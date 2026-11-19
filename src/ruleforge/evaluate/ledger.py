from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from ruleforge.evaluate.identity import (
    RETRY_SCOPE_COMPONENT,
    RETRY_SCOPE_SCENARIO_BUNDLE,
    RETRY_SCOPES,
)
from ruleforge.scenarios.manifest import SAMPLING_SCHEMA_VERSION
from ruleforge.core.constants import FULL_RULESET_KEY, PUBLISHED_COPY_FNAME, PUBLISHED_LEDGER_FNAME
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.artifacts import write_jsonl_records
from ruleforge.core.resume import count_complete_jsonl_records

logger = logging.getLogger(__name__)

LEDGER_FNAME = "attempts.jsonl"
RAW_OUTPUTS_FNAME = "raw_outputs.jsonl"
# The schema of a published copy's form of a draw's ledger (:data:`PUBLISHED_LEDGER_FNAME`, :func:`published_form`).
PUBLISHED_LEDGER_SCHEMA = 1
JUDGED_ON_FIELD = "judged_on"

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_TRANSPORT_FAILED = "transport_failed"
ATTEMPT_STATUSES = (STATUS_ACCEPTED, STATUS_REJECTED, STATUS_TRANSPORT_FAILED)
# The statuses that consume a semantic attempt: every component was generated and judged.
SEMANTIC_STATUSES = (STATUS_ACCEPTED, STATUS_REJECTED)


class LedgerError(SamplingDesignError):
    """Raised when a ledger violates an invariant it is meant to guarantee."""


def raw_output_sha256(text: Optional[str]) -> Optional[str]:
    """Digest of a component's raw text, or None when nothing came back."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


@dataclass(frozen=True)
class ComponentRecord:
    """One component of one attempted bundle, as the ledger records it."""

    component_id: str
    valid: bool
    failure_reason: Optional[str]
    judgment: Optional[str]
    finish_reason: Optional[str]
    generation_seed: Optional[int]
    seed_sent: bool
    raw_output_sha256: Optional[str]
    provider_request_ref: Optional[str] = None
    request_block_ref: Optional[str] = None
    rule_judgments: Optional[dict[str, str]] = None
    preempted_rules: tuple[str, ...] = ()
    # Lets a rebuilt trace prove it shows the prompt the judge saw.
    prompt_sha256: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["preempted_rules"] = list(self.preempted_rules)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComponentRecord":
        data = dict(payload)
        data["preempted_rules"] = tuple(data.get("preempted_rules") or ())
        return cls(**data)


@dataclass(frozen=True)
class AttemptRecord:
    """One attempted bundle: the immutable scientific key plus its outcome. A ``component``-scope row carries only the
    components its attempt generated. ``judged_on`` is the stack digest of a local judge's run."""

    campaign_id: str
    condition_id: str
    scenario_id: str
    decoder_draw_id: int
    semantic_attempt_id: int
    request_block_id: str
    status: str
    seed_capability: str
    components: tuple[ComponentRecord, ...]
    started_at: str
    completed_at: str
    failure_reason: Optional[str] = None
    schema_version: int = SAMPLING_SCHEMA_VERSION
    retry_scope: str = RETRY_SCOPE_SCENARIO_BUNDLE
    judged_on: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in ATTEMPT_STATUSES:
            raise LedgerError(f"unknown attempt status {self.status!r}")
        if self.retry_scope not in RETRY_SCOPES:
            raise LedgerError(f"unknown retry scope {self.retry_scope!r}")
        if self.schema_version != SAMPLING_SCHEMA_VERSION:
            raise LedgerError(
                f"attempt record schema_version {self.schema_version!r} is not "
                f"{SAMPLING_SCHEMA_VERSION}"
            )
        if self.status == STATUS_ACCEPTED and not all(c.valid for c in self.components):
            raise LedgerError("an accepted attempt cannot carry an invalid component")

    @property
    def key(self) -> tuple[str, int]:
        return (self.scenario_id, self.decoder_draw_id)

    @property
    def consumes_attempt(self) -> bool:
        return self.status in SEMANTIC_STATUSES

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = [component.to_dict() for component in self.components]
        if self.retry_scope == RETRY_SCOPE_SCENARIO_BUNDLE:
            del payload["retry_scope"]
        if self.judged_on is None:
            del payload["judged_on"]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttemptRecord":
        data = dict(payload)
        try:
            data["components"] = tuple(
                ComponentRecord.from_dict(item) for item in data.get("components", [])
            )
            return cls(**data)
        except TypeError as error:
            raise LedgerError(f"malformed attempt record: {error}") from error


def raw_row_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    """Return the ledger row a raw row belongs to: (scenario, attempt, block)."""
    return (
        str(row["scenario_id"]),
        int(row["semantic_attempt_id"]),
        str(row["request_block_id"]),
    )


@dataclass
class LedgerState:
    """What a draw's ledger says, indexed for the runner and the gate."""

    records: list[AttemptRecord] = field(default_factory=list)
    accepted: dict[str, AttemptRecord] = field(default_factory=dict)
    attempts_consumed: dict[str, int] = field(default_factory=dict)
    block_ids: set[str] = field(default_factory=set)
    vouched: set[tuple[str, str]] = field(default_factory=set)
    retry_scope: Optional[str] = None
    # Component scope: each scenario's bundle, and the components settled so far with their rows.
    bundles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    settled: dict[str, dict[str, tuple[AttemptRecord, ComponentRecord]]] = field(default_factory=dict)

    def next_attempt(self, scenario_id: str) -> int:
        return self.attempts_consumed.get(scenario_id, 0)

    def is_accepted(self, scenario_id: str) -> bool:
        return scenario_id in self.accepted

    def components_at(self, scenario_id: str, attempt: int, bundle: Sequence[str]) -> tuple[str, ...]:
        """Return the components of ``bundle`` a component-scope ``attempt`` generates: those no earlier attempt
        settled."""
        settled = self.settled.get(scenario_id, {})
        return tuple(
            cid for cid in bundle if cid not in settled or settled[cid][0].semantic_attempt_id >= attempt
        )

    def accepted_components(self, scenario_id: str) -> dict[str, tuple[AttemptRecord, ComponentRecord]]:
        """An accepted bundle's components, in bundle order, each with the row that generated it."""
        record = self.accepted[scenario_id]
        if record.retry_scope == RETRY_SCOPE_SCENARIO_BUNDLE:
            return {component.component_id: (record, component) for component in record.components}
        settled = self.settled[scenario_id]
        return {cid: settled[cid] for cid in self.bundles[scenario_id]}

    def accepted_bundle(self, scenario_id: str) -> AttemptRecord:
        """Return the accepted row with every component of the accepted bundle, under either scope."""
        record = self.accepted[scenario_id]
        if record.retry_scope == RETRY_SCOPE_SCENARIO_BUNDLE:
            return record
        components = tuple(component for _, component in self.accepted_components(scenario_id).values())
        return replace(record, components=components)

    def add(self, record: AttemptRecord, *, where: str = "") -> None:
        """Index one more row, enforcing the ledger's invariants."""
        sid = record.scenario_id
        if self.retry_scope not in (None, record.retry_scope):
            raise LedgerError(
                f"{where} was recorded under retry scope {record.retry_scope!r}; the ledger's rows are "
                f"{self.retry_scope!r}, and one draw is retried one way"
            )
        if sid in self.accepted:
            raise LedgerError(
                f"{where} attempts scenario {sid[:12]} after it was already "
                "accepted; a key is immutable once accepted"
            )
        if (sid, record.request_block_id) in self.vouched:
            raise LedgerError(
                f"{where} records block {record.request_block_id} twice for "
                f"scenario {sid[:12]}"
            )
        consumed = self.attempts_consumed.get(sid, 0)
        if record.semantic_attempt_id != consumed:
            raise LedgerError(
                f"{where} has semantic_attempt_id {record.semantic_attempt_id} for "
                f"scenario {sid[:12]}; the ledger holds {consumed} consumed "
                f"attempt(s), so {consumed} was expected"
            )
        if record.retry_scope == RETRY_SCOPE_COMPONENT:
            self._settle(record, where)
        self.retry_scope = record.retry_scope
        self.records.append(record)
        self.block_ids.add(record.request_block_id)
        self.vouched.add((sid, record.request_block_id))
        if record.consumes_attempt:
            self.attempts_consumed[sid] = consumed + 1
            if record.status == STATUS_ACCEPTED:
                self.accepted[sid] = record

    def _settle(self, record: AttemptRecord, where: str) -> None:
        sid = record.scenario_id
        carried = tuple(component.component_id for component in record.components)
        bundle = self.bundles.get(sid, carried)
        settled = self.settled.get(sid, {})
        expected = tuple(cid for cid in bundle if cid not in settled)
        if not carried or carried != expected:
            raise LedgerError(
                f"{where} carries components {list(carried)} for scenario {sid[:12]}; its unsettled "
                f"components are {list(expected)}"
            )
        if record.consumes_attempt:
            newly = {c.component_id: (record, c) for c in record.components if c.valid}
            if (len(newly) == len(carried)) != (record.status == STATUS_ACCEPTED):
                raise LedgerError(
                    f"{where} is {record.status!r} for scenario {sid[:12]} with "
                    f"{len(carried) - len(newly)} component(s) left unsettled; a component-scope row is "
                    "accepted exactly when it settles the last"
                )
            settled = {**settled, **newly}
        self.bundles[sid] = bundle
        self.settled[sid] = settled


class DrawLedger:
    """Reader and writer for one draw directory's ledger and raw outputs."""

    def __init__(
        self,
        directory: Path | str,
        *,
        campaign_id: str,
        condition_id: str,
        decoder_draw_id: int,
    ) -> None:
        self.directory = Path(directory)
        self.campaign_id = campaign_id
        self.condition_id = condition_id
        self.decoder_draw_id = int(decoder_draw_id)

    @property
    def ledger_path(self) -> Path:
        return self.directory / LEDGER_FNAME

    @property
    def raw_path(self) -> Path:
        return self.directory / RAW_OUTPUTS_FNAME


    def accepted(self) -> "LedgerState | PublishedLedger":
        """The draw's accepted attempts: the strict replay of its ledger (:meth:`load`), or on a published copy the
        replay's outcome as the copy holds it (:data:`PUBLISHED_LEDGER_FNAME`)."""
        mode = draw_mode(self.directory)
        if not mode.published:
            return self.load()
        if mode.form is None:
            raise LedgerError(f"{self.directory} is in a published copy of a campaign root that leaves this "
                              "condition's attempt ledgers out: read its accepted attempts from a root you ran")
        return PublishedLedger(self.directory / PUBLISHED_LEDGER_FNAME, mode.form,
                               identity=(self.campaign_id, self.condition_id, self.decoder_draw_id))

    def _check_identity(self, record: AttemptRecord, line: int) -> None:
        expected = (self.campaign_id, self.condition_id, self.decoder_draw_id)
        actual = (record.campaign_id, record.condition_id, record.decoder_draw_id)
        if actual != expected:
            raise LedgerError(
                f"{self.ledger_path}:{line} belongs to {actual}, not {expected}"
            )

    def load(self, *, repair: bool = False) -> LedgerState:
        """Read the ledger and index it, checking every invariant. ``repair`` truncates a torn trailing record and drops
        unvouched raw rows; without it a torn tail is refused. A draw of a published copy is refused: the replay, the
        history, cost and repair read the ledger itself, which a published copy holds only as its accepted attempts."""
        if draw_mode(self.directory).published:
            raise LedgerError(f"{self.directory} is in a published copy of a campaign root, which holds a draw's ledger "
                              f"as its accepted attempts alone ({PUBLISHED_LEDGER_FNAME}), not the ledger a replay, the "
                              "attempt history, cost and repair read; run on a root you ran")
        state = LedgerState()
        n_rows = count_complete_jsonl_records(self.ledger_path, repair=repair)
        if self.ledger_path.exists():
            with self.ledger_path.open("rb") as handle:
                for line, raw in enumerate(handle, start=1):
                    if line > n_rows:
                        # Nothing after a torn record can be trusted.
                        raise LedgerError(
                            f"{self.ledger_path} has a torn trailing record; "
                            "load with repair=True from the runner, or fix it by hand"
                        )
                    record = AttemptRecord.from_dict(json.loads(raw))
                    self._check_identity(record, line)
                    state.add(record, where=f"{self.ledger_path}:{line}")
        if repair:
            self._drop_orphan_raw_rows(state)
        return state

    def _drop_orphan_raw_rows(self, state: LedgerState) -> None:
        count_complete_jsonl_records(self.raw_path, repair=True)
        if not self.raw_path.exists():
            return
        vouched = {
            (r.scenario_id, r.semantic_attempt_id, r.request_block_id)
            for r in state.records
        }
        kept: list[dict[str, Any]] = []
        dropped = 0
        for row in self.iter_raw():
            if raw_row_key(row) in vouched:
                kept.append(row)
            else:
                dropped += 1
        if dropped:
            logger.warning(
                "%s: dropping %d raw output row(s) no ledger row vouches for (an "
                "interrupted block; the bundle will be attempted again)",
                self.raw_path,
                dropped,
            )
            write_jsonl_records(self.raw_path, kept)

    def iter_raw(self) -> Iterator[dict[str, Any]]:
        """Stream raw-output rows, in the order they were written."""
        if not self.raw_path.exists():
            return
        with self.raw_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    yield json.loads(raw)


    def append(
        self,
        records: Sequence[AttemptRecord],
        raw_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Append one block's raw rows, then its ledger rows, durably."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for record in records:
            self._check_identity(record, 0)
        if raw_rows:
            self._append_lines(self.raw_path, (dict(row) for row in raw_rows))
        if records:
            self._append_lines(self.ledger_path, (record.to_dict() for record in records))

    @staticmethod
    def _append_lines(path: Path, payloads) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads
            )
            handle.flush()
            os.fsync(handle.fileno())


# The published-copy record's key listing the conditions whose draws hold their ledger's published form.
LEDGER_CONDITIONS_KEY = "ledger_conditions"


@dataclass(frozen=True)
class DrawMode:
    """How a draw's ledger is read: as recorded (the ledger itself, replayed), or as a published copy holds it (its
    published form, or none where the copy leaves the condition's ledgers out)."""

    published: bool
    form: Optional[dict[str, Any]] = None


def _published_root(directory: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    for parent in directory.parents:
        marker = parent / PUBLISHED_COPY_FNAME
        if marker.is_file():
            try:
                record = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise LedgerError(f"{marker}: not a readable published-copy record ({type(error).__name__})") from None
            if not isinstance(record, dict):
                raise LedgerError(f"{marker}: not a published-copy record")
            return parent, record
    return None


def _read_form(path: Path) -> dict[str, Any]:
    try:
        form = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LedgerError(f"{path}: not a readable published form ({type(error).__name__})") from None
    if not isinstance(form, dict) or form.get("schema_version") != PUBLISHED_LEDGER_SCHEMA:
        raise LedgerError(f"{path}: not a published form of schema {PUBLISHED_LEDGER_SCHEMA}")
    return form


def draw_mode(draw_dir: Path | str) -> DrawMode:
    """How a draw's ledger is read, every reader's one decision: recorded (no published-copy record above it, no
    published form), or published (a published-copy record above it, no recorded ledger, the published form where the
    record says the copy holds the condition's ledgers). Every other combination is refused by name."""
    directory = Path(draw_dir)
    form_path = directory / PUBLISHED_LEDGER_FNAME
    marked = _published_root(directory)
    if marked is None:
        if form_path.exists():
            raise LedgerError(f"{directory} holds a published ledger form ({PUBLISHED_LEDGER_FNAME}) but lies in no "
                              f"published copy (no {PUBLISHED_COPY_FNAME} above it): a recorded root holds ledgers")
        return DrawMode(False)
    root, record = marked
    if (directory / LEDGER_FNAME).exists():
        raise LedgerError(f"{directory} holds a recorded ledger ({LEDGER_FNAME}) in a published copy "
                          f"({root / PUBLISHED_COPY_FNAME}), which holds ledgers in their published form alone")
    if form_path.exists():
        return DrawMode(True, _read_form(form_path))
    condition = directory.parent.relative_to(root).as_posix()
    if condition in (record.get(LEDGER_CONDITIONS_KEY) or ()):
        raise LedgerError(f"{directory} lacks its ledger's published form ({PUBLISHED_LEDGER_FNAME}), which its "
                          f"published copy holds for every draw of {condition}")
    return DrawMode(True)


# The parts of a draw's replayed ledger its published form can hold: of each accepted scenario, its accepted attempt
# and of the full-ruleset call its bundle holds the attempt and block that generated it, its judgment, rule judgments
# and raw output's digest; of the draw, the first and last start time of its attempts, the judging stacks its rows
# name and whether it holds an attempt. A root's readers choose which (ruleforge.hub.PROFILES).
CALL_PARTS = {"call": ("semantic_attempt_id", "request_block_id"), "judgment": ("judgment",),
              "rule_judgments": ("rule_judgments",), "raw_output_sha256": ("raw_output_sha256",)}
FORM_PARTS = ("accepted_attempt", *CALL_PARTS, "started_at", JUDGED_ON_FIELD, "has_attempts")


def _call(record: AttemptRecord, component: ComponentRecord, parts: Sequence[str]) -> dict[str, Any]:
    values = {"semantic_attempt_id": record.semantic_attempt_id, "request_block_id": record.request_block_id,
              "judgment": component.judgment, "rule_judgments": component.rule_judgments,
              "raw_output_sha256": component.raw_output_sha256}
    return {name: values[name] for part in parts if part in CALL_PARTS for name in CALL_PARTS[part]}


def published_form(draw_dir: Path | str, *, campaign_id: str, condition_id: str, decoder_draw_id: int,
                   parts: Sequence[str] = FORM_PARTS) -> dict:
    """The published form of a draw's ledger, from its strict replay: of :data:`FORM_PARTS`, the ``parts`` its root's
    rebuilds read, and beside its accepted scenarios the draw's identity, which a reader of them checks."""
    unknown = sorted(set(parts) - set(FORM_PARTS))
    if unknown:
        raise LedgerError(f"{unknown} are not parts of a published ledger form ({list(FORM_PARTS)})")
    state = DrawLedger(draw_dir, campaign_id=campaign_id, condition_id=condition_id,
                       decoder_draw_id=decoder_draw_id).load()
    form: dict[str, Any] = {"schema_version": PUBLISHED_LEDGER_SCHEMA}
    if "has_attempts" in parts:
        form["has_attempts"] = bool(state.records)
    if "started_at" in parts:
        stamps = sorted(record.started_at for record in state.records if record.started_at)
        form["started_at"] = {"first": stamps[0], "last": stamps[-1]} if stamps else None
    if JUDGED_ON_FIELD in parts:
        stacks = Counter(record.judged_on for record in state.records if record.judged_on is not None)
        form[JUDGED_ON_FIELD] = dict(sorted(stacks.items()))
    if any(part in parts for part in ("accepted_attempt", *CALL_PARTS)):
        form.update(campaign_id=campaign_id, condition_id=condition_id, decoder_draw_id=int(decoder_draw_id))
        accepted = []
        for scenario_id, record in state.accepted.items():
            full = state.accepted_components(scenario_id).get(FULL_RULESET_KEY)
            row: dict[str, Any] = {"scenario_id": scenario_id}
            if "accepted_attempt" in parts:
                row["semantic_attempt_id"] = record.semantic_attempt_id
            if any(part in parts for part in CALL_PARTS):
                row[FULL_RULESET_KEY] = None if full is None else _call(*full, parts)
            accepted.append(row)
        form["accepted"] = accepted
    return form


def accepted_calls(draw_dir: Path | str, *, campaign_id: str, condition_id: str,
                   decoder_draw_id: int) -> frozenset[tuple[str, int, str]]:
    """The (scenario, attempt, block) of each accepted full-ruleset call of a draw, from its ledger's strict replay: the
    raw outputs a rebuild reads of the draw."""
    state = DrawLedger(draw_dir, campaign_id=campaign_id, condition_id=condition_id,
                       decoder_draw_id=decoder_draw_id).load()
    calls = (state.accepted_components(sid).get(FULL_RULESET_KEY) for sid in state.accepted)
    return frozenset((record.scenario_id, record.semantic_attempt_id, record.request_block_id)
                     for record, _ in filter(None, calls))


def published_form_text(draw_dir: Path | str, **identity: Any) -> bytes:
    """:func:`published_form` as a published copy holds it: one JSON document."""
    return (json.dumps(published_form(draw_dir, **identity)) + "\n").encode("utf-8")


def _not_held(path: Path, what: str) -> LedgerError:
    return LedgerError(f"{path} is the published form of a draw's attempt ledger, which holds what its root's rebuilds "
                       f"read: not {what}; read it from a root you ran")


class _Held:
    """Values a published form holds of one record: reading one it does not hold is refused by name."""

    def __init__(self, path: Path, what: str, values: Mapping[str, Any]) -> None:
        self._path, self._what, self._values = path, what, dict(values)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._values:
            raise _not_held(self._path, f"{self._what}'s {name}")
        return self._values[name]


class PublishedComponent(_Held):
    """An accepted bundle's full-ruleset call, as a published form holds it (``component_id`` and the parts held)."""

    def __init__(self, path: Path, values: Mapping[str, Any]) -> None:
        super().__init__(path, "a full-ruleset call", {"component_id": FULL_RULESET_KEY, **{
            name: values[name] for name in ("judgment", "rule_judgments", "raw_output_sha256") if name in values}})


class PublishedLedger:
    """A draw's accepted attempts as its published form holds them, read as :class:`LedgerState` gives them; the
    ledger's history (its records, consumed attempts, blocks, settled components) is not there and is refused, and so
    is any part of the form its root's rebuilds do not read."""

    def __init__(self, path: Path, form: Mapping[str, Any], *, identity: tuple[str, str, int]) -> None:
        self.path = path
        self._rows = form.get("accepted")
        actual = (form.get("campaign_id"), form.get("condition_id"), form.get("decoder_draw_id"))
        if self._rows is not None and actual != identity:
            raise LedgerError(f"{path} belongs to {actual}, not {identity}")
        self._bundles: dict[str, _Held] = {}
        self._calls: dict[str, _Held] = {}
        for row in self._rows or ():
            sid, full = row["scenario_id"], row.get(FULL_RULESET_KEY)
            components = () if full is None else (PublishedComponent(path, full),)
            self._bundles[sid] = _Held(path, "an accepted bundle", {
                "scenario_id": sid, "components": components,
                **({"semantic_attempt_id": int(row["semantic_attempt_id"])} if "semantic_attempt_id" in row else {})})
            if full is not None:
                self._calls[sid] = _Held(path, "a full-ruleset call's attempt", {
                    "scenario_id": sid, **{name: full[name] for name in ("semantic_attempt_id", "request_block_id")
                                           if name in full}})

    @property
    def accepted(self) -> dict[str, _Held]:
        if self._rows is None:
            raise _not_held(self.path, "its accepted scenarios")
        return self._bundles

    def is_accepted(self, scenario_id: str) -> bool:
        return scenario_id in self.accepted

    def accepted_bundle(self, scenario_id: str) -> _Held:
        return self.accepted[scenario_id]

    def accepted_components(self, scenario_id: str) -> dict[str, tuple[_Held, PublishedComponent]]:
        return {component.component_id: (self._calls[scenario_id], component)
                for component in self.accepted[scenario_id].components}

    def __getattr__(self, name: str) -> Any:
        raise _not_held(self.path, f"the ledger's {name}")


def _form_part(mode: DrawMode, path: Path, part: str) -> Any:
    if part not in mode.form:
        raise _not_held(path, f"the draw's {part}")
    return mode.form[part]


def has_attempts(draw_dir: Path | str) -> bool:
    """Whether a draw's ledger holds at least one complete attempt, reading its first line only (or, on a published
    copy, as its published form records)."""
    mode = draw_mode(draw_dir)
    if mode.published:
        return mode.form is not None and bool(
            _form_part(mode, Path(draw_dir) / PUBLISHED_LEDGER_FNAME, "has_attempts"))
    path = Path(draw_dir) / LEDGER_FNAME
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
    if not first.endswith("\n"):
        return False
    try:
        return isinstance(json.loads(first), dict)
    except json.JSONDecodeError:
        return False


def named_stacks(draw_dir: Path | str) -> Counter:
    """How many of a draw's complete ledger rows each judging stack names (:data:`JUDGED_ON_FIELD`), or on a published
    copy as its published form records."""
    mode = draw_mode(draw_dir)
    if mode.published:
        return Counter({} if mode.form is None else _form_part(mode, Path(draw_dir) / PUBLISHED_LEDGER_FNAME,
                                                                JUDGED_ON_FIELD))
    path = Path(draw_dir) / LEDGER_FNAME
    named: Counter = Counter()
    if not path.is_file():
        return named
    token = json.dumps(JUDGED_ON_FIELD).encode("utf-8")
    with path.open("rb") as handle:
        for _, line in zip(range(count_complete_jsonl_records(path)), handle):
            if token in line:
                named[json.loads(line)[JUDGED_ON_FIELD]] += 1
    return named


def attempt_stamps(draw_dir: Path | str) -> list[str]:
    """The start times a draw's ledger rows record, or on a published copy the first and last of them."""
    mode = draw_mode(draw_dir)
    if mode.published:
        stamps = None if mode.form is None else _form_part(mode, Path(draw_dir) / PUBLISHED_LEDGER_FNAME, "started_at")
        return [] if stamps is None else [stamps["first"], stamps["last"]]
    path = Path(draw_dir) / LEDGER_FNAME
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [str(row["started_at"]) for row in rows if row.get("started_at")]
