from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Optional

from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.paths import resolve
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    REPLAY_RECORD_FNAME,
    RUNS_FNAME,
    SAMPLING_MANIFEST_FNAME,
    derive_generation_seed,
    draw_dirname,
    load_sampling_manifest,
    refuse_published,
    run_log_problem,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import (
    canonical_digest,
    sha256_file,
)
from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME, AttemptRecord, DrawLedger, LedgerState
from ruleforge.evaluate.runner import REQUEST_BLOCKS_FNAME
from ruleforge.evaluate.sharding import (
    CodeState,
    code_state_key,
    describe_code_state,
    parse_shard_dirname,
    recorded_code_states,
)
from ruleforge.evaluate.stack import (
    LEDGER_ROWS_KEY,
    LOCAL_MODEL_TYPE,
    describe_changes,
    run_spans,
    runtime_changes,
    unlogged_stacks,
)

# The record's kind, and the manifest field it amends.
CAP_AMENDMENT = "semantic_attempt_cap"
AMENDED_FIELD = "max_semantic_attempts"
AMENDMENT_SCHEMA_VERSION = 1
# Files of an amended draw: those a resume appends to, and the provenance it never rewrites.
APPENDED_FILES = (LEDGER_FNAME, RAW_OUTPUTS_FNAME, REQUEST_BLOCKS_FNAME, RUNS_FNAME)
FIXED_FILES = (DRAW_PROVENANCE_FNAME,)
# A draw's files by digest, keyed as a label replay's record keys them.
LEDGER_DIGESTS = {"attempts_sha256": LEDGER_FNAME, "raw_outputs_sha256": RAW_OUTPUTS_FNAME}
REPLAYED_DIGESTS = {**LEDGER_DIGESTS, "runs_sha256": RUNS_FNAME, "draw_provenance_sha256": DRAW_PROVENANCE_FNAME}
OUTCOME_ACCEPTED, OUTCOME_EXHAUSTED, OUTCOME_PENDING, OUTCOME_ABSENT = (
    "accepted", "exhausted", "pending", "not in this root")


class AmendmentError(SamplingDesignError):
    """Raised when an amendment cannot be recorded, read, or held to what it allows."""


def file_snapshot(path: Path) -> dict[str, Any]:
    """A file's size and SHA-256."""
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def draw_digests(draw_dir: Path | str, files: Mapping[str, str] = LEDGER_DIGESTS) -> dict[str, str]:
    """A draw's ``files`` by SHA-256, keyed as they are (:data:`LEDGER_DIGESTS`, :data:`REPLAYED_DIGESTS`)."""
    return {key: sha256_file(Path(draw_dir) / name) for key, name in files.items()}


def component_prompts(records: Iterable[AttemptRecord], scenario_id: str, where: str) -> dict[str, str]:
    """Return ``{component id: prompt digest}`` over a scenario's ledger rows; a prompt that changes between attempts is
    refused."""
    found: dict[str, set[Optional[str]]] = {}
    for record in records:
        if record.scenario_id == scenario_id:
            for component in record.components:
                found.setdefault(component.component_id, set()).add(component.prompt_sha256)
    mixed = sorted(cid for cid, digests in found.items() if len(digests) != 1 or None in digests)
    if not found or mixed:
        raise AmendmentError(
            f"{where}: scenario {scenario_id[:12]} records no prompt digest, or more than one, for component(s) "
            f"{mixed or 'all'}: its prompts cannot be held to the ones its attempts were sent"
        )
    return {cid: str(next(iter(digests))) for cid, digests in found.items()}


def bundle_prompt_digest(prompts: Mapping[str, str]) -> str:
    """One digest of a bundle's prompts: its ``[component id, prompt digest]`` pairs in bundle order."""
    return canonical_digest([[component_id, digest] for component_id, digest in prompts.items()])


def check_seeds(records: Iterable[AttemptRecord], *, generation_seed_base: int, where: str,
                identity_id: Optional[str] = None) -> None:
    """Refuse a component whose recorded seed is not what ``derive_generation_seed`` gives it (none where none was
    sent)."""
    for record in records:
        for component in record.components:
            expected = derive_generation_seed(
                generation_seed_base=generation_seed_base, campaign_id=record.campaign_id,
                condition_id=record.condition_id if identity_id is None else identity_id,
                scenario_id=record.scenario_id,
                decoder_draw_id=record.decoder_draw_id, semantic_attempt_id=record.semantic_attempt_id,
                component_id=component.component_id,
            ) if component.seed_sent else None
            if component.generation_seed != expected:
                raise AmendmentError(
                    f"{where}: {record.scenario_id[:12]}/{component.component_id} at attempt "
                    f"{record.semantic_attempt_id} records seed {component.generation_seed}, not the {expected} the "
                    "runner derives for it"
                )


def _code_state_dict(state: CodeState) -> dict[str, Any]:
    return {"code_revision": state[0], "code_dirty": state[1], "code_state_sha256": state[2]}


@dataclass(frozen=True)
class AmendedBundle:
    """One bundle the amendment gives more attempts: where its draw is, what it used, what it was sent."""

    condition_id: str
    decoder_draw_id: int
    scenario_id: str
    # ``shard-2-of-8``, or None for the campaign root.
    shard: Optional[str]
    # The old cap.
    attempts: int
    prompts_sha256: str
    files: Mapping[str, Mapping[str, Any]]

    def describe(self) -> str:
        return (f"{self.condition_id} {draw_dirname(self.decoder_draw_id)} scenario {self.scenario_id[:12]} "
                f"({self.shard or 'the campaign root'})")

    def to_dict(self) -> dict[str, Any]:
        return {"condition_id": self.condition_id, "decoder_draw_id": self.decoder_draw_id,
                "scenario_id": self.scenario_id, "shard": self.shard, "attempts": self.attempts,
                "prompts_sha256": self.prompts_sha256, "files": {name: dict(entry) for name, entry in self.files.items()}}


@dataclass(frozen=True)
class CapAmendment:
    """A raised ``max_semantic_attempts``, as the manifests of a campaign's roots record it."""

    old_cap: int
    new_cap: int
    recorded_at: str
    decided_by: str
    decided_on: str
    reason: str
    ledger: str
    # The code that recorded the amendment, which the resume runs from.
    code_state: CodeState
    prior_code_states: tuple[CodeState, ...]
    bundles: tuple[AmendedBundle, ...]
    conditions: tuple[str, ...]
    checks: Mapping[str, Any]
    # ``(condition id, decoder draw, scenario id)`` of each bundle carried over exhausted, which keeps the old cap.
    held: tuple[tuple[str, int, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": CAP_AMENDMENT, "field": AMENDED_FIELD, "schema_version": AMENDMENT_SCHEMA_VERSION,
            "old_cap": self.old_cap, "new_cap": self.new_cap, "recorded_at": self.recorded_at,
            "decided_by": self.decided_by, "decided_on": self.decided_on, "reason": self.reason,
            "ledger": self.ledger, "code_state": _code_state_dict(self.code_state),
            "prior_code_states": [_code_state_dict(state) for state in self.prior_code_states],
            "bundles": [bundle.to_dict() for bundle in self.bundles], "conditions": list(self.conditions),
            "checks": dict(self.checks),
            **({"held": [{"condition_id": condition_id, "decoder_draw_id": draw, "scenario_id": scenario_id}
                         for condition_id, draw, scenario_id in self.held]} if self.held else {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapAmendment":
        if not isinstance(payload, Mapping) or (payload.get("kind"), payload.get("field"), payload.get(
                "schema_version")) != (CAP_AMENDMENT, AMENDED_FIELD, AMENDMENT_SCHEMA_VERSION):
            raise AmendmentError(
                f"an amendment of another kind or version than this build reads ({CAP_AMENDMENT} of {AMENDED_FIELD}, "
                f"schema {AMENDMENT_SCHEMA_VERSION})"
            )
        try:
            return cls(
                old_cap=int(payload["old_cap"]), new_cap=int(payload["new_cap"]),
                recorded_at=str(payload["recorded_at"]), decided_by=str(payload["decided_by"]),
                decided_on=str(payload["decided_on"]), reason=str(payload["reason"]), ledger=str(payload["ledger"]),
                code_state=code_state_key(payload["code_state"]),
                prior_code_states=tuple(code_state_key(state) for state in payload["prior_code_states"]),
                bundles=tuple(AmendedBundle(**{**bundle, "decoder_draw_id": int(bundle["decoder_draw_id"])})
                              for bundle in payload["bundles"]),
                conditions=tuple(str(condition_id) for condition_id in payload["conditions"]),
                checks=dict(payload["checks"]),
                held=tuple((str(item["condition_id"]), int(item["decoder_draw_id"]), str(item["scenario_id"]))
                           for item in payload.get("held", ())),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AmendmentError(f"malformed {CAP_AMENDMENT} amendment: {error!r}") from error

    def bundles_of(self, condition_id: str, decoder_draw_id: int, shard: Optional[str]) -> list[AmendedBundle]:
        """The amended bundles of one draw where it is held (a shard root's label, or None for the campaign root)."""
        return [bundle for bundle in self.bundles
                if (bundle.condition_id, bundle.decoder_draw_id, bundle.shard) == (condition_id, decoder_draw_id, shard)]

    def held_caps(self, condition_id: str, decoder_draw_id: int) -> dict[str, int]:
        """Return ``{scenario id: old cap}`` of the bundles one draw holds at the old cap."""
        return {scenario_id: self.old_cap for held_condition, draw, scenario_id in self.held
                if (held_condition, draw) == (condition_id, decoder_draw_id)}

    def code_state_refusal(
        self, found: Mapping[CodeState, Collection[tuple[Optional[str], str]]], condition_id: str
    ) -> Optional[str]:
        """Return why a condition's recorded code states are not what this amendment allows, or None when they are."""
        named = {(bundle.shard, draw_dirname(bundle.decoder_draw_id))
                 for bundle in self.bundles if bundle.condition_id == condition_id}
        others = sorted((state for state in found if state != self.code_state), key=str)
        outside = sorted(set(found.get(self.code_state, ())) - named, key=str)
        problems = [f"{describe_code_state(state)} is no code state the campaign ran before it"
                    for state in others if state not in self.prior_code_states]
        if len(others) > 1:
            problems.append(f"{len(others)} code states besides the amendment's ({'; '.join(map(describe_code_state, others))})")
        if outside:
            where = ", ".join(f"{shard or 'the campaign root'} {draw}" for shard, draw in outside)
            problems.append(f"its own, {describe_code_state(self.code_state)}, ran {where}, which it does not name")
        return "; ".join(problems) or None

    def summary(self) -> dict[str, Any]:
        """The record without its bundles' file snapshots, the code states described: what a disclosure shows."""
        record = self.to_dict()
        record.pop("bundles")
        record["code_state_described"] = describe_code_state(self.code_state)
        return record


def cap_amendment(manifest: Mapping[str, Any]) -> Optional[CapAmendment]:
    """The campaign's cap amendment, None when its manifest records none; refused unless it records exactly one."""
    records = manifest.get(MANIFEST_AMENDMENTS_KEY)
    if records is None:
        return None
    if not isinstance(records, list) or len(records) != 1:
        raise AmendmentError(
            f"campaign {manifest.get('campaign_id')!r} records {MANIFEST_AMENDMENTS_KEY} {records!r}: this build reads "
            "one cap amendment, no more"
        )
    return CapAmendment.from_dict(records[0])


def _appended_rows(draw_dir: Path, name: str, snapshot: Mapping[str, Any], where: str) -> list[dict[str, Any]]:
    path = draw_dir / name
    data = path.read_bytes() if path.is_file() else b""
    head, tail = data[: snapshot["bytes"]], data[snapshot["bytes"]:]
    if len(head) != snapshot["bytes"] or hashlib.sha256(head).hexdigest() != snapshot["sha256"]:
        raise AmendmentError(
            f"{where}: {name} no longer begins with the {snapshot['bytes']} bytes it held when the amendment was "
            "recorded; a resume only appends"
        )
    if tail and not tail.endswith(b"\n"):
        raise AmendmentError(f"{where}: {name} ends in a torn row: resume the draw to its end before reading it")
    # Split on newline bytes: a raw U+2028 or U+0085 in a model's text would cut a row under str.splitlines().
    try:
        return [json.loads(line) for line in tail.split(b"\n")[:-1]]
    except ValueError as error:
        raise AmendmentError(f"{where}: {name} gained a row that is not JSON ({error})") from error


def _identity_id(draw_dir: Path, condition_id: str) -> str:
    from ruleforge.evaluate.campaign import CONDITION_FNAME, read_condition  # campaign imports this module

    path = draw_dir.parent / CONDITION_FNAME
    if not path.is_file():
        return condition_id
    condition = read_condition(path)
    if condition.condition_id != condition_id:
        raise AmendmentError(f"{path} registers {condition.condition_id!r}, not the draw's {condition_id!r}")
    return condition.identity_id


def check_resumed_draw(
    draw_dir: Path | str,
    amendment: CapAmendment,
    *,
    campaign_id: str,
    condition_id: str,
    decoder_draw_id: int,
    shard: Optional[str],
    generation_seed_base: int,
    finished: bool = True,
) -> Optional[dict[str, Any]]:
    """Hold an amended draw to the amendment (its recorded bytes, then only the amended bundles' rows; None if not
    named) and return each amended bundle's outcome, appended row count and code state."""
    draw_dir = Path(draw_dir)
    bundles = amendment.bundles_of(condition_id, decoder_draw_id, shard)
    if not bundles:
        return None
    where = f"{draw_dir} (amended: {', '.join(bundle.describe() for bundle in bundles)})"
    files = bundles[0].files
    changed = [name for name in FIXED_FILES if file_snapshot(draw_dir / name) != dict(files[name])]
    if changed:
        raise AmendmentError(f"{where}: {changed} changed since the amendment was recorded; a resume never rewrites them")
    appended = {name: _appended_rows(draw_dir, name, files[name], where) for name in APPENDED_FILES}
    state = DrawLedger(draw_dir, campaign_id=campaign_id, condition_id=condition_id, decoder_draw_id=decoder_draw_id).load()
    split = len(state.records) - len(appended[LEDGER_FNAME])
    before, after = state.records[:split], state.records[split:]
    names = {bundle.scenario_id for bundle in bundles}
    prompts = {sid: component_prompts(before, sid, where) for sid in names}
    window = range(amendment.old_cap, amendment.new_cap)
    problems = [f"{bundle.scenario_id[:12]}'s earlier attempts were sent other prompts than the amendment records"
                for bundle in bundles if bundle_prompt_digest(prompts[bundle.scenario_id]) != bundle.prompts_sha256]
    problems += [f"{LEDGER_FNAME} gained a row of {record.scenario_id[:12]} at attempt {record.semantic_attempt_id}"
                 for record in after if record.scenario_id not in names or record.semantic_attempt_id not in window]
    problems += [f"{record.scenario_id[:12]}/{component.component_id} at attempt {record.semantic_attempt_id} was "
                 "sent another prompt than its earlier attempts"
                 for record in after if record.scenario_id in names for component in record.components
                 if component.prompt_sha256 != prompts[record.scenario_id].get(component.component_id)]
    problems += [f"{RAW_OUTPUTS_FNAME} gained a row of {str(row.get('scenario_id'))[:12]} at attempt "
                 f"{row.get('semantic_attempt_id')}" for row in appended[RAW_OUTPUTS_FNAME]
                 if row.get("scenario_id") not in names or row.get("semantic_attempt_id") not in window]
    problems += [f"{REQUEST_BLOCKS_FNAME} gained block {row.get('request_block_id')}"
                 for row in appended[REQUEST_BLOCKS_FNAME]
                 if row.get("semantic_attempt_id") not in window or not set(row.get("scenario_ids") or ()) <= names]
    # Every run logs itself before its first attempt, so rows with no run logged beside them cannot be the resume's.
    if after and not appended[RUNS_FNAME]:
        problems.append(f"{LEDGER_FNAME} gained {len(after)} row(s), and {RUNS_FNAME} logs no run since the amendment "
                        "was recorded")
    # A whole run cut from the log would give its rows to the run before it.
    unbound = run_log_problem(draw_dir, required=bool(after) and any(LEDGER_ROWS_KEY in row
                                                                     for row in appended[RUNS_FNAME]))
    if unbound:
        problems.append(unbound)
    unlogged = unlogged_stacks(draw_dir, state.records)
    if unlogged:
        problems.append(unlogged)
    # Only a local judge's run that appended rows is held to its stack.
    spans, wrong = run_spans(appended[RUNS_FNAME], start=split, end=len(state.records))
    if wrong:
        problems.append(f"{RUNS_FNAME}: {wrong}")
    recorded = json.loads((draw_dir / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
    for span in spans:
        row = span.row
        if code_state_key(row) != amendment.code_state or row.get("max_semantic_attempts") != amendment.new_cap:
            problems.append(f"{RUNS_FNAME} logs a run of {describe_code_state(code_state_key(row))} stating "
                            f"{row.get('max_semantic_attempts')} attempts")
        changed = runtime_changes(recorded.get("runtime") or {}, row.get("runtime") or {})
        if span.stack_change:
            problems.append(f"{RUNS_FNAME} logs a stack change ({describe_changes(span.stack_change)})")
        elif (recorded.get("model_type") == LOCAL_MODEL_TYPE and changed
              and (span.appended or (span.appended is None and after))):
            problems.append(f"{RUNS_FNAME} logs a run on another stack ({describe_changes(changed)}) and no stack "
                            "change, and the rows it appended " + ("cannot be told from the others'"
                                                                   if span.appended is None else "are its"))
    if problems:
        raise AmendmentError(f"{where}: " + "; ".join(problems))
    check_seeds(after, generation_seed_base=generation_seed_base, where=where,
                identity_id=_identity_id(draw_dir, condition_id))
    outcomes = {sid: _outcome(state, sid, amendment.new_cap) for sid in sorted(names)}
    pending = sorted(sid[:12] for sid, entry in outcomes.items() if entry["outcome"] == OUTCOME_PENDING)
    if pending and finished:
        raise AmendmentError(
            f"{where}: {pending} still pending under the amended cap: resume the draw under the amendment (relaunch "
            "it stating the new cap, from the amendment's code state) before reading it"
        )
    return {"appended_attempt_rows": len(after), "bundles": outcomes,
            "resumed_by": describe_code_state(amendment.code_state)}


def _outcome(state: LedgerState, scenario_id: str, new_cap: int) -> dict[str, Any]:
    if state.is_accepted(scenario_id):
        attempt = state.accepted[scenario_id].semantic_attempt_id
        return {"outcome": OUTCOME_ACCEPTED, "accepted_at_attempt": attempt, "attempts": attempt + 1}
    used = state.next_attempt(scenario_id)
    return {"outcome": OUTCOME_EXHAUSTED if used >= new_cap else OUTCOME_PENDING, "attempts": used}


def _replay_record(root: Path, manifest: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    path = root / REPLAY_RECORD_FNAME
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_bytes())
        found = (record["campaign_id"], record["label_rule"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AmendmentError(f"{path} is no label replay's record ({error!r}): the root's amended draws cannot be "
                             "read as a re-score's") from error
    if found != (manifest["campaign_id"], manifest["label_rule"]):
        raise AmendmentError(f"{path} records a re-score of campaign {found[0]!r} under {found[1]!r}, not this root's "
                             f"{manifest['campaign_id']!r} under {manifest['label_rule']!r}")
    return record


def check_root_draw(
    root: Path | str, amendment: CapAmendment, *, campaign_id: str, condition_id: str, decoder_draw_id: int,
    generation_seed_base: int, finished: bool = True,
) -> dict[str, dict[str, Any]]:
    """Hold a campaign root's draw to the amendment, resumed in place or merged from shards; return ``{scenario id:
    outcome}``."""
    root = Path(root)
    draw_dir = root / condition_id / draw_dirname(decoder_draw_id)
    in_place = check_resumed_draw(draw_dir, amendment, campaign_id=campaign_id, condition_id=condition_id,
                                  decoder_draw_id=decoder_draw_id, shard=None,
                                  generation_seed_base=generation_seed_base, finished=finished)
    outcomes = dict(in_place["bundles"]) if in_place is not None else {}
    layouts: dict[int, list[AmendedBundle]] = {}
    for bundle in amendment.bundles:
        if (bundle.condition_id, bundle.decoder_draw_id) == (condition_id, decoder_draw_id) and bundle.shard is not None:
            parsed = parse_shard_dirname(bundle.shard)
            if parsed is None:
                raise AmendmentError(f"{bundle.describe()}: {bundle.shard!r} names no shard root")
            layouts.setdefault(parsed[1], []).append(bundle)
    from ruleforge.evaluate.shard_merge import check_merged_draw  # shard_merge imports this module

    for count, named in sorted(layouts.items()):
        where = f"{draw_dir} (amended: {', '.join(bundle.describe() for bundle in named)})"
        try:
            held = check_merged_draw(root, condition_id, decoder_draw_id, num_shards=count).get("amended", {})
        except SamplingDesignError as error:
            raise AmendmentError(f"{where}: merged from shard roots, it cannot be held to them ({error})") from error
        for bundle in named:
            if bundle.shard not in held:
                raise AmendmentError(f"{where}: {bundle.shard} holds no amended bundle of the draw to hold it to")
            outcomes[bundle.scenario_id] = held[bundle.shard]["bundles"][bundle.scenario_id]
    return outcomes


def _derived_outcomes(
    root: Path, record: Mapping[str, Any], amendment: CapAmendment, condition_id: str, decoder_draw_id: int, *,
    campaign_id: str, generation_seed_base: int,
) -> dict[str, dict[str, Any]]:
    # A derived root's draw is held to the re-score its record describes, and the source that re-score read to the
    # amendment.
    draw_dir = root / condition_id / draw_dirname(decoder_draw_id)
    bundles = [bundle for bundle in amendment.bundles
               if (bundle.condition_id, bundle.decoder_draw_id) == (condition_id, decoder_draw_id)]
    where = f"{draw_dir} (a derived root's amended draw, {', '.join(bundle.describe() for bundle in bundles)})"
    try:
        condition = record["conditions"][condition_id]
        entry = condition["draws"][str(decoder_draw_id)]
        replayed, sources = dict(entry["replayed_draw"]), list(entry["source_ledgers"])
        fingerprint, from_shards, origin = condition["fingerprint"], condition["from_shards"], str(condition["source"])
    except (KeyError, TypeError, ValueError) as error:
        raise AmendmentError(f"{where}: {REPLAY_RECORD_FNAME} records no re-score of it to hold it to ({error!r}); "
                             "it is no re-score this version reads") from error
    try:
        provenance = json.loads((draw_dir / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
        written = draw_digests(draw_dir, REPLAYED_DIGESTS)
    except (OSError, ValueError) as error:
        raise AmendmentError(f"{where}: its files cannot be read to hold them to the re-score ({error!r})") from error
    problems = []
    if provenance.get("condition_fingerprint") != fingerprint:
        problems.append(f"its {DRAW_PROVENANCE_FNAME} names another condition than the re-score wrote")
    changed = [name for key, name in REPLAYED_DIGESTS.items() if written[key] != replayed.get(key)]
    if changed:
        problems.append(f"its {changed} are not the files the re-score wrote")
    labels = sorted({str(bundle.shard) for bundle in bundles if bundle.shard is not None})
    held_in_place = any(bundle.shard is None for bundle in bundles)
    source_root = None
    try:
        source_root = resolve(origin)
        read = {resolve(str(source["path"])).resolve(): source for source in sources}
        if from_shards:
            named = {(source_root / label / condition_id / draw_dirname(decoder_draw_id)).resolve() for label in labels}
            if held_in_place:
                problems.append(f"{REPLAY_RECORD_FNAME} names no draw of a campaign root as its source")
            elif not named <= set(read):
                problems.append(f"{REPLAY_RECORD_FNAME} names no draw of the shard roots holding its amended "
                                "bundles as its source")
            elif any(draw_digests(path) != {key: source.get(key) for key in LEDGER_DIGESTS}
                     for path, source in read.items()):
                problems.append(f"its source shard draws in {source_root} are not the ones the re-score read")
            elif any(cap_amendment(load_sampling_manifest(source_root / label / SAMPLING_MANIFEST_FNAME)) != amendment
                     for label in labels):
                problems.append(f"its source shard roots in {source_root} record another cap amendment than this root")
        else:
            source_dir = source_root / condition_id / draw_dirname(decoder_draw_id)
            if list(read) != [source_dir.resolve()]:
                problems.append(f"{REPLAY_RECORD_FNAME} names no draw of a campaign root as its source")
            elif draw_digests(source_dir) != {key: sources[0].get(key) for key in LEDGER_DIGESTS}:
                problems.append(f"its source draw {source_dir} is not the one the re-score read")
            elif cap_amendment(load_sampling_manifest(source_root / SAMPLING_MANIFEST_FNAME)) != amendment:
                problems.append(f"its source root {source_root} records another cap amendment than this root")
    except (OSError, ValueError, KeyError, TypeError, ArtifactRootError) as error:
        problems.append(f"its source draw cannot be read to hold it to the amendment ({error!r})")
    if problems:
        raise AmendmentError(f"{where}: " + "; ".join(problems))
    if from_shards:
        for label in labels:
            check_resumed_draw(source_root / label / condition_id / draw_dirname(decoder_draw_id), amendment,
                               campaign_id=campaign_id, condition_id=condition_id, decoder_draw_id=decoder_draw_id,
                               shard=label, generation_seed_base=generation_seed_base)
    else:
        check_root_draw(source_root, amendment, campaign_id=campaign_id, condition_id=condition_id,
                        decoder_draw_id=decoder_draw_id, generation_seed_base=generation_seed_base)
    state = DrawLedger(draw_dir, campaign_id=campaign_id, condition_id=condition_id,
                       decoder_draw_id=decoder_draw_id).load()
    return {bundle.scenario_id: _outcome(state, bundle.scenario_id, amendment.new_cap) for bundle in bundles}


def disclosure(root: Path | str, manifest: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return what an analysis of ``root`` reports of its cap amendment (record, outcomes, code states) once each
    amended draw is held to it, or None."""
    amendment = cap_amendment(manifest)
    if amendment is None:
        return None
    root = Path(root)
    refuse_published(root, "a cap amendment's disclosure", "the amended draws' attempt ledgers")
    campaign_id = str(manifest["campaign_id"])
    seed_base = int(manifest["generation_seed_base"])
    record = _replay_record(root, manifest)
    held: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    bundles = []
    for bundle in amendment.bundles:
        draw_dir = root / bundle.condition_id / draw_dirname(bundle.decoder_draw_id)
        entry: dict[str, Any] = {"condition_id": bundle.condition_id, "decoder_draw_id": bundle.decoder_draw_id,
                                 "scenario_id": bundle.scenario_id, "held_by": bundle.shard or "campaign root",
                                 "attempts_at_old_cap": bundle.attempts}
        if not (draw_dir / LEDGER_FNAME).is_file():
            if (draw_dir / COMPLETION_FNAME).is_file():
                raise AmendmentError(f"{draw_dir} holds a {COMPLETION_FNAME} but no {LEDGER_FNAME}, which the amended "
                                     "bundle's outcome is read from; disclose the amendment from a root that holds it")
            bundles.append({**entry, "outcome": OUTCOME_ABSENT})
            continue
        key = (bundle.condition_id, bundle.decoder_draw_id)
        if key not in held:
            held[key] = (
                _derived_outcomes(root, record, amendment, *key, campaign_id=campaign_id, generation_seed_base=seed_base)
                if record is not None else
                check_root_draw(root, amendment, campaign_id=campaign_id, condition_id=key[0], decoder_draw_id=key[1],
                                generation_seed_base=seed_base, finished=False)
            )
        bundles.append({**entry, **held[key][bundle.scenario_id]})
    conditions = sorted({bundle.condition_id for bundle in amendment.bundles})
    code_states = {
        condition_id: {describe_code_state(state): draws
                       for state, draws in sorted(recorded_code_states(root, condition_id).items(), key=str)}
        for condition_id in conditions
    }
    return {**amendment.summary(), "bundles": bundles, "code_states": code_states}

