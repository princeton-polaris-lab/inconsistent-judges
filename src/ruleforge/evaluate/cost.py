from __future__ import annotations

import dataclasses
import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import yaml

from ruleforge.core.paths import artifact_uri_or_path
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    COMPONENT_KINDS,
    DRAW_PROVENANCE_FNAME,
    KIND_FULL_SET,
    RETRY_SCOPE_COMPONENT,
    ConditionSpec,
    draw_dirname,
    refuse_published,
)
from ruleforge.evaluate.identity import component_kind as kind_of
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import (
    request_namespace,
    sha256_file,
)
from ruleforge.evaluate.ledger import LEDGER_FNAME, AttemptRecord, ComponentRecord, DrawLedger, LedgerState
from ruleforge.evaluate.outputs import CampaignRoot
from ruleforge.evaluate.runner import REQUEST_BLOCKS_FNAME, STATE_COMPLETE
from ruleforge.env import package_path
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.record_names import PARTIAL_KIND
from ruleforge.judges.batch_api import (
    PARTIAL_FNAME,
    PARTIAL_SCHEMA_VERSION,
    BatchUsage,
    batch_custom_id,
    billed_body,
    prompt_of_body,
)
from ruleforge.core.resume import count_complete_jsonl_records

# Bumped when a record this module writes changes shape.
BATCH_COST_SCHEMA_VERSION = 4
RATES_SCHEMA_VERSION = 1

USAGE_FNAME = "usage.json"
DEFAULT_RATES = package_path("configs/pricing/batch_rates.yaml")

# Buckets an extrapolation scales with the prompt, and those it scales with the generation.
INPUT_FIELDS = ("input_tokens", "cached_tokens", "cache_write_tokens")
OUTPUT_FIELDS = ("output_tokens", "reasoning_tokens")
TOKEN_FIELDS = (*INPUT_FIELDS, *OUTPUT_FIELDS)
USAGE_FIELDS = ("requests", *TOKEN_FIELDS)
# USD per million tokens.
RATE_FIELDS = ("input", "cached_input", "cache_write", "output")
_RATE_TEXT_FIELDS = ("tier", "as_of", "source")
_RATE_KEYS = frozenset({*RATE_FIELDS, *_RATE_TEXT_FIELDS})
_IDENTITY_KEYS = frozenset(field.name for field in dataclasses.fields(ConditionSpec))
JUDGE_FIELDS = ("model_name", "model_type", "model_path", "adapter_path", "adapter_sha256", "adapter_config_sha256",
                "model_weights_sha256", "serving_dtype", "endpoint", "reasoning_effort", "enable_thinking",
                "temperature", "max_new_tokens", "send_max_output_tokens", "seed_capability",
                "sampling_truncation", "judge_extension")
PROMPT_FIELDS = ("prompt_style", "individual_rule_context", "judge_type", "evaluation_mode", "rule_set_name",
                 "rule_set_sha256", "rule_levels_sha256", "prompt_shape_version", "extra", "task")
PROTOCOL_FIELDS = ("forced_json_requirement", "validation_schema_version", "label_rule", "retry_scope", "strict_think",
                   "response_contract")
# (full-set components, rule components)
Shape = tuple[int, int]
_FRESH_TOLERANCE = 1e-9


class BatchCostError(SamplingDesignError):
    """A usage record, rate file, plan or campaign that cannot be read as this module reads it."""


@dataclass(frozen=True)
class Usage:
    """Token buckets of one or more requests: counts as read, floats once scaled to a design."""

    requests: float = 0
    input_tokens: float = 0
    cached_tokens: float = 0
    cache_write_tokens: float = 0
    output_tokens: float = 0
    reasoning_tokens: float = 0

    @classmethod
    def read(cls, path: Path) -> "Usage":
        """A batch's ``usage.json``: exactly :data:`USAGE_FIELDS`, each a non-negative count."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise BatchCostError(f"{path}: not a readable usage record ({type(error).__name__})") from None
        if not isinstance(payload, dict) or set(payload) != set(USAGE_FIELDS):
            raise BatchCostError(f"{path}: a usage record holds exactly {list(USAGE_FIELDS)}")
        for name in USAGE_FIELDS:
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BatchCostError(f"{path}: {name} must be a non-negative count, got {value!r}")
        return cls(**payload)

    @classmethod
    def of_body(cls, body: Mapping[str, Any]) -> "Usage":
        """One billed OpenAI response body's usage, as the batch layer counts it into ``usage.json``."""
        counts = BatchUsage()
        counts.add_body(body)
        return cls(**counts.to_dict())

    @classmethod
    def _summed(cls, add_usage: Callable[[BatchUsage, Mapping[str, Any]], None], payload: Mapping[str, Any]) -> "Usage":
        counts = BatchUsage()
        add_usage(counts, payload)
        return cls(**counts.to_dict())

    @classmethod
    def of_anthropic_message(cls, message: Mapping[str, Any]) -> "Usage":
        """Return one billed Anthropic message's usage, counted as ``anthropic_batch.add_usage`` counts it."""
        from ruleforge.judges.anthropic_batch import add_usage

        return cls._summed(add_usage, message)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(**{name: getattr(self, name) + getattr(other, name) for name in USAGE_FIELDS})

    @property
    def fresh_input_tokens(self) -> float:
        """Input neither read from the cache nor written to it, billed at the plain input rate."""
        return self.input_tokens - self.cached_tokens - self.cache_write_tokens

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in USAGE_FIELDS}


def total(usages: Iterable[Usage]) -> Usage:
    return sum(usages, Usage())


def _openai_result(record: Mapping[str, Any]) -> tuple[Any, Optional[Usage]]:
    body = billed_body(record)
    return record.get("custom_id"), None if body is None else Usage.of_body(body)


def _anthropic_result(record: Mapping[str, Any]) -> tuple[Any, Optional[Usage]]:
    from ruleforge.judges.anthropic_batch import outcome_from_record

    _, message = outcome_from_record(record)
    return record.get("custom_id"), None if message is None else Usage.of_anthropic_message(message)


# How a batch's output lines name and bill each request, by ``model_type``.
RESULT_READERS: dict[str, Callable[[Mapping[str, Any]], tuple[Any, Optional[Usage]]]] = {
    "openai_batch": _openai_result,
    "anthropic_batch": _anthropic_result,
}


def _openai_request(line: Mapping[str, Any]) -> tuple[Any, dict[str, str]]:
    return line.get("custom_id"), prompt_of_body(line["body"], line["url"])


def _anthropic_request(line: Mapping[str, Any]) -> tuple[Any, dict[str, str]]:
    from ruleforge.judges.anthropic_batch import prompt_of_params

    return line.get("custom_id"), prompt_of_params(line["params"])


# How a batch's input lines name each request and its prompt, by ``model_type``.
REQUEST_READERS: dict[str, Callable[[Mapping[str, Any]], tuple[Any, dict[str, str]]]] = {
    "openai_batch": _openai_request,
    "anthropic_batch": _anthropic_request,
}


def sent_requests(directory: Path, model_type: str) -> list[tuple[str, str]]:
    """Return each request of a batch's input files in order: its custom id and its prompt's digest."""
    read = REQUEST_READERS.get(model_type)
    if read is None:
        raise BatchCostError(f"batch {directory.name}: a {model_type!r} judge's input lines are not read request by "
                             f"request (only {', '.join(sorted(REQUEST_READERS))} judges' are)")
    sent = []
    for path in sorted((directory / "inputs").glob("part-*.jsonl")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                custom_id, entry = read(json.loads(line))
                sent.append((str(custom_id), prompt_sha256(entry)))
    return sent


def request_usage(directory: Path, model_type: str, whole: Optional[Usage] = None) -> dict[str, Usage]:
    """Return each billed request's usage in a batch's outputs, by custom id, refused unless they sum to its
    ``usage.json``."""
    def read_usage(path: Path) -> Usage:
        return Usage.read(path) if whole is None else whole

    read = RESULT_READERS.get(model_type)
    if read is None:
        raise BatchCostError(f"batch {directory.name}: a {model_type!r} judge's outputs are not read request by "
                             f"request (only {', '.join(sorted(RESULT_READERS))} judges' are); its usage is read "
                             f"whole, from {USAGE_FNAME}, so it cannot be restricted to some scenarios or one kind of "
                             f"component")
    parts = sorted(path for path in (directory / "outputs").glob("part-*.jsonl")
                   if not path.name.endswith(".errors.jsonl"))
    if not parts:
        marker = read_partial_marker(directory, read_usage)
        if marker is None or marker["requests_returned"]:
            raise BatchCostError(f"batch {directory.name} holds no downloaded outputs to read request by request")
        return {}
    usages: dict[str, Usage] = {}
    seen: set[str] = set()
    for path in parts:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    raise BatchCostError(f"{path}:{number} is not a result record") from None
                custom_id, usage = read(record) if isinstance(record, dict) else (None, None)
                if not isinstance(custom_id, str) or custom_id in seen:
                    raise BatchCostError(f"{path}:{number}: a result names its own custom id, once")
                seen.add(custom_id)
                if usage is not None:
                    usages[custom_id] = usage
    if total(usages.values()) != read_usage(directory / USAGE_FNAME):
        raise BatchCostError(f"batch {directory.name}: its outputs' billed requests do not sum to its {USAGE_FNAME}")
    return usages


@dataclass(frozen=True)
class Rates:
    """One model's Batch prices, USD per million tokens, with when and where they were read."""

    model: str
    tier: str
    input: float
    cached_input: float
    cache_write: float
    output: float
    as_of: str
    source: str


def _versioned_yaml(path: Path | str, key: str, version: int) -> Any:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise BatchCostError(f"{path}: not a readable YAML file ({type(error).__name__})") from None
    if not isinstance(payload, dict) or set(payload) != {"schema_version", key}:
        raise BatchCostError(f"{path}: the file holds exactly schema_version and {key}")
    if payload["schema_version"] != version:
        raise BatchCostError(f"{path}: schema_version {payload['schema_version']!r}, this build reads {version}")
    return payload[key]


def load_rates(path: Path | str) -> dict[str, Rates]:
    """The rate file: ``schema_version`` and ``rates``, a mapping of model id to :class:`Rates`' fields."""
    entries = _versioned_yaml(path, "rates", RATES_SCHEMA_VERSION)
    if not isinstance(entries, dict) or not entries:
        raise BatchCostError(f"{path}: rates must map model ids to their prices")
    rates = {}
    for model, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != _RATE_KEYS:
            raise BatchCostError(f"{path}: {model!r} must give exactly {sorted(_RATE_KEYS)}")
        for name in RATE_FIELDS:
            value = entry[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise BatchCostError(f"{path}: {model!r} {name} must be a non-negative price, got {value!r}")
        for name in _RATE_TEXT_FIELDS:
            if not isinstance(entry[name], str) or not entry[name].strip():
                raise BatchCostError(f"{path}: {model!r} {name} must be text (quote a date)")
        rates[str(model)] = Rates(model=str(model), **entry)
    return rates


def price(usage: Usage, rates: Rates) -> dict[str, float]:
    """USD per bucket and in total (reasoning is inside ``output_tokens``, never charged twice)."""
    fresh = usage.fresh_input_tokens
    if fresh < -_FRESH_TOLERANCE * usage.input_tokens:
        raise BatchCostError(
            f"cached ({usage.cached_tokens}) and cache-written ({usage.cache_write_tokens}) tokens exceed the input "
            f"({usage.input_tokens}); this formula prices input that counts both")
    parts = {
        "input": max(fresh, 0.0) * rates.input,
        "cached_input": usage.cached_tokens * rates.cached_input,
        "cache_write": usage.cache_write_tokens * rates.cache_write,
        "output": usage.output_tokens * rates.output,
    }
    parts = {name: value / 1e6 for name, value in parts.items()}
    return {**parts, "total": sum(parts.values())}


@dataclass(frozen=True)
class SentComponent:
    """A component a request block sent that no ledger row records, read from the block's plan and its batch's input
    file."""

    component_id: str
    provider_request_ref: str
    request_block_ref: str
    prompt_sha256: str


@dataclass(frozen=True)
class SentRow:
    """A bundle's components a request block sent that no ledger row records."""

    scenario_id: str
    semantic_attempt_id: int
    request_block_id: str
    components: tuple[SentComponent, ...]


@dataclass(frozen=True)
class PartialBatch:
    """A collected batch that ended without completing: ``returned`` of its ``expected`` requests answered and billed.
    The missing requests are not a random sample: they include the slowest and those that failed at the provider."""

    key: str
    returned: int
    expected: int
    states: tuple[str, ...]
    missing_reasons: Mapping[str, int]
    draw: Optional[int] = None
    block_id: Optional[str] = None
    rows: tuple[SentRow, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"batch": self.key, "returned": self.returned, "expected": self.expected, "states": list(self.states),
                "missing_reasons": dict(self.missing_reasons), "draw": self.draw, "request_block_id": self.block_id,
                "from_block_plan": bool(self.rows)}


def read_partial_marker(
    directory: Path, read_usage: Callable[[Path], Usage] = Usage.read,
) -> Optional[dict[str, Any]]:
    """Return a batch directory's ``partial.json``, or None; refused unless it matches the batch's ``usage.json``."""
    path = directory / PARTIAL_FNAME
    if not path.is_file():
        return None
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BatchCostError(f"{path}: not a readable partial marker ({type(error).__name__})") from None
    if (not isinstance(marker, dict) or marker.get("kind") != PARTIAL_KIND
            or marker.get("schema_version") != PARTIAL_SCHEMA_VERSION):
        raise BatchCostError(f"{path}: not a partial marker of schema {PARTIAL_SCHEMA_VERSION}")
    usage = marker.get("usage")
    if not isinstance(usage, dict) or set(usage) != set(USAGE_FIELDS) or Usage(**usage) != read_usage(
            directory / USAGE_FNAME):
        raise BatchCostError(f"{directory.name}: its {USAGE_FNAME} is not the usage its {PARTIAL_FNAME} was written "
                             f"with; collect it again (ruleforge campaign collect)")
    return marker


def _partial_batch(key: str, marker: Mapping[str, Any], **block: Any) -> PartialBatch:
    return PartialBatch(
        key=key, returned=int(marker["requests_returned"]), expected=int(marker["requests_expected"]),
        states=tuple(sorted({str(part["state"]) for part in marker["parts"].values()})),
        missing_reasons=dict(marker["missing_reasons"]), **block)


def _unrecorded_blocks(campaign: CampaignRoot, condition_id: str, recorded: set[str]) -> Iterator[tuple[int, dict]]:
    # A block with no members is left out: only a rebase empties one, and its batch was priced in the parent's reading.
    for draw in range(campaign.n_draws):
        path = campaign.root / condition_id / draw_dirname(draw) / REQUEST_BLOCKS_FNAME
        count = count_complete_jsonl_records(path)
        if not count:
            continue
        with path.open(encoding="utf-8") as handle:
            for _, line in zip(range(count), handle):
                block = json.loads(line)
                if block["scenario_ids"] and block["request_block_id"] not in recorded:
                    yield draw, block


def block_namespace(campaign_id: str, condition: ConditionSpec, block_id: str) -> str:
    """Return the provider namespace of a recorded block, taken over the block's identity id as the runner addressed
    it."""
    return request_namespace(condition.identity_block_id(campaign_id, block_id))


def _block_rows(
    campaign: CampaignRoot, condition: ConditionSpec, draw: int, block: Mapping[str, Any], directory: Path,
) -> tuple[SentRow, ...]:
    """Return what a request block sent; refused unless the batch's requests are exactly the plan's, in order."""
    draw_dir = campaign.root / condition.condition_id / draw_dirname(draw)
    component_ids = tuple(json.loads((draw_dir / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))["component_ids"])
    members = [str(scenario_id) for scenario_id in block["scenario_ids"]]
    attempt = int(block["semantic_attempt_id"])
    if condition.retry_scope == RETRY_SCOPE_COMPONENT:
        ledger = DrawLedger(draw_dir, campaign_id=campaign.campaign_id, condition_id=condition.condition_id,
                            decoder_draw_id=draw)
        state = ledger.load() if ledger.ledger_path.is_file() else LedgerState()
        planned = [state.components_at(scenario_id, attempt, component_ids) for scenario_id in members]
    else:
        planned = [component_ids] * len(members)
    namespace = block_namespace(campaign.campaign_id, condition, str(block["request_block_id"]))
    sent = sent_requests(directory, condition.model_type)
    expected = [batch_custom_id(index, 0, None, namespace) for index in range(sum(map(len, planned)))]
    if [custom_id for custom_id, _ in sent] != expected:
        raise BatchCostError(
            f"batch {directory.name}: its {len(sent)} request(s) are not request block {block['request_block_id']}'s "
            f"plan of {len(expected)} (its members' components, in order, under the block's namespace)")
    rows, position = [], 0
    for scenario_id, components in zip(members, planned):
        rows.append(SentRow(scenario_id=scenario_id, semantic_attempt_id=attempt,
                            request_block_id=str(block["request_block_id"]), components=tuple(
                                SentComponent(component_id=component_id, provider_request_ref=custom_id,
                                              request_block_ref=directory.name, prompt_sha256=digest)
                                for component_id, (custom_id, digest) in zip(
                                    components, sent[position:position + len(components)]))))
        position += len(components)
    return tuple(rows)


@dataclass(frozen=True)
class ConditionUsage:
    """One condition's batches, as its ledgers name them, and their summed usage."""

    campaign: CampaignRoot
    condition: ConditionSpec
    batch_root: Path
    batches: tuple[str, ...]
    # Named batches with no usage.json: requests whose usage is not counted.
    missing: tuple[str, ...]
    usage: Usage
    batch_usages: Mapping[str, Usage] = field(hash=False)
    draws: int
    scenarios: frozenset[str]
    bundles: int
    grid_requests: int
    # Each bundle's first-attempt components by kind.
    shapes: frozenset[Shape]
    scope: Optional[frozenset[str]] = None
    kind: Optional[str] = None
    partial: tuple[PartialBatch, ...] = ()
    partial_requests: tuple[int, int] = (0, 0)
    first_attempt: bool = False

    @property
    def campaign_id(self) -> str:
        return self.campaign.campaign_id

    @property
    def condition_id(self) -> str:
        return self.condition.condition_id

    @property
    def label(self) -> str:
        return f"{self.campaign_id}/{self.condition_id}"

    @property
    def measured(self) -> bool:
        """Every request it sent is counted: it named a batch, and every batch it named has its usage."""
        return bool(self.batches) and not self.missing

    def value(self, key: str) -> Any:
        """A condition-identity field's value here."""
        return getattr(self.condition, key)


@dataclass(frozen=True)
class Measured:
    """A sum of conditions: what a report row or an estimate's source stands on."""

    label: str
    members: tuple[str, ...]
    batches: int
    usage: Usage
    scenarios: frozenset[str]
    bundles: int
    grid_requests: int
    measured: bool
    shapes: frozenset[Shape]
    partial: tuple[PartialBatch, ...] = ()
    partial_requests: tuple[int, int] = (0, 0)

    @property
    def complete(self) -> bool:
        """Return whether every request its batches were sent returned."""
        return self.partial_requests[0] == self.partial_requests[1]

    @property
    def retry_overhead(self) -> Optional[float]:
        """Return requests billed beyond one pass over every attempted bundle, as a fraction of it; None unless all are
        counted."""
        if not self.measured or not self.grid_requests or not self.complete:
            return None
        return self.usage.requests / self.grid_requests - 1

    def partial_text(self) -> str:
        """``returned/expected`` of its partial batches' requests, ``-`` without one."""
        returned, expected = self.partial_requests
        return f"{returned:,}/{expected:,}" if expected else "-"

    def to_dict(self) -> dict[str, Any]:
        returned, expected = self.partial_requests
        return {"label": self.label, "conditions": list(self.members), "batches": self.batches,
                "usage": self.usage.to_dict(), "scenarios": len(self.scenarios), "bundles": self.bundles,
                "grid_requests": self.grid_requests, "shapes": _shapes_record(self.shapes),
                "measured": self.measured, "retry_overhead": self.retry_overhead,
                "partial": None if not self.partial else {
                    "returned": returned, "expected": expected,
                    "batches": [batch.to_dict() for batch in self.partial]}}


def combine(label: str, conditions: Sequence[ConditionUsage]) -> Measured:
    return Measured(
        label=label,
        members=tuple(condition.label for condition in conditions),
        batches=sum(len(condition.batches) for condition in conditions),
        usage=total(condition.usage for condition in conditions),
        scenarios=frozenset().union(*(condition.scenarios for condition in conditions)),
        bundles=sum(condition.bundles for condition in conditions),
        grid_requests=sum(condition.grid_requests for condition in conditions),
        measured=all(condition.measured for condition in conditions),
        shapes=frozenset().union(*(condition.shapes for condition in conditions)),
        partial=tuple(batch for condition in conditions for batch in condition.partial),
        partial_requests=(sum(condition.partial_requests[0] for condition in conditions),
                          sum(condition.partial_requests[1] for condition in conditions)),
    )


def _shapes_record(shapes: Iterable[Shape]) -> list[dict[str, int]]:
    return [dict(zip(COMPONENT_KINDS, shape)) for shape in sorted(shapes)]


@dataclass(frozen=True)
class Attribution:
    """Every given root's conditions with their usage, and the batch root's batches no given root names."""

    batch_root: Path
    roots: tuple[tuple[str, Path], ...]
    conditions: tuple[ConditionUsage, ...]
    unattributed: Mapping[str, Usage]
    unread: tuple[str, ...]
    missing: Mapping[str, str]
    unattributed_partial: tuple[str, ...] = ()

    @property
    def unattributed_usage(self) -> Usage:
        return total(self.unattributed.values())


@dataclass
class _LedgerReading:
    keys: set[str] = field(default_factory=set)
    blocks: set[str] = field(default_factory=set)
    # Batch key -> custom id -> (scenario, component).
    requests: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    scenarios: set[str] = field(default_factory=set)
    draws: int = 0
    bundles: int = 0
    grid_requests: int = 0
    shapes: set[Shape] = field(default_factory=set)


def _check_kind(kind: Optional[str], *, where: str) -> None:
    if kind is not None and kind not in COMPONENT_KINDS:
        raise BatchCostError(f"{where}: component_kind {kind!r} is not one of {COMPONENT_KINDS}")


_Rows = list[tuple[AttemptRecord | SentRow, list[tuple[ComponentRecord | SentComponent, str]]]]


def _ledger_rows(
    campaign: CampaignRoot, condition_id: str, scenarios: Optional[frozenset[str]], kind: Optional[str],
    partial: Sequence[PartialBatch] = (), first_attempt: bool = False,
) -> Iterator[tuple[DrawLedger, _Rows]]:
    sent: dict[int, list[SentRow]] = {}
    for batch in partial:
        if batch.rows:
            sent.setdefault(batch.draw, []).extend(batch.rows)
    ledgers = {ledger.decoder_draw_id: ledger for ledger in campaign.ledgers(condition_id)}
    for draw in sorted(ledgers.keys() | sent.keys()):
        ledger = ledgers.get(draw) or DrawLedger(campaign.root / condition_id / draw_dirname(draw),
                                                 campaign_id=campaign.campaign_id, condition_id=condition_id,
                                                 decoder_draw_id=draw)
        records = ledger.load().records if draw in ledgers else []
        rows: _Rows = []
        for record in [*records, *sent.get(draw, ())]:
            if (scenarios is not None and record.scenario_id not in scenarios) or (
                    first_attempt and record.semantic_attempt_id != 0):
                continue
            kept = [(component, kind_of(component.component_id)) for component in record.components]
            rows.append((record, [(component, its_kind) for component, its_kind in kept
                                  if kind is None or its_kind == kind]))
        yield ledger, rows


def _read_ledgers(
    campaign: CampaignRoot, condition_id: str, scenarios: Optional[frozenset[str]] = None,
    kind: Optional[str] = None, *, by_request: bool = False, partial: Sequence[PartialBatch] = (),
    first_attempt: bool = False,
) -> _LedgerReading:
    by_request = by_request or scenarios is not None or kind is not None or first_attempt
    reading = _LedgerReading()
    for ledger, rows in _ledger_rows(campaign, condition_id, scenarios, kind, partial, first_attempt):
        first: dict[str, tuple[int, Shape]] = {}
        for record, kept in rows:
            reading.blocks.add(record.request_block_id)
            for component, _ in kept:
                if not component.request_block_ref:
                    continue
                key = Path(component.request_block_ref).name
                reading.keys.add(key)
                if by_request:
                    if not component.provider_request_ref:
                        raise BatchCostError(f"{ledger.ledger_path}: a component of batch {key} names no request")
                    reading.requests.setdefault(key, {})[component.provider_request_ref] = (
                        record.scenario_id, component.component_id)
            earliest = first.get(record.scenario_id)
            if earliest is None or record.semantic_attempt_id < earliest[0]:
                full = sum(its_kind == KIND_FULL_SET for _, its_kind in kept)
                first[record.scenario_id] = (record.semantic_attempt_id, (full, len(kept) - full))
        reading.draws += 1
        reading.scenarios.update(first)
        reading.bundles += len(first)
        reading.grid_requests += sum(sum(shape) for _, shape in first.values())
        reading.shapes.update(shape for _, shape in first.values())
    return reading


def _narrowed(
    condition: ConditionUsage, scenarios: Optional[Iterable[str]], kind: Optional[str],
) -> tuple[Optional[frozenset[str]], Optional[str]]:
    _check_kind(kind, where=condition.label)
    if kind is not None and condition.kind not in (None, kind):
        raise BatchCostError(f"{condition.label} is restricted to its {condition.kind} components; it has no {kind} "
                             f"components to read")
    scope = condition.scope
    wanted = None if scenarios is None else frozenset(scenarios)
    if wanted is not None and not condition.scenarios <= wanted:
        scope = wanted if scope is None else scope & wanted
    return scope, kind or condition.kind


def _reading(
    condition: ConditionUsage, scope: Optional[frozenset[str]], kind: Optional[str],
    first_attempt: Optional[bool] = None,
) -> ConditionUsage:
    first_attempt = condition.first_attempt if first_attempt is None else first_attempt
    reading = _read_ledgers(condition.campaign, condition.condition_id, scope, kind, by_request=True,
                            partial=condition.partial, first_attempt=first_attempt)
    usage, absent, batch_usages = Usage(), [], {}
    partial_keys = {batch.key for batch in condition.partial}
    returned = expected = 0
    for batch, refs in sorted(reading.requests.items()):
        known = condition.batch_usages.get(batch)
        if known is None and not (condition.batch_root / batch / USAGE_FNAME).is_file():
            absent.append(batch)
            continue
        billed = request_usage(condition.batch_root / batch, condition.condition.model_type, known)
        batch_usages[batch] = total(billed.values())
        for ref, item in refs.items():
            if batch in partial_keys:
                expected += 1
                returned += ref in billed
            if ref not in billed:  # a failed request, or one a partial batch never returned, bills nothing
                continue
            usage += billed[ref]
    return dataclasses.replace(
        condition, batches=tuple(sorted(reading.keys)), missing=tuple(absent), usage=usage, batch_usages=batch_usages,
        draws=reading.draws, scenarios=frozenset(reading.scenarios), bundles=reading.bundles,
        grid_requests=reading.grid_requests, shapes=frozenset(reading.shapes), scope=scope, kind=kind,
        partial=tuple(batch for batch in condition.partial if batch.key in reading.requests),
        partial_requests=(returned, expected), first_attempt=first_attempt)


def restrict(
    condition: ConditionUsage, scenarios: Optional[Iterable[str]], cache: Optional[dict] = None, *,
    kind: Optional[str] = None, first_attempt: bool = False,
) -> ConditionUsage:
    """Return ``condition`` over ``scenarios``, components of ``kind`` and optionally first-attempt rows, always within
    its own."""
    scope, kind = _narrowed(condition, scenarios, kind)
    first_attempt = first_attempt or condition.first_attempt
    if scope == condition.scope and kind == condition.kind and first_attempt == condition.first_attempt:
        return condition
    key = (condition.label, scope, kind, first_attempt)
    if cache is not None and key in cache:
        return cache[key]
    restricted = _reading(condition, scope, kind, first_attempt)
    if cache is not None:
        cache[key] = restricted
    return restricted


def attribute(
    roots: Sequence[Path | str], batch_root: Path | str, *, scenarios: Optional[Iterable[str]] = None,
    component_kind: Optional[str] = None,
) -> Attribution:
    """Attribute every batch the roots' ledgers name to its campaign and condition; ``scenarios`` or ``component_kind``
    narrow each."""
    _check_kind(component_kind, where="attribute")
    batch_root = Path(batch_root)
    if not batch_root.is_dir():
        raise BatchCostError(f"batch root {batch_root} is not a directory")
    campaigns: dict[Path, CampaignRoot] = {}
    owner: dict[str, str] = {}
    readings: list[tuple[CampaignRoot, ConditionSpec, str, _LedgerReading]] = []
    for root in map(Path, roots):
        resolved = root.resolve()
        if resolved in campaigns:
            continue
        campaign = CampaignRoot.open(root)
        if any(other.campaign_id == campaign.campaign_id for other in campaigns.values()):
            raise BatchCostError(f"{root}: another given root is campaign {campaign.campaign_id!r} too; give one of them")
        campaigns[resolved] = campaign
        for condition_id, condition in campaign.conditions.items():
            reading = _read_ledgers(campaign, condition_id)
            label = f"{campaign.campaign_id}/{condition_id}"
            for key in reading.keys:
                if owner.setdefault(key, label) != label:
                    raise BatchCostError(f"batch {key} is named by {owner[key]} and by {label}")
            readings.append((campaign, condition, label, reading))
    directories = sorted(directory for directory in batch_root.iterdir() if directory.is_dir())
    # One read of each usage.json: everything is checked against and priced from the same bytes.
    read_usage = functools.cache(Usage.read)
    markers = {directory.name: marker for directory in directories
               if (marker := read_partial_marker(directory, read_usage)) is not None}
    partial = _attribute_partial(readings, markers, owner, batch_root)
    conditions: list[ConditionUsage] = []
    missing: dict[str, str] = {}
    for campaign, condition, label, reading in readings:
        found = partial.get(label, ())
        if any(batch.rows for batch in found):
            reading = _read_ledgers(campaign, condition.condition_id, partial=found)
        batch_usages, absent = {}, []
        for key in sorted(reading.keys):
            path = batch_root / key / USAGE_FNAME
            if path.is_file():
                batch_usages[key] = read_usage(path)
            else:
                absent.append(key)
                missing[key] = label
        conditions.append(ConditionUsage(
            campaign=campaign, condition=condition, batch_root=batch_root, batches=tuple(sorted(reading.keys)),
            missing=tuple(absent), usage=total(batch_usages.values()), batch_usages=batch_usages,
            draws=reading.draws, scenarios=frozenset(reading.scenarios),
            bundles=reading.bundles, grid_requests=reading.grid_requests, shapes=frozenset(reading.shapes),
            partial=found, partial_requests=(sum(batch.returned for batch in found),
                                             sum(batch.expected for batch in found))))
    if scenarios is not None or component_kind is not None:
        wanted = None if scenarios is None else frozenset(scenarios)
        conditions = [restrict(condition, wanted, kind=component_kind) for condition in conditions]
    unattributed: dict[str, Usage] = {}
    unread: list[str] = []
    for directory in directories:
        if directory.name in owner:
            continue
        try:
            unattributed[directory.name] = read_usage(directory / USAGE_FNAME)
        except BatchCostError:
            unread.append(directory.name)
    return Attribution(batch_root=batch_root, roots=tuple((c.campaign_id, path) for path, c in campaigns.items()),
                       conditions=tuple(conditions), unattributed=unattributed, unread=tuple(unread),
                       missing=missing, unattributed_partial=tuple(sorted(set(markers) - set(owner))))


def _attribute_partial(
    readings: Sequence[tuple[CampaignRoot, ConditionSpec, str, _LedgerReading]], markers: Mapping[str, dict],
    owner: dict[str, str], batch_root: Path,
) -> dict[str, tuple[PartialBatch, ...]]:
    found: dict[str, list[PartialBatch]] = {}
    by_prefix: dict[str, list[str]] = {}
    for key, marker in sorted(markers.items()):
        if key in owner:
            found.setdefault(owner[key], []).append(_partial_batch(key, marker))
        elif marker.get("custom_id_prefix"):
            by_prefix.setdefault(marker["custom_id_prefix"], []).append(key)
    if by_prefix:
        for campaign, condition, label, reading in readings:
            for draw, block in _unrecorded_blocks(campaign, condition.condition_id, reading.blocks):
                namespace = block_namespace(campaign.campaign_id, condition, str(block["request_block_id"]))
                keys = by_prefix.get(namespace, [])
                if len(keys) > 1:
                    raise BatchCostError(f"partial batches {keys} both carry request block {block['request_block_id']}"
                                         f"'s namespace; one block buys one batch")
                if not keys:
                    continue
                key = keys[0]
                if key in owner:  # a block id names its condition and draw, and is recorded once
                    raise BatchCostError(f"batch {key} is matched by two request blocks ({owner[key]}, and "
                                         f"{block['request_block_id']} of {label}); a block's record is written once")
                owner[key] = label
                rows = _block_rows(campaign, condition, draw, block, batch_root / key)
                found.setdefault(label, []).append(_partial_batch(
                    key, markers[key], draw=draw, block_id=str(block["request_block_id"]), rows=rows))
    return {label: tuple(sorted(batches, key=lambda batch: batch.key)) for label, batches in found.items()}


def group_conditions(conditions: Sequence[ConditionUsage], by: str) -> list[Measured]:
    """Sum the conditions by ``condition``, ``campaign`` or a condition-identity field, in order of first appearance."""
    if by not in ("condition", "campaign") and by not in _IDENTITY_KEYS:
        raise BatchCostError(f"cannot group by {by!r}: condition, campaign or one of {sorted(_IDENTITY_KEYS)}")
    groups: dict[str, list[ConditionUsage]] = {}
    for condition in conditions:
        if by == "condition":
            label = condition.label
        elif by == "campaign":
            label = condition.campaign_id
        else:
            label = f"{condition.campaign_id}:{by}={condition.value(by)}"
        groups.setdefault(label, []).append(condition)
    return [combine(label, members) for label, members in groups.items()]


def _costs(usage: Usage, rates: Sequence[Rates]) -> dict[str, dict[str, float]]:
    return {rate.model: price(usage, rate) for rate in rates}


def _price_cells(usage: Usage, rates: Sequence[Rates]) -> list[str]:
    return [f"{cost['total']:,.2f}" for cost in _costs(usage, rates).values()]


def _rates_record(path: Path, rates: Sequence[Rates]) -> dict[str, Any]:
    return {"path": artifact_uri_or_path(path), "sha256": sha256_file(path),
            "priced": {rate.model: dataclasses.asdict(rate) for rate in rates}}


def _attribution_record(attribution: Attribution, rates: Sequence[Rates]) -> dict[str, Any]:
    unattributed = attribution.unattributed_usage
    return {
        "batch_root": artifact_uri_or_path(attribution.batch_root),
        "roots": [{"campaign_id": cid, "path": artifact_uri_or_path(path)} for cid, path in attribution.roots],
        "conditions": [{"campaign_id": c.campaign_id, "condition_id": c.condition_id,
                        "condition": c.condition.to_dict(),
                        "batches": list(c.batches), "missing": list(c.missing),
                        "usage": c.usage.to_dict(), "draws": c.draws, "scenarios": len(c.scenarios),
                        "bundles": c.bundles, "grid_requests": c.grid_requests, "shapes": _shapes_record(c.shapes),
                        "partial": [batch.to_dict() for batch in c.partial]}
                       for c in attribution.conditions],
        "unattributed": {"batches": sorted(attribution.unattributed), "usage": unattributed.to_dict(),
                         "cost": _costs(unattributed, rates), "partial": list(attribution.unattributed_partial)},
        "unread": list(attribution.unread),
        "missing": dict(attribution.missing),
    }


def _millions(value: float) -> str:
    return f"{value / 1e6:,.2f}"


_COLUMNS = ("requests", "input M", "cached M", "cache-write M", "output M", "reasoning M")


def _usage_cells(usage: Usage) -> list[str]:
    return [f"{usage.requests:,.0f}", *(_millions(getattr(usage, name)) for name in TOKEN_FIELDS)]


def _overhead(measured: Measured) -> str:
    overhead = measured.retry_overhead
    return "-" if overhead is None else f"{overhead:+.2%}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    return ["  ".join(str(cell).rjust(width) if index else str(cell).ljust(width)
                      for index, (cell, width) in enumerate(zip(row, widths))) for row in (headers, *rows)]


def usage_report(
    attribution: Attribution, rates: Sequence[Rates], *, group_by: str
) -> tuple[dict[str, Any], list[str]]:
    """Return the usage command's record and text."""
    groups = group_conditions(attribution.conditions, group_by)
    partial = any(group.partial for group in groups)
    headers = ["group", "batches", *_COLUMNS, "retry", *(["partial"] if partial else []),
               *(f"$ {rate.model}" for rate in rates)]
    rows = [[group.label, str(group.batches), *_usage_cells(group.usage), _overhead(group),
             *([group.partial_text()] if partial else []), *_price_cells(group.usage, rates)] for group in groups]
    unattributed = attribution.unattributed_usage
    rows.append([f"(unattributed: {len(attribution.unattributed)} batch(es))", str(len(attribution.unattributed)),
                 *_usage_cells(unattributed), "-", *(["-"] if partial else []), *_price_cells(unattributed, rates)])
    lines = [f"batch root {attribution.batch_root}: {len(attribution.conditions)} condition(s) of "
             f"{len(attribution.roots)} campaign(s)", *_table(headers, rows)]
    if partial:
        lines.append("partial: requests returned / sent by the batches that ended without completing (collected); "
                     "the bill is what returned, and the draw stays incomplete")
    if attribution.unattributed_partial:
        lines.append(f"{len(attribution.unattributed_partial)} unattributed batch(es) ended without completing "
                     f"(partial.json): {', '.join(attribution.unattributed_partial)}")
    if attribution.unread:
        lines.append(f"{len(attribution.unread)} unattributed batch(es) hold no readable usage.json: "
                     f"{', '.join(attribution.unread)}")
    record = {"schema_version": BATCH_COST_SCHEMA_VERSION, "kind": "usage", "group_by": group_by,
              "groups": [{**group.to_dict(), "cost": _costs(group.usage, rates)} for group in groups],
              **_attribution_record(attribution, rates)}
    return record, lines


def chosen_rates(path: Path | str, names: Optional[Sequence[str]], default: Sequence[str] = ()) -> list[Rates]:
    """The rate file's entries for ``default``'s models it prices, then each of ``names`` (refused if unpriced)."""
    rates = load_rates(path)
    unknown = [name for name in names or () if name not in rates]
    if unknown:
        raise BatchCostError(f"{path} has no rates for {unknown}; it has {sorted(rates)}")
    chosen = dict.fromkeys([*(name for name in default if name in rates), *(names or ())])
    return [rates[name] for name in chosen]


def selected_conditions(conditions: Sequence[ConditionUsage], wanted: Optional[Sequence[str]]) -> list[ConditionUsage]:
    """The ``wanted`` ids of ``conditions`` (all when none is named), each refused unless a root holds it."""
    if not wanted:
        return list(conditions)
    missing = sorted(set(wanted) - {condition.condition.condition_id for condition in conditions})
    if missing:
        raise BatchCostError(f"--condition {missing}: no --root campaign holds it")
    return [condition for condition in conditions if condition.condition.condition_id in wanted]


def _finished(draw_dir: Path) -> bool:
    try:
        return json.loads((draw_dir / COMPLETION_FNAME).read_text(encoding="utf-8")).get("state") == STATE_COMPLETE
    except (OSError, ValueError, AttributeError):
        return False


def refuse_unledgered(conditions: Sequence[ConditionUsage]) -> None:
    """Refuse conditions with a complete draw that holds no attempt ledger, which its usage is read from."""
    missing = [f"{condition.label} {draw_dirname(draw)}" for condition in conditions
               for draw in range(condition.campaign.n_draws)
               if _finished(condition.campaign.root / condition.condition_id / draw_dirname(draw))
               and not (condition.campaign.root / condition.condition_id / draw_dirname(draw) / LEDGER_FNAME).is_file()]
    if missing:
        raise BatchCostError(f"{len(missing)} complete draw(s) hold no {LEDGER_FNAME}, which their usage is read from: "
                             f"{', '.join(missing[:6])}{', ...' if len(missing) > 6 else ''}; read their cost from a "
                             "root that holds their ledgers, or name the other conditions with --condition")


def campaign_usage(
    roots: Sequence[Path | str], batch_root: Path | str, *, rates: Path | str = DEFAULT_RATES,
    price: Sequence[str] = (), scenarios: Optional[Path | str] = None, component_kind: Optional[str] = None,
    conditions: Sequence[str] = (), group_by: str = "condition",
) -> tuple[dict[str, Any], list[str], Attribution]:
    """Return what the batches of ``roots``' conditions used and cost: the record, its text and the attribution."""
    for root in roots:
        refuse_published(root, "campaign cost usage", "each draw's attempt ledger and request blocks in full")
    manifest = None if scenarios is None else ScenarioManifest.build(scenarios)
    attribution = attribute(roots, batch_root, scenarios=None if manifest is None else manifest.scenario_ids,
                            component_kind=component_kind)
    attribution = dataclasses.replace(attribution,
                                      conditions=tuple(selected_conditions(attribution.conditions, conditions)))
    refuse_unledgered(attribution.conditions)
    judges = sorted({condition.condition.model_name for condition in attribution.conditions})
    priced = chosen_rates(rates, price, judges)
    record, lines = usage_report(attribution, priced, group_by=group_by)
    if manifest is not None:
        lines.insert(0, f"over the {manifest.record_count} scenarios of {scenarios} only (the unattributed row is "
                        "whole batches: no given root names their scenarios)")
        record["scenarios"] = {"path": artifact_uri_or_path(scenarios), "sha256": manifest.source_sha256,
                               "count": manifest.record_count}
    if component_kind is not None:
        lines.insert(0, f"{component_kind} requests only (the unattributed row is whole batches)")
        record["component_kind"] = component_kind
    if conditions:
        lines.insert(0, f"only condition(s) {sorted(conditions)} of the --root campaigns (the others' batches are in "
                        "no row, the unattributed one included)")
        record["selected_conditions"] = sorted(conditions)
    record["rates"] = _rates_record(Path(rates), priced)
    return record, lines, attribution
