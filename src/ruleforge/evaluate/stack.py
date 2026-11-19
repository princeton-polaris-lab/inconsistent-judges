from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ruleforge.evaluate.identity import (
    DRAW_PROVENANCE_FNAME,
    RUNS_FNAME,
    run_log_problem,
)
from ruleforge.scenarios.manifest import SAMPLING_SCHEMA_VERSION
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import (
    canonical_digest,
    canonical_json,
)
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.ledger import JUDGED_ON_FIELD, LEDGER_FNAME, AttemptRecord, named_stacks
from ruleforge.evaluate.ledger import has_attempts as draw_has_attempts
from ruleforge.judges.backend import PRECISION_ENV_DEFAULTS, precision_setting
from ruleforge.core.resume import count_complete_jsonl_records

STACK_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "triton")
PRECISION_ENV_KEY = "precision_env"
ENGINE_ARGS_KEY = "engine_args"
# Engine arguments that say where and how wide an engine runs, not what it computes, so neither guard compares them.
ENGINE_PLACEMENT_ARGS = frozenset({
    "model", "tensor_parallel_size", "disable_log_stats", "max_num_seqs",
    "gpu_memory_utilization", "max_num_batched_tokens",
})
# The binding engine arguments a contrast holds equal besides the stack.
CONTRAST_ENGINE_ARGS = ("dtype", "attention_config", "max_model_len")
# Compared between one condition's own draws only: a contrast may pair differently configured engines on purpose.
CONDITION_ONLY_FIELDS = frozenset({ENGINE_ARGS_KEY})
LOCAL_MODEL_TYPE = "local"
CONDITION_STACK_FNAME = "condition_stack.json"
# ``runs.jsonl`` key of a row logging that a run added attempts on another stack, as allowed.
STACK_CHANGE_KEY = "stack_change"
# ``runs.jsonl`` key: the ledger rows held when the run started, which binds later rows to the run that appended them.
LEDGER_ROWS_KEY = "ledger_rows"
# ``attempts.jsonl`` field: the digest of the stack a row was judged on, so a reader takes it from the row itself.
JUDGED_ON_KEY = JUDGED_ON_FIELD

# ``{package or precision variable: version or setting}``; None means not recorded.
Signature = dict[str, Optional[str]]
# ``{engine-argument digest: the binding engine arguments it digests}``.
EngineArgsIndex = dict[str, dict[str, Any]]


def recorded_precision(runtime: Mapping[str, Any]) -> dict[str, Optional[str]]:
    """Return the precision settings a runtime record names; a value a local judge would refuse is kept as recorded."""
    recorded = runtime.get(PRECISION_ENV_KEY)
    recorded = recorded if isinstance(recorded, Mapping) else {}
    return {
        name: precision_setting(name, recorded[name], strict=False) if name in recorded else None
        for name in PRECISION_ENV_DEFAULTS
    }


def stack_signature(runtime: Mapping[str, Any]) -> Signature:
    """Return ``{package: version, precision variable: setting}`` of a runtime record."""
    return {
        **{name: runtime.get(f"{name}_version") for name in STACK_PACKAGES},
        **recorded_precision(runtime),
    }


def binding_engine_args(runtime: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return a runtime record's engine arguments, placement aside; an argument recorded as None is left out."""
    recorded = runtime.get(ENGINE_ARGS_KEY)
    if not isinstance(recorded, Mapping):
        return None
    return {
        name: value for name, value in recorded.items() if name not in ENGINE_PLACEMENT_ARGS and value is not None
    }


def judgment_signature(runtime: Mapping[str, Any]) -> Signature:
    """Return what a draw's judgments came from: the stack signature, the contrast arguments and a digest of all binding
    arguments."""
    binding = binding_engine_args(runtime)
    contrast = {
        f"{ENGINE_ARGS_KEY}.{name}": None if binding is None else canonical_json(binding.get(name))
        for name in CONTRAST_ENGINE_ARGS
    }
    digest = None if binding is None else canonical_digest(binding)[:16]
    return {**stack_signature(runtime), **contrast, ENGINE_ARGS_KEY: digest}


def stack_digest(runtime: Mapping[str, Any]) -> str:
    """Return the digest of a runtime record's ``judgment_signature``."""
    return canonical_digest(judgment_signature(runtime))


def _differing(first: Mapping[str, Optional[str]], second: Mapping[str, Optional[str]]) -> list[str]:
    return [
        name
        for name in first
        if first.get(name) is not None and second.get(name) is not None and first[name] != second[name]
    ]


def stack_changes(first: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return ``{name: (first value, current value)}`` for stack fields that differ between two runtime records."""
    before, now = stack_signature(first), stack_signature(current)
    return {name: (before[name], now[name]) for name in _differing(before, now)}


def _binding_changes(before: Mapping[str, Any], now: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {
        f"{ENGINE_ARGS_KEY}.{name}": (before.get(name), now.get(name))
        for name in sorted(set(before) | set(now))
        if canonical_json(before.get(name)) != canonical_json(now.get(name))
    }


def engine_args_changes(first: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Return ``{"engine_args.<name>": (first value, current value)}`` for binding arguments that differ."""
    before, now = binding_engine_args(first), binding_engine_args(current)
    if before is None or now is None:
        return {}
    return _binding_changes(before, now)


def runtime_changes(first: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """What a runner may not change within a draw or a condition: the stack and the binding engine arguments."""
    return {**stack_changes(first, current), **engine_args_changes(first, current)}


def describe_changes(changed: Mapping[str, Sequence[Any]]) -> str:
    """``vllm 0.19.0 -> 0.28.0, torch ...``."""
    return ", ".join(f"{name} {before} -> {now}" for name, (before, now) in changed.items())


def signatures_conflict(first: Mapping[str, Optional[str]], second: Mapping[str, Optional[str]]) -> bool:
    """Whether two signatures name different values of one field."""
    return bool(_differing(first, second))


def across_conditions(signature: Mapping[str, Optional[str]]) -> Signature:
    """What a contrast compares of a draw's signature: the stack and :data:`CONTRAST_ENGINE_ARGS`."""
    return {name: value for name, value in signature.items() if name not in CONDITION_ONLY_FIELDS}


def describe_signature(signature: Mapping[str, Optional[str]]) -> str:
    """``vllm 0.28.0+cu129 / ... / TRITON_F32_DEFAULT=unset / engine_args=<digest>``: the fields it names."""
    shown = [
        f"{name} {value}" if name in STACK_PACKAGES else f"{name}={value}"
        for name, value in signature.items()
        if value is not None
    ]
    return " / ".join(shown) or "no stack recorded"


def describe_signatures(signatures: Sequence[Mapping[str, Optional[str]]]) -> str:
    """:func:`describe_signature` of each, joined by `` + ``."""
    return " + ".join(describe_signature(signature) for signature in signatures)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def recorded_runtime(draw_dir: Path | str) -> dict[str, Any]:
    """The runtime a draw's provenance records (its judgments' first stack); {} without one."""
    path = Path(draw_dir) / DRAW_PROVENANCE_FNAME
    if not path.is_file():
        return {}
    return dict(_read_object(path).get("runtime") or {})


def _complete_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for _, line in zip(range(count_complete_jsonl_records(path)), handle)]


def unlogged_stacks(draw_dir: Path | str, records: Optional[Iterable[AttemptRecord]] = None) -> Optional[str]:
    """Return what is wrong with the stacks a draw's ledger rows were judged on, or None; reads only."""
    draw_dir = Path(draw_dir)
    named = (named_stacks(draw_dir) if records is None
             else Counter(record.judged_on for record in records if record.judged_on is not None))
    if not named:
        return None
    path = draw_dir / DRAW_PROVENANCE_FNAME
    provenance = _read_object(path) if path.is_file() else {}
    recorded = dict(provenance.get("runtime") or {})
    rows = _complete_rows(draw_dir / RUNS_FNAME)
    changes = [dict(row.get("runtime") or {}) for row in rows if row.get(STACK_CHANGE_KEY)]
    runs = {stack_digest(row.get("runtime") or {}): dict(row.get("runtime") or {})
            for row in rows if STACK_CHANGE_KEY not in row}
    held = {stack_digest(recorded), *map(stack_digest, changes)}
    local = provenance.get("model_type") == LOCAL_MODEL_TYPE
    problems = []
    for stack, count in sorted(named.items()):
        if stack in held:
            continue
        runtime = runs.get(stack)
        if runtime is None:
            problems.append(
                f"{count} row(s) of its {LEDGER_FNAME} were judged on a stack ({JUDGED_ON_KEY} {stack[:12]}) that "
                f"neither its {DRAW_PROVENANCE_FNAME} nor any complete row of its {RUNS_FNAME} records: the run that "
                "judged them is gone from the run log (cut from it, rolled back with the gate that counted it, or "
                "logged behind bytes no reader reads), and they would read as judged on the stacks left")
            continue
        changed = runtime_changes(recorded, runtime)
        if local and changed and all(runtime_changes(change, runtime) for change in changes):
            problems.append(
                f"{count} row(s) of its {LEDGER_FNAME} were judged on another stack than its provenance's "
                f"({describe_changes(changed)}), and its {RUNS_FNAME} logs no stack change to it: they would read as "
                "judged on the provenance's stack")
    return "; ".join(problems) or None


def judgment_runtimes(draw_dir: Path | str) -> list[dict[str, Any]]:
    """Return the runtimes a draw's recorded attempts came from; refused when the run log lost a run or a stack."""
    draw_dir = Path(draw_dir)
    if not draw_has_attempts(draw_dir):
        return []
    cut = run_log_problem(draw_dir) or unlogged_stacks(draw_dir)
    if cut:
        raise SamplingDesignError(f"{draw_dir}: {cut}; its judgments' stacks cannot be read from it")
    changes = [dict(row.get("runtime") or {}) for row in _complete_rows(draw_dir / RUNS_FNAME)
               if row.get(STACK_CHANGE_KEY)]
    return [recorded_runtime(draw_dir), *changes]


@dataclass(frozen=True)
class RunSpan:
    """One run a draw's run log records: its own row, the ledger rows it appended, any stack change it logged."""

    row: Mapping[str, Any]
    # None when the run log cannot say.
    appended: Optional[int]
    stack_change: Optional[Mapping[str, Any]]


def run_spans(
    rows: Sequence[Mapping[str, Any]], *, start: int, end: int
) -> tuple[list[RunSpan], Optional[str]]:
    """Return the runs ``rows`` log, each with the ledger rows it appended, and what is wrong with the log (None if
    nothing); ``start`` and ``end`` are the ledger's row counts when ``rows`` began and now."""
    spans: list[dict[str, Any]] = []
    problem = None
    for row in rows:
        if STACK_CHANGE_KEY not in row:
            spans.append({"row": row, "stack_change": None})
        elif spans:
            spans[-1]["stack_change"] = spans[-1]["stack_change"] or row[STACK_CHANGE_KEY]
        else:
            problem = "a stack change logged before any run"
    bounds = [span["row"].get(LEDGER_ROWS_KEY) for span in spans]
    known = problem is None and all(type(bound) is int for bound in bounds)
    edges = [*bounds, end]
    if known and spans and (edges[0] != start or any(later < earlier for earlier, later in zip(edges, edges[1:]))):
        problem = (f"its runs' ledger boundaries {bounds} are not its ledger's (rows {start} to {end}): the rows "
                   "cannot be told to the runs that appended them")
        known = False
    return [
        RunSpan(span["row"], edges[index + 1] - edges[index] if known else None, span["stack_change"])
        for index, span in enumerate(spans)
    ], problem


def draw_dirs(condition_dir: Path | str) -> list[Path]:
    """A condition's draw directories, in draw order."""
    return sorted(path for path in Path(condition_dir).glob("draw-*") if path.is_dir())


def condition_signatures(
    condition_dir: Path | str, *, args_by_digest: Optional[EngineArgsIndex] = None
) -> list[Signature]:
    """Return the distinct judgment signatures of a condition's recorded attempts, in first-seen order."""
    seen: list[Signature] = []
    for draw_dir in draw_dirs(condition_dir):
        for runtime in judgment_runtimes(draw_dir):
            signature = judgment_signature(runtime)
            if args_by_digest is not None and signature[ENGINE_ARGS_KEY] is not None:
                args_by_digest.setdefault(signature[ENGINE_ARGS_KEY], binding_engine_args(runtime))
            if signature not in seen:
                seen.append(signature)
    return seen


def signatures_mixed(signatures: Sequence[Mapping[str, Optional[str]]]) -> bool:
    """Whether any two of one condition's signatures conflict (its stacks or binding engine arguments)."""
    return any(
        signatures_conflict(a, b) for i, a in enumerate(signatures) for b in signatures[i + 1 :]
    )


def contrast_conflict(
    first: Sequence[Mapping[str, Optional[str]]], second: Sequence[Mapping[str, Optional[str]]]
) -> Optional[str]:
    """Return why two conditions' judgments may not be contrasted, or None; a condition without stacks conflicts with
    nothing."""
    for label, signatures in (("the first", first), ("the second", second)):
        if signatures_mixed(signatures):
            return (
                f"{label} member's judgments come from several stacks or engine configurations "
                f"({describe_signatures(signatures)})"
            )
    stacks_first = [across_conditions(signature) for signature in first]
    stacks_second = [across_conditions(signature) for signature in second]
    if any(signatures_conflict(a, b) for a in stacks_first for b in stacks_second):
        return f"{describe_signatures(stacks_first)} against {describe_signatures(stacks_second)}"
    return None


# Top-level keys of an analysis made under ``allow_stack_mismatch``. An output without them reports no mismatch.
ALLOW_STACK_MISMATCH_KEY = "allow_stack_mismatch"
STACK_MISMATCHES_KEY = "stack_mismatches"
INFERENCE_STACKS_KEY = "inference_stacks"
CONDITION_SCOPE = "condition"


def mismatch_record(scope: str, conditions: Sequence[str], reason: str) -> dict[str, Any]:
    """One :data:`STACK_MISMATCHES_KEY` entry."""
    return {"scope": scope, "conditions": list(conditions), "reason": reason}


def override_record(allow_stack_mismatch: bool, mismatches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the top-level stack keys of an analysis output, only under the override."""
    if not allow_stack_mismatch:
        return {}
    return {ALLOW_STACK_MISMATCH_KEY: True, STACK_MISMATCHES_KEY: [dict(record) for record in mismatches]}


def stacks_record(allow_stack_mismatch: bool, signatures: Sequence[Mapping[str, Optional[str]]]) -> dict[str, Any]:
    """Return a condition's or model's ``INFERENCE_STACKS_KEY`` entry, only under the override."""
    return {INFERENCE_STACKS_KEY: [dict(signature) for signature in signatures]} if allow_stack_mismatch else {}


def describe_engine_args(signatures: Sequence[Mapping[str, Optional[str]]], *, args_by_digest: EngineArgsIndex) -> str:
    """Return a description like ``engine_args.max_model_len 4 -> 4.0`` of the arguments behind differing digests."""
    digests = dict.fromkeys(signature.get(ENGINE_ARGS_KEY) for signature in signatures)
    held = [digest for digest in digests if digest in args_by_digest]
    if len(held) < 2:
        return ""
    first = args_by_digest[held[0]]
    return "; ".join(describe_changes(_binding_changes(first, args_by_digest[digest])) for digest in held[1:])


def mixed_conditions(
    stacks: Mapping[str, Sequence[Mapping[str, Optional[str]]]],
    *,
    args_by_digest: Optional[EngineArgsIndex] = None,
) -> list[dict[str, Any]]:
    """Return a record for every condition whose own judgments come from several stacks or engine configurations."""
    records = []
    for condition_id, signatures in stacks.items():
        if not signatures_mixed(signatures):
            continue
        detail = describe_engine_args(signatures, args_by_digest=args_by_digest or {})
        records.append(mismatch_record(
            CONDITION_SCOPE,
            [condition_id],
            "its judgments come from several stacks or engine configurations "
            f"({describe_signatures(signatures)}{'; ' + detail if detail else ''})",
        ))
    return records


def shared_stack_conflicts(
    members: Mapping[str, Sequence[Mapping[str, Optional[str]]]],
    *,
    scope: str,
    args_by_digest: Optional[EngineArgsIndex] = None,
) -> list[dict[str, Any]]:
    """Return why ``members`` (condition id -> stacks) do not share one stack, at most one record per member; empty when
    they do."""
    records = mixed_conditions(members, args_by_digest=args_by_digest)
    consistent: dict[str, Sequence[Mapping[str, Optional[str]]]] = {}
    for condition_id, signatures in members.items():
        if signatures_mixed(signatures):
            continue
        for earlier, theirs in consistent.items():
            reason = contrast_conflict(theirs, signatures)
            if reason is not None:
                records.append(mismatch_record(scope, [earlier, condition_id], reason))
                break
        else:
            consistent[condition_id] = signatures
    return records


def describe_mismatches(records: Sequence[Mapping[str, Any]], *, limit: int = 3) -> str:
    """``a vs b: reason; ...`` for the first ``limit`` records, and how many more."""
    shown = "; ".join(f"{' vs '.join(record['conditions'])}: {record['reason']}" for record in records[:limit])
    return shown + (f"; and {len(records) - limit} more" if len(records) > limit else "")


def recorded_mismatches(estimates: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The stack mismatches an analysis output reports; empty for one without the key."""
    return [dict(record) for record in estimates.get(STACK_MISMATCHES_KEY) or []]


def recorded_mismatch_refusal(estimates: Mapping[str, Any], where: str) -> Optional[str]:
    """Return why a producer refuses an analysis output that reports stack mismatches, or None."""
    records = recorded_mismatches(estimates)
    if not records:
        return None
    return (
        f"{where} reports {len(records)} inference-stack mismatch(es), analysed under "
        f"allow_stack_mismatch ({describe_mismatches(records)}); pass --allow-stack-mismatch to build "
        "from it anyway, on the record"
    )


def read_condition_stack(condition_dir: Path | str) -> Optional[dict[str, Any]]:
    """The condition's :data:`CONDITION_STACK_FNAME` record, or None."""
    path = Path(condition_dir) / CONDITION_STACK_FNAME
    return _read_object(path) if path.is_file() else None


def condition_stack(condition_dir: Path | str) -> Optional[dict[str, Any]]:
    """Return the stack a condition's judgments started on, or None while it holds none."""
    with_attempts = [path for path in draw_dirs(condition_dir) if draw_has_attempts(path)]
    if not with_attempts:
        return None
    record = read_condition_stack(condition_dir)
    if record is not None:
        return record
    first = with_attempts[0]
    runtime = recorded_runtime(first)
    return {
        "stack": stack_signature(runtime),
        "runtime": runtime,
        "source": f"{first.name}/{DRAW_PROVENANCE_FNAME}",
    }


def write_condition_stack(
    condition_dir: Path | str,
    *,
    runtime: Mapping[str, Any],
    condition_id: str,
    condition_fingerprint: str,
    source: str,
    recorded_at: str,
) -> Path:
    """Write the condition's record: the stack, the full runtime (engine arguments included), where from."""
    return write_json_sidecar(Path(condition_dir) / CONDITION_STACK_FNAME, {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "condition_id": condition_id,
        "condition_fingerprint": condition_fingerprint,
        "stack": stack_signature(runtime),
        "runtime": dict(runtime),
        "source": source,
        "recorded_at": recorded_at,
    })
