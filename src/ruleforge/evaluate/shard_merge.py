from __future__ import annotations

import itertools
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Iterator, Mapping, NamedTuple, Optional, Sequence

from ruleforge.rulesets.variants import IMPLICATED_RULES_FNAME
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    campaign_completion,
    check_condition_id,
    check_judge_configuration,
    condition_of,
    refuse_retired_root,
    registers,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    FULL_RULESET_COMPONENT,
    RUN_LOG_ROWS_KEY,
    RUNS_FNAME,
    ConditionSpec,
    differing_manifest_fields,
    draw_dirname,
    load_sampling_manifest,
    request_block_id,
    sampling_config_from_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.amendment import CapAmendment, cap_amendment, check_resumed_draw
from ruleforge.evaluate.configured import MARKER_FNAME as CONFIGURED_MARKER_FNAME
from ruleforge.evaluate.configured import check_marker
from ruleforge.evaluate.ledger import AttemptRecord, DrawLedger, LedgerState, raw_output_sha256, raw_row_key
from ruleforge.core.provenance import SNAPSHOT_DIRNAME
from ruleforge.evaluate.identity import BUNDLE_OUTCOMES_FNAME
from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.evaluate.runner import (
    EVALUATIONS_FNAME,
    PROVENANCE_RUN_FIELDS,
    REQUEST_BLOCKS_FNAME,
    STATE_COMPLETE,
    DrawSpec,
    draw_completion,
    draw_ended,
    utc_now,
    write_draw_summary,
)
from ruleforge.evaluate.sharding import (
    SHARD_RECORD_FNAME,
    ShardAssignment,
    ShardError,
    blocks_per_draw,
    campaign_lock,
    describe_code_state,
    layout_code_states,
    parse_shard_dirname,
    shards_root,
)
from ruleforge.evaluate.stack import (
    CONDITION_STACK_FNAME,
    condition_stack,
    describe_changes,
    draw_dirs,
    judgment_runtimes,
    read_condition_stack,
    runtime_changes,
)
from ruleforge.core.resume import count_complete_jsonl_records

logger = logging.getLogger(__name__)

# The scratch directory draws are assembled in, beside the campaign root.
MERGING_SUFFIX = ".merging-"
_ROW_VIEWS = (BUNDLE_OUTCOMES_FNAME, EVALUATIONS_FNAME, JUDGE_OUTPUTS_FNAME)


@dataclass(frozen=True)
class Shard:
    """One shard root of a condition, and the units its ``shard.json`` records."""

    root: Path
    assignment: ShardAssignment
    record: Mapping[str, Any]

    @property
    def label(self) -> str:
        return self.assignment.label

    def draw_dir(self, condition_id: str, draw: int) -> Path:
        return self.root / condition_id / draw_dirname(draw)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_lines(path: Path) -> list[str]:
    count = count_complete_jsonl_records(path)
    if not count:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line for _, line in zip(range(count), handle)]


def discover_shards(shards_dir: Path | str, *, num_shards: Optional[int] = None) -> list[Shard]:
    """Return every shard root of one condition's layout, in shard order; ``num_shards`` picks a layout when there are
    several."""
    directory = Path(shards_dir)
    layouts: dict[int, dict[int, Path]] = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            parsed = parse_shard_dirname(path.name)
            if parsed is not None and path.is_dir():
                layouts.setdefault(parsed[1], {})[parsed[0]] = path
    if not layouts:
        raise ShardError(f"{directory} holds no shard roots (shard-<i>-of-<n>)")
    if num_shards is None:
        if len(layouts) > 1:
            raise ShardError(
                f"{directory} holds shards of {len(layouts)} layouts ({', '.join(f'of {n}' for n in sorted(layouts))}); "
                "name the one to merge (num_shards)"
            )
        num_shards = next(iter(layouts))
    found = layouts.get(num_shards, {})
    missing = [index for index in range(num_shards) if index not in found]
    if missing:
        raise ShardError(
            f"{directory} lacks shard(s) {missing} of {num_shards}: their units would be missing from the merge; "
            "pull the launch that ran them (ruleforge jobs pull <its id>, once its runs have ended), or launch those "
            "no launch has (ruleforge jobs launch campaign ... narrowed to this condition, --shards "
            f"{num_shards} --only {','.join(map(str, missing))})"
        )
    shards = []
    for index in range(num_shards):
        path = found[index] / SHARD_RECORD_FNAME
        if not path.is_file():
            raise ShardError(f"{found[index]} has no {SHARD_RECORD_FNAME}: its job never planned; launch it")
        record = _read_json(path)
        assignment = ShardAssignment.from_record(record)
        if (assignment.index, assignment.count) != (index, num_shards):
            raise ShardError(f"{path} records shard {assignment.label}, not its directory's")
        shards.append(Shard(found[index], assignment, record))
    return shards


def check_units(shards: Sequence[Shard]) -> None:
    """Refuse shards of different plans, and units run twice or by no shard."""
    first = shards[0].assignment
    for shard in shards[1:]:
        plan = (shard.assignment.draw_ids, shard.assignment.blocks_per_draw)
        if plan != (first.draw_ids, first.blocks_per_draw):
            raise ShardError(
                f"{shard.label} was planned over draws {list(plan[0])} x {plan[1]} request block(s) and "
                f"{shards[0].label} over draws {list(first.draw_ids)} x {first.blocks_per_draw}: not one plan's shards"
            )
    counts = Counter(unit for shard in shards for unit in shard.assignment.units)
    expected = {(draw, block) for draw in first.draw_ids for block in range(first.blocks_per_draw)}
    overlapping = sorted(unit for unit, count in counts.items() if count > 1)
    missing = sorted(expected - set(counts))
    stray = sorted(set(counts) - expected)
    if overlapping or missing or stray:
        raise ShardError(
            f"the shards' units are not the plan's, each once: {len(overlapping)} run by several shards "
            f"{overlapping[:5]}, {len(missing)} by none {missing[:5]}, {len(stray)} outside the plan {stray[:5]}"
        )


@dataclass(frozen=True)
class _Context:
    campaign_id: str
    condition: ConditionSpec
    condition_payload: Mapping[str, Any]
    sampling_manifest: Mapping[str, Any]
    scenario_manifest: ScenarioManifest
    reference_runtime: Mapping[str, Any]

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return self.scenario_manifest.scenario_ids


def _shards_and_draws(pairs: Sequence[tuple[str, str]]) -> str:
    by_shard: dict[str, list[str]] = {}
    for label, draw in pairs:
        by_shard.setdefault(label, []).append(draw)
    return ", ".join(f"{label} ({', '.join(draws)})" for label, draws in by_shard.items())


def _check_code_states(shards: Sequence[Shard], condition_id: str, amendment: Optional[CapAmendment]) -> None:
    found = layout_code_states(((shard.label, shard.root) for shard in shards), condition_id)
    refusal = None if amendment is None else amendment.code_state_refusal(found, condition_id)
    if len(found) > 1 and (amendment is None or refusal is not None):
        described = "; ".join(
            f"{describe_code_state(state)}: {_shards_and_draws(where)}" for state, where in sorted(found.items(), key=str)
        )
        raise ShardError(
            f"the shards ran {len(found)} code states ({described}). An unsharded run records one per process, "
            "but a merged draw mixing them would hide which judged what, and a runs.jsonl row, once written, "
            "stays: move aside the roots of the shards on the state to drop and run those shards again from "
            "the other's commit (ruleforge jobs launch campaign ... narrowed to this condition, --shards <n> "
            "--only <i,j> "
            "--expect-commit <commit>)"
            + ("" if refusal is None else f". The campaign's cap amendment does not allow this mix: {refusal}")
        )


def _check_identity(shards: Sequence[Shard], condition_id: str) -> _Context:
    first = shards[0]
    manifest = load_sampling_manifest(first.root / SAMPLING_MANIFEST_FNAME)
    scenarios = ScenarioManifest.load(first.root / SCENARIO_MANIFEST_FNAME)
    config = sampling_config_from_manifest(manifest)
    plan = first.assignment
    expected_blocks = blocks_per_draw(scenarios.record_count, config.scenarios_per_request_block)
    if plan.blocks_per_draw != expected_blocks or not set(plan.draw_ids) <= set(range(config.num_decoder_draws)):
        raise ShardError(
            f"the shards were planned over draws {list(plan.draw_ids)} x {plan.blocks_per_draw} request block(s), "
            f"but their campaign has draws 0-{config.num_decoder_draws - 1} of {expected_blocks} block(s) each"
        )
    payload = _read_json(first.root / condition_id / CONDITION_FNAME)
    runtimes: list[tuple[str, Mapping[str, Any]]] = []
    for shard in shards:
        where = f"{shard.label} ({shard.root})"
        record = shard.record
        if record.get("condition_id") != condition_id or record.get("campaign_id") != manifest["campaign_id"]:
            raise ShardError(
                f"{where} ran condition {record.get('condition_id')!r} of campaign {record.get('campaign_id')!r}, "
                f"not {condition_id!r} of {manifest['campaign_id']!r}"
            )
        differing = differing_manifest_fields(manifest, load_sampling_manifest(shard.root / SAMPLING_MANIFEST_FNAME))
        if differing:
            raise ShardError(f"{where}: its campaign manifest differs from {first.label}'s ({', '.join(differing)})")
        if ScenarioManifest.load(shard.root / SCENARIO_MANIFEST_FNAME).manifest_sha256 != scenarios.manifest_sha256:
            raise ShardError(f"{where} froze another scenario manifest than {first.label}")
        if _read_json(shard.root / condition_id / CONDITION_FNAME) != payload:
            raise ShardError(
                f"{where} registers another condition identity than {first.label} (fingerprint, prompt, model, "
                "rule set or validation rule): not one condition's shards"
            )
        if record.get("condition_fingerprint") != payload["fingerprint"]:
            raise ShardError(f"{where}: its shard record names another condition fingerprint than its condition.json")
        stack = read_condition_stack(shard.root / condition_id)
        if stack is None:
            raise ShardError(f"{where} has no {CONDITION_STACK_FNAME}: its job never started a draw; launch it")
        runtimes.append((f"{where} {CONDITION_STACK_FNAME}", stack["runtime"]))
        for draw in shard.assignment.draws:
            draw_dir = shard.draw_dir(condition_id, draw)
            runtimes.extend((f"{where} {draw_dir.name}", runtime) for runtime in judgment_runtimes(draw_dir))
    _check_code_states(shards, condition_id, cap_amendment(manifest))
    reference = runtimes[0][1]
    for where, runtime in runtimes:
        changed = runtime_changes(reference, runtime)
        if changed:
            raise ShardError(
                f"{where} judged on another inference stack or engine configuration than {runtimes[0][0]} "
                f"({describe_changes(changed)}): a merged condition would pool two stacks' judgments"
            )
    context = _Context(
        campaign_id=str(manifest["campaign_id"]),
        condition=condition_of(payload, first.root / condition_id / CONDITION_FNAME),
        condition_payload=payload,
        sampling_manifest=manifest,
        scenario_manifest=scenarios,
        reference_runtime=reference,
    )
    for shard in shards:
        _check_marker(shard.root, context)
    return context


def _check_marker(place: Path, context: _Context) -> None:
    if os.path.lexists(place / CONFIGURED_MARKER_FNAME):
        check_marker(place, campaign_id=context.campaign_id,
                     scenario_manifest_sha256=context.scenario_manifest.manifest_sha256)


def _check_campaign_root(root: Path, context: _Context) -> None:
    manifest_path = root / SAMPLING_MANIFEST_FNAME
    if manifest_path.exists():
        differing = differing_manifest_fields(load_sampling_manifest(manifest_path), context.sampling_manifest)
        if differing:
            raise ShardError(f"{manifest_path} describes another campaign than the shards ({', '.join(differing)})")
    scenario_path = root / SCENARIO_MANIFEST_FNAME
    if scenario_path.exists() and (
        ScenarioManifest.load(scenario_path).manifest_sha256 != context.scenario_manifest.manifest_sha256
    ):
        raise ShardError(f"{scenario_path} froze another population than the shards")
    _check_marker(root, context)
    condition_dir = root / context.condition.condition_id
    condition_path = condition_dir / CONDITION_FNAME
    if condition_path.exists() and not registers(_read_json(condition_path), context.condition,
                                                 context.condition_payload.get("schema_version")):
        raise ShardError(f"{condition_path} registers {context.condition.condition_id!r} with another identity")
    judged = condition_stack(condition_dir)
    if judged is None:
        return
    held = [(f"{condition_dir.name}/{CONDITION_STACK_FNAME} ({judged.get('source')})", judged["runtime"])]
    held += [(str(draw_dir), runtime) for draw_dir in draw_dirs(condition_dir) for runtime in judgment_runtimes(draw_dir)]
    for where, runtime in held:
        changed = runtime_changes(runtime, context.reference_runtime)
        if changed:
            raise ShardError(
                f"{root}'s {where} holds judgments of the condition on another stack or engine "
                f"({describe_changes(changed)}) than the shards judged on"
            )


def _check_shard_draw(shard: Shard, context: _Context, draw: int, scope: Sequence[str]) -> tuple[Path, bool]:
    condition = context.condition
    draw_dir = shard.draw_dir(condition.condition_id, draw)
    path = draw_dir / COMPLETION_FNAME
    if not path.is_file():
        raise ShardError(f"{shard.label} has not run draw {draw} ({path} is missing): launch or resume it")
    gate = _read_json(path)
    expected = {**shard.assignment.draw_shard(draw).to_dict(), "scenarios": len(scope)}
    if gate.get("shard") != expected:
        raise ShardError(f"{path} is not {shard.label}'s gate for its request blocks {expected['request_blocks']}")
    state = gate.get("state")
    if not draw_ended(gate):
        raise ShardError(
            f"{shard.label} has not finished: draw {draw} is {state!r} with {gate.get('pending')} "
            "bundle(s) pending; resume it (relaunch the same shard, from the commit it ran) before merging"
        )
    identity = (gate.get("campaign_id"), gate.get("condition_fingerprint"), gate.get("scenario_manifest_sha256"))
    if identity != (context.campaign_id, condition.fingerprint, context.scenario_manifest.manifest_sha256):
        raise ShardError(f"{path} gates another campaign, condition or population than the shards'")
    return draw_dir, state == STATE_COMPLETE


class _Entry(NamedTuple):
    """One ledger row of a shard, keyed for the canonical order: attempt, transport round, manifest position."""

    attempt: int
    transport_round: int
    position: int
    shard_index: int
    record: AttemptRecord

    @property
    def raw_key(self) -> tuple[int, str, int, str]:
        """The shard's raw rows and block of this row: (shard, scenario, attempt, block id as the shard ran it)."""
        record = self.record
        return (self.shard_index, record.scenario_id, record.semantic_attempt_id, record.request_block_id)


@dataclass
class _DrawParts:

    entries: list[_Entry] = field(default_factory=list)
    block_rows: dict[tuple[int, str], Mapping[str, Any]] = field(default_factory=dict)
    raw: dict[tuple[Any, ...], list[dict[str, Any]]] = field(default_factory=dict)
    views: dict[str, dict[str, str]] = field(default_factory=lambda: {name: {} for name in _ROW_VIEWS})
    owner: dict[str, int] = field(default_factory=dict)
    # Each shard draw a cap amendment resumed: what its resume appended.
    amended: dict[str, dict[str, Any]] = field(default_factory=dict)
    implicated: list[Optional[bytes]] = field(default_factory=list)
    provenance: list[tuple[Shard, bytes]] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    complete: bool = True


def _check_view_row(name: str, row: Mapping[str, Any], bundle: AttemptRecord, judge_type: str) -> bool:
    judgments = {component.component_id: component.judgment for component in bundle.components}
    if name == JUDGE_OUTPUTS_FNAME:
        traces = [(component_id, trace) for components in row["judge_outputs"].values()
                  for component_id, by_judge in components.items() for trace in by_judge.values()]
        return sorted(component_id for component_id, _ in traces) == sorted(judgments) and all(
            trace.get("judgment") == judgments[component_id] for component_id, trace in traces
        )
    if row.get("semantic_attempt_id") != bundle.semantic_attempt_id:
        return False
    if name == EVALUATIONS_FNAME:
        expected = {component_id: {judge_type: label} for component_id, label in judgments.items()}
        return list(row["evaluations"].values()) == [expected]
    labels = {**(row.get("rule_judgments") or {}), FULL_RULESET_COMPONENT: row.get("full_set_judgment")}
    return all(judgments.get(component_id) == label for component_id, label in labels.items())


def _read_views(parts: _DrawParts, draw_dir: Path, state: LedgerState, scope: Sequence[str], judge_type: str) -> None:
    for name in _ROW_VIEWS:
        lines = _complete_lines(draw_dir / name)
        rows = {json.loads(line)["scenario_id"]: line for line in lines}
        if len(lines) != len(scope) or set(rows) != set(scope):
            raise ShardError(f"{draw_dir / name} does not hold its units' {len(scope)} scenarios, each once")
        for sid, line in rows.items():
            if not _check_view_row(name, json.loads(line), state.accepted_bundle(sid), judge_type):
                raise ShardError(
                    f"{draw_dir / name}: scenario {sid[:12]}'s row is not the bundle its ledger accepted"
                )
        parts.views[name].update(rows)


def _read_draw(shards: Sequence[Shard], context: _Context, draw: int) -> _DrawParts:
    condition = context.condition
    config = sampling_config_from_manifest(context.sampling_manifest)
    amendment = cap_amendment(context.sampling_manifest)
    positions = {sid: index for index, sid in enumerate(context.scenario_ids)}
    parts = _DrawParts()
    for shard in shards:
        scope = shard.assignment.draw_shard(draw).scenario_ids(context.scenario_ids, config.scenarios_per_request_block)
        draw_dir, complete = _check_shard_draw(shard, context, draw, scope)
        resumed = None if amendment is None else check_resumed_draw(
            draw_dir, amendment, campaign_id=context.campaign_id, condition_id=condition.condition_id,
            decoder_draw_id=draw, shard=shard.label, generation_seed_base=config.generation_seed_base,
        )
        if resumed is not None:
            parts.amended[shard.label] = resumed
        ledger = DrawLedger(
            draw_dir, campaign_id=context.campaign_id, condition_id=condition.condition_id, decoder_draw_id=draw
        )
        state = ledger.load()
        exhausted = {
            sid for sid in scope
            if not state.is_accepted(sid) and state.next_attempt(sid) >= config.max_semantic_attempts
        }
        if (
            set(state.accepted) | exhausted != set(scope)
            or {record.scenario_id for record in state.records} - set(scope)
            or complete == bool(exhausted)
        ):
            raise ShardError(
                f"{draw_dir}: its ledger is not its units' as its gate says (accepted {len(state.accepted)} and "
                f"exhausted {len(exhausted)} of {len(scope)})"
            )
        rounds: Counter = Counter()
        for record in state.records:
            key = (record.scenario_id, record.semantic_attempt_id)
            parts.entries.append(_Entry(
                record.semantic_attempt_id, rounds[key], positions[record.scenario_id], shard.assignment.index, record
            ))
            rounds[key] += 1
        for line in _complete_lines(draw_dir / REQUEST_BLOCKS_FNAME):
            row = json.loads(line)
            parts.block_rows[(shard.assignment.index, row["request_block_id"])] = row
        for row in ledger.iter_raw():
            parts.raw.setdefault((shard.assignment.index, *raw_row_key(row)), []).append(row)
        if complete:
            _read_views(parts, draw_dir, state, scope, condition.judge_type)
            implicated = draw_dir / IMPLICATED_RULES_FNAME
            parts.implicated.append(implicated.read_bytes() if implicated.is_file() else None)
        parts.complete = parts.complete and complete
        parts.owner.update({sid: shard.assignment.index for sid in scope})
        parts.provenance.append((shard, (draw_dir / DRAW_PROVENANCE_FNAME).read_bytes()))
        parts.runs.extend(_complete_lines(draw_dir / RUNS_FNAME))
    if len(set(parts.implicated)) > 1:
        raise ShardError(f"the shards of draw {draw} wrote different {IMPLICATED_RULES_FNAME} sidecars")
    first = json.loads(parts.provenance[0][1])
    for shard, text in parts.provenance[1:]:
        other = json.loads(text)
        differing = sorted(
            key for key in set(first) | set(other) if key not in PROVENANCE_RUN_FIELDS and first.get(key) != other.get(key)
        )
        if differing:
            raise ShardError(f"{shard.label}'s draw {draw} provenance differs from {parts.provenance[0][0].label}'s "
                             f"({', '.join(differing)})")
    return parts


def _vouched_raw(parts: _DrawParts, entry: _Entry, draw: int) -> list[dict[str, Any]]:
    rows = parts.raw.pop(entry.raw_key, [])
    digests: dict[str, Optional[str]] = {}
    for row in rows:
        if row["component_id"] in digests:
            raise ShardError(f"draw {draw}: two raw outputs of {entry.record.scenario_id[:12]}/{row['component_id']}")
        digests[row["component_id"]] = raw_output_sha256(row.get("raw_response"))
    components = {component.component_id: component for component in entry.record.components}
    stray = sorted(set(digests) - set(components))
    missing = sorted(
        component_id for component_id, component in components.items()
        if component.raw_output_sha256 is not None and digests.get(component_id) != component.raw_output_sha256
    )
    if stray or missing:
        raise ShardError(
            f"draw {draw}: the raw outputs of {entry.record.scenario_id[:12]} at attempt "
            f"{entry.record.semantic_attempt_id} are not its ledger row's (missing or other text for {missing}, "
            f"no component for {stray})"
        )
    return rows


class BlockUnit(NamedTuple):
    """One ledger row as an unsharded run's block planning sees it."""

    attempt: int
    transport_round: int
    position: int
    scenario_id: str
    recorded_at: str


def replan_blocks(
    units: Sequence[BlockUnit], *, campaign_id: str, condition_id: str, draw: int, block_size: Optional[int],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return each row's block id and the ``request_blocks.jsonl`` rows an unsharded run forms over a draw's ledger
    rows."""
    ids: list[str] = []
    block_rows: list[dict[str, Any]] = []
    next_index: Counter = Counter()
    for (attempt, _), group in itertools.groupby(units, key=lambda unit: unit[:2]):
        group = list(group)
        size = block_size or len(group)
        for start in range(0, len(group), size):
            chunk = group[start : start + size]
            index = next_index[attempt]
            next_index[attempt] += 1
            block_id = request_block_id(
                campaign_id=campaign_id, condition_id=condition_id, decoder_draw_id=draw,
                semantic_attempt_id=attempt, block_index=index,
            )
            ids.extend(block_id for _ in chunk)
            block_rows.append({
                "request_block_id": block_id,
                "block_index": index,
                "semantic_attempt_id": attempt,
                "scenario_ids": [unit.scenario_id for unit in chunk],
                "recorded_at": min(unit.recorded_at for unit in chunk),
            })
    return ids, block_rows


def _write_draw(parts: _DrawParts, context: _Context, draw: int, destination: Path) -> dict[str, int]:
    condition = context.condition
    config = sampling_config_from_manifest(context.sampling_manifest)
    parts.entries.sort(key=lambda entry: entry[:3])
    units = []
    for entry in parts.entries:
        row = parts.block_rows.get((entry.shard_index, entry.record.request_block_id))
        if row is None:
            raise ShardError(
                f"draw {draw}: a ledger row names block {entry.record.request_block_id}, which its shard's "
                f"{REQUEST_BLOCKS_FNAME} does not record"
            )
        units.append(BlockUnit(entry.attempt, entry.transport_round, entry.position, entry.record.scenario_id,
                               str(row["recorded_at"])))
    ids, block_rows = replan_blocks(
        units, campaign_id=context.campaign_id, condition_id=condition.condition_id, draw=draw,
        block_size=config.scenarios_per_request_block,
    )
    new_ids = {entry.raw_key: block_id for entry, block_id in zip(parts.entries, ids)}
    state = LedgerState()
    records = []
    raw_rows: list[dict[str, Any]] = []
    relabelled = 0
    for entry in parts.entries:
        new_id = new_ids[entry.raw_key]
        relabelled += new_id != entry.record.request_block_id
        merged = replace(entry.record, request_block_id=new_id)
        state.add(merged, where=f"merged draw {draw}")
        records.append(merged)
        raw_rows.extend({**row, "request_block_id": new_id} for row in _vouched_raw(parts, entry, draw))
    if parts.raw:
        raise ShardError(f"draw {draw}: {len(parts.raw)} raw output group(s) no ledger row vouches for")
    destination.mkdir(parents=True)
    DrawLedger(
        destination, campaign_id=context.campaign_id, condition_id=condition.condition_id, decoder_draw_id=draw
    ).append(records, raw_rows)
    (destination / REQUEST_BLOCKS_FNAME).write_text("".join(json.dumps(row) + "\n" for row in block_rows),
                                                    encoding="utf-8")
    provenance = json.loads(parts.provenance[0][1])
    if parts.complete:
        _write_views(parts, context, draw, destination, new_ids, provenance)
    (destination / DRAW_PROVENANCE_FNAME).write_bytes(parts.provenance[0][1])
    (destination / RUNS_FNAME).write_text("".join(parts.runs), encoding="utf-8")
    spec = DrawSpec(
        campaign_id=context.campaign_id, condition=condition, decoder_draw_id=draw, config=config,
        scenario_manifest=context.scenario_manifest, judge_type=provenance["judge_type"],
    )
    write_json_sidecar(destination / COMPLETION_FNAME, draw_completion(
        state, spec=spec, scope_ids=context.scenario_ids, components_per_bundle=len(provenance["component_ids"]),
        materialized=parts.complete, updated_at=utc_now(), runs_logged=len(parts.runs),
    ))
    amended = {"amended": dict(parts.amended)} if parts.amended else {}
    return {"attempt_rows": len(records), "request_blocks": len(block_rows), "relabelled_rows": relabelled, **amended}


def _write_views(
    parts: _DrawParts, context: _Context, draw: int, destination: Path,
    new_ids: Mapping[tuple[int, str, int, str], str], provenance: Mapping[str, Any],
) -> None:
    ids = context.scenario_ids
    for name in (BUNDLE_OUTCOMES_FNAME, EVALUATIONS_FNAME):
        (destination / name).write_text("".join(parts.views[name][sid] for sid in ids), encoding="utf-8")
    with (destination / JUDGE_OUTPUTS_FNAME).open("w", encoding="utf-8") as handle:
        for sid in ids:
            row = json.loads(parts.views[JUDGE_OUTPUTS_FNAME][sid])
            for components in row["judge_outputs"].values():
                for by_judge in components.values():
                    for trace in by_judge.values():
                        for attempt in trace["attempts"]:
                            attempt["request_block_id"] = new_ids[(
                                parts.owner[sid], sid, attempt["semantic_attempt_id"], attempt["request_block_id"]
                            )]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if parts.implicated[0] is not None:
        (destination / IMPLICATED_RULES_FNAME).write_bytes(parts.implicated[0])
    write_draw_summary(
        destination,
        [json.loads(parts.views[EVALUATIONS_FNAME][sid]) for sid in ids],
        model_name=provenance["model_name"],
        judge_type=provenance["judge_type"],
        evaluation_mode=context.condition.evaluation_mode,
        decoder_draw_id=draw,
        condition_id=context.condition.condition_id,
        components_per_bundle=len(provenance["component_ids"]),
    )


def _same_draw(existing: Path, merged: Path) -> bool:
    names = sorted(path.name for path in merged.iterdir())
    if sorted(path.name for path in existing.iterdir()) != names:
        return False
    for name in names:
        if name == COMPLETION_FNAME:
            ours, theirs = _read_json(merged / name), _read_json(existing / name)
            ours.pop("updated_at")
            theirs.pop("updated_at", None)
            if RUN_LOG_ROWS_KEY not in theirs:
                ours.pop(RUN_LOG_ROWS_KEY)
            if ours != theirs:
                return False
        elif (merged / name).read_bytes() != (existing / name).read_bytes():
            return False
    return True


def _check_destinations(root: Path, places: Sequence[Path]) -> None:
    # A file the merge writes in the root goes through a rename, which replaces a link in its place rather than
    # following it.
    from ruleforge.evaluate.api import _is_campaign_root

    for place in places:
        current = root
        for part in place.parts:
            current = current / part
            if current.is_symlink():
                raise ShardError(f"{current} is a symlink: the merge would write through it, out of the campaign root "
                                 f"{root}; replace it with a directory of the root's own")
            if _is_campaign_root(current):
                raise ShardError(f"{current} is a campaign root of its own inside {root}: a merge never writes into "
                                 "another root")


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _marked(root: Path) -> bool:
    marker = root / CONFIGURED_MARKER_FNAME
    return marker.is_file() and not marker.is_symlink()


def _copy_if_absent(source: Path, destination: Path) -> None:
    if not destination.exists():
        _copy(source, destination)


def check_merged_draw(
    campaign_root: Path | str, condition_id: str, draw: int, *, num_shards: Optional[int] = None,
    shards_dir: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Hold a merged draw to its shard roots as they stand: merge them again into a throwaway directory and compare file
    for file. Reads only and takes no lock; returns the merge's report of the draw."""
    root = Path(campaign_root)
    directory = Path(shards_dir) if shards_dir is not None else shards_root(root) / condition_id
    existing = root / condition_id / draw_dirname(draw)
    shards = discover_shards(directory, num_shards=num_shards)
    check_units(shards)
    context = _check_identity(shards, condition_id)
    _check_campaign_root(root, context)
    covering = [shard for shard in shards if draw in shard.assignment.draws]
    if not covering:
        raise ShardError(f"no shard root in {directory} holds draw {draw}: {existing} cannot be held to its shards")
    with tempfile.TemporaryDirectory(prefix="ruleforge-merged-draw-") as scratch:
        merged = Path(scratch) / draw_dirname(draw)
        counts = _write_draw(_read_draw(covering, context, draw), context, draw, merged)
        if not (existing.is_dir() and _same_draw(existing, merged)):
            raise ShardError(
                f"{existing} is not the merge of its shard roots in {directory} as they stand: the root's draw, or a "
                "shard's, changed since the merge (merge the draw again: move the root's aside, then "
                "`ruleforge campaign merge-shards`)"
            )
    return counts


@contextmanager
def root_writing(campaign_root: Path | str) -> Iterator[None]:
    """Hold ``campaign_root``'s write lock if ``ruleforge.evaluate.api`` created it; in any other root nothing is
    taken."""
    from ruleforge.evaluate.api import _own_marker, _root_lock

    location = Path(os.path.realpath(campaign_root))
    if not _own_marker(location):
        yield
        return
    with _root_lock(location):
        yield


def merge_condition_shards(
    campaign_root: Path | str,
    condition_id: str,
    *,
    shards_dir: Optional[Path | str] = None,
    num_shards: Optional[int] = None,
    writing: Optional[Callable[[], ContextManager[Any]]] = None,
) -> dict[str, Any]:
    """Merge one condition's shard roots into ``campaign_root`` under ``writing()`` and the campaign lock, recompute the
    gate and return what was merged; ``shards_dir`` defaults to ``<campaign_root>.shards/<condition_id>``."""
    from ruleforge.evaluate.api import refuse_protected

    refuse_protected(campaign_root, "the campaign root a shard merge writes", campaign_root=True)
    with (root_writing(campaign_root) if writing is None else writing()), campaign_lock(campaign_root):
        return _merge_condition_shards(Path(campaign_root), condition_id, shards_dir=shards_dir,
                                       num_shards=num_shards)


def _merge_condition_shards(
    root: Path, condition_id: str, *, shards_dir: Optional[Path | str], num_shards: Optional[int]
) -> dict[str, Any]:
    refuse_retired_root(root)  # under the locks, before anything is copied
    directory = Path(shards_dir) if shards_dir is not None else shards_root(root) / condition_id
    shards = discover_shards(directory, num_shards=num_shards)
    check_units(shards)
    context = _check_identity(shards, condition_id)
    _check_campaign_root(root, context)
    # Held under the lock, so no registration lands between these checks and the copy.
    check_condition_id(root, context.condition)
    check_judge_configuration(root, [context.condition])
    condition_dir = root / condition_id
    # A condition with no judgments takes the record of the run whose judgments come first.
    replace_stack = condition_stack(condition_dir) is None
    report: dict[str, Any] = {}
    root.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{root.name}{MERGING_SUFFIX}", dir=root.parent))
    try:
        staged: list[tuple[int, Path, Path]] = []
        for draw in shards[0].assignment.draw_ids:
            covering = [shard for shard in shards if draw in shard.assignment.draws]
            parts = _read_draw(covering, context, draw)
            temporary = scratch / draw_dirname(draw)
            counts = _write_draw(parts, context, draw, temporary)
            staged.append((draw, temporary, condition_dir / draw_dirname(draw)))
            report[str(draw)] = {"shards": [shard.label for shard in covering], **counts}
        archives = [archive for shard in shards
                    for archive in sorted((shard.root / SNAPSHOT_DIRNAME).glob("*.tar.gz"))]
        _check_destinations(root, [Path(condition_id), *(final.relative_to(root) for _, _, final in staged),
                                   *([Path(SNAPSHOT_DIRNAME)] if archives else [])])
        conflicts = [final for _, temporary, final in staged if final.exists() and not _same_draw(final, temporary)]
        if conflicts:
            raise ShardError(
                f"{root} already holds draw(s) {[path.name for path in conflicts]} of {condition_id!r} with other "
                "content; a merge adds draws, never replaces one (move them aside to merge these)"
            )
        first = shards[0]
        if not os.path.lexists(root / SAMPLING_MANIFEST_FNAME) and _marked(first.root):
            _copy_if_absent(first.root / CONFIGURED_MARKER_FNAME, root / CONFIGURED_MARKER_FNAME)
        # Scenario manifest first: a merge stopped between the two copies can be run again.
        for name in (SCENARIO_MANIFEST_FNAME, SAMPLING_MANIFEST_FNAME):
            _copy_if_absent(first.root / name, root / name)
        for archive in archives:
            _copy_if_absent(archive, root / SNAPSHOT_DIRNAME / archive.name)
        _copy_if_absent(first.root / condition_id / CONDITION_FNAME, condition_dir / CONDITION_FNAME)
        if replace_stack:
            _copy(first.root / condition_id / CONDITION_STACK_FNAME, condition_dir / CONDITION_STACK_FNAME)
        for draw, temporary, final in staged:
            merged = not final.exists()
            if merged:
                os.replace(temporary, final)
            report[str(draw)]["state"] = "merged" if merged else "already merged"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    gate = campaign_completion(root)
    logger.info("Merged %d shard(s) of %s into %s: %s", len(shards), condition_id, root, report)
    return {
        "campaign_root": str(root),
        "condition_id": condition_id,
        "shards_dir": str(directory),
        "num_shards": len(shards),
        "draws": report,
        "condition_state": {
            draw: entry["state"] for draw, entry in gate["conditions"][condition_id]["draws"].items()
        },
        "campaign_state": gate["state"],
    }
