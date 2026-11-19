from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Optional, Sequence

from ruleforge.evaluate.chain import (
    CHAIN_KEY,
    RebaseError,
    _refuse,
    rebase_chain,
    rebase_hop,
)
from ruleforge.core.paths import artifact_uri_or_path
from ruleforge.evaluate.amendment import AMENDED_FIELD, AmendmentError, cap_amendment
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    REBASE_FNAME,
    SCENARIO_MANIFEST_FNAME,
    campaign_completion,
    campaign_manifest,
    check_judge_configuration,
    load_conditions,
    rebase_record,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.identity import (
    BUNDLE_OUTCOMES_FNAME,
    CAMPAIGN_MODE_SCREENING,
    COMPLETION_FNAME,
    SHARED_BANK_DESIGN_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    MANIFEST_RUN_FIELDS,
    ConditionSpec,
    SamplingConfig,
    differing_manifest_fields,
    draw_dirname,
    load_sampling_manifest,
    run_log_damage,
    run_log_problem,
    sampling_config_from_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.files import (
    json_sidecar_text,
    write_json_sidecar,
)
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.bank import group_by_model, load_design
from ruleforge.evaluate.identity import REPLAY_RECORD_FNAME
from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME, DrawLedger
from ruleforge.core.provenance import SNAPSHOT_DIRNAME, code_state, write_code_snapshot
from ruleforge.evaluate.configured import MARKER_FNAME as CONFIGURED_MARKER_FNAME
from ruleforge.evaluate.identity import (
    DRAW_PROVENANCE_FNAME,
    RUNS_FNAME,
)
from ruleforge.evaluate.runner import (
    EVALUATIONS_FNAME,
    REQUEST_BLOCKS_FNAME,
    STATE_COMPLETE,
    STATE_INCOMPLETE_EXHAUSTED,
    CampaignRunError,
    draw_lock_path,
)
from ruleforge.evaluate.shard_merge import Shard, discover_shards, merge_condition_shards
from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.core.record_names import PER_SCENARIO_FNAME, SUMMARY_FNAME
from ruleforge.rulesets.variants import IMPLICATED_RULES_FNAME
from ruleforge.evaluate.sharding import campaign_lock
from ruleforge.evaluate.stack import CONDITION_STACK_FNAME, unlogged_stacks

logger = logging.getLogger(__name__)

#: Every file the runner's ``materialize()`` derives from a draw's ledger.
DRAW_VIEW_FNAMES = (
    BUNDLE_OUTCOMES_FNAME,
    EVALUATIONS_FNAME,
    JUDGE_OUTPUTS_FNAME,
    PER_SCENARIO_FNAME,
    SUMMARY_FNAME,
    IMPLICATED_RULES_FNAME,
)

REBASE_SCHEMA_VERSION = 3
REBASE_KIND = "population_rebase"
REPOINTED_MANIFEST_FIELDS = ("scenario_manifest_sha256", "scenario_source_sha256", "expected_scenarios")

FILTERED_LEDGER_FNAMES = (LEDGER_FNAME, RAW_OUTPUTS_FNAME)
#: The request-block record: removed scenarios taken out of each membership.
FILTERED_BLOCKS_FNAME = REQUEST_BLOCKS_FNAME
#: Re-pointed at the target manifest; its run fields untouched.
REWRITTEN_SIDECAR_FNAME = DRAW_PROVENANCE_FNAME
#: Copied byte for byte: the draw's run log.
COPIED_SIDECAR_FNAMES = (RUNS_FNAME,)
_CONDITION_FNAMES = (CONDITION_FNAME, CONDITION_STACK_FNAME)
#: Not copied: the gate and every view, rebuilt by the resumed runner.
SKIPPED_DRAW_FNAMES = (COMPLETION_FNAME, *DRAW_VIEW_FNAMES)
KNOWN_DRAW_FNAMES = frozenset(
    (*FILTERED_LEDGER_FNAMES, FILTERED_BLOCKS_FNAME, REWRITTEN_SIDECAR_FNAME,
     *COPIED_SIDECAR_FNAMES, *SKIPPED_DRAW_FNAMES)
)
REQUIRED_DRAW_FNAMES = (
    *FILTERED_LEDGER_FNAMES, FILTERED_BLOCKS_FNAME, REWRITTEN_SIDECAR_FNAME,
    *COPIED_SIDECAR_FNAMES, COMPLETION_FNAME,
)

_ROOT_FILES = frozenset(
    {SAMPLING_MANIFEST_FNAME, SCENARIO_MANIFEST_FNAME, COMPLETION_FNAME, SHARED_BANK_DESIGN_FNAME, REBASE_FNAME,
     CONFIGURED_MARKER_FNAME}
)
_AGGREGATE_DIRNAME = "aggregate"
_ANALYSIS_DIRNAME = "analysis"
_HYDRA_DIRNAME = "hydra"
_ROOT_DIRS = frozenset({_AGGREGATE_DIRNAME, _ANALYSIS_DIRNAME, _HYDRA_DIRNAME, SNAPSHOT_DIRNAME})
_LEFT_BEHIND = (_ROOT_DIRS - {SNAPSHOT_DIRNAME}) | {COMPLETION_FNAME, REBASE_FNAME, CONFIGURED_MARKER_FNAME}
PARTIAL_MARKER = ".partial-"


@dataclass(frozen=True)
class DrawRef:
    """One draw of the parent."""

    condition: ConditionSpec
    decoder_draw_id: int
    #: The root the draw is read from: the parent's, or a shard source's merged root.
    source_root: Path
    carried_exhausted: tuple[str, ...] = ()

    @property
    def relative(self) -> Path:
        return Path(self.condition.condition_id) / draw_dirname(self.decoder_draw_id)


@dataclass(frozen=True)
class ShardSource:
    """A condition the parent holds only in shard roots, merged into a throwaway root at the plan (read only)."""

    condition_id: str
    shards_dir: Path
    shards: tuple[Shard, ...]
    #: The merged root (a campaign root holding this condition alone), under ``holder``.
    merged_root: Path
    #: Every file of every shard root as the plan found it: ``{path: (size, mtime_ns)}``.
    stamps: Mapping[str, tuple[int, int]]
    holder: tempfile.TemporaryDirectory = field(repr=False, compare=False)


@dataclass(frozen=True)
class RebasePlan:
    """Everything :func:`execute_rebase` needs, checked against the parent read-only."""

    parent_root: Path
    population_path: Path
    parent_manifest: Mapping[str, Any]
    parent_manifest_file_sha256: str
    config: SamplingConfig
    parent_scenarios: ScenarioManifest
    target: ScenarioManifest
    conditions: Mapping[str, ConditionSpec]
    draws: tuple[DrawRef, ...]
    #: Scenario ids: retained and added in target order, removed in parent order.
    retained: tuple[str, ...]
    removed: tuple[str, ...]
    added: tuple[str, ...]
    parent_gate_mtime_ns: int
    #: The reverted cap amendment: ``{"field", "from", "to", "amendments"}`` (the parent's record verbatim).
    cap_reversion: Optional[Mapping[str, Any]] = None
    shard_sources: tuple[ShardSource, ...] = ()
    parent_chain: tuple[Mapping[str, Any], ...] = ()

    @property
    def campaign_id(self) -> str:
        return str(self.parent_manifest["campaign_id"])

    @property
    def n_draws(self) -> int:
        return int(self.parent_manifest["expected_decoder_draws"])

    def close(self) -> None:
        for source in self.shard_sources:
            source.holder.cleanup()


def rebased_sampling_manifest(
    parent: Mapping[str, Any], fresh: Mapping[str, Any], *, revert_cap: bool = False
) -> dict[str, Any]:
    """The parent's campaign manifest with the population and run fields of ``fresh``."""
    base = dict(parent)
    repointed = [*REPOINTED_MANIFEST_FIELDS, *MANIFEST_RUN_FIELDS]
    if revert_cap:
        base.pop(MANIFEST_AMENDMENTS_KEY)
        repointed.append(AMENDED_FIELD)
    return {**base, **{key: fresh[key] for key in repointed}}


def plan_rebase(
    parent_root: Path | str,
    population_path: Path | str,
    *,
    approved_sha256: Optional[str] = None,
    shard_sources: Sequence[Path | str] = (),
    carry_exhausted: Sequence[str] = (),
) -> RebasePlan:
    """Check a parent root and a revised population; write nothing in either."""
    root = Path(parent_root)
    _refuse(root.is_dir(), f"{root} is not a campaign root")
    _refuse(not (root / REPLAY_RECORD_FNAME).exists(), (
        f"{root} is a derived root ({REPLAY_RECORD_FNAME}): rebase the root it was re-scored from and "
        "re-score the rebased root (ruleforge campaign rescore), so its replay record describes the rows it was "
        "made from"
    ))
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    _refuse(manifest.get("campaign_mode") != CAMPAIGN_MODE_SCREENING, (
        f"campaign {manifest.get('campaign_id')!r} is a screening campaign: its population "
        "is cut from its ledgers by the freeze step, never rebased"
    ))
    parent_scenarios = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME)
    _refuse(
        (parent_scenarios.manifest_sha256, parent_scenarios.source_sha256, parent_scenarios.record_count)
        == (manifest["scenario_manifest_sha256"], manifest["scenario_source_sha256"], manifest["expected_scenarios"]),
        f"{root}: {SCENARIO_MANIFEST_FNAME} is not the population {SAMPLING_MANIFEST_FNAME} names",
    )
    config = sampling_config_from_manifest(manifest)

    population = Path(population_path)
    if approved_sha256 is not None:
        actual = sha256_file(population)
        _refuse(actual == str(approved_sha256), (
            f"population SHA-256 {actual} does not match approved {approved_sha256}"
        ))
    target = ScenarioManifest.build(population, expected_count=config.expected_scenario_count)
    _refuse(target.manifest_sha256 != parent_scenarios.manifest_sha256, (
        f"{population.name} freezes the parent's own manifest "
        f"({target.manifest_sha256[:12]}...); there is nothing to rebase"
    ))
    parent_ids = set(parent_scenarios.scenario_ids)
    target_ids = set(target.scenario_ids)
    retained = tuple(sid for sid in target.scenario_ids if sid in parent_ids)
    _refuse(bool(retained), (
        f"{population.name} shares no scenario with the parent; a fresh campaign, not a rebase"
    ))
    removed = tuple(sid for sid in parent_scenarios.scenario_ids if sid not in target_ids)
    config, reversion = _rebased_config(manifest, config, frozenset(removed))
    expected = campaign_manifest(str(manifest["campaign_id"]), config, target)
    rebased = rebased_sampling_manifest(manifest, expected, revert_cap=reversion is not None)
    differing = differing_manifest_fields(rebased, expected)
    _refuse(not differing, (
        f"{root}/{SAMPLING_MANIFEST_FNAME} is not the manifest this code builds for its own "
        f"config ({', '.join(differing)} differ); a rebased root could not be resumed"
    ))
    carried = frozenset(carry_exhausted)
    stray = sorted(sid[:12] for sid in carried if sid not in set(retained))
    _refuse(not stray, f"carry_exhausted names scenario(s) the target does not keep: {stray}")

    try:
        parent_record = rebase_record(root)
    except CampaignRunError as error:
        raise RebaseError(str(error)) from error
    parent_chain = () if parent_record is None else tuple(rebase_chain(parent_record))

    # Read first: whatever a job writes from here on is newer than this.
    gate_mtime_ns = _root_gate_mtime_ns(root)
    conditions = load_conditions(root)
    if (root / SHARED_BANK_DESIGN_FNAME).exists():
        group_by_model(conditions, load_design(root))
    n_draws = int(manifest["expected_decoder_draws"])
    draws = _check_parent_gate(
        root, manifest,
        [DrawRef(condition, draw, root) for condition in conditions.values() for draw in range(n_draws)],
        frozenset(removed), carried,
    )
    _check_inventory(root, conditions, n_draws)
    _check_no_live_writer(root, n_draws, gate_mtime_ns)
    sources: list[ShardSource] = []
    try:
        for directory in shard_sources:
            source, condition, source_draws = _open_shard_source(
                Path(directory), manifest, parent_scenarios, conditions, n_draws, frozenset(removed), carried,
            )
            sources.append(source)
            conditions = {**conditions, condition.condition_id: condition}
            draws = (*draws, *source_draws)
        if sources:
            try:
                check_judge_configuration(root, [conditions[source.condition_id] for source in sources])
            except CampaignRunError as error:
                raise RebaseError(str(error)) from error
        unused = sorted(sid[:12] for sid in carried if not any(sid in ref.carried_exhausted for ref in draws))
        _refuse(not unused, f"carry_exhausted names scenario(s) no parent draw holds exhausted: {unused}")
    except BaseException:
        for source in sources:
            source.holder.cleanup()
        raise
    return RebasePlan(
        parent_root=root,
        population_path=population,
        parent_manifest=manifest,
        parent_manifest_file_sha256=sha256_file(root / SAMPLING_MANIFEST_FNAME),
        config=config,
        parent_scenarios=parent_scenarios,
        target=target,
        conditions=conditions,
        draws=tuple(draws),
        retained=retained,
        removed=removed,
        added=tuple(sid for sid in target.scenario_ids if sid not in parent_ids),
        parent_gate_mtime_ns=gate_mtime_ns,
        cap_reversion=reversion,
        shard_sources=tuple(sources),
        parent_chain=parent_chain,
    )


def _rebased_config(
    manifest: Mapping[str, Any], config: SamplingConfig, removed: frozenset[str]
) -> tuple[SamplingConfig, Optional[dict[str, Any]]]:
    try:
        amendment = cap_amendment(manifest)
    except AmendmentError as error:
        raise RebaseError(str(error)) from error
    if amendment is None:
        return config, None
    kept = [bundle.describe() for bundle in amendment.bundles if bundle.scenario_id not in removed]
    _refuse(not kept, (
        f"campaign {manifest['campaign_id']!r}'s cap amendment (ledger {amendment.ledger}) names bundle(s) the "
        f"target keeps ({kept[:3]}): a rebase reverts an amendment only when every bundle it names is removed"
    ))
    _refuse(config.max_semantic_attempts == amendment.new_cap, (
        f"campaign {manifest['campaign_id']!r} records {AMENDED_FIELD} {config.max_semantic_attempts}, not its "
        f"amendment's new cap {amendment.new_cap}"
    ))
    reversion = {
        "field": AMENDED_FIELD,
        "from": amendment.new_cap,
        "to": amendment.old_cap,
        "amendments": list(manifest[MANIFEST_AMENDMENTS_KEY]),
    }
    return replace(config, max_semantic_attempts=amendment.old_cap), reversion


def _open_shard_source(
    directory: Path,
    manifest: Mapping[str, Any],
    parent_scenarios: ScenarioManifest,
    parent_conditions: Mapping[str, ConditionSpec],
    n_draws: int,
    removed: frozenset[str],
    carried: frozenset[str],
) -> tuple[ShardSource, ConditionSpec, tuple[DrawRef, ...]]:
    """A condition held in shard roots, merged into a throwaway root and checked as a parent condition is."""
    shards = tuple(discover_shards(directory))
    condition_id = str(shards[0].record["condition_id"])
    _refuse(condition_id not in parent_conditions, (
        f"{directory}: {condition_id} is registered in the parent root too; a shard source is a condition the "
        "parent holds only in shard roots"
    ))
    stamps = _shard_stamps(shards)
    holder = tempfile.TemporaryDirectory(prefix="ruleforge-rebase-shards-")
    try:
        merged = Path(holder.name) / "merged"
        merge_condition_shards(merged, condition_id, shards_dir=directory)
        differing = differing_manifest_fields(manifest, load_sampling_manifest(merged / SAMPLING_MANIFEST_FNAME))
        _refuse(not differing, (
            f"{directory}: its shards ran another campaign than the parent ({', '.join(differing)} differ)"
        ))
        _refuse(
            ScenarioManifest.load(merged / SCENARIO_MANIFEST_FNAME).manifest_sha256 == parent_scenarios.manifest_sha256,
            f"{directory}: its shards froze another population than the parent",
        )
        registered = load_conditions(merged)
        draws = _check_parent_gate(
            merged, manifest,
            [DrawRef(registered[condition_id], draw, merged) for draw in range(n_draws)],
            removed, carried,
        )
        _check_inventory(merged, registered, n_draws)
    except BaseException:
        holder.cleanup()
        raise
    source = ShardSource(condition_id, directory, shards, merged, stamps, holder)
    return source, registered[condition_id], draws


def _shard_stamps(shards: Sequence[Shard]) -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for shard in shards for path in sorted(shard.root.rglob("*")) if path.is_file()
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RebaseError(f"cannot read {path}: {error}") from error
    _refuse(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def _check_parent_gate(
    root: Path, manifest: Mapping[str, Any], draws: Sequence[DrawRef], removed: frozenset[str],
    carried: frozenset[str],
) -> tuple[DrawRef, ...]:
    """Every draw finished and naming the parent's identity; read only."""
    gate = campaign_completion(root, write=False)
    _refuse(gate["state"] in (STATE_COMPLETE, STATE_INCOMPLETE_EXHAUSTED), (
        f"campaign {manifest['campaign_id']!r} is {gate['state']!r}, not complete; only a "
        "finished root (complete, or exhausted at scenarios the target removes) is rebased"
    ))
    identity = {
        "campaign_id": manifest["campaign_id"],
        "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
    }
    checked = []
    for ref in draws:
        draw_dir = root / ref.relative
        completion = _read_json(draw_dir / COMPLETION_FNAME)
        provenance = _read_json(draw_dir / DRAW_PROVENANCE_FNAME)
        wanted = {
            **identity,
            "condition_fingerprint": ref.condition.fingerprint,
            "decoder_draw_id": ref.decoder_draw_id,
        }
        for name, record in ((COMPLETION_FNAME, completion), (DRAW_PROVENANCE_FNAME, provenance)):
            wrong = sorted(key for key, value in wanted.items() if record.get(key) != value)
            _refuse(not wrong, f"{draw_dir}/{name} names another {', '.join(wrong)}")
        damage = run_log_damage(draw_dir)
        cut = run_log_problem(draw_dir) or unlogged_stacks(draw_dir) or (damage[0] if damage else None)
        _refuse(not cut, f"{draw_dir}: {cut}")
        if completion.get("state") == STATE_COMPLETE:
            _refuse(
                completion.get("materialized") is True and (draw_dir / BUNDLE_OUTCOMES_FNAME).is_file(),
                f"{draw_dir} is not complete and materialised",
            )
            checked.append(ref)
            continue
        _refuse(completion.get("state") == STATE_INCOMPLETE_EXHAUSTED and completion.get("pending") == 0, (
            f"{draw_dir} is {completion.get('state')!r}, not finished: only a draw complete and materialised, or "
            "exhausted with nothing pending, is rebased"
        ))
        exhausted = set(completion.get("exhausted") or ())
        kept = sorted(sid[:12] for sid in exhausted - removed - carried)
        _refuse(not kept, (
            f"{draw_dir} holds scenario(s) {kept} exhausted, which the target keeps: name each to carry it over "
            "exhausted as the parent holds it (plan_rebase's carry_exhausted), or rebase a parent that "
            "finished it"
        ))
        checked.append(replace(ref, carried_exhausted=tuple(sorted(exhausted & carried))))
    return tuple(checked)


def _root_gate_mtime_ns(root: Path) -> int:
    gate = root / COMPLETION_FNAME
    _refuse(gate.is_file(), (
        f"{gate} is missing: a finished job writes it, and without it a live writer cannot "
        "be ruled out; once nothing runs on the parent, re-gate it "
        "(ruleforge.evaluate.campaign.campaign_completion) and plan again"
    ))
    return gate.stat().st_mtime_ns


def _check_no_live_writer(root: Path, n_draws: int, gate_mtime_ns: int) -> None:
    stamped: list[Path] = []
    for registration in sorted(root.glob(f"**/{CONDITION_FNAME}")):
        stamped.append(registration)
        stamped.extend(
            gate
            for draw in range(n_draws)
            if (gate := registration.parent / draw_dirname(draw) / COMPLETION_FNAME).exists()
        )
    newer = [path.relative_to(root).as_posix() for path in stamped if path.stat().st_mtime_ns > gate_mtime_ns]
    _refuse(not newer, (
        f"{root} has a live writer: {len(newer)} registration(s) or draw gate(s) are newer than "
        f"its {COMPLETION_FNAME} (first: {newer[:3]}). Wait until every job on the parent has "
        "stopped (the last rewrites the gate); if none is running, a job died after stamping "
        "a draw -- re-gate the parent (ruleforge.evaluate.campaign.campaign_completion) and plan again"
    ))


def _check_inventory(root: Path, conditions: Mapping[str, ConditionSpec], n_draws: int) -> None:
    condition_dirs = {Path(cid) for cid in conditions}
    prefix_dirs = {parent for path in condition_dirs for parent in path.parents if parent != Path(".")}
    draw_names = {draw_dirname(draw) for draw in range(n_draws)}
    lock_names = {draw_lock_path(Path(name)).name for name in draw_names}
    unknown: list[str] = []
    for entry in sorted(root.iterdir()):
        if Path(entry.name) in condition_dirs | prefix_dirs:
            continue
        known_dir = entry.is_dir() and entry.name in _ROOT_DIRS
        if not (known_dir or (entry.is_file() and entry.name in _ROOT_FILES)):
            unknown.append(entry.name)
    for relative in sorted(condition_dirs | prefix_dirs):
        for entry in sorted((root / relative).iterdir()):
            child = relative / entry.name
            if child in condition_dirs or child in prefix_dirs:
                continue
            if relative in condition_dirs and entry.name in _CONDITION_FNAMES and entry.is_file():
                continue
            if relative in condition_dirs and entry.name in lock_names and entry.is_file():
                continue
            if relative in condition_dirs and entry.name in draw_names and entry.is_dir():
                unknown.extend(_unknown_draw_files(entry, child))
                continue
            unknown.append(child.as_posix())
    _refuse(not unknown, (
        f"{root} holds {len(unknown)} file(s) a rebase does not know how to carry over "
        f"(first: {unknown[:5]}); move them aside or teach the rebase about them"
    ))


def _unknown_draw_files(draw_dir: Path, relative: Path) -> list[str]:
    names = {entry.name: entry for entry in draw_dir.iterdir()}
    unknown = [
        (relative / name).as_posix()
        for name, entry in sorted(names.items())
        if name not in KNOWN_DRAW_FNAMES or not entry.is_file()
    ]
    unknown.extend(
        f"{(relative / name).as_posix()} (missing)" for name in REQUIRED_DRAW_FNAMES if name not in names
    )
    return unknown


@dataclass
class _Totals:
    attempt_rows_dropped: int = 0
    raw_output_rows_dropped: int = 0
    block_memberships_dropped: int = 0
    request_blocks_rewritten: int = 0
    request_blocks_emptied: int = 0
    views_skipped: int = 0
    bytes_written: int = 0
    #: The most attempts any retained bundle used in any draw: the evidence for a cap reversion.
    max_retained_attempts_used: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _lines(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        for line in handle:
            _refuse(line.endswith(b"\n"), f"{path} ends in a torn record; a complete draw has none")
            yield line


def _filter_ledger(
    source: Path, destination: BinaryIO, removed: frozenset[str], tokens: Sequence[bytes]
) -> tuple[int, int]:
    dropped = kept_bytes = 0
    for line in _lines(source):
        if any(token in line for token in tokens) and json.loads(line)["scenario_id"] in removed:
            dropped += 1
            continue
        kept_bytes += len(line)
        destination.write(line)
    return dropped, kept_bytes


def _filter_blocks(
    source: Path, destination: BinaryIO, removed: frozenset[str], tokens: Sequence[bytes], totals: _Totals
) -> int:
    written = 0
    for line in _lines(source):
        if any(token in line for token in tokens):
            row = json.loads(line)
            _refuse(
                (json.dumps(row) + "\n").encode("utf-8") == line,
                f"{source}: a request-block row does not re-serialise to its own bytes",
            )
            kept = [sid for sid in row["scenario_ids"] if sid not in removed]
            dropped = len(row["scenario_ids"]) - len(kept)
            totals.block_memberships_dropped += dropped
            totals.request_blocks_rewritten += int(dropped > 0)
            totals.request_blocks_emptied += int(dropped > 0 and not kept)
            line = (json.dumps({**row, "scenario_ids": kept}) + "\n").encode("utf-8")
        written += len(line)
        destination.write(line)
    return written


def _rebase_draw(plan: RebasePlan, ref: DrawRef, destination: Path, totals: _Totals) -> None:
    source = ref.source_root / ref.relative
    removed = frozenset(plan.removed)
    tokens = [sid.encode("ascii") for sid in plan.removed]
    destination.mkdir(parents=True)
    for name in FILTERED_LEDGER_FNAMES:
        with (destination / name).open("xb") as handle:
            dropped, kept = _filter_ledger(source / name, handle, removed, tokens)
        totals.bytes_written += kept
        if name == LEDGER_FNAME:
            totals.attempt_rows_dropped += dropped
        else:
            totals.raw_output_rows_dropped += dropped
    with (destination / FILTERED_BLOCKS_FNAME).open("xb") as handle:
        totals.bytes_written += _filter_blocks(source / FILTERED_BLOCKS_FNAME, handle, removed, tokens, totals)
    provenance = _read_json(source / REWRITTEN_SIDECAR_FNAME)
    provenance.update(
        scenario_manifest_sha256=plan.target.manifest_sha256,
        expected_scenarios=plan.target.record_count,
    )
    totals.bytes_written += len(json_sidecar_text(provenance).encode("utf-8"))
    write_json_sidecar(destination / REWRITTEN_SIDECAR_FNAME, provenance)
    for name in COPIED_SIDECAR_FNAMES:
        totals.bytes_written += (source / name).stat().st_size
        shutil.copyfile(source / name, destination / name)
    totals.views_skipped += sum(1 for name in DRAW_VIEW_FNAMES if (source / name).exists())
    _check_copies(destination)
    used = _verify_draw(plan, ref, destination)
    totals.max_retained_attempts_used = max(totals.max_retained_attempts_used, used)


def _verify_draw(plan: RebasePlan, ref: DrawRef, directory: Path) -> int:
    state = DrawLedger(
        directory,
        campaign_id=plan.campaign_id,
        condition_id=ref.condition.condition_id,
        decoder_draw_id=ref.decoder_draw_id,
    ).load(repair=False)
    seen = {record.scenario_id for record in state.records}
    stray = seen - set(plan.retained)
    _refuse(not stray, f"{directory}: rows remain for {len(stray)} scenario(s) outside the retained set")
    carried = set(ref.carried_exhausted)
    missing = [sid for sid in plan.retained if sid not in carried and not state.is_accepted(sid)]
    _refuse(not missing, f"{directory}: {len(missing)} retained scenario(s) have no accepted bundle")
    cap = plan.config.max_semantic_attempts
    unfinished = sorted(sid[:12] for sid in carried if state.is_accepted(sid) or state.next_attempt(sid) < cap)
    _refuse(not unfinished, f"{directory}: carried scenario(s) {unfinished} are not exhausted at the cap {cap}")
    used = max((state.next_attempt(sid) for sid in plan.retained), default=0)
    _refuse(used <= cap, (
        f"{directory}: a retained bundle used {used} attempts, more than the rebased root's cap {cap}: "
        "the reverted cap would strand it"
    ))
    return used


def _check_copies(directory: Path) -> None:
    linked = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file() and path.stat().st_nlink != 1
    )
    _refuse(not linked, f"{directory}: {len(linked)} file(s) share an inode with another (first: {linked[:3]})")


def _check_parent_unchanged(plan: RebasePlan) -> None:
    _check_no_live_writer(plan.parent_root, plan.n_draws, plan.parent_gate_mtime_ns)
    _refuse(sha256_file(plan.parent_root / SAMPLING_MANIFEST_FNAME) == plan.parent_manifest_file_sha256, (
        f"{plan.parent_root}/{SAMPLING_MANIFEST_FNAME} changed while the rebase read the root (a cap amendment's "
        "record, or a merge); plan again once nothing writes it"
    ))
    for source in plan.shard_sources:
        _refuse(_shard_stamps(source.shards) == dict(source.stamps), (
            f"{source.shards_dir}: a shard root changed while the rebase read it; plan again once no job of "
            f"{source.condition_id} runs"
        ))


def _ledger_digests(directory: Path) -> dict[str, str]:
    return {
        "attempts_sha256": sha256_file(directory / LEDGER_FNAME),
        "raw_outputs_sha256": sha256_file(directory / RAW_OUTPUTS_FNAME),
    }


def _shard_source_record(source: ShardSource, n_draws: int) -> dict[str, Any]:
    draws = {}
    for draw in range(n_draws):
        draws[str(draw)] = {
            "shards": [
                {"shard": shard.label, **_ledger_digests(shard.draw_dir(source.condition_id, draw))}
                for shard in source.shards if (shard.draw_dir(source.condition_id, draw) / LEDGER_FNAME).is_file()
            ],
            "merged": _ledger_digests(source.merged_root / source.condition_id / draw_dirname(draw)),
        }
    return {
        "condition_id": source.condition_id,
        "shards_dir": artifact_uri_or_path(source.shards_dir),
        "num_shards": len(source.shards),
        "draws": draws,
    }


def _build(
    plan: RebasePlan,
    root: Path,
    payload: dict[str, Any],
    sampling_manifest: Mapping[str, Any],
    snapshot_files: Sequence[tuple[Path, Path]],
    code: Mapping[str, Any],
    *,
    keep_draws: bool,
) -> None:
    """Write the rebased root into the (empty) directory ``root``, checking every part."""
    totals = _Totals()
    plan.target.write(root / SCENARIO_MANIFEST_FNAME)
    write_json_sidecar(root / SAMPLING_MANIFEST_FNAME, sampling_manifest)
    if payload["copied"]["figure2_design"]:
        shutil.copyfile(plan.parent_root / SHARED_BANK_DESIGN_FNAME, root / SHARED_BANK_DESIGN_FNAME)
    for source_root, path in snapshot_files:
        target = root / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():  # a shard source's archive of a digest the parent's already holds
            shutil.copyfile(path, target)
    write_code_snapshot(root, dict(code))
    for condition_id, source_root in _condition_roots(plan).items():
        (root / condition_id).mkdir(parents=True)
        for name in _CONDITION_FNAMES:
            if (source_root / condition_id / name).is_file():
                shutil.copyfile(source_root / condition_id / name, root / condition_id / name)
    for ref in plan.draws:
        _rebase_draw(plan, ref, root / ref.relative, totals)
        if not keep_draws:
            shutil.rmtree(root / ref.relative)
    registered = load_conditions(root)
    _refuse(
        {cid: spec.fingerprint for cid, spec in registered.items()}
        == {cid: spec.fingerprint for cid, spec in plan.conditions.items()},
        f"{root}: the copied registrations are not the parent's",
    )
    _check_copies(root)
    _check_parent_unchanged(plan)
    payload["totals"] = totals.to_dict()
    write_json_sidecar(root / REBASE_FNAME, payload)


def _condition_roots(plan: RebasePlan) -> dict[str, Path]:
    return {ref.condition.condition_id: ref.source_root for ref in plan.draws}


def _partial_builds(destination: Path) -> list[str]:
    if not destination.parent.is_dir():
        return []
    prefix = f"{destination.name}{PARTIAL_MARKER}"
    return sorted(entry.name for entry in destination.parent.iterdir() if entry.name.startswith(prefix))


def execute_rebase(plan: RebasePlan, new_root: Path | str, *, dry_run: bool = False) -> dict[str, Any]:
    """Build the rebased root at ``new_root`` (which must not exist); return the report."""
    destination = Path(new_root)
    _refuse(not os.path.lexists(destination), f"{destination} already exists; a rebase builds a new root")
    _refuse(
        plan.parent_root.resolve() not in destination.resolve().parents,
        f"{destination} is inside the parent root {plan.parent_root}; a rebase never writes there",
    )
    leftover = _partial_builds(destination)
    _refuse(not leftover, (
        f"{destination.parent} holds {leftover}: a rebase into {destination.name} that did not finish (a crash). "
        "Check that none is running, remove it by hand, and run the rebase again"
    ))
    code = code_state()
    sampling_manifest = rebased_sampling_manifest(
        plan.parent_manifest, campaign_manifest(plan.campaign_id, plan.config, plan.target, code),
        revert_cap=plan.cap_reversion is not None,
    )
    snapshot_files = [
        (source_root, path)
        for source_root in (plan.parent_root, *(source.merged_root for source in plan.shard_sources))
        if (source_root / SNAPSHOT_DIRNAME).is_dir()
        for path in sorted(p for p in (source_root / SNAPSHOT_DIRNAME).rglob("*") if p.is_file())
    ]
    payload = {
        "schema_version": REBASE_SCHEMA_VERSION,
        "kind": REBASE_KIND,
        "parent": {
            "root": artifact_uri_or_path(plan.parent_root),
            "campaign_id": plan.campaign_id,
            "scenario_manifest_sha256": plan.parent_scenarios.manifest_sha256,
            "scenario_source_sha256": plan.parent_scenarios.source_sha256,
            "sampling_manifest_sha256": plan.parent_manifest_file_sha256,
        },
        "target": {
            "population": artifact_uri_or_path(plan.population_path),
            "source_sha256": plan.target.source_sha256,
            "scenario_manifest_sha256": plan.target.manifest_sha256,
            "count": plan.target.record_count,
        },
        "removed_ids": list(plan.removed),
        "added_ids": list(plan.added),
        "retained_count": len(plan.retained),
        "n_conditions": len(plan.conditions),
        "n_draws": len(plan.draws),
        "copied": {
            "figure2_design": (plan.parent_root / SHARED_BANK_DESIGN_FNAME).is_file(),
            "code_snapshots": len({path.relative_to(source_root) for source_root, path in snapshot_files}),
            "condition_stacks": sum(
                (source_root / condition_id / CONDITION_STACK_FNAME).is_file()
                for condition_id, source_root in _condition_roots(plan).items()
            ),
        },
        "left_behind": sorted(entry.name for entry in plan.parent_root.iterdir() if entry.name in _LEFT_BEHIND),
        "cap_reversion": None if plan.cap_reversion is None else dict(plan.cap_reversion),
        "carried_exhausted": {
            ref.relative.as_posix(): list(ref.carried_exhausted) for ref in plan.draws if ref.carried_exhausted
        },
        "shard_sources": [_shard_source_record(source, plan.n_draws) for source in plan.shard_sources],
        "created_at": sampling_manifest["created_at"],
        "code_state": dict(code),
    }
    # The chain: the parent's rebases (its own record's), then this one.
    payload[CHAIN_KEY] = [*(dict(hop) for hop in plan.parent_chain), rebase_hop(payload)]
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="ruleforge-rebase-dry-run-") as scratch:
            _build(plan, Path(scratch), payload, sampling_manifest, snapshot_files, code, keep_draws=False)
        return {**payload, "dry_run": True, "new_root": str(destination)}

    destination.parent.mkdir(parents=True, exist_ok=True)
    with campaign_lock(destination):
        partial = destination.with_name(f"{destination.name}{PARTIAL_MARKER}{uuid.uuid4().hex[:12]}")
        partial.mkdir()
        try:
            _build(plan, partial, payload, sampling_manifest, snapshot_files, code, keep_draws=True)
            _refuse(not os.path.lexists(destination), f"{destination} appeared while the rebase ran")
            partial.rename(destination)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    logger.info(
        "Rebased %s onto %s at %s: %d removed, %d added, %d draws",
        plan.parent_root, plan.population_path.name, destination,
        len(plan.removed), len(plan.added), len(plan.draws),
    )
    return {**payload, "dry_run": False, "new_root": str(destination)}
