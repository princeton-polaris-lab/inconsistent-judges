from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from ruleforge.evaluate.amendment import AMENDED_FIELD, AmendmentError, CapAmendment, cap_amendment, check_seeds
import math

from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    load_conditions,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    RUNS_FNAME,
    differing_manifest_fields,
    draw_dirname,
    load_sampling_manifest,
    run_log_damage,
    run_log_problem,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.campaign import rebase_record
from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME, DrawLedger
from ruleforge.evaluate.chain import (
    CHAIN_KEY,
    RebaseError,
    rebase_chain,
)
from ruleforge.evaluate.campaign import REBASE_FNAME
from tests.support.prompts import prompt_renderer, rule_sets_by_digest
from tests.support.rebase import REPOINTED_MANIFEST_FIELDS
from ruleforge.evaluate.identity import BUNDLE_OUTCOMES_FNAME
from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records
from ruleforge.core.resume import count_complete_jsonl_records

from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.evaluate.runner import (
    EVALUATIONS_FNAME,
    REQUEST_BLOCKS_FNAME,
    CampaignRunError,
)
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.evaluate.shard_merge import discover_shards, merge_condition_shards
from ruleforge.evaluate.stack import (
    JUDGED_ON_KEY,
    LOCAL_MODEL_TYPE,
    STACK_CHANGE_KEY,
    describe_changes,
    judgment_runtimes,
    recorded_runtime,
    runtime_changes,
    stack_changes,
    unlogged_stacks,
)


#: Slack on the ``1/n`` bound for floating-point rounding of the estimates.
RATE_TOLERANCE = 1e-9
#: The first rebase schema whose record carries a chain; an earlier record is read as one hop.
CHAIN_SCHEMA_VERSION = 3


class RebaseCheckError(SamplingDesignError):
    """An input the checks cannot read: not a rebased root, a missing ruleset, an unreadable file."""


def _report() -> dict[str, Any]:
    return {"problems": [], "notes": []}


def _number(value: Any) -> Optional[float]:
    """``value`` as a finite float; None when it is not one (a NaN, an infinity, a null, a string)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def rate_changes(
    before: Mapping[str, Any], after: Mapping[str, Any], *, n_scenarios: int | Mapping[str, int],
    units: float = 1.0, swaps: int = 1,
) -> dict[str, Any]:
    """Check that each named rate moved by at most the swapped scenarios' share (``swaps / n``)."""
    if not isinstance(swaps, int) or isinstance(swaps, bool) or swaps < 1:
        raise RebaseCheckError(f"swaps must be a positive integer, not {swaps!r}")
    report = _report()
    report["bound"] = {}
    report["changes"] = {}
    report["swaps"] = swaps
    shares = units * swaps
    whose = "one scenario's" if shares == 1 else f"{shares:g} scenarios' share"
    for name in sorted(set(before) | set(after)):
        if name not in before or name not in after:
            report["problems"].append(f"{name}: in one of the two only")
            continue
        first, second = _number(before[name]), _number(after[name])
        if first is None or second is None:
            report["problems"].append(f"{name}: not a finite number ({before[name]!r} -> {after[name]!r})")
            continue
        bound = shares / (n_scenarios if isinstance(n_scenarios, int) else n_scenarios[name])
        change = second - first
        report["bound"][name], report["changes"][name] = bound, change
        if not abs(change) <= bound + RATE_TOLERANCE:
            report["problems"].append(f"{name}: moved by {change:+.6f}, more than {whose} {bound:.6f}")
    return report


#: The draw views that hold one row per scenario, compared row for row.
PER_SCENARIO_VIEWS = (BUNDLE_OUTCOMES_FNAME, EVALUATIONS_FNAME, JUDGE_OUTPUTS_FNAME)
#: The sampling-manifest fields a rebase may change (with the run fields): the population, and the cap.
_REBASED_FIELDS = frozenset({*REPOINTED_MANIFEST_FIELDS, AMENDED_FIELD, MANIFEST_AMENDMENTS_KEY})
#: A draw file ending in a torn row (bytes after its last newline): a run killed mid-append, and its repair.
TORN_ROW = ("ends in a torn row (a run killed mid-append): resume the draw, whose runner drops it, then check "
            "again")


def _rebase_record(rebased_root: Path) -> dict[str, Any]:
    path = rebased_root / REBASE_FNAME
    if not path.is_file():
        raise RebaseCheckError(f"{rebased_root} holds no {REBASE_FNAME}: not a rebased root")
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        yield from handle


def _ends_torn(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(max(os.fstat(handle.fileno()).st_size - 1, 0))
        return handle.read(1) not in (b"\n", b"")


def _scenario_of(line: bytes) -> str:
    return str(json.loads(line)["scenario_id"])


def _compare_ledger(
    parent: Path, rebased: Path, removed: frozenset[str], added: frozenset[str], *,
    raised: frozenset[str] = frozenset(), base: int = 0,
) -> tuple[Optional[str], int, int]:
    """Compare a rebased ledger with its parent's; returns (problem or None, added rows, rows naming no stack)."""
    tokens = [sid.encode("ascii") for sid in removed]
    others = _lines(rebased)
    for number, line in enumerate(_lines(parent), start=1):
        if any(token in line for token in tokens) and _scenario_of(line) in removed:
            continue
        if next(others, None) != line:
            return f"the parent's line {number}, kept, is not the next line of the rebased file", 0, 0
    count = unnamed = 0
    for count, line in enumerate(others, start=1):
        if not line.endswith(b"\n"):
            return f"it {TORN_ROW}", 0, 0
        row = json.loads(line)
        scenario_id = str(row["scenario_id"])
        if scenario_id not in added and scenario_id not in raised:
            return f"row {count} after the parent's is not an added scenario's", 0, 0
        if scenario_id in raised and int(row["semantic_attempt_id"]) < base:
            return (f"row {count} after the parent's is a raised scenario's ({scenario_id[:12]}) at attempt "
                    f"{row['semantic_attempt_id']}, below the old cap {base}: no resume after the raise appends one"), 0, 0
        unnamed += JUDGED_ON_KEY not in row
    return None, count, unnamed


def _log_problems(draw: Path, *, required: bool = False) -> list[str]:
    damage = run_log_damage(draw)
    torn = damage and (f"{damage[0]}; resume the draw, whose runner drops it, then check again" if damage[1]
                       else damage[0])
    found = (run_log_problem(draw, required=required), torn, unlogged_stacks(draw))
    return [f"{draw.as_posix()}: {problem}" for problem in found if problem]


def _compare_blocks(parent: Path, rebased: Path, removed: frozenset[str], added: frozenset[str]) -> Optional[str]:
    others = iter(rebased.read_bytes().splitlines(keepends=True))
    for number, line in enumerate(parent.read_bytes().splitlines(keepends=True), start=1):
        row = json.loads(line)
        if removed & set(row["scenario_ids"]):
            line = (json.dumps({**row, "scenario_ids": [s for s in row["scenario_ids"] if s not in removed]})
                    + "\n").encode("utf-8")
        if next(others, None) != line:
            return f"the parent's block row {number} is not the next row of the rebased blocks"
    for number, line in enumerate(others, start=1):
        if not line.endswith(b"\n"):
            return f"it {TORN_ROW}"
        if not set(json.loads(line)["scenario_ids"]) <= added:
            return f"block row {number} after the parent's holds a scenario that is not an added one"
    return None


def _view_rows(path: Path) -> dict[str, bytes]:
    return {_scenario_of(line): line for line in path.read_bytes().splitlines(keepends=True)}


@contextmanager
def merged_shard_sources(shard_sources: Sequence[Path | str]) -> Iterator[dict[str, Path]]:
    """Each shard source's condition and the throwaway root it is merged into, as the rebase merged it."""
    with ExitStack() as stack:
        roots = {}
        for directory in shard_sources:
            holder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ruleforge-rebase-check-")))
            condition_id = str(discover_shards(directory)[0].record["condition_id"])
            merge_condition_shards(holder / "merged", condition_id, shards_dir=directory)
            roots[condition_id] = holder / "merged"
        yield roots


def _source_roots(parent_root: Path, rebased_root: Path, merged: Mapping[str, Path], report: dict[str, Any],
                  ) -> dict[str, Path]:
    expected = {condition_id: parent_root for condition_id in load_conditions(parent_root)}
    for condition_id, root in merged.items():
        if condition_id in expected:
            report["problems"].append(f"{condition_id} is registered in the parent and in a shard source")
        expected[condition_id] = root
    held = set(load_conditions(rebased_root))
    report["problems"] += [f"{rebased_root} does not register {condition_id}, which "
                           + ("the parent" if expected[condition_id] == parent_root else "a shard source") + " holds"
                           for condition_id in sorted(set(expected) - held)]
    report["problems"] += [f"{rebased_root} registers {condition_id}, which neither the parent nor a given shard "
                           "source holds" for condition_id in sorted(held - set(expected))]
    return {condition_id: root for condition_id, root in expected.items() if condition_id in held}


def _draw_pairs(roots: Mapping[str, Path], rebased_root: Path, n_draws: int) -> Iterator[tuple[str, Path, Path]]:
    """``(condition id/draw-NNN, the source draw, the rebased draw)`` of every draw of ``roots``' conditions."""
    for condition_id, source_root in sorted(roots.items()):
        for draw in range(n_draws):
            relative = f"{condition_id}/{draw_dirname(draw)}"
            yield relative, source_root / relative, rebased_root / relative


def _check_draw(
    source: Path, rebased: Path, removed: frozenset[str], added: frozenset[str], target: str, report: dict[str, Any],
    *, raised: frozenset[str] = frozenset(), base: int = 0,
) -> None:
    where = rebased.as_posix()
    resumed, unnamed = {}, {}
    gained = added | raised
    for name in (LEDGER_FNAME, RAW_OUTPUTS_FNAME):
        problem, resumed[name], unnamed[name] = _compare_ledger(source / name, rebased / name, removed, added,
                                                                raised=raised, base=base)
        if problem:
            report["problems"].append(f"{where}/{name}: {problem}")
    problem = _compare_blocks(source / REQUEST_BLOCKS_FNAME, rebased / REQUEST_BLOCKS_FNAME, removed, gained)
    if problem:
        report["problems"].append(f"{where}/{REQUEST_BLOCKS_FNAME}: {problem}")
    parent_runs, runs = (source / RUNS_FNAME).read_bytes(), (rebased / RUNS_FNAME).read_bytes()
    if not runs.startswith(parent_runs):
        report["problems"].append(f"{where}/{RUNS_FNAME}: the parent's run log is not its beginning")
    elif resumed[LEDGER_FNAME] and len(runs) == len(parent_runs):
        report["problems"].append(f"{where}/{RUNS_FNAME}: {resumed[LEDGER_FNAME]} added row(s) in its {LEDGER_FNAME}, "
                                  "and no run logged after the parent's")
    for path, required in ((source, False), (rebased, bool(resumed[LEDGER_FNAME]))):
        report["problems"] += _log_problems(path, required=required)
    provenance = json.loads((source / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
    if unnamed[LEDGER_FNAME] and provenance.get("model_type") == LOCAL_MODEL_TYPE:
        report["problems"].append(f"{where}/{LEDGER_FNAME}: {unnamed[LEDGER_FNAME]} added row(s) name no stack they "
                                  f"were judged on ({JUDGED_ON_KEY}): recorded by a runner older than this build's")
    rebased_provenance = json.loads((rebased / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
    if rebased_provenance.pop("scenario_manifest_sha256", None) != target:
        report["problems"].append(f"{where}/{DRAW_PROVENANCE_FNAME} does not name the target manifest")
    provenance.pop("scenario_manifest_sha256", None)
    for record in (provenance, rebased_provenance):
        record.pop("expected_scenarios", None)
    if rebased_provenance != provenance:
        report["problems"].append(f"{where}/{DRAW_PROVENANCE_FNAME} is not the parent's, re-pointed")
    for name in PER_SCENARIO_VIEWS:
        if not ((source / name).is_file() and (rebased / name).is_file()):
            continue
        before, after = _view_rows(source / name), _view_rows(rebased / name)
        differing = sorted(sid[:12] for sid in set(before) - removed - raised if after.get(sid) != before[sid])
        if differing:
            report["problems"].append(f"{where}/{name}: retained scenario(s) {differing[:5]} are not the parent's rows")
    if not (source / BUNDLE_OUTCOMES_FNAME).is_file():
        report["notes"].append(f"{where}: the parent draw is not materialised; its outcomes are compared through "
                               "its ledger")
    elif not (rebased / BUNDLE_OUTCOMES_FNAME).is_file():
        report["notes"].append(f"{where}: not materialised (not resumed yet, or exhausted); its views were not "
                               "compared, its ledger was")


def _difference(parent: Path, rebased: Path) -> tuple[list[str], list[str]]:
    parent_ids = ScenarioManifest.load(parent / SCENARIO_MANIFEST_FNAME).scenario_ids
    rebased_ids = ScenarioManifest.load(rebased / SCENARIO_MANIFEST_FNAME).scenario_ids
    kept, before = set(rebased_ids), set(parent_ids)
    return [sid for sid in parent_ids if sid not in kept], [sid for sid in rebased_ids if sid not in before]


def added_scenarios(parent_root: Path | str, rebased_root: Path | str) -> list[str]:
    """The scenarios ``rebased_root``'s population holds that ``parent_root``'s does not, in manifest order."""
    return _difference(Path(parent_root), Path(rebased_root))[1]


def _carried(roots: Mapping[str, Path], n_draws: int, removed: frozenset[str]) -> dict[str, list[str]]:
    """``{condition id/draw-NNN: retained scenarios exhausted in the parent's draw}``, as the rebase reads them."""
    carried = {}
    for condition_id, source_root in sorted(roots.items()):
        for draw in range(n_draws):
            relative = f"{condition_id}/{draw_dirname(draw)}"
            gate = json.loads((source_root / relative / COMPLETION_FNAME).read_text(encoding="utf-8"))
            exhausted = sorted(set(gate.get("exhausted") or ()) - removed)
            if exhausted:
                carried[relative] = exhausted
    return carried


def _check_record(
    parent: Path, rebased: Path, record: Mapping[str, Any], merged: Mapping[str, Path], n_conditions: int,
    difference: tuple[list[str], list[str]], carried: Mapping[str, list[str]], report: dict[str, Any], *,
    parent_manifest: Mapping[str, Any],
) -> None:
    removed, added = difference
    rebased_scenarios = ScenarioManifest.load(rebased / SCENARIO_MANIFEST_FNAME)
    n_draws = int(parent_manifest["expected_decoder_draws"])
    found = {
        "removed_ids": removed, "added_ids": added, "retained_count": len(rebased_scenarios.scenario_ids) - len(added),
        "n_conditions": n_conditions, "n_draws": n_conditions * n_draws,
        "shard sources' conditions": sorted(merged),
        "parent's sampling manifest": sha256_file(parent / SAMPLING_MANIFEST_FNAME),
        "target manifest": rebased_scenarios.manifest_sha256,
        "carried_exhausted": dict(carried),
    }
    recorded = {
        **{key: record.get(key) for key in ("removed_ids", "added_ids", "retained_count", "n_conditions", "n_draws")},
        "shard sources' conditions": sorted(str(source.get("condition_id")) for source in record.get("shard_sources")
                                            or ()),
        "parent's sampling manifest": (record.get("parent") or {}).get("sampling_manifest_sha256"),
        "target manifest": (record.get("target") or {}).get("scenario_manifest_sha256"),
        "carried_exhausted": record.get("carried_exhausted"),
    }
    schema = record.get("schema_version")
    if isinstance(schema, int) and schema >= CHAIN_SCHEMA_VERSION:
        parent_scenarios = ScenarioManifest.load(parent / SCENARIO_MANIFEST_FNAME)
        own = {
            "parent": {"root": (record.get("parent") or {}).get("root"),
                       "campaign_id": parent_manifest["campaign_id"],
                       "scenario_manifest_sha256": parent_scenarios.manifest_sha256,
                       "scenario_source_sha256": parent_scenarios.source_sha256,
                       "sampling_manifest_sha256": found["parent's sampling manifest"]},
            "target": {"scenario_manifest_sha256": rebased_scenarios.manifest_sha256,
                       "source_sha256": rebased_scenarios.source_sha256},
            "removed_ids": list(removed), "added_ids": list(added),
        }
        try:
            parent_record = rebase_record(parent)
            found[CHAIN_KEY] = [*([] if parent_record is None else rebase_chain(parent_record)), own]
        except (CampaignRunError, RebaseError, KeyError, TypeError) as error:
            report["problems"].append(f"{REBASE_FNAME}'s {CHAIN_KEY} cannot be checked: {error}")
        else:
            recorded[CHAIN_KEY] = record.get(CHAIN_KEY)
    else:
        report["notes"].append(f"{REBASE_FNAME} is schema {schema!r}, from before the chain: it records "
                               "one hop, its own")
    report["problems"] += [f"{REBASE_FNAME} records {name} {recorded[name]!r}; the roots hold {value!r}"
                           for name, value in found.items() if recorded[name] != value]


def _named_carried(amendment: Optional[CapAmendment], carried: Mapping[str, list[str]]) -> dict[str, set[str]]:
    named: dict[str, set[str]] = {}
    for bundle in () if amendment is None else amendment.bundles:
        relative = f"{bundle.condition_id}/{draw_dirname(bundle.decoder_draw_id)}"
        if bundle.scenario_id in carried.get(relative, ()):
            named.setdefault(relative, set()).add(bundle.scenario_id)
    return named


def _check_manifest(
    parent_manifest: Mapping[str, Any], manifest: Mapping[str, Any], record: Mapping[str, Any], target: str,
    added: frozenset[str], carried: Mapping[str, list[str]], report: dict[str, Any], *,
    amendment: Optional[CapAmendment], named_carried: Mapping[str, set[str]],
) -> int:
    """Check the rebased sampling manifest against the parent's; returns the parent's frozen cap."""
    problems = report["problems"]
    differing = sorted(set(differing_manifest_fields(parent_manifest, manifest)) - _REBASED_FIELDS)
    if differing:
        problems.append(f"the rebased sampling manifest differs from the parent's in {differing}")
    if manifest.get("scenario_manifest_sha256") != target:
        problems.append("the rebased sampling manifest does not name its own scenario manifest")
    parent_amendment, reversion = cap_amendment(parent_manifest), record.get("cap_reversion")
    if parent_amendment is None:
        base = parent_manifest[AMENDED_FIELD]
        if reversion is not None:
            problems.append(f"{REBASE_FNAME} records a cap_reversion, but the parent records no cap amendment")
    else:
        base = parent_amendment.old_cap
        reverted = {"field": AMENDED_FIELD, "from": parent_manifest[AMENDED_FIELD], "to": base,
                    "amendments": parent_manifest[MANIFEST_AMENDMENTS_KEY]}
        if reversion != reverted:
            caps = None if reversion is None else {key: reversion.get(key) for key in ("from", "to")}
            problems.append(f"{REBASE_FNAME}'s cap_reversion {caps} is not the parent's amendment: from "
                            f"{parent_manifest[AMENDED_FIELD]} back to its old cap {base}, its record verbatim")
    expected = base if amendment is None else (amendment.new_cap if amendment.old_cap == base else None)
    if manifest[AMENDED_FIELD] != expected:
        problems.append(
            f"the rebased root records {AMENDED_FIELD} {manifest[AMENDED_FIELD]}: neither the parent's frozen cap "
            f"{base} nor a recorded amendment's raised cap from it")
    if amendment is not None:
        def carried_in(condition_id: str, draw: int) -> Sequence[str]:
            return carried.get(f"{condition_id}/{draw_dirname(draw)}", ())

        foreign = sorted(bundle.describe() for bundle in amendment.bundles if bundle.scenario_id not in added
                         and bundle.scenario_id not in named_carried.get(
                             f"{bundle.condition_id}/{draw_dirname(bundle.decoder_draw_id)}", ()))
        if foreign:
            problems.append(f"the rebased root's cap amendment names bundle(s) of scenarios the rebase neither added "
                            f"nor carried over exhausted ({foreign[:3]}): only an added scenario's cap, or a "
                            "carried-over bundle's named for the raise, is raised after a rebase")
        unheld = sorted(f"{condition_id}/{draw_dirname(draw)} {sid[:12]}" for condition_id, draw, sid in amendment.held
                        if sid not in carried_in(condition_id, draw))
        if unheld:
            problems.append(f"the rebased root's cap amendment holds bundle(s) {unheld[:3]} at the old cap that the "
                            "parent's draws do not hold exhausted: only a carried-over bundle is held")
        kept = {(condition_id, draw_dirname(draw), sid) for condition_id, draw, sid in amendment.held}
        named = {(bundle.condition_id, draw_dirname(bundle.decoder_draw_id), bundle.scenario_id)
                 for bundle in amendment.bundles}
        both = sorted(f"{condition_id}/{draw} {sid[:12]}" for condition_id, draw, sid in kept & named)
        if both:
            problems.append(f"the rebased root's cap amendment both holds and names bundle(s) {both[:3]}")
        unkept = sorted(f"{relative} {sid[:12]}" for relative, sids in carried.items() for sid in sids
                        if (*relative.rsplit("/", 1), sid) not in kept | named)
        if unkept:
            problems.append(f"the rebased root's cap amendment neither holds at the old cap nor names bundle(s) "
                            f"{unkept[:3]} that the parent's draws hold exhausted: a resume would judge a carried-over "
                            "bundle again, unrecorded")
    return base


def check_retained(
    parent_root: Path | str, rebased_root: Path | str, *, merged: Optional[Mapping[str, Path]] = None,
) -> dict[str, Any]:
    """Check that every retained judgment of the rebased root is the parent's."""
    parent, rebased, merged = Path(parent_root), Path(rebased_root), dict(merged or {})
    record = _rebase_record(rebased)
    parent_manifest = load_sampling_manifest(parent / SAMPLING_MANIFEST_FNAME)
    manifest = load_sampling_manifest(rebased / SAMPLING_MANIFEST_FNAME)
    report = _report()
    roots = _source_roots(parent, rebased, merged, report)
    n_conditions = len(load_conditions(parent)) + len(merged)
    n_draws = int(parent_manifest["expected_decoder_draws"])
    difference = _difference(parent, rebased)
    removed, added = (frozenset(ids) for ids in difference)
    carried = _carried(roots, n_draws, removed)
    _check_record(parent, rebased, record, merged, n_conditions, difference, carried, report,
                  parent_manifest=parent_manifest)
    target = ScenarioManifest.load(rebased / SCENARIO_MANIFEST_FNAME).manifest_sha256
    amendment = cap_amendment(manifest)
    raised = _named_carried(amendment, carried)
    base = _check_manifest(parent_manifest, manifest, record, target, added, carried, report, amendment=amendment,
                           named_carried=raised)
    for condition_id, source_root in sorted(roots.items()):
        if (source_root / condition_id / CONDITION_FNAME).read_bytes() != (
                rebased / condition_id / CONDITION_FNAME).read_bytes():
            report["problems"].append(f"{rebased / condition_id / CONDITION_FNAME} is not the parent's")
    for relative, source, draw in _draw_pairs(roots, rebased, n_draws):
        _check_draw(source, draw, removed, added, target, report, raised=frozenset(raised.get(relative, ())),
                    base=base)
    report["conditions"], report["draws"] = len(roots), len(roots) * n_draws
    return report


def estimate_changes(before_path: Path | str, after_path: Path | str, *, swaps: int = 1) -> dict[str, Any]:
    """:func:`rate_changes` over two analyses' estimates (``aggregate/estimates.json``)."""
    before, after = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (before_path, after_path))
    rates = [{cid: row.get("estimate") for cid, row in payload["conditions"].items()} for payload in (before, after)]
    counts = {
        cid: min(int(payload["conditions"][cid].get("n_scenarios", payload["n_scenarios"]))
                 for payload in (before, after))
        for cid in set(rates[0]) & set(rates[1])
    }
    return rate_changes(rates[0], rates[1], n_scenarios=counts, swaps=swaps)


def check_added(
    parent_root: Path | str, rebased_root: Path | str, *, population: Path | str, rule_sets: Sequence[Path | str],
) -> dict[str, Any]:
    """Check that every added scenario's attempt carries the seeds and prompts its parent's campaign gives it."""
    parent, rebased = Path(parent_root), Path(rebased_root)
    _rebase_record(rebased)
    added = set(_difference(parent, rebased)[1])  # from the two populations, never the record's own list
    parent_manifest = load_sampling_manifest(parent / SAMPLING_MANIFEST_FNAME)
    manifest = load_sampling_manifest(rebased / SAMPLING_MANIFEST_FNAME)
    if ScenarioManifest.build(population).manifest_sha256 != manifest["scenario_manifest_sha256"]:
        raise RebaseCheckError(f"{population} is not the population {rebased} froze")
    rows = {sid: row for row in iter_jsonl_records(population) if (sid := compute_scenario_id(row)) in added}
    rules = rule_sets_by_digest(rule_sets)
    # The parent's campaign, never the rebased root's record of it: the seeds R must carry are the parent's.
    campaign_id, seed_base = str(parent_manifest["campaign_id"]), int(parent_manifest["generation_seed_base"])
    cap = int(manifest[AMENDED_FIELD])
    report = _report()
    report["outcomes"] = {}
    seeds = prompts = 0
    for condition_id, spec in sorted(load_conditions(rebased).items()):
        if spec.rule_set_sha256 not in rules:
            raise RebaseCheckError(f"no --rule-set has the digest of {condition_id}'s rule set ({spec.rule_set_name})")
        render = prompt_renderer(spec, list(iter_jsonl_records(rules[spec.rule_set_sha256])))
        for draw in range(int(parent_manifest["expected_decoder_draws"])):
            directory = rebased / condition_id / draw_dirname(draw)
            where = f"{condition_id} {draw_dirname(draw)}"
            if _ends_torn(directory / LEDGER_FNAME):
                report["problems"].append(f"{where}: its {LEDGER_FNAME} {TORN_ROW}")
                continue
            # The ledger refuses a row naming another campaign, condition or draw than the parent's.
            state = DrawLedger(directory, campaign_id=campaign_id, condition_id=condition_id,
                               decoder_draw_id=draw).load(repair=False)
            records = [item for item in state.records if item.scenario_id in added]
            try:
                check_seeds(records, generation_seed_base=seed_base, where=where, identity_id=spec.identity_id)
            except AmendmentError as error:
                report["problems"].append(str(error))
            seeds += sum(1 for item in records for component in item.components if component.seed_sent)
            for item in records:
                for component in item.components:
                    prompts += 1
                    if prompt_sha256(render(rows[item.scenario_id], component.component_id)) != component.prompt_sha256:
                        report["problems"].append(f"{where}: {item.scenario_id[:12]}/{component.component_id} at "
                                                  f"attempt {item.semantic_attempt_id} was sent another prompt than "
                                                  "today's renderer builds")
            for sid in sorted(added):
                outcome = (f"accepted at attempt {state.accepted[sid].semantic_attempt_id}" if state.is_accepted(sid)
                           else "exhausted" if state.next_attempt(sid) >= cap else "pending")
                report["outcomes"].setdefault(sid, {})[where] = outcome
    report["seeds_checked"], report["prompts_checked"] = seeds, prompts
    return report


def _resume_rows(source: Path, rebased: Path) -> list[dict[str, Any]]:
    tail = (rebased / RUNS_FNAME).read_bytes()[len((source / RUNS_FNAME).read_bytes()):]
    return [json.loads(line) for line in tail.split(b"\n") if line.strip()]


def _added_stacks(source: Path, rebased: Path, removed: frozenset[str]) -> tuple[int, Counter]:
    tokens = [sid.encode("ascii") for sid in removed]
    kept = sum(1 for line in _lines(source / LEDGER_FNAME)
               if not (any(token in line for token in tokens) and _scenario_of(line) in removed))
    path = rebased / LEDGER_FNAME
    with path.open("rb") as handle:
        lines = [line for _, line in zip(range(count_complete_jsonl_records(path)), handle)][kept:]
    return len(lines), Counter(json.loads(line).get(JUDGED_ON_KEY) for line in lines)


def _resume_stacks(
    relative: str, source: Path, draw: Path, reference: Mapping[str, Any], removed: frozenset[str],
    report: dict[str, Any],
) -> None:
    gained, stacks = _added_stacks(source, draw, removed)
    unbound = run_log_problem(draw, required=gained > 0)
    if unbound:
        report["problems"].append(f"{relative}: {unbound}")
    unnamed = stacks.get(None, 0)
    if unnamed:
        report["problems"].append(f"{relative}: {unnamed} added row(s) of its {LEDGER_FNAME} name no stack they "
                                  f"were judged on ({JUDGED_ON_KEY}): their stack cannot be told")
    rows = _resume_rows(source, draw)
    changes = [row.get("runtime") or {} for row in rows if row.get(STACK_CHANGE_KEY)]
    for row in rows:
        runtime = row.get("runtime") or {}
        if STACK_CHANGE_KEY in row or not runtime_changes(reference, runtime) or not all(
                runtime_changes(change, runtime) for change in changes):
            continue
        report["notes"].append(f"{relative}: a resume run logged another runtime and no stack change, and judged "
                               "no row (the runner lets one through only when it judges nothing)")


def check_stacks(
    parent_root: Path | str, rebased_root: Path | str, *, merged: Optional[Mapping[str, Path]] = None,
) -> dict[str, Any]:
    """Check that each rebased draw's judgments come from its parent's stack, or from a logged stack change."""
    parent, rebased = Path(parent_root), Path(rebased_root)
    n_draws = int(load_sampling_manifest(parent / SAMPLING_MANIFEST_FNAME)["expected_decoder_draws"])
    removed = frozenset(_difference(parent, rebased)[0])
    report = _report()
    report["stack_changes"] = {}
    providers, compared = 0, 0
    for relative, source, draw in _draw_pairs(_source_roots(parent, rebased, dict(merged or {}), report), rebased,
                                              n_draws):
        unread = _log_problems(source) + _log_problems(draw)
        if unread:
            report["problems"] += unread
            continue
        before, after = judgment_runtimes(source), judgment_runtimes(draw)
        if after[: len(before)] != before:
            report["problems"].append(f"{relative}: its judgments' runtimes do not begin with the parent's")
            continue
        if json.loads((source / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8")).get("model_type") != (
                LOCAL_MODEL_TYPE):
            providers += 1  # a provider judge is held to no stack (the runner's guard): nothing to compare
            continue
        compared += 1
        if len(after) > len(before):
            report["stack_changes"][relative] = [
                describe_changes(stack_changes(before[0], runtime)) for runtime in after[len(before):]]
        _resume_stacks(relative, source, draw, recorded_runtime(source), removed, report)
    if providers:
        report["notes"].append(f"{providers} draw(s) of a provider judge: no stack to hold them to (not applicable)")
    if compared and not report["stack_changes"] and not report["problems"]:
        report["notes"].append(f"every one of the {compared} local judge's draw(s) judged on its parent's stack")
    return report
