from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ruleforge.core import record_names
from ruleforge.core.errors import ArtifactRootError, SamplingDesignError
from ruleforge.core.paths import resolve
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME
from ruleforge.evaluate.amendment import cap_amendment
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME, load_conditions
from ruleforge.evaluate.estimates import CampaignEstimatesError, campaign_settings_differences, load_estimates
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_FINAL,
    MANIFEST_AMENDMENTS_KEY,
    SAMPLING_MANIFEST_FNAME,
    ConditionSpec,
    load_sampling_manifest,
)
from ruleforge.evaluate.inference import DrawGrid, iter_jsonl, load_condition_outcomes, outcome_grid
from ruleforge.evaluate.ledger import attempt_stamps
from ruleforge.evaluate.runner import REQUEST_BLOCKS_FNAME
from ruleforge.evaluate.stack import (
    LOCAL_MODEL_TYPE,
    STACK_PACKAGES,
    condition_signatures,
    describe_signatures,
    draw_dirs,
    signatures_mixed,
)
from ruleforge.experiments.checks import require
from ruleforge.scenarios.manifest import ScenarioManifest

#: A provider judge's stack: none this process can record.
PROVIDER = "provider"
SHORT_STACK_PACKAGES = (("vllm", "vLLM"), ("torch", "torch"), ("transformers", "transformers"))


class CampaignRootError(CampaignEstimatesError):
    """A campaign root cannot give the arms an experiment reads from it."""


_require = partial(require, error=CampaignRootError)


@dataclass
class Root:
    """One campaign root: its estimates, sampling manifest, registered conditions and scenarios."""

    key: str
    uri: str
    path: Path
    payload: Dict[str, Any]
    manifest: Dict[str, Any]
    conditions: Dict[str, ConditionSpec]
    scenario_ids: Tuple[str, ...]


#: How a reader's refusal ends where a root's estimates disagree with what it reads beside them; readers take no lock,
#: so a write during a read shows up here.
CHANGED_WHILE_READ = ("the campaign changed while it was read; rerun this once no shard merge, cap amendment record "
                      "or analysis of it is running")


def open_roots(
    uris: Mapping[str, str], where: str, *, allow_incomplete: bool, allow_stack_mismatch: bool,
) -> Dict[str, Root]:
    """Return each root (``{key: artifact:// URI}``) with its estimates, manifests, conditions and scenarios, refused
    unless the roots are one experiment of final campaigns; read-only and lock-free."""
    try:
        paths = {key: resolve(uri) for key, uri in uris.items()}
    except (ArtifactRootError, ValueError) as error:
        raise CampaignRootError(f"{where}: a root does not resolve: {error}") from error
    _require([(path.is_dir(), f"root {key} ({uris[key]}) is not in the artifact root: fetch it (ruleforge data fetch)")
              for key, path in paths.items()], where)
    estimates = {key: path / AGGREGATE_DIRNAME / ESTIMATES_FNAME for key, path in paths.items()}
    _require([(path.is_file(), f"root {key} ({uris[key]}) has no {AGGREGATE_DIRNAME}/{ESTIMATES_FNAME}: "
                               "analyse it first (ruleforge analyze ROOT)")
              for key, path in estimates.items()], where)
    payloads = load_estimates(
        list(estimates.values()),
        allow_incomplete=allow_incomplete, allow_stack_mismatch=allow_stack_mismatch,
    )
    roots = {}
    for (key, path), payload in zip(paths.items(), payloads):
        scenarios = ScenarioManifest.load(path / SCENARIO_MANIFEST_FNAME)
        manifest = load_sampling_manifest(path / SAMPLING_MANIFEST_FNAME)
        _require(
            [
                (scenarios.manifest_sha256 == payload["scenario_manifest_sha256"],
                 "its scenario manifest is not the one its estimates name"),
                (manifest.get("campaign_id") == payload["campaign_id"],
                 "its sampling manifest names another campaign than its estimates"),
                (allow_incomplete or manifest.get("campaign_mode") == CAMPAIGN_MODE_FINAL,
                 f"campaign_mode {manifest.get('campaign_mode')!r}; the paper's goldens need a final campaign (pass "
                 "--allow-incomplete for a scratch build)"),
                (_analysed_cap(payload) == _current_cap(manifest),
                 f"its estimates were analysed at max_semantic_attempts {payload.get('max_semantic_attempts')!r}, "
                 f"its sampling manifest records {manifest.get('max_semantic_attempts')!r} now, or the two record "
                 "different cap amendments: estimates older than the manifest (re-run analyze_campaign "
                 f"on it), or {CHANGED_WHILE_READ}"),
            ],
            f"root {key}",
        )
        # The experiments read each root's recorded estimates and never analyse it again.
        roots[key] = Root(key, uris[key], path, payload, manifest, load_conditions(path, skip_retired=True),
                          scenarios.scenario_ids)
    return roots


def _current_cap(manifest: Mapping[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
    amendment = cap_amendment(manifest)
    return manifest.get("max_semantic_attempts"), None if amendment is None else amendment.summary()


def _analysed_cap(payload: Mapping[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]]]:
    disclosed = payload.get(MANIFEST_AMENDMENTS_KEY) or [None]
    record = None if disclosed[0] is None else {
        key: value for key, value in disclosed[0].items() if key not in ("bundles", "code_states")}
    return payload.get("max_semantic_attempts"), record


def registered_condition(root: Root, condition_id: str, where: str) -> ConditionSpec:
    """Return the registered spec of the condition ``condition_id`` names in ``root``, refused unless the root registers
    exactly one and its estimates analysed it."""
    try:
        spec = record_names.locate_condition(root.conditions, condition_id)
    except SamplingDesignError as error:
        raise CampaignRootError(f"{where}: root {root.key}: {error}") from error
    _require(
        [(spec.condition_id in root.payload["conditions"],
          f"{condition_id} is not analysed in root {root.key}'s estimates (it analysed "
          f"{sorted(public_id_of(root, cid) for cid in root.payload['conditions'])})")],
        where,
    )
    return spec


def public_id_of(root: Root, condition_id: str) -> str:
    """Return the public id of a condition ``root``'s estimates analyse."""
    spec = root.conditions.get(condition_id)
    if spec is not None:
        return spec.public_id
    return record_names.public_condition_id(condition_id, root.payload["conditions"][condition_id]["prompt_style"])


def check_campaign_settings(roots: Sequence[Root], where: str) -> None:
    """Check that the roots a judge's arms come from share the campaign settings that shape an arm's outcomes, each
    root's attempts at the cap its campaign froze."""
    differing = campaign_settings_differences({root.key: root.manifest for root in roots})
    _require([(not differing, f"its arms' roots differ in {'; '.join(differing)}: a contrast across them would be "
                              "the campaigns' as well as the prompts'")], where)


def short_stack(signatures: Sequence[Mapping[str, Optional[str]]], extra: Sequence[str] = ()) -> str:
    """Return ``vLLM 0.19.0, torch 2.10.0``-style text: the first recorded version of each short-stack package, then of
    each ``extra`` field."""
    parts = []
    shown_names = dict(SHORT_STACK_PACKAGES)
    for name, shown in (*SHORT_STACK_PACKAGES, *((name, name) for name in extra if name not in shown_names)):
        version = next((signature.get(name) for signature in signatures if signature.get(name)), None)
        if version is not None:
            parts.append(f"{shown} {version}")
    return ", ".join(parts) or "no stack recorded"


def _request_dates(condition_dir: Path) -> Dict[str, Optional[str]]:
    """When a condition's first and last requests went out, over its request blocks' ``recorded_at`` and its attempts'
    ``started_at`` (a rebased draw's blocks miss some attempts); None when none records one."""
    stamps = sorted(
        [str(row["recorded_at"]) for draw in draw_dirs(condition_dir) if (draw / REQUEST_BLOCKS_FNAME).is_file()
         for row in iter_jsonl(draw / REQUEST_BLOCKS_FNAME) if row.get("recorded_at")]
        + [stamp for draw in draw_dirs(condition_dir) for stamp in attempt_stamps(draw)]
    )
    return {"first_request_block": stamps[0] if stamps else None, "last_request_block": stamps[-1] if stamps else None}


def arm_stack(root: Root, spec: ConditionSpec, where: str) -> Dict[str, Any]:
    """What an arm's judgments ran on: a local judge's recorded stacks (refused when they mix), or ``provider``."""
    condition_dir = root.path / spec.condition_id
    if spec.model_type != LOCAL_MODEL_TYPE:
        return {"kind": PROVIDER, "model_type": spec.model_type, "signatures": [],
                "description": f"{PROVIDER} ({spec.model_type}): no inference stack recorded",
                "short": PROVIDER, "dates": _request_dates(condition_dir)}
    signatures = condition_signatures(condition_dir)
    _require([(bool(signatures) and all(any(signature.get(name) for name in STACK_PACKAGES) for signature in signatures),
               f"{spec.condition_id}'s draws record no inference stack (no {list(STACK_PACKAGES)} version in their "
               "runtime): a local arm's stack is what its contrasts are held to")], where)
    _require([(not signatures_mixed(signatures),
               f"{spec.condition_id}'s own judgments come from several stacks or engine configurations "
               f"({describe_signatures(signatures)}); no contrast pairs it")], where)
    return {"kind": LOCAL_MODEL_TYPE, "model_type": spec.model_type, "signatures": [dict(s) for s in signatures],
            "description": describe_signatures(signatures), "short": short_stack(signatures)}


def arm_outcomes(
    root: Root, spec: ConditionSpec, n_draws: int, where: str
) -> Tuple[Dict[Tuple[str, int], Dict[str, Any]], DrawGrid]:
    """Return the arm's bundle outcomes (``{(scenario_id, draw): row}``) and its flat grid, refused unless the grid is
    its estimates' own."""
    summary = root.payload["conditions"][spec.condition_id]
    ids = root.scenario_ids
    cells, provenance = load_condition_outcomes(
        root.path / spec.condition_id, expected_scenario_ids=ids, expected_draws=n_draws,
        scenario_manifest_sha256=root.payload["scenario_manifest_sha256"],
    )
    grid = outcome_grid(spec.condition_id, cells, scenario_ids=ids, n_draws=n_draws, identity_id=spec.identity_id)
    _require(
        [
            (provenance["condition_fingerprint"] == spec.fingerprint == summary["condition_fingerprint"],
             f"{spec.condition_id}: its draws, its condition.json and its estimates name different fingerprints"),
            (grid.digest == summary["grid_digest"],
             f"{spec.condition_id}: its grid rebuilt from the draws is not its estimates' own (digest)"),
        ],
        where,
    )
    return cells, grid
