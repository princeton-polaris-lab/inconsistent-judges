from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
import json
from typing import Any, Mapping, Optional

import numpy as np

from ruleforge.evaluate.estimates import frozen_settings, public_amendments
from ruleforge.evaluate.inference import (
    DrawGrid,
    INFERENCE_METHOD,
    PRIMARY_INTERVAL_BASIS,
    TEST_METHOD,
    holm_adjust,
    load_condition_outcomes,
    outcome_grid,
)
from ruleforge.core.paths import to_uri
from ruleforge.evaluate.aggregate import missing_rulesets, ruleset_families
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME, load_conditions
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_FINAL,
    SAMPLING_MANIFEST_FNAME,
    ConditionSpec,
    load_sampling_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.stack import STACK_MISMATCHES_KEY, recorded_mismatch_refusal, recorded_mismatches

DATA_VERSION = 6
TRANSITION_METHOD = "independent_draw_expected_shares_v1"


def expected_transition_shares(before: DrawGrid, after: DrawGrid) -> dict[str, float]:
    """Return the mean of all cross-condition draw pairs, paired only by scenario: four proportions summing to one, with
    ``regressed - fixed == after.rate - before.rate``."""
    if set(before.scenario_ids) != set(after.scenario_ids):
        raise ValueError("transition grids must cover identical scenarios")
    after = after.restricted_to(before.scenario_ids)
    for grid in (before, after):
        if grid.n_draws < 2 or not np.isin(grid.values, (0, 1)).all():
            raise ValueError("expected transitions require repeated binary outcomes")
    p, q = before.scenario_means, after.scenario_means
    return {
        "both_consistent": float(np.mean((1 - p) * (1 - q))),
        "fixed": float(np.mean(p * (1 - q))),
        "regressed": float(np.mean((1 - p) * q)),
        "both_inconsistent": float(np.mean(p * q)),
    }


def _interval(row: Mapping, *, reverse: bool = False) -> dict:
    values = np.asarray([row["estimate"], row["ci_low"], row["ci_high"]], dtype=float)
    if not np.isfinite(values).all() or values[1] > values[2]:
        raise ValueError("invalid estimate or interval")
    if row.get("interval_basis") != PRIMARY_INTERVAL_BASIS:
        raise ValueError("unsupported interval basis")
    mean, low, high = values
    return {
        "mean": float(-mean if reverse else mean),
        "ci_low": float(-high if reverse else low),
        "ci_high": float(-low if reverse else high),
    }


def build_summary(
    estimates: Mapping, grids: Mapping[str, DrawGrid], labels: Mapping[str, str],
    specs: Optional[Mapping[str, ConditionSpec]] = None,
) -> dict:
    """Return one ruleset family's summary; mapping order sets plot order, and ``specs`` (each condition's spec by id)
    gives the golden its public names."""
    if estimates.get("state") != "complete" or estimates.get("excluded_conditions"):
        raise ValueError("a complete campaign without excluded conditions is required")
    if estimates.get("inference_method") != INFERENCE_METHOD:
        raise ValueError("unsupported inference method")
    conditions = estimates["conditions"]
    if (
        len(labels) < 2
        or set(labels) != set(conditions)
        or set(grids) != set(conditions)
        or len(set(labels.values())) != len(labels)
        or not all(labels.values())
    ):
        raise ValueError("labels and grids must cover every condition exactly once")
    keys = [f"{after} - {before}" for before, after in combinations(labels.values(), 2)]
    if len(set(keys)) != len(keys):
        raise ValueError("display labels produce ambiguous contrast keys")
    families = list(estimates["ruleset_families"].values())
    if (
        len(families) != 1
        or set(families[0]["members"]) != set(conditions)
        or families[0].get("missing_rulesets")
    ):
        raise ValueError("exactly one complete ruleset family is required")
    n, draws = estimates["n_scenarios"], estimates["n_decoder_draws"]
    reference_ids = next(iter(grids.values())).scenario_ids
    rates = {}
    for cid, label in labels.items():
        row, grid = conditions[cid], grids[cid]
        if (
            grid.scenario_ids != reference_ids
            or grid.n_scenarios != n
            or grid.n_draws != draws
            or draws < 2
            or not np.isin(grid.values, (0, 1)).all()
            or row["n_scenarios"] != n
            or row["n_decoder_draws"] != draws
            or row["grid_digest"] != grid.digest
            or not np.isclose(row["estimate"], grid.rate, atol=1e-12, rtol=0)
        ):
            raise ValueError(
                f"{cid}: aggregate does not match the complete binary grid"
            )
        spec = (specs or {}).get(cid)
        named = {} if spec is None else {"prompt_style": spec.prompt_style}
        rates[label] = {**row, **named, **_interval(row), "condition_id": cid if spec is None else spec.public_id}
        rates[label].pop("estimate")
    contrasts = families[0]["contrasts"]
    expected_pairs = {frozenset(pair) for pair in combinations(labels, 2)}
    actual_pairs = [frozenset((row["a"], row["b"])) for row in contrasts]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("missing or duplicate ruleset contrasts")
    for row in contrasts:
        if not np.isfinite(row["p_value"]) or not 0 <= row["p_value"] <= 1:
            raise ValueError("invalid contrast p-value")
    adjusted = holm_adjust(contrasts)
    for row, corrected in zip(contrasts, adjusted, strict=True):
        a, b = row["a"], row["b"]
        if (
            row["family_size"] != len(contrasts)
            or row.get("test_method") != TEST_METHOD
            or row.get("inference_method") != INFERENCE_METHOD
            or row["n_scenarios"] != n
            or row["n_decoder_draws_a"] != draws
            or row["n_decoder_draws_b"] != draws
            or row["grid_digest_a"] != grids[a].digest
            or row["grid_digest_b"] != grids[b].digest
            or not np.isclose(
                row["estimate"], grids[a].rate - grids[b].rate, atol=1e-12, rtol=0
            )
            or not np.isclose(row["p_holm"], corrected["p_holm"], atol=1e-12, rtol=0)
        ):
            raise ValueError("contrast identity or Holm family disagrees with campaign")
    deltas, transitions = {}, {}
    lookup = {frozenset((row["a"], row["b"])): row for row in contrasts}
    for before, after in combinations(labels, 2):
        key = f"{labels[after]} - {labels[before]}"
        row = lookup[frozenset((before, after))]
        deltas[key] = {
            **_interval(row, reverse=row["a"] == before),
            **{
                k: row[k]
                for k in (
                    "p_value",
                    "p_holm",
                    "test_method",
                    "family_size",
                    "interval_basis",
                )
            },
        }
        transitions[key] = expected_transition_shares(grids[before], grids[after])
    return {
        "data_version": DATA_VERSION,
        "source": "stochastic_campaign",
        "num_paired_scenarios": n,
        "num_total_scenarios": n,
        "n_decoder_draws": draws,
        "dropped_scenarios": {},
        "labels": list(labels.values()),
        "inference_method": estimates["inference_method"],
        "num_bootstrap": estimates["bootstrap"]["n_resamples"],
        "confidence_level": estimates["bootstrap"]["ci_level"],
        "random_state": estimates["bootstrap"]["seed"],
        "randomization_test": estimates["randomization_test"],
        "transition_method": TRANSITION_METHOD,
        "transition_units": "share",
        "rulesets": rates,
        "paired_deltas": deltas,
        "paired_transitions": transitions,
    }


def require_materialized(provenance: Mapping[str, Any], cid: str, *, fingerprint: str, campaign_id: str,
                         accepted: int) -> None:
    """Refuse a condition's draws unless each completion record holds every scenario accepted and materialized under
    this campaign and condition, at this fingerprint."""
    if provenance["condition_fingerprint"] != fingerprint:
        raise ValueError(f"{cid}: draw fingerprint disagrees with condition")
    for completion in provenance["draws"]:
        if (
            not completion.get("materialized")
            or completion.get("campaign_id") != campaign_id
            or completion.get("condition_id") != cid
            or completion.get("accepted") != accepted
        ):
            raise ValueError(
                f"{cid}: incomplete materialization or wrong draw identity"
            )


def build_from_campaign(
    estimates_path: Path, ruleset_labels: Mapping[str, str], *, allow_stack_mismatch: bool = False
) -> dict:
    """Load and verify the stored aggregate against every materialized draw, read-only; stack mismatches are refused
    unless ``allow_stack_mismatch``."""
    estimates_path = Path(estimates_path)
    root = estimates_path.parent.parent
    data = estimates_path.read_bytes()  # read once: the digest recorded is of the estimates read
    estimates = json.loads(data.decode("utf-8"))
    refusal = recorded_mismatch_refusal(estimates, str(estimates_path))
    if refusal is not None and not allow_stack_mismatch:
        raise ValueError(refusal)
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    scenarios = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME)
    conditions = load_conditions(root)
    if manifest["campaign_mode"] != CAMPAIGN_MODE_FINAL:
        raise ValueError("paper figures require a final campaign")
    for key, value in {
        "campaign_id": manifest["campaign_id"],
        "scenario_manifest_sha256": scenarios.manifest_sha256,
        "n_scenarios": scenarios.record_count,
        "n_decoder_draws": manifest["expected_decoder_draws"],
        **{
            k: manifest[k]
            for k in (
                "temperature",
                "truncation",
                "max_semantic_attempts",
                "forced_json_requirement",
            )
        },
    }.items():
        if estimates.get(key) != value:
            raise ValueError(f"aggregate {key} disagrees with campaign manifest")
    if (
        scenarios.record_count != manifest["expected_scenarios"]
        or scenarios.manifest_sha256 != manifest["scenario_manifest_sha256"]
        or set(conditions) != set(estimates["conditions"])
    ):
        raise ValueError("registered population or conditions disagree with aggregate")
    families = ruleset_families(conditions)
    if len(families) != 1 or set(next(iter(families.values()))) != set(conditions):
        raise ValueError("conditions must form one ruleset-only family")
    if missing_rulesets(conditions, list(conditions)):
        raise ValueError("declared ruleset family is incomplete")
    by_tag = {dict(spec.extra).get("ruleset"): cid for cid, spec in conditions.items()}
    if len(by_tag) != len(conditions) or set(by_tag) != set(ruleset_labels):
        raise ValueError("ruleset labels must cover registered rulesets exactly")
    labels = {by_tag[tag]: label for tag, label in ruleset_labels.items()}
    grids = {}
    for cid, spec in conditions.items():
        row = estimates["conditions"][cid]
        if row["condition_fingerprint"] != spec.fingerprint:
            raise ValueError(f"{cid}: stale condition fingerprint")
        cells, provenance = load_condition_outcomes(
            root / cid,
            expected_scenario_ids=scenarios.scenario_ids,
            expected_draws=manifest["expected_decoder_draws"],
            scenario_manifest_sha256=scenarios.manifest_sha256,
        )
        require_materialized(provenance, cid, fingerprint=spec.fingerprint, campaign_id=manifest["campaign_id"],
                             accepted=scenarios.record_count)
        grids[cid] = outcome_grid(
            cid,
            cells,
            scenario_ids=scenarios.scenario_ids,
            n_draws=manifest["expected_decoder_draws"],
            identity_id=spec.identity_id,
        )
    result = build_summary(estimates, grids, labels, conditions)
    first = next(iter(conditions.values()))
    result["provenance"] = {
        "complete": True,
        "campaign_root": to_uri(root),
        "estimates": {
            "uri": to_uri(estimates_path),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "scenario_manifest_sha256": scenarios.manifest_sha256,
        "scenario_source_sha256": scenarios.source_sha256,
        "scenario_set": scenarios.source_name,
        "temperature": manifest["temperature"],
        "truncation": manifest["truncation"],
        # The cap its campaign froze; an amended root's raise is disclosed beside it (provenance.amendments).
        "max_semantic_attempts": frozen_settings(manifest)["max_semantic_attempts"],
        "judge": {
            "model_name": first.model_name,
            "judge_type": first.judge_type,
            "individual_rule_context": first.individual_rule_context,
            "evaluation_mode": first.evaluation_mode,
        },
    }
    mismatches = recorded_mismatches(estimates)
    if mismatches:
        result["provenance"][STACK_MISMATCHES_KEY] = mismatches
    result["provenance"].update(public_amendments(estimates))
    return result
