from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ruleforge.evaluate.inference import (
    BOOTSTRAP_TEST_METHOD,
    INFERENCE_METHOD,
    TEST_METHOD,
    DrawGrid,
    RepeatedSamplingError,
    clustered_paired_randomization_statistic,
    iter_jsonl,
    outcome_grid,
    percentile_bootstrap_p,
    scenario_cluster_bootstrap,
    statistic_interval,
)
from ruleforge.evaluate.scoring import PHI_CLASSES, PHI_MAP, calculate_inconsistency
from ruleforge.evaluate.identity import (
    FORCED_JSON_COMPLETE_RULE_MAP,
    FULL_RULESET_COMPONENT,
    ConditionSpec,
    draw_dirname,
)
from ruleforge.core.digests import sha256_file
from ruleforge.core.record_names import SAME_CALL_RATE_KEY
from ruleforge.evaluate.ledger import DrawLedger, raw_output_sha256
from ruleforge.evaluate.validation import ComponentSchema, strict_think_answer
from ruleforge.tasks.base import FORCED_JSON
from ruleforge.core.artifacts import iter_jsonl_records
from ruleforge.core.constants import EVALUATIONS_FNAME

YES_PHI = PHI_MAP["yes"]
KAPPA_METHOD = "stratified_cohen_kappa_v1"
# Positive when the member's judgments track its own per-rule maxima better than the reference's do.
KAPPA_ORIENTATION = "kappa(member) - kappa(reference)"
KAPPA_TEST_METHOD = BOOTSTRAP_TEST_METHOD
# The scenario field whose values stratify the chance agreement.
KAPPA_STRATUM_FIELD = "source"
SINGLE_STRATUM = "all"
M_STRATA = {
    "M=yes": lambda m: m == YES_PHI,
    "M!=yes": lambda m: m != YES_PHI,
    "M=unsure": lambda m: m == PHI_MAP["unsure"],
    "M=no": lambda m: m == PHI_MAP["no"],
}
STRATIFIED_ORIENTATION = "rate(member) - rate(reference)"
# The harm flag: the member's M != yes rate above the reference's by more than this, with an interval that excludes 0.
HARM_STRATUM = "M!=yes"
HARM_THRESHOLD = 0.10
DECOMPOSITION_METHOD = "independent_draw_expected_shares_v1"
# Positive when the member is less inconsistent.
DECOMPOSITION_ORIENTATION = "cell(reference, reference) - cell(member, member), split into judgment and reference parts"
SAME_CALL_METHOD = "post_hoc_complete_rule_map_reclassification_v1"


class DiscriminationError(RepeatedSamplingError):
    """The judgments, the strata or the stored outputs are not what these statistics need."""


@dataclass(frozen=True)
class VerdictGrids:
    """One condition's phi(F) and M as ``(N, S)`` grids over the manifest's scenarios."""

    full: DrawGrid
    rule_max: DrawGrid

    @classmethod
    def from_cells(
        cls, condition_id: str, cells: Mapping[tuple[str, int], Mapping[str, Any]], *,
        scenario_ids: Sequence[str], n_draws: int, identity_id: Optional[str] = None,
    ) -> "VerdictGrids":
        """Build one from bundle outcomes' ``full_set_phi`` and ``max_rule_phi``."""
        return cls(*(
            outcome_grid(condition_id, cells, scenario_ids=scenario_ids, n_draws=n_draws, quantity=quantity,
                         identity_id=identity_id)
            for quantity in ("full_set_phi", "max_rule_phi")
        ))

    @property
    def condition_id(self) -> str:
        return self.full.condition_id

    @property
    def digests(self) -> dict[str, str]:
        return {self.full.quantity: self.full.digest, self.rule_max.quantity: self.rule_max.digest}

    def shares(self, which: str, target: float) -> np.ndarray:
        """Per scenario, the share of draws whose ``which`` (``full`` or ``rule_max``) is ``target``."""
        return (getattr(self, which).values == target).mean(axis=1)


@dataclass(frozen=True)
class Strata:
    """Each scenario's stratum, in manifest order; ``field`` None when the population carries none."""

    field: Optional[str]
    labels: tuple[str, ...]

    @property
    def names(self) -> list[str]:
        return sorted(set(self.labels))

    def indicator(self, name: str) -> np.ndarray:
        return np.array([label == name for label in self.labels], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "counts": {name: self.labels.count(name) for name in self.names}}


def read_strata(
    condition_roots: Sequence[Path], scenario_ids: Sequence[str], field: str = KAPPA_STRATUM_FIELD
) -> Strata:
    """Return each scenario's stratum from the rows judged; a field carried on only some rows is refused."""
    found: dict[str, Optional[str]] = {}
    for root in condition_roots:
        path = Path(root) / draw_dirname(0) / EVALUATIONS_FNAME
        seen: dict[str, Optional[str]] = {}
        for row in iter_jsonl(path):
            value = (row.get("scenario") or {}).get(field)
            seen[str(row["scenario_id"])] = None if value is None else str(value)
        if set(seen) != set(scenario_ids):
            raise DiscriminationError(f"{path}: its scenario rows are not the campaign's {len(scenario_ids)} scenarios")
        for sid, value in seen.items():
            if found.setdefault(sid, value) != value:
                raise DiscriminationError(
                    f"{path}: scenario {sid[:12]} has {field} {value!r}, another condition's rows {found[sid]!r}"
                )
    values = [found[sid] for sid in scenario_ids]
    if all(value is None for value in values):
        return Strata(None, (SINGLE_STRATUM,) * len(values))
    if any(value is None for value in values):
        raise DiscriminationError(
            f"{sum(value is None for value in values)} of {len(values)} scenarios carry no {field!r}: the chance "
            "agreement is taken within strata, and a partly stratified population has none"
        )
    return Strata(field, tuple(values))


def _require_same_scenarios(*grids: VerdictGrids) -> None:
    ids = {grid.full.scenario_ids for grid in grids} | {grid.rule_max.scenario_ids for grid in grids}
    if len(ids) != 1:
        raise DiscriminationError("the judgment grids must cover the same scenarios in the same order")


def kappa_features(grids: VerdictGrids, strata: Strata) -> np.ndarray:
    """Return per-scenario features: agreement, class shares of phi(F) and M per stratum, and stratum indicators."""
    if len(strata.labels) != grids.full.n_scenarios:
        raise DiscriminationError(f"{len(strata.labels)} strata for {grids.full.n_scenarios} scenarios")
    agree = (grids.full.values == grids.rule_max.values).mean(axis=1)
    indicators = [strata.indicator(name) for name in strata.names]
    blocks = [agree[:, None]]
    for which in ("full", "rule_max"):
        blocks.extend(
            (indicator * grids.shares(which, target))[:, None] for indicator in indicators for target in PHI_CLASSES
        )
    blocks.extend(indicator[:, None] for indicator in indicators)
    return np.hstack(blocks)


def kappa_parts(means: np.ndarray, n_strata: int) -> np.ndarray:
    """Return ``(n, 3)``: kappa, P_o and P_e from ``kappa_features`` means; kappa is NaN where ``P_e`` is 1."""
    n_classes = len(PHI_CLASSES)
    width = n_strata * n_classes
    p_o = means[:, 0]
    full = means[:, 1 : 1 + width].reshape(-1, n_strata, n_classes)
    rule_max = means[:, 1 + width : 1 + 2 * width].reshape(-1, n_strata, n_classes)
    weights = means[:, 1 + 2 * width :]
    joint = (full * rule_max).sum(axis=2)
    p_e = np.divide(joint, weights, out=np.zeros_like(joint), where=weights > 0).sum(axis=1)
    margin = 1.0 - p_e
    kappa = np.divide(p_o - p_e, margin, out=np.full_like(p_o, np.nan), where=margin > 1e-12)
    return np.column_stack([kappa, p_o, p_e])


def kappa_summary(grids: VerdictGrids, strata: Strata, **bootstrap: Any) -> dict[str, Any]:
    """One arm's kappa with its interval, P_o and P_e."""
    point, replicates = scenario_cluster_bootstrap(
        kappa_features(grids, strata), lambda means: kappa_parts(means, len(strata.names)), **bootstrap
    )
    return {
        **statistic_interval(point[0], replicates[:, 0], bootstrap["ci_level"]),
        "p_observed": float(point[1]),
        "p_chance": float(point[2]),
        "method": KAPPA_METHOD,
        "strata": strata.to_dict(),
    }


def kappa_contrast(
    reference: VerdictGrids,
    member: VerdictGrids,
    strata: Strata,
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
    n_permutations: int,
) -> dict[str, Any]:
    """Return ``kappa(member) - kappa(reference)``: a paired interval, the bootstrap p that decides it, and a
    randomization p, which tests exchangeability (equal kappas with shifted marginals lack it)."""
    _require_same_scenarios(reference, member)
    n_strata = len(strata.names)
    ref_features, member_features = kappa_features(reference, strata), kappa_features(member, strata)
    width = ref_features.shape[1]

    def both(means: np.ndarray) -> np.ndarray:
        ref = kappa_parts(means[:, :width], n_strata)[:, 0]
        mem = kappa_parts(means[:, width:], n_strata)[:, 0]
        return np.column_stack([ref, mem, mem - ref])

    point, replicates = scenario_cluster_bootstrap(
        np.hstack([ref_features, member_features]), both, n_resamples=n_resamples, ci_level=ci_level, seed=seed,
    )
    test = clustered_paired_randomization_statistic(
        ref_features, member_features, lambda means: kappa_parts(means, n_strata)[:, :1],
        n_permutations=n_permutations, seed=seed,
    )
    defined = bool(np.isfinite(point[2]))
    return {
        "a": reference.condition_id,
        "b": member.condition_id,
        "orientation": KAPPA_ORIENTATION,
        "kappa_a": statistic_interval(point[0], replicates[:, 0], ci_level),
        "kappa_b": statistic_interval(point[1], replicates[:, 1], ci_level),
        **statistic_interval(point[2], replicates[:, 2], ci_level),
        "n_scenarios": reference.full.n_scenarios,
        "n_decoder_draws_a": reference.full.n_draws,
        "n_decoder_draws_b": member.full.n_draws,
        # An undefined difference stays in the family at p = 1, so the Holm family keeps its declared size.
        "p_value": percentile_bootstrap_p(replicates[:, 2]) if defined else 1.0,
        "p_value_defined": defined,
        "p_randomization": test["p_value"],
        "randomization_test_method": TEST_METHOD,
        "n_permutations": test["n_permutations"],
        "n_undefined_permutations": test["n_undefined_permutations"],
        "undefined_permutations": test["undefined_permutations"],
        "statistic": KAPPA_METHOD,
        "strata": strata.to_dict(),
        "inference_method": INFERENCE_METHOD,
        "test_method": KAPPA_TEST_METHOD,
        "grid_digests_a": reference.digests,
        "grid_digests_b": member.digests,
        "bootstrap_seed": seed,
    }


def stratified_features(grids: VerdictGrids) -> np.ndarray:
    """Per scenario and :data:`M_STRATA` stratum: the share of draws inconsistent with M in it, and in it."""
    full, rule_max = grids.full.values, grids.rule_max.values
    columns = []
    for member_of in M_STRATA.values():
        inside = member_of(rule_max)
        columns.append((inside & (full != rule_max)).mean(axis=1))
        columns.append(inside.mean(axis=1))
    return np.column_stack(columns)


def _ratios(means: np.ndarray) -> np.ndarray:
    numerators, denominators = means[:, 0::2], means[:, 1::2]
    return np.divide(numerators, denominators, out=np.full_like(numerators, np.nan), where=denominators > 0)


def arm_statistics(
    grids: VerdictGrids, strata: Strata, *, n_resamples: int, ci_level: float, seed: int
) -> dict[str, Any]:
    """Return one arm's kappa, yes-rate, always-yes floor and M-stratified rates, each with its interval."""
    bootstrap = dict(n_resamples=n_resamples, ci_level=ci_level, seed=seed)
    linear = np.column_stack([grids.shares("full", YES_PHI), 1.0 - grids.shares("rule_max", YES_PHI)])
    point, replicates = scenario_cluster_bootstrap(linear, lambda means: means, **bootstrap)
    stratified = stratified_features(grids)
    ratio_point, ratio_replicates = scenario_cluster_bootstrap(stratified, _ratios, **bootstrap)
    draws_in = np.rint(stratified[:, 1::2].sum(axis=0) * grids.full.n_draws).astype(int)
    return {
        "n_scenarios": grids.full.n_scenarios,
        "n_decoder_draws": grids.full.n_draws,
        "kappa": kappa_summary(grids, strata, **bootstrap),
        "yes_rate": statistic_interval(point[0], replicates[:, 0], ci_level),
        "always_yes_floor": statistic_interval(point[1], replicates[:, 1], ci_level),
        "m_stratified": {
            name: {**statistic_interval(ratio_point[i], ratio_replicates[:, i], ci_level), "n_draws": int(draws_in[i])}
            for i, name in enumerate(M_STRATA)
        },
        "grid_digests": grids.digests,
    }


def stratified_differences(
    reference: VerdictGrids, member: VerdictGrids, *, n_resamples: int, ci_level: float, seed: int
) -> dict[str, Any]:
    """Return ``rate(member) - rate(reference)`` per M stratum with a paired interval, and the harm flag;
    descriptive."""
    _require_same_scenarios(reference, member)
    ref, mem = stratified_features(reference), stratified_features(member)
    width = ref.shape[1]
    point, replicates = scenario_cluster_bootstrap(
        np.hstack([ref, mem]), lambda means: _ratios(means[:, width:]) - _ratios(means[:, :width]),
        n_resamples=n_resamples, ci_level=ci_level, seed=seed,
    )
    differences = {
        name: statistic_interval(point[i], replicates[:, i], ci_level) for i, name in enumerate(M_STRATA)
    }
    harm = differences[HARM_STRATUM]
    flagged = (
        harm["estimate"] is not None and harm["ci_low"] is not None
        and harm["estimate"] > HARM_THRESHOLD and harm["ci_low"] > 0
    )
    return {
        "orientation": STRATIFIED_ORIENTATION,
        "differences": differences,
        "harm_flag": {"stratum": HARM_STRATUM, "threshold": HARM_THRESHOLD, "flagged": bool(flagged)},
    }


# ``F_<judgments' arm>|M_<references' arm>``.
DECOMPOSITION_CELLS = (
    ("reference", "reference"), ("member", "reference"), ("reference", "member"), ("member", "member"),
)


def cross_inconsistency(verdicts: VerdictGrids, references: VerdictGrids) -> np.ndarray:
    """Return per scenario the share of independent (judgment draw, reference draw) pairs that disagree."""
    agree = sum(verdicts.shares("full", c) * references.shares("rule_max", c) for c in PHI_CLASSES)
    return 1.0 - agree


def cross_arm_decomposition(
    reference: VerdictGrids, member: VerdictGrids, *, n_resamples: int, ci_level: float, seed: int
) -> dict[str, Any]:
    """Return each arm's judgments against each arm's per-rule maxima, with ``verdict_effect`` and ``reference_effect``
    summing to ``total``."""
    _require_same_scenarios(reference, member)
    arms = {"reference": reference, "member": member}
    cells = np.column_stack([cross_inconsistency(arms[f], arms[m]) for f, m in DECOMPOSITION_CELLS])

    def effects(means: np.ndarray) -> np.ndarray:
        rr, mr, rm, mm = (means[:, i] for i in range(4))
        verdict = 0.5 * ((rr - mr) + (rm - mm))
        references = 0.5 * ((rr - rm) + (mr - mm))
        return np.column_stack([rr, mr, rm, mm, rr - mm, verdict, references])

    point, replicates = scenario_cluster_bootstrap(
        cells, effects, n_resamples=n_resamples, ci_level=ci_level, seed=seed
    )
    names = [f"F_{f}|M_{m}" for f, m in DECOMPOSITION_CELLS] + ["total", "verdict_effect", "reference_effect"]
    summary = {name: statistic_interval(point[i], replicates[:, i], ci_level) for i, name in enumerate(names)}
    return {
        "method": DECOMPOSITION_METHOD,
        "orientation": DECOMPOSITION_ORIENTATION,
        "cells": {name: summary[name] for name in names[:4]},
        **{name: summary[name] for name in names[4:]},
    }


@dataclass(frozen=True)
class MapOutcome:
    """One accepted full-set output's rule map; ``in_context_contradiction`` only for a complete map, else
    ``reason``."""

    complete: bool
    in_context_contradiction: Optional[float] = None
    reason: Optional[str] = None


def load_rules(path: Path | str, expected_sha256: str) -> list[dict[str, Any]]:
    """The ruleset a condition judged, refused unless its file digest is the condition's."""
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DiscriminationError(
            f"{path} has SHA-256 {actual[:12]}..., the conditions judged ruleset {expected_sha256[:12]}...: "
            "the same-call rate must be reclassified against the ruleset the campaign validated with"
        )
    return list(iter_jsonl_records(path))


def map_schema(condition: ConditionSpec, rules: Sequence[Mapping[str, Any]]) -> ComponentSchema:
    """The campaign's validator for this condition, with the complete rule map required."""
    return ComponentSchema.for_condition(condition, rules, forced_json_requirement=FORCED_JSON_COMPLETE_RULE_MAP)


def reclassify_forced_maps(
    condition_root: Path,
    condition: ConditionSpec,
    *,
    campaign_id: str,
    scenario_ids: Sequence[str],
    n_draws: int,
    rules: Sequence[Mapping[str, Any]],
    cells: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, int], MapOutcome]:
    """Return each accepted full-set output's rule map, reread from its draw's ledger and raw outputs; read-only."""
    if condition.contract != FORCED_JSON:
        raise DiscriminationError(f"{condition.condition_id}: the same-call rate needs the forced prompt, not "
                                  f"{condition.prompt_style!r}")
    schema = map_schema(condition, rules)
    if set(schema.expected_rule_ids) != set(next(iter(cells.values()))["rule_judgments"]):
        raise DiscriminationError(f"{condition.condition_id}: the ruleset's rules are not the rules its bundles judged")
    outcomes: dict[tuple[str, int], MapOutcome] = {}
    for draw in range(n_draws):
        where = f"{condition.condition_id}/{draw_dirname(draw)}"
        ledger = DrawLedger(
            Path(condition_root) / draw_dirname(draw), campaign_id=campaign_id,
            condition_id=condition.condition_id, decoder_draw_id=draw,
        )
        state = ledger.accepted()
        raw = {
            (row["scenario_id"], int(row["semantic_attempt_id"]), row["request_block_id"]): row
            for row in ledger.iter_raw()
            if row["component_id"] == FULL_RULESET_COMPONENT
        }
        for sid in scenario_ids:
            if not state.is_accepted(sid):
                raise DiscriminationError(f"{where}: scenario {sid[:12]} has no accepted bundle")
            record, component = state.accepted_components(sid)[FULL_RULESET_COMPONENT]
            row = raw.get((sid, record.semantic_attempt_id, record.request_block_id))
            if row is None or raw_output_sha256(row["raw_response"]) != component.raw_output_sha256:
                raise DiscriminationError(
                    f"{where}: scenario {sid[:12]}'s accepted full-set output is missing from the raw outputs or "
                    "is not the text the ledger recorded"
                )
            text = strict_think_answer(row["raw_response"]) if condition.strict_think else row["completion"]
            verdict = schema.classify_forced(text, require_map=False)
            recorded = cells[(sid, draw)]["full_set_judgment"]
            if not verdict.valid or verdict.judgment != component.judgment or component.judgment != recorded:
                raise DiscriminationError(
                    f"{where}: scenario {sid[:12]}'s stored full-set output reads as {verdict.judgment!r}, the ledger "
                    f"says {component.judgment!r} and the bundle outcome {recorded!r}"
                )
            mapped = schema.classify_forced(text, require_map=True)
            if not mapped.valid:
                outcomes[(sid, draw)] = MapOutcome(False, reason=mapped.failure_reason)
                continue
            contradiction = calculate_inconsistency(verdict.judgment, list(mapped.rule_judgments.values()))["contradiction"]
            outcomes[(sid, draw)] = MapOutcome(True, contradiction)
    return outcomes


def same_call_statistics(
    outcomes: Mapping[tuple[str, int], MapOutcome],
    *,
    scenario_ids: Sequence[str],
    n_draws: int,
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    """Return one arm's map completeness and same-call rate over complete maps; never contrast them between arms."""
    complete = np.array([[outcomes[(sid, d)].complete for d in range(n_draws)] for sid in scenario_ids], dtype=float)
    contradiction = np.array(
        [[outcomes[(sid, d)].in_context_contradiction or 0.0 for d in range(n_draws)] for sid in scenario_ids]
    )
    features = np.column_stack([complete.mean(axis=1), contradiction.mean(axis=1)])

    def statistic(means: np.ndarray) -> np.ndarray:
        rate = np.divide(means[:, 1], means[:, 0], out=np.full(means.shape[0], np.nan), where=means[:, 0] > 0)
        return np.column_stack([means[:, 0], rate])

    point, replicates = scenario_cluster_bootstrap(
        features, statistic, n_resamples=n_resamples, ci_level=ci_level, seed=seed
    )
    reasons = Counter(str(outcome.reason) for outcome in outcomes.values() if not outcome.complete)
    return {
        "method": SAME_CALL_METHOD,
        "between_arm_contrast": None,
        "n_outputs": int(complete.size),
        "n_complete_maps": int(complete.sum()),
        "incomplete_reasons": dict(sorted(reasons.items())),
        "completeness": statistic_interval(point[0], replicates[:, 0], ci_level),
        SAME_CALL_RATE_KEY: statistic_interval(point[1], replicates[:, 1], ci_level),
    }
