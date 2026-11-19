from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ruleforge.experiments.arm_records import ISOLATED
from ruleforge.experiments.checks import require
from ruleforge.experiments.leniency.golden import validate_as_written, validate_payload
from ruleforge.experiments.leniency.schema import (
    AGREEMENT_METHOD,
    ALONE,
    DATA_VERSION,
    DIRECTION_COUNT_FIELDS,
    FEW_INFORMATIVE_BELOW,
    GOLDEN_SOURCE,
    IN_CONTEXT,
    LABELS,
    METRIC,
    TRANSITION_METHOD,
    RuleContextError,
    check_holm,
    direction_counts,
)
from ruleforge.experiments.labels import (
    campaign_label,
    display_order,
    rule_label_number,
)
from ruleforge.evaluate.estimates import (
    SHARED_DESIGN_FIELDS,
    estimates_source,
    is_complete,
    load_estimates,
)
from ruleforge.evaluate.multiplicity import ALPHA, HOLM_MC_MARGIN_SE, holm_flags
from ruleforge.evaluate.inference import (
    INFERENCE_METHOD,
    PRIMARY_INTERVAL_BASIS,
    TEST_METHOD,
    load_condition_outcomes,
    rule_grids,
)
from ruleforge.evaluate.report import transition_summary
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.rulesets.rules import build_rule_id_index, normalize_preempting_rule_id
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME, load_conditions
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_FINAL,
    FORCED_JSON_COMPLETE_RULE_MAP,
    FULL_RULESET_COMPONENT,
    draw_dirname,
    load_sampling_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.ledger import DrawLedger
from ruleforge.prompts.styles import (
    FORCED_GRPO,
    FORCED_JSON_STYLES,
    lists_preemption,
    replace_exactly,
)
from ruleforge.core.artifacts import iter_jsonl_records

TRIGGER = "yes"
ADJUSTMENT = "holm"
ADJUSTMENT_FAMILY = "holm_within_model"
#: Condition fields that fix what the two sides of every row mean; joined conditions must agree on all of them.
JOINED_DESIGN_FIELDS = (
    "individual_rule_context",
    "judge_type",
    "label_rule",
    "evaluation_mode",
    "prompt_shape_version",
    "validation_schema_version",
)
_PHI = np.array([PHI_MAP[label] for label in LABELS])
_YES = LABELS.index(TRIGGER)
_NO = LABELS.index("no")

DEFINITION = (
    "For each model's forced-prompt condition (forced-grpo, complete_rule_map) "
    "of the stochastic campaign and each rule r: every one of the N frozen "
    "scenarios is judged in S accepted temperature-1 draws, and each accepted "
    "bundle carries both the separate single-rule call's judgment of r "
    "('alone', rule_judgments) and the same draw's full-set response's own "
    "entry for r ('in context', in_context_rule_judgments), so each side is a "
    "complete N x S grid and nothing is dropped. rate = mean over scenarios of "
    "the mean over draws of 1[judgment == 'yes']; delta = rate_in_context - "
    "rate_alone, so a negative delta means r is deemed violated LESS often "
    "inside the full ruleset. yes_difference is the exact integer behind it "
    "(in-context minus alone 'yes' judgments over the N x S cells); every "
    "direction and tie -- the counts, the colours, the rule order -- is read "
    "from it, never from the float delta. ci_low/ci_high are the campaign's "
    "scenario-cluster bootstrap interval and p_value its scenario-clustered "
    "paired randomization test, read from aggregate/estimates.json after both "
    "grids are rebuilt from the draws and matched by digest; p_holm is Holm "
    "across each model's rules, recomputed. p_value is a Monte Carlo p over B "
    "= n_permutations sign flips: p_value_is_floor marks the smallest value it "
    "can take, 1/(B+1), a bound rather than a point value; holm_mc_margin "
    "marks a Holm decision another permutation seed could flip -- a step j "
    f"lies within {HOLM_MC_MARGIN_SE:g} Monte Carlo standard errors of alpha "
    "when |m_j p_(j) - alpha| <= "
    # " / B", spaced: ")/B" reads as a rooted path to the absolute-path scan (ruleforge.core.paths.absolute_strings).
    f"{HOLM_MC_MARGIN_SE:g} m_j sqrt(p_(j)(1-p_(j)) / B), and a Holm-significant "
    "row is marked when ANY step up to its own does (its rejection needs "
    "every one of them to pass), a non-significant row only when EVERY "
    "failing step up to its own (m_j p_(j) > alpha) does (its rejection "
    "would need every one of them to flip); neither changes a decision. "
    "FLAT: preempted_rules is ignored "
    "wherever a judgment exists. phi_* are the same means under yes=1, "
    "unsure=0.5, no=0. n_informative_scenarios counts the scenarios whose "
    "draw-mean in-context minus alone difference is non-zero; few_informative "
    f"flags fewer than {FEW_INFORMATIVE_BELOW}; degenerate flags a zero-width "
    "interval. transitions are expected independent-draw shares "
    f"({TRANSITION_METHOD}): per (scenario, rule) cell p(a) q(b), p and q the "
    "alone and in-context label shares over its S draws, averaged over cells "
    "-- every alone draw paired with every in-context draw of the scenario, "
    "not only its own bundle. draw_to_draw is the same share for two distinct "
    f"draws of one side ({AGREEMENT_METHOD}: sum over labels of c(c-1) / "
    "(S(S-1))), the agreement each side has with itself: the cross-side "
    "unchanged share would equal both if context had no effect, and in "
    "expectation it cannot exceed their geometric mean. by_verdict splits "
    "delta by the in-context response's own full_set_judgment, which the "
    "forced prompt ties every rule entry to; named_preempted takes out the "
    "cells whose response listed r under preempted_rules. Both are "
    "decompositions over (scenario, draw) cells whose contributions sum to "
    "delta (named plus its unnamed remainder), not corrections; a model's "
    "share_of_pooled_delta is null where pooled_yes_difference, the exact "
    "count behind pooled_delta, is 0."
)


def _no_preemption_definition(style: str) -> str:
    """:data:`DEFINITION` edited for a forced style whose prompt asks for no ``preempted_rules``; each passage must
    occur exactly once, so a stale edit fails at import."""
    edits = (
        ("(forced-grpo, complete_rule_map)", f"({style}, complete_rule_map)"),
        (
            "FLAT: preempted_rules is ignored wherever a judgment exists.",
            "FLAT: this prompt asks for no preempted_rules, so every judgment counts.",
        ),
        (
            "named_preempted takes out the cells whose response listed r under preempted_rules. Both are "
            "decompositions over (scenario, draw) cells whose contributions sum to delta (named plus its unnamed "
            "remainder), not corrections;",
            "named_preempted is null (not applicable, never zero): this prompt asks for no preempted_rules. "
            "by_verdict is a decomposition over (scenario, draw) cells whose contributions sum to delta, not a "
            "correction;",
        ),
    )
    text = DEFINITION
    for old, new in edits:
        text = replace_exactly(text, old, new)
    return text


DEFINITIONS = {
    style: DEFINITION if lists_preemption(style) else _no_preemption_definition(style) for style in FORCED_JSON_STYLES
}


def definition(prompt_style: str) -> str:
    """Return the provenance definition of a golden of ``prompt_style``'s forced conditions."""
    return DEFINITIONS[prompt_style]


_require = partial(require, error=RuleContextError)


@dataclass(frozen=True)
class Rule:
    """One rule of the inventory: its number (the join key), id and title."""

    number: int
    rule_id: str
    title: str


def load_rules(path: Path | str) -> List[Rule]:
    """Return the rules to analyse, sorted by number, from a ``{"id": "Rule N", "rule": title}`` JSONL."""
    rules: List[Rule] = []
    for record in iter_jsonl_records(path):
        number = rule_label_number(record.get("id"))
        title = record.get("rule")
        _require(
            [
                (number is not None, f"rule id {record.get('id')!r} names no rule number"),
                (number not in {rule.number for rule in rules}, f"rule number {number} appears twice"),
                (isinstance(title, str) and bool(title.strip()), f"rule {record.get('id')!r} has no title"),
            ],
            str(path),
        )
        rules.append(Rule(int(number), str(record["id"]).strip(), title.strip()))
    _require([(bool(rules), "no rules")], str(path))
    return sorted(rules, key=lambda rule: rule.number)


@dataclass(frozen=True)
class ConditionCells:
    """One forced condition's judgments as dense label arrays: ``alone``/``in_context`` are ``(N, S, R)`` indices into
    :data:`LABELS`, and ``named`` marks cells whose response listed the rule as preempted (None if the prompt asks
    for no such list)."""

    condition_id: str
    scenario_ids: Tuple[str, ...]
    alone: np.ndarray
    in_context: np.ndarray
    verdict: np.ndarray
    named: Optional[np.ndarray]
    unresolved_preempted_ids: Optional[int]
    outcomes: Mapping[Tuple[str, int], Mapping[str, Any]]

    @property
    def n_scenarios(self) -> int:
        return int(self.alone.shape[0])

    @property
    def n_draws(self) -> int:
        return int(self.alone.shape[1])

    @property
    def difference(self) -> np.ndarray:
        """``1[in context == yes] - 1[alone == yes]`` per (scenario, draw, rule) cell."""
        return (self.in_context == _YES).astype(int) - (self.alone == _YES).astype(int)


def _label_index(value: Any, where: str) -> int:
    _require([(value in LABELS, f"judgment {value!r} is not one of {LABELS}")], where)
    return LABELS.index(value)


def _accepted_full_sets(
    draw_dir: Path, *, campaign_id: str, condition_id: str, draw: int
) -> Dict[str, Tuple[int, Any]]:
    """Return ``{scenario_id: (semantic_attempt_id, full-set component)}`` from the ledger's accepted attempts,
    read-only."""
    state = DrawLedger(
        draw_dir, campaign_id=campaign_id, condition_id=condition_id, decoder_draw_id=draw
    ).accepted()
    out: Dict[str, Tuple[int, Any]] = {}
    for sid in state.accepted:
        record = state.accepted_bundle(sid)
        full = [c for c in record.components if c.component_id == FULL_RULESET_COMPONENT]
        _require(
            [(len(full) == 1, f"accepted attempt of {sid[:12]} has {len(full)} full-set components")],
            str(draw_dir),
        )
        out[sid] = (record.semantic_attempt_id, full[0])
    return out


def require_materialized(draws: Mapping[str, Any], *, fingerprint: str, campaign_id: str, condition_id: str,
                         where: str) -> None:
    """Refuse a condition's draws unless each completion record holds it materialized under this campaign and
    condition, at this fingerprint."""
    _require(
        [
            (draws["condition_fingerprint"] == fingerprint, "the draws record another condition fingerprint"),
            (
                all(
                    record.get("materialized")
                    and record.get("campaign_id") == campaign_id
                    and record.get("condition_id") == condition_id
                    for record in draws["draws"]
                ),
                "a draw is not materialized under this campaign and condition",
            ),
        ],
        where,
    )


def read_condition(
    root: Path,
    condition_id: str,
    *,
    campaign_id: str,
    scenario_ids: Sequence[str],
    n_draws: int,
    manifest_sha256: str,
    rules: Sequence[Rule],
    reads_preemption: bool = True,
) -> Tuple[ConditionCells, Dict[str, Any]]:
    """Return every accepted bundle of one forced condition, with its preemption list when ``reads_preemption``, and the
    draws' provenance."""
    outcomes, provenance = load_condition_outcomes(
        root / condition_id,
        expected_scenario_ids=scenario_ids,
        expected_draws=n_draws,
        scenario_manifest_sha256=manifest_sha256,
    )
    position = {rule.rule_id: k for k, rule in enumerate(rules)}
    index = build_rule_id_index([{"id": rule.rule_id} for rule in rules])
    shape = (len(scenario_ids), n_draws, len(rules))
    alone = np.empty(shape, dtype=np.int8)
    in_context = np.empty(shape, dtype=np.int8)
    verdict = np.empty(shape[:2], dtype=np.int8)
    named = np.zeros(shape, dtype=bool) if reads_preemption else None
    unresolved = 0 if reads_preemption else None
    for draw in range(n_draws):
        draw_dir = root / condition_id / draw_dirname(draw)
        accepted = _accepted_full_sets(
            draw_dir, campaign_id=campaign_id, condition_id=condition_id, draw=draw
        )
        for i, sid in enumerate(scenario_ids):
            row = outcomes[(sid, draw)]
            where = f"{condition_id} {draw_dirname(draw)} scenario {sid[:12]}"
            attempt, full = accepted.get(sid, (None, None))
            alone_map = row.get("rule_judgments") or {}
            context_map = row.get("in_context_rule_judgments") or {}
            _require(
                [
                    (set(alone_map) == set(position), "rule_judgments do not cover exactly the rules file's ids"),
                    (
                        set(context_map) == set(position),
                        "in_context_rule_judgments do not cover exactly the rules file's ids",
                    ),
                    (full is not None, "no accepted ledger row"),
                    (
                        full is not None
                        and attempt == row.get("semantic_attempt_id")
                        and full.judgment == row.get("full_set_judgment")
                        and full.rule_judgments == context_map,
                        "the accepted full-set response is not the one the bundle outcome was built from",
                    ),
                ],
                where,
            )
            for rule_id, k in position.items():
                alone[i, draw, k] = _label_index(alone_map[rule_id], where)
                in_context[i, draw, k] = _label_index(context_map[rule_id], where)
            verdict[i, draw] = _label_index(row["full_set_judgment"], where)
            for raw in full.preempted_rules if reads_preemption else ():
                canonical = normalize_preempting_rule_id(raw, index)
                if canonical is None:
                    unresolved += 1
                else:
                    named[i, draw, position[canonical]] = True
    arrays = ConditionCells(
        condition_id=condition_id,
        scenario_ids=tuple(scenario_ids),
        alone=alone,
        in_context=in_context,
        verdict=verdict,
        named=named,
        unresolved_preempted_ids=unresolved,
        outcomes=outcomes,
    )
    return arrays, provenance


def rows_by_rule(
    rows: Sequence[Mapping[str, Any]], rules: Sequence[Rule], where: str
) -> Dict[int, Mapping[str, Any]]:
    """Return the aggregate's rows keyed by parsed rule number; they arrive in string order, so a positional join would
    mispair them."""
    numbers = [rule_label_number(row.get("rule_id")) for row in rows]
    wanted = [rule.number for rule in rules]
    covered = sorted(n for n in numbers if n is not None)
    _require(
        [
            (None not in numbers, "a rule_context row names no rule number"),
            (len(set(numbers)) == len(numbers), "two rule_context rows name one rule"),
            (covered == wanted, f"rule_context rows cover rules {covered}, the rules file {wanted}"),
        ],
        where,
    )
    by_number = dict(zip(numbers, rows))
    # The grids are rebuilt by the row's own key, so it must be the rules file's spelling.
    spellings = []
    for rule in rules:
        stored = by_number[rule.number]["rule_id"]
        message = f"row {stored!r} is spelled differently from the rules file's {rule.rule_id!r}"
        spellings.append((stored == rule.rule_id, message))
    _require(spellings, where)
    return by_number


def verify_rule_rows(
    arrays: ConditionCells,
    by_number: Mapping[int, Mapping[str, Any]],
    rules: Sequence[Rule],
    *,
    identity_id: Optional[str] = None,
) -> None:
    """Refuse rows that are not the aggregate of these draws, or a Holm family that is not these rules'."""
    cid = arrays.condition_id
    common = dict(scenario_ids=arrays.scenario_ids, n_draws=arrays.n_draws, trigger=TRIGGER, identity_id=identity_id)
    for rule in rules:
        row = by_number[rule.number]
        rule_id = row["rule_id"]
        in_context = rule_grids(
            cid, arrays.outcomes, rule_id=rule_id, field="in_context_rule_judgments", **common
        )
        alone = rule_grids(cid, arrays.outcomes, rule_id=rule_id, **common)
        estimate = float(row["estimate"])
        sides = (row.get("a"), row.get("b"))
        draws = (row.get("n_decoder_draws_a"), row.get("n_decoder_draws_b"))
        _require(
            [
                (sides == (IN_CONTEXT, ALONE), f"contrast is {sides[0]!r} - {sides[1]!r}"),
                (row.get("grid_digest_a") == in_context.digest, "in-context grid digest differs from the draws"),
                (row.get("grid_digest_b") == alone.digest, "alone grid digest differs from the draws"),
                (
                    math.isclose(estimate, in_context.rate - alone.rate, abs_tol=1e-12),
                    "estimate is not the grids' difference",
                ),
                (row.get("n_scenarios") == arrays.n_scenarios, "n_scenarios differs"),
                (draws == (arrays.n_draws,) * 2, "draw count differs"),
                (
                    row.get("interval_basis") == PRIMARY_INTERVAL_BASIS,
                    f"interval basis {row.get('interval_basis')!r}",
                ),
                (
                    row.get("inference_method") == INFERENCE_METHOD,
                    f"inference method {row.get('inference_method')!r}",
                ),
                (row.get("test_method") == TEST_METHOD, f"test method {row.get('test_method')!r}"),
                (
                    row.get("family_size") == len(rules),
                    f"family_size {row.get('family_size')!r}, not {len(rules)}",
                ),
            ],
            f"{cid} {rule_id}",
        )
    # ALPHA must be the aggregate's own Holm level: the stored flags were decided there.
    check_holm([by_number[rule.number] for rule in rules], alpha=ALPHA, where=cid)


def _shares(labels: np.ndarray) -> np.ndarray:
    return np.stack([(labels == k).mean(axis=-1) for k in range(len(LABELS))], axis=-1)


def _matrix(values: np.ndarray) -> Dict[str, Dict[str, float]]:
    return {a: {b: float(values[i, j]) for j, b in enumerate(LABELS)} for i, a in enumerate(LABELS)}


def expected_transitions(alone: np.ndarray, in_context: np.ndarray) -> np.ndarray:
    """Mean over cells of ``p(a) q(b)`` for ``(cells, S)`` label arrays; sums to 1."""
    p, q = _shares(alone), _shares(in_context)
    return p.T @ q / p.shape[0]


def draw_to_draw(labels: np.ndarray) -> np.ndarray:
    """Return the joint label shares of two distinct draws of one side, averaged over cells; its trace is a reference,
    not a ceiling."""
    n_draws = labels.shape[-1]
    counts = _shares(labels) * n_draws
    joint = counts.T @ counts - np.diag(counts.sum(axis=0))
    return joint / (labels.shape[0] * n_draws * (n_draws - 1))


def _agreement(matrix: np.ndarray) -> Dict[str, Any]:
    summary = transition_summary(_matrix(matrix))
    return {
        "matrix": _matrix(matrix),
        **{key: float(summary[key]) for key in ("unchanged_fraction", "chance_agreement", "kappa")},
    }


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _share_of(contribution: float, pooled: float) -> Optional[float]:
    """Return a contribution's share of a pooled delta, None where it is 0; ``+ 0.0`` keeps ``-0.0`` out of JSON."""
    return float(contribution / pooled) + 0.0 if pooled else None


def rule_effect(rule: Rule, row: Mapping[str, Any], arrays: ConditionCells, k: int) -> Dict[str, Any]:
    """One rule's golden row: the aggregate's inference plus the descriptions."""
    alone, in_context = arrays.alone[:, :, k], arrays.in_context[:, :, k]
    difference = arrays.difference[:, :, k]
    n_cells = difference.size
    rate_alone, rate_in_context = float((alone == _YES).mean()), float((in_context == _YES).mean())
    delta = float(row["estimate"])
    by_verdict = {
        label: float(difference[arrays.verdict == j].sum() / n_cells) for j, label in enumerate(LABELS)
    }
    # Identities of a complete grid: the pooled cell mean is the rate, and each decomposition sums back to delta.
    assert math.isclose(delta, rate_in_context - rate_alone, abs_tol=1e-12)
    assert math.isclose(sum(by_verdict.values()), delta, abs_tol=1e-12)
    n_informative = int(np.count_nonzero(difference.sum(axis=1)))
    phi_alone, phi_in_context = _PHI[alone].mean(), _PHI[in_context].mean()
    return {
        "rule": rule.number,
        "rule_id": rule.rule_id,
        "title": rule.title,
        "n_scenarios": arrays.n_scenarios,
        "n_decoder_draws": arrays.n_draws,
        "rate_alone": rate_alone,
        "rate_in_context": rate_in_context,
        "delta": delta,
        "yes_difference": int(difference.sum()),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "sd": float(row["sd"]),
        "se_analytic": float(row["se_analytic"]),
        "degenerate": bool(row["ci_low"] == row["ci_high"]),
        "p_value": float(row["p_value"]),
        "n_permutations": int(row["n_permutations"]),
        "p_holm": float(row["p_holm"]),
        "significant_holm": bool(row["significant_holm"]),
        "grid_digest_in_context": row["grid_digest_a"],
        "grid_digest_alone": row["grid_digest_b"],
        "n_informative_scenarios": n_informative,
        "few_informative": n_informative < FEW_INFORMATIVE_BELOW,
        "phi_alone": float(phi_alone),
        "phi_in_context": float(phi_in_context),
        "phi_delta": float(phi_in_context - phi_alone),
        "transitions": _matrix(expected_transitions(alone, in_context)),
        "by_verdict": by_verdict,
        "named_preempted": None if arrays.named is None else {
            "n_cells": int(arrays.named[:, :, k].sum()),
            "delta_contribution": float(difference[arrays.named[:, :, k]].sum() / n_cells),
        },
    }



def model_summary(effects: Sequence[Mapping[str, Any]], arrays: ConditionCells) -> Dict[str, Any]:
    """Return the summary pooled over every (scenario, draw, rule) cell, with the exact count, direction and Holm
    counts."""
    deltas = np.array([row["delta"] for row in effects])
    phi_alone, phi_in_context = _PHI[arrays.alone].mean(), _PHI[arrays.in_context].mean()
    return {
        "n_rules": len(effects),
        "n_cells": int(arrays.alone.size),
        "pooled_rate_alone": float((arrays.alone == _YES).mean()),
        "pooled_rate_in_context": float((arrays.in_context == _YES).mean()),
        "pooled_delta": float(deltas.mean()),
        "pooled_yes_difference": sum(int(row["yes_difference"]) for row in effects),
        "pooled_phi_alone": float(phi_alone),
        "pooled_phi_in_context": float(phi_in_context),
        "pooled_phi_delta": float(phi_in_context - phi_alone),
        **dict(zip(DIRECTION_COUNT_FIELDS, direction_counts(effects))),
    }


def flips(arrays: ConditionCells) -> Dict[str, Any]:
    """The model's pooled alone -> in-context shares with both references."""
    n_draws = arrays.n_draws
    alone = arrays.alone.transpose(0, 2, 1).reshape(-1, n_draws)
    in_context = arrays.in_context.transpose(0, 2, 1).reshape(-1, n_draws)
    return {
        "method": TRANSITION_METHOD,
        "units": "share of (scenario x rule) cells",
        "n_cells": int(alone.shape[0]),
        **_agreement(expected_transitions(alone, in_context)),
        "draw_to_draw": {
            "method": AGREEMENT_METHOD,
            ALONE: _agreement(draw_to_draw(alone)),
            IN_CONTEXT: _agreement(draw_to_draw(in_context)),
        },
    }


def _exact_pooled(pooled_delta: float, pooled_yes_difference: int) -> float:
    """Return the pooled delta a share is taken of: 0 where its exact count is 0, since rounding would leave noise."""
    return pooled_delta if pooled_yes_difference else 0.0


def verdict_decomposition(arrays: ConditionCells, pooled_delta: float, pooled_yes_difference: int) -> Dict[str, Any]:
    """The model's cells split by the in-context response's own judgment."""
    difference = arrays.difference
    pooled = _exact_pooled(pooled_delta, pooled_yes_difference)
    strata = {}
    for j, label in enumerate(LABELS):
        mask = arrays.verdict == j
        n_cells = int(mask.sum()) * arrays.alone.shape[2]
        contribution = float(difference[mask].sum() / difference.size)
        strata[label] = {
            "n_responses": int(mask.sum()),
            "share_of_responses": float(mask.mean()),
            "rate_alone": _ratio((arrays.alone[mask] == _YES).sum(), n_cells),
            "rate_in_context": _ratio((arrays.in_context[mask] == _YES).sum(), n_cells),
            "delta_contribution": contribution,
            "share_of_pooled_delta": _share_of(contribution, pooled),
        }
    return {"strata": strata}


def named_decomposition(
    arrays: ConditionCells, pooled_delta: float, pooled_yes_difference: int
) -> Optional[Dict[str, Any]]:
    """Return the cells whose response named their rule under preempted_rules, or None for a condition whose prompt asks
    for no such list."""
    named = arrays.named
    if named is None:
        return None
    difference = arrays.difference
    contribution = float(difference[named].sum() / difference.size)
    pooled = _exact_pooled(pooled_delta, pooled_yes_difference)
    return {
        "n_named_cells": int(named.sum()),
        "share_of_cells": float(named.mean()),
        "n_responses_naming": int(named.any(axis=2).sum()),
        "in_context_no_rate": _ratio((arrays.in_context[named] == _NO).sum(), named.sum()),
        "delta_contribution": contribution,
        "share_of_pooled_delta": _share_of(contribution, pooled),
        "unresolved_preempted_ids": arrays.unresolved_preempted_ids,
    }


def _campaign_inputs(
    payload: Mapping[str, Any], *, allow_incomplete: bool
) -> Tuple[Path, Dict[str, Any], ScenarioManifest, Dict[str, Any]]:
    """Return the root beside ``<root>/aggregate/estimates.json``, checked against it, read-only."""
    path = Path(payload["_path"])
    root = path.parent.parent
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    scenarios = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME)
    _require(
        [
            (
                payload.get("campaign_id") == manifest.get("campaign_id"),
                "campaign_id differs from the root's sampling manifest",
            ),
            (
                payload.get("scenario_manifest_sha256")
                == scenarios.manifest_sha256
                == manifest.get("scenario_manifest_sha256"),
                "the estimates, the sampling manifest and the scenario manifest name different populations",
            ),
            (payload.get("n_scenarios") == scenarios.record_count, "n_scenarios differs from the scenario manifest"),
            (
                payload.get("n_decoder_draws") == manifest.get("expected_decoder_draws"),
                "n_decoder_draws differs from the sampling manifest",
            ),
            (int(payload.get("n_decoder_draws") or 0) >= 2, "draw-to-draw agreement needs at least two draws"),
            (
                allow_incomplete or manifest.get("campaign_mode") == CAMPAIGN_MODE_FINAL,
                f"campaign_mode {manifest.get('campaign_mode')!r}; the paper's figures need a final campaign",
            ),
        ],
        str(path),
    )
    return root, manifest, scenarios, load_conditions(root)


def analyze_condition(
    payload: Mapping[str, Any],
    condition_id: str,
    *,
    root: Path,
    manifest: Mapping[str, Any],
    scenarios: ScenarioManifest,
    conditions: Mapping[str, Any],
    rules: Sequence[Rule],
    rules_sha256: str,
    prompt_style: str = FORCED_GRPO,
) -> Tuple[str, Dict[str, Any], ConditionCells]:
    """Return one forced condition's golden block under its model's label, and the verified cells behind it."""
    where = f"{payload['_path']} {condition_id}"
    spec = conditions.get(condition_id)
    _require([(spec is not None, "no registered condition.json")], where)
    summary = payload["conditions"].get(condition_id) or {}
    _require(
        [
            (
                spec.fingerprint == summary.get("condition_fingerprint"),
                "the estimates carry another condition fingerprint",
            ),
            (
                spec.rule_set_sha256 == rules_sha256,
                f"the condition judged ruleset {spec.rule_set_sha256[:12]}..., "
                f"not the rules file's {rules_sha256[:12]}...",
            ),
            (
                (spec.prompt_style, spec.forced_json_requirement) == (prompt_style, FORCED_JSON_COMPLETE_RULE_MAP),
                f"not a forced ({prompt_style}, complete_rule_map) condition",
            ),
            (
                spec.individual_rule_context == ISOLATED,
                f"individual_rule_context {spec.individual_rule_context!r}, not {ISOLATED!r}: its "
                "rule_judgments are not rules judged alone",
            ),
        ],
        where,
    )
    label = campaign_label(spec.model_name)
    arrays, draws = read_condition(
        root,
        condition_id,
        campaign_id=manifest["campaign_id"],
        scenario_ids=scenarios.scenario_ids,
        n_draws=int(manifest["expected_decoder_draws"]),
        manifest_sha256=scenarios.manifest_sha256,
        rules=rules,
        reads_preemption=lists_preemption(prompt_style),
    )
    require_materialized(draws, fingerprint=spec.fingerprint, campaign_id=manifest["campaign_id"],
                         condition_id=condition_id, where=where)
    by_number = rows_by_rule(payload["forced_conditions"][condition_id]["rule_context"], rules, where)
    verify_rule_rows(arrays, by_number, rules, identity_id=spec.identity_id)
    effects = [rule_effect(rule, by_number[rule.number], arrays, k) for k, rule in enumerate(rules)]
    for row, flags in zip(effects, holm_flags(effects, alpha=ALPHA)):
        row.update(flags)
    headline = model_summary(effects, arrays)
    pooled = (headline["pooled_delta"], headline["pooled_yes_difference"])
    return label, {
        "condition_id": spec.public_id,
        "model_name": spec.model_name,
        "campaign_id": manifest["campaign_id"],
        "effects": effects,
        "summary": headline,
        "flips": flips(arrays),
        "by_verdict": verdict_decomposition(arrays, *pooled),
        "named_preempted": named_decomposition(arrays, *pooled),
        "attempt_statistics": summary.get("attempt_statistics"),
    }, arrays


def build_payload(
    estimates_paths: Sequence[Path],
    rules_path: Path | str,
    *,
    allow_incomplete: bool = False,
    allow_stack_mismatch: bool = False,
    prompt_style: str = FORCED_GRPO,
) -> Dict[str, Any]:
    """Return the golden: every forced ``prompt_style`` condition of every joined root, verified."""
    return build_payload_cells(
        estimates_paths, rules_path, allow_incomplete=allow_incomplete,
        allow_stack_mismatch=allow_stack_mismatch, prompt_style=prompt_style,
    )[0]


def build_payload_cells(
    estimates_paths: Sequence[Path],
    rules_path: Path | str,
    *,
    allow_incomplete: bool = False,
    allow_stack_mismatch: bool = False,
    prompt_style: str = FORCED_GRPO,
) -> Tuple[Dict[str, Any], Dict[str, ConditionCells]]:
    """Return :func:`build_payload`'s golden and each label's verified cells; one golden holds one forced style, and it
    is validated as written."""
    _require(
        [(prompt_style in FORCED_JSON_STYLES, f"prompt_style {prompt_style!r} is not one of {FORCED_JSON_STYLES}")],
        "the golden",
    )
    payloads = load_estimates(
        estimates_paths, allow_incomplete=allow_incomplete, allow_stack_mismatch=allow_stack_mismatch
    )
    rules = load_rules(rules_path)
    rules_sha256 = sha256_file(Path(rules_path))
    models: Dict[str, Dict[str, Any]] = {}
    cells: Dict[str, ConditionCells] = {}
    campaigns: List[Dict[str, Any]] = []
    complete = True
    # (condition_id, design) of the first forced condition read.
    reference: Optional[Tuple[str, Dict[str, Any]]] = None
    for payload in payloads:
        root, manifest, scenarios, conditions = _campaign_inputs(payload, allow_incomplete=allow_incomplete)
        complete = complete and is_complete(payload) and manifest.get("campaign_mode") == CAMPAIGN_MODE_FINAL
        # A forced condition with no condition.json stays in, for analyze_condition to refuse.
        forced = sorted(
            condition_id for condition_id in payload.get("forced_conditions") or {}
            if condition_id not in conditions or conditions[condition_id].prompt_style == prompt_style
        )
        for condition_id in forced:
            label, block, arrays = analyze_condition(
                payload,
                condition_id,
                root=root,
                manifest=manifest,
                scenarios=scenarios,
                conditions=conditions,
                rules=rules,
                rules_sha256=rules_sha256,
                prompt_style=prompt_style,
            )
            earlier = models.get(label, {}).get("condition_id")
            _require(
                [(label not in models, f"forced conditions {earlier} and {condition_id} share it")],
                f"golden label {label}",
            )
            design = {field: getattr(conditions[condition_id], field) for field in JOINED_DESIGN_FIELDS}
            reference = reference or (condition_id, design)
            differing = [
                f"{field} {design[field]!r} vs {reference[1][field]!r}"
                for field in JOINED_DESIGN_FIELDS
                if design[field] != reference[1][field]
            ]
            _require(
                [(not differing, f"differs from {reference[0]} on " + ", ".join(differing))],
                condition_id,
            )
            models[label] = block
            cells[label] = arrays
        campaigns.append({**estimates_source(payload),
                          "forced_conditions": sorted(conditions[condition_id].public_id for condition_id in forced)})
    _require([(bool(models), f"no forced ({prompt_style}, complete_rule_map) condition")], "the estimates")
    order = display_order(models)
    first = payloads[0]
    golden = {
        "data_version": DATA_VERSION,
        "source": GOLDEN_SOURCE,
        "metric": METRIC,
        "rules": [{"number": rule.number, "id": rule.rule_id, "title": rule.title} for rule in rules],
        "models": {label: models[label] for label in order},
        "settings": {
            "alpha": ALPHA,
            "holm_mc_margin_se": HOLM_MC_MARGIN_SE,
            **reference[1],
            "trigger": TRIGGER,
            "adjustment": ADJUSTMENT,
            "adjustment_family": ADJUSTMENT_FAMILY,
            "family_size": len(rules),
            "inference_method": first["inference_method"],
            "interval_basis": PRIMARY_INTERVAL_BASIS,
            "test_method": TEST_METHOD,
            "transition_method": TRANSITION_METHOD,
            "agreement_method": AGREEMENT_METHOD,
            "few_informative_below": FEW_INFORMATIVE_BELOW,
            "bootstrap": first["bootstrap"],
            "randomization_test": first["randomization_test"],
            "n_scenarios": first["n_scenarios"],
            "n_decoder_draws": first["n_decoder_draws"],
            "prompt_style": prompt_style,
            "forced_json_requirement": FORCED_JSON_COMPLETE_RULE_MAP,
        },
        "provenance": {
            "complete": complete,
            "campaigns": campaigns,
            "models": order,
            "model_labels": {models[label]["model_name"]: label for label in order},
            **{field: first.get(field) for field in SHARED_DESIGN_FIELDS},
            "rules_file": {"name": Path(rules_path).name, "sha256": rules_sha256, "n_rules": len(rules)},
            "pairing": "scenario_id x decoder draw within one accepted bundle; rules joined by parsed number",
            "definition": definition(prompt_style),
        },
    }
    validate_as_written(golden, validate_payload, "the golden")
    return golden, cells


CSV_COLUMNS = (
    "model", "rule", "title", "n_scenarios", "n_decoder_draws",
    "rate_alone", "rate_in_context", "delta", "yes_difference", "ci_low", "ci_high",
    "p_value", "p_value_is_floor", "p_holm", "significant_holm", "holm_mc_margin",
    "n_informative_scenarios", "few_informative", "degenerate",
    "phi_alone", "phi_in_context", "phi_delta",
    *(f"share_alone_{a}_in_context_{b}" for a in LABELS for b in LABELS),
    *(f"full_set_{label}_delta_contribution" for label in LABELS),
    "named_preempted_delta_contribution",
    "metric", "inference_method", "test_method",
)


def csv_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return one row per (model, rule) in :data:`CSV_COLUMNS` order, models in paper order."""
    settings = payload["settings"]
    rows = []
    for label in display_order(payload["models"]):
        for row in payload["models"][label]["effects"]:
            flat = {
                **row,
                "model": label,
                "metric": payload["metric"],
                "inference_method": settings["inference_method"],
                "test_method": settings["test_method"],
                **{f"share_alone_{a}_in_context_{b}": row["transitions"][a][b] for a in LABELS for b in LABELS},
                **{f"full_set_{v}_delta_contribution": row["by_verdict"][v] for v in LABELS},
                # Empty for a style that asks for no preempted_rules: not applicable, not 0.
                "named_preempted_delta_contribution": (row["named_preempted"] or {}).get("delta_contribution"),
            }
            rows.append({column: flat[column] for column in CSV_COLUMNS})
    return rows
