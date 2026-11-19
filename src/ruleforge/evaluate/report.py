from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ruleforge.core.errors import RuleForgeError
from ruleforge.evaluate.scoring import PHI_MAP, calculate_inconsistency

logger = logging.getLogger(__name__)

FLAT = "flat"
HIERARCHICAL = "hierarchical"
VALID_METRICS = (FLAT, HIERARCHICAL)

# One per dropped scenario, so these sum with ``len(rows)`` to the input count.
SCENARIO_DROP_REASONS = (
    "unparseable_full_set",
    "no_applicability",
    "unparseable_rule",
    "no_rule_evaluations",
)

# Judge calls, not scenarios: never part of the scenario accounting.
CALL_DROP_COUNTERS = ("unparseable_rule_calls",)


class ScoringError(ValueError, RuleForgeError):
    """Raised when a frozen row cannot be scored."""


def _validate_metric(metric: str) -> None:
    if metric not in VALID_METRICS:
        raise ScoringError(f"unknown metric {metric!r}; valid: {list(VALID_METRICS)}")


def _rule_judgments(row: Mapping[str, Any], *, applicable_only: bool) -> list[str]:
    # A missing rule list must not score as a consistent judge.
    if "rules" not in row:
        raise ScoringError("row has no 'rules' list")
    judgments: list[str] = []
    for rule in row["rules"]:
        if not isinstance(rule, Mapping):
            raise ScoringError(f"malformed rule entry: {rule!r}")
        if applicable_only:
            # Strict: raw judge text such as ``yes.`` must not count as applicable.
            flag = str(rule.get("not_applicable", "no")).strip().lower()
            if flag not in ("yes", "no"):
                raise ScoringError(f"unparseable not_applicable value {flag!r}")
            if flag == "yes":
                continue
        judgment = rule.get("judgment")
        if judgment not in PHI_MAP:
            raise ScoringError(f"unparseable rule judgment {judgment!r}")
        judgments.append(judgment)
    return judgments


def _full_judgment(row: Mapping[str, Any]) -> str:
    judgment = row.get("full_set_judgment")
    if judgment not in PHI_MAP:
        raise ScoringError(f"unparseable full-set judgment {judgment!r}")
    return judgment


def _score_applicable(full_judgment: str, applicable: list[str], n_rules: int) -> dict[str, float]:
    if n_rules and not applicable:
        return {"contradiction": 0.0, "penalty": 0.0}
    return calculate_inconsistency(full_judgment, applicable)


def score_row(row: Mapping[str, Any], metric: str) -> dict[str, float]:
    """Return ``{"contradiction": 0.0|1.0, "penalty": float}`` for one row; ``metric`` selects whether preempted rules
    are dropped."""
    _validate_metric(metric)
    return _score_applicable(
        _full_judgment(row),
        _rule_judgments(row, applicable_only=metric == HIERARCHICAL),
        len(row.get("rules", ())),
    )


def flat_contradiction(row: Mapping[str, Any]) -> float:
    """Return the flat contradiction, with no preemption filtering."""
    return score_row(row, FLAT)["contradiction"]


def anchor_stratum(row: Mapping[str, Any], metric: str = FLAT) -> str:
    """Return a row's anchor bucket, the strongest per-rule judgment: ``yes``, ``unsure`` or ``no``."""
    _validate_metric(metric)
    judgments = _rule_judgments(row, applicable_only=metric == HIERARCHICAL)
    return max(judgments, key=lambda judgment: PHI_MAP[judgment], default="no")


# Row and column order of every transition matrix, strongest judgment first.
TRANSITION_LABELS = ("yes", "unsure", "no")


def empty_transitions() -> dict[str, dict[str, int]]:
    """Return a zero 3x3 transition matrix in ``TRANSITION_LABELS`` order."""
    return {b: {t: 0 for t in TRANSITION_LABELS} for b in TRANSITION_LABELS}


def transition_summary(matrix: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Return total, diagonal count, stability fraction, chance floor and kappa; quote the fraction, floor and kappa
    together."""
    total = sum(matrix[b][t] for b in TRANSITION_LABELS for t in TRANSITION_LABELS)
    unchanged = sum(matrix[label][label] for label in TRANSITION_LABELS)
    observed = unchanged / total if total else 0.0
    # Marginals over TRANSITION_LABELS: a matrix with an extra key would otherwise inflate p_e without inflating
    # ``total``.
    row_totals = {
        label: sum(matrix[label][t] for t in TRANSITION_LABELS)
        for label in TRANSITION_LABELS
    }
    column_totals = {
        label: sum(matrix[base][label] for base in TRANSITION_LABELS)
        for label in TRANSITION_LABELS
    }
    expected = (
        sum(row_totals[label] * column_totals[label] for label in TRANSITION_LABELS)
        / (total * total)
        if total
        else 0.0
    )
    kappa = (
        (observed - expected) / (1.0 - expected) if total and expected != 1.0 else 0.0
    )
    return {
        "n": total,
        "unchanged": unchanged,
        "unchanged_fraction": observed,
        "chance_agreement": expected,
        "kappa": kappa,
    }


def judgment_transitions(
    base_rows: Mapping[str, Mapping[str, Any]],
    tuned_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return base -> tuned transition counts (``per_rule``, ``anchors``, ``full_set``) over the scenarios both runs
    scored, raising on differing rule sets, a repeated ``rule_id``, a ruleless row or no common scenario."""
    anchors = empty_transitions()
    full_set = empty_transitions()
    per_rule = empty_transitions()
    base_marginals = {
        key: {label: 0 for label in TRANSITION_LABELS}
        for key in ("anchors", "full_set", "per_rule")
    }
    per_rule_flips: dict[str, int] = {}
    hashes = sorted(set(base_rows) & set(tuned_rows))
    if not hashes:
        raise ScoringError(
            f"no common scenario between the runs ({len(base_rows)} base "
            f"rows, {len(tuned_rows)} tuned) -- nothing to pair"
        )
    for digest in hashes:
        base, tuned = base_rows[digest], tuned_rows[digest]
        if not base["rules"] or not tuned["rules"]:
            raise ScoringError(f"{digest}: a row has no rule judgments")
        base_anchor = anchor_stratum(base)
        base_full_set = _full_judgment(base)
        anchors[base_anchor][anchor_stratum(tuned)] += 1
        full_set[base_full_set][_full_judgment(tuned)] += 1
        base_marginals["anchors"][base_anchor] += 1
        base_marginals["full_set"][base_full_set] += 1
        base_rules = {rule["rule_id"]: rule["judgment"] for rule in base["rules"]}
        tuned_rules = {rule["rule_id"]: rule["judgment"] for rule in tuned["rules"]}
        if len(base_rules) != len(base["rules"]) or len(tuned_rules) != len(
            tuned["rules"]
        ):
            raise ScoringError(f"{digest}: duplicate rule_id within a row")
        if set(base_rules) != set(tuned_rules):
            raise ScoringError(f"{digest}: base and tuned judged different rule sets")
        for rule_id, base_judgment in base_rules.items():
            tuned_judgment = tuned_rules[rule_id]
            per_rule[base_judgment][tuned_judgment] += 1
            base_marginals["per_rule"][base_judgment] += 1
            if base_judgment != tuned_judgment:
                per_rule_flips[rule_id] = per_rule_flips.get(rule_id, 0) + 1
    return {
        "n_common": len(hashes),
        "n_base": len(base_rows),
        "n_tuned": len(tuned_rows),
        "anchors": anchors,
        "full_set": full_set,
        "per_rule": per_rule,
        # Counted directly from the base records, so a consumer can detect an accidental transpose.
        "base_marginals": base_marginals,
        "summaries": {
            "anchors": transition_summary(anchors),
            "full_set": transition_summary(full_set),
            "per_rule": transition_summary(per_rule),
        },
        "per_rule_flips": dict(
            sorted(per_rule_flips.items(), key=lambda item: (-item[1], item[0]))
        ),
    }


def build_per_scenario_rows(
    scenarios: Sequence[Mapping[str, Any]],
    model_name: str,
    judge_type: str,
    *,
    ruleset_key: str,
    require_not_applicable: bool = True,
    applicability_judge_type: str | None = None,
    drop_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return per-scenario rows, filling ``drop_counts`` with why scenarios were excluded;
    ``applicability_judge_type`` names the ``not_applicable`` flags' strategy, and ``require_not_applicable=False``
    keeps a failed preemption pass."""
    from ruleforge.core.digests import row_hash
    from ruleforge.prompts.parse import read_label

    if applicability_judge_type is None:
        applicability_judge_type = judge_type

    drops = drop_counts if drop_counts is not None else {}
    for reason in (*SCENARIO_DROP_REASONS, *CALL_DROP_COUNTERS):
        drops.setdefault(reason, 0)

    rows: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        model_evals = scenario.get("evaluations", {}).get(model_name, {})
        full_judgment = read_label(
            model_evals.get(ruleset_key, {}).get(judge_type)
        )
        if full_judgment is None:
            drops["unparseable_full_set"] += 1
            continue

        not_applicable = scenario.get("not_applicable", {}).get(model_name, {})
        if require_not_applicable and not not_applicable:
            drops["no_applicability"] += 1
            continue

        rule_rows: list[dict[str, Any]] = []
        n_unparseable = 0
        for rule_id, by_judge in model_evals.items():
            if rule_id == ruleset_key:
                continue
            judgment = read_label(by_judge.get(judge_type))
            if judgment is None:
                n_unparseable += 1
                continue
            flag = read_label(
                not_applicable.get(rule_id, {}).get(applicability_judge_type)
            )
            rule_rows.append(
                {
                    "rule_id": rule_id,
                    "judgment": judgment,
                    "phi": PHI_MAP[judgment],
                    "not_applicable": flag if flag is not None else "no",
                }
            )

        if n_unparseable:
            # Any unparseable per-rule call drops the scenario: no field records truncation, so a truncated row would
            # read as a smaller ruleset.
            logger.warning(
                "Scenario %d has %d unparseable per-rule judgment(s); "
                "excluding it rather than scoring a truncated rule set.",
                index,
                n_unparseable,
            )
            drops["unparseable_rule"] += 1
            drops["unparseable_rule_calls"] += n_unparseable
            continue

        if not rule_rows:
            logger.warning(
                "Scenario %d has no per-rule evaluations at all; excluding it "
                "rather than scoring it as consistent.",
                index,
            )
            drops["no_rule_evaluations"] += 1
            continue

        applicable = [rule for rule in rule_rows if rule["not_applicable"] != "yes"]
        summary = _score_applicable(
            full_judgment, [rule["judgment"] for rule in applicable], len(rule_rows)
        )
        scenario_body = scenario.get("scenario", {})
        rows.append(
            {
                "prompt_response_hash": row_hash(scenario),
                "scenario_index": index,
                "prompt": scenario_body.get("prompt", ""),
                "response": scenario_body.get("response", ""),
                "ruleset_key": ruleset_key,
                "strategy": judge_type,
                "full_set_judgment": full_judgment,
                "full_set_phi": PHI_MAP[full_judgment],
                "rules": rule_rows,
                "applicable_rules": [rule["rule_id"] for rule in applicable],
                "applicable_rule_count": len(applicable),
                "max_applicable_rule_phi": max(
                    (rule["phi"] for rule in applicable), default=0.0
                ),
                # False means ``contradiction`` holds the flat number, so exclude this row from any preemption-aware
                # aggregate.
                "has_applicability": bool(not_applicable),
                "contradiction": summary["contradiction"],
                "regularization_penalty": summary["penalty"],
            }
        )
    return rows


def _is_exhausted(
    n_rows: int, n_input_scenarios: Optional[int], accounted: int
) -> bool:
    # ``incomplete`` must mean there is more to try: a judge that always runs out of tokens would otherwise resume
    # forever.
    if not n_rows:
        # With a dead backend every full-set judgment is None, and the accounting would call a run that scored nothing
        # attempted.
        return False
    if not n_input_scenarios:
        return False
    if n_rows >= n_input_scenarios:
        # More rows than inputs means a different population.
        return False
    return accounted >= n_input_scenarios


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    n_input_scenarios: Optional[int] = None,
    drop_counts: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Return the summary payload (means, coverage, drops, ``status``: ``complete``, ``exhausted`` or ``incomplete``);
    zero is the best score, so the counts and ``status`` tell a run that scored nothing."""
    hierarchical_rows = [row for row in rows if row.get("has_applicability", True)]
    hierarchical = [float(row["contradiction"]) for row in hierarchical_rows]
    penalties = [float(row["regularization_penalty"]) for row in hierarchical_rows]
    flat = [flat_contradiction(row) for row in rows]

    def mean(values: Sequence[float]) -> Optional[float]:
        # None: nothing to average is not a score of zero.
        return float(np.mean(values)) if values else None

    coverage = (
        len(rows) / n_input_scenarios if n_input_scenarios else (1.0 if rows else None)
    )
    # len(rows): a row with no applicability data was still scored.
    accounted = len(rows) + sum(
        (drop_counts or {}).get(reason, 0) for reason in SCENARIO_DROP_REASONS
    )
    if coverage == 1.0:
        status = "complete"
    elif _is_exhausted(len(rows), n_input_scenarios, accounted):
        status = "exhausted"
    else:
        status = "incomplete"

    return {
        "n_scenarios": len(rows),
        "n_input_scenarios": n_input_scenarios,
        "n_hierarchical_scenarios": len(hierarchical_rows),
        "coverage": coverage,
        "status": status,
        "contradiction_arr": hierarchical,
        "regularization_penalty_arr": penalties,
        "flat_contradiction_arr": flat,
        "average_contradiction": mean(hierarchical),
        "average_regularization_penalty": mean(penalties),
        "average_flat_contradiction": mean(flat),
        "drops": dict(drop_counts) if drop_counts else None,
    }
