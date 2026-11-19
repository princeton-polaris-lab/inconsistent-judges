from __future__ import annotations

from typing import Any, Mapping, Sequence

from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.evaluate.report import (
    FLAT,
)
from ruleforge.evaluate.multiplicity import ALPHA, HOLM_MC_MARGIN_SE, adjust_for_multiplicity, holm_flags
from ruleforge.experiments.labels import NAIVE_SLOT, SLOTS, STYLE_LABELS
from ruleforge.prompts.styles import ENCOURAGED, FORCED, NAIVE_RULES_ABOVE


#: The prompt style each slot's arm ran (the slots are the paper's, ``labels.SLOTS``).
SLOT_STYLES = {NAIVE_SLOT: NAIVE_RULES_ABOVE, ENCOURAGED: ENCOURAGED, FORCED: FORCED}

#: Every ordered pair the table reports.
TABLE_CONTRASTS = ((NAIVE_SLOT, ENCOURAGED), (NAIVE_SLOT, FORCED), (ENCOURAGED, FORCED))


FORCED_B = "forced-b"


#: The table's two families, each over every contrast of its table.
BH_METHOD = "fdr_bh"
HOLM_METHOD = "holm"
TABLE_ADJUSTMENT_FAMILY = "all_table_style_contrasts"
#: Every column :func:`add_table_multiplicity` writes; :func:`validate_table` strips these before recomputing.
TABLE_MULTIPLICITY_KEYS = (
    "p_bh", "adjustment_method", "significant", "significant_bh",
    "p_holm", "significant_holm", "holm_mc_margin", "adjustment_family", "adjustment_family_n",
)
HOLM_MC_MARGIN_NOTE = (
    f"holm_mc_margin marks a Holm decision another permutation seed could flip (a step within {HOLM_MC_MARGIN_SE:g} "
    "Monte Carlo standard errors of alpha)"
)


#: The quantity every table rate is (the full-set judgment against the strongest single-rule judgment, binary), and the
#: figure's population and bars in drawn order.
TABLE_METRIC = FLAT
BAR_POPULATION = "per_bar"
PROMPT_BARS = (*SLOTS, FORCED_B)
#: The golden's schema version and source, which every producer writes and a ``--from_golden`` re-render checks.
TABLE_SCHEMA_VERSION = 2
TABLE_SOURCE = "stochastic_campaign"
#: The keys of each judge's same-call against separate-call comparison copied into ``diagnostics.separate_vs_same_call``.
SAME_VS_SEPARATE_KEYS = ("rate_a", "rate_b", "estimate", "ci_low", "ci_high", "p_value", "n_permutations")


def table_row(
    label: str,
    model_name: str,
    baseline: str,
    treatment: str,
    contrast: Mapping[str, Any],
    *,
    test_method: str,
) -> dict[str, Any]:
    """Return one table row, before :func:`add_table_multiplicity`: ``baseline`` minus ``treatment`` of one judge."""
    difference = float(contrast["estimate"])
    return {
        "label": label,
        "model_name": model_name,
        "baseline": baseline,
        "treatment": treatment,
        "baseline_label": STYLE_LABELS[baseline],
        "treatment_label": STYLE_LABELS[treatment],
        "n_scenarios": int(contrast["n_scenarios"]),
        "n_decoder_draws": int(contrast["n_decoder_draws_a"]),
        "baseline_rate": float(contrast["rate_a"]),
        "treatment_rate": float(contrast["rate_b"]),
        # Positive means the treatment prompt is the more consistent one.
        "difference": difference,
        "ci_low": float(contrast["ci_low"]),
        "ci_high": float(contrast["ci_high"]),
        "sd": float(contrast["sd"]),
        "interval_basis": contrast["interval_basis"],
        "two_stage": contrast.get("two_stage"),
        "p_value": float(contrast["p_value"]),
        "n_permutations": int(contrast["n_permutations"]),
        "test_method": test_method,
        "inference_method": contrast["inference_method"],
        "bootstrap_seed": contrast.get("bootstrap_seed"),
        "grid_digest_a": contrast.get("grid_digest_a"),
        "grid_digest_b": contrast.get("grid_digest_b"),
        "improvement": bool(difference > 0),
    }


def rate_bar(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return one figure bar from a condition's rate summary: ``{mean, ci_low, ci_high, n, degenerate}``."""
    low, high = float(summary["ci_low"]), float(summary["ci_high"])
    return {
        "mean": float(summary["estimate"]),
        "ci_low": low,
        "ci_high": high,
        "n": int(summary["n_scenarios"]),
        "degenerate": bool(low == high),
    }


def add_table_multiplicity(rows: Sequence[Mapping[str, Any]], *, alpha: float = ALPHA) -> list[dict[str, Any]]:
    """Add both families over every row of a table (BH in ``p_bh``, Holm in ``p_holm`` with its margin flag), with
    Holm reported: ``adjustment_method`` names it and ``significant`` is its decision."""
    bh = adjust_for_multiplicity(rows, method=BH_METHOD, alpha=alpha)
    holm = adjust_for_multiplicity(rows, method=HOLM_METHOD, alpha=alpha)
    out: list[dict[str, Any]] = []
    for row_bh, row_holm, flags in zip(bh, holm, holm_flags(rows, alpha=alpha)):
        row = dict(row_bh)
        row["p_bh"] = row.pop("p_adjusted")
        row["significant_bh"] = row.pop("significant_adjusted")
        row["p_holm"] = float(row_holm["p_adjusted"])
        row["significant_holm"] = bool(row_holm["significant_adjusted"])
        row["holm_mc_margin"] = flags["holm_mc_margin"]
        row["adjustment_method"] = HOLM_METHOD
        row["significant"] = row["significant_holm"]
        row["adjustment_family"] = TABLE_ADJUSTMENT_FAMILY
        row["adjustment_family_n"] = len(rows)
        out.append(row)
    return out


def validate_table(table: Sequence[Mapping[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    """Refuse a table whose stored families disagree with a recomputation at ``alpha``."""
    if not table:
        raise CampaignEstimatesError("the table has no rows")
    stripped = [{k: v for k, v in row.items() if k not in TABLE_MULTIPLICITY_KEYS} for row in table]
    fresh = add_table_multiplicity(stripped, alpha=alpha)
    for stored, recomputed in zip(table, fresh):
        for key in ("p_bh", "p_holm"):
            if abs(float(stored[key]) - float(recomputed[key])) > 1e-9:
                raise CampaignEstimatesError(
                    f"{stored['label']} {stored['baseline']}->{stored['treatment']}: stored "
                    f"{key}={stored[key]} but recomputing over the table gives {recomputed[key]}"
                )
        for key in ("significant_bh", "significant_holm", "holm_mc_margin"):
            if bool(stored[key]) != bool(recomputed[key]):
                raise CampaignEstimatesError(f"{stored['label']}: stored {key} disagrees with the recomputation")
        if int(stored["adjustment_family_n"]) != len(table):
            raise CampaignEstimatesError("adjustment_family_n is not the table's row count")
    return [dict(row) for row in table]
