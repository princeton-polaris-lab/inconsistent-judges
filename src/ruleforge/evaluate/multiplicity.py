from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

ALPHA = 0.05
#: A Holm decision resting on a step within this many Monte Carlo standard errors of alpha is flagged
#: ``holm_mc_margin``; a flag, never a change of decision.
HOLM_MC_MARGIN_SE = 3.0


def adjust_for_multiplicity(
    table: Sequence[Mapping[str, Any]],
    method: str = "holm",
    alpha: float = ALPHA,
) -> list[dict[str, Any]]:
    """Return a copy of each row with ``p_adjusted``, ``adjustment_method`` and ``significant_adjusted`` under any
    ``multipletests`` method (``holm``, ``fdr_bh``, ...)."""
    from statsmodels.stats.multitest import multipletests

    rows = [dict(row) for row in table]
    if not rows:
        return rows
    rejected, adjusted, _, _ = multipletests(
        [row["p_value"] for row in rows], alpha=alpha, method=method
    )
    for row, p_adjusted, reject in zip(rows, adjusted, rejected):
        row["p_adjusted"] = float(p_adjusted)
        row["adjustment_method"] = method
        row["significant_adjusted"] = bool(reject)
    return rows


def holm_flags(rows: Sequence[Mapping[str, Any]], *, alpha: float) -> list[dict[str, bool]]:
    """Return per row of one Holm family ``p_value_is_floor`` and ``holm_mc_margin``: whether another permutation seed
    could flip the decision."""
    p_values = [float(row["p_value"]) for row in rows]
    significant = [row["significant_adjusted"]
                   for row in adjust_for_multiplicity([{"p_value": p} for p in p_values], method="holm", alpha=alpha)]
    flags: list[dict[str, bool]] = [{} for _ in rows]
    # Some step so far is near alpha, and every failing step so far is.
    any_near, failing_near = False, True
    for step, i in enumerate(sorted(range(len(rows)), key=p_values.__getitem__)):
        p, n_permutations = p_values[i], int(rows[i]["n_permutations"])
        multiplier = len(rows) - step
        product, se = multiplier * p, multiplier * math.sqrt(p * (1 - p) / n_permutations)
        near = abs(product - alpha) <= HOLM_MC_MARGIN_SE * se
        any_near = any_near or near
        if p > alpha / multiplier:
            failing_near = failing_near and near
        flags[i] = {
            "p_value_is_floor": p <= (1 + 1e-9) / (n_permutations + 1),
            "holm_mc_margin": any_near if significant[i] else failing_near,
        }
    return flags
