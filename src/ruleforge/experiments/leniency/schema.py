from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Sequence, Tuple

from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.evaluate.inference import holm_adjust
from ruleforge.evaluate.report import FLAT, TRANSITION_LABELS
from ruleforge.experiments.checks import require

DATA_VERSION = 5
GOLDEN_SOURCE = "stochastic_campaign"
#: Preemption is ignored wherever a judgment exists; under complete_rule_map every rule carries one.
METRIC = FLAT
#: The three judgment labels, in the order every matrix and array uses.
LABELS = TRANSITION_LABELS
#: Transition unit of a (scenario, rule) cell: over the S x S (alone, in-context) draw pairs of that scenario, mean_i
#: p_i(a) q_i(b).
TRANSITION_METHOD = "independent_draw_expected_shares_v1"
#: The reference each side is read against: two distinct draws of one side.
AGREEMENT_METHOD = "distinct_draw_pair_agreement_v1"
#: Below this many scenarios with a non-zero draw-mean difference a row is flagged as few-informative.
FEW_INFORMATIVE_BELOW = 10
IN_CONTEXT, ALONE = "in_context", "alone"


class RuleContextError(CampaignEstimatesError):
    """The campaign cannot make the leniency golden as stated."""


_require = partial(require, error=RuleContextError)


def check_holm(rows: Sequence[Mapping[str, Any]], *, alpha: float, where: str) -> None:
    """Refuse rows (one model's rules) whose stored columns are not Holm's."""
    fresh = holm_adjust([{"p_value": float(row["p_value"])} for row in rows], alpha=alpha)
    for row, recomputed in zip(rows, fresh):
        stored = float(row["p_holm"])
        _require(
            [
                (
                    abs(stored - recomputed["p_holm"]) <= 1e-12,
                    f"stored p_holm {stored} but Holm over the model's {len(rows)} rules "
                    f"gives {recomputed['p_holm']}",
                ),
                (
                    bool(row["significant_holm"]) == recomputed["significant_holm"],
                    "stored significant_holm disagrees with the recomputation",
                ),
            ],
            f"{where} {row['rule_id']}",
        )


def _sign(value: float) -> int:
    return int(value > 0) - int(value < 0)


def direction(row: Mapping[str, Any]) -> int:
    """Return -1, 0 or +1 (lower, tied or higher in context) from the exact count, never the sign of ``delta``, whose
    rounding would break ties."""
    return _sign(int(row["yes_difference"]))


def direction_counts(rows: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int, int, int]:
    """Rows lower, higher and tied by :func:`direction`, then the Holm-significant ones lower and higher."""
    signs = [direction(row) for row in rows]
    significant = [sign for sign, row in zip(signs, rows) if row["significant_holm"]]
    return signs.count(-1), signs.count(1), signs.count(0), significant.count(-1), significant.count(1)


#: A model summary's :func:`direction_counts`, by name.
DIRECTION_COUNT_FIELDS = ("n_rules_lower", "n_rules_higher", "n_rules_tied", "n_holm_lower", "n_holm_higher")
