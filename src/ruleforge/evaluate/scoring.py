from __future__ import annotations

from typing import Dict, List

PHI_MAP: Dict[str, float] = {"yes": 1.0, "unsure": 0.5, "no": 0.0}
# In the order every class-indexed feature uses: no, unsure, yes.
PHI_CLASSES = (0.0, 0.5, 1.0)


def calculate_inconsistency(
    full_judgment: str,
    rule_judgments: List[str],
) -> Dict[str, float]:
    """Return ``{"contradiction", "penalty"}`` comparing the full-set judgment to the strongest per-rule judgment;
    raise ValueError with no rules."""
    strongest = max_rule_phi(rule_judgments)
    phi_full = PHI_MAP[full_judgment]
    return {
        "contradiction": float(phi_full != strongest),
        "penalty": abs(phi_full - strongest),
    }


def max_rule_phi(rule_judgments: List[str]) -> float:
    """Return the strongest per-rule judgment's phi; raise ValueError with no rules."""
    if not rule_judgments:
        raise ValueError("no per-rule judgments to compare the full-set judgment with")
    return max(PHI_MAP[judgment] for judgment in rule_judgments)
