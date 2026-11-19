from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ruleforge.prompts.context import (
    IMPLICATED_RANGES_RULE_CONTEXT,
    IMPLICATED_RULE_CONTEXT,
    VALID_INDIVIDUAL_RULE_CONTEXTS,
)
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT
from ruleforge.rulesets.levels import require_known_levels, superior_ids
from ruleforge.rulesets.rules import (
    RULE_REFERENCE_TEXT_FIELDS,
    build_implicated_rules_context,
    build_rule_implication_map,
    format_rule,
    is_meta_rule_id,
    looks_like_rule_range,
)

logger = logging.getLogger(__name__)

IMPLICATED_RULES_FNAME = "implicated_rules.json"

# (rule id, the text the judge is shown, pre-formatted context rules or None)
RuleVariant = tuple[str, str, List[str] | None]


def _checked_levels(
    rules: List[Dict[str, Any]], individual_rule_context: str, levels: Mapping[str, str] | None
) -> Mapping[str, str] | None:
    if (individual_rule_context == HIERARCHY_RULE_CONTEXT) != (levels is not None):
        raise ValueError(
            f"individual_rule_context {individual_rule_context!r} "
            + ("judges each rule with its superiors and needs each rule's level"
               if levels is None else "reads no rule levels; only 'superiors' does")
        )
    if levels is not None:
        require_known_levels((str(rule.get("id", "")).strip() for rule in rules), levels)
    return levels


def rule_set_text(
    rules: List[Dict[str, Any]],
    individual_rule_context: str = "isolated",
    levels: Mapping[str, str] | None = None,
) -> str:
    """Return the full-ruleset prompt's rule text, one rule per line; under ``superiors`` each line carries its
    level."""
    levels = _checked_levels(rules, individual_rule_context.lower(), levels)
    return "\n".join(
        format_rule(rule, level=None if levels is None else levels[str(rule.get("id", "")).strip()])
        for rule in rules
    )


def build_rule_variants(
    rules: List[Dict[str, Any]],
    individual_rule_context: str,
    levels: Mapping[str, str] | None = None,
) -> tuple[List[RuleVariant], Dict[str, Any] | None]:
    """Return the non-meta rules' prompt variants in file order and the implication map (``implicated`` and
    ``implicated_ranges``); ``superiors`` needs ``levels``, a rule's context being the rules that outrank it."""
    individual_rule_context = individual_rule_context.lower()
    if individual_rule_context not in VALID_INDIVIDUAL_RULE_CONTEXTS:
        raise ValueError(
            f"Invalid individual_rule_context: {individual_rule_context!r}"
        )
    levels = _checked_levels(rules, individual_rule_context, levels)
    if levels is not None:
        order = [str(rule.get("id", "")).strip() for rule in rules]
        tagged = {rule_id: format_rule(rule, level=levels[rule_id]) for rule_id, rule in zip(order, rules)}
        return [
            (rule_id, tagged[rule_id], [tagged[other] for other in superior_ids(rule_id, levels, order)] or None)
            for rule_id in order
            if not is_meta_rule_id(rule_id)
        ], None
    implication_map = None
    rules_by_id: Dict[str, Dict[str, Any]] = {}
    if individual_rule_context in (IMPLICATED_RULE_CONTEXT, IMPLICATED_RANGES_RULE_CONTEXT):
        implication_map = build_rule_implication_map(
            rules, expand_ranges=individual_rule_context == IMPLICATED_RANGES_RULE_CONTEXT)
        rules_by_id = {str(rule.get("id", "")).strip(): rule for rule in rules}
    variants: List[RuleVariant] = []
    for rule in rules:
        rule_id = str(rule.get("id", "")).strip()
        if is_meta_rule_id(rule_id):
            continue
        context_rules = None
        if implication_map is not None:
            context_rules = build_implicated_rules_context(
                rule_id, implication_map, rules_by_id
            )
        variants.append((rule_id, format_rule(rule), context_rules))
    return variants, implication_map


def warn_unexpanded_ranges(rules: List[Dict[str, Any]], individual_rule_context: str) -> bool:
    """Log one warning, and return True, when a new run's ``implicated`` context would read something like a range in
    ``rules``, well formed or not, as its first rule; it never refuses."""
    if individual_rule_context != IMPLICATED_RULE_CONTEXT:
        return False
    named = [str(rule.get("id", "")).strip() for rule in rules
             if any(looks_like_rule_range(rule.get(field)) for field in RULE_REFERENCE_TEXT_FIELDS)]
    if named:
        logger.warning(f"individual_rule_context {IMPLICATED_RULE_CONTEXT!r} reads each rule range of {named} as its "
                       f"first rule, as the paper's records did; {IMPLICATED_RANGES_RULE_CONTEXT!r} reads it as every "
                       "rule in it")
    return bool(named)


def write_implicated_sidecar(
    directory: Path,
    individual_rule_context: str,
    implication_map: Dict[str, Any] | None,
) -> Path | None:
    """Write the implication map beside a run's evaluations; return its path, or None when there is no map."""
    if implication_map is None:
        return None
    sidecar_path = Path(directory) / IMPLICATED_RULES_FNAME
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(
            {
                "individual_rule_context": individual_rule_context,
                "text_fields": list(RULE_REFERENCE_TEXT_FIELDS),
                "implicated_rules": implication_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote implicated-rules sidecar to {sidecar_path}")
    return sidecar_path
