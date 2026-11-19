from __future__ import annotations

from typing import List

from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT

# Alone, with the rules it references (a range read as its first rule), with the rules it references and every rule
# of each range it names, or with the rules that outrank it.
IMPLICATED_RULE_CONTEXT, IMPLICATED_RANGES_RULE_CONTEXT = "implicated", "implicated_ranges"
VALID_INDIVIDUAL_RULE_CONTEXTS = {"isolated", IMPLICATED_RULE_CONTEXT, IMPLICATED_RANGES_RULE_CONTEXT,
                                  HIERARCHY_RULE_CONTEXT}

IMPLICATED_RULES_HEADER = "# Referenced Rules (Context Only)"

_CONTEXT_RULES_INSTRUCTIONS = (
    "The rules listed below are provided as context for interpreting the rule "
    "set under evaluation. Use them however the rule set under "
    "evaluation makes them relevant. Your final "
    "judgment must state only whether the rule set under evaluation is "
    "violated."
)


def build_context_rules_section(
    context_rules: List[str] | None, header: str = IMPLICATED_RULES_HEADER
) -> str:
    """Render a rule's context block under ``header``, or "" when there is none."""
    if not context_rules:
        return ""
    joined_rules = "\n".join(context_rules)
    return (
        f"\n{header}\n"
        f"{_CONTEXT_RULES_INSTRUCTIONS}\n\n"
        f"{joined_rules}\n"
    )
