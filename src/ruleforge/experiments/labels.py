from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional

from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.prompts.styles import (
    ENCOURAGED,
    FORCED,
    FORCED_GRPO,
    NAIVE_HIERARCHY,
    NAIVE_RULES_ABOVE,
)

#: Campaign judge name (``judge_model.model_name``) to golden label; an unlisted judge is refused, not guessed.
CAMPAIGN_MODEL_LABELS = {
    "Olmo-3-32B-Think": "Olmo3-32B",
    "gemma-4-31B-it": "gemma4-31b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-6.1-sol": "gpt-6.1-sol",
    "claude-haiku-5-5": "claude-haiku-5-5",
    "Muse-Glimmer-30B": "muse-glimmer-30b",
    "Qwen3-8B": "Qwen3-8B",
}


def campaign_label(model_name: str) -> str:
    """Return the golden label of a campaign judge; raise for one not listed."""
    try:
        return CAMPAIGN_MODEL_LABELS[model_name]
    except KeyError:
        raise CampaignEstimatesError(
            f"no golden label for judge {model_name!r}; add it to CAMPAIGN_MODEL_LABELS "
            "(and MODEL_DISPLAY_LABELS) rather than letting a new model land unnamed"
        ) from None


LUNA_LABEL = "gpt-5.6-luna"
HAIKU_LABEL = "claude-haiku-5-5"
GLIMMER_LABEL = "muse-glimmer-30b"
#: Golden label to the name a figure or table prints, in paper order.
MODEL_DISPLAY_LABELS = {
    "Olmo3-32B": "Olmo-3-32B",
    "gemma4-31b": "Gemma-4-31B",
    "gpt-oss-120b": "GPT-OSS-120B",
    GLIMMER_LABEL: "Muse-Glimmer-30B",
    "Qwen3-8B": "Qwen3-8B",
    LUNA_LABEL: "GPT-5.6 Luna",
    HAIKU_LABEL: "Claude Haiku 5.5",
    "gpt-6.1-sol": "GPT-6.1 Sol",
}


def display_order(labels: Iterable[str]) -> List[str]:
    """Sort models in paper order (`MODEL_DISPLAY_LABELS`), then alphabetically."""
    rank = {label: index for index, label in enumerate(MODEL_DISPLAY_LABELS)}
    return sorted(labels, key=lambda label: (rank.get(label, len(rank)), label))


#: The paper's three prompt slots, in paper order. The paper prints its first arm as "naive"; that arm ran the
#: naive-rules-above prompt.
NAIVE_SLOT = "naive"
SLOTS = (NAIVE_SLOT, ENCOURAGED, FORCED)
STYLE_LABELS = {
    NAIVE_SLOT: "naive",
    ENCOURAGED: "encouraged",
    FORCED: "forced",
    FORCED_GRPO: "forced (GRPO)",
    NAIVE_RULES_ABOVE: "naive (rules above)",
    NAIVE_HIERARCHY: "naive (hierarchy shown)",
}
PAPER_STYLE_LABELS = {slot: STYLE_LABELS[slot] for slot in SLOTS}


#: Rule labels in a forced judgment look like "Rule 42" (the space is optional); artifacts key the same rule as
#: "openai_rules42". `(?:Meta-)?` collapses "Meta-Rule 5" and "Rule 5" to one number.
_RULE_LABEL = re.compile(r"^\s*(?:Meta-)?Rule\s*(\d+)\s*$", re.IGNORECASE)


def rule_label_number(label: Any) -> Optional[int]:
    """Return the rule number in a judgment's label, or None if it names no rule."""
    match = _RULE_LABEL.match(str(label))
    return int(match.group(1)) if match else None
