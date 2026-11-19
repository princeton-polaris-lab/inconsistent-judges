from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.identity import FULL_RULESET_COMPONENT, ConditionSpec
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT, HIERARCHY_STYLES
from ruleforge.rulesets.levels import RuleLevels
from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.tasks.base import DEFAULT_TASK, TaskSettings
from ruleforge.tasks.registry import load_task


def rule_sets_by_digest(paths: Sequence[Path | str]) -> dict[str, Path]:
    """Return ruleset files by SHA-256; each path is a file, or a directory giving every ``*.jsonl`` under it."""
    found: dict[str, Path] = {}
    for given in paths:
        path = Path(given)
        for file in (sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]):
            found[sha256_file(file)] = file
    return found


def prompt_renderer(
    spec: ConditionSpec, rules: Sequence[Mapping[str, Any]], *, levels: Optional[RuleLevels] = None,
) -> Callable[[Mapping[str, Any], str], dict[str, str]]:
    """Return ``(scenario row, component id) -> prompt`` as a Model Spec condition sends it, with its levels."""
    task = load_task(DEFAULT_TASK)
    context = HIERARCHY_RULE_CONTEXT if spec.prompt_style in HIERARCHY_STYLES else spec.individual_rule_context
    settings = TaskSettings(spec.prompt_style, context, spec.judge_type)
    by_id = levels.by_id if context == HIERARCHY_RULE_CONTEXT else None
    rules = [dict(rule) for rule in rules]
    full_set = rule_set_text(rules, context, by_id)
    variants, _ = build_rule_variants(rules, context, by_id)
    texts = {rule_id: (text, context) for rule_id, text, context in variants}

    def render(row: Mapping[str, Any], component_id: str) -> dict[str, str]:
        text, context = (full_set, None) if component_id == FULL_RULESET_COMPONENT else texts[component_id]
        system_prompt = task.render(text, dict(row), settings, context_rules=context)
        return {"system_prompt": system_prompt, "user_prompt": ""}

    return render
