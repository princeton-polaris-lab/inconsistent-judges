from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ruleforge.core.record_names import identity_condition_id
from ruleforge.tasks.spec import ModelSpecTaskSpec
from ruleforge.prompts.styles import (
    FORCED_GRPO,
    NAIVE_RULES_ABOVE,
    no_interpretation_question_block,
)
from ruleforge.rulesets.rules import format_rule

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES = REPO_ROOT / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"

class _Spec:
    """The slice of a task spec `build_eval_prompt` reads."""

    interpretation_rules: dict = {}

    def __init__(self, style: str) -> None:
        self.reasoning_prompt_style = style


def test_naive_keeps_the_trailing_space_after_the_question():
    """One space, and the only thing between the question and the format block."""
    assert no_interpretation_question_block(NAIVE_RULES_ABOVE) == " \n"


def _artifact_root() -> Path:
    root = os.environ.get("RULEFORGE_ARTIFACT_ROOT")
    if not root:
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    return Path(root)


FINAL_CAMPAIGN = "sampling/final-2026-09"
FINAL_CAMPAIGN_JUDGE = "gemma-4-31B-it"
FROZEN_POPULATION = "populations/frozen_500.jsonl"
BUNDLES_CHECKED = 3


@pytest.mark.cluster
@pytest.mark.parametrize("style", [NAIVE_RULES_ABOVE, FORCED_GRPO])
def test_the_builder_reproduces_the_published_campaign_s_recorded_prompts(style):
    """The campaign recorded sha256(system prompt + RS + user prompt) for every component it sent."""
    import itertools

    from ruleforge.rulesets.variants import build_rule_variants
    from ruleforge.core.artifacts import prompt_sha256
    from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records

    root = _artifact_root()
    # The arm's recorded directory: its id names the style by the key the records name it by (the one table).
    condition_id = identity_condition_id(f"{FINAL_CAMPAIGN_JUDGE}/{style}", style)
    attempts = root / FINAL_CAMPAIGN / condition_id / "draw-000" / "attempts.jsonl"
    if not attempts.is_file():
        pytest.skip(f"missing published artifact {attempts}")
    scenarios = {compute_scenario_id(row): row for row in iter_jsonl_records(root / FROZEN_POPULATION)}
    rules = [json.loads(line) for line in RULES.read_text().splitlines() if line.strip()]
    texts = {"full_ruleset": ("\n".join(format_rule(rule) for rule in rules), None)}
    texts.update({rule_id: (text, context) for rule_id, text, context in build_rule_variants(rules, "isolated")[0]})
    checked = 0
    with attempts.open() as handle:
        for line in itertools.islice(handle, BUNDLES_CHECKED):
            record = json.loads(line)
            scenario = dict(scenarios[record["scenario_id"]])
            assert [c["component_id"] for c in record["components"]] == list(texts)
            for component in record["components"]:
                text, context = texts[component["component_id"]]
                prompt = ModelSpecTaskSpec.build_eval_prompt(
                    _Spec(style), text, scenario, "No_Interpretation", context_rules=context
                )
                assert prompt_sha256({"system_prompt": prompt, "user_prompt": ""}) == component["prompt_sha256"]
                checked += 1
    assert checked == BUNDLES_CHECKED * len(texts) == BUNDLES_CHECKED * 58
