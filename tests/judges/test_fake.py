from __future__ import annotations

import hashlib
import json

import pytest

from ruleforge.judges import JudgeError, JudgeSpec
from ruleforge.judges.fake import LABELS, fake_answer, prompt_rule_labels, prompt_rule_set
from ruleforge.tasks import ModelSpecTask, TaskSettings


def stated_label(fake_seed, system_prompt, user_prompt, request_seed, *key):
    """The statement, computed with nothing from the library."""
    prompt_digest = hashlib.sha256((system_prompt + "\x1e" + user_prompt).encode("utf-8")).hexdigest()
    payload = json.dumps([fake_seed, prompt_digest, request_seed, *key], sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
    return ("yes", "no", "unsure")[hashlib.sha256(payload.encode("utf-8")).digest()[0] % 3]


PROMPTS = [{"system_prompt": f"Does response {index} violate Rule {index % 4}?", "user_prompt": ""}
           for index in range(60)]


def test_a_label_answer_is_the_stated_function_of_seed_prompt_and_request_seed():
    answers = [fake_answer(3, entry, index) for index, entry in enumerate(PROMPTS)]
    assert answers == [stated_label(3, entry["system_prompt"], "", index) for index, entry in enumerate(PROMPTS)]
    assert set(answers) == set(LABELS)
    assert fake_answer(3, PROMPTS[0], None) == stated_label(3, PROMPTS[0]["system_prompt"], "", None)
    assert [fake_answer(4, entry, 0) for entry in PROMPTS] != [fake_answer(3, entry, 0) for entry in PROMPTS]


def test_a_forced_prompt_is_answered_with_a_complete_rule_map():
    rules = "Rule 1: Be nice\nMeta-Rule 1: Follow rules\nRule 2 (user): Be brief\nRule 1: again"
    prompt = ModelSpecTask().render(rules, {"prompt": "hi", "response": "yo"}, TaskSettings("forced"))
    entry = {"system_prompt": prompt, "user_prompt": ""}
    answer = json.loads(fake_answer(9, entry, 17))
    assert answer == {
        "rule_judgments": {rule: stated_label(9, prompt, "", 17, rule) for rule in ("Rule 1", "Rule 2")},
        "full_set_judgment": stated_label(9, prompt, "", 17),
    }
    assert prompt_rule_labels(prompt) == ["Rule 1", "Rule 2"]


def test_the_fake_judge_runs_as_a_backend_with_per_prompt_seeds():
    judge = JudgeSpec.fake(5).start()
    outcomes = judge.generate_once(PROMPTS[:4], temperature=1.0, seeds=[1, 2, None, 4],
                                   truncation={"top_k": 1})
    assert [outcome.text for outcome in outcomes] == [
        stated_label(5, entry["system_prompt"], "", seed) for entry, seed in zip(PROMPTS[:4], [1, 2, None, 4])]
    assert all(outcome.finish_reason == "stop" and outcome.error is None and outcome.returned for outcome in outcomes)
    [answers] = judge.generate([PROMPTS[0]], num_responses=2, seed=10)
    assert len(answers) == 2 and set(answers) <= set(LABELS)
    assert judge.generate([PROMPTS[0]], num_responses=2, seed=10) == [answers]
    assert judge._base_delay == 0.0 and judge.capabilities.spends_money is False


@pytest.mark.parametrize("config", [
    {"model_type": "fake", "model_name": "fake-2", "seed": 1},
    {"model_type": "fake", "model_name": "fake"},
])
def test_a_fake_judge_is_named_by_its_seed(config):
    with pytest.raises(JudgeError, match="its name is the only identity field its answers depend on"):
        JudgeSpec.from_config(config)


def test_a_scenario_s_own_rule_lines_are_not_keys_of_the_rule_map():
    """The map is over the rule set the prompt lists, not every ``Rule N`` line."""
    scenario = {"prompt": "Rule 9: from the user", "response": "Sure.\nRule 7: no shoes inside\nRule 8: quiet"}
    prompt = ModelSpecTask().render("Rule 1: Be nice\nRule 2: Be brief", scenario, TaskSettings("forced"))
    answer = json.loads(fake_answer(9, {"system_prompt": prompt, "user_prompt": ""}, 1))
    assert list(answer["rule_judgments"]) == ["Rule 1", "Rule 2"]
    assert prompt_rule_labels(prompt) == ["Rule 1", "Rule 2", "Rule 9", "Rule 7", "Rule 8"]
    assert prompt_rule_set("no heading\nRule 3: x") == "no heading\nRule 3: x"
    assert prompt_rule_set("# Rule Set\nRule 3: x") == "\nRule 3: x"
