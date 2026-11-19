from __future__ import annotations

import json
import random
from dataclasses import fields
from pathlib import Path

import pytest

from ruleforge.optimize import RAConfig
from ruleforge.tasks import RewardWeights
from ruleforge.tasks.history import (
    _RUBRIC_ORIGINAL_STANDARDS,
    _RUBRIC_MAP,
    build_scenario_set_string,
    format_intent_feedback_rubric,
    format_ra_history_string,
    format_scenario_string,
    format_scenario_xml,
    format_sg_history_string,
    read_scenario_set_from_file,
    sample_history_scenarios,
    sample_seed_scenarios,
)

KEYS = ("prompt", "response")
SCENARIO = {"prompt": "p", "response": "r", "source": "not shown"}


def experience(index: int, reward: float) -> dict:
    return {"scenario": {"prompt": f"p{index}", "response": f"r{index}"}, "entropy_reward": reward, "penalty": 0.0,
            "effective_reward": reward}


def test_the_weights_default_to_the_paper_s_and_are_held_as_floats():
    amender = RAConfig()
    assert RewardWeights() == RewardWeights(intent=amender.intent_weight, entropy_method=amender.entropy_method,
                                            intent_method=amender.intent_method)
    assert [field.name for field in fields(RewardWeights)] == ["intent", "entropy_method", "intent_method"]
    weights = RewardWeights(intent=0)
    assert repr(weights.intent) == "0.0"
    with pytest.raises(AttributeError):
        weights.intent = 0.5


def test_a_scenario_renders_its_task_s_fields_alone():
    assert format_scenario_string(SCENARIO, KEYS) == "Prompt: p\n\nResponse: r\n"
    assert format_scenario_xml(SCENARIO, KEYS) == "<PROMPT>\np\n</PROMPT>\n\n<RESPONSE>\nr\n</RESPONSE>\n"
    assert format_scenario_string({"facts_of_case": "f"}, ("facts_of_case",)) == "Facts of case: f\n"
    assert build_scenario_set_string([SCENARIO], KEYS, num_hash_prefix=1) == "# Scenario 1\nPrompt: p\n\nResponse: r\n"


def test_the_intent_feedback_names_the_flagged_rules_by_score_with_the_rubric_s_text():
    scores = [{"id": "Rule 1", "score": 0}, {"id": "Rule 2", "score": 2}, {"id": "Rule 3", "score": 1},
              {"id": "Rule 4", "score": 2}]
    assert format_intent_feedback_rubric(0.5, scores, "rubric_orig_standards") == (
        "Intent Penalty: 0.5\n\tIntent Penalty Breakdown:\n"
        f"\t\tRules that scored 1: Rule 3\n\t\tExplanation of score 1: {_RUBRIC_ORIGINAL_STANDARDS['1']}\n"
        f"\t\tRules that scored 2: Rule 2, Rule 4\n\t\tExplanation of score 2: {_RUBRIC_ORIGINAL_STANDARDS['2']}\n")
    assert format_intent_feedback_rubric(0.0, scores[:1], "rubric_orig_standards") == "Intent Penalty: 0.0\n"
    assert format_intent_feedback_rubric(1.0, [], "rubric_orig_standards", rule_deletion_detected=True) == (
        "Intent Penalty: 1.0 (Rule deletion detected - automatic maximum penalty)\n")
    assert format_intent_feedback_rubric(0.25, scores[2:3], "nope").endswith("Explanation of score 1: \n")
    assert sorted(_RUBRIC_MAP) == ["rubric_orig_standards"]


def ra_trajectory(action: str, reward: float, *, deleted: bool = False) -> dict:
    return {"action": action, "reward": {
        "effective_reward": reward, "entropy_reward": reward + 0.5, "regularization_penalty": 0.5,
        "per_scenario_rewards": [{"scenario": experience(0, 0.25)["scenario"], "entropy_reward": 0.25}],
        "penalty_breakdown": {"intent": {"value": 0.25, "weight": 1.0, "details": {
                                  "penalty": 0.25, "per_rule_scores": [{"id": "Rule 1", "score": 1}],
                                  "rule_deletion_detected": deleted}}}}}


def test_the_amender_s_history_shows_the_penalties_its_weights_score_and_the_last_episode_s_experiences():
    history = [ra_trajectory("Rule 1: first", 0.1), ra_trajectory("Rule 1: last", 0.2, deleted=True)]
    both = format_ra_history_string(history, KEYS, intent_weight=1.0, intent_method="rubric_orig_standards")
    assert both == (
        "<TRAJECTORY_1>\n<ACTION>\nRule 1: last\n</ACTION>\n<REWARD>\nOverall Total Reward: 0.2\n"
        "Overall Entropy Reward: 0.7\n"
        "Intent Penalty: 0.25 (Rule deletion detected - automatic maximum penalty)\n</REWARD>\n"
        "<EXPERIENCES>\n<STATE>\n<PROMPT>\np0\n</PROMPT>\n\n<RESPONSE>\nr0\n</RESPONSE>\n\n</STATE>\n"
        "<REWARD>\nScenario Entropy Reward: 0.25\n</REWARD>\n</EXPERIENCES>\n</TRAJECTORY_1>\n"
        "<TRAJECTORY_0>\n<ACTION>\nRule 1: first\n</ACTION>\n<REWARD>\nOverall Total Reward: 0.1\n"
        "Overall Entropy Reward: 0.6\n"
        "Intent Penalty: 0.25\n\tIntent Penalty Breakdown:\n\t\tRules that scored 1: Rule 1\n"
        f"\t\tExplanation of score 1: {_RUBRIC_ORIGINAL_STANDARDS['1']}\n</REWARD>\n</TRAJECTORY_0>\n")
    bare = format_ra_history_string([{"action": "a", "reward": {"effective_reward": 1.0, "entropy_reward": 1.0}}],
                                    KEYS, intent_weight=0.0)
    assert bare == ("<TRAJECTORY_0>\n<ACTION>\na\n</ACTION>\n<REWARD>\nOverall Total Reward: 1.0\n"
                    "Overall Entropy Reward: 1.0\n</REWARD>\n<EXPERIENCES>\n</EXPERIENCES>\n</TRAJECTORY_0>\n")


def test_the_amender_s_history_refuses_an_action_that_is_not_a_ruleset_s_text():
    with pytest.raises(TypeError, match="a trajectory's action is the ruleset's text, got PosixPath"):
        format_ra_history_string([ra_trajectory(Path("rules.jsonl"), 0.1)], KEYS)


def test_the_generator_s_history_shows_every_scenario_of_its_last_episode_and_the_kept_ones_before():
    history = [{"state": "Rule 1: a\n", "accumulated_scenarios": [experience(1, 0.9)],
                "all_generated": [experience(1, 0.9), experience(2, 0.1)], "reward": 0.9},
               {"state": "Rule 1: b\n", "accumulated_scenarios": [], "all_generated": [experience(3, 0.5)],
                "reward": 0.5}]

    def shown(index: int, reward: float) -> str:
        return (f"<EXPERIENCE>\n<ACTION><PROMPT>\np{index}\n</PROMPT>\n\n<RESPONSE>\nr{index}\n</RESPONSE>\n</ACTION>\n"
                f"<REWARD>\n{reward}\n</REWARD>\n</EXPERIENCE>\n")

    assert format_sg_history_string(history, KEYS) == (
        "<TRAJECTORY_2>\n<STATE>\nRule 1: b\n</STATE>\n" + shown(3, 0.5) + "</TRAJECTORY_2>\n"
        "<TRAJECTORY_1>\n" + shown(1, 0.9) + "</TRAJECTORY_1>\n")


def test_the_generator_s_examples_come_from_a_seed_file_then_from_its_last_episode(tmp_path):
    path = tmp_path / "seeds.jsonl"
    path.write_text("".join(json.dumps({**SCENARIO, "prompt": f"p{i}"}) + "\n" for i in range(4)), encoding="utf-8")
    assert read_scenario_set_from_file(path, KEYS) == [{"prompt": f"p{i}", "response": "r"} for i in range(4)]
    state = random.getstate()
    drawn = sample_seed_scenarios(path, KEYS, 2, random.Random(3))
    assert drawn == sample_seed_scenarios(path, KEYS, 2, random.Random(3))
    assert drawn == random.Random(3).sample(read_scenario_set_from_file(path, KEYS), k=2)
    assert len(sample_seed_scenarios(path, KEYS, 10, random.Random(3))) == 4
    assert random.getstate() == state
    last = {"all_generated": [experience(1, 0.2), experience(2, 0.9), experience(3, 0.5)]}
    assert sample_history_scenarios([{"all_generated": []}, last], 1) == {
        "good_examples": [experience(2, 0.9)["scenario"]], "bad_examples": [experience(1, 0.2)["scenario"]]}
    with pytest.raises(ValueError, match="No history"):
        sample_history_scenarios([])
