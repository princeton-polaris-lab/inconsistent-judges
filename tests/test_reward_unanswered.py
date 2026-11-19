from __future__ import annotations

import json

import pytest

from ruleforge.optimize.metrics.reward import compute_ra_reward_standard, compute_sg_reward_standard
from ruleforge.tasks import RewardWeights
from tests.support.traces import traced

#: Neither penalty is scored: the entropy reward alone.
NO_PENALTIES = RewardWeights(intent=0.0)


class _Judge:
    """Answers `answers` in order, cycling; None means validation failed."""

    def __init__(self, answers):
        self.model_name = "stub-judge"
        self.seed = 16
        self.answers = list(answers)
        self.index = 0

    def generate_with_validation(self, prompts, **_kwargs):
        out = []
        for _ in prompts:
            out.append(self.answers[self.index % len(self.answers)])
            self.index += 1
        return out

    def generate_with_json_validation(self, prompts, **_kwargs):
        return [{"preempting_rules": []} for _ in prompts]

    # The reward keeps every judge call's trace beside its evaluations.
    def generate_with_validation_and_traces(self, prompts, **kwargs):
        return traced(self.generate_with_validation(prompts, **kwargs), "judgment")


@pytest.fixture
def reward_inputs(tmp_path):
    rules = tmp_path / "rules.jsonl"
    rules.write_text(
        json.dumps({"id": "Rule 1", "rule": "be helpful"}) + "\n", encoding="utf-8"
    )
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        "".join(
            json.dumps({"prompt": f"p{n}", "response": f"r{n}"}) + "\n"
            for n in range(2)
        ),
        encoding="utf-8",
    )
    return rules, scenarios


def _spec(modelspec, rules):
    spec = modelspec()
    spec.initial_rule_set_path = rules
    return spec


def test_a_fully_answered_run_scores_from_every_scenario(
    tmp_path, reward_inputs, modelspec
):
    rules, scenarios = reward_inputs
    reward = compute_ra_reward_standard(
        _spec(modelspec, rules), scenarios, rules, tmp_path / "a", _Judge(["yes"]), weights=NO_PENALTIES
    )
    assert reward["entropy_reward"] == pytest.approx(1.0)
    assert len(reward["per_scenario_rewards"]) == 2


def test_scenarios_nobody_answered_are_dropped_not_scored_as_consensus(
    tmp_path, reward_inputs, modelspec
):
    rules, scenarios = reward_inputs
    reward = compute_ra_reward_standard(
        _spec(modelspec, rules), scenarios, rules, tmp_path / "b", _Judge([None]), weights=NO_PENALTIES
    )
    assert reward["entropy_reward"] == 0.0
    assert reward["per_scenario_rewards"] == []


def test_a_partially_answered_run_averages_only_what_was_answered(
    tmp_path, reward_inputs, modelspec
):
    rules, scenarios = reward_inputs
    reward = compute_ra_reward_standard(
        _spec(modelspec, rules), scenarios, rules, tmp_path / "c", _Judge(["yes", None]), weights=NO_PENALTIES
    )
    assert len(reward["per_scenario_rewards"]) == 1
    assert reward["entropy_reward"] == pytest.approx(1.0)


@pytest.mark.parametrize("which", ["ra", "sg"])
def test_a_scenario_whose_panel_lost_a_judgment_is_left_out_of_the_reward_and_the_count_is_kept(
    tmp_path, reward_inputs, modelspec, caplog, which
):
    """The second scenario's panel lost one of two judgments; read as agreement, it would raise the amender's reward."""
    rules, scenarios = reward_inputs
    spec = _spec(lambda: modelspec(judge_types=["No_Interpretation", "Narrow"]), rules)
    score = compute_ra_reward_standard if which == "ra" else compute_sg_reward_standard
    with caplog.at_level("WARNING"):
        reward = score(spec, scenarios, rules, tmp_path, _Judge(["yes", "no", "no", None]), weights=NO_PENALTIES)
    assert [entry["scenario"]["prompt"] for entry in reward["per_scenario_rewards"]] == ["p0"]
    assert reward["entropy_reward"] == pytest.approx(0.0 if which == "ra" else 1.0)
    assert reward["judgment_count"] == 2
    assert f"{which.upper()} reward: 1 of 2 scenarios had an incomplete panel" in caplog.text


def test_the_reward_layer_drops_a_record_with_no_full_ruleset_entropy():
    from ruleforge.optimize.metrics.entropy import get_scenario_full_ruleset_entropy

    answered = {"scenario": "a", "entropy": {"m": {"full_ruleset": 0.8}}}
    unanswered = {"scenario": "b", "entropy": {"m": {"Rule 1": 0.5}}}
    scored = [
        value
        for record in (answered, unanswered)
        if (value := get_scenario_full_ruleset_entropy(record)) is not None
    ]
    assert scored == [0.8]
    assert 1.0 - sum(scored) / len(scored) == pytest.approx(0.2)


def test_the_sg_reward_applies_the_same_rule(tmp_path, reward_inputs, modelspec):
    rules, scenarios = reward_inputs
    reward = compute_sg_reward_standard(
        _spec(modelspec, rules), scenarios, rules, tmp_path / "d", _Judge([None]), weights=NO_PENALTIES
    )
    assert reward["entropy_reward"] == 0.0
    assert reward["per_scenario_rewards"] == []


def test_each_reward_s_per_scenario_values_average_to_its_entropy_reward(tmp_path, reward_inputs, modelspec, monkeypatch):
    """The amender's mean is 1 - mean entropy, the generator's the mean entropy."""
    import ruleforge.optimize.metrics.reward as reward_module

    records = [{"scenario": name, "entropy": {"m": {"full_ruleset": value}}, "judgment_count": {"m": {"full_ruleset": 1}}}
               for name, value in (("split", 0.75), ("agreed", 0.25))]
    monkeypatch.setattr(reward_module, "run_judgments", lambda **kwargs: None)
    monkeypatch.setattr(reward_module, "compute_entropy_from_evals", lambda **kwargs: records)
    rules, scenarios = reward_inputs
    spec = _spec(modelspec, rules)
    amender = compute_ra_reward_standard(spec, scenarios, rules, tmp_path, _Judge(["yes"]), weights=NO_PENALTIES)
    assert amender["per_scenario_rewards"] == [{"scenario": "split", "entropy_reward": 0.25},
                                               {"scenario": "agreed", "entropy_reward": 0.75}]
    assert amender["entropy_reward"] == amender["effective_reward"] == 0.5
    records[0]["entropy"]["m"]["full_ruleset"] = 0.5
    amender = compute_ra_reward_standard(spec, scenarios, rules, tmp_path, _Judge(["yes"]), weights=NO_PENALTIES)
    assert amender["entropy_reward"] == pytest.approx(0.625)  # 1 - mean(0.5, 0.25), not the raw entropies' mean
    assert sum(entry["entropy_reward"] for entry in amender["per_scenario_rewards"]) / 2 == amender["entropy_reward"]
    generator = compute_sg_reward_standard(spec, scenarios, rules, tmp_path, _Judge(["yes"]), weights=NO_PENALTIES)
    assert [entry["entropy_reward"] for entry in generator["per_scenario_rewards"]] == [0.5, 0.25]
    assert generator["entropy_reward"] == pytest.approx(0.375)


def test_an_unknown_reward_method_is_refused(tmp_path):
    from ruleforge.optimize.metrics.reward import compute_ra_reward, compute_sg_reward

    with pytest.raises(ValueError, match=r"Unknown RA reward method: 'nope'\. Valid: \['standard'\]"):
        compute_ra_reward("nope", None, tmp_path, tmp_path, tmp_path, None, weights=NO_PENALTIES)
    with pytest.raises(ValueError, match=r"Unknown SG reward method: 'nope'\. Valid: \['standard'\]"):
        compute_sg_reward("nope", None, tmp_path, tmp_path, tmp_path, None, weights=NO_PENALTIES)


@pytest.mark.parametrize("which", ["ra", "sg"])
def test_both_rewards_drop_a_record_with_no_full_ruleset_entropy_and_say_so(
    tmp_path, reward_inputs, modelspec, monkeypatch, caplog, which
):
    import ruleforge.optimize.metrics.reward as reward_module

    answered = {"scenario": "a", "entropy": {"m": {"full_ruleset": 0.75}}, "judgment_count": {"m": {"full_ruleset": 1}}}
    unanswered = {"scenario": "b", "entropy": {"m": {"Rule 1": 0.5}}, "judgment_count": {"m": {"full_ruleset": 0}}}
    monkeypatch.setattr(reward_module, "run_judgments", lambda **kwargs: None)
    monkeypatch.setattr(reward_module, "compute_entropy_from_evals", lambda **kwargs: [answered, unanswered])
    rules, scenarios = reward_inputs
    score = compute_ra_reward_standard if which == "ra" else compute_sg_reward_standard
    with caplog.at_level("WARNING", logger=reward_module.__name__):
        reward = score(_spec(modelspec, rules), scenarios, rules, tmp_path, _Judge(["yes"]), weights=NO_PENALTIES)
    assert reward["entropy_reward"] == pytest.approx(0.25 if which == "ra" else 0.75)
    assert [entry["scenario"] for entry in reward["per_scenario_rewards"]] == ["a"]
    assert f"{which.upper()} reward: 1 of 2 scenarios had an incomplete panel" in caplog.text
