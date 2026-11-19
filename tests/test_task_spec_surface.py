from __future__ import annotations

import json
import random

import pytest

from ruleforge.optimize import scoring as scoring_module
from ruleforge.optimize.scoring import RewardSettings, ra_reward, sg_reward
from ruleforge.tasks.spec import LegalTaskSpec
from ruleforge.tasks.spec import ModelSpecTaskSpec
from ruleforge.tasks import LegalTask, ModelSpecTask, RewardWeights

GENERATED = [{"scenario": {"prompt": f"p{i}", "response": f"r{i}", "facts_of_case": f"f{i}",
                           "legal_question": f"q{i}"}, "entropy_reward": 0.1 * i, "penalty": 0.0,
              "effective_reward": 0.1 * i} for i in range(3)]
SG_HISTORY = [{"state": "Rule 1: x", "accumulated_scenarios": GENERATED[:1], "all_generated": GENERATED,
               "reward": 0.5}]


def _seeds(tmp_path, keys):
    path = tmp_path / f"seeds_{keys[0]}.jsonl"
    path.write_text("".join(json.dumps({key: f"{key} {i}" for key in keys}) + "\n" for i in range(4)),
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def specs(task_cfg, tmp_path):
    modelspec_seeds = _seeds(tmp_path, ("prompt", "response"))
    legal_seeds = _seeds(tmp_path, ("facts_of_case", "legal_question"))

    def build(kind, seed_type="none"):
        cfg = task_cfg(task_spec=kind)
        cfg.scenario_generation.initial_scenario_seed_type = seed_type
        cfg.scenario_generation.good_seed_scenario_path = modelspec_seeds
        cfg.scenario_generation.bad_seed_scenario_path = modelspec_seeds
        cfg.scenario_generation.seed_scenario_path = legal_seeds
        return ModelSpecTaskSpec(cfg) if kind == "modelspec" else LegalTaskSpec(cfg)

    return build


WEIGHTS = RewardWeights(intent=0.75)


@pytest.mark.parametrize("kind", ["modelspec", "legal"])
def test_the_agents_descriptions_and_priors(specs, kind):
    spec = specs(kind)
    assert "scenarios" in spec.get_sg_environment_description()
    assert "revising a set of" in spec.get_ra_environment_description()
    assert "Conditions for the Optimal Set of Scenarios" in spec.get_sg_prior()
    assert "0.75*(intent_penalty)" in spec.get_ra_prior(WEIGHTS)
    task = {"modelspec": ModelSpecTask, "legal": LegalTask}[kind]()
    assert (spec.get_sg_environment_description(), spec.get_ra_environment_description(), spec.get_sg_prior(),
            spec.get_ra_prior(WEIGHTS)) == (task.sg_environment_description(), task.ra_environment_description(),
                                            task.sg_prior(), task.ra_prior(WEIGHTS))


@pytest.mark.parametrize("kind", ["modelspec", "legal"])
def test_the_history_strings_are_the_shared_formatters(specs, kind, tmp_path):
    spec = specs(kind)
    assert isinstance(spec.build_ra_history_string([], WEIGHTS), str)
    reward = {"effective_reward": 0.5, "entropy_reward": 0.5, "per_scenario_rewards": [],
              "penalty_breakdown": {"intent": {"details": {"penalty": 0.0, "per_rule_scores": []}}}}
    shown = spec.build_ra_history_string([{"action": "Rule 1: x", "reward": reward}], WEIGHTS)
    assert "<ACTION>\nRule 1: x\n</ACTION>\n" in shown
    with pytest.raises(TypeError, match="the ruleset's text"):
        spec.build_ra_history_string([{"action": tmp_path / "rules.jsonl", "reward": reward}], WEIGHTS)
    assert "p2" in spec.build_sg_history_string(SG_HISTORY) or "f2" in spec.build_sg_history_string(SG_HISTORY)


@pytest.mark.parametrize(("kind", "seed_type"), [("modelspec", "wildchat"), ("legal", "single_seed")])
def test_the_scenario_generator_s_prompt_takes_its_examples_from_the_seeds_then_the_history(specs, kind, seed_type):
    spec = specs(kind, seed_type)
    first = spec.build_sg_posterior_sampling_system_prompt([], [], random.Random(3))
    assert first == spec.build_sg_posterior_sampling_system_prompt([], [], random.Random(3))
    assert len({spec.build_sg_posterior_sampling_system_prompt([], [], random.Random(seed)) for seed in range(20)}) > 1
    later = spec.build_sg_posterior_sampling_system_prompt([GENERATED[0]["scenario"]], SG_HISTORY, random.Random(3))
    assert "You must generate 3 scenarios" in first and "Environment Description Given to the Agent" in first
    assert "# Example" in first and "Already Generated Scenarios (DO NOT DUPLICATE)" in later
    assert "# Examples of Good Scenarios" in later and "# Examples of Bad Scenarios" in later
    assert "Already Generated" not in specs(kind).build_sg_posterior_sampling_system_prompt([], [], random.Random(3))


@pytest.mark.parametrize("kind", ["modelspec", "legal"])
def test_the_rewards_are_the_metrics_reward_methods_under_the_run_s_settings(specs, kind, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(scoring_module, "compute_ra_reward", lambda **kwargs: calls.append(("ra", kwargs)) or "ra")
    monkeypatch.setattr(scoring_module, "compute_sg_reward", lambda **kwargs: calls.append(("sg", kwargs)) or "sg")
    spec = specs(kind)
    assert not hasattr(spec, "ra_reward") and not hasattr(spec, "intent_weight")
    settings = RewardSettings(weights=WEIGHTS, ra_reward_method="ra-method", sg_reward_method="sg-method",
                              intent_temperature=0.5, intent_seed=3, panel_num_tries=4,
                              panel_max_retry_seconds=60.0)
    paths = {"scenario_set_path": tmp_path / "s", "rule_set_path": tmp_path / "r", "output_dir": tmp_path}
    assert ra_reward(spec, settings, **paths, backend="backend") == "ra"
    assert sg_reward(spec, settings, **paths, backend="backend") == "sg"
    assert calls == [
        ("ra", {"ra_reward_method": "ra-method", "task_spec": spec, **paths, "backend": "backend", "weights": WEIGHTS,
                "intent_temperature": 0.5, "intent_seed": 3, "panel_num_tries": 4, "panel_max_retry_seconds": 60.0}),
        ("sg", {"sg_reward_method": "sg-method", "task_spec": spec, **paths, "backend": "backend",
                "weights": WEIGHTS, "panel_num_tries": 4, "panel_max_retry_seconds": 60.0}),
    ]


def test_the_modelspec_spec_parses_labels_and_validates_forced_answers(specs, task_cfg):
    spec = specs("modelspec")
    assert spec.parse_judgment("Unsure.") == "unsure" and spec.eval_response_validator is None
    forced = ModelSpecTaskSpec(task_cfg(reasoning_prompt_style="forced-grpo"))
    assert forced.eval_response_validator('{"full_set_judgment": "Yes"}') == "yes"
    assert specs("legal").parse_judgment("no") == "no"
