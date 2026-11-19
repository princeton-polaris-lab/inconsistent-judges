from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import types
from dataclasses import fields
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from ruleforge.judges import Capabilities, JudgeError, JudgeOptions, JudgeSpec
from ruleforge.optimize import (
    DEFAULT_SEED,
    INITIAL_RULES_FNAME,
    RUN_RECORD_DIR,
    RUN_RECORD_FNAME,
    SEED_FNAME,
    NullLogger,
    OptimizeConfig,
    RAConfig,
    RewardSettings,
    RewardWeights,
    SGConfig,
    ValidatedJudge,
    WandbLogger,
    optimize,
    seed_paths,
)
from ruleforge.optimize.api import _claim
from ruleforge.optimize.scoring import WEIGHT_KEYS
from ruleforge.rulesets import Rule, Ruleset, RulesetError
from ruleforge.tasks.spec import build_task_spec
from ruleforge.tasks import ModelSpecTask, Task, TaskError
from ruleforge.judges.backend import ModelBackend
from ruleforge.judges.batch_api import BatchSpendNotConfirmed, spend_confirmed
from tests.support.scripted import RuleForgeScript, ScriptedFailure, ScriptedJudge, scripted_backend

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "optimize"
RULES = FIXTURES / "rules.jsonl"
SEEDS = {"good": FIXTURES / "good_seeds.jsonl", "bad": FIXTURES / "bad_seeds.jsonl"}
PROMPT = {"system_prompt": "Say something.", "user_prompt": ""}


def composed_config():
    """``configs/config.yaml`` as Hydra composes it, with no override."""
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
            return compose(config_name="config")
    finally:
        GlobalHydra.instance().clear()


def answering(*answers):
    """A script answering successive requests with ``answers`` in turn; an exception is raised."""
    queue = list(answers)

    def answer(system_prompt: str, user_prompt: str) -> str:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return answer


def attempt_seeds(seed: int, attempts: int, prompt=None) -> list[int]:
    """The seeds a judge's sampler gets in a validated call's first ``attempts`` attempts at ``prompt``."""
    from ruleforge.judges.backend import prompt_attempt_seed

    return [prompt_attempt_seed(seed, PROMPT if prompt is None else prompt, attempt)
            for attempt in range(1, attempts + 1)]


def validated(answer, **options) -> ValidatedJudge:
    options = {"num_tries": 3, "max_retry_seconds": None, **options}
    return ValidatedJudge(ScriptedJudge(answer), **options)


def test_the_defaults_are_the_composed_hydra_config_key_by_key():
    composed = composed_config()
    built = OptimizeConfig().to_config(task=composed.task_spec, rules=composed.rule_amendment.initial_rule_set_path,
                                       seeds=seed_paths(composed.scenario_generation))
    for section in ("scenario_generation", "rule_amendment"):
        assert OmegaConf.to_container(built[section]) == OmegaConf.to_container(composed[section]), section
    # The checkout's config names the paper's prompt; a new run defaults to naive.
    assert OmegaConf.to_container(built.eval_cfg) == {**OmegaConf.to_container(composed.eval_cfg),
                                                      "reasoning_prompt_style": "naive"}
    assert (built.task_spec, built.max_episodes) == (composed.task_spec, composed.max_episodes)
    defaults = OptimizeConfig()
    assert defaults.num_tries == composed.judge_model.num_tries
    assert defaults.max_retry_seconds is OmegaConf.select(composed.judge_model, "max_retry_seconds") is None
    # Every field is a key there, and the dataclasses are the sections' keys less the run's inputs.
    assert {item.name for item in fields(SGConfig)} == set(composed.scenario_generation) - {
        "good_seed_scenario_path", "bad_seed_scenario_path"}
    assert {item.name for item in fields(RAConfig)} == set(composed.rule_amendment) - {"initial_rule_set_path"}


def test_the_rewards_are_read_from_the_composed_config_as_the_paper_scores_them():
    from ruleforge.optimize.metrics.intent import DEFAULT_INTENT_SEED, DEFAULT_INTENT_TEMPERATURE

    composed = RewardSettings.from_config(composed_config())
    assert composed == RewardSettings() == RewardSettings.from_config(OptimizeConfig().to_config(
        task="modelspec", rules=RULES, seeds=SEEDS))
    assert composed.weights == RewardWeights() and (composed.ra_reward_method, composed.sg_reward_method) == (
        "standard", "standard")
    assert (composed.intent_temperature, composed.intent_seed) == (DEFAULT_INTENT_TEMPERATURE, DEFAULT_INTENT_SEED)


def test_the_rewards_read_every_setting_and_default_the_methods_and_intent_sampling_a_config_lacks():
    config = OptimizeConfig(rule_amendment=RAConfig(intent_weight=0.25, entropy_method="optimistic",
                                                    intent_method="rubric_baseline"),
                            intent_temperature=0.5, intent_seed=7).to_config(task="modelspec", rules=RULES,
                                                                             seeds=SEEDS)
    config.rule_amendment.ra_reward_method = "ra-method"
    config.scenario_generation.sg_reward_method = "sg-method"
    assert RewardSettings.from_config(config) == RewardSettings(
        weights=RewardWeights(intent=0.25, entropy_method="optimistic", intent_method="rubric_baseline"),
        ra_reward_method="ra-method", sg_reward_method="sg-method", intent_temperature=0.5, intent_seed=7)
    weights_only = OmegaConf.create({"rule_amendment": {"intent_weight": 1.0, "entropy_method": "consensus",
                                                        "intent_method": "rubric_orig_standards"},
                                     "scenario_generation": {}, "eval_cfg": {}})
    assert RewardSettings.from_config(weights_only) == RewardSettings()


@pytest.mark.parametrize("task", ["modelspec", "legal"])
@pytest.mark.parametrize("missing", [("intent_weight",), ("intent_weight", "intent_method"), tuple(WEIGHT_KEYS.values())])
def test_the_rewards_refuse_a_config_lacking_a_weight_or_method(task, missing):
    config = OptimizeConfig().to_config(task=task, rules=RULES, seeds=SEEDS if task == "modelspec" else {
        "single": SEEDS["good"]})
    for key in missing:
        del config.rule_amendment[key]
    named = ", ".join(f"rule_amendment.{key}" for key in missing)
    with pytest.raises(ValueError, match=re.escape(f"the run's config sets no {named}: set each")):
        RewardSettings.from_config(config)


@pytest.mark.parametrize("key, value", [("smrc_weight", 0.5), ("smrc_method", "hierarchical")])
def test_the_rewards_refuse_a_config_asking_for_the_penalty_this_version_does_not_compute(key, value):
    config = OptimizeConfig().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    config.rule_amendment[key] = value
    with pytest.raises(ValueError, match=rf"rule_amendment\.{key}={value!r}, but this version computes no"):
        RewardSettings.from_config(config)


def test_the_rewards_read_a_config_naming_the_penalty_as_the_recorded_runs_do():
    config = OptimizeConfig().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    config.rule_amendment.smrc_weight = 0.0
    config.rule_amendment.smrc_method = "hierarchical_all_rules"
    assert RewardSettings.from_config(config) == RewardSettings()
    config.rule_amendment.smrc_weight = 0  # a recorded zero may be an integer
    assert RewardSettings.from_config(config) == RewardSettings()


def test_to_config_lays_out_the_inputs_with_the_task_s_seeding():
    built = OptimizeConfig(judge_types=("Narrow",), rule_amendment=RAConfig(entropy_method="optimistic")).to_config(
        task="legal", rules="r.jsonl", seeds={"single": "s.jsonl"})
    assert built.scenario_generation.seed_scenario_path == "s.jsonl"
    assert (built.scenario_generation.initial_scenario_seed_type, built.scenario_generation.num_seed_scenarios,
            built.scenario_generation.entropy_method) == ("single_seed", 1, "optimistic")
    assert built.rule_amendment.initial_rule_set_path == "r.jsonl"
    assert list(built.eval_cfg.judge_types) == ["Narrow"] and built.judge_model is None
    assert seed_paths(built.scenario_generation) == {"single": Path("s.jsonl")}
    unseeded = OptimizeConfig(scenario_generation=SGConfig(initial_scenario_seed_type="none", num_seed_scenarios=2))
    built = unseeded.to_config(task="modelspec", rules="r.jsonl", seeds={})
    assert seed_paths(built.scenario_generation) == {} and built.scenario_generation.num_seed_scenarios == 2
    assert seed_paths(OmegaConf.create({"good_seed_scenario_path": None, "seed_scenario_path": "s.jsonl"})) == {
        "single": Path("s.jsonl")}
    assert unseeded.seeding("legal") == ("none", ())
    assert OptimizeConfig().seeding("modelspec") == ("wildchat", ("good", "bad"))


@pytest.mark.parametrize(("config", "task", "seeds", "match"), [
    (OptimizeConfig(), "modelspec", {"ugly": "u.jsonl"}, r"reads the seed files \['good', 'bad'\]"),
    (OptimizeConfig(), "legal", {"good": "g.jsonl", "bad": "b.jsonl"}, r"reads the seed files \['single'\]"),
    (OptimizeConfig(scenario_generation=SGConfig(initial_scenario_seed_type="none")), "modelspec",
     {"good": "g.jsonl", "bad": "b.jsonl"}, r"reads the seed files \[\]"),
    (OptimizeConfig(scenario_generation=SGConfig(initial_scenario_seed_type="wildchat")), "legal",
     {"single": "s.jsonl"}, "seeded by one of"),
    (OptimizeConfig(scenario_generation=SGConfig(initial_scenario_seed_type="single_seed")), "modelspec",
     {"single": "s.jsonl"}, "seeded by one of"),
    (OptimizeConfig(), "other", {}, "not a built-in task"),
])
def test_to_config_refuses_a_seeding_the_task_does_not_handle_or_seeds_it_does_not_read(config, task, seeds, match):
    with pytest.raises(ValueError, match=match):
        config.to_config(task=task, rules="r.jsonl", seeds=seeds)


def test_the_scenario_generator_s_entropy_method_is_the_amender_s():
    assert OptimizeConfig(scenario_generation=SGConfig(entropy_method="consensus")).scenario_generation
    with pytest.raises(ValueError, match="scores the scenario generator's reward too"):
        OptimizeConfig(scenario_generation=SGConfig(entropy_method="optimistic"))


def test_another_judge_runs_the_backend_s_own_loop_with_its_seeds():
    script = ("", "first", "second")
    judge = ScriptedJudge(answering(*script))
    from_judge = ValidatedJudge(judge, num_tries=3, max_retry_seconds=None).generate([PROMPT], num_responses=2)
    backend = scripted_backend(OmegaConf.create({"model_name": "m", "num_tries": 3}), answering(*script))
    from_backend = ValidatedJudge(backend, num_tries=3, max_retry_seconds=None).generate([PROMPT], num_responses=2)
    assert from_judge == from_backend == [["first", "second"]]
    first, second = attempt_seeds(DEFAULT_SEED, 2)
    assert [request["seed"] for request in judge.requests] == [first, first + 1, second]


PROMPTS = [PROMPT, {"system_prompt": "Say something else.", "user_prompt": ""}]


@pytest.mark.parametrize(("call", "answer"), [
    (lambda judge: judge.generate(PROMPTS), "text"),
    (lambda judge: judge.generate_with_validation(PROMPTS, ("yes", "no")), "yes"),
    (lambda judge: judge.generate_with_json_validation(PROMPTS), '{"rules": []}'),
])
def test_a_raised_request_is_retried_as_an_empty_answer_is(call, answer):
    results, requests = [], []
    for failure in ("", ScriptedFailure("down")):
        seen = set()

        def script(system_prompt, user_prompt, failure=failure, seen=seen):
            if system_prompt not in seen:
                seen.add(system_prompt)
                return answering(failure)(system_prompt, user_prompt)
            return answer

        judge = ScriptedJudge(script)
        results.append(call(ValidatedJudge(judge, num_tries=3, max_retry_seconds=None)))
        requests.append([(request["system_prompt"], request["seed"]) for request in judge.requests])
    assert results[0] == results[1] and requests[0] == requests[1] and len(requests[0]) == 4
    assert None not in results[0]


class Unreliable(ScriptedJudge):
    """A judge whose calls themselves fail: ``failures`` raised in turn (None answers normally)."""

    def __init__(self, failures, answer, **options):
        super().__init__(answer, **options)
        self.failures = list(failures)

    def generate_once(self, prompts, **kwargs):
        failure = self.failures.pop(0) if self.failures else None
        if failure is not None:
            raise failure
        return super().generate_once(prompts, **kwargs)


def test_a_failed_call_is_retried_and_a_fatal_one_propagates():
    retried = ValidatedJudge(Unreliable([RuntimeError("blip")], answering("yes")), num_tries=2,
                             max_retry_seconds=None)
    assert retried.generate_with_validation([PROMPT], ("yes",)) == ["yes"]
    fatal = ValidatedJudge(Unreliable([BatchSpendNotConfirmed("confirm")], answering("yes")), num_tries=2,
                           max_retry_seconds=None)
    with pytest.raises(BatchSpendNotConfirmed):
        fatal.generate([PROMPT])


class Short(ScriptedJudge):
    """A judge returning one outcome too few."""

    def generate_once(self, prompts, **kwargs):
        return super().generate_once(prompts, **kwargs)[:-1]


def test_a_judge_that_answers_too_few_prompts_fails_the_attempt():
    judge = Short(answering("yes", "yes", "yes", "yes"))
    assert ValidatedJudge(judge, num_tries=2, max_retry_seconds=None).generate([PROMPT, PROMPT]) == [None, None]
    assert len(judge.requests) == 4


def test_the_retry_budget_and_the_backoff_are_the_backend_s(monkeypatch):
    clock = {"now": 0.0}
    slept = []

    def monotonic():
        clock["now"] += 10.0
        return clock["now"]

    monkeypatch.setattr("ruleforge.judges.backend.time", types.SimpleNamespace(monotonic=monotonic,
                                                                              sleep=slept.append))
    budgeted = ScriptedJudge(answering("", "", ""))
    ValidatedJudge(budgeted, num_tries=3, max_retry_seconds=5).generate([PROMPT])
    assert len(budgeted.requests) == 1 and slept == []
    billed = ScriptedJudge(answering("", "late"), capabilities=Capabilities(seed="unsupported", spends_money=True))
    with spend_confirmed():
        assert ValidatedJudge(billed, num_tries=2, max_retry_seconds=None).generate([PROMPT]) == [["late"]]
    assert slept == [1.0]


@pytest.mark.parametrize("call", ["generate", "generate_with_validation", "generate_with_json_validation",
                                  "generate_with_traces", "generate_with_validation_and_traces",
                                  "generate_with_json_validation_and_traces"])
def test_a_judge_that_bills_each_call_makes_none_unconfirmed(call):
    arguments = {"generate_with_validation": (("yes", "no"),),
                 "generate_with_validation_and_traces": (("yes", "no"),)}.get(call, ())
    billed = ScriptedJudge(answering("yes"), capabilities=Capabilities(seed="unsupported", spends_money=True))
    with pytest.raises(BatchSpendNotConfirmed, match="pass --yes-spend"):
        getattr(ValidatedJudge(billed, num_tries=1, max_retry_seconds=None), call)([PROMPT], *arguments)
    assert billed.requests == []
    with spend_confirmed():
        getattr(ValidatedJudge(billed, num_tries=1, max_retry_seconds=None), call)([PROMPT], *arguments)
    batch = ScriptedJudge(answering("yes"), capabilities=Capabilities(seed="unsupported", spends_money=True,
                                                                      batch=True))
    getattr(ValidatedJudge(batch, num_tries=1, max_retry_seconds=None), call)([PROMPT], *arguments)
    assert len(billed.requests) == len(batch.requests) == 1


def test_temperatures_and_seeds_reach_the_judge_as_the_backend_resolves_them(caplog):
    free = ScriptedJudge(answering("a", "b"))
    ValidatedJudge(free, num_tries=1, max_retry_seconds=None).generate([PROMPT])
    pinned = ScriptedJudge(answering("a"), capabilities=Capabilities(seed="unsupported", fixed_temperature=1.0))
    ValidatedJudge(pinned, num_tries=1, max_retry_seconds=None).generate([PROMPT], temperature=0.2)
    assert free.requests[0]["temperature"] is None and pinned.requests[0]["temperature"] == 1.0
    assert "forces temperature=1.0" in caplog.text
    seeded = ScriptedJudge(answering("a"))
    seeded.spec = JudgeSpec(model_type="scripted", model_name="s",
                            options=JudgeOptions({"model_type": "scripted", "model_name": "s", "seed": 7}))
    ValidatedJudge(seeded, num_tries=1, max_retry_seconds=None).generate([PROMPT])
    assert seeded.requests[0]["seed"] == attempt_seeds(7, 1)[0]


def test_a_model_backend_keeps_its_budget_and_takes_the_wrapper_s_attempts():
    config = OmegaConf.create({"model_name": "m", "max_retry_seconds": 300})
    with pytest.raises(JudgeError, match="reads its retry budget from its model config"):
        ValidatedJudge(scripted_backend(config, answering()), num_tries=2, max_retry_seconds=None)
    calls = []
    backend = scripted_backend(config, lambda system_prompt, user_prompt: calls.append(1) or "")
    assert backend.num_tries == 5
    judge = ValidatedJudge(backend, num_tries=2, max_retry_seconds=300)
    assert judge.generate([PROMPT]) == [None] and len(calls) == 2
    assert (judge.model_name, judge.model_type) == ("m", "fake")


def test_the_wrapper_names_its_judge_and_refuses_what_it_cannot_retry():
    panel = ScriptedJudge(answering(), model_name="panel")
    judge = ValidatedJudge(panel, num_tries=1, max_retry_seconds=None)
    assert (judge.model_name, judge.model_type, judge.judge) == ("panel", "scripted", panel)
    with pytest.raises(TypeError, match="started Judge"):
        ValidatedJudge(JudgeSpec.openai("gpt-x"), num_tries=1, max_retry_seconds=None)
    with pytest.raises(ValueError, match="num_tries"):
        validated(answering(), num_tries=0)
    with pytest.raises(ValueError, match="positive"):
        validated(answering(), max_retry_seconds=0)


def test_the_loggers(monkeypatch):
    assert NullLogger().log({"reward": 1.0}, step=3) is None
    logged = []
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(log=lambda metrics, step: logged.append(
        (metrics, step))))
    metrics = {"rule_set_idx": 0}
    WandbLogger().log(metrics, step=2)
    assert logged == [(metrics, 2)] and logged[0][0] is metrics


def test_importing_the_optimizer_imports_no_w_and_b():
    probe = "import sys, ruleforge.optimize.trainer, ruleforge.optimize; print('wandb' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=REPO)
    assert result.stdout.strip() == "False"


def small(**changes) -> OptimizeConfig:
    """A one-episode run over two judge types, small batches and two candidates."""
    return OptimizeConfig(**{
        "max_episodes": 1, "judge_types": ("No_Interpretation", "Narrow"),
        "scenario_generation": SGConfig(num_seed_scenarios=1, target_num_scenarios=2, num_scenarios_per_batch=2,
                                        max_accumulation_iterations=1),
        "rule_amendment": RAConfig(best_of_n=2), **changes})


def test_optimize_runs_the_paper_s_settings_end_to_end(tmp_path):
    """The paper's settings for two episodes, every judge type of its panel, a scripted judge and agent."""
    run = optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()),
                   agent=ScriptedJudge(RuleForgeScript()), seeds=SEEDS, out=tmp_path / "run",
                   config=OptimizeConfig(max_episodes=2))
    assert (run.last_episode, run.stopped_by) == (2, ("max_episodes",))
    assert run.ruleset_path == tmp_path / "run" / "episode_2" / "rule_amendment" / "optimal_rule_set.jsonl"
    written = json.loads((run.ruleset_path.parent / "reward_optimal.jsonl").read_text(encoding="utf-8"))
    assert run.reward == written["reward"] and run.reward["effective_reward"] == 0.75
    assert [rule.id for rule in run.ruleset.rules] == ["Meta-Rule 1", "Rule 1", "Rule 2", "Rule 3"]
    accumulated = (tmp_path / "run" / "episode_2" / "accumulated_scenario_set.jsonl").read_text().splitlines()
    # each scenario-generation step's scenarios whose panel is complete; the script drops some judgments
    assert len(accumulated) == 36


@pytest.mark.parametrize("given", ["spec", "started"])
def test_an_unconfirmed_judge_that_bills_each_call_is_refused_before_the_run_claims_its_output(tmp_path, monkeypatch,
                                                                                              given):
    requests: list = []
    billed = Capabilities(seed="local_deterministic", per_request_errors=True, spends_money=True)

    def start(self):
        backend = scripted_backend(self.options.config(), RuleForgeScript(), requests=requests)
        backend.capabilities = billed
        return backend

    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: billed)
    monkeypatch.setattr(JudgeSpec, "start", start)
    # a billed judge backs off between attempts; the clock stands still here
    monkeypatch.setattr("ruleforge.judges.backend.time", types.SimpleNamespace(monotonic=lambda: 0.0,
                                                                              sleep=lambda seconds: None))
    judge = JudgeSpec.fake(1) if given == "spec" else ScriptedJudge(RuleForgeScript(), capabilities=billed)
    call = dict(task=ModelSpecTask(), judge=judge, seeds=SEEDS, out=tmp_path / "run",
                config=OptimizeConfig(max_episodes=0))
    with pytest.raises(BatchSpendNotConfirmed, match="pass --yes-spend"):
        optimize(RULES, **call)
    assert not (tmp_path / "run").exists() and requests == [] and getattr(judge, "requests", []) == []
    with spend_confirmed():
        assert optimize(RULES, **call).last_episode == 0
    assert requests or judge.requests


def test_optimize_takes_a_ruleset_or_its_file_and_a_task_by_name(tmp_path):
    for index, rules in enumerate((Ruleset.load(RULES), str(RULES))):
        run = optimize(rules, task="modelspec", judge=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                       out=tmp_path / str(index), config=small())
        assert run.last_episode == 1


class OtherTask(Task):
    name = "other"
    scenario_fields = ("prompt",)
    styles = {"plain": "label"}
    default_style = "plain"

    def render(self, rule_text, scenario, settings, *, context_rules=None):
        return rule_text


@pytest.mark.parametrize(("arguments", "error", "match"), [
    ({"task": OtherTask()}, TaskError, "built-in tasks"),
    ({"seeds": {"good": SEEDS["good"]}}, ValueError, "seed files"),
    ({"seeds": {**SEEDS, "single": SEEDS["good"]}}, ValueError, "seed files"),
    ({"rules": Ruleset(rules=(Rule("Rule 1", "Be kind"),), name="kind", sha256="0" * 64)}, RulesetError, "no file"),
    ({"judge": object()}, TypeError, "Judge or a JudgeSpec"),
])
def test_optimize_refuses_inputs_it_cannot_run(tmp_path, arguments, error, match):
    call = {"rules": RULES, "task": ModelSpecTask(), "judge": ScriptedJudge(RuleForgeScript()), "seeds": SEEDS,
            "out": tmp_path / "run", "config": small(), **arguments}
    with pytest.raises(error, match=match):
        optimize(call.pop("rules"), **call)
    assert not (tmp_path / "run" / "episode_0").exists()


def test_optimize_refuses_a_style_the_task_keeps_only_to_read_its_records(tmp_path, monkeypatch):
    monkeypatch.setattr(ModelSpecTask, "record_only_styles", frozenset({"forced-grpo"}))
    judge = ScriptedJudge(RuleForgeScript())
    with pytest.raises(TaskError, match="prompt style 'forced-grpo' of task 'modelspec' is recognised only to read the "
                                        "records that ran it") as refused:
        optimize(RULES, task=ModelSpecTask(), judge=judge, seeds=SEEDS, out=tmp_path / "run",
                 config=small(reasoning_prompt_style="forced-grpo"))
    assert refused.value.setting == "prompt_style" and not (tmp_path / "run").exists()


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(("rows", "comes_back"), [
    # Not rule headers: the text of every rule is one rule, line_1 (the ids lost, the run optimizing one "rule").
    ([{"id": "Guideline A", "rule": "Be honest."}, {"id": "Guideline B", "rule": "Be kind."}],
     "('line_1', 'Guideline A: Be honest.\\nGuideline B: Be kind.')"),
    ([{"id": "Rule 1.1", "rule": "Be honest."}], "('Rule 11', 'Be honest.')"),  # the id's digits
    ([{"id": "openai_rules7", "rule": "Be kind."}], "('Rule 7', 'Be kind.')"),
    ([{"id": "Rule 1", "rule": "Be kind. Rule 2: be brief."}], "('Rule 1', 'Be kind.')"),
    ([{"id": "Rule 1", "rule": "Be kind.\n\n  Always."}], "('Rule 1', 'Be kind.\\nAlways.')"),
    ([{"id": "Rule 1", "rule": "Be kind. "}], "('Rule 1', 'Be kind.')"),
])
def test_optimize_refuses_a_ruleset_its_own_text_would_change(tmp_path, rows, comes_back):
    rules = write_jsonl(tmp_path / "rules.jsonl", rows)
    with pytest.raises(RulesetError, match=re.escape(f"would come back as {comes_back}")):
        optimize(rules, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                 out=tmp_path / "run", config=small())
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("ids", [("Rule 1", "Rule 2", "Rule 1"), ("Meta-Rule 1", "Meta-Rule 1", "Rule 1")])
def test_optimize_refuses_two_rules_with_one_id(tmp_path, ids):
    rules = write_jsonl(tmp_path / "rules.jsonl", [{"id": rule_id, "rule": f"Be kind {index}."}
                                                   for index, rule_id in enumerate(ids)])
    with pytest.raises(RulesetError, match=f"ruleset rules.jsonl: '{ids[0]}' names the rule '{ids[0]}' names"):
        optimize(rules, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                 out=tmp_path / "run", config=small())
    assert not (tmp_path / "run").exists()


def test_optimize_runs_a_loaded_ruleset_only_while_it_is_its_file_s(tmp_path):
    rules = tmp_path / "rules.jsonl"
    rules.write_bytes(RULES.read_bytes())
    loaded = Ruleset.load(rules)
    call = {"task": ModelSpecTask(), "judge": ScriptedJudge(RuleForgeScript()), "seeds": SEEDS,
            "out": tmp_path / "run", "config": small()}
    with pytest.raises(RulesetError, match="not its file's"):
        optimize(dataclasses.replace(loaded, rules=loaded.rules[:-1]), **call)
    with rules.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "Rule 4", "rule": "Be brief"}\n')
    with pytest.raises(RulesetError, match="not the expected"):
        optimize(loaded, **call)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(("content", "match"), [
    ('{"facts_of_case": "f", "legal_question": "q"}\n', r"lacks \['prompt', 'response'\]"),
    ("", "holds no scenarios"),
    ("not json\n", "cannot read"),
    (None, "cannot read"),
])
def test_optimize_refuses_seed_files_without_the_task_s_scenarios(tmp_path, content, match):
    bad = tmp_path / "seeds.jsonl"
    if content is not None:
        bad.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()),
                 seeds={"good": SEEDS["good"], "bad": bad}, out=tmp_path / "run", config=small())
    assert not (tmp_path / "run").exists()


class FirstScenarioPrompt(BaseException):
    """Raised by a script at the scenario generator's first prompt, carrying it (past every retry loop)."""


def stop_at_scenario_generation(system_prompt: str, user_prompt: str) -> str:
    if re.search(r"You must generate \d+ scenarios\.", system_prompt):  # either task's scenario-generation prompt
        raise FirstScenarioPrompt(system_prompt)
    return ""


LEGAL = REPO / "tests" / "fixtures" / "legal"
LEGAL_SEED = {"facts_of_case": "A mine operator was cited twice for one violation.",
              "legal_question": "Do both citations count toward its history?"}


@pytest.mark.parametrize(("task", "rules", "seeding", "shown", "hidden"), [
    # The legal task's own seeding by default (its base config's single_seed), not the Model Spec task's.
    ("legal", LEGAL / "legal_gms_rules.jsonl", None, ["# Example Scenarios", LEGAL_SEED["facts_of_case"]], []),
    ("legal", LEGAL / "legal_gms_rules.jsonl", "none", [], ["# Example Scenarios", LEGAL_SEED["facts_of_case"]]),
    ("modelspec", RULES, None, ["# Examples of Good Scenarios", "Is this mole cancerous?"], []),
    ("modelspec", RULES, "none", [], ["# Examples of Good Scenarios"]),
])
def test_the_seed_files_reach_the_scenario_generator_s_first_prompt(tmp_path, task, rules, seeding, shown, hidden):
    seeds = ({} if seeding == "none" else
             {"single": write_jsonl(tmp_path / "seeds.jsonl", [LEGAL_SEED])} if task == "legal" else SEEDS)
    config = small(scenario_generation=SGConfig(initial_scenario_seed_type=seeding))
    with pytest.raises(FirstScenarioPrompt) as stopped:
        optimize(rules, task=task, judge=ScriptedJudge(stop_at_scenario_generation), seeds=seeds,
                 out=tmp_path / "run", config=config)
    prompt = stopped.value.args[0]
    assert all(text in prompt for text in shown) and not any(text in prompt for text in hidden)


def test_an_unseeded_run_needs_no_seed_files(tmp_path):
    config = small(scenario_generation=SGConfig(initial_scenario_seed_type="none", num_seed_scenarios=1,
                                                target_num_scenarios=2, num_scenarios_per_batch=2,
                                                max_accumulation_iterations=1))
    run = optimize(RULES, task="modelspec", judge=ScriptedJudge(RuleForgeScript()), seeds={}, out=tmp_path / "run",
                   config=config)
    record = json.loads((run.out / RUN_RECORD_DIR / RUN_RECORD_FNAME).read_text(encoding="utf-8"))
    assert run.last_episode == 1 and record["seeds"] == {}
    assert record["config"]["scenario_generation"]["initial_scenario_seed_type"] == "none"


def test_a_task_spec_needs_the_seed_files_of_its_seeding_alone():
    unseeded = OptimizeConfig(scenario_generation=SGConfig(initial_scenario_seed_type="none")).to_config(
        task="modelspec", rules="r.jsonl", seeds={})
    assert build_task_spec(unseeded).good_seed_scenario_path is None
    unseeded.scenario_generation.initial_scenario_seed_type = "wildchat"
    with pytest.raises(ValueError, match="reads scenario_generation.good_seed_scenario_path"):
        build_task_spec(unseeded)


def test_optimize_writes_into_a_directory_of_its_own(tmp_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "notes.txt").write_text("mine")
    (tmp_path / "file").write_text("a file")
    for out in (tmp_path / "taken", tmp_path / "file"):
        with pytest.raises(FileExistsError, match="not an empty directory"):
            optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()), seeds=SEEDS, out=out,
                     config=small())
    (tmp_path / "empty").mkdir()
    assert optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                    out=tmp_path / "empty", config=small()).last_episode == 1


@pytest.fixture
def started_specs(monkeypatch):
    """Every ``JudgeSpec.start`` answered by a scripted ModelBackend, its start and ``close`` recorded."""
    record = {"started": [], "closed": []}

    def start(spec):
        backend = scripted_backend(spec.options.config(), RuleForgeScript())
        record["started"].append(backend)
        return backend

    monkeypatch.setattr(JudgeSpec, "start", start)
    monkeypatch.setattr(ModelBackend, "close", lambda backend: record["closed"].append(backend))
    with spend_confirmed():
        yield record


def test_a_judge_given_as_a_spec_is_started_with_the_run_s_budget_and_closed(tmp_path, started_specs):
    optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x", max_retry_seconds=300), seeds=SEEDS,
             out=tmp_path / "a", config=small())
    assert [judge.max_retry_seconds for judge in started_specs["started"]] == [None]
    assert started_specs["closed"] == started_specs["started"]
    optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x", max_retry_seconds=60), seeds=SEEDS,
             out=tmp_path / "b", config=small(max_retry_seconds=60))
    assert started_specs["started"][1].max_retry_seconds == 60.0
    caller_s = scripted_backend(OmegaConf.create({"model_name": "gpt-x", "max_retry_seconds": 300}),
                                RuleForgeScript())
    with pytest.raises(JudgeError, match="retry budget"):
        optimize(RULES, task=ModelSpecTask(), judge=caller_s, seeds=SEEDS, out=tmp_path / "c", config=small())
    assert len(started_specs["closed"]) == 2 and not (tmp_path / "c").exists()


def test_the_agent_shares_the_judge_s_instance_as_the_program_shares_its_backend(tmp_path, monkeypatch,
                                                                              started_specs):
    seen = []
    real_validated = ValidatedJudge.__init__

    def record(self, judge, **options):
        seen.append(judge)
        real_validated(self, judge, **options)

    monkeypatch.setattr(ValidatedJudge, "__init__", record)
    script = ScriptedJudge(RuleForgeScript())
    optimize(RULES, task=ModelSpecTask(), judge=script, agent=script, seeds=SEEDS, out=tmp_path / "same",
             config=small())
    assert seen == [script]
    optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x", seed=9),
             agent=JudgeSpec.openai("gpt-x", seed=9), seeds=SEEDS, out=tmp_path / "specs", config=small())
    assert len(started_specs["started"]) == 1 and seen[1:] == started_specs["started"]
    optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x"), agent=JudgeSpec.openai("gpt-y"),
             seeds=SEEDS, out=tmp_path / "two", config=small())
    assert [backend.model_name for backend in started_specs["started"][1:]] == ["gpt-x", "gpt-y"]
    assert seen[2:] == started_specs["started"][1:]


def test_each_started_judge_is_closed_though_a_close_before_it_raises(tmp_path, monkeypatch, started_specs):
    def close(backend):
        started_specs["closed"].append(backend)
        if backend.model_name == "gpt-y":
            raise RuntimeError("the agent's engine did not stop")

    monkeypatch.setattr(ModelBackend, "close", close)
    with pytest.raises(RuntimeError, match="the agent's engine did not stop"):
        optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x"), agent=JudgeSpec.openai("gpt-y"),
                 seeds=SEEDS, out=tmp_path / "run", config=small())
    assert [backend.model_name for backend in started_specs["closed"]] == ["gpt-y", "gpt-x"]


@pytest.mark.parametrize(("agent", "differing"), [
    (JudgeSpec.openai("gpt-x", seed=9), "['seed']"),
    (JudgeSpec.openai("gpt-x", temperature=0.7, reasoning_level="high"), "['reasoning_level', 'temperature']"),
])
def test_an_agent_spec_of_the_judge_s_model_with_other_options_is_refused(tmp_path, started_specs, agent,
                                                                        differing):
    with pytest.raises(JudgeError, match=re.escape(differing)):
        optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x"), agent=agent, seeds=SEEDS,
                 out=tmp_path / "refused", config=small())
    assert started_specs["started"] == [] and not (tmp_path / "refused").exists()
    own = scripted_backend(OmegaConf.create({"model_name": "gpt-x", "seed": 9}), RuleForgeScript())
    optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x"), agent=own, seeds=SEEDS,
             out=tmp_path / "own", config=small())
    assert len(started_specs["started"]) == 1 and started_specs["closed"] == started_specs["started"]


def test_optimize_claims_out_before_a_judge_starts_and_records_the_run(tmp_path, monkeypatch, started_specs):
    out = tmp_path / "run"
    start = JudgeSpec.start
    refused = []

    def start_after_a_second_run_tries(spec):
        with pytest.raises(FileExistsError, match="not an empty directory"):
            optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()), seeds=SEEDS, out=out,
                     config=small())
        refused.append(spec.model_name)
        return start(spec)

    monkeypatch.setattr(JudgeSpec, "start", start_after_a_second_run_tries)
    optimize(RULES, task=ModelSpecTask(), judge=JudgeSpec.openai("gpt-x", seed=3), seeds=SEEDS, out=out,
             config=small())
    assert refused == ["gpt-x"]
    record = json.loads((out / RUN_RECORD_DIR / RUN_RECORD_FNAME).read_text(encoding="utf-8"))
    snapshot = out / RUN_RECORD_DIR / INITIAL_RULES_FNAME
    assert snapshot.read_bytes() == RULES.read_bytes()
    assert record["ruleset"] == {"name": "rules.jsonl", "sha256": Ruleset.load(RULES).sha256, "source": str(RULES)}
    assert record["seeds"] == {role: {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                               for role, path in SEEDS.items()}
    for role, path in SEEDS.items():
        copy = out / RUN_RECORD_DIR / SEED_FNAME.format(role=role)
        assert copy.read_bytes() == path.read_bytes()
        assert record["config"]["scenario_generation"][f"{role}_seed_scenario_path"] == str(copy)
    assert (record["task"], record["agent"], record["num_tries"], record["max_retry_seconds"]) == (
        "modelspec", "judge", 5, None)
    assert record["judge"] == {"model_type": "openai", "model_name": "gpt-x", "started_by": "optimize",
                               "options": JudgeSpec.openai("gpt-x", seed=3).options.written()}
    assert record["config"]["rule_amendment"]["initial_rule_set_path"] == str(snapshot)
    assert record["config"]["scenario_generation"]["initial_scenario_seed_type"] == "wildchat"
    two = optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript(), model_name="panel"),
                   agent=ScriptedJudge(RuleForgeScript(), model_name="agent"), seeds=SEEDS, out=tmp_path / "two",
                   config=small())
    record = json.loads((two.out / RUN_RECORD_DIR / RUN_RECORD_FNAME).read_text(encoding="utf-8"))
    assert [(record[role]["model_name"], record[role]["started_by"]) for role in ("judge", "agent")] == [
        ("panel", "caller"), ("agent", "caller")]


def test_of_two_runs_given_one_out_only_one_claims_it(tmp_path, monkeypatch):
    out = tmp_path / "run"
    assert _claim(out) == out / RUN_RECORD_DIR
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(()))  # the other run claimed it after this one looked
    with pytest.raises(FileExistsError, match="claimed by another run"):
        _claim(out)


class Interrupted(ScriptedJudge):
    """A judge whose first call's outcomes carry text and also an error, or a failed block."""

    def __init__(self, answer, failure, **options):
        super().__init__(answer, **options)
        self.failure = failure

    def generate_once(self, prompts, **kwargs):
        outcomes = super().generate_once(prompts, **kwargs)
        if len(self.requests) == len(prompts):
            return [dataclasses.replace(outcome, **self.failure) for outcome in outcomes]
        return outcomes


@pytest.mark.parametrize("failure", [{"error": "connection interrupted"}, {"block_failed": True}])
def test_an_answer_that_failed_in_transport_is_retried_whatever_text_it_carries(failure):
    judge = Interrupted(answering("yes", "no"), failure)
    assert ValidatedJudge(judge, num_tries=2, max_retry_seconds=None).generate_with_validation(
        [PROMPT], ("yes", "no")) == ["no"]
    assert len(judge.requests) == 2
