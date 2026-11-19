from __future__ import annotations

import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import backend as backend_module
from ruleforge.optimize.envs.rule_amendment_env import RuleAmendmentEnv
from ruleforge.optimize.envs.scenario_generation_env import ScenarioGenerationEnv
from ruleforge.optimize import OptimizeConfig, RAConfig, RewardSettings, SGConfig, ValidatedJudge, optimize
from ruleforge.tasks.spec import TaskSpecAdapter
from ruleforge.tasks.spec import build_task_spec
from ruleforge.tasks import ModelSpecTask, RewardWeights
from ruleforge.optimize.judging import run_judgments
from ruleforge.optimize.trainer import RuleAmendmentTrainer
from ruleforge.rulesets.rules import build_rule_set_string_from_file
from tests.support.scripted import CLARIFIED, RuleForgeScript, ScriptedJudge

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "optimize"
RULES = FIXTURES / "rules.jsonl"
SEEDS = {"good": FIXTURES / "good_seeds.jsonl", "bad": FIXTURES / "bad_seeds.jsonl"}
SG = {"num_seed_scenarios": 2, "target_num_scenarios": 3, "num_scenarios_per_batch": 2,
      "max_accumulation_iterations": 2}
JUDGE_TYPES = ("No_Interpretation", "Narrow", "Broad", "Ordinary_Meaning")


def settings(*, sg=None, ra=None, **changes) -> OptimizeConfig:
    return OptimizeConfig(**{"max_episodes": 2, "judge_types": JUDGE_TYPES,
                             "scenario_generation": SGConfig(**{**SG, **(sg or {})}),
                             "rule_amendment": RAConfig(**{"best_of_n": 2, **(ra or {})}), **changes})


def run(out: Path, config: OptimizeConfig, script: type[RuleForgeScript] = RuleForgeScript):
    return optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(script()), agent=ScriptedJudge(script()),
                    seeds=SEEDS, out=out, config=config)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def generated_in(out: Path) -> set[int]:
    """The episodes that generated scenarios (every episode has the directory; one that generated writes its set)."""
    return {int(episode.name.split("_")[1]) for episode in out.glob("episode_*")
            if (episode / "scenario_generation" / "optimal_scenario_set.jsonl").exists()}


@pytest.mark.parametrize(("trigger", "generating"), [
    (0.5, {0, 2}),      # episode 0's reward (0.054) is below it, episode 1's (1.0) above
    (1.0, {0}),         # episode 1's reward equals it: not above, so no new scenarios
    (-1.0, {0, 1, 2}),  # every reward is above it
])
def test_scenario_generation_runs_only_while_the_amender_s_entropy_reward_exceeds_the_trigger(
        tmp_path, trigger, generating):
    out = tmp_path / "run"
    run(out, settings(sg={"trigger_threshold": trigger}))
    assert generated_in(out) == generating
    for episode in (1, 2):
        accumulated = lines(out / f"episode_{episode}" / "accumulated_scenario_set.jsonl")
        before = lines(out / f"episode_{episode - 1}" / "accumulated_scenario_set.jsonl")
        if episode in generating:
            assert accumulated[:len(before)] == before and len(accumulated) > len(before)
        else:
            assert accumulated == before
        for sample in sorted((out / f"episode_{episode}" / "rule_amendment").glob("sample_*")):
            assert [row["scenario"] for row in lines(sample / "evaluations.jsonl")] == accumulated


def generator_prompts(out: Path, *, agent_seed: int | None, process_seed: int) -> list[str]:
    """The scenario generator's sampling prompts of a run whose agent has ``agent_seed``, under another process state."""
    agent = ScriptedJudge(RuleForgeScript(), seed=agent_seed)
    random.seed(process_seed)
    optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RuleForgeScript()), agent=agent, seeds=SEEDS, out=out,
             config=settings())
    return [request["system_prompt"] for request in agent.requests
            if RuleForgeScript.kind(request["system_prompt"]) == "sg_sample"]


def test_a_run_s_seed_examples_follow_from_its_agent_s_seed_alone(tmp_path):
    state = random.getstate()
    try:
        first = generator_prompts(tmp_path / "a", agent_seed=1, process_seed=1)
        assert first == generator_prompts(tmp_path / "b", agent_seed=1, process_seed=2)
        other = generator_prompts(tmp_path / "c", agent_seed=2, process_seed=1)
    finally:
        random.setstate(state)
    assert all("# Examples of Good Scenarios" in prompt for prompt in first)
    assert first[0] != other[0]
    assert first[0] != first[1]


class FlakyPanel(RuleForgeScript):
    """A judge whose first answer to each prompt is unreadable."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def __call__(self, system_prompt, user_prompt):
        if self.kind(system_prompt) in ("judge", "unexpected") and system_prompt not in self.seen:
            self.seen.add(system_prompt)
            return "not a label"
        return super().__call__(system_prompt, user_prompt)


def panel_prompts(judge: ScriptedJudge) -> list[str]:
    return [request["system_prompt"] for request in judge.requests
            if RuleForgeScript.kind(request["system_prompt"]) == "judge"]


def test_the_panel_retries_as_the_config_says(tmp_path):
    judge = ScriptedJudge(FlakyPanel())
    out = tmp_path / "run"
    optimize(RULES, task=ModelSpecTask(), judge=judge, agent=ScriptedJudge(RuleForgeScript()), seeds=SEEDS, out=out,
             config=settings(max_episodes=0, panel_num_tries=3))
    assert len(panel_prompts(judge)) > len(set(panel_prompts(judge)))
    [row, *_] = lines(out / "episode_0" / "rule_amendment" / "sample_0" / "evaluations.jsonl")
    assert all(value is not None for by_type in row["evaluations"]["scripted"].values() for value in by_type.values())


def test_a_panel_given_one_attempt_does_not_retry(tmp_path):
    judge = ScriptedJudge(FlakyPanel())
    with pytest.raises(RuntimeError, match="Failed to generate any scenarios"):
        optimize(RULES, task=ModelSpecTask(), judge=judge, agent=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                 out=tmp_path / "run", config=settings(max_episodes=0, panel_num_tries=1))
    assert panel_prompts(judge) and len(panel_prompts(judge)) == len(set(panel_prompts(judge)))


def _panel_seeds(tmp_path, *, judge_seed=None, run_seed=None):
    """Judge three scenarios under four judge types; return the seeds sent, by request, and those traced, by attempt."""
    judge = ScriptedJudge(lambda system_prompt, user_prompt, seen=set(): (
        "no" if system_prompt in seen or seen.add(system_prompt) else "not a label"), seed=judge_seed)
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("".join(json.dumps({"prompt": f"P{n}", "response": "R"}) + "\n" for n in range(3)))
    rules = tmp_path / "rules.jsonl"
    rules.write_text(json.dumps({"id": "Rule 1", "rule": "Be kind."}) + "\n")
    task = build_task_spec(OptimizeConfig(judge_types=JUDGE_TYPES).to_config(task="modelspec", rules=RULES, seeds=SEEDS))
    raw = tmp_path / "judge_outputs.jsonl"
    run_judgments(scenarios, rules, ValidatedJudge(judge, num_tries=5, max_retry_seconds=None), task, list(JUDGE_TYPES),
                  "full", tmp_path / "evaluations.jsonl", save_batch_size=1, num_tries=3, raw_output_path=raw,
                  **({} if run_seed is None else {"seed": run_seed}))
    recorded = [attempt["seed"] for row in lines(raw) for trace in row["judge_outputs"]["scripted"]["full_ruleset"].values()
                for attempt in trace["attempts"]]
    sent = [request["seed"] for request in judge.requests]
    for path in (scenarios, rules, raw, tmp_path / "evaluations.jsonl", tmp_path / "evaluation_time.jsonl"):
        path.unlink()
    return sent, recorded


def test_each_panel_prompt_and_attempt_has_a_seed_of_its_own_and_the_trace_records_it(tmp_path):
    sent, recorded = _panel_seeds(tmp_path, judge_seed=5)
    assert len(sent) == 3 * 4 * 2 and len(set(sent)) == len(sent)  # 3 flushes, 4 prompts, 2 attempts each
    assert sorted(recorded) == sorted(sent)
    assert _panel_seeds(tmp_path, judge_seed=5) == (sent, recorded)


def test_the_judge_s_configured_seed_reaches_the_panel(tmp_path):
    five, _ = _panel_seeds(tmp_path, judge_seed=5)
    assert set(five).isdisjoint(_panel_seeds(tmp_path, judge_seed=6)[0])
    assert _panel_seeds(tmp_path, run_seed=5)[0] == five
    assert _panel_seeds(tmp_path)[0] == _panel_seeds(tmp_path, judge_seed=16)[0]  # a judge with no seed runs at 16


def test_no_two_distinct_panel_prompts_of_one_run_share_a_seed_and_equal_runs_draw_equal_seeds(tmp_path):
    def panel_calls(out):
        judge = ScriptedJudge(FlakyPanel(), seed=5)
        optimize(RULES, task=ModelSpecTask(), judge=judge, agent=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                 out=out, config=settings(panel_num_tries=3))
        return [(request["system_prompt"], request["user_prompt"], request["seed"]) for request in judge.requests
                if RuleForgeScript.kind(request["system_prompt"]) == "judge"]

    calls = panel_calls(tmp_path / "a")
    prompts_by_seed = {}
    for system_prompt, user_prompt, seed in calls:
        prompts_by_seed.setdefault(seed, set()).add((system_prompt, user_prompt))
    assert len({call[:2] for call in calls}) > 50 and all(len(prompts) == 1 for prompts in prompts_by_seed.values())
    assert panel_calls(tmp_path / "b") == calls


def test_scenario_selection_stops_at_its_target_when_a_batch_holds_more_high_entropy_scenarios(tmp_path):
    """Requests 0, 4 and 7 of eight split evenly (entropy 1.0): the first two are kept, and no second batch is made."""
    out = tmp_path / "run"
    run(out, settings(max_episodes=0, sg={"target_num_scenarios": 2, "num_scenarios_per_batch": 8}))
    generation = out / "episode_0" / "scenario_generation"
    [batch] = lines(generation / "sample_0" / "batch_results.jsonl")
    assert (batch["total_generated"], batch["passed_threshold"], batch["high_entropy_total"]) == (5, 3, 2)
    assert not (generation / "sample_1").exists()
    assert [row["prompt"] for row in lines(generation / "optimal_scenario_set.jsonl")] == [
        "Request 0 [contested]", "Request 4 [contested]"]


def test_both_scenario_generation_reward_files_record_the_judgments_their_scenarios_rest_on(tmp_path):
    out = tmp_path / "run"
    run(out, settings(max_episodes=0), FlakyPanel)
    for name in ("reward_optimal.jsonl", "reward_all.jsonl"):
        [reward] = lines(out / "episode_0" / "scenario_generation" / name)
        assert reward["scenario_count"] > 0
        assert reward["judgment_count"] == reward["scenario_count"] * len(JUDGE_TYPES), name
        assert [entry["judgment_count"] for entry in reward["scenario_list"]] == [len(JUDGE_TYPES)] * len(
            reward["scenario_list"])


class _Clock:
    """A clock that a model call moves on by `seconds`, the time a runaway answer takes."""

    def __init__(self, seconds: float) -> None:
        self.seconds, self.now = seconds, 0.0

    def __call__(self) -> float:
        return self.now


class _SlowFailures(ScriptedJudge):
    def __init__(self, clock: _Clock) -> None:
        super().__init__(lambda system_prompt, user_prompt: "no verdict")
        self.clock = clock

    def generate_once(self, prompts, **kwargs):
        self.clock.now += self.clock.seconds
        return super().generate_once(prompts, **kwargs)


def _judgments(tmp_path, monkeypatch, caplog, *, save_batch_size, **budget):
    """Judge two scenarios under one judge type, each attempt taking 21 minutes and failing."""
    clock = _Clock(21 * 60)
    monkeypatch.setattr(backend_module, "time", SimpleNamespace(monotonic=clock, sleep=lambda seconds: None))
    judge = _SlowFailures(clock)
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text("".join(json.dumps({"prompt": f"P{n}", "response": "R"}) + "\n" for n in range(2)))
    rules = tmp_path / "rules.jsonl"
    rules.write_text(json.dumps({"id": "Rule 1", "rule": "Be kind."}) + "\n")
    task = build_task_spec(OptimizeConfig(judge_types=JUDGE_TYPES).to_config(task="modelspec", rules=RULES, seeds=SEEDS))
    output = tmp_path / "evaluations.jsonl"
    with caplog.at_level("WARNING"):
        run_judgments(scenarios, rules, ValidatedJudge(judge, num_tries=5, max_retry_seconds=None), task,
                      ["No_Interpretation"], "full", output, save_batch_size=save_batch_size, num_tries=30,
                      raw_output_path=tmp_path / "judge_outputs.jsonl", **budget)
    return judge, lines(output)


def test_a_flush_stops_retrying_at_its_time_budget_and_drops_what_still_fails(tmp_path, monkeypatch, caplog):
    """The budget starts when the first attempt returns: at 21 minutes an attempt, an hour holds three more."""
    judge, rows = _judgments(tmp_path, monkeypatch, caplog, save_batch_size=2, max_retry_seconds=3600.0)
    assert len(judge.requests) == 2 * 4
    assert all(row["evaluations"]["scripted"]["full_ruleset"]["No_Interpretation"] is None for row in rows)
    assert any("Retry budget of 3600s exhausted" in record.getMessage() for record in caplog.records)
    assert any("2/2 prompts have no valid answer after 4 of 30 attempts; they are dropped" in record.getMessage()
               for record in caplog.records)


def test_each_flush_has_a_time_budget_of_its_own(tmp_path, monkeypatch, caplog):
    judge, _ = _judgments(tmp_path, monkeypatch, caplog, save_batch_size=1, max_retry_seconds=3600.0)
    assert len(judge.requests) == 2 * 4


def test_a_panel_with_no_time_budget_uses_every_attempt(tmp_path, monkeypatch, caplog):
    judge, _ = _judgments(tmp_path, monkeypatch, caplog, save_batch_size=1)
    assert len(judge.requests) == 2 * 30


def test_the_run_s_time_budget_reaches_the_panel(tmp_path, monkeypatch):
    """A flush whose every attempt takes 21 minutes: the run's config decides how many are made."""
    clock = _Clock(21 * 60)
    monkeypatch.setattr(backend_module, "time", SimpleNamespace(monotonic=clock, sleep=lambda seconds: None))

    class Script(FlakyPanel):
        def __call__(self, system_prompt, user_prompt):
            if RuleForgeScript.kind(system_prompt) == "judge":
                clock.now += clock.seconds
                return "no verdict"
            return super().__call__(system_prompt, user_prompt)

    judge = ScriptedJudge(Script())
    with pytest.raises(RuntimeError, match="Failed to generate any scenarios"):
        optimize(RULES, task=ModelSpecTask(), judge=judge, agent=ScriptedJudge(RuleForgeScript()), seeds=SEEDS,
                 out=tmp_path / "run", config=settings(max_episodes=0, panel_num_tries=30,
                                                       panel_max_retry_seconds=1.0))
    assert len(panel_prompts(judge)) < 30 * len(set(panel_prompts(judge)))


def test_the_panel_s_time_budget_is_an_hour_by_default_and_null_or_positive():
    assert OptimizeConfig().panel_max_retry_seconds == 3600.0
    run_config = OptimizeConfig().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    assert run_config.eval_cfg.panel_max_retry_seconds == 3600.0
    assert RewardSettings.from_config(run_config).panel_max_retry_seconds == 3600.0
    run_config.eval_cfg.panel_max_retry_seconds = None
    assert RewardSettings.from_config(run_config).panel_max_retry_seconds is None
    run_config.eval_cfg.panel_max_retry_seconds = 0
    with pytest.raises(ValueError, match="panel_max_retry_seconds=0: it must be positive"):
        RewardSettings.from_config(run_config)


def test_the_panel_retries_up_to_thirty_attempts_by_default_and_the_agents_five():
    assert OptimizeConfig().panel_num_tries == 30 and OptimizeConfig().num_tries == 5
    run_config = OptimizeConfig().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    assert run_config.eval_cfg.panel_num_tries == 30
    assert RewardSettings.from_config(run_config).panel_num_tries == 30


def test_the_amender_s_prompts_show_each_ruleset_s_text_and_no_path(tmp_path):
    out = tmp_path / "run"
    agent = ScriptedJudge(RuleForgeScript())
    judge = ScriptedJudge(RuleForgeScript())
    optimize(RULES, task=ModelSpecTask(), judge=judge, agent=agent, seeds=SEEDS, out=out, config=settings())
    amender = [request["user_prompt"] for request in agent.requests
               if RuleForgeScript.kind(request["system_prompt"]) in ("ra_sample", "ra_update")]
    assert len(amender) >= 4
    initial = build_rule_set_string_from_file(RULES)
    assert all(f"<TRAJECTORY_0>\n<ACTION>\n{initial}\n</ACTION>\n" in prompt for prompt in amender)
    kept = build_rule_set_string_from_file(out / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl")
    assert CLARIFIED in kept
    assert any(f"<TRAJECTORY_1>\n<ACTION>\n{kept}\n</ACTION>\n" in prompt for prompt in amender)
    for request in agent.requests + judge.requests:
        for text in (request["system_prompt"], request["user_prompt"]):
            assert str(tmp_path) not in text and ".jsonl" not in text


class CostlyClarity(RuleForgeScript):
    """The intent rubric scores every rule 4 once two are clarified: the clearest candidate costs the most."""

    def _intent(self, index, system_prompt, user_prompt):
        answer = super()._intent(index, system_prompt, user_prompt)
        revised = system_prompt[system_prompt.index("# Revised Ruleset:"):]
        if index and revised.count(CLARIFIED) >= 2:
            rules = json.loads(answer)["rules"]
            answer = json.dumps({"rules": [{**rule, "score": 4} for rule in rules]})
        return answer


def test_the_trigger_reads_the_kept_candidate_s_entropy_reward(tmp_path):
    """In episode 1 the clearer candidate (entropy 1.0) loses on its penalty to one at 0.728, below the trigger."""
    out = tmp_path / "run"
    run(out, settings(sg={"trigger_threshold": 0.9}), CostlyClarity)
    rewards = json.loads((out / "episode_1" / "rule_amendment" / "reward_all.jsonl").read_text(encoding="utf-8"))
    assert [round(rewards[index]["entropy_reward"], 3) for index in ("0", "1")] == [0.728, 1.0]
    assert lines(out / "episode_1" / "rule_amendment" / "reward_optimal.jsonl")[0]["optimal_rule_set_index"] == 0
    assert generated_in(out) == {0}


def trainer_for(tmp_path: Path, config: OptimizeConfig | None = None, *, judge_answer=None, agent_answer=None,
                logger=None, rules: Path = RULES, seeds=None, run_config=None, task=None) -> RuleAmendmentTrainer:
    """A trainer over ``config`` with scripted judges, its task spec built from the config unless ``task`` is given."""
    if run_config is None:
        run_config = (config or settings()).to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    options = {"num_tries": 5, "max_retry_seconds": None}
    return RuleAmendmentTrainer(
        run_config, task=build_task_spec(cfg=run_config) if task is None else task,
        judge=ValidatedJudge(ScriptedJudge(judge_answer or RuleForgeScript()), **options),
        agent=ValidatedJudge(ScriptedJudge(agent_answer or RuleForgeScript()), **options),
        rules=rules, seeds=SEEDS if seeds is None else seeds, output_dir=tmp_path / "out", logger=logger)


class Recorder:
    def __init__(self):
        self.logged = []

    def log(self, metrics, *, step):
        self.logged.append((dict(metrics), step))


def reward(entropy: float, effective: float) -> dict:
    return {"entropy_reward": entropy, "regularization_penalty": entropy - effective, "effective_reward": effective,
            "per_scenario_rewards": [], "penalty_breakdown": {}}


def test_the_candidate_with_the_highest_effective_reward_is_kept(tmp_path):
    """Three candidates: the clearest (entropy reward 0.9) loses on its penalty."""
    recorder = Recorder()
    trainer = trainer_for(tmp_path, logger=recorder)
    rewards = {0: reward(0.9, 0.2), 1: reward(0.5, 0.4), 2: reward(0.6, 0.4)}
    trainer.ra_psrl.posterior_sampling = lambda prev_rule_set_path, trace_path=None: ["patch 0", "patch 1", "patch 2"]
    trainer.ra_psrl.apply_patch_to_jsonl = lambda previous, patch: f"Rule 1: {patch.read_text()}"
    trainer.ra_psrl.posterior_update = lambda trace_path=None: "posterior"
    trainer.ra_env.get_reward = lambda rule_set_path, output_dir: rewards[int(output_dir.name.split("_")[1])]
    trainer.episode_num = 1
    ra_dir = tmp_path / "out" / "episode_1" / "rule_amendment"
    ra_dir.mkdir(parents=True)

    entropy, kept, terminated = trainer.ra_step(ra_dir=ra_dir, prev_rule_set_path=RULES)

    assert (entropy, kept, terminated) == (0.5, ra_dir / "optimal_rule_set.jsonl", False)
    assert lines(kept) == [{"id": "Rule 1", "rule": "patch 1"}]
    assert lines(ra_dir / "reward_optimal.jsonl") == [{"optimal_rule_set_index": 1, "reward": rewards[1]}]
    assert trainer.ra_env.reward_history == [0.4] and trainer.optimal_rule_set_path == kept
    assert trainer.ra_psrl.history[-1] == {"action": "Rule 1: patch 1", "reward": rewards[1]}
    assert [(metrics["rule_set_idx"], step) for metrics, step in recorder.logged] == [(0, 1), (1, 1), (2, 1)]


def ra_env(**ra) -> RuleAmendmentEnv:
    return RuleAmendmentEnv(task_spec=None, ra_cfg=OmegaConf.create(ra), backend=None, rewards=RewardSettings())


@pytest.mark.parametrize(("history", "stops"), [
    ([0.95, 0.95], False),            # fewer episodes than the window
    ([0.95, 0.95, 0.95], True),
    ([0.2, 0.95, 0.91, 0.95], True),  # ... over the last three alone
    ([0.95, 0.9, 0.95], False),       # at the threshold is not above it
    ([0.5, 0.4, 0.3], True),
    ([0.5, 0.4, 0.4], False),         # a plateau is not a decrease
    ([0.3, 0.4, 0.5], False),
])
def test_the_amender_stops_above_its_threshold_or_when_strictly_worse(history, stops):
    env = ra_env(termination_episodes=3, termination_threshold=0.9)
    for value in history:
        env.update_optimal_reward(value)
    assert env.check_termination() is stops


@pytest.mark.parametrize(("history", "stops"), [
    ([0.05, 0.05], False),
    ([0.05, 0.05, 0.05], True),
    ([0.9, 0.05, 0.0, 0.05], True),
    ([0.05, 0.1, 0.05], False),  # at the threshold is not below it
])
def test_the_generator_stops_below_its_threshold(history, stops):
    env = ScenarioGenerationEnv(task_spec=None, sg_cfg=OmegaConf.create(
        {"termination_episodes": 3, "termination_threshold": 0.1}), backend=None, rewards=RewardSettings())
    for value in history:
        env.update_optimal_reward(value)
    assert env.check_termination() is stops


@pytest.mark.parametrize(("config", "last_episode", "stopped_by"), [
    (settings(max_episodes=5, ra={"termination_episodes": 2, "termination_threshold": 0.5}), 2, ("rule_amendment",)),
    (settings(max_episodes=5, sg={"termination_episodes": 1, "termination_threshold": 0.1}), 2,
     ("scenario_generation",)),
    (settings(sg={"termination_episodes": 1, "termination_threshold": 1.0}), 0, ("scenario_generation",)),
    (settings(ra={"termination_episodes": 2, "termination_threshold": -1.0},
              sg={"termination_episodes": 2, "termination_threshold": 1.0, "trigger_threshold": -1.0}), 1,
     ("rule_amendment", "scenario_generation")),
    (settings(max_episodes=0), 0, ("max_episodes",)),
    (settings(), 2, ("max_episodes",)),
])
def test_a_run_stops_when_either_agent_s_rule_fires_or_at_max_episodes(tmp_path, config, last_episode, stopped_by):
    out = tmp_path / "run"
    result = run(out, config)
    assert (result.last_episode, result.stopped_by) == (last_episode, stopped_by)
    assert sorted(path.name for path in out.glob("episode_*")) == [f"episode_{n}" for n in range(last_episode + 1)]


@pytest.mark.parametrize("window", [1, 0])
def test_a_termination_window_below_two_is_refused_before_a_run_starts(tmp_path, window):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match=f"rule_amendment.termination_episodes={window}"):
        run(out, settings(ra={"termination_episodes": window}))
    assert not out.exists()
    with pytest.raises(ValueError, match=f"rule_amendment.termination_episodes={window}"):
        trainer_for(tmp_path, settings(ra={"termination_episodes": window}))


def test_a_single_rule_context_is_refused_since_the_panel_judges_the_whole_ruleset(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="eval_cfg.individual_rule_context='implicated': the panel judges"):
        run(out, settings(individual_rule_context="implicated"))
    assert not out.exists()


@pytest.mark.parametrize("window", [0, -2])
def test_a_generator_termination_window_below_one_is_refused_before_a_run_starts(tmp_path, window):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match=f"scenario_generation.termination_episodes={window}"):
        run(out, settings(sg={"termination_episodes": window}))
    assert not out.exists()


@pytest.mark.parametrize("section", ["ra", "sg"])
@pytest.mark.parametrize("window", [15.0, "15", True])
def test_a_termination_window_that_is_no_integer_is_refused_before_a_run_starts(tmp_path, section, window):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="termination_episodes=.*it must be an integer"):
        run(out, settings(**{section: {"termination_episodes": window}}))
    assert not out.exists()


def test_max_episodes_bounds_the_loop_on_its_own(tmp_path):
    trainer = trainer_for(tmp_path, settings(max_episodes=1))
    trainer.terminate_training = lambda ra_terminated, sg_terminated: False
    trainer.train()
    assert sorted(path.name for path in (tmp_path / "out").glob("episode_*")) == ["episode_0", "episode_1"]


def test_the_trainer_refuses_inputs_its_config_does_not_name(tmp_path):
    with pytest.raises(ValueError, match="hand it the files both read"):
        trainer_for(tmp_path, rules=FIXTURES / "good_seeds.jsonl")
    with pytest.raises(ValueError, match="hand it the files both read"):
        trainer_for(tmp_path, seeds={"good": SEEDS["good"]})


def fewer_rules(tmp_path: Path) -> Path:
    """RULES without its last rule (another ruleset the intent penalty would compare every revision with)."""
    path = tmp_path / "fewer.jsonl"
    path.write_text("".join(RULES.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    return path


def test_the_trainer_refuses_a_task_spec_not_built_from_its_config_as_it_stands(tmp_path):
    other = fewer_rules(tmp_path)
    first = settings().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    spec = build_task_spec(cfg=first)
    second = settings().to_config(task="modelspec", rules=other, seeds=SEEDS)
    with pytest.raises(ValueError, match=r"built from another config than the trainer's \(\['rule_amendment'\] "):
        trainer_for(tmp_path, run_config=second, task=spec, rules=other)
    first.rule_amendment.initial_rule_set_path = str(other)
    with pytest.raises(ValueError, match="or the config changed since"):
        trainer_for(tmp_path, run_config=first, task=spec, rules=other)
    first.rule_amendment.initial_rule_set_path = str(RULES)
    first.rule_amendment.intent_weight = 0.5
    with pytest.raises(ValueError, match=r"\['rule_amendment'\] differ\), or the config changed since"):
        trainer_for(tmp_path, run_config=first, task=spec)


@pytest.mark.parametrize(("key", "built", "trained"), [
    ("rule_amendment.intent_weight", "1.0", "0.5"),  # the reward the trainer would score: 0.5, its record 0.5 too
    ("eval_cfg.evaluation_temperature", "0.1", "0.7"),  # a setting the task spec caches when it is built
])
def test_the_trainer_refuses_a_task_spec_whose_config_resolved_otherwise(tmp_path, monkeypatch, key, built, trained):
    run_config = settings().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    OmegaConf.update(run_config, key, "${oc.env:REVIEW_SETTING}")
    monkeypatch.setenv("REVIEW_SETTING", built)
    spec = build_task_spec(cfg=run_config)
    monkeypatch.setenv("REVIEW_SETTING", trained)
    section = key.split(".")[0]
    with pytest.raises(ValueError, match=re.escape(f"(['{section}'] differ), or the config changed since")):
        trainer_for(tmp_path, run_config=run_config, task=spec)
    monkeypatch.setenv("REVIEW_SETTING", built)
    trainer = trainer_for(tmp_path, run_config=run_config, task=spec)
    assert trainer.rewards == RewardSettings.from_config(run_config)


def test_the_trainer_reads_its_rewards_from_its_config_once(tmp_path):
    trainer = trainer_for(tmp_path, settings(ra={"intent_weight": 0.25}))
    assert trainer.rewards.weights == RewardWeights(intent=0.25)
    assert trainer.ra_env.rewards is trainer.sg_env.rewards is trainer.rewards
    assert trainer.ra_psrl.weights is trainer.rewards.weights
    assert "entropy_reward - (0.0*(smrc_penalty) + 0.25*(intent_penalty))" in trainer.ra_psrl.prior


@pytest.mark.parametrize("attribute", ["initial_rule_set_path", "good_seed_scenario_path", "bad_seed_scenario_path"])
def test_the_trainer_refuses_files_its_task_spec_does_not_read(tmp_path, attribute):
    run_config = settings().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    spec = build_task_spec(cfg=run_config)
    setattr(spec, attribute, fewer_rules(tmp_path))
    with pytest.raises(ValueError, match=r"its task spec reads .*: hand it the files both read"):
        trainer_for(tmp_path, run_config=run_config, task=spec)


def test_the_trainer_refuses_a_task_spec_that_reads_no_initial_ruleset(tmp_path):
    run_config = settings().to_config(task="modelspec", rules=RULES, seeds=SEEDS)
    adapter = TaskSpecAdapter(ModelSpecTask(), run_config)
    assert (adapter.initial_rule_set_path, adapter.seed_scenario_paths()) == (None, {})
    with pytest.raises(ValueError, match=r"task spec \(TaskSpecAdapter\) reads no initial ruleset"):
        trainer_for(tmp_path, run_config=run_config, task=adapter)


def test_the_trainer_refuses_an_initial_ruleset_with_two_ids_for_one_rule(tmp_path):
    """``Rule 7`` and ``Rule 07`` are one rule to the metrics (both shown to a judge as Rule 7)."""
    rules = tmp_path / "rules.jsonl"
    rules.write_text('{"id": "Rule 7", "rule": "Be kind"}\n{"id": "Rule 07", "rule": "Be brief"}\n',
                     encoding="utf-8")
    run_config = settings().to_config(task="modelspec", rules=rules, seeds=SEEDS)
    with pytest.raises(ValueError, match=re.escape(f"the initial ruleset {rules}: 'Rule 07' names the rule 'Rule 7' "
                                                  "names")):
        trainer_for(tmp_path, run_config=run_config, rules=rules)


@pytest.mark.parametrize(("rows", "named"), [
    # A rule header in a rule's text: episode 0 would write Rule 1, Rule 2, Rule 2.
    ([{"id": "Rule 1", "rule": "Apply Rule 2: be kind."}, {"id": "Rule 2", "rule": "Be brief."}],
     "its rule 0 ('Rule 1', 'Apply Rule 2: be kind.') would come back as ('Rule 1', 'Apply')"),
    ([{"id": "Guideline A", "rule": "Be honest."}], "its rule 0 ('Guideline A', 'Be honest.') would come back as "
                                                    "('line_1', 'Guideline A: Be honest.')"),
    ([{"id": "Rule 1", "rule": "Be kind.\n\nAlways."}], "would come back as ('Rule 1', 'Be kind.\\nAlways.')"),
])
def test_the_trainer_refuses_an_initial_ruleset_its_own_text_would_change(tmp_path, rows, named):
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    run_config = settings().to_config(task="modelspec", rules=rules, seeds=SEEDS)
    with pytest.raises(ValueError, match=re.escape(f"the initial ruleset {rules}: ") + ".*" + re.escape(named)):
        trainer_for(tmp_path, run_config=run_config, rules=rules)
    assert not (tmp_path / "out").exists()


def test_the_trainer_takes_an_id_that_comes_back_as_the_rule_it_names(tmp_path):
    """``openai_rules7`` is shown, scored and saved as ``Rule 7``, the rule every metric keys it by."""
    rules = tmp_path / "rules.jsonl"
    rules.write_text('{"id": "openai_rules1", "rule": "Be kind."}\n{"id": "openai_rules2", "rule": "Be brief."}\n',
                     encoding="utf-8")
    run_config = settings().to_config(task="modelspec", rules=rules, seeds=SEEDS)
    assert trainer_for(tmp_path, run_config=run_config, rules=rules).initial_rule_set_path == rules


def ra_step_over(trainer, patches, *, patched=None):
    """One rule-amendment step over ``patches``, its candidates scored by index; returns its directory and result."""
    trainer.ra_psrl.posterior_sampling = lambda prev_rule_set_path, trace_path=None: patches
    if patched is not None:
        trainer.ra_psrl.apply_patch_to_jsonl = patched
    trainer.ra_psrl.posterior_update = lambda trace_path=None: "posterior"
    scored = []

    def get_reward(rule_set_path, output_dir):
        scored.append(output_dir.name)
        return reward(0.5, 0.5)

    trainer.ra_env.get_reward = get_reward
    trainer.episode_num = 1
    ra_dir = trainer.output_dir / "episode_1" / "rule_amendment"
    ra_dir.mkdir(parents=True)
    return ra_dir, scored, trainer.ra_step(ra_dir=ra_dir, prev_rule_set_path=RULES)


def test_a_patch_that_gives_two_rules_one_id_is_skipped_before_it_is_scored(tmp_path):
    trainer = trainer_for(tmp_path)
    ra_dir, scored, (_, kept, _) = ra_step_over(
        trainer, ["two ids", "fine"],
        patched=lambda previous, patch: ("Rule 1: Be kind\nRule 2: Be brief\nRule 1: Be fair."
                                         if patch.read_text() == "two ids" else "Rule 1: Be kind"))
    assert scored == ["sample_1"]
    assert lines(kept) == [{"id": "Rule 1", "rule": "Be kind"}]
    assert not (ra_dir / "sample_0" / "posterior_sample.jsonl").exists()
    assert re.search(r"the ruleset patch .*sample_0/posterior_sample\.patch makes: 'Rule 1' names the rule 'Rule 1' "
                     "names", (ra_dir / "sample_0" / "patch_error.txt").read_text())


@pytest.mark.parametrize(("patch", "error"), [
    ("no markers here", "No SEARCH/REPLACE block found"),
    ("<<<<<<< SEARCH\nRule 77: not in the ruleset at all\n=======\nRule 77: x\n>>>>>>> REPLACE\n",
     "Search text not found in text"),
    ("<<<<<<< SEARCH\n=======\nRule 9: x\n>>>>>>> REPLACE\n", "A SEARCH section cannot be empty"),
])
def test_a_patch_that_cannot_be_applied_is_skipped_and_its_error_kept(tmp_path, patch, error):
    trainer = trainer_for(tmp_path)
    first = build_rule_set_string_from_file(RULES).splitlines()[0]
    fine = f"<<<<<<< SEARCH\n{first}\n=======\n{first} {CLARIFIED}\n>>>>>>> REPLACE\n"
    ra_dir, scored, (_, kept, _) = ra_step_over(trainer, [patch, fine])
    assert scored == ["sample_1"]
    assert CLARIFIED in lines(kept)[0]["rule"]
    assert error in (ra_dir / "sample_0" / "patch_error.txt").read_text()
    assert (ra_dir / "sample_0" / "posterior_sample.patch").read_text() == patch


def test_when_every_patch_fails_the_previous_ruleset_stands(tmp_path):
    trainer = trainer_for(tmp_path)
    ra_dir, scored, (_, kept, terminated) = ra_step_over(trainer, ["nothing", "nor this"])
    assert scored == ["sample_2"] and terminated is False
    assert kept.read_text() == RULES.read_text()
    assert lines(ra_dir / "reward_optimal.jsonl")[0]["optimal_rule_set_index"] == 2
    assert trainer.ra_psrl.history[-1]["action"] == build_rule_set_string_from_file(RULES)


@pytest.mark.parametrize(("patched", "error"), [
    ("Rule 1: See Rule 3: this is a quoted example.", "would come back as ('Rule 1', 'See')"),
    ("Rule 1: Be kind. Rule 2: Be brief.", "would come back as ('Rule 1', 'Be kind.')"),
    ("there are no rules", "has text before its first rule header"),
    ("A note first.\nRule 1: Be kind.", "has text before its first rule header"),
])
def test_a_patch_whose_text_is_not_rules_is_skipped(tmp_path, patched, error):
    trainer = trainer_for(tmp_path)
    ra_dir, scored, (_, kept, _) = ra_step_over(
        trainer, ["bad", "fine"],
        patched=lambda previous, patch: patched if patch.read_text() == "bad" else "Rule 1: Be kind")
    assert scored == ["sample_1"]
    assert lines(kept) == [{"id": "Rule 1", "rule": "Be kind"}]
    assert not (ra_dir / "sample_0" / "posterior_sample.jsonl").exists()
    assert error in (ra_dir / "sample_0" / "patch_error.txt").read_text()


def test_a_patch_whose_rule_runs_over_lines_is_kept_as_written(tmp_path):
    trainer = trainer_for(tmp_path)
    _, scored, (_, kept, _) = ra_step_over(
        trainer, ["fine"], patched=lambda previous, patch: "Meta-Rule 1: Order them.\n\nRule 1: Be kind\n  to all.\n")
    assert scored == ["sample_0"]
    assert lines(kept) == [{"id": "Meta-Rule 1", "rule": "Order them."}, {"id": "Rule 1", "rule": "Be kind\nto all."}]


def test_a_patch_that_leaves_no_rule_is_skipped(tmp_path):
    trainer = trainer_for(tmp_path)
    ra_dir, scored, _ = ra_step_over(trainer, ["all gone"], patched=lambda previous, patch: "")
    assert scored == ["sample_1"]
    assert "leaves no rule" in (ra_dir / "sample_0" / "patch_error.txt").read_text()


class EveryPatchFails(RuleForgeScript):
    def _ra_sample(self, index, system_prompt, user_prompt):
        return "<<<<<<< SEARCH\nRule 99: nothing like this is in the ruleset\n=======\nRule 99: x\n>>>>>>> REPLACE"


def test_a_run_whose_patches_all_fail_keeps_going(tmp_path):
    out = tmp_path / "run"
    result = run(out, settings(), EveryPatchFails)
    assert result.last_episode == 2 and result.stopped_by == ("max_episodes",)
    assert result.ruleset_path.read_text() == (out / "episode_0" / "rule_amendment" / "optimal_rule_set.jsonl").read_text()


class ShortRubric(RuleForgeScript):
    """A rubric that leaves a rule unscored once a revision is on the table."""

    def _intent(self, index, system_prompt, user_prompt):
        answer = json.loads(super()._intent(index, system_prompt, user_prompt))
        revised = system_prompt[system_prompt.index("# Revised Ruleset:"):]
        return json.dumps({"rules": answer["rules"][:-1]}) if CLARIFIED in revised else json.dumps(answer)


def test_a_candidate_the_rubric_will_not_score_is_skipped_and_its_error_kept(tmp_path):
    out = tmp_path / "run"
    result = run(out, settings(), ShortRubric)
    assert result.last_episode == 2 and result.ruleset_path.read_text() == (
        out / "episode_0" / "rule_amendment" / "optimal_rule_set.jsonl").read_text()
    amended = out / "episode_1" / "rule_amendment"
    errors = sorted(path.parent.name for path in amended.glob("sample_*/patch_error.txt"))
    assert errors == ["sample_0", "sample_1"]
    assert "did not score every revised rule" in (amended / "sample_1" / "patch_error.txt").read_text()
    assert not (amended / "sample_1" / "intent_penalty.json").exists()
    assert list(json.loads((amended / "reward_all.jsonl").read_text())) == ["2"]


class NoRubric(RuleForgeScript):
    def _intent(self, index, system_prompt, user_prompt):
        return "no JSON"


def test_a_ruleset_nothing_can_score_ends_the_run(tmp_path):
    with pytest.raises(ValueError, match="no candidate of episode 0 could be scored: .*did not score every revised"):
        run(tmp_path / "run", settings(), NoRubric)


class RunawayAmender(RuleForgeScript):
    """A thought that never closes but holds a parseable block, and updates of that kind for every attempt."""

    def _ra_sample(self, index, system_prompt, user_prompt):
        block = super()._ra_sample(index + 1, system_prompt, user_prompt)
        return f"<|channel>thought\nPerhaps:\n{block}" if index < 2 else block

    def _ra_update(self, index, system_prompt, user_prompt):
        return "<|channel>thought\nla la" if index < 5 else super()._ra_update(index, system_prompt, user_prompt)


def test_an_unclosed_thought_never_reaches_the_patch_parser(tmp_path):
    out = tmp_path / "run"
    optimize(RULES, task=ModelSpecTask(), judge=ScriptedJudge(RunawayAmender()),
             agent=ScriptedJudge(RunawayAmender(), strip_thinking_model="gemma4"), seeds=SEEDS, out=out,
             config=settings(max_episodes=1))
    amended = out / "episode_1" / "rule_amendment"
    for name in ("sample_0", "sample_1"):
        assert "holds an unclosed thought" in (amended / name / "patch_error.txt").read_text()
        assert not (amended / name / "posterior_sample.jsonl").exists()
    assert list(json.loads((amended / "reward_all.jsonl").read_text())) == ["2"]
    assert lines(amended / "optimal_rule_set.jsonl") == lines(RULES)
    assert lines(amended / "prior.jsonl")[0]["prior"] == lines(out / "episode_0" / "rule_amendment" / "prior.jsonl")[0]["prior"]
    [trace] = lines(amended / "prior_trace.jsonl")
    assert len(trace["attempts"]) == 5 and len(trace["retries"]) == 4


def test_an_agent_that_never_answers(tmp_path):
    trainer = trainer_for(tmp_path, agent_answer=lambda system_prompt, user_prompt: "")
    for agent in (trainer.ra_psrl, trainer.sg_psrl):
        prior = agent.prior
        assert agent.posterior_update() == prior == agent.prior
    with pytest.raises(ValueError, match="No output from posterior_sampling generation in RA agent"):
        trainer.ra_psrl.posterior_sampling(prev_rule_set_path=RULES)
    with pytest.raises(ValueError, match="No output from posterior_sampling generation in SG agent"):
        trainer.sg_psrl.posterior_sampling(current_rule_set="Rule 1: Be kind", already_generated_scenarios=[])


def test_the_environments_need_a_reset_first(tmp_path):
    trainer = trainer_for(tmp_path)
    with pytest.raises(ValueError, match="Call reset"):
        trainer.ra_env.get_reward(rule_set_path=RULES, output_dir=tmp_path)
    with pytest.raises(ValueError, match="Call reset"):
        trainer.sg_env.get_reward(scenario_set_path=RULES, output_dir=tmp_path)
    with pytest.raises(ValueError, match="Call reset"):
        trainer.sg_env.get_obs()


class Mute(RuleForgeScript):
    """A scenario generator whose batches hold no scenario."""

    def _sg_sample(self, index, system_prompt, user_prompt):
        return "No scenarios today."


def test_a_generator_that_writes_no_scenario_stops_the_run(tmp_path):
    with pytest.raises(RuntimeError, match="Failed to generate any scenarios after 2 batches"):
        run(tmp_path / "run", settings(), Mute)
