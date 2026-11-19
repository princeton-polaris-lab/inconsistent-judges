from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ruleforge.optimize import OptimizeRun, select_best
from ruleforge.optimize.api import RUN_RECORD_DIR, RUN_RECORD_FNAME
from ruleforge.optimize.select import (
    RunSummary,
    contiguous_episode_rewards,
    exclusion,
    extract_metrics,
    rank,
    run_seed,
    summarize,
    sweep_table,
)

REPO = Path(__file__).resolve().parents[1]
COMMAND = ["-m", "ruleforge", "optimize", "select"]
TORN = "torn"


def _reward(effective: float) -> dict:
    return {"effective_reward": effective, "entropy_reward": effective + 0.25, "regularization_penalty": 0.25,
            "penalty_breakdown": {"intent": {"value": 0.25, "weight": 1.0}}}


def make_run(run: Path, rewards: list, *, seed=None, agent_seed=None, ruleset: bool = True) -> Path:
    """A run directory as the trainer lays it out."""
    for episode, effective in enumerate(rewards):
        amendment = run / f"episode_{episode}" / "rule_amendment"
        amendment.mkdir(parents=True)
        if effective is None:
            continue
        reward_file = amendment / "reward_optimal.jsonl"
        if effective == TORN:
            reward_file.write_text('{"reward": {"effective_re', encoding="utf-8")
            continue
        reward_file.write_text(json.dumps({"reward": _reward(effective)}) + "\n", encoding="utf-8")
        if ruleset or episode < len(rewards) - 1:
            (amendment / "optimal_rule_set.jsonl").write_text(
                json.dumps({"id": "Rule 1", "rule": f"{run.name} episode {episode}"}) + "\n", encoding="utf-8")
    (run / ".hydra").mkdir(parents=True, exist_ok=True)
    config = {"judge_model": {"seed": seed}, "agent_model": {"seed": agent_seed}}
    (run / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return run


def test_a_reward_file_is_flattened_and_a_torn_or_missing_one_is_none(tmp_path):
    path = tmp_path / "reward_optimal.jsonl"
    path.write_text(json.dumps({"reward": _reward(0.5)}) + "\n", encoding="utf-8")
    assert extract_metrics(path) == {"effective_reward": 0.5, "entropy_reward": 0.75, "regularization_penalty": 0.25,
                                     "intent_penalty": 0.25}
    path.write_text('{"reward": {"effective_re', encoding="utf-8")
    assert extract_metrics(path) is None
    path.write_text('{"reward": {"effective_reward": 1}}\n', encoding="utf-8")
    assert extract_metrics(path) is None
    assert extract_metrics(tmp_path / "absent.jsonl") is None


def test_the_complete_episodes_run_in_order_to_the_first_without_a_reward(tmp_path):
    run = make_run(tmp_path / "run", [0.1, 0.2, None, 0.4])
    (run / "episode_x").mkdir()
    (run / "episode_9").write_text("not a directory", encoding="utf-8")
    assert contiguous_episode_rewards(run) == [
        (n, run / f"episode_{n}" / "rule_amendment" / "reward_optimal.jsonl") for n in (0, 1)]
    # the directories there, in order: a number missing between two is not a gap
    gapped = make_run(tmp_path / "gapped", [0.1, 0.2, 0.3, 0.4])
    for n in (0, 2):
        shutil.rmtree(gapped / f"episode_{n}")
    assert [n for n, _ in contiguous_episode_rewards(gapped)] == [1, 3]
    assert summarize(gapped).last_episode == 3


def test_a_runs_seed_is_its_judges_else_its_agents_from_whichever_record_it_has(tmp_path, monkeypatch):
    assert run_seed(make_run(tmp_path / "a", [0.5], seed=3, agent_seed=4)) == 3
    assert run_seed(make_run(tmp_path / "b", [0.5], seed=None, agent_seed=4)) == 4
    assert run_seed(make_run(tmp_path / "c", [0.5], seed="x")) is None
    (tmp_path / "d" / ".hydra").mkdir(parents=True)
    (tmp_path / "d" / ".hydra" / "config.yaml").write_text("", encoding="utf-8")
    assert run_seed(tmp_path / "d") is None
    (tmp_path / "e" / ".hydra").mkdir(parents=True)
    (tmp_path / "e" / ".hydra" / "config.yaml").write_text("judge_model: gemma\n", encoding="utf-8")
    assert run_seed(tmp_path / "e") is None
    assert run_seed(tmp_path / "f") is None

    def optimizer_run(name: str, record: dict) -> Path:
        (tmp_path / name / RUN_RECORD_DIR).mkdir(parents=True)
        (tmp_path / name / RUN_RECORD_DIR / RUN_RECORD_FNAME).write_text(json.dumps(record), encoding="utf-8")
        return tmp_path / name

    assert run_seed(optimizer_run("g", {"judge": {"options": {"seed": 7}}, "agent": "judge"})) == 7
    assert run_seed(optimizer_run("h", {"judge": {"options": {"seed": None}},
                                        "agent": {"options": {"seed": 8}}})) == 8
    assert run_seed(optimizer_run("i", {"judge": {"options": {}}, "agent": "judge"})) is None
    assert run_seed(tmp_path / "judge.seed.5") == 5
    assert run_seed(optimizer_run("agent.seed.6", {"judge": {"options": {}}, "agent": "judge"})) == 6
    assert run_seed(optimizer_run("judge.seed.2", {"judge": {"options": {"seed": 9}}, "agent": "judge"})) == 9
    assert run_seed(optimizer_run("agent.seed.3", {"judge": {"options": {"seed": 16}},
                                                   "agent": {"options": {"seed": 3}}})) == 3
    assert run_seed(optimizer_run("agent.seed.4", {"judge": {"options": {"seed": 16}}, "agent": "judge"})) == 16
    assert run_seed(optimizer_run("agent.seed.7", {"judge": {"options": {"seed": 16}},
                                                   "agent": {"options": {}}})) == 7
    assert run_seed(make_run(tmp_path / "agent.seed.1", [0.5], seed=16, agent_seed=5)) == 5
    assert run_seed(make_run(tmp_path / "judge.seed.8", [0.5], seed=None, agent_seed=4)) == 8
    assert run_seed(tmp_path / "judge.seed.x") is None and run_seed(tmp_path / "judge.seed.5.runs") is None

    def seeds(judge, agent) -> dict:
        return {"judge": {"options": {"seed": judge}}, "agent": {"options": {"seed": agent}}}

    # A seed given in Hydra's spelling with underscores (--sweep agent.seed=1_000) names its member so: that axis's.
    assert run_seed(optimizer_run("agent.seed.1_000", seeds(16, 1000))) == 1000
    assert run_seed(tmp_path / "judge.seed.1_0_0") == 100 and run_seed(tmp_path / "judge.seed.1__0") is None

    assert run_seed(optimizer_run("n/judge.seed.3/agent.seed.4", seeds(3, 4))) == 4
    assert run_seed(optimizer_run("n/agent.seed.2/rule_amendment.best_of_n.2", seeds(16, 2))) == 16
    assert run_seed(optimizer_run("n/agent.seed.6/rule_amendment.best_of_n.2", seeds(16, None))) == 16
    assert run_seed(optimizer_run("n/agent.seed.5/judge.temperature.0.5", seeds(1, 5))) == 1
    assert run_seed(optimizer_run("n/agent.seed.8/judge.temperature.0.5", seeds(None, 5))) == 5
    three = "n/agent.seed.10/rule_amendment.best_of_n.2/scenario_generation.trigger_threshold.0.7"
    assert run_seed(optimizer_run(three, seeds(16, 10))) == 16
    assert run_seed(optimizer_run("n/judge.seed.5/eval_cfg.x.y/agent.seed.1/judge.temperature.0.1", seeds(5, 1))) == 5
    monkeypatch.chdir(tmp_path / "n" / "agent.seed.2")  # nor, for a relative path, the directory it is read in
    assert run_seed(Path("rule_amendment.best_of_n.2")) == 16
    assert run_seed(optimizer_run("q/agent.seed.7/x.1/y.2", seeds(16, 7))) == 16
    assert run_seed(optimizer_run("q/agent.seed.7/mine/rule_amendment.best_of_n.2", seeds(16, 7))) == 16
    assert run_seed(optimizer_run("runs/agent.seed.3/1", seeds(1, 3))) == 1
    assert run_seed(optimizer_run("runs/agent.seed.3/2", seeds(2, 3))) == 2
    assert run_seed(make_run(tmp_path / "m" / "agent.seed.9" / "0", [0.5], seed=16, agent_seed=9)) == 16
    assert run_seed(make_run(tmp_path / "m" / "agent.seed.9" / "rule_amendment.best_of_n.2", [0.5], seed=16,
                             agent_seed=9)) == 16
    assert run_seed(optimizer_run("jobs/16", seeds(3, 16))) == 3
    assert run_seed(optimizer_run("jobs/2", seeds(7, 2))) == 7
    assert run_seed(optimizer_run("exp/16/1", seeds(1, 16))) == 1
    assert run_seed(optimizer_run("exp/16/3", seeds(3, 16))) == 3
    assert run_seed(optimizer_run("t/10", seeds(16, 10))) == 16  # an --out template naming {agent.seed} alone
    assert run_seed(optimizer_run("v/10", seeds(None, 10))) == 10  # ... under a judge with no seed: the agent's
    assert run_seed(optimizer_run("t/8", {"judge": {"options": {"seed": 8}}, "agent": "judge"})) == 8
    assert run_seed(make_run(tmp_path / "multirun" / "2", [0.5], seed=16, agent_seed=2)) == 16


def _recorded_run(path, *, config_seed, record):
    run = make_run(path, [0.5, 0.6], seed=config_seed)
    (run / RUN_RECORD_DIR).mkdir()
    (run / RUN_RECORD_DIR / RUN_RECORD_FNAME).write_text(json.dumps(record), encoding="utf-8")
    return run


def test_a_run_s_record_and_config_that_disagree_on_its_seed_are_refused(tmp_path):
    record = {"judge": {"options": {"seed": 7}}, "agent": "judge"}
    run = _recorded_run(tmp_path / "sweep" / "0", config_seed=9, record=record)
    with pytest.raises(ValueError, match=r"its run record holds judge seed 7 .* holds 9"):
        run_seed(run)
    with pytest.raises(ValueError, match="its run record holds judge seed 7"):
        sweep_table([tmp_path / "sweep"])


def test_a_run_s_record_gives_its_seed_ahead_of_its_config(tmp_path):
    assert run_seed(_recorded_run(tmp_path / "a", config_seed=7,
                                  record={"judge": {"options": {"seed": 7}}, "agent": "judge"})) == 7
    # agents on the judge's engine: the agents' own config seed is not used, so it is no conflict
    run = _recorded_run(tmp_path / "b", config_seed=7, record={"judge": {"options": {"seed": 7}}, "agent": "judge"})
    config = yaml.safe_load((run / ".hydra" / "config.yaml").read_text())
    config["agent_model"] = {"seed": 99}
    (run / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config))
    assert run_seed(run) == 7
    assert run_seed(_recorded_run(tmp_path / "c", config_seed=5, record={"judge": {"options": {}}, "agent": "judge"})) == 5
    separate = {"judge": {"options": {}}, "agent": {"options": {"seed": 3}}}
    run = _recorded_run(tmp_path / "d", config_seed=5, record=separate)
    config = yaml.safe_load((run / ".hydra" / "config.yaml").read_text())
    config["judge_model"], config["agent_model"] = {}, {"seed": 4}
    (run / ".hydra" / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="its run record holds agent seed 3"):
        run_seed(run)


def test_a_sweep_s_runs_are_its_numbered_jobs_its_recorded_runs_and_its_seed_members(tmp_path):
    from ruleforge.cli import optimize as optimize_cli
    from ruleforge.cli import records
    from ruleforge.optimize import select as select_module

    sweep = tmp_path / "sweep"
    make_run(sweep / "0", [0.5, 0.6], seed=1)
    member = make_run(sweep / "judge.max_new_tokens.4", [0.5, 0.7])
    shutil.rmtree(member / ".hydra")
    (member / RUN_RECORD_DIR).mkdir()
    (member / RUN_RECORD_DIR / RUN_RECORD_FNAME).write_text(json.dumps({"judge": {"options": {"seed": 3}}}))
    shutil.rmtree(make_run(sweep / "judge.seed.2", [0.5, 0.6]) / ".hydra")
    make_run(sweep / "notes", [0.5, 0.9])  # named and recorded as no sweep names a run: not read
    shutil.rmtree(sweep / "notes" / ".hydra")
    (sweep / "judge.seed.3" / "judge.temperature.0.1").mkdir(parents=True)
    (sweep / "selection.json").write_text("{}")
    for written in (".submitit", "selection.json.runs", "judge.seed.2.runs", "wandb"):
        (sweep / written / "x").mkdir(parents=True)
    named = make_run(sweep / "1.runs", [0.5, 0.65])  # a member as an --out template names it: '{judge.seed}.runs'
    shutil.rmtree(named / ".hydra")
    (named / RUN_RECORD_DIR).mkdir()
    (named / RUN_RECORD_DIR / RUN_RECORD_FNAME).write_text(json.dumps({"judge": {"options": {"seed": 1}}}))
    assert select_module._RECORDS_SUFFIX == records.RUNS_SUFFIX
    assert select_module._WANDB_DIRNAME == optimize_cli.WANDB_DIRNAME
    rows, notes = sweep_table([sweep])
    assert [(row["job_num"], row["seed"], row["effective_reward"]) for row in rows] == [
        (None, 3, 0.7), (None, 1, 0.65), (0, 1, 0.6), (None, 2, 0.6)]
    assert notes == [
        f"SKIP {sweep / 'judge.seed.3'}: not a run: named as a seed axis's member, with no run record or episode in "
        "it (a sweep over several keys holds its runs a directory further down)",
        f"SKIP {sweep / 'notes'}: not a run: no job number for its name, no run record in it"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root may read every directory")
def test_a_directory_the_user_may_not_read_is_noted_and_the_rest_of_its_sweep_tabled(tmp_path):
    sweep = tmp_path / "sweep"
    make_run(sweep / "0", [0.5, 0.6], seed=1)
    modes = {"7": 0, ".hidden": 0, "locked": 0, "x.runs": 0, "judge.seed.2": 0o100, "listless": 0o100}
    for name in modes:
        (sweep / name / "episode_0").mkdir(parents=True)
    (sweep / "listless" / ".hydra").mkdir()
    (sweep / "listless" / ".hydra" / "config.yaml").write_text("judge_model: {seed: 3}\n", encoding="utf-8")
    try:
        for name, mode in modes.items():
            (sweep / name).chmod(mode)
        rows, notes = sweep_table([sweep])
    finally:
        for name in modes:
            (sweep / name).chmod(0o700)
    denied = os.strerror(errno.EACCES)
    assert [(row["job_num"], row["seed"]) for row in rows] == [(0, 1)]
    assert notes == [f"SKIP {sweep / '7'}: no complete episode",
                     f"SKIP {sweep / 'judge.seed.2'}: cannot be read ({denied})",
                     f"SKIP {sweep / 'listless'}: no complete episode",
                     *(f"SKIP {sweep / name}: cannot be read ({denied})" for name in ("locked", "x.runs"))]


@pytest.mark.skipif(os.geteuid() == 0, reason="root may read every directory")
@pytest.mark.parametrize("locked", ["episode_1", ".hydra/config.yaml", f"{RUN_RECORD_DIR}/{RUN_RECORD_FNAME}"],
                         ids=["episode", "hydra-record", "run-record"])
def test_a_run_holding_a_part_the_user_may_not_read_is_noted_and_the_rest_of_its_sweep_tabled(tmp_path, locked):
    sweep = tmp_path / "sweep"
    make_run(sweep / "0", [0.5, 0.6], seed=1)
    parts = []
    for name in ("7", "judge.seed.2", "mine"):
        if locked.startswith(RUN_RECORD_DIR):
            recorded_run(sweep / name, judge_seed=2, agent_seed=2)
        else:
            make_run(sweep / name, [0.5, 0.7], seed=2)
        parts.append(sweep / name / locked)
    try:
        for part in parts:
            part.chmod(0)
        rows, notes = sweep_table([sweep])
    finally:
        for part in parts:
            part.chmod(0o700)
    denied = os.strerror(errno.EACCES)
    assert [(row["job_num"], row["seed"]) for row in rows] == [(0, 1)]
    assert notes == [f"SKIP {sweep / name}: cannot be read ({denied})" for name in ("7", "judge.seed.2", "mine")]
    assert len(sweep_table([sweep])[0]) == 4


def recorded_run(run: Path, *, judge_seed: int, agent_seed: int) -> Path:
    """A run as :func:`make_run` lays it out, with the optimizer's run record naming each role's seed."""
    make_run(run, [0.5, 0.6])
    shutil.rmtree(run / ".hydra")
    (run / RUN_RECORD_DIR).mkdir()
    (run / RUN_RECORD_DIR / RUN_RECORD_FNAME).write_text(json.dumps(
        {"judge": {"options": {"seed": judge_seed}}, "agent": {"options": {"seed": agent_seed}}}), encoding="utf-8")
    return run


def test_a_judge_seed_sweep_in_a_directory_named_by_an_agent_seed_is_tabled_by_its_judges_seeds(tmp_path):
    for judge_seed in (2, 1):
        recorded_run(tmp_path / "runs" / "agent.seed.3" / str(judge_seed), judge_seed=judge_seed, agent_seed=3)
    rows, notes = sweep_table([tmp_path / "runs" / "agent.seed.3"])
    assert [(row["job_num"], row["seed"]) for row in rows] == [(1, 1), (2, 2)] and notes == []
    for job, judge_seed in ((0, 2), (1, 1)):
        make_run(tmp_path / "hydra" / "agent.seed.3" / str(job), [0.5, 0.6], seed=judge_seed, agent_seed=3)
    rows, notes = sweep_table([tmp_path / "hydra" / "agent.seed.3"])
    assert [(row["job_num"], row["seed"]) for row in rows] == [(1, 1), (0, 2)] and notes == []


def test_runs_named_as_another_key_s_members_under_an_agent_seed_directory_are_ranked_by_their_judges_seeds(tmp_path):
    sweep = tmp_path / "runs" / "agent.seed.3"
    runs = [recorded_run(sweep / f"judge.temperature.{judge_seed}", judge_seed=judge_seed, agent_seed=3)
            for judge_seed in (10, 2)]
    assert [run_seed(run) for run in runs] == [10, 2]
    winner = runs[1] / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl"
    assert select_best(runs).path == winner and select_best(runs[::-1]).path == winner
    assert [summary.run for summary in rank(summarize(run) for run in runs)] == [runs[1], runs[0]]
    rows, notes = sweep_table([sweep])
    assert [(row["seed"], row["job_num"]) for row in rows] == [(2, None), (10, None)] and notes == []
    assert rows[0]["optimal_rule_set_path"] == str(winner.resolve())


def test_a_run_is_summarized_by_its_last_complete_episode_and_its_peak(tmp_path):
    run = make_run(tmp_path / "run", [0.25, 0.75, 0.5, TORN, 0.9], seed=2)
    summary = summarize(run)
    assert summary == RunSummary(
        run=run, last_episode=2, reward_path=run / "episode_2" / "rule_amendment" / "reward_optimal.jsonl",
        metrics={"effective_reward": 0.5, "entropy_reward": 0.75, "regularization_penalty": 0.25,
                 "intent_penalty": 0.25},
        peak_episode=1, peak_effective_reward=0.75,
        unreadable=run / "episode_3" / "rule_amendment" / "reward_optimal.jsonl")
    assert summary.ruleset_path == run / "episode_2" / "rule_amendment" / "optimal_rule_set.jsonl"
    assert summary.seed == 2
    assert len({summary, summarize(run)}) == 1
    assert summarize(make_run(tmp_path / "flat", [0.5, 0.5])).peak_episode == 0


def test_a_run_with_no_complete_episode_has_no_last_episode(tmp_path):
    torn = summarize(make_run(tmp_path / "torn", [TORN], seed=1))
    assert (torn.last_episode, torn.reward_path, torn.ruleset_path, torn.metrics) == (None, None, None, {})
    assert torn.unreadable == tmp_path / "torn" / "episode_0" / "rule_amendment" / "reward_optimal.jsonl"
    assert summarize(tmp_path / "empty").unreadable is None


def test_an_optimize_run_is_read_from_its_out(tmp_path):
    run = make_run(tmp_path / "run", [0.5, 0.6], seed=5)
    result = OptimizeRun(out=run, last_episode=1, stopped_by=("max_episodes",),
                         ruleset_path=run / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl", reward={})
    assert summarize(result) == summarize(run) == summarize(str(run))


def test_a_run_is_excluded_with_no_complete_episode_below_the_minimum_or_without_its_ruleset(tmp_path):
    assert exclusion(summarize(make_run(tmp_path / "a", [TORN]))) == "no complete episode"
    only_initial = summarize(make_run(tmp_path / "b", [0.9]))
    assert exclusion(only_initial) == "last complete episode is 0 (< min_last_episode 1)"
    assert exclusion(only_initial, min_last_episode=0) is None
    assert exclusion(summarize(make_run(tmp_path / "c", [0.1, 0.2], ruleset=False))) == (
        "episode_1 missing optimal_rule_set.jsonl")
    assert exclusion(summarize(make_run(tmp_path / "d", [0.1, 0.2]))) is None


def test_runs_rank_by_last_reward_then_lower_seed_then_the_order_given(tmp_path):
    def summary(name, effective, seed):
        (tmp_path / name / ".hydra").mkdir(parents=True)
        (tmp_path / name / ".hydra" / "config.yaml").write_text(yaml.safe_dump({"judge_model": {"seed": seed}}),
                                                               encoding="utf-8")
        return RunSummary(run=tmp_path / name, last_episode=1, metrics={"effective_reward": effective})

    runs = [summary("low", 0.1, 1), summary("unseeded", 0.5, None), summary("seed9", 0.5, 9),
            summary("seed2", 0.5, 2), summary("seed2-again", 0.5, 2), summary("high", 0.9, 30)]
    assert [s.run.name for s in rank(runs)] == ["high", "seed2", "seed2-again", "seed9", "unseeded", "low"]


def test_select_best_returns_the_winners_last_ruleset(tmp_path, caplog):
    runs = [
        make_run(tmp_path / "peaked", [0.5, 0.99, 0.6], seed=1),
        make_run(tmp_path / "steady", [0.5, 0.7], seed=2),
        make_run(tmp_path / "initial", [0.95], seed=3),
        make_run(tmp_path / "torn", [TORN], seed=4),
        make_run(tmp_path / "lost", [0.1, 0.98], seed=5, ruleset=False),
    ]
    with caplog.at_level(logging.WARNING, logger="ruleforge.optimize.select"):
        best = select_best(runs)
    assert best.path == tmp_path / "steady" / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl"
    assert [rule.text for rule in best.rules] == ["steady episode 1"]
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"skipping {runs[2]}: last complete episode is 0 (< min_last_episode 1)",
        f"{runs[3] / 'episode_0' / 'rule_amendment' / 'reward_optimal.jsonl'}: unreadable (killed mid-write?); "
        "the run's complete episodes end before it",
        f"skipping {runs[3]}: no complete episode",
        f"skipping {runs[4]}: episode_1 missing optimal_rule_set.jsonl",
    ]
    assert select_best(runs, min_last_episode=0).path == runs[2] / "episode_0" / "rule_amendment" / (
        "optimal_rule_set.jsonl")


def torn_records(run: Path, *, hydra: bool = True) -> Path:
    """``run``'s records as a run killed while it started leaves them."""
    if hydra:
        (run / ".hydra").mkdir(parents=True, exist_ok=True)
        (run / ".hydra" / "config.yaml").write_text("judge_model: {seed: [1", encoding="utf-8")
    else:
        shutil.rmtree(run / ".hydra", ignore_errors=True)
        (run / RUN_RECORD_DIR).mkdir(parents=True)
        (run / RUN_RECORD_DIR / RUN_RECORD_FNAME).write_text("", encoding="utf-8")
    return run


@pytest.mark.parametrize("hydra", [True, False], ids=["hydra-record", "run-record"])
def test_an_excluded_runs_records_are_never_read_and_a_selectable_ones_are(tmp_path, hydra):
    valid = make_run(tmp_path / "valid", [0.5, 0.7], seed=2)
    killed = torn_records(tmp_path / "killed", hydra=hydra)
    initial = torn_records(make_run(tmp_path / "initial", [0.9]), hydra=hydra)
    lost = torn_records(make_run(tmp_path / "lost", [0.1, 0.9], ruleset=False), hydra=hydra)
    assert [exclusion(summarize(run)) for run in (killed, initial, lost)] == [
        "no complete episode", "last complete episode is 0 (< min_last_episode 1)",
        "episode_1 missing optimal_rule_set.jsonl"]
    assert select_best([killed, initial, lost, valid]).path == (
        valid / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl")
    rows, notes = sweep_table([tmp_path])
    assert [row["seed"] for row in rows] == [2] and len(notes) == 3
    torn_records(valid, hydra=hydra)
    with pytest.raises((yaml.YAMLError, json.JSONDecodeError)):
        select_best([valid])
    with pytest.raises((yaml.YAMLError, json.JSONDecodeError)):
        sweep_table([tmp_path])


def test_select_best_refuses_when_no_run_can_be_selected(tmp_path):
    initial = make_run(tmp_path / "initial", [0.9])
    with pytest.raises(ValueError, match=re.escape(f"no run can be selected ({initial}: last complete episode is 0 ")):
        select_best([initial])
    with pytest.raises(ValueError, match="no run can be selected: none given"):
        select_best([])


def test_the_script_tables_the_same_reading_and_picks_select_bests_winner(tmp_path):
    first, second = tmp_path / "sweep" / "launch", tmp_path / "sweep" / "reruns"
    make_run(first / "0", [0.5, 0.7], seed=1)
    make_run(first / "1", [0.5, 0.8], seed=2)
    make_run(first / "2", [0.9], seed=3)  # episode 0 alone
    make_run(first / "3", [0.4, TORN], seed=4)  # its episode 1 torn: read as ending at episode 0
    (first / "notes").mkdir()
    make_run(second / "0", [0.5, 0.8], seed=1)  # ties job 1 of the launch, at a lower seed
    output = tmp_path / "selection.json"
    result = subprocess.run([sys.executable, *COMMAND, "--sweep-dir", str(first), "--sweep-dir", str(second),
                             "--output", str(output)], capture_output=True, text=True, check=True)
    table = json.loads(output.read_text(encoding="utf-8"))
    rows = table["all_seeds"]
    assert [(row["sweep_dir"], row["job_num"], row["seed"]) for row in rows] == [
        (str(second), 0, 1), (str(first), 1, 2), (str(first), 0, 1)]
    assert rows[0] == table["winner"] == {
        "sweep_dir": str(second), "job_num": 0, "seed": 1,
        "optimal_rule_set_path": str((second / "0" / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl")
                                     .resolve()),
        "last_episode": 1, "peak_episode": 1, "peak_effective_reward": 0.8,
        "effective_reward": 0.8, "entropy_reward": 0.8 + 0.25, "regularization_penalty": 0.25, "intent_penalty": 0.25}
    jobs = [first / name for name in ("0", "1", "2", "3")] + [second / "0"]
    assert str(select_best(jobs).path.resolve()) == table["winner"]["optimal_rule_set_path"]
    assert f"SKIP {first / '2'}: last complete episode is 0 (< min_last_episode 1)" in result.stdout
    assert f"WARN {first / '3' / 'episode_1' / 'rule_amendment' / 'reward_optimal.jsonl'}: unparseable" in (
        result.stdout)
    assert "Winner: seed=1 (job 0, episode 1)" in result.stdout


def test_the_script_skips_a_job_killed_while_it_started_without_reading_its_records(tmp_path):
    sweep = tmp_path / "sweep"
    make_run(sweep / "0", [0.5, 0.7], seed=1)
    torn_records(sweep / "1")
    torn_records(make_run(sweep / "2", [0.9]), hydra=False)
    result = subprocess.run([sys.executable, *COMMAND, "--sweep-dir", str(sweep), "--output",
                             str(tmp_path / "selection.json")], capture_output=True, text=True, check=True)
    assert f"SKIP {sweep / '1'}: no complete episode" in result.stdout
    assert f"SKIP {sweep / '2'}: last complete episode is 0 (< min_last_episode 1)" in result.stdout
    assert "Winner: seed=1 (job 0, episode 1)" in result.stdout


def test_the_script_refuses_a_sweep_with_nothing_selectable_or_no_directory(tmp_path):
    make_run(tmp_path / "sweep" / "0", [0.9])
    ended = subprocess.run([sys.executable, *COMMAND, "--sweep-dir", str(tmp_path / "sweep"), "--output",
                            str(tmp_path / "selection.json")], capture_output=True, text=True)
    assert ended.returncode == 2 and "No completed runs found" in ended.stderr
    assert not (tmp_path / "selection.json").exists() and not (tmp_path / "selection.json.runs").exists()
    missing = subprocess.run([sys.executable, *COMMAND, "--sweep-dir", str(tmp_path / "absent")],
                             capture_output=True, text=True)
    assert missing.returncode == 2 and "sweep dir not found" in missing.stderr
