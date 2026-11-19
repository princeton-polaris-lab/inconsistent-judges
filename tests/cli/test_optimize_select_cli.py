from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from ruleforge.cli import records
from ruleforge.optimize.select import sweep_table
from tests.test_optimize_select import TORN, make_run


def test_select_writes_the_table_and_prints_each_row_and_the_winner(cli, tmp_path):
    first, second = tmp_path / "launch", tmp_path / "reruns"
    make_run(first / "0", [0.5, 0.7], seed=1)
    make_run(first / "1", [0.5, 0.8], seed=2)
    make_run(first / "2", [0.9], seed=3)
    make_run(first / "3", [0.4, TORN], seed=4)
    make_run(second / "0", [0.5, 0.8], seed=1)
    rows, notes = sweep_table([first, second])
    status, out, _ = cli("optimize", "select", "--sweep-dir", first, "--sweep-dir", second)
    output = first / "selection.json"
    table = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0 and table == {"winner": rows[0], "all_seeds": rows}
    lines = out.splitlines()
    assert len(notes) == 3 and lines[:3] == notes and lines[3].split() == ["seed", "job", "last_ep", "effective", "entropy", "penalty",
                                                       "peak_ep", "peak_eff"]
    assert lines[5].split()[:4] == ["1", "0", "1", "0.8000"]
    assert lines[-3:] == ["Winner: seed=1 (job 0, episode 1)", f"Ruleset: {rows[0]['optimal_rule_set_path']}",
                          f"Selection table written to {output}"]
    assert next(records.runs_dir(output).iterdir()).name.endswith("-optimize-select")


def test_select_refuses_nothing_selectable_or_a_missing_sweep_before_writing(cli, tmp_path):
    make_run(tmp_path / "sweep" / "0", [0.9])
    status, out, error = cli("optimize", "select", "--sweep-dir", tmp_path / "sweep", "--output", "s.json",
                             "--min-last-episode", "0")
    assert status == 0 and (Path.cwd() / "s.json").is_file()
    status, out, error = cli("optimize", "select", "--sweep-dir", tmp_path / "sweep")
    assert status == 2 and "No completed runs found" in error and out.startswith("SKIP")
    assert "named by its job number" not in error
    assert not (tmp_path / "sweep" / "selection.json").exists()
    assert not records.runs_dir(tmp_path / "sweep" / "selection.json").exists()
    status, _, error = cli("optimize", "select", "--sweep-dir", tmp_path / "absent")
    assert status == 2 and "sweep dir not found" in error


@pytest.mark.skipif(os.geteuid() == 0, reason="root may read every directory")
def test_select_notes_a_directory_it_may_not_read_and_selects_from_the_rest(cli, tmp_path):
    make_run(tmp_path / "sweep" / "0", [0.5, 0.6], seed=1)
    member = make_run(tmp_path / "sweep" / "judge.seed.2", [0.5, 0.7], seed=2)
    locked = [tmp_path / "sweep" / "locked", tmp_path / "other" / "judge.seed.1", member / "episode_1"]
    for directory in locked:
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0)
    try:
        status, out, error = cli("optimize", "select", "--sweep-dir", tmp_path / "sweep")
        refused = cli("optimize", "select", "--sweep-dir", tmp_path / "other")
    finally:
        for directory in locked:
            directory.chmod(0o700)
    assert status == 0 and (tmp_path / "sweep" / "selection.json").is_file(), error
    assert out.splitlines()[:2] == [f"SKIP {directory}: cannot be read ({os.strerror(errno.EACCES)})"
                                    for directory in (member, locked[0])]
    assert "Winner: seed=1 (job 0, episode 1)" in out
    status, out, error = refused
    assert status == 2 and out == f"SKIP {locked[1]}: cannot be read ({os.strerror(errno.EACCES)})\n"
    assert "No completed runs found in the given sweep dir(s)." in error and "named by its job number" not in error


def test_optimize_without_a_verb_is_still_the_optimizer():
    from ruleforge.cli import app
    from ruleforge.cli import optimize as optimize_cli

    args = app.build_parser().parse_args(["optimize", "--rules", "r.jsonl"])
    assert args.verb is None and args.handler is optimize_cli.run_optimize and args.takes_overrides
    selected = app.build_parser().parse_args(["optimize", "select", "--sweep-dir", "d"])
    assert selected.handler is optimize_cli.run_select and not selected.takes_overrides


def test_select_refuses_optimize_s_own_options_before_reading_anything(cli, tmp_path):
    make_run(tmp_path / "sweep" / "0", [0.9])
    status, _, error = cli("optimize", "--slurm", "example", "--dry-run", "select", "--sweep-dir", tmp_path / "sweep")
    assert status == 2 and "takes none of optimize's own options (given: --dry-run, --slurm)" in error, error
    assert not (tmp_path / "sweep" / "selection.json").exists() and not (tmp_path / "sweep.runs").exists()


def test_select_reads_and_writes_a_home_spelling_as_the_home_directory_s(cli, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    make_run(tmp_path / "sweep" / "0", [0.9])
    status, _, error = cli("optimize", "select", "--sweep-dir=~/sweep", "--output=~/s.json", "--min-last-episode", "0")
    assert status == 0 and (tmp_path / "s.json").is_file() and records.runs_dir(tmp_path / "s.json").is_dir(), error


#: The scripted optimizer's smallest run: one episode, two scenarios, two judge types.
SMALL_RUN = ("--episodes", "1", "--", "eval_cfg.judge_types=[No_Interpretation,Narrow]",
             "scenario_generation.num_seed_scenarios=1", "scenario_generation.target_num_scenarios=2",
             "scenario_generation.num_scenarios_per_batch=2", "scenario_generation.max_accumulation_iterations=1")


def _sweep(cli, monkeypatch, *flags, overrides=()) -> None:
    from ruleforge.judges import JudgeSpec
    from tests.support.scripted import RuleForgeScript, scripted_backend
    from tests.test_optimize import RULES, SEEDS

    monkeypatch.setattr(JudgeSpec, "start", lambda spec: scripted_backend(spec.options.config(), RuleForgeScript()))
    status, _, error = cli("optimize", "--rules", RULES, "--seeds-good", SEEDS["good"], "--seeds-bad", SEEDS["bad"],
                           "--yes-spend", *flags, *SMALL_RUN, *overrides)
    assert status == 0, error


def test_select_reads_a_default_optimize_sweep(cli, monkeypatch):
    _sweep(cli, monkeypatch, "--judge", "openai:gpt-x", "--out", "runs/mine", "--sweep", "judge.seed=3,1,2",
           overrides=["rule_amendment.best_of_n=2"])
    sweep = Path.cwd() / "runs" / "mine"
    members = [f"judge.seed.{seed}" for seed in (1, 2, 3)]
    assert sorted(path.name for path in sweep.iterdir()) == sorted([*members, *(f"{name}.runs" for name in members)])
    status, out, error = cli("optimize", "select", "--sweep-dir", sweep)
    assert status == 0, error
    table = json.loads((sweep / "selection.json").read_text(encoding="utf-8"))
    rows = table["all_seeds"]
    assert [(row["seed"], row["job_num"], row["last_episode"]) for row in rows] == [(1, None, 1), (2, None, 1),
                                                                                    (3, None, 1)]
    assert table["winner"] == rows[0] and rows[0]["optimal_rule_set_path"] == str(
        (sweep / "judge.seed.1" / "episode_1" / "rule_amendment" / "optimal_rule_set.jsonl").resolve())
    assert not [line for line in out.splitlines() if line.startswith("SKIP")]
    assert out.splitlines()[2].split()[:2] == ["1", "-"] and "Winner: seed=1 (episode 1)" in out
    status, again, error = cli("optimize", "select", "--sweep-dir", sweep, "--output", Path.cwd() / "again.json")
    assert status == 0 and not [line for line in again.splitlines() if line.startswith("SKIP")], error


#: An agent-seed sweep's options: agents of their own, the judge's one seed apart from every agent seed.
AGENT_SWEEP = ("--judge", "openai:gpt-x", "--agent", "openai:gpt-y")
AGENT_OVERRIDES = ["++judge_model.seed=16", "rule_amendment.best_of_n=2"]


def _seeds_and_members(path: Path) -> list[tuple[int, str]]:
    """Each row's seed and the member directory its ruleset is in (``<member>/episode_<n>/rule_amendment/...``)."""
    rows = json.loads(path.read_text(encoding="utf-8"))["all_seeds"]
    return [(row["seed"], str(Path(row["optimal_rule_set_path"]).parents[2].relative_to(Path.cwd() / "runs")))
            for row in rows]


def test_select_reads_an_agent_seed_sweep_s_members_by_the_agent_s_seed(cli, monkeypatch):
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/a", "--sweep", "agent.seed=10,2,3", overrides=AGENT_OVERRIDES)
    sweep = Path.cwd() / "runs" / "a"
    status, out, error = cli("optimize", "select", "--sweep-dir", sweep)
    assert status == 0, error
    assert _seeds_and_members(sweep / "selection.json") == [(2, "a/agent.seed.2"), (3, "a/agent.seed.3"),
                                                            (10, "a/agent.seed.10")]
    assert "Winner: seed=2 (episode 1)" in out
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/t/agent.seed.{agent.seed}", "--sweep", "agent.seed=10,2,3",
           overrides=AGENT_OVERRIDES)
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "t")
    assert status == 0 and "Winner: seed=2 (episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "runs" / "t" / "selection.json") == [
        (2, "t/agent.seed.2"), (3, "t/agent.seed.3"), (10, "t/agent.seed.10")]
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/n", "--sweep", "rule_amendment.best_of_n=2",
           "--sweep", "agent.seed=10,2", overrides=["++judge_model.seed=16"])
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "n" /
                             "rule_amendment.best_of_n.2", "--output", Path.cwd() / "n.json")
    assert status == 0 and "Winner: seed=2 (episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "n.json") == [(2, "n/rule_amendment.best_of_n.2/agent.seed.2"),
                                                         (10, "n/rule_amendment.best_of_n.2/agent.seed.10")]
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/k", "--sweep", "agent.seed=10,2",
           "--sweep", "rule_amendment.best_of_n=2", overrides=["++judge_model.seed=16"])
    first = [Path.cwd() / "runs" / "k" / f"agent.seed.{seed}" for seed in (10, 2)]
    status, out, error = cli("optimize", "select", "--sweep-dir", first[0], "--sweep-dir", first[1], "--output",
                             Path.cwd() / "k.json")
    assert status == 0 and "Winner: seed=16 (episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "k.json") == [(16, "k/agent.seed.10/rule_amendment.best_of_n.2"),
                                                         (16, "k/agent.seed.2/rule_amendment.best_of_n.2")]


def test_select_reads_an_agent_seed_given_with_underscores_by_that_seed(cli, monkeypatch):
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/s", "--sweep", "agent.seed=1_000,20",
           overrides=AGENT_OVERRIDES)
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "s")
    assert status == 0 and "Winner: seed=20 (episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "runs" / "s" / "selection.json") == [(20, "s/agent.seed.20"),
                                                                                (1000, "s/agent.seed.1_000")]


def test_select_reads_a_member_named_by_a_number_by_its_recorded_seed(cli, monkeypatch):
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/u/{agent.seed}", "--sweep", "agent.seed=10,2,3",
           overrides=AGENT_OVERRIDES)
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "u")
    assert status == 0 and "Winner: seed=16 (job 10, episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "runs" / "u" / "selection.json") == [(16, "u/10"), (16, "u/2"),
                                                                                (16, "u/3")]
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/v/{agent.seed}", "--sweep", "agent.seed=10,2,3",
           overrides=["rule_amendment.best_of_n=2"])
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "v")
    assert status == 0 and "Winner: seed=2 (job 2, episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "runs" / "v" / "selection.json") == [(2, "v/2"), (3, "v/3"), (10, "v/10")]
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/{rule_amendment.best_of_n}/{judge.seed}",
           "--sweep", "rule_amendment.best_of_n=2", "--sweep", "judge.seed=3,1,2", overrides=["++agent_model.seed=2"])
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "2")
    assert status == 0 and "Winner: seed=1 (job 1, episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "runs" / "2" / "selection.json") == [(1, "2/1"), (2, "2/2"), (3, "2/3")]
    _sweep(cli, monkeypatch, *AGENT_SWEEP, "--out", "runs/w/agent.seed.{agent.seed}/{judge.seed}",
           "--sweep", "agent.seed=3", "--sweep", "judge.seed=2,1")
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "w" / "agent.seed.3")
    assert status == 0 and "Winner: seed=1 (job 1, episode 1)" in out, error
    assert _seeds_and_members(Path.cwd() / "runs" / "w" / "agent.seed.3" / "selection.json") == [
        (1, "w/agent.seed.3/1"), (2, "w/agent.seed.3/2")]


def test_select_says_where_a_two_key_sweep_s_seed_members_are(cli, monkeypatch):
    """A default sweep over two keys nests one directory per key in its --sweep order."""
    _sweep(cli, monkeypatch, "--judge", "openai:gpt-x", "--out", "runs/first", "--sweep", "judge.seed=1,2",
           "--sweep", "rule_amendment.best_of_n=2")
    sweep = Path.cwd() / "runs" / "first"
    status, out, error = cli("optimize", "select", "--sweep-dir", sweep)
    assert status == 2 and not (sweep / "selection.json").exists()
    assert out.splitlines() == [f"SKIP {sweep / f'judge.seed.{seed}'}: not a run: named as a seed axis's member, with "
                                "no run record or episode in it (a sweep over several keys holds its runs a directory "
                                "further down)" for seed in (1, 2)]
    assert "No completed runs found" in error and "nests one directory per key" in error
    status, out, error = cli("optimize", "select", "--sweep-dir", sweep / "judge.seed.1", "--sweep-dir",
                             sweep / "judge.seed.2", "--output", Path.cwd() / "first.json")
    rows = json.loads((Path.cwd() / "first.json").read_text(encoding="utf-8"))["all_seeds"]
    assert status == 0 and [row["seed"] for row in rows] == [1, 2], error
    _sweep(cli, monkeypatch, "--judge", "openai:gpt-x", "--out", "runs/last", "--sweep", "rule_amendment.best_of_n=2",
           "--sweep", "judge.seed=1,2")
    status, out, error = cli("optimize", "select", "--sweep-dir", Path.cwd() / "runs" / "last" /
                             "rule_amendment.best_of_n.2")
    assert status == 0 and "Winner: seed=1 (episode 1)" in out, error


def test_select_says_how_a_sweep_s_runs_are_named_when_it_finds_none(cli, tmp_path):
    make_run(tmp_path / "sweep" / "seed-1", [0.5, 0.9])
    (tmp_path / "sweep" / "seed-1" / ".hydra" / "config.yaml").unlink()
    status, out, error = cli("optimize", "select", "--sweep-dir", tmp_path / "sweep")
    assert status == 2 and out.startswith(f"SKIP {tmp_path / 'sweep' / 'seed-1'}: ")
    assert "No completed runs found" in error and "named by its job number" in error and "run record" in error
    assert "(judge.seed.<seed>, agent.seed.<seed>) and holding its episodes" in error
