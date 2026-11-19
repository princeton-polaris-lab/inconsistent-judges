from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from hydra.core.hydra_config import HydraConfig

import ruleforge.optimize  # noqa: F401 -- imported (subclassing ModelBackend) before a test stubs the program's backend
from ruleforge.launch.cli import main
from tests.conftest import requires_git_worktree
from tests.support.hydra_runs import run_program, through_hydra
from tests.support.scripted import RuleForgeScript, scripted_backend

REPO = Path(__file__).resolve().parents[1]
PROGRAM = "ruleforge.launch.programs.optimize"
pytestmark = requires_git_worktree
#: The paper's run: the ten-seed spec's run_name has no default.
PAPER = ("--var", "run_name=openai_10seed")


@pytest.fixture(autouse=True)
def _cluster(monkeypatch):
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu")
    monkeypatch.setenv("RULEFORGE_SLURM_ACCOUNT", "myaccount")


def _rendered(capsys, spec: str, *argv: str) -> dict:
    assert main(["job", "--spec", spec, "--dry-run", "--json", "--allow-dirty", *argv], repo=REPO) == 0
    return json.loads(capsys.readouterr().out)


def test_the_task_and_the_seeds_are_the_command_line_s_to_change(capsys):
    printed = _rendered(capsys, "ruleforge_modelspec", "--var", "task=modelspec/added", "--var", "run_name=added",
                        "--axis", "judge_model.seed=11..12")
    assert all(job["argv"][:3] == ["python", "-m", PROGRAM] and job["workdir"] == "." for job in printed["jobs"])
    assert [job["argv"][3:5] + job["argv"][-1:] for job in printed["jobs"]] == [
        ["task=modelspec/added", "run_name=added", "judge_model.seed=11"],
        ["task=modelspec/added", "run_name=added", "judge_model.seed=12"]]
    assert main(["job", "--spec", "ruleforge_modelspec", "--var", "task=modelspec/added", "--dry-run",
                 "--allow-dirty"], repo=REPO) == 2
    assert "needs --var for run_name" in capsys.readouterr().err


@pytest.mark.parametrize(("spec", "extra"), [("ruleforge_modelspec", ("--var", "run_name=r")), ("ruleforge_smoke", ())])
def test_the_sweep_s_prompt_is_a_launch_variable(capsys, spec, extra):
    default = _rendered(capsys, spec, *extra)["jobs"][0]["argv"]
    assert "eval_cfg.reasoning_prompt_style=naive" in default
    chosen = _rendered(capsys, spec, *extra, "--var", "prompt=naive-rules-above")["jobs"][0]["argv"]
    assert "eval_cfg.reasoning_prompt_style=naive-rules-above" in chosen
    assert len(chosen) == len(default)


@pytest.mark.parametrize(("spec", "extra"), [("ruleforge_modelspec", PAPER), ("ruleforge_smoke", ())])
def test_each_job_of_a_paper_launch_shares_the_judge_s_engine_by_the_rule_its_runs_were_launched_under(
        capsys, monkeypatch, tmp_path, spec, extra):
    """No paper launch names an openai_compatible judge: agents share its engine when model type and name match."""
    from ruleforge.launch.programs import check_jobs, config_dir
    from ruleforge.launch.programs import optimize as program

    printed = _rendered(capsys, spec, *extra)
    decided = []

    def check(cfg, *, workdir=None):
        agent, judge = cfg.agent_model, cfg.judge_model
        named = (agent.model_type, agent.model_name) == (judge.model_type, judge.model_name)
        decided.append((program._shares_engine(cfg), named))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.setenv("WANDB_GROUP_TIMESTAMP", printed["launch_id"])
    for job in printed["jobs"]:
        assert job["argv"][:3] == ["python", "-m", PROGRAM]
        with through_hydra():
            check_jobs(program.run, config_dir(), program.CONFIG_NAME, job["argv"][3:], check)
    assert len(decided) == len(printed["jobs"]) and all(shared == named for shared, named in decided)


#: Each seed's last-episode effective reward in the fake sweep: seed 7 wins.
REWARDS = {seed: 0.5 + (0.3 if seed == 7 else 0.01 * seed) for seed in range(1, 11)}


def _episodes(directory: Path, seed: int) -> None:
    """What the trainer writes of each episode that `ruleforge optimize select` reads (three episodes a seed)."""
    for episode in range(3):
        amendment = directory / f"episode_{episode}" / "rule_amendment"
        amendment.mkdir(parents=True)
        effective = REWARDS[seed] - 0.1 * (2 - episode)
        reward = {"effective_reward": effective, "entropy_reward": effective + 0.05, "regularization_penalty": 0.05,
                  "penalty_breakdown": {"intent": {"value": 0.05}}}
        (amendment / "reward_optimal.jsonl").write_text(json.dumps({"reward": reward}) + "\n", encoding="utf-8")
        (amendment / "optimal_rule_set.jsonl").write_text(json.dumps({"id": "Rule 1", "rule": f"r{seed}"}) + "\n",
                                                          encoding="utf-8")


def _run_optimize(overrides: list[str], monkeypatch, fail: str | None = None, script=None) -> list[dict]:
    """The optimizer's program with ``overrides``, as the job runs it, with backends scripted and the trainer stubbed."""
    runs: list[dict] = []

    def failing(where: str) -> None:
        if fail == where:
            raise RuntimeError(f"{where} failed")

    def backend(model_cfg):
        failing("engine")
        return scripted_backend(model_cfg, script or (lambda system_prompt, user_prompt: ""))

    class Trainer:
        def __init__(self, cfg, **injected):
            failing("trainer")
            self.cfg = cfg

        def train(self):
            failing("train")
            runs[-1]["output_dir"] = HydraConfig.get().runtime.output_dir
            _episodes(Path(runs[-1]["output_dir"]), self.cfg.judge_model.seed)

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: failing("wandb") or runs.append(kwargs) or Run()
    fake_wandb.log = lambda metrics, step: runs[-1].setdefault("logged", []).append(step)
    fake_wandb.Settings = dict
    fake_trainer = types.ModuleType("ruleforge.optimize.trainer")
    fake_trainer.RuleAmendmentTrainer = Trainer
    fake_trainer.check_run_inputs = lambda config, rules: None  # the stubbed trainer reads no ruleset
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    if script is None:
        monkeypatch.setitem(sys.modules, "ruleforge.optimize.trainer", fake_trainer)
    monkeypatch.setattr("ruleforge.judges.backend.ModelBackend", backend)  # the program imports it from there
    monkeypatch.setattr(sys, "argv", ["optimize.py", *overrides, "hydra/job_logging=none", "hydra/hydra_logging=none"])
    monkeypatch.setattr(HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)  # the run sets it: unset it after
    monkeypatch.setattr("ruleforge.launch.programs.check_top", Path.cwd)
    run_program(PROGRAM)
    return runs


def test_the_seeds_land_where_optimize_select_finds_them_and_the_best_is_picked(capsys, monkeypatch, tmp_path):
    printed = _rendered(capsys, "ruleforge_modelspec", *PAPER)
    launch_id = printed["launch_id"]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.setenv("WANDB_GROUP_TIMESTAMP", launch_id)
    sweep = tmp_path / "experiments" / "results" / "sweep" / "modelspec" / "openai" / launch_id
    for member, job in enumerate(printed["jobs"]):
        [run] = _run_optimize(job["argv"][3:], monkeypatch)
        assert (run["name"], run["group"], run["job_type"]) == (
            f"openai_10seed-{member}", f"{launch_id}-openai_10seed", "optimize")
        assert Path(run["output_dir"]) == sweep / str(member)
        saved = yaml.safe_load((sweep / str(member) / ".hydra" / "config.yaml").read_text(encoding="utf-8"))
        assert saved["judge_model"]["seed"] == member + 1 and saved["judge_model"]["model_name"] == "gemma-4-31B-it"
    capsys.readouterr()
    result = subprocess.run([sys.executable, "-m", "ruleforge", "optimize", "select", "--sweep-dir", str(sweep)],
                            capture_output=True, text=True, check=False, cwd=REPO)
    assert result.returncode == 0, result.stderr
    selection = json.loads((sweep / "selection.json").read_text(encoding="utf-8"))
    assert [(row["seed"], row["job_num"]) for row in sorted(selection["all_seeds"], key=lambda row: row["seed"])] == [
        (seed, seed - 1) for seed in range(1, 11)]
    winner = selection["winner"]
    assert (winner["seed"], winner["job_num"], winner["last_episode"]) == (7, 6, 2)
    assert winner["optimal_rule_set_path"] == str(sweep / "6" / "episode_2" / "rule_amendment" /
                                                  "optimal_rule_set.jsonl")


@pytest.mark.parametrize("fail", ["wandb", "engine", "trainer", "train"])
def test_a_seed_that_fails_before_or_while_training_fails_its_job(capsys, monkeypatch, tmp_path, fail):
    printed = _rendered(capsys, "ruleforge_smoke")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.setenv("WANDB_GROUP_TIMESTAMP", printed["launch_id"])
    with pytest.raises((SystemExit, RuntimeError)) as ended:
        _run_optimize(printed["jobs"][0]["argv"][3:], monkeypatch, fail=fail)
    if ended.type is SystemExit:
        assert ended.value.code == 1
    else:
        assert str(ended.value) == f"{fail} failed"


def test_the_smoke_runs_its_episode_through_the_program(capsys, monkeypatch, tmp_path):
    printed = _rendered(capsys, "ruleforge_smoke")
    (tmp_path / "data").symlink_to(REPO / "data")  # the configs name data/... from the job's directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.setenv("WANDB_GROUP_TIMESTAMP", printed["launch_id"])
    script = RuleForgeScript()
    [run] = _run_optimize(printed["jobs"][0]["argv"][3:], monkeypatch, script=script)
    assert script.unexpected == []
    amendment = (tmp_path / "experiments" / "results" / "sweep" / "modelspec" / "openai" / printed["launch_id"] / "0"
                 / "episode_1" / "rule_amendment")
    assert (amendment / "reward_optimal.jsonl").is_file() and (amendment / "optimal_rule_set.jsonl").is_file()
    assert sorted(set(run["logged"])) == [0, 1]
