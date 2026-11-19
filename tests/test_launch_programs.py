from __future__ import annotations

import contextlib
import json
import os
import re
import runpy
import sys
import types
from pathlib import Path

import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from ruleforge.launch import campaign, train
from ruleforge.launch import programs
from tests.support.hydra_runs import run_program, through_hydra

REPO = Path(__file__).resolve().parents[1]


def _run(module: str, arguments: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [module, *arguments])
    run_program(f"ruleforge.launch.programs.{module}")


def _run_refused(module: str, arguments: list[str], monkeypatch, refusal: type, match: str) -> None:
    with pytest.raises(refusal, match=match):
        _run(module, arguments, monkeypatch)


def test_the_programs_compose_the_checkout_s_configs_and_refuse_where_there_is_none(monkeypatch):
    assert programs.config_dir() == str(REPO / "configs") and programs.REPO_ROOT == REPO
    monkeypatch.setattr(programs, "CONFIG_DIR", REPO / "no-configs")
    with pytest.raises(programs.NoCheckout, match="no-configs is no directory: the launcher's job-side programs"):
        programs.config_dir()


@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign", "train_grpo", "optimize", "build_grpo_data"])
def test_a_program_runs_only_from_the_top_of_the_checkout_it_imported(tmp_path, monkeypatch, module):
    from ruleforge.core.provenance import CheckoutMismatch

    monkeypatch.chdir(REPO)
    assert programs.check_top() == REPO
    elsewhere = tmp_path / "other-checkout"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _run_refused(module, ["--cfg", "job"], monkeypatch, CheckoutMismatch,
                 f"runs from {elsewhere}, not from {REPO}, the top of the checkout it imported ruleforge from")


def test_the_rows_job_is_the_paper_s_rows_build_from_the_checkout_s_top(monkeypatch):
    from ruleforge.experiments.training.pins import PINNED_PROTECTED
    from ruleforge.grpo import grpo_data
    from ruleforge.launch.programs import build_grpo_data as job

    calls = []
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(grpo_data, "main", lambda argv=None, pins=None: calls.append((argv, pins)) or 3)
    assert job.main(["--out", "o"]) == 3
    monkeypatch.setattr(sys, "argv", ["build_grpo_data", "--out", "o"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ruleforge.launch.programs.build_grpo_data", run_name="__main__")
    assert exit_info.value.code == 3 and calls == [(["--out", "o"], PINNED_PROTECTED), (None, PINNED_PROTECTED)]


@pytest.mark.parametrize("words, module", [(campaign.PROGRAM, "run_campaign"),
                                           (campaign.PREFLIGHT_PROGRAM, "preflight_campaign"),
                                           (train.PROGRAM, "train_grpo")])
def test_the_launcher_runs_each_program_as_a_module_from_the_checkout_s_top(words, module):
    assert words == ("python", "-m", f"ruleforge.launch.programs.{module}")
    assert campaign.WORKDIR == train.WORKDIR == "."


@pytest.mark.parametrize("exported", [None, {"RANK": "1", "WORLD_SIZE": "2"}])
def test_the_grpo_program_runs_one_rank_of_its_composed_config(tmp_path, monkeypatch, exported, caplog):
    import ruleforge.grpo.trainer as trainer

    ran = []
    monkeypatch.setattr(trainer, "run_grpo", lambda config: ran.append((config, HydraConfig.get().job.name)))
    monkeypatch.setattr(trainer, "export_submitit_distributed_env", lambda environ: exported)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)  # the run sets it: unset it after
    monkeypatch.chdir(REPO)  # the checkout's top, where the launcher runs the job
    caplog.set_level("INFO")
    _run("train_grpo", ["run_name=r1", f"hydra.run.dir={tmp_path / 'hydra'}", "hydra/job_logging=none",
                        "hydra/hydra_logging=none"], monkeypatch)
    [(config, job)] = ran
    assert config["run_name"] == "r1" and config["output_dir"] == str(tmp_path / "artifacts" / "training" / "r1")
    assert config["data"]["rules"] == "data/rules/modelspec/openai_rules_all.jsonl" and job == "train_grpo"
    assert ("submitit task -> torch.distributed rank 1 of 2" in caplog.text) == (exported is not None)


def _hydra(mode: str, tmp_path: Path):
    from omegaconf import OmegaConf

    return OmegaConf.create({"job": {"name": "optimize", "num": 3}, "mode": {"name": mode},
                             "runtime": {"output_dir": str(tmp_path / "run")},
                             "overrides": {"task": []}})


def _optimize_program(tmp_path, monkeypatch, llms, stop):
    import types

    from ruleforge.launch.programs import optimize as program

    runs, trained = [], []

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: runs.append(kwargs) or Run()
    fake_wandb.Settings = dict
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: _hydra("MULTIRUN", tmp_path)))
    built = iter(llms)
    monkeypatch.setattr(program, "ModelBackend", lambda model: types.SimpleNamespace(llm=next(built)))
    monkeypatch.setattr(program, "check_run_inputs", lambda cfg, rules: None)
    monkeypatch.setattr(program, "build_task_spec", lambda cfg: types.SimpleNamespace(check_new_run=lambda: None))
    monkeypatch.setattr(program, "validated", lambda backend: backend)
    monkeypatch.setattr(program, "seed_paths", lambda section: [])
    monkeypatch.setattr(program, "WandbLogger", lambda: None)
    monkeypatch.setattr(program, "ensure_job_logging", lambda **kwargs: None)
    monkeypatch.setattr(program, "cleanup_vllm", stop)
    monkeypatch.setattr(program, "RuleAmendmentTrainer",
                        lambda cfg, **injected: types.SimpleNamespace(train=lambda: trained.append(injected)))
    return program, runs, trained


def _optimize_cfg(agent: str):
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "run_name": "r", "wandb": {"entity": "e", "project": "p", "group": "g", "dir": "w"},
        "rule_amendment": {"initial_rule_set_path": "rules.jsonl"}, "scenario_generation": {},
        "judge_model": {"model_type": "local", "model_name": "judge"},
        "agent_model": {"model_type": "local", "model_name": "judge" if agent == "judge" else "agent"}})


@pytest.mark.parametrize(("agent", "llms", "cleaned"), [
    ("judge", ("engine", None), ["engine"]),          # one backend for both: its engine stopped once
    ("other", ("judge-engine", "agent-engine"), ["judge-engine", "agent-engine"]),
    ("other", (None, "agent-engine"), ["agent-engine"]),
    ("other", (None, None), []),
])
def test_ruleforge_s_program_names_its_run_by_its_job_and_stops_the_engines_it_started(tmp_path, monkeypatch, agent,
                                                                                     llms, cleaned):
    stopped = []
    program, runs, trained = _optimize_program(tmp_path, monkeypatch, llms, stopped.append)
    program.run(_optimize_cfg(agent))
    [run] = runs
    assert (run["name"], run["job_type"], run["group"]) == ("r-3", "optimize", "g")
    [injected] = trained
    assert (injected["agent"] is injected["judge"]) == (agent == "judge")
    assert stopped == cleaned


@pytest.mark.parametrize("agent", ["judge", "other"])
def test_ruleforge_s_program_refuses_an_unconfirmed_judge_that_bills_each_call_before_it_claims_its_run(
        tmp_path, monkeypatch, agent):
    from omegaconf import OmegaConf

    from ruleforge.judges import Capabilities, JudgeSpec
    from ruleforge.judges.batch_api import BatchSpendNotConfirmed

    program, runs, trained = _optimize_program(tmp_path, monkeypatch, (None, None), lambda llm: None)
    billed = Capabilities(seed="local_deterministic", per_request_errors=True, spends_money=True)
    paid = "judge" if agent == "judge" else "agent"
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: billed if self.model_name == paid else
                        Capabilities(seed="local_deterministic", per_request_errors=True))
    with pytest.raises(BatchSpendNotConfirmed, match="pass --yes-spend"):
        program.run(_optimize_cfg(agent))
    assert not (tmp_path / "run").exists() and runs == trained == []
    program.run(OmegaConf.merge(_optimize_cfg(agent), {"confirm_spend": True}))
    assert len(trained) == 1 and (tmp_path / "run" / ".ruleforge-optimize" / "run.json").is_file()


def _compatible_cfg(judge_url: str, agent_url: str):
    from omegaconf import OmegaConf

    def model(url: str) -> dict:
        return {"model_type": "openai_compatible", "model_name": "same-model", "base_url": url, "api_key_env": None}

    return OmegaConf.merge(_optimize_cfg("judge"), {"judge_model": model(judge_url), "agent_model": model(agent_url)})


@pytest.mark.parametrize("confirmed", [False, True], ids=["unconfirmed", "confirmed"])
def test_ruleforge_s_program_refuses_agents_on_another_server_of_the_judge_s_model_as_optimize_does(tmp_path,
                                                                                                   monkeypatch,
                                                                                                   confirmed):
    from omegaconf import OmegaConf

    from ruleforge.judges import JudgeError, JudgeSpec
    from ruleforge.optimize.api import _shares

    program, runs, trained = _optimize_program(tmp_path, monkeypatch, ("judge", "agent"), lambda llm: None)
    cfg = OmegaConf.merge(_compatible_cfg("http://judge.test/v1", "http://agent.test/v1"), {"confirm_spend": confirmed})
    with pytest.raises(JudgeError, match=r"with other options \['base_url'\]") as python:
        _shares(JudgeSpec.from_config(cfg.agent_model), JudgeSpec.from_config(cfg.judge_model))
    for checked in (program.check_spend, program.run):
        with pytest.raises(JudgeError) as refused:
            checked(cfg)
        assert str(refused.value) == str(python.value)
    assert not (tmp_path / "run").exists() and runs == trained == []


def test_ruleforge_s_program_serves_agents_on_the_judge_s_own_server_from_its_one_engine(tmp_path, monkeypatch):
    program, _, trained = _optimize_program(tmp_path, monkeypatch, ("engine",), lambda llm: None)
    program.check_spend(_compatible_cfg("http://judge.test/v1", "http://judge.test/v1"))
    program.run(_compatible_cfg("http://judge.test/v1", "http://judge.test/v1"))  # a second engine would be refused
    [injected] = trained
    record = json.loads((tmp_path / "run" / ".ruleforge-optimize" / "run.json").read_text(encoding="utf-8"))
    assert injected["agent"] is injected["judge"] and record["agent"] == "judge"


def test_ruleforge_s_launch_refuses_an_unconfirmed_judge_that_bills_each_call_before_hydra_writes(tmp_path,
                                                                                                    monkeypatch):
    from ruleforge.launch.programs import optimize as program

    from ruleforge.judges.batch_api import BatchSpendNotConfirmed

    for name in ("ARTIFACT_ROOT", "MODEL_ROOT", "BATCH_ROOT"):
        monkeypatch.setenv(f"RULEFORGE_{name}", str(tmp_path / "roots" / name.lower()))
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")  # Hydra raises the check's own error, under a tracer or not
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: contextlib.nullcontext()
    fake_wandb.Settings = dict
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    engines, trained = [], []
    monkeypatch.setattr(program, "ModelBackend", lambda model: engines.append(model) or types.SimpleNamespace(llm=None))
    monkeypatch.setattr(program, "validated", lambda backend: backend)
    monkeypatch.setattr(program, "WandbLogger", lambda: None)
    monkeypatch.setattr(program, "RuleAmendmentTrainer",
                        lambda cfg, **injected: types.SimpleNamespace(train=lambda: trained.append(cfg.run_name)))
    monkeypatch.setattr(HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)  # the run sets it: unset it after
    monkeypatch.chdir(REPO)  # the checkout's top, where the launcher runs the job
    run_dir = tmp_path / "hydra"
    argv = ["optimize", "model@judge_model=gpt-5.6-luna", "model@agent_model=gpt-5.6-luna", "run_name=paid",
            f"hydra.run.dir={run_dir}"]

    def launch(*extra):
        monkeypatch.setattr(sys, "argv", [*argv, *extra])
        with through_hydra():
            program.main()

    before = sorted(tmp_path.rglob("*"))
    with pytest.raises(BatchSpendNotConfirmed, match="pass --yes-spend"):
        launch()
    assert sorted(tmp_path.rglob("*")) == before and engines == trained == []
    launch("+confirm_spend=true")
    assert trained == ["paid"] and len(engines) == 1
    assert (run_dir / ".ruleforge-optimize" / "run.json").is_file() and (run_dir / "trainer.log").is_file()


def test_ruleforge_s_program_refuses_a_style_the_task_keeps_only_to_read_its_records(tmp_path, monkeypatch):
    import types

    from omegaconf import OmegaConf

    from ruleforge.launch.programs import optimize as program
    from ruleforge.optimize import OptimizeConfig
    from ruleforge.tasks import ModelSpecTask, TaskError

    fixtures = REPO / "tests" / "fixtures" / "optimize"
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: pytest.fail("a W&B run opened")
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: _hydra("RUN", tmp_path)))
    monkeypatch.setattr(program, "ModelBackend", lambda model: pytest.fail("a judge loaded"))
    cfg = OmegaConf.merge(
        OptimizeConfig(reasoning_prompt_style="forced-grpo").to_config(
            task="modelspec", rules=fixtures / "rules.jsonl",
            seeds={"good": fixtures / "good_seeds.jsonl", "bad": fixtures / "bad_seeds.jsonl"}),
        {"run_name": "r", "wandb": {"entity": "e", "project": "p", "group": "g", "dir": "w"}})
    monkeypatch.setattr(ModelSpecTask, "record_only_styles", frozenset({"forced-grpo"}))
    with pytest.raises(TaskError, match="prompt style 'forced-grpo' of task 'modelspec' is recognised only to read"):
        program.run(cfg)


def test_an_engine_that_fails_to_stop_skips_no_other_and_fails_the_job(tmp_path, monkeypatch):
    for failing in ("judge-engine", "agent-engine"):
        stopped = []

        def stop(llm):
            stopped.append(llm)
            if llm == failing:
                raise SystemExit(1)

        program, _, _ = _optimize_program(tmp_path / failing, monkeypatch, ("judge-engine", "agent-engine"), stop)
        with pytest.raises(SystemExit) as exit_info:
            program.run(_optimize_cfg("other"))
        assert exit_info.value.code == 1 and stopped == ["judge-engine", "agent-engine"]


def test_ruleforge_s_program_stops_the_judge_s_engine_when_the_agent_s_backend_fails_to_build(tmp_path, monkeypatch):
    import types

    stopped, built = [], []
    program, _, _ = _optimize_program(tmp_path, monkeypatch, (), stopped.append)

    def backend(model):
        if built:
            raise RuntimeError("the agent's backend failed to build")
        built.append(model)
        return types.SimpleNamespace(llm="judge-engine")

    monkeypatch.setattr(program, "ModelBackend", backend)
    with pytest.raises(RuntimeError, match="the agent's backend failed to build"):
        program.run(_optimize_cfg("other"))
    assert stopped == ["judge-engine"]


def test_ruleforge_s_program_refuses_a_hydra_mode_it_does_not_know(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from ruleforge.launch.programs import optimize as program

    monkeypatch.setitem(sys.modules, "wandb", __import__("types").ModuleType("wandb"))
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: _hydra("SWEEP", tmp_path)))
    with pytest.raises(ValueError, match="Unknown hydra mode"):
        program.run(OmegaConf.create({"run_name": "r"}))


def test_a_campaign_job_stops_its_judge_s_engine_when_its_draws_fail(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    import ruleforge.judges.models as models
    from ruleforge.judges.backend import ModelBackend
    from ruleforge.launch.programs import run_campaign as program

    engine, stopped = object(), []
    backend = ModelBackend.__new__(ModelBackend)  # a backend holding a started engine, without loading one
    backend.llm = engine
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: _hydra("RUN", tmp_path)))
    for name in ("ensure_job_logging", "preflight", "execution_settings", "build_task_spec", "build_plan"):
        monkeypatch.setattr(program, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(program, "ModelBackend", lambda model: backend)

    def draws(*args, **kwargs):
        raise RuntimeError("the draws failed")

    monkeypatch.setattr(program, "run_plan", draws)
    monkeypatch.setattr(models, "cleanup_vllm", lambda llm, **kwargs: stopped.append((llm, kwargs)))
    with pytest.raises(RuntimeError, match="the draws failed"):
        program.run(OmegaConf.create({"judge_model": {}}))
    assert stopped == [(engine, {"raise_on_error": False})] and backend.llm is None


PINNED = "artifact://sampling/pub"
#: The pinned root's refusal (``ruleforge.evaluate.api._refuse_pinned``), wherever on its path a place is.
PINNED_REFUSAL = r"\(artifact://sampling/pub\S*\) is artifact://sampling/pub, or on its path, a published run"
#: A file handler added to Hydra's own logging (``hydra.hydra_logging``), whose ``filename`` an argument then gives.
HYDRA_LOG_FILE = "+hydra.hydra_logging.handlers.file.class=logging.FileHandler"
MADE_ON_THE_WAY = r"names two places: \S+ as written out, and {made}, which a write handed it as given makes on the way"


def _tree(top: Path) -> list[str]:
    return sorted(path.relative_to(top).as_posix() for path in top.rglob("*"))


class TaskReached(AssertionError):
    """A campaign program's task is reached: Hydra has made the job's places."""


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    """An artifact root whose ``sampling/pub`` a committed golden pins (empty)."""
    from hydra_plugins.hydra_submitit_launcher.submitit_launcher import BaseSubmititLauncher

    import ruleforge.core.provenance as provenance
    import ruleforge.evaluate.campaign as campaign_module
    from ruleforge.evaluate import api

    def reached(*args, **kwargs):
        raise TaskReached("the task was reached: Hydra had made the job's directories, saved configs and logs")

    def launch(*args, **kwargs):
        raise AssertionError("the submitit launcher was reached: a sweep was let through to SLURM")

    monkeypatch.setattr(BaseSubmititLauncher, "launch", launch)
    monkeypatch.setattr(provenance, "require_checkout", reached)
    monkeypatch.setattr(campaign_module, "preflight", reached)
    artifacts = tmp_path / "artifacts"
    (artifacts / "sampling" / "pub").mkdir(parents=True)
    monkeypatch.setattr(api, "pinned_roots", lambda: {PINNED: ("leniency",)})
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")  # Hydra re-raises what it would print
    monkeypatch.setattr(HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)  # a run sets it: unset it after
    monkeypatch.chdir(REPO)
    return artifacts


@pytest.mark.parametrize("spelled", ["default", "override"])
@pytest.mark.parametrize("multirun", [False, True], ids=["run", "sweep"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_campaign_program_run_directly_on_a_pinned_root_is_refused_before_hydra_makes_anything(
        tmp_path, pinned, monkeypatch, module, multirun, spelled):
    from ruleforge.evaluate.api import CampaignError

    unpinned = ("hydra.run.dir=${campaign_root}/hydra/preflight/${condition_id}/${now:%Y-%m-%d_%H-%M-%S}"
                if not multirun else "hydra.sweep.dir=${campaign_root}/hydra/x")
    override = {"default": [], "override": [unpinned]}[spelled]
    before = _tree(tmp_path)
    _run_refused(module, [*(["-m"] if multirun else []), "campaign_id=pub", *override], monkeypatch, CampaignError,
                 f"{pinned}/sampling/pub {PINNED_REFUSAL}")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(("multirun", "arguments"), [
    (False, ["hydra.run.dir={pub}/hydra/x"]),
    (False, ["hydra.output_subdir={pub}/x"]),
    (False, ["hydra.job_logging.handlers.file.filename={pub}/x.log"]),
    (False, [HYDRA_LOG_FILE, "+hydra.hydra_logging.handlers.file.filename={pub}/h.log"]),
    (True, ["hydra.sweep.dir={pub}/hydra"]),
    (True, ["hydra.sweep.subdir={pub}/x"]),
    (True, ["hydra.job_logging.handlers.file.filename={pub}/x.log"]),
    (True, [HYDRA_LOG_FILE, "+hydra.hydra_logging.handlers.file.filename={pub}/h.log"]),
    (True, ["hydra/launcher=submitit_slurm_gpu", "hydra.launcher.submitit_folder={pub}/s"]),
], ids=["run.dir", "output_subdir", "log-file", "hydra-log-file", "sweep.dir", "sweep.subdir", "sweep-log-file",
        "sweep-hydra-log-file", "submitit-folder"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_place_hydra_writes_on_a_pinned_root_is_refused_before_hydra_makes_anything(
        tmp_path, pinned, monkeypatch, module, multirun, arguments):
    from ruleforge.evaluate.api import CampaignError

    before = _tree(tmp_path)
    _run_refused(module, [*(["-m"] if multirun else []), "campaign_id=new",
                          *(argument.format(pub=pinned / "sampling" / "pub") for argument in arguments)],
                 monkeypatch, CampaignError, PINNED_REFUSAL)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("arguments", [
    ["-m", "campaign_id=new", "hydra.sweep.dir={pub}/s", "execution.num_shards=range(0,0)"],
    ["-m", "campaign_id=new", "hydra.sweep.dir={pub}/hydra", "judge_model=glob(nomatch*)"],
    ["-m", "campaign_id=new", "hydra.sweep.dir={elsewhere}", "execution.num_shards=range(0,0)", HYDRA_LOG_FILE,
     "+hydra.hydra_logging.handlers.file.filename={pub}/h.log"],
    ["--help", "campaign_id=new", HYDRA_LOG_FILE, "+hydra.hydra_logging.handlers.file.filename={pub}/h.log"],
    ["--hydra-help", HYDRA_LOG_FILE, "+hydra.hydra_logging.handlers.file.filename={pub}/h.log"],
], ids=["sweep-of-an-empty-range", "sweep-of-an-empty-glob", "empty-sweep-hydra-log-file", "help-hydra-log-file",
        "hydra-help-hydra-log-file"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_command_that_runs_no_job_is_held_where_hydra_still_writes(tmp_path, pinned, monkeypatch, module,
                                                                       arguments):
    from ruleforge.evaluate.api import CampaignError

    before = _tree(tmp_path)
    _run_refused(module, [argument.format(pub=pinned / "sampling" / "pub", elsewhere=pinned / "elsewhere")
                          for argument in arguments], monkeypatch, CampaignError, PINNED_REFUSAL)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_sweep_with_no_job_outside_a_pinned_root_runs(tmp_path, pinned, monkeypatch, module):
    before = _tree(tmp_path)
    _run(module, ["-m", "campaign_id=new", f"hydra.sweep.dir={pinned}/elsewhere", "execution.num_shards=range(0,0)"],
         monkeypatch)
    assert sorted(set(_tree(tmp_path)) - set(before)) == ["artifacts/elsewhere", "artifacts/elsewhere/multirun.yaml"]


@pytest.mark.parametrize("multirun", [False, True], ids=["run", "sweep"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_the_jobs_are_composed_under_the_version_base_hydra_main_runs_them_under(tmp_path, pinned, monkeypatch,
                                                                                   module, multirun):
    import ruleforge.core.provenance as provenance
    import ruleforge.evaluate.campaign as campaign_module
    from hydra import version

    from ruleforge.evaluate import api
    from ruleforge.evaluate.api import CampaignError
    from ruleforge.launch.programs import run_campaign

    def fresh():  # a fresh process's version base: unset (monkeypatch restores the suite's after the test)
        monkeypatch.setattr(version.VersionBase.instance(), "version_base", version._UNSPECIFIED_)

    def arguments(leaf):
        place = f"{pinned}/sampling/${{hydra:runtime.version_base}}/{leaf}"
        return (["-m", "campaign_id=new", f"hydra.sweep.dir={tmp_path}/sweeps", f"hydra.sweep.subdir={place}"]
                if multirun else ["campaign_id=new", f"hydra.run.dir={place}"])

    seen = []

    def reached(*args, **kwargs):
        seen.append(HydraConfig.get().runtime.version_base)
        raise TaskReached("the task was reached: Hydra had made the job's directories, saved configs and logs")

    monkeypatch.setattr(provenance, "require_checkout", reached)
    monkeypatch.setattr(campaign_module, "preflight", reached)
    fresh()
    _run_refused(module, arguments("ran"), monkeypatch, TaskReached, "the task was reached")  # on no pinned root
    [main_base] = seen  # the version base hydra.main composed the job under
    fresh()
    assert programs.hydra_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                               arguments("x")).jobs[0].hydra.runtime.version_base == main_base
    monkeypatch.setattr(api, "pinned_roots",
                        lambda: {f"artifact://sampling/{main_base}": ("leniency",)})
    before = _tree(tmp_path)
    fresh()
    _run_refused(module, arguments("x"), monkeypatch, CampaignError,
                 rf"\(artifact://sampling/{main_base}/x\) is artifact://sampling/{main_base}, or on its path")
    assert _tree(tmp_path) == before and seen == [main_base]


def test_every_job_of_a_sweep_is_checked_before_hydra_makes_any(tmp_path, pinned, monkeypatch):
    from ruleforge.evaluate.api import CampaignError

    sampling = pinned / "sampling"
    before = _tree(tmp_path)
    _run_refused("run_campaign", ["-m", "campaign_id=new", f"campaign_root={sampling}/new,{sampling}/pub"], monkeypatch,
                 CampaignError, f"{sampling}/pub {PINNED_REFUSAL}")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("multirun", [False, True], ids=["run", "sweep"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_relative_place_under_chdir_is_held_from_the_job_s_output_directory(tmp_path, pinned, monkeypatch, module,
                                                                               multirun):
    from ruleforge.evaluate.api import CampaignError

    output = ([f"hydra.sweep.dir={pinned}/elsewhere", "hydra.sweep.subdir=a"] if multirun
              else [f"hydra.run.dir={pinned}/elsewhere/a"])
    relatives = ("hydra.job_logging.handlers.file.filename=../../sampling/pub/x.log", "campaign_root=../../sampling/pub")

    def refused(match):
        before = _tree(tmp_path)
        for relative in relatives:
            _run_refused(module, [*(["-m"] if multirun else []), "campaign_id=new", *output, "hydra.job.chdir=true",
                                  relative], monkeypatch, CampaignError, match)
        assert _tree(tmp_path) == before

    refused(MADE_ON_THE_WAY.format(made=re.escape(str(pinned / "elsewhere" / "a"))))
    (pinned / "elsewhere" / "a").mkdir(parents=True)
    refused(rf"{pinned}/sampling/pub\S* {PINNED_REFUSAL}")


@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_sweep_s_jobs_are_composed_from_their_own_overrides(tmp_path, pinned, monkeypatch, module):
    from ruleforge.evaluate.api import CampaignError

    before = _tree(tmp_path)
    _run_refused(module, ["-m", "campaign_id=new", "eval_cfg.reasoning_prompt_style=naive-rules-above,pub",
                          f"campaign_root={pinned}/sampling/${{eval_cfg.reasoning_prompt_style}}",
                          f"hydra.sweep.dir={pinned}/sweeps", "hydra.sweep.subdir=${campaign_root}/x"],
                 monkeypatch, CampaignError, f"{pinned}/sampling/pub {PINNED_REFUSAL}")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_hydra_s_own_log_file_in_a_sweep_is_held_as_the_master_config_names_it(tmp_path, pinned, monkeypatch, module):
    """Hydra configures its own logging (``hydra.hydra_logging``) from a sweep's master config, never a job's."""
    from ruleforge.evaluate import api
    from ruleforge.evaluate.api import CampaignError

    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/naive-rules-above": ("leniency",)})
    (pinned / "sampling" / "naive-rules-above").mkdir()
    before = _tree(tmp_path)
    _run_refused(module, ["-m", "campaign_id=new", "eval_cfg.reasoning_prompt_style=forced,x", HYDRA_LOG_FILE,
                          f"+hydra.hydra_logging.handlers.file.filename={pinned}/sampling/"
                          "${eval_cfg.reasoning_prompt_style}/h.log"],
                 monkeypatch, CampaignError, rf"{pinned}/sampling/naive-rules-above/h.log \(artifact://sampling/naive-rules-above/h.log\) is")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("multirun", [False, True], ids=["run", "sweep"])
def test_a_file_hydra_saves_through_a_symlink_into_a_pinned_root_is_refused(tmp_path, pinned, monkeypatch, multirun):
    from ruleforge.evaluate.api import CampaignError

    directory = pinned / "elsewhere"
    saved = directory / "multirun.yaml" if multirun else directory / ".hydra" / "config.yaml"
    saved.parent.mkdir(parents=True)
    saved.symlink_to(pinned / "sampling" / "pub" / "x.yaml")
    before = _tree(tmp_path)
    place = ["-m", f"hydra.sweep.dir={directory}"] if multirun else [f"hydra.run.dir={directory}"]
    _run_refused("run_campaign", [*place, "campaign_id=new"], monkeypatch, CampaignError, PINNED_REFUSAL)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(("multirun", "arguments", "made"), [
    (False, ["campaign_id=new", "campaign_root={pub}/nx/../../new"], "nx"),
    (False, ["campaign_id=new", "hydra.run.dir={pub}/nx/../../elsewhere"], "nx"),
    (False, ["campaign_id=new", "hydra.run.dir={art}/out", "hydra.output_subdir=../sampling/pub/nn/../../cfg"], "nn"),
    (False, ["campaign_id=new", "hydra.job_logging.handlers.file.filename={pub}/nl/../../x.log"], "nl"),
    (True, ["campaign_id=new", "campaign_root={pub}/nx/../../new"], "nx"),
    (True, ["campaign_id=new", "hydra.sweep.dir={pub}/ny/../../sweeps"], "ny"),
    (True, ["campaign_id=new", "hydra.sweep.dir={art}/sweeps", "hydra.sweep.subdir=../sampling/pub/ns/../../x"], "ns"),
    (True, ["campaign_id=new", "hydra.output_subdir={pub}/nn/../../cfg"], "nn"),
    (True, ["campaign_id=new", "hydra/launcher=submitit_slurm_gpu",
            "hydra.launcher.submitit_folder={pub}/nf/../../submitit"], "nf"),
], ids=["campaign_root", "run.dir", "output_subdir", "log-file", "sweep-campaign_root", "sweep.dir", "sweep.subdir",
        "sweep-output_subdir", "submitit-folder"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_dotdot_after_a_directory_not_there_yet_is_refused_before_hydra_makes_it(
        tmp_path, pinned, monkeypatch, module, multirun, arguments, made):
    from ruleforge.evaluate.api import CampaignError

    for directory in ("out", "sweeps"):  # Hydra's output and sweep directories, made before the places below them
        (pinned / directory).mkdir()
    before = _tree(tmp_path)
    pub = pinned / "sampling" / "pub"
    _run_refused(module, [*(["-m"] if multirun else []),
                          *(argument.format(pub=pub, art=pinned) for argument in arguments)],
                 monkeypatch, CampaignError, MADE_ON_THE_WAY.format(made=re.escape(str(pub / made))))
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("multirun", [False, True], ids=["run", "sweep"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_dotdot_through_directories_that_are_there_names_one_place(tmp_path, pinned, monkeypatch, module, multirun):
    (pinned / "elsewhere").mkdir()
    place = ["-m", f"hydra.sweep.dir={pinned}/elsewhere/../sweeps"] if multirun else [
        f"hydra.run.dir={pinned}/elsewhere/../new"]
    _run_refused(module, [*place, "campaign_id=new", f"campaign_root={pinned}/elsewhere/../sampling/new"],
                 monkeypatch, TaskReached, "the task was reached")
    assert _tree(pinned / "sampling" / "pub") == []
    assert (pinned / ("sweeps" if multirun else "new")).is_dir()


@pytest.mark.parametrize("linked", [True, False], ids=["hard-link", "own-file"])
@pytest.mark.parametrize("multirun", [False, True], ids=["run", "sweep"])
def test_a_file_hydra_writes_that_has_another_name_is_refused(tmp_path, pinned, monkeypatch, multirun, linked):
    from ruleforge.evaluate.api import CampaignError

    pub_file = pinned / "sampling" / "pub" / "draws.jsonl"
    pub_file.write_text('{"recorded": true}\n', encoding="utf-8")
    directory = pinned / "elsewhere"
    saved = directory / "multirun.yaml" if multirun else directory / ".hydra" / "config.yaml"
    saved.parent.mkdir(parents=True)
    if linked:
        saved.hardlink_to(pub_file)
    else:
        saved.write_text("old: config\n", encoding="utf-8")
    place = ["-m", f"hydra.sweep.dir={directory}"] if multirun else [f"hydra.run.dir={directory}"]
    if linked:
        before = _tree(tmp_path)
        _run_refused("run_campaign", [*place, "campaign_id=new"], monkeypatch, CampaignError,
                     rf"{saved} is a file with 2 names \(hard links\)")
        assert _tree(tmp_path) == before
    else:
        _run_refused("run_campaign", [*place, "campaign_id=new"], monkeypatch, TaskReached, "the task was reached")
    assert pub_file.read_text(encoding="utf-8") == '{"recorded": true}\n'


@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_a_shard_of_a_pinned_root_and_a_root_in_another_campaign_s_are_refused_before_hydra_makes_anything(
        tmp_path, pinned, monkeypatch, module):
    from ruleforge.evaluate.api import CampaignError

    recorded = pinned / "sampling" / "recorded"
    recorded.mkdir()
    (recorded / "sampling_manifest.json").write_text("{}\n", encoding="utf-8")
    before = _tree(tmp_path)
    shard = ["campaign_id=pub", "execution.num_shards=2", "execution.shard_index=0",
             "job_root=${campaign_root}.shards/${condition_id}/shard-0-of-2"]
    _run_refused(module, shard, monkeypatch, CampaignError, f"{pinned}/sampling/pub {PINNED_REFUSAL}")
    _run_refused(module, ["campaign_id=x", f"campaign_root={recorded}/child"], monkeypatch, CampaignError,
                 f"the campaign root a configured job writes {recorded}/child is inside the campaign root {recorded}")
    assert _tree(tmp_path) == before


def test_a_rerun_of_a_pickled_config_or_another_sweeper_s_sweep_is_refused(tmp_path, pinned, monkeypatch):
    before = _tree(tmp_path)
    _run_refused("run_campaign", ["--experimental-rerun", str(tmp_path / "config.pickle")], monkeypatch,
                 programs.UncheckedJobs, "--experimental-rerun runs a pickled config")
    _run_refused("run_campaign", ["-m", "campaign_id=new", "hydra.sweeper._target_=x.Sweeper"], monkeypatch,
                 programs.UncheckedJobs, "hydra.sweeper x.Sweeper: this program lays out only")
    assert _tree(tmp_path) == before


@pytest.mark.parametrize(("arguments", "printed"), [(["--cfg", "job"], "campaign_id: pub"),
                                                    (["--help"], "preflight_campaign is powered by Hydra.")],
                         ids=["cfg", "help"])
def test_a_command_that_runs_no_job_is_not_held(tmp_path, pinned, monkeypatch, capsys, arguments, printed):
    before = _tree(tmp_path)
    exits = pytest.raises(SystemExit, match="^0$") if "--help" in arguments else contextlib.nullcontext()
    with exits:  # Hydra exits after its help
        _run("preflight_campaign", [*arguments, "campaign_id=pub"], monkeypatch)
    assert printed in capsys.readouterr().out and _tree(tmp_path) == before


def test_a_command_hydra_refuses_before_it_composes_is_not_held(tmp_path, pinned, monkeypatch):
    from ruleforge.launch.programs import run_campaign

    log = [HYDRA_LOG_FILE, f"+hydra.hydra_logging.handlers.file.filename={pinned}/sampling/pub/h.log"]
    resolve, commands = ["--hydra-help", "--resolve", *log], ["-m", "--cfg", "job", "campaign_id=new", *log]
    for arguments in (resolve, commands):
        assert programs.hydra_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                                   arguments) == programs.HydraJobs(None, [], False)
    before = _tree(tmp_path)
    _run_refused("run_campaign", resolve, monkeypatch, ValueError, "The --resolve flag can only be used in")
    _run_refused("run_campaign", commands, monkeypatch, ValueError, "Only one of --run, --multirun, --cfg, --info")
    assert _tree(tmp_path) == before


def test_the_jobs_checked_are_hydra_s_own(tmp_path, pinned, monkeypatch):
    from ruleforge.evaluate.campaign import check_job_roots
    from ruleforge.launch.programs import run_campaign

    seen = []

    def check(cfg, *, workdir):
        seen.append((str(cfg.condition_id), str(cfg.campaign_root), "hydra" in cfg, workdir,
                     check_job_roots(cfg, workdir=workdir)[1]))

    other = tmp_path / "other"
    extra = tmp_path / "extra"
    extra.mkdir()
    programs.check_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                        ["-m", "--config-dir", str(extra), "campaign_id=a", "eval_cfg.reasoning_prompt_style=naive-rules-above,forced",
                         f"+hydra.job.env_set.RULEFORGE_ARTIFACT_ROOT={other}"], check)
    assert seen == [(f"gemma-4-31B-it/{style}", f"{other}/sampling/a", False, None, other / "sampling" / "a")
                    for style in ("naive-rules-above", "forced")]
    launch, jobs, sweep = programs.hydra_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                                              ["campaign_id=a"])
    assert not sweep and launch == jobs[0] and launch is not jobs[0]
    HydraConfig.instance().set_config(jobs[0])
    places = programs.hydra_places(jobs[0], programs.output_directory(jobs[0], sweep=False))
    output = places[0][0]
    assert output.startswith(f"{pinned}/sampling/a/hydra/gemma-4-31B-it/")
    assert places == [(output, "Hydra's output directory"),
                      (f"{output}/.hydra", "the directory Hydra saves the job's configs in (hydra.output_subdir)"),
                      *((f"{output}/.hydra/{name}", f"the {name} Hydra saves there") for name in programs.SAVED_CONFIGS),
                      (f"{output}/run_campaign.log", "a log file of hydra.job_logging")]
    assert programs.hydra_launch_places(launch, sweep=False) == []  # Hydra's own logging writes no file
    master, jobs, sweep = programs.hydra_jobs(
        run_campaign.run, programs.config_dir(), "sampling_campaign",
        ["-m", "campaign_id=a", "hydra.output_subdir=null", "hydra/job_logging=none"])
    HydraConfig.instance().set_config(master)
    assert sweep and programs.hydra_launch_places(master, sweep=True) == [
        (f"{pinned}/sampling/a/hydra", "Hydra's sweep directory"),
        (f"{pinned}/sampling/a/hydra/multirun.yaml", "the multirun.yaml Hydra saves there")]
    HydraConfig.instance().set_config(jobs[0])
    [(output, _)] = programs.hydra_places(jobs[0], programs.output_directory(jobs[0], sweep=True))
    assert output.startswith(f"{pinned}/sampling/a/hydra/gemma-4-31B-it/")


def test_each_job_of_a_sweep_resolves_its_own_overrides(tmp_path, pinned):
    from ruleforge.launch.programs import run_campaign

    seen = []
    programs.check_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                        ["-m", f"hydra.sweep.dir={tmp_path}/sweeps", "campaign_id=a,b",
                         f"campaign_root={tmp_path}/r/${{campaign_id}}",
                         "eval_cfg.reasoning_prompt_style=naive-rules-above,forced", "condition_extra=job-${hydra:job.num}"],
                        lambda cfg, *, workdir: seen.append((str(cfg.campaign_root), str(cfg.condition_extra))))
    assert seen == [(f"{tmp_path}/r/{campaign}", f"job-{number}")
                    for number, campaign in enumerate(("a", "a", "b", "b"))]


@pytest.mark.parametrize("arguments", [["-m", "hydra.mode=RUN"], ["hydra.mode=MULTIRUN"]], ids=["-m", "mode"])
def test_a_sweep_is_laid_out_whichever_way_the_command_line_asks_for_it(tmp_path, pinned, arguments):
    from hydra.types import RunMode

    from ruleforge.launch.programs import run_campaign

    master, jobs, sweep = programs.hydra_jobs(
        run_campaign.run, programs.config_dir(), "sampling_campaign",
        [*arguments, "campaign_id=a", "eval_cfg.reasoning_prompt_style=naive-rules-above,forced"])
    assert sweep and master.hydra.mode == RunMode.MULTIRUN and [str(job.condition_id) for job in jobs] == [
        "gemma-4-31B-it/naive-rules-above", "gemma-4-31B-it/forced"]


def test_a_job_whose_output_directory_does_not_resolve_is_refused_after_its_own_check(tmp_path, pinned, monkeypatch):
    from omegaconf.errors import InterpolationResolutionError

    before = _tree(tmp_path)
    _run_refused("run_campaign", ["campaign_id=new", f"hydra.run.dir='${{rf.unpinned:{pinned}/sampling/pub}}/x'"],
                 monkeypatch, InterpolationResolutionError, PINNED_REFUSAL)
    assert _tree(tmp_path) == before


def test_a_job_under_chdir_reads_its_relative_paths_from_its_output_directory(tmp_path, pinned):
    from ruleforge.launch.programs import run_campaign

    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    seen = []
    for chdir in ("true", "false"):
        programs.check_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                            ["campaign_id=a", f"hydra.run.dir={tmp_path}/link/out", f"hydra.job.chdir={chdir}",
                             "hydra.job_logging.handlers.file.filename=x.log"],
                            lambda cfg, *, workdir: seen.append(workdir))
    assert seen == [str(real / "out"), None]
    _, [job], _ = programs.hydra_jobs(run_campaign.run, programs.config_dir(), "sampling_campaign",
                                      ["campaign_id=a", f"hydra.run.dir={tmp_path}/link/out", "hydra.job.chdir=true",
                                       "hydra.job_logging.handlers.file.filename=x.log"])
    HydraConfig.instance().set_config(job)
    assert programs.hydra_places(job, programs.output_directory(job, sweep=False))[-1] == (
        f"{real}/out/x.log", "a log file of hydra.job_logging")


@pytest.mark.parametrize("sharded", [False, True])
def test_the_launcher_s_preflight_directory_is_held_as_the_config_s_own(tmp_path, pinned, sharded):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError

    run_dir = campaign.SHARD_PREFLIGHT_RUN_DIR if sharded else campaign.PREFLIGHT_RUN_DIR
    where = {False: "{root}/hydra/preflight/gemma-4-31B-it/naive-rules-above/", True: "{root}.shards/gemma-4-31B-it/naive-rules-above/preflight/"}

    def resolved(campaign_id):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
            cfg = compose(config_name="sampling_campaign", overrides=[f"campaign_id={campaign_id}", run_dir],
                          return_hydra_config=True)
        GlobalHydra.instance().clear()
        return OmegaConf.select(cfg, "hydra.run.dir")

    assert resolved("new").startswith(where[sharded].format(root=pinned / "sampling" / "new"))
    if sharded:  # the shards beside a published root are on no pinned path; the program refuses its campaign root
        assert resolved("pub").startswith(where[sharded].format(root=pinned / "sampling" / "pub"))
    else:
        with pytest.raises(InterpolationResolutionError, match=PINNED_REFUSAL):
            resolved("pub")


def test_a_campaign_job_s_hydra_directories_are_under_its_root_when_no_golden_pins_it(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from ruleforge.evaluate.campaign import unpinned

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="sampling_campaign", overrides=["campaign_id=c"], return_hydra_config=True)
    GlobalHydra.instance().clear()
    hydra = OmegaConf.to_container(cfg.hydra, resolve=False)
    assert hydra["sweep"]["dir"] == "${rf.unpinned:${job_root}}/hydra"
    assert OmegaConf.select(cfg, "hydra.sweep.dir") == f"{tmp_path}/sampling/c/hydra"
    assert unpinned(tmp_path / "x") == str(tmp_path / "x")


@pytest.mark.parametrize("pending, finished", [(0, True), (1, False)], ids=["nothing-pending", "one-pending"])
def test_a_screening_draw_is_finished_when_nothing_is_pending(tmp_path, monkeypatch, pending, finished):
    from omegaconf import OmegaConf

    from ruleforge.evaluate.runner import STATE_INCOMPLETE_EXHAUSTED
    from ruleforge.launch.programs import run_campaign as program

    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: _hydra("RUN", tmp_path)))
    for name in ("require_checkout", "ensure_job_logging", "preflight", "execution_settings", "build_task_spec"):
        monkeypatch.setattr(program, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(program, "ModelBackend", lambda model: types.SimpleNamespace(close=lambda: None))
    plan = types.SimpleNamespace(config=types.SimpleNamespace(campaign_mode="screening"),
                                 condition=types.SimpleNamespace(condition_id="m/naive-rules-above"))
    monkeypatch.setattr(program, "build_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(program, "run_plan",
                        lambda *args, **kwargs: {0: {"state": STATE_INCOMPLETE_EXHAUSTED, "pending": pending}})
    if finished:
        program.run(OmegaConf.create({"judge_model": {}}))
    else:
        with pytest.raises(RuntimeError, match=r"1 draw\(s\) of m/naive-rules-above did not complete"):
            program.run(OmegaConf.create({"judge_model": {}}))


@pytest.mark.parametrize(("extra", "confirmed"), [({}, False), ({"confirm_spend": True}, True)], ids=["unset", "set"])
def test_ruleforge_s_program_confirms_its_spend_by_its_override(tmp_path, monkeypatch, extra, confirmed):
    import types

    from omegaconf import OmegaConf

    from ruleforge.judges import batch_api

    program, _, _ = _optimize_program(tmp_path, monkeypatch, ("engine", None), lambda _engine: None)
    seen = []
    monkeypatch.setattr(program, "RuleAmendmentTrainer", lambda cfg, **injected: types.SimpleNamespace(
        train=lambda: seen.append(batch_api.spend_is_confirmed())))
    program.run(OmegaConf.merge(_optimize_cfg("judge"), extra))
    assert seen == [confirmed]
    with pytest.raises(batch_api.BatchSpendNotConfirmed, match="confirm_spend is true or false, got 'false'"):
        program.run(OmegaConf.merge(_optimize_cfg("judge"), {"confirm_spend": "false"}))
    assert seen == [confirmed]


@pytest.fixture
def retired_root(tmp_path, modelspec, monkeypatch):
    """A configured campaign root holding a valid retired registration, beside an ordinary place; a launch program's
    task raises ``TaskReached`` at its entry."""
    import ruleforge.core.provenance as provenance
    import ruleforge.optimize.job_logging as job_logging
    from ruleforge.evaluate.campaign import build_plan
    from tests.evaluate.test_retired_roots import plant_retired
    from tests.test_sampling_campaign import backend_for, compose_campaign

    def reached(*args, **kwargs):
        raise TaskReached("the task was reached: Hydra had made the job's directories, saved configs and logs")

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    (tmp_path / "campaign").mkdir()
    plan = build_plan(compose_campaign(tmp_path / "campaign"), backend=backend_for("naive-rules-above"),
                      task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    plant_retired(plan.campaign_root, plan.condition_root / "condition.json")
    monkeypatch.setattr(provenance, "require_checkout", reached)
    monkeypatch.setattr(job_logging, "ensure_job_logging", reached)
    monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")  # Hydra re-raises what it would print
    monkeypatch.setattr(HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)  # a run sets it: unset it after
    monkeypatch.chdir(REPO)
    return plan.campaign_root


@pytest.mark.parametrize(("module", "arguments"), [
    ("optimize", ["hydra.run.dir={root}/hydra/optimize"]),
    ("optimize", ["hydra.run.dir={elsewhere}", "wandb.dir={root}/wandb"]),
    ("train_grpo", ["run_name=r1", "hydra.run.dir={root}/hydra/train"]),
    ("train_grpo", ["run_name=r1", "hydra.run.dir={elsewhere}", "output_dir={root}/run"]),
], ids=["optimize-run-dir", "optimize-wandb-dir", "train-run-dir", "train-output-dir"])
def test_a_launch_program_writes_nothing_in_a_campaign_root_and_runs_elsewhere(tmp_path, retired_root, monkeypatch,
                                                                              module, arguments):
    """The optimization and training programs hold Hydra's places and their own outputs to the campaign-root
    protection before Hydra makes anything; an ordinary place reaches the task."""
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    before = contents(retired_root)
    places = {"root": retired_root, "elsewhere": tmp_path / "elsewhere" / module}
    _run_refused(module, [argument.format(**places) for argument in arguments], monkeypatch, CampaignError,
                 f"is inside the campaign root {retired_root}")
    assert contents(retired_root) == before and not places["elsewhere"].exists()
    ordinary = [argument for argument in arguments if "{root}" not in argument]
    _run_refused(module, [*(argument.format(**places) for argument in ordinary),
                          *([] if any("hydra.run.dir" in argument for argument in ordinary)
                            else [f"hydra.run.dir={places['elsewhere']}"])], monkeypatch, TaskReached, "was reached")
    assert contents(retired_root) == before and places["elsewhere"].is_dir()


def test_a_job_s_unset_place_is_not_checked_and_a_relative_one_is_read_from_its_workdir(tmp_path, retired_root):
    from ruleforge.evaluate.api import CampaignError

    programs.refuse_outputs([(None, "an unset place"), ("run", "a relative place")], str(tmp_path))
    with pytest.raises(CampaignError, match=f"is inside the campaign root {retired_root}"):
        programs.refuse_outputs([(None, "an unset place"), ("run", "a relative place")], str(retired_root))


BATCH = "gpt-5.6-luna-batch"


@pytest.mark.parametrize(("arguments", "env"), [
    ([f"model@judge_model={BATCH}", f"model@agent_model={BATCH}"], "{root}/batch"),
    (["model@judge_model=gemma4-31b-tp-2", f"model@agent_model={BATCH}"], "{root}/batch"),
    (["model@judge_model=claude-haiku-5-5-batch", f"model@agent_model={BATCH}", "agent_model.batch_root={root}/batch"],
     "{tmp}/batch"),
    ([f"model@judge_model={BATCH}", f"model@agent_model={BATCH}", "hydra.job.chdir=true",
      "judge_model.batch_root={relative}"], "{tmp}/batch"),
], ids=["shared-engine", "separate-agent", "separate-both", "relative-to-the-job"])
def test_the_optimizer_holds_each_engine_s_batch_root_before_hydra_and_its_backends(tmp_path, retired_root, monkeypatch,
                                                                                     arguments, env):
    """Every distinct batch root of the engines the run builds is refused in a campaign root before Hydra makes anything
    and before a backend is built; an ordinary one reaches the task."""
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    def built(*args, **kwargs):
        raise AssertionError("a backend was built")

    monkeypatch.setattr("ruleforge.judges.backend.ModelBackend.__init__", built)
    elsewhere = tmp_path / "elsewhere" / "optimize"
    elsewhere.mkdir(parents=True)  # a run directory there already, so a relative place reads from it as it is
    places = {"root": retired_root, "tmp": tmp_path, "relative": os.path.relpath(retired_root / "batch", elsewhere)}
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", env.format(**places))
    before = contents(retired_root)
    common = ["+confirm_spend=true", f"hydra.run.dir={elsewhere}", f"wandb.dir={tmp_path / 'wandb'}"]
    _run_refused("optimize", [*(argument.format(**places) for argument in arguments), *common], monkeypatch,
                 CampaignError, f"batch_root .* is inside the campaign root {retired_root}")
    assert contents(retired_root) == before and not any(elsewhere.iterdir())
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    ordinary = [argument.format(**places).replace(str(retired_root), str(tmp_path / "ordinary"))
                for argument in arguments if "{relative}" not in argument]
    _run_refused("optimize", [*ordinary, *common], monkeypatch, TaskReached, "was reached")
    assert contents(retired_root) == before


@pytest.mark.parametrize("module", ["optimize", "train_grpo"])
@pytest.mark.parametrize("planted", ["trainer.log", "episode_0"], ids=["fallback-log", "link-below"])
def test_a_link_in_a_job_s_own_output_directory_is_refused_before_the_task(tmp_path, retired_root, monkeypatch, module,
                                                                           planted):
    """With Hydra's logging off, the log the task opens in its output directory, or anything else there, linked into a
    campaign root is refused before the task; the root is unchanged."""
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    run_dir = tmp_path / "elsewhere" / module
    run_dir.mkdir(parents=True)
    (run_dir / planted).symlink_to(retired_root / f"escaped-{planted}")
    before = contents(retired_root)
    arguments = [f"hydra.run.dir={run_dir}", "hydra/job_logging=none", "hydra/hydra_logging=none",
                 *(["run_name=r1"] if module == "train_grpo" else [])]
    _run_refused(module, arguments, monkeypatch, CampaignError, f"{run_dir / planted}, below .* is a symlink")
    assert contents(retired_root) == before and not (retired_root / f"escaped-{planted}").exists()
    (run_dir / planted).unlink()
    _run_refused(module, arguments, monkeypatch, TaskReached, "was reached")


@pytest.mark.parametrize("module", ["optimize", "train_grpo"])
def test_a_fallback_log_with_another_name_in_a_campaign_root_is_refused_before_the_task(tmp_path, retired_root,
                                                                                        monkeypatch, module):
    """A hard link shows on no path: the log the task would append to is refused while it has a second name."""
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    run_dir = tmp_path / "elsewhere" / module
    run_dir.mkdir(parents=True)
    kept = retired_root / "kept.log"
    kept.write_text("recorded\n")
    os.link(kept, run_dir / "trainer.log")
    before = contents(retired_root)
    arguments = [f"hydra.run.dir={run_dir}", "hydra/job_logging=none", "hydra/hydra_logging=none",
                 *(["run_name=r1"] if module == "train_grpo" else [])]
    _run_refused(module, arguments, monkeypatch, CampaignError, r"trainer\.log is a file with 2 names \(hard links\)")
    assert contents(retired_root) == before
    (run_dir / "trainer.log").unlink()
    _run_refused(module, arguments, monkeypatch, TaskReached, "was reached")


def test_the_grpo_output_directory_is_a_tree_the_run_owns_with_no_link_below(tmp_path, retired_root, monkeypatch):
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    output = tmp_path / "run"
    (output / "checkpoint-1").mkdir(parents=True)
    (output / "checkpoint-1" / "adapter").symlink_to(retired_root, target_is_directory=True)
    before = contents(retired_root)
    _run_refused("train_grpo", ["run_name=r1", f"output_dir={output}", f"hydra.run.dir={tmp_path / 'hydra'}"],
                 monkeypatch, CampaignError, "checkpoint-1/adapter, below the run's output directory .* is a symlink")
    assert contents(retired_root) == before and not (tmp_path / "hydra").exists()


def test_the_optimizer_s_transports_derive_their_work_directories_under_the_guard_its_command_holds(tmp_path,
                                                                                                  monkeypatch):
    """As in ``ruleforge optimize``, a batch work directory that is a link is refused before a transport writes it."""
    from omegaconf import OmegaConf

    from ruleforge.judges import batch_api

    program, _, _ = _optimize_program(tmp_path, monkeypatch, ("engine", None), lambda _engine: None)
    batches = tmp_path / "batch"
    batches.mkdir()
    (batches / "linked").symlink_to(tmp_path, target_is_directory=True)
    derived = []

    def train():
        derived.append(batch_api.work_directory(batches, "plain"))
        derived.append(batch_api.work_directory(batches, "linked"))

    monkeypatch.setattr(program, "RuleAmendmentTrainer", lambda cfg, **injected: types.SimpleNamespace(train=train))
    with pytest.raises(batch_api.BatchDestinationRefused, match=f"{batches / 'linked'} is a symlink"):
        program.run(OmegaConf.merge(_optimize_cfg("judge"), {"confirm_spend": True}))
    assert derived == [batches / "plain"] and not batch_api.work_directories_guarded()


@pytest.mark.parametrize("link", ["symlink", "hard-link"])
@pytest.mark.parametrize(("module", "logging", "planted"), [
    ("run_campaign", "off", "trainer.log"),  # the log its task opens with Hydra's logging off (ensure_job_logging)
    ("run_campaign", "default", "run_campaign.log"),  # Hydra's job log
    ("preflight_campaign", "default", "preflight_campaign.log"),  # its task opens no log of its own
], ids=["run-fallback-log", "run-job-log", "preflight-job-log"])
def test_a_campaign_program_refuses_a_linked_log_file_before_hydra_and_its_task(tmp_path, retired_root, monkeypatch,
                                                                                module, logging, planted, link):
    """A log file a campaign job writes that is a symlink, or a file with a second name, into a campaign root is refused
    before Hydra writes and before the task; the root is unchanged, and the same command without the link runs."""
    import ruleforge.evaluate.campaign as campaign_module
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    def reached(*args, **kwargs):
        raise TaskReached("the task was reached")

    monkeypatch.setattr(campaign_module, "preflight", reached)  # the preflight program's task starts with it
    run_dir = tmp_path / "elsewhere" / module
    run_dir.mkdir(parents=True)
    escaped = retired_root / f"escaped-{planted}"
    if link == "symlink":
        (run_dir / planted).symlink_to(escaped)
        refusal = re.escape(f"{run_dir / planted} is a symlink: a write there would follow it to {escaped}")
    else:
        kept = retired_root / f"kept-{planted}"
        kept.write_text("recorded\n")
        os.link(kept, run_dir / planted)
        refusal = re.escape(f"{run_dir / planted} is a file with 2 names (hard links)")
    before = contents(retired_root)
    arguments = ["campaign_id=new", f"hydra.run.dir={run_dir}",
                 *(["hydra/job_logging=none", "hydra/hydra_logging=none"] if logging == "off" else [])]
    _run_refused(module, arguments, monkeypatch, CampaignError, refusal)
    assert contents(retired_root) == before and not escaped.exists()
    assert [path.name for path in run_dir.iterdir()] == [planted]  # Hydra made nothing
    (run_dir / planted).unlink()
    _run_refused(module, arguments, monkeypatch, TaskReached, "was reached")
    assert contents(retired_root) == before and not escaped.exists() and (run_dir / ".hydra").is_dir()


@pytest.mark.parametrize(("arguments", "planted"), [
    (["hydra.run.dir={run}"], "{run}/.hydra/config.yaml"),
    (["-m", "hydra.sweep.dir={run}", "execution.num_shards=range(0,0)"], "{run}/multirun.yaml"),
    (["hydra.run.dir={run}/job", HYDRA_LOG_FILE, "+hydra.hydra_logging.handlers.file.filename={run}/h.log"],
     "{run}/h.log"),
], ids=["saved-config", "master-config", "hydra-log"])
@pytest.mark.parametrize("module", ["run_campaign", "preflight_campaign"])
def test_each_file_hydra_writes_for_a_campaign_program_is_refused_while_it_is_a_symlink(tmp_path, retired_root,
                                                                                       monkeypatch, module, arguments,
                                                                                       planted):
    from ruleforge.evaluate.api import CampaignError
    from tests.evaluate.test_retired_roots import contents

    run = tmp_path / "elsewhere"
    place = Path(planted.format(run=run))
    place.parent.mkdir(parents=True)
    escaped = retired_root / f"escaped-{place.name}"
    place.symlink_to(escaped)
    before = contents(retired_root)
    _run_refused(module, [*(argument.format(run=run) for argument in arguments), "campaign_id=new"], monkeypatch,
                 CampaignError, re.escape(f"{place} is a symlink: a write there would follow it to {escaped}"))
    assert contents(retired_root) == before and not escaped.exists()


def test_only_a_file_place_is_refused_for_being_a_symlink(tmp_path):
    """A directory place, a dangling link among them, is out of this check's scope; a file place that is a link is not."""
    from ruleforge.evaluate.api import CampaignError

    dangling, linked = tmp_path / "dangling", tmp_path / "linked"
    dangling.symlink_to(tmp_path / "absent")
    linked.symlink_to(tmp_path, target_is_directory=True)
    for place in (dangling, linked, tmp_path / "absent"):
        programs.refuse_place(str(place), "a directory Hydra makes")
    for place in (dangling, linked):
        with pytest.raises(CampaignError, match=re.escape(f"a log file {place} is a symlink")):
            programs.refuse_place(str(place), "a log file", file=True)
    programs.refuse_place(str(tmp_path / "absent"), "a log file", file=True)
