from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ruleforge.core.constants import (
    EVALUATIONS_FNAME,
    INTENT_TRACE_FNAME,
    JUDGE_OUTPUTS_FNAME,
    PATCH_FNAME,
    POSTERIOR_SAMPLING_TRACE_FNAME,
    PRIOR_FNAME,
    PRIOR_TRACE_FNAME,
)
from ruleforge.optimize import RUN_RECORD_DIR, RUN_RECORD_FNAME, OptimizeConfig, RAConfig, SGConfig, optimize
from ruleforge.tasks import ModelSpecTask
from tests.support.hydra_runs import kept_logging, run_program
from tests.support.scripted import RuleForgeScript, ScriptedJudge, scripted_backend

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "optimize"
JUDGE_TYPES = ("No_Interpretation", "Narrow", "Broad", "Ordinary_Meaning")
#: The judge's name in the frozen evaluations: the default judge config's (configs/model/gemma4-31b-tp-2.yaml).
MODEL_NAME = "gemma-4-31B-it"
#: The token ceiling of every request the Hydra job sends: that config's ``max_new_tokens``, which no caller overrides.
SCRIPT_MAX_NEW_TOKENS = 65536
SEEDS = {"good": FIXTURES / "good_seeds.jsonl", "bad": FIXTURES / "bad_seeds.jsonl"}
#: The scripted run's settings, beside the paper's defaults (configs/task/modelspec/base.yaml).
SETTINGS = {
    "max_episodes": 2,
    "scenario_generation.num_seed_scenarios": 2,
    "scenario_generation.target_num_scenarios": 3,
    "scenario_generation.num_scenarios_per_batch": 2,
    "scenario_generation.max_accumulation_iterations": 2,
    "rule_amendment.best_of_n": 2,
    # One attempt, so the scripted failures stay in the frozen trace.
    "eval_cfg.panel_num_tries": 1,
}
RUNS = {"intent": {}}
NOT_TRACED = (".hydra/", "trainer.log", f"{RUN_RECORD_DIR}/")
CALL_TRACES = frozenset({JUDGE_OUTPUTS_FNAME, INTENT_TRACE_FNAME, PRIOR_TRACE_FNAME, POSTERIOR_SAMPLING_TRACE_FNAME})


def trace_path(name: str) -> Path:
    return FIXTURES / f"trace_{name}.json"


def request_stream(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """How many requests a run sent, and a digest of them; a run's output path in a prompt would change the digest."""

    def digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    rows = sorted(json.dumps([digest(request["system_prompt"]), digest(request["user_prompt"]),
                              request["temperature"], request["seed"]]) for request in requests)
    return {"count": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()}


def _rows(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _kept(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [candidate for attempt in trace["attempts"] for candidate in attempt["candidates"] if candidate["accepted"]]


def check_call_traces(out: Path) -> None:
    """Every model call's trace under ``out`` keeps the answer the run used, beside the output it produced."""
    found = {name: list(out.rglob(name)) for name in CALL_TRACES}
    assert found[JUDGE_OUTPUTS_FNAME] and found[PRIOR_TRACE_FNAME] and found[POSTERIOR_SAMPLING_TRACE_FNAME]
    for evaluations in out.rglob(EVALUATIONS_FNAME):
        outputs, rows = _rows(evaluations.with_name(JUDGE_OUTPUTS_FNAME)), _rows(evaluations)
        assert [row["scenario_id"] for row in outputs] == [row["scenario_id"] for row in rows]
        for output, row in zip(outputs, rows):
            for model, variants in output["judge_outputs"].items():
                for variant, by_type in variants.items():
                    for judge_type, trace in by_type.items():
                        assert trace["judgment"] == row["evaluations"][model][variant][judge_type]
                        assert len(_kept(trace)) == (trace["judgment"] is not None)
    for intent in found[INTENT_TRACE_FNAME]:
        trace = json.loads(intent.read_text(encoding="utf-8"))
        assert [candidate["completion"] for candidate in _kept(trace)] and trace["result"] is not None
    for prior_trace in found[PRIOR_TRACE_FNAME]:
        [trace] = _rows(prior_trace)
        assert trace["call"] == "posterior_update"
        [prior] = _rows(prior_trace.with_name(PRIOR_FNAME))
        assert prior["prior"] == (trace["result"][0] if trace["result"] else prior["prior"])
    for sampling in found[POSTERIOR_SAMPLING_TRACE_FNAME]:
        [trace] = _rows(sampling)
        assert trace["call"] == "posterior_sampling"
        if sampling.parent.name == "rule_amendment":
            for candidate in _kept(trace):
                patch = sampling.parent / f"sample_{candidate['result_index']}" / PATCH_FNAME
                assert patch.read_text(encoding="utf-8") == candidate["completion"]


def collect(out: Path, logged: list[dict[str, Any]], requests: list[dict[str, Any]]) -> dict[str, Any]:
    """The trace of a run that wrote ``out``, logged ``logged`` and sent ``requests``."""
    check_call_traces(out)
    files = {}
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out).as_posix()
        if path.is_file() and not relative.startswith(NOT_TRACED) and path.name not in CALL_TRACES:
            files[relative] = path.read_text(encoding="utf-8")
    return {"files": files, "logged": logged, "requests": request_stream(requests)}


@pytest.fixture
def isolated_run(monkeypatch):
    """What a run changes in this process, put back after it."""
    clock = types.SimpleNamespace(time=lambda: 0.0)
    monkeypatch.setattr("ruleforge.optimize.judging.time", clock)
    with kept_logging():
        yield


def _overrides(name: str, rules: Path = FIXTURES / "rules.jsonl") -> list[str]:
    settings = {**SETTINGS, **RUNS[name]}
    return [
        "run_name=trace",
        f"eval_cfg.judge_types=[{','.join(JUDGE_TYPES)}]",
        f"rule_amendment.initial_rule_set_path={rules}",
        f"scenario_generation.good_seed_scenario_path={SEEDS['good']}",
        f"scenario_generation.bad_seed_scenario_path={SEEDS['bad']}",
        *(f"{key}={value}" for key, value in settings.items()),
    ]


def _recorded(logged: list[dict[str, Any]], metrics: Any, step: int) -> None:
    logged.append(json.loads(json.dumps({"step": step, "metrics": metrics})))


def run_script_path(name: str, out: Path, monkeypatch, script: RuleForgeScript,
                    rules: Path = FIXTURES / "rules.jsonl", *, extra: tuple[str, ...] = (),
                    started: list[str] | None = None) -> dict[str, Any]:
    """The Hydra job for run ``name`` into ``out``, its backends scripted by ``script``; returns its trace."""
    logged: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    started = [] if started is None else started

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = lambda **kwargs: started.append("wandb") or Run()
    fake_wandb.log = lambda metrics, step: _recorded(logged, metrics, step)
    fake_wandb.Settings = dict
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    # The program imports ModelBackend from here when it runs.
    monkeypatch.setattr("ruleforge.judges.backend.ModelBackend",
                        lambda model_cfg: started.append("backend") or scripted_backend(model_cfg, script,
                                                                                        requests=requests))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.chdir(REPO)  # the checkout's top, where the launcher runs the job
    monkeypatch.setattr(sys, "argv", ["optimize.py", *_overrides(name, rules), *extra, f"hydra.run.dir={out}",
                                      "hydra/job_logging=none", "hydra/hydra_logging=none"])
    monkeypatch.setattr(HydraConfig.instance(), "cfg", HydraConfig.instance().cfg)
    run_program("ruleforge.launch.programs.optimize")
    assert script.unexpected == []
    assert {request["max_new_tokens"] for request in requests} == {SCRIPT_MAX_NEW_TOKENS}
    return collect(out, logged, requests)


def optimize_config(name: str) -> OptimizeConfig:
    """Run ``name``'s settings as an :class:`OptimizeConfig`."""
    settings = {**SETTINGS, **RUNS[name]}
    sections = {section: {key.split(".", 1)[1]: value for key, value in settings.items()
                          if key.startswith(f"{section}.")} for section in ("scenario_generation", "rule_amendment")}
    return OptimizeConfig(max_episodes=settings["max_episodes"],
                          scenario_generation=SGConfig(**sections["scenario_generation"]),
                          rule_amendment=RAConfig(**sections["rule_amendment"]), judge_types=JUDGE_TYPES,
                          reasoning_prompt_style="naive-rules-above",
                          panel_num_tries=settings["eval_cfg.panel_num_tries"])


class Recorder:
    """A RunLogger keeping what it is given."""

    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []

    def log(self, metrics: Any, *, step: int) -> None:
        _recorded(self.logged, metrics, step)


def run_optimize(name: str, out: Path, failure: str) -> dict[str, Any]:
    """:func:`optimize` with run ``name``'s settings into ``out``, a scripted judge and a scripted agent."""
    judge_script, agent_script = RuleForgeScript(failure=failure), RuleForgeScript(failure=failure)
    recorder = Recorder()
    handlers = list(logging.getLogger().handlers)
    judge, agent = ScriptedJudge(judge_script, model_name=MODEL_NAME), ScriptedJudge(agent_script)
    optimize(FIXTURES / "rules.jsonl", task=ModelSpecTask(), judge=judge, agent=agent, seeds=SEEDS, out=out,
             config=optimize_config(name), logger=recorder)
    assert logging.getLogger().handlers == handlers
    assert judge_script.unexpected == agent_script.unexpected == []
    assert {request["max_new_tokens"] for request in judge.requests + agent.requests} == {None}
    return collect(out, recorder.logged, judge.requests + agent.requests)


def frozen(name: str, trace: dict[str, Any]) -> dict[str, Any]:
    """Run ``name``'s committed trace, (re)written from ``trace`` under ``RULEFORGE_REGENERATE_OPTIMIZE_TRACE=1``."""
    path = trace_path(name)
    if os.environ.get("RULEFORGE_REGENERATE_OPTIMIZE_TRACE") == "1":  # pragma: no cover - deliberate regeneration
        path.write_text(json.dumps(trace, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("failure", ["empty", "raise"])
@pytest.mark.parametrize("name", sorted(RUNS))
def test_the_program_reproduces_the_frozen_trace(name, failure, tmp_path, monkeypatch, isolated_run):
    trace = run_script_path(name, tmp_path / "run", monkeypatch, RuleForgeScript(failure=failure))
    if failure == "empty":
        assert trace == frozen(name, trace)
    else:
        assert trace == json.loads(trace_path(name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("failure", ["empty", "raise"])
@pytest.mark.parametrize("name", sorted(RUNS))
def test_optimize_reproduces_the_frozen_trace(name, failure, tmp_path, isolated_run):
    assert run_optimize(name, tmp_path / "run", failure) == json.loads(trace_path(name).read_text(encoding="utf-8"))


def test_the_program_refuses_an_initial_ruleset_its_own_text_would_change(tmp_path, monkeypatch, isolated_run):
    rules = tmp_path / "rules.jsonl"
    rules.write_text('{"id": "Rule 1", "rule": "Apply Rule 2: be kind."}\n{"id": "Rule 2", "rule": "Be brief."}\n',
                     encoding="utf-8")
    script, started = RuleForgeScript(), []
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")  # Hydra re-raises the run's error rather than exiting
    with pytest.raises(ValueError, match=r"its rule 0 \('Rule 1', 'Apply Rule 2: be kind\.'\) would come back as"):
        run_script_path("intent", tmp_path / "run", monkeypatch, script, rules=rules, started=started)
    assert started == [] and not (tmp_path / "run" / "episode_0").exists()


def test_the_program_refuses_a_config_lacking_a_reward_weight_before_starting_a_judge(tmp_path, monkeypatch,
                                                                                  isolated_run):
    started: list[str] = []
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(ValueError, match=r"the run's config sets no rule_amendment\.intent_method: set each"):
        run_script_path("intent", tmp_path / "run", monkeypatch, RuleForgeScript(),
                        extra=("~rule_amendment.intent_method",), started=started)
    assert started == [] and not (tmp_path / "run" / "episode_0").exists()


def test_the_program_refuses_a_used_run_directory_before_it_writes(tmp_path, monkeypatch, isolated_run):
    out = tmp_path / "run"
    run_script_path("intent", out, monkeypatch, RuleForgeScript())
    written = {path: path.read_bytes() for path in out.glob("episode_*/*/*") if path.is_file()}
    started: list[str] = []
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(FileExistsError, match="a run writes into a directory of its own"):
        run_script_path("intent", out, monkeypatch, RuleForgeScript(), started=started)
    assert started == []
    assert written and written == {path: path.read_bytes() for path in out.glob("episode_*/*/*") if path.is_file()}


def test_the_program_refuses_a_directory_holding_an_unclaimed_run_s_episodes(tmp_path, monkeypatch, isolated_run):
    out = tmp_path / "run"
    (out / "episode_0").mkdir(parents=True)
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(FileExistsError, match="holds a run's episodes"):
        run_script_path("intent", out, monkeypatch, RuleForgeScript())
    assert not (out / RUN_RECORD_DIR).exists() and not (out / "episode_0" / "rule_amendment").exists()


def _run_record(out: Path) -> dict[str, Any]:
    return json.loads((out / RUN_RECORD_DIR / RUN_RECORD_FNAME).read_text(encoding="utf-8"))


def test_a_shared_engine_s_seed_and_settings_are_recorded(tmp_path, monkeypatch, isolated_run):
    out, started = tmp_path / "run", []
    run_script_path("intent", out, monkeypatch, RuleForgeScript(), extra=("judge_model.seed=3",), started=started)
    record = _run_record(out)
    assert started.count("backend") == 1 and record["agent"] == "judge"
    assert record["judge"]["model_name"] == MODEL_NAME and record["judge"]["options"]["seed"] == 3
    assert record["agent_settings_not_used"] == {"seed": 16}


def test_a_seed_named_for_the_agents_alone_under_a_shared_engine_is_refused(tmp_path, monkeypatch, isolated_run):
    out, started = tmp_path / "run", []
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    with pytest.raises(ValueError, match=r"agent_model\.seed=5 is not the seed .*judge_model\.seed=16"):
        run_script_path("intent", out, monkeypatch, RuleForgeScript(), extra=("agent_model.seed=5",), started=started)
    assert started == [] and not (out / RUN_RECORD_DIR).exists()


def test_the_same_seed_named_for_both_under_a_shared_engine_is_taken(tmp_path, monkeypatch, isolated_run):
    out = tmp_path / "run"
    run_script_path("intent", out, monkeypatch, RuleForgeScript(), extra=("judge_model.seed=5", "agent_model.seed=5"))
    assert _run_record(out)["agent_settings_not_used"] == {}


def test_agents_on_another_model_have_an_engine_and_a_record_of_their_own(tmp_path, monkeypatch, isolated_run):
    out, started = tmp_path / "run", []
    run_script_path("intent", out, monkeypatch, RuleForgeScript(),
                    extra=("agent_model.model_name=another", "agent_model.seed=5"), started=started)
    record = _run_record(out)
    assert started.count("backend") == 2 and record["agent_settings_not_used"] == {}
    assert (record["agent"]["model_name"], record["agent"]["options"]["seed"]) == ("another", 5)


def test_a_run_s_config_records_the_panel_s_retries(tmp_path, monkeypatch, isolated_run):
    out = tmp_path / "run"
    run_script_path("intent", out, monkeypatch, RuleForgeScript())
    saved = yaml.safe_load((out / ".hydra" / "config.yaml").read_text())["eval_cfg"]
    assert (saved["panel_num_tries"], saved["panel_max_retry_seconds"]) == (1, 3600)
    composed = OmegaConf.load(REPO / "configs" / "config.yaml")
    assert composed.eval_cfg.panel_num_tries == 30 and composed.eval_cfg.panel_max_retry_seconds == 3600
