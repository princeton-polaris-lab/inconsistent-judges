from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from dataclasses import replace

import pytest
from omegaconf import OmegaConf

from ruleforge.launch.environment import LAUNCHER_CONFIG
from ruleforge.launch import control, records, train
from ruleforge.launch.cli import main
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.records import LaunchRecord
from tests.support.git import _git, make_repo
from tests.test_launch_backends import Completed, FakePopen, _executors

REPO = Path(__file__).resolve().parents[1]
DATA = ["--config-name", "training/grpo/qwen3_8b", "reward.truncated=zero"]
#: sacctmgr on a cluster where jit preempts normal and nothing preempts compact.
QOS = Completed({"sacctmgr": [(0, "normal|\njit|normal\ncompact|\n", "")]})


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clean checkout holding the repository's configs and the launcher config; the artifact root beside it."""
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu-partition")
    monkeypatch.delenv("RULEFORGE_SLURM_ACCOUNT", raising=False)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    checkout = make_repo(tmp_path)
    shutil.copytree(REPO / "configs", checkout / "configs", dirs_exist_ok=True)
    shutil.copy(REPO / LAUNCHER_CONFIG, checkout / LAUNCHER_CONFIG)
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "configs")
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    (tmp_path / "artifacts").mkdir()
    return checkout


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _records(tmp_path: Path) -> list[Path]:
    root = tmp_path / "artifacts" / ".ruleforge-launch"
    return sorted(root.iterdir()) if root.is_dir() else []


def hydra_dir(argv) -> str:
    """The Hydra directory a command names, below the artifact root (``${output_dir}`` is training/<run>)."""
    [run_dir] = [item.split("=", 1)[1] for item in argv if item.startswith("hydra.run.dir=")]
    [run_name] = [item.split("=", 1)[1] for item in argv if item.startswith("run_name=")]
    return run_dir.replace("${output_dir}", f"training/{run_name}")


def write_hydra(directory: Path, overrides, *, loss_weight: float = 4.0) -> None:
    """What Hydra writes into a preflight's directory: its composed config and overrides."""
    (directory / ".hydra").mkdir(parents=True)
    (directory / ".hydra" / "config.yaml").write_text(OmegaConf.to_yaml({
        "anchor": {"loss_weight": loss_weight}, "run_name": "r", "output_dir": "${oc.env:NOPE_X}"}))
    (directory / ".hydra" / "overrides.yaml").write_text(OmegaConf.to_yaml(list(overrides)))


def overrides_of(argv) -> list[str]:
    return [item for item in argv[len(train.PROGRAM):]
            if not item.startswith("-") and "=" in item and not item.startswith("hydra.")]


class Preflight(FakePopen):
    """A local preflight: it writes the Hydra files (unless it refuses) and exits ``code``; ``log`` says when."""

    def __init__(self, log: list, code: int = 0, loss_weight: float = 4.0):
        super().__init__([(code, ["GRPO plan: 2400 train rows"])])
        self.log, self.code, self.loss_weight = log, code, loss_weight

    def __call__(self, argv, **kwargs):
        self.log.append("preflight")
        if self.code == 0:
            write_hydra(Path(kwargs["env"]["RULEFORGE_ARTIFACT_ROOT"]) / hydra_dir(argv), overrides_of(argv),
                        loss_weight=self.loss_weight)
        return super().__call__(argv, **kwargs)


def logged(log: list):
    """A SLURM executor whose submissions are logged in ``log``."""
    jobs, factory = _executors()

    def executor(folder, max_num_timeout):
        made = factory(folder, max_num_timeout)
        submit = made.submit
        made.submit = lambda command: (log.append("submit"), submit(command))[1]
        return made

    return jobs, executor


SLURM = ["train", "--backend", "slurm", "--qos", "compact", "--gpu-count", "2", "--run-name", "r1"]


def test_a_slurm_run_is_preflighted_here_then_checked_then_submitted_as_one_task_per_gpu(repo, capsys):
    log: list[str] = []
    jobs, executor = logged(log)
    argv = [*SLURM, "--expect", "anchor.loss_weight=4.0", "--json", "--", *DATA, "anchor.loss_weight=4"]
    assert main(argv, repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 0
    assert log == ["preflight", "submit"]
    printed = _json(capsys)
    record = LaunchRecord.load(printed["launch_id"])
    [job] = record.jobs
    launch_id = record.launch_id
    assert job["argv"] == [*train.PROGRAM, *DATA, "anchor.loss_weight=4", "run_name=r1",
                           "resume_from_checkpoint=null",
                           f"hydra.run.dir=${{output_dir}}/hydra/{launch_id}/rank${{oc.env:SLURM_PROCID,0}}"]
    assert train.PROGRAM == ("python", "-m", "ruleforge.launch.programs.train_grpo") and train.WORKDIR == "."
    assert job["job_id"] == "r1" and job["workdir"] == "." and job["outputs"] == []
    submitit = job["request"]["submitit"]
    assert submitit["tasks_per_node"] == 2 and submitit["cpus_per_task"] == 8 and submitit["mem_gb"] == 240
    assert submitit["slurm_additional_parameters"] == {"qos": "compact", "no_requeue": True}
    assert jobs["jobs"][0][3] == 0 and record.data["options"]["rank_per_gpu"] is True
    run = record.data["train"]
    assert run["name"] == "r1" and run["output_dir"] == "training/r1" and run["ranks"] == 2
    assert (run["save_steps"], run["max_steps"], run["resume"]) == (100, 600, "none")
    assert run["expect"] == {"anchor.loss_weight": 4.0} and run["push"] == []
    preflight = run["preflight"]
    assert preflight["where"] == "local" and preflight["hydra_dir"] == f"training/r1/hydra/{launch_id}/preflight"
    assert preflight["command"][-2:] == ["preflight_only=true",
                                         f"hydra.run.dir=${{output_dir}}/hydra/{launch_id}/preflight"]
    assert preflight["config"]["anchor"] == {"loss_weight": 4.0} and "anchor.loss_weight=4" in preflight["overrides"]


def test_expect_is_checked_against_the_preflight_s_config_before_anything_is_submitted(repo, tmp_path, capsys):
    log: list[str] = []
    _, executor = logged(log)
    argv = [*SLURM, "--expect", "anchor.loss_weight=4.0", "--", *DATA, "anchor.loss_weight=4.0"]
    assert main(argv, repo=repo, popen=Preflight(log, loss_weight=1.0), executor=executor, run=QOS) == 2
    assert log == ["preflight"] and _records(tmp_path) == []
    assert "--expect: in the preflight's config, anchor.loss_weight is 1.0, not 4.0" in capsys.readouterr().err
    assert main([*SLURM, "--expect", "anchor.loss_weight=4.0", "--", *DATA], repo=repo, popen=Preflight(log),
                executor=executor, run=QOS) == 2
    assert log == ["preflight"]
    assert "in the config composed here, anchor.loss_weight is 1.0, not 4.0" in capsys.readouterr().err


def test_a_refused_preflight_submits_nothing_and_preflight_only_stops_after_it(repo, tmp_path, capsys):
    log: list[str] = []
    _, executor = logged(log)
    assert main([*SLURM, "--", *DATA], repo=repo, popen=Preflight(log, code=1), executor=executor, run=QOS) == 1
    assert log == ["preflight"] and _records(tmp_path) == []
    assert "the preflight refused r1 (exit 1): nothing was submitted" in capsys.readouterr().out
    assert main([*SLURM, "--preflight-only", "--", *DATA], repo=repo, popen=Preflight(log), executor=executor,
                run=QOS) == 0
    assert log == ["preflight", "preflight"] and _records(tmp_path) == []
    assert "r1 passed its preflight; nothing was submitted" in capsys.readouterr().out


def test_a_dry_run_prints_the_preflight_and_the_run_and_submits_nothing(repo, tmp_path, capsys):
    log: list[str] = []
    _, executor = logged(log)
    assert main([*SLURM, "--dry-run", "--", *DATA], repo=repo, popen=Preflight(log), executor=executor,
                run=QOS) == 0
    out = capsys.readouterr().out
    assert log == [] and _records(tmp_path) == [] and "== dry run: nothing was submitted" in out
    assert re.search(r"== preflight \(local\): python -m ruleforge\.launch\.programs\.train_grpo .* preflight_only=true", out)


def test_a_run_resumes_as_asked_and_only_one_that_resumes_latest_is_requeued(repo, tmp_path, capsys):
    dry = [*SLURM[:3], "--gpu-count", "2", "--run-name", "r1", "--dry-run", "--json"]
    # latest: requeued (after a preemption or the wall), whatever the QOS, even when sacctmgr cannot say.
    run = Completed({"sacctmgr": [(1, "", "unreachable")]})
    assert main([*dry, "--qos", "normal", "--resume", "latest", "--", *DATA], repo=repo, run=run) == 0
    [job] = _json(capsys)["jobs"]
    assert "resume_from_checkpoint=latest" in job["argv"] and job["resources"]["resumable"] is True
    assert job["request"]["submitit"]["slurm_additional_parameters"]["requeue"] is True
    assert job["request"]["max_num_timeout"] == records.TIMEOUT_REQUEUES
    assert run.calls == [["sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,MaxWall"]]
    for extra, message in ((["--qos", "normal"], "QOS normal may be preempted"),
                           ([], "no --qos is named"),):
        assert main([*dry, *extra, "--", *DATA], repo=repo, run=QOS) == 2
        assert message in capsys.readouterr().err
    assert main([*dry, "--qos", "compact", "--", *DATA], repo=repo, run=run) == 2
    assert "sacctmgr cannot say whether QOS compact may be preempted" in capsys.readouterr().err
    checkpoint = "training/r0/checkpoint-200"
    assert main([*dry, "--qos", "compact", "--resume", checkpoint, "--", *DATA], repo=repo, run=QOS) == 0
    [job] = _json(capsys)["jobs"]
    assert f"resume_from_checkpoint={tmp_path}/artifacts/{checkpoint}" in job["argv"]
    assert job["resources"]["resumable"] is False


@pytest.mark.parametrize("extra, message", [
    (["run_name=x"], "reaches 'run_name'"),
    (["hydra.run.dir=/x"], "reaches 'hydra.run.dir'"),
    (["+resume_from_checkpoint=latest"], "reaches 'resume_from_checkpoint'"),
    (["output_root=/x"], "reaches 'output_root'"),
    (["preflight_only=true"], "reaches 'preflight_only'"),
    (["hydra.output_subdir=/published/run/.hydra"], "gives Hydra's 'hydra.output_subdir' a value"),
    (["hydra.output_subdir=../../../published"], "gives Hydra's 'hydra.output_subdir' a value"),
    (["+hydra.job_logging.handlers.evil={class:logging.FileHandler,filename:/published/x}"],
     "gives Hydra's 'hydra.job_logging.handlers.evil' a value"),
    (["hydra.job.env_set.RULEFORGE_ARTIFACT_ROOT=/published"], "gives Hydra's 'hydra.job.env_set.RULEFORGE_ARTIFACT_ROOT'"),
    (["hydra.job.name=../../../../../../published/provenance"], "gives Hydra's 'hydra.job.name' a value"),
    (["hydra.searchpath=[file:///elsewhere]"], "gives Hydra's 'hydra.searchpath' a value"),
    (["-m"], "no -m"),
    (["grpo.save_steps=null"], "no whole save, step and batch counts"),
    (["grpo.num_generations=0"], "no whole save, step and batch counts"),
    (["--config-dir", "/elsewhere"], "--config-dir: a job's config is one of configs/"),
])
def test_what_the_verb_sets_and_a_config_it_cannot_read_are_refused(repo, capsys, extra, message):
    assert main([*SLURM, "--dry-run", "--", *DATA, *extra], repo=repo, run=QOS) == 2
    assert message in capsys.readouterr().err


def test_an_export_the_run_s_config_reads_is_refused_as_a_campaign_s_is(repo, capsys, monkeypatch):
    monkeypatch.delenv("STEPS_HERE", raising=False)
    steps = "grpo.max_steps=${oc.env:STEPS_HERE,100}"
    assert main([*SLURM, "--env", "STEPS_HERE=7", "--dry-run", "--", *DATA, steps], repo=repo, run=QOS) == 2
    assert "--env STEPS_HERE: a training job's config reads it" in capsys.readouterr().err
    assert main([*SLURM, "--env", "OTHER=7", "--dry-run", "--", *DATA, steps], repo=repo, run=QOS) == 0


def test_a_run_takes_gpus_whole_cpu_shares_a_name_and_a_training_backend(repo, capsys):
    assert main(["train", "--backend", "slurm", "--qos", "compact", "--run-name", "r1", "--dry-run", "--", *DATA],
                repo=repo, run=QOS) == 2
    assert "a training run takes GPUs" in capsys.readouterr().err
    assert main([*SLURM, "--cpus", "9", "--dry-run", "--", *DATA], repo=repo, run=QOS) == 2
    assert "--cpus 9 over 2 ranks" in capsys.readouterr().err
    assert main(["train", "--backend", "slurm", "--gpu-count", "1", "--run-name", "bad name", "--dry-run"],
                repo=repo) == 2
    assert "--run-name 'bad name'" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["train", "--backend", "local", "--gpu-count", "1", "--run-name", "r1"], repo=repo)
    assert "invalid choice: 'local'" in capsys.readouterr().err
    assert main(["train", "--gpu-count", "1", "--run-name", "r1", "--dry-run"], repo=repo) == 2
    assert "--backend is required" in capsys.readouterr().err


def test_the_gemma_run_spec_renders_the_measured_layout(repo, capsys):
    common = ["train", "--spec", "grpo_train_gemma_run", "--run-name", "g1", "--dry-run", "--json"]
    assert main(common, repo=repo, run=QOS) == 2
    assert "needs --var for condition, rows" in capsys.readouterr().err
    common += ["--var", "rows=grpo_data/gemma-screen-2026-10-01-train", "--var",
               "condition=sampling/gemma-screen-2026-10-01-confirm/b/gemma-4-31B-it/grpo-v2-anchors/confirm/condition.json",
               "--expect", "grpo.max_steps=600", "--expect", "grpo.save_steps=100", "--expect", "model.master_dtype=null"]
    assert main(common, repo=repo, run=QOS) == 2
    assert "QOS normal may be preempted" in capsys.readouterr().err
    assert main([*common, "--resume", "latest"], repo=repo, run=QOS) == 0
    printed = _json(capsys)
    assert printed["train"]["per_step"] == {"completions": 32, "prompts": 4, "replay_rows": 8}
    assert (printed["train"]["max_steps"], printed["train"]["save_steps"]) == (600, 100)
    [job] = printed["jobs"]
    root = "${oc.env:RULEFORGE_ARTIFACT_ROOT}"
    assert job["argv"][:5] == [*train.PROGRAM, "--config-name", "training/grpo/gemma4_31b"]
    assert job["argv"][5:9] == [
        f"data.train={root}/grpo_data/gemma-screen-2026-10-01-train/grpo_rows.jsonl",
        f"data.replay={root}/grpo_data/gemma-screen-2026-10-01-train/replay_rows.jsonl",
        f"data.anchor_condition={root}/sampling/gemma-screen-2026-10-01-confirm/b/gemma-4-31B-it/grpo-v2-anchors/confirm/"
        "condition.json", "reward.truncated=zero"]
    assert "resume_from_checkpoint=latest" in job["argv"]
    resources = job["resources"]
    assert (resources["gpus"], resources["mem_gb"], resources["timeout_min"], resources["resumable"]) == (
        4, 320, 2880, True)
    assert job["request"]["submitit"]["slurm_additional_parameters"]["requeue"] is True
    # rank 0's metric lines reach its log as printed (block-buffered on NFS, unread at the saves)
    assert job["env"] == ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "MAX_JOBS=8", "PYTHONUNBUFFERED=1"]


def test_the_gemma_pilot_launches_200_steps_under_the_run_spec(repo, capsys):
    common = ["train", "--spec", "grpo_train_gemma_run", "--run-name", "gemma-grpo-pilot", "--dry-run", "--json",
              "--resume", "latest", "--var", "rows=grpo_data/gemma-pilot-train", "--var",
              "condition=sampling/s-confirm/b/gemma-4-31B-it/grpo-v2-anchors/confirm/condition.json",
              "--expect", "grpo.max_steps=200", "--expect", "grpo.save_steps=100", "--expect",
              "model.master_dtype=null", "--expect", "anchor.loss_weight=4.0"]
    assert main([*common, "--", "anchor.loss_weight=4.0"], repo=repo, run=QOS) == 2
    assert "grpo.max_steps" in capsys.readouterr().err
    assert main([*common, "--", "anchor.loss_weight=4.0", "grpo.max_steps=200"], repo=repo, run=QOS) == 0
    printed = _json(capsys)
    assert (printed["train"]["max_steps"], printed["train"]["save_steps"]) == (200, 100)
    assert printed["train"]["per_step"] == {"completions": 32, "prompts": 4, "replay_rows": 8}
    [job] = printed["jobs"]
    assert job["argv"][9:11] == ["anchor.loss_weight=4.0", "grpo.max_steps=200"]


def test_the_gemma_mix_run_launches_the_spec_as_it_is_at_anchor_weight_1(repo, capsys):
    common = ["train", "--spec", "grpo_train_gemma_run", "--run-name", "gemma-grpo-mix", "--dry-run", "--json",
              "--resume", "latest", "--var", "rows=grpo_data/gemma-mix-train", "--var",
              "condition=sampling/s-confirm/b/gemma-4-31B-it/grpo-v2-anchors/confirm/condition.json",
              "--expect", "grpo.max_steps=600", "--expect", "grpo.save_steps=100", "--expect",
              "model.master_dtype=null", "--expect", "anchor.loss_weight=1.0"]
    assert main(common, repo=repo, run=QOS) == 0
    printed = _json(capsys)
    assert (printed["train"]["max_steps"], printed["train"]["save_steps"]) == (600, 100)
    [job] = printed["jobs"]
    assert not any(part.startswith("anchor.") for part in job["argv"])
    assert main([*common, "--", "anchor.loss_weight=4.0"], repo=repo, run=QOS) == 2
    assert "anchor.loss_weight is 4.0, not 1.0" in capsys.readouterr().err


def _warm_start(repo, capsys, tmp_path, *, overrides=None) -> tuple[int, dict, list[str]]:
    init = f"{tmp_path}/artifacts/training/gemma-grpo-mix/checkpoint-400"
    run_name = "gemma-grpo-mix-warm"
    given = [f"model.init_adapter={init}", "grpo.max_steps=300", "grpo.seed=1"]
    command = ["train", "--spec", "grpo_train_gemma_run", "--run-name", run_name, "--dry-run", "--json",
               "--resume", "latest", "--var", f"rows=grpo_data/{run_name}", "--var",
               "condition=sampling/s-confirm/b/gemma-4-31B-it/grpo-v2-anchors/confirm/condition.json",
               "--expect", "grpo.max_steps=300", "--expect", "grpo.save_steps=100", "--expect",
               "model.master_dtype=null", "--expect", "anchor.loss_weight=1.0", "--expect", "grpo.seed=1",
               "--expect", f"model.init_adapter={init}"]
    code = main([*command, "--", *(given if overrides is None else overrides)], repo=repo, run=QOS)
    return code, _json(capsys) if code == 0 else {}, given


def test_a_warm_start_launches_under_the_run_spec(repo, capsys, tmp_path):
    code, printed, given = _warm_start(repo, capsys, tmp_path)
    assert code == 0
    assert (printed["train"]["max_steps"], printed["train"]["save_steps"]) == (300, 100)
    assert printed["train"]["per_step"] == {"completions": 32, "prompts": 4, "replay_rows": 8}
    [job] = printed["jobs"]
    assert job["argv"][9:12] == given
    assert train.expectations(["grpo.max_steps=300", "grpo.seed=1"]) == {"grpo.max_steps": 300, "grpo.seed": 1}
    assert train.same_value(0.0, 0) and train.same_value(0, 0.0)
    code, _, _ = _warm_start(repo, capsys, tmp_path, overrides=[given[0], *given[2:]])
    assert code == 2 and "grpo.max_steps is 600, not 300" in capsys.readouterr().err


def test_the_gemma_config_composes_to_the_reward_set(repo, capsys, tmp_path):
    pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from ruleforge.grpo import trainer as tn
    from ruleforge.grpo.rewards import GrpoReward
    from ruleforge.core.digests import canonical_digest
    from ruleforge.core.artifacts import load_jsonl_records
    from tests.test_training_rewards import GEMMA_REWARD_DIGEST

    code, printed, _ = _warm_start(repo, capsys, tmp_path)
    assert code == 0
    _, cfg = train.plan(repo, printed["train"]["arguments"], run_name=printed["train"]["name"], resume="latest",
                        ranks=4)
    reward = OmegaConf.to_container(cfg.reward, resolve=True)
    assert reward == {"truncated": "zero", "lambda_int": 0.3, "lambda_comp": 0.1, "favoured_no_rate": True}
    assert canonical_digest(tn.resume_free({"reward": reward})) == GEMMA_REWARD_DIGEST
    rules = load_jsonl_records((REPO / train.WORKDIR / cfg.data.rules).resolve())  # the program's working directory
    reward_set = tn.configured_reward_set(reward, rules=rules, num_generations=int(cfg.grpo.num_generations),
                                          factory=GrpoReward)
    assert [func.__name__ for func in reward_set.reward_funcs] == list(tn.REWARD_FUNC_NAMES)
    assert reward_set.reward_weights == [1.0, 0.3, 0.1] and reward_set.favoured_no_rate
    assert tn.reward_form(reward_set) == (
        "sum (1 flat_soft_reward + 0.3 internal_consistency_reward + 0.1 map_completeness_reward)")
    assert cfg.model.init_adapter.endswith("gemma-grpo-mix/checkpoint-400") and cfg.grpo.seed == 1


def test_the_step_is_pinned_only_when_asked_and_a_count_no_whole_accumulation_gives_is_named_so(repo, capsys):
    assert main([*SLURM, "--dry-run", "--", *DATA], repo=repo, run=QOS) == 0
    assert "== r1: 2 rank(s); each step 32 completions (4 prompts) and 8 replay rows" in capsys.readouterr().out
    assert main([*SLURM, "--completions-per-step", "30", "--replay-rows-per-step", "8", "--dry-run", "--", *DATA],
                repo=repo, run=QOS) == 2
    assert ("32 completions, not 30 (--completions-per-step; no grpo.gradient_accumulation_steps does on 2 "
            "rank(s) at 2 per device)") in capsys.readouterr().err
    assert main([*SLURM, "--replay-rows-per-step", "0", "--dry-run", "--json", "--", *DATA, "anchor.enabled=false"],
                repo=repo, run=QOS) == 0
    assert _json(capsys)["train"]["per_step"]["replay_rows"] == 0


def test_a_slurm_run_s_limit_is_within_its_qos_s_wall(repo, capsys):
    walls = Completed({"sacctmgr": [(0, "normal||2-00:00:00\njit|normal|\ncompact||1-00:00:00\n", "")]})
    assert main([*SLURM, "--timeout-min", "2880", "--dry-run", "--", *DATA], repo=repo, run=walls) == 2
    assert "--timeout-min 2880 is past QOS compact's MaxWall (1440 minutes)" in capsys.readouterr().err
    assert main([*SLURM, "--timeout-min", "1440", "--dry-run", "--", *DATA], repo=repo, run=walls) == 0


def test_a_run_is_launched_again_only_once_its_other_launch_has_ended(repo, tmp_path, capsys):
    log: list[str] = []
    jobs, executor = logged(log)
    latest = [*SLURM, "--resume", "latest", "--json", "--", *DATA]
    assert main(latest, repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 0
    first = _json(capsys)["launch_id"]
    [run] = LaunchRecord.load(first).runs
    running = Completed({"sacctmgr": QOS.answers["sacctmgr"], "sacct": [(0, f"{run['slurm_job']}|RUNNING\n", "")]})
    assert main(latest, repo=repo, popen=Preflight(log), executor=executor, run=running) == 2
    err = capsys.readouterr().err
    assert f"{first} writes training/r1 (1 running): its run has not ended" in err and "--ignore-launch" in err
    assert log == ["preflight", "submit"] and len(_records(tmp_path)) == 1
    assert main(latest, repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 2
    assert "(1 unknown)" in capsys.readouterr().err
    assert main([*SLURM, "--resume", "latest", "--ignore-launch", first, "--json", "--", *DATA], repo=repo,
                popen=Preflight(log), executor=executor, run=running) == 0
    second = LaunchRecord.load(_json(capsys)["launch_id"])
    assert second.data["ignored_launches"] == [first] and log[-2:] == ["preflight", "submit"]
    [again] = second.runs
    ended = Completed({"sacctmgr": QOS.answers["sacctmgr"],
                       "sacct": [(0, f"{run['slurm_job']}|CANCELLED by 1\n{again['slurm_job']}|COMPLETED\n", "")]})
    assert main(latest, repo=repo, popen=Preflight(log), executor=executor, run=ended) == 0
    assert "ignored_launches" not in LaunchRecord.load(_json(capsys)["launch_id"]).data


def test_a_fresh_run_is_refused_where_a_run_has_started_and_a_resume_is_not(repo, tmp_path, capsys):
    log: list[str] = []
    _, executor = logged(log)
    (tmp_path / "artifacts" / "training" / "r1" / "code_snapshots").mkdir(parents=True)
    assert main([*SLURM, "--", *DATA], repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 2
    assert "training/r1 holds a run that has started (its code snapshot)" in capsys.readouterr().err
    assert log == []
    assert main([*SLURM, "--dry-run", "--", *DATA], repo=repo, run=QOS) == 0
    capsys.readouterr()
    for step in (100, 200):
        checkpoint = tmp_path / "artifacts" / "training" / "r1" / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "smrc_provenance.json").write_text("{}")
    assert main([*SLURM, "--resume", "latest", "--json", "--", *DATA], repo=repo, popen=Preflight(log),
                executor=executor, run=QOS) == 0
    record = LaunchRecord.load(_json(capsys)["launch_id"])
    assert log == ["preflight", "submit"] and record.data["train"]["saved_before"] == 200
    assert main(["stop", record.launch_id, "--after-save", "200"], repo=repo, run=QOS) == 2
    assert "training/r1 held checkpoint-200 before this launch" in capsys.readouterr().err


def test_a_run_whose_directory_leads_into_a_campaign_root_is_refused_before_its_preflight(repo, tmp_path, capsys):
    published = tmp_path / "artifacts" / "sampling" / "published"
    published.mkdir(parents=True)
    (published / "sampling_manifest.json").write_text("{}")
    (tmp_path / "artifacts" / "training").mkdir()
    (tmp_path / "artifacts" / "training" / "r1").symlink_to(published, target_is_directory=True)
    log: list[str] = []
    _, executor = logged(log)
    assert main([*SLURM, "--", *DATA], repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 2
    assert re.search(r"the run's directory \S*/sampling/published is a campaign root", capsys.readouterr().err)
    assert log == [] and _records(tmp_path) == []
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]
    assert main([*SLURM, "--dry-run", "--", *DATA], repo=repo, run=QOS) == 0


def _digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        digest.update(str(path.relative_to(directory)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize("planted", ["hydra", "run directory's parent"])
def test_a_run_whose_hydra_directory_or_run_directory_is_reached_through_a_symlink_is_refused(repo, tmp_path, capsys,
                                                                                            planted):
    artifacts = tmp_path / "artifacts"
    published = artifacts / "sampling" / "published"
    published.mkdir(parents=True)
    (published / "sampling_manifest.json").write_text("{}")
    if planted == "hydra":
        (artifacts / "training" / "r1").mkdir(parents=True)
        (artifacts / "training" / "r1" / "hydra").symlink_to(published, target_is_directory=True)
        target = published
        refusal = r"the run's Hydra directory \S*/training/r1/hydra is a symlink"
    else:
        target = artifacts / "elsewhere"
        (target / "r1").mkdir(parents=True)
        (artifacts / "training").symlink_to(artifacts / "elsewhere", target_is_directory=True)
        refusal = (r"the run's directory \S*/training/r1, as its jobs spell it, is not \S*/elsewhere/r1, where the "
                   r"launch's record and a pull name it: it is reached through a symlink")
    before = _digest(target)
    log: list[str] = []
    jobs, executor = logged(log)
    assert main([*SLURM, "--", *DATA], repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 2
    assert re.search(refusal, capsys.readouterr().err)
    assert log == [] and jobs["jobs"] == [] and _records(tmp_path) == [] and _digest(target) == before
    assert main([*SLURM, "--dry-run", "--", *DATA], repo=repo, run=QOS) == 0


@pytest.mark.parametrize("argument, refusal", [
    ("hydra.job.name=../../../../../../{published}/provenance", r"gives Hydra's 'hydra\.job\.name' a value"),
    ("hydra/job_logging=../../../../escape", r"the config 'hydra/job_logging/\.\./\.\./\.\./\.\./escape' is read "
                                             r"from outside configs/ and Hydra's own configs"),
])
def test_a_run_whose_log_file_hydra_would_name_outside_its_run_directory_is_refused_before_its_preflight(
        repo, tmp_path, capsys, argument, refusal):
    artifacts = tmp_path / "artifacts"
    published = artifacts / "sampling" / "published"
    published.mkdir(parents=True)
    (published / "sampling_manifest.json").write_text("{}")
    escape = repo.parent / "escape.yaml"  # configs/hydra/job_logging/../../../../escape: beside the checkout
    escape.write_text(f"version: 1\nhandlers:\n  file:\n    class: logging.FileHandler\n"
                      f"    filename: {published}/provenance.log\nroot:\n  handlers: [file]\n")
    before = _digest(published)
    log: list[str] = []
    jobs, executor = logged(log)
    extra = argument.format(published=str(published).lstrip("/"))
    assert main([*SLURM, "--", *DATA, extra], repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 2
    assert re.search(refusal, capsys.readouterr().err)
    assert log == [] and jobs["jobs"] == [] and _records(tmp_path) == [] and _digest(published) == before


@pytest.mark.parametrize("planted, refusal", [
    # an entry of the run's directory, which the run writes in (W&B's files, its completion logs, its alarms)
    ("wandb", r"\S*/training/r1/wandb, in the run's directory, is a symlink: the run writes its files there"),
    ("completions", r"\S*/training/r1/completions, in the run's directory, is a symlink"),
    ("alarms.jsonl", r"\S*/training/r1/alarms\.jsonl, in the run's directory, is a symlink"),
    # below a subtree that is the run's alone: a save, a completion log, a code snapshot
    ("checkpoint-100/model.safetensors", r"a directory the run writes \S*/training/r1/checkpoint-100: "
                                         r"\S*/checkpoint-100/model\.safetensors, below it, is a symlink"),
    ("completions/completions_00001.parquet", r"\S*/r1/completions: \S*/completions_00001\.parquet, below it"),
    ("code_snapshots/x.tar.gz", r"\S*/r1/code_snapshots: \S*/code_snapshots/x\.tar\.gz, below it, is a symlink"),
])
def test_a_run_whose_directory_holds_a_symlink_where_it_writes_is_refused(repo, tmp_path, capsys, planted, refusal):
    artifacts = tmp_path / "artifacts"
    published = artifacts / "sampling" / "published"
    published.mkdir(parents=True)
    (published / "sampling_manifest.json").write_text("{}")
    run = artifacts / "training" / "r1"
    link = run / planted
    link.parent.mkdir(parents=True, exist_ok=True)
    if "/" in planted:
        link.symlink_to(published / "sampling_manifest.json")
    else:
        link.symlink_to(published, target_is_directory=True)
    before = _digest(published)
    for resume in ([], ["--resume", "latest"]):
        log: list[str] = []
        jobs, executor = logged(log)
        argv = [*SLURM, *resume, "--", *DATA]
        assert main(argv, repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 2
        assert re.search(refusal, capsys.readouterr().err)
        assert log == [] and jobs["jobs"] == [] and _records(tmp_path) == [] and _digest(published) == before
    assert main([*SLURM, "--dry-run", "--", *DATA], repo=repo, run=QOS) == 0


def test_w_and_b_s_own_links_deep_in_its_directory_are_left_alone(repo, tmp_path, capsys):
    """W&B's core links its debug log in each run's own directory (``wandb/offline-run-*/logs/debug-core.log``)."""
    logs = tmp_path / "artifacts" / "training" / "r1" / "wandb" / "offline-run-x" / "logs"
    logs.mkdir(parents=True)
    (logs / "debug-core.log").symlink_to(tmp_path / "core.log")
    train.check_root(train.TrainRun("r1", "training/r1", "latest", "latest", 1, 100, 300, {}, ()), "L1",
                     str(tmp_path / "artifacts" / "training" / "r1"))


def test_a_launch_of_the_run_recorded_or_started_during_the_preflight_refuses_this_one(repo, tmp_path, capsys):
    log: list[str] = []
    _, executor = logged(log)
    latest = [*SLURM, "--resume", "latest", "--json", "--", *DATA]
    states = Completed({"sacctmgr": QOS.answers["sacctmgr"]})
    other: dict = {}

    class Overlapped(Preflight):
        """The first launch's preflight, during which the same run is launched again (from another shell, say)."""

        def __call__(self, argv, **kwargs):
            if not other:
                other["status"] = main(latest, repo=repo, popen=Preflight(log), executor=executor, run=QOS)
                other["record"] = LaunchRecord.load(_json(capsys)["launch_id"])
                [run] = other["record"].runs
                states.answers["sacct"] = [(0, f"{run['slurm_job']}|RUNNING\n", "")]
            return super().__call__(argv, **kwargs)

    assert main(latest, repo=repo, popen=Overlapped(log), executor=executor, run=states) == 2
    assert other["status"] == 0 and log == ["preflight", "submit", "preflight"]
    assert f"{other['record'].launch_id} writes training/r1 (1 running): its run has not ended" in \
        capsys.readouterr().err
    assert _records(tmp_path) == [other["record"].directory]
    snapshot = tmp_path / "artifacts" / "training" / "r2" / "code_snapshots"

    class Started(Preflight):
        def __call__(self, argv, **kwargs):
            snapshot.mkdir(parents=True)
            return super().__call__(argv, **kwargs)

    fresh = [*SLURM[:-1], "r2", "--", *DATA]
    assert main(fresh, repo=repo, popen=Started(log), executor=executor, run=QOS) == 2
    assert "training/r2 holds a run that has started (its code snapshot)" in capsys.readouterr().err
    assert log == ["preflight", "submit", "preflight", "preflight"] and len(_records(tmp_path)) == 1


def test_the_job_verb_refuses_a_second_live_writer_of_one_output(repo, tmp_path, capsys):
    _, executor = logged([])
    job = ["job", "--backend", "slurm", "--output", "training/r/merged/checkpoint-200", "--json", "--", "true"]
    assert main(job, repo=repo, executor=executor) == 0
    [run] = LaunchRecord.load(_json(capsys)["launch_id"]).runs
    running = Completed({"sacct": [(0, f"{run['slurm_job']}|RUNNING\n", "")]})
    assert main(job, repo=repo, executor=executor, run=running) == 2
    assert "writes training/r/merged/checkpoint-200 (1 running)" in capsys.readouterr().err
    other = ["job", "--backend", "slurm", "--output", "training/r/merged/checkpoint-300", "--", "true"]
    assert main(other, repo=repo, executor=executor, run=running) == 0


def test_expectations_are_plain_assignments_typed_as_hydra_types_them():
    assert train.expectations(["anchor.loss_weight=4.0", "reward.truncated=zero", "data.replay=null",
                               "grpo.report_to=[wandb]"]) == {
        "anchor.loss_weight": 4.0, "reward.truncated": "zero", "data.replay": None, "grpo.report_to": ["wandb"]}
    for bad in ("+x=1", "++x=1", "~x", "x=1,2", "g@pkg=a", "hydra/launcher=x", "=1"):
        with pytest.raises(LaunchError, match="KEY=VALUE, a config key"):
            train.expectations([bad])
    with pytest.raises(LaunchError, match="--expect names x twice"):
        train.expectations(["x=1", "x=2"])


def test_a_config_value_is_resolved_where_it_can_be_else_read_as_written(monkeypatch):
    monkeypatch.delenv("NOPE_X", raising=False)
    cfg = OmegaConf.create({"a": {"b": 1, "c": "${a.b}", "d": "${oc.env:NOPE_X}", "e": [1, 2]}, "f": {"g": 2}})
    assert train.config_value(cfg, "a.c") == 1 and train.config_value(cfg, "a.e") == [1, 2]
    assert train.config_value(cfg, "a.d") == "${oc.env:NOPE_X}" and train.config_value(cfg, "f") == {"g": 2}
    assert train.config_value(cfg, "a.z") is train.config_value(cfg, "q") is train._ABSENT
    assert train.config_value(cfg, "a.d.x") is train._ABSENT
    assert train.check_expectations(cfg, {"a.c": 1.0, "a.e": [1, 2]}, where="w") == {"a.c": 1, "a.e": [1, 2]}
    with pytest.raises(LaunchError, match="in w, a.z is absent, not 1; a.b is 1, not 2"):
        train.check_expectations(cfg, {"a.z": 1, "a.b": 2}, where="w")


def test_an_expected_boolean_matches_only_a_boolean_and_a_whole_number_its_float():
    cfg = OmegaConf.create({"a": {"on": True, "off": False, "w": 4, "n": 0}})
    assert train.check_expectations(cfg, {"a.on": True, "a.off": False, "a.w": 4.0, "a.n": 0}, where="w")
    for key, value in (("a.on", 1), ("a.off", 0), ("a.n", False), ("a.w", True)):
        with pytest.raises(LaunchError, match=f"{key} is"):
            train.check_expectations(cfg, {key: value}, where="w")


def test_a_preflight_that_left_no_hydra_files_is_refused(tmp_path):
    with pytest.raises(LaunchError, match="the preflight left no .hydra/config.yaml, .hydra/overrides.yaml"):
        train.read_preflight(tmp_path)


def test_training_resources_are_per_gpu_unless_given():
    assert train.training_resources({}, 2) == {"cpus": 16, "mem_gb": 240}
    assert train.training_resources({"cpus": 12, "mem_gb": 200}, 4) == {"cpus": 12, "mem_gb": 200}
    assert control.TRAIN_BACKENDS == ("slurm",)  # a local run has no GPUs of its own


class SaveBackend:
    """A training run's backend."""

    def __init__(self, landed: dict, polls, fail=False, blips=()):
        self.landed, self.polls, self.fail, self.blips = landed, list(polls), fail, set(blips)
        self.calls, self.asked, self.stopped = 0, [], []

    def exists(self, path):
        self.asked.append(path)
        if len(self.asked) - 1 in self.blips:
            raise LaunchError("runpod_run.py exists failed (255): ssh: connect to host: Connection timed out")
        return any(path.endswith(suffix) and self.calls >= poll for suffix, poll in self.landed.items())

    def states(self, record):
        self.calls += 1
        return {"r1": {"state": self.polls[min(self.calls, len(self.polls)) - 1]}}

    def stop(self, record, run):
        if self.fail:
            raise LaunchError("scancel failed")
        self.stopped.append(run["job_id"])
        return {"requested": True}


def train_record(tmp_path, **train_block) -> LaunchRecord:
    record = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L1", {
        "launch_id": "L1", "backend": "fake", "jobs": [{"job_id": "r1", "outputs": [], "resources": {
            "kind": "gpu", "timeout_min": 60, "resumable": False}}],
        "train": {"name": "r1", "output_dir": "training/r1", "save_steps": 100, "max_steps": 600, **train_block}})
    record.add_run({"job_id": "r1", "slurm_job": "7"})
    return record


def test_stop_after_save_waits_for_the_save_then_stops_and_never_writes_stop_json(tmp_path):
    record = train_record(tmp_path)
    backend = SaveBackend({"checkpoint-200/smrc_provenance.json": 2}, ["pending", "running"])
    said: list[str] = []
    clock = records_clock()
    status, outcome = train.stop_after_save(record, backend, 200, echo=said.append, sleep=clock.sleep, clock=clock)
    assert status == 0 and backend.stopped == ["r1"] and clock.sleeps == [30.0, 30.0]
    assert said == ["== waiting for r1's checkpoint-200 (1 pending)", "== waiting for r1's checkpoint-200 (1 running)"]
    assert outcome["landed"] == "training/r1/checkpoint-200/smrc_provenance.json"
    assert outcome["next_save"] == ["training/r1/checkpoint-300", "training/r1/tmp-checkpoint-300"]
    [logged] = json.loads((record.directory / records.STOP_FNAME).read_text())
    assert logged["after_save"] == 200 and logged["reason"] == "after checkpoint-200" and "r1" in logged["stopped"]
    assert [path.name for path in tmp_path.rglob("STOP.json")] == []


def test_stop_after_save_refuses_when_the_next_save_has_begun_and_warns_when_one_began_as_it_stopped(tmp_path):
    record = train_record(tmp_path)
    late = SaveBackend({"checkpoint-200/smrc_provenance.json": 0, "/tmp-checkpoint-300": 0}, ["running"])
    said: list[str] = []
    status, outcome = train.stop_after_save(record, late, 200, echo=said.append)
    assert status == train.STOP_REFUSED and late.stopped == [] and outcome["late"] == [
        "training/r1/tmp-checkpoint-300"]
    assert said[0].startswith("== late: training/r1/tmp-checkpoint-300 exists") and "stop --after-save 300" in said[0]
    assert not (record.directory / records.STOP_FNAME).exists()

    class Racing(SaveBackend):
        def stop(self, record, run):
            self.landed["/checkpoint-300"] = 0
            return super().stop(record, run)

    said.clear()
    racing = Racing({"checkpoint-200/smrc_provenance.json": 0}, ["running"])
    status, outcome = train.stop_after_save(record, racing, 200, reason="tripwire", echo=said.append)
    assert status == 0 and outcome["late"] == ["training/r1/checkpoint-300"]
    assert "the next save began as the run was stopped" in said[0]
    assert json.loads((record.directory / records.STOP_FNAME).read_text())[0]["reason"] == "tripwire"


def test_stop_after_save_ends_without_stopping_when_the_run_ends_first_or_the_wait_times_out(tmp_path):
    record = train_record(tmp_path)
    status, outcome = train.stop_after_save(record, SaveBackend({}, ["failed"]), 100, echo=lambda line: None)
    assert status == records.WAIT_FAILED and outcome["ended"] is True
    clock = records_clock()
    status, outcome = train.stop_after_save(record, SaveBackend({}, ["running"]), 100, timeout_min=1,
                                            sleep=clock.sleep, clock=clock)
    assert status == train.STOP_REFUSED and outcome["timed_out"] is True and clock.sleeps == [30.0, 30.0]
    ended = SaveBackend({"checkpoint-100/smrc_provenance.json": 1}, ["succeeded"])
    assert train.stop_after_save(record, ended, 100)[0] == 0 and ended.stopped == []
    status, outcome = train.stop_after_save(record, SaveBackend({"checkpoint-100/smrc_provenance.json": 0},
                                                                ["running"], fail=True), 100)
    assert status == records.WAIT_FAILED and outcome["stop"]["stop_errors"] == {"r1": "scancel failed"}


def test_stop_after_save_asks_again_when_the_backend_cannot_answer_and_never_stops_blind(tmp_path):
    record = train_record(tmp_path)
    landed = {"checkpoint-200/smrc_provenance.json": 4}
    said: list[str] = []
    clock = records_clock()
    blip = SaveBackend(landed, ["running"], blips={2})
    status, outcome = train.stop_after_save(record, blip, 200, echo=said.append, sleep=clock.sleep, clock=clock)
    assert status == 0 and blip.stopped == ["r1"]
    assert said[1].startswith("== could not ask whether checkpoint-200 has landed (LaunchError: runpod_run.py")
    assert said[2] == "== waiting for r1's checkpoint-200 (1 running)"
    unsure = SaveBackend({"checkpoint-200/smrc_provenance.json": 0}, ["running"], blips={1})
    status, outcome = train.stop_after_save(record, unsure, 200, echo=said.append)
    assert status == train.STOP_REFUSED and unsure.stopped == [] and outcome["unasked"] == [
        "training/r1/checkpoint-300"]
    assert "nothing was stopped; stop again" in said[-1]
    after = SaveBackend({"checkpoint-200/smrc_provenance.json": 0}, ["running"], blips={3})
    status, outcome = train.stop_after_save(record, after, 200, echo=said.append)
    assert status == 0 and after.stopped == ["r1"] and outcome["unasked"] == ["training/r1/checkpoint-300"]
    assert "whether the next save began as it stopped could not be asked" in said[-1]
    clock = records_clock()
    ended = SaveBackend({}, ["failed"], blips=set(range(10)))
    status, outcome = train.stop_after_save(record, ended, 200, timeout_min=1, sleep=clock.sleep, clock=clock)
    assert status == train.STOP_REFUSED and outcome["timed_out"] and outcome["errors"] and ended.stopped == []


def test_the_save_after_a_last_one_short_of_the_interval_is_the_run_s_end(tmp_path):
    """The trainer also saves at max_steps: with 550 steps, the save after checkpoint-500 is checkpoint-550."""
    record = train_record(tmp_path, max_steps=550)
    late = SaveBackend({"checkpoint-500/smrc_provenance.json": 0, "/checkpoint-550": 0}, ["running"])
    status, outcome = train.stop_after_save(record, late, 500)
    assert status == train.STOP_REFUSED and outcome["late"] == ["training/r1/checkpoint-550"]


def test_a_save_the_directory_held_before_the_launch_is_not_one_to_stop_after(tmp_path):
    record = train_record(tmp_path, saved_before=200)
    with pytest.raises(LaunchError, match="training/r1 held checkpoint-200 before this launch .* after 200"):
        train.stop_after_save(record, SaveBackend({}, ["running"]), 200)
    backend = SaveBackend({"checkpoint-300/smrc_provenance.json": 0}, ["running"])
    assert train.stop_after_save(record, backend, 300)[0] == 0 and backend.stopped == ["r1"]


def test_the_saves_before_a_launch_are_asked_last_first_and_a_fresh_run_has_none():
    run = train.TrainRun("r1", "training/r1", "latest", "latest", 2, 100, 550, {}, ())
    held = SaveBackend({"checkpoint-300/smrc_provenance.json": 0}, [])
    assert train.saved_before(held, run) == 300
    assert held.asked == [f"training/r1/checkpoint-{step}/smrc_provenance.json" for step in (550, 500, 400, 300)]
    assert train.saved_before(SaveBackend({}, []), run) == 0
    fresh = SaveBackend({}, [])
    assert train.saved_before(fresh, replace(run, resume="none")) == 0 and fresh.asked == []


@pytest.mark.parametrize("step", [0, 150, 600, 700])
def test_stop_after_save_names_a_save_before_the_last(tmp_path, step):
    with pytest.raises(LaunchError, match=f"--after-save {step}: a save of r1 before its last"):
        train.stop_after_save(train_record(tmp_path), SaveBackend({}, ["running"]), step)


def test_stop_after_save_is_for_a_training_launch(tmp_path):
    record = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L1", {"launch_id": "L1", "jobs": []})
    with pytest.raises(LaunchError, match="L1 is not a training launch"):
        train.stop_after_save(record, SaveBackend({}, []), 100)


def records_clock():
    from tests.test_launch_records import Clock

    return Clock()


def test_the_stop_verb_after_a_save_through_slurm(repo, tmp_path, capsys):
    log: list[str] = []
    _, executor = logged(log)
    assert main([*SLURM, "--json", "--", *DATA], repo=repo, popen=Preflight(log), executor=executor, run=QOS) == 0
    record = LaunchRecord.load(_json(capsys)["launch_id"])
    [run] = record.runs
    checkpoint = tmp_path / "artifacts" / "training" / "r1" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "smrc_provenance.json").write_text("{}")
    running = Completed({"sacct": [(0, f"{run['slurm_job']}|RUNNING\n", "")], "scancel": [(0, "", "")]})
    assert main(["stop", record.launch_id, "--after-save", "100", "--reason", "smoke"], repo=repo,
                run=running) == 0
    assert "== stop requested for r1" in capsys.readouterr().out and ["scancel", run["slurm_job"]] in running.calls
    (checkpoint.parent / "tmp-checkpoint-200").mkdir()
    assert main(["stop", record.launch_id, "--after-save", "100", "--json"], repo=repo, run=running) == 3
    assert _json(capsys)["late"] == ["training/r1/tmp-checkpoint-200"]
    assert main(["stop", record.launch_id, "--timeout-min", "5"], repo=repo, run=running) == 2
    assert "--timeout-min bounds the wait of --after-save" in capsys.readouterr().err
    assert [path.name for path in (tmp_path / "artifacts").rglob("STOP.json")] == []


def test_a_preflight_s_config_is_read_here(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    argv = [*train.PROGRAM, "run_name=r1", "hydra.run.dir=${output_dir}/hydra/L/preflight"]
    write_hydra(tmp_path / hydra_dir(argv), overrides_of(argv))
    found = train.preflight_config("training/r1/hydra/L/preflight")
    assert found["overrides"] == ["run_name=r1"] and found["config"]["anchor"]["loss_weight"] == 4.0


def test_the_run_spec_trains_qwen3_8b_on_gemma_s_recipe(repo, capsys):
    inputs = ["--var", "rows=grpo_data/qwen-mix", "--var",
              "condition=sampling/a-bank/b-000/Qwen3-8B/grpo-v2-anchors/bank/condition.json"]
    run = ["train", "--spec", "grpo_train_run", "--var", "recipe=training/grpo/qwen3_8b_mix", "--run-name", "q1",
           *inputs,
           "--expect", "grpo.max_steps=500", "--expect", "grpo.save_steps=100", "--expect", "anchor.loss_weight=1.0",
           "--expect", "reward.favoured_no_rate=true", "--resume", "latest", "--dry-run", "--json"]
    assert main(run, repo=repo, run=QOS) == 0
    printed = _json(capsys)
    assert printed["train"]["per_step"] == {"completions": 32, "prompts": 4, "replay_rows": 8}
    assert (printed["train"]["ranks"], printed["train"]["max_steps"], printed["train"]["save_steps"]) == (4, 500, 100)
    [job] = printed["jobs"]
    root = "${oc.env:RULEFORGE_ARTIFACT_ROOT}"
    assert job["argv"][:9] == [
        *train.PROGRAM, "--config-name", "training/grpo/qwen3_8b_mix",
        f"data.train={root}/grpo_data/qwen-mix/grpo_rows.jsonl", f"data.replay={root}/grpo_data/qwen-mix/replay_rows.jsonl",
        f"data.anchor_condition={root}/sampling/a-bank/b-000/Qwen3-8B/grpo-v2-anchors/bank/condition.json",
        "reward.truncated=zero"]
    resources = job["resources"]
    assert (resources["gpus"], resources["mem_gb"], resources["timeout_min"], resources["qos"], resources["resumable"]) == (
        4, 480, 2880, "normal", True)


def test_a_run_on_a_profile_takes_its_qos_for_the_wall_and_requeue_checks(repo, tmp_path, capsys):
    profile = tmp_path / "site.yaml"
    profile.write_text("partition: p\naccount: a\nqos: {gpu: compact, cpu: c}\ngpus_per_node: 8\nexclude: [n1]\n"
                       "defaults: {timeout_min: 30, cpus: 2, mem_gb: 8, mem_gb_per_gpu: 16}\nsetup: []\n")
    walls = Completed({"sacctmgr": [(0, "normal||2-00:00:00\njit|normal|\ncompact||1-00:00:00\n", "")]})
    argv = ["train", "--backend", "slurm", "--gpu-count", "2", "--run-name", "r1", "--profile", str(profile),
            "--dry-run", "--json"]
    assert main([*argv, "--timeout-min", "2880", "--", *DATA], repo=repo, run=walls) == 2
    assert "--timeout-min 2880 is past QOS compact's MaxWall (1440 minutes)" in capsys.readouterr().err
    assert main([*argv, "--", *DATA], repo=repo, run=walls) == 0
    printed = _json(capsys)
    [job] = printed["jobs"]
    assert printed["profile"] == str(profile) and printed["train"]["push"] == []
    assert job["request"]["submitit"]["slurm_additional_parameters"] == {"account": "a", "qos": "compact",
                                                                         "no_requeue": True, "exclude": "n1"}
