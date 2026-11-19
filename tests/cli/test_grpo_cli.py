from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import pytest

import ruleforge.grpo.merging as merge_module
import ruleforge.grpo.trainer as trainer
import ruleforge.grpo as grpo_package
from ruleforge.cli import grpo, records
from ruleforge.grpo.lora_parity import DEFAULT_TARGET_MODULES
from ruleforge.grpo.merging import MergeExpectation
from ruleforge.grpo import RowsBuild


def recorder(monkeypatch, module, name, result):
    real = getattr(module, name)
    seen = []

    @functools.wraps(real)
    def record(*args, **kwargs):
        seen.append(kwargs)
        return result

    monkeypatch.setattr(module, name, record)
    return seen


def test_rows_passes_its_flags_as_the_api_s_arguments(cli, monkeypatch, tmp_path):
    build = RowsBuild(out=tmp_path / "o", train=tmp_path / "t", replay=tmp_path / "r", scenarios=tmp_path / "s",
                      manifest=tmp_path / "m", anchor_condition="a", rules=tmp_path / "rules", rules_sha256="x",
                      markers={"train": "y"})
    seen = recorder(monkeypatch, grpo_package, "rows", build)
    status, out, _ = cli("grpo", "rows", "--anchors", "a", "--scenarios", "train.jsonl", "--rules", "r.jsonl",
                         "--model", "m", "--out", "out", "--require-protected", "test.jsonl", "--prompt-draws", "1",
                         "--allow-no-held-out", "--class-a-share-cap", "0.2")
    assert status == 0 and out["anchor_condition"] == "a" and out["markers"] == {"train": "y"} and "payload" not in out
    call = seen[0]
    assert (call["prompt_draws"], call["allow_no_held_out"], call["class_a_share_cap"]) == (1, True, 0.2)
    assert call["scenarios"] == str(Path.cwd() / "train.jsonl") and call["pins"] == {}
    assert (Path.cwd() / "out.runs").is_dir()


def test_merge_takes_its_expectation_as_the_adapter_s_fields(cli, monkeypatch):
    seen = recorder(monkeypatch, merge_module, "merge_adapter", {"result": {"weights_sha256": "w"}})
    status, out, _ = cli("grpo", "merge", "--base", "/models/Qwen3-8B", "--adapter", "ckpt", "--out", "merged",
                         "--expect-rank", "32", "--expect-alpha", "64", "--expect-base-model", "Qwen3-8B")
    assert status == 0 and out == {"result": {"weights_sha256": "w"}}
    call = seen[0]
    assert call["expect"] == MergeExpectation(rank=32, alpha=64.0, base_model="Qwen3-8B")
    assert call["expect"].target_modules == DEFAULT_TARGET_MODULES and call["threads"] == 8
    assert call["adapter_dir"] == str(Path.cwd() / "ckpt") and call["out_dir"] == str(Path.cwd() / "merged")


def test_each_function_command_says_what_it_does_in_grpo_s_help():
    assert all(command.summary and "(" not in command.summary for command in grpo.COMMANDS.values())


def test_a_function_command_s_flags_follow_its_parameters():
    parser = argparse.ArgumentParser()
    grpo.add_function_arguments(parser, grpo.COMMANDS["merge"])
    args = parser.parse_args(["--threads", "2", "--cast-bf16", "--expect-target-modules", "q_proj", "k_proj",
                              "--training-step", "7"])
    assert (args.threads, args.cast_bf16, args.expect__target_modules, args.training_step_override) == (
        2, True, ["q_proj", "k_proj"], 7)
    rows = argparse.ArgumentParser()
    grpo.add_function_arguments(rows, grpo.COMMANDS["rows"])
    assert "--pins" not in rows.format_help() and "--require-protected" in rows.format_help()


def test_train_runs_grpo_on_its_composed_config_and_records_the_run(cli, monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: seen.append(cfg))
    exported = []
    monkeypatch.setattr(trainer, "export_submitit_distributed_env", lambda environ: exported.append(True))
    status, out, _ = cli("grpo", "train", "--run-name", "r", "--model", "/m", "--output-root", tmp_path / "out", "--",
                         "data.train=/d/t", "data.replay=/d/r",
                         "data.rules=/d/rules", "data.rules_sha256=x", "data.anchor_condition=/d/c",
                         "reward.truncated=zero")
    output_dir = tmp_path / "out" / "r"
    assert status == 0 and out == {"output_dir": str(output_dir), "preflight_only": False} and exported == [True]
    assert seen[0]["grpo"]["max_steps"] == 600 and seen[0]["output_dir"] == str(output_dir)
    assert next(records.runs_dir(output_dir).iterdir()).name.endswith("-grpo-train")


@pytest.mark.parametrize("flag, allowed", [((), False), (("--allow-unscreened-rows",), True)])
def test_train_takes_unscreened_rows_only_when_asked(cli, monkeypatch, tmp_path, flag, allowed):
    seen = []
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: seen.append(cfg))
    monkeypatch.setattr(trainer, "export_submitit_distributed_env", lambda environ: None)
    status, _, error = cli("grpo", "train", "--run-name", "r", "--model", "/m", "--output-root", tmp_path / "out", *flag,
                           "--", "data.train=/d/t", "data.replay=/d/r",
                           "data.rules=/d/rules", "data.rules_sha256=x", "data.anchor_condition=/d/c",
                           "reward.truncated=zero")
    assert status == 0, error
    assert seen[0]["data"]["allow_unscreened_rows"] is allowed


def test_a_preflight_without_the_base_model_s_directory_says_so(cli, monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "no-weights"))
    config = Path(__file__).resolve().parents[2] / "configs/training/grpo/gemma4_31b.yaml"
    status, _, error = cli("grpo", "train", "--config", config, "--run-name", "R", "--preflight-only", "--",
                           "data.train=x", "data.replay=y", "data.anchor_condition=z", "reward.truncated=zero")
    assert status != 0 and "is not a directory: set RULEFORGE_MODEL_ROOT" in error and "Repo id" not in error


def test_train_without_the_training_stack_is_refused_in_one_line_naming_what_installs_it(cli, monkeypatch):
    monkeypatch.setitem(sys.modules, "trl", None)
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: pytest.fail("trained"))
    status, _, error = cli("grpo", "train", "--run-name", "r", "--output-root", "out")
    assert status == 2 and error == ("ruleforge grpo: MissingDependency: training needs trl, which this environment "
                                     "lacks: add --group grpo to uv sync, or pip install -e '.[grpo]' from the checkout\n")


TRAIN_DATA = ("data.train=/d/t", "data.replay=/d/r", "data.rules=/d/rules", "data.rules_sha256=x",
              "data.anchor_condition=/d/c", "reward.truncated=zero")


def test_train_needs_the_training_stack_only_where_it_trains(cli, monkeypatch, tmp_path):
    """A submission imports no trainer, a preflight needs no TRL or PEFT, and a training run needs both."""
    for name in ("trl", "peft", "ruleforge.grpo.trainer"):
        monkeypatch.setitem(sys.modules, name, None)
    status, out, error = cli("grpo", "train", "--run-name", "r", "--output-root", tmp_path / "out", "--slurm",
                             "example", "--dry-run", "--json")
    assert status == 0 and [job["job_id"] for job in out["jobs"]] == ["grpo-train"], error
    monkeypatch.setitem(sys.modules, "ruleforge.grpo.trainer", trainer)
    seen = []
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: seen.append(cfg))
    monkeypatch.setattr(trainer, "export_submitit_distributed_env", lambda environ: None)
    status, _, error = cli("grpo", "train", "--run-name", "r", "--model", "/m", "--output-root", tmp_path / "out",
                           "--preflight-only", "--", *TRAIN_DATA)
    assert status == 0 and seen[0]["preflight_only"] is True, error
    status, _, error = cli("grpo", "train", "--run-name", "r", "--model", "/m", "--output-root", tmp_path / "out",
                           "--", *TRAIN_DATA)
    assert status == 2 and "training needs trl, peft, which this environment lacks" in error and len(seen) == 1


def test_train_needs_its_required_keys(cli, monkeypatch):
    monkeypatch.setattr(trainer, "run_grpo", lambda cfg: pytest.fail("trained"))
    status, _, error = cli("grpo", "train", "--run-name", "r", "--output-root", "out")
    assert status == 2 and "Missing mandatory value" in error


@pytest.mark.parametrize("argv, gpus, tasks", [
    (["grpo", "train", "--run-name", "r", "--output-root", "out", "--gpus", "2"], 2, {"tasks_per_node": 2}),
    (["grpo", "rows", "--anchors", "a", "--scenarios", "s", "--rules", "r", "--model", "m", "--out", "o"], 0, {}),
])
def test_a_grpo_command_is_one_job_under_slurm(cli, argv, gpus, tasks):
    status, out, _ = cli(*argv, "--slurm", "example", "--dry-run", "--json")
    assert status == 0
    [job] = out["jobs"]
    assert job["resources"]["gpus"] == gpus and job["argv"][:3] == ["ruleforge", *argv[:2]]
    assert "--slurm" not in job["argv"] and "--json" not in job["argv"]
    assert {key: job["request"]["submitit"][key] for key in tasks} == tasks


def test_a_training_job_under_slurm_is_sized_per_rank_as_the_launcher_s_train_verb_sizes_it(cli):
    from ruleforge.launch import train

    train_argv = ["grpo", "train", "--run-name", "r", "--output-root", "out", "--slurm", "example", "--dry-run",
                  "--json"]
    status, out, error = cli(*train_argv, "--gpus", "2")
    assert status == 0, error
    [job] = out["jobs"]
    assert (job["resources"]["cpus"], job["resources"]["mem_gb"]) == (2 * train.CPUS_PER_GPU, 2 * train.MEM_GB_PER_GPU)
    assert job["request"]["submitit"]["cpus_per_task"] == train.CPUS_PER_GPU
    status, out, _ = cli(*train_argv, "--gpus", "2", "--cpus", "6", "--mem-gb", "100")
    assert status == 0 and (out["jobs"][0]["resources"]["cpus"], out["jobs"][0]["resources"]["mem_gb"]) == (6, 100)
    status, _, error = cli(*train_argv, "--gpus", "2", "--cpus", "9")
    assert status == 2 and "--cpus 9 over 2 ranks" in error


def _published(path):
    path.mkdir(parents=True)
    (path / "sampling_manifest.json").write_text("{}")
    return path


def test_a_link_or_a_campaign_root_below_the_out_is_refused_before_anything_is_written(cli, monkeypatch, tmp_path):
    seen = recorder(monkeypatch, grpo_package, "rows", None)
    published = _published(tmp_path / "published")
    out = Path.cwd() / "out"
    out.mkdir()
    (out / "data").symlink_to(published)
    argv = ("grpo", "rows", "--anchors", "a", "--scenarios", "p", "--rules", "r", "--model", "m", "--out", "out",
            "--require-protected", "t")
    for submission in ((), ("--slurm", "example", "--dry-run")):
        status, _, error = cli(*argv, *submission)
        assert status == 2 and f"{out / 'data'}, below the command's target {out}, is a symlink" in error
    (out / "data").unlink()
    _published(out / "rows" / "inner")
    status, _, error = cli(*argv)
    assert status == 2 and f"the command's target {out} holds a campaign root of its own" in error
    assert not seen and not records.runs_dir(out).exists()
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]


def test_train_s_output_directory_is_checked_before_a_job_is_submitted(cli, tmp_path):
    published = _published(tmp_path / "published")
    status, _, error = cli("grpo", "train", "--run-name", "r", "--output-root", published, "--slurm", "example",
                           "--dry-run")
    assert status == 2 and "is inside the campaign root" in error
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]


@pytest.mark.parametrize("command, argv", [
    ("rows", ("--anchors", "a", "--scenarios", "s", "--rules", "r", "--model", "m")),
    ("merge", ("--base", "b", "--adapter", "c", "--expect-rank", "32", "--expect-alpha", "64",
               "--expect-base-model", "Qwen3-8B"))])
def test_a_linked_partial_build_is_refused_before_the_api_runs(cli, monkeypatch, tmp_path, command, argv):
    """A build is written to ``<out>.partial`` beside its target, then renamed there."""
    module, name = (grpo_package, "rows") if command == "rows" else (merge_module, "merge_adapter")
    seen = recorder(monkeypatch, module, name, None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (Path.cwd() / "out.partial").symlink_to(elsewhere, target_is_directory=True)
    status, _, error = cli("grpo", command, *argv, "--out", "out")
    assert status == 2 and "its .partial directory" in error and "is a symlink" in error, error
    assert not seen and not records.runs_dir(Path.cwd() / "out").exists() and not list(elsewhere.iterdir())
    assert grpo.COMMANDS[command].beside == (merge_module.PARTIAL_SUFFIX,)


@pytest.mark.parametrize("out", [".ruleforge-launch", ".."])
def test_a_target_meeting_the_launch_records_is_refused_before_anything_is_submitted(cli, monkeypatch, tmp_path, out):
    from ruleforge.cli import slurm

    monkeypatch.setattr(slurm, "submit", lambda *args, **kwargs: pytest.fail("a job was submitted"))
    artifacts = tmp_path / "outer" / "artifacts"
    artifacts.mkdir(parents=True)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(artifacts))
    target = Path(os.path.normpath(artifacts / out))
    status, _, error = cli("grpo", "rows", "--anchors", "a", "--scenarios", "s", "--rules", "r", "--model", "m",
                           "--out", target, "--slurm", "example")
    assert status == 2 and "the run: its --out" in error and "the launch's records" in error, error
    status, _, error = cli("grpo", "train", "--run-name", "r", "--slurm", "example", "--", f"output_dir={target}")
    assert status == 2 and "the run: its output_dir" in error and "the launch's records" in error, error
    assert sorted(path.relative_to(tmp_path) for path in (tmp_path / "outer").rglob("*")) == [
        Path("outer") / "artifacts"]
