from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import pytest

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.provenance import repo_root
from ruleforge.launch import environment
from ruleforge.launch.environment import (
    LAUNCHER_CONFIG,
    LaunchError,
    artifact_relative,
    check_env_name,
    env_script,
    launcher_setup,
    new_run_id,
    relative_subpath,
    run_workdir,
)
from ruleforge.launch.environment import LaunchError as JobsLaunchError

REPO = Path(__file__).resolve().parents[1]

#: The GPU launcher's setup lines, byte for byte, as every paper run ran under them.
LAUNCHER_SETUP = [
    "export VLLM_NO_USAGE_STATS=1",
    "export MAX_JOBS=8",
    "export VLLM_FLOAT32_MATMUL_PRECISION=highest",
    "export VLLM_BATCH_INVARIANT=0",
    "unset TRITON_F32_DEFAULT",
    "export TIKTOKEN_RS_CACHE_DIR=$HOME/.cache/tiktoken",
]


def test_the_repo_root_is_the_checkout_the_module_came_from():
    assert repo_root() == REPO and (repo_root() / LAUNCHER_CONFIG).is_file()


def test_a_launch_error_is_every_launch_s_refusal_and_a_library_error():
    assert JobsLaunchError is LaunchError and issubclass(LaunchError, ValueError)
    assert issubclass(LaunchError, RuleForgeError)


def test_the_launcher_setup_is_the_gpu_launchers_own(monkeypatch):
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.delenv("RULEFORGE_SLURM_PARTITION", raising=False)
    assert launcher_setup(REPO) == LAUNCHER_SETUP
    monkeypatch.setenv("MAX_JOBS", "3")
    assert "export MAX_JOBS=3" in launcher_setup(REPO)


def test_the_env_script_runs_the_setup_then_the_exports_then_the_command():
    script = env_script(["export A=1", "unset B"], env=["A=two words", "C=$HOME"])
    assert script.splitlines() == ["set -e", "export A=1", "unset B", "export A='two words'", "export C='$HOME'",
                                   'exec "$@"']
    result = subprocess.run(["bash", "-c", script, "prog", "bash", "-c", 'echo "$A|$C|${B-unset}"'],
                            capture_output=True, text=True, check=True, env={"B": "set", "PATH": "/usr/bin:/bin"})
    assert result.stdout == "two words|$HOME|unset\n"
    assert "TIKTOKEN_RS_CACHE_DIR" in env_script([], env=["TIKTOKEN_RS_CACHE_DIR=/c"])  # a cache, not a credential


def test_a_failing_setup_line_stops_the_job_before_its_command():
    script = env_script(["false", "export A=1"])
    result = subprocess.run(["bash", "-c", script, "prog", "bash", "-c", "echo ran"], capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin"})
    assert (result.returncode, result.stdout) == (1, "")


@pytest.mark.parametrize(("env", "message"), [
    (["NOEQUALS"], "expected NAME=VALUE"),
    (["1BAD=x"], "not an environment variable name"),
    (["RULEFORGE_ARTIFACT_ROOT=/x"], "the launcher owns"),
    (["RULEFORGE_MODEL_ROOT=/x"], "the launcher owns"),
    (["SLURM_JOB_ID=1"], "the launcher owns"),  # a draw's provenance reads it
    (["PYTHONPATH=/x"], "the launcher owns"),
    (["A=1", "A=2"], "more than once"),
    (["HF_TOKEN=x"], "looks like a credential"),
    (["OPENAI_API_KEY=x"], "looks like a credential"),
])
def test_the_exports_are_checked(env, message):
    with pytest.raises(LaunchError, match=message):
        env_script([], env=env)


def test_a_name_the_caller_reserves_is_refused_too():
    check_env_name("MINE", flag="--env")
    with pytest.raises(LaunchError, match="--env 'MINE': the launcher owns that variable"):
        check_env_name("MINE", flag="--env", reserved=("MINE",))


@pytest.mark.parametrize("name", ["RULEFORGE_MODAL_RUN_ID", "RULEFORGE_RUNPOD_RUN_ID", "RUNPOD_POD_ID", "MODAL_TASK_ID"])
def test_the_retired_runners_run_ids_are_no_launcher_names_since_no_draw_reads_them(name):
    assert env_script([], env=[f"{name}=x"]).splitlines()[1] == f"export {name}=x"
    assert not hasattr(environment, "PROVENANCE_RUN_ID_ENV") and "RUNPOD_" not in environment.RESERVED_ENV_PREFIXES


def test_launch_ids_name_the_second_the_commit_and_a_random_suffix():
    moment = datetime.datetime(2026, 9, 25, 5, 6, 7, tzinfo=datetime.timezone.utc)
    run_id = new_run_id("0123456789abcdef", now=moment)
    assert run_id.startswith("20260925T050607Z-01234567-") and len(run_id) == 30
    assert new_run_id("0123456789abcdef").split("-")[1] == "01234567"
    assert environment.utc_now().endswith("+00:00")


def test_artifact_paths_are_relative_to_the_root(tmp_path):
    env = {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path)}
    (tmp_path / "sampling" / "x").mkdir(parents=True)
    assert artifact_relative("artifact://sampling/x", env) == "sampling/x"
    assert artifact_relative("sampling//x/./", env) == "sampling/x"
    assert artifact_relative(str(tmp_path / "sampling" / "x"), env) == "sampling/x"
    with pytest.raises(ValueError, match="not under"):
        artifact_relative("/somewhere/else", env)
    for bad in ("", ".", "a/../b", "artifact://"):
        with pytest.raises(ValueError, match="below the root"):
            artifact_relative(bad, env)


def test_a_home_relative_path_is_resolved_against_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "root" / "populations").mkdir(parents=True)
    env = {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path / "root")}
    assert artifact_relative("~/root/populations", env) == "populations"


def test_a_subpath_and_a_workdir_stay_below_their_root():
    assert relative_subpath("a/b", what="x") == "a/b"
    with pytest.raises(ValueError, match="x '/a'"):
        relative_subpath("/a", what="x")
    assert run_workdir("") == run_workdir(".") == "." and run_workdir("scripts/sampling") == "scripts/sampling"
    with pytest.raises(ValueError, match="--workdir '../up'"):
        run_workdir("../up")
