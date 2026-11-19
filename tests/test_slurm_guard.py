from __future__ import annotations

import os
import subprocess

import pytest

from tests.support import slurm_guard


@pytest.fixture
def record():
    """What the test made the guard record, taken back so the suite's own check does not fail the test."""
    since = slurm_guard.mark()
    reached: list[str] = []
    yield lambda: reached.extend(slurm_guard.reached_since(since, forget=True)) or reached


@pytest.mark.parametrize("call", [
    lambda: subprocess.run(["sbatch", "--version"], check=False),
    lambda: subprocess.run("srun --version", shell=True, check=False),
    lambda: subprocess.run(["true && scancel --version"], shell=True, check=False),
    lambda: subprocess.Popen(["scontrol", "--version"], env={"PATH": "/usr/bin:/bin"}),
], ids=["by-name", "shell", "shell-list", "own-environment"])
def test_a_slurm_command_is_refused_before_it_starts(record, call):
    """By name through the suite's ``PATH`` or a child's own, in a shell command line."""
    if not slurm_guard.real_commands():
        pytest.skip("no SLURM command on this machine")
    try:
        result = call()
    except slurm_guard.SlurmReached as error:
        assert "the suite never reaches SLURM" in str(error)
    else:
        code = result.wait() if isinstance(result, subprocess.Popen) else result.returncode
        assert code == slurm_guard.STUB_EXIT
    assert record()


def test_the_cluster_s_own_command_is_refused_by_its_path(record):
    if not slurm_guard.real_commands():
        pytest.skip("no SLURM command on this machine")
    with pytest.raises(slurm_guard.SlurmReached, match="a child process running /"):
        subprocess.run([sorted(slurm_guard.real_commands())[0], "--version"], check=False)
    assert record()


def test_a_script_a_child_runs_reaches_the_stub_on_its_own_path(record, tmp_path):
    real = sorted({os.path.dirname(command) for command in slurm_guard.real_commands()})
    if not real:
        pytest.skip("no SLURM command on this machine: a child finds none by name")
    first = tmp_path / "bin"
    path = os.pathsep.join([str(first), real[0], "/bin"])
    result = subprocess.run(["sh", "-c", 'sbatch --version; echo "$?"; echo "$PATH"'], env={"PATH": path},
                            capture_output=True, text=True, check=True)
    code, seen = result.stdout.splitlines()
    assert int(code) == slurm_guard.STUB_EXIT and seen == os.pathsep.join([str(first), slurm_guard.stubs(), real[0],
                                                                           "/bin"])
    assert record() == ["stub: \tsbatch --version"]  # its own environment: no PYTEST_CURRENT_TEST to name the test
    entries = os.environ["PATH"].split(os.pathsep)
    assert entries.index(slurm_guard.stubs()) < min(entries.index(directory) for directory in real if directory in
                                                    entries)


def test_a_child_that_inherits_a_path_a_test_set_without_the_stubs_gets_them(record, tmp_path, monkeypatch):
    real = sorted({os.path.dirname(command) for command in slurm_guard.real_commands()})
    if not real:
        pytest.skip("no SLURM command on this machine: a child finds none by name")
    first = tmp_path / "bin"
    monkeypatch.setenv("PATH", os.pathsep.join([str(first), real[0], "/bin"]))
    result = subprocess.run(["sh", "-c", 'sbatch --version; echo "$?"; echo "$PATH"'], capture_output=True, text=True,
                            check=True)
    code, seen = result.stdout.splitlines()
    assert int(code) == slurm_guard.STUB_EXIT and seen == os.pathsep.join([str(first), slurm_guard.stubs(), real[0],
                                                                           "/bin"])
    assert len(record()) == 1
    monkeypatch.undo()
    assert slurm_guard._with_stubs(os.environ)[0] is os.environ  # the suite's PATH holds them: inherited as it is


def test_submitit_s_slurm_executor_is_refused_unless_the_test_only_renders_one(record, tmp_path):
    import submitit

    with pytest.raises(slurm_guard.SlurmReached, match="submitit's SLURM executor"):
        submitit.AutoExecutor(folder=str(tmp_path), cluster="slurm")
    assert record()


@pytest.mark.allow_slurm_executor
def test_a_test_marked_so_may_build_one(record, tmp_path, monkeypatch):
    import submitit

    monkeypatch.setattr(submitit.slurm.slurm.SlurmExecutor, "affinity", classmethod(lambda cls: 2))
    assert submitit.AutoExecutor(folder=str(tmp_path), cluster="slurm").cluster == "slurm" and not record()


def test_other_programs_run_as_they_did(record):
    assert subprocess.run(["true"], check=False).returncode == 0
    assert subprocess.run("echo sbatchless", shell=True, capture_output=True, text=True).stdout == "sbatchless\n"
    assert not record()
