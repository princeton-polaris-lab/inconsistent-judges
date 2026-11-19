from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import submitit
from omegaconf import OmegaConf

from ruleforge.launch import backends, control
from ruleforge.launch.backends import (
    Command,
    CommandFailed,
    LocalBackend,
    ResumableCommand,
    SlurmBackend,
    backend_for,
    preflight_request,
    run_preflight,
    wall_minutes,
)
from ruleforge.launch.environment import LAUNCHER_CONFIG, env_script, launcher_setup
from ruleforge.launch.profile import Profile
from ruleforge.launch import records
from ruleforge.launch.jobs import JobSpec, Resources
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.records import LaunchRecord

REPO = Path(__file__).resolve().parents[1]
HEAD = "2403d53a491454d063ab2d86a04e650c2a6a8974"
CODE = {"commit": HEAD, "dirty": False, "code_state_sha256": "c"}


#: A launcher config naming a cluster: the real one's setup lines, its own partition and account.
SITE = "partition: gpu-partition\naccount: lab-account\nsignal_delay_s: 120\n"


def _repo(tmp_path: Path, launcher: str | None = None) -> Path:
    """A checkout holding a launcher config: :data:`SITE` after the real one's setup lines, or ``launcher``."""
    repo = tmp_path / "checkout"
    config = repo / LAUNCHER_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    if launcher is None:
        setup = OmegaConf.to_container(OmegaConf.load(REPO / LAUNCHER_CONFIG).setup, resolve=False)
        launcher = SITE + OmegaConf.to_yaml({"setup": setup})
    config.write_text(launcher)
    return repo


def _spec(job_id="j", *, gpus=0, qos=None, resumable=False, **overrides) -> JobSpec:
    resources = Resources("gpu" if gpus else "cpu", cpus=2, gpus=gpus, mem_gb=8, timeout_min=30, qos=qos,
                          resumable=resumable)
    return JobSpec(job_id, "job", 0, ("python", "x.py"), resources, **overrides)


def _record(tmp_path: Path, backend, specs) -> LaunchRecord:
    specs = [backend.prepare(spec) for spec in specs]
    jobs = [{**spec.to_json(), "request": request} for spec, request in zip(specs, backend.render(specs))]
    return LaunchRecord.create(tmp_path / "launches" / "L1", {"launch_id": "L1", "backend": backend.name,
                                                               "code": dict(CODE), "jobs": jobs})


class Completed:
    """Records each command and answers it from ``answers`` (by the command's first word), in turn."""

    def __init__(self, answers=None):
        self.answers = {key: list(value) for key, value in (answers or {}).items()}
        self.calls: list[list[str]] = []
        self.envs: list = []

    def __call__(self, argv, **kwargs):
        assert kwargs["check"] is False
        self.calls.append(list(argv))
        self.envs.append(kwargs.get("env"))
        key = next((word for word in argv if word in self.answers), None)
        if key is None:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        answer = self.answers[key].pop(0) if len(self.answers[key]) > 1 else self.answers[key][0]
        if isinstance(answer, BaseException):
            raise answer
        code, stdout, stderr = answer
        return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr=stderr)


class FakePopen:
    """A child process: prints ``lines`` and exits ``code`` (in turn, one per launch)."""

    def __init__(self, launches):
        self.launches = list(launches)
        self.calls: list[tuple[list[str], str]] = []
        self.envs: list = []
        self.killed: list[int] = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs["cwd"]))
        self.envs.append(kwargs["env"])
        code, lines = self.launches.pop(0)
        process = type("Process", (), {})()
        process.stdout = io.StringIO("".join(line + "\n" for line in lines))
        process.wait = lambda: code
        process.kill = lambda launch=len(self.calls): self.killed.append(launch)
        return process


class FakeJob:
    def __init__(self, job_id):
        self.job_id = job_id


class FakeExecutor:
    """submitit's executor, as far as the backend uses it; ids from ``next_id``; ``fail_at`` raises that submit."""

    def __init__(self, log, folder, max_num_timeout, *, fail_at=None):
        self.log, self.folder, self.max_num_timeout, self.fail_at = log, folder, max_num_timeout, fail_at
        self.parameters = {}
        self.batching = False

    def update_parameters(self, **parameters):
        self.parameters = parameters

    def submit(self, command):
        self.log["submitted"] += 1
        if self.log["submitted"] == self.fail_at:
            raise RuntimeError("sbatch: error: Batch job submission failed")
        self.log["jobs"].append((command, dict(self.parameters), self.batching, self.max_num_timeout))
        base = 99 + len(self.log["executors"])
        return FakeJob(f"{base}_{sum(1 for job in self.log['jobs'] if job[2])-1}" if self.batching else str(base))

    @contextlib.contextmanager
    def batch(self):
        self.batching = True
        yield
        self.batching = False


def _executors(fail_at=None):
    log = {"submitted": 0, "jobs": [], "executors": []}

    def factory(folder, max_num_timeout):
        log["executors"].append((folder, max_num_timeout))
        return FakeExecutor(log, folder, max_num_timeout, fail_at=fail_at)

    return log, factory


def test_the_cluster_comes_from_the_launcher_config(tmp_path):
    assert SlurmBackend(_repo(tmp_path)).site() == {"partition": "gpu-partition", "account": "lab-account",
                                                    "signal_delay_s": 120}
    bare = _repo(tmp_path / "bare", "partition: p\nsetup: []\n")
    assert SlurmBackend(bare).site() == {"partition": "p", "account": None, "signal_delay_s": None}


def test_a_cpu_job_gets_the_normal_qos_unless_it_names_one(tmp_path):
    backend = SlurmBackend(_repo(tmp_path))
    assert backend.prepare(_spec()).resources.qos == "normal"
    assert backend.prepare(_spec(qos="jit", gpus=1)).resources.qos == "jit"
    assert backend.prepare(_spec(gpus=1)).resources.qos is None
    assert LocalBackend(tmp_path).prepare(_spec()).resources.qos is None


def test_a_slurm_request_is_the_launchers_setup_then_the_jobs_exports(tmp_path):
    repo = _repo(tmp_path)
    backend = SlurmBackend(repo)
    [request] = backend.render([backend.prepare(_spec(env=("HF_HUB_OFFLINE=1",), workdir="scripts",
                                                      resumable=True))])
    assert request["submitit"]["slurm_setup"] == [*launcher_setup(repo), "export HF_HUB_OFFLINE=1"]
    assert request["submitit"]["slurm_signal_delay_s"] == 120
    assert request["submitit"]["slurm_additional_parameters"] == {"account": "lab-account", "qos": "normal",
                                                                  "requeue": True}
    assert request["max_num_timeout"] == backends.TIMEOUT_REQUEUES
    assert request["argv"] == ["python", "x.py"] and request["cwd"] == str(repo / "scripts")
    bare = SlurmBackend(_repo(tmp_path / "bare", "partition: p\nsetup: []\n"))
    [plain] = bare.render([_spec(gpus=1)])
    assert "slurm_signal_delay_s" not in plain["submitit"] and plain["max_num_timeout"] == 0
    assert plain["cwd"] == str(bare.repo)


def test_slurm_submits_one_job_or_one_array_per_set_of_parameters(tmp_path):
    log, factory = _executors()
    backend = SlurmBackend(_repo(tmp_path), executor=factory, echo=lambda line: None)
    record = _record(tmp_path, backend, [_spec("a"), _spec("b"), _spec("c", gpus=1, resumable=True)])
    backend.submit(record)
    assert [run["slurm_job"] for run in record.runs] == ["100_0", "100_1", "101"]
    assert json.loads(record.path.read_text())["runs"] == record.runs
    (first, parameters, batched, _), _, (third, alone, single, timeouts) = log["jobs"]
    assert batched and parameters["name"] == "L1" and type(first) is Command
    assert first.argv == ["python", "x.py"] and first.cwd == str(backend.repo)
    assert first.bin_dir == str(Path(sys.executable).parent)
    assert not single and alone["name"] == "c" and type(third) is ResumableCommand and timeouts == 3
    assert log["executors"][0] == (record.directory / "submitit", 0)


def test_a_slurm_launch_that_dies_part_way_keeps_the_jobs_it_submitted(tmp_path):
    log, factory = _executors(fail_at=2)
    backend = SlurmBackend(_repo(tmp_path), executor=factory, echo=lambda line: None)
    record = _record(tmp_path, backend, [_spec("a", gpus=1), _spec("b")])
    with pytest.raises(RuntimeError, match="submission failed"):
        backend.submit(record)
    assert json.loads(record.path.read_text())["runs"] == [{"job_id": "a", "slurm_job": "100"},
                                                           {"job_id": "b", "slurm_job": None}]


class QueuedThenRaises(FakeExecutor):
    """submitit's submit when sbatch queues the job and the call then raises: queued, and no job id comes back."""

    def submit(self, command):
        super().submit(command)
        raise OSError("[Errno 116] Stale file handle: moving the submission's pickle")


def test_a_slurm_submission_that_raises_after_sbatch_never_reads_as_not_launched(tmp_path):
    log = {"submitted": 0, "jobs": [], "executors": []}
    backend = SlurmBackend(_repo(tmp_path), executor=lambda folder, timeouts: QueuedThenRaises(log, folder, timeouts),
                           run=Completed(), echo=lambda line: None)
    record = _record(tmp_path, backend, [_spec("a"), _spec("b")])
    with pytest.raises(OSError, match="Stale file handle"):
        backend.submit(record)
    assert log["submitted"] == 1 and json.loads(record.path.read_text())["runs"] == [  # one array: both unread
        {"job_id": "a", "slurm_job": None}, {"job_id": "b", "slurm_job": None}]
    assert backend.states(record) == {"a": {"state": "submitting"}, "b": {"state": "submitting"}}
    assert backend.run.calls == []  # no job id to ask SLURM about
    record.finish(OSError("Stale file handle"))
    unread = {"state": "unknown", "error": backends.UNREAD_SUBMISSION}
    assert backend.states(record) == {"a": unread, "b": unread}
    stopped = records.stop(record, backend)
    assert stopped["stopped"] == {} and set(stopped["stop_errors"]) == {"a", "b"}
    assert "look by hand (squeue --me: the job named a or L1), scancel what it queued, then pass --ignore-launch L1" \
        in stopped["stop_errors"]["a"] and backend.run.calls == []
    record.update_run("a", slurm_job="11")
    backend.run = Completed({"sacct": [(0, "11|RUNNING\n", "")]})
    assert backend.states(record) == {"a": {"state": "running", "slurm_state": "RUNNING"}, "b": unread}
    assert backend.run.calls[0][-1] == "11"
    with pytest.raises(LaunchError, match="records no run of c"):
        record.update_runs({"a": {"slurm_job": "12"}, "c": {"slurm_job": "13"}})
    assert record.run_for("a")["slurm_job"] == "11"


def test_a_slurm_array_is_recorded_whole_before_any_of_its_lines_is_echoed(tmp_path):
    def echo(line):
        raise BrokenPipeError("the terminal went away")

    backend = SlurmBackend(_repo(tmp_path), executor=_executors()[1], echo=echo)
    record = _record(tmp_path, backend, [_spec("a"), _spec("b"), _spec("c")])
    with pytest.raises(BrokenPipeError):
        backend.submit(record)
    assert [run["slurm_job"] for run in json.loads(record.path.read_text())["runs"]] == ["100_0", "100_1", "100_2"]


@pytest.mark.allow_slurm_executor  # builds it and submits nothing (tests/support/slurm_guard.py)
def test_the_slurm_executor_is_submitits_on_slurm(tmp_path, monkeypatch):
    # submitit refuses to build one where no srun is on PATH (CI's runners): say there is one.
    monkeypatch.setattr(submitit.slurm.slurm.SlurmExecutor, "affinity", classmethod(lambda cls: 2))
    executor = backends.slurm_executor(tmp_path / "submitit", 3)
    assert executor.cluster == "slurm" and executor._executor.max_num_timeout == 3


def test_slurm_rows_spread_a_pending_array_over_its_tasks():
    text = "7|COMPLETED\n8_0|RUNNING\n8_[1-3%2]|PENDING\n9_[5,7]|PENDING\nnoise\n"
    assert backends.slurm_rows(text) == {"7": "COMPLETED", "8_0": "RUNNING", "8_1": "PENDING", "8_2": "PENDING",
                                         "8_3": "PENDING", "9_5": "PENDING", "9_7": "PENDING"}
    assert backends.slurm_state("CANCELLED by 1000") == "stopped" and backends.slurm_state("") == "unknown"
    assert backends.slurm_state("RUNNING+") == "running" and backends.slurm_state("WEIRD") == "unknown"


def test_slurm_states_come_from_sacct_else_squeue(tmp_path):
    run = Completed({"sacct": [(0, "11|TIMEOUT\n12|PENDING\n", "")], "squeue": [(0, "13|RUNNING\n", "")]})
    backend = SlurmBackend(_repo(tmp_path), run=run)
    record = _record(tmp_path, backend, [_spec("a"), _spec("b"), _spec("c"), _spec("d"), _spec("e")])
    for job, slurm_job in zip("abcd", ("11", "12", "13", "14")):
        record.add_run({"job_id": job, "slurm_job": slurm_job})
    record.finish()
    states = backend.states(record)
    assert {job: entry["state"] for job, entry in states.items()} == {
        "a": "failed", "b": "pending", "c": "running", "d": "unknown", "e": "not-launched"}
    assert states["a"]["slurm_state"] == "TIMEOUT"
    assert run.calls[0][:6] == ["sacct", "-X", "-n", "-P", "-o", "JobID,State"] and run.calls[0][-1] == "11,12,13,14"
    assert run.calls[1][-1] == "13,14"
    # No accounting (sacct fails), no SLURM here at all: squeue alone, then nothing.
    run = Completed({"sacct": [(1, "", "accounting disabled")], "squeue": [(0, "11|RUNNING\n", "")]})
    assert SlurmBackend(backend.repo, run=run).states(record)["a"]["state"] == "running"
    run = Completed({"sacct": [FileNotFoundError("sacct")], "squeue": [FileNotFoundError("squeue")]})
    assert {entry["state"] for entry in SlurmBackend(backend.repo, run=run).states(record).values()} == {
        "unknown", "not-launched"}


@pytest.mark.parametrize("raw", ["PREEMPTED", "NODE_FAIL"])
def test_a_resumable_job_that_slurm_requeues_is_pending_not_failed(tmp_path, raw):
    run = Completed({"sacct": [(0, f"11|{raw}\n12|{raw}\n", "")], "scancel": [(0, "", "")]})
    backend = SlurmBackend(_repo(tmp_path), run=run)
    record = _record(tmp_path, backend, [_spec("resumable", resumable=True), _spec("plain")])
    record.add_run({"job_id": "resumable", "slurm_job": "11"})
    record.add_run({"job_id": "plain", "slurm_job": "12"})
    record.finish()
    states = backend.states(record)
    assert states == {"resumable": {"state": "pending", "slurm_state": raw},
                      "plain": {"state": "failed", "slurm_state": raw}}
    clock = iter(range(0, 10**6, 60))
    code, _ = records.wait(record, backend, timeout_min=1, sleep=lambda seconds: None, clock=lambda: next(clock))
    assert code == records.WAIT_TIMED_OUT
    assert records.stop(record, backend)["stopped"] == {"resumable": {"slurm_job": "11", "requested": True}}
    assert run.calls[-1] == ["scancel", "11"]


def test_a_launch_with_no_runs_asks_nothing(tmp_path):
    run = Completed()
    backend = SlurmBackend(_repo(tmp_path), run=run)
    assert backend.states(_record(tmp_path, backend, [_spec()])) == {"j": {"state": "submitting"}}
    assert run.calls == []


def test_slurm_stops_through_scancel_and_pulls_nothing(tmp_path):
    run = Completed({"scancel": [(0, "", ""), (1, "", "scancel: error: Invalid job id")]})
    backend = SlurmBackend(_repo(tmp_path), run=run)
    record = _record(tmp_path, backend, [_spec()])
    assert backend.stop(record, {"job_id": "j", "slurm_job": "11"}) == {"slurm_job": "11", "requested": True}
    with pytest.raises(LaunchError, match="Invalid job id"):
        backend.stop(record, {"job_id": "j", "slurm_job": "12"})
    assert backend.pull(record, "sampling/x") is None and run.calls == [["scancel", "11"], ["scancel", "12"]]


def test_the_shared_filesystem_backends_look_under_the_artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    (tmp_path / "training" / "run" / "checkpoint-100").mkdir(parents=True)
    backend = SlurmBackend(tmp_path)
    assert backend.exists("training/run/checkpoint-100") and backend.exists("artifact://training/run")
    assert not backend.exists(str(tmp_path / "training" / "run" / "checkpoint-200"))


def test_a_slurm_jobs_command_fails_the_job_with_its_status(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "tool").write_text("#!/bin/bash\nexit 7\n")
    (bin_dir / "tool").chmod(0o755)
    assert Command(["true"], tmp_path, str(bin_dir))() == 0
    with pytest.raises(CommandFailed, match="exited with status 3"):
        Command(["bash", "-c", "exit 3"], tmp_path, str(bin_dir))()
    # The environment's bin comes first on PATH.
    with pytest.raises(CommandFailed, match="tool exited with status 7"):
        Command(["tool"], tmp_path, str(bin_dir))()
    # The warning signal before the wall does not end the command; the job's handler does.
    assert Command(["bash", "-c", "kill -USR2 $$; echo alive"], tmp_path, str(bin_dir))() == 0


def test_the_environments_bin_goes_first_on_path():
    assert backends.environment_bin() == str(Path(sys.executable).parent)
    assert backends.with_bin("/venv/bin", {"PATH": "/usr/bin", "A": "1"}) == {"PATH": "/venv/bin:/usr/bin", "A": "1"}
    assert backends.with_bin("/venv/bin", {})["PATH"] == "/venv/bin"


def test_a_resumable_command_is_requeued_with_itself():
    command = ResumableCommand(["python", "x.py"], "/repo", "/venv/bin")
    assert isinstance(command, submitit.helpers.Checkpointable)
    again = command.checkpoint()
    assert again.function is command and again.args == ()


def test_the_warning_signal_is_ignored_in_the_command():
    before = signal.getsignal(signal.SIGUSR2)
    try:
        backends.ignore_warning_signal()
        assert signal.getsignal(signal.SIGUSR2) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGUSR2, before)


def test_a_local_job_runs_in_the_foreground_under_the_setup_and_is_logged(tmp_path):
    repo = _repo(tmp_path)
    (repo / "scripts").mkdir()
    said: list[str] = []
    backend = LocalBackend(repo, echo=said.append)
    ok = JobSpec("ok", "job", 0, ("bash", "-c", 'echo $VLLM_BATCH_INVARIANT $A; pwd; echo "${PATH%%:*}"; echo err >&2'),
                 Resources("cpu", 1, 0, 1, 5), workdir="scripts", env=("A=a b",))
    bad = JobSpec("bad", "job", 0, ("bash", "-c", "exit 4"), Resources("cpu", 1, 0, 1, 5))
    record = _record(tmp_path, backend, [ok, bad])
    assert record.jobs[0]["request"]["argv"][:2] == ["bash", "-c"] and record.jobs[0]["request"]["argv"][3:5] == [
        "ruleforge-launch", "bash"]
    backend.submit(record)
    log = Path(record.runs[0]["log"]).read_text()
    # The launching environment's bin comes first on PATH.
    assert log == f"0 a b\n{repo / 'scripts'}\n{Path(sys.executable).parent}\nerr\n" and "0 a b" in said
    assert [run["exit_code"] for run in record.runs] == [0, 4]
    assert {job: entry["state"] for job, entry in backend.states(record).items()} == {"ok": "succeeded",
                                                                                       "bad": "failed"}
    assert backend.foreground and not SlurmBackend(repo).foreground


def _failing_echo(line):
    if line == "started":
        raise BrokenPipeError("the terminal went away")


def test_a_local_launch_that_fails_while_its_job_runs_kills_the_job_and_never_reads_its_run_as_ended(tmp_path):
    repo = _repo(tmp_path)
    pid_file = tmp_path / "pid"
    backend = LocalBackend(repo, echo=_failing_echo)
    long = JobSpec("long", "job", 0, ("bash", "-c", f"echo $$ > {pid_file}; echo started; exec sleep 600"),
                   Resources("cpu", 1, 0, 1, 5))
    record = _record(tmp_path, backend, [long, _spec("next")])
    with pytest.raises(BrokenPipeError):
        backend.submit(record)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    [run] = json.loads(record.path.read_text())["runs"]
    assert "exit_code" not in run and run["error"] == "BrokenPipeError: the terminal went away"
    record.finish(BrokenPipeError("the terminal went away"))
    assert backend.states(record) == {"long": {"state": "unknown", "error": "BrokenPipeError: the terminal went away"},
                                      "next": {"state": "not-launched"}}


def test_a_process_a_local_job_started_without_exec_outlives_the_kill_so_its_run_is_never_ended(tmp_path):
    repo = _repo(tmp_path)
    writer_pid = tmp_path / "writer.pid"
    backend = LocalBackend(repo, echo=_failing_echo)
    wrapper = JobSpec("wrapper", "job", 0, ("bash", "-c", f"sleep 600 > /dev/null 2>&1 & echo $! > {writer_pid}; "
                                                          "echo started; wait"),
                      Resources("cpu", 1, 0, 1, 5), outputs=("out/x",))
    record = _record(tmp_path, backend, [wrapper])
    with pytest.raises(BrokenPipeError):
        backend.submit(record)
    writer = int(writer_pid.read_text())
    try:
        os.kill(writer, 0)
        record.finish(BrokenPipeError("the terminal went away"))
        states = backend.states(record)
        assert states["wrapper"]["state"] == "unknown" and states["wrapper"]["state"] not in records.ENDED
        assert records.live_launches([record], lambda other: {"wrapper": {"out/x"}}, backend.states) == [
            {"launch_id": "L1", "shared": {"wrapper": {"out/x"}}, "states": {"wrapper": states["wrapper"]},
             "record_gone": False}]
    finally:
        os.kill(writer, signal.SIGKILL)


def test_a_local_launch_that_fails_before_its_child_starts_records_its_run_failed(tmp_path):
    def popen(*args, **kwargs):
        raise FileNotFoundError("bash")

    backend = LocalBackend(_repo(tmp_path), popen=popen)
    record = _record(tmp_path, backend, [_spec("a")])
    with pytest.raises(FileNotFoundError):
        backend.submit(record)
    [run] = json.loads(record.path.read_text())["runs"]
    assert run["exit_code"] is None and run["error"] == "FileNotFoundError: bash"
    record.finish(FileNotFoundError("bash"))
    assert backend.states(record) == {"a": {"state": "failed", "exit_code": None}}


def test_a_local_run_is_never_stopped_by_the_launcher(tmp_path):
    backend = LocalBackend(_repo(tmp_path))
    record = _record(tmp_path, backend, [_spec("a")])
    record.add_run({"job_id": "a", "log": "a.log", "error": "BrokenPipeError: x"})
    record.finish(BrokenPipeError("x"))
    entry = records.stop(record, backend, reason="test")
    assert entry["stopped"] == {} and "look for it by hand" in entry["stop_errors"]["a"]
    assert "--ignore-launch L1" in entry["stop_errors"]["a"]


def test_a_local_run_with_no_exit_status_never_reads_as_ended(tmp_path):
    backend = LocalBackend(_repo(tmp_path))
    record = _record(tmp_path, backend, [_spec("a"), _spec("b")])
    record.add_run({"job_id": "a", "log": "a.log"})
    assert backend.states(record) == {"a": {"state": "submitting"}, "b": {"state": "submitting"}}
    record.data["state"] = "failed"
    assert backend.states(record) == {"a": {"state": "unknown"}, "b": {"state": "not-launched"}}
    assert {entry["state"] for entry in backend.states(record).values()} & set(records.ENDED) == {"not-launched"}


@pytest.mark.parametrize("kind", [LocalBackend, SlurmBackend])
def test_a_stop_during_a_launch_ends_its_submission(tmp_path, kind):
    backend = kind(_repo(tmp_path), executor=_executors()[1], popen=FakePopen([]), echo=lambda line: None)
    record = _record(tmp_path, backend, [_spec("a")])
    (record.directory / records.STOP_FNAME).write_text("[]")
    with pytest.raises(LaunchError, match="stop of L1 was requested"):
        backend.submit(record)
    assert record.runs == []


def test_a_backend_says_what_it_does_on_stderr(tmp_path, capsys):
    LocalBackend(tmp_path).echo("== job: exit status 0")
    assert capsys.readouterr() == ("", "== job: exit status 0\n")


def test_backends_by_name(tmp_path):
    assert isinstance(backend_for("local", tmp_path), LocalBackend)
    assert set(backends.BACKENDS) == {"slurm", "local"}
    with pytest.raises(LaunchError, match="backend 'k8s'"):
        backend_for("k8s", tmp_path)


def _profile(tmp_path: Path, **overrides) -> Profile:
    values = dict(path=tmp_path / "site.yaml", partition="p", account="acct", qos_gpu="g", qos_cpu="c",
                  gpus_per_node=4, timeout_min=30, cpus=2, mem_gb=8, mem_gb_per_gpu=16, signal_delay_s=60,
                  setup=("export A=1",), exclude=("n1", "n2"))
    return Profile(**{**values, **overrides})


def test_a_slurm_backend_on_a_profile_takes_its_cluster_setup_and_excluded_nodes(tmp_path):
    backend = SlurmBackend(tmp_path / "no-checkout", profile=_profile(tmp_path))
    gpu, cpu = backend.prepare(_spec("g", gpus=2, qos="mine")), backend.prepare(_spec("c"))
    requests = backend.render([gpu, cpu])
    for request in requests:
        assert request["submitit"]["slurm_partition"] == "p" and request["submitit"]["slurm_setup"] == ["export A=1"]
        assert request["submitit"]["slurm_signal_delay_s"] == 60
    assert requests[0]["submitit"]["slurm_additional_parameters"] == {
        "account": "acct", "qos": "mine", "no_requeue": True, "exclude": "n1,n2"}
    assert "qos" not in requests[1]["submitit"]["slurm_additional_parameters"]
    assert SlurmBackend(tmp_path).prepare(_spec("c")).resources.qos == backends.CPU_QOS
    with pytest.raises(LaunchError, match="big asks for 5 GPUs; a node of site has 4"):
        backend.render([_spec("big", gpus=5)])
    [fenceless] = SlurmBackend(tmp_path, profile=_profile(tmp_path, exclude=())).render([_spec()])
    assert "exclude" not in fenceless["submitit"]["slurm_additional_parameters"]


def test_a_preflight_runs_here_under_the_launchers_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    repo = _repo(tmp_path)
    spec = JobSpec("p", "preflight", 0, ("bash", "-c", "echo $MARK; exit 4", "{artifact_root}"),
                   backends.PREFLIGHT_RESOURCES, env=("MARK=hello",))
    request = preflight_request(SlurmBackend(repo), spec)
    assert request["where"] == "local" and request["shown"] == ["bash", "-c", "echo $MARK; exit 4", str(tmp_path)]
    assert request["argv"][:2] == ["bash", "-c"] and request["cwd"] == str(repo)
    said: list[str] = []
    assert run_preflight(SlurmBackend(repo), request, echo=said.append) == 4 and said == ["hello"]
    assert control.PREFLIGHT_WHERE == ("local", "none")
    assert request["argv"][2] == env_script(launcher_setup(repo), spec.env)


def test_a_preflight_on_a_profile_runs_under_the_profile_s_setup_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    repo = _repo(tmp_path)
    spec = JobSpec("p", "preflight", 0, ("bash", "-c", "echo $A"), backends.PREFLIGHT_RESOURCES)
    backend = SlurmBackend(repo, profile=_profile(tmp_path))
    request = preflight_request(backend, spec)
    assert request["argv"][2] == env_script(["export A=1"], ()) == env_script(backend.setup_lines(), ())
    assert request["argv"][2] != env_script(launcher_setup(repo), ())
    said: list[str] = []
    assert run_preflight(backend, request, echo=said.append) == 0 and said == ["1"]
    assert LocalBackend(repo).setup_lines() == launcher_setup(repo) == SlurmBackend(repo).setup_lines()


def test_each_backend_fills_the_roots_as_its_jobs_see_them(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/weights")
    spec = JobSpec("j", "job", 0, ("python", "x.py", "--base", "{model_root}/Qwen3-8B", "--in",
                                   "{artifact_root}/training/r", "${artifact_root}", "{member}"),
                   Resources("cpu", cpus=1, mem_gb=1, timeout_min=1), env=("OUT={artifact_root}/o",))
    here = LocalBackend(tmp_path).prepare(spec)
    assert here.argv[3:] == ("/weights/Qwen3-8B", "--in", f"{tmp_path}/artifacts/training/r", "${artifact_root}",
                             "{member}")
    assert here.env == (f"OUT={tmp_path}/artifacts/o",)
    # A SLURM job reads this filesystem; a CPU job still gets its QOS. A spec with no root is left as it is.
    slurm = SlurmBackend(_repo(tmp_path)).prepare(spec)
    assert slurm.argv == here.argv and slurm.resources.qos == backends.CPU_QOS
    plain = _spec()
    assert LocalBackend(tmp_path).prepare(plain) is plain
    monkeypatch.delenv("RULEFORGE_MODEL_ROOT")
    with pytest.raises(LaunchError, match=r"{model_root} is \$RULEFORGE_MODEL_ROOT, which is unset"):
        LocalBackend(tmp_path).prepare(spec)


def test_the_model_root_is_read_from_its_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/weights/")
    spec = JobSpec("j", "job", 0, ("python", "x.py", "--base", "{model_root}Qwen3-8B"),
                   Resources("cpu", cpus=1, mem_gb=1, timeout_min=1))
    assert LocalBackend(tmp_path).prepare(spec).argv[3] == "/weights/Qwen3-8B"


def test_a_training_run_on_slurm_is_one_task_per_gpu_with_its_share_of_the_cpus(tmp_path):
    spec = JobSpec("run", "job", 0, ("python", "train_grpo.py"), Resources("gpu", cpus=16, gpus=2, mem_gb=240,
                                                                             timeout_min=60, qos="compact"))
    [request] = SlurmBackend(_repo(tmp_path), {"rank_per_gpu": True}).render([spec])
    assert request["submitit"]["tasks_per_node"] == 2 and request["submitit"]["cpus_per_task"] == 8
    assert request["submitit"]["gpus_per_node"] == 2 and request["submitit"]["mem_gb"] == 240
    [single] = SlurmBackend(_repo(tmp_path), {"rank_per_gpu": True}).render([replace(
        spec, resources=replace(spec.resources, gpus=1))])
    [plain] = SlurmBackend(_repo(tmp_path)).render([spec])
    assert "tasks_per_node" not in single["submitit"] and "tasks_per_node" not in plain["submitit"]
    assert plain["submitit"]["cpus_per_task"] == 16


def test_the_qos_another_qos_may_preempt_and_each_qos_s_wall_come_from_sacctmgr(tmp_path):
    run = Completed({"sacctmgr": [(0, "normal||2-00:00:00\njit|normal|\ncards|normal,low|x\n"
                                      "compact||1-00:00:00\nodd\n", ""),
                                  (1, "", "sacctmgr: error: slurmdbd unreachable"),
                                  OSError("no sacctmgr")]})
    backend = SlurmBackend(_repo(tmp_path), run=run)
    limits = backend.qos_limits()
    assert limits.preempted == {"normal", "low"} and limits.max_wall_min == {"normal": 2880, "compact": 1440}
    assert run.calls[0] == ["sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,MaxWall"]
    assert backend.qos_limits() is None and backend.qos_limits() is None


@pytest.mark.parametrize("text, minutes", [("1-00:00:00", 1440), ("12:30:59", 750), ("2-01:02:03", 2942),
                                           ("", None), ("UNLIMITED", None), ("1:00", None), ("x-01:00:00", None)])
def test_a_wall_is_read_as_sacctmgr_prints_it(text, minutes):
    assert wall_minutes(text) == minutes
