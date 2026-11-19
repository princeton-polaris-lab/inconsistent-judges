from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

import submitit

from ruleforge.core.errors import TracebackError
from ruleforge.launch.environment import (LAUNCHER_CONFIG, MODEL_ROOT_ENV, artifact_relative, env_script, export_lines,
                                          launcher_setup)
from ruleforge.launch.jobs import (
    TIMEOUT_REQUEUES,
    JobSpec,
    Resources,
    fill,
    submitit_parameters,
)
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.records import (
    FAILED, PENDING, RUNNING, STOPPED, SUBMITTING, SUCCEEDED, UNKNOWN, LaunchRecord,
    run_states,
)
from ruleforge.env import artifact_root, env_value

if TYPE_CHECKING:
    from ruleforge.launch.profile import Profile

Runner = Callable[..., subprocess.CompletedProcess]

#: Placeholders `prepare` fills in a job's command and exports.
ROOTS = ("artifact_root", "model_root")
_ROOT = re.compile(r"(?<!\$)\{(" + "|".join(ROOTS) + r")\}")

#: A CPU job's QOS when none is asked for: `compact` requires a GPU.
CPU_QOS = "normal"
DEPENDENCY = "afterok:{}"
SLURM_STATES = {
    **dict.fromkeys(("PENDING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESV_DEL_HOLD", "CONFIGURING"), PENDING),
    **dict.fromkeys(("RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT", "SIGNALING", "RESIZING", "STOPPED"), RUNNING),
    "COMPLETED": SUCCEEDED,
    "CANCELLED": STOPPED,
    **dict.fromkeys(("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED",
                     "REVOKED", "SPECIAL_EXIT"), FAILED),
}
#: States SLURM requeues a `--requeue` job after; pending for a resumable job.
REQUEUED_WHEN_RESUMABLE = ("PREEMPTED", "NODE_FAIL")
_ARRAY_ROW = re.compile(r"^(\d+)_\[([^\]]+)\]$")


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class Backend:
    """What every backend does alike."""

    name = ""
    #: Whether `submit` returns only once its runs have ended.
    foreground = False

    def __init__(
        self, repo: Path | str, options: Optional[Mapping[str, Any]] = None, *, run: Runner = subprocess.run,
        popen: Callable[..., Any] = subprocess.Popen, executor: Optional[Callable[..., Any]] = None,
        echo: Callable[[str], None] = _stderr,
    ) -> None:
        self.repo = Path(repo)
        self.options = dict(options or {})
        self.run = run
        self.popen = popen
        self.executor = executor or slurm_executor
        self.echo = echo

    def prepare(self, spec: JobSpec) -> JobSpec:
        """Return the spec with the artifact and weights roots filled into its command and exports."""
        used = sorted({match.group(1) for item in (*spec.argv, *spec.env) for match in _ROOT.finditer(item)})
        if not used:
            return spec
        values = {name: self.root(name) for name in used}
        return replace(spec, argv=tuple(fill(item, values) for item in spec.argv),
                       env=tuple(fill(item, values) for item in spec.env))

    def root(self, name: str) -> str:
        """Return a root of `ROOTS` as a job sees it (this machine's)."""
        if name == "artifact_root":
            return str(artifact_root())
        model_root = env_value("MODEL_ROOT")
        if model_root is None:
            raise LaunchError(f"{{model_root}} is ${MODEL_ROOT_ENV}, which is unset: "
                              "export the weights root")
        return model_root

    def setup_lines(self) -> list[str]:
        """Return the lines every job runs before its command."""
        return launcher_setup(self.repo)

    def render(self, specs: Sequence[JobSpec]) -> list[dict[str, Any]]:
        """Return each job's request: what the record keeps and `submit` sends."""
        raise NotImplementedError

    def submit(self, record: LaunchRecord) -> None:
        raise NotImplementedError

    def states(self, record: LaunchRecord) -> dict[str, dict[str, Any]]:
        return run_states(record, self.found(record) if record.runs else {})

    def found(self, record: LaunchRecord) -> dict[str, dict[str, Any]]:
        """Return `{job id: {"state": ...}}` for the record's runs the backend can report on."""
        raise NotImplementedError

    def pull(self, record: LaunchRecord, path: str) -> None:
        """Nothing to copy: the jobs wrote the shared filesystem."""

    def stop(self, record: LaunchRecord, run: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        return (artifact_root() / artifact_relative(path)).exists()


def environment_bin() -> str:
    """Return this environment's `bin` directory, whose `python` a job runs."""
    return str(Path(sys.executable).parent)


def with_bin(bin_dir: str, environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return `environ` (default this process's) with `bin_dir` first on `PATH`."""
    environ = dict(os.environ if environ is None else environ)
    return {**environ, "PATH": os.pathsep.join(part for part in (bin_dir, environ.get("PATH")) if part)}


def stream(popen: Callable[..., Any], argv: Sequence[str], *, cwd: Path | str, echo: Callable[[str], None],
           log: Any = None, merge_stderr: bool = False, env: Optional[Mapping[str, str]] = None) -> int:
    """Run `argv`, handing each output line to `echo` (and `log`) as it comes, and return its exit status; the child is
    killed and reaped if the launch fails while reading it."""
    process = popen(list(argv), cwd=str(cwd), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT if merge_stderr else None, text=True, errors="replace", bufsize=1,
                    env=None if env is None else dict(env))
    try:
        for line in process.stdout:
            echo(line.rstrip("\n"))
            if log is not None:
                log.write(line)
        return process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise


class CommandFailed(RuntimeError, TracebackError):
    """A job's command exited non-zero: the SLURM job fails with it."""


def ignore_warning_signal() -> None:
    """Ignore the warning signal SLURM sends before the wall, so the job's own handler decides what happens."""
    signal.signal(signal.SIGUSR2, signal.SIG_IGN)


class Command:
    """What a SLURM job runs: `argv` from `cwd` with `bin_dir` first on `PATH`; a non-zero exit fails the job."""

    def __init__(self, argv: Sequence[str], cwd: Path | str, bin_dir: str) -> None:
        self.argv = [str(part) for part in argv]
        self.cwd = str(cwd)
        self.bin_dir = bin_dir

    def __call__(self) -> int:
        code = subprocess.run(self.argv, cwd=self.cwd, env=with_bin(self.bin_dir), check=False,
                              preexec_fn=ignore_warning_signal).returncode
        if code != 0:
            raise CommandFailed(f"{shlex.join(self.argv)} exited with status {code}")
        return code


class ResumableCommand(Command, submitit.helpers.Checkpointable):
    """A command submitit requeues with the same arguments when the job reaches its wall."""


def _with_dependency(job: Mapping[str, Any], submitted: Mapping[str, str]) -> Mapping[str, Any]:
    if not job.get("depends_on"):
        return job
    extra = job["request"]["submitit"]["slurm_additional_parameters"]
    dependency = DEPENDENCY.format(":".join(submitted[name] for name in job["depends_on"]))
    submitit_request = {**job["request"]["submitit"], "slurm_additional_parameters": {**extra, "dependency": dependency}}
    return {**job, "request": {**job["request"], "submitit": submitit_request}}


def slurm_executor(folder: Path, max_num_timeout: int) -> Any:
    return submitit.AutoExecutor(folder=str(folder), cluster="slurm", slurm_max_num_timeout=max_num_timeout)


def slurm_rows(text: str) -> dict[str, str]:
    """Return `sacct`'s or `squeue`'s `id|state` lines as `{job id: state}`, a pending array row spread over its tasks.
    """
    rows = {}
    for line in text.splitlines():
        job, separator, state = line.strip().partition("|")
        if not separator:
            continue
        match = _ARRAY_ROW.match(job)
        if match is None:
            rows[job] = state
            continue
        for part in match.group(2).split("%")[0].split(","):
            first, _, last = part.partition("-")
            for index in range(int(first), int(last or first) + 1):
                rows[f"{match.group(1)}_{index}"] = state
    return rows


UNREAD_SUBMISSION = "its submission raised after sbatch was asked: sbatch may have queued it, and no job id was read"


def slurm_state(raw: str, *, resumable: bool = False) -> str:
    """Return a run's state for a raw SLURM state; a resumable job's requeue states are `pending`."""
    word = (raw.split() or [""])[0].rstrip("+")
    return PENDING if resumable and word in REQUEUED_WHEN_RESUMABLE else SLURM_STATES.get(word, UNKNOWN)


@dataclass(frozen=True)
class QosLimits:
    """What `sacctmgr` says of the QOS: which another may preempt, and each one's longest wall in minutes."""

    preempted: frozenset[str]
    max_wall_min: Mapping[str, int]


def wall_minutes(text: str) -> Optional[int]:
    """Return a SLURM wall (`[D-]HH:MM:SS`) in whole minutes, or None for anything else."""
    days, _, clock = text.strip().rpartition("-")
    parts = clock.split(":")
    if len(parts) != 3 or not all(part.isdigit() for part in parts) or not (days == "" or days.isdigit()):
        return None
    hours, minutes, _ = map(int, parts)
    return (int(days or 0) * 24 + hours) * 60 + minutes


class SlurmBackend(Backend):
    """SLURM, with the cluster and setup lines of the launcher config or of `profile` when given."""

    name = "slurm"

    def __init__(self, repo: Path | str, options: Optional[Mapping[str, Any]] = None, *,
                 profile: Optional["Profile"] = None, **kwargs: Any) -> None:
        super().__init__(repo, options, **kwargs)
        self.profile = profile

    def site(self) -> dict[str, Any]:
        """Return the cluster: partition, account, the warning signal's lead and the excluded nodes."""
        if self.profile is not None:
            return self.profile.site()
        from omegaconf import OmegaConf

        config = OmegaConf.load(self.repo / LAUNCHER_CONFIG)
        return {"partition": config.partition, "account": config.get("account"),
                "signal_delay_s": config.get("signal_delay_s")}

    def prepare(self, spec: JobSpec) -> JobSpec:
        spec = super().prepare(spec)
        # A profile's CPU job takes the profile's CPU QOS, or with none the partition's default.
        if spec.resources.gpus or spec.resources.qos or self.profile is not None:
            return spec
        return replace(spec, resources=replace(spec.resources, qos=CPU_QOS))

    def tasks(self, spec: JobSpec) -> dict[str, int]:
        """Return a GPU job's tasks under the `rank_per_gpu` option: one per GPU, the CPUs split among them."""
        ranks = spec.resources.gpus if self.options.get("rank_per_gpu") else 1
        return {} if ranks < 2 else {"tasks_per_node": ranks, "cpus_per_task": spec.resources.cpus // ranks}

    def setup_lines(self) -> list[str]:
        """Return the lines every job runs before its command: the launcher config's or the profile's."""
        return list(self.profile.setup) if self.profile is not None else super().setup_lines()

    def render(self, specs: Sequence[JobSpec]) -> list[dict[str, Any]]:
        site, setup = self.site(), self.setup_lines()
        delay = {} if site["signal_delay_s"] is None else {"slurm_signal_delay_s": site["signal_delay_s"]}
        known = {spec.job_id for spec in specs}
        requests = []
        for spec in specs:
            unknown = sorted(set(spec.depends_on) - known)
            if unknown:
                raise LaunchError(f"{spec.job_id} depends on {', '.join(unknown)}, which this launch does not submit")
            if self.profile is not None:
                self.profile.check_gpus(spec.job_id, spec.resources.gpus)
            submitit_request = {**submitit_parameters(spec, site), **self.tasks(spec),
                                "slurm_setup": [*setup, *export_lines(spec.env)], **delay}
            if spec.depends_on:
                submitit_request["slurm_additional_parameters"] = {
                    **submitit_request["slurm_additional_parameters"],
                    "dependency": DEPENDENCY.format(":".join(f"{{{job}}}" for job in spec.depends_on))}
            requests.append({
                "submitit": submitit_request,
                "max_num_timeout": TIMEOUT_REQUEUES if spec.resources.resumable else 0,
                "argv": list(spec.argv),
                "cwd": str(self.repo / spec.workdir),
                "bin": environment_bin(),
            })
        return requests

    def submit(self, record: LaunchRecord) -> None:
        """Submit the jobs in waves, each once the jobs it depends on are submitted; jobs sharing their parameters go out
        as one array."""
        pending = list(record.jobs)
        while pending:
            submitted = {run["job_id"]: run["slurm_job"] for run in record.runs if run["slurm_job"] is not None}
            wave = [job for job in pending if all(name in submitted for name in job.get("depends_on") or ())]
            if not wave:
                raise LaunchError(f"jobs {[job['job_id'] for job in pending]} depend on one another in a cycle: none "
                                  "can be submitted first")
            self._submit_wave(record, [_with_dependency(job, submitted) for job in wave])
            pending = [job for job in pending if job not in wave]

    def _submit_wave(self, record: LaunchRecord, wave: Sequence[Mapping[str, Any]]) -> None:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for job in wave:
            request = job["request"]
            key = json.dumps([{**request["submitit"], "name": None}, request["max_num_timeout"]], sort_keys=True)
            groups.setdefault(key, []).append(job)
        for jobs in groups.values():
            record.check_not_stopped()
            first = jobs[0]["request"]
            executor = self.executor(record.directory / "submitit", first["max_num_timeout"])
            name = jobs[0]["job_id"] if len(jobs) == 1 else record.launch_id
            executor.update_parameters(**{**first["submitit"], "name": name})
            commands = [(ResumableCommand if job["resources"]["resumable"] else Command)(
                job["request"]["argv"], job["request"]["cwd"], job["request"]["bin"]) for job in jobs]
            # recorded before sbatch is asked: a submission that raises after sbatch answered may still have queued the job
            record.add_runs([{"job_id": job["job_id"], "slurm_job": None} for job in jobs])
            if len(commands) == 1:
                handles = [executor.submit(commands[0])]
            else:
                with executor.batch():
                    handles = [executor.submit(command) for command in commands]
            record.update_runs({job["job_id"]: {"slurm_job": str(handle.job_id)}
                                for job, handle in zip(jobs, handles)})
            for job, handle in zip(jobs, handles):
                self.echo(f"== {job['job_id']}: SLURM job {handle.job_id}")

    def _ask(self, argv: Sequence[str]) -> Optional[str]:
        try:
            result = self.run(list(argv), capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout if result.returncode == 0 else None

    def _query(self, argv: Sequence[str]) -> dict[str, str]:
        return slurm_rows(self._ask(argv) or "")

    def qos_limits(self) -> Optional[QosLimits]:
        """Return what `sacctmgr` says of the QOS, or None when it cannot say."""
        text = self._ask(["sacctmgr", "-nP", "show", "qos", "format=Name,Preempt,MaxWall"])
        if text is None:
            return None
        preempted: set[str] = set()
        walls: dict[str, int] = {}
        for line in text.splitlines():
            name, _, rest = line.partition("|")
            preempt, _, wall = rest.partition("|")
            preempted.update(filter(None, preempt.split(",")))
            minutes = wall_minutes(wall)
            if name and minutes is not None:
                walls[name] = minutes
        return QosLimits(frozenset(preempted), walls)

    def found(self, record: LaunchRecord) -> dict[str, dict[str, Any]]:
        """Return each run's state from sacct, else squeue; a run whose job id was never read is `submitting` while its
        launch is and `unknown` after."""
        unread = {run["job_id"]: {"state": SUBMITTING} if record.data.get("state") == "submitting" else
                  {"state": UNKNOWN, "error": UNREAD_SUBMISSION} for run in record.runs if run["slurm_job"] is None}
        ids = [run["slurm_job"] for run in record.runs if run["slurm_job"] is not None]
        if not ids:
            return unread
        raw = self._query(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(ids)])
        missing = [job for job in ids if job not in raw]
        if missing:
            raw.update(self._query(["squeue", "-h", "-r", "-o", "%i|%T", "-j", ",".join(missing)]))
        resumable = {job["job_id"]: bool(job["resources"]["resumable"]) for job in record.jobs}
        return {**unread, **{run["job_id"]: {"state": slurm_state(raw[run["slurm_job"]],
                                                                  resumable=resumable[run["job_id"]]),
                                             "slurm_state": raw[run["slurm_job"]]}
                             for run in record.runs if run["slurm_job"] in raw}}

    def stop(self, record: LaunchRecord, run: Mapping[str, Any]) -> dict[str, Any]:
        if run["slurm_job"] is None:
            raise LaunchError(f"{run['job_id']}: {UNREAD_SUBMISSION}; look by hand (squeue --me: the job named "
                              f"{run['job_id']} or {record.launch_id}), scancel what it queued, then pass "
                              f"--ignore-launch {record.launch_id}")
        result = self.run(["scancel", run["slurm_job"]], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise LaunchError(f"scancel {run['slurm_job']} failed ({result.returncode}): {result.stderr.strip()}")
        return {"slurm_job": run["slurm_job"], "requested": True}


class LocalBackend(Backend):
    """This machine: each job in the foreground under the launcher config's setup lines, or `setup` when given."""

    name = "local"
    foreground = True

    def __init__(self, repo: Path | str, options: Optional[Mapping[str, Any]] = None, *,
                 setup: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        super().__init__(repo, options, **kwargs)
        self.setup = None if setup is None else list(setup)

    def setup_lines(self) -> list[str]:
        return super().setup_lines() if self.setup is None else list(self.setup)

    def render(self, specs: Sequence[JobSpec]) -> list[dict[str, Any]]:
        setup = self.setup_lines()
        return [{"argv": ["bash", "-c", env_script(setup, spec.env), "ruleforge-launch", *spec.argv],
                 "cwd": str(self.repo / spec.workdir), "bin": environment_bin()} for spec in specs]

    def submit(self, record: LaunchRecord) -> None:
        """Run each job in turn in the foreground, recording its run before the child starts and its exit status once it
        ends."""
        for job in record.jobs:
            record.check_not_stopped()
            log = record.directory / "logs" / f"{job['job_id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            record.add_run({"job_id": job["job_id"], "log": str(log)})
            child = _Child(self.popen)
            try:
                with log.open("w", encoding="utf-8") as handle:
                    code = stream(child, job["request"]["argv"], cwd=job["request"]["cwd"], echo=self.echo,
                                  log=handle, merge_stderr=True, env=with_bin(job["request"]["bin"]))
            except BaseException as error:
                record.update_run(job["job_id"], error=f"{type(error).__name__}: {error}",
                                  **({} if child.started else {"exit_code": None}))
                raise
            record.update_run(job["job_id"], exit_code=code)
            self.echo(f"== {job['job_id']}: exit status {code} (log {log})")

    def found(self, record: LaunchRecord) -> dict[str, dict[str, Any]]:
        """Return a run's state from its exit status; one with none is `submitting` while its launch is and `unknown`
        after."""
        return {run["job_id"]: (
            {"state": SUCCEEDED if run["exit_code"] == 0 else FAILED, "exit_code": run["exit_code"]} if "exit_code" in run
            else {"state": SUBMITTING if record.data.get("state") == "submitting" else UNKNOWN,
                  **({"error": run["error"]} if run.get("error") else {})}) for run in record.runs}

    def stop(self, record: LaunchRecord, run: Mapping[str, Any]) -> dict[str, Any]:
        """Nothing to stop: a local job ran in its launch's foreground."""
        raise LaunchError(f"{run['job_id']} ran on this machine in its launch's foreground: no launcher can stop it "
                          "now; look for it by hand (pgrep), stop it, then pass --ignore-launch "
                          f"{record.launch_id}")


class _Child:

    def __init__(self, popen: Callable[..., Any]) -> None:
        self.popen = popen
        self.started = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        process = self.popen(*args, **kwargs)
        self.started = True
        return process


BACKENDS: dict[str, type[Backend]] = {"slurm": SlurmBackend, "local": LocalBackend}


def backend_for(name: str, repo: Path | str, options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> Backend:
    """Return the backend `name` built with `options`."""
    if name not in BACKENDS:
        raise LaunchError(f"backend {name!r}: one of {', '.join(sorted(BACKENDS))}")
    return BACKENDS[name](repo, options, **kwargs)


PREFLIGHT_RESOURCES = Resources("cpu", cpus=4, mem_gb=24, timeout_min=30)


def preflight_request(backend: Backend, spec: JobSpec) -> dict[str, Any]:
    """Return a preflight's request: run here, under the setup lines the launch's jobs run under."""
    here = LocalBackend(backend.repo, setup=backend.setup_lines())
    spec = here.prepare(spec)
    [request] = here.render([spec])
    return {**request, "where": "local", "shown": list(spec.argv)}


def run_preflight(backend: Backend, request: Mapping[str, Any], *, echo: Callable[[str], None]) -> int:
    """Run one preflight and return its exit status."""
    return stream(backend.popen, request["argv"], cwd=request["cwd"], echo=echo, merge_stderr=True,
                  env=with_bin(request["bin"]))
