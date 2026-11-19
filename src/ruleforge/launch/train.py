from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.grpo.run_files import PROVENANCE_FNAME
from ruleforge.launch import records
from ruleforge.launch.backends import PREFLIGHT_RESOURCES, QosLimits
from ruleforge.launch.campaign import place
from ruleforge.launch.environment import artifact_relative
from ruleforge.launch.guards import (
    check_exports,
    check_name,
    check_overrides,
    check_spelled,
    compose,
    parse_override,
    refuse_linked_way,
)
from ruleforge.launch.environment import LaunchError
from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root
from ruleforge.core.provenance import SNAPSHOT_DIRNAME

#: Where the run's program runs: the checkout's top.
WORKDIR = "."
PROGRAM = ("python", "-m", "ruleforge.launch.programs.train_grpo")
DEFAULT_CONFIG = "training/grpo/qwen3_8b"
#: What the verb sets on the run itself, so an argument may not.
RESERVED = ("run_name", "output_root", "output_dir", "resume_from_checkpoint", "preflight_only", "hydra.run.dir")
RESUME_FRESH, RESUME_LATEST = "none", "latest"
#: A rank's CPUs and host memory (GB) when not given: each rank loads the fp32 master weights on the host first.
CPUS_PER_GPU, MEM_GB_PER_GPU = 8, 120
PREFLIGHT = replace(PREFLIGHT_RESOURCES, mem_gb=32, timeout_min=60)
#: What the trainer writes below the run's directory, each a subtree of the run's alone.
COMPLETIONS_DIRNAME = "completions"
SAVE_PREFIXES = ("checkpoint-", "tmp-checkpoint-")
#: The variable naming a rank when Hydra composes its directory (RANK is exported only after).
RANK_VARIABLE = "SLURM_PROCID"
HYDRA_FILES = (".hydra/config.yaml", ".hydra/overrides.yaml")
STOP_REFUSED = records.WAIT_TIMED_OUT
PER_STEP_OPTIONS = {"completions": "--completions-per-step", "replay_rows": "--replay-rows-per-step"}
_ASK_ERRORS = (LaunchError, OSError)
_ABSENT = object()


@dataclass(frozen=True)
class TrainRun:
    """One training launch's run: its name, output directory, how it resumes, its ranks, its save and step counts, what
    one step trains on and its program's Hydra arguments."""

    name: str
    output_dir: str
    resume: str
    resume_value: str
    ranks: int
    save_steps: int
    max_steps: int
    per_step: Mapping[str, int]
    arguments: tuple[str, ...]

    @property
    def settings(self) -> tuple[str, ...]:
        """Return what the verb sets on every command of the run."""
        return f"run_name={self.name}", f"resume_from_checkpoint={self.resume_value}"

    def hydra_dir(self, launch_id: str, name: str) -> str:
        """Return a directory of the launch's under the run's Hydra directory."""
        return f"{self.output_dir}/hydra/{launch_id}/{name}"

    def command(self, launch_id: str) -> tuple[str, ...]:
        """Return the run's command: one program per SLURM task."""
        return (*PROGRAM, *self.arguments, *self.settings,
                f"hydra.run.dir=${{output_dir}}/hydra/{launch_id}/rank${{oc.env:{RANK_VARIABLE},0}}")

    def preflight(self, launch_id: str) -> tuple[str, ...]:
        """Return the run's preflight command, in its own Hydra directory."""
        return (*PROGRAM, *self.arguments, *self.settings, "preflight_only=true",
                f"hydra.run.dir=${{output_dir}}/hydra/{launch_id}/preflight")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def resume_value(resume: str) -> str:
    """Return `--resume` as the run's `resume_from_checkpoint`: null, latest, or a checkpoint path."""
    if resume == RESUME_FRESH:
        return "null"
    if resume == RESUME_LATEST:
        return RESUME_LATEST
    return place(resume, what="--resume").named


def per_step(cfg: DictConfig, ranks: int) -> dict[str, int]:
    """Return what one optimizer step trains on over `ranks`: completions, prompts and replay rows."""
    grpo, anchor = cfg.grpo, cfg.anchor
    completions = ranks * int(grpo.per_device_train_batch_size) * int(grpo.gradient_accumulation_steps)
    rows = ranks * int(anchor.per_device_batch_size) * int(anchor.anchor_accum) if anchor.enabled else 0
    return {"completions": completions, "prompts": completions // int(grpo.num_generations), "replay_rows": rows}


def check_per_step(cfg: DictConfig, found: Mapping[str, int], expected: Mapping[str, Optional[int]],
                   ranks: int) -> None:
    """Refuse a run whose step trains on other than `expected`."""
    per_rank = {"completions": ("grpo.gradient_accumulation_steps", int(cfg.grpo.per_device_train_batch_size)),
                "replay_rows": ("anchor.anchor_accum", int(cfg.anchor.per_device_batch_size))}
    wrong = []
    for key, value in expected.items():
        if value is None or found[key] == value:
            continue
        knob, each = per_rank[key]
        fix = (f"{knob}={value // (ranks * each)}" if value % (ranks * each) == 0
               else f"no {knob} does on {ranks} rank(s) at {each} per device")
        wrong.append(f"{found[key]} {key.replace('_', ' ')}, not {value} ({PER_STEP_OPTIONS[key]}; {fix})")
    if wrong:
        raise LaunchError(f"on {ranks} rank(s) one step would train on " + "; ".join(wrong) + ": the ranks and "
                          "the accumulations go together (grpo.gradient_accumulation_steps and anchor.anchor_accum after --)")


def plan(repo: Path, arguments: Sequence[str], *, run_name: str, resume: str, ranks: int,
         expected_per_step: Optional[Mapping[str, Optional[int]]] = None,
         env: Sequence[str] = ()) -> tuple[TrainRun, DictConfig]:
    """Return the run and its config, composed from `arguments` and what the verb sets; an export of `env` that the
    config reads is refused."""
    check_name(run_name, what="--run-name")
    if ranks < 1:
        raise LaunchError("a training run takes GPUs: give --gpu-count")
    check_overrides(arguments, reserved=RESERVED)
    value = resume_value(resume)
    cfg = compose(repo, [*arguments, f"run_name={run_name}", f"resume_from_checkpoint={value}"],
                  default_config=DEFAULT_CONFIG, env=env, what="a training job")
    check_exports(cfg, env, what="a training job")
    try:
        output_dir = artifact_relative(str(cfg.output_dir))
        save_steps, max_steps = int(cfg.grpo.save_steps), int(cfg.grpo.max_steps)
        geometry = per_step(cfg, ranks)
    except (OmegaConfBaseException, ValueError, ArtifactRootError, TypeError, ZeroDivisionError) as error:
        raise LaunchError(f"the run's config gives no output directory below the artifact root, or no whole save, "
                          f"step and batch counts (output_dir, grpo.save_steps, grpo.max_steps, the per-device "
                          f"batches and accumulations): {error}") from None
    check_per_step(cfg, geometry, expected_per_step or {}, ranks)
    return TrainRun(run_name, output_dir, resume, value, ranks, save_steps, max_steps, geometry,
                    tuple(arguments)), cfg


def requeued(qos: Optional[str], limits: Optional[QosLimits], resume: str) -> bool:
    """Return whether a SLURM run is requeued: exactly when it resumes `latest`; any other is refused under a QOS another
    may preempt."""
    if resume == RESUME_LATEST:
        return True
    why = ("no --qos is named (the account's default may be preempted)" if qos is None
           else f"sacctmgr cannot say whether QOS {qos} may be preempted" if limits is None
           else f"QOS {qos} may be preempted (sacctmgr: another QOS's Preempt names it)" if qos in limits.preempted
           else None)
    if why is None:
        return False
    raise LaunchError(f"{why}: a preempted run that does not resume would end there. Launch with --resume latest "
                      "(requeued, it resumes from its newest checkpoint), or under a QOS nothing preempts")


def check_wall(qos: Optional[str], limits: Optional[QosLimits], timeout_min: int) -> None:
    """Refuse a SLURM run whose limit is past its QOS's `MaxWall`."""
    wall = None if qos is None or limits is None else limits.max_wall_min.get(qos)
    if wall is not None and timeout_min > wall:
        raise LaunchError(f"--timeout-min {timeout_min} is past QOS {qos}'s MaxWall ({wall} minutes), under which "
                          f"the job would pend for good: give --timeout-min {wall} or less (a run resuming latest "
                          "is requeued after its wall)")


def training_resources(settings: Mapping[str, Any], ranks: int) -> dict[str, Any]:
    """Return the CPUs and memory a run asks for: per-rank defaults unless given."""
    cpus = CPUS_PER_GPU * ranks if settings.get("cpus") is None else settings["cpus"]
    if cpus % ranks:
        raise LaunchError(f"--cpus {cpus} over {ranks} ranks: each SLURM task takes an equal share, so give a "
                          f"multiple of {ranks}")
    return {"cpus": cpus, "mem_gb": MEM_GB_PER_GPU * ranks if settings.get("mem_gb") is None else settings["mem_gb"]}


def expectations(items: Sequence[str]) -> dict[str, Any]:
    """Return `--expect KEY=VALUE` items as `{key: value}`, each value typed as Hydra types an override."""
    expected: dict[str, Any] = {}
    for item in items:
        override = parse_override(item)
        key = override.key_or_group
        plain = not (override.is_sweep_override() or override.is_add() or override.is_force_add()
                     or override.is_delete() or override.package is not None)
        if not plain or not key or "/" in key:
            raise LaunchError(f"--expect {item!r}: KEY=VALUE, a config key and the value it must hold")
        if key in expected:
            raise LaunchError(f"--expect names {key} twice")
        expected[key] = override.value()
    return expected


def config_value(cfg: DictConfig, key: str) -> Any:
    """Return a key's value, resolved where this machine can resolve it, else as written; `_ABSENT` when absent."""
    try:
        value = OmegaConf.select(cfg, key, default=_ABSENT)
    except OmegaConfBaseException:
        value = OmegaConf.to_container(cfg, resolve=False)
        for part in key.split("."):
            value = value.get(part, _ABSENT) if isinstance(value, dict) else _ABSENT
    return OmegaConf.to_container(value, resolve=False) if isinstance(value, (DictConfig, ListConfig)) else value


def same_value(found: Any, expected: Any) -> bool:
    """Return whether a config's value is the one expected (a boolean only to a boolean, a whole number to a float).
    """
    if isinstance(found, bool) or isinstance(expected, bool):
        return isinstance(found, bool) and isinstance(expected, bool) and found == expected
    return found == expected


def check_expectations(cfg: DictConfig, expected: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """Refuse a config whose value of an expected key is not the one expected; return the values found."""
    found = {key: config_value(cfg, key) for key in expected}
    wrong = [f"{key} is {'absent' if found[key] is _ABSENT else repr(found[key])}, not {value!r}"
             for key, value in expected.items() if found[key] is _ABSENT or not same_value(found[key], value)]
    if wrong:
        raise LaunchError(f"--expect: in {where}, " + "; ".join(wrong))
    return found


def read_preflight(directory: Path) -> dict[str, Any]:
    """Return what a preflight composed: its `.hydra/config.yaml` and overrides."""
    missing = [name for name in HYDRA_FILES if not (directory / name).is_file()]
    if missing:
        raise LaunchError(f"the preflight left no {', '.join(missing)} in {directory}")
    config, overrides = (OmegaConf.to_container(OmegaConf.load(directory / name), resolve=False)
                         for name in HYDRA_FILES)
    return {"config": config, "overrides": overrides}


def preflight_config(hydra_dir: str) -> dict[str, Any]:
    """Return what the preflight composed, from its Hydra directory below the artifact root."""
    return read_preflight(artifact_root() / hydra_dir)


def check_root(run: TrainRun, launch_id: str, written: str) -> None:
    """Refuse a launch of `run` whose directory, or any subtree the run writes, holds a symlink; nothing is written."""
    from ruleforge.evaluate.api import refuse_protected

    directory = artifact_root() / run.output_dir
    refuse_protected(directory, "the run's directory")
    check_spelled(written, directory, what="the run's directory")
    refuse_protected(directory / "hydra", "the run's Hydra directory")
    refuse_protected(directory / "hydra" / launch_id, "the launch's Hydra directory", below=True)
    for entry in sorted(directory.iterdir()) if directory.is_dir() else ():
        if entry.is_symlink():
            raise LaunchError(f"{entry}, in the run's directory, is a symlink: the run writes its files there and a "
                              f"write would follow it to {os.path.realpath(entry)}; replace it with a file or "
                              "directory of its own")
        if entry.name in (COMPLETIONS_DIRNAME, SNAPSHOT_DIRNAME) or entry.name.startswith(SAVE_PREFIXES):
            refuse_linked_way(directory, entry, what="a directory the run writes", below=True)


def started(backend: Any, run: TrainRun) -> bool:
    """Return whether a run has started in the run's directory (its code snapshot is there)."""
    return bool(backend.exists(f"{run.output_dir}/{SNAPSHOT_DIRNAME}"))


def saved_before(backend: Any, run: TrainRun) -> int:
    """Return the newest save the run's directory holds, 0 for none."""
    if run.resume == RESUME_FRESH:
        return 0
    steps = sorted({*range(run.save_steps, run.max_steps, run.save_steps), run.max_steps}, reverse=True)
    return next((step for step in steps if backend.exists(f"{run.output_dir}/checkpoint-{step}/{PROVENANCE_FNAME}")),
                0)


def stop_after_save(
    record: records.LaunchRecord, backend: Any, step: int, *, reason: Optional[str] = None,
    timeout_min: Optional[float] = None, poll_seconds: float = records.POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
    echo: Callable[[str], None] = lambda line: None,
) -> tuple[int, dict[str, Any]]:
    """Stop a training launch's run right after `checkpoint-<step>` has landed; return the exit status (0 stopped, 1 the
    run ended before it, `STOP_REFUSED` when the next save had begun or the wait timed out) and what was done."""
    records.check_writes(record)
    run = record.refresh().data.get("train")
    if run is None:
        raise LaunchError(f"{record.launch_id} is not a training launch (ruleforge jobs launch train): --after-save "
                          "waits for a training run's save")
    save_steps, max_steps, output = run["save_steps"], run["max_steps"], run["output_dir"]
    if step < save_steps or step % save_steps or step >= max_steps:
        raise LaunchError(f"--after-save {step}: a save of {run['name']} before its last (every {save_steps} steps "
                          f"up to {max_steps}, where it ends by itself)")
    before = run.get("saved_before") or 0
    if step <= before:
        raise LaunchError(f"--after-save {step}: {output} held checkpoint-{before} before this launch (the save it "
                          f"resumed from, or an older trajectory's), so a provenance there says nothing of this "
                          f"launch's run: name a save after {before}")
    landed = f"{output}/checkpoint-{step}/{PROVENANCE_FNAME}"
    following = min(step + save_steps, max_steps)  # the trainer also saves at max_steps
    next_save = [f"{output}/{prefix}checkpoint-{following}" for prefix in ("", "tmp-")]
    outcome: dict[str, Any] = {"after_save": step, "landed": landed, "next_save": next_save}
    deadline = clock() + (records.default_wait_minutes(record) if timeout_min is None else timeout_min) * 60
    errors: list[str] = []

    def ask(path: str) -> Optional[bool]:
        try:
            return bool(backend.exists(path))
        except _ASK_ERRORS as error:
            errors.append(f"{type(error).__name__}: {error}")
            return None

    said = None
    while True:
        errors.clear()
        if ask(landed):
            break
        states = backend.states(record.refresh())
        if all(entry["state"] in records.ENDED for entry in states.values()):
            there = ask(landed)
            if there:
                break
            if there is False:
                echo(f"== {run['name']} ended ({records.summary(states)}) before checkpoint-{step} landed: nothing "
                     "to stop")
                return records.WAIT_FAILED, {**outcome, "ended": True, "states": states}
        if clock() >= deadline:
            echo(f"== checkpoint-{step} had not landed by the wait's limit: stop again, or stop now (without "
                 "--after-save)")
            return STOP_REFUSED, {**outcome, "timed_out": True, "states": states, **({"errors": errors} if errors
                                                                                    else {})}
        line = (f"== could not ask whether checkpoint-{step} has landed ({errors[-1]}): asking again" if errors
                else f"== waiting for {run['name']}'s checkpoint-{step} ({records.summary(states)})")
        if line != said:
            echo(line)
            said = line
        sleep(poll_seconds)
    found = {path: ask(path) for path in next_save}
    begun = [path for path, there in found.items() if there]
    if begun:
        echo(f"== late: {', '.join(begun)} exists, so the save after checkpoint-{step} has begun and a stop now "
             "could cut it; nothing was stopped. Stop by hand (ruleforge jobs stop, at once) once it has landed, or "
             f"stop --after-save {following}")
        return STOP_REFUSED, {**outcome, "late": begun}
    unasked = [path for path, there in found.items() if there is None]
    if unasked:
        echo(f"== could not ask whether {', '.join(unasked)} exists ({errors[-1]}), so whether the next save has "
             "begun: nothing was stopped; stop again")
        return STOP_REFUSED, {**outcome, "unasked": unasked, "errors": errors}
    entry = records.stop(record, backend, reason=reason or f"after checkpoint-{step}",
                         details={"after_save": step, "landed": landed})
    found = {path: ask(path) for path in next_save}
    begun_while_stopping = [path for path, there in found.items() if there]
    unasked = [path for path, there in found.items() if there is None]
    if begun_while_stopping:
        echo(f"== the next save began as the run was stopped: {', '.join(begun_while_stopping)} has no provenance, "
             "so a resume skips it (resume_from_checkpoint=latest); move it aside before evaluating the run's saves")
    if unasked:
        echo(f"== the run was stopped, but whether the next save began as it stopped could not be asked "
             f"({errors[-1]}): look for {', '.join(unasked)} before evaluating the run's saves")
    failed = bool(entry.get("stop_errors") or entry.get("submitting"))
    return (records.WAIT_FAILED if failed else 0), {
        **outcome, "stop": entry, **({"late": begun_while_stopping} if begun_while_stopping else {}),
        **({"unasked": unasked} if unasked else {})}
