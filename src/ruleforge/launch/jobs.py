from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from ruleforge.launch.environment import LaunchError
from ruleforge.launch.environment import env_pairs, run_workdir

#: The `Resources.kind` of an action the planner runs itself; no backend renders it.
LOCAL = "local"
#: How many times submitit requeues a resumable job that reached its wall.
TIMEOUT_REQUEUES = 3
MAX_AXES = 2
_AXIS_KEY = re.compile(r"(?:\+\+|\+|~)?[A-Za-z_][\w.@/-]*")
_AXIS_PAIR = re.compile(r"(?<=[\w.@/-])\+")
_AXIS_RANGE = re.compile(r"(-?[0-9]+)\.\.(-?[0-9]+)")


def fill(text: str, values: Mapping[str, str]) -> str:
    """Return `text` with each `{name}` of `values` replaced; every other brace is left as it is."""
    pattern = re.compile(r"(?<!\$)\{(" + "|".join(re.escape(name) for name in values) + r")\}")
    return pattern.sub(lambda match: str(values[match.group(1)]), text)


@dataclass(frozen=True)
class Resources:
    """What a job asks the scheduler for (`kind`: cpu, gpu or local); `resumable` means SLURM may requeue it."""

    kind: str
    cpus: int = 0
    gpus: int = 0
    mem_gb: int = 0
    timeout_min: int = 0
    qos: str | None = None
    resumable: bool = False


@dataclass(frozen=True)
class JobSpec:
    """One job: `argv` run from the repository root's `workdir`, after every job in `depends_on`, with its `env` exports
    and the artifact paths it writes (`outputs`)."""

    job_id: str
    kind: str
    step: int
    argv: tuple[str, ...]
    resources: Resources
    depends_on: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    workdir: str = "."
    env: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        env_pairs(self.env)
        run_workdir(self.workdir)

    def to_json(self) -> dict[str, Any]:
        fields = {name: list(value) if isinstance(value, tuple) else value for name, value in asdict(self).items()}
        env = fields.pop("env")
        return {**fields, "groups": [], "env": env}


def axis_keys(key: str) -> tuple[str, ...]:
    """Return a paired axis's keys (`a+b`: both take each value), else the one key."""
    return tuple(_AXIS_PAIR.split(key))


def parse_axis(text: str) -> tuple[str, tuple[str, ...]]:
    """Return `KEY=VALUES` as its key and values; `A..B` is the whole numbers from A to B."""
    key, separator, spelled = text.partition("=")
    if not separator or not all(_AXIS_KEY.fullmatch(part) for part in axis_keys(key)):
        raise LaunchError(f"--axis {text!r}: expected KEY=VALUES, KEY an override's key (a,b,c or A..B)")
    return key, tuple(expand_values(text, spelled, "--axis", LaunchError))


def expand_values(text: str, spelled: str, flag: str, error: type[Exception]) -> list[str]:
    """Return the values `spelled` lists (`A..B` expanded); an empty, backwards or repeated value is refused with
    `error`."""
    values: list[str] = []
    for item in spelled.split(","):
        bounds = _AXIS_RANGE.fullmatch(item)
        if bounds is None and item:
            values.append(item)
            continue
        if bounds is None or int(bounds.group(1)) > int(bounds.group(2)):
            raise error(f"{flag} {text!r}: {item!r} is not a value (an empty one, or a range that runs backwards)")
        values += [str(number) for number in range(int(bounds.group(1)), int(bounds.group(2)) + 1)]
    repeated = sorted({value for value in values if values.count(value) > 1})
    if repeated:
        raise error(f"{flag} {text!r} gives {', '.join(repeated)} more than once")
    return values


def members(spec: JobSpec, axes: Sequence[str] = ()) -> list[JobSpec]:
    """Return the jobs `spec` fans out into over `axes`: one per point of their product, `KEY=VALUE` appended to its
    command and `{member}` filled with its index."""
    parsed = [parse_axis(axis) for axis in axes]
    keys = [axis_keys(key) for key, _ in parsed]
    every = [part for group in keys for part in group]
    if len(parsed) > MAX_AXES or len(set(every)) != len(every):
        raise LaunchError(f"--axis {', '.join(axes)}: at most {MAX_AXES} axes, each on its own key")
    jobs = []
    for index, point in enumerate(itertools.product(*(values for _, values in parsed))):
        member = {"member": str(index)}
        argv = (*spec.argv, *(f"{key}={value}" for group, value in zip(keys, point) for key in group))
        jobs.append(replace(spec, job_id=f"{spec.job_id}-{index}" if parsed else spec.job_id,
                            argv=tuple(fill(item, member) for item in argv),
                            env=tuple(fill(item, member) for item in spec.env),
                            outputs=tuple(fill(item, member) for item in spec.outputs)))
    shared = sorted({path for job in jobs for path in job.outputs
                     if sum(path in other.outputs for other in jobs) > 1})
    if shared:
        raise LaunchError(f"--axis {', '.join(axes)}: every member would write {', '.join(shared)}: put {{member}} "
                          "in its path")
    return jobs


def _scheduled(spec: JobSpec) -> Resources:
    if spec.resources.kind == LOCAL:
        raise LaunchError(f"{spec.job_id} is a local action, not a scheduler job")
    return spec.resources


def submitit_parameters(spec: JobSpec, slurm: Mapping[str, Any]) -> dict[str, Any]:
    """Return `submitit.AutoExecutor.update_parameters` keywords for a spec on a cluster; only a resumable job may be
    requeued."""
    resources = _scheduled(spec)
    extra = {
        **({"account": slurm["account"]} if slurm.get("account") else {}),
        **({"qos": resources.qos} if resources.qos else {}),
        **({"requeue": True} if resources.resumable else {"no_requeue": True}),
        **({"exclude": ",".join(slurm["exclude"])} if slurm.get("exclude") else {}),
    }
    return {
        "name": spec.job_id, "slurm_partition": slurm["partition"], "timeout_min": resources.timeout_min,
        "cpus_per_task": resources.cpus, "mem_gb": resources.mem_gb,
        **({"gpus_per_node": resources.gpus} if resources.gpus else {}),
        "slurm_additional_parameters": extra,
    }
