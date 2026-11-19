from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import env_value, package_path
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.jobs import Resources

CLUSTER_DIR = "CLUSTER_DIR"
EXAMPLE = "configs/cluster/example.yaml"


@dataclass(frozen=True)
class Profile:
    """One cluster: where its jobs go and what each asks for when the command says nothing else."""

    path: Path
    partition: str
    account: Optional[str]
    qos_gpu: Optional[str]
    qos_cpu: Optional[str]
    gpus_per_node: int
    timeout_min: int
    cpus: int
    mem_gb: int
    mem_gb_per_gpu: int
    signal_delay_s: Optional[int]
    setup: tuple[str, ...]
    exclude: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> "Profile":
        """Return the profile at `path`; a missing key is refused."""
        try:
            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            defaults, qos = data.get("defaults") or {}, data.get("qos") or {}
            exclude = data.get("exclude") or []
            if not isinstance(exclude, list) or not all(isinstance(node, str) and node for node in exclude):
                raise ValueError(f"exclude is a list of node names, not {exclude!r}")
            return cls(path=Path(os.path.abspath(path)), partition=str(data["partition"]), account=data.get("account"),
                       qos_gpu=qos.get("gpu"), qos_cpu=qos.get("cpu"), gpus_per_node=int(data["gpus_per_node"]),
                       timeout_min=int(defaults["timeout_min"]), cpus=int(defaults["cpus"]),
                       mem_gb=int(defaults["mem_gb"]), mem_gb_per_gpu=int(defaults["mem_gb_per_gpu"]),
                       signal_delay_s=defaults.get("signal_delay_s"),
                       setup=tuple(str(line) for line in data.get("setup") or ()), exclude=tuple(exclude))
        except (OSError, OmegaConfBaseException, KeyError, TypeError, ValueError, AttributeError,
                yaml.YAMLError) as error:
            raise LaunchError(f"{path}: not a cluster profile ({type(error).__name__}: {error}); see "
                              f"{package_path(EXAMPLE)}") from error

    def site(self) -> dict[str, Any]:
        """Return the cluster as the SLURM backend renders a job's request: partition, account, warning lead and excluded
        nodes."""
        return {"partition": self.partition, "account": self.account, "signal_delay_s": self.signal_delay_s,
                "exclude": list(self.exclude)}

    def qos(self, gpus: int) -> Optional[str]:
        """Return the QOS a job of `gpus` GPUs gets when nothing else names one."""
        return self.qos_gpu if gpus else self.qos_cpu

    def resources(self, gpus: int, *, cpus: Optional[int] = None, mem_gb: Optional[int] = None,
                  timeout_min: Optional[int] = None, qos: Optional[str] = None, resumable: bool = False) -> Resources:
        """Return what a job of `gpus` GPUs asks for: what is given, else this profile's defaults and QOS."""
        default_mem = self.mem_gb_per_gpu * gpus if gpus else self.mem_gb
        return Resources(kind="gpu" if gpus else "cpu", gpus=gpus, cpus=self.cpus if cpus is None else cpus,
                         mem_gb=default_mem if mem_gb is None else mem_gb,
                         timeout_min=self.timeout_min if timeout_min is None else timeout_min,
                         qos=self.qos(gpus) if qos is None else qos, resumable=resumable)

    def check_gpus(self, job_id: str, gpus: int) -> None:
        """Refuse a job asking for more GPUs than one of the cluster's nodes has."""
        if gpus > self.gpus_per_node:
            raise LaunchError(f"{job_id} asks for {gpus} GPUs; a node of {self.path.stem} has {self.gpus_per_node}")


def profile_dirs(cluster_dir: Optional[Path]) -> list[Path]:
    """Return the directories a profile is looked for in, in order."""
    from ruleforge.core.paths import repository_root

    dirs = [Path(cluster_dir)] if cluster_dir is not None else []
    named = env_value(CLUSTER_DIR)
    if named:
        dirs.append(Path(named))
    try:
        dirs.append(repository_root() / "configs" / "cluster")
    except ArtifactRootError:
        pass
    return [*dirs, package_path("configs/cluster")]


def find_profile(name: str, cluster_dir: Optional[Path] = None) -> Profile:
    """Return the profile `name` names: a YAML file, or `<name>.yaml` in the first of `profile_dirs` that holds one.
    """
    given = Path(name).expanduser()
    if given.suffix in (".yaml", ".yml"):
        if not given.is_file():
            raise LaunchError(f"{name}: no such profile file")
        return Profile.load(given)
    dirs = profile_dirs(cluster_dir)
    for directory in dirs:
        if (directory / f"{name}.yaml").is_file():
            return Profile.load(directory / f"{name}.yaml")
    raise LaunchError(f"no profile {name}.yaml in {', '.join(str(path) for path in dirs)}")
