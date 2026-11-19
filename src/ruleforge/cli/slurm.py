from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from ruleforge.cli.common import CliError
from ruleforge.cli.sweep import Place
from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.optional import require
from ruleforge.core.provenance import code_state
from ruleforge.launch.profile import CLUSTER_DIR as CLUSTER_DIR
from ruleforge.launch.profile import Profile as Profile
from ruleforge.launch.profile import find_profile as find_profile
from ruleforge.launch.profile import profile_dirs as profile_dirs

if TYPE_CHECKING:
    from ruleforge.launch.jobs import JobSpec

#: The command a job runs, found on its ``PATH`` (the submitting environment's ``bin``).
PROGRAM = "ruleforge"
#: Options only the submission reads, each with the count of values it takes.
SUBMISSION_OPTIONS = {"--slurm": 1, "--cluster-dir": 1, "--dry-run": 0, "--gpus": 1, "--timeout-min": 1,
                      "--cpus": 1, "--mem-gb": 1, "--json": 0}


@dataclass(frozen=True)
class Request:
    """What a command asks of the cluster: GPUs (0: a CPU job), and any of the profile's defaults it overrides."""

    gpus: int = 0
    timeout_min: Optional[int] = None
    cpus: Optional[int] = None
    mem_gb: Optional[int] = None
    resumable: bool = False


def request(args: argparse.Namespace, gpus: int, *, resumable: bool = False) -> Request:
    """Return what a command's job asks for, the profile's defaults where its SLURM options give none."""
    return Request(gpus=gpus, timeout_min=args.timeout_min, cpus=args.cpus, mem_gb=args.mem_gb, resumable=resumable)


def job(job_id: str, argv: Sequence[str], profile: Profile, request: Request, *,
        depends_on: Sequence[str] = ()) -> "JobSpec":
    """Return one job running ``ruleforge <argv>``, sized from ``request`` over the profile's defaults."""
    from ruleforge.launch.jobs import JobSpec

    if request.gpus > profile.gpus_per_node:
        raise CliError(f"{job_id} asks for {request.gpus} GPUs; a node of {profile.path.stem} has "
                       f"{profile.gpus_per_node}")
    resources = profile.resources(request.gpus, cpus=request.cpus, mem_gb=request.mem_gb,
                                  timeout_min=request.timeout_min, resumable=request.resumable)
    kind = next(item for item in argv if not item.startswith("-"))
    return JobSpec(job_id=job_id, kind=kind, step=0, argv=(PROGRAM, *argv), resources=resources,
                   depends_on=tuple(depends_on))


def backend(profile: Profile, repo: Path, **kwargs: Any) -> Any:
    """Return the SLURM backend of ``ruleforge.launch`` on the profile's cluster."""
    from ruleforge.launch.backends import SlurmBackend

    class ProfileBackend(SlurmBackend):
        def prepare(self, spec: "JobSpec") -> "JobSpec":
            return spec  # each job's QOS is the profile's (a CPU job gets the profile's CPU QOS, never a default)

    return ProfileBackend(repo, profile=profile, **kwargs)


NO_ARTIFACT_ROOT = ("--slurm records its launch under the artifact root ({}); set RULEFORGE_ARTIFACT_ROOT, or pass "
                    "--dry-run")


def launch_places() -> list[Place]:
    """Return the places a launch writes, as places of the command's inventory."""
    from ruleforge.cli.records import protected
    from ruleforge.launch.records import records_places

    try:
        places = records_places()
    except ArtifactRootError as error:
        raise CliError(NO_ARTIFACT_ROOT.format(error)) from error
    return [Place(None, place.name, place.path, own=False, protect=protected(place.path, place.what))
            for place in places]


def submit(specs: Sequence["JobSpec"], profile: Profile, *, verb: str, dry_run: bool,
           as_json: bool = False, **backend_kwargs: Any) -> int:
    """Submit ``specs`` as one launch, or under ``dry_run`` print each job's request."""
    require("--slurm", "submitit")
    from ruleforge.launch.cli import Output, launch
    from ruleforge.launch.environment import LaunchError
    from ruleforge.launch.records import new_launch_id

    state = code_state()
    code = {"commit": state["code_revision"] or "unversioned", "dirty": bool(state["code_dirty"]),
            "code_state_sha256": state["code_state_sha256"]}
    out = Output(as_json, follow="ruleforge jobs")
    try:
        # With no artifact root set, the launch's first check raises ArtifactRootError, refused below before anything is
        # written.
        return launch(backend(profile, Path(os.getcwd()), echo=out.say, **backend_kwargs), specs, verb=verb,
                      launch_id=new_launch_id(code["commit"]), code=code, spec_record=None, out=out, dry_run=dry_run,
                      refuse=dict, extra={"profile": str(profile.path)})
    except ArtifactRootError as error:
        raise CliError(NO_ARTIFACT_ROOT.format(error)) from error
    except LaunchError as error:
        raise CliError(str(error)) from error


def passthrough_argv(argv: Sequence[str], *, drop: Mapping[str, int]) -> list[str]:
    """Return ``argv`` without the options ``drop`` names."""
    kept: list[str] = []
    skip = 0
    for index, item in enumerate(argv):
        if skip:
            skip -= 1
            continue
        if item == "--":
            kept += argv[index:]
            break
        option, equals, _ = item.partition("=")
        if option in drop:
            skip = 0 if equals else drop[option]
            continue
        kept.append(item)
    return kept
