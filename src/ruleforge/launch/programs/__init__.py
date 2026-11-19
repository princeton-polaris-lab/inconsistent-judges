from __future__ import annotations

import copy
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional, Sequence

from ruleforge.core.credentials import refused_arguments
from ruleforge.core.errors import RuleForgeError
from ruleforge.core.provenance import CheckoutMismatch, repo_root

REPO_ROOT = repo_root()
CONFIG_DIR = REPO_ROOT / "configs"
BASIC_SWEEPER = "hydra._internal.core_plugins.basic_sweeper.BasicSweeper"
#: `hydra.main`'s `version_base`: None is the installed hydra-core's own semantics.
VERSION_BASE: Optional[str] = None


class NoCheckout(RuleForgeError):
    """A job-side program run where no checkout's `configs/` is (an installed wheel)."""


class UncheckedJobs(RuleForgeError):
    """A command line whose jobs a checked program cannot lay out before Hydra runs them."""


class HydraJobs(NamedTuple):
    """What Hydra composes and runs for a program's command line."""

    #: The config Hydra composes before any job and configures its logging from; None when it composes none.
    launch: Optional[Any]
    jobs: list[Any]
    sweep: bool


def config_dir() -> str:
    """Return the checkout's config directory as Hydra's `config_path` takes it; refused where it is none."""
    if not Path(CONFIG_DIR).is_dir():
        raise NoCheckout(f"{CONFIG_DIR} is no directory: the launcher's job-side programs compose a checkout's "
                         "configs/ and run from a checkout's top (a launch's jobs do); run it from one")
    return str(CONFIG_DIR)


def check_top() -> Path:
    """Return the working directory, refused unless it is the top of the checkout these programs came from."""
    here = Path.cwd().resolve()
    if here != REPO_ROOT:
        raise CheckoutMismatch(
            f"this program runs from {here}, not from {REPO_ROOT}, the top of the checkout it imported ruleforge from: "
            "it would run that checkout's code and configs on the paths it reads from here. Run it from the top of "
            "the checkout whose code it should run, under that checkout's environment (uv run there) or with its "
            "src/ first on PYTHONPATH")
    return here


def run_program(task: Callable[[Any], Any], config_name: str, *, check: Callable[..., Any],
                protected: bool = False, opens_log: bool = False) -> None:
    """Run `task` under Hydra over the checkout's configs, every job Hydra would run checked first with `check`.
    ``protected``: a program that writes no campaign root, every place Hydra or its task writes in the job's output
    directory held to ``refuse_protected`` too (``task_places``). ``opens_log``: a task that calls
    ``ensure_job_logging``, whose log (``task_log``) is held as Hydra's files are.
    """
    if refused_arguments(sys.argv[1:], Path(sys.argv[0]).name):
        raise SystemExit(2)
    import hydra

    config_path = config_dir()
    check_top()
    check_jobs(task, config_path, config_name, sys.argv[1:], check, protected=protected, opens_log=opens_log)
    hydra.main(version_base=VERSION_BASE, config_path=config_path, config_name=config_name)(task)()


def check_jobs(task: Callable[[Any], Any], config_path: str, config_name: str, arguments: Sequence[str],
               check: Callable[..., Any], *, protected: bool = False, opens_log: bool = False) -> None:
    """Check every job Hydra would run `task` on for `arguments` (its config with `check`, then every place Hydra writes
    for it and the command line, with ``refuse_protected`` too when ``protected``) before Hydra makes anything."""
    from hydra._internal.utils import run_and_report

    run_and_report(lambda: _check_jobs(task, config_path, config_name, arguments, check, protected, opens_log))


def refuse_outputs(places: Iterable[tuple[Optional[str], str]], workdir: Optional[str] = None, *,
                   below: bool = False) -> None:
    """Refuse each place a job writes (None: none) where nothing but a campaign's own writers may write: in or at a
    campaign root, on a published run's path or through a link (``refuse_protected``), with ``below`` a link anywhere
    under it too. A relative place is read from ``workdir`` when the job runs there."""
    from ruleforge.evaluate.api import refuse_protected

    for place, what in places:
        if place is not None:
            refuse_protected(place if workdir is None else os.path.join(workdir, place), what, below=below)


def task_log(job: Any) -> tuple[str, str]:
    """Return the log ``ensure_job_logging`` opens in the job's output directory when Hydra's logging is off or a
    library reset it, with what it is."""
    from ruleforge.optimize.job_logging import LOG_FNAME

    return (os.path.join(str(job.hydra.runtime.output_dir), LOG_FNAME),
            "the log the task opens in the job's output directory")


def task_places(job: Any) -> list[tuple[str, str]]:
    """Return what a protected program's task writes in its job's output directory beside Hydra: the directory itself,
    a tree the job owns, and its log (``task_log``)."""
    return [(str(job.hydra.runtime.output_dir), "the job's output directory"), task_log(job)]


def selected(cfg: Any, key: str) -> Optional[str]:
    """Return ``cfg``'s ``key`` as a string, or None where the config sets no value there that resolves."""
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, key, throw_on_missing=False, throw_on_resolution_failure=False)
    return None if value is None else str(value)


def _check_jobs(task: Callable[[Any], Any], config_path: str, config_name: str, arguments: Sequence[str],
                check: Callable[..., Any], protected: bool = False, opens_log: bool = False) -> None:
    from hydra.core.hydra_config import HydraConfig
    from hydra.core.utils import env_override
    from omegaconf import open_dict, read_write

    launch, jobs, sweep = hydra_jobs(task, config_path, config_name, arguments)
    if launch is None:
        return
    held = HydraConfig.instance().cfg
    try:
        for job in jobs:
            HydraConfig.instance().set_config(job)
            try:
                output, unresolved = output_directory(job, sweep=sweep), None
            except Exception as error:
                output, unresolved = "", error
            task_cfg = copy.deepcopy(job)
            with read_write(task_cfg), open_dict(task_cfg):
                del task_cfg["hydra"]
            with env_override(job.hydra.job.env_set):
                check(task_cfg, workdir=None if unresolved is not None else job_workdir(job))
            if unresolved is not None:
                raise unresolved
            places = hydra_places(job, output)
            _refuse_places(places, hydra_files(job))
            if protected:
                refuse_outputs(places)
                refuse_outputs(task_places(job), below=True)
            if opens_log:
                refuse_place(*task_log(job), file=True)
        HydraConfig.instance().set_config(launch)
        places = hydra_launch_places(launch, sweep=sweep)
        _refuse_places(places, hydra_launch_files(launch, sweep=sweep))
        if protected:
            refuse_outputs(places)
    finally:
        HydraConfig.instance().cfg = held


def hydra_jobs(task: Callable[[Any], Any], config_path: str, config_name: str,
               arguments: Sequence[str]) -> HydraJobs:
    """Return the configs Hydra would compose and run `task` on for `arguments`, composed as `hydra.main` composes them;
    a pickled rerun or another sweeper's sweep is refused."""
    from hydra import version
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import (
        create_automatic_config_search_path,
        detect_calling_file_or_module_from_task_function,
        detect_task_name,
        get_args_parser,
    )
    from hydra.core.config_search_path import SearchPathQuery
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.core.utils import validate_config_path
    from hydra.types import RunMode
    from omegaconf import OmegaConf, open_dict

    version.setbase(VERSION_BASE)
    args = get_args_parser().parse_args(list(arguments))
    if args.experimental_rerun is not None:
        raise UncheckedJobs("--experimental-rerun runs a pickled config, whose places this program cannot hold before "
                            "it writes them; run the job again with its arguments")
    config_name = config_name if args.config_name is None else args.config_name
    config_path = config_path if args.config_path is None else args.config_path
    validate_config_path(config_path)
    calling_file, calling_module = detect_calling_file_or_module_from_task_function(task)
    search_path = create_automatic_config_search_path(calling_file, calling_module, config_path)
    if args.config_dir is not None:
        search_path.prepend(provider="command-line", path=f"file://{os.path.abspath(args.config_dir)}",
                            anchor=SearchPathQuery(provider="schema"))
    hydra = Hydra.create_main_hydra2(task_name=detect_task_name(calling_file, calling_module),
                                     config_search_path=search_path)
    try:
        if args.help:
            return HydraJobs(hydra.compose_config(config_name, list(args.overrides), run_mode=RunMode.RUN), [], False)
        if args.resolve and args.cfg is None:
            return HydraJobs(None, [], False)
        if args.hydra_help:
            return HydraJobs(hydra.compose_config(None, list(args.overrides), run_mode=RunMode.RUN), [], False)
        commands = args.run + (args.cfg is not None) + args.multirun + args.shell_completion + (args.info is not None)
        if commands > 1 or not (args.run or args.multirun or commands == 0):
            return HydraJobs(None, [], False)
        overrides = list(args.overrides)
        mode = hydra.get_mode(config_name=config_name, overrides=overrides)
        if mode is None:
            mode = RunMode.MULTIRUN if args.multirun else RunMode.RUN
            overrides.append(f"hydra.mode={mode.name}")
        elif args.multirun and mode == RunMode.RUN:
            mode = RunMode.MULTIRUN
            overrides.append("hydra.mode=MULTIRUN")
        if mode == RunMode.RUN:
            job = hydra.compose_config(config_name, overrides, run_mode=RunMode.RUN)
            return HydraJobs(copy.deepcopy(job), [job], False)
        master = hydra.compose_config(config_name, overrides, run_mode=RunMode.MULTIRUN)
        sweeper = OmegaConf.select(master, "hydra.sweeper._target_")
        if sweeper != BASIC_SWEEPER:
            raise UncheckedJobs(f"hydra.sweeper {sweeper}: this program lays out only {BASIC_SWEEPER}'s jobs, to "
                                "check each before Hydra makes anything")
        params = OmegaConf.select(master, "hydra.sweeper.params") or {}
        task_overrides = OmegaConf.to_container(master.hydra.overrides.task, resolve=False)
        sweep = OverridesParser.create(config_loader=hydra.config_loader).parse_overrides(
            [*(f"{key}={value}" for key, value in params.items()), *task_overrides])
        jobs = [hydra.config_loader.load_sweep_config(master, list(job))
                for batch in BasicSweeper.split_arguments(sweep, None) for job in batch]
        for number, job in enumerate(jobs):
            with open_dict(job):
                job.hydra.job.id = job.hydra.job.num = number
        return HydraJobs(master, jobs, True)
    finally:
        GlobalHydra.instance().clear()


SAVED_CONFIGS = ("config.yaml", "hydra.yaml", "overrides.yaml")
MULTIRUN_CONFIG = "multirun.yaml"


def output_directory(job: Any, *, sweep: bool) -> str:
    """Return the directory Hydra makes for `job`, resolved as `run_job` resolves it; sets the job's
    `hydra.runtime.output_dir`."""
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf, open_dict, read_write

    keys = ("hydra.sweep.dir", "hydra.sweep.subdir") if sweep else ("hydra.run.dir",)
    output = os.path.join(*(str(OmegaConf.select(job, key)) for key in keys))
    with read_write(job.hydra.runtime), open_dict(job.hydra.runtime):
        job.hydra.runtime.output_dir = os.path.abspath(output)
    HydraConfig.instance().set_config(job)
    return output


def job_workdir(job: Any) -> Optional[str]:
    """Return the directory `job` reads its relative paths from under `hydra.job.chdir`, else None."""
    return os.path.realpath(job.hydra.runtime.output_dir) if job.hydra.job.chdir else None


def _saved_directory(job: Any) -> str:
    return os.path.join(job.hydra.runtime.output_dir, str(job.hydra.output_subdir))


def hydra_places(job: Any, output: str) -> list[tuple[str, str]]:
    """Return the places Hydra writes for one job, each with what it is: its output directory, its saved configs and its
    log files."""
    places = [(output, "Hydra's output directory")]
    if job.hydra.output_subdir is not None:
        places.append((_saved_directory(job), "the directory Hydra saves the job's configs in (hydra.output_subdir)"))
    return [*places, *hydra_files(job)]


def hydra_files(job: Any) -> list[tuple[str, str]]:
    """Return the files among ``hydra_places``: the configs Hydra saves and its job log files."""
    files = []
    if job.hydra.output_subdir is not None:
        files += [(os.path.join(_saved_directory(job), name), f"the {name} Hydra saves there") for name in SAVED_CONFIGS]
    workdir = job_workdir(job)
    files += [(str(handler.filename) if workdir is None else os.path.join(workdir, str(handler.filename)),
               "a log file of hydra.job_logging") for handler in _handlers(job, "hydra.job_logging")]
    return files


def _hydra_log_files(config: Any) -> list[tuple[str, str]]:
    return [(str(handler.filename), "a log file of hydra.hydra_logging")
            for handler in _handlers(config, "hydra.hydra_logging")]


def _master_config(config: Any) -> tuple[str, str]:
    return os.path.join(str(config.hydra.sweep.dir), MULTIRUN_CONFIG), f"the {MULTIRUN_CONFIG} Hydra saves there"


def hydra_launch_places(config: Any, *, sweep: bool) -> list[tuple[str, str]]:
    """Return the places Hydra writes for a command line beside its jobs': its logging files and, for a sweep, its
    directory, master config and submitit folder."""
    from omegaconf import OmegaConf

    places = _hydra_log_files(config)
    if sweep:
        places += [(str(config.hydra.sweep.dir), "Hydra's sweep directory"), _master_config(config)]
        folder = OmegaConf.select(config, "hydra.launcher.submitit_folder")
        if folder is not None:
            places.append((str(folder), "submitit's folder (hydra.launcher.submitit_folder)"))
    return places


def hydra_launch_files(config: Any, *, sweep: bool) -> list[tuple[str, str]]:
    """Return the files among ``hydra_launch_places``: Hydra's own log files and, for a sweep, its master config."""
    return [*_hydra_log_files(config), *([_master_config(config)] if sweep else [])]


def _refuse_places(places: Sequence[tuple[str, str]], files: Sequence[tuple[str, str]]) -> None:
    named = {place for place, _ in files}
    for place, what in places:
        refuse_place(place, what, file=place in named)


def refuse_place(place: str, what: str, *, file: bool = False) -> None:
    """Refuse a place Hydra or a task writes on a pinned root's path (through a symlink too), or a file there with a
    second name (a hard link); a ``file`` place that is itself a symlink too, which a write would follow.
    Not looked at here: a directory place or anything below it (a campaign's may sit in its own root, whose entry checks
    cover the directories its jobs write; a protected program's own trees go to ``refuse_outputs``), a symlinked parent,
    a link planted after entry, third-party caches and the scheduler's stdout and stderr."""
    from ruleforge.evaluate.api import CampaignError, refuse_pinned

    refuse_pinned(place, what)
    try:
        entry = os.lstat(place)
    except OSError:
        return
    if file and stat.S_ISLNK(entry.st_mode):  # a dangling one too: the write would make its target
        raise CampaignError(f"{what} {place} is a symlink: a write there would follow it to {os.path.realpath(place)}; "
                            "remove it, or replace it with a file of its own")
    try:
        info = os.stat(place)
    except OSError:
        return
    if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
        raise CampaignError(f"{what} {place} is a file with {info.st_nlink} names (hard links): a write there would "
                            "change it under each of them, which no path shows -- a file of a root a committed golden "
                            "pins among them, possibly; remove it, or replace it with a file of its own")


def _handlers(config: Any, node: str) -> list[Any]:
    from omegaconf import OmegaConf

    return [handler for handler in (OmegaConf.select(config, f"{node}.handlers") or {}).values()
            if "filename" in handler]
