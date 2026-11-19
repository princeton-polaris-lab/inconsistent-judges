from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.cli.common import CliError, rebased
from ruleforge.cli.sweep import Member, Place, check_places
from ruleforge.core import paths
from ruleforge.core.provenance import code_state, package_version

RUNS_SUFFIX = ".runs"
ARGV_FNAME, CONFIG_FNAME, LOG_FNAME = "argv.json", "resolved_config.yaml", "log.txt"
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
#: Every run record this process has opened, in order; a command that opened one has started.
OPENED: list[Path] = []


def runs_dir(target: Path) -> Path:
    """Where ``target``'s run records go: ``<target>.runs`` beside it."""
    target = Path(os.path.abspath(target))
    return target.with_name(f"{target.name}{RUNS_SUFFIX}")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def config_text(cfg: Optional[DictConfig]) -> str:
    """Return the composed config as YAML, resolved; as written when some value does not resolve here."""
    if cfg is None:
        return "# no config: this command takes its arguments alone\n"
    try:
        return OmegaConf.to_yaml(cfg, resolve=True)
    except OmegaConfBaseException as error:
        return f"# unresolved ({type(error).__name__}: {str(error).splitlines()[0]})\n" + OmegaConf.to_yaml(cfg)


def print_dry_run(target: Path, cfg: Optional[DictConfig]) -> None:
    """Print what a run here would write and the config its record would keep, for a dry run that writes nothing."""
    print(f"== dry run: nothing was run or written; the run would write {target} with\n{config_text(cfg)}", end="",
          flush=True)


def protected(place: Path | str, what: str, **options: Any) -> Callable[[], None]:
    """Return the check of a place of an inventory, made when it is called."""
    def check() -> None:
        from ruleforge.evaluate.api import refuse_protected

        refuse_protected(place, what, **options)

    return check


def target_places(member: Optional[Member], target: Path, what: str, *, campaign_root: bool = False,
                  claimed: Sequence[Path] = ()) -> list[Place]:
    """Return ``target`` and its run records' directory as places of a command's inventory."""
    runs = runs_dir(target)
    return [Place(member, what, target, protect=protected(target, "the command's target", campaign_root=campaign_root,
                                                            below=not campaign_root, claimed=tuple(claimed))),
            Place(member, "run records' directory", runs, own=False,
                  protect=protected(runs, "its run records' directory"))]


def check_place(target: Path, *, campaign_root: bool = False, artifact_root: bool = False,
                claimed: Sequence[Path] = ()) -> Path:
    """Return ``target``'s ``.runs`` directory, once both are places a record may be made."""
    for place in target_places(None, target, "target", campaign_root=campaign_root, claimed=claimed):
        if not (artifact_root and place.own):  # the target itself is the command's own place
            place.protect()
    return runs_dir(target)


def absolute_path(path: Optional[Path | str]) -> Optional[Path]:
    """Return ``path`` absolute against the working directory, its ``~`` expanded (a spelling the kernel would take
    elsewhere refused); None stays None."""
    return None if path is None else Path(rebased(os.fspath(path), Path.cwd()))


def one_place(path: Path | str, what: str) -> Path:
    """Return ``path`` absolute and normalised, each ``..`` taken lexically; refuse a spelling the kernel would take
    elsewhere, naming it as ``what``."""
    try:
        return paths.one_place(path)
    except ValueError as error:
        raise CliError(f"{what} {error}") from error


def check_outputs(outputs: Sequence[Path], *, beside: Sequence[Path] = (), way_out: str,
                  claimed: Sequence[Path] = ()) -> None:
    """Check a command's outputs, each with its run records' directory beside it."""
    places = [place for output in outputs for place in target_places(None, Path(output), "an output",
                                                                     claimed=claimed)]
    places += [Place(None, "a file written beside an output", Path(path), own=False,
                     protect=protected(path, "a file the command writes beside its output")) for path in beside]
    check_places(places, way_out=way_out)


BATCH_ROOT_KIND = "batch root"


def _batch_roots(judges: Sequence[Any], *, campaign: Optional[Path]) -> list[tuple[Any, str]]:
    found: list[tuple[Any, str]] = []
    for spec in judges:
        if spec is None or not spec.capabilities().batch:
            continue
        try:
            root = spec.options.batch_root
        except OmegaConfBaseException:
            if campaign is None:
                raise
            continue
        if campaign is not None:
            from ruleforge.evaluate.api import bound_batch_root

            found.append((spec, bound_batch_root(campaign, root)))
        elif root:
            found.append((spec, root))
    return found


def batch_places(member: Optional[Member], judges: Sequence[Any], *, home: Optional[Path] = None,
                 campaign: Optional[Path] = None) -> list[Place]:
    """Return each batch judge's batch root as a place of a command's inventory."""
    home = campaign if campaign is not None else home
    return [Place(member, f"batch judge {spec.model_name}'s batch_root", Path(root), own=False, home=home, holds=True,
                  kind=BATCH_ROOT_KIND,
                  protect=None if campaign is not None else protected(
                      root, f"the batch judge {spec.model_name}'s batch_root"))
            for spec, root in _batch_roots(judges, campaign=campaign)]


def check_batch_roots(*judges: Any) -> None:
    """Refuse a run whose batch judge keeps its paid state where a write may not reach."""
    for place in batch_places(None, judges):
        place.protect()


@contextmanager
def guarded_batch_writes() -> Iterator[None]:
    """Check, for the block, every batch work directory a transport derives."""
    from ruleforge.evaluate.api import batch_work_guard
    from ruleforge.judges.batch_api import BatchDestinationRefused, guard_work_directories

    try:
        with guard_work_directories(batch_work_guard()):
            yield
    except BatchDestinationRefused as error:
        raise CliError(f"a batch transport's write was refused before it was made: {error}") from error


def open_record(target: Path, subcommand: str, argv: Sequence[str], cfg: Optional[DictConfig], *,
                campaign_root: bool = False, artifact_root: bool = False, claimed: Sequence[Path] = ()) -> Path:
    """Make the run record of ``subcommand`` writing ``target``, its place checked first."""
    parent = check_place(target, campaign_root=campaign_root, artifact_root=artifact_root, claimed=claimed)
    parent.mkdir(parents=True, exist_ok=True)
    stem = f"{_now()}-{subcommand.replace(' ', '-')}"
    attempt = 0
    while True:
        directory = parent / (stem if attempt == 0 else f"{stem}-{attempt}")
        try:
            directory.mkdir()
            break
        except FileExistsError:
            attempt += 1
    OPENED.append(directory)
    record = {"argv": list(argv), "cwd": os.getcwd(), "subcommand": subcommand, "started_at": _now(),
              "python": sys.executable, "package_version": package_version(), "code": code_state()}
    (directory / ARGV_FNAME).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (directory / CONFIG_FNAME).write_text(config_text(cfg), encoding="utf-8")
    return directory


def ended(error: BaseException) -> logging.LogRecord:
    """Return the line a run record's log ends with when its block ended on ``error``."""
    return logging.LogRecord("ruleforge.cli", logging.ERROR, __file__, 0, "the run ended on %s: %s",
                             (type(error).__name__, error), None)


# The API clients' loggers, held above DEBUG, where they log each request's body (the whole prompt), or each request.
QUIETED = {"openai": logging.INFO, "anthropic": logging.INFO, "httpx": logging.WARNING, "httpcore": logging.WARNING}


def _not_quieted(record: logging.LogRecord) -> bool:
    return record.levelno >= QUIETED.get(record.name.split(".", 1)[0], logging.NOTSET)


def log_handler(directory: Path) -> logging.Handler:
    """Return the handler of a run record's ``log.txt``, which drops the API clients' quieted lines itself, so a client
    that sets its own logger's level cannot undo it."""
    handler = logging.FileHandler(directory / LOG_FNAME, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.addFilter(_not_quieted)
    return handler


@contextmanager
def logging_to(directory: Optional[Path], *, verbose: bool = False) -> Iterator[None]:
    """Log to stderr and to the run record's ``log.txt`` for the block."""
    root = logging.getLogger()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    handlers[0].setLevel(logging.DEBUG if verbose else logging.INFO)
    handlers[0].setFormatter(logging.Formatter(LOG_FORMAT))
    handlers[0].addFilter(_not_quieted)
    if directory is not None:
        handlers.append(log_handler(directory))
    previous = root.level
    root.setLevel(logging.DEBUG)
    for handler in handlers:
        root.addHandler(handler)
    try:
        yield
    except BaseException as error:
        if directory is not None:
            handlers[-1].handle(ended(error))
        raise
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous)


def describe(payload: Any) -> str:
    """Return a result as the JSON a command prints."""
    return json.dumps(payload, indent=1, sort_keys=True, default=str)
