from __future__ import annotations

import argparse
import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from ruleforge.cli import records
from ruleforge.cli.common import CliError, subcommand
from ruleforge.core.optional import require
from ruleforge.launch.control import add_control_verbs, add_launch_verbs

PROGRAM = "ruleforge"
#: What the ``ruleforge`` parser adds to a launch verb's arguments, which the launcher's verb never reads.
COMMAND_KEYS = ("command", "verbose", "overrides", "takes_overrides")


def register(sub: argparse._SubParsersAction) -> None:
    """Add ``jobs`` and its verbs to the ``ruleforge`` command."""
    text = "launch a job, a campaign or a training run, and follow it by its record: launch, status, wait, pull, stop"
    parser = subcommand(sub, "jobs", run_jobs, text)
    verbs = parser.add_subparsers(dest="verb", required=True)
    text = "launch a job, a campaign or a training run (the launcher's launch verbs)"
    launching = verbs.add_parser("launch", help=text, description=text, allow_abbrev=False)
    for kind in add_launch_verbs(launching.add_subparsers(dest="verb", required=True),
                                 program=f"{PROGRAM} jobs launch", allow_abbrev=False).values():
        kind.set_defaults(handler=run_launch, takes_overrides=True)
    add_control_verbs(verbs, allow_abbrev=False)


def launcher_verb(args: argparse.Namespace, handler: Any, **options: Any) -> int:
    """Run a launcher verb; its refusals reach the command line's boundary, which picks the exit status."""
    from ruleforge.launch.cli import run_verb

    try:
        return run_verb(args, handler, refusals=(), **options)
    except (KeyError, yaml.YAMLError) as error:  # the launcher's refusals the command line would take for failures
        raise CliError(f"{type(error).__name__}: {error}") from error


def run_launch(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run ``jobs launch <verb>`` through the launcher's own entry."""
    require("launching and following jobs", "submitit")
    from ruleforge.launch.cli import LAUNCH_HANDLERS

    launched = argparse.Namespace(**{key: value for key, value in vars(args).items() if key not in COMMAND_KEYS})
    launched.args = list(args.overrides)
    launched.handler = LAUNCH_HANDLERS[args.verb]
    launched.command_line = [PROGRAM, *argv]
    return launcher_verb(launched, launched.handler, program=f"{PROGRAM} jobs launch", follow=f"{PROGRAM} jobs")


def check_places(args: argparse.Namespace) -> list[Path]:
    """Refuse a verb that would write where nothing may, before it runs or records anything."""
    pulls = args.verb == "pull" or (args.verb == "wait" and args.pull)
    if not pulls and args.verb != "stop":
        return []
    from ruleforge.launch.records import LaunchRecord, check_writes

    record = LaunchRecord.load(args.launch)
    roots = check_writes(record, merges=pulls)
    for root in roots:
        records.check_place(root, campaign_root=True)
    return roots


def open_records(args: argparse.Namespace, argv: Sequence[str], roots: Sequence[Path]) -> list[Path]:
    """Return the run records of a pull that merges into ``roots``, one beside each."""
    return [records.open_record(root, f"jobs {args.verb}", argv, None, campaign_root=True) for root in roots]


@contextmanager
def logging_to(directories: Sequence[Path]) -> Iterator[None]:
    """Log, for the block, to each run record's ``log.txt`` and to stderr; with no record, logging is left as it is."""
    with ExitStack() as stack:
        if directories:
            stack.enter_context(records.logging_to(directories[0]))
        others = []
        for directory in directories[1:]:
            handler = records.log_handler(directory)
            logging.getLogger().addHandler(handler)
            stack.callback(handler.close)
            stack.callback(logging.getLogger().removeHandler, handler)
            others.append(handler)
        try:
            yield
        except BaseException as error:
            for handler in others:
                handler.handle(records.ended(error))
            raise


def run_jobs(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run ``jobs <verb>`` through the launcher's own verb, the places it writes checked first."""
    require("launching and following jobs", "submitit")
    from ruleforge.launch.cli import CONTROL_HANDLERS

    verb = CONTROL_HANDLERS[args.verb]

    def checked(args: argparse.Namespace, out: Any, repo: Path, backend_kwargs: Mapping[str, Any]) -> int:
        roots = check_places(args)
        with logging_to(open_records(args, argv, roots)):
            return verb(args, out, repo, backend_kwargs)

    return launcher_verb(args, checked, program="ruleforge jobs")
