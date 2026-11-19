from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ruleforge.cli import records
from ruleforge.cli.common import CliError


class LazyParser(argparse.ArgumentParser):
    """An argument parser that copies in another parser's arguments the first time it parses or formats its help."""

    def __init__(self, *args: Any, source: Optional[Callable[[], argparse.ArgumentParser]] = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._source = source

    def _load(self) -> None:
        if self._source is None:
            return
        source, self._source = self._source(), None
        self._add_container_actions(source)
        self._defaults.update(source._defaults)
        self.description = self.description or source.description
        self.epilog = self.epilog or source.epilog
        self.formatter_class = source.formatter_class

    def parse_known_args(self, args: Any = None, namespace: Any = None) -> Any:
        self._load()
        return super().parse_known_args(args, namespace)

    def format_usage(self) -> str:
        self._load()
        return super().format_usage()

    def format_help(self) -> str:
        self._load()
        return super().format_help()


@dataclass(frozen=True)
class Step:
    """One step of a tool: its help and the flags naming what it writes, reads and keeps."""

    help: str
    writes: tuple[str, ...] = ()
    creates: tuple[str, ...] = ()
    beside: Optional[Callable[[Path], Sequence[Path]]] = None

    def outputs(self, args: argparse.Namespace) -> list[Path]:
        """Return what a run of the step given ``args`` writes, each path absolute."""
        created = [records.absolute_path(path) for dest in self.creates if (path := getattr(args, dest, None)) is not None
                   and not records.absolute_path(path).exists()]
        return [*self.given(args), *created]

    def given(self, args: argparse.Namespace) -> list[Path]:
        """Return each output ``args`` gives, absolute."""
        return [records.absolute_path(path) for dest in self.writes if (path := getattr(args, dest, None)) is not None]

    def sidecars(self, args: argparse.Namespace) -> list[Path]:
        """The files a run of the step given ``args`` writes beside its outputs (:attr:`beside`)."""
        if self.beside is None:
            return []
        return [sidecar for path in self.given(args) for sidecar in self.beside(path)]


@dataclass(frozen=True)
class Tool:
    """A library tool's command line: its module, parser builder, ``main`` and steps."""

    module: str
    steps: Mapping[str, Step]
    parser: str = "_build_parser"
    main: str = "main"

    def load(self) -> Any:
        return importlib.import_module(self.module)

    def build(self) -> argparse.ArgumentParser:
        """Return the tool's parser, whose subcommands are its steps."""
        return getattr(self.load(), self.parser)()

    def source(self, step: str) -> Callable[[], argparse.ArgumentParser]:
        """Return what builds ``step``'s parser."""
        return lambda: subcommands(self.build())[step]


def subcommands(parser: argparse.ArgumentParser) -> Mapping[str, argparse.ArgumentParser]:
    """Return a tool's parser's subcommands by name."""
    [choices] = [action.choices for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    return choices


def lazy_group(parser: argparse.ArgumentParser, dest: str) -> argparse._SubParsersAction:
    """Return ``parser``'s subcommands group, each a :class:`LazyParser`."""
    return parser.add_subparsers(dest=dest, required=True, parser_class=LazyParser)


def mount(group: argparse._SubParsersAction, name: str, tool: Tool, text: str, *, path: Sequence[str]) -> None:
    """Add ``tool`` to ``group`` as the command ``name`` with one subcommand per step."""
    steps = lazy_group(group.add_parser(name, help=text, description=text, allow_abbrev=False), "step")
    for step in tool.steps:
        mount_step(steps, step, tool, step, path=(*path, name))


def mount_step(group: argparse._SubParsersAction, name: str, tool: Tool, step: str, *, path: Sequence[str]) -> None:
    """Add ``tool``'s ``step`` to ``group`` (a :func:`lazy_group`) as the command ``name``."""
    words = (*path, name)
    parser = group.add_parser(name, help=tool.steps[step].help, allow_abbrev=False, add_help=False,
                              source=tool.source(step))
    parser.set_defaults(handler=lambda args, argv: run_tool(args, argv, tool, step, words), takes_overrides=False)


def arguments_after(argv: Sequence[str], path: Sequence[str]) -> list[str]:
    """Return the arguments ``argv`` gives after the command words ``path``."""
    words = tuple(path)
    for start in range(len(argv) - len(words) + 1):
        if tuple(argv[start:start + len(words)]) == words:
            return list(argv[start + len(words):])
    raise ValueError(f"{list(argv)} does not name the command {' '.join(words)}")


def tool_arguments(argv: Sequence[str], tool: Tool, step: str, path: Sequence[str]) -> list[str]:
    """Return what ``tool``'s ``main`` is given: its step's name, then the command's arguments."""
    return [step, *arguments_after(argv, path)]


WAY_OUT = "give each output a place of its own, apart from the step's other outputs, their sidecars and run records"


def output_dests(tool: Tool, step: str) -> tuple[str, ...]:
    """Return the dests of the flags naming what ``tool``'s ``step`` writes or holds."""
    spec = tool.steps[step]
    return (*spec.writes, *spec.creates)


def absolute_outputs(arguments: Sequence[str], tool: Tool, step: str) -> list[str]:
    """Return ``arguments`` with each output flag's value absolute."""
    dests = output_dests(tool, step)
    options = {option for action in tool.source(step)()._actions if action.dest in dests
               for option in action.option_strings}
    rewritten = list(arguments)
    for index, token in enumerate(rewritten):
        option, joined, value = token.partition("=")
        if token in options and index + 1 < len(rewritten) and rewritten[index + 1].startswith("~"):
            rewritten[index + 1] = str(records.absolute_path(rewritten[index + 1]))
        elif joined and option in options and value.startswith("~"):
            rewritten[index] = f"{option}={records.absolute_path(value)}"
    return rewritten


def handed(arguments: Sequence[str], tool: Tool, step: str) -> argparse.Namespace:
    """Return the tool's own parse of ``arguments``, so the places checked are the places it writes."""
    given = tool.build().parse_args(list(arguments))
    for dest in output_dests(tool, step):
        value = getattr(given, dest, None)
        if value is not None and os.fspath(value).startswith("~"):
            raise CliError(f"{value} ({dest}): the tool would take this '~' as given (<cwd>/~/...), where the command "
                           "checks the home directory's place; give the flag whole, as --flag PATH or --flag=PATH, or "
                           "the path absolute")
    return given


def run_tool(args: argparse.Namespace, argv: Sequence[str], tool: Tool, step: str, path: Sequence[str]) -> int:
    """Run ``tool``'s ``main`` on the command's arguments, every place the step writes checked first."""
    rest = absolute_outputs(tool_arguments(argv, tool, step, path), tool, step)
    given = handed(rest, tool, step)
    spec = tool.steps[step]
    outputs = spec.outputs(given)
    records.check_outputs(outputs, beside=spec.sidecars(given), way_out=WAY_OUT)
    dry_run = getattr(given, "dry_run", False)
    record = records.open_record(outputs[0], " ".join(path), argv, None) if outputs and not dry_run else None
    with records.logging_to(record, verbose=args.verbose):
        return getattr(tool.load(), tool.main)(rest)
