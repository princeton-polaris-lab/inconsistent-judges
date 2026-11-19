from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.paths import one_place


class CliError(RuleForgeError):
    """A command refused before it changed anything (its message says why, and what to run instead)."""


FAKE_NOTE = ("note: the fake judge answers each prompt with a label hashed from the prompt, not a model's judgment; "
             "its numbers check the pipeline and measure no model")


def note_fake(model_types: Iterable[str]) -> None:
    """Print :data:`FAKE_NOTE` on stderr when one of ``model_types`` is the fake judge's."""
    from ruleforge.evaluate.runner import FAKE_MODEL_TYPE

    if FAKE_MODEL_TYPE in model_types:
        print(FAKE_NOTE, file=sys.stderr, flush=True)


def subcommand(sub: argparse._SubParsersAction, name: str, handler: Callable[..., int], text: str, *,
               overrides: bool = False, **defaults: Any) -> argparse.ArgumentParser:
    """Add subcommand ``name`` to ``sub``, run by ``handler``; ``overrides``: it takes ``-- key=value`` overrides."""
    parser = sub.add_parser(name, help=text, description=text, allow_abbrev=False)
    parser.set_defaults(handler=handler, takes_overrides=overrides, **defaults)
    return parser


@dataclass(frozen=True)
class Assignment:
    """One value a layer above the packaged config sets: its config key, its Hydra override and how the user spelled it."""

    key: str
    override: str
    spelling: str
    files: Mapping[str, Any] = field(default_factory=dict, compare=False)
    group: bool = field(default=False, compare=False)
    leaves: tuple[str, ...] = field(default=(), compare=False)

    def sets(self) -> tuple[str, ...]:
        """The keys whose values this assignment sets."""
        return self.leaves or (self.key,)


def overlap(first: Assignment, second: Assignment) -> Optional[str]:
    """Return the key two assignments both set, or None, counting a key set inside another's mapping."""
    for one in first.sets():
        for other in second.sets():
            if one == other:
                return one
            if one.startswith(other + ".") and not second.group:
                return one
            if other.startswith(one + ".") and not first.group:
                return other
    return None


# Hydra reads a backslash as an escape only in a run that ends at a quote, the closing one included.
_BEFORE_A_QUOTE = re.compile(r"(\\*)('|\Z)")


def _escaped(found: re.Match[str]) -> str:
    backslashes, quote = found.groups()
    return backslashes * 2 + ("\\'" if quote else "")


def quoted(value: Any) -> str:
    """Return ``value`` as a Hydra override's value, strings single-quoted so their ``/``, ``,`` or ``=`` stay literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(quoted(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + _BEFORE_A_QUOTE.sub(_escaped, str(value)) + "'"


def setting(key: str, value: Any, spelling: str) -> Assignment:
    """The assignment of ``value`` to config key ``key``."""
    return Assignment(key=key, override=f"{key}={quoted(value)}", spelling=spelling)


def flag_settings(args: argparse.Namespace, table: Mapping[str, tuple[str, bool]], cwd: Path,
                  special: Sequence[str] = ()) -> list[Assignment]:
    """Return the assignments a command's value flags make, paths rebased against ``cwd``."""
    found = []
    for name, (key, is_path) in table.items():
        value = getattr(args, name)
        if value is not None:
            found.append(setting(key, rebased(value, cwd, special) if is_path else value,
                                 f"--{name.replace('_', '-')} {value}"))
    return found


def is_literal_path(value: Any, special: Sequence[str] = ()) -> bool:
    """Return whether ``value`` is a path written out: no interpolation, URI or ``special`` word."""
    return (isinstance(value, str) and value != "" and value not in special and "${" not in value
            and "://" not in value)


def is_literal_relative(value: Any, special: Sequence[str] = ()) -> bool:
    """Return whether ``value`` is a literal path that is not absolute once its ``~`` is expanded."""
    return is_literal_path(value, special) and not os.path.isabs(os.path.expanduser(value))


def rebased(value: Any, base: Path, special: Sequence[str] = ()) -> Any:
    """Return ``value`` (a path or a list) with each literal relative path made absolute against ``base``; refuse one
    the kernel would take elsewhere (a ``..`` after a symlink)."""
    if isinstance(value, (list, tuple)):
        return [rebased(item, base, special) for item in value]
    if not is_literal_path(value, special) or os.path.isabs(value):
        return value
    try:
        return str(one_place(value, base=base))
    except ValueError as error:
        raise CliError(str(error)) from error
