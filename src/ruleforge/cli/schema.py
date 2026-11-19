from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Callable, Mapping, Optional, Union

#: A path key's kinds: one path, a sequence of them, a mapping whose values are paths.
PATH, SEQUENCE, MAPPING = "path", "sequence", "mapping"
JUDGE_PATH_OPTIONS = ("model_path", "adapter_path", "batch_root")
#: A required parameter's value in a config: Hydra's mandatory marker.
REQUIRED = "???"
_PATH_WORD = re.compile(r"\b(?:Path|Pathish)\b")


def path_kind(annotation: Any) -> Optional[str]:
    """What path an annotation holds (:data:`PATH`, :data:`SEQUENCE` or :data:`MAPPING`), or None."""
    if isinstance(annotation, str):
        return PATH if _PATH_WORD.search(annotation) else None
    if isinstance(annotation, type) and issubclass(annotation, PurePath):
        return PATH
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (Union, types.UnionType):
        kinds = {path_kind(arg) for arg in args} - {None}
        return SEQUENCE if SEQUENCE in kinds else MAPPING if MAPPING in kinds else PATH if kinds else None
    if origin in (collections.abc.Sequence, list, tuple) and any(path_kind(arg) == PATH for arg in args):
        return SEQUENCE
    if origin in (collections.abc.Mapping, dict) and args and path_kind(args[-1]) == PATH:
        return MAPPING
    return None


def annotations_of(target: Any) -> dict[str, Any]:
    """Return ``target``'s annotations, evaluated where they can be."""
    raw = dict(getattr(target, "__annotations__", {}))
    if not raw and not dataclasses.is_dataclass(target):
        raw = {name: parameter.annotation for name, parameter in inspect.signature(target).parameters.items()}
    try:
        return {**raw, **typing.get_type_hints(target)}
    except (NameError, TypeError):
        return raw


def defaults_of(target: Any) -> dict[str, Any]:
    """Return each parameter's default; a required one's is :data:`REQUIRED`."""
    if dataclasses.is_dataclass(target):
        found = {}
        for item in dataclasses.fields(target):
            if item.default is not dataclasses.MISSING:
                found[item.name] = item.default
            elif item.default_factory is not dataclasses.MISSING:
                found[item.name] = item.default_factory()
            else:
                found[item.name] = REQUIRED
        return found
    return {name: REQUIRED if parameter.default is inspect.Parameter.empty else parameter.default
            for name, parameter in inspect.signature(target).parameters.items()
            if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}


@dataclass(frozen=True)
class Parameter:
    """A config key bound to a parameter of an API callable, and what path its annotation holds."""

    key: str
    target: Any = field(repr=False)
    name: str
    entry: bool = False

    @property
    def kind(self) -> Optional[str]:
        kind = path_kind(annotations_of(self.target).get(self.name, typing.Any))
        return PATH if self.entry and kind is not None else kind


def bind(target: Any, *, prefix: str = "", only: Optional[tuple[str, ...]] = None,
         keys: Mapping[str, str] = types.MappingProxyType({}), exclude: tuple[str, ...] = ()) -> tuple[Parameter, ...]:
    """Return ``target``'s parameters bound to config keys."""
    names = only if only is not None else tuple(defaults_of(target))
    return tuple(Parameter(keys.get(name, prefix + name), target, name) for name in names if name not in exclude)


@dataclass(frozen=True)
class Operation:
    """One command's schema: its parameters, judge nodes and declared path keys."""

    name: str
    parameters: tuple[Parameter, ...]
    judges: tuple[str, ...] = ()
    declared: Mapping[str, str] = field(default_factory=dict)
    special: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def path_keys(self) -> dict[str, str]:
        """Every path key, with its kind."""
        keys = {parameter.key: parameter.kind for parameter in self.parameters if parameter.kind is not None}
        for node in self.judges:
            keys.update({f"{node}.{option}": PATH for option in JUDGE_PATH_OPTIONS})
        return {**keys, **self.declared}

    def callables(self) -> tuple[Callable[..., Any], ...]:
        """The API callables its keys are bound to, each once."""
        return tuple(dict.fromkeys(parameter.target for parameter in self.parameters))


def matches(pattern: str, key: str) -> bool:
    """Return whether the dotted ``pattern`` ends with the dotted ``key``'s segments (``*`` matching any one)."""
    want, have = pattern.split("."), key.split(".")
    return len(have) <= len(want) and all(w in ("*", h) for w, h in zip(want[len(want) - len(have):], have))
