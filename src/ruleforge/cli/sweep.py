from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from ruleforge.cli.common import CliError
from ruleforge.launch.jobs import expand_values

CONDITION, CAMPAIGN = "condition", "campaign"
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Axes:
    """An operation's sweepable keys: each short name's config key and level."""

    keys: Mapping[str, tuple[str, str]]
    prefixes: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    files: tuple[str, ...] = ()
    label: Optional[Callable[[str, str], Optional[str]]] = None

    def find(self, name: str) -> tuple[str, str, str]:
        """``(short name, config key, level)`` for a swept name, short or long."""
        if name in self.keys:
            return (name, *self.keys[name])
        for short, (key, level) in self.keys.items():
            if name == key:
                return (short, key, level)
        for short, (prefix, level) in self.prefixes.items():
            for given, rest in ((short, name[len(short):]), (prefix, name[len(prefix):])):
                if name.startswith(given) and rest:
                    return (f"{short}{rest}", f"{prefix}{rest}", level)
        known = sorted([*self.keys, *(f"{prefix}<key>" for prefix in self.prefixes)])
        raise CliError(f"--sweep {name}: not a key this command sweeps; it sweeps {', '.join(known)}")


@dataclass(frozen=True)
class Axis:
    name: str
    key: str
    level: str
    values: tuple[str, ...]


def parse(texts: Sequence[str], axes: Axes) -> list[Axis]:
    """Parse each ``--sweep KEY=VALUES``: a key swept once, values given once each, none empty."""
    parsed: list[Axis] = []
    for text in texts:
        name, separator, spelled = text.partition("=")
        if not separator or not name:
            raise CliError(f"--sweep {text!r}: expected KEY=VALUES (a,b,c or A..B)")
        short, key, level = axes.find(name)
        values = expand_values(text, spelled, "--sweep", CliError)
        if any(axis.key == key for axis in parsed):
            raise CliError(f"--sweep sweeps {short} twice: give its values in one --sweep")
        parsed.append(Axis(short, key, level, tuple(values)))
    return parsed


@dataclass(frozen=True)
class Member:
    """One point of a sweep: each swept key's value, in axis order."""

    index: int
    values: tuple[tuple[Axis, str], ...]

    def at(self, level: str) -> tuple[tuple[Axis, str], ...]:
        return tuple((axis, value) for axis, value in self.values if axis.level == level)

    def describe(self) -> str:
        return ", ".join(f"{axis.name}={value}" for axis, value in self.values) or "(no sweep)"


def members(axes: Sequence[Axis]) -> list[Member]:
    """The product of ``axes``' values, the first outermost; one member with no value when nothing is swept."""
    return [Member(index, tuple(zip(axes, point)))
            for index, point in enumerate(itertools.product(*(axis.values for axis in axes)))]


def label(axis: Axis, value: str, axes: Axes) -> str:
    """Return how a member's value names it: a file's stem, else the value; refused unless it is a name segment."""
    custom = axes.label(axis.name, value) if axes.label is not None else None
    found = custom if custom is not None else Path(value).stem if axis.name in axes.files else value
    if not _LABEL.fullmatch(found):
        raise CliError(f"--sweep {axis.name}={value}: its label {found!r} cannot name a member (letters, digits, "
                       f"'.', '_' and '-' only); name the members with a template naming {{{axis.name}}}")
    return found


def segments(values: Sequence[tuple[Axis, str]], axes: Axes, *, separator: str) -> str:
    """``<separator><name>.<label>`` for each value: a member's addition to a default name."""
    return "".join(f"{separator}{axis.name}.{label(axis, value, axes)}" for axis, value in values)


def fill(template: str, values: Sequence[tuple[Axis, str]], axes: Axes, *, what: str) -> str:
    """``template`` with ``{<name>}`` filled by each value's label, every swept key's name required in it."""
    missing = [axis.name for axis, _ in values if f"{{{axis.name}}}" not in template]
    if missing:
        raise CliError(f"{what} {template!r} under a sweep is a template naming {{key}} for every swept key of its "
                       f"level: it names none of {', '.join(f'{{{name}}}' for name in missing)}")
    for axis, value in values:
        template = template.replace(f"{{{axis.name}}}", label(axis, value, axes))
    return template


def composed_value(cfg: Any, key: str) -> Any:
    """Return the value ``key`` holds in a composed config, as plain data."""
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    node = OmegaConf.select(cfg, key, throw_on_missing=False)
    if not isinstance(node, (DictConfig, ListConfig)):
        return node
    try:
        return OmegaConf.to_container(node, resolve=True)
    except OmegaConfBaseException:
        return OmegaConf.to_container(node, resolve=False)


def refuse_equal_values(planned: Sequence[tuple[Member, Any]], axes: Sequence[Axis]) -> None:
    """Refuse two values of one swept key that compose to one value."""
    for axis in axes:
        seen: dict[str, str] = {}
        for member, cfg in planned:
            given = next(value for swept, value in member.values if swept.key == axis.key)
            found = composed_value(cfg, axis.key)
            shown = json.dumps(found, sort_keys=True, default=str)
            other = seen.setdefault(shown, given)
            if other != given:
                raise CliError(f"--sweep {axis.name}: {other} and {given} compose to one value ({shown}), so the "
                               "members they name would be one under two names; give each value once")


def refuse_collisions(destinations: Sequence[tuple[Member, tuple[str, ...]]], *, way_out: str) -> None:
    """Refuse members whose destinations coincide, naming them and ``way_out`` (the template that separates
    them), before any member runs."""
    seen: dict[tuple[str, ...], Member] = {}
    for member, destination in destinations:
        if destination in seen:
            raise CliError(f"sweep members {seen[destination].describe()} and {member.describe()} would both write "
                           f"{' / '.join(destination)}; {way_out}")
        seen[destination] = member


@dataclass(frozen=True)
class Place:
    """A place a command writes: ``own`` if a member writes it alone, else one it may share."""

    member: Optional[Member]
    what: str
    path: Path
    own: bool = True
    home: Optional[Path] = None
    holds: bool = False
    kind: Optional[str] = None
    protect: Optional[Callable[[], None]] = field(default=None, compare=False, repr=False)


def _real(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _lies_in_home(place: Place, owner_real: Path) -> bool:
    return not place.own and place.home is not None and _real(place.home) == owner_real


def _overlap(place: Place, real: Path, other: Place, other_real: Path) -> Optional[str]:
    if real == other_real:
        if _lies_in_home(place, other_real) or _lies_in_home(other, real) or (
                place.kind is not None and place.kind == other.kind):
            return None
        return f"are one place, {real}"
    for inner, inner_real, outer, outer_real in ((place, real, other, other_real), (other, other_real, place, real)):
        if inner_real.is_relative_to(outer_real):
            if _lies_in_home(inner, outer_real) or outer.holds:
                return None
            return f"lie one inside the other ({other_real}, {real})"
    return None


def _whose(place: Place) -> str:
    return "its" if place.member is not None else "the launch's"


def refuse_overlapping(places: Sequence[Place], *, way_out: str) -> None:
    """Refuse two of ``places`` that are one place, or one inside the other, before any member runs."""
    found = [(place, _real(place.path)) for place in places]
    for index, (place, real) in enumerate(found):
        for other, other_real in found[:index]:
            relation = _overlap(place, real, other, other_real)
            if relation is None:
                continue
            members_ = [item.member for item in (other, place) if item.member is not None]
            if len(members_) == 2 and members_[0] is not members_[1]:
                who = f"sweep members {members_[0].describe()} and {members_[1].describe()}"
                where = f"{other.what} {other.path} and {place.what} {place.path}"
            else:
                who = ("the launch" if not members_ else "the run" if not members_[0].values
                       else f"sweep member {members_[0].describe()}")
                where = " and ".join(f"{_whose(item)} {item.what} {item.path}" for item in (other, place))
            raise CliError(f"{who}: {where} {relation} as a write reaches them, so one would write into the "
                           f"other; {way_out}")


def check_places(places: Sequence[Place], *, way_out: str) -> None:
    """Check a command's inventory before anything is written or submitted."""
    for place in places:
        if place.protect is not None:
            place.protect()
    refuse_overlapping(places, way_out=way_out)
