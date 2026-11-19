from __future__ import annotations

import functools
import os
import tempfile
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.cli.common import Assignment, CliError, is_literal_relative, overlap, quoted, rebased
from ruleforge.cli.judging import MODEL_GROUP, composition_refusal
from ruleforge.cli.schema import MAPPING, Operation, matches
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.record_names import public_blocks
from ruleforge.env import package_path

PRIMARY, OPERATION, USER = "__ruleforge_primary__", "__ruleforge_operation__", "__ruleforge_config__"
SIBLINGS = "__ruleforge_siblings__"
MODELS = "__ruleforge_models_"
#: The search path scheme whose files have their relative paths rebased.
SCHEME = "rfconfig"


@dataclass
class Rebasing:
    """What :class:`RebasingSource` rebases while a composition runs."""

    patterns: Mapping[str, str] = field(default_factory=dict)
    special: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    origins: dict[str, str] = field(default_factory=dict)
    bases: Mapping[Path, Path] = field(default_factory=dict)

    def base(self, directory: Path) -> Path:
        """Return the directory the relative paths of a file in ``directory`` are read from."""
        found = Path(os.path.realpath(directory))
        for tree, base in self.bases.items():
            if found.is_relative_to(os.path.realpath(tree)):
                return Path(base)
        return directory

    def rebase(self, node: DictConfig, directory: Path, where: str) -> None:
        """Rebase ``node``'s literal relative path values against ``directory``."""
        directory = self.base(directory)
        plain = OmegaConf.to_container(node, resolve=False)
        if not isinstance(plain, dict):
            return
        for key, value in _leaves(plain):
            hit = next((pattern for pattern in self.patterns if matches(pattern, key)), None)
            if hit is None or key == "defaults" or key.startswith("defaults."):
                continue
            special = self.special.get(hit, ())
            if self.patterns[hit] == MAPPING and isinstance(value, dict):
                moved: Any = {name: rebased(item, directory, special) for name, item in value.items()}
            else:
                moved = rebased(value, directory, special) if isinstance(value, (str, list)) else value
            if moved != value:
                OmegaConf.update(node, key, moved, merge=False)
            if isinstance(value, str) and "${" in value:
                self.origins.setdefault(hit, where)


def _leaves(plain: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    for name, value in plain.items():
        key = f"{prefix}{name}"
        yield key, value
        if isinstance(value, dict):
            yield from _leaves(value, f"{key}.")


_ACTIVE: list[Rebasing] = []


@functools.lru_cache(maxsize=1)
def _register_source() -> None:
    from hydra._internal.core_plugins.file_config_source import FileConfigSource
    from hydra.core.plugins import Plugins
    from hydra.plugins.config_source import ConfigResult

    class RebasingSource(FileConfigSource):
        """A directory of YAML files whose literal relative path values are rebased against each file's own directory."""

        @staticmethod
        def scheme() -> str:
            return SCHEME

        def load_config(self, config_path: str) -> ConfigResult:
            result = super().load_config(config_path)
            location = Path(os.path.realpath(os.path.join(self.path, self._normalize_file_name(config_path))))
            if config_path.split("/", 1)[0] == MODEL_GROUP:
                from ruleforge.judges.base import as_packaged

                result.config = as_packaged(result.config)
            _ACTIVE[-1].rebase(result.config, location.parent, str(location))
            return result

    Plugins.instance().register(RebasingSource)


@contextmanager
def _rebasing(state: Rebasing) -> Iterator[None]:
    _register_source()
    _ACTIVE.append(state)
    try:
        yield
    finally:
        _ACTIVE.pop()


def _siblings(scratch: Path, directory: Path) -> Path:
    """Return a scratch directory linking each YAML file directly in ``directory``: what a defaults list can name
    without naming a group."""
    place = scratch / SIBLINGS
    place.mkdir()
    for path in sorted(directory.glob("*.yaml")):
        (place / path.name).symlink_to(path.resolve())
    return place


def _model_groups(scratch: Path) -> list[Path]:
    """Return a scratch directory per model config directory, so a bare model name is looked up ahead of the package's."""
    from ruleforge.judges.base import MODEL_CONFIG_DIRS

    packaged = Path(os.path.realpath(package_path("configs") / MODEL_GROUP))
    places = []
    for index, directory in enumerate(MODEL_CONFIG_DIRS):
        if Path(os.path.realpath(directory)) == packaged:
            continue
        place = scratch / f"{MODELS}{index}"
        place.mkdir()
        (place / MODEL_GROUP).symlink_to(Path(directory).resolve(), target_is_directory=True)
        places.append(place)
    return places


def _save(directory: Path, relative: str, node: Any) -> None:
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(node) if not isinstance(node, DictConfig) else node, path)


def parse_overrides(texts: Sequence[str], operation: Operation, cwd: Path) -> list[Assignment]:
    """Return ``-- key=value`` overrides as assignments, literal relative paths made absolute against ``cwd``."""
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import OverrideParseException

    parser = OverridesParser.create()
    assignments = []
    patterns = operation.path_keys()
    for text in texts:
        try:
            parsed = parser.parse_overrides([text])[0]
        except OverrideParseException as error:
            raise CliError(f"-- {text}: not a Hydra override ({error})") from error
        key = parsed.package if parsed.package else parsed.key_or_group
        kind = patterns.get(key)
        override = text
        group = bool(parsed.package)
        given = None if parsed.is_delete() or group else parsed.value()
        # A mapping is merged into its node: it sets its entries (an empty one inside it, that entry), not the node.
        leaves = tuple(f"{key}.{name}" for name, entry in _leaves(given) if not isinstance(entry, dict) or not entry
                       ) if isinstance(given, dict) else ()
        if kind is not None and not parsed.is_delete():
            value = parsed.value()
            special = operation.special.get(key, ())
            if isinstance(value, dict):
                value = {name: rebased(item, cwd, special) for name, item in value.items()}
            elif isinstance(value, (str, list)):
                value = rebased(value, cwd, special)
            prefix = "++" if parsed.is_force_add() else "+" if parsed.is_add() else ""
            override = f"{prefix}{key}={_dict_text(value) if isinstance(value, dict) else quoted(value)}"
        assignments.append(Assignment(key=key, override=override, spelling=text, group=group, leaves=leaves))
    return assignments


def override_key(text: str) -> Optional[str]:
    """The config key a ``-- key=value`` override sets (a group override's package), or None when it is none."""
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import OverrideParseException

    try:
        parsed = OverridesParser.create().parse_overrides([text])[0]
    except OverrideParseException:
        return None
    return parsed.package if parsed.package else parsed.key_or_group


def without_keys(texts: Sequence[str], keys: Sequence[str]) -> list[str]:
    """Return ``texts`` (overrides) less those setting one of ``keys``."""
    return [text for text in texts if override_key(text) not in keys]


def _dict_text(value: Mapping[str, Any]) -> str:
    return "{" + ",".join(f"{name}:{quoted(item)}" for name, item in value.items()) + "}"


def refuse_conflicts(flags: Sequence[Assignment], overrides: Sequence[Assignment],
                     swept: Sequence[Assignment]) -> None:
    """Refuse a key two of the flags and overrides set, and a swept key either sets explicitly."""
    explicit = [(item, item.spelling) for item in flags] + [(item, f"-- {item.spelling}") for item in overrides]
    for index, (item, shown) in enumerate(explicit):
        for earlier, earlier_shown in explicit[:index]:
            key = overlap(earlier, item)
            if key is not None:
                raise CliError(f"{earlier_shown} and {shown} both set {key}: give it once")
    for item in swept:
        for other, _ in explicit:
            key = overlap(other, item)
            if key is not None:
                raise CliError(f"--sweep {item.spelling} sweeps {item.key}, which {other.spelling} also sets"
                               f"{'' if key == item.key else f' ({key})'}: a swept key's values come from the sweep "
                               "alone")


def load_user_config(path: Path, rebasing: Rebasing) -> DictConfig:
    """Return the ``--config`` file, its blocks under today's keys and its literal relative paths rebased against its
    directory."""
    if not path.is_file():
        raise CliError(f"--config {path}: no such file")
    try:
        node = OmegaConf.load(path)
    except (OSError, OmegaConfBaseException, ValueError, yaml.YAMLError) as error:
        raise CliError(f"--config {path} cannot be read as YAML ({error})") from error
    if not isinstance(node, DictConfig):
        raise CliError(f"--config {path} is not a mapping")
    plain = OmegaConf.to_container(node, resolve=False)
    try:
        public = public_blocks(plain)
    except SamplingDesignError as error:
        raise CliError(f"--config {path}: {error}") from error
    if public != plain:
        node = OmegaConf.create(public)
    rebasing.rebase(node, path.parent, str(path))
    return node


def compose(operation: Operation, packaged: Any, *, config_file: Optional[Path] = None,
            layers: Sequence[Sequence[Assignment]] = (), cwd: Optional[Path] = None,
            config_bases: Mapping[Path, Path] = types.MappingProxyType({})) -> DictConfig:
    """Return the composed config: ``packaged``, the ``--config`` file and the ``layers`` of assignments, in that order,
    every path key checked absolute."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.errors import HydraException

    cwd = Path(cwd or os.getcwd())
    rebasing = Rebasing(patterns=operation.path_keys(), special=operation.special, bases=dict(config_bases))
    node = OmegaConf.load(package_path(f"configs/{packaged}")) if isinstance(packaged, str) else packaged
    user = None
    if config_file is not None:
        config_file = Path(rebased(str(config_file), cwd))
        user = load_user_config(config_file, rebasing)
    with tempfile.TemporaryDirectory(prefix="ruleforge-config-") as scratch:
        _save(Path(scratch), f"{OPERATION}.yaml", node)
        defaults = [OPERATION]
        if user is not None:
            _save(Path(scratch), f"{USER}.yaml", user)
            defaults.append(USER)
        _save(Path(scratch), f"{PRIMARY}.yaml", {"defaults": [*defaults, "_self_"]})
        for item in (item for layer in layers for item in layer):
            for relative, file_node in item.files.items():
                _save(Path(scratch), relative, file_node)
        searchpath = []
        if user is not None:
            beside = config_file.parent if operation.judges else _siblings(Path(scratch), config_file.parent)
            searchpath.append(f"{SCHEME}://{beside}")
        if operation.judges:
            searchpath += [f"{SCHEME}://{place}" for place in _model_groups(Path(scratch))]
            searchpath.append(f"file://{package_path('configs')}")
        overrides = [f"hydra.searchpath=[{','.join(quoted(entry) for entry in searchpath)}]",
                     *(item.override for layer in layers for item in layer)]
        try:
            with _rebasing(rebasing), initialize_config_dir(version_base=None, config_dir=scratch):
                cfg = hydra_compose(PRIMARY, overrides=overrides)
        except (HydraException, OmegaConfBaseException) as error:
            refusal = composition_refusal(error, operation.judges)
            raise CliError(refusal or f"the {operation.name} config does not compose: {error}") from error
    check_paths(cfg, operation, rebasing.origins)
    return cfg


def _child(cfg: DictConfig, name: str) -> Any:
    try:
        return OmegaConf.select(cfg, name, throw_on_missing=False)
    except OmegaConfBaseException:
        return None


def _keys(cfg: Any, pattern: str, prefix: str = "") -> list[str]:
    head, _, rest = pattern.partition(".")
    if not isinstance(cfg, DictConfig):
        return []
    names = [str(name) for name in cfg.keys()] if head == "*" else [head] if head in cfg else []
    if not rest:
        return [f"{prefix}{name}" for name in names]
    return [key for name in names for key in _keys(_child(cfg, name), rest, f"{prefix}{name}.")]


def check_paths(cfg: DictConfig, operation: Operation, origins: Mapping[str, str]) -> None:
    """Refuse a path key whose resolved value is not absolute, naming the key, its value and where it was
    written."""
    for pattern, kind in operation.path_keys().items():
        special = operation.special.get(pattern, ())
        for key in _keys(cfg, pattern):
            try:
                value = OmegaConf.select(cfg, key, throw_on_missing=False)
            except OmegaConfBaseException:
                continue  # read where it is used, which names the unset variable or the missing key
            if isinstance(value, (DictConfig, ListConfig)):
                value = OmegaConf.to_container(value, resolve=True)
            values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
            relative = [item for item in values if is_literal_relative(item, special)]
            if relative:
                where = origins.get(pattern, "a flag, an override or an interpolation")
                raise CliError(f"{key} = {value!r} ({where}) is not an absolute path once resolved; write it "
                               "relative to the file that names it, or absolute")
