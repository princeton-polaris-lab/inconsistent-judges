from __future__ import annotations

import os
import re
import shlex
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Collection, Iterator, Mapping, Optional, Sequence

from hydra.core.default_element import GroupDefault
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Override
from hydra.errors import HydraException

from ruleforge.core.errors import RuleForgeError
from ruleforge.env import ENV_RESOLVER, PREFIX
from ruleforge.launch.environment import LaunchError
from ruleforge.core.digests import sha256_file
from ruleforge.core.paths import walk_strictly
from ruleforge.core.provenance import code_state

#: A launch id, a job's name, a run id: it becomes a file name and a SLURM job name.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
COMMIT = re.compile(r"[0-9a-f]{7,40}")
#: Keys a launcher sets on each campaign job itself; an argument may not reach them.
RESERVED_OVERRIDES = ("execution.num_shards", "execution.shard_index", "job_root", "hydra.run.dir", "campaign_id",
                      "campaign_root")
#: The only keys of Hydra's own config an argument may set: neither places a write.
HYDRA_TAKEN = ("hydra.verbose", "hydra.job.chdir")
#: Keys that start a sweep without `-m` or say how it is launched.
MULTIRUN_KEYS = ("hydra.mode", "hydra.sweep", "hydra.sweeper", "hydra.launcher")
#: The only Hydra flags a campaign job passes; Hydra's parser reads abbreviations of `-m` too.
CONFIG_FLAGS = ("--config-name", "-cn", "--config-path", "-cp", "--config-dir", "-cd")
ENV_RESOLVERS = ("oc.env", ENV_RESOLVER)
#: Resolvers that parse a string as config when the value is resolved: a read inside is seen only then.
REPARSING_RESOLVERS = ("oc.decode", "oc.create")
#: A switch's spellings, on and off (empty is unset, so off).
SWITCH_ON, SWITCH_OFF = ("1", "true", "yes", "on"), ("0", "false", "no", "off", "")
CAMPAIGN_MARKER = "sampling_manifest.json"
CONFIG_DIR = "configs"


def check_name(value: str, *, what: str) -> str:
    if not NAME.fullmatch(str(value)):
        raise LaunchError(f"{what} {value!r}: letters, digits, '.', '_' and '-', starting with a letter or digit "
                          "(at most 128)")
    return str(value)


def same_commit(first: str, second: str) -> bool:
    """Return whether two possibly abbreviated shas name one commit."""
    length = min(len(first), len(second))
    return length > 0 and first[:length] == second[:length]


def code_record(repo: Path | str) -> dict[str, Any]:
    """Return the checkout's commit, dirty flag and code-state digest."""
    state = code_state(Path(repo))
    if not state["code_revision"]:
        raise LaunchError(f"{repo} is not a git checkout: a launch records the commit it runs")
    return {"commit": state["code_revision"], "dirty": bool(state["code_dirty"]),
            "code_state_sha256": state["code_state_sha256"]}


def check_code(code: Mapping[str, Any], *, expect_commit: Optional[str] = None, allow_dirty: bool = False) -> None:
    """Refuse a checkout at another commit than `expect_commit` or, unless `allow_dirty`, with uncommitted changes."""
    if expect_commit is not None:
        if not COMMIT.fullmatch(expect_commit):
            raise LaunchError(f"--expect-commit {expect_commit!r}: a commit sha, 7 to 40 hex digits")
        if not same_commit(code["commit"], expect_commit):
            raise LaunchError(f"this checkout is at {code['commit']}, not --expect-commit {expect_commit}: launch "
                              "from a worktree of that commit")
    if code["dirty"] and not allow_dirty:
        raise LaunchError("this checkout has uncommitted changes, which its jobs would run and record as dirty: "
                          "commit them, or pass --allow-dirty")


def _launcher_sets(reserved: Sequence[str]) -> str:
    return (f"the launcher sets {', '.join(reserved)} on each job itself, which no argument reaches, nor a dict above "
            "one of them (the config's root included) or a key below one, and submits each job as one run (no -m, and "
            "no hydra.mode, sweep or launcher)")


def reaches(key: str, names: Sequence[str]) -> bool:
    """Return whether a config key is one of `names`, a dict above one or a key below one."""
    return any(key == name or name.startswith(f"{key}.") or key.startswith(f"{name}.") for name in names)


def _reserved(key: str, reserved: Sequence[str]) -> bool:
    return (key == "" or reaches(key, reserved)
            or any(key == multirun or key.startswith(f"{multirun}.") for multirun in MULTIRUN_KEYS))


def parse_override(argument: str) -> Override:
    """Return an override as Hydra's own parser reads it; refused when Hydra could not read it."""
    try:
        return OverridesParser.create().parse_override(argument)
    except HydraException as error:
        raise LaunchError(f"{argument!r} is not an override Hydra reads: {error}") from None


def placement(group: str, package: Optional[str] = None, *, parent: str = "") -> str:
    """Return where Hydra merges a config group's option, by Hydra's own package rules."""
    # no option name: Hydra substitutes `_name_` only below version_base 1.2
    default = GroupDefault(group=group, package=package)
    default.update_parent(parent, placement(parent) if parent else "")
    return default.get_final_package()


def _keys(override: Override, config_dir: str) -> list[str]:
    key = override.key_or_group
    if not key:
        return [""]
    return list(dict.fromkeys([key.replace("/", "."), placement(key, override.package),
                               placement(key, override.package, parent=config_dir)]))


def override_keys(argument: str, *, config_dir: str = "") -> list[str]:
    """Return the config keys an override reaches: the key it sets and where a config group's option is merged."""
    return _keys(parse_override(argument), config_dir)


def hydra_arguments(arguments: Sequence[str]) -> tuple[str, list[str], list[str]]:
    """Return a job's Hydra arguments as Hydra's parser reads them: the config name, the overrides and the config flags;
    any other flag is refused."""
    config_name, overrides, flags, rest = "", [], [], list(arguments)
    while rest:
        argument = rest.pop(0)
        if not argument.startswith("-"):
            overrides.append(argument)
            continue
        flag, equals, value = argument.partition("=")
        if flag not in CONFIG_FLAGS:
            raise LaunchError(
                f"{argument!r}: a job's Hydra flags are {', '.join(CONFIG_FLAGS)} only, and the launcher "
                "submits each job as one run (no -m, in any spelling Hydra's parser reads)"
            )
        if not equals and rest and not rest[0].startswith("-"):
            value = rest.pop(0)
        if flag in ("--config-name", "-cn"):
            config_name = value
        else:
            flags.append(flag)
    return config_name, overrides, flags


def check_overrides(arguments: Sequence[str], *, reserved: Sequence[str] = RESERVED_OVERRIDES) -> list[str]:
    """Return a job's Hydra arguments, refused when one sets what the launcher sets, asks for a multirun, gives a value
    to a key of Hydra's own (but `HYDRA_TAKEN`) or reads a config from outside the checkout."""
    config_name, overrides, flags = hydra_arguments(arguments)
    check_config_source(config_name, flags)
    sets = _launcher_sets(reserved)
    config_dir = config_name.rpartition("/")[0]
    primary = placement(config_dir) if config_dir else ""
    if primary and _reserved(primary, reserved):
        raise LaunchError(f"--config-name {config_name!r}: Hydra merges a config in {config_dir}/ at {primary!r}; "
                          f"{sets}")
    for argument in overrides:
        override = parse_override(argument)
        if override.is_sweep_override():
            raise LaunchError(f"{argument!r} is a sweep, which Hydra runs only as a multirun; {sets}")
        reached = [key for key in _keys(override, config_dir) if _reserved(key, reserved)]
        if reached:
            raise LaunchError(f"{argument!r} reaches {reached[0] or 'the root'!r}; {sets}")
        if _hydra_key(override.key_or_group):
            raise LaunchError(f"{argument!r} gives Hydra's {override.key_or_group!r} a value: of Hydra's own keys a "
                              f"job's arguments set {' and '.join(HYDRA_TAKEN)} only, since every other places the "
                              "job's writes outside its run directory (its saved config, its log files and their "
                              "names), sets the environment its places resolve under, the directories its configs are "
                              "read from or code run at its start, or rewrites Hydra's record of the run, where no "
                              "check of the launch looks; a committed option of a Hydra config group "
                              "(hydra/job_logging=NAME) is taken")
    return list(arguments)


def check_config_source(config_name: str, flags: Sequence[str]) -> None:
    """Refuse a config read from anywhere but the checkout's `configs/`."""
    if flags:
        raise LaunchError(f"{', '.join(flags)}: a job's config is one of {CONFIG_DIR}/, named by --config-name")
    if config_name.startswith("/"):
        raise LaunchError(f"--config-name {config_name!r}: Hydra reads a name led by '/' as a path ('//dir/x' is the "
                          f"file /dir/x.yaml, outside {CONFIG_DIR}/); a job's config is one of {CONFIG_DIR}/, named "
                          "by its path there with no leading '/'")


class _Computed(RuleForgeError):
    """A name an environment read computes, which no parse can tell."""


def _computed(*args: Any, **kwargs: Any) -> Any:
    raise _Computed()


def env_reads(value: str) -> tuple[set[str], bool]:
    """Return the environment variables a config value reads, as OmegaConf parses it, and whether it may read a name it
    computes."""
    from omegaconf.errors import GrammarParseError
    from omegaconf.grammar.gen.OmegaConfGrammarParser import OmegaConfGrammarParser
    from omegaconf.grammar_parser import parse
    from omegaconf.grammar_visitor import GrammarVisitor

    try:
        pending = [parse(value)]
    except GrammarParseError:
        return set(), False
    names: set[str] = set()
    computed = False
    while pending:
        node = pending.pop()
        if isinstance(node, OmegaConfGrammarParser.InterpolationResolverContext):
            resolver, arguments = node.resolverName().getText(), node.sequence()
            if node.resolverName().interpolation() or resolver in REPARSING_RESOLVERS:
                computed = True
            elif resolver in ENV_RESOLVERS and arguments is not None:
                first = arguments.getChild(0)
                try:
                    name = (GrammarVisitor(_computed, _computed, None).visit(first)
                            if isinstance(first, OmegaConfGrammarParser.ElementContext) else "")
                except Exception:
                    computed = True
                else:
                    if resolver == "oc.env":
                        names |= set() if name is None else {str(name)}
                    else:
                        names |= {PREFIX + str(name)}
        pending += list(getattr(node, "children", None) or ())
    return names, computed


OC_ENV_UNSET = re.compile(r"Environment variable '(.+)' not found")


def unset_variable(error: BaseException) -> Optional[str]:
    """Return the environment variable whose absence failed a config value's resolution, or None."""
    from ruleforge.env import EnvUnsetError

    rf_unset = re.compile(rf"({re.escape(PREFIX)}\S+) is unset: ")
    seen: set[int] = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, EnvUnsetError) and (found := rf_unset.match(str(current))):
            return found.group(1)
        if isinstance(current, KeyError) and current.args and isinstance(current.args[0], str) and (
                found := OC_ENV_UNSET.fullmatch(current.args[0])):
            return found.group(1)
        current = current.__cause__ or current.__context__
    return None


def check_exports(cfg: Any, env: Sequence[str], *, what: str) -> None:
    """Refuse an export of a variable the job's composed config reads: the launch checks the config under this
    environment, so export it where the launch runs instead."""
    from omegaconf import OmegaConf

    read: set[str] = set()
    computing: list[str] = []
    pending = [OmegaConf.to_container(cfg, resolve=False)]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending += list(value.values())
        elif isinstance(value, list):
            pending += value
        elif isinstance(value, str) and "${" in value:
            names, computed = env_reads(value)
            read |= names
            computing += [value] if computed else []
    given = sorted({item.partition("=")[0] for item in env})
    exported = sorted(set(given) & read)
    if exported:
        raise LaunchError(f"--env {', '.join(exported)}: {what}'s config reads it, and the launch composes that "
                          "config, and checks the places it names, under this environment, where the job would "
                          "resolve it under the export: export it where the launch runs instead")
    if computing and given:
        raise LaunchError(f"--env {', '.join(given)}: {what}'s config reads a variable whose name it computes "
                          f"({min(computing)!r}), which may be one of these, and the launch composes that config, and "
                          "checks the places it names, under this environment: export them where the launch runs "
                          "instead, or name the variable in the config")


def _hydra_key(key: str) -> bool:
    return ((key == "hydra" or key.startswith("hydra."))
            and not any(key == taken or key.startswith(f"{taken}.") for taken in HYDRA_TAKEN))


def compose(repo: Path | str, arguments: Sequence[str], *, default_config: str, env: Sequence[str] = (),
            what: str = "a job") -> Any:
    """Return the config Hydra composes for a job's arguments from the checkout's `configs/`; a config from outside them,
    one that does not compose, or one whose files the exports would change is refused."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf.errors import OmegaConfBaseException

    config_name, overrides, flags = hydra_arguments(arguments)
    check_config_source(config_name, flags)
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(Path(repo).resolve() / CONFIG_DIR), version_base=None):
            loader, name = GlobalHydra.instance().config_loader(), config_name or default_config
            defaults = check_composed_files(loader, name, overrides)
            check_exported_defaults(loader, name, overrides, defaults, env, what=what)
            return hydra_compose(config_name=name, overrides=list(overrides))
    except (HydraException, OmegaConfBaseException) as error:
        raise LaunchError(f"{shlex.join(arguments)}: the config does not compose: {type(error).__name__}: "
                          f"{error}") from None


def check_composed_files(loader: Any, config_name: str, overrides: Sequence[str]) -> list[Any]:
    """Refuse a composition that reads a config file from outside the directories Hydra searches; return the defaults
    list."""
    from hydra.types import RunMode

    defaults = loader.compute_defaults_list(config_name, list(overrides), RunMode.RUN).defaults
    for default in defaults:
        path = default.config_path or ""
        if path.startswith("/") or ".." in path.split("/"):
            raise LaunchError(f"the config {path!r}{' (the primary config)' if default.primary else ''} is read from "
                              f"outside {CONFIG_DIR}/ and Hydra's own configs: a job composes only the checkout's "
                              "committed configs; name the option or config by its name there")
    return defaults


@contextmanager
def _exported(env: Sequence[str]) -> Iterator[None]:
    pairs = [item.partition("=")[::2] for item in env]
    saved = {name: os.environ.get(name) for name, _ in pairs}
    try:
        os.environ.update(dict(pairs))
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check_exported_defaults(loader: Any, config_name: str, overrides: Sequence[str], defaults: Sequence[Any],
                            env: Sequence[str], *, what: str) -> None:
    """Refuse a job's exports under which its config's defaults list would differ from `defaults`."""
    if not env:
        return
    from hydra.types import RunMode
    from omegaconf.errors import OmegaConfBaseException

    names = ", ".join(sorted({item.partition("=")[0] for item in env}))
    try:
        with _exported(env):
            again = loader.compute_defaults_list(config_name, list(overrides), RunMode.RUN).defaults
    except (HydraException, OmegaConfBaseException) as error:
        raise LaunchError(f"--env {names}: {what}'s config does not compose under the exports, as the job would "
                          f"compose it ({type(error).__name__}: {str(error).splitlines()[0]}), where the launch "
                          "composed it under this environment: a config group's choice reads them; export them where "
                          "the launch runs instead") from None
    here, there = ([f"{item.config_path} at {item.package}" if item.package else str(item.config_path)
                    for item in listed] for listed in (defaults, again))
    if here != there:
        job = ", ".join(dict.fromkeys(item for item in there if item not in here)) or "the same files in another order"
        launch = ", ".join(dict.fromkeys(item for item in here if item not in there)) or "them"
        raise LaunchError(f"--env {names}: {what}'s config chooses its files by the environment (a config group's "
                          f"choice reads it): under the exports the job would compose {job} where the launch "
                          f"composed, and checked the places of, {launch}; export them where the launch runs "
                          "instead, or name the choice in the arguments")


def switch(value: str, *, what: str) -> bool:
    """Return a switch's value; anything but the on and off spellings is refused."""
    if value in SWITCH_ON:
        return True
    if value in SWITCH_OFF:
        return False
    raise LaunchError(f"{what} {value!r} is not a switch: {'|'.join(SWITCH_ON)} or {'|'.join(SWITCH_OFF[:-1])}")


def check_campaign_root(root: str, *, holds_campaign: bool) -> None:
    """Refuse a campaign root named explicitly that holds no campaign yet, unless a fresh one is meant."""
    if not holds_campaign:
        raise LaunchError(f"--campaign-root {root} holds no campaign (no {CAMPAIGN_MARKER}): a mistyped root would "
                          "start a fresh campaign there; pass --new-root if a fresh root is meant")


def check_population(path: Optional[Path], expected: Optional[str]) -> Optional[str]:
    """Return a population's approved digest; a population without its digest, or the reverse, is refused."""
    if path is None and expected is None:
        return None
    if expected is None:
        raise LaunchError("--population needs --expect-population-sha256, its approved digest")
    if path is None:
        raise LaunchError("--expect-population-sha256 without --population would check nothing")
    if not path.is_file():
        raise LaunchError(f"--population {path} is not a file")
    actual = sha256_file(path)
    if actual != expected:
        raise LaunchError(f"--population {path}: its digest {actual} is not the expected {expected}")
    return actual


def check_packing(pack_draws: bool, max_prompts: Optional[int], max_draws: Optional[int]) -> None:
    """Refuse packing bounds given while packing is off."""
    if not pack_draws and (max_prompts is not None or max_draws is not None):
        raise LaunchError("--pack-max-prompts / --pack-max-draws tune packing, which only --pack-draws turns on")


def check_spelled(written: Path | str, recorded: Path, *, what: str) -> None:
    """Refuse a root its jobs reach through a symlink: as the jobs spell it, it must be the root the record names."""
    if os.path.abspath(written) != os.path.abspath(recorded):
        raise LaunchError(f"{what} {written}, as its jobs spell it, is not {recorded}, where the launch's record "
                          "and a pull name it: it is reached through a symlink, and its jobs would write beside and "
                          "below the link's spelling. Name the root itself (below the artifact root, no link on its "
                          "way)")


def refuse_linked_way(base: Path, destination: Path, *, what: str, below: bool = False, roots: bool = False,
                      own: Collection[Path] = ()) -> None:
    """Refuse `destination` when a directory on the way from `base` to it, or it, is a symlink; with `below`, anything
    below it too; with `roots`, another campaign's root on the way or below it too. Nothing is written."""
    is_root = _campaign_root_test(own) if roots else (lambda path: False)
    current = base
    for part in destination.relative_to(base).parts:
        current = current / part
        if current.is_symlink():
            raise LaunchError(f"{what} {destination}: {current}, on its way, is a symlink: a write would follow it to "
                              f"{os.path.realpath(current)}; replace it with a directory of its own")
        if not current.exists():
            return
        if is_root(current):
            raise LaunchError(f"{what} {destination}: {current}, on its way, is another campaign's root, which a write "
                              "of this one would land in; move it out of the way")
    if not below or not destination.is_dir():
        return
    try:
        for path, entry in sorted(walk_strictly(destination, missing_ok=True)):
            if stat.S_ISLNK(entry.st_mode):
                raise LaunchError(f"{what} {destination}: {path}, below it, is a symlink: a write there would follow "
                                  f"it to {os.path.realpath(path)}; replace it with a file or directory of its own")
            if stat.S_ISDIR(entry.st_mode) and is_root(path):
                raise LaunchError(f"{what} {destination}: {path}, below it, is another campaign's root, which a write "
                                  "of this one would land in; move it out of the way")
    except OSError as error:
        raise LaunchError(f"{what} {destination} cannot be walked whole ({error}): an entry below it that cannot be "
                          "listed or examined may be, or hold, a link a write would follow; make it readable"
                          ) from error


def _campaign_root_test(own: Collection[Path]) -> Any:
    from ruleforge.evaluate.api import _is_campaign_root

    owned = {Path(path) for path in own}
    return lambda path: Path(path) not in owned and _is_campaign_root(Path(path))
