from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.cli.common import Assignment, CliError, rebased
from ruleforge.cli.schema import JUDGE_PATH_OPTIONS
from ruleforge.judges.base import judge_forms

MODEL_GROUP = "model"
JUDGE_HELP = judge_forms()


def is_named(value: str) -> bool:
    """Return whether ``value`` names a model config of the model group (not a file, not built from its spelling)."""
    return (":" not in value and value != "fake" and "/" not in value and "\\" not in value
            and Path(value).suffix not in (".yaml", ".yml"))


def judge_node(value: str, *, cwd: Path) -> DictConfig:
    """Return the model config ``value`` names, its path options absolute."""
    from ruleforge.judges import JudgeError, resolve

    if is_named(value):
        from ruleforge.judges.base import model_config_names, model_config_path

        named = model_config_path(value)
        if named is None:
            raise CliError(unknown_judge(value, model_config_names()))
        return _loaded(named)
    path = Path(value).expanduser()
    if path.suffix in (".yaml", ".yml") or path.is_file():
        return _loaded(path if path.is_absolute() else cwd / path)
    try:
        spec = resolve(value)
    except (JudgeError, OmegaConfBaseException, TypeError, ValueError) as error:
        raise CliError(f"--judge {value}: {error}") from error
    return OmegaConf.create(spec.options.written())


def unknown_judge(name: str, known: Sequence[str]) -> str:
    """Return the refusal of a judge name that is no model config's."""
    return f"no model config named {name!r} (there are {', '.join(known)}); a judge is {judge_forms()}"


def composition_refusal(error: BaseException, nodes: Sequence[str]) -> Optional[str]:
    """Return what Hydra's refusal to select one of ``nodes``' model configs means, or None for any other refusal: no
    judge given, or a name that is no model config's."""
    missing = getattr(error, "missing_cfg_file", None)
    if isinstance(missing, str) and missing.startswith(f"{MODEL_GROUP}/"):
        return unknown_judge(missing.removeprefix(f"{MODEL_GROUP}/"), getattr(error, "options", None) or ())
    if any(f"You must specify '{MODEL_GROUP}@{node}'" in str(error) for node in nodes):
        return f"no judge given: --judge is required, and takes {judge_forms()}"
    return None


def judge_assignment(value: str, spelling: str, *, node: str, cwd: Path, name: str) -> Assignment:
    """Return the assignment for a judge given by flag or sweep: a config's name selects it from the model group;
    anything else becomes an option of the group and is selected."""
    if is_named(value):
        return Assignment(key=node, override=f"{MODEL_GROUP}@{node}={value}", spelling=spelling, group=True)
    return Assignment(key=node, override=f"{MODEL_GROUP}@{node}={name}", spelling=spelling, group=True,
                      files={f"{MODEL_GROUP}/{name}.yaml": judge_node(value, cwd=cwd)})


def judge_flag(args: argparse.Namespace, name: str, node: str, cwd: Path) -> Optional[Assignment]:
    """Return the assignment ``--<name>`` makes of config node ``node``, when it is given."""
    value = getattr(args, name)
    if value is None:
        return None
    return judge_assignment(value, f"--{name} {value}", node=node, cwd=cwd, name=f"__ruleforge_{name}__")


def engine_gpus(spec: Any) -> int:
    """The GPUs a judge's local engine takes (its tensor-parallel size); none for a judge without one."""
    return int(spec.options.get("tp_size") or 1) if spec.capabilities().local_engine else 0


def _loaded(path: Path) -> DictConfig:
    try:
        node = OmegaConf.load(path)
    except (OSError, OmegaConfBaseException, ValueError, yaml.YAMLError) as error:
        raise CliError(f"--judge {path}: not a model config ({error})") from error
    if not isinstance(node, DictConfig):
        raise CliError(f"--judge {path}: not a model config (a mapping)")
    from ruleforge.judges.base import as_packaged

    node = as_packaged(node)  # its roots read by the library's rule, whether it was named or given by path
    written = OmegaConf.to_container(node, resolve=False)
    for option in JUDGE_PATH_OPTIONS:
        if isinstance(written.get(option), str):
            node[option] = rebased(written[option], path.parent)
    return node


def judge_spec(node: Any) -> Any:
    """Return the judge a composed model config node describes, a batch transport's ``batch_root`` absolute."""
    from ruleforge.judges import JudgeError, JudgeSpec

    try:
        return absolute_batch_root(JudgeSpec.from_config(node))
    except (JudgeError, OmegaConfBaseException, TypeError, ValueError) as error:
        raise CliError(f"the judge's model config is refused: {error}") from error


def absolute_batch_root(spec: Any) -> Any:
    """Return ``spec`` with a batch transport's ``batch_root`` absolute."""
    try:
        given = spec.options.batch_root if spec.capabilities().batch else None
    except OmegaConfBaseException:
        return spec
    if not given:
        return spec
    place = os.path.join(os.getcwd(), os.path.expanduser(str(given)))
    return spec if place == given else spec.with_options(batch_root=place)
