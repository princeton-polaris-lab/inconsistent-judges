from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pytest
from omegaconf import OmegaConf

from ruleforge.cli import evaluate, grpo, optimize
from ruleforge.cli.schema import (
    JUDGE_PATH_OPTIONS,
    MAPPING,
    PATH,
    REQUIRED,
    SEQUENCE,
    Operation,
    Parameter,
    annotations_of,
    bind,
    defaults_of,
    matches,
    path_kind,
)
from ruleforge.env import package_path


@pytest.mark.parametrize("annotation, kind", [
    (Path, PATH), (Union[Path, str], PATH), (Optional[Union[Path, str]], PATH), ("Path | str", PATH),
    (Sequence[Union[Path, str]], SEQUENCE), (Union[Path, str, Sequence[Union[Path, str]]], SEQUENCE),
    (Mapping[str, Union[Path, str]], MAPPING), (Union[Mapping[str, Path], None], MAPPING), (str, None), (int, None),
    (Sequence[str], None), (Mapping[str, str], None), (Any, None), ("Optional[int]", None), (list[Path], SEQUENCE),
])
def test_an_annotation_says_what_path_it_holds(annotation, kind):
    assert path_kind(annotation) == kind


def _paths_of(target: Any) -> dict[str, str]:
    return {name: kind for name, annotation in annotations_of(target).items()
            if (kind := path_kind(annotation)) is not None and name in defaults_of(target)}


def test_evaluate_binds_every_path_parameter_of_the_api_it_calls():
    from ruleforge.evaluate.api import ConditionSettings, SamplingSettings
    from ruleforge.evaluate.api import evaluate as evaluate_api
    from ruleforge.judges import JudgeSpec
    from ruleforge.evaluate.packing import ExecutionSettings

    operation = evaluate.operation()
    bound = {(parameter.target, parameter.name) for parameter in operation.parameters}
    for target in (evaluate_api, ConditionSettings, SamplingSettings, ExecutionSettings):
        assert {(target, name) for name in _paths_of(target)} <= bound, target
    assert set(_paths_of(JudgeSpec.local)) <= set(JUDGE_PATH_OPTIONS)
    assert operation.path_keys() == {
        "rule_set_path": PATH, "scenario_set_path": PATH, "campaign_root": PATH, "rule_levels_path": PATH,
        **{f"judge_model.{option}": PATH for option in JUDGE_PATH_OPTIONS}}


def test_optimize_binds_its_ruleset_its_seed_files_and_its_directory():
    from ruleforge.optimize import SEED_PATH_KEYS
    from ruleforge.optimize.api import optimize as optimize_api

    operation = optimize.operation()
    assert {name for name in _paths_of(optimize_api)} == {"rules", "seeds", "out"}
    assert {parameter.name for parameter in operation.parameters} == {"rules", "seeds", "out"}
    assert operation.path_keys() == {
        "rule_amendment.initial_rule_set_path": PATH, "out": PATH,
        **{f"scenario_generation.{key}": PATH for key in SEED_PATH_KEYS.values()},
        **{f"{node}.{option}": PATH for node in ("judge_model", "agent_model") for option in JUDGE_PATH_OPTIONS}}


@pytest.mark.parametrize("name, expected", [
    ("rows", {"anchors": SEQUENCE, "scenarios": PATH, "rules": PATH, "model": PATH, "out": PATH,
              "protected": SEQUENCE, "require_protected": SEQUENCE}),
    ("anchors build", {"bank": SEQUENCE, "full_set": SEQUENCE, "top_up": SEQUENCE, "reference": PATH, "rules": PATH,
                       "out": PATH, "population": PATH, "blocks": PATH}),
    ("anchors route", {"bank": SEQUENCE, "reference": PATH, "rules": PATH, "population": PATH, "out": PATH,
                       "blocks": PATH}),
    ("merge", {"base_dir": PATH, "adapter_dir": PATH, "out_dir": PATH}),
])
def test_a_function_command_s_path_keys_are_its_api_function_s_path_parameters(name, expected):
    command = grpo.COMMANDS[name]
    assert command.operation().path_keys() == expected
    assert _paths_of(command.function()).keys() <= expected.keys()
    config = OmegaConf.to_container(command.config())
    for key, (target, parameter) in command.keys().items():
        default = command.defaults.get(key, defaults_of(target)[parameter])
        head, _, rest = key.partition(".")
        assert (config[head][rest] if rest else config[key]) == grpo._plain(default)


def test_train_declares_the_path_keys_of_the_config_run_grpo_reads():
    packaged = OmegaConf.to_container(OmegaConf.load(package_path("configs/grpo/base.yaml")), resolve=False)
    operation = grpo.train_operation()
    for key in operation.path_keys():
        node = packaged
        for part in key.split("."):
            node = node[part]
    assert operation.special == {"resume_from_checkpoint": ("latest",)}


def test_every_path_option_of_a_packaged_model_config_is_a_judge_path_option():
    for path in sorted((package_path("configs") / "model").glob("*.yaml")):
        options = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        named = {name for name in options if name.endswith(("_path", "_root"))}
        assert named <= set(JUDGE_PATH_OPTIONS), path.name


def test_a_settings_dataclass_and_a_function_bind_alike():
    @dataclasses.dataclass
    class Settings:
        out: Path
        items: list = dataclasses.field(default_factory=list)
        label: str = "x"

    def function(rules: Union[Path, str], *args: Any, count: int = 1, **kwargs: Any) -> None:
        return None

    assert defaults_of(Settings) == {"out": REQUIRED, "items": [], "label": "x"}
    assert defaults_of(function) == {"rules": REQUIRED, "count": 1}
    bound = bind(function, prefix="call.", exclude=("count",))
    assert [(parameter.key, parameter.kind) for parameter in bound] == [("call.rules", PATH)]
    assert Operation("x", bind(Settings), declared={"extra": PATH}).path_keys() == {"out": PATH, "extra": PATH}
    assert Operation("x", bind(function)).callables() == (function,)
    assert Parameter("seed", function, "rules", entry=True).kind == PATH
    assert Parameter("label", Settings, "label").kind is None


def test_an_annotation_only_a_type_checker_resolves_is_read_as_written():
    def function(out=None, other=1):
        return None

    function.__annotations__ = {"out": "Unknown | Path", "other": "Unknown"}
    assert annotations_of(function) == {"out": "Unknown | Path", "other": "Unknown"}
    assert Parameter("out", function, "out").kind == PATH and Parameter("other", function, "other").kind is None


@pytest.mark.parametrize("pattern, key, found", [
    ("judge_model.model_path", "model_path", True), ("judge_model.model_path", "judge_model.model_path", True),
    ("judge_model.model_path", "agent_model.model_path", False), ("out", "foo.out", False),
    ("ruleforge.rulesets.*.path", "rulesets.a.path", True), ("ruleforge.rulesets.*.path", "rulesets.a.other", False),
])
def test_a_path_pattern_matches_a_key_written_below_its_place(pattern, key, found):
    assert matches(pattern, key) is found
