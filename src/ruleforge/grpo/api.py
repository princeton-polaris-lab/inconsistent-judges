from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from ruleforge.evaluate.api import EXECUTION_FNAME
from ruleforge.grpo.grpo_data import (
    GRPO_ROWS_FNAME,
    MANIFEST_FNAME,
    PROMPT_DRAWS,
    REPEAT_CAP,
    REPLAY_ROWS_FNAME,
    SCENARIOS_FNAME,
    CampaignPrompts,
    build_grpo_data,
)
from ruleforge.judges.completion_format import GrpoDataError
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.identity import ConditionSpec
from ruleforge.scenarios import ScenarioSet
from ruleforge.tasks import FORCED_JSON, Task, TaskError, TaskSettings, check_task_identity, load_task, task_locator


@dataclass(frozen=True)
class PromptBuilder:
    """A condition's judge prompts, rendered by its task under its settings."""

    task: Task
    settings: TaskSettings
    locator: str

    @classmethod
    def for_condition(cls, condition: ConditionSpec, root: Path, *, task: Any = None) -> "PromptBuilder":
        """Return the builder of `condition`'s prompts: `task` when given, else the task its root locates; refused unless
        the task is the one recorded and its style answers in forced JSON."""
        where = f"condition {condition.condition_id}"
        try:
            if task is None:
                record = Path(root) / condition.condition_id / EXECUTION_FNAME
                spec = json.loads(record.read_text(encoding="utf-8"))["task_locator"] if record.is_file() \
                    else condition.task
            else:
                spec = task
            found = load_task(spec)
            check_task_identity(found, task_name=condition.task, response_contract=condition.response_contract,
                                prompt_style=condition.prompt_style, where=where)
            settings = TaskSettings(condition.prompt_style, condition.individual_rule_context, condition.judge_type)
            found.check(settings)
            contract = found.contract(condition.prompt_style)
        except (TaskError, OSError, ValueError, KeyError) as error:
            raise GrpoDataError(f"{where}: its task cannot render its prompts ({error})") from error
        if contract != FORCED_JSON:
            raise GrpoDataError(
                f"{where} answers style {condition.prompt_style!r} under the {contract!r} contract: GRPO scores "
                f"{FORCED_JSON!r} answers alone (its reward reads a full-set judgment beside a complete rule map), so "
                "its rows are built from a forced-JSON condition's anchors")
        return cls(task=found, settings=settings, locator=task_locator(spec))

    @property
    def individual_rule_context(self) -> str:
        return self.settings.rule_context

    @property
    def reasoning_prompt_style(self) -> str:
        return self.settings.prompt_style

    def task_identity(self, judge_type: str) -> dict[str, Any]:
        return self.task.identity(dataclasses.replace(self.settings, judge_type=judge_type))

    def build_eval_prompt(self, rule_text: str, scenario: Mapping[str, Any], judge_type: str, *,
                          context_rules: Optional[list[str]] = None) -> str:
        """Render the prompt, once `scenario` carries every field the task reads."""
        try:
            self.task.check_scenarios([scenario])
        except TaskError as error:
            raise GrpoDataError(f"task {self.task.identifier!r} cannot render a scenario of this build: it lacks a "
                                f"field the task reads ({error})") from error
        return self.task.render(rule_text, scenario, dataclasses.replace(self.settings, judge_type=judge_type),
                                context_rules=context_rules)

    def prompts(self, rules: Ruleset) -> CampaignPrompts:
        """Return every component prompt of `rules` under this builder."""
        return CampaignPrompts.build(task_spec=self, rules=rules.rows(), judge_type=self.settings.judge_type)

    def rendering(self) -> dict[str, Any]:
        """Return what the rows' manifest records of the task."""
        return {"task": self.task.identity(self.settings), "task_locator": self.locator}


@dataclass(frozen=True)
class RowsBuild:
    """A built row set: its directory, files, anchor condition, ruleset and marker digests, from its manifest."""

    out: Path
    train: Path
    replay: Path
    scenarios: Path
    manifest: Path
    anchor_condition: str
    rules: Path
    rules_sha256: str
    markers: Mapping[str, str] = field(hash=False)
    payload: Mapping[str, Any] = field(default_factory=dict, hash=False, repr=False, compare=False)

    @classmethod
    def read(cls, out: Path | str) -> "RowsBuild":
        """Read the build under `out` from its manifest."""
        directory = Path(out)
        payload = json.loads((directory / MANIFEST_FNAME).read_text(encoding="utf-8"))
        rules = payload["rules"]
        return cls(out=directory, train=directory / GRPO_ROWS_FNAME, replay=directory / REPLAY_ROWS_FNAME,
                   scenarios=directory / SCENARIOS_FNAME, manifest=directory / MANIFEST_FNAME,
                   anchor_condition=str(payload["anchor_condition"]), rules=Path(rules["path"]),
                   rules_sha256=str(rules["sha256"]), markers=dict(payload["markers"]), payload=payload)

    def train_overrides(self) -> dict[str, str]:
        """Return the training config's data keys for this build."""
        return {"data.train": str(self.train), "data.replay": str(self.replay), "data.rules": str(self.rules),
                "data.rules_sha256": self.rules_sha256, "data.anchor_condition": self.anchor_condition}


Pathish = Union[Path, str]


def _scenario_file(scenarios: Union[ScenarioSet, Pathish]) -> Path:
    if not isinstance(scenarios, ScenarioSet):
        return Path(scenarios)
    try:
        fresh = ScenarioSet.load(scenarios.path, expect_sha256=scenarios.sha256)
    except (OSError, ValueError) as error:
        raise GrpoDataError(f"scenario set {scenarios.path} is not the file it names ({error}); load it again from "
                            "the file (ScenarioSet.load)") from error
    if fresh.manifest != scenarios.manifest or fresh.rows != tuple(scenarios.rows):
        raise GrpoDataError(f"scenario set {scenarios.path} holds rows other than its file's (changed in memory since "
                            "it was loaded, or built by hand): what is built from it is recorded by the file; load "
                            "it again from the file (ScenarioSet.load)")
    return fresh.path


def rows(anchors: Union[Pathish, Sequence[Pathish]], scenarios: Union[ScenarioSet, Pathish], rules: Pathish,
         model: Pathish, out: Pathish, *, task: Any = None, protected: Sequence[Pathish] = (),
         require_protected: Sequence[Pathish] = (), allow_no_held_out: bool = False,
         pins: Union[Mapping[str, str], Sequence[Any]] = (), prompt_draws: int = PROMPT_DRAWS,
         repeat_cap: int = REPEAT_CAP, class_a_share_cap: Optional[float] = None, **kw: Any) -> RowsBuild:
    """Build the GRPO rows and the replay rows under `out` and return them, from `scenarios` that carry a disjointness
    marker, with `protected` the files the second pass adds and `pins` protected files the marker must also cover."""
    if isinstance(pins, Mapping):
        pinned = dict(pins)
    elif not pins:
        pinned = {}
    else:
        raise GrpoDataError(f"pins name protected files by digest, {{name: sha256}}; got {pins!r}")
    build_grpo_data(
        anchors=anchors, scenarios=_scenario_file(scenarios), rules=rules, model=model, out=out, protected=protected,
        required_protected=require_protected, allow_no_held_out=allow_no_held_out, pinned_protected=pinned,
        prompt_draws=prompt_draws, repeat_cap=repeat_cap, class_a_share_cap=class_a_share_cap,
        task_spec_for=lambda condition, root: PromptBuilder.for_condition(condition, root, task=task), **kw)
    return RowsBuild.read(out)


#: The training run's base config: the package's when packaged, else the repository's.
BASE_CONFIG_PACKAGED = ("configs", "grpo", "base.yaml")
BASE_CONFIG_REPOSITORY = ("configs", "training", "grpo", "base.yaml")
#: What the library's base config sets apart from the paper's: no reporting unless asked.
LIBRARY_DEFAULTS: Mapping[str, Any] = {"grpo.report_to": []}
LAUNCHER_KEY = "hydra"


def base_config_path() -> Path:
    """Return the base config `train` composes when given none: the packaged one, else the repository's."""
    from importlib import resources

    packaged = resources.files("ruleforge").joinpath(*BASE_CONFIG_PACKAGED)
    if packaged.is_file():
        return Path(str(packaged))
    from ruleforge.core.paths import repository_root

    return repository_root().joinpath(*BASE_CONFIG_REPOSITORY)


def training_config(config: Union[None, Mapping[str, Any], Pathish] = None, *,
                    overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return the complete resolved config `train` runs: `config` (None: the base config) with `overrides` applied by
    dotted key; an unknown key without `+`, or a value left `???`, is refused."""
    from omegaconf import OmegaConf, open_dict
    from omegaconf.errors import OmegaConfBaseException

    if config is None:
        composed = OmegaConf.load(base_config_path())
        for key, value in LIBRARY_DEFAULTS.items():
            OmegaConf.update(composed, key, value, merge=False)
    elif isinstance(config, Mapping):
        composed = OmegaConf.create(dict(config))
    else:
        composed = OmegaConf.load(Path(config))
        if "defaults" in composed:
            raise GrpoDataError(f"{config} composes others (its defaults list): compose it first, with the config's "
                                "launcher, and pass the composed mapping")
    composed.pop(LAUNCHER_KEY, None)
    OmegaConf.set_struct(composed, True)
    for given, value in (overrides or {}).items():
        key, adds = given.removeprefix("+"), given.startswith("+")
        try:
            if adds:
                with open_dict(composed):
                    OmegaConf.update(composed, key, value, merge=False, force_add=True)
            else:
                OmegaConf.update(composed, key, value, merge=False)
        except OmegaConfBaseException as error:
            detail = str(error).splitlines()[0]
            raise GrpoDataError(f"override {given!r}: {detail} (a key the config lacks takes a '+')") from error
    missing = sorted(_missing(OmegaConf.to_container(composed, resolve=False)))
    if missing:
        raise GrpoDataError(f"the training config leaves {missing} unset (???): give each as an override")
    return OmegaConf.to_container(composed, resolve=True)


def _missing(node: Any, prefix: str = "") -> list[str]:
    if isinstance(node, Mapping):
        return [key for name, value in node.items() for key in _missing(value, f"{prefix}{name}.")]
    if isinstance(node, list):
        return [key for index, value in enumerate(node) for key in _missing(value, f"{prefix}{index}.")]
    return [prefix[:-1]] if node == "???" else []


@dataclass(frozen=True)
class TrainRun:
    """A training run: its output directory and the config it ran under."""

    output_dir: Path
    config: Mapping[str, Any] = field(hash=False, repr=False, compare=False)

    def checkpoint(self, step: int) -> Path:
        """Return the run's save at `step`, refused unless finished."""
        from ruleforge.grpo.run_files import SAVE_FILES

        directory = self.output_dir / f"checkpoint-{step}"
        absent = [name for name in SAVE_FILES if not (directory / name).is_file()]
        if absent:
            raise GrpoDataError(f"{directory} is not a finished save (it lacks {absent})")
        return directory

    def checkpoints(self) -> list[int]:
        """Return the steps of the run's finished saves, in order."""
        from ruleforge.grpo.run_files import SAVE_FILES, checkpoint_steps

        return [step for step in checkpoint_steps(self.output_dir)
                if all((self.output_dir / f"checkpoint-{step}" / name).is_file() for name in SAVE_FILES)]

    def merge_expectation(self, step: Optional[int] = None) -> Any:
        """Return what a merge of the run's save at `step` (else the newest) must find."""
        from ruleforge.grpo.merging import MergeExpectation
        from ruleforge.grpo.run_files import PROVENANCE_FNAME

        steps = self.checkpoints()
        if not steps:
            raise GrpoDataError(f"{self.output_dir} holds no finished save: a merge expectation reads one's preset")
        save = self.checkpoint(steps[-1] if step is None else step)
        record = json.loads((save / PROVENANCE_FNAME).read_text(encoding="utf-8"))
        preset = record["run"]["identity"]["preset"]
        return MergeExpectation(rank=int(preset["r"]), alpha=float(preset["alpha"]),
                                base_model=Path(str(self.config["model"]["path"])).name,
                                target_modules=tuple(preset["target_modules"]),
                                module_scope=preset.get("module_scope"))


def train(config: Union[None, Mapping[str, Any], Pathish] = None, *,
          overrides: Optional[Mapping[str, Any]] = None) -> TrainRun:
    """Train a run under `config` and `overrides` and return it; with `preflight_only`, check everything and train
    nothing."""
    from ruleforge.grpo.trainer import run_grpo

    resolved = training_config(config, overrides=overrides)
    run_grpo(resolved)
    return TrainRun(output_dir=Path(resolved["output_dir"]), config=resolved)


def _saved_run_name(adapter: Pathish) -> Optional[str]:
    from ruleforge.grpo.run_files import PROVENANCE_FNAME

    provenance = Path(adapter) / PROVENANCE_FNAME
    if not provenance.is_file():
        return None
    record = json.loads(provenance.read_text(encoding="utf-8"))
    return ((record.get("run") or {}).get("config") or {}).get("run_name")


def merge(base: Pathish, adapter: Pathish, out: Pathish, *, expect: Any, threads: int = 8,
          expect_base_sha256: Optional[str] = None, training_run: Optional[str] = None) -> Path:
    """Merge `adapter` into `base` in fp32 under `out` and return the merged directory."""
    from ruleforge.grpo.merging import merge_adapter

    merge_adapter(base, adapter, out, expect=expect, threads=threads, expect_base_sha256=expect_base_sha256,
                  training_run=_saved_run_name(adapter) if training_run is None else training_run)
    return Path(out)


__all__ = ["PromptBuilder", "RowsBuild", "TrainRun", "base_config_path", "merge", "rows", "train", "training_config"]
