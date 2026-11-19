from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

from ruleforge.prompts.context import VALID_INDIVIDUAL_RULE_CONTEXTS
from ruleforge.prompts.parse import LABELS, read_label
from ruleforge.prompts.interpreters import INTERPRETATION_RULES
from ruleforge.prompts.styles import (
    FORCED_JSON_STYLES,
    NAIVE,
    NO_INTERPRETATION,
    VALID_REASONING_PROMPT_STYLES,
    hierarchy_refusal,
    no_interpretation_only_refusal,
    validate_forced_response,
)
from ruleforge.tasks.base import (
    DEFAULT_TASK, FORCED_JSON, OptimizableTask, Task, TaskError, TaskSettings, styles_listing,
)
from ruleforge.tasks.history import RATrajectory, RewardWeights, SGTrajectory
from ruleforge.tasks.legal import LegalTask, render_legal_prompt
from ruleforge.tasks.modelspec import ModelSpecTask, render_modelspec_prompt
from ruleforge.tasks.registry import load_task

CONFIG_SECTIONS = ("task_spec", "eval_cfg", "rule_amendment", "scenario_generation")


def config_snapshot(cfg: Any) -> dict[str, Any]:
    """Return the config sections a task spec reads, interpolations resolved; a missing section is None."""
    snapshot = {}
    for key in CONFIG_SECTIONS:
        value = cfg.get(key)
        snapshot[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return snapshot


class BaseTaskSpec(ABC):
    """The config-bound task surface; ``task`` is the task whose prompts and identity it serves."""

    task: Task
    # The ruleset the intent penalty compares every revision with; None scores no intent penalty.
    initial_rule_set_path: Optional[Path] = None

    def __init__(self, cfg: DictConfig):
        self._read_config(cfg)
        if self.individual_rule_context not in VALID_INDIVIDUAL_RULE_CONTEXTS:
            raise ValueError(
                f"Invalid individual_rule_context: {self.individual_rule_context!r}"
            )
        self.reasoning_prompt_style = str(
            getattr(self.eval_cfg, "reasoning_prompt_style", NAIVE)
        ).lower()
        if self.reasoning_prompt_style not in VALID_REASONING_PROMPT_STYLES:
            raise ValueError(
                f"Invalid reasoning_prompt_style: {self.reasoning_prompt_style!r}; "
                f"valid: {styles_listing(VALID_REASONING_PROMPT_STYLES)}"
            )
        refusal = hierarchy_refusal(self.reasoning_prompt_style, self.individual_rule_context)
        if refusal is not None:
            raise ValueError(refusal)

    def _read_config(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.built_from = config_snapshot(cfg)
        self.task_spec_name = str(cfg.task_spec)
        self.ra_cfg = cfg.rule_amendment
        self.sg_cfg = cfg.scenario_generation
        self.agent_model_cfg = cfg.agent_model
        self.judge_model_cfg = cfg.judge_model
        self.eval_cfg = cfg.eval_cfg
        self.individual_rule_context = str(
            getattr(self.eval_cfg, "individual_rule_context", "isolated")
        ).lower()

    def seed_scenario_paths(self) -> Dict[str, Path]:
        """Return the seed scenario files by their ``scenario_generation`` key."""
        return {}

    def task_settings(self, judge_type: str) -> TaskSettings:
        """Return the settings this spec judges under with ``judge_type``."""
        return TaskSettings(self.reasoning_prompt_style, self.individual_rule_context, judge_type)

    def task_identity(self, judge_type: str) -> dict[str, Any]:
        """Return what a condition judged by this spec with ``judge_type`` records of its task."""
        return self.task.identity(self.task_settings(judge_type))

    def check_new_run(self) -> None:
        """Refuse a new run under a record-only style; called where a run starts, not where a spec is built."""
        refusal = self.task.new_run_refusal(self.reasoning_prompt_style)
        if refusal is not None:
            raise TaskError(refusal, setting="prompt_style")

    def _optimizable(self) -> OptimizableTask:
        if not isinstance(self.task, OptimizableTask):
            raise TaskError(f"task {self.task.identifier!r} is not an OptimizableTask: the rule-amendment optimizer "
                            "cannot run it")
        return self.task

    @abstractmethod
    def scenario_seeding(self) -> Mapping[str, Any]:
        """Return the scenario generator's seeding: its type, the examples per prompt and the seed files."""

    def get_sg_environment_description(self) -> str:
        return self._optimizable().sg_environment_description()

    def get_ra_environment_description(self) -> str:
        return self._optimizable().ra_environment_description()

    def get_sg_prior(self) -> str:
        return self._optimizable().sg_prior()

    def get_ra_prior(self, weights: RewardWeights) -> str:
        """Return the rule amender's prior, stating the run's reward ``weights``."""
        return self._optimizable().ra_prior(weights)

    def build_ra_history_string(self, history: List[RATrajectory], weights: RewardWeights) -> str:
        return self._optimizable().ra_history(history, weights)

    def build_sg_history_string(self, history: List[SGTrajectory]) -> str:
        return self._optimizable().sg_history(history)

    def build_sg_posterior_sampling_system_prompt(
        self,
        already_generated_scenarios: List[Dict[str, Any]],
        history: List[SGTrajectory],
        rng: random.Random,
    ) -> str:
        """Return the scenario generator's system prompt under the config's batch size and seeding, any seed-file
        example drawn with ``rng``."""
        return self._optimizable().sg_system_prompt(
            per_batch=self.sg_cfg.num_scenarios_per_batch, seeds=self.scenario_seeding(),
            already=already_generated_scenarios, history=history, rng=rng,
        )

    @property
    def eval_response_validator(self) -> Optional[Callable[[str], Optional[str]]]:
        """Return a validator for a judge response, or None to read a label with ``valid_eval_responses``."""
        return None

    @abstractmethod
    def parse_judgment(self, response_text: Any) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def build_eval_prompt(
        self,
        rule_text: str,
        scenario: Dict[str, Any],
        judge_type: str,
        *,
        context_rules: Optional[List[str]] = None,
    ) -> str:
        """Return the judge prompt for one rule variant; empty ``context_rules`` must leave it unchanged."""
        raise NotImplementedError


def seed_file(sg_cfg: Any, key: str, *, read_by: str) -> Optional[Path]:
    """Return the seed file ``key`` names, or None; a config seeded by ``read_by`` must name it."""
    value = sg_cfg.get(key)
    if value is None:
        if sg_cfg.get("initial_scenario_seed_type") == read_by:
            raise ValueError(f"the {read_by!r} seeding reads scenario_generation.{key}, which the config does not "
                             "name")
        return None
    return Path(value)


def task_identity_of(task_spec: Any, judge_type: str) -> dict[str, Any]:
    """Return what a condition judged by ``task_spec`` records of its task; a spec with no task records the default
    task's."""
    identity = getattr(task_spec, "task_identity", None)
    if identity is None:
        return {
            "task": DEFAULT_TASK,
            "prompt_style": task_spec.reasoning_prompt_style,
            "individual_rule_context": task_spec.individual_rule_context,
            "judge_type": judge_type,
            "response_contract": None,
        }
    return identity(judge_type)


class TaskSpecAdapter(BaseTaskSpec):
    """Any task behind the task-spec surface; every style, rule context and judge type pairing is approved by the
    task."""

    def __init__(self, task: Task, cfg: DictConfig):
        self.task = task
        self._read_config(cfg)
        style = str(getattr(self.eval_cfg, "reasoning_prompt_style", task.default_style))
        self.reasoning_prompt_style = style.lower() if style not in task.styles and style.lower() in task.styles \
            else style
        self.judge_types = [str(judge_type)
                            for judge_type in getattr(self.eval_cfg, "judge_types", [NO_INTERPRETATION])]
        for judge_type in self.judge_types:
            task.check(self.task_settings(judge_type))
        self.interpretation_rules = task.interpretations
        self.valid_scenario_keys = tuple(task.scenario_fields)
        self.valid_eval_responses = LABELS
        self.evaluation_temperature = float(getattr(self.eval_cfg, "evaluation_temperature", 0.1))
        self.save_batch_size = int(getattr(self.eval_cfg, "save_batch_size", 10))
        if isinstance(task, OptimizableTask):
            self.scenario_block_pattern = task.scenario_block_pattern

    def scenario_seeding(self) -> Mapping[str, Any]:
        """Return the config's seeding type, example count and every ``*seed_scenario_path`` key it sets."""
        return {
            "type": str(getattr(self.sg_cfg, "initial_scenario_seed_type", "none")),
            "count": int(getattr(self.sg_cfg, "num_seed_scenarios", 1)),
            "paths": {key: str(value) for key, value in self.sg_cfg.items() if str(key).endswith("seed_scenario_path")},
        }

    @property
    def eval_response_validator(self) -> Optional[Callable[[str], Optional[str]]]:
        if isinstance(self.task, OptimizableTask):
            return self.task.response_validator(self.reasoning_prompt_style)
        return validate_forced_response if self.task.contract(self.reasoning_prompt_style) == FORCED_JSON else None

    def parse_judgment(self, response_text: Any) -> Optional[str]:
        if isinstance(self.task, OptimizableTask):
            return self.task.parse_judgment(response_text)
        return read_label(response_text)

    def build_eval_prompt(
        self,
        rule_text: str,
        scenario: Dict[str, Any],
        judge_type: str,
        *,
        context_rules: Optional[List[str]] = None,
    ) -> str:
        return self.task.render(rule_text, scenario, self.task_settings(judge_type), context_rules=context_rules)


class ModelSpecTaskSpec(BaseTaskSpec):
    """The Model Spec task's spec."""

    def __init__(self, cfg: DictConfig):
        self.task = ModelSpecTask()
        super().__init__(cfg=cfg)
        self.scenario_block_pattern = self.task.scenario_block_pattern
        self.interpretation_rules = INTERPRETATION_RULES
        self.initial_rule_set_path = Path(self.ra_cfg.initial_rule_set_path)
        self.judge_types = list(self.eval_cfg.judge_types)
        refusal = no_interpretation_only_refusal(self.reasoning_prompt_style, self.judge_types)
        if refusal is not None:
            raise ValueError(refusal)
        self.evaluation_temperature = float(getattr(self.eval_cfg, "evaluation_temperature", 0.1))
        self.save_batch_size = int(getattr(self.eval_cfg, "save_batch_size", 10))

        self.num_seed_scenarios = int(getattr(self.sg_cfg, "num_seed_scenarios", 5))
        self.good_seed_scenario_path = seed_file(self.sg_cfg, "good_seed_scenario_path", read_by="wildchat")
        self.bad_seed_scenario_path = seed_file(self.sg_cfg, "bad_seed_scenario_path", read_by="wildchat")

        self.valid_scenario_keys = ("prompt", "response")
        self.valid_eval_responses = LABELS
    
    def seed_scenario_paths(self) -> Dict[str, Path]:
        held = {"good_seed_scenario_path": self.good_seed_scenario_path,
                "bad_seed_scenario_path": self.bad_seed_scenario_path}
        return {key: path for key, path in held.items() if path is not None}

    def scenario_seeding(self) -> Mapping[str, Any]:
        return {"type": self.sg_cfg.initial_scenario_seed_type, "count": self.num_seed_scenarios,
                "paths": self.seed_scenario_paths()}

    def parse_judgment(self, response_text: Any) -> Optional[str]:
        return read_label(response_text)

    @property
    def eval_response_validator(self):
        if self.reasoning_prompt_style in FORCED_JSON_STYLES:
            return validate_forced_response
        return None

    def build_eval_prompt(
        self,
        rule_text: str,
        scenario: Dict[str, Any],
        judge_type: str,
        *,
        context_rules: Optional[List[str]] = None,
    ) -> str:
        interpretations = {} if judge_type == NO_INTERPRETATION else self.interpretation_rules
        return render_modelspec_prompt(
            self.reasoning_prompt_style, rule_text, scenario, judge_type, context_rules=context_rules,
            interpretations=interpretations,
        )


class LegalTaskSpec(BaseTaskSpec):
    """The legal task's spec: a court answering a yes/no/unsure question about a statute under the naive style only."""

    def __init__(self, cfg: DictConfig):
        self.task = LegalTask()
        super().__init__(cfg=cfg)
        if self.reasoning_prompt_style != NAIVE:
            raise ValueError(
                "LegalTaskSpec supports only reasoning_prompt_style='naive', "
                f"got {self.reasoning_prompt_style!r}"
            )
        self.scenario_block_pattern = self.task.scenario_block_pattern
        self.interpretation_rules = INTERPRETATION_RULES
        self.initial_rule_set_path = Path(self.ra_cfg.initial_rule_set_path)
        self.judge_types = list(self.eval_cfg.judge_types)
        self.evaluation_temperature = float(getattr(self.eval_cfg, "evaluation_temperature", 0.1))
        self.save_batch_size = int(getattr(self.eval_cfg, "save_batch_size", 10))

        self.num_seed_scenarios = int(getattr(self.sg_cfg, "num_seed_scenarios", 1))
        self.seed_scenario_path = seed_file(self.sg_cfg, "seed_scenario_path", read_by="single_seed")

        self.valid_scenario_keys = ("facts_of_case", "legal_question")
        self.valid_eval_responses = LABELS
    
    def seed_scenario_paths(self) -> Dict[str, Path]:
        return {} if self.seed_scenario_path is None else {"seed_scenario_path": self.seed_scenario_path}

    def scenario_seeding(self) -> Mapping[str, Any]:
        return {"type": self.sg_cfg.initial_scenario_seed_type, "count": self.num_seed_scenarios,
                "paths": self.seed_scenario_paths()}

    def parse_judgment(self, response_text: Any) -> Optional[str]:
        return read_label(response_text)

    def build_eval_prompt(
        self,
        rule_text: str,
        scenario: Dict[str, Any],
        judge_type: str,
        *,
        context_rules: Optional[List[str]] = None,
    ) -> str:
        interpretations = {} if judge_type == NO_INTERPRETATION else self.interpretation_rules
        return render_legal_prompt(rule_text, scenario, judge_type, context_rules=context_rules,
                                   interpretations=interpretations)


_TASK_SPECS = {
    "modelspec": ModelSpecTaskSpec,
    "legal": LegalTaskSpec,
}


def build_task_spec(cfg: DictConfig) -> BaseTaskSpec:
    """Return the task spec ``cfg.task_spec`` names: a built-in's own, or any other task behind ``TaskSpecAdapter``."""
    task_spec_name = str(cfg.task_spec)
    cls = _TASK_SPECS.get(task_spec_name)
    if cls is not None:
        return cls(cfg=cfg)
    try:
        task = load_task(task_spec_name)
    except TaskError as error:
        raise ValueError(
            f"Unknown task_spec '{task_spec_name}'. "
            f"Valid options: {sorted(_TASK_SPECS.keys())}, or a task load_task finds ({error})"
        ) from error
    return TaskSpecAdapter(task, cfg)
