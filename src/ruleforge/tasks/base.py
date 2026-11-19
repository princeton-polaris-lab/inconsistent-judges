from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.prompts.context import VALID_INDIVIDUAL_RULE_CONTEXTS
from ruleforge.prompts.interpreters import INTERPRETATION_RULES
from ruleforge.prompts.parse import read_label
from ruleforge.prompts.styles import NO_INTERPRETATION, validate_forced_response
from ruleforge.tasks.history import RATrajectory, RewardWeights, SGTrajectory

# Response contracts: a bare label, or a JSON object with ``full_set_judgment`` and a ``rule_judgments`` map.
LABEL = "label"
FORCED_JSON = "forced_json"
RESPONSE_CONTRACTS = (LABEL, FORCED_JSON)
BUILTIN_TASKS = ("modelspec", "legal")
DEFAULT_TASK = "modelspec"
DEFAULT_RULE_CONTEXT = "isolated"

_TASK_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:@[A-Za-z0-9][A-Za-z0-9._+-]*)?$")


class TaskError(ValueError, RuleForgeError):
    """Raised for a task, style or pairing of settings a task refuses; ``setting`` names the setting at fault."""

    def __init__(self, message: str, *, setting: Optional[str] = None) -> None:
        super().__init__(message)
        self.setting = setting


def styles_listing(styles: Iterable[str], record_only: Iterable[str] = ()) -> str:
    """Return the listing an unknown-style refusal names: the selectable styles, then the record-only ones."""
    kept = sorted(record_only)
    selectable = str(sorted(set(styles) - set(kept)))
    return f"{selectable}; recognised only to read the records that ran them: {kept}" if kept else selectable


def is_task_identifier(value: Any) -> bool:
    """Return whether ``value`` is a task identifier: a name, or ``name@version``."""
    return isinstance(value, str) and _TASK_IDENTIFIER.match(value) is not None


@dataclass(frozen=True)
class TaskSettings:
    """The settings one condition judges under; every field is condition identity."""

    prompt_style: str
    # "isolated": each rule alone; "implicated": with the rules it references, a range read as its first rule;
    # "implicated_ranges": the same, a range read as every rule in it; "superiors": with the rules above it.
    rule_context: str = DEFAULT_RULE_CONTEXT
    judge_type: str = NO_INTERPRETATION


class Task(ABC):
    """A judging task (scenario fields, prompt styles and their contracts, judge prompts); subclasses declare the class
    attributes and implement ``render``, and may extend ``check``."""

    name: ClassVar[str]
    version: ClassVar[str] = ""
    scenario_fields: ClassVar[tuple[str, ...]]
    # prompt style -> response contract (LABEL or FORCED_JSON)
    styles: ClassVar[Mapping[str, str]]
    # Styles recognised only to read the records that ran them; no new run selects one.
    record_only_styles: ClassVar[frozenset[str]] = frozenset()
    default_style: ClassVar[str]
    interpretations: ClassVar[Mapping[str, str]] = INTERPRETATION_RULES

    @property
    def identifier(self) -> str:
        """Return the task's recorded identity: its name, or ``name@version``."""
        return f"{self.name}@{self.version}" if self.version else self.name

    @property
    def builtin(self) -> bool:
        """Return whether this is one of the shipped tasks itself (a subclass of one is not)."""
        return bool(type(self).__dict__.get("_builtin", False))

    def check(self, settings: TaskSettings) -> None:
        """Refuse settings this task cannot judge under: an unknown style, rule context or judge type."""
        if settings.prompt_style not in self.styles:
            raise TaskError(
                f"task {self.identifier!r} has no prompt style {settings.prompt_style!r}; its styles: "
                f"{styles_listing(self.styles, self.record_only_styles)}", setting="prompt_style")
        if settings.rule_context not in VALID_INDIVIDUAL_RULE_CONTEXTS:
            raise TaskError(
                f"rule context {settings.rule_context!r} is not one of {sorted(VALID_INDIVIDUAL_RULE_CONTEXTS)}",
                setting="rule_context")
        if settings.judge_type != NO_INTERPRETATION and settings.judge_type not in self.interpretations:
            raise TaskError(
                f"judge type {settings.judge_type!r} is neither {NO_INTERPRETATION!r} nor an interpretation "
                f"strategy of task {self.identifier!r}", setting="judge_type")

    def new_run_refusal(self, style: str) -> Optional[str]:
        """Return why a new run may not select ``style`` (it is record-only), or None."""
        if style not in self.record_only_styles:
            return None
        return (f"prompt style {style!r} of task {self.identifier!r} is recognised only to read the records that ran "
                "it; a new run cannot select it")

    def check_scenarios(self, scenarios: Iterable[Mapping[str, Any]]) -> None:
        """Refuse scenario rows (or a ``ScenarioSet``) lacking a field in ``scenario_fields``."""
        rows = getattr(scenarios, "rows", scenarios)
        for index, row in enumerate(rows):
            missing = [name for name in self.scenario_fields if name not in row]
            if missing:
                raise TaskError(f"scenario row {index} lacks {missing}, which task {self.identifier!r} reads")

    @abstractmethod
    def render(
        self,
        rule_text: str,
        scenario: Mapping[str, Any],
        settings: TaskSettings,
        *,
        context_rules: Optional[Sequence[str]] = None,
    ) -> str:
        """Return the judge's system prompt for one rule text and one scenario. ``context_rules`` are pre-formatted
        rules shown as context; None or empty must leave the prompt unchanged."""

    def contract(self, style: str) -> str:
        """Return ``style``'s response contract."""
        if style not in self.styles:
            raise TaskError(f"task {self.identifier!r} has no prompt style {style!r}", setting="prompt_style")
        return self.styles[style]

    def identity(self, settings: TaskSettings) -> dict[str, Any]:
        """Return what a condition records of its task, once ``check`` approves the settings."""
        self.check(settings)
        return {
            "task": self.identifier,
            "prompt_style": settings.prompt_style,
            "individual_rule_context": settings.rule_context,
            "judge_type": settings.judge_type,
            "response_contract": None if self.builtin else self.contract(settings.prompt_style),
        }


class OptimizableTask(Task):
    """A task the rule-amendment optimizer can run: the prompts and strings its two agents are given."""

    # Parses the scenario generator's output; its named groups are the scenario fields.
    scenario_block_pattern: ClassVar[re.Pattern[str]]

    @abstractmethod
    def sg_environment_description(self) -> str: ...

    @abstractmethod
    def ra_environment_description(self) -> str: ...

    @abstractmethod
    def sg_prior(self) -> str: ...

    @abstractmethod
    def ra_prior(self, weights: RewardWeights) -> str:
        """Return the rule amender's prior, which states the reward's penalty ``weights``."""

    @abstractmethod
    def sg_system_prompt(
        self, *, per_batch: int, seeds: Mapping[str, Any], already: Sequence[Mapping[str, Any]],
        history: Sequence[SGTrajectory], rng: random.Random,
    ) -> str:
        """Return the scenario generator's system prompt; ``seeds`` holds the seeding ``type``, the example ``count``
        per prompt and the seed file ``paths`` by config key, an example from a seed file drawn with ``rng``."""

    @abstractmethod
    def ra_history(self, history: Sequence[RATrajectory], weights: RewardWeights) -> str:
        """Return the rule amender's history, its rewards broken down as ``weights`` composes them."""

    @abstractmethod
    def sg_history(self, history: Sequence[SGTrajectory]) -> str:
        """Return the scenario generator's history."""

    def parse_judgment(self, text: Any) -> Optional[str]:
        """Return a judge's label from its answer, or None."""
        return read_label(text)

    def response_validator(self, style: str) -> Optional[Callable[[str], Optional[str]]]:
        """Return the forced JSON validator for a forced style, else None (the label tuple applies)."""
        return validate_forced_response if self.contract(style) == FORCED_JSON else None
