from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ruleforge.rulesets.rules import (
    build_rule_id_index,
    is_meta_rule_id,
    normalize_preempting_rule_id,
)
from ruleforge.evaluate.identity import (
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_BOTH,
    EVALUATION_MODE_FULL,
    EVALUATION_MODES,
    FORCED_JSON_COMPLETE_RULE_MAP,
    FORCED_JSON_REQUIREMENTS,
    FULL_RULESET_COMPONENT,
    LABEL_RULE,
    ConditionSpec,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.prompts.parse import LABELS, THINK_CLOSE, holds_an_open_thought, read_label
from ruleforge.prompts.styles import (
    FORCED_JSON_STYLES,
    VALID_REASONING_PROMPT_STYLES,
    has_json_object,
    lists_preemption,
    parse_full_set_json,
)
from ruleforge.tasks.base import FORCED_JSON, LABEL, RESPONSE_CONTRACTS
from ruleforge.judges.backend import GenerationOutcome
from ruleforge.judges.batch_api import is_truncated

# Why a component failed validation. The first two are transport outcomes; the rest consume a semantic attempt.
TRANSPORT_ERROR = "transport_error"
MISSING_PROVIDER_RESULT = "missing_provider_result"
TRUNCATED = "truncated"
EMPTY_RESPONSE = "empty_response"
INVALID_JSON = "invalid_json"
# Valid JSON without a top-level judgment. Recorded runs name this failure invalid_json.
MISSING_FULL_SET_JUDGMENT = "missing_full_set_judgment"
MISSING_REQUIRED_FIELD = "missing_required_field"
INVALID_LABEL = "invalid_label"
# Kept apart from INVALID_LABEL: only ``complete_rule_map`` rejects this.
INVALID_RULE_LABEL = "invalid_rule_label"
MISSING_RULE_IDS = "missing_rule_ids"
DUPLICATE_RULE_IDS = "duplicate_rule_ids"
UNEXPECTED_RULE_IDS = "unexpected_rule_ids"
UNCLOSED_THINK = "unclosed_think"

TRANSPORT_REASONS = (TRANSPORT_ERROR, MISSING_PROVIDER_RESULT)


class _PairsDict(dict):
    """``json.loads`` object hook that remembers every key: the complete-rule-map rule must see a repeated ``"Rule
    7"``."""

    __slots__ = ("pairs",)

    def __init__(self, pairs):
        self.pairs = tuple(pairs)
        super().__init__(self.pairs)


@dataclass(frozen=True)
class ComponentVerdict:
    """What validation decided about one component."""

    valid: bool
    judgment: Optional[str] = None
    failure_reason: Optional[str] = None
    # Rule id -> label, only for a forced object validated under ``complete_rule_map``.
    rule_judgments: Optional[dict[str, str]] = None
    preempted_rules: tuple[str, ...] = ()

    @property
    def is_transport_failure(self) -> bool:
        return self.failure_reason in TRANSPORT_REASONS


def _invalid(reason: str) -> ComponentVerdict:
    return ComponentVerdict(valid=False, failure_reason=reason)


def strict_think_answer(raw: str) -> str:
    """Return the raw output after its last ``</think>``, which is what ``strict_think`` validates."""
    return raw.rsplit(THINK_CLOSE, 1)[-1]


def _json_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in LABELS else None


def check_label_rule(spec: ConditionSpec) -> None:
    """Refuse a label condition recorded under another label rule; its recorded judgments are read as they stand."""
    if spec.label_rule != LABEL_RULE and spec.contract != FORCED_JSON:
        raise SamplingDesignError(
            f"condition {spec.condition_id!r} recorded its label answers under {spec.label_rule!r}: this version reads "
            f"a label answer with one reader, recorded as {LABEL_RULE!r}, and reads such a condition's records as they "
            "stand, never judging it or reading its answers again")


@dataclass(frozen=True)
class ComponentSchema:
    """The validation rule for a condition's components, built once from the rule set."""

    prompt_style: Optional[str]
    forced_json_requirement: str
    expected_rule_ids: tuple[str, ...]
    rule_id_index: dict[str, str] = field(default_factory=dict)
    evaluation_mode: str = EVALUATION_MODE_BOTH
    strict_think: bool = False
    # ``label`` or ``forced_json``; None derives it from a built-in ``prompt_style``.
    response_contract: Optional[str] = None
    lists_preemption: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.response_contract is None:
            if self.prompt_style not in VALID_REASONING_PROMPT_STYLES:
                raise SamplingDesignError(f"unknown prompt style {self.prompt_style!r}")
            object.__setattr__(
                self, "response_contract", FORCED_JSON if self.prompt_style in FORCED_JSON_STYLES else LABEL)
            object.__setattr__(self, "lists_preemption", lists_preemption(self.prompt_style))
        elif self.response_contract not in RESPONSE_CONTRACTS:
            raise SamplingDesignError(f"unknown response contract {self.response_contract!r}")
        if self.lists_preemption is None:
            object.__setattr__(self, "lists_preemption", False)
        if not isinstance(self.lists_preemption, bool) or (
                self.lists_preemption and self.response_contract != FORCED_JSON):
            raise SamplingDesignError("lists_preemption is a forced contract's, and a bool")
        if self.evaluation_mode not in EVALUATION_MODES:
            raise SamplingDesignError(f"unknown evaluation_mode {self.evaluation_mode!r}")
        if self.forced_json_requirement not in FORCED_JSON_REQUIREMENTS:
            raise SamplingDesignError(
                f"unknown forced_json_requirement {self.forced_json_requirement!r}"
            )
        if any(is_meta_rule_id(rule_id) for rule_id in self.expected_rule_ids):
            raise SamplingDesignError("expected_rule_ids must not include meta-rules")

    @staticmethod
    def _rule_ids(rules: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        expected = tuple(
            str(rule["id"]).strip()
            for rule in rules
            if not is_meta_rule_id(str(rule.get("id", "")))
        )
        if not expected:
            raise SamplingDesignError("a rule set needs at least one non-meta rule")
        return {"expected_rule_ids": expected, "rule_id_index": build_rule_id_index(list(rules))}

    @classmethod
    def from_rules(
        cls,
        rules: Sequence[Mapping[str, Any]],
        *,
        prompt_style: str,
        forced_json_requirement: str,
        evaluation_mode: str = EVALUATION_MODE_BOTH,
        strict_think: bool = False,
    ) -> "ComponentSchema":
        """The schema of a built-in style over ``rules``: the style decides the contract."""
        return cls(
            prompt_style=prompt_style,
            forced_json_requirement=forced_json_requirement,
            evaluation_mode=evaluation_mode,
            strict_think=strict_think,
            **cls._rule_ids(rules),
        )

    @classmethod
    def for_condition(
        cls, spec: ConditionSpec, rules: Optional[Sequence[Mapping[str, Any]]] = None, **overrides: Any
    ) -> "ComponentSchema":
        """Return a condition's schema; a label condition recorded under another label rule is refused."""
        check_label_rule(spec)
        return cls._of(spec, rules, overrides)

    @classmethod
    def for_recorded_judgments(cls, spec: ConditionSpec, component_ids: Sequence[str]) -> "ComponentSchema":
        """Return the schema a condition's recorded judgments are scored by; no answer is read again."""
        return cls._of(spec, None, {"expected_rule_ids": tuple(
            component_id for component_id in component_ids if component_id != FULL_RULESET_COMPONENT)})

    @classmethod
    def _of(cls, spec: ConditionSpec, rules: Optional[Sequence[Mapping[str, Any]]],
            overrides: Mapping[str, Any]) -> "ComponentSchema":
        recorded = spec.response_contract
        fields = {
            "prompt_style": spec.prompt_style,
            "response_contract": recorded,
            "lists_preemption": None if recorded is None else spec.lists_preemption,
            "forced_json_requirement": spec.forced_json_requirement,
            "evaluation_mode": spec.evaluation_mode,
            "strict_think": spec.strict_think,
            **(cls._rule_ids(rules) if rules is not None else {}),
            **overrides,
        }
        return cls(**fields)

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Every component a bundle of this mode must contain, full set first."""
        if self.evaluation_mode == EVALUATION_MODE_FULL:
            return (FULL_RULESET_COMPONENT,)
        if self.evaluation_mode == EVALUATION_MODE_BANK:
            return tuple(self.expected_rule_ids)
        return (FULL_RULESET_COMPONENT, *self.expected_rule_ids)

    def classify(self, component_id: str, outcome: GenerationOutcome) -> ComponentVerdict:
        """Decide validity for one component's generation outcome: only a provider error or nothing returned is
        transport; an empty returned generation consumes an attempt, so a silent judge is not retried forever."""
        if outcome.error:
            return _invalid(TRANSPORT_ERROR)
        if not outcome.returned:
            return _invalid(MISSING_PROVIDER_RESULT)
        if is_truncated(outcome.finish_reason):
            return _invalid(TRUNCATED)
        text = outcome.text
        if self.strict_think:
            raw = outcome.raw_response or ""
            if raw.strip() and THINK_CLOSE not in raw:
                return _invalid(UNCLOSED_THINK)
            text = strict_think_answer(raw)
        if not isinstance(text, str) or not text.strip():
            return _invalid(EMPTY_RESPONSE)
        if holds_an_open_thought(text):
            return _invalid(UNCLOSED_THINK)
        if self.response_contract != FORCED_JSON:
            judgment = read_label(text)
            if judgment is None:
                return _invalid(INVALID_LABEL)
            return ComponentVerdict(valid=True, judgment=judgment)
        require_map = (
            component_id == FULL_RULESET_COMPONENT
            and self.forced_json_requirement == FORCED_JSON_COMPLETE_RULE_MAP
        )
        return self.classify_forced(text, require_map=require_map)

    def classify_forced(self, text: str, *, require_map: bool) -> ComponentVerdict:
        """Validate a forced answer's judgment and, with ``require_map``, its rule map (``text`` excludes reasoning);
        where the style's contract asks for ``preempted_rules``, a present non-null non-list value rejects it."""
        verdict = parse_full_set_json(text, object_pairs_hook=_PairsDict)
        if verdict is None:
            return _invalid(MISSING_FULL_SET_JUDGMENT if has_json_object(text) else INVALID_JSON)
        judgment = _json_label(verdict.get("full_set_judgment"))
        if judgment is None:
            return _invalid(INVALID_LABEL)
        listed = verdict.get("preempted_rules") if self.lists_preemption else None
        if listed is not None and not isinstance(listed, list):
            return _invalid(MISSING_REQUIRED_FIELD)
        preempted = tuple(item for item in listed or () if isinstance(item, str))
        if not require_map:
            return ComponentVerdict(valid=True, judgment=judgment, preempted_rules=preempted)
        rule_judgments = verdict.get("rule_judgments")
        if not isinstance(rule_judgments, _PairsDict):
            return _invalid(MISSING_REQUIRED_FIELD)
        expected = set(self.expected_rule_ids)
        mapped: dict[str, str] = {}
        for key, value in rule_judgments.pairs:
            canonical = normalize_preempting_rule_id(key, self.rule_id_index)
            if canonical is None or canonical not in expected:
                return _invalid(UNEXPECTED_RULE_IDS)
            if canonical in mapped:
                return _invalid(DUPLICATE_RULE_IDS)
            label = _json_label(value)
            if label is None:
                return _invalid(INVALID_RULE_LABEL)
            mapped[canonical] = label
        if set(mapped) != expected:
            return _invalid(MISSING_RULE_IDS)
        ordered = {rule_id: mapped[rule_id] for rule_id in self.expected_rule_ids}
        return ComponentVerdict(
            valid=True,
            judgment=judgment,
            rule_judgments=ordered,
            preempted_rules=preempted,
        )
