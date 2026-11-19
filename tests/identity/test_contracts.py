from __future__ import annotations

import dataclasses
import json
from collections import defaultdict

import pytest

from ruleforge.evaluate import cost as batch_cost
from ruleforge.evaluate.aggregate import _families, style_families
from ruleforge.evaluate.identity import _OMITTED_AT_DEFAULT, LABEL_RULE, ConditionSpec
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.core.record_names import public_style
from ruleforge.evaluate.validation import ComponentSchema
from ruleforge.prompts.styles import (
    FORCED_JSON_STYLES,
    HIERARCHY_RULE_CONTEXT,
    NAIVE_HIERARCHY,
    PREEMPTION_LISTING_STYLES,
    VALID_REASONING_PROMPT_STYLES,
)
from ruleforge.tasks import FORCED_JSON, LABEL, ModelSpecTask, TaskSettings
from tests.identity import published as pub
from tests.identity.test_tables import PROMPTS, _rule_texts, prompt_scenarios
from tests.support.recorded_root import recorded_condition

ROWS = pub.read_fixture(pub.CONDITIONS)
PUBLISHED = [row for row in pub.rows_of(ROWS, "condition")]
RULES = [{"id": "Rule 1", "rule": "a"}, {"id": "Rule 2", "rule": "b"}, {"id": "Meta-Rule 1", "rule": "m"}]
NEW_FIELDS = {"task": "modelspec", "response_contract": None, "judge_extension": None}


def _spec(payload) -> ConditionSpec:
    return recorded_condition(payload)


def _condition(style="naive", **overrides) -> ConditionSpec:
    paired = {"individual_rule_context": HIERARCHY_RULE_CONTEXT, "rule_levels_sha256": "0" * 64} \
        if style == NAIVE_HIERARCHY else {}
    fields = dict(condition_id=f"m/{style}", model_name="m", model_type="local", prompt_style=style,
                  individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
                  rule_set_name="r.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
                  forced_json_requirement="complete_rule_map")
    fields.update(paired)
    fields.update(overrides)
    return ConditionSpec(**fields)


def _expected(style: str) -> tuple[str, bool]:
    return (FORCED_JSON if style in FORCED_JSON_STYLES else LABEL), style in PREEMPTION_LISTING_STYLES


def _old_schema(spec: ConditionSpec) -> ComponentSchema:
    return ComponentSchema.from_rules(
        RULES, prompt_style=spec.prompt_style, forced_json_requirement=spec.forced_json_requirement,
        evaluation_mode=spec.evaluation_mode, strict_think=spec.strict_think)


@pytest.mark.parametrize("style", VALID_REASONING_PROMPT_STYLES)
def test_a_built_in_style_s_contract_is_its_tuple_membership(style):
    spec = _condition(style)
    schema = ComponentSchema.for_condition(spec, RULES)
    assert (spec.contract, spec.lists_preemption) == _expected(style)
    assert (schema.response_contract, schema.lists_preemption, schema.prompt_style) == (*_expected(style), style)
    assert schema == _old_schema(spec)


def test_every_published_condition_s_contract_is_its_style_s():
    as_recorded = set()
    for row in PUBLISHED:
        spec = _spec(row["payload"])
        assert (spec.contract, spec.lists_preemption) == _expected(spec.prompt_style), row["path"]
        if spec.label_rule != LABEL_RULE and spec.contract != FORCED_JSON:
            as_recorded.add(row["root"])
            with pytest.raises(SamplingDesignError, match="as they stand"):
                ComponentSchema.for_condition(spec, RULES)
            continue
        assert ComponentSchema.for_condition(spec, RULES) == _old_schema(spec), row["path"]
    assert as_recorded == {"hierarchy-aware-2026-10-pop-a22af097"}


def test_a_recorded_contract_decides_for_a_task_that_is_not_built_in():
    """An outside task's label arm is read as labels whatever its style is named."""
    label = _condition("forced-grpo", task="demo@1", response_contract=LABEL)
    forced = _condition("plain", task="demo@1", response_contract=FORCED_JSON)
    named_like_ours = _condition("forced-grpo", task="demo@1", response_contract=FORCED_JSON)
    assert (label.contract, label.lists_preemption) == (LABEL, False)
    assert (forced.contract, forced.lists_preemption) == (FORCED_JSON, False)
    assert (named_like_ours.contract, named_like_ours.lists_preemption) == (FORCED_JSON, False)
    schema = ComponentSchema.for_condition(forced, RULES)
    verdict = schema.classify("full_ruleset", _outcome(json.dumps({
        "rule_judgments": {"Rule 1": "no", "Rule 2": "yes"}, "preempted_rules": ["Rule 2"],
        "full_set_judgment": "yes"})))
    assert (verdict.valid, verdict.judgment, verdict.preempted_rules) == (True, "yes", ())
    assert ComponentSchema.for_condition(label, RULES).classify("full_ruleset", _outcome("no")).judgment == "no"


def _outcome(text):
    from ruleforge.judges.backend import GenerationOutcome

    return GenerationOutcome(text=text, raw_response=text, finish_reason="stop")


def test_the_schema_keeps_its_old_constructor_and_refuses_a_contract_it_does_not_know():
    schema = ComponentSchema(prompt_style="forced-grpo", forced_json_requirement="label_only",
                             expected_rule_ids=("Rule 1",))
    assert (schema.response_contract, schema.lists_preemption) == (FORCED_JSON, True)
    positional = ComponentSchema("forced-grpo", "complete_rule_map", ("Rule 1",), {"rule 1": "Rule 1"}, "full", True)
    assert (positional.prompt_style, positional.forced_json_requirement, positional.expected_rule_ids,
            positional.rule_id_index, positional.evaluation_mode, positional.strict_think,
            positional.response_contract, positional.lists_preemption) == (
        "forced-grpo", "complete_rule_map", ("Rule 1",), {"rule 1": "Rule 1"}, "full", True, FORCED_JSON, True)
    explicit = ComponentSchema(prompt_style=None, forced_json_requirement="label_only", expected_rule_ids=("Rule 1",),
                               response_contract=LABEL)
    assert (explicit.prompt_style, explicit.lists_preemption) == (None, False)
    with pytest.raises(SamplingDesignError, match="unknown prompt style None"):
        ComponentSchema(None, "label_only", ("Rule 1",))
    with pytest.raises(SamplingDesignError, match="unknown response contract 'prose'"):
        ComponentSchema(prompt_style=None, forced_json_requirement="label_only", expected_rule_ids=("Rule 1",),
                        response_contract="prose")
    with pytest.raises(SamplingDesignError, match="lists_preemption is a forced contract's"):
        ComponentSchema(prompt_style=None, forced_json_requirement="label_only", expected_rule_ids=("Rule 1",),
                        response_contract=LABEL, lists_preemption=True)
    with pytest.raises(SamplingDesignError, match="lists_preemption is a forced contract's"):
        ComponentSchema(prompt_style=None, forced_json_requirement="label_only", expected_rule_ids=("Rule 1",),
                        response_contract=FORCED_JSON, lists_preemption="yes")


def test_a_built_in_task_s_unknown_style_is_refused_by_its_schema():
    """A built-in task's condition records no contract, so its style must be one the built-in task knows."""
    with pytest.raises(SamplingDesignError, match="task 'modelspec' has no prompt style 'forced-v3'"):
        _condition("forced-v3")
    spec = _condition("forced")
    object.__setattr__(spec, "prompt_style", "forced-v3")
    with pytest.raises(SamplingDesignError, match="unknown prompt style 'forced-v3'"):
        ComponentSchema.for_condition(spec, RULES)


def test_a_schema_without_rules_takes_its_rule_ids_from_the_overrides():
    spec = _condition("forced-grpo")
    schema = ComponentSchema.for_condition(spec, expected_rule_ids=("Rule 1",), rule_id_index={"rule 1": "Rule 1"})
    assert (schema.component_ids, schema.response_contract) == (("full_ruleset", "Rule 1"), FORCED_JSON)


def test_style_families_of_every_published_root_are_unchanged():
    by_root = defaultdict(dict)
    for row in PUBLISHED:
        spec = _spec(row["payload"])
        by_root[row["root"]][spec.condition_id] = spec
    for root, conditions in by_root.items():
        before = sorted(sorted(members) for members in _families(conditions, ("prompt_style",
                                                                                "forced_json_requirement")))
        assert sorted(sorted(members) for members in style_families(conditions).values()) == before, root


def test_an_outside_label_arm_and_forced_arm_are_one_style_family():
    label = _condition("plain", condition_id="m/plain", task="demo@1", response_contract=LABEL)
    forced = _condition("forced", condition_id="m/forced", task="demo@1", response_contract=FORCED_JSON)
    other_task = _condition("forced", condition_id="x/forced", task="other", response_contract=FORCED_JSON)
    families = style_families({c.condition_id: c for c in (label, forced, other_task)})
    assert sorted(sorted(members) for members in families.values()) == [["m/forced", "m/plain"], ["x/forced"]]


def test_each_new_field_is_omitted_at_its_default():
    for name, default in NEW_FIELDS.items():
        assert _OMITTED_AT_DEFAULT[name] == default


def test_every_published_fingerprint_is_unchanged_by_the_new_fields():
    for row in PUBLISHED:
        spec = _spec(row["payload"])
        assert not set(NEW_FIELDS) & set(spec.to_dict()), row["path"]
        assert spec.fingerprint == row["payload"]["fingerprint"], row["path"]


@pytest.mark.parametrize(
    ("base", "field", "values"),
    [
        ({}, "task", ("modelspec", "legal")),
        ({"task": "demo@1"}, "response_contract", (LABEL, FORCED_JSON)),
        ({"model_type": "myjudge"}, "judge_extension",
         ('{"options":{},"provider":"p","version":"1"}', '{"options":{"k":1},"provider":"p","version":"1"}')),
    ],
)
def test_each_new_field_enters_the_identity_once_set(base, field, values):
    first, second = (_condition("naive", **base, **{field: value}) for value in values)
    assert field in second.to_dict() and second.fingerprint != first.fingerprint
    assert (field in first.to_dict()) == (values[0] != NEW_FIELDS[field])
    assert ConditionSpec.from_dict(second.to_dict()) == second


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task": "demo@1"}, "a task that is not built in"),
        ({"response_contract": LABEL}, "a task that is not built in"),
        ({"task": "demo", "response_contract": "prose"}, "response_contract must be one of"),
        ({"task": "bad name"}, "task must be a task name"),
        ({"model_type": "myjudge"}, "records its options as its extension"),
        ({"judge_extension": "{}"}, "records its options as its extension"),
        ({"model_type": "myjudge", "judge_extension": ""}, "judge_extension must be a non-empty string"),
    ],
)
def test_the_new_fields_refuse_a_condition_nobody_could_have_run(overrides, message):
    with pytest.raises(SamplingDesignError, match=message):
        _condition("naive", **overrides)


def test_only_a_built_in_task_s_hierarchy_pairing_is_refused_by_the_identity():
    """The levels digest is a ``superiors`` condition's whatever the task."""
    with pytest.raises(SamplingDesignError, match="belongs to the hierarchy prompt"):
        _condition("naive", individual_rule_context=HIERARCHY_RULE_CONTEXT, rule_levels_sha256="0" * 64)
    outside = _condition("ranked", individual_rule_context=HIERARCHY_RULE_CONTEXT, rule_levels_sha256="0" * 64,
                         task="demo@1", response_contract=LABEL)
    assert outside.to_dict()["rule_levels_sha256"] == "0" * 64
    with pytest.raises(SamplingDesignError, match="names its levels file by content"):
        _condition("ranked", individual_rule_context=HIERARCHY_RULE_CONTEXT, task="demo@1", response_contract=LABEL)
    with pytest.raises(SamplingDesignError, match="rule_levels_sha256 must be a SHA-256 hex digest"):
        _condition("ranked", individual_rule_context=HIERARCHY_RULE_CONTEXT, rule_levels_sha256="x",
                   task="demo@1", response_contract=LABEL)


def test_batch_cost_s_partition_holds_every_field_once_with_the_judge_fields_apart():
    parts = (batch_cost.JUDGE_FIELDS, batch_cost.PROMPT_FIELDS, batch_cost.PROTOCOL_FIELDS, ("condition_id",))
    names = [name for part in parts for name in part]
    assert sorted(names) == sorted(field.name for field in dataclasses.fields(ConditionSpec))
    assert len(names) == len(set(names))
    assert "judge_extension" in batch_cost.JUDGE_FIELDS
    assert "task" in batch_cost.PROMPT_FIELDS and "response_contract" in batch_cost.PROTOCOL_FIELDS


def test_every_published_condition_s_task_fields_are_the_modelspec_task_s_identity():
    task = ModelSpecTask()
    for row in PUBLISHED:
        spec = _spec(row["payload"])
        identity = task.identity(TaskSettings(spec.prompt_style, spec.individual_rule_context, spec.judge_type))
        rebuilt = dataclasses.replace(spec, prompt_style=identity["prompt_style"],
                                      individual_rule_context=identity["individual_rule_context"],
                                      judge_type=identity["judge_type"], task=identity["task"],
                                      response_contract=identity["response_contract"])
        assert rebuilt.fingerprint == row["payload"]["fingerprint"], row["path"]


def test_the_prompt_table_is_rendered_again_through_the_task():
    scenarios = prompt_scenarios()
    task = ModelSpecTask()
    mismatched = []
    for row in pub.rows_of(PROMPTS, "prompt"):
        text, context = _rule_texts(row["rule_set_sha256"], row["individual_rule_context"],
                                    row.get("rule_levels_sha256"))[row["component_id"]]
        settings = TaskSettings(public_style(row["prompt_style"]), row["individual_rule_context"], row["judge_type"])
        task.check(settings)
        prompt = task.render(text, scenarios[row["scenario_id"]], settings, context_rules=context)
        if prompt_sha256({"system_prompt": prompt, "user_prompt": ""}) != row["prompt_sha256"]:
            mismatched.append((row["condition_id"], row["component_id"]))
    assert mismatched == []
