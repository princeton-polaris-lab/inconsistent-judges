from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import pytest

from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.aggregate import analyze_campaign, style_families
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    build_plan,
    load_conditions,
    run_plan,
    validate_campaign_cfg,
)
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.evaluate.identity import ConditionSpec
from ruleforge.evaluate.ledger import DrawLedger
from ruleforge.evaluate.runner import STATE_COMPLETE, DrawRunner
from ruleforge.tasks.spec import TaskSpecAdapter
from ruleforge.tasks.spec import build_task_spec
from ruleforge.tasks import FORCED_JSON, LABEL, ModelSpecTask, TaskError, register_task
from ruleforge.tasks import registry as registry_module
from ruleforge.core.artifacts import load_jsonl_records
from tests.test_sampling_campaign import compose_campaign
from tests.test_sampling_runner import FakeBackend
from tests.test_tasks import DemoTask


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    monkeypatch.setattr(registry_module, "_REGISTERED", {})
    register_task("demo", DemoTask)


def _arm(tmp_path: Path, style: str, *, judge: FakeBackend, **overrides):
    """One condition of the demo task in the root ``tmp_path/camp``: planned, run, and its plan returned."""
    cfg = compose_campaign(tmp_path, **{"task_spec": "demo", "eval_cfg.reasoning_prompt_style": style,
                                        "campaign_root": str(tmp_path / "camp"), **overrides})
    task_spec = build_task_spec(cfg)
    assert isinstance(task_spec, TaskSpecAdapter)
    judge.max_new_tokens = 1024
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    completions = run_plan(plan, backend=judge, task_spec=task_spec)
    assert {draw: record["state"] for draw, record in completions.items()} == {0: STATE_COMPLETE, 1: STATE_COMPLETE}
    return plan


def _components(plan, draw=0):
    ledger = DrawLedger(plan.condition_root / f"draw-{draw:03d}", campaign_id=plan.campaign_id,
                        condition_id=plan.condition.condition_id, decoder_draw_id=draw)
    return [component for record in ledger.load().records for component in record.components]


def _preempting_answer():
    return json.dumps({"rule_judgments": {"Rule 1": "yes", "Rule 2": "no"}, "preempted_rules": ["Rule 1"],
                       "full_set_judgment": "no"})


def test_an_outside_forced_arm_records_its_task_and_contract_and_reads_no_preemption(tmp_path):
    """The forced arm's answers list a preempted rule."""
    judge = FakeBackend(style="forced-grpo", script={(i, "full_ruleset"): [_preempting_answer()] for i in range(3)})
    plan = _arm(tmp_path, "json", judge=judge)
    condition = plan.condition
    assert (condition.task, condition.response_contract, condition.contract) == ("demo@1.0", FORCED_JSON, FORCED_JSON)
    recorded = json.loads((plan.condition_root / CONDITION_FNAME).read_text(encoding="utf-8"))
    assert (recorded["task"], recorded["response_contract"]) == ("demo@1.0", FORCED_JSON)
    components = _components(plan)
    full = [c for c in components if c.component_id == "full_ruleset"]
    assert full and all(c.valid and c.rule_judgments == {"Rule 1": "yes", "Rule 2": "no"} for c in full)
    assert all(component.preempted_rules == () for component in components)
    raw = load_jsonl_records(plan.condition_root / "draw-000" / "raw_outputs.jsonl")
    assert any('"preempted_rules": ["Rule 1"]' in (row["completion"] or "") for row in raw)


def test_the_built_in_forced_contract_still_reads_the_preempted_rules(tmp_path, modelspec):
    """The same answer under ``forced-grpo`` of the Model Spec task: its contract asks for the list, which is read."""
    cfg = compose_campaign(tmp_path, **{"eval_cfg.reasoning_prompt_style": "forced-grpo",
                                        "campaign_root": str(tmp_path / "builtin")})
    judge = FakeBackend(style="forced-grpo", script={(i, "full_ruleset"): [_preempting_answer()] for i in range(3)})
    judge.max_new_tokens = 1024
    task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    run_plan(plan, backend=judge, task_spec=task_spec)
    assert "task" not in json.loads((plan.condition_root / CONDITION_FNAME).read_text(encoding="utf-8"))
    full = [c for c in _components(plan) if c.component_id == "full_ruleset"]
    assert full and all(c.preempted_rules == ("Rule 1",) for c in full)


def test_an_outside_label_arm_and_forced_arm_share_a_root_a_style_family_and_an_analysis(tmp_path):
    forced_plan = _arm(tmp_path, "json", judge=FakeBackend(style="forced-grpo"))
    label_plan = _arm(tmp_path, "plain", judge=FakeBackend(style="naive"))
    assert (label_plan.condition.response_contract, label_plan.condition.contract) == (LABEL, LABEL)
    conditions = load_conditions(tmp_path / "camp")
    assert set(conditions) == {forced_plan.condition.condition_id, label_plan.condition.condition_id}
    assert sorted(sorted(members) for members in style_families(conditions).values()) == [sorted(conditions)]
    payload = analyze_campaign(tmp_path / "camp")
    # The forced arm gets a same-call rate by its contract (its own rule maps); the label arm none; one style family.
    assert set(payload["forced_conditions"]) == {forced_plan.condition.condition_id}
    assert payload["forced_conditions"][forced_plan.condition.condition_id]["method_b_rate"]
    assert [sorted(group["members"]) for group in payload["style_families"].values()] == [sorted(conditions)]


def test_an_outside_style_judged_with_its_references_is_approved_by_its_task(tmp_path):
    plan = _arm(tmp_path, "referenced", judge=FakeBackend(style="naive"),
                **{"eval_cfg.individual_rule_context": "implicated"})
    condition = plan.condition
    assert (condition.prompt_style, condition.individual_rule_context, condition.response_contract) == (
        "referenced", "implicated", LABEL)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"task_spec": "demo", "eval_cfg.reasoning_prompt_style": "nope"},
         "reasoning_prompt_style='nope' must be one of ('plain', 'json', 'referenced')"),
        ({"task_spec": "demo", "eval_cfg.reasoning_prompt_style": "referenced"},
         "eval_cfg: referenced is judged with the implicated context"),
        ({"task_spec": "demo", "eval_cfg.reasoning_prompt_style": "plain", "interpretation": "Telepathy",
          "eval_cfg.judge_types": "[Telepathy]"}, "interpretation: judge type 'Telepathy'"),
        ({"task_spec": "absent"}, "task_spec: no task 'absent'"),
    ],
)
def test_preflight_asks_the_campaign_s_task(tmp_path, overrides, message):
    with pytest.raises(CampaignRunError, match=message.replace("(", r"\(").replace(")", r"\)")):
        validate_campaign_cfg(compose_campaign(tmp_path, **overrides))


class EditedDemoTask(DemoTask):
    """The demo task edited in place, name and version kept: ``plain`` asks for the forced JSON object."""

    styles = {**DemoTask.styles, "plain": FORCED_JSON}


def test_the_runner_refuses_a_task_spec_that_is_not_its_condition_s_task(tmp_path, task_cfg):
    """The condition's contract reads the answers and the task spec renders the prompts."""
    plan = _arm(tmp_path, "plain", judge=FakeBackend(style="naive"))

    def runner(task_spec, **spec_changes):
        draw_plan = dataclasses.replace(plan, **spec_changes)
        return campaign_module.draw_runner(draw_plan, 0, backend=FakeBackend(style="naive"), task_spec=task_spec)

    other = TaskSpecAdapter(ModelSpecTask(), task_cfg(reasoning_prompt_style="naive"))
    with pytest.raises(CampaignRunError, match=r"records task 'demo@1.0', prompt_style 'plain', response_contract "
                                               r"'label' but the task spec renders task 'modelspec', prompt_style "
                                               r"'naive', response_contract None; build the plan"):
        runner(other)
    edited = TaskSpecAdapter(EditedDemoTask(), task_cfg(task_spec="demo", reasoning_prompt_style="plain"))
    with pytest.raises(CampaignRunError, match=r"records response_contract 'label' but the task spec renders "
                                               r"response_contract 'forced_json'"):
        runner(edited)
    implicated = TaskSpecAdapter(DemoTask(), task_cfg(task_spec="demo", reasoning_prompt_style="plain",
                                                      individual_rule_context="implicated"))
    with pytest.raises(CampaignRunError, match=r"records individual_rule_context 'isolated' but the task spec "
                                               r"renders individual_rule_context 'implicated'"):
        runner(implicated)
    same = TaskSpecAdapter(DemoTask(), task_cfg(task_spec="demo", reasoning_prompt_style="plain"))
    with pytest.raises(CampaignRunError, match=r"records judge_type 'No_Interpretation' but the task spec renders "
                                               r"judge_type 'Narrow'"):
        runner(same, judge_type="Narrow")
    with pytest.raises(CampaignRunError, match=r"condition '[^']+': judge type 'Telepathy'"):
        runner(same, judge_type="Telepathy")  # the task refuses the settings it would render
    assert isinstance(runner(same), DrawRunner)


def _built_in_condition(style="forced-grpo", context="isolated", **task) -> ConditionSpec:
    """A condition of the built-in task, or of the task ``task`` names (its ``task`` and ``response_contract``)."""
    return ConditionSpec(
        condition_id=f"m/{style}", model_name="m", model_type="local", prompt_style=style,
        individual_rule_context=context, judge_type="No_Interpretation", evaluation_mode="both",
        rule_set_name="rules.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
        forced_json_requirement="complete_rule_map", **task)


def test_the_grpo_rows_task_spec_refuses_another_task_s_condition():
    """The GRPO rows' task spec renders the Model Spec task's prompts alone."""
    from ruleforge.grpo import grpo_data

    built_in = _built_in_condition("naive")
    outside = dataclasses.replace(built_in, task="demo@1.0", response_contract=LABEL)
    assert grpo_data.condition_task_spec(built_in).reasoning_prompt_style == "naive"
    with pytest.raises(grpo_data.GrpoDataError, match="condition m/naive: its prompts are task 'demo@1.0''s"):
        grpo_data.condition_task_spec(outside)


def test_a_task_spec_whose_task_refuses_its_settings_registers_no_condition():
    def refuse(judge_type):
        raise TaskError(f"no {judge_type} here")

    stand_in = types.SimpleNamespace(task_spec_name="demo", task_identity=refuse)
    with pytest.raises(CampaignRunError, match="task spec demo: no Narrow here"):
        campaign_module.condition_task_identity(stand_in, "Narrow")
