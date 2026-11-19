from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from functools import lru_cache

import pytest

from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.evaluate.identity import (
    FULL_RULESET_COMPONENT,
    LABEL_RULE,
    derive_generation_seed,
    request_block_id,
)
from ruleforge.core.digests import request_namespace
from ruleforge.core.record_names import public_style
from ruleforge.evaluate.ledger import (
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_TRANSPORT_FAILED,
    AttemptRecord,
    ComponentRecord,
    raw_output_sha256,
)
from ruleforge.evaluate.runner import (
    BUNDLE_FAILURE_COMPONENT_INVALID,
    BUNDLE_FAILURE_TRANSPORT,
    bundle_status,
)
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.evaluate.validation import (
    EMPTY_RESPONSE,
    INVALID_JSON,
    INVALID_LABEL,
    INVALID_RULE_LABEL,
    MISSING_FULL_SET_JUDGMENT,
    MISSING_PROVIDER_RESULT,
    MISSING_REQUIRED_FIELD,
    TRANSPORT_ERROR,
    TRUNCATED,
    UNCLOSED_THINK,
    ComponentSchema,
)
from ruleforge.core.artifacts import compute_scenario_id, load_jsonl_records
from ruleforge.judges.backend import GenerationOutcome
from ruleforge.prompts.parse import THINK_CLOSE
from ruleforge.tasks.base import FORCED_JSON
from ruleforge.rulesets.levels import load_rule_levels
from tests.identity import published as pub

SEEDS = pub.read_fixture(pub.SEEDS)
PROMPTS = pub.read_fixture(pub.PROMPTS)
SETTLEMENT = pub.read_fixture(pub.SETTLEMENT)
RULE_FILES = pub.repo_rule_files()
FILE_BY_DIGEST = {sha: path for (_, sha), path in RULE_FILES.items()}

PROMPT_COMBOS = {
    ("forced-grpo", "isolated", "No_Interpretation"): 50,
    ("forced", "isolated", "No_Interpretation"): 50,
    ("encouraged", "isolated", "No_Interpretation"): 50,
    ("naive-rules-above", "implicated", "No_Interpretation"): 50,
    ("naive-rules-above", "isolated", "No_Interpretation"): 44,
    ("naive-hierarchy", "superiors", "No_Interpretation"): 43,
}


def test_the_seed_table_holds_ninety_recorded_seeds_over_attempts_and_components():
    seeds = pub.rows_of(SEEDS, "seed")
    assert len(seeds) == 90
    assert {row["semantic_attempt_id"] > 0 for row in seeds} == {True, False}
    assert {row["component_id"] == FULL_RULESET_COMPONENT for row in seeds} == {True, False}
    assert len({row["condition_id"] for row in seeds}) >= 10


def test_every_recorded_generation_seed_is_derived_from_its_tuple():
    mismatched = [
        row for row in pub.rows_of(SEEDS, "seed")
        if derive_generation_seed(
            generation_seed_base=row["generation_seed_base"], campaign_id=row["campaign_id"],
            condition_id=row["condition_id"], scenario_id=row["scenario_id"],
            decoder_draw_id=row["decoder_draw_id"], semantic_attempt_id=row["semantic_attempt_id"],
            component_id=row["component_id"],
        ) != row["generation_seed"]
    ]
    assert mismatched == []


def test_every_recorded_block_id_and_provider_namespace_is_derived_from_the_block():
    blocks = pub.rows_of(SEEDS, "block")
    assert len(blocks) == 19
    for row in blocks:
        index = int(re.fullmatch(r".*/block-(\d{4})", row["request_block_id"]).group(1))
        assert request_block_id(
            campaign_id=row["campaign_id"], condition_id=row["condition_id"], decoder_draw_id=row["decoder_draw_id"],
            semantic_attempt_id=row["semantic_attempt_id"], block_index=index,
        ) == row["request_block_id"]
        assert row["provider_request_ref"].split("-", 1)[0] == request_namespace(row["request_block_id"])


@lru_cache(maxsize=None)
def _rule_texts(rule_set_sha256: str, context: str, levels_sha256: str | None = None,
                ) -> dict[str, tuple[str, object]]:
    """``{component id: (rule text, context rules)}`` as the campaign runner composes them."""
    path = FILE_BY_DIGEST[rule_set_sha256]
    rules = [dict(rule) for rule in load_jsonl_records(path)]
    levels = None if levels_sha256 is None else load_rule_levels(path, FILE_BY_DIGEST[levels_sha256]).by_id
    texts = {FULL_RULESET_COMPONENT: (rule_set_text(rules, context, levels), None)}
    variants, _ = build_rule_variants(rules, context, levels)
    texts.update({rule_id: (text, rules_in_context) for rule_id, text, rules_in_context in variants})
    return texts


def modelspec_task_spec(prompt_style: str, judge_type: str, individual_rule_context: str = "isolated"):
    """A ModelSpec task spec that renders ``prompt_style`` prompts for ``judge_type``; no file it names is opened."""
    from omegaconf import OmegaConf

    from ruleforge.tasks.spec import ModelSpecTaskSpec

    cfg = OmegaConf.create({
        "task_spec": "modelspec",
        "rule_amendment": {"initial_rule_set_path": "unused.jsonl"},
        "scenario_generation": {"good_seed_scenario_path": "unused.jsonl", "bad_seed_scenario_path": "unused.jsonl"},
        "agent_model": {},
        "judge_model": {},
        "eval_cfg": {"judge_types": [judge_type], "reasoning_prompt_style": prompt_style,
                     "individual_rule_context": individual_rule_context},
    })
    return ModelSpecTaskSpec(cfg)


def task_spec_render(row, scenario) -> str:
    """The prompt the published campaigns' task spec renders for a prompt-table row."""
    text, context = _rule_texts(row["rule_set_sha256"], row["individual_rule_context"],
                                row.get("rule_levels_sha256"))[row["component_id"]]
    task_spec = modelspec_task_spec(public_style(row["prompt_style"]), row["judge_type"],
                                    row["individual_rule_context"])
    return task_spec.build_eval_prompt(rule_text=text, scenario=dict(scenario), judge_type=row["judge_type"],
                                       context_rules=context)


def prompt_scenarios() -> dict[str, dict]:
    scenarios = {row["scenario_id"]: row["row"] for row in pub.rows_of(PROMPTS, "scenario")}
    assert all(compute_scenario_id(row) == sid for sid, row in scenarios.items())
    return scenarios


def test_the_prompt_table_covers_every_published_style_and_rule_context():
    rows = pub.rows_of(PROMPTS, "prompt")
    assert dict(Counter((public_style(r["prompt_style"]), r["individual_rule_context"], r["judge_type"])
                        for r in rows)) == PROMPT_COMBOS
    assert {r["rule_set_name"] for r in rows} == {"openai_rules_all.jsonl", "openai_rules_all_full.jsonl",
                                                  "seed5_winner.jsonl", "seed9_runner_up.jsonl"}
    assert {r["component_id"] == FULL_RULESET_COMPONENT for r in rows} == {True, False}


def test_every_recorded_prompt_digest_is_the_prompt_rendered_again():
    scenarios = prompt_scenarios()
    mismatched = [
        (row["condition_id"], row["scenario_id"][:12], row["component_id"])
        for row in pub.rows_of(PROMPTS, "prompt")
        if prompt_sha256({"system_prompt": task_spec_render(row, scenarios[row["scenario_id"]]), "user_prompt": ""})
        != row["prompt_sha256"]
    ]
    assert mismatched == []


def recorded_outcome(component, raw, *, strict_think: bool) -> GenerationOutcome:
    """The generation outcome a component was validated from, rebuilt from its raw row."""
    if raw is None:
        if component["failure_reason"] == TRANSPORT_ERROR:
            return GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="transport error")
        assert component["failure_reason"] == MISSING_PROVIDER_RESULT, component
        return GenerationOutcome(text=None, raw_response=None, finish_reason=None)
    raw_response = ""
    if strict_think and not raw.get("text_elided"):
        raw_response = (THINK_CLOSE if raw["raw_think_closed"] else "") + raw["raw_answer"]
    return GenerationOutcome(text=raw["completion"], raw_response=raw_response, finish_reason=raw["finish_reason"],
                             error=raw["error"])


def _schema(row) -> ComponentSchema:
    condition = row["condition"]
    rules = load_jsonl_records(RULE_FILES[(condition["rule_set_name"], condition["rule_set_sha256"])])
    return ComponentSchema.from_rules(
        rules, prompt_style=public_style(condition["prompt_style"]),
        forced_json_requirement=condition["forced_json_requirement"], evaluation_mode=condition["evaluation_mode"],
        strict_think=bool(condition["strict_think"]),
    )


VERDICT_FIELDS = ("valid", "failure_reason", "judgment", "rule_judgments", "preempted_rules")
#: Recorded runs name a JSON answer without a top-level verdict invalid_json.
RECORDED_REASON = {MISSING_FULL_SET_JUDGMENT: INVALID_JSON}


def settle(row, schema: ComponentSchema) -> tuple[list[ComponentRecord], list[str]]:
    """Each component of a recorded attempt row classified again; and every field that differs from the record."""
    raws = {raw["row"]["component_id"]: raw["row"] for raw in row["raw"]}
    strict = bool(row["condition"]["strict_think"])
    rebuilt, differences = [], []
    for component in row["attempt"]["components"]:
        raw = raws.get(component["component_id"])
        if raw is not None:
            assert raw["raw_output_sha256"] == component["raw_output_sha256"]
        verdict = schema.classify(component["component_id"], recorded_outcome(component, raw, strict_think=strict))
        verdict = replace(verdict, failure_reason=RECORDED_REASON.get(verdict.failure_reason, verdict.failure_reason))
        recorded = ComponentRecord.from_dict(component)
        for name in VERDICT_FIELDS:
            if getattr(verdict, name) != getattr(recorded, name):
                differences.append(f"{component['component_id']}.{name}")
        rebuilt.append(ComponentRecord.from_dict({**component, **{name: getattr(verdict, name)
                                                                  for name in VERDICT_FIELDS}}))
    return rebuilt, differences


def test_the_settlement_table_covers_every_recorded_outcome_kind():
    categories = Counter(json.dumps(row["category"]) for row in SETTLEMENT)
    assert len(categories) == 62 and max(categories.values()) <= 2
    assert {row["category"][4] for row in SETTLEMENT} == {"lenient", "lenient_last_line", "lenient_markup"}
    statuses = {row["attempt"]["status"] for row in SETTLEMENT}
    assert statuses == {STATUS_ACCEPTED, STATUS_REJECTED}
    assert {row["category"][2] for row in SETTLEMENT} == {"scenario_bundle", "component"}
    reasons = {reason for row in SETTLEMENT for reason in row["category"][7]}
    assert reasons == {"duplicate_rule_ids", "empty_response", "invalid_json", "invalid_label", "missing_rule_ids",
                       "truncated", "unclosed_think", "unexpected_rule_ids"}


def _read_as_recorded(row, schema: ComponentSchema) -> bool:
    return row["condition"]["label_rule"] != LABEL_RULE and schema.response_contract != FORCED_JSON


def test_every_recorded_attempt_row_is_settled_again_from_its_raw_outputs():
    failures, as_recorded = [], set()
    for row in SETTLEMENT:
        record = AttemptRecord.from_dict(row["attempt"])
        assert record.to_dict() == row["attempt"]
        schema = _schema(row)
        if _read_as_recorded(row, schema):
            as_recorded.add(row["root"])
            rebuilt, differences = list(record.components), []
        else:
            rebuilt, differences = settle(row, schema)
        status = bundle_status(rebuilt)
        if differences or status != (record.status, record.failure_reason):
            failures.append(f"{row['source']['path']}:{row['source']['line']}: {differences} {status}")
    assert failures == []
    assert as_recorded == {"hierarchy-aware-2026-10-pop-a22af097"}


def test_a_recorded_answer_without_a_top_level_verdict_stays_rejected_under_its_own_reason():
    fresh = Counter()
    for row in SETTLEMENT:
        if _read_as_recorded(row, schema := _schema(row)):
            continue
        raws = {raw["row"]["component_id"]: raw["row"] for raw in row["raw"]}
        for component in row["attempt"]["components"]:
            if component["failure_reason"] == INVALID_JSON:
                verdict = schema.classify(component["component_id"], recorded_outcome(
                    component, raws.get(component["component_id"]), strict_think=bool(row["condition"]["strict_think"])))
                assert not verdict.valid
                fresh[verdict.failure_reason] += 1
    assert set(fresh) == {INVALID_JSON, MISSING_FULL_SET_JUDGMENT}


RULES = [{"id": "Rule 1", "rule": "Be kind."}, {"id": "Rule 2", "rule": "Be brief."}]


def _schema_for(style: str, **kwargs) -> ComponentSchema:
    return ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement="complete_rule_map", **kwargs)


@pytest.mark.parametrize(
    ("style", "outcome", "options", "expected"),
    [
        # A provider error: transport, consuming no attempt.
        ("naive-rules-above", GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="429"), {},
         TRANSPORT_ERROR),
        # Nothing came back for the request: transport too.
        ("naive-rules-above", GenerationOutcome(text=None, raw_response=None, finish_reason=None), {}, MISSING_PROVIDER_RESULT),
        # Returned, but cut at the token ceiling: semantic, whatever the text.
        ("naive-rules-above", GenerationOutcome(text="yes", raw_response="yes", finish_reason="length"), {}, TRUNCATED),
        ("forced-grpo", GenerationOutcome(text='{"full_set_judgment": "no"}', raw_response="<think>still going",
                                          finish_reason="stop"), {"strict_think": True}, UNCLOSED_THINK),
        ("forced-grpo", GenerationOutcome(text='{"full_set_judgment": "no", "rule_judgments": {"Rule 1": "probably", '
                                             '"Rule 2": "no"}}', raw_response="x", finish_reason="stop"), {},
         INVALID_RULE_LABEL),
        # Under strict_think the answer is the text after the last </think>: blank is empty.
        ("naive-rules-above", GenerationOutcome(text="no", raw_response="<think>no</think>  ", finish_reason="stop"),
         {"strict_think": True}, EMPTY_RESPONSE),
        # A forced full-set answer with no rule map under complete_rule_map.
        ("forced-grpo", GenerationOutcome(text='{"full_set_judgment": "no"}', raw_response="x", finish_reason="stop"),
         {}, MISSING_REQUIRED_FIELD),
        ("forced", GenerationOutcome(text="not json", raw_response="not json", finish_reason="stop"), {},
         INVALID_JSON),
    ],
)
def test_an_outcome_no_root_recorded_gets_the_contract_s_verdict(style, outcome, options, expected):
    verdict = _schema_for(style, **options).classify(FULL_RULESET_COMPONENT, outcome)
    assert (verdict.valid, verdict.failure_reason) == (False, expected)
    assert verdict.is_transport_failure == (expected in (TRANSPORT_ERROR, MISSING_PROVIDER_RESULT))


def _component(component_id: str, verdict) -> ComponentRecord:
    return ComponentRecord(component_id=component_id, valid=verdict.valid, failure_reason=verdict.failure_reason,
                           judgment=verdict.judgment, finish_reason="stop", generation_seed=None, seed_sent=False,
                           raw_output_sha256=raw_output_sha256("x"))


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        (("yes", "no", "no"), (STATUS_ACCEPTED, None)),
        # One returned invalid: rejected, the attempt consumed (a transport sibling does not change that).
        (("yes", "maybe", None), (STATUS_REJECTED, BUNDLE_FAILURE_COMPONENT_INVALID)),
        # Transport alone: transport-failed, nothing consumed.
        (("yes", None, "no"), (STATUS_TRANSPORT_FAILED, BUNDLE_FAILURE_TRANSPORT)),
    ],
)
def test_a_bundle_s_status_is_decided_by_its_components(outcomes, expected):
    schema = _schema_for("naive-rules-above")
    components = []
    for component_id, text in zip(schema.component_ids, outcomes):
        outcome = (GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="503") if text is None
                   else GenerationOutcome(text=text, raw_response=text, finish_reason="stop"))
        components.append(_component(component_id, schema.classify(component_id, outcome)))
    assert bundle_status(components) == expected


@pytest.mark.parametrize(
    ("answers", "reasons"),
    [
        # An empty answer beside a label the one rule cannot read (recorded under lenient by Haiku's ruleset-size bank).
        ((("", "stop"), ("maybe", "stop"), ("no", "stop")), (EMPTY_RESPONSE, INVALID_LABEL, None)),
        ((("maybe", "stop"), ("yes", "length"), ("no", "stop")), (INVALID_LABEL, TRUNCATED, None)),
    ],
)
def test_a_bundle_s_two_invalid_components_keep_their_reasons_and_reject_it(answers, reasons):
    schema = _schema_for("naive-rules-above")
    components = [_component(component_id, schema.classify(component_id, GenerationOutcome(
        text=text, raw_response=text, finish_reason=finish)))
                  for component_id, (text, finish) in zip(schema.component_ids, answers)]
    assert tuple(component.failure_reason for component in components) == reasons
    assert bundle_status(components) == (STATUS_REJECTED, BUNDLE_FAILURE_COMPONENT_INVALID)
