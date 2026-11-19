from __future__ import annotations

import json

import pytest

from ruleforge.evaluate import validation
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.validation import ComponentSchema, ComponentVerdict
from ruleforge.judges.backend import GenerationOutcome

RULES = [
    {"id": "Rule 1", "rule": "one"},
    {"id": "Rule 2", "rule": "two"},
    {"id": "Meta-Rule 1", "rule": "meta"},
]


def outcome(text, finish_reason="stop", error=None, raw=None):
    return GenerationOutcome(
        text=text, raw_response=text if raw is None else raw, finish_reason=finish_reason, error=error
    )


NOTHING = GenerationOutcome(text=None, raw_response=None, finish_reason=None)


def forced(full="no", rules=None, preempted=None, **extra):
    payload = {
        "rule_judgments": {"Rule 1": "no", "Rule 2": "yes"} if rules is None else rules,
        "preempted_rules": [] if preempted is None else preempted,
        "full_set_judgment": full,
    }
    payload.update(extra)
    return json.dumps(payload)


@pytest.fixture
def label_schema():
    return ComponentSchema.from_rules(RULES, prompt_style="naive-rules-above", forced_json_requirement="complete_rule_map")


@pytest.fixture
def forced_schema():
    return ComponentSchema.from_rules(RULES, prompt_style="forced-grpo", forced_json_requirement="complete_rule_map")


def test_schema_lists_the_full_set_then_every_non_meta_rule(label_schema):
    assert label_schema.component_ids == ("full_ruleset", "Rule 1", "Rule 2")
    assert label_schema.rule_id_index["rule 2"] == "Rule 2"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(prompt_style="terse", forced_json_requirement="label_only", expected_rule_ids=("Rule 1",)), "prompt style"),
        (dict(prompt_style="naive-rules-above", forced_json_requirement="loose", expected_rule_ids=("Rule 1",)), "forced_json_requirement"),
        (dict(prompt_style="naive-rules-above", forced_json_requirement="label_only", expected_rule_ids=("Meta-Rule 1",)), "meta-rules"),
    ],
)
def test_schema_refuses_bad_settings(kwargs, message):
    with pytest.raises(SamplingDesignError, match=message):
        ComponentSchema(**kwargs)


def test_schema_refuses_a_rule_set_with_only_meta_rules():
    with pytest.raises(SamplingDesignError, match="non-meta"):
        ComponentSchema.from_rules([{"id": "Meta-Rule 1"}], prompt_style="naive-rules-above", forced_json_requirement="label_only")


@pytest.mark.parametrize("style", ["naive-rules-above", "forced-grpo"])
@pytest.mark.parametrize(
    ("out", "reason"),
    [
        (outcome("yes", error="ConnectionError: x"), validation.TRANSPORT_ERROR),
        (NOTHING, validation.MISSING_PROVIDER_RESULT),
        (GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="boom", block_failed=True), validation.TRANSPORT_ERROR),
        (outcome("yes", finish_reason="length"), validation.TRUNCATED),
        (outcome(forced(), finish_reason="incomplete:max_output_tokens"), validation.TRUNCATED),
        (outcome(None, raw=""), validation.EMPTY_RESPONSE),
        (outcome("   "), validation.EMPTY_RESPONSE),
        (GenerationOutcome(text=None, raw_response=None, finish_reason="completed"), validation.EMPTY_RESPONSE),
        (GenerationOutcome(text=None, raw_response="", finish_reason="length"), validation.TRUNCATED),
    ],
)
def test_transport_and_truncation_are_invalid_with_their_own_reason(style, out, reason):
    schema = ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement="complete_rule_map")
    verdict = schema.classify("full_ruleset", out)
    assert verdict == ComponentVerdict(valid=False, failure_reason=reason)
    assert verdict.is_transport_failure == (reason in validation.TRANSPORT_REASONS)


@pytest.mark.parametrize("text, judgment", [("yes", "yes"), ("  No \n", "no"), ("UNSURE", "unsure")])
def test_a_bare_label_is_normalized(label_schema, text, judgment):
    assert label_schema.classify("Rule 1", outcome(text)) == ComponentVerdict(valid=True, judgment=judgment)


@pytest.mark.parametrize("text, judgment", [("yes.", "yes"), ("yes\nno", "yes"), ("Final Judgment: no", "no")])
def test_a_label_the_reader_reads_is_valid(label_schema, text, judgment):
    assert label_schema.classify("Rule 1", outcome(text)) == ComponentVerdict(valid=True, judgment=judgment)


@pytest.mark.parametrize("text", ["I think yes", "maybe", "Rule 1: yes\nRule 2: no"])
def test_a_label_the_reader_does_not_read_is_invalid(label_schema, text):
    assert label_schema.classify("Rule 1", outcome(text)) == ComponentVerdict(
        valid=False, failure_reason=validation.INVALID_LABEL
    )


def test_a_complete_forced_object_is_valid_with_its_rule_map(forced_schema):
    text = "Reasoning first.\n" + forced(full="Yes", rules={"rule 2": "unsure", "Rule 1": "no"}, preempted=["Rule 2", 7])
    verdict = forced_schema.classify("full_ruleset", outcome(text))
    assert verdict.valid and verdict.judgment == "yes"
    assert verdict.rule_judgments == {"Rule 1": "no", "Rule 2": "unsure"}
    assert verdict.preempted_rules == ("Rule 2",)


def test_an_inconsistent_but_well_formed_object_is_data_not_a_retry(forced_schema):
    verdict = forced_schema.classify("full_ruleset", outcome(forced(full="no", rules={"Rule 1": "yes", "Rule 2": "yes"})))
    assert verdict.valid and verdict.judgment == "no"


def test_the_last_verdict_object_wins(forced_schema):
    text = forced(full="yes") + "\nOn reflection:\n" + forced(full="no")
    assert forced_schema.classify("full_ruleset", outcome(text)).judgment == "no"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("no json here", validation.INVALID_JSON),
        ('{"rule_judgments": {"Rule 1": "no"}}', validation.MISSING_FULL_SET_JUDGMENT),
        ('{"rule_judgments": {"Rule 1": "no", "Rule 2": "yes", "full_set_judgment": "no"}}',
         validation.MISSING_FULL_SET_JUDGMENT),
        ('{"rule_judgments": {"Rule 1": "no", "Rule 2": "yes"}, "full_set_judiction": "no"}',
         validation.MISSING_FULL_SET_JUDGMENT),
        ('prose {braces} then {"rule_judgments": {}}', validation.MISSING_FULL_SET_JUDGMENT),
        # The outer object is broken: its parsable rule map is no answer, and the failure stays bad JSON.
        ('{"rule_judgments": {"Rule 1": "no"} "full_set_judgment": "no"}', validation.INVALID_JSON),
        ('{"rule_judgments": {"Rule 1": "no"}, "full_set_judgment": "no"', validation.INVALID_JSON),
        ("{not json}", validation.INVALID_JSON),
        (forced(full="maybe"), validation.INVALID_LABEL),
        (forced(full=3), validation.INVALID_LABEL),
        (json.dumps({"full_set_judgment": "no"}), validation.MISSING_REQUIRED_FIELD),
        (json.dumps({"full_set_judgment": "no", "rule_judgments": ["Rule 1"]}), validation.MISSING_REQUIRED_FIELD),
        (forced(rules={"Rule 1": "no"}), validation.MISSING_RULE_IDS),
        (forced(rules={"Rule 1": "no", "Rule 2": "yes", "Rule 3": "no"}), validation.UNEXPECTED_RULE_IDS),
        (forced(rules={"Rule 1": "no", "Rule 2": "yes", "Meta-Rule 1": "no"}), validation.UNEXPECTED_RULE_IDS),
        (forced(rules={"Rule 1": "no", "Rule 2": "yes", "the vibe": "no"}), validation.UNEXPECTED_RULE_IDS),
        (forced(rules={"Rule 1": "no", "Rule 2": "probably"}), validation.INVALID_RULE_LABEL),
    ],
)
def test_forced_full_set_failures_are_named(forced_schema, text, reason):
    assert forced_schema.classify("full_ruleset", outcome(text)) == ComponentVerdict(
        valid=False, failure_reason=reason
    )


@pytest.mark.parametrize("listed", [True, False, 3, 0, 1.5, "Rule 1", "", {"Rule 1": "yes"}, {}])
@pytest.mark.parametrize("require_map", [True, False])
def test_a_preemption_list_that_is_no_list_rejects_the_answer(forced_schema, listed, require_map):
    text = json.dumps(
        {"rule_judgments": {"Rule 1": "no", "Rule 2": "yes"}, "preempted_rules": listed, "full_set_judgment": "no"})
    assert forced_schema.classify_forced(text, require_map=require_map) == ComponentVerdict(
        valid=False, failure_reason=validation.MISSING_REQUIRED_FIELD
    )


def test_a_null_or_absent_preemption_list_is_an_empty_one(forced_schema):
    payload = {"rule_judgments": {"Rule 1": "no", "Rule 2": "yes"}, "full_set_judgment": "no"}
    for text in (json.dumps(payload), json.dumps({**payload, "preempted_rules": None})):
        verdict = forced_schema.classify_forced(text, require_map=True)
        assert verdict.valid and verdict.preempted_rules == ()


def test_a_repeated_rule_key_is_a_duplicate_even_when_spelled_differently(forced_schema):
    text = '{"rule_judgments": {"Rule 1": "no", "rule1": "yes", "Rule 2": "no"}, "preempted_rules": [], "full_set_judgment": "no"}'
    assert forced_schema.classify("full_ruleset", outcome(text)).failure_reason == validation.DUPLICATE_RULE_IDS


def test_single_rule_forced_components_need_only_the_label(forced_schema):
    text = json.dumps({"rule_judgments": {"Rule 1": "yes"}, "preempted_rules": [], "full_set_judgment": "yes"})
    verdict = forced_schema.classify("Rule 1", outcome(text))
    assert verdict.valid and verdict.judgment == "yes" and verdict.rule_judgments is None


def test_label_only_requirement_skips_the_rule_map_on_the_full_set():
    schema = ComponentSchema.from_rules(RULES, prompt_style="forced-grpo", forced_json_requirement="label_only")
    verdict = schema.classify("full_ruleset", outcome(forced(rules={"Rule 1": "no"})))
    assert verdict.valid and verdict.rule_judgments is None


#: The forced styles whose contract asks for no preempted_rules: the paper's.
UNLISTED = ("forced",)


def without_preemption(full="no", rules=None):
    """A forced answer as a contract without ``preempted_rules`` asks for it: no preempted_rules at all."""
    payload = json.loads(forced(full=full, rules=rules))
    del payload["preempted_rules"]
    return json.dumps(payload)


FORCED_TEXTS = [
    forced(full="Yes", rules={"rule 2": "unsure", "Rule 1": "no"}, preempted=["Rule 2", 7]),
    forced(full="no", rules={"Rule 1": "yes", "Rule 2": "yes"}),
    without_preemption(),
    without_preemption(full="unsure", rules={"Rule 2": "no", "rule1": "no"}),
    forced(full="yes") + "\nOn reflection:\n" + without_preemption(full="no"),
    "no json here",
    forced(full="maybe"),
    json.dumps({"full_set_judgment": "no"}),
    without_preemption(rules={"Rule 1": "no"}),
    forced(rules={"Rule 1": "no", "Rule 2": "yes", "Rule 3": "no"}),
    without_preemption(rules={"Rule 1": "no", "Rule 2": "probably"}),
    '{"rule_judgments": {"Rule 1": "no", "rule1": "yes", "Rule 2": "no"}, "full_set_judgment": "no"}',
]


@pytest.mark.parametrize("style", UNLISTED)
@pytest.mark.parametrize("requirement", ["complete_rule_map", "label_only"])
@pytest.mark.parametrize("component", ["full_ruleset", "Rule 1"])
@pytest.mark.parametrize("text", FORCED_TEXTS)
def test_an_unlisted_forced_prompt_is_validated_exactly_as_the_forced_prompt(text, component, requirement, style):
    schemas = {each: ComponentSchema.from_rules(RULES, prompt_style=each, forced_json_requirement=requirement)
               for each in ("forced-grpo", style)}
    forced_verdict = schemas["forced-grpo"].classify(component, outcome(text))
    other = schemas[style].classify(component, outcome(text))
    assert other.preempted_rules == ()
    assert other == ComponentVerdict(**{**forced_verdict.__dict__, "preempted_rules": ()})


@pytest.mark.parametrize("style", UNLISTED)
def test_an_answer_without_preempted_rules_means_nothing_preempted_never_unparseable(forced_schema, style):
    schema = ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement="complete_rule_map")
    for each in (schema, forced_schema):
        verdict = each.classify("full_ruleset", outcome(without_preemption(full="yes")))
        assert verdict.valid and verdict.judgment == "yes" and verdict.preempted_rules == ()
        assert verdict.rule_judgments == {"Rule 1": "no", "Rule 2": "yes"}


@pytest.mark.parametrize("style", UNLISTED)
def test_a_list_an_unlisted_forced_judge_writes_anyway_is_not_read(forced_schema, style):
    schema = ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement="complete_rule_map")
    text = forced(full="no", preempted=["Rule 1"])
    assert forced_schema.classify("full_ruleset", outcome(text)).preempted_rules == ("Rule 1",)
    verdict = schema.classify("full_ruleset", outcome(text))
    assert verdict.valid and verdict.preempted_rules == ()
    assert schema.classify_forced(text, require_map=False) == ComponentVerdict(valid=True, judgment="no")


def test_strict_think_applies_to_an_unlisted_forced_prompt_as_to_the_forced_one():
    raw = "<think>" + without_preemption(full="yes") + "</think>" + without_preemption(full="no")
    for style in ("forced-grpo", *UNLISTED):
        verdict = strict(style).classify("full_ruleset", thought(raw, stripped="ignored"))
        assert verdict.valid and verdict.judgment == "no"
    for style in UNLISTED:
        assert strict(style).classify("full_ruleset", thought("<think>open")).failure_reason == (
            validation.UNCLOSED_THINK)


LABEL_TEXTS = ["yes", "  No \n", "UNSURE", "yes.", "I think yes", "maybe", "yes\nno", "Final Judgment: yes",
               "**unsure**", "Rule 1: yes\nRule 2: no\n\nAnswer: no", forced(full="yes"), "   ",
               "---\nno", "no<tool_call|>no", "yes<tool_call|>no", "Final Judgment: no<channel|>"]


@pytest.mark.parametrize("component", ["full_ruleset", "Rule 1"])
@pytest.mark.parametrize("text", LABEL_TEXTS)
def test_the_revised_label_styles_are_validated_exactly_as_the_published_label_styles(text, component):
    """The revised naive and encouraged prompts answer with a bare label, as the published naive prompt does."""
    for published, revised in (("naive-rules-above", "naive"), ("naive-rules-above", "encouraged")):
        verdicts = {
            style: ComponentSchema.from_rules(
                RULES, prompt_style=style, forced_json_requirement="complete_rule_map",
            ).classify(component, outcome(text))
            for style in (published, revised)
        }
        assert verdicts[revised] == verdicts[published]
        assert verdicts[revised].rule_judgments is None and verdicts[revised].preempted_rules == ()


def strict(style="forced-grpo", requirement="label_only"):
    return ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement=requirement, strict_think=True)


def thought(raw, *, stripped=None, finish_reason="stop"):
    """A thinking judge's outcome: ``raw`` as vLLM returned it, the stripper's text beside it."""
    return GenerationOutcome(text=raw if stripped is None else stripped, raw_response=raw, finish_reason=finish_reason)


@pytest.mark.parametrize("style", ["naive-rules-above", "forced-grpo"])
def test_strict_think_refuses_a_finished_output_whose_thinking_never_closed(style):
    answer = "no" if style == "naive-rules-above" else forced(full="no")
    raw = f"<think>The full set says {answer}"
    lax = ComponentSchema.from_rules(RULES, prompt_style=style, forced_json_requirement="label_only")
    assert lax.classify("Rule 1", thought(raw, stripped=answer)).valid
    verdict = strict(style).classify("Rule 1", thought(raw, stripped=answer))
    assert verdict == ComponentVerdict(valid=False, failure_reason=validation.UNCLOSED_THINK)
    assert not verdict.is_transport_failure


def test_strict_think_reads_the_answer_after_the_last_closing_tag():
    raw = "<think>maybe yes</think>\n\n" + forced(full="yes") + "<think>second thoughts</think>\n" + forced(full="no")
    # the stripper's text (both blocks removed, the answers run together) is not what is validated
    verdict = strict().classify("full_ruleset", thought(raw, stripped="ignored"))
    assert verdict.valid and verdict.judgment == "no"
    # an orphaned closing tag (the template opened the block) reads the same way
    assert strict("naive-rules-above").classify("Rule 1", thought("reasoning</think>\n\nunsure", stripped="x")).judgment == "unsure"
    assert validation.strict_think_answer(raw) == "\n" + forced(full="no")
    assert validation.strict_think_answer("no tag") == "no tag"


@pytest.mark.parametrize(
    ("out", "reason"),
    [
        # a closed block with no answer after it: the stripper's fallback would hand on the thinking
        (thought("<think>it is yes</think>\n\n", stripped="<think>it is yes</think>"), validation.EMPTY_RESPONSE),
        (thought("", stripped=None), validation.EMPTY_RESPONSE),
        (GenerationOutcome(text=None, raw_response=None, finish_reason="stop"), validation.EMPTY_RESPONSE),
        # cut short inside the thinking: truncated, before anything else
        (thought("<think>still going", finish_reason="length"), validation.TRUNCATED),
        (thought("<think>x</think>maybe"), validation.INVALID_JSON),
    ],
)
def test_strict_think_keeps_the_other_reasons(out, reason):
    assert strict().classify("full_ruleset", out) == ComponentVerdict(valid=False, failure_reason=reason)


def test_strict_think_is_a_schema_setting_off_by_default():
    assert ComponentSchema.from_rules(RULES, prompt_style="naive-rules-above", forced_json_requirement="label_only").strict_think is False
    assert validation.UNCLOSED_THINK not in validation.TRANSPORT_REASONS
