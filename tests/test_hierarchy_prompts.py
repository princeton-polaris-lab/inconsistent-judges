from __future__ import annotations

import dataclasses
import hashlib
import itertools
import re

import pytest

from ruleforge.experiments.hierarchy.analysis import APPROVED_PROMPT_SHA256
from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.prompts.context import _CONTEXT_RULES_INSTRUCTIONS, IMPLICATED_RULES_HEADER, build_context_rules_section
from ruleforge.evaluate.identity import ConditionSpec
from ruleforge.core.errors import SamplingDesignError
from ruleforge.prompts.styles import (
    HIERARCHY_RULE_CONTEXT,
    ENCOURAGED,
    FORCED_GRPO,
    HIGHER_RANKED_RULES_HEADER,
    NAIVE_RULES_ABOVE,
    NAIVE_HIERARCHY,
    NAIVE,
    RANKING_BLOCK,
    VALID_REASONING_PROMPT_STYLES,
    hierarchy_refusal,
    no_interpretation_only_refusal,
    ranking_section,
    reasoning_block,
)
from ruleforge.core.artifacts import load_jsonl_records
from ruleforge.rulesets.rules import format_rule
from ruleforge.rulesets.levels import LEVEL_ORDER, load_rule_levels

PLACEHOLDER = {"prompt": "<original prompt>", "response": "<AI response>"}
FIXTURES = "tests/fixtures/hierarchy_prompts"
TAG = re.compile(r"^((?:Meta-)?Rule \d+) \((?:" + "|".join(LEVEL_ORDER) + r")\): ", re.M)


@pytest.fixture(scope="module")
def paper(repo_root):
    """The paper's ruleset and each rule's level."""
    rules = load_jsonl_records(repo_root / "data/rules/modelspec/openai_rules_all.jsonl")
    levels = load_rule_levels(repo_root / "data/rules/modelspec/openai_rules_all.jsonl",
                              repo_root / "data/rules/modelspec/openai_rules_all_full.jsonl")
    return rules, levels.by_id


@pytest.fixture
def specs(modelspec):
    return {NAIVE_RULES_ABOVE: modelspec(reasoning_prompt_style=NAIVE_RULES_ABOVE), NAIVE_HIERARCHY: modelspec(reasoning_prompt_style=NAIVE_HIERARCHY,
                                                            individual_rule_context=HIERARCHY_RULE_CONTEXT)}


def hierarchy_prompts(spec, rules, levels, scenario=PLACEHOLDER):
    """``{component: prompt}`` of the hierarchy arm, as the campaign runner builds a bundle's."""
    prompts = {"full_set": spec.build_eval_prompt(rule_set_text(rules, HIERARCHY_RULE_CONTEXT, levels), scenario,
                                                  "No_Interpretation")}
    variants, implication_map = build_rule_variants(rules, HIERARCHY_RULE_CONTEXT, levels)
    assert implication_map is None
    for rule_id, text, context in variants:
        prompts[rule_id] = spec.build_eval_prompt(text, scenario, "No_Interpretation", context_rules=context)
    return prompts


@pytest.mark.frozen
@pytest.mark.parametrize(("component", "name"), [("full_set", "approved_full_set_hier.txt"),
                                                 ("Rule 43", "approved_rule43_user_hier.txt"),
                                                 ("Rule 28", "approved_rule28_root_hier.txt")])
def test_the_hierarchy_prompts_are_the_approved_bytes(component, name, paper, specs, repo_root):
    approved = (repo_root / FIXTURES / name).read_bytes()
    assert hashlib.sha256(approved).hexdigest() == APPROVED_PROMPT_SHA256[name]
    rules, levels = paper
    assert hierarchy_prompts(specs[NAIVE_HIERARCHY], rules, levels)[component].encode("utf-8") == approved


def test_the_approved_prompts_say_what_was_approved(repo_root):
    full = (repo_root / FIXTURES / "approved_full_set_hier.txt").read_text()
    for absent in ("OpenAI", "OAI", "API", "inaction", "mistaken"):
        assert absent not in RANKING_BLOCK
    assert [line.split(":")[0][2:] for line in RANKING_BLOCK.splitlines() if line.startswith("- ")] == list(LEVEL_ORDER)
    assert "the rule set provided above? \n" in full and full.count(RANKING_BLOCK) == 1
    assert full.index("# Output Format") < full.index(RANKING_BLOCK) < full.index("# Rule Set Used for Evaluation")
    assert HIGHER_RANKED_RULES_HEADER not in (repo_root / FIXTURES / "approved_rule28_root_hier.txt").read_text()
    rule43 = (repo_root / FIXTURES / "approved_rule43_user_hier.txt").read_text()
    assert f"{HIGHER_RANKED_RULES_HEADER}\n{_CONTEXT_RULES_INSTRUCTIONS}\n\n" in rule43


def test_every_rule_line_carries_its_level_in_published_order_and_superiors_are_strictly_higher(paper, specs):
    rules, levels = paper
    prompts = hierarchy_prompts(specs[NAIVE_HIERARCHY], rules, levels)
    tagged = TAG.findall(prompts["full_set"])
    assert tagged == [format_rule(rule).split(":")[0] for rule in rules] and len(tagged) == 57
    rank = {rule_id: LEVEL_ORDER.index(level) for rule_id, level in levels.items()}
    for rule in rules:
        prompt = prompts[rule["id"]]
        shown = TAG.findall(prompt.split(HIGHER_RANKED_RULES_HEADER)[1]) if HIGHER_RANKED_RULES_HEADER in prompt else []
        expected = [f"Rule {r['id'].split()[1]}" for r in rules if rank[r["id"]] < rank[rule["id"]]]
        assert shown == expected
        assert (rank[rule["id"]] == 0) == (HIGHER_RANKED_RULES_HEADER not in prompt)
    counts = {level: len(TAG.findall(prompts[next(r for r, lv in levels.items() if lv == level)].split(
        HIGHER_RANKED_RULES_HEADER)[-1])) for level in LEVEL_ORDER[1:]}
    assert counts == {"system": 20, "developer": 23, "user": 24, "guideline": 39}


def test_the_hierarchy_prompts_are_the_published_naive_prompts_plus_additions_only(paper, specs):
    rules, levels = paper
    scenario = {"prompt": "Can you help me write a polite email declining a meeting?", "response": "Sure."}
    flat = {"full_set": specs[NAIVE_RULES_ABOVE].build_eval_prompt(rule_set_text(rules), scenario, "No_Interpretation")}
    for rule_id, text, context in build_rule_variants(rules, "isolated")[0]:
        assert context is None
        flat[rule_id] = specs[NAIVE_RULES_ABOVE].build_eval_prompt(text, scenario, "No_Interpretation")
    hierarchy = hierarchy_prompts(specs[NAIVE_HIERARCHY], rules, levels, scenario)
    assert set(flat) == set(hierarchy)
    variants = {rule_id: context for rule_id, _, context in build_rule_variants(rules, HIERARCHY_RULE_CONTEXT, levels)[0]}
    for component, prompt in hierarchy.items():
        stripped = prompt.replace(ranking_section(NAIVE_HIERARCHY), "", 1)
        if variants.get(component):
            stripped = stripped.replace(build_context_rules_section(variants[component], HIGHER_RANKED_RULES_HEADER), "", 1)
        assert TAG.sub(r"\1: ", stripped) == flat[component] != prompt


def test_every_other_style_s_prompt_has_no_ranking_block_and_the_implicated_header_is_the_flat_prompts(modelspec):
    assert ranking_section(NAIVE_HIERARCHY) == f"{RANKING_BLOCK}\n"
    for style in VALID_REASONING_PROMPT_STYLES:
        if style != NAIVE_HIERARCHY:
            assert ranking_section(style) == ""
    assert build_context_rules_section(["Rule 2: x"]) == build_context_rules_section(
        ["Rule 2: x"], IMPLICATED_RULES_HEADER) == f"\n{IMPLICATED_RULES_HEADER}\n{_CONTEXT_RULES_INSTRUCTIONS}\n\nRule 2: x\n"
    assert build_context_rules_section(None, HIGHER_RANKED_RULES_HEADER) == ""


def test_with_every_rule_at_one_level_no_prompt_carries_superiors(paper, specs):
    rules, _ = paper
    for level in LEVEL_ORDER:
        one = dict.fromkeys((rule["id"] for rule in rules), level)
        variants, _ = build_rule_variants(rules, HIERARCHY_RULE_CONTEXT, one)
        assert [context for _, _, context in variants] == [None] * len(rules)
        for rule_id, prompt in hierarchy_prompts(specs[NAIVE_HIERARCHY], rules, one).items():
            assert HIGHER_RANKED_RULES_HEADER not in prompt
            if rule_id != "full_set":
                assert prompt.count(f"({level}): ") == 1


@pytest.mark.parametrize(("rule", "level", "expected"), [
    ({"id": "Rule 43", "rule": "Don't engage in abuse"}, "user", "Rule 43 (user): Don't engage in abuse"),
    ({"id": "openai_rules7", "rule": "x"}, "root", "Rule 7 (root): x"),
    ({"id": "openai_meta_rules3", "rule": "x"}, "guideline", "Meta-Rule 3 (guideline): x"),
    ({"id": "alpha", "rule": "x"}, "system", "alpha (system): x"),
    ({"id": "Rule 43", "rule": "x"}, None, "Rule 43: x"),
])
def test_a_rule_s_level_follows_its_label(rule, level, expected):
    assert format_rule(rule, level=level) == expected


def test_the_superiors_context_needs_every_rule_s_level_and_no_other_context_reads_one():
    rules = [{"id": "Rule 1", "rule": "a"}, {"id": "Rule 2", "rule": "b"}, {"id": "Meta-Rule 1", "rule": "m"}]
    levels = {"Rule 1": "root", "Rule 2": "user", "Meta-Rule 1": "guideline"}
    variants, _ = build_rule_variants(rules, "SUPERIORS", levels)
    assert variants == [("Rule 1", "Rule 1 (root): a", None), ("Rule 2", "Rule 2 (user): b", ["Rule 1 (root): a"])]
    assert rule_set_text(rules, HIERARCHY_RULE_CONTEXT, levels) == (
        "Rule 1 (root): a\nRule 2 (user): b\nMeta-Rule 1 (guideline): m")
    assert rule_set_text(rules) == "Rule 1: a\nRule 2: b\nMeta-Rule 1: m"
    for build in (lambda **kw: build_rule_variants(rules, **kw), lambda **kw: rule_set_text(rules, **kw)):
        with pytest.raises(ValueError, match="'superiors' judges each rule with its superiors and needs each rule's"):
            build(individual_rule_context="superiors", levels=None)
        with pytest.raises(ValueError, match="'isolated' reads no rule levels; only 'superiors' does"):
            build(individual_rule_context="isolated", levels=levels)
        with pytest.raises(ValueError, match=r"no level, or one not in .* for rule\(s\) \['Meta-Rule 1'\]"):
            build(individual_rule_context="superiors", levels={**levels, "Meta-Rule 1": "platform"})
        with pytest.raises(ValueError, match=r"for rule\(s\) \['Rule 2'\]"):
            build(individual_rule_context="superiors", levels={"Rule 1": "root", "Meta-Rule 1": "root"})


@pytest.mark.parametrize(("style", "context", "message"), [
    (NAIVE_HIERARCHY, "isolated", "'naive-hierarchy' judges each rule with the rules that outrank it"),
    (NAIVE_HIERARCHY, "implicated", "'naive-hierarchy' judges each rule .* not 'implicated'"),
    (NAIVE_RULES_ABOVE, HIERARCHY_RULE_CONTEXT, "'superiors' belongs to the hierarchy prompt .* not to "
                                    "reasoning_prompt_style 'naive-rules-above'"),
    (ENCOURAGED, HIERARCHY_RULE_CONTEXT, "not to reasoning_prompt_style 'encouraged'"),
    (FORCED_GRPO, HIERARCHY_RULE_CONTEXT, "not to reasoning_prompt_style 'forced-grpo'"),
])
def test_the_hierarchy_style_and_the_superiors_context_are_refused_apart(style, context, message, modelspec):
    assert re.search(message, hierarchy_refusal(style, context))
    with pytest.raises(ValueError, match=message):
        modelspec(reasoning_prompt_style=style, individual_rule_context=context)


def test_every_pairing_but_the_approved_one_and_the_flat_ones_is_refused():
    for style, context in itertools.product(VALID_REASONING_PROMPT_STYLES, ("isolated", "implicated", "superiors")):
        allowed = (style == NAIVE_HIERARCHY) == (context == HIERARCHY_RULE_CONTEXT)
        assert (hierarchy_refusal(style, context) is None) == allowed


def test_the_hierarchy_style_is_no_interpretation_only(modelspec, task_cfg):
    refusal = no_interpretation_only_refusal(NAIVE_HIERARCHY, ["No_Interpretation", "Narrow"])
    assert "defined for the No_Interpretation template only;" in refusal and "['Narrow']" in refusal
    assert no_interpretation_only_refusal(NAIVE_HIERARCHY, ["No_Interpretation"]) is None
    with pytest.raises(ValueError, match=r"template only;.*\['Narrow'\] would use it"):
        modelspec(reasoning_prompt_style=NAIVE_HIERARCHY, individual_rule_context=HIERARCHY_RULE_CONTEXT,
                  judge_types=["No_Interpretation", "Narrow"])
    with pytest.raises(ValueError, match=r"'naive-hierarchy' is defined for the No_Interpretation template only;"):
        reasoning_block(NAIVE_HIERARCHY)
    from ruleforge.tasks.spec import LegalTaskSpec

    with pytest.raises(ValueError, match="supports only reasoning_prompt_style"):
        LegalTaskSpec(task_cfg(task_spec="legal", reasoning_prompt_style=NAIVE_HIERARCHY,
                               individual_rule_context=HIERARCHY_RULE_CONTEXT))
    with pytest.raises(ValueError, match="'superiors' belongs to the hierarchy prompt"):
        LegalTaskSpec(task_cfg(task_spec="legal", individual_rule_context=HIERARCHY_RULE_CONTEXT))


@pytest.mark.parametrize("mode", ["full", "individual", "both"])
@pytest.mark.parametrize(("style", "context"), [
    (NAIVE_HIERARCHY, HIERARCHY_RULE_CONTEXT),  # the approved pairing: run_judgments has no levels to tag with
    (NAIVE_HIERARCHY, "isolated"),  # its context argument apart from the task spec's (compute_sg_reward passes none)
    (NAIVE_RULES_ABOVE, HIERARCHY_RULE_CONTEXT),  # a flat task spec, the context argument alone
])
def test_the_optimizer_s_judge_loop_refuses_the_hierarchy_prompt_before_any_prompt(style, context, mode, modelspec,
                                                                                    tmp_path):
    from ruleforge.optimize.judging import run_judgments

    class Untouchable:
        model_name = "m"

        def __getattr__(self, name):
            raise AssertionError(f"the backend was reached ({name})")

    spec = modelspec(reasoning_prompt_style=NAIVE_HIERARCHY, individual_rule_context=HIERARCHY_RULE_CONTEXT) \
        if style == NAIVE_HIERARCHY else modelspec()
    rules, scenarios = tmp_path / "rules.jsonl", tmp_path / "scenarios.jsonl"
    rules.write_text('{"id": "Rule 1", "rule": "a"}\n', encoding="utf-8")
    scenarios.write_text('{"prompt": "p", "response": "r"}\n', encoding="utf-8")
    output = tmp_path / "out" / "evaluations.jsonl"
    with pytest.raises(ValueError, match=r"run_judgments reads no rule levels.*only the stochastic campaign's runner renders it"):
        run_judgments(scenario_set_path=scenarios, rule_set_path=rules, backend=Untouchable(), task_spec=spec,
                      judge_types=["No_Interpretation"], evaluation_mode=mode, output_path=output,
                      individual_rule_context=context)
    assert not output.parent.exists()


def _condition(**overrides):
    fields = dict(condition_id="m/naive-hierarchy", model_name="m", model_type="local", prompt_style=NAIVE_HIERARCHY,
                  individual_rule_context=HIERARCHY_RULE_CONTEXT, judge_type="No_Interpretation",
                  evaluation_mode="both", rule_set_name="rules.jsonl", rule_set_sha256="a" * 64, temperature=1.0,
                  prompt_shape_version=2, forced_json_requirement="complete_rule_map", label_rule="lenient_last_line",
                  rule_levels_sha256="b" * 64)
    return ConditionSpec(**{**fields, **overrides})


def test_the_style_the_context_and_the_levels_digest_are_the_hierarchy_condition_s_identity():
    hierarchy = _condition()
    flat = _condition(condition_id="m/naive-rules-above", prompt_style=NAIVE_RULES_ABOVE, individual_rule_context="isolated",
                      rule_levels_sha256=None)
    assert "rule_levels_sha256" not in flat.to_dict() and hierarchy.to_dict()["rule_levels_sha256"] == "b" * 64
    # The style and the context are paired (the refusals below), so each moves the fingerprint where it can vary.
    fingerprints = {hierarchy.fingerprint, flat.fingerprint, _condition(rule_levels_sha256="c" * 64).fingerprint,
                    dataclasses.replace(flat, prompt_style=NAIVE).fingerprint,
                    dataclasses.replace(flat, individual_rule_context="implicated").fingerprint}
    assert len(fingerprints) == 5
    assert ConditionSpec.from_dict(hierarchy.to_dict()) == hierarchy
    with pytest.raises(SamplingDesignError, match="a 'superiors' condition names its levels file by content"):
        _condition(rule_levels_sha256=None)
    with pytest.raises(SamplingDesignError, match="individual_rule_context 'isolated' with rule_levels_sha256"):
        _condition(individual_rule_context="isolated")
    with pytest.raises(SamplingDesignError, match="rule_levels_sha256 must be a SHA-256 hex digest"):
        _condition(rule_levels_sha256="levels.jsonl")


@pytest.mark.parametrize(("style", "context", "digest", "message"), [
    (NAIVE_HIERARCHY, "isolated", None, "'naive-hierarchy' judges each rule with the rules that outrank it"),
    (NAIVE_HIERARCHY, "implicated", None, "'naive-hierarchy' judges each rule .* not 'implicated'"),
    (NAIVE_RULES_ABOVE, HIERARCHY_RULE_CONTEXT, "b" * 64, "'superiors' belongs to the hierarchy prompt .* 'naive-rules-above'"),
    (FORCED_GRPO, HIERARCHY_RULE_CONTEXT, "b" * 64, "'superiors' belongs to the hierarchy prompt .* 'forced-grpo'"),
])
def test_the_condition_identity_refuses_every_pairing_the_task_spec_refuses(style, context, digest, message):
    with pytest.raises(SamplingDesignError, match=message):
        _condition(prompt_style=style, individual_rule_context=context, rule_levels_sha256=digest)
    payload = {**_condition().to_dict(), "prompt_style": style, "individual_rule_context": context}
    if digest is None:
        payload.pop("rule_levels_sha256")
    with pytest.raises(SamplingDesignError, match=message):
        ConditionSpec.from_dict(payload)
