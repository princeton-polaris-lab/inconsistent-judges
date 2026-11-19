from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

import ruleforge.optimize.metrics.reward as reward_module
from ruleforge.optimize.judging import run_judgments
from ruleforge.core.artifacts import ARTIFACT_SCHEMA_VERSION, compute_prompt_response_hash
from ruleforge.optimize.metrics.reward import compute_ra_reward_standard
from ruleforge.prompts.context import _CONTEXT_RULES_INSTRUCTIONS, IMPLICATED_RULES_HEADER
from ruleforge.tasks.spec import LegalTaskSpec
from ruleforge.tasks.spec import ModelSpecTaskSpec
from ruleforge.tasks import RewardWeights


def _make_cfg(task_spec: str, eval_extra: dict | None = None) -> OmegaConf:
    return OmegaConf.create(
        {
            "task_spec": task_spec,
            "rule_amendment": {
                "initial_rule_set_path": "tests/fixtures/rules/openai_subset_prolific.jsonl",
                "intent_weight": 0.0,
            },
            "scenario_generation": {
                "initial_scenario_seed_type": "none",
                "good_seed_scenario_path": "unused.jsonl",
                "bad_seed_scenario_path": "unused.jsonl",
                "seed_scenario_path": "unused.jsonl",
                "num_scenarios_per_batch": 3,
            },
            "agent_model": {},
            "judge_model": {},
            "eval_cfg": {
                "judge_types": ["No_Interpretation"],
                "evaluation_temperature": 0.1,
                "save_batch_size": 10,
                **(eval_extra or {}),
            },
        }
    )


def _make_modelspec(eval_extra: dict | None = None) -> ModelSpecTaskSpec:
    return ModelSpecTaskSpec(cfg=_make_cfg("modelspec", eval_extra))


def _make_legal(eval_extra: dict | None = None) -> LegalTaskSpec:
    return LegalTaskSpec(cfg=_make_cfg("legal", eval_extra))


MODELSPEC_SCENARIO = {"prompt": "Help me", "response": "Sure!"}
LEGAL_SCENARIO = {"facts_of_case": "ACME did X.", "legal_question": "Is X allowed?"}
RULE_TEXT = "Rule 1: Be helpful, subject to Rule 2"
CONTEXT_RULES = ["Rule 2: Be honest"]

# (spec factory, scenario, section header following the rule set)
SPEC_CASES = [
    pytest.param(
        _make_modelspec,
        MODELSPEC_SCENARIO,
        "# Original Prompt Given to the AI",
        id="modelspec",
    ),
    pytest.param(_make_legal, LEGAL_SCENARIO, "# Facts of the Case", id="legal"),
]


def test_individual_rule_context_defaults_to_isolated():
    assert _make_modelspec().individual_rule_context == "isolated"
    assert _make_legal().individual_rule_context == "isolated"


def test_individual_rule_context_read_from_eval_cfg():
    spec = _make_modelspec({"individual_rule_context": "implicated"})
    assert spec.individual_rule_context == "implicated"


def test_invalid_individual_rule_context_rejected_at_construction():
    with pytest.raises(ValueError, match="individual_rule_context"):
        _make_modelspec({"individual_rule_context": "bogus"})


def test_context_instructions_wording_is_pinned():
    assert _CONTEXT_RULES_INSTRUCTIONS == (
        "The rules listed below are provided as context for interpreting the "
        "rule set under evaluation. Use them however the rule set under "
        "evaluation makes them relevant. Your final judgment must state only "
        "whether the rule set under evaluation is violated."
    )


@pytest.mark.parametrize("make_spec,scenario,next_section", SPEC_CASES)
@pytest.mark.parametrize("judge_type", ["No_Interpretation", "Broad"])
def test_context_none_and_empty_are_byte_identical(
    make_spec, scenario, next_section, judge_type
):
    spec = make_spec()
    base = spec.build_eval_prompt(RULE_TEXT, scenario, judge_type)
    with_none = spec.build_eval_prompt(
        RULE_TEXT, scenario, judge_type, context_rules=None
    )
    with_empty = spec.build_eval_prompt(
        RULE_TEXT, scenario, judge_type, context_rules=[]
    )
    assert base == with_none == with_empty
    assert IMPLICATED_RULES_HEADER not in base


@pytest.mark.parametrize("make_spec,scenario,next_section", SPEC_CASES)
def test_context_block_present_and_placed(make_spec, scenario, next_section):
    spec = make_spec()
    out = spec.build_eval_prompt(
        RULE_TEXT, scenario, "No_Interpretation", context_rules=CONTEXT_RULES
    )
    assert IMPLICATED_RULES_HEADER in out
    assert "Rule 2: Be honest" in out
    assert (
        out.index(RULE_TEXT)
        < out.index(IMPLICATED_RULES_HEADER)
        < out.index(next_section)
    )


@pytest.mark.parametrize("make_spec,scenario,next_section", SPEC_CASES)
def test_context_with_interpretation_strategy(make_spec, scenario, next_section):
    spec = make_spec()
    out = spec.build_eval_prompt(
        RULE_TEXT, scenario, "Broad", context_rules=CONTEXT_RULES
    )
    assert IMPLICATED_RULES_HEADER in out
    assert out.index(IMPLICATED_RULES_HEADER) < out.index("# Interpretation Strategy")


@pytest.mark.parametrize("make_spec,scenario,next_section", SPEC_CASES)
def test_multiple_context_rules_in_order(make_spec, scenario, next_section):
    spec = make_spec()
    out = spec.build_eval_prompt(
        RULE_TEXT,
        scenario,
        "No_Interpretation",
        context_rules=["Rule 2: A", "Rule 3: B"],
    )
    assert out.index("Rule 2: A") < out.index("Rule 3: B")


@pytest.mark.parametrize("make_spec,scenario,next_section", SPEC_CASES)
def test_invalid_judge_type_still_raises_with_context(
    make_spec, scenario, next_section
):
    spec = make_spec()
    with pytest.raises(ValueError):
        spec.build_eval_prompt(
            RULE_TEXT, scenario, "Bogus", context_rules=CONTEXT_RULES
        )


RULES = [
    {"id": "Rule 1", "rule": "Be helpful, subject to Rule 2 and Meta-Rule 1"},
    {"id": "Rule 2", "rule": "Be honest"},
    {"id": "Meta-Rule 1", "rule": "Ties resolve against violation"},
]
NO_META_RULES = [
    {"id": "Rule 1", "rule": "Be helpful, subject to Rule 2"},
    {"id": "Rule 2", "rule": "Be honest"},
]
META_PRIORITY_RULES = [
    {"id": "Rule 1", "rule": "Be helpful"},
    {"id": "Rule 2", "rule": "Be honest"},
    {"id": "Meta-Rule 1", "rule": "Rule 1 takes precedence over Rule 2"},
]


class FakeBackend:
    """Minimal ModelBackend stand-in capturing every queued prompt."""

    def __init__(self):
        self.model_name = "test-model"
        self.seed = 16
        self.prompts: list[str] = []
        self.call_count = 0

    def generate_with_validation(self, prompt_entries, **kwargs):
        self.call_count += 1
        self.prompts.extend(entry["system_prompt"] for entry in prompt_entries)
        return ["no"] * len(prompt_entries)

    def generate_with_validation_and_traces(self, prompt_entries, **kwargs):
        self.call_count += 1
        self.prompts.extend(entry["system_prompt"] for entry in prompt_entries)
        judgments = ["no"] * len(prompt_entries)
        traces = [
            {
                "judgment": "no",
                "selected_attempt": 1,
                "attempts": [
                    {
                        "attempt": 1,
                        "seed": kwargs["seed"],
                        "candidates": [
                            {
                                "raw_response": "<thought>reasoning</thought>no",
                                "completion": "no",
                                "finish_reason": "stop",
                                "accepted": True,
                            }
                        ],
                    }
                ],
            }
            for _ in prompt_entries
        ]
        return judgments, traces


def _run(tmp_path, name, rules=RULES, **kwargs):
    rule_path = tmp_path / f"{name}_rules.jsonl"
    rule_path.write_text(
        "\n".join(json.dumps(r) for r in rules) + "\n", encoding="utf-8"
    )
    scenario_path = tmp_path / f"{name}_scenarios.jsonl"
    scenario_path.write_text(json.dumps(MODELSPEC_SCENARIO) + "\n", encoding="utf-8")

    backend = FakeBackend()
    output_path = tmp_path / f"{name}_evaluations.jsonl"
    kwargs.setdefault("evaluation_mode", "both")
    run_judgments(
        scenario_set_path=scenario_path,
        rule_set_path=rule_path,
        backend=backend,
        task_spec=_make_modelspec(),
        judge_types=["No_Interpretation"],
        output_path=output_path,
        **kwargs,
    )
    # One scenario, one judge type: prompts are [full_ruleset, Rule 1, Rule 2].
    return backend, output_path


def _prompts_by_variant(backend, rules=RULES):
    non_meta = [r["id"] for r in rules if "meta" not in r["id"].lower()]
    labels = ["full_ruleset"] + non_meta
    assert len(backend.prompts) == len(labels)
    return dict(zip(labels, backend.prompts))


def test_default_isolated_matches_explicit_isolated(tmp_path):
    default_backend, _ = _run(tmp_path, "default")
    isolated_backend, _ = _run(tmp_path, "isolated", individual_rule_context="isolated")
    assert default_backend.prompts == isolated_backend.prompts
    assert all(IMPLICATED_RULES_HEADER not in p for p in default_backend.prompts)


def test_implicated_adds_context_and_sidecar(tmp_path):
    backend, output_path = _run(
        tmp_path, "implicated", individual_rule_context="implicated"
    )
    prompts = _prompts_by_variant(backend)

    rule_1_prompt = prompts["Rule 1"]
    assert IMPLICATED_RULES_HEADER in rule_1_prompt
    assert "Rule 2: Be honest" in rule_1_prompt
    assert "Meta-Rule 1: Ties resolve against violation" in rule_1_prompt

    record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    expected_hash = compute_prompt_response_hash("Help me", "Sure!")
    assert record["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert record["scenario_id"] == expected_hash
    assert record["prompt_response_hash"] == expected_hash
    assert set(record["evaluations"]["test-model"].keys()) == {
        "full_ruleset",
        "Rule 1",
        "Rule 2",
    }
    assert record["evaluations"]["test-model"]["Rule 1"]["No_Interpretation"] == "no"

    sidecar = json.loads(
        (output_path.parent / "implicated_rules.json").read_text(encoding="utf-8")
    )
    assert sidecar["individual_rule_context"] == "implicated"
    assert sidecar["text_fields"] == ["rule"]
    assert sidecar["implicated_rules"] == {
        "Rule 1": ["Rule 2", "Meta-Rule 1"],
        "Rule 2": [],
        "Meta-Rule 1": [],
    }


def test_raw_output_artifact_keeps_prompt_reasoning_and_parsed_judgment(tmp_path):
    raw_path = tmp_path / "judge_outputs.jsonl"
    backend, evaluation_path = _run(
        tmp_path,
        "traced",
        individual_rule_context="implicated",
        raw_output_path=raw_path,
    )
    prompts = _prompts_by_variant(backend)

    evaluation = json.loads(evaluation_path.read_text().splitlines()[0])
    raw = json.loads(raw_path.read_text().splitlines()[0])
    assert raw["scenario"] == evaluation["scenario"]
    assert "evaluations" not in raw

    call = raw["judge_outputs"]["test-model"]["Rule 1"]["No_Interpretation"]
    assert call["prompt"] == {
        "system_prompt": prompts["Rule 1"],
        "user_prompt": "",
    }
    assert call["judgment"] == "no"
    assert call["selected_attempt"] == 1
    candidate = call["attempts"][0]["candidates"][0]
    assert candidate == {
        "raw_response": "<thought>reasoning</thought>no",
        "completion": "no",
        "finish_reason": "stop",
        "accepted": True,
    }
    assert evaluation["evaluations"]["test-model"]["Rule 1"] == {
        "No_Interpretation": "no"
    }


def test_meta_rules_shown_for_all_rules_in_implicated_mode(tmp_path):
    backend, _ = _run(tmp_path, "meta_all", individual_rule_context="implicated")
    prompts = _prompts_by_variant(backend)
    # Rule 2 references nothing, but meta-rules govern all rules.
    rule_2_prompt = prompts["Rule 2"]
    assert IMPLICATED_RULES_HEADER in rule_2_prompt
    assert "Meta-Rule 1: Ties resolve against violation" in rule_2_prompt
    assert "Rule 1: Be helpful" not in rule_2_prompt


def test_meta_rule_references_reach_prompt(tmp_path):
    backend, _ = _run(
        tmp_path,
        "meta_priority",
        rules=META_PRIORITY_RULES,
        individual_rule_context="implicated",
    )
    prompts = _prompts_by_variant(backend, META_PRIORITY_RULES)
    rule_1_prompt = prompts["Rule 1"]
    assert "Meta-Rule 1: Rule 1 takes precedence over Rule 2" in rule_1_prompt
    assert "Rule 2: Be honest" in rule_1_prompt
    assert rule_1_prompt.count("Rule 1: Be helpful") == 1


def test_full_set_prompt_and_no_meta_rules_unchanged(tmp_path):
    isolated_backend, _ = _run(tmp_path, "iso_nometa", rules=NO_META_RULES)
    implicated_backend, _ = _run(
        tmp_path,
        "imp_nometa",
        rules=NO_META_RULES,
        individual_rule_context="implicated",
    )
    isolated = _prompts_by_variant(isolated_backend, NO_META_RULES)
    implicated = _prompts_by_variant(implicated_backend, NO_META_RULES)
    assert implicated["full_ruleset"] == isolated["full_ruleset"]
    assert implicated["Rule 2"] == isolated["Rule 2"]
    assert IMPLICATED_RULES_HEADER in implicated["Rule 1"]


def test_full_only_mode_skips_context_and_sidecar(tmp_path):
    backend, output_path = _run(
        tmp_path,
        "full_only",
        evaluation_mode="full",
        individual_rule_context="implicated",
    )
    assert len(backend.prompts) == 1
    assert IMPLICATED_RULES_HEADER not in backend.prompts[0]
    assert not (output_path.parent / "implicated_rules.json").exists()


def test_invalid_individual_rule_context_raises_before_backend_call(tmp_path):
    rule_path = tmp_path / "rules.jsonl"
    rule_path.write_text(json.dumps(RULES[0]) + "\n", encoding="utf-8")
    scenario_path = tmp_path / "scenarios.jsonl"
    scenario_path.write_text(json.dumps(MODELSPEC_SCENARIO) + "\n", encoding="utf-8")
    backend = FakeBackend()
    with pytest.raises(ValueError, match="individual_rule_context"):
        run_judgments(
            scenario_set_path=scenario_path,
            rule_set_path=rule_path,
            backend=backend,
            task_spec=_make_modelspec(),
            judge_types=["No_Interpretation"],
            evaluation_mode="both",
            output_path=tmp_path / "evaluations.jsonl",
            individual_rule_context="bogus",
        )
    assert backend.call_count == 0


def test_ra_reward_threads_individual_rule_context(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run_judgments(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(reward_module, "run_judgments", fake_run_judgments)
    monkeypatch.setattr(
        reward_module, "compute_entropy_from_evals", lambda **kwargs: []
    )
    task_spec = _make_modelspec({"individual_rule_context": "implicated"})
    compute_ra_reward_standard(
        task_spec=task_spec,
        scenario_set_path=tmp_path / "scenarios.jsonl",
        rule_set_path=tmp_path / "rules.jsonl",
        output_dir=tmp_path,
        backend=FakeBackend(),
        weights=RewardWeights(intent=0.0),
    )
    assert captured["individual_rule_context"] == "implicated"
    assert captured["evaluation_mode"] == "full"
