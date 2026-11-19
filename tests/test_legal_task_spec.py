from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from ruleforge.tasks.spec import LegalTaskSpec
from ruleforge.tasks.history import format_scenario_xml
from ruleforge.optimize.input_output import parse_scenarios_to_jsonl


def _make_spec() -> LegalTaskSpec:
    cfg = OmegaConf.create({
        "task_spec": "legal",
        "rule_amendment": {
            "initial_rule_set_path": "tests/fixtures/legal/legal_gms_rules.jsonl",
        },
        "scenario_generation": {
            "initial_scenario_seed_type": "single_seed",
            "seed_scenario_path": "tests/fixtures/legal/legal_gms_seeds.jsonl",
            "num_seed_scenarios": 5,
            "num_scenarios_per_batch": 3,
        },
        "agent_model": {},
        "judge_model": {},
        "eval_cfg": {
            "judge_types": ["No_Interpretation"],
            "evaluation_temperature": 0.1,
            "save_batch_size": 10,
        },
    })
    return LegalTaskSpec(cfg=cfg)


SAMPLE_TWO_SCENARIOS = """# Scenario 1
Subset of Rule IDs: [R1, R2]
Possible Ambiguous Topics and Value Tradeoffs: [retroactivity, finality of orders]
Facts of the Case: ACME is a mine operator with 5 prior assessed violations finalized within the 15-month lookback window. Two additional citations from before the window were finalized inside it.
Legal Question: Should the court count the two earlier citations toward ACME's history of previous violations?

# Scenario 2
Subset of Rule IDs: [R1]
Possible Ambiguous Topics and Value Tradeoffs: [definition of "repeat violation"]
Facts of the Case: BetaMine received 4 distinct citations under the same standard within the 15-month period.
Legal Question: Are the four citations "repeat violations" for purposes of Rule 1?
"""


class TestParseScenariosToJsonl:
    def test_parses_two_scenarios(self, tmp_path):
        spec = _make_spec()
        out_path = tmp_path / "scenarios.jsonl"
        parse_scenarios_to_jsonl(spec.scenario_block_pattern,SAMPLE_TWO_SCENARIOS, out_path)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["subset_rule_ids"] == "[R1, R2]"
        assert first["ambiguous_topics_and_tradeoffs"] == "[retroactivity, finality of orders]"
        assert first["facts_of_case"].startswith("ACME is a mine operator")
        assert first["legal_question"].startswith("Should the court count")

        second = json.loads(lines[1])
        assert second["subset_rule_ids"] == "[R1]"
        assert second["facts_of_case"].startswith("BetaMine")
        assert second["legal_question"].startswith("Are the four citations")

    def test_empty_input_writes_empty_file(self, tmp_path):
        spec = _make_spec()
        out_path = tmp_path / "scenarios.jsonl"
        parse_scenarios_to_jsonl(spec.scenario_block_pattern,"", out_path)
        assert out_path.read_text() == ""

    def test_extra_whitespace_tolerated(self, tmp_path):
        spec = _make_spec()
        text = """

   # Scenario 1

   Subset of Rule IDs:   [R1]

   Possible Ambiguous Topics and Value Tradeoffs:   [topic]

   Facts of the Case:   Some facts here.

   Legal Question:   A yes/no question?

"""
        out_path = tmp_path / "scenarios.jsonl"
        parse_scenarios_to_jsonl(spec.scenario_block_pattern,text, out_path)
        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["subset_rule_ids"] == "[R1]"
        assert entry["facts_of_case"] == "Some facts here."
        assert entry["legal_question"] == "A yes/no question?"

    def test_multiline_facts_and_question(self, tmp_path):
        spec = _make_spec()
        text = """# Scenario 1
Subset of Rule IDs: [R1]
Possible Ambiguous Topics and Value Tradeoffs: [topic]
Facts of the Case: Line one of facts.
Line two of facts continues here.
Legal Question: A multi-line legal question
that wraps to a second line?
"""
        out_path = tmp_path / "scenarios.jsonl"
        parse_scenarios_to_jsonl(spec.scenario_block_pattern,text, out_path)
        entry = json.loads(out_path.read_text().strip())
        assert "Line one of facts." in entry["facts_of_case"]
        assert "Line two of facts continues here." in entry["facts_of_case"]
        assert "wraps to a second line" in entry["legal_question"]


class TestFormatScenarioForHistory:
    def test_uses_legal_field_names(self):
        spec = _make_spec()
        result = format_scenario_xml(
            {"facts_of_case": "Some facts.", "legal_question": "A question?"},
            spec.valid_scenario_keys,
        )
        assert "<FACTS_OF_CASE>" in result
        assert "<LEGAL_QUESTION>" in result
        assert "Some facts." in result
        assert "A question?" in result


class TestBuildEvalPrompt:
    def test_no_interpretation_uses_legal_framing(self):
        spec = _make_spec()
        prompt = spec.build_eval_prompt(
            rule_text="Rule 1: ...",
            scenario={"facts_of_case": "Facts.", "legal_question": "A question?"},
            judge_type="No_Interpretation",
        )
        assert "court should resolve a legal question" in prompt
        assert "Facts." in prompt
        assert "A question?" in prompt
        assert "Rule 1: ..." in prompt

    def test_invalid_judge_type_raises(self):
        spec = _make_spec()
        with pytest.raises(ValueError, match="Invalid judge type"):
            spec.build_eval_prompt(
                rule_text="r",
                scenario={"facts_of_case": "f", "legal_question": "q"},
                judge_type="bogus",
            )
