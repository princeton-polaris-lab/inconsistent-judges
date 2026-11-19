from __future__ import annotations

import inspect

import pytest

from ruleforge.optimize.metrics.entropy import (
    ENTROPY_METHODS,
    aggregate_entropy,
    calculate_consensus_entropy_yes_no_unsure,
    entropy_for_scenario,
    extract_rule_judgments,
    get_scenario_full_ruleset_entropy,
    iter_entropy_values,
)


def parse_fn(x):
    if not isinstance(x, str):
        return None
    x = x.strip().lower()
    return x if x in ("yes", "no", "unsure") else None


def make_scenario(model_to_variant_to_judgments):
    """Build an evaluation scenario record."""
    evaluations = {}
    for model, variants in model_to_variant_to_judgments.items():
        evaluations[model] = {}
        for variant, judgments in variants.items():
            evaluations[model][variant] = {
                f"jt_{i}": j for i, j in enumerate(judgments)
            }
    return {"scenario": {"prompt": "p", "response": "r"}, "evaluations": evaluations}


class TestConsensusEntropy:
    def test_all_yes_zero_entropy(self):
        assert calculate_consensus_entropy_yes_no_unsure(["yes"] * 5) == 0.0

    def test_all_no_zero_entropy(self):
        assert calculate_consensus_entropy_yes_no_unsure(["no"] * 5) == 0.0

    def test_balanced_yes_no_max_entropy(self):
        result = calculate_consensus_entropy_yes_no_unsure(["yes", "yes", "no", "no"])
        assert result == pytest.approx(1.0)

    def test_unsure_splits_50_50(self):
        # 2 unsure → 1 yes-half + 1 no-half → balanced → entropy 1
        result = calculate_consensus_entropy_yes_no_unsure(["unsure", "unsure"])
        assert result == pytest.approx(1.0)

    def test_empty_returns_zero(self):
        assert calculate_consensus_entropy_yes_no_unsure([]) == 0.0


class TestExtractRuleJudgments:
    def test_extracts_per_model_per_variant(self):
        scenario = make_scenario({"M": {"full_ruleset": ["yes", "no"], "R1": ["yes"]}})
        result = extract_rule_judgments(scenario["evaluations"], parse_fn)
        assert result == {"M": {"full_ruleset": ["yes", "no"], "R1": ["yes"]}}

    def test_skips_unparseable(self):
        scenario = make_scenario({"M": {"full_ruleset": ["yes", "garbage", "no"]}})
        result = extract_rule_judgments(scenario["evaluations"], parse_fn)
        assert result == {"M": {"full_ruleset": ["yes", "no"]}}


class TestIterEntropyValues:
    def test_flattens_nested(self):
        result = iter_entropy_values({"M1": {"full_ruleset": 0.5, "R1": 0.0}, "M2": {"full_ruleset": 1.0}})
        assert sorted(result) == [0.0, 0.5, 1.0]

    def test_empty(self):
        assert iter_entropy_values({}) == []


class TestGetScenarioFullRulesetEntropy:
    def test_averages_across_models(self):
        record = {"entropy": {"M1": {"full_ruleset": 0.4}, "M2": {"full_ruleset": 0.6}}}
        assert get_scenario_full_ruleset_entropy(record) == pytest.approx(0.5)

    def test_no_full_ruleset_returns_none_not_zero(self):
        record = {"entropy": {"M1": {"R1": 0.5}}}
        assert get_scenario_full_ruleset_entropy(record) is None


class TestEntropyForScenario:
    def test_returns_entropy_record(self):
        scenario = make_scenario({"M": {"full_ruleset": ["yes", "no"], "R1": ["yes", "yes"]}})
        record = entropy_for_scenario(scenario, "consensus", parse_fn)
        assert record is not None
        assert record["entropy"]["M"]["full_ruleset"] == pytest.approx(1.0)
        assert record["entropy"]["M"]["R1"] == 0.0
        assert record["judgment_count"]["M"]["full_ruleset"] == 2

    def test_returns_none_when_no_evaluations(self):
        scenario = {"scenario": {}, "evaluations": {}}
        record = entropy_for_scenario(scenario, "consensus", parse_fn)
        assert record is None

    def test_invalid_method_raises(self):
        scenario = make_scenario({"M": {"full_ruleset": ["yes"]}})
        with pytest.raises(ValueError, match="Invalid entropy_method"):
            entropy_for_scenario(scenario, "bogus", parse_fn)


class TestAggregateEntropy:
    def test_collects_records_for_all_valid_scenarios(self):
        scenarios = [
            make_scenario({"M": {"full_ruleset": ["yes", "no"]}}),
            make_scenario({"M": {"full_ruleset": ["yes", "yes"]}}),
        ]
        records = aggregate_entropy(scenarios, "consensus", parse_fn)
        assert len(records) == 2
        assert records[0]["entropy"]["M"]["full_ruleset"] == pytest.approx(1.0)
        assert records[1]["entropy"]["M"]["full_ruleset"] == 0.0

    def test_skips_scenarios_with_no_evaluations(self):
        scenarios = [
            make_scenario({"M": {"full_ruleset": ["yes", "no"]}}),
            {"scenario": {}, "evaluations": {}},
        ]
        records = aggregate_entropy(scenarios, "consensus", parse_fn)
        assert len(records) == 1

    def test_empty_input_returns_empty(self):
        assert aggregate_entropy([], "consensus", parse_fn) == []


class TestEntropyMethodsRegistry:
    EXPECTED_KEYS = {"consensus"}

    def test_registry_has_expected_keys(self):
        assert set(ENTROPY_METHODS.keys()) == self.EXPECTED_KEYS

    def test_registry_values_are_callable(self):
        for fn in ENTROPY_METHODS.values():
            assert callable(fn)

    def test_registered_functions_share_signature(self):
        for name, fn in ENTROPY_METHODS.items():
            params = list(inspect.signature(fn).parameters.keys())
            assert params == ["data_list"], f"{name!r} has unexpected params: {params}"


class TestEmptyJudgmentsAreNotConsensus:
    """A rule key nobody answered must not read as agreement."""

    def _scenario(self, judgments):
        return {
            "scenario": "s",
            "evaluations": {
                "judge": {
                    rule_key: {judge: value for judge, value in by_judge.items()}
                    for rule_key, by_judge in judgments.items()
                }
            },
        }

    def test_an_unanswered_rule_is_excluded_rather_than_scored_zero(self):
        from ruleforge.optimize.metrics.entropy import entropy_for_scenario
        from ruleforge.prompts.parse import read_label

        record = entropy_for_scenario(
            self._scenario(
                {
                    "full_ruleset": {"A": "yes", "B": "no"},
                    "Rule 1": {"A": None, "B": None},
                }
            ),
            "consensus",
            read_label,
        )
        assert "Rule 1" not in record["entropy"]["judge"]
        assert record["judgment_count"]["judge"]["Rule 1"] == 0
        assert "full_ruleset" in record["entropy"]["judge"]

    def test_a_partially_answered_rule_is_still_scored(self):
        from ruleforge.optimize.metrics.entropy import entropy_for_scenario
        from ruleforge.prompts.parse import read_label

        record = entropy_for_scenario(
            self._scenario({"Rule 1": {"A": "yes", "B": None}}),
            "consensus",
            read_label,
        )
        assert record["entropy"]["judge"]["Rule 1"] == 0.0
        assert record["judgment_count"]["judge"]["Rule 1"] == 1


class TestInputsOutsideTheShape:
    """What each kernel and reader does with input that holds no yes/no answer, or not the shape it expects."""

    def test_a_list_of_other_labels_alone_has_zero_entropy(self):
        assert ENTROPY_METHODS["consensus"](["maybe", "n/a"]) == 0.0

    def test_an_empty_list_has_zero_entropy(self):
        assert ENTROPY_METHODS["consensus"]([]) == 0.0

    def test_a_model_or_variant_that_is_not_a_mapping_is_skipped(self):
        evaluations = {"M1": "not a mapping", "M2": {"full_ruleset": "yes"}, "M3": {"full_ruleset": {"jt": "no"}}}
        assert extract_rule_judgments(evaluations, parse_fn) == {"M3": {"full_ruleset": ["no"]}}

    def test_a_variant_map_that_is_not_a_mapping_is_refused(self):
        with pytest.raises(ValueError, match="Variant map is not a dict"):
            iter_entropy_values({"M": 0.5})

    def test_an_entropy_that_is_not_a_number_is_refused(self):
        with pytest.raises(ValueError, match="Unable to coerce entropy value to float: high"):
            iter_entropy_values({"M": {"full_ruleset": "high"}})

    def test_a_full_ruleset_entropy_that_is_not_a_number_or_not_a_mapping_is_skipped(self):
        record = {"entropy": {"M1": 0.3, "M2": {"full_ruleset": "high"}, "M3": {"full_ruleset": 0.25}}}
        assert get_scenario_full_ruleset_entropy(record) == 0.25
        assert get_scenario_full_ruleset_entropy({"entropy": {"M1": 0.3, "M2": {"full_ruleset": "x"}}}) is None
