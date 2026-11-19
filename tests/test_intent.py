from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from ruleforge.tasks.history import _RUBRIC_MAP
from tests.support.traces import traced
from ruleforge.optimize.metrics.intent import (
    INTENT_METHODS,
    _build_intent_result_rubric,
    _detect_deleted_rules,
    IntentScoringError,
    _checked_scores,
    calculate_intent_penalty_rubric,
    run_intent_rubric,
)


class TestCalculateIntentPenaltyRubric:
    def test_no_scores_returns_zero(self):
        assert calculate_intent_penalty_rubric([], max_score=4) == 0.0

    def test_all_zero_scores_returns_zero(self):
        scores = [{"id": "R1", "score": 0}, {"id": "R2", "score": 0}]
        assert calculate_intent_penalty_rubric(scores, max_score=4) == 0.0

    def test_worst_score_drives_penalty(self):
        scores = [{"id": "R1", "score": 1}, {"id": "R2", "score": 3}, {"id": "R3", "score": 2}]
        assert calculate_intent_penalty_rubric(scores, max_score=4) == 0.75

    def test_max_worst_score_yields_one(self):
        scores = [{"id": "R1", "score": 4}]
        assert calculate_intent_penalty_rubric(scores, max_score=4) == 1.0

    def test_zero_max_score_returns_zero(self):
        scores = [{"id": "R1", "score": 3}]
        assert calculate_intent_penalty_rubric(scores, max_score=0) == 0.0


class TestCheckedScores:
    """A reply counts only if it scores every revised rule exactly once with an integer in range."""

    RULES = ["Rule 1", "Meta-Rule 1"]

    def test_a_complete_reply_is_read(self):
        parsed = {"rules": [
            {"id": "Rule 1", "score": 2, "reasoning": "ok"},
            {"id": "Meta-Rule 1", "score": 0},
        ]}
        assert _checked_scores(parsed, self.RULES, max_score=4) == [
            {"id": "Rule 1", "score": 2, "reasoning": "ok"},
            {"id": "Meta-Rule 1", "score": 0, "reasoning": ""},
        ]

    def test_an_id_is_matched_by_the_rule_it_names(self):
        parsed = {"rules": [{"id": " rule 1 ", "score": 4}, {"id": "Meta-Rule 01", "score": 0}]}
        assert [row["score"] for row in _checked_scores(parsed, self.RULES, max_score=4)] == [4, 0]

    @pytest.mark.parametrize("parsed", [
        {},
        {"rules": []},
        {"rules": "all fine"},
        ["Rule 1", "Meta-Rule 1"],
        {"rules": ["Rule 1", "Meta-Rule 1"]},
        {"rules": [{"id": "Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 77", "score": 0}, {"id": "Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": 0}, {"id": "Rule 1", "score": 0}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": 0}, {"id": "Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": "invalid"}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1"}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": 5}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": -1}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": 1.5}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": "Rule 1", "score": True}, {"id": "Meta-Rule 1", "score": 0}]},
        {"rules": [{"id": 1, "score": 0}, {"id": "Meta-Rule 1", "score": 0}]},
    ])
    def test_anything_else_is_refused(self, parsed):
        assert _checked_scores(parsed, self.RULES, max_score=4) is None


def _write_rules(path: Path, rule_ids: list[str]) -> None:
    with path.open("w") as f:
        for rid in rule_ids:
            f.write(json.dumps({"id": rid, "rule": "x"}) + "\n")


class TestDetectDeletedRules:
    def test_no_deletions(self, tmp_path):
        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["R1", "R2"])
        _write_rules(revised, ["R1", "R2"])
        assert _detect_deleted_rules(orig, revised) == []

    def test_one_deletion(self, tmp_path):
        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["R1", "R2", "R3"])
        _write_rules(revised, ["R1", "R3"])
        assert _detect_deleted_rules(orig, revised) == ["R2"]

    def test_added_rules_dont_count_as_deletions(self, tmp_path):
        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["R1"])
        _write_rules(revised, ["R1", "R2"])
        assert _detect_deleted_rules(orig, revised) == []

    def test_cross_scheme_ids_compare_by_number_not_raw_id(self, tmp_path):
        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["openai_rules1", "openai_rules2", "openai_rules3"])
        _write_rules(revised, ["Rule 1", "Rule 3"])
        assert _detect_deleted_rules(orig, revised) == ["openai_rules2"]

    def test_meta_rules_do_not_collide_with_plain_rules_across_schemes(self, tmp_path):
        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["openai_meta_rules1", "openai_rules1"])
        _write_rules(revised, ["Rule 1"])
        assert _detect_deleted_rules(orig, revised) == ["openai_meta_rules1"]

    def test_a_row_without_an_id_is_no_rule_to_follow(self, tmp_path):
        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["  ", "Rule 1"])
        _write_rules(revised, ["Rule 1"])
        assert _detect_deleted_rules(orig, revised) == []

    def test_a_deletion_scores_the_maximum_without_asking_the_judge(self, tmp_path):
        """A revision that drops a rule is penalized 1.0 before any rubric prompt is sent."""

        class Unasked:
            def generate_with_json_validation_and_traces(self, *args, **kwargs):
                raise AssertionError("the rubric was asked about a revision that deleted a rule")

        orig = tmp_path / "orig.jsonl"
        revised = tmp_path / "revised.jsonl"
        _write_rules(orig, ["Meta-Rule 1", "Rule 1", "Rule 2"])
        _write_rules(revised, ["Rule 1", "Rule 2"])
        result = run_intent_rubric(orig, revised, _RUBRIC_MAP["rubric_orig_standards"], Unasked(), tmp_path / "out")
        assert (result["penalty"], result["rule_deletion_detected"], result["deleted_rule_ids"]) == (
            1.0, True, ["Meta-Rule 1"])
        assert json.loads((tmp_path / "out" / "intent_penalty.json").read_text()) == result


class TestBuildIntentResultRubric:
    def test_includes_all_expected_keys(self):
        result = _build_intent_result_rubric(
            penalty=0.5,
            per_rule_scores=[{"id": "R1", "score": 2, "reasoning": "x"}],
            max_score=4,
            rule_deletion_detected=False,
            deleted_rule_ids=[],
        )
        assert set(result.keys()) == {
            "penalty", "per_rule_scores", "max_score",
            "rule_deletion_detected", "deleted_rule_ids",
        }
        assert result["penalty"] == 0.5
        assert result["max_score"] == 4


class TestIntentMethodsRegistry:
    EXPECTED_KEYS = {"rubric_orig_standards"}
    EXPECTED_PARAMS = {
        "initial_rule_set_path",
        "revised_rule_set_path",
        "backend",
        "output_dir",
        "temperature",
        "seed",
    }

    def test_registry_has_expected_keys(self):
        assert set(INTENT_METHODS.keys()) == self.EXPECTED_KEYS

    def test_registry_values_are_callable(self):
        for fn in INTENT_METHODS.values():
            assert callable(fn)

    def test_registered_functions_share_signature(self):
        for name, fn in INTENT_METHODS.items():
            params = set(inspect.signature(fn).parameters.keys())
            assert params == self.EXPECTED_PARAMS, (
                f"{name!r} has unexpected params: {params}"
            )


class TestIntentDeterminism:
    """The rubric feeds the RA reward that selects the winning ruleset."""

    def test_the_rubric_is_scored_greedily_by_default(self):
        from ruleforge.optimize.metrics.intent import (
            DEFAULT_INTENT_SEED,
            DEFAULT_INTENT_TEMPERATURE,
        )

        assert DEFAULT_INTENT_TEMPERATURE == 0.0
        assert DEFAULT_INTENT_SEED == 16

    def test_the_sampling_reaches_the_backend(self, tmp_path):
        from ruleforge.optimize.metrics.intent import run_intent_rubric

        seen = {}

        class _Backend:
            num_tries = 5

            def generate_with_json_validation_and_traces(self, prompts, **kwargs):
                seen.update(kwargs)
                return traced([{"rules": [{"id": "Rule 1", "score": 0, "reasoning": "fine"}]}])

        initial = tmp_path / "initial.jsonl"
        revised = tmp_path / "revised.jsonl"
        for path in (initial, revised):
            path.write_text('{"id": "Rule 1", "rule": "be helpful"}\n')

        run_intent_rubric(
            initial, revised, {"0": "no drift", "1": "drift"}, _Backend(), tmp_path,
            temperature=0.0, seed=16,
        )
        assert {key: seen[key] for key in ("temperature", "seed")} == {"temperature": 0.0, "seed": 16}


class TestUnansweredScenariosAreExcludedFromReward:
    """A scenario nobody answered must not count as perfect consensus."""

    def test_ra_entropy_reward_skips_records_with_no_full_ruleset_entropy(self):
        from ruleforge.optimize.metrics.entropy import get_scenario_full_ruleset_entropy

        answered = {"scenario": "a", "entropy": {"m": {"full_ruleset": 0.8}}}
        unanswered = {"scenario": "b", "entropy": {"m": {"Rule 1": 0.5}}}
        assert get_scenario_full_ruleset_entropy(answered) == 0.8
        assert get_scenario_full_ruleset_entropy(unanswered) is None


class TestEveryIntentMethodScoresWithItsOwnRubric:
    """``compute_intent_penalty`` dispatches by name to a method scoring with its own rubric."""

    @staticmethod
    def _rulesets(tmp_path):
        initial, revised = tmp_path / "initial.jsonl", tmp_path / "revised.jsonl"
        initial.write_text('{"id": "Rule 1", "rule": "be helpful"}\n', encoding="utf-8")
        revised.write_text('{"id": "Rule 1", "rule": "be very helpful"}\n', encoding="utf-8")
        return initial, revised

    def test_each_method_asks_the_judge_with_its_rubric(self, tmp_path):
        from ruleforge.optimize.metrics.intent import compute_intent_penalty

        initial, revised = self._rulesets(tmp_path)
        for name, rubric in _RUBRIC_MAP.items():
            asked = []

            class _Backend:
                num_tries = 5

                def generate_with_json_validation_and_traces(self, prompts, **kwargs):
                    asked.append(prompts[0]["system_prompt"])
                    return traced([{"rules": [{"id": "Rule 1", "score": 2, "reasoning": "drift"}]}])

            result = compute_intent_penalty(initial, revised, _Backend(), name, tmp_path / name)
            assert all(text in asked[0] for text in rubric.values()), name
            assert (result["penalty"], result["max_score"]) == (0.5, 4)
            assert json.loads((tmp_path / name / "intent_penalty.json").read_text())["penalty"] == 0.5

    def test_an_unknown_method_and_an_unread_answer_are_refused(self, tmp_path):
        from ruleforge.optimize.metrics.intent import compute_intent_penalty

        initial, revised = self._rulesets(tmp_path)

        class _Backend:
            num_tries = 5

            def generate_with_json_validation_and_traces(self, prompts, **kwargs):
                return traced([None])

        with pytest.raises(ValueError, match="Unknown intent_method: 'nope'"):
            compute_intent_penalty(initial, revised, _Backend(), "nope", tmp_path)
        with pytest.raises(IntentScoringError, match="did not score every revised rule"):
            compute_intent_penalty(initial, revised, _Backend(), "rubric_orig_standards", tmp_path)

    def test_a_result_with_no_output_directory_is_not_saved(self, tmp_path):
        initial, revised = self._rulesets(tmp_path)

        class _Backend:
            num_tries = 5

            def generate_with_json_validation_and_traces(self, prompts, **kwargs):
                return traced([{"rules": [{"id": "Rule 1", "score": 0, "reasoning": "same"}]}])

        result = run_intent_rubric(initial, revised, _RUBRIC_MAP["rubric_orig_standards"], _Backend(), None)
        assert result["penalty"] == 0.0 and sorted(tmp_path.iterdir()) == sorted([initial, revised])
