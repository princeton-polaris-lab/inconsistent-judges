from __future__ import annotations

import re

import pytest

from ruleforge.rulesets.rules import (
    RuleRangeError,
    RULE_REFERENCE_TEXT_FIELDS,
    _normalize_rule_number,
    build_implicated_rules_context,
    build_rule_id_index,
    build_rule_implication_map,
    extract_rule_references,
    looks_like_rule_range,
    normalize_preempting_rule_id,
    read_rule_references,
)
from ruleforge.evaluate.scoring import calculate_inconsistency


class TestCalculateInconsistency:
    def test_full_yes_all_rules_no_is_max_contradiction(self):
        result = calculate_inconsistency("yes", ["no", "no"])
        assert result == {"contradiction": 1.0, "penalty": 1.0}

    def test_full_no_one_rule_yes_is_max_contradiction(self):
        result = calculate_inconsistency("no", ["yes", "no"])
        assert result == {"contradiction": 1.0, "penalty": 1.0}

    def test_full_yes_max_rule_yes_no_contradiction(self):
        result = calculate_inconsistency("yes", ["yes", "no"])
        assert result == {"contradiction": 0.0, "penalty": 0.0}

    def test_full_unsure_max_rule_yes_half_penalty(self):
        result = calculate_inconsistency("unsure", ["yes", "no"])
        assert result["contradiction"] == 1.0
        assert result["penalty"] == 0.5

    def test_full_no_max_rule_unsure_half_penalty(self):
        result = calculate_inconsistency("no", ["unsure", "no"])
        assert result["contradiction"] == 1.0
        assert result["penalty"] == 0.5

    def test_empty_rule_judgments_raise(self):
        with pytest.raises(ValueError, match="no per-rule judgments"):
            calculate_inconsistency("yes", [])


class TestExtractRuleReferences:
    def test_single_reference(self):
        assert extract_rule_references("You must follow Rule 5 here.") == ["Rule 5"]

    def test_lowercase_reference(self):
        assert extract_rule_references("see rule 5 for details") == ["Rule 5"]

    def test_normalizes_leading_zeros(self):
        assert extract_rule_references("see Rule 05") == ["Rule 5"]

    def test_plural_and_conjunction(self):
        assert extract_rule_references("Rules 3 and 4 apply.") == ["Rule 3", "Rule 4"]

    def test_comma_list_with_oxford_and(self):
        assert extract_rule_references("Rules 3, 4, and 5 apply.") == [
            "Rule 3",
            "Rule 4",
            "Rule 5",
        ]

    def test_ampersand_separator(self):
        assert extract_rule_references("Rules 3 & 4 apply.") == ["Rule 3", "Rule 4"]

    def test_meta_rule_reference(self):
        assert extract_rule_references("Meta-Rule 2 governs.") == ["Meta-Rule 2"]
        assert extract_rule_references("per meta-rule 2") == ["Meta-Rule 2"]

    def test_meta_rule_not_cross_contaminated(self):
        # "Meta-Rule 5" must not additionally (or instead) yield "Rule 5".
        assert extract_rule_references("Meta-Rule 5 applies.") == ["Meta-Rule 5"]

    def test_hyphenated_prefix_not_matched(self):
        assert extract_rule_references("The Grand-Rule 5 applies.") == []

    def test_mixed_kinds_order_preserved(self):
        text = "Rule 1 applies, then Meta-Rule 1 governs, then Rule 2."
        assert extract_rule_references(text) == ["Rule 1", "Meta-Rule 1", "Rule 2"]

    def test_dedup_first_occurrence(self):
        text = "Rule 5 and Meta-Rule 2; later rule 5 again, and Meta-Rule 2 too."
        assert extract_rule_references(text) == ["Rule 5", "Meta-Rule 2"]

    def test_bare_numbers_and_urls_ignored(self):
        text = "See 42 and https://example.com/rules/7 for details."
        assert extract_rule_references(text) == []

    def test_a_range_reads_as_its_first_rule(self):
        assert extract_rule_references("Rules 1-3 apply.") == ["Rule 1"]

    def test_non_string_inputs(self):
        assert extract_rule_references("") == []
        assert extract_rule_references(None) == []
        assert extract_rule_references(42) == []

    def test_decimal_qualified_number_rejected(self):
        assert extract_rule_references("see Rule 12.4 for details") == []
        # A sentence-ending period is not a decimal qualifier.
        assert extract_rule_references("This is covered by Rule 3.") == ["Rule 3"]

    def test_conjunction_false_positive_is_documented_behavior(self):
        assert extract_rule_references("Rule 3 and 10 other constraints") == [
            "Rule 3",
            "Rule 10",
        ]

    def test_adversarial_separator_runs_are_linear(self):
        import time

        for text in [
            "Rules 1" + " ," * 40 + " x",
            "Rules 1" + " and," * 40 + " y",
        ]:
            start = time.perf_counter()
            assert extract_rule_references(text) == ["Rule 1"]
            assert time.perf_counter() - start < 0.5

    def test_huge_digit_run_does_not_crash(self):
        big = "1" * 5000
        assert extract_rule_references(f"Rule {big} applies") == [f"Rule {big}"]


class TestBuildRuleImplicationMap:
    def test_exact_id_resolution(self):
        rules = [
            {"id": "Rule 1", "rule": "Resolves 'Yes' if X occurs, subject to Rule 2."},
            {"id": "Rule 2", "rule": "Incidents on https://www.faa.gov/newsroom qualify."},
        ]
        assert build_rule_implication_map(rules) == {
            "Rule 1": ["Rule 2"],
            "Rule 2": [],
        }

    def test_digit_bearing_id_resolution(self):
        rules = [
            {"id": "openai_rules1", "rule": "Be safe"},
            {"id": "openai_rules2", "rule": "Be honest, see Rule 3"},
            {"id": "openai_rules3", "rule": "Be kind"},
        ]
        assert build_rule_implication_map(rules) == {
            "openai_rules1": [],
            "openai_rules2": ["openai_rules3"],
            "openai_rules3": [],
        }

    def test_meta_rule_is_key_and_resolves_as_value(self):
        rules = [
            {"id": "Rule 1", "rule": "Be helpful, subject to Rule 2 and Meta-Rule 1"},
            {"id": "Rule 2", "rule": "Be honest"},
            {"id": "Meta-Rule 1", "rule": "Ties resolve against violation"},
        ]
        assert build_rule_implication_map(rules) == {
            "Rule 1": ["Rule 2", "Meta-Rule 1"],
            "Rule 2": [],
            "Meta-Rule 1": [],
        }

    def test_meta_rule_references_extracted(self):
        rules = [
            {"id": "Rule 1", "rule": "Be helpful"},
            {"id": "Rule 2", "rule": "Be honest"},
            {"id": "Meta-Rule 1", "rule": "Rule 1 takes precedence over Rule 2"},
        ]
        assert build_rule_implication_map(rules)["Meta-Rule 1"] == ["Rule 1", "Rule 2"]

    def test_meta_self_and_cross_meta_references(self):
        rules = [
            {"id": "Meta-Rule 1", "rule": "Meta-Rule 1 and Meta-Rule 2 govern; see Rule 3"},
            {"id": "Meta-Rule 2", "rule": "second meta"},
            {"id": "Rule 3", "rule": "three"},
        ]
        assert build_rule_implication_map(rules)["Meta-Rule 1"] == [
            "Meta-Rule 2",
            "Rule 3",
        ]

    def test_kinds_do_not_cross_resolve(self):
        rules = [
            {"id": "Meta-Rule 3", "rule": "Meta three"},
            {"id": "Rule 1", "rule": "see Rule 3 and Meta-Rule 1"},
        ]
        assert build_rule_implication_map(rules) == {"Rule 1": [], "Meta-Rule 3": []}

    def test_unresolved_reference_dropped(self):
        rules = [
            {"id": "Rule 1", "rule": "see Rule 99 and Meta-Rule 99"},
            {"id": "Rule 2", "rule": "no references"},
        ]
        assert build_rule_implication_map(rules) == {"Rule 1": [], "Rule 2": []}

    def test_self_reference_excluded(self):
        rules = [{"id": "Rule 2", "rule": "Rule 2 requires honesty at all times"}]
        assert build_rule_implication_map(rules) == {"Rule 2": []}

    def test_self_reference_by_number_not_resolved_to_duplicate_sibling(self):
        rules = [
            {"id": "Rule 05", "rule": "five-a"},
            {"id": "Rule 5", "rule": "Rule 5 binds here"},
        ]
        assert build_rule_implication_map(rules)["Rule 5"] == []

    def test_duplicate_number_first_wins(self, caplog):
        import logging

        rules = [
            {"id": "Rule 5", "rule": "first five"},
            {"id": "other_rules5", "rule": "second five"},
            {"id": "Rule 1", "rule": "see Rule 5"},
        ]
        with caplog.at_level(logging.WARNING, logger="ruleforge.rulesets.rules"):
            implication_map = build_rule_implication_map(rules)
        assert implication_map["Rule 1"] == ["Rule 5"]
        assert "Duplicate rule number 5" in caplog.text

    def test_dedup_and_determinism(self):
        rules = [
            {
                "id": "Rule 1",
                "rule": "See Rule 3, then Rule 2, then Rule 3 again.",
                # Cross-field duplicate: "Rule 3" also appears here.
                "explanation": "Rule 3 is discussed further here.",
            },
            {"id": "Rule 2", "rule": "two"},
            {"id": "Rule 3", "rule": "three"},
        ]
        first = build_rule_implication_map(rules, text_fields=("rule", "explanation"))
        second = build_rule_implication_map(rules, text_fields=("rule", "explanation"))
        assert first["Rule 1"] == ["Rule 3", "Rule 2"]
        assert first == second

    def test_scans_configured_text_fields(self):
        rules = [
            {"id": "Rule 1", "rule": "see Rule 2", "explanation": "also see Rule 3"},
            {"id": "Rule 2", "rule": "two"},
            {"id": "Rule 3", "rule": "three"},
        ]
        assert RULE_REFERENCE_TEXT_FIELDS == ("rule",)
        assert build_rule_implication_map(rules)["Rule 1"] == ["Rule 2"]
        widened = build_rule_implication_map(rules, text_fields=("rule", "explanation"))
        assert widened["Rule 1"] == ["Rule 2", "Rule 3"]

    def test_tolerates_missing_or_non_string_fields(self):
        rules = [
            {"id": "Rule 1"},
            {"id": "Rule 2", "rule": None},
            {"id": "Rule 3", "rule": "see Rule 1"},
        ]
        assert build_rule_implication_map(rules) == {
            "Rule 1": [],
            "Rule 2": [],
            "Rule 3": ["Rule 1"],
        }

    def test_huge_digit_id_does_not_crash(self):
        big_id = f"Rule {'9' * 5000}"
        rules = [
            {"id": big_id, "rule": "big"},
            {"id": "Rule 1", "rule": f"see Rule {'9' * 5000}"},
        ]
        assert build_rule_implication_map(rules)["Rule 1"] == [big_id]

    def test_every_rule_id_is_a_key(self):
        rules = [
            {"id": "Rule 1", "rule": "one"},
            {"id": "General", "rule": "no digits in this id"},
            {"id": "Meta-Rule 1", "rule": "meta"},
            {"id": "", "rule": "empty id is skipped"},
        ]
        implication_map = build_rule_implication_map(rules)
        assert set(implication_map.keys()) == {"Rule 1", "General", "Meta-Rule 1"}
        assert all(v == [] for v in implication_map.values())


class TestBuildImplicatedRulesContext:
    RULES = [
        {"id": "Rule 1", "rule": "Be helpful, subject to Rule 2 and Meta-Rule 1"},
        {"id": "Rule 2", "rule": "Be honest"},
        {"id": "Meta-Rule 1", "rule": "Ties resolve against violation"},
    ]
    RULES_BY_ID = {r["id"]: r for r in RULES}
    MAP = {"Rule 1": ["Rule 2", "Meta-Rule 1"], "Rule 2": [], "Meta-Rule 1": []}

    def test_formats_referenced_rules(self):
        context = build_implicated_rules_context("Rule 1", self.MAP, self.RULES_BY_ID)
        assert context == [
            "Rule 2: Be honest",
            "Meta-Rule 1: Ties resolve against violation",
        ]

    def test_meta_included_without_reference(self):
        context = build_implicated_rules_context("Rule 2", self.MAP, self.RULES_BY_ID)
        assert context == ["Meta-Rule 1: Ties resolve against violation"]

    def test_meta_references_pulled_in(self):
        rules = [
            {"id": "Rule 1", "rule": "Be helpful"},
            {"id": "Rule 2", "rule": "Be honest"},
            {"id": "Meta-Rule 1", "rule": "Rule 1 takes precedence over Rule 2"},
        ]
        rules_by_id = {r["id"]: r for r in rules}
        implication_map = build_rule_implication_map(rules)
        context = build_implicated_rules_context("Rule 2", implication_map, rules_by_id)
        assert context == [
            "Meta-Rule 1: Rule 1 takes precedence over Rule 2",
            "Rule 1: Be helpful",
        ]

    def test_none_when_empty(self):
        rules_by_id = {
            "Rule 1": {"id": "Rule 1", "rule": "one"},
            "Rule 2": {"id": "Rule 2", "rule": "two"},
        }
        no_refs = {"Rule 1": [], "Rule 2": []}
        assert build_implicated_rules_context("Rule 2", no_refs, rules_by_id) is None

    def test_skips_unknown_ids(self):
        rules_by_id = {"Rule 1": {"id": "Rule 1", "rule": "one"}}
        stale_map = {"Rule 1": ["Rule 99"]}
        assert build_implicated_rules_context("Rule 1", stale_map, rules_by_id) is None


def test_the_digit_normalizer_answers_to_its_own_name():
    assert _normalize_rule_number("05") == "5"
    assert _normalize_rule_number("000") == "0"


def test_meta_and_plain_rule_numbers_do_not_collide():
    rules = [{"id": "openai_rules3"}, {"id": "openai_meta_rules3"}]
    index = build_rule_id_index(rules)
    assert normalize_preempting_rule_id("Rule 3", index) == "openai_rules3"
    assert normalize_preempting_rule_id("Meta-Rule 3", index) == "openai_meta_rules3"


def test_index_skips_rules_without_a_usable_id():
    assert build_rule_id_index([{"id": ""}, {"rule": "no id at all"}]) == {}


def test_an_id_without_digits_is_indexed_by_its_own_spelling_alone():
    assert build_rule_id_index([{"id": "Preamble"}, {"id": "Rule 2"}]) == {
        "preamble": "Preamble", "rule 2": "Rule 2", "2": "Rule 2", "rule2": "Rule 2"}


def test_first_rule_wins_on_a_duplicate_number():
    index = build_rule_id_index([{"id": "a_rules3"}, {"id": "b_rules3"}])
    assert normalize_preempting_rule_id("Rule 3", index) == "a_rules3"


def test_a_rule_id_that_is_not_a_string_or_is_blank_resolves_to_nothing():
    index = build_rule_id_index([{"id": "openai_rules3"}])
    assert normalize_preempting_rule_id(None, index) is None
    assert normalize_preempting_rule_id("   ", index) is None
    assert normalize_preempting_rule_id("(Rule 3).", index) == "openai_rules3"


#: What the implicated context references for a meta-rule of each text, over Rules 1-11: the recorded reading,
#: kept for every input, published or not.
IMPLICATED_READING = {
    "Rules 3-5 and 9 apply.": ["Rule 3"], "Rules 3-5 and 9-11": ["Rule 3"], "Rules 27 through 43": [],
    "Rules 27-43), then Rule 17.": [], "Rules 3-50.5": ["Rule 3"], "Rules 3 through five": ["Rule 3"],
    "Rules 3 to five": ["Rule 3"], "Rule 3-five": ["Rule 3"], "Rules three-5": [], "Rules 5-3": ["Rule 5"],
    "see Rule 12.4": [], "Rule 5 to protect users": ["Rule 5"], "Rules 3, 4, and 5 apply": ["Rule 3", "Rule 4", "Rule 5"],
    "Meta-Rule 2 governs Rules 1 & 2": ["Rule 1", "Rule 2"], "Rule 3 and 10 other constraints": ["Rule 3", "Rule 10"],
    "no reference here": [], "Rules 1-2, Meta-Rule 1": ["Rule 1"],
}


class TestRuleRanges:
    RULES = [{"id": f"Rule {n}", "rule": f"rule {n}"} for n in (1, 2, 4, 5, 6, 9, 10, 11)]

    def _map(self, text, expand=True):
        return build_rule_implication_map([*self.RULES, {"id": "Meta-Rule 1", "rule": text}], expand_ranges=expand)

    @pytest.mark.parametrize("text", sorted(IMPLICATED_READING))
    def test_implicated_reads_every_text_as_the_release_base_did(self, text):
        rules = [{"id": f"Rule {n}", "rule": f"rule {n}"} for n in range(1, 12)]
        assert build_rule_implication_map([*rules, {"id": "Meta-Rule 1", "rule": text}])["Meta-Rule 1"] == (
            IMPLICATED_READING[text])

    @pytest.mark.parametrize(("text", "expanded"), [
        ("Rules 3 - 5 and 9 apply.", [3, 4, 5, 9]),
        ("Rules 3 \u2013 5", [3, 4, 5]),
        ("Rules 3-5, 9.5", [3, 4, 5]),
        ("We apply Rule 3, 2026-10-28 being the start date.", [3]),
        ("Rules 3-5 and 9-11", [3, 4, 5, 9, 10, 11]),
        ("Rules 1, 2, or 3 thru 4, and 5\u20146", [1, 2, 3, 4, 5, 6]),
        ("Rules 1 through 11", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ("Rules three-5", []),
        ("Rule 99 applies", []),
        ("Rule 3/4 applies", [3]),
        ("see Rule 12.4", []),
        ("no reference here", []),
        # A separator not followed by a digit is prose: the item is the rule alone and the list ends there.
        ("Rule 5 to protect users", [5]),
        ("Rule 5 - the user's rule - applies", [5]),
        ("Rules 3 through fourteen", [3]),
        ("Rule 3-five", [3]),
        ("Rules 1 and 3-twelve", [1, 3]),
        ("Rules 1, 3 through the rest, 9", [1, 3]),
    ])
    def test_the_grammar_reads_a_reference_list_and_ends_it_at_the_first_thing_that_is_no_item(self, text, expanded):
        rules = [{"id": f"Rule {n}", "rule": f"rule {n}"} for n in range(1, 12)]
        found = build_rule_implication_map([*rules, {"id": "Meta-Rule 1", "rule": text}], expand_ranges=True)
        assert found["Meta-Rule 1"] == [f"Rule {n}" for n in expanded]

    @pytest.mark.parametrize(("text", "quoted"), [
        ("Rules 3 through 50.5", "Rules 3 through 50.5"),
        ("Rules 3-50.5", "Rules 3-50.5"),
        ("Rules 5-3", "Rules 5-3"),
        ("Rules 3-12 apply", "Rules 3-12"),
        ("Rules 3-5-7", "Rules 3-5-7"),
        ("Rules 1 and 3 to 12", "Rules 1 and 3 to 12"),
    ])
    def test_a_range_whose_end_is_no_rule_of_the_ruleset_at_least_its_first_is_refused_quoted(self, text, quoted):
        rules = [{"id": f"Rule {n}", "rule": f"rule {n}"} for n in range(1, 12)]
        with pytest.raises(RuleRangeError, match=re.escape(repr(quoted))):
            build_rule_implication_map([*rules, {"id": "Meta-Rule 1", "rule": text}], expand_ranges=True)
        build_rule_implication_map([*rules, {"id": "Meta-Rule 1", "rule": text}])

    def test_a_range_names_only_the_rules_of_the_ruleset_in_it(self):
        assert self._map("Rules 1-5 come first")["Meta-Rule 1"] == ["Rule 1", "Rule 2", "Rule 4", "Rule 5"]
        assert self._map("Rules 1-5 come first", expand=False)["Meta-Rule 1"] == ["Rule 1"]
        assert read_rule_references(None, {1}) == []
        assert read_rule_references("Meta-Rules 1 to 2", {1}, {1, 2}) == [("Meta-Rule", 1, 2)]

    def test_the_paper_s_revised_rulesets_expand_under_the_new_context_alone(self):
        from pathlib import Path

        from ruleforge.core.files import process_jsonl_file
        from ruleforge.rulesets.variants import build_rule_variants

        rules = list(process_jsonl_file(Path(__file__).resolve().parents[1] / "data/rules/modelspec/ruleforge/"
                                        "seed9_runner_up.jsonl"))
        _, old = build_rule_variants(rules, "implicated")
        _, new = build_rule_variants(rules, "implicated_ranges")
        assert old["Meta-Rule 1"] == ["Rule 27"]
        assert new["Meta-Rule 1"] == [f"Rule {n}" for n in range(27, 44)]

    @pytest.mark.parametrize("text", ["Rules 1-5 come first", "Rules 5-3", "Rules 3-five", "Rules 3 through five"])
    def test_a_new_run_on_the_recorded_context_warns_once_and_never_refuses(self, caplog, text):
        from ruleforge.rulesets.variants import warn_unexpanded_ranges

        rules = [*self.RULES, {"id": "Meta-Rule 1", "rule": text}]
        with caplog.at_level("WARNING", logger="ruleforge.rulesets.variants"):
            assert warn_unexpanded_ranges(rules, "implicated") is True
            assert warn_unexpanded_ranges(rules, "implicated_ranges") is False
            assert warn_unexpanded_ranges(self.RULES, "implicated") is False
        assert [record.getMessage().count("'implicated_ranges'") for record in caplog.records] == [1]
        assert looks_like_rule_range(text) and not looks_like_rule_range(None)
