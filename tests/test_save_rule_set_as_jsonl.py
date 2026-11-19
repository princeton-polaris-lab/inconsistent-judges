import json
import pytest
from pathlib import Path

from ruleforge.rulesets.rules import save_rule_set_as_jsonl


@pytest.fixture
def temp_jsonl_file(tmp_path):
    return tmp_path / "test_rules.jsonl"


def read_jsonl(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def test_save_single_rule(temp_jsonl_file):
    text = "Rule 1: Provide value"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Provide value"


def test_save_multiple_rules(temp_jsonl_file):
    text = "Rule 1: Provide value\nRule 2: Stay consistent\nRule 3: Be helpful"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 3
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Provide value"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Stay consistent"
    assert result[2]["id"] == "Rule 3"
    assert result[2]["rule"] == "Be helpful"


def test_save_meta_rules(temp_jsonl_file):
    text = "Meta-Rule 1: Top level guidance\nMeta-Rule 5: Another meta rule"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Meta-Rule 1"
    assert result[0]["rule"] == "Top level guidance"
    assert result[1]["id"] == "Meta-Rule 5"
    assert result[1]["rule"] == "Another meta rule"


def test_save_mixed_rules_and_meta_rules(temp_jsonl_file):
    text = (
        "Rule 1: First directive\n"
        "Meta-Rule 5: Meta directive\n"
        "Rule 7: Another directive"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 3
    assert result[0]["id"] == "Rule 1"
    assert result[1]["id"] == "Meta-Rule 5"
    assert result[2]["id"] == "Rule 7"


def test_empty_lines_are_skipped(temp_jsonl_file):
    text = "Rule 1: First rule\n\n\nRule 2: Second rule\n\n"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[1]["id"] == "Rule 2"


def test_lines_without_colon_kept_by_default(temp_jsonl_file):
    text = "Rule 1: Valid rule\nThis line has no colon\nRule 2: Another valid rule"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Valid rule\nThis line has no colon"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Another valid rule"


def test_lines_without_colon_skipped_when_specified(temp_jsonl_file):
    text = "Rule 1: Valid rule\nThis line has no colon\nRule 2: Another valid rule"
    save_rule_set_as_jsonl(text, temp_jsonl_file, on_missing_colon="skip")

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Valid rule"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Another valid rule"


def test_custom_content_key(temp_jsonl_file):
    text = "Rule 1: Provide value\nRule 2: Stay consistent"
    save_rule_set_as_jsonl(text, temp_jsonl_file, content_key="guideline")

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[0]["guideline"] == "Provide value"
    assert "rule" not in result[0]


def test_space_separated_rules_normalized(temp_jsonl_file):
    text = "Rule 1: First rule Rule 2: Second rule"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "First rule"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Second rule"


def test_space_separated_meta_rules_normalized(temp_jsonl_file):
    text = "Meta-Rule 1: First meta Meta-Rule 2: Second meta"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Meta-Rule 1"
    assert result[0]["rule"] == "First meta"
    assert result[1]["id"] == "Meta-Rule 2"
    assert result[1]["rule"] == "Second meta"


def test_whitespace_handling(temp_jsonl_file):
    text = "  Rule 1: Provide value  \n  Rule 2: Stay consistent  "
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Provide value"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Stay consistent"


def test_colon_in_description(temp_jsonl_file):
    text = "Rule 1: Time format is HH:MM:SS"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Time format is HH:MM:SS"


def test_non_standard_rule_ids(temp_jsonl_file):
    # Non-standard ids with colons need the skip mode.
    text = "Guideline A: Do this\nPrinciple 5: Do that"
    save_rule_set_as_jsonl(text, temp_jsonl_file, on_missing_colon="skip")

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Guideline A"
    assert result[0]["rule"] == "Do this"
    assert result[1]["id"] == "Principle 5"
    assert result[1]["rule"] == "Do that"


def test_unicode_content(temp_jsonl_file):
    text = "Rule 1: Use emoji 😀 and symbols ≈ ± ∞"
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 1
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Use emoji 😀 and symbols ≈ ± ∞"


def test_real_world_example(temp_jsonl_file):
    # The example of test_search_replace.py.
    text = (
        "Rule 1: Provide value\n"
        "Rule 2: Stay consistent\n"
        "Meta-Rule 5: Meta directive\n"
        "Rule 7: Another directive"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 4
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Provide value"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Stay consistent"
    assert result[2]["id"] == "Meta-Rule 5"
    assert result[2]["rule"] == "Meta directive"
    assert result[3]["id"] == "Rule 7"
    assert result[3]["rule"] == "Another directive"


def test_multiline_rule_with_bullet_points_kept(temp_jsonl_file):
    text = (
        "Rule 1: Original guidance\n"
        "  - keep old detail line\n"
        "  - another old detail\n"
        "Rule 2: Leave untouched"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert (
        result[0]["rule"]
        == "Original guidance\n- keep old detail line\n- another old detail"
    )
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Leave untouched"


def test_multiline_rule_with_bullet_points_skipped(temp_jsonl_file):
    text = (
        "Rule 1: Original guidance\n"
        "  - keep old detail line\n"
        "  - another old detail\n"
        "Rule 2: Leave untouched"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file, on_missing_colon="skip")

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert result[0]["rule"] == "Original guidance"
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Leave untouched"


def test_multiline_meta_rule_with_details_kept(temp_jsonl_file):
    text = (
        "Meta-Rule 1: Rule hierarchy and conflict resolution\n"
        "  (1) Safety rules have highest priority\n"
        "  (2) Integrity rules come next\n"
        "Rule 1: Regular rule"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Meta-Rule 1"
    assert result[0]["rule"] == (
        "Rule hierarchy and conflict resolution\n"
        "(1) Safety rules have highest priority\n"
        "(2) Integrity rules come next"
    )
    assert result[1]["id"] == "Rule 1"
    assert result[1]["rule"] == "Regular rule"


def test_multiline_meta_rule_with_details_skipped(temp_jsonl_file):
    text = (
        "Meta-Rule 1: Rule hierarchy and conflict resolution\n"
        "  (1) Safety rules have highest priority\n"
        "  (2) Integrity rules come next\n"
        "Rule 1: Regular rule"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file, on_missing_colon="skip")

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 2
    assert result[0]["id"] == "Meta-Rule 1"
    assert result[0]["rule"] == "Rule hierarchy and conflict resolution"
    assert result[1]["id"] == "Rule 1"
    assert result[1]["rule"] == "Regular rule"


def test_preamble_before_first_rule_kept(temp_jsonl_file):
    text = (
        "Some preamble text\n"
        "Another preamble line\n"
        "Rule 1: First rule\n"
        "Rule 2: Second rule"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    assert len(result) == 3
    assert result[0]["id"] == "line_1"
    assert result[0]["rule"] == "Some preamble text\nAnother preamble line"
    assert result[1]["id"] == "Rule 1"
    assert result[1]["rule"] == "First rule"
    assert result[2]["id"] == "Rule 2"
    assert result[2]["rule"] == "Second rule"


def test_multiline_rule_with_colon_in_continuation(temp_jsonl_file):
    text = (
        "Rule 1: Main guidance\n"
        "Details: more info here\n"
        "Time format: HH:MM:SS\n"
        "Rule 2: Another rule"
    )
    save_rule_set_as_jsonl(text, temp_jsonl_file)

    result = read_jsonl(temp_jsonl_file)

    # "Details:" and "Time format:" don't match "Rule X:" or "Meta-Rule X:" pattern
    assert len(result) == 2
    assert result[0]["id"] == "Rule 1"
    assert (
        result[0]["rule"]
        == "Main guidance\nDetails: more info here\nTime format: HH:MM:SS"
    )
    assert result[1]["id"] == "Rule 2"
    assert result[1]["rule"] == "Another rule"


def test_text_with_no_rule_start_is_one_preamble_rule_or_nothing(temp_jsonl_file):
    save_rule_set_as_jsonl("A preamble\nand its second line", temp_jsonl_file)
    assert read_jsonl(temp_jsonl_file) == [{"id": "line_1", "rule": "A preamble\nand its second line"}]
    save_rule_set_as_jsonl("", temp_jsonl_file)
    assert read_jsonl(temp_jsonl_file) == []


def test_blank_lines_are_skipped_when_lines_without_a_colon_are(temp_jsonl_file):
    # a blank run before a rule start folds into the split, so this one sits before a line that is not one
    save_rule_set_as_jsonl("Rule 1: a\n\n   \nno colon here\nRule 2: b", temp_jsonl_file, on_missing_colon="skip")
    assert read_jsonl(temp_jsonl_file) == [{"id": "Rule 1", "rule": "a"}, {"id": "Rule 2", "rule": "b"}]
