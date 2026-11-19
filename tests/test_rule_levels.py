from __future__ import annotations

import json

import pytest

from ruleforge.rulesets import levels as rl
from tests.test_leniency_campaign import RULES, write_rules

#: The study's levels, by rule number (RULES' eleven): no rule is a system rule, so the system level holds none.
STUDY_LEVELS = {1: "developer", 2: "guideline", 3: "guideline", 4: "guideline", 5: "root", 6: "root", 7: "root",
                8: "user", 9: "user", 10: "user", 11: "user"}


def write_levels(path, rules=RULES, levels=None):
    """A levels file aligned with ``rules`` (Model Spec ids, the rule text, ``levels[k]`` for the k-th rule)."""
    levels = levels or STUDY_LEVELS
    path.write_text("".join(json.dumps({"id": f"openai_rules{k}", "rule_type": levels[k], "rule": rule["rule"],
                                        "explanation": "..."}) + "\n" for k, rule in enumerate(rules, start=1)))
    return path


def test_the_level_order_is_the_model_spec_s_chain_of_command():
    assert rl.LEVEL_ORDER == ("root", "system", "developer", "user", "guideline")
    assert "Model Spec" in rl.LEVEL_SOURCE and rl.LEVEL_FIELD == "rule_type"


def test_the_real_levels_file_aligns_with_the_paper_s_ruleset(repo_root):
    levels = rl.load_rule_levels(repo_root / "data/rules/modelspec/openai_rules_all.jsonl",
                                 repo_root / "data/rules/modelspec/openai_rules_all_full.jsonl")
    by_level = levels.rules_by_level()
    assert {level: len(ids) for level, ids in by_level.items()} == {
        "root": 20, "system": 3, "developer": 1, "user": 15, "guideline": 18}
    assert by_level["root"] == [f"Rule {n}" for n in range(20, 40)]
    assert by_level["system"] == ["Rule 40", "Rule 41", "Rule 42"]
    assert by_level["developer"] == ["Rule 1"]
    assert by_level["user"] == [f"Rule {n}" for n in range(43, 58)]
    assert by_level["guideline"] == [f"Rule {n}" for n in range(2, 20)]
    assert levels.ruleset["sha256"].startswith("a99062f00a9f")
    assert levels.levels_file == {"name": "openai_rules_all_full.jsonl", "sha256": (
        "99eac16d5fc4c8cd028b091b6cab23badcd1573bd33d9765f0ca436e91f7130a")}
    assert levels.rank["Rule 20"] == 0 and levels.rank["Rule 2"] == 4
    assert levels.by_id["Rule 43"] == "user" and list(levels.by_id) == list(levels.rule_ids)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda rules, levelled: (rules, levelled[:-1]), "11 rules but 10 levelled rules"),
        (lambda rules, levelled: ([], []), "the ruleset holds no rule"),
        (lambda rules, levelled: ([{**rules[0], "id": " "}, *rules[1:]], levelled), "has no id, or two share one"),
        (lambda rules, levelled: ([{**rules[0], "id": None}, *rules[1:]], levelled), "has no id, or two share one"),
        (lambda rules, levelled: ([{**rules[0], "id": 1}, *rules[1:]], levelled), "has no id, or two share one"),
        (lambda rules, levelled: ([rules[0], {**rules[1], "id": "Rule 1"}, *rules[2:]], levelled),
         "has no id, or two share one"),
        (lambda rules, levelled: (rules, [levelled[1], levelled[0], *levelled[2:]]),
         r"rule text differs at position\(s\) \['1 \(Rule 1\)', '2 \(Rule 2\)'\]"),
        (lambda rules, levelled: ([{"id": "Rule 1"}, *rules[1:]], [{"rule_type": "root"}, *levelled[1:]]),
         r"differs at position\(s\) \['1 \(Rule 1\)'\]"),
        (lambda rules, levelled: (rules, [{**levelled[0], "rule_type": "platform"}, *levelled[1:]]),
         r"no level, or one not in .* at position\(s\) \[\"1 \(Rule 1: 'platform'\)\"\]"),
        (lambda rules, levelled: (rules, [{k: v for k, v in levelled[0].items() if k != "rule_type"}, *levelled[1:]]),
         r"at position\(s\) \['1 \(Rule 1: None\)'\]"),
    ],
)
def test_the_levels_file_is_refused_unless_it_aligns_row_by_row(tmp_path, edit, match):
    levelled = [json.loads(line) for line in write_levels(tmp_path / "full.jsonl").read_text().splitlines()]
    rules, levelled = edit([dict(rule) for rule in RULES], levelled)
    write_rules(tmp_path / "rules.jsonl", rules)
    (tmp_path / "full.jsonl").write_text("".join(json.dumps(row) + "\n" for row in levelled))
    with pytest.raises(rl.RuleLevelsError, match=match):
        rl.load_rule_levels(tmp_path / "rules.jsonl", tmp_path / "full.jsonl")


def test_a_rule_s_superiors_are_the_rules_of_strictly_higher_levels_in_file_order():
    order = [f"Rule {k}" for k in STUDY_LEVELS]
    levels = {f"Rule {k}": level for k, level in STUDY_LEVELS.items()}
    assert rl.superior_ids("Rule 5", levels, order) == []  # root: nothing outranks it, not even another root rule
    assert rl.superior_ids("Rule 1", levels, order) == ["Rule 5", "Rule 6", "Rule 7"]  # developer
    assert rl.superior_ids("Rule 8", levels, order) == ["Rule 1", "Rule 5", "Rule 6", "Rule 7"]  # user
    assert rl.superior_ids("Rule 2", levels, order) == [  # guideline: everything but the guidelines
        "Rule 1", "Rule 5", "Rule 6", "Rule 7", "Rule 8", "Rule 9", "Rule 10", "Rule 11"]
    with pytest.raises(rl.RuleLevelsError, match=r"for rule\(s\) \['Rule 12'\]"):
        rl.superior_ids("Rule 1", levels, [*order, "Rule 12"])
    with pytest.raises(rl.RuleLevelsError, match=r"for rule\(s\) \['Rule 1'\]"):
        rl.superior_ids("Rule 1", {**levels, "Rule 1": "platform"}, order)
