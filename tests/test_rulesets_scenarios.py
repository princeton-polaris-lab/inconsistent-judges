from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge import scenarios as scenarios_module
from ruleforge.rulesets.subsets import WITH_REPLACEMENT, build_manifest, load_manifest
from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.rulesets import Rule, Ruleset, RulesetError
from ruleforge.rulesets.rules import round_trip_refusal
from ruleforge.core.errors import SamplingDesignError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import sha256_file
from ruleforge.scenarios import ScenarioSet, ScenarioSetError
from ruleforge.core.artifacts import compute_scenario_id
from ruleforge.tasks.history import read_scenario_set_from_file
from ruleforge.rulesets.levels import load_rule_levels

REPO = Path(__file__).resolve().parents[1]
RULES = [{"id": " Rule 1 ", "rule": "Be kind.", "rule_type": "root"},
         {"id": "Rule 2", "rule": "See Rule 1.", "rule_type": "user"},
         {"id": "Meta-Rule 1", "rule": "Read Rule 2 first.", "rule_type": "guideline"}]
LEVELS = {"Rule 1": "root", "Rule 2": "user", "Meta-Rule 1": "guideline"}


def _write(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _rulesets(*roots: Path) -> dict[Path, list[dict]]:
    """Every JSONL file under ``roots`` whose rows are all rules (objects with an ``id`` and a ``rule``)."""
    found = {}
    for root in roots:
        for path in sorted(root.rglob("*.jsonl")):
            try:
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except ValueError:
                continue
            if rows and all(isinstance(row, dict) and {"id", "rule"} <= set(row) for row in rows):
                found[path] = rows
    return found


def test_every_ruleset_of_the_repository_comes_back_from_the_text_the_optimizer_writes():
    rulesets = _rulesets(REPO / "data" / "rules", REPO / "tests" / "fixtures")
    assert {path.relative_to(REPO).as_posix() for path in rulesets} >= {
        "data/rules/modelspec/openai_rules_all.jsonl", "data/rules/modelspec/openai_rules_all_full.jsonl",
        "data/rules/modelspec/ruleforge/seed5_winner.jsonl", "data/rules/modelspec/ruleforge/seed9_runner_up.jsonl",
        "tests/fixtures/optimize/rules.jsonl", "tests/fixtures/rules/openai_subset_prolific.jsonl"}
    for path, rows in rulesets.items():
        assert round_trip_refusal(rows, str(path), exact_ids=False) is None, path
        annotated = any(not str(row["id"]).strip().startswith(("Rule ", "Meta-Rule ")) for row in rows)
        assert (round_trip_refusal(rows, str(path), exact_ids=True) is None) is not annotated, path


@pytest.mark.parametrize(("rows", "exact", "comes_back"), [
    ([{"id": "Rule 1", "rule": "Be kind. Rule 2: be brief."}], False, "its rule 0 ('Rule 1', 'Be kind. Rule 2: be "
                                                                     "brief.') would come back as ('Rule 1', 'Be kind.')"),
    ([{"id": "openai_rules7", "rule": "Be kind."}], True, "would come back as ('Rule 7', 'Be kind.')"),
    ([{"id": "Guideline A", "rule": "Be kind."}], False, "would come back as ('line_1', 'Guideline A: Be kind.')"),
    ([{"id": "Rule 1", "rule": " Be kind."}], False, "would come back as ('Rule 1', 'Be kind.')"),
])
def test_a_ruleset_that_would_not_come_back_is_refused_naming_its_first_such_rule(rows, exact, comes_back):
    refusal = round_trip_refusal(rows, "ruleset r", exact_ids=exact)
    assert refusal.startswith("ruleset r: the optimizer writes a ruleset as 'Rule <n>: <text>'") and comes_back in refusal
    assert round_trip_refusal([{"id": "openai_rules7", "rule": "Be kind."}], "r", exact_ids=False) is None


def test_the_round_trip_reads_a_one_shot_iterable_as_a_list():
    rows = [{"id": "Rule 1", "rule": "a"}, {"id": "Rule 2", "rule": "b"}]
    assert round_trip_refusal(iter(rows), "r", exact_ids=True) is None
    assert round_trip_refusal((row for row in rows), "r", exact_ids=False) is None


def test_a_ruleset_is_named_by_its_file_s_basename_and_bytes(tmp_path):
    path = _write(tmp_path / "mine.jsonl", RULES)
    ruleset = Ruleset.load(path)
    assert (ruleset.name, ruleset.sha256, ruleset.path) == ("mine.jsonl", sha256_file(path), path)
    assert ruleset.rules[0] == Rule(id="Rule 1", text="Be kind.", fields=RULES[0])
    assert [rule.is_meta for rule in ruleset.rules] == [False, False, True]
    assert ruleset.rows() == RULES and ruleset.non_meta_ids() == ("Rule 1", "Rule 2")
    assert Ruleset.load(path, name="pinned.jsonl", expect_sha256=sha256_file(path)).name == "pinned.jsonl"


def test_every_published_ruleset_is_named_as_its_conditions_name_it():
    from tests.identity import published as pub

    for (name, sha), path in pub.repo_rule_files().items():
        ruleset = Ruleset.load(path)
        assert (ruleset.name, ruleset.sha256) == (name, sha)


@pytest.mark.parametrize(
    ("content", "message"),
    [("", "holds no rules"), ('{"id": "Rule 1"}\n', "row 0 is not a rule"),
     ('{"id": 1, "rule": "x"}\n', "row 0 is not a rule"), ("not json\n", "cannot read ruleset")],
)
def test_a_file_that_is_not_a_ruleset_is_refused(tmp_path, content, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RulesetError, match=message):
        Ruleset.load(path)


def test_a_ruleset_other_than_the_expected_one_is_refused(tmp_path):
    path = _write(tmp_path / "mine.jsonl", RULES)
    with pytest.raises(RulesetError, match="not the expected " + "0" * 64):
        Ruleset.load(path, expect_sha256="0" * 64)
    with pytest.raises(RulesetError, match="cannot read ruleset"):
        Ruleset.load(tmp_path / "absent.jsonl")


def test_a_ruleset_s_prompt_texts_are_the_campaign_s(tmp_path):
    path = _write(tmp_path / "mine.jsonl", RULES)
    levels_path = _write(tmp_path / "levels.jsonl", RULES)
    ruleset = Ruleset.load(path)
    levels = ruleset.with_levels(levels_path)
    assert levels.by_id == load_rule_levels(path, levels_path).by_id == LEVELS
    rows = [dict(rule) for rule in RULES]
    assert ruleset.text() == rule_set_text(rows)
    assert ruleset.text(levels) == ruleset.text(LEVELS) == rule_set_text(rows, "superiors", LEVELS)
    for context in ("isolated", "implicated"):
        assert ruleset.variants(context) == build_rule_variants(rows, context)
    assert ruleset.variants("superiors", levels) == build_rule_variants(rows, "superiors", LEVELS)


def test_a_ruleset_s_subsets_are_the_committed_subset_files(tmp_path):
    meta, entries = load_manifest(REPO / "data" / "rules" / "subset_manifest")
    ruleset = Ruleset.load(REPO / meta["source_ruleset"]["path"], expect_sha256=meta["source_ruleset"]["sha256"])
    written = ruleset.subsets(seeds=[0, 3], strategy=WITH_REPLACEMENT, out=tmp_path)
    assert sorted(written) == [0, 3] and len(written[0]) == 56
    regenerated = build_manifest(tmp_path / WITH_REPLACEMENT)
    assert regenerated == {path: digest for path, digest in entries.items() if path.split("/")[0] in
                           ("seed_0", "seed_3")}


def test_a_ruleset_with_no_file_has_no_levels_or_subsets(tmp_path):
    ruleset = Ruleset(rules=(Rule(id="Rule 1", text="x"),), name="unsaved", sha256="0" * 64)
    assert ruleset.rules[0].row() == {}
    with pytest.raises(RulesetError, match="ruleset unsaved has no file"):
        ruleset.with_levels(tmp_path / "levels.jsonl")
    with pytest.raises(RulesetError, match="has no file"):
        ruleset.subsets(seeds=[0], strategy=WITH_REPLACEMENT, out=tmp_path)


ROWS = [{"prompt": "p0", "response": "r0", "source": "a", "n": 1},
        {"prompt": "p1", "response": "r1", "source": "b"},
        {"conversation": [{"role": "user", "content": "hi"}], "source": "c"}]


def test_a_scenario_set_keeps_every_row_whole_and_in_order(tmp_path):
    path = _write(tmp_path / "pool.jsonl", ROWS)
    scenarios = ScenarioSet.load(path, expect_sha256=sha256_file(path), expect_count=3)
    assert list(scenarios.rows) == ROWS and list(scenarios) == ROWS and len(scenarios) == 3
    assert scenarios.ids == tuple(compute_scenario_id(row) for row in ROWS)
    assert scenarios.manifest == ScenarioManifest.build(path)
    assert (scenarios.sha256, scenarios.path) == (sha256_file(path), path)
    assert scenarios.by_id() == dict(zip(scenarios.ids, ROWS))


def test_the_population_is_the_manifest_the_published_campaigns_froze():
    population = ScenarioSet.load(REPO / "data" / "populations" / "frozen_500_r3.jsonl", expect_count=500)
    assert population.manifest.manifest_sha256.startswith("a22af097")
    assert population.sha256.startswith("c16f66f4")


def test_a_scenario_set_line_that_does_not_parse_is_refused_by_the_optimizer_s_reader_too(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\nnot json\n", encoding="utf-8")
    for read in (lambda: read_scenario_set_from_file(path, ("prompt", "response")), lambda: ScenarioSet.load(path)):
        with pytest.raises(ValueError, match=":2: invalid JSON"):
            read()


def test_a_scenario_set_other_than_the_expected_one_is_refused(tmp_path):
    path = _write(tmp_path / "pool.jsonl", ROWS)
    with pytest.raises(ScenarioSetError, match="not the expected " + "0" * 64):
        ScenarioSet.load(path, expect_sha256="0" * 64)
    with pytest.raises(SamplingDesignError, match="requires exactly 4"):
        ScenarioSet.load(path, expect_count=4)


def test_a_scenario_file_that_changes_while_it_is_read_is_refused(tmp_path, monkeypatch):
    path = _write(tmp_path / "pool.jsonl", ROWS)
    monkeypatch.setattr(scenarios_module, "load_jsonl_hashed", lambda source: (ROWS[:2], sha256_file(source)))
    with pytest.raises(ScenarioSetError, match="changed while it was read"):
        ScenarioSet.load(path)
