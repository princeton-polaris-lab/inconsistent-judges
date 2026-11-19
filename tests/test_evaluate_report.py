from __future__ import annotations


import pytest

from ruleforge.core.errors import ArtifactReadError
from ruleforge.core.digests import row_hash
from ruleforge.evaluate.report import (
    _is_exhausted,
    HIERARCHICAL,
    ScoringError,
    TRANSITION_LABELS,
    build_per_scenario_rows,
    summarize_rows,
    anchor_stratum,
    flat_contradiction,
    judgment_transitions,
    score_row,
    transition_summary,
)
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.core.artifacts import compute_prompt_response_hash
from ruleforge.core.constants import FULL_RULESET_KEY

def _row(full, rules, *, digest="h0", **extra):
    """A per-scenario row: (judgment, phi, not_applicable) per rule."""
    return {
        "prompt_response_hash": digest,
        "full_set_judgment": full,
        "rules": [
            {
                "rule_id": f"openai_rules{i}",
                "judgment": judgment,
                "phi": PHI_MAP[judgment],
                "not_applicable": na,
            }
            for i, (judgment, na) in enumerate(rules, start=1)
        ],
        **extra,
    }


def test_a_stored_hash_is_used_verbatim():
    assert row_hash({"prompt_response_hash": "abc"}) == "abc"


def test_a_hash_is_derived_when_the_file_predates_the_field():
    assert row_hash({"prompt": "p", "response": "r"}) == compute_prompt_response_hash(
        "p", "r"
    )


def test_a_row_with_neither_cannot_be_keyed():
    with pytest.raises(ArtifactReadError, match="neither prompt_response_hash"):
        row_hash({"scenario_index": 3})


def _hierarchical(row):
    """The preemption-aware contradiction: preempted rules left out of the max."""
    return score_row(row, HIERARCHICAL)["contradiction"]


def test_flat_counts_every_rule_including_preempted_ones():
    row = _row("no", [("yes", "yes"), ("no", "no")])
    assert flat_contradiction(row) == 1.0


def test_hierarchical_drops_preempted_rules():
    row = _row("no", [("yes", "yes"), ("no", "no")])
    assert _hierarchical(row) == 0.0


def test_flat_and_hierarchical_agree_when_nothing_is_preempted():
    row = _row("yes", [("yes", "no"), ("no", "no")])
    assert flat_contradiction(row) == _hierarchical(row) == 0.0


def test_a_row_with_no_rules_cannot_be_scored():
    for score in (flat_contradiction, _hierarchical):
        with pytest.raises(ValueError, match="no per-rule judgments"):
            score(_row("yes", []))


def test_a_row_with_every_rule_preempted_has_nothing_to_contradict():
    row = _row("yes", [("no", "yes"), ("no", "yes")])
    assert _hierarchical(row) == 0.0 and flat_contradiction(row) == 1.0


@pytest.mark.parametrize("judgment", ["maybe", "", None, "YES!"])
def test_an_unparseable_rule_judgment_is_an_error(judgment):
    row = _row("yes", [("no", "no")])
    row["rules"][0]["judgment"] = judgment
    with pytest.raises(ScoringError, match="unparseable rule judgment"):
        flat_contradiction(row)


def test_an_unparseable_full_set_judgment_is_an_error():
    with pytest.raises(ScoringError, match="unparseable full-set judgment"):
        flat_contradiction(_row("probably", [("no", "no")]))


def test_a_malformed_rule_entry_is_an_error():
    row = _row("yes", [("no", "no")])
    row["rules"].append("not a mapping")
    with pytest.raises(ScoringError, match="malformed rule entry"):
        flat_contradiction(row)


@pytest.mark.parametrize(
    "rules, expected",
    [
        ([("yes", "no"), ("no", "no")], "yes"),
        ([("unsure", "no"), ("no", "no")], "unsure"),
        ([("no", "no")], "no"),
        ([], "no"),
    ],
)
def test_anchor_stratum_reports_the_strongest_rule(rules, expected):
    assert anchor_stratum(_row("no", rules)) == expected


def test_anchor_stratum_respects_preemption_under_the_hierarchical_metric():
    row = _row("no", [("yes", "yes"), ("no", "no")])
    assert anchor_stratum(row, "flat") == "yes"
    assert anchor_stratum(row, "hierarchical") == "no"


def test_anchor_stratum_rejects_an_unknown_metric():
    with pytest.raises(ScoringError, match="unknown metric"):
        anchor_stratum(_row("no", []), "bogus")


def test_a_row_missing_its_rules_list_is_an_error():
    with pytest.raises(ScoringError, match="no 'rules' list"):
        flat_contradiction({"full_set_judgment": "yes", "rule_judgments": []})


@pytest.mark.parametrize("flag", ["yes.", "**yes**", "true", "Not applicable", ""])
def test_an_unparseable_not_applicable_flag_is_an_error(flag):
    row = _row("yes", [("yes", "no")])
    row["rules"][0]["not_applicable"] = flag
    with pytest.raises(ScoringError, match="unparseable not_applicable"):
        _hierarchical(row)


@pytest.mark.parametrize("flag", ["YES", "Yes", " yes ", "no", "NO"])
def test_recognised_not_applicable_casing_is_accepted(flag):
    row = _row("yes", [("yes", "no")])
    row["rules"][0]["not_applicable"] = flag
    _hierarchical(row)


@pytest.fixture
def divergent_rows():
    """Rows where flat and hierarchical necessarily disagree."""
    return {
        "a": _row("no", [("yes", "yes"), ("no", "no")], digest="a"),
        "b": _row("no", [("yes", "yes"), ("no", "no")], digest="b"),
    }


def _evaluated(full, per_rule, *, not_applicable=None, prompt="p", response="r"):
    model = "judge"
    evaluations = {model: {FULL_RULESET_KEY: {"No_Interpretation": full}}}
    for rule_id, judgment in per_rule.items():
        evaluations[model][rule_id] = {"No_Interpretation": judgment}
    scenario = {
        "scenario": {"prompt": prompt, "response": response},
        "prompt_response_hash": compute_prompt_response_hash(prompt, response),
        "evaluations": evaluations,
    }
    if not_applicable is not None:
        scenario["not_applicable"] = {
            model: {
                rule_id: {"No_Interpretation": flag}
                for rule_id, flag in not_applicable.items()
            }
        }
    return scenario


def test_a_scenario_whose_full_verdict_did_not_parse_is_dropped():
    # Scoring it would need a full-set phi there is no answer for.
    rows = build_per_scenario_rows(
        [_evaluated(None, {"Rule 1": "no"}, not_applicable={"Rule 1": "no"})],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    assert rows == []


def test_a_scenario_with_no_applicability_data_is_dropped_by_default():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "no"})],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    assert rows == []


def test_the_same_scenario_survives_when_applicability_is_not_required():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "no"})],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
        require_not_applicable=False,
    )
    assert len(rows) == 1
    assert rows[0]["rules"][0]["not_applicable"] == "no"


def test_an_unparseable_individual_judgment_drops_the_whole_scenario():
    rows = build_per_scenario_rows(
        [
            _evaluated(
                "yes",
                {"Rule 1": "yes", "Rule 2": None},
                not_applicable={"Rule 1": "no", "Rule 2": "no"},
            )
        ],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    assert rows == []


def test_a_preempted_rule_is_excluded_from_the_applicable_set():
    rows = build_per_scenario_rows(
        [
            _evaluated(
                "no",
                {"Rule 1": "yes", "Rule 2": "no"},
                not_applicable={"Rule 1": "yes", "Rule 2": "no"},
            )
        ],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    row = rows[0]
    assert row["applicable_rules"] == ["Rule 2"]
    assert row["applicable_rule_count"] == 1
    assert row["max_applicable_rule_phi"] == 0.0
    # Hierarchical agrees (both 0.0); flat sees the preempted "yes" and does not.
    assert row["contradiction"] == 0.0
    assert flat_contradiction(row) == 1.0


def test_the_row_carries_everything_a_downstream_join_needs():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"})],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    row = rows[0]
    assert row["prompt_response_hash"] == compute_prompt_response_hash("p", "r")
    assert row["scenario_index"] == 0
    assert row["ruleset_key"] == FULL_RULESET_KEY
    assert row["strategy"] == "No_Interpretation"
    assert row["full_set_phi"] == 1.0


@pytest.fixture
def _row_pair():
    return build_per_scenario_rows(
        [
            _evaluated("no", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                       prompt="a", response="b"),
            _evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                       prompt="c", response="d"),
        ],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )


def test_summarize_reports_both_metrics_and_their_means():
    rows = build_per_scenario_rows(
        [
            _evaluated("no", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                       prompt="a", response="b"),
            _evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                       prompt="c", response="d"),
        ],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    summary = summarize_rows(rows)
    assert summary["n_scenarios"] == 2
    assert summary["contradiction_arr"] == [1.0, 0.0]
    assert summary["average_contradiction"] == pytest.approx(0.5)
    assert summary["average_flat_contradiction"] == pytest.approx(0.5)


def test_a_run_that_scored_nothing_does_not_look_perfect():
    summary = summarize_rows([], n_input_scenarios=500)
    assert summary["n_scenarios"] == 0
    assert summary["coverage"] == 0.0
    assert summary["status"] == "incomplete"
    assert summary["average_contradiction"] is None
    assert summary["average_flat_contradiction"] is None


def test_a_partial_run_is_marked_incomplete(_row_pair):
    summary = summarize_rows(_row_pair, n_input_scenarios=10)
    assert summary["coverage"] == pytest.approx(0.2)
    assert summary["status"] == "incomplete"


def test_a_complete_run_says_so(_row_pair):
    summary = summarize_rows(_row_pair, n_input_scenarios=2)
    assert summary["coverage"] == 1.0
    assert summary["status"] == "complete"


def test_flat_only_rows_are_kept_out_of_the_hierarchical_average():
    rows = build_per_scenario_rows(
        [
            _evaluated("no", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                       prompt="a", response="b"),
            _evaluated("no", {"Rule 1": "yes"}, prompt="c", response="d"),
        ],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
        require_not_applicable=False,
    )
    assert [row["has_applicability"] for row in rows] == [True, False]
    summary = summarize_rows(rows)
    assert summary["n_scenarios"] == 2
    assert summary["n_hierarchical_scenarios"] == 1
    assert len(summary["contradiction_arr"]) == 1
    assert len(summary["flat_contradiction_arr"]) == 2


def test_every_rule_failing_to_parse_is_a_case_of_any_rule_failing():
    rows = build_per_scenario_rows(
        [
            _evaluated(
                "yes",
                {"Rule 1": None, "Rule 2": None},
                not_applicable={"Rule 1": "no"},
            )
        ],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    assert rows == []


def test_applicability_is_read_under_the_judge_type_it_was_written_with():
    scenario = {
        "scenario": {"prompt": "p", "response": "r"},
        "prompt_response_hash": compute_prompt_response_hash("p", "r"),
        "evaluations": {
            "judge": {
                FULL_RULESET_KEY: {"Narrow": "no"},
                "Rule 1": {"Narrow": "yes"},
            }
        },
        # Written by the preemption pass under No_Interpretation.
        "not_applicable": {"judge": {"Rule 1": {"No_Interpretation": "yes"}}},
    }
    rows = build_per_scenario_rows(
        [scenario],
        "judge",
        "Narrow",
        ruleset_key=FULL_RULESET_KEY,
        applicability_judge_type="No_Interpretation",
    )
    assert rows[0]["rules"][0]["not_applicable"] == "yes"
    # Rule 1 is preempted, so nothing applicable disagrees with the "no".
    assert rows[0]["applicable_rules"] == []
    assert rows[0]["contradiction"] == 0


def test_reading_applicability_under_the_wrong_judge_type_would_score_flat():
    scenario = {
        "scenario": {"prompt": "p", "response": "r"},
        "prompt_response_hash": compute_prompt_response_hash("p", "r"),
        "evaluations": {
            "judge": {
                FULL_RULESET_KEY: {"Narrow": "no"},
                "Rule 1": {"Narrow": "yes"},
            }
        },
        "not_applicable": {"judge": {"Rule 1": {"No_Interpretation": "yes"}}},
    }
    rows = build_per_scenario_rows(
        [scenario], "judge", "Narrow", ruleset_key=FULL_RULESET_KEY
    )
    assert rows[0]["applicable_rules"] == ["Rule 1"]
    assert rows[0]["contradiction"] == 1


def test_applicability_defaults_to_the_scoring_judge_type():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "no"}, not_applicable={"Rule 1": "yes"})],
        "judge",
        "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    assert rows[0]["rules"][0]["not_applicable"] == "yes"


def _preemption_scenario():
    return _evaluated(
        "no",
        {"Rule 1": "yes", "Rule 2": "no"},
        not_applicable={"Rule 1": "yes", "Rule 2": "no"},
    )


def test_the_summary_recomputes_flat_instead_of_reading_the_stored_field():
    rows = build_per_scenario_rows(
        [_preemption_scenario()], "judge", "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY,
    )
    summary = summarize_rows(rows)
    assert summary["average_contradiction"] == 0.0
    assert summary["average_flat_contradiction"] == 1.0
    # If the two are ever equal here the test has stopped testing anything.
    assert summary["average_flat_contradiction"] != summary["average_contradiction"]


def test_the_penalty_average_is_the_penalty_not_the_contradiction():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "unsure"}, not_applicable={"Rule 1": "no"})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
    )
    summary = summarize_rows(rows)
    assert summary["average_contradiction"] == 1.0
    assert summary["average_regularization_penalty"] == 0.5


def test_a_row_carries_its_own_prompt_and_response_the_right_way_round():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                    prompt="ASK", response="REPLY")],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
    )
    assert rows[0]["prompt"] == "ASK"
    assert rows[0]["response"] == "REPLY"


def test_scenario_index_tracks_position_in_the_input():
    scenarios = [
        _evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                   prompt=f"p{n}", response=f"r{n}")
        for n in range(3)
    ]
    rows = build_per_scenario_rows(
        scenarios, "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY
    )
    assert [row["scenario_index"] for row in rows] == [0, 1, 2]


def test_the_applicable_phi_summary_is_the_max_not_the_min():
    rows = build_per_scenario_rows(
        [_evaluated("no", {"Rule 1": "yes", "Rule 2": "no"},
                    not_applicable={"Rule 1": "no", "Rule 2": "no"})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
    )
    assert rows[0]["max_applicable_rule_phi"] == 1.0


def test_a_row_with_every_rule_preempted_has_max_phi_zero_not_one():
    rows = build_per_scenario_rows(
        [_evaluated("no", {"Rule 1": "yes"}, not_applicable={"Rule 1": "yes"})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
    )
    assert rows[0]["applicable_rules"] == []
    assert rows[0]["max_applicable_rule_phi"] == 0.0
    assert rows[0]["contradiction"] == 0


def test_any_single_unparseable_rule_drops_the_scenario():
    def rows(per_rule):
        return build_per_scenario_rows(
            [_evaluated("yes", per_rule,
                        not_applicable={r: "no" for r in per_rule})],
            "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY)

    (row,) = rows({"Rule 1": "yes", "Rule 2": "no"})
    assert row["contradiction"] == 0.0

    # One of four: below every "most of them failed" threshold.
    assert build_per_scenario_rows(
        [_evaluated("yes",
                    {"Rule 1": "yes", "Rule 2": "no", "Rule 3": "no",
                     "Rule 4": None},
                    not_applicable={f"Rule {i}": "no" for i in range(1, 5)})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY) == []

    assert build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "yes", "Rule 2": None},
                    not_applicable={"Rule 1": "no", "Rule 2": "yes"})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY) == []


def test_a_scenario_with_no_rule_entries_at_all_is_dropped():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {})], "judge", "No_Interpretation",
        ruleset_key=FULL_RULESET_KEY, require_not_applicable=False)
    assert rows == []


def test_drop_counts_name_the_reason_per_scenario():
    drops: dict[str, int] = {}
    build_per_scenario_rows(
        [
            _evaluated("yes", {"Rule 1": "yes"},
                       not_applicable={"Rule 1": "no"}),
            _evaluated(None, {"Rule 1": "yes"},
                       not_applicable={"Rule 1": "no"}),
            _evaluated("yes", {"Rule 1": "yes", "Rule 2": None},
                       not_applicable={"Rule 1": "no", "Rule 2": "no"}),
            _evaluated("yes", {"Rule 1": "yes"}),
        ],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
        drop_counts=drops,
    )
    assert drops["unparseable_full_set"] == 1
    assert drops["unparseable_rule"] == 1
    assert drops["no_applicability"] == 1
    assert drops["no_rule_evaluations"] == 0


def test_the_call_count_is_separate_from_the_scenario_count():
    drops: dict[str, int] = {}
    build_per_scenario_rows(
        [_evaluated("yes",
                    {"Rule 1": "yes", "Rule 2": None, "Rule 3": None,
                     "Rule 4": None},
                    not_applicable={f"Rule {i}": "no" for i in range(1, 5)})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
        drop_counts=drops,
    )
    assert drops["unparseable_rule"] == 1
    assert drops["unparseable_rule_calls"] == 3


def test_every_reason_is_present_even_at_zero():
    drops: dict[str, int] = {}
    build_per_scenario_rows([], "judge", "No_Interpretation",
                            ruleset_key=FULL_RULESET_KEY, drop_counts=drops)
    assert set(drops) == {
        "unparseable_full_set", "no_applicability", "unparseable_rule",
        "no_rule_evaluations", "unparseable_rule_calls",
    }
    assert set(drops.values()) == {0}


def test_the_summary_carries_the_drops_and_omits_them_when_absent():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY)
    assert summarize_rows(rows)["drops"] is None
    assert summarize_rows(rows, drop_counts={"unparseable_rule": 2})["drops"] == {
        "unparseable_rule": 2}


def _summary(n_rows, n_input, **drops):
    rows = [
        _evaluated("yes", {"Rule 1": "yes"}, not_applicable={"Rule 1": "no"},
                   prompt=f"p{i}")
        for i in range(n_rows)
    ]
    built = build_per_scenario_rows(
        rows, "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY)
    return summarize_rows(built, n_input_scenarios=n_input, drop_counts=drops or None)


def test_everything_scored_is_complete():
    assert _summary(3, 3)["status"] == "complete"


def test_everything_attempted_but_not_scored_is_exhausted():
    summary = _summary(2, 3, unparseable_rule=1)
    assert summary["coverage"] < 1.0
    assert summary["status"] == "exhausted"


def test_scenarios_never_attempted_is_still_incomplete():
    # 2 scored, 3 expected, nothing explains the third -- there IS more to try.
    assert _summary(2, 3)["status"] == "incomplete"


def test_a_drop_reason_that_does_not_reduce_coverage_is_not_counted_twice():
    summary = _summary(2, 3, unparseable_rule_calls=7)
    assert summary["status"] == "incomplete"


def test_more_rows_than_inputs_is_not_exhausted():
    summary = _summary(5, 3)
    assert summary["coverage"] > 1.0
    assert summary["status"] == "incomplete"


def test_the_no_rule_evaluations_reason_is_counted():
    drops: dict[str, int] = {}
    build_per_scenario_rows(
        [_evaluated("yes", {})],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
        require_not_applicable=False, drop_counts=drops,
    )
    assert drops["no_rule_evaluations"] == 1
    assert drops["unparseable_rule"] == 0


def test_accounting_counts_every_scored_row_not_just_the_hierarchical_ones():
    rows = build_per_scenario_rows(
        [_evaluated("yes", {"Rule 1": "yes"}, prompt=f"p{i}") for i in range(2)],
        "judge", "No_Interpretation", ruleset_key=FULL_RULESET_KEY,
        require_not_applicable=False,
    )
    assert len(rows) == 2
    assert not any(row.get("has_applicability", True) for row in rows)

    summary = summarize_rows(rows, n_input_scenarios=3,
                             drop_counts={"unparseable_rule": 1})
    assert summary["n_hierarchical_scenarios"] == 0
    assert summary["status"] == "exhausted"


@pytest.mark.parametrize("n_rows, n_input, accounted, expected", [
    (2, 3, 3, True),      # attempted and explained
    (0, 3, 3, False),     # A dead backend explains everything and scored nothing
    (5, 3, 5, False),     # more rows than inputs -- a different population
    (2, 3, 2, False),     # a scenario is unaccounted for; there is more to try
    (2, None, 2, False),  # no expected count to compare against
    (2, 0, 2, False),
])
def test_the_exhausted_predicate_directly(n_rows, n_input, accounted, expected):
    assert _is_exhausted(n_rows, n_input, accounted) is expected


def _flip_row(full_set, rule_judgments, rule_ids=None):
    rule_ids = rule_ids or [f"r{i}" for i in range(len(rule_judgments))]
    return {
        "full_set_judgment": full_set,
        "rules": [
            {"rule_id": rid, "judgment": judgment}
            for rid, judgment in zip(rule_ids, rule_judgments)
        ],
    }


def test_transitions_join_rules_by_id_not_position():
    base = {"h1": _flip_row("no", ["yes", "no"], ["a", "b"])}
    tuned = {"h1": {
        "full_set_judgment": "yes",
        "rules": [
            {"rule_id": "b", "judgment": "no"},
            {"rule_id": "a", "judgment": "yes"},
        ],
    }}
    result = judgment_transitions(base, tuned)
    assert result["n_common"] == 1
    assert result["summaries"]["per_rule"]["unchanged"] == 2
    assert result["per_rule_flips"] == {}
    assert result["anchors"]["yes"]["yes"] == 1
    assert result["full_set"]["no"]["yes"] == 1


def test_transitions_count_flips_and_rank_rules():
    base = {
        "h1": _flip_row("no", ["yes", "no", "unsure"]),
        "h2": _flip_row("no", ["no", "no", "no"]),
        "only_base": _flip_row("no", ["no"]),
    }
    tuned = {
        "h1": _flip_row("yes", ["no", "no", "no"]),   # r0 yes->no, r2 unsure->no
        "h2": _flip_row("no", ["yes", "no", "no"]),   # r0 no->yes
        "only_tuned": _flip_row("no", ["no"]),
    }
    result = judgment_transitions(base, tuned)
    assert result["n_common"] == 2
    assert result["per_rule"]["yes"]["no"] == 1
    assert result["per_rule"]["unsure"]["no"] == 1
    assert result["per_rule"]["no"]["yes"] == 1
    assert result["per_rule"]["no"]["no"] == 3
    # r0 flipped in both scenarios; r2 once; ties broken by rule_id.
    assert list(result["per_rule_flips"].items()) == [("r0", 2), ("r2", 1)]
    # Anchors: h1 yes->no, h2 no->yes.
    assert result["anchors"]["yes"]["no"] == 1
    assert result["anchors"]["no"]["yes"] == 1
    assert result["summaries"]["anchors"]["unchanged_fraction"] == 0.0


def test_transitions_refuse_duplicate_rule_ids():
    base = {"h1": _flip_row("no", ["yes", "no"], ["a", "a"])}
    tuned = {"h1": _flip_row("no", ["yes", "no"], ["a", "b"])}
    with pytest.raises(ScoringError, match="duplicate rule_id"):
        judgment_transitions(base, tuned)


def test_transitions_refuse_mismatched_rule_sets():
    base = {"h1": _flip_row("no", ["yes"], ["a"])}
    tuned = {"h1": _flip_row("no", ["yes"], ["b"])}
    with pytest.raises(ScoringError, match="different rule sets"):
        judgment_transitions(base, tuned)


def test_transition_summary_ignores_keys_outside_the_label_set():
    matrix = {b: {t: 0 for t in TRANSITION_LABELS} for b in TRANSITION_LABELS}
    matrix["yes"]["yes"] = 10
    clean = transition_summary(matrix)
    matrix["yes"]["abstain"] = 1_000
    assert transition_summary(matrix) == clean


def test_transition_summary_handles_the_empty_matrix():
    empty = {b: {t: 0 for t in TRANSITION_LABELS} for b in TRANSITION_LABELS}
    assert transition_summary(empty) == {
        "n": 0, "unchanged": 0, "unchanged_fraction": 0.0,
        "chance_agreement": 0.0, "kappa": 0.0}


def test_kappa_corrects_for_the_base_rate():
    """A degenerate tuned run that answers "yes" everywhere looks stable on the raw fraction and earns kappa 0."""
    degenerate = {b: {t: 0 for t in TRANSITION_LABELS} for b in TRANSITION_LABELS}
    degenerate["yes"]["yes"] = 93
    degenerate["no"]["yes"] = 7
    summary = transition_summary(degenerate)
    assert summary["unchanged_fraction"] == 0.93
    assert summary["chance_agreement"] == 0.93  # the floor equals p_o here
    assert summary["kappa"] == 0.0
    diagonal = {b: {t: 0 for t in TRANSITION_LABELS} for b in TRANSITION_LABELS}
    diagonal["yes"]["yes"] = 50
    diagonal["no"]["no"] = 50
    assert transition_summary(diagonal)["kappa"] == 1.0
    saturated = {b: {t: 0 for t in TRANSITION_LABELS} for b in TRANSITION_LABELS}
    saturated["yes"]["yes"] = 10
    assert transition_summary(saturated)["kappa"] == 0.0


def test_transitions_record_the_per_arm_denominators():
    base = {
        "h1": _flip_row("no", ["yes"]),
        "only_base": _flip_row("no", ["no"]),
    }
    tuned = {"h1": _flip_row("no", ["yes"])}
    result = judgment_transitions(base, tuned)
    assert (result["n_common"], result["n_base"], result["n_tuned"]) == (1, 2, 1)


def test_transitions_refuse_disjoint_runs():
    with pytest.raises(ScoringError, match="no common scenario"):
        judgment_transitions(
            {"a": _flip_row("no", ["no"])}, {"b": _flip_row("no", ["no"])})


def test_transitions_refuse_a_row_with_no_rules():
    base = {"h1": _flip_row("no", [])}
    tuned = {"h1": _flip_row("no", [])}
    with pytest.raises(ScoringError, match="no rule judgments"):
        judgment_transitions(base, tuned)


def test_transitions_refuse_raw_judge_text_as_a_full_set_verdict():
    base = {"h1": _flip_row("Yes.", ["yes"])}
    tuned = {"h1": _flip_row("yes", ["yes"])}
    with pytest.raises(ScoringError, match="unparseable full-set judgment"):
        judgment_transitions(base, tuned)
