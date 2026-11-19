from __future__ import annotations

import pytest

pytest.importorskip("statsmodels")

from ruleforge.evaluate.multiplicity import (
    ALPHA,
    adjust_for_multiplicity,
)
from ruleforge.experiments.labels import NAIVE_SLOT, STYLE_LABELS, rule_label_number
from ruleforge.experiments.prompt_ablation.analysis import TABLE_CONTRASTS
from ruleforge.prompts.styles import ENCOURAGED, FORCED


def test_the_encouraged_to_forced_contrast_is_included():
    """It is one of only two significant rows on the published runs."""
    assert (ENCOURAGED, FORCED) in TABLE_CONTRASTS


def _table_with_p(values):
    return [
        {
            "label": "M",
            "baseline": NAIVE_SLOT,
            "treatment": ENCOURAGED,
            "p_value": p,
            "significant": p < ALPHA,
        }
        for p in values
    ]


def test_correction_raises_every_p_value():
    adjusted = adjust_for_multiplicity(_table_with_p([0.001, 0.02, 0.30, 0.80]))
    for row in adjusted:
        assert row["p_adjusted"] >= row["p_value"]
        assert row["adjustment_method"] == "holm"


def test_the_default_correction_is_holm():
    values = [0.01, 0.012, 0.014, 0.6]
    assert adjust_for_multiplicity(_table_with_p(values)) == adjust_for_multiplicity(_table_with_p(values), method="holm")
    bh = adjust_for_multiplicity(_table_with_p(values), method="fdr_bh")
    assert [r["p_adjusted"] for r in bh] != [r["p_adjusted"] for r in adjust_for_multiplicity(_table_with_p(values))]


def test_correction_can_demote_a_marginal_result():
    adjusted = adjust_for_multiplicity(_table_with_p([0.04] + [0.6] * 8))
    assert adjusted[0]["significant"] is True
    assert adjusted[0]["significant_adjusted"] is False


def test_a_strong_result_survives_correction():
    adjusted = adjust_for_multiplicity(_table_with_p([1e-6] + [0.6] * 8))
    assert adjusted[0]["significant_adjusted"] is True


def test_bonferroni_is_stricter_than_benjamini_hochberg():
    values = [0.004, 0.005, 0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    bh = adjust_for_multiplicity(_table_with_p(values), method="fdr_bh")
    bonferroni = adjust_for_multiplicity(_table_with_p(values), method="bonferroni")
    assert sum(r["significant_adjusted"] for r in bonferroni) <= sum(
        r["significant_adjusted"] for r in bh
    )


def test_correcting_an_empty_table_is_not_an_error():
    assert adjust_for_multiplicity([]) == []


def test_correction_does_not_mutate_the_input():
    original = _table_with_p([0.01, 0.5])
    adjust_for_multiplicity(original)
    assert all("p_adjusted" not in row for row in original)


def test_a_rule_label_parses_with_or_without_whitespace():
    assert rule_label_number("Rule 5") == 5
    assert rule_label_number("Rule5") == 5
    assert rule_label_number("rule5") == 5
    assert rule_label_number("Meta-Rule 12") == 12
    assert rule_label_number("Meta-Rule12") == 12


def test_a_label_that_names_no_rule_does_not_parse():
    assert rule_label_number("Meta-Rule") is None
    assert rule_label_number("19") is None
    assert rule_label_number("X05") is None
    assert rule_label_number("Rule 5 and Rule 6") is None
    assert rule_label_number("not a rule") is None


def test_the_prompt_table_styles_are_the_paper_s_three_labelled_plainly_and_the_others_by_family():
    from ruleforge.experiments.labels import PAPER_STYLE_LABELS, SLOTS
    from ruleforge.experiments.prompt_ablation.analysis import SLOT_STYLES
    from ruleforge.prompts.styles import VALID_REASONING_PROMPT_STYLES

    assert SLOTS == (NAIVE_SLOT, ENCOURAGED, FORCED) == ("naive", "encouraged", "forced")
    assert SLOT_STYLES == {"naive": "naive-rules-above", "encouraged": "encouraged", "forced": "forced"}
    assert STYLE_LABELS == {"naive": "naive", "encouraged": "encouraged", "forced": "forced",
                            "forced-grpo": "forced (GRPO)", "naive-rules-above": "naive (rules above)",
                            "naive-hierarchy": "naive (hierarchy shown)"}
    assert set(STYLE_LABELS) == set(VALID_REASONING_PROMPT_STYLES)
    assert {slot: PAPER_STYLE_LABELS[slot] for slot in SLOTS} == {slot: slot for slot in SLOTS}


def test_the_label_tuple_has_one_definition():
    from ruleforge.judges import fake
    from ruleforge.prompts import parse, styles
    from ruleforge.tasks import spec

    assert parse.LABELS == ("yes", "no", "unsure")
    assert fake.LABELS is parse.LABELS and styles.LABELS is parse.LABELS and spec.LABELS is parse.LABELS
