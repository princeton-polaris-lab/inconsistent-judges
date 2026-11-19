from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os

import pytest

from ruleforge.evaluate.multiplicity import ALPHA
from ruleforge.experiments.prompt_ablation.analysis import (
    FORCED_B,
    TABLE_CONTRASTS,
)
from ruleforge.experiments.labels import NAIVE_SLOT, display_order
from ruleforge.prompts.styles import ENCOURAGED, FORCED
from ruleforge.experiments.paper_outputs import SLOT_NAMES
from ruleforge.core.paths import resolve
from ruleforge.experiments import table as experiment_table

GOLDEN = "results/prompt_ablation/prompt_ablation.json"
ARM_MAP = "configs/paper/prompt_ablation_arms.yaml"
ARM_MAP_SHA256 = "31938b3bc18cc6181e89b3dc9abbc152e091a791dc3f79f5e6ed3b7fd9a3ec2a"
ARM_MAP_VARIABLES = {"pop": "pop-a22af097"}

HAIKU = "claude-haiku-5-5"
GLIMMER = "muse-glimmer-30b"
#: The paper's order: the open judges, Muse-Glimmer-30B with them, then the provider judges.
MODELS = ["Olmo3-32B", "gemma4-31b", "gpt-oss-120b", GLIMMER, "gpt-5.6-luna", HAIKU]
MODEL_NAMES = {"Olmo3-32B": "Olmo-3-32B-Think", "gemma4-31b": "gemma-4-31B-it", "gpt-oss-120b": "gpt-oss-120b",
               GLIMMER: "Muse-Glimmer-30B", "gpt-5.6-luna": "gpt-5.6-luna", HAIKU: HAIKU}
#: The provider judges, each with no inference stack recorded: (model_type, retry scope).
PROVIDERS = {"gpt-5.6-luna": ("openai_batch", "scenario_bundle"), HAIKU: ("anthropic_batch", "component")}
#: Each judge's retry scope: the bundle's but for the two judged by component.
RETRY_SCOPES = {label: "component" if label in (GLIMMER, HAIKU) else "scenario_bundle" for label in MODELS}
LABEL_RULE = "lenient_last_line"
N_SCENARIOS = 500
N_DRAWS = 5
DRAWN_BARS = [NAIVE_SLOT, ENCOURAGED, FORCED, FORCED_B]
#: Each slot's prompt style: the published naive prompt, the paper's encouraged and forced ones.
SLOT_STYLES = {NAIVE_SLOT: "naive-rules-above", ENCOURAGED: "encouraged", FORCED: "forced"}

SAMPLING = "artifact://sampling"
PUBLISHED = f"{SAMPLING}/final-2026-09-pop-a22af097-rescored-{LABEL_RULE}"
PUBLISHED_LUNA = f"{SAMPLING}/final-2026-09-luna-pop-a22af097-rescored-{LABEL_RULE}"
GEMMA_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-pop-a22af097-rescored-{LABEL_RULE}"
OPEN_REVISED = f"{SAMPLING}/prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-{LABEL_RULE}"
LUNA_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-{LABEL_RULE}"
HAIKU_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-haiku-sum"
GLIMMER_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-glimmer-pop-a22af097"
#: The arm map's seven roots, in its order: campaign id, estimates SHA-256, the conditions the table reads there.
CAMPAIGNS = {
    PUBLISHED: ("final-2026-09", "e469d5ca0dffac5aef6f43414e6146dedb0ad8c4cead05969d46e1f89379dab2",
                ["Olmo-3-32B-Think/naive-rules-above", "gemma-4-31B-it/naive-rules-above", "gpt-oss-120b/naive-rules-above"]),
    PUBLISHED_LUNA: ("final-2026-09-luna", "9c5793739ccd2bb8e5c68d394e4a55d0091ce4325c2fe351656b8ea58bd431f7",
                     ["gpt-5.6-luna/naive-rules-above"]),
    GEMMA_REVISED: ("prompt-ablation-v2-2026-10", "89750ce3b41584bf9c7f82147930cd2b28628a8dc5bb46c125677749e27928ed",
               ["gemma-4-31B-it/encouraged", "gemma-4-31B-it/forced"]),
    OPEN_REVISED: ("prompt-ablation-v2-open-2026-10", "2c5c6e378ce858816331338128534542e368ff6a9278cf61a446681b85508e1a",
              ["Olmo-3-32B-Think/encouraged", "Olmo-3-32B-Think/forced", "gpt-oss-120b/encouraged",
               "gpt-oss-120b/forced"]),
    LUNA_REVISED: ("prompt-ablation-v2-2026-10-luna", "5816b7933a3863584713943a40e1cdb085a781616c862b4a29cb04c5fc197035",
              ["gpt-5.6-luna/encouraged", "gpt-5.6-luna/forced"]),
    HAIKU_REVISED: ("prompt-ablation-v2-2026-10-haiku-sum",
               "345e4dbeb5a9b9e93f371cbf6b2d8e1855270ec39d7a6d291a73df444e966696",
               ["claude-haiku-5-5/encouraged", "claude-haiku-5-5/forced", "claude-haiku-5-5/naive-rules-above"]),
    GLIMMER_REVISED: ("prompt-ablation-v2-2026-10-glimmer",
                 "f4f93185f95cc206cd2f921020a9afba7ec110a52d41bead66a23a41aaa1a304",
                 ["Muse-Glimmer-30B/encouraged", "Muse-Glimmer-30B/forced", "Muse-Glimmer-30B/naive-rules-above"]),
}
ARM_ROOTS = {
    "Olmo3-32B": {NAIVE_SLOT: PUBLISHED, ENCOURAGED: OPEN_REVISED, FORCED: OPEN_REVISED},
    "gemma4-31b": {NAIVE_SLOT: PUBLISHED, ENCOURAGED: GEMMA_REVISED, FORCED: GEMMA_REVISED},
    "gpt-oss-120b": {NAIVE_SLOT: PUBLISHED, ENCOURAGED: OPEN_REVISED, FORCED: OPEN_REVISED},
    GLIMMER: {NAIVE_SLOT: GLIMMER_REVISED, ENCOURAGED: GLIMMER_REVISED, FORCED: GLIMMER_REVISED},
    "gpt-5.6-luna": {NAIVE_SLOT: PUBLISHED_LUNA, ENCOURAGED: LUNA_REVISED, FORCED: LUNA_REVISED},
    HAIKU: {NAIVE_SLOT: HAIKU_REVISED, ENCOURAGED: HAIKU_REVISED, FORCED: HAIKU_REVISED},
}
#: A retired style's condition is listed by its recorded id.
IGNORED = {PUBLISHED_LUNA: ["gpt-5.6-luna/forced-grpo", "gpt-5.6-luna/good"]}
SCENARIO_MANIFEST_SHA256 = "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"
OLD_STACK = "vLLM 0.19.0, torch 2.10.0"
NEW_STACK = "vLLM 0.28.0+cu129, torch 2.13.0+cu129, transformers 5.17.0"

PAPER = {
    ("Olmo3-32B", NAIVE_SLOT, ENCOURAGED): (
        0.750800, 0.724400, 0.026400, 0.006400, 0.046000,
        1.017990e-02, 5.089949e-02, 1.308844e-02,
    ),
    ("Olmo3-32B", NAIVE_SLOT, FORCED): (
        0.750800, 0.656400, 0.094400, 0.071600, 0.116400,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("Olmo3-32B", ENCOURAGED, FORCED): (
        0.724400, 0.656400, 0.068000, 0.047600, 0.088400,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("gemma4-31b", NAIVE_SLOT, ENCOURAGED): (
        0.720400, 0.663200, 0.057200, 0.039600, 0.074800,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("gemma4-31b", NAIVE_SLOT, FORCED): (
        0.720400, 0.719600, 0.000800, -0.016000, 0.017600,
        9.621604e-01, 1.000000e+00, 9.621604e-01,
    ),
    ("gemma4-31b", ENCOURAGED, FORCED): (
        0.663200, 0.719600, -0.056400, -0.075610, -0.037200,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("gpt-oss-120b", NAIVE_SLOT, ENCOURAGED): (
        0.627600, 0.512000, 0.115600, 0.094000, 0.137200,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("gpt-oss-120b", NAIVE_SLOT, FORCED): (
        0.627600, 0.472400, 0.155200, 0.132000, 0.178400,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("gpt-oss-120b", ENCOURAGED, FORCED): (
        0.512000, 0.472400, 0.039600, 0.020400, 0.058800,
        3.999960e-05, 3.599964e-04, 7.199928e-05,
    ),
    (GLIMMER, NAIVE_SLOT, ENCOURAGED): (
        0.442400, 0.382400, 0.060000, 0.039200, 0.080400,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    (GLIMMER, NAIVE_SLOT, FORCED): (
        0.442400, 0.502400, -0.060000, -0.083600, -0.037600,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    (GLIMMER, ENCOURAGED, FORCED): (
        0.382400, 0.502400, -0.120000, -0.141600, -0.098800,
        9.999900e-06, 1.799982e-04, 1.999980e-05,
    ),
    ("gpt-5.6-luna", NAIVE_SLOT, ENCOURAGED): (
        0.436000, 0.412400, 0.023600, 0.002400, 0.046000,
        3.756962e-02, 1.502785e-01, 4.508355e-02,
    ),
    ("gpt-5.6-luna", NAIVE_SLOT, FORCED): (
        0.436000, 0.400400, 0.035600, 0.011600, 0.060400,
        4.969950e-03, 2.981970e-02, 6.881470e-03,
    ),
    ("gpt-5.6-luna", ENCOURAGED, FORCED): (
        0.412400, 0.400400, 0.012000, -0.004800, 0.028800,
        1.729383e-01, 5.188148e-01, 1.945556e-01,
    ),
    (HAIKU, NAIVE_SLOT, ENCOURAGED): (
        0.585600, 0.584400, 0.001200, -0.016000, 0.018800,
        9.287107e-01, 1.000000e+00, 9.621604e-01,
    ),
    (HAIKU, NAIVE_SLOT, FORCED): (
        0.585600, 0.615200, -0.029600, -0.046800, -0.012400,
        9.299907e-04, 6.509935e-03, 1.394986e-03,
    ),
    (HAIKU, ENCOURAGED, FORCED): (
        0.584400, 0.615200, -0.030800, -0.047600, -0.014800,
        2.999970e-04, 2.399976e-03, 4.909042e-04,
    ),
}

SAME_CALL_RATE = {"Olmo3-32B": 0.0044, "gemma4-31b": 0.0000, "gpt-oss-120b": 0.0004, GLIMMER: 0.0008,
                 "gpt-5.6-luna": 0.0000, HAIKU: 0.0340}

#: The rows no family stars.
NOT_SIGNIFICANT = [("gemma4-31b", NAIVE_SLOT, FORCED), ("gpt-5.6-luna", ENCOURAGED, FORCED), (HAIKU, NAIVE_SLOT, ENCOURAGED)]
#: The rows BH and raw alpha star and Holm does not (p_holm 0.0509 and 0.150 over the eighteen).
BH_NOT_HOLM = [("Olmo3-32B", NAIVE_SLOT, ENCOURAGED), ("gpt-5.6-luna", NAIVE_SLOT, ENCOURAGED)]
SIGNIFICANT = sorted(key for key in PAPER if key not in NOT_SIGNIFICANT)
HOLM_SIGNIFICANT = sorted(key for key in SIGNIFICANT if key not in BH_NOT_HOLM)
CROSS_STACK = sorted((label, NAIVE_SLOT, treatment) for label in ("Olmo3-32B", "gemma4-31b", "gpt-oss-120b")
                     for treatment in (ENCOURAGED, FORCED))


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table(payload):
    return payload["table"]


@pytest.fixture(scope="module")
def by_key(table):
    return {(r["label"], r["baseline"], r["treatment"]): r for r in table}


@pytest.fixture(scope="module")
def bar_block(payload):
    return payload["bars"]


@pytest.fixture(scope="module")
def driver(repo_root):
    pytest.importorskip("statsmodels", reason="needs the viz group")
    module = importlib.import_module("ruleforge.experiments.prompt_ablation.build")
    return module


def test_every_row_s_arms_are_their_slots_template_styles(payload, table):
    assert "family" not in payload["provenance"]
    for row in table:
        assert row["baseline_style"] == SLOT_STYLES[row["baseline"]] == SLOT_STYLES[SLOT_NAMES[row["baseline_label"]]]
        assert row["treatment_style"] == SLOT_STYLES[row["treatment"]] == SLOT_STYLES[SLOT_NAMES[row["treatment_label"]]]


def test_every_paper_row_reproduces(by_key):
    for key, expected in PAPER.items():
        row = by_key[key]
        rate_a, rate_b, difference, ci_low, ci_high, p_value, p_holm, p_bh = expected
        assert row["n_scenarios"] == N_SCENARIOS, key
        assert row["n_decoder_draws"] == N_DRAWS, key
        assert row["baseline_rate"] == pytest.approx(rate_a, abs=1e-6), key
        assert row["treatment_rate"] == pytest.approx(rate_b, abs=1e-6), key
        # baseline minus treatment: its SIGN carries whether the treatment helped.
        assert row["difference"] == pytest.approx(difference, abs=1e-6), key
        assert row["difference"] == pytest.approx(row["baseline_rate"] - row["treatment_rate"]), key
        assert row["ci_low"] == pytest.approx(ci_low, abs=1e-6), key
        assert row["ci_high"] == pytest.approx(ci_high, abs=1e-6), key
        assert row["p_value"] == pytest.approx(p_value, rel=1e-6), key
        assert row["p_holm"] == pytest.approx(p_holm, rel=1e-6), key
        assert row["p_bh"] == pytest.approx(p_bh, rel=1e-6), key
        assert row["improvement"] is (difference > 0), key
        assert row["significant"] is row["significant_holm"], key
        assert row["n_permutations"] == 100_000, key
        assert row["test_method"] == "scenario_clustered_paired_randomization_v1", key
        assert row["interval_basis"] == "scenario_cluster_bootstrap", key


def test_the_table_covers_every_model_and_contrast_in_paper_order(table):
    assert len(table) == 18
    assert [(r["label"], r["baseline"], r["treatment"]) for r in table] == [
        (label, baseline, treatment) for label in display_order(MODELS) for baseline, treatment in TABLE_CONTRASTS
    ]
    assert [(r["baseline_label"], r["treatment_label"]) for r in table[:3]] == [
        ("naive", "encouraged"), ("naive", "forced"), ("encouraged", "forced")]


def test_thirteen_of_eighteen_rows_are_holm_significant_and_fifteen_under_bh_and_raw_alpha(by_key):
    def flagged(flag):
        return sorted(key for key, row in by_key.items() if row[flag])

    assert flagged("significant_holm") == flagged("significant") == HOLM_SIGNIFICANT
    assert flagged("significant_bh") == SIGNIFICANT
    assert sorted(key for key, row in by_key.items() if row["p_value"] < ALPHA) == SIGNIFICANT
    assert len(HOLM_SIGNIFICANT) == 13 and len(SIGNIFICANT) == 15
    for key in NOT_SIGNIFICANT:
        row = by_key[key]
        assert row["p_value"] > ALPHA, key
        assert row["ci_low"] < 0.0 < row["ci_high"], key


def test_both_families_recompute_over_all_eighteen_rows(table):
    multipletests = pytest.importorskip("statsmodels.stats.multitest", reason="needs the viz group").multipletests
    raw = [row["p_value"] for row in table]
    _, holm, _, _ = multipletests(raw, alpha=ALPHA, method="holm")
    _, bh, _, _ = multipletests(raw, alpha=ALPHA, method="fdr_bh")
    assert [row["p_holm"] for row in table] == pytest.approx(list(holm), rel=1e-9)
    assert [row["p_bh"] for row in table] == pytest.approx(list(bh), rel=1e-9)
    for row in table:
        assert row["adjustment_method"] == "holm"
        assert row["adjustment_family"] == "all_table_style_contrasts"
        assert row["adjustment_family_n"] == 18
        assert row["p_holm"] >= row["p_bh"] >= row["p_value"]
        assert row["significant_holm"] is (row["p_holm"] < ALPHA)
        assert row["significant_bh"] is (row["p_bh"] < ALPHA)


def test_each_row_names_its_arms_and_crosses_stacks_only_where_the_map_declares(by_key):
    for (label, baseline, treatment), row in by_key.items():
        model = MODEL_NAMES[label]
        assert row["model_name"] == model
        assert row["baseline_condition"] == f"{model}/{SLOT_STYLES[baseline]}"
        assert row["treatment_condition"] == f"{model}/{SLOT_STYLES[treatment]}"
        assert row["baseline_root"] == ARM_ROOTS[label][baseline]
        assert row["treatment_root"] == ARM_ROOTS[label][treatment]
        assert row["paired_within_root"] is (row["baseline_root"] == row["treatment_root"])
        if (label, baseline, treatment) in CROSS_STACK:
            record = row["cross_stack"]
            assert (record["declared"], record["baseline_stack"], record["treatment_stack"]) == (
                "naive", OLD_STACK, NEW_STACK)
            assert record["fields"] == ["torch", "vllm"]
            assert record["conflict"].startswith("vllm 0.19.0 / torch 2.10.0 against vllm 0.28.0+cu129")
        else:
            assert row["cross_stack"] is None
    assert sorted(key for key, row in by_key.items() if row["cross_stack"] is not None) == CROSS_STACK


def test_the_provenance_names_the_map_each_arm_and_the_declaration(payload):
    provenance = payload["provenance"]
    assert provenance["arm_map"] == {"name": "prompt_ablation_arms.yaml", "sha256": ARM_MAP_SHA256,
                                     "variables": ARM_MAP_VARIABLES, "version": 1}
    assert provenance["label_rules"] == {label: LABEL_RULE for label in MODELS}
    assert provenance["retry_scopes"] == RETRY_SCOPES
    assert "recomputed over every contrast in this table" in provenance["multiplicity"]
    for label in MODELS:
        model = MODEL_NAMES[label]
        for slot, arm in provenance["arms"][label].items():
            assert arm["condition_id"] == f"{model}/{SLOT_STYLES[slot]}" and arm["prompt_style"] == SLOT_STYLES[slot]
            assert arm["root"] == ARM_ROOTS[label][slot] and arm["label_rule"] == LABEL_RULE
            stack = arm["stack"]
            if label in PROVIDERS:
                assert (arm["model_type"], stack["kind"], stack["short"]) == (PROVIDERS[label][0], "provider",
                                                                              "provider")
                assert set(stack["dates"]) == {"first_request_block", "last_request_block"}
            else:
                assert (arm["model_type"], stack["kind"]) == ("local", "local")
                old = slot == NAIVE_SLOT and label != GLIMMER
                assert stack["short"] == (OLD_STACK if old else NEW_STACK), (label, slot)
    cross = provenance["cross_stack"]
    assert cross["declared"] == ["Olmo3-32B", "gemma4-31b", "gpt-oss-120b"]
    assert sorted(map(tuple, cross["contrasts"])) == CROSS_STACK
    replay = provenance["label_replay"]
    assert set(replay) == set(CAMPAIGNS)
    for root, source in ((PUBLISHED, "lenient"), (PUBLISHED_LUNA, "lenient"), (GEMMA_REVISED, "lenient"),
                         (OPEN_REVISED, "lenient_markup"), (LUNA_REVISED, "lenient_markup")):
        assert (replay[root]["label_rule"], replay[root]["source_label_rule"]) == (LABEL_RULE, source), root
        assert replay[root]["tool"] == "smrc.sampling.label_replay"
    assert replay[HAIKU_REVISED] is None and replay[GLIMMER_REVISED] is None
    assert provenance["ignored_conditions"] == IGNORED


def test_no_revised_naive_condition_is_an_arm_and_the_golden_has_no_block_beside_the_table(payload):
    assert set(payload) == {"schema_version", "table", "rates", "bars", "diagnostics", "provenance"}
    named = {arm["condition_id"] for arms in payload["provenance"]["arms"].values() for arm in arms.values()}
    listed = {condition for conditions in payload["provenance"]["ignored_conditions"].values()
              for condition in conditions}
    for conditions in (named, set(payload["diagnostics"]["conditions"]), listed):
        assert not [condition for condition in conditions if condition.endswith("/naive")]
    assert "robustness" not in payload["provenance"]["definition"]


def test_every_bar_rests_on_the_full_design(bar_block):
    assert set(bar_block) == set(MODELS)
    for label, entry in bar_block.items():
        assert list(entry) == ["per_bar"], label
        bars = entry["per_bar"]
        assert set(bars) == set(DRAWN_BARS), label
        for bar, stats in bars.items():
            assert stats["n"] == N_SCENARIOS, (label, bar)
            assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"], (label, bar)
            assert stats["degenerate"] is (stats["ci_low"] == stats["ci_high"]), (label, bar)


def test_the_degenerate_intervals_are_the_two_zero_same_call_bars(bar_block):
    degenerate = sorted((label, bar) for label, entry in bar_block.items()
                        for bar, stats in entry["per_bar"].items() if stats["degenerate"])
    assert degenerate == [("gemma4-31b", FORCED_B), ("gpt-5.6-luna", FORCED_B)]
    for label, _ in degenerate:
        assert bar_block[label]["per_bar"][FORCED_B]["mean"] == 0.0


def test_same_call_rates_reproduce(bar_block, payload):
    for label, expected in SAME_CALL_RATE.items():
        bars = bar_block[label]["per_bar"]
        assert bars[FORCED_B]["mean"] == pytest.approx(expected, abs=1e-4), label
        diagnostic = payload["diagnostics"]["separate_vs_same_call"][label]
        assert diagnostic["rate_a"] == pytest.approx(bars[FORCED]["mean"]), label
        assert diagnostic["rate_b"] == pytest.approx(bars[FORCED_B]["mean"]), label
        assert diagnostic["n_permutations"] == 100_000


def test_the_figure_and_the_table_are_one_set_of_numbers(payload, bar_block, by_key):
    assert set(payload["rates"]) == set(MODELS)
    for label in MODELS:
        rates = payload["rates"][label]
        assert rates["n_common"] == N_SCENARIOS, label
        assert set(rates["rates"]) == {NAIVE_SLOT, ENCOURAGED, FORCED}, label
        for style, rate in rates["rates"].items():
            assert bar_block[label]["per_bar"][style]["mean"] == pytest.approx(rate), (label, style)
        for baseline, treatment in TABLE_CONTRASTS:
            row = by_key[(label, baseline, treatment)]
            assert row["baseline_rate"] == pytest.approx(rates["rates"][baseline]), (label, baseline)
            assert row["treatment_rate"] == pytest.approx(rates["rates"][treatment]), (label, treatment)


def test_the_golden_names_its_campaigns_and_design(payload):
    provenance = payload["provenance"]
    assert payload["schema_version"] == provenance["schema_version"] == 2
    assert provenance["source"] == "stochastic_campaign"
    assert provenance["complete"] is True
    assert provenance["models"] == display_order(MODELS)
    assert provenance["model_labels"] == {model: label for label, model in MODEL_NAMES.items()}
    assert provenance["alpha"] == ALPHA
    assert provenance["metric"] == "flat"
    assert provenance["scenario_manifest_sha256"] == SCENARIO_MANIFEST_SHA256
    assert provenance["n_scenarios"] == N_SCENARIOS
    assert provenance["n_decoder_draws"] == N_DRAWS
    assert provenance["temperature"] == 1.0
    assert provenance["truncation"] == "model_default"
    assert provenance["forced_json_requirement"] == "complete_rule_map"
    assert provenance["inference_method"] == "hierarchical_bootstrap_v2"
    assert provenance["bootstrap"] == {"n_resamples": 10_000, "ci_level": 0.95, "seed": 16}
    assert provenance["randomization_test"] == {"n_permutations": 100_000, "seed": 16}
    assert provenance["bar_settings"] == {
        "population": "per_bar",
        "interval_basis": "scenario_cluster_bootstrap",
        "inference_method": "hierarchical_bootstrap_v2",
        "bootstrap": provenance["bootstrap"],
        "n_scenarios": N_SCENARIOS,
        "n_decoder_draws": N_DRAWS,
        "bars": DRAWN_BARS,
    }
    campaigns = {c["root"]: c for c in provenance["campaigns"]}
    assert list(campaigns) == list(CAMPAIGNS)
    for root, (campaign_id, sha256, conditions) in CAMPAIGNS.items():
        campaign = campaigns[root]
        assert campaign["campaign_id"] == campaign_id
        assert campaign["estimates"] == f"{root}/aggregate/estimates.json"
        assert campaign["estimates_sha256"] == sha256
        assert campaign["conditions"] == conditions
        assert campaign["state"] == "complete"
        assert campaign["max_semantic_attempts"] == 20 and "amendments" not in campaign
        assert campaign.get("retry_scope") == ("component" if root in (HAIKU_REVISED, GLIMMER_REVISED) else None), root
    assert payload["diagnostics"]["excluded_conditions"] == {}
    assert set(payload["diagnostics"]["conditions"]) == {
        f"{MODEL_NAMES[label]}/{style}" for label in MODELS for style in SLOT_STYLES.values()}


def test_no_golden_value_is_a_bare_nan(repo_root):
    text = (repo_root / GOLDEN).read_text(encoding="utf-8")
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text, token


def test_the_table_names_the_arm_map_s_roots():
    (row,) = [row for row in experiment_table.TABLE if row.name == "prompt-ablation"]
    assert list(row.runs) == list(CAMPAIGNS)


def test_the_committed_arm_map_is_the_one_the_golden_names(repo_root):
    assert hashlib.sha256((repo_root / ARM_MAP).read_bytes()).hexdigest() == ARM_MAP_SHA256


@pytest.fixture
def renderings(driver, payload, table, tmp_path):
    """The CSV and LaTeX table the producer writes from the golden at its alpha and interval level."""
    alpha = payload["provenance"]["alpha"]
    driver.write_csv(table, tmp_path / "table.csv", alpha=alpha)
    driver.write_latex(table, tmp_path / "table.tex", alpha=alpha,
                       ci_level=payload["provenance"]["bootstrap"]["ci_level"])
    return {"csv": tmp_path / "table.csv", "tex": tmp_path / "table.tex"}


def test_the_csv_is_the_goldens_table(renderings, table, by_key):
    with renderings["csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "model", "prompt_a", "prompt_b", "n_scenarios", "n_draws",
            "rate_a", "rate_b", "difference", "diff_ci_low", "diff_ci_high",
            "randomization_p", "p_holm", "significant_holm", "p_bh", "significant_bh",
            "adjustment_method", "adjustment_family", "adjustment_family_n",
            "metric", "inference_method", "style_a", "style_b", "cross_stack",
        ]
        rows = list(reader)
    keys = [(row["model"], row["prompt_a"], row["prompt_b"]) for row in rows]
    assert keys == [(r["label"], r["baseline"], r["treatment"]) for r in table]
    for key, row in zip(keys, rows):
        golden = by_key[key]
        assert float(row["rate_a"]) == pytest.approx(golden["baseline_rate"], abs=5e-7)
        assert float(row["rate_b"]) == pytest.approx(golden["treatment_rate"], abs=5e-7)
        assert float(row["difference"]) == pytest.approx(golden["difference"], abs=5e-7)
        assert float(row["p_holm"]) == pytest.approx(golden["p_holm"], rel=1e-6)
        assert row["significant_holm"] == str(golden["significant_holm"]).lower()
        assert (row["style_a"], row["style_b"]) == (golden["baseline"], golden["treatment"])
        assert row["cross_stack"] == str(key in CROSS_STACK).lower()
    assert sorted(key for key, row in zip(keys, rows) if row["significant_holm"] == "true") == HOLM_SIGNIFICANT
    assert sorted(key for key, row in zip(keys, rows) if row["significant_bh"] == "true") == SIGNIFICANT


def test_the_latex_names_the_prompts_plainly_and_daggers_the_cross_stack_rows(renderings, driver, table):
    """The "Plain names": naive / encouraged / forced, no version label, no list of prompts in the note."""
    latex = renderings["tex"].read_text(encoding="utf-8")
    assert "(v2)" not in latex and "_v2" not in latex and "Prompts:" not in latex
    assert latex.count(r"$^{\dagger}$") == 7  # six rows and the note
    assert latex.count(r"naive $\rightarrow$ encouraged$^{\dagger}$") == 3
    assert latex.count(r"naive $\rightarrow$ forced$^{\dagger}$") == 3
    assert latex.count(r"encouraged $\rightarrow$ forced &") == 6
    assert r"Claude Haiku 5.5 & naive $\rightarrow$ encouraged &" in latex
    assert r"Muse-Glimmer-30B & naive $\rightarrow$ encouraged &" in latex
    assert (f"different inference stacks ({OLD_STACK} against {NEW_STACK}), an exception the table's arm map "
            "declares for the naive arm.") in latex
    assert "$N=500$" in latex and "$D=5$" in latex and "all 18 contrasts" in latex
    assert r"\textbf{\textless 0.001}" in latex and "0.000 " not in latex
    for row in table:
        rendered_p = driver._format_latex_p(row["p_holm"], significant=bool(row["significant_holm"]))
        assert rendered_p in latex, (row["label"], row["baseline"], row["treatment"])


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_arm_map(driver, repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    estimates = {root: resolve(f"{root}/aggregate/estimates.json") for root in CAMPAIGNS}
    if not all(path.is_file() for path in estimates.values()):
        pytest.skip("the prompt-ablation roots are not on this machine")
    for root, path in estimates.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == CAMPAIGNS[root][1], root
    rebuilt = {name: tmp_path / name for name in ("table.json", "table.csv", "table.tex")}
    variables = [word for name, value in ARM_MAP_VARIABLES.items() for word in ("--arm-map-var", f"{name}={value}")]
    assert driver.main(["--arm-map", str(repo_root / ARM_MAP), *variables, "--golden", str(rebuilt["table.json"]),
                        "--csv", str(rebuilt["table.csv"]), "--latex", str(rebuilt["table.tex"])]) == 0
    assert rebuilt["table.json"].read_bytes() == (repo_root / GOLDEN).read_bytes()


def test_the_golden_flags_only_olmo_s_naive_to_encouraged_row_inside_monte_carlo_noise(table):
    assert [(r["label"], r["baseline"], r["treatment"]) for r in table if r["holm_mc_margin"]] == [
        ("Olmo3-32B", NAIVE_SLOT, ENCOURAGED)]
    assert all(type(r["holm_mc_margin"]) is bool for r in table)
    assert not {"p_adjusted", "significant_adjusted"} & {key for row in table for key in row}
