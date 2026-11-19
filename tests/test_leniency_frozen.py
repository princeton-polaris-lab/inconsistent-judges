from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import os

import pytest

from ruleforge.experiments.labels import display_order
from ruleforge.experiments.paper_outputs import SLOT_NAMES
from ruleforge.experiments.leniency.analysis import CSV_COLUMNS
from ruleforge.experiments.leniency.schema import AGREEMENT_METHOD, DATA_VERSION, LABELS, TRANSITION_METHOD, direction
from ruleforge.experiments.leniency.golden import validate_payload
from ruleforge.experiments.leniency.figure import (
    direction_significance_counts,
    main_text_rule_order,
    rule_order_by_mean_delta,
    style_note,
)
from ruleforge.core.paths import resolve
from ruleforge.experiments import table

pytestmark = pytest.mark.frozen

GOLDEN = "results/leniency/leniency.json"
RULES_FILE = "data/rules/modelspec/openai_rules_all.jsonl"
PROMPT_STYLE = "forced"

HAIKU = "claude-haiku-5-5"
GLIMMER = "muse-glimmer-30b"
MODELS = ["Olmo3-32B", "gemma4-31b", "gpt-oss-120b", GLIMMER, "gpt-5.6-luna", HAIKU]
#: The four judges the published (frozen_500) golden held.
PUBLISHED_FOUR = ["Olmo3-32B", "gemma4-31b", "gpt-oss-120b", "gpt-5.6-luna"]
FIVE = [*PUBLISHED_FOUR, HAIKU]
N_SCENARIOS, N_DRAWS, N_RULES = 500, 5, 57
RULE_CELLS = N_SCENARIOS * N_DRAWS
MODEL_CELLS = RULE_CELLS * N_RULES

SAMPLING = "artifact://sampling"
LABEL_RULE = "lenient_last_line"
GEMMA_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-pop-a22af097-rescored-{LABEL_RULE}"
OPEN_REVISED = f"{SAMPLING}/prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-{LABEL_RULE}"
LUNA_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-{LABEL_RULE}"
HAIKU_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-haiku-sum"
GLIMMER_REVISED = f"{SAMPLING}/prompt-ablation-v2-2026-10-glimmer-pop-a22af097"
CAMPAIGNS = {
    GEMMA_REVISED: ("prompt-ablation-v2-2026-10", "89750ce3b41584bf9c7f82147930cd2b28628a8dc5bb46c125677749e27928ed",
               ["gemma-4-31B-it/forced"], None),
    OPEN_REVISED: ("prompt-ablation-v2-open-2026-10", "2c5c6e378ce858816331338128534542e368ff6a9278cf61a446681b85508e1a",
              ["Olmo-3-32B-Think/forced", "gpt-oss-120b/forced"], None),
    LUNA_REVISED: ("prompt-ablation-v2-2026-10-luna", "5816b7933a3863584713943a40e1cdb085a781616c862b4a29cb04c5fc197035",
              ["gpt-5.6-luna/forced"], None),
    HAIKU_REVISED: ("prompt-ablation-v2-2026-10-haiku-sum",
               "345e4dbeb5a9b9e93f371cbf6b2d8e1855270ec39d7a6d291a73df444e966696",
               ["claude-haiku-5-5/forced"], "component"),
    GLIMMER_REVISED: ("prompt-ablation-v2-2026-10-glimmer",
                 "f4f93185f95cc206cd2f921020a9afba7ec110a52d41bead66a23a41aaa1a304",
                 ["Muse-Glimmer-30B/forced"], "component"),
}
CONDITIONS = {
    "Olmo3-32B": ("prompt-ablation-v2-open-2026-10", "Olmo-3-32B-Think/forced"),
    "gemma4-31b": ("prompt-ablation-v2-2026-10", "gemma-4-31B-it/forced"),
    "gpt-oss-120b": ("prompt-ablation-v2-open-2026-10", "gpt-oss-120b/forced"),
    "gpt-5.6-luna": ("prompt-ablation-v2-2026-10-luna", "gpt-5.6-luna/forced"),
    HAIKU: ("prompt-ablation-v2-2026-10-haiku-sum", "claude-haiku-5-5/forced"),
    GLIMMER: ("prompt-ablation-v2-2026-10-glimmer", "Muse-Glimmer-30B/forced"),
}
SCENARIO_MANIFEST_SHA256 = "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"
RULES_SHA256 = "a99062f00a9ff616cba4d548e9cd59460dc4ee52fc46e3de4a4eb2579a260228"

SUMMARY = {
    "Olmo3-32B": (-11207, 0.1037, 0.0250, 52, 4, 1, 45, 0),
    "gemma4-31b": (-10586, 0.0873, 0.0130, 51, 0, 6, 41, 0),
    "gpt-oss-120b": (-10224, 0.1286, 0.0568, 48, 7, 2, 37, 0),
    "gpt-5.6-luna": (-7864, 0.1423, 0.0871, 52, 5, 0, 36, 4),
    HAIKU: (-8886, 0.0841, 0.0218, 53, 1, 3, 38, 0),
    GLIMMER: (-20984, 0.1972, 0.0500, 49, 6, 2, 42, 0),
}
POOLED_DELTA = {"Olmo3-32B": -0.0786, "gemma4-31b": -0.0743, "gpt-oss-120b": -0.0717, "gpt-5.6-luna": -0.0552,
                HAIKU: -0.0624, GLIMMER: -0.1473}
#: Every Holm-significant increase, by (model, rule): its yes_difference.
HOLM_INCREASES = {("gpt-5.6-luna", 15): 100, ("gpt-5.6-luna", 21): 146, ("gpt-5.6-luna", 23): 140,
                  ("gpt-5.6-luna", 35): 54}
EVERYWHERE_SIGNIFICANT_DECREASES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 16, 22, 24, 26, 34, 42, 43, 44, 45, 47, 49, 51, 52,
                                    55]
ALL_FIVE_SIGNIFICANT_DECREASES = [1, 2, 3, 4, 5, 6, 9, 10, 11, 16, 22, 24, 34, 44, 45, 49, 51, 52, 55]
ALL_SIX_SIGNIFICANT_DECREASES = ALL_FIVE_SIGNIFICANT_DECREASES
COMPACT_RULES = [16, 5, 45, 4, 3, 55, 6, 10, 49, 11, 21, 23, 15, 35]
#: The caption's counts over all 342 model-rule estimates, with the number of each within the Monte Carlo margin.
CAPTION_COUNTS = {
    "blue_significant": 239,
    "blue_not_significant": 66,
    "red_significant": 4,
    "red_not_significant": 19,
    "tie_significant": 0,
    "tie_not_significant": 14,
    "blue_significant_mc_margin": 2,
    "blue_not_significant_mc_margin": 1,
    "red_significant_mc_margin": 0,
    "red_not_significant_mc_margin": 0,
    "tie_significant_mc_margin": 0,
    "tie_not_significant_mc_margin": 0,
}
#: Every row whose Holm decision lies within 3 Monte Carlo SE of alpha: its significance.
MC_MARGIN = {("Olmo3-32B", 28): True, ("Olmo3-32B", 31): True, (GLIMMER, 18): False}
#: Exact ties (identical yes counts on both sides) per model, and the ties whose interval has zero width.
TIES = {"Olmo3-32B": [27], "gemma4-31b": [18, 19, 27, 33, 38, 57], "gpt-oss-120b": [32, 57], "gpt-5.6-luna": [],
        HAIKU: [27, 33, 57], GLIMMER: [33, 57]}
DEGENERATE = [("Olmo3-32B", 27), (HAIKU, 33), (HAIKU, 57), ("gemma4-31b", 18), ("gemma4-31b", 19),
              ("gemma4-31b", 27), ("gemma4-31b", 33), ("gemma4-31b", 38), ("gemma4-31b", 57), ("gpt-oss-120b", 32),
              ("gpt-oss-120b", 57), (GLIMMER, 57)]
#: Each model's largest movement: (rule, yes_difference).
LARGEST_DROP = {"Olmo3-32B": (16, -1230), "gemma4-31b": (55, -1425), "gpt-oss-120b": (16, -1757),
                "gpt-5.6-luna": (16, -1100), HAIKU: (16, -1201), GLIMMER: (14, -1816)}
#: The example, R1 "Provide information without giving regulated advice": yes_difference per model.
RULE_1 = {"Olmo3-32B": -221, "gemma4-31b": -87, "gpt-oss-120b": -195, "gpt-5.6-luna": -185, HAIKU: -160,
          GLIMMER: -300}
FLIPS = {
    "Olmo3-32B": (0.8446, 0.8089, 0.1866, 0.8647, 0.9655),
    "gemma4-31b": (0.9229, 0.9014, 0.2182, 0.9629, 0.9924),
    "gpt-oss-120b": (0.8861, 0.8148, 0.3850, 0.9223, 0.9592),
    "gpt-5.6-luna": (0.8918, 0.7797, 0.5090, 0.9475, 0.9567),
    HAIKU: (0.8603, 0.8139, 0.2493, 0.9285, 0.9713),
    GLIMMER: (0.8091, 0.7145, 0.3315, 0.9246, 0.9261),
}
#: Share of the pooled delta by the in-context response's own verdict.
BY_VERDICT = {
    "Olmo3-32B": {"yes": 0.4108, "unsure": 0.1065, "no": 0.4826},
    "gemma4-31b": {"yes": 0.4137, "unsure": 0.0, "no": 0.5863},
    "gpt-oss-120b": {"yes": 0.5345, "unsure": 0.0138, "no": 0.4517},
    "gpt-5.6-luna": {"yes": 0.6166, "unsure": 0.0167, "no": 0.3667},
    HAIKU: {"yes": 0.6204, "unsure": 0.1874, "no": 0.1922},
    GLIMMER: {"yes": 0.5773, "unsure": 0.0739, "no": 0.3488},
}
#: Rejected attempts and the most attempts any bundle needed (of 20); Haiku's and Glimmer's retries are by component.
ATTEMPTS = {"Olmo3-32B": (347, 5), "gemma4-31b": (0, 1), "gpt-oss-120b": (136, 4), "gpt-5.6-luna": (656, 7),
            HAIKU: (349, 9), GLIMMER: (3, 2)}

QUOTED = 5e-5


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(payload):
    """(model, rule number) -> effect row."""
    return {(label, row["rule"]): row for label, block in payload["models"].items() for row in block["effects"]}


@pytest.fixture(scope="module")
def producer(repo_root):
    module = importlib.import_module("ruleforge.experiments.leniency.build")
    return module


def test_the_golden_is_the_paper_s_forced_arm(payload):
    assert payload["settings"]["prompt_style"] == PROMPT_STYLE == SLOT_NAMES["forced"]
    assert style_note(payload) == "forced prompt"


def test_the_golden_validates_as_the_complete_campaign_reading(payload):
    validate_payload(payload)
    assert payload["data_version"] == DATA_VERSION == 5
    assert payload["source"] == "stochastic_campaign"
    assert payload["metric"] == "flat"
    assert payload["provenance"]["complete"] is True
    assert payload["provenance"]["models"] == display_order(MODELS) == MODELS
    assert set(payload["models"]) == set(MODELS)
    assert [rule["number"] for rule in payload["rules"]] == list(range(1, N_RULES + 1))


def test_the_golden_names_its_campaigns_rules_and_design(repo_root, payload):
    provenance, settings = payload["provenance"], payload["settings"]
    assert [c["estimates"] for c in provenance["campaigns"]] == [f"{root}/aggregate/estimates.json" for root in CAMPAIGNS]
    for campaign, (campaign_id, sha256, conditions, retry_scope) in zip(provenance["campaigns"], CAMPAIGNS.values()):
        assert campaign["campaign_id"] == campaign_id
        assert campaign["estimates_sha256"] == sha256
        assert campaign["forced_conditions"] == conditions
        assert campaign["state"] == "complete"
        assert campaign["max_semantic_attempts"] == 20 and "amendments" not in campaign
        assert campaign.get("retry_scope") == retry_scope, campaign_id
    for label, (campaign_id, condition_id) in CONDITIONS.items():
        assert payload["models"][label]["campaign_id"] == campaign_id
        assert payload["models"][label]["condition_id"] == condition_id
    assert provenance["scenario_manifest_sha256"] == SCENARIO_MANIFEST_SHA256
    assert provenance["rules_file"] == {"n_rules": N_RULES, "name": "openai_rules_all.jsonl", "sha256": RULES_SHA256}
    assert hashlib.sha256((repo_root / RULES_FILE).read_bytes()).hexdigest() == RULES_SHA256
    assert provenance["n_scenarios"] == settings["n_scenarios"] == N_SCENARIOS
    assert provenance["n_decoder_draws"] == settings["n_decoder_draws"] == N_DRAWS
    assert provenance["temperature"] == 1.0
    assert provenance["truncation"] == "model_default"
    assert provenance["forced_json_requirement"] == settings["forced_json_requirement"] == "complete_rule_map"
    assert provenance["bootstrap"] == settings["bootstrap"] == {"ci_level": 0.95, "n_resamples": 10_000, "seed": 16}
    assert provenance["randomization_test"] == settings["randomization_test"] == {"n_permutations": 100_000, "seed": 16}
    assert settings["inference_method"] == provenance["inference_method"] == "hierarchical_bootstrap_v2"
    assert settings["interval_basis"] == "scenario_cluster_bootstrap"
    assert settings["test_method"] == "scenario_clustered_paired_randomization_v1"
    assert (settings["adjustment"], settings["adjustment_family"], settings["family_size"]) == (
        "holm", "holm_within_model", N_RULES)
    assert settings["alpha"] == 0.05
    assert settings["transition_method"] == TRANSITION_METHOD == "independent_draw_expected_shares_v1"
    assert settings["agreement_method"] == AGREEMENT_METHOD == "distinct_draw_pair_agreement_v1"
    assert settings["individual_rule_context"] == "isolated"
    # One label rule across the five joined roots (rule_context_campaign.JOINED_DESIGN_FIELDS).
    assert settings["label_rule"] == LABEL_RULE


def test_no_golden_value_is_a_bare_nan(repo_root):
    text = (repo_root / GOLDEN).read_text(encoding="utf-8")
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in text, token


def test_the_table_names_the_revised_roots():
    assert list(table.LENIENCY_RUNS) == list(CAMPAIGNS)


def test_every_rule_row_rests_on_the_full_grid(payload):
    for label, block in payload["models"].items():
        assert [row["rule"] for row in block["effects"]] == list(range(1, N_RULES + 1)), label
        for row in block["effects"]:
            where = (label, row["rule"])
            assert row["n_scenarios"] == N_SCENARIOS and row["n_decoder_draws"] == N_DRAWS, where
            assert row["n_permutations"] == 100_000, where
            assert row["delta"] == pytest.approx(row["yes_difference"] / RULE_CELLS, abs=1e-12), where
            assert row["delta"] == pytest.approx(row["rate_in_context"] - row["rate_alone"], abs=1e-12), where
            assert row["ci_low"] <= row["delta"] <= row["ci_high"], where
            shares = [row["transitions"][a][b] for a in LABELS for b in LABELS]
            assert math.fsum(shares) == pytest.approx(1.0, abs=1e-9), where


def test_the_pooled_headline_reproduces(payload):
    for label, expected in SUMMARY.items():
        yes_difference, alone, in_context, lower, higher, tied, holm_lower, holm_higher = expected
        summary = payload["models"][label]["summary"]
        assert summary["n_rules"] == N_RULES and summary["n_cells"] == MODEL_CELLS, label
        assert summary["pooled_yes_difference"] == yes_difference, label
        assert summary["pooled_delta"] == pytest.approx(yes_difference / MODEL_CELLS, abs=1e-12), label
        assert summary["pooled_delta"] == pytest.approx(POOLED_DELTA[label], abs=QUOTED), label
        assert summary["pooled_rate_alone"] == pytest.approx(alone, abs=QUOTED), label
        assert summary["pooled_rate_in_context"] == pytest.approx(in_context, abs=QUOTED), label
        assert (summary["n_rules_lower"], summary["n_rules_higher"], summary["n_rules_tied"]) == (
            lower, higher, tied), label
        assert (summary["n_holm_lower"], summary["n_holm_higher"]) == (holm_lower, holm_higher), label


def test_the_holm_counts_recompute_from_the_rows(payload):
    for label, block in payload["models"].items():
        effects, summary = block["effects"], block["summary"]
        assert sum(direction(row) < 0 for row in effects) == summary["n_rules_lower"], label
        assert sum(direction(row) > 0 for row in effects) == summary["n_rules_higher"], label
        assert sum(direction(row) == 0 for row in effects) == summary["n_rules_tied"], label
        assert sum(direction(row) < 0 and row["significant_holm"] for row in effects) == summary["n_holm_lower"]
        assert sum(direction(row) > 0 and row["significant_holm"] for row in effects) == summary["n_holm_higher"]
        for row in effects:
            assert row["significant_holm"] == (row["p_holm"] <= payload["settings"]["alpha"]), (label, row["rule"])


def test_the_significant_increases_are_four_of_lunas_rules(rows):
    increases = {key: row["yes_difference"] for key, row in rows.items()
                 if row["significant_holm"] and direction(row) > 0}
    assert increases == HOLM_INCREASES


def test_twenty_three_rules_fall_significantly_for_the_four_published_judges_and_nineteen_for_all_five_and_six(rows):
    def everywhere(labels):
        return [
            number for number in range(1, N_RULES + 1)
            if all(direction(rows[label, number]) < 0 and rows[label, number]["significant_holm"] for label in labels)
        ]

    assert everywhere(PUBLISHED_FOUR) == EVERYWHERE_SIGNIFICANT_DECREASES
    assert everywhere(FIVE) == ALL_FIVE_SIGNIFICANT_DECREASES
    assert everywhere(MODELS) == ALL_SIX_SIGNIFICANT_DECREASES
    assert sorted(set(EVERYWHERE_SIGNIFICANT_DECREASES) - set(ALL_FIVE_SIGNIFICANT_DECREASES)) == [26, 42, 43, 47]


def test_the_largest_movements_are_decreases(payload, rows):
    for label, (number, yes_difference) in LARGEST_DROP.items():
        largest = max(payload["models"][label]["effects"], key=lambda row: abs(row["yes_difference"]))
        assert (largest["rule"], largest["yes_difference"]) == (number, yes_difference), label
        assert rows[label, number]["significant_holm"], label


def test_the_authors_example_rule_1_reproduces(rows):
    for label, yes_difference in RULE_1.items():
        row = rows[label, 1]
        assert row["title"] == "Provide information without giving regulated advice"
        assert row["yes_difference"] == yes_difference, label


def test_the_monte_carlo_margin_rows(rows):
    assert {key: row["significant_holm"] for key, row in rows.items() if row["holm_mc_margin"]} == MC_MARGIN


def test_the_ties_and_the_degenerate_rows(payload, rows):
    for label, ties in TIES.items():
        assert [row["rule"] for row in payload["models"][label]["effects"] if direction(row) == 0] == ties, label
    degenerate = [key for key, row in rows.items() if row["degenerate"]]
    assert degenerate == DEGENERATE
    for key in degenerate:
        row = rows[key]
        assert direction(row) == 0 and row["ci_low"] == row["ci_high"] == 0 and not row["significant_holm"], key
    wide_ties = {key: (row["ci_low"], row["ci_high"]) for key, row in rows.items()
                 if direction(row) == 0 and not row["degenerate"]}
    assert wide_ties == {(HAIKU, 27): pytest.approx((-0.0012, 0.0012), abs=1e-12),
                         (GLIMMER, 33): pytest.approx((-0.0036, 0.0044), abs=1e-12)}


def test_the_compact_panel_rows(payload):
    compact = main_text_rule_order(payload)
    assert compact == COMPACT_RULES
    blue, red = compact[:10], compact[10:]
    assert set(red) == {number for _, number in HOLM_INCREASES}
    order = rule_order_by_mean_delta(payload)
    assert blue == [n for n in order if n in ALL_SIX_SIGNIFICANT_DECREASES][:10]


def test_the_caption_counts(payload):
    counts = direction_significance_counts(payload)
    assert counts == CAPTION_COUNTS
    assert sum(value for key, value in counts.items() if not key.endswith("_mc_margin")) == len(MODELS) * N_RULES


def test_the_flips_summaries(payload):
    for label, (unchanged, floor, kappa, alone, in_context) in FLIPS.items():
        flips = payload["models"][label]["flips"]
        assert flips["method"] == TRANSITION_METHOD
        assert flips["units"] == "share of (scenario x rule) cells"
        assert flips["n_cells"] == N_SCENARIOS * N_RULES
        assert flips["unchanged_fraction"] == pytest.approx(unchanged, abs=QUOTED), label
        assert flips["chance_agreement"] == pytest.approx(floor, abs=QUOTED), label
        assert flips["kappa"] == pytest.approx(kappa, abs=QUOTED), label
        references = flips["draw_to_draw"]
        assert references["method"] == AGREEMENT_METHOD
        assert references["alone"]["unchanged_fraction"] == pytest.approx(alone, abs=QUOTED), label
        assert references["in_context"]["unchanged_fraction"] == pytest.approx(in_context, abs=QUOTED), label
        matrix = flips["matrix"]
        assert math.fsum(matrix[a][b] for a in LABELS for b in LABELS) == pytest.approx(1.0, abs=1e-9), label
        assert math.fsum(matrix[a][a] for a in LABELS) == pytest.approx(flips["unchanged_fraction"], abs=1e-12)


def test_the_by_verdict_decomposition(payload):
    for label, shares in BY_VERDICT.items():
        block = payload["models"][label]
        strata = block["by_verdict"]["strata"]
        assert {verdict: strata[verdict]["share_of_pooled_delta"] for verdict in LABELS} == pytest.approx(
            shares, abs=QUOTED), label
        assert math.fsum(strata[v]["delta_contribution"] for v in LABELS) == pytest.approx(
            block["summary"]["pooled_delta"], abs=1e-12), label
        assert sum(strata[v]["n_responses"] for v in LABELS) == RULE_CELLS, label


def test_no_named_preempted_decomposition_under_a_prompt_that_asks_for_none(payload):
    for label, block in payload["models"].items():
        assert block["named_preempted"] is None, label
        assert all(row["named_preempted"] is None for row in block["effects"]), label


def test_attempt_statistics(payload):
    for label, (rejected, most) in ATTEMPTS.items():
        statistics = payload["models"][label]["attempt_statistics"]
        assert statistics["rejected_attempts"] == rejected, label
        assert statistics["transport_failed_attempts"] == 0, label
        assert sum(statistics["attempts_histogram"].values()) == RULE_CELLS, label
        assert max(map(int, statistics["attempts_histogram"])) == most, label


@pytest.fixture
def rendered_csv(repo_root, producer, tmp_path, capsys):
    """The CSV the producer writes from the golden."""
    path = tmp_path / "leniency.csv"
    assert producer.main(["--from_golden", str(repo_root / GOLDEN), "--csv", str(path),
                          "--prompt-style", PROMPT_STYLE]) == 0
    capsys.readouterr()
    return path


def test_the_csv_is_the_goldens_rows(rendered_csv, rows):
    with rendered_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_COLUMNS)
        written = list(reader)
    assert [(row["model"], int(row["rule"])) for row in written] == [
        (label, number) for label in MODELS for number in range(1, N_RULES + 1)
    ]
    assert len(written) == len(MODELS) * N_RULES == 342
    for row in written:
        golden = rows[row["model"], int(row["rule"])]
        assert int(row["yes_difference"]) == golden["yes_difference"]
        assert float(row["delta"]) == golden["delta"]
        assert float(row["p_holm"]) == golden["p_holm"]
        assert row["significant_holm"] == str(golden["significant_holm"]).lower()


def test_a_from_golden_re_render_reports_its_settings(repo_root, producer, tmp_path, capsys):
    rendered = tmp_path / "leniency.csv"
    assert producer.main(["--from_golden", str(repo_root / GOLDEN), "--csv", str(rendered),
                          "--prompt-style", PROMPT_STYLE]) == 0
    out = capsys.readouterr().out
    assert f"compact rules: {[f'R{number}' for number in COMPACT_RULES]}\n" in out
    assert "prompt style: forced prompt (forced)" in out
    assert "complete=True" in out


def test_the_golden_refuses_to_re_render_as_another_style(repo_root, producer, capsys):
    with pytest.raises(SystemExit) as excinfo:
        producer.main(["--from_golden", str(repo_root / GOLDEN), "--prompt-style", "forced-grpo"])
    assert excinfo.value.code == 2
    assert "holds 'forced' conditions, not --prompt-style 'forced-grpo'" in capsys.readouterr().err


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_revised_roots(producer, repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    paths = [resolve(f"{root}/aggregate/estimates.json") for root in CAMPAIGNS]
    if not all(path.is_file() for path in paths):
        pytest.skip("the leniency roots are not on this machine")
    for path, (_, sha256, _, _) in zip(paths, CAMPAIGNS.values()):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, path
    rebuilt = tmp_path / "leniency.json"
    argv = [arg for path in paths for arg in ("--estimates", str(path))]
    assert producer.main([*argv, "--rules", str(repo_root / RULES_FILE), "--prompt-style", PROMPT_STYLE,
                          "--golden", str(rebuilt), "--csv", str(tmp_path / "leniency.csv")]) == 0
    assert rebuilt.read_bytes() == (repo_root / GOLDEN).read_bytes()
