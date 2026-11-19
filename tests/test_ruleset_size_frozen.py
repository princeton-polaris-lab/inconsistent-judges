from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from ruleforge.experiments.ruleset_size.curves import CONTRADICTION, INCONSISTENCY_RATE
from ruleforge.experiments import table
from ruleforge.core.paths import resolve

GOLDEN = "results/ruleset_size/curves.json"
PRODUCER = "ruleforge.experiments.ruleset_size.build"

LABELS = ["GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "Muse-Glimmer-30B", "GPT-5.6 Luna", "Claude Haiku 5.5"]
GLIMMER = "Muse-Glimmer-30B"
SIZES = list(range(2, 58))
N_SCENARIOS, N_SEEDS, N_DRAWS = 500, 10, 5
N_OBS = N_SCENARIOS * N_SEEDS * N_DRAWS

#: The one label rule every curve's root records.
LABEL_RULE = "lenient_last_line"
RESCORED = f"rescored-{LABEL_RULE}"
#: Each judge's root of size 57, then the published root its bank and smaller sizes come from.
PAIRS = [
    ("k57-shuffled-2026-11", f"fig2-2026-09-pop-a22af097-{RESCORED}"),
    ("k57-shuffled-2026-11-luna", f"fig2-2026-09-luna-pop-a22af097-{RESCORED}"),
    ("k57-shuffled-2026-11-haiku", f"fig2-2026-10-haiku-sum-{RESCORED}"),
    ("k57-shuffled-2026-11-glimmer", "fig2-2026-10-glimmer-pop-a22af097"),
]
ROOTS = [f"artifact://sampling/{size_57}" for size_57, _ in PAIRS]

#: Per curve: the campaign that judged its bank, that campaign's root, the judge, the sha256 of its joined estimates
#: (in its size-57 root), its retry scope and its size-57 root.
CAMPAIGNS = {
    "GPT-OSS-120B": ("fig2-2026-09", f"fig2-2026-09-pop-a22af097-{RESCORED}", "gpt-oss-120b", "0cd3bce41131b2b3c2d36f46299b6f9cacaad01887a066e052bc665a5167e70c",
                     "scenario_bundle", "k57-shuffled-2026-11"),
    "Gemma-4-31B": ("fig2-2026-09", f"fig2-2026-09-pop-a22af097-{RESCORED}", "gemma-4-31B-it", "0cd3bce41131b2b3c2d36f46299b6f9cacaad01887a066e052bc665a5167e70c",
                    "scenario_bundle", "k57-shuffled-2026-11"),
    "Olmo-3-32B": ("fig2-2026-09", f"fig2-2026-09-pop-a22af097-{RESCORED}", "Olmo-3-32B-Think", "0cd3bce41131b2b3c2d36f46299b6f9cacaad01887a066e052bc665a5167e70c",
                   "scenario_bundle", "k57-shuffled-2026-11"),
    "Muse-Glimmer-30B": ("fig2-2026-10-glimmer", "fig2-2026-10-glimmer-pop-a22af097", "Muse-Glimmer-30B", "dbfac1cc816a6e1c100be2fcc1c4c9dd65e7798762bcf0a386fdea6efbc9228c",
                         "component", "k57-shuffled-2026-11-glimmer"),
    "GPT-5.6 Luna": ("fig2-2026-09-luna", f"fig2-2026-09-luna-pop-a22af097-{RESCORED}", "gpt-5.6-luna", "28c19551231e45073733916b4d1cbd4684f4bcf235b09c53192497748ebe2ceb",
                     "scenario_bundle", "k57-shuffled-2026-11-luna"),
    "Claude Haiku 5.5": ("fig2-2026-10-haiku-sum", f"fig2-2026-10-haiku-sum-{RESCORED}", "claude-haiku-5-5", "0467eb5d796e430405767f1944be49553836dacb6d2602b8bef4d017507e6d94",
                         "component", "k57-shuffled-2026-11-haiku"),
}

SCENARIO_MANIFEST_SHA256 = "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"
SUBSET_CHECKSUMS = "data/rules/subset_manifest_shuffled_full/manifest.sha256"
SOURCE_RULESET = "data/rules/modelspec/openai_rules_all_full.jsonl"

#: (inconsistent judgments of 25,000, ci_low, ci_high) at each landmark.
LANDMARKS = {
    "GPT-OSS-120B": {2: (3834, 0.105359, 0.199042), 10: (9356, 0.272000, 0.479844),
                     30: (13529, 0.482159, 0.598681), 57: (15144, 0.567959, 0.642001)},
    "Gemma-4-31B": {2: (3337, 0.064679, 0.222084), 10: (9660, 0.262396, 0.510561),
                    30: (14303, 0.476276, 0.656322), 57: (17923, 0.680958, 0.751480)},
    "Olmo-3-32B": {2: (5641, 0.166079, 0.282240), 10: (11715, 0.394478, 0.548282),
                   30: (17318, 0.649878, 0.737321), 57: (19610, 0.752959, 0.814961)},
    "Muse-Glimmer-30B": {2: (3939, 0.095679, 0.224643), 10: (8307, 0.232720, 0.432600),
                         30: (12199, 0.407880, 0.557121), 57: (12020, 0.425077, 0.529561)},
    "GPT-5.6 Luna": {2: (2889, 0.089119, 0.140480), 10: (6469, 0.202040, 0.320120),
                     30: (8668, 0.297719, 0.397841), 57: (10434, 0.377919, 0.455681)},
    "Claude Haiku 5.5": {2: (4962, 0.124679, 0.281241), 10: (10399, 0.355839, 0.473600),
                         30: (13684, 0.491599, 0.604242), 57: (15850, 0.590919, 0.675681)},
}

CONTRASTS = {
    ("GPT-OSS-120B", "net_rise"): (+0.452400, +0.385798, +0.519440),
    ("GPT-OSS-120B", "upper_vs_middle"): (+0.038113, +0.014308, +0.066692),
    ("GPT-OSS-120B", "last_step"): (-0.006200, -0.023001, +0.009481),
    ("Gemma-4-31B", "net_rise"): (+0.583440, +0.482319, +0.666841),
    ("Gemma-4-31B", "upper_vs_middle"): (+0.135516, +0.111858, +0.159630),
    ("Gemma-4-31B", "last_step"): (-0.003000, -0.013040, +0.005760),
    ("Olmo-3-32B", "net_rise"): (+0.558760, +0.496840, +0.625080),
    ("Olmo-3-32B", "upper_vs_middle"): (+0.069658, +0.040291, +0.101098),
    ("Olmo-3-32B", "last_step"): (-0.000720, -0.019961, +0.017841),
    ("Muse-Glimmer-30B", "net_rise"): (+0.323240, +0.218560, +0.417562),
    ("Muse-Glimmer-30B", "upper_vs_middle"): (-0.021651, -0.068073, +0.024080),
    ("Muse-Glimmer-30B", "last_step"): (+0.005600, -0.063603, +0.065481),
    ("GPT-5.6 Luna", "net_rise"): (+0.301800, +0.252160, +0.352761),
    ("GPT-5.6 Luna", "upper_vs_middle"): (+0.054851, +0.027083, +0.084066),
    ("GPT-5.6 Luna", "last_step"): (-0.011680, -0.026280, +0.003441),
    ("Claude Haiku 5.5", "net_rise"): (+0.435520, +0.328160, +0.535002),
    ("Claude Haiku 5.5", "upper_vs_middle"): (+0.040611, +0.008222, +0.080761),
    ("Claude Haiku 5.5", "last_step"): (+0.000160, -0.020120, +0.019881),
}
CONTRAST_LABELS = {
    "net_rise": "K57 - K2",
    "upper_vs_middle": "K56 - mean(K20..K30)",
    "last_step": "K57 - K56",
}

REJECTED_ATTEMPTS = {
    "GPT-OSS-120B": (46, 507),
    "Gemma-4-31B": (9, 114),
    "Olmo-3-32B": (11, 236),
    "Muse-Glimmer-30B": (2, 21),
    "GPT-5.6 Luna": (0, 0),
    "Claude Haiku 5.5": (561, 2203),
}
MAX_ATTEMPTS = {
    "GPT-OSS-120B": (3, 2),
    "Gemma-4-31B": (2, 4),
    "Olmo-3-32B": (8, 4),
    "Muse-Glimmer-30B": (2, 2),
    "GPT-5.6 Luna": (1, 1),
    "Claude Haiku 5.5": (28, 15),
}
MAX_SEMANTIC_ATTEMPTS = 20
AMENDED = {"Claude Haiku 5.5": 30}


@pytest.fixture(scope="module")
def curves(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def points(curves):
    """Every point by label and size: {label: {size: point}}."""
    return {curve["label"]: {p["size"]: p for p in curve["points"]} for curve in curves}


@pytest.fixture(scope="module")
def means(points):
    return {label: {size: p["mean"] for size, p in by_size.items()} for label, by_size in points.items()}


@pytest.fixture(scope="module")
def producer():
    return importlib.import_module(PRODUCER)


def test_the_table_names_each_judge_s_size_57_root_and_the_root_of_its_bank():
    assert list(table.RULESET_SIZE_PAIRS) == PAIRS
    assert list(table.RULESET_SIZE_RUNS) == ROOTS
    assert list(table.RULESET_SIZE_BANK_RUNS) == [f"artifact://sampling/{bank}" for _, bank in PAIRS]
    (row,) = [row for row in table.TABLE if row.name == "ruleset-size"]
    assert set(row.runs) == {f"artifact://sampling/{name}" for pair in PAIRS for name in pair}
    assert all(step.inputs == table.RULESET_SIZE_BANK_RUNS for step in row.builds if step.golden is not None)


def test_every_curve_names_its_campaign_and_estimates_file(curves):
    assert [curve["label"] for curve in curves] == LABELS
    for curve in curves:
        campaign_id, root, model_name, sha256, retry_scope, k57 = CAMPAIGNS[curve["label"]]
        campaign = curve["campaign"]
        assert campaign["campaign_id"] == campaign_id
        assert campaign["model_name"] == campaign["condition_prefix"] == model_name
        assert campaign["estimates"] == f"artifact://sampling/{k57}/aggregate/figure2_estimates.json"
        assert campaign["root"] == f"artifact://sampling/{root}"
        # The size-57 root ran under the bank campaign's retry scope, at the cap that campaign held in force.
        assert campaign["sizes_from"] == {"57": {"campaign_id": k57, "root": f"artifact://sampling/{k57}",
                                                 "retry_scope": retry_scope,
                                                 "max_semantic_attempts": AMENDED.get(curve["label"],
                                                                                      MAX_SEMANTIC_ATTEMPTS)}}
        assert campaign["estimates_sha256"] == sha256
        assert campaign["state"] == "complete"
        assert campaign["retry_scope"] == retry_scope
        amendments = campaign.get("amendments", [])
        assert [(a["old_cap"], a["new_cap"]) for a in amendments] == (
            [(MAX_SEMANTIC_ATTEMPTS, AMENDED[curve["label"]])] if curve["label"] in AMENDED else []), curve["label"]


def test_one_experiment_one_population_one_subset_design(repo_root, curves):
    subset_checksums = hashlib.sha256((repo_root / SUBSET_CHECKSUMS).read_bytes()).hexdigest()
    source_ruleset = hashlib.sha256((repo_root / SOURCE_RULESET).read_bytes()).hexdigest()
    ablation = json.loads((repo_root / "results/prompt_ablation/prompt_ablation.json").read_text(encoding="utf-8"))
    ruleforge = json.loads((repo_root / "results/ruleforge/ruleforge.json").read_text(encoding="utf-8"))
    assert ablation["provenance"]["scenario_manifest_sha256"] == SCENARIO_MANIFEST_SHA256
    assert ruleforge["provenance"]["scenario_manifest_sha256"] == SCENARIO_MANIFEST_SHA256
    for curve in curves:
        campaign = curve["campaign"]
        assert campaign["scenario_manifest_sha256"] == SCENARIO_MANIFEST_SHA256
        assert campaign["design"] == {
            "design": "shared_individual_bank_v1",
            "subset_seeds": list(range(N_SEEDS)),
            "subset_sizes": SIZES,
            "sampling_strategy": "with_replacement_shuffled_full",
            "subset_manifest_sha256": subset_checksums,
            "source_ruleset_sha256": source_ruleset,
        }
        assert campaign["temperature"] == 1.0
        assert campaign["truncation"] == "model_default"
        assert campaign["label_rule"] == LABEL_RULE
        assert campaign["max_semantic_attempts"] == MAX_SEMANTIC_ATTEMPTS
        assert curve["bank"]["n_rules"] == 57
        assert curve["subsets"]["n_conditions"] == N_SEEDS * len(SIZES)


def test_the_golden_was_written_by_the_campaign_producer(producer, curves):
    assert producer.validate_curves(curves) == curves
    for curve in curves:
        assert curve["code_version"] == producer.CURVE_VERSION == 5
        assert curve["metric"] == CONTRADICTION
        assert curve["quantity"] == INCONSISTENCY_RATE
        assert curve["strategy"] == "No_Interpretation"
        assert curve["derived_from"] == "stochastic campaign shared_individual_bank_v1"
        assert curve["ci_estimator"] == "scenario_subset_cluster_bootstrap"
        assert curve["inference_method"] == "crossed_scenario_subset_bankshared_decoder_bootstrap_v2"
        assert (curve["ci_level"], curve["n_resamples"], curve["rng_seed"]) == (0.95, 10_000, 16)
        assert (curve["n_scenarios"], curve["n_subset_seeds"], curve["n_decoder_draws"]) == (
            N_SCENARIOS, N_SEEDS, N_DRAWS,
        )
        assert curve["n_expected"] == N_OBS


def test_no_golden_value_is_a_bare_nan(repo_root):
    text = (repo_root / GOLDEN).read_text(encoding="utf-8")
    for token in ("NaN", "Infinity"):
        assert token not in text, token


def test_every_point_rests_on_the_complete_grid(curves):
    for curve in curves:
        assert [point["size"] for point in curve["points"]] == SIZES, curve["label"]
        for point in curve["points"]:
            where = f"{curve['label']} K={point['size']}"
            assert point["n_obs"] == N_OBS, where
            assert (point["n_scenarios"], point["n_seeds"], point["n_draws"]) == (N_SCENARIOS, N_SEEDS, N_DRAWS)
            # A mean over 25,000 binary judgments is an integer over 25,000.
            assert point["mean"] * N_OBS == pytest.approx(round(point["mean"] * N_OBS), abs=1e-6), where
            assert point["ci_low"] <= point["mean"] <= point["ci_high"], where
            two_stage = point["two_stage"]
            assert two_stage["ci_low"] <= point["mean"] <= two_stage["ci_high"], where


def test_every_landmark_reproduces(curves):
    checked = 0
    for curve in curves:
        by_size = {point["size"]: point for point in curve["points"]}
        for size, (inconsistent, ci_low, ci_high) in LANDMARKS[curve["label"]].items():
            row, where = by_size[size], f"{curve['label']} K={size}"
            assert row["mean"] == pytest.approx(inconsistent / N_OBS, abs=1e-12), where
            assert row["ci_low"] == pytest.approx(ci_low, abs=5e-7), where
            assert row["ci_high"] == pytest.approx(ci_high, abs=5e-7), where
            checked += 1
    assert checked == 6 * 4, "six models at four landmarks"


def test_every_contrast_reproduces_and_is_its_curves_own_difference(curves, means):
    checked = 0
    for curve in curves:
        label = curve["label"]
        assert set(curve["contrasts"]) == set(CONTRAST_LABELS), label
        for name, row in curve["contrasts"].items():
            estimate, ci_low, ci_high = CONTRASTS[(label, name)]
            assert row["contrast"] == CONTRAST_LABELS[name]
            assert row["estimate"] == pytest.approx(estimate, abs=5e-7), (label, name)
            assert row["ci_low"] == pytest.approx(ci_low, abs=5e-7), (label, name)
            assert row["ci_high"] == pytest.approx(ci_high, abs=5e-7), (label, name)
            assert row["excludes_zero"] is (ci_low > 0 or ci_high < 0), (label, name)
            minuend = sum(means[label][k] for k in row["minuend_sizes"]) / len(row["minuend_sizes"])
            subtrahend = sum(means[label][k] for k in row["subtrahend_sizes"]) / len(row["subtrahend_sizes"])
            assert row["estimate"] == pytest.approx(minuend - subtrahend, abs=1e-12), (label, name)
            checked += 1
    assert checked == len(CONTRASTS) == 18


def test_every_curve_s_adjacent_rises_count_its_rising_steps(curves):
    rises = {curve["label"]: curve["adjacent_rises"] for curve in curves}
    assert rises == {"GPT-OSS-120B": 32, "Gemma-4-31B": 35, "Olmo-3-32B": 31, GLIMMER: 31,
                     "GPT-5.6 Luna": 33, "Claude Haiku 5.5": 31}
    for curve in curves:
        steps = [b["mean"] > a["mean"] for a, b in zip(curve["points"], curve["points"][1:])]
        assert sum(steps) == curve["adjacent_rises"], curve["label"]


def test_no_size_is_one_ruleset_for_every_seed(curves):
    for curve in curves:
        assert [p["size"] for p in curve["points"] if p["shares_one_ruleset"]] == [], curve["label"]


def test_rejected_attempts_are_recorded_and_no_transport_failed(curves):
    for curve in curves:
        bank = curve["bank"]["attempt_statistics"]
        subsets = curve["subsets"]["attempt_statistics"]
        assert (bank["rejected_attempts"], subsets["rejected_attempts"]) == REJECTED_ATTEMPTS[curve["label"]]
        assert sum(bank["attempts_histogram"].values()) == N_SCENARIOS * N_DRAWS
        assert sum(subsets["attempts_histogram"].values()) == N_SEEDS * len(SIZES) * N_SCENARIOS * N_DRAWS
        assert bank["transport_failed_attempts"] == subsets["transport_failed_attempts"] == 0
        most = (max(map(int, bank["attempts_histogram"])), max(map(int, subsets["attempts_histogram"])))
        cap = AMENDED.get(curve["label"], MAX_SEMANTIC_ATTEMPTS)
        assert most == MAX_ATTEMPTS[curve["label"]] and max(most) <= cap, curve["label"]


@pytest.mark.cluster
def test_the_goldens_rebuild_byte_for_byte_from_the_campaign_estimates(producer, curves, repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    paths = [resolve(uri) / "aggregate" / "figure2_estimates.json" for uri in ROOTS]
    if not all(path.is_file() for path in paths):
        pytest.skip("the ruleset-size roots are not on this machine")
    for curve in curves:
        estimates = resolve(curve["campaign"]["estimates"])
        _, _, _, sha256, _, _ = CAMPAIGNS[curve["label"]]
        assert hashlib.sha256(estimates.read_bytes()).hexdigest() == sha256
    rebuilt = tmp_path / Path(GOLDEN).name
    argv = []
    for path in paths:
        argv += ["--estimates", str(path)]
    assert producer.main([*argv, "--golden", str(rebuilt)]) == 0
    assert rebuilt.read_bytes() == (repo_root / GOLDEN).read_bytes()
