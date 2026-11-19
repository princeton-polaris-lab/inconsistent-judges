from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ruleforge.core.paths import resolve
from ruleforge.evaluate.bank import default_contrasts
from ruleforge.experiments import table
from ruleforge.experiments.ruleset_size import build_repeated_draw_baseline
from ruleforge.experiments.ruleset_size import repeated_draw_baseline as rdb

pytestmark = pytest.mark.frozen

PROGRAM = "ruleforge.experiments.ruleset_size.build_repeated_draw_baseline"
GOLDEN = "results/ruleset_size/repeated_draw_baseline.json"
CURVES = "results/ruleset_size/curves.json"
SIZES = list(range(2, 58))
#: Per judge, the inconsistency rate, the repeated-draw baseline and the excess at the smallest and largest size, to
#: four places.
ENDS = {
    "GPT-OSS-120B": (0.1534, 0.1205, 0.0329, 0.6058, 0.0927, 0.5131),
    "Gemma-4-31B": (0.1335, 0.0788, 0.0546, 0.7169, 0.0538, 0.6632),
    "Olmo-3-32B": (0.2256, 0.2026, 0.0231, 0.7844, 0.1237, 0.6607),
    "Muse-Glimmer-30B": (0.1576, 0.1156, 0.042, 0.4808, 0.1074, 0.3734),
    "GPT-5.6 Luna": (0.1156, 0.0806, 0.035, 0.4174, 0.0874, 0.33),
    "Claude Haiku 5.5": (0.1985, 0.1305, 0.068, 0.634, 0.1056, 0.5284),
}
#: Per judge, the net rise (K57 - K2) of the excess and of the repeated-draw baseline, to four places.
NET_RISE = {
    "GPT-OSS-120B": (0.4801, -0.0277),
    "Gemma-4-31B": (0.6085, -0.0251),
    "Olmo-3-32B": (0.6377, -0.0789),
    "Muse-Glimmer-30B": (0.3314, -0.0082),
    "GPT-5.6 Luna": (0.295, 0.0068),
    "Claude Haiku 5.5": (0.4604, -0.0249),
}

#: Per judge, the full-set and separate-call repeat disagreement and the lenient and strict shares at the smallest and
#: largest size, to four places: the statistics that are not symmetric in the two calls.
DIRECTED_ENDS = {
    "GPT-OSS-120B": (0.1076, 0.1333, 0.1193, 0.034, 0.159, 0.0264, 0.6055, 0.0003),
    "Gemma-4-31B": (0.0694, 0.0882, 0.115, 0.0185, 0.0511, 0.0564, 0.7169, 0.0),
    "Olmo-3-32B": (0.1506, 0.2546, 0.149, 0.0767, 0.1543, 0.093, 0.783, 0.0014),
    "Muse-Glimmer-30B": (0.1045, 0.1267, 0.1202, 0.0374, 0.2148, 0.0, 0.4808, 0.0),
    "GPT-5.6 Luna": (0.0731, 0.0881, 0.0904, 0.0252, 0.0935, 0.0812, 0.4162, 0.0012),
    "Claude Haiku 5.5": (0.11, 0.151, 0.1294, 0.0691, 0.1352, 0.076, 0.6311, 0.0029),
}
#: Per judge, the rule-median inconsistency rate at the smallest and largest size and its net rise, to four places.
RULE_MEDIAN = {
    "GPT-OSS-120B": (0.1303, 0.5998, 0.4696),
    "Gemma-4-31B": (0.124, 0.6956, 0.5716),
    "Olmo-3-32B": (0.1818, 0.781, 0.5992),
    "Muse-Glimmer-30B": (0.1462, 0.4808, 0.3346),
    "GPT-5.6 Luna": (0.1034, 0.4059, 0.3025),
    "Claude Haiku 5.5": (0.1864, 0.6281, 0.4418),
}

#: Per judge, the median inconsistency rate (both sides at their median over the draws) at the smallest and largest
#: size and its net rise, to four places.
MEDIAN = {
    "GPT-OSS-120B": (0.112, 0.6138, 0.5018),
    "Gemma-4-31B": (0.1194, 0.7008, 0.5814),
    "Olmo-3-32B": (0.1576, 0.8072, 0.6496),
    "Muse-Glimmer-30B": (0.1302, 0.493, 0.3628),
    "GPT-5.6 Luna": (0.0918, 0.4122, 0.3204),
    "Claude Haiku 5.5": (0.1756, 0.6338, 0.4582),
}


@pytest.fixture(scope="module")
def payload(repo_root):
    return build_repeated_draw_baseline.load_golden(repo_root / GOLDEN)


@pytest.fixture(scope="module")
def curves(repo_root):
    return {curve["label"]: curve for curve in json.loads((repo_root / CURVES).read_text(encoding="utf-8"))}


def test_the_golden_defines_every_statistic_and_names_the_campaign_s_bootstrap(payload):
    assert (payload["analysis"], payload["schema_version"], payload["definition"]) == (
        rdb.ANALYSIS, rdb.SCHEMA_VERSION, rdb.DEFINITION)
    assert payload["statistics"] == list(rdb.STATISTICS) and set(rdb.STATISTICS) <= set(rdb.DEFINITION)
    assert payload["bootstrap"] == {"n_resamples": 10_000, "ci_level": 0.95, "seed": 16,
                                    "interval_basis": "scenario_subset_cluster_bootstrap"}
    assert (payload["subset_sizes"], payload["contrasts"]) == (SIZES, default_contrasts(SIZES))
    assert (payload["n_scenarios"], payload["n_subset_seeds"], payload["n_decoder_draws"]) == (500, 10, 5)


def test_each_judge_s_inconsistency_is_its_ruleset_size_curve_exactly(payload, curves):
    assert [judge["label"] for judge in payload["judges"]] == list(curves)
    for judge in payload["judges"]:
        curve = curves[judge["label"]]
        assert judge["model_name"] == curve["campaign"]["model_name"]
        assert judge["campaign"] == {key: curve["campaign"][key] for key in judge["campaign"]}
        assert {"campaign_id", "estimates", "estimates_sha256"} <= set(judge["campaign"])
        assert judge["grid_digests"] == {"bank": curve["bank"]["grid_digest"],
                                         "subsets": curve["subsets"]["grid_digest"]}
        assert [point["size"] for point in judge["points"]] == SIZES
        for point, row in zip(judge["points"], curve["points"]):
            assert point["inconsistency"] == {"estimate": row["mean"], "ci_low": row["ci_low"],
                                              "ci_high": row["ci_high"], "sd": row["sd"]}, (judge["label"], row["size"])
        keys = ("minuend_sizes", "subtrahend_sizes", "estimate", "ci_low", "ci_high", "sd")
        assert judge["contrasts"]["inconsistency"] == {name: {key: row[key] for key in keys}
                                                       for name, row in curve["contrasts"].items()}


def test_the_derived_statistics_are_their_definitions_at_every_size(payload):
    for judge in payload["judges"]:
        for point in judge["points"]:
            value = {name: point[name]["estimate"] for name in rdb.STATISTICS}
            assert value["repeated_draw_baseline"] == pytest.approx(
                (value["full_repeat"] + value["separate_repeat"]) / 2)
            assert value["excess"] == pytest.approx(value["inconsistency"] - value["repeated_draw_baseline"])
            assert value["lenient_minus_strict"] == pytest.approx(value["lenient"] - value["strict"])
            assert value["lenient"] + value["strict"] == pytest.approx(value["inconsistency"])
            for name in rdb.STATISTICS:
                assert point[name]["ci_low"] <= point[name]["estimate"] <= point[name]["ci_high"], (judge["label"], name)
        assert sorted(judge["contrasts"]) == sorted(rdb.STATISTICS)
        assert all(sorted(contrasts) == ["last_step", "net_rise", "upper_vs_middle"]
                   for contrasts in judge["contrasts"].values())


def test_the_pinned_ends_and_net_rises(payload):
    for judge in payload["judges"]:
        first, last = judge["points"][0], judge["points"][-1]
        assert tuple(round(point[name]["estimate"], 4) for point in (first, last)
                     for name in ("inconsistency", "repeated_draw_baseline", "excess")) == ENDS[judge["label"]], \
            judge["label"]
        assert tuple(round(point[name]["estimate"], 4) for point in (first, last)
                     for name in ("full_repeat", "separate_repeat", "lenient", "strict")) == DIRECTED_ENDS[
                         judge["label"]], judge["label"]
        assert tuple(round(judge["contrasts"][name]["net_rise"]["estimate"], 4)
                     for name in ("excess", "repeated_draw_baseline")) == NET_RISE[judge["label"]], judge["label"]
        median = (first["rule_median_inconsistency"], last["rule_median_inconsistency"],
                  judge["contrasts"]["rule_median_inconsistency"]["net_rise"])
        assert tuple(round(row["estimate"], 4) for row in median) == RULE_MEDIAN[judge["label"]], judge["label"]
        both = (first["median_inconsistency"], last["median_inconsistency"],
                judge["contrasts"]["median_inconsistency"]["net_rise"])
        assert tuple(round(row["estimate"], 4) for row in both) == MEDIAN[judge["label"]], judge["label"]


def test_the_csv_renders_a_row_per_judge_and_size_from_the_golden(repo_root, tmp_path, capsys):
    out = tmp_path / "repeated_draw_baseline.csv"
    assert build_repeated_draw_baseline.main(["--from_golden", str(repo_root / GOLDEN), "--csv", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert lines[0] == ",".join(rdb.CSV_COLUMNS) and len(lines) == 1 + 6 * len(SIZES)
    assert capsys.readouterr().out.splitlines()[0] == "GPT-OSS-120B"


def test_the_ruleset_size_experiment_builds_it_and_draws_it_with_a_csv():
    (row,) = [row for row in table.TABLE if row.name == "ruleset-size"]
    (step,) = [step for step in row.builds if step.id == "ruleset_size_repeated_draw_baseline"]
    (curves,) = [step for step in row.builds if step.id == "ruleset_size"]
    assert step.words[2] == PROGRAM and step.words[3:-2] == curves.words[3:-2]
    assert step.words[-2:] == ["--golden", f"{{out}}/{GOLDEN}"]
    assert (step.golden.name, step.golden.path, step.golden.rerun_tolerance) == (
        "ruleset_size_repeated_draw_baseline", GOLDEN, 0.005)
    plots = [plot for plot in row.plots if f"repo://{GOLDEN}" in plot.words]
    assert sorted(word for plot in plots for word in plot.words if word.startswith("{out}/")) == [
        "{out}/ruleset_size_repeated_draw_baseline.csv", "{out}/ruleset_size_repeated_draw_baseline.pdf",
        "{out}/ruleset_size_repeated_draw_baseline.png"]


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_paper_roots(repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("no artifact root is set")
    estimates = [resolve(run) / "aggregate" / "figure2_estimates.json" for run in table.RULESET_SIZE_RUNS]
    if not all(path.is_file() for path in estimates):
        pytest.skip("the ruleset-size roots are not in this artifact root")
    out = tmp_path / "repeated_draw_baseline.json"
    built = subprocess.run([sys.executable, "-m", PROGRAM, *(word for path in estimates for word in ("--estimates", str(path))),
                            "--golden", str(out)], cwd=repo_root,
                           env={**os.environ, "SOURCE_DATE_EPOCH": "0", "OMP_NUM_THREADS": "4",
                                "OPENBLAS_NUM_THREADS": "4"}, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    assert out.read_bytes() == (repo_root / GOLDEN).read_bytes()
