import importlib
import json
import os

import numpy as np
import pytest

from ruleforge.evaluate.inference import (
    DrawGrid,
    hierarchical_bootstrap,
    clustered_paired_randomization_p,
    holm_adjust,
)
from ruleforge.experiments.ruleforge.comparison import expected_transition_shares
from ruleforge.experiments import table
from ruleforge.core.paths import resolve
from ruleforge.scenarios.manifest import ScenarioManifest

INITIAL, WINNER, RUNNER_UP = "Initial", "Seed 5 (winner)", "Seed 9 (runner-up)"
REVISED = ("Revised ruleset 1", "Revised ruleset 2")
GOLDEN = "results/ruleforge/ruleforge.json"
ROOT = "artifact://sampling/fig4-2026-09-pop-a22af097-rescored-lenient_last_line"
AUDIT_GRID = "tests/fixtures/ruleforge/audit_grid.json"


def _close(figure):
    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text())


@pytest.fixture(scope="module")
def grids(repo_root):
    fixture = json.loads((repo_root / AUDIT_GRID).read_text())
    manifest = ScenarioManifest.from_dict(fixture["scenario_manifest"])
    result = {}
    for cid, row in fixture["conditions"].items():
        grid = DrawGrid(
            cid,
            "contradiction",
            manifest.scenario_ids,
            np.array([[int(v) for v in line] for line in row["outcomes"]]),
        )
        assert grid.digest == row["grid_digest"]
        result[cid] = grid
    return result


def test_design_and_campaign_provenance(payload, repo_root):
    assert payload["data_version"] == 6
    assert payload["source"] == "stochastic_campaign"
    assert payload["num_total_scenarios"] == 500
    assert payload["n_decoder_draws"] == 5
    assert payload["labels"] == [INITIAL, WINNER, RUNNER_UP]
    assert payload["transition_method"] == "independent_draw_expected_shares_v1"
    assert payload["transition_units"] == "share"
    assert payload["inference_method"] == "hierarchical_bootstrap_v2"
    assert payload["provenance"]["complete"]
    assert payload["provenance"]["temperature"] == 1
    assert (
        payload["provenance"]["scenario_manifest_sha256"]
        == "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"
    )
    assert payload["provenance"]["scenario_set"] == "frozen_500_r3.jsonl"
    # The frozen cap; the root took no retry ladder, so no amendment is disclosed beside it.
    assert payload["provenance"]["max_semantic_attempts"] == 20 and "amendments" not in payload["provenance"]
    assert payload["provenance"]["estimates"] == {
        "uri": f"{ROOT}/aggregate/estimates.json",
        "sha256": "94d9387c1b668cffdb63ca067e019eba3afe2e19ec71a02783d529e4a7a49059",
    }
    assert payload["provenance"]["campaign_root"] == ROOT
    assert table.RULEFORGE_RUNS == (ROOT,)


def test_the_audit_grid_is_the_paper_root_s(payload, repo_root):
    fixture = json.loads((repo_root / AUDIT_GRID).read_text(encoding="utf-8"))
    assert fixture["campaign_root"] == ROOT
    assert fixture["scenario_manifest"]["manifest_sha256"] == payload["provenance"]["scenario_manifest_sha256"]
    assert fixture["scenario_manifest"]["source_name"] == payload["provenance"]["scenario_set"]
    for row in payload["rulesets"].values():
        assert fixture["conditions"][row["condition_id"]]["condition_fingerprint"] == row["condition_fingerprint"]


def test_rates_and_intervals_reproduce_from_frozen_draws(payload, grids):
    boot = hierarchical_bootstrap(
        grids,
        n_resamples=payload["num_bootstrap"],
        ci_level=payload["confidence_level"],
        seed=payload["random_state"],
    )
    assert [
        payload["rulesets"][label]["mean"] for label in payload["labels"]
    ] == pytest.approx([0.7268, 0.5888, 0.5520], abs=1e-12)
    for row in payload["rulesets"].values():
        cid = row["condition_id"]
        assert row["grid_digest"] == grids[cid].digest
        assert row["mean"] == pytest.approx(grids[cid].rate, abs=1e-12)
        for key in ("ci_low", "ci_high"):
            assert row[key] == pytest.approx(boot["conditions"][cid][key], abs=1e-12)


def test_paired_inference_reproduces_with_holm_over_three(payload, grids):
    rows = []
    labels = payload["labels"]
    for i, before in enumerate(labels):
        for after in labels[i + 1 :]:
            a = grids[payload["rulesets"][before]["condition_id"]]
            b = grids[payload["rulesets"][after]["condition_id"]]
            boot = hierarchical_bootstrap(
                {"before": a, "after": b},
                contrasts=[("after", "before")],
                n_resamples=10000,
                ci_level=0.95,
                seed=16,
            )
            test = clustered_paired_randomization_p(
                a, b, n_permutations=100000, seed=16
            )
            actual = payload["paired_deltas"][f"{after} - {before}"]
            expected = boot["contrasts"][0]
            assert actual["mean"] == pytest.approx(expected["estimate"], abs=1e-12)
            for key in ("ci_low", "ci_high"):
                assert actual[key] == pytest.approx(expected[key], abs=1e-12)
            assert actual["p_value"] == test["p_value"]
            rows.append({**test, "actual": actual})
    for row in holm_adjust(rows):
        assert row["actual"]["p_holm"] == row["p_holm"]
        assert row["actual"]["family_size"] == 3
        assert (
            row["actual"]["test_method"] == "scenario_clustered_paired_randomization_v1"
        )


def test_expected_transitions_reproduce_and_reconcile_with_deltas(payload, grids):
    expected = {
        f"{WINNER} - {INITIAL}": (0.17848, 0.04048),
        f"{RUNNER_UP} - {INITIAL}": (0.21448, 0.03968),
        f"{RUNNER_UP} - {WINNER}": (0.15408, 0.11728),
    }
    labels = payload["labels"]
    for i, before in enumerate(labels):
        for after in labels[i + 1 :]:
            key = f"{after} - {before}"
            row = payload["paired_transitions"][key]
            a, b = (
                grids[payload["rulesets"][label]["condition_id"]]
                for label in (before, after)
            )
            assert row == pytest.approx(expected_transition_shares(a, b), abs=1e-12)
            assert (row["fixed"], row["regressed"]) == pytest.approx(
                expected[key], abs=1e-12
            )
            assert sum(row.values()) == pytest.approx(1)
            assert row["regressed"] - row["fixed"] == pytest.approx(
                payload["paired_deltas"][key]["mean"]
            )


def test_standalone_panels_render_without_titles_or_integer_transition_counts(
    payload, repo_root, tmp_path
):
    plot = importlib.import_module("ruleforge.experiments.ruleforge.plot")
    for panel in ("a", "b"):
        fig = plot.build_figure(payload, panel=panel)
        assert len(fig.axes) == 1 and not fig.texts and fig._suptitle is None
        ax = fig.axes[0]
        assert ax.get_title(loc="left") == ""
        if panel == "a":
            assert [p.get_height() for p in ax.patches] == pytest.approx(
                [0.7268, 0.5888, 0.5520]
            )
            assert [t.get_text() for t in ax.get_xticklabels()] == [INITIAL, *REVISED]
        else:
            assert [t.get_text() for t in ax.get_yticklabels()] == list(REVISED)
            assert [p.get_width() for p in ax.patches] == pytest.approx(
                [-0.17848, -0.21448, 0.04048, 0.03968]
            )
            assert {"17.8%", "21.4%", "4.0%"} <= {
                t.get_text() for t in ax.texts
            }
            assert ax.get_xlabel() == "Expected share per draw"
        _close(fig)
        path = tmp_path / f"figure4{panel}.pdf"
        plot.draw(payload, path, panel=panel)
        assert b"/Type3" not in path.read_bytes()
        assert b"FontFile2" in path.read_bytes()


def test_each_panel_draws_its_own_data_alone(payload):
    plot = importlib.import_module("ruleforge.experiments.ruleforge.plot")
    for panel, other in (("a", "paired_transitions"), ("b", "rulesets")):
        fig = plot.build_figure({key: value for key, value in payload.items() if key != other}, panel=panel)
        assert len(fig.axes) == 1 and tuple(fig.get_size_inches()) == (5.5, 4.0)
        _close(fig)


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_campaign_estimates(payload, repo_root, tmp_path, monkeypatch):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    estimates = resolve(payload["provenance"]["estimates"]["uri"])
    if not estimates.is_file():
        pytest.skip("the RuleForge campaign root is not on this machine")
    driver = importlib.import_module("ruleforge.experiments.ruleforge.build")
    rebuilt = tmp_path / "ruleforge.json"
    monkeypatch.setattr("sys.argv", ["build_ruleset_campaign.py", "--estimates", str(estimates), "--golden", str(rebuilt)])
    driver.main()
    assert rebuilt.read_bytes() == (repo_root / GOLDEN).read_bytes()
