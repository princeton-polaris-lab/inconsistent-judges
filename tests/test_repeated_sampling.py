from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ruleforge.evaluate import inference as rs
from ruleforge.evaluate.inference import (
    DrawGrid,
    RepeatedSamplingError,
    clustered_paired_randomization_p,
    grid_from_cells,
    hierarchical_bootstrap,
    holm_adjust,
    load_condition_outcomes,
    outcome_grid,
    rule_grids,
)

IDS = ("s0", "s1", "s2", "s3")


def grid(values, condition_id="c", quantity="contradiction", ids=IDS):
    return DrawGrid(condition_id, quantity, ids, np.array(values, dtype=float))


def test_the_point_estimate_is_the_mean_of_scenario_means():
    g = grid([[1, 0], [1, 1], [0, 0], [0, 1]])
    assert g.n_scenarios == 4 and g.n_draws == 2
    assert list(g.scenario_means) == [0.5, 1.0, 0.0, 0.5]
    assert g.rate == 0.5 == g.values.mean()


def test_the_digest_covers_ids_draws_values_and_identity():
    base = grid([[1, 0], [1, 1], [0, 0], [0, 1]])
    assert base.digest == grid([[1, 0], [1, 1], [0, 0], [0, 1]]).digest
    assert base.digest != grid([[1, 0], [1, 1], [0, 0], [1, 1]]).digest
    assert base.digest != grid([[1, 0], [1, 1], [0, 0], [0, 1]], condition_id="d").digest
    assert base.digest != grid([[1, 0], [1, 1], [0, 0], [0, 1]], quantity="q").digest
    assert base.digest != grid([[1, 0], [1, 1], [0, 0], [0, 1]], ids=("a", "b", "c", "d")).digest


@pytest.mark.parametrize(
    ("values", "ids", "message"),
    [
        ([1, 0, 1, 0], IDS, "must be"),
        ([[1], [0]], IDS, "scenario ids for 2 rows"),
        (np.zeros((0, 2)), (), "scenario ids for 0 rows"),
        (np.zeros((4, 0)), IDS, "at least one draw"),
        ([[1], [0], [1], [0]], ("a", "a", "b", "c"), "duplicate scenario ids"),
        ([[1, np.nan], [0, 0], [1, 1], [0, 0]], IDS, "missing"),
    ],
)
def test_a_grid_that_is_not_dense_and_validated_is_refused(values, ids, message):
    with pytest.raises(RepeatedSamplingError, match=message):
        DrawGrid("c", "q", ids, np.array(values, dtype=float))


def test_grids_are_not_mappings_so_legacy_scenario_keyed_code_cannot_take_them():
    g = grid([[1, 0], [1, 1], [0, 0], [0, 1]])
    with pytest.raises((TypeError, AttributeError)):
        set(g)  # type: ignore[arg-type]


def test_restricted_to_reorders_and_refuses_a_missing_id():
    g = grid([[1, 0], [1, 1], [0, 0], [0, 1]])
    sub = g.restricted_to(("s2", "s0"))
    assert sub.scenario_ids == ("s2", "s0") and sub.values.tolist() == [[0, 0], [1, 0]]
    with pytest.raises(RepeatedSamplingError, match="absent from the grid"):
        g.restricted_to(("s9",))


def test_grid_from_cells_refuses_holes_and_extras():
    cells = {(sid, d): float(i + d) for i, sid in enumerate(IDS) for d in range(2)}
    built = grid_from_cells("c", "q", IDS, 2, cells)
    assert built.values.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4]]
    del cells[("s1", 1)]
    with pytest.raises(RepeatedSamplingError, match="1 missing, 0 unexpected"):
        grid_from_cells("c", "q", IDS, 2, cells)
    cells[("s1", 1)] = 0.0
    cells[("zz", 0)] = 1.0
    with pytest.raises(RepeatedSamplingError, match="0 missing, 1 unexpected"):
        grid_from_cells("c", "q", IDS, 2, cells)


def write_condition(root: Path, *, draws=2, state="complete", ids=IDS, fingerprint="fp",
                    manifest="m", forced=True, mutate=None):
    for draw in range(draws):
        draw_dir = root / f"draw-{draw:03d}"
        draw_dir.mkdir(parents=True)
        completion = {
            "state": state, "decoder_draw_id": draw, "condition_fingerprint": fingerprint,
            "scenario_manifest_sha256": manifest,
        }
        rows = []
        for i, sid in enumerate(ids):
            row = {
                "scenario_id": sid, "decoder_draw_id": draw,
                "contradiction": float((i + draw) % 2), "penalty": 0.5,
                "rule_judgments": {"Rule 1": "yes" if i % 2 else "no", "Rule 2": "no"},
                "in_context_rule_judgments": {"Rule 1": "yes", "Rule 2": "no"} if forced else None,
                "in_context_contradiction": 1.0 if forced else None,
            }
            rows.append(row)
        if mutate is not None:
            mutate(draw, completion, rows)
        (draw_dir / "completion.json").write_text(json.dumps(completion))
        (draw_dir / "bundle_outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_a_complete_condition_loads_into_grids(tmp_path):
    root = write_condition(tmp_path / "c")
    with (root / "draw-000" / "bundle_outcomes.jsonl").open("a") as handle:
        handle.write("\n")
    cells, provenance = load_condition_outcomes(root, expected_scenario_ids=IDS, expected_draws=2, scenario_manifest_sha256="m")
    assert len(cells) == 8 and provenance["condition_fingerprint"] == "fp"
    assert provenance["n_draws"] == 2 and len(provenance["draws"]) == 2
    g = outcome_grid("c", cells, scenario_ids=IDS, n_draws=2)
    assert g.values.tolist() == [[0, 1], [1, 0], [0, 1], [1, 0]]
    b = outcome_grid("c", cells, scenario_ids=IDS, n_draws=2, quantity="in_context_contradiction")
    assert b.rate == 1.0
    alone = rule_grids("c", cells, scenario_ids=IDS, n_draws=2, rule_id="Rule 1")
    in_context = rule_grids("c", cells, scenario_ids=IDS, n_draws=2, rule_id="Rule 1", field="in_context_rule_judgments")
    assert alone.values[:, 0].tolist() == [0, 1, 0, 1] and in_context.rate == 1.0
    with pytest.raises(RepeatedSamplingError, match="carries no rule_judgments\\['Rule 9'\\]"):
        rule_grids("c", cells, scenario_ids=IDS, n_draws=2, rule_id="Rule 9")


def test_a_label_only_condition_refuses_the_in_context_grid(tmp_path):
    root = write_condition(tmp_path / "c", forced=False)
    cells, _ = load_condition_outcomes(root, expected_scenario_ids=IDS, expected_draws=2)
    with pytest.raises(RepeatedSamplingError, match="does not carry that quantity"):
        outcome_grid("c", cells, scenario_ids=IDS, n_draws=2, quantity="in_context_contradiction")
    with pytest.raises(RepeatedSamplingError, match="carries no in_context_rule_judgments"):
        rule_grids("c", cells, scenario_ids=IDS, n_draws=2, rule_id="Rule 1", field="in_context_rule_judgments")


def test_loading_refuses_every_way_a_condition_can_be_incomplete(tmp_path):
    with pytest.raises(RepeatedSamplingError, match="expected draw directories"):
        load_condition_outcomes(write_condition(tmp_path / "a", draws=1), expected_scenario_ids=IDS, expected_draws=2)
    with pytest.raises(RepeatedSamplingError, match="not 'complete'"):
        load_condition_outcomes(write_condition(tmp_path / "b", state="in_progress"), expected_scenario_ids=IDS, expected_draws=2)

    def wrong_draw(draw, completion, rows):
        completion["decoder_draw_id"] = 9
    with pytest.raises(RepeatedSamplingError, match="is for draw 9"):
        load_condition_outcomes(write_condition(tmp_path / "c", mutate=wrong_draw), expected_scenario_ids=IDS, expected_draws=2)
    with pytest.raises(RepeatedSamplingError, match="not the campaign's"):
        load_condition_outcomes(write_condition(tmp_path / "d"), expected_scenario_ids=IDS, expected_draws=2, scenario_manifest_sha256="other")

    def duplicate(draw, completion, rows):
        rows.append(dict(rows[0]))
    with pytest.raises(RepeatedSamplingError, match="appears twice"):
        load_condition_outcomes(write_condition(tmp_path / "e", mutate=duplicate), expected_scenario_ids=IDS, expected_draws=2)

    def foreign_row(draw, completion, rows):
        rows[0]["decoder_draw_id"] = 5
    with pytest.raises(RepeatedSamplingError, match="claims draw 5"):
        load_condition_outcomes(write_condition(tmp_path / "f", mutate=foreign_row), expected_scenario_ids=IDS, expected_draws=2)

    def short(draw, completion, rows):
        rows.pop()
    with pytest.raises(RepeatedSamplingError, match="1 missing, 0 extra"):
        load_condition_outcomes(write_condition(tmp_path / "g", mutate=short), expected_scenario_ids=IDS, expected_draws=2)

    def drift(draw, completion, rows):
        completion["condition_fingerprint"] = f"fp{draw}"
    with pytest.raises(RepeatedSamplingError, match="disagree on the condition fingerprint"):
        load_condition_outcomes(write_condition(tmp_path / "h", mutate=drift), expected_scenario_ids=IDS, expected_draws=2)


def test_loading_refuses_unreadable_completion_records(tmp_path):
    root = write_condition(tmp_path / "c")
    (root / "draw-000" / "completion.json").write_text("{")
    with pytest.raises(RepeatedSamplingError, match="cannot read"):
        load_condition_outcomes(root, expected_scenario_ids=IDS, expected_draws=2)
    (root / "draw-000" / "completion.json").write_text("[]")
    with pytest.raises(RepeatedSamplingError, match="not a JSON object"):
        load_condition_outcomes(root, expected_scenario_ids=IDS, expected_draws=2)


def random_grid(rng, n=30, s=5, p=0.4, condition_id="c", quantity="contradiction"):
    # Scenario-level difficulty plus decoder noise, so both components exist.
    difficulty = rng.uniform(0.05, 0.95, size=n)
    values = (rng.uniform(size=(n, s)) < difficulty[:, None]).astype(float)
    ids = tuple(f"s{i}" for i in range(n))
    return DrawGrid(condition_id, quantity, ids, values)


def test_estimates_intervals_and_decomposition_have_the_expected_shape():
    rng = np.random.default_rng(0)
    a = random_grid(rng, condition_id="a")
    b = random_grid(rng, condition_id="b")
    result = hierarchical_bootstrap({"a": a, "b": b}, contrasts=[("a", "b")], n_resamples=600, seed=3)
    assert result["method"] == "hierarchical_bootstrap_v2"
    assert result["resampling"]["n_resamples"] == 600 and result["resampling"]["seed"] == 3
    for name, g in (("a", a), ("b", b)):
        row = result["conditions"][name]
        assert row["estimate"] == g.rate
        assert row["ci_low"] <= row["estimate"] <= row["ci_high"]
        assert row["n_scenarios"] == 30 and row["n_decoder_draws"] == 5
        assert row["grid_digest"] == g.digest
        assert row["sd"] > 0 and row["two_stage"]["sd"] > 0
        assert row["interval_basis"] == "scenario_cluster_bootstrap"
        assert row["variance_components"]["within_draw"] > 0
        assert row["two_stage"]["ci_low"] <= row["estimate"] <= row["two_stage"]["ci_high"]
    contrast = result["contrasts"][0]
    assert contrast["estimate"] == pytest.approx(a.rate - b.rate)
    assert contrast["inner_resample"] == "independent"
    assert contrast["ci_low"] <= contrast["estimate"] <= contrast["ci_high"]


def test_a_constant_paired_difference_has_a_zero_width_scenario_component():
    rng = np.random.default_rng(1)
    a = random_grid(rng, condition_id="a")
    values = a.values.copy()
    values[:, 0] = 1.0 - values[:, 0]
    b = DrawGrid("b", "contradiction", a.scenario_ids, values)
    result = hierarchical_bootstrap({"a": a, "b": b}, contrasts=[("a", "b")], n_resamples=200)
    contrast = result["contrasts"][0]
    assert contrast["sd"] != 0.0
    shifted = DrawGrid("b", "contradiction", a.scenario_ids, a.values + 0.25)
    result = hierarchical_bootstrap({"a": a, "b": shifted}, contrasts=[("a", "b")], n_resamples=200)
    contrast = result["contrasts"][0]
    assert contrast["estimate"] == pytest.approx(-0.25)
    assert contrast["sd"] == pytest.approx(0.0, abs=1e-12)
    assert contrast["se_analytic"] == pytest.approx(0.0, abs=1e-12)
    assert contrast["two_stage"]["sd"] > 0


def test_same_artifact_pairs_share_the_inner_resample_and_independent_ones_do_not():
    rng = np.random.default_rng(2)
    a = random_grid(rng, condition_id="a")
    shifted = DrawGrid("b", "contradiction", a.scenario_ids, a.values + 0.25)
    joint = hierarchical_bootstrap(
        {"a": a, "b": shifted}, contrasts=[("a", "b")], shared_inner=[{"a", "b"}], n_resamples=200
    )
    independent = hierarchical_bootstrap({"a": a, "b": shifted}, contrasts=[("a", "b")], n_resamples=200)
    assert joint["contrasts"][0]["inner_resample"] == "joint"
    assert joint["contrasts"][0]["two_stage"]["sd"] == pytest.approx(0.0, abs=1e-12)
    assert joint["contrasts"][0]["ci_low"] == pytest.approx(-0.25) == joint["contrasts"][0]["ci_high"]
    assert independent["contrasts"][0]["two_stage"]["sd"] > 0
    assert independent["contrasts"][0]["sd"] == pytest.approx(joint["contrasts"][0]["sd"], abs=1e-12)
    assert joint["resampling"]["shared_inner"] == [["a", "b"]]


def test_every_condition_sees_the_same_scenario_positions():
    rng = np.random.default_rng(4)
    a = random_grid(rng, condition_id="a")
    twin = DrawGrid("twin", "contradiction", a.scenario_ids, a.values.copy())
    result = hierarchical_bootstrap(
        {"a": a, "twin": twin}, contrasts=[("a", "twin")], shared_inner=[["a", "twin"]], n_resamples=100
    )
    contrast = result["contrasts"][0]
    assert contrast["ci_low"] == contrast["ci_high"] == 0.0
    assert contrast["sd"] == 0.0 and contrast["two_stage"]["sd"] == 0.0


def test_a_grid_without_decoder_variation_has_a_zero_decoder_component():
    values = np.tile(np.array([[1.0], [0.0], [1.0], [0.0]]), (1, 3))
    g = DrawGrid("c", "q", IDS, values)
    result = hierarchical_bootstrap({"c": g}, n_resamples=100)
    row = result["conditions"]["c"]
    assert row["variance_components"]["within_draw"] == 0.0
    assert row["variance_components"]["decoder_share_of_variance"] == 0.0
    assert row["sd"] > 0
    assert row["two_stage"]["sd"] == pytest.approx(row["sd"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grids": {}}, "no grids"),
        ({"grids": {"a": grid([[1], [0], [1], [0]])}}, "1 decoder draw"),
        ({"grids": {"a": grid([[1, 0]] * 4), "b": grid([[1, 0]] * 4, ids=("a", "b", "c", "d"))}}, "scenario ids differ"),
        ({"grids": {"a": grid([[1, 0]] * 4)}, "contrasts": [("a", "zz")]}, "unknown grid"),
        ({"grids": {"a": grid([[1, 0]] * 4)}, "shared_inner": [["a", "zz"]]}, "unknown grid"),
        ({"grids": {"a": grid([[1, 0]] * 4), "b": grid([[1, 0, 1]] * 4)}, "shared_inner": [["a", "b"]]}, "mixes draw counts"),
        ({"grids": {"a": grid([[1, 0]] * 4)}, "n_resamples": 5}, "n_resamples"),
    ],
)
def test_the_bootstrap_refuses_designs_it_cannot_honour(kwargs, message):
    kwargs = {"n_resamples": 100, **kwargs}
    with pytest.raises((RepeatedSamplingError, ValueError), match=message):
        hierarchical_bootstrap(**kwargs)


def test_no_difference_gives_p_one_and_a_consistent_difference_gives_a_small_p():
    same = grid([[1, 0], [1, 1], [0, 0], [0, 1]])
    assert clustered_paired_randomization_p(same, same, n_permutations=200)["p_value"] == 1.0
    ids = tuple(f"s{i}" for i in range(12))
    ones = DrawGrid("a", "q", ids, np.ones((12, 3)))
    zeros = DrawGrid("b", "q", ids, np.zeros((12, 3)))
    result = clustered_paired_randomization_p(ones, zeros, n_permutations=2500, seed=1)
    assert result["observed_difference"] == 1.0
    assert result["p_value"] < 0.01
    assert result["method"] == "scenario_clustered_paired_randomization_v1"
    assert result["n_permutations"] == 2500 and result["n_scenarios"] == 12


def test_the_test_sees_scenario_means_not_flattened_cells():
    rng = np.random.default_rng(5)
    ids = tuple(f"s{i}" for i in range(20))
    a_means = rng.integers(0, 2, size=20).astype(float)
    b_means = rng.integers(0, 2, size=20).astype(float)
    a5 = DrawGrid("a", "q", ids, np.tile(a_means[:, None], (1, 5)))
    b5 = DrawGrid("b", "q", ids, np.tile(b_means[:, None], (1, 5)))
    a1 = DrawGrid("a", "q", ids, a_means[:, None])
    b1 = DrawGrid("b", "q", ids, b_means[:, None])
    p5 = clustered_paired_randomization_p(a5, b5, n_permutations=1000, seed=9)["p_value"]
    p1 = clustered_paired_randomization_p(a1, b1, n_permutations=1000, seed=9)["p_value"]
    assert p5 == p1


def test_the_test_refuses_misaligned_grids_and_too_few_permutations():
    a = grid([[1, 0]] * 4)
    with pytest.raises(RepeatedSamplingError, match="same scenarios"):
        clustered_paired_randomization_p(a, grid([[1, 0]] * 4, ids=("a", "b", "c", "d")))
    with pytest.raises(RepeatedSamplingError, match="at least 100"):
        clustered_paired_randomization_p(a, a, n_permutations=10)


def test_holm_adjust_stamps_the_test_method():
    rows = holm_adjust([{"rule": 1, "p_value": 0.001}, {"rule": 2, "p_value": 0.5}])
    assert rows[0]["p_holm"] == pytest.approx(0.002) and rows[0]["significant_holm"] is True
    assert rows[1]["significant_holm"] is False
    assert all(row["test_method"] == rs.TEST_METHOD for row in rows)
    assert "p_adjusted" not in rows[0] and "adjustment_method" not in rows[0]
    assert holm_adjust([]) == []
    bootstrap = holm_adjust([{"p_value": 0.01}], test_method=rs.BOOTSTRAP_TEST_METHOD)
    assert bootstrap[0]["test_method"] == "scenario_cluster_bootstrap_percentile_test_v1"


def test_overlapping_shared_inner_groups_merge_transitively():
    rng = np.random.default_rng(11)
    a = random_grid(rng, condition_id="a")
    b = DrawGrid("b", "q", a.scenario_ids, a.values.copy())
    c = DrawGrid("c", "q", a.scenario_ids, a.values.copy())
    for groups in ([["a", "b"], ["b", "c"]], [["b", "c"], ["a", "b"]]):
        result = hierarchical_bootstrap(
            {"a": a, "b": b, "c": c}, contrasts=[("a", "c")], shared_inner=groups, n_resamples=100
        )
        assert result["contrasts"][0]["inner_resample"] == "joint"
        assert result["contrasts"][0]["two_stage"]["sd"] == 0.0


def test_variance_components_follow_the_one_way_random_effects_formulas():
    from ruleforge.evaluate.inference import variance_components

    values = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    parts = variance_components(values, projected_n=500)
    within = (1 / 3 + 0 + 0) / 3                       # mean sample variance across draws
    between = 0.25 - within / 4                        # var of means (0.5, 1, 0) less its decoder share
    assert parts["within_draw"] == pytest.approx(within)
    assert parts["between_scenario"] == pytest.approx(between)
    assert parts["icc"] == pytest.approx(between / (between + within))
    assert parts["se_analytic"] == pytest.approx(np.sqrt(between / 3 + within / (3 * 4)))
    assert parts["decoder_share_of_variance"] == pytest.approx((within / 4) / (between + within / 4))
    projection = parts["se_projection"]
    assert projection["n_scenarios"] == 500
    by_draws = [projection["by_draws"][k] for k in ("1", "2", "5", "10", "20")]
    assert by_draws == sorted(by_draws, reverse=True)
    assert projection["infinite_draws"] == pytest.approx(np.sqrt(between / 500))
    assert by_draws[-1] > projection["infinite_draws"]
    assert by_draws[2] == pytest.approx(np.sqrt(between / 500 + within / (500 * 5)))


def test_variance_components_clip_a_negative_between_estimate_and_flag_a_constant_grid():
    from ruleforge.evaluate.inference import variance_components

    rng = np.random.default_rng(5)
    noise = (rng.uniform(size=(40, 5)) < 0.5).astype(float)
    parts = variance_components(noise)
    assert parts["between_scenario"] >= 0.0 and parts["within_draw"] > 0
    assert parts["se_projection"]["infinite_draws"] == pytest.approx(np.sqrt(parts["between_scenario"] / 500))
    constant = variance_components(np.ones((3, 2)))
    assert constant["icc"] is None and constant["decoder_share_of_variance"] is None
    with pytest.raises(RepeatedSamplingError, match="at least two"):
        variance_components(np.ones((1, 3)))
    with pytest.raises(RepeatedSamplingError, match="at least two"):
        variance_components(np.ones((3, 1)))


def test_the_two_stage_interval_is_the_conservative_one_on_a_noisy_grid():
    """Pure decoder noise: the two-stage resample over-counts it by (2S-1)/S."""
    rng = np.random.default_rng(9)
    noise = (rng.uniform(size=(200, 5)) < 0.5).astype(float)
    g = DrawGrid("c", "q", tuple(f"s{i}" for i in range(200)), noise)
    row = hierarchical_bootstrap({"c": g}, n_resamples=2000, seed=1)["conditions"]["c"]
    assert row["two_stage"]["sd"] > row["sd"]
    assert row["two_stage"]["sd"] / row["sd"] == pytest.approx(np.sqrt(9 / 5), rel=0.15)
    assert row["sd"] == pytest.approx(row["variance_components"]["se_analytic"], rel=0.1)
