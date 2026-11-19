from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
import pytest

from ruleforge.evaluate.bank import analyze_shared_bank_campaign, default_contrasts, max_rule_median
from ruleforge.evaluate.bootstrap import percentile_bounds
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME
from ruleforge.evaluate.inference import (
    CURVE_INFERENCE_METHOD,
    CURVE_INTERVAL_BASIS,
    RepeatedSamplingError,
    bank_shared_curve_bootstrap,
    crossed_cluster_indices,
    crossed_cluster_means,
    load_condition_outcomes,
)
from ruleforge.evaluate.scoring import PHI_CLASSES, PHI_MAP, calculate_inconsistency
from ruleforge.experiments.ruleset_size import build_repeated_draw_baseline
from ruleforge.experiments.ruleset_size import repeated_draw_baseline as rdb
from ruleforge.experiments.ruleset_size.figure import SERIES_COLORS
from ruleforge.scenarios.manifest import ScenarioManifest
from tests.test_sampling_ruleset_size import _tamper, run_fake_campaign

LABELS = {value: label for label, value in PHI_MAP.items()}


def _phi(rng, shape):
    return rng.choice(PHI_CLASSES, size=shape)


def _brute_force(phi):
    *cells, n_draws = phi.shape
    out = np.empty(cells)
    pairs = [(a, b) for a in range(n_draws) for b in range(n_draws) if a != b]
    for index in itertools.product(*map(range, cells)):
        out[index] = sum(phi[index][a] != phi[index][b] for a, b in pairs) / len(pairs)
    return out


@pytest.mark.parametrize("n_draws", [2, 3, 5])
def test_the_repeat_disagreement_is_the_mean_over_ordered_pairs_of_distinct_draws(n_draws):
    phi = _phi(np.random.default_rng(n_draws), (3, 4, 2, n_draws))
    assert np.array_equal(rdb.repeat_disagreement(phi), _brute_force(phi))
    # One label throughout agrees in every pair; two labels split evenly over two draws disagree in both.
    assert rdb.repeat_disagreement(np.zeros((1, 3))).tolist() == [0.0]
    assert rdb.repeat_disagreement(np.array([[0.0, 1.0]])).tolist() == [1.0]


def test_the_inconsistency_cell_is_the_mean_of_the_bank_s_contradiction_over_draws():
    rng = np.random.default_rng(4)
    shape, n_rules = (2, 6, 3, 4), 3
    full_labels = rng.choice(list(PHI_MAP), size=shape)
    rule_labels = rng.choice(list(PHI_MAP), size=(*shape, n_rules))
    full = np.vectorize(PHI_MAP.get)(full_labels)
    rule_phi = np.vectorize(PHI_MAP.get)(rule_labels)
    separate = rule_phi.max(axis=-1)
    contradiction = np.empty(shape)
    for index in itertools.product(*map(range, shape)):
        contradiction[index] = calculate_inconsistency(str(full_labels[index]),
                                                       [str(label) for label in rule_labels[index]])["contradiction"]
    # Each rule's upper median over the four draws, then the strongest rule's, cell by cell.
    median = np.empty(shape[:3])
    for index in itertools.product(*map(range, shape[:3])):
        median[index] = max(sorted(rule_phi[index][:, rule])[2] for rule in range(n_rules))
    cells, observed = rdb.cell_statistics(full, separate, median)
    assert sorted(cells) == sorted(observed) == sorted(rdb.BASE)
    assert np.array_equal(cells["inconsistency"], contradiction.mean(axis=3))
    assert np.array_equal(observed["inconsistency"], contradiction.mean(axis=(1, 2, 3)))
    assert np.array_equal(cells["lenient"] + cells["strict"], cells["inconsistency"])
    assert np.array_equal(cells["full_repeat"], _brute_force(full))
    assert np.array_equal(cells["separate_repeat"], _brute_force(separate))
    assert np.array_equal(cells["rule_median_inconsistency"], (full != median[..., None]).mean(axis=3))
    assert np.array_equal(observed["rule_median_inconsistency"], (full != median[..., None]).mean(axis=(1, 2, 3)))
    # The full-set verdict's upper median over the four draws against the same strongest rule's, cell by cell.
    full_median = np.empty(shape[:3])
    for index in itertools.product(*map(range, shape[:3])):
        full_median[index] = sorted(full[index])[2]
    assert np.array_equal(cells["median_inconsistency"], (full_median != median).astype(float))
    assert np.array_equal(observed["median_inconsistency"], (full_median != median).mean(axis=(1, 2)))


def _cells(full, separate):
    return rdb.cell_statistics(full, separate, np.zeros(full.shape[:3]))


def _null(rng, shape, *, shift=0.0):
    """F and M drawn independently per draw from one distribution per cell; ``shift`` moves F's toward yes."""
    probabilities = rng.dirichlet(np.ones(3), size=shape[:-1])
    moved = np.clip(probabilities + np.array([-shift, 0.0, shift]), 0.0, None)
    moved /= moved.sum(axis=-1, keepdims=True)

    def draw(p):
        cumulative = p.cumsum(axis=-1)[..., None, :]
        return np.asarray(PHI_CLASSES)[(rng.random((*shape, 1)) > cumulative).sum(axis=-1)]

    return draw(moved), draw(probabilities), moved, probabilities


#: The null simulation's tolerance: its mean excess is a mean of 40,000 cells, each within [-1, 1], whose standard
#: error is about 0.002.
NULL_TOLERANCE = 0.01


def test_under_the_null_the_excess_is_zero_and_the_plug_in_baseline_is_biased_low():
    full, separate, _, probabilities = _null(np.random.default_rng(7), (2, 1000, 20, 5))
    cells, _ = _cells(full, separate)
    baseline = (cells["full_repeat"] + cells["separate_repeat"]) / 2
    assert abs((cells["inconsistency"] - baseline).mean()) < NULL_TOLERANCE
    assert abs((cells["lenient"] - cells["strict"]).mean()) < NULL_TOLERANCE

    def plug_in(phi):
        shares = np.stack([(phi == value).mean(axis=-1) for value in PHI_CLASSES])
        return 1.0 - (shares ** 2).sum(axis=0)

    # The plug-in baseline has expectation w (D-1)/D, so its excess is about w/D: well above the tolerance.
    plug_in_excess = cells["inconsistency"] - (plug_in(full) + plug_in(separate)) / 2
    w = 1.0 - (probabilities ** 2).sum(axis=-1)
    assert plug_in_excess.mean() > 10 * NULL_TOLERANCE
    assert plug_in_excess.mean() == pytest.approx(w.mean() / 5, abs=NULL_TOLERANCE)


def test_off_the_null_the_excess_is_half_the_squared_distance_between_the_two_distributions():
    full, separate, moved, probabilities = _null(np.random.default_rng(8), (2, 1000, 20, 5), shift=0.3)
    cells, _ = _cells(full, separate)
    excess = cells["inconsistency"] - (cells["full_repeat"] + cells["separate_repeat"]) / 2
    expected = 0.5 * ((moved - probabilities) ** 2).sum(axis=-1)
    assert excess.mean() == pytest.approx(expected.mean(), abs=NULL_TOLERANCE) and expected.mean() > 0.02
    assert (cells["lenient"] - cells["strict"]).mean() < -NULL_TOLERANCE  # F leans to yes: it exceeds M more often


NO, UNSURE, YES = (PHI_MAP[label] for label in ("no", "unsure", "yes"))


@pytest.mark.parametrize(("draws", "expected"), [
    # Rule A says yes only in draw 0 and rule B only in draw 1: each draw's maximum is yes in two draws of three, but
    # neither rule's median is.
    ([[YES, NO], [NO, YES], [NO, NO]], NO),
    ([[YES, NO], [YES, YES], [NO, NO]], YES),
    # One yes among two unsures moves a rule's median from no to unsure, never to yes.
    ([[NO], [NO], [UNSURE], [UNSURE], [YES]], UNSURE),
    ([[NO], [NO], [UNSURE], [UNSURE], [NO]], NO),
    ([[YES], [NO], [YES], [NO], [YES]], YES),
    # With an even count the median is the upper one: a yes in half the draws makes the rule's median yes.
    ([[NO], [YES]], YES),
    ([[NO], [YES], [NO], [YES]], YES),
    ([[NO], [YES], [NO], [NO]], NO),
])
def test_the_rule_median_takes_each_rule_s_median_over_the_draws_before_the_strongest_rule(draws, expected):
    rule_phi = np.array(draws, dtype=float)
    assert max_rule_median(rule_phi) == expected
    assert max_rule_median(np.stack([rule_phi, np.zeros_like(rule_phi)])).tolist() == [expected, NO]
    assert rdb.draw_median(rule_phi[:, 0]) == expected
    assert rdb.draw_median(np.stack([rule_phi[:, 0], np.zeros(len(draws))])).tolist() == [expected, NO]


def test_the_median_inconsistency_drops_a_verdict_the_full_set_gives_in_a_minority_of_its_draws():
    # The strongest rule's median is no in every cell. The full set says yes in one, two and three of five draws.
    full = np.zeros((1, 3, 1, 5))
    for scenario, n_yes in enumerate((1, 2, 3)):
        full[0, scenario, 0, :n_yes] = YES
    cells, _ = rdb.cell_statistics(full, full, np.zeros((1, 3, 1)))
    assert cells["rule_median_inconsistency"][0, :, 0].tolist() == [0.2, 0.4, 0.6]
    assert cells["median_inconsistency"][0, :, 0].tolist() == [0.0, 0.0, 1.0]


def _rare_yes(rng, sizes, p, *, n_scen=300, n_seed=4, n_draws=5):
    """Every call's modal judgment no, the full-set call always no, each separate call yes with probability ``p``."""
    full = np.zeros((len(sizes), n_scen, n_seed, n_draws))
    separate, median = np.empty(full.shape), np.empty(full.shape[:3])
    for k, size in enumerate(sizes):
        rule_phi = (rng.random((n_scen, n_seed, n_draws, size)) < p).astype(float)
        separate[k], median[k] = rule_phi.max(axis=-1), max_rule_median(rule_phi)
    return full, separate, median


def test_at_a_per_call_yes_probability_of_0_04_the_rate_rises_with_k_and_the_rule_median_rate_stays_near_zero():
    # The rate and the excess rise with K, all of it lenient; a rule's median is yes with probability about 0.0006.
    _, observed = rdb.cell_statistics(*_rare_yes(np.random.default_rng(9), (2, 10, 57), 0.04))
    baseline = (observed["full_repeat"] + observed["separate_repeat"]) / 2
    assert observed["inconsistency"][-1] > 0.8 and (observed["inconsistency"] - baseline)[-1] > 0.7
    assert np.array_equal(observed["lenient"], observed["inconsistency"])
    assert observed["rule_median_inconsistency"].max() < 0.05
    # The full set always says no, so its median is no and the two median rates agree.
    assert np.array_equal(observed["median_inconsistency"], observed["rule_median_inconsistency"])


def test_at_a_per_call_yes_probability_of_0_15_the_rule_median_rate_rises_with_k_too():
    # A rule's median is yes with probability q = P(Bin(5, 0.15) >= 3) = 0.027, so M~ is yes with 1 - (1 - q)^K: about
    # 0.05, 0.24 and 0.79 at K = 2, 10 and 57, below the rate's 0.28, 0.80 and 1.00 but rising as it does.
    sizes = (2, 10, 57)
    _, observed = rdb.cell_statistics(*_rare_yes(np.random.default_rng(10), sizes, 0.15))
    q = 1 - sum(math.comb(5, n) * 0.15 ** n * 0.85 ** (5 - n) for n in range(3))
    assert observed["rule_median_inconsistency"] == pytest.approx([1 - (1 - q) ** size for size in sizes], abs=0.03)
    assert observed["inconsistency"] == pytest.approx([1 - 0.85 ** size for size in sizes], abs=0.03)
    assert (np.diff(observed["rule_median_inconsistency"]) > 0.15).all()


SIZES = [2, 3, 5, 57]
SETTINGS = {"n_resamples": 400, "ci_level": 0.95, "seed": 16}


@pytest.fixture(scope="module")
def arrays():
    rng = np.random.default_rng(12)
    shape = (len(SIZES), 30, 4, 5)
    full = _phi(rng, shape)
    return full, np.where(rng.random(shape) < 0.3, _phi(rng, shape), full), _phi(rng, shape[:3])


@pytest.fixture(scope="module")
def joint(arrays):
    return rdb.joint_bootstrap(*arrays, sizes=SIZES, contrasts=default_contrasts(SIZES), **SETTINGS)


def test_the_joint_bootstrap_s_inconsistency_rows_are_the_curve_bootstrap_s_primary_ones(arrays, joint):
    full, separate, _ = arrays
    curve = bank_shared_curve_bootstrap((full != separate).astype(float), sizes=SIZES,
                                        contrasts=default_contrasts(SIZES), **SETTINGS)
    points, contrasts = joint["inconsistency"]["points"], joint["inconsistency"]["contrasts"]
    assert points == [{key: value for key, value in point.items() if key != "two_stage"} for point in curve["points"]]
    assert contrasts == {name: {key: value for key, value in row.items() if key != "two_stage"}
                         for name, row in curve["contrasts"].items()}
    assert list(joint) == list(rdb.STATISTICS) and list(contrasts) == ["net_rise", "last_step"]


def test_every_statistic_s_interval_is_read_from_one_set_of_replicates_and_derived_ones_replicate_by_replicate(
        arrays, joint):
    cells, observed = rdb.cell_statistics(*arrays)
    scen, seeds = crossed_cluster_indices(np.random.default_rng(SETTINGS["seed"]), SETTINGS["n_resamples"], 30, 4)
    replicates = {name: crossed_cluster_means(cells[name], scen, seeds) for name in rdb.BASE}
    baseline = (replicates["full_repeat"] + replicates["separate_repeat"]) / 2
    derived = {"repeated_draw_baseline": baseline, "excess": replicates["inconsistency"] - baseline,
               "lenient_minus_strict": replicates["lenient"] - replicates["strict"], **replicates}
    for name in rdb.STATISTICS:
        points, contrasts = joint[name]["points"], joint[name]["contrasts"]
        for index, point in enumerate(points):
            low, high = percentile_bounds(derived[name][:, index], SETTINGS["ci_level"])
            assert (point["ci_low"], point["ci_high"], point["sd"]) == (low, high, float(derived[name][:, index].std()))
        rise = derived[name][:, -1] - derived[name][:, 0]
        assert (contrasts["net_rise"]["ci_low"], contrasts["net_rise"]["ci_high"]) == percentile_bounds(rise, 0.95)
    assert [point["estimate"] for point in joint["repeated_draw_baseline"]["points"]] == list(
        (observed["full_repeat"] + observed["separate_repeat"]) / 2)
    # The excess's interval is its own replicates', not the difference of the two intervals' bounds.
    excess, rate, first = (joint[name]["points"][0]
                           for name in ("excess", "inconsistency", "repeated_draw_baseline"))
    assert excess["estimate"] == pytest.approx(rate["estimate"] - first["estimate"])
    assert excess["ci_low"] != pytest.approx(rate["ci_low"] - first["ci_low"])


def test_the_joint_bootstrap_refuses_arrays_of_different_shapes_and_a_rule_median_cell_no_row_fills(arrays):
    full, separate, median = arrays
    with pytest.raises(RepeatedSamplingError, match="the full-set array is"):
        rdb.joint_bootstrap(full, separate[:, :, :, :4], median, sizes=SIZES, contrasts=(), **SETTINGS)
    with pytest.raises(RepeatedSamplingError, match=r"and its rule-median array \(4, 30, 3\)"):
        rdb.joint_bootstrap(full, separate, median[:, :, :3], sizes=SIZES, contrasts=(), **SETTINGS)
    holed = median.copy()
    holed[1, 2, 0] = np.nan
    with pytest.raises(RepeatedSamplingError, match="a rule-median cell is missing"):
        rdb.joint_bootstrap(full, separate, holed, sizes=SIZES, contrasts=(), **SETTINGS)


MODEL = "gemma-4-31B-it"


def _rule_ids(root, condition_id):
    path = root / condition_id / "draw-000" / "bundle_outcomes.jsonl"
    return json.loads(path.read_text().splitlines()[0])["ruleset_rule_ids"]


def _vary(root):
    """Make three cells' judgments change across draws in known directions (the fake judge answers every draw alike)."""
    ids = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME).scenario_ids
    outside = min(set(_rule_ids(root, f"{MODEL}/subsets/s00/k04")) - set(_rule_ids(root, f"{MODEL}/subsets/s00/k02")))
    _tamper(root, f"{MODEL}/subsets/s00/k02", 0, lambda rows: [
        row.__setitem__("full_set_judgment", "yes") for row in rows if row["scenario_id"] == ids[1]])
    _tamper(root, f"{MODEL}/bank", 1, lambda rows: [
        row["rule_judgments"].update(dict.fromkeys(row["rule_judgments"], "yes") if row["scenario_id"] == ids[2]
                                     else {outside: "yes"} if row["scenario_id"] == ids[1] else {})
        for row in rows])


def _expected_phi(root, n_draws=2, seeds=(0, 1), sizes=(2, 3, 4)):
    ids = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME).scenario_ids
    load = {"expected_scenario_ids": ids, "expected_draws": n_draws}
    bank, _ = load_condition_outcomes(root / f"{MODEL}/bank", **load)
    full, separate = np.empty((2, len(sizes), len(ids), len(seeds), n_draws))
    median = np.empty((len(sizes), len(ids), len(seeds)))
    for (j, seed), (k, size) in itertools.product(enumerate(seeds), enumerate(sizes)):
        rows, _ = load_condition_outcomes(root / f"{MODEL}/subsets/s{seed:02d}/k{size:02d}", **load)
        for (i, sid), draw in itertools.product(enumerate(ids), range(n_draws)):
            row = rows[(sid, draw)]
            full[k, i, j, draw] = PHI_MAP[row["full_set_judgment"]]
            separate[k, i, j, draw] = max(PHI_MAP[bank[(sid, draw)]["rule_judgments"][rule]]
                                          for rule in row["ruleset_rule_ids"])
        for i, sid in enumerate(ids):
            rules = rows[(sid, 0)]["ruleset_rule_ids"]
            median[k, i, j] = max(sorted(PHI_MAP[bank[(sid, draw)]["rule_judgments"][rule]]
                                         for draw in range(n_draws))[n_draws // 2] for rule in rules)
    return full, separate, median


def _campaign(tmp_path, modelspec, monkeypatch, *, vary):
    import ruleforge.evaluate.bank as driver

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    if vary:
        _vary(root)
    analyze_shared_bank_campaign(root)
    estimates = root / "aggregate" / "figure2_estimates.json"
    # The fake judge runs under the conditions of a paper judge, whose label the build looks up by model name.
    _rewrite(estimates, lambda payload: payload["models"]["gemma-4-31B-it"].__setitem__("model_name", "gemma-4-31B-it"))
    return root, estimates


@pytest.fixture
def campaign(tmp_path, modelspec, monkeypatch):
    """A complete fake campaign root under the artifact root and its estimates."""
    return _campaign(tmp_path, modelspec, monkeypatch, vary=False)


@pytest.fixture
def varied_campaign(tmp_path, modelspec, monkeypatch):
    """The fake campaign with three cells whose verdicts change across draws (:func:`_vary`)."""
    return _campaign(tmp_path, modelspec, monkeypatch, vary=True)


def _rewrite(path, mutate):
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_the_reader_joins_each_subset_draw_to_the_bank_draw_of_its_index(varied_campaign):
    root, estimates = varied_campaign
    design, phi = rdb.read_phi(root, MODEL, n_draws=2)
    entry = json.loads(estimates.read_text())["models"][MODEL]
    assert design.to_dict() == json.loads(estimates.read_text())["design"]
    assert phi.full.shape == phi.separate.shape == (3, 3, 2, 2) and np.isfinite(phi.full).all()
    assert phi.bank_digest == entry["bank"]["grid_digest"]
    assert np.array_equal(phi.inconsistency, phi.full != phi.separate)
    rates = phi.inconsistency.mean(axis=(1, 2, 3))
    assert rates.tolist() == [point["estimate"] for point in entry["points"]]
    full, separate, median = _expected_phi(root)
    assert np.array_equal(phi.full, full) and np.array_equal(phi.separate, separate)
    assert phi.max_rule_median.shape == (3, 3, 2) and np.array_equal(phi.max_rule_median, median)
    # The varied cells, by hand: F is the full-set verdict, M the maximum over the subset's own rules of the separate
    # calls, draw d against draw d. Size 2 at seed 0 does not judge the rule scenario 1's bank says yes to; size 4 does.
    assert (phi.full[0, 1, 0].tolist(), phi.separate[0, 1, 0].tolist()) == ([1.0, 0.0], [0.0, 0.0])
    assert (phi.full[2, 1, 0].tolist(), phi.separate[2, 1, 0].tolist()) == ([0.0, 0.0], [0.0, 1.0])
    assert {(tuple(phi.full[k, 2, j]), tuple(phi.separate[k, 2, j])) for k in range(3) for j in range(2)} == {
        ((1.0, 1.0), (0.0, 1.0))}
    # With two draws the upper median is the larger: a rule that says yes in one draw of two raises M~.
    assert (phi.max_rule_median[0, 1, 0], phi.max_rule_median[2, 1, 0], phi.max_rule_median[1, 2, 1]) == (0.0, 1.0, 1.0)
    cells, _ = rdb.cell_statistics(phi.full, phi.separate, phi.max_rule_median)
    names = ("inconsistency", "full_repeat", "separate_repeat", "lenient", "strict", "rule_median_inconsistency",
             "median_inconsistency")
    # With two draws the full set's upper median is its larger draw too.
    assert [cells[name][0, 1, 0] for name in names] == [0.5, 1.0, 0.0, 0.0, 0.5, 0.5, 1.0]
    assert [cells[name][2, 1, 0] for name in names] == [0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 1.0]
    assert [cells[name][1, 2, 1] for name in names] == [0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0]


def test_the_reader_takes_each_rule_s_median_over_three_draws_before_the_strongest_rule(tmp_path, modelspec,
                                                                                         monkeypatch):
    import ruleforge.evaluate.bank as driver

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver, **{"sampling.num_decoder_draws": 3})
    ids = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME).scenario_ids
    first, second = _rule_ids(root, f"{MODEL}/subsets/s00/k02")
    # Scenario 1's bank says yes to the first rule of seed 0, size 2 in draw 0 only and to the second in draw 1 only.
    for draw, rule in ((0, first), (1, second)):
        _tamper(root, f"{MODEL}/bank", draw, lambda rows, rule=rule: [
            row["rule_judgments"].__setitem__(rule, "yes") for row in rows if row["scenario_id"] == ids[1]])
    _, phi = rdb.read_phi(root, MODEL, n_draws=3)
    full, separate, median = _expected_phi(root, n_draws=3)
    assert np.array_equal(phi.full, full) and np.array_equal(phi.separate, separate)
    assert np.array_equal(phi.max_rule_median, median)
    # The maximum is yes in two draws of three, but neither rule's median is: M~ is no.
    assert (phi.full[0, 1, 0].tolist(), phi.separate[0, 1, 0].tolist()) == ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    assert phi.max_rule_median[0, 1, 0] == 0.0
    cells, _ = rdb.cell_statistics(phi.full, phi.separate, phi.max_rule_median)
    assert (cells["inconsistency"][0, 1, 0], cells["rule_median_inconsistency"][0, 1, 0],
            cells["median_inconsistency"][0, 1, 0]) == (2 / 3, 0.0, 0.0)


def test_the_build_holds_every_inconsistency_row_to_its_estimates_and_records_its_provenance(campaign):
    _, estimates = campaign
    payload = rdb.build_payload([estimates])
    assert (payload["analysis"], payload["schema_version"], payload["definition"]) == (
        rdb.ANALYSIS, rdb.SCHEMA_VERSION, rdb.DEFINITION)
    assert payload["statistics"] == list(rdb.STATISTICS) and payload["subset_sizes"] == [2, 3, 4]
    assert payload["bootstrap"] == {"n_resamples": 150, "ci_level": 0.95, "seed": 16,
                                    "interval_basis": "scenario_subset_cluster_bootstrap"}
    assert payload["contrasts"] == default_contrasts([2, 3, 4])
    published = json.loads(estimates.read_text())
    (judge,) = payload["judges"]
    entry = published["models"]["gemma-4-31B-it"]
    assert (judge["label"], judge["model_name"]) == ("Gemma-4-31B", "gemma-4-31B-it")
    assert judge["campaign"] == {"campaign_id": "ruleset-size", "estimates": "artifact://root/aggregate/figure2_estimates.json",
                                 "estimates_sha256": hashlib.sha256(estimates.read_bytes()).hexdigest()}
    assert judge["grid_digests"] == {"bank": entry["bank"]["grid_digest"], "subsets": entry["subsets"]["grid_digest"]}
    for point, row in zip(judge["points"], entry["points"]):
        assert point["size"] == row["size"] and list(point) == ["size", *rdb.STATISTICS]
        assert point["inconsistency"] == {key: row[key] for key in ("estimate", "ci_low", "ci_high", "sd")}
    assert list(judge["contrasts"]) == list(rdb.STATISTICS)
    # The fake judge answers each call the same way in every draw: no two calls of a cell disagree.
    assert {point[name]["estimate"] for point in judge["points"] for name in ("full_repeat", "separate_repeat")} == {0.0}
    assert all(point["excess"]["estimate"] == point["inconsistency"]["estimate"] for point in judge["points"])



def test_the_build_s_statistics_are_the_root_s_own_cells_with_the_full_set_verdict_as_f(varied_campaign):
    root, estimates = varied_campaign
    (judge,) = rdb.build_payload([estimates])["judges"]
    cells, _ = rdb.cell_statistics(*_expected_phi(root))
    for index, point in enumerate(judge["points"]):
        for name in rdb.BASE:
            assert point[name]["estimate"] == pytest.approx(cells[name][index].mean()), (point["size"], name)
    assert judge["points"][0]["full_repeat"]["estimate"] > 0 and judge["points"][0]["separate_repeat"]["estimate"] > 0
    assert judge["points"][0]["strict"]["estimate"] > 0 and judge["points"][-1]["lenient"]["estimate"] > 0


def test_the_build_reads_one_judge_in_process_and_several_in_a_pool_then_orders_them_as_the_legend(monkeypatch):
    shared = {"interval_basis": CURVE_INTERVAL_BASIS, "inference_method": CURVE_INFERENCE_METHOD, "bootstrap": {},
              "subset_sizes": [2, 3], "n_scenarios": 3, "n_subset_seeds": 2, "n_decoder_draws": 2}
    payloads = [{**shared, "models": {"b": {}, "a": {}}}, {**shared, "models": {"c": {}}}]
    labels = {"a": "Gemma-4-31B", "b": "Some judge", "c": "GPT-OSS-120B"}
    monkeypatch.setattr(rdb, "load_estimates", lambda paths, allow_incomplete: [payloads[int(p)] for p in paths])
    monkeypatch.setattr(rdb, "judge_record", lambda payload, model: {"label": labels[model]})
    pools = []

    class Pool:
        def __init__(self, processes, mp_context):
            pools.append((processes, mp_context.get_start_method()))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, *iterables):
            return map(function, *iterables)

    monkeypatch.setattr(rdb, "ProcessPoolExecutor", Pool)
    order = ["GPT-OSS-120B", "Gemma-4-31B", "Some judge"]
    assert [judge["label"] for judge in rdb.build_payload(["0", "1"], workers=8)["judges"]] == order
    assert pools == [(3, "spawn")]
    assert [judge["label"] for judge in rdb.build_payload(["0", "1"])["judges"]] == order
    assert [judge["label"] for judge in rdb.build_payload(["1"], workers=8)["judges"]] == ["GPT-OSS-120B"]
    assert len(pools) == 1


def test_the_program_uses_every_cpu_where_the_platform_cannot_say_which_it_may_run_on(monkeypatch):
    monkeypatch.delattr(build_repeated_draw_baseline.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(build_repeated_draw_baseline.os, "cpu_count", lambda: None)
    assert build_repeated_draw_baseline._usable_cpus() == 1
    monkeypatch.setattr(build_repeated_draw_baseline.os, "cpu_count", lambda: 6)
    assert build_repeated_draw_baseline._usable_cpus() == 6


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda payload: payload["models"]["gemma-4-31B-it"]["bank"].__setitem__("grid_digest", "0" * 64),
     "its bank's grid digest is not its estimates'"),
    (lambda payload: payload["models"]["gemma-4-31B-it"]["subsets"].__setitem__("grid_digest", "0" * 64),
     "its inconsistency grid's digest is not its estimates'"),
    (lambda payload: payload["design"].__setitem__("sampling_strategy", "other"), "its root's design is not"),
    (lambda payload: payload["models"]["gemma-4-31B-it"]["points"][1].__setitem__("ci_low", -1.0),
     "its inconsistency rows are not its estimates' points"),
    (lambda payload: payload["models"]["gemma-4-31B-it"]["contrasts"]["net_rise"].__setitem__("sd", 9.0),
     "its inconsistency contrasts are not its estimates'"),
    (lambda payload: payload.__setitem__("interval_basis", "other"), "their interval basis is 'other'"),
    (lambda payload: payload.__setitem__("inference_method", "other"), "their inference method is 'other'"),
])
def test_the_build_refuses_estimates_its_root_does_not_reproduce(campaign, mutate, message):
    _, estimates = campaign
    _rewrite(estimates, mutate)
    with pytest.raises(rdb.RepeatedDrawBaselineError, match=message):
        rdb.build_payload([estimates])


def test_the_reader_refuses_a_judge_its_root_does_not_hold_and_an_outcome_it_cannot_join(campaign):
    root, _ = campaign
    with pytest.raises(rdb.RepeatedDrawBaselineError, match="registers no conditions of 'claude-haiku-5-5'"):
        rdb.read_phi(root, "claude-haiku-5-5", n_draws=2)
    _tamper(root, "gemma-4-31B-it/subsets/s00/k02", 0, lambda rows: rows[2].__setitem__("ruleset_rule_ids", ["x", "y"]))
    with pytest.raises(RepeatedSamplingError, match="judged a different subset"):
        rdb.read_phi(root, "gemma-4-31B-it", n_draws=2)


def test_the_reader_refuses_a_cell_no_row_fills_rather_than_count_it_inconsistent(campaign, monkeypatch):
    import ruleforge.evaluate.bank as bank

    root, _ = campaign
    joined = bank._subset_judgments
    monkeypatch.setattr(bank, "_subset_judgments",
                        lambda *args, **kwargs: itertools.islice(joined(*args, **kwargs), 1, None))
    first = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME).scenario_ids[0]
    with pytest.raises(RepeatedSamplingError, match=f"^{MODEL}/subsets/s00/k02: {first[:12]}/draw 0 is missing$"):
        rdb.read_phi(root, MODEL, n_draws=2)


def test_the_program_builds_the_golden_csv_and_figure_and_redraws_them_from_the_golden(campaign, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    _, estimates = campaign
    out = tmp_path / "out"
    golden, table, figure = (out / f"repeated_draw_baseline.{suffix}" for suffix in ("json", "csv", "png"))
    assert build_repeated_draw_baseline.main(["--estimates", str(estimates), "--golden", str(golden),
                                              "--csv", str(table), "--figure", str(figure), "--dpi", "40"]) == 0
    written = build_repeated_draw_baseline.load_golden(golden)
    assert written == json.loads(json.dumps(rdb.build_payload([estimates]), sort_keys=True))
    assert golden.read_text() == json.dumps(written, indent=2, sort_keys=True) + "\n"
    assert capsys.readouterr().out == rdb.summary(written)
    lines = table.read_text().splitlines()
    assert lines[0] == ",".join(rdb.CSV_COLUMNS) and len(lines) == 1 + 3 and figure.stat().st_size
    again = tmp_path / "again.csv"
    assert build_repeated_draw_baseline.main(["--from_golden", str(golden), "--csv", str(again)]) == 0
    assert again.read_bytes() == table.read_bytes()


def test_the_summary_prints_each_judge_s_ends_and_net_rise(campaign):
    _, estimates = campaign
    text = rdb.summary(rdb.build_payload([estimates])).splitlines()
    assert text[0] == "Gemma-4-31B"
    assert text[1].startswith("  K=2      inconsistency ") and text[2].startswith("  K=4      inconsistency ")
    assert text[3].startswith("  net rise inconsistency +") and "excess +" in text[3]
    assert text[-1] == "1 judge(s); N=3 x J=2 x D=2; scenario_subset_cluster_bootstrap"


@pytest.mark.parametrize(("words", "message"), [
    (["--from_golden", "g.json", "--golden", "again.json"],
     "--from_golden re-renders an existing golden; do not also pass --golden"),
    (["--from_golden", "other.json"], "other.json is not a ruleset_size_repeated_draw_baseline payload of schema 1"),
    (["--estimates", "missing.json"], "No such file or directory"),
    ([], "one of the arguments --estimates --from_golden is required"),
])
def test_the_program_refuses_what_it_cannot_build_or_draw(tmp_path, monkeypatch, capsys, words, message):
    (tmp_path / "g.json").write_text(json.dumps(_payload(["Gemma-4-31B"])))
    (tmp_path / "other.json").write_text(json.dumps({"analysis": "hierarchy", "schema_version": 1}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        build_repeated_draw_baseline.main(words)
    assert exit_info.value.code == 2 and message in capsys.readouterr().err.splitlines()[-1]


def test_a_fault_below_the_build_keeps_its_traceback(tmp_path, monkeypatch):
    """A numpy shape mismatch or a bad JSON value is a fault, not a usage error."""
    def fail(*args, **kwargs):
        raise ValueError("operands could not be broadcast together")

    monkeypatch.setattr(build_repeated_draw_baseline, "build_payload", fail)
    with pytest.raises(ValueError, match="could not be broadcast"):
        build_repeated_draw_baseline.main(["--estimates", str(tmp_path / "estimates.json")])


def _payload(labels, sizes=tuple(range(2, 58))):
    def interval(value):
        return {"estimate": value, "ci_low": value - 0.03, "ci_high": value + 0.03, "sd": 0.015}

    def point(size, offset):
        rate = 0.1 + offset + 0.6 * (size - sizes[0]) / max(1, sizes[-1] - sizes[0])
        baseline = 0.08 + offset / 2
        values = {"inconsistency": rate, "full_repeat": baseline, "separate_repeat": baseline,
                  "repeated_draw_baseline": baseline, "excess": rate - baseline, "lenient": rate, "strict": 0.0,
                  "lenient_minus_strict": rate, "rule_median_inconsistency": rate - 0.01,
                  "median_inconsistency": rate - 0.02}
        return {"size": size, **{name: interval(values[name]) for name in rdb.STATISTICS}}

    contrasts = {name: {"net_rise": {"minuend_sizes": [sizes[-1]], "subtrahend_sizes": [sizes[0]], **interval(0.5)}}
                 for name in rdb.STATISTICS}
    return {"analysis": rdb.ANALYSIS, "schema_version": rdb.SCHEMA_VERSION, "n_scenarios": 500, "n_subset_seeds": 10,
            "n_decoder_draws": 5, "bootstrap": {"interval_basis": "scenario_subset_cluster_bootstrap"},
            "judges": [{"label": label, "model_name": label, "points": [point(size, 0.02 * index) for size in sizes],
                        "contrasts": contrasts} for index, label in enumerate(labels)]}


def _overlaps(figure):
    from matplotlib.text import Text

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [text for text in figure.findobj(Text) if text.get_visible() and text.get_text()
             and (text.axes is None or text.axes.get_visible())]
    boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in texts]
    return [(a, b) for (a, box_a), (b, box_b) in itertools.combinations(boxes, 2) if box_a.overlaps(box_b)]


@pytest.mark.parametrize(("count", "shape"), [(1, (1, 1)), (4, (2, 2)), (5, (2, 3)), (6, (2, 3))])
def test_the_figure_draws_a_panel_per_judge_its_rate_and_dashed_baseline_with_bands_and_its_median_rate(
        count, shape):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D

    from ruleforge.experiments.ruleset_size import repeated_draw_baseline_figure as figure_module

    labels = list(SERIES_COLORS)[:count]
    payload = _payload(labels)
    figure = figure_module.build_figure(payload)
    try:
        panels = [axes for axes in figure.axes if axes.get_visible()]
        assert figure.axes[0].get_gridspec().get_geometry() == shape and len(panels) == count
        assert [axes.get_title() for axes in panels] == labels
        assert figure.get_supxlabel() == "Ruleset size" and figure.get_supylabel() == "Inconsistency rate"
        assert figure.get_suptitle() == "" and all(axes.get_ylim() == (0.0, 1.0) for axes in panels)
        for axes, label, judge in zip(panels, labels, payload["judges"]):
            baseline, rate, median = axes.lines
            assert (baseline.get_linestyle(), rate.get_linestyle(), median.get_linestyle()) == ("--", "-", ":")
            assert (to_hex(rate.get_color()), to_hex(baseline.get_color()), to_hex(median.get_color())) == (
                to_hex(SERIES_COLORS[label]), to_hex(figure_module.BASELINE_COLOR),
                to_hex(figure_module.MEDIAN_COLOR))
            assert list(median.get_ydata()) == [point["median_inconsistency"]["estimate"]
                                                for point in judge["points"]]
            assert median.get_zorder() > rate.get_zorder() and median.get_linewidth() < rate.get_linewidth()
            assert len(axes.collections) == 2  # one band each for the rate and the baseline; the median has none
        assert [tick for tick in panels[0].get_xticks()] == [2, 10, 20, 30, 40, 50, 57]
        [legend] = figure.legends
        assert [text.get_text() for text in legend.get_texts()] == [
            "Inconsistency rate", "Median inconsistency rate", "Repeated-draw baseline"]
        assert [(to_hex(line.get_color()), line.get_linestyle()) for line in legend.findobj(Line2D)] == [
            *((to_hex(SERIES_COLORS[label]), "-") for label in labels), (to_hex(figure_module.MEDIAN_COLOR), ":"),
            (to_hex(figure_module.BASELINE_COLOR), "--")]
        assert _overlaps(figure) == []
        labelled = [index for index, axes in enumerate(panels)
                    if any(label.get_visible() and label.get_text() for label in axes.get_xticklabels())]
        assert labelled == [index for index in range(count) if index + shape[1] >= count]
    finally:
        plt.close(figure)


def test_a_judge_without_a_colour_takes_one_no_drawn_judge_has_and_few_sizes_get_their_own_ticks(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from ruleforge.experiments.ruleset_size import repeated_draw_baseline_figure as figure_module

    figure = figure_module.build_figure(_payload(["GPT-OSS-120B", "Some judge"], sizes=(2, 3, 4)))
    try:
        assert to_hex(figure.axes[1].lines[1].get_color()) == to_hex(list(SERIES_COLORS.values())[1])
        assert list(figure.axes[0].get_xticks()) == [2, 4]
    finally:
        plt.close(figure)
    out = tmp_path / "figure.pdf"
    figure_module.draw(_payload(["GPT-OSS-120B"]), out)
    assert out.read_bytes().startswith(b"%PDF")


def test_the_figure_draws_every_judge_of_the_golden(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from ruleforge.experiments.ruleset_size import repeated_draw_baseline_figure as figure_module

    drawn = []
    monkeypatch.setattr(figure_module, "draw", lambda payload, out, dpi=None: drawn.append((payload, dpi)))
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps(_payload(["GPT-OSS-120B", "Claude Haiku 5.5"])))
    assert build_repeated_draw_baseline.main(["--from_golden", str(golden), "--figure", str(tmp_path / "f.png"),
                                              "--dpi", "200"]) == 0
    [(payload, dpi)] = drawn
    assert [judge["label"] for judge in payload["judges"]] == ["GPT-OSS-120B", "Claude Haiku 5.5"] and dpi == 200
