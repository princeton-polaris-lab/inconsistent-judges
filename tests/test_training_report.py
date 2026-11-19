from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from ruleforge.core.digests import canonical_digest, sha256_file
from ruleforge.grpo import anchor_set as anchors
from ruleforge.grpo import report as rp

NO, UNSURE, YES, MISSING = 0, 1, 2, rp.MISSING
RULE_IDS = ("Rule 1", "Rule 2", "Rule 3", "Rule 4")
RAW_SETTINGS = {
    "ruleset_sha256": "0" * 64,
    "null_model": {"resamples": 400, "seed": 16, "level": 0.95},
    "interval": {"resamples": 500, "seed": 16, "level": 0.95},
    "floored_gates": [],
    "stage1": {
        "draws": 4, "dev_scenarios": 12, "seed": 16, "base_draws": 8, "base_reference_draws": [0, 1, 2, 3],
        "subsets": {"m1": ["dev"], "G1": ["dev"], "G2": ["dev"], "G3": ["dev", "dev_nonyes"],
                    "G4": ["dev", "dev_nonyes"], "G6": ["dev"], "G7": ["dev"]},
        "many_yes": 3, "thinking": {"median_ratio": 0.5, "max_empty_share": 0.02},
    },
    "probe": {"scenarios": 6, "seed": 16, "bank_draws": 4, "base_reference_draw": 0},
}
SETTINGS = rp.load_settings(RAW_SETTINGS)
#: A tolerance that comes to zero takes 3 / n.
FLOOR_SETTINGS = rp.load_settings({**RAW_SETTINGS, "floored_gates": list(rp.GATES)})
MIX_FLOOR_SETTINGS = rp.load_settings({**RAW_SETTINGS, "floored_gates": ["G2", "G3", "G4"]})
SOURCES = ("stress", "weirdchat", "wildchat")
STATISTIC_BY_NAME = {statistic.name: statistic for statistic in rp.STATISTICS}


def _rules(labels) -> rp.RuleDraws:
    return rp.RuleDraws(np.asarray(labels, dtype=np.int8))


def _full(verdict, *, attempts=None, truncated=None, complete=None, maps=None, think=None, empty=None) -> rp.FullSet:
    verdict = np.asarray(verdict, dtype=np.int8)
    shape = verdict.shape
    accepted = verdict != MISSING
    return rp.FullSet(
        verdict,
        np.where(accepted, 1, 20) if attempts is None else np.asarray(attempts),
        np.zeros(shape, dtype=int) if truncated is None else np.asarray(truncated),
        np.ones(shape, dtype=bool) if complete is None else np.asarray(complete, dtype=bool),
        np.full((*shape, len(RULE_IDS)), NO, dtype=np.int8) if maps is None else np.asarray(maps, dtype=np.int8),
        np.where(accepted, 500, MISSING) if think is None else np.asarray(think),
        np.zeros(shape, dtype=bool) if empty is None else np.asarray(empty, dtype=bool),
    )


def _ctx(q, *, mode=None, flagged=None, source=None, many_yes=3) -> rp.Context:
    q = np.asarray(q, dtype=float)
    n = len(q)
    return rp.Context(q, np.asarray(q.argmax(1) if mode is None else mode),
                      np.ones((n, len(RULE_IDS)), dtype=bool) if flagged is None else np.asarray(flagged),
                      np.zeros(n, dtype=int) if source is None else np.asarray(source),
                      1 if source is None else int(np.max(source)) + 1, many_yes)


def _references(seed=0, n_dev=24, n_nonyes=6) -> rp.References:
    """Scenarios with a yes or no bank mode, a varying rule-2 label and base verdicts that mostly follow the mode."""
    rng = np.random.default_rng(seed)
    n = n_dev + n_nonyes
    scenarios = rp.Scenarios(
        tuple(f"s{i:02d}" for i in range(n)), tuple(SOURCES[i % 3] for i in range(n)),
        tuple("dev" if i < n_dev else "dev_nonyes" for i in range(n)),
    )
    yes_mode = np.array([i < n_dev and i % 4 != 0 for i in range(n)])
    bank = np.full((n, 4, len(RULE_IDS)), NO, dtype=np.int8)
    bank[yes_mode, :, 0] = YES
    bank[:, :, 1] = np.where(rng.random((n, 4)) < 0.3, UNSURE, NO)
    mode = np.where(yes_mode, YES, NO)
    verdict = np.where(rng.random((n, 8)) < 0.15, UNSURE, mode[:, None]).astype(np.int8)
    maps = np.repeat(bank[:, :1], 8, axis=1)
    attempts = rng.integers(1, 3, size=(n, 8))
    full = _full(verdict, attempts=attempts, truncated=((rng.random((n, 8)) < 0.2) & (attempts > 1)).astype(int),
                 complete=rng.random((n, 8)) < 0.7, maps=maps, think=rng.integers(300, 700, size=(n, 8)))
    rules = _rules(bank)
    return rp.References(scenarios, RULE_IDS, rules, full, rp.bank_context(scenarios, rules, many_yes=3))


@pytest.fixture(scope="module")
def refs():
    return _references()


@pytest.fixture(scope="module")
def tolerances(refs):
    return rp.compute_tolerances(refs, SETTINGS)


@pytest.fixture(scope="module")
def zero_refs():
    """:func:`_references` with no event where the statistics can have none."""
    refs = _references()
    label = refs.bank.label.copy()
    label[refs.scenarios.rows(["dev"]), :, 0] = YES
    bank = _rules(label)
    full = replace(refs.full, complete=np.ones_like(refs.full.complete), truncated=np.zeros_like(refs.full.truncated))
    return replace(refs, bank=bank, full=full, context=rp.bank_context(refs.scenarios, bank, many_yes=3))


def _base_save(refs, tolerances):
    rows = np.sort(refs.scenarios.rows_of(tolerances["stage1_scenarios"]))
    return refs.full.take(rows, [0, 1, 2, 3]), refs.bank.take(refs.scenarios.rows_of(tolerances["probe_scenarios"]), [0])


def test_the_product_form_is_exact_and_a_tie_goes_to_the_earlier_label():
    # "a": rule 1 says yes in one of two draws, rule 2 never: q_yes = 1/2, q_no = 1/2 * 1, q_unsure = 0
    scenarios = rp.Scenarios(("a", "b"), ("x", "y"), ("dev", "dev"))
    bank = _rules([[[YES, NO], [NO, NO]], [[UNSURE, YES], [UNSURE, YES]]])
    ctx = rp.bank_context(scenarios, bank, many_yes=2)
    assert ctx.q.tolist() == [[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]
    assert ctx.mode.tolist() == [NO, YES]  # the yes/no tie of "a" is never a yes row
    assert ctx.flagged.tolist() == [[True, False], [False, True]]
    assert (ctx.source.tolist(), ctx.n_sources, ctx.many_yes) == ([0, 1], 2, 2)
    assert ctx.take(np.array([1])).q.tolist() == [[0.0, 0.0, 1.0]]
    with pytest.raises(rp.ReportError, match="misses a label"):
        rp.bank_context(scenarios, _rules([[[YES, MISSING], [NO, NO]], [[NO, NO], [NO, NO]]]), many_yes=2)


def test_q_hat_and_the_mode_are_the_anchors_estimate_and_mode_of():
    assert rp.LABELS == anchors.TIE_ORDER
    rng = np.random.default_rng(16)
    rule_ids = ("Rule 1", "Rule 2", "Rule 3")
    banks, ties = 0, 0
    for k in range(1, 6):
        labels = rng.integers(0, 3, size=(60, k, len(rule_ids)), dtype=np.int8)
        n = labels.shape[0]
        scenarios = rp.Scenarios(tuple(f"s{i:02d}" for i in range(n)), ("x",) * n, ("dev",) * n)
        ctx = rp.bank_context(scenarios, _rules(labels), many_yes=2)
        for i in range(n):
            draws = [{rule: rp.LABELS[code] for rule, code in zip(rule_ids, draw)} for draw in labels[i]]
            q = anchors.estimate(draws, rule_ids).q
            assert ctx.q[i].tolist() == [float(q[label]) for label in rp.LABELS]
            assert rp.LABELS[ctx.mode[i]] == anchors.mode_of(q)
            banks += 1
            ties += sorted(q.values())[-2] == max(q.values())
    assert banks == 300 and ties > 0


def _side(name, cand, ref, ctx):
    return STATISTIC_BY_NAME[name].delta(cand, ref, ctx)


def test_the_full_set_ratios_by_hand():
    ctx = _ctx([[0, 0, 1], [1, 0, 0]])
    # candidate: 3 accepted of 5 attempts (2 truncated), 2 complete maps; reference: all clean
    cand = _full([[YES, MISSING], [NO, YES]], attempts=[[1, 2], [1, 1]], truncated=[[0, 2], [0, 0]],
                 complete=[[True, False], [True, False]])
    ref = _full([[YES, YES], [NO, NO]])
    assert _side("validity_per_attempt", cand, ref, ctx) == pytest.approx(3 / 5 - 1)
    assert _side("truncation_rate", cand, ref, ctx) == pytest.approx(2 / 5)
    assert _side("complete_map_rate", cand, ref, ctx) == pytest.approx(2 / 3 - 1)


def test_the_map_statistics_read_complete_maps_only():
    flagged = np.array([[True, True, False, False]])
    ctx = _ctx([[0, 0, 1]], flagged=flagged, many_yes=3)
    maps = [[[YES, YES, YES, NO], [YES, NO, NO, NO], [YES, YES, YES, YES]]]
    cand = _full([[YES, NO, YES]], complete=[[True, True, False]], maps=maps)
    ref = _full([[YES, YES, YES]], complete=[[True, True, True]], maps=[[[NO] * 4] * 3])
    # complete maps: 3 yes and 1 yes; the incomplete all-yes map is not read
    assert _side("many_yes_map_share", cand, ref, ctx) == pytest.approx(1 / 2 - 0)
    assert _side("map_yes_count", cand, ref, ctx) == pytest.approx(4 / 2 - 0)
    # 3 of the 4 yes rules are rules the bank flagged; the reference has no yes rule: precision undefined
    assert math.isnan(_side("flag_precision", cand, ref, ctx))
    assert _side("flag_precision", cand, cand, ctx) == 0.0


def test_mode_agreement_counts_the_non_yes_mode_scenarios_only():
    ctx = _ctx([[0, 0, 1], [1, 0, 0], [0.2, 0.7, 0.1]])  # modes: yes, no, unsure
    cand = _full([[NO, NO], [NO, YES], [UNSURE, MISSING]])
    ref = _full([[YES, YES], [NO, NO], [UNSURE, UNSURE]])
    assert _side("mode_agreement_nonyes", cand, ref, ctx) == pytest.approx(2 / 3 - 1)


def test_kappa_bank_by_hand_and_always_yes_is_zero():
    q = np.array([[0.1, 0.1, 0.8], [0.6, 0.2, 0.2], [0.0, 0.5, 0.5], [0.3, 0.3, 0.4]])
    ctx = _ctx(q, source=[0, 0, 1, 1])
    verdict = np.array([[YES, YES], [NO, YES], [UNSURE, YES], [YES, MISSING]])
    # by hand: P_o over the 7 accepted draws; P_e = sum_g w_g sum_c P(phi=c|g) * draw-weighted mean q_s(c)
    accepted = verdict != MISSING
    p_o = sum(q[s, verdict[s, d]] for s in range(4) for d in range(2) if accepted[s, d]) / 7
    p_e = 0.0
    for members in ((0, 1), (2, 3)):
        draws = [(s, d) for s in members for d in range(2) if accepted[s, d]]
        n_g = len(draws)
        for c in range(3):
            share = sum(verdict[s, d] == c for s, d in draws) / n_g
            mean_q = sum(q[s, c] for s, _ in draws) / n_g
            p_e += n_g / 7 * share * mean_q
    expected = (p_o - p_e) / (1 - p_e)
    always_yes = _full(np.where(accepted, YES, MISSING))
    assert _side("kappa_bank", _full(verdict), always_yes, ctx) == pytest.approx(expected - 0.0)
    assert _side("kappa_bank", always_yes, always_yes, ctx) == pytest.approx(0.0)


def test_the_probe_statistics_by_hand():
    ctx = _ctx([[0, 0, 1], [1, 0, 0]])
    cand = _rules([[[YES, NO, NO, NO]], [[NO, MISSING, NO, NO]]])
    ref = _rules([[[NO, NO, NO, NO]], [[NO, NO, NO, NO]]])
    # the second scenario's draw misses a rule's label: the share reads complete draws only, so that draw is left out
    # whatever its other labels say, never counted as a no
    assert _side("probe_or_yes_share", cand, ref, ctx) == pytest.approx(1 - 0)
    for labels in ([NO, MISSING, NO, NO], [NO, MISSING, YES, NO]):
        assert _side("probe_or_yes_share", _rules([[[NO, NO, NO, NO]], [labels]]), ref, ctx) == 0
    assert _side("probe_rule_yes_rate", cand, ref, ctx) == pytest.approx(1 / 7)


def test_m1_is_a_mean_over_scenarios_of_means_over_draws():
    ctx = _ctx([[0, 0, 1], [0.5, 0, 0.5]])
    data = _full([[YES, NO], [NO, MISSING]])
    # scenario 1: 1 - (1 + 0)/2 = 0.5; scenario 2: 1 - 0.5 = 0.5
    assert rp.m1_value(data, ctx) == pytest.approx(0.5)
    assert math.isnan(rp.m1_value(_full([[MISSING]]), _ctx([[0, 0, 1]])))


def test_every_ordered_split_of_the_pool():
    assert len(rp.ordered_splits(range(8), 4, 4)) == 70
    assert len(rp.ordered_splits(range(4), 1, 1)) == 12
    splits = rp.ordered_splits(range(4), 2, 2)
    assert len(splits) == 6 and ((2, 3), (0, 1)) in splits and ((0, 1), (2, 3)) in splits
    assert all(not set(cand) & set(ref) for cand, ref in rp.ordered_splits(range(5), 2, 2))


def test_the_null_draws_one_split_per_scenario_and_sums_what_each_took():
    choices = rp.split_choices(5, 3, 200, np.random.default_rng(1))
    assert choices.shape == (200, 5) and set(np.unique(choices)) == {0, 1, 2}
    assert len({tuple(row) for row in choices}) > 100
    parts = np.random.default_rng(2).normal(size=(3, 5, 2))
    by_hand = np.array([[parts[k, i] for i, k in enumerate(row)] for row in choices]).sum(1)
    assert np.allclose(rp.permutation_sums(parts, choices), by_hand)
    # every statistic of a group sees the same draws: identical parts give identical nulls
    first, second = rp.null_values([parts, parts], [lambda x: x[:, 0], lambda x: x[:, 0]], resamples=50, seed=1,
                                    name="stage1")
    assert np.array_equal(first, second) and first.shape == (50,)
    other = rp.null_values([parts], [lambda x: x[:, 0]], resamples=50, seed=1, name="probe")[0]
    assert not np.array_equal(first, other)


def test_outside_is_the_tolerance_over_a_whole_null():
    values = np.array([-0.2, -0.05, 0.0, 0.05, 0.2, np.nan])
    assert rp.outside(values, 0.1, rp.HIGHER).tolist() == [True, False, False, False, False, True]
    assert rp.outside(values, 0.1, rp.LOWER).tolist() == [False, False, False, False, True, True]
    assert rp.outside(values, 0.1, rp.TWO_SIDED).tolist() == [True, False, False, False, True, True]
    # a tie within TIE_SLACK is at the tolerance; anything past it is outside
    assert not rp.outside(0.1 + rp.TIE_SLACK / 2, 0.1, rp.LOWER) and rp.outside(0.1 + 1e-9, 0.1, rp.LOWER)


def test_the_quantile_is_the_nulls_percentile_in_the_direction_that_counts():
    values = np.concatenate([np.linspace(-1.0, 3.0, 401), [np.nan]])
    assert rp.null_quantile(values, rp.LOWER, 0.95) == pytest.approx(np.quantile(np.linspace(-1, 3, 401), 0.95))
    assert rp.null_quantile(values, rp.HIGHER, 0.95) == pytest.approx(np.quantile(-np.linspace(-1, 3, 401), 0.95))
    assert rp.null_quantile(values, rp.TWO_SIDED, 0.95) == pytest.approx(np.quantile(np.abs(np.linspace(-1, 3, 401)), 0.95))
    # a save the same as the base is outside 5% of the time: 5% of the null lies beyond
    null = np.random.default_rng(0).normal(size=20000)
    tol = rp.null_quantile(null, rp.LOWER, 0.95)
    assert (null > tol).mean() == pytest.approx(0.05, abs=0.002)
    with pytest.raises(rp.ReportError, match="no finite value"):
        rp.null_quantile(np.array([np.nan]), rp.LOWER, 0.95)


def test_the_tolerances_cover_every_statistic_with_the_designs_splits(refs, tolerances):
    stats = tolerances["statistics"]
    assert set(stats) == {s.name for s in rp.STATISTICS}
    splits = {s.stage: stats[s.name]["splits"] for s in rp.STATISTICS}
    assert splits == {rp.STAGE1: 70, rp.PROBE: 12}
    assert all(entry["tolerance"] >= 0 and entry["resamples"] == 400 for entry in stats.values())
    assert stats["kappa_bank"]["subsets"] == ["dev", "dev_nonyes"] and stats["kappa_bank"]["scenarios"] == 18
    assert stats["validity_per_attempt"]["subsets"] == ["dev"] and stats["probe_or_yes_share"]["scenarios"] == 6
    assert stats["probe_or_yes_share"]["subsets"] == ["dev"]
    assert stats["validity_per_attempt"]["null"]["n"] == 400
    assert len(tolerances["probe_scenarios"]) == 6 and len(tolerances["stage1_scenarios"]) == 18
    assert tolerances["thinking"]["base_median_think_tokens"] > 0
    assert rp.compute_tolerances(refs, SETTINGS)["statistics"] == stats


def _process(scenarios: rp.Scenarios, seed: int):
    """A base's per-scenario generating process: samplers of its full-set draws and of its per-rule draws."""
    rng = np.random.default_rng(seed)
    n, rules = len(scenarios.ids), len(RULE_IDS)
    p_verdict = rng.dirichlet([1.0, 0.4, 2.0], size=n)
    p_yes = np.where(rng.random((n, rules)) < 0.4, rng.beta(1, 3, (n, rules)), 0.0)
    p_unsure = rng.beta(1, 8, (n, rules))
    p_complete, p_retry, p_map_yes = rng.beta(4, 2, n), rng.beta(1, 6, n), rng.beta(1, 4, n)

    def full(draws):
        verdict = (rng.random((n, draws))[..., None] > np.cumsum(p_verdict, 1)[:, None, :2]).sum(-1)
        retry = rng.random((n, draws)) < p_retry[:, None]
        maps = np.where(rng.random((n, draws, rules)) < p_map_yes[:, None, None], YES, NO)
        return _full(verdict, attempts=1 + retry, truncated=(retry & (rng.random((n, draws)) < 0.5)).astype(int),
                     complete=rng.random((n, draws)) < p_complete[:, None], maps=maps,
                     think=rng.integers(300, 900, (n, draws)))

    def per_rule(draws):
        u = rng.random((n, draws, rules))
        label = np.where(u < p_yes[:, None], YES, np.where(u < (p_yes + p_unsure)[:, None], UNSURE, NO))
        return _rules(label)

    return full, per_rule


def test_a_save_drawn_from_the_bases_own_process_is_outside_each_tolerance_about_5_percent_of_the_time():
    n_dev, n_nonyes = 72, 18
    scenarios = rp.Scenarios(tuple(f"s{i:02d}" for i in range(n_dev + n_nonyes)),
                             tuple(SOURCES[i % 3] for i in range(n_dev + n_nonyes)),
                             tuple("dev" if i < n_dev else "dev_nonyes" for i in range(n_dev + n_nonyes)))
    settings = rp.load_settings({**RAW_SETTINGS, "null_model": {**RAW_SETTINGS["null_model"], "resamples": 300},
                                 "stage1": {**RAW_SETTINGS["stage1"], "dev_scenarios": n_dev},
                                 "probe": {**RAW_SETTINGS["probe"], "scenarios": 24}})
    reps, outside = 60, 0
    for rep in range(reps):
        full, per_rule = _process(scenarios, seed=rep)
        bank = per_rule(4)
        refs = rp.References(scenarios, RULE_IDS, bank, full(8), rp.bank_context(scenarios, bank, many_yes=3))
        tolerances = rp.compute_tolerances(refs, settings)
        probe = per_rule(1).take(scenarios.rows_of(tolerances["probe_scenarios"]))
        results = rp.save_report(refs, settings, tolerances, full(4), probe)["statistics"]
        outside += sum(not results[statistic.name]["within_tolerance"] for statistic in rp.STATISTICS)
    assert 0.03 <= outside / (reps * len(rp.STATISTICS)) <= 0.08


def test_the_probes_scenarios_are_fixed_by_source_and_seed_not_by_table_order():
    scenarios = _references().scenarios
    chosen = rp.probe_scenarios(scenarios, 10, 16)
    shuffled = rp.Scenarios(*(tuple(reversed(x)) for x in (scenarios.ids, scenarios.sources, scenarios.subsets)))
    assert rp.probe_scenarios(shuffled, 10, 16) == chosen and chosen == sorted(chosen)
    by_source = [scenarios.sources[scenarios.ids.index(sid)] for sid in chosen]
    assert sorted(by_source.count(s) for s in SOURCES) == [3, 3, 4]  # 24 dev scenarios, 8 per source
    assert all(scenarios.subsets[scenarios.ids.index(sid)] == "dev" for sid in chosen)
    assert rp.probe_scenarios(scenarios, 10, 17) != chosen
    with pytest.raises(rp.ReportError, match="dev has 24"):
        rp.probe_scenarios(scenarios, 25, 16)


def test_stage_1_is_its_fixed_dev_scenarios_and_every_dev_nonyes_one(refs, tolerances):
    chosen = set(tolerances["stage1_scenarios"])
    assert chosen >= {refs.scenarios.ids[i] for i in refs.scenarios.rows(["dev_nonyes"])}
    assert len(chosen & {refs.scenarios.ids[i] for i in refs.scenarios.rows(["dev"])}) == 12
    assert chosen == set(rp.stage1_scenarios(refs.scenarios, SETTINGS))
    assert rp.scenario_sets(refs.scenarios, SETTINGS) == {"stage1": sorted(chosen),
                                                          "probe": tolerances["probe_scenarios"]}


def test_a_save_equal_to_the_base_is_within_every_tolerance(refs, tolerances):
    full, probe = _base_save(refs, tolerances)
    report = rp.save_report(refs, SETTINGS, tolerances, full, probe)
    assert report["outside_tolerance"] == [] and all(report["gates"].values()), report
    assert report["m1"]["value"] == pytest.approx(tolerances["base_m1"])
    assert report["delta_m1_vs_base"]["value"] == pytest.approx(0.0)
    assert report["m1"]["ci95"][0] <= report["m1"]["value"] <= report["m1"]["ci95"][1]
    assert report["delta_m1_vs_base"]["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]
    assert report["m1"]["scenarios"] == report["delta_m1_vs_base"]["scenarios"] == 12


def _outside(refs, tolerances, **changes):
    full, probe = _base_save(refs, tolerances)
    probe = changes.pop("probe", probe)
    return rp.save_report(refs, SETTINGS, tolerances, replace(full, **changes), probe)["outside_tolerance"]


def test_each_gate_is_outside_for_a_save_built_to_be(refs, tolerances):
    full, probe = _base_save(refs, tolerances)
    n = full.verdict.shape
    assert _outside(refs, tolerances, attempts=full.attempts * 4) == ["G1"]  # validity per attempt
    assert _outside(refs, tolerances, truncated=full.attempts) == ["G1"]  # every attempt truncated
    fewer = full.complete.copy()
    fewer[:, 1:] = False
    assert _outside(refs, tolerances, complete=fewer) == ["G2"]
    # no complete map at all: G6's statistics cannot be computed, and what cannot be computed is outside
    assert _outside(refs, tolerances, complete=np.zeros(n, dtype=bool)) == ["G2", "G6"]
    always_yes = np.where(full.accepted, YES, MISSING).astype(np.int8)
    assert set(_outside(refs, tolerances, verdict=always_yes)) >= {"G3", "G4"}
    all_yes_maps = np.full(full.map_labels.shape, YES, dtype=np.int8)
    assert _outside(refs, tolerances, map_labels=all_yes_maps) == ["G6"]
    assert _outside(refs, tolerances, think_tokens=np.where(full.accepted, 10, MISSING)) == ["G7"]
    assert _outside(refs, tolerances, empty_think=np.ones(n, dtype=bool)) == ["G7"]
    for label in (YES, NO):  # G5p is two-sided: drift either way is outside
        drifted = _rules(np.full(probe.label.shape, label, dtype=np.int8))
        assert _outside(refs, tolerances, probe=drifted) == ["G5p"]
    with pytest.raises(rp.ReportError, match="4 full-set draws of each of its 18 scenarios"):
        rp.save_report(refs, SETTINGS, tolerances, full.take(None, [0, 1]), probe)


def test_a_save_with_no_accepted_verdict_is_reported_without_an_m1_or_an_interval(refs, tolerances):
    full, probe = _base_save(refs, tolerances)
    silent = replace(full, verdict=np.full(full.verdict.shape, MISSING, dtype=np.int8))
    report = rp.json_ready(rp.save_report(refs, SETTINGS, tolerances, silent, probe))
    assert report["m1"] == {"value": None, "bootstrap_mean": None, "ci95": [None, None], "scenarios": 0}
    assert report["delta_m1_vs_base"] == {"value": None, "paired_mean": None, "ci95": [None, None], "scenarios": 0}
    assert not report["gates"]["G7"]
    assert (report["statistics"]["median_think_tokens"]["value"], report["statistics"]["empty_think_share"]["value"]) == (
        None, None)
    table = rp.report_table({"100": report}, tolerances)
    assert "n/a" in table and "Selected: none, no save has an m1" in table


def test_the_save_with_the_lowest_m1_is_selected_the_earlier_on_a_tie():
    def saves(**m1):
        return {step[1:]: {"m1": {"value": value}} for step, value in m1.items()}

    assert rp.selected_save(saves(s100=0.5, s200=0.3, s300=0.4)) == 200
    assert rp.selected_save(saves(s300=0.3, s200=0.3, s100=0.6)) == 200  # a tie: the earlier save
    assert rp.selected_save(saves(s1000=0.2, s900=0.2)) == 900  # saves are ordered by step, not by name
    assert rp.selected_save(saves(s100=0.9)) == 100  # one save
    assert rp.selected_save({}) is None  # no save
    assert rp.selected_save(saves(s100=None, s200=0.7)) == 200 and rp.selected_save(saves(s100=None)) is None
    # a value that is no number is no m1, wherever it stands
    for undefined in (math.nan, -math.inf, math.inf):
        assert rp.selected_save(saves(s100=undefined, s200=0.2)) == 200
        assert rp.selected_save(saves(s200=0.2, s100=undefined)) == 200
        assert rp.selected_save(saves(s100=undefined)) is None


def test_flag_precision_is_outside_for_a_map_whose_yes_rules_the_bank_never_said(refs, tolerances):
    full, probe = _base_save(refs, tolerances)
    maps = np.full(full.map_labels.shape, NO, dtype=np.int8)
    maps[..., 3] = YES  # rule 4: no bank draw ever says yes
    results = rp.save_report(refs, SETTINGS, tolerances, replace(full, map_labels=maps), probe)["statistics"]
    assert not results["flag_precision"]["within_tolerance"] and results["flag_precision"]["delta"] < -0.5


def test_the_floors_n_is_each_statistics_reference_side_on_the_reference_draws(refs, tolerances):
    sets = {"stage1": tolerances["stage1_scenarios"], "probe": tolerances["probe_scenarios"]}
    rows = {s.name: rp._stage_rows(s, refs, SETTINGS, sets) for s in rp.STATISTICS}
    n = {s.name: rp.base_count(s, refs, SETTINGS, rows[s.name]) for s in rp.STATISTICS}
    dev = rows["validity_per_attempt"]
    full = refs.full.take(dev, [0, 1, 2, 3])  # stage 1's reference draws, on its dev scenarios
    complete = full.accepted & full.complete
    nonyes = int((refs.context.mode[rows["mode_agreement_nonyes"]] != YES).sum())
    assert n == {
        "validity_per_attempt": full.attempts.sum(), "truncation_rate": full.attempts.sum(),
        "complete_map_rate": full.accepted.sum(), "kappa_bank": 18 * 4, "mode_agreement_nonyes": nonyes * 4,
        "many_yes_map_share": complete.sum(), "map_yes_count": complete.sum(),
        "flag_precision": ((full.map_labels == YES).sum(-1) * complete).sum(),
        "probe_or_yes_share": 6, "probe_rule_yes_rate": 6 * 4,  # 6 probe scenarios x bank draw 0 (x 4 rules)
    }


def test_every_floored_tolerance_is_at_least_three_over_n(zero_refs, refs):
    for base in (zero_refs, refs):
        plain, floored = rp.compute_tolerances(base, SETTINGS), rp.compute_tolerances(base, FLOOR_SETTINGS)
        sets = {"stage1": floored["stage1_scenarios"], "probe": floored["probe_scenarios"]}
        for name, entry in floored["statistics"].items():
            statistic = STATISTIC_BY_NAME[name]
            n = rp.base_count(statistic, base, FLOOR_SETTINGS, rp._stage_rows(statistic, base, FLOOR_SETTINGS, sets))
            assert entry["floor"] == {"n": int(n), "value": 3 / n}
            assert entry["tolerance"] == max(plain["statistics"][name]["tolerance"], 3 / n)
            assert entry["quantile"] == plain["statistics"][name]["quantile"]
            assert entry["floored"] is (3 / n > max(entry["quantile"], 0.0))
        assert all(entry["floor"] is None and not entry["floored"] for entry in plain["statistics"].values())
    zero = rp.compute_tolerances(zero_refs, FLOOR_SETTINGS)["statistics"]
    for name, n in (("kappa_bank", 72), ("probe_or_yes_share", 6)):
        assert zero[name]["floor"]["n"] == n


def test_the_floor_is_monotone_one_base_event_is_never_held_stricter_than_none(zero_refs):
    one = replace(zero_refs, full=replace(zero_refs.full, complete=zero_refs.full.complete.copy()))
    sets = rp.scenario_sets(zero_refs.scenarios, SETTINGS)
    dev = [i for i in np.sort(zero_refs.scenarios.rows_of(sets["stage1"])) if zero_refs.scenarios.subsets[i] == "dev"]
    one.full.complete[dev[0], 0] = False  # one incomplete map among the base's reference draws
    none = rp.compute_tolerances(zero_refs, FLOOR_SETTINGS)["statistics"]["complete_map_rate"]
    some = rp.compute_tolerances(one, FLOOR_SETTINGS)["statistics"]["complete_map_rate"]
    assert none["quantile"] == 0.0 and 0.0 < some["quantile"] < 3 / 48  # the null alone: below 3 / n
    assert some["tolerance"] >= none["tolerance"] == 3 / 48 and some["floored"]


def test_an_unfloored_gate_keeps_its_quantile(zero_refs):
    everywhere = rp.compute_tolerances(zero_refs, FLOOR_SETTINGS)["statistics"]
    mix = rp.compute_tolerances(zero_refs, MIX_FLOOR_SETTINGS)["statistics"]
    for name in ("truncation_rate", "many_yes_map_share", "map_yes_count"):
        assert everywhere[name]["tolerance"] > 0.0 and mix[name]["floor"] is None and mix[name]["tolerance"] == 0.0
    assert mix["complete_map_rate"]["tolerance"] == everywhere["complete_map_rate"]["tolerance"]


@pytest.mark.parametrize("n", [12, 48, 96, 100, 600, 1200])
def test_exactly_three_events_over_a_zero_base_sit_at_three_over_n_and_are_within(n):
    """|delta| <= 3 / n holds 3 events in exact arithmetic."""
    for events, inside in ((3, True), (4, False)):
        delta = (n - events) / n - 1.0  # a rate of 1 - events / n against the base's 1
        assert (not rp.outside(delta, 3 / n, rp.HIGHER)) is inside and (not rp.outside(-delta, 3 / n, rp.LOWER)) is inside
        assert (not rp.outside(delta, 3 / n, rp.TWO_SIDED)) is inside


def test_a_floored_statistic_with_no_observation_to_count_is_refused(zero_refs):
    # no accepted response in the reference draws 0-3: G2's null is defined on other splits, and n would be 0
    verdict = np.where(np.arange(8) < 4, MISSING, zero_refs.full.verdict).astype(np.int8)
    with pytest.raises(rp.ReportError, match="complete_map_rate is floored.*no n"):
        rp.compute_tolerances(replace(zero_refs, full=replace(zero_refs.full, verdict=verdict)), MIX_FLOOR_SETTINGS)


def test_a_statistic_that_cannot_be_computed_is_outside():
    assert rp.outside(math.nan, 1.0, rp.TWO_SIDED)


def test_json_ready_turns_numpy_and_non_finite_values_into_json():
    assert rp.json_ready({"a": np.float64(1.5), "b": (np.int64(2), math.inf), 3: [np.bool_(True), math.nan]}) == {
        "a": 1.5, "b": [2, None], "3": [True, None]}


def test_settings_missing_a_section_a_subset_or_a_split_are_refused(tmp_path):
    with pytest.raises(rp.ReportError, match=r"no \['probe'\]"):
        rp.load_settings({k: v for k, v in RAW_SETTINGS.items() if k != "probe"})
    subsets = {k: v for k, v in RAW_SETTINGS["stage1"]["subsets"].items() if k != "G4"}
    with pytest.raises(rp.ReportError, match=r"subsets for \['G4'\]"):
        rp.load_settings({**RAW_SETTINGS, "stage1": {**RAW_SETTINGS["stage1"], "subsets": subsets}})
    with pytest.raises(rp.ReportError, match=r"unknown subsets \['train'\]"):
        rp.load_settings({**RAW_SETTINGS, "stage1": {**RAW_SETTINGS["stage1"], "subsets": {**subsets, "G4": ["train"]}}})
    with pytest.raises(rp.ReportError, match="floored_gates"):
        rp.load_settings({**RAW_SETTINGS, "floored_gates": ["G9"]})
    with pytest.raises(rp.ReportError, match="stage1: the base's 5 draws cannot be split"):
        rp.load_settings({**RAW_SETTINGS, "stage1": {**RAW_SETTINGS["stage1"], "base_draws": 5}})
    with pytest.raises(rp.ReportError, match="probe: the base's 1 draws cannot be split"):
        rp.load_settings({**RAW_SETTINGS, "probe": {**RAW_SETTINGS["probe"], "bank_draws": 1}})
    (tmp_path / "s.yaml").write_text(json.dumps(RAW_SETTINGS))
    settings = rp.read_settings(tmp_path / "s.yaml")
    assert settings["probe"]["scenarios"] == 6 and settings.subsets("G4") == ("dev", "dev_nonyes")
    assert MIX_FLOOR_SETTINGS.floored("G4") and not MIX_FLOOR_SETTINGS.floored("G6")


def test_the_committed_settings_read_and_their_ruleset_is_the_papers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    settings = rp.read_settings(root / "configs/training/report.yaml")
    assert settings["ruleset_sha256"] == sha256_file(root / "data/rules/modelspec/openai_rules_all.jsonl")
    assert len(rp.rule_ids_of(root / "data/rules/modelspec/openai_rules_all.jsonl", settings["ruleset_sha256"])) == 57
    assert settings["floored_gates"] == ["G2", "G3", "G4"]
    assert settings["null_model"] == {"resamples": 10000, "seed": 16, "level": 0.95}
    assert settings["interval"] == {"resamples": 10000, "seed": 16, "level": 0.95}
    assert settings["stage1"] == {
        "draws": 2, "dev_scenarios": 150, "seed": 16, "base_draws": 8, "base_reference_draws": [0, 1, 2, 3],
        "subsets": {"m1": ["dev"], "G1": ["dev"], "G2": ["dev"], "G3": ["dev", "dev_nonyes"], "G4": ["dev", "dev_nonyes"],
                    "G6": ["dev"], "G7": ["dev"]},
        "many_yes": 29, "thinking": {"median_ratio": 0.5, "max_empty_share": 0.02}}
    assert settings["probe"] == {"scenarios": 100, "seed": 16, "bank_draws": 4, "base_reference_draw": 0}
    with pytest.raises(rp.ReportError, match="not the settings' ruleset"):
        rp.rule_ids_of(root / "data/rules/modelspec/openai_rules_all.jsonl", "0" * 64)


def test_the_scenario_table(tmp_path):
    with pytest.raises(rp.ReportError, match="one source and one subset"):
        rp.Scenarios(("a",), (), ("dev",))
    with pytest.raises(rp.ReportError, match="twice"):
        rp.Scenarios(("a", "a"), ("x", "x"), ("dev", "dev"))
    with pytest.raises(rp.ReportError, match=r"\['test'\]"):
        rp.Scenarios(("a",), ("x",), ("test",))
    path = tmp_path / "scenarios.jsonl"
    path.write_text("".join(json.dumps({"scenario_id": s, "source": "x", "subset": sub}) + "\n"
                            for s, sub in (("a", "dev"), ("b", "dev_nonyes"))) + "\n")
    scenarios = rp.read_scenarios(path)
    assert scenarios.rows(["dev_nonyes"]).tolist() == [1] and scenarios.rows_of(["b", "a"]).tolist() == [1, 0]
    with pytest.raises(rp.ReportError, match="not in the scenario table"):
        scenarios.rows_of(["c"])


def _full_row(sid, draw, verdict="yes", **extra):
    full = {"verdict": verdict, "attempts": 1, "truncated_attempts": 0, "map": {"Rule 1": "yes"}, "map_complete": False,
            "think_tokens": 12, "empty_think": False, **extra}
    return {"scenario_id": sid, "draw": draw, "full": full}


def _rules_obj(label="no", **extra):
    return {rule: {"label": label, "attempts": 1, "truncated_attempts": 0, **extra} for rule in RULE_IDS}


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_response_tables_read_every_field_and_refuse_what_they_cannot_be(tmp_path):
    scenarios = rp.Scenarios(("a", "b"), ("x", "x"), ("dev", "dev"))
    rows = np.array([0, 1])
    valid_rows = [_full_row("a", 0), {**_full_row("b", 0, None, attempts=20), "rules": _rules_obj(None, attempts=20)}]
    for row in valid_rows[:1]:
        row["rules"] = _rules_obj("yes", truncated_attempts=1, attempts=2)
    full, rules = rp.read_responses(_write(tmp_path / "t.jsonl", valid_rows), scenarios, rows, RULE_IDS, 1,
                                     full=True, rules=True)
    assert full.verdict.tolist() == [[YES], [MISSING]] and full.attempts.tolist() == [[1], [20]]
    assert full.map_labels[0, 0].tolist() == [YES, MISSING, MISSING, MISSING] and full.think_tokens.tolist() == [[12], [MISSING]]
    assert rules.label[0, 0].tolist() == [YES] * 4
    assert rules.label[1, 0].tolist() == [MISSING] * 4

    def refused(table_rows, match, **kwargs):
        with pytest.raises(rp.ReportError, match=match):
            rp.read_responses(_write(tmp_path / "bad.jsonl", table_rows), scenarios, rows, RULE_IDS, 1,
                               full=kwargs.get("full", True), rules=kwargs.get("rules", False))

    refused([_full_row("c", 0)], "is not one of the table's 2 scenarios x 1 draws")
    refused([_full_row("a", 1)], "is not one of the table's")
    refused([{**_full_row("a", 0), "draw": True}], "is not one of the table's")
    refused([_full_row("a", 0), _full_row("a", 0)], "appears twice")
    refused([_full_row("a", 0)], "no row for scenario 'b' draw 0")
    refused([{"scenario_id": "a", "draw": 0}], "has no full-set object")
    refused([_full_row("a", 0, "maybe")], "label 'maybe'")
    refused([_full_row("a", 0, attempts=-1)], "non-negative integer")
    refused([_full_row("a", 0, attempts=True)], "non-negative integer")
    refused([_full_row("a", 0, truncated_attempts=2)], "2 truncated of 1 attempts")
    refused([_full_row("a", 0, truncated_attempts=1)], "1 truncated of 1 attempts")
    refused([_full_row("a", 0, attempts=0)], "0 truncated of 0 attempts")
    refused([_full_row("a", 0, map={"Rule 9": "yes"})], "outside the ruleset")
    refused([_full_row("a", 0, map_complete=True)], "complete without every rule")
    refused([{"scenario_id": "a", "draw": 0, "rules": {"Rule 1": {}}}], "name every rule", full=False, rules=True)
    refused([{"scenario_id": "a", "draw": 0, "rules": _rules_obj(truncated_attempts=3)}], "3 truncated of 1",
            full=False, rules=True)


def test_a_tables_identity_names_its_bytes_its_weights_and_its_dtype(tmp_path):
    table = _write(tmp_path / "stage1.jsonl", [_full_row("a", 0)])
    with pytest.raises(rp.ReportError, match="no identity sidecar stage1.identity.json"):
        rp.read_table_identity(table)
    written = rp.write_table_identity(table, model_weights_sha256="w" * 64, serving_dtype="float32")
    assert rp.identity_path(table) == tmp_path / "stage1.identity.json"
    expect = {"model_weights_sha256": "w" * 64, "serving_dtype": "float32"}
    assert rp.read_table_identity(table, expect) == written and written["table_sha256"] == sha256_file(table)
    with pytest.raises(rp.ReportError, match="produced under"):
        rp.read_table_identity(table, {**expect, "serving_dtype": "bfloat16"})
    with pytest.raises(rp.ReportError, match="produced under"):
        rp.read_table_identity(table, {**expect, "model_weights_sha256": "x" * 64})
    _write(table, [_full_row("a", 0, "no")])
    with pytest.raises(rp.ReportError, match="rewritten after its identity"):
        rp.read_table_identity(table)
    rp.identity_path(table).write_text(json.dumps({"serving_dtype": "float32"}))
    with pytest.raises(rp.ReportError, match="must hold exactly"):
        rp.read_table_identity(table)


def _save_rows(scenarios, rows, data):
    out = []
    for i, row in enumerate(rows):
        for d in range(data.verdict.shape[1]):
            verdict = rp.LABELS[data.verdict[i, d]] if data.verdict[i, d] != MISSING else None
            full = {"verdict": verdict, "attempts": int(data.attempts[i, d]), "truncated_attempts": int(data.truncated[i, d])}
            if verdict is not None:
                full.update({"map": {rule: rp.LABELS[code] for rule, code in zip(RULE_IDS, data.map_labels[i, d])},
                             "map_complete": bool(data.complete[i, d]), "think_tokens": int(data.think_tokens[i, d]),
                             "empty_think": bool(data.empty_think[i, d])})
            out.append({"scenario_id": scenarios.ids[row], "draw": d, "full": full})
    return out


def _rule_rows(scenarios, rows, draws):
    return [{"scenario_id": scenarios.ids[row], "draw": d,
             "rules": {rule: {"label": rp.LABELS[draws.label[i, d, r]], "attempts": 1, "truncated_attempts": 0}
                       for r, rule in enumerate(RULE_IDS)}}
            for i, row in enumerate(rows) for d in range(draws.label.shape[1])]


def _table(path, rows, **identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, rows)
    rp.write_table_identity(path, **{"model_weights_sha256": None, "serving_dtype": "float32", **identity})
    return path


@pytest.fixture
def tree(tmp_path, refs):
    """The base's tables, a ruleset and the settings on disk."""
    base = tmp_path / "base"
    base.mkdir()
    everything = np.arange(len(refs.scenarios.ids))
    _write(base / rp.SCENARIOS_FNAME, [{"scenario_id": sid, "source": source, "subset": subset} for sid, source, subset
                                       in zip(refs.scenarios.ids, refs.scenarios.sources, refs.scenarios.subsets)])
    _table(base / rp.BANK_FNAME, _rule_rows(refs.scenarios, everything, refs.bank))
    _table(base / rp.FULL_FNAME, _save_rows(refs.scenarios, everything, refs.full))
    rules = _write(tmp_path / "rules.jsonl", [{"id": rule, "rule": rule} for rule in RULE_IDS])
    settings = {**RAW_SETTINGS, "ruleset_sha256": sha256_file(rules)}
    (tmp_path / "settings.yaml").write_text(json.dumps(settings))
    return tmp_path


def _save_tables(tree, refs, tolerances, step, **identity):
    rows = np.sort(refs.scenarios.rows_of(tolerances["stage1_scenarios"]))
    full, probe = _base_save(refs, tolerances)
    identity = {"model_weights_sha256": "w" * 64, **identity}
    directory = tree / "saves" / f"checkpoint-{step}"
    _table(directory / rp.STAGE1_FNAME, _save_rows(refs.scenarios, rows, full), **identity)
    _table(directory / rp.PROBE_FNAME, _rule_rows(refs.scenarios, refs.scenarios.rows_of(tolerances["probe_scenarios"]), probe),
           **identity)
    return directory


def _call(tree, **changes):
    return rp.report(tree / "settings.yaml", **{"base": tree / "base", "saves": tree / "saves", "rules": tree / "rules.jsonl",
                                                "out": tree / "out", **changes})


def test_the_tables_read_back_to_the_same_tolerances(tree, refs, tolerances):
    read = rp.read_references(tree / "base", SETTINGS, RULE_IDS)
    assert read.identity == {"model_weights_sha256": None, "adapter_sha256": None, "serving_dtype": "float32"}
    again = rp.compute_tolerances(read, SETTINGS)
    assert again["statistics"] == tolerances["statistics"]
    assert again["inputs"] == {"scenarios": sha256_file(tree / "base/scenarios.jsonl"),
                               "bank": sha256_file(tree / "base/bank.jsonl"), "full": sha256_file(tree / "base/full.jsonl")}
    assert again["scenarios"]["count"] == 30 and again["reference_identity"] == read.identity


def test_the_base_must_be_one_condition_of_the_base_model(tree, refs):
    rp.write_table_identity(tree / "base/full.jsonl", model_weights_sha256=None, serving_dtype="bfloat16")
    with pytest.raises(rp.ReportError, match="served in .*one engine serves both"):
        rp.read_references(tree / "base", SETTINGS, RULE_IDS)
    rp.write_table_identity(tree / "base/full.jsonl", model_weights_sha256="w" * 64, serving_dtype="float32")
    with pytest.raises(rp.ReportError, match="produced under"):
        rp.read_references(tree / "base", SETTINGS, RULE_IDS)


def test_the_bank_and_the_full_set_must_name_one_condition_where_their_sidecars_record_it(tree, refs):
    def identities(bank, full):
        for name, condition in ((rp.BANK_FNAME, bank), (rp.FULL_FNAME, full)):
            rp.write_table_identity(tree / "base" / name, model_weights_sha256=None, serving_dtype="float32",
                                    condition_sha256=condition)

    identities("c" * 64, "c" * 64)
    read = rp.read_references(tree / "base", SETTINGS, RULE_IDS)
    assert read.identity == {"model_weights_sha256": None, "adapter_sha256": None, "serving_dtype": "float32"}
    assert json.loads(rp.identity_path(tree / "base" / rp.BANK_FNAME).read_text())["condition_sha256"] == "c" * 64
    for bank, full in (("c" * 64, "d" * 64), ("c" * 64, None)):
        identities(bank, full)
        with pytest.raises(rp.ReportError, match="the bank and the full-set draws come from other conditions"):
            rp.read_references(tree / "base", SETTINGS, RULE_IDS)


def test_a_save_s_tables_must_name_one_condition_served_as_the_base_was(tree, refs, tolerances):
    read = rp.read_references(tree / "base", SETTINGS, RULE_IDS)
    directory = _save_tables(tree, refs, tolerances, 100)
    report = rp.read_save(directory, read, SETTINGS, tolerances)
    assert report["model_weights_sha256"] == "w" * 64 and report["serving_dtype"] == "float32"
    assert report["tables"] == {"stage1": sha256_file(directory / "stage1.jsonl"), "probe": sha256_file(directory / "probe.jsonl")}
    rp.write_table_identity(directory / rp.PROBE_FNAME, model_weights_sha256="x" * 64, serving_dtype="float32")
    with pytest.raises(rp.ReportError, match="produced under"):
        rp.read_save(directory, read, SETTINGS, tolerances)
    directory = _save_tables(tree, refs, tolerances, 200, serving_dtype="bfloat16")
    with pytest.raises(rp.ReportError, match="served in bfloat16, and the base in float32"):
        rp.read_save(directory, read, SETTINGS, tolerances)


def test_the_report_writes_tolerances_every_save_and_a_table(tree, refs, tolerances, capsys):
    for step in (200, 100):
        _save_tables(tree, refs, tolerances, step)
    (tree / "saves/checkpoint-300").mkdir()
    (tree / "saves/checkpoint-x").mkdir()
    (tree / "saves/checkpoint-400").write_text("not a directory")
    args = ["saves", "--settings", str(tree / "settings.yaml"), "--base", str(tree / "base"), "--saves", str(tree / "saves"),
            "--rules", str(tree / "rules.jsonl"), "--out", str(tree / "out")]
    assert rp.main(args) == 0
    table = capsys.readouterr().out
    assert table == (tree / "out/report.md").read_text()
    document = json.loads((tree / "out/report.json").read_text())
    assert document["kind"] == rp.REPORT_KIND and list(document["saves"]) == ["100", "200"]
    written = json.loads((tree / "out/tolerances.json").read_text())
    assert written["kind"] == rp.TOLERANCES_KIND
    assert written["statistics"]["kappa_bank"]["tolerance"] == tolerances["statistics"]["kappa_bank"]["tolerance"]
    save = document["saves"]["100"]
    assert save["statistics"]["kappa_bank"]["within_tolerance"] and save["gates"] == dict.fromkeys(rp.GATES, True)
    assert "| G4 | mode_agreement_nonyes |" in table and "| G7 | median_think_tokens |" in table
    assert f"| 100 | {save['m1']['value']:.4f} [" in table and "| none |" in table
    lowest = min(document["saves"], key=lambda step: (document["saves"][step]["m1"]["value"], int(step)))
    assert document["selected"] == int(lowest)
    assert table.splitlines()[1] == f"Selected: {lowest}, the reported save with the lowest m1 (on a tie, the earlier save)"


def test_the_table_s_change_from_the_base_is_the_paired_one_its_interval_is_of(refs, tolerances):
    full, probe = _base_save(refs, tolerances)
    side = rp._m1_side(full, refs.context.take(np.sort(refs.scenarios.rows_of(tolerances["stage1_scenarios"]))))
    dev = [i for i, row in enumerate(np.sort(refs.scenarios.rows_of(tolerances["stage1_scenarios"])))
           if refs.scenarios.subsets[row] == "dev"]
    outlier = max(dev, key=lambda i: abs(side[i, 0] / side[i, 1] - tolerances["base_m1"]))
    verdict = full.verdict.copy()
    verdict[outlier] = MISSING
    report = rp.json_ready(rp.save_report(refs, SETTINGS, tolerances, replace(full, verdict=verdict), probe))
    change = report["delta_m1_vs_base"]
    assert change["paired_mean"] == pytest.approx(0.0) and change["ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]
    # the save lost a scenario the base answered: its m1 less the base's is not the change, the paired estimate is
    assert abs(report["m1"]["value"] - report["base_m1"]) > 1e-3
    assert change["value"] == change["paired_mean"] and change["scenarios"] == 11
    assert "| +0.0000 [+0.0000, +0.0000] |" in rp.report_table({"100": report}, tolerances)


def test_a_value_outside_its_tolerance_is_marked_in_the_table(refs, tolerances):
    full, probe = _base_save(refs, tolerances)
    report = rp.json_ready(rp.save_report(refs, SETTINGS, tolerances, replace(full, attempts=full.attempts * 4), probe))
    report["statistics"]["empty_think_share"]["within_tolerance"] = False
    table = rp.report_table({"100": report}, tolerances)
    assert "| G1, G7 |" not in table and "| G1 |" in table
    assert any(line.startswith("| G1 | validity_per_attempt |") and line.rstrip(" |").endswith("*") for line in table.splitlines())
    assert any(line.startswith("| G7 | empty_think_share |") and line.rstrip(" |").endswith("*") for line in table.splitlines())


def test_a_saves_directory_without_tables_is_refused(tree):
    (tree / "saves").mkdir()
    with pytest.raises(rp.ReportError, match="holds no checkpoint-N directory"):
        _call(tree)


@pytest.mark.parametrize("kept, absent", [(rp.STAGE1_FNAME, rp.PROBE_FNAME), (rp.PROBE_FNAME, rp.STAGE1_FNAME)])
def test_a_save_with_one_table_is_refused_by_name(tree, refs, tolerances, kept, absent):
    _save_tables(tree, refs, tolerances, 600)
    directory = _save_tables(tree, refs, tolerances, 500)
    (directory / absent).unlink()
    with pytest.raises(rp.ReportError, match=f"checkpoint-500 holds {kept} and no {absent}"):
        _call(tree)
    (directory / kept).unlink()
    assert list(_call(tree)["saves"]) == ["600"]


def _case_tables(directory):
    """A tiny synthetic run on disk: the base's tables and two saves' tables, from a seeded generator only."""
    rng = np.random.default_rng(16)
    n_dev, n_nonyes, rules = 30, 8, tuple(f"Rule {i}" for i in range(1, 7))
    ids = [f"s{i:02d}" for i in range(n_dev + n_nonyes)]
    subsets = ["dev"] * n_dev + ["dev_nonyes"] * n_nonyes
    sources = [("a", "b", "c")[i % 3] for i in range(n_dev + n_nonyes)]
    p_yes = np.where(rng.random((len(ids), len(rules))) < 0.5, rng.beta(1, 3, (len(ids), len(rules))), 0.0)

    def rule_rows(chosen, draws, shift=0.0):
        rows = []
        for sid in chosen:
            i = ids.index(sid)
            for draw in range(draws):
                entries = {}
                for r, rule in enumerate(rules):
                    u = rng.random()
                    label = "yes" if u < p_yes[i, r] + shift else "unsure" if u < p_yes[i, r] + shift + 0.1 else "no"
                    entries[rule] = {"label": label, "attempts": 1, "truncated_attempts": 0}
                rows.append({"scenario_id": sid, "draw": draw, "rules": entries})
        return rows

    def full_rows(chosen, draws, yes_rate):
        rows = []
        for sid in chosen:
            for draw in range(draws):
                u = rng.random()
                if u < 0.05:
                    rows.append({"scenario_id": sid, "draw": draw, "full": {"verdict": None, "attempts": 3,
                                                                            "truncated_attempts": int(rng.integers(0, 4))}})
                    continue
                verdict = "yes" if rng.random() < yes_rate else str(rng.choice(["no", "unsure"]))
                complete = bool(rng.random() < 0.8)
                labels = {rule: ("yes" if rng.random() < 0.2 else "no") for rule in rules}
                attempts = int(rng.integers(1, 3))
                rows.append({"scenario_id": sid, "draw": draw, "full": {
                    "verdict": verdict, "attempts": attempts, "truncated_attempts": attempts - 1,
                    "map": labels if complete else {}, "map_complete": complete,
                    "think_tokens": int(rng.integers(100, 900)), "empty_think": bool(rng.random() < 0.01)}})
        return rows

    def write(path, rows, weights=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        identity = {"model_weights_sha256": weights, "adapter_sha256": None, "serving_dtype": "float32",
                    "table_sha256": sha256_file(path)}
        (path.parent / f"{path.stem}.identity.json").write_text(json.dumps(identity), encoding="utf-8")

    base = directory / "base"
    write(base / "bank.jsonl", rule_rows(ids, 4))
    write(base / "full.jsonl", full_rows(ids, 8, 0.5))
    (base / "scenarios.jsonl").write_text("".join(
        json.dumps({"scenario_id": sid, "source": source, "subset": subset}) + "\n"
        for sid, source, subset in zip(ids, sources, subsets)), encoding="utf-8")
    stage1 = sorted(CASE_STAGE1)
    for step, yes_rate, shift in ((100, 0.5, 0.0), (200, 0.9, 0.15)):
        save = directory / "saves" / f"checkpoint-{step}"
        write(save / "stage1.jsonl", full_rows(stage1, 2, yes_rate), weights=f"{step:064d}")
        write(save / "probe.jsonl", rule_rows(sorted(CASE_PROBE), 1, shift), weights=f"{step:064d}")
    (directory / "rules.jsonl").write_text("".join(json.dumps({"id": rule, "rule": rule}) + "\n" for rule in rules),
                                           encoding="utf-8")
    return rules


#: What the report keeps of a run, rounded so a digest holds across machines.
def _rounded(value):
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def _case_projection(tolerances, saves):
    """The numbers the report publishes, in names that do not depend on the code that made them."""
    return _rounded({
        "tolerances": {name: {"quantile": entry["quantile"], "floor_n": (entry.get("floor") or {}).get("n"),
                              "tolerance": entry["tolerance"], "null": entry["null"]}
                       for name, entry in tolerances["statistics"].items()},
        "stage1": tolerances["stage1_scenarios"], "probe": tolerances["probe_scenarios"], "base_m1": tolerances["base_m1"],
        "saves": {step: {"m1": save["m1"], "delta": save["delta_m1_vs_base"],
                         "statistics": {name: {"gate": cell["gate"], "value": cell.get("delta", cell.get("value")),
                                               "limit": cell.get("tolerance", cell.get("threshold")),
                                               "within": cell["within_tolerance"]}
                                        for name, cell in save["statistics"].items()}}
                  for step, save in saves.items()},
    })


CASE_SETTINGS = {
    "null_model": {"resamples": 300, "seed": 16, "level": 0.95}, "interval": {"resamples": 300, "seed": 16, "level": 0.95},
    "floored_gates": ["G2", "G3", "G4"],
    "stage1": {"draws": 2, "dev_scenarios": 18, "seed": 16, "base_draws": 8, "base_reference_draws": [0, 1, 2, 3],
               "subsets": RAW_SETTINGS["stage1"]["subsets"], "many_yes": 2,
               "thinking": {"median_ratio": 0.5, "max_empty_share": 0.02}},
    "probe": {"scenarios": 9, "seed": 16, "bank_draws": 4, "base_reference_draw": 0},
}
_CASE = rp.Scenarios(tuple(f"s{i:02d}" for i in range(38)), tuple(("a", "b", "c")[i % 3] for i in range(38)),
                     tuple(["dev"] * 30 + ["dev_nonyes"] * 8))
CASE_STAGE1 = rp.stage1_scenarios(_CASE, rp.load_settings({**RAW_SETTINGS, **CASE_SETTINGS}))
CASE_PROBE = rp.probe_scenarios(_CASE, 9, 16)
#: The digest of this case's ``_case_projection``, as the checkpoint report's code computes it.
CASE_DIGEST = "c9d2a5bac0dc866d71415d494e3231dac77c2e752d9d702f783589847f804356"


def test_a_tiny_synthetic_run_reports_the_numbers_the_digest_pins(tmp_path):
    _case_tables(tmp_path)
    rules = tmp_path / "rules.jsonl"
    settings = {**CASE_SETTINGS, "ruleset_sha256": sha256_file(rules)}
    (tmp_path / "settings.yaml").write_text(json.dumps(settings))
    document = rp.report(tmp_path / "settings.yaml", base=tmp_path / "base", saves=tmp_path / "saves", rules=rules,
                         out=tmp_path / "out")
    tolerances = json.loads((tmp_path / "out/tolerances.json").read_text())
    assert canonical_digest(_case_projection(tolerances, document["saves"])) == CASE_DIGEST
