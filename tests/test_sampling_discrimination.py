from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ruleforge.evaluate.inference import (
    BOOTSTRAP_TEST_METHOD,
    TEST_METHOD,
    DrawGrid,
    load_condition_outcomes,
    RepeatedSamplingError,
    clustered_paired_randomization_p,
    clustered_paired_randomization_statistic,
    percentile_bootstrap_p,
    scenario_cluster_bootstrap,
    statistic_interval,
)
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.aggregate import policy_discrimination, analyze_campaign, policy_families
from ruleforge.evaluate.campaign import build_plan, load_conditions, run_plan
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.identity import load_sampling_manifest
from ruleforge.evaluate.discrimination import (
    SINGLE_STRATUM,
    DiscriminationError,
    Strata,
    VerdictGrids,
    arm_statistics,
    cross_arm_decomposition,
    cross_inconsistency,
    kappa_contrast,
    kappa_features,
    kappa_parts,
    kappa_summary,
    load_rules,
    same_call_statistics,
    read_strata,
    reclassify_forced_maps,
    stratified_differences,
)
from ruleforge.evaluate.runner import CampaignRunError
from tests.test_sampling_campaign import backend_for
from tests.test_sampling_training import POLICIES, _training_like_campaign, family_of, policy_spec, run_policy_conditions, write_adapter

PHI = {"yes": 1.0, "unsure": 0.5, "no": 0.0}
BOOTSTRAP = dict(n_resamples=200, ci_level=0.95, seed=16)


def verdicts(full, rule_max, name="c") -> VerdictGrids:
    """Grids from label rows, one row per scenario."""
    ids = tuple(f"{i:064x}" for i in range(len(full)))
    to_phi = np.vectorize(PHI.__getitem__)
    return VerdictGrids(
        DrawGrid(name, "full_set_phi", ids, to_phi(np.array(full))),
        DrawGrid(name, "max_rule_phi", ids, to_phi(np.array(rule_max))),
    )


#: Four scenarios, two draws, two sources (the worked example in the test below).
HAND = verdicts(
    [["yes", "yes"], ["yes", "no"], ["no", "no"], ["unsure", "yes"]],
    [["yes", "yes"], ["yes", "yes"], ["no", "yes"], ["unsure", "no"]],
)
HAND_STRATA = Strata("source", ("A", "A", "B", "B"))


def point(features, statistic):
    return statistic(np.asarray(features).mean(axis=0, keepdims=True))[0]


def loop_kappa(grids: VerdictGrids, strata: Strata) -> float:
    """The kappa formula written as a loop, draw by draw."""
    rows = [
        (strata.labels[i], grids.full.values[i, d], grids.rule_max.values[i, d])
        for i in range(grids.full.n_scenarios) for d in range(grids.full.n_draws)
    ]
    p_o = np.mean([f == m for _, f, m in rows])
    p_e = 0.0
    for stratum in set(strata.labels):
        inside = [(f, m) for s, f, m in rows if s == stratum]
        weight = len(inside) / len(rows)
        p_e += weight * sum(np.mean([f == c for f, _ in inside]) * np.mean([m == c for _, m in inside]) for c in PHI.values())
    return (p_o - p_e) / (1 - p_e)


def test_kappa_equals_a_hand_computation():
    kappa, p_o, p_e = kappa_parts(kappa_features(HAND, HAND_STRATA).mean(axis=0, keepdims=True), 2)[0]
    assert (kappa, p_o, p_e) == pytest.approx((1 / 7, 5 / 8, 9 / 16))
    # One stratum: F (y 4, n 3, u 1), M (y 5, n 2, u 1); P_e = 27/64, kappa = 13/37.
    single = Strata(None, (SINGLE_STRATUM,) * 4)
    assert kappa_parts(kappa_features(HAND, single).mean(axis=0, keepdims=True), 1)[0, 0] == pytest.approx(13 / 37)
    summary = kappa_summary(HAND, HAND_STRATA, **BOOTSTRAP)
    assert summary["estimate"] == pytest.approx(1 / 7) and summary["p_chance"] == pytest.approx(9 / 16)
    assert summary["strata"] == {"field": "source", "counts": {"A": 2, "B": 2}}
    assert summary["ci_low"] <= summary["estimate"] <= summary["ci_high"]


def test_kappa_is_the_design_formula_on_random_draws():
    rng = np.random.default_rng(3)
    labels = np.array(list(PHI))
    grids = verdicts(labels[rng.integers(0, 3, (40, 5))], labels[rng.integers(0, 3, (40, 5))])
    strata = Strata("source", tuple(rng.choice(["w", "s", "x"], 40)))
    features = kappa_features(grids, strata)
    assert kappa_parts(features.mean(axis=0, keepdims=True), 3)[0, 0] == pytest.approx(loop_kappa(grids, strata))


def test_an_always_yes_judge_has_kappa_zero_and_a_constant_arm_none():
    always_yes = verdicts([["yes", "yes"]] * 4, [["yes", "no"], ["no", "no"], ["yes", "yes"], ["unsure", "yes"]])
    assert kappa_summary(always_yes, HAND_STRATA, **BOOTSTRAP)["estimate"] == pytest.approx(0.0, abs=1e-12)
    constant = verdicts([["yes", "yes"]] * 4, [["yes", "yes"]] * 4)
    summary = kappa_summary(constant, HAND_STRATA, **BOOTSTRAP)
    assert summary["estimate"] is None and summary["ci_low"] is None and summary["n_undefined_replicates"] == 200


def test_the_kappa_contrast_is_member_minus_reference_with_a_bootstrap_p():
    member = verdicts([["yes", "yes"], ["yes", "yes"], ["no", "yes"], ["unsure", "no"]],
                      [["yes", "yes"], ["yes", "yes"], ["no", "yes"], ["unsure", "no"]], name="m")
    row = kappa_contrast(HAND, member, HAND_STRATA, n_permutations=500, **BOOTSTRAP)
    assert (row["a"], row["b"], row["orientation"]) == ("c", "m", "kappa(member) - kappa(reference)")
    assert row["kappa_a"]["estimate"] == pytest.approx(1 / 7) and row["kappa_b"]["estimate"] == pytest.approx(1.0)
    assert row["estimate"] == pytest.approx(1 - 1 / 7) and row["p_value_defined"]
    fa, fb = kappa_features(HAND, HAND_STRATA), kappa_features(member, HAND_STRATA)
    _, replicates = scenario_cluster_bootstrap(
        np.hstack([fa, fb]),
        lambda m: kappa_parts(m[:, fa.shape[1]:], 2)[:, :1] - kappa_parts(m[:, : fa.shape[1]], 2)[:, :1],
        **BOOTSTRAP,
    )
    assert row["p_value"] == percentile_bootstrap_p(replicates[:, 0]) and row["test_method"] == BOOTSTRAP_TEST_METHOD
    assert 0 < row["p_randomization"] <= 1 and row["randomization_test_method"] == TEST_METHOD
    assert row["n_permutations"] == 500 and row["grid_digests_b"] == member.digests
    constant = verdicts([["yes", "yes"]] * 4, [["yes", "yes"]] * 4, name="k")
    undefined = kappa_contrast(HAND, constant, HAND_STRATA, n_permutations=500, **BOOTSTRAP)
    assert undefined["estimate"] is None and (undefined["p_value"], undefined["p_value_defined"]) == (1.0, False)
    assert undefined["p_randomization"] is None
    with pytest.raises(DiscriminationError, match="same scenarios"):
        kappa_contrast(HAND, verdicts([["yes", "yes"]], [["yes", "yes"]]), HAND_STRATA, n_permutations=500, **BOOTSTRAP)
    with pytest.raises(DiscriminationError, match="3 strata for 4 scenarios"):
        kappa_features(HAND, Strata("source", ("A", "A", "B")))


def test_the_bootstrap_p_reads_the_difference_off_its_replicates():
    # 3 of 99 replicates at or below 0: p = 2 * (3 + 1) / 100.
    assert percentile_bootstrap_p(np.r_[-0.2, 0.0, -1e-3, np.linspace(0.1, 1, 96)]) == pytest.approx(0.08)
    replicates = -np.linspace(0.01, 1, 199)
    assert percentile_bootstrap_p(replicates) == pytest.approx(2 / 200)
    assert statistic_interval(-0.5, replicates, 0.95)["ci_high"] < 0
    # undefined replicates are left out, as the interval leaves them; float noise around 0 is 0
    assert percentile_bootstrap_p(np.r_[np.nan, np.linspace(0.1, 1, 99)]) == pytest.approx(2 / 100)
    assert percentile_bootstrap_p(np.full(5, np.nan)) == 1.0
    assert percentile_bootstrap_p(np.full(200, 1e-15)) == 1.0
    assert percentile_bootstrap_p(np.r_[-1.0, 1.0]) == 1.0


def _shifted_arm(rng, f_yes, m_yes, rho, n, s):
    """F copies M with probability rho, else an independent draw at a scenario-level yes-rate near f_yes."""
    pf = rng.beta(f_yes * 4, (1 - f_yes) * 4, size=n)
    pm = rng.beta(m_yes * 4, (1 - m_yes) * 4, size=n)
    rule_max = (rng.random((n, s)) < pm[:, None]).astype(float)
    full = np.where(rng.random((n, s)) < rho, rule_max, (rng.random((n, s)) < pf[:, None]).astype(float))
    return full, rule_max


def _copy_probability(f_yes, m_yes, kappa):
    """The rho at which :func:`_shifted_arm`'s population kappa is ``kappa`` (closed form, bisected)."""
    def population(rho):
        p_o = rho + (1 - rho) * (f_yes * m_yes + (1 - f_yes) * (1 - m_yes))
        f_yes_share = rho * m_yes + (1 - rho) * f_yes
        p_e = f_yes_share * m_yes + (1 - f_yes_share) * (1 - m_yes)
        return (p_o - p_e) / (1 - p_e)

    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if population(mid) < kappa else (low, mid)
    return (low + high) / 2


@pytest.mark.slow
def test_equal_kappas_with_shifted_marginals_are_not_called_an_increase():
    """The size of family 2's test where the training comparison expects to be -- the policy yes-shifted, its kappa the base's."""
    n, s, reps = 500, 5, 1000
    rng = np.random.default_rng(91)
    strata = Strata("source", tuple("abc"[i % 3] for i in range(n)))
    base, policy = (0.3, 0.93), (0.97, 0.95)
    rho_base, rho_policy = _copy_probability(*base, 0.05), _copy_probability(*policy, 0.05)
    ids = tuple(f"{i:064x}" for i in range(n))
    swap = bootstrap = 0
    for rep in range(reps):
        arms = [
            VerdictGrids(DrawGrid(name, "full_set_phi", ids, full), DrawGrid(name, "max_rule_phi", ids, rule_max))
            for name, (full, rule_max) in (
                ("a", _shifted_arm(rng, *base, rho_base, n, s)), ("b", _shifted_arm(rng, *policy, rho_policy, n, s))
            )
        ]
        row = kappa_contrast(*arms, strata, n_resamples=1000, ci_level=0.95, seed=16 + rep, n_permutations=1000)
        swap += row["p_randomization"] < 0.05 and row["estimate"] > 0
        bootstrap += row["p_value"] < 0.05 and row["estimate"] > 0
    assert bootstrap / reps <= 0.025 < 0.05 < swap / reps


def test_the_stratified_rates_yes_rate_and_floor_equal_a_hand_computation():
    arm = arm_statistics(HAND, HAND_STRATA, **BOOTSTRAP)
    # M = yes on 5 draws, 2 inconsistent; M != yes on 3, 1 (y vs n); M = unsure 1, 0; M = no 2, 1.
    rates = {name: (entry["estimate"], entry["n_draws"]) for name, entry in arm["m_stratified"].items()}
    assert rates == pytest.approx({"M=yes": (2 / 5, 5), "M!=yes": (1 / 3, 3), "M=unsure": (0.0, 1), "M=no": (1 / 2, 2)})
    assert arm["yes_rate"]["estimate"] == pytest.approx(4 / 8) and arm["always_yes_floor"]["estimate"] == pytest.approx(3 / 8)
    assert arm["kappa"]["estimate"] == pytest.approx(1 / 7)
    assert (arm["n_scenarios"], arm["n_decoder_draws"]) == (4, 2) and arm["grid_digests"] == HAND.digests


def test_the_harm_flag_fires_only_on_a_rise_above_the_threshold_that_excludes_zero():
    worse = verdicts([["yes", "yes"], ["yes", "yes"], ["yes", "yes"], ["yes", "yes"]],
                     [["yes", "yes"], ["yes", "yes"], ["no", "yes"], ["unsure", "no"]], name="w")
    flagged = stratified_differences(HAND, worse, **BOOTSTRAP)
    assert flagged["orientation"] == "rate(member) - rate(reference)"
    assert flagged["differences"]["M!=yes"]["estimate"] == pytest.approx(1 - 1 / 3)
    assert flagged["harm_flag"] == {"stratum": "M!=yes", "threshold": 0.1, "flagged": True}
    assert stratified_differences(HAND, HAND, **BOOTSTRAP)["harm_flag"]["flagged"] is False
    all_yes = verdicts([["yes", "yes"]] * 4, [["yes", "yes"]] * 4, name="y")
    undefined = stratified_differences(HAND, all_yes, **BOOTSTRAP)
    assert undefined["differences"]["M!=yes"]["estimate"] is None and not undefined["harm_flag"]["flagged"]


REFERENCE_2 = verdicts([["yes", "no"], ["no", "no"]], [["yes", "yes"], ["no", "yes"]], name="r")
MEMBER_2 = verdicts([["yes", "yes"], ["yes", "no"]], [["yes", "yes"], ["yes", "yes"]], name="m")


def test_the_decomposition_equals_a_hand_computation():
    parts = cross_arm_decomposition(REFERENCE_2, MEMBER_2, **BOOTSTRAP)
    cells = {name: entry["estimate"] for name, entry in parts["cells"].items()}
    assert cells == pytest.approx({"F_reference|M_reference": 0.5, "F_member|M_reference": 0.25,
                                   "F_reference|M_member": 0.75, "F_member|M_member": 0.25})
    # total = 1/2 - 1/4; verdicts: ((1/2 - 1/4) + (3/4 - 1/4)) / 2; references: ((1/2 - 3/4) + (1/4 - 1/4)) / 2
    assert parts["total"]["estimate"] == pytest.approx(0.25)
    assert parts["verdict_effect"]["estimate"] == pytest.approx(0.375)
    assert parts["reference_effect"]["estimate"] == pytest.approx(-0.125)
    assert parts["method"] == "independent_draw_expected_shares_v1"


def test_a_cross_cell_is_the_mean_over_every_draw_pair():
    rng = np.random.default_rng(5)
    labels = np.array(list(PHI))
    a = verdicts(labels[rng.integers(0, 3, (6, 3))], labels[rng.integers(0, 3, (6, 3))], name="a")
    b = verdicts(labels[rng.integers(0, 3, (6, 3))], labels[rng.integers(0, 3, (6, 3))], name="b")
    pairs = [
        np.mean([a.full.values[i, d] != b.rule_max.values[i, e] for d in range(3) for e in range(3)]) for i in range(6)
    ]
    assert cross_inconsistency(a, b) == pytest.approx(pairs)
    # within one arm its expectation is the flat rate's; here it is a different estimate of it
    parts = cross_arm_decomposition(a, b, **BOOTSTRAP)
    assert parts["verdict_effect"]["estimate"] + parts["reference_effect"]["estimate"] == pytest.approx(parts["total"]["estimate"])


def _binary_grids(seed=11, n=30, s=3):
    rng = np.random.default_rng(seed)
    ids = tuple(f"{i:064x}" for i in range(n))
    return (DrawGrid("a", "q", ids, rng.integers(0, 2, (n, s)).astype(float)),
            DrawGrid("b", "q", ids, rng.integers(0, 2, (n, s)).astype(float)))


def test_with_the_mean_as_statistic_the_randomization_is_the_flat_rates():
    a, b = _binary_grids()
    flat = clustered_paired_randomization_p(a, b, n_permutations=5000, seed=16)
    general = clustered_paired_randomization_statistic(
        a.scenario_means[:, None], b.scenario_means[:, None], lambda means: means, n_permutations=5000, seed=16,
    )
    assert general["p_value"] == flat["p_value"] and general["observed_difference"] == pytest.approx(flat["observed_difference"])
    assert general["method"] == flat["method"] and general["n_undefined_permutations"] == 0


def test_the_cluster_bootstrap_resamples_scenario_rows_together():
    a, b = _binary_grids(n=9)
    features = np.column_stack([a.scenario_means, b.scenario_means])
    point_value, replicates = scenario_cluster_bootstrap(features, lambda m: m[:, :1] - m[:, 1:], n_resamples=700, ci_level=0.95, seed=4)
    rng = np.random.default_rng(4)
    expected = []
    for start in range(0, 700, 500):
        picks = rng.integers(0, 9, size=(min(500, 700 - start), 9))
        expected.extend(features[picks].mean(axis=1)[:, 0] - features[picks].mean(axis=1)[:, 1])
    assert replicates[:, 0] == pytest.approx(expected)
    assert point_value[0] == pytest.approx(a.rate - b.rate)


def test_undefined_replicates_and_permutations_are_counted():
    features = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    def ratio(m):
        return np.divide(m[:, :1], m[:, 1:], out=np.full((m.shape[0], 1), np.nan), where=m[:, 1:] > 0)

    value, replicates = scenario_cluster_bootstrap(features, ratio, n_resamples=200, ci_level=0.95, seed=1)
    summary = statistic_interval(value[0], replicates[:, 0], 0.95)
    assert summary["estimate"] == 2.0 and summary["n_undefined_replicates"] > 0
    assert statistic_interval(float("nan"), np.full(5, np.nan), 0.95) == {
        "estimate": None, "ci_low": None, "ci_high": None, "sd": None, "interval_basis": "scenario_cluster_bootstrap",
        "statistic_resampling": "recomputed_on_every_replicate", "n_undefined_replicates": 5,
    }
    other = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    test = clustered_paired_randomization_statistic(features, other, ratio, n_permutations=200, seed=2)
    assert test["n_undefined_permutations"] > 0 and 0 < test["p_value"] <= 1
    undefined = clustered_paired_randomization_statistic(
        np.zeros((3, 2)), other, ratio, n_permutations=200, seed=2
    )
    assert undefined["p_value"] is None and undefined["observed_difference"] is None


@pytest.mark.parametrize(
    ("a", "b", "kwargs", "message"),
    [
        (np.zeros((1, 2)), np.zeros((1, 2)), {}, "n_scenarios >= 2"),
        (np.zeros((3, 2)), np.zeros((3, 3)), {}, "differ in shape"),
        (np.array([[0.0], [np.nan]]), np.zeros((2, 1)), {}, "missing"),
        (np.zeros((3, 2)), np.zeros((3, 2)), {"n_permutations": 10}, "at least 100"),
    ],
)
def test_the_randomization_refuses_what_it_cannot_permute(a, b, kwargs, message):
    with pytest.raises(RepeatedSamplingError, match=message):
        clustered_paired_randomization_statistic(a, b, lambda m: m[:, :1], **kwargs)


def _write_rows(root: Path, rows) -> Path:
    path = root / "draw-000" / "evaluations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"scenario_id": sid, "scenario": scenario}) + "\n" for sid, scenario in rows))
    return root


def test_the_strata_come_from_the_judged_scenario_rows(tmp_path):
    a = _write_rows(tmp_path / "a", [("s1", {"source": "stress"}), ("s2", {"source": "weird"})])
    b = _write_rows(tmp_path / "b", [("s2", {"source": "weird"}), ("s1", {"source": "stress"})])
    strata = read_strata([a, b], ["s1", "s2"])
    assert strata == Strata("source", ("stress", "weird")) and strata.names == ["stress", "weird"]
    none = _write_rows(tmp_path / "c", [("s1", {}), ("s2", {})])
    assert read_strata([none], ["s1", "s2"]) == Strata(None, (SINGLE_STRATUM,) * 2)
    partial = _write_rows(tmp_path / "d", [("s1", {"source": "x"}), ("s2", {})])
    with pytest.raises(DiscriminationError, match="1 of 2 scenarios carry no 'source'"):
        read_strata([partial], ["s1", "s2"])
    other = _write_rows(tmp_path / "e", [("s1", {"source": "wild"}), ("s2", {"source": "weird"})])
    with pytest.raises(DiscriminationError, match="another condition's rows 'stress'"):
        read_strata([a, other], ["s1", "s2"])
    with pytest.raises(DiscriminationError, match="not the campaign's 3 scenarios"):
        read_strata([a], ["s1", "s2", "s3"])


@pytest.fixture
def policy_root(tmp_path, modelspec, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    return run_policy_conditions(tmp_path, modelspec)


def _inputs(root: Path, condition_id: str):
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    scenarios = ScenarioManifest.load(root / "scenario_manifest.json")
    cells, _ = load_condition_outcomes(root / condition_id, expected_scenario_ids=scenarios.scenario_ids,
                                       expected_draws=manifest["expected_decoder_draws"])
    return manifest, scenarios, cells


@pytest.mark.parametrize("style", ["forced", "forced-grpo"])
def test_same_call_is_reread_from_the_accepted_raw_outputs(policy_root, tmp_path, style):
    cid = "qwen/figure5/base"
    spec = dataclasses.replace(load_conditions(policy_root)[cid], prompt_style=style)
    manifest, scenarios, cells = _inputs(policy_root, cid)
    rules = load_rules(tmp_path / "rules.jsonl", spec.rule_set_sha256)
    outcomes = reclassify_forced_maps(policy_root / cid, spec, campaign_id=manifest["campaign_id"],
                                      scenario_ids=scenarios.scenario_ids, n_draws=2, rules=rules, cells=cells)
    # every full-set answer is forced("no") with an all-no complete map: a same-call rate of 0 everywhere
    assert len(outcomes) == 6 and all(o.complete and o.in_context_contradiction == 0.0 for o in outcomes.values())
    summary = same_call_statistics(outcomes, scenario_ids=scenarios.scenario_ids, n_draws=2, **BOOTSTRAP)
    assert summary["completeness"]["estimate"] == 1.0 and summary["method_b_rate"]["estimate"] == 0.0
    assert summary["incomplete_reasons"] == {} and summary["between_arm_contrast"] is None


def test_same_call_reads_a_published_copy_s_accepted_calls_alone_as_the_recorded_root(policy_root, tmp_path):
    import shutil

    from ruleforge import hub
    from ruleforge.core.constants import PUBLISHED_COPY_FNAME, PUBLISHED_LEDGER_FNAME
    from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME

    cid = "qwen/figure5/base"
    spec = dataclasses.replace(load_conditions(policy_root)[cid], prompt_style="forced")
    manifest, scenarios, cells = _inputs(policy_root, cid)
    rules = load_rules(tmp_path / "rules.jsonl", spec.rule_set_sha256)
    kwargs = dict(campaign_id=manifest["campaign_id"], scenario_ids=scenarios.scenario_ids, n_draws=2, rules=rules,
                  cells=cells)
    # a superseded full-ruleset row beside the accepted ones: the published copy leaves it out
    raw = policy_root / cid / "draw-000" / RAW_OUTPUTS_FNAME
    first = json.loads(raw.read_text().splitlines()[0])
    raw.write_text(raw.read_text() + json.dumps({**first, "component_id": "full_ruleset", "semantic_attempt_id": 9})
                   + "\n")
    recorded = reclassify_forced_maps(policy_root / cid, spec, **kwargs)
    published = tmp_path / "published"
    shutil.copytree(policy_root, published)
    (published / PUBLISHED_COPY_FNAME).write_text("{}")
    for draw in ("draw-000", "draw-001"):
        relative = f"{cid}/{draw}"
        (published / relative / LEDGER_FNAME).unlink()
        (published / relative / PUBLISHED_LEDGER_FNAME).write_bytes(hub._published_ledger(
            policy_root, f"{relative}/{LEDGER_FNAME}", hub.PROFILES["training-qwen"].ledger))
        source = (policy_root / relative / RAW_OUTPUTS_FNAME).read_bytes()
        (published / relative / RAW_OUTPUTS_FNAME).write_bytes(hub._published_raw_outputs(
            policy_root, f"{relative}/{RAW_OUTPUTS_FNAME}", source, relative, hub.Publication(),
            hub.PROFILES["training-gemma"].projections[RAW_OUTPUTS_FNAME]))
    rows = [json.loads(line) for line in (published / cid / "draw-000" / RAW_OUTPUTS_FNAME).read_text().splitlines()]
    assert len(rows) == len(scenarios.scenario_ids) and all(row["component_id"] == "full_ruleset" for row in rows)
    assert reclassify_forced_maps(published / cid, spec, **kwargs) == recorded


def test_same_call_refuses_outputs_that_are_not_what_the_ledger_recorded(policy_root, tmp_path):
    cid = "qwen/figure5/base"
    spec = load_conditions(policy_root)[cid]
    manifest, scenarios, cells = _inputs(policy_root, cid)
    rules = load_rules(tmp_path / "rules.jsonl", spec.rule_set_sha256)
    kwargs = dict(campaign_id=manifest["campaign_id"], scenario_ids=scenarios.scenario_ids, n_draws=2, rules=rules)
    first = next(iter(cells))
    edited = {**cells, first: {**cells[first], "full_set_judgment": "yes"}}
    with pytest.raises(DiscriminationError, match="reads as 'no', the ledger says 'no' and the bundle outcome 'yes'"):
        reclassify_forced_maps(policy_root / cid, spec, cells=edited, **kwargs)
    with pytest.raises(DiscriminationError, match="not the rules its bundles judged"):
        reclassify_forced_maps(policy_root / cid, spec, cells=cells, **{**kwargs, "rules": rules[:1]})
    with pytest.raises(DiscriminationError, match="has no accepted bundle"):
        reclassify_forced_maps(policy_root / cid, spec, cells=cells, **{**kwargs, "scenario_ids": ["f" * 64]})
    with pytest.raises(DiscriminationError, match="the same-call rate needs the forced prompt"):
        reclassify_forced_maps(policy_root / cid, dataclasses.replace(spec, prompt_style="naive"), cells=cells, **kwargs)
    raw = policy_root / cid / "draw-000" / "raw_outputs.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in rows:
        if row["component_id"] == "full_ruleset":
            row["raw_response"] += " "
            break
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(DiscriminationError, match="not the text the ledger recorded"):
        reclassify_forced_maps(policy_root / cid, spec, cells=cells, **kwargs)


def test_the_ruleset_must_be_the_one_the_conditions_judged(tmp_path):
    path = tmp_path / "rules.jsonl"
    path.write_text('{"id": "Rule 1", "rule": "r"}\n')
    with pytest.raises(DiscriminationError, match="the conditions judged ruleset aaaaaaaaaaaa"):
        load_rules(path, "a" * 64)


def test_a_forced_policy_family_needs_its_ruleset(policy_root, tmp_path):
    with pytest.raises(CampaignRunError, match="pass rule_set_path"):
        analyze_campaign(policy_root, n_permutations=100, n_resamples=100)
    payload = analyze_campaign(policy_root, n_permutations=100, n_resamples=100, rule_set_path=tmp_path / "rules.jsonl")
    (family,) = payload["policy_families"].values()
    assert set(family["arms"]) == set(POLICIES) and all("method_b" in arm for arm in family["arms"].values())
    rows = family["discrimination_contrasts"]
    assert [(row["policy_a"], row["policy_b"]) for row in rows] == [("base", "ckpt125"), ("base", "ckpt200")]
    assert all(row["family_size"] == 2 and row["family"] == "discrimination_contrasts" for row in rows)
    assert set(family["decompositions"]) == set(family["stratified_differences"]) == {"ckpt125", "ckpt200"}
    assert family["strata"] == {"field": None, "counts": {SINGLE_STRATUM: 3}}


def test_a_contradiction_that_is_not_the_verdicts_is_refused(policy_root, tmp_path):
    outcomes = policy_root / "qwen/figure5/ckpt200" / "draw-000" / "bundle_outcomes.jsonl"
    rows = [json.loads(line) for line in outcomes.read_text().splitlines()]
    rows[0]["contradiction"] = 1.0 - rows[0]["contradiction"]
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(CampaignRunError, match="contradiction is not phi"):
        analyze_campaign(policy_root, n_permutations=100, n_resamples=100, rule_set_path=tmp_path / "rules.jsonl")


def test_a_policy_family_of_label_prompts_needs_no_ruleset_and_reports_no_same_call(tmp_path, modelspec, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    for policy in POLICIES:
        adapter = None if policy == "base" else write_adapter(tmp_path / "adapters", policy)
        cfg = _training_like_campaign(tmp_path, policy, adapter, **{"eval_cfg.reasoning_prompt_style": "naive"})
        backend = backend_for("naive", script={(0, "Rule 2"): ["yes"]} if policy == "base" else None)
        backend.adapter_path = adapter
        task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
        run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    payload = analyze_campaign(tmp_path / "root", n_permutations=100, n_resamples=100)
    (family,) = payload["policy_families"].values()
    assert not any("method_b" in arm for arm in family["arms"].values())
    assert family["arms"]["base"]["m_stratified"]["M=yes"]["estimate"] == 1.0
    assert [row["family_size"] for row in family["discrimination_contrasts"]] == [2, 2]


def test_a_family_with_no_analysed_member_gets_empty_blocks():
    registered = family_of(policy_spec("base", None), policy_spec("ckpt125", "1" * 64))
    name = next(iter(policy_families(registered)))
    groups = {name: {"missing_policies": ["ckpt200"]}}
    rows = policy_discrimination(
        groups, registered, {}, {}, condition_dirs={}, campaign_ids={}, n_draws=2, rule_set_path=None, stacks={},
        allow_stack_mismatch=False, n_resamples=100, ci_level=0.95, seed=16, n_permutations=100,
    )
    assert rows == [] and groups[name]["arms"] == {} and groups[name]["strata"] is None
    assert groups[name]["discrimination_contrasts"] == [] and groups[name]["decompositions"] == {}
