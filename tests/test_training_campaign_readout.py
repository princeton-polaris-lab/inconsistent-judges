from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from ruleforge.experiments.training import campaign_readout as producer
from ruleforge.experiments.training.campaign_readout import (
    TrainingReadoutError,
    build_from_campaign,
    claim,
    disclosure,
    read_parity,
    read_parity_arms,
)
from ruleforge.experiments.training.figure import build_figure, draw, panel_values
from ruleforge.grpo.lora_parity import PARITY_PRESETS, thresholds_to_json
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import load_conditions
from ruleforge.core.files import write_json_sidecar
from tests.support.parity import preset_thresholds
from tests.support.training_campaign import (
    FIXTURE,
    GEMMA,
    PARITY,
    POLICIES,
    RUNTIME,
    _edit,
    _roots,  # noqa: F401 - an autouse fixture
    assert_panel_a_draws_the_flat_rates_alone,
    mark_final,
    run_training_campaign,
    scripted_answers,
    source_of,
    write_parity,
)
from tests.test_sampling_campaign import set_recorded_runtime


@pytest.fixture
def root(tmp_path, modelspec):
    root = run_training_campaign(tmp_path, modelspec)
    analyze_campaign(root, n_resamples=200, n_permutations=2000, rule_set_path=tmp_path / "rules.jsonl")
    mark_final(root)
    return root


@pytest.fixture
def parity(tmp_path):
    """Each policy's prefill and decode reports."""
    decode = {"rho": 0.9998, "rho_ci": [0.9990, 1.0007], "rho_halfwidth": 0.00085}
    return {
        "grpo": [write_parity(tmp_path, "grpo"), write_parity(tmp_path, "grpo", "decode", buckets={"all": decode})],
        "final": [
            write_parity(tmp_path, "final", buckets={"all": {"rho": 0.9993, "rho_ci": [0.9981, 1.0004]}}),
            write_parity(tmp_path, "final", "decode", buckets={"all": decode}),
        ],
    }


@pytest.fixture
def golden(root, parity):
    return build_from_campaign(root / "aggregate" / "estimates.json", parity=parity,
                               labels={"base": "Qwen3-8B", "grpo": "GRPO", "final": "GRPO (final)"})


def hand_kappa(policy: str) -> float:
    """Cohen's kappa of phi(F) against M, chance agreement within source -- the design's formula, by loops."""
    phi = {"yes": 1.0, "unsure": 0.5, "no": 0.0}
    rows = []
    for (i, _), answer in scripted_answers(policy).items():
        rows.append((source_of(i), phi[answer["F"]], max(phi[answer["Rule 1"]], phi[answer["Rule 2"]])))
    p_o = sum(f == m for _, f, m in rows) / len(rows)
    p_e = 0.0
    for source in {s for s, _, _ in rows}:
        inside = [(f, m) for s, f, m in rows if s == source]
        weight = len(inside) / len(rows)
        p_e += weight * sum(
            (sum(f == c for f, _ in inside) / len(inside)) * (sum(m == c for _, m in inside) / len(inside))
            for c in (0.0, 0.5, 1.0)
        )
    return (p_o - p_e) / (1 - p_e)


def test_the_arms_are_the_hand_derived_numbers(golden):
    arms = golden["arms"]
    assert list(arms) == list(POLICIES) and golden["reference_policy"] == "base"
    assert [arms[p]["flat"]["estimate"] for p in POLICIES] == pytest.approx([22 / 24, 0.0, 3 / 24])
    assert [arms[p]["kappa"]["estimate"] for p in POLICIES] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert [hand_kappa(p) for p in POLICIES] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    # base: P_o 2/24, P_e = 1/2 * 0 (stress: F never yes, M always yes) + 1/2 * (1 * 2/12) (wildchat)
    assert (arms["base"]["kappa"]["p_observed"], arms["base"]["kappa"]["p_chance"]) == pytest.approx((1 / 12, 1 / 12))
    assert [arms[p]["yes_rate"]["estimate"] for p in POLICIES] == pytest.approx([0.0, 21 / 24, 1.0])
    assert [arms[p]["always_yes_floor"]["estimate"] for p in POLICIES] == pytest.approx([2 / 24, 3 / 24, 3 / 24])
    stratified = {p: arms[p]["m_stratified"] for p in POLICIES}
    assert {p: (s["M=yes"]["estimate"], s["M!=yes"]["estimate"]) for p, s in stratified.items()} == pytest.approx(
        {"base": (1.0, 0.0), "grpo": (0.0, 0.0), "final": (0.0, 1.0)}
    )
    assert {p: (s["M=yes"]["n_draws"], s["M!=yes"]["n_draws"]) for p, s in stratified.items()} == {
        "base": (22, 2), "grpo": (21, 3), "final": (21, 3),
    }
    assert stratified["grpo"]["M=unsure"] == {**stratified["grpo"]["M=unsure"], "estimate": 0.0, "n_draws": 1}
    assert stratified["base"]["M=unsure"]["estimate"] is None and stratified["base"]["M=unsure"]["n_draws"] == 0
    same_call = {p: arms[p]["same_call"] for p in POLICIES}
    assert {p: (m["n_complete_maps"], m["n_outputs"]) for p, m in same_call.items()} == {
        "base": (22, 24), "grpo": (24, 24), "final": (22, 24),
    }
    assert {p: m["same_call_rate"]["estimate"] for p, m in same_call.items()} == pytest.approx(
        {"base": 1 / 22, "grpo": 0.0, "final": 1.0}
    )
    assert same_call["base"]["incomplete_reasons"] == {"missing_rule_ids": 2}
    assert all(m["between_arm_contrast"] is None for m in same_call.values())
    # grpo's malformed first answer: one bundle completed at its second attempt, one component regenerated.
    statistics = arms["grpo"]["attempt_statistics"]
    assert statistics["attempts_histogram"] == {"1": 23, "2": 1}
    assert statistics["component_attempts_histogram"] == {"1": 71, "2": 1}
    assert statistics["invalid_components"] == {"invalid_json": 1}
    assert golden["strata"] == {"field": "source", "counts": {"stress": 6, "wildchat": 6}}


def test_two_holm_families_of_the_declared_size_never_pair_the_policies(golden):
    for policy, comparison in golden["comparisons"].items():
        for family in ("flat", "kappa"):
            row = comparison[family]
            assert (row["policy_a"], row["policy_b"]) == ("base", policy)
            assert row["family_size"] == len(POLICIES) - 1
    assert set(golden["comparisons"]) == {"grpo", "final"}
    grpo, final = golden["comparisons"]["grpo"], golden["comparisons"]["final"]
    assert grpo["flat"]["estimate"] == pytest.approx(22 / 24) and final["flat"]["estimate"] == pytest.approx(19 / 24)
    assert grpo["kappa"]["estimate"] == pytest.approx(1.0) and final["kappa"]["estimate"] == pytest.approx(0.0, abs=1e-12)
    for family in ("flat", "kappa"):
        rows = sorted((golden["comparisons"][p][family] for p in ("grpo", "final")), key=lambda r: r["p_value"])
        assert rows[0]["p_holm"] == pytest.approx(min(1.0, 2 * rows[0]["p_value"]))
        assert rows[1]["p_holm"] == pytest.approx(max(rows[0]["p_holm"], rows[1]["p_value"]))
    assert {grpo["flat"]["test_method"], grpo["kappa"]["test_method"]} == {
        "scenario_clustered_paired_randomization_v1", "scenario_cluster_bootstrap_percentile_test_v1",
    }
    assert grpo["kappa"]["test_method"] == "scenario_cluster_bootstrap_percentile_test_v1"
    assert "p_randomization" in grpo["kappa"] and "p_randomization" not in grpo["flat"]
    defined = 200 - grpo["kappa"]["n_undefined_replicates"]
    assert 0 < grpo["kappa"]["n_undefined_replicates"] < 50 and grpo["kappa"]["p_value"] == pytest.approx(2 / (defined + 1))
    # final's kappa does not move at all: 0 on every replicate, and every permutation is as extreme.
    assert final["kappa"]["p_value"] == 1.0 and final["kappa"]["p_randomization"] == 1.0


def test_the_claim_rule_harm_flag_and_decomposition(golden):
    grpo, final = golden["comparisons"]["grpo"], golden["comparisons"]["final"]
    assert (grpo["claim"], final["claim"]) == ("more_consistent", "yes_shift")
    assert grpo["flat"]["significant_holm"] and grpo["kappa"]["significant_holm"]
    assert final["flat"]["significant_holm"] and not final["kappa"]["significant_holm"]
    # final's M != yes rate rises from 0 to 1 with an interval at 1: flagged; grpo's does not move.
    assert final["harm_flag"] == {"stratum": "M!=yes", "threshold": 0.1, "flagged": True}
    assert final["stratified_differences"]["differences"]["M!=yes"]["estimate"] == pytest.approx(1.0)
    assert grpo["harm_flag"]["flagged"] is False
    for comparison in (grpo, final):
        parts = comparison["decomposition"]
        assert parts["verdict_effect"]["estimate"] + parts["reference_effect"]["estimate"] == pytest.approx(
            parts["total"]["estimate"]
        )
    # grpo's verdicts against the base's per-rule maxima: every S x S pair of scenario i.
    assert grpo["decomposition"]["cells"]["F_reference|M_reference"]["estimate"] == pytest.approx(22 / 24)


def test_the_disclosure_is_printed_beside_the_result(golden):
    record = golden["disclosure"]
    assert record["regime"] == "merged_fp32" and record["serving_dtype"] == "float32"
    assert record["engine"] == {"vllm_version": "0.28.0+cu129", "attention_backend": "TRITON_ATTN",
                                **RUNTIME["precision_env"]}
    assert record["parity_b"]["grpo"]["prefill"]["rho"] == 1.0001
    assert record["parity_b"]["final"]["prefill"]["rho_ci"] == [0.9981, 1.0004]
    assert {policy: list(arms) for policy, arms in record["parity_b"].items()} == {
        "grpo": ["prefill", "decode"], "final": ["prefill", "decode"],
    }
    assert record["parity_b"]["grpo"]["decode"]["segment"] == "completion"
    text = "\n".join(record["lines"])
    for phrase in ("Each GRPO policy arm (grpo, final) is its checkpoint merged into the Qwen3-8B weights in fp32 "
                   "and served in fp32", "TF32; TRITON_F32_DEFAULT unset", "VLLM_FLOAT32_MATMUL_PRECISION=highest",
                   "grpo: prefill rho = 1.0001 [0.9996, 1.0009], decode rho = 0.9998 [0.9990, 1.0007]",
                   "half-width is at most 0.01", "not the bf16 Qwen3-8B of the ruleset-size experiment", "bf16 colocated engine",
                   "measures the merged policy itself"):
        assert phrase in text


def _assert_same(expected, actual, path="golden"):
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        assert math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12), f"{path}: {actual} != {expected}"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and sorted(actual) == sorted(expected), f"{path}: keys differ"
        for key in expected:
            _assert_same(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: lengths differ"
        for index, (e, a) in enumerate(zip(expected, actual)):
            _assert_same(e, a, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def test_an_arm_without_same_call_statistics_has_none_in_the_golden():
    assert producer._same_call(None) is None


def test_the_producer_reproduces_the_committed_synthetic_readout(golden):
    payload = json.loads(json.dumps(golden, sort_keys=True))
    if os.environ.get("RULEFORGE_REGENERATE_TRAINING_FIXTURE") == "1":  # pragma: no cover - deliberate regeneration
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _assert_same(json.loads(FIXTURE.read_text(encoding="utf-8")), payload)


def test_the_panels_draw_the_golden(tmp_path):
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert list(panel_values(committed, "a")) == ["Qwen3-8B", "GRPO", "GRPO (final)"]
    # the flat rates alone: the floors (pinned above) stay in the golden, unread by panel a
    assert_panel_a_draws_the_flat_rates_alone(committed, ["0.917", "0.000", "0.125"])
    assert [committed["arms"][p]["kappa"]["estimate"] is not None for p in POLICIES] == [True] * len(POLICIES)
    assert committed["arms"]["final"]["m_stratified"]["M!=yes"]["n_draws"] == 3
    draw(committed, tmp_path / "a.pdf", "a")
    assert (tmp_path / "a.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="unknown training panel"):
        panel_values(committed, "b")
    four = json.loads(json.dumps(committed))
    four["policies"].append("extra")
    four["arms"]["extra"] = four["arms"]["final"]
    with pytest.raises(ValueError, match="at most 3"):
        panel_values(four, "a")
    # a golden whose two arms share a display label (edited by hand: the producers refuse one) is not drawn
    alike = json.loads(json.dumps(committed))
    alike["arms"]["final"]["label"] = alike["arms"]["grpo"]["label"]
    with pytest.raises(ValueError, match=r"display labels \['GRPO'\] name more than one arm"):
        panel_values(alike, "a")
    undefined = json.loads(json.dumps(committed))
    undefined["arms"]["final"]["flat"] = {"estimate": None, "ci_low": None, "ci_high": None}
    figure = build_figure(undefined, "a")
    assert "undefined" in [text.get_text() for text in figure.axes[0].texts]


def _estimates(root: Path) -> Path:
    return root / "aggregate" / "estimates.json"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.update(state="in_progress"), "is not complete"),
        (lambda e: e.update(inference_method="mcnemar"), "unsupported inference method"),
        (lambda e: e.update(temperature=0.7), "temperature is not the campaign manifest's"),
        (lambda e: e.update(n_scenarios=500), "n_scenarios is not the manifest's"),
        (lambda e: next(iter(e["policy_families"].values())).update(missing_policies=["x"]), "are missing"),
        (lambda e: next(iter(e["policy_families"].values())).update(excluded_policies={"final": "x"}), "were excluded"),
        (lambda e: next(iter(e["policy_families"].values())).update(orientation="x"), "flat family's orientation"),
        (lambda e: next(iter(e["policy_families"].values())).update(kappa_orientation="x"), "kappa family's orientation"),
        (lambda e: next(iter(e["policy_families"].values())).update(strata={"field": None}), "recorded strata"),
        (lambda e: e["conditions"]["qwen/figure5/grpo"].update(estimate=0.5), "flat rate is not its draws'"),
        (lambda e: next(iter(e["policy_families"].values()))["arms"]["grpo"]["kappa"].update(estimate=0.5),
         "kappa is not its draws'"),
        (lambda e: next(iter(e["policy_families"].values()))["arms"]["grpo"]["yes_rate"].update(estimate=0.5),
         "yes-rate is not"),
        (lambda e: next(iter(e["policy_families"].values()))["arms"]["grpo"]["always_yes_floor"].update(estimate=0.5),
         "always-yes floor is not"),
        (lambda e: next(iter(e["policy_families"].values()))["arms"]["grpo"].update(grid_digests={}),
         "judgment grids are not"),
        (lambda e: e["conditions"]["qwen/figure5/grpo"].update(condition_fingerprint="x"), "stale fingerprint"),
        (lambda e: next(iter(e["policy_families"].values()))["contrasts"].pop(), "not the reference-against-each"),
        (lambda e: next(iter(e["policy_families"].values()))["discrimination_contrasts"][0].update(family_size=3),
         "family_size is not 2"),
        (lambda e: next(iter(e["policy_families"].values()))["contrasts"][0].update(test_method="x"),
         "another test method"),
        (lambda e: next(iter(e["policy_families"].values()))["contrasts"][0].update(inference_method="x"),
         "another inference method"),
        (lambda e: next(iter(e["policy_families"].values()))["contrasts"][0].update(p_value=2.0),
         "is not a probability"),
        (lambda e: next(iter(e["policy_families"].values()))["discrimination_contrasts"][1].update(p_holm=0.0),
         "Holm p disagrees"),
        (lambda e: next(iter(e["policy_families"].values()))["discrimination_contrasts"][1].update(significant_holm=True),
         "kappa contrasts significance disagrees"),
        (lambda e: next(iter(e["policy_families"].values()))["discrimination_contrasts"][0].update(
            test_method="scenario_clustered_paired_randomization_v1"), "kappa contrasts carry another test method"),
        (lambda e: next(iter(e["policy_families"].values()))["contrasts"][0].update(estimate=0.1),
         r"not rate\(a\) - rate\(b\)"),
        (lambda e: next(iter(e["policy_families"].values()))["discrimination_contrasts"][0].update(estimate=0.1),
         r"not kappa\(b\) - kappa\(a\)"),
        (lambda e: e.update(policy_families={}), "0 policy families"),
        (lambda e: e["policy_families"].update(other=e["policy_families"][next(iter(e["policy_families"]))]),
         "2 policy families"),
    ],
)
def test_the_producer_refuses_estimates_that_are_not_the_campaigns(root, parity, change, message):
    _edit(_estimates(root), change)
    with pytest.raises(TrainingReadoutError, match=message):
        build_from_campaign(_estimates(root), parity=parity)


def test_the_producer_refuses_a_non_final_campaign_a_missing_parity_and_unknown_labels(root, parity, tmp_path):
    with pytest.raises(TrainingReadoutError, match=r"no parity report for \['final \(prefill\)', 'final \(decode\)'\]"):
        build_from_campaign(_estimates(root), parity={"grpo": parity["grpo"]})
    with pytest.raises(TrainingReadoutError, match=r"no parity report for \['final \(decode\)'\]"):
        build_from_campaign(_estimates(root), parity={**parity, "final": parity["final"][0]})
    with pytest.raises(TrainingReadoutError, match="which are not policies"):
        build_from_campaign(_estimates(root), parity={**parity, "base": parity["grpo"]})
    with pytest.raises(TrainingReadoutError, match="display labels"):
        build_from_campaign(_estimates(root), parity=parity, labels={"other": "x"})
    # two arms of one display label would be drawn as one bar (the panels key an arm by it)
    with pytest.raises(TrainingReadoutError, match=r"display labels \['GRPO'\] name more than one arm"):
        build_from_campaign(_estimates(root), parity=parity, labels={"grpo": "GRPO", "final": "GRPO"})
    _edit(root / "sampling_manifest.json", lambda m: m.update(campaign_mode="pilot"))
    with pytest.raises(TrainingReadoutError, match="campaign_mode 'pilot'"):
        build_from_campaign(_estimates(root), parity=parity)
    scratch = build_from_campaign(_estimates(root), parity={"grpo": parity["grpo"]}, allow_incomplete=True)
    assert scratch["provenance"]["complete"] is False and set(scratch["disclosure"]["parity_b"]) == {"grpo"}


def test_the_files_read_are_recorded_by_the_digest_of_the_bytes_read(root, parity, monkeypatch):
    path = _estimates(root)
    read = hashlib.sha256(path.read_bytes()).hexdigest()
    real = producer.family_golden

    def golden(*args, **kwargs):
        write_json_sidecar(path, {**json.loads(path.read_text()), "created_at": "2026-10-11T00:00:00Z"})
        return real(*args, **kwargs)

    monkeypatch.setattr(producer, "family_golden", golden)
    built = build_from_campaign(path, parity=parity)
    assert built["provenance"]["estimates"]["sha256"] == read != hashlib.sha256(path.read_bytes()).hexdigest()
    assert built["disclosure"]["parity_b"]["grpo"]["prefill"]["report_sha256"] == hashlib.sha256(
        parity["grpo"][0].read_bytes()).hexdigest()


def test_a_stack_mismatch_is_built_only_on_the_record(root, parity):
    mismatch = {"scope": "policy_contrasts", "conditions": ["qwen/figure5/base", "qwen/figure5/grpo"], "reason": "r"}
    _edit(_estimates(root), lambda e: e.update(allow_stack_mismatch=True, stack_mismatches=[mismatch]))
    with pytest.raises(TrainingReadoutError, match="inference-stack mismatch"):
        build_from_campaign(_estimates(root), parity=parity)
    built = build_from_campaign(_estimates(root), parity=parity, allow_stack_mismatch=True)
    assert built["provenance"]["stack_mismatches"] == [mismatch]


def test_the_arms_must_share_one_engine_and_a_serving_dtype(root, parity):
    set_recorded_runtime(root / "qwen" / "figure5" / "final", "draw-001",
                         precision_env={**RUNTIME["precision_env"], "TRITON_F32_DEFAULT": "ieee"})
    with pytest.raises(TrainingReadoutError, match="different engine settings.*TRITON_F32_DEFAULT"):
        build_from_campaign(_estimates(root), parity=parity)


def test_a_serving_dtype_outside_both_regimes_is_refused(root, parity, monkeypatch):
    monkeypatch.setattr(producer, "REGIMES", {"bfloat16": "bf16_fallback"})
    with pytest.raises(TrainingReadoutError, match="serving dtypes.*declares dtype bfloat16"):
        build_from_campaign(_estimates(root), parity=parity)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "lora"}, "not 'merged'"),
        ({"passed": False, "failures": ["serving gap"]}, "did not pass"),
        ({"buckets": {"all": {"rho": None}}}, "is not a number"),
        ({"segment": "prompt"}, "segment 'prompt' is neither parity arm"),
        ({"buckets": {"all": {"rho": 0.97, "rho_ci": [0.95, 0.99], "rho_halfwidth": 0.01}}},
         r"outside the fp32 gate \[0.99, inf\]"),
        (
            {"rho_gated": {"all": False}, "buckets": {"all": {"rho": 0.995, "rho_ci": [0.93, 1.06], "rho_halfwidth": 0.065}}},
            r"rho gate did not run \(rho_gated \{'all': False\}\): the interval's half-width 0.0650 is over 0.01",
        ),
        ({"rho_gated": {}}, "rho gate did not run"),
        ({"rho_gated": {"all": True, "b0,512": False}}, "rho gate did not run"),
        ({"rho_gated": {"all": False}, "inconclusive": ["all: effect too weak"]}, "rho gate did not run"),
        ({"thresholds": {"rho_range": None, "rho_precision": 0.01}}, r"rho range None, not fp32's gate"),
        ({"thresholds": {"rho_range": [0.9, "inf"], "rho_precision": 0.01}}, "rho range"),
        ({"thresholds": {"rho_range": ["-inf", "inf"], "rho_precision": 0.01}}, r"rho range \['-inf', 'inf'\]"),
        ({"thresholds": {"rho_range": [0.99, "infinity"], "rho_precision": 0.01}}, "rho range"),
        ({"thresholds": {"rho_range": [0.99, "inf"], "rho_precision": None}}, "rho precision None"),
        ({"thresholds": {"rho_range": [0.99, "inf"], "rho_precision": 0.05}}, "rho precision 0.05"),
    ],
)
def test_a_parity_report_must_be_a_passing_merged_one(tmp_path, changes, message):
    with pytest.raises(TrainingReadoutError, match=message):
        read_parity(write_parity(tmp_path, **changes), regime="merged_fp32", policy="grpo")


def test_a_report_as_compare_records_it_clears_the_gate(tmp_path):
    # `compare` writes the unbounded upper limit as the string "inf" (thresholds_to_json), not a JSON number.
    assert PARITY["thresholds"]["rho_range"] == [0.99, "inf"]
    record = read_parity(write_parity(tmp_path), regime="merged_fp32", policy="grpo")
    assert record["passed"] is True and record["rho"] == 1.0001


def test_a_policy_has_one_parity_report_per_arm(tmp_path):
    prefill, decode = write_parity(tmp_path), write_parity(tmp_path, arm="decode")
    arms = read_parity_arms([decode, prefill], regime="merged_fp32", policy="grpo")
    assert list(arms) == ["prefill", "decode"] and arms["decode"]["segment"] == "completion"
    with pytest.raises(TrainingReadoutError, match="two prefill parity reports for 'grpo'"):
        read_parity_arms([prefill, write_parity(tmp_path, "other")], regime="merged_fp32", policy="grpo")


def test_the_bf16_fallback_reports_rho_against_its_threshold(tmp_path):
    record = read_parity(write_parity(tmp_path, buckets={"all": {"rho": 0.85}}, rho_gated={}, thresholds={}),
                         regime="bf16_fallback", policy="grpo")
    assert record["rho"] == 0.85 and record["rho_ci"] is None and record["arm"] == "prefill"
    engine = {"vllm_version": None, "attention_backend": "FLASH_ATTN", "TRITON_F32_DEFAULT": "weird",
              "VLLM_FLOAT32_MATMUL_PRECISION": "highest"}
    common = dict(engine=engine, policies=["grpo"], reference="base", model_name="Qwen3-8B")
    lines = disclosure(regime="bf16_fallback", parity={"grpo": {"prefill": record}}, **common)
    text = "\n".join(lines)
    assert "Under the bf16 fallback, each GRPO policy arm" in text and "in prefill and decode arms): grpo: prefill rho = 0.8500" in text
    assert "For grpo, rho is below 0.9" in text and "served in bf16 on vLLM an unrecorded vLLM version" in text
    assert "carry the same rounding of the adapter as that engine" in text
    assert "measures the merged policy itself" not in text
    above = read_parity(write_parity(tmp_path, "high", buckets={"all": {"rho": 0.95}}), regime="bf16_fallback", policy="g")
    decode = read_parity(write_parity(tmp_path, "high", "decode", buckets={"all": {"rho": 0.89}}),
                         regime="bf16_fallback", policy="g")
    kept = "\n".join(disclosure(regime="bf16_fallback", parity={"g": {"prefill": above}}, **common))
    assert "rho is below" not in kept
    either = "\n".join(disclosure(regime="bf16_fallback", parity={"g": {"prefill": above, "decode": decode}}, **common))
    assert "For g, rho is below 0.9" in either
    fp32 = "\n".join(disclosure(regime="merged_fp32", parity={}, **common))
    assert "TRITON_F32_DEFAULT=weird" in fp32 and "no parity report (scratch build)" in fp32
    assert "measures the merged policy itself" in fp32 and "same rounding" not in fp32


@pytest.mark.parametrize(
    ("flat", "kappa", "expected"),
    [
        ((True, 0.2), (True, 0.1), "more_consistent"),
        ((True, 0.2), (False, 0.1), "yes_shift"),
        ((True, 0.2), (True, -0.1), "yes_shift"),
        ((True, 0.2), (True, None), "yes_shift"),
        ((False, 0.2), (True, 0.1), "null"),
        ((True, -0.2), (True, 0.1), "higher_inconsistency"),
    ],
)
def test_the_claim_rule(flat, kappa, expected):
    assert claim({"significant_holm": flat[0], "estimate": flat[1]},
                 {"significant_holm": kappa[0], "estimate": kappa[1]}) == expected


def test_a_root_whose_conditions_are_not_the_family_is_refused(root, parity):
    conditions = load_conditions(root)
    assert set(conditions) == {f"qwen/figure5/{p}" for p in POLICIES}
    _edit(_estimates(root), lambda e: e["conditions"].update(extra=e["conditions"]["qwen/figure5/base"]))
    with pytest.raises(TrainingReadoutError, match="conditions outside the family"):
        build_from_campaign(_estimates(root), parity=parity)


def test_an_undefined_kappa_is_reported_as_such_and_never_counts_for_the_claim(tmp_path, modelspec, parity):
    """An always-yes arm whose per-rule max is always yes has P_e = 1."""
    root = run_training_campaign(tmp_path, modelspec, degenerate_final=True)
    estimates = analyze_campaign(root, n_resamples=200, n_permutations=2000, rule_set_path=tmp_path / "rules.jsonl")
    mark_final(root)
    (family,) = estimates["policy_families"].values()
    row = family["discrimination_contrasts"][1]
    assert row["policy_b"] == "final" and row["estimate"] is None and row["kappa_b"]["estimate"] is None
    assert (row["p_value"], row["p_value_defined"], row["family_size"]) == (1.0, False, 2)
    built = build_from_campaign(_estimates(root), parity=parity)
    assert built["arms"]["final"]["kappa"]["estimate"] is None
    assert built["comparisons"]["final"]["claim"] == "yes_shift"


def _arm(dtype="bfloat16", adapter=None, weights=None, path="Qwen3-8B"):
    from types import SimpleNamespace

    return SimpleNamespace(serving_dtype=dtype, adapter_sha256=adapter, model_weights_sha256=weights, model_path=path)


def _lora_arms(**changes):
    arms = {"base": _arm(), "grpo": _arm(adapter="a" * 64), "final": _arm(adapter="b" * 64)}
    return {**arms, **changes}


def test_the_serving_regime_is_read_off_the_arms_identities():
    merged = {"base": _arm("float32"), "grpo": _arm("float32", weights="w" * 64, path="merged-1")}
    assert producer.serving_regime(merged, "base", "root") == "merged_fp32"
    fallback = {name: _arm("bfloat16", weights=arm.model_weights_sha256, path=arm.model_path)
                for name, arm in merged.items()}
    assert producer.serving_regime(fallback, "base", "root") == "bf16_fallback"
    assert producer.serving_regime(_lora_arms(), "base", "root") == producer.LORA_REGIME


@pytest.mark.parametrize(("arms", "message"), [
    ({"base": _arm("float32"), "grpo": _arm("bfloat16")}, r"serving dtypes \['bfloat16', 'float32'\]"),
    ({"base": _arm(None), "grpo": _arm(None)}, r"serving dtypes \['None'\].*declares dtype bfloat16"),
    (_lora_arms(base=_arm("float32"), grpo=_arm("float32", adapter="a" * 64), final=_arm("float32", adapter="b" * 64)),
     "served in float32; LoRA serving is bf16"),
    (_lora_arms(base=_arm(adapter="c" * 64)), "the reference 'base' serves an adapter"),
    (_lora_arms(final=_arm()), r"policies \['final'\] serve no adapter beside policies that do"),
    (_lora_arms(final=_arm(adapter="b" * 64, weights="w" * 64)), "every arm serves the base weights"),
    (_lora_arms(final=_arm(adapter="b" * 64, path="Qwen3-8B-other")), "every arm serves the base weights"),
    # a policy arm serving the base: no adapter and no weights digest of its own, in either dtype
    ({"base": _arm(), "grpo": _arm()}, r"policies \['grpo'\] serve the reference's weights with no adapter"),
    ({"base": _arm("float32"), "grpo": _arm("float32", weights="w" * 64, path="merged-1"), "final": _arm("float32")},
     r"policies \['final'\] serve the reference's weights"),
    ({"base": _arm(weights="w" * 64, path="m"), "grpo": _arm(weights="w" * 64, path="m")},
     r"policies \['grpo'\] serve the reference's weights"),
])
def test_a_root_that_mixes_serving_regimes_is_refused(arms, message):
    with pytest.raises(TrainingReadoutError, match=message):
        producer.serving_regime(arms, "base", "root")


#: What a report of ``lora_parity compare --preset b_lora --check b`` changes of :data:`PARITY`.
LORA_PARITY = {"mode": "lora", "thresholds": thresholds_to_json(preset_thresholds("b_lora")),
               "buckets": {"all": {"rho": 0.991, "rho_ci": [0.984, 0.998], "rho_halfwidth": 0.007}}}


def write_lora_parity(tmp_path: Path, name: str = "grpo", arm: str = "prefill", **changes) -> Path:
    return write_parity(tmp_path, name, arm, **{**LORA_PARITY, **changes})


def test_lora_servings_parity_is_a_lora_mode_report_gated_within_b_loras_window(tmp_path):
    assert producer.RHO_WINDOW_LORA == PARITY_PRESETS["b_lora"]["rho_range"]
    assert producer.RHO_PRECISION_LORA == PARITY_PRESETS["b_lora"]["rho_precision"]
    assert (producer.RHO_GATE_FP32, producer.RHO_PRECISION_FP32) == (PARITY_PRESETS["b"]["rho_range"][0],
                                                                     PARITY_PRESETS["b"]["rho_precision"])
    record = read_parity(write_lora_parity(tmp_path), regime="lora_bf16", policy="grpo")
    assert record["mode"] == "lora" and record["rho"] == 0.991 and record["arm"] == "prefill"
    with pytest.raises(TrainingReadoutError, match="mode 'lora', not 'merged'"):
        read_parity(write_lora_parity(tmp_path), regime="merged_fp32", policy="grpo")


@pytest.mark.parametrize(("changes", "message"), [
    ({"mode": "merged"}, "mode 'merged', not 'lora'"),
    ({"buckets": {"all": {"rho": 1.07, "rho_ci": [1.06, 1.08]}}}, r"outside the b_lora gate \[0.95, 1.05\]"),
    ({"buckets": {"all": {"rho": 0.93, "rho_ci": [0.92, 0.94]}}}, r"outside the b_lora gate"),
    ({"thresholds": thresholds_to_json(preset_thresholds("b"))}, "not b_lora's gate"),
    ({"thresholds": {"rho_range": [0.9, 1.1], "rho_precision": 0.01}}, "not b_lora's gate"),
    ({"thresholds": {"rho_range": [0.95], "rho_precision": 0.01}}, "not b_lora's gate"),
    ({"thresholds": {"rho_range": [0.95, 1.05], "rho_precision": 0.02}}, "rho precision 0.02 is not b_lora's 0.01"),
    ({"rho_gated": {"all": False}}, "rho gate did not run"),
])
def test_a_lora_parity_report_outside_b_lora_is_refused(tmp_path, changes, message):
    with pytest.raises(TrainingReadoutError, match=message):
        read_parity(write_lora_parity(tmp_path, **changes), regime="lora_bf16", policy="grpo")


def test_the_lora_disclosure_names_the_unmerged_adapter_and_its_engine(tmp_path):
    record = read_parity(write_lora_parity(tmp_path), regime="lora_bf16", policy="grpo")
    engine = {"vllm_version": "0.28.0", "attention_backend": "FLASH_ATTN", "TRITON_F32_DEFAULT": "unset",
              "VLLM_FLOAT32_MATMUL_PRECISION": "highest"}
    text = "\n".join(disclosure(regime="lora_bf16", engine=engine, parity={"grpo": {"prefill": record}},
                                policies=["grpo"], reference="base", model_name="Qwen3-8B"))
    assert "LoRA adapter, unmerged, served through vLLM's LoRA path in bf16" in text
    assert "the base arm serves none, and its engine is built without LoRA" in text
    assert "gated within [0.95, 1.05]" in text and "grpo: prefill rho = 0.9910 [0.9840, 0.9980]" in text
    assert "served in bf16 on vLLM 0.28.0" in text and "merged in" not in text and "checkpoint merged" not in text


def test_a_lora_served_root_reads_lora_parity_and_discloses_lora_serving(root, tmp_path, monkeypatch):
    seen = {}

    def regime(specs, reference, where):
        seen.update(policies=sorted(specs), reference=reference)
        return producer.LORA_REGIME

    monkeypatch.setattr(producer, "serving_regime", regime)
    reports = {policy: [write_lora_parity(tmp_path, policy), write_lora_parity(tmp_path, policy, "decode")]
               for policy in ("grpo", "final")}
    built = build_from_campaign(_estimates(root), parity=reports)
    assert seen == {"policies": sorted(POLICIES), "reference": "base"}
    assert built["disclosure"]["regime"] == "lora_bf16" and built["provenance"]["complete"] is True
    assert built["disclosure"]["parity_b"]["grpo"]["decode"]["mode"] == "lora"
    assert any("LoRA adapter, unmerged" in line for line in built["disclosure"]["lines"])
    assert all("adapter_sha256" not in arm for arm in built["arms"].values())
    with pytest.raises(TrainingReadoutError, match="mode 'merged', not 'lora'"):
        build_from_campaign(_estimates(root), parity={"grpo": write_parity(tmp_path)}, allow_incomplete=True)


GEMMA_PAIR = ("base", "grpo")


@pytest.fixture
def gemma_root(tmp_path, modelspec):
    """A Gemma-shaped training root as configs/launch/training_arm_gemma.yaml lays one out."""
    root = run_training_campaign(tmp_path, modelspec, GEMMA_PAIR, judge=GEMMA)
    analyze_campaign(root, n_resamples=200, n_permutations=2000, rule_set_path=tmp_path / "rules.jsonl")
    mark_final(root)
    return root


def test_a_gemma_pair_reads_as_one_family_and_builds_a_complete_golden_without_parity_b(gemma_root):
    labels = {"base": "Gemma 4 31B-it", "grpo": "GRPO (checkpoint-200)"}
    golden = build_from_campaign(_estimates(gemma_root), parity={}, labels=labels)
    assert golden["policies"] == list(GEMMA_PAIR) and golden["reference_policy"] == "base"
    assert golden["provenance"]["complete"] is True and golden["provenance"]["strict_think"] is False
    assert golden["provenance"]["judge"]["model_name"] == "gemma-4-31B-it"
    assert {policy: arm["condition_id"] for policy, arm in golden["arms"].items()} == {
        "base": "gemma-4-31B-it/figure5/base", "grpo": "gemma-4-31B-it/figure5/grpo"}
    assert (golden["arms"]["base"]["model_path"], golden["arms"]["base"]["model_weights_sha256"]) == (
        "gemma-4-31B-it", None)
    assert golden["arms"]["grpo"]["model_path"] == "grpo" and golden["arms"]["grpo"]["model_weights_sha256"]
    qwen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for policy in GEMMA_PAIR:
        for key in ("flat", "kappa", "yes_rate", "always_yes_floor"):
            assert golden["arms"][policy][key]["estimate"] == pytest.approx(qwen["arms"][policy][key]["estimate"])
        assert golden["arms"][policy]["same_call"]["same_call_rate"]["estimate"] == pytest.approx(
            qwen["arms"][policy]["same_call"]["same_call_rate"]["estimate"])
    comparison = golden["comparisons"]["grpo"]
    assert (comparison["flat"]["family_size"], comparison["kappa"]["family_size"]) == (1, 1)
    assert comparison["flat"]["estimate"] == pytest.approx(22 / 24) and comparison["kappa"]["estimate"] == pytest.approx(1.0)
    assert comparison["claim"] == "more_consistent" and comparison["harm_flag"]["flagged"] is False
    assert set(comparison) >= {"decomposition", "stratified_differences"}
    record = golden["disclosure"]
    assert (record["regime"], record["serving_dtype"], record["parity_b"]) == ("merged_fp32", "float32", {})
    assert record["parity_b_not_built"] == producer.parity_not_built("gemma-4-31B-it")
    text = "\n".join(record["lines"])
    for phrase in ("merged into the gemma-4-31B-it weights in fp32", "Parity B was not built for gemma-4-31B-it",
                   "no rho is reported", "not the bf16 gemma-4-31B-it of the ruleset-size experiment"):
        assert phrase in text
    assert "Qwen3-8B" not in text and "rho =" not in text and "no parity report (scratch build)" not in text


def test_a_parity_report_for_a_judge_parity_b_was_never_built_for_is_refused(gemma_root, tmp_path):
    with pytest.raises(TrainingReadoutError, match=r"parity B was never built for gemma-4-31B-it: pass none"):
        build_from_campaign(_estimates(gemma_root), parity={"grpo": write_parity(tmp_path)})
    with pytest.raises(TrainingReadoutError, match="parity B was never built"):
        build_from_campaign(_estimates(gemma_root), parity={"grpo": write_parity(tmp_path)}, allow_incomplete=True)
    _edit(gemma_root / "sampling_manifest.json", lambda m: m.update(strict_think=True))
    with pytest.raises(TrainingReadoutError, match="gemma-4-31B-it's campaign validated with strict_think on, but its thinking"):
        build_from_campaign(_estimates(gemma_root), parity={}, allow_incomplete=True)
    _edit(gemma_root / "sampling_manifest.json", lambda m: m.pop("strict_think"))
    _edit(gemma_root / "sampling_manifest.json", lambda m: m.update(campaign_mode="pilot"))
    with pytest.raises(TrainingReadoutError, match="campaign_mode 'pilot'"):
        build_from_campaign(_estimates(gemma_root), parity={})


def test_parity_b_exists_for_every_judge_but_those_whose_thinking_closes_otherwise_than_with_a_think_tag(root, parity):
    for model_name, strict in (("Qwen3-8B", True), ("gemma-4-31B-it", False)):
        assert producer.parity_b_built(model_name, strict, "root") is strict
    _edit(root / "sampling_manifest.json", lambda m: m.pop("strict_think"))
    for reports, flags in ((parity, {}), ({}, {}), ({}, {"allow_incomplete": True})):
        with pytest.raises(TrainingReadoutError, match="Qwen3-8B's campaign validated with strict_think off, which only"):
            build_from_campaign(_estimates(root), parity=reports, **flags)


def test_parity_b_s_sentence_alone_gives_way_to_the_reason_in_every_regime():
    engine = {"vllm_version": "0.28.0", "attention_backend": "TRITON_ATTN", "TRITON_F32_DEFAULT": "unset",
              "VLLM_FLOAT32_MATMUL_PRECISION": "highest"}
    for regime in ("merged_fp32", "bf16_fallback", "lora_bf16"):
        common = dict(regime=regime, engine=engine, parity={}, policies=["grpo"], reference="base",
                      model_name="gemma-4-31B-it")
        built, not_built = disclosure(**common), disclosure(**common, parity_built=False)
        changed = [(a, b) for a, b in zip(built, not_built, strict=True) if a != b]
        assert len(changed) == 1 and changed[0][1] == producer.parity_not_built("gemma-4-31B-it"), regime
        assert "Parity B" in changed[0][0] and "no parity report (scratch build)" in changed[0][0]
