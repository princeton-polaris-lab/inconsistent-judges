from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ruleforge.experiments import table
from ruleforge.experiments.reproduce.rerun import (
    Finding,
    Interval,
    RerunError,
    claim_findings,
    differences,
    extra_files,
    intervals,
    read_golden,
    select,
    verify_golden,
    verify_rerun,
)
from ruleforge.experiments.table import Experiment, Golden, Step

REPO_ROOT = Path(__file__).resolve().parents[1]


#: The verifier (``ruleforge reproduce EXPERIMENT verify`` runs it).
verify_script = importlib.import_module("ruleforge.experiments.reproduce.verify_rerun")
def _entry(name="g", path="results/g.json", rerun_tolerance=0.005):
    return Golden(name, path, rerun_tolerance)


def _point(mean, low, high):
    return {"mean": mean, "ci_low": low, "ci_high": high}


def _fatal(findings):
    return [f for f in findings if f.fatal]


def _checks(findings):
    return sorted({f["check"] for f in findings})


def test_a_mean_with_ci_low_and_ci_high_is_one_interval():
    """`results/ruleset_size/curves.json` spells it this way, 348 times."""
    found = intervals(_point(0.87, 0.83, 0.91))
    assert len(found) == 1
    assert (found[0].estimate_key, found[0].estimate) == ("mean", 0.87)
    assert (found[0].low, found[0].high) == (0.83, 0.91)


def test_a_suffixed_ci_list_binds_to_its_own_stem():
    node = {"inconsistency_graded": 0.86, "kappa_lin": 0.58, "inconsistency_graded_ci": [0.82, 0.90]}
    found = intervals(node)
    assert [(i.estimate_key, i.low, i.high) for i in found] == [
        ("inconsistency_graded", 0.82, 0.90)]


def test_a_ci_list_falls_back_to_the_mean_spelling_of_its_stem():
    node = {"base_mean": 0.79, "base_ci": [0.75, 0.83]}
    assert intervals(node)[0].estimate_key == "base_mean"


def test_a_bare_ci_binds_to_the_change_it_brackets():
    node = {"change": 0.00037, "pct_change": -13.5, "ci": [0.00016, 0.00059]}
    found = intervals(node)
    assert (found[0].estimate_key, found[0].estimate) == ("change", 0.00037)


def test_an_analytic_bound_pair_drops_qualifiers_to_find_its_estimate():
    node = {"kappa_lin": 0.58,
            "kappa_lin_analytic_lo": 0.56, "kappa_lin_analytic_hi": 0.60}
    found = intervals(node)
    assert [(i.estimate_key, i.low, i.high) for i in found] == [
        ("kappa_lin", 0.56, 0.60)]


def test_a_lone_bound_is_not_an_interval():
    assert intervals({"agreement_ci_low": 0.887, "macro_agreement": 0.893}) == []


def test_a_non_numeric_bound_pair_is_not_an_interval():
    assert intervals({"x_lo": "n/a", "x_hi": 1.0, "x": 0.5}) == []


def test_a_ci_key_that_is_not_a_pair_of_numbers_is_ignored():
    assert intervals({"x_ci": [0.1, 0.2, 0.3], "x": 0.15}) == []
    assert intervals({"x_ci": "unavailable", "x": 0.15}) == []
    assert intervals({"x_ci": [0.1, "high"], "x": 0.15}) == []


def test_a_bool_is_not_an_estimate():
    node = {"ci": [0.0, 1.0], "mean": True, "value": 0.5}
    assert intervals(node)[0].estimate_key == "value"


def test_an_interval_with_no_estimate_anywhere_still_reports_its_bounds():
    found = intervals({"x_ci": [0.1, 0.2]})
    assert found[0].estimate is None and found[0].name == "x_ci"


def test_a_bound_pair_with_no_estimate_anywhere_still_reports_its_bounds():
    found = intervals({"x_lo": 0.1, "x_hi": 0.2})
    assert (found[0].estimate, found[0].name, found[0].keys) == (
        None, "x_lo", ("x_lo", "x_hi"))


def test_interval_predicates():
    interval = Interval("ci", "mean", 0.5, 0.4, 0.6, ("ci", "mean"))
    assert interval.contains(0.4) and interval.contains(0.6)
    assert not interval.contains(0.61)
    assert interval.excludes_zero()
    assert Interval("ci", None, None, -0.6, -0.4, ()).excludes_zero()
    assert not Interval("ci", None, None, -0.1, 0.6, ()).excludes_zero()
    assert interval.overlaps(Interval("ci", None, None, 0.6, 0.9, ()))
    assert not interval.overlaps(Interval("ci", None, None, 0.61, 0.9, ()))


def test_an_estimate_outside_the_committed_interval_is_reported_and_counted():
    findings, counts = claim_findings("g", _point(0.87, 0.83, 0.91),
                                      _point(0.79, 0.75, 0.83), 0.005)
    assert (counts["containment"], counts["missed"]) == (1, 1) and _fatal(findings) == []
    assert [f.check for f in findings] == ["containment"]
    assert "outside the committed interval" in findings[0].detail


def test_an_estimate_inside_the_committed_interval_passes():
    findings, counts = claim_findings("g", _point(0.87, 0.83, 0.91),
                                      _point(0.85, 0.81, 0.89), 0.005)
    assert findings == [] and counts["containment"] == counts["overlap"] == 1


def test_disjoint_intervals_are_reported_even_when_no_estimate_is_recorded():
    findings, _ = claim_findings("g", {"x_ci": [0.1, 0.2]},
                                 {"x_ci": [0.3, 0.4]}, 0.005)
    assert [(f.check, f.fatal) for f in findings] == [("overlap", False)]


def test_a_missing_rerun_estimate_is_reported_beside_the_overlap_check():
    findings, counts = claim_findings("g", {"x": 0.15, "x_ci": [0.1, 0.2]},
                                      {"x_ci": [0.9, 1.0]}, 0.005)
    assert counts["containment"] == 0 and counts["overlap"] == 1
    assert sorted((f.check, f.fatal) for f in findings) == [("missing", True), ("overlap", False)]


def test_a_sign_flip_is_reported_when_the_committed_interval_excludes_zero():
    findings, counts = claim_findings(
        "g", {"beta": -0.05, "beta_ci": [-0.09, -0.01]},
        {"beta": 0.05, "beta_ci": [0.01, 0.09]}, 0.005)
    assert counts["sign"] == 1
    assert _checks([f.as_dict() for f in findings]) == [
        "containment", "overlap", "sign"]
    assert _fatal(findings) == []


def test_a_boolean_claim_flip_is_reported():
    findings, counts = claim_findings(
        "g", {"p": 0.003, "significant_holm": True},
        {"p": 0.62, "significant_holm": False}, 0.005)
    assert counts["flag"] == 1
    flips = [f for f in findings if f.check == "flag"]
    assert len(flips) == 1 and not flips[0].fatal
    assert "boolean claim flipped" in flips[0].detail


def test_an_unchanged_boolean_is_counted_but_not_reported():
    findings, counts = claim_findings(
        "g", {"significant_holm": True}, {"significant_holm": True}, 0.005)
    assert counts["flag"] == 1 and findings == []


def test_a_bool_only_mapping_is_claim_bearing():
    findings, _ = claim_findings(
        "g", {"flags": {"significant": True}}, {}, 0.005)
    assert any(f.check == "missing" and f.fatal for f in findings)


def test_a_sign_flip_is_not_claimed_when_the_interval_straddles_zero():
    """A golden commits to a direction only where the CI excludes zero; doing otherwise promotes noise to a claim."""
    findings, counts = claim_findings(
        "g", {"beta": -0.01, "beta_ci": [-0.09, 0.07]},
        {"beta": 0.01, "beta_ci": [-0.07, 0.09]}, 0.005)
    assert counts["sign"] == 0 and findings == []


def test_a_rerun_estimate_of_exactly_zero_loses_a_claimed_direction():
    findings, _ = claim_findings("g", {"b": -0.05, "b_ci": [-0.09, -0.01]},
                                 {"b": 0.0, "b_ci": [-0.09, -0.01]}, 0.005)
    assert "sign" in [f.check for f in findings]


def test_containment_asks_whether_the_committed_interval_holds_the_rerun():
    findings, counts = claim_findings("g", _point(0.85, 0.80, 0.90),
                                      _point(0.84, 0.838, 0.842), 0.005)
    assert findings == [] and counts["containment"] == 1

    findings, _ = claim_findings("g", _point(0.85, 0.849, 0.851),
                                 _point(0.86, 0.80, 0.90), 0.005)
    assert [f.check for f in findings] == ["containment"]


def test_the_sign_gate_reads_the_committed_interval_not_the_reruns():
    findings, counts = claim_findings(
        "g", {"beta": -0.05, "beta_ci": [-0.09, -0.01]},
        {"beta": 0.01, "beta_ci": [-0.07, 0.09]}, 0.005)
    assert counts["sign"] == 1
    assert "sign" in [f.check for f in findings]

    findings, counts = claim_findings(
        "g", {"beta": -0.01, "beta_ci": [-0.09, 0.07]},
        {"beta": 0.05, "beta_ci": [0.01, 0.09]}, 0.005)
    assert counts["sign"] == 0
    assert "sign" not in [f.check for f in findings]


def test_an_interval_the_rerun_dropped_cannot_be_checked_and_is_fatal():
    findings, _ = claim_findings("g", {"x": 0.15, "x_ci": [0.1, 0.2]},
                                 {"x": 0.15, "x_ci": None}, 0.005)
    assert [(f.check, f.fatal) for f in findings] == [("missing", True)]


def test_a_bare_number_flips_sign_fatally_when_both_sides_clear_the_tolerance():
    findings, counts = claim_findings("g", {"difference": -0.0028},
                                      {"difference": 0.0028}, 0.0005)
    assert counts["bare_sign"] == 1
    assert [(f.check, f.fatal) for f in findings] == [("sign", False)]


def test_a_bare_number_wobbling_across_zero_below_the_tolerance_is_not_a_flip():
    """The prompt table's null rows sit at |improvement| ~0.006 with p ~0.88."""
    findings, _ = claim_findings("g", {"improvement": -0.0062},
                                 {"improvement": 0.0031}, 0.005)
    assert findings == []


def test_a_bare_number_that_keeps_its_sign_is_not_a_finding():
    findings, _ = claim_findings("g", {"difference": -0.0028},
                                 {"difference": -0.0031}, 0.0005)
    assert findings == []
    findings, _ = claim_findings("g", {"beta_K": -0.0028}, {"beta_K": -0.0031}, 0.0005)
    assert [(f.check, f.fatal) for f in findings] == [("design", True)]


def test_bounds_are_not_re_checked_as_bare_numbers():
    findings, counts = claim_findings(
        "g", {"mean": 0.5, "ci_low": -0.02, "ci_high": 0.9},
        {"mean": 0.5, "ci_low": 0.02, "ci_high": 0.9}, 0.001)
    assert counts["bare_sign"] == 0 and findings == []


def test_where_a_run_happened_is_not_compared_and_what_it_compared_is():
    findings, counts = claim_findings("g", {"created_at": "first run", "n_resamples": 25, "rate": 0.5},
                                      {"created_at": "second run", "n_resamples": 25, "rate": 0.5}, 0.0)
    assert findings == [] and counts["bare_sign"] == 1 and counts["design"] == 1
    findings, _ = claim_findings("g", {"note": "first run"}, {"note": "second run"}, 0.0)
    assert [(f.check, f.path) for f in _fatal(findings)] == [("design", "note")]


@pytest.mark.parametrize(("path", "value"), [
    ((0, "label"), "Another judge"),
    ((0, "n_scenarios"), 1),
    ((0, "n_decoder_draws"), 4),
    ((0, "points", 0, "size"), 999),
    ((0, "points", 0, "n_scenarios"), 499),
])
def test_a_rerun_that_measured_another_judge_size_or_sample_is_fatal(path, value):
    committed, entry = _published("ruleset_size_curves")
    rerun = json.loads(json.dumps(committed))
    node = rerun
    for step in path[:-1]:
        node = node[step]
    before, node[path[-1]] = node[path[-1]], value
    where = "".join(f"[{step}]" if isinstance(step, int) else f".{step}" for step in path).lstrip(".")
    findings = _fatal(_findings_of(committed, rerun, entry))
    assert [(f.check, f.path) for f in findings] == [("design", where)]
    assert f"committed {before!r}, rerun {value!r}" in findings[0].detail


@pytest.mark.parametrize(("name", "path", "value"), [
    ("prompt_ablation", ("table", 0, "baseline"), "forced"),
    ("prompt_ablation", ("table", 0, "treatment"), "naive"),
    ("prompt_ablation", ("table", 0, "baseline_condition"), "OtherModel/forced"),
    ("prompt_ablation", ("table", 0, "baseline_label"), "forced"),
    ("prompt_ablation", ("table", 0, "treatment_label"), None),
    ("ruleset_size_curves", (0, "subsets", "n_conditions"), 1),
    ("hierarchy", ("judges", 0, "judge"), "Another judge"),
    ("leniency", ("rules", 0, "id"), "Rule 99"),
    ("ruleforge_final_eval", ("labels", 0), "Another ruleset"),
])
def test_a_rerun_of_another_comparison_is_fatal(name, path, value):
    committed, entry = _published(name)
    rerun = json.loads(json.dumps(committed))
    node = rerun
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    assert [f.check for f in _fatal(_findings_of(committed, rerun, entry))] == ["design"]
    assert verify_golden(entry, committed, rerun)["findings"]


@pytest.mark.parametrize(("name", "path", "value"), [
    ("leniency", ("models", "Olmo3-32B", "effects", 0, "rule"), 57),
    ("leniency", ("settings", "alpha"), 0.1),
    ("ruleset_size_curves", (0, "ci_level"), 0.9),
    ("ruleset_size_curves", (0, "full_set_key"), "Rule 1"),
    ("training_gemma", ("arms", "base", "reference"), False),
])
def test_a_field_no_rule_names_is_compared_strictly(name, path, value):
    committed, entry = _published(name)
    rerun = json.loads(json.dumps(committed))
    node = rerun
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    assert [f.check for f in _fatal(_findings_of(committed, rerun, entry))] == ["design"]


@pytest.mark.parametrize(("name", "path", "value"), [
    ("leniency", ("provenance", "rules_file", "n_rules"), 58),
    ("prompt_ablation", ("provenance", "arms", "Olmo3-32B", "encouraged", "prompt_style"), "naive"),
    ("hierarchy", ("provenance", "bootstrap", "seed"), 17),
    ("hierarchy", ("provenance", "n_scenarios"), 499),
    ("ruleforge_final_eval", ("provenance", "judge", "model_name"), "another-judge"),
    ("prompt_ablation", ("provenance", "multiplicity"), "fdr_bh in p_holm"),
    ("training_gemma", ("disclosure", "lines", 8), "Bonferroni runs within each family"),
    ("leniency", ("provenance", "bootstrap", "seed"), 57),
    ("leniency", ("provenance", "bootstrap", "n_resamples"), 1),
    ("prompt_ablation", ("provenance", "alpha"), 0.9),
    ("ruleset_size_curves", (0, "campaign", "model_name"), "other-judge"),
    ("hierarchy", ("judges", 0, "secondary", "level_shift", "root", "n_rules"), 21),
    # A retry cap and a condition a cap amendment names are design, not where the run happened.
    ("ruleset_size_curves", (5, "campaign", "amendments", 0, "new_cap"), 1),
    ("ruleset_size_curves", (5, "campaign", "amendments", 0, "old_cap"), 1),
    ("ruleset_size_curves", (5, "campaign", "amendments", 0, "conditions", 0), "another-judge/bank"),
    ("ruleset_size_curves", (5, "campaign", "amendments", 0, "bundles", 0, "condition_id"), "another-judge/bank"),
    ("hierarchy", ("levels", "rules", "root", 0), "Rule 1"),
    ("hierarchy", ("judges", 0, "flat", "stack", "model_type"), "remote"),
    ("prompt_ablation", ("table", 0, "paired_within_root"), True),
    ("training_gemma", ("provenance", "arms", "final", "model_path"), "checkpoint-500"),
    ("training_gemma", ("provenance", "campaigns", 0, "rebase", "added_ids", 0), "0" * 64),
])
def test_a_design_value_inside_any_subtree_is_strict(name, path, value):
    committed, entry = _published(name)
    rerun = json.loads(json.dumps(committed))
    node = rerun
    for step in path[:-1]:
        node = node[step]
    node[path[-1]] = value
    assert [f.check for f in _fatal(_findings_of(committed, rerun, entry))] == ["design"]


def test_a_field_no_rule_names_stays_strict_and_a_level_named_root_is_no_path():
    from ruleforge.experiments.reproduce.rerun import SOURCE, STRICT, role

    assert role("new_threshold", 0.5) == role("new_label", "x") == role("new_count", 3) == STRICT
    assert role("new_threshold", 0.5, "provenance.stack.new_threshold") == STRICT
    assert role("new_cap", 30, "[5].campaign.amendments[0].new_cap") == STRICT
    assert role("paired_within_root", False, "table[0].paired_within_root") == STRICT
    assert role("root", "Rule 20", "levels.rules.root[0]") == role("root", "artifact://x", "root") == STRICT
    assert role("root", "artifact://x", "provenance.campaigns[0].root") == SOURCE
    assert role("short", "vLLM 0.28", "judges[0].flat.stack.short") == STRICT
    committed, entry = _published("hierarchy")
    rerun = json.loads(json.dumps(committed))
    del rerun["judges"][0]["secondary"]["level_shift"]["root"]
    assert [(f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))] == [
        ("missing", "judges[0].secondary.level_shift.root")]


def test_the_roots_a_joined_curve_came_from_are_where_it_ran_and_their_retry_policy_is_design():
    from ruleforge.experiments.reproduce.rerun import SOURCE, STRICT, role

    size = {"campaign_id": "k57", "root": "artifact://sampling/k57", "retry_scope": "component",
            "max_semantic_attempts": 30}
    assert role("root", "x", "[0].campaign.root") == role("root", "x", "judges[0].campaign.root") == SOURCE
    assert role("root", "x", "[0].campaign.sizes_from.57.root") == SOURCE
    assert role("campaign_id", "k57", "[0].campaign.sizes_from.57.campaign_id") == SOURCE
    assert role("retry_scope", "component", "[0].campaign.sizes_from.57.retry_scope") == STRICT
    assert role("max_semantic_attempts", 30, "[0].campaign.sizes_from.57.max_semantic_attempts") == STRICT
    for name in ("ruleset_size_curves", "ruleset_size_repeated_draw_baseline"):
        committed, entry = _published(name)
        curves = committed if isinstance(committed, list) else committed["judges"]
        where = "[0].campaign" if isinstance(committed, list) else "judges[0].campaign"
        for curve in curves:
            curve["campaign"].update(root="artifact://sampling/bank", sizes_from={"57": dict(size)})
        moved = json.loads(json.dumps(committed))
        for curve in moved if isinstance(moved, list) else moved["judges"]:
            curve["campaign"].update(root="artifact://sampling/mine", sizes_from={"57": {**size, "root": "x",
                                                                                         "campaign_id": "mine"}})
        assert _fatal(_findings_of(committed, moved, entry)) == []
        # A rerun in one root does not say how its size 57 was retried, and one under another cap measured otherwise.
        alone = json.loads(json.dumps(committed))
        for curve in alone if isinstance(alone, list) else alone["judges"]:
            del curve["campaign"]["root"], curve["campaign"]["sizes_from"]
        assert (f"{where}.sizes_from", "missing") in {(f.path, f.check) for f in _fatal(_findings_of(committed, alone,
                                                                                                         entry))}
        capped = json.loads(json.dumps(committed))
        (capped if isinstance(capped, list) else capped["judges"])[0]["campaign"]["sizes_from"]["57"].update(
            max_semantic_attempts=20)
        assert [(f.check, f.path) for f in _fatal(_findings_of(committed, capped, entry))] == [
            ("design", f"{where}.sizes_from.57.max_semantic_attempts")]


def test_where_and_when_a_run_happened_is_one_notice_never_fatal():
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    rerun["table"][0].update(baseline_root="artifact://sampling/mine", grid_digest_a="0" * 64)
    report = verify_golden(entry, committed, rerun)
    assert [(f["check"], f["fatal"]) for f in report["findings"] if f["check"] != "drift"] == [("environment", False)]


def test_a_flipped_significance_flag_and_the_claim_derived_from_it_are_reported_not_fatal():
    from ruleforge.experiments.training.campaign_readout import claim

    committed, entry = _published("training_gemma")
    rerun = json.loads(json.dumps(committed))
    row = rerun["comparisons"]["final"]
    row["kappa"]["significant_holm"] = not row["kappa"]["significant_holm"]
    row["claim"] = claim(row["flat"], row["kappa"])
    findings = _findings_of(committed, rerun, entry)
    assert _fatal(findings) == [] and {f.check for f in findings} >= {"flag", "derived"}
    for value in (None, "different_judge_different_prompt"):
        fabricated = json.loads(json.dumps(rerun))
        fabricated["comparisons"]["final"]["claim"] = value
        assert [(f.check, f.path) for f in _fatal(_findings_of(committed, fabricated, entry))] == [
            ("design", "comparisons.final.claim")]
    del rerun["comparisons"]["final"]["kappa"]
    assert ("design", "comparisons.final.claim") in {(f.check, f.path) for f in _fatal(_findings_of(committed, rerun,
                                                                                                    entry))}
    assert claim_findings("g", {"p": [{"created_at": "x"}]}, {"p": None}, 0.005)[0] == []


#: Each golden's leaves by how they are compared. A new field is strict until a rule of the verifier says otherwise, and
#: this table moves with it.
ROLES = {
    "ruleset_size_curves": {"strict": 4201, "result": 2957, "environment": 87},
    "ruleset_size_repeated_draw_baseline": {"strict": 1429, "result": 14160, "environment": 48},
    "prompt_ablation": {"strict": 852, "result": 831, "environment": 239},
    "leniency": {"strict": 2695, "result": 11062, "environment": 712},
    "ruleforge_final_eval": {"strict": 85, "result": 108, "environment": 14},
    "training_gemma": {"strict": 998, "result": 171, "environment": 105},
    "training_qwen": {"strict": 1005, "result": 190, "environment": 109},
    "grpo_judgment_changes": {"strict": 281, "result": 399, "environment": 34},
    "grpo_frozen_anchors": {"strict": 234, "result": 460, "environment": 36},
    "hierarchy": {"strict": 407, "result": 546, "environment": 120},
    "hierarchy_ladder": {"strict": 512, "result": 528, "environment": 76},
}


def _roles(node, key="", path="", found=None):
    from collections import Counter

    from ruleforge.experiments.reproduce.rerun import role

    found = Counter() if found is None else found
    if isinstance(node, dict):
        for name, value in node.items():
            _roles(value, name, f"{path}.{name}" if path else name, found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _roles(value, key, f"{path}[{index}]", found)
    else:
        found[role(key, node, path)] += 1
    return found


def test_every_leaf_of_every_golden_is_strict_unless_a_rule_names_its_group():
    assert {entry.name: dict(_roles(_published(entry.name)[0])) for entry in table.goldens()} == ROLES


_V028 = "vllm 0.28.0+cu129 / torch 2.13.0+cu129 / transformers 5.17.0 / tokenizers 0.23.2 / triton 3.7.1"
_M028 = "vllm <version> / torch <version> / transformers <version> / tokenizers <version> / triton <version>"
_SETTINGS = "VLLM_FLOAT32_MATMUL_PRECISION=highest / TRITON_F32_DEFAULT=unset / VLLM_BATCH_INVARIANT=0"
_FP32 = '"float32"'


def _engine(dtype: str, backend: str) -> str:
    return (f"{_SETTINGS} / engine_args.dtype={dtype} / engine_args.attention_config={{\"backend\":\"{backend}\"}} / "
            "engine_args.max_model_len=null")


#: Every stack description the committed goldens hold, and what is compared of it.
DESCRIBED = [
    ("provider (anthropic_batch): no inference stack recorded",
     "provider (anthropic_batch): no inference stack recorded"),
    ("provider (openai_batch): no inference stack recorded", "provider (openai_batch): no inference stack recorded"),
    ("provider", "provider"),
    ("vLLM 0.19.0, torch 2.10.0", "vLLM <version>, torch <version>"),
    ("vLLM 0.28.0+cu129, torch 2.13.0+cu129, transformers 5.17.0",
     "vLLM <version>, torch <version>, transformers <version>"),
    ("vllm 0.19.0 / torch 2.10.0", "vllm <version> / torch <version>"),
    *((f"vllm 0.19.0 / torch 2.10.0 against {_V028} / {_engine('null', backend)}",
       f"vllm <version> / torch <version> against {_M028} / {_engine('null', backend)}")
      for backend in ("FLASH_ATTN", "TRITON_ATTN")),
    *((f"{_V028} / {_engine(dtype, backend)} / engine_args={digest}",
       f"{_M028} / {_engine(dtype, backend)} / engine_args={digest}")
      for dtype, backend, digest in ((_FP32, "TRITON_ATTN", "820b4a9eadfb90be"),
                                     (_FP32, "TRITON_ATTN", "fd77bf42fef3e8de"),
                                     ("null", "FLASH_ATTN", "4fb7b98f0cf575c6"),
                                     ("null", "FLASH_ATTN", "51df8d972a9cbf87"),
                                     ("null", "FLASH_ATTN", "9320284de5d2c6e5"),
                                     ("null", "TRITON_ATTN", "b4de527ca7469bb7"))),
]


@pytest.mark.parametrize(("description", "compared"), DESCRIBED)
def test_a_stack_description_is_compared_with_only_its_package_versions_masked(description, compared):
    from ruleforge.experiments.reproduce.rerun import unversioned

    assert unversioned(description) == compared


def test_the_table_holds_every_stack_description_of_the_goldens_and_each_name_a_version_follows():
    from ruleforge.evaluate.stack import STACK_PACKAGES
    from ruleforge.experiments.campaign_roots import SHORT_STACK_PACKAGES
    from ruleforge.experiments.reproduce.rerun import _STACK_DESCRIPTION, VERSIONED_NAMES

    def described(node, path=""):
        if isinstance(node, dict):
            return {text for name, value in node.items() for text in described(value, f"{path}.{name}" if path else name)}
        if isinstance(node, list):
            return {text for index, value in enumerate(node) for text in described(value, f"{path}[{index}]")}
        return {node} if _STACK_DESCRIPTION.search(path) and node is not None else set()

    found = {text for entry in table.goldens() for text in described(_published(entry.name)[0])}
    assert found == {description for description, _ in DESCRIBED}
    assert set(VERSIONED_NAMES) == {*STACK_PACKAGES, *(shown for _, shown in SHORT_STACK_PACKAGES)}
    assert all(any(f"{name} " in text for text in found) for name in VERSIONED_NAMES)


_GEMMA_STACKS = [("provenance", "arms", "base", "stack"), ("provenance", "arms", "final", "stack"),
                 ("provenance", "engine", "stack")]


def _where(path) -> str:
    return "".join(f"[{step}]" if isinstance(step, int) else f".{step}" for step in path).lstrip(".")


def _described(name, edits):
    committed, entry = _published(name)
    rerun = json.loads(json.dumps(committed))
    for path, old, new in edits:
        node = rerun
        for step in path[:-1]:
            node = node[step]
        assert old in node[path[-1]]
        node[path[-1]] = node[path[-1]].replace(old, new)
    return committed, rerun, entry


@pytest.mark.parametrize(("name", "edits"), [
    ("training_gemma", [(path, 'dtype="float32"', 'dtype="bfloat16"') for path in _GEMMA_STACKS]),
    # the context limit
    ("training_gemma", [(("provenance", "arms", "final", "stack"), "max_model_len=null", "max_model_len=4096")]),
    ("hierarchy", [(("judges", 0, "flat", "stack", "description"), "max_model_len=null", "max_model_len=4096")]),
    ("prompt_ablation", [(("table", 0, "cross_stack", "conflict"), "max_model_len=null", "max_model_len=4096")]),
    # a precision setting
    ("prompt_ablation", [(("provenance", "arms", "Olmo3-32B", "encouraged", "stack", "description"),
                          "VLLM_BATCH_INVARIANT=0", "VLLM_BATCH_INVARIANT=1")]),
    # the provider that served the judge
    ("prompt_ablation", [(("provenance", "arms", "claude-haiku-5-5", "encouraged", "stack", "description"),
                          "anthropic_batch", "openai_batch")]),
    # the attention backend
    ("prompt_ablation", [(("provenance", "arms", "gemma4-31b", "encouraged", "stack", "description"),
                          "TRITON_ATTN", "FLASH_ATTN")]),
])
def test_a_serving_setting_or_provider_inside_a_stack_description_is_design(name, edits):
    committed, rerun, entry = _described(name, edits)
    assert sorted((f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))) == sorted(
        ("design", _where(path)) for path, _, _ in edits)


def test_a_package_version_that_moved_inside_a_stack_description_is_one_notice():
    edits = [*((path, "vllm 0.28.0+cu129", "vllm 0.29.1+cu130") for path in _GEMMA_STACKS),
             (("provenance", "engine", "short"), "torch 2.13.0+cu129", "torch 2.14.0+cu130")]
    committed, rerun, entry = _described("training_gemma", edits)
    rerun["disclosure"]["engine"]["vllm_version"] = "0.29.1+cu130"
    report = verify_golden(entry, committed, rerun)
    assert [(f["check"], f["fatal"]) for f in report["findings"] if f["check"] != "drift"] == [("environment", False)]
    assert report["checks_run"]["environment"] == len(edits) + 1
    # Every description, short stack and stack conflict of the prompt table on another vLLM.
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed).replace("0.28.0+cu129", "0.29.1+cu130"))
    report = verify_golden(entry, committed, rerun)
    assert [(f["check"], f["fatal"]) for f in report["findings"] if f["check"] != "drift"] == [("environment", False)]


def test_a_judge_or_row_the_rerun_adds_is_fatal_and_where_it_ran_is_not_compared():
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    rerun["rates"]["another-judge"] = rerun["rates"]["gpt-oss-120b"]
    assert [(f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))] == [
        ("extra", "rates.another-judge")]
    rerun = json.loads(json.dumps(committed))
    rerun["table"][0].update(baseline_root="artifact://sampling/mine", grid_digest_a="0" * 64)
    for campaign in rerun["provenance"]["campaigns"]:
        campaign.update(estimates="artifact://sampling/mine/aggregate/estimates.json", campaign_id="mine")
    assert _fatal(_findings_of(committed, rerun, entry)) == []
    rerun["provenance"]["campaigns"] = []
    assert [f.path for f in _fatal(_findings_of(committed, rerun, entry))] == ["provenance.campaigns[]"]


def test_a_design_value_the_rerun_lost_is_fatal():
    committed, entry = _published("ruleset_size_curves")
    rerun = json.loads(json.dumps(committed))
    del rerun[0]["label"]
    assert [(f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))] == [("missing", "[0].label")]
    baseline, entry = _published("ruleset_size_repeated_draw_baseline")
    rerun = json.loads(json.dumps(baseline))
    rerun["subset_sizes"] = [None] * len(rerun["subset_sizes"])
    assert [(f.check, f.path) for f in _fatal(_findings_of(baseline, rerun, entry))] == [
        ("design", f"subset_sizes[{index}]") for index in range(len(rerun["subset_sizes"]))]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_rerun_value_that_is_no_finite_number_is_fatal(value):
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    rerun["table"][0]["p_value"] = value
    assert [(f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))] == [
        ("missing", "table[0].p_value")]
    rerun = json.loads(json.dumps(committed))
    rerun["table"][0]["ci_low"] = value
    assert ("missing", "table[0].difference") in {(f.check, f.path)
                                                 for f in _fatal(_findings_of(committed, rerun, entry))}


def test_a_value_lost_from_a_list_of_numbers_is_fatal():
    findings, counts = claim_findings("g", {"flagged": [0.25, -0.5, True]}, {"flagged": [0.25, None, False]}, 0.005)
    assert [(f.check, f.path, f.fatal) for f in findings] == [("missing", "flagged[1]", True),
                                                               ("flag", "flagged[2]", False)]
    findings, counts = claim_findings("g", {"p": [0.25, -0.5]}, {"p": [-0.25, -0.5]}, 0.005)
    assert [(f.check, f.path, f.fatal) for f in findings] == [("sign", "p[0]", False)] and counts["bare_sign"] == 2


def test_the_walk_reaches_into_lists_and_reports_the_index():
    committed = {"points": [_point(0.9, 0.85, 0.95), _point(0.8, 0.75, 0.85)]}
    rerun = {"points": [_point(0.9, 0.85, 0.95), _point(0.1, 0.05, 0.15)]}
    findings, _ = claim_findings("g", committed, rerun, 0.005)
    assert [(f.check, f.path) for f in findings] == [
        ("containment", "points[1].mean"), ("overlap", "points[1].mean")]


def test_a_length_change_on_a_claim_bearing_list_is_fatal():
    findings, counts = claim_findings("g", {"p": [_point(0.9, 0.8, 1.0)]},
                                      {"p": []}, 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing"]
    assert "never ran" in findings[0].detail
    assert counts["overlap"] == 0


def test_a_length_change_on_an_interval_free_list_is_left_to_compare():
    findings, counts = claim_findings("g", {"p": [1, 2, 3]}, {"p": [1, 2]}, 0.005)
    assert [(f.check, f.path) for f in _fatal(findings)] == [("missing", "p[]")] and counts["overlap"] == 0
    findings, _ = claim_findings(
        "g", {"p": [{"meta": {"created_at": "x"}}, {"meta": {"created_at": "y"}}]},
        {"p": [{"meta": {"created_at": "x"}}]}, 0.005)
    assert findings == []
    assert [f.check for f in claim_findings("g", {"p": [None, None]}, {"p": [None]}, 0.005)[0]] == ["missing"]


def test_a_length_change_on_a_table_of_numeric_rows_is_fatal():
    row = {"model": "gpt-oss", "b": 25, "c": 7, "p_value": 0.001}
    findings, _ = claim_findings(
        "g", {"table": [row, row, row]}, {"table": [row]}, 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing"]
    assert findings[0].path == "table[]"


def test_a_claim_nested_below_the_list_entries_still_makes_the_length_fatal():
    wrapped = {"point": _point(0.9, 0.8, 1.0)}
    findings, _ = claim_findings("g", {"p": [wrapped, wrapped]}, {"p": [wrapped]},
                                 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing"]


def test_a_partial_length_change_is_caught_where_it_happened():
    good = [_point(0.9, 0.8, 1.0)]
    findings, counts = claim_findings(
        "g",
        {"a": {"points": good}, "b": {"points": [_point(0.5, 0.4, 0.6)] * 2}},
        {"a": {"points": good}, "b": {"points": [_point(0.5, 0.4, 0.6)]}},
        0.005,
    )
    assert [f.check for f in _fatal(findings)] == ["missing"]
    assert findings[0].path == "b.points[]"
    assert counts["overlap"] == 1


def test_a_rerun_that_drops_its_point_estimates_cannot_pass_silently():
    findings, counts = claim_findings(
        "g", _point(0.87, 0.83, 0.91), {"ci_low": 0.83, "ci_high": 0.91}, 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing"]
    assert counts["containment"] == 0

    findings, _ = claim_findings(
        "g", {"ci_low": 0.83, "ci_high": 0.91}, _point(0.87, 0.83, 0.91), 0.005)
    assert [(f.check, f.path) for f in findings] == [("extra", "mean")]


def test_an_interval_bounded_by_exactly_zero_claims_no_direction():
    assert not Interval("ci", None, None, 0.0, 0.5, ()).excludes_zero()
    assert not Interval("ci", None, None, -0.5, 0.0, ()).excludes_zero()
    findings, counts = claim_findings(
        "g", {"b": 0.2, "b_ci": [0.0, 0.5]}, {"b": -0.01, "b_ci": [-0.1, 0.4]},
        0.005)
    assert counts["sign"] == 0
    assert "sign" not in [f.check for f in findings]


def test_a_dropped_mapping_key_with_claims_under_it_is_fatal():
    committed = {"flat_full": {"model": {"beta": -0.5, "beta_ci": [-0.9, -0.1]}}}
    findings, counts = claim_findings("g", committed, {"other": 1}, 0.005)
    assert [(f.check, f.path) for f in _fatal(findings)] == [("missing", "flat_full"), ("extra", "other")]
    assert counts["containment"] == 0

    findings, _ = claim_findings("g", {"a": {"mean": 1.0}}, {"b": 2}, 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing", "extra"]

    findings, _ = claim_findings("g", {"a": 1.0}, {"a": 1.0, "extra": 9}, 0.005)
    assert [(f.check, f.path) for f in findings] == [("extra", "extra")]
    findings, _ = claim_findings("g", {"note": "x", "a": 1.0}, {"a": 1.0}, 0.005)
    assert [(f.check, f.path) for f in findings] == [("missing", "note")]
    findings, _ = claim_findings("g", {"campaign_root": "x", "a": 1.0}, {"a": 1.0, "created_at": "y"}, 0.005)
    assert findings == []


@pytest.mark.parametrize("before, after", [(1.0, [1.0]), (1.0, None), (1.0, "1.0"), (1.0, True),
                                           (True, None), (True, "true"), (True, 1), (False, [False])])
def test_a_claim_leaf_that_stops_being_its_type_is_fatal(before, after):
    findings, _ = claim_findings("g", {"significant": before}, {"significant": after}, 0.005)
    assert [(f.check, f.path, f.fatal) for f in findings] == [("missing", "significant", True)]


def test_a_label_turned_null_and_a_null_turned_number_are_both_fatal():
    findings, _ = claim_findings("g", {"a": "x", "b": None}, {"a": None, "b": 1.0}, 0.005)
    assert [(f.check, f.path) for f in findings] == [("design", "a"), ("design", "b")]


@pytest.mark.parametrize("value", [1.0, True, False])
def test_a_deleted_claim_leaf_is_fatal(value):
    findings, _ = claim_findings("g", {"a": value, "b": "x"}, {"b": "x"}, 0.005)
    assert [(f.check, f.path, f.fatal) for f in findings] == [("missing", "a", True)]


def test_a_claim_bearing_subtree_turned_null_or_another_container_is_fatal():
    row = _point(0.5, 0.4, 0.6)
    for committed, rerun, path in [({"table": [row]}, {"table": None}, "table"),
                                   ({"table": [row]}, {"table": {"row": row}}, "table"),
                                   ({"model": row}, {"model": None}, "model"),
                                   ({"model": row}, {"model": [row]}, "model"),
                                   ({"model": {"rate": 0.5}}, {"model": 0.5}, "model")]:
        findings, _ = claim_findings("g", committed, rerun, 0.005)
        assert [(f.check, f.path) for f in _fatal(findings)] == [("missing", path)], (committed, rerun)


def test_a_subtree_that_records_only_where_a_run_happened_turned_null_is_left_to_compare():
    findings, _ = claim_findings("g", {"stack": {"dates": {"first_request_block": "x"}}, "campaign": {"estimates": "a"}},
                                 {"stack": None, "campaign": None}, 0.005)
    assert findings == []
    findings, _ = claim_findings("g", {"stack": {"short": "vLLM 0.19"}}, {"stack": None}, 0.005)
    assert [(f.check, f.path) for f in _fatal(findings)] == [("missing", "stack")]
    findings, _ = claim_findings("g", {"provenance": {"judge": "x"}}, {"provenance": None}, 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing"]
    findings, _ = claim_findings("g", {"labels": ["a"]}, {"labels": None}, 0.005)
    assert [f.check for f in _fatal(findings)] == ["missing"]


def _published(name: str) -> tuple[dict, Golden]:
    [entry] = [golden for golden in table.goldens() if golden.name == name]
    return json.loads((REPO_ROOT / entry.path).read_text(encoding="utf-8")), entry


def _findings_of(committed, rerun, entry):
    return claim_findings(entry.name, committed, rerun, entry.rerun_tolerance)[0]


def test_the_prompt_table_turned_null_fails_the_tolerance_check():
    committed, entry = _published("prompt_ablation")
    findings = _findings_of(committed, {**committed, "table": None}, entry)
    assert [(f.check, f.path) for f in _fatal(findings)] == [("missing", "table")]


@pytest.mark.parametrize("key", ["baseline_rate", "significant_holm"])
def test_a_prompt_table_leaf_turned_null_fails_the_tolerance_check(key):
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    assert isinstance(rerun["table"][0][key], (bool, float))
    rerun["table"][0][key] = None
    assert [(f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))] == [
        ("missing", f"table[0].{key}")]


def test_a_prompt_table_boolean_deleted_fails_the_tolerance_check():
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    del rerun["table"][0]["significant_holm"]
    assert [(f.check, f.path) for f in _fatal(_findings_of(committed, rerun, entry))] == [
        ("missing", "table[0].significant_holm")]


def _nulled(node, kind):
    if isinstance(node, dict):
        return {key: None if type(value) in kind else _nulled(value, kind) for key, value in node.items()}
    return [_nulled(value, kind) for value in node] if isinstance(node, list) else node


@pytest.mark.parametrize("kind", [(int, float), (bool,)], ids=["numbers", "booleans"])
def test_the_prompt_table_with_every_claim_leaf_null_fails_verify(kind):
    committed, entry = _published("prompt_ablation")
    report = verify_golden(entry, committed, _nulled(committed, kind))
    fatal = {(f["check"], f["path"]) for f in report["findings"] if f["fatal"]}
    assert ("missing", "table[0].significant_holm" if bool in kind else "table[0].baseline_rate") in fatal


def test_a_failure_reason_a_new_sample_does_not_hold_is_not_fatal():
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    reasons = rerun["diagnostics"]["conditions"]["gpt-5.6-luna/forced"]["attempt_statistics"]["invalid_components"]
    reasons["missing_full_set_judgment"] = reasons.pop("invalid_json")
    report = verify_golden(entry, committed, rerun)
    assert [f for f in report["findings"] if f["fatal"]] == [] and report["differences"] == 2


def test_an_attempt_count_a_new_sample_does_not_reach_is_not_fatal():
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    del rerun["diagnostics"]["conditions"]["gpt-5.6-luna/forced"]["attempt_statistics"]["attempts_histogram"]["7"]
    assert _fatal(_findings_of(committed, rerun, entry)) == []


def test_how_a_run_s_attempts_went_is_where_it_happened_and_not_compared():
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    del rerun["diagnostics"]["conditions"]["gpt-5.6-luna/forced"]["attempt_statistics"]["rejected_attempts"]
    assert _fatal(_findings_of(committed, rerun, entry)) == []


@pytest.mark.parametrize("change", ["flip", "drop"])
def test_the_monte_carlo_margin_flag_is_noise_and_a_significance_flag_flip_is_reported(change):
    committed, entry = _published("prompt_ablation")
    rerun = json.loads(json.dumps(committed))
    assert rerun["table"][0]["holm_mc_margin"] is True
    if change == "flip":
        rerun["table"][0]["holm_mc_margin"] = False
    else:
        del rerun["table"][0]["holm_mc_margin"]
    report = verify_golden(entry, committed, rerun)
    assert [(f["check"], f["fatal"]) for f in report["findings"]] == [("drift", False)]
    rerun["table"][0]["significant_holm"] = not rerun["table"][0]["significant_holm"]
    assert [(f.check, f.path, f.fatal) for f in _findings_of(committed, rerun, entry)] == [
        ("flag", "table[0].significant_holm", False)]
    rerun = json.loads(json.dumps(committed))
    rerun["table"][0]["significant_bh"] = not rerun["table"][0]["significant_bh"]
    assert [(f.check, f.path, f.fatal) for f in _findings_of(committed, rerun, entry)] == [
        ("flag", "table[0].significant_bh", False)]


def test_each_published_golden_compared_with_itself_has_no_findings():
    for entry in table.goldens():
        committed = read_golden(REPO_ROOT / entry.path)
        assert _findings_of(committed, committed, entry) == [], entry.name


def _shifted(row: dict, estimate: str) -> dict:
    """``row`` with its estimate and interval moved up by 1.5 half-widths."""
    shift = 1.5 * (row["ci_high"] - row["ci_low"]) / 2
    return {**row, estimate: row[estimate] + shift, "ci_low": row["ci_low"] + shift, "ci_high": row["ci_high"] + shift}


def test_a_prompt_table_row_shifted_past_its_interval_is_reported():
    committed, entry = _published("prompt_ablation")
    table = list(committed["table"])
    table[0] = _shifted(table[0], "difference")
    assert [(f.check, f.path, f.fatal) for f in _findings_of(committed, {**committed, "table": table}, entry)] == [
        ("containment", "table[0].difference", False)]
    assert _findings_of(committed, committed, entry) == []


def test_a_leniency_effect_shifted_past_its_interval_is_reported():
    committed, entry = _published("leniency")
    rerun = json.loads(json.dumps(committed))
    effects = rerun["models"]["Olmo3-32B"]["effects"]
    effects[0] = _shifted(effects[0], "delta")
    assert [(f.check, f.path, f.fatal) for f in _findings_of(committed, rerun, entry)] == [
        ("containment", "models.Olmo3-32B.effects[0].delta", False)]


def test_a_json_golden_reads_as_its_own_shape(tmp_path):
    path = tmp_path / "curves.json"
    path.write_text(json.dumps([{"mean": 1.0}]))
    assert read_golden(path) == [{"mean": 1.0}]


def test_unreadable_json_names_the_file_rather_than_raising_a_decode_error(tmp_path):
    path = tmp_path / "curves.json"
    path.write_text("{not json")
    with pytest.raises(RerunError, match="cannot be read as JSON"):
        read_golden(path)


def test_drift_within_tolerance_is_not_reported_at_all():
    report = verify_golden(_entry(), [_point(0.87, 0.83, 0.91)],
                           [_point(0.871, 0.83, 0.91)])
    assert report["differences"] == 0 and report["findings"] == []


def test_drift_beyond_tolerance_is_advisory_when_a_tolerance_is_given():
    report = verify_golden(_entry(), [_point(0.87, 0.83, 0.91)],
                           [_point(0.86, 0.83, 0.91)])
    assert [(f["check"], f["fatal"]) for f in report["findings"]] == [
        ("drift", False)]
    assert report["difference_paths"] == ["[0].mean"]


def test_drift_is_reported_even_when_the_tolerance_is_zero():
    report = verify_golden(_entry(rerun_tolerance=0.0), {"estimate": 1.0}, {"estimate": 1.5})
    assert [(f["check"], f["fatal"]) for f in report["findings"]] == [("drift", False)]
    assert report["largest_differences"] == [{"path": "estimate", "was": 1.0, "now": 1.5}]


def test_a_report_carries_no_prose_to_echo():
    report = verify_golden(_entry(), [1], [1])
    assert set(report) == {"name", "path", "status", "tolerance", "differences", "difference_paths",
                           "largest_differences", "checks_run", "findings"}


def test_selection_defaults_to_every_golden():
    assert [g.name for g in select([_entry(name="a"), _entry(name="b")])] == ["a", "b"]


def test_selection_keeps_the_table_s_order_not_the_order_asked_for():
    assert [g.name for g in select([_entry(name="a"), _entry(name="b")], ["b", "a"])] == ["a", "b"]


def test_an_unknown_selection_raises_rather_than_verifying_nothing():
    with pytest.raises(RerunError, match="no golden named typo"):
        select([_entry(name="a")], ["typo"])


@pytest.fixture
def tree(tmp_path):
    """A repo root, a rerun root, and one golden."""

    def _build(committed, rerun, entry=None):
        entry = entry or _entry()
        repo = tmp_path / "repo"
        run = tmp_path / "rerun"
        (repo / "results").mkdir(parents=True)
        (repo / entry.path).write_text(json.dumps(committed))
        if rerun is not None:
            (run / "results").mkdir(parents=True)
            (run / entry.path).write_text(json.dumps(rerun))
        return repo, run, [entry]

    return _build


def test_a_clean_rerun_is_ok(tree):
    repo, run, goldens = tree([_point(0.9, 0.85, 0.95)],
                               [_point(0.9, 0.85, 0.95)])
    report = verify_rerun(goldens, repo, run)
    assert report["ok"] and report["fatal"] == []


def test_a_moved_estimate_is_reported_and_a_structural_change_makes_the_report_not_ok(tree):
    repo, run, goldens = tree([_point(0.9, 0.85, 0.95)], [_point(0.1, 0.05, 0.15)])
    report = verify_rerun(goldens, repo, run)
    assert report["ok"] and sorted(_checks(report["advisory"])) == ["containment", "drift", "overlap"]
    lost = verify_golden(_entry(), [_point(0.9, 0.85, 0.95)], [])
    assert [(f["check"], f["fatal"]) for f in lost["findings"] if f["check"] != "drift"] == [("missing", True)]


def test_a_missing_rerun_of_a_reproducible_golden_is_fatal(tree):
    repo, run, goldens = tree([1], None)
    run.mkdir()
    report = verify_rerun(goldens, repo, run)
    assert not report["ok"]
    assert report["fatal"][0]["check"] == "missing"
    assert report["goldens"][0]["status"] == "skipped"


def test_a_table_golden_missing_from_the_repo_is_a_structural_error(tree):
    repo, run, goldens = tree([1], [1])
    (repo / "results" / "g.json").unlink()
    with pytest.raises(RerunError, match="in the experiment table but not committed"):
        verify_rerun(goldens, repo, run)


def test_advisory_findings_do_not_sink_the_run(tree):
    repo, run, goldens = tree([_point(0.9, 0.85, 0.95)],
                               [_point(0.89, 0.85, 0.95)])
    report = verify_rerun(goldens, repo, run)
    assert report["ok"] and [f["check"] for f in report["advisory"]] == ["drift"]


def test_the_run_s_report_holds_its_goldens_and_findings_alone(tree):
    repo, run, goldens = tree([1], [1])
    assert set(verify_rerun(goldens, repo, run)) == {"tool", "goldens", "fatal", "advisory", "ok"}


def _copied_goldens(repo_root, rerun):
    for golden in table.goldens():
        target = rerun / golden.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo_root / golden.path).read_bytes())
    return rerun


def test_the_committed_goldens_verify_against_themselves_and_none_is_skipped(repo_root, tmp_path):
    report = verify_rerun(table.goldens(), repo_root, _copied_goldens(repo_root, tmp_path / "rerun"))
    assert report["ok"], report["fatal"][:3]
    assert [g["name"] for g in report["goldens"]] == [g.name for g in table.goldens()]
    assert {g["status"] for g in report["goldens"]} == {"checked"}


def test_the_real_goldens_are_checked_and_not_merely_walked(repo_root, tmp_path):
    rerun = _copied_goldens(repo_root, tmp_path / "rerun")
    counts = {g["name"]: g.get("checks_run", {}) for g in verify_rerun(table.goldens(), repo_root, rerun)["goldens"]}
    assert counts["ruleset_size_curves"]["containment"] >= 100
    name = "prompt_ablation"
    golden, _ = _published(name)
    judges = golden["bars"]
    assert set(golden["diagnostics"]["separate_vs_same_call"]) == set(judges) and len(judges) >= 4
    expected = sum(len(judge["per_bar"]) + 1 for judge in judges.values()) + len(golden["table"])
    assert counts[name]["containment"] == expected >= 20
    assert counts[name]["bare_sign"] > 300


@pytest.fixture
def cli(tree, monkeypatch):
    def _build(committed, rerun, entry=None):
        repo, run, goldens = tree(committed, rerun, entry)
        monkeypatch.setattr(table, "TABLE", (Experiment("x", builds=(Step("b", "python -c pass", golden=goldens[0]),)),))
        return run, ["--rerun-root", str(run), "--repo-root", str(repo)]

    return _build


def test_the_script_exits_zero_when_every_check_passes(cli, capsys):
    _, argv = cli([_point(0.9, 0.85, 0.95)], [_point(0.9, 0.85, 0.95)])
    assert verify_script.main(argv) == 0
    out = capsys.readouterr().out
    assert "every structural check passed" in out
    assert "intervals compared       1" in out and "outside the interval     0 (" in out


def test_the_script_exits_one_and_names_the_first_failure(cli, capsys):
    _, argv = cli([{**_point(0.9, 0.85, 0.95), "label": "A"}], [{**_point(0.9, 0.85, 0.95), "label": "B"}])
    assert verify_script.main(argv) == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err and "design" in captured.err
    assert "FAIL g:" in captured.out


def test_the_script_prints_no_prose(cli, capsys):
    _, argv = cli([1], [1])
    assert verify_script.main(argv) == 0
    out = capsys.readouterr().out
    assert "machine-checked" not in out and "prose" not in out


def test_the_script_prints_one_table_per_golden_with_the_misses_expected_by_chance(cli, capsys):
    committed = [{**_point(0.9, 0.85, 0.95), "sd": 0.025}, {**_point(0.5, 0.45, 0.55), "sd": 0.025}]
    rerun = [_point(0.97, 0.92, 1.0), {**_point(0.5, 0.45, 0.55), "extra_note": "x"}]
    _, argv = cli(committed, rerun)
    assert verify_script.main(argv) == 1
    out = capsys.readouterr().out
    assert "intervals compared       2" in out and "outside the interval     1 (0.3 expected by chance)" in out
    assert "[0].mean 0.9 -> 0.97" in out.split("largest differences")[1] and "note g: extra" not in out
    assert "FAIL g: extra at [1].extra_note" in out


def test_an_interval_with_no_stored_standard_deviation_has_no_expectation(cli, capsys):
    committed = [{**_point(0.9, 0.85, 0.95), "sd": 0.025}, _point(0.5, 0.45, 0.55)]
    _, argv = cli(committed, [{**_point(0.97, 0.92, 1.0), "sd": 0.025}, _point(0.5, 0.45, 0.55)])
    assert verify_script.main(argv) == 0
    assert ("outside the interval     1 (0.2 expected by chance; expectation unavailable for 1 with no stored "
            "standard deviation)") in capsys.readouterr().out


def test_the_script_fails_a_golden_the_rerun_lacks(cli, capsys):
    run, argv = cli([1], None)
    run.mkdir()
    assert verify_script.main(argv) == 1
    out = capsys.readouterr().out
    assert "skip g:" in out and "FAIL g: missing" in out


def test_the_script_exits_two_on_a_structural_problem(cli, capsys):
    _, argv = cli([1], [1])
    assert verify_script.main(argv + ["--only", "typo"]) == 2
    assert "cannot verify this rerun" in capsys.readouterr().err


def test_the_terminal_output_is_truncated_but_the_report_is_not(
    cli, tmp_path, capsys
):
    committed = [{**_point(0.9, 0.85, 0.95), "label": "A"} for _ in range(verify_script.MAX_LINES + 5)]
    rerun = [{**point, "label": "B"} for point in committed]
    _, argv = cli(committed, rerun)
    out = tmp_path / "report.json"
    assert verify_script.main(argv + ["--json", str(out)]) == 1
    printed = capsys.readouterr().out
    assert printed.count("  FAIL ") == verify_script.MAX_LINES
    assert "--json has all of them" in printed
    report = json.loads(out.read_text())
    assert len(report["fatal"]) == len(committed)


def test_the_script_writes_a_json_report(cli, tmp_path):
    _, argv = cli([1], [1])
    out = tmp_path / "report.json"
    verify_script.main(argv + ["--json", str(out)])
    report = json.loads(out.read_text())
    assert report["tool"] == "verify_rerun"


def test_the_report_records_the_rerun_root_as_an_artifact_uri(
    cli, tmp_path, monkeypatch
):
    _, argv = cli([1], [1])
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    out = tmp_path / "report.json"
    verify_script.main(argv + ["--json", str(out)])
    assert json.loads(out.read_text())["rerun_root"] == "artifact://rerun"


def test_the_rerun_root_falls_back_to_a_plain_path_outside_the_artifact_root(
    cli, tmp_path, monkeypatch
):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)
    _, argv = cli([1], [1])
    out = tmp_path / "report.json"
    verify_script.main(argv + ["--json", str(out)])
    assert json.loads(out.read_text())["rerun_root"].endswith("rerun")


def test_the_script_accepts_an_artifact_uri_for_the_rerun_root(
    cli, tmp_path, monkeypatch, capsys
):
    _, argv = cli([1], [1])
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    argv[argv.index("--rerun-root") + 1] = "artifact://rerun"
    assert verify_script.main(argv) == 0
    assert "artifact://rerun" in capsys.readouterr().out


def test_a_finding_renders_for_a_human():
    finding = Finding("g", "sign", "", "beta flipped", fatal=True)
    assert finding.render() == "g: sign at .: beta flipped"
    assert finding.as_dict()["fatal"] is True


def _tree_and_report(argv):
    return (Path(argv[argv.index(flag) + 1]) for flag in ("--rerun-root", "--repo-root"))


def test_the_script_refuses_the_checkout_as_its_own_rerun_whatever_its_spelling(cli, monkeypatch, capsys):
    _, argv = cli([1], [1])
    _, repo = _tree_and_report(argv)
    monkeypatch.chdir(repo)
    for given in (".", str(repo), "results/.."):
        argv[argv.index("--rerun-root") + 1] = given
        assert verify_script.main(argv) == 2
        assert "is this checkout" in capsys.readouterr().err


def test_the_script_refuses_a_tree_holding_the_checkout_s_goldens_or_naming_nothing_or_two_places(
        cli, tmp_path, monkeypatch, capsys):
    _, argv = cli([1], [1])
    run, repo = _tree_and_report(argv)
    (run / "results" / "g.json").unlink()
    (run / "results" / "g.json").symlink_to(repo / "results" / "g.json")
    assert verify_script.main(argv) == 2
    assert "or holds its results/g.json" in capsys.readouterr().err
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere" / "child")
    monkeypatch.chdir(tmp_path)
    for given, message in [("missing/../rerun", "is no directory"), ("link/..", "names two places"),
                           (str(repo / "results" / "g.json"), "is no directory")]:
        argv[argv.index("--rerun-root") + 1] = given
        assert verify_script.main(argv) == 2
        assert message in capsys.readouterr().err, given


def test_the_script_s_report_never_replaces_what_the_check_reads_or_the_paper_commits(cli, tmp_path, capsys):
    _, argv = cli([1], [1])
    run, repo = _tree_and_report(argv)
    rebuilt = tmp_path / "rebuilt.json"
    rebuilt.write_text("[1]")
    (run / "results" / "g.json").unlink()
    (run / "results" / "g.json").symlink_to(rebuilt)
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere" / "child")
    (tmp_path / "linked.json").symlink_to(tmp_path / "elsewhere" / "linked.json")
    before = {path: path.read_bytes() for path in (rebuilt, repo / "results" / "g.json")}
    for report, message in [(repo / "results" / "report.json", "the checkout's results/"),
                            (run / "results" / "report.json", "the tree's results/"),
                            (rebuilt, "which the check reads"),
                            (tmp_path / "link" / ".." / "r.json", "names two places"),
                            (tmp_path / "linked.json", "is a symlink")]:
        assert verify_script.main(argv + ["--json", str(report)]) == 2, report
        captured = capsys.readouterr()
        assert message in captured.err and "verifying" not in captured.out, (report, captured.err)
    assert {path: path.read_bytes() for path in before} == before
    assert not (run / "results" / "report.json").exists()
    assert not (tmp_path / "elsewhere" / "r.json").exists() and not (tmp_path / "elsewhere" / "linked.json").exists()
    assert verify_script.main(argv + ["--json", str(tmp_path / "report.json")]) == 0


def test_a_report_is_never_written_into_this_checkout_s_committed_files_whatever_repo_root_names(tmp_path):
    from ruleforge.core.paths import repository_root
    from ruleforge.experiments.reproduce.verify_rerun import report_place

    golden = repository_root() / "results" / "ruleforge" / "ruleforge.json"
    for given in (golden, repository_root() / "results" / "report.json"):
        with pytest.raises(RerunError, match=r"lies in the checkout's results/"):
            report_place(given, repo_root=tmp_path / "reference", tree=tmp_path / "tree", goldens=[])


def test_an_installed_program_with_no_checkout_of_its_own_holds_the_reference_s_committed_files(tmp_path,
                                                                                                monkeypatch):
    from ruleforge.core.errors import ArtifactRootError
    from ruleforge.experiments.reproduce import verify_rerun

    def no_checkout():
        raise ArtifactRootError("no checkout")

    monkeypatch.setattr(verify_rerun, "repository_root", no_checkout)
    reference = tmp_path / "reference"
    with pytest.raises(RerunError, match=r"lies in the checkout's results/"):
        verify_rerun.report_place(reference / "results" / "report.json", repo_root=reference, tree=tmp_path / "tree",
                                  goldens=[])
    assert verify_rerun.report_place(tmp_path / "out" / "report.json", repo_root=reference, tree=tmp_path / "tree",
                                     goldens=[]) == tmp_path / "out" / "report.json"


def test_a_moved_value_is_reported_with_both_sides():
    assert differences({"a": 1.0}, {"a": 1.0}) == {}
    assert differences({"a": 1.0}, {"a": 2.0})["a"] == {"was": 1.0, "now": 2.0, "kind": "changed"}


def test_a_difference_inside_the_tolerance_is_not_a_difference():
    assert not differences({"a": 1.0}, {"a": 1.0001}, 1e-3)
    assert differences({"a": 1.0}, {"a": 1.1}, 1e-3)
    assert not differences({"a": 1}, {"a": 1.0005}, 1e-2)


def test_added_and_removed_keys_are_differences_at_any_depth():
    assert differences({}, {"a": 1})["a"]["kind"] == "added"
    assert differences({"a": 1}, {})["a"]["kind"] == "removed"
    assert differences({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}})["a.c"]["kind"] == "added"
    assert differences({"a": {"b": 1, "c": 2}}, {"a": {"b": 1}})["a.c"]["kind"] == "removed"


@pytest.mark.parametrize("before, after", [(True, 1.0), (1.0, True), ("x", "y"), (1.0, "1.0"), ({"b": 1}, 1),
                                           (float("nan"), 1.0), (1.0, float("nan")), (2 ** 53, 2 ** 53 + 1)])
def test_a_leaf_of_another_kind_or_value_is_a_difference_at_any_tolerance(before, after):
    assert differences({"a": before}, {"a": after}, 1e9 if isinstance(before, float) else 0.0)


@pytest.mark.parametrize("value", [True, "x", float("nan"), float("inf"), 2 ** 53])
def test_a_leaf_compares_equal_to_itself(value):
    assert not differences({"a": value}, {"a": value})


def test_differences_are_keyed_by_path_in_sorted_order():
    assert list(differences({"b": 1, "a": 1}, {"b": 2, "a": 2})) == ["a", "b"]
    assert "rows[0].p" in differences({"rows": [{"p": 0.1}]}, {"rows": [{"p": 0.9}]})
    assert not differences({"rows": [{"p": 0.10}]}, {"rows": [{"p": 0.1001}]}, 1e-3)
    assert differences([1.0, 2.0], [1.0, 3.0]) and not differences([1.0, 2.0], [1.0, 2.0])


def test_a_length_change_is_reported_as_a_length_difference():
    assert differences({"rows": [1, 2]}, {"rows": [1, 2, 3]})["rows"] == {"was": 2, "now": 3, "kind": "length"}


def test_a_file_of_the_rerun_tree_no_golden_is_fails_within_the_selection(tmp_path):
    repo, run = tmp_path / "repo", tmp_path / "rerun"
    entries = [_entry(name="a", path="results/a/g.json"), _entry(name="b", path="results/b/g.json")]
    for base in (repo, run):
        for entry in entries:
            (base / entry.path).parent.mkdir(parents=True, exist_ok=True)
            (base / entry.path).write_text(json.dumps({"rate": 0.5}))
    (run / "results" / "b" / "unexpected.json").write_text("{}")
    report = verify_rerun(entries, repo, run)
    assert not report["ok"] and [(f["check"], f["path"]) for f in report["fatal"]] == [
        ("extra", "results/b/unexpected.json")]
    assert verify_rerun(entries[:1], repo, run)["ok"]  # --only a: b's directory is not this check's
    for entry in table.goldens():
        (run / entry.path).parent.mkdir(parents=True, exist_ok=True)
    (run / "results" / "elsewhere.json").write_text("{}")
    assert "results/elsewhere.json" in {finding.path for finding in extra_files(table.goldens(), run)}
