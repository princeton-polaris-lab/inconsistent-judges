from __future__ import annotations

import importlib
import json

import pytest

from ruleforge.core import record_names
from ruleforge.experiments.prompt_ablation import analysis

MANIFEST = "3" * 64
BOOTSTRAP = {"n_resamples": 10000, "ci_level": 0.95, "seed": 16}
RANDOMIZATION = {"n_permutations": 100000, "seed": 16}
OPEN_RATES = {"naive": 0.575, "encouraged": 0.527, "forced": 0.428}
LUNA_RATES = {"naive": 0.381, "encouraged": 0.388, "forced": 0.353}
OPEN_PS = [5e-5, 5e-5, 5e-5]
LUNA_PS = [0.45, 0.023, 0.00027]


@pytest.fixture(scope="module")
def driver(repo_root):
    module = importlib.import_module("ruleforge.experiments.prompt_ablation.build")
    return module


def _summary(rate, n=500, draws=5):
    return {
        "estimate": rate, "ci_low": max(0.0, rate - 0.03), "ci_high": rate + 0.03, "sd": 0.015,
        "interval_basis": "scenario_cluster_bootstrap", "two_stage": {"ci_low": rate - 0.04, "ci_high": rate + 0.04, "sd": 0.02},
        "variance_components": {"decoder_share_of_variance": 0.1}, "n_scenarios": n, "n_decoder_draws": draws,
        "grid_digest": "g" * 64, "condition_fingerprint": "f" * 64, "sampling_truncation": "model_default",
        "attempt_statistics": {"invalid_attempt_rate": 0.01, "attempts_histogram": {"1": n * draws}},
    }


def _contrast(a, b, rate_a, rate_b, p):
    return {
        "a": a, "b": b, "inner_resample": "independent", "estimate": rate_a - rate_b,
        "ci_low": rate_a - rate_b - 0.02, "ci_high": rate_a - rate_b + 0.02, "sd": 0.01,
        "interval_basis": "scenario_cluster_bootstrap", "two_stage": {"ci_low": 0, "ci_high": 0, "sd": 0},
        "se_analytic": 0.01, "rate_a": rate_a, "rate_b": rate_b, "n_scenarios": 500,
        "n_decoder_draws_a": 5, "n_decoder_draws_b": 5, "p_value": p, "n_permutations": 100000,
        "inference_method": "hierarchical_bootstrap_v2", "grid_digest_a": "a" * 64, "grid_digest_b": "b" * 64,
        "bootstrap_seed": 16, "family": "style_contrasts", "family_size": 3, "p_holm": min(1.0, p * 3),
        "significant_holm": p * 3 < 0.05, "test_method": "scenario_clustered_paired_randomization_v1",
    }


def _payload(campaign_id, model, rates, ps, *, state="complete", manifest=MANIFEST, excluded=None,
             styles=("naive", "encouraged", "forced"), forced=True, bootstrap=BOOTSTRAP, randomization=RANDOMIZATION,
             recorded=False, family=None):
    """A root's estimates, ``styles`` and ``rates`` by slot."""
    def key(slot):
        style = (family or analysis.SLOT_STYLES).get(slot, slot)
        return record_names.identity_key(style) if recorded else style

    conditions = {f"{model}/{key(s)}": dict(_summary(rates[s]), prompt_style=key(s), model_name=model) for s in styles}
    pairs = [("naive", "encouraged"), ("naive", "forced"), ("encouraged", "forced")]
    contrasts = [
        _contrast(f"{model}/{key(a)}", f"{model}/{key(b)}", rates[a], rates[b], p)
        for (a, b), p in zip(pairs, ps) if a in styles and b in styles
    ]
    payload = {
        "schema_version": 2, "campaign_id": campaign_id, "state": state,
        "inference_method": "hierarchical_bootstrap_v2", "n_scenarios": 500, "n_decoder_draws": 5,
        "temperature": 1.0, "max_semantic_attempts": 20, "forced_json_requirement": "complete_rule_map",
        "truncation": "model_default", "scenario_manifest_sha256": manifest,
        "bootstrap": bootstrap, "randomization_test": randomization,
        "conditions": conditions,
        "style_families": {"+".join(sorted(conditions)): {"members": sorted(conditions), "contrasts": contrasts}},
        "forced_conditions": {}, "excluded_conditions": excluded or {}, "created_at": "2026-09-12T00:00:00+00:00",
    }
    if forced and "forced" in styles:
        payload["forced_conditions"][f"{model}/{key('forced')}"] = {
            "method_a_vs_b": dict(_contrast(f"{model}/{key('forced')}", "method_b", rates["forced"], 0.002, 1e-5),
                                  inner_resample="joint"),
            "method_b_rate": _summary(0.002),
            "rule_context": [],
        }
    return payload


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Two estimates files under a temporary artifact root, as the real run has."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))

    def write(name, payload):
        path = tmp_path / "sampling" / name / "aggregate" / "estimates.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return {
        "open": write("gpu", _payload("gpu", "gpt-oss-120b", OPEN_RATES, OPEN_PS)),
        "luna": write("luna", _payload("luna", "gpt-5.6-luna", LUNA_RATES, LUNA_PS)),
        "write": write,
    }


def test_two_roots_make_one_table_with_families_over_every_row(driver, roots):
    from statsmodels.stats.multitest import multipletests

    payload = driver.build_payload([roots["open"], roots["luna"]])
    table = payload["table"]
    # Paper order: gpt-oss before Luna, and the three contrasts in TABLE_CONTRASTS order.
    assert [(r["label"], r["baseline"], r["treatment"]) for r in table] == [
        ("gpt-oss-120b", "naive", "encouraged"), ("gpt-oss-120b", "naive", "forced"),
        ("gpt-oss-120b", "encouraged", "forced"), ("gpt-5.6-luna", "naive", "encouraged"),
        ("gpt-5.6-luna", "naive", "forced"), ("gpt-5.6-luna", "encouraged", "forced"),
    ]
    raw = [r["p_value"] for r in table]
    _, holm, _, _ = multipletests(raw, alpha=0.05, method="holm")
    _, bh, _, _ = multipletests(raw, alpha=0.05, method="fdr_bh")
    assert [r["p_holm"] for r in table] == pytest.approx(list(holm))
    assert [r["p_bh"] for r in table] == pytest.approx(list(bh))
    assert all(r["adjustment_family_n"] == 6 for r in table)
    assert [r["p_holm"] for r in table[:3]] == pytest.approx([6 * 5e-5] * 3)
    assert table[4]["p_holm"] == pytest.approx(2 * 0.023)
    assert table[3]["p_holm"] == pytest.approx(0.45)
    luna_naive_forced = table[4]
    assert luna_naive_forced["difference"] == pytest.approx(0.381 - 0.353)
    assert luna_naive_forced["improvement"] is True and luna_naive_forced["significant"] is True
    prov = payload["provenance"]
    assert prov["complete"] is True and prov["metric"] == "flat" and prov["alpha"] == 0.05
    assert [c["estimates"] for c in prov["campaigns"]] == [
        "artifact://sampling/gpu/aggregate/estimates.json", "artifact://sampling/luna/aggregate/estimates.json",
    ]
    assert prov["model_labels"] == {"gpt-oss-120b": "gpt-oss-120b", "gpt-5.6-luna": "gpt-5.6-luna"}
    assert prov["bar_settings"]["interval_basis"] == "scenario_cluster_bootstrap"
    assert prov["bar_settings"]["inference_method"] == "hierarchical_bootstrap_v2"
    assert prov["bar_settings"]["bars"] == ["naive", "encouraged", "forced", "forced-b"]


def test_a_recorded_root_makes_the_table_its_public_spelling_makes(driver, roots):
    def build(recorded):
        def key(style):
            return record_names.identity_key(style) if recorded else style

        excluded = {f"gpt-oss-120b/{key('naive')}": {"draw_states": {"0": "complete"}}}
        path = roots["write"](f"gpu-{recorded}", _payload("gpu", "gpt-oss-120b", OPEN_RATES, OPEN_PS,
                                                          recorded=recorded, excluded=excluded))
        assert sorted(json.loads(path.read_text())["conditions"]) == sorted(
            f"gpt-oss-120b/{key(analysis.SLOT_STYLES[slot])}" for slot in OPEN_RATES)
        payload = driver.build_payload([path, roots["luna"]], allow_incomplete=True)
        for campaign in payload["provenance"]["campaigns"]:
            for field in ("estimates", "estimates_sha256", "created_at"):
                campaign.pop(field)
        return payload

    recorded, public = build(True), build(False)
    assert recorded == public
    assert sorted(recorded["diagnostics"]["conditions"]) == sorted(
        [*(f"gpt-oss-120b/{analysis.SLOT_STYLES[slot]}" for slot in OPEN_RATES),
         *(f"gpt-5.6-luna/{analysis.SLOT_STYLES[slot]}" for slot in LUNA_RATES)])
    assert recorded["diagnostics"]["excluded_conditions"] == {"gpt-oss-120b/naive": {"draw_states": {"0": "complete"}}}
    assert recorded["provenance"]["campaigns"][0]["conditions"] == sorted(
        f"gpt-oss-120b/{analysis.SLOT_STYLES[slot]}" for slot in OPEN_RATES)


def test_a_root_of_a_style_outside_the_slots_is_refused(driver, roots):
    for family, style in (({**analysis.SLOT_STYLES, "forced": "forced-grpo"}, "forced-grpo"), ({"naive": "naive"}, "naive")):
        other = roots["write"](f"gpu-{style}", _payload("gpu", "gpt-oss-120b", OPEN_RATES, OPEN_PS, recorded=True,
                                                         family=family))
        with pytest.raises(driver.CampaignTableError, match=f"gpt-oss-120b/{style}: unknown prompt style '{style}'"):
            driver.build_payload([other])


def test_bars_block_is_what_the_panels_and_the_site_read(driver, roots):
    bar_block = driver.build_payload([roots["open"], roots["luna"]])["bars"]
    assert set(bar_block) == {"gpt-oss-120b", "gpt-5.6-luna"}
    bars = bar_block["gpt-oss-120b"]["per_bar"]
    assert set(bars) == {"naive", "encouraged", "forced", "forced-b"}
    assert bars["naive"] == {"mean": 0.575, "ci_low": pytest.approx(0.545), "ci_high": pytest.approx(0.605), "n": 500, "degenerate": False}
    assert bars["forced-b"]["mean"] == 0.002  # the forced arm's same-call rate
    assert "common" not in bar_block["gpt-oss-120b"]  # one population: the complete grid


def test_refuses_roots_that_are_not_one_experiment(driver, roots):
    gemma = {"naive": 0.5, "encouraged": 0.5, "forced": 0.5}
    other = roots["write"]("other", _payload("other", "gemma-4-31B-it", gemma, [0.5] * 3, manifest="4" * 64))
    with pytest.raises(driver.CampaignTableError, match="scenario_manifest_sha256"):
        driver.build_payload([roots["open"], other])
    fewer_perms = roots["write"]("perms", _payload("perms", "gemma-4-31B-it", gemma, [0.5] * 3, randomization={"n_permutations": 20000, "seed": 16}))
    with pytest.raises(driver.CampaignTableError, match="randomization_test"):
        driver.build_payload([roots["open"], fewer_perms])
    narrower = roots["write"]("ci", _payload("ci", "gemma-4-31B-it", gemma, [0.5] * 3, bootstrap={"n_resamples": 10000, "ci_level": 0.90, "seed": 16}))
    with pytest.raises(driver.CampaignTableError, match="bootstrap"):
        driver.build_payload([roots["open"], narrower])


def test_refuses_a_condition_that_appears_in_two_roots(driver, roots):
    dup = roots["write"]("dup", _payload("dup", "gpt-oss-120b", OPEN_RATES, OPEN_PS))
    with pytest.raises(driver.CampaignTableError, match="appears in both"):
        driver.build_payload([roots["open"], dup])


def test_refuses_an_unknown_judge_rather_than_guessing_a_label(driver, roots):
    odd = roots["write"]("odd", _payload("odd", "mystery-7b", OPEN_RATES, OPEN_PS))
    with pytest.raises(driver.CampaignTableError, match="no golden label"):
        driver.build_payload([odd])


def test_refuses_an_arm_that_is_not_a_prompt_style(driver, roots):
    payload = json.loads(roots["open"].read_text())
    payload["conditions"]["gpt-oss-120b/rule_context"] = payload["conditions"]["gpt-oss-120b/naive-rules-above"]
    weird = roots["write"]("weird", payload)
    with pytest.raises(driver.CampaignTableError, match="unknown prompt style"):
        driver.build_payload([weird])


def test_no_estimates_is_refused(driver, roots):
    with pytest.raises(driver.CampaignTableError, match="at least one"):
        driver.build_payload([])


def test_a_model_split_across_complete_roots_is_refused_on_the_strict_path(driver, roots, caplog):
    first = roots["write"]("split-a", _payload("split-a", "gemma-4-31B-it", {"naive": 0.63, "encouraged": 0.69, "forced": 0.65}, [0.5], styles=("naive", "encouraged"), forced=False))
    second = roots["write"]("split-b", _payload("split-b", "gemma-4-31B-it", {"naive": 0.63, "encouraged": 0.69, "forced": 0.65}, [], styles=("forced",)))
    with pytest.raises(driver.CampaignTableError, match="no naive vs forced contrast"):
        driver.build_payload([roots["luna"], first, second])
    payload = driver.build_payload([roots["luna"], first, second], allow_incomplete=True)
    assert [(r["label"], r["baseline"], r["treatment"]) for r in payload["table"] if r["label"] == "gemma4-31b"] == [("gemma4-31b", "naive", "encouraged")]
    assert "row omitted (scratch build)" in caplog.text
    assert set(payload["bars"]["gemma4-31b"]["per_bar"]) == {"naive", "encouraged", "forced", "forced-b"}


def test_an_incomplete_campaign_is_refused_unless_it_is_a_scratch_build(driver, roots, caplog):
    partial = roots["write"]("partial", _payload(
        "partial", "Olmo-3-32B-Think", {"naive": 0.736}, [], styles=("naive",), forced=False,
        state="in_progress", excluded={"Olmo-3-32B-Think/encouraged": {"draw_states": {}}},
    ))
    with pytest.raises(driver.CampaignTableError, match="complete campaign"):
        driver.build_payload([roots["open"], partial])
    payload = driver.build_payload([roots["open"], partial], allow_incomplete=True)
    assert payload["provenance"]["complete"] is False
    assert [r["label"] for r in payload["table"]] == ["gpt-oss-120b"] * 3
    assert "Olmo3-32B" not in payload["bars"]
    assert "Olmo3-32B" in payload["rates"]
    assert "left off the figure" in caplog.text
    assert set(payload["diagnostics"]["excluded_conditions"]) == {"Olmo-3-32B-Think/encouraged"}


def test_a_complete_model_missing_a_drawn_bar_is_refused_on_the_strict_path(driver, roots):
    grouped = driver.conditions_by_model([json.loads(roots["open"].read_text())])
    with pytest.raises(driver.CampaignTableError, match="forced-b"):
        driver.build_bars(grouped, {})


def test_validate_table_refuses_a_tampered_family(driver, roots):
    table = driver.build_payload([roots["open"], roots["luna"]])["table"]
    assert driver.validate_table(table, alpha=driver.ALPHA)[0]["p_holm"] == table[0]["p_holm"]
    tampered = [dict(r) for r in table]
    tampered[4]["p_holm"] = 0.001
    with pytest.raises(driver.CampaignTableError, match="stored p_holm"):
        driver.validate_table(tampered, alpha=driver.ALPHA)
    flipped = [dict(r) for r in table]
    flipped[3]["significant_holm"] = True
    with pytest.raises(driver.CampaignTableError, match="significant_holm"):
        driver.validate_table(flipped, alpha=driver.ALPHA)
    short = [dict(r) for r in table]
    short[0]["adjustment_family_n"] = 7
    with pytest.raises(driver.CampaignTableError, match="row count"):
        driver.validate_table(short, alpha=driver.ALPHA)
    with pytest.raises(driver.CampaignTableError, match="no rows"):
        driver.validate_table([], alpha=driver.ALPHA)


def test_csv_and_latex_render_the_holm_column_from_the_validated_table(driver, roots, tmp_path):
    table = driver.build_payload([roots["open"], roots["luna"]])["table"]
    csv_path, tex_path = tmp_path / "t.csv", tmp_path / "t.tex"
    driver.write_csv(table, csv_path)
    driver.write_latex(table, tex_path)
    lines = csv_path.read_text().splitlines()
    header = lines[0].split(",")
    assert header[:3] == ["model", "prompt_a", "prompt_b"] and "p_holm" in header and "metric" in header
    assert len(lines) == 7
    first = dict(zip(header, lines[1].split(",")))
    assert first["adjustment_family_n"] == "6" and first["metric"] == "flat" and first["significant_holm"] == "true"
    tex = tex_path.read_text()
    assert r"\begin{table}" in tex and "Holm-adjusted" in tex and r"\textbf{" in tex
    assert "$N=500$" in tex and "$D=5$" in tex and r"\alpha=0.05" in tex
    # A three-decimal column never prints 0.000: small p-values are a bound, bold in text mode.
    assert "0.000" not in tex
    assert r"\textbf{\textless 0.001}" in tex  # Luna encouraged->forced, Holm 8.1e-4, significant
    assert r"\textbf{$" not in tex


def test_alpha_is_threaded_from_the_build_to_every_rendering(driver, roots, tmp_path):
    golden, csv_path, tex_path = tmp_path / "g.json", tmp_path / "t.csv", tmp_path / "t.tex"
    argv = ["--estimates", str(roots["open"]), "--estimates", str(roots["luna"]),
            "--golden", str(golden), "--csv", str(csv_path), "--latex", str(tex_path)]
    assert driver.main(argv + ["--alpha", "0.01"]) == 0
    stored = json.loads(golden.read_text())
    assert stored["provenance"]["alpha"] == 0.01
    assert stored["table"][4]["significant_holm"] is False  # Luna naive->forced, Holm 0.046, not < 0.01
    assert csv_path.exists() and r"\alpha=0.01" in tex_path.read_text()
    assert driver.main(["--from_golden", str(golden), "--csv", str(tmp_path / "again.csv")]) == 0
    assert (tmp_path / "again.csv").read_bytes() == csv_path.read_bytes()
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden), "--alpha", "0.05", "--csv", str(tmp_path / "no.csv")])


def test_main_writes_a_golden_the_from_golden_path_re_renders_identically(driver, roots, tmp_path):
    golden, csv1, csv2 = tmp_path / "g.json", tmp_path / "1.csv", tmp_path / "2.csv"
    assert driver.main(["--estimates", str(roots["open"]), "--estimates", str(roots["luna"]),
                        "--golden", str(golden), "--csv", str(csv1), "--latex", str(tmp_path / "1.tex")]) == 0
    stored = json.loads(golden.read_text())
    assert stored["schema_version"] == analysis.TABLE_SCHEMA_VERSION
    assert driver.main(["--from_golden", str(golden), "--csv", str(csv2)]) == 0
    assert csv1.read_bytes() == csv2.read_bytes()
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden), "--estimates", str(roots["open"])])
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(dict(stored, schema_version=1)))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(stale)])
    with pytest.raises(SystemExit):
        driver.main([])
    scratch = tmp_path / "scratch.json"
    scratch.write_text(json.dumps(dict(stored, provenance=dict(stored["provenance"], complete=False))))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(scratch), "--csv", str(tmp_path / "3.csv")])
    assert driver.main(["--from_golden", str(scratch), "--allow-incomplete", "--csv", str(tmp_path / "3.csv")]) == 0


STACK_MISMATCH = [{
    "scope": "style_contrasts", "conditions": ["gpt-oss-120b/naive-rules-above", "gpt-oss-120b/encouraged"],
    "reason": "vllm 0.19 / torch 2.10 / transformers 5.5 against vllm 0.28 / torch 2.13 / transformers 5.17",
}]


def test_estimates_reporting_a_stack_mismatch_are_built_only_on_the_record(driver, roots, tmp_path):
    mixed = roots["write"]("mixed", dict(
        _payload("mixed", "gpt-oss-120b", OPEN_RATES, OPEN_PS), allow_stack_mismatch=True,
        stack_mismatches=STACK_MISMATCH,
    ))
    with pytest.raises(driver.CampaignTableError, match=(
        r"mixed/aggregate/estimates.json reports 1 inference-stack mismatch\(es\).*"
        r"gpt-oss-120b/naive-rules-above vs gpt-oss-120b/encouraged"
    )):
        driver.build_payload([mixed, roots["luna"]])
    payload = driver.build_payload([mixed, roots["luna"]], allow_stack_mismatch=True)
    flagged, luna = payload["provenance"]["campaigns"]
    assert flagged["stack_mismatches"] == STACK_MISMATCH and "stack_mismatches" not in luna
    clean = roots["write"]("clean", dict(
        _payload("clean", "gpt-oss-120b", OPEN_RATES, OPEN_PS), allow_stack_mismatch=True, stack_mismatches=[],
    ))
    assert "stack_mismatches" not in driver.build_payload([clean, roots["luna"]])["provenance"]["campaigns"][0]
    golden = tmp_path / "g.json"
    argv = ["--estimates", str(mixed), "--estimates", str(roots["luna"]), "--golden", str(golden)]
    with pytest.raises(SystemExit):
        driver.main(argv)
    assert driver.main([*argv, "--allow-stack-mismatch"]) == 0
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden), "--csv", str(tmp_path / "t.csv")])
    assert driver.main(["--from_golden", str(golden), "--allow-stack-mismatch", "--csv", str(tmp_path / "t.csv")]) == 0


def test_main_draws_both_panels(driver, roots, tmp_path):
    pytest.importorskip("matplotlib")
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    assert driver.main(["--estimates", str(roots["open"]), "--estimates", str(roots["luna"]),
                        "--figure_styles", str(a), "--figure_methods", str(b), "--dpi", "50"]) == 0
    assert all(p.stat().st_size > 1_000 for p in (a, b))


def test_mixed_interval_bases_are_refused(driver, roots):
    payload = json.loads(roots["open"].read_text())
    payload["conditions"]["gpt-oss-120b/encouraged"]["interval_basis"] = "percentile_bootstrap"
    mixed = roots["write"]("mixed", payload)
    with pytest.raises(driver.CampaignTableError, match="interval bases"):
        driver.build_payload([mixed])


def test_the_latex_prints_the_goldens_interval_level_not_a_literal(driver, roots, tmp_path):
    ninety = {"n_resamples": 10000, "ci_level": 0.90, "seed": 16}
    a = roots["write"]("a90", _payload("a90", "gpt-oss-120b", OPEN_RATES, OPEN_PS, bootstrap=ninety))
    b = roots["write"]("b90", _payload("b90", "gpt-5.6-luna", LUNA_RATES, LUNA_PS, bootstrap=ninety))
    golden, tex = tmp_path / "g.json", tmp_path / "t.tex"
    assert driver.main(["--estimates", str(a), "--estimates", str(b), "--golden", str(golden), "--latex", str(tex)]) == 0
    text = tex.read_text()
    assert r"[90\% CI]" in text and r"with 90\% scenario-cluster" in text and "95" not in text
    stored = json.loads(golden.read_text())
    del stored["provenance"]["alpha"]
    (tmp_path / "noalpha.json").write_text(json.dumps(stored))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(tmp_path / "noalpha.json"), "--csv", str(tmp_path / "x.csv")])


def _committed_rows(repo_root):
    golden = json.loads((repo_root / "results/prompt_ablation/prompt_ablation.json").read_text(encoding="utf-8"))
    return golden["table"]


def test_the_margin_flag_marks_only_olmo_s_naive_to_encouraged_row_and_moves_no_holm_field(repo_root):
    committed = _committed_rows(repo_root)
    stripped = [{k: v for k, v in row.items() if k not in analysis.TABLE_MULTIPLICITY_KEYS} for row in committed]
    fresh = analysis.add_table_multiplicity(stripped, alpha=0.05)
    assert [(r["label"], r["baseline"], r["treatment"]) for r in fresh if r["holm_mc_margin"]] == [
        ("Olmo3-32B", "naive", "encouraged")]
    assert fresh[0]["holm_mc_margin"] is True and fresh[0]["significant_holm"] is False
    assert all(type(row["holm_mc_margin"]) is bool for row in fresh)
    for before, after in zip(committed, fresh):
        for key in ("p_value", "p_holm", "significant_holm"):
            assert repr(after[key]) == repr(before[key]), key


def test_a_stored_margin_flag_the_recomputation_refuses():
    rows = analysis.add_table_multiplicity(
        [{"label": "m", "baseline": "naive", "treatment": "forced", "p_value": p, "n_permutations": 100_000}
         for p in (0.01018, 0.9, 1e-5, 1e-5, 1e-5)], alpha=0.05)
    assert analysis.validate_table(rows, alpha=0.05) == rows
    rows[0] = {**rows[0], "holm_mc_margin": not rows[0]["holm_mc_margin"]}
    with pytest.raises(analysis.CampaignEstimatesError, match="stored holm_mc_margin disagrees"):
        analysis.validate_table(rows, alpha=0.05)
