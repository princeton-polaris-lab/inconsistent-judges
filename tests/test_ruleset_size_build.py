from __future__ import annotations

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest

from ruleforge.evaluate.estimates import public_amendment

PRODUCER = "ruleforge.experiments.ruleset_size.build"
SIZES = list(range(2, 58))
BOOTSTRAP = {"n_resamples": 10000, "ci_level": 0.95, "seed": 16}
MANIFEST = "3" * 64


@pytest.fixture(scope="module")
def driver():
    return importlib.import_module(PRODUCER)


def _points(base):
    out = []
    for size in SIZES:
        mean = min(0.95, base + 0.4 * (size - 2) / 55) - (0.03 if size == 57 else 0.0)
        out.append({"size": size, "estimate": mean, "ci_low": mean - 0.02, "ci_high": mean + 0.02, "sd": 0.01,
                    "two_stage": {"ci_low": mean - 0.03, "ci_high": mean + 0.03, "sd": 0.015}})
    return out


def _contrast(estimate, minuend, subtrahend):
    return {"minuend_sizes": minuend, "subtrahend_sizes": subtrahend, "estimate": estimate,
            "ci_low": estimate - 0.01, "ci_high": estimate + 0.01, "sd": 0.005,
            "two_stage": {"ci_low": estimate - 0.02, "ci_high": estimate + 0.02, "sd": 0.01}}


def _model(model_name, base):
    points = _points(base)
    by_size = {p["size"]: p["estimate"] for p in points}
    middle = sum(by_size[k] for k in range(20, 31)) / 11
    return {
        "model_name": model_name,
        "judge_type": "No_Interpretation",
        "bank": {"condition_id": f"{model_name}/bank", "condition_fingerprint": "b" * 64, "n_rules": 57,
                 "grid_digest": "g" * 64, "attempt_statistics": {"attempts_histogram": {"1": 2500}}},
        "subsets": {"n_conditions": 560, "grid_digest": "s" * 64, "attempt_statistics": {"attempts_histogram": {"1": 1400000}}},
        "one_ruleset_sizes": [57],
        "points": points,
        "contrasts": {
            "net_rise": _contrast(by_size[57] - by_size[2], [57], [2]),
            "upper_vs_middle": _contrast(by_size[56] - middle, [56], list(range(20, 31))),
            "last_step": _contrast(by_size[57] - by_size[56], [57], [56]),
        },
        "adjacent_rises": sum(by_size[k + 1] > by_size[k] for k in SIZES[:-1]),
    }


def _estimates(campaign_id, models, *, state="complete", **overrides):
    payload = {
        "schema_version": 2, "campaign_id": campaign_id, "state": state,
        "design": {"design": "shared_individual_bank_v1", "subset_seeds": list(range(10)), "subset_sizes": SIZES,
                   "sampling_strategy": "with_replacement", "subset_manifest_sha256": "m" * 64, "source_ruleset_sha256": "r" * 64},
        "inference_method": "crossed_scenario_subset_bankshared_decoder_bootstrap_v2",
        "interval_basis": "scenario_subset_cluster_bootstrap",
        "n_scenarios": 500, "n_decoder_draws": 5, "n_subset_seeds": 10, "subset_sizes": SIZES,
        "temperature": 1.0, "max_semantic_attempts": 20, "truncation": "model_default", "label_rule": "lenient",
        "scenario_manifest_sha256": MANIFEST, "bootstrap": BOOTSTRAP,
        "contrasts": [{"name": "net_rise", "minuend": [57], "subtrahend": [2]},
                      {"name": "upper_vs_middle", "minuend": [56], "subtrahend": list(range(20, 31))},
                      {"name": "last_step", "minuend": [57], "subtrahend": [56]}],
        "models": {name: _model(name, base) for name, base in models.items()},
        "created_at": "2026-09-20T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))

    def write(name, payload):
        path = tmp_path / "sampling" / name / "aggregate" / "figure2_estimates.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return {
        "open": write("open", _estimates("open", {"Olmo-3-32B-Think": 0.2, "gemma-4-31B-it": 0.1, "gpt-oss-120b": 0.15})),
        "luna": write("luna", _estimates("luna", {"gpt-5.6-luna": 0.12})),
        "glimmer": write("glimmer", _estimates("glimmer", {"Muse-Glimmer-30B": 0.1})),
        "write": write,
    }


def test_two_roots_make_four_curves_in_legend_order(driver, roots):
    curves = driver.build_curves([roots["open"], roots["luna"]])
    assert [c["label"] for c in curves] == ["GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "GPT-5.6 Luna"]
    gemma = next(c for c in curves if c["label"] == "Gemma-4-31B")
    assert gemma["code_version"] == 5 and gemma["quantity"] == "inconsistency_rate" and gemma["metric"] == "contradiction"
    assert gemma["ci_estimator"] == "scenario_subset_cluster_bootstrap" and gemma["n_expected"] == 500 * 10 * 5
    assert gemma["strategy"] == "No_Interpretation"
    assert [p["size"] for p in gemma["points"]] == SIZES
    assert gemma["points"][0]["mean"] == pytest.approx(0.1) and gemma["points"][0]["n_obs"] == 25000
    assert gemma["points"][0]["two_stage"]["sd"] == 0.015
    assert [p["size"] for p in gemma["points"] if p["shares_one_ruleset"]] == [57]
    assert gemma["contrasts"]["net_rise"]["contrast"] == "K57 - K2"
    assert gemma["contrasts"]["last_step"]["contrast"] == "K57 - K56"
    assert gemma["contrasts"]["upper_vs_middle"]["contrast"] == "K56 - mean(K20..K30)"
    assert gemma["contrasts"]["net_rise"]["excludes_zero"] is True
    # mean(57) = base + 0.4 - 0.03, mean(56) = base + 0.4 * 54 / 55.
    assert gemma["contrasts"]["last_step"]["estimate"] == pytest.approx(0.4 / 55 - 0.03)
    assert gemma["campaign"]["estimates"] == "artifact://sampling/open/aggregate/figure2_estimates.json"
    assert len(gemma["campaign"]["estimates_sha256"]) == 64 and gemma["campaign"]["model_name"] == "gemma-4-31B-it"
    assert gemma["campaign"]["design"]["design"] == "shared_individual_bank_v1"
    luna = curves[-1]
    assert luna["campaign"]["campaign_id"] == "luna" and luna["label"] == "GPT-5.6 Luna"


def test_a_root_retried_by_component_joins_the_others_and_its_curve_names_its_scope(driver, roots):
    unnamed = roots["write"]("unnamed", _estimates("unnamed", {"claude-opus-9": 0.11}, retry_scope="component"))
    with pytest.raises(driver.CampaignCurveError, match="claude-opus-9"):
        driver.build_curves([roots["open"], unnamed])
    haiku = roots["write"]("haiku", _estimates("haiku", {"claude-haiku-5-5": 0.11}, retry_scope="component"))
    curves = {c["campaign"]["campaign_id"]: c for c in driver.build_curves([roots["open"], roots["luna"], haiku])}
    assert curves["haiku"]["campaign"]["retry_scope"] == "component"
    assert {curves[name]["campaign"]["retry_scope"] for name in ("open", "luna")} == {"scenario_bundle"}
    assert {curves[name]["campaign"]["label_rule"] for name in curves} == {"lenient"}
    assert all("retry_scope" not in curve["campaign"] for curve in driver.build_curves([roots["open"], roots["luna"]]))
    bundle = roots["write"]("bundle", _estimates("bundle", {"Muse-Glimmer-30B": 0.12}, retry_scope="scenario_bundle"))
    assert all("retry_scope" not in curve["campaign"] for curve in driver.build_curves([roots["open"], bundle]))
    assert "retry_scope" not in driver.SHARED_DESIGN_FIELDS + driver.CAMPAIGN_SHARED_FIELDS
    other = roots["write"]("other", _estimates("other", {"Muse-Glimmer-30B": 0.1}, label_rule="lenient_last_line"))
    with pytest.raises(driver.CampaignCurveError, match="label_rule 'lenient_last_line' differs"):
        driver.build_curves([roots["open"], other])


def test_three_roots_make_five_curves_in_legend_order(driver, roots):
    curves = driver.build_curves([roots["glimmer"], roots["luna"], roots["open"]])
    assert [c["label"] for c in curves] == [
        "GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "Muse-Glimmer-30B", "GPT-5.6 Luna",
    ]
    glimmer = curves[3]
    assert glimmer["campaign"]["campaign_id"] == "glimmer" and glimmer["campaign"]["model_name"] == "Muse-Glimmer-30B"
    assert driver.validate_curves(curves) == curves


def test_the_estimates_are_recorded_by_the_digest_of_the_bytes_read(driver, roots):
    path = roots["open"]
    read = hashlib.sha256(path.read_bytes()).hexdigest()
    [payload] = driver.load_estimates([path], allow_incomplete=False)
    path.write_text(json.dumps({**json.loads(path.read_text()), "created_at": "2026-10-11T00:00:00Z"}))
    curve = driver.build_curve(payload, "gemma-4-31B-it", [57])
    assert curve["campaign"]["estimates_sha256"] == read != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.__setitem__("state", "in_progress"), "campaign state 'in_progress'"),
        (lambda p: p["design"].__setitem__("design", "literal_bundle"), "is not 'shared_individual_bank_v1'"),
        (lambda p: p.__setitem__("n_decoder_draws", 3), "n_decoder_draws 3 differs"),
        (lambda p: p.__setitem__("scenario_manifest_sha256", "4" * 64), "scenario_manifest_sha256"),
        (lambda p: p.__setitem__("subset_sizes", SIZES[:-1]), "subset_sizes"),
    ],
)
def test_a_root_that_is_not_the_same_experiment_is_refused(driver, roots, mutate, message):
    payload = _estimates("other", {"gpt-5.6-luna": 0.12})
    mutate(payload)
    other = roots["write"]("other", payload)
    with pytest.raises(driver.CampaignCurveError, match=message):
        driver.build_curves([roots["open"], other])


def test_an_incomplete_root_can_only_be_a_scratch_build(driver, roots):
    partial = roots["write"]("partial", _estimates("partial", {"gpt-5.6-luna": 0.12}, state="in_progress"))
    curves = driver.build_curves([partial], allow_incomplete=True)
    assert curves[0]["campaign"]["state"] == "in_progress"
    with pytest.raises(driver.CampaignCurveError, match="incomplete campaign"):
        driver.validate_curves(curves)


def test_an_unknown_judge_or_a_model_in_two_roots_is_refused(driver, roots):
    unknown = roots["write"]("unknown", _estimates("unknown", {"mystery-judge": 0.1}))
    with pytest.raises(driver.CampaignCurveError, match="no golden label for judge 'mystery-judge'"):
        driver.build_curves([unknown])
    twice = roots["write"]("twice", _estimates("twice", {"gpt-oss-120b": 0.1}))
    with pytest.raises(driver.CampaignCurveError, match="appears in both"):
        driver.build_curves([roots["open"], twice])
    with pytest.raises(driver.CampaignCurveError, match="at least one"):
        driver.build_curves([])


def test_a_campaign_judged_under_another_strategy_is_labelled_by_it_not_the_papers(driver, roots):
    payload = _estimates("narrow", {"gpt-5.6-luna": 0.12})
    payload["models"]["gpt-5.6-luna"]["judge_type"] = "Narrow"
    narrow = roots["write"]("narrow", payload)
    assert driver.build_curves([narrow])[0]["strategy"] == "Narrow"
    payload["models"]["gpt-5.6-luna"].pop("judge_type")
    unstamped = roots["write"]("unstamped", payload)
    with pytest.raises(driver.CampaignCurveError, match="record no judge_type"):
        driver.build_curves([unstamped])


def test_a_curve_whose_points_do_not_cover_the_design_is_refused(driver, roots):
    payload = _estimates("short", {"gpt-5.6-luna": 0.12})
    payload["models"]["gpt-5.6-luna"]["points"].pop()
    short = roots["write"]("short", payload)
    with pytest.raises(driver.CampaignCurveError, match="points cover"):
        driver.build_curves([short])


def test_validate_refuses_a_foreign_or_mixed_golden(driver, roots):
    curves = driver.build_curves([roots["open"]])
    assert driver.validate_curves(curves) == curves
    with pytest.raises(driver.CampaignCurveError, match="no curves"):
        driver.validate_curves([])
    old = dict(curves[0], code_version=4)
    with pytest.raises(driver.CampaignCurveError, match="code_version 4"):
        driver.validate_curves([old])
    graded = dict(curves[0], quantity="mean_graded_penalty")
    with pytest.raises(driver.CampaignCurveError, match="not the inconsistency_rate"):
        driver.validate_curves([graded])
    mixed = [curves[0], dict(curves[1], n_resamples=5)]
    with pytest.raises(driver.CampaignCurveError, match="disagree on n_resamples"):
        driver.validate_curves(mixed)
    fewer = dict(curves[1], points=curves[1]["points"][:-1])
    with pytest.raises(driver.CampaignCurveError, match="covers different sizes"):
        driver.validate_curves([curves[0], fewer])
    with pytest.raises(driver.CampaignCurveError, match="duplicate curve labels"):
        driver.validate_curves([curves[0], curves[0]])


def test_a_golden_hand_merged_from_two_campaigns_is_refused_on_re_render(driver, roots, tmp_path):
    other = roots["write"](
        "other", _estimates("other", {"gpt-5.6-luna": 0.12}, temperature=0.7, scenario_manifest_sha256="9" * 64)
    )
    merged = driver.build_curves([roots["open"]]) + driver.build_curves([other])
    with pytest.raises(driver.CampaignCurveError, match="campaign.scenario_manifest_sha256"):
        driver.validate_curves(merged)
    golden = tmp_path / "merged.json"
    golden.write_text(json.dumps(merged))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden)])


def test_the_golden_and_the_figure_come_from_one_set_of_curves(driver, roots, tmp_path):
    pytest.importorskip("matplotlib")
    golden = tmp_path / "curves.json"
    figure = tmp_path / "ruleset_size.png"
    assert driver.main([
        "--estimates", str(roots["open"]), "--estimates", str(roots["luna"]),
        "--golden", str(golden), "--figure", str(figure), "--dpi", "50",
    ]) == 0
    curves = json.loads(golden.read_text())
    assert [c["label"] for c in curves] == ["GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "GPT-5.6 Luna"]
    assert all({2, 10, 30, 57} <= {point["size"] for point in curve["points"]} for curve in curves)
    row = curves[2]["contrasts"]["net_rise"]
    assert row["contrast"] == "K57 - K2" and row["excludes_zero"] is True
    assert figure.exists() and figure.stat().st_size > 5_000


def test_the_figure_draws_every_curve_of_the_golden_built_or_read(driver, roots, tmp_path, monkeypatch):
    drawn = []
    monkeypatch.setattr(driver, "draw", lambda curves, _path, dpi=None: drawn.append([c["label"] for c in curves]))
    golden = tmp_path / "curves.json"
    assert driver.main([
        "--estimates", str(roots["open"]), "--estimates", str(roots["luna"]), "--estimates", str(roots["glimmer"]),
        "--golden", str(golden), "--figure", str(tmp_path / "f.png"),
    ]) == 0
    assert driver.main(["--from_golden", str(golden), "--figure", str(tmp_path / "g.png")]) == 0
    labels = [c["label"] for c in json.loads(golden.read_text())]
    assert drawn == [labels, labels] and len(labels) == 5


def test_estimates_reporting_a_stack_mismatch_are_built_only_on_the_record(driver, roots, tmp_path):
    record = {"scope": "gemma-4-31B-it", "conditions": ["gemma-4-31B-it/bank", "gemma-4-31B-it/subsets/s00/k02"],
              "reason": "vllm 0.19 / torch 2.10 / transformers 5.5 against vllm 0.28 / torch 2.13 / transformers 5.17"}
    mixed = roots["write"]("mixed", _estimates(
        "mixed", {"Olmo-3-32B-Think": 0.2, "gemma-4-31B-it": 0.1}, allow_stack_mismatch=True, stack_mismatches=[record],
    ))
    with pytest.raises(driver.CampaignCurveError, match=r"reports 1 inference-stack mismatch.*gemma-4-31B-it/bank vs"):
        driver.build_curves([mixed])
    curves = {c["campaign"]["model_name"]: c for c in driver.build_curves([mixed], allow_stack_mismatch=True)}
    assert curves["gemma-4-31B-it"]["campaign"]["stack_mismatches"] == [record]
    assert "stack_mismatches" not in curves["Olmo-3-32B-Think"]["campaign"]
    with pytest.raises(driver.CampaignCurveError, match=r"curve 'Gemma-4-31B' reports 1 inference-stack mismatch"):
        driver.validate_curves(list(curves.values()))
    assert len(driver.validate_curves(list(curves.values()), allow_stack_mismatch=True)) == 2
    golden = tmp_path / "curves.json"
    with pytest.raises(SystemExit):
        driver.main(["--estimates", str(mixed), "--golden", str(golden)])
    assert driver.main(["--estimates", str(mixed), "--golden", str(golden), "--allow-stack-mismatch"]) == 0
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden)])
    assert driver.main(["--from_golden", str(golden), "--allow-stack-mismatch"]) == 0


def test_from_golden_refuses_what_it_did_not_write(driver, roots, tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps([{"label": "GPT-OSS-120B", "code_version": 4}]))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(old)])
    not_a_list = tmp_path / "dict.json"
    not_a_list.write_text(json.dumps({"table": []}))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(not_a_list)])
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(old), "--estimates", str(roots["open"])])
    with pytest.raises(SystemExit):
        driver.main([])


def test_haiku_s_curve_lands_named_after_luna_its_scope_named_beside_the_others(driver, roots):
    haiku = roots["write"]("haiku", _estimates("haiku", {"claude-haiku-5-5": 0.09}, retry_scope="component"))
    curves = driver.build_curves([roots["open"], roots["luna"], haiku])
    assert [c["label"] for c in curves] == ["GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "GPT-5.6 Luna",
                                            "Claude Haiku 5.5"]
    by_label = {c["label"]: c["campaign"] for c in curves}
    assert by_label["Claude Haiku 5.5"]["retry_scope"] == "component"
    assert by_label["Claude Haiku 5.5"]["model_name"] == "claude-haiku-5-5"
    assert {by_label[label]["retry_scope"] for label in by_label if label != "Claude Haiku 5.5"} == {"scenario_bundle"}
    assert driver.validate_curves(json.loads(json.dumps(curves))) == curves
    assert driver.judge_label("claude-haiku-5-5") == "Claude Haiku 5.5"


def test_glimmer_s_curve_lands_named_with_the_open_judges_its_scope_named(driver, roots):
    haiku = roots["write"]("haiku", _estimates("haiku", {"claude-haiku-5-5": 0.09}, retry_scope="component"))
    glimmer = roots["write"]("glimmer-component", _estimates("glimmer", {"Muse-Glimmer-30B": 0.1},
                                                             retry_scope="component"))
    curves = driver.build_curves([glimmer, roots["open"], roots["luna"], haiku])
    assert [c["label"] for c in curves] == ["GPT-OSS-120B", "Gemma-4-31B", "Olmo-3-32B", "Muse-Glimmer-30B",
                                            "GPT-5.6 Luna", "Claude Haiku 5.5"]
    by_label = {c["label"]: c["campaign"] for c in curves}
    assert by_label["Muse-Glimmer-30B"]["retry_scope"] == "component"
    assert by_label["Muse-Glimmer-30B"]["model_name"] == "Muse-Glimmer-30B"
    assert {label for label, campaign in by_label.items() if campaign["retry_scope"] == "component"} == {
        "Muse-Glimmer-30B", "Claude Haiku 5.5"}
    assert driver.validate_curves(json.loads(json.dumps(curves))) == curves
    assert driver.judge_label("Muse-Glimmer-30B") == "Muse-Glimmer-30B"


def test_a_judge_s_legend_label_is_its_display_name_and_the_published_ones_stand(driver):
    assert {name: driver.judge_label(name) for name in (
        "gpt-oss-120b", "gemma-4-31B-it", "Olmo-3-32B-Think", "gpt-5.6-luna", "Qwen3-8B", "claude-haiku-5-5",
        "Muse-Glimmer-30B")} == {
        "gpt-oss-120b": "GPT-OSS-120B", "gemma-4-31B-it": "Gemma-4-31B", "Olmo-3-32B-Think": "Olmo-3-32B",
        "gpt-5.6-luna": "GPT-5.6 Luna", "Qwen3-8B": "Qwen3-8B", "claude-haiku-5-5": "Claude Haiku 5.5",
        "Muse-Glimmer-30B": "Muse-Glimmer-30B"}
    assert driver.judge_label("gpt-6.1-sol") == "GPT-6.1 Sol"
    with pytest.raises(driver.CampaignCurveError, match="no golden label for judge 'mystery-7b'"):
        driver.judge_label("mystery-7b")


def _amendment(old_cap=20, new_cap=50):
    return {"kind": "semantic_attempt_cap", "field": "max_semantic_attempts", "old_cap": old_cap, "new_cap": new_cap,
            "ledger": "note 143", "checks": {}, "bundles": [{"condition_id": "Muse-Glimmer-30B/subsets/s03/k12", "decoder_draw_id": 1,
                                                         "scenario_id": "a" * 64, "outcome": "accepted"}]}


def test_a_root_that_took_a_retry_ladder_joins_at_its_frozen_cap_its_amendment_beside_it(driver, roots):
    laddered = roots["write"]("laddered", _estimates("laddered", {"Muse-Glimmer-30B": 0.1}, max_semantic_attempts=50,
                                                     amendments=[_amendment()]))
    curves = driver.build_curves([roots["open"], roots["luna"], laddered])
    campaigns = {c["campaign"]["campaign_id"]: c["campaign"] for c in curves}
    assert campaigns["laddered"]["max_semantic_attempts"] == 20
    assert campaigns["laddered"]["amendments"] == [public_amendment(_amendment())]
    assert all("amendments" not in campaigns[name] for name in ("open", "luna"))
    assert driver.validate_curves(json.loads(json.dumps(curves))) == curves
    other = roots["write"]("other", _estimates("other", {"Muse-Glimmer-30B": 0.1}, max_semantic_attempts=50,
                                               amendments=[_amendment(old_cap=5)]))
    with pytest.raises(driver.CampaignCurveError, match="campaign.max_semantic_attempts"):
        driver.build_curves([roots["open"], other])
    mixed = [*curves[:1], {**curves[1], "campaign": {**curves[1]["campaign"], "max_semantic_attempts": 5}}]
    with pytest.raises(driver.CampaignCurveError, match="campaign.max_semantic_attempts"):
        driver.validate_curves(json.loads(json.dumps(mixed)))


@pytest.mark.parametrize("amendments", [[_amendment(new_cap=40)], [_amendment(), _amendment()], ["not a record"]])
def test_estimates_whose_cap_is_not_their_amendment_s_raise_are_refused(driver, roots, amendments):
    bad = roots["write"]("bad", _estimates("bad", {"Muse-Glimmer-30B": 0.1}, max_semantic_attempts=50,
                                           amendments=amendments))
    with pytest.raises(driver.CampaignCurveError, match="one amendment raising the cap to it"):
        driver.build_curves([roots["open"], bad])


def test_the_one_ruleset_sizes_are_those_whose_seeds_judged_one_file():
    from ruleforge.evaluate.bank import SharedBankDesign, one_ruleset_sizes

    def spec(digest):
        return SimpleNamespace(rule_set_sha256=digest)

    design = SharedBankDesign(subset_seeds=(0, 1), subset_sizes=(2, 57), sampling_strategy="with_replacement",
                              subset_manifest_sha256="m", source_ruleset_sha256="r")
    entry = {"subsets": {(0, 2): spec("a"), (1, 2): spec("b"), (0, 57): spec("c"), (1, 57): spec("c")}}
    assert one_ruleset_sizes(entry, design) == [57]
    entry["subsets"][(1, 57)] = spec("d")  # each seed's own order of the full set
    assert one_ruleset_sizes(entry, design) == []


def test_estimates_that_record_no_one_ruleset_sizes_are_read_from_their_root_s_conditions(driver, tmp_path, modelspec,
                                                                                          monkeypatch):
    from ruleforge.evaluate import bank
    from tests.test_sampling_ruleset_size import run_fake_campaign

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    root, *_ = run_fake_campaign(tmp_path, modelspec, bank)
    payload = bank.analyze_shared_bank_campaign(root, n_resamples=100)
    assert "one_ruleset_sizes" not in payload["models"]["gemma-4-31B-it"]
    payload["models"]["gemma-4-31B-it"]["model_name"] = "gemma-4-31B-it"
    estimates = root / "aggregate" / "figure2_estimates.json"
    estimates.write_text(json.dumps(payload))
    (curve,) = driver.build_curves([estimates])
    assert [point["size"] for point in curve["points"] if point["shares_one_ruleset"]] == [4]


def test_estimates_away_from_their_root_build_as_estimates_did_before_the_conditions_were_read(
        driver, tmp_path, modelspec, monkeypatch):
    from ruleforge.evaluate import bank
    from ruleforge.rulesets.subsets import WITH_REPLACEMENT_SHUFFLED_FULL
    from tests.test_sampling_ruleset_size import run_fake_campaign

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    for name, strategy in (("file_order", "with_replacement"), ("shuffled", WITH_REPLACEMENT_SHUFFLED_FULL)):
        (tmp_path / name).mkdir()
        root, *_ = run_fake_campaign(tmp_path / name, modelspec, bank, strategy=strategy)
        payload = bank.analyze_shared_bank_campaign(root, n_resamples=100)
        payload["models"]["gemma-4-31B-it"]["model_name"] = "gemma-4-31B-it"
        estimates = root / "aggregate" / "figure2_estimates.json"
        estimates.write_text(json.dumps(payload))
        moved = tmp_path / name / "elsewhere" / "figure2_estimates.json"
        moved.parent.mkdir()
        moved.write_text(estimates.read_text())
        if strategy == WITH_REPLACEMENT_SHUFFLED_FULL:
            with pytest.raises(driver.CampaignCurveError, match="lies away from its root"):
                driver.build_curves([moved])
            continue
        (here,), (there,) = driver.build_curves([estimates]), driver.build_curves([moved])
        assert [point["size"] for point in there["points"] if point["shares_one_ruleset"]] == [4]
        assert {**there, "campaign": {}} == {**here, "campaign": {}}
