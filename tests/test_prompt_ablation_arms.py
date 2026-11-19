from __future__ import annotations

import dataclasses
import importlib
import json
import os
import shutil
from contextlib import contextmanager

import numpy as np
import pytest

from ruleforge.core import record_names
from ruleforge.experiments import paper_outputs as pf
from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.experiments import campaign_roots as cr
from ruleforge.experiments.prompt_ablation import arms as ta
from ruleforge.experiments.labels import NAIVE_SLOT, SLOTS, STYLE_LABELS
from ruleforge.experiments.prompt_ablation.analysis import PROMPT_BARS, TABLE_CONTRASTS
from ruleforge.prompts.styles import ENCOURAGED, FORCED
from ruleforge.evaluate.inference import DrawGrid, clustered_paired_randomization_p
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.campaign import campaign_completion, load_conditions, register_condition
from ruleforge.evaluate.identity import ConditionSpec, recorded_fingerprint
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.ledger import AttemptRecord
from tests.test_leniency_campaign import (
    BOOTSTRAP,
    N,
    PERMUTATIONS,
    S,
    _component,
    _judgments,
    artifact_root,
    recorded_id,
    register_as_recorded,
    without_estimates_files,
    write_rules,
)

OLMO, GEMMA, OSS, LUNA = "Olmo-3-32B-Think", "gemma-4-31B-it", "gpt-oss-120b", "gpt-5.6-luna"
LABELS = {OLMO: "Olmo3-32B", GEMMA: "gemma4-31b", OSS: "gpt-oss-120b", LUNA: "gpt-5.6-luna"}
OLD = {"vllm_version": "0.19.0", "torch_version": "2.10.0"}
NEW = {"vllm_version": "0.28.0+cu129", "torch_version": "2.13.0+cu129", "transformers_version": "5.17.0"}
OLD_SHORT, NEW_SHORT = "vLLM 0.19.0, torch 2.10.0", "vLLM 0.28.0+cu129, torch 2.13.0+cu129, transformers 5.17.0"
MARKUP = {"label_rule": "lenient_markup"}
PUBLISHED_STYLES = ("naive-rules-above", "forced-grpo")
REVISED_STYLES = ("encouraged", "forced")
#: The roots of the study, as (name, [(model, styles, model_type, runtime)]).
ROOTS = {
    "published": ("final-2026-09-rescored-lenient_markup",
                  [(OLMO, PUBLISHED_STYLES, "local", OLD), (GEMMA, PUBLISHED_STYLES, "local", OLD),
                   (OSS, PUBLISHED_STYLES, "local", OLD)]),
    "published_luna": ("final-2026-09-luna-rescored-lenient_markup", [(LUNA, PUBLISHED_STYLES, "openai_batch", None)]),
    "gemma": ("prompt-ablation-v2-2026-10-rescored-lenient_markup",
                 [(GEMMA, ("naive", *REVISED_STYLES), "local", NEW)]),
    "open": ("prompt-ablation-v2-open-2026-10", [(OLMO, REVISED_STYLES, "local", NEW), (OSS, REVISED_STYLES, "local", NEW)]),
    "luna": ("prompt-ablation-v2-2026-10-luna", [(LUNA, REVISED_STYLES, "openai_batch", None)]),
}
REASON = "an accepted trade-off: the published naive arms ran on another stack"


def write_condition(base, name, *, model, style, seed, model_type="local", runtime=None, design=None,
                    request_dates=("2026-09-12T05:05:19+00:00", "2026-09-12T06:00:00+00:00"), recorded=False):
    """One condition's complete draws in ``base/sampling/name``, its manifests too."""
    root = base / "sampling" / name
    root.mkdir(parents=True, exist_ok=True)
    scenarios = base / "scenarios.jsonl"
    if not scenarios.exists():
        scenarios.write_text("".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) + "\n" for i in range(N)))
    manifest = ScenarioManifest.build(scenarios)
    manifest.write(root / "scenario_manifest.json")
    rules_path = base / "rules.jsonl"
    if not rules_path.exists():
        write_rules(rules_path)
    (root / "sampling_manifest.json").write_text(json.dumps({
        "schema_version": 2, "campaign_id": name, "campaign_mode": "final",
        "scenario_manifest_sha256": manifest.manifest_sha256, "expected_scenarios": N,
        "expected_decoder_draws": S, "temperature": 1.0, "max_semantic_attempts": 20,
        "forced_json_requirement": "complete_rule_map", "truncation": "model_default", "bootstrap": BOOTSTRAP,
    }))
    provider = model_type != "local"
    spec = ConditionSpec(
        condition_id=recorded_id(model, style) if recorded else f"{model}/{style}", model_name=model,
        model_type=model_type, prompt_style=style,
        rule_set_name="rules.jsonl", rule_set_sha256=sha256_file(rules_path),
        temperature=1.0, forced_json_requirement="complete_rule_map",
        sampling_truncation="provider_default" if provider else {"top_p": 1.0, "top_k": 0, "min_p": 0.0},
        seed_capability="unsupported" if provider else "local_deterministic",
        **{"individual_rule_context": "isolated", "judge_type": "No_Interpretation", "evaluation_mode": "both",
           "prompt_shape_version": 2, **MARKUP, **(design or {})},
    )
    (register_as_recorded if recorded else register_condition)(root, spec)
    cells = _judgments(seed, rule_10_rises=True, unsure_verdicts=True, preempt=False)
    for s in range(S):
        draw_dir = root / spec.condition_id / f"draw-{s:03d}"
        draw_dir.mkdir(parents=True)
        outcomes, ledger = [], []
        for i, sid in enumerate(manifest.scenario_ids):
            alone, in_context, verdict, _ = cells[(i, s)]
            attempt = 1 if (i, s) == (0, 0) else 0
            common = dict(campaign_id=name, condition_id=spec.condition_id, scenario_id=sid, decoder_draw_id=s,
                          seed_capability=spec.seed_capability, started_at=request_dates[0], completed_at="t1")
            if attempt:
                ledger.append(AttemptRecord(**common, semantic_attempt_id=0, status="rejected",
                                            request_block_id=f"{name}/draw-{s}/attempt-0/block-0000",
                                            components=(_component("full_ruleset", None),)))
            full = _component("full_ruleset", verdict, rule_judgments=dict(in_context), preempted_rules=())
            ledger.append(AttemptRecord(
                **common, semantic_attempt_id=attempt, status="accepted",
                request_block_id=f"{name}/draw-{s}/attempt-{attempt}/block-0000",
                components=(full, *(_component(rule_id, label) for rule_id, label in alone.items())),
            ))
            outcomes.append({
                "scenario_id": sid, "decoder_draw_id": s, "semantic_attempt_id": attempt,
                "full_set_judgment": verdict, "rule_judgments": alone, "in_context_rule_judgments": in_context,
                "contradiction": float(verdict == "no" and "yes" in alone.values()),
                "in_context_contradiction": float(verdict == "no" and "yes" in in_context.values()),
            })
        (draw_dir / "attempts.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in ledger))
        (draw_dir / "bundle_outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in outcomes))
        (draw_dir / "request_blocks.jsonl").write_text("".join(
            json.dumps({"request_block_id": f"{name}/draw-{s}/block-{k}", "recorded_at": stamp}) + "\n"
            for k, stamp in enumerate(request_dates)))
        if runtime is not None:
            (draw_dir / "draw_provenance.json").write_text(json.dumps({"runtime": runtime}))
        (draw_dir / "completion.json").write_text(json.dumps({
            "schema_version": 2, "campaign_id": name, "condition_id": spec.condition_id,
            "condition_fingerprint": spec.fingerprint, "decoder_draw_id": s, "state": "complete",
            "materialized": True, "scenario_manifest_sha256": manifest.manifest_sha256, "expected_scenarios": N,
            "accepted": N, "attempts_histogram": {"1": N - int(s == 0), **({"2": 1} if s == 0 else {})},
            "rejected_attempts": int(s == 0), "transport_failed_attempts": 0,
            "invalid_components": {"invalid_json": 1} if s == 0 else {},
        }))
    return root


def build_study_root(base, key, *, roots=ROOTS, designs=None, runtimes=None, recorded=False):
    """One of the study's roots, written and aggregated."""
    name, members = roots[key]
    seed = sum(map(ord, name)) % 997
    for model, styles, model_type, runtime in members:
        for index, style in enumerate(styles):
            cid = f"{model}/{style}"
            write_condition(base, name, model=model, style=style, seed=seed + 7 * index + len(model),
                            model_type=model_type, runtime=(runtimes or {}).get(cid, runtime),
                            design=(designs or {}).get(cid), recorded=recorded)
    analyze_campaign(base / "sampling" / name, n_permutations=PERMUTATIONS)
    return base / "sampling" / name


def arm_map(roots=None, judges=None, **top):
    """The study's arm map as a dict: the committed map's plan over the study's roots."""
    roots = roots or {key: f"artifact://sampling/{name}" for key, (name, _) in ROOTS.items()}

    def arms(model, naive_root, revised_root, **extra):
        return {"naive": {"root": naive_root, "condition": f"{model}/naive-rules-above"},
                "encouraged": {"root": revised_root, "condition": f"{model}/encouraged"},
                "forced": {"root": revised_root, "condition": f"{model}/forced"}, **extra}

    judges = judges or {
        OLMO: arms(OLMO, "published", "open", cross_stack="naive"),
        GEMMA: arms(GEMMA, "published", "gemma", cross_stack="naive"),
        OSS: arms(OSS, "published", "open", cross_stack="naive"),
        LUNA: arms(LUNA, "published_luna", "luna"),
    }
    return {"version": 1, "roots": roots, "judges": judges,
            **({"cross_stack_reason": REASON} if any("cross_stack" in j for j in judges.values()) else {}), **top}


def write_map(path, payload):
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    base = tmp_path_factory.mktemp("arms")
    with artifact_root(base):
        roots = {key: build_study_root(base, key) for key in ROOTS}
        path = write_map(base / "arms.yaml", arm_map())
        payload = ta.build_payload(path)
    return {"base": base, "roots": roots, "map": path, "payload": payload}


@pytest.fixture(scope="module")
def driver(repo_root):
    module = importlib.import_module("ruleforge.experiments.prompt_ablation.build")
    return module


def grid(root, cid):
    """The flat inconsistency per (scenario, draw), recounted from the bundle outcomes: an N x S array."""
    ids = json.loads((root / "scenario_manifest.json").read_text())["scenario_ids"]
    out = np.zeros((N, S))
    for s in range(S):
        rows = {r["scenario_id"]: r for r in map(json.loads, (root / cid / f"draw-{s:03d}" / "bundle_outcomes.jsonl")
                                                  .read_text().splitlines())}
        for i, sid in enumerate(ids):
            out[i, s] = rows[sid]["contradiction"]
    return out


def randomization_p(a, b):
    ids = tuple(f"s{i}" for i in range(N))
    return clustered_paired_randomization_p(DrawGrid("a", "q", ids, a), DrawGrid("b", "q", ids, b),
                                            n_permutations=PERMUTATIONS, seed=16)["p_value"]


def holm_by_hand(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        adjusted[k] = running
    return adjusted


def bh_by_hand(ps):
    order = sorted(range(len(ps)), key=lambda k: ps[k], reverse=True)
    adjusted, running = [0.0] * len(ps), 1.0
    for position, k in enumerate(order):
        rank = len(ps) - position
        running = min(running, len(ps) * ps[k] / rank)
        adjusted[k] = running
    return adjusted


def _rebuild(study, tmp_path, change=None, map_change=None, root_overrides=None):
    """The study with its map edited (``map_change``) and/or a root rebuilt (``root_overrides``: key -> kwargs)."""
    base = tmp_path
    for key, (name, _) in ROOTS.items():
        if root_overrides and key in root_overrides:
            continue
        shutil.copytree(study["roots"][key], base / "sampling" / name)
    shutil.copy(study["base"] / "scenarios.jsonl", base / "scenarios.jsonl")
    shutil.copy(study["base"] / "rules.jsonl", base / "rules.jsonl")
    with artifact_root(base):
        for key, kwargs in (root_overrides or {}).items():
            build_study_root(base, key, **kwargs)
    payload = arm_map()
    if map_change is not None:
        map_change(payload)
    path = write_map(base / "arms.yaml", payload)
    if change is not None:
        change(base)
    return base, path


def _build(base, path):
    with artifact_root(base):
        return ta.build_payload(path)


#: Root writes into a read-only tree, so a test that a reader writes nothing means nothing as root.
UNPRIVILEGED = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes a read-only tree")


@contextmanager
def read_only(tree):
    """``chmod -R a-w`` on ``tree`` for the block."""
    modes = {path: path.stat().st_mode for path in (tree, *tree.rglob("*"))}
    for path, mode in modes.items():
        path.chmod(mode & ~0o222)
    try:
        assert not os.access(tree, os.W_OK)
        yield
    finally:
        for path, mode in modes.items():
            path.chmod(mode)


@UNPRIVILEGED
def test_a_read_only_copy_of_the_roots_gives_the_same_golden_byte_for_byte(study, tmp_path):
    base = shutil.copytree(study["base"], tmp_path / "copy")
    with artifact_root(base):
        writable = json.dumps(ta.build_payload(base / "arms.yaml"), sort_keys=True)
        with read_only(base / "sampling"):
            assert json.dumps(ta.build_payload(base / "arms.yaml"), sort_keys=True) == writable
    assert writable == json.dumps(study["payload"], sort_keys=True)


def test_the_table_is_twelve_rows_in_paper_order_each_recounted_from_the_draws(study):
    table = study["payload"]["table"]
    order = [LABELS[m] for m in (OLMO, GEMMA, OSS, LUNA)]
    assert [(r["label"], r["baseline"], r["treatment"]) for r in table] == [
        (label, a, b) for label in order for a, b in TABLE_CONTRASTS]
    arms = study["payload"]["provenance"]["arms"]
    for row in table:
        a_arm, b_arm = arms[row["label"]][row["baseline"]], arms[row["label"]][row["treatment"]]
        roots = {uri: study["base"] / uri.removeprefix("artifact://") for uri in (a_arm["root"], b_arm["root"])}
        a, b = grid(roots[a_arm["root"]], a_arm["condition_id"]), grid(roots[b_arm["root"]], b_arm["condition_id"])
        assert row["difference"] == pytest.approx(a.mean() - b.mean(), abs=1e-12)
        assert (row["baseline_rate"], row["treatment_rate"]) == (pytest.approx(a.mean()), pytest.approx(b.mean()))
        assert row["p_value"] == randomization_p(a, b)
        assert row["ci_low"] <= row["difference"] <= row["ci_high"]
        assert (row["baseline_label"], row["treatment_label"]) == (STYLE_LABELS[row["baseline"]],
                                                                   STYLE_LABELS[row["treatment"]])
        assert (row["baseline_style"], row["treatment_style"]) == (a_arm["prompt_style"], b_arm["prompt_style"])
        assert (row["n_scenarios"], row["n_decoder_draws"], row["n_permutations"]) == (N, S, PERMUTATIONS)
        assert row["paired_within_root"] is (a_arm["root"] == b_arm["root"])
        assert row["improvement"] is (row["difference"] > 0) and row["significant"] is row["significant_holm"]
    styles = {(r["baseline_style"], r["treatment_style"]) for r in table}
    assert styles == {("naive-rules-above", "encouraged"), ("naive-rules-above", "forced"), ("encouraged", "forced")}


def test_holm_and_bh_run_over_all_twelve_contrasts_as_published(study):
    table = study["payload"]["table"]
    ps = [row["p_value"] for row in table]
    assert len(ps) == 12
    for row, holm, bh in zip(table, holm_by_hand(ps), bh_by_hand(ps)):
        assert row["p_holm"] == pytest.approx(holm, rel=1e-12) and row["p_bh"] == pytest.approx(bh, rel=1e-12)
        assert row["significant_holm"] is (holm < 0.05) and row["significant_bh"] is (bh < 0.05)
        assert (row["adjustment_family"], row["adjustment_family_n"]) == ("all_table_style_contrasts", 12)


def test_holm_and_bh_by_hand_on_a_small_case():
    ps = [0.01, 0.04, 0.03, 0.005, 0.2, 0.0001, 0.5, 0.02, 0.06, 0.001, 0.9, 0.03]
    from ruleforge.experiments.prompt_ablation.analysis import add_table_multiplicity

    rows = add_table_multiplicity([{"p_value": p, "n_permutations": 100_000} for p in ps])
    # Holm: the k-th smallest p times (12 - k + 1), running maximum; BH: p times 12 / rank, running minimum from the top.
    assert holm_by_hand(ps)[5] == pytest.approx(12 * 0.0001) and bh_by_hand(ps)[10] == pytest.approx(0.9)
    assert [row["p_holm"] for row in rows] == pytest.approx(holm_by_hand(ps))
    assert [row["p_bh"] for row in rows] == pytest.approx(bh_by_hand(ps))


def test_a_pair_within_one_root_is_the_aggregate_s_own_row(study):
    payload = study["payload"]
    for row in payload["table"]:
        if not row["paired_within_root"]:
            continue
        name = row["baseline_root"].removeprefix("artifact://sampling/")
        estimates = json.loads((study["base"] / "sampling" / name / "aggregate" / "estimates.json").read_text())
        stored = ta._aggregate_row(estimates, row["baseline_condition"], row["treatment_condition"])
        assert (row["difference"], row["ci_low"], row["ci_high"], row["p_value"]) == (
            stored["estimate"], stored["ci_low"], stored["ci_high"], stored["p_value"])
    assert sum(row["paired_within_root"] for row in payload["table"]) == 4  # every encouraged -> forced


def test_the_figure_block_rates_and_diagnostics_come_from_each_arm_s_root(study):
    payload = study["payload"]
    arms = payload["provenance"]["arms"]
    for label, block in payload["bars"].items():
        assert list(block) == ["per_bar"] and set(block["per_bar"]) == set(PROMPT_BARS)
        for slot in SLOTS:
            arm = arms[label][slot]
            estimates = json.loads((study["base"] / arm["root"].removeprefix("artifact://") / "aggregate" /
                                    "estimates.json").read_text())
            summary = estimates["conditions"][arm["condition_id"]]
            assert block["per_bar"][slot]["mean"] == summary["estimate"] == payload["rates"][label]["rates"][slot]
            diagnostics = payload["diagnostics"]["conditions"][arm["condition_id"]]
            assert (diagnostics["slot"], diagnostics["prompt_style"], diagnostics["grid_digest"]) == (
                slot, arm["prompt_style"], summary["grid_digest"])
        forced = arms[label][FORCED]
        estimates = json.loads((study["base"] / forced["root"].removeprefix("artifact://") / "aggregate" /
                                "estimates.json").read_text())
        assert block["per_bar"]["forced-b"]["mean"] == (
            estimates["forced_conditions"][forced["condition_id"]]["method_b_rate"]["estimate"])
    settings = payload["provenance"]["bar_settings"]
    assert settings["bars"] == ["naive", "encouraged", "forced", "forced-b"] and settings["population"] == "per_bar"


def test_the_naive_rows_cross_stacks_only_as_declared_and_are_recorded(study):
    payload = study["payload"]
    flagged = [(r["label"], r["baseline"], r["treatment"]) for r in payload["table"] if r["cross_stack"] is not None]
    assert flagged == [(LABELS[m], NAIVE_SLOT, slot) for m in (OLMO, GEMMA, OSS) for slot in (ENCOURAGED, FORCED)]
    for row in payload["table"]:
        if row["cross_stack"] is not None:
            record = row["cross_stack"]
            assert (record["declared"], record["baseline_stack"], record["treatment_stack"], record["fields"]) == (
                "naive", OLD_SHORT, NEW_SHORT, ["torch", "vllm"])
            assert "vllm 0.19.0" in record["conflict"] and "vllm 0.28.0+cu129" in record["conflict"]
    cross = payload["provenance"]["cross_stack"]
    assert cross["declared"] == sorted(LABELS[m] for m in (OLMO, GEMMA, OSS)) and "reason" not in cross
    assert cross["contrasts"] == [list(key) for key in flagged]
    luna = payload["provenance"]["arms"]["gpt-5.6-luna"]
    for slot in SLOTS:
        assert luna[slot]["stack"]["kind"] == "provider" and luna[slot]["model_type"] == "openai_batch"
        assert luna[slot]["stack"]["dates"] == {"first_request_block": "2026-09-12T05:05:19+00:00",
                                                "last_request_block": "2026-09-12T06:00:00+00:00"}
    gemma = payload["provenance"]["arms"]["gemma4-31b"]
    assert gemma[NAIVE_SLOT]["stack"]["short"] == OLD_SHORT and gemma[ENCOURAGED]["stack"]["short"] == NEW_SHORT
    assert gemma[NAIVE_SLOT]["stack"]["description"] == "vllm 0.19.0 / torch 2.10.0"


def _without_declaration(model):
    def change(payload):
        payload["judges"][model].pop("cross_stack")
    return change


def test_an_undeclared_cross_stack_contrast_is_refused(study, tmp_path):
    base, path = _rebuild(study, tmp_path, map_change=_without_declaration(GEMMA))
    with pytest.raises(ta.ArmMapError, match=r"gemma4-31b \(gemma-4-31B-it\): naive vs encouraged would compare "
                                             r"judgments from different inference stacks .*vllm 0\.19\.0"):
        _build(base, path)


def _runtime(key, condition_id, runtime, draws=range(S)):
    def change(base):
        for s in draws:
            path = base / "sampling" / ROOTS[key][0] / condition_id / f"draw-{s:03d}" / "draw_provenance.json"
            path.write_text(json.dumps({"runtime": runtime}))
    return change


def test_encouraged_against_forced_across_stacks_is_refused_even_where_naive_is_declared(study, tmp_path):
    change = _runtime("gemma", f"{GEMMA}/forced", {**NEW, "vllm_version": "0.29.0+cu129"})
    base, path = _rebuild(study, tmp_path, change=change)
    with pytest.raises(ta.ArmMapError, match=r"encouraged vs forced would compare judgments from different inference "
                                             r"stacks .*0\.29\.0\+cu129.*only the naive arm"):
        _build(base, path)


def test_an_arm_whose_own_draws_mix_stacks_is_refused(study, tmp_path):
    change = _runtime("open", f"{OLMO}/encouraged", {**NEW, "vllm_version": "0.29.0"}, draws=[1])
    base, path = _rebuild(study, tmp_path, change=change)
    with pytest.raises(cr.CampaignRootError, match=f"{OLMO}/encouraged's own judgments come from several stacks"):
        _build(base, path)


@pytest.mark.parametrize(
    "map_change, change, message",
    [
        (lambda p: p["judges"][LUNA].update(cross_stack="naive"), None,
         "gpt-5.6-luna .*remove the stale declaration"),
        (None, _runtime("published", f"{GEMMA}/naive-rules-above", NEW),
         "gemma4-31b .*shares its other arms' stack .*remove the stale declaration"),
    ],
    ids=["a provider judge", "one stack"],
)
def test_a_declaration_that_excuses_nothing_is_refused_as_stale(study, tmp_path, map_change, change, message):
    base, path = _rebuild(study, tmp_path, change=change, map_change=map_change)
    with pytest.raises(ta.ArmMapError, match=message):
        _build(base, path)


@pytest.mark.parametrize(
    "naive_runtime, revised_runtime, field",
    [
        ({**OLD, "engine_args": {"max_model_len": 4096}}, {**NEW, "engine_args": {}}, "max_model_len"),
        ({**NEW, "engine_args": {"dtype": "float16"}}, {**NEW, "engine_args": {"dtype": "bfloat16"}}, "dtype"),
    ],
    ids=["the stack and the context window", "a serving dtype on one stack"],
)
def test_a_declared_naive_contrast_that_differs_in_a_serving_setting_is_refused(study, tmp_path, naive_runtime,
                                                                                 revised_runtime, field):
    def change(base):
        _runtime("published", f"{OLMO}/naive-rules-above", naive_runtime)(base)
        for style in REVISED_STYLES:
            _runtime("open", f"{OLMO}/{style}", revised_runtime)(base)
    base, path = _rebuild(study, tmp_path, change=change)
    with pytest.raises(ta.ArmMapError, match=rf"Olmo3-32B \(Olmo-3-32B-Think\): naive vs encouraged differs in "
                                             rf"\['engine_args\.{field}'\] beyond the inference stack's packages"):
        _build(base, path)


def test_a_local_arm_whose_draws_record_no_stack_is_refused(study, tmp_path):
    def change(base):
        for s in range(S):
            draw = base / "sampling" / ROOTS["open"][0] / f"{OSS}/encouraged" / f"draw-{s:03d}"
            (draw / "draw_provenance.json").unlink()
    base, path = _rebuild(study, tmp_path, change=change)
    with pytest.raises(cr.CampaignRootError, match=f"{OSS}/encouraged's draws record no inference stack"):
        _build(base, path)


def test_conditions_the_map_does_not_name_are_listed_never_rows(study, caplog):
    payload = study["payload"]
    published = f"artifact://sampling/{ROOTS['published'][0]}"
    assert payload["provenance"]["ignored_conditions"] == {
        published: sorted(f"{m}/forced-grpo" for m in (OLMO, GEMMA, OSS)),
        f"artifact://sampling/{ROOTS['published_luna'][0]}": [f"{LUNA}/forced-grpo"],
        f"artifact://sampling/{ROOTS['gemma'][0]}": [f"{GEMMA}/naive"],
    }
    assert f"{GEMMA}/naive" not in payload["diagnostics"]["conditions"]
    assert list(payload) == ["schema_version", "table", "rates", "bars", "diagnostics", "provenance"]
    with artifact_root(study["base"]), caplog.at_level("INFO"):
        ta.build_payload(study["map"])
    assert "conditions the arm map does not name, so no row" in caplog.text


def _retire(base):
    """The Luna root as analysed before its third style was retired: that style's condition file, registered under its
    recorded key with its recorded fingerprint, and its estimates row."""
    root = base / "sampling" / ROOTS["published_luna"][0]
    fields = {**json.loads((root / LUNA / "forced-grpo" / "condition.json").read_text()),
              "condition_id": f"{LUNA}/good", "prompt_style": "good"}
    (root / LUNA / "good").mkdir()
    (root / LUNA / "good" / "condition.json").write_text(json.dumps({**fields, "fingerprint": recorded_fingerprint(fields)}))
    estimates = root / "aggregate" / "estimates.json"
    analysed = json.loads(estimates.read_text())
    analysed["conditions"][f"{LUNA}/good"] = {**analysed["conditions"][f"{LUNA}/forced-grpo"], "prompt_style": "good"}
    estimates.write_text(json.dumps(analysed))


def _files(root):
    return {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _edit_json(path, **fields):
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def test_a_condition_of_a_retired_style_is_listed_by_its_recorded_id_and_never_read(study, tmp_path):
    """A root analysed before its style was retired: its condition file and its estimates row stay, and no number moves."""
    base, path = _rebuild(study, tmp_path, change=_retire)
    payload = _build(base, path)
    luna = f"artifact://sampling/{ROOTS['published_luna'][0]}"
    assert payload["provenance"]["ignored_conditions"][luna] == [f"{LUNA}/forced-grpo", f"{LUNA}/good"]
    assert all(payload[part] == study["payload"][part] for part in ("table", "rates", "bars", "diagnostics"))


def test_a_root_holding_a_retired_style_s_condition_is_never_analysed_or_gated_again(study, tmp_path):
    """Without that condition, an analysis would change the bootstrap and Holm families its estimates record."""
    base, _ = _rebuild(study, tmp_path, change=_retire)
    root = base / "sampling" / ROOTS["published_luna"][0]
    before = _files(root)
    message = r"registers a condition of a retired style \('good'\).*change every family its analysis records"
    with artifact_root(base):
        for step in (lambda: analyze_campaign(root, n_permutations=PERMUTATIONS), lambda: campaign_completion(root),
                     lambda: load_conditions(root)):
            with pytest.raises(SamplingDesignError, match=message):
                step()
    assert {path: data for path, data in _files(root).items() if path in before} == before
    assert sorted(load_conditions(root, skip_retired=True)) == [f"{LUNA}/forced-grpo", f"{LUNA}/naive-rules-above"]


@pytest.mark.parametrize("corrupt, message", [
    (lambda root: _edit_json(root / LUNA / "forced-grpo" / "condition.json", prompt_style="good"),
     "recorded fingerprint '[0-9a-f]{64}' does not match the condition it describes"),
    (lambda root: (root / LUNA / "good").rename(root / LUNA / "elsewhere"),
     f"registers '{LUNA}/good' outside its own directory"),
    (lambda root: _edit_json(root / LUNA / "good" / "condition.json", condition_id=f"{LUNA}/../{LUNA}/good"),
     "condition_id must be <segment>/<segment>"),
], ids=["an active registration's style set to a retired key", "a retired condition outside its directory",
        "a retired condition's id with a parent segment"])
def test_a_registration_that_is_not_its_retired_condition_s_own_is_refused_never_skipped(study, tmp_path, corrupt,
                                                                                       message):
    from tests.identity.generate_fixtures import _conditions

    base, _ = _rebuild(study, tmp_path, change=_retire)
    root = base / "sampling" / ROOTS["published_luna"][0]
    corrupt(root)
    for read in (lambda: load_conditions(root, skip_retired=True), lambda: _conditions(base, [root.name])):
        with pytest.raises(SamplingDesignError, match=message):
            read()


def test_the_provenance_names_the_map_each_arm_and_each_root(study):
    provenance = study["payload"]["provenance"]
    assert "family" not in provenance and (provenance["complete"], provenance["schema_version"]) == (True, 2)
    assert provenance["arm_map"]["name"] == "arms.yaml" and provenance["arm_map"]["version"] == 1
    assert provenance["label_rules"] == {label: "lenient_markup" for label in LABELS.values()}
    assert provenance["models"] == [LABELS[m] for m in (OLMO, GEMMA, OSS, LUNA)]
    assert provenance["model_labels"] == LABELS
    campaigns = {c["root"]: c for c in provenance["campaigns"]}
    assert campaigns[f"artifact://sampling/{ROOTS['open'][0]}"]["conditions"] == sorted(
        f"{m}/{s}" for m in (OLMO, OSS) for s in REVISED_STYLES)
    assert all(c["estimates"].startswith("artifact://sampling/") for c in provenance["campaigns"])
    assert set(provenance["label_replay"].values()) == {None}
    assert provenance["n_scenarios"] == N and provenance["randomization_test"]["n_permutations"] == PERMUTATIONS
    assert provenance["definition"] == ta.DEFINITION and "_path" not in json.dumps(provenance)


def test_a_derived_root_s_replay_record_is_named(study, tmp_path):
    def change(base):
        (base / "sampling" / ROOTS["published"][0] / "label_replay.json").write_text(json.dumps({
            "tool": "smrc.sampling.label_replay", "code_state_sha256": "c" * 64, "source_label_rule": "lenient",
            "label_rule": "lenient_markup"}))
    base, path = _rebuild(study, tmp_path, change=change)
    record = _build(base, path)["provenance"]["label_replay"][f"artifact://sampling/{ROOTS['published'][0]}"]
    assert (record["name"], record["source_label_rule"], record["label_rule"]) == (
        "label_replay.json", "lenient", "lenient_markup")


def _drop(model, slot):
    def change(payload):
        payload["judges"][model].pop(slot)
    return change


@pytest.mark.parametrize(
    "map_change, message",
    [
        (_drop(OSS, "forced"), r"judge gpt-oss-120b: no \['forced'\]"),
        (lambda p: p["judges"][OSS].update(forced=p["judges"][OSS]["encouraged"]),
         r"named more than once: an arm is one slot of one judge"),
        (lambda p: p["judges"][OLMO].update(forced={"root": "open", "condition": f"{OSS}/forced"}),
         r"named more than once"),
        (lambda p: p["judges"][OLMO].update(encouraged={"root": "open", "condition": f"{OSS}/encouraged"}),
         r"named more than once"),
        (lambda p: p.update(extra=1), r"unknown key\(s\) \['extra'\]"),
        (lambda p: p["judges"][OLMO].update(notes="x"), r"judge Olmo-3-32B-Think: unknown key\(s\) \['notes'\]"),
        (lambda p: p["judges"][OLMO]["naive"].update(style="naive"), r"naive: unknown key\(s\) \['style'\]"),
        (lambda p: p["judges"][OLMO]["naive"].update(root="elsewhere"), "root 'elsewhere' is not one of the map's"),
        (lambda p: p["judges"][OLMO]["naive"].update(condition=""), "condition '' is not a condition id"),
        (lambda p: p["judges"][OLMO].update(naive=["x"]), "naive: list is not a mapping"),
        (lambda p: p.update(version=2), "version 2, not 1"),
        (lambda p: p.update(version=True), "version True, not 1"),
        (lambda p: p.update(family="paper"), r"unknown key\(s\) \['family'\]"),
        (lambda p: p.update(roots={}), "roots is not a non-empty mapping"),
        (lambda p: p.update(judges=[]), "judges is not a non-empty mapping"),
        (lambda p: p["roots"].update(published="/abs/path"), "root 'published': '/abs/path' is not an artifact:// URI"),
        (lambda p: p["roots"].update(published=["artifact://sampling/a"]),
         r"root 'published': \['artifact://sampling/a'\] is not an artifact:// URI"),
        (lambda p: p["roots"].update(spare=p["roots"]["published"]), "two roots name one URI"),
        (lambda p: p["roots"].update(spare="artifact://sampling/spare"), r"root\(s\) \['spare'\] hold no arm"),
        (lambda p: p["judges"][OLMO].update(cross_stack="encouraged"),
         "only the naive arm's contrasts may be declared cross-stack"),
        (lambda p: p["judges"][GEMMA].update(robustness=[{"root": "gemma", "condition": f"{GEMMA}/naive"}]),
         r"judge gemma-4-31B-it: unknown key\(s\) \['robustness'\]"),
        (lambda p: p.pop("cross_stack_reason"), "declare cross_stack, so the map must state cross_stack_reason"),
        (lambda p: p.update(cross_stack_reason="  "), "declare cross_stack, so the map must state cross_stack_reason"),
        (lambda p: [p["judges"][m].pop("cross_stack") for m in (OLMO, GEMMA, OSS)],
         "cross_stack_reason without a judge that declares cross_stack"),
    ],
    ids=["a missing arm", "an arm in two slots", "another judge's arm", "two judges' arm", "a top-level typo",
         "a judge typo", "an arm typo", "an unknown root", "an empty condition", "an arm list", "another version",
         "a boolean version", "a family key", "no roots", "no judges", "an absolute root", "a list root",
         "a root twice", "a root with no arm", "a declaration beyond naive", "a robustness arm", "no reason",
         "a blank reason", "a reason with no declaration"],
)
def test_an_arm_map_that_names_its_arms_ambiguously_or_incompletely_is_refused(tmp_path, map_change, message):
    payload = arm_map()
    map_change(payload)
    with pytest.raises(ta.ArmMapError, match=message):
        ta.load_arm_map(write_map(tmp_path / "arms.yaml", payload))


def _insert(after, line):
    def edit(text):
        assert f"\n{after}\n" in f"\n{text}"
        return f"\n{text}".replace(f"\n{after}\n", f"\n{after}\n{line}\n", 1)[1:]
    return edit


@pytest.mark.parametrize(
    "edit, message",
    [
        (_insert("roots:", "  published: artifact://sampling/elsewhere"),
         r"line \d+: key 'published' appears twice in one mapping"),
        (_insert("judges:", f"  {GEMMA}:\n    naive: {{root: published, condition: {GEMMA}/naive-rules-above}}"),
         rf"line \d+: key '{GEMMA}' appears twice in one mapping"),
        (_insert(f"  {OLMO}:", f"    naive: {{root: open, condition: {OLMO}/encouraged}}"),
         r"line \d+: key 'naive' appears twice in one mapping"),
        (lambda text: text + "version: 1\n", r"line \d+: key 'version' appears twice in one mapping"),
        (_insert("roots:", "  2: artifact://sampling/two"), r"line \d+: key 2 is not a name"),
        (_insert("roots:", "  no: artifact://sampling/no"), r"line \d+: key False is not a name"),
    ],
    ids=["a root twice", "a judge twice", "a slot twice", "a top-level key twice", "a number", "a YAML 1.1 boolean"],
)
def test_an_arm_map_whose_yaml_repeats_a_key_or_keys_by_a_non_name_is_refused(tmp_path, edit, message):
    """PyYAML would keep the last of a repeated key silently."""
    path = write_map(tmp_path / "arms.yaml", arm_map())
    path.write_text(edit(path.read_text()))
    with pytest.raises(ta.ArmMapError, match=f"arm map arms.yaml: {message}"):
        ta.load_arm_map(path)


@pytest.mark.parametrize("text, message", [("[1, 2]", "list is not a mapping"), (": :\n  - [", "cannot read it")],
                         ids=["a list", "not YAML"])
def test_an_arm_map_that_is_not_a_mapping_of_yaml_is_refused(tmp_path, text, message):
    path = tmp_path / "arms.yaml"
    path.write_text(text)
    with pytest.raises(ta.ArmMapError, match=message):
        ta.load_arm_map(path)
    with pytest.raises(ta.ArmMapError, match="cannot read it"):
        ta.load_arm_map(tmp_path / "missing.yaml")


def _arm(model, slot, root, condition):
    def change(payload):
        payload["judges"][model][slot] = {"root": root, "condition": condition}
    return change


@pytest.mark.parametrize(
    "map_change, message",
    [
        (_arm(OLMO, "forced", "open", f"{OLMO}/forced-grpo"),
         f"root open: no registered condition has the public id '{OLMO}/forced-grpo'"),
        (_arm(OLMO, "encouraged", "published", f"{OLMO}/forced-grpo"),
         "Olmo-3-32B-Think/forced-grpo's style 'forced-grpo' is not its encouraged slot's 'encouraged'"),
        (_arm(OLMO, "naive", "published", f"{OLMO}/forced-grpo"), "style 'forced-grpo' is not its naive slot's 'naive-rules-above'"),
        (_arm(OLMO, "naive", "published", f"{GEMMA}/forced-grpo"),
         f"{GEMMA}/forced-grpo is {GEMMA}'s, not {OLMO}'s"),
    ],
    ids=["a condition the root does not register", "a GRPO style in the encouraged slot", "a forced style as naive",
         "another judge's condition"],
)
def test_an_arm_the_roots_do_not_hold_as_named_is_refused(study, tmp_path, map_change, message):
    base, path = _rebuild(study, tmp_path, map_change=map_change)
    with pytest.raises((ta.ArmMapError, cr.CampaignRootError), match=message):
        _build(base, path)


def test_an_arm_its_root_registers_but_never_analysed_is_refused(study, tmp_path):
    from ruleforge.evaluate.campaign import load_conditions

    def register_late(base):
        root = base / "sampling" / ROOTS["open"][0]
        spec = load_conditions(root)[f"{OLMO}/encouraged"]
        register_condition(root, dataclasses.replace(spec, condition_id=f"{OLMO}/naive", prompt_style="naive"))

    base, path = _rebuild(study, tmp_path, change=register_late,
                          map_change=_arm(OLMO, "naive", "open", f"{OLMO}/naive"))
    with pytest.raises(cr.CampaignRootError, match=f"{OLMO}/naive is not analysed in root open's estimates"):
        _build(base, path)


@pytest.fixture(scope="module")
def recorded_study(tmp_path_factory):
    """The study's draws in roots written as the published campaigns write them, under recorded style keys."""
    base = tmp_path_factory.mktemp("recorded-arms")
    with artifact_root(base):
        roots = {key: build_study_root(base, key, recorded=True) for key in ROOTS}
        path = write_map(base / "arms.yaml", arm_map())
        payload = ta.build_payload(path)
    return {"base": base, "roots": roots, "map": path, "payload": payload}


def test_recorded_roots_and_public_ones_give_one_golden(study, recorded_study):
    recorded, public = recorded_study["payload"], study["payload"]
    names = json.loads((recorded_study["roots"]["gemma"] / "aggregate" / "estimates.json").read_text())["conditions"]
    assert sorted(names) == sorted(recorded_id(GEMMA, style) for style in ("naive", *REVISED_STYLES))
    assert not set(names) & {f"{GEMMA}/{style}" for style in ("naive", *REVISED_STYLES)}
    assert without_estimates_files(recorded) == without_estimates_files(public)
    provenance = recorded["provenance"]
    assert provenance["arms"]["gemma4-31b"][FORCED]["condition_id"] == f"{GEMMA}/forced"
    assert provenance["ignored_conditions"][f"artifact://sampling/{ROOTS['gemma'][0]}"] == [f"{GEMMA}/naive"]
    assert {c["root"]: c["conditions"] for c in provenance["campaigns"]}[
        f"artifact://sampling/{ROOTS['gemma'][0]}"] == [f"{GEMMA}/encouraged", f"{GEMMA}/forced"]


def test_an_analysed_condition_no_condition_json_registers_is_named_by_its_public_id(recorded_study, tmp_path):
    base = shutil.copytree(recorded_study["base"], tmp_path / "copy")
    root = base / "sampling" / ROOTS["gemma"][0]
    revised = recorded_id(GEMMA, "naive")
    (root / revised / "condition.json").unlink()
    _edit_estimates("gemma", lambda p: p["conditions"][revised].update(
        prompt_style=record_names.identity_key("naive")))(base)
    payload = _build(base, base / "arms.yaml")
    assert payload["provenance"]["ignored_conditions"][f"artifact://sampling/{ROOTS['gemma'][0]}"] == [
        f"{GEMMA}/naive"]


def test_a_root_registering_a_recorded_condition_and_its_public_twin_is_refused(recorded_study, tmp_path):
    from ruleforge.evaluate.campaign import condition_payload

    base = shutil.copytree(recorded_study["base"], tmp_path / "copy")
    root = base / "sampling" / ROOTS["open"][0]
    twin = dataclasses.replace(load_conditions(root)[recorded_id(OLMO, FORCED)], condition_id=f"{OLMO}/{FORCED}")
    (root / twin.condition_id).mkdir()
    (root / twin.condition_id / "condition.json").write_text(json.dumps(condition_payload(twin)))
    with pytest.raises(cr.CampaignRootError, match=f"root open: 2 registered conditions have the public id "
                                             f"'{OLMO}/{FORCED}'"):
        _build(base, base / "arms.yaml")


def test_a_judge_s_arms_under_different_label_rules_are_refused(study, tmp_path):
    designs = {f"{m}/naive-rules-above": {"label_rule": "lenient"} for m in (OLMO, GEMMA, OSS)}
    base, path = _rebuild(study, tmp_path, root_overrides={"published": {"designs": designs}})
    with pytest.raises(ta.ArmMapError, match=r"Olmo3-32B \(Olmo-3-32B-Think\): its arms were judged under different "
                                             r"label rules \(naive lenient, encouraged lenient_markup, forced "
                                             r"lenient_markup\): one rule per judge's arms"):
        _build(base, path)


def test_a_judge_s_arms_whose_identities_differ_beyond_the_style_are_refused(study, tmp_path):
    designs = {f"{OSS}/forced": {"max_new_tokens": 4096}}
    base, path = _rebuild(study, tmp_path, root_overrides={"open": {"designs": designs}})
    with pytest.raises(ta.ArmMapError, match=r"its forced arm differs from its naive arm in \['max_new_tokens'\]"):
        _build(base, path)


def test_a_judge_retried_by_component_joins_judges_retried_by_the_bundle_and_every_scope_is_named(study, tmp_path):
    assert "retry_scopes" not in study["payload"]["provenance"]
    component = {"retry_scope": "component"}
    base, path = _rebuild(study, tmp_path, root_overrides={
        "luna": {"designs": {f"{LUNA}/{style}": component for style in REVISED_STYLES}},
        "published_luna": {"designs": {f"{LUNA}/{style}": component for style in PUBLISHED_STYLES}},
    })
    for key in ("published_luna", "luna"):
        _edit_manifest(key, retry_scope="component")(base)
    payload = _build(base, path)
    provenance = payload["provenance"]
    assert provenance["retry_scopes"] == {**{LABELS[m]: "scenario_bundle" for m in (OLMO, GEMMA, OSS)},
                                          LABELS[LUNA]: "component"}
    assert provenance["label_rules"] == {label: "lenient_markup" for label in LABELS.values()}
    assert [row["label"] for row in payload["table"]] == [row["label"] for row in study["payload"]["table"]]
    campaigns = {c["campaign_id"]: c for c in provenance["campaigns"]}
    assert all("retry_scope" not in c for c in campaigns.values())


def test_a_judge_s_arms_retried_under_different_scopes_are_refused(study, tmp_path):
    base, path = _rebuild(study, tmp_path, root_overrides={
        "luna": {"designs": {f"{LUNA}/{style}": {"retry_scope": "component"} for style in REVISED_STYLES}}})
    with pytest.raises(ta.ArmMapError, match=r"its encouraged arm differs from its naive arm in \['retry_scope'\]"):
        _build(base, path)


def test_an_unknown_judge_and_two_judges_under_one_label_are_refused(study, tmp_path):
    payload = arm_map()
    payload["judges"]["mystery-model"] = payload["judges"].pop(LUNA)
    base, _ = _rebuild(study, tmp_path)
    with pytest.raises(CampaignEstimatesError, match="no golden label for judge 'mystery-model'"):
        _build(base, write_map(base / "arms.yaml", payload))
    payload["judges"]["claude-haiku-5-5"] = payload["judges"].pop("mystery-model")
    assert "claude-haiku-5-5" in ta.judges_by_label(ta.load_arm_map(write_map(base / "haiku.yaml", payload)))
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(ta.campaign_label.__globals__["CAMPAIGN_MODEL_LABELS"], LUNA, "gpt-oss-120b")
        with pytest.raises(ta.ArmMapError, match="two judges share a golden label"):
            _build(base, write_map(base / "arms.yaml", arm_map()))


def test_roots_that_do_not_resolve_or_are_not_analysed_are_refused(study, tmp_path):
    base, path = _rebuild(study, tmp_path, change=lambda b: (
        b / "sampling" / ROOTS["luna"][0] / "aggregate" / "estimates.json").unlink())
    with pytest.raises(cr.CampaignRootError, match="root luna .* has no aggregate/estimates.json: analyse it first"):
        _build(base, path)
    shutil.rmtree(base / "sampling" / ROOTS["luna"][0])
    with pytest.raises(cr.CampaignRootError, match=r"root luna .* is not in the artifact root: fetch it \(ruleforge "
                                                   r"data fetch\)"):
        _build(base, path)
    payload = arm_map()
    payload["roots"]["luna"] = "artifact://../outside"
    with pytest.raises(cr.CampaignRootError, match="a root does not resolve"):
        _build(base, write_map(base / "arms.yaml", payload))
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)
        with pytest.raises(cr.CampaignRootError, match="a root does not resolve: RULEFORGE_ARTIFACT_ROOT is unset"):
            ta.build_payload(path)


def _edit_manifest(key, **fields):
    def change(base):
        path = base / "sampling" / ROOTS[key][0] / "sampling_manifest.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))
        if "max_semantic_attempts" in fields:
            estimates = base / "sampling" / ROOTS[key][0] / "aggregate" / "estimates.json"
            estimates.write_text(json.dumps({**json.loads(estimates.read_text()),
                                             "max_semantic_attempts": fields["max_semantic_attempts"]}))
    return change


def test_a_judge_s_arms_from_roots_run_under_other_campaign_settings_are_refused(study, tmp_path):
    base, path = _rebuild(study, tmp_path, change=_edit_manifest("open", max_semantic_attempts=3,
                                                                  scenarios_per_request_block=7))
    with pytest.raises(cr.CampaignRootError, match=r"Olmo3-32B \(Olmo-3-32B-Think\): its arms' roots differ in "
                                             r"max_semantic_attempts \(published 20, open 3\); "
                                             r"scenarios_per_request_block \(published None, open 7\)"):
        _build(base, path)
    # Luna's roots may hold other settings than the open judges' (a Batch API block, say): only a judge's own count.
    _edit_manifest("open", max_semantic_attempts=20, scenarios_per_request_block=None)(base)
    for key in ("published_luna", "luna"):
        _edit_manifest(key, scenarios_per_request_block=500)(base)
    assert _build(base, path)["table"] == study["payload"]["table"]


def test_a_root_that_is_no_final_campaign_or_names_another_campaign_is_refused(study, tmp_path):
    base, path = _rebuild(study, tmp_path, change=_edit_manifest("luna", campaign_mode="screening"))
    with pytest.raises(cr.CampaignRootError, match="root luna: campaign_mode 'screening'; the paper's goldens need a final "
                                             "campaign"):
        _build(base, path)
    with artifact_root(base):
        scratch = ta.build_payload(path, allow_incomplete=True)
    assert scratch["table"] == study["payload"]["table"] and scratch["provenance"]["complete"] is False
    _edit_manifest("luna", campaign_mode="final", campaign_id="elsewhere")(base)
    with pytest.raises(cr.CampaignRootError, match="root luna: its sampling manifest names another campaign"):
        _build(base, path)


def test_a_build_at_another_alpha_reads_every_significance_at_it(study):
    alpha = 0.5
    with artifact_root(study["base"]):
        payload = ta.build_payload(study["map"], alpha=alpha)
    assert payload["provenance"]["alpha"] == alpha
    for row in payload["table"]:
        assert row["significant"] is row["significant_holm"] is (row["p_holm"] < alpha)
        assert row["adjustment_method"] == "holm"
        assert row["significant_bh"] is (row["p_bh"] < alpha)


def test_roots_that_are_not_one_experiment_are_refused(study, tmp_path):
    def change(base):
        path = base / "sampling" / ROOTS["luna"][0] / "aggregate" / "estimates.json"
        payload = json.loads(path.read_text())
        payload["bootstrap"]["n_resamples"] = 300
        path.write_text(json.dumps(payload))
    base, path = _rebuild(study, tmp_path, change=change)
    with pytest.raises(CampaignEstimatesError, match="bootstrap .* differs .* not one experiment"):
        _build(base, path)


def _edit_estimates(key, edit):
    def change(base):
        path = base / "sampling" / ROOTS[key][0] / "aggregate" / "estimates.json"
        payload = json.loads(path.read_text())
        edit(payload)
        path.write_text(json.dumps(payload))
    return change


@pytest.mark.parametrize(
    "change, message",
    [
        (_edit_estimates("open", lambda p: p["conditions"][f"{OLMO}/encouraged"].update(grid_digest="0" * 64)),
         f"{OLMO}/encouraged: its grid rebuilt from the draws is not its estimates' own"),
        (_edit_estimates("open", lambda p: p["conditions"][f"{OLMO}/encouraged"].update(condition_fingerprint="f")),
         f"{OLMO}/encouraged: its draws, its condition.json and its estimates name different fingerprints"),
        (_edit_estimates("open", lambda p: next(iter(p["style_families"].values()))["contrasts"][0].update(
            p_value=0.5)), r"the recomputed p_value .* is not the aggregate's 0.5"),
        (_edit_estimates("luna", lambda p: p["forced_conditions"].clear()),
         f"{LUNA}/forced has no same-call rate in root luna's estimates"),
        (_edit_estimates("luna", lambda p: p["conditions"][f"{LUNA}/encouraged"].update(interval_basis="other")),
         r"the arms carry 2 interval bases"),
        (lambda base: (base / "sampling" / ROOTS["open"][0] / f"{OSS}/encouraged" / "condition.json").unlink(),
         f"root open: no registered condition has the public id '{OSS}/encouraged'"),
        (lambda base: [_edit_estimates(key, lambda p: p.update(scenario_manifest_sha256="x" * 64))(base)
                       for key in ROOTS],
         "root published: its scenario manifest is not the one its estimates name"),
    ],
    ids=["a grid digest", "a fingerprint", "an aggregate row", "no same-call rate", "two interval bases",
         "an unregistered arm", "another scenario manifest"],
)
def test_estimates_that_do_not_describe_the_draws_beside_them_are_refused(study, tmp_path, change, message):
    base, path = _rebuild(study, tmp_path, change=change)
    with pytest.raises((ta.ArmMapError, CampaignEstimatesError, ValueError), match=message):
        _build(base, path)


def test_validate_cross_stack_reads_only_declared_naive_records():
    ok = {"label": "x", "baseline": NAIVE_SLOT, "treatment": ENCOURAGED,
          "cross_stack": {"declared": "naive", "baseline_stack": "a", "treatment_stack": "b", "fields": ["vllm"],
                          "conflict": "c"}}
    ta.validate_cross_stack([ok, {"label": "x", "baseline": ENCOURAGED, "treatment": FORCED, "cross_stack": None}, {}])
    with pytest.raises(ta.ArmMapError, match="x encouraged->forced: cross_stack on a row whose baseline is not"):
        ta.validate_cross_stack([{**ok, "baseline": ENCOURAGED, "treatment": FORCED}])
    with pytest.raises(ta.ArmMapError, match="its cross_stack record is not a declared exception"):
        ta.validate_cross_stack([{**ok, "cross_stack": {"declared": "naive"}}])
    with pytest.raises(ta.ArmMapError, match="its cross_stack record is not a declared exception"):
        ta.validate_cross_stack([{**ok, "cross_stack": "yes"}])
    for fields in ([], ["engine_args.dtype"], None):
        with pytest.raises(ta.ArmMapError, match="the stack packages they differ in"):
            ta.validate_cross_stack([{**ok, "cross_stack": {**ok["cross_stack"], "fields": fields}}])


def test_a_golden_whose_flags_disagree_with_its_own_provenance_is_refused(study, driver, tmp_path, capsys):
    payload = json.loads(json.dumps(study["payload"]))
    table, provenance = payload["table"], payload["provenance"]
    ta.validate_cross_stack(table, provenance)
    ta.validate_cross_stack([{**row, "cross_stack": None} for row in table], {})
    dropped = [dict(row) for row in table]
    dropped[0]["cross_stack"] = None
    for rows, record, message in (
        (dropped, provenance, "are not the ones provenance.cross_stack.contrasts lists"),
        (table, {}, "are not the ones provenance.cross_stack.contrasts lists"),
        (table, {"cross_stack": {**provenance["cross_stack"], "declared": ["gemma4-31b"]}},
         "a flagged row's judge is not one provenance.cross_stack.declared lists"),
    ):
        with pytest.raises(ta.ArmMapError, match=message):
            ta.validate_cross_stack(rows, record)
    payload["table"] = dropped
    path = tmp_path / "dropped.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(path), "--csv", str(tmp_path / "t.csv")])
    assert "provenance.cross_stack.contrasts" in capsys.readouterr().err and not (tmp_path / "t.csv").exists()


def test_short_stacks_and_request_dates_read_what_is_recorded(tmp_path):
    assert cr.short_stack([{"vllm": None}, {"vllm": "0.28.0", "torch": "2.13.0"}]) == "vLLM 0.28.0, torch 2.13.0"
    assert cr.short_stack([]) == "no stack recorded"
    assert cr.short_stack([{"vllm": "0.28.0", "tokenizers": "0.22.1"}], ["tokenizers", "vllm"]) == (
        "vLLM 0.28.0, tokenizers 0.22.1")
    (tmp_path / "draw-000").mkdir()
    assert cr._request_dates(tmp_path) == {"first_request_block": None, "last_request_block": None}
    (tmp_path / "draw-000" / "request_blocks.jsonl").write_text(json.dumps({"recorded_at": None}) + "\n")
    assert cr._request_dates(tmp_path)["first_request_block"] is None
    # A rebased draw: one request block for every scenario.
    (tmp_path / "draw-000" / "request_blocks.jsonl").write_text(
        json.dumps({"recorded_at": "2026-09-12T05:05:19+00:00"}) + "\n")
    (tmp_path / "draw-000" / "attempts.jsonl").write_text("".join(json.dumps({"started_at": stamp}) + "\n" for stamp in (
        "2026-09-12T05:05:19+00:00", "2026-10-28T02:28:11+00:00", None)))
    assert cr._request_dates(tmp_path) == {"first_request_block": "2026-09-12T05:05:19+00:00",
                                           "last_request_block": "2026-10-28T02:28:11+00:00"}


def test_main_builds_from_the_arm_map_and_flags_the_cross_stack_rows(study, driver, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = {name: tmp_path / name for name in ("t.json", "t.csv", "t.tex", "a.png", "b.png")}
    with artifact_root(study["base"]):
        assert driver.main(["--arm-map", str(study["map"]), "--golden", str(out["t.json"]), "--csv", str(out["t.csv"]),
                            "--latex", str(out["t.tex"]), "--figure_styles", str(out["a.png"]),
                            "--figure_methods", str(out["b.png"]), "--dpi", "40"]) == 0
    printed = capsys.readouterr().out
    golden = json.loads(out["t.json"].read_text())
    assert golden == json.loads(json.dumps(study["payload"]))
    assert printed.count("(cross-stack)") == 6 and "robustness" not in printed
    header, *rows = out["t.csv"].read_text().splitlines()
    columns = header.split(",")
    assert columns[-3:] == ["style_a", "style_b", "cross_stack"] and len(rows) == 12
    flags = [(row.split(",")[0], row.split(",")[1], row.split(",")[-1]) for row in rows]
    assert flags[:3] == [("Olmo3-32B", "naive", "true"), ("Olmo3-32B", "naive", "true"),
                         ("Olmo3-32B", "encouraged", "false")]
    assert all(flag == "false" for label, _, flag in flags if label == "gpt-5.6-luna")
    assert rows[0].split(",")[-3:-1] == ["naive", "encouraged"]
    latex = out["t.tex"].read_text()
    assert latex.count(r"$^{\dagger}$") == 7  # six rows and the note
    # Plain names: the rows say naive / encouraged / forced and the note lists no prompts, no version label.
    assert "(v2)" not in latex and "Prompts:" not in latex
    assert r"naive $\rightarrow$ encouraged$^{\dagger}$" in latex
    assert ("The two arms were judged on different inference stacks (vLLM 0.19.0, torch 2.10.0 against vLLM "
            "0.28.0+cu129, torch 2.13.0+cu129, transformers 5.17.0)") in latex
    for path in ("a.png", "b.png"):
        assert out[path].stat().st_size > 0
    again = {name: tmp_path / f"again.{name}" for name in ("csv", "tex")}
    assert driver.main(["--from_golden", str(out["t.json"]), "--csv", str(again["csv"]),
                        "--latex", str(again["tex"])]) == 0
    assert again["csv"].read_text() == out["t.csv"].read_text()
    assert again["tex"].read_text() == latex


def test_a_table_with_no_cross_stack_row_carries_no_cross_stack_note(study, driver):
    assert driver._cross_stack_note([dict(row, cross_stack=None) for row in study["payload"]["table"]]) == ""


def test_main_refuses_mixed_inputs_a_refusal_and_a_tampered_cross_stack_flag(study, driver, tmp_path, capsys):
    with pytest.raises(SystemExit):
        driver.main(["--arm-map", str(study["map"]), "--estimates", "x.json"])
    assert "--arm-map builds from the roots the arm map names" in capsys.readouterr().err
    bad = write_map(tmp_path / "bad.yaml", {**arm_map(), "version": 9})
    with artifact_root(study["base"]), pytest.raises(SystemExit):
        driver.main(["--arm-map", str(bad)])
    assert "version 9, not 1" in capsys.readouterr().err
    golden = json.loads(json.dumps(study["payload"]))
    golden["table"][2]["cross_stack"] = golden["table"][0]["cross_stack"]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(golden))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(path)])
    assert "cross_stack on a row whose baseline is not the naive slot" in capsys.readouterr().err


def test_the_paper_s_files_take_an_arm_map_s_table_only(study, driver, tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    monkeypatch.setattr(driver, "repository_root", lambda: repo)
    golden = tmp_path / "arms.json"
    golden.write_text(json.dumps(study["payload"]))
    none = json.loads(json.dumps(study["payload"]))
    del none["provenance"]["arm_map"]
    no_map = tmp_path / "none.json"
    no_map.write_text(json.dumps(none))
    paper_csv = repo / "results" / "prompt_ablation" / "prompt_ablation_table.csv"
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(no_map), "--csv", str(paper_csv)])
    assert "hold(s) the paper's files; this output is not one of them" in capsys.readouterr().err
    assert not paper_csv.exists()
    elsewhere = repo / "scratch" / "table.json"
    assert driver.main(["--from_golden", str(no_map), "--golden", str(elsewhere)]) == 0
    assert json.loads(elsewhere.read_text()) == none
    assert driver.main(["--from_golden", str(golden), "--csv", str(paper_csv)]) == 0 and paper_csv.exists()


def test_the_paper_s_files_take_only_the_slots_template_arms(study, driver, repo_root, tmp_path, monkeypatch,
                                                                capsys):
    repo = tmp_path / "repo"
    monkeypatch.setattr(driver, "repository_root", lambda: repo)
    paper_csv = repo / "results" / "prompt_ablation" / "prompt_ablation_table.csv"
    assert ta.paper_table_refusal(json.loads((repo_root / "results/prompt_ablation/prompt_ablation.json").read_text())) is None
    forced_grpo = json.loads(json.dumps(study["payload"]))
    for row in forced_grpo["table"]:
        if row["label"] == "gemma4-31b" and row["treatment"] == FORCED:
            row["treatment_style"] = "forced-grpo"
    path = tmp_path / "forced_grpo.json"
    path.write_text(json.dumps(forced_grpo))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(path), "--csv", str(paper_csv)])
    assert "arms ['gemma4-31b forced forced-grpo'] are not their slots' template styles" in capsys.readouterr().err
    assert not paper_csv.exists()


def test_the_paper_s_directories_refuse_anything_but_the_paper_s_outputs(tmp_path):
    paper = tmp_path / "results" / "prompt_ablation" / "x.json"
    elsewhere = tmp_path / "scratch" / "x.json"
    assert pf.paper_output_refusal([paper, None], True, repo_root=tmp_path) is None
    assert pf.paper_output_refusal([elsewhere, None], False, repo_root=tmp_path) is None
    refusal = pf.paper_output_refusal([elsewhere, paper, tmp_path / "results" / "a.png"], False, repo_root=tmp_path)
    assert "a.png" in refusal and "x.json" in refusal and str(elsewhere) not in refusal
    assert pf.paper_paths([tmp_path / "figures" / "a.png"], repo_root=tmp_path) == []
    assert "the paper's files; this output is not one of them" in refusal
    assert pf.paper_paths([tmp_path / "results" / "leniency" / "leniency.json"], repo_root=tmp_path) != []
    assert pf.paper_paths([elsewhere], repo_root=tmp_path) == []
    assert pf.SLOT_NAMES == {"naive": "naive", "encouraged": "encouraged", "forced": "forced"}


def test_the_leniency_golden_joins_the_three_revised_roots_one_forced_arm_per_judge(study, repo_root, tmp_path,
                                                                                    monkeypatch, capsys):
    pytest.importorskip("matplotlib")
    from ruleforge.experiments.leniency import analysis as rc

    estimates = [study["roots"][key] / "aggregate" / "estimates.json" for key in ("gemma", "open", "luna")]
    with artifact_root(study["base"]):
        golden = rc.build_payload(estimates, study["base"] / "rules.jsonl", prompt_style="forced")
    assert list(golden["models"]) == [LABELS[m] for m in (OLMO, GEMMA, OSS, LUNA)]
    assert {block["condition_id"] for block in golden["models"].values()} == {
        f"{m}/forced" for m in (OLMO, GEMMA, OSS, LUNA)}
    assert golden["settings"]["label_rule"] == "lenient_markup" and golden["settings"]["prompt_style"] == "forced"
    assert all(block["named_preempted"] is None for block in golden["models"].values())
    module = importlib.import_module("ruleforge.experiments.leniency.build")
    argv = [arg for path in estimates for arg in ("--estimates", str(path))]
    common = [*argv, "--rules", str(study["base"] / "rules.jsonl"), "--prompt-style", "forced"]
    repo = tmp_path / "repo"
    monkeypatch.setattr(module, "repository_root", lambda: repo)
    paper_csv = repo / "results" / "leniency" / "leniency.csv"
    with artifact_root(study["base"]):
        assert module.main([*common, "--csv", str(paper_csv)]) == 0 and paper_csv.is_file()
        capsys.readouterr()
        assert module.main([*common, "--golden", str(tmp_path / "rc.json"), "--csv", str(tmp_path / "rc.csv"),
                            "--figure", str(tmp_path / "a.png"), "--figure_all", str(tmp_path / "b.png"),
                            "--figure_flips", str(tmp_path / "c.png"), "--dpi", "40"]) == 0
    assert json.loads((tmp_path / "rc.json").read_text()) == json.loads(json.dumps(golden))
    assert "prompt style: forced prompt (forced)" in capsys.readouterr().out
    assert all((tmp_path / name).stat().st_size > 0 for name in ("rc.csv", "a.png", "b.png", "c.png"))


def test_the_leniency_golden_refuses_a_revised_root_under_another_label_rule(study, tmp_path):
    from ruleforge.experiments.leniency import analysis as rc
    from ruleforge.experiments.leniency.schema import RuleContextError

    base, _ = _rebuild(study, tmp_path, root_overrides={
        "luna": {"designs": {f"{LUNA}/forced": {"label_rule": "lenient"}}}})
    estimates = [base / "sampling" / ROOTS[key][0] / "aggregate" / "estimates.json"
                 for key in ("gemma", "open", "luna")]
    with artifact_root(base), pytest.raises(RuleContextError, match="label_rule 'lenient' vs 'lenient_markup'"):
        rc.build_payload(estimates, base / "rules.jsonl", prompt_style="forced")


SAMPLING = "artifact://sampling"
GEMMA_REVISED_ROOT = f"{SAMPLING}/prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line"
PUBLISHED_RESCORED = f"{SAMPLING}/final-2026-09-pop-a22af097-rescored-lenient_last_line"
SHARED_ROW_KEYS = ("baseline_rate", "treatment_rate", "difference", "ci_low", "ci_high", "p_value", "n_permutations",
                   "baseline_condition", "treatment_condition", "baseline_root", "treatment_root",
                   "paired_within_root", "cross_stack")


def _real_root(uri):
    from ruleforge.core.paths import resolve

    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    path = resolve(uri)
    if not (path / "aggregate" / "estimates.json").is_file():
        pytest.skip(f"{uri} is not on this machine (or not analysed yet)")
    return path


def _gemma_map(tmp_path, naive_root, **extra):
    """A one-judge map: Gemma's published naive arm in ``naive_root``, its other arms in its prompt-ablation root."""
    judge = {"naive": {"root": "naive", "condition": f"{GEMMA}/naive-rules-above"},
             "encouraged": {"root": "gemma", "condition": f"{GEMMA}/encouraged"},
             "forced": {"root": "gemma", "condition": f"{GEMMA}/forced"}, **extra}
    payload = {"version": 1, "roots": {"naive": naive_root, "gemma": GEMMA_REVISED_ROOT},
               "judges": {GEMMA: judge}}
    if "cross_stack" in extra:
        payload["cross_stack_reason"] = "the published naive arms ran on another stack"
    return write_map(tmp_path / "gemma.yaml", payload)


@pytest.mark.cluster
def test_gemma_s_re_scored_published_naive_crosses_stacks_only_as_declared_on_the_real_roots(tmp_path, repo_root):
    _real_root(GEMMA_REVISED_ROOT)
    _real_root(PUBLISHED_RESCORED)
    with pytest.raises(ta.ArmMapError, match="naive vs encouraged would compare judgments from different inference"):
        ta.build_payload(_gemma_map(tmp_path, PUBLISHED_RESCORED))
    payload = ta.build_payload(_gemma_map(tmp_path, PUBLISHED_RESCORED, cross_stack="naive"))
    golden = json.loads((repo_root / "results/prompt_ablation/prompt_ablation.json").read_text(encoding="utf-8"))
    label = LABELS[GEMMA]
    paper = {(r["baseline"], r["treatment"]): r for r in golden["table"] if r["label"] == label}
    rows = {(r["baseline"], r["treatment"]): r for r in payload["table"]}
    assert list(rows) == list(paper) == list(TABLE_CONTRASTS)
    for key, row in rows.items():
        assert {k: row[k] for k in SHARED_ROW_KEYS} == {k: paper[key][k] for k in SHARED_ROW_KEYS}, key
    assert [rows[key]["cross_stack"] is not None for key in TABLE_CONTRASTS] == [True, True, False]
    assert rows[(NAIVE_SLOT, ENCOURAGED)]["cross_stack"]["baseline_stack"].startswith("vLLM 0.19.0")
    assert payload["bars"] == {label: golden["bars"][label]}
    assert payload["provenance"]["arms"] == {label: golden["provenance"]["arms"][label]}
    assert payload["provenance"]["label_rules"] == {label: "lenient_last_line"}
