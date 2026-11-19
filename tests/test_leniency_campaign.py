from __future__ import annotations

import contextlib
import copy
import importlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest

from ruleforge.experiments.leniency import analysis as rc
from ruleforge.experiments.leniency import golden as lg
from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.evaluate.inference import holm_adjust
from ruleforge.experiments.leniency import schema as ls
from ruleforge.experiments.leniency.schema import LABELS, RuleContextError
from ruleforge.core.digests import sha256_file
from ruleforge.core.paths import absolute_strings
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.core import record_names
from ruleforge.evaluate.campaign import CONDITION_FNAME, condition_payload, register_condition
from ruleforge.evaluate.identity import ConditionSpec
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.ledger import AttemptRecord, ComponentRecord
from ruleforge.prompts.styles import FORCED_JSON_STYLES

SCRIPT = "scripts/prompt_ablation/build_rule_context_campaign.py"
N_RULES, N, S = 11, 16, 3
RULES = [{"id": f"Rule {n}", "rule": f"Title of rule {n}" + (" with a long tail" * 4 if n == 3 else "")}
         for n in range(1, N_RULES + 1)]
BOOTSTRAP = {"n_resamples": 200, "ci_level": 0.95, "seed": 16}
PERMUTATIONS = 4000
#: Raw preemption ids a chatty judge writes: two resolve, two do not.
PREEMPTED = ["rule 2.", "Rule 3", "Meta-Rule 1", "Rule 99"]


def _judgments(seed, *, rule_10_rises, unsure_verdicts, preempt):
    """``{(i, s): (alone, in_context, verdict, preempted)}`` with known structure."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(N):
        for s in range(S):
            alone, in_context = {}, {}
            for n in range(1, N_RULES + 1):
                if n == 1:
                    a, c = ("yes" if rng.random() < 0.8 else "no"), ("yes" if rng.random() < 0.1 else "no")
                elif n == 2:
                    a = c = str(rng.choice(LABELS))
                elif n == 10:
                    low, high = ("yes" if rng.random() < 0.05 else "no"), ("yes" if rng.random() < 0.75 else "no")
                    a, c = (low, high) if rule_10_rises else (high, low)
                else:
                    a, c = (str(rng.choice(LABELS, p=p)) for p in ([0.2, 0.1, 0.7], [0.15, 0.1, 0.75]))
                alone[f"Rule {n}"], in_context[f"Rule {n}"] = a, c
            verdict = "no" if i % 3 == 0 else "unsure" if unsure_verdicts and i % 5 == 1 else "yes"
            out[(i, s)] = (alone, in_context, verdict, list(PREEMPTED) if preempt and i % 4 == 1 else [])
    return out


def _component(component_id, judgment, **extra):
    return ComponentRecord(
        component_id=component_id, valid=judgment is not None, failure_reason=None if judgment else "invalid_json",
        judgment=judgment, finish_reason="stop", generation_seed=7, seed_sent=True, raw_output_sha256="0" * 64,
        **extra,
    )


def recorded_id(model, style):
    """The id a published campaign records ``model``'s ``style`` condition under: ``{model}/{key}``."""
    return f"{model}/{record_names.identity_key(style)}"


def register_as_recorded(root, spec):
    """Write ``spec``'s ``condition.json`` in ``root`` by hand, as a published campaign records it."""
    payload = condition_payload(spec)
    payload["prompt_style"] = record_names.identity_key(spec.prompt_style)
    path = Path(root) / spec.condition_id / CONDITION_FNAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def without_estimates_files(golden):
    """``golden`` without what tells two roots of one campaign apart but their names."""
    golden = copy.deepcopy(golden)
    for campaign in golden["provenance"]["campaigns"]:
        del campaign["estimates_sha256"], campaign["created_at"]
    return golden


def build_root(base, name, *, model="gpt-oss-120b", condition_id=None, seed=1, rule_10_rises=True,
               unsure_verdicts=True, preempt=True, rules_path=None, bootstrap=BOOTSTRAP, mode="final",
               design=None, styles=("forced-grpo",), recorded=False):
    """Write a complete campaign root under ``base/sampling/name`` and aggregate it."""
    root = base / "sampling" / name
    for index, style in enumerate(styles):
        estimates = _write_condition(
            base, name, model=model, condition_id=condition_id, seed=seed + index,
            rule_10_rises=rule_10_rises != bool(index),
            unsure_verdicts=unsure_verdicts, preempt=preempt and rc.lists_preemption(style), rules_path=rules_path,
            bootstrap=bootstrap, mode=mode, design=design, style=style, first=index == 0, recorded=recorded,
        )
    analyze_campaign(root, n_permutations=PERMUTATIONS)
    return estimates


def _write_condition(base, name, *, model, condition_id, seed, rule_10_rises, unsure_verdicts, preempt, rules_path,
                     bootstrap, mode, design, style, first, recorded=False):
    """One forced condition's draws in ``base/sampling/name`` (the root's manifests too when ``first``)."""
    root = base / "sampling" / name
    root.mkdir(parents=True, exist_ok=not first)
    scenarios = base / "scenarios.jsonl"
    if not scenarios.exists():
        scenarios.write_text("".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) + "\n" for i in range(N)))
    manifest = ScenarioManifest.build(scenarios)
    manifest.write(root / "scenario_manifest.json")
    rules_path = rules_path or write_rules(base / "rules.jsonl")
    (root / "sampling_manifest.json").write_text(json.dumps({
        "schema_version": 2, "campaign_id": name, "campaign_mode": mode,
        "scenario_manifest_sha256": manifest.manifest_sha256, "expected_scenarios": N,
        "expected_decoder_draws": S, "temperature": 1.0, "max_semantic_attempts": 20,
        "forced_json_requirement": "complete_rule_map", "truncation": "model_default", "bootstrap": bootstrap,
    }))
    design = {"individual_rule_context": "isolated", "judge_type": "No_Interpretation",
              "evaluation_mode": "both", "prompt_shape_version": 2, **(design or {})}
    spec = ConditionSpec(
        condition_id=condition_id or (recorded_id(model, style) if recorded else f"{model}/{style}"),
        model_name=model, model_type="local",
        prompt_style=style, rule_set_name="rules.jsonl",
        rule_set_sha256=sha256_file(Path(rules_path)), temperature=1.0,
        forced_json_requirement="complete_rule_map",
        sampling_truncation={"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0},
        **design,
    )
    (register_as_recorded if recorded else register_condition)(root, spec)
    cells = _judgments(seed, rule_10_rises=rule_10_rises, unsure_verdicts=unsure_verdicts, preempt=preempt)
    for s in range(S):
        draw_dir = root / spec.condition_id / f"draw-{s:03d}"
        draw_dir.mkdir(parents=True)
        outcomes, ledger = [], []
        for i, sid in enumerate(manifest.scenario_ids):
            alone, in_context, verdict, preempted = cells[(i, s)]
            attempt = 1 if (i, s) == (0, 0) else 0
            common = dict(campaign_id=name, condition_id=spec.condition_id, scenario_id=sid, decoder_draw_id=s,
                          seed_capability="local_deterministic", started_at="t0", completed_at="t1")
            if attempt:
                ledger.append(AttemptRecord(**common, semantic_attempt_id=0, status="rejected",
                                            request_block_id=f"{name}/draw-{s}/attempt-0/block-0000",
                                            components=(_component("full_ruleset", None),)))
            full = _component("full_ruleset", verdict, rule_judgments=dict(in_context), preempted_rules=tuple(preempted))
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
        (draw_dir / "completion.json").write_text(json.dumps({
            "schema_version": 2, "campaign_id": name, "condition_id": spec.condition_id,
            "condition_fingerprint": spec.fingerprint, "decoder_draw_id": s, "state": "complete",
            "materialized": True, "scenario_manifest_sha256": manifest.manifest_sha256, "expected_scenarios": N,
            "accepted": N, "attempts_histogram": {"1": N - int(s == 0), **({"2": 1} if s == 0 else {})},
            "rejected_attempts": int(s == 0), "transport_failed_attempts": 0,
            "invalid_components": {"invalid_json": 1} if s == 0 else {},
        }))
    return root / "aggregate" / "estimates.json"


def write_rules(path, rules=RULES):
    path.write_text("".join(json.dumps(rule) + "\n" for rule in rules))
    return path


@contextlib.contextmanager
def artifact_root(path):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RULEFORGE_ARTIFACT_ROOT", str(path))
        yield


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    """Two roots, as the paper's run has: the open judge here, Luna there."""
    base = tmp_path_factory.mktemp("campaign")
    with artifact_root(base):
        open_root = build_root(base, "gpu", model="gpt-oss-120b", seed=1)
        luna_root = build_root(base, "luna", model="gpt-5.6-luna", seed=2, rule_10_rises=False,
                               unsure_verdicts=False, preempt=False)
        payload = rc.build_payload([open_root, luna_root], base / "rules.jsonl")
    return {"base": base, "open": open_root, "luna": luna_root, "rules": base / "rules.jsonl", "payload": payload}


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A fresh artifact root for tests that damage a campaign."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    return tmp_path


def _edit_json(path, change):
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload))


def _edit_jsonl(path, change):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _bundles(estimates, condition_id):
    """Every outcome row and accepted full-set component of a condition, by (i, s)."""
    root = estimates.parent.parent
    ids = json.loads((root / "scenario_manifest.json").read_text())["scenario_ids"]
    out = {}
    for s in range(S):
        draw_dir = root / condition_id / f"draw-{s:03d}"
        rows = {r["scenario_id"]: r for r in map(json.loads, (draw_dir / "bundle_outcomes.jsonl").read_text().splitlines())}
        accepted = {r["scenario_id"]: r for r in map(json.loads, (draw_dir / "attempts.jsonl").read_text().splitlines())
                    if r["status"] == "accepted"}
        for i, sid in enumerate(ids):
            out[(i, s)] = (rows[sid], accepted[sid]["components"][0])
    return out


def test_the_golden_has_one_block_per_forced_condition_in_paper_order(campaign):
    payload = campaign["payload"]
    assert payload["data_version"] == 5 and payload["source"] == "stochastic_campaign"
    assert list(payload["models"]) == ["gpt-oss-120b", "gpt-5.6-luna"]
    assert payload["settings"]["family_size"] == N_RULES and payload["settings"]["n_decoder_draws"] == S
    assert payload["settings"]["transition_method"] == "independent_draw_expected_shares_v1"
    assert {key: payload["settings"][key] for key in rc.JOINED_DESIGN_FIELDS} == {
        "individual_rule_context": "isolated", "judge_type": "No_Interpretation", "label_rule": "lenient_last_line",
        "evaluation_mode": "both", "prompt_shape_version": 2, "validation_schema_version": 3,
    }
    assert payload["settings"]["alpha"] == 0.05 and payload["settings"]["holm_mc_margin_se"] == 3.0
    prov = payload["provenance"]
    assert prov["complete"] is True
    assert [c["estimates"] for c in prov["campaigns"]] == [
        "artifact://sampling/gpu/aggregate/estimates.json", "artifact://sampling/luna/aggregate/estimates.json",
    ]
    assert prov["campaigns"][0]["forced_conditions"] == ["gpt-oss-120b/forced-grpo"]
    assert prov["rules_file"]["name"] == "rules.jsonl" and prov["rules_file"]["n_rules"] == N_RULES
    assert prov["model_labels"] == {"gpt-oss-120b": "gpt-oss-120b", "gpt-5.6-luna": "gpt-5.6-luna"}
    block = payload["models"]["gpt-oss-120b"]
    assert block["attempt_statistics"]["rejected_attempts"] == 1
    assert block["attempt_statistics"]["attempts_histogram"] == {"1": N * S - 1, "2": 1}


def test_rows_arrive_in_string_order_and_are_joined_by_rule_number(campaign):
    estimates = json.loads(campaign["open"].read_text())
    rows = estimates["forced_conditions"]["gpt-oss-120b/forced-grpo"]["rule_context"]
    assert [row["rule_id"] for row in rows][:4] == ["Rule 1", "Rule 10", "Rule 11", "Rule 2"]
    by_id = {row["rule_id"]: row for row in rows}
    effects = campaign["payload"]["models"]["gpt-oss-120b"]["effects"]
    assert [row["rule"] for row in effects] == list(range(1, N_RULES + 1))
    for effect in effects:
        stored = by_id[f"Rule {effect['rule']}"]
        assert effect["title"].startswith(f"Title of rule {effect['rule']}")
        assert effect["delta"] == stored["estimate"] and effect["p_holm"] == stored["p_holm"]
        assert effect["grid_digest_in_context"] == stored["grid_digest_a"]
    assert effects[0]["delta"] < -0.4 and effects[9]["delta"] > 0.4
    assert effects[1]["delta"] == 0 and effects[1]["degenerate"] and effects[1]["few_informative"]
    assert effects[1]["yes_difference"] == 0 and ls.direction(effects[1]) == 0
    assert ls.direction(effects[0]) == -1 and ls.direction(effects[9]) == 1
    assert effects[1]["n_informative_scenarios"] == 0
    assert effects[0]["significant_holm"] and effects[9]["significant_holm"]


def test_every_identity_holds_against_an_independent_recount(campaign):
    effects = campaign["payload"]["models"]["gpt-oss-120b"]["effects"]
    bundles = _bundles(campaign["open"], "gpt-oss-120b/forced-grpo")
    index = {"rule 2.": "Rule 2", "Rule 3": "Rule 3"}
    for effect in effects:
        rule_id = effect["rule_id"]
        cells = {key: (row["rule_judgments"][rule_id], row["in_context_rule_judgments"][rule_id],
                       row["full_set_judgment"], rule_id in {index.get(p) for p in full["preempted_rules"]})
                 for key, (row, full) in bundles.items()}
        diff = {key: int(c == "yes") - int(a == "yes") for key, (a, c, _, _) in cells.items()}
        delta = sum(diff.values()) / (N * S)
        assert effect["yes_difference"] == sum(diff.values())
        assert effect["delta"] == pytest.approx(delta, abs=1e-12)
        assert effect["delta"] == pytest.approx(effect["rate_in_context"] - effect["rate_alone"], abs=1e-12)
        assert sum(effect["by_verdict"].values()) == pytest.approx(delta, abs=1e-12)
        for verdict in LABELS:
            assert effect["by_verdict"][verdict] == pytest.approx(
                sum(d for key, d in diff.items() if cells[key][2] == verdict) / (N * S), abs=1e-12)
        named = sum(d for key, d in diff.items() if cells[key][3]) / (N * S)
        unnamed = sum(d for key, d in diff.items() if not cells[key][3]) / (N * S)
        assert effect["named_preempted"]["delta_contribution"] == pytest.approx(named, abs=1e-12)
        assert named + unnamed == pytest.approx(delta, abs=1e-12)
        shares = {(a, b): 0.0 for a in LABELS for b in LABELS}
        for i in range(N):
            for s, t in itertools.product(range(S), repeat=2):
                shares[(cells[(i, s)][0], cells[(i, t)][1])] += 1 / (N * S * S)
        assert {(a, b): effect["transitions"][a][b] for a in LABELS for b in LABELS} == pytest.approx(shares)
        assert sum(shares.values()) == pytest.approx(1.0)
        phi = {"yes": 1.0, "unsure": 0.5, "no": 0.0}
        assert effect["phi_alone"] == pytest.approx(sum(phi[a] for a, *_ in cells.values()) / (N * S))
        informative = sum(1 for i in range(N) if sum(diff[(i, s)] for s in range(S)) != 0)
        assert effect["n_informative_scenarios"] == informative
    assert effects[1]["named_preempted"]["n_cells"] == 4 * S and effects[2]["named_preempted"]["n_cells"] == 4 * S


def test_the_model_blocks_pool_the_cells_and_carry_both_agreement_references(campaign):
    block = campaign["payload"]["models"]["gpt-oss-120b"]
    effects, summary = block["effects"], block["summary"]
    assert summary["n_cells"] == N * S * N_RULES
    assert summary["pooled_delta"] == pytest.approx(np.mean([row["delta"] for row in effects]))
    assert summary["n_rules_tied"] == sum(row["yes_difference"] == 0 for row in effects) >= 1
    assert summary["n_rules_lower"] == sum(row["yes_difference"] < 0 for row in effects)
    assert summary["n_holm_higher"] >= 1 and summary["n_holm_lower"] >= 1
    assert summary["pooled_yes_difference"] == sum(row["yes_difference"] for row in effects) != 0
    assert sum(e["delta_contribution"] for e in block["by_verdict"]["strata"].values()) == pytest.approx(summary["pooled_delta"])
    assert sum(e["share_of_pooled_delta"] for e in block["by_verdict"]["strata"].values()) == pytest.approx(1.0)
    assert block["by_verdict"]["strata"]["no"]["n_responses"] == 6 * S  # i = 0, 3, ..., 15
    flips = block["flips"]
    assert flips["unchanged_fraction"] == pytest.approx(sum(flips["matrix"][a][a] for a in LABELS))
    assert flips["n_cells"] == N * N_RULES
    bundles = _bundles(campaign["open"], "gpt-oss-120b/forced-grpo")
    for side, field in (("alone", "rule_judgments"), ("in_context", "in_context_rule_judgments")):
        agree = pairs = 0
        for i in range(N):
            for rule in RULES:
                labels = [bundles[(i, s)][0][field][rule["id"]] for s in range(S)]
                for s, t in itertools.permutations(range(S), 2):
                    agree += labels[s] == labels[t]
                    pairs += 1
        reference = flips["draw_to_draw"][side]
        assert reference["unchanged_fraction"] == pytest.approx(agree / pairs)
        assert sum(v for row in reference["matrix"].values() for v in row.values()) == pytest.approx(1.0)
    named = block["named_preempted"]
    assert named["n_named_cells"] == 2 * 4 * S and named["n_responses_naming"] == 4 * S
    assert named["unresolved_preempted_ids"] == 2 * 4 * S  # "Meta-Rule 1" and "Rule 99" on every naming response
    # Luna names nothing and never answers unsure overall: undefined, not zero.
    luna = campaign["payload"]["models"]["gpt-5.6-luna"]
    assert luna["named_preempted"]["in_context_no_rate"] is None and luna["named_preempted"]["n_named_cells"] == 0
    assert luna["by_verdict"]["strata"]["unsure"]["rate_alone"] is None
    assert luna["by_verdict"]["strata"]["unsure"]["n_responses"] == 0


def test_holm_is_recomputed_within_each_model(campaign):
    from statsmodels.stats.multitest import multipletests

    for block in campaign["payload"]["models"].values():
        raw = [row["p_value"] for row in block["effects"]]
        rejected, holm, _, _ = multipletests(raw, alpha=0.05, method="holm")
        assert [row["p_holm"] for row in block["effects"]] == pytest.approx(list(holm))
        assert [row["significant_holm"] for row in block["effects"]] == list(rejected)


def test_small_helpers_never_divide_by_zero():
    assert rc._ratio(1, 0) is None and rc._ratio(1, 4) == 0.25
    assert rc._share_of(0.1, 0.0) is None and rc._share_of(-0.02, -0.04) == pytest.approx(0.5)
    assert str(rc._share_of(0.0, -0.04)) == "0.0"  # an empty stratum, not -0.0


def _family(p_values, n_permutations=100_000):
    return [{"rule_id": f"Rule {n}", "p_value": p, "n_permutations": n_permutations}
            for n, p in enumerate(p_values, 1)]


def test_holm_flags_mark_the_floor_and_decisions_inside_monte_carlo_noise():
    floor = 1 / 100_001
    flags = rc.holm_flags(_family([floor, floor, 0.0249, 0.2]), alpha=0.05)
    assert [f["p_value_is_floor"] for f in flags] == [True, True, False, False]
    assert [f["holm_mc_margin"] for f in flags] == [False, False, True, False]
    # A step is near alpha when |m_j p - alpha| <= 3 m_j sqrt(p(1-p)/B).
    assert abs(0.0249 - 0.05 / 2) < 3 * np.sqrt(0.0249 * (1 - 0.0249) / 100_000)
    rows = _family([0.0165, 0.02, 0.9])
    assert [f["holm_mc_margin"] for f in rc.holm_flags(rows, alpha=0.05)] == [True, True, False]
    assert abs(0.02 - 0.05 / 2) > 3 * np.sqrt(0.02 * 0.98 / 100_000)  # the own-step rule alone would miss it
    reviewers = _family([0.0154967, 0.0232494, 0.9])
    assert [f["holm_mc_margin"] for f in rc.holm_flags(reviewers, alpha=0.05)] == [True, True, False]

    def se(m, p):
        return m * np.sqrt(p * (1 - p) / 100_000)

    assert abs(3 * 0.0154967 - 0.05) <= 3 * se(3, 0.0154967)
    assert abs(2 * 0.0232494 - 0.05) > 3 * se(2, 0.0232494) and 2 * 0.0232494 > 3 * 0.0154967
    assert [f["holm_mc_margin"] for f in rc.holm_flags(_family([0.0171, 0.02, 0.9]), alpha=0.05)] == [
        True, True, False]
    assert not any(f["holm_mc_margin"] for f in rc.holm_flags(_family([0.03, 0.04, 0.9]), alpha=0.05))
    # Ties share the first tied step (Holm's own value), whatever the order.
    tied = _family([0.5, 0.0166, 0.0166])
    assert [f["holm_mc_margin"] for f in rc.holm_flags(tied, alpha=0.05)] == [False, True, True]
    assert rc.holm_flags(tied[::-1], alpha=0.05) == rc.holm_flags(tied, alpha=0.05)[::-1]
    assert "significant_holm" not in rc.holm_flags(rows, alpha=0.05)[0]


def test_a_non_significant_row_is_flagged_only_when_every_failing_step_could_flip():
    def se(m, p):
        return m * np.sqrt(p * (1 - p) / 100_000)

    p_values = [0.0012] + [0.0012 + 1e-7 * k for k in range(1, 48)] + [0.0295, 0.9]
    m = len(p_values)
    flags = [f["holm_mc_margin"] for f in rc.holm_flags(_family(p_values), alpha=0.05)]
    failing = [j for j, p in enumerate(p_values) if (m - j) * p > 0.05]
    assert failing == [*range(9), 48, 49]
    assert all(((m - j) * p_values[j] - 0.05) / se(m - j, p_values[j]) < 1.83 for j in range(9))
    assert ((2 * 0.0295) - 0.05) / se(2, 0.0295) == pytest.approx(8.41, abs=0.01)
    assert (50 * 0.0012 - 0.05) <= 3 * se(50, 0.0012) and 50 * 0.0012 > 2 * 0.0295  # the binding step is near
    assert flags == [True] * 48 + [False, False]
    assert [f["holm_mc_margin"] for f in rc.holm_flags(_family([0.02, 0.0255, 0.9]), alpha=0.05)] == [
        False, False, False]
    assert abs(2 * 0.0255 - 0.05) <= 3 * se(2, 0.0255) and 3 * 0.02 - 0.05 > 3 * se(3, 0.02)


def test_the_golden_carries_the_monte_carlo_flags(campaign):
    for block in campaign["payload"]["models"].values():
        effects = block["effects"]
        assert all(isinstance(row["p_value_is_floor"], bool) and isinstance(row["holm_mc_margin"], bool)
                   for row in effects)
        assert [{k: row[k] for k in ("p_value_is_floor", "holm_mc_margin")} for row in effects] == (
            rc.holm_flags(effects, alpha=0.05))
    effects = campaign["payload"]["models"]["gpt-oss-120b"]["effects"]
    # Rule 10 rises in all 16 scenarios, so no sign flip is as extreme; Rule 2 never moves.
    assert effects[9]["p_value_is_floor"] and effects[9]["p_value"] == 1 / (PERMUTATIONS + 1)
    assert not effects[1]["p_value_is_floor"]


def test_directions_come_from_the_exact_count_not_the_float_delta():
    """Equal yes counts in different scenarios can leave the estimate at -2e-19."""
    zeros = np.zeros((1, 1, 1), dtype=np.int8)
    arrays = rc.ConditionCells(
        condition_id="x", scenario_ids=("a",), alone=zeros, in_context=zeros,
        verdict=np.zeros((1, 1), dtype=np.int8), named=np.zeros((1, 1, 1), dtype=bool),
        unresolved_preempted_ids=0, outcomes={},
    )
    effects = [
        {"delta": -2.168e-19, "yes_difference": 0, "significant_holm": False},
        {"delta": -0.1, "yes_difference": -250, "significant_holm": True},
        {"delta": 0.04, "yes_difference": 100, "significant_holm": True},
    ]
    summary = rc.model_summary(effects, arrays)
    assert (summary["n_rules_lower"], summary["n_rules_higher"], summary["n_rules_tied"]) == (1, 1, 1)
    assert (summary["n_holm_lower"], summary["n_holm_higher"]) == (1, 1)
    assert summary["pooled_yes_difference"] == -150


def test_shares_of_a_pooled_delta_whose_exact_count_is_zero_are_undefined():
    y, n = LABELS.index("yes"), LABELS.index("no")
    alone = np.array([[[y, y]], [[y, n]], [[n, n]]], dtype=np.int8)  # (N=3, S=1, R=2)
    in_context = np.array([[[y, n]], [[y, n]], [[y, n]]], dtype=np.int8)
    named = np.zeros(alone.shape, dtype=bool)
    named[0, 0, 1] = True  # the -1 cell of the second rule
    arrays = rc.ConditionCells(
        condition_id="x", scenario_ids=("a", "b", "c"), alone=alone, in_context=in_context,
        verdict=np.array([[n], [y], [y]], dtype=np.int8), named=named, unresolved_preempted_ids=0, outcomes={},
    )

    def rate(grid):
        return float(np.mean([(row == y).mean() for row in grid]))

    effects = [
        {"delta": rate(in_context[:, :, k]) - rate(alone[:, :, k]),
         "yes_difference": int(arrays.difference[:, :, k].sum()), "significant_holm": False}
        for k in range(2)
    ]
    assert [row["yes_difference"] for row in effects] == [1, -1]
    summary = rc.model_summary(effects, arrays)
    assert summary["pooled_yes_difference"] == 0 and summary["pooled_delta"] != 0  # 2.8e-17
    by_verdict = rc.verdict_decomposition(arrays, summary["pooled_delta"], summary["pooled_yes_difference"])
    named_block = rc.named_decomposition(arrays, summary["pooled_delta"], summary["pooled_yes_difference"])
    assert by_verdict["strata"]["no"]["delta_contribution"] == pytest.approx(-1 / 6)
    assert named_block["delta_contribution"] == pytest.approx(-1 / 6)
    assert all(stratum["share_of_pooled_delta"] is None for stratum in by_verdict["strata"].values())
    assert named_block["share_of_pooled_delta"] is None
    assert abs(rc._share_of(-1 / 6, summary["pooled_delta"])) > 1e15
    kept = rc.verdict_decomposition(arrays, 0.25, 1)
    assert kept["strata"]["no"]["share_of_pooled_delta"] == pytest.approx((-1 / 6) / 0.25)


def test_roots_that_are_not_one_experiment_are_refused(scratch):
    first = build_root(scratch, "a", model="gpt-oss-120b")
    other = build_root(scratch, "b", model="gpt-5.6-luna", bootstrap={"n_resamples": 300, "ci_level": 0.95, "seed": 16})
    with pytest.raises(CampaignEstimatesError, match="bootstrap .* not one experiment"):
        rc.build_payload([first, other], scratch / "rules.jsonl")


def test_an_unknown_judge_is_refused_rather_than_guessed(scratch):
    odd = build_root(scratch, "odd", model="mystery-7b")
    with pytest.raises(CampaignEstimatesError, match="no golden label for judge 'mystery-7b'"):
        rc.build_payload([odd], scratch / "rules.jsonl")


def test_gpt_6_1_sol_and_qwen3_8b_are_named_judges():
    from ruleforge.experiments.labels import campaign_label

    assert campaign_label("gpt-6.1-sol") == "gpt-6.1-sol" and campaign_label("Qwen3-8B") == "Qwen3-8B"


def test_two_conditions_of_one_judge_are_refused(scratch):
    first = build_root(scratch, "a", model="gpt-oss-120b")
    second = build_root(scratch, "b", model="gpt-oss-120b", condition_id="gpt-oss-120b-rerun/forced-grpo")
    with pytest.raises(RuleContextError, match="golden label gpt-oss-120b: forced conditions"):
        rc.build_payload([first, second], scratch / "rules.jsonl")


def test_a_condition_whose_alone_is_not_a_rule_judged_alone_is_refused(scratch):
    estimates = build_root(scratch, "a", design={"individual_rule_context": "implicated"})
    with pytest.raises(RuleContextError, match="individual_rule_context 'implicated', not 'isolated'"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_joined_conditions_must_share_the_judging_design(scratch):
    first = build_root(scratch, "a", model="gpt-oss-120b")
    other = build_root(scratch, "b", model="gpt-5.6-luna", design={"judge_type": "Narrow", "label_rule": "lenient"})
    with pytest.raises(RuleContextError, match=(
        "gpt-5.6-luna/forced-grpo: differs from gpt-oss-120b/forced-grpo on "
        "judge_type 'Narrow' vs 'No_Interpretation', label_rule 'lenient' vs 'lenient_last_line'"
    )):
        rc.build_payload([first, other], scratch / "rules.jsonl")


def test_claude_haiku_5_5_joins_the_rule_context_under_one_label_rule_with_no_special_case(scratch):
    """The published judges' forced arms re-scored under lenient_last_line, Claude Haiku 5.5's forced arm."""
    design = {"label_rule": "lenient_last_line"}
    roots = [build_root(scratch, name, model=model, seed=seed, design=design, styles=("forced",))
             for name, model, seed in (("open", "gpt-oss-120b", 1), ("luna", "gpt-5.6-luna", 2),
                                       ("haiku", "claude-haiku-5-5", 3))]
    _edit_json(scratch / "sampling" / "haiku" / "sampling_manifest.json", lambda m: m.update(retry_scope="component"))
    payload = rc.build_payload(roots[::-1], scratch / "rules.jsonl", prompt_style="forced")
    assert list(payload["models"]) == ["gpt-oss-120b", "gpt-5.6-luna", "claude-haiku-5-5"]
    assert payload["settings"]["label_rule"] == "lenient_last_line"
    assert payload["models"]["claude-haiku-5-5"]["condition_id"] == "claude-haiku-5-5/forced"
    markup = build_root(scratch, "gemma", model="gemma-4-31B-it", seed=4, design={"label_rule": "lenient_markup"},
                        styles=("forced",))
    with pytest.raises(RuleContextError, match="label_rule 'lenient_markup' vs 'lenient_last_line'"):
        rc.build_payload([roots[2], markup], scratch / "rules.jsonl", prompt_style="forced")


def test_muse_glimmer_30b_joins_the_rule_context_in_its_place_with_no_special_case(scratch):
    """Muse-Glimmer-30B's forced arm."""
    design = {"label_rule": "lenient_last_line"}
    roots = [build_root(scratch, name, model=model, seed=seed, design=design, styles=("forced",))
             for name, model, seed in (("open", "gpt-oss-120b", 1), ("luna", "gpt-5.6-luna", 2),
                                       ("glimmer", "Muse-Glimmer-30B", 5))]
    _edit_json(scratch / "sampling" / "glimmer" / "sampling_manifest.json",
               lambda m: m.update(retry_scope="component"))
    payload = rc.build_payload(roots[::-1], scratch / "rules.jsonl", prompt_style="forced")
    assert payload["provenance"]["models"] == ["gpt-oss-120b", "muse-glimmer-30b", "gpt-5.6-luna"]
    assert payload["models"]["muse-glimmer-30b"]["condition_id"] == "Muse-Glimmer-30B/forced"
    assert payload["provenance"]["model_labels"]["Muse-Glimmer-30B"] == "muse-glimmer-30b"


def test_a_rules_file_the_conditions_did_not_judge_is_refused(scratch):
    estimates = build_root(scratch, "a")
    edited = write_rules(scratch / "other.jsonl", [dict(rule, rule=rule["rule"] + ".") for rule in RULES])
    with pytest.raises(RuleContextError, match="not the rules file's"):
        rc.build_payload([estimates], edited)


def test_a_grid_digest_or_estimate_that_does_not_match_the_draws_is_refused(scratch):
    estimates = build_root(scratch, "a")
    cid = "gpt-oss-120b/forced-grpo"

    def tamper(payload):
        row = next(r for r in payload["forced_conditions"][cid]["rule_context"] if r["rule_id"] == "Rule 4")
        row["grid_digest_b"] = "0" * 64
        row["estimate"] += 0.01
        row["test_method"] = "mcnemar"

    _edit_json(estimates, tamper)
    with pytest.raises(RuleContextError, match="Rule 4: alone grid digest .*estimate is not .*test method 'mcnemar'"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_a_stored_holm_column_that_disagrees_is_refused(scratch):
    estimates = build_root(scratch, "a")

    def rows(p):
        return p["forced_conditions"]["gpt-oss-120b/forced-grpo"]["rule_context"]

    _edit_json(estimates, lambda p: rows(p)[3].update(p_holm=rows(p)[3]["p_holm"] / 2))
    with pytest.raises(RuleContextError, match="stored p_holm"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    _edit_json(estimates, lambda p: rows(p)[3].update(p_holm=rows(p)[3]["p_holm"] * 2,
                                                      significant_holm=not rows(p)[3]["significant_holm"]))
    with pytest.raises(RuleContextError, match="significant_holm disagrees"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def _rule_row(payload, number, cid="gpt-oss-120b/forced-grpo"):
    """The aggregate's ``Rule {number}`` row of ``cid`` (its rows arrive in string order)."""
    return next(row for row in payload["forced_conditions"][cid]["rule_context"] if row["rule_id"] == f"Rule {number}")


def _off_the_lattice(payload, cid="gpt-oss-120b/forced-grpo"):
    """Every row's p a quarter step below the permutation test's lattice, Holm re-decided over them."""
    rows = payload["forced_conditions"][cid]["rule_context"]
    for row in rows:
        row["p_value"] -= 0.25 / (PERMUTATIONS + 1)
    for row, fresh in zip(rows, holm_adjust(rows)):
        row.update(p_holm=fresh["p_holm"], significant_holm=fresh["significant_holm"])


#: What the build copies from the aggregate without recomputing it, and validate_payload range-checks on re-read.
COPIED_UNCHECKED = [
    (lambda p: _rule_row(p, 4).update(ci_high=1.5), r"Rule 4: interval \[[^]]*, 1\.5\] is not ordered within \[-1, 1\]"),
    (lambda p: _rule_row(p, 4).update(ci_low=_rule_row(p, 4)["ci_high"] + 0.125), r"Rule 4: interval .* is not ordered"),
    (lambda p: _rule_row(p, 5).update(n_permutations=PERMUTATIONS + 1),
     rf"Rule 5: n_permutations {PERMUTATIONS + 1}, not the settings' randomization test's {PERMUTATIONS}"),
    (_off_the_lattice, f"is not a probability on the {PERMUTATIONS}-permutation test's lattice"),
    (lambda p: p["bootstrap"].update(ci_level=95), r"bootstrap .* is not a ci_level in \(0, 1\)"),
]


def test_the_build_refuses_a_golden_its_own_re_read_would_refuse(scratch):
    estimates = build_root(scratch, "a")
    original = estimates.read_text()
    for change, message in COPIED_UNCHECKED:
        estimates.write_text(original)
        _edit_json(estimates, change)
        with pytest.raises(RuleContextError, match=f"^the golden as built is one its own re-read refuses: .*{message}"):
            rc.build_payload([estimates], scratch / "rules.jsonl")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows[:-1], "rule_context rows cover rules"),
        (lambda rows: rows + [dict(rows[0])], "two rule_context rows name one rule"),
        (lambda rows: [dict(rows[0], rule_id="overall")] + rows[1:], "names no rule number"),
        (lambda rows: [dict(r, rule_id="rule 4") if r["rule_id"] == "Rule 4" else r for r in rows], "spelled differently"),
    ],
)
def test_rule_rows_must_cover_the_rules_file_exactly(scratch, change, message):
    estimates = build_root(scratch, "a")
    cid = "gpt-oss-120b/forced-grpo"
    _edit_json(estimates, lambda p: p["forced_conditions"][cid].update(rule_context=change(p["forced_conditions"][cid]["rule_context"])))
    with pytest.raises(RuleContextError, match=message):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_the_estimates_must_describe_the_root_beside_them(scratch):
    estimates = build_root(scratch, "a")
    _edit_json(estimates.parent.parent / "sampling_manifest.json", lambda p: p.update(campaign_id="elsewhere"))
    with pytest.raises(RuleContextError, match="campaign_id differs"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_a_pilot_campaign_is_a_scratch_build_only(scratch):
    estimates = build_root(scratch, "a", mode="pilot")
    with pytest.raises(RuleContextError, match="need a final campaign"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    payload = rc.build_payload([estimates], scratch / "rules.jsonl", allow_incomplete=True)
    assert payload["provenance"]["complete"] is False


STACK_MISMATCH = [{"scope": "condition", "conditions": ["gpt-oss-120b/forced-grpo"],
                   "reason": "its judgments come from several stacks (vllm 0.19 / torch ? / transformers ? + vllm 0.28 / torch ? / transformers ?)"}]


def test_estimates_reporting_a_stack_mismatch_are_built_only_on_the_record(scratch, driver, tmp_path):
    estimates = build_root(scratch, "a")
    _edit_json(estimates, lambda p: p.update(allow_stack_mismatch=True, stack_mismatches=STACK_MISMATCH))
    with pytest.raises(CampaignEstimatesError, match=r"reports 1 inference-stack mismatch.*gpt-oss-120b/forced-grpo: its"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    payload = rc.build_payload([estimates], scratch / "rules.jsonl", allow_stack_mismatch=True)
    assert payload["provenance"]["campaigns"][0]["stack_mismatches"] == STACK_MISMATCH
    golden = tmp_path / "g.json"
    argv = ["--estimates", str(estimates), "--rules", str(scratch / "rules.jsonl"), "--golden", str(golden),
            "--prompt-style", "forced-grpo"]
    with pytest.raises(SystemExit):
        driver.main(argv)
    assert driver.main([*argv, "--allow-stack-mismatch"]) == 0
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden)])
    assert driver.main(["--from_golden", str(golden), "--allow-stack-mismatch"]) == 0


def test_a_condition_the_root_does_not_register_or_describe_is_refused(scratch):
    estimates = build_root(scratch, "a")
    cid = "gpt-oss-120b/forced-grpo"
    _edit_json(estimates, lambda p: p["conditions"][cid].update(condition_fingerprint="f" * 64))
    with pytest.raises(RuleContextError, match="another condition fingerprint"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    _edit_json(estimates, lambda p: p["forced_conditions"].update({"ghost/forced-grpo": p["forced_conditions"][cid]}))
    with pytest.raises(RuleContextError, match="ghost/forced-grpo: no registered condition.json"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    _edit_json(estimates, lambda p: p.update(forced_conditions={}))
    with pytest.raises(RuleContextError, match="no forced"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_draws_that_name_another_condition_or_are_not_materialized_are_refused(scratch):
    estimates = build_root(scratch, "a")
    draw = estimates.parent.parent / "gpt-oss-120b" / "forced-grpo" / "draw-001" / "completion.json"
    _edit_json(draw, lambda p: p.update(materialized=False))
    with pytest.raises(RuleContextError, match="not materialized"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    for s in range(S):
        path = draw.parent.parent / f"draw-{s:03d}" / "completion.json"
        _edit_json(path, lambda p: p.update(condition_fingerprint="e" * 64, materialized=True))
    with pytest.raises(RuleContextError, match="draws record another condition fingerprint"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_an_outcome_the_ledger_does_not_vouch_for_is_refused(scratch):
    estimates = build_root(scratch, "a")
    draw_dir = estimates.parent.parent / "gpt-oss-120b" / "forced-grpo" / "draw-002"

    def flip(rows):
        rows[5]["in_context_rule_judgments"]["Rule 7"] = "unsure" if rows[5]["in_context_rule_judgments"]["Rule 7"] != "unsure" else "no"
        return rows

    _edit_jsonl(draw_dir / "bundle_outcomes.jsonl", flip)
    with pytest.raises(RuleContextError, match="not the one the bundle outcome was built from"):
        rc.build_payload([estimates], scratch / "rules.jsonl")
    _edit_jsonl(draw_dir / "attempts.jsonl", lambda rows: rows[:5] + rows[6:])
    with pytest.raises(RuleContextError, match="no accepted ledger row"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_an_accepted_row_without_one_full_set_component_is_refused(scratch):
    estimates = build_root(scratch, "a")
    draw_dir = estimates.parent.parent / "gpt-oss-120b" / "forced-grpo" / "draw-001"

    def drop_full(rows):
        rows[3]["components"] = rows[3]["components"][1:]
        return rows

    _edit_jsonl(draw_dir / "attempts.jsonl", drop_full)
    with pytest.raises(RuleContextError, match="has 0 full-set components"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


def test_an_outcome_over_other_rules_or_labels_is_refused(scratch):
    estimates = build_root(scratch, "a")
    root = estimates.parent.parent
    draw_dir = root / "gpt-oss-120b" / "forced-grpo" / "draw-000"
    rules = write_rules(scratch / "short.jsonl", RULES[:-1])
    # Straight to the reader: the build would refuse the short file's digest first.
    with pytest.raises(RuleContextError, match="do not cover exactly the rules file's ids"):
        rc.read_condition(
            root, "gpt-oss-120b/forced-grpo", campaign_id="a",
            scenario_ids=ScenarioManifest.load(root / "scenario_manifest.json").scenario_ids,
            n_draws=S, manifest_sha256=ScenarioManifest.load(root / "scenario_manifest.json").manifest_sha256,
            rules=rc.load_rules(rules),
        )

    def odd_label(rows):
        rows[2]["rule_judgments"]["Rule 4"] = "maybe"
        return rows

    _edit_jsonl(draw_dir / "bundle_outcomes.jsonl", odd_label)
    with pytest.raises(RuleContextError, match="judgment 'maybe' is not one of"):
        rc.build_payload([estimates], scratch / "rules.jsonl")


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([{"id": "Overview", "rule": "x"}], "names no rule number"),
        ([{"id": "Rule 1", "rule": "x"}, {"id": "rule1", "rule": "y"}], "appears twice"),
        ([{"id": "Rule 1", "rule": "  "}], "has no title"),
        ([], "no rules"),
    ],
)
def test_a_malformed_rules_file_is_refused(tmp_path, lines, message):
    with pytest.raises(RuleContextError, match=message):
        rc.load_rules(write_rules(tmp_path / "r.jsonl", lines))


def test_rules_load_sorted_by_number(tmp_path):
    rules = rc.load_rules(write_rules(tmp_path / "r.jsonl", [RULES[9], RULES[0], RULES[1]]))
    assert [(r.number, r.rule_id) for r in rules] == [(1, "Rule 1"), (2, "Rule 2"), (10, "Rule 10")]


def test_the_golden_carries_no_string_the_registry_reads_as_an_absolute_path(campaign):
    assert list(absolute_strings(campaign["payload"])) == []
    assert list(absolute_strings(rc.DEFINITION)) == []


def test_validate_payload_refuses_stale_or_tampered_goldens(campaign):
    payload = campaign["payload"]
    lg.validate_payload(payload)

    def oss(p):
        return p["models"]["gpt-oss-120b"]["effects"]

    def head(p):
        return p["models"]["gpt-oss-120b"]["summary"]

    not_quotient = r"delta .* is not yes_difference / \(n_scenarios x n_decoder_draws\)"
    cases = [
        (lambda p: p.update(data_version=4), "data_version 4, not 5"),
        (lambda p: oss(p).pop(), r"effects cover rules \[1, .*10\], the golden's"),
        (lambda p: p["models"]["gpt-5.6-luna"]["effects"][4].update(rule=99), "effects cover rules"),
        (lambda p: p["settings"].update(family_size=12), "family_size 12, 11 rules"),
        (lambda p: p["rules"].pop(), "family_size 11, 10 rules"),
        (lambda p: oss(p)[1].update(yes_difference=1), not_quotient),
        (lambda p: oss(p)[0].update(yes_difference=0), not_quotient),
        (lambda p: oss(p)[0].update(yes_difference=5), not_quotient),
        (lambda p: oss(p)[0].update(yes_difference=oss(p)[0]["yes_difference"] + 1), f"= .* / {N * S}"),
        (lambda p: oss(p)[0].update(yes_difference=float(oss(p)[0]["yes_difference"])), "is not an integer"),
        (lambda p: oss(p)[1].update(yes_difference=False), "yes_difference False is not an integer"),
        (lambda p: oss(p)[1].update(yes_difference="0"), "yes_difference '0' is not an integer"),
        # Every row on the settings' grid, and the settings a real grid.
        (lambda p: oss(p)[4].update(n_scenarios=N + 1), f"grid {N + 1} x {S}, the settings' {N} x {S}"),
        (lambda p: oss(p)[4].update(n_decoder_draws=S - 1), f"grid {N} x {S - 1}, the settings' {N} x {S}"),
        (lambda p: p["settings"].update(n_decoder_draws=S + 1), f"grid {N} x {S}, the settings' {N} x {S + 1}"),
        (lambda p: p["settings"].update(n_scenarios=0), "settings grid 0 x 3 is not two positive integers"),
        (lambda p: p["settings"].update(n_decoder_draws=3.0), "settings grid 16 x 3.0 is not two positive"),
        (lambda p: _flip(oss(p)[0], "few_informative"), "Rule 1: few_informative True disagrees with n_informative"),
        (lambda p: oss(p)[1].update(n_informative_scenarios=10), "few_informative True disagrees .* 10 < 10"),
        (lambda p: p["settings"].update(few_informative_below=9), "few_informative_below 9, not this module's 10"),
        (lambda p: oss(p)[4].update(n_informative_scenarios=N + 1),
         f"Rule 5: n_informative_scenarios {N + 1} is not a count of the {N} scenarios"),
        (lambda p: oss(p)[1].update(n_informative_scenarios=-1), "n_informative_scenarios -1 is not a count"),
        (lambda p: oss(p)[0].update(n_informative_scenarios=0, few_informative=True),
         r"Rule 1: n_informative_scenarios 0 is not a count of the \d+ scenarios that can hold yes_difference -\d+"),
        # An interval is not recomputable from a golden, so it is held to its order and range.
        (lambda p: oss(p)[4].update(ci_low=oss(p)[4]["ci_high"] + 0.1), r"Rule 5: interval \[.*\] is not ordered"),
        (lambda p: oss(p)[4].update(ci_high=1.5), r"Rule 5: interval \[.*, 1.5\] is not ordered within \[-1, 1\]"),
        (lambda p: p["settings"].update(holm_mc_margin_se=2.0), "holm_mc_margin_se 2.0, not this module's 3.0"),
        (lambda p: head(p).update(pooled_yes_difference=0), "pooled_yes_difference 0 is not the sum"),
        (lambda p: head(p).update(pooled_yes_difference=None), "pooled_yes_difference None"),
        (lambda p: head(p).update(n_holm_lower=head(p)["n_holm_lower"] + 1),
         r"summary counts \{.*'n_holm_lower': \d+.*\} do not recount from the rows"),
        (lambda p: head(p).update(n_holm_higher=head(p)["n_holm_higher"] + 1), "summary counts .* do not recount"),
        (lambda p: head(p).update(n_rules_lower=head(p)["n_rules_lower"] - 1,
                                  n_rules_tied=head(p)["n_rules_tied"] + 1), "summary counts .* do not recount"),
        (lambda p: head(p).update(n_rules_higher=float(head(p)["n_rules_higher"])), "n_rules_higher .* is not an integer"),
        (lambda p: head(p).update(n_holm_lower=True), "n_holm_lower True is not an integer"),
        (lambda p: head(p).pop("n_rules_tied"), "gpt-oss-120b summary: n_rules_tied is missing"),
        (lambda p: _flip(oss(p)[4], "p_value_is_floor"), "p_value_is_floor / holm"),
        (lambda p: _flip(p["models"]["gpt-oss-120b"]["effects"][4], "holm_mc_margin"), "holm_mc_margin disagree"),
        (lambda p: p["models"]["gpt-oss-120b"]["effects"][4].update(p_holm=0.5), "stored p_holm"),
        (lambda p: p["models"]["gpt-oss-120b"]["effects"][4]["transitions"]["yes"].update(no=0.9), "shares do not sum"),
        (lambda p: p["models"]["gpt-oss-120b"]["effects"][4]["by_verdict"].update(no=0.9), "by-judgment"),
        (lambda p: p["models"]["gpt-oss-120b"]["effects"][4].update(rate_alone=0.9), "rate_in_context - rate_alone"),
    ]
    for change, message in cases:
        tampered = copy.deepcopy(payload)
        change(tampered)
        with pytest.raises(RuleContextError, match=message):
            lg.validate_payload(tampered)
    # Float noise on an exact tie is not a disagreement; a 1e-11 delta on it is.
    noisy = copy.deepcopy(payload)
    noisy["models"]["gpt-oss-120b"]["effects"][1]["delta"] = -2.168e-19
    lg.validate_payload(noisy)
    noisy["models"]["gpt-oss-120b"]["effects"][1]["delta"] = 1e-11
    with pytest.raises(RuleContextError, match=not_quotient):
        lg.validate_payload(noisy)


def _flip(row, key):
    row[key] = not row[key]


def _refused(payload, change, message):
    tampered = copy.deepcopy(payload)
    change(tampered)
    with pytest.raises(RuleContextError, match=message):
        lg.validate_payload(tampered)


def test_validate_payload_refuses_mistyped_fields_before_any_arithmetic(campaign):
    payload = campaign["payload"]

    def oss(p):
        return p["models"]["gpt-oss-120b"]["effects"]

    row = oss(payload)[4]
    cases = [
        # Grid fields: integers, not floats or booleans -- even value-equal ones.
        (lambda p: oss(p)[4].update(n_scenarios=float(N)), f"Rule 5: n_scenarios {float(N)} is not an integer"),
        (lambda p: oss(p)[4].update(n_scenarios=500.0), "n_scenarios 500.0 is not an integer"),
        (lambda p: oss(p)[4].update(n_decoder_draws=True), "n_decoder_draws True is not an integer"),
        (lambda p: p["settings"].update(n_scenarios=float(N)), f"settings grid {float(N)} x {S} is not two positive"),
        (lambda p: p["settings"].update(n_decoder_draws=True), f"settings grid {N} x True is not two positive"),
        (lambda p: p["settings"].update(family_size=float(N_RULES)), f"family_size {float(N_RULES)}, {N_RULES} rules"),
        (lambda p: oss(p)[4].update(rule="5"), "rule '5' is not an integer"),
        (lambda p: oss(p)[4].update(n_informative_scenarios="3"), "n_informative_scenarios '3' is not an integer"),
        (lambda p: oss(p)[4].update(n_permutations=4000.0), "n_permutations 4000.0 is not an integer"),
        (lambda p: oss(p)[4].update(delta=str(row["delta"])), f"delta '{row['delta']}' is not a real number"),
        (lambda p: oss(p)[4].update(delta=float("nan")), "delta nan is not a real number"),
        (lambda p: oss(p)[4].update(rate_alone=True), "rate_alone True is not a real number"),
        (lambda p: oss(p)[4].update(rate_in_context=None), "rate_in_context None is not a real number"),
        (lambda p: oss(p)[4].update(p_value=str(row["p_value"])), "p_value '.*' is not a real number"),
        (lambda p: oss(p)[4].update(p_holm=False), "p_holm False is not a real number"),
        (lambda p: oss(p)[4].pop("ci_low"), "Rule 5: ci_low is missing"),
        # The five flags: booleans, not 0/1 or strings.
        (lambda p: oss(p)[1].update(few_informative=1), "Rule 2: few_informative 1 is not a boolean"),
        (lambda p: oss(p)[4].update(degenerate="false"), "degenerate 'false' is not a boolean"),
        (lambda p: oss(p)[4].update(significant_holm=int(row["significant_holm"])), "significant_holm [01] is not a"),
        (lambda p: oss(p)[4].update(p_value_is_floor=None), "p_value_is_floor None is not a boolean"),
        (lambda p: oss(p)[4].pop("holm_mc_margin"), "holm_mc_margin is missing"),
        (lambda p: oss(p)[4].pop("rule_id"), r"gpt-oss-120b effects\[4\]: rule_id is missing"),
        (lambda p: oss(p).__setitem__(4, 7), r"gpt-oss-120b effects\[4\]: int is not an object"),
        # Every row's p is over the settings' permutation count (a reader compares those between goldens).
        (lambda p: oss(p)[4].update(n_permutations=0), f"n_permutations 0, not the settings' randomization test's {PERMUTATIONS}"),
        (lambda p: oss(p)[4].update(n_permutations=PERMUTATIONS + 1), f"n_permutations {PERMUTATIONS + 1}, not the"),
        (lambda p: p["settings"].pop("randomization_test"), "randomization_test.n_permutations None is not a positive"),
        (lambda p: p["settings"]["randomization_test"].update(n_permutations=float(PERMUTATIONS)),
         f"randomization_test.n_permutations {float(PERMUTATIONS)} is not a positive integer"),
        (lambda p: oss(p)[4].update(p_value=1.5), "p_value 1.5 is not a probability"),
        (lambda p: oss(p)[4]["transitions"]["yes"].update(no="0.1"), "transitions is not a"),
        (lambda p: oss(p)[4]["transitions"].pop("unsure"), "transitions is not a"),
        (lambda p: oss(p)[4]["by_verdict"].pop("unsure"), "by_verdict is not a"),
        (lambda p: oss(p)[4].update(by_verdict=[0.0, 0.0, 0.0]), "by_verdict is not a"),
        (lambda p: p.clear() or p.update(data_version=5, source="stochastic_campaign", metric="flat"),
         "settings is not an object"),
        (lambda p: p["rules"][0].update(number="1"), "rules is not a non-empty list"),
        (lambda p: p.update(models={}), "models is not a non-empty object"),
        (lambda p: p["models"].update(extra=[]), "models is not a non-empty object"),
        (lambda p: p["provenance"].update(complete="yes"), "provenance.complete is not a boolean"),
        (lambda p: p["settings"].pop("alpha"), "alpha None is not a level in"),
        (lambda p: p["models"]["gpt-oss-120b"].pop("effects"), "gpt-oss-120b: effects is not a list"),
    ]
    for change, message in cases:
        _refused(payload, change, message)
    with pytest.raises(RuleContextError, match="list is not an object"):
        lg.validate_payload([payload])


def test_validate_payload_ties_the_pooled_figures_to_the_rows(campaign):
    payload = campaign["payload"]

    def head(p):
        return p["models"]["gpt-oss-120b"]["summary"]

    def strata(p):
        return p["models"]["gpt-oss-120b"]["by_verdict"]["strata"]

    def named(p):
        return p["models"]["gpt-oss-120b"]["named_preempted"]

    summary = head(payload)
    cells = N * S * N_RULES
    assert summary["n_cells"] == cells and summary["pooled_yes_difference"] != 0
    assert abs(summary["pooled_delta"] - summary["pooled_yes_difference"] / cells) <= 1e-12
    cases = [
        (lambda p: head(p).update(pooled_delta=summary["pooled_delta"] + 2e-12),
         rf"pooled_delta .* is not pooled_yes_difference / n_cells = {summary['pooled_yes_difference']} / {cells}"),
        (lambda p: head(p).update(pooled_delta=str(summary["pooled_delta"])), "pooled_delta '.*' is not a real number"),
        (lambda p: head(p).update(n_cells=cells + 1), f"n_cells {cells + 1}, not {N_RULES} rules x {N * S}"),
        (lambda p: head(p).update(n_cells=float(cells)), f"n_cells {float(cells)} is not an integer"),
        (lambda p: head(p).update(pooled_rate_alone=summary["pooled_rate_alone"] + 1 / cells),
         rf"pooled_rate_alone .* is not the rows' 'yes' count \d+ / {cells}"),
        (lambda p: head(p).update(pooled_rate_in_context=True), "pooled_rate_in_context True is not a real number"),
        (lambda p: p["models"]["gpt-oss-120b"].pop("summary"), "gpt-oss-120b summary: NoneType is not an object"),
        (lambda p: p["models"]["gpt-oss-120b"]["effects"][4].update(
            rate_alone=payload["models"]["gpt-oss-120b"]["effects"][4]["rate_alone"] + 0.001,
            rate_in_context=payload["models"]["gpt-oss-120b"]["effects"][4]["rate_in_context"] + 0.001),
         f"Rule 5: rate_alone .* is not a count of 'yes' over the {N * S} cells"),
        (lambda p: strata(p)["unsure"].update(share_of_pooled_delta=None),
         "by_verdict unsure share_of_pooled_delta None with pooled_yes_difference -?\\d+: it is null exactly"),
        (lambda p: named(p).update(share_of_pooled_delta=None), "named_preempted share_of_pooled_delta None with"),
        (lambda p: named(p).pop("share_of_pooled_delta"), "named_preempted share_of_pooled_delta '\\(missing\\)'"),
        (lambda p: head(p).update(pooled_yes_difference=0),
         "by_verdict yes share_of_pooled_delta .* with pooled_yes_difference 0: it is null exactly"),
        (lambda p: strata(p)["no"].update(share_of_pooled_delta=2 * strata(payload)["no"]["share_of_pooled_delta"]),
         "by_verdict no share_of_pooled_delta .* is not delta_contribution / pooled_delta"),
        (lambda p: named(p).update(share_of_pooled_delta=str(named(payload)["share_of_pooled_delta"])),
         "named_preempted share_of_pooled_delta '.*' is not delta_contribution / pooled_delta"),
        (lambda p: strata(p)["yes"].update(delta_contribution=strata(payload)["yes"]["delta_contribution"] + 0.01),
         "by-judgment contributions do not sum to pooled_delta"),
        (lambda p: strata(p)["yes"].update(delta_contribution="0.1"), "by_verdict yes: delta_contribution '0.1' is not"),
        (lambda p: strata(p)["yes"].update(share_of_responses=strata(payload)["yes"]["share_of_responses"] + 0.01),
         f"by_verdict yes share_of_responses .* is not n_responses / the cells = \\d+ / {N * S}"),
        (lambda p: (strata(p)["yes"].update(n_responses=strata(p)["yes"]["n_responses"] + 1),
                    strata(p)["no"].update(n_responses=strata(p)["no"]["n_responses"] - 1)),
         "by_verdict yes share_of_responses .* is not n_responses / the cells"),
        (lambda p: strata(p)["no"].update(n_responses=strata(p)["no"]["n_responses"] + 1),
         f"by_verdict n_responses \\[.*\\] are not counts summing to the {N * S} \\(scenario, draw\\) cells"),
        (lambda p: _move_responses(strata(p), "unsure", "no", strata(p)["unsure"]["n_responses"] + 1),
         r"by_verdict n_responses \[.*-1.*\] are not counts summing"),
        (lambda p: strata(p)["no"].update(n_responses=float(strata(p)["no"]["n_responses"])),
         "by_verdict no: n_responses .* is not an integer"),
        (lambda p: strata(p)["unsure"].pop("share_of_responses"), "by_verdict unsure: share_of_responses is missing"),
        (lambda p: strata(p).pop("no"), "by_verdict.strata is not an object over"),
        (lambda p: p["models"]["gpt-oss-120b"].update(by_verdict=[]), "by_verdict.strata is not an object over"),
        (lambda p: p["models"]["gpt-oss-120b"].update(named_preempted=None), "named_preempted: NoneType is not an"),
    ]
    for change, message in cases:
        _refused(payload, change, message)


def _shifted(matrix, amount):
    """``matrix`` with ``amount`` of its yes -> yes share moved to yes -> no (the shares still sum to 1)."""
    matrix["yes"]["yes"] -= amount
    matrix["yes"]["no"] += amount


def _tiny(matrix, share=1e-200):
    """``matrix`` all 0.0 but yes -> yes = ``share``."""
    for a in LABELS:
        matrix[a].update({b: 0.0 for b in LABELS})
    matrix["yes"]["yes"] = share


def _deep(levels, leaf=1):
    """``leaf`` nested in ``levels`` arrays (built without recursion; ``json.loads`` reads one about 10,000 deep)."""
    for _ in range(levels):
        leaf = [leaf]
    return leaf


def test_json_equal_and_the_depth_bound_meet_any_depth_without_recursion():
    deep = _deep(5000)
    assert lg.json_equal(deep, _deep(5000)) and not lg.json_equal(deep, _deep(5000, leaf=1.0))
    assert not lg.json_equal(deep, _deep(4999)) and not lg.json_equal(_deep(4999), deep)
    assert lg.json_equal({"a": [1, (2,)]}, {"a": (1, [2])}) and not lg.json_equal({"a": 1}, {"b": 1})
    assert not lg.json_equal([1, 2], [1]) and not lg.json_equal(True, 1) and not lg.json_equal(500.0, 500)
    assert [lg.nesting_depth(value) for value in (1, [], {"a": [1]}, ({"a": ()},))] == [0, 1, 2, 3]
    assert lg.nesting_depth(deep) == lg.MAX_DEPTH + 1
    lg.require_shallow(_deep(lg.MAX_DEPTH), "x")
    with pytest.raises(RuleContextError, match=f"^x: nested past {lg.MAX_DEPTH} levels: not a golden this pipeline"):
        lg.require_shallow(_deep(lg.MAX_DEPTH + 1), "x")


_OSS = "gpt-oss-120b"


def _block(p):
    return p["models"][_OSS]


@pytest.mark.parametrize("change, message", [
    (lambda p: p.update(metric="smrc"), "golden: metric 'smrc', not 'flat'"),
    (lambda p: p["settings"].pop("inference_method"), "inference_method None, not this module's 'hierarchical_boot"),
    (lambda p: p["settings"].update(test_method="x"), "test_method 'x', not this module's 'scenario_clustered_paired"),
    (lambda p: p["settings"].update(agreement_method="x"), "agreement_method 'x', not this module's"),
    (lambda p: p["settings"].pop("bootstrap"), r"golden: bootstrap None is not a ci_level in \(0, 1\)"),
    (lambda p: p["settings"]["bootstrap"].update(ci_level="x"), r"bootstrap \{.*'ci_level': 'x'.*\} is not a ci_level"),
    (lambda p: p["settings"]["bootstrap"].update(ci_level=10**30), "bootstrap .* is not a ci_level in"),
    (lambda p: p["settings"]["bootstrap"].update(n_resamples=0), "bootstrap .* is not a ci_level in .* positive integer"),
    (lambda p: p["settings"]["bootstrap"].pop("seed"), "bootstrap .* with an integer seed"),
    # The rules the figures label rows by, and each row's own title (the CSV's).
    (lambda p: p["rules"][0].pop("title"), "rules is not a non-empty list of objects with distinct integer numbers and "
                                           "string titles"),
    (lambda p: p["rules"].append(dict(p["rules"][0])), "rules is not a non-empty list of objects with distinct"),
    (lambda p: _block(p)["effects"][0].update(title="WRONG TITLE"), "Rule 1: title 'WRONG TITLE' is not the rules'"),
    (lambda p: _block(p)["effects"][0].pop("title"), "Rule 1: title is missing"),
    # The phi means the CSV prints: means of PHI_MAP's values, phi_delta their difference.
    (lambda p: _block(p)["effects"][0].update(phi_delta="abc"), "Rule 1: phi_delta 'abc' is not a real number"),
    (lambda p: _block(p)["effects"][0].update(phi_delta=0.9),
     r"Rule 1: phi_alone .* are not means in \[0, 1\] with phi_delta 0.9 their difference"),
    (lambda p: _block(p)["effects"][0].update(phi_alone=1.5, phi_delta=_block(p)["effects"][0]["phi_in_context"] - 1.5),
     "Rule 1: phi_alone 1.5 .* are not means in"),
    (lambda p: p["provenance"].update(campaigns=5), "provenance.campaigns is not a list of campaign records"),
    (lambda p: p["provenance"]["campaigns"][0].pop("forced_conditions"), "provenance.campaigns is not a list"),
    (lambda p: _block(p).update(model_name="gemma-4-31B-it"),
     f"{_OSS}: model_name 'gemma-4-31B-it' is not the judge CAMPAIGN_MODEL_LABELS labels '{_OSS}'"),
    (lambda p: _block(p).update(model_name=["gpt-oss-120b"]), r"model_name \['gpt-oss-120b'\] is not the judge"),
    (lambda p: _block(p).update(condition_id=f"{_OSS}/naive-rules-above"),
     f"{_OSS}: condition_id '{_OSS}/naive-rules-above' is not a forced condition that the record of its campaign"),
    (lambda p: _block(p).update(campaign_id="elsewhere"), "condition_id .* of its campaign 'elsewhere' lists"),
    # The flips the summary and the flips figure print: typed, the rows' mean, its agreement figures its own.
    (lambda p: _block(p).pop("flips"), f"{_OSS} flips: NoneType is not an object"),
    (lambda p: _block(p)["flips"].update(unchanged_fraction=0.12), "flips unchanged_fraction 0.12 is not its matrix's"),
    (lambda p: _shifted(_block(p)["flips"]["matrix"], 0.2), f"{_OSS} flips: matrix is not the mean of the rules'"),
    (lambda p: _block(p)["flips"].update(method="x"), "method 'x', not 'independent_draw_expected_shares_v1'"),
    (lambda p: _block(p)["flips"]["matrix"]["yes"].pop("no"), f"{_OSS} flips: matrix is not a"),
    (lambda p: _block(p)["flips"]["draw_to_draw"].pop("alone"), f"{_OSS} draw_to_draw alone: NoneType is not an object"),
    (lambda p: _block(p)["flips"].update(draw_to_draw=[]), f"{_OSS} flips: draw_to_draw is not an object"),
    (lambda p: _block(p)["flips"]["draw_to_draw"].update(method="x"), "draw_to_draw method 'x', not"),
    (lambda p: _block(p)["flips"]["draw_to_draw"]["in_context"].update(kappa=0.5),
     "draw_to_draw in_context kappa 0.5 is not its matrix's"),
    (lambda p: _shifted(_block(p)["flips"]["draw_to_draw"]["alone"]["matrix"], -2.0),
     "draw_to_draw alone matrix is not non-negative shares summing to 1"),
    (lambda p: _tiny(_block(p)["flips"]["draw_to_draw"]["alone"]["matrix"]),
     f"{_OSS} flips: draw_to_draw alone matrix is not non-negative shares summing to 1"),
    (lambda p: _tiny(_block(p)["flips"]["draw_to_draw"]["in_context"]["matrix"], 5e-324),
     "draw_to_draw in_context matrix is not non-negative shares summing to 1"),
    (lambda p: _tiny(_block(p)["flips"]["matrix"]), "flips matrix is not non-negative shares summing to 1"),
    (lambda p: _block(p)["flips"]["draw_to_draw"]["alone"].pop("chance_agreement"), "chance_agreement is missing"),
    # The named decomposition the summary prints: typed, its cells and contribution the rows' sum and mean.
    (lambda p: _block(p)["named_preempted"].pop("n_named_cells"), "named_preempted: n_named_cells is missing"),
    (lambda p: _block(p)["named_preempted"].update(n_named_cells=_block(p)["named_preempted"]["n_named_cells"] + 1),
     "named_preempted n_named_cells .* is not the rules' sum"),
    (lambda p: _block(p)["named_preempted"].update(
        delta_contribution=_block(p)["named_preempted"]["delta_contribution"] + 0.01),
     "named_preempted delta_contribution .* is not the rules' mean"),
    (lambda p: _block(p)["named_preempted"].pop("in_context_no_rate"), r"named_preempted in_context_no_rate '\(missing\)'"),
    (lambda p: _block(p)["named_preempted"].update(in_context_no_rate=1.5), "in_context_no_rate 1.5 is not a rate in"),
    (lambda p: _block(p)["named_preempted"].update(in_context_no_rate=None), "in_context_no_rate None is not a rate"),
    (lambda p: _block(p)["effects"][0]["named_preempted"].update(n_cells=N * S + 1),
     f"Rule 1: named_preempted n_cells {N * S + 1} is not a count of the {N * S} cells"),
    # A number past 2**53 is refused where it is typed: the arithmetic on it would raise an OverflowError.
    (lambda p: _block(p)["effects"][0].update(yes_difference=10**400),
     r"Rule 1: yes_difference 10{400} is not an integer within \+-2\*\*53"),
    (lambda p: p["settings"].update(n_scenarios=10**400), f"settings grid 10{{400}} x {S} is not two positive"),
    (lambda p: p["settings"]["randomization_test"].update(n_permutations=10**400),
     "randomization_test.n_permutations 10{400} is not a positive integer"),
    (lambda p: _block(p)["summary"].update(pooled_rate_alone=1e308),
     r"summary: pooled_rate_alone 1e\+308 is not a real number within"),
    (lambda p: _block(p)["effects"][0]["transitions"]["yes"].update(no=1e306), "transitions is not a"),
    (lambda p: _block(p)["by_verdict"]["strata"]["yes"].update(n_responses=-(10**400)), "n_responses -10{400}"),
    (lambda p: _block(p)["effects"][0].update(ci_high=float("inf")), "Rule 1: ci_high inf is not a real number"),
    # A p its settings' test cannot produce: 0, or off its lattice k / (B + 1).
    (lambda p: _block(p)["effects"][4].update(p_value=0.0),
     f"Rule 5: p_value 0.0 is not a probability on the {PERMUTATIONS}-permutation test's lattice"),
    (lambda p: _block(p)["effects"][4].update(p_value=0.123456789), "Rule 5: p_value 0.123456789 is not a probability on"),
    (lambda p: p["settings"].update(few_informative_below=float(ls.FEW_INFORMATIVE_BELOW)),
     f"few_informative_below {float(ls.FEW_INFORMATIVE_BELOW)}, not this module's {ls.FEW_INFORMATIVE_BELOW}"),
    (lambda p: p["settings"].update(holm_mc_margin_se=int(rc.HOLM_MC_MARGIN_SE)),
     f"holm_mc_margin_se {int(rc.HOLM_MC_MARGIN_SE)}, not this module's {rc.HOLM_MC_MARGIN_SE}"),
    (lambda p: p.update(data_version=float(ls.DATA_VERSION)), f"data_version {float(ls.DATA_VERSION)}, not"),
    (lambda p: p["rules"][0].update(note=_deep(1200)), f"^golden: nested past {lg.MAX_DEPTH} levels"),
    (lambda p: p.update(definition=_deep(5000)), f"^golden: nested past {lg.MAX_DEPTH} levels"),
])
def test_validate_payload_checks_every_field_its_renderers_read(campaign, change, message):
    _refused(campaign["payload"], change, message)


def _move_responses(strata, source, target, n):
    cells = sum(stratum["n_responses"] for stratum in strata.values())
    for label, step in ((source, -n), (target, n)):
        count = strata[label]["n_responses"] + step
        strata[label].update(n_responses=count, share_of_responses=count / cells)


def _without_context_effect(block):
    for row in block["effects"]:
        row.update(yes_difference=0, delta=0.0, rate_in_context=row["rate_alone"],
                   by_verdict={label: 0.0 for label in LABELS})
        row["named_preempted"]["delta_contribution"] = 0.0
    summary = block["summary"]
    summary.update(pooled_yes_difference=0, pooled_delta=0.0, pooled_rate_in_context=summary["pooled_rate_alone"],
                   **dict(zip(ls.DIRECTION_COUNT_FIELDS, ls.direction_counts(block["effects"]))))
    for part in [*block["by_verdict"]["strata"].values(), block["named_preempted"]]:
        part.update(delta_contribution=0.0, share_of_pooled_delta=None)


def test_a_golden_whose_pooled_count_is_zero_validates_only_with_null_shares(campaign):
    payload = copy.deepcopy(campaign["payload"])
    _without_context_effect(payload["models"]["gpt-oss-120b"])
    lg.validate_payload(payload)
    payload["models"]["gpt-oss-120b"]["named_preempted"]["share_of_pooled_delta"] = 0.0
    with pytest.raises(RuleContextError, match="named_preempted share_of_pooled_delta 0.0 with pooled_yes_difference 0"):
        lg.validate_payload(payload)


def test_csv_rows_carry_every_column_in_paper_order(campaign):
    rows = rc.csv_rows(campaign["payload"])
    assert len(rows) == 2 * N_RULES
    assert [row["model"] for row in rows[:: N_RULES]] == ["gpt-oss-120b", "gpt-5.6-luna"]
    assert list(rows[0]) == list(rc.CSV_COLUMNS)
    first = rows[0]
    assert first["rule"] == 1 and first["metric"] == "flat"
    assert first["yes_difference"] < 0 and isinstance(first["p_value_is_floor"], bool)
    assert isinstance(first["holm_mc_margin"], bool)
    assert sum(first[f"share_alone_{a}_in_context_{b}"] for a in LABELS for b in LABELS) == pytest.approx(1.0)
    assert sum(first[f"full_set_{v}_delta_contribution"] for v in LABELS) == pytest.approx(first["delta"])


@pytest.fixture(scope="module")
def figures():
    pytest.importorskip("matplotlib")
    from ruleforge.experiments.leniency import figure as leniency_figure

    return leniency_figure


def _effects(payload, label):
    return {row["rule"]: row for row in payload["models"][label]["effects"]}


def test_rule_order_selection_and_caption_counts(figures, campaign):
    payload = copy.deepcopy(campaign["payload"])
    order = figures.rule_order_by_mean_delta(payload)
    totals = {n: sum(_effects(payload, m)[n]["yes_difference"] for m in payload["models"]) for n in order}
    assert order == sorted(order, key=lambda n: (totals[n], n)) and order[0] == 1
    counts = figures.direction_significance_counts(payload)
    cells = [key for key in counts if not key.endswith("_mc_margin")]
    assert list(counts) == cells + [f"{cell}_mc_margin" for cell in cells]  # the six counts first, as published
    assert sum(counts[cell] for cell in cells) == 2 * N_RULES and counts["tie_not_significant"] >= 2  # Rule 2
    effects = [row for model in payload["models"].values() for row in model["effects"]]
    assert sum(counts[f"{cell}_mc_margin"] for cell in cells) == sum(row["holm_mc_margin"] for row in effects)
    _effects(payload, "gpt-oss-120b")[1]["holm_mc_margin"] = True
    _effects(payload, "gpt-5.6-luna")[2]["holm_mc_margin"] = True
    marked = figures.direction_significance_counts(payload)
    assert marked["blue_significant_mc_margin"] == counts["blue_significant_mc_margin"] + 1
    assert marked["tie_not_significant_mc_margin"] == counts["tie_not_significant_mc_margin"] + 1
    assert {k: v for k, v in marked.items() if k in cells} == {k: v for k, v in counts.items() if k in cells}
    for label in payload["models"]:
        for row in payload["models"][label]["effects"]:
            yes = -row["rule"] if row["rule"] != 7 else 2
            row.update(significant_holm=row["rule"] in (1, 4, 5, 6), yes_difference=yes, delta=yes / (N * S))
    _effects(payload, "gpt-5.6-luna")[7]["significant_holm"] = True
    assert figures.main_text_rule_order(payload, n_blue=2) == [6, 5, 7]
    assert figures.main_text_rule_order(payload) == [6, 5, 4, 1, 7]
    for label in payload["models"]:
        for row in payload["models"][label]["effects"]:
            row["significant_holm"] = False
    assert figures.main_text_rule_order(payload) == figures.rule_order_by_mean_delta(payload)


def test_exactly_tied_rules_are_ordered_by_number_whatever_the_model_order(figures):
    counts = {24: (-353, -202, -131, -371), 50: (-472, -532, -55, 2), 2: (-32, -135, -131, -78),
              39: (-107, -24, -162, -83)}
    deltas = {24: (-0.1412, -0.0808, -0.0524, -0.14839999999999998),
              50: (-0.18880000000000002, -0.2128, -0.02200000000000002, 0.0007999999999999674),
              2: (-0.012799999999999999, -0.054, -0.05239999999999999, -0.031200000000000002),
              39: (-0.0428, -0.009600000000000001, -0.06480000000000002, -0.03319999999999999)}
    labels = ["Olmo3-32B", "gemma4-31b", "gpt-5.6-luna", "gpt-oss-120b"]
    float_mean = {n: sum(deltas[n]) / 4 for n in counts}
    assert float_mean[50] < float_mean[24] and float_mean[39] < float_mean[2]  # noise decides a float key

    def payload(model_order):
        return {"models": {labels[m]: {"effects": [
            {"rule": n, "yes_difference": counts[n][m], "delta": deltas[n][m], "n_scenarios": 500,
             "n_decoder_draws": 5} for n in counts]} for m in model_order}}

    for model_order in itertools.permutations(range(4)):
        assert figures.rule_order_by_mean_delta(payload(model_order)) == [24, 50, 2, 39]


def test_counts_and_colours_read_a_float_noise_tie_as_a_tie(figures):
    from ruleforge.experiments import style as theme

    row = {"delta": -2.168e-19, "yes_difference": 0, "degenerate": False, "significant_holm": False,
           "holm_mc_margin": False}
    assert figures.mark_color(row) == theme.TICK_COLOR
    counts = figures.direction_significance_counts({"models": {"m": {"effects": [row]}}})
    assert counts["tie_not_significant"] == 1 and counts["blue_not_significant"] == 0


def test_the_forest_draws_one_mark_per_model_and_rule(figures, campaign):
    import matplotlib.pyplot as plt

    payload = copy.deepcopy(campaign["payload"])
    luna = _effects(payload, "gpt-5.6-luna")
    luna[5].update(degenerate=True, delta=0.05, ci_low=0.05, ci_high=0.05, significant_holm=True)
    luna[1]["holm_mc_margin"] = True
    _effects(payload, "gpt-oss-120b")[4].update(holm_mc_margin=True, significant_holm=False)
    figure = figures.build_figure(payload, compact=False)
    try:
        assert len(figure.axes) == 2
        assert [ax.get_title() for ax in figure.axes] == [
            "GPT-OSS-120B", "GPT-5.6 Luna",
        ]
        order = figures.rule_order_by_mean_delta(payload)
        for ax, label in zip(figure.axes, payload["models"]):
            effects = _effects(payload, label)
            assert len(ax.collections) == N_RULES and len(ax.lines) == N_RULES + 1  # whiskers + zero line
            for marker, whisker, number in zip(ax.collections, ax.lines[1:], order):
                row = effects[number]
                assert marker.get_offsets()[0, 0] == pytest.approx(row["delta"])
                filled = tuple(marker.get_facecolor()[0]) == tuple(marker.get_edgecolor()[0])
                assert filled == row["significant_holm"]
                faded = figures.FEW_INFORMATIVE_ALPHA if row["few_informative"] else None
                assert marker.get_alpha() == whisker.get_alpha() == faded
                assert (whisker.get_linestyle() != "-") == row["few_informative"]
            assert not ax.texts
        # RGB only: a faded (few-informative) marker carries its alpha too.
        colors = {number: tuple(m.get_edgecolor()[0][:3]) for m, number in zip(figure.axes[1].collections, order)}
        from matplotlib.colors import to_rgba

        from ruleforge.experiments import style as theme

        assert colors[5] == to_rgba(theme.TICK_COLOR)[:3]
        assert colors[2] == to_rgba(theme.TICK_COLOR)[:3]
        assert colors[1] == to_rgba(theme.DARK_BLUE)[:3]
        labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
        assert labels[0].startswith("R1 · ")
        long = next(label for label in labels if label.startswith("R3 · "))
        assert long.endswith("…") and len(long) <= len("R3 · ") + figures.TITLE_CHARS
        long_title = "Be thorough but efficient, while respecting length limits"
        assert figures.tick_label(15, long_title) == "R15 · Be thorough but efficient…"
        assert figures.tick_label(9, "x" * 40) == "R9 · " + "x" * (figures.TITLE_CHARS - 1) + "…"
        assert figures.tick_label(10, "Be warm") == "R10 · Be warm"
        legend = [t.get_text() for t in figure.legends[0].get_texts()]
        assert legend[-1] == "95% scenario-cluster bootstrap CI" and legend[0] == "Lower in full ruleset"
        assert any(row["few_informative"] for row in luna.values())
        assert [t.get_text() for t in figure.legends[1].get_texts()] == ["Fewer than 10 informative scenarios"]
        assert figures.FIGURE_TITLE in [t.get_text() for t in figure.texts]
        assert len({ax.get_xlim() for ax in figure.axes}) == 1
    finally:
        plt.close(figure)
    for label in payload["models"]:
        for row in payload["models"][label]["effects"]:
            row["few_informative"] = False
    selected = figures.main_text_rule_order(payload)
    _effects(payload, "gpt-oss-120b")[4]["holm_mc_margin"] = 4 in selected
    assert 1 in selected and luna[1]["holm_mc_margin"]
    compact = figures.build_figure(payload)
    try:
        assert all(len(ax.collections) == len(selected) for ax in compact.axes)
        assert compact.get_figheight() == figures.MIN_HEIGHT
        grid = (compact.axes[0].get_position().x0 + compact.axes[-1].get_position().x1) / 2
        title = next(t for t in compact.texts if t.get_text() == figures.FIGURE_TITLE)
        anchor = compact.legends[0].get_bbox_to_anchor().transformed(compact.transFigure.inverted())
        assert grid == pytest.approx(figures.PANEL_CENTER)
        assert title.get_position()[0] == figures.HEADER_CENTER == 0.5
        assert compact._supxlabel.get_position()[0] == figures.PANEL_CENTER
        assert anchor.x0 == pytest.approx(figures.HEADER_CENTER)
        assert title.get_fontsize() == figures.TITLE_FONTSIZE
        assert len(compact.legends) == 1
        assert not any("Monte Carlo" in t.get_text() for t in compact.texts)
    finally:
        plt.close(compact)
    for label in payload["models"]:
        for row in payload["models"][label]["effects"]:
            row["holm_mc_margin"] = False
    bare = figures.build_figure(payload)
    try:
        assert len(bare.legends) == 1 and not any(ax.texts for ax in bare.axes)
    finally:
        plt.close(bare)


def test_the_flips_figure_prints_shares_agreement_and_references(figures, campaign):
    import matplotlib.pyplot as plt

    payload = campaign["payload"]
    figure = figures.build_flips_figure(payload)
    try:
        assert len(figure.axes) == 2
        for ax, label in zip(figure.axes, payload["models"]):
            flips = payload["models"][label]["flips"]
            assert ax.get_title().endswith(f"{flips['unchanged_fraction']:.0%} unchanged")
            drawn = [t.get_text() for t in ax.texts]
            assert f"{100 * flips['matrix']['no']['no']:.1f}%" in drawn
            header = next(text for text in drawn if text.startswith("floor "))
            assert f"κ {flips['kappa']:.2f}" in header
            assert f"alone {flips['draw_to_draw']['alone']['unchanged_fraction']:.0%}" in header
            assert ax.get_xlabel() == "in full ruleset" and ax.get_ylabel() == "judged alone"
        assert figures.FLIPS_FOOTER in [t.get_text() for t in figure.texts]
    finally:
        plt.close(figure)
    assert figures._share_text(0.0) == "0.0%" and figures._share_text(0.0004) == "<0.1%"
    assert figures._share_text(0.123) == "12.3%"


def test_draw_writes_png_and_pdf(figures, campaign, tmp_path):
    payload = campaign["payload"]
    figures.draw(payload, tmp_path / "f.png", dpi=40)
    figures.draw(payload, tmp_path / "all.pdf", compact=False)
    figures.draw_flips(payload, tmp_path / "flips.png", dpi=40)
    for name in ("f.png", "all.pdf", "flips.png"):
        assert (tmp_path / name).stat().st_size > 1_000
    assert b"/Type3" not in (tmp_path / "all.pdf").read_bytes()


@pytest.fixture(scope="module")
def driver(repo_root):
    module = importlib.import_module("ruleforge.experiments.leniency.build")
    return module


def test_main_writes_every_output_and_re_renders_from_the_golden(driver, campaign, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    golden, csv1, csv2 = tmp_path / "g.json", tmp_path / "1.csv", tmp_path / "2.csv"
    with artifact_root(campaign["base"]):
        assert driver.main([
            "--estimates", str(campaign["open"]), "--estimates", str(campaign["luna"]), "--prompt-style", "forced-grpo",
            "--rules", str(campaign["rules"]), "--golden", str(golden), "--csv", str(csv1),
            "--figure", str(tmp_path / "c.png"), "--figure_all", str(tmp_path / "a.pdf"),
            "--figure_flips", str(tmp_path / "f.png"), "--dpi", "40",
        ]) == 0
    out = capsys.readouterr().out
    assert "pooled delta" in out and "draws agree alone" in out and "compact rules" in out
    assert "of them significant); few informative" in out and "'blue_significant_mc_margin': " in out
    assert "judged 'no' in context on n/a" in out
    assert "prompt style" not in out
    stored = json.loads(golden.read_text())
    assert stored == json.loads(json.dumps(campaign["payload"]))
    header = csv1.read_text().splitlines()[0].split(",")
    assert header == list(driver.CSV_COLUMNS)
    assert ",true," in csv1.read_text() and "True" not in csv1.read_text()
    assert all((tmp_path / name).stat().st_size > 1_000 for name in ("c.png", "a.pdf", "f.png"))
    assert driver.main(["--from_golden", str(golden), "--csv", str(csv2)]) == 0
    assert csv1.read_bytes() == csv2.read_bytes()
    assert capsys.readouterr().out == out
    assert "by judgment full=yes" in out  # label order, not the golden's sort_keys order


def test_main_refuses_mixed_sources_stale_and_scratch_goldens(driver, campaign, tmp_path):
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps(campaign["payload"]))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden), "--estimates", str(campaign["open"])])
    with pytest.raises(SystemExit):
        driver.main([])
    stale = tmp_path / "v4.json"
    stale.write_text(json.dumps(dict(campaign["payload"], data_version=4)))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(stale)])
    scratch = tmp_path / "scratch.json"
    scratch.write_text(json.dumps(dict(campaign["payload"], provenance=dict(campaign["payload"]["provenance"], complete=False))))
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(scratch)])
    assert driver.main(["--from_golden", str(scratch), "--allow-incomplete"]) == 0
    short_rules = write_rules(tmp_path / "x.jsonl", RULES[:3])
    with pytest.raises(SystemExit):
        driver.main(["--estimates", str(campaign["open"]), "--rules", str(short_rules), "--prompt-style", "forced-grpo"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda row: row.update(n_scenarios=500.0), "n_scenarios 500.0 is not an integer"),
        (lambda row: row.update(delta="-0.1"), "delta '-0.1' is not a real number"),
        (lambda row: row.update(few_informative=1), "few_informative 1 is not a boolean"),
        (lambda row: row.pop("p_value"), "p_value is missing"),
    ],
)
def test_main_refuses_a_mistyped_golden_as_a_usage_error(driver, campaign, tmp_path, capsys, change, message):
    payload = copy.deepcopy(campaign["payload"])
    change(payload["models"]["gpt-oss-120b"]["effects"][4])
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as stopped:
        driver.main(["--from_golden", str(golden)])
    assert stopped.value.code == 2 and message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p["models"]["gpt-oss-120b"].pop("flips"), "flips: NoneType is not an object"),
        (lambda p: p["models"]["gpt-oss-120b"]["named_preempted"].pop("n_named_cells"), "n_named_cells is missing"),
        (lambda p: p["rules"][0].pop("title"), "rules is not a non-empty list"),
        (lambda p: p.pop("metric"), "metric None, not 'flat'"),
        (lambda p: p["provenance"].update(campaigns=5), "provenance.campaigns is not a list"),
        (lambda p: p["models"]["gpt-oss-120b"]["effects"][0].update(yes_difference=10**400), "is not an integer within"),
        (lambda p: _tiny(p["models"]["gpt-oss-120b"]["flips"]["draw_to_draw"]["alone"]["matrix"]),
         "draw_to_draw alone matrix is not non-negative shares summing to 1"),
        (lambda p: p["rules"][0].update(note=_deep(1200)), f"golden: nested past {lg.MAX_DEPTH} levels"),
    ],
)
def test_main_refuses_a_golden_its_renderers_would_crash_on_as_a_usage_error(
    driver, campaign, tmp_path, capsys, change, message
):
    payload = copy.deepcopy(campaign["payload"])
    change(payload)
    golden = tmp_path / "g.json"
    golden.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as stopped:
        driver.main(["--from_golden", str(golden), "--csv", str(tmp_path / "x.csv")])
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "x.csv").exists()


UNREADABLE = {
    "RecursionError": ("[" * 20_000 + "]" * 20_000).encode(),
    "JSONDecodeError": b'{"data_version": 5,',
    "UnicodeDecodeError": b'{"source": "\xff"}',
    "ValueError": b'{"data_version": ' + b"9" * 5_000 + b"}",
    "UnicodeEncodeError": b'{"source": "\\ud800"}',
}


def unreadable(tmp_path, kind):
    """A file json.loads meets as ``kind`` (:data:`UNREADABLE`), and the refusal lg.read_golden names it with."""
    path = tmp_path / f"unreadable_{kind}.json"
    path.write_bytes(UNREADABLE[kind])
    return path, rf"{re.escape(str(path))} cannot be read as JSON \({kind}: "


@pytest.mark.parametrize("kind", UNREADABLE)
def test_read_golden_refuses_a_file_json_cannot_read_naming_it(tmp_path, kind):
    class Refusal(RuleContextError):
        pass

    path, message = unreadable(tmp_path, kind)
    with pytest.raises(Refusal, match=f"^somewhere: {message}"):
        lg.read_golden(path, "somewhere", error=Refusal)
    with pytest.raises(RuleContextError, match=f"^somewhere: {message}") as refused:
        lg.read_golden(str(path), "somewhere")
    assert type(refused.value) is RuleContextError
    readable = tmp_path / "readable.json"
    readable.write_text('{"a": [1, 2.5, "x"]}')
    assert lg.read_golden(readable, "somewhere") == {"a": [1, 2.5, "x"]}


def test_read_golden_refuses_a_missing_file_naming_it(tmp_path):
    missing = tmp_path / "no_such_golden.json"
    with pytest.raises(RuleContextError, match=rf"^somewhere: {re.escape(str(missing))} cannot be read as JSON "
                                               r"\(FileNotFoundError: "):
        lg.read_golden(missing, "somewhere")


@pytest.mark.parametrize("kind", UNREADABLE)
def test_main_refuses_a_golden_json_cannot_read_as_a_usage_error(driver, tmp_path, capsys, kind):
    path, message = unreadable(tmp_path, kind)
    with pytest.raises(SystemExit) as stopped:
        driver.main(["--from_golden", str(path), "--csv", str(tmp_path / "x.csv")])
    assert stopped.value.code == 2 and re.search(f"error: golden: {message}", capsys.readouterr().err)
    assert not (tmp_path / "x.csv").exists()
    with pytest.raises(RuleContextError, match=f"^golden: {message}"):
        driver.load_golden(path, allow_incomplete=False)


def test_the_build_and_a_golden_re_read_render_rules_in_one_order(driver, figures, campaign, tmp_path):
    import matplotlib.pyplot as plt

    golden = tmp_path / "g.json"
    with artifact_root(campaign["base"]):
        driver.main(["--estimates", str(campaign["open"]), "--estimates", str(campaign["luna"]),
                     "--rules", str(campaign["rules"]), "--golden", str(golden), "--prompt-style", "forced-grpo"])
    built, reread = campaign["payload"], driver.load_golden(golden, allow_incomplete=False)
    assert list(built["models"]) != list(reread["models"])
    assert figures.rule_order_by_mean_delta(built) == figures.rule_order_by_mean_delta(reread)
    assert figures.main_text_rule_order(built) == figures.main_text_rule_order(reread)
    for compact in (True, False):
        one, two = figures.build_figure(built, compact=compact), figures.build_figure(reread, compact=compact)
        try:
            assert [t.get_text() for t in one.axes[0].get_yticklabels()] == [
                t.get_text() for t in two.axes[0].get_yticklabels()]
            assert [ax.get_title() for ax in one.axes] == [ax.get_title() for ax in two.axes]
        finally:
            plt.close(one)
            plt.close(two)


def test_main_turns_refusals_from_the_layers_below_into_usage_errors(driver, scratch, capsys):
    from ruleforge.core.errors import SamplingDesignError
    from ruleforge.evaluate.ledger import LedgerError
    from ruleforge.evaluate.runner import CampaignRunError

    assert issubclass(LedgerError, SamplingDesignError) and issubclass(CampaignRunError, SamplingDesignError)
    estimates = build_root(scratch, "a")
    root = estimates.parent.parent
    draw_dir = root / "gpt-oss-120b" / "forced-grpo" / "draw-001"
    argv = ["--estimates", str(estimates), "--rules", str(scratch / "rules.jsonl"), "--prompt-style", "forced-grpo"]

    def refused(message):
        with pytest.raises(SystemExit) as stopped:
            driver.main(argv)
        assert stopped.value.code == 2 and message in capsys.readouterr().err

    _edit_json(draw_dir / "completion.json", lambda p: p.update(state="running"))
    refused("completion state is 'running'")
    _edit_json(draw_dir / "completion.json", lambda p: p.update(state="complete"))
    _edit_jsonl(draw_dir / "attempts.jsonl", lambda rows: rows + [rows[-1]])
    refused("after it was already accepted")
    (root / "sampling_manifest.json").write_text("{")
    refused("cannot read sampling manifest")


#: A forced style whose contract asks for no preempted_rules.
UNLISTED = "forced"


@pytest.fixture(scope="module")
def both_styles(tmp_path_factory):
    base = tmp_path_factory.mktemp("both")
    with artifact_root(base):
        estimates = build_root(base, "both", model="gemma-4-31B-it", styles=("forced-grpo", UNLISTED))
        goldens = {style: rc.build_payload([estimates], base / "rules.jsonl", prompt_style=style)
                   for style in ("forced-grpo", UNLISTED)}
    return {"base": base, "estimates": estimates, "goldens": goldens}


def test_a_golden_of_the_paper_s_forced_style_names_it_in_its_flips_figure_only(figures, both_styles):
    import matplotlib.pyplot as plt

    paper, other = both_styles["goldens"][UNLISTED], both_styles["goldens"]["forced-grpo"]
    assert figures.style_note(paper) == "forced prompt"
    assert figures.style_note(other) == ""
    for golden in (paper, other):
        for compact in (True, False):
            figure = figures.build_figure(golden, compact=compact)
            assert [ax.get_title() for ax in figure.axes] == ["Gemma-4-31B"]
            plt.close(figure)
    texts = {style: [text.get_text() for text in figures.build_flips_figure(golden).texts]
             for style, golden in both_styles["goldens"].items()}
    assert f"{figures.FLIPS_FIGURE_TITLE} (forced prompt)" in texts[UNLISTED]
    assert figures.FLIPS_FIGURE_TITLE in texts["forced-grpo"]
    plt.close("all")


def test_a_root_with_both_forced_styles_gives_one_golden_per_style(both_styles):
    estimates = json.loads(both_styles["estimates"].read_text())
    assert sorted(estimates["forced_conditions"]) == sorted(["gemma-4-31B-it/forced-grpo", f"gemma-4-31B-it/{UNLISTED}"])
    for style, golden in both_styles["goldens"].items():
        assert golden["settings"]["prompt_style"] == style
        assert list(golden["models"]) == ["gemma4-31b"]
        assert golden["models"]["gemma4-31b"]["condition_id"] == f"gemma-4-31B-it/{style}"
        assert golden["provenance"]["campaigns"][0]["forced_conditions"] == [f"gemma-4-31B-it/{style}"]
        assert golden["provenance"]["definition"] == rc.definition(style)
        lg.validate_payload(golden)
        rows = {row["rule_id"]: row for row in estimates["forced_conditions"][f"gemma-4-31B-it/{style}"]["rule_context"]}
        for effect in golden["models"]["gemma4-31b"]["effects"]:
            assert (effect["delta"], effect["p_holm"]) == (rows[effect["rule_id"]]["estimate"], rows[effect["rule_id"]]["p_holm"])
    assert both_styles["goldens"]["forced-grpo"]["provenance"]["definition"] == rc.DEFINITION


def test_the_preemption_decomposition_is_not_applicable_without_preemption_never_zero(both_styles):
    control = both_styles["goldens"]["forced-grpo"]["models"]["gemma4-31b"]
    block = both_styles["goldens"][UNLISTED]["models"]["gemma4-31b"]
    assert control["named_preempted"]["n_named_cells"] > 0
    assert all(row["named_preempted"] is not None for row in control["effects"])
    assert block["named_preempted"] is None
    assert all(row["named_preempted"] is None for row in block["effects"])
    assert set(block) == set(control) and set(block["effects"][0]) == set(control["effects"][0])
    assert sum(block["by_verdict"]["strata"][v]["delta_contribution"] for v in LABELS) == pytest.approx(
        block["summary"]["pooled_delta"])
    definition = rc.definition(UNLISTED)
    assert f"({UNLISTED}, complete_rule_map)" in definition and "not applicable, never zero" in definition
    assert "listed r under preempted_rules" not in definition and "(forced-grpo," not in definition
    rows = rc.csv_rows(both_styles["goldens"][UNLISTED])
    assert {row["named_preempted_delta_contribution"] for row in rows} == {None}
    assert all(isinstance(row["named_preempted_delta_contribution"], float)
               for row in rc.csv_rows(both_styles["goldens"]["forced-grpo"]))


def test_the_reader_reads_no_preemption_list_for_a_style_that_asks_for_none(both_styles):
    root = both_styles["estimates"].parent.parent
    manifest = ScenarioManifest.load(root / "scenario_manifest.json")
    common = dict(campaign_id="both", scenario_ids=manifest.scenario_ids, n_draws=S,
                  manifest_sha256=manifest.manifest_sha256, rules=rc.load_rules(both_styles["base"] / "rules.jsonl"))
    listed, _ = rc.read_condition(root, "gemma-4-31B-it/forced-grpo", **common)
    unlisted, _ = rc.read_condition(root, "gemma-4-31B-it/forced-grpo", reads_preemption=False, **common)
    assert listed.named.any() and listed.unresolved_preempted_ids == 2 * 4 * S
    assert unlisted.named is None and unlisted.unresolved_preempted_ids is None
    assert (unlisted.alone == listed.alone).all() and (unlisted.in_context == listed.in_context).all()
    assert rc.named_decomposition(unlisted, -0.1, -3) is None


def test_a_golden_s_preemption_decomposition_must_follow_its_style(both_styles):
    control, other = both_styles["goldens"]["forced-grpo"], both_styles["goldens"][UNLISTED]

    def block(p):
        return p["models"]["gemma4-31b"]

    _refused(other, lambda p: block(p).update(named_preempted=block(control)["named_preempted"]),
             f"gemma4-31b named_preempted: .* is not null: the {UNLISTED} prompt asks for no preempted_rules")
    _refused(other, lambda p: block(p)["effects"][3].update(named_preempted={"n_cells": 0, "delta_contribution": 0.0}),
             f"gemma4-31b Rule 4 named_preempted: .* is not null: the {UNLISTED} prompt")
    _refused(other, lambda p: block(p)["effects"][3].pop("named_preempted"),
             "gemma4-31b Rule 4 named_preempted: is missing \\(null where the prompt asks for no preempted_rules\\)")
    _refused(other, lambda p: block(p).pop("named_preempted"), "gemma4-31b named_preempted: is missing")
    _refused(control, lambda p: block(p)["effects"][3].update(named_preempted=None),
             "gemma4-31b Rule 4 named_preempted: NoneType is not an object")
    _refused(control, lambda p: block(p)["effects"][3]["named_preempted"].update(n_cells=1.0),
             "Rule 4 named_preempted: n_cells 1.0 is not an integer")
    # A golden over the other style's rows would be read under the wrong rules.
    _refused(other, lambda p: p["settings"].update(prompt_style="forced-grpo"), "named_preempted: None")
    for style in ("encouraged", None):
        _refused(control, lambda p, style=style: p["settings"].update(prompt_style=style),
                 f"prompt_style {style!r} is not one of the forced styles")


def test_only_a_forced_style_builds_a_golden_and_a_root_without_it_is_refused(scratch):
    estimates = build_root(scratch, "a")
    with pytest.raises(RuleContextError, match="prompt_style 'encouraged' is not one of"):
        rc.build_payload([estimates], scratch / "rules.jsonl", prompt_style="encouraged")
    with pytest.raises(RuleContextError, match=f"no forced \\({UNLISTED}, complete_rule_map\\) condition"):
        rc.build_payload([estimates], scratch / "rules.jsonl", prompt_style=UNLISTED)


def test_the_definition_s_derivation_refuses_a_passage_it_no_longer_finds(monkeypatch):
    monkeypatch.setattr(rc, "DEFINITION", "a published definition reworded")
    with pytest.raises(ValueError, match="expected exactly one occurrence of '\\(forced-grpo, complete_rule_map\\)', found 0"):
        rc._no_preemption_definition(UNLISTED)
    monkeypatch.undo()
    assert set(rc.DEFINITIONS) == set(FORCED_JSON_STYLES) and rc.DEFINITIONS["forced-grpo"] == rc.DEFINITION


def test_main_builds_and_re_renders_a_golden_of_another_forced_style(driver, both_styles, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    golden, first, second = tmp_path / "g.json", tmp_path / "1.csv", tmp_path / "2.csv"
    with artifact_root(both_styles["base"]):
        assert driver.main(["--estimates", str(both_styles["estimates"]), "--rules", str(both_styles["base"] / "rules.jsonl"),
                            "--prompt-style", UNLISTED, "--golden", str(golden), "--csv", str(first)]) == 0
    printed = capsys.readouterr().out
    assert "named: not applicable (the prompt asks for no preempted_rules)" in printed
    assert "prompt style: forced prompt (forced)" in printed
    assert driver.main(["--from_golden", str(golden), "--csv", str(second), "--prompt-style", UNLISTED]) == 0
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(SystemExit):
        driver.main(["--from_golden", str(golden), "--prompt-style", "forced-grpo"])
    assert f"holds '{UNLISTED}' conditions, not --prompt-style 'forced-grpo'" in capsys.readouterr().err
    header, row = first.read_text().splitlines()[:2]
    assert row.split(",")[header.split(",").index("named_preempted_delta_contribution")] == ""
    assert json.loads(golden.read_text()) == json.loads(json.dumps(both_styles["goldens"][UNLISTED]))


def test_a_build_takes_the_paper_s_forced_style_unless_told_another(driver, both_styles, tmp_path):
    golden = tmp_path / "g.json"
    with artifact_root(both_styles["base"]):
        assert driver.main(["--estimates", str(both_styles["estimates"]), "--rules", str(both_styles["base"] / "rules.jsonl"),
                            "--golden", str(golden)]) == 0
    assert json.loads(golden.read_text()) == json.loads(json.dumps(both_styles["goldens"]["forced"]))


def test_a_golden_of_another_style_is_refused_anywhere_among_the_paper_s_results(driver, both_styles, tmp_path, capsys,
                                                                                monkeypatch):
    monkeypatch.setattr(driver, "repository_root", lambda: tmp_path)
    golden = tmp_path / "results" / "leniency" / "leniency.json"
    argv = ["--estimates", str(both_styles["estimates"]), "--rules", str(both_styles["base"] / "rules.jsonl"),
            "--golden", str(golden)]
    with artifact_root(both_styles["base"]), pytest.raises(SystemExit):
        driver.main([*argv, "--prompt-style", "forced-grpo"])
    assert "hold(s) the paper's files; this output is not one of them" in capsys.readouterr().err
    assert not golden.exists()
    with artifact_root(both_styles["base"]):
        assert driver.main([*argv, "--prompt-style", "forced"]) == 0
    assert golden.is_file()


REVISED_STYLES = ("naive", "encouraged", "forced")


@pytest.fixture(scope="module")
def revised_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("revised")
    with artifact_root(base):
        estimates = build_root(base, "revised", model="gemma-4-31B-it", styles=REVISED_STYLES)
        golden = rc.build_payload([estimates], base / "rules.jsonl", prompt_style="forced")
    return {"base": base, "estimates": estimates, "golden": golden}


def test_the_revised_forced_arm_builds_its_own_golden_with_no_preemption_decomposition(revised_root, figures):
    import matplotlib.pyplot as plt

    estimates = json.loads(revised_root["estimates"].read_text())
    assert sorted(estimates["forced_conditions"]) == ["gemma-4-31B-it/forced"]
    golden = revised_root["golden"]
    lg.validate_payload(golden)
    block = golden["models"]["gemma4-31b"]
    assert golden["settings"]["prompt_style"] == "forced" and block["condition_id"] == "gemma-4-31B-it/forced"
    assert golden["provenance"]["campaigns"][0]["forced_conditions"] == ["gemma-4-31B-it/forced"]
    assert block["named_preempted"] is None and all(row["named_preempted"] is None for row in block["effects"])
    definition = rc.definition("forced")
    assert golden["provenance"]["definition"] == definition
    assert "(forced, complete_rule_map)" in definition and "not applicable, never zero" in definition
    assert definition == rc.DEFINITIONS["forced"] == rc._no_preemption_definition("forced")
    assert figures.style_note(golden) == "forced prompt"
    figure = figures.build_figure(golden)
    assert [ax.get_title() for ax in figure.axes] == ["Gemma-4-31B"]
    plt.close(figure)
    flips = figures.build_flips_figure(golden)
    assert [text.get_text() for text in flips.texts if "(v2)" in text.get_text()] == []
    assert any(text.get_text().endswith("(forced prompt)") for text in flips.texts)
    plt.close(flips)
    with pytest.raises(RuleContextError, match="prompt_style 'encouraged' is not one of"):
        rc.build_payload([revised_root["estimates"]], revised_root["base"] / "rules.jsonl", prompt_style="encouraged")


def test_main_builds_the_revised_forced_golden_as_the_published_one_is_built(driver, revised_root, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    golden = tmp_path / "g.json"
    with artifact_root(revised_root["base"]):
        assert driver.main(["--estimates", str(revised_root["estimates"]), "--rules", str(revised_root["base"] / "rules.jsonl"),
                            "--prompt-style", "forced", "--golden", str(golden),
                            "--csv", str(tmp_path / "g.csv"), "--figure", str(tmp_path / "g.png")]) == 0
    assert "prompt style: forced prompt (forced)" in capsys.readouterr().out
    assert json.loads(golden.read_text()) == json.loads(json.dumps(revised_root["golden"]))
    assert (tmp_path / "g.png").stat().st_size > 0


@pytest.mark.parametrize("style", ["forced-grpo", UNLISTED])
def test_a_recorded_root_and_a_public_one_give_one_golden(tmp_path_factory, style):
    goldens, roots = {}, {}
    for recorded in (False, True):
        base = tmp_path_factory.mktemp(f"recorded-{recorded}")
        with artifact_root(base):
            estimates = build_root(base, "revised", model="gemma-4-31B-it", styles=(style,), recorded=recorded,
                                   preempt=rc.lists_preemption(style))
            goldens[recorded] = rc.build_payload([estimates], base / "rules.jsonl", prompt_style=style)
            roots[recorded] = json.loads(estimates.read_text())
    assert sorted(roots[True]["forced_conditions"]) == [recorded_id("gemma-4-31B-it", style)] != [
        f"gemma-4-31B-it/{style}"]
    assert without_estimates_files(goldens[True]) == without_estimates_files(goldens[False])
    block = goldens[True]["models"]["gemma4-31b"]
    assert block["condition_id"] == f"gemma-4-31B-it/{style}" == goldens[True]["provenance"]["campaigns"][0][
        "forced_conditions"][0]
    assert goldens[True]["settings"]["prompt_style"] == style
    lg.validate_payload(goldens[True])
