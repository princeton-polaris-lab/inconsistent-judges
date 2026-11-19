from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from ruleforge.core.artifacts import load_jsonl_records, write_jsonl_records
from ruleforge.core.digests import sha256_file
from ruleforge.grpo import report as rp
from ruleforge.grpo import report_tables as rt

REPO = Path(__file__).resolve().parents[1]
RULES = REPO / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"
SETTINGS = REPO / "configs" / "training" / "report.yaml"
#: Each published run: its campaign prefix, its base model, and the published save with what it records of G4.
RUNS = {"gemma-grpo-mix-2026-10-04": ("gemma-dev-2026-10-01", "gemma-4-31B-it", "600", -0.0530, 0.0227),
        "qwen-grpo-mix-2026-10-11": ("grpo-v2-dev-a25-2026-09-26", "Qwen3-8B", "300", -0.2763, 0.0395)}
#: What the runs' calibration records holds that the report does not compute: no save ran the second-stage campaign.
NOT_COMPUTED = {"or_yes_share", "own_rules_shift", "rule_yes_rate", "rule_agreement", "rule_invalid_rate", "rule_length_rate"}
TOLERANCE_KEYS = ("name", "gate", "stage", "direction", "definition", "subsets", "scenarios", "splits", "resamples",
                  "quantile", "null")


def differences(got: Any, want: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Return where `got` is not exactly `want`."""
    if isinstance(want, Mapping) and isinstance(got, Mapping):
        keys = sorted(set(want) | set(got))
        return [row for key in keys for row in differences(got.get(key, "<absent>"), want.get(key, "<absent>"),
                                                           f"{path}/{key}")]
    if isinstance(want, list) and isinstance(got, list) and len(want) == len(got):
        return [row for i, (g, w) in enumerate(zip(got, want)) for row in differences(g, w, f"{path}/{i}")]
    return [] if got == want else [(path, got, want)]


def tolerance_differences(tolerances: Mapping[str, Any], recorded: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return every tolerance's differences from a run's calibration record."""
    found = differences({key: tolerances[key] for key in ("inputs", "reference_identity", "scenarios", "stage1_scenarios",
                                                          "probe_scenarios")},
                        {key: recorded[key] for key in ("inputs", "reference_identity", "scenarios", "stage1_scenarios",
                                                        "probe_scenarios")})
    found += differences(tolerances["base_m1"], recorded["base"]["m1"], "/base_m1")
    found += differences(tolerances["thinking"]["base_median_think_tokens"], recorded["g7"]["base_median_think_tokens"],
                         "/thinking/median")
    found += differences(tolerances["thinking"]["base_empty_think_share"], recorded["g7"]["base_empty_think_share"],
                         "/thinking/empty")
    statistics = {name: entry for name, entry in recorded["statistics"].items() if name not in NOT_COMPUTED}
    found += differences(sorted(tolerances["statistics"]), sorted(statistics), "/statistics")
    for name, entry in statistics.items():
        got = tolerances["statistics"][name]
        floor = entry["floor"] and {"n": entry["floor"]["n"], "value": entry["floor"]["value"]}
        found += differences({**{key: got[key] for key in TOLERANCE_KEYS}, "tolerance": got["tolerance"], "floor": got["floor"],
                              "floored": got["floored"]},
                             {**{key: entry[key] for key in TOLERANCE_KEYS}, "tolerance": entry["delta"], "floor": floor,
                              "floored": name in recorded["floored"]}, f"/statistics/{name}")
    return found


def save_differences(save: Mapping[str, Any], recorded: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return one save's differences from its `results/checkpoint-N.json`."""
    delta = recorded["delta_m1_vs_base"]
    want = {"m1": recorded["m1"], "base_m1": recorded["base_m1"],
            "delta_m1_vs_base": {key: delta[key] for key in ("value", "paired_mean", "ci95", "scenarios")},
            "gates": recorded["gates"], "tables": recorded["tables"],
            "model_weights_sha256": recorded["identity"]["weights_sha256"], "adapter_sha256": None,
            "serving_dtype": recorded["identity"]["serving_dtype"],
            "outside_tolerance": [gate for gate in rp.GATES if not recorded["gates"][gate]], "statistics": {}}
    for name, entry in recorded["statistics"].items():
        kept = {key: value for key, value in entry.items() if key not in ("passed", "gating")}
        want["statistics"][name] = {**kept, "within_tolerance": entry["passed"]}
    return differences(dict(save), want)


def run_differences(tree: Path, out: Path) -> dict[str, Any]:
    """Run the report over one run's recorded tables and return its differences from the run's records."""
    document = rp.report(SETTINGS, base=tree / "selection" / "base", saves=tree / "selection" / "checkpoints", rules=RULES,
                         out=out)
    tolerances = json.loads((out / rp.TOLERANCES_FNAME).read_text(encoding="utf-8"))
    calibration = json.loads((tree / "prereg" / "calibration.json").read_text(encoding="utf-8"))
    return {"tolerances": tolerance_differences(tolerances, calibration),
            "saves": {step: save_differences(save, json.loads((tree / "results" / f"checkpoint-{step}.json").read_text(
                encoding="utf-8"))) for step, save in document["saves"].items()},
            "document": document}


def _tree(run: str) -> Path:
    tree = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / "selection" / run
    if not (tree / "selection" / "base" / rp.BANK_FNAME).is_file():
        pytest.skip(f"needs {run}'s dev tables (RULEFORGE_ARTIFACT_ROOT at its root)")
    return tree


@pytest.mark.cluster
@pytest.mark.parametrize("run", sorted(RUNS))
def test_the_report_restates_a_published_run_s_record(run, tmp_path):
    tree, (_, _, published, g4_delta, g4_tolerance) = _tree(run), RUNS[run]
    found = run_differences(tree, tmp_path)
    assert found["tolerances"] == [] and all(rows == [] for rows in found["saves"].values()), found
    saves = found["document"]["saves"]
    assert found["document"]["selected"] == int(published)  # the run's lowest dev inconsistency
    g4 = saves[published]["statistics"]["mode_agreement_nonyes"]
    assert (round(g4["delta"], 4), round(g4["tolerance"], 4), g4["within_tolerance"]) == (g4_delta, g4_tolerance, False)


@pytest.mark.cluster
@pytest.mark.parametrize("run", sorted(RUNS))
def test_a_published_run_s_tables_read_back_from_its_campaign_roots(run, tmp_path):
    tree, (prefix, model, *_) = _tree(run), RUNS[run]
    models = Path(os.environ.get("RULEFORGE_MODEL_ROOT", "/nonexistent")) / model
    if not models.is_dir():
        pytest.skip(f"needs {model}'s tokenizer (RULEFORGE_MODEL_ROOT)")
    root, settings = tree.parents[1], rp.read_settings(SETTINGS)
    sets = rt.stage_sets(rp.read_scenarios(tree / "selection" / "base" / rp.SCENARIOS_FNAME), settings)
    rule_ids, rows = rp.rule_ids_of(RULES, settings["ruleset_sha256"]), load_jsonl_records(RULES)
    tokens, family = rt._tokens(models), rt.directory_family(models)
    jobs = [("bank", root / "sampling" / f"{prefix}-references_bank", "base", tree / "selection" / "base" / rp.BANK_FNAME),
            ("full", root / "sampling" / f"{prefix}-references_full", "base", tree / "selection" / "base" / rp.FULL_FNAME)]
    for save in rp.checkpoint_steps(tree / "selection" / "checkpoints"):
        label = rp.checkpoint_name(save)
        jobs += [(rp.STAGE1, tree / "campaigns" / f"{prefix}-stage1", label,
                  tree / "selection" / "checkpoints" / label / rp.STAGE1_FNAME),
                 (rp.PROBE, tree / "campaigns" / f"{prefix}-probe", label,
                  tree / "selection" / "checkpoints" / label / rp.PROBE_FNAME)]
    for name, campaign, policy, recorded in jobs:
        kind = rt.KINDS[name]
        source = rt.open_dev_condition(campaign, rt.find_condition(campaign, policy=policy, evaluation_mode=kind.evaluation_mode),
                                       evaluation_mode=kind.evaluation_mode)
        table = rt.read_table(source, kind, rules=rows, rule_ids=rule_ids, draws=rt.expected_draws(kind, settings),
                              scenario_ids=sets[kind.scenario_set], ruleset_sha256=settings["ruleset_sha256"],
                              count_tokens=tokens, thinking=family)
        write_jsonl_records(tmp_path / "table.jsonl", table)
        assert sha256_file(tmp_path / "table.jsonl") == sha256_file(recorded), (name, policy)
