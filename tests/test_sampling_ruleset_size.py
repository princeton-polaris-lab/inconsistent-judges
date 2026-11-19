from __future__ import annotations

import dataclasses
from contextlib import contextmanager
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from ruleforge.rulesets import subsets as subsets_module
from ruleforge.evaluate.inference import (
    CURVE_INFERENCE_METHOD,
    CURVE_INTERVAL_BASIS,
    RepeatedSamplingError,
    bank_shared_curve_bootstrap,
)
from ruleforge.evaluate import bank as shared_bank
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import CONDITION_FNAME, build_plan, expected_requests_per_draw, run_plan
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_FULL,
    SHARED_BANK_DESIGN_FNAME,
    ConditionSpec,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.bank import (
    SharedBankDesign,
    SharedBankDesignError,
    analyze_shared_bank_campaign,
    bank_condition_id,
    default_contrasts,
    load_design,
    parse_condition_id,
    plan_conditions,
    subset_condition_id,
    verify_subsets,
    write_design,
)
from ruleforge.evaluate.runner import CampaignRunError, STATE_COMPLETE
from ruleforge.evaluate.validation import ComponentSchema
from tests.support.subsets import render_manifest
from tests.test_sampling_runner import FakeBackend, restamp_stacks, write_inputs

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "sampling_ruleset_size.yaml"
RULES = [{"id": f"Rule {i}", "rule": word} for i, word in enumerate(("one", "two", "three", "four"), start=1)]
SEEDS = (0, 1)
SIZES = (2, 3, 4)


@pytest.fixture(scope="module")
def driver():
    """The shared-bank job's code (``ruleforge.evaluate.bank``, which ``ruleforge bank run`` runs)."""
    import ruleforge.evaluate.bank

    return ruleforge.evaluate.bank


@pytest.fixture(autouse=True)
def _artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def write_subsets(tmp_path: Path, rules=RULES, seeds=SEEDS, strategy=subsets_module.WITH_REPLACEMENT):
    """A committed-style subset tree and manifest for a tiny ruleset."""
    rule_path = tmp_path / "rules_full.jsonl"
    rule_path.write_text("".join(json.dumps(r) + "\n" for r in rules), encoding="utf-8")
    subset_root = tmp_path / "subsets"
    strategy_root = subsets_module.strategy_root(subset_root, strategy)
    for seed in seeds:
        subsets_module.generate_subset_files(
            rule_path, subsets_module.seed_dir(subset_root, strategy, seed), strategy, seed,
        )
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    entries = subsets_module.build_manifest(strategy_root)
    (manifest_dir / "manifest.sha256").write_text(render_manifest(entries), encoding="utf-8")
    (manifest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sampling_strategy": strategy,
                "source_ruleset": {"path": rule_path.name, "sha256": sha256_file(rule_path), "n_rules": len(rules)},
                "files_manifest": "manifest.sha256",
            }
        ),
        encoding="utf-8",
    )
    return rule_path, subset_root, manifest_dir


def make_design(rule_path, manifest_dir, seeds=SEEDS, sizes=SIZES, **overrides):
    fields = dict(
        subset_seeds=seeds,
        subset_sizes=sizes,
        sampling_strategy=subsets_module.WITH_REPLACEMENT,
        subset_manifest_sha256=sha256_file(manifest_dir / "manifest.sha256"),
        source_ruleset_sha256=sha256_file(rule_path),
    )
    fields.update(overrides)
    return SharedBankDesign(**fields)


def compose_ruleset_size(tmp_path: Path, rule_path, subset_root, manifest_dir, **overrides):
    scenarios = write_inputs(tmp_path)
    base = {
        "campaign_id": "ruleset-size",
        "campaign_root": str(tmp_path / "root"),
        "scenario_set_path": str(scenarios),
        "approved_scenario_sha256": "null",
        "rule_set_path": str(rule_path),
        "sampling.campaign_mode": "pilot",
        "sampling.num_decoder_draws": 2,
        "sampling.scenarios_per_request_block": "null",
        "sampling.bootstrap.n_resamples": 150,
        "ruleset_size.subset_root": str(subset_root),
        "ruleset_size.subset_manifest_dir": str(manifest_dir),
        "ruleset_size.approved_subset_manifest_sha256": sha256_file(manifest_dir / "manifest.sha256"),
        "ruleset_size.subset_seeds": "[0,1]",
    }
    base.update(overrides)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="sampling_ruleset_size", overrides=[f"{k}={v}" for k, v in base.items()])


SCRIPT = {(0, "Rule 2"): ["yes"], (2, "full_ruleset"): ["yes"]}


def run_fake_campaign(tmp_path, modelspec, driver, *, script=SCRIPT, strategy=subsets_module.WITH_REPLACEMENT,
                      **overrides):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path, strategy=strategy)
    overrides.setdefault("ruleset_size.sampling_strategy", strategy)
    cfg = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **overrides)
    design, planned, report = driver.plan_job(cfg)
    root = Path(str(cfg.campaign_root))
    root.mkdir(parents=True, exist_ok=True)
    write_design(root, design)
    backend = FakeBackend(style="naive", script=script)
    backend.max_new_tokens = 1024
    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    completions = {}
    for item in planned:
        plan = build_plan(driver.condition_cfg(cfg, item), backend=backend, task_spec=task_spec)
        completions[item.condition_id] = run_plan(plan, backend=backend, task_spec=task_spec)
    return root, cfg, design, planned, report, completions, subset_root


def fake_backend(script=SCRIPT, **attributes):
    """The fake judge every driver test judges with (its token ceiling the condition identity names)."""
    backend = FakeBackend(style="naive", script=script)
    backend.max_new_tokens = 1024
    for name, value in attributes.items():
        setattr(backend, name, value)
    return backend


def job_specs(cfg, planned, modelspec, backend=None, **task):
    """Each planned condition's identity as the job's judge and task spec give it (``job_conditions``)."""
    task_spec = modelspec(**{"reasoning_prompt_style": "naive", "evaluation_temperature": 1.0, **task})
    return shared_bank.job_conditions(cfg, planned, backend=backend or fake_backend(), task_spec=task_spec)


def subset_rule_ids(subset_root, seed, size):
    path = shared_bank.subset_path(subset_root, subsets_module.WITH_REPLACEMENT, seed, size)
    return [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]


def test_design_round_trips_and_freezes_once(tmp_path):
    rule_path, _, manifest_dir = write_subsets(tmp_path)
    design = make_design(rule_path, manifest_dir, sizes=(4, 2, 3))
    assert design.subset_sizes == (2, 3, 4) and design.n_conditions_per_model == 1 + 2 * 3
    assert SharedBankDesign.from_dict(design.to_dict()) == design
    root = tmp_path / "root"
    root.mkdir()
    assert write_design(root, design) == root / SHARED_BANK_DESIGN_FNAME
    assert write_design(root, design).exists()
    assert load_design(root) == design
    with pytest.raises(SharedBankDesignError, match="different shared-bank design"):
        write_design(root, make_design(rule_path, manifest_dir, seeds=(0, 1, 2)))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"design": "literal_bundle"}, "design must be one of"),
        ({"sampling_strategy": "nested"}, "sampling_strategy must be one of"),
        ({"seeds": (0,)}, "at least two distinct non-negative seeds"),
        ({"seeds": (0, 0)}, "at least two distinct non-negative seeds"),
        ({"seeds": (-1, 0)}, "at least two distinct non-negative seeds"),
        ({"sizes": ()}, "one or more distinct sizes"),
        ({"sizes": (2, 2)}, "one or more distinct sizes"),
        ({"sizes": (1, 2)}, "one or more distinct sizes"),
    ],
)
def test_a_malformed_design_is_refused(tmp_path, overrides, message):
    rule_path, _, manifest_dir = write_subsets(tmp_path)
    with pytest.raises(SharedBankDesignError, match=message):
        make_design(rule_path, manifest_dir, **overrides)


def test_a_design_of_one_size_plans_that_size_alone(tmp_path):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    design = make_design(rule_path, manifest_dir, sizes=(3,))
    assert design.subset_sizes == (3,) and design.n_conditions_per_model == 1 + len(design.subset_seeds)
    planned = plan_conditions(design, "m", subset_root=subset_root, bank_rule_set_path=rule_path, include_bank=False)
    assert [item.condition_id for item in planned] == [f"m/subsets/s{seed:02d}/k03" for seed in design.subset_seeds]


def test_loading_a_design_refuses_a_missing_or_foreign_file(tmp_path):
    with pytest.raises(SharedBankDesignError, match="missing"):
        load_design(tmp_path)
    (tmp_path / SHARED_BANK_DESIGN_FNAME).write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(SharedBankDesignError, match="schema_version"):
        load_design(tmp_path)
    (tmp_path / SHARED_BANK_DESIGN_FNAME).write_text(json.dumps({"schema_version": 2, "design": "shared_individual_bank_v1"}))
    with pytest.raises(SharedBankDesignError, match="malformed shared-bank design"):
        load_design(tmp_path)


def test_condition_ids_are_parsed_back_exactly():
    assert parse_condition_id(bank_condition_id("gemma-4-31B-it")) == ("gemma-4-31B-it", "bank", None, None)
    assert parse_condition_id(subset_condition_id("Olmo-3-32B-Think", 3, 57)) == ("Olmo-3-32B-Think", "subset", 3, 57)
    with pytest.raises(SharedBankDesignError, match="neither a bank"):
        parse_condition_id("gemma-4-31B-it/naive")


def test_verify_subsets_checks_every_needed_file_against_the_manifest(tmp_path):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    design = make_design(rule_path, manifest_dir)
    digests = verify_subsets(subset_root, manifest_dir, design)
    assert set(digests) == {(seed, size) for seed in SEEDS for size in SIZES}
    # A tampered subset is a different dataset.
    path = shared_bank.subset_path(subset_root, subsets_module.WITH_REPLACEMENT, 1, 3)
    original = path.read_text()
    path.write_text(original.rstrip("\n") + " \n")
    with pytest.raises(SharedBankDesignError, match="different dataset"):
        verify_subsets(subset_root, manifest_dir, design)
    path.unlink()
    with pytest.raises(SharedBankDesignError, match="is missing"):
        verify_subsets(subset_root, manifest_dir, design)
    path.write_text(original)
    with pytest.raises(SharedBankDesignError, match="not in the committed subset manifest"):
        verify_subsets(subset_root, manifest_dir, make_design(rule_path, manifest_dir, seeds=(0, 7)))


def test_verify_subsets_refuses_a_foreign_manifest_ruleset_strategy_or_meta_rule(tmp_path):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    with pytest.raises(SharedBankDesignError, match="digests to"):
        verify_subsets(subset_root, manifest_dir, make_design(rule_path, manifest_dir, subset_manifest_sha256="0" * 64))
    with pytest.raises(SharedBankDesignError, match="source ruleset"):
        verify_subsets(subset_root, manifest_dir, make_design(rule_path, manifest_dir, source_ruleset_sha256="0" * 64))
    meta = json.loads((manifest_dir / "manifest.json").read_text())
    meta["sampling_strategy"] = subsets_module.WITHOUT_REPLACEMENT
    (manifest_dir / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(SharedBankDesignError, match="the design 'with_replacement'"):
        verify_subsets(subset_root, manifest_dir, make_design(rule_path, manifest_dir))
    # A subset carrying a meta-rule (re-pinned so only the rule check fires).
    meta["sampling_strategy"] = subsets_module.WITH_REPLACEMENT
    (manifest_dir / "manifest.json").write_text(json.dumps(meta))
    path = shared_bank.subset_path(subset_root, subsets_module.WITH_REPLACEMENT, 0, 2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["id"] = "Meta-Rule 1"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    entries = subsets_module.build_manifest(subsets_module.strategy_root(subset_root, subsets_module.WITH_REPLACEMENT))
    (manifest_dir / "manifest.sha256").write_text(render_manifest(entries))
    with pytest.raises(SharedBankDesignError, match="non-meta rules"):
        verify_subsets(subset_root, manifest_dir, make_design(rule_path, manifest_dir))


def test_plan_is_bank_first_then_seed_major(tmp_path):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    design = make_design(rule_path, manifest_dir)
    planned = plan_conditions(design, "m", subset_root=subset_root, bank_rule_set_path=rule_path)
    assert [item.condition_id for item in planned] == [
        "m/bank", "m/subsets/s00/k02", "m/subsets/s00/k03", "m/subsets/s00/k04",
        "m/subsets/s01/k02", "m/subsets/s01/k03", "m/subsets/s01/k04",
    ]
    assert planned[0].evaluation_mode == EVALUATION_MODE_BANK and planned[0].rule_set_path == rule_path
    assert planned[0].extra == {"design": "shared_individual_bank_v1", "role": "bank"}
    assert planned[4].evaluation_mode == EVALUATION_MODE_FULL
    assert planned[4].extra == {
        "design": "shared_individual_bank_v1", "role": "subset", "subset_seed": 1, "subset_size": 2,
        "bank_condition_id": "m/bank",
    }
    assert planned[4].rule_set_path == shared_bank.subset_path(subset_root, subsets_module.WITH_REPLACEMENT, 1, 2)
    share = plan_conditions(design, "m", subset_root=subset_root, bank_rule_set_path=rule_path, seeds=[1], include_bank=False)
    assert [item.condition_id for item in share] == ["m/subsets/s01/k02", "m/subsets/s01/k03", "m/subsets/s01/k04"]
    with pytest.raises(SharedBankDesignError, match="distinct members"):
        plan_conditions(design, "m", subset_root=subset_root, bank_rule_set_path=rule_path, seeds=[1, 1])
    with pytest.raises(SharedBankDesignError, match="distinct members"):
        plan_conditions(design, "m", subset_root=subset_root, bank_rule_set_path=rule_path, seeds=[5])
    other = tmp_path / "other_rules.jsonl"
    other.write_text(json.dumps({"id": "Rule 1", "rule": "x"}) + "\n")
    with pytest.raises(SharedBankDesignError, match="not the subsets' source ruleset"):
        plan_conditions(design, "m", subset_root=subset_root, bank_rule_set_path=other)


def test_the_paper_contrasts_appear_only_where_their_sizes_exist():
    assert default_contrasts([57]) == []
    assert default_contrasts([2, 3, 4]) == [
        {"name": "net_rise", "minuend": [4], "subtrahend": [2]},
        {"name": "last_step", "minuend": [4], "subtrahend": [3]},
    ]
    # K=56 with only part of the middle range: no differently defined twin.
    assert [c["name"] for c in default_contrasts([2, 20, 56])] == ["net_rise", "last_step"]
    full = default_contrasts(range(2, 58))
    assert [c["name"] for c in full] == ["net_rise", "upper_vs_middle", "last_step"]
    assert full[0] == {"name": "net_rise", "minuend": [57], "subtrahend": [2]}
    assert full[1] == {"name": "upper_vs_middle", "minuend": [56], "subtrahend": list(range(20, 31))}
    assert full[2] == {"name": "last_step", "minuend": [57], "subtrahend": [56]}


def test_the_driver_plans_a_share_and_decides_who_runs_the_bank(tmp_path, driver):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    cfg = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir)
    design, planned, report = driver.plan_job(cfg)
    assert design.subset_sizes == SIZES and design.subset_seeds == SEEDS
    assert report["runs_bank"] is True and report["conditions_this_job"] == 7
    assert report["conditions_per_model"] == 7 and report["scenarios"] == 3
    # bank: 3 scenarios x 4 rules; subsets: 6 x 3 scenarios; two draws.
    assert report["requests_per_draw_this_job"] == 12 + 18 and report["requests_total_this_job"] == 60
    assert report["decoder_draw_ids"] == [0, 1]
    # A draw share plans (and would buy) only its own draws, and a bad share is refused here.
    draw_share = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, decoder_draw_ids="[1]")
    assert driver.plan_job(draw_share)[2]["decoder_draw_ids"] == [1]
    assert driver.plan_job(draw_share)[2]["requests_total_this_job"] == 30
    for bad in ("[2]", "[0,0]", "[]"):
        with pytest.raises(SharedBankDesignError, match="decoder_draw_ids"):
            driver.plan_job(compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, decoder_draw_ids=bad))
    with pytest.raises(SharedBankDesignError, match="never sharded"):
        driver.plan_job(compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{
            "execution.num_shards": 2, "execution.shard_index": 0,
        }))
    share = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{"ruleset_size.jobs_seeds": "[1]"})
    _, planned, report = driver.plan_job(share)
    assert report["runs_bank"] is False and [i.condition_id for i in planned] == [
        f"gemma-4-31B-it/subsets/s01/k0{size}" for size in SIZES
    ]
    assert driver.runs_bank(compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{"ruleset_size.run_bank": "true"}), design, [1])
    assert not driver.runs_bank(compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{"ruleset_size.run_bank": "false"}), design, [0])
    with pytest.raises(SharedBankDesignError, match="run_bank"):
        driver.runs_bank(compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{"ruleset_size.run_bank": "sometimes"}), design, [0])
    explicit = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{"ruleset_size.subset_sizes": "[2,4]"})
    assert driver.design_from_cfg(explicit).subset_sizes == (2, 4)


def test_condition_cfg_carries_one_conditions_identity(tmp_path, driver):
    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    cfg = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir)
    _, planned, _ = driver.plan_job(cfg)
    ccfg = driver.condition_cfg(cfg, planned[2])
    assert ccfg.condition_id == "gemma-4-31B-it/subsets/s00/k03"
    assert ccfg.evaluation_mode == EVALUATION_MODE_FULL
    assert ccfg.rule_set_path.endswith("seed_0/subset_size_03.jsonl")
    assert dict(ccfg.condition_extra)["subset_size"] == 3
    assert cfg.condition_id == "gemma-4-31B-it/bank" and cfg.evaluation_mode == EVALUATION_MODE_BANK


def test_every_condition_completes_with_its_own_bundle_shape(tmp_path, modelspec, driver):
    root, cfg, design, planned, report, completions, subset_root = run_fake_campaign(tmp_path, modelspec, driver)
    assert all(c["state"] == STATE_COMPLETE for done in completions.values() for c in done.values())
    bank = json.loads((root / "gemma-4-31B-it" / "bank" / "draw-000" / COMPLETION_FNAME).read_text())
    assert bank["components_per_bundle"] == 4 and bank["accepted"] == 3
    subset = json.loads((root / "gemma-4-31B-it" / "subsets" / "s00" / "k02" / "draw-001" / COMPLETION_FNAME).read_text())
    assert subset["components_per_bundle"] == 1
    bank_rows = [json.loads(line) for line in (root / "gemma-4-31B-it" / "bank" / "draw-000" / "bundle_outcomes.jsonl").read_text().splitlines()]
    assert set(bank_rows[0]["rule_judgments"]) == {"Rule 1", "Rule 2", "Rule 3", "Rule 4"}
    assert bank_rows[0]["full_set_judgment"] is None and bank_rows[0]["contradiction"] is None
    subset_rows = [json.loads(line) for line in (root / "gemma-4-31B-it" / "subsets" / "s00" / "k02" / "draw-000" / "bundle_outcomes.jsonl").read_text().splitlines()]
    assert subset_rows[0]["ruleset_rule_ids"] == subset_rule_ids(subset_root, 0, 2)
    assert subset_rows[0]["rule_judgments"] is None and subset_rows[0]["contradiction"] is None
    assert not (root / "gemma-4-31B-it" / "bank" / "draw-000" / "smrc_per_scenario.jsonl").exists()
    summary = json.loads((root / "gemma-4-31B-it" / "bank" / "draw-000" / "smrc_summary.json").read_text())
    assert summary["evaluation_mode"] == EVALUATION_MODE_BANK and summary["n_accepted"] == 3
    identity = driver.job_identity(cfg, report)
    specs = job_specs(cfg, planned, modelspec)
    for item in planned:
        assert driver.condition_complete(root, specs[item.condition_id], 2, **identity)
    unplanned = dataclasses.replace(specs[planned[1].condition_id], condition_id="gemma-4-31B-it/subsets/s09/k02")
    assert not driver.condition_complete(root, unplanned, 2, **identity)
    record = json.loads((root / "gemma-4-31B-it" / "bank" / "draw-001" / COMPLETION_FNAME).read_text())
    record["state"] = "in_progress"
    (root / "gemma-4-31B-it" / "bank" / "draw-001" / COMPLETION_FNAME).write_text(json.dumps(record))
    assert not driver.condition_complete(root, specs["gemma-4-31B-it/bank"], 2, **identity)


def test_draw_shares_of_one_condition_complete_it_between_them(tmp_path, modelspec, driver):
    root, cfg, _, planned, report, first, _ = run_fake_campaign(tmp_path, modelspec, driver, decoder_draw_ids="[0]")
    assert all(set(done) == {0} and done[0]["state"] == STATE_COMPLETE for done in first.values())
    identity = driver.job_identity(cfg, report)
    specs = job_specs(cfg, planned, modelspec)
    assert not any(driver.condition_complete(root, specs[item.condition_id], 2, **identity) for item in planned)
    with pytest.raises(CampaignRunError, match="not complete"):
        analyze_shared_bank_campaign(root)
    cfg.decoder_draw_ids = [1]
    backend = FakeBackend(style="naive", script=SCRIPT)
    backend.max_new_tokens = 1024
    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    for item in planned:
        second = run_plan(build_plan(driver.condition_cfg(cfg, item), backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
        assert set(second) == {1} and second[1]["state"] == STATE_COMPLETE
    assert all(driver.condition_complete(root, specs[item.condition_id], 2, **identity) for item in planned)
    assert analyze_shared_bank_campaign(root, n_resamples=100)["state"] == STATE_COMPLETE


def test_a_component_scope_root_names_its_retry_scope_in_its_estimates(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver, **{"sampling.retry_scope": "component"})
    payload = analyze_shared_bank_campaign(root, n_resamples=100)
    assert payload["state"] == STATE_COMPLETE and payload["retry_scope"] == "component"


@pytest.mark.parametrize("strategy", subsets_module.VALID_SAMPLING_STRATEGIES)
def test_a_one_size_design_plans_completes_and_reads_one_point_and_no_contrast(tmp_path, modelspec, driver, capsys,
                                                                                strategy):
    from ruleforge.cli import main

    root, _, design, planned, _, completions, _ = run_fake_campaign(
        tmp_path, modelspec, driver, strategy=strategy, **{"ruleset_size.subset_sizes": "[4]"})
    assert (design.subset_sizes, design.sampling_strategy) == ((4,), strategy)
    assert [item.condition_id for item in planned] == [
        "gemma-4-31B-it/bank", "gemma-4-31B-it/subsets/s00/k04", "gemma-4-31B-it/subsets/s01/k04"]
    assert all(c["state"] == STATE_COMPLETE for done in completions.values() for c in done.values())
    payload = analyze_shared_bank_campaign(root, n_resamples=100)
    assert (payload["state"], payload["subset_sizes"], payload["contrasts"]) == (STATE_COMPLETE, [4], [])
    (model,) = payload["models"].values()
    assert len(model["points"]) == 1 and model["contrasts"] == {} and model["adjacent_rises"] == 0
    capsys.readouterr()
    out_file = tmp_path / "analysis" / "figure2_estimates.json"
    assert main(["bank", "analyze", str(root), "--out", str(out_file), "--n-resamples", "120"]) == 0
    assert "gemma-4-31B-it: K=4 " in (out := capsys.readouterr().out) and "one size, so no contrast" in out


def test_an_amended_root_s_estimates_disclose_the_amendment_beside_the_raised_cap(tmp_path, modelspec, driver,
                                                                                  monkeypatch):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    assert "amendments" not in analyze_shared_bank_campaign(root, n_resamples=100)
    seen = []
    disclosed = {"old_cap": 20, "new_cap": 50, "bundles": []}
    monkeypatch.setattr(shared_bank, "amendment_disclosure", lambda *args: seen.append(args) or disclosed)
    payload = analyze_shared_bank_campaign(root, n_resamples=100)
    assert payload["amendments"] == [disclosed] and seen and Path(seen[0][0]) == Path(root)
    assert json.loads((Path(root) / "aggregate" / "figure2_estimates.json").read_text())["amendments"] == [disclosed]


def test_the_analysis_joins_each_subset_to_the_bank_of_the_same_draw(tmp_path, modelspec, driver):
    root, cfg, design, planned, _, _, subset_root = run_fake_campaign(tmp_path, modelspec, driver)
    payload = analyze_shared_bank_campaign(root)
    assert payload["state"] == STATE_COMPLETE and payload["inference_method"] == CURVE_INFERENCE_METHOD
    assert payload["interval_basis"] == CURVE_INTERVAL_BASIS
    assert "retry_scope" not in payload
    assert payload["design"] == design.to_dict() and payload["n_subset_seeds"] == 2 and payload["n_decoder_draws"] == 2
    assert payload["subset_sizes"] == [2, 3, 4] and payload["contrasts"] == default_contrasts(SIZES)
    entry = payload["models"]["gemma-4-31B-it"]
    assert entry["judge_type"] == "No_Interpretation"
    assert entry["bank"]["condition_id"] == "gemma-4-31B-it/bank" and entry["bank"]["n_rules"] == 4
    assert entry["subsets"]["n_conditions"] == 6
    for point in entry["points"]:
        holds = [("Rule 2" in subset_rule_ids(subset_root, seed, point["size"])) for seed in SEEDS]
        expected = np.mean([(1 + int(h)) / 3 for h in holds])
        assert point["estimate"] == pytest.approx(expected), point["size"]
        assert point["ci_low"] <= point["estimate"] <= point["ci_high"]
    assert entry["points"][-1]["estimate"] == pytest.approx(2 / 3)
    assert entry["contrasts"]["net_rise"]["estimate"] == pytest.approx(
        entry["points"][-1]["estimate"] - entry["points"][0]["estimate"]
    )
    assert isinstance(entry["adjacent_rises"], int)
    assert (root / "aggregate" / "figure2_estimates.json").exists()
    with pytest.raises(CampaignRunError, match="shared-bank campaign"):
        analyze_campaign(root)


def _record_vllm(root, pattern, version):
    for provenance in root.glob(f"{pattern}/draw-*/draw_provenance.json"):
        payload = json.loads(provenance.read_text())
        payload["runtime"]["vllm_version"] = version
        provenance.write_text(json.dumps(payload))
        restamp_stacks(provenance.parent)


def test_a_bank_and_subsets_on_different_stacks_are_not_joined(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    _record_vllm(root, "gemma-4-31B-it/subsets/*/*", "0.28.0+cu129")
    # the same release in another CUDA build is another stack
    _record_vllm(root, "gemma-4-31B-it/bank", "0.28.0+cu130")
    with pytest.raises(CampaignRunError, match=r"vllm 0.28.0\+cu130 .* against vllm 0.28.0\+cu129"):
        analyze_shared_bank_campaign(root, n_resamples=100)
    _record_vllm(root, "gemma-4-31B-it/bank", "0.28.0+cu129")
    clean = analyze_shared_bank_campaign(root, n_resamples=100)
    assert not {"allow_stack_mismatch", "stack_mismatches"} & set(clean)
    assert "inference_stacks" not in clean["models"]["gemma-4-31B-it"]
    assert "stack" not in (root / "aggregate" / "figure2_estimates.json").read_text()
    # the bank on another release: every subset joined to it would pair two engines
    _record_vllm(root, "gemma-4-31B-it/bank", "0.19.0")
    with pytest.raises(CampaignRunError, match=(
        r"do not share one inference stack \(gemma-4-31B-it/bank vs gemma-4-31B-it/subsets/s00/k02: "
        r"vllm 0.19.0 .* against vllm 0.28.0.*; and 3 more\)"
    )):
        analyze_shared_bank_campaign(root, n_resamples=100)
    payload = analyze_shared_bank_campaign(root, n_resamples=100, allow_stack_mismatch=True)
    assert payload["allow_stack_mismatch"] is True
    assert [r["conditions"] for r in payload["stack_mismatches"]] == [
        ["gemma-4-31B-it/bank", f"gemma-4-31B-it/subsets/s{seed:02d}/k{size:02d}"] for seed in SEEDS for size in SIZES
    ]
    assert {r["scope"] for r in payload["stack_mismatches"]} == {"gemma-4-31B-it"}
    assert [s["vllm"] for s in payload["models"]["gemma-4-31B-it"]["inference_stacks"]] == ["0.19.0", "0.28.0+cu129"]
    written = json.loads((root / "aggregate" / "figure2_estimates.json").read_text())
    assert written["stack_mismatches"] == payload["stack_mismatches"]
    _record_vllm(root, "gemma-4-31B-it/bank", "0.28.0+cu129")
    provenance = root / "gemma-4-31B-it" / "subsets" / "s01" / "k03" / "draw-001" / "draw_provenance.json"
    mixed = json.loads(provenance.read_text())
    mixed["runtime"]["vllm_version"] = "0.19.0"
    provenance.write_text(json.dumps(mixed))
    restamp_stacks(provenance.parent)
    with pytest.raises(CampaignRunError, match="s01/k03: its judgments come from several stacks"):
        analyze_shared_bank_campaign(root, n_resamples=100)
    mixed["runtime"].update(vllm_version="0.28.0+cu129", engine_args={"max_model_len": 8192})
    provenance.write_text(json.dumps(mixed))
    restamp_stacks(provenance.parent)
    first = provenance.parents[1] / "draw-000" / "draw_provenance.json"
    unmixed = json.loads(first.read_text())
    unmixed["runtime"]["engine_args"] = {"max_model_len": 4096}
    first.write_text(json.dumps(unmixed))
    restamp_stacks(first.parent)
    with pytest.raises(CampaignRunError, match=r"s01/k03: its judgments come from several stacks or engine "
                                               r"configurations \(.*; engine_args.max_model_len 4096 -> 8192\)"):
        analyze_shared_bank_campaign(root, n_resamples=100)


def _tamper(root, condition_id, draw, mutate):
    path = root / condition_id / f"draw-{draw:03d}" / "bundle_outcomes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _reregister(root, condition_id, mutate) -> str:
    path = root / condition_id / CONDITION_FNAME
    payload = json.loads(path.read_text())
    mutate(payload)
    spec = ConditionSpec.from_dict({k: v for k, v in payload.items() if k not in ("schema_version", "fingerprint")})
    payload["fingerprint"] = spec.fingerprint
    path.write_text(json.dumps(payload))
    return spec.fingerprint


def test_the_analysis_refuses_a_root_whose_conditions_do_not_match_the_design(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    wider = SharedBankDesign.from_dict({**load_design(root).to_dict(), "subset_seeds": [0, 1, 2]})
    (root / SHARED_BANK_DESIGN_FNAME).write_text(json.dumps({"schema_version": 2, **wider.to_dict()}))
    with pytest.raises(SharedBankDesignError, match="design condition\\(s\\) missing"):
        analyze_shared_bank_campaign(root)


def test_the_analysis_refuses_a_bank_or_subset_registered_under_the_wrong_identity(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    cid = "gemma-4-31B-it/subsets/s00/k02"
    _reregister(root, cid, lambda p: p["extra"].__setitem__("subset_size", 3))
    with pytest.raises(SharedBankDesignError, match="disagree with the id's seed/size"):
        analyze_shared_bank_campaign(root)
    _reregister(root, cid, lambda p: p["extra"].update(subset_size=2, role="bank"))
    with pytest.raises(SharedBankDesignError, match="do not name design"):
        analyze_shared_bank_campaign(root)
    _reregister(root, cid, lambda p: (p["extra"].__setitem__("role", "subset"), p.__setitem__("evaluation_mode", "both")))
    with pytest.raises(SharedBankDesignError, match="registered as 'both'"):
        analyze_shared_bank_campaign(root)


def test_the_analysis_refuses_a_model_judged_under_two_strategies(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    fingerprint = _reregister(root, "gemma-4-31B-it/subsets/s01/k02", lambda p: p.__setitem__("judge_type", "Narrow"))
    for draw in (0, 1):
        completion = root / "gemma-4-31B-it" / "subsets" / "s01" / "k02" / f"draw-{draw:03d}" / COMPLETION_FNAME
        record = json.loads(completion.read_text())
        record["condition_fingerprint"] = fingerprint
        completion.write_text(json.dumps(record))
    with pytest.raises(SharedBankDesignError, match="more than one judge configuration \\(judge_type: 'No_Interpretation' "
                                                 "\\(gemma-4-31B-it/bank\\) vs 'Narrow' "
                                                 "\\(gemma-4-31B-it/subsets/s01/k02\\)\\)"):
        analyze_shared_bank_campaign(root)


@pytest.mark.parametrize(("field", "value", "shown"), [("prompt_style", "forced-grpo", "'forced-grpo'"),
                                                     ("max_new_tokens", 77, "77")])
def test_the_analysis_refuses_a_model_whose_conditions_hold_two_judge_configurations(tmp_path, modelspec, driver,
                                                                                    field, value, shown):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    cid = "gemma-4-31B-it/subsets/s01/k02"
    _reregister(root, cid, lambda p: p.__setitem__(field, value))
    with pytest.raises(SharedBankDesignError, match=rf"its conditions in this root hold more than one judge configuration "
                                                 rf"\({field}: [^()]+ \(gemma-4-31B-it/bank\) vs {shown} \({cid}\)"):
        analyze_shared_bank_campaign(root)
    assert shared_bank.configuration_conflicts(list(shared_bank.load_conditions(root).values())).keys() == {field}


def test_the_analysis_refuses_a_bank_judged_on_another_ruleset(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    fingerprint = _reregister(root, "gemma-4-31B-it/bank", lambda p: p.__setitem__("rule_set_sha256", "0" * 64))
    for draw in (0, 1):
        completion = root / "gemma-4-31B-it" / "bank" / f"draw-{draw:03d}" / COMPLETION_FNAME
        record = json.loads(completion.read_text())
        record["condition_fingerprint"] = fingerprint
        completion.write_text(json.dumps(record))
    with pytest.raises(SharedBankDesignError, match="not the design's source ruleset"):
        analyze_shared_bank_campaign(root)


@pytest.mark.parametrize(
    ("condition_id", "mutate", "message"),
    [
        ("gemma-4-31B-it/bank", lambda rows: rows[0].__setitem__("rule_judgments", {}), "carries no rule judgments"),
        ("gemma-4-31B-it/bank", lambda rows: rows[1].__setitem__("rule_judgments", {"Rule 9": "no"}), "different rule set"),
        ("gemma-4-31B-it/subsets/s00/k02", lambda rows: rows[0].__setitem__("ruleset_rule_ids", None), "not a full-only outcome"),
        ("gemma-4-31B-it/subsets/s00/k02", lambda rows: [r.__setitem__("ruleset_rule_ids", ["Rule 1"]) for r in rows], "names 1 rules, not 2"),
        ("gemma-4-31B-it/subsets/s00/k02", lambda rows: rows[2].__setitem__("ruleset_rule_ids", ["Rule 1", "Rule 3"]), "judged a different subset"),
        ("gemma-4-31B-it/subsets/s00/k02", lambda rows: [r.__setitem__("ruleset_rule_ids", ["Rule 1", "Rule 9"]) for r in rows], "no judgment for rule"),
    ],
)
def test_the_analysis_refuses_outcomes_that_cannot_be_joined(tmp_path, modelspec, driver, condition_id, mutate, message):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    _tamper(root, condition_id, 0, mutate)
    with pytest.raises(RepeatedSamplingError, match=message):
        analyze_shared_bank_campaign(root)


def test_a_screening_root_is_never_analysed(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    manifest_path = root / "sampling_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["campaign_mode"] = "screening"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CampaignRunError, match="screening campaign"):
        analyze_shared_bank_campaign(root)


def test_the_curve_bootstrap_keeps_every_size_of_a_scenario_on_one_draw():
    rng = np.random.default_rng(3)
    draw_effect = rng.integers(0, 2, size=(1, 6, 1, 3)).astype(float)
    values = np.broadcast_to(draw_effect, (4, 6, 2, 3)).copy()
    out = bank_shared_curve_bootstrap(
        values, sizes=[2, 3, 4, 5],
        contrasts=[{"name": "net_rise", "minuend": [5], "subtrahend": [2]}],
        n_resamples=300, seed=1,
    )
    assert out["method"] == CURVE_INFERENCE_METHOD and out["interval_basis"] == CURVE_INTERVAL_BASIS
    assert [p["size"] for p in out["points"]] == [2, 3, 4, 5]
    assert out["points"][0]["estimate"] == pytest.approx(values.mean())
    assert out["contrasts"]["net_rise"] == {
        "minuend_sizes": [5], "subtrahend_sizes": [2], "estimate": 0.0,
        "ci_low": 0.0, "ci_high": 0.0, "sd": 0.0,
        "two_stage": {"ci_low": 0.0, "ci_high": 0.0, "sd": 0.0},
    }
    assert out["n_scenarios"] == 6 and out["n_subset_seeds"] == 2 and out["n_decoder_draws"] == 3
    values[3] = np.clip(values[3] + 1.0, 0, 1)
    out = bank_shared_curve_bootstrap(values, sizes=[2, 3, 4, 5], n_resamples=300, seed=1)
    assert out["points"][3]["estimate"] == pytest.approx(1.0)
    assert out["points"][3]["ci_low"] <= 1.0 <= out["points"][3]["ci_high"]
    assert out["points"][3]["two_stage"]["ci_low"] <= 1.0 <= out["points"][3]["two_stage"]["ci_high"]
    assert out["contrasts"] == {}


def test_the_two_stage_reading_is_the_wider_one_and_chunking_does_not_move_a_number(monkeypatch):
    """Decoder noise inside a cell is already in the cell mean at the estimator's scale."""
    from ruleforge.evaluate import inference as rs

    rng = np.random.default_rng(11)
    values = (rng.random((3, 40, 4, 5)) < 0.4).astype(float)
    out = bank_shared_curve_bootstrap(values, sizes=[2, 30, 57], n_resamples=400, seed=5)
    for point in out["points"]:
        assert point["two_stage"]["sd"] >= point["sd"]
        assert point["ci_low"] <= point["estimate"] <= point["ci_high"]
    monkeypatch.setattr(rs, "_CURVE_CHUNK", 7)
    monkeypatch.setattr(rs, "_CURVE_CHUNK_TWO_STAGE", 3)
    assert bank_shared_curve_bootstrap(values, sizes=[2, 30, 57], n_resamples=400, seed=5) == out


def test_a_contrast_named_twice_is_refused():
    with pytest.raises(RepeatedSamplingError, match="named twice"):
        bank_shared_curve_bootstrap(
            np.zeros((2, 3, 2, 2)), sizes=[2, 3], n_resamples=100,
            contrasts=[{"name": "x", "minuend": [3], "subtrahend": [2]}] * 2,
        )


@pytest.mark.parametrize(
    ("values", "sizes", "contrasts", "message"),
    [
        (np.zeros((2, 3, 2)), [2, 3], (), "must be \\(sizes, scenarios"),
        (np.zeros((2, 3, 2, 2)), [2, 2], (), "size slices"),
        (np.zeros((2, 3, 2, 2)), [2], (), "size slices"),
        (np.zeros((2, 1, 2, 2)), [2, 3], (), "at least two scenarios"),
        (np.zeros((2, 3, 1, 2)), [2, 3], (), "at least two scenarios"),
        (np.zeros((2, 3, 2, 1)), [2, 3], (), "at least two scenarios"),
        (np.full((2, 3, 2, 2), np.nan), [2, 3], (), "missing"),
        (np.zeros((2, 3, 2, 2)), [2, 3], [{"name": "x", "minuend": [9], "subtrahend": [2]}], "outside"),
        (np.zeros((2, 3, 2, 2)), [2, 3], [{"name": "x", "minuend": [], "subtrahend": [2]}], "both sides"),
    ],
)
def test_the_curve_bootstrap_refuses_a_malformed_array_or_contrast(values, sizes, contrasts, message):
    with pytest.raises(RepeatedSamplingError, match=message):
        bank_shared_curve_bootstrap(values, sizes=sizes, contrasts=contrasts, n_resamples=100)


def test_the_cli_prints_each_models_curve(tmp_path, modelspec, driver, capsys):
    from ruleforge.cli import main

    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    out_file = tmp_path / "analysis" / "figure2_estimates.json"
    assert main(["bank", "analyze", str(root), "--out", str(out_file), "--n-resamples", "120"]) == 0
    out = capsys.readouterr().out
    assert "gemma-4-31B-it: K=2" in out and "net rise" in out and "adjacent steps rise" in out
    assert json.loads(out.strip().splitlines()[-1])["n_subset_seeds"] == 2
    assert json.loads(out_file.read_text())["bootstrap"]["n_resamples"] == 120
    _record_vllm(root, "gemma-4-31B-it/subsets/*/*", "0.28.0")
    _record_vllm(root, "gemma-4-31B-it/bank", "0.19.0")
    again = ["bank", "analyze", str(root), "--out", str(out_file), "--n-resamples", "120", "--replace"]
    assert main(again) == 1 and "do not share one inference stack" in capsys.readouterr().err
    assert main([*again, "--allow-stack-mismatch"]) == 0
    assert json.loads(out_file.read_text())["allow_stack_mismatch"] is True


def test_copying_a_complete_root_keeps_it_analysable(tmp_path, modelspec, driver):
    root, *_ = run_fake_campaign(tmp_path, modelspec, driver)
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    assert analyze_shared_bank_campaign(copy)["models"]["gemma-4-31B-it"]["subsets"]["n_conditions"] == 6


def test_an_unknown_bundle_shape_is_refused_everywhere(tmp_path):
    with pytest.raises(SamplingDesignError, match="unknown evaluation_mode"):
        ComponentSchema.from_rules(RULES, prompt_style="naive", forced_json_requirement="complete_rule_map", evaluation_mode="individual")
    manifest = ScenarioManifest.build(write_inputs(tmp_path))
    assert expected_requests_per_draw(manifest, RULES, "individual_bank") == 3 * 4
    assert expected_requests_per_draw(manifest, RULES, "full") == 3
    assert expected_requests_per_draw(manifest, RULES) == 3 * 5
    with pytest.raises(SamplingDesignError, match="unknown evaluation_mode"):
        expected_requests_per_draw(manifest, RULES, "individual")
    with pytest.raises(SamplingDesignError, match="evaluation_mode must be one of"):
        ConditionSpec(
            condition_id="m/x", model_name="m", model_type="local", prompt_style="naive",
            individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="individual",
            rule_set_name="r", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
            forced_json_requirement="complete_rule_map",
        )


def test_a_gate_from_another_population_or_campaign_does_not_skip_a_condition(tmp_path, modelspec, driver):
    root, cfg, _, planned, report, _, _ = run_fake_campaign(tmp_path, modelspec, driver)
    identity = driver.job_identity(cfg, report)
    assert identity == {"scenario_manifest_sha256": report["scenario_manifest_sha256"], "campaign_id": "ruleset-size"}
    spec = job_specs(cfg, planned, modelspec)[planned[0].condition_id]
    assert driver.condition_complete(root, spec, 2, **identity)
    assert not driver.condition_complete(root, spec, 2, **{**identity, "scenario_manifest_sha256": "0" * 64})
    assert not driver.condition_complete(root, spec, 2, **{**identity, "campaign_id": "ruleset-size-other"})
    gate = root / spec.condition_id / "draw-001" / COMPLETION_FNAME
    record = json.loads(gate.read_text())
    gate.write_text(json.dumps({**record, "condition_fingerprint": "0" * 64}))
    assert not driver.condition_complete(root, spec, 2, **identity)
    gate.write_text(json.dumps(record))
    with pytest.raises(TypeError):
        driver.condition_complete(root, spec, 2)
    with pytest.raises(TypeError):
        driver.condition_complete(root, spec, 2, identity["scenario_manifest_sha256"], identity["campaign_id"])


@pytest.mark.parametrize("change", ["prompt_style", "max_new_tokens"])
@pytest.mark.parametrize("packed", [False, True], ids=["sequential", "packed"])
def test_a_root_registering_a_condition_under_another_identity_is_refused_not_skipped(tmp_path, modelspec, driver,
                                                                                      change, packed):
    root, cfg, _, planned, report, _, _ = run_fake_campaign(tmp_path, modelspec, driver)
    identity = driver.job_identity(cfg, report)
    task = {"reasoning_prompt_style": "forced-grpo"} if change == "prompt_style" else {}
    backend = fake_backend(**({"max_new_tokens": 2048} if change == "max_new_tokens" else {}))
    other = job_specs(cfg, planned, modelspec, backend=backend, **task)
    assert all(other[item.condition_id].fingerprint != spec.fingerprint
               for item, spec in zip(planned, job_specs(cfg, planned, modelspec).values()))
    with pytest.raises(CampaignRunError, match="already registers condition .* with a different identity"):
        driver.condition_complete(root, other[planned[0].condition_id], 2, **identity)
    before = {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}
    run = driver.run_packed_conditions if packed else driver.run_conditions
    task_spec = modelspec(**{"reasoning_prompt_style": "naive", "evaluation_temperature": 1.0, **task})
    with pytest.raises(CampaignRunError, match="different identity"):
        run(cfg, planned, identity=identity, backend=backend, task_spec=task_spec)
    assert backend.blocks == [] and {path: path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()} == before
    judged = fake_backend()
    assert run(cfg, planned, identity=identity, backend=judged,
               task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)) == {}
    assert judged.blocks == []


def _files(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("change", ["prompt_style", "max_new_tokens"])
@pytest.mark.parametrize("packed", [False, True], ids=["sequential", "packed"])
def test_a_seed_share_under_another_judge_configuration_is_refused_before_anything_is_written(
        tmp_path, modelspec, driver, change, packed):
    root, cfg, _, _, report, _, _ = run_fake_campaign(tmp_path, modelspec, driver, **{"ruleset_size.jobs_seeds": "[0]"})
    share = OmegaConf.merge(cfg, {"ruleset_size": {"jobs_seeds": [1]}})
    _, planned, _ = driver.plan_job(share)
    assert planned and all("/s01/" in item.condition_id for item in planned)
    task = {"reasoning_prompt_style": "forced-grpo"} if change == "prompt_style" else {}
    backend = fake_backend(**({"max_new_tokens": 2048} if change == "max_new_tokens" else {}))
    shown = "'forced-grpo'" if change == "prompt_style" else "2048"
    before = _files(root)
    run = driver.run_packed_conditions if packed else driver.run_conditions
    identity = driver.job_identity(share, report)
    with pytest.raises(CampaignRunError, match=rf"more than one judge configuration \({change}: {shown} "
                                               rf"\(gemma-4-31B-it/subsets/s01/k02\) vs [^()]+ \(gemma-4-31B-it/bank\)"):
        run(share, planned, identity=identity, backend=backend,
            task_spec=modelspec(**{"reasoning_prompt_style": "naive", "evaluation_temperature": 1.0, **task}))
    assert backend.blocks == [] and _files(root) == before
    judged = fake_backend()
    assert run(share, planned, identity=identity, backend=judged,
               task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)) == {}
    assert judged.blocks and analyze_shared_bank_campaign(root, n_resamples=100)["state"] == STATE_COMPLETE


def test_a_registration_made_meanwhile_under_another_configuration_is_refused_under_the_lock(tmp_path, modelspec,
                                                                                             driver):
    from ruleforge.evaluate import campaign as campaign_module

    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    cfg = compose_ruleset_size(tmp_path, rule_path, subset_root, manifest_dir, **{"ruleset_size.jobs_seeds": "[1]"})
    design, planned, report = driver.plan_job(cfg)
    root = Path(str(cfg.campaign_root))
    root.mkdir()
    write_design(root, design)
    whole = OmegaConf.merge(cfg, {"ruleset_size": {"jobs_seeds": None}})
    _, every, _ = driver.plan_job(whole)
    bank_spec = job_specs(whole, every[:1], modelspec, reasoning_prompt_style="forced-grpo")[every[0].condition_id]
    entered = []

    @contextmanager
    def writing():
        if not entered:
            campaign_module.register_condition(root, bank_spec)  # another job's, just before this one's lock
        entered.append(True)
        yield

    backend = fake_backend()
    with pytest.raises(CampaignRunError, match=r"prompt_style: 'naive' \(gemma-4-31B-it/subsets/s01/k02\) vs "
                                               r"'forced-grpo' \(gemma-4-31B-it/bank\)"):
        driver.run_conditions(cfg, planned, identity=driver.job_identity(cfg, report), backend=backend,
                              task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0),
                              writing=writing)
    assert backend.blocks == [] and sorted(_files(root)) == [SHARED_BANK_DESIGN_FNAME, f"{every[0].condition_id}/condition.json"]


def test_every_registration_into_a_shared_bank_root_holds_its_model_to_one_configuration(tmp_path, modelspec,
                                                                                        driver):
    from ruleforge.evaluate import campaign as campaign_module

    root, cfg, _, planned, _, _, _ = run_fake_campaign(tmp_path, modelspec, driver)
    other = job_specs(cfg, planned[1:2], modelspec, reasoning_prompt_style="forced-grpo")[planned[1].condition_id]
    stray = dataclasses.replace(other, condition_id="gemma-4-31B-it/subsets/s05/k02")
    with pytest.raises(CampaignRunError, match="more than one judge configuration \\(prompt_style: "):
        campaign_module.register_condition(root, stray)
    assert not (root / stray.condition_id).exists()
    outside = dataclasses.replace(other, condition_id="gemma-4-31B-it/other")
    assert campaign_module.register_condition(root, outside).is_file()
    with pytest.raises(SharedBankDesignError, match="'gemma-4-31B-it/other' is neither a bank"):
        analyze_shared_bank_campaign(root)


@pytest.mark.parametrize("packed", ["false", "true"])
def test_bank_run_runs_every_condition_then_gates_the_campaign(tmp_path, monkeypatch, packed):
    import ruleforge.judges.backend as backend_module
    from ruleforge.cli import main

    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    scenarios = write_inputs(tmp_path)
    overrides = [
        "campaign_id=ruleset-size", f"campaign_root={tmp_path / 'root'}", f"scenario_set_path={scenarios}",
        "approved_scenario_sha256=null", f"rule_set_path={rule_path}", "sampling.campaign_mode=pilot",
        "sampling.num_decoder_draws=2", "sampling.scenarios_per_request_block=null",
        f"ruleset_size.subset_root={subset_root}", f"ruleset_size.subset_manifest_dir={manifest_dir}",
        f"ruleset_size.approved_subset_manifest_sha256={sha256_file(manifest_dir / 'manifest.sha256')}",
        "ruleset_size.subset_seeds=[0,1]", f"execution.pack_draws={packed}",
    ]
    backends = []

    def fake_backend(model_cfg):
        backend = FakeBackend(style="naive", script=SCRIPT)
        backend.max_new_tokens, backend.llm = 1024, None
        backends.append(backend)
        return backend

    monkeypatch.setattr(backend_module, "ModelBackend", fake_backend)
    assert main(["bank", "run", "--config", str(CONFIG), "--", *overrides]) == 0
    gate = json.loads((tmp_path / "root" / COMPLETION_FNAME).read_text())
    assert gate["state"] == STATE_COMPLETE and gate["n_conditions"] == 7
    # Packed: every draw's attempt-0 bundles in one call; per block otherwise.
    assert len(backends[0].blocks) == (1 if packed == "true" else 14)
    assert analyze_shared_bank_campaign(tmp_path / "root", n_resamples=100)["state"] == STATE_COMPLETE
