from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ruleforge.core.digests import sha256_file
from ruleforge.evaluate import bank
from ruleforge.evaluate.bank import (
    BANK_ROOT_KEY,
    ONE_RULESET_SIZES_KEY,
    SharedBankDesignError,
    analyze_shared_bank_campaign,
    committed_subsets,
    read_conditions,
)
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.experiments.ruleset_size import build as curves_build
from ruleforge.experiments.ruleset_size import repeated_draw_baseline as rdb
from ruleforge.rulesets import subsets as subsets_module
from tests.test_sampling_ruleset_size import SEEDS, _record_vllm, _reregister, run_fake_campaign

MODEL = "gemma-4-31B-it"
#: Answers, taken call by call, that move with the scenario, rule, seed, size and draw, so that the curve's cells differ.
FIRST_SCRIPT = {(0, "Rule 2"): ["yes", "no"], (1, "Rule 1"): ["unsure", "yes"], (2, "Rule 3"): ["no", "yes"],
                **{(index, "full_ruleset"): ["yes", "no", "unsure", "no", "yes", "yes", "no"][index:] * 3
                   for index in range(3)}}
SECOND_SCRIPT = {(index, "full_ruleset"): ["no", "yes", "yes", "unsure"][index:] * 2 for index in range(3)}


@pytest.fixture(autouse=True)
def _artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def _two_roots(tmp_path, modelspec, monkeypatch, *, bank_overrides=None, **overrides):
    """A root with the bank and sizes 2 to 4 in file order, and a root of size 4 alone in each seed's own order."""
    (tmp_path / "first").mkdir()
    bank_root, *_ = run_fake_campaign(tmp_path / "first", modelspec, bank, script=FIRST_SCRIPT,
                                      **(bank_overrides or {}))
    return bank_root, _second_root(tmp_path, modelspec, monkeypatch, **overrides)


def _second_root(tmp_path, modelspec, monkeypatch, **overrides):
    (tmp_path / "second").mkdir()
    root, *_ = run_fake_campaign(
        tmp_path / "second", modelspec, bank, script=SECOND_SCRIPT,
        strategy=subsets_module.WITH_REPLACEMENT_SHUFFLED_FULL,
        **{"campaign_id": "k4", "ruleset_size.subset_sizes": "[4]", "ruleset_size.run_bank": "false", **overrides})
    checkout = tmp_path / "checkout"
    shutil.copytree(tmp_path / "second" / "manifest", checkout / "data" / "rules" / "shuffled")
    monkeypatch.setattr(bank, "repository_root", lambda: checkout)
    return root


def _amended_bank_root(tmp_path, modelspec, monkeypatch, old_cap, new_cap):
    """The first root of ``_two_roots`` at ``old_cap``, one bundle exhausted, then amended to ``new_cap`` and resumed."""
    from ruleforge.evaluate import campaign
    from ruleforge.evaluate.campaign import build_plan, run_plan
    from ruleforge.scenarios.manifest import ScenarioManifest
    from tests.support.cap_amendment import amend_attempt_cap
    from tests.test_sampling_ruleset_size import compose_ruleset_size
    from tests.test_sampling_runner import FakeBackend

    first, before, after = tmp_path / "first", {"code_revision": "a" * 40}, {"code_revision": "b" * 40}
    first.mkdir()
    monkeypatch.setattr(campaign, "write_code_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, "code_state", lambda: {**before, "code_dirty": False, "code_state_sha256": "1" * 64})
    script = {**FIRST_SCRIPT, (0, "full_ruleset"): ["maybe"] * old_cap + ["yes"]}
    bank_root, cfg, _, planned, *_ = run_fake_campaign(first, modelspec, bank, script=script,
                                                       **{"sampling.max_semantic_attempts": old_cap})
    exhausted = (f"{MODEL}/subsets/s00/k02", 0, ScenarioManifest.load(bank_root / "scenario_manifest.json").scenario_ids[0])
    code = {**after, "code_dirty": False, "code_state_sha256": "2" * 64}
    amend_attempt_cap(bank_root, new_cap=new_cap, bundles=[exhausted], decided_by="a test", decided_on="2026-11-16",
                      reason="a bundle exhausted at the old cap", ledger="none", population_path=str(cfg.scenario_set_path),
                      rules_path=[first / "rules_full.jsonl", *first.glob("subsets/**/*.jsonl")], record=True, code=code)
    monkeypatch.setattr(campaign, "code_state", lambda: dict(code))
    resumed = compose_ruleset_size(first, first / "rules_full.jsonl", first / "subsets", first / "manifest",
                                   **{"sampling.max_semantic_attempts": new_cap})
    backend, task = FakeBackend(style="naive"), modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)
    backend.max_new_tokens = 1024
    for item in planned:
        run_plan(build_plan(bank.condition_cfg(resumed, item), backend=backend, task_spec=task), backend=backend,
                 task_spec=task)
    return bank_root


def _named(estimates):
    """The estimates with the fake judge named as the condition prefix names it, so the curve has a legend label."""
    payload = json.loads(estimates.read_text())
    payload["models"][MODEL]["model_name"] = MODEL
    estimates.write_text(json.dumps(payload))
    return estimates


def _rows(condition_dir, draw):
    path = Path(condition_dir) / f"draw-{draw:03d}" / "bundle_outcomes.jsonl"
    return {row["scenario_id"]: row for row in map(json.loads, path.read_text().splitlines())}


def test_a_bank_root_supplies_the_bank_and_the_sizes_a_root_lacks_and_leaves_them_as_they_were(
        tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    alone = analyze_shared_bank_campaign(bank_root, n_resamples=100)
    joined = analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    assert json.loads((root / "aggregate" / "figure2_estimates.json").read_text()) == joined
    assert joined["campaign_id"] == "k4" and joined["state"] == "complete"
    assert joined["design"] == {**alone["design"], "sampling_strategy": subsets_module.WITH_REPLACEMENT_SHUFFLED_FULL,
                                "subset_manifest_sha256": sha256_file(tmp_path / "second/manifest/manifest.sha256")}
    assert joined[BANK_ROOT_KEY] == {"campaign_id": "ruleset-size", "root": "artifact://first/root", "subset_sizes": [2, 3],
                                     "max_semantic_attempts": 20}
    assert joined["subset_sizes"] == [2, 3, 4] and joined["contrasts"] == alone["contrasts"]
    mine, theirs = joined["models"][MODEL], alone["models"][MODEL]
    assert mine["bank"] == theirs["bank"] and mine["points"][:2] == theirs["points"][:2]
    assert all(point["ci_low"] < point["ci_high"] for point in mine["points"])
    assert mine["points"][2]["estimate"] != theirs["points"][2]["estimate"]
    assert mine["subsets"]["grid_digest"] != theirs["subsets"]["grid_digest"]
    assert mine["subsets"]["n_conditions"] == theirs["subsets"]["n_conditions"] == 6
    # Seeds 0 and 1 happen to draw one order of these four rules, so the one-ruleset sizes follow the files judged;
    # an analysis of one root records none.
    assert mine[ONE_RULESET_SIZES_KEY] == [4] and ONE_RULESET_SIZES_KEY not in theirs
    assert "root" not in alone and BANK_ROOT_KEY not in alone
    found = read_conditions(root, bank_root)
    flags = []
    for seed in SEEDS:
        for draw in range(2):
            banks = _rows(found.dirs[f"{MODEL}/bank"], draw)
            for scenario, row in _rows(found.dirs[f"{MODEL}/subsets/s{seed:02d}/k04"], draw).items():
                separate = max(PHI_MAP[label] for label in banks[scenario]["rule_judgments"].values())
                flags.append(PHI_MAP[row["full_set_judgment"]] != separate)
    assert found.dirs[f"{MODEL}/subsets/s00/k04"].is_relative_to(root)
    assert found.dirs[f"{MODEL}/subsets/s00/k03"].is_relative_to(bank_root)
    assert mine["points"][2]["estimate"] == pytest.approx(np.mean(flags))


#: The keys, in order, an analysis of one root writes at the top and for each model: its estimates file keeps the
#: bytes of an analysis without a bank root but for the time it was written.
ONE_ROOT_KEYS = ["schema_version", "campaign_id", "state", "design", "inference_method", "interval_basis",
                 "n_scenarios", "n_decoder_draws", "n_subset_seeds", "subset_sizes", "temperature",
                 "max_semantic_attempts", "truncation", "label_rule", "scenario_manifest_sha256", "bootstrap",
                 "contrasts", "models", "created_at"]
ONE_ROOT_MODEL_KEYS = ["model_name", "judge_type", "bank", "subsets", "points", "contrasts", "adjacent_rises"]


def test_an_analysis_of_one_root_writes_what_it_wrote_before_roots_could_be_joined(tmp_path, modelspec, monkeypatch):
    from ruleforge.core.files import json_sidecar_text

    bank_root, _ = _two_roots(tmp_path, modelspec, monkeypatch)
    payload = analyze_shared_bank_campaign(bank_root, n_resamples=100)
    assert list(payload) == ONE_ROOT_KEYS and list(payload["models"][MODEL]) == ONE_ROOT_MODEL_KEYS
    assert (bank_root / "aggregate" / "figure2_estimates.json").read_text() == json_sidecar_text(payload)


def test_a_bank_root_of_another_design_or_beside_a_bank_of_its_own_is_refused(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    design = json.loads((root / "figure2_design.json").read_text())
    (root / "figure2_design.json").write_text(json.dumps({**design, "source_ruleset_sha256": "0" * 64}))
    with pytest.raises(SharedBankDesignError, match="design differs from .* in source_ruleset_sha256"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    (root / "figure2_design.json").write_text(json.dumps(design))
    shutil.copytree(bank_root / MODEL / "bank", root / MODEL / "bank")
    with pytest.raises(SharedBankDesignError, match=f"registers a bank of its own \\({MODEL}/bank\\)"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    shutil.rmtree(root / MODEL / "bank")
    (bank_root / MODEL / "bank" / "condition.json").unlink()
    with pytest.raises(SharedBankDesignError, match="1 design condition\\(s\\) missing"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_a_root_holding_a_size_its_design_does_not_name_is_refused_rather_than_read_over_the_bank_root_s(
        tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    shutil.copytree(bank_root / MODEL / "subsets" / "s01" / "k03", root / MODEL / "subsets" / "s01" / "k03")
    with pytest.raises(SharedBankDesignError, match=f"registers {MODEL}/subsets/s01/k03, a size its own design does "
                                                    "not name"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_a_bank_root_subset_that_is_not_the_design_s_file_is_refused(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    pinned = committed_subsets(bank.load_design(root))
    monkeypatch.setattr(bank, "committed_subsets", lambda design: {**pinned, "seed_1/subset_size_03.jsonl": "0" * 64})
    with pytest.raises(SharedBankDesignError, match=f"first/root/{MODEL}/subsets/s01/k03 judged a ruleset that is not "
                                                    "the with_replacement_shuffled_full design's subset"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_the_root_s_own_subsets_are_checked_against_the_manifest_its_design_names(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    # Another committed manifest, the file-order one: its size 4 is not the file the root's seeds judged.
    shutil.copytree(tmp_path / "first" / "manifest", tmp_path / "checkout" / "data" / "rules" / "file_order")
    design = json.loads((root / "figure2_design.json").read_text())
    named = {**design, "subset_manifest_sha256": sha256_file(tmp_path / "first" / "manifest" / "manifest.sha256")}
    (root / "figure2_design.json").write_text(json.dumps(named))
    with pytest.raises(SharedBankDesignError, match="the subset manifest is 'with_replacement', the design "
                                                    "'with_replacement_shuffled_full'"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    (root / "figure2_design.json").write_text(json.dumps({**named, "sampling_strategy": "with_replacement"}))
    with pytest.raises(SharedBankDesignError, match=f"second/root/{MODEL}/subsets/s00/k04 judged a ruleset that is "
                                                    "not the with_replacement design's subset seed_0/subset_size_04"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_the_checkout_s_subset_manifests_are_found_by_their_checksum_file_s_digest(tmp_path, monkeypatch):
    shuffled = Path(__file__).resolve().parents[1] / "data" / "rules" / "subset_manifest_shuffled_full"
    meta, pinned = subsets_module.load_manifest(shuffled)
    design = bank.SharedBankDesign(subset_seeds=tuple(meta["seeds"]), subset_sizes=tuple(range(2, 58)),
                                   sampling_strategy=meta["sampling_strategy"],
                                   subset_manifest_sha256=sha256_file(shuffled / "manifest.sha256"),
                                   source_ruleset_sha256=meta["source_ruleset"]["sha256"])
    assert committed_subsets(design) == pinned and len(pinned) == 1400
    with pytest.raises(SharedBankDesignError, match="no subset manifest under data/rules digests to 000000000000"):
        committed_subsets(dataclasses.replace(design, subset_manifest_sha256="0" * 64))


@pytest.mark.parametrize(("old_cap", "new_cap"), [(20, 30), (5, 20)])
def test_a_bank_root_whose_cap_an_amendment_raised_joins_a_root_run_at_the_cap_in_force(
        tmp_path, modelspec, monkeypatch, old_cap, new_cap):
    bank_root = _amended_bank_root(tmp_path, modelspec, monkeypatch, old_cap, new_cap)
    root = _second_root(tmp_path, modelspec, monkeypatch, **{"sampling.max_semantic_attempts": new_cap})
    payload = analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    record = payload[BANK_ROOT_KEY]
    assert (payload["max_semantic_attempts"], record["max_semantic_attempts"]) == (new_cap, new_cap)
    assert [(amendment["old_cap"], amendment["new_cap"]) for amendment in record["amendments"]] == [(old_cap, new_cap)]
    assert "amendments" not in payload


def test_a_bank_root_with_another_amendment_than_a_cap_s_is_refused_on_either_side(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    for place, campaign_id in ((bank_root, "ruleset-size"), (root, "k4")):
        manifest = json.loads((place / "sampling_manifest.json").read_text())
        changed = {**manifest, "amendments": [{"kind": "temperature", "field": "temperature", "schema_version": 1}]}
        (place / "sampling_manifest.json").write_text(json.dumps(changed))
        with pytest.raises(CampaignRunError, match=f"campaign '{campaign_id}' records the amendment\\(s\\) "
                                                   r"\['temperature'\] \(an amendment of another kind"):
            analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
        (place / "sampling_manifest.json").write_text(json.dumps(manifest))


def test_a_bank_root_that_ran_other_settings_or_another_judge_configuration_is_refused(tmp_path, modelspec,
                                                                                       monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch, **{"sampling.max_semantic_attempts": 30})
    with pytest.raises(CampaignRunError, match="ran other sampling settings than campaign 'k4' "
                                               r"\(max_semantic_attempts 20 vs 30\)"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    _reregister(root, f"{MODEL}/subsets/s00/k04", lambda payload: payload.update(max_new_tokens=512))
    with pytest.raises(SharedBankDesignError, match=f"{MODEL}: .* hold more than one judge configuration "
                                                    r"\(max_new_tokens: 1024"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_a_bank_root_is_complete_for_what_it_supplies(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    (bank_root / MODEL / "subsets" / "s00" / "k04" / "draw-001" / "completion.json").unlink()
    assert analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)["state"] == "complete"
    (bank_root / MODEL / "subsets" / "s01" / "k02" / "draw-001" / "completion.json").unlink()
    with pytest.raises(CampaignRunError, match=f"not complete for the conditions this analysis takes from it.*"
                                               f'"{MODEL}/subsets/s01/k02": {{"1": "in_progress"}}'):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_a_size_judged_on_another_stack_than_the_bank_root_s_is_refused_or_marked(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    _record_vllm(bank_root, f"{MODEL}/**", "0.19.0")
    _record_vllm(root, f"{MODEL}/subsets/*/*", "0.28.0+cu129")
    with pytest.raises(CampaignRunError, match=f"do not share one inference stack \\({MODEL}/bank vs "
                                               f"{MODEL}/subsets/s00/k04: vllm 0.19.0 .* against vllm 0.28.0"):
        analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    payload = analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root, allow_stack_mismatch=True)
    assert [record["conditions"] for record in payload["stack_mismatches"]] == [
        [f"{MODEL}/bank", f"{MODEL}/subsets/s{seed:02d}/k04"] for seed in SEEDS]
    _record_vllm(root, f"{MODEL}/subsets/*/*", "0.19.0")
    assert "stack_mismatches" not in analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)


def test_the_cli_analyses_a_root_joined_to_a_bank_root_into_a_file(tmp_path, modelspec, monkeypatch, capsys):
    from ruleforge.cli import main

    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    out = tmp_path / "analysis" / "figure2_estimates.json"
    assert main(["bank", "analyze", str(root), "--bank-root", str(bank_root), "--out", str(out),
                 "--n-resamples", "100"]) == 0
    written = json.loads(out.read_text())
    assert written[BANK_ROOT_KEY]["campaign_id"] == "ruleset-size" and written["subset_sizes"] == [2, 3, 4]
    assert (written["root"], written[BANK_ROOT_KEY]["root"]) == ("artifact://second/root", "artifact://first/root")
    assert f"{MODEL}: K=2 " in capsys.readouterr().out
    assert not (root / "aggregate").exists() and not (bank_root / "aggregate").exists()
    (curve,) = curves_build.build_curves([_named(out)])
    assert curve["campaign"]["sizes_from"] == {"4": {"campaign_id": "k4", "root": "artifact://second/root",
                                                     "retry_scope": "scenario_bundle", "max_semantic_attempts": 20}}
    (judge,) = rdb.build_payload([out])["judges"]
    assert judge["campaign"] == {key: curve["campaign"][key] for key in judge["campaign"]}
    assert [point["inconsistency"]["estimate"] for point in judge["points"]] == [p["mean"] for p in curve["points"]]


def test_a_published_copy_is_never_bank_analysed_as_a_root_or_as_its_bank_root(tmp_path, modelspec, monkeypatch,
                                                                             capsys):
    from ruleforge.cli import main
    from ruleforge.core.constants import PUBLISHED_COPY_FNAME
    from ruleforge.evaluate.identity import PublishedCopyError

    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    out = tmp_path / "analysis" / "figure2_estimates.json"
    for copy in (bank_root, root):
        (copy / PUBLISHED_COPY_FNAME).write_text("{}")
        assert main(["bank", "analyze", str(root), "--bank-root", str(bank_root), "--out", str(out)]) == 2
        assert (f"{copy} is a published copy of a campaign root (published_copy.json): it holds only the files, rows "
                "and fields the paper's rebuilds read, not its manifests and each draw's completion record in full, "
                "which bank analyze reads") in capsys.readouterr().err
        with pytest.raises(PublishedCopyError, match="which bank analyze reads"):
            bank.shared_bank_estimates(root, bank_root=bank_root)
        (copy / PUBLISHED_COPY_FNAME).unlink()
    assert not out.exists()


def test_the_curves_name_the_bank_root_s_campaign_and_the_root_each_other_size_came_from(tmp_path, modelspec,
                                                                                         monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch, bank_overrides={"sampling.max_semantic_attempts": 30},
                                 **{"sampling.max_semantic_attempts": 30})
    disclosed = {"old_cap": 20, "new_cap": 30, "bundles": []}
    monkeypatch.setattr(bank, "amendment_disclosure",
                        lambda found, manifest: disclosed if Path(found) == bank_root.resolve() else None)
    payload = analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    estimates = _named(root / "aggregate" / "figure2_estimates.json")
    (curve,) = curves_build.build_curves([estimates])
    campaign = curve["campaign"]
    # The campaign that judged the bank, with the cap it froze and the amendment that raised it ...
    assert (campaign["campaign_id"], campaign["max_semantic_attempts"]) == ("ruleset-size", 20)
    assert campaign["amendments"] == [disclosed] and "amendments" not in payload
    # ... the joined estimates, the bank root and the root of the size it lacked.
    assert campaign["estimates"] == "artifact://second/root/aggregate/figure2_estimates.json"
    assert campaign["root"] == "artifact://first/root"
    assert campaign["sizes_from"] == {"4": {"campaign_id": "k4", "root": "artifact://second/root",
                                            "retry_scope": "scenario_bundle", "max_semantic_attempts": 30}}
    assert campaign["design"] == payload["design"]
    assert [point["shares_one_ruleset"] for point in curve["points"]] == [False, False, True]
    assert [point["mean"] for point in curve["points"]] == [
        point["estimate"] for point in payload["models"][MODEL]["points"]]
    assert curves_build.validate_curves([curve]) == [curve]


def test_the_repeated_draw_baseline_reads_both_roots_as_the_joined_analysis_did(tmp_path, modelspec, monkeypatch):
    bank_root, root = _two_roots(tmp_path, modelspec, monkeypatch)
    payload = analyze_shared_bank_campaign(root, n_resamples=100, bank_root=bank_root)
    design, phi = rdb.read_phi(root, MODEL, n_draws=2, bank_root=bank_root)
    assert design.to_dict() == payload["design"]
    assert bank.subset_grid_digest(phi.inconsistency) == payload["models"][MODEL]["subsets"]["grid_digest"]
    estimates = _named(root / "aggregate" / "figure2_estimates.json")
    (judge,) = rdb.build_payload([estimates])["judges"]
    (curve,) = curves_build.build_curves([estimates])
    assert judge["campaign"] == {key: curve["campaign"][key] for key in judge["campaign"]}
    assert set(judge["campaign"]) == {"campaign_id", "estimates", "estimates_sha256", "root", "sizes_from"}
    assert [point["inconsistency"]["estimate"] for point in judge["points"]] == [
        point["mean"] for point in curve["points"]]
