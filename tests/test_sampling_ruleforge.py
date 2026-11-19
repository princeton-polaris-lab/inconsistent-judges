from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.aggregate import analyze_campaign, missing_rulesets, ruleset_families, style_families
from ruleforge.evaluate.campaign import (
    build_plan,
    load_conditions,
    non_meta_rule_count,
    preflight,
    run_plan,
    validate_campaign_cfg,
)
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.core.artifacts import iter_jsonl_records
from tests.test_sampling_campaign import backend_for, compose_campaign, run_two_conditions

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_DIR = REPO_ROOT  # the jobs' working directory: the configs' relative paths are the checkout's
RULESETS = ("initial", "seed5_winner", "seed9_runner_up")


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def compose_ruleforge(ruleset: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="sampling_ruleforge", overrides=["campaign_id=ruleforge", f"ruleforge.ruleset={ruleset}"])


@pytest.mark.parametrize("ruleset", RULESETS)
def test_each_ruleset_composes_into_its_own_implicated_condition(ruleset, tmp_path):
    cfg = compose_ruleforge(ruleset)
    validate_campaign_cfg(cfg)
    assert cfg.condition_id == f"gemma-4-31B-it/figure4/{ruleset}"
    assert dict(cfg.condition_extra) == {
        "figure": "figure4", "ruleset": ruleset, "ruleset_family": "+".join(RULESETS),
    }
    assert list(cfg.ruleforge.rulesets) == list(RULESETS)
    assert cfg.eval_cfg.individual_rule_context == "implicated"
    assert cfg.eval_cfg.reasoning_prompt_style == "naive-rules-above" and cfg.evaluation_mode == "both"
    assert cfg.sampling.campaign_mode == "final" and cfg.sampling.num_decoder_draws == 5
    assert cfg.sampling.max_semantic_attempts == 20 and "label_rule" not in cfg.sampling
    assert cfg.scenario_set_path == str(tmp_path / "artifacts" / "populations" / "frozen_500_r3.jsonl")


def test_the_config_pins_the_published_rulesets():
    provenance = json.loads(
        (REPO_ROOT / "tests" / "fixtures" / "ruleforge" / "campaign_input.json").read_text(encoding="utf-8")
    )
    published = [entry["ruleset"]["sha256"] for entry in provenance["conditions"]]
    for ruleset, digest in zip(RULESETS, published, strict=True):
        cfg = compose_ruleforge(ruleset)
        assert cfg.approved_rule_set_sha256 == digest
        path = (JOB_DIR / str(cfg.rule_set_path)).resolve()
        assert sha256_file(path) == digest, path
        assert non_meta_rule_count(tuple(iter_jsonl_records(path))) == 57


def test_preflight_refuses_a_drifted_ruleset(tmp_path):
    cfg = compose_campaign(tmp_path, approved_rule_set_sha256="0" * 64)
    with pytest.raises(CampaignRunError, match="rule set SHA-256 .* does not match approved 0000"):
        preflight(cfg)
    rules = tmp_path / "rules.jsonl"
    config, manifest = preflight(compose_campaign(tmp_path, approved_rule_set_sha256=sha256_file(rules)))
    assert manifest.record_count == 3 and config.num_decoder_draws == 2


def test_an_unknown_ruleset_cannot_be_composed():
    with pytest.raises(Exception, match="nonesuch"):
        str(compose_ruleforge("nonesuch").rule_set_path)


FAMILY = "initial+seed5_winner+seed9_runner_up"


def run_ruleset_conditions(tmp_path, modelspec, labels=RULESETS, *, family=FAMILY, same_rules=False, axis=None):
    """Naive conditions over different rulesets."""
    scripts = {"initial": {(0, "Rule 2"): ["yes"]}, "seed5_winner": {(1, "Rule 1"): ["yes"]}, "seed9_runner_up": None}
    for label in labels:
        rules = tmp_path / f"{label}.jsonl"
        body = "shared" if same_rules else label
        rules.write_text(
            "".join(json.dumps({"id": f"Rule {n}", "rule": f"{body} rule {n}"}) + "\n" for n in (1, 2)),
            encoding="utf-8",
        )
        extra = {"figure": "figure4", "ruleset": label}
        if family is not None:
            extra["ruleset_family"] = family
        if axis is not None:
            extra["axis"] = axis[label]
        cfg = compose_campaign(
            tmp_path,
            rule_set_path=str(rules),
            condition_id=f"gemma/figure4/{label}",
            condition_extra="{" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + "}",
        )
        backend = backend_for("naive-rules-above", script=scripts.get(label))
        task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
        run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    return tmp_path / "root"


def test_the_three_rulesets_are_one_holm_family_of_three_contrasts(tmp_path, modelspec):
    root = run_ruleset_conditions(tmp_path, modelspec)
    conditions = load_conditions(root)
    members = [f"gemma/figure4/{label}" for label in RULESETS]
    assert ruleset_families(conditions) == {"+".join(members): members}
    assert missing_rulesets(conditions, members) == []
    assert sorted(style_families(conditions).values()) == [[member] for member in members]

    payload = analyze_campaign(root, n_permutations=200, n_resamples=150)
    family = payload["ruleset_families"]["+".join(members)]
    assert family["members"] == members and "missing_rulesets" not in family
    pairs = [(row["a"], row["b"]) for row in family["contrasts"]]
    assert pairs == [(members[0], members[1]), (members[0], members[2]), (members[1], members[2])]
    # one of three scenarios is inconsistent under initial, another under the winner, none under the runner-up
    assert [row["estimate"] for row in family["contrasts"]] == pytest.approx([0.0, 1 / 3, 1 / 3])
    for row in family["contrasts"]:
        assert row["family"] == "ruleset_contrasts" and row["family_size"] == 3
        assert row["inner_resample"] == "independent" and row["p_holm"] >= row["p_value"]
    assert all(entry["contrasts"] == [] for entry in payload["style_families"].values())
    row = payload["conditions"][members[1]]
    assert row["extra"] == {"figure": "figure4", "ruleset": "seed5_winner", "ruleset_family": FAMILY}
    assert row["rule_set_name"] == "seed5_winner.jsonl" and len(row["rule_set_sha256"]) == 64


def test_a_family_short_of_a_declared_ruleset_reports_no_contrast(tmp_path, modelspec):
    root = run_ruleset_conditions(tmp_path, modelspec, labels=RULESETS[:2])
    conditions = load_conditions(root)
    members = [f"gemma/figure4/{label}" for label in RULESETS[:2]]
    assert missing_rulesets(conditions, members) == ["seed9_runner_up"]
    with pytest.raises(CampaignRunError, match="missing declared rulesets .*seed9_runner_up"):
        analyze_campaign(root, n_permutations=100, n_resamples=100)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100, allow_incomplete_conditions=True)
    assert payload["ruleset_families"] == {
        "+".join(members): {"members": members, "contrasts": [], "missing_rulesets": ["seed9_runner_up"]}
    }


def test_a_new_sweep_s_tags_are_contrasted_once_the_family_declares_them(tmp_path, modelspec):
    labels = ("initial", "seed3_winner", "seed7_runner_up")
    root = run_ruleset_conditions(tmp_path, modelspec, labels=labels, family="+".join(labels))
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100)
    (family,) = payload["ruleset_families"].values()
    assert len(family["contrasts"]) == 3


def test_an_undeclared_family_is_contrasted_as_it_stands(tmp_path, modelspec):
    root = run_ruleset_conditions(tmp_path, modelspec, labels=RULESETS[:2], family=None)
    payload = analyze_campaign(root, n_permutations=100, n_resamples=100)
    (family,) = payload["ruleset_families"].values()
    (contrast,) = family["contrasts"]
    assert contrast["family_size"] == 1 and contrast["estimate"] == pytest.approx(0.0)


def test_another_design_axis_keeps_conditions_out_of_one_family(tmp_path, modelspec):
    root = run_ruleset_conditions(
        tmp_path, modelspec, labels=RULESETS[:2], family=None, axis={"initial": "a", "seed5_winner": "b"}
    )
    assert ruleset_families(load_conditions(root)) == {}


def test_two_conditions_of_a_family_over_one_ruleset_are_refused(tmp_path, modelspec):
    root = run_ruleset_conditions(tmp_path, modelspec, family=None)
    conditions = dict(load_conditions(root))
    # Forge the runner-up onto the winner's ruleset: three members, two digests.
    import dataclasses

    winner = conditions["gemma/figure4/seed5_winner"]
    conditions["gemma/figure4/seed9_runner_up"] = dataclasses.replace(
        conditions["gemma/figure4/seed9_runner_up"],
        rule_set_name=winner.rule_set_name, rule_set_sha256=winner.rule_set_sha256,
    )
    with pytest.raises(CampaignRunError, match="two of them judge the same one"):
        ruleset_families(conditions)


def test_a_prompt_style_campaign_has_no_ruleset_family(tmp_path, modelspec):
    root, _ = run_two_conditions(tmp_path, modelspec)
    assert ruleset_families(load_conditions(root)) == {}
    assert analyze_campaign(root, n_permutations=100, n_resamples=100)["ruleset_families"] == {}
