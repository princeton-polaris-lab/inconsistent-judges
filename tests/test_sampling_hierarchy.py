from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ruleforge.rulesets.variants import build_rule_variants, rule_set_text
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    build_plan,
    campaign_rule_levels,
    load_conditions,
    preflight,
    run_plan,
    validate_campaign_cfg,
)
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.prompts.styles import (
    HIERARCHY_RULE_CONTEXT,
    HIGHER_RANKED_RULES_HEADER,
    NAIVE_RULES_ABOVE,
    NAIVE_HIERARCHY,
    RANKING_BLOCK,
)
from ruleforge.core.artifacts import load_jsonl_records
from ruleforge.rulesets.levels import load_rule_levels
from tests.support.hydra_runs import run_program
from tests.test_sampling_campaign import compose_campaign
from tests.test_sampling_runner import RULES, FakeBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
HIER = NAIVE_HIERARCHY
#: RULES' levels, in file order (its meta-rule rides in the full set only).
LEVELS = {"Rule 1": "root", "Rule 2": "user", "Meta-Rule 1": "guideline"}


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)


def write_levels(path: Path, levels=LEVELS) -> Path:
    path.write_text("".join(json.dumps({"id": rule["id"], "rule_type": levels[rule["id"]], "rule": rule["rule"]}) + "\n"
                            for rule in RULES), encoding="utf-8")
    return path


def hierarchy_overrides(tmp_path: Path, **extra) -> dict:
    """The hierarchy arm's keys, as the hierarchy config composes them, over the campaign tests' config."""
    levels = write_levels(tmp_path / "levels.jsonl")
    return {"eval_cfg.reasoning_prompt_style": HIER, "eval_cfg.individual_rule_context": HIERARCHY_RULE_CONTEXT,
            "+rule_levels_path": str(levels), "+approved_rule_levels_sha256": sha256_file(levels), **extra}


def run(tmp_path, modelspec, *, hierarchy: bool, campaign_id="camp", judge=None, **extra):
    """One condition of ``campaign_id`` (root ``tmp_path/<campaign_id>``): the flat control or the hierarchy arm."""
    overrides = {"campaign_id": campaign_id, "campaign_root": str(tmp_path / campaign_id), **extra}
    if hierarchy:
        overrides = {**hierarchy_overrides(tmp_path), **overrides}
    cfg = compose_campaign(tmp_path, **overrides)
    style = HIER if hierarchy else NAIVE_RULES_ABOVE
    judge = judge or FakeBackend(style=style)
    judge.max_new_tokens = getattr(judge, "max_new_tokens", 1024)
    task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0,
                          individual_rule_context=HIERARCHY_RULE_CONTEXT if hierarchy else "isolated")
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    run_plan(plan, backend=judge, task_spec=task_spec)
    return plan, tmp_path / campaign_id


def sent_prompts(root: Path, condition_id: str) -> dict:
    """``{(scenario prompt, component): the prompt sent}`` from draw 0's traced outputs."""
    sent = {}
    for row in load_jsonl_records(root / condition_id / "draw-000" / "judge_outputs.jsonl"):
        for component, by_type in next(iter(row["judge_outputs"].values())).items():
            sent[(row["scenario"]["prompt"], component)] = by_type["No_Interpretation"]["prompt"]["system_prompt"]
    return sent


def test_the_hierarchy_arm_runs_beside_its_flat_control_with_its_levels_in_its_identity(tmp_path, modelspec):
    from ruleforge.grpo.grpo_data import CampaignPrompts

    flat_plan, root = run(tmp_path, modelspec, hierarchy=False)
    plan, _ = run(tmp_path, modelspec, hierarchy=True)
    conditions = load_conditions(root)
    flat, hierarchy = conditions[f"gemma-4-31B-it/{NAIVE_RULES_ABOVE}"], conditions[f"gemma-4-31B-it/{HIER}"]
    digest = sha256_file(tmp_path / "levels.jsonl")
    assert (hierarchy.prompt_style, hierarchy.individual_rule_context, hierarchy.rule_levels_sha256) == (
        HIER, HIERARCHY_RULE_CONTEXT, digest)
    assert flat.rule_levels_sha256 is None and plan.rule_levels.by_id == LEVELS and flat_plan.rule_levels is None
    registered = json.loads((root / hierarchy.condition_id / CONDITION_FNAME).read_text())
    assert registered["rule_levels_sha256"] == digest
    assert "rule_levels_sha256" not in json.loads((root / flat.condition_id / CONDITION_FNAME).read_text())
    flat_payload, hier_payload = flat.to_dict(), hierarchy.to_dict()
    assert sorted(key for key in set(flat_payload) | set(hier_payload)
                  if flat_payload.get(key) != hier_payload.get(key)) == [
        "condition_id", "individual_rule_context", "prompt_style", "rule_levels_sha256"]
    rules = load_jsonl_records(tmp_path / "rules.jsonl")
    hier_spec = modelspec(reasoning_prompt_style=HIER, individual_rule_context=HIERARCHY_RULE_CONTEXT)
    variants = {rule_id: (text, context) for rule_id, text, context in
                build_rule_variants(rules, HIERARCHY_RULE_CONTEXT, LEVELS)[0]}
    hier_sent, flat_sent = sent_prompts(root, hierarchy.condition_id), sent_prompts(root, flat.condition_id)
    assert set(hier_sent) == set(flat_sent) and {component for _, component in hier_sent} == {
        "full_ruleset", "Rule 1", "Rule 2"}
    built = CampaignPrompts.build(task_spec=hier_spec, rules=rules, judge_type="No_Interpretation",
                                  levels=plan.rule_levels.by_id)
    for (prompt, component), sent in hier_sent.items():
        scenario = {"prompt": prompt, "response": prompt.replace("prompt", "response")}
        text, context = ((rule_set_text(rules, HIERARCHY_RULE_CONTEXT, LEVELS), None) if component == "full_ruleset"
                         else variants[component])
        assert sent == hier_spec.build_eval_prompt(text, scenario, "No_Interpretation", context_rules=context)
        assert built.entry(scenario, component) == {"system_prompt": sent, "user_prompt": ""}
        assert RANKING_BLOCK in sent and RANKING_BLOCK not in flat_sent[(prompt, component)]
        assert (HIGHER_RANKED_RULES_HEADER in sent) == (component == "Rule 2")
    full = next(sent for (_, component), sent in hier_sent.items() if component == "full_ruleset")
    assert "Rule 1 (root): rule 1 body\nRule 2 (user): rule 2 body\nMeta-Rule 1 (guideline): meta body" in full
    assert "Rule 1: rule 1 body\nRule 2: rule 2 body\nMeta-Rule 1: meta body" in next(
        sent for (_, component), sent in flat_sent.items() if component == "full_ruleset")


@pytest.mark.parametrize(("overrides", "message"), [
    ({"eval_cfg.reasoning_prompt_style": HIER}, r"eval_cfg: reasoning_prompt_style 'naive-hierarchy' judges each "
                                                r"rule with the rules that outrank it"),
    ({"eval_cfg.individual_rule_context": HIERARCHY_RULE_CONTEXT}, "eval_cfg: individual_rule_context 'superiors' "
                                                                  "belongs to the hierarchy prompt"),
    ({"eval_cfg.reasoning_prompt_style": HIER, "eval_cfg.individual_rule_context": HIERARCHY_RULE_CONTEXT},
     "rule_levels_path=None: a 'superiors' condition reads each rule's level from one"),
    ({"+rule_levels_path": "levels.jsonl"}, "rule_levels_path=levels.jsonl: .* no other condition reads one"),
    ({"+approved_rule_levels_sha256": "ab"}, "approved_rule_levels_sha256=ab but no rule_levels_path"),
    ({"eval_cfg.reasoning_prompt_style": HIER, "eval_cfg.individual_rule_context": HIERARCHY_RULE_CONTEXT,
      "interpretation": "Narrow", "eval_cfg.judge_types": "[Narrow]"}, r"defined for the No_Interpretation template "
                                                                        r"only;"),
])
def test_the_campaign_check_refuses_any_pairing_but_the_approved_one(tmp_path, overrides, message):
    with pytest.raises(CampaignRunError, match=message):
        validate_campaign_cfg(compose_campaign(tmp_path, **overrides))


def test_the_preflight_aligns_the_levels_with_the_ruleset_and_holds_them_to_their_digest(tmp_path):
    cfg = compose_campaign(tmp_path, **hierarchy_overrides(tmp_path))
    config, manifest = preflight(cfg)
    levels = campaign_rule_levels(cfg)
    assert levels.by_id == LEVELS and levels.levels_file["sha256"] == sha256_file(tmp_path / "levels.jsonl")
    assert campaign_rule_levels(compose_campaign(tmp_path)) is None
    misaligned = write_levels(tmp_path / "other.jsonl", {**LEVELS, "Rule 2": "platform"})
    for extra, message in (
        ({"+approved_rule_levels_sha256": "ab" * 32}, f"rule levels SHA-256 {sha256_file(tmp_path / 'levels.jsonl')} "
                                                      f"does not match approved {'ab' * 32}"),
        ({"+rule_levels_path": str(misaligned), "+approved_rule_levels_sha256": "null"},
         r"no level, or one not in .* \[\"2 \(Rule 2: 'platform'\)\"\]"),
        ({"+rule_levels_path": str(tmp_path / "none.jsonl")}, "rule_levels_path .*none.jsonl does not exist"),
    ):
        with pytest.raises(CampaignRunError, match=message):
            preflight(compose_campaign(tmp_path, **{**hierarchy_overrides(tmp_path), **extra}))


def test_the_runner_refuses_a_superiors_task_spec_without_the_levels(tmp_path, modelspec):
    from ruleforge.evaluate.campaign import draw_runner

    cfg = compose_campaign(tmp_path, **hierarchy_overrides(tmp_path))
    judge = FakeBackend(style=HIER)
    judge.max_new_tokens = 1024
    task_spec = modelspec(reasoning_prompt_style=HIER, evaluation_temperature=1.0,
                          individual_rule_context=HIERARCHY_RULE_CONTEXT)
    plan = build_plan(cfg, backend=judge, task_spec=task_spec)
    import dataclasses

    with pytest.raises(ValueError, match="'superiors' judges each rule with its superiors and needs each rule's level"):
        draw_runner(dataclasses.replace(plan, rule_levels=None), 0, backend=judge, task_spec=task_spec)


def test_the_preflight_script_reports_the_rule_context_and_the_levels(tmp_path, capsys, monkeypatch):
    cfg = compose_campaign(tmp_path, **hierarchy_overrides(tmp_path))
    script = "ruleforge.launch.programs.preflight_campaign"
    common = [str(script), "campaign_id=camp", f"campaign_root={tmp_path / 'root'}",
              f"scenario_set_path={cfg.scenario_set_path}", "approved_scenario_sha256=null",
              f"rule_set_path={cfg.rule_set_path}", "sampling.campaign_mode=pilot", f"hydra.run.dir={tmp_path / 'h'}"]
    reports = {}
    monkeypatch.chdir(REPO_ROOT)  # the checkout's top, where the launcher runs the preflight
    for name, extra in (("flat", []), ("hierarchy", [f"{key}={value}" for key, value in
                                                     hierarchy_overrides(tmp_path).items()])):
        monkeypatch.setattr(sys, "argv", [*common, *extra])
        run_program(script)
        reports[name] = json.loads(capsys.readouterr().out)
    assert (reports["flat"]["individual_rule_context"], reports["flat"]["rule_levels"]) == ("isolated", None)
    assert reports["hierarchy"]["individual_rule_context"] == HIERARCHY_RULE_CONTEXT
    assert reports["hierarchy"]["rule_levels"] == {
        "name": "levels.jsonl", "sha256": sha256_file(tmp_path / "levels.jsonl"),
        "rules_per_level": {"root": 1, "system": 0, "developer": 0, "user": 1, "guideline": 1}}
    assert not (tmp_path / "root").exists()


def test_every_bundle_of_the_paper_s_population_renders_under_the_hierarchy_prompt(modelspec):
    population = REPO_ROOT / "data" / "populations" / "frozen_500_r3.jsonl"
    assert sha256_file(population) == "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4"
    rules_path = REPO_ROOT / "data/rules/modelspec/openai_rules_all.jsonl"
    rules = load_jsonl_records(rules_path)
    levels = load_rule_levels(rules_path, REPO_ROOT / "data/rules/modelspec/openai_rules_all_full.jsonl").by_id
    flat, hier = modelspec(), modelspec(reasoning_prompt_style=HIER, individual_rule_context=HIERARCHY_RULE_CONTEXT)
    flat_variants, _ = build_rule_variants(rules, "isolated")
    hier_variants, _ = build_rule_variants(rules, HIERARCHY_RULE_CONTEXT, levels)
    assert len(hier_variants[42][2]) == 24 and hier_variants[27][2] is None  # Rule 43 (user), Rule 28 (root)
    full_flat, full_hier = rule_set_text(rules), rule_set_text(rules, HIERARCHY_RULE_CONTEXT, levels)
    lengths = {"flat_full": 0, "hier_full": 0, "flat_rule": 0, "hier_rule": 0}
    scenarios = load_jsonl_records(population)
    assert len(scenarios) == 500
    for scenario in scenarios:
        lengths["flat_full"] += len(flat.build_eval_prompt(full_flat, scenario, "No_Interpretation"))
        lengths["hier_full"] += len(hier.build_eval_prompt(full_hier, scenario, "No_Interpretation"))
        for (_, text, _), (_, tagged, context) in zip(flat_variants, hier_variants):
            lengths["flat_rule"] += len(flat.build_eval_prompt(text, scenario, "No_Interpretation"))
            lengths["hier_rule"] += len(hier.build_eval_prompt(tagged, scenario, "No_Interpretation",
                                                               context_rules=context))
    assert lengths["hier_rule"] / lengths["flat_rule"] == pytest.approx(1.555, abs=0.001)
    assert lengths["hier_full"] / lengths["flat_full"] == pytest.approx(1.233, abs=0.001)
