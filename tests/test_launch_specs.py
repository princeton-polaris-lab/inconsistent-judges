from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest
import yaml

from ruleforge.core.record_names import RECORDED_BLOCK_KEYS
from ruleforge.launch import cli, control
from ruleforge.launch.campaign import PREFLIGHT_PROGRAM, PROGRAM, resolve_condition
from ruleforge.launch.jobs import JobSpec, Resources, fill, members
from ruleforge.core import record_names
from ruleforge.core.paths import absolute_strings
from ruleforge.evaluate.campaign import load_conditions
from ruleforge.evaluate.identity import LABEL_RULE
from ruleforge.grpo.anchor_set import condition_key
from tests.support.launch_records import launch_site, model_words_now

REPO = Path(__file__).resolve().parents[1]
SPECS = sorted((REPO / control.SPECS_DIR).glob("*.yaml"))
#: Each committed spec's verb: a new spec must be named here, so it is read as its verb reads it.
VERBS = {"ruleforge_modelspec": "job", "ruleforge_smoke": "job", "ruleforge_rulesets": "campaign",
         "prompt_ablation_naive": "campaign", "merge_fp32": "job", "gemma_anchors": "campaign",
         "gemma_references": "campaign", "grpo_rows_gemma": "job",
         "grpo_train_gemma_run": "train", "batch_judge": "campaign",
         "prompt_ablation": "campaign", "training_arm_gemma": "campaign",
         "hierarchy": "campaign", "grpo_train_run": "train", "grpo_anchors": "campaign",
         "training_arms": "campaign",
         "ruleset_size_open_models": "job", "ruleset_size_batch_judge": "job"}
#: A campaign spec's variables as the paper, or the run it records, gave them.
EXAMPLES = {
    "ruleforge_rulesets": {"campaign_id": "fig4-2026-09", "prompt": "naive-rules-above"},
    "prompt_ablation_naive": {"campaign_id": "final-2026-09"},
    "batch_judge": {"campaign_id": "final-2026-09-luna", "model": "gpt-5.6-luna-batch",
                    "population": "populations/frozen_500.jsonl",
                    "population_sha256": "fa9bdc5eb6b9bac8e7b42a27f4ab9f6b60bd5a0e4ac4ae3a63190eb00e487411"},
    "prompt_ablation": {"campaign_id": "prompt-ablation-v2-2026-10"},
    # One model a launch; the three together compose every condition the open judges' launches run
    "hierarchy": {"campaign_id": "hierarchy-aware-2026-10",
                        "models": "gpt-oss-tp-2,olmo3-32b-tp-2,gemma4-31b-tp-2"},
    "gemma_anchors": {"stage": "bank", "campaign_prefix": "gemma-anchors-2026-10-01"},
    "gemma_references": {"stage": "references_bank", "campaign_prefix": "gemma-dev-2026-10-01",
                         "population": "populations/gemma-run/stage1.jsonl", "population_sha256": "ab"},
    "training_arm_gemma": {
        "campaign_id": "fig5-gemma-2026-10-07",
        "checkpoint": "selection/gemma-grpo-mix-2026-10-04/modal_root/finetuning/gemma-grpo-mix-2026-10-04/merged/"
                      "checkpoint-200",
        "checkpoint_sha256": "5f161da83561d405d917aab198b6f79433413669f786a1f2a982c389677b704e"},
    # the specs every model's run takes, with Qwen3-8B's values: its base screen, and its training arm
    "grpo_anchors": {"judge": "qwen3-8b", "weights": "Qwen3-8B", "strict_think": "true", "stage": "mixscreen",
                     "campaign_prefix": "qwen-mix-2026-10-11", "request_block": "250", "timeout_min": "90"},
    "training_arms": {"serving": "merged_fp32", "gpus": "1", "mem_gb": "96",
                     "campaign_id": "fig5-qwen-mix-2026-10-11-checkpoint-100", "policy": "qwen-mix",
                     "checkpoint": "selection/qwen-grpo-mix-2026-10-11/modal_root/finetuning/"
                                   "qwen-grpo-mix-2026-10-11/merged/checkpoint-100",
                     "checkpoint_sha256": "ab" * 32},
}
HOME_OR_USER = re.compile(r"(^|[^\w.])~/|\$\{?(HOME|USER|LOGNAME)\b|oc\.env:\s*(HOME|USER|LOGNAME)\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
#: The ``launch campaign`` command lines the published campaigns ran, one file per spec.
RECIPES = Path(__file__).parent / "fixtures" / "launch" / "recipes"
COLLAPSED = re.compile(r"\S\s+#\s{2,}\S")


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif node is not None:
        yield str(node)


def _options(verb: str) -> dict:
    parser = cli.build_parser()
    [verbs] = [action for action in parser._actions if action.dest == "verb"]
    return cli.spec_options(verbs.choices[verb])


def test_spec_discovery_is_not_vacuous():
    assert {path.stem for path in SPECS} == set(VERBS), "name each spec's verb in VERBS"
    assert {name for name, verb in VERBS.items() if verb == "campaign"} == set(EXAMPLES)


def _declared(path: Path) -> dict:
    data, declared, _ = cli._read_spec(path)
    return cli._preset(path, data, declared, REPO)[2] if cli.PRESET_KEY in data else declared


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_a_spec_reads_as_its_verb_reads_it(path):
    declared = _declared(path)
    required = EXAMPLES.get(path.stem) or {
        name: "x" for name in declared if (declared[name] if isinstance(declared, dict) else None) is None}
    verb = VERBS[path.stem]
    values, arguments, record, _ = cli.load_spec(path, required, "L1", _options(verb), repo=REPO, verb=verb)
    assert record["path"] == f"{control.SPECS_DIR}/{path.name}" and arguments and values["backend"]
    if verb == "job":
        with pytest.raises(cli.LaunchError, match="unknown key.*the job verb's long options"):
            cli.load_spec(REPO / control.SPECS_DIR / "ruleforge_rulesets.yaml", {"campaign_id": "x"}, "L1",
                          _options("job"), repo=REPO)
    if verb == "train":
        assert not {"run-name", "resume", "name", "axis", "resumable", "workdir", "output"} & set(_options("train"))


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_a_campaign_spec_s_conditions_compose_from_the_committed_configs(name, tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))  # a Batch judge's paid state, set where it launches
    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", EXAMPLES[name], "L1",
                                            _options("campaign"), repo=REPO, verb="campaign")
    template = JobSpec("t", "campaign", 0, (*PROGRAM, *arguments), Resources("gpu", gpus=1))
    conditions = members(template, values.get("axis") or ())
    knobs = [f"campaign_id={values['campaign_id']}"] if "campaign_id" in values else []
    if "root_template" in values:
        knobs.append(f"campaign_root={tmp_path}/{values['root_template'].replace('{block}', 'b-000')}")
    resolved = [resolve_condition(REPO, [*member.argv[len(PROGRAM):], *knobs]) for member in conditions]
    assert len({entry["condition_id"] for entry in resolved}) == len(conditions)
    for entry in resolved:
        assert entry["campaign_root"].startswith(str(tmp_path)) and entry["condition_id"]


GEMMA_DESIGN = {
    "judge_model.model_name": "gemma-4-31B-it", "judge_model.model_path": "/models/gemma-4-31B-it",
    "judge_model.strip_thinking_model": "gemma4", "judge_model.enable_thinking": True,
    "judge_model.attention_backend": "TRITON_ATTN", "judge_model.max_new_tokens": 65536,
    "judge_model.adapter_path": None, "approved_model_weights_sha256": None, "approved_adapter_sha256": None,
    "eval_cfg.reasoning_prompt_style": "forced-grpo", "eval_cfg.individual_rule_context": "isolated",
    "interpretation": "No_Interpretation", "sampling.strict_think": False,
    "sampling.forced_json_requirement": "label_only", "sampling.retry_scope": "component",
    "sampling.truncation": "model_default", "sampling.temperature": 1.0,
    "sampling.max_semantic_attempts": 20, "approved_rule_set_sha256": (
        "a99062f00a9ff616cba4d548e9cd59460dc4ee52fc46e3de4a4eb2579a260228"),
}


def _compose_spec(name: str, variables: dict, tmp_path: Path):
    from ruleforge.launch.guards import compose

    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", variables, "L1",
                                            _options("campaign"), repo=REPO, verb="campaign")
    knobs = [f"campaign_root={tmp_path}/root", "scenario_set_path=/p.jsonl", "approved_scenario_sha256=ab"]
    return values, compose(REPO, [*arguments, *knobs], default_config="sampling_campaign")


def _gemma_config(name: str, variables: dict, tmp_path: Path):
    from omegaconf import OmegaConf

    values, cfg = _compose_spec(name, variables, tmp_path)
    return values, {key: OmegaConf.select(cfg, key) for key in (*GEMMA_DESIGN, "judge_model.dtype",
                                                                 "judge_model.tp_size", "condition_id",
                                                                 "evaluation_mode", "sampling.num_decoder_draws",
                                                                 "sampling.generation_seed_base",
                                                                 "sampling.scenarios_per_request_block",
                                                                 "sampling.campaign_mode", "rule_set_path")}


def test_the_gemma_campaigns_judge_one_design_on_their_declared_engines(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    prefix = {"campaign_prefix": "gemma-x"}
    # (variables, evaluation mode, draws, seed base, dtype, TP, scenarios a vLLM call)
    cases = {
        ("gemma_anchors", "bank"): ({**prefix, "stage": "bank"}, "individual_bank", 2, 16, "bfloat16", 1, 50),
        ("gemma_anchors", "top_up"): ({**prefix, "stage": "top_up"}, "individual_bank", 4, 17, "bfloat16", 1, 50),
        ("gemma_anchors", "full"): ({**prefix, "stage": "full", "request_block": "250"}, "full", 2, 16, "bfloat16", 1,
                                    250),
        # the two-stage screen's stages' banks
        ("gemma_anchors", "confirm"): ({**prefix, "stage": "confirm"}, "individual_bank", 4, 17, "bfloat16", 1, 50),
        ("gemma_anchors", "yes_side"): ({**prefix, "stage": "yes_side"}, "individual_bank", 2, 16, "bfloat16", 1, 50),
        # the base screen of the mixed-verdict run: the full set, 8 fresh draws, seed base 20
        ("gemma_anchors", "mixscreen"): ({**prefix, "stage": "mixscreen", "request_block": "250"}, "full", 8, 20,
                                         "bfloat16", 1, 250),
        ("gemma_references", "references_bank"): (
            {**EXAMPLES["gemma_references"]}, "individual_bank", 4, 16, "float32", 2, 50),
        ("gemma_references", "references_full"): (
            {**EXAMPLES["gemma_references"], "stage": "references_full"}, "full", 8, 16, "float32", 2, 50),
    }
    for (name, stage), (variables, mode, draws, seed_base, dtype, tp, block) in cases.items():
        values, cfg = _gemma_config(name, variables, tmp_path)
        design = dict(GEMMA_DESIGN)
        if stage == "screen_rules":  # the prone ruleset in place of the 57 rules, by its approved digest
            design["approved_rule_set_sha256"] = variables["rules_sha256"]
            assert cfg["rule_set_path"] == f"{tmp_path}/{variables['rules']}"
        else:
            assert cfg["rule_set_path"] == "data/rules/modelspec/openai_rules_all.jsonl"
        assert {key: cfg[key] for key in GEMMA_DESIGN} == design, (name, stage)
        assert cfg["sampling.campaign_mode"] == ("screening" if stage.startswith("screen") else "pilot"), stage
        assert (cfg["evaluation_mode"], cfg["sampling.num_decoder_draws"], cfg["sampling.generation_seed_base"],
                cfg["judge_model.dtype"], cfg["judge_model.tp_size"], cfg["sampling.scenarios_per_request_block"]) == (
            mode, draws, seed_base, dtype, tp, block), stage
        assert cfg["condition_id"].startswith("gemma-4-31B-it/grpo-v2-")
        assert (values["backend"], values["gpu_count"], values["qos"], values["resumable"]) == (
            "slurm", tp, "normal", True)
        assert values.get("cpus") == (None if name == "gemma_references" else 4), name
        assert values.get("mem_gb") == (160 if name == "gemma_references" else None), name


def test_the_gemma_base_screen_is_the_full_sets_condition_but_for_its_draws_and_seed_base(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")

    def flat(node: dict, prefix: str = "") -> dict:
        return {key: value for name, child in node.items() for key, value in (
            flat(child, f"{prefix}{name}.") if isinstance(child, dict) else {f"{prefix}{name}": child}).items()}

    def composed(stage: str) -> dict:
        variables = {"campaign_prefix": "gemma-x", "stage": stage, "request_block": "250"}
        return flat(OmegaConf.to_container(_compose_spec("gemma_anchors", variables, tmp_path)[1], resolve=True))

    full, screen = composed("full"), composed("mixscreen")
    assert sorted(key for key in set(full) | set(screen) if full.get(key) != screen.get(key)) == [
        "campaign_id", "condition_id", "grpo_anchors.stage", "grpo_anchors.stages.full.request_block",
        "grpo_anchors.stages.mixscreen.request_block", "sampling.generation_seed_base", "sampling.num_decoder_draws"]


def _spec_arguments(name: str, variables: dict) -> tuple[dict, list[str]]:
    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", variables, "L1",
                                            _options("campaign"), repo=REPO, verb="campaign")
    return values, arguments


#: The evaluation design the arms' configs compose to (Qwen3-8B's strict_think on and its engine's cap).
TRAINING_DESIGN = {"config": "configs/sampling_training.yaml", "prompt_style": "forced-grpo", "judge_type": "No_Interpretation",
                  "individual_rule_context": "isolated", "method": "flat_separate_call", "draws": 5, "temperature": 1.0,
                  "truncation": "model_default", "attempts": 20, "retry_scope": "component", "arms": {"reference": "base"},
                  "validity": {"forced_json_requirement": "label_only", "strict_think": True},
                  "tests": {"bootstrap": {"resamples": 10000, "seed": 16, "level": 0.95}}, "max_new_tokens": 32768}
GEMMA_TRAINING_DESIGN = {**TRAINING_DESIGN, "max_new_tokens": 65536,
                        "validity": {"forced_json_requirement": "label_only", "strict_think": False}}
GEMMA_SERVING = {"engine": "configs/model/gemma4-31b-fp32.yaml", "dtype": "float32",
                 "campaign_serving": "configs/training_serving/gemma_merged_fp32.yaml"}
#: What Gemma's run recorded of its serving.
GEMMA_RUN_RECORD = "selection/gemma-grpo-mix-2026-10-04/prereg/prereg.json"
RULESET_PATH = "data/rules/modelspec/openai_rules_all.jsonl"
RULESET_SHA256 = "a99062f00a9ff616cba4d548e9cd59460dc4ee52fc46e3de4a4eb2579a260228"
#: The population the published runs' dev scenarios are held apart from.
DEV_HELD_APART_FROM = "fa9bdc5eb6b9bac8e7b42a27f4ab9f6b60bd5a0e4ac4ae3a63190eb00e487411"
PAPER_SCENARIOS = 500
PAPER_POPULATION = {"name": "frozen_500_r3",
                    "sha256": "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4"}


def _gemma_training_arms(variables: dict) -> tuple[dict, list[str], dict]:
    """training_arm_gemma's values, its arguments and ``{policy: the composed arm}``."""
    values, arguments = _spec_arguments("training_arm_gemma", variables)
    arms = _members_composed(values, arguments, knobs=(f"campaign_id={variables['campaign_id']}",),
                             key=lambda flat: flat["training"]["policy"])
    return values, arguments, arms


def test_gemma_s_training_arms_are_the_design_on_gemma_s_fp32_engine(tmp_path, monkeypatch):
    from ruleforge.evaluate.sharding import assign_units, blocks_per_draw

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    design, serving = GEMMA_TRAINING_DESIGN, GEMMA_SERVING
    engine = yaml.safe_load((REPO / serving["engine"]).read_text(encoding="utf-8"))
    values, arguments, arms = _gemma_training_arms(EXAMPLES["training_arm_gemma"])
    assert arguments[:3] == ["--config-name", Path(design["config"]).stem,
                             f"training_serving={Path(serving['campaign_serving']).stem}"]
    assert sorted(arms) == ["base", "grpo"]
    for policy, cfg in arms.items():
        flat = _flat(cfg)
        assert {key: flat[key] for key in (
            "eval_cfg.reasoning_prompt_style", "sampling.forced_json_requirement", "sampling.strict_think",
            "sampling.num_decoder_draws", "sampling.max_semantic_attempts", "sampling.retry_scope",
            "approved_scenario_sha256", "sampling.temperature", "sampling.truncation", "judge_model.max_new_tokens",
            "interpretation", "eval_cfg.individual_rule_context", "approved_rule_set_sha256",
            "sampling.bootstrap.n_resamples", "sampling.bootstrap.seed", "sampling.bootstrap.ci_level")} == {
            "eval_cfg.reasoning_prompt_style": design["prompt_style"],
            "sampling.forced_json_requirement": design["validity"]["forced_json_requirement"],
            "sampling.strict_think": design["validity"]["strict_think"],
            "sampling.num_decoder_draws": design["draws"], "sampling.max_semantic_attempts": design["attempts"],
            "sampling.retry_scope": design["retry_scope"], "approved_scenario_sha256": PAPER_POPULATION["sha256"],
            "sampling.temperature": design["temperature"], "sampling.truncation": design["truncation"],
            "judge_model.max_new_tokens": design["max_new_tokens"], "interpretation": design["judge_type"],
            "eval_cfg.individual_rule_context": design["individual_rule_context"],
            "approved_rule_set_sha256": RULESET_SHA256,
            "sampling.bootstrap.n_resamples": design["tests"]["bootstrap"]["resamples"],
            "sampling.bootstrap.seed": design["tests"]["bootstrap"]["seed"],
            "sampling.bootstrap.ci_level": design["tests"]["bootstrap"]["level"]}, policy
        assert (design["method"], flat["evaluation_mode"], flat["sampling.campaign_mode"]) == (
            "flat_separate_call", "both", "final")
        assert flat["scenario_set_path"] == f"{tmp_path}/populations/{PAPER_POPULATION['name']}.jsonl"
        assert flat["rule_set_path"].endswith(RULESET_PATH)
        assert {key: flat[f"judge_model.{key}"] for key in (
            "model_name", "dtype", "tp_size", "strip_thinking_model", "reasoning_parser", "attention_backend",
            "enable_thinking", "max_new_tokens")} == {key: engine[key] for key in (
                "model_name", "dtype", "tp_size", "strip_thinking_model", "reasoning_parser", "attention_backend",
                "enable_thinking", "max_new_tokens")}
        assert (flat["judge_model.dtype"], flat["judge_model.tp_size"], flat["judge_model.strip_thinking_model"]) == (
            serving["dtype"], 2, "gemma4")
        assert flat["condition_id"] == f"gemma-4-31B-it/figure5/{policy}"
        assert (flat["condition_extra.figure"], flat["condition_extra.policy_family"],
                flat["condition_extra.policy_reference"]) == ("figure5", "base+grpo", design["arms"]["reference"])
        assert flat["campaign_root"] == f"{tmp_path}/sampling/{EXAMPLES['training_arm_gemma']['campaign_id']}"
    assert (values["backend"], values["gpu_count"], values["mem_gb"], values["timeout_min"], values["qos"],
            values["resumable"], values["pack_draws"]) == ("slurm", engine["tp_size"], 160, 360, "normal", True, True)
    assert values["mem_gb"] > 131 and "cpus" not in values  # the launcher's 4 CPUs a GPU
    assert (values["population"], values["expect_population_sha256"]) == (
        f"populations/{PAPER_POPULATION['name']}.jsonl", PAPER_POPULATION["sha256"])
    cfg = arms["base"]
    blocks = blocks_per_draw(PAPER_SCENARIOS, cfg.sampling.scenarios_per_request_block)
    shards = assign_units(range(cfg.sampling.num_decoder_draws), blocks, values["shards"])
    assert (blocks, values["shards"], [len(units) for units in shards]) == (10, 25, [2] * 25)


def test_gemma_s_training_arms_differ_in_the_policy_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    example = EXAMPLES["training_arm_gemma"]
    _, _, arms = _gemma_training_arms(example)
    base, policy = _flat(arms["base"]), _flat(arms["grpo"])
    assert {key: (base.get(key), policy.get(key)) for key in set(base) | set(policy)
            if base.get(key) != policy.get(key)} == {
        "approved_model_weights_sha256": (None, example["checkpoint_sha256"]),
        "condition_extra.policy": ("base", "grpo"),
        "condition_id": ("gemma-4-31B-it/figure5/base", "gemma-4-31B-it/figure5/grpo"),
        "training.policy": ("base", "grpo"),
        "judge_model.model_path": ("/models/gemma-4-31B-it", f"{tmp_path}/{example['checkpoint']}")}
    args = cli.build_parser().parse_args(["campaign", "--spec", "training_arm_gemma", "--axis", "training.policy=grpo",
                                          *[item for name, value in example.items()
                                            for item in ("--var", f"{name}={value}")]])
    args.args = []
    settings, _, _ = cli.job_settings(args, args.options_parser, "L1", REPO)
    assert settings["axis"] == ["training.policy=grpo"] and settings["shards"] == 25
    with pytest.raises(cli.LaunchError, match="gives base more than once"):
        _gemma_training_arms({**example, "policy": "base"})


def test_gemma_s_training_arms_pass_their_preflights_as_the_launcher_plans_them(tmp_path, monkeypatch):
    from ruleforge.grpo import merging as merge_module
    from ruleforge.launch.campaign import CampaignOptions, plan
    from ruleforge.launch.guards import compose
    from ruleforge.evaluate.runner import CampaignRunError
    from ruleforge.evaluate.campaign import preflight
    from ruleforge.core.digests import sha256_file
    from tests.test_sampling_weights import write_merged, write_weights

    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.chdir(REPO)  # the configs' relative paths are the checkout's, where the jobs run
    write_weights(tmp_path / "models" / "gemma-4-31B-it", seed=-2.0)
    example = EXAMPLES["training_arm_gemma"]
    merged = write_merged((artifacts / example["checkpoint"]).parent, "checkpoint-200", seed=3.0)
    digest = merge_module.weights_digest(merged)
    population = artifacts / "populations" / f"{PAPER_POPULATION['name']}.jsonl"
    population.parent.mkdir(parents=True)
    population.write_text("".join(f'{{"prompt": "p{index}", "response": "r{index}"}}\n'
                                  for index in range(PAPER_SCENARIOS)), encoding="utf-8")

    def preflighted(checkpoint_sha256: str) -> dict:
        values, arguments = _spec_arguments("training_arm_gemma", {**example, "checkpoint_sha256": checkpoint_sha256})
        options = CampaignOptions.from_settings({**values, "expect_population_sha256": sha256_file(population)})
        template = JobSpec("training_arm_gemma", "campaign", 0, PROGRAM,
                           Resources("gpu", gpus=values["gpu_count"]))
        conditions = plan(REPO, template, arguments, values["axis"], options, exists=lambda root: False)
        # the checkpoint arm first: its shards head the job array (the stop rules read it before most of the base)
        assert [(condition.condition_id, condition.shards) for condition in conditions] == [
            ("gemma-4-31B-it/figure5/grpo", 25), ("gemma-4-31B-it/figure5/base", 25)]
        composed = {}
        for condition in conditions:
            argv = condition.preflight(0).argv
            assert argv[:3] == PREFLIGHT_PROGRAM and argv[-1].startswith("hydra.run.dir=")
            cfg = compose(REPO, list(argv[3:-1]), default_config="sampling_campaign")
            composed[str(cfg.training.policy)] = cfg
        return composed

    arms = preflighted(digest)
    for cfg in arms.values():
        config, manifest = preflight(cfg)
        assert (manifest.record_count, config.num_decoder_draws, cfg.execution.num_shards, cfg.execution.shard_index,
                cfg.execution.pack_draws) == (PAPER_SCENARIOS, 5, 25, 0, True)
    with pytest.raises(CampaignRunError, match=f"model weights SHA-256 {digest} .* does not match approved "
                                               f"{example['checkpoint_sha256']}"):
        preflight(preflighted(example["checkpoint_sha256"])["grpo"])
    base, spec = _condition_spec(arms["base"]), _condition_spec(arms["grpo"])
    assert (spec.model_path, spec.model_weights_sha256, spec.serving_dtype, spec.strict_think) == (
        "checkpoint-200", digest, "float32", False)
    assert (base.model_path, base.model_weights_sha256, base.serving_dtype, dict(base.extra)["policy"]) == (
        "gemma-4-31B-it", None, "float32", "base")
    swapped_back = dataclasses.replace(spec, condition_id=base.condition_id, model_path=base.model_path,
                                       model_weights_sha256=base.model_weights_sha256, extra=base.extra)
    assert spec.fingerprint != base.fingerprint and swapped_back.fingerprint == base.fingerprint


@pytest.mark.cluster
def test_gemma_s_training_inputs_are_its_recorded_serving_a_merged_checkpoint_the_base_and_the_frozen_500():
    import json
    import os

    from ruleforge.grpo import merging as merge_module
    from ruleforge.core.digests import sha256_file

    root, models = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")), os.environ.get("RULEFORGE_MODEL_ROOT")
    example = EXAMPLES["training_arm_gemma"]
    checkpoint = root / example["checkpoint"]
    if not checkpoint.is_dir() or not models:
        pytest.skip("needs the Gemma artifact root (its merged checkpoint-200) and RULEFORGE_MODEL_ROOT")
    manifest = json.loads((checkpoint / merge_module.MANIFEST_FNAME).read_text(encoding="utf-8"))
    assert manifest["result"]["weights_sha256"] == example["checkpoint_sha256"]
    assert merge_module.weights_digest(checkpoint) == example["checkpoint_sha256"]
    base = Path(models) / "gemma-4-31B-it"
    assert Path(manifest["base"]["path"]).name == base.name
    assert merge_module.weights_digest(base) == manifest["base"]["weights_sha256"]
    assert sha256_file(root / "populations" / "frozen_500.jsonl") == DEV_HELD_APART_FROM
    recorded = json.loads((root / GEMMA_RUN_RECORD).read_text(encoding="utf-8"))["serving"]
    assert {key: recorded[key] for key in ("engine", "dtype")} == {key: GEMMA_SERVING[key] for key in ("engine", "dtype")}


def _composed_members(name: str, variables: dict) -> tuple[dict, dict]:
    """The spec's resolved vars, and ``{(model, style): the composed config}`` of every condition it launches."""
    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", variables, "L1",
                                            _options("campaign"), repo=REPO, verb="campaign")
    return values, _members_composed(values, arguments)


def _members_composed(
    settings: dict, arguments: list[str],
    knobs: tuple[str, ...] = ("campaign_root=/r", "scenario_set_path=/p.jsonl", "approved_scenario_sha256=ab"),
    key=lambda flat: (flat["judge_model"]["model_name"], flat["eval_cfg"]["reasoning_prompt_style"]),
) -> dict:
    """``{key: the composed config}`` of every condition a campaign launch fans out to, ``knobs`` composed after."""
    from omegaconf import OmegaConf

    from ruleforge.launch.guards import compose

    template = JobSpec("t", "campaign", 0, (*PROGRAM, *arguments), Resources("gpu", gpus=1))
    out = {}
    for member in members(template, settings["axis"]):
        cfg = compose(REPO, [*member.argv[len(PROGRAM):], *knobs], default_config="sampling_campaign")
        out[key(OmegaConf.to_container(cfg, resolve=True))] = cfg
    return out


def _header_recipes(name: str, campaign_id: str) -> list[tuple[dict, dict]]:
    return [(settings, _members_composed(settings, arguments))
            for settings, arguments in _header_jobs(name, campaign_id)]


def _header_jobs(name: str, campaign_id: str) -> list[tuple[dict, list[str]]]:
    import shlex

    commands, pending = [], ""
    for line in (RECIPES / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        pending = f"{pending} {line.strip()}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1]
            continue
        if pending.startswith("launch campaign ") and f"campaign_id={campaign_id}" in pending.split():
            commands.append(shlex.split(pending)[1:])
        pending = ""
    jobs = []
    for argv in commands:
        split = argv.index("--") if "--" in argv else len(argv)
        args = cli.build_parser().parse_args(argv[:split])
        args.args = argv[split + 1:]  # what follows ``--``, which cli.main splits off before parsing
        settings, arguments, _ = cli.job_settings(args, args.options_parser, "L1", REPO)
        jobs.append((settings, arguments))
    return jobs


def _walk(node: dict, prefix: str = "") -> dict:
    return {key: value for name, child in node.items() for key, value in (
        _walk(child, f"{prefix}{name}.") if isinstance(child, dict) else {f"{prefix}{name}": child}).items()}


def _flat(cfg, *, resolve: bool = True) -> dict:
    from omegaconf import OmegaConf

    return _walk(OmegaConf.to_container(cfg, resolve=resolve))


def _condition_spec(cfg, *, model_type: str = "local", seed_capability: str = "local_deterministic"):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from ruleforge.evaluate.campaign import condition_extra, condition_spec_for
    from ruleforge.evaluate.identity import SamplingConfig

    backend = SimpleNamespace(model_name=str(cfg.judge_model.model_name), model_type=model_type,
                              max_new_tokens=int(cfg.judge_model.max_new_tokens), seed_capability=seed_capability)
    task_spec = SimpleNamespace(reasoning_prompt_style=str(cfg.eval_cfg.reasoning_prompt_style),
                                individual_rule_context=str(cfg.eval_cfg.individual_rule_context))
    return condition_spec_for(
        condition_id=str(cfg.condition_id), backend=backend, task_spec=task_spec,
        config=SamplingConfig.from_mapping(OmegaConf.to_container(cfg.sampling, resolve=True)),
        judge_type=str(cfg.interpretation), rule_set_path=(REPO / str(cfg.rule_set_path)),
        model_cfg=cfg.judge_model, extra=condition_extra(cfg), evaluation_mode=str(cfg.evaluation_mode),
        rule_levels_sha256=(hashlib.sha256((REPO / str(cfg.rule_levels_path)).read_bytes()).hexdigest()
                            if cfg.get("rule_levels_path") else None),
    )


RECORDED_STYLES = "naive-rules-above,forced-grpo"
RECORDED_REVISED_STYLES = "naive,encouraged,forced"
#: The paper's three prompts: the published naive prompt beside the paper's encouraged and forced ones.
PAPER_STYLES = "naive-rules-above,encouraged,forced"
LUNA_FINAL_OVERRIDES = ["sampling.campaign_mode=final", "sampling.num_decoder_draws=5", "sampling.max_semantic_attempts=20",
                        "sampling.scenarios_per_request_block=500",
                        "model@judge_model=gpt-5.6-luna-batch", "model@agent_model=gpt-5.6-luna-batch"]
LUNA_FINAL_FINGERPRINTS = {
    "naive-rules-above": "c813a2685fc4d1754db675780a8e48de3e82e2248374e20f66c42b655d2ea752",
    "forced-grpo": "a6b1d045f792655a195bb56816d7f5df97a0b38b3ebfc2428b4edaa3e30b93d0",
}
RECORDED_LABEL_RULE = "lenient"


def _as_recorded(spec, label_rule: str = RECORDED_LABEL_RULE):
    assert spec.label_rule == LABEL_RULE
    return dataclasses.replace(spec, label_rule=label_rule)


BATCH_PILOT = {"campaign_id": "pilot-batch-2026-10", "model": "gpt-5.6-luna-batch",
               "population": "sampling/subsets/frozen_500_12_seed16_by_source.jsonl",
               "population_sha256": "3be28ab38b0ec0261e8c5f4235db15524284d9931d51283fa13f496883799c8c",
               "request_block": "12", "mode": "screening", "draws": "1"}


def _batch_spec(cfg):
    """The condition identity of an openai_batch judge on the responses endpoint, which sends no seed."""
    return _condition_spec(cfg, model_type="openai_batch", seed_capability="unsupported")


def test_the_batch_judge_spec_is_luna_s_prompt_table_launch_and_a_pilot_differs_in_the_campaign_alone(tmp_path, monkeypatch):
    from ruleforge.launch.guards import compose

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "/batch")
    values, luna = _composed_members("batch_judge", {**EXAMPLES["batch_judge"], "styles": RECORDED_STYLES})
    assert (values["backend"], values["population"], values["expect_population_sha256"], values["campaign_id"]) == (
        "local", "populations/frozen_500.jsonl", "fa9bdc5eb6b9bac8e7b42a27f4ab9f6b60bd5a0e4ac4ae3a63190eb00e487411",
        "final-2026-09-luna")
    pilot_values, pilot = _composed_members("batch_judge", {**BATCH_PILOT, "styles": RECORDED_STYLES})
    assert sorted(luna) == [("gpt-5.6-luna", style) for style in ("forced-grpo", "naive-rules-above")]
    for style, fingerprint in LUNA_FINAL_FINGERPRINTS.items():
        recorded = compose(REPO, [*LUNA_FINAL_OVERRIDES, f"eval_cfg.reasoning_prompt_style={style}", "campaign_root=/r",
                                  "scenario_set_path=/p.jsonl", "approved_scenario_sha256=ab"],
                           default_config="sampling_campaign")
        arm, cut = luna[("gpt-5.6-luna", style)], pilot[("gpt-5.6-luna", style)]
        flat_arm, flat_cut = map(_flat, (arm, cut))
        assert flat_arm == _flat(recorded)
        assert _as_recorded(_batch_spec(arm)).fingerprint == fingerprint
        assert sorted(key for key in flat_arm if flat_arm[key] != flat_cut.get(key)) == [
            "sampling.campaign_mode", "sampling.num_decoder_draws", "sampling.scenarios_per_request_block"]
        assert _as_recorded(_batch_spec(cut)).fingerprint == fingerprint
        assert [flat_cut[key] for key in (
            "sampling.campaign_mode", "sampling.num_decoder_draws", "sampling.scenarios_per_request_block",
            "sampling.max_semantic_attempts", "sampling.forced_json_requirement",
            "eval_cfg.individual_rule_context", "interpretation")] == [
            "screening", 1, 12, 20, "complete_rule_map", "isolated", "No_Interpretation"]
    assert (pilot_values["population"], pilot_values["campaign_id"]) == (BATCH_PILOT["population"], "pilot-batch-2026-10")


def test_the_prompt_table_specs_launch_the_paper_s_styles_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "/batch")
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    _, naive = _composed_members("prompt_ablation_naive", EXAMPLES["prompt_ablation_naive"])
    _, recorded = _composed_members("prompt_ablation_naive", {**EXAMPLES["prompt_ablation_naive"], "styles": RECORDED_STYLES})
    assert sorted(naive) == [(model, "naive-rules-above") for model in ("Olmo-3-32B-Think", "gemma-4-31B-it", "gpt-oss-120b")]
    assert {key: _flat(cfg) for key, cfg in naive.items()} == {key: _flat(recorded[key]) for key in naive}
    _, luna = _composed_members("batch_judge", EXAMPLES["batch_judge"])
    assert sorted(luna) == [("gpt-5.6-luna", style) for style in ("encouraged", "forced", "naive-rules-above")]


#: Each paper style's published arm, the same condition but for the style: encouraged has no counterpart of its own, so
#: its arm is held to the published naive arm.
REVISED_COUNTERPARTS = {"naive": "naive-rules-above", "encouraged": "naive-rules-above", "forced": "forced-grpo"}


BATCH_PILOT_REVISED = {**BATCH_PILOT, "campaign_id": "pilot-batch-revised-2026-10", "styles": RECORDED_REVISED_STYLES}


def test_the_revised_pilot_is_the_published_pilot_but_for_the_style(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from ruleforge.evaluate.runner import CampaignRunError
    from ruleforge.evaluate.campaign import expected_requests_per_draw, validate_campaign_cfg
    from ruleforge.scenarios.manifest import ScenarioManifest
    from ruleforge.core.artifacts import load_jsonl_records

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "/batch")
    values, published = _composed_members("batch_judge", {**BATCH_PILOT, "styles": RECORDED_STYLES})
    revised_values, revised = _composed_members("batch_judge", BATCH_PILOT_REVISED)
    model = "gpt-5.6-luna"
    assert sorted(revised) == sorted((model, style) for style in REVISED_COUNTERPARTS)
    assert sorted(key for key in set(values) | set(revised_values) if values.get(key) != revised_values.get(key)) == [
        "axis", "campaign_id"]
    assert (revised_values["campaign_id"], revised_values["axis"]) == (
        "pilot-batch-revised-2026-10", ["eval_cfg.reasoning_prompt_style=naive,encouraged,forced"])
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(f'{{"prompt": "p{index}", "response": "r{index}"}}\n' for index in range(12)))
    rules = load_jsonl_records(REPO / "data/rules/modelspec/openai_rules_all.jsonl")
    for style, counterpart in REVISED_COUNTERPARTS.items():
        arm, reference = revised[(model, style)], published[(model, counterpart)]
        flat_arm, flat_reference = _flat(arm), _flat(reference)
        assert sorted(key for key in set(flat_arm) | set(flat_reference)
                      if flat_arm.get(key) != flat_reference.get(key)) == ["condition_id", "eval_cfg.reasoning_prompt_style"]
        assert [flat_arm[key] for key in ("condition_id", "interpretation", "evaluation_mode", "sampling.campaign_mode",
                                          "sampling.num_decoder_draws", "sampling.scenarios_per_request_block",
                                          "sampling.max_semantic_attempts",
                                          "sampling.forced_json_requirement", "eval_cfg.individual_rule_context")] == [
            f"{model}/{style}", "No_Interpretation", "both", "screening", 1, 12, 20, "complete_rule_map",
            "isolated"]
        validate_campaign_cfg(arm)
        other = arm.copy()
        OmegaConf.update(other, "interpretation", "Narrow")
        if style == "naive":
            OmegaConf.update(other, "eval_cfg.judge_types", ["Narrow"])
            validate_campaign_cfg(other)
        else:
            with pytest.raises(CampaignRunError, match=f"'{style}' is defined for the No_Interpretation template only"):
                validate_campaign_cfg(other)
        spec, reference_spec = _batch_spec(arm), _batch_spec(reference)
        assert spec.fingerprint != reference_spec.fingerprint
        assert dataclasses.replace(spec, prompt_style=counterpart).fingerprint == reference_spec.fingerprint
        assert expected_requests_per_draw(ScenarioManifest.build(cut), rules, spec.evaluation_mode) == 696


def test_the_ruleset_campaign_s_prompt_is_a_launch_variable(tmp_path, monkeypatch):
    from ruleforge.evaluate.campaign import validate_campaign_cfg

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    example = {"campaign_id": EXAMPLES["ruleforge_rulesets"]["campaign_id"]}
    _, default = _composed_members("ruleforge_rulesets", example)
    assert {style for _, style in default} == {"naive"}
    _, chosen = _composed_members("ruleforge_rulesets", {**example, "prompt": "naive-rules-above"})
    assert {style for _, style in chosen} == {"naive-rules-above"}
    for cfg in chosen.values():
        validate_campaign_cfg(cfg)


HAIKU_PILOT_REVISED = {**BATCH_PILOT_REVISED, "campaign_id": "pilot-haiku-v2-2026-10", "model": "claude-haiku-5-5-batch",
                  "population": "sampling/subsets/frozen_500_r1_12_seed16_by_source.jsonl", "styles": PAPER_STYLES}


def test_the_haiku_revised_pilot_is_the_batch_pilot_but_for_the_judge_and_the_header_launches_it(tmp_path, monkeypatch):
    from ruleforge.evaluate import cost as batch_cost
    from ruleforge.evaluate.campaign import expected_requests_per_draw, validate_campaign_cfg
    from ruleforge.scenarios.manifest import ScenarioManifest
    from ruleforge.core.artifacts import load_jsonl_records
    from ruleforge.judges.backend import ModelBackend

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    reference_values, references = _composed_members("batch_judge", {**BATCH_PILOT_REVISED, "styles": PAPER_STYLES})
    values, haiku = _composed_members("batch_judge", HAIKU_PILOT_REVISED)
    [(settings, recipe)] = _header_recipes("batch_judge", HAIKU_PILOT_REVISED["campaign_id"])
    assert {key: settings[key] for key in values if key != "name"} == {key: values[key] for key in values
                                                                      if key != "name"}
    assert {key: _flat(cfg) for key, cfg in recipe.items()} == {key: _flat(cfg) for key, cfg in haiku.items()}
    assert sorted(key for key in set(values) | set(reference_values) if values.get(key) != reference_values.get(key)) == [
        "campaign_id", "population"]
    assert (values["population"], values["expect_population_sha256"]) == (
        "sampling/subsets/frozen_500_r1_12_seed16_by_source.jsonl",
        "3be28ab38b0ec0261e8c5f4235db15524284d9931d51283fa13f496883799c8c")
    model = "claude-haiku-5-5"
    assert sorted(haiku) == sorted((model, style) for style in PAPER_STYLES.split(","))
    cut = tmp_path / "cut.jsonl"
    cut.write_text("".join(f'{{"prompt": "p{index}", "response": "r{index}"}}\n' for index in range(12)))
    rules = load_jsonl_records(REPO / "data/rules/modelspec/openai_rules_all.jsonl")
    requests = 0
    for style in PAPER_STYLES.split(","):
        arm, reference = haiku[(model, style)], references[("gpt-5.6-luna", style)]
        flat_arm, flat_reference = _flat(arm), _flat(reference)
        assert {key.split(".")[0] for key in set(flat_arm) | set(flat_reference)
                if flat_arm.get(key) != flat_reference.get(key)} == {"condition_id", "judge_model", "agent_model"}
        assert [flat_arm[key] for key in ("condition_id", "judge_model.model_type", "interpretation",
                                          "evaluation_mode", "sampling.campaign_mode", "sampling.num_decoder_draws",
                                          "sampling.scenarios_per_request_block", "sampling.max_semantic_attempts",
                                          "sampling.forced_json_requirement",
                                          "eval_cfg.individual_rule_context")] == [
            f"{model}/{style}", "anthropic_batch", "No_Interpretation", "both", "screening", 1, 12, 20,
            "complete_rule_map", "isolated"]
        validate_campaign_cfg(arm)
        backend = ModelBackend(arm.judge_model)
        assert (backend.model_type, backend.seed_capability) == ("anthropic_batch", "unsupported")
        spec = _condition_spec(arm, model_type=backend.model_type, seed_capability=backend.seed_capability)
        reference_spec = _batch_spec(reference)
        for part in (batch_cost.PROMPT_FIELDS, batch_cost.PROTOCOL_FIELDS):
            assert {name: getattr(spec, name) for name in part} == {name: getattr(reference_spec, name) for name in part}
        assert sorted(name for name in batch_cost.JUDGE_FIELDS if getattr(spec, name) != getattr(reference_spec, name)) == [
            "endpoint", "max_new_tokens", "model_name", "model_type", "send_max_output_tokens"]
        assert (spec.endpoint, spec.send_max_output_tokens, spec.reasoning_effort,
                spec.max_new_tokens, spec.temperature, spec.sampling_truncation, spec.seed_capability) == (
            None, True, "medium", 16384, 1.0, "provider_default", "unsupported")
        requests += expected_requests_per_draw(ScenarioManifest.build(cut), rules, spec.evaluation_mode)
    assert requests == 2_088


#: Claude Haiku 5.5's final recipe: the forced style's pilot pass into its own root, and the full runs.
HAIKU_FORCED_PASS = {**HAIKU_PILOT_REVISED, "campaign_id": "pilot-haiku-forced-v2-2026-10", "styles": "forced",
                     "retry_scope": "component"}
HAIKU_REVISED_FULL = "prompt-ablation-v2-2026-10-haiku-sum"


def test_haiku_s_final_recipe_is_its_pilot_but_for_the_retry_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    model = "claude-haiku-5-5"
    _, pilot = _composed_members("batch_judge", HAIKU_PILOT_REVISED)
    values, forced = _composed_members("batch_judge", HAIKU_FORCED_PASS)
    [(settings, recipe)] = _header_recipes("batch_judge", HAIKU_FORCED_PASS["campaign_id"])
    assert {key: settings[key] for key in values if key != "name"} == {key: values[key] for key in values
                                                                      if key != "name"}
    assert {key: _flat(cfg) for key, cfg in recipe.items()} == {key: _flat(cfg) for key, cfg in forced.items()}
    assert sorted(forced) == [(model, "forced")]
    flat_pilot, flat_forced = _flat(pilot[(model, "forced")]), _flat(forced[(model, "forced")])
    assert {key for key in set(flat_pilot) | set(flat_forced) if flat_pilot.get(key) != flat_forced.get(key)} == {
        "sampling.retry_scope"}
    assert (values["campaign_id"], settings["campaign_id"]) == (HAIKU_FORCED_PASS["campaign_id"],) * 2
    assert (flat_forced["sampling.retry_scope"], flat_pilot["sampling.retry_scope"]) == ("component", "scenario_bundle")
    pilot_spec = _condition_spec(pilot[(model, "forced")], model_type="anthropic_batch", seed_capability="unsupported")
    forced_spec = _condition_spec(forced[(model, "forced")], model_type="anthropic_batch",
                                  seed_capability="unsupported")
    assert pilot_spec.fingerprint != forced_spec.fingerprint
    assert (pilot_spec.label_rule, forced_spec.label_rule) == (LABEL_RULE, LABEL_RULE)
    assert dataclasses.replace(forced_spec, retry_scope="scenario_bundle").fingerprint == pilot_spec.fingerprint
    [(full_settings, full)] = _header_recipes("batch_judge", HAIKU_REVISED_FULL)
    assert sorted(full) == sorted((model, style) for style in PAPER_STYLES.split(","))
    assert (full_settings["population"], full_settings["expect_population_sha256"]) == (
        "populations/frozen_500_r3.jsonl", "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4")
    for style in PAPER_STYLES.split(","):
        flat = _flat(full[(model, style)])
        assert [flat[key] for key in ("sampling.campaign_mode", "sampling.num_decoder_draws",
                                      "sampling.scenarios_per_request_block", "sampling.max_semantic_attempts",
                                      "sampling.retry_scope")] == ["final", 5, 500, 20, "component"]
        assert {key for key in set(flat) | set(flat_forced) if flat.get(key) != flat_forced.get(key)} <= {
            "campaign_id", "campaign_root", "condition_id", "eval_cfg.reasoning_prompt_style", "scenario_set_path",
            "approved_scenario_sha256", "sampling.campaign_mode", "sampling.num_decoder_draws",
            "sampling.scenarios_per_request_block"}


RECORDED_REVISED_FINGERPRINTS = {
    "prompt-ablation-v2-2026-10-luna": {
        "encouraged": "1cb8f9128f0eb381321a3f31def7905150ede21904ce95630229f283c245d165",
        "forced": "e0f8040ae8cb23ea59357b1e9091ae943e6f232e97140c136bcfc0b7be148075"},
}
RECORDED_REVISED_LABEL_RULES = {"prompt-ablation-v2-2026-10-luna": "lenient_markup"}


def test_the_batch_judges_revised_runs_compose_to_the_fingerprints_their_roots_recorded(tmp_path, monkeypatch):
    from ruleforge.judges.backend import ModelBackend

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    composed = {LUNA_REVISED: {key: cfg for _, arms in _header_recipes("batch_judge", LUNA_REVISED)
                          for key, cfg in arms.items()}}
    found = {}
    for campaign, arms in composed.items():
        for (_, style), cfg in arms.items():
            backend = ModelBackend(cfg.judge_model)
            found.setdefault(campaign, {})[style] = _as_recorded(_condition_spec(
                cfg, model_type=backend.model_type, seed_capability=backend.seed_capability),
                RECORDED_REVISED_LABEL_RULES[campaign]).fingerprint
    assert found == RECORDED_REVISED_FINGERPRINTS


def test_the_revised_arms_are_prompt_table_s_gemma_arms_but_for_the_prompt(tmp_path, monkeypatch):
    from ruleforge.evaluate.campaign import validate_campaign_cfg

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    published_values, published = _composed_members("prompt_ablation_naive", {**EXAMPLES["prompt_ablation_naive"],
                                                                      "styles": RECORDED_STYLES})
    values, arms = _composed_members("prompt_ablation", EXAMPLES["prompt_ablation"])
    gemma = "gemma-4-31B-it"
    assert sorted(arms) == sorted((gemma, style) for style in OPEN_REVISED_STYLES)
    _, recorded = _composed_members("prompt_ablation", {**EXAMPLES["prompt_ablation"],
                                                           "styles": RECORDED_REVISED_STYLES})
    assert {key: _flat(cfg) for key, cfg in arms.items()} == {key: _flat(recorded[key]) for key in arms}
    arms = recorded
    assert sorted(arms) == sorted((gemma, style) for style in REVISED_COUNTERPARTS)
    for style, counterpart in REVISED_COUNTERPARTS.items():
        validate_campaign_cfg(arms[(gemma, style)])
        arm, reference = _flat(arms[(gemma, style)]), _flat(published[(gemma, counterpart)])
        assert sorted(key for key in set(arm) | set(reference) if arm.get(key) != reference.get(key)) == [
            "condition_id", "eval_cfg.reasoning_prompt_style"]
        assert (arm["condition_id"], arm["eval_cfg.reasoning_prompt_style"]) == (f"{gemma}/{style}", style)
        assert (arm["sampling.num_decoder_draws"], arm["sampling.max_semantic_attempts"], arm["judge_model.tp_size"],
                arm["sampling.campaign_mode"], arm["sampling.forced_json_requirement"],
                arm["interpretation"], arm["eval_cfg.individual_rule_context"]) == (
            5, 20, 1, "final", "complete_rule_map", "No_Interpretation", "isolated")
    differing = sorted(key for key in set(values) | set(published_values) if values.get(key) != published_values.get(key))
    assert differing == ["axis", "campaign_id", "shards", "timeout_min"]
    assert (values["shards"], values["timeout_min"], values["qos"], values["resumable"]) == (5, 360, "normal", True)
    assert (values["population"], values["expect_population_sha256"]) == (
        published_values["population"], published_values["expect_population_sha256"])


@pytest.mark.cluster
def test_each_revised_arm_is_its_published_counterpart_s_condition_but_for_the_style_by_fingerprint():
    import os

    published = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / "sampling/final-2026-09"
    if not (published / "gemma-4-31B-it").is_dir() or not os.environ.get("RULEFORGE_MODEL_ROOT"):
        pytest.skip("needs the published root and RULEFORGE_MODEL_ROOT")
    _, arms = _composed_members("prompt_ablation", {**EXAMPLES["prompt_ablation"], "styles": RECORDED_REVISED_STYLES})
    assert sorted(style for _, style in arms) == sorted(REVISED_COUNTERPARTS)
    registered = load_conditions(published, skip_retired=True)  # the published arms, read as recorded
    for (model, style), cfg in arms.items():
        spec = _as_recorded(_condition_spec(cfg))
        reference = record_names.locate_condition(registered, f"{model}/{REVISED_COUNTERPARTS[style]}").fingerprint
        assert spec.fingerprint != reference
        assert dataclasses.replace(spec, prompt_style=REVISED_COUNTERPARTS[style]).fingerprint == reference


OPEN_REVISED = "prompt-ablation-v2-open-2026-10"
OPEN_REVISED_STYLES = ("encouraged", "forced")
OPEN_REVISED_ENGINES = {
    "Olmo-3-32B-Think": {"tp_size": 1, "attention_backend": "FLASH_ATTN", "max_model_len": None, "max_num_seqs": None,
                         "max_new_tokens": 32768, "enable_thinking": True, "reasoning_parser": "olmo3",
                         "strip_thinking_model": "olmo3", "gpt_oss_reasoning_level": None},
    "gpt-oss-120b": {"tp_size": 1, "attention_backend": "FLASH_ATTN", "max_model_len": None, "max_num_seqs": 512,
                     "max_new_tokens": 65536, "enable_thinking": True, "reasoning_parser": "openai_gptoss",
                     "strip_thinking_model": "openai_gptoss", "gpt_oss_reasoning_level": "high"},
}
OPEN_REVISED_SHARDS = {"Olmo-3-32B-Think": 10, "gpt-oss-120b": 5}


def test_the_open_revised_recipes_launch_prompt_table_s_open_arms_but_for_the_prompt(tmp_path, monkeypatch):
    from ruleforge.evaluate.campaign import validate_campaign_cfg

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    published_values, published = _composed_members("prompt_ablation_naive", {**EXAMPLES["prompt_ablation_naive"],
                                                                      "styles": RECORDED_STYLES})
    shards = {}
    for settings, arms in _header_recipes("prompt_ablation", OPEN_REVISED):
        [model] = {model for model, _ in arms}
        assert sorted(arms) == sorted((model, style) for style in OPEN_REVISED_STYLES)
        for (_, style), cfg in arms.items():
            arm, reference = _flat(cfg), _flat(published[(model, REVISED_COUNTERPARTS[style])])
            assert sorted(key for key in set(arm) | set(reference) if arm.get(key) != reference.get(key)) == [
                "condition_id", "eval_cfg.reasoning_prompt_style"]
            assert arm["condition_id"] == f"{model}/{style}"
            assert [arm[key] for key in ("sampling.num_decoder_draws", "sampling.max_semantic_attempts",
                                         "sampling.scenarios_per_request_block", "sampling.campaign_mode",
                                         "sampling.forced_json_requirement", "interpretation",
                                         "eval_cfg.individual_rule_context", "evaluation_mode")] == [
                5, 20, 50, "final", "complete_rule_map", "No_Interpretation", "isolated", "both"]
            assert {key: arm.get(f"judge_model.{key}") for key in OPEN_REVISED_ENGINES[model]} == OPEN_REVISED_ENGINES[model]
            validate_campaign_cfg(cfg)
        differing = sorted(key for key in (set(settings) | set(published_values)) - {"name"}
                           if settings.get(key) != published_values.get(key))
        assert differing == ["axis", "campaign_id", "shards", "timeout_min"]
        assert (settings["campaign_id"], settings["timeout_min"], settings["qos"], settings["resumable"],
                settings["gpu_count"], settings["backend"]) == (OPEN_REVISED, 360, "normal", True, 1, "slurm")
        assert (settings["population"], settings["expect_population_sha256"]) == (
            published_values["population"], published_values["expect_population_sha256"])
        shards[model] = settings["shards"]
    assert shards == OPEN_REVISED_SHARDS


ADDED_SINCE_PUBLISHED = {
    "execution.allow_stack_change": False, "execution.pack_draws": False, "execution.pack_max_prompts": 4096,
    "execution.pack_max_draws": None, "execution.num_shards": None, "execution.shard_index": None,
    "job_root": "${campaign_root}", "sampling.strict_think": False,
    "approved_rule_set_sha256": None, "approved_adapter_sha256": None, "approved_model_weights_sha256": None,
}
#: What the campaign verb sets for each launch, its root and population (placeholders in the composed configs).
LAUNCH_KEYS = ("campaign_id", "campaign_root", "scenario_set_path", "approved_scenario_sha256")
_ABSENT = "<absent>"
RETIRED_SINCE_PUBLISHED = {
    "sampling.label_rule": (_ABSENT, "lenient"), "eval_cfg.preemption_judge_type": (_ABSENT, None),
    "rule_amendment.smrc_method": (_ABSENT, "hierarchical_all_rules"), "rule_amendment.smrc_weight": (_ABSENT, 0.0),
}
RAN_TWO_LEVELS_DOWN = "../../"


def _rebased(value):
    if isinstance(value, str) and value.startswith(RAN_TWO_LEVELS_DOWN):
        return value[len(RAN_TWO_LEVELS_DOWN):]
    return value


def _named_now(value):
    if isinstance(value, list):
        return [_named_now(item) for item in value]
    if isinstance(value, str):
        return value.replace("${oc.env:SMRC_", "${oc.env:RULEFORGE_").replace("${smrc_judge_type}", "${interpretation}")
    return value


def _check_recorded_run(cfg, recorded: Path, *, style: str, counterpart: str) -> None:
    composed = _flat(cfg, resolve=False)
    ran = {key: _named_now(_rebased(value)) for key, value in _walk(yaml.safe_load(recorded.read_text(encoding="utf-8"))).items()}
    ran["interpretation"] = ran.pop("smrc_judge_type")  # the key the published runs recorded
    ran["eval_cfg.reasoning_prompt_style"] = record_names.public_style(ran["eval_cfg.reasoning_prompt_style"])
    differences = {key: (composed.get(key, _ABSENT), ran.get(key, _ABSENT)) for key in set(composed) | set(ran)
                   if composed.get(key, _ABSENT) != ran.get(key, _ABSENT)}
    assert {key: new for key, (new, old) in differences.items() if old == _ABSENT} == ADDED_SINCE_PUBLISHED
    assert {key: pair for key, pair in differences.items() if pair[1] != _ABSENT and key not in LAUNCH_KEYS} == {
        "eval_cfg.reasoning_prompt_style": (style, counterpart), **RETIRED_SINCE_PUBLISHED}
    assert ran["scenario_set_path"].endswith("/populations/frozen_500.jsonl")
    assert ran["approved_scenario_sha256"] == EXAMPLES["batch_judge"]["population_sha256"]


def _swapped_back(spec, counterpart: str) -> list[str]:
    assert spec.label_rule == LABEL_RULE
    return [dataclasses.replace(spec, **change).fingerprint for change in (
        {}, {"prompt_style": counterpart}, {"label_rule": RECORDED_LABEL_RULE},
        {"prompt_style": counterpart, "label_rule": RECORDED_LABEL_RULE})]


@pytest.mark.cluster
def test_each_open_revised_arm_is_its_published_arm_but_for_the_style_and_the_label_rule_by_fingerprint_and_engine():
    import json
    import os

    root = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / "sampling/final-2026-09"
    if not root.is_dir() or not os.environ.get("RULEFORGE_MODEL_ROOT"):
        pytest.skip("needs the published root and RULEFORGE_MODEL_ROOT")
    arms = {key: cfg for _, composed in _header_recipes("prompt_ablation", OPEN_REVISED) for key, cfg in composed.items()}
    assert sorted(arms) == sorted((model, style) for model in OPEN_REVISED_ENGINES for style in OPEN_REVISED_STYLES)
    registered = load_conditions(root, skip_retired=True)  # the published arms, read as recorded
    for (model, style), cfg in arms.items():
        counterpart = REVISED_COUNTERPARTS[style]
        recorded = record_names.locate_condition(registered, f"{model}/{counterpart}")
        assert [fingerprint == recorded.fingerprint
                for fingerprint in _swapped_back(_condition_spec(cfg), counterpart)] == [False, False, False, True]
        runs = sorted(root.glob(f"hydra/*/{recorded.condition_id}/*/.hydra/config.yaml"))
        assert runs, (model, counterpart)
        for run in runs:
            _check_recorded_run(cfg, run, style=style, counterpart=counterpart)
        for draw in range(5):
            provenance = json.loads(
                (root / recorded.condition_id / f"draw-{draw:03d}/draw_provenance.json").read_text())
            assert (provenance["tp_size"], provenance["attention_backend"]) == (1, "FLASH_ATTN")


#: GPT-5.6 Luna's paper-prompt arms: batch_judge's header recipes, one launch per style into one root.
LUNA_REVISED = "prompt-ablation-v2-2026-10-luna"


def test_the_luna_revised_recipes_launch_luna_s_prompt_table_arms_but_for_the_prompt(tmp_path, monkeypatch):
    from ruleforge.launch.guards import compose
    from ruleforge.evaluate.campaign import expected_requests_per_draw, validate_campaign_cfg
    from ruleforge.scenarios.manifest import ScenarioManifest
    from ruleforge.core.artifacts import load_jsonl_records

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "/batch")
    published_values, _ = _composed_members("batch_judge", EXAMPLES["batch_judge"])
    population = tmp_path / "population.jsonl"
    population.write_text("".join(f'{{"prompt": "p{index}", "response": "r{index}"}}\n' for index in range(500)))
    rules = load_jsonl_records(REPO / "data/rules/modelspec/openai_rules_all.jsonl")
    styles, first_passes = [], 0
    for settings, arms in _header_recipes("batch_judge", LUNA_REVISED):
        [((model, style), arm)] = arms.items()
        counterpart = REVISED_COUNTERPARTS[style]
        recorded = compose(REPO, [*LUNA_FINAL_OVERRIDES, f"eval_cfg.reasoning_prompt_style={counterpart}",
                                  "campaign_root=/r", "scenario_set_path=/p.jsonl", "approved_scenario_sha256=ab"],
                           default_config="sampling_campaign")
        flat_arm, flat_recorded = _flat(arm), _flat(recorded)
        assert sorted(key for key in set(flat_arm) | set(flat_recorded)
                      if flat_arm.get(key) != flat_recorded.get(key)) == [
            "condition_id", "eval_cfg.reasoning_prompt_style"]
        assert [flat_arm[key] for key in ("condition_id", "judge_model.model_type",
                                          "sampling.campaign_mode", "sampling.num_decoder_draws",
                                          "sampling.max_semantic_attempts", "sampling.scenarios_per_request_block",
                                          "sampling.forced_json_requirement", "interpretation")] == [
            f"gpt-5.6-luna/{style}", "openai_batch", "final", 5, 20, 500, "complete_rule_map",
            "No_Interpretation"]
        validate_campaign_cfg(arm)
        spec = _batch_spec(arm)
        assert [fingerprint == LUNA_FINAL_FINGERPRINTS[counterpart] for fingerprint in _swapped_back(spec, counterpart)
                ] == [False, False, False, True]
        draw = expected_requests_per_draw(ScenarioManifest.build(population), rules, spec.evaluation_mode)
        assert draw == 29_000
        first_passes += draw * flat_arm["sampling.num_decoder_draws"]
        assert sorted(key for key in (set(settings) | set(published_values)) - {"name"}
                      if settings.get(key) != published_values.get(key)) == ["axis", "campaign_id"]
        assert (settings["campaign_id"], settings["axis"], settings["backend"]) == (
            LUNA_REVISED, [f"eval_cfg.reasoning_prompt_style={style}"], "local")
        styles.append(style)
    assert sorted(styles) == sorted(OPEN_REVISED_STYLES) and first_passes == 290_000


@pytest.mark.cluster
def test_each_luna_revised_arm_is_its_published_run_but_for_the_style_and_the_label_rule(monkeypatch):
    import os

    root = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / "sampling/final-2026-09-luna"
    if not root.is_dir():
        pytest.skip("needs the published root")
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", os.environ.get("RULEFORGE_BATCH_ROOT") or "/batch")
    recipes = _header_recipes("batch_judge", LUNA_REVISED)
    assert sorted(style for _, arms in recipes for _, style in arms) == sorted(OPEN_REVISED_STYLES)
    registered = load_conditions(root, skip_retired=True)  # the published arms, read as recorded
    for _, arms in recipes:
        [((model, style), cfg)] = arms.items()
        counterpart = REVISED_COUNTERPARTS[style]
        recorded = record_names.locate_condition(registered, f"{model}/{counterpart}")
        assert recorded.fingerprint == LUNA_FINAL_FINGERPRINTS[counterpart]
        assert _swapped_back(_batch_spec(cfg), counterpart)[-1] == recorded.fingerprint
        run = root / "hydra" / "luna" / f"{Path(recorded.condition_id).name}-login" / ".hydra" / "config.yaml"
        _check_recorded_run(cfg, run, style=style, counterpart=counterpart)


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_a_spec_s_header_recipes_keep_their_line_continuations(path):
    header = [line[1:] for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert [line for line in header if COLLAPSED.search(line)] == []


@pytest.mark.parametrize("path", sorted(RECIPES.glob("*.txt")), ids=lambda path: path.stem)
def test_a_fixture_recipe_keeps_its_line_continuations(path):
    assert [line for line in path.read_text(encoding="utf-8").splitlines() if COLLAPSED.search(line)] == []


@pytest.mark.parametrize("path", SPECS, ids=lambda path: path.stem)
def test_a_spec_names_no_absolute_path_and_no_identity(path):
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=cli.SpecLoader) or {}
    assert list(absolute_strings(data)) == []
    texts = list(_strings(data))
    assert [text for text in texts if HOME_OR_USER.search(text) or EMAIL.search(text)] == []
    comments = [line for line in path.read_text(encoding="utf-8").splitlines() if "#" in line]
    assert [line for line in comments if EMAIL.search(line)] == []


JOB_EXAMPLES = {
    "grpo_rows_gemma": ({"anchors": "anchors/a", "scenarios": "populations/s.jsonl",
                         "require_protected": "populations/grpo_v2_dev.jsonl",
                         "dev_nonyes": "populations/gemma_dev_nonyes.jsonl", "out": "grpo_data/b"},
                        ("ruleforge.launch.programs.build_grpo_data", "ruleforge.grpo.grpo_data")),
    "merge_fp32": ({"run": "r", "step": "200"}, None),
}
ROOTS = {"artifact_root": "/artifacts", "model_root": "/models"}


def _job_spec(name: str) -> tuple[dict, list[str]]:
    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", JOB_EXAMPLES[name][0], "L1",
                                            _options("job"), repo=REPO)
    return values, [fill(item, ROOTS) for item in arguments]


@pytest.mark.parametrize("name", sorted(JOB_EXAMPLES))
def test_a_job_spec_s_command_parses_as_its_program_parses_it(name):
    import importlib
    import importlib.util

    from ruleforge.cli.app import build_parser

    values, argv = _job_spec(name)
    module = JOB_EXAMPLES[name][1]
    assert argv[0] == "python" and (argv[1:3] == ["-m", "ruleforge"]) == (module is None)
    if module is None:
        build_parser().parse_args(argv[3:])
    else:
        program, parser = module
        assert argv[1:3] == ["-m", program] and importlib.util.find_spec(program) is not None
        assert values.get("workdir", ".") == "."
        importlib.import_module(parser)._build_parser().parse_args(argv[3:])


def test_the_gemma_rows_spec_s_defaults_are_the_published_build_s():
    """grpo_rows_gemma: by default 2,400 draws under the verdict mix."""
    from ruleforge.grpo import grpo_data

    _, argv = _job_spec("grpo_rows_gemma")
    args = grpo_data._build_parser().parse_args(argv[3:])
    assert (args.prompt_draws, args.no_share, args.no_margin, args.repeat_cap) == (2400, 0.35, 0.3, 5)
    assert [str(path) for path in args.anchors] == ["/artifacts/anchors/a"]
    values = {**JOB_EXAMPLES["grpo_rows_gemma"][0], "prompt_draws": "800"}
    _, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / "grpo_rows_gemma.yaml", values, "L1", _options("job"),
                                       repo=REPO)
    assert grpo_data._build_parser().parse_args([fill(item, ROOTS) for item in arguments][3:]).prompt_draws == 800
    args = grpo_data._build_parser().parse_args([*argv[3:], "--anchors", "/artifacts/anchors/b"])
    assert [str(path) for path in args.anchors] == ["/artifacts/anchors/a", "/artifacts/anchors/b"]


@pytest.mark.parametrize("name", ["batch_judge", "prompt_ablation", "prompt_ablation_naive"])
def test_no_spec_names_a_label_rule(name, tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    assert "label_rule" not in _compose_spec(name, EXAMPLES[name], tmp_path)[1].sampling
    assert {path.stem for path in SPECS if "label_rule" in path.read_text()} == set()


#: Each spec that names a retry scope, and its default: the bundle scope of every root its examples name.
RETRY_SCOPE_DEFAULTS = {"batch_judge": "scenario_bundle", "hierarchy": "scenario_bundle",
                        "prompt_ablation": "scenario_bundle"}


@pytest.mark.parametrize("name", sorted(RETRY_SCOPE_DEFAULTS))
def test_the_retry_scope_is_a_spec_variable_whose_default_is_its_roots(name, tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    assert _compose_spec(name, EXAMPLES[name], tmp_path)[1].sampling.retry_scope == RETRY_SCOPE_DEFAULTS[name]
    assert _compose_spec(name, {**EXAMPLES[name], "retry_scope": "component"}, tmp_path)[1].sampling.retry_scope == (
        "component")
    specs = {path.stem for path in SPECS if "{retry_scope}" in path.read_text()}
    assert specs == set(RETRY_SCOPE_DEFAULTS)


GLIMMER = "prompt-ablation-v2-2026-10-glimmer"
GLIMMER_ROOT = "sampling/prompt-ablation-v2-2026-10-glimmer-pop-6f528dcb"
#: The prompt ablation's three arms of a new judge: naive the published naive prompt, encouraged and forced the paper's texts.
GLIMMER_STYLES = ("naive-rules-above", "encouraged", "forced")
GLIMMER_OWN = {"condition_id", "sampling.retry_scope"}
#: The fingerprints the three condition.json files of Glimmer's prompt-ablation root record.
GLIMMER_FINGERPRINTS = {
    "naive-rules-above": "409dcd61f25450aca7aaf07739e873251fa7a6a2d1df2167477434ec8d164559",
    "encouraged": "e4823dd5a2e13f63cdc84e2e40efc112cd1a3fb60e709083f35c6f652d1cc670",
    "forced": "9537285848138848c172c724d974cb69ae506af2175eb70f95f263fa0c74dc99",
}
OLMO_ARM_ROOTS = {"naive-rules-above": "final-2026-09-pop-a22af097-rescored-lenient_last_line",
                  "encouraged": "prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line",
                  "forced": "prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line"}


def _glimmer_arms() -> tuple[dict, dict]:
    """The settings of the header's Glimmer launch, and ``{style: the composed config}`` of its conditions."""
    [(settings, arms)] = _header_recipes("prompt_ablation", GLIMMER)
    assert sorted(arms) == sorted(("Muse-Glimmer-30B", style) for style in GLIMMER_STYLES)
    return settings, {style: cfg for (_, style), cfg in arms.items()}


def test_glimmer_s_recipe_launches_prompt_table_s_open_arms_but_for_the_judge_and_the_scope(tmp_path, monkeypatch):
    from ruleforge.evaluate.campaign import validate_campaign_cfg

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    references = {style: _flat(cfg) for _, arms in _header_recipes("prompt_ablation", OPEN_REVISED)
                  for (model, style), cfg in arms.items() if model == "Olmo-3-32B-Think"}
    references["naive-rules-above"] = _flat(_composed_members("prompt_ablation_naive", EXAMPLES["prompt_ablation_naive"])[1][
        ("Olmo-3-32B-Think", "naive-rules-above")])
    settings, arms = _glimmer_arms()
    for style, cfg in arms.items():
        arm, reference = _flat(cfg), references[style]
        differing = {key for key in set(arm) | set(reference) if arm.get(key) != reference.get(key)}
        assert GLIMMER_OWN < differing and all(key.startswith(("judge_model.", "agent_model."))
                                               for key in differing - GLIMMER_OWN), style
        assert (arm["condition_id"], arm["sampling.retry_scope"]) == (f"Muse-Glimmer-30B/{style}", "component")
        assert [arm[key] for key in ("sampling.num_decoder_draws", "sampling.max_semantic_attempts",
                                     "sampling.scenarios_per_request_block", "sampling.campaign_mode",
                                     "sampling.forced_json_requirement", "interpretation",
                                     "eval_cfg.individual_rule_context", "evaluation_mode", "sampling.temperature",
                                     "sampling.truncation", "sampling.strict_think")] == [
            5, 20, 50, "final", "complete_rule_map", "No_Interpretation", "isolated", "both", 1.0, "model_default",
            False]
        assert {key: arm[f"judge_model.{key}"] for key in (
            "model_path", "tp_size", "strip_thinking_model", "reasoning_parser", "max_new_tokens", "enable_thinking",
            "attention_backend", "language_model_only", "gpt_oss_reasoning_level", "max_model_len")} == {
            "model_path": "/models/Muse-Glimmer-30B", "tp_size": 1, "strip_thinking_model": "muse_glimmer",
            "reasoning_parser": "muse_glimmer", "max_new_tokens": 65536, "enable_thinking": True,
            "attention_backend": "FLASH_ATTN", "language_model_only": True, "gpt_oss_reasoning_level": None,
            "max_model_len": None}
        validate_campaign_cfg(cfg)
    assert (settings["campaign_id"], settings["campaign_root"], settings["new_root"], settings["backend"],
            settings["gpu_count"]) == (GLIMMER, GLIMMER_ROOT, True, "slurm", 1)
    assert (settings["population"], settings["expect_population_sha256"]) == (
        "populations/frozen_500_r3.jsonl", "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4")


@pytest.mark.cluster
def test_glimmer_s_conditions_compose_to_their_recorded_fingerprints_and_differ_from_olmo_s_arms_as_declared():
    root = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / "sampling"
    models = os.environ.get("RULEFORGE_MODEL_ROOT")
    if not all((root / name).is_dir() for name in OLMO_ARM_ROOTS.values()) or not models or not (
            Path(models) / "Muse-Glimmer-30B").is_dir():
        pytest.skip("needs the shipped re-scored roots, and Muse-Glimmer-30B under RULEFORGE_MODEL_ROOT")
    specs = {style: _condition_spec(cfg) for style, cfg in _glimmer_arms()[1].items()}
    assert {style: spec.fingerprint for style, spec in specs.items()} == GLIMMER_FINGERPRINTS
    registered = {name: load_conditions(root / name) for name in set(OLMO_ARM_ROOTS.values())}
    for style, spec in specs.items():
        olmo = record_names.locate_condition(registered[OLMO_ARM_ROOTS[style]], f"Olmo-3-32B-Think/{style}").to_dict()
        mine = spec.to_dict()
        differing = {key for key in set(olmo) | set(mine)
                     if olmo.get(key) != mine.get(key)}  # a key at its default is omitted (retry_scope's)
        assert differing == {"condition_id", "model_name", "model_path", "max_new_tokens", "sampling_truncation",
                             "retry_scope"}, style
        assert (mine["model_path"], mine["max_new_tokens"], mine["label_rule"], mine["retry_scope"]) == (
            "Muse-Glimmer-30B", 65536, "lenient_last_line", "component")


HIERARCHY = "hierarchy-aware-2026-10"
HIERARCHY_GLIMMER, HIERARCHY_QWEN = "hierarchy-aware-2026-10-glimmer", "hierarchy-aware-2026-10-qwen3-8b"
#: batch_judge's header recipes of the API judges' hierarchy arms.
HIERARCHY_HAIKU, HIERARCHY_LUNA = "hierarchy-aware-2026-10-haiku-sum", "hierarchy-aware-2026-10-luna-sum"
FLAT, HIER = "naive-rules-above", "naive-hierarchy"
HIERARCHY_SHARDS = {"gpt-oss-120b": 3, "Olmo-3-32B-Think": 7, "gemma-4-31B-it": 8}
LEVELS_PATH, LEVELS_SHA256 = (
    "data/rules/modelspec/openai_rules_all_full.jsonl",
    "99eac16d5fc4c8cd028b091b6cab23badcd1573bd33d9765f0ca436e91f7130a")
HIERARCHY_MAP = {f"hierarchy.styles.{FLAT}.rule_context": "isolated", f"hierarchy.styles.{FLAT}.levels": None,
                 f"hierarchy.styles.{FLAT}.levels_sha256": None,
                 f"hierarchy.styles.{HIER}.rule_context": "superiors", f"hierarchy.styles.{HIER}.levels": LEVELS_PATH,
                 f"hierarchy.styles.{HIER}.levels_sha256": LEVELS_SHA256}
HIERARCHY_ARM = {"eval_cfg.individual_rule_context": "superiors", "rule_levels_path": LEVELS_PATH,
                 "approved_rule_levels_sha256": LEVELS_SHA256}
HIERARCHY_FINGERPRINTS = {
    "Olmo-3-32B-Think/naive-rules-above": "5a94234f9e8b0c2923a4e15e2588b17ca10b7fe1610d0d6378f308ea558de207",
    "Olmo-3-32B-Think/naive-hierarchy": "7f6d2420d895460cb558f7493769253368c9b2516d814d90bf870eb1462d2bc9",
    "gemma-4-31B-it/naive-rules-above": "17646ddc22e27d2c9698c6db0f35c328d7f25c55f06df6d0b156e67a83a9dbd5",
    "gemma-4-31B-it/naive-hierarchy": "133406f222433317e4a4ce012b7abd50c2ac75e22cb99bf4f6275bff3e0c8b76",
    "gpt-oss-120b/naive-rules-above": "5cdb1d75ef852ef840d4157cf8ad9475856e2cf9c92d96927f6b795a8e6e9a7e",
    "gpt-oss-120b/naive-hierarchy": "5c43a079569d05f9b2c2c67a10ad130ba47ebb2c85494db94eda87daff1cd2e0",
    "Muse-Glimmer-30B/naive-hierarchy": "ac7dc6980f5e721a3a7d8654f8c1d3ff9a33c9918b15f13aa417eca3b3e94c2a",
    "Qwen3-8B/naive-rules-above": "7dd42b6ccb076767176f165771bb23c9e0401d4306388faaadbf166f86960427",
    "Qwen3-8B/naive-hierarchy": "519da5b17fc7accf43fec93e8a7f3c263afa3809d7b842b0df3bf8ea773d835b",
    "claude-haiku-5-5/naive-hierarchy": "cf3206a9b45c049e46bc2601d0ce73f96c18948490e3a637685d8b6dcc2bfaab",
    "gpt-5.6-luna/naive-hierarchy": "8a3576aa23aa0bd7f72aa347b3a34feb87811e7569236d0fd439c721cdfb916c",
}
API_FLAT_FINGERPRINTS = {
    "claude-haiku-5-5": "806fc5b850b61bed1a5d4d5dafa07400a970b82118ade8450e1bbba397b4392a",
    "gpt-5.6-luna": "0fcaa2ed99d95253a0bcd2cb2ba92af7ab344e780db2317c33cb89d7509fb0ae",
}
#: The roots the hierarchy arms ran in, under the artifact root.
HIERARCHY_ROOTS = ("hierarchy-aware-2026-10-pop-a22af097", HIERARCHY_GLIMMER, HIERARCHY_QWEN, HIERARCHY_HAIKU,
                   HIERARCHY_LUNA)


def _differences(arm: dict, reference: dict) -> dict:
    return {key: arm.get(key) for key in set(arm) | set(reference) if arm.get(key) != reference.get(key)}


def _expected_hierarchy(model: str, style: str, reference: dict) -> dict:
    expected = {"condition_id": f"{model}/{style}", "eval_cfg.reasoning_prompt_style": style, **HIERARCHY_MAP,
                **(HIERARCHY_ARM if style == HIER else {})}
    return {key: value for key, value in expected.items() if value != reference.get(key)}


def _revised_counterparts() -> dict:
    open_revised = {key: cfg for _, arms in _header_recipes("prompt_ablation", OPEN_REVISED) for key, cfg in arms.items()}
    _, gemma = _composed_members("prompt_ablation", EXAMPLES["prompt_ablation"])
    return {**open_revised, **gemma}


def test_the_hierarchy_recipes_launch_each_open_judge_s_revised_arm_but_for_the_style_and_the_context(tmp_path, monkeypatch):
    from ruleforge.evaluate.campaign import validate_campaign_cfg

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    counterparts = _revised_counterparts()
    revised_values, _ = _composed_members("prompt_ablation", EXAMPLES["prompt_ablation"])
    shards = {}
    for settings, arms in _header_recipes("hierarchy", HIERARCHY):
        [model] = {model for model, _ in arms}
        assert sorted(arms) == [(model, style) for style in (HIER, FLAT)]
        for (_, style), cfg in arms.items():
            arm, reference = _flat(cfg), _flat(counterparts[(model, "encouraged")])
            assert _differences(arm, reference) == _expected_hierarchy(model, style, reference)
            assert [arm[key] for key in ("sampling.num_decoder_draws", "sampling.max_semantic_attempts",
                                         "sampling.scenarios_per_request_block", "sampling.campaign_mode",
                                         "sampling.retry_scope", "interpretation", "evaluation_mode",
                                         "judge_model.tp_size")] == [
                5, 20, 50, "final", "scenario_bundle", "No_Interpretation", "both", 1]
            validate_campaign_cfg(cfg)
        assert sorted(key for key in (set(settings) | set(revised_values)) - {"name"}
                      if settings.get(key) != revised_values.get(key)) == ["axis", "campaign_id", "shards"]
        assert (settings["campaign_id"], settings["timeout_min"], settings["qos"], settings["resumable"],
                settings["gpu_count"], settings["backend"]) == (HIERARCHY, 360, "normal", True, 1, "slurm")
        assert (settings["population"], settings["expect_population_sha256"]) == (
            "populations/frozen_500_r3.jsonl", "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4")
        shards[model] = settings["shards"]
    assert shards == HIERARCHY_SHARDS


def test_glimmer_s_and_qwen_s_hierarchy_recipes_are_their_flat_runs_but_for_the_style_and_the_context(
        tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    [(settings, arms)] = _header_recipes("hierarchy", HIERARCHY_GLIMMER)
    assert sorted(arms) == [("Muse-Glimmer-30B", HIER)] and settings["campaign_id"] == HIERARCHY_GLIMMER
    naive = _flat(_glimmer_arms()[1]["naive-rules-above"])
    arm = _flat(arms[("Muse-Glimmer-30B", HIER)])
    assert _differences(arm, naive) == _expected_hierarchy("Muse-Glimmer-30B", HIER, naive)
    [(settings, arms)] = _header_recipes("hierarchy", HIERARCHY_QWEN)
    assert sorted(arms) == [("Qwen3-8B", HIER), ("Qwen3-8B", FLAT)] and settings["campaign_id"] == HIERARCHY_QWEN
    flat, hierarchy = (_flat(arms[("Qwen3-8B", style)]) for style in (FLAT, HIER))
    assert _differences(hierarchy, flat) == {"condition_id": f"Qwen3-8B/{HIER}",
                                             "eval_cfg.reasoning_prompt_style": HIER, **HIERARCHY_ARM}
    assert [flat[key] for key in ("sampling.retry_scope", "judge_model.model_path", "judge_model.tp_size",
                                  "sampling.max_semantic_attempts", "sampling.campaign_mode")] == [
        "component", "/models/Qwen3-8B", 1, 20, "final"]


def test_the_api_judges_hierarchy_recipes_compose_to_their_recorded_fingerprints(tmp_path, monkeypatch):
    from ruleforge.core.errors import SamplingDesignError
    from ruleforge.core.artifacts import load_jsonl_records
    from ruleforge.evaluate.campaign import expected_requests_per_draw, validate_campaign_cfg
    from ruleforge.judges.backend import ModelBackend
    from ruleforge.scenarios.manifest import ScenarioManifest

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    population = tmp_path / "population.jsonl"
    population.write_text("".join(f'{{"prompt": "p{index}", "response": "r{index}"}}\n' for index in range(500)))
    rules = load_jsonl_records(REPO / "data/rules/modelspec/openai_rules_all.jsonl")
    swaps = {"prompt_style": FLAT, "individual_rule_context": "isolated", "rule_levels_sha256": None}
    for campaign_id, model, scope in ((HIERARCHY_HAIKU, "claude-haiku-5-5", "component"),
                                      (HIERARCHY_LUNA, "gpt-5.6-luna", "scenario_bundle")):
        [(settings, arms)] = _header_recipes("batch_judge", campaign_id)
        assert sorted(arms) == [(model, HIER)]
        cfg = arms[(model, HIER)]
        validate_campaign_cfg(cfg)
        backend = ModelBackend(cfg.judge_model)
        spec = _condition_spec(cfg, model_type=backend.model_type, seed_capability=backend.seed_capability)
        assert spec.fingerprint == HIERARCHY_FINGERPRINTS[f"{model}/{HIER}"]
        assert (spec.label_rule, spec.retry_scope, spec.rule_levels_sha256) == (LABEL_RULE, scope, LEVELS_SHA256)
        flat = dataclasses.replace(spec, condition_id=f"{model}/{FLAT}", **swaps)
        assert flat.fingerprint == API_FLAT_FINGERPRINTS[model]
        with pytest.raises(SamplingDesignError, match=f"'{HIER}' judges each rule with the rules that outrank"):
            dataclasses.replace(spec, **{key: value for key, value in swaps.items() if key != "prompt_style"})
        assert expected_requests_per_draw(ScenarioManifest.build(population), rules, spec.evaluation_mode) * 5 == (
            145_000)
        assert [settings[key] for key in ("population", "expect_population_sha256", "axis", "backend")] == [
            "populations/frozen_500_r3.jsonl", "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4",
            [f"eval_cfg.reasoning_prompt_style={HIER}"], "local"]
        flat_cfg = _flat(cfg)
        assert [flat_cfg[key] for key in ("sampling.campaign_mode", "sampling.num_decoder_draws",
                                          "sampling.scenarios_per_request_block", "sampling.max_semantic_attempts",
                                          "sampling.retry_scope", "rule_levels_path")] == [
            "final", 5, 500, 20, scope, LEVELS_PATH]


@pytest.mark.cluster
def test_the_hierarchy_recipes_compose_to_the_conditions_the_roots_registered():
    root = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / "sampling"
    models = os.environ.get("RULEFORGE_MODEL_ROOT")
    if not all((root / name).is_dir() for name in HIERARCHY_ROOTS) or not models:
        pytest.skip("needs the hierarchy roots and RULEFORGE_MODEL_ROOT")
    composed = {}
    for campaign_id in (HIERARCHY, HIERARCHY_GLIMMER, HIERARCHY_QWEN):
        for _, arms in _header_recipes("hierarchy", campaign_id):
            for (model, style), cfg in arms.items():
                if not (Path(models) / Path(str(cfg.judge_model.model_path)).name).is_dir():
                    pytest.skip(f"needs {model}'s weights under RULEFORGE_MODEL_ROOT")
                composed[f"{model}/{style}"] = _condition_spec(cfg)
    registered = {}
    for name in HIERARCHY_ROOTS[:3]:
        registered.update({spec.public_id: spec for spec in load_conditions(root / name).values()})
    assert sorted(composed) == sorted(public for public in HIERARCHY_FINGERPRINTS
                                      if not public.startswith(("claude", "gpt-5.6")))
    for public, spec in composed.items():
        recorded = registered[public]
        assert recorded.fingerprint == HIERARCHY_FINGERPRINTS[public]
        if recorded.label_rule != LABEL_RULE:
            assert (recorded.label_rule, public.split("/")[0]) in {
                ("lenient_markup", model) for model in HIERARCHY_SHARDS}
            spec = _as_recorded(spec, recorded.label_rule)
        assert spec.to_dict() == {**recorded.to_dict(), "condition_id": spec.condition_id}
        assert spec.fingerprint == recorded.fingerprint


RECORDED_RENDERS = Path(__file__).resolve().parent / "fixtures" / "launch" / "recorded_spec_renders.json"
PRESETS = {"grpo_train_gemma_run": "grpo_train_run", "gemma_anchors": "grpo_anchors", "training_arm_gemma": "training_arms"}
RECORDED_DIFFERENCES = {"20261001T183746Z-35a23e81-a96d": {"env": ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                                                                   "MAX_JOBS=8"]}}
#: Where the published and rerun roots hold their launch records; new launches write RECORDS_DIRNAME.
RECORDED_LAUNCHES_DIRNAME = ".smrc-launch"
#: What the recorded launches called a spec, and the name now.
RECORDED_SPECS = {"figure4_rulesets": "ruleforge_rulesets", "figure5_arm_gemma": "training_arm_gemma", "figure5_arms": "training_arms",
                  "prompt_ablation_v2": "prompt_ablation", "table1_final": "prompt_ablation_naive"}
#: What their arguments called a config block, and its name now.
RECORDED_KEYS = {f"{recorded}.": f"{public}." for public, recorded in RECORDED_BLOCK_KEYS.items()}
#: What their arguments called a config and the serving group, and the names now.
RECORDED_WORDS = {**RECORDED_KEYS, "sampling_figure5": "sampling_training", "sampling_grpo_v2_anchors": "sampling_training_anchors",
                  "figure5_serving=": "training_serving=", "finetuning/grpo/": "training/grpo/",
                  "ruleforge.launch.programs.train": "ruleforge.launch.programs.optimize",
                  "table1_final": "prompt_ablation_naive"}


def _spec_path_now(path: str) -> str:
    parent, _, name = path.rpartition("/")
    stem = name.removesuffix(".yaml")
    return f"{parent}/{RECORDED_SPECS.get(stem, stem)}.yaml"


def _launch_words_now(arguments: list, site: Optional[str]) -> list:
    """A recorded job's arguments under today's names, its model configs read against ``site``, its record's partition."""
    words = _named_now(arguments)
    for old, new in RECORDED_WORDS.items():
        words = [word.replace(old, new) for word in words]
    return model_words_now(words, site)


def _without_retired(variables: dict) -> dict:
    return {name: value for name, value in variables.items() if name not in ("prereg", "prereg_sha256")}


def _arguments_without_retired(arguments: list) -> list:
    return [word for word in arguments if not word.startswith(("prereg.path=", "prereg.sha256="))]


def _keys_now(words: list) -> list:
    """Recorded arguments with each config key under this version's name."""
    for old, new in RECORDED_KEYS.items():
        words = [word.replace(old, new) for word in words]
    return words


def _recorded_renders() -> dict:
    renders = json.loads(RECORDED_RENDERS.read_text(encoding="utf-8"))
    for entry in renders["specs"].values():
        for render in entry["renders"]:
            render["vars"] = _without_retired(render["vars"])
            render["arguments"] = _keys_now(_arguments_without_retired(render["arguments"]))
            if "axis" in render["values"]:
                render["values"]["axis"] = _keys_now(render["values"]["axis"])
    return renders


def _rendered(name: str, verb: str, variables: dict) -> tuple[dict, list[str], dict]:
    values, arguments, record, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", variables, "L1",
                                                 _options(verb), repo=REPO, verb=verb)
    return json.loads(json.dumps(values)), arguments, record


def test_a_recorded_job_s_model_config_matches_today_s_only_by_its_own_partition():
    today = ["model@judge_model=gemma4-31b-tp-2"]
    assert _launch_words_now(["model@judge_model=gemma4-31b-tp-2-somesite"], "somesite") == today
    for recorded, site in (("gemma4-31b-tp-2-fp32", "somesite"), ("gemma4-31b-tp-2-typo", "somesite"),
                           ("gemma4-31b-tp-2-othersite", "somesite"), ("gemma4-31b-tp-2-somesite", None)):
        assert _launch_words_now([f"model@judge_model={recorded}"], site) != today, (recorded, site)


def test_each_preset_launches_what_its_spec_launched_before_it_became_one():
    import hashlib

    renders = _recorded_renders()
    assert sorted(renders["specs"]) == sorted(PRESETS)
    for name, entry in renders["specs"].items():
        verb = entry["verb"]
        preset = yaml.load((REPO / control.SPECS_DIR / f"{name}.yaml").read_text(encoding="utf-8"), Loader=cli.SpecLoader)
        assert sorted(preset) == [cli.PRESET_KEY, "vars"] and preset[cli.PRESET_KEY] == PRESETS[name]
        assert VERBS[name] == VERBS[PRESETS[name]] == verb
        for render in entry["renders"]:
            assert bool(render["launches"]) != ("synthetic" in render), (name, render["vars"])
            values, arguments, record = _rendered(name, verb, render["vars"])
            assert (values, arguments) == (render["values"], render["arguments"]), (name, render["vars"])
            general = _rendered(PRESETS[name], verb, {**preset["vars"], **render["vars"]})
            assert general[:2] == (render["values"], render["arguments"]), (name, render["vars"])
            spec = REPO / control.SPECS_DIR / f"{PRESETS[name]}.yaml"
            assert record["preset_of"] == {"path": f"{control.SPECS_DIR}/{spec.name}",
                                           "sha256": hashlib.sha256(spec.read_bytes()).hexdigest()}
            assert "preset_of" not in general[2]


@pytest.mark.cluster
def test_each_recorded_gemma_launch_of_a_preset_s_spec_is_what_the_preset_renders_now():
    import os

    renders = _recorded_renders()
    launches = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent")) / RECORDED_LAUNCHES_DIRNAME
    named = {launch for entry in renders["specs"].values() for render in entry["renders"]
             for launch in render["launches"]}
    if not any((launches / launch / "launch.json").is_file() for launch in named):
        pytest.skip("needs the Gemma artifact root's launch records")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(launches.glob("*/launch.json"))]
    checked = []
    for record in records:
        if record["launch_id"] not in named:
            continue
        stem = Path(record["spec"]["path"]).stem
        name = RECORDED_SPECS.get(stem, stem)
        values, arguments, _ = _rendered(name, renders["specs"][name]["verb"], _without_retired(record["spec"]["vars"]))
        expected = {"env": values.get("env", []), **RECORDED_DIFFERENCES.get(record["launch_id"], {})}
        for job in record["jobs"]:
            resources, where = job["resources"], (record["launch_id"], job["job_id"])
            launched = _launch_words_now(_arguments_without_retired(job["argv"]), launch_site(job))
            assert launched[2:2 + len(arguments)] == arguments, where
            assert (resources["gpus"], resources["timeout_min"], resources["qos"]) == (
                values.get("gpu_count", resources["gpus"]), values["timeout_min"], values["qos"]), where
            assert resources["mem_gb"] == values.get("mem_gb", 24 * resources["gpus"]), where
            assert job["env"] == expected["env"], where
        checked.append(record["launch_id"])
    assert sorted(named) == sorted(checked) and set(RECORDED_DIFFERENCES) <= named


QWEN_BANK_FINGERPRINT = "88863a87b84af6261ee1d7b2169737e950acd612437795ccdb8c14ea42b662db"
QWEN_FULL_FINGERPRINT = "1fcebab861c3ecbbad7edf8029077d2cfe8b0d31a19f827aa440ce701ab33d77"
QWEN_BANK_KEY = "634630905846ce45d1ca20bc3bb8417613ed1c25d1e84a31f614846187a90db9"
QWEN_FULL_KEY = "847ac6b0c91f30e430d76802209a66485c8180ea82a7eade5977ff9205815250"


def test_the_anchor_spec_given_qwen3_8b_s_values_is_qwen3_8b_s_recorded_anchor_conditions(tmp_path, monkeypatch):
    from tests.test_sampling_weights import write_weights

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    write_weights(tmp_path / "models" / "Qwen3-8B")
    qwen = {key: EXAMPLES["grpo_anchors"][key] for key in ("judge", "weights", "strict_think", "campaign_prefix")}
    bank, full = (QWEN_BANK_FINGERPRINT, QWEN_BANK_KEY), (QWEN_FULL_FINGERPRINT, QWEN_FULL_KEY)
    stages = {"bank": (bank, 2, 16, "125"), "top_up": (bank, 4, 17, "125"), "confirm": (bank, 4, 17, "125"),
              "full": (full, 2, 16, "250"), "mixscreen": (full, 8, 20, "250")}
    for stage, ((fingerprint, key), draws, seed_base, block) in stages.items():
        values, cfg = _compose_spec("grpo_anchors", {**qwen, "stage": stage, "request_block": block}, tmp_path)
        spec = _condition_spec(cfg)
        assert condition_key(spec) == key, stage
        assert spec.label_rule == LABEL_RULE and spec.fingerprint != fingerprint, stage
        recorded = _as_recorded(spec)
        assert (recorded.fingerprint, condition_key(recorded)) == (fingerprint, key), stage
        assert (spec.condition_id, spec.strict_think, spec.serving_dtype, cfg.judge_model.tp_size) == (
            f"Qwen3-8B/grpo-v2-anchors/{stage}", True, "bfloat16", 1)
        assert (cfg.sampling.num_decoder_draws, cfg.sampling.generation_seed_base,
                cfg.sampling.scenarios_per_request_block) == (draws, seed_base, int(block))
        assert (values["backend"], values["gpu_count"], values["qos"], values["resumable"], values["root_template"]) == (
            "slurm", 1, "normal", True, f"sampling/{qwen['campaign_prefix']}-{stage}/{{block}}")
    _, gemma = _compose_spec("gemma_anchors", {"campaign_prefix": "g", "stage": "bank"}, tmp_path)
    assert (gemma.judge_model.model_name, gemma.sampling.strict_think) == ("gemma-4-31B-it", False)


#: The serving option of the Qwen3-8B run's training arm, and the fp32 engine it composes.
QWEN_SERVING = {"engine": "configs/model/qwen3-8b-fp32.yaml", "dtype": "float32", "campaign_serving": "configs/training_serving/merged_fp32.yaml"}


def test_the_training_spec_given_qwen3_8b_s_values_is_the_design_on_qwen3_8b_s_fp32_engine(
        tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    design, serving = TRAINING_DESIGN, QWEN_SERVING
    engine = yaml.safe_load((REPO / serving["engine"]).read_text(encoding="utf-8"))
    example = EXAMPLES["training_arms"]
    values, arguments = _spec_arguments("training_arms", example)
    assert arguments[:3] == ["--config-name", Path(design["config"]).stem,
                             f"training_serving={Path(serving['campaign_serving']).stem}"]
    arms = _members_composed(values, arguments, knobs=(f"campaign_id={example['campaign_id']}",),
                             key=lambda flat: flat["training"]["policy"])
    assert sorted(arms) == ["base", example["policy"]]
    for policy, cfg in arms.items():
        flat = _flat(cfg)
        assert (flat["sampling.strict_think"], flat["sampling.forced_json_requirement"]) == (
            design["validity"]["strict_think"], design["validity"]["forced_json_requirement"])
        assert (flat["eval_cfg.reasoning_prompt_style"], flat["sampling.num_decoder_draws"],
                flat["approved_scenario_sha256"], flat["evaluation_mode"]) == (
            design["prompt_style"], design["draws"], PAPER_POPULATION["sha256"], "both")
        assert {key: flat[f"judge_model.{key}"] for key in ("model_name", "dtype", "tp_size", "attention_backend")} == {
            key: engine[key] for key in ("model_name", "dtype", "tp_size", "attention_backend")}
        assert flat["condition_id"] == f"Qwen3-8B/figure5/{policy}"
    base, policy = _flat(arms["base"]), _flat(arms[example["policy"]])
    assert sorted(key for key in set(base) | set(policy) if base.get(key) != policy.get(key)) == [
        "approved_model_weights_sha256", "condition_extra.policy", "condition_id", "judge_model.model_path",
        "training.policy"]
    assert (values["backend"], values["gpu_count"], values["mem_gb"], values["shards"], values["axis"]) == (
        "slurm", engine["tp_size"], 96, 25, [f"training.policy={example['policy']},base"])


RERUN_ROOT = os.environ.get("RULEFORGE_RERUN_ROOT")
MOVED_COMMAND_SPECS = frozenset({"configs/launch/merge_fp32.yaml", "configs/launch/ruleforge_smoke.yaml"})
REWRITTEN_RECIPE_SPECS = frozenset({
    "configs/launch/batch_judge.yaml", "configs/launch/ruleforge_rulesets.yaml", "configs/launch/training_arms.yaml",
    "configs/launch/grpo_train_run.yaml", "configs/launch/prompt_ablation.yaml", "configs/launch/prompt_ablation_naive.yaml"})


@pytest.mark.cluster
@pytest.mark.skipif(not RERUN_ROOT, reason="the rerun's root is the cluster's: set RULEFORGE_RERUN_ROOT")
def test_a_spec_the_rerun_launched_keeps_the_bytes_its_records_name():
    named: dict[str, set[str]] = {}
    for path in sorted(Path(RERUN_ROOT).glob(f"{RECORDED_LAUNCHES_DIRNAME}/*/launch.json")):
        spec = json.loads(path.read_text(encoding="utf-8")).get("spec") or {}
        for entry in (spec, spec.get("preset_of") or {}):
            if entry.get("path"):
                named.setdefault(_spec_path_now(entry["path"]), set()).add(entry["sha256"])
    changed = MOVED_COMMAND_SPECS | REWRITTEN_RECIPE_SPECS
    assert named and changed <= set(named)
    now = {path: {hashlib.sha256((REPO / path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()}
           for path in named}
    assert {path: digests for path, digests in named.items() if path not in changed and digests != now[path]} == {}
    assert [path for path in sorted(changed) if named[path] == now[path]] == []
