from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ruleforge.grpo import anchor_set as anchors
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.campaign import condition_spec_for, preflight, validate_campaign_cfg
from ruleforge.evaluate.runner import CampaignRunError
from tests.test_sampling_campaign import backend_for
from tests.test_sampling_training import JOB_DIR
from tests.test_sampling_runner import write_inputs
from tests.test_sampling_weights import write_base, write_merged

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    monkeypatch.chdir(JOB_DIR)  # the configs' relative paths are the checkout's
    write_base(tmp_path / "models")


def _compose(name: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _spec(cfg, modelspec):
    config, _ = preflight(cfg)
    return condition_spec_for(
        condition_id=str(cfg.condition_id), backend=backend_for("forced-grpo"),
        task_spec=modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0), config=config,
        judge_type=str(cfg.interpretation), rule_set_path=Path(str(cfg.rule_set_path)), model_cfg=cfg.judge_model,
        extra=campaign_module.condition_extra(cfg), evaluation_mode=str(cfg.evaluation_mode),
    ), config


def _population(tmp_path):
    return f"scenario_set_path={write_inputs(tmp_path)}", "approved_scenario_sha256=null", "sampling.campaign_mode=pilot"


def test_the_serving_option_puts_every_arm_on_its_engine(tmp_path):
    cfg = _compose("sampling_training", "campaign_id=training", "training.family=base", "training_serving=merged_fp32")
    validate_campaign_cfg(cfg)
    assert (cfg.judge_model.dtype, cfg.judge_model.attention_backend) == ("float32", "TRITON_ATTN")
    assert campaign_module.judge_serving_dtype(cfg.judge_model) == "float32"
    assert cfg.judge_model.model_path == str(tmp_path / "models" / "Qwen3-8B")
    assert cfg.judge_model.get("adapter_path") is None and cfg.approved_adapter_sha256 is None
    assert cfg.approved_model_weights_sha256 is None


DEV = ("grpo_dev.campaign_prefix=grpo-dev-t",)


@pytest.mark.parametrize("stage, mode, draws, campaign_mode", [
    ("references_bank", "individual_bank", 4, "pilot"), ("references_full", "full", 8, "pilot"),
    ("stage1", "full", 2, "pilot"), ("probe", "individual_bank", 1, "screening"),
])
def test_every_dev_stage_composes_into_its_own_campaign(tmp_path, modelspec, stage, mode, draws, campaign_mode):
    cfg = _compose("sampling_training_dev", *DEV, f"grpo_dev.stage={stage}", f"scenario_set_path={write_inputs(tmp_path)}",
                   "approved_scenario_sha256=null")
    spec, config = _spec(cfg, modelspec)
    assert cfg.campaign_id == f"grpo-dev-t-{stage}" and spec.condition_id == f"Qwen3-8B/grpo-v2-dev/{stage}/base"
    assert (spec.evaluation_mode, config.num_decoder_draws, config.campaign_mode) == (mode, draws, campaign_mode)
    assert dict(spec.extra) == {"figure": "grpo_v2_dev", "stage": stage, "policy": "base", "policy_family": None,
                                "policy_reference": None}


def test_a_dev_reference_condition_is_the_training_base_arms_but_for_its_label_and_mode(tmp_path, modelspec):
    training, _ = _spec(_compose("sampling_training", "campaign_id=training", "training.family=base",
                                 "training_serving=merged_fp32", *_population(tmp_path)), modelspec)
    for stage in ("references_bank", "references_full"):
        dev, _ = _spec(_compose("sampling_training_dev", *DEV, f"grpo_dev.stage={stage}",
                                "training_serving=merged_fp32", f"scenario_set_path={write_inputs(tmp_path)}",
                                "approved_scenario_sha256=null"), modelspec)
        anchors.check_condition_projection(dev, training, side=anchors.SIDE_DEV)
    with pytest.raises(anchors.AnchorError, match="serving_dtype"):
        dev, _ = _spec(_compose("sampling_training_dev", *DEV, "grpo_dev.stage=references_bank",
                                "training_serving=merged_fp32", "judge_model.dtype=bfloat16",
                                f"scenario_set_path={write_inputs(tmp_path)}", "approved_scenario_sha256=null"), modelspec)
        anchors.check_condition_projection(dev, training, side=anchors.SIDE_DEV)


def test_a_checkpoint_is_declared_to_a_dev_campaign_as_the_serving_names_it(tmp_path, modelspec):
    from ruleforge.grpo import merging as merge_module

    merged = write_merged(tmp_path / "grpo" / "merged", "checkpoint-300")
    cfg = _compose("sampling_training_dev", *DEV, "grpo_dev.stage=stage1", "training.policy=checkpoint-300",
                   f"+training.policies.checkpoint-300={{model_path: {merged}, "
                   f"weights_sha256: {merge_module.weights_digest(merged)}}}",
                   f"scenario_set_path={write_inputs(tmp_path)}", "approved_scenario_sha256=null")
    spec, _ = _spec(cfg, modelspec)
    assert spec.model_weights_sha256 == merge_module.weights_digest(merged) and spec.serving_dtype == "float32"
    assert spec.condition_id == "Qwen3-8B/grpo-v2-dev/stage1/checkpoint-300"


def _save_words(option, merged, digest, population, *extra):
    return (f"training_serving={option}", "training.policy=checkpoint-10",
            f"+training.policies.checkpoint-10={{model_path: {merged}, weights_sha256: '{digest}'}}", "grpo_dev.stage=stage1",
            "grpo_dev.campaign_prefix=grpo-dev-t", f"scenario_set_path={population}", *extra)


def test_a_merged_save_is_its_qwen_campaign_s_weights(tmp_path):
    cfg = _compose("sampling_training_dev", *_save_words("merged_fp32", tmp_path / "merged/checkpoint-10", "a" * 64,
                                                         tmp_path / "p.jsonl", "judge_model.tp_size=2"))
    assert (cfg.judge_model.model_name, cfg.judge_model.tp_size, cfg.judge_model.reasoning_parser) == ("Qwen3-8B", 2, "qwen3")
    assert str(cfg.judge_model.model_path).endswith("merged/checkpoint-10")


def test_gemmas_serving_option_serves_gemma_with_strict_think_off(tmp_path, modelspec):
    from ruleforge.grpo import merging as merge_module
    from tests.test_sampling_weights import write_weights

    write_weights(tmp_path / "models" / "gemma-4-31B-it", seed=-2.0)
    references = _compose("sampling_training_dev", *DEV, "grpo_dev.stage=references_full", "training.policy=base",
                          "training_serving=gemma_merged_fp32", *_population(tmp_path))
    base, _ = _spec(references, modelspec)
    merged = write_merged(tmp_path / "volume" / "gemma-run" / "merged", "checkpoint-100", seed=3.0)
    digest = merge_module.weights_digest(merged)
    checkpoint = _compose("sampling_training_dev", *DEV, "grpo_dev.stage=stage1", "training_serving=gemma_merged_fp32",
                          "training.policy=checkpoint-100",
                          f"+training.policies.checkpoint-100={{model_path: {merged}, weights_sha256: '{digest}'}}",
                          f"scenario_set_path={write_inputs(tmp_path)}", "approved_scenario_sha256=null",
                          "sampling.campaign_mode=pilot")
    ckpt, _ = _spec(checkpoint, modelspec)
    for cfg in (references, checkpoint):
        assert (cfg.judge_model.model_name, cfg.judge_model.dtype, cfg.judge_model.tp_size) == (
            "gemma-4-31B-it", "float32", 2)
        assert (cfg.judge_model.strip_thinking_model, cfg.sampling.strict_think) == ("gemma4", False)
    assert (base.condition_id, ckpt.condition_id) == ("gemma-4-31B-it/grpo-v2-dev/references_full/base",
                                                      "gemma-4-31B-it/grpo-v2-dev/stage1/checkpoint-100")
    assert (base.model_path, base.model_weights_sha256) == ("gemma-4-31B-it", None)
    assert (ckpt.model_weights_sha256, base.serving_dtype, ckpt.serving_dtype) == (digest, "float32", "float32")
    differ = {"condition_id", "evaluation_mode", "model_path", "model_weights_sha256", "extra"}
    assert {k: v for k, v in base.to_dict().items() if k not in differ} == {
        k: v for k, v in ckpt.to_dict().items() if k not in differ}
    with pytest.raises(CampaignRunError, match="strict_think"):
        preflight(_compose("sampling_training_dev", *DEV, "grpo_dev.stage=references_full", "training.policy=base",
                           "training_serving=gemma_merged_fp32", "sampling.strict_think=true", *_population(tmp_path)))


def test_gemmas_references_project_onto_its_anchors_condition(tmp_path, modelspec):
    from ruleforge.launch import cli, control
    from tests.test_launch_specs import _options
    from tests.test_sampling_weights import write_weights

    write_weights(tmp_path / "models" / "gemma-4-31B-it", seed=-2.0)

    def composed(name, variables, *knobs):
        _, arguments, _, _ = cli.load_spec(REPO_ROOT / control.SPECS_DIR / f"{name}.yaml", variables, "L1",
                                           _options("campaign"), repo=REPO_ROOT, verb="campaign")
        return _compose(*arguments[1:2], *arguments[2:], *knobs, *_population(tmp_path))

    references = composed("gemma_references", {"stage": "references_bank", "campaign_prefix": "gemma-dev-t",
                                               "population": "p.jsonl", "population_sha256": "ab"})
    bank = composed("gemma_anchors", {"stage": "bank", "campaign_prefix": "gemma-anchors-t"},
                    f"campaign_root={tmp_path / 'artifacts' / 'sampling' / 'gemma-anchors-t-bank' / 'b'}")
    reference, _ = _spec(references, modelspec)
    anchor, _ = _spec(bank, modelspec)
    assert (reference.serving_dtype, anchor.serving_dtype) == ("float32", "bfloat16")
    anchors.check_reference(reference)
    anchors.check_condition_projection(anchor, reference, side=anchors.SIDE_TRAIN)
    with pytest.raises(anchors.AnchorError, match="strict_think"):
        anchors.check_condition_projection(dataclasses.replace(anchor, strict_think=True), reference,
                                           side=anchors.SIDE_TRAIN)
