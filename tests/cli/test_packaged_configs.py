from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ruleforge.cli import evaluate as evaluate_cli
from ruleforge.cli import optimize as optimize_cli
from ruleforge.cli.common import setting
from ruleforge.cli.compose import compose
from ruleforge.cli.judging import judge_assignment
from ruleforge.env import package_path
from ruleforge.judges.batch_api import BatchSpendNotConfirmed

REPO = Path(__file__).resolve().parents[2]


def _values(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def test_each_packaged_model_config_is_its_checkout_twin_with_its_roots_renamed():
    from ruleforge.judges.base import PACKAGED_ROOTS, as_packaged

    assert dict(PACKAGED_ROOTS) == {"${oc.env:RULEFORGE_MODEL_ROOT}": "${rf.env:MODEL_ROOT}",
                                    "${oc.env:RULEFORGE_BATCH_ROOT}": "${rf.env:BATCH_ROOT,null}"}
    packaged = {path.stem: path for path in (package_path("configs") / "model").glob("*.yaml")}
    checkout = {path.stem: path for path in (REPO / "configs" / "model").glob("*.yaml")}
    assert packaged.keys() == checkout.keys() and len(packaged) >= 10
    for name, path in packaged.items():
        twin = OmegaConf.to_container(as_packaged(OmegaConf.load(checkout[name])), resolve=False)
        unchanged = OmegaConf.to_container(as_packaged(OmegaConf.load(path)), resolve=False)
        assert _values(path) == twin == unchanged, name
    nested = OmegaConf.create({"a": "${oc.env:RULEFORGE_MODEL_ROOT}/w", "b": {"c": ["${oc.env:RULEFORGE_BATCH_ROOT}", 1]},
                               "d": None, "e": "${oc.env:RULEFORGE_ARTIFACT_ROOT}"})
    assert OmegaConf.to_container(as_packaged(nested), resolve=False) == {
        "a": "${rf.env:MODEL_ROOT}/w", "b": {"c": ["${rf.env:BATCH_ROOT,null}", 1]}, "d": None,
        "e": "${oc.env:RULEFORGE_ARTIFACT_ROOT}"}


def test_a_root_inside_a_longer_value_is_renamed_without_a_default(monkeypatch):
    """Only a whole ``${oc.env:RULEFORGE_BATCH_ROOT}`` takes the packaged default (null)."""
    from omegaconf.errors import InterpolationResolutionError

    from ruleforge.judges.base import as_packaged

    for name in ("RULEFORGE_BATCH_ROOT", "RULEFORGE_MODEL_ROOT"):
        monkeypatch.delenv(name, raising=False)
    written = {"batch_root": "${oc.env:RULEFORGE_BATCH_ROOT}/x", "whole": "${oc.env:RULEFORGE_BATCH_ROOT}",
               "model_path": "${oc.env:RULEFORGE_MODEL_ROOT}/w"}
    node = as_packaged(OmegaConf.create(written))
    assert OmegaConf.to_container(node, resolve=False) == {"batch_root": "${rf.env:BATCH_ROOT}/x",
                                                           "whole": "${rf.env:BATCH_ROOT,null}",
                                                           "model_path": "${rf.env:MODEL_ROOT}/w"}
    assert node.whole is None
    for key in ("batch_root", "model_path"):
        with pytest.raises(InterpolationResolutionError, match="EnvUnsetError.*is unset"):
            getattr(node, key)
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", "/b")
    assert (node.batch_root, node.whole) == ("/b/x", "/b")


def test_the_panel_configs_are_the_checkout_s_but_for_the_prompt_a_new_run_defaults_to():
    for path in (package_path("configs") / "eval_rule_set").glob("*.yaml"):
        packaged, checkout = path.read_text(), (REPO / "configs" / "eval_rule_set" / path.name).read_text()
        assert packaged.replace('reasoning_prompt_style: "naive"', 'reasoning_prompt_style: "naive-rules-above"') == checkout
    assert 'reasoning_prompt_style: "naive"\n' in (package_path("configs") / "eval_rule_set" / "default.yaml").read_text()


def test_the_training_config_is_the_checkout_s_less_its_paths_and_wandb():
    packaged = _values(package_path("configs/grpo/base.yaml"))
    checkout = _values(REPO / "configs" / "training" / "grpo" / "base.yaml")
    del checkout["hydra"]
    assert packaged.pop("output_root") == "${rf.env:ARTIFACT_ROOT}/training"
    assert checkout.pop("output_root") == "${oc.env:RULEFORGE_ARTIFACT_ROOT}/training"
    assert (packaged["data"].pop("rules"), packaged["data"].pop("rules_sha256")) == ("???", "???")
    del checkout["data"]["rules"], checkout["data"]["rules_sha256"]
    assert packaged["wandb"].pop("entity") is None and checkout["wandb"].pop("entity").startswith("${oc.env:")
    assert packaged["grpo"].pop("report_to") == [] and checkout["grpo"].pop("report_to") == ["wandb"]
    assert packaged == checkout


def test_evaluate_s_defaults_are_the_api_s(inputs, tmp_path):
    from ruleforge.evaluate.api import ConditionSettings, SamplingSettings
    from ruleforge.evaluate.packing import ExecutionSettings

    scenarios, rules = inputs
    cfg = compose(evaluate_cli.operation(), "evaluate.yaml", cwd=tmp_path, layers=([
        setting("rule_set_path", str(rules), "--rules"), setting("scenario_set_path", str(scenarios), "--scenarios"),
        judge_assignment("fake", "--judge fake", node="judge_model", cwd=tmp_path,
                         name="__judge__")],))
    call = evaluate_cli.convert(cfg)
    assert call.sampling == SamplingSettings()
    assert call.condition == ConditionSettings(task="modelspec", extra={})
    assert call.execution == ExecutionSettings()
    assert (call.out, call.campaign_id, call.condition_id, call.confirm_spend, call.relocate_batch_root) == (
        None, None, None, False, False)
    cfg.confirm_spend = "false"
    with pytest.raises(BatchSpendNotConfirmed, match="confirm_spend is true or false, got 'false'"):
        evaluate_cli.convert(cfg)


def test_optimize_s_defaults_are_the_api_s(tmp_path):
    from ruleforge.optimize import OptimizeConfig

    cfg = compose(optimize_cli.operation(), "optimize.yaml", cwd=tmp_path, layers=([
        setting("rule_amendment.initial_rule_set_path", "/r.jsonl", "--rules"), setting("out", "/o", "--out"),
        judge_assignment("fake", "--judge fake", node="judge_model", cwd=tmp_path,
                         name="__judge__")],))
    call = optimize_cli.convert(cfg)
    assert call["config"] == OptimizeConfig() and call["seeds"] == {} and call["task"] == "modelspec"
