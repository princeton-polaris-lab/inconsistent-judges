from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest
from omegaconf import OmegaConf

from ruleforge.core.record_names import identity_key
from ruleforge.launch import cli, control
from ruleforge.launch.guards import compose
from ruleforge.launch.jobs import JobSpec, Resources, members
from tests.test_launch_specs import EXAMPLES, REPO, _options

#: The recorded prompt key each launch spec and sampling config composes to; every paper path keeps it. RuleForge's specs
#: are the exception, since its rerun runs on naive.
PROMPT_KEYS = Path(__file__).parent / "fixtures" / "launch" / "prompt_keys.json"
#: The specs that judge, each with the variables a launch must give beyond ``EXAMPLES``.
CAMPAIGN_SPECS = {"batch_judge": {}, "hierarchy": {}, "prompt_ablation": {}, "prompt_ablation_naive": {},
                  "ruleforge_rulesets": {}, "training_arm_gemma": {}, "training_arms": {},
                  "gemma_references": {}, "gemma_anchors": {"stage": "bank", "campaign_prefix": "c"},
                  "grpo_anchors": {}}
JOB_SPECS = {"ruleforge_modelspec": ("config", {"run_name": "r"}), "ruleforge_smoke": ("config", {}),
             "ruleset_size_open_models": ("sampling_ruleset_size", {"campaign_id": "c", "seeds": "[0]"}),
             "ruleset_size_batch_judge": ("sampling_ruleset_size", {"campaign_id": "c", "shares": "[0]"})}
#: The specs that run no judge prompt of their own.
NO_PROMPT_SPECS = {"grpo_rows_gemma", "grpo_train_gemma_run", "grpo_train_run", "merge_fp32"}
SAMPLING_CONFIGS = {"sampling_campaign": ["campaign_id=c"], "sampling_ruleforge": ["campaign_id=c"],
                    "sampling_ruleset_size": ["campaign_id=c"], "sampling_training": ["campaign_id=c"],
                    "sampling_training_anchors": ["campaign_id=c"], "sampling_training_dev": ["campaign_id=c"]}
KNOBS = ("campaign_root=/r", "scenario_set_path=/p.jsonl", "approved_scenario_sha256=ab")


RULEFORGE_SPECS = {"spec/ruleforge_modelspec", "spec/ruleforge_rulesets", "spec/ruleforge_smoke"}
NAIVE_KEY, PAPER_NAIVE_KEY = "legacy_v2", "legacy"


def _key(cfg: Any) -> str:
    return identity_key(str(OmegaConf.select(cfg, "eval_cfg.reasoning_prompt_style")))


def _campaign_keys(name: str, given: dict) -> Iterator[list[str]]:
    variables = {key: value for key, value in EXAMPLES.get(name, {}).items() if key != "prompt"} | given
    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", variables, "L1",
                                            _options("campaign"), repo=REPO, verb="campaign")
    template = JobSpec("t", "campaign", 0, ("program", *arguments), Resources("gpu", gpus=1))
    for member in members(template, values.get("axis") or []):
        cfg = compose(REPO, [*member.argv[1:], *KNOBS], default_config="sampling_campaign")
        yield [str(OmegaConf.select(cfg, "judge_model.model_name")), _key(cfg)]


def _job_keys(name: str, config: str, given: dict) -> Iterator[list[str]]:
    values, arguments, _, _ = cli.load_spec(REPO / control.SPECS_DIR / f"{name}.yaml", given, "L1",
                                            _options("job"), repo=REPO, verb="job")
    template = JobSpec("t", "job", 0, ("program", *arguments), Resources("gpu", gpus=1))
    for member in members(template, values.get("axis") or []):
        overrides = [word for word in member.argv[1:] if "=" in word and not word.startswith("hydra.")]
        cfg = compose(REPO, [*overrides, *(() if config == "config" else KNOBS)], default_config=config)
        yield [str(OmegaConf.select(cfg, "judge_model.model_name")), _key(cfg)]


def composed_keys() -> dict[str, list[list[str]]]:
    """Return each spec's and sampling config's sorted ``[model, recorded prompt key]`` rows, as they compose."""
    keys: dict[str, list[list[str]]] = {}
    for name, given in CAMPAIGN_SPECS.items():
        keys[f"spec/{name}"] = sorted(_campaign_keys(name, given))
    for name, (config, given) in JOB_SPECS.items():
        keys[f"spec/{name}"] = sorted(_job_keys(name, config, given))
    for name, overrides in SAMPLING_CONFIGS.items():
        cfg = compose(REPO, [*overrides, *KNOBS], default_config=name)
        keys[f"config/{name}"] = [[str(OmegaConf.select(cfg, "judge_model.model_name")), _key(cfg)]]
    return keys


@pytest.fixture
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")


def test_every_spec_is_listed_as_judging_or_not():
    assert {path.stem for path in (REPO / control.SPECS_DIR).glob("*.yaml")} == {
        *CAMPAIGN_SPECS, *JOB_SPECS, *NO_PROMPT_SPECS}


def test_every_paper_path_composes_to_the_recorded_key_it_had_and_ruleforge_to_naive(_env):
    recorded = json.loads(PROMPT_KEYS.read_text(encoding="utf-8"))
    composed = composed_keys()
    assert set(composed) == set(recorded)
    assert {name: rows for name, rows in composed.items() if name not in RULEFORGE_SPECS} == {
        name: rows for name, rows in recorded.items() if name not in RULEFORGE_SPECS}
    for name in RULEFORGE_SPECS:
        assert composed[name] == [[model, NAIVE_KEY] for model, key in recorded[name] if key == PAPER_NAIVE_KEY]


def test_a_ruleforge_spec_given_the_paper_s_prompt_composes_to_the_recorded_key(_env):
    recorded = json.loads(PROMPT_KEYS.read_text(encoding="utf-8"))
    paper = {"prompt": "naive-rules-above"}
    assert sorted(_campaign_keys("ruleforge_rulesets", paper)) == recorded["spec/ruleforge_rulesets"]
    for spec in ("ruleforge_modelspec", "ruleforge_smoke"):
        config, given = JOB_SPECS[spec]
        assert sorted(_job_keys(spec, config, {**given, **paper})) == recorded[f"spec/{spec}"]
