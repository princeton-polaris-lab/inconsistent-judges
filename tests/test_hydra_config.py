from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from ruleforge.core.paths import absolute_strings as _literal_absolute_paths

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = str(REPO_ROOT / "configs")
TASK_DIR = REPO_ROOT / "configs" / "task"
MODEL_DIR = REPO_ROOT / "configs" / "model"

def _discover_task_overrides() -> list[str]:
    overrides = []
    for base_dir in sorted(TASK_DIR.iterdir()):
        if not base_dir.is_dir():
            continue
        for yaml in sorted(base_dir.glob("*.yaml")):
            if yaml.stem == "base":
                continue
            overrides.append(f"task={base_dir.name}/{yaml.stem}")
    return overrides


TASK_OVERRIDES = _discover_task_overrides()


def _discover_model_configs() -> list[str]:
    return [yaml.name for yaml in sorted(MODEL_DIR.glob("*.yaml"))]


MODEL_CONFIGS = _discover_model_configs()

KNOWN_ABSOLUTE_MODEL_PATHS: set[str] = set()

PORTABLE_MODEL_CONFIGS = [
    name for name in MODEL_CONFIGS if name not in KNOWN_ABSOLUTE_MODEL_PATHS
]


def _scan_config_for_absolute_paths(yaml_path: Path) -> list[tuple[str, str]]:
    container = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=False)
    return list(_literal_absolute_paths(container))

def _resolve_data_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def _compose_cfg(task_override: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=[task_override])
    return cfg


# Top-level keys read by trainer.py and hydra dir interpolation
TOP_LEVEL_KEYS = {"task_spec", "task_name", "max_episodes", "run_name"}

# scenario_generation keys read by trainer + envs + agents + task specs
SG_REQUIRED_KEYS = {
    "initial_scenario_seed_type",
    "num_seed_scenarios",
    "target_num_scenarios",
    "num_scenarios_per_batch",
    "entropy_threshold",
    "entropy_method",
    "max_accumulation_iterations",
    "termination_threshold",
    "termination_episodes",
    "trigger_threshold",
    "posterior_update_temperature",
    "posterior_sampling_temperature",
}

# rule_amendment keys read by trainer + envs + agents + task specs
RA_REQUIRED_KEYS = {
    "initial_rule_set_path",
    "intent_weight",
    "entropy_method",
    "intent_method",
    "best_of_n",
    "termination_threshold",
    "termination_episodes",
    "posterior_update_temperature",
    "posterior_sampling_temperature",
}

# eval_cfg keys read by task spec __init__
EVAL_REQUIRED_KEYS = {
    "judge_types",
    "save_batch_size",
    "evaluation_temperature",
    "individual_rule_context",
}

# Valid task_spec names (must be registered in registry.py)
VALID_TASK_SPECS = {"modelspec", "legal"}

VALID_SEED_TYPES = {
    "modelspec": {"none", "wildchat"},
    "legal": {"none", "single_seed"},
}

SEED_TYPE_EXTRA_KEYS = {
    "wildchat": {"good_seed_scenario_path", "bad_seed_scenario_path"},
    "single_seed": {"seed_scenario_path"},
    "none": set(),
}


@pytest.mark.parametrize("task_override", TASK_OVERRIDES, ids=TASK_OVERRIDES)
class TestTaskConfig:
    """Validate every discovered task config against pipeline expectations."""

    @pytest.fixture(autouse=True)
    def _model_root_env(self, monkeypatch):
        monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/dev/null/model-root-unused-in-tests")

    def test_config_composes(self, task_override):
        cfg = _compose_cfg(task_override)
        assert cfg is not None

    def test_top_level_keys(self, task_override):
        cfg = _compose_cfg(task_override)
        missing = TOP_LEVEL_KEYS - set(OmegaConf.to_container(cfg, resolve=True).keys())
        assert not missing, f"Missing top-level keys: {missing}"

    def test_task_spec_is_valid(self, task_override):
        cfg = _compose_cfg(task_override)
        assert cfg.task_spec in VALID_TASK_SPECS, (
            f"task_spec='{cfg.task_spec}' not in {VALID_TASK_SPECS}"
        )

    def test_scenario_generation_keys(self, task_override):
        cfg = _compose_cfg(task_override)
        sg = OmegaConf.to_container(cfg.scenario_generation, resolve=True)
        missing = SG_REQUIRED_KEYS - set(sg.keys())
        assert not missing, f"Missing scenario_generation keys: {missing}"

    def test_rule_amendment_keys(self, task_override):
        cfg = _compose_cfg(task_override)
        ra = OmegaConf.to_container(cfg.rule_amendment, resolve=True)
        missing = RA_REQUIRED_KEYS - set(ra.keys())
        assert not missing, f"Missing rule_amendment keys: {missing}"

    def test_eval_cfg_keys(self, task_override):
        cfg = _compose_cfg(task_override)
        ev = OmegaConf.to_container(cfg.eval_cfg, resolve=True)
        missing = EVAL_REQUIRED_KEYS - set(ev.keys())
        assert not missing, f"Missing eval_cfg keys: {missing}"

    def test_seed_type_valid_for_task(self, task_override):
        cfg = _compose_cfg(task_override)
        seed_type = cfg.scenario_generation.initial_scenario_seed_type
        allowed = VALID_SEED_TYPES.get(cfg.task_spec, set())
        assert seed_type in allowed, (
            f"seed type '{seed_type}' not valid for task_spec '{cfg.task_spec}'. "
            f"Allowed: {allowed}"
        )

    def test_seed_type_extra_keys(self, task_override):
        cfg = _compose_cfg(task_override)
        seed_type = cfg.scenario_generation.initial_scenario_seed_type
        extra = SEED_TYPE_EXTRA_KEYS.get(seed_type, set())
        if not extra:
            return
        sg = OmegaConf.to_container(cfg.scenario_generation, resolve=True)
        missing = extra - set(sg.keys())
        assert not missing, (
            f"seed type '{seed_type}' requires SG keys {extra}, missing: {missing}"
        )

    def test_data_files_exist(self, task_override):
        cfg = _compose_cfg(task_override)
        ra = cfg.rule_amendment
        sg = cfg.scenario_generation

        rule_path = _resolve_data_path(ra.initial_rule_set_path)
        assert rule_path.exists(), f"Rule set file not found: {rule_path}"

        seed_type = sg.initial_scenario_seed_type
        paths_to_check = []
        if seed_type == "single_seed":
            paths_to_check.append(sg.seed_scenario_path)
        elif seed_type == "wildchat":
            paths_to_check.append(sg.good_seed_scenario_path)
            paths_to_check.append(sg.bad_seed_scenario_path)

        for p in paths_to_check:
            pp = _resolve_data_path(p)
            assert pp.exists(), f"Seed scenario file not found: {pp}"

    def test_rule_set_is_valid_jsonl(self, task_override):
        cfg = _compose_cfg(task_override)
        rule_path = _resolve_data_path(cfg.rule_amendment.initial_rule_set_path)

        with open(rule_path) as f:
            lines = [raw.strip() for raw in f if raw.strip()]

        assert len(lines) > 0, f"Rule set file is empty: {rule_path}"
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert "id" in obj, f"Line {i} missing 'id' key in {rule_path}"
            assert "rule" in obj, f"Line {i} missing 'rule' key in {rule_path}"

    def test_numeric_types(self, task_override):
        cfg = _compose_cfg(task_override)
        # These are all cast with float() or int() in the pipeline
        float(cfg.max_episodes)
        float(cfg.scenario_generation.target_num_scenarios)
        float(cfg.scenario_generation.entropy_threshold)
        float(cfg.scenario_generation.trigger_threshold)
        float(cfg.scenario_generation.posterior_update_temperature)
        float(cfg.scenario_generation.posterior_sampling_temperature)
        int(cfg.scenario_generation.max_accumulation_iterations)
        float(cfg.rule_amendment.intent_weight)
        int(cfg.rule_amendment.best_of_n)
        float(cfg.rule_amendment.posterior_update_temperature)
        float(cfg.rule_amendment.posterior_sampling_temperature)

    def test_model_configs_present(self, task_override):
        cfg = _compose_cfg(task_override)
        assert hasattr(cfg, "judge_model"), "Missing judge_model config"
        assert hasattr(cfg, "agent_model"), "Missing agent_model config"
        assert hasattr(cfg.judge_model, "model_type"), "judge_model missing model_type"
        assert hasattr(cfg.agent_model, "model_type"), "agent_model missing model_type"


def _compose_eval_cfg(option: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="config", overrides=[f"eval_rule_set@eval_cfg={option}"]
        )
    return cfg


def test_prompt_ablation_eval_cfg_composes_with_every_required_key():
    cfg = _compose_eval_cfg("prompt_ablation")
    ev = OmegaConf.to_container(cfg.eval_cfg, resolve=True)
    missing = EVAL_REQUIRED_KEYS - set(ev.keys())
    assert not missing, f"Missing eval_cfg keys: {missing}"


def test_prompt_ablation_pins_the_judge_panel_to_the_paper_strategy():
    cfg = _compose_eval_cfg("prompt_ablation")
    assert list(cfg.eval_cfg.judge_types) == ["No_Interpretation"]


def test_prompt_ablation_carries_the_manipulated_variable():
    cfg = _compose_eval_cfg("prompt_ablation")
    assert cfg.eval_cfg.reasoning_prompt_style == "naive-rules-above"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        overridden = compose(
            config_name="config",
            overrides=[
                "eval_rule_set@eval_cfg=prompt_ablation",
                "eval_cfg.reasoning_prompt_style=forced-grpo",
            ],
        )
    assert overridden.eval_cfg.reasoning_prompt_style == "forced-grpo"
    assert list(overridden.eval_cfg.judge_types) == ["No_Interpretation"]


def test_model_config_discovery_is_not_vacuous():
    discovered = set(MODEL_CONFIGS)
    stale = KNOWN_ABSOLUTE_MODEL_PATHS - discovered
    assert not stale, f"allowlisted but absent from configs/model/: {sorted(stale)}"
    assert PORTABLE_MODEL_CONFIGS, (
        "every model config is allowlisted — the scan below checks nothing"
    )
    assert LOCAL_MODEL_CONFIGS, (
        "no local model config discovered — the contract tests below check nothing"
    )


@pytest.mark.parametrize(
    "model_config", PORTABLE_MODEL_CONFIGS, ids=PORTABLE_MODEL_CONFIGS
)
def test_model_config_holds_no_literal_absolute_path(model_config):
    found = _scan_config_for_absolute_paths(MODEL_DIR / model_config)
    assert found == [], (
        f"absolute path(s) in configs/model/{model_config}: {found}. Move the "
        "location into ${oc.env:VAR} (no absolute default) and export it from "
        "the launch script."
    )


@pytest.mark.parametrize(
    "model_config",
    sorted(KNOWN_ABSOLUTE_MODEL_PATHS),
    ids=sorted(KNOWN_ABSOLUTE_MODEL_PATHS),
)
def test_the_declared_model_path_debt_is_real_and_not_stale(model_config):
    assert _scan_config_for_absolute_paths(MODEL_DIR / model_config), (
        f"configs/model/{model_config} no longer holds an absolute path; remove "
        "it from KNOWN_ABSOLUTE_MODEL_PATHS so the rule applies to it."
    )


LOCAL_MODEL_CONFIGS = [
    name
    for name in MODEL_CONFIGS
    if OmegaConf.load(MODEL_DIR / name).get("model_type") == "local"
]


@pytest.mark.parametrize(
    "model_config", LOCAL_MODEL_CONFIGS, ids=LOCAL_MODEL_CONFIGS
)
def test_a_local_model_path_resolves_under_the_exported_root(
    model_config, monkeypatch
):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/exported/root")
    cfg = OmegaConf.load(MODEL_DIR / model_config)
    assert str(cfg.model_path).startswith("/exported/root/")


#: The key each API transport asks for reasoning with, and the value that asks.
REASONING_REQUEST_KEYS = {
    "anthropic_batch": ("thinking_display", "summarized"),
    "anthropic": ("thinking_display", "summarized"),
    "openai_batch": ("reasoning_summary", "auto"),
    "openai": ("reasoning_summary", "auto"),
}
#: API configs exempt from the rule, each with why.
REASONING_REQUEST_EXEMPT: dict[str, str] = {}
API_MODEL_CONFIGS = [
    name for name in MODEL_CONFIGS if OmegaConf.load(MODEL_DIR / name).get("model_type") != "local"
]


@pytest.mark.parametrize("model_config", API_MODEL_CONFIGS, ids=API_MODEL_CONFIGS)
def test_every_api_judge_config_asks_its_provider_to_show_its_reasoning(model_config):
    cfg = OmegaConf.load(MODEL_DIR / model_config)
    if model_config in REASONING_REQUEST_EXEMPT:
        return
    model_type = cfg.get("model_type")
    assert model_type in REASONING_REQUEST_KEYS, (
        f"{model_config}: model_type {model_type!r} has no reasoning request key here; add it, or exempt the config"
    )
    key, value = REASONING_REQUEST_KEYS[model_type]
    assert cfg.get(key) == value, f"{model_config} must set {key}: {value}"
    if key == "reasoning_summary":
        # The responses endpoint sends a summary only beside an effort.
        assert cfg.get("reasoning_level") or cfg.get("gpt_oss_reasoning_level"), model_config


def test_the_reasoning_request_rule_is_not_vacuous():
    assert API_MODEL_CONFIGS and not set(REASONING_REQUEST_EXEMPT) - set(API_MODEL_CONFIGS)


def test_an_unset_model_root_fails_naming_the_variable(monkeypatch):
    monkeypatch.delenv("RULEFORGE_MODEL_ROOT", raising=False)
    cfg = OmegaConf.load(MODEL_DIR / "gemma4-31b-tp-2.yaml")
    with pytest.raises(InterpolationResolutionError, match="RULEFORGE_MODEL_ROOT"):
        _ = cfg.model_path


#: Stand-ins for what a weights directory may ship (None: no file at all).
SHIPPED_VARIANTS = {
    "no_file": None,
    "gemma_like": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "do_sample": True},
    "every_key": {"top_k": 40, "top_p": 0.9, "min_p": 0.05, "repetition_penalty": 1.1, "presence_penalty": 1.5},
    "nulls": {"top_k": None, "top_p": 0.8},
}


def _resolution_before_default_truncation(policy: str, model_type: str, model_path: Path):
    """The truncation resolution without ``default_truncation``: what every config without the key must match."""
    from ruleforge.judges.models import TRUNCATION_KEYS, UNTRUNCATED_SAMPLING

    if model_type != "local":
        return "provider_default"
    if policy == "none":
        return dict(UNTRUNCATED_SAMPLING)
    path = model_path / "generation_config.json"
    shipped = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        shipped = {key: payload[key] for key in TRUNCATION_KEYS if payload.get(key) is not None}
    return {**UNTRUNCATED_SAMPLING, **shipped}


def test_no_committed_model_config_declares_a_default_truncation():
    declared = {
        name: OmegaConf.load(MODEL_DIR / name).get("default_truncation") for name in MODEL_CONFIGS
    }
    assert {name: value for name, value in declared.items() if value is not None} == {}


@pytest.mark.parametrize("model_config", MODEL_CONFIGS, ids=MODEL_CONFIGS)
def test_every_committed_model_config_resolves_its_truncation_as_before(model_config, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ruleforge.evaluate.campaign import resolve_sampling_truncation

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    cfg = OmegaConf.load(MODEL_DIR / model_config)
    model_type = str(cfg.model_type)
    weights = Path(str(cfg.model_path)) if model_type == "local" else tmp_path / "unused"
    weights.mkdir(parents=True, exist_ok=True)
    for variant, shipped in SHIPPED_VARIANTS.items():
        (weights / "generation_config.json").unlink(missing_ok=True)
        if shipped is not None:
            (weights / "generation_config.json").write_text(json.dumps(shipped), encoding="utf-8")
        for policy in ("model_default", "none"):
            now = resolve_sampling_truncation(SimpleNamespace(truncation=policy), model_type, cfg)
            before = _resolution_before_default_truncation(policy, model_type, weights)
            assert json.dumps(now, sort_keys=True) == json.dumps(before, sort_keys=True), (variant, policy)


def _resolved(name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(MODEL_DIR / name), resolve=True)


def test_the_fp32_engine_is_the_bf16_config_with_dtype_float32_on_triton_attn(monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/exported/root")
    bf16, fp32 = _resolved("qwen3-8b.yaml"), _resolved("qwen3-8b-fp32.yaml")
    assert {key: (bf16.get(key), fp32.get(key)) for key in bf16.keys() | fp32.keys() if bf16.get(key) != fp32.get(key)} == {
        "dtype": (None, "float32"), "attention_backend": ("FLASH_ATTN", "TRITON_ATTN"),
    }
    assert (fp32["model_name"], fp32["model_path"], fp32["max_new_tokens"], fp32["max_model_len"]) == (
        "Qwen3-8B", "/exported/root/Qwen3-8B", 32768, None
    )
    assert fp32["enable_thinking"] is True and "adapter_path" not in fp32


def test_the_fp32_config_reaches_the_engine_as_float32_on_triton_attn(monkeypatch):
    from ruleforge.judges.backend import local_engine_settings
    from ruleforge.judges.models import local_engine_kwargs

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/exported/root")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    engine = local_engine_kwargs(**local_engine_settings(OmegaConf.load(MODEL_DIR / "qwen3-8b-fp32.yaml"), engine_fields=()))
    assert (engine["dtype"], engine["attention_config"]) == ("float32", {"backend": "TRITON_ATTN"})
    assert (engine["model"], engine["max_model_len"], engine["enable_lora"]) == ("/exported/root/Qwen3-8B", None, False)


def test_only_the_fp32_engines_declare_a_dtype():
    declared = {name: OmegaConf.load(MODEL_DIR / name).get("dtype") for name in MODEL_CONFIGS}
    assert {name: value for name, value in declared.items() if value is not None} == {
        "qwen3-8b-fp32.yaml": "float32", "gemma4-31b-fp32.yaml": "float32"}


def test_the_gemma_fp32_engine_is_gemmas_tp_2_config_with_the_dtype_alone(monkeypatch):
    from ruleforge.judges.backend import local_engine_settings
    from ruleforge.judges.models import local_engine_kwargs

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/exported/root")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    bf16, fp32 = _resolved("gemma4-31b-tp-2.yaml"), _resolved("gemma4-31b-fp32.yaml")
    assert {key: (bf16.get(key), fp32.get(key)) for key in bf16.keys() | fp32.keys()
            if bf16.get(key) != fp32.get(key)} == {"dtype": (None, "float32"), "disable_log_stats": (False, None)}
    assert (fp32["tp_size"], fp32["attention_backend"], fp32["model_name"]) == (2, "TRITON_ATTN", "gemma-4-31B-it")
    engine = local_engine_kwargs(**local_engine_settings(OmegaConf.load(MODEL_DIR / "gemma4-31b-fp32.yaml"),
                                                         engine_fields=()))
    assert (engine["dtype"], engine["attention_config"]) == ("float32", {"backend": "TRITON_ATTN"})
    assert engine["tensor_parallel_size"] == 2 and engine["model"] == "/exported/root/gemma-4-31B-it"


@pytest.mark.parametrize("model_config", LOCAL_MODEL_CONFIGS, ids=LOCAL_MODEL_CONFIGS)
def test_no_config_serves_float32_on_a_backend_without_fp32_and_an_unset_dtype_is_not_passed(model_config, monkeypatch):
    """FLASH_ATTN and FLASHINFER serve only fp16 and bf16."""
    from ruleforge.judges.backend import FP32_ATTENTION_BACKENDS, canonical_serving_dtype, local_engine_settings
    from ruleforge.judges.models import local_engine_kwargs

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/exported/root")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    cfg = OmegaConf.load(MODEL_DIR / model_config)
    dtype = canonical_serving_dtype(cfg.get("dtype"))
    backend = cfg.get("attention_backend")
    assert dtype != "float32" or backend in FP32_ATTENTION_BACKENDS, (model_config, backend)
    assert not (dtype == "float32" and backend in ("FLASH_ATTN", "FLASHINFER"))
    engine = local_engine_kwargs(**local_engine_settings(cfg, engine_fields=()))
    assert engine.get("dtype") == dtype and ("dtype" in engine) == (dtype is not None)


def test_the_fp32_engine_resolves_the_existing_qwen3_8b_truncation_for_the_base_and_a_merged_dir(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ruleforge.evaluate.campaign import resolve_sampling_truncation
    from tests.test_sampling_weights import QWEN3_TRUNCATION, write_base, write_merged

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    write_base(tmp_path / "models")
    merged = write_merged(tmp_path / "grpo")
    recorded = json.dumps(QWEN3_TRUNCATION, sort_keys=True)
    policy = SimpleNamespace(truncation="model_default")
    for name in ("qwen3-8b.yaml", "qwen3-8b-fp32.yaml"):
        cfg = OmegaConf.load(MODEL_DIR / name)
        assert json.dumps(resolve_sampling_truncation(policy, "local", cfg), sort_keys=True) == recorded, name
        cfg.model_path = str(merged)
        assert json.dumps(resolve_sampling_truncation(policy, "local", cfg), sort_keys=True) == recorded, name


def _scan_text(tmp_path: Path, body: str) -> list[tuple[str, str]]:
    yaml_path = tmp_path / "synthetic.yaml"
    yaml_path.write_text(body, encoding="utf-8")
    return _scan_config_for_absolute_paths(yaml_path)


def test_a_bare_cluster_path_is_flagged(tmp_path):
    assert _scan_text(tmp_path, 'model_path: "/scratch/shared/x"\n') == [
        ("model_path", "/scratch/shared/x")
    ]


def test_an_env_interpolation_with_a_suffix_is_accepted(tmp_path):
    body = 'model_path: "${oc.env:RULEFORGE_UNSET_MODEL_ROOT}/gemma-4-31B-it"\n'
    assert _scan_text(tmp_path, body) == []


def test_a_bare_env_interpolation_is_accepted(tmp_path):
    assert _scan_text(tmp_path, 'model_path: "${oc.env:RULEFORGE_UNSET_MODEL_PATH}"\n') == []


def test_an_absolute_env_default_is_flagged(tmp_path):
    body = 'model_path: "${oc.env:RULEFORGE_UNSET_ROOT,/scratch/shared/x}"\n'
    assert _scan_text(tmp_path, body) == [
        ("model_path", "${oc.env:RULEFORGE_UNSET_ROOT,/scratch/shared/x}")
    ]


def test_a_relative_env_default_is_accepted(tmp_path):
    body = 'model_path: "${oc.env:RULEFORGE_UNSET_ROOT,../../models}/qwen"\n'
    assert _scan_text(tmp_path, body) == []


def test_a_path_named_only_in_a_comment_is_ignored(tmp_path):
    body = "# base config points at /scratch/shared/x\nmodel_path: null\n"
    assert _scan_text(tmp_path, body) == []


def test_relative_paths_and_non_strings_are_ignored(tmp_path):
    body = (
        'initial_rule_set_path: "../../data/rules/modelspec.jsonl"\n'
        "tp_size: 2\n"
        "enable_thinking: true\n"
        "reasoning_parser: null\n"
    )
    assert _scan_text(tmp_path, body) == []


def test_a_url_value_is_not_mistaken_for_a_path(tmp_path):
    assert _scan_text(tmp_path, 'base_url: "https://router.example/api/v1"\n') == []


def test_nested_and_listed_values_are_scanned_and_located(tmp_path):
    body = (
        "lora:\n"
        '  adapter_path: "/home/user/models/adapter"\n'
        "extra_args:\n"
        '  - "--download-dir=relative/cache"\n'
        '  - "/home/user/models/second"\n'
    )
    assert _scan_text(tmp_path, body) == [
        ("lora.adapter_path", "/home/user/models/adapter"),
        ("extra_args[1]", "/home/user/models/second"),
    ]


LAUNCHER_DIR = REPO_ROOT / "configs" / "hydra" / "launcher"
LAUNCHERS = sorted(path.stem for path in LAUNCHER_DIR.glob("*.yaml"))


def _launcher(launcher: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"hydra/launcher={launcher}", "hydra.sweep.dir=/sweep"],
                      return_hydra_config=True)
    return cfg.hydra.launcher


def _launcher_setup(launcher: str) -> list:
    return OmegaConf.to_container(_launcher(launcher).setup, resolve=True)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_every_launcher_caps_compile_jobs(launcher, monkeypatch):
    """A job's first vLLM 0.28 engine start JIT-compiles with one compiler per core unless MAX_JOBS caps it."""
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.delenv("MAX_JOBS", raising=False)
    assert "export MAX_JOBS=8" in _launcher_setup(launcher)
    monkeypatch.setenv("MAX_JOBS", "4")
    assert "export MAX_JOBS=4" in _launcher_setup(launcher)


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_every_launcher_pins_the_precision_settings_to_the_library_defaults(launcher, monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.setenv("VLLM_FLOAT32_MATMUL_PRECISION", "high")
    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    setup = _launcher_setup(launcher)
    assert "export VLLM_FLOAT32_MATMUL_PRECISION=highest" in setup
    assert "unset TRITON_F32_DEFAULT" in setup
    assert not any(line.startswith("export TRITON_F32_DEFAULT") for line in setup)
    assert [line for line in setup if "VLLM_BATCH_INVARIANT" in line] == ["export VLLM_BATCH_INVARIANT=0"]
    # vLLM 0.28 reads no VLLM_ATTENTION_BACKEND: a model config's attention_backend picks the kernel
    assert not any("VLLM_ATTENTION_BACKEND" in line for line in setup)


def test_every_launcher_a_job_is_submitted_with_is_checked_here():
    from ruleforge.launch.environment import LAUNCHER_CONFIG

    assert [path for top in ("envs", "tools") for path in (REPO_ROOT / top).rglob("*.sh")
            if "hydra/launcher=" in path.read_text()] == []
    assert (REPO_ROOT / LAUNCHER_CONFIG).parent == LAUNCHER_DIR and Path(LAUNCHER_CONFIG).stem in LAUNCHERS


#: The launchers that name no cluster: the submitting shell's RULEFORGE_SLURM_PARTITION and RULEFORGE_SLURM_ACCOUNT do.
GENERIC_LAUNCHERS = ("submitit_slurm_gpu",)


def _sbatch_script(launcher: str, monkeypatch) -> str:
    import submitit
    from hydra.utils import instantiate
    from hydra_plugins.hydra_submitit_launcher.config import BaseQueueConf

    # submitit refuses to build an executor where no srun is on PATH (CI's runners): say there is one.
    monkeypatch.setattr(submitit.slurm.slurm.SlurmExecutor, "affinity", classmethod(lambda cls: 2))
    params = dict(instantiate(_launcher(launcher)).params)
    executor = submitit.AutoExecutor(folder=params.pop("submitit_folder"), cluster="slurm",
                                     slurm_max_num_timeout=params.pop("max_num_timeout"))
    base = set(OmegaConf.structured(BaseQueueConf).keys())
    executor.update_parameters(**{key if key in base else f"slurm_{key}": value for key, value in params.items()})
    return executor._executor._make_submission_file_text("COMMAND", "uid")


@pytest.mark.allow_slurm_executor  # renders the sbatch script; nothing is submitted (tests/support/slurm_guard.py)
@pytest.mark.parametrize("launcher", GENERIC_LAUNCHERS)
def test_a_generic_launcher_takes_its_cluster_from_the_environment(launcher, monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.delenv("RULEFORGE_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("RULEFORGE_SLURM_ACCOUNT", raising=False)
    with pytest.raises(InterpolationResolutionError, match="RULEFORGE_SLURM_PARTITION"):
        _sbatch_script(launcher, monkeypatch)
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu")
    lines = _sbatch_script(launcher, monkeypatch).splitlines()
    assert "#SBATCH --partition=gpu" in lines and "#SBATCH --no-requeue" in lines
    assert not any(line.startswith(("#SBATCH --account", "#SBATCH --mail")) for line in lines)
    monkeypatch.setenv("RULEFORGE_SLURM_ACCOUNT", "myaccount")
    assert "#SBATCH --account=myaccount" in _sbatch_script(launcher, monkeypatch).splitlines()


GRPO_CONFIGS = sorted(
    f"training/grpo/{path.stem}" for path in (REPO_ROOT / "configs" / "training" / "grpo").glob("*.yaml")
    if path.stem != "base"
)


def _grpo_config(name: str, monkeypatch, *overrides):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", "/artifacts")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return compose(config_name=name, overrides=["run_name=grpo-test", *overrides], return_hydra_config=True)


def test_the_grpo_config_discovery_is_not_vacuous():
    assert GRPO_CONFIGS == ["training/grpo/gemma4_31b", "training/grpo/qwen3_8b", "training/grpo/qwen3_8b_mix"]


GRPO_RECIPES = {
    "training/grpo/qwen3_8b": {"batch": (8, 2, 8), "anchor_accum": 4, "tp": 1, "utilization": 0.30,
                                 "plain": ["model.norm"], "wandb_group": "grpo", "max_steps": 600},
    "training/grpo/gemma4_31b": {"batch": (8, 1, 8), "anchor_accum": 2, "tp": 4, "utilization": 0.40,
                                   "plain": ["model.language_model.norm"], "wandb_group": "gemma-grpo",
                                   "max_steps": 600},
    # the paper's Qwen3-8B run: 600 steps configured, 500 trained, which its rerun trains
    "training/grpo/qwen3_8b_mix": {"batch": (8, 2, 4), "anchor_accum": 2, "tp": 1, "utilization": 0.30,
                                     "plain": ["model.norm"], "wandb_group": "qwen-grpo-mix", "max_steps": 500},
}


@pytest.mark.parametrize("name", GRPO_CONFIGS)
def test_a_grpo_config_composes_with_its_inputs_left_to_the_run(name, monkeypatch):
    cfg = _grpo_config(name, monkeypatch)
    for section, key in (("data", "train"), ("data", "replay"), ("data", "anchor_condition"), ("reward", "truncated")):
        assert OmegaConf.is_missing(cfg[section], key), (section, key)
    assert cfg.output_dir == "/artifacts/training/grpo-test"
    assert OmegaConf.to_container(cfg.hydra.run, resolve=True)["dir"].startswith("/artifacts/training/grpo-test/hydra/")
    assert cfg.model.init_adapter is None
    assert _grpo_config(name, monkeypatch, "model.init_adapter=/a/checkpoint-200").model.init_adapter == (
        "/a/checkpoint-200")
    # the pinned ruleset, from the program's working directory (the checkout's top)
    rules = (REPO_ROOT / cfg.data.rules).resolve()
    assert rules == REPO_ROOT / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"
    import hashlib

    assert hashlib.sha256(rules.read_bytes()).hexdigest() == cfg.data.rules_sha256
    grpo = cfg.grpo
    recipe = GRPO_RECIPES[name]
    assert (grpo.learning_rate, grpo.warmup_steps, grpo.max_steps, grpo.save_steps, grpo.save_total_limit) == (
        2e-5, 10, recipe["max_steps"], 100, None)
    # a constant rate after the warmup: a run cut short is the longer run's first steps
    assert grpo.lr_scheduler_type == "constant_with_warmup"
    assert (grpo.beta, grpo.loss_type, grpo.scale_rewards, grpo.multi_objective_aggregation) == (
        0.0, "dapo", "batch", "sum_then_normalize")
    assert (grpo.num_generations, grpo.per_device_train_batch_size, grpo.gradient_accumulation_steps) == recipe["batch"]
    assert (grpo.vllm_tensor_parallel_size, grpo.vllm_gpu_memory_utilization) == (recipe["tp"], recipe["utilization"])
    assert grpo.steps_per_generation == grpo.gradient_accumulation_steps
    # the trainer shuffles the rows: their prefix balance (grpo_data.interleave) does not reach a run
    assert grpo.shuffle_dataset is True
    assert (grpo.epsilon, grpo.epsilon_high, grpo.delta) == (0.2, None, None)
    assert (grpo.top_entropy_quantile, grpo.off_policy_mask_threshold) == (1.0, None)
    assert cfg.sync_probe.plain_modules == recipe["plain"]
    assert (grpo.vllm_importance_sampling_mode, grpo.vllm_importance_sampling_clip_max) == ("token_truncate", 2.0)
    assert (grpo.max_completion_length, cfg.vllm_length_margin, cfg.data.prompt_token_cap) == (6144, 5120, 5000)
    assert (cfg.reward.lambda_int, cfg.reward.lambda_comp) == (0.3, 0.1)
    assert (cfg.anchor.loss_weight, cfg.anchor.mix_every_n_steps, cfg.anchor.anchor_accum) == (
        1.0, 1, recipe["anchor_accum"])
    for derived in ("temperature", "top_k", "top_p", "min_p", "reward_weights", "model_init_kwargs", "run_name"):
        assert derived not in grpo, derived


@pytest.mark.parametrize("name", GRPO_CONFIGS)
def test_a_grpo_config_reports_to_wandb_in_its_family(name, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    cfg = _grpo_config(name, monkeypatch)
    assert list(cfg.grpo.report_to) == ["wandb"] and cfg.grpo.log_completions is True
    assert OmegaConf.to_container(cfg.wandb, resolve=True) == {
        "project": "ruleforge", "entity": None, "group": GRPO_RECIPES[name]["wandb_group"], "log_completions_table": False}
    monkeypatch.setenv("WANDB_ENTITY", "a-team")
    assert _grpo_config(name, monkeypatch).wandb.entity == "a-team"


def test_the_qwen3_8b_grpo_config_trains_the_local_qwen3_8b(monkeypatch):
    cfg = _grpo_config("training/grpo/qwen3_8b", monkeypatch, "data.train=t", "data.replay=r",
                       "data.anchor_condition=c", "reward.truncated=zero")
    assert cfg.model.path == "/models/Qwen3-8B" and cfg.model.attn_implementation == "sdpa"
    assert not OmegaConf.is_missing(cfg.data, "train")
    # Qwen3-8B's run keeps the config it trained under: no Gemma-only key
    assert "master_dtype" not in cfg.model and "fsdp" not in cfg.grpo and "fsdp_config" not in cfg.grpo
    assert "favoured_no_rate" not in cfg.reward  # nor the favoured-no yes rate
    # base.yaml's warm-start key is there, null: no part of its config digest (trainer.resume_free)
    assert cfg.model.init_adapter is None


def test_the_gemma4_31b_grpo_config_shards_an_fp32_master_under_colocated_tensor_parallel_engines(monkeypatch):
    from ruleforge.launch.train import per_step
    from transformers.models.gemma4 import modeling_gemma4

    cfg = _grpo_config("training/grpo/gemma4_31b", monkeypatch, "data.train=t", "data.replay=r",
                       "data.anchor_condition=c", "reward.truncated=zero")
    assert cfg.model.path == "/models/gemma-4-31B-it" and cfg.model.master_dtype is None
    fsdp = OmegaConf.to_container(cfg.grpo.fsdp_config)
    assert cfg.grpo.fsdp is True and fsdp == {
        "version": 2, "transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer"], "reshard_after_forward": True,
        "cpu_ram_efficient_loading": False, "state_dict_type": "FULL_STATE_DICT"}
    assert hasattr(modeling_gemma4, fsdp["transformer_layer_cls_to_wrap"][0])
    assert per_step(cfg, 4) == {"completions": 32, "prompts": 4, "replay_rows": 8}
    # TP splits Gemma 4's 32 heads and its 4 global KV heads (at 8, vLLM would replicate the KV heads)
    assert 32 % cfg.grpo.vllm_tensor_parallel_size == 0 and 4 % cfg.grpo.vllm_tensor_parallel_size == 0
    assert (cfg.grpo.max_completion_length, cfg.data.prompt_token_cap, cfg.vllm_length_margin) == (6144, 5000, 5120)
    assert cfg.reward.favoured_no_rate is True


def test_the_qwen3_8b_mix_recipe_is_qwen3_8b_s_config_with_gemma_s_recipe_on_4_ranks(monkeypatch):
    from ruleforge.launch.train import per_step
    from tests.support.diffs import diff

    overrides = ("data.train=t", "data.replay=r", "data.anchor_condition=c", "reward.truncated=zero")
    mix = _grpo_config("training/grpo/qwen3_8b_mix", monkeypatch, *overrides)
    qwen = _grpo_config("training/grpo/qwen3_8b", monkeypatch, *overrides)
    gemma = _grpo_config("training/grpo/gemma4_31b", monkeypatch, *overrides)
    differences = diff(*(OmegaConf.to_container(cfg, resolve=False) for cfg in (qwen, mix)))
    assert {key: value for key, value in differences.items()
            if key != "hydra.job.config_name" and not key.startswith("hydra.runtime")} == {
        "reward.favoured_no_rate": (None, True), "anchor.anchor_accum": (4, 2),
        "grpo.gradient_accumulation_steps": (8, 4), "wandb.group": ("grpo", "qwen-grpo-mix"), "grpo.max_steps": (600, 500)}
    assert per_step(mix, 4) == {"completions": 32, "prompts": 4, "replay_rows": 8}
    assert per_step(qwen, 2) == per_step(mix, 4)
    for key in ("reward.lambda_int", "reward.lambda_comp", "reward.favoured_no_rate", "anchor.loss_weight", "anchor.kind", "grpo.learning_rate",
                "grpo.save_steps", "grpo.num_generations", "grpo.max_completion_length", "grpo.beta",
                "grpo.loss_type", "grpo.max_grad_norm"):
        assert OmegaConf.select(mix, key) == OmegaConf.select(gemma, key), key
