from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ruleforge.judges.lora import (
    DEFAULT_MAX_LORA_RANK,
    resolve_lora_adapter_metadata,
    resolve_lora_max_rank,
    stable_adapter_int_id,
)
from ruleforge.judges.token_usage import TokenUsage, TokenUsageTracker


@pytest.fixture
def adapter(tmp_path):
    """An adapter laid out as training writes it: <run>/<stage>/<dir>/<file>."""
    directory = tmp_path / "run_12162" / "checkpoints" / "final_adapter"
    directory.mkdir(parents=True)
    path = directory / "adapter_model.safetensors"
    path.write_bytes(b"weights")
    return path


def test_metadata_comes_from_the_training_layout(adapter):
    directory, name, run_id = resolve_lora_adapter_metadata(adapter)
    assert directory == adapter.parent
    assert name == "final_adapter"
    assert run_id == "run_12162"


def test_a_missing_adapter_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        resolve_lora_adapter_metadata(tmp_path / "absent.safetensors")


def test_pointing_at_the_directory_is_an_error(adapter):
    with pytest.raises(ValueError, match="must point to an adapter file"):
        resolve_lora_adapter_metadata(adapter.parent)


def test_a_relative_path_yields_the_resolved_run_id_not_an_empty_one(
    tmp_path, monkeypatch
):
    ids = {}
    for run in ("runA", "runB"):
        directory = tmp_path / run / "stage" / "final_adapter"
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_bytes(b"weights")
        monkeypatch.chdir(tmp_path / run / "stage")
        adapter_dir, name, run_id = resolve_lora_adapter_metadata(
            type(tmp_path)("final_adapter/adapter_model.safetensors")
        )
        assert adapter_dir.is_absolute()
        assert name == "final_adapter"
        ids[run] = run_id
    assert ids == {"runA": "runA", "runB": "runB"}


def test_a_too_shallow_layout_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(b"weights")

    shallow = type(path)("/a/adapter_model.safetensors")
    monkeypatch.setattr(
        type(path), "resolve", lambda self: shallow, raising=False
    )
    with pytest.raises(ValueError, match="Unable to derive adapter ID"):
        resolve_lora_adapter_metadata(path)


def test_build_lora_request_carries_the_resolved_rank_and_id(adapter, monkeypatch):
    (adapter.parent / "adapter_config.json").write_text(json.dumps({"r": 32}))

    class _LoRARequest:
        def __init__(self, name, int_id, path):
            self.name, self.int_id, self.path = name, int_id, path

    fake = types.ModuleType("vllm.lora.request")
    fake.LoRARequest = _LoRARequest
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake)

    from ruleforge.judges.lora import build_lora_request, lora_serving

    resolved_dir = adapter.parent.resolve()
    request, metadata = build_lora_request(adapter)
    assert request.name == "final_adapter"
    assert request.int_id == stable_adapter_int_id(str(resolved_dir))
    assert request.path == str(resolved_dir)
    assert metadata == {
        "path": str(resolved_dir),
        "name": "final_adapter",
        "id": "run_12162",
        "vllm_int_id": stable_adapter_int_id(str(resolved_dir)),
        "max_lora_rank": 32,
    }
    served = lora_serving(adapter)
    assert isinstance(served.request, _LoRARequest)
    assert (served.request.int_id, served.request.path) == (request.int_id, request.path)
    assert served.metadata == metadata
    assert served.engine_settings() == {"enable_lora": True, "max_lora_rank": 32}


def test_same_named_checkpoints_of_different_runs_never_share_a_registry_key(
    tmp_path,
):
    """The real GRPO store is <group>/<run>/<checkpoint>/<file>."""
    keys = {}
    for run in ("run_A", "run_B"):
        directory = tmp_path / "grpo" / run / "checkpoint-125"
        directory.mkdir(parents=True)
        path = directory / "adapter_model.safetensors"
        path.write_bytes(b"weights")
        adapter_dir, name, run_id = resolve_lora_adapter_metadata(path)
        assert name == "checkpoint-125"
        assert run_id == "grpo"
        keys[run] = stable_adapter_int_id(str(adapter_dir))
    assert keys["run_A"] != keys["run_B"]


def test_the_name_keyed_equality_residue_is_the_one_documented():
    module = pytest.importorskip("vllm.lora.request", reason="needs the vllm group")
    a = module.LoRARequest("checkpoint-125", 111, "/store/grpo/run_A/checkpoint-125")
    b = module.LoRARequest("checkpoint-125", 222, "/store/grpo/run_B/checkpoint-125")
    assert a == b
    assert len({a, b}) == 1


@pytest.mark.parametrize(
    "rank, expected",
    [(32, 32), (64, 64), (8, DEFAULT_MAX_LORA_RANK), (16, 16)],
)
def test_the_ceiling_is_the_adapter_rank_or_the_floor(adapter, rank, expected):
    (adapter.parent / "adapter_config.json").write_text(json.dumps({"r": rank}))
    assert resolve_lora_max_rank(adapter) == expected


@pytest.mark.parametrize(
    "content",
    ['{"r": "not a number"}', "{}", "not json at all", '{"r": null}'],
)
def test_an_unusable_config_falls_back_to_the_floor(adapter, content, caplog):
    (adapter.parent / "adapter_config.json").write_text(content)
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.lora"):
        assert resolve_lora_max_rank(adapter) == DEFAULT_MAX_LORA_RANK
    assert "falling back" in caplog.text


def test_an_absent_config_falls_back_to_the_floor(adapter):
    assert resolve_lora_max_rank(adapter) == DEFAULT_MAX_LORA_RANK


def test_a_custom_floor_is_respected(adapter):
    (adapter.parent / "adapter_config.json").write_text(json.dumps({"r": 4}))
    assert resolve_lora_max_rank(adapter, floor=8) == 8


def test_the_integer_id_is_stable_across_processes():
    assert stable_adapter_int_id("run_12162") == stable_adapter_int_id("run_12162")


def test_different_runs_get_different_ids():
    assert stable_adapter_int_id("run_a") != stable_adapter_int_id("run_b")


def test_ids_are_positive_and_fit_in_31_bits():
    for name in ("run_1", "x", "", "a" * 200, "unicode-ünïcode"):
        value = stable_adapter_int_id(name)
        assert 0 < value < 2**31


class _Engine:
    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


class _Tokenizer:
    """Counts one token per whitespace-separated word."""

    def __init__(self, *, callable_api=True, batched=False):
        self._callable = callable_api
        self._batched = batched

    def __call__(self, text, **_kwargs):
        if not self._callable:
            raise TypeError("not callable")
        ids = list(range(len(text.split())))
        return {"input_ids": [ids] if self._batched else ids}

    def encode(self, text, **_kwargs):
        return list(range(len(text.split())))

    def apply_chat_template(self, chat, **_kwargs):
        return " ".join(turn["content"] for turn in chat if turn["content"])


def _production_files(repo_root):
    return [
        path
        for tree in ("src", "scripts")
        for path in (repo_root / tree).rglob("*.py")
    ]


def test_usage_reads_an_object_response():
    usage = TokenUsage.from_response(
        _Engine(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        10,
        5,
        15,
    )


def test_usage_reads_a_mapping_response():
    usage = TokenUsage.from_response(
        {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    )
    assert usage.total_tokens == 7


def test_a_missing_total_is_derived():
    # A response reporting input and output and no total, as Anthropic's does.
    usage = TokenUsage.from_response({"prompt_tokens": 3, "completion_tokens": 4})
    assert usage.total_tokens == 7


def test_absent_usage_counts_as_zero():
    assert TokenUsage.from_response(None) == TokenUsage()


def test_non_numeric_fields_count_as_zero():
    assert TokenUsage.from_response({"prompt_tokens": "many"}).prompt_tokens == 0


def test_the_tracker_accumulates_and_counts_calls():
    tracker = TokenUsageTracker()
    tracker.add(TokenUsage(1, 2, 3))
    tracker.add(TokenUsage(10, 20, 30))
    assert (tracker.prompt_tokens, tracker.completion_tokens, tracker.total_tokens, tracker.calls) == (11, 22, 33, 2)


def test_a_fresh_tracker_is_empty():
    tracker = TokenUsageTracker()
    assert (tracker.prompt_tokens, tracker.completion_tokens, tracker.total_tokens, tracker.calls) == (0, 0, 0, 0)


def _local_cfg(**overrides):
    from omegaconf import OmegaConf

    base = {
        "model_type": "local",
        "model_name": "qwen3-8b",
        "model_path": "/models/Qwen3-8B",
        "tp_size": 1,
        "reasoning_parser": None,
        "speculative_method": None,
        "num_speculative_tokens": None,
        "max_model_len": None,
        "attention_backend": None,
        "enable_thinking": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


VLLM_FIELDS = frozenset({"model", "max_model_len", "reasoning_parser", "max_num_seqs", "disable_log_stats",
                         "language_model_only", "attention_backend", "seed", "dtype", "enforce_eager"})


@pytest.fixture
def captured_local_init(monkeypatch):
    """Stub ``init_local_model``, returning the kwargs it is called with."""
    import ruleforge.judges.models as models_module

    seen = {}

    def _init(**kwargs):
        seen.update(kwargs)
        return object(), _Tokenizer()

    monkeypatch.setattr(models_module, "init_local_model", _init)
    monkeypatch.setattr(models_module, "call_local_model", lambda *a, **k: [[""]])
    monkeypatch.setattr(models_module, "engine_argument_names", lambda: VLLM_FIELDS)
    monkeypatch.setattr("ruleforge.judges.providers.require", lambda purpose, *modules: None)  # vLLM stubbed here
    return seen


def test_a_local_backend_counts_a_prompt_s_tokens_as_it_sends_it(monkeypatch):
    import ruleforge.judges.models as models_module
    from ruleforge.judges.backend import ModelBackend

    class Counting(_Tokenizer):
        def apply_chat_template(self, chat, **kwargs):
            return " ".join(turn["content"] for turn in chat)

        def encode(self, text, **kwargs):
            return text.split()

    engine = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=262144, dtype="bfloat16"))
    monkeypatch.setattr(models_module, "init_local_model", lambda **kwargs: (engine, Counting()))
    monkeypatch.setattr(models_module, "engine_argument_names", lambda: VLLM_FIELDS)
    monkeypatch.setattr("ruleforge.judges.providers.require", lambda purpose, *modules: None)
    backend = ModelBackend(_local_cfg())
    assert backend.prompt_token_count({"system_prompt": "one two", "user_prompt": "three"}) == 3
    assert backend.prompt_token_limit == 262144


def test_an_engine_that_resolved_no_limit_reports_none():
    from ruleforge.judges.engine_state import engine_max_model_len

    assert engine_max_model_len(object()) is None


def test_a_backend_without_an_adapter_does_not_enable_lora(captured_local_init):
    from ruleforge.judges.backend import ModelBackend

    backend = ModelBackend(_local_cfg())
    assert backend.adapter_path is None
    assert backend.adapter_metadata is None
    assert captured_local_init["enable_lora"] is False
    assert captured_local_init["max_lora_rank"] is None


def test_an_adapter_raises_the_rank_ceiling_before_the_engine_is_built(
    adapter, captured_local_init, monkeypatch
):
    (adapter.parent / "adapter_config.json").write_text(json.dumps({"r": 32}))
    _fake_lora_request(monkeypatch)

    from ruleforge.judges.backend import ModelBackend

    backend = ModelBackend(_local_cfg(adapter_path=str(adapter)))
    assert backend.adapter_path == adapter
    assert backend.adapter_metadata["name"] == "final_adapter"
    assert captured_local_init["enable_lora"] is True
    assert captured_local_init["max_lora_rank"] == 32


def test_an_api_backend_records_the_adapter_path_but_never_serves_it(monkeypatch):
    from omegaconf import OmegaConf

    import ruleforge.judges.models as models_module

    monkeypatch.setattr(models_module, "init_openai_client", lambda: object())
    from ruleforge.judges.backend import ModelBackend

    backend = ModelBackend(
        OmegaConf.create(
            {"model_type": "openai", "model_name": "gpt-5.6-luna", "adapter_path": "/x/y/z/a.safetensors"}
        )
    )
    assert str(backend.adapter_path).endswith("a.safetensors")
    assert backend.adapter_metadata is None


def test_language_model_only_reaches_the_backend_init(captured_local_init):
    from ruleforge.judges.backend import ModelBackend

    ModelBackend(_local_cfg())
    assert captured_local_init["language_model_only"] is False
    ModelBackend(_local_cfg(language_model_only=True))
    assert captured_local_init["language_model_only"] is True


@pytest.mark.parametrize("flag", [False, True])
def test_language_model_only_reaches_vllm_only_when_set(monkeypatch, flag):
    import sys
    import types

    import ruleforge.judges.models as models_module

    seen = {}

    class _LLM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_AutoTokenizer))
    models_module.init_local_model(model_path="/models/x", tp_size=1, language_model_only=flag)
    if flag:
        assert seen["language_model_only"] is True
    else:
        assert "language_model_only" not in seen


def test_a_local_gpt_oss_backend_loads_the_harmony_vocab_before_its_engine(captured_local_init, monkeypatch):
    import ruleforge.judges.models as models_module
    from ruleforge.judges.backend import ModelBackend

    events = []
    engine = models_module.init_local_model
    monkeypatch.setattr(models_module, "harmony_encoding", lambda: events.append("vocab"))
    monkeypatch.setattr(models_module, "init_local_model", lambda **kw: events.append("engine") or engine(**kw))
    ModelBackend(_local_cfg(strip_thinking_model="qwen3"))
    assert events == ["engine"]
    events.clear()
    ModelBackend(_local_cfg(model_name="gpt-oss-120b", strip_thinking_model="openai_gptoss"))
    assert events == ["vocab", "engine"]

    def unreadable():
        raise RuntimeError("failed to download or load vocab file")

    events.clear()
    monkeypatch.setattr(models_module, "harmony_encoding", unreadable)
    with pytest.raises(RuntimeError, match="vocab file"):
        ModelBackend(_local_cfg(model_name="gpt-oss-120b", strip_thinking_model="openai_gptoss"))
    assert events == []


@pytest.mark.parametrize("extra", [{}, {"model_name": "gpt-oss-120b", "strip_thinking_model": "openai_gptoss"}],
                         ids=["local", "gpt-oss"])
def test_a_local_judge_without_vllm_is_refused_in_one_line_before_anything_is_loaded(monkeypatch, extra):
    import ruleforge.judges.lora as lora_module
    import ruleforge.judges.models as models_module
    from ruleforge.core.optional import MissingDependency
    from ruleforge.judges.backend import ModelBackend

    def reached(*args, **kwargs):
        raise AssertionError("a part of the vLLM stack was reached")

    monkeypatch.setitem(sys.modules, "vllm", None)
    for name in ("harmony_encoding", "engine_argument_names", "init_local_model"):
        monkeypatch.setattr(models_module, name, reached)
    monkeypatch.setattr(lora_module, "build_lora_request", reached)
    with pytest.raises(MissingDependency) as refused:
        ModelBackend(_local_cfg(adapter_path="/adapters/a/adapter_model.safetensors", **extra))
    assert str(refused.value) == ("serving a local judge needs vllm, which this environment lacks: add --group vllm to "
                                  "uv sync, or pip install -e '.[vllm]' from the checkout")


def test_max_num_seqs_reaches_the_engine_only_when_configured(captured_local_init):
    """gpt-oss-120b on one GPU needs a lower sampler warm-up cap."""
    from ruleforge.judges.backend import ModelBackend

    ModelBackend(_local_cfg())
    assert captured_local_init["max_num_seqs"] is None
    ModelBackend(_local_cfg(max_num_seqs=512))
    assert captured_local_init["max_num_seqs"] == 512


def test_engine_args_reach_the_engine_as_a_plain_dict(captured_local_init):
    from ruleforge.judges.backend import ModelBackend

    ModelBackend(_local_cfg())
    assert captured_local_init["engine_args"] == {}
    ModelBackend(_local_cfg(engine_args={"enable_prefix_caching": False, "gpu_memory_utilization": 0.8}))
    assert captured_local_init["engine_args"] == {"enable_prefix_caching": False, "gpu_memory_utilization": 0.8}
    assert type(captured_local_init["engine_args"]) is dict


def test_engine_args_must_be_a_mapping(captured_local_init):
    from ruleforge.judges.backend import local_engine_settings

    with pytest.raises(ValueError, match="engine_args must be a mapping"):
        local_engine_settings(_local_cfg(engine_args=["enable_prefix_caching=false"]))
    from types import SimpleNamespace

    plain = SimpleNamespace(**dict(_local_cfg()), engine_args={"a": 1})
    assert local_engine_settings(plain)["engine_args"] == {"a": 1}


def test_the_settings_are_exactly_the_plumbed_keys(captured_local_init):
    from ruleforge.judges.backend import LOCAL_ENGINE_SETTING_KEYS, local_engine_settings

    assert tuple(local_engine_settings(_local_cfg())) == LOCAL_ENGINE_SETTING_KEYS


def test_a_model_config_key_naming_an_unplumbed_engine_argument_is_refused(captured_local_init):
    from ruleforge.judges.backend import ModelBackend, local_engine_settings

    with pytest.raises(ValueError, match=r"model config key\(s\) \['enforce_eager'\] name vLLM engine arguments "
                                         r"that no model config key passes to vLLM.*under engine_args$"):
        ModelBackend(_local_cfg(enforce_eager=True))
    plumbed = _local_cfg(max_model_len=4096, reasoning_parser="qwen3", max_num_seqs=8, disable_log_stats=False,
                         language_model_only=True, seed=16, enforce_eager=None)
    assert local_engine_settings(plumbed)["max_model_len"] == 4096
    # the caller's field list instead of vLLM's
    assert local_engine_settings(_local_cfg(enforce_eager=True), engine_fields=())["model_path"] == "/models/Qwen3-8B"
    with pytest.raises(ValueError, match=r"\['cpu_offload_gb'\]"):
        local_engine_settings(_local_cfg(cpu_offload_gb=4), engine_fields=("cpu_offload_gb",))


def test_the_engine_argument_names_are_vllms_engine_args_fields(monkeypatch):
    import dataclasses
    import sys
    import types

    from ruleforge.judges.models import engine_argument_names

    @dataclasses.dataclass
    class EngineArgs:
        model: str = ""
        enforce_eager: bool = False

    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", types.SimpleNamespace(EngineArgs=EngineArgs))
    assert engine_argument_names() == {"model", "enforce_eager"}


def test_every_refused_engine_argument_is_one_vllm_has():
    pytest.importorskip("vllm.engine.arg_utils", reason="needs the vllm group")
    from ruleforge.judges.backend import LOCAL_ENGINE_SETTING_KEYS
    from ruleforge.judges.models import _ENGINE_ARGS_OWNED, ENGINE_ARGS_OUTSIDE_IDENTITY, engine_argument_names

    names = engine_argument_names()
    assert ENGINE_ARGS_OUTSIDE_IDENTITY <= names and _ENGINE_ARGS_OWNED <= names
    assert {"reasoning_parser", "max_model_len", "max_num_seqs", "disable_log_stats"} <= set(LOCAL_ENGINE_SETTING_KEYS) & names


@pytest.mark.parametrize(
    "module",
    [
        "vllm.entrypoints.llm",
        "vllm.entrypoints.openai.cli_args",
        "vllm.tool_parsers",
    ],
)
def test_the_modules_an_engine_imports_at_construction_load(module):
    """The vllm group resolves to versions vLLM can import, checked without a GPU."""
    pytest.importorskip("vllm", reason="needs the vllm group")
    importlib.import_module(module)


def _configured_reasoning_parsers() -> list[str]:
    import yaml

    configs = (Path(__file__).resolve().parents[1] / "configs" / "model").glob("*.yaml")
    names = {(yaml.safe_load(path.read_text()) or {}).get("reasoning_parser") for path in configs}
    return sorted(name for name in names if isinstance(name, str))


@pytest.mark.parametrize("parser", _configured_reasoning_parsers())
def test_every_configured_reasoning_parser_loads(parser):
    pytest.importorskip("vllm", reason="needs the vllm group")
    from vllm.reasoning import ReasoningParserManager

    assert ReasoningParserManager.get_reasoning_parser(parser) is not None


def _init_with_fake_vllm(monkeypatch, **kwargs):
    import sys
    import types

    import ruleforge.judges.models as models_module

    seen = {}

    class _LLM:
        def __init__(self, **engine_kwargs):
            seen.update(engine_kwargs)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_AutoTokenizer))
    models_module.init_local_model(model_path="/models/x", tp_size=1, **kwargs)
    return seen


def test_engine_args_reach_vllm_verbatim(monkeypatch):
    seen = _init_with_fake_vllm(monkeypatch, engine_args={"enable_prefix_caching": False})
    assert seen["enable_prefix_caching"] is False
    assert "enable_prefix_caching" not in _init_with_fake_vllm(monkeypatch)


def test_engine_args_may_not_respell_a_setting_with_its_own_parameter(monkeypatch):
    import ruleforge.judges.models as models_module

    with pytest.raises(ValueError, match=r"\['max_num_seqs'\].*own parameters, not engine_args \(from the model "
                                         r"config's own keys, e.g. tp_size, attention_backend, max_num_seqs, and "
                                         r"the served adapter\)$"):
        models_module.init_local_model(model_path="/m", tp_size=1, engine_args={"max_num_seqs": 8})
    # vLLM's attention_backend respells attention_config's backend, which the model config's key sets
    with pytest.raises(ValueError, match=r"\['attention_backend'\].*own parameters"):
        models_module.local_engine_kwargs(model_path="/m", tp_size=1, engine_args={"attention_backend": "FLASH_ATTN"})
    with pytest.raises(ValueError, match=r"dtype by the model config's dtype key \(condition identity: "
                                         r"serving_dtype\) or a tool's --dtype\)$"):
        models_module.local_engine_kwargs(model_path="/m", tp_size=1, engine_args={"dtype": "float32"})


def test_the_owned_engine_arguments_are_exactly_what_local_engine_kwargs_sets(monkeypatch):
    import ruleforge.judges.models as models_module

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    every = models_module.local_engine_kwargs(
        model_path="/m", tp_size=2, reasoning_parser="qwen3", speculative_method="mtp",
        num_speculative_tokens=3, max_model_len=4096, attention_backend="TRITON_ATTN",
        enable_thinking=True, disable_log_stats=False, enable_lora=True, max_lora_rank=32,
        max_num_seqs=8, language_model_only=True, dtype="float32",
    )
    assert set(every) | {"attention_backend"} == models_module._ENGINE_ARGS_OWNED


@pytest.mark.parametrize("key", ["tokenizer_mode", "quantization", "kv_cache_dtype", "hf_overrides", "seed", "revision"])
def test_engine_args_that_change_the_served_function_are_refused(key):
    from ruleforge.judges.models import ENGINE_ARGS_OUTSIDE_IDENTITY, local_engine_kwargs

    assert key in ENGINE_ARGS_OUTSIDE_IDENTITY and "dtype" not in ENGINE_ARGS_OUTSIDE_IDENTITY
    with pytest.raises(ValueError, match=rf"engine_args \['{key}'\] change the function the judge serves"):
        local_engine_kwargs(model_path="/m", tp_size=1, engine_args={key: "x"})
    with pytest.raises(ValueError, match=r"\['dtype'\].*own parameters"):
        local_engine_kwargs(model_path="/m", tp_size=1, engine_args={"dtype": "float32"})


def test_a_model_config_may_not_name_an_engine_argument_outside_the_identity(captured_local_init):
    from ruleforge.judges.backend import ModelBackend, local_engine_settings

    with pytest.raises(ValueError, match=r"model config key\(s\) \['quantization'\] name vLLM engine "
                                         r"arguments.*\['quantization'\] change the served function, which the "
                                         r"condition identity does not record.*as dtype is through serving_dtype\)$"):
        local_engine_settings(_local_cfg(dtype="float32", quantization="fp8"), engine_fields=())
    assert local_engine_settings(_local_cfg(dtype="float32"), engine_fields=())["dtype"] == "float32"
    ModelBackend(_local_cfg(dtype="float32"))
    assert captured_local_init["dtype"] == "float32"
    assert local_engine_settings(_local_cfg(seed=16, dtype=None))["model_path"] == "/models/Qwen3-8B"


@pytest.mark.parametrize("given, canonical", [
    (None, None), ("float32", "float32"), ("FLOAT", "float32"), ("torch.float32", "float32"),
    ("bfloat16", "bfloat16"), ("torch.bfloat16", "bfloat16"), ("half", "float16"), ("float16", "float16"),
])
def test_a_serving_dtype_is_read_in_its_one_spelling(given, canonical):
    """As vLLM reads it (lower-cased, torch. dropped, its aliases): one condition identity, one engine argument."""
    from ruleforge.judges.backend import canonical_serving_dtype
    from ruleforge.judges.models import local_engine_kwargs

    assert canonical_serving_dtype(given) == canonical
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype=given).get("dtype") == canonical


@pytest.mark.parametrize("given", ["auto", "fp8", "float64", ""])
def test_a_dtype_that_is_not_a_serving_dtype_is_refused(given):
    """auto is the key left unset, its one spelling; the rest are no dtype vLLM serves weights in."""
    from ruleforge.judges.backend import canonical_serving_dtype

    with pytest.raises(ValueError, match=rf"dtype {given!r} is not one of \['float32', 'bfloat16', 'float16'\]"):
        canonical_serving_dtype(given)


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER", "FLASHMLA"])
def test_float32_on_an_attention_backend_that_serves_no_fp32_is_refused(backend):
    """FlashAttention and FlashInfer declare only fp16 and bf16 (vLLM 0.28), as a backend declaring nothing does."""
    from ruleforge.judges.backend import FP32_ATTENTION_BACKENDS, canonical_serving_dtype
    from ruleforge.judges.models import local_engine_kwargs

    message = rf"dtype float32 with attention_backend {backend}: that backend serves only fp16 and bf16"
    with pytest.raises(ValueError, match=message):
        canonical_serving_dtype("float32", backend)
    with pytest.raises(ValueError, match=message):
        local_engine_kwargs(model_path="/m", tp_size=1, dtype="float", attention_backend=backend)
    assert canonical_serving_dtype("bfloat16", backend) == "bfloat16"
    for serves_fp32 in (*sorted(FP32_ATTENTION_BACKENDS), None):
        assert canonical_serving_dtype("float32", serves_fp32) == "float32"
    assert FP32_ATTENTION_BACKENDS == {"TRITON_ATTN", "FLEX_ATTENTION"}


def test_what_a_started_engine_resolved_is_read_off_it():
    from types import SimpleNamespace

    from ruleforge.judges.engine_state import engine_attention_backend, engine_resolution

    def engine(backend, cache_dtype="auto"):
        config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype),
                                 attention_config=SimpleNamespace(backend=backend))
        return SimpleNamespace(model_config=SimpleNamespace(dtype="torch.float32"),
                               llm_engine=SimpleNamespace(vllm_config=config))

    assert engine_resolution(engine(SimpleNamespace(name="TRITON_ATTN"))) == {
        "engine_dtype": "float32", "kv_cache_dtype": "float32", "attention_backend": "TRITON_ATTN",
    }
    assert engine_attention_backend(engine("FLEX_ATTENTION")) == "FLEX_ATTENTION"
    assert engine_attention_backend(engine(None)) is None
    assert engine_resolution(engine(None, cache_dtype="fp8"))["kv_cache_dtype"] == "fp8"
    assert engine_resolution(object()) == {"engine_dtype": None, "kv_cache_dtype": None, "attention_backend": None}


def test_the_training_tools_import_without_the_models_module():
    probe = (
        "import sys; import ruleforge.grpo.merging, ruleforge.grpo.lora_parity, "
        "ruleforge.grpo.report; "
        "print(sorted(name for name in ('ruleforge.judges.models', 'vllm', 'openai_harmony') "
        "if name in sys.modules))"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_a_local_backend_records_its_weights_directory_and_what_its_engine_resolved(captured_local_init, monkeypatch):
    from types import SimpleNamespace

    import ruleforge.judges.models as models_module
    from ruleforge.judges.backend import ModelBackend

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="auto"),
                             attention_config=SimpleNamespace(backend=SimpleNamespace(name="TRITON_ATTN")))
    llm = SimpleNamespace(model_config=SimpleNamespace(dtype="torch.float32"),
                          llm_engine=SimpleNamespace(vllm_config=config))

    def _init(**kwargs):
        captured_local_init.update(kwargs)
        return llm, _Tokenizer()

    monkeypatch.setattr(models_module, "init_local_model", _init)
    backend = ModelBackend(_local_cfg(dtype="float32", attention_backend="TRITON_ATTN"))
    assert backend.model_path == "/models/Qwen3-8B" and backend.llm is llm
    assert backend.engine_resolved == {"engine_dtype": "float32", "kv_cache_dtype": "float32",
                                       "attention_backend": "TRITON_ATTN"}
    assert backend.engine_args["dtype"] == captured_local_init["dtype"] == "float32"


def test_the_qwen3_8b_fp32_config_reaches_the_engine_as_written(captured_local_init, monkeypatch):
    from omegaconf import OmegaConf

    from ruleforge.judges.backend import ModelBackend

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/exported/root")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    configs = Path(__file__).resolve().parents[1] / "configs" / "model"
    fp32 = ModelBackend(OmegaConf.load(configs / "qwen3-8b-fp32.yaml"))
    assert (captured_local_init["dtype"], captured_local_init["attention_backend"]) == ("float32", "TRITON_ATTN")
    bf16 = ModelBackend(OmegaConf.load(configs / "qwen3-8b.yaml"))
    assert captured_local_init["dtype"] is None
    assert {key: value for key, value in fp32.engine_args.items() if bf16.engine_args.get(key) != value} == {
        "dtype": "float32", "attention_config": {"backend": "TRITON_ATTN"},
    }
    assert "dtype" not in bf16.engine_args and set(bf16.engine_args) | {"dtype"} == set(fp32.engine_args)


def test_a_negative_max_model_len_is_refused():
    """-1 lets vLLM fit the window to the KV memory left, which placement changes and no guard compares."""
    from ruleforge.judges.models import local_engine_kwargs

    with pytest.raises(ValueError, match=r"max_model_len=-1: a negative value lets vLLM fit the context window"):
        local_engine_kwargs(model_path="/m", tp_size=1, max_model_len=-1)
    assert local_engine_kwargs(model_path="/m", tp_size=1, max_model_len=4096)["max_model_len"] == 4096
    assert local_engine_kwargs(model_path="/m", tp_size=1)["max_model_len"] is None


def test_an_fp32_engine_is_refused_under_batch_invariance(monkeypatch):
    """VLLM_BATCH_INVARIANT routes every unquantized linear through a TF32 Triton matmul."""
    from ruleforge.judges.models import local_engine_kwargs

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    for dtype in ("float32", "float", "torch.float32", "FLOAT32", "Float"):  # vLLM lower-cases the dtype
        with pytest.raises(ValueError, match="float32 engine with VLLM_BATCH_INVARIANT on"):
            local_engine_kwargs(model_path="/m", tp_size=1, dtype=dtype)
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype="bfloat16")["dtype"] == "bfloat16"
    assert "dtype" not in local_engine_kwargs(model_path="/m", tp_size=1)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype="float32")["dtype"] == "float32"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "yes")
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT='yes' is not an integer"):
        local_engine_kwargs(model_path="/m", tp_size=1, dtype="float32")


@pytest.mark.parametrize("precision", ["high", "medium"])
def test_an_fp32_engine_is_refused_under_a_reduced_gemm_precision(monkeypatch, precision):
    """The worker passes VLLM_FLOAT32_MATMUL_PRECISION to torch.set_float32_matmul_precision."""
    from ruleforge.judges.models import local_engine_kwargs

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setenv("VLLM_FLOAT32_MATMUL_PRECISION", precision)
    with pytest.raises(ValueError, match=rf"float32 engine with VLLM_FLOAT32_MATMUL_PRECISION={precision}: .* "
                                         "Unset it or set it to highest$"):
        local_engine_kwargs(model_path="/m", tp_size=1, dtype="float32")
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype="bfloat16")["dtype"] == "bfloat16"
    monkeypatch.setenv("VLLM_FLOAT32_MATMUL_PRECISION", "highest")
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype="float32")["dtype"] == "float32"
    monkeypatch.delenv("VLLM_FLOAT32_MATMUL_PRECISION")  # unset: vLLM's default, highest
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype="float32")["dtype"] == "float32"


def test_the_precision_settings_read_an_unset_variable_as_its_library_default():
    from ruleforge.judges.backend import UNSET, batch_invariant_enabled, precision_env

    assert precision_env({}) == {
        "VLLM_FLOAT32_MATMUL_PRECISION": "highest", "TRITON_F32_DEFAULT": UNSET, "VLLM_BATCH_INVARIANT": "0",
    }
    assert precision_env({"TRITON_F32_DEFAULT": "tf32x3"})["TRITON_F32_DEFAULT"] == "tf32x3"
    assert batch_invariant_enabled({"VLLM_BATCH_INVARIANT": "1"}) and not batch_invariant_enabled({})


@pytest.mark.parametrize("name, value, recorded", [
    # vLLM reads batch invariance as an integer: any non-zero one is on
    ("VLLM_BATCH_INVARIANT", "01", "1"), ("VLLM_BATCH_INVARIANT", "2", "1"), ("VLLM_BATCH_INVARIANT", " 1 ", "1"),
    ("VLLM_BATCH_INVARIANT", "00", "0"), ("VLLM_BATCH_INVARIANT", "-0", "0"),
    # ... and the GEMM precision as torch applies it, spelled exactly
    ("VLLM_FLOAT32_MATMUL_PRECISION", "high", "high"), ("VLLM_FLOAT32_MATMUL_PRECISION", "medium", "medium"),
    # Triton reads its default case-insensitively, an empty value as unset
    ("TRITON_F32_DEFAULT", "", "unset"), ("TRITON_F32_DEFAULT", "tf32", "tf32"), ("TRITON_F32_DEFAULT", "TF32x3", "tf32x3"),
])
def test_the_precision_settings_are_recorded_as_their_library_reads_them(name, value, recorded):
    from ruleforge.judges.backend import precision_env

    assert precision_env({name: value})[name] == recorded


@pytest.mark.parametrize("value", ["", "yes", "true", "1.0"])
def test_a_batch_invariance_setting_vllm_refuses_is_refused(value):
    from ruleforge.judges.backend import batch_invariant_enabled, precision_env

    for read in (precision_env, batch_invariant_enabled):
        with pytest.raises(ValueError, match=rf"VLLM_BATCH_INVARIANT={value!r} is not an integer"):
            read({"VLLM_BATCH_INVARIANT": value})


@pytest.mark.parametrize("value", ["HIGH", "Medium", "HIGHEST"])
def test_a_mis_cased_gemm_precision_is_refused(value):
    from ruleforge.judges.backend import precision_env

    with pytest.raises(ValueError, match=rf"VLLM_FLOAT32_MATMUL_PRECISION={value!r} is mis-cased: .* spell it exactly"):
        precision_env({"VLLM_FLOAT32_MATMUL_PRECISION": value})


def test_an_exact_highest_clears_the_tf32_start_the_override_gives():
    """Why an exactly spelled GEMM precision is recorded as applied."""
    pytest.importorskip("torch")
    probe = (
        "import torch; torch.set_float32_matmul_precision('highest'); "
        "print(torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32)"
    )
    env = {**os.environ, "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["highest", "False"]


def test_a_setting_vllm_refuses_is_recorded_as_given_where_no_engine_reads_it():
    from ruleforge.judges.backend import precision_env

    refused = {"VLLM_BATCH_INVARIANT": "true", "VLLM_FLOAT32_MATMUL_PRECISION": "fastest"}
    with pytest.raises(ValueError, match=r"VLLM_FLOAT32_MATMUL_PRECISION='fastest' is not one of \['highest'"):
        precision_env({"VLLM_FLOAT32_MATMUL_PRECISION": "fastest"})
    assert precision_env(refused, strict=False) == {**refused, "TRITON_F32_DEFAULT": "unset"}
    mis_cased = {**refused, "VLLM_FLOAT32_MATMUL_PRECISION": "HIGH"}
    assert precision_env(mis_cased, strict=False) == {**mis_cased, "TRITON_F32_DEFAULT": "unset"}
    assert precision_env({"VLLM_BATCH_INVARIANT": "01"}, strict=False)["VLLM_BATCH_INVARIANT"] == "1"


def _fake_lora_request(monkeypatch):
    """build_lora_request without vllm; its metadata carries the rank as the real one does."""
    import ruleforge.judges.lora as lora_module

    monkeypatch.setattr(lora_module, "build_lora_request", lambda path: (
        ("REQUEST", str(path)), {"name": path.parent.name, "max_lora_rank": lora_module.resolve_lora_max_rank(path)}
    ))


def test_lora_serving_enables_lora_exactly_for_an_adapter_at_its_rank(adapter, monkeypatch):
    from ruleforge.judges.lora import LoraServing, lora_serving

    _fake_lora_request(monkeypatch)
    (adapter.parent / "adapter_config.json").write_text(json.dumps({"r": 32}))
    assert lora_serving(None) == LoraServing()
    assert lora_serving(None).engine_settings() == {"enable_lora": False, "max_lora_rank": None}
    served = lora_serving(adapter)
    assert (served.enable_lora, served.max_lora_rank, served.request[0]) == (True, 32, "REQUEST")
    assert served.metadata["max_lora_rank"] == 32
    assert served.engine_settings() == {"enable_lora": True, "max_lora_rank": 32}


def test_an_api_backend_has_no_engine(monkeypatch):
    from omegaconf import OmegaConf

    import ruleforge.judges.models as models_module
    from ruleforge.judges.backend import ModelBackend

    monkeypatch.setattr(models_module, "init_openai_client", lambda: object())
    backend = ModelBackend(OmegaConf.create({"model_type": "openai", "model_name": "gpt-5.6-luna"}))
    assert backend.engine_args is None


def test_the_engine_kwargs_set_only_what_is_asked_for(monkeypatch):
    from ruleforge.judges.models import local_engine_kwargs

    # an fp32 engine below; whatever the shell set
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.delenv("VLLM_FLOAT32_MATMUL_PRECISION", raising=False)

    plain = local_engine_kwargs(model_path="/m", tp_size=1)
    assert plain["attention_config"] is None
    assert not {"max_lora_rank", "max_num_seqs", "language_model_only", "dtype"} & set(plain)
    assert local_engine_kwargs(model_path="/m", tp_size=1, dtype="float32")["dtype"] == "float32"
    spec = local_engine_kwargs(model_path="/m", tp_size=1, speculative_method="mtp", num_speculative_tokens=3)
    assert spec["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 3}
    # no count: vLLM's own default for the method
    assert local_engine_kwargs(model_path="/m", tp_size=1, speculative_method="mtp")["speculative_config"] == {
        "method": "mtp",
    }
    with pytest.raises(ValueError, match=r"\['enable_lora'\].*own parameters"):
        local_engine_kwargs(model_path="/m", tp_size=1, engine_args={"enable_lora": True})


def test_cleanup_shuts_the_engine_core_down(monkeypatch, caplog):
    import time
    from types import SimpleNamespace

    from ruleforge.judges.models import cleanup_vllm

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    calls = []
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: calls.append("core"))))
    cleanup_vllm(llm)
    assert calls == ["core"]
    # an engine without a core is an error: fatal by default, logged from a finally block
    broken = SimpleNamespace(llm_engine=SimpleNamespace())
    with pytest.raises(SystemExit):
        cleanup_vllm(broken)
    with caplog.at_level(logging.ERROR):
        cleanup_vllm(broken, raise_on_error=False)
    assert "Error during vLLM cleanup" in caplog.text
