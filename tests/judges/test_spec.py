from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import (
    MODEL_CONFIG_DIR,
    BatchJudge,
    Capabilities,
    Judge,
    JudgeError,
    JudgeOptions,
    JudgeSpec,
    check,
    resolve,
)
from ruleforge.judges import base as judges_base
from ruleforge.judges import providers
from ruleforge.judges.base import identity_fields
from ruleforge.judges import backend as backend_module
from ruleforge.judges import models
from ruleforge.judges.backend import BUILTIN_MODEL_TYPES, GenerationOutcome, ModelBackend

MODEL_CONFIGS = sorted(path.stem for path in MODEL_CONFIG_DIR.glob("*.yaml"))


@pytest.fixture
def model_root(tmp_path, monkeypatch):
    """A model root holding every committed local config's weights directory, each with a generation config."""
    root = tmp_path / "models"
    for name in MODEL_CONFIGS:
        cfg = OmegaConf.load(MODEL_CONFIG_DIR / f"{name}.yaml")
        if cfg.model_type == "local":
            weights = root / OmegaConf.to_container(cfg)["model_path"].rsplit("/", 1)[-1]
            weights.mkdir(parents=True, exist_ok=True)
            (weights / "generation_config.json").write_text(json.dumps({"top_k": 20, "top_p": 0.95}))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(root))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    return root


@pytest.fixture
def nothing_built(monkeypatch):
    """Every engine and client constructor refuses: what reads a spec must build none of them."""
    def refuse(*args, **kwargs):
        raise AssertionError("an engine or client was built")

    for name in ("init_local_model", "init_openai_client", "init_anthropic_client"):
        monkeypatch.setattr(models, name, refuse)
    monkeypatch.setattr(ModelBackend, "__init__", refuse)


def _legacy_backend(cfg) -> SimpleNamespace:
    """The facts a campaign reads off a constructed backend, as ``ModelBackend`` reads them from its config."""
    seed = providers.PROVIDERS[str(cfg.model_type)].seed_capability(cfg)
    return SimpleNamespace(model_type=str(cfg.model_type), model_name=str(cfg.model_name),
                           max_new_tokens=int(cfg.get("max_new_tokens", 2**16)), seed_capability=seed)


def test_the_model_config_directory_is_not_empty():
    assert {"qwen3-8b", "gpt-5.6-luna-batch", "claude-haiku-5-5-batch"} <= set(MODEL_CONFIGS)


@pytest.mark.parametrize("name", MODEL_CONFIGS)
def test_every_committed_model_config_reads_as_a_spec_with_no_engine_or_client(name, model_root, nothing_built):
    spec = JudgeSpec.from_config(name)
    cfg = OmegaConf.load(MODEL_CONFIG_DIR / f"{name}.yaml")
    assert (spec.model_type, spec.model_name) == (str(cfg.model_type), str(cfg.model_name))
    assert spec == JudgeSpec.from_config(MODEL_CONFIG_DIR / f"{name}.yaml") == JudgeSpec.from_config(cfg)
    identity = spec.identity(verify_weights=True)
    assert identity == providers.backend_identity(_legacy_backend(cfg), cfg)
    capabilities = spec.capabilities()
    assert capabilities.per_request_errors and capabilities.seed == identity.seed_capability
    assert capabilities.spends_money == (spec.model_type != "local")
    # What the request shows of the reasoning, never identity: Luna's summary, Haiku's summarized display.
    shown = {key: cfg.get(key) for key in ("reasoning_summary", "thinking_display") if cfg.get(key)}
    assert spec.request_settings() == shown


def test_a_batch_config_s_identity_needs_no_batch_root(monkeypatch):
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT", raising=False)
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT", raising=False)
    spec = JudgeSpec.from_config("gpt-5.6-luna-batch")
    assert spec.identity().endpoint == "responses" and spec.options.batch_root is None
    written = JudgeSpec.from_config(MODEL_CONFIG_DIR / "gpt-5.6-luna-batch.yaml")
    assert written.identity() == spec.identity()
    assert written.options.written() == spec.options.written() and written.options.batch_root is None


def test_from_config_takes_overrides_and_refuses_what_it_cannot_find(tmp_path, model_root):
    assert JudgeSpec.from_config("qwen3-8b", tp_size=4).options.tp_size == 4
    with pytest.raises(JudgeError, match="no model config named 'nope'"):
        JudgeSpec.from_config("nope")
    with pytest.raises(JudgeError, match="no model config at"):
        JudgeSpec.from_config(tmp_path / "absent.yaml")
    with pytest.raises(JudgeError, match="a model config names its model_type"):
        JudgeSpec.from_config({"model_name": "m"})
    with pytest.raises(JudgeError, match="names its model_name"):
        JudgeSpec.from_config({"model_type": "openai"})
    with pytest.raises(JudgeError, match="a judge is configured by"):
        JudgeSpec.from_config(3)
    with pytest.raises(JudgeError, match="no judge provider for model_type 'telepathy'"):
        JudgeSpec.from_config({"model_type": "telepathy", "model_name": "x"})


@pytest.mark.parametrize("name", ["qwen", "gemma", "gpt-oss"])
def test_a_local_config_naming_no_thinking_stripper_is_refused_when_it_is_read(model_root, name):
    with pytest.raises(JudgeError, match=rf"strip_thinking_model {name!r} names no thinking-token stripper: "
                                         r"one of \['gemma4', 'muse_glimmer', 'olmo3', 'openai_gptoss', 'qwen3'\]"):
        JudgeSpec.from_config("qwen3-8b", strip_thinking_model=name)
    assert JudgeSpec.from_config("qwen3-8b", strip_thinking_model=None).options.get("strip_thinking_model") is None


def test_options_are_read_only_resolved_on_read_and_cut_from_their_parent(monkeypatch):
    monkeypatch.setenv("RULEFORGE_TEST_ROOT", "/r")
    parent = OmegaConf.create({"base": "/b", "judge_model": {"model_type": "fake", "model_name": "fake-1", "seed": 1,
                                                             "path": "${base}/x", "env": "${oc.env:RULEFORGE_TEST_ROOT}",
                                                             "later": "${oc.env:RULEFORGE_UNSET_FOR_TEST}",
                                                             "nested": {"a": [1, 2]}}})
    options = JudgeOptions(parent.judge_model)
    parent.base = "/changed"
    assert (options.path, options["env"], options.nested, options.absent) == ("/b/x", "/r", {"a": [1, 2]}, None)
    assert options.written()["later"] == "${oc.env:RULEFORGE_UNSET_FOR_TEST}" and len(options) == 7
    assert list(options) == ["model_type", "model_name", "seed", "path", "env", "later", "nested"]
    monkeypatch.setenv("RULEFORGE_UNSET_FOR_TEST", "/now")
    assert options.later == "/now"
    with pytest.raises(KeyError):
        options["absent"]
    with pytest.raises(AttributeError):
        options._hidden
    with pytest.raises(AttributeError, match="read-only"):
        options.seed = 2
    assert options.replace(seed=2).seed == 2 and options.seed == 1
    assert options == JudgeOptions(options) and options != JudgeOptions({"seed": 1})
    assert (options == {"seed": 1}) is False and "JudgeOptions(" in repr(options)
    config = options.config()
    config.seed = 3
    assert options.seed == 1


def test_the_typed_constructors_write_the_configs_the_yaml_files_do(tmp_path, nothing_built):
    weights = tmp_path / "Qwen3-8B"
    weights.mkdir()
    local = JudgeSpec.local(weights, dtype="float32", attention_backend="TRITON_ATTN", enable_thinking=True)
    assert (local.model_type, local.model_name, local.options.tp_size, local.options.model_path) == (
        "local", "Qwen3-8B", 1, str(weights))
    assert local.identity().serving_dtype == "float32" and local.identity().truncation_source == (str(weights), None)
    assert JudgeSpec.local(weights, model_name="label").model_name == "label"
    openai = JudgeSpec.openai("gpt-5.6-luna", reasoning_level="medium", reasoning_summary="auto")
    assert openai.request_settings() == {"reasoning_summary": "auto"}
    assert openai.identity().send_max_output_tokens is False
    anthropic = JudgeSpec.anthropic("claude-haiku-5-5", reasoning_level="medium")
    assert anthropic.capabilities().fixed_temperature == 1.0 and anthropic.identity().send_max_output_tokens is True
    assert JudgeSpec.fake(7).model_name == "fake-7" and JudgeSpec.fake().options.seed == 1
    with pytest.raises(TypeError, match=r"takes no option\(s\) \['reasoning_summary'\]"):
        JudgeSpec.anthropic("m", reasoning_summary="auto")


def test_with_options_is_read_again_by_the_provider():
    spec = JudgeSpec.fake(2)
    assert spec.with_options(max_new_tokens=10).identity().max_new_tokens == 10
    with pytest.raises(JudgeError, match="is named 'fake-3'"):
        spec.with_options(seed=3)


@pytest.mark.parametrize("name, model_type", [("gpt-5.6-luna", "openai"), ("claude-haiku-5-5-batch", "anthropic_batch")])
def test_start_is_the_backend_today_s_config_builds(name, model_type, monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path))
    monkeypatch.setattr(models, "init_openai_client", lambda: object())
    spec = JudgeSpec.from_config(name)
    started = spec.start()
    direct = ModelBackend(OmegaConf.load(MODEL_CONFIG_DIR / f"{name}.yaml"))
    assert isinstance(started, Judge) and isinstance(started, ModelBackend) and started.model_type == model_type
    keys = ("model_type", "model_name", "max_new_tokens", "seed", "temperature", "num_tries", "fixed_temperature",
            "max_retry_seconds", "seed_capability", "_base_delay", "capabilities")
    assert {key: getattr(started, key) for key in keys} == {key: getattr(direct, key) for key in keys}
    assert started.spec == spec and started.usage is started.token_tracker
    assert started.identity() == spec.identity(verify_weights=False)
    started.close()


def test_a_backend_s_identity_is_its_config_as_it_was_built_from(monkeypatch):
    sent = []
    monkeypatch.setattr(models, "init_openai_client", lambda: object())
    monkeypatch.setattr(models, "_make_openai_caller", lambda client, model, effort, summary, **kw: sent.append(effort))
    cfg = OmegaConf.create({"model_type": "openai", "model_name": "gpt-x", "reasoning_level": "low"})
    backend = ModelBackend(cfg)
    cfg.reasoning_level = "high"
    cfg.max_new_tokens = 7
    assert sent == ["low"]
    assert backend.identity().reasoning_effort == "low" and backend.identity().max_new_tokens == 2**16
    assert backend.spec.options["reasoning_level"] == "low" and "max_new_tokens" not in backend.spec.options


def test_closing_a_local_backend_shuts_its_engine_down_once(monkeypatch, stub_backend):
    seen = []
    monkeypatch.setattr(models, "cleanup_vllm", lambda llm, raise_on_error: seen.append((llm, raise_on_error)))
    backend = stub_backend(llm="engine")
    backend.close()
    backend.close()
    assert seen == [("engine", False)] and backend.llm is None


def test_the_backend_dispatches_on_the_built_in_model_types_alone():
    assert set(providers.PROVIDERS) == set(BUILTIN_MODEL_TYPES)
    with pytest.raises(ValueError, match="Unsupported model_type: 'telepathy'"):
        providers.builtin_provider("telepathy")
    assert backend_module.BUILTIN_MODEL_TYPES is BUILTIN_MODEL_TYPES


def test_a_spec_whose_provider_names_another_judge_is_refused(monkeypatch):
    spec = JudgeSpec.fake(1)
    monkeypatch.setattr(providers.FakeProvider, "identity",
                        lambda self, spec, verify_weights=True: providers.config_identity(
                            "fake", spec.options.config(), seed_capability="unsupported", model_name="other"))
    with pytest.raises(JudgeError, match="named its judge 'fake' 'other'"):
        spec.identity()


def test_a_built_in_identity_carrying_an_extension_is_refused(monkeypatch):
    from dataclasses import replace

    real = providers.FakeProvider.identity
    monkeypatch.setattr(providers.FakeProvider, "identity",
                        lambda self, spec, verify_weights=True: replace(real(self, spec), extension="{}"))
    with pytest.raises(JudgeError, match="a built-in one's carries none"):
        JudgeSpec.fake(1).identity()


def test_the_abstract_judges_declare_the_contract():
    assert {"generate_once", "usage"} <= Judge.__abstractmethods__
    assert {"estimate_block_tokens", "submit_block", "poll_block"} <= BatchJudge.__abstractmethods__

    class Minimal(Judge):
        spec = JudgeSpec.fake(1)

        def generate_once(self, prompts, **kwargs):
            return []

        @property
        def usage(self):
            return None

    judge = Minimal()
    assert judge.identity() == JudgeSpec.fake(1).identity() and judge.close() is None


def test_identity_fields_are_plain_data(tmp_path):
    adapter = tmp_path / "checkpoint-1"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8}))
    spec = JudgeSpec.local(tmp_path / "Qwen3-8B", adapter_path=str(adapter / "adapter_model.safetensors"))
    fields = identity_fields(spec.identity())
    assert fields["adapter"]["label"] == "checkpoint-1/adapter_model.safetensors"
    assert fields["truncation_source"] == [str(tmp_path / "Qwen3-8B"), None]
    assert identity_fields(JudgeSpec.fake(1).identity())["adapter"] is None
    json.dumps(fields)


def test_resolve_reads_names_shorthands_and_specs(tmp_path, nothing_built):
    spec = JudgeSpec.fake(4)
    assert resolve(spec) is spec
    assert resolve("fake") == JudgeSpec.fake(1) and resolve("fake:4") == spec
    assert resolve("openai:gpt-5.6-luna") == JudgeSpec.openai("gpt-5.6-luna")
    assert resolve(f"local:{tmp_path}").model_type == "local"
    assert resolve("gpt-5.6-luna").model_type == "openai"
    assert resolve({"model_type": "fake", "model_name": "fake-4", "seed": 4}) == spec
    with pytest.raises(JudgeError, match="no model config named 'fake:x'"):
        resolve("fake:x")


def test_check_probes_the_fake_and_only_reads_a_paid_judge(monkeypatch):
    monkeypatch.setattr(models, "init_openai_client", lambda: pytest.fail("a paid judge was started"))
    report = check("fake")
    assert report.ok and report.probe.returned and report.probe.text in ("yes", "no", "unsure")
    assert report.identity.model_name == "fake-1" and report.request_settings == {}
    paid = check(JudgeSpec.openai("m"))
    assert paid.probe is None and paid.ok and paid.capabilities == Capabilities(
        seed="unsupported", per_request_errors=True, spends_money=True)


def test_probing_a_paid_judge_asks_the_spend_confirmation_before_it_starts(monkeypatch):
    from ruleforge.judges.batch_api import BatchSpendNotConfirmed

    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started before the spend was confirmed"))
    with pytest.raises(BatchSpendNotConfirmed):
        check("openai:m", probe=True)


def test_a_failed_probe_is_reported(monkeypatch):
    class Silent(SimpleNamespace):
        def generate_once(self, prompts, **kwargs):
            return [GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="down")]

        def close(self):
            self.closed = True

    silent = Silent()
    monkeypatch.setattr(JudgeSpec, "start", lambda self: silent)
    report = check("fake", probe=True)
    assert not report.ok and silent.closed


def test_the_config_directory_is_the_checkout_s():
    assert judges_base.MODEL_CONFIG_DIR == MODEL_CONFIG_DIR and MODEL_CONFIG_DIR.parts[-2:] == ("configs", "model")


def test_a_spec_is_hashable_and_equal_specs_hash_alike():
    assert len({JudgeSpec.fake(1), JudgeSpec.fake(1), JudgeSpec.fake(2)}) == 2
    assert JudgeOptions({"a": 1}) == JudgeOptions({"a": 1.0}) and hash(JudgeOptions({"a": 1})) == hash(
        JudgeOptions({"a": 1.0}))


def test_a_sync_anthropic_ceiling_the_sdk_would_refuse_is_refused_as_a_spec():
    """The anthropic SDK refuses a non-streaming call above its ceiling before any request, on every call alike."""
    import anthropic

    from ruleforge.judges.base import ANTHROPIC_DEFAULT_MAX_TOKENS
    from ruleforge.judges.providers import ANTHROPIC_SYNC_MAX_TOKENS

    client = anthropic.Anthropic(api_key="dummy")
    client._calculate_nonstreaming_timeout(ANTHROPIC_SYNC_MAX_TOKENS, None)
    with pytest.raises(ValueError, match="Streaming is required"):
        client._calculate_nonstreaming_timeout(ANTHROPIC_SYNC_MAX_TOKENS + 1, None)
    assert JudgeSpec.anthropic("claude-haiku-5-5").identity().max_new_tokens == ANTHROPIC_DEFAULT_MAX_TOKENS == 16384
    assert JudgeSpec.anthropic("m", max_new_tokens=ANTHROPIC_SYNC_MAX_TOKENS).identity().max_new_tokens == 21333
    with pytest.raises(JudgeError, match="the anthropic SDK refuses above 21333"):
        JudgeSpec.anthropic("m", max_new_tokens=ANTHROPIC_SYNC_MAX_TOKENS + 1)
