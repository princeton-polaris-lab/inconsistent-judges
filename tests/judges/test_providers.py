from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import Capabilities, JudgeError, JudgeIdentity, JudgeSpec, Provider, extension, register_judge
from ruleforge.judges import providers
from ruleforge.evaluate.campaign import condition_spec_for
from ruleforge.evaluate.identity import SamplingConfig
from ruleforge.tasks import ModelSpecTask, TaskSettings
from ruleforge.judges import anthropic_batch, openai_batch
from ruleforge.judges.backend import ModelBackend


class EchoProvider(Provider):
    """A provider of one's own: its judge answers by an option, which its identity names."""

    def identity(self, spec, *, verify_weights=True):
        return JudgeIdentity(model_name=spec.model_name, model_type=spec.model_type, seed_capability="unsupported",
                             max_new_tokens=8, extension=extension("echo", "1.0", {"answer": spec.options.answer}))

    def capabilities(self, spec):
        return Capabilities(seed="unsupported", per_request_errors=True)

    def start(self, spec):
        return SimpleNamespace(spec=spec)


@pytest.fixture
def registry(monkeypatch):
    """A clean registry of providers of one's own, restored afterwards."""
    monkeypatch.setattr(providers, "_REGISTERED", {})
    return providers._REGISTERED


def test_a_registered_provider_s_extension_reaches_the_condition(registry, tmp_path):
    register_judge("echo", EchoProvider)
    spec = JudgeSpec.from_config({"model_type": "echo", "model_name": "echo-1", "answer": "yes"})
    identity = spec.identity()
    assert identity.extension == '{"options":{"answer":"yes"},"provider":"echo","version":"1.0"}'
    assert spec.capabilities().per_request_errors and spec.request_settings() == {}
    assert spec.start().spec is spec
    rules = tmp_path / "rules.jsonl"
    rules.write_text('{"id": "Rule 1", "rule": "be brief"}\n')
    from ruleforge.rulesets import Ruleset

    condition = condition_spec_for(
        condition_id="echo-1/naive", config=SamplingConfig(campaign_mode="pilot"), judge=identity,
        task=ModelSpecTask().identity(TaskSettings("naive")), ruleset=Ruleset.load(rules))
    assert condition.judge_extension == identity.extension and condition.to_dict()["judge_extension"]
    other = JudgeSpec.from_config({"model_type": "echo", "model_name": "echo-1", "answer": "no"}).identity()
    assert condition_spec_for(condition_id="echo-1/naive", config=SamplingConfig(campaign_mode="pilot"), judge=other,
                              task=ModelSpecTask().identity(TaskSettings("naive")),
                              ruleset=Ruleset.load(rules)).fingerprint != condition.fingerprint


def test_a_registered_provider_s_identity_without_its_extension_is_refused(registry):
    class Bare(EchoProvider):
        def identity(self, spec, *, verify_weights=True):
            return JudgeIdentity(model_name=spec.model_name, model_type=spec.model_type,
                                 seed_capability="unsupported", max_new_tokens=8)

    register_judge("bare", Bare())
    with pytest.raises(JudgeError, match="a registered judge's identity carries its extension"):
        JudgeSpec.from_config({"model_type": "bare", "model_name": "b"}).identity()


@pytest.mark.parametrize("name", ["openai", "fake", "local", "anthropic_batch"])
def test_a_built_in_model_type_cannot_be_registered(registry, name):
    with pytest.raises(JudgeError, match="is a built-in model_type"):
        register_judge(name, EchoProvider)


@pytest.mark.parametrize("name, provider, match", [
    ("", EchoProvider, "under a model_type name"),
    ("echo", object, "is not a ruleforge.judges.Provider"),
])
def test_register_judge_refuses_what_is_not_a_provider_under_a_name(registry, name, provider, match):
    with pytest.raises(JudgeError, match=match):
        register_judge(name, provider)


class _EntryPoint(SimpleNamespace):
    def load(self):
        self.loaded = getattr(self, "loaded", 0) + 1
        return self.target


def test_an_installed_provider_is_found_by_its_model_type_and_loaded_once(registry, monkeypatch):
    entries = [_EntryPoint(name="other", value="pkg:Other", target=EchoProvider),
               _EntryPoint(name="echo", value="pkg:Echo", target=EchoProvider)]
    seen = []
    monkeypatch.setattr(providers.metadata, "entry_points",
                        lambda group: seen.append(group) or list(entries))
    found = providers.provider_for("echo")
    assert isinstance(found, EchoProvider) and providers.provider_for("echo") is found
    assert seen == [providers.ENTRY_POINT_GROUP] and entries[1].loaded == 1 and not hasattr(entries[0], "loaded")
    with pytest.raises(JudgeError, match="no judge provider for model_type 'missing'"):
        providers.provider_for("missing")


def test_an_installed_provider_named_like_a_built_in_is_refused(registry, monkeypatch):
    monkeypatch.setattr(providers.metadata, "entry_points",
                        lambda group: [_EntryPoint(name="openai", value="pkg:Spoof", target=EchoProvider)])
    with pytest.raises(JudgeError, match="entry point 'openai' .* is named like a built-in model_type"):
        providers.provider_for("echo")
    assert providers.provider_for("openai") is providers.PROVIDERS["openai"]


def test_the_backend_serves_no_registered_model_type(registry):
    register_judge("echo", EchoProvider)
    with pytest.raises(ValueError, match="Unsupported model_type: 'echo'"):
        ModelBackend(OmegaConf.create({"model_type": "echo", "model_name": "e"}))


def test_a_campaign_built_from_a_registered_backend_records_its_extension(registry):
    register_judge("echo", EchoProvider)
    backend = SimpleNamespace(model_type="echo", model_name="label", max_new_tokens=99, seed_capability="unsupported")
    identity = providers.backend_identity(backend, OmegaConf.create({"model_type": "echo", "model_name": "e",
                                                                    "answer": "no"}))
    assert (identity.model_name, identity.max_new_tokens) == ("label", 99)
    assert identity.extension == extension("echo", "1.0", {"answer": "no"})


def test_the_windowed_transports_are_the_ones_with_windowed_ops():
    modules = {"openai_batch": openai_batch, "anthropic_batch": anthropic_batch}
    assert set(providers.WINDOWED_MODEL_TYPES) == {
        name for name, module in modules.items() if hasattr(module, "make_windowed_block_ops")}


@pytest.mark.parametrize("model_type, enqueued, windowed", [
    ("openai_batch", 1000, True), ("openai_batch", None, False), ("anthropic_batch", 1000, False),
])
def test_a_batch_transport_is_windowed_when_it_can_keep_blocks_in_flight(model_type, enqueued, windowed, tmp_path):
    spec = JudgeSpec.from_config({"model_type": model_type, "model_name": "m", "batch_root": str(tmp_path),
                                  "max_enqueued_tokens": enqueued, "fixed_temperature": 1.0,
                                  "send_max_output_tokens": True})
    assert spec.capabilities() == Capabilities(
        seed="unsupported", batch=True,
        windowed=windowed, per_request_errors=True, spends_money=True, fixed_temperature=1.0)
    backend = spec.start()
    assert backend.capabilities == spec.capabilities()
    assert (backend.max_enqueued_tokens is not None) is windowed


def test_a_local_judge_s_capabilities(tmp_path):
    assert JudgeSpec.local(tmp_path / "w", fixed_temperature=1.0).capabilities() == Capabilities(
        seed="local_deterministic", per_request_errors=True, sampler_truncation=True, local_engine=True,
        fixed_temperature=1.0)
    assert JudgeSpec.fake(1).capabilities() == Capabilities(seed="local_deterministic", per_request_errors=True)


@pytest.mark.parametrize("config, match", [
    ({"model_type": "anthropic_batch", "batch_endpoint": "responses", "send_max_output_tokens": True},
     "has one endpoint"),
    ({"model_type": "anthropic", "reasoning_summary": "auto", "send_max_output_tokens": True},
     "sends no reasoning_summary"),
    ({"model_type": "anthropic", "reasoning_level": "huge", "send_max_output_tokens": True},
     "unknown Claude effort level"),
    ({"model_type": "openai_batch", "batch_endpoint": "carrier_pigeon"}, "unknown batch endpoint"),
])
def test_a_config_its_judge_could_not_start_from_is_refused_as_a_spec(config, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        JudgeSpec.from_config({**config, "model_name": "m", "batch_root": str(tmp_path)}).identity()


def test_a_judge_s_serving_dtype_refusal_names_the_judge(tmp_path):
    with pytest.raises(JudgeError, match="judge_model w: dtype float32 with attention_backend FLASH_ATTN"):
        JudgeSpec.local(tmp_path / "w", dtype="float32", attention_backend="FLASH_ATTN").identity()


def test_a_local_judge_s_derived_weights_are_verified_or_read_as_recorded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("ruleforge.judges.weights.model_weights_identity",
                        lambda path: calls.append(("verified", path)) or SimpleNamespace(sha256="v" * 64))
    monkeypatch.setattr("ruleforge.judges.weights.recorded_model_weights",
                        lambda path: calls.append(("recorded", path)) or SimpleNamespace(sha256="r" * 64))
    spec = JudgeSpec.local(tmp_path / "merged")
    assert spec.identity().model_weights_sha256 == "v" * 64
    assert spec.identity(verify_weights=False).model_weights_sha256 == "r" * 64
    assert [kind for kind, _ in calls] == ["verified", "recorded"] and Path(calls[0][1]).name == "merged"


def test_the_provider_base_reads_a_spec_from_any_config(registry):
    register_judge("echo", EchoProvider)
    with pytest.raises(JudgeError, match="'echo' judge's config names its model_name"):
        JudgeSpec.from_config({"model_type": "echo"})


def test_a_campaign_built_from_a_backend_reports_a_refused_judge_as_its_own(tmp_path):
    from ruleforge.evaluate.runner import CampaignRunError

    rules = tmp_path / "rules.jsonl"
    rules.write_text('{"id": "Rule 1", "rule": "be brief"}\n')
    backend = SimpleNamespace(model_type="local", model_name="w", max_new_tokens=8,
                              seed_capability="local_deterministic")
    model_cfg = OmegaConf.create({"model_type": "local", "model_name": "w", "model_path": str(tmp_path / "w"),
                                  "dtype": "float32", "attention_backend": "FLASH_ATTN"})
    with pytest.raises(CampaignRunError, match="judge_model w: dtype float32"):
        condition_spec_for(condition_id="w/naive", backend=backend, model_cfg=model_cfg, rule_set_path=rules,
                           task_spec=SimpleNamespace(reasoning_prompt_style="naive", individual_rule_context="isolated"),
                           judge_type="No_Interpretation", config=SamplingConfig(campaign_mode="pilot"))
