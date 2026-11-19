from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ruleforge.judges import JudgeError, JudgeIdentity, JudgeSpec
from ruleforge.judges import anthropic_batch, models, openai_batch

PROMPT = {"system_prompt": "Rule 1: be brief. Does the response break it?", "user_prompt": ""}
BLOCK = "camp/cond/draw-000/attempt-0/block-0000"
BATCH_MODULES = {"openai_batch": openai_batch, "anthropic_batch": anthropic_batch}
ABSENT = object()


def _config(model_type: str, send, **extra):
    config = {"model_type": model_type, "model_name": "m", "max_new_tokens": 512, **extra}
    if send is not ABSENT:
        config["send_max_output_tokens"] = send
    return config


@pytest.fixture
def clients(monkeypatch):
    """Every sync provider's client, recording the requests it is sent."""
    sent: dict[str, dict] = {}

    openai = MagicMock()
    openai.responses.create.side_effect = lambda **kw: sent.update(openai=kw) or SimpleNamespace(
        output_text="no", output=[], usage=None, status="completed")
    claude = MagicMock()
    claude.messages.create.side_effect = lambda **kw: sent.update(anthropic=kw) or {
        "content": [{"type": "text", "text": "no"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3}}

    monkeypatch.setattr(models, "init_openai_client", lambda: openai)
    monkeypatch.setattr(models, "init_anthropic_client", lambda: claude)
    for name, module in BATCH_MODULES.items():
        def collect(runner, requests, *, namespace, provider, tracker, summaries_requested=False, name=name):
            sent[name] = requests[0].body
            return [{"raw_response": "no", "completion": "no", "finish_reason": "stop", "error": None}
                    for _ in requests]

        monkeypatch.setattr(module, "collect_block_details", collect)
    return sent


def _sent_body(spec: JudgeSpec, sent: dict) -> dict:
    judge = spec.start()
    [outcome] = judge.generate_once([dict(PROMPT)], temperature=1.0, seeds=[5], request_block_id=BLOCK)
    assert outcome.text == "no" and outcome.error is None
    return sent[spec.model_type]


def _sent_effort(body: Any) -> Any:
    if isinstance(body, dict):
        for key, value in body.items():
            if key in ("effort", "thinking_level") and isinstance(value, str):
                return value.lower()
            found = _sent_effort(value)
            if found is not None:
                return found
    return None


def _carries_ceiling(model_type: str, body: dict) -> bool:
    if model_type in ("openai", "openai_batch"):
        return body.get("max_output_tokens") == 512 or body.get("max_completion_tokens") == 512
    return body["max_tokens"] == 512  # anthropic, anthropic_batch: required by the Messages API


#: (model type, send_max_output_tokens as configured) -> the identity's value, or None for a refused config.
SEND_TABLE = {
    ("openai", ABSENT): False, ("openai", True): True, ("openai", False): False,
    ("anthropic", True): True, ("anthropic", False): None, ("anthropic", ABSENT): None,
    ("openai_batch", ABSENT): False, ("openai_batch", True): True, ("openai_batch", False): False,
    ("anthropic_batch", True): True, ("anthropic_batch", False): None, ("anthropic_batch", ABSENT): None,
}
SEEDS = {"openai": "unsupported", "anthropic": "unsupported", "openai_batch": "unsupported",
         "anthropic_batch": "unsupported"}


@pytest.mark.parametrize("model_type, send", list(SEND_TABLE), ids=lambda value: getattr(value, "__name__", None)
                         or ("absent" if value is ABSENT else str(value)))
def test_the_identity_is_section_zero_s_and_the_request_agrees(model_type, send, clients, tmp_path, monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    extra = {"batch_root": str(tmp_path)} if model_type in BATCH_MODULES else {}
    expected = SEND_TABLE[(model_type, send)]
    if expected is None:
        with pytest.raises(ValueError, match="set send_max_output_tokens: true"):
            JudgeSpec.from_config(_config(model_type, send, **extra))
        return
    effort = "medium"
    spec = JudgeSpec.from_config(_config(model_type, send, reasoning_level=effort, **extra))
    assert spec.identity() == JudgeIdentity(
        model_name="m", model_type=model_type, seed_capability=SEEDS[model_type], max_new_tokens=512,
        reasoning_effort=effort, send_max_output_tokens=expected)
    body = _sent_body(spec, clients)
    assert _carries_ceiling(model_type, body) is expected
    assert _sent_effort(body) == spec.identity().reasoning_effort


@pytest.mark.parametrize("model_type", ["fake", "local"])
@pytest.mark.parametrize("key", ["reasoning_level", "gpt_oss_reasoning_level"])
def test_a_reasoning_level_no_request_carries_is_refused(model_type, key, tmp_path):
    config = {"model_type": model_type, "model_name": "fake-1" if model_type == "fake" else "m", "seed": 1,
              "model_path": str(tmp_path / "w"), key: "high"}
    if model_type == "local" and key == "gpt_oss_reasoning_level":
        assert JudgeSpec.from_config(config).identity().reasoning_effort == "high"
        return
    with pytest.raises(JudgeError, match=f"model_type '{model_type}' sends no {key}"):
        JudgeSpec.from_config(config)


@pytest.mark.parametrize("model_type", ["openai", "anthropic", "local", "fake"])
def test_a_batch_endpoint_off_a_batch_transport_is_refused(model_type, tmp_path):
    config = {"model_type": model_type, "model_name": "fake-1" if model_type == "fake" else "m", "seed": 1,
              "model_path": str(tmp_path / "w"), "send_max_output_tokens": True, "max_new_tokens": 512,
              "batch_endpoint": "chat_completions"}
    with pytest.raises(JudgeError, match=f"model_type '{model_type}' sends no batch request"):
        JudgeSpec.from_config(config)


@pytest.mark.parametrize("model_type", ["local", "fake"])
def test_a_judge_with_no_request_records_no_ceiling(model_type, tmp_path):
    config = ({"model_type": "local", "model_name": "w", "model_path": str(tmp_path / "w")} if model_type == "local"
              else {"model_type": "fake", "model_name": "fake-1", "seed": 1})
    spec = JudgeSpec.from_config({**config, "send_max_output_tokens": True})
    assert spec.identity().send_max_output_tokens is None


#: Configs whose requests may carry a reasoning summary: (model type, extra config).
SUMMARY_CONFIGS = [("openai", {}), ("openai_batch", {"batch_endpoint": "responses"}),
                   ("openai_batch", {"batch_endpoint": "chat_completions"})]


@pytest.mark.parametrize("model_type, extra", SUMMARY_CONFIGS)
@pytest.mark.parametrize("effort", [None, "medium"])
@pytest.mark.parametrize("summary", [None, "auto"])
def test_request_settings_carry_the_summary_exactly_when_the_body_does(model_type, extra, effort, summary, clients,
                                                                      tmp_path, monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    config = {"model_type": model_type, "model_name": "m", "reasoning_level": effort, "reasoning_summary": summary,
              **extra, **({"batch_root": str(tmp_path)} if model_type in BATCH_MODULES else {})}
    spec = JudgeSpec.from_config(config)
    body = _sent_body(spec, clients)
    carried = (body.get("reasoning") or {}).get("summary")
    assert spec.request_settings() == ({} if carried is None else {"reasoning_summary": carried})
    assert "batch_root" not in spec.request_settings()


@pytest.mark.parametrize("model_type", ["local", "anthropic", "fake", "anthropic_batch"])
def test_no_other_judge_has_request_settings(model_type, tmp_path):
    config = {"model_type": model_type, "model_name": "fake-16" if model_type == "fake" else "m",
              "batch_root": str(tmp_path), "send_max_output_tokens": True, "max_new_tokens": 512}
    assert JudgeSpec.from_config(config).request_settings() == {}
