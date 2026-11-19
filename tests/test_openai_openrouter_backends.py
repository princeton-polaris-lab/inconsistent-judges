from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import models
from ruleforge.judges.backend import ModelBackend, _optional_str
from ruleforge.judges.batch_api import BatchExecutionError, BatchSpendNotConfirmed


class _MinimalSpec:
    individual_rule_context = False
    interpretation_rules: dict = {}
    valid_eval_responses = ("yes", "no", "unsure")
    eval_response_validator = None

    def build_eval_prompt(self, rule_text, scenario, judge_type, context_rules=None):
        return f"{judge_type}:{rule_text}:{scenario.get('prompt', '')}"


def _text(detail):
    return detail["completion"] or None


MODELS_LOGGER = "ruleforge.judges.models"


@pytest.fixture(autouse=True)
def _no_ambient_base_url(monkeypatch):
    for name in models.OPENAI_REDIRECT_ENVS:
        monkeypatch.delenv(name, raising=False)


def _responses_obj(output_text=None, output=(), usage=None):
    return SimpleNamespace(output_text=output_text, output=list(output), usage=usage)


def _message_item(*texts):
    return SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=t) for t in texts],
    )


def _chat_obj(content, usage=None):
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def _usage(prompt=11, completion=5, total=16):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        self.exited = True
        return False

    def __iter__(self):
        return iter(self._events)


class _Tracker:
    def __init__(self):
        self.seen = []

    def add(self, usage):
        self.seen.append(usage)


def test_openai_caller_sends_the_developer_role_and_a_reasoning_object():
    """The exact request body behind the published Luna ruleset-size campaign."""
    client = MagicMock()
    client.responses.create.return_value = _responses_obj(output_text=" yes ")
    caller = models._make_openai_caller(client, "gpt-5.6-luna", "medium", "auto")

    assert _text(caller("system", "user", 1.0, 99, None)) == "yes"

    assert client.responses.create.call_args.kwargs == {
        "model": "gpt-5.6-luna",
        "input": [
            {"role": "developer", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 1.0,
        "reasoning": {"effort": "medium", "summary": "auto"},
    }


@pytest.mark.parametrize(
    "system_prompt,user_prompt,expected_input",
    [
        ("system", "", [{"role": "developer", "content": "system"}]),
        ("system", "   \n", [{"role": "developer", "content": "system"}]),
        ("", "user", [{"role": "user", "content": "user"}]),
        (
            "system",
            "user",
            [
                {"role": "developer", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        ),
        ("  ", "\t", []),
    ],
)
def test_openai_caller_drops_blank_turns(system_prompt, user_prompt, expected_input):
    client = MagicMock()
    client.responses.create.return_value = _responses_obj(output_text="ok")
    caller = models._make_openai_caller(client, "gpt-5.6-luna", "medium", "auto")

    caller(system_prompt, user_prompt, 1.0, 99, None)

    assert client.responses.create.call_args.kwargs["input"] == expected_input


@pytest.mark.parametrize(
    "effort,summary,expected",
    [
        ("medium", "auto", {"reasoning": {"effort": "medium", "summary": "auto"}}),
        ("low", None, {"reasoning": {"effort": "low"}}),
        (None, "auto", {}),
        (None, None, {}),
    ],
)
def test_openai_caller_omits_reasoning_fields_that_were_not_configured(
    effort, summary, expected
):
    client = MagicMock()
    client.responses.create.return_value = _responses_obj(output_text="ok")
    caller = models._make_openai_caller(client, "gpt-5.6-luna", effort, summary)

    caller("system", "", 1.0, 99, None)

    kwargs = client.responses.create.call_args.kwargs
    assert {k: v for k, v in kwargs.items() if k == "reasoning"} == expected


def test_openai_caller_handles_empty_and_failed_responses(caplog):
    client = MagicMock()
    caller = models._make_openai_caller(client, "gpt-5.6-luna", "medium", "auto")

    client.responses.create.return_value = _responses_obj(output_text="   ")
    assert caller("system", "", 1.0, 99, None) == {
        "raw_response": "", "completion": "", "finish_reason": None, "error": None, "reasoning_summaries": []}

    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        client.responses.create.side_effect = RuntimeError("offline")
        assert caller("system", "", 1.0, 99, None) == {
            "raw_response": "", "completion": "", "finish_reason": None, "error": "RuntimeError: offline",
            "reasoning_summaries": []}

    assert "OpenAI API error" in caplog.text
    assert "gpt-5.6-luna" in caplog.text
    assert "offline" in caplog.text
    assert "Traceback" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_openai_caller_reports_usage_to_the_tracker_and_survives_without_one():
    client = MagicMock()
    client.responses.create.return_value = _responses_obj(
        output_text="ok", usage=_usage()
    )
    tracker = _Tracker()

    tracked = models._make_openai_caller(
        client, "gpt-5.6-luna", "medium", "auto", tracker=tracker
    )
    assert _text(tracked("system", "", 1.0, 99, None)) == "ok"
    assert [(u.prompt_tokens, u.completion_tokens) for u in tracker.seen] == [(11, 5)]

    untracked = models._make_openai_caller(client, "gpt-5.6-luna", "medium", "auto")
    assert _text(untracked("system", "", 1.0, 99, None)) == "ok"
    assert len(tracker.seen) == 1


@pytest.mark.parametrize(
    "response,expected",
    [
        # The SDK's aggregated convenience property wins when it has content.
        (_responses_obj(output_text=" yes ", output=[_message_item("no")]), "yes"),
        # Absent (the raw-JSON case): walk output[] instead.
        (_responses_obj(output=[_message_item(" yes\n")]), "yes"),
        # Present but blank: same fallback, because "" is not an answer.
        (_responses_obj(output_text="   ", output=[_message_item("yes")]), "yes"),
        (_responses_obj(output=[_message_item("ye", "s")]), "yes"),
        # Reasoning items are skipped; only `message` items carry the answer.
        (
            _responses_obj(
                output=[
                    SimpleNamespace(type="reasoning", content=[]),
                    _message_item("yes"),
                ]
            ),
            "yes",
        ),
        (
            _responses_obj(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(type="refusal", text="nope"),
                            SimpleNamespace(type="output_text", text="yes"),
                        ],
                    )
                ]
            ),
            "yes",
        ),
        # A non-string `text` is ignored rather than crashing the join.
        (
            _responses_obj(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text=None)],
                    )
                ]
            ),
            None,
        ),
        (_responses_obj(output=[SimpleNamespace(type="message", content=None)]), None),
        (_responses_obj(), None),
        (SimpleNamespace(), None),
    ],
)
def test_responses_text_reads_both_the_sdk_and_the_raw_json_shape(response, expected):
    """``output_text`` is SDK-only; a batch output file has no such field."""
    assert models._responses_text(response) == expected


def test_stream_responses_accumulates_deltas_and_takes_usage_from_completed():
    stream = _FakeStream(
        [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_text.delta", delta=" ye"),
            SimpleNamespace(type="response.reasoning_summary_text.delta", delta="XXX"),
            SimpleNamespace(type="response.output_text.delta", delta="s "),
            SimpleNamespace(type="response.output_text.delta", delta=None),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(usage=_usage(prompt=3, completion=4, total=7)),
            ),
        ]
    )
    client = MagicMock()
    client.responses.stream.return_value = stream

    text, usage, finish_reason, summaries, reasoning = models._stream_responses_detail(client, {"model": "m"})

    assert (text, summaries, reasoning) == ("yes", [], None)
    assert (usage.prompt_tokens, usage.completion_tokens) == (3, 4)
    assert client.responses.stream.call_args.kwargs == {"model": "m"}
    assert stream.entered and stream.exited


@pytest.mark.parametrize("events", [
    [SimpleNamespace(type="response.output_text.delta", delta="partial")],
    [],
], ids=["cut-after-text", "no-event"])
def test_stream_responses_refuses_a_stream_cut_short_as_a_failed_call(events):
    client = MagicMock()
    client.responses.stream.return_value = _FakeStream(events)

    with pytest.raises(models.CallFailed, match="ended before response.completed"):
        models._stream_responses_detail(client, {})


def test_stream_responses_returns_none_for_a_completed_stream_with_no_text():
    client = MagicMock()
    client.responses.stream.return_value = _FakeStream(
        [SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None))]
    )

    assert models._stream_responses_detail(client, {}) == (None, None, None, [], None)


def test_openai_caller_streams_through_the_context_manager_when_asked():
    stream = _FakeStream(
        [
            SimpleNamespace(type="response.output_text.delta", delta="ye"),
            SimpleNamespace(type="response.output_text.delta", delta="s"),
            SimpleNamespace(
                type="response.completed", response=SimpleNamespace(usage=_usage())
            ),
        ]
    )
    client = MagicMock()
    client.responses.stream.return_value = stream
    tracker = _Tracker()
    caller = models._make_openai_caller(
        client, "gpt-5.6-luna", "medium", "auto", tracker=tracker, stream=True
    )

    assert _text(caller("system", "", 1.0, 99, None)) == "yes"

    client.responses.create.assert_not_called()
    assert client.responses.stream.call_args.kwargs == {
        "model": "gpt-5.6-luna",
        "input": [{"role": "developer", "content": "system"}],
        "temperature": 1.0,
        "reasoning": {"effort": "medium", "summary": "auto"},
    }
    assert [u.total_tokens for u in tracker.seen] == [16]


def test_seed_warning_fires_exactly_once_and_names_the_provider(caplog):
    state = {}
    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        for seed in range(31337, 31387):
            models._warn_seed_ignored_once(state, "OpenAI", seed)

    warnings = [r for r in caplog.records if "no seed parameter" in r.getMessage()]
    assert len(warnings) == 1
    assert "OpenAI" in warnings[0].getMessage()
    assert "31337" in warnings[0].getMessage()
    assert state == {"warned": True}


def test_seed_warning_is_silent_when_no_seed_was_supplied(caplog):
    state = {}
    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        for _ in range(5):
            models._warn_seed_ignored_once(state, "The OpenAI Responses API", None)

    assert caplog.text == ""
    assert state == {}


def test_the_responses_caller_warns_once_about_the_seed_it_cannot_send(caplog):
    factory, provider_text = models._make_openai_caller, "OpenAI"
    client = MagicMock()
    client.responses.create.return_value = _responses_obj(output_text="ok")
    client.chat.completions.create.return_value = _chat_obj("ok")
    caller = factory(client, "model", "medium")

    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        caller("system", "", 1.0, 99, 7)
        caller("system", "", 1.0, 99, 8)
        caller("system", "", 1.0, 99, None)

    warnings = [r for r in caplog.records if "no seed parameter" in r.getMessage()]
    assert len(warnings) == 1
    assert provider_text in warnings[0].getMessage()

    with caplog.at_level(logging.WARNING, logger=MODELS_LOGGER):
        factory(client, "model", "medium")("system", "", 1.0, 99, 7)
    said = [r for r in caplog.records if "no seed parameter" in r.getMessage()]
    assert len(said) == 2


@pytest.mark.parametrize("init_fn,env_var", [(models.init_openai_client, "OPENAI_API_KEY")])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_init_requires_a_nonblank_api_key(monkeypatch, init_fn, env_var, value):
    if value is None:
        monkeypatch.delenv(env_var, raising=False)
    else:
        monkeypatch.setenv(env_var, value)

    with pytest.raises(ValueError, match=env_var):
        init_fn()


def test_init_openai_refuses_an_ambient_base_url_naming_it_not_its_value(monkeypatch):
    """The SDK reads OPENAI_BASE_URL on its own."""
    monkeypatch.setenv("OPENAI_API_KEY", "api")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ValueError, match="OPENAI_BASE_URL set: the openai SDK would use it") as refused:
        models.init_openai_client()
    assert "127.0.0.1" not in str(refused.value)


def test_init_openai_passes_only_the_key(monkeypatch):
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "  api  ")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    client = models.init_openai_client()

    assert isinstance(client, FakeOpenAI)
    assert calls == [{"api_key": "api"}]


@pytest.mark.parametrize(
    "init_fn,env_var,message",
    [
        (models.init_openai_client, "OPENAI_API_KEY", "Failed to initialize OpenAI"),
    ],
)
def test_init_failure_never_puts_the_api_key_in_an_exception_or_a_log(
    monkeypatch, caplog, init_fn, env_var, message
):
    class BrokenOpenAI:
        def __init__(self, **kwargs):
            raise RuntimeError(f"bad credentials: {kwargs!r}")

    monkeypatch.setenv(env_var, "SENSITIVE_TEST_KEY")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=BrokenOpenAI))

    with caplog.at_level(logging.ERROR, logger=MODELS_LOGGER):
        with pytest.raises(RuntimeError, match=message) as exc_info:
            init_fn()

    assert "SENSITIVE_TEST_KEY" not in str(exc_info.value)
    assert "SENSITIVE_TEST_KEY" not in caplog.text
    assert exc_info.value.__cause__ is None


def _sync_config(model_type, **extra):
    return OmegaConf.create(
        {
            "model_type": model_type,
            "model_name": "test-model",
            "max_new_tokens": 100,
            "temperature": 1.0,
            "num_tries": 1,
            "seed": 3,
            **extra,
        }
    )


def _capturing_caller(seen):
    def call(system, user, temperature, max_tokens, seed):
        seen["call"] = (system, user, temperature, max_tokens, seed)
        return {"raw_response": "ok", "completion": "ok", "finish_reason": None, "error": None}

    return call


@pytest.mark.parametrize("stream", [True, False])
def test_backend_wires_the_openai_caller_with_effort_summary_and_stream(
    monkeypatch, stream
):
    client = object()
    seen = {}

    def fake_make_caller(
        given_client, model_name, reasoning_effort, reasoning_summary, **kwargs
    ):
        seen.update(
            client=given_client,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            **kwargs,
        )
        return _capturing_caller(seen)

    monkeypatch.setattr(models, "init_openai_client", lambda: client)
    monkeypatch.setattr(models, "_make_openai_caller", fake_make_caller)

    backend = ModelBackend(
        _sync_config(
            "openai",
            reasoning_level="medium",
            reasoning_summary="auto",
            stream=stream,
        )
    )

    assert backend.generate([{"system_prompt": "s", "user_prompt": ""}]) == [["ok"]]
    assert seen["client"] is client
    assert seen["model_name"] == "test-model"
    assert seen["reasoning_effort"] == "medium"
    assert seen["reasoning_summary"] == "auto"
    assert seen["stream"] is stream
    assert seen["tracker"] is backend.token_tracker
    assert backend._base_delay == 1.0


@pytest.mark.parametrize(
    "config_extra,expected",
    [
        ({"reasoning_level": "medium"}, "medium"),
        ({"gpt_oss_reasoning_level": "low"}, "low"),
        ({"reasoning_level": "high", "gpt_oss_reasoning_level": "low"}, "high"),
        # An explicit null falls through to the other spelling.
        ({"reasoning_level": None, "gpt_oss_reasoning_level": "low"}, "low"),
        ({}, None),
        ({"reasoning_level": None, "gpt_oss_reasoning_level": None}, None),
    ],
)
def test_reasoning_effort_reads_both_spellings_and_never_invents_one(
    config_extra, expected
):
    cfg = OmegaConf.create({"model_type": "openai", "model_name": "m", **config_extra})

    assert ModelBackend._reasoning_effort(cfg) == expected


def test_reasoning_effort_does_not_fall_back_to_high():
    silent = OmegaConf.create({"model_type": "openai", "model_name": "m"})

    assert ModelBackend._reasoning_effort(silent) is None
    assert ModelBackend._reasoning_effort(silent) != "high"


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("auto", "auto"), (3, "3")],
)
def test_optional_str_treats_absent_and_null_alike(value, expected):
    cfg = OmegaConf.create({"present": value})

    assert _optional_str(cfg, "present") == expected
    assert _optional_str(cfg, "never_written") is None


def _batch_config(model_type, **extra):
    return OmegaConf.create(
        {
            "model_type": model_type,
            "model_name": "test-model",
            "max_new_tokens": 100,
            "temperature": 1.0,
            "num_tries": 1,
            "seed": 3,
            **extra,
        }
    )


@pytest.mark.parametrize("model_type", ["openai_batch"])
@pytest.mark.parametrize("batch_root", [None, "", 0])
def test_a_batch_provider_refuses_to_invent_a_batch_root(model_type, batch_root):
    extra = {} if batch_root is None else {"batch_root": batch_root}

    with pytest.raises(ValueError, match="batch_root"):
        ModelBackend(_batch_config(model_type, **extra))


@pytest.mark.parametrize(
    "model_type,endpoint,expected_path,expected_max_tokens",
    [
        ("openai_batch", None, "/v1/responses", False),
        ("openai_batch", "responses", "/v1/responses", False),
        ("openai_batch", "chat_completions", "/v1/chat/completions", False),
    ],
)
def test_batch_endpoint_names_resolve_to_url_paths(
    monkeypatch, tmp_path, model_type, endpoint, expected_path, expected_max_tokens
):
    import ruleforge.judges.openai_batch as openai_batch

    seen = {}

    def fake_make_batch_generate(**kwargs):
        seen.update(kwargs)
        return ("plain", "detailed")

    monkeypatch.setattr(openai_batch, "make_batch_generate", fake_make_batch_generate)

    extra = {"batch_root": str(tmp_path), "reasoning_level": "medium"}
    if endpoint is not None:
        extra["batch_endpoint"] = endpoint

    backend = ModelBackend(_batch_config(model_type, **extra))

    assert seen["endpoint"] == expected_path
    assert seen["send_max_output_tokens"] is expected_max_tokens
    assert seen["root"] == tmp_path
    assert seen["model_name"] == "test-model"
    assert seen["reasoning_effort"] == "medium"
    assert seen["poll_seconds"] == 60.0
    assert seen["timeout_seconds"] is None
    assert seen["tracker"] is backend.token_tracker
    assert (backend._batch_generate, backend._batch_generate_with_details) == (
        "plain",
        "detailed",
    )


@pytest.mark.parametrize("model_type", ["openai_batch"])
def test_a_batch_provider_forwards_the_summary_and_numeric_settings(
    monkeypatch, tmp_path, model_type
):
    import ruleforge.judges.openai_batch as openai_batch

    seen = {}

    def fake_make_batch_generate(**kwargs):
        seen.update(kwargs)
        return (object(), object())

    monkeypatch.setattr(openai_batch, "make_batch_generate", fake_make_batch_generate)

    ModelBackend(
        _batch_config(
            model_type,
            batch_root=str(tmp_path),
            batch_endpoint="responses",
            reasoning_summary="auto",
            batch_poll_seconds=5,
            batch_timeout_seconds=90,
            send_max_output_tokens=True,
        )
    )

    assert seen["reasoning_summary"] == "auto"
    assert seen["send_max_output_tokens"] is True
    assert isinstance(seen["poll_seconds"], float)
    assert seen["poll_seconds"] == 5.0
    assert isinstance(seen["timeout_seconds"], float)
    assert seen["timeout_seconds"] == 90.0


@pytest.mark.parametrize(
    "model_type,endpoint",
    [("openai_batch", "responses"), ("openai_batch", "chat_completions")],
)
def test_a_batch_backend_defines_both_generate_callables(
    tmp_path, model_type, endpoint
):
    backend = ModelBackend(
        _batch_config(
            model_type,
            batch_root=str(tmp_path / "batch"),
            batch_endpoint=endpoint,
            reasoning_level="medium",
        )
    )

    assert callable(backend._batch_generate)
    assert callable(backend._batch_generate_with_details)
    assert backend._batch_generate is not backend._batch_generate_with_details
    assert backend._base_delay == 1.0
    assert not (tmp_path / "batch").exists()


def test_an_unknown_batch_endpoint_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown batch endpoint"):
        ModelBackend(
            _batch_config(
                "openai_batch",
                batch_root=str(tmp_path),
                batch_endpoint="completions",
            )
        )


def test_an_unknown_model_type_is_still_rejected():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        ModelBackend(OmegaConf.create({"model_type": "openai_ish", "model_name": "x"}))


@pytest.mark.parametrize("send_ceiling", [True, False])
def test_the_sync_caller_matches_the_batch_adapter_on_the_output_ceiling(send_ceiling):
    client = MagicMock()
    client.responses.create.return_value = _responses_obj(output_text="no")
    caller = models._make_openai_caller(
        client, "m", "medium", None, send_max_output_tokens=send_ceiling
    )

    caller("system", "", 1.0, 4096, None)
    kwargs = client.responses.create.call_args.kwargs
    assert ("max_output_tokens" in kwargs) is send_ceiling
    if send_ceiling:
        assert kwargs["max_output_tokens"] == 4096


def test_the_sync_caller_and_the_batch_adapter_build_the_same_body():
    from ruleforge.judges.batch_api import RESPONSES_ENDPOINT, build_batch_requests

    client = MagicMock()
    client.responses.create.return_value = _responses_obj(output_text="no")
    models._make_openai_caller(client, "gpt-5.6-luna", "medium", "auto")(
        "judge this", "", 1.0, 65536, None
    )
    sync_body = dict(client.responses.create.call_args.kwargs)

    batch_body = build_batch_requests(
        [{"system_prompt": "judge this", "user_prompt": ""}],
        endpoint=RESPONSES_ENDPOINT,
        model="gpt-5.6-luna",
        temperature=1.0,
        num_responses=1,
        seed=None,
        reasoning_effort="medium",
        reasoning_summary="auto",
        max_output_tokens=None,
    )[0].body

    assert sync_body == batch_body


@pytest.mark.parametrize(
    "error",
    [
        BatchSpendNotConfirmed("pass --yes-spend"),
        BatchExecutionError("status checks in a row failed"),
    ],
    ids=["spend-not-confirmed", "batch-unreadable"],
)
def test_the_retry_loop_does_not_absorb_a_fatal_provider_error(stub_backend, error):
    attempts = {"n": 0}

    def _explode(*_args, **_kwargs):
        attempts["n"] += 1
        raise error

    backend = stub_backend(batch_generate_fn=_explode, num_tries=3)

    with pytest.raises(type(error)):
        backend.generate_with_validation(
            [{"system_prompt": "s", "user_prompt": ""}], ("yes", "no")
        )
    assert attempts["n"] == 1, "a fatal error was retried"


@pytest.mark.parametrize(
    "model_type,save_batch_size,expect_warning",
    [
        ("openai_batch", 10, True),
        ("openai_batch", 500, False),
        ("local", 10, False),
    ],
)
def test_a_serialising_save_batch_size_is_announced_on_a_batch_provider(
    tmp_path, caplog, model_type, save_batch_size, expect_warning
):
    from ruleforge.optimize import judging

    rules = tmp_path / "rules.jsonl"
    rules.write_text(json.dumps({"id": "Rule 1", "rule": "body"}) + "\n")
    scenarios = tmp_path / "scen.jsonl"
    scenarios.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")

    backend = SimpleNamespace(
        model_name="m",
        model_type=model_type,
        seed=16,
        generate_with_validation=lambda prompts, **_: ["no"] * len(prompts),
    )

    with caplog.at_level(logging.WARNING):
        judging.run_judgments(
            scenario_set_path=scenarios,
            rule_set_path=rules,
            backend=backend,
            task_spec=_MinimalSpec(),
            judge_types=["No_Interpretation"],
            evaluation_mode="full",
            output_path=tmp_path / "evaluations.jsonl",
            save_batch_size=save_batch_size,
        )

    assert ("separate batch job" in caplog.text) is expect_warning


def test_a_provider_exception_is_recorded_in_the_trace_for_every_pending_prompt(
    stub_backend,
):
    calls = {"n": 0}

    def _explode(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("provider exploded")

    backend = stub_backend(batch_generate_fn=_explode, num_tries=2)
    judgments, traces = backend.generate_with_validation_and_traces(
        [{"system_prompt": "a", "user_prompt": ""}, {"system_prompt": "b", "user_prompt": ""}],
        ("yes", "no"),
    )

    assert judgments == [None, None]
    assert calls["n"] == 2, "a non-fatal error should still be retried"
    for trace in traces:
        assert len(trace["attempts"]) == 2
        assert all("provider exploded" in a["error"] for a in trace["attempts"])
        assert all(a["candidates"] == [] for a in trace["attempts"])
        assert trace["selected_attempt"] is None


def test_a_custom_response_validator_drives_the_traced_path_too(stub_backend):
    backend = stub_backend(
        batch_generate_fn=lambda *_a, **_k: [['{"full_set_judgment": "yes"}']]
    )

    judgments, traces = backend.generate_with_validation_and_traces(
        [{"system_prompt": "a", "user_prompt": ""}],
        ("yes", "no"),
        response_validator=lambda text: "yes" if "yes" in text else None,
    )

    assert judgments == ["yes"]
    assert traces[0]["selected_attempt"] == 1

    backend = stub_backend(batch_generate_fn=lambda *_a, **_k: [["nonsense"]])
    judgments, _ = backend.generate_with_validation_and_traces(
        [{"system_prompt": "a", "user_prompt": ""}],
        ("yes", "no"),
        response_validator=lambda _text: None,
    )
    assert judgments == [None]


def test_a_custom_response_validator_drives_the_untraced_path_too(stub_backend):
    backend = stub_backend(
        batch_generate_fn=lambda *_a, **_k: [['{"full_set_judgment": "no"}']]
    )
    assert backend.generate_with_validation(
        [{"system_prompt": "a", "user_prompt": ""}],
        ("yes", "no"),
        response_validator=lambda text: "no" if "no" in text else None,
    ) == ["no"]

    backend = stub_backend(batch_generate_fn=lambda *_a, **_k: [["nonsense"]])
    assert backend.generate_with_validation(
        [{"system_prompt": "a", "user_prompt": ""}],
        ("yes", "no"),
        response_validator=lambda _text: None,
    ) == [None]


def test_the_local_closures_forward_every_generation_argument(monkeypatch):
    import ruleforge.judges.models as models_module
    from omegaconf import OmegaConf

    seen = {}

    class _Tok:
        def apply_chat_template(self, *_a, **_k):
            return ""

    monkeypatch.setattr(
        models_module, "init_local_model", lambda **_k: (object(), _Tok())
    )
    monkeypatch.setattr(models_module, "engine_argument_names", frozenset)  # no vLLM here
    monkeypatch.setattr("ruleforge.judges.providers.require", lambda purpose, *modules: None)  # vLLM stubbed here
    monkeypatch.setattr(
        models_module,
        "call_local_model",
        lambda *a, **k: seen.update(plain=k) or [["yes"]],
    )
    monkeypatch.setattr(
        models_module,
        "call_local_model_with_details",
        lambda *a, **k: seen.update(detail=k)
        or [[{"raw_response": "r", "completion": "yes", "finish_reason": "stop"}]],
    )

    cfg = OmegaConf.create(
        {
            "model_type": "local",
            "model_name": "local-test",
            "model_path": "/nonexistent",
            "tp_size": 1,
            "reasoning_parser": None,
            "speculative_method": None,
            "num_speculative_tokens": None,
            "max_model_len": None,
            "attention_backend": None,
            "enable_thinking": False,
            "strip_thinking_model": "qwen3",
            "gpt_oss_reasoning_level": None,
            "max_new_tokens": 128,
            "temperature": 0.7,
            "num_tries": 1,
            "seed": 5,
        }
    )
    backend = ModelBackend(cfg)

    backend._batch_generate([{"system_prompt": "s", "user_prompt": ""}], 0.25, 64, 3, 11)
    assert seen["plain"] == {
        "temperature": 0.25,
        "max_new_tokens": 64,
        "gpt_oss_reasoning_level": None,
        "num_responses": 3,
        "seed": 11,
        "strip_thinking_model": "qwen3",
        "lora_request": None,
    }

    backend._batch_generate_with_details(
        [{"system_prompt": "s", "user_prompt": ""}], 0.25, 64, 3, 11
    )
    assert seen["detail"] == {**seen["plain"], "truncation": None}
    backend._batch_generate_with_details(
        [{"system_prompt": "s", "user_prompt": ""}], 0.25, 64, 3, 11, truncation={"top_k": 64}
    )
    assert seen["detail"]["truncation"] == {"top_k": 64}
