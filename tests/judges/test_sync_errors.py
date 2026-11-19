from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ruleforge.judges import JudgeSpec
from ruleforge.evaluate.validation import EMPTY_RESPONSE, TRANSPORT_ERROR, TRUNCATED, ComponentSchema
from ruleforge.judges import anthropic_batch, models
from ruleforge.judges.batch_api import BatchExecutionError, BatchSpendNotConfirmed

PROMPT = {"system_prompt": "Rule 1: be brief. Does the response break it? Answer yes, no or unsure.",
          "user_prompt": ""}
SCHEMA = ComponentSchema.from_rules([{"id": "Rule 1", "rule": "be brief"}], prompt_style="naive",
                                    forced_json_requirement="label_only")


def _openai(monkeypatch, behaviour):
    client = MagicMock()
    client.responses.create.side_effect = behaviour
    monkeypatch.setattr(models, "init_openai_client", lambda: client)
    return JudgeSpec.openai("m").start()


def _anthropic(monkeypatch, behaviour):
    client = MagicMock()
    client.messages.create.side_effect = behaviour
    monkeypatch.setattr(models, "init_anthropic_client", lambda: client)
    return JudgeSpec.anthropic("m").start()


def _raise(error):
    def behaviour(**kwargs):
        raise error

    return behaviour


BLANK = {
    "openai": lambda **kw: SimpleNamespace(output_text="  ", output=[], usage=None, status="completed"),
    "anthropic": lambda **kw: {"content": [], "stop_reason": "end_turn"},
}
STARTERS = {"openai": _openai, "anthropic": _anthropic}


@pytest.mark.parametrize("provider", list(STARTERS))
def test_a_sync_transport_failure_is_a_transport_error_not_an_empty_answer(provider, monkeypatch):
    judge = STARTERS[provider](monkeypatch, _raise(RuntimeError("rate limited")))
    [outcome] = judge.generate_once([dict(PROMPT)], temperature=1.0)
    assert outcome.error == "RuntimeError: rate limited" and not outcome.block_failed and not outcome.returned
    assert SCHEMA.classify("full_ruleset", outcome).failure_reason == TRANSPORT_ERROR
    assert judge._batch_generate([dict(PROMPT)], 1.0, 16, 1, None) == [[""]]


@pytest.mark.parametrize("provider", list(STARTERS))
def test_a_returned_blank_answer_is_an_empty_response_and_consumes_an_attempt(provider, monkeypatch):
    judge = STARTERS[provider](monkeypatch, BLANK[provider])
    [outcome] = judge.generate_once([dict(PROMPT)], temperature=1.0)
    assert outcome.error is None and outcome.returned and outcome.text is None
    assert SCHEMA.classify("full_ruleset", outcome).failure_reason == EMPTY_RESPONSE


@pytest.mark.parametrize("provider", list(STARTERS))
@pytest.mark.parametrize("error", [BatchSpendNotConfirmed("confirm"), BatchExecutionError("unreadable")])
def test_a_fatal_provider_error_still_propagates(provider, error, monkeypatch):
    judge = STARTERS[provider](monkeypatch, _raise(error))
    with pytest.raises(type(error)):
        judge.generate_once([dict(PROMPT)], temperature=1.0)


def test_the_validated_path_retries_a_failed_sync_call(monkeypatch):
    answers = iter([RuntimeError("blip"), SimpleNamespace(output_text="no", output=[], usage=None,
                                                          status="completed")])

    def behaviour(**kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    judge = _openai(monkeypatch, behaviour)
    judge._base_delay = 0.0
    assert judge.generate_with_validation([dict(PROMPT)], ("yes", "no", "unsure"), num_tries=2) == ["no"]


@pytest.mark.parametrize("response, finish_reason", [
    (SimpleNamespace(output_text="no", output=[], status="completed"), "completed"),
    (SimpleNamespace(output_text="no", output=[], status="incomplete",
                     incomplete_details=SimpleNamespace(reason="max_output_tokens")), "incomplete:max_output_tokens"),
    (SimpleNamespace(output_text="no", output=[], status="incomplete", incomplete_details=None), "incomplete"),
    (SimpleNamespace(output_text="no", output=[]), None),
])
def test_the_responses_finish_reason_is_spelled_as_the_batch_spells_it(response, finish_reason):
    assert models._responses_finish_reason(response) == finish_reason


def test_a_truncated_responses_answer_is_classified_truncated(monkeypatch):
    judge = _openai(monkeypatch, lambda **kw: SimpleNamespace(
        output_text="ye", output=[], usage=None, status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens")))
    [outcome] = judge.generate_once([dict(PROMPT)], temperature=1.0)
    assert SCHEMA.classify("full_ruleset", outcome).failure_reason == TRUNCATED


def test_a_streamed_response_takes_usage_and_finish_reason_from_its_terminal_event():
    usage = SimpleNamespace(input_tokens=3, output_tokens=4)
    incomplete = SimpleNamespace(usage=usage, status="incomplete",
                                 incomplete_details=SimpleNamespace(reason="max_output_tokens"))

    class Stream:
        def __enter__(self):
            return iter([SimpleNamespace(type="response.output_text.delta", delta="ye"),
                         SimpleNamespace(type="response.incomplete", response=incomplete)])

        def __exit__(self, *exc):
            return False

    client = MagicMock()
    client.responses.stream.return_value = Stream()
    assert models._stream_responses_detail(client, {}) == ("ye", usage, "incomplete:max_output_tokens", [], None)


def test_the_anthropic_caller_sends_the_batch_s_request_and_reads_its_answer(caplog):
    client = MagicMock()
    client.messages.create.return_value = {
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": " no "}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 11, "output_tokens": 5}}
    tracker = SimpleNamespace(seen=[], add=lambda usage: tracker.seen.append(usage))
    caller = models._make_anthropic_caller(client, "claude-haiku-5-5", "medium", tracker=tracker)
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.models"):
        assert caller("system", "", 1.0, 64, 7) == {
            "raw_response": "no", "completion": "no", "finish_reason": "end_turn", "error": None}
        caller("system", "", 1.0, 64, 8)
    assert client.messages.create.call_args.kwargs == anthropic_batch.build_request_params(
        "system", "", model="claude-haiku-5-5", max_tokens=64, effort="medium")
    assert [(usage.prompt_tokens, usage.completion_tokens) for usage in tracker.seen] == [(11, 5), (11, 5)]
    assert caplog.text.count("has no seed parameter") == 1


@pytest.mark.parametrize("message, completion, finish_reason", [
    ({"content": [{"type": "text", "text": "yes"}], "stop_reason": "refusal",
      "stop_details": {"category": "cyber"}}, "", "refusal:cyber"),
    ({"content": [{"type": "text", "text": "{\"full"}], "stop_reason": "max_tokens"}, "{\"full", "max_tokens"),
])
def test_an_anthropic_refusal_or_ceiling_is_read_as_the_batch_reads_it(message, completion, finish_reason):
    client = MagicMock()
    client.messages.create.return_value = message
    detail = models._make_anthropic_caller(client, "m")("s", "", 1.0, 64, None)
    assert (detail["completion"], detail["finish_reason"], detail["error"]) == (completion, finish_reason, None)


def test_an_anthropic_temperature_it_cannot_send_is_refused_before_the_call():
    client = MagicMock()
    with pytest.raises(ValueError, match="sends no temperature"):
        models._make_anthropic_caller(client, "m")("s", "", 0.5, 64, None)
    client.messages.create.assert_not_called()


def test_the_anthropic_client_is_the_batch_transport_s(monkeypatch):
    monkeypatch.setattr(anthropic_batch, "default_client", lambda: "client")
    assert models.init_anthropic_client() == "client"


class _Stream:
    """The SDK's Responses stream handle: a context manager over its events."""

    def __init__(self, events):
        self.events = list(events)

    def __enter__(self):
        return iter(self.events)

    def __exit__(self, *exc):
        return False


def _streamed_openai(monkeypatch, events):
    client = MagicMock()
    client.responses.stream.side_effect = lambda **kw: _Stream(events)
    monkeypatch.setattr(models, "init_openai_client", lambda: client)
    return JudgeSpec.openai("m", stream=True).start()


DELTA = SimpleNamespace(type="response.output_text.delta", delta="yes")
FAILED_RESPONSE = SimpleNamespace(status="failed", usage=SimpleNamespace(input_tokens=5, output_tokens=1),
                                  error=SimpleNamespace(code="server_error", message="upstream died"))

NO_ANSWER = {
    "responses-status-failed": (
        _openai, lambda **kw: SimpleNamespace(output_text="yes", output=[], usage=None, status="failed",
                                              error=SimpleNamespace(code="server_error", message="boom")),
        "CallFailed: the response's status is 'failed' (server_error: boom)"),
    "responses-status-cancelled": (
        _openai, lambda **kw: SimpleNamespace(output_text="yes", output=[], usage=None, status="cancelled",
                                              error=None),
        "CallFailed: the response's status is 'cancelled'"),
    "stream-response-failed": (
        _streamed_openai, [DELTA, SimpleNamespace(type="response.failed", response=FAILED_RESPONSE)],
        "CallFailed: the response's status is 'failed' (server_error: upstream died)"),
    "stream-response-failed-bare": (
        _streamed_openai, [DELTA, SimpleNamespace(type="response.failed", response=None)],
        "CallFailed: the stream reported response.failed"),
    "stream-error-event": (
        _streamed_openai, [DELTA, SimpleNamespace(type="error", code="rate_limit_exceeded", message="slow down")],
        "CallFailed: the stream sent an error event (rate_limit_exceeded: slow down)"),
    "stream-error-event-bare": (
        _streamed_openai, [DELTA, SimpleNamespace(type="error")], "CallFailed: the stream sent an error event"),
    "stream-cut-short": (
        _streamed_openai, [DELTA], "CallFailed: the stream ended before response.completed or response.incomplete"),
}


@pytest.mark.parametrize("case", list(NO_ANSWER))
def test_a_call_that_returned_without_an_answer_is_a_transport_error(case, monkeypatch):
    start, sent, error = NO_ANSWER[case]
    judge = start(monkeypatch, sent)
    [outcome] = judge.generate_once([dict(PROMPT)], temperature=1.0)
    assert outcome.error == error and not outcome.returned
    assert SCHEMA.classify("full_ruleset", outcome).failure_reason == TRANSPORT_ERROR
    assert judge._batch_generate([dict(PROMPT)], 1.0, 16, 1, None) == [[""]]


def test_a_failed_response_s_reported_usage_is_still_counted(monkeypatch):
    judge = _streamed_openai(monkeypatch, [DELTA, SimpleNamespace(type="response.failed", response=FAILED_RESPONSE)])
    judge.generate_once([dict(PROMPT)], temperature=1.0)
    assert (judge.usage.prompt_tokens, judge.usage.completion_tokens) == (5, 1)


def test_a_streamed_answer_that_completes_is_returned(monkeypatch):
    completed = SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None, status="completed"))
    [outcome] = _streamed_openai(monkeypatch, [DELTA, completed]).generate_once([dict(PROMPT)], temperature=1.0)
    assert (outcome.text, outcome.finish_reason, outcome.error) == ("yes", "completed", None)
