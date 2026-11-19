from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import logging

import pytest
from omegaconf import OmegaConf

from ruleforge.judges.backend import ModelBackend


@pytest.mark.parametrize(
    "prompts,match",
    [
        (["not-a-dict"], "must be a dictionary"),
        ([{"system_prompt": 1, "user_prompt": "user"}], "must be strings"),
        ([{"system_prompt": "system", "user_prompt": 1}], "must be strings"),
        ([{"system_prompt": " ", "user_prompt": "\t"}], "nonblank content"),
        ([{}], "nonblank content"),
    ],
)
def test_backend_rejects_malformed_prompts_without_provider_calls(prompts, match):
    backend = object.__new__(ModelBackend)
    backend.fixed_temperature = None
    backend.max_retry_seconds = None
    backend._batch_generate = MagicMock()
    backend.temperature = 1.0
    backend.max_new_tokens = 10
    backend.seed = 1
    backend.num_tries = 2

    with pytest.raises(ValueError, match=match):
        backend.generate(prompts)

    backend._batch_generate.assert_not_called()


def _deadline_backend():
    backend = object.__new__(ModelBackend)
    backend.fixed_temperature = None
    backend.max_retry_seconds = None
    backend.model_name = "test"
    backend.temperature = 1.0
    backend.max_new_tokens = 10
    backend.seed = 1
    backend.num_tries = 5
    backend.fixed_temperature = None
    backend.max_retry_seconds = 1.0
    backend._base_delay = 0.0
    backend._batch_generate = MagicMock(return_value=[[""]])
    return backend


def test_backend_retry_deadline_stops_before_backoff(monkeypatch):
    backend = _deadline_backend()
    monkeypatch.setattr(
        "ruleforge.judges.backend.time.monotonic",
        MagicMock(side_effect=[0.0, 2.0]),
    )

    result = backend.generate([{"system_prompt": "system", "user_prompt": ""}])

    assert result == [None]
    assert backend._batch_generate.call_count == 1


def test_backend_retry_deadline_stops_after_backoff(monkeypatch):
    backend = _deadline_backend()
    backend._backoff = MagicMock()
    monkeypatch.setattr(
        "ruleforge.judges.backend.time.monotonic",
        MagicMock(side_effect=[0.0, 0.5, 2.0]),
    )

    result = backend.generate([{"system_prompt": "system", "user_prompt": ""}])

    assert result == [None]
    assert backend._backoff.call_count == 1
    assert backend._batch_generate.call_count == 1


def test_backend_rejects_nonpositive_retry_budget():
    config = OmegaConf.create(
        {
            "model_type": "unsupported",
            "model_name": "test",
            "max_retry_seconds": 0,
        }
    )
    with pytest.raises(ValueError, match="max_retry_seconds must be positive"):
        ModelBackend(config)


def test_fixed_temperature_applies_to_judgment_and_json_paths(stub_backend):
    temperatures = []
    responses = iter([["yes"], ['{"value": 1}']])

    def batch_generate(prompts, temperature, max_tokens, num_responses, seed):
        temperatures.append(temperature)
        return [next(responses)]

    backend = stub_backend(
        batch_generate, temperature=1.0, fixed_temperature=1.0, num_tries=1
    )
    prompts = [{"system_prompt": "system", "user_prompt": ""}]

    assert backend.generate_with_validation(
        prompts, valid_responses=("yes", "no"), temperature=0.1
    ) == ["yes"]
    assert backend.generate_with_json_validation(prompts, temperature=0.0) == [
        {"value": 1}
    ]
    assert temperatures == [1.0, 1.0]


def test_an_overridden_temperature_is_announced(stub_backend, caplog):
    backend = stub_backend(lambda *_a, **_k: [["yes"]], fixed_temperature=1.0)
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.backend"):
        backend.generate_with_validation(
            [{"system_prompt": "s", "user_prompt": ""}],
            valid_responses=("yes",),
            temperature=0.0,
        )
    assert "forces temperature=1.0" in caplog.text
    assert "will not get it" in caplog.text


def test_generate_announces_an_overridden_temperature_too(stub_backend, caplog):
    backend = stub_backend(lambda *_a, **_k: [["out"]], fixed_temperature=1.0)
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.backend"):
        backend.generate(
            [{"system_prompt": "s", "user_prompt": ""}], temperature=0.2
        )
    assert "forces temperature=1.0" in caplog.text


def test_no_warning_when_the_caller_agrees_with_the_pin(stub_backend, caplog):
    backend = stub_backend(lambda *_a, **_k: [["yes"]], fixed_temperature=1.0)
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.backend"):
        backend.generate_with_validation(
            [{"system_prompt": "s", "user_prompt": ""}],
            valid_responses=("yes",),
            temperature=1.0,
        )
    assert "forces temperature" not in caplog.text


def test_a_slow_first_attempt_does_not_consume_the_retry_budget(
    stub_backend, monkeypatch
):
    """The retry budget bounds retries, not the initial call."""
    now = [0.0]
    attempts = []

    def batch_generate(batch, *_args, **_kwargs):
        attempts.append(len(batch))
        if len(attempts) == 1:
            now[0] += 600.0  # the first pass burns ten times the budget
            return [["no"]]
        return [["yes"]]

    backend = stub_backend(batch_generate, max_retry_seconds=60.0)
    monkeypatch.setattr("ruleforge.judges.backend.time.monotonic", lambda: now[0])
    monkeypatch.setattr("ruleforge.judges.backend.ModelBackend._backoff", lambda *_: None)

    result = backend.generate_with_validation(
        [{"system_prompt": "s", "user_prompt": ""}],
        valid_responses=("yes",),
        num_tries=3,
    )

    assert len(attempts) == 2, "a slow first attempt must not cancel the retry"
    assert result == ["yes"]


def test_the_budget_still_stops_retries_once_it_is_armed(stub_backend, monkeypatch):
    attempts = []

    def batch_generate(batch, *_args, **_kwargs):
        attempts.append(len(batch))
        return [["no"]]

    backend = stub_backend(batch_generate, max_retry_seconds=10.0)
    # First attempt is instant; the budget then expires before attempt 2.
    clock = iter([0.0, 0.0, 999.0])
    monkeypatch.setattr(
        "ruleforge.judges.backend.time.monotonic", lambda: next(clock, 999.0)
    )
    monkeypatch.setattr("ruleforge.judges.backend.ModelBackend._backoff", lambda *_: None)

    backend.generate_with_validation(
        [{"system_prompt": "s", "user_prompt": ""}],
        valid_responses=("yes",),
        num_tries=5,
    )
    assert len(attempts) == 1


def test_an_unsupported_model_type_is_rejected():
    from omegaconf import OmegaConf

    from ruleforge.judges.backend import ModelBackend

    with pytest.raises(ValueError, match="Unsupported model_type"):
        ModelBackend(OmegaConf.create({"model_type": "telepathy", "model_name": "x"}))


@pytest.mark.parametrize("usage, expected", [
    (SimpleNamespace(prompt_token_count=11, candidates_token_count=5,
                     total_token_count=16), (11, 5, 16)),
    (SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
     (7, 3, 10)),
    # Anthropic-shaped, no total
    (SimpleNamespace(input_tokens=4, output_tokens=6), (4, 6, 10)),
])
def test_token_usage_reads_every_provider_spelling(usage, expected):
    """A missing attribute reads as 0 rather than raising."""
    from ruleforge.judges.token_usage import TokenUsage

    parsed = TokenUsage.from_response(usage)
    assert (parsed.prompt_tokens, parsed.completion_tokens,
            parsed.total_tokens) == expected


def _answering(stub_backend, *answers):
    """A judge whose engine gives ``answers`` in turn, each a (text, finish_reason); ``calls`` counts the calls."""
    calls = []

    def details(prompts, temperature, max_new_tokens, num_responses, seed):
        text, reason = answers[len(calls)]
        calls.append(seed)
        return [[{"raw_response": text, "completion": text, "finish_reason": reason}] for _ in prompts]

    backend = stub_backend(lambda *args: pytest.fail("the texts alone carry no finish reason"),
                           _batch_generate_with_details=details)
    backend.calls = calls
    return backend


PANEL_PROMPT = [{"system_prompt": "judge", "user_prompt": ""}]


@pytest.mark.parametrize("finish_reason", ["length", "incomplete:max_output_tokens", "max_tokens"])
@pytest.mark.parametrize("traced", [False, True])
def test_a_judgment_cut_off_at_the_output_ceiling_is_asked_again_as_a_campaign_rejects_it(stub_backend,
                                                                                          finish_reason, traced):
    backend = _answering(stub_backend, ("Rule 10: be warm.\n    *   Yes.\n\n    *", finish_reason), ("no", "stop"))
    if traced:
        results, traces = backend.generate_with_validation_and_traces(PANEL_PROMPT, ("yes", "no", "unsure"))
        assert [[c["accepted"] for c in a["candidates"]] for a in traces[0]["attempts"]] == [[False], [True]]
    else:
        results = backend.generate_with_validation(PANEL_PROMPT, ("yes", "no", "unsure"))
    assert results == ["no"] and len(backend.calls) == 2


def test_a_json_judgment_cut_off_at_the_output_ceiling_is_asked_again(stub_backend):
    backend = _answering(stub_backend, ('{"score": 1} and then', "length"), ('{"score": 2}', "stop"))
    assert backend.generate_with_json_validation(PANEL_PROMPT) == [{"score": 2}] and len(backend.calls) == 2


def test_a_free_form_answer_cut_off_is_the_caller_s_to_judge(stub_backend):
    backend = _answering(stub_backend, ("a long draft", "length"))
    assert backend.generate(PANEL_PROMPT) == [["a long draft"]] and len(backend.calls) == 1


def test_a_retry_keeps_no_more_candidates_than_a_prompt_still_needs(stub_backend):
    # Attempt 1 answers prompt 0 once and prompt 1 not at all; attempt 2 asks both for two, prompt 1's need.
    script = [[["A", ""], ["", ""]], [["A2", "A3"], ["B1", "B2"]]]
    calls = []

    def details(prompts, temperature, max_new_tokens, num_responses, seed):
        calls.append(num_responses)
        rows = script[len(calls) - 1]
        return [[{"raw_response": text, "completion": text, "finish_reason": "stop"} for text in row] for row in rows]

    backend = stub_backend(_batch_generate_with_details=details)
    prompts = [{"system_prompt": "a", "user_prompt": ""}, {"system_prompt": "b", "user_prompt": ""}]
    results, traces = backend.generate_with_traces(prompts, num_responses=2)
    assert results == [["A", "A2"], ["B1", "B2"]] and calls == [2, 2]
    accepted = [[c["completion"] for a in t["attempts"] for c in a["candidates"] if c["accepted"]] for t in traces]
    assert accepted == [["A", "A2"], ["B1", "B2"]]
    assert [c["result_index"] for a in traces[0]["attempts"] for c in a["candidates"] if c["accepted"]] == [0, 1]


def test_a_json_judgment_that_is_a_list_is_kept_whole(stub_backend):
    backend = _answering(stub_backend, ('```json\n[1, 2, 3]\n```', "stop"))
    assert backend.generate_with_json_validation(PANEL_PROMPT) == [[1, 2, 3]]
