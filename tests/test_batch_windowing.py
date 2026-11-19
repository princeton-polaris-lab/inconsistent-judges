from __future__ import annotations

import sys
import types

import pytest

from ruleforge.judges.batch_api import (
    BatchExecutionError,
    BatchRequest,
    CHAT_COMPLETIONS_ENDPOINT,
    RESPONSES_ENDPOINT,
    estimate_enqueued_tokens,
    _request_input_text,
)
from ruleforge.judges.openai_batch import make_windowed_block_ops

from tests.test_openai_batch import (
    FakeOpenAI,
    build_runner,
    make_requests,
)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def confirmed(monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)


def test_check_is_false_while_running_then_true_once_completed(tmp_path, confirmed):
    client = FakeOpenAI(status_script=["in_progress"])
    runner = build_runner(tmp_path, client)
    requests = make_requests()
    runner.submit(requests)
    assert runner.check(requests) is False
    assert runner.check(requests) is True


def test_check_raises_when_a_shard_ends_in_a_terminal_failure(tmp_path, confirmed):
    client = FakeOpenAI(status_script=["failed"])
    runner = build_runner(tmp_path, client)
    requests = make_requests()
    runner.submit(requests)
    with pytest.raises(BatchExecutionError, match="non-completed terminal"):
        runner.check(requests)


def test_check_refuses_a_shard_that_was_never_submitted(tmp_path):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client)
    requests = make_requests()
    runner.prepare(requests)
    with pytest.raises(BatchExecutionError, match="never submitted"):
        runner.check(requests)


def test_check_tolerates_a_transient_status_error_as_still_running(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client)
    requests = make_requests()
    runner.submit(requests)

    calls = {"n": 0}
    real_retrieve = client.batches.retrieve

    def flaky(batch_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 slow down")
        return real_retrieve(batch_id)

    client.batches.retrieve = flaky
    assert runner.check(requests) is False
    assert runner.check(requests) is True


def _ops(tmp_path, client):
    return make_windowed_block_ops(
        model_name="test-model",
        root=tmp_path,
        endpoint=RESPONSES_ENDPOINT,
        client_factory=lambda: client,
    )


@pytest.fixture(autouse=True)
def offline_encoding(monkeypatch):
    """A stand-in ``tiktoken`` whose encoding counts words."""
    requested = []

    def get_encoding(name):
        requested.append(name)
        return types.SimpleNamespace(encode=str.split)

    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=get_encoding))
    return requested


PROMPTS = [{"system_prompt": "judge this"}, {"system_prompt": "and this one too"}]
BLOCK = "camp|cond|d0|a0|b0"


def test_estimate_returns_positive_input_tokens_without_touching_the_api(tmp_path):
    estimate, _submit, _poll = _ops(tmp_path, client=None)
    tokens = estimate(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=None, request_block_id=BLOCK)
    assert tokens > 0


def test_seeds_on_the_responses_endpoint_are_ignored_with_a_warning(tmp_path):
    estimate, _submit, _poll = _ops(tmp_path, client=None)
    with_seeds = estimate(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=[1, 2], request_block_id=BLOCK)
    without = estimate(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=None, request_block_id=BLOCK)
    assert with_seeds == without


def test_submit_then_poll_returns_details_once_the_batch_completes(tmp_path, confirmed):
    client = FakeOpenAI(status_script=["in_progress"])
    estimate, submit, poll = _ops(tmp_path, client)
    submit(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=None, request_block_id=BLOCK)
    assert len(client.create_calls) == 1
    assert poll(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=None, request_block_id=BLOCK) is None
    details = poll(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=None, request_block_id=BLOCK)
    assert details is not None
    assert len(details) == len(PROMPTS)
    assert all("completion" in d and "provider_request_ref" in d for d in details)


def test_windowed_submit_addresses_the_same_batch_as_the_blocking_path(tmp_path, confirmed):
    from ruleforge.judges.openai_batch import OpenAIBatchRunner, build_block_requests

    _namespace, requests = build_block_requests(
        PROMPTS, 1.0, None, None, BLOCK,
        endpoint=RESPONSES_ENDPOINT, model="test-model",
        reasoning_effort=None, reasoning_summary=None, send_max_output_tokens=False,
    )
    runner = OpenAIBatchRunner(root=tmp_path, client_factory=lambda: FakeOpenAI())
    blocking_dir = runner.workdir(requests)

    client = FakeOpenAI()
    _estimate, submit, _poll = _ops(tmp_path, client)
    submit(PROMPTS, temperature=1.0, max_new_tokens=None, seeds=None, request_block_id=BLOCK)
    assert (blocking_dir / "state.json").exists()


def test_estimate_enqueued_tokens_sums_over_the_request_set(offline_encoding):
    requests = [
        BatchRequest(custom_id="a", body={"input": [{"role": "user", "content": "one two three"}]}),
        BatchRequest(custom_id="b", body={"input": [{"role": "user", "content": "four five"}]}),
    ]
    total = estimate_enqueued_tokens(requests, endpoint=RESPONSES_ENDPOINT)
    assert total == 5  # the stand-in encoding counts words: 3 + 2
    assert offline_encoding == ["o200k_base"]


@pytest.mark.parametrize(
    "body, endpoint, expected_substrings",
    [
        ({"input": "a bare string prompt"}, RESPONSES_ENDPOINT, ["bare string"]),
        ({"input": [{"role": "user", "content": "hello there"}]}, RESPONSES_ENDPOINT, ["hello there"]),
        (
            {"input": [{"role": "user", "content": [{"type": "input_text", "text": "part one"}, {"text": "part two"}]}]},
            RESPONSES_ENDPOINT,
            ["part one", "part two"],
        ),
        ({"messages": [{"role": "system", "content": "chat sys"}]}, CHAT_COMPLETIONS_ENDPOINT, ["chat sys"]),
        ({"input": [42, "loose"]}, RESPONSES_ENDPOINT, ["42", "loose"]),
        ({"input": 12345}, RESPONSES_ENDPOINT, [""]),  # neither str nor list -> empty
    ],
)
def test_request_input_text_extracts_prompt_text(body, endpoint, expected_substrings):
    text = _request_input_text(body, endpoint)
    for substring in expected_substrings:
        assert substring in text
