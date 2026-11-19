from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import anthropic_batch
from ruleforge.judges.anthropic_batch import (
    ANTHROPIC_BATCH_ENDPOINT,
    DEFAULT_MAX_SHARD_BYTES,
    NO_TEXT,
    PART_COMPLETED,
    PART_INCOMPLETE,
    STATE_VERSION,
    AnthropicBatchRunner,
    add_usage,
    build_request_params,
    build_requests,
    check_custom_id,
    check_temperature,
    create_was_refused,
    effort_level,
    fingerprint,
    make_batch_generate,
    make_block_generate,
    outcome_from_message,
    outcome_from_record,
    part_state,
    prompt_of_params,
    request_line,
    result_record,
    write_shards,
)
from ruleforge.judges.backend import BATCH_MODEL_TYPES, SEED_UNSUPPORTED, ModelBackend
from ruleforge.judges.batch_api import (
    BatchAlreadyClaimed,
    BatchExecutionError,
    BatchRequest,
    BatchSpendNotConfirmed,
    BatchUsage,
    is_truncated,
)
from ruleforge.judges.models import EMPTY_USER_TURN
from ruleforge.judges.token_usage import TokenUsageTracker

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-haiku-5-5"
PROMPTS = [{"system_prompt": f"judge {i}", "user_prompt": ""} for i in range(3)]
#: A usage block whose every bucket is distinguishable (Anthropic's input_tokens is the uncached remainder).
USAGE = {"input_tokens": 11, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 2, "output_tokens": 30,
         "output_tokens_details": {"thinking_tokens": 20}, "service_tier": "batch"}
SENTINEL_KEY = "sk-ant-test-SENTINEL-not-a-key"


def text(value: str) -> Dict[str, Any]:
    return {"type": "text", "text": value}


def thinking() -> Dict[str, Any]:
    """A thinking block as the omitted display returns it: empty text, a signature."""
    return {"type": "thinking", "thinking": "", "signature": "sig"}


def message(*blocks, stop: Optional[str] = "end_turn", usage=USAGE, **extra) -> Dict[str, Any]:
    body: Dict[str, Any] = {"id": "msg_1", "type": "message", "role": "assistant", "model": MODEL,
                            "content": list(blocks), "stop_reason": stop, "stop_sequence": None, **extra}
    if usage is not None:
        body["usage"] = usage
    return body


def succeeded(custom_id: str, *blocks, **kwargs) -> Dict[str, Any]:
    return {"custom_id": custom_id, "result": {"type": "succeeded", "message": message(*blocks, **kwargs)}}


def success(custom_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """An answer naming its custom id, after a thinking block."""
    return succeeded(custom_id, thinking(), text(f"answer::{custom_id}"))


def failed(custom_id: str, kind: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"type": kind}
    if kind == "errored":
        result["error"] = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    return {"custom_id": custom_id, "result": result}


def sdk_result(record: Dict[str, Any]) -> Any:
    """A result line as the SDK's results() decoder hands it back."""
    from anthropic._models import construct_type_unchecked
    from anthropic.types.messages import MessageBatchIndividualResponse

    return construct_type_unchecked(value=record, type_=MessageBatchIndividualResponse)


def sdk_batch(batch_id: str, status: str, counts: Dict[str, int], *, ended: bool = False) -> Any:
    from anthropic.types.messages import MessageBatch

    return MessageBatch.model_validate({
        "id": batch_id, "type": "message_batch", "processing_status": status, "request_counts": counts,
        "created_at": "2026-10-13T00:00:00Z", "expires_at": "2026-10-14T00:00:00Z", "archived_at": None,
        "cancel_initiated_at": None, "ended_at": "2026-10-13T01:00:00Z" if ended else None,
        "results_url": f"https://api.example/v1/messages/batches/{batch_id}/results" if ended else None})


class FakeAnthropic:
    """Scriptable stand-in for ``anthropic.Anthropic``."""

    def __init__(self, *, statuses: Optional[List[Any]] = None,
                 responder: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]] = success,
                 shuffle: bool = False, extra_lines: tuple = (), create_failures: Optional[List[Any]] = None,
                 created_id: Optional[str] = "auto") -> None:
        self.batches: Dict[str, List[Dict[str, Any]]] = {}
        self.create_calls: List[List[Dict[str, Any]]] = []
        self.option_calls: List[Dict[str, Any]] = []
        self.retrieve_calls: List[str] = []
        self.results_calls: List[str] = []
        self.statuses = list(statuses or [])
        self.responder = responder
        self.shuffle = shuffle
        self.extra_lines = list(extra_lines)
        self.create_failures = list(create_failures or [])
        self.created_id = created_id
        self.messages = SimpleNamespace(batches=SimpleNamespace(
            create=self._create, retrieve=self._retrieve, results=self._results))

    def with_options(self, **options):
        self.option_calls.append(options)
        return self

    def _create(self, *, requests):
        requests = json.loads(json.dumps(list(requests)))
        failure = self.create_failures.pop(0) if self.create_failures else None
        self.create_calls.append(requests)
        if failure is not None and failure[0] == "before":
            raise failure[1]
        batch_id = f"msgbatch_{len(self.batches):02d}"
        self.batches[batch_id] = requests
        if failure is not None:
            raise failure[1]
        counts = {"processing": len(requests), "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        if self.created_id != "auto":
            return SimpleNamespace(id=self.created_id, processing_status="in_progress")
        return sdk_batch(batch_id, "in_progress", counts)

    def records(self, batch_id: str) -> List[Dict[str, Any]]:
        records = [self.responder(request["custom_id"], request["params"]) for request in self.batches[batch_id]]
        records = [record for record in records if record is not None]
        if self.shuffle:
            records.reverse()
        return records + self.extra_lines

    def _retrieve(self, batch_id):
        self.retrieve_calls.append(batch_id)
        step = self.statuses.pop(0) if self.statuses else "ended"
        if isinstance(step, BaseException):
            raise step
        if not isinstance(step, str):
            return step
        counts = {"processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        if step != "ended":
            counts["processing"] = len(self.batches[batch_id])
            return sdk_batch(batch_id, step, counts)
        for record in self.records(batch_id):
            kind = record.get("result", {}).get("type")
            if kind in counts:
                counts[kind] += 1
        return sdk_batch(batch_id, "ended", counts, ended=True)

    def _results(self, batch_id):
        self.results_calls.append(batch_id)
        return iter([sdk_result(record) for record in self.records(batch_id)])


def requests_for(prompts=PROMPTS, **overrides) -> List[BatchRequest]:
    kwargs = dict(model=MODEL, max_tokens=16384, num_responses=1, seed=5, effort="medium")
    kwargs.update(overrides)
    return build_requests(prompts, **kwargs)


def make_runner(tmp_path: Path, client: Any = None, **kwargs) -> AnthropicBatchRunner:
    """A runner over ``client``; with none, any API call fails the test."""

    def factory():
        if client is None:
            pytest.fail("this path must make no API call")
        return client

    kwargs.setdefault("sleep", lambda seconds: None)
    return AnthropicBatchRunner(root=tmp_path / "batch", model=MODEL, client_factory=factory, **kwargs)


@pytest.fixture(autouse=True)
def _no_spend_no_key_no_gate(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", *anthropic_batch.REFUSED_SDK_ENVS):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def confirmed(monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)


def test_a_request_carries_the_model_the_cap_the_prompt_thinking_and_effort_and_nothing_else():
    """No temperature, top_p, top_k, seed, stop sequence, prefill, cache_control or fallbacks."""
    assert build_request_params("sys", "", model=MODEL, max_tokens=16384, effort="medium") == {
        "model": MODEL, "max_tokens": 16384, "system": "sys", "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
        "messages": [{"role": "user", "content": EMPTY_USER_TURN}],
    }
    assert build_request_params("", "u", model=MODEL, max_tokens=7) == {
        "model": MODEL, "max_tokens": 7, "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": "u"}]}
    params = build_request_params("s", "u", model=MODEL, max_tokens=1, effort="low")
    assert params["messages"][-1]["role"] == "user" and len(params["messages"]) == 1


@pytest.mark.parametrize("system_prompt, user_prompt", [("s", ""), ("s", "u"), ("", "u"), ("s", "  ")])
def test_a_request_reads_back_as_the_prompt_it_was_built_from(system_prompt, user_prompt):
    params = build_request_params(system_prompt, user_prompt, model=MODEL, max_tokens=5)
    expected_user = user_prompt if user_prompt.strip() else ""
    assert prompt_of_params(params) == {"system_prompt": system_prompt, "user_prompt": expected_user}


def test_the_request_bytes_and_their_address_are_pinned():
    requests = build_requests([{"system_prompt": "judge é 0", "user_prompt": ""},
                               {"system_prompt": "judge 1", "user_prompt": "u"}],
                              model=MODEL, max_tokens=16384, num_responses=1, seed=None, effort="medium",
                              namespace="ns0123456789abcdef")
    assert request_line(requests[0]) == (
        b'{"custom_id":"ns0123456789abcdef-p000000-c00","params":{"max_tokens":16384,"messages":[{"content":'
        b'"Respond according to the system instruction.","role":"user"}],"model":"claude-haiku-5-5","output_config":'
        b'{"effort":"medium"},"system":"judge \xc3\xa9 0","thinking":{"type":"adaptive"}}}\n')
    assert request_line(requests[1]) == (
        b'{"custom_id":"ns0123456789abcdef-p000001-c00","params":{"max_tokens":16384,"messages":[{"content":"u",'
        b'"role":"user"}],"model":"claude-haiku-5-5","output_config":{"effort":"medium"},"system":"judge 1",'
        b'"thinking":{"type":"adaptive"}}}\n')
    assert fingerprint(requests) == "3bc109e674b7d156a17da6ef9a8901eb"
    assert fingerprint(requests[:1]) != fingerprint(requests) != fingerprint(list(reversed(requests)))


def test_build_requests_keys_by_seed_or_namespace_and_sends_neither():
    keyed = requests_for(num_responses=2)
    assert [r.custom_id for r in keyed[:2]] == ["s5-p000000-c00", "s5-p000000-c01"]
    assert all("seed" not in json.dumps(r.body) for r in keyed)
    assert keyed[0].body is keyed[1].body
    spaced = requests_for(namespace="nsabc")
    assert [r.custom_id for r in spaced] == [f"nsabc-p{i:06d}-c00" for i in range(3)]
    with pytest.raises(ValueError, match="max_tokens must be at least 1"):
        requests_for(max_tokens=0)


@pytest.mark.parametrize("custom_id", ["a" * 65, "", "ns/x-p000000-c00", "s1.5-p0", "ns x", "é"])
def test_a_custom_id_the_api_does_not_take_is_refused_before_any_spend(custom_id):
    with pytest.raises(ValueError, match="is not one the Message Batches API takes"):
        check_custom_id(custom_id)


def test_every_custom_id_the_repo_writes_is_one_the_api_takes_unchanged():
    from ruleforge.core.digests import request_namespace
    from ruleforge.judges.batch_api import batch_custom_id

    namespaces = [None, request_namespace("camp/model/style/draw-004/attempt-19/block-0099"),
                  request_namespace("pilot-haiku-v2-2026-10/claude-haiku-5-5/forced/draw-000/attempt-0/"
                                    "block-0000")]
    seen = set()
    for namespace in namespaces:
        for seed in (None, 0, 16, -5, 2**31 - 1, -(2**63)):
            for prompt_index in (0, 999_999, 12_345_678):
                for candidate_index in (0, 99):
                    custom_id = batch_custom_id(prompt_index, candidate_index, seed, namespace)
                    assert check_custom_id(custom_id) == custom_id
                    seen.add(custom_id)
    assert len(seen) == len(namespaces[1:]) * 3 * 2 + 6 * 3 * 2


@pytest.mark.parametrize("value, expected", [(None, None), ("medium", "medium"), (" HIGH ", "high"), ("max", "max")])
def test_effort_level_normalizes_a_config_value(value, expected):
    assert effort_level(value) == expected


def test_effort_level_refuses_a_level_the_model_does_not_take():
    with pytest.raises(ValueError, match="unknown Claude effort level"):
        effort_level("minimal")


def test_a_temperature_other_than_the_providers_is_refused():
    check_temperature(1.0)
    check_temperature(1)
    with pytest.raises(ValueError, match="sends no temperature"):
        check_temperature(0.7)


def test_write_shards_splits_by_requests_and_by_bytes_and_refuses_what_cannot_be_sent(tmp_path):
    requests = requests_for(prompts=[{"system_prompt": f"p{i}", "user_prompt": ""} for i in range(5)])
    shards = write_shards(requests, tmp_path / "a", max_requests=2)
    assert [s["requests"] for s in shards] == [2, 2, 1]
    line = len(request_line(requests[0]))
    assert [s["requests"] for s in write_shards(requests, tmp_path / "b", max_bytes=2 * line + 1)] == [2, 2, 1]
    assert b"".join((tmp_path / "a" / s["path"]).read_bytes() for s in shards) == b"".join(map(request_line, requests))
    with pytest.raises(ValueError, match="no requests"):
        write_shards([], tmp_path / "c")
    with pytest.raises(ValueError, match="over the 10-byte batch limit"):
        write_shards(requests, tmp_path / "d", max_bytes=10)
    with pytest.raises(ValueError, match="appears twice"):
        write_shards([requests[0], requests[0]], tmp_path / "e")
    with pytest.raises(ValueError, match="is not one the Message Batches API takes"):
        write_shards([BatchRequest(custom_id="no/slash", body=requests[0].body)], tmp_path / "f")
    assert DEFAULT_MAX_SHARD_BYTES < 256_000_000 and anthropic_batch.DEFAULT_MAX_SHARD_REQUESTS < 100_000


def test_a_shard_is_sized_as_the_wire_may_carry_it_every_non_ascii_character_escaped(tmp_path):
    wide = requests_for(prompts=[{"system_prompt": "é" * 100, "user_prompt": ""}] * 2)
    utf8 = len(request_line(wide[0]))
    escaped = anthropic_batch._wire_size(request_line(wide[0]))
    assert escaped == utf8 - 1 - 200 + 600  # no newline; 2 UTF-8 bytes become 6 escaped ones
    assert [s["requests"] for s in write_shards(wide, tmp_path / "w", max_bytes=escaped + utf8)] == [1, 1]


def test_the_answer_is_the_text_blocks_alone():
    outcome = outcome_from_message(message(thinking(), text(" no "), {"type": "redacted_thinking", "data": "x"},
                                           text("pe ")))
    assert (outcome.text, outcome.finish_reason, outcome.error) == ("no pe", "end_turn", None)


def test_a_refusal_is_a_returned_answer_with_no_text_and_its_category():
    refused = outcome_from_message(message(text("partial"), stop="refusal",
                                           stop_details={"type": "refusal", "category": "cyber"}))
    assert (refused.text, refused.finish_reason, refused.error) == (None, "refusal:cyber", None)
    for details in (None, {"category": None}, {"category": ""}, "odd"):
        extra = {} if details is None else {"stop_details": details}
        assert outcome_from_message(message(stop="refusal", **extra)).finish_reason == "refusal:unspecified"


def test_max_tokens_keeps_its_partial_text_and_reads_as_truncated():
    partial = outcome_from_message(message(thinking(), text('{"verdict"'), stop="max_tokens"))
    assert (partial.text, partial.finish_reason) == ('{"verdict"', "max_tokens") and is_truncated("max_tokens")
    inside = outcome_from_message(message(thinking(), stop="max_tokens"))
    assert (inside.text, inside.finish_reason) == (None, "max_tokens")


def test_a_message_with_no_text_is_a_returned_answer_with_its_stop_reason():
    assert outcome_from_message(message(thinking())).finish_reason == "end_turn"
    blank = outcome_from_message(message(text("  "), stop=None))
    assert (blank.text, blank.finish_reason) == (None, NO_TEXT)
    odd = outcome_from_message({"content": "not a list", "stop_reason": 7})
    assert (odd.text, odd.finish_reason) == (None, NO_TEXT)
    skipped = outcome_from_message({"content": ["x", {"type": "text", "text": 3}, text("ok")]})
    assert (skipped.text, skipped.finish_reason) == ("ok", None)


@pytest.mark.parametrize("result, error", [
    ({"type": "errored", "error": {"type": "error", "error": {"type": "overloaded_error", "message": "m"}}},
     "errored:overloaded_error"),
    ({"type": "errored", "error": {"type": "invalid_request_error"}}, "errored:invalid_request_error"),
    ({"type": "errored", "error": None}, "errored"),
    ({"type": "errored"}, "errored"),
    ({"type": "canceled"}, "canceled"),
    ({"type": "expired"}, "expired"),
])
def test_a_failed_result_is_that_requests_failure_and_bills_nothing(result, error):
    outcome, billed = outcome_from_record({"custom_id": "a", "result": result})
    assert (outcome.text, outcome.finish_reason, outcome.error, billed) == (None, None, error, None)


@pytest.mark.parametrize("record", [{"custom_id": "a"}, {"custom_id": "a", "result": {"type": "weird"}},
                                    {"custom_id": "a", "result": {"type": "succeeded", "message": None}}])
def test_a_result_of_no_known_shape_is_refused(record):
    with pytest.raises(BatchExecutionError, match="of no type this build reads"):
        outcome_from_record(record)


def test_a_succeeded_result_bills_its_message():
    record = succeeded("a", text("yes"))
    outcome, billed = outcome_from_record(record)
    assert outcome.text == "yes" and billed is record["result"]["message"]


def test_usage_maps_onto_the_schema_input_counting_the_cached_and_the_written():
    usage = BatchUsage()
    add_usage(usage, message(text("x")))
    assert usage.to_dict() == {"input_tokens": 17, "cached_tokens": 4, "cache_write_tokens": 2,
                               "output_tokens": 30, "reasoning_tokens": 20, "requests": 1}
    add_usage(usage, message(text("x"), usage={"input_tokens": 5, "output_tokens": True, "cache_read_input_tokens":
                                               None, "output_tokens_details": "odd"}))
    add_usage(usage, message(text("x"), usage=None))
    assert usage.to_dict() == {"input_tokens": 22, "cached_tokens": 4, "cache_write_tokens": 2,
                               "output_tokens": 30, "reasoning_tokens": 20, "requests": 3}


def test_a_result_record_is_the_sdks_own_decoding_written_back_whole():
    record = succeeded("a", thinking(), text("no"), stop="refusal", stop_details={"category": "bio"},
                       usage={**USAGE, "future_field": 1})
    assert result_record(sdk_result(record)) == record  # fields this SDK's types lack (stop_details) are kept
    assert result_record(record) == record and result_record(record) is not record


def test_part_state_reads_the_end_from_the_counts():
    assert part_state({}) is None
    assert part_state({"processing_status": "in_progress"}) == "in_progress"
    assert part_state({"processing_status": "ended", "request_counts": {"succeeded": 2, "errored": 1}}) == PART_COMPLETED
    assert part_state({"processing_status": "ended"}) == PART_COMPLETED
    for kind in ("canceled", "expired"):
        assert part_state({"processing_status": "ended", "request_counts": {kind: 1}}) == PART_INCOMPLETE


@pytest.mark.parametrize("status, refused", [(400, True), (404, True), (413, True), (429, True), (408, False),
                                             (409, False), (499, False), (500, False), (529, False), (None, False),
                                             (True, False), ("400", False)])
def test_only_a_definite_4xx_says_a_create_made_no_batch(status, refused):
    error = RuntimeError("x")
    error.status_code = status
    assert create_was_refused(error) is refused
    assert create_was_refused(RuntimeError("timeout")) is False


def test_preparing_makes_no_api_call_and_needs_no_key(tmp_path):
    runner = make_runner(tmp_path)
    requests = requests_for()
    state = runner.prepare(requests)
    assert state["parts"] == {"part-0000.jsonl": {}} and state["model"] == MODEL
    assert state["endpoint"] == ANTHROPIC_BATCH_ENDPOINT and state["state_version"] == STATE_VERSION
    assert runner.prepare(requests) == state
    with pytest.raises(ValueError, match="is for 'other', not this runner's"):
        runner.prepare(requests_for(model="other"))


def test_submitting_without_confirmation_refuses_before_any_client_exists(tmp_path):
    with pytest.raises(BatchSpendNotConfirmed):
        make_runner(tmp_path).submit(requests_for())


def test_a_run_creates_once_inline_without_sdk_retries_and_records_the_batch_before_waiting(tmp_path, confirmed):
    client = FakeAnthropic(statuses=["in_progress", "canceling", "ended"])
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    seen: List[Dict[str, Any]] = []
    client.messages.batches.retrieve = lambda batch_id, real=client._retrieve: (
        seen.append(json.loads((runner.workdir(requests) / "state.json").read_text())) or real(batch_id))
    outcomes, usage = runner.run(requests)
    assert client.option_calls == [{"max_retries": 0}] and len(client.create_calls) == 1
    directory = runner.workdir(requests)
    assert client.create_calls[0] == [json.loads(line) for line in
                                      (directory / "inputs" / "part-0000.jsonl").read_text().splitlines()]
    assert seen[0]["parts"]["part-0000.jsonl"]["batch_id"] == "msgbatch_00"
    assert "create_started_at" in seen[0]["parts"]["part-0000.jsonl"]
    assert [outcomes[r.custom_id].text for r in requests] == [f"answer::{r.custom_id}" for r in requests]
    assert usage.to_dict() == {"input_tokens": 51, "cached_tokens": 12, "cache_write_tokens": 6,
                               "output_tokens": 90, "reasoning_tokens": 60, "requests": 3}
    assert json.loads((directory / "usage.json").read_text()) == usage.to_dict()
    state = json.loads((directory / "state.json").read_text())
    part = state["parts"]["part-0000.jsonl"]
    assert (part["processing_status"], part["request_counts"]["succeeded"], part["ended_at"]) == (
        "ended", 3, "2026-10-13T01:00:00+00:00")
    assert not (directory / "reasoning_summaries.jsonl").exists()  # the omitted display returns no thinking text


def test_a_rerun_reattaches_and_buys_nothing_without_asking_again(tmp_path, confirmed, monkeypatch):
    client = FakeAnthropic()
    requests = requests_for()
    make_runner(tmp_path, client).run(requests)
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: False)
    outcomes, _ = make_runner(tmp_path, client).run(requests)
    assert len(client.create_calls) == 1 and len(client.results_calls) == 1
    assert outcomes[requests[0].custom_id].text.startswith("answer::")


def test_a_batch_created_by_another_process_before_the_claim_is_not_bought_again(tmp_path, confirmed, monkeypatch):
    client = FakeAnthropic()
    requests = requests_for()
    stale = make_runner(tmp_path, client)
    stale.prepare(requests)
    make_runner(tmp_path, client).submit(requests)
    real_prepare = stale.prepare
    stale.prepare = lambda reqs: {**real_prepare(reqs), "parts": {"part-0000.jsonl": {}}}
    stale.submit(requests)
    assert len(client.create_calls) == 1


def test_a_state_another_process_wrote_after_the_existence_check_is_read_never_replaced(tmp_path, confirmed,
                                                                                      monkeypatch):
    client = FakeAnthropic()
    requests = requests_for()
    state_path = make_runner(tmp_path).workdir(requests) / "state.json"
    real = anthropic_batch.write_bytes_atomic

    def racing(path, payload):
        real(path, payload)
        if not state_path.exists():  # the other process prepares, buys and records here
            monkeypatch.setattr(anthropic_batch, "write_bytes_atomic", real)
            make_runner(tmp_path, client).submit(requests)

    monkeypatch.setattr(anthropic_batch, "write_bytes_atomic", racing)
    state = make_runner(tmp_path, client).submit(requests)
    assert state["parts"]["part-0000.jsonl"]["batch_id"] == "msgbatch_00" and len(client.create_calls) == 1
    assert json.loads(state_path.read_text())["parts"]["part-0000.jsonl"]["batch_id"] == "msgbatch_00"
    assert not list(state_path.parent.glob("*.tmp"))


def test_a_held_claim_refuses_a_second_creator(tmp_path, confirmed):
    runner = make_runner(tmp_path, FakeAnthropic())
    requests = requests_for()
    runner.prepare(requests)
    (runner.workdir(requests) / "creating.lock").write_text("123")
    with pytest.raises(BatchAlreadyClaimed):
        runner.submit(requests)


def test_an_input_file_changed_after_preparing_is_never_sent(tmp_path, confirmed):
    client = FakeAnthropic()
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    runner.prepare(requests)
    (runner.workdir(requests) / "inputs" / "part-0000.jsonl").write_bytes(b"{}\n")
    with pytest.raises(BatchExecutionError, match="changed on disk"):
        runner.submit(requests)
    assert client.create_calls == []


class Refused(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("error", [Refused(500), Refused(408), TimeoutError("read timed out"),
                                   ConnectionResetError("reset")])
def test_a_create_that_may_have_made_its_batch_is_never_repeated(tmp_path, confirmed, error):
    client = FakeAnthropic(create_failures=[("after", error)])
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="does not say whether the batch was made") as raised:
        runner.submit(requests)
    assert raised.value.__cause__ is error
    state = json.loads((runner.workdir(requests) / "state.json").read_text())
    assert state["parts"]["part-0000.jsonl"]["create_started_at"]
    for _ in range(2):
        with pytest.raises(BatchExecutionError, match="started at .* and recorded no batch"):
            make_runner(tmp_path, client).submit(requests)
    assert len(client.create_calls) == 1


def test_a_create_the_api_refused_is_cleared_and_may_be_retried(tmp_path, confirmed):
    client = FakeAnthropic(create_failures=[("before", Refused(429))])
    requests = requests_for()
    with pytest.raises(Refused):
        make_runner(tmp_path, client).submit(requests)
    state = json.loads((make_runner(tmp_path).workdir(requests) / "state.json").read_text())
    assert state["parts"]["part-0000.jsonl"] == {}
    make_runner(tmp_path, client).run(requests)
    assert len(client.create_calls) == 2 and len(client.batches) == 1


def test_a_create_returning_no_id_is_refused_and_never_created_again(tmp_path, confirmed):
    client = FakeAnthropic(created_id=None)
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="returned no batch id"):
        make_runner(tmp_path, client).submit(requests)
    with pytest.raises(BatchExecutionError, match="recorded no batch"):
        make_runner(tmp_path, client).submit(requests)
    assert len(client.create_calls) == 1


def test_a_batch_whose_record_could_not_be_saved_is_named_and_never_bought_again(tmp_path, confirmed, monkeypatch):
    client = FakeAnthropic()
    requests = requests_for()
    real = anthropic_batch.write_json_atomic

    def failing(path, value):
        if any(part.get("batch_id") for part in value.get("parts", {}).values()):
            raise OSError("disk full")
        real(path, value)

    monkeypatch.setattr(anthropic_batch, "write_json_atomic", failing)
    with pytest.raises(BatchExecutionError, match=r"msgbatch_00 was created .* by hand"):
        make_runner(tmp_path, client).submit(requests)
    monkeypatch.setattr(anthropic_batch, "write_json_atomic", real)
    with pytest.raises(BatchExecutionError, match="recorded no batch"):
        make_runner(tmp_path, client).submit(requests)
    assert len(client.create_calls) == 1


@pytest.mark.parametrize("found", [True, False])
def test_a_mark_resolved_by_hand_resumes(tmp_path, confirmed, found):
    client = FakeAnthropic(create_failures=[("after", Refused(503))])
    requests = requests_for()
    with pytest.raises(BatchExecutionError):
        make_runner(tmp_path, client).submit(requests)
    state_path = make_runner(tmp_path).workdir(requests) / "state.json"
    state = json.loads(state_path.read_text())
    part = state["parts"]["part-0000.jsonl"]
    if found:
        part["batch_id"] = "msgbatch_00"
    else:
        del part["create_started_at"]
    state_path.write_text(json.dumps(state))
    outcomes, _ = make_runner(tmp_path, client).run(requests)
    assert len(client.create_calls) == (1 if found else 2)
    assert all(outcome.text for outcome in outcomes.values())


def test_a_state_from_another_version_or_model_is_refused(tmp_path):
    runner = make_runner(tmp_path)
    requests = requests_for()
    runner.prepare(requests)
    state_path = runner.workdir(requests) / "state.json"
    state = json.loads(state_path.read_text())
    state_path.write_text(json.dumps({**state, "state_version": 99}))
    with pytest.raises(BatchExecutionError, match="state version 99"):
        runner.prepare(requests)
    state_path.write_text(json.dumps({**state, "model": "other"}))
    with pytest.raises(BatchExecutionError, match="was prepared for 'other'"):
        runner.prepare(requests)


@pytest.mark.parametrize("kwargs, message", [
    ({"poll_seconds": 0}, "poll_seconds"), ({"timeout_seconds": 0}, "timeout_seconds"),
    ({"max_poll_errors": 0}, "max_poll_errors"), ({"model": ""}, "needs a model name")])
def test_the_runner_refuses_a_nonsense_setting(tmp_path, kwargs, message):
    settings = {"root": tmp_path, "model": MODEL, **kwargs}
    with pytest.raises(ValueError, match=message):
        AnthropicBatchRunner(**settings)


def test_the_default_client_reads_the_key_from_the_environment_alone_and_never_logs_it(monkeypatch, caplog):
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
        anthropic_batch.default_client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"  {SENTINEL_KEY}  ")
    with caplog.at_level(logging.DEBUG):
        client = anthropic_batch.default_client()
    import anthropic

    assert isinstance(client, anthropic.Anthropic) and client.api_key == SENTINEL_KEY
    assert SENTINEL_KEY not in caplog.text
    assert str(client.base_url).rstrip("/") == "https://api.anthropic.com" and client.auth_token is None


@pytest.mark.parametrize("names", [("ANTHROPIC_BASE_URL",), ("ANTHROPIC_AUTH_TOKEN",),
                                   ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")])
def test_the_default_client_refuses_an_ambient_host_or_second_credential_naming_it_not_its_value(monkeypatch, names):
    """The SDK reads ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN on its own."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    for name in names:
        monkeypatch.setenv(name, "VALUE-NEVER-ECHOED")
    with pytest.raises(ValueError, match=" and ".join(names) + " set") as raised:
        anthropic_batch.default_client()
    assert "VALUE-NEVER-ECHOED" not in str(raised.value) and SENTINEL_KEY not in str(raised.value)


def test_polling_waits_through_every_live_state_and_transient_errors(tmp_path, confirmed, caplog):
    client = FakeAnthropic(statuses=["in_progress", RuntimeError("503"), "canceling", "ended"])
    runner = make_runner(tmp_path, client)
    with caplog.at_level(logging.WARNING):
        runner.run(requests_for())
    assert len(client.retrieve_calls) == 4 and "waiting rather than giving up" in caplog.text


def test_a_status_read_as_a_plain_mapping_or_with_no_counts_is_recorded_as_given(tmp_path, confirmed):
    ended = SimpleNamespace(processing_status="ended", results_url="u", ended_at="2026-10-13T01:00:00Z",
                            request_counts={"succeeded": 3, "errored": 0, "canceled": 0, "expired": 0})
    client = FakeAnthropic(statuses=[SimpleNamespace(processing_status="in_progress"), ended])
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    seen = []
    real = runner._sweep
    runner._sweep = lambda *args: (seen.append(json.loads(json.dumps(args[0]["parts"]))), real(*args))[1]
    outcomes, _ = runner.run(requests)
    assert len(outcomes) == 3
    part = json.loads((runner.workdir(requests) / "state.json").read_text())["parts"]["part-0000.jsonl"]
    assert (part["request_counts"]["succeeded"], part["ended_at"], part["expires_at"]) == (
        3, "2026-10-13T01:00:00Z", None)
    assert seen[1]["part-0000.jsonl"]["request_counts"] is None


def test_consecutive_status_errors_past_the_cap_give_up_loudly(tmp_path, confirmed):
    client = FakeAnthropic(statuses=[RuntimeError("401")] * 3)
    with pytest.raises(BatchExecutionError, match="3 status checks in a row failed"):
        make_runner(tmp_path, client, max_poll_errors=3).run(requests_for())


def test_a_timeout_stops_the_wait_and_says_the_work_is_paid_for(tmp_path, confirmed):
    ticks = iter([0.0, 5.0, 11.0])
    client = FakeAnthropic(statuses=["in_progress"] * 5)
    with pytest.raises(BatchExecutionError, match="did not finish within 10.0s"):
        make_runner(tmp_path, client, timeout_seconds=10.0, monotonic=lambda: next(ticks)).run(requests_for())


def test_polling_a_never_submitted_input_is_refused(tmp_path):
    runner = make_runner(tmp_path, FakeAnthropic())
    requests = requests_for()
    runner.prepare(requests)
    with pytest.raises(BatchExecutionError, match="was never submitted"):
        runner.poll(requests)
    with pytest.raises(BatchExecutionError, match="was never submitted"):
        runner.refresh(requests)


@pytest.mark.parametrize("kind", ["canceled", "expired"])
def test_a_batch_that_ended_without_completing_raises_naming_it_and_is_never_resubmitted(tmp_path, confirmed, kind):
    client = FakeAnthropic(responder=lambda cid, params: failed(cid, kind) if cid.endswith("01-c00") else success(
        cid, params))
    requests = requests_for()
    for _ in range(2):
        with pytest.raises(BatchExecutionError, match="ended without completing .*ruleforge campaign collect"):
            make_runner(tmp_path, client).run(requests)
    assert len(client.create_calls) == 1 and client.results_calls == []


def test_errored_requests_alone_complete_a_batch_as_that_requests_failure(tmp_path, confirmed, caplog):
    client = FakeAnthropic(responder=lambda cid, params: failed(cid, "errored") if cid.endswith("01-c00") else success(
        cid, params))
    requests = requests_for()
    with caplog.at_level(logging.WARNING):
        outcomes, usage = make_runner(tmp_path, client).run(requests)
    assert outcomes[requests[1].custom_id].error == "errored:overloaded_error"
    assert usage.requests == 2 and "failed: errored:overloaded_error" in caplog.text


def test_a_downloaded_output_is_not_fetched_again_and_an_unended_batch_is_not_fetched(tmp_path, confirmed):
    client = FakeAnthropic()
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    runner.run(requests)
    runner.download(requests)
    assert len(client.results_calls) == 1
    other = requests_for(seed=9)
    runner.submit(other)
    with pytest.raises(BatchExecutionError, match="has not ended with results"):
        runner.download(other)


def test_results_are_joined_by_custom_id_not_by_position(tmp_path, confirmed):
    client = FakeAnthropic(shuffle=True)
    requests = requests_for()
    outcomes, _ = make_runner(tmp_path, client).run(requests)
    assert [outcomes[r.custom_id].text for r in requests] == [f"answer::{r.custom_id}" for r in requests]


@pytest.mark.parametrize("extra, message", [
    ({"result": {"type": "succeeded", "message": message(text("x"))}}, "carries no custom id"),
    (succeeded("s5-p000099-c00", text("x")), "unknown custom id"),
    (succeeded("s5-p000000-c00", text("x")), "repeats custom id"),
])
def test_a_line_with_no_unknown_or_repeated_custom_id_is_refused(tmp_path, confirmed, extra, message):
    client = FakeAnthropic(extra_lines=(extra,))
    with pytest.raises(BatchExecutionError, match=message):
        make_runner(tmp_path, client).run(requests_for())


def test_a_request_with_no_result_and_a_line_that_is_not_json_are_refused(tmp_path, confirmed):
    client = FakeAnthropic(responder=lambda cid, params: None if cid.endswith("02-c00") else success(cid, params))
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="1 of 3 request.s. have no result"):
        runner.run(requests)
    part = runner.workdir(requests) / "outputs" / "part-0000.jsonl"
    part.write_text(part.read_text() + "\n\nnot json\n")
    with pytest.raises(BatchExecutionError, match="is not JSON"):
        runner.read_results(requests)


def test_a_partial_reading_keeps_the_unfinished_as_failures(tmp_path, confirmed):
    client = FakeAnthropic(responder=lambda cid, params: (
        failed(cid, "canceled") if cid.endswith("01-c00") else success(cid, params)))
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="ended without completing"):
        runner.run(requests)
    runner.refresh(requests)
    runner.download(requests)
    with pytest.raises(BatchExecutionError, match="was canceled"):
        runner.read_results(requests)
    outcomes, usage = runner.read_results(requests, partial=True)
    assert outcomes[requests[1].custom_id].error == "canceled" and usage.requests == 2


def test_a_request_with_no_result_is_refused_in_a_partial_reading_too(tmp_path, confirmed):
    client = FakeAnthropic(responder=lambda cid, params: (
        failed(cid, "canceled") if cid.endswith("01-c00") else success(cid, params)))
    runner = make_runner(tmp_path, client, max_shard_requests=2)
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="ended without completing"):
        runner.run(requests)
    runner.refresh(requests)
    outputs = runner.download(requests)
    part = outputs / "part-0000.jsonl"
    part.write_text(part.read_text().split("\n")[0] + "\n")
    with pytest.raises(BatchExecutionError, match="part-0000.jsonl holds .*'succeeded': 1.* but the API counted "
                                                  ".*'succeeded': 1.*'canceled': 1"):
        runner.read_results(requests, partial=True)
    part.unlink()
    with pytest.raises(BatchExecutionError, match="2 of 3 request.s. have no result.*damaged"):
        runner.read_results(requests, partial=True)
    assert not (runner.workdir(requests) / "usage.json").exists()


def test_a_results_file_whose_types_are_not_the_api_counts_is_refused(tmp_path, confirmed):
    counts = {"processing": 0, "succeeded": 2, "errored": 1, "canceled": 0, "expired": 0}
    client = FakeAnthropic(statuses=[sdk_batch("msgbatch_00", "ended", counts, ended=True)])
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="part-0000.jsonl holds .*'succeeded': 3.* but the API counted "
                                                  ".*'succeeded': 2.* for batch msgbatch_00"):
        runner.run(requests)
    assert not (runner.workdir(requests) / "usage.json").exists()


def test_a_results_file_of_no_ended_batch_is_refused(tmp_path, confirmed):
    client = FakeAnthropic()
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    runner.run(requests)
    outputs = runner.workdir(requests) / "outputs"
    (outputs / "part-0000.jsonl").rename(outputs / "part-0007.jsonl")
    with pytest.raises(BatchExecutionError, match="part-0007.jsonl belongs to no ended batch"):
        runner.read_results(requests)


def test_two_downloads_at_once_never_share_a_temporary_file(tmp_path, confirmed):
    client = FakeAnthropic()
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    runner.submit(requests)
    runner.poll(requests)
    real_results = client.messages.batches.results
    other = make_runner(tmp_path, client)

    def racing(batch_id):
        items = iter(real_results(batch_id))
        yield next(items)
        if len(client.results_calls) == 1:  # the first download, one line written: the second runs whole
            other.download(requests)
        yield from items

    client.messages.batches.results = racing
    outputs = runner.download(requests)
    assert len(client.results_calls) == 2 and sorted(path.name for path in outputs.iterdir()) == ["part-0000.jsonl"]
    outcomes, usage = runner.read_results(requests)
    assert usage.requests == 3 and all(outcome.text for outcome in outcomes.values())


def test_a_download_that_fails_midway_leaves_no_file_behind(tmp_path, confirmed):
    client = FakeAnthropic()
    runner = make_runner(tmp_path, client)
    requests = requests_for()
    runner.submit(requests)
    runner.poll(requests)
    real_results = client.messages.batches.results

    def broken(batch_id):
        yield next(iter(real_results(batch_id)))
        raise ConnectionResetError("reset mid-download")

    client.messages.batches.results = broken
    with pytest.raises(ConnectionResetError):
        runner.download(requests)
    assert list((runner.workdir(requests) / "outputs").iterdir()) == []


def test_an_answer_carrying_a_unicode_line_separator_is_read_whole_and_refusals_are_counted(tmp_path, confirmed,
                                                                                          caplog):
    def respond(cid, params):
        if cid.endswith("00-c00"):
            return succeeded(cid, text("a b"))
        return succeeded(cid, text("x"), stop="refusal", stop_details={"category": "general_harms"})

    with caplog.at_level(logging.WARNING):
        outcomes, usage = make_runner(tmp_path, FakeAnthropic(responder=respond)).run(requests_for())
    assert outcomes["s5-p000000-c00"].text == "a b"
    assert outcomes["s5-p000001-c00"].finish_reason == "refusal:general_harms" and usage.requests == 3
    assert "2 refusal(s) by category: {'refusal:general_harms': 2}" in caplog.text


def test_several_input_files_are_several_batches_joined_back_together(tmp_path, confirmed):
    client = FakeAnthropic()
    runner = make_runner(tmp_path, client)
    runner.max_shard_requests = 2
    requests = requests_for()
    outcomes, usage = runner.run(requests)
    assert len(client.create_calls) == 2 and usage.requests == 3 and len(outcomes) == 3


def test_make_batch_generate_round_trips_text_details_and_usage(tmp_path, confirmed, caplog):
    client = FakeAnthropic(responder=lambda cid, params: succeeded(
        cid, thinking(), text("no"), stop="max_tokens" if cid.endswith("01-c00") else "end_turn"))
    tracker = TokenUsageTracker()
    plain, detailed = make_batch_generate(model_name=MODEL, root=tmp_path / "batch", reasoning_effort="medium",
                                          client_factory=lambda: client, tracker=tracker, poll_seconds=1)
    with caplog.at_level(logging.WARNING):
        assert plain(PROMPTS, 1.0, 64, 1, 7) == [["no"], ["no"], ["no"]]
    assert "has no seed parameter; the requested seed 7 is ignored" in caplog.text
    assert [d[0]["finish_reason"] for d in detailed(PROMPTS, 1.0, 64, 1, 7)] == ["end_turn", "max_tokens", "end_turn"]
    sent = client.create_calls[0][0]["params"]
    assert (sent["max_tokens"], sent["output_config"], "temperature" in sent) == (64, {"effort": "medium"}, False)
    assert tracker.prompt_tokens == 2 * 51 and len(client.create_calls) == 1
    with pytest.raises(ValueError, match="sends no temperature"):
        plain(PROMPTS, 0.0, 64, 1, 7)


def test_a_retry_after_a_failed_batch_call_resumes_the_batch_it_bought(tmp_path, confirmed):
    client = FakeAnthropic()
    real = client._results
    calls = {"n": 0}

    def flaky(batch_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("download dropped")
        return real(batch_id)

    client.messages.batches.results = flaky
    plain, _ = make_batch_generate(model_name=MODEL, root=tmp_path / "batch", client_factory=lambda: client)
    with pytest.raises(ConnectionError):
        plain(PROMPTS, 1.0, 64, 1, 11)
    assert plain(PROMPTS, 1.0, 64, 1, 99) == [[f"answer::s11-p{i:06d}-c00"] for i in range(3)]
    assert len(client.create_calls) == 1


def test_make_block_generate_keys_by_the_block_sends_its_cap_and_no_seed(tmp_path, confirmed, caplog):
    from ruleforge.core.digests import request_namespace

    client = FakeAnthropic()
    block = make_block_generate(model_name=MODEL, root=tmp_path / "batch", reasoning_effort="medium",
                                client_factory=lambda: client)
    with caplog.at_level(logging.WARNING):
        details = block(PROMPTS, 1.0, 16384, [1, 2, 3], "c/m/s/draw-000/attempt-0/block-0000")
    namespace = request_namespace("c/m/s/draw-000/attempt-0/block-0000")
    assert [d["provider_request_ref"] for d in details] == [f"{namespace}-p{i:06d}-c00" for i in range(3)]
    assert all(d["completion"] == f"answer::{d['provider_request_ref']}" for d in details)
    assert details[0]["request_block_ref"].endswith(Path(details[0]["request_block_ref"]).name)
    sent = [request["params"] for request in client.create_calls[0]]
    assert {params["max_tokens"] for params in sent} == {16384} and "has no seed parameter" in caplog.text
    assert all(set(params) == {"model", "max_tokens", "messages", "system", "thinking", "output_config"}
               for params in sent)
    assert block(PROMPTS, 1.0, 16384, None, "c/m/s/draw-000/attempt-0/block-0000") == details
    assert len(client.create_calls) == 1


class MockAPI:
    """The three Message Batches endpoints over ``httpx.MockTransport``: records every request it is sent."""

    def __init__(self, *, fail_create: Optional[int] = None) -> None:
        self.sent: List[Any] = []
        self.created: Dict[str, Any] = {}
        self.fail_create = fail_create

    def handle(self, request):
        import httpx

        self.sent.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            if self.fail_create is not None:
                return httpx.Response(self.fail_create, json={"type": "error", "error": {
                    "type": "api_error", "message": "boom"}})
            body = json.loads(request.content)
            self.created["msgbatch_wire"] = body["requests"]
            return httpx.Response(200, json=self.batch("in_progress", len(body["requests"])))
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_wire":
            return httpx.Response(200, json=self.batch("ended", len(self.created["msgbatch_wire"])))
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_wire/results":
            lines = [success(r["custom_id"], r["params"]) for r in self.created["msgbatch_wire"]]
            return httpx.Response(200, content="".join(json.dumps(line) + "\n" for line in lines).encode())
        return httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": path}})

    @staticmethod
    def batch(status: str, count: int) -> Dict[str, Any]:
        ended = status == "ended"
        return {"id": "msgbatch_wire", "type": "message_batch", "processing_status": status,
                "request_counts": {"processing": 0 if ended else count, "succeeded": count if ended else 0,
                                   "errored": 0, "canceled": 0, "expired": 0},
                "created_at": "2026-10-13T00:00:00Z", "expires_at": "2026-10-14T00:00:00Z", "archived_at": None,
                "cancel_initiated_at": None, "ended_at": "2026-10-13T00:05:00Z" if ended else None,
                "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_wire/results" if ended else None}


def _sdk_client(api: MockAPI):
    import anthropic
    import httpx

    return lambda: anthropic.Anthropic(api_key=SENTINEL_KEY, http_client=httpx.Client(
        transport=httpx.MockTransport(api.handle)))


def test_the_sdk_sends_the_input_file_s_requests_verbatim_effort_included_and_the_key_in_its_header_alone(
        tmp_path, confirmed, caplog):
    """anthropic 0.84 types ``output_config.effort`` without ``xhigh`` and ``thinking`` without ``display``."""
    api = MockAPI()
    runner = AnthropicBatchRunner(root=tmp_path / "batch", model=MODEL, client_factory=_sdk_client(api),
                                  sleep=lambda seconds: None)
    requests = requests_for(namespace="nsabc")
    with caplog.at_level(logging.DEBUG):
        outcomes, usage = runner.run(requests)
    create = api.sent[0]
    directory = runner.workdir(requests)
    assert (create.method, create.url.path) == ("POST", "/v1/messages/batches")
    assert json.loads(create.content) == {"requests": [
        json.loads(line) for line in (directory / "inputs" / "part-0000.jsonl").read_text().splitlines()]}
    sent = json.loads(create.content)["requests"][0]["params"]
    assert (sent["model"], sent["max_tokens"], sent["output_config"], sent["thinking"]) == (
        MODEL, 16384, {"effort": "medium"}, {"type": "adaptive"})
    assert not {"temperature", "top_p", "top_k", "stop_sequences", "metadata"} & set(sent)
    assert all(request.headers["x-api-key"] == SENTINEL_KEY for request in api.sent)
    assert outcomes["nsabc-p000002-c00"].text == "answer::nsabc-p000002-c00" and usage.requests == 3
    assert SENTINEL_KEY not in caplog.text
    for path in directory.rglob("*"):
        if path.is_file():
            assert SENTINEL_KEY not in path.read_text(encoding="utf-8")


def test_the_sdk_does_not_retry_a_create_and_a_5xx_create_is_never_repeated(tmp_path, confirmed):
    api = MockAPI(fail_create=500)
    runner = AnthropicBatchRunner(root=tmp_path / "batch", model=MODEL, client_factory=_sdk_client(api))
    requests = requests_for()
    with pytest.raises(BatchExecutionError, match="InternalServerError"):
        runner.submit(requests)
    assert len(api.sent) == 1  # the SDK's own retries are off for a create
    with pytest.raises(BatchExecutionError, match="recorded no batch"):
        runner.submit(requests)
    assert len(api.sent) == 1


def test_the_sdk_s_4xx_create_is_refused_and_may_be_retried(tmp_path, confirmed):
    api = MockAPI(fail_create=400)
    runner = AnthropicBatchRunner(root=tmp_path / "batch", model=MODEL, client_factory=_sdk_client(api))
    requests = requests_for()
    import anthropic

    with pytest.raises(anthropic.BadRequestError):
        runner.submit(requests)
    api.fail_create = None
    runner.submit(requests)
    assert len(api.sent) == 2


def test_every_module_imports_without_the_sdk_and_none_imports_it():
    code = (
        "import sys, importlib, pkgutil\n"
        "sys.modules['anthropic'] = None\n"
        "import ruleforge\n"
        "names = [module.name for module in pkgutil.walk_packages(ruleforge.__path__, 'ruleforge.')]\n"
        "assert 'ruleforge.judges.anthropic_batch' in names and len(names) > 100, names\n"
        "for name in names:\n"
        "    importlib.import_module(name)\n"
        "from ruleforge.judges import anthropic_batch\n"
        "import os; os.environ['ANTHROPIC_API_KEY'] = 'x'\n"
        "try:\n"
        "    anthropic_batch.default_client()\n"
        "except ImportError:\n"
        "    print('lazy')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy"
    loaded = subprocess.run([sys.executable, "-c", "import sys; import ruleforge.judges.anthropic_batch, "
                             "ruleforge.judges.batch_collect, ruleforge.evaluate.cost; print('anthropic' in sys.modules)"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert loaded.stdout.strip() == "False", loaded.stderr


def _config(**extra):
    return OmegaConf.create({"model_type": "anthropic_batch", "model_name": MODEL, "max_new_tokens": 100,
                             "temperature": 1.0, "num_tries": 1, "seed": 3, "send_max_output_tokens": True, **extra})


def test_the_backend_builds_the_anthropic_adapters_and_declares_no_seed(monkeypatch, tmp_path):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(anthropic_batch, "make_batch_generate", lambda **k: seen.update(batch=k) or ("p", "d"))
    monkeypatch.setattr(anthropic_batch, "make_block_generate", lambda **k: seen.update(block=k) or "block")
    backend = ModelBackend(_config(batch_root=str(tmp_path), reasoning_level="medium", batch_poll_seconds=5,
                                   batch_timeout_seconds=60))
    assert "anthropic_batch" in BATCH_MODEL_TYPES and backend.seed_capability == SEED_UNSUPPORTED
    assert (backend._batch_generate, backend._batch_generate_with_details, backend._generate_request_block) == (
        "p", "d", "block")
    assert seen["batch"] == seen["block"] == {
        "model_name": MODEL, "root": tmp_path, "reasoning_effort": "medium", "thinking_display": None,
        "poll_seconds": 5.0, "timeout_seconds": 60.0, "tracker": backend.token_tracker}


@pytest.mark.parametrize("extra, message", [
    ({"batch_endpoint": "responses"}, "has one endpoint"),
    ({"reasoning_summary": "auto"}, "sends no reasoning_summary"),
    ({"send_max_output_tokens": False}, "set send_max_output_tokens: true"),
])
def test_the_backend_refuses_a_setting_the_transport_cannot_honour(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        ModelBackend(_config(batch_root=str(tmp_path), **extra))
    with pytest.raises(ValueError, match="batch_root"):
        ModelBackend(_config())


def test_an_absent_cap_switch_is_refused_too(tmp_path):
    config = _config(batch_root=str(tmp_path))
    del config["send_max_output_tokens"]
    with pytest.raises(ValueError, match="set send_max_output_tokens: true"):
        ModelBackend(config)


def test_a_token_budget_falls_back_to_one_block_at_a_time(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        backend = ModelBackend(_config(batch_root=str(tmp_path), max_enqueued_tokens=1000))
    assert backend.max_enqueued_tokens is None and "no windowed" in caplog.text


def test_generate_once_through_the_backend_sends_the_cap_and_no_seed(monkeypatch, tmp_path, confirmed):
    client = FakeAnthropic(responder=lambda cid, params: succeeded(cid, text("x"), stop="refusal",
                                                                   stop_details={"category": "bio"}))
    monkeypatch.setattr(anthropic_batch, "default_client", lambda: client)
    backend = ModelBackend(_config(batch_root=str(tmp_path), reasoning_level="medium"))
    outcomes = backend.generate_once(PROMPTS, temperature=1.0, seeds=None, request_block_id="b")
    assert all(o.returned and o.text is None and o.finish_reason == "refusal:bio" for o in outcomes)
    assert {request["params"]["max_tokens"] for request in client.create_calls[0]} == {100}


def test_the_model_config_composes_into_an_anthropic_batch_judge(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "claude-haiku-5-5-batch.yaml")
    assert (cfg.model_type, cfg.model_name, cfg.max_new_tokens, cfg.send_max_output_tokens) == (
        "anthropic_batch", MODEL, 16384, True)
    assert (cfg.reasoning_level, cfg.temperature, cfg.fixed_temperature, cfg.max_enqueued_tokens) == (
        "medium", 1, 1.0, None)
    assert cfg.batch_root == str(tmp_path / "batch") and "include_thoughts" not in cfg
    backend = ModelBackend(cfg)
    assert backend.seed_capability == SEED_UNSUPPORTED and backend.resolved_temperature(0.0) == 1.0
    assert round(696 * cfg.max_new_tokens * 0.25 / 1e6, 2) == 2.85


def test_a_haiku_condition_records_what_its_requests_carry():
    from ruleforge.evaluate.campaign import condition_spec_for
    from ruleforge.evaluate.identity import SamplingConfig

    cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / "claude-haiku-5-5-batch.yaml")
    backend = SimpleNamespace(model_name=MODEL, model_type="anthropic_batch", max_new_tokens=16384,
                              seed_capability=SEED_UNSUPPORTED)
    task_spec = SimpleNamespace(reasoning_prompt_style="naive", individual_rule_context="isolated")
    spec = condition_spec_for(condition_id=f"{MODEL}/naive", backend=backend, task_spec=task_spec,
                              config=SamplingConfig(), judge_type="No_Interpretation",
                              rule_set_path=REPO_ROOT / "data/rules/modelspec/openai_rules_all.jsonl", model_cfg=cfg)
    assert (spec.endpoint, spec.reasoning_effort, spec.max_new_tokens, spec.send_max_output_tokens,
            spec.seed_capability, spec.temperature, spec.sampling_truncation) == (
        None, "medium", 16384, True, "unsupported", 1.0, "provider_default")


def test_a_one_request_campaign_runs_end_to_end_through_the_batch_path(monkeypatch, tmp_path, modelspec):
    """The smoke test's design, offline: one scenario, draw 0, the full ruleset only, one attempt."""
    from tests.test_sampling_campaign import compose_campaign

    from ruleforge.evaluate.campaign import build_plan, run_plan
    from ruleforge.evaluate.runner import STATE_COMPLETE

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    client = FakeAnthropic(responder=lambda cid, params: succeeded(cid, thinking(), text("no")))
    monkeypatch.setattr(anthropic_batch, "default_client", lambda: client)
    cfg = compose_campaign(tmp_path, **{
        "model@judge_model": "claude-haiku-5-5-batch", "model@agent_model": "claude-haiku-5-5-batch",
        "evaluation_mode": "full", "sampling.num_decoder_draws": 2, "decoder_draw_ids": "[0]",
        "sampling.max_semantic_attempts": 1})
    scenarios = Path(cfg.scenario_set_path)
    scenarios.write_text(scenarios.read_text().splitlines()[0] + "\n")
    backend = ModelBackend(cfg.judge_model)
    task_spec = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    assert (plan.condition.seed_capability, plan.condition.endpoint, plan.condition.send_max_output_tokens) == (
        SEED_UNSUPPORTED, None, True)
    assert (plan.condition.reasoning_effort, plan.condition.max_new_tokens) == ("medium", 16384)
    with pytest.raises(BatchSpendNotConfirmed):
        run_plan(plan, backend=backend, task_spec=task_spec)
    [prepared] = [p for p in (tmp_path / "batch").iterdir() if p.is_dir()]
    assert client.create_calls == [] and (prepared / "inputs" / "part-0000.jsonl").is_file()

    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    backend = ModelBackend(cfg.judge_model)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    completions = run_plan(plan, backend=backend, task_spec=task_spec)
    assert completions[0]["state"] == STATE_COMPLETE
    [[line]] = client.create_calls
    assert [line] == [json.loads(text) for text in (prepared / "inputs" / "part-0000.jsonl").read_text().splitlines()]
    params = line["params"]
    assert (params["model"], params["max_tokens"], params["output_config"], params["thinking"]) == (
        MODEL, 16384, {"effort": "medium"}, {"type": "adaptive", "display": "summarized"})
    assert params["messages"] == [{"role": "user", "content": EMPTY_USER_TURN}]
    assert "# Rule Set Used for Evaluation" in params["system"] and "temperature" not in params
    assert json.loads((prepared / "usage.json").read_text())["requests"] == 1


@pytest.mark.parametrize("overrides, message", [
    ({"execution.pack_draws": "true"}, "needs a 'local' judge"),
    ({"sampling.strict_think": "true"}, "is 'anthropic_batch'"),
    ({"execution.num_shards": 2, "execution.shard_index": 0}, "num_shards needs a 'local' judge"),
    ({"+judge_model.adapter_path": "/x/y.safetensors"}, "'anthropic_batch' judge; only a local judge serves a LoRA"),
    ({"+judge_model.dtype": "float32"}, "'anthropic_batch' judge; only a local judge serves weights"),
    ({"+judge_model.default_truncation": "{top_k: 20}"}, "default_truncation is set for a 'anthropic_batch' judge"),
])
def test_preflight_treats_the_anthropic_judge_as_a_batch_judge(tmp_path, monkeypatch, overrides, message):
    from tests.test_sampling_campaign import compose_campaign

    from ruleforge.evaluate.runner import CampaignRunError
    from ruleforge.evaluate.campaign import preflight

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    cfg = compose_campaign(tmp_path, **{"model@judge_model": "claude-haiku-5-5-batch", **overrides})
    with pytest.raises(CampaignRunError, match=message):
        preflight(cfg)


def test_a_resubmit_buys_only_the_parts_that_have_no_batch(tmp_path, confirmed):
    """Two shards, both created; the second's record lost: a resubmit creates the second alone, the first kept."""
    client = FakeAnthropic()
    requests = requests_for()
    runner = make_runner(tmp_path, client, max_shard_requests=2)
    state = runner.submit(requests)
    assert len(client.create_calls) == 2 and sorted(state["parts"]) == ["part-0000.jsonl", "part-0001.jsonl"]
    state_path = runner.workdir(requests) / "state.json"
    recorded = json.loads(state_path.read_text())
    recorded["parts"]["part-0001.jsonl"] = {}
    state_path.write_text(json.dumps(recorded))
    again = make_runner(tmp_path, client, max_shard_requests=2).submit(requests)
    assert len(client.create_calls) == 3
    assert again["parts"]["part-0000.jsonl"]["batch_id"] == state["parts"]["part-0000.jsonl"]["batch_id"]
    assert again["parts"]["part-0001.jsonl"]["batch_id"] not in (None, state["parts"]["part-0001.jsonl"]["batch_id"])
