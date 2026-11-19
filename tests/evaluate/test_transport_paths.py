from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ruleforge.judges import JudgeSpec
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.ledger import STATUS_ACCEPTED, STATUS_TRANSPORT_FAILED
from ruleforge.evaluate.runner import REQUEST_BLOCKS_FNAME, STATE_COMPLETE, TransportExhausted
from ruleforge.judges import models
from ruleforge.judges.backend import GenerationOutcome
from tests.test_sampling_runner import FakeBackend, build_runner, read_jsonl, sent_block, sid_of, write_inputs

#: The id a judge is sent for a block (the condition's identity id in place of its id).
BLOCK = sent_block("camp/fake/naive/draw-000/attempt-{attempt}/block-{index:04d}")


@pytest.fixture
def inputs(tmp_path):
    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


class RaisingOnce(FakeBackend):
    """The call for ``block`` raises once."""

    def __init__(self, block, **kwargs):
        super().__init__(**kwargs)
        self.raise_on = {block}

    def generate_once(self, prompts, *, request_block_id="", **kwargs):
        if request_block_id in self.raise_on:
            self.raise_on.discard(request_block_id)
            self.blocks.append((request_block_id, len(prompts), None))
            return [GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="ConnectionError: x",
                                      block_failed=True) for _ in prompts]
        return super().generate_once(prompts, request_block_id=request_block_id, **kwargs)


def test_a_raised_call_resubmits_the_same_block_id_and_records_nothing(tmp_path, inputs, modelspec):
    backend = RaisingOnce(BLOCK.format(attempt=0, index=0))
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["transport_failed_attempts"] == 0
    assert [block[0] for block in backend.blocks] == [BLOCK.format(attempt=0, index=0)] * 2
    assert [row["block_index"] for row in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)] == [0]
    assert {record.status for record in runner.ledger.load().records} == {STATUS_ACCEPTED}


def test_a_per_request_error_replans_that_scenario_alone_at_the_same_attempt_with_its_seeds(
        tmp_path, inputs, modelspec):
    failed = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="RuntimeError: rate limited")
    backend = FakeBackend(script={(1, "Rule 2"): [failed, "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, scenarios_per_request_block=2)
    assert runner.run()["state"] == STATE_COMPLETE
    blocks = {block_id: (count, seeds) for block_id, count, seeds in backend.blocks}
    assert list(blocks) == [BLOCK.format(attempt=0, index=index) for index in (0, 1, 2)]
    first, retried = blocks[BLOCK.format(attempt=0, index=0)], blocks[BLOCK.format(attempt=0, index=2)]
    # Two scenarios of three components each, then the second scenario alone under its own seeds.
    assert (first[0], retried[0]) == (6, 3) and retried[1] == first[1][3:]
    recorded = read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)
    assert [(row["block_index"], row["semantic_attempt_id"], row["scenario_ids"]) for row in recorded][-1] == (
        2, 0, [sid_of(inputs, 1)])
    rows = [(record.status, record.semantic_attempt_id) for record in runner.ledger.load().records
            if record.scenario_id == sid_of(inputs, 1)]
    assert rows == [(STATUS_TRANSPORT_FAILED, 0), (STATUS_ACCEPTED, 0)]


def test_a_crash_after_the_block_is_recorded_and_before_its_outcomes_are_is_the_first_path(
        tmp_path, inputs, modelspec, monkeypatch):
    backend = FakeBackend()
    runner = build_runner(tmp_path, inputs, modelspec, backend)

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "record", crash)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    assert runner.ledger.load().records == [] and len(read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)) == 1
    resumed = build_runner(tmp_path, inputs, modelspec, backend)
    assert resumed.run()["state"] == STATE_COMPLETE
    assert [block[0] for block in backend.blocks] == [BLOCK.format(attempt=0, index=0)] * 2
    assert backend.blocks[0][2] == backend.blocks[1][2]


def test_a_sync_provider_s_failed_call_is_a_recorded_transport_failure(tmp_path, inputs, modelspec, monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    calls = []

    def respond(**kwargs):
        prompt = kwargs["input"][0]["content"]
        calls.append(prompt)
        if "prompt 1" in prompt and "rule 2 body" in prompt and sum("rule 2 body" in p and "prompt 1" in p
                                                                     for p in calls) == 1:
            raise RuntimeError("rate limited")
        return SimpleNamespace(output_text="no", output=[], usage=None, status="completed")

    client = MagicMock()
    client.responses.create.side_effect = respond
    monkeypatch.setattr(models, "init_openai_client", lambda: client)
    judge = JudgeSpec.openai("fake-judge", fixed_temperature=1.0).start()
    runner = build_runner(tmp_path, inputs, modelspec, judge, scenarios_per_request_block=2)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["transport_failed_attempts"] == 1
    assert completion["invalid_components"] == {"transport_error": 1} and completion["attempts_histogram"] == {"1": 3}
    rows = [(record.status, record.semantic_attempt_id) for record in runner.ledger.load().records
            if record.scenario_id == sid_of(inputs, 1)]
    assert rows == [(STATUS_TRANSPORT_FAILED, 0), (STATUS_ACCEPTED, 0)]


def _failing_request(failing, answer, fail):
    """A provider's answer to each request: the second request of every batch fails while ``failing["on"]``."""
    return lambda custom_id, body: fail(custom_id, body) if failing["on"] and custom_id.endswith("-p000001-c00") \
        else answer(custom_id, body)


def _campaign_buys_the_failed_request_again(tmp_path, inputs, modelspec, backend_of, created):
    """The cap stops a provider failing one request; a recovered resume buys it once more, never a completed batch."""
    failing = {"on": True}
    runner = build_runner(tmp_path, inputs, modelspec, backend_of(failing), scenarios_per_request_block=3)
    with pytest.raises(TransportExhausted, match="failed in transport 2 time"):
        runner.run()
    assert [len(batch) for batch in created()] == [9, 3]
    failing["on"] = False
    resumed = build_runner(tmp_path, inputs, modelspec, backend_of(failing), scenarios_per_request_block=3)
    assert resumed.run()["state"] == STATE_COMPLETE
    assert [len(batch) for batch in created()] == [9, 3, 3]
    rows = [(r.status, r.semantic_attempt_id) for r in resumed.ledger.load().records if r.scenario_id == sid_of(inputs, 0)]
    assert rows == [(STATUS_TRANSPORT_FAILED, 0)] * 2 + [(STATUS_ACCEPTED, 0)]
    blocks = read_jsonl(resumed.draw_dir / REQUEST_BLOCKS_FNAME)
    assert [(row["semantic_attempt_id"], row["scenario_ids"]) for row in blocks][1:] == [(0, [sid_of(inputs, 0)])] * 2


def test_a_request_that_errored_inside_a_completed_anthropic_batch_is_bought_again_in_a_fresh_block(
        tmp_path, inputs, modelspec, monkeypatch):
    from omegaconf import OmegaConf

    from ruleforge.judges import anthropic_batch
    from ruleforge.judges.backend import ModelBackend
    from tests.test_anthropic_batch import FakeAnthropic, failed, succeeded, text

    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    clients = []

    def backend_of(failing):
        clients.append(FakeAnthropic(responder=_failing_request(
            failing, lambda cid, params: succeeded(cid, text("no")), lambda cid, params: failed(cid, "errored"))))
        monkeypatch.setattr(anthropic_batch, "default_client", lambda: clients[-1])
        return ModelBackend(OmegaConf.create({
            "model_type": "anthropic_batch", "model_name": "claude-haiku-5-5", "batch_root": str(tmp_path / "batch"),
            "send_max_output_tokens": True, "max_new_tokens": 100, "fixed_temperature": 1.0, "temperature": 1.0}))

    _campaign_buys_the_failed_request_again(tmp_path, inputs, modelspec, backend_of,
                                            lambda: [batch for client in clients for batch in client.create_calls])


def test_a_responses_body_that_failed_inside_a_completed_openai_batch_is_bought_again_in_a_fresh_block(
        tmp_path, inputs, modelspec, monkeypatch):
    from omegaconf import OmegaConf

    from ruleforge.judges import openai_batch
    from ruleforge.judges.backend import ModelBackend
    from tests.test_openai_batch import FakeOpenAI, success_record

    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)

    def answered(custom_id, body):
        record = success_record(custom_id, body)
        record["response"]["body"]["output"][0]["content"][0]["text"] = "no"
        return record

    def failed_body(custom_id, body):
        record = answered(custom_id, body)
        record["response"]["body"].update(status="failed", error={"code": "server_error", "message": "try again"})
        return record

    client = FakeOpenAI()

    def backend_of(failing):
        client.responder = _failing_request(failing, answered, failed_body)
        backend = ModelBackend(OmegaConf.create({"model_type": "openai_batch", "model_name": "test-model",
                                                 "batch_root": str(tmp_path / "batch")}))
        backend._generate_request_block = openai_batch.make_block_generate(
            model_name="test-model", root=tmp_path / "batch", client_factory=lambda: client)
        return backend

    def created():
        return [client.uploads[call["input_file_id"]].decode().splitlines() for call in client.create_calls]

    _campaign_buys_the_failed_request_again(tmp_path, inputs, modelspec, backend_of, created)
