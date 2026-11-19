from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from ruleforge.judges import anthropic_batch, batch_api, batch_collect, openai_batch
from ruleforge.judges.batch_api import (
    CHAT_COMPLETIONS_ENDPOINT,
    PARTIAL_FNAME,
    RESPONSES_ENDPOINT,
    BatchExecutionError,
    build_batch_requests,
    prompt_of_body,
    write_json_atomic,
)
from ruleforge.judges.batch_collect import NO_RESULT, BatchCollectRefused, collect, main, transport_of
from tests.test_anthropic_batch import FakeAnthropic, failed, success
from tests.test_anthropic_batch import requests_for as anthropic_requests_for

NAMESPACE = "ns0123456789abcdef"
EXPIRED, CANCELLED, FAILED = "expired", "cancelled", "failed"


def no_client() -> Any:
    raise AssertionError("no provider client may be built on this path")


def usage_of(index: int) -> Dict[str, Any]:
    """Request ``index``'s usage: every bucket distinct per request, so a sum names which requests it counted."""
    return {"input_tokens": 100 + index, "output_tokens": 10 + index,
            "input_tokens_details": {"cached_tokens": index, "cache_write_tokens": 2 * index},
            "output_tokens_details": {"reasoning_tokens": index}}


def answered(custom_id: str, index: int) -> Dict[str, Any]:
    return {"custom_id": custom_id, "response": {"status_code": 200, "body": {
        "status": "completed", "usage": usage_of(index),
        "output": [{"type": "message", "content": [{"type": "output_text", "text": f"no {index}"}]}]}}}


class EndedOpenAI:
    """The OpenAI SDK client as the collect calls it: ``batches.retrieve`` and ``files.content``, nothing else."""

    def __init__(self, lines: Dict[str, list], endings: Dict[str, Dict[str, Any]]) -> None:
        self.lines, self.endings = lines, endings
        self.calls: list[tuple[str, str]] = []
        self.batches = SimpleNamespace(retrieve=self._retrieve)
        self.files = SimpleNamespace(content=self._content)

    def _ending(self, batch_id: str) -> Dict[str, Any]:
        ending = dict(self.endings[batch_id])
        indices = range(len(self.lines[batch_id]))
        ending.setdefault("returned", list(indices))
        returned = ending["returned"] or []
        code = {EXPIRED: "batch_expired", CANCELLED: "batch_cancelled"}.get(ending["status"])
        ending.setdefault("errors", {index: {"error": {"code": code, "message": "not finished"}}
                                     for index in indices if index not in returned and code})
        ending.setdefault("counts", {"total": len(self.lines[batch_id]), "completed": len(returned),
                                     "failed": len(ending["errors"])})
        return ending

    def _retrieve(self, batch_id: str):
        self.calls.append(("retrieve", batch_id))
        ending = self._ending(batch_id)
        counts = ending["counts"]
        return SimpleNamespace(
            id=batch_id, status=ending["status"],
            output_file_id=None if ending["returned"] is None else f"out-{batch_id}",
            error_file_id=f"err-{batch_id}" if ending["errors"] else None,
            request_counts=None if counts is None else SimpleNamespace(model_dump=lambda: dict(counts)))

    def _content(self, file_id: str):
        self.calls.append(("content", file_id))
        kind, _, batch_id = file_id.partition("-")
        ending, lines = self._ending(batch_id), self.lines[batch_id]
        if kind == "out":
            records = [answered(lines[index]["custom_id"], index) for index in ending["returned"]]
            records += ending.get("extra", [])
        else:
            records = [{"custom_id": lines[index]["custom_id"], **body} for index, body in ending["errors"].items()]
        payload = "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")
        return SimpleNamespace(write_to_file=lambda path: Path(path).write_bytes(payload))


def submitted(root: Path, count: int = 4, *, per_shard: int = 4, endpoint: str = RESPONSES_ENDPOINT,
              namespace: Optional[str] = NAMESPACE):
    """Prepare ``count`` requests under ``root`` with the real OpenAI runner and record each shard as submitted."""
    requests = build_batch_requests(
        [{"system_prompt": f"judge {index}"} for index in range(count)], endpoint=endpoint, model="gpt-6.1-sol",
        temperature=1.0, num_responses=1, seed=None, namespace=namespace)
    runner = openai_batch.OpenAIBatchRunner(root=root, endpoint=endpoint, client_factory=no_client,
                                            max_shard_requests=per_shard)
    runner.prepare(requests)
    directory = runner.workdir(requests)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    lines = {}
    for number, shard in enumerate(state["shards"]):
        batch_id = f"batch-{number}"
        state["parts"][shard["path"]] = {"batch_id": batch_id, "file_id": f"file-{number}", "status": "in_progress"}
        lines[batch_id] = [json.loads(line) for line in (directory / "inputs" / shard["path"]).read_text(
            encoding="utf-8").splitlines()]
    write_json_atomic(directory / "state.json", state)
    return directory, lines


def collect_openai(directory: Path, client: EndedOpenAI) -> Dict[str, Any]:
    return collect(directory, openai_client=lambda: client)


def snapshot(directory: Path) -> Dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*"))
            if path.is_file()}


def test_an_expired_batch_is_collected_from_its_outputs_and_error_file_and_marked_partial(tmp_path):
    directory, lines = submitted(tmp_path / "batch", per_shard=2)
    client = EndedOpenAI(lines, {"batch-0": {"status": "completed"}, "batch-1": {"status": EXPIRED, "returned": [0]}})
    marker = collect_openai(directory, client)
    ids = [line["custom_id"] for shard in lines.values() for line in shard]
    assert marker == {
        "kind": "smrc-partial-batch", "schema_version": 1, "transport": "openai_batch", "endpoint": RESPONSES_ENDPOINT,
        "requests_expected": 4, "requests_returned": 3, "requests_missing": 1, "missing_reasons": {"batch_expired": 1},
        "missing_custom_ids": [ids[3]], "custom_id_prefix": NAMESPACE,
        "parts": {"part-0000.jsonl": {"state": "completed", "id": "batch-0",
                                      "provider_counts": {"total": 2, "completed": 2, "failed": 0}},
                  "part-0001.jsonl": {"state": EXPIRED, "id": "batch-1",
                                      "provider_counts": {"total": 2, "completed": 1, "failed": 1}}},
        "usage": {"requests": 3, "input_tokens": 100 + 101 + 100, "cached_tokens": 0 + 1 + 0,
                  "cache_write_tokens": 0 + 2 + 0, "output_tokens": 10 + 11 + 10, "reasoning_tokens": 0 + 1 + 0}}
    assert json.loads((directory / "usage.json").read_text()) == marker["usage"]
    assert json.loads((directory / PARTIAL_FNAME).read_text()) == marker
    assert sorted(path.name for path in (directory / "outputs").iterdir()) == [
        "part-0000.jsonl", "part-0001.errors.jsonl", "part-0001.jsonl"]
    state = json.loads((directory / "state.json").read_text())
    assert [part["status"] for part in state["parts"].values()] == ["completed", EXPIRED]
    assert client.calls == [("retrieve", "batch-0"), ("retrieve", "batch-1"), ("content", "out-batch-0"),
                            ("content", "out-batch-1"), ("content", "err-batch-1")]
    assert batch_collect.summary(directory, marker) == (
        f"{directory.name}: openai_batch (completed, expired): 3 of 4 request(s) returned, 1 missing (batch_expired "
        f"1); usage.json and partial.json written")


def test_a_collect_run_again_writes_the_same_bytes_and_downloads_nothing_again(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0, 2]}})
    first = collect_openai(directory, client)
    before = snapshot(directory)
    client.calls.clear()
    assert collect_openai(directory, client) == first
    assert snapshot(directory) == before
    assert client.calls == [("retrieve", "batch-0")]


def test_a_cancelled_batch_and_its_failed_and_unreturned_requests(tmp_path):
    """A cancelled batch: two answers, one errored output line (status 500: billed nothing), one in neither file."""
    directory, lines = submitted(tmp_path / "batch", count=6, per_shard=6)
    failed = {"custom_id": lines["batch-0"][2]["custom_id"], "response": {"status_code": 500, "body": {}}}
    client = EndedOpenAI(lines, {"batch-0": {
        "status": CANCELLED, "returned": [0, 1], "extra": [failed],
        "errors": {3: {"error": {"code": "batch_cancelled"}}, 4: {"response": {"status_code": 429}}, 5: {}}}})
    marker = collect_openai(directory, client)
    assert (marker["requests_returned"], marker["requests_missing"]) == (2, 4)
    assert marker["missing_reasons"] == {"batch_cancelled": 1, "error": 1, "status 429": 1, "status 500": 1}
    assert marker["usage"]["requests"] == 2 and marker["parts"]["part-0000.jsonl"]["state"] == CANCELLED


def test_a_failed_batch_with_no_output_file_bills_nothing(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    marker = collect_openai(directory, EndedOpenAI(lines, {"batch-0": {"status": FAILED, "returned": None,
                                                                       "counts": None}}))
    assert (marker["requests_returned"], marker["missing_reasons"]) == (0, {NO_RESULT: 4})
    assert marker["usage"] == {"requests": 0, "input_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0,
                               "output_tokens": 0, "reasoning_tokens": 0}
    assert marker["parts"]["part-0000.jsonl"]["provider_counts"] is None
    assert not list((directory / "outputs").iterdir())


@pytest.mark.parametrize("status", ["validating", "in_progress", "finalizing", "cancelling"])
def test_a_batch_still_running_is_refused_and_nothing_is_written(tmp_path, status):
    directory, lines = submitted(tmp_path / "batch", per_shard=2)
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0]}, "batch-1": {"status": status}})
    with pytest.raises(BatchCollectRefused, match=rf"\{{'part-0001.jsonl': '{status}'\}} have not ended.*never "
                                                   "cancels"):
        collect_openai(directory, client)
    assert sorted(path.name for path in directory.iterdir()) == ["inputs", "state.json"]
    assert client.calls == [("retrieve", "batch-0"), ("retrieve", "batch-1")]


def test_a_batch_whose_every_part_completed_is_the_normal_path_s(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    with pytest.raises(BatchCollectRefused, match="every part completed.*relaunch its campaign"):
        collect_openai(directory, EndedOpenAI(lines, {"batch-0": {"status": "completed"}}))
    assert not (directory / "usage.json").exists()


def test_a_part_with_completed_requests_and_no_output_file_yet_is_refused(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": None,
                                             "counts": {"total": 4, "completed": 3, "failed": 1}}})
    with pytest.raises(BatchCollectRefused, match="counts 3 completed request.*names no output file"):
        collect_openai(directory, client)
    assert not (directory / "outputs").exists()


def test_an_openai_output_file_cut_short_is_refused_not_billed_short(tmp_path):
    """An expired batch of 3 whose provider counts 2 completed."""
    directory, lines = submitted(tmp_path / "batch", count=3, per_shard=3)
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0, 1]}})
    (directory / "outputs").mkdir()
    short = directory / "outputs" / "part-0000.jsonl"
    short.write_text(json.dumps(answered(lines["batch-0"][0]["custom_id"], 0)) + "\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(BatchExecutionError, match=r"part-0000.jsonl answers 1 request\(s\) without an error, but "
                                                      r"OpenAI counted 2 completed in shard part-0000.jsonl \(batch "
                                                      r"batch-0\).*Delete it and rerun"):
            collect_openai(directory, client)
        assert not (directory / "usage.json").exists() and not (directory / PARTIAL_FNAME).exists()
    assert ("content", "out-batch-0") not in client.calls
    short.unlink()
    marker = collect_openai(directory, client)
    assert (marker["requests_returned"], marker["missing_reasons"]) == (2, {"batch_expired": 1})
    assert marker["usage"]["requests"] == 2 and marker["usage"]["input_tokens"] == 100 + 101


def test_an_openai_output_file_with_no_provider_counts_to_hold_it_to_is_refused(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0], "counts": None}})
    with pytest.raises(BatchCollectRefused, match="names an output file for part-0000.jsonl.*gives no request counts"):
        collect_openai(directory, client)
    assert not (directory / "outputs").exists()


def test_an_answer_to_a_request_the_batch_never_sent_is_refused(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0], "extra": [answered("stranger", 9)],
                                             "counts": {"total": 4, "completed": 2, "failed": 3}}})
    with pytest.raises(BatchExecutionError, match="answers unknown custom id 'stranger'"):
        collect_openai(directory, client)
    assert not (directory / PARTIAL_FNAME).exists()


def test_a_directory_is_collected_only_as_the_batch_its_name_and_files_say(tmp_path):
    directory, lines = submitted(tmp_path / "batch")
    client = EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0]}})
    renamed = tmp_path / "batch" / ("0" * 32)
    shutil.copytree(directory, renamed)
    with pytest.raises(BatchCollectRefused, match=f"address {directory.name}, not this directory"):
        collect_openai(renamed, client)
    shard = directory / "inputs" / "part-0000.jsonl"
    original = shard.read_bytes()
    shard.write_bytes(original.replace(b"judge 0", b"judge 9"))
    with pytest.raises(BatchCollectRefused, match="not the input file its state.json recorded"):
        collect_openai(directory, client)
    # a line for another endpoint, its digest made to match: refused before any address is computed
    shard.write_bytes(original.replace(RESPONSES_ENDPOINT.encode(), CHAT_COMPLETIONS_ENDPOINT.encode(), 1))
    state = json.loads((directory / "state.json").read_text())
    state["shards"][0]["sha256"] = openai_batch.sha256_bytes(shard.read_bytes())
    write_json_atomic(directory / "state.json", state)
    with pytest.raises(BatchCollectRefused, match="is for '/v1/chat/completions', not the batch's '/v1/responses'"):
        collect_openai(directory, client)
    assert client.calls == []


def test_only_an_openai_or_an_anthropic_work_directory_is_collected(tmp_path):
    with pytest.raises(BatchCollectRefused, match="holds no state.json"):
        collect(tmp_path)
    inline = {"state_version": 1, "endpoint": CHAT_COMPLETIONS_ENDPOINT, "model": "m", "requests": 1, "parts": {}}
    for state in (inline, {"endpoint": "/v1/embeddings", "shards": []}):
        with pytest.raises(BatchCollectRefused, match="neither an OpenAI Batch work directory.s nor an Anthropic one.s"):
            transport_of(state)
    assert transport_of({"endpoint": RESPONSES_ENDPOINT, "shards": []}) == "openai_batch"
    assert transport_of({"endpoint": anthropic_batch.ANTHROPIC_BATCH_ENDPOINT, "model": "m", "shards": []}) == (
        "anthropic_batch")


def test_a_batch_of_another_kind_of_custom_id_carries_no_prefix(tmp_path):
    """Custom ids keyed by a seed share no request namespace (a non-campaign batch)."""
    directory, lines = submitted(tmp_path / "batch", namespace=None)
    marker = collect_openai(directory, EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [1]}}))
    assert marker["custom_id_prefix"] == "snone"
    assert batch_api.custom_id_prefix(["a-p000000-c00", "b-p000001-c00"]) is None


def test_the_refresh_reads_every_part_and_refuses_one_never_submitted(tmp_path):
    directory, lines = submitted(tmp_path / "batch", per_shard=2)
    state = json.loads((directory / "state.json").read_text())
    state["parts"]["part-0000.jsonl"]["status"] = "completed"
    del state["parts"]["part-0001.jsonl"]["batch_id"]
    write_json_atomic(directory / "state.json", state)
    client = EndedOpenAI(lines, {"batch-0": {"status": "completed"}})
    with pytest.raises(BatchExecutionError, match="shard part-0001.jsonl was never submitted"):
        collect_openai(directory, client)
    assert client.calls == [("retrieve", "batch-0")]


def anthropic_submitted(tmp_path: Path, client: FakeAnthropic, *, per_part: Optional[int] = None):
    """Three requests prepared and submitted by the real Anthropic runner, each batch held by the fake client."""
    requests = anthropic_requests_for(namespace=NAMESPACE)
    runner = anthropic_batch.AnthropicBatchRunner(root=tmp_path / "batch", model="claude-haiku-5-5",
                                                  client_factory=lambda: client,
                                                  max_shard_requests=per_part or len(requests))
    runner.submit(requests)
    return runner.workdir(requests), [request.custom_id for request in requests]


@pytest.fixture
def spend(monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)


def test_an_anthropic_batch_that_ended_with_canceled_requests_is_collected_partial(tmp_path, spend):
    """A Message Batch ends ``ended`` whatever happened."""
    kinds = {"p000000": None, "p000001": "canceled", "p000002": "expired"}

    def respond(custom_id, params):
        kind = kinds[custom_id.split("-")[1]]
        return success(custom_id, params) if kind is None else failed(custom_id, kind)

    client = FakeAnthropic(responder=respond)
    directory, keys = anthropic_submitted(tmp_path, client)
    marker = collect(directory, anthropic_client=lambda: client, openai_client=no_client)
    assert marker["transport"] == "anthropic_batch" and marker["custom_id_prefix"] == NAMESPACE
    assert (marker["requests_expected"], marker["requests_returned"], marker["missing_custom_ids"]) == (
        3, 1, keys[1:])
    assert marker["missing_reasons"] == {"canceled": 1, "expired": 1}
    assert marker["parts"]["part-0000.jsonl"]["state"] == anthropic_batch.PART_INCOMPLETE
    assert marker["parts"]["part-0000.jsonl"]["provider_counts"]["canceled"] == 1
    assert marker["usage"] == json.loads((directory / "usage.json").read_text()) and marker["usage"]["requests"] == 1
    before = snapshot(directory)
    assert collect(directory, anthropic_client=lambda: client) == marker
    assert snapshot(directory) == before and client.results_calls == ["msgbatch_00"]
    assert len(client.create_calls) == 1


def test_an_anthropic_results_file_cut_short_is_refused_not_billed_short(tmp_path, spend):
    client = FakeAnthropic(responder=lambda cid, params: (
        failed(cid, "expired") if cid.endswith("p000002-c00") else success(cid, params)))
    directory, _ = anthropic_submitted(tmp_path, client)
    whole = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in client.records("msgbatch_00")]
    (directory / "outputs").mkdir()
    (directory / "outputs" / "part-0000.jsonl").write_text(whole[0] + "\n" + whole[2] + "\n")
    with pytest.raises(BatchExecutionError, match="holds .*'succeeded': 1.* but the API counted .*'succeeded': 2"):
        collect(directory, anthropic_client=lambda: client)
    assert client.results_calls == []
    assert not (directory / "usage.json").exists() and not (directory / batch_api.PARTIAL_FNAME).exists()


def test_an_anthropic_batch_of_two_parts_one_complete_is_collected_whole(tmp_path, spend):
    client = FakeAnthropic(responder=lambda cid, params: failed(cid, "errored") if cid.endswith("p000000-c00")
                           else failed(cid, "expired") if cid.endswith("p000002-c00") else success(cid, params))
    directory, keys = anthropic_submitted(tmp_path, client, per_part=2)
    marker = collect(directory, anthropic_client=lambda: client)
    assert {name: part["state"] for name, part in marker["parts"].items()} == {
        "part-0000.jsonl": anthropic_batch.PART_COMPLETED, "part-0001.jsonl": anthropic_batch.PART_INCOMPLETE}
    assert marker["missing_reasons"] == {"errored:overloaded_error": 1, "expired": 1}
    assert marker["requests_returned"] == 1


@pytest.mark.parametrize("statuses, refusal", [(["in_progress"], "have not ended.*an Anthropic Message Batch's "
                                                                  "requests 24 h after it was created"),
                                               (["canceling"], "have not ended"),
                                               (["ended"], "every part completed")])
def test_an_anthropic_batch_running_or_completed_is_refused(tmp_path, spend, statuses, refusal):
    client = FakeAnthropic(statuses=statuses)
    directory, _ = anthropic_submitted(tmp_path, client)
    with pytest.raises(BatchCollectRefused, match=refusal):
        collect(directory, anthropic_client=lambda: client)
    assert not (directory / "outputs").exists() and len(client.create_calls) == 1


def test_an_anthropic_directory_is_collected_only_as_the_batch_its_name_says(tmp_path, spend):
    client = FakeAnthropic(responder=lambda cid, params: failed(cid, "canceled"))
    directory, _ = anthropic_submitted(tmp_path, client)
    renamed = directory.parent / ("2" * 32)
    shutil.copytree(directory, renamed)
    with pytest.raises(BatchCollectRefused, match="not the batch its name says"):
        collect(renamed, anthropic_client=lambda: client)
    assert client.retrieve_calls == []


@pytest.mark.parametrize("endpoint", [RESPONSES_ENDPOINT, CHAT_COMPLETIONS_ENDPOINT])
@pytest.mark.parametrize("entry", [{"system_prompt": "judge", "user_prompt": ""},
                                   {"system_prompt": "judge", "user_prompt": "ask"},
                                   {"system_prompt": "", "user_prompt": "ask"}])
def test_a_body_reads_back_as_the_prompt_it_was_built_from(endpoint, entry):
    [request] = build_batch_requests([entry], endpoint=endpoint, model="m", temperature=1.0, num_responses=1,
                                     seed=None)
    assert prompt_of_body(request.body, endpoint) == entry
    empty = {"system_prompt": "", "user_prompt": ""}
    assert prompt_of_body({}, endpoint) == empty
    assert prompt_of_body({"input": [{"role": "assistant", "content": "x"}],
                           "messages": [{"role": "assistant", "content": "x"}]}, endpoint) == empty


def test_the_command_collects_each_directory_and_exits_2_on_a_refusal(tmp_path, monkeypatch, capsys):
    done, lines = submitted(tmp_path / "batch")
    running, more = submitted(tmp_path / "other")
    client = EndedOpenAI({**lines}, {"batch-0": {"status": EXPIRED, "returned": [0]}})
    monkeypatch.setattr(openai_batch, "default_client", lambda: client)
    assert main([str(done)]) == 0
    assert capsys.readouterr().out.strip() == batch_collect.summary(
        done, json.loads((done / PARTIAL_FNAME).read_text()))
    client.endings["batch-0"] = {"status": "in_progress"}
    client.lines = more
    assert main([str(running), str(tmp_path / "nothing")]) == 2
    err = capsys.readouterr().err
    assert "have not ended" in err and "holds no state.json" in err


def test_the_command_without_a_key_is_refused_by_name(tmp_path, monkeypatch, capsys):
    directory, _ = submitted(tmp_path / "batch")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main([str(directory)]) == 2
    assert "OPENAI_API_KEY environment variable is required" in capsys.readouterr().err


def test_ruleforge_campaign_collect_runs_the_command(tmp_path, monkeypatch, capsys):
    from ruleforge.cli import main as cli_main

    directory, lines = submitted(tmp_path / "batch")
    monkeypatch.setattr(openai_batch, "default_client",
                        lambda: EndedOpenAI(lines, {"batch-0": {"status": EXPIRED, "returned": [0, 1, 2]}}))
    assert cli_main(["campaign", "collect", str(directory)]) == 0
    assert "3 of 4 request(s) returned" in capsys.readouterr().out
