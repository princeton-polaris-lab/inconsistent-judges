from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from ruleforge.judges import openai_batch
from ruleforge.judges.models import OPENAI_REDIRECT_ENVS
from ruleforge.judges.batch_api import (
    CHAT_COMPLETIONS_ENDPOINT,
    RESPONSES_ENDPOINT,
    BatchExecutionError,
    BatchRequest,
    BatchSpendNotConfirmed,
    billed_body,
    build_batch_requests,
    request_line,
)
from ruleforge.judges.openai_batch import (
    STATE_VERSION,
    OpenAIBatchRunner,
    default_client,
    make_batch_generate,
    write_shards,
)
from ruleforge.judges.token_usage import TokenUsage


USAGE = {
    "input_tokens": 11,
    "output_tokens": 5,
    "input_tokens_details": {"cached_tokens": 4},
    "output_tokens_details": {"reasoning_tokens": 2},
}


def success_record(custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """A completed /v1/responses result whose text names its custom_id."""
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": f"answer::{custom_id}"}
                        ],
                    }
                ],
                "usage": USAGE,
            },
        },
    }


class FakeOpenAI:
    """Scriptable stand-in for the OpenAI SDK client."""

    def __init__(
        self,
        *,
        status_script: Optional[List[str]] = None,
        responder=success_record,
        emit_error_file: bool = False,
        request_counts: str = "model_dump",
        shuffle: bool = False,
        create_failures: Optional[List[Any]] = None,
    ) -> None:
        self.uploads: Dict[str, bytes] = {}
        self.upload_calls: List[Dict[str, Any]] = []
        self.create_calls: List[Dict[str, Any]] = []
        self.option_calls: List[Dict[str, Any]] = []
        # ("before" | "after", error): a create that raises before or after it made its batch
        self.create_failures = list(create_failures or [])
        self.retrieved: List[str] = []
        self.downloaded: List[str] = []
        self.status_script = list(status_script or [])
        self.responder = responder
        self.emit_error_file = emit_error_file
        self.request_counts = request_counts
        self.shuffle = shuffle
        self._batch_input: Dict[str, str] = {}
        self.files = SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve)

    def _create_file(self, *, file, purpose):
        payload = file.read()
        file_id = f"file-{len(self.uploads):02d}"
        self.uploads[file_id] = payload
        self.upload_calls.append(
            {"purpose": purpose, "name": Path(file.name).name, "bytes": payload}
        )
        return SimpleNamespace(id=file_id)

    def _content(self, file_id: str):
        self.downloaded.append(file_id)
        kind, _, batch_id = file_id.partition("-")
        payload = {
            "out": self._output_payload,
            "err": self._error_payload,
        }[kind](batch_id)
        return SimpleNamespace(
            write_to_file=lambda path: Path(path).write_bytes(payload)
        )

    def _shard_lines(self, batch_id: str) -> List[Dict[str, Any]]:
        shard = self.uploads[self._batch_input[batch_id]]
        return [json.loads(line) for line in shard.splitlines()]

    def _output_payload(self, batch_id: str) -> bytes:
        records = []
        for payload in self._shard_lines(batch_id):
            record = self.responder(payload["custom_id"], payload["body"])
            if record is not None:
                records.append(json.dumps(record))
        if self.shuffle:
            records.reverse()
        return "".join(f"{record}\n" for record in records).encode("utf-8")

    def _error_payload(self, batch_id: str) -> bytes:
        custom_id = self._shard_lines(batch_id)[0]["custom_id"]
        record = {"custom_id": custom_id, "error": {"message": "rate limited"}}
        return (json.dumps(record) + "\n").encode("utf-8")

    def with_options(self, **options):
        self.option_calls.append(options)
        return self

    def _create_batch(self, **kwargs):
        failure = self.create_failures.pop(0) if self.create_failures else None
        self.create_calls.append(kwargs)
        if failure is not None and failure[0] == "before":
            raise failure[1]
        batch_id = f"batch{len(self.create_calls) - 1:02d}"
        self._batch_input[batch_id] = kwargs["input_file_id"]
        if failure is not None:
            raise failure[1]
        return SimpleNamespace(id=batch_id, status="validating")

    def _counts(self, batch_id: str) -> Dict[str, int]:
        total = len(self._shard_lines(batch_id))
        completed = sum(
            billed_body(json.loads(line)) is not None
            for line in self._output_payload(batch_id).decode("utf-8").splitlines()
        )
        return {"total": total, "completed": completed, "failed": total - completed}

    def _retrieve(self, batch_id: str):
        self.retrieved.append(batch_id)
        status = self.status_script.pop(0) if self.status_script else "completed"
        done = status == "completed"
        counts = {
            "model_dump": SimpleNamespace(
                model_dump=lambda: self._counts(batch_id)
            ),
            "plain": SimpleNamespace(total=2),
            "none": None,
        }[self.request_counts]
        return SimpleNamespace(
            id=batch_id,
            status=status,
            output_file_id=f"out-{batch_id}" if done else None,
            error_file_id=f"err-{batch_id}" if done and self.emit_error_file else None,
            request_counts=counts,
        )


class FakeClock:
    """Monotonic stand-in returning scripted ticks, then holding the last."""

    def __init__(self, ticks: List[float]) -> None:
        self.ticks = ticks
        self.calls = 0

    def __call__(self) -> float:
        tick = self.ticks[min(self.calls, len(self.ticks) - 1)]
        self.calls += 1
        return tick


def explode() -> Any:
    """A client factory that fails if anything tries to construct a client."""
    raise AssertionError("no API client may be constructed on this path")


class Tracker:
    """Minimal ModelBackend.token_tracker stand-in."""

    def __init__(self) -> None:
        self.added: List[TokenUsage] = []

    def add(self, usage: TokenUsage) -> None:
        self.added.append(usage)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """Neither a real key, a spend confirmation nor a redirecting base URL may leak in from the shell."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for name in OPENAI_REDIRECT_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def confirmed(monkeypatch):
    """The environment gate that stands in front of every billable call."""
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)


def make_requests(count: int = 2, prefix: str = "p") -> List[BatchRequest]:
    """``count`` requests with identical body sizes and distinct ids."""
    return build_batch_requests(
        [{"system_prompt": f"{prefix}{index}"} for index in range(count)],
        endpoint=RESPONSES_ENDPOINT,
        model="test-model",
        temperature=1.0,
        num_responses=1,
        seed=0,
    )


def build_runner(root: Path, client: Optional[FakeOpenAI] = None, **kwargs):
    """Runner wired to ``client``; without one, constructing a client fails."""
    return OpenAIBatchRunner(
        root=root,
        client_factory=(lambda: client) if client is not None else explode,
        **kwargs,
    )


def read_state(runner: OpenAIBatchRunner, requests) -> Dict[str, Any]:
    return json.loads(
        (runner.workdir(requests) / "state.json").read_text(encoding="utf-8")
    )


def write_state(runner: OpenAIBatchRunner, requests, state: Dict[str, Any]) -> None:
    (runner.workdir(requests) / "state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )


def test_write_shards_refuses_an_empty_request_list(tmp_path):
    with pytest.raises(ValueError, match="no requests"):
        write_shards([], tmp_path / "inputs", endpoint=RESPONSES_ENDPOINT)

    assert not (tmp_path / "inputs").exists()


def test_write_shards_names_the_request_that_can_never_fit(tmp_path):
    requests = make_requests(2)
    with pytest.raises(ValueError, match=r"request 's0-p000000-c00' is \d+ bytes"):
        write_shards(
            requests, tmp_path / "inputs", endpoint=RESPONSES_ENDPOINT, max_bytes=10
        )


def test_write_shards_rotates_on_the_byte_budget(tmp_path):
    requests = make_requests(5)
    line_bytes = len(request_line(requests[0], RESPONSES_ENDPOINT))
    directory = tmp_path / "inputs"

    shards = write_shards(
        requests,
        directory,
        endpoint=RESPONSES_ENDPOINT,
        max_bytes=2 * line_bytes,
    )

    assert [shard["path"] for shard in shards] == [
        "part-0000.jsonl",
        "part-0001.jsonl",
        "part-0002.jsonl",
    ]
    assert [shard["requests"] for shard in shards] == [2, 2, 1]
    assert all(shard["bytes"] <= 2 * line_bytes for shard in shards)
    written = b"".join((directory / shard["path"]).read_bytes() for shard in shards)
    assert written == b"".join(
        request_line(request, RESPONSES_ENDPOINT) for request in requests
    )


def test_write_shards_rotates_on_the_request_count(tmp_path):
    """The 50,000-request-per-file limit is enforced independently of bytes."""
    requests = make_requests(5)

    shards = write_shards(
        requests, tmp_path / "inputs", endpoint=RESPONSES_ENDPOINT, max_requests=2
    )

    assert [shard["requests"] for shard in shards] == [2, 2, 1]
    assert [shard["path"] for shard in shards] == [
        "part-0000.jsonl",
        "part-0001.jsonl",
        "part-0002.jsonl",
    ]


def test_write_shard_descriptors_describe_the_bytes_on_disk(tmp_path):
    requests = make_requests(3)
    directory = tmp_path / "inputs"

    shards = write_shards(requests, directory, endpoint=RESPONSES_ENDPOINT)

    assert len(shards) == 1
    payload = (directory / shards[0]["path"]).read_bytes()
    assert shards[0] == {
        "path": "part-0000.jsonl",
        "requests": 3,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert payload.count(b"\n") == 3


def test_default_client_requires_an_api_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        default_client()


def test_default_client_rejects_a_blank_api_key(monkeypatch):
    """A whitespace-only key is an unset key that the SDK would accept."""
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        default_client()


def test_default_client_passes_the_key_to_the_sdk(monkeypatch):
    seen = {}

    class FakeSDKClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "  sk-test  ")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeSDKClient))

    client = default_client()

    assert isinstance(client, FakeSDKClient)
    assert seen == {"api_key": "sk-test"}


def test_default_client_refuses_an_ambient_base_url_naming_it_not_its_value(monkeypatch):
    """The SDK reads OPENAI_BASE_URL on its own."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ValueError, match="OPENAI_BASE_URL set: the openai SDK would use it") as refused:
        default_client()
    assert "127.0.0.1" not in str(refused.value) and "sk-test" not in str(refused.value)


def test_runner_rejects_an_endpoint_it_cannot_read_results_from(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported batch endpoint '/v1/embeddings'"):
        build_runner(tmp_path, endpoint="/v1/embeddings")


@pytest.mark.parametrize("poll_seconds", [0, -1.0])
def test_runner_rejects_a_nonpositive_poll_interval(tmp_path, poll_seconds):
    with pytest.raises(ValueError, match="poll_seconds must be positive"):
        build_runner(tmp_path, poll_seconds=poll_seconds)


@pytest.mark.parametrize("timeout_seconds", [0, -5.0])
def test_runner_rejects_a_nonpositive_timeout(tmp_path, timeout_seconds):
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        build_runner(tmp_path, timeout_seconds=timeout_seconds)


def test_runner_accepts_no_timeout_and_a_string_root(tmp_path):
    runner = build_runner(str(tmp_path / "batches"), timeout_seconds=None)

    assert isinstance(runner.root, Path)
    assert runner.timeout_seconds is None


def test_workdir_is_the_same_directory_for_the_same_requests(tmp_path):
    runner = build_runner(tmp_path)

    assert runner.workdir(make_requests(3)) == runner.workdir(make_requests(3))
    assert runner.workdir(make_requests(3)).parent == tmp_path


def test_workdir_changes_when_any_request_changes(tmp_path):
    runner = build_runner(tmp_path)
    baseline = runner.workdir(make_requests(3))

    assert runner.workdir(make_requests(3, prefix="q")) != baseline
    assert runner.workdir(make_requests(4)) != baseline


def test_prepare_writes_shards_and_state_without_calling_the_api(tmp_path):
    runner = build_runner(tmp_path, max_shard_requests=2)
    requests = make_requests(3)

    state = runner.prepare(requests)

    directory = runner.workdir(requests)
    assert sorted(path.name for path in (directory / "inputs").iterdir()) == [
        "part-0000.jsonl",
        "part-0001.jsonl",
    ]
    assert state["state_version"] == STATE_VERSION
    assert state["endpoint"] == RESPONSES_ENDPOINT
    assert state["requests"] == 3
    assert state["parts"] == {"part-0000.jsonl": {}, "part-0001.jsonl": {}}
    assert state == read_state(runner, requests)
    assert runner._client is None


def test_prepare_returns_the_existing_state_instead_of_rewriting_it(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(2)
    runner.prepare(requests)

    state = read_state(runner, requests)
    state["parts"]["part-0000.jsonl"] = {"batch_id": "batch-already-paid-for"}
    write_state(runner, requests, state)

    resumed = runner.prepare(requests)

    assert resumed["parts"]["part-0000.jsonl"] == {"batch_id": "batch-already-paid-for"}
    assert read_state(runner, requests) == state


def test_a_state_another_process_wrote_after_the_existence_check_is_read_never_replaced(tmp_path, monkeypatch):
    requests = make_requests(2)
    theirs = build_runner(tmp_path / "other").prepare(requests)
    theirs["parts"]["part-0000.jsonl"] = {"batch_id": "batch-theirs"}
    runner = build_runner(tmp_path / "mine")
    state_path = runner.workdir(requests) / "state.json"
    real = openai_batch.write_bytes_atomic

    def racing(path, payload):
        real(path, payload)
        if not state_path.exists():  # the other process prepares, buys and records here
            openai_batch.write_json_atomic(state_path, theirs)

    monkeypatch.setattr(openai_batch, "write_bytes_atomic", racing)
    assert runner.prepare(requests)["parts"]["part-0000.jsonl"] == {"batch_id": "batch-theirs"}
    assert read_state(runner, requests) == theirs
    assert not list(state_path.parent.glob("*.tmp"))


def test_a_batch_created_by_another_process_before_the_claim_is_not_bought_again(tmp_path, confirmed):
    """submit reads the state in prepare, outside the creation claim."""
    requests = make_requests(2)
    client = FakeOpenAI()
    theirs = build_runner(tmp_path, client)
    theirs.prepare(requests)

    def factory():
        theirs.submit(requests)  # this process has read the state with no batch id; the other one buys meanwhile
        return client

    state = OpenAIBatchRunner(root=tmp_path, client_factory=factory).submit(requests)
    assert len(client.create_calls) == 1 and len(client.upload_calls) == 1
    assert state["parts"]["part-0000.jsonl"]["batch_id"] == "batch00"
    assert read_state(theirs, requests)["parts"]["part-0000.jsonl"]["batch_id"] == "batch00"


def test_load_state_refuses_a_state_file_from_another_version(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(2)
    runner.prepare(requests)
    state = read_state(runner, requests)
    state["state_version"] = STATE_VERSION + 1
    write_state(runner, requests, state)

    with pytest.raises(BatchExecutionError, match="state version"):
        runner.prepare(requests)


def test_load_state_refuses_a_state_file_prepared_for_another_endpoint(tmp_path):
    """Results are parsed by endpoint, so a mismatch reads bodies with the wrong extractor."""
    runner = build_runner(tmp_path)
    requests = make_requests(2)
    runner.prepare(requests)
    state = read_state(runner, requests)
    state["endpoint"] = CHAT_COMPLETIONS_ENDPOINT
    write_state(runner, requests, state)

    with pytest.raises(BatchExecutionError, match="prepared for endpoint"):
        runner.prepare(requests)


def test_submit_refuses_to_spend_without_the_environment_confirmation(tmp_path):
    client = FakeOpenAI()
    factory_calls = []

    runner = OpenAIBatchRunner(
        root=tmp_path,
        client_factory=lambda: factory_calls.append(1) or client,
    )
    requests = make_requests(2)

    with pytest.raises(BatchSpendNotConfirmed, match="--yes-spend"):
        runner.submit(requests)

    assert factory_calls == []
    assert client.upload_calls == []
    assert (runner.workdir(requests) / "inputs" / "part-0000.jsonl").is_file()


def test_submit_uploads_each_shard_and_records_its_batch(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(
        tmp_path,
        client,
        completion_window="24h",
        metadata={"model": "test-model"},
    )
    requests = make_requests(2)

    state = runner.submit(requests)

    assert [call["purpose"] for call in client.upload_calls] == ["batch"]
    assert client.upload_calls[0]["name"] == "part-0000.jsonl"
    assert client.upload_calls[0]["bytes"] == (
        runner.workdir(requests) / "inputs" / "part-0000.jsonl"
    ).read_bytes()
    assert client.create_calls == [
        {
            "input_file_id": "file-00",
            "endpoint": RESPONSES_ENDPOINT,
            "completion_window": "24h",
            "metadata": {"model": "test-model"},
        }
    ]
    part = dict(state["parts"]["part-0000.jsonl"])
    assert part.pop("create_started_at").endswith("+00:00")
    assert part == {
        "file_id": "file-00",
        "batch_id": "batch00",
        "status": "validating",
    }
    assert read_state(runner, requests) == state


def test_submit_sends_no_metadata_when_none_was_configured(tmp_path, confirmed):
    """metadata=None rather than an empty dict: the SDK rejects {} on some versions."""
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, metadata=None)

    runner.submit(make_requests(2))

    assert client.create_calls[0]["metadata"] is None


def test_submit_is_a_no_op_when_every_shard_already_has_a_batch(tmp_path, monkeypatch):
    client = FakeOpenAI()
    requests = make_requests(2)
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    paid_state = build_runner(tmp_path, client).submit(requests)
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: False)

    resumed = build_runner(tmp_path).submit(requests)

    assert resumed == paid_state
    assert len(client.create_calls) == 1
    assert len(client.upload_calls) == 1


class Refused(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_a_create_is_sent_without_sdk_retries_and_marked_before_it_is_sent(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client)
    requests = make_requests(2)
    seen: List[Dict[str, Any]] = []
    client.batches.create = lambda real=client._create_batch, **kwargs: seen.append(read_state(runner, requests)) or (
        real(**kwargs))
    part = runner.submit(requests)["parts"]["part-0000.jsonl"]
    assert client.option_calls == [{"max_retries": 0}] and len(client.create_calls) == 1
    assert seen[0]["parts"]["part-0000.jsonl"]["create_started_at"] and part["batch_id"] == "batch00"


@pytest.mark.parametrize("error", [Refused(500), Refused(408), TimeoutError("read timed out"),
                                   ConnectionResetError("reset")])
def test_a_create_that_may_have_made_its_batch_is_never_repeated(tmp_path, confirmed, error):
    client = FakeOpenAI(create_failures=[("after", error)])
    requests = make_requests(2)
    with pytest.raises(BatchExecutionError, match="does not say whether the batch was made") as raised:
        build_runner(tmp_path, client).submit(requests)
    assert raised.value.__cause__ is error
    part = read_state(build_runner(tmp_path), requests)["parts"]["part-0000.jsonl"]
    assert part["create_started_at"] and part["file_id"] == "file-00" and "batch_id" not in part
    for _ in range(2):
        with pytest.raises(BatchExecutionError, match=r"started at .* and recorded no batch.*input_file_id file-00"):
            build_runner(tmp_path, client).submit(requests)
    assert len(client.create_calls) == 1 and len(client.upload_calls) == 1


def test_a_create_the_api_refused_is_cleared_and_may_be_retried(tmp_path, confirmed):
    client = FakeOpenAI(create_failures=[("before", Refused(429))])
    requests = make_requests(2)
    with pytest.raises(Refused):
        build_runner(tmp_path, client).submit(requests)
    assert "create_started_at" not in read_state(build_runner(tmp_path), requests)["parts"]["part-0000.jsonl"]
    assert build_runner(tmp_path, client).submit(requests)["parts"]["part-0000.jsonl"]["batch_id"] == "batch01"
    assert len(client.create_calls) == 2 and len(client.upload_calls) == 1


def test_submit_refuses_a_shard_that_changed_after_it_was_prepared(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client)
    requests = make_requests(2)
    runner.prepare(requests)

    shard = runner.workdir(requests) / "inputs" / "part-0000.jsonl"
    shard.write_bytes(shard.read_bytes() + b'{"custom_id":"smuggled"}\n')

    with pytest.raises(BatchExecutionError, match="prepared shard changed on disk"):
        runner.submit(requests)

    assert client.upload_calls == []


def test_submit_reuses_an_uploaded_file_and_skips_a_submitted_shard(
    tmp_path, confirmed
):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, max_shard_requests=1)
    requests = make_requests(2)
    runner.prepare(requests)

    state = read_state(runner, requests)
    state["parts"]["part-0000.jsonl"] = {"batch_id": "batch-live", "status": "validating"}
    state["parts"]["part-0001.jsonl"] = {"file_id": "file-uploaded-earlier"}
    write_state(runner, requests, state)

    resumed = runner.submit(requests)

    assert client.upload_calls == []
    assert [call["input_file_id"] for call in client.create_calls] == [
        "file-uploaded-earlier"
    ]
    assert resumed["parts"]["part-0000.jsonl"]["batch_id"] == "batch-live"
    assert resumed["parts"]["part-0001.jsonl"]["batch_id"] == "batch00"


def test_poll_refuses_to_wait_on_a_shard_that_was_never_submitted(tmp_path):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client)
    requests = make_requests(2)
    runner.prepare(requests)

    with pytest.raises(BatchExecutionError, match="part-0000.jsonl was never submitted"):
        runner.poll(requests)


def test_poll_waits_while_a_batch_is_in_progress_and_records_the_result(
    tmp_path, confirmed
):
    client = FakeOpenAI(status_script=["validating", "in_progress", "completed"])
    sleeps: List[float] = []
    runner = build_runner(tmp_path, client, poll_seconds=7.5, sleep=sleeps.append)
    requests = make_requests(2)
    runner.submit(requests)

    state = runner.poll(requests)

    assert sleeps == [7.5, 7.5]
    assert client.retrieved == ["batch00", "batch00", "batch00"]
    assert state["parts"]["part-0000.jsonl"]["status"] == "completed"
    assert state["parts"]["part-0000.jsonl"]["output_file_id"] == "out-batch00"
    assert state["parts"]["part-0000.jsonl"]["error_file_id"] is None
    assert state["parts"]["part-0000.jsonl"]["request_counts"] == {
        "total": 2,
        "completed": 2,
        "failed": 0,
    }
    assert state["parts"]["part-0000.jsonl"][openai_batch.COUNTS_STATUS] == "completed"
    assert read_state(runner, requests) == state


@pytest.mark.parametrize("request_counts", ["plain", "none"])
def test_poll_survives_a_provider_without_dumpable_request_counts(
    tmp_path, confirmed, request_counts
):
    client = FakeOpenAI(request_counts=request_counts)
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)

    state = runner.poll(requests)

    assert state["parts"]["part-0000.jsonl"]["status"] == "completed"
    assert "request_counts" not in state["parts"]["part-0000.jsonl"]


def test_poll_gives_up_after_the_timeout_and_says_where_the_state_is(
    tmp_path, confirmed
):
    client = FakeOpenAI(status_script=["in_progress", "in_progress"])
    sleeps: List[float] = []
    runner = build_runner(
        tmp_path,
        client,
        timeout_seconds=60.0,
        sleep=sleeps.append,
        monotonic=FakeClock([0.0, 1.0, 999.0]),
    )
    requests = make_requests(2)
    runner.submit(requests)

    with pytest.raises(BatchExecutionError) as excinfo:
        runner.poll(requests)

    assert "did not finish within 60.0s" in str(excinfo.value)
    assert str(runner.workdir(requests)) in str(excinfo.value)
    assert sleeps == [runner.poll_seconds]


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_poll_raises_when_a_shard_ends_in_a_terminal_failure(
    tmp_path, confirmed, status
):
    client = FakeOpenAI(status_script=[status])
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)

    with pytest.raises(BatchExecutionError, match="non-completed terminal"):
        runner.poll(requests)
    assert len(client.create_calls) == 1


def test_poll_does_not_re_retrieve_a_shard_already_terminal_in_state(
    tmp_path, confirmed
):
    client = FakeOpenAI(status_script=["completed"])
    runner = build_runner(
        tmp_path, client, max_shard_requests=1, sleep=lambda _: None
    )
    requests = make_requests(2)
    runner.submit(requests)

    state = read_state(runner, requests)
    state["parts"]["part-0000.jsonl"]["status"] = "completed"
    state["parts"]["part-0000.jsonl"][openai_batch.COUNTS_STATUS] = "completed"
    state["parts"]["part-0000.jsonl"]["output_file_id"] = "out-batch00"
    write_state(runner, requests, state)

    polled = runner.poll(requests)

    assert client.retrieved == ["batch01"]
    assert polled["parts"]["part-0000.jsonl"]["output_file_id"] == "out-batch00"
    assert polled["parts"]["part-0001.jsonl"]["status"] == "completed"


def _head_record(part: Dict[str, Any], batch: Any) -> None:
    """A record step that keeps an earlier status response's counts."""
    part["status"] = batch.status
    part["output_file_id"] = getattr(batch, "output_file_id", None)
    part["error_file_id"] = getattr(batch, "error_file_id", None)
    counts = getattr(batch, "request_counts", None)
    if counts is not None and hasattr(counts, "model_dump"):
        part["request_counts"] = counts.model_dump()


def _legacy_terminal_state(runner, requests, client, monkeypatch) -> None:
    """A state.json holding a stale in-progress count (2 of 3) on a completed shard."""
    statuses = iter(
        [
            SimpleNamespace(
                status="in_progress", output_file_id=None, error_file_id=None,
                request_counts=SimpleNamespace(
                    model_dump=lambda: {"total": 3, "completed": 2, "failed": 0}),
            ),
            SimpleNamespace(
                status="completed", output_file_id="out-batch00", error_file_id=None,
                request_counts=None,
            ),
        ]
    )
    original = client.batches.retrieve
    client.batches.retrieve = lambda batch_id: next(statuses)
    with monkeypatch.context() as patch:
        patch.setattr(OpenAIBatchRunner, "_record", staticmethod(_head_record))
        runner.poll(requests)
    client.batches.retrieve = original
    part = read_state(runner, requests)["parts"]["part-0000.jsonl"]
    assert (part["status"], part["request_counts"]["completed"]) == ("completed", 2)
    assert openai_batch.COUNTS_STATUS not in part


@pytest.mark.parametrize("counts", ["none", "model_dump"])
def test_a_terminal_shard_an_older_recorder_left_stale_counts_on_is_re_read_on_the_normal_path(
    tmp_path, confirmed, monkeypatch, counts
):
    client = FakeOpenAI(request_counts=counts)
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(3)
    directory = runner.workdir(requests)
    runner.submit(requests)
    _legacy_terminal_state(runner, requests, client, monkeypatch)
    whole = client._output_payload("batch00").decode("utf-8").splitlines()
    _seed_outputs(runner, requests, "part-0000.jsonl", whole[:2])
    match = {
        "none": "has an output file but no completed count",
        "model_dump": r"answers 2 request\(s\) without an error, but OpenAI counted 3",
    }[counts]

    with pytest.raises(BatchExecutionError, match=match):
        runner.run(requests)

    assert client.retrieved == ["batch00"]
    assert not (directory / "usage.json").exists()
    assert client.downloaded == [] and len(client.create_calls) == 1
    part = read_state(runner, requests)["parts"]["part-0000.jsonl"]
    assert part[openai_batch.COUNTS_STATUS] == "completed"
    assert openai_batch.completed_count(part) == {"none": None, "model_dump": 3}[counts]


def test_a_whole_output_under_an_older_recorder_s_state_is_read_after_one_re_read(
    tmp_path, confirmed, monkeypatch
):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(3)
    runner.submit(requests)
    _legacy_terminal_state(runner, requests, client, monkeypatch)

    outcomes, usage = runner.run(requests)
    assert usage.requests == 3 and all(outcome.text for outcome in outcomes.values())
    assert client.retrieved == ["batch00"]
    runner.run(requests)
    assert client.retrieved == ["batch00"]


def test_read_results_holds_no_output_to_counts_an_older_recorder_left(tmp_path, confirmed, monkeypatch):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(3)
    runner.submit(requests)
    _legacy_terminal_state(runner, requests, client, monkeypatch)
    whole = client._output_payload("batch00").decode("utf-8").splitlines()
    _seed_outputs(runner, requests, "part-0000.jsonl", whole[:2])
    with pytest.raises(BatchExecutionError, match="has an output file but no completed count"):
        runner.read_results(requests)
    assert not (runner.workdir(requests) / "usage.json").exists()


def test_download_writes_output_and_error_files_and_leaves_no_temp_behind(
    tmp_path, confirmed
):
    client = FakeOpenAI(emit_error_file=True)
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)
    runner.poll(requests)

    outputs = runner.download(requests)

    assert sorted(path.name for path in outputs.iterdir()) == [
        "part-0000.errors.jsonl",
        "part-0000.jsonl",
    ]
    assert client.downloaded == ["out-batch00", "err-batch00"]
    assert list(outputs.glob("*.tmp")) == []
    assert "answer::s0-p000000-c00" in (outputs / "part-0000.jsonl").read_text(
        encoding="utf-8"
    )
    assert "rate limited" in (outputs / "part-0000.errors.jsonl").read_text(
        encoding="utf-8"
    )


def test_two_downloads_at_once_never_share_a_temporary_file(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)
    runner.poll(requests)
    real_content = client.files.content
    other = build_runner(tmp_path, client)

    def racing(file_id):
        real_content(file_id)
        payload = client._output_payload(file_id.partition("-")[2])

        def write_to_file(path):
            with Path(path).open("wb") as handle:
                if len(client.downloaded) == 1:  # the first download has opened its file: the second runs whole
                    other.download(requests)
                handle.write(payload)

        return SimpleNamespace(write_to_file=write_to_file)

    client.files.content = racing
    outputs = runner.download(requests)
    assert client.downloaded == ["out-batch00", "out-batch00"]
    assert sorted(path.name for path in outputs.iterdir()) == ["part-0000.jsonl"]
    outcomes, usage = runner.read_results(requests)
    assert usage.requests == 2 and all(outcome.text for outcome in outcomes.values())


def test_a_download_that_fails_midway_leaves_no_file_behind(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)
    runner.poll(requests)

    def broken(path):
        Path(path).write_bytes(b'{"custom_id": "s0-p000000-c00"')
        raise ConnectionResetError("reset mid-download")

    client.files.content = lambda file_id: SimpleNamespace(write_to_file=broken)
    with pytest.raises(ConnectionResetError):
        runner.download(requests)
    assert list((runner.workdir(requests) / "outputs").iterdir()) == []


def test_download_skips_a_shard_with_no_file_ids(tmp_path, confirmed):
    """A failed batch has no output file; asking for file id None would raise."""
    client = FakeOpenAI(status_script=["failed"])
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)
    with pytest.raises(BatchExecutionError):
        runner.poll(requests)

    outputs = runner.download(requests)

    assert list(outputs.iterdir()) == []
    assert client.downloaded == []


def test_download_leaves_an_already_downloaded_file_alone(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(2)
    runner.submit(requests)
    runner.poll(requests)
    outputs = runner.download(requests)
    (outputs / "part-0000.jsonl").write_text("kept\n", encoding="utf-8")

    runner.download(requests)

    assert (outputs / "part-0000.jsonl").read_text(encoding="utf-8") == "kept\n"
    assert client.downloaded == ["out-batch00"]


def test_a_cached_output_file_cut_short_is_refused_not_billed_short(tmp_path, confirmed):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(3)
    runner.submit(requests)
    runner.poll(requests)
    whole = client._output_payload("batch00").decode("utf-8").splitlines()
    directory = runner.workdir(requests)
    outputs = _seed_outputs(runner, requests, "part-0000.jsonl", whole[:2])
    runner.download(requests)
    assert client.downloaded == []
    with pytest.raises(
        BatchExecutionError,
        match=r"answers 2 request\(s\) without an error, but OpenAI counted 3 "
        r"completed in shard part-0000.jsonl \(batch batch00\)",
    ):
        runner.read_results(requests)
    assert not (directory / "usage.json").exists()
    (outputs / "part-0000.jsonl").unlink()
    runner.download(requests)
    outcomes, usage = runner.read_results(requests)
    assert usage.requests == 3 and all(outcome.text for outcome in outcomes.values())


def test_a_shard_with_an_output_file_and_no_counts_is_refused_on_the_normal_path(
    tmp_path, confirmed
):
    client = FakeOpenAI(request_counts="none")
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(3)
    directory = runner.workdir(requests)
    runner.submit(requests)
    runner.poll(requests)
    whole = client._output_payload("batch00").decode("utf-8").splitlines()
    _seed_outputs(runner, requests, "part-0000.jsonl", whole[:2])
    for _ in range(2):
        with pytest.raises(
            BatchExecutionError,
            match=r"shard part-0000.jsonl \(batch batch00, completed\) has an output "
            r"file but no completed count.*nothing is read",
        ):
            runner.run(requests)
        assert not (directory / "usage.json").exists()
    assert client.downloaded == []
    state = read_state(runner, requests)
    state["parts"]["part-0000.jsonl"]["output_file_id"] = None
    write_state(runner, requests, state)
    with pytest.raises(BatchExecutionError, match="has an output file but no completed count"):
        runner.read_results(requests)
    assert not (directory / "usage.json").exists()


def test_counts_a_later_status_response_does_not_carry_are_dropped_not_kept(
    tmp_path, confirmed
):
    client = FakeOpenAI()
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    requests = make_requests(3)
    runner.submit(requests)
    statuses = iter(
        [
            SimpleNamespace(
                status="in_progress", output_file_id=None, error_file_id=None,
                request_counts=SimpleNamespace(
                    model_dump=lambda: {"total": 3, "completed": 2, "failed": 0}),
            ),
            SimpleNamespace(
                status="completed", output_file_id="out-batch00", error_file_id=None,
                request_counts=None,
            ),
        ]
    )
    client.batches.retrieve = lambda batch_id: next(statuses)
    runner.poll(requests)
    assert "request_counts" not in read_state(runner, requests)["parts"]["part-0000.jsonl"]
    whole = client._output_payload("batch00").decode("utf-8").splitlines()
    _seed_outputs(runner, requests, "part-0000.jsonl", whole[:2])
    with pytest.raises(BatchExecutionError, match="no completed count"):
        runner.read_results(requests)


@pytest.mark.parametrize(
    "counts, expected",
    [
        ({"total": 3, "completed": 2, "failed": 1}, 2),
        ({"total": 0, "completed": 0, "failed": 0}, 0),
        (None, None),
        ({}, None),
        ({"completed": None}, None),
        ({"completed": True}, None),
        ({"completed": -1}, None),
        ({"completed": "2"}, None),
        ("2", None),
    ],
)
def test_a_completed_count_is_a_non_negative_integer_or_none(counts, expected):
    part = {"status": "completed", openai_batch.COUNTS_STATUS: "completed"}
    if counts is not None:
        part["request_counts"] = counts
    assert openai_batch.completed_count(part) == expected


@pytest.mark.parametrize(
    "status, counts_status",
    [
        ("completed", None),
        ("completed", "in_progress"),  # counts from another response than the status's
        ("in_progress", "in_progress"),  # not ended: the count can still grow
        (None, None),
    ],
)
def test_only_the_counts_of_the_status_get_that_saw_a_shard_end_are_its_completed_count(
    status, counts_status
):
    part = {"status": status, "request_counts": {"total": 3, "completed": 2, "failed": 0}}
    if counts_status is not None:
        part[openai_batch.COUNTS_STATUS] = counts_status
    assert openai_batch.counts_final(part) is False
    assert openai_batch.completed_count(part) is None


def test_an_output_file_no_shard_names_is_refused(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    _seed_outputs(
        runner, requests, "part-0007.jsonl", [json.dumps(success_record("s0-p000000-c00", {}))]
    )
    with pytest.raises(BatchExecutionError, match=r"\['part-0007.jsonl'\] that no shard"):
        runner.read_results(requests)
    assert not (runner.workdir(requests) / "usage.json").exists()


def _seed_outputs(runner, requests, name, lines, *, completed=None) -> Path:
    runner.prepare(requests)
    directory = runner.workdir(requests)
    if completed is not None:
        state = read_state(runner, requests)
        state["parts"]["part-0000.jsonl"].update({
            "status": "completed", openai_batch.COUNTS_STATUS: "completed",
            "request_counts": {
                "total": len(requests), "completed": completed, "failed": len(requests) - completed}})
        openai_batch.write_json_atomic(directory / "state.json", state)
    outputs = directory / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


def test_read_results_maps_ids_to_text_and_totals_the_usage(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(2)
    directory = runner.workdir(requests)
    _seed_outputs(
        runner,
        requests,
        "part-0000.jsonl",
        [
            json.dumps(success_record("s0-p000000-c00", {})),
            "",
            "   ",
            json.dumps(success_record("s0-p000001-c00", {})),
        ],
        completed=2,
    )

    outcomes, usage = runner.read_results(requests)

    assert set(outcomes) == {"s0-p000000-c00", "s0-p000001-c00"}
    assert outcomes["s0-p000001-c00"].text == "answer::s0-p000001-c00"
    assert outcomes["s0-p000001-c00"].finish_reason == "completed"
    assert usage.to_dict() == {
        "input_tokens": 22,
        "cached_tokens": 8,
        "cache_write_tokens": 0,
        "output_tokens": 10,
        "reasoning_tokens": 4,
        "requests": 2,
    }
    assert json.loads((directory / "usage.json").read_text(encoding="utf-8")) == (
        usage.to_dict()
    )


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_read_results_splits_lines_on_newlines_only(tmp_path, separator):
    """A model's text may hold a raw Unicode line separator, which json.dumps(ensure_ascii=False) leaves raw."""
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    record = success_record("s0-p000000-c00", {})
    record["response"]["body"]["output"][0]["content"][0]["text"] = f"no{separator}really"
    _seed_outputs(runner, requests, "part-0000.jsonl", [json.dumps(record, ensure_ascii=False)], completed=1)

    outcomes, _ = runner.read_results(requests)

    assert outcomes["s0-p000000-c00"].text == f"no{separator}really"


def test_read_results_rejects_a_line_with_no_custom_id(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    _seed_outputs(
        runner, requests, "part-0000.jsonl", [json.dumps({"response": {"body": {}}})]
    )

    with pytest.raises(BatchExecutionError, match="has no custom_id"):
        runner.read_results(requests)


def test_read_results_rejects_a_duplicate_custom_id(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    line = json.dumps(success_record("s0-p000000-c00", {}))
    _seed_outputs(runner, requests, "part-0000.jsonl", [line, line])

    with pytest.raises(BatchExecutionError, match="duplicate custom_id"):
        runner.read_results(requests)


def test_read_results_rejects_an_id_no_request_carries(tmp_path):
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    _seed_outputs(runner, requests, "part-0000.jsonl", [json.dumps(success_record("s9-p000000-c00", {}))],
                  completed=1)

    with pytest.raises(BatchExecutionError, match="unknown custom id 's9-p000000-c00'"):
        runner.read_results(requests)
    assert not (runner.workdir(requests) / "usage.json").exists()


@pytest.mark.parametrize(
    "record,expected_error",
    [
        (
            {
                "custom_id": "s0-p000000-c00",
                "error": {"message": "model overloaded"},
                "response": None,
            },
            "model overloaded",
        ),
        (
            {
                "custom_id": "s0-p000000-c00",
                "response": {"status_code": 500, "body": {"usage": USAGE}},
            },
            "status 500",
        ),
    ],
    ids=["error-field", "status-500"],
)
def test_read_results_keeps_a_failed_request_and_does_not_bill_it(
    tmp_path, caplog, record, expected_error
):
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    _seed_outputs(runner, requests, "part-0000.jsonl", [json.dumps(record)], completed=0)

    with caplog.at_level(logging.WARNING, logger=openai_batch.logger.name):
        outcomes, usage = runner.read_results(requests)

    assert outcomes["s0-p000000-c00"].text is None
    assert expected_error in outcomes["s0-p000000-c00"].error
    assert usage.to_dict()["requests"] == 0
    assert usage.to_dict()["input_tokens"] == 0
    assert "s0-p000000-c00 failed" in caplog.text


def test_read_results_ignores_the_error_file_next_to_the_output_file(tmp_path):
    """.errors.jsonl is a provider report, not results -- parsing it duplicates ids."""
    runner = build_runner(tmp_path)
    requests = make_requests(1)
    _seed_outputs(
        runner,
        requests,
        "part-0000.jsonl",
        [json.dumps(success_record("s0-p000000-c00", {}))],
        completed=1,
    )
    _seed_outputs(
        runner,
        requests,
        "part-0000.errors.jsonl",
        [json.dumps({"custom_id": "s0-p000000-c00", "error": {"message": "boom"}})],
    )

    outcomes, usage = runner.read_results(requests)

    assert outcomes["s0-p000000-c00"].text == "answer::s0-p000000-c00"
    assert usage.requests == 1


def test_run_completes_a_whole_batch_end_to_end(tmp_path, confirmed):
    def with_summary(custom_id, body):
        record = success_record(custom_id, body)
        record["response"]["body"]["output"].insert(
            0,
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": f"thought::{custom_id}"}],
            },
        )
        return record

    client = FakeOpenAI(
        status_script=["in_progress", "completed"],
        responder=with_summary,
        emit_error_file=True,
    )
    runner = build_runner(
        tmp_path, client, max_shard_requests=2, sleep=lambda _: None
    )
    requests = make_requests(3)

    outcomes, usage = runner.run(requests)

    assert sorted(outcomes) == [
        "s0-p000000-c00",
        "s0-p000001-c00",
        "s0-p000002-c00",
    ]
    assert outcomes["s0-p000002-c00"].text == "answer::s0-p000002-c00"
    assert outcomes["s0-p000002-c00"].reasoning_summaries == (
        "thought::s0-p000002-c00",
    )
    assert usage.requests == 3
    assert usage.input_tokens == 33
    directory = runner.workdir(requests)
    assert (directory / "usage.json").is_file()
    assert (directory / "reasoning_summaries.jsonl").is_file()
    assert len(client.create_calls) == 2
    assert len(client.downloaded) == 4


def test_run_resumes_a_finished_batch_without_spending_again(tmp_path, confirmed):
    client = FakeOpenAI()
    requests = make_requests(2)
    first_outcomes, _ = build_runner(
        tmp_path, client, sleep=lambda _: None
    ).run(requests)

    second_outcomes, _ = build_runner(
        tmp_path, client, sleep=lambda _: None
    ).run(requests)

    assert second_outcomes == first_outcomes
    assert len(client.create_calls) == 1
    assert client.downloaded == ["out-batch00"]


def test_make_batch_generate_builds_both_callables_without_credentials(tmp_path):
    generate, generate_with_details = make_batch_generate(
        model_name="gpt-5.6-luna", root=tmp_path
    )

    assert callable(generate)
    assert callable(generate_with_details)


def test_batch_generate_returns_text_per_prompt_and_empty_for_failures(
    tmp_path, confirmed
):
    def flaky(custom_id, body):
        if custom_id.endswith("p000001-c00"):
            return None
        if custom_id.endswith("p000002-c00"):
            return {"custom_id": custom_id, "error": {"message": "overloaded"}}
        return success_record(custom_id, body)

    client = FakeOpenAI(responder=flaky)
    generate, _ = make_batch_generate(
        model_name="test-model",
        root=tmp_path,
        client_factory=lambda: client,
        poll_seconds=1.0,
    )

    results = generate([{"system_prompt": f"prompt-{i}"} for i in range(3)], 1.0, 64, 1, 3)

    assert results == [["answer::s3-p000000-c00"], [""], [""]]


def test_batch_generate_with_details_carries_a_real_finish_reason(
    tmp_path, confirmed
):
    def truncated(custom_id, body):
        return {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": '{"Rule 1": "y'}],
                        }
                    ],
                    "usage": USAGE,
                },
            },
        }

    client = FakeOpenAI(responder=truncated)
    _, generate_with_details = make_batch_generate(
        model_name="test-model", root=tmp_path, client_factory=lambda: client
    )

    details = generate_with_details([{"system_prompt": "prompt"}], 1.0, 64, 1, 0)

    assert details == [
        [
            {
                "raw_response": '{"Rule 1": "y',
                "completion": '{"Rule 1": "y',
                "finish_reason": "incomplete:max_output_tokens",
            }
        ]
    ]


def test_batch_generate_details_are_empty_strings_for_a_missing_result(
    tmp_path, confirmed
):
    client = FakeOpenAI(responder=lambda custom_id, body: None)
    _, generate_with_details = make_batch_generate(
        model_name="test-model", root=tmp_path, client_factory=lambda: client
    )

    details = generate_with_details([{"system_prompt": "prompt"}], 1.0, 64, 1, 0)

    assert details == [
        [{"raw_response": "", "completion": "", "finish_reason": None}]
    ]


def test_batch_results_stay_aligned_with_prompts_when_they_come_back_shuffled(
    tmp_path, confirmed
):
    def echo(custom_id, body):
        record = success_record(custom_id, body)
        record["response"]["body"]["output"][0]["content"][0]["text"] = (
            f"reply to {body['input'][0]['content']}"
        )
        return record

    client = FakeOpenAI(responder=echo, shuffle=True)
    generate, _ = make_batch_generate(
        model_name="test-model", root=tmp_path, client_factory=lambda: client
    )
    prompts = [{"system_prompt": f"prompt-{index}"} for index in range(4)]

    results = generate(prompts, 1.0, 64, 1, 5)

    assert results == [[f"reply to prompt-{index}"] for index in range(4)]


def test_batch_generate_returns_a_grid_of_prompts_by_candidates(tmp_path, confirmed):
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="test-model", root=tmp_path, client_factory=lambda: client
    )

    results = generate([{"system_prompt": "a"}, {"system_prompt": "b"}], 1.0, 64, 3, 0)

    assert [len(row) for row in results] == [3, 3]
    assert results[1] == [
        "answer::s0-p000001-c00",
        "answer::s0-p000001-c01",
        "answer::s0-p000001-c02",
    ]


def test_batch_generate_reports_one_token_usage_per_batch(tmp_path, confirmed):
    tracker = Tracker()
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="test-model",
        root=tmp_path,
        client_factory=lambda: client,
        tracker=tracker,
    )

    generate([{"system_prompt": "a"}, {"system_prompt": "b"}], 1.0, 64, 1, 0)

    assert tracker.added == [
        TokenUsage(prompt_tokens=22, completion_tokens=10, total_tokens=32)
    ]


@pytest.mark.parametrize(
    "send_max_output_tokens,expected",
    [(False, None), (True, 4096)],
    ids=["omitted", "included"],
)
def test_send_max_output_tokens_controls_the_request_body(
    tmp_path, confirmed, send_max_output_tokens, expected
):
    """Default False keeps the request byte-identical to the published campaign."""
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="gpt-5.6-luna",
        root=tmp_path,
        client_factory=lambda: client,
        reasoning_effort="medium",
        reasoning_summary="auto",
        send_max_output_tokens=send_max_output_tokens,
    )

    generate([{"system_prompt": "prompt"}], 1.0, 4096, 1, 0)

    body = json.loads(client.upload_calls[0]["bytes"].splitlines()[0])["body"]
    assert body.get("max_output_tokens") == expected
    assert body["model"] == "gpt-5.6-luna"
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_the_committed_config_prepares_its_model_id_and_settings_without_a_call(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from ruleforge.judges.backend import ModelBackend

    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path))
    config = Path(__file__).resolve().parents[1] / "configs" / "model" / "gpt-5.6-luna-batch.yaml"
    backend = ModelBackend(OmegaConf.load(config))
    with pytest.raises(BatchSpendNotConfirmed):
        backend.generate_once([{"system_prompt": "judge this"}], temperature=1.0,
                               request_block_id="pilot/gpt-5.6-luna/naive/draw-000/attempt-0/block-0000")
    [part] = tmp_path.glob("*/inputs/part-0000.jsonl")
    [line] = [json.loads(text) for text in part.read_text().splitlines()]
    assert (line["method"], line["url"]) == ("POST", RESPONSES_ENDPOINT)
    assert line["body"] == {"model": "gpt-5.6-luna", "input": [{"role": "developer", "content": "judge this"}],
                            "temperature": 1.0, "reasoning": {"effort": "medium", "summary": "auto"}}
    assert (backend.model_name, backend.seed_capability, backend.max_new_tokens) == ("gpt-5.6-luna", "unsupported", 65536)


def test_batch_generate_tags_every_batch_with_its_model(tmp_path, confirmed):
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="gpt-5.6-luna",
        root=tmp_path,
        client_factory=lambda: client,
        completion_window="24h",
    )

    generate([{"system_prompt": "prompt"}], 1.0, 64, 1, 0)

    assert client.create_calls[0]["metadata"] == {
        "model": "gpt-5.6-luna",
        "source": "smrc",
    }
    assert client.create_calls[0]["completion_window"] == "24h"


def test_a_failed_status_check_waits_instead_of_abandoning_the_batch(
    tmp_path, confirmed, caplog
):
    client = FakeOpenAI(status_script=["in_progress"])
    failures = {"left": 2}
    real_retrieve = client.batches.retrieve

    def flaky(batch_id):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("429 rate limited")
        return real_retrieve(batch_id)

    client.batches.retrieve = flaky
    requests = make_requests(2)
    runner = build_runner(tmp_path, client, sleep=lambda _: None)
    runner.submit(requests)

    with caplog.at_level(logging.WARNING):
        state = runner.poll(requests)

    assert [part["status"] for part in state["parts"].values()] == ["completed"]
    assert "still running" in caplog.text
    assert len(client.create_calls) == 1


def test_a_failed_status_check_still_honours_the_timeout(tmp_path, confirmed):
    client = FakeOpenAI()
    client.batches.retrieve = _raise_always
    requests = make_requests(1)
    ticks = iter([0.0, 1.0])
    runner = build_runner(
        tmp_path,
        client,
        timeout_seconds=5.0,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks, 99.0),
    )
    runner.submit(requests)

    with pytest.raises(BatchExecutionError, match="did not finish within"):
        runner.poll(requests)


def _raise_always(_batch_id):
    raise RuntimeError("endpoint down")


def test_a_retry_after_a_failed_batch_re_enters_it_instead_of_buying_another(
    tmp_path, confirmed
):
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="test-model",
        root=tmp_path / "batches",
        client_factory=lambda: client,
        poll_seconds=0.01,
        timeout_seconds=0.05,
    )
    prompts = [{"system_prompt": "a", "user_prompt": ""}]

    client.batches.retrieve = _raise_always
    with pytest.raises(Exception):
        generate(prompts, 1.0, 64, 1, 111)
    assert len(client.create_calls) == 1

    # Attempt 2 comes back with a different seed, as the retry loop would.
    client.batches.retrieve = lambda batch_id: SimpleNamespace(
        id=batch_id,
        status="completed",
        output_file_id=f"out-{batch_id}",
        error_file_id=None,
        request_counts=SimpleNamespace(model_dump=lambda: client._counts(batch_id)),
    )
    answers = generate(prompts, 1.0, 64, 1, 222)
    assert len(client.create_calls) == 1, "a second batch was bought"
    assert answers == [["answer::s111-p000000-c00"]]


def test_a_batch_that_returned_answers_does_re_ask_on_the_next_retry(
    tmp_path, confirmed
):
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="test-model",
        root=tmp_path / "batches",
        client_factory=lambda: client,
        poll_seconds=0.01,
    )
    prompts = [{"system_prompt": "a", "user_prompt": ""}]

    generate(prompts, 1.0, 64, 1, 111)
    generate(prompts, 1.0, 64, 1, 222)
    assert len(client.create_calls) == 2


def test_the_batch_adapter_warns_that_the_responses_endpoint_drops_the_seed(
    tmp_path, confirmed, caplog
):
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="test-model",
        root=tmp_path / "batches",
        client_factory=lambda: client,
        poll_seconds=0.01,
    )
    with caplog.at_level(logging.WARNING):
        generate([{"system_prompt": "a", "user_prompt": ""}], 1.0, 64, 1, 16)
        generate([{"system_prompt": "b", "user_prompt": ""}], 1.0, 64, 1, 16)

    warnings = [r for r in caplog.records if "no seed parameter" in r.getMessage()]
    assert len(warnings) == 1
    assert "16" in warnings[0].getMessage()


def test_no_seed_warning_when_the_batch_targets_chat_completions(
    tmp_path, confirmed, caplog
):
    client = FakeOpenAI()
    generate, _ = make_batch_generate(
        model_name="test-model",
        root=tmp_path / "batches",
        endpoint=CHAT_COMPLETIONS_ENDPOINT,
        client_factory=lambda: client,
        poll_seconds=0.01,
    )
    with caplog.at_level(logging.WARNING):
        generate([{"system_prompt": "a", "user_prompt": ""}], 1.0, 64, 1, 16)

    assert "no seed parameter" not in caplog.text
    submitted = json.loads(next(iter(client.uploads.values())).decode().splitlines()[0])
    assert submitted["body"]["seed"] == 16


def test_a_permanently_failing_status_check_gives_up_instead_of_waiting_forever(
    tmp_path, confirmed
):
    client = FakeOpenAI()
    client.batches.retrieve = _raise_always
    requests = make_requests(1)
    runner = build_runner(
        tmp_path, client, sleep=lambda _: None, max_poll_errors=3, timeout_seconds=None
    )
    runner.submit(requests)

    with pytest.raises(BatchExecutionError, match="status checks in a row failed"):
        runner.poll(requests)


def test_a_recovered_status_check_resets_the_failure_count(tmp_path, confirmed):
    client = FakeOpenAI(status_script=["in_progress"] * 6)
    calls = {"n": 0}
    real = client.batches.retrieve

    def alternating(batch_id):
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("429")
        return real(batch_id)

    client.batches.retrieve = alternating
    requests = make_requests(1)
    runner = build_runner(
        tmp_path, client, sleep=lambda _: None, max_poll_errors=2
    )
    runner.submit(requests)
    state = runner.poll(requests)
    assert [part["status"] for part in state["parts"].values()] == ["completed"]


def test_a_nonpositive_poll_error_bound_is_refused(tmp_path):
    with pytest.raises(ValueError, match="max_poll_errors must be positive"):
        build_runner(tmp_path, max_poll_errors=0)
