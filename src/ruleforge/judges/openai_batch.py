from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ruleforge.core.record_names import BATCH_METADATA_SOURCE
from ruleforge.judges.batch_api import (
    blocking_generators,
    block_details_from_outcomes,
    collect_block_details,
    estimate_enqueued_tokens,
    CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_POLL_SECONDS,
    RESPONSES_ENDPOINT,
    TERMINAL_STATUSES,
    TEXT_EXTRACTORS,
    BatchExecutionError,
    BatchOutcome,
    BatchRequest,
    BatchUsage,
    ResumeGuard,
    claim_for_creation,
    build_batch_requests,
    create_was_refused,
    fingerprint,
    log_batch_completion,
    outcome_from_record,
    private_temporary,
    request_line,
    require_spend_confirmation,
    reshape_outcomes,
    responses_summaries_requested,
    sha256_bytes,
    write_bytes_atomic,
    write_json_atomic,
    write_json_once,
    write_reasoning_summaries,
    work_directory,
    track_usage,
    utc_now,
)
from ruleforge.judges.models import OPENAI_REDIRECT_ENVS, _warn_seed_ignored_once, refuse_ambient_sdk_envs

logger = logging.getLogger(__name__)

# Bumped when the on-disk state shape changes; an older directory is refused.
STATE_VERSION = 1

# Status checks in a row that may fail before the wait is abandoned.
DEFAULT_MAX_POLL_ERRORS = 20

# Below the API's 200 MB / 50,000-request per-file limits.
DEFAULT_MAX_SHARD_BYTES = 180_000_000
DEFAULT_MAX_SHARD_REQUESTS = 45_000


def write_shards(
    requests: Sequence[BatchRequest],
    directory: Path,
    *,
    endpoint: str,
    max_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_requests: int = DEFAULT_MAX_SHARD_REQUESTS,
) -> List[Dict[str, Any]]:
    """Split requests into batch input files; return one descriptor per shard (path, request count, bytes, SHA-256)."""
    if not requests:
        raise ValueError("cannot prepare a batch with no requests")
    directory.mkdir(parents=True, exist_ok=True)

    shards: List[Dict[str, Any]] = []
    lines: List[bytes] = []
    size = 0

    def close_shard() -> None:
        payload = b"".join(lines)
        name = f"part-{len(shards):04d}.jsonl"
        write_bytes_atomic(directory / name, payload)
        shards.append(
            {
                "path": name,
                "requests": len(lines),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        lines.clear()

    for request in requests:
        line = request_line(request, endpoint)
        if len(line) > max_bytes:
            raise ValueError(
                f"request {request.custom_id!r} is {len(line)} bytes, over the "
                f"{max_bytes}-byte shard limit; it can never be submitted"
            )
        if lines and (size + len(line) > max_bytes or len(lines) >= max_requests):
            close_shard()
            size = 0
        lines.append(line)
        size += len(line)
    close_shard()
    return shards


# Recorded with every status GET: counts beside a status it does not name may be an in-progress count.
COUNTS_STATUS = "request_counts_status"


def counts_final(part: Mapping[str, Any]) -> bool:
    """Return whether a shard's record is a terminal status GET's, so its ``request_counts`` are final."""
    status = part.get("status")
    return status in TERMINAL_STATUSES and part.get(COUNTS_STATUS) == status


def completed_count(part: Mapping[str, Any]) -> Optional[int]:
    """Return the final ``request_counts.completed`` of a shard (the requests OpenAI bills), or None."""
    if not counts_final(part):
        return None
    counts = part.get("request_counts")
    value = counts.get("completed") if isinstance(counts, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def default_client() -> Any:
    """Return an OpenAI client, refused without a key or while ``OPENAI_BASE_URL`` is set."""
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    refuse_ambient_sdk_envs(OPENAI_REDIRECT_ENVS, sdk="openai", server="api.openai.com")
    from openai import OpenAI

    return OpenAI(api_key=api_key)


@dataclass
class OpenAIBatchRunner:
    """Prepare, submit, poll and collect one batch; ``client_factory`` is called only when a networked call is
    needed."""

    root: Path
    endpoint: str = RESPONSES_ENDPOINT
    client_factory: Callable[[], Any] = default_client
    poll_seconds: float = DEFAULT_POLL_SECONDS
    timeout_seconds: Optional[float] = None
    max_poll_errors: int = DEFAULT_MAX_POLL_ERRORS
    completion_window: str = DEFAULT_COMPLETION_WINDOW
    metadata: Optional[Dict[str, str]] = None
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES
    max_shard_requests: int = DEFAULT_MAX_SHARD_REQUESTS
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.endpoint not in TEXT_EXTRACTORS:
            raise ValueError(
                f"unsupported batch endpoint {self.endpoint!r}; "
                f"expected one of {sorted(TEXT_EXTRACTORS)}"
            )
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_poll_errors <= 0:
            raise ValueError("max_poll_errors must be positive")
        self.root = Path(self.root)
        self._client: Any | None = None

    def workdir(self, requests: Sequence[BatchRequest]) -> Path:
        """Return the content-addressed directory for this request set."""
        return work_directory(self.root, fingerprint(requests, self.endpoint))

    def _client_or_init(self) -> Any:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def prepare(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Write the request files and the state record, or return the state already there; makes no API call."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        if state_path.is_file():
            return self._load_state(state_path)

        shards = write_shards(
            requests,
            directory / "inputs",
            endpoint=self.endpoint,
            max_bytes=self.max_shard_bytes,
            max_requests=self.max_shard_requests,
        )
        state = {
            "state_version": STATE_VERSION,
            "endpoint": self.endpoint,
            "completion_window": self.completion_window,
            "requests": len(requests),
            "shards": shards,
            "parts": {shard["path"]: {} for shard in shards},
        }
        # Another process may have prepared and bought this request set since the check above.
        if not write_json_once(state_path, state):
            return self._load_state(state_path)
        logger.info(
            "Prepared %d batch request(s) in %d shard(s) at %s",
            len(requests),
            len(shards),
            directory,
        )
        return state

    def _load_state(self, state_path: Path) -> Dict[str, Any]:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        version = state.get("state_version")
        if version != STATE_VERSION:
            raise BatchExecutionError(
                f"{state_path} was written by state version {version!r}; this "
                f"build writes {STATE_VERSION}. Refusing to resume a batch whose "
                "shard accounting may differ -- inspect it by hand."
            )
        if state.get("endpoint") != self.endpoint:
            raise BatchExecutionError(
                f"{state_path} was prepared for endpoint {state.get('endpoint')!r}, "
                f"not {self.endpoint!r}"
            )
        return state

    def submit(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Upload and create a batch for every shard not already submitted; with none missing, makes no API call."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        state = self.prepare(requests)
        if all(part.get("batch_id") for part in state["parts"].values()):
            return state
        require_spend_confirmation()
        client = self._client_or_init()

        with claim_for_creation(directory):
            state = self._load_state(state_path)
            self._create_missing_batches(state, state_path, directory, client)
        return state

    def _create_missing_batches(self, state, state_path, directory, client) -> None:
        for shard in state["shards"]:
            relative = shard["path"]
            part = state["parts"].setdefault(relative, {})
            if part.get("batch_id"):
                continue
            if part.get("create_started_at"):
                raise BatchExecutionError(self._unrecorded_create(relative, part, state_path))
            path = directory / "inputs" / relative
            if sha256_bytes(path.read_bytes()) != shard["sha256"]:
                raise BatchExecutionError(f"prepared shard changed on disk: {path}")
            if not part.get("file_id"):
                with path.open("rb") as handle:
                    uploaded = client.files.create(file=handle, purpose="batch")
                part["file_id"] = uploaded.id
                # Persisted before the batch is created: a forgotten batch is one paid for twice.
                write_json_atomic(state_path, state)
            # Persisted before the one create call: a create is not idempotent, so a failure after it keeps this mark.
            part["create_started_at"] = utc_now().isoformat()
            write_json_atomic(state_path, state)
            try:
                batch = client.with_options(max_retries=0).batches.create(
                    input_file_id=part["file_id"],
                    endpoint=self.endpoint,
                    completion_window=self.completion_window,
                    metadata=dict(self.metadata) if self.metadata else None,
                )
            except Exception as exc:
                if not create_was_refused(exc):
                    raise BatchExecutionError(
                        f"batches.create failed ({type(exc).__name__}: {exc}) in a way that does not say whether "
                        f"the batch was made, so it is not retried. "
                        + self._unrecorded_create(relative, part, state_path)) from exc
                del part["create_started_at"]
                write_json_atomic(state_path, state)
                raise
            part["batch_id"] = batch.id
            part["status"] = batch.status
            write_json_atomic(state_path, state)
            logger.info("Submitted %s as batch %s", relative, batch.id)

    @staticmethod
    def _unrecorded_create(name: str, part: Mapping[str, Any], state_path: Path) -> str:
        return (
            f"{name}: a batches.create started at {part.get('create_started_at')} and recorded no batch, so it may "
            f"have made one, and a second create would buy it twice. List the account's batches "
            f"(client.batches.list()): one with input_file_id {part.get('file_id')} is it -- write its id as "
            f"parts[{name!r}].batch_id in {state_path} and rerun to wait on it; if there is none, delete "
            f"parts[{name!r}].create_started_at there and rerun to create it."
        )

    def _sweep(
        self, state: Dict[str, Any], state_path: Path, client: Any
    ) -> tuple[List[str], bool]:
        pending: List[str] = []
        errored = False
        for relative, part in state["parts"].items():
            batch_id = part.get("batch_id")
            if not batch_id:
                raise BatchExecutionError(f"shard {relative} was never submitted")
            if counts_final(part):
                continue
            try:
                batch = client.batches.retrieve(batch_id)
            except Exception as exc:
                # A failed status check says nothing about the running, paid batch; a retry here would buy another.
                logger.warning(
                    "Status check for %s failed (%s); the batch is still "
                    "running, so waiting rather than giving up",
                    batch_id,
                    exc,
                )
                errored = True
                pending.append(f"{relative}=unknown")
                continue
            self._record(part, batch)
            if batch.status not in TERMINAL_STATUSES:
                pending.append(f"{relative}={batch.status}")
        write_json_atomic(state_path, state)
        return pending, errored

    @staticmethod
    def _record(part: Dict[str, Any], batch: Any) -> None:
        part["status"] = batch.status
        part[COUNTS_STATUS] = batch.status
        part["output_file_id"] = getattr(batch, "output_file_id", None)
        part["error_file_id"] = getattr(batch, "error_file_id", None)
        counts = getattr(batch, "request_counts", None)
        if counts is not None and hasattr(counts, "model_dump"):
            part["request_counts"] = counts.model_dump()
        else:
            part.pop("request_counts", None)

    def refresh(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Re-read every shard's status once into ``state.json`` and return the state; a failed GET raises."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        state = self._load_state(state_path)
        client = self._client_or_init()
        for relative, part in state["parts"].items():
            batch_id = part.get("batch_id")
            if not batch_id:
                raise BatchExecutionError(f"shard {relative} was never submitted")
            self._record(part, client.batches.retrieve(batch_id))
        write_json_atomic(state_path, state)
        return state

    def _finished_or_raise(self, state: Dict[str, Any], directory: Path) -> None:
        failed = {
            relative: part.get("status")
            for relative, part in state["parts"].items()
            if part.get("status") != "completed"
        }
        if failed:
            # Returning empty answers would make the retry layer buy the whole batch again under a fresh seed.
            raise BatchExecutionError(
                f"batch shard(s) ended in a non-completed terminal "
                f"state: {failed}. Nothing is retried automatically, "
                f"because a retry would re-buy every request. Any "
                f"partial output is downloadable from {directory} "
                f"(the batch ids are in state.json); decide there "
                f"whether to collect it or resubmit."
            )

    def poll(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Block until every submitted shard reaches a terminal status, which may not be success."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        state = self._load_state(state_path)
        client = self._client_or_init()
        deadline = (
            self.monotonic() + self.timeout_seconds
            if self.timeout_seconds is not None
            else None
        )

        consecutive_errors = 0
        while True:
            pending, errored = self._sweep(state, state_path, client)
            if not pending:
                self._finished_or_raise(state, directory)
                return state
            consecutive_errors = consecutive_errors + 1 if errored else 0
            if consecutive_errors >= self.max_poll_errors:
                raise BatchExecutionError(
                    f"{consecutive_errors} status checks in a row failed; the "
                    f"batch may still be running, and its state is in "
                    f"{directory}. This is a permanent-looking failure -- a "
                    f"rotated key, a deleted batch, the wrong org -- not the "
                    f"transient blip the retry tolerates. Fix it and rerun to "
                    f"resume; the work is already paid for."
                )
            if deadline is not None and self.monotonic() >= deadline:
                raise BatchExecutionError(
                    f"batch did not finish within {self.timeout_seconds}s; it is "
                    f"still running and its state is in {directory}. Rerun to "
                    "resume waiting -- the work is already paid for."
                )
            logger.info(
                "Waiting on %d batch shard(s): %s", len(pending), ", ".join(pending)
            )
            self.sleep(self.poll_seconds)

    def check(self, requests: Sequence[BatchRequest]) -> bool:
        """Run one non-blocking status sweep and return whether every shard has completed, raising on a failed terminal
        state."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        state = self._load_state(state_path)
        client = self._client_or_init()
        pending, _errored = self._sweep(state, state_path, client)
        if pending:
            return False
        self._finished_or_raise(state, directory)
        return True

    def download(self, requests: Sequence[BatchRequest]) -> Path:
        """Fetch output and error files for every terminal shard."""
        directory = self.workdir(requests)
        state = self._load_state(directory / "state.json")
        client = self._client_or_init()
        outputs = directory / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)

        for relative, part in state["parts"].items():
            stem = Path(relative).stem
            for key, suffix in (
                ("output_file_id", ".jsonl"),
                ("error_file_id", ".errors.jsonl"),
            ):
                file_id = part.get(key)
                if not file_id:
                    continue
                destination = outputs / f"{stem}{suffix}"
                if destination.exists():
                    continue
                temporary = private_temporary(destination)
                try:
                    client.files.content(file_id).write_to_file(temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                logger.info("Downloaded %s -> %s", file_id, destination)
        return outputs

    def read_results(
        self, requests: Sequence[BatchRequest]
    ) -> tuple[Dict[str, BatchOutcome], BatchUsage]:
        """Return custom_id -> outcome (``text=None`` without a usable answer) and the batch's token usage; raises on an
        id no request carries, or unless each shard answers exactly the requests OpenAI counted completed."""
        directory = self.workdir(requests)
        outputs = directory / "outputs"
        parts = self._load_state(directory / "state.json")["parts"]
        known = {request.custom_id for request in requests}
        outcomes: Dict[str, BatchOutcome] = {}
        usage = BatchUsage()
        billed: Dict[str, int] = {}

        for path in sorted(outputs.glob("part-*.jsonl")):
            if path.name.endswith(".errors.jsonl"):
                continue
            billed[path.name] = 0
            # Split on newlines only: a model's text may hold a raw U+2028 or U+0085, which str.splitlines() splits on.
            for line in path.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                record = json.loads(line)
                custom_id = record.get("custom_id")
                if not isinstance(custom_id, str):
                    raise BatchExecutionError(
                        f"batch output line in {path} has no custom_id"
                    )
                if custom_id not in known:
                    raise BatchExecutionError(f"batch output line in {path} answers unknown custom id {custom_id!r}")
                if custom_id in outcomes:
                    raise BatchExecutionError(
                        f"duplicate custom_id {custom_id!r} in {path}"
                    )
                outcome, body = outcome_from_record(record, self.endpoint)
                if body is None:
                    logger.warning(
                        "Batch request %s failed: %s", custom_id, outcome.error
                    )
                else:
                    usage.add_body(body)
                    billed[path.name] += 1
                outcomes[custom_id] = outcome

        self._check_completed(directory, parts, billed)
        write_json_atomic(directory / "usage.json", usage.to_dict())
        write_reasoning_summaries(directory, outcomes)
        return outcomes, usage

    @staticmethod
    def _check_completed(
        directory: Path,
        parts: Mapping[str, Mapping[str, Any]],
        billed: Mapping[str, int],
    ) -> None:
        files = {f"{Path(name).stem}.jsonl": name for name in parts}
        stray = sorted(set(billed) - set(files))
        if stray:
            raise BatchExecutionError(
                f"{directory / 'outputs'} holds output file(s) {stray} that no "
                f"shard of its state.json names"
            )
        for file_name, name in sorted(files.items()):
            completed = completed_count(parts[name])
            if completed is None:
                if parts[name].get("output_file_id") or file_name in billed:
                    raise BatchExecutionError(
                        f"shard {name} (batch {parts[name].get('batch_id')}, "
                        f"{parts[name].get('status')}) has an output file but no "
                        f"completed count recorded from the OpenAI status response "
                        f"that saw it end, so {directory / 'outputs' / file_name} "
                        f"cannot be held to the requests OpenAI billed for it; "
                        f"nothing is read. Re-read its status (OpenAIBatchRunner."
                        f"refresh, read-only, or a rerun, whose poll re-reads it) and "
                        f"run again once it carries request_counts"
                    )
                continue
            found = billed.get(file_name, 0)
            if found != completed:
                raise BatchExecutionError(
                    f"{directory / 'outputs' / file_name} answers {found} "
                    f"request(s) without an error, but OpenAI counted {completed} "
                    f"completed in shard {name} (batch "
                    f"{parts[name].get('batch_id')}) when it ended: the file is "
                    f"not the shard's whole output, and reading it would bill "
                    f"less than was billed. Delete it and rerun to download it "
                    f"again"
                )

    def run(
        self, requests: Sequence[BatchRequest]
    ) -> tuple[Dict[str, BatchOutcome], BatchUsage]:
        """Prepare, submit, wait, download and collect, in one blocking call."""
        self.submit(requests)
        self.poll(requests)
        self.download(requests)
        return self.read_results(requests)


def _runner_settings(
    *,
    model_name: str,
    root: Path,
    endpoint: str,
    poll_seconds: float,
    timeout_seconds: Optional[float],
    completion_window: str,
    client_factory: Optional[Callable[[], Any]],
) -> Dict[str, Any]:
    runner_kwargs: Dict[str, Any] = dict(
        root=Path(root),
        endpoint=endpoint,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        completion_window=completion_window,
        metadata={"model": model_name, "source": BATCH_METADATA_SOURCE},
    )
    if client_factory is not None:
        runner_kwargs["client_factory"] = client_factory
    return runner_kwargs


def make_block_generate(
    *,
    model_name: str,
    root: Path,
    endpoint: str = RESPONSES_ENDPOINT,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    send_max_output_tokens: bool = False,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    client_factory: Optional[Callable[[], Any]] = None,
    tracker: Any | None = None,
) -> Callable[
    [List[Dict[str, str]], float, int, Optional[Sequence[Optional[int]]], str],
    List[Dict[str, Any]],
]:
    """Return ``_generate_block(prompts, temperature, max_new_tokens, seeds, request_block_id)``, one attempt per call,
    no retry or ``ResumeGuard``: the block id gives the namespace, so a rerun re-addresses the paid batch."""
    runner_kwargs = _runner_settings(
        model_name=model_name,
        root=root,
        endpoint=endpoint,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        completion_window=completion_window,
        client_factory=client_factory,
    )
    summarized = responses_summaries_requested(endpoint, reasoning_effort, reasoning_summary)
    _seed_warning: Dict[str, bool] = {}

    def _generate_block(
        prompts: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        seeds: Optional[Sequence[Optional[int]]],
        request_block_id: str,
    ) -> List[Dict[str, Any]]:
        namespace, requests = prepare_block_requests(
            prompts,
            temperature,
            max_new_tokens,
            seeds,
            request_block_id,
            endpoint=endpoint,
            model=model_name,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            send_max_output_tokens=send_max_output_tokens,
            seed_warning=_seed_warning,
        )
        return collect_block_details(
            OpenAIBatchRunner(**runner_kwargs),
            requests,
            namespace=namespace,
            provider="OpenAI",
            tracker=tracker,
            summaries_requested=summarized,
        )

    return _generate_block


def prepare_block_requests(
    prompts: List[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    seeds: Optional[Sequence[Optional[int]]],
    request_block_id: str,
    *,
    endpoint: str,
    model: str,
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
    send_max_output_tokens: bool,
    seed_warning: Dict[str, bool],
) -> tuple[str, List[BatchRequest]]:
    """Warn once about seeds the responses endpoint drops, then return the block's request set."""
    if endpoint != CHAT_COMPLETIONS_ENDPOINT and seeds is not None:
        _warn_seed_ignored_once(
            seed_warning,
            "The batch responses endpoint",
            next((s for s in seeds if s is not None), None),
        )
    return build_block_requests(
        prompts,
        temperature,
        max_new_tokens,
        seeds,
        request_block_id,
        endpoint=endpoint,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        send_max_output_tokens=send_max_output_tokens,
    )


def build_block_requests(
    prompts: List[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    seeds: Optional[Sequence[Optional[int]]],
    request_block_id: str,
    *,
    endpoint: str,
    model: str,
    reasoning_effort: Optional[str],
    reasoning_summary: Optional[str],
    send_max_output_tokens: bool,
) -> tuple[str, List[BatchRequest]]:
    """Return the (namespace, requests) of one block, addressed by the block id."""
    from ruleforge.core.digests import request_namespace

    namespace = request_namespace(request_block_id)
    requests = build_batch_requests(
        prompts,
        endpoint=endpoint,
        model=model,
        temperature=temperature,
        num_responses=1,
        seed=None,
        seeds=seeds if endpoint == CHAT_COMPLETIONS_ENDPOINT else None,
        namespace=namespace,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        max_output_tokens=max_new_tokens if send_max_output_tokens else None,
    )
    return namespace, requests


def make_windowed_block_ops(
    *,
    model_name: str,
    root: Path,
    endpoint: str = RESPONSES_ENDPOINT,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    send_max_output_tokens: bool = False,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    client_factory: Optional[Callable[[], Any]] = None,
    tracker: Any | None = None,
) -> tuple[Callable[..., int], Callable[..., None], Callable[..., Optional[List[Dict[str, Any]]]]]:
    """Return the ``(estimate, submit, poll)`` a windowed scheduler drives ``openai_batch`` with."""
    runner_kwargs = _runner_settings(
        model_name=model_name,
        root=root,
        endpoint=endpoint,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        completion_window=completion_window,
        client_factory=client_factory,
    )
    summarized = responses_summaries_requested(endpoint, reasoning_effort, reasoning_summary)
    _seed_warning: Dict[str, bool] = {}

    def _prepare(prompts, temperature, max_new_tokens, seeds, request_block_id):
        return prepare_block_requests(
            prompts,
            temperature,
            max_new_tokens,
            seeds,
            request_block_id,
            endpoint=endpoint,
            model=model_name,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            send_max_output_tokens=send_max_output_tokens,
            seed_warning=_seed_warning,
        )

    def estimate(prompts, *, temperature, max_new_tokens, seeds, request_block_id):
        _, requests = _prepare(prompts, temperature, max_new_tokens, seeds, request_block_id)
        return estimate_enqueued_tokens(requests, endpoint=endpoint)

    def submit(prompts, *, temperature, max_new_tokens, seeds, request_block_id):
        _, requests = _prepare(prompts, temperature, max_new_tokens, seeds, request_block_id)
        OpenAIBatchRunner(**runner_kwargs).submit(requests)

    def poll(prompts, *, temperature, max_new_tokens, seeds, request_block_id):
        namespace, requests = _prepare(prompts, temperature, max_new_tokens, seeds, request_block_id)
        runner = OpenAIBatchRunner(**runner_kwargs)
        if not runner.check(requests):
            return None
        runner.download(requests)
        outcomes, usage = runner.read_results(requests)
        return block_details_from_outcomes(
            outcomes,
            usage,
            requests,
            namespace=namespace,
            provider="OpenAI",
            tracker=tracker,
            workdir=str(runner.workdir(requests)),
            summaries_requested=summarized,
        )

    return estimate, submit, poll


def make_batch_generate(
    *,
    model_name: str,
    root: Path,
    endpoint: str = RESPONSES_ENDPOINT,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    send_max_output_tokens: bool = False,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    client_factory: Optional[Callable[[], Any]] = None,
    tracker: Any | None = None,
) -> tuple[
    Callable[[List[Dict[str, str]], float, int, int, Optional[int]], List[List[str]]],
    Callable[
        [List[Dict[str, str]], float, int, int, Optional[int]],
        List[List[Dict[str, Any]]],
    ],
]:
    """Return ``(_batch_generate, _batch_generate_with_details)`` for ``model_type: openai_batch``. One call blocks
    until a whole batch clears. ``send_max_output_tokens`` defaults to False, so the model's own ceiling applies."""
    runner_kwargs = _runner_settings(
        model_name=model_name,
        root=root,
        endpoint=endpoint,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        completion_window=completion_window,
        client_factory=client_factory,
    )

    summarized = responses_summaries_requested(endpoint, reasoning_effort, reasoning_summary)
    resume_guard = ResumeGuard()
    _seed_warning: Dict[str, bool] = {}

    def _run(
        prompts: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        num_responses: int,
        seed: Optional[int],
    ) -> List[List[BatchOutcome]]:
        seed = resume_guard.seed_for(seed)
        resume_guard.arm(seed)
        if endpoint == RESPONSES_ENDPOINT:
            # The responses endpoint has no seed parameter: the configured seed reaches only the custom ids.
            _warn_seed_ignored_once(
                _seed_warning, "The batch responses endpoint", seed
            )
        requests = build_batch_requests(
            prompts,
            endpoint=endpoint,
            model=model_name,
            temperature=temperature,
            num_responses=num_responses,
            seed=seed,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_output_tokens=max_new_tokens if send_max_output_tokens else None,
        )
        outcomes, usage = OpenAIBatchRunner(**runner_kwargs).run(requests)
        track_usage(tracker, usage)
        resume_guard.clear()
        grid = reshape_outcomes(
            outcomes,
            num_prompts=len(prompts),
            num_responses=num_responses,
            seed=seed,
        )
        log_batch_completion(grid, "OpenAI")
        return grid

    return blocking_generators(_run, summaries_requested=summarized)
