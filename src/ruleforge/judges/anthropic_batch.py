from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ruleforge.judges.backend import ANOTHER_SERVER
from ruleforge.judges.batch_api import (
    DEFAULT_POLL_SECONDS,
    BatchExecutionError,
    BatchOutcome,
    BatchRequest,
    BatchUsage,
    ResumeGuard,
    batch_custom_id,
    blocking_generators,
    claim_for_creation,
    collect_block_details,
    create_was_refused,
    log_batch_completion,
    private_temporary,
    require_spend_confirmation,
    reshape_outcomes,
    sha256_bytes,
    write_bytes_atomic,
    write_json_atomic,
    write_json_once,
    write_reasoning_summaries,
    work_directory,
    track_usage,
    utc_now,
)
from ruleforge.judges.models import EMPTY_USER_TURN, _warn_seed_ignored_once

logger = logging.getLogger(__name__)

# Bumped when the on-disk state shape changes; an older directory is refused.
STATE_VERSION = 1

# Hashed into every content address, so work directories of different transports cannot collide in a shared root.
ANTHROPIC_BATCH_ENDPOINT = "/v1/messages/batches"

API_KEY_ENV = "ANTHROPIC_API_KEY"
# Variables the SDK would read on its own (another host, a second credential), refused while set.
REFUSED_SDK_ENVS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN")

# Status checks in a row that may fail before the wait is abandoned.
DEFAULT_MAX_POLL_ERRORS = 20

# The API allows 100,000 requests and 256 MB per batch; these keep a margin.
DEFAULT_MAX_SHARD_REQUESTS = 90_000
DEFAULT_MAX_SHARD_BYTES = 200_000_000

CUSTOM_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

# ``output_config.effort`` values.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# Sent on every request and not configurable, so no condition identity records it.
THINKING = {"type": "adaptive"}

# ``thinking.display`` values: ``summarized`` returns a readable summary, ``omitted`` (the default) leaves blocks empty.
DISPLAY_SUMMARIZED, DISPLAY_OMITTED = "summarized", "omitted"
THINKING_DISPLAYS = (DISPLAY_SUMMARIZED, DISPLAY_OMITTED)

# None may be sent, so requests sample at the provider's default.
PROVIDER_TEMPERATURE = 1.0

# ``processing_status`` once every request has a result.
PROCESSING_ENDED = "ended"
# A part ended with every request answered or errored, or with some canceled or expired.
PART_COMPLETED, PART_INCOMPLETE = "completed", "ended_incomplete"
TERMINAL_PART_STATES = frozenset({PART_COMPLETED, PART_INCOMPLETE})
UNFINISHED_RESULTS = ("canceled", "expired")

RESULT_SUCCEEDED, RESULT_ERRORED = "succeeded", "errored"

# The stop reason of a declined request; the finish reasons below are prefixed so they never read as the provider's.
REFUSAL = "refusal"
REFUSAL_UNSPECIFIED = "unspecified"
NO_TEXT = "no_text"


def default_client() -> Any:
    """Return an ``anthropic.Anthropic`` client keyed from ``ANTHROPIC_API_KEY``, refused while ``REFUSED_SDK_ENVS`` are
    set."""
    api_key = (os.environ.get(API_KEY_ENV) or "").strip()
    if not api_key:
        raise ValueError(f"{API_KEY_ENV} environment variable is required")
    ambient = [name for name in REFUSED_SDK_ENVS if (os.environ.get(name) or "").strip()]
    if ambient:
        raise ValueError(f"{' and '.join(ambient)} set: the anthropic SDK would use it for every call, so the batches "
                         f"would not go to api.anthropic.com on {API_KEY_ENV} alone; unset it for this transport "
                         f"({ANOTHER_SERVER})")
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def _sdk_client() -> Any:
    return default_client()


def effort_level(value: Optional[str]) -> Optional[str]:
    """Return a config's reasoning level as ``output_config.effort`` carries it, or None."""
    if value is None:
        return None
    level = str(value).strip().lower()
    if level not in EFFORT_LEVELS:
        raise ValueError(f"unknown Claude effort level {value!r}; expected one of {EFFORT_LEVELS}")
    return level


def display_setting(value: Optional[str]) -> Optional[str]:
    """Return a config's ``thinking_display`` as ``thinking.display`` carries it, or None."""
    if value is None:
        return None
    display = str(value).strip().lower()
    if display not in THINKING_DISPLAYS:
        raise ValueError(f"unknown thinking display {value!r}; expected one of {THINKING_DISPLAYS}")
    return display


def check_custom_id(custom_id: str) -> str:
    """Return ``custom_id`` if the API accepts it, else raise ValueError."""
    if not isinstance(custom_id, str) or not CUSTOM_ID_PATTERN.fullmatch(custom_id):
        raise ValueError(f"custom id {custom_id!r} is not one the Message Batches API takes "
                         f"(^[a-zA-Z0-9_-]{{1,64}}$)")
    return custom_id


def check_temperature(temperature: float) -> None:
    """Refuse any temperature other than the provider's 1.0: requests carry none."""
    if float(temperature) != PROVIDER_TEMPERATURE:
        raise ValueError(f"an anthropic_batch request sends no temperature and samples at the provider's "
                         f"{PROVIDER_TEMPERATURE}; {temperature!r} cannot be honoured (Claude Haiku 5.5 answers a "
                         f"non-default temperature with a 400)")


def build_request_params(
    system_prompt: str, user_prompt: str, *, model: str, max_tokens: int, effort: Optional[str] = None,
    display: Optional[str] = None,
) -> Dict[str, Any]:
    """Return one Messages API request; ``max_tokens`` is required, and no sampling parameter, seed or prefill is
    sent."""
    thinking: Dict[str, Any] = dict(THINKING)
    if display is not None:
        thinking["display"] = display
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": [{"role": "user", "content": user_prompt if user_prompt.strip() else EMPTY_USER_TURN}],
        "thinking": thinking,
    }
    if system_prompt:
        params["system"] = system_prompt
    if effort is not None:
        params["output_config"] = {"effort": effort}
    return params


def build_requests(
    prompts: Sequence[Mapping[str, str]],
    *,
    model: str,
    max_tokens: int,
    num_responses: int,
    seed: Optional[int],
    effort: Optional[str] = None,
    namespace: Optional[str] = None,
    display: Optional[str] = None,
) -> List[BatchRequest]:
    """Return one request per (prompt, candidate); the seed appears in the custom id alone."""
    if int(max_tokens) < 1:
        raise ValueError(f"max_tokens must be at least 1, got {max_tokens!r}")
    requests: List[BatchRequest] = []
    for prompt_index, entry in enumerate(prompts):
        params = build_request_params(entry.get("system_prompt", ""), entry.get("user_prompt", ""), model=model,
                                      max_tokens=max_tokens, effort=effort, display=display)
        for candidate_index in range(num_responses):
            requests.append(BatchRequest(
                custom_id=check_custom_id(batch_custom_id(prompt_index, candidate_index, seed, namespace)),
                body=params))
    return requests


def prompt_of_params(params: Mapping[str, Any]) -> Dict[str, str]:
    """Return the ``{system_prompt, user_prompt}`` entry a request was built from."""
    user = params["messages"][0]["content"]
    return {"system_prompt": params.get("system") or "", "user_prompt": "" if user == EMPTY_USER_TURN else user}


def request_line(request: BatchRequest) -> bytes:
    """Return one input-file line, ``{"custom_id": ..., "params": ...}``, with keys sorted and separators fixed."""
    payload = {"custom_id": request.custom_id, "params": request.body}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"


def _wire_size(line: bytes) -> int:
    """Return the most bytes a line can take on the wire: every non-ASCII character escaped."""
    return len(json.dumps(json.loads(line), ensure_ascii=True, separators=(",", ":")).encode("ascii"))


def fingerprint(requests: Sequence[BatchRequest]) -> str:
    """Return the content address of a request set: the endpoint and every line, in order."""
    digest = hashlib.sha256(f"{ANTHROPIC_BATCH_ENDPOINT}\0".encode("utf-8"))
    for request in requests:
        digest.update(request_line(request))
    return digest.hexdigest()[:32]


def write_shards(
    requests: Sequence[BatchRequest],
    directory: Path,
    *,
    max_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_requests: int = DEFAULT_MAX_SHARD_REQUESTS,
) -> List[Dict[str, Any]]:
    """Write the input file(s) under ``directory``; return one descriptor (path, requests, bytes, SHA-256) per file."""
    if not requests:
        raise ValueError("cannot prepare a batch with no requests")
    seen: set = set()
    groups: List[List[bytes]] = [[]]
    size = 0
    for request in requests:
        check_custom_id(request.custom_id)
        if request.custom_id in seen:
            raise ValueError(f"custom id {request.custom_id!r} appears twice; a batch's ids must be unique")
        seen.add(request.custom_id)
        line = request_line(request)
        wire = _wire_size(line)
        if wire > max_bytes:
            raise ValueError(f"request {request.custom_id!r} is {wire} bytes, over the {max_bytes}-byte batch "
                             f"limit; it can never be submitted")
        if groups[-1] and (size + wire > max_bytes or len(groups[-1]) >= max_requests):
            groups.append([])
            size = 0
        groups[-1].append(line)
        size += wire
    directory.mkdir(parents=True, exist_ok=True)
    shards: List[Dict[str, Any]] = []
    for index, lines in enumerate(groups):
        payload = b"".join(lines)
        name = f"part-{index:04d}.jsonl"
        write_bytes_atomic(directory / name, payload)
        shards.append({"path": name, "requests": len(lines), "bytes": len(payload), "sha256": sha256_bytes(payload)})
    return shards


def _lines(payload: bytes) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in payload.decode("utf-8").split("\n") if line.strip()]


def _count(mapping: Any, key: str) -> int:
    value = mapping.get(key) if isinstance(mapping, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def add_usage(usage: BatchUsage, message: Mapping[str, Any]) -> None:
    """Add one succeeded message's ``usage`` to the buckets; ``input_tokens`` includes cache reads and writes."""
    reported = message.get("usage")
    fresh, read, written = (_count(reported, key) for key in
                            ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    usage.input_tokens += fresh + read + written
    usage.cached_tokens += read
    usage.cache_write_tokens += written
    usage.output_tokens += _count(reported, "output_tokens")
    details = reported.get("output_tokens_details") if isinstance(reported, Mapping) else None
    usage.reasoning_tokens += _count(details, "thinking_tokens")
    usage.requests += 1


def _blocks_of(message: Mapping[str, Any], kind: str, field: str) -> List[str]:
    content = message.get("content")
    return [block[field] for block in (content if isinstance(content, list) else [])
            if isinstance(block, Mapping) and block.get("type") == kind and isinstance(block.get(field), str)]


def outcome_from_message(message: Mapping[str, Any]) -> BatchOutcome:
    """Return the answer, finish reason and reasoning summaries of one succeeded message."""
    summaries = tuple(text for text in _blocks_of(message, "thinking", "thinking") if text.strip())
    stop = message.get("stop_reason")
    stop = stop if isinstance(stop, str) else None
    if stop == REFUSAL:
        details = message.get("stop_details")
        category = details.get("category") if isinstance(details, Mapping) else None
        return BatchOutcome(
            finish_reason=f"{REFUSAL}:{category if isinstance(category, str) and category else REFUSAL_UNSPECIFIED}",
            reasoning_summaries=summaries)
    text = "".join(_blocks_of(message, "text", "text")).strip() or None
    if text is None:
        return BatchOutcome(finish_reason=stop or NO_TEXT, reasoning_summaries=summaries)
    return BatchOutcome(text=text, finish_reason=stop, reasoning_summaries=summaries)


def _describe_error(error: Any) -> str:
    inner = error.get("error") if isinstance(error, Mapping) else None
    kind = inner.get("type") if isinstance(inner, Mapping) else None
    if not isinstance(kind, str) and isinstance(error, Mapping):
        kind = error.get("type")
    return f"{RESULT_ERRORED}:{kind}" if isinstance(kind, str) and kind else RESULT_ERRORED


def outcome_from_record(record: Mapping[str, Any]) -> tuple[BatchOutcome, Optional[Mapping[str, Any]]]:
    """Return one result line as an outcome and the message to bill for (None: it bills nothing); refuses other
    shapes."""
    result = record.get("result")
    kind = result.get("type") if isinstance(result, Mapping) else None
    if kind == RESULT_SUCCEEDED and isinstance(result.get("message"), Mapping):
        return outcome_from_message(result["message"]), result["message"]
    if kind == RESULT_ERRORED:
        return BatchOutcome(error=_describe_error(result.get("error"))), None
    if kind in UNFINISHED_RESULTS:
        return BatchOutcome(error=kind), None
    raise BatchExecutionError(f"result for {record.get('custom_id')!r} is of no type this build reads ({kind!r})")


def result_record(item: Any) -> Dict[str, Any]:
    """Return one decoded result as the dict the output file keeps."""
    if isinstance(item, Mapping):
        return dict(item)
    return item.to_dict(mode="json")


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value if isinstance(value, str) else None


def _dump(value: Any) -> Any:
    if value is None:
        return None
    return dict(value) if isinstance(value, Mapping) else value.model_dump()


def part_counts(part: Mapping[str, Any]) -> Dict[str, int]:
    """Return a part's last recorded ``request_counts``."""
    counts = part.get("request_counts")
    return {key: int(value) for key, value in counts.items()} if isinstance(counts, Mapping) else {}


def part_state(part: Mapping[str, Any]) -> Optional[str]:
    """Return a part's state: its ``processing_status`` until it ended, then ``PART_COMPLETED`` or
    ``PART_INCOMPLETE``."""
    status = part.get("processing_status")
    if status != PROCESSING_ENDED:
        return status
    counts = part_counts(part)
    return PART_INCOMPLETE if any(counts.get(key) for key in UNFINISHED_RESULTS) else PART_COMPLETED


@dataclass
class AnthropicBatchRunner:
    """Prepare, submit, poll, download and collect one Message Batch; ``client_factory`` is called only when needed."""

    root: Path
    model: str
    client_factory: Callable[[], Any] = _sdk_client
    poll_seconds: float = DEFAULT_POLL_SECONDS
    timeout_seconds: Optional[float] = None
    max_poll_errors: int = DEFAULT_MAX_POLL_ERRORS
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES
    max_shard_requests: int = DEFAULT_MAX_SHARD_REQUESTS
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    clock: Callable[[], datetime] = utc_now

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("an Anthropic batch needs a model name")
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
        return work_directory(self.root, fingerprint(requests))

    def _client_or_init(self) -> Any:
        if self._client is None:
            self._client = self.client_factory()
        return self._client

    def prepare(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Write the input file(s) and the state record, or return the state already there; makes no API call."""
        for request in requests:
            if request.body.get("model") != self.model:
                raise ValueError(f"request {request.custom_id!r} is for {request.body.get('model')!r}, not this "
                                 f"runner's {self.model!r}")
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        if state_path.is_file():
            return self._load_state(state_path)
        shards = write_shards(requests, directory / "inputs", max_bytes=self.max_shard_bytes,
                              max_requests=self.max_shard_requests)
        state = {"state_version": STATE_VERSION, "endpoint": ANTHROPIC_BATCH_ENDPOINT, "model": self.model,
                 "requests": len(requests), "shards": shards, "parts": {shard["path"]: {} for shard in shards}}
        if not write_json_once(state_path, state):
            return self._load_state(state_path)
        logger.info("Prepared %d Anthropic batch request(s) in %d input file(s) at %s", len(requests), len(shards),
                    directory)
        return state

    def _load_state(self, state_path: Path) -> Dict[str, Any]:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        version = state.get("state_version")
        if version != STATE_VERSION:
            raise BatchExecutionError(f"{state_path} was written by state version {version!r}; this build writes "
                                      f"{STATE_VERSION}. Refusing to resume it -- inspect it by hand.")
        if state.get("endpoint") != ANTHROPIC_BATCH_ENDPOINT or state.get("model") != self.model:
            raise BatchExecutionError(f"{state_path} was prepared for {state.get('model')!r} on "
                                      f"{state.get('endpoint')!r}, not {self.model!r} on {ANTHROPIC_BATCH_ENDPOINT!r}")
        return state

    def submit(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Create a batch for every input file that has none yet; a create that recorded no batch is refused, never
        repeated."""
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

    def _create_missing_batches(
        self, state: Dict[str, Any], state_path: Path, directory: Path, client: Any
    ) -> None:
        if all(state["parts"].get(shard["path"], {}).get("batch_id") for shard in state["shards"]):
            return
        payloads: Dict[str, bytes] = {}
        for shard in state["shards"]:
            name = shard["path"]
            part = state["parts"].setdefault(name, {})
            if not part.get("batch_id") and part.get("create_started_at"):
                raise BatchExecutionError(self._unrecorded_create(name, shard, part, state_path))
            path = directory / "inputs" / name
            payloads[name] = path.read_bytes()
            if sha256_bytes(payloads[name]) != shard["sha256"]:
                raise BatchExecutionError(f"prepared input file changed on disk: {path}")
        for shard in state["shards"]:
            name = shard["path"]
            part = state["parts"][name]
            if part.get("batch_id"):
                continue
            lines = _lines(payloads[name])
            # Persisted before the call: a create is not idempotent, so a failure after it keeps this mark.
            part["create_started_at"] = self.clock().isoformat()
            write_json_atomic(state_path, state)
            try:
                batch = client.with_options(max_retries=0).messages.batches.create(requests=lines)
            except Exception as exc:
                if not create_was_refused(exc):
                    raise BatchExecutionError(
                        f"messages.batches.create failed ({type(exc).__name__}: {exc}) in a way that does not say "
                        f"whether the batch was made, so it is not retried. "
                        + self._unrecorded_create(name, shard, part, state_path)) from exc
                del part["create_started_at"]
                write_json_atomic(state_path, state)
                raise
            batch_id = getattr(batch, "id", None)
            if not isinstance(batch_id, str) or not batch_id:
                raise BatchExecutionError("messages.batches.create returned no batch id. "
                                          + self._unrecorded_create(name, shard, part, state_path))
            part["batch_id"] = batch_id
            self._record(part, batch)
            try:
                write_json_atomic(state_path, state)
            except Exception as exc:
                raise BatchExecutionError(
                    f"Anthropic batch {batch_id} was created for {name}, but recording it in {state_path} failed "
                    f"({exc}). Write \"batch_id\": \"{batch_id}\" into parts[{name!r}] there by hand, then rerun to "
                    f"wait on it; until then a rerun refuses rather than buy it again.") from exc
            logger.info("Submitted %s (%d request(s)) as Anthropic batch %s", name, len(lines), batch_id)

    @staticmethod
    def _unrecorded_create(name: str, shard: Mapping[str, Any], part: Mapping[str, Any], state_path: Path) -> str:
        return (
            f"{name}: a messages.batches.create of {shard['requests']} request(s) started at "
            f"{part.get('create_started_at')} and recorded no batch, so it may have made one, and a second create "
            f"would buy it twice. List the account's batches (client.messages.batches.list()): "
            f"one created just after that time with {shard['requests']} request(s) whose results carry "
            f"this file's custom ids is it -- write its id as parts[{name!r}].batch_id in {state_path} and rerun to "
            f"wait on it; if there is none, delete parts[{name!r}].create_started_at there and "
            f"rerun to create it."
        )

    @staticmethod
    def _record(part: Dict[str, Any], batch: Any) -> None:
        part["processing_status"] = getattr(batch, "processing_status", None)
        part["request_counts"] = _dump(getattr(batch, "request_counts", None))
        for key in ("ended_at", "expires_at", "cancel_initiated_at"):
            part[key] = _iso(getattr(batch, key, None))
        part["results_url"] = getattr(batch, "results_url", None)

    def _sweep(self, state: Dict[str, Any], state_path: Path, client: Any) -> tuple[List[str], bool]:
        pending: List[str] = []
        errored = False
        for name, part in state["parts"].items():
            batch_id = part.get("batch_id")
            if not batch_id:
                raise BatchExecutionError(f"input file {name} was never submitted")
            if part.get("processing_status") == PROCESSING_ENDED:
                continue
            try:
                batch = client.messages.batches.retrieve(batch_id)
            except Exception as exc:
                # A failed status check says nothing about the batch, which is running and paid for.
                logger.warning("Status check for %s failed (%s); the batch is still running, so waiting rather "
                               "than giving up", batch_id, exc)
                errored = True
                pending.append(f"{name}=unknown")
                continue
            self._record(part, batch)
            if part["processing_status"] != PROCESSING_ENDED:
                pending.append(f"{name}={part['processing_status']}")
        write_json_atomic(state_path, state)
        return pending, errored

    def refresh(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Re-read every part once into ``state.json`` and return the state; a failed call raises."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        state = self._load_state(state_path)
        client = self._client_or_init()
        for name, part in state["parts"].items():
            batch_id = part.get("batch_id")
            if not batch_id:
                raise BatchExecutionError(f"input file {name} was never submitted")
            self._record(part, client.messages.batches.retrieve(batch_id))
        write_json_atomic(state_path, state)
        return state

    @staticmethod
    def _finished_or_raise(state: Dict[str, Any], directory: Path) -> None:
        incomplete = {name: {"batch_id": part.get("batch_id"), "request_counts": part_counts(part)}
                      for name, part in state["parts"].items() if part_state(part) != PART_COMPLETED}
        if incomplete:
            raise BatchExecutionError(
                f"Anthropic batch(es) ended without completing (canceled or expired requests): {incomplete}. "
                f"Nothing is resubmitted automatically, because a resubmission re-buys every request. Collect and "
                f"bill what returned with: ruleforge campaign collect {directory}")

    def poll(self, requests: Sequence[BatchRequest]) -> Dict[str, Any]:
        """Block until every batch has ended; raise unless each completed."""
        directory = self.workdir(requests)
        state_path = directory / "state.json"
        state = self._load_state(state_path)
        client = self._client_or_init()
        deadline = self.monotonic() + self.timeout_seconds if self.timeout_seconds is not None else None
        consecutive_errors = 0
        while True:
            pending, errored = self._sweep(state, state_path, client)
            if not pending:
                self._finished_or_raise(state, directory)
                return state
            consecutive_errors = consecutive_errors + 1 if errored else 0
            if consecutive_errors >= self.max_poll_errors:
                raise BatchExecutionError(
                    f"{consecutive_errors} status checks in a row failed; the batch may still be running, and its "
                    f"state is in {directory}. This looks permanent -- a rotated key, a deleted batch -- not the "
                    f"transient blip the retry tolerates. Fix it and rerun to resume; the work is already paid for.")
            if deadline is not None and self.monotonic() >= deadline:
                raise BatchExecutionError(
                    f"batch did not finish within {self.timeout_seconds}s; it is still running and its state is in "
                    f"{directory}. Rerun to resume waiting -- the work is already paid for.")
            logger.info("Waiting on %d Anthropic batch(es): %s", len(pending), ", ".join(pending))
            self.sleep(self.poll_seconds)

    def download(self, requests: Sequence[BatchRequest]) -> Path:
        """Fetch every ended batch's results not already on disk, one line per result."""
        directory = self.workdir(requests)
        state = self._load_state(directory / "state.json")
        client = self._client_or_init()
        outputs = directory / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        for name, part in state["parts"].items():
            destination = outputs / name
            if destination.exists():
                continue
            if part.get("processing_status") != PROCESSING_ENDED or not part.get("results_url"):
                raise BatchExecutionError(f"batch {part.get('batch_id')} ({name}) has not ended with results to "
                                          f"download; its status is {part.get('processing_status')!r}")
            temporary = private_temporary(destination)
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    for item in client.messages.batches.results(part["batch_id"]):
                        handle.write(json.dumps(result_record(item), ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")) + "\n")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            logger.info("Downloaded the results of %s -> %s", part["batch_id"], destination)
        return outputs

    def read_results(
        self, requests: Sequence[BatchRequest], *, partial: bool = False
    ) -> tuple[Dict[str, BatchOutcome], BatchUsage]:
        """Join the results to the requests by custom id and count the usage; a missing, unknown or repeated id or a
        count mismatch raises (a damaged file), as does a canceled or expired result unless ``partial``."""
        directory = self.workdir(requests)
        parts = self._load_state(directory / "state.json")["parts"]
        expected = [request.custom_id for request in requests]
        known = set(expected)
        outcomes: Dict[str, BatchOutcome] = {}
        usage = BatchUsage()
        for path in sorted((directory / "outputs").glob("part-*.jsonl")):
            kinds: Counter = Counter()
            for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BatchExecutionError(f"{path}:{number} is not JSON: {exc}") from None
                custom_id = record.get("custom_id") if isinstance(record, dict) else None
                if not isinstance(custom_id, str):
                    raise BatchExecutionError(f"{path}:{number} carries no custom id")
                if custom_id not in known:
                    raise BatchExecutionError(f"{path}:{number} answers unknown custom id {custom_id!r}")
                if custom_id in outcomes:
                    raise BatchExecutionError(f"{path}:{number} repeats custom id {custom_id!r}")
                outcome, message = outcome_from_record(record)
                if outcome.error in UNFINISHED_RESULTS and not partial:
                    raise BatchExecutionError(f"{path}:{number}: request {custom_id} was {outcome.error}, so its "
                                              f"batch ended without completing; collect it with "
                                              f"ruleforge campaign collect {directory}")
                if message is None:
                    logger.warning("Anthropic batch request %s failed: %s", custom_id, outcome.error)
                else:
                    add_usage(usage, message)
                outcomes[custom_id] = outcome
                kinds[record["result"]["type"]] += 1
            self._check_counts(path, parts.get(path.name), kinds)
        missing = [custom_id for custom_id in expected if custom_id not in outcomes]
        if missing:
            raise BatchExecutionError(f"{len(missing)} of {len(expected)} request(s) have no result in "
                                      f"{directory / 'outputs'} (first: {missing[:3]}); an ended batch's results "
                                      f"hold every request, so the file is damaged: delete it and rerun to download "
                                      f"it again")
        refusals = Counter(outcome.finish_reason for outcome in outcomes.values()
                           if (outcome.finish_reason or "").startswith(f"{REFUSAL}:"))
        if refusals:
            logger.warning("Anthropic batch %s: %d refusal(s) by category: %s", directory.name,
                           sum(refusals.values()), dict(sorted(refusals.items())))
        write_json_atomic(directory / "usage.json", usage.to_dict())
        write_reasoning_summaries(directory, outcomes)
        return outcomes, usage

    @staticmethod
    def _check_counts(path: Path, part: Optional[Mapping[str, Any]], kinds: Mapping[str, int]) -> None:
        if part is None or part.get("processing_status") != PROCESSING_ENDED:
            raise BatchExecutionError(f"{path} belongs to no ended batch in its state.json; it is not results this "
                                      f"build downloaded")
        counted = {kind: _count(part.get("request_counts"), kind)
                   for kind in (RESULT_SUCCEEDED, RESULT_ERRORED, *UNFINISHED_RESULTS)}
        found = {kind: kinds.get(kind, 0) for kind in counted}
        if found != counted:
            raise BatchExecutionError(
                f"{path} holds {found} result(s) by type, but the API counted {counted} for batch "
                f"{part.get('batch_id')} when it ended: the file is not the batch's whole results. Delete it and "
                f"rerun to download it again")

    def run(self, requests: Sequence[BatchRequest]) -> tuple[Dict[str, BatchOutcome], BatchUsage]:
        """Prepare, submit, wait, download and collect, in one blocking call."""
        self.submit(requests)
        self.poll(requests)
        self.download(requests)
        return self.read_results(requests)


def _runner_settings(
    *, model_name: str, root: Path, poll_seconds: float, timeout_seconds: Optional[float],
    client_factory: Optional[Callable[[], Any]],
) -> Dict[str, Any]:
    runner_kwargs: Dict[str, Any] = dict(root=Path(root), model=model_name, poll_seconds=poll_seconds,
                                         timeout_seconds=timeout_seconds)
    if client_factory is not None:
        runner_kwargs["client_factory"] = client_factory
    return runner_kwargs


def make_block_generate(
    *,
    model_name: str,
    root: Path,
    reasoning_effort: Optional[str] = None,
    thinking_display: Optional[str] = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    tracker: Any | None = None,
) -> Callable[[List[Dict[str, str]], float, int, Optional[Sequence[Optional[int]]], str], List[Dict[str, Any]]]:
    """Return the one-attempt request-block callable for ``anthropic_batch``, as in
    ``openai_batch.make_block_generate``."""
    runner_kwargs = _runner_settings(model_name=model_name, root=root, poll_seconds=poll_seconds,
                                     timeout_seconds=timeout_seconds, client_factory=client_factory)
    effort = effort_level(reasoning_effort)
    display = display_setting(thinking_display)
    seed_warning: Dict[str, bool] = {}

    def _generate_block(
        prompts: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        seeds: Optional[Sequence[Optional[int]]],
        request_block_id: str,
    ) -> List[Dict[str, Any]]:
        from ruleforge.core.digests import request_namespace

        check_temperature(temperature)
        if seeds is not None:
            _warn_seed_ignored_once(seed_warning, "The Anthropic batch path",
                                    next((seed for seed in seeds if seed is not None), None))
        namespace = request_namespace(request_block_id)
        requests = build_requests(prompts, model=model_name, max_tokens=max_new_tokens, num_responses=1, seed=None,
                                  effort=effort, namespace=namespace, display=display)
        return collect_block_details(AnthropicBatchRunner(**runner_kwargs), requests, namespace=namespace,
                                     provider="Anthropic", tracker=tracker,
                                     summaries_requested=display == DISPLAY_SUMMARIZED)

    return _generate_block


def make_batch_generate(
    *,
    model_name: str,
    root: Path,
    reasoning_effort: Optional[str] = None,
    thinking_display: Optional[str] = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    tracker: Any | None = None,
) -> tuple[
    Callable[[List[Dict[str, str]], float, int, int, Optional[int]], List[List[str]]],
    Callable[[List[Dict[str, str]], float, int, int, Optional[int]], List[List[Dict[str, Any]]]],
]:
    """Return ``(_batch_generate, _batch_generate_with_details)`` for ``anthropic_batch``."""
    runner_kwargs = _runner_settings(model_name=model_name, root=root, poll_seconds=poll_seconds,
                                     timeout_seconds=timeout_seconds, client_factory=client_factory)
    effort = effort_level(reasoning_effort)
    display = display_setting(thinking_display)
    summarized = display == DISPLAY_SUMMARIZED
    resume_guard = ResumeGuard()
    seed_warning: Dict[str, bool] = {}

    def _run(
        prompts: List[Dict[str, str]], temperature: float, max_new_tokens: int, num_responses: int,
        seed: Optional[int],
    ) -> List[List[BatchOutcome]]:
        check_temperature(temperature)
        seed = resume_guard.seed_for(seed)
        resume_guard.arm(seed)
        _warn_seed_ignored_once(seed_warning, "The Anthropic batch path", seed)
        requests = build_requests(prompts, model=model_name, max_tokens=max_new_tokens, num_responses=num_responses,
                                  seed=seed, effort=effort, display=display)
        outcomes, usage = AnthropicBatchRunner(**runner_kwargs).run(requests)
        track_usage(tracker, usage)
        resume_guard.clear()
        grid = reshape_outcomes(outcomes, num_prompts=len(prompts), num_responses=num_responses, seed=seed)
        log_batch_completion(grid, "Anthropic")
        return grid

    return blocking_generators(_run, summaries_requested=summarized)
