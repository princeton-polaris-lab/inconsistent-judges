from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from ruleforge.core.errors import RuleForgeError
from ruleforge.judges.token_usage import TokenUsage

logger = logging.getLogger(__name__)

DEFAULT_POLL_SECONDS = 60.0
DEFAULT_COMPLETION_WINDOW = "24h"

RESPONSES_ENDPOINT = "/v1/responses"
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

# ``completed`` is the only one that carries results.
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})

ENDPOINT_ALIASES = {
    "responses": RESPONSES_ENDPOINT,
    "chat_completions": CHAT_COMPLETIONS_ENDPOINT,
}


def resolve_endpoint(name: str) -> str:
    """Return a config's endpoint name as its URL path; a URL path passes unchanged."""
    if name in ENDPOINT_ALIASES:
        return ENDPOINT_ALIASES[name]
    if name in TEXT_EXTRACTORS:
        return name
    raise ValueError(
        f"unknown batch endpoint {name!r}; expected one of "
        f"{sorted(ENDPOINT_ALIASES)} or {sorted(TEXT_EXTRACTORS)}"
    )


class BatchSpendNotConfirmed(RuleForgeError, RuntimeError):
    """Raised when a billable call is attempted unconfirmed."""


class BatchExecutionError(RuntimeError, RuleForgeError):
    """Raised when a batch cannot be prepared, resumed, or read back."""


class BatchDestinationRefused(BatchExecutionError):
    """A batch work directory the active guard refused before the transport read or wrote anything in it."""


# Errors the retry loops let through: the next attempt would fail the same way, or buy the same work again.
FATAL_PROVIDER_ERRORS = (BatchSpendNotConfirmed, BatchExecutionError)


_WORK_DIRECTORY_GUARD: ContextVar[Optional[Callable[[Path, Path], None]]] = ContextVar(
    "work_directory_guard", default=None)


def work_directory(root: Path, name: str) -> Path:
    """Return ``root / name`` once the active guard, if any, accepts it."""
    directory = Path(root) / name
    guard = _WORK_DIRECTORY_GUARD.get()
    if guard is not None:
        guard(Path(root), directory)
    return directory


def work_directories_guarded() -> bool:
    """Return whether a guard is set."""
    return _WORK_DIRECTORY_GUARD.get() is not None


@contextmanager
def guard_work_directories(guard: Callable[[Path, Path], None]) -> Iterator[None]:
    """Run ``guard(batch_root, work_directory)`` on every work directory a transport derives inside the block."""
    token = _WORK_DIRECTORY_GUARD.set(guard)
    try:
        yield
    finally:
        _WORK_DIRECTORY_GUARD.reset(token)


@dataclass(frozen=True)
class BatchRequest:
    """One request in a batch: an id the results are joined on, and a body."""

    custom_id: str
    body: Dict[str, Any]


@dataclass(frozen=True)
class BatchOutcome:
    """What came back for one request; ``text`` is None when it errored, never returned, or produced no text."""

    text: Optional[str] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    reasoning_summaries: tuple = ()
    reasoning: Optional[str] = None


@dataclass
class BatchUsage:
    """The token buckets a batch is billed against: fresh, cached and cache-written input bill at different rates."""

    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0

    def add_body(self, body: Mapping[str, Any]) -> None:
        """Add one response body's reported usage."""
        usage = body.get("usage") or {}
        # Responses nests details under input_/output_tokens_details, chat under prompt_/completion_tokens_details.
        input_details = (
            usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        )
        output_details = (
            usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )

        def _int(value: Any) -> int:
            return int(value) if isinstance(value, (int, float)) else 0

        self.input_tokens += _int(usage.get("input_tokens")) or _int(
            usage.get("prompt_tokens")
        )
        self.output_tokens += _int(usage.get("output_tokens")) or _int(
            usage.get("completion_tokens")
        )
        self.cached_tokens += _int(input_details.get("cached_tokens"))
        self.cache_write_tokens += _int(input_details.get("cache_write_tokens"))
        self.reasoning_tokens += _int(output_details.get("reasoning_tokens"))
        self.requests += 1

    def to_dict(self) -> Dict[str, int]:
        """Return a serializable snapshot, for the run's usage record."""
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "requests": self.requests,
        }


def build_responses_body(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the body of one ``/v1/responses`` request; that endpoint calls the system turn ``developer``."""
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "developer", "content": system_prompt})
    if user_prompt.strip():
        messages.append({"role": "user", "content": user_prompt})

    body: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        reasoning: Dict[str, str] = {"effort": reasoning_effort}
        if reasoning_summary is not None:
            reasoning["summary"] = reasoning_summary
        body["reasoning"] = reasoning
    if max_output_tokens is not None:
        body["max_output_tokens"] = int(max_output_tokens)
    return body


def build_chat_body(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the body of one ``/v1/chat/completions`` request; only this endpoint accepts ``seed``."""
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt.strip():
        messages.append({"role": "user", "content": user_prompt})

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if max_output_tokens is not None:
        body["max_completion_tokens"] = int(max_output_tokens)
    if seed is not None:
        body["seed"] = int(seed)
    return body


BODY_BUILDERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    RESPONSES_ENDPOINT: build_responses_body,
    CHAT_COMPLETIONS_ENDPOINT: build_chat_body,
}


def batch_custom_id(
    prompt_index: int,
    candidate_index: int,
    seed: Optional[int],
    namespace: Optional[str] = None,
) -> str:
    """Return a stable id for one (prompt, candidate) pair, prefixed by ``namespace`` or else the seed, so a retry after
    unusable answers addresses a new batch (``ResumeGuard`` covers one after a failure)."""
    prefix = namespace if namespace is not None else f"s{'none' if seed is None else seed}"
    return f"{prefix}-p{prompt_index:06d}-c{candidate_index:02d}"


def custom_id_prefix(custom_ids: Iterable[str]) -> Optional[str]:
    """Return the prefix all custom ids share (a namespace or ``s<seed>``), or None when they share none."""
    prefixes = {str(custom_id).rsplit("-p", 1)[0] for custom_id in custom_ids}
    return prefixes.pop() if len(prefixes) == 1 else None


def build_batch_requests(
    prompts: Sequence[Mapping[str, str]],
    *,
    endpoint: str,
    model: str,
    temperature: float,
    num_responses: int,
    seed: Optional[int],
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    seeds: Optional[Sequence[Optional[int]]] = None,
    namespace: Optional[str] = None,
) -> List[BatchRequest]:
    """Expand prompt dicts into one request per candidate; ``seeds``, if given, holds one nullable seed per prompt."""
    if seeds is not None and len(seeds) != len(prompts):
        raise ValueError(
            f"seeds must align with prompts, got {len(seeds)} seeds for "
            f"{len(prompts)} prompts"
        )
    if endpoint not in BODY_BUILDERS:
        raise ValueError(
            f"unsupported batch endpoint {endpoint!r}; "
            f"expected one of {sorted(BODY_BUILDERS)}"
        )
    if endpoint != RESPONSES_ENDPOINT and reasoning_summary is not None:
        logger.warning(
            "reasoning_summary=%r is only carried by %s; %s has no summary "
            "block, so no reasoning summaries will be saved.",
            reasoning_summary,
            RESPONSES_ENDPOINT,
            endpoint,
        )
    requests: List[BatchRequest] = []
    for prompt_index, entry in enumerate(prompts):
        system_prompt = entry.get("system_prompt", "")
        user_prompt = entry.get("user_prompt", "")
        prompt_seed = seeds[prompt_index] if seeds is not None else seed
        for candidate_index in range(num_responses):
            if endpoint == RESPONSES_ENDPOINT:
                body = build_responses_body(
                    system_prompt,
                    user_prompt,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=reasoning_summary,
                    max_output_tokens=max_output_tokens,
                )
            else:
                body = build_chat_body(
                    system_prompt,
                    user_prompt,
                    model=model,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    seed=None if prompt_seed is None else prompt_seed + candidate_index,
                )
            requests.append(
                BatchRequest(
                    custom_id=batch_custom_id(
                        prompt_index, candidate_index, seed, namespace
                    ),
                    body=body,
                )
            )
    return requests


def responses_text(body: Mapping[str, Any]) -> Optional[str]:
    """Return the assistant text of a ``/v1/responses`` body, case kept (forced answers have ``"Rule 1"`` keys)."""
    texts: List[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if not isinstance(content, Mapping) or content.get("type") != "output_text":
                continue
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    joined = "".join(texts).strip()
    return joined or None


def chat_text(body: Mapping[str, Any]) -> Optional[str]:
    """Return the assistant text of a ``/v1/chat/completions`` body."""
    for choice in body.get("choices") or []:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return None


def responses_finish_reason(body: Mapping[str, Any]) -> Optional[str]:
    """Return why a ``/v1/responses`` generation stopped, which tells a truncated answer from a brief one."""
    status = body.get("status")
    if status == "incomplete":
        reason = (body.get("incomplete_details") or {}).get("reason")
        return f"incomplete:{reason}" if isinstance(reason, str) else "incomplete"
    return status if isinstance(status, str) else None


# The statuses of a Responses body that answered; ``incomplete`` is a truncated answer.
_RESPONSES_ANSWERED = ("completed", "incomplete")


def responses_failure(response: Any) -> Optional[str]:
    """Return why a ``/v1/responses`` body or SDK response holds no answer (its status is not an answer's), or None."""
    status = _read(response, "status")
    if not isinstance(status, str) or status in _RESPONSES_ANSWERED:
        return None
    error = _read(response, "error")
    named = [str(value) for value in (_read(error, "code"), _read(error, "message")) if value]
    return f"the response's status is {status!r}" + (f" ({': '.join(named)})" if named else "")


def chat_finish_reason(body: Mapping[str, Any]) -> Optional[str]:
    """Return why a ``/v1/chat/completions`` generation stopped."""
    for choice in body.get("choices") or []:
        if isinstance(choice, Mapping) and isinstance(choice.get("finish_reason"), str):
            return choice["finish_reason"]
    return None


def _read(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _items(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _reasoning_parts(body: Any, field: str, part_type: str) -> List[str]:
    texts: List[str] = []
    for item in _items(_read(body, "output")):
        if _read(item, "type") != "reasoning":
            continue
        for part in _items(_read(item, field)):
            if _read(part, "type") != part_type:
                continue
            text = _read(part, "text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def reasoning_summaries(body: Any) -> List[str]:
    """Return a ``/v1/responses`` body's reasoning summary texts; an empty list is normal."""
    return _reasoning_parts(body, "summary", "summary_text")


def _joined_reasoning(texts: Sequence[str]) -> Optional[str]:
    joined = "\n".join(texts)
    return joined if joined.strip() else None


def responses_reasoning(body: Any) -> Optional[str]:
    """Return the ``reasoning_text`` content of a ``/v1/responses`` body, or None."""
    return _joined_reasoning(_reasoning_parts(body, "content", "reasoning_text"))


# The ``reasoning_details`` entry types whose text is readable (OpenRouter), and the field holding it.
_REASONING_DETAIL_TEXT = {"reasoning.text": "text", "reasoning.summary": "summary"}


def message_reasoning(message: Any) -> Optional[str]:
    """Return the reasoning a chat message or delta carried apart from its content, or None."""
    for key in ("reasoning", "reasoning_content"):
        value = _read(message, key)
        if isinstance(value, str) and value.strip():
            return value
    texts: List[str] = []
    for detail in _items(_read(message, "reasoning_details")):
        kind = _read(detail, "type")
        field = _REASONING_DETAIL_TEXT.get(kind) if isinstance(kind, str) else None
        text = _read(detail, field) if field else None
        if isinstance(text, str):
            texts.append(text)
    return _joined_reasoning(texts)


def chat_reasoning(body: Any) -> Optional[str]:
    """Return the reasoning of a ``/v1/chat/completions`` body's first choice, or None."""
    first = next(iter(_items(_read(body, "choices"))), None)
    message = _read(first, "message")
    return None if message is None else message_reasoning(message)


TEXT_EXTRACTORS: Dict[str, Callable[[Mapping[str, Any]], Optional[str]]] = {
    RESPONSES_ENDPOINT: responses_text,
    CHAT_COMPLETIONS_ENDPOINT: chat_text,
}

FINISH_REASON_EXTRACTORS: Dict[str, Callable[[Mapping[str, Any]], Optional[str]]] = {
    RESPONSES_ENDPOINT: responses_finish_reason,
    CHAT_COMPLETIONS_ENDPOINT: chat_finish_reason,
}

REASONING_EXTRACTORS: Dict[str, Callable[[Mapping[str, Any]], Optional[str]]] = {
    RESPONSES_ENDPOINT: responses_reasoning,
    CHAT_COMPLETIONS_ENDPOINT: chat_reasoning,
}


def prompt_of_body(body: Mapping[str, Any], endpoint: str) -> Dict[str, str]:
    """Return the ``{system_prompt, user_prompt}`` entry a request body was built from."""
    system_role = "developer" if endpoint == RESPONSES_ENDPOINT else "system"
    turns = body.get("input") if endpoint == RESPONSES_ENDPOINT else body.get("messages")
    entry = {"system_prompt": "", "user_prompt": ""}
    for turn in turns or []:
        if turn.get("role") == system_role:
            entry["system_prompt"] = turn["content"]
        elif turn.get("role") == "user":
            entry["user_prompt"] = turn["content"]
    return entry


# Marks a work directory whose batch ended without completing and was collected.
PARTIAL_FNAME = "partial.json"
PARTIAL_SCHEMA_VERSION = 1


def billed_body(record: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the response body one result record bills for, or None when the request failed."""
    response = record.get("response") or {}
    status_code = response.get("status_code")
    if record.get("error") or (status_code is not None and int(status_code) != 200):
        return None
    return response.get("body") or {}


def outcome_from_record(
    record: Mapping[str, Any], endpoint: str
) -> tuple[BatchOutcome, Optional[Mapping[str, Any]]]:
    """Return one result record's outcome and the body to bill for (None when the request failed); a billed Responses
    body whose status is not an answer's is a failed request."""
    body = billed_body(record)
    if body is None:
        status_code = (record.get("response") or {}).get("status_code")
        return (
            BatchOutcome(error=str(record.get("error") or f"status {status_code}")),
            None,
        )
    failure = responses_failure(body) if endpoint == RESPONSES_ENDPOINT else None
    if failure is not None:
        return BatchOutcome(error=failure), body
    return (
        BatchOutcome(
            text=TEXT_EXTRACTORS[endpoint](body),
            finish_reason=FINISH_REASON_EXTRACTORS[endpoint](body),
            reasoning_summaries=tuple(reasoning_summaries(body)),
            reasoning=REASONING_EXTRACTORS[endpoint](body),
        ),
        body,
    )


def sha256_bytes(payload: bytes) -> str:
    """Return the hex digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def request_line(request: BatchRequest, endpoint: str) -> bytes:
    """Serialize one request as a line of a batch input file; the bytes are pinned by a test."""
    payload = {
        "custom_id": request.custom_id,
        "method": "POST",
        "url": endpoint,
        "body": request.body,
    }
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def fingerprint(requests: Sequence[BatchRequest], endpoint: str) -> str:
    """Return the content address of a request set: the same bytes share one batch, any change buys a new one."""
    digest = hashlib.sha256()
    digest.update(endpoint.encode("utf-8"))
    digest.update(b"\0")
    for request in requests:
        digest.update(request_line(request, endpoint))
    return digest.hexdigest()[:32]


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON via a temp file and rename: a truncated state file would lose the record of a paid batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # The pid keeps jobs sharing a batch root from racing on one temp path.
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` via a temp file and rename, so a reader sees the old bytes or the new."""
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def private_temporary(path: Path) -> Path:
    """Return a temporary name beside ``path`` that no other writer uses, for a download renamed into place."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def write_json_once(path: Path, value: Any) -> bool:
    """Write JSON to ``path`` only if no file is there, whole or not at all; return True if this call wrote it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    except OSError:  # no hard links on this filesystem (EPERM, ENOTSUP)
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
    finally:
        temporary.unlink()
    return True


_SPEND_CONFIRMED: ContextVar[bool] = ContextVar("spend_confirmed", default=False)


@contextmanager
def spend_confirmed() -> Iterator[None]:
    """Confirm the spend of every billable call made inside the block, in this thread or task alone."""
    token = _SPEND_CONFIRMED.set(True)
    try:
        yield
    finally:
        _SPEND_CONFIRMED.reset(token)


def config_confirms_spend(cfg: Mapping[str, Any]) -> bool:
    """Return a config's ``confirm_spend`` setting, which is true or false and nothing else."""
    value = cfg.get("confirm_spend", False)
    if value is True or value is False:
        return value
    raise BatchSpendNotConfirmed(f"confirm_spend is true or false, got {value!r}")


def spend_is_confirmed() -> bool:
    """Whether the calling context confirmed billable calls (:func:`spend_confirmed`)."""
    return _SPEND_CONFIRMED.get()


def require_spend_confirmation() -> None:
    """Refuse a billable call unless the calling context confirms it; call just before creating a batch."""
    if not spend_is_confirmed():
        raise BatchSpendNotConfirmed(
            "a paid judge's call spends money; pass --yes-spend (ruleforge jobs launch: "
            "-- +confirm_spend=true; in Python: with batch_api.spend_confirmed()) to confirm. Preparing and "
            "inspecting requests needs no confirmation."
        )


class BatchAlreadyClaimed(BatchExecutionError):
    """Raised when another process holds the creation lock for this batch."""


@contextmanager
def claim_for_creation(directory: Path) -> Iterator[None]:
    """Hold an exclusive claim on a work directory while batches are created, so two jobs cannot buy one batch twice; a
    crashed holder's claim file is removed by hand."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "creating.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise BatchAlreadyClaimed(
            f"another process holds {lock} and may be creating this batch right "
            f"now; refusing to create a second one. If no such process is "
            f"running, it died mid-submit -- check the provider's dashboard for "
            f"an unrecorded batch, then delete the lock."
        ) from None
    try:
        os.write(handle, str(os.getpid()).encode("utf-8"))
        os.close(handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


# A request's reasoning summaries beside its answer: a list, ``[]`` for none returned, absent when none were asked for.
REASONING_SUMMARIES_KEY = "reasoning_summaries"

# A request's full reasoning text beside its answer, absent when none came back. Encrypted reasoning is never stored.
REASONING_KEY = "reasoning"

REASONING_FIELDS = (REASONING_SUMMARIES_KEY, REASONING_KEY)


def reasoning_fields(detail: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the reasoning fields a detail dict carries."""
    return {key: detail[key] for key in REASONING_FIELDS if key in detail}


def responses_summaries_requested(
    endpoint: str, reasoning_effort: Optional[str], reasoning_summary: Optional[str]
) -> bool:
    """Return whether a built request asks for a reasoning summary: the responses endpoint, beside an effort."""
    return (
        endpoint == RESPONSES_ENDPOINT
        and reasoning_effort is not None
        and reasoning_summary is not None
    )


def summary_fields(outcome: BatchOutcome, requested: bool) -> Dict[str, Any]:
    """Return a detail dict's reasoning entries for one outcome."""
    return {
        **({REASONING_SUMMARIES_KEY: list(outcome.reasoning_summaries)} if requested else {}),
        **({REASONING_KEY: outcome.reasoning} if outcome.reasoning is not None else {}),
    }


def write_reasoning_summaries(
    directory: Path, outcomes: Mapping[str, BatchOutcome]
) -> Optional[Path]:
    """Write the batch's reasoning summaries, keyed by custom id; return the path, or None if the provider returned
    none."""
    summaries = {
        custom_id: outcome.reasoning_summaries
        for custom_id, outcome in outcomes.items()
        if outcome.reasoning_summaries
    }
    if not summaries:
        return None
    path = directory / "reasoning_summaries.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for custom_id in sorted(summaries):
            handle.write(
                json.dumps(
                    {"custom_id": custom_id, "summaries": list(summaries[custom_id])},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


_UNSET = object()


class ResumeGuard:
    """Makes a retry after a failed batch re-enter that batch instead of buying another: the seed is part of its
    address, so a raising call's seed is replayed on the next; after unusable answers a fresh seed re-asks."""

    def __init__(self) -> None:
        self._seed: Any = _UNSET

    def seed_for(self, seed: Optional[int]) -> Optional[int]:
        """Return the seed to address this attempt with."""
        return seed if self._seed is _UNSET else self._seed

    def arm(self, seed: Optional[int]) -> None:
        """Remember the seed a call is about to use."""
        self._seed = seed

    def clear(self) -> None:
        """Forget it: the batch returned, so the next retry re-asks."""
        self._seed = _UNSET


def reshape_outcomes(
    outcomes: Mapping[str, BatchOutcome],
    *,
    num_prompts: int,
    num_responses: int,
    seed: Optional[int],
    namespace: Optional[str] = None,
) -> List[List[BatchOutcome]]:
    """Index outcomes by id into the prompts x candidates grid; a cell with no result gets an empty outcome."""
    return [
        [
            outcomes.get(
                batch_custom_id(prompt_index, candidate_index, seed, namespace),
                BatchOutcome(),
            )
            for candidate_index in range(num_responses)
        ]
        for prompt_index in range(num_prompts)
    ]


def _request_input_text(body: Mapping[str, Any], endpoint: str) -> str:
    parts = body.get("messages") if endpoint == CHAT_COMPLETIONS_ENDPOINT else body.get("input")
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    chunks: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            chunks.append(str(part))
            continue
        content = part.get("content", "")
        if isinstance(content, list):
            chunks.append(
                " ".join(
                    piece.get("text", "") if isinstance(piece, dict) else str(piece)
                    for piece in content
                )
            )
        else:
            chunks.append(str(content))
    return "\n".join(chunks)


def estimate_enqueued_tokens(requests: Sequence[BatchRequest], *, endpoint: str) -> int:
    """Estimate a request set's enqueued input tokens with ``o200k_base``; set a windowing budget below the real
    ceiling."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")
    return sum(len(encoding.encode(_request_input_text(request.body, endpoint))) for request in requests)


def track_usage(tracker: Any | None, usage: BatchUsage) -> None:
    """Add a finished batch's input and output tokens to ``tracker``, if one is set."""
    if tracker is not None:
        tracker.add(
            TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            )
        )


def blocking_generators(
    run: Callable[[List[Dict[str, str]], float, int, int, Optional[int]], List[List[BatchOutcome]]],
    *,
    summaries_requested: bool = False,
) -> tuple[
    Callable[[List[Dict[str, str]], float, int, int, Optional[int]], List[List[str]]],
    Callable[[List[Dict[str, str]], float, int, int, Optional[int]], List[List[Dict[str, Any]]]],
]:
    """Return ``(_batch_generate, _batch_generate_with_details)`` over a transport's blocking ``run``."""

    def _batch_generate(
        prompts: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        num_responses: int,
        seed: Optional[int],
    ) -> List[List[str]]:
        return [
            [outcome.text or "" for outcome in candidates]
            for candidates in run(prompts, temperature, max_new_tokens, num_responses, seed)
        ]

    def _batch_generate_with_details(
        prompts: List[Dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        num_responses: int,
        seed: Optional[int],
    ) -> List[List[Dict[str, Any]]]:
        return [
            [
                {
                    "raw_response": outcome.text or "",
                    "completion": outcome.text or "",
                    "finish_reason": outcome.finish_reason,
                    **summary_fields(outcome, summaries_requested),
                }
                for outcome in candidates
            ]
            for candidates in run(prompts, temperature, max_new_tokens, num_responses, seed)
        ]

    return _batch_generate, _batch_generate_with_details


def block_details_from_outcomes(
    outcomes: Mapping[str, "BatchOutcome"],
    usage: Any,
    requests: Sequence[BatchRequest],
    *,
    namespace: str,
    provider: str,
    tracker: Any | None,
    workdir: str,
    summaries_requested: bool = False,
) -> List[Dict[str, Any]]:
    """Record usage and return one detail dict per prompt from a completed block's outcomes."""
    track_usage(tracker, usage)
    grid = reshape_outcomes(
        outcomes, num_prompts=len(requests), num_responses=1, seed=None, namespace=namespace
    )
    log_batch_completion(grid, provider)
    return [
        {
            "raw_response": candidates[0].text,
            "completion": candidates[0].text,
            "finish_reason": candidates[0].finish_reason,
            "error": candidates[0].error,
            "provider_request_ref": requests[index].custom_id,
            "request_block_ref": workdir,
            **summary_fields(candidates[0], summaries_requested),
        }
        for index, candidates in enumerate(grid)
    ]


def collect_block_details(
    runner: Any,
    requests: Sequence[BatchRequest],
    *,
    namespace: str,
    provider: str,
    tracker: Any | None,
    summaries_requested: bool = False,
) -> List[Dict[str, Any]]:
    """Run one request block through a batch runner and return per-prompt details."""
    outcomes, usage = runner.run(requests)
    return block_details_from_outcomes(
        outcomes,
        usage,
        requests,
        namespace=namespace,
        provider=provider,
        tracker=tracker,
        workdir=str(runner.workdir(requests)),
        summaries_requested=summaries_requested,
    )


# 4xx codes that do not say whether a create made its batch. Any other 4xx means no batch exists.
AMBIGUOUS_CLIENT_ERROR_CODES = frozenset({408, 409, 499})


def create_was_refused(error: BaseException) -> bool:
    """Return whether a failed ``batches.create`` certainly made no batch, so it may be retried."""
    status = getattr(error, "status_code", None)
    return (isinstance(status, int) and not isinstance(status, bool) and 400 <= status < 500
            and status not in AMBIGUOUS_CLIENT_ERROR_CODES)


def utc_now() -> datetime:
    """Return the timezone-aware wall clock."""
    return datetime.now(timezone.utc)


def is_truncated(finish_reason: Optional[str]) -> bool:
    """Return whether a generation stopped at the output ceiling (responses: ``incomplete``; chat: ``length``)."""
    reason = finish_reason or ""
    return reason in ("length", "MAX_TOKENS", "max_tokens") or reason.startswith("incomplete")


def log_batch_completion(
    outcome_grid: Sequence[Sequence[BatchOutcome]], provider: str
) -> None:
    """Report how much of a finished batch is usable, and how much is partial."""
    flat = [outcome for row in outcome_grid for outcome in row]
    answered = sum(1 for outcome in flat if outcome.text is not None)
    truncated = sum(1 for outcome in flat if is_truncated(outcome.finish_reason))
    if truncated:
        logger.warning(
            "%d %s response(s) stopped before finishing; their text is partial, "
            "not short",
            truncated,
            provider,
        )
    logger.info("%s batch complete: %d/%d returned text", provider, answered, len(flat))
