from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Mapping, Any, Callable, Dict, List, Optional, Sequence

from typing import TYPE_CHECKING

from ruleforge.core.errors import RuleForgeError, TracebackError
from ruleforge.judges.backend import (
    ANOTHER_SERVER,
    DEFAULT_MAX_NEW_TOKENS,
    STRIP_FAILED_KEY,
    batch_invariant_enabled,
    canonical_serving_dtype,
    precision_setting,
)
from ruleforge.judges.batch_api import (
    FATAL_PROVIDER_ERRORS,
    REASONING_KEY,
    REASONING_SUMMARIES_KEY,
    RESPONSES_ENDPOINT,
    build_responses_body,
    reasoning_summaries,
    responses_failure,
    responses_reasoning,
    responses_summaries_requested,
)
from ruleforge.judges.token_usage import TokenUsage
from ruleforge.judges.tokenizer import check_tokenizer_fidelity, has_tokenizer_json

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from vllm import LLM

logger = logging.getLogger(__name__)

# One provider call with no retry, returning a detail dict: ``raw_response``, ``completion``, ``finish_reason``,
# ``error``, and any reasoning the provider showed (``reasoning_summaries``, ``reasoning``).
DetailCallFn = Callable[[str, str, float, int, Optional[int]], Dict[str, Any]]
#                        sys   usr   temp  max_tok seed -> detail

try:  # pragma: no cover - depends on the optional vllm group
    import openai_harmony as _openai_harmony
except ImportError:  # pragma: no cover - depends on the optional vllm group
    _openai_harmony = None

_harmony_lock = threading.Lock()
_harmony_loaded: Any = None


def harmony_encoding() -> Any:
    """Return gpt-oss's harmony encoding, loaded on first call (it may download the vocab); raises if harmony is
    missing."""
    global _harmony_loaded
    if _openai_harmony is None:
        raise NotImplementedError("openai_harmony is not installed")
    with _harmony_lock:
        if _harmony_loaded is None:
            _harmony_loaded = _openai_harmony.load_harmony_encoding(
                _openai_harmony.HarmonyEncodingName.HARMONY_GPT_OSS
            )
        return _harmony_loaded


def _strip_harmony_thinking(output: Dict[str, Any]) -> str:
    encoding = harmony_encoding()
    token_ids = output.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("Missing token_ids for harmony stripping")
    try:
        messages = encoding.parse_messages_from_completion_tokens(
            token_ids, _openai_harmony.Role.ASSISTANT
        )
    except Exception as exc:
        raise ValueError("Unable to decode assistant messages from token IDs") from exc
    final_contents = [
        message.content
        for message in messages
        if getattr(message, "channel", None) == "final"
    ]
    if final_contents:
        return final_contents[-1][-1].text
    raise ValueError("No final assistant messages decoded from token IDs")


def _strip_think_tags(
    output: Dict[str, Any], strip_before_orphaned_closing: bool = True
) -> str:
    text = output.get("text")
    if not isinstance(text, str):
        raise ValueError("Missing 'text' field for think-tag stripping")
    _THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
    stripped = _THINK_TAG_PATTERN.sub("", text)
    if strip_before_orphaned_closing and "</think>" in stripped:
        last_idx = stripped.rfind("</think>")
        stripped = stripped[last_idx + len("</think>") :]
    stripped = stripped.strip()
    if not stripped:
        raise ValueError("No content remaining after stripping thinking tags")
    return stripped


def _strip_gemma4_thinking(output: Dict[str, Any]) -> str:
    text = output.get("text")
    _GEMMA4_THINKING_PATTERN = re.compile(
        r"<\|channel>thought.*?<channel\|>", re.DOTALL
    )
    if not isinstance(text, str):
        raise ValueError("Missing 'text' field for gemma4 thinking stripping")
    stripped = _GEMMA4_THINKING_PATTERN.sub("", text)
    # Handle orphaned closing tag (model omitted opening <|channel>)
    last_idx = stripped.rfind("<channel|>")
    if last_idx != -1:
        stripped = stripped[last_idx + len("<channel|>") :]
    stripped = stripped.strip()
    if not stripped:
        raise ValueError("No content remaining after stripping Gemma 4 thinking tags")
    return stripped


# Muse Glimmer's channel framing: ``<|start|>assistant to=<recipient><|message|>`` and a body ended by ``<|eom|>`` (more
# follows) or ``<|eot|>`` (a stop id vLLM leaves out of the text). ``self`` is reasoning, ``user`` the answer.
# The framing tokens' ids. A generation's ``token_ids`` show which were really emitted; text alone cannot tell.
GLIMMER_SPECIAL_IDS = {"<|start|>": 200022, "<|message|>": 200023, "<|eom|>": 200007, "<|eot|>": 200008}
_GLIMMER_FIRST_HEADER = re.compile(r"\s*(?:<\|start\|>assistant\s*)?to=([^\s<]+)<\|message\|>")
_GLIMMER_NEXT_HEADER = re.compile(r"\s*<\|start\|>assistant\s*to=([^\s<]+)<\|message\|>")
_GLIMMER_MARKER = re.compile("|".join(re.escape(token) for token in GLIMMER_SPECIAL_IDS))


def _glimmer_markers_real(text: str, token_ids: Sequence[int]) -> bool:
    emitted_ids = Counter(token_ids)
    for token, token_id in GLIMMER_SPECIAL_IDS.items():
        written, emitted = text.count(token), emitted_ids[token_id]
        if written > emitted or (written != emitted and token != "<|eot|>"):
            return False
    return True


def muse_glimmer_answer_span(text: str, token_ids: Optional[Sequence[int]] = None) -> Optional[tuple[int, int]]:
    """Return the ``(start, end)`` of the last ``to=user`` message's body, or None when there is none or the framing is
    unclear (framing written as ordinary tokens, per ``token_ids``)."""
    if token_ids is not None and not _glimmer_markers_real(text, token_ids):
        return None
    answer, position, header_pattern = None, 0, _GLIMMER_FIRST_HEADER
    while True:
        header = header_pattern.match(text, position)
        if header is None:
            return None
        end = _GLIMMER_MARKER.search(text, header.end())
        if header.group(1) == "user":
            answer = (header.end(), len(text) if end is None else end.start())
        if end is None:
            return answer
        marker, header_pattern = end.group(0), _GLIMMER_NEXT_HEADER
        if marker == "<|eom|>":
            position = end.end()
            if not text[position:].strip():  # the turn stopped after a message (cut off before the next)
                return answer
        elif marker == "<|start|>":
            position = end.start()
        elif marker == "<|eot|>":
            return answer if not text[end.end():].strip() else None
        else:  # <|message|> inside a body
            return None


def _strip_muse_glimmer_thinking(output: Dict[str, Any]) -> str:
    """Return the last ``to=user`` message's body stripped, or an empty answer so that the validator asks again."""
    text = output.get("text")
    if not isinstance(text, str):
        return ""
    try:  # vLLM's ids, a list; anything else that is not a sequence of ids is no answer
        token_ids = None if output.get("token_ids") is None else [int(i) for i in output["token_ids"]]
    except (TypeError, ValueError):
        return ""
    span = muse_glimmer_answer_span(text, token_ids)
    return "" if span is None else text[span[0]:span[1]].strip()


# vLLM's own SamplingParams defaults, with no truncation. vLLM applies a model's generation_config.json only when no
# SamplingParams is passed, so every explicit SamplingParams starts from these.
UNTRUNCATED_SAMPLING: Dict[str, float | int] = {
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}
TRUNCATION_KEYS = tuple(UNTRUNCATED_SAMPLING)


def validate_default_truncation(values: Mapping[str, Any] | None) -> Dict[str, float | int]:
    """Return a model config's ``default_truncation`` checked (only ``TRUNCATION_KEYS``, finite) and in canonical
    types."""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(
            f"default_truncation must be a mapping over {list(TRUNCATION_KEYS)}, got "
            f"{type(values).__name__}"
        )
    unknown = sorted(str(key) for key in values if key not in UNTRUNCATED_SAMPLING)
    if unknown:
        raise ValueError(
            f"default_truncation keys {unknown} are not sampler truncation keys; only "
            f"{list(TRUNCATION_KEYS)} are applied (a presence or frequency penalty, or a "
            "temperature, is not part of the truncation a campaign samples with)"
        )
    canonical: Dict[str, float | int] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"default_truncation.{key} must be a finite number, got {value!r}")
        if isinstance(UNTRUNCATED_SAMPLING[key], int):
            if not float(value).is_integer():
                raise ValueError(f"default_truncation.{key} must be an integer, got {value!r}")
            canonical[key] = int(value)
        else:
            canonical[key] = float(value)
    return canonical


def model_default_truncation(
    model_path: str | Path, default_truncation: Mapping[str, Any] | None = None
) -> Dict[str, float | int]:
    """Return the truncation a model ships in ``generation_config.json``, with ``default_truncation`` filling missing
    keys."""
    fallback = validate_default_truncation(default_truncation)
    path = Path(model_path) / "generation_config.json"
    shipped: Dict[str, float | int] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        shipped = {key: payload[key] for key in TRUNCATION_KEYS if payload.get(key) is not None}
    for key in sorted(fallback.keys() & shipped.keys()):
        if fallback[key] != shipped[key]:
            logger.warning(
                "%s ships %s=%r; the model config's default_truncation %s=%r is not applied",
                path, key, shipped[key], key, fallback[key],
            )
    return {**fallback, **shipped}


_MODEL_STRIP_FN: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "openai_gptoss": _strip_harmony_thinking,
    "qwen3": _strip_think_tags,
    "gemma4": _strip_gemma4_thinking,
    "olmo3": _strip_think_tags,
    "muse_glimmer": _strip_muse_glimmer_thinking,
}

# The ``strip_thinking_model`` values a local config may name.
THINKING_STRIPPERS = tuple(sorted(_MODEL_STRIP_FN))

# The ``strip_thinking_model`` values whose thinking closes with a literal ``</think>``.
THINK_TAG_MODELS = frozenset(name for name, strip in _MODEL_STRIP_FN.items() if strip is _strip_think_tags)


# How a thought opens in each family that writes it in its text; harmony fails on a turn with no final message, and Muse
# Glimmer's stripper returns no answer.
_THOUGHT_OPENINGS = {"gemma4": "<|channel>thought", **dict.fromkeys(THINK_TAG_MODELS, "<think>")}


def _opens_a_thought(answer: str, strip_thinking_model: str | None) -> bool:
    """Return whether a stripped answer still opens its family's thought, one never closed: ``<think>`` only where the
    answer opens (text can write it), Gemma's special-token opening anywhere."""
    opening = _THOUGHT_OPENINGS.get(strip_thinking_model)
    return opening is not None and (
        answer.startswith(opening) or (strip_thinking_model == "gemma4" and opening in answer))


def holds_a_thought(answer: str, strip_thinking_model: str | None) -> bool:
    """Return whether an answer still holds its family's thought once stripped, or holds nothing else."""
    return strip_thinking_model in _THOUGHT_OPENINGS and _extract_text_and_strip_status(
        {"text": answer}, strip_thinking_model)[1]


def _extract_text_and_strip_status(
    candidate: Dict[str, Any],
    strip_thinking_model: str | None,
) -> tuple[str, bool]:
    """Return the candidate's answer and whether a configured stripper could not run or left an unclosed thought, which
    leaves no answer: the thought can end on a line that reads as a judgment."""
    if not strip_thinking_model:
        raw = candidate.get("text", "")
        return (raw.strip() if isinstance(raw, str) else ""), False
    strip_fn = _MODEL_STRIP_FN.get(strip_thinking_model)
    if strip_fn is None:
        logger.warning(f"No thinking-token stripper for model '{strip_thinking_model}', so no answer is read")
        return "", True
    try:
        answer = strip_fn(candidate)
    except (ValueError, NotImplementedError) as e:
        logger.warning(f"Failed to strip thinking tokens: {e}")
        return "", True
    if _opens_a_thought(answer, strip_thinking_model):
        logger.warning("The answer holds a thought that never closed, so no answer is read")
        return "", True
    return answer, False


def parse_json_from_response(response: str) -> Dict[str, Any] | None:
    if not response:
        return None
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return None


def _generate_with_vllm(
    llm: LLM,
    prompts: Sequence[str | Mapping[str, Any]],
    temperature: float,
    max_new_tokens: int,
    num_responses: int,
    seed: int | None | Sequence[int | None],
    skip_special_tokens: bool = False,
    lora_request: Any | None = None,
    truncation: Mapping[str, float | int] | None = None,
) -> List[List[Dict[str, Any]]]:
    """Run one vLLM generate call and return its candidates; ``seed`` is one seed or one per prompt."""
    from vllm import SamplingParams

    effective = {**UNTRUNCATED_SAMPLING, **dict(truncation or {})}

    def _params(prompt_seed: int | None) -> Any:
        return SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=num_responses,
            seed=prompt_seed,
            skip_special_tokens=skip_special_tokens,
            **effective,
        )

    if isinstance(seed, (list, tuple)):
        if len(seed) != len(prompts):
            raise ValueError(
                f"per-prompt seeds must align with prompts, got {len(seed)} seeds "
                f"for {len(prompts)} prompts"
            )
        sampling_params: Any = [_params(prompt_seed) for prompt_seed in seed]
    else:
        sampling_params = _params(seed)
    request_outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        **({"lora_request": lora_request} if lora_request is not None else {}),
    )
    if not request_outputs:
        return [[] for _ in prompts]

    normalized: List[List[Dict[str, Any]]] = []
    for request in request_outputs:
        prompt_outputs: List[Dict[str, Any]] = []
        for candidate in getattr(request, "outputs", []):
            prompt_outputs.append(
                {
                    "text": candidate.text,
                    "token_ids": getattr(candidate, "token_ids", None),
                    "finish_reason": getattr(candidate, "finish_reason", None),
                }
            )
        normalized.append(prompt_outputs)
    return normalized


def format_prompt_for_generation(
    tokenizer: AutoTokenizer,
    prompt_entry: Dict[str, str],
    *,
    gpt_oss_reasoning_level: str | None = None,
) -> str:
    """Render one prompt entry as it is sent to vLLM, as one user turn if the chat template rejects a system role."""
    system_prompt = prompt_entry.get("system_prompt", "")
    user_prompt = prompt_entry.get("user_prompt", "")
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        if gpt_oss_reasoning_level:
            return tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=gpt_oss_reasoning_level,
            )
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        merged = "\n".join(part for part in (system_prompt, user_prompt) if part)
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            tokenize=False,
            add_generation_prompt=True,
        )


def prompt_token_ids(tokenizer: Any, rendered: str) -> List[int]:
    """Return the token ids of a rendered prompt, adding no special tokens (the chat template already wrote them)."""
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def call_local_model(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt_batch: List[Dict[str, str]],
    temperature: float = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    gpt_oss_reasoning_level: str | None = None,
    num_responses: int = 1,
    seed: int | None | Sequence[int | None] = None,
    strip_thinking_model: str | None = None,
    lora_request: Any | None = None,
) -> List[List[str]]:
    """Format prompts, generate via vLLM and strip thinking tokens; return clean strings."""
    detailed = call_local_model_with_details(
        llm,
        tokenizer,
        prompt_batch,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        gpt_oss_reasoning_level=gpt_oss_reasoning_level,
        num_responses=num_responses,
        seed=seed,
        strip_thinking_model=strip_thinking_model,
        lora_request=lora_request,
    )
    return [
        [candidate["completion"] for candidate in prompt_outputs]
        for prompt_outputs in detailed
    ]


def call_local_model_with_details(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt_batch: List[Dict[str, str]],
    temperature: float = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    gpt_oss_reasoning_level: str | None = None,
    num_responses: int = 1,
    seed: int | None | Sequence[int | None] = None,
    strip_thinking_model: str | None = None,
    lora_request: Any | None = None,
    truncation: Mapping[str, float | int] | None = None,
) -> List[List[Dict[str, Any]]]:
    """Return local completions as detail dicts (``completion`` the stripped text, ``raw_response`` the text before it);
    a failed strip or unclosed thought gives an empty ``completion`` and ``strip_failed: True``."""
    if not prompt_batch:
        raise ValueError("prompt_batch must not be empty")

    formatted = [
        format_prompt_for_generation(
            tokenizer, entry, gpt_oss_reasoning_level=gpt_oss_reasoning_level
        )
        for entry in prompt_batch
    ]
    raw_outputs = _generate_with_vllm(
        llm,
        [{"prompt_token_ids": prompt_token_ids(tokenizer, text)} for text in formatted],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_responses=num_responses,
        seed=seed,
        lora_request=lora_request,
        truncation=truncation,
    )

    def _detail(candidate: Dict[str, Any]) -> Dict[str, Any]:
        completion, strip_failed = _extract_text_and_strip_status(
            candidate, strip_thinking_model
        )
        text = candidate.get("text", "")
        return {
            "raw_response": text if isinstance(text, str) else "",
            "completion": completion,
            "finish_reason": candidate.get("finish_reason"),
            **({STRIP_FAILED_KEY: True} if strip_failed else {}),
        }

    return [
        [_detail(candidate) for candidate in prompt_outputs]
        for prompt_outputs in raw_outputs
    ]


# Bumped when a prompt's wire shape changes (which role carries the text, what a missing turn becomes). Recorded in run
# provenance; a resume across versions raises.
PROMPT_SHAPE_VERSION = 2

# Sent as the user turn when the caller has none: the Messages API needs one and evaluation prompts are system-only.
EMPTY_USER_TURN = "Respond according to the system instruction."

def _answered(
    text: Optional[str],
    finish_reason: Optional[str],
    *,
    summaries: Optional[Sequence[str]] = None,
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    value = text or ""
    return {
        "raw_response": value, "completion": value, "finish_reason": finish_reason, "error": None,
        **({REASONING_SUMMARIES_KEY: list(summaries)} if summaries is not None else {}),
        **({REASONING_KEY: reasoning} if reasoning and reasoning.strip() else {}),
    }


class CallFailed(RuntimeError, TracebackError):
    """A provider call that came back without an answer (failed status, error event, cut-off stream, no choice)."""

    def __init__(self, message: str, usage: Any = None) -> None:
        super().__init__(message)
        self.usage = usage


def _failed(provider: str, model_name: str, exc: BaseException, tracker: Any | None = None, *,
            summaries_requested: bool = False) -> Dict[str, Any]:
    if isinstance(exc, CallFailed) and tracker is not None:
        tracker.add(TokenUsage.from_response(exc.usage))
    logger.warning("%s API error (model=%s): %s", provider, model_name, exc)
    return {"raw_response": "", "completion": "", "finish_reason": None, "error": f"{type(exc).__name__}: {exc}",
            **({REASONING_SUMMARIES_KEY: []} if summaries_requested else {})}


def _warn_seed_ignored_once(state: Dict[str, bool], provider: str, seed: Any) -> None:
    if seed is None or state.get("warned"):
        return
    state["warned"] = True
    logger.warning(
        "%s has no seed parameter; the requested seed %s is ignored. Retries "
        "vary only through sampling, so a run at temperature 0 will repeat "
        "itself rather than explore.",
        provider,
        seed,
    )


def _responses_text(response: Any) -> Optional[str]:
    aggregated = getattr(response, "output_text", None)
    if isinstance(aggregated, str) and aggregated.strip():
        return aggregated.strip()

    texts: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            text = getattr(content, "text", None)
            if isinstance(text, str):
                texts.append(text)
    joined = "".join(texts).strip()
    return joined or None


def _responses_finish_reason(response: Any) -> Optional[str]:
    status = getattr(response, "status", None)
    if status == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        return f"incomplete:{reason}" if isinstance(reason, str) else "incomplete"
    return status if isinstance(status, str) else None


_RESPONSES_TERMINAL_EVENTS = ("response.completed", "response.incomplete")


_RESPONSES_REASONING_EVENTS = {
    "response.reasoning_summary_text.done": "summaries",
    "response.reasoning_text.done": "reasoning",
}


def _stream_responses_detail(
    client: Any, request_kwargs: Dict[str, Any]
) -> tuple[Optional[str], Any, Optional[str], List[str], Optional[str]]:
    text: List[str] = []
    usage = None
    finish_reason = None
    ended = False
    parts: Dict[str, List[str]] = {"summaries": [], "reasoning": []}
    with client.responses.stream(**request_kwargs) as stream:
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str):
                    text.append(delta)
            elif event_type in _RESPONSES_REASONING_EVENTS:
                part = getattr(event, "text", None)
                if isinstance(part, str):
                    parts[_RESPONSES_REASONING_EVENTS[event_type]].append(part)
            elif event_type in _RESPONSES_TERMINAL_EVENTS:
                response = getattr(event, "response", None)
                usage = getattr(response, "usage", None)
                finish_reason = _responses_finish_reason(response)
                ended = True
            elif event_type == "response.failed":
                response = getattr(event, "response", None)
                raise CallFailed(responses_failure(response) or "the stream reported response.failed",
                                 usage=getattr(response, "usage", None))
            elif event_type == "error":
                named = [str(value) for value in (getattr(event, "code", None), getattr(event, "message", None))
                         if value]
                raise CallFailed("the stream sent an error event" + (f" ({': '.join(named)})" if named else ""))
    if not ended:
        raise CallFailed("the stream ended before response.completed or response.incomplete")
    joined = "".join(text).strip()
    reasoning = "\n".join(parts["reasoning"])
    return (joined or None), usage, finish_reason, parts["summaries"], (reasoning if reasoning.strip() else None)


def _make_openai_caller(
    client: Any,
    model_name: str,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    tracker: Any | None = None,
    stream: bool = False,
    send_max_output_tokens: bool = False,
) -> DetailCallFn:
    """Return a DetailCallFn that makes one OpenAI Responses API call, the same request ``openai_batch`` sends."""
    seed_warning: Dict[str, bool] = {}
    summarized = responses_summaries_requested(RESPONSES_ENDPOINT, reasoning_effort, reasoning_summary)

    def _call(
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> Dict[str, Any]:
        _warn_seed_ignored_once(seed_warning, "The OpenAI Responses API", seed)
        request_kwargs = build_responses_body(
            system_prompt,
            user_prompt,
            model=model_name,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            max_output_tokens=max_new_tokens if send_max_output_tokens else None,
        )

        try:
            if stream:
                text, usage, finish_reason, summaries, reasoning = _stream_responses_detail(
                    client, request_kwargs
                )
            else:
                response = client.responses.create(**request_kwargs)
                text, usage = _responses_text(response), getattr(response, "usage", None)
                finish_reason = _responses_finish_reason(response)
                failure = responses_failure(response)
                if failure is not None:
                    raise CallFailed(failure, usage=usage)
                summaries, reasoning = reasoning_summaries(response), responses_reasoning(response)
            if tracker is not None:
                tracker.add(TokenUsage.from_response(usage))
            return _answered(text, finish_reason, summaries=summaries if summarized else None,
                             reasoning=reasoning)
        except FATAL_PROVIDER_ERRORS:
            raise
        except Exception as exc:
            return _failed("OpenAI", model_name, exc, tracker, summaries_requested=summarized)

    return _call


def _make_anthropic_caller(
    client: Any,
    model_name: str,
    reasoning_effort: Optional[str] = None,
    tracker: Any | None = None,
    thinking_display: Optional[str] = None,
) -> DetailCallFn:
    """Return a DetailCallFn that makes one Anthropic Messages API call, the same request and reading as
    ``anthropic_batch``."""
    from ruleforge.judges.anthropic_batch import (
        DISPLAY_SUMMARIZED,
        build_request_params,
        check_temperature,
        display_setting,
        effort_level,
        outcome_from_message,
        result_record,
    )

    effort = effort_level(reasoning_effort)
    display = display_setting(thinking_display)
    summarized = display == DISPLAY_SUMMARIZED
    seed_warning: Dict[str, bool] = {}

    def _call(
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> Dict[str, Any]:
        check_temperature(temperature)
        _warn_seed_ignored_once(seed_warning, "The Anthropic Messages API", seed)
        params = build_request_params(
            system_prompt, user_prompt, model=model_name, max_tokens=max_new_tokens, effort=effort,
            display=display,
        )
        try:
            message = result_record(client.messages.create(**params))
            if tracker is not None:
                tracker.add(TokenUsage.from_response(message.get("usage")))
            outcome = outcome_from_message(message)
            return _answered(outcome.text, outcome.finish_reason,
                             summaries=outcome.reasoning_summaries if summarized else None)
        except FATAL_PROVIDER_ERRORS:
            raise
        except Exception as exc:
            return _failed("Anthropic", model_name, exc, summaries_requested=summarized)

    return _call


def _api_generate_batch_with_details(
    call_fn: DetailCallFn,
    prompt_batch: List[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    num_responses: int,
    seed: int | None | Sequence[int | None],
) -> List[List[Dict[str, Any]]]:
    """Return one detail dict per candidate for any API DetailCallFn; a candidate's seed is offset by its index."""
    per_prompt = isinstance(seed, (list, tuple))
    if per_prompt and len(seed) != len(prompt_batch):
        raise ValueError(
            f"per-prompt seeds must align with prompts, got {len(seed)} seeds "
            f"for {len(prompt_batch)} prompts"
        )
    results: List[List[Dict[str, Any]]] = []
    for prompt_index, entry in enumerate(prompt_batch):
        sys_prompt = entry.get("system_prompt", "")
        usr_prompt = entry.get("user_prompt", "")
        prompt_seed = seed[prompt_index] if per_prompt else seed
        candidates: List[Dict[str, Any]] = []
        for resp_idx in range(num_responses):
            resp_seed = prompt_seed + resp_idx if prompt_seed is not None else None
            candidates.append(
                call_fn(sys_prompt, usr_prompt, temperature, max_new_tokens, resp_seed)
            )
        results.append(candidates)
    return results


def _api_generate_batch(
    call_fn: DetailCallFn,
    prompt_batch: List[Dict[str, str]],
    temperature: float,
    max_new_tokens: int,
    num_responses: int,
    seed: int | None | Sequence[int | None],
) -> List[List[str]]:
    return [
        [detail.get("completion") or "" for detail in candidates]
        for candidates in _api_generate_batch_with_details(
            call_fn, prompt_batch, temperature, max_new_tokens, num_responses, seed
        )
    ]


class _ThinkingTokenizer:
    """Tokenizer proxy that passes the configured ``enable_thinking`` to ``apply_chat_template``, false as well as true:
    a template that thinks unless told otherwise (Qwen3's) ignores a missing argument."""

    def __init__(self, tokenizer, enable_thinking: bool):
        self._tokenizer = tokenizer
        self._enable_thinking = bool(enable_thinking)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", self._enable_thinking)
        return self._tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        # Python looks special methods up on the type, so ``__getattr__`` would not forward this call.
        return self._tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


# Engine arguments ``init_local_model`` sets itself, so ``engine_args`` may not repeat them.
_ENGINE_ARGS_OWNED = frozenset({
    "model", "tensor_parallel_size", "trust_remote_code", "reasoning_parser",
    "speculative_config", "max_model_len", "attention_config", "attention_backend", "disable_log_stats",
    "enable_lora", "max_lora_rank", "max_num_seqs", "language_model_only", "dtype",
})
# Engine arguments that change what a judge serves but no condition identity records; neither ``engine_args`` nor a
# top-level key may carry them.
ENGINE_ARGS_OUTSIDE_IDENTITY = frozenset({
    "kv_cache_dtype", "kv_cache_dtype_skip_layers", "mamba_cache_dtype", "mamba_ssm_cache_dtype",
    "quantization", "quantization_config", "load_format", "model_weights", "hf_overrides", "hf_config_path",
    "config_format", "model_class_overrides", "model_impl", "revision", "code_revision", "tokenizer",
    "tokenizer_revision", "tokenizer_mode", "skip_tokenizer_init", "generation_config",
    "override_generation_config", "logits_processors", "lora_dtype", "linear_backend",
    "enable_bf16x3_router_gemm", "use_fp64_gumbel", "seed",
})


def engine_argument_names() -> frozenset[str]:
    """Return the name of every vLLM engine argument (imports vLLM)."""
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs

    return frozenset(field.name for field in dataclasses.fields(EngineArgs))


def local_engine_kwargs(
    model_path: Path,
    tp_size: int,
    reasoning_parser: str | None = None,
    speculative_method: str | None = None,
    num_speculative_tokens: int | None = None,
    max_model_len: int | None = None,
    attention_backend: str | None = None,
    enable_thinking: bool = False,
    disable_log_stats: bool = True,
    enable_lora: bool = False,
    max_lora_rank: int | None = None,
    max_num_seqs: int | None = None,
    language_model_only: bool = False,
    dtype: str | None = None,
    engine_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the keyword arguments ``init_local_model`` builds ``vllm.LLM`` with, without importing vLLM. Refuses
    settings outside a condition's identity, a negative ``max_model_len`` (the window would depend on placement), and
    float32 under ``VLLM_BATCH_INVARIANT`` (vLLM then runs linear layers at TF32, which rounds the weights)."""
    extra_engine_args = dict(engine_args or {})
    owned = sorted(extra_engine_args.keys() & _ENGINE_ARGS_OWNED)
    if owned:
        route = (
            "from the model config's own keys, e.g. tp_size, attention_backend, max_num_seqs, and the "
            "served adapter"
        )
        if "dtype" in owned:
            route += "; dtype by the model config's dtype key (condition identity: serving_dtype) or a tool's --dtype"
        raise ValueError(
            f"engine_args {owned} are set through init_local_model's own parameters, not engine_args ({route})"
        )
    outside = sorted(extra_engine_args.keys() & ENGINE_ARGS_OUTSIDE_IDENTITY)
    if outside:
        raise ValueError(
            f"engine_args {outside} change the function the judge serves, which the condition identity "
            "does not record, so two draws of one condition could be served differently; plumb a model "
            "config key into the identity for it first (as serving_dtype is for dtype)"
        )
    if max_model_len is not None and int(max_model_len) < 0:
        raise ValueError(
            f"max_model_len={max_model_len}: a negative value lets vLLM fit the context window to the KV-cache "
            "memory left, so which prompts and generations fit would follow placement (gpu_memory_utilization, "
            "tensor_parallel_size), which the stack guard does not compare; set the window, or null for the "
            "model's native one"
        )
    dtype = canonical_serving_dtype(dtype, attention_backend)
    if dtype == "float32":
        _check_float32_precision()
    speculative_config = None
    if speculative_method is not None:
        speculative_config = {"method": speculative_method}
        if num_speculative_tokens is not None:
            speculative_config["num_speculative_tokens"] = num_speculative_tokens
    return {
        "model": model_path,
        "tensor_parallel_size": tp_size,
        "trust_remote_code": True,
        "reasoning_parser": reasoning_parser,
        "speculative_config": speculative_config,
        "max_model_len": max_model_len,
        "attention_config": {"backend": attention_backend} if attention_backend else None,
        # False logs engine throughput periodically, which tells a long decode from a hang.
        "disable_log_stats": disable_log_stats,
        # The ceiling cannot be raised after construction.
        "enable_lora": enable_lora,
        **({"max_lora_rank": max_lora_rank} if max_lora_rank else {}),
        **({"max_num_seqs": int(max_num_seqs)} if max_num_seqs else {}),
        **({"language_model_only": True} if language_model_only else {}),
        **({"dtype": dtype} if dtype is not None else {}),
        **extra_engine_args,
    }


def _check_float32_precision() -> None:
    """Refuse a float32 engine that this process's precision settings would round to TF32 or bf16."""
    if batch_invariant_enabled():
        raise ValueError(
            "a float32 engine with VLLM_BATCH_INVARIANT on: vLLM then runs every unquantized linear layer "
            "through a Triton matmul at TF32, rounding the weights to a 10-bit mantissa (fp32 serving keeps them "
            "unrounded). Unset VLLM_BATCH_INVARIANT or set it to 0"
        )
    gemm = precision_setting("VLLM_FLOAT32_MATMUL_PRECISION", os.environ.get("VLLM_FLOAT32_MATMUL_PRECISION"))
    if gemm != "highest":
        raise ValueError(
            f"a float32 engine with VLLM_FLOAT32_MATMUL_PRECISION={gemm}: the vLLM worker passes it to "
            "torch.set_float32_matmul_precision, so every fp32 GEMM runs at reduced precision (TF32 under "
            "high, bf16 under medium), rounding the weights (fp32 serving keeps them unrounded). Unset it or set it "
            "to highest"
        )


def init_local_model(
    model_path: Path,
    tp_size: int,
    reasoning_parser: str | None = None,
    speculative_method: str | None = None,
    num_speculative_tokens: int | None = None,
    max_model_len: int | None = None,
    attention_backend: str | None = None,
    enable_thinking: bool = False,
    disable_log_stats: bool = True,
    enable_lora: bool = False,
    max_lora_rank: int | None = None,
    max_num_seqs: int | None = None,
    language_model_only: bool = False,
    dtype: str | None = None,
    engine_args: Mapping[str, Any] | None = None,
) -> tuple[LLM, AutoTokenizer]:
    """Build the vLLM engine and the tokenizer prompts are rendered with, refusing a tokenizer the checkpoint disagrees
    with; ``max_num_seqs`` caps concurrent sequences (the warm-up sends that many, which large vocabularies can run out
    of memory on); ``language_model_only`` serves a multimodal checkpoint as text."""
    llm_kwargs = local_engine_kwargs(
        model_path=model_path,
        tp_size=tp_size,
        reasoning_parser=reasoning_parser,
        speculative_method=speculative_method,
        num_speculative_tokens=num_speculative_tokens,
        max_model_len=max_model_len,
        attention_backend=attention_backend,
        enable_thinking=enable_thinking,
        disable_log_stats=disable_log_stats,
        enable_lora=enable_lora,
        max_lora_rank=max_lora_rank,
        max_num_seqs=max_num_seqs,
        language_model_only=language_model_only,
        dtype=dtype,
        engine_args=engine_args,
    )
    from transformers import AutoTokenizer
    from vllm import LLM

    logger.info(f"Initializing local model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        check_tokenizer_fidelity(tokenizer, model_path, where="HF")
        tokenizer = _ThinkingTokenizer(tokenizer, enable_thinking)
        if attention_backend:
            logger.info("Using attention backend: %s", attention_backend)

        llm = LLM(**llm_kwargs)
        if has_tokenizer_json(model_path):
            check_tokenizer_fidelity(llm.get_tokenizer(), model_path, where="vLLM engine")
        logger.info("Local model initialized.")
    except RuleForgeError:
        raise  # a refusal (a tokenizer that is not the checkpoint's) reaches the caller as itself
    except Exception as e:
        logger.error(f"Error initializing local model: {e}")
        raise RuntimeError(f"Failed to initialize local model: {e}") from e
    return llm, tokenizer


# Variables the openai SDK reads on its own, which would send calls to a server other than the recorded one.
OPENAI_REDIRECT_ENVS = ("OPENAI_BASE_URL",)


def refuse_ambient_sdk_envs(names: Sequence[str], *, sdk: str, server: str) -> None:
    """Refuse while any of ``names`` is set: the SDK would follow it and the calls would not go to ``server``."""
    ambient = [name for name in names if (os.environ.get(name) or "").strip()]
    if ambient:
        raise ValueError(f"{' and '.join(ambient)} set: the {sdk} SDK would use it for every call, so the requests "
                         f"would not go to {server}, the server the condition records; unset it for this judge "
                         f"({ANOTHER_SERVER})")


def init_openai_client():
    """Return an OpenAI client keyed from ``OPENAI_API_KEY``, refused while ``OPENAI_BASE_URL`` is set."""
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    refuse_ambient_sdk_envs(OPENAI_REDIRECT_ENVS, sdk="openai", server="api.openai.com")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return client
    except Exception:
        # Raised ``from None``: an SDK that echoes its arguments into the exception would put the key in a log.
        logger.error("Failed to initialize OpenAI client")
        raise RuntimeError("Failed to initialize OpenAI client") from None


def init_anthropic_client():
    """Return an Anthropic client keyed from ``ANTHROPIC_API_KEY``, refused while a redirecting variable is set."""
    from ruleforge.judges.anthropic_batch import default_client

    return default_client()


def cleanup_vllm(llm: "LLM", raise_on_error: bool = True) -> None:
    """Shut down a vLLM engine; ``raise_on_error=False`` keeps a failure from masking an in-flight exception."""
    import gc
    import time

    logger.info("Cleaning up vLLM model...")
    try:
        # Without this, tensor-parallel workers survive and the process hangs at exit.
        llm.llm_engine.engine_core.shutdown()
        del llm
        gc.collect()
        logger.info("Waiting for vLLM cleanup to complete...")
        time.sleep(10)
        logger.info("vLLM model cleanup complete")
    except Exception as e:
        logger.error(f"Error during vLLM cleanup: {e}")
        if raise_on_error:
            raise SystemExit(1) from e
