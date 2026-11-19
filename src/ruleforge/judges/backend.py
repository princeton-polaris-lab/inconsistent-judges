from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import random
import time
from typing import (
    Mapping,
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from omegaconf import DictConfig, OmegaConf

from ruleforge.core.digests import canonical_json
from ruleforge.judges.batch_api import (
    FATAL_PROVIDER_ERRORS,
    REASONING_FIELDS,
    REASONING_KEY,
    REASONING_SUMMARIES_KEY,
    is_truncated,
)
from ruleforge.prompts.parse import LABELS, holds_an_open_thought, read_label
from ruleforge.judges.token_usage import TokenUsageTracker

if TYPE_CHECKING:
    from vllm import LLM

    from ruleforge.judges.base import JudgeIdentity, JudgeSpec

logger = logging.getLogger(__name__)

T = TypeVar("T")


# What a backend does with a generation seed. ``local_deterministic``: reaches the sampler; vLLM is not batch-invariant,
# so a prompt replays only beside the same call's prompts. ``provider_best_effort``: accepted, no replay promised.
# ``unsupported``: no seed sent; a replicate is identified by its decoder draw and a fresh call.
SEED_UNSUPPORTED = "unsupported"
SEED_LOCAL_DETERMINISTIC = "local_deterministic"
SEED_PROVIDER_BEST_EFFORT = "provider_best_effort"
SEED_CAPABILITIES = (
    SEED_UNSUPPORTED,
    SEED_LOCAL_DETERMINISTIC,
    SEED_PROVIDER_BEST_EFFORT,
)

BATCH_MODEL_TYPES = ("openai_batch", "anthropic_batch")
COMPATIBLE_MODEL_TYPE = "openai_compatible"
ANOTHER_SERVER = (f"a server that speaks OpenAI's Chat Completions is reached with model_type "
                  f"{COMPATIBLE_MODEL_TYPE!r} and its base_url")
# Any other model type belongs to a registered provider.
BUILTIN_MODEL_TYPES = ("local", "openai", "anthropic", "fake", *BATCH_MODEL_TYPES, COMPATIBLE_MODEL_TYPE)
DEFAULT_MAX_NEW_TOKENS = 2**16
DEFAULT_SEED = 16
# Model config keys a reasoning level is read from, in that order.
REASONING_KEYS = ("reasoning_level", "gpt_oss_reasoning_level")

# Set on a local judge's detail when its thinking-token stripper could not run or left an unclosed thought, which
# leaves it no answer.
STRIP_FAILED_KEY = "strip_failed"

TRACE_DETAIL_KEYS = (*REASONING_FIELDS, STRIP_FAILED_KEY)

# Decides which of one prompt's candidates a validated call kept, given the texts and the count of valid ones.
AcceptedFn = Callable[[List[str], int], List[bool]]


def _first_if_valid(candidates: List[str], num_valid: int) -> List[bool]:
    return [index == 0 and num_valid > 0 for index in range(len(candidates))]


def _non_empty(candidates: List[str]) -> List[str]:
    return [candidate for candidate in candidates if candidate and candidate.strip()]


def _each_non_empty(candidates: List[str], num_valid: int) -> List[bool]:
    kept = [bool(candidate and candidate.strip()) for candidate in candidates]
    return [keep and kept[:index].count(True) < num_valid for index, keep in enumerate(kept)]


def _texts(details: List[Dict[str, Any]]) -> List[str]:
    return [detail.get("completion") or "" for detail in details]


def _free_form_validator(details: List[Dict[str, Any]]) -> Tuple[Optional[List[str]], int]:
    # A free-form answer cut off at the output ceiling is kept: its caller sees the finish reason and decides.
    non_empty = _non_empty(_texts(details))
    return (non_empty, len(non_empty)) if non_empty else (None, 0)


def _judged_text(details: List[Dict[str, Any]]) -> str:
    """Return the text a judgment is read from: the first candidate's, or none when it stopped at the output ceiling or
    holds a thought that never closed, as a campaign rejects it (``is_truncated``, ``holds_an_open_thought``)."""
    first = details[0] if details else {}
    text = (first.get("completion") or "").strip()
    return "" if is_truncated(first.get("finish_reason")) or holds_an_open_thought(text) else text


def _json_validator(details: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
    from ruleforge.judges.models import parse_json_from_response

    text = _judged_text(details)
    parsed = parse_json_from_response(text) if text else None
    return (parsed, 1) if parsed else (None, 0)


def _checked_json_validator(
    check: Callable[[Any], bool] | None,
) -> Callable[[List[Dict[str, Any]]], Tuple[Optional[Dict[str, Any]], int]]:
    def _validator(details: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], int]:
        parsed, valid = _json_validator(details)
        return (parsed, valid) if valid and (check is None or check(parsed)) else (None, 0)

    return _validator


def prompt_attempt_seed(seed: int, prompt: Mapping[str, str], attempt: int) -> int:
    """Return the sampler seed of one attempt at one prompt: a hash of the configured seed, the prompt's text and the
    attempt number, so equal calls draw equal seeds and different prompts different ones."""
    payload = [int(seed), prompt.get("system_prompt", ""), prompt.get("user_prompt", ""), int(attempt)]
    return int.from_bytes(hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8], "big") % 2**31


@dataclass(frozen=True)
class GenerationOutcome:
    """What one prompt produced in one attempt, unvalidated: ``text`` the stripped non-empty completion or None,
    ``error`` this prompt's transport failure, ``block_failed`` the whole call's; reasoning is in neither ``text`` nor
    ``raw_response``."""

    text: Optional[str]
    raw_response: Optional[str]
    finish_reason: Optional[str]
    error: Optional[str] = None
    provider_request_ref: Optional[str] = None
    request_block_ref: Optional[str] = None
    block_failed: bool = False
    reasoning_summaries: Optional[tuple[str, ...]] = None
    reasoning: Optional[str] = None

    @property
    def returned(self) -> bool:
        """Return whether the provider returned anything for this prompt."""
        return self.error is None and (
            self.raw_response is not None or self.finish_reason is not None
        )


def _optional_str(model_cfg: DictConfig, key: str) -> str | None:
    value = getattr(model_cfg, key, None)
    return None if value is None else str(value)


def _optional_int(model_cfg: DictConfig, key: str) -> int | None:
    value = getattr(model_cfg, key, None)
    return None if value is None else int(value)


UNSET = "unset"
# Environment variables that change a local engine's arithmetic without being engine arguments, each with its library's
# default: vLLM's fp32 GEMM precision, Triton's ``tl.dot`` precision, vLLM's batch invariance.
PRECISION_ENV_DEFAULTS = {
    "VLLM_FLOAT32_MATMUL_PRECISION": "highest",
    "TRITON_F32_DEFAULT": UNSET,
    "VLLM_BATCH_INVARIANT": "0",
}


# Spelled as ``torch.set_float32_matmul_precision`` requires.
_MATMUL_PRECISIONS = ("highest", "high", "medium")


def _matmul_precision(value: str) -> str:
    """Return ``VLLM_FLOAT32_MATMUL_PRECISION`` exactly, refusing a mis-cased value: vLLM accepts it but torch ignores
    it."""
    if value in _MATMUL_PRECISIONS:
        return value
    if value.lower() in _MATMUL_PRECISIONS:
        raise ValueError(
            f"VLLM_FLOAT32_MATMUL_PRECISION={value!r} is mis-cased: vLLM accepts it, but torch ignores it and "
            f"keeps the precision the worker already has; spell it exactly as one of {list(_MATMUL_PRECISIONS)}"
        )
    raise ValueError(
        f"VLLM_FLOAT32_MATMUL_PRECISION={value!r} is not one of {list(_MATMUL_PRECISIONS)}, which vLLM requires"
    )


def _integer_flag(value: str) -> str:
    try:
        return str(int(bool(int(value))))
    except ValueError:
        raise ValueError(f"VLLM_BATCH_INVARIANT={value!r} is not an integer, which vLLM requires") from None


_PRECISION_ENV_READERS: Dict[str, Callable[[str], Optional[str]]] = {
    "VLLM_FLOAT32_MATMUL_PRECISION": _matmul_precision,
    "TRITON_F32_DEFAULT": lambda value: value.lower() or None,
    "VLLM_BATCH_INVARIANT": _integer_flag,
}


def precision_setting(name: str, value: Any, *, strict: bool = True) -> str:
    """Return one precision variable as its library reads it; a refused value raises, or with ``strict=False`` stays as
    given."""
    if value is None:
        return PRECISION_ENV_DEFAULTS[name]
    try:
        read = _PRECISION_ENV_READERS[name](str(value))
    except ValueError:
        if strict:
            raise
        return str(value)
    return PRECISION_ENV_DEFAULTS[name] if read is None else read


def precision_env(environ: Mapping[str, str] | None = None, *, strict: bool = True) -> Dict[str, str]:
    """Return the precision variables as ``environ`` sets them."""
    environ = os.environ if environ is None else environ
    return {name: precision_setting(name, environ.get(name), strict=strict) for name in PRECISION_ENV_DEFAULTS}


def batch_invariant_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether vLLM runs batch-invariant."""
    environ = os.environ if environ is None else environ
    return precision_setting("VLLM_BATCH_INVARIANT", environ.get("VLLM_BATCH_INVARIANT")) == "1"


# The dtypes a local engine may serve its weights in. Unset means vLLM's ``auto``, which downcasts float32 to bf16.
SERVING_DTYPES = ("float32", "bfloat16", "float16")
_SERVING_DTYPE_ALIASES = {"float": "float32", "half": "float16"}
# The attention backends that serve a float32 model; vLLM raises at start on any other selected explicitly.
FP32_ATTENTION_BACKENDS = frozenset({"TRITON_ATTN", "FLEX_ATTENTION"})


def canonical_serving_dtype(dtype: Any, attention_backend: Any = None) -> Optional[str]:
    """Return ``dtype`` as one of ``SERVING_DTYPES``, or None when unset; ``auto`` and float32 on another backend are
    refused."""
    if dtype is None:
        return None
    name = str(dtype).lower().removeprefix("torch.")
    name = _SERVING_DTYPE_ALIASES.get(name, name)
    if name not in SERVING_DTYPES:
        raise ValueError(
            f"dtype {dtype!r} is not one of {list(SERVING_DTYPES)}; leave it unset for vLLM's auto (the "
            "checkpoint's own dtype, but bf16 for a float32 checkpoint on Hopper)"
        )
    if name == "float32" and attention_backend is not None and str(attention_backend) not in FP32_ATTENTION_BACKENDS:
        raise ValueError(
            f"dtype float32 with attention_backend {attention_backend}: that backend serves only fp16 and bf16, "
            f"and vLLM refuses it at start; name one of {sorted(FP32_ATTENTION_BACKENDS)} (the fp32 engine, "
            "configs/model/qwen3-8b-fp32.yaml, names TRITON_ATTN)"
        )
    return name


# Model config keys passed on to ``init_local_model``; ``engine_args`` carries any further vLLM engine argument.
LOCAL_ENGINE_SETTING_KEYS = (
    "model_path", "tp_size", "reasoning_parser", "speculative_method", "num_speculative_tokens",
    "max_model_len", "attention_backend", "enable_thinking", "disable_log_stats", "max_num_seqs",
    "language_model_only", "dtype", "engine_args",
)
# ``seed`` seeds each generation request, never the engine.
_MODEL_CONFIG_OWN_KEYS = frozenset({"seed"})


def local_engine_settings(
    model_cfg: DictConfig, *, engine_fields: Optional[Collection[str]] = None
) -> Dict[str, Any]:
    """Return ``init_local_model``'s arguments for a local model config, LoRA aside. A top-level key naming another vLLM
    engine argument is refused, because it would be dropped silently: use ``engine_args``."""
    from ruleforge.judges.models import ENGINE_ARGS_OUTSIDE_IDENTITY, engine_argument_names

    fields = frozenset(engine_argument_names() if engine_fields is None else engine_fields)
    unplumbed_names = (
        (fields | ENGINE_ARGS_OUTSIDE_IDENTITY) - set(LOCAL_ENGINE_SETTING_KEYS) - _MODEL_CONFIG_OWN_KEYS
    )
    unplumbed = sorted(key for key in unplumbed_names if getattr(model_cfg, key, None) is not None)
    if unplumbed:
        message = (
            f"model config key(s) {unplumbed} name vLLM engine arguments that no model config key passes "
            "to vLLM, so they would be dropped silently; set such an argument under engine_args"
        )
        outside = [key for key in unplumbed if key in ENGINE_ARGS_OUTSIDE_IDENTITY]
        if outside:
            message += (
                f". {outside} change the served function, which the condition identity does not record, "
                "so engine_args refuses them too (each needs a model config key plumbed into the identity "
                "first, as dtype is through serving_dtype)"
            )
        raise ValueError(message)
    engine_args = getattr(model_cfg, "engine_args", None)
    if engine_args is not None and not isinstance(engine_args, Mapping):
        raise ValueError(
            f"engine_args must be a mapping of vLLM engine arguments, got {type(engine_args).__name__}"
        )
    return {
        "model_path": model_cfg.model_path,
        "tp_size": model_cfg.tp_size,
        "reasoning_parser": model_cfg.reasoning_parser,
        "speculative_method": model_cfg.speculative_method,
        "num_speculative_tokens": model_cfg.num_speculative_tokens,
        "max_model_len": model_cfg.max_model_len,
        "attention_backend": model_cfg.attention_backend,
        "enable_thinking": bool(model_cfg.enable_thinking),
        "disable_log_stats": bool(getattr(model_cfg, "disable_log_stats", True)),
        "max_num_seqs": getattr(model_cfg, "max_num_seqs", None),
        "language_model_only": bool(getattr(model_cfg, "language_model_only", False)),
        "dtype": getattr(model_cfg, "dtype", None),
        "engine_args": (
            OmegaConf.to_container(engine_args, resolve=True)
            if isinstance(engine_args, DictConfig)
            else dict(engine_args or {})
        ),
    }


def _label_validator(
    valid_responses: Collection[str], response_validator: Callable[[str], Optional[str]] | None
) -> Callable[[List[Dict[str, Any]]], Tuple[Optional[str], int]]:
    """Return the candidate check of a validated generation: ``response_validator``, else the label reader, over the
    text a judgment is read from. Raises ValueError when no candidate could ever be accepted."""
    valid_set = {v.strip().lower() for v in valid_responses}
    if response_validator is None and not valid_set <= set(LABELS):
        raise ValueError(f"valid_responses {sorted(valid_set)} are not labels the label reader reads ({LABELS}); "
                         "pass a response_validator that reads them")

    def _validator(details: List[Dict[str, Any]]) -> Tuple[str | None, int]:
        text = _judged_text(details)
        if response_validator is not None:
            judgment = response_validator(text) if text else None
            return (judgment, 1) if judgment is not None else (None, 0)
        label = read_label(text)
        return (label, 1) if label in valid_set else (None, 0)

    return _validator


class ModelBackend:
    """Unified backend for local vLLM and API models, with retries, thinking-token stripping and output validation built
    in."""

    #: A local engine's token count of a prompt as it would send it, and the prompt length it refuses; None elsewhere.
    prompt_token_count: Optional[Callable[[Dict[str, str]], int]] = None
    prompt_token_limit: Optional[int] = None

    def __init__(self, model_cfg: DictConfig):
        self.model_type: str = str(model_cfg.model_type)
        self.model_name: str = str(model_cfg.model_name)
        self.max_new_tokens: int = int(getattr(model_cfg, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        self.seed: int = int(getattr(model_cfg, "seed", DEFAULT_SEED))
        self.temperature: float = float(getattr(model_cfg, "temperature", 1.0))
        self.num_tries: int = int(getattr(model_cfg, "num_tries", 5))

        fixed_temperature = getattr(model_cfg, "fixed_temperature", None)
        self.fixed_temperature: float | None = (
            float(fixed_temperature) if fixed_temperature is not None else None
        )
        max_retry_seconds = getattr(model_cfg, "max_retry_seconds", None)
        self.max_retry_seconds: float | None = (
            float(max_retry_seconds) if max_retry_seconds is not None else None
        )
        if self.max_retry_seconds is not None and self.max_retry_seconds <= 0:
            raise ValueError("max_retry_seconds must be positive")

        adapter_path = getattr(model_cfg, "adapter_path", None)
        self.adapter_path: Path | None = Path(adapter_path) if adapter_path else None
        self.adapter_metadata: Dict[str, Any] | None = None
        self.engine_args: Dict[str, Any] | None = None
        # What the engine resolved at start; the campaign runner checks both against the condition.
        self.model_path: str | None = None
        self.engine_resolved: Dict[str, Any] | None = None
        # Local vLLM reports no usage, so this stays at zero there.
        self.token_tracker = TokenUsageTracker()

        # (prompts, temperature, max_new_tokens, num_responses, seed) -> List[List[str]]
        self._batch_generate: Callable[
            [List[Dict[str, str]], float, int, int, Optional[int]],
            List[List[str]],
        ]
        self._base_delay: float
        self.llm: LLM | None = (
            None
        )
        self.seed_capability: str = SEED_UNSUPPORTED
        self._generate_request_block: Optional[
            Callable[
                [List[Dict[str, str]], float, int, Optional[Sequence[Optional[int]]], str],
                List[Dict[str, Any]],
            ]
        ] = None

        # A detached snapshot: later config changes alter neither what this backend sends nor what it says it is.
        from ruleforge.judges.base import JudgeOptions

        self._options = JudgeOptions(model_cfg)
        from ruleforge.judges.providers import builtin_provider

        provider = builtin_provider(self.model_type)
        provider.configure(self, model_cfg)
        self.capabilities = provider.capabilities_of(model_cfg)

    @property
    def spec(self) -> JudgeSpec:
        """Return the ``JudgeSpec`` of the model config this backend was built from."""
        from ruleforge.judges.base import JudgeSpec

        return JudgeSpec(model_type=self.model_type, model_name=self.model_name, options=self._options)

    @property
    def usage(self) -> TokenUsageTracker:
        """Return the token usage the provider has reported."""
        return self.token_tracker

    def identity(self) -> JudgeIdentity:
        """Return its spec's ``JudgeIdentity``."""
        return self.spec.identity(verify_weights=False)

    def close(self) -> None:
        """Shut a local engine down; does nothing for a provider."""
        if self.llm is not None:
            from ruleforge.judges.models import cleanup_vllm

            llm, self.llm = self.llm, None
            cleanup_vllm(llm, raise_on_error=False)

    @staticmethod
    def _reasoning_effort(model_cfg: DictConfig) -> str | None:
        """Return the configured reasoning level, or None: an unset level is never upgraded on a metered provider."""
        for key in REASONING_KEYS:
            value = _optional_str(model_cfg, key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _make_seed_rng(seed: int | None) -> random.Random | None:
        return random.Random(seed) if seed is not None else None

    @staticmethod
    def _next_seed(rng: random.Random | None) -> int | None:
        return rng.randrange(2**31) if rng is not None else None

    @staticmethod
    def _validate_prompts(prompts: List[Dict[str, str]]) -> None:
        for index, entry in enumerate(prompts):
            if not isinstance(entry, dict):
                raise ValueError(f"prompt {index} must be a dictionary")
            system_prompt = entry.get("system_prompt", "")
            user_prompt = entry.get("user_prompt", "")
            if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
                raise ValueError(
                    f"prompt {index} system_prompt and user_prompt must be strings"
                )
            if not system_prompt.strip() and not user_prompt.strip():
                raise ValueError(f"prompt {index} must contain nonblank content")

    def _backoff(self, attempt: int) -> None:
        if self._base_delay <= 0:
            return
        delay = min(self._base_delay * (2**attempt), 60.0)
        time.sleep(delay)

    def _warn_if_temperature_pinned(self, requested: float | None) -> None:
        if (
            self.fixed_temperature is not None
            and requested is not None
            and requested != self.fixed_temperature
        ):
            logger.warning(
                "%s forces temperature=%s; the requested %s is ignored, so a "
                "caller relying on greedy decoding will not get it.",
                self.model_name,
                self.fixed_temperature,
                requested,
            )

    def resolved_temperature(self, requested: float | None) -> float:
        """Return the temperature a request would be sent at; a provider-pinned ``fixed_temperature`` wins."""
        return self._resolve(requested, None, None, None)[0]

    def _resolve(
        self,
        temperature: float | None,
        max_new_tokens: int | None,
        seed: int | None,
        num_tries: int | None,
    ) -> Tuple[float, int, int, int]:
        return (
            (
                self.fixed_temperature
                if self.fixed_temperature is not None
                else temperature
                if temperature is not None
                else self.temperature
            ),
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            seed if seed is not None else self.seed,
            num_tries if num_tries is not None else self.num_tries,
        )

    def _call_validate_retry(
        self,
        prompts: List[Dict[str, str]],
        validator_fn: Callable[[List[Dict[str, Any]]], Tuple[T | None, int]],
        temperature: float,
        max_new_tokens: int,
        num_responses: int,
        seed: int,
        num_tries: int,
        attempt_traces: List[List[Dict[str, Any]]] | None = None,
        accepted_fn: AcceptedFn = _first_if_valid,
        max_retry_seconds: float | None = None,
    ) -> List[T | None]:
        """Retry each prompt until it has ``num_responses`` valid candidates or the retries run out (None for a prompt
        with none). ``validator_fn(details)`` returns a partial result and its count of valid candidates;
        ``attempt_traces`` receives every attempt; ``max_retry_seconds`` replaces the config's budget."""
        n = len(prompts)
        results: List[T | None] = [None] * n
        remaining: List[int] = [num_responses] * n
        pending_idx: List[int] = list(range(n))
        seed_rng = self._make_seed_rng(seed)
        # Where the seed reaches a sampler each prompt's attempt has its own seed (``prompt_attempt_seed``), not one
        # shared random stream; a batch transport addresses its batch by the attempt's one seed.
        per_prompt = seed_rng is not None and getattr(self, "seed_capability", None) == SEED_LOCAL_DETERMINISTIC

        retry_seconds = self.max_retry_seconds if max_retry_seconds is None else max_retry_seconds
        # Armed after the first attempt returns: one pass over a slow reasoning model can outlast the whole budget.
        retry_deadline: float | None = None

        def budget_exhausted() -> bool:
            return retry_deadline is not None and time.monotonic() >= retry_deadline

        def arm_retry_budget(current: float | None) -> float | None:
            if retry_seconds is None or current is not None:
                return current
            return time.monotonic() + retry_seconds

        attempts_made = num_tries
        for attempt in range(num_tries):
            if not pending_idx:
                break
            if attempt > 0:
                if budget_exhausted():
                    logger.warning(
                        "Retry budget of %.0fs exhausted before attempt %d",
                        retry_seconds,
                        attempt + 1,
                    )
                    attempts_made = attempt
                    break
                self._backoff(attempt - 1)
                if budget_exhausted():
                    logger.warning(
                        "Retry budget of %.0fs exhausted during backoff before "
                        "attempt %d",
                        retry_seconds,
                        attempt + 1,
                    )
                    attempts_made = attempt
                    break
            logger.info(f"Attempt {attempt + 1}/{num_tries} of model generation")

            batch = [prompts[i] for i in pending_idx]
            request_n = max(remaining[i] for i in pending_idx)
            if per_prompt:
                prompt_seeds = [prompt_attempt_seed(seed, prompt, attempt + 1) for prompt in prompts]
                attempt_seed: Any = [prompt_seeds[i] for i in pending_idx]
            else:
                prompt_seeds = [self._next_seed(seed_rng)] * n
                attempt_seed = prompt_seeds[0]

            try:
                # Details, where the judge has them, so a candidate is validated with its finish reason.
                detail_fn = getattr(self, "_batch_generate_with_details", None)
                if detail_fn is not None:
                    batch_details = detail_fn(
                        batch,
                        temperature,
                        max_new_tokens,
                        request_n,
                        attempt_seed,
                    )
                else:
                    batch_details = [
                        [
                            {
                                "raw_response": candidate,
                                "completion": candidate,
                                "finish_reason": None,
                            }
                            for candidate in candidates
                        ]
                        for candidates in self._batch_generate(
                            batch,
                            temperature,
                            max_new_tokens,
                            request_n,
                            attempt_seed,
                        )
                    ]
                batch_results = [_texts(candidates) for candidates in batch_details]
            except FATAL_PROVIDER_ERRORS:
                # A misconfiguration, not a flaky call: retrying it would end in a complete file of null judgments.
                raise
            except Exception as exc:
                logger.warning(f"Attempt {attempt + 1}/{num_tries} failed: {exc}")
                if attempt_traces is not None:
                    for orig_idx in pending_idx:
                        attempt_traces[orig_idx].append(
                            {
                                "attempt": attempt + 1,
                                "seed": prompt_seeds[orig_idx],
                                "error": str(exc),
                                "candidates": [],
                            }
                        )
                retry_deadline = arm_retry_budget(retry_deadline)
                continue
            retry_deadline = arm_retry_budget(retry_deadline)

            still_pending: List[int] = []
            for j, orig_idx in enumerate(pending_idx):
                candidates = batch_results[j] if j < len(batch_results) else []
                details = batch_details[j] if j < len(batch_details) else []
                partial, num_valid = validator_fn(details)
                # A retry asks every pending prompt for the most any still needs; each keeps what it needs (a free-form
                # list may keep more).
                if num_valid > remaining[orig_idx]:
                    num_valid = remaining[orig_idx]
                    partial = partial[:num_valid]
                if attempt_traces is not None:
                    kept = accepted_fn(candidates, num_valid)
                    attempt_traces[orig_idx].append(
                        {
                            "attempt": attempt + 1,
                            "seed": prompt_seeds[orig_idx],
                            "candidates": [
                                {
                                    "raw_response": detail.get("raw_response", ""),
                                    "completion": detail.get("completion", ""),
                                    "finish_reason": detail.get("finish_reason"),
                                    **{
                                        key: detail[key]
                                        for key in TRACE_DETAIL_KEYS
                                        if key in detail
                                    },
                                    "accepted": bool(index < len(kept) and kept[index]),
                                }
                                for index, detail in enumerate(details)
                            ],
                        }
                    )
                if partial is not None and num_valid > 0:
                    if results[orig_idx] is None:
                        results[orig_idx] = partial
                    else:
                        results[orig_idx] = results[orig_idx] + partial
                    remaining[orig_idx] -= num_valid

                if remaining[orig_idx] > 0:
                    still_pending.append(orig_idx)

            pending_idx = still_pending
            logger.info(
                f"Pending prompts after attempt {attempt + 1}: {len(pending_idx)}"
            )

        if pending_idx:
            logger.warning(
                f"{len(pending_idx)}/{n} prompts have no valid answer after {attempts_made} of {num_tries} attempts; "
                "they are dropped"
            )

        return results

    def generate_once(
        self,
        prompts: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
        request_block_id: str = "",
        truncation: Optional[Mapping[str, Any]] = None,
    ) -> List[GenerationOutcome]:
        """Submit every prompt exactly once and return one outcome each, leaving retries to the caller (``seeds``: one
        per prompt or None; ``request_block_id``: a block's address); a fatal provider error propagates, any other
        transport exception is every prompt's ``error``."""
        self._validate_prompts(prompts)
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError(
                f"seeds must align with prompts, got {len(seeds)} seeds for "
                f"{len(prompts)} prompts"
            )
        self._warn_if_temperature_pinned(temperature)
        resolved_temperature, resolved_max_tokens, _, _ = self._resolve(
            temperature, max_new_tokens, None, None
        )
        seed_list = list(seeds) if seeds is not None else None
        block_fn = getattr(self, "_generate_request_block", None)
        detail_fn = getattr(self, "_batch_generate_with_details", None)
        try:
            if block_fn is not None:
                details = block_fn(
                    prompts, resolved_temperature, resolved_max_tokens, seed_list,
                    request_block_id,
                )
            elif detail_fn is not None:
                extra = {"truncation": truncation} if self.model_type == "local" else {}
                details = [
                    candidates[0] if candidates else {}
                    for candidates in detail_fn(
                        prompts, resolved_temperature, resolved_max_tokens, 1, seed_list, **extra
                    )
                ]
            else:
                details = [
                    {"raw_response": candidates[0], "completion": candidates[0]}
                    if candidates
                    else {}
                    for candidates in self._batch_generate(
                        prompts, resolved_temperature, resolved_max_tokens, 1, seed_list
                    )
                ]
        except FATAL_PROVIDER_ERRORS:
            raise
        except Exception as exc:
            logger.warning("A request block failed in transport: %s", exc)
            return [
                GenerationOutcome(text=None, raw_response=None, finish_reason=None,
                                  error=f"{type(exc).__name__}: {exc}", block_failed=True)
                for _ in prompts
            ]
        if len(details) != len(prompts):
            raise RuntimeError(
                f"provider returned {len(details)} results for {len(prompts)} prompts"
            )
        return self._outcomes_from_details(details)

    @staticmethod
    def _outcomes_from_details(
        details: Sequence[Mapping[str, Any]],
    ) -> List[GenerationOutcome]:
        outcomes = []
        for detail in details:
            completion = detail.get("completion")
            raw = detail.get("raw_response")
            summaries = detail.get(REASONING_SUMMARIES_KEY)
            reasoning = detail.get(REASONING_KEY)
            outcomes.append(
                GenerationOutcome(
                    text=completion if isinstance(completion, str) and completion else None,
                    raw_response=raw if isinstance(raw, str) else None,
                    finish_reason=detail.get("finish_reason"),
                    error=detail.get("error"),
                    provider_request_ref=detail.get("provider_request_ref"),
                    request_block_ref=detail.get("request_block_ref"),
                    reasoning_summaries=None if summaries is None else tuple(summaries),
                    reasoning=reasoning if isinstance(reasoning, str) and reasoning else None,
                )
            )
        return outcomes

    def _resolve_block_args(
        self, temperature: float | None, seeds: Optional[Sequence[Optional[int]]]
    ) -> tuple[float, int | None, Optional[list]]:
        self._warn_if_temperature_pinned(temperature)
        resolved_temperature, resolved_max_tokens, _, _ = self._resolve(
            temperature, None, None, None
        )
        return resolved_temperature, resolved_max_tokens, (list(seeds) if seeds is not None else None)

    def estimate_block_tokens(
        self,
        prompts: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
        request_block_id: str = "",
    ) -> int:
        """Return the estimated enqueued input tokens for one block (windowed backends only)."""
        self._validate_prompts(prompts)
        temp, max_tokens, seed_list = self._resolve_block_args(temperature, seeds)
        return self._estimate_request_block(
            prompts, temperature=temp, max_new_tokens=max_tokens,
            seeds=seed_list, request_block_id=request_block_id,
        )

    def submit_block(
        self,
        prompts: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
        request_block_id: str = "",
    ) -> None:
        """Create the batch for one block without waiting (windowed backends only); a block already submitted costs
        nothing."""
        self._validate_prompts(prompts)
        temp, max_tokens, seed_list = self._resolve_block_args(temperature, seeds)
        self._submit_request_block(
            prompts, temperature=temp, max_new_tokens=max_tokens,
            seeds=seed_list, request_block_id=request_block_id,
        )

    def poll_block(
        self,
        prompts: List[Dict[str, str]],
        *,
        temperature: float | None = None,
        seeds: Optional[Sequence[Optional[int]]] = None,
        request_block_id: str = "",
    ) -> Optional[List[GenerationOutcome]]:
        """Return one block's outcomes once its batch has completed, else None (windowed backends only)."""
        self._validate_prompts(prompts)
        temp, max_tokens, seed_list = self._resolve_block_args(temperature, seeds)
        details = self._poll_request_block(
            prompts, temperature=temp, max_new_tokens=max_tokens,
            seeds=seed_list, request_block_id=request_block_id,
        )
        if details is None:
            return None
        if len(details) != len(prompts):
            raise RuntimeError(
                f"provider returned {len(details)} results for {len(prompts)} prompts"
            )
        return self._outcomes_from_details(details)

    def _traced(
        self,
        prompts: List[Dict[str, str]],
        validator_fn: Callable[[List[Dict[str, Any]]], Tuple[T | None, int]],
        resolved: Tuple[float, int, int, int],
        num_responses: int,
        result_key: str,
        accepted_fn: AcceptedFn = _first_if_valid,
        max_retry_seconds: float | None = None,
    ) -> Tuple[List[T | None], List[Dict[str, Any]]]:
        """Run the validated retry loop keeping every attempt: one ``{result_key, selected_attempt, attempts}`` trace
        per prompt."""
        temperature, max_new_tokens, seed, num_tries = resolved
        attempt_traces: List[List[Dict[str, Any]]] = [[] for _ in prompts]
        results = self._call_validate_retry(
            prompts,
            validator_fn,
            temperature,
            max_new_tokens,
            num_responses,
            seed,
            num_tries,
            attempt_traces=attempt_traces,
            accepted_fn=accepted_fn,
            max_retry_seconds=max_retry_seconds,
        )
        traces = []
        for result, attempts in zip(results, attempt_traces):
            selected_attempt = next(
                (
                    attempt["attempt"]
                    for attempt in attempts
                    if any(
                        candidate.get("accepted")
                        for candidate in attempt.get("candidates", [])
                    )
                ),
                None,
            )
            traces.append(
                {
                    result_key: result,
                    "selected_attempt": selected_attempt,
                    "attempts": attempts,
                }
            )
        return results, traces

    def generate(
        self,
        prompts: List[Dict[str, str]],
        num_responses: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        num_tries: int | None = None,
    ) -> List[List[str] | None]:
        """Return up to ``num_responses`` non-empty candidates per prompt, or None for a prompt with none."""

        self._validate_prompts(prompts)
        self._warn_if_temperature_pinned(temperature)
        t, m, s, n = self._resolve(temperature, max_new_tokens, seed, num_tries)
        return self._call_validate_retry(prompts, _free_form_validator, t, m, num_responses, s, n)

    def generate_with_traces(
        self,
        prompts: List[Dict[str, str]],
        num_responses: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        num_tries: int | None = None,
    ) -> Tuple[List[List[str] | None], List[Dict[str, Any]]]:
        """Return what ``generate`` returns and one trace per prompt; a kept candidate's ``result_index`` is its result
        position."""
        self._validate_prompts(prompts)
        self._warn_if_temperature_pinned(temperature)
        results, traces = self._traced(
            prompts,
            _free_form_validator,
            self._resolve(temperature, max_new_tokens, seed, num_tries),
            num_responses,
            "result",
            accepted_fn=_each_non_empty,
        )
        for trace in traces:
            kept = (
                candidate
                for attempt in trace["attempts"]
                for candidate in attempt["candidates"]
                if candidate["accepted"]
            )
            for index, candidate in enumerate(kept):
                candidate["result_index"] = index
        return results, traces

    def generate_with_validation(
        self,
        prompts: List[Dict[str, str]],
        valid_responses: tuple[str, ...],
        temperature: float | None = None,
        seed: int | None = None,
        num_tries: int | None = None,
        response_validator: Callable[[str], Optional[str]] | None = None,
        max_retry_seconds: float | None = None,
    ) -> List[str | None]:
        """Return one validated response per prompt (None once its retries run out), read by
        ``ruleforge.prompts.parse.read_label`` or ``response_validator`` (None rejects and retries);
        ``max_retry_seconds`` replaces the config's budget."""
        self._validate_prompts(prompts)
        self._warn_if_temperature_pinned(temperature)
        validator = _label_validator(valid_responses, response_validator)
        t, m, s, n = self._resolve(temperature, None, seed, num_tries)
        return self._call_validate_retry(prompts, validator, t, m, 1, s, n, max_retry_seconds=max_retry_seconds)

    def generate_with_validation_and_traces(
        self,
        prompts: List[Dict[str, str]],
        valid_responses: tuple[str, ...],
        temperature: float | None = None,
        seed: int | None = None,
        num_tries: int | None = None,
        response_validator: Callable[[str], Optional[str]] | None = None,
        max_retry_seconds: float | None = None,
    ) -> Tuple[List[str | None], List[Dict[str, Any]]]:
        """Return validated judgments and one ``{judgment, selected_attempt, attempts}`` trace per prompt."""
        self._validate_prompts(prompts)
        self._warn_if_temperature_pinned(temperature)
        return self._traced(
            prompts,
            _label_validator(valid_responses, response_validator),
            self._resolve(temperature, None, seed, num_tries),
            1,
            "judgment",
            max_retry_seconds=max_retry_seconds,
        )

    def generate_with_json_validation(
        self,
        prompts: List[Dict[str, str]],
        temperature: float | None = None,
        seed: int | None = None,
        num_tries: int | None = None,
    ) -> List[Dict[str, Any] | None]:
        """Return one parsed JSON dict per prompt, or None when its retries ran out."""
        self._validate_prompts(prompts)
        self._warn_if_temperature_pinned(temperature)
        t, m, s, n = self._resolve(temperature, None, seed, num_tries)
        return self._call_validate_retry(prompts, _json_validator, t, m, 1, s, n)

    def generate_with_json_validation_and_traces(
        self,
        prompts: List[Dict[str, str]],
        temperature: float | None = None,
        seed: int | None = None,
        num_tries: int | None = None,
        check: Callable[[Any], bool] | None = None,
    ) -> Tuple[List[Dict[str, Any] | None], List[Dict[str, Any]]]:
        """Return parsed JSON dicts and one ``{result, selected_attempt, attempts}`` trace per prompt; an answer ``check``
        refuses counts as a failed attempt."""
        self._validate_prompts(prompts)
        self._warn_if_temperature_pinned(temperature)
        return self._traced(
            prompts,
            _checked_json_validator(check),
            self._resolve(temperature, None, seed, num_tries),
            1,
            "result",
        )
