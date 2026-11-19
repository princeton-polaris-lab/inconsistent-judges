from __future__ import annotations

from typing import Any


def vllm_config(llm: Any) -> Any:
    """Return the engine's ``VllmConfig``, or None."""
    return getattr(getattr(llm, "llm_engine", None), "vllm_config", None)


def engine_dtype(llm: Any) -> str | None:
    """Return the model dtype the engine resolved (``float32``, ``bfloat16``), or None."""
    config = getattr(llm, "model_config", None)
    return None if config is None else str(config.dtype).removeprefix("torch.")


def kv_cache_dtype(llm: Any) -> str | None:
    """Return the KV-cache dtype, ``auto`` resolved to the model's, or None."""
    value = getattr(getattr(vllm_config(llm), "cache_config", None), "cache_dtype", None)
    if value is None:
        return None
    return engine_dtype(llm) if str(value) == "auto" else str(value)


def engine_attention_backend(llm: Any) -> str | None:
    """Return the explicitly configured attention backend, or None when vLLM chose one itself."""
    backend = getattr(getattr(vllm_config(llm), "attention_config", None), "backend", None)
    return None if backend is None else getattr(backend, "name", str(backend))


def engine_max_model_len(llm: Any) -> int | None:
    """Return the longest sequence the engine resolved, or None."""
    return getattr(getattr(llm, "model_config", None), "max_model_len", None)


def engine_resolution(llm: Any) -> dict[str, str | None]:
    """Return the engine's resolved model dtype, KV-cache dtype and attention backend."""
    return {
        "engine_dtype": engine_dtype(llm),
        "kv_cache_dtype": kv_cache_dtype(llm),
        "attention_backend": engine_attention_backend(llm),
    }
