from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TokenUsage:
    """Usage reported by a single provider call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response(cls, usage: Any) -> "TokenUsage":
        """Read usage off a provider response (object or mapping); a missing field counts as zero."""
        if usage is None:
            return cls()
        if isinstance(usage, Mapping):
            read = usage.get
        else:
            def read(name: str, default: int = 0) -> Any:
                return getattr(usage, name, default)

        def as_int(name: str) -> int:
            value = read(name, 0)
            return int(value) if isinstance(value, (int, float)) else 0

        prompt = as_int("prompt_tokens") or as_int("prompt_token_count") \
            or as_int("input_tokens")
        completion = as_int("completion_tokens") or as_int("candidates_token_count") \
            or as_int("output_tokens")
        total = as_int("total_tokens") or as_int("total_token_count")
        return cls(prompt, completion, total or prompt + completion)


@dataclass
class TokenUsageTracker:
    """Cumulative usage across the calls of one run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: TokenUsage) -> None:
        """Accumulate one call's usage."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.calls += 1
