from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def completion_text(completion: Any) -> str:
    """Return the text of a string or TRL conversational completion."""
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, Sequence) or not completion:
        return ""
    last = completion[-1]
    if not isinstance(last, Mapping):
        return ""
    content = last.get("content")
    return content if isinstance(content, str) else ""


def should_run_anchor_step(
    *,
    global_step: int,
    mix_every_n_steps: int,
    sync_gradients: bool,
) -> bool:
    """Return whether the anchor loss runs at this optimizer step."""
    if mix_every_n_steps <= 0:
        raise ValueError("mix_every_n_steps must be positive")
    return sync_gradients and (global_step + 1) % mix_every_n_steps == 0

