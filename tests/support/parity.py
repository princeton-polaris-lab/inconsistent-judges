from __future__ import annotations

from typing import Any

from ruleforge.grpo.lora_parity import PARITY_PRESETS, ParityThresholds


def preset_thresholds(name: str, **overrides: Any) -> ParityThresholds:
    """Return the `ParityThresholds` of a `PARITY_PRESETS` entry with `overrides` over it, as `compare --preset` does."""
    return ParityThresholds(**{**PARITY_PRESETS[name], **overrides})
