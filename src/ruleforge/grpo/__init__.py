import importlib
from typing import Any

_API = {
    "PromptBuilder": "ruleforge.grpo.api",
    "RowsBuild": "ruleforge.grpo.api",
    "TrainRun": "ruleforge.grpo.api",
    "merge": "ruleforge.grpo.api",
    "rows": "ruleforge.grpo.api",
    "train": "ruleforge.grpo.api",
}


_MODULES = frozenset({"anchor_set", "exclusion", "report"})


def __getattr__(name: str) -> Any:
    """Return a toolkit name from its module, or one of the toolkit's modules."""
    if name in _MODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    elif name in _API:
        value = getattr(importlib.import_module(_API[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_API, *_MODULES})


__all__ = ["PromptBuilder", "RowsBuild", "TrainRun", "anchor_set", "exclusion", "merge", "report", "rows", "train"]
