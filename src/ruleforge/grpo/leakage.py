from __future__ import annotations

from pathlib import Path

from ruleforge.core.errors import RuleForgeError


class LeakageError(RuntimeError, RuleForgeError):
    """Two sets that must be disjoint are not."""


#: A marker's `identity` when the gate also compared normalized prompts; `exclusion` imports it from here.
PROMPT_IDENTITY = "normalized_prompt"


def disjoint_marker_path(out_path: Path) -> Path:
    """Return the sidecar path for `out_path`: `<out_path>.disjoint.json`."""
    return out_path.with_suffix(out_path.suffix + ".disjoint.json")
