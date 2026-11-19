from __future__ import annotations

import re
from typing import Sequence


def in_scope(path: str, target_modules: Sequence[str], module_scope: str | None = None) -> bool:
    """Return whether module `path` is a target: its leaf is one of `target_modules`, below `module_scope` when given.
    """
    return path.rpartition(".")[2] in target_modules and (module_scope is None or path.startswith(module_scope))


def peft_target_modules(target_modules: Sequence[str], module_scope: str | None = None) -> list[str] | str:
    """Return PEFT's `target_modules`: the leaves unscoped, else one full-match regex over the module path."""
    if module_scope is None:
        return list(target_modules)
    leaves = "|".join(re.escape(target) for target in target_modules)
    return rf"{re.escape(module_scope)}.*\.(?:{leaves})"

