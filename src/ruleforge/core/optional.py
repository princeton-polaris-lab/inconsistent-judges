from __future__ import annotations

import importlib.util

from ruleforge.core.errors import RuleForgeError


class MissingDependency(ImportError, RuleForgeError):
    """Raised when a command needs an optional dependency this environment lacks."""


#: Each optional dependency a command checks for: the extra that installs it, and the uv group that does in a checkout.
INSTALLED_BY = {
    "matplotlib": ("viz", "viz"),
    "submitit": ("slurm", "dev"),
    "huggingface_hub": ("hub", "dev"),
    "zstandard": ("hub", "dev"),
    "torch": ("grpo", "grpo"),
    "trl": ("grpo", "grpo"),
    "peft": ("grpo", "grpo"),
    "vllm": ("vllm", "vllm"),
}


def _installed(module: str) -> bool:
    # A directory of that name on the search path alone is a namespace portion, imported or not: it has no origin, and
    # it is not the package.
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.origin is not None


def require(purpose: str, *modules: str) -> None:
    """Refuse ``purpose`` in one line naming what installs them when any of ``modules`` is not installed; nothing is
    imported."""
    missing = [module for module in modules if not _installed(module)]
    if not missing:
        return
    extras = ",".join(sorted({INSTALLED_BY[module][0] for module in missing}))
    groups = " ".join(f"--group {group}" for group in sorted({INSTALLED_BY[module][1] for module in missing}))
    raise MissingDependency(f"{purpose} needs {', '.join(missing)}, which this environment lacks: add {groups} to "
                            f"uv sync, or pip install -e '.[{extras}]' from the checkout")
