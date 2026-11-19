from __future__ import annotations

from importlib import import_module
from typing import Any

from ruleforge.env import register_resolvers as _register_resolvers

_register_resolvers()

_EXPORTS = {
    **dict.fromkeys(("Rule", "Ruleset"), "ruleforge.rulesets"),
    "ScenarioSet": "ruleforge.scenarios",
    **dict.fromkeys(("Capabilities", "GenerationOutcome", "Judge", "JudgeIdentity", "JudgeSpec", "register_judge"),
                    "ruleforge.judges"),
    **dict.fromkeys(("OptimizableTask", "Task", "TaskSettings", "load_task", "register_task"), "ruleforge.tasks"),
    **dict.fromkeys(("Campaign", "CampaignResult", "ConditionSettings", "Contrast", "Estimate", "Estimates",
                     "EvalResult", "ExecutionSettings", "Kappa", "SamplingSettings"), "ruleforge.evaluate"),
}
_PACKAGES = ("evaluate", "grpo", "optimize")

__all__ = sorted((*_EXPORTS, *_PACKAGES))


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from ruleforge.core.provenance import package_version

        return package_version()
    if name in _PACKAGES:
        return import_module(f"ruleforge.{name}")
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ruleforge' has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, "__version__"})
