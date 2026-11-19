from __future__ import annotations

import importlib
import sys
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ruleforge.evaluate.results import EvalResult

_EXPORTS = {
    **dict.fromkeys((
        "ARTIFACT_ROOT_ENV", "DEFAULT_BATCH_DIRNAME", "EXECUTION_FNAME", "LOCK_FNAME", "MARKER_FNAME",
        "RUN_LOCK_FNAME", "AnalysisExists", "Approvals", "Campaign", "CampaignError", "Condition",
        "ConditionSettings", "ReadOnlyCampaign", "SamplingSettings", "default_campaign_id", "default_root",
        "evaluate", "pinned_roots"), "api"),
    **dict.fromkeys((
        "CampaignResult", "Contrast", "Estimate", "Estimates", "EvalResult", "Kappa", "ValidityPolicy"), "results"),
    "BootstrapSettings": "identity",
    "ExecutionSettings": "packing",
}

__all__ = [
    "ARTIFACT_ROOT_ENV", "AnalysisExists", "Approvals", "BootstrapSettings", "Campaign", "CampaignError",
    "CampaignResult", "Condition", "ConditionSettings", "Contrast", "DEFAULT_BATCH_DIRNAME", "EXECUTION_FNAME",
    "Estimate", "Estimates", "EvalResult", "ExecutionSettings", "Kappa", "LOCK_FNAME", "MARKER_FNAME",
    "RUN_LOCK_FNAME", "ReadOnlyCampaign", "SamplingSettings", "ValidityPolicy",
    "default_campaign_id", "default_root", "evaluate", "pinned_roots",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


class _CallablePackage(types.ModuleType):
    """This package, callable as :func:`evaluate`: the package and the function share the name ``evaluate``."""

    def __call__(self, *args: Any, **kwargs: Any) -> EvalResult:
        return self.evaluate(*args, **kwargs)


sys.modules.get(__name__).__class__ = _CallablePackage
