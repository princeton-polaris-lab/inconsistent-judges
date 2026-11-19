from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ruleforge.judges.base import (
    MODEL_CONFIG_DIR,
    SHORTHANDS,
    BatchJudge,
    Capabilities,
    Judge,
    JudgeError,
    JudgeIdentity,
    JudgeOptions,
    JudgeSpec,
    Provider,
    extension,
    judge_forms,
    refuse_credential,
)
from ruleforge.judges.providers import ENTRY_POINT_GROUP, PROVIDERS, provider_for, register_judge
from ruleforge.judges.backend import BUILTIN_MODEL_TYPES, SEED_UNSUPPORTED, GenerationOutcome


def _compatible(text: str) -> JudgeSpec:
    from ruleforge.judges.compatible import split_shorthand

    model, base_url = split_shorthand(text)
    return JudgeSpec.openai_compatible(model, base_url)


_NAMED_CONSTRUCTORS = {"openai": JudgeSpec.openai, "anthropic": JudgeSpec.anthropic, "local": JudgeSpec.local,
                       "openai_compatible": _compatible}
PROBE_PROMPT = {"system_prompt": "Answer with one word, yes, no or unsure: is this message a check?", "user_prompt": ""}


def resolve(judge: Any) -> JudgeSpec:
    """Return the judge ``judge`` names: a spec, a model config, or one of ``SHORTHANDS``."""
    if isinstance(judge, JudgeSpec):
        return judge
    if isinstance(judge, str):
        refuse_credential(judge)
        kind, colon, rest = judge.partition(":")
        if kind == "fake" and (not colon or rest.lstrip("-").isdigit()):
            return JudgeSpec.fake(int(rest) if colon else 1)
        if colon and kind in _NAMED_CONSTRUCTORS and rest:
            return _NAMED_CONSTRUCTORS[kind](rest)
    return JudgeSpec.from_config(judge)


@dataclass(frozen=True)
class CheckReport:
    """What ``check`` found: the judge's spec, identity, capabilities, request settings and probe answer (if asked)."""

    spec: JudgeSpec
    identity: JudgeIdentity
    capabilities: Capabilities
    request_settings: Mapping[str, Any]
    probe: Optional[GenerationOutcome] = None

    @property
    def ok(self) -> bool:
        """Return whether the probe, when made, came back answered."""
        return self.probe is None or (self.probe.returned and self.probe.text is not None)


def check(judge: Any, *, probe: Optional[bool] = None) -> CheckReport:
    """Check a judge without a campaign: resolve it, verify its weights, read its settings, and (by default only for a
    free judge with no local engine) send it one probe prompt."""
    spec = resolve(judge)
    identity = spec.identity(verify_weights=True)
    capabilities = spec.capabilities()
    settings = spec.request_settings()
    if probe is None:
        probe = not (capabilities.spends_money or capabilities.local_engine)
    outcome = None
    if probe:
        if capabilities.spends_money:
            from ruleforge.judges.batch_api import require_spend_confirmation

            require_spend_confirmation()
        started = spec.start()
        try:
            seeds = None if capabilities.seed == SEED_UNSUPPORTED else [0]
            outcome = started.generate_once([dict(PROBE_PROMPT)], temperature=1.0, seeds=seeds)[0]
        finally:
            started.close()
    return CheckReport(spec=spec, identity=identity, capabilities=capabilities, request_settings=settings,
                       probe=outcome)


__all__ = [
    "BUILTIN_MODEL_TYPES", "ENTRY_POINT_GROUP", "MODEL_CONFIG_DIR", "PROBE_PROMPT", "PROVIDERS", "SHORTHANDS",
    "BatchJudge", "Capabilities", "CheckReport", "GenerationOutcome", "Judge", "JudgeError", "JudgeIdentity",
    "JudgeOptions", "JudgeSpec", "Provider", "check", "extension", "judge_forms", "provider_for", "register_judge",
    "resolve",
]
