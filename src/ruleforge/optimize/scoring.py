from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ruleforge.optimize.metrics.reward import compute_ra_reward, compute_sg_reward
from ruleforge.optimize.config import OptimizeConfig
from ruleforge.tasks.history import RewardDict, RewardWeights

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

_DEFAULTS = OptimizeConfig()
WEIGHT_KEYS = {"intent": "intent_weight", "entropy_method": "entropy_method", "intent_method": "intent_method"}
#: The rule-vs-ruleset penalty, which is not computed: a config may name it at these values and no others.
RETIRED_PENALTY = {"smrc_weight": 0.0, "smrc_method": "hierarchical_all_rules"}
_ABSENT = object()


def _panel_budget(eval_cfg: Any) -> Optional[float]:
    budget = getattr(eval_cfg, "panel_max_retry_seconds", _DEFAULTS.panel_max_retry_seconds)
    if budget is None:
        return None
    if float(budget) <= 0:
        raise ValueError(f"eval_cfg.panel_max_retry_seconds={budget}: it must be positive, or null for no limit")
    return float(budget)


@dataclass(frozen=True)
class RewardSettings:
    """How a run scores both rewards: the rule amender's weights and methods, each reward's method, the intent rubric's
    sampling and the panel's attempts per judgment and retry budget per flush."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    ra_reward_method: str = "standard"
    sg_reward_method: str = "standard"
    intent_temperature: float = _DEFAULTS.intent_temperature
    intent_seed: int = _DEFAULTS.intent_seed
    panel_num_tries: int = _DEFAULTS.panel_num_tries
    panel_max_retry_seconds: Optional[float] = _DEFAULTS.panel_max_retry_seconds

    @classmethod
    def from_config(cls, config: Any) -> "RewardSettings":
        """Return the settings a run's config holds; a config lacking a reward weight or method, or asking for the
        rule-vs-ruleset penalty, is refused."""
        ra, sg, eval_cfg = config.rule_amendment, config.scenario_generation, config.eval_cfg
        given = {field: getattr(ra, key, _ABSENT) for field, key in WEIGHT_KEYS.items()}
        missing = [WEIGHT_KEYS[field] for field, value in given.items() if value is _ABSENT]
        if missing:
            raise ValueError(
                f"the run's config sets no rule_amendment.{', rule_amendment.'.join(missing)}: set each of the "
                f"reward's weights and methods ({', '.join(WEIGHT_KEYS.values())}; RAConfig's defaults are the "
                "paper's, and a task spec's own default does not apply)")
        for key, recorded in RETIRED_PENALTY.items():
            value = getattr(ra, key, _ABSENT)
            if value is not _ABSENT and value != recorded:
                raise ValueError(
                    f"the run's config sets rule_amendment.{key}={value!r}, but this version computes no "
                    f"rule-vs-ruleset penalty: remove the key (or set it to {recorded!r})")
        weights = RewardWeights(
            intent=float(given["intent"]),
            entropy_method=str(given["entropy_method"]),
            intent_method=str(given["intent_method"]),
        )
        return cls(
            weights=weights,
            ra_reward_method=str(getattr(ra, "ra_reward_method", "standard")),
            sg_reward_method=str(getattr(sg, "sg_reward_method", "standard")),
            intent_temperature=float(getattr(eval_cfg, "intent_temperature", _DEFAULTS.intent_temperature)),
            intent_seed=int(getattr(eval_cfg, "intent_seed", _DEFAULTS.intent_seed)),
            panel_num_tries=int(getattr(eval_cfg, "panel_num_tries", _DEFAULTS.panel_num_tries)),
            panel_max_retry_seconds=_panel_budget(eval_cfg),
        )


def ra_reward(task_spec: Any, settings: RewardSettings, *, scenario_set_path: Path, rule_set_path: Path,
              output_dir: Path, backend: "ValidatedJudge") -> RewardDict:
    """Return the rule amender's reward for the ruleset on the scenarios, judgments written under `output_dir`."""
    return compute_ra_reward(
        ra_reward_method=settings.ra_reward_method,
        task_spec=task_spec,
        scenario_set_path=scenario_set_path,
        rule_set_path=rule_set_path,
        output_dir=output_dir,
        backend=backend,
        weights=settings.weights,
        intent_temperature=settings.intent_temperature,
        intent_seed=settings.intent_seed,
        panel_num_tries=settings.panel_num_tries,
        panel_max_retry_seconds=settings.panel_max_retry_seconds,
    )


def sg_reward(task_spec: Any, settings: RewardSettings, *, scenario_set_path: Path, rule_set_path: Path,
              output_dir: Path, backend: "ValidatedJudge") -> RewardDict:
    """Return the scenario generator's reward for the scenarios under the ruleset, judgments written
    under `output_dir`."""
    return compute_sg_reward(
        sg_reward_method=settings.sg_reward_method,
        task_spec=task_spec,
        scenario_set_path=scenario_set_path,
        rule_set_path=rule_set_path,
        output_dir=output_dir,
        backend=backend,
        weights=settings.weights,
        panel_num_tries=settings.panel_num_tries,
        panel_max_retry_seconds=settings.panel_max_retry_seconds,
    )


__all__ = ["WEIGHT_KEYS", "RewardSettings", "RewardWeights", "ra_reward", "sg_reward"]
