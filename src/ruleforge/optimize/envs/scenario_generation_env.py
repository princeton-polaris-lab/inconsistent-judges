from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

from omegaconf import DictConfig

from ruleforge.optimize.scoring import RewardSettings, sg_reward
from ruleforge.tasks.spec import BaseTaskSpec
from ruleforge.rulesets.rules import build_rule_set_string_from_file
from ruleforge.tasks.history import RewardDict

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)


class ScenarioGenerationEnv:
    """Scenario-generation environment: holds the ruleset and scores a scenario set's reward under it."""

    def __init__(
        self,
        task_spec: BaseTaskSpec,
        sg_cfg: DictConfig,
        backend: "ValidatedJudge",
        *,
        rewards: RewardSettings,
    ):
        self.task_spec = task_spec
        self.sg_cfg = sg_cfg
        self.backend = backend
        self.rewards = rewards

        self.reward_history: List[float] = []
        self.rule_set_path: Path | None = None

    def get_obs(self) -> str:
        """Return the current ruleset as text."""
        if self.rule_set_path is None:
            raise ValueError(
                "rule_set_path must be set before calling get_obs(). Call reset() first."
            )
        return build_rule_set_string_from_file(self.rule_set_path)

    def get_reward(self, scenario_set_path: Path, output_dir: Path) -> RewardDict:
        """Return the reward of the scenarios at `scenario_set_path` under the current ruleset."""
        if self.rule_set_path is None:
            raise ValueError("rule_set_path must be set before calling get_reward(). Call reset() first.")
        return sg_reward(
            self.task_spec,
            self.rewards,
            scenario_set_path=scenario_set_path,
            rule_set_path=self.rule_set_path,
            output_dir=output_dir,
            backend=self.backend,
        )

    def reset(self, rule_set_path: Path):
        self.rule_set_path = rule_set_path

    def update_optimal_reward(self, optimal_effective_reward: float) -> None:
        self.reward_history.append(optimal_effective_reward)
        logger.info(f"Updated optimal reward: {optimal_effective_reward}")
        logger.info(f"Reward history: {list(self.reward_history)}")

    def check_termination(self) -> bool:
        """Return True when the last `termination_episodes` rewards are all below the threshold."""
        if len(self.reward_history) < self.sg_cfg.termination_episodes:
            return False
        recent_rewards = list(self.reward_history[-self.sg_cfg.termination_episodes:])
        if all(r < self.sg_cfg.termination_threshold for r in recent_rewards):
            logger.info(
                f"SG termination condition met: scenario generator is failing to generate good scenarios. "
                f"All {self.sg_cfg.termination_episodes} recent rewards below threshold {self.sg_cfg.termination_threshold}"
            )
            return True
        return False
