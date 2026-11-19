from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

from omegaconf import DictConfig

from ruleforge.optimize.scoring import RewardSettings, ra_reward
from ruleforge.tasks.spec import BaseTaskSpec
from ruleforge.tasks.history import RewardDict

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)


class RuleAmendmentEnv:
    """Rule-amendment environment: holds the scenario set and scores a ruleset's reward on it."""

    def __init__(
        self,
        task_spec: BaseTaskSpec,
        ra_cfg: DictConfig,
        backend: "ValidatedJudge",
        *,
        rewards: RewardSettings,
    ):
        self.task_spec = task_spec
        self.ra_cfg = ra_cfg
        self.backend = backend
        self.rewards = rewards

        self.reward_history: List[float] = []
        self.scenario_set_path: Path | None = None

    def get_reward(self, rule_set_path: Path, output_dir: Path) -> RewardDict:
        """Return the reward of the ruleset at `rule_set_path` on the current scenario set."""
        if self.scenario_set_path is None:
            raise ValueError("scenario_set_path must be set before calling get_reward(). Call reset() first.")
        return ra_reward(
            self.task_spec,
            self.rewards,
            rule_set_path=rule_set_path,
            scenario_set_path=self.scenario_set_path,
            output_dir=output_dir,
            backend=self.backend,
        )

    def reset(self, scenario_set_path: Path):
        self.scenario_set_path = scenario_set_path

    def update_optimal_reward(self, optimal_effective_reward: float) -> None:
        self.reward_history.append(optimal_effective_reward)

    def check_termination(self) -> bool:
        """Return True when the last `termination_episodes` rewards all exceed the threshold or strictly decrease."""
        termination_episodes = self.ra_cfg.termination_episodes
        termination_threshold = self.ra_cfg.termination_threshold
        if len(self.reward_history) < termination_episodes:
            return False
        recent_rewards = list(self.reward_history[-termination_episodes:])

        above_threshold = all(r > termination_threshold for r in recent_rewards)
        if above_threshold:
            logger.info(f"Termination condition met: RA is consistently getting high rewards. "
                        f"All {termination_episodes} recent rewards above threshold {termination_threshold}")
            return True

        is_decreasing = all(
            recent_rewards[i] > recent_rewards[i + 1]
            for i in range(len(recent_rewards) - 1)
        )
        if is_decreasing:
            logger.info(f"Termination condition met: RA agent is consistently getting worse. "
                        f"Rewards strictly decreasing for last{termination_episodes} episodes")
            return True
        return False
