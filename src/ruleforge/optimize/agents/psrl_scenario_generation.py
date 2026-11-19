from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from omegaconf import DictConfig

from ruleforge.optimize.agents.traces import generate_finished, revised_prior
from ruleforge.tasks.spec import BaseTaskSpec
from ruleforge.tasks.history import SGTrajectory

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)


class PSRLScenarioGeneration:
    """Scenario-generation agent: samples scenario batches from a free-text prior it revises each episode."""

    def __init__(
        self,
        task_spec: BaseTaskSpec,
        sg_cfg: DictConfig,
        backend: "ValidatedJudge",
    ):
        self.task_spec = task_spec
        self.sg_cfg = sg_cfg
        self.backend = backend
        # the seed examples follow from the agent's seed, the one a seed sweep varies
        self.example_rng = random.Random(backend.seed)
        # a batch that keeps no scenario leaves the next batch's prompt unchanged, so each batch needs its own seed
        self.batch_seed_rng = random.Random(backend.seed)

        self.history: List[SGTrajectory] = []
        self.prior: str = self.task_spec.get_sg_prior()

    def posterior_update(self, trace_path: Optional[Path] = None) -> str:
        """Rewrite the prior from the agent's history and return it; the call's trace goes to `trace_path`."""
        system_prompt = f"""# Task
You are a Bayesian posterior distribution for a real-world sequential decision-making problem. Given a current prior belief about the environment and single trajectory observation, you should produce the posterior distribution that accurately reflects knowledge about possibly stochastic environment transitions and environment rewards based on the observed history.

# Output Format
In practice, updating the posterior distribution means revising the current knowledge in the prior. You may edit, add, or remove any knowledge about the optimal set of scenarios. You are encouraged to edit the knowledge about the optimal set of scenarios. You may edit, add, or remove any knowledge about the reward function or stochasticity. You are not allowed to add any scenarios from the scenario set to the posterior distribution. You should not state explicit values for probabilities. You should not state or imply that scenario content is not related to reward magnitude. You must only output the posterior distribution and not mention that your output is a posterior distribution.

# Definitions
The history of the agent's interactions with the environment is made up of a sequence of trajectories denoted by XML <TRAJECTORY_i> </TRAJECTORY_i> tags, where i is the index of the trajectory. The most recent iteration's trajectory observation and all previous trajectories are a sequence experiences, denoted by XML <EXPERIENCE> </EXPERIENCE> tags, of action and reward tuples, respectively wrapped in XML <ACTION> </ACTION> and <REWARD> </REWARD> tags. The most recent iteration's trajectory also contains a state, wrapped in XML <STATE> </STATE> tags, that is shared by all action and reward tuples in the most recent trajectory. You are not provided the state for any previous trajectories.

# Posterior Distribution Conditions
The posterior distribution must always be complete and describe all sources of uncertainty the agent has about the world. There can be uncertainty about a stochastic transition or reward. The posterior distribution should take into account all information provided in the observed trajectory to update the prior belief about the environment. Be direct and don't show your work. You cannot make any assumptions about the agent and the action selections used to generate the trajectory observation. Never try to model beliefs about the agent. Do not say anything beyond providing the posterior distribution. The agent's interactions with the environment will generate rewards and the posterior distribution should keep track of how any and all rewards are generated. Information and knowledge in the current prior belief about the environment should never be discarded from the posterior distribution. If there is knowledge in the current prior belief about the environment that is unaffected by the trajectory observation, then this knowledge should not be changed and must be repeated exactly in the posterior distribution. Do not say anything to distinguish between old knowledge that is being retained and updated knowledge.

# Environment Description Given to the Agent
{self.task_spec.get_sg_environment_description()}"""

        user_prompt = f"""# Current Prior
{self.prior}

# History of Agent's Interactions with the Environment
{self.get_history_string()}"""

        prompt_batch = [{"system_prompt": system_prompt, "user_prompt": user_prompt}]
        answers, unfinished = generate_finished(
            self.backend,
            "posterior_update",
            prompt_batch,
            trace_path,
            temperature=float(self.sg_cfg.posterior_update_temperature),
        )

        posterior = revised_prior("SG", self.prior, answers, unfinished)
        self.prior = posterior
        return posterior

    def posterior_sampling(
        self,
        current_rule_set: str,
        already_generated_scenarios: List[Dict[str, Any]],
        trace_path: Optional[Path] = None,
    ) -> str:
        """Return one batch of scenarios as text, given the current ruleset and the scenarios already generated; "" for a
        batch whose attempts all failed to finish."""
        system_prompt = self.task_spec.build_sg_posterior_sampling_system_prompt(
            already_generated_scenarios=already_generated_scenarios,
            history=self.history,
            rng=self.example_rng,
        )
        user_prompt = f"""# Agent's Current Knowledge About the Optimal Scenario Set
{self.prior}

# Current Rule Set
{current_rule_set}

# History of Agent's Interactions with the Environment
{self.get_history_string()}"""

        prompt_batch = [{"system_prompt": system_prompt, "user_prompt": user_prompt}]
        answers, unfinished = generate_finished(
            self.backend,
            "posterior_sampling",
            prompt_batch,
            trace_path,
            seed=self.batch_seed_rng.randrange(2**31),
            temperature=float(self.sg_cfg.posterior_sampling_temperature),
        )

        if not answers:
            raise ValueError("No output from posterior_sampling generation in SG agent")
        if 0 in unfinished:
            logger.warning(f"A scenario batch {unfinished[0]}; reading no scenarios from it")
            return ""

        return answers[0]

    def update_history(self, trajectory: SGTrajectory) -> None:
        """Append a completed trajectory to the agent's history."""
        self.history.append(trajectory)

    def get_history_string(self) -> str:
        """Return the history as the task spec formats it for the prompt."""
        return self.task_spec.build_sg_history_string(history=self.history)
