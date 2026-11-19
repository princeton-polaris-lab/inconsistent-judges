from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from omegaconf import DictConfig

from ruleforge.optimize.agents.traces import generate_finished, generate_traced, revised_prior
from ruleforge.tasks.spec import BaseTaskSpec
from ruleforge.tasks.history import RATrajectory, RewardDict, RewardWeights
from ruleforge.rulesets.rules import build_rule_set_string_from_file
from ruleforge.optimize.input_output import read_string_from_file
from ruleforge.optimize.search_replace import SearchAndReplace

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)


class PSRLRuleAmendment:
    """Rule-amendment agent: samples search/replace patches from a free-text prior it revises each episode."""

    def __init__(
        self,
        task_spec: BaseTaskSpec,
        ra_cfg: DictConfig,
        backend: "ValidatedJudge",
        *,
        weights: RewardWeights,
    ):
        self.task_spec = task_spec
        self.ra_cfg = ra_cfg
        self.backend = backend
        self.weights = weights

        self.history: List[RATrajectory] = []
        self.prior: str = self.task_spec.get_ra_prior(self.weights)
        self.search_and_replace = SearchAndReplace()
        self.unfinished_samples: Dict[int, str] = {}

    def posterior_update(self, trace_path: Optional[Path] = None) -> str:
        """Rewrite the prior from the agent's history and return it; the call's trace goes to `trace_path`."""
        system_prompt = f"""# Task
You are a Bayesian posterior distribution for a real-world sequential decision-making problem. Given a current prior belief about the environment and a history of agent trajectory observations, you should produce the posterior distribution that accurately reflects knowledge about possibly stochastic environment rewards and how to optimize the overall reward based on the observed history of the agent's interactions with the environment.

# Output Format
In practice, updating the posterior distribution means revising the current knowledge in the prior. You must follow the exact structure of the prior and cannot add new sections. You may edit existing conditions for the optimal set of rules, add new conditions, or remove existing conditions. You are encouraged to edit the conditions for the optimal set of rules. You are not allowed to add any rules from the rule set to the posterior distribution. You must only output the posterior distribution and not mention that your output is a posterior distribution.

# Definitions
The reward consists of a positive entropy reward, which increases as the ambiguity of the rule set decreases, and a negative regularization term which may consist of multiple penalties. The history of the agent's interactions with the environment is made up of a sequence of trajectories denoted by XML <TRAJECTORY_i> </TRAJECTORY_i> tags, where i is the index of the trajectory. Each trajectory contains an <ACTION> tag with the rule set proposed by the agent, followed by a <REWARD> tag with overall ruleset-level rewards (total reward, entropy reward, and penalties). The most recent trajectory also includes an <EXPERIENCES> block containing per-scenario details with <STATE> (prompt and response) and <REWARD> (scenario-level entropy reward) tags. The overall ruleset-level reward is the average of the scenario-level entropy reward minus the regularization penalty. <TRAJECTORY_0> contains the original ruleset as the <ACTION>. If any rules are deleted from the original ruleset by the agent, the intent penalty is automatically set to 1.0 for that trajectory.

# Posterior Distribution Conditions
The posterior distribution should take into account all information provided in the observed trajectory to update the prior belief about the environment. Be direct and clear. Do not say anything beyond providing the posterior distribution. The agent's interactions with the environment will generate rewards and the posterior distribution should capture what can be inferred from the agent's interactions with the environment and the rewards observed. Information and knowledge in the current prior belief about the environment should be updated as needed but never entirely discarded from the posterior distribution. If there is knowledge in the current prior belief about the environment that is unaffected by the trajectory observation, then this knowledge should not be changed and must be repeated exactly in the posterior distribution. Do not say anything to distinguish between old knowledge that is being retained and updated knowledge.

# Environment Description Given to the Agent
{self.task_spec.get_ra_environment_description()}"""

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
            temperature=float(self.ra_cfg.posterior_update_temperature),
        )

        posterior = revised_prior("RA", self.prior, answers, unfinished)

        logger.info("Rule set posterior updated")
        logger.debug(f"Posterior: {posterior}")
        self.prior = posterior
        return posterior

    def posterior_sampling(
        self,
        prev_rule_set_path: Path,
        trace_path: Optional[Path] = None,
    ) -> List[str]:
        """Return `best_of_n` candidate patches; the patch at index i is the trace's candidate with `result_index` i, and
        `unfinished_samples` says why a patch at an index must not be read."""
        prev_rule_set_str = build_rule_set_string_from_file(prev_rule_set_path)

        system_prompt = f"""# Task
You are an expert at revising rules trying to aid an agent in revising a rule set to be optimal. Given all of the knowledge you currently have about the optimal set of rules, you must generate a revision of the rule set that you think is optimal, i.e. one that maximizes the overall reward. To produce this revision, you must diligently read through all of the input information.

All rules about rules must be either (1) included as meta-rules (with format Meta-Rule <number>: <rule text>) or (2) written in a regular rule's text without the meta-rule label (e.g., Rule 1: Be helpful except when Rule 2 is violated). Meta-rules must come before other regular rules.

# Output Format
You must only output the revision of the rule set as a *SEARCH/REPLACE block* defined using the rules below. You cannot output anything other than the *SEARCH/REPLACE block*.
    ## *SEARCH/REPLACE block* Rules

    Every *SEARCH/REPLACE block* must use this format:
    1. The start of search block: <<<<<<< SEARCH
    2. A contiguous chunk of lines to search for in the existing rule set
    3. The dividing line: =======
    4. The lines to replace into the rule set
    5. The end of the replace block: >>>>>>> REPLACE

    Every *SEARCH* section must *EXACTLY MATCH* text in the previous rule set, character for character.
    The *SEARCH section cannot be empty.
    If the rule set contains data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the rule set, including the container markup.

    *SEARCH/REPLACE* blocks will *only* replace the first match occurrence.
    Include multiple unique *SEARCH/REPLACE* blocks if needed.
    Include enough lines in each SEARCH section to uniquely match each set of lines that need to change.

    Keep *SEARCH/REPLACE* blocks concise.
    Break large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the rule set.
    Include just the changing lines, and a few surrounding lines if needed for uniqueness.
    Do not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.

    To move text within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.

    If you have multiple *SEARCH/REPLACE* blocks, make sure that subsequent *SEARCH/REPLACE* blocks account for any changes made by previous *SEARCH/REPLACE* blocks.

# Environment Description Given to the Agent
{self.task_spec.get_ra_environment_description()}"""

        user_prompt = f"""# Previous Rule Set
{prev_rule_set_str}
# Agent's Current Knowledge About the Optimal Rule Set
{self.prior}

# History of Agent's Interactions with the Environment
{self.get_history_string()}"""

        prompt_batch = [{"system_prompt": system_prompt, "user_prompt": user_prompt}]
        best_of_n = int(self.ra_cfg.best_of_n)
        answers, self.unfinished_samples = generate_traced(
            self.backend,
            "posterior_sampling",
            prompt_batch,
            trace_path,
            temperature=float(self.ra_cfg.posterior_sampling_temperature),
            num_responses=best_of_n,
        )
        if not answers:
            raise ValueError("No output from posterior_sampling generation in RA agent")

        logger.info(f"Posterior sampling answers: {len(answers)} of the expected {best_of_n}")
        return answers

    def apply_patch_to_jsonl(
        self, prev_jsonl_path: Path, patch_path: Path
    ) -> str:
        """Apply a patch to a ruleset file and return the patched text."""
        text = build_rule_set_string_from_file(prev_jsonl_path)
        patch = read_string_from_file(patch_path)
        for search_text, replace_text in self.search_and_replace.extract_multiple_search_replace_blocks(patch):
            text = self.search_and_replace.search_replace(text, search_text, replace_text)
        return text

    def choose_optimal_rule_set(
        self,
        reward_map: Dict[int, RewardDict],
    ) -> int:
        """Return the index of the candidate with the highest effective reward."""

        effective_reward_map = {
            idx: reward["effective_reward"]
            for idx, reward in reward_map.items()
        }
        optimal_idx = max(effective_reward_map, key=effective_reward_map.get)
        return optimal_idx

    def update_history(self, trajectory: RATrajectory) -> None:
        """Append a completed trajectory to the agent's history."""
        self.history.append(trajectory)

    def get_history_string(self) -> str:
        """Return the history as the task spec formats it for the prompt."""
        return self.task_spec.build_ra_history_string(history=self.history, weights=self.weights)
