from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

import numpy as np
from shutil import copyfile
from omegaconf import DictConfig

from ruleforge.optimize.agents.psrl_rule_amendment import PSRLRuleAmendment
from ruleforge.optimize.agents.psrl_scenario_generation import PSRLScenarioGeneration
from ruleforge.core.constants import OPTIMAL_RULE_SET_FNAME, OPTIMAL_SCENARIO_SET_FNAME, PRIOR_FNAME, POSTERIOR_SAMPLE_FNAME, BATCH_RESULTS_FNAME, REWARD_ALL_FNAME, REWARD_OPTIMAL_FNAME, ACCUMULATED_SCENARIO_SET_FNAME, PATCH_ERROR_FNAME, PATCH_FNAME, PRIOR_TRACE_FNAME, POSTERIOR_SAMPLING_TRACE_FNAME
from ruleforge.optimize.envs.rule_amendment_env import RuleAmendmentEnv
from ruleforge.optimize.envs.scenario_generation_env import ScenarioGenerationEnv
from ruleforge.optimize.config import SEED_PATH_KEYS, seed_paths
from ruleforge.optimize.loggers import NullLogger, RunLogger
from ruleforge.optimize.metrics.intent import IntentScoringError
from ruleforge.optimize.scoring import RewardSettings
from ruleforge.rulesets.rules import duplicate_rule_id_refusal, round_trip_refusal, rule_text_refusal
from ruleforge.tasks.spec import BaseTaskSpec, config_snapshot
from ruleforge.optimize.job_logging import log_with_header
from ruleforge.rulesets.rules import (
    build_rule_set_string_from_file,
    parse_rule_set_text,
    save_rule_set_as_jsonl,
)
from ruleforge.core.files import process_jsonl_file
from ruleforge.optimize.input_output import (
    save_string_to_file,
    save_data_to_jsonl,
    parse_scenarios_to_jsonl,
)
from ruleforge.tasks.history import (
    RATrajectory,
    RewardDict,
    SGSingleExperience,
    SGTrajectory,
)

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)


def _check_inputs(config: DictConfig, task: BaseTaskSpec, rules: Path, seeds: Mapping[str, Path]) -> None:
    built, current = task.built_from, config_snapshot(config)
    if built != current:
        differing = sorted(key for key in set(built) | set(current) if built.get(key) != current.get(key))
        raise ValueError(f"the trainer's task spec was built from another config than the trainer's ({differing} "
                         "differ), or the config changed since: build the task spec from the trainer's config")
    if task.initial_rule_set_path is None:
        raise ValueError(f"the trainer's task spec ({type(task).__name__}) reads no initial ruleset, which the "
                         "intent penalty compares every revision with: run a built-in task's spec")
    configured_rules = Path(config.rule_amendment.initial_rule_set_path)
    configured_seeds = seed_paths(config.scenario_generation)
    roles = {key: role for role, key in SEED_PATH_KEYS.items()}
    held_rules = Path(task.initial_rule_set_path)
    held_seeds = {roles[key]: Path(path) for key, path in task.seed_scenario_paths().items()}
    named_by_config = rules == configured_rules and dict(seeds) == configured_seeds
    read_by_task = rules == held_rules and all(seeds.get(role) == path for role, path in held_seeds.items())
    if not (named_by_config and read_by_task):
        raise ValueError(
            f"the trainer is handed rules {rules} and seeds {dict(seeds)}; its config names {configured_rules} and "
            f"{configured_seeds}, its task spec reads {held_rules} and {held_seeds}: hand it the files both read")


def _refuse_duplicate_ids(rule_ids: List[str], holder: str) -> None:
    refusal = duplicate_rule_id_refusal(rule_ids, holder)
    if refusal is not None:
        raise ValueError(refusal)


def _refuse_changed_rules(rules: List[Dict[str, Any]], holder: str) -> None:
    refusal = round_trip_refusal(rules, holder, exact_ids=False)
    if refusal is not None:
        raise ValueError(refusal)


def check_run_inputs(config: DictConfig, rules: Path) -> RewardSettings:
    """Check what the trainer refuses before it needs a judge (the initial ruleset's ids and text, both termination
    windows, the panel's rule context, the reward settings) and return the reward settings."""
    context = config.eval_cfg.get("individual_rule_context", "isolated")
    if context != "isolated":
        raise ValueError(f"eval_cfg.individual_rule_context={context!r}: the panel judges the whole ruleset in one call "
                         "and never a rule alone, so no single-rule context applies; leave it 'isolated'")
    for section in ("rule_amendment", "scenario_generation"):
        # the window slices the reward history, which takes an integer
        given = config[section].termination_episodes
        if type(given) is not int:
            raise ValueError(f"{section}.termination_episodes={given!r}: it must be an integer")
    window = config.rule_amendment.termination_episodes
    if window < 2:
        raise ValueError(f"rule_amendment.termination_episodes={window}: the amender stops on a strict decrease over "
                         "this window, which takes at least 2 episodes")
    window = config.scenario_generation.termination_episodes
    if window < 1:
        raise ValueError(f"scenario_generation.termination_episodes={window}: the generator stops when this many last "
                         "rewards are all low, which takes at least 1 episode")
    initial_rules = list(process_jsonl_file(rules))
    holder = f"the initial ruleset {rules}"
    _refuse_duplicate_ids([str(row.get("id", "")) for row in initial_rules], holder)
    _refuse_changed_rules(initial_rules, holder)
    return RewardSettings.from_config(config)


class RuleAmendmentTrainer:
    """The episode loop over the judges, task spec and inputs it is handed, writing episodes under `output_dir`;
    `config` is the composed Hydra config; `rules` and `seeds` are the files it names and `task` was built from."""

    def __init__(
        self,
        config: DictConfig,
        *,
        task: BaseTaskSpec,
        judge: "ValidatedJudge",
        agent: "ValidatedJudge",
        rules: Path,
        seeds: Mapping[str, Path],
        output_dir: Path,
        logger: RunLogger | None = None,
    ):
        self.cfg = config
        self.output_dir = Path(output_dir)
        logging.getLogger(__name__).info(f"Working directory : {Path.cwd()}")
        logging.getLogger(__name__).info(f"Output directory  : {self.output_dir}")

        self.ra_cfg = config.rule_amendment
        self.sg_cfg = config.scenario_generation

        self.initial_rule_set_path = Path(rules)
        self.seeds = {role: Path(path) for role, path in seeds.items()}
        _check_inputs(config, task, self.initial_rule_set_path, self.seeds)
        rewards = check_run_inputs(config, self.initial_rule_set_path)

        self.judge = judge
        self.agent = agent
        self.task_spec = task
        self.rewards = rewards
        self.run_logger: RunLogger = logger if logger is not None else NullLogger()

        self.ra_env = RuleAmendmentEnv(
            task_spec=self.task_spec,
            ra_cfg=self.ra_cfg,
            backend=self.judge,
            rewards=self.rewards,
        )
        self.sg_env = ScenarioGenerationEnv(
            task_spec=self.task_spec,
            sg_cfg=self.sg_cfg,
            backend=self.judge,
            rewards=self.rewards,
        )

        self.ra_psrl = PSRLRuleAmendment(
            task_spec=self.task_spec,
            ra_cfg=self.ra_cfg,
            backend=self.agent,
            weights=self.rewards.weights,
        )
        self.sg_psrl = PSRLScenarioGeneration(
            task_spec=self.task_spec,
            sg_cfg=self.sg_cfg,
            backend=self.agent,
        )

        self.accumulated_optimal_scenarios: List[SGSingleExperience] = []
        self.curr_ra_avg_entropy_reward: float = -1.0
        self.episode_num: int = 0
        self.max_episodes = int(self.cfg.max_episodes)
        self.optimal_rule_set_path: Path | None = None
        self.stopped_by: tuple[str, ...] = ()

    def train(self):
        logger.info("Starting training")
        self.episode_num = 0
        optimal_rule_set_path, sg_terminated, ra_terminated = self.handle_episode_zero()
        if self.terminate_training(ra_terminated, sg_terminated):
            return

        for episode in range(1, self.max_episodes + 1):
            self.episode_num = episode
            log_with_header(f"Episode {self.episode_num}")
            episode_dir, sg_dir, ra_dir = self._create_episode_dirs(episode_num=self.episode_num)

            threshold = float(self.sg_cfg.trigger_threshold)
            if self.curr_ra_avg_entropy_reward > threshold:
                logger.info(f"Avg entropy reward ({self.curr_ra_avg_entropy_reward:.4f}) > threshold ({threshold}). Generating new scenarios.")
                self.sg_env.reset(rule_set_path=optimal_rule_set_path)
                sg_terminated = self.sg_step(sg_dir=sg_dir)
            else:
                logger.info(f"Avg entropy reward ({self.curr_ra_avg_entropy_reward:.4f}) <= threshold ({threshold}). Skipping scenario generation.")
            accumulated_scenario_path = self._save_accumulated_scenario_set(episode_dir=episode_dir)

            self.ra_env.reset(scenario_set_path=accumulated_scenario_path)
            self.curr_ra_avg_entropy_reward, optimal_rule_set_path, ra_terminated = self.ra_step(ra_dir=ra_dir, prev_rule_set_path=optimal_rule_set_path)

            if self.terminate_training(ra_terminated, sg_terminated):
                return


    def sg_step(self, sg_dir: Path) -> bool:
        """Run one scenario-generation episode, accumulating batches until the target count or the iteration cap; return
        whether the generator stops."""
        log_with_header(f"Scenario Generation Episode {self.episode_num}")

        high_entropy_scenarios: List[SGSingleExperience] = []
        all_generated_scenarios: List[SGSingleExperience] = []
        batch_idx = 0
        target = int(self.sg_cfg.target_num_scenarios)
        entropy_threshold = float(self.sg_cfg.entropy_threshold)
        max_iters = int(self.sg_cfg.max_accumulation_iterations)

        log_with_header(f"Scenario set iterative generation for episode {self.episode_num} (target: {target} scenarios, threshold: {entropy_threshold})")

        while len(high_entropy_scenarios) < target and batch_idx < max_iters:
            logger.info(
                f"Batch {batch_idx}: {len(high_entropy_scenarios)}/{target} high-entropy scenarios collected, generating more..."
            )

            batch_dir = self._create_batch_dir(sg_dir, batch_idx)

            reward = self._generate_and_evaluate_scenario_batch(
                batch_dir=batch_dir,
                already_generated_scenarios=[s["scenario"] for s in high_entropy_scenarios],
            )
            # per-batch rewards only: averages are taken after accumulation
            scenarios_with_rewards: List[SGSingleExperience] = reward["per_scenario_rewards"]

            all_generated_scenarios.extend(scenarios_with_rewards)

            high_entropy = [
                s for s in scenarios_with_rewards
                if s["entropy_reward"] >= entropy_threshold
            ]
            high_entropy.sort(key=lambda x: x["entropy_reward"], reverse=True)

            logger.info(f"Batch {batch_idx}: {len(high_entropy)}/{len(scenarios_with_rewards)} passed entropy threshold ({entropy_threshold})")

            for scenario in high_entropy:
                if len(high_entropy_scenarios) >= target:
                    break
                high_entropy_scenarios.append(scenario)

            save_data_to_jsonl(
                {
                    "batch_idx": batch_idx,
                    "total_generated": len(scenarios_with_rewards),
                    "passed_threshold": len(high_entropy),
                    "high_entropy_total": len(high_entropy_scenarios),
                    "entropy_threshold": entropy_threshold,
                },
                batch_dir / BATCH_RESULTS_FNAME,
            )
            batch_idx += 1

        if len(high_entropy_scenarios) < target:
            logger.warning(f"Reached max iterations ({max_iters}) with only {len(high_entropy_scenarios)}/{target} above threshold. Falling back to top {target} by entropy.")
            if not all_generated_scenarios:
                raise RuntimeError(f"Failed to generate any scenarios after {batch_idx} batches.")
            all_generated_scenarios.sort(key=lambda x: x["entropy_reward"], reverse=True)
            high_entropy_scenarios = all_generated_scenarios[:target]
            logger.info(f"Selected top {len(high_entropy_scenarios)} scenarios by entropy reward")

        log_with_header(f"Iterative generation complete: {len(high_entropy_scenarios)} scenarios in {batch_idx} batches")

        out_path = sg_dir / OPTIMAL_SCENARIO_SET_FNAME
        scenario_set = [s["scenario"] for s in high_entropy_scenarios]
        save_data_to_jsonl(scenario_set, out_path)

        def _create_scenario_set_log_dict(scenarios: List[SGSingleExperience], num_batches: int) -> Dict[str, Any]:
            def _get_average(key: str, scenarios: List[SGSingleExperience]) -> float:
                return float(np.mean([s[key] for s in scenarios])) if scenarios else 0.0
            return {
                "average_entropy_reward": _get_average("entropy_reward", scenarios),
                "average_effective_reward": _get_average("effective_reward", scenarios),
                "average_penalty": _get_average("penalty", scenarios),
                "num_batches": num_batches,
                "scenario_count": len(scenarios),
                "judgment_count": sum(s["judgment_count"] for s in scenarios),
                "scenario_list": list(scenarios),
            }

        out_path_optimal = sg_dir / REWARD_OPTIMAL_FNAME
        optimal_scenarios_dict = _create_scenario_set_log_dict(high_entropy_scenarios, batch_idx)
        save_data_to_jsonl(optimal_scenarios_dict, out_path_optimal)

        out_path_all = sg_dir / REWARD_ALL_FNAME
        all_scenarios_dict = _create_scenario_set_log_dict(all_generated_scenarios, batch_idx)
        save_data_to_jsonl(all_scenarios_dict, out_path_all)

        self.run_logger.log(
            {
                "optimal_scenarios": optimal_scenarios_dict,
                "all_scenarios": all_scenarios_dict,
            },
            step=self.episode_num,
        )

        trajectory = SGTrajectory(
            state=self.sg_env.get_obs(),
            accumulated_scenarios=high_entropy_scenarios,
            all_generated=all_generated_scenarios,
            reward=optimal_scenarios_dict["average_effective_reward"],
        )
        self.sg_psrl.update_history(trajectory)

        self.sg_env.update_optimal_reward(optimal_scenarios_dict["average_effective_reward"])
        terminated = self.sg_env.check_termination()

        logger.info(f"Scenario set posterior update for episode {self.episode_num}")
        posterior = self.sg_psrl.posterior_update(trace_path=sg_dir / PRIOR_TRACE_FNAME)
        out_path = sg_dir / PRIOR_FNAME
        save_data_to_jsonl({"prior": posterior}, out_path)
        logger.info(f"Posterior saved to {out_path}")
        logger.debug(f"Posterior: {posterior}")

        self.accumulated_optimal_scenarios.extend(high_entropy_scenarios)
        logger.info(
            f"Added {len(high_entropy_scenarios)} scenarios from episode {self.episode_num}. "
            f"Total accumulated: {len(self.accumulated_optimal_scenarios)}"
        )

        return terminated

    def _generate_and_evaluate_scenario_batch(
        self,
        batch_dir: Path,
        already_generated_scenarios: List[Dict[str, Any]],
    ) -> RewardDict:
        scenario_set_path = batch_dir / POSTERIOR_SAMPLE_FNAME

        sampled_scenarios_str = self.sg_psrl.posterior_sampling(
            current_rule_set=self.sg_env.get_obs(),
            already_generated_scenarios=already_generated_scenarios,
            trace_path=batch_dir / POSTERIOR_SAMPLING_TRACE_FNAME,
        )
        parse_scenarios_to_jsonl(self.task_spec.scenario_block_pattern, sampled_scenarios_str, scenario_set_path)
        logger.info(f"Scenarios posterior sample saved to {scenario_set_path}")
        
        return self.sg_env.get_reward(scenario_set_path=scenario_set_path, output_dir=batch_dir)

    def ra_step(self, ra_dir: Path, prev_rule_set_path: Path) -> tuple[float, Path, bool]:
        """Run one rule-amendment episode; return (average entropy reward, optimal ruleset path, terminated)."""
        log_with_header(f"Rule Amendment Episode {self.episode_num}")

        logger.info(f"Rule set posterior sampling for episode {self.episode_num}")

        posterior_samples = self.ra_psrl.posterior_sampling(
            prev_rule_set_path=prev_rule_set_path,
            trace_path=ra_dir / POSTERIOR_SAMPLING_TRACE_FNAME,
        )
        ruleset_dirs: Dict[int, Path] = {}
        for i, patch in enumerate(posterior_samples):
            ruleset_dir = self._create_batch_dir(ra_dir, i)
            posterior_patch_path = ruleset_dir / PATCH_FNAME
            save_string_to_file(patch, posterior_patch_path)
            logger.info(f"Patch saved to {posterior_patch_path}")

            try:
                new_rule_set = self._patched_rule_set(prev_rule_set_path, posterior_patch_path,
                                                      unfinished=self.ra_psrl.unfinished_samples.get(i))
            except ValueError as error:
                logger.warning(f"Skipping candidate {i}: {error}")
                save_string_to_file(str(error), ruleset_dir / PATCH_ERROR_FNAME)
                continue
            save_rule_set_as_jsonl(new_rule_set, ruleset_dir / POSTERIOR_SAMPLE_FNAME)
            logger.debug(f"Rule set {i} saved to {ruleset_dir}. New rule set: {new_rule_set}")

            ruleset_dirs[i] = ruleset_dir

        optimal_reward_dict, optimal_rule_set_path, terminated = self._evaluate_rule_sets_and_update_history(
            ra_dir=ra_dir, rule_set_dirs=ruleset_dirs, previous=(len(posterior_samples), prev_rule_set_path))

        logger.info(f"Rule amendment posterior update for episode {self.episode_num}")
        posterior = self.ra_psrl.posterior_update(trace_path=ra_dir / PRIOR_TRACE_FNAME)
        out_path = ra_dir / PRIOR_FNAME
        save_data_to_jsonl({"prior": posterior}, out_path)
        logger.info(f"Posterior saved to {out_path}")
        logger.debug(f"Posterior: {posterior}")

        return optimal_reward_dict["entropy_reward"], optimal_rule_set_path, terminated

    def _patched_rule_set(self, prev_rule_set_path: Path, patch_path: Path, *, unfinished: str | None) -> str:
        if unfinished is not None:
            raise ValueError(f"the ruleset patch {patch_path} {unfinished}")
        new_rule_set = self.ra_psrl.apply_patch_to_jsonl(prev_rule_set_path, patch_path)
        refusal = rule_text_refusal(new_rule_set, f"the ruleset patch {patch_path}")
        if refusal is not None:
            raise ValueError(refusal)
        rule_ids = [rule_id for rule_id, _ in parse_rule_set_text(new_rule_set)]
        if not rule_ids:
            raise ValueError(f"the ruleset patch {patch_path} leaves no rule")
        _refuse_duplicate_ids(rule_ids, f"the ruleset patch {patch_path} makes")
        return new_rule_set

    def _score_candidates(self, rule_set_dirs: Mapping[int, Path]) -> tuple[Dict[int, RewardDict], List[str]]:
        reward_map: Dict[int, RewardDict] = {}
        errors: List[str] = []
        for idx, rule_set_dir in rule_set_dirs.items():
            rule_set_path = rule_set_dir / POSTERIOR_SAMPLE_FNAME
            try:
                reward_map[idx] = self.ra_env.get_reward(rule_set_path=rule_set_path, output_dir=rule_set_dir)
            except IntentScoringError as error:
                logger.warning(f"Skipping candidate {idx}: {error}")
                save_string_to_file(str(error), rule_set_dir / PATCH_ERROR_FNAME)
                errors.append(str(error))
        return reward_map, errors

    def _evaluate_rule_sets_and_update_history(
        self,
        ra_dir: Path,
        rule_set_dirs: dict[int, Path],
        previous: tuple[int, Path] | None = None,
    ) -> tuple[RewardDict, Path, bool]:
        """Score the candidates and keep the best; with no candidate left, `previous` (its sample index and ruleset)
        stands, scored on this episode's scenarios."""
        logger.info(f"Rule set evaluation for episode {self.episode_num}")
        reward_map, errors = self._score_candidates(rule_set_dirs)

        if not reward_map and previous is not None:
            logger.warning(f"Episode {self.episode_num}: no candidate applied and scored; keeping the previous ruleset")
            index, previous_path = previous
            rule_set_dirs[index] = self._create_batch_dir(ra_dir, index)
            copyfile(previous_path, rule_set_dirs[index] / POSTERIOR_SAMPLE_FNAME)
            reward_map, errors = self._score_candidates({index: rule_set_dirs[index]})
        if not reward_map:
            raise ValueError(f"no candidate of episode {self.episode_num} could be scored: {errors[-1]}")

        save_data_to_jsonl(reward_map, ra_dir / REWARD_ALL_FNAME)

        for idx, reward_dict in reward_map.items():
            self.run_logger.log({"rule_set_idx": idx, "reward": reward_dict}, step=self.episode_num)

        optimal_idx = self.ra_psrl.choose_optimal_rule_set(reward_map=reward_map)
        optimal_posterior_path = rule_set_dirs[optimal_idx] / POSTERIOR_SAMPLE_FNAME
        optimal_rule_set_path = ra_dir / OPTIMAL_RULE_SET_FNAME
        copyfile(optimal_posterior_path, optimal_rule_set_path)
        self.optimal_rule_set_path = optimal_rule_set_path
        optimal_reward_dict: RewardDict = reward_map[optimal_idx]
        save_data_to_jsonl(
            {
                "optimal_rule_set_index": optimal_idx,
                "reward": optimal_reward_dict,
            },
            ra_dir / REWARD_OPTIMAL_FNAME,
        )

        trajectory = RATrajectory(
            action=build_rule_set_string_from_file(optimal_rule_set_path),
            reward=optimal_reward_dict,
        )
        self.ra_psrl.update_history(trajectory)

        self.ra_env.update_optimal_reward(optimal_reward_dict["effective_reward"])

        terminated = self.ra_env.check_termination()
        return optimal_reward_dict, optimal_rule_set_path, terminated

    def handle_episode_zero(self) -> tuple[Path, bool, bool]:
        """Generate the initial scenarios and evaluate the initial ruleset; return (optimal ruleset path, sg_terminated,
        ra_terminated)."""
        episode_zero_dir, sg_dir, ra_dir, ruleset_dir, rule_set_path = self._initialize_episode_zero_dirs()
        log_with_header("Initial scenario generation")

        self.sg_env.reset(rule_set_path=rule_set_path)
        sg_terminated = self.sg_step(sg_dir=sg_dir)

        accumulated_scenario_path = self._save_accumulated_scenario_set(episode_dir=episode_zero_dir)

        log_with_header("Initial rule set evaluation")
        self.ra_env.reset(scenario_set_path=accumulated_scenario_path)

        rule_set_dirs: dict[int, Path] = {0: ruleset_dir}

        optimal_reward_dict, optimal_rule_set_path, ra_terminated = self._evaluate_rule_sets_and_update_history(ra_dir=ra_dir, rule_set_dirs=rule_set_dirs)
        logger.info(f"Initial rule set reward dict:\n\t{optimal_reward_dict}")

        self.curr_ra_avg_entropy_reward = optimal_reward_dict["entropy_reward"]

        return optimal_rule_set_path, sg_terminated, ra_terminated

    def terminate_training(self, ra_terminated: bool, sg_terminated: bool) -> bool:
        if ra_terminated or sg_terminated:
            log_with_header(f"Episode {self.episode_num} terminated (RA={ra_terminated}, SG={sg_terminated})")
            self.stopped_by = tuple(
                name for name, fired in (("rule_amendment", ra_terminated), ("scenario_generation", sg_terminated))
                if fired
            )
            return True
        if self.episode_num >= self.max_episodes:
            log_with_header(f"Episode {self.episode_num} terminated (reached max_episodes ({self.max_episodes}))")
            self.stopped_by = ("max_episodes",)
            return True
        return False

    def _initialize_episode_zero_dirs(self) -> tuple[Path, Path, Path, Path, Path]:
        episode_zero_dir, sg_dir, ra_dir = self._create_episode_dirs(episode_num=0)
        rule_set_string = build_rule_set_string_from_file(self.initial_rule_set_path)

        ruleset_dir = self._create_batch_dir(ra_dir, 0)
        rule_set_path = ruleset_dir / POSTERIOR_SAMPLE_FNAME
        save_rule_set_as_jsonl(rule_set_string, rule_set_path)

        sg_prior_path = sg_dir / PRIOR_FNAME
        save_data_to_jsonl({"prior": self.task_spec.get_sg_prior()}, sg_prior_path)
        logger.info(f"Scenario generation prior saved to {sg_prior_path}")

        ra_prior_path = ra_dir / PRIOR_FNAME
        save_data_to_jsonl({"prior": self.task_spec.get_ra_prior(self.rewards.weights)}, ra_prior_path)
        logger.info(f"Rule amendment prior saved to {ra_prior_path}")

        return episode_zero_dir, sg_dir, ra_dir, ruleset_dir, rule_set_path

    def _create_episode_dirs(self, episode_num: int) -> tuple[Path, Path, Path]:
        episode_dir = self.output_dir / f"episode_{episode_num}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        sg_dir = episode_dir / "scenario_generation"
        sg_dir.mkdir(parents=True, exist_ok=True)
        ra_dir = episode_dir / "rule_amendment"
        ra_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir, sg_dir, ra_dir

    def _create_batch_dir(self, parent_dir: Path, batch_idx: int) -> Path:
        batch_dir = parent_dir / f"sample_{batch_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    def _save_accumulated_scenario_set(self, episode_dir: Path) -> Path:
        accumulated_path = episode_dir / ACCUMULATED_SCENARIO_SET_FNAME
        accumulated_scenario_set = [experience["scenario"] for experience in self.accumulated_optimal_scenarios]
        save_data_to_jsonl(accumulated_scenario_set, accumulated_path)
        logger.info(f"Saved {len(accumulated_scenario_set)} accumulated scenarios to {accumulated_path}")
        return accumulated_path
