from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List
import logging

from ruleforge.tasks.spec import BaseTaskSpec
from ruleforge.tasks.history import (
    PenaltyBreakdownEntry,
    RASingleExperience,
    RewardDict,
    RewardWeights,
    SGSingleExperience,
)

from ruleforge.optimize.judging import run_judgments
from ruleforge.optimize.metrics.entropy import compute_entropy_from_evals, get_scenario_full_ruleset_entropy
from ruleforge.optimize.metrics.intent import DEFAULT_INTENT_SEED, DEFAULT_INTENT_TEMPERATURE, compute_intent_penalty

from ruleforge.core.constants import (
    EVALUATIONS_FNAME,
    ENTROPY_FNAME,
    FULL_RULESET_KEY,
    JUDGE_OUTPUTS_FNAME,
)

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)

def aggregate_reward(
    entropy_reward: float,
    penalty_breakdown: Dict[str, PenaltyBreakdownEntry],
    per_scenario_rewards: List[RASingleExperience] | List[SGSingleExperience],
    judgment_count: int,
) -> RewardDict:
    """Return the reward dict: `effective_reward = entropy_reward - sum(penalty * weight)`, with the per-scenario rewards,
    the penalty breakdown and the count of panel judgments the entropy reward is computed from."""
    weighted_sum = sum(p["value"] * p["weight"] for p in penalty_breakdown.values())
    effective_reward = entropy_reward - weighted_sum

    result: RewardDict = {
        "entropy_reward": entropy_reward,
        "regularization_penalty": weighted_sum,
        "effective_reward": effective_reward,
        "per_scenario_rewards": per_scenario_rewards,
        "penalty_breakdown": penalty_breakdown,
        "judgment_count": judgment_count,
    }

    return result


def _full_ruleset_judgments(record: Dict[str, Any]) -> List[int]:
    """Return how many full-ruleset judgments each model's panel gave a scenario's entropy record."""
    return [counts.get(FULL_RULESET_KEY, 0) for counts in record.get("judgment_count", {}).values()]

def _scored_entropy(
    task_spec: BaseTaskSpec,
    scenario_set_path: Path,
    rule_set_path: Path,
    output_dir: Path,
    backend: "ValidatedJudge",
    *,
    mode: str,
    rule_context: str,
    entropy_method: str,
    label: str,
    num_tries: int,
    max_retry_seconds: float | None,
) -> tuple[list, list, int]:
    """Judge the scenarios and return their entropy records, the scored ``(record, entropy)`` pairs of the scenarios
    whose panel is complete, and the count of judgments those hold."""
    evaluation_path = output_dir / EVALUATIONS_FNAME
    run_judgments(
        scenario_set_path=scenario_set_path,
        rule_set_path=rule_set_path,
        backend=backend,
        task_spec=task_spec,
        judge_types=task_spec.judge_types,
        evaluation_mode=mode,
        output_path=evaluation_path,
        save_batch_size=task_spec.save_batch_size,
        evaluation_temperature=task_spec.evaluation_temperature,
        individual_rule_context=rule_context,
        raw_output_path=output_dir / JUDGE_OUTPUTS_FNAME,
        num_tries=num_tries,
        max_retry_seconds=max_retry_seconds,
    )
    entropy_records = compute_entropy_from_evals(
        evaluations_path=evaluation_path,
        output_path=output_dir / ENTROPY_FNAME,
        parse_fn=task_spec.parse_judgment,
        entropy_method=entropy_method,
    )
    # A judgment the panel lost is no vote: a panel missing one would score fewer voices as more agreement.
    panel_size = len(task_spec.judge_types)
    scored = [
        (r, value)
        for r in entropy_records
        if all(count == panel_size for count in _full_ruleset_judgments(r))
        and (value := get_scenario_full_ruleset_entropy(r)) is not None
    ]
    if len(scored) < len(entropy_records):
        logger.warning(
            "%s: %d of %d scenarios had an incomplete panel (fewer than %d full-ruleset judgments) and were left "
            "out of the reward.",
            label,
            len(entropy_records) - len(scored),
            len(entropy_records),
            panel_size,
        )
    return entropy_records, scored, sum(sum(_full_ruleset_judgments(r)) for r, _ in scored)


def compute_ra_reward_standard(
    task_spec: BaseTaskSpec,
    scenario_set_path: Path,
    rule_set_path: Path,
    output_dir: Path,
    backend: "ValidatedJudge",
    *,
    weights: RewardWeights,
    intent_temperature: float = DEFAULT_INTENT_TEMPERATURE,
    intent_seed: int = DEFAULT_INTENT_SEED,
    panel_num_tries: int = 1,
    panel_max_retry_seconds: float | None = None,
) -> RewardDict:
    """Return the rule amender's reward (judge the scenarios, score entropy and the intent penalty, aggregate) under
    the panel's attempts per judgment and retry budget per flush that a run's config sets (`RewardSettings`)."""

    evaluation_path = output_dir / EVALUATIONS_FNAME
    entropy_records, scored, judgment_count = _scored_entropy(
        task_spec, scenario_set_path, rule_set_path, output_dir, backend,
        mode="full", rule_context=task_spec.individual_rule_context,
        entropy_method=weights.entropy_method, label="RA reward", num_tries=panel_num_tries,
        max_retry_seconds=panel_max_retry_seconds,
    )
    if not entropy_records:
        # an empty scenario set must not score as perfect consensus
        logger.warning(f"No entropy records for {evaluation_path}; RA entropy reward set to 0.0")
    entropy_reward = (
        1.0 - sum(value for _, value in scored) / len(scored) if scored else 0.0
    )

    intent_penalty = 0.0
    intent_result = {}
    if weights.intent > 0:
        intent_result = compute_intent_penalty(
            initial_rule_set_path=task_spec.initial_rule_set_path,
            revised_rule_set_path=rule_set_path,
            backend=backend,
            intent_method=weights.intent_method,
            output_dir=output_dir,
            temperature=intent_temperature,
            seed=intent_seed,
        )
        intent_penalty = intent_result["penalty"]

    per_scenario_rewards: List[RASingleExperience] = [
        RASingleExperience(scenario=record["scenario"], entropy_reward=1.0 - value)
        for record, value in scored
    ]

    penalty_breakdown = {}
    if weights.intent > 0:
        penalty_breakdown["intent"] = PenaltyBreakdownEntry(value=intent_penalty, weight=weights.intent, details=intent_result)
    reward_dict = aggregate_reward(
        entropy_reward=entropy_reward,
        penalty_breakdown=penalty_breakdown,
        per_scenario_rewards=per_scenario_rewards,
        judgment_count=judgment_count,
    )

    logger.info(f"Rule Amendment Reward: {reward_dict}")
    return reward_dict


def compute_sg_reward_standard(
    task_spec: BaseTaskSpec,
    scenario_set_path: Path,
    rule_set_path: Path,
    output_dir: Path,
    backend: "ValidatedJudge",
    *,
    weights: RewardWeights,
    panel_num_tries: int = 1,
    panel_max_retry_seconds: float | None = None,
) -> RewardDict:
    """Return the scenario generator's reward, the mean full-ruleset entropy, under the panel's attempts per judgment
    and retry budget per flush that a run's config sets (`RewardSettings`)."""
    _, scored, judgment_count = _scored_entropy(
        task_spec, scenario_set_path, rule_set_path, output_dir, backend,
        mode="full", rule_context="isolated", entropy_method=weights.entropy_method, label="SG reward",
        num_tries=panel_num_tries, max_retry_seconds=panel_max_retry_seconds,
    )
    entropy_reward = (
        sum(value for _, value in scored) / len(scored) if scored else 0.0
    )

    per_scenario_rewards = [
        SGSingleExperience(
            scenario=record["scenario"],
            entropy_reward=value,
            penalty=0.0,
            effective_reward=value,
            judgment_count=sum(_full_ruleset_judgments(record)),
        )
        for record, value in scored
    ]

    reward_dict = aggregate_reward(entropy_reward=entropy_reward, penalty_breakdown={},
                                   per_scenario_rewards=per_scenario_rewards, judgment_count=judgment_count)

    logger.info(f"Scenario Generation Reward: {reward_dict}")
    return reward_dict


def compute_ra_reward(
    ra_reward_method: str,
    task_spec: BaseTaskSpec,
    scenario_set_path: Path,
    rule_set_path: Path,
    output_dir: Path,
    backend: "ValidatedJudge",
    **settings: Any,
) -> RewardDict:
    """Return the rule amender's reward by the method `settings` names."""
    reward_fn = RA_REWARD_METHODS.get(ra_reward_method)
    if reward_fn is None:
        raise ValueError(f"Unknown RA reward method: {ra_reward_method!r}. Valid: {list(RA_REWARD_METHODS.keys())}")
    return reward_fn(task_spec, scenario_set_path, rule_set_path, output_dir, backend, **settings)


def compute_sg_reward(
    sg_reward_method: str,
    task_spec: BaseTaskSpec,
    scenario_set_path: Path,
    rule_set_path: Path,
    output_dir: Path,
    backend: "ValidatedJudge",
    **settings: Any,
) -> RewardDict:
    """Return the scenario generator's reward by the method `settings` names."""
    reward_fn = SG_REWARD_METHODS.get(sg_reward_method)
    if reward_fn is None:
        raise ValueError(f"Unknown SG reward method: {sg_reward_method!r}. Valid: {list(SG_REWARD_METHODS.keys())}")
    return reward_fn(task_spec, scenario_set_path, rule_set_path, output_dir, backend, **settings)


RA_REWARD_METHODS: Dict[str, Callable] = {
    "standard": compute_ra_reward_standard,
}

SG_REWARD_METHODS: Dict[str, Callable] = {
    "standard": compute_sg_reward_standard,
}