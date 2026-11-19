from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from omegaconf import DictConfig, OmegaConf

#: Config key of each seed file role.
SEED_PATH_KEYS = {"good": "good_seed_scenario_path", "bad": "bad_seed_scenario_path", "single": "seed_scenario_path"}
#: Seed file roles each seeding reads.
SEEDING_ROLES = {"none": (), "wildchat": ("good", "bad"), "single_seed": ("single",)}
#: Each built-in task's seedings (the first is its default) and its default number of seed examples.
TASK_SEEDING: dict[str, tuple[tuple[str, ...], int]] = {
    "modelspec": (("wildchat", "none"), 5),
    "legal": (("single_seed", "none"), 1),
}
PAPER_JUDGE_TYPES = (
    "No_Interpretation", "Narrow", "Broad", "Ordinary_Meaning", "Omitted_Case", "General_Terms",
    "Negative_Implication", "Grammar_Based", "Whole_Principle", "Consistent_Usage", "Surplus_Meaning",
    "Harmonization", "Specific_over_General",
)


@dataclass(frozen=True)
class SGConfig:
    """The scenario generator's settings."""

    #: None: the task's default.
    initial_scenario_seed_type: Optional[str] = None
    #: None: the task's default.
    num_seed_scenarios: Optional[int] = None
    target_num_scenarios: int = 20
    num_scenarios_per_batch: int = 10
    entropy_threshold: float = 0.9
    #: None: the rule amender's.
    entropy_method: Optional[str] = None
    max_accumulation_iterations: int = 3
    termination_threshold: float = 0.1
    termination_episodes: int = 3
    trigger_threshold: float = 0.5
    posterior_update_temperature: float = 0.2
    posterior_sampling_temperature: float = 1.0


@dataclass(frozen=True)
class RAConfig:
    """The rule amender's settings."""

    intent_weight: float = 1.0
    #: Scores both agents' entropy rewards.
    entropy_method: str = "consensus"
    intent_method: str = "rubric_orig_standards"
    best_of_n: int = 5
    termination_threshold: float = 0.9
    termination_episodes: int = 15
    posterior_update_temperature: float = 0.2
    posterior_sampling_temperature: float = 1.0


#: Fields a Hydra config holds under `eval_cfg`.
EVAL_FIELDS = ("judge_types", "individual_rule_context", "save_batch_size",
               "evaluation_temperature", "reasoning_prompt_style", "intent_temperature", "intent_seed",
               "panel_num_tries", "panel_max_retry_seconds")


@dataclass(frozen=True)
class OptimizeConfig:
    """A run's settings: both agents', the judge panel's (`panel_num_tries` attempts per judgment, within
    `panel_max_retry_seconds` per flush of `save_batch_size` scenarios) and the agents' and the intent rubric's validated
    retries (`num_tries` attempts within `max_retry_seconds`; None is no limit)."""

    max_episodes: int = 25
    scenario_generation: SGConfig = field(default_factory=SGConfig)
    rule_amendment: RAConfig = field(default_factory=RAConfig)
    judge_types: tuple[str, ...] = PAPER_JUDGE_TYPES
    individual_rule_context: str = "isolated"
    save_batch_size: int = 10
    evaluation_temperature: float = 0.1
    reasoning_prompt_style: str = "naive"
    intent_temperature: float = 0.0
    intent_seed: int = 16
    panel_num_tries: int = 30
    panel_max_retry_seconds: Optional[float] = 3600.0
    num_tries: int = 5
    max_retry_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        given = self.scenario_generation.entropy_method
        if given is not None and given != self.rule_amendment.entropy_method:
            raise ValueError(
                f"scenario_generation.entropy_method {given!r} differs from rule_amendment.entropy_method "
                f"{self.rule_amendment.entropy_method!r}, which scores the scenario generator's reward too: set "
                "the rule amender's")

    def seeding(self, task: str) -> tuple[str, tuple[str, ...]]:
        """Return the seeding of `task`'s scenario generator and the seed file roles it reads."""
        if task not in TASK_SEEDING:
            raise ValueError(f"task {task!r} is not a built-in task ({sorted(TASK_SEEDING)})")
        seedings, _ = TASK_SEEDING[task]
        seed_type = self.scenario_generation.initial_scenario_seed_type or seedings[0]
        if seed_type not in seedings:
            raise ValueError(f"task {task!r}'s scenario generator is seeded by one of {list(seedings)}, not "
                             f"{seed_type!r} (scenario_generation.initial_scenario_seed_type)")
        return seed_type, SEEDING_ROLES[seed_type]

    def to_config(self, *, task: str, rules: Path | str, seeds: Mapping[str, Path | str]) -> DictConfig:
        """Return these settings and the run's inputs laid out as a composed Hydra config."""
        seed_type, roles = self.seeding(task)
        if set(seeds) != set(roles):
            raise ValueError(f"the {seed_type!r} seeding of task {task!r}'s scenario generator reads the seed files "
                             f"{list(roles)}; seeds names {sorted(seeds)}")
        _, default_count = TASK_SEEDING[task]
        given = self.scenario_generation
        scenario_generation = {
            **asdict(given),
            "initial_scenario_seed_type": seed_type,
            "num_seed_scenarios": default_count if given.num_seed_scenarios is None else given.num_seed_scenarios,
            "entropy_method": self.rule_amendment.entropy_method,
            **{SEED_PATH_KEYS[role]: str(path) for role, path in seeds.items()},
        }
        rule_amendment = {**asdict(self.rule_amendment), "initial_rule_set_path": str(rules)}
        eval_cfg: dict[str, Any] = {name: getattr(self, name) for name in EVAL_FIELDS}
        eval_cfg["judge_types"] = list(self.judge_types)
        return OmegaConf.create({
            "task_spec": task, "max_episodes": self.max_episodes, "scenario_generation": scenario_generation,
            "rule_amendment": rule_amendment, "eval_cfg": eval_cfg, "judge_model": None, "agent_model": None,
        })


def seed_paths(scenario_generation: Any) -> dict[str, Path]:
    """Return the seed scenario files a `scenario_generation` config names, by role."""
    return {role: Path(scenario_generation[key]) for role, key in SEED_PATH_KEYS.items()
            if scenario_generation.get(key) is not None}


__all__ = ["EVAL_FIELDS", "OptimizeConfig", "PAPER_JUDGE_TYPES", "RAConfig", "SEEDING_ROLES", "SEED_PATH_KEYS",
           "SGConfig", "TASK_SEEDING", "seed_paths"]
