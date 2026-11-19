from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Tuple, TypedDict

from ruleforge.core.files import process_jsonl_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardWeights:
    """How the rule amender's reward is composed: entropy reward minus ``intent`` times the intent penalty."""

    intent: float = 1.0
    entropy_method: str = "consensus"
    intent_method: str = "rubric_orig_standards"

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", float(self.intent))


class RASingleExperience(TypedDict):
    """Per-scenario experience in a rule-amendment trajectory."""
    scenario: Dict[str, Any]
    # 1 - the scenario's entropy (higher = consensus); their mean is the ruleset's entropy reward.
    entropy_reward: float

class RATrajectory(TypedDict):
    """One rule-amendment episode: the text of the ruleset it produced and its aggregate reward."""
    action: str
    reward: RewardDict

class SGSingleExperience(TypedDict):
    """Per-scenario experience in a scenario-generation trajectory."""
    scenario: Dict[str, Any]
    # The scenario's entropy (higher = disagreement).
    entropy_reward: float
    penalty: float
    effective_reward: float
    judgment_count: int


class SGTrajectory(TypedDict):
    """One scenario-generation episode: the ruleset seen, the scenarios kept, every scenario generated, the reward."""
    state: str
    accumulated_scenarios: List[SGSingleExperience]
    all_generated: List[SGSingleExperience]
    reward: float

class PenaltyBreakdownEntry(TypedDict):
    """One penalty's value and weight, with optional ``details``."""
    value: float
    weight: float
    details: NotRequired[Dict[str, Any]]

class RewardDict(TypedDict):
    """Aggregate reward dictionary for a RA or SG episode."""
    entropy_reward: float
    regularization_penalty: float
    effective_reward: float
    per_scenario_rewards: List[RASingleExperience] | List[SGSingleExperience]
    penalty_breakdown: Dict[str, PenaltyBreakdownEntry]
    # The panel judgments the entropy reward is computed from; a scenario whose panel lost one is left out.
    judgment_count: int


def format_scenario_string(scenario: Dict[str, Any], valid_keys: Tuple[str, ...]) -> str:
    """Render a scenario's ``valid_keys`` as ``Key: value`` lines."""
    def _format_key(key: str) -> str:
        key = key.replace("_", " ")
        return key.capitalize()

    parts = [f"{_format_key(key)}: {value}\n" for key, value in scenario.items() if key in valid_keys]
    return "\n".join(parts)

def format_scenario_xml(scenario: Dict[str, Any], valid_keys: Tuple[str, ...]) -> str:
    """Render a scenario's ``valid_keys`` as upper-cased XML tags."""
    parts = [f"<{key.upper()}>\n{value}\n</{key.upper()}>\n" for key, value in scenario.items() if key in valid_keys]
    return "\n".join(parts)


def build_scenario_set_string(
    scenario_set_file: Path | List[Dict[str, Any]],
    valid_keys: Tuple[str, ...],
    num_hash_prefix: int = 2,
) -> str:
    """Render a scenario set (a JSONL path or a list of rows) as numbered sections."""
    if isinstance(scenario_set_file, Path):
        scenarios = list(process_jsonl_file(scenario_set_file))
    elif isinstance(scenario_set_file, list):
        scenarios = scenario_set_file
    else:
        raise ValueError(
            "scenario_set_file must be a path to a JSONL file or a list of dictionaries."
        )

    out = ""
    for i, scenario in enumerate(scenarios):
        out += (
            f"{'#' * num_hash_prefix} Scenario {i + 1}\n"
            f"{format_scenario_string(scenario, valid_keys)}"
        )
    return out

def read_scenario_set_from_file(file_path: Path, valid_keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Read a scenario set from a JSONL file, keeping only ``valid_keys``."""
    return [
        {k: v for k, v in scenario.items() if k in valid_keys}
        for scenario in process_jsonl_file(file_path)
    ]


_RUBRIC_ORIGINAL_STANDARDS = {
    "0": "A reasonable rule amender tasked with clarifying this rule would readily produce this revision. Every element of the Revised Rule has a clear root, precursor, or germ in the Original Rule. The revision merely clarifies principles already in effect in the Original Rule without introducing substantive changes to its scope or effect. A rational connection between the Original Rule and the Revised Rule can reasonably be discerned. The revision does not introduce unnecessary specificity that is more extensive than necessary to serve the original rule's purpose. Examples of such acceptable revisions include: minor rewordings, illustrative (non-mandatory) examples, elaboration of what was already implicit, and meta-rules that establish a reasonable hierarchy or interpretation guidance among existing rules.",
    "1": "The revision goes beyond minimal clarification but reflects a defensible judgment call. A reasonable rule amender could justify every addition by pointing to a root, precursor, or germ in the Original Rule's text, though some additions are debatably beyond strict clarification. The revision introduces some changes that change the scope or effect of the Original Rule but these changes are reasonable and not too far from the original rule's intent. The revision does not introduce unnecessary specificity that is more extensive than necessary to serve the original rule's purpose. A rational connection between the Original and Revised Rule can reasonably be discerned.",
    "2": "The revision raises moderate concerns. It may introduce substantive changes that significantly alter the scope or effect of the Original Rule, or some elements of the Revised Rule have only a tenuous root, precursor, or germ in the Original Rule. However, the change is not arbitrary and capricious — a rational connection between the Original Rule and the Revised Rule can still be discerned, even if strained. The revision does not mandate exact verbatim phrases or numeric thresholds, and does not produce an absurd or irrational result.",
    "3": "The revision raises serious concerns. It either (a) introduces specificity that is more extensive than necessary to fulfill the original rule's purpose, such as mandating exact verbatim phrases or numeric thresholds, OR (b) a rational connection between the Original Rule and the Revised Rule cannot reasonably be discerned. However, the revision does not produce an absurd or internally contradictory result.",
    "4": "The revision is absurd: it either (a) consists of a disposition that no reasonable person could intend, OR (b) applying it would lead to an absurd and irrational result. Examples of such unacceptable revisions include: self-contradictory logic, impossible conditions, rules that completely negate the original's objective, and meta-rules that completely nullify the existing ruleset.",
}

_RUBRIC_MAP: Dict[str, Dict[str, str]] = {
    "rubric_orig_standards": _RUBRIC_ORIGINAL_STANDARDS,
}

def format_intent_feedback_rubric(
    penalty: float,
    per_rule_scores: List[Dict[str, Any]],
    intent_method: str,
    rule_deletion_detected: bool = False,
) -> str:
    """Render the intent penalty for the agent's history: flagged rules grouped by score, each with its rubric text."""
    rubric = _RUBRIC_MAP.get(intent_method, {})
    intent_str = f"Intent Penalty: {penalty}"

    if rule_deletion_detected:
        return intent_str + " (Rule deletion detected - automatic maximum penalty)\n"

    flagged = [r for r in per_rule_scores if r["score"] > 0]
    if not flagged:
        return intent_str + "\n"

    scores_present = sorted({r["score"] for r in flagged})
    lines = [intent_str, "\tIntent Penalty Breakdown:"]

    for score_level in scores_present:
        rules_at_level = [r for r in flagged if r["score"] == score_level]
        rule_ids = [r["id"] for r in rules_at_level]
        rubric_text = rubric.get(str(score_level), "")
        lines.append(f"\t\tRules that scored {score_level}: {', '.join(rule_ids)}")
        lines.append(f"\t\tExplanation of score {score_level}: {rubric_text}")

    return "\n".join(lines) + "\n"


def format_ra_history_string(history: List[RATrajectory],
                            valid_keys: Tuple[str, ...],
                            intent_weight: float = 1.0, 
                            intent_method: str = "rubric_orig_standards") -> str:
    """Render the rule amender's history; only the latest trajectory shows per-scenario experiences."""
    history_string = ""
    for i in range(len(history) - 1, -1, -1):
        trajectory = history[i]
        history_string += f"<TRAJECTORY_{i}>\n"

        action = trajectory["action"]
        if not isinstance(action, str):
            raise TypeError(f"a trajectory's action is the ruleset's text, got {action!r}")
        reward = trajectory["reward"]
        effective_reward = reward["effective_reward"]
        entropy_reward = reward["entropy_reward"]

        history_string += f"<ACTION>\n{action}\n</ACTION>\n"
        history_string += (
            f"<REWARD>\n"
            f"Overall Total Reward: {effective_reward}\n"
            f"Overall Entropy Reward: {entropy_reward}\n"
        )
        if intent_weight > 0:
            intent_details = reward["penalty_breakdown"]["intent"]["details"]
            intent_feedback_str = format_intent_feedback_rubric(
                penalty=intent_details["penalty"],
                per_rule_scores=intent_details["per_rule_scores"],
                intent_method=intent_method,
                rule_deletion_detected=intent_details.get("rule_deletion_detected", False),
            )
            history_string += f"{intent_feedback_str}"
        history_string += "</REWARD>\n"

        if i == len(history) - 1:
            per_scenario_rewards = reward.get("per_scenario_rewards", [])
            history_string += "<EXPERIENCES>\n"
            for exp in per_scenario_rewards:
                state_block = format_scenario_xml(exp["scenario"], valid_keys)
                exp_entropy = exp["entropy_reward"]
                history_string += (
                    f"<STATE>\n{state_block}\n</STATE>\n"
                    f"<REWARD>\nScenario Entropy Reward: {exp_entropy}\n</REWARD>\n"
                )
            history_string += "</EXPERIENCES>\n"

        history_string += f"</TRAJECTORY_{i}>\n"

    logger.debug(f"RA history string: {history_string}")
    return history_string


def format_sg_history_string(history: List[SGTrajectory], valid_keys: Tuple[str, ...]) -> str:
    """Render the scenario generator's history; only the latest trajectory shows the ruleset and every scenario."""
    history_string = ""
    for i in range(len(history) - 1, -1, -1):
        trajectory = history[i]
        history_string += f"<TRAJECTORY_{i + 1}>\n"

        if i == len(history) - 1:
            rule_set = trajectory["state"]
            history_string += f"<STATE>\n{rule_set}</STATE>\n"
            scenarios_to_show = trajectory.get("all_generated", [])
        else:
            scenarios_to_show = trajectory.get("accumulated_scenarios", [])

        for exp in scenarios_to_show:
            scenario = exp["scenario"]
            exp_reward = exp["effective_reward"]
            history_string += (
                f"<EXPERIENCE>\n"
                f"<ACTION>{format_scenario_xml(scenario, valid_keys)}</ACTION>\n"
                f"<REWARD>\n{exp_reward}\n</REWARD>\n"
                f"</EXPERIENCE>\n"
            )

        history_string += f"</TRAJECTORY_{i + 1}>\n"

    logger.debug(f"SG history string: {history_string}")
    return history_string


def sample_seed_scenarios(path: Path, valid_keys: Tuple[str, ...], num_seed_scenarios: int,
                          rng: random.Random) -> List[Dict[str, Any]]:
    """Sample up to ``num_seed_scenarios`` rows from a scenario set file with the run's generator ``rng``."""
    scenarios = read_scenario_set_from_file(path, valid_keys)
    return rng.sample(scenarios, k=min(num_seed_scenarios, len(scenarios)))


def sample_history_scenarios(history: List[SGTrajectory], num_examples: int = 1) -> Dict[str, List[Dict[str, Any]]]:
    """Return the last trajectory's top and bottom ``num_examples`` scenarios by reward as good and bad examples."""
    if len(history) == 0:
        raise ValueError("No history to get good and bad examples from.")

    all_generated = list(history[-1].get("all_generated", []))

    scored = [
        (exp["scenario"], exp["effective_reward"])
        for exp in all_generated
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    good_examples = [s for s, _ in scored[:num_examples]]
    bad_examples = [s for s, _ in scored[-num_examples:]]
    return {"good_examples": good_examples, "bad_examples": bad_examples}
