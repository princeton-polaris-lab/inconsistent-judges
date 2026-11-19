from __future__ import annotations

import random
import re
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Optional, Sequence

from ruleforge.prompts.context import build_context_rules_section
from ruleforge.prompts.interpreters import INTERPRETATION_RULES
from ruleforge.prompts.styles import NAIVE, NO_INTERPRETATION, hierarchy_refusal
from ruleforge.tasks.base import LABEL, OptimizableTask, TaskError, TaskSettings
from ruleforge.tasks.history import (
    RATrajectory,
    RewardWeights,
    SGTrajectory,
    build_scenario_set_string,
    format_ra_history_string,
    format_sg_history_string,
    sample_history_scenarios,
    sample_seed_scenarios,
)


def render_legal_prompt(
    rule_text: str,
    scenario: Mapping[str, Any],
    judge_type: str,
    *,
    context_rules: Optional[Sequence[str]] = None,
    interpretations: Mapping[str, str] = INTERPRETATION_RULES,
) -> str:
    """Return the legal judge prompt for one statute text and one ``{facts_of_case, legal_question}`` scenario."""
    facts_of_case = scenario.get("facts_of_case")
    legal_question = scenario.get("legal_question")
    context_section = build_context_rules_section(context_rules)
    if judge_type == NO_INTERPRETATION:
        return f"""# Task
You are an expert in legal reasoning tasked with deciding how a court should resolve a legal question related to a U.S. statute or regulation. Given the facts of a case, the legal question, and the relevant statute or regulation, you must decide whether the court should answer the legal question as 'yes' or 'no'.

# Output Format
Respond with ONLY "yes", "no", or "unsure". You cannot output anything other than "yes", "no", or "unsure".

# Relevant Statute or Regulation
{rule_text}
{context_section}
# Facts of the Case
{facts_of_case}

# Legal Question
{legal_question}
"""
    elif judge_type in interpretations:
        return f"""# Task
You are an expert in legal reasoning tasked with deciding how a court should resolve a legal question related to a U.S. statute or regulation. While some laws may seem clear, many contain ambiguities that must be resolved by the judge. To do so, judges use various tools of interpretation. You are given the facts of a case, the legal question, and the relevant statute or regulation. Your task is to use the provided interpretation strategy to understand the given statute or regulation, and then decide whether the court should answer the legal question as 'yes' or 'no'.

# Output Format
Respond with ONLY "yes", "no", or "unsure". You cannot output anything other than "yes", "no", or "unsure".

# Relevant Statute or Regulation
{rule_text}
{context_section}
# Interpretation Strategy
{interpretations[judge_type]}

# Facts of the Case
{facts_of_case}

# Legal Question
{legal_question}
"""
    else:
        raise ValueError(f"Invalid judge type: {judge_type}")


class LegalTask(OptimizableTask):
    """The legal task: ``{facts_of_case, legal_question}`` scenarios under the ``naive`` prompt only."""

    _builtin: ClassVar[bool] = True
    name: ClassVar[str] = "legal"
    scenario_fields: ClassVar[tuple[str, ...]] = ("facts_of_case", "legal_question")
    styles: ClassVar[Mapping[str, str]] = MappingProxyType({NAIVE: LABEL})
    default_style: ClassVar[str] = NAIVE

    def check(self, settings: TaskSettings) -> None:
        """Refuse every style but ``naive``, and the ``superiors`` context."""
        if settings.prompt_style != NAIVE:
            raise TaskError(
                f"the legal task supports only reasoning_prompt_style='naive', got {settings.prompt_style!r}",
                setting="prompt_style")
        super().check(settings)
        refusal = hierarchy_refusal(settings.prompt_style, settings.rule_context)
        if refusal is not None:
            raise TaskError(refusal, setting="rule_context")

    def render(
        self,
        rule_text: str,
        scenario: Mapping[str, Any],
        settings: TaskSettings,
        *,
        context_rules: Optional[Sequence[str]] = None,
    ) -> str:
        return render_legal_prompt(rule_text, scenario, settings.judge_type, context_rules=context_rules,
                                   interpretations=self.interpretations)

    scenario_block_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*# Scenario \d+\s*\n+"
        r"\s*Subset of Rule IDs:\s*(?P<subset_rule_ids>.*?)\s*\n+"
        r"\s*Possible Ambiguous Topics and Value Tradeoffs:\s*(?P<ambiguous_topics_and_tradeoffs>.*?)\s*\n+"
        r"\s*Facts of the Case:\s*(?P<facts_of_case>.*?)\s*\n+"
        r"\s*Legal Question:\s*(?P<legal_question>.*?)(?=\n\s*# Scenario|\s*\Z)",
        re.DOTALL | re.MULTILINE,
    )

    def sg_system_prompt(
        self, *, per_batch: int, seeds: Mapping[str, Any], already: Sequence[Mapping[str, Any]],
        history: Sequence[SGTrajectory], rng: random.Random,
    ) -> str:
        """Return the scenario generator's prompt; ``single_seed`` seeding shows seed-file examples, then history
        ones."""
        system_prompt = f"""# Task
You are an expert at creating legal scenarios that ambiguously implicate a set of statutory or regulatory rules. Given a set of rules, the optimal scenarios are those for which it is genuinely hard to determine whether a court should answer the legal question 'yes' or 'no' under the rules. You are aiding an agent in revising a set of scenarios to be optimal. Given all of the knowledge you currently have about the optimal set of scenarios, you must generate a set of scenarios that you think is optimal. To produce this revision, you must diligently read through all of the input information. Then, for each scenario, follow this three-step process: (1) Identify a subset of rules (as a list of rule IDs) that, when put in conflict, yields ambiguous scenarios (format with M<rule_id> for meta-rules and R<rule_id> for regular rules); (2) Ideate briefly about possible topics and value tradeoffs of ambiguous scenarios (as a list of sentences) that put the subset of rules into conflict; (3) Generate a scenario based on steps (1) and (2). You must generate {per_batch} scenarios.

# Output Format
You must only output the set of {per_batch} scenarios using the following format:

    # Scenario 1
    Subset of Rule IDs: [<rule_id>, <rule_id>, ...]
    Possible Ambiguous Topics and Value Tradeoffs: [<topic_and_value_tradeoff>, <topic_and_value_tradeoff>, ...]
    Facts of the Case: <facts_of_case>
    Legal Question: <legal_question>
    # Scenario 2
    Subset of Rule IDs: [<rule_id>, <rule_id>, ...]
    Possible Ambiguous Topics and Value Tradeoffs: [<topic_and_value_tradeoff>, <topic_and_value_tradeoff>, ...]
    Facts of the Case: <facts_of_case>
    Legal Question: <legal_question>
    # ...
"""

        if already:
            system_prompt += (
                f"\n# Already Generated Scenarios (DO NOT DUPLICATE)\n"
                f"The following scenarios have already been generated for this episode. "
                f"You MUST NOT generate scenarios with facts of the case or legal questions that "
                f"are duplicates or very similar to these.\n\n"
                f"{build_scenario_set_string(already, self.scenario_fields)}"
            )

        match seeds["type"]:
            case "none":
                pass
            case "single_seed":
                if len(history) == 0:
                    seed_scenarios = sample_seed_scenarios(seeds["paths"]["seed_scenario_path"], self.scenario_fields, seeds["count"], rng)
                    system_prompt += (
                        f"\n# Example Scenarios\n"
                        f"{build_scenario_set_string(seed_scenarios, self.scenario_fields)}"
                    )
                else:
                    history_scenarios = sample_history_scenarios(history, seeds["count"])
                    system_prompt += f"\n# Examples of Good Scenarios\n{build_scenario_set_string(history_scenarios['good_examples'], self.scenario_fields)}"
                    system_prompt += f"\n# Examples of Bad Scenarios\n{build_scenario_set_string(history_scenarios['bad_examples'], self.scenario_fields)}"

        system_prompt += f"\n# Environment Description Given to the Agent\n{self.sg_environment_description()}"

        return system_prompt

    def ra_history(self, history: Sequence[RATrajectory], weights: RewardWeights) -> str:
        return format_ra_history_string(
            history=history,
            valid_keys=self.scenario_fields,
            intent_weight=weights.intent,
            intent_method=weights.intent_method,
        )

    def sg_history(self, history: Sequence[SGTrajectory]) -> str:
        return format_sg_history_string(history=history, valid_keys=self.scenario_fields)

    def sg_environment_description(self) -> str:
        return (
            "You are an agent generating legal scenarios (facts of a case and a legal question) that make it difficult "
            "for the court to decide how to apply a set or subset of statutory or regulatory rules. There is an optimal set of scenarios such that "
            "there is maximal ambiguity in how the rules should be applied to the scenarios "
            "when interpreted by different reasonable interpretation strategies. You will construct a set of scenarios "
            "at each timestep. For each scenario, you will see a set of rules and the associated "
            "reward with each rule based on your scenario quality. Rewards are in the interval "
            "[0, 1]. Your only available action is to output a set of scenarios."
        )

    def ra_environment_description(self) -> str:
        return (
            "You are an agent revising a set of statutory or regulatory rules. There is an "
            "optimal set of rules such that there is no ambiguity in how a court should apply "
            "them under all reasonable interpretive strategies for all possible scenarios. You "
            "will construct a revision for a set of rules at each timestep. For each revision, "
            "you will see a set of legal scenarios and the associated reward with each scenario "
            "based on your revision quality. Rewards are in the interval [-1, 1]. Your only "
            "available action is to output a revision of the set of rules."
        )

    def sg_prior(self) -> str:
        return """
**Conditions for the Optimal Set of Scenarios (in order of importance from lowest to highest number):**
Condition 1: The optimal set of scenarios creates maximal ambiguity when applying the rules to the scenarios, i.e. different reasonable interpretation strategies would yield different answers to the legal question. You will be directly rewarded for high ambiguity through the entropy reward in the reward function.
Condition 2: Each scenario in the optimal set of scenarios should focus on a specific rule or a subset of rules to create maximal ambiguity. Diverse subsets of rules should be reflected in the optimal set of scenarios. Some examples of subsets of rules that empirically create ambiguity include:
Condition 3: The optimal set of scenarios creates interesting scenarios.
Condition 4: The optimal set of scenarios uses realistic language and is not contrived.
Condition 5: The optimal set of scenarios contains scenarios diverse in length.
Condition 6: The optimal set of scenarios has no duplicates.

**Reward and Stochasticity Knowledge:**
- Rewards are in the interval of [0, 1].
"""

    def ra_prior(self, weights: RewardWeights) -> str:
        return f"""
**Conditions for the Optimal Set of Rules (in order of importance from lowest to highest number):**
Condition 1: The optimal set of rules has a similar structure to the original set of rules.
Condition 2: The optimal set of rules retains the essential meaning of the original set of rules.
Condition 3: The optimal set of rules is specific in its wording and does not allow for ambiguity in its interpretation. You will be directly rewarded for low ambiguity through the entropy reward in the reward function.
Condition 4: The optimal set of rules may have meta-rules. A non-exhaustive list of examples of meta-rules include interpretive strategies for one or multiple rules and specifying a hierarchy of rules (sometimes specifying a hierarchy is better done within a specific rule rather than as a meta-rule). Meta-rules should be specified before rules and should follow the same structure as rules.
Condition 5: The optimal set of rules preserves the original ruleset's intent. You will be directly penalized for low intent preservation in the reward function. DO NOT delete any rules from the original ruleset because rule deletion automatically triggers the maximum intent penalty of 1.0. If no rule is deleted, then each revised rule is individually scored on a 0-4 rubric measuring how faithfully the revised rule preserves the original rule's intent (0 = best intent preservation, 4 = worst intent preservation). The intent penalty is then computed based on the worst-scoring rule: penalty = max(per-rule scores) / 4.
Condition 6: The optimal set of rules must balance maximizing the entropy reward while reducing the intent penalty.

**Reward and Stochasticity Knowledge:**
- Overall rewards are in the interval of [-1, 1].
- Entropy reward is in the interval of [0, 1]. The less ambiguous the ruleset, the higher the entropy reward.
- Intent penalty is in the interval of [0, 1]. If any rules in the original ruleset are entirely deleted from the revised ruleset, the intent penalty is automatically set to the maximum value of 1. The less the revised ruleset deviates from the original ruleset's intent, the lower the intent penalty. The intent penalty is max(per-rule scores) / 4, where each rule is scored 0-4 on a rubric (0 = best, 4 = worst).
- Overall rewards are exactly equal to the entropy_reward - ({weights.intent}*(intent_penalty)).
"""
