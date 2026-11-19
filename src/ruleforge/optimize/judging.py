from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ruleforge.judges.base import Judge, JudgeError, JudgeSpec
from ruleforge.prompts.context import VALID_INDIVIDUAL_RULE_CONTEXTS
from ruleforge.rulesets.variants import RuleVariant, build_rule_variants, write_implicated_sidecar
from ruleforge.tasks.spec import BaseTaskSpec
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT, HIERARCHY_STYLES
from ruleforge.core.artifacts import make_evaluation_record
from ruleforge.judges.backend import BATCH_MODEL_TYPES, ModelBackend
from ruleforge.judges.batch_api import REASONING_KEY, REASONING_SUMMARIES_KEY, require_spend_confirmation
from ruleforge.core.constants import FULL_RULESET_KEY
from ruleforge.rulesets.rules import build_rule_set_string_from_file
from ruleforge.core.files import process_jsonl_file
from ruleforge.optimize.input_output import save_data_to_jsonl
from ruleforge.core.resume import prepare_resume_skip

logger = logging.getLogger(__name__)

DEFAULT_SEED = 16



def _configured_seed(judge: Judge) -> int:
    seed = judge.spec.options.get("seed")
    return DEFAULT_SEED if seed is None else int(seed)


class _GenerateOnceBackend(ModelBackend):
    """`ModelBackend`'s validated retry loop over another judge's `generate_once`."""

    def __init__(self, judge: Judge, *, num_tries: int, max_retry_seconds: Optional[float]) -> None:
        self.judge = judge
        self.model_type = judge.spec.model_type
        self.model_name = judge.spec.model_name
        self.seed = _configured_seed(judge)
        self.num_tries = num_tries
        self.temperature = None
        self.max_new_tokens = None
        self.fixed_temperature = judge.capabilities.fixed_temperature
        self.seed_capability = judge.capabilities.seed
        self.max_retry_seconds = max_retry_seconds
        # a billed judge is remote: back off between attempts
        self._base_delay = 1.0 if judge.capabilities.spends_money else 0.0

    def _outcomes(self, prompts: List[Dict[str, str]], temperature: Optional[float], max_new_tokens: Optional[int],
                  num_responses: int, seed: Union[int, List[int], None]) -> List[List[Any]]:
        candidates: List[List[Any]] = [[] for _ in prompts]
        prompt_seeds = seed if isinstance(seed, list) else [seed] * len(prompts)
        for index in range(num_responses):
            seeds = None if seed is None else [prompt_seed + index for prompt_seed in prompt_seeds]
            outcomes = self.judge.generate_once(prompts, temperature=temperature, max_new_tokens=max_new_tokens,
                                                seeds=seeds)
            if len(outcomes) != len(prompts):
                raise RuntimeError(f"{self.model_name} returned {len(outcomes)} outcomes for {len(prompts)} prompts")
            for row, outcome in zip(candidates, outcomes):
                row.append(outcome)
        return candidates

    @staticmethod
    def _text(outcome: Any) -> str:
        failed = outcome.error is not None or outcome.block_failed
        return "" if failed else outcome.text or ""

    def _batch_generate_with_details(self, prompts: List[Dict[str, str]], temperature: Optional[float],
                                     max_new_tokens: Optional[int], num_responses: int,
                                     seed: Union[int, List[int], None]) -> List[List[Dict[str, Any]]]:
        return [[{
            "raw_response": outcome.raw_response or "",
            "completion": self._text(outcome),
            "finish_reason": outcome.finish_reason,
            **({REASONING_SUMMARIES_KEY: list(outcome.reasoning_summaries)}
               if outcome.reasoning_summaries is not None else {}),
            **({REASONING_KEY: outcome.reasoning} if outcome.reasoning is not None else {}),
        } for outcome in row] for row in self._outcomes(prompts, temperature, max_new_tokens, num_responses, seed)]


def refuse_unconfirmed(*judges: Union[Judge, JudgeSpec, None]) -> None:
    """Refuse a judge among ``judges`` (None for no judge), started or not, that bills each call unless the calling
    context confirmed the spend, as a campaign's run refuses it; a batch judge refuses at its own submission."""
    for judge in judges:
        if judge is None:
            continue
        capabilities = judge.capabilities() if isinstance(judge, JudgeSpec) else judge.capabilities
        if capabilities.spends_money and not capabilities.batch:
            require_spend_confirmation()


class ValidatedJudge:
    """A judge as the optimizer calls it: each generate call retried up to `num_tries` times within `max_retry_seconds`,
    with a traced form that also returns every attempt."""

    def __init__(self, judge: Judge, *, num_tries: int, max_retry_seconds: Optional[float]) -> None:
        if not isinstance(judge, Judge):
            raise TypeError(f"a ValidatedJudge wraps a started Judge (JudgeSpec.start()), got {judge!r}")
        if int(num_tries) < 1:
            raise ValueError(f"num_tries must be at least 1, got {num_tries}")
        if max_retry_seconds is not None and max_retry_seconds <= 0:
            raise ValueError("max_retry_seconds must be positive")
        self.judge = judge
        self.num_tries = int(num_tries)
        self.max_retry_seconds = None if max_retry_seconds is None else float(max_retry_seconds)
        if isinstance(judge, ModelBackend):
            if judge.max_retry_seconds != self.max_retry_seconds:
                raise JudgeError(
                    f"{judge.model_name} reads its retry budget from its model config (max_retry_seconds="
                    f"{judge.max_retry_seconds}), not {self.max_retry_seconds}: set it there")
            self._backend: ModelBackend = judge
        else:
            self._backend = _GenerateOnceBackend(judge, num_tries=self.num_tries,
                                                 max_retry_seconds=self.max_retry_seconds)

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def model_type(self) -> str:
        return self._backend.model_type

    @property
    def seed(self) -> int:
        """Return the seed of this judge's sampling, from its model config."""
        return self._backend.seed

    @property
    def strip_thinking_model(self) -> Optional[str]:
        """Return the model family whose thinking this judge's answers are framed in, from its model config."""
        return self.judge.spec.options.get("strip_thinking_model")

    def prompt_size(self, prompt: Dict[str, str]) -> Tuple[Optional[int], Optional[int]]:
        """Return a prompt's token count and the length the model refuses; None where the judge has no cheap count."""
        count = self._backend.prompt_token_count
        return (None if count is None else count(prompt)), self._backend.prompt_token_limit

    def _tries(self, num_tries: Optional[int]) -> int:
        return self.num_tries if num_tries is None else num_tries

    def _calling(self) -> ModelBackend:
        """Return the backend for a call, refused unconfirmed for a judge that bills each call (:func:`refuse_unconfirmed`;
        a run makes that refusal before it writes, and this one stands behind it)."""
        refuse_unconfirmed(self.judge)
        return self._backend

    def generate(self, prompts: List[Dict[str, str]], num_responses: int = 1, temperature: Optional[float] = None,
                 max_new_tokens: Optional[int] = None, seed: Optional[int] = None,
                 num_tries: Optional[int] = None) -> List[List[str] | None]:
        """Return `num_responses` non-empty candidates per prompt (None when none came)."""
        return self._calling().generate(prompts, num_responses=num_responses, temperature=temperature,
                                      max_new_tokens=max_new_tokens, seed=seed, num_tries=self._tries(num_tries))

    def generate_with_validation(self, prompts: List[Dict[str, str]], valid_responses: Tuple[str, ...],
                                 temperature: Optional[float] = None, seed: Optional[int] = None,
                                 num_tries: Optional[int] = None,
                                 response_validator: Optional[Callable[[str], Optional[str]]] = None,
                                 max_retry_seconds: Optional[float] = None,
                                 ) -> List[str | None]:
        """Return one validated label per prompt (None when the attempts run out); `max_retry_seconds` replaces the
        judge's budget for this call."""
        return self._calling().generate_with_validation(prompts, valid_responses, temperature=temperature, seed=seed,
                                                      num_tries=self._tries(num_tries),
                                                      response_validator=response_validator,
                                                      max_retry_seconds=max_retry_seconds)

    def generate_with_json_validation(self, prompts: List[Dict[str, str]], temperature: Optional[float] = None,
                                      seed: Optional[int] = None,
                                      num_tries: Optional[int] = None) -> List[Dict[str, Any] | None]:
        """Return one parsed JSON object per prompt (None when the attempts run out)."""
        return self._calling().generate_with_json_validation(prompts, temperature=temperature, seed=seed,
                                                           num_tries=self._tries(num_tries))

    def generate_with_traces(self, prompts: List[Dict[str, str]], num_responses: int = 1,
                             temperature: Optional[float] = None, max_new_tokens: Optional[int] = None,
                             seed: Optional[int] = None, num_tries: Optional[int] = None,
                             ) -> Tuple[List[List[str] | None], List[Dict[str, Any]]]:
        """Return `generate`'s result and one trace per prompt."""
        return self._calling().generate_with_traces(prompts, num_responses=num_responses, temperature=temperature,
                                                  max_new_tokens=max_new_tokens, seed=seed,
                                                  num_tries=self._tries(num_tries))

    def generate_with_validation_and_traces(self, prompts: List[Dict[str, str]], valid_responses: Tuple[str, ...],
                                            temperature: Optional[float] = None, seed: Optional[int] = None,
                                            num_tries: Optional[int] = None,
                                            response_validator: Optional[Callable[[str], Optional[str]]] = None,
                                            max_retry_seconds: Optional[float] = None,
                                            ) -> Tuple[List[str | None], List[Dict[str, Any]]]:
        """Return `generate_with_validation`'s answers and one trace per prompt."""
        return self._calling().generate_with_validation_and_traces(
            prompts, valid_responses, temperature=temperature, seed=seed, num_tries=self._tries(num_tries),
            response_validator=response_validator, max_retry_seconds=max_retry_seconds)

    def generate_with_json_validation_and_traces(self, prompts: List[Dict[str, str]],
                                                 temperature: Optional[float] = None, seed: Optional[int] = None,
                                                 num_tries: Optional[int] = None,
                                                 check: Optional[Callable[[Any], bool]] = None,
                                                 ) -> Tuple[List[Dict[str, Any] | None], List[Dict[str, Any]]]:
        """Return `generate_with_json_validation`'s objects and one trace per prompt; an answer `check` refuses counts as
        a failed attempt."""
        return self._calling().generate_with_json_validation_and_traces(prompts, temperature=temperature, seed=seed,
                                                                      num_tries=self._tries(num_tries), check=check)


#: Below this, a batch-provider run is warned: each flush is its own batch job with its own completion window.
MIN_BATCH_PROVIDER_SAVE_SIZE = 100


def run_judgments(
    scenario_set_path: Path,
    rule_set_path: Path,
    backend: "ValidatedJudge",
    task_spec: BaseTaskSpec,
    judge_types: List[str],
    evaluation_mode: str,
    output_path: Path,
    save_batch_size: int = 10,
    num_tries: int = 1,
    seed: Optional[int] = None,
    evaluation_temperature: float = 0.1,
    individual_rule_context: str = "isolated",
    resume: bool = False,
    raw_output_path: Path | None = None,
    max_retry_seconds: float | None = None,
) -> None:
    """Judge every scenario under the ruleset, once per judge type, writing the evaluations JSONL to `output_path`:
    rule context 'isolated' judges each rule alone, 'implicated' adds the rules it references ('superiors' and
    hierarchy-style tasks are refused), and seeds derive from `seed`, by default the judge's own."""
    start_time = time.time()

    evaluation_mode = evaluation_mode.lower()
    if evaluation_mode not in {"full", "individual", "both"}:
        raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

    individual_rule_context = individual_rule_context.lower()
    if individual_rule_context not in VALID_INDIVIDUAL_RULE_CONTEXTS:
        raise ValueError(
            f"Invalid individual_rule_context: {individual_rule_context!r}"
        )
    style = getattr(task_spec, "reasoning_prompt_style", None)
    if individual_rule_context == HIERARCHY_RULE_CONTEXT or style in HIERARCHY_STYLES:
        raise ValueError(
            f"run_judgments reads no rule levels, so it cannot render the hierarchy prompt "
            f"(reasoning_prompt_style {style!r}, individual_rule_context {individual_rule_context!r}): "
            "only the stochastic campaign's runner renders it, from a levels file"
        )

    # appending is refused by default: a doubled scenario changes every metric's denominator
    skip_count = prepare_resume_skip(output_path, resume, label="evaluations")
    if raw_output_path is not None:
        if raw_output_path == output_path:
            raise ValueError("raw_output_path must differ from output_path")
        raw_skip_count = prepare_resume_skip(
            raw_output_path, resume, label="judge outputs"
        )
        if raw_skip_count != skip_count:
            raise ValueError(
                f"Cannot resume mismatched artifacts: {output_path} has "
                f"{skip_count} complete rows but {raw_output_path} has "
                f"{raw_skip_count}."
            )

    evaluate_full = evaluation_mode in {"full", "both"}
    evaluate_individual = evaluation_mode in {"individual", "both"}

    if not judge_types:
        raise ValueError("judge_types must be non-empty")

    rules = list(process_jsonl_file(rule_set_path))
    if not rules:
        raise ValueError(f"No rules found in {rule_set_path}")

    rule_set_string = build_rule_set_string_from_file(rule_set_path)

    individual_rule_variants: List[RuleVariant] = []
    if evaluate_individual:
        individual_rule_variants, implication_map = build_rule_variants(
            rules, individual_rule_context
        )
        write_implicated_sidecar(
            output_path.parent, individual_rule_context, implication_map
        )

    model_name = backend.model_name

    logger.info(f"Evaluating {rule_set_path} against {scenario_set_path}")
    logger.info(f"Mode: {evaluation_mode}, judge_types: {judge_types}")

    if getattr(backend, "model_type", None) in BATCH_MODEL_TYPES and (
        save_batch_size < MIN_BATCH_PROVIDER_SAVE_SIZE
    ):
        logger.warning(
            "save_batch_size=%d against batch provider %r: each flush is a "
            "separate batch job that blocks until it clears, so this sweep "
            "will take %d sequential completion windows. Set it to the "
            "scenario count to submit one batch instead.",
            save_batch_size,
            backend.model_type,
            save_batch_size,
        )

    prompt_entries: List[Dict[str, Any]] = []

    current_batch: List[Dict[str, Any]] = []
    current_raw_batch: List[Dict[str, Any]] = []

    def flush_and_save():
        if not prompt_entries:
            return
        generation_kwargs = {
            "valid_responses": task_spec.valid_eval_responses,
            "num_tries": num_tries,
            "seed": backend.seed if seed is None else seed,
            "temperature": evaluation_temperature,
            "response_validator": task_spec.eval_response_validator,
            **({} if max_retry_seconds is None else {"max_retry_seconds": max_retry_seconds}),
        }
        if raw_output_path is None:
            results = backend.generate_with_validation(
                prompt_entries, **generation_kwargs
            )
            traces = None
        else:
            results, traces = backend.generate_with_validation_and_traces(
                prompt_entries, **generation_kwargs
            )
        for idx, entry in enumerate(prompt_entries):
            entry["variant_results"][entry["judge_type"]] = results[idx]
            if traces is not None:
                entry["variant_traces"][entry["judge_type"]] = {
                    "prompt": {
                        "system_prompt": entry["system_prompt"],
                        "user_prompt": entry["user_prompt"],
                    },
                    **traces[idx],
                }
        prompt_entries.clear()

        if current_batch:  # pragma: no branch -- a queued prompt always has its scenario queued
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a") as f:
                for item in current_batch:
                    f.write(json.dumps(item) + "\n")
            current_batch.clear()
        if raw_output_path is not None and current_raw_batch:
            raw_output_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_output_path.open("a") as f:
                for item in current_raw_batch:
                    f.write(json.dumps(item) + "\n")
            current_raw_batch.clear()

    examined_count = 0
    for scenario_index, scenario in enumerate(process_jsonl_file(scenario_set_path)):
        if scenario_index < skip_count:
            continue
        examined_count += 1
        eval_dict = make_evaluation_record(scenario, {model_name: {}})
        model_evals = eval_dict["evaluations"][model_name]
        if raw_output_path is not None:
            raw_dict = {
                key: value for key, value in eval_dict.items() if key != "evaluations"
            }
            raw_dict["judge_outputs"] = {model_name: {}}
            model_traces = raw_dict["judge_outputs"][model_name]
        else:
            raw_dict = None
            model_traces = None

        def queue_variant(
            rule_text: str,
            variant_key: str,
            context_rules: List[str] | None = None,
        ) -> Dict[str, Any]:
            variant_results: Dict[str, Any] = {}
            variant_traces: Dict[str, Any] = {}
            for judge_type in judge_types:
                eval_prompt = task_spec.build_eval_prompt(
                    rule_text=rule_text,
                    scenario=scenario,
                    judge_type=judge_type,
                    context_rules=context_rules,
                )
                prompt_entries.append(
                    {
                        "system_prompt": eval_prompt,
                        "user_prompt": "",
                        "variant_results": variant_results,
                        "variant_traces": variant_traces,
                        "judge_type": judge_type,
                    }
                )
            if model_traces is not None:
                model_traces[variant_key] = variant_traces
            return variant_results

        if evaluate_full:
            model_evals[FULL_RULESET_KEY] = queue_variant(
                rule_set_string, FULL_RULESET_KEY
            )

        if evaluate_individual:
            for rule_id, rule_prompt_text, context_rules in individual_rule_variants:
                model_evals[rule_id] = queue_variant(
                    rule_prompt_text, rule_id, context_rules
                )

        current_batch.append(eval_dict)
        if raw_dict is not None:
            current_raw_batch.append(raw_dict)
        logger.debug(f"Queued scenario {examined_count}")

        if len(current_batch) >= save_batch_size:
            flush_and_save()

    if current_batch or current_raw_batch or prompt_entries:
        flush_and_save()

    if not output_path.exists():
        # an absent file would crash readers mid-episode; an empty one is zero records
        logger.warning(f"No scenarios evaluated; writing empty {output_path}")
        save_data_to_jsonl([], output_path)

    elapsed = time.time() - start_time
    logger.info(
        f"Evaluated {examined_count} scenarios in {elapsed:.2f}s -> {output_path}"
    )
    save_data_to_jsonl(
        {
            "seconds": round(elapsed, 2),
            "minutes": round(elapsed / 60, 2),
            "scenarios": examined_count,
        },
        output_path.parent / "evaluation_time.jsonl",
    )


__all__ = ["DEFAULT_SEED", "MIN_BATCH_PROVIDER_SAVE_SIZE", "ValidatedJudge", "run_judgments"]
