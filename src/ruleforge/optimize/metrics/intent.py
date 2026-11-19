from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Callable

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.constants import INTENT_TRACE_FNAME
from ruleforge.core.files import process_jsonl_file
from ruleforge.rulesets.rules import build_rule_set_string_from_file, rule_identity
from ruleforge.tasks.history import (
    _RUBRIC_MAP,
)

if TYPE_CHECKING:
    from ruleforge.optimize.judging import ValidatedJudge

logger = logging.getLogger(__name__)

#: Greedy.
DEFAULT_INTENT_TEMPERATURE = 0.0
DEFAULT_INTENT_SEED = 16
#: One retry at temperature 0.
GREEDY_TRIES = 2


class IntentScoringError(ValueError, RuleForgeError):
    """The rubric gave no usable reply for a revision: the candidate is skipped, never scored as no penalty."""


def calculate_intent_penalty_rubric(per_rule_scores: List[Dict[str, Any]], max_score: int) -> float:
    """Return the worst per-rule score divided by `max_score`, in [0, 1]; 0.0 without scores."""
    if max_score <= 0:
        return 0.0
    worst = max((r["score"] for r in per_rule_scores), default=0)
    return worst / max_score


def _rule_identities(rule_set_path: Path) -> Dict[tuple, str]:
    identities: Dict[tuple, str] = {}
    for rule in process_jsonl_file(rule_set_path):
        rule_id = str(rule.get("id", "")).strip()
        if not rule_id:
            continue
        identities.setdefault(rule_identity(rule_id), rule_id)
    return identities


def _detect_deleted_rules(original_path: Path, revised_path: Path) -> List[str]:
    original = _rule_identities(original_path)
    revised = _rule_identities(revised_path)
    return sorted(
        raw_id for key, raw_id in original.items() if key not in revised
    )


def _format_rubric(rubric: Dict[str, str]) -> str:
    return "\n".join(f"  {score} - {desc}" for score, desc in rubric.items())


def _build_intent_prompt(original_ruleset: str, revised_ruleset: str, rubric_str: str, max_score: int) -> str:
    return f"""You are an expert at evaluating rule revisions. The revisions are made by a rule amender tasked with refining the language of the rules to make them less ambiguous. Your task is to assess how well the Revised Ruleset preserves the intent of the Original Ruleset.

# Task Details
You must assign a score to each revised rule based on the given Scoring Rubric. Compare each rule in the Revised Ruleset to its corresponding rule in the Original Ruleset (match them by the "id" field). If a rule in the Revised Ruleset does not have a corresponding "id" in the Original Ruleset, evaluate it in the context of the Original Ruleset as a whole. For each Revised Rule, assign an integer score from 0 (best) to {max_score} (worst) using the rubric below. If a Revised Rule is identical to its corresponding Original Rule, assign it a score of 0.

# Scoring Rubric
{rubric_str}

# OUTPUT FORMAT
Return a valid JSON object with this exact structure:
{{
  "rules": [
    {{
      "id": "<id of the revised rule>",
      "reasoning": "<brief explanation for the assigned score>",
      "score": <integer score from 0 to {max_score}>
    }}
  ]
}}
Score ALL revised rules, including those that are unchanged (score 0).

# Original Ruleset:
{original_ruleset}

# Revised Ruleset:
{revised_ruleset}
"""


def _names_one_rule(reply_id: str, rule_ids: Iterable[str]) -> bool:
    """Return whether a reply's id is one of `rule_ids` or holds a single number: `rule_identity` joins every digit, so
    'Rules 1 and 2' and 'Rule 1.2' would both read as Rule 12."""
    return reply_id in rule_ids or len(re.findall(r"\d+", reply_id)) <= 1


def _checked_scores(parsed: Any, rule_ids: Iterable[str], max_score: int) -> Optional[List[Dict[str, Any]]]:
    """Return a reply's per-rule scores, or None unless it scores every rule of `rule_ids` exactly once with an integer
    from 0 to `max_score`."""
    entries = parsed.get("rules") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return None
    rule_ids = list(rule_ids)
    expected = {rule_identity(rule_id) for rule_id in rule_ids}
    scored: Dict[tuple, Dict[str, Any]] = {}
    for entry in entries:
        rule_id, score = (entry.get("id"), entry.get("score")) if isinstance(entry, dict) else (None, None)
        if not isinstance(rule_id, str) or type(score) is not int or not 0 <= score <= max_score:
            return None
        identity = rule_identity(rule_id.strip())
        if not _names_one_rule(rule_id.strip(), rule_ids) or identity not in expected or identity in scored:
            return None
        scored[identity] = {"id": rule_id, "score": score, "reasoning": entry.get("reasoning", "")}
    return list(scored.values()) if len(scored) == len(expected) else None


def run_intent_rubric(
    initial_rule_set_path: Path,
    revised_rule_set_path: Path,
    rubric: Dict[str, str],
    backend: "ValidatedJudge",
    output_dir: Path,
    temperature: float = DEFAULT_INTENT_TEMPERATURE,
    seed: int = DEFAULT_INTENT_SEED,
) -> Dict[str, Any]:
    """Score a revision against the original ruleset with an LLM rubric and return the result dict."""
    max_score = max(int(k) for k in rubric.keys())

    deleted_rule_ids = _detect_deleted_rules(initial_rule_set_path, revised_rule_set_path)
    if deleted_rule_ids:
        logger.info(f"Rule deletion detected ({len(deleted_rule_ids)} rules deleted: {deleted_rule_ids}). Setting intent penalty to 1.0.")
        result = _build_intent_result_rubric(
            penalty=1.0,
            per_rule_scores=[],
            max_score=max_score,
            rule_deletion_detected=True,
            deleted_rule_ids=deleted_rule_ids,
        )
        _save_intent_result(result, output_dir)
        return result

    original_str = build_rule_set_string_from_file(initial_rule_set_path)
    revised_str = build_rule_set_string_from_file(revised_rule_set_path)
    rubric_str = _format_rubric(rubric)
    prompt = _build_intent_prompt(original_str, revised_str, rubric_str, max_score)

    entry = {"system_prompt": prompt, "user_prompt": ""}
    revised_ids = list(_rule_identities(revised_rule_set_path).values())
    results, traces = backend.generate_with_json_validation_and_traces(
        [entry],
        temperature=temperature,
        seed=seed,
        num_tries=min(GREEDY_TRIES, backend.num_tries) if temperature == 0 else None,
        check=lambda parsed: _checked_scores(parsed, revised_ids, max_score) is not None,
    )
    # the trace is written before parsing, so an unparseable response is still on disk
    _save_intent_trace({"prompt": entry, **traces[0]}, output_dir)
    per_rule_scores = _checked_scores(results[0], revised_ids, max_score)
    if per_rule_scores is None:
        raise IntentScoringError(f"The intent rubric did not score every revised rule of {revised_rule_set_path} "
                                 f"exactly once with an integer from 0 to {max_score}")
    penalty = calculate_intent_penalty_rubric(per_rule_scores, max_score)
    result = _build_intent_result_rubric(
        penalty=penalty,
        per_rule_scores=per_rule_scores,
        max_score=max_score,
        rule_deletion_detected=False,
        deleted_rule_ids=[],
    )
    _save_intent_result(result, output_dir)
    return result


def _build_intent_result_rubric(
    penalty: float,
    per_rule_scores: List[Dict[str, Any]],
    max_score: int,
    rule_deletion_detected: bool,
    deleted_rule_ids: List[str],
) -> Dict[str, Any]:
    logger.info(f"Intent penalty: {penalty} (max_score={max_score})")
    for r in per_rule_scores:
        if r["score"] > 0:
            logger.info(f"  {r['id']}: score={r['score']} — {r['reasoning']}")
    return {
        "penalty": penalty,
        "per_rule_scores": per_rule_scores,
        "max_score": max_score,
        "rule_deletion_detected": rule_deletion_detected,
        "deleted_rule_ids": deleted_rule_ids,
    }


def _save_intent_trace(trace: Dict[str, Any], output_dir: Optional[Path]) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / INTENT_TRACE_FNAME, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)


def _save_intent_result(result: Dict[str, Any], output_dir: Optional[Path]) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "intent_penalty.json", "w") as f:
        json.dump(result, f, indent=2)


def _rubric_method(key: str) -> Callable[..., Dict[str, Any]]:

    def method(
        initial_rule_set_path: Path,
        revised_rule_set_path: Path,
        backend: "ValidatedJudge",
        output_dir: Path,
        temperature: float = DEFAULT_INTENT_TEMPERATURE,
        seed: int = DEFAULT_INTENT_SEED,
    ) -> Dict[str, Any]:
        return run_intent_rubric(
            initial_rule_set_path,
            revised_rule_set_path,
            _RUBRIC_MAP[key],
            backend,
            output_dir,
            temperature=temperature,
            seed=seed,
        )

    method.__name__ = method.__qualname__ = f"intent_{key}"
    return method


intent_rubric_orig_standards = _rubric_method("rubric_orig_standards")


def compute_intent_penalty(
    initial_rule_set_path: Path,
    revised_rule_set_path: Path,
    backend: "ValidatedJudge",
    intent_method: str,
    output_dir: Path,
    temperature: float = DEFAULT_INTENT_TEMPERATURE,
    seed: int = DEFAULT_INTENT_SEED,
) -> Dict[str, Any]:
    """Return the intent penalty of a revised ruleset against the original, with its per-rule scores."""
    logger.info(f"Computing intent penalty with method: {intent_method}")
    intent_method_fn = INTENT_METHODS.get(intent_method)
    if intent_method_fn is None:
        raise ValueError(
            f"Unknown intent_method: {intent_method!r}. Valid: {list(INTENT_METHODS.keys())}"
        )
    return intent_method_fn(
        initial_rule_set_path,
        revised_rule_set_path,
        backend,
        output_dir,
        temperature=temperature,
        seed=seed,
    )


INTENT_METHODS: Dict[str, Callable] = {
    "rubric_orig_standards": intent_rubric_orig_standards,
}
