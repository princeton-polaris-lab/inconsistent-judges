from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ruleforge.core.constants import FULL_RULESET_KEY
from ruleforge.core.files import process_jsonl_file
from ruleforge.optimize.input_output import save_data_to_jsonl

logger = logging.getLogger(__name__)


def calculate_consensus_entropy_yes_no_unsure(
    data_list: List[str],
) -> float:
    """Return the two-class entropy of the judgments; each 'unsure' counts half yes, half no."""
    if not data_list:
        return 0.0

    yes_count = data_list.count("yes") + 0.5 * data_list.count("unsure")
    no_count = data_list.count("no") + 0.5 * data_list.count("unsure")

    yes_no_total = yes_count + no_count
    if yes_no_total == 0:
        return 0.0
    probs = [yes_count / yes_no_total, no_count / yes_no_total]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def extract_rule_judgments(
    evaluations: Dict[str, Any],
    parse_fn: Callable,
) -> Dict[str, Dict[str, List[str]]]:
    """Return the parsed judgments as {model: {variant: [judgment, ...]}}."""
    judgments_by_model: Dict[str, Dict[str, List[str]]] = {}
    unparseable_count = 0

    for model_name, model_evals in evaluations.items():
        if not isinstance(model_evals, dict):
            continue

        judgments_by_rule_key: Dict[str, List[str]] = {}
        for rule_key, rule_evals in model_evals.items():
            if not isinstance(rule_evals, dict):
                continue

            judgments: List[str] = []
            for _judge_type, raw_judgment in rule_evals.items():
                judgment = parse_fn(raw_judgment)
                if judgment:
                    judgments.append(judgment)
                else:
                    unparseable_count += 1
            judgments_by_rule_key[rule_key] = judgments

        if judgments_by_rule_key:
            judgments_by_model[model_name] = judgments_by_rule_key

    if unparseable_count:
        logger.warning(f"Skipped {unparseable_count} unparseable judgments during extraction")

    return judgments_by_model


def iter_entropy_values(entropy_map: Dict[str, Dict[str, float]]) -> List[float]:
    """Flatten {model: {variant: entropy}} into a list."""
    flattened: List[float] = []
    for variant_map in entropy_map.values():
        if not isinstance(variant_map, dict):
            raise ValueError("Variant map is not a dict")
        for value in variant_map.values():
            try:
                flattened.append(float(value))
            except (TypeError, ValueError) as e:
                raise ValueError(f"Unable to coerce entropy value to float: {value}") from e
    return flattened


def get_scenario_full_ruleset_entropy(
    entropy_record: Dict[str, Any],
    full_ruleset_key: str = FULL_RULESET_KEY,
) -> Optional[float]:
    """Return the mean full-ruleset entropy across models, or None when nobody answered."""
    entropy_map = entropy_record.get("entropy", {})
    values = []
    for model_entropies in entropy_map.values():
        if isinstance(model_entropies, dict):
            v = model_entropies.get(full_ruleset_key)
            if v is not None:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
    return sum(values) / len(values) if values else None


def entropy_for_scenario(
    scenario: Dict[str, Any],
    entropy_method: str,
    parse_fn: Callable,
) -> Optional[Dict[str, Any]]:
    """Return one scenario's entropy record, or None when no judgment parsed."""
    entropy_fn = ENTROPY_METHODS.get(entropy_method)
    if entropy_fn is None:
        raise ValueError(
            f"Invalid entropy_method: {entropy_method!r}. "
            f"Valid options: {list(ENTROPY_METHODS.keys())}"
        )

    evaluations = scenario.get("evaluations", {})
    judgments_by_model = extract_rule_judgments(evaluations, parse_fn)
    if not judgments_by_model:
        return None

    entropy_by_model: Dict[str, Dict[str, float]] = {}
    judgment_counts: Dict[str, Dict[str, int]] = {}
    for model, judgments_by_rule_key in judgments_by_model.items():
        entropy_by_model[model] = {}
        judgment_counts[model] = {}
        for rule_key, judgment_list in judgments_by_rule_key.items():
            judgment_counts[model][rule_key] = len(judgment_list)
            if judgment_list:
                entropy_by_model[model][rule_key] = entropy_fn(judgment_list)
            else:
                # omitted, not scored 0.0: zero entropy would read as perfect consensus and raise the reward
                logger.warning(
                    "No parseable judgments for %s/%s; excluding it from the "
                    "entropy mean rather than scoring it as consensus.",
                    model,
                    rule_key,
                )

    if not iter_entropy_values(entropy_by_model):
        return None

    return {
        "scenario": scenario.get("scenario", ""),
        "evaluations": evaluations,
        "judgments": judgments_by_model,
        "entropy": entropy_by_model,
        "judgment_count": judgment_counts,
    }


def aggregate_entropy(
    scenarios: Iterable[Dict[str, Any]],
    entropy_method: str,
    parse_fn: Callable,
) -> List[Dict[str, Any]]:
    """Return the entropy records of the scenarios that have parseable judgments."""
    results: List[Dict[str, Any]] = []
    skipped = 0
    for i, scenario in enumerate(scenarios):
        record = entropy_for_scenario(scenario, entropy_method, parse_fn)
        if record is None:
            logger.warning(f"Skipping scenario {i}: no parseable judgments")
            skipped += 1
            continue
        results.append(record)
    if skipped:
        logger.warning(f"Entropy: skipped {skipped} scenarios")
    return results


def _log_entropy_stats(records: List[Dict[str, Any]]) -> None:
    all_values = [v for r in records for v in iter_entropy_values(r.get("entropy", {}))]
    if all_values:
        logger.info(f"Entropy stats: min={min(all_values):.4f}, max={max(all_values):.4f}, mean={sum(all_values)/len(all_values):.4f}")


def compute_entropy_from_evals(
    evaluations_path: str | Path,
    output_path: str | Path,
    parse_fn: Callable,
    entropy_method: str = "consensus",
) -> List[Dict[str, Any]]:
    """Read an evaluations JSONL, compute per-scenario entropy and write the entropy JSONL."""
    logger.info(f"Computing entropy for {evaluations_path} using method '{entropy_method}'")
    records = aggregate_entropy(
        process_jsonl_file(evaluations_path),
        entropy_method=entropy_method,
        parse_fn=parse_fn,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data_to_jsonl(records, output_path)

    logger.info(f"Computed entropy for {len(records)} records -> {output_path}")
    _log_entropy_stats(records)
    return records


ENTROPY_METHODS = {
    "consensus": calculate_consensus_entropy_yes_no_unsure,
}
