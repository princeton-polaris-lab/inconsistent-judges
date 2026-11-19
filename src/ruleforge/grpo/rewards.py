from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from ruleforge.grpo.anchor_set import ANCHOR_CLASSES, CLASS_A, TIE_ORDER, mode_of
from ruleforge.judges.completion_format import CompletionFormat
from ruleforge.grpo.primitives import completion_text
from ruleforge.evaluate.scoring import PHI_MAP, max_rule_phi
from ruleforge.evaluate.identity import EVALUATION_MODE_FULL, FORCED_JSON_LABEL_ONLY
from ruleforge.evaluate.validation import EMPTY_RESPONSE, TRUNCATED, ComponentSchema
from ruleforge.prompts.parse import THINK_CLOSE, THINK_OPEN
from ruleforge.prompts.styles import FORCED_GRPO

#: Qwen3's finish ids (eos and pad) and `</think>` id.
QWEN3_FINISH_TOKEN_IDS = (151645, 151643)
QWEN3_THINK_END_TOKEN_ID = 151668
UNCLOSED_THINK = "unclosed_think"

LAMBDA_INT = 0.3
LAMBDA_COMP = 0.1

TRUNCATED_ZERO = "zero"
TRUNCATED_NONE = "none"
TRUNCATED_POLICIES = (TRUNCATED_ZERO, TRUNCATED_NONE)

#: The dataset columns the reward reads, one value per completion.
Q_COLUMNS = {"yes": "q_yes", "unsure": "q_unsure", "no": "q_no"}
CLASS_COLUMN = "anchor_class"
TOPPED_UP_COLUMN = "topped_up"
BASE_YES_RULES_COLUMN = "base_yes_rules"

#: Anchor-margin buckets of |q̂_yes - q̂_no|.
MARGIN_BUCKETS = (("0.0-0.2", 0.2), ("0.2-0.4", 0.4), ("0.4-1.0", math.inf))
SHORT_THINK_TOKENS = 32
METRIC_PREFIX = "consistency"


@dataclass(frozen=True)
class ParsedCompletion:
    """What one completion is, as the reward reads it."""

    finished: bool
    closed_think: bool
    #: Completion tokens before the last closing marker; None without one.
    think_tokens: Optional[int]
    verdict: Optional[str]
    failure: Optional[str]
    rule_map: Optional[dict[str, str]]
    #: A finished completion whose reasoning holds only whitespace.
    empty_think: bool = False
    #: What follows the last closing marker, stripped, when the reasoning closed.
    answer: Optional[str] = None
    preempted_rules: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.verdict is not None

    @property
    def complete_map(self) -> bool:
        return self.rule_map is not None


def parse_completion(
    text: str,
    completion_ids: Sequence[int],
    *,
    schema: ComponentSchema,
    finish_token_ids: Sequence[int] = QWEN3_FINISH_TOKEN_IDS,
    think_end_token_id: int = QWEN3_THINK_END_TOKEN_ID,
    decode: Optional[Callable[[Sequence[int]], str]] = None,
    think_open: str = THINK_OPEN,
) -> ParsedCompletion:
    """Parse one completion (finished, reasoning closed, the answer after the last closing marker, the judgment and the
    rule map), reading the closing marker from the ids when `decode` is given."""
    ids = list(completion_ids)
    finished = bool(ids) and ids[-1] in finish_token_ids
    think_tokens = len(ids) - 1 - ids[::-1].index(think_end_token_id) if think_end_token_id in ids else None
    closed = THINK_CLOSE in text if decode is None else think_tokens is not None

    def invalid(reason: str, **seen: Any) -> ParsedCompletion:
        return ParsedCompletion(finished, closed, think_tokens, None, reason, None, **seen)

    if not finished:
        return invalid(TRUNCATED)
    if not closed:
        return invalid(UNCLOSED_THINK)
    if decode is None:
        reasoning, answer = (part.strip() for part in text.rsplit(THINK_CLOSE, 1))
    else:
        reasoning, answer = decode(ids[:think_tokens]).strip(), decode(ids[think_tokens + 1 :]).strip()
    seen = dict(empty_think=not reasoning.removeprefix(think_open).strip(), answer=answer)
    if not answer:
        return invalid(EMPTY_RESPONSE, **seen)
    labelled = schema.classify_forced(answer, require_map=False)
    if not labelled.valid:
        return invalid(labelled.failure_reason, **seen)
    mapped = schema.classify_forced(answer, require_map=True)
    return ParsedCompletion(
        finished, closed, think_tokens, labelled.judgment, None, mapped.rule_judgments,
        preempted_rules=labelled.preempted_rules, **seen,
    )


@dataclass(frozen=True)
class RewardTerms:
    """One completion's terms; all None for a truncated completion under `truncated="none"`."""

    flat: Optional[float]
    internal: Optional[float]
    completeness: Optional[float]

    def total(self, lambda_int: float, lambda_comp: float) -> Optional[float]:
        """Return R less its validity gate."""
        if self.flat is None:
            return None
        return self.flat + lambda_int * self.internal + lambda_comp * self.completeness


def or_consistent(parsed: ParsedCompletion) -> bool:
    """Return whether a complete map's judgment is the OR of its own per-rule judgments."""
    return PHI_MAP[parsed.verdict] == max_rule_phi(list(parsed.rule_map.values()))


def score(parsed: ParsedCompletion, q: Mapping[str, float], *, truncated: str) -> RewardTerms:
    """Return the terms of one parsed completion against its row's q̂: 0 or None for a truncated one, 0 for an invalid
    answer; r_int and r_comp are gated on a complete map."""
    if not parsed.finished:
        value = 0.0 if truncated == TRUNCATED_ZERO else None
        return RewardTerms(value, value, value)
    if not parsed.valid:
        return RewardTerms(0.0, 0.0, 0.0)
    if not parsed.complete_map:
        return RewardTerms(float(q[parsed.verdict]), 0.0, 0.0)
    return RewardTerms(float(q[parsed.verdict]), 1.0 if or_consistent(parsed) else 0.0, 1.0)


def margin_bucket(q_yes: float, q_no: float) -> str:
    """Return the anchor-margin bucket of |q̂_yes - q̂_no|, rounded to 12 decimals first."""
    margin = round(abs(q_yes - q_no), 12)
    return next(name for name, upper in MARGIN_BUCKETS if margin < upper)


@dataclass(frozen=True)
class RowColumns:
    """The dataset columns of one batch, checked and aligned with its completions."""

    q: list[dict[str, float]]
    classes: list[str]
    topped_up: list[bool]
    base_yes_rules: list[frozenset[str]]

    @classmethod
    def read(cls, columns: Mapping[str, Any], n: int) -> "RowColumns":
        wanted = (*Q_COLUMNS.values(), CLASS_COLUMN, TOPPED_UP_COLUMN, BASE_YES_RULES_COLUMN)
        missing = [name for name in wanted if name not in columns]
        if missing:
            raise ValueError(f"the GRPO dataset lacks the reward columns {missing}")
        short = [name for name in wanted if len(columns[name]) != n]
        if short:
            raise ValueError(f"reward columns {short} do not align with the {n} completions")
        q = [{verdict: float(columns[name][i]) for verdict, name in Q_COLUMNS.items()} for i in range(n)]
        for i, row in enumerate(q):
            if not all(0.0 <= value <= 1.0 for value in row.values()) or abs(sum(row.values()) - 1.0) > 1e-9:
                raise ValueError(f"row {i}: q-hat {row} is not a distribution over the three judgments")
        classes = [str(value) for value in columns[CLASS_COLUMN]]
        unknown = sorted(set(classes) - set(ANCHOR_CLASSES))
        if unknown:
            raise ValueError(f"unknown anchor classes {unknown}")
        # a string would read as the set of its characters
        strings = [i for i, rules in enumerate(columns[BASE_YES_RULES_COLUMN]) if isinstance(rules, (str, bytes))]
        if strings:
            raise ValueError(f"row {strings[0]}: {BASE_YES_RULES_COLUMN} is a string "
                             f"({columns[BASE_YES_RULES_COLUMN][strings[0]]!r}), not a list of rule ids")
        return cls(
            q=q,
            classes=classes,
            topped_up=[bool(value) for value in columns[TOPPED_UP_COLUMN]],
            base_yes_rules=[frozenset(str(rule) for rule in rules) for rules in columns[BASE_YES_RULES_COLUMN]],
        )


def _batch_rate(flags: Sequence[bool]) -> float:
    return sum(1 for flag in flags if flag) / len(flags)


def _pooled(name: str, values: Sequence[float]) -> dict[str, float]:
    return {f"{name}/num": float(sum(values)), f"{name}/den": float(len(values))}


def _yes_rules(rule_map: Mapping[str, str]) -> list[str]:
    return [rule for rule, label in rule_map.items() if label == "yes"]


def zero_variance_group_share(totals: Sequence[Optional[float]], num_generations: int) -> float:
    """Return the share of groups (runs of `num_generations`) whose scored R do not vary."""
    if not totals or len(totals) % num_generations:
        raise ValueError(f"{len(totals)} completions are not whole groups of {num_generations}")
    groups = [totals[start : start + num_generations] for start in range(0, len(totals), num_generations)]
    flat = 0
    for group in groups:
        scored = [value for value in group if value is not None]
        flat += len(scored) < 2 or max(scored) == min(scored)
    return flat / len(groups)


def batch_telemetry(
    parsed: Sequence[ParsedCompletion],
    terms: Sequence[RewardTerms],
    columns: RowColumns,
    *,
    num_generations: int,
    lambda_int: float,
    lambda_comp: float,
    favoured_no_rate: bool = False,
) -> tuple[dict[str, float], dict[str, list[Any]]]:
    """Return `(metrics, extras)` of one batch under fixed names: a rate over a part that may be empty is logged as
    `<name>/num` and `<name>/den`, so TRL's mean over ranks pools correctly."""
    totals = [term.total(lambda_int, lambda_comp) for term in terms]
    finished_totals = [total if item.finished else None for item, total in zip(parsed, totals)]
    modes = [mode_of(q) for q in columns.q]
    buckets = [margin_bucket(q["yes"], q["no"]) for q in columns.q]
    rows = list(zip(parsed, columns.classes, modes, columns.base_yes_rules))
    verdicts = [(item.verdict, cls, mode) for item, cls, mode, _ in rows if item.valid]
    maps = [(item.rule_map, _yes_rules(item.rule_map), cls, base) for item, cls, _, base in rows if item.complete_map]
    thinks = [item.think_tokens for item in parsed if item.think_tokens is not None]
    metrics: dict[str, float] = {
        "valid_rate": _batch_rate([item.valid for item in parsed]),
        "clipped_rate": _batch_rate([not item.finished for item in parsed]),
        "unclosed_think_rate": _batch_rate([item.finished and not item.closed_think for item in parsed]),
        "empty_think_rate": _batch_rate([item.empty_think for item in parsed]),
        # text and ids disagree on </think>; meaningful on the text path only
        "think_marker_mismatch_rate": _batch_rate(
            [(item.think_tokens is not None) != item.closed_think for item in parsed]
        ),
        "complete_map_rate": _batch_rate([item.complete_map for item in parsed]),
        "zero_variance_group_share": zero_variance_group_share(totals, num_generations),
        "zero_variance_group_share_finished": zero_variance_group_share(finished_totals, num_generations),
    }
    pooled: dict[str, list[float]] = {
        "yes_rate": [verdict == "yes" for verdict, _, _ in verdicts],
        "yes_rate_class_a": [verdict == "yes" for verdict, cls, _ in verdicts if cls == CLASS_A],
        **({"yes_rate_favoured_no": [verdict == "yes" for verdict, _, mode in verdicts if mode == "no"]}
           if favoured_no_rate else {}),
        "mode_rate_nonyes_rows": [verdict == mode for verdict, _, mode in verdicts if mode != "yes"],
        "unsure_rate_majority_yes_rows": [verdict == "unsure" for verdict, cls, _ in verdicts if cls != CLASS_A],
        "map_yes_labels_mean": [len(yes) for _, yes, _, _ in maps],
        "map_majority_yes_share": [2 * len(yes) > len(rule_map) for rule_map, yes, _, _ in maps],
        "map_all_yes_share": [len(yes) == len(rule_map) for rule_map, yes, _, _ in maps],
        "map_all_no_share": [set(rule_map.values()) == {"no"} for rule_map, _, _, _ in maps],
        **{
            f"map_all_no_share_class_{name}": [
                set(rule_map.values()) == {"no"} for rule_map, _, cls, _ in maps if cls == name
            ]
            for name in ANCHOR_CLASSES
        },
        "flag_precision": [rule in base for _, yes, _, base in maps for rule in yes],
        "flag_overlap": [bool(set(yes) & base) for _, yes, _, base in maps if yes],
        "think_tokens_mean": thinks,
        "think_short_share": [count < SHORT_THINK_TOKENS for count in thinks],
    }
    for name, _ in MARGIN_BUCKETS:
        members = [i for i, bucket in enumerate(buckets) if bucket == name]
        scored = [terms[i] for i in members if terms[i].flat is not None]
        metrics[f"margin_{name}/share"] = len(members) / len(parsed)
        pooled[f"margin_{name}/r_flat"] = [term.flat for term in scored]
        pooled[f"margin_{name}/r_int"] = [term.internal for term in scored]
        pooled[f"margin_{name}/r_comp"] = [term.completeness for term in scored]
        for verdict in TIE_ORDER:
            pooled[f"margin_{name}/{verdict}_rate"] = [parsed[i].verdict == verdict for i in members if parsed[i].valid]
    for name, values in pooled.items():
        metrics.update(_pooled(name, values))
    extras = {
        "anchor_class": list(columns.classes),
        "topped_up": list(columns.topped_up),
        "margin_bucket": buckets,
        "map_yes_labels": [len(_yes_rules(item.rule_map)) if item.complete_map else None for item in parsed],
        "think_tokens": [item.think_tokens for item in parsed],
        "parse_failure": [item.failure for item in parsed],
    }
    return {f"{METRIC_PREFIX}/{name}": value for name, value in metrics.items()}, extras


class GrpoReward:
    """The three reward functions over one ruleset, sharing one parse per completion, with `truncated` required and
    `completion_format` replacing the default (Qwen3) finish and think-end ids."""

    def __init__(
        self,
        rules: Sequence[Mapping[str, Any]],
        *,
        truncated: str,
        num_generations: int,
        lambda_int: float = LAMBDA_INT,
        lambda_comp: float = LAMBDA_COMP,
        finish_token_ids: Sequence[int] = QWEN3_FINISH_TOKEN_IDS,
        think_end_token_id: int = QWEN3_THINK_END_TOKEN_ID,
        completion_format: Optional[CompletionFormat] = None,
        favoured_no_rate: bool = False,
    ) -> None:
        if truncated not in TRUNCATED_POLICIES:
            raise ValueError(f"truncated must be one of {TRUNCATED_POLICIES}, got {truncated!r}")
        if num_generations < 1:
            raise ValueError(f"num_generations must be positive, got {num_generations}")
        self.decode: Optional[Callable[[Sequence[int]], str]] = None
        self.think_open = THINK_OPEN
        self.other_stop_token_ids: tuple[int, ...] = ()
        if completion_format is not None:
            if tuple(finish_token_ids) != QWEN3_FINISH_TOKEN_IDS or think_end_token_id != QWEN3_THINK_END_TOKEN_ID:
                raise ValueError("pass the finish and think-end ids, or a completion_format, not both")
            finish_token_ids = completion_format.finish_token_ids
            think_end_token_id = completion_format.think_end_token_id
            self.decode = completion_format.decode
            self.think_open = completion_format.think_open
            self.other_stop_token_ids = tuple(completion_format.other_stop_token_ids)
        self.schema = ComponentSchema.from_rules(
            rules,
            prompt_style=FORCED_GRPO,
            forced_json_requirement=FORCED_JSON_LABEL_ONLY,
            evaluation_mode=EVALUATION_MODE_FULL,
        )
        self.truncated = truncated
        self.num_generations = num_generations
        self.lambda_int = lambda_int
        self.lambda_comp = lambda_comp
        self.favoured_no_rate = bool(favoured_no_rate)
        self.finish_token_ids = tuple(finish_token_ids)
        self.think_end_token_id = think_end_token_id
        self._cache: dict[tuple[str, tuple[int, ...]], ParsedCompletion] = {}

    @property
    def reward_funcs(self) -> list[Callable[..., list[Optional[float]]]]:
        return [self.flat_soft_reward, self.internal_consistency_reward, self.map_completeness_reward]

    @property
    def reward_weights(self) -> list[float]:
        return [1.0, self.lambda_int, self.lambda_comp]

    def parse(self, completions: Sequence[Any], completion_ids: Sequence[Sequence[int]]) -> list[ParsedCompletion]:
        """Parse each completion once; the cache keeps the current batch only."""
        if len(completions) != len(completion_ids):
            raise ValueError(f"{len(completions)} completions but {len(completion_ids)} completion_ids")
        keys = [(completion_text(completion), tuple(ids)) for completion, ids in zip(completions, completion_ids)]
        parsed = []
        for text, ids in keys:
            cached = self._cache.get((text, ids))
            if cached is None:
                cached = parse_completion(
                    text, ids, schema=self.schema, finish_token_ids=self.finish_token_ids,
                    think_end_token_id=self.think_end_token_id, decode=self.decode, think_open=self.think_open,
                )
            parsed.append(cached)
        self._cache = dict(zip(keys, parsed))
        return parsed

    def _terms(
        self, completions: Sequence[Any], completion_ids: Sequence[Sequence[int]], columns: Mapping[str, Any]
    ) -> tuple[list[ParsedCompletion], list[RewardTerms], RowColumns]:
        parsed = self.parse(completions, completion_ids)
        rows = RowColumns.read(columns, len(parsed))
        terms = [score(item, q, truncated=self.truncated) for item, q in zip(parsed, rows.q)]
        return parsed, terms, rows

    def flat_soft_reward(self, completions, completion_ids, **columns) -> list[Optional[float]]:
        """Return `valid * q̂[φ(judgment)]` per completion and log the batch's telemetry."""
        parsed, terms, rows = self._terms(completions, completion_ids, columns)
        metrics, extras = batch_telemetry(
            parsed, terms, rows,
            num_generations=self.num_generations, lambda_int=self.lambda_int, lambda_comp=self.lambda_comp,
            favoured_no_rate=self.favoured_no_rate,
        )
        if self.other_stop_token_ids:
            # logged only under a format with other stop ids: every rank runs one format
            ended = [bool(ids) and ids[-1] in self.other_stop_token_ids for ids in completion_ids]
            metrics[f"{METRIC_PREFIX}/other_stop_rate"] = _batch_rate(ended)
            metrics[f"{METRIC_PREFIX}/cap_clipped_rate"] = _batch_rate(
                [not item.finished and not stopped for item, stopped in zip(parsed, ended)])
        log_metric, log_extra = columns.get("log_metric"), columns.get("log_extra")
        if log_metric is not None:
            for name, value in metrics.items():
                log_metric(name, value)
        if log_extra is not None:
            for name, values in extras.items():
                log_extra(name, values)
        return [term.flat for term in terms]

    def internal_consistency_reward(self, completions, completion_ids, **columns) -> list[Optional[float]]:
        """Return `valid * comp * 1[φ(judgment) = max_i φ(map_i)]` per completion."""
        return [term.internal for term in self._terms(completions, completion_ids, columns)[1]]

    def map_completeness_reward(self, completions, completion_ids, **columns) -> list[Optional[float]]:
        """Return `valid * comp` per completion."""
        return [term.completeness for term in self._terms(completions, completion_ids, columns)[1]]
