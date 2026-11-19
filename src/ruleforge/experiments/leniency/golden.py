from __future__ import annotations

import json
import math
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from ruleforge.evaluate.inference import INFERENCE_METHOD, PRIMARY_INTERVAL_BASIS, TEST_METHOD
from ruleforge.evaluate.multiplicity import HOLM_MC_MARGIN_SE, holm_flags
from ruleforge.evaluate.report import transition_summary
from ruleforge.experiments.checks import require
from ruleforge.experiments.goldens import read_json_golden
from ruleforge.experiments.labels import CAMPAIGN_MODEL_LABELS
from ruleforge.experiments.leniency.schema import (
    AGREEMENT_METHOD,
    ALONE,
    DATA_VERSION,
    DIRECTION_COUNT_FIELDS,
    FEW_INFORMATIVE_BELOW,
    GOLDEN_SOURCE,
    IN_CONTEXT,
    LABELS,
    METRIC,
    TRANSITION_METHOD,
    RuleContextError,
    check_holm,
    direction_counts,
)
from ruleforge.prompts.styles import FORCED_JSON_STYLES, lists_preemption

_require = partial(require, error=RuleContextError)


#: The largest magnitude a golden's number may have: past it an integer is no longer exact as a float.
MAX_MAGNITUDE = 2**53
#: The deepest a golden's objects and arrays may nest: past a few hundred levels ``==`` or ``repr`` raises
#: RecursionError.
MAX_DEPTH = 64


def nesting_depth(value: Any, *, limit: int = MAX_DEPTH) -> int:
    """Return how deep ``value`` nests, without recursion; any depth past ``limit`` is reported as ``limit + 1``."""
    deepest, stack = 0, [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, Mapping):
            children = node.values()
        elif isinstance(node, (list, tuple)):
            children = node
        else:
            continue
        deepest = max(deepest, depth + 1)
        if deepest > limit:
            return limit + 1
        stack.extend((child, depth + 1) for child in children)
    return deepest


def require_shallow(value: Any, where: str, *, error: type[RuleContextError] = RuleContextError) -> None:
    """Refuse a golden nested past :data:`MAX_DEPTH`, before anything compares or prints a value of it."""
    _require(
        [(nesting_depth(value) <= MAX_DEPTH, f"nested past {MAX_DEPTH} levels: not a golden this pipeline wrote")],
        where,
        error=error,
    )


read_golden = partial(read_json_golden, error=RuleContextError)


def validate_as_written(
    payload: Mapping[str, Any],
    validate: Callable[[Mapping[str, Any]], None],
    where: str,
    *,
    error: type[RuleContextError] = RuleContextError,
) -> None:
    """Refuse a just-built golden that its own re-read (``validate``, as ``--from_golden`` runs it) refuses."""
    try:
        validate(json.loads(json.dumps(payload, sort_keys=True)))
    except RuleContextError as refusal:
        raise error(f"{where} as built is one its own re-read refuses: {refusal}") from refusal


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and abs(value) <= MAX_MAGNITUDE


def _is_real(value: Any) -> bool:
    return _is_number(value) and abs(value) <= MAX_MAGNITUDE


def _is_label_map(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == set(LABELS) and all(map(_is_real, value.values()))


def _is_label_matrix(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == set(LABELS) and all(map(_is_label_map, value.values()))


def json_equal(a: Any, b: Any) -> bool:
    """Return ``a == b`` without Python's coercions between JSON types (``500.0`` is not ``500``, ``True`` is not
    ``1``), walked with a stack."""
    pairs = [(a, b)]
    while pairs:
        a, b = pairs.pop()
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                return False
            pairs.extend((a[key], b[key]) for key in a)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            pairs.extend(zip(a, b))
        elif not (type(a) is type(b) and a == b):
            return False
    return True


def _is_yes_share(rate: float, n_cells: int) -> bool:
    return 0 <= rate <= 1 and abs(rate * n_cells - round(rate * n_cells)) <= 1e-9


_TYPES = {
    "a string": lambda value: isinstance(value, str),
    "an integer": _is_int,
    "a real number": _is_real,
    "a boolean": lambda value: isinstance(value, bool),
}
#: What :func:`validate_payload` reads from each effects row and each model's summary, by JSON type; checked before any
#: arithmetic.
ROW_TYPES = {
    "a string": ("rule_id", "title"),
    "an integer": (
        "rule", "n_scenarios", "n_decoder_draws", "yes_difference", "n_informative_scenarios", "n_permutations",
    ),
    "a real number": (
        "delta", "rate_alone", "rate_in_context", "ci_low", "ci_high", "p_value", "p_holm",
        "phi_alone", "phi_in_context", "phi_delta",
    ),
    "a boolean": ("few_informative", "degenerate", "significant_holm", "p_value_is_floor", "holm_mc_margin"),
}
SUMMARY_TYPES = {
    "an integer": ("n_cells", "pooled_yes_difference", *DIRECTION_COUNT_FIELDS),
    "a real number": ("pooled_delta", "pooled_rate_alone", "pooled_rate_in_context"),
}
#: What :func:`validate_payload` reads from each stratum of a ``by_verdict`` decomposition, by JSON type.
STRATUM_TYPES = {"an integer": ("n_responses",), "a real number": ("delta_contribution", "share_of_responses")}
#: What the summary and the flips figure read from each agreement block beside its matrix, by JSON type.
AGREEMENT_TYPES = {"a real number": ("unchanged_fraction", "chance_agreement", "kappa")}
METHODS = {
    "inference_method": INFERENCE_METHOD,
    "interval_basis": PRIMARY_INTERVAL_BASIS,
    "test_method": TEST_METHOD,
    "transition_method": TRANSITION_METHOD,
    "agreement_method": AGREEMENT_METHOD,
}
#: The range of a delta, or of a difference of two rates: a stored interval's default bounds.
DELTA_RANGE = (-1.0, 1.0)


def interval_check(record: Mapping[str, Any], bounds: Tuple[float, float] = DELTA_RANGE) -> Tuple[bool, str]:
    """Check that a typed record's ``[ci_low, ci_high]`` is ordered within ``bounds``."""
    low, high = bounds
    return (
        low <= record["ci_low"] <= record["ci_high"] <= high,
        f"interval [{record['ci_low']!r}, {record['ci_high']!r}] is not ordered within [{low:g}, {high:g}]",
    )


def p_value_check(p_value: float, n_permutations: int) -> Tuple[bool, str]:
    """Check a typed Monte Carlo p over ``B`` sign flips against its lattice: ``(extreme + 1) / (B + 1)`` for a whole
    ``0 <= extreme <= B``."""
    steps = p_value * (n_permutations + 1)
    return (
        0 < p_value <= 1 and round(steps) >= 1 and abs(steps - round(steps)) <= 1e-6,
        f"p_value {p_value!r} is not a probability on the {n_permutations}-permutation test's lattice "
        f"k / {n_permutations + 1}, 1 <= k <= {n_permutations + 1}",
    )


def inference_checks(
    row: Mapping[str, Any], *, n_permutations: int, bounds: Tuple[float, float] = DELTA_RANGE
) -> List[Tuple[bool, str]]:
    """Check a typed tested row's permutation count, p and interval, none of which a golden can recompute."""
    return [
        (
            row["n_permutations"] == n_permutations,
            f"n_permutations {row['n_permutations']!r}, not the settings' randomization test's {n_permutations}",
        ),
        p_value_check(row["p_value"], n_permutations),
        interval_check(row, bounds),
    ]


def informative_checks(row: Mapping[str, Any]) -> List[Tuple[bool, str]]:
    """Check a typed effects row's ``n_informative_scenarios`` against what its grid and exact count allow, and
    recompute ``few_informative`` from it."""
    n_informative = row["n_informative_scenarios"]
    return [
        (
            0 <= n_informative <= row["n_scenarios"]
            and abs(row["yes_difference"]) <= n_informative * row["n_decoder_draws"],
            f"n_informative_scenarios {n_informative!r} is not a count of the {row['n_scenarios']!r} scenarios that "
            f"can hold yes_difference {row['yes_difference']!r} at {row['n_decoder_draws']!r} draws each",
        ),
        (
            row["few_informative"] == (n_informative < FEW_INFORMATIVE_BELOW),
            f"few_informative {row['few_informative']!r} disagrees with "
            f"n_informative_scenarios {n_informative!r} < {FEW_INFORMATIVE_BELOW}",
        ),
    ]


def _type_refusal(record: Mapping[str, Any], key: str, kind: str) -> str:
    if key not in record:
        return f"{key} is missing"
    value = record[key]
    past = _is_number(value) and not abs(value) <= MAX_MAGNITUDE
    return f"{key} {value!r} is not {kind}" + (" within +-2**53" if past else "")


def typed(record: Any, types: Mapping[str, Sequence[str]], where: str) -> Mapping[str, Any]:
    """Return ``record``, refused unless it is an object whose fields have these JSON types (a number within
    :data:`MAX_MAGNITUDE`)."""
    _require([(isinstance(record, Mapping), f"{type(record).__name__} is not an object")], where)
    _require(
        [
            (_TYPES[kind](record.get(key)), _type_refusal(record, key, kind))
            for kind, keys in types.items()
            for key in keys
        ],
        where,
    )
    return record


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Refuse a golden this module would not have written, as a :class:`RuleContextError`, never a TypeError, KeyError
    or the like."""
    require_shallow(payload, "golden")
    _require([(isinstance(payload, Mapping), f"{type(payload).__name__} is not an object")], "golden")
    _require(
        [
            (
                json_equal(payload.get("data_version"), DATA_VERSION),
                f"data_version {payload.get('data_version')!r}, not {DATA_VERSION}; rebuild it from --estimates",
            ),
            (payload.get("source") == GOLDEN_SOURCE, f"source {payload.get('source')!r}, not {GOLDEN_SOURCE!r}"),
            (payload.get("metric") == METRIC, f"metric {payload.get('metric')!r}, not {METRIC!r}"),
        ],
        "golden",
    )
    settings, rules, models, provenance = (payload.get(key) for key in ("settings", "rules", "models", "provenance"))
    campaigns = provenance.get("campaigns") if isinstance(provenance, Mapping) else None
    _require(
        [
            (isinstance(settings, Mapping), "settings is not an object"),
            (
                isinstance(rules, list)
                and bool(rules)
                and all(
                    isinstance(rule, Mapping) and _is_int(rule.get("number")) and isinstance(rule.get("title"), str)
                    for rule in rules
                )
                and len({rule["number"] for rule in rules}) == len(rules),
                "rules is not a non-empty list of objects with distinct integer numbers and string titles",
            ),
            (
                isinstance(models, Mapping) and bool(models) and all(isinstance(b, Mapping) for b in models.values()),
                "models is not a non-empty object of model blocks",
            ),
            (
                isinstance(provenance, Mapping) and isinstance(provenance.get("complete"), bool),
                "provenance.complete is not a boolean",
            ),
            (
                isinstance(campaigns, list)
                and all(
                    isinstance(campaign, Mapping)
                    and isinstance(campaign.get("campaign_id"), str)
                    and isinstance(campaign.get("forced_conditions"), list)
                    for campaign in campaigns
                ),
                "provenance.campaigns is not a list of campaign records, each with a campaign_id and a "
                "forced_conditions list",
            ),
        ],
        "golden",
    )
    titles = {rule["number"]: rule["title"] for rule in rules}
    family_size, few_below, margin_se, alpha = (
        settings.get(key) for key in ("family_size", "few_informative_below", "holm_mc_margin_se", "alpha")
    )
    grid = (settings.get("n_scenarios"), settings.get("n_decoder_draws"))
    prompt_style = settings.get("prompt_style")
    randomization = settings.get("randomization_test")
    n_permutations = randomization.get("n_permutations") if isinstance(randomization, Mapping) else None
    bootstrap = settings.get("bootstrap")
    _require(
        [
            (
                prompt_style in FORCED_JSON_STYLES,
                f"prompt_style {prompt_style!r} is not one of the forced styles {FORCED_JSON_STYLES}",
            ),
            (_is_int(family_size) and family_size == len(titles), f"family_size {family_size!r}, {len(titles)} rules"),
            (
                json_equal(few_below, FEW_INFORMATIVE_BELOW),
                f"few_informative_below {few_below!r}, not this module's {FEW_INFORMATIVE_BELOW}: "
                "its few_informative flags were drawn at another threshold",
            ),
            (
                json_equal(margin_se, HOLM_MC_MARGIN_SE),
                f"holm_mc_margin_se {margin_se!r}, not this module's {HOLM_MC_MARGIN_SE!r}: "
                "its holm_mc_margin flags cannot be recomputed here",
            ),
            *(
                (settings.get(key) == value, f"{key} {settings.get(key)!r}, not this module's {value!r}")
                for key, value in METHODS.items()
            ),
            (
                all(_is_int(n) and n > 0 for n in grid),
                f"settings grid {grid[0]!r} x {grid[1]!r} is not two positive integers",
            ),
            (_is_real(alpha) and 0 < alpha < 1, f"alpha {alpha!r} is not a level in (0, 1)"),
            (
                _is_int(n_permutations) and n_permutations > 0,
                f"randomization_test.n_permutations {n_permutations!r} is not a positive integer",
            ),
            # The figures name the interval's level.
            (
                isinstance(bootstrap, Mapping)
                and _is_real(bootstrap.get("ci_level"))
                and 0 < bootstrap["ci_level"] < 1
                and _is_int(bootstrap.get("n_resamples"))
                and bootstrap["n_resamples"] > 0
                and _is_int(bootstrap.get("seed")),
                f"bootstrap {bootstrap!r} is not a ci_level in (0, 1) over a positive integer n_resamples, with an "
                "integer seed",
            ),
        ],
        "golden",
    )
    for label, block in models.items():
        _validate_identity(label, block, campaigns)
        _validate_model(
            label, block, titles=titles, grid=grid, alpha=alpha, prompt_style=prompt_style,
            n_permutations=n_permutations,
        )


def _validate_identity(label: str, block: Mapping[str, Any], campaigns: Sequence[Mapping[str, Any]]) -> None:
    model_name, campaign_id, condition_id = (block.get(key) for key in ("model_name", "campaign_id", "condition_id"))
    _require(
        [
            (
                isinstance(model_name, str) and CAMPAIGN_MODEL_LABELS.get(model_name) == label,
                f"model_name {model_name!r} is not the judge CAMPAIGN_MODEL_LABELS labels {label!r}",
            ),
            (
                isinstance(condition_id, str)
                and any(
                    campaign["campaign_id"] == campaign_id and condition_id in campaign["forced_conditions"]
                    for campaign in campaigns
                ),
                f"condition_id {condition_id!r} is not a forced condition that the record of its campaign "
                f"{campaign_id!r} lists in provenance.campaigns",
            ),
        ],
        label,
    )


def _named_part(
    container: Mapping[str, Any], types: Mapping[str, Sequence[str]], prompt_style: str, where: str
) -> Optional[Mapping[str, Any]]:
    part = container.get("named_preempted")
    if lists_preemption(prompt_style):
        return typed(part, types, where)
    _require(
        [
            (
                "named_preempted" in container and part is None,
                f"{part!r} is not null: the {prompt_style} prompt asks for no preempted_rules"
                if "named_preempted" in container else "is missing (null where the prompt asks for no preempted_rules)",
            )
        ],
        where,
    )
    return None


def _validate_model(
    label: str,
    block: Mapping[str, Any],
    *,
    titles: Mapping[int, str],
    grid: Tuple[int, int],
    alpha: float,
    prompt_style: str,
    n_permutations: int,
) -> None:
    effects = block.get("effects")
    _require([(isinstance(effects, list), "effects is not a list")], label)
    for k, row in enumerate(effects):
        name = row.get("rule_id") if isinstance(row, Mapping) and isinstance(row.get("rule_id"), str) else f"effects[{k}]"
        where = f"{label} {name}"
        typed(row, ROW_TYPES, where)
        _named_part(
            row, {"an integer": ("n_cells",), "a real number": ("delta_contribution",)}, prompt_style,
            f"{where} named_preempted",
        )
        transitions = row.get("transitions")
        _require(
            [
                *inference_checks(row, n_permutations=n_permutations),
                (
                    _is_label_matrix(transitions),
                    f"transitions is not a {{label: {{label: number}}}} matrix over {LABELS}",
                ),
                (_is_label_map(row.get("by_verdict")), f"by_verdict is not a {{label: number}} map over {LABELS}"),
            ],
            where,
        )
    covered = sorted(row["rule"] for row in effects)
    _require([(covered == sorted(titles), f"effects cover rules {covered}, the golden's {sorted(titles)}")], label)
    check_holm(effects, alpha=alpha, where=label)
    n_cells = grid[0] * grid[1]
    for row, flags in zip(effects, holm_flags(effects, alpha=alpha)):
        delta, yes_difference = row["delta"], row["yes_difference"]
        shares = sum(value for matrix_row in row["transitions"].values() for value in matrix_row.values())
        named = row["named_preempted"]
        _require(
            [
                (
                    (row["n_scenarios"], row["n_decoder_draws"]) == grid,
                    f"grid {row['n_scenarios']!r} x {row['n_decoder_draws']!r}, the settings' "
                    f"{grid[0]} x {grid[1]}",
                ),
                (row["title"] == titles[row["rule"]], f"title {row['title']!r} is not the rules' {titles[row['rule']]!r}"),
                # Exact to float rounding, so it also fixes delta's sign and ties.
                (
                    abs(delta - yes_difference / n_cells) <= 1e-12,
                    f"delta {delta!r} is not yes_difference / (n_scenarios x n_decoder_draws) "
                    f"= {yes_difference} / {n_cells}",
                ),
                # The rates are whole 'yes' counts over the grid.
                *(
                    (
                        _is_yes_share(row[f"rate_{side}"], n_cells),
                        f"rate_{side} {row[f'rate_{side}']!r} is not a count of 'yes' over the {n_cells} cells",
                    )
                    for side in (ALONE, IN_CONTEXT)
                ),
                # The phi means are means of PHI_MAP's values, in [0, 1]; the draws they need are not in the golden.
                (
                    0 <= row["phi_alone"] <= 1
                    and 0 <= row["phi_in_context"] <= 1
                    and abs(row["phi_delta"] - (row["phi_in_context"] - row["phi_alone"])) <= 1e-12,
                    f"phi_alone {row['phi_alone']!r} / phi_in_context {row['phi_in_context']!r} are not means in "
                    f"[0, 1] with phi_delta {row['phi_delta']!r} their difference",
                ),
                (
                    named is None or 0 <= named["n_cells"] <= n_cells,
                    f"named_preempted n_cells {(named or {}).get('n_cells')!r} is not a count of the {n_cells} cells",
                ),
                (math.isclose(shares, 1.0, abs_tol=1e-9), "transition shares do not sum to 1"),
                (
                    math.isclose(sum(row["by_verdict"].values()), delta, abs_tol=1e-9),
                    "by-judgment contributions do not sum to delta",
                ),
                (
                    math.isclose(row["rate_in_context"] - row["rate_alone"], delta, abs_tol=1e-9),
                    "delta is not rate_in_context - rate_alone",
                ),
                *informative_checks(row),
                (
                    row["degenerate"] == (row["ci_low"] == row["ci_high"]),
                    f"degenerate {row['degenerate']!r} disagrees with the interval "
                    f"[{row['ci_low']!r}, {row['ci_high']!r}]",
                ),
                (
                    all(row[key] == value for key, value in flags.items()),
                    "p_value_is_floor / holm_mc_margin disagree with the recomputation",
                ),
            ],
            f"{label} {row['rule_id']}",
        )
    _validate_pooled(label, block, effects, n_cells=n_cells, n_rules=len(titles), prompt_style=prompt_style)
    _validate_flips(label, block.get("flips"), effects)


def _validate_pooled(
    label: str,
    block: Mapping[str, Any],
    effects: Sequence[Mapping[str, Any]],
    *,
    n_cells: int,
    n_rules: int,
    prompt_style: str,
) -> None:
    summary = typed(block.get("summary"), SUMMARY_TYPES, f"{label} summary")
    by_verdict = block.get("by_verdict")
    strata = by_verdict.get("strata") if isinstance(by_verdict, Mapping) else None
    _require(
        [(isinstance(strata, Mapping) and set(strata) == set(LABELS), f"by_verdict.strata is not an object over {LABELS}")],
        label,
    )
    parts = {f"by_verdict {v}": strata[v] for v in LABELS}
    for name, part in parts.items():
        typed(part, STRATUM_TYPES, f"{label} {name}")
    named = _named_part(
        block, {"an integer": ("n_named_cells",), "a real number": ("delta_contribution",)}, prompt_style,
        f"{label} named_preempted",
    )
    if named is not None:
        parts["named_preempted"] = named
    pooled, cells = summary["pooled_yes_difference"], n_cells * n_rules
    yes = {side: sum(round(row[f"rate_{side}"] * n_cells) for row in effects) for side in (ALONE, IN_CONTEXT)}
    counts = {key: summary[key] for key in DIRECTION_COUNT_FIELDS}
    recount = dict(zip(DIRECTION_COUNT_FIELDS, direction_counts(effects)))
    responses = [strata[v]["n_responses"] for v in LABELS]
    _require(
        [
            (
                pooled == sum(row["yes_difference"] for row in effects),
                f"pooled_yes_difference {pooled!r} is not the sum of the rules' yes_difference",
            ),
            (summary["n_cells"] == cells, f"n_cells {summary['n_cells']!r}, not {n_rules} rules x {n_cells}"),
            (counts == recount, f"summary counts {counts} do not recount from the rows: {recount}"),
            (
                min(responses) >= 0 and sum(responses) == n_cells,
                f"by_verdict n_responses {responses} are not counts summing to the {n_cells} (scenario, draw) cells",
            ),
            *(
                (
                    abs(strata[v]["share_of_responses"] - strata[v]["n_responses"] / n_cells) <= 1e-12,
                    f"by_verdict {v} share_of_responses {strata[v]['share_of_responses']!r} is not n_responses / "
                    f"the cells = {strata[v]['n_responses']} / {n_cells}",
                )
                for v in LABELS
            ),
            (
                abs(summary["pooled_delta"] - pooled / cells) <= 1e-12,
                f"pooled_delta {summary['pooled_delta']!r} is not pooled_yes_difference / n_cells "
                f"= {pooled} / {cells}",
            ),
            *(
                (
                    abs(summary[f"pooled_rate_{side}"] - yes[side] / cells) <= 1e-12,
                    f"pooled_rate_{side} {summary[f'pooled_rate_{side}']!r} is not the rows' 'yes' count "
                    f"{yes[side]} / {cells}",
                )
                for side in (ALONE, IN_CONTEXT)
            ),
            *(
                (
                    "share_of_pooled_delta" in part and (part["share_of_pooled_delta"] is None) == (pooled == 0),
                    f"{name} share_of_pooled_delta {part.get('share_of_pooled_delta', '(missing)')!r} with "
                    f"pooled_yes_difference {pooled}: it is null exactly where that count is 0",
                )
                for name, part in parts.items()
            ),
            *([] if named is None else _named_checks(named, effects)),
        ],
        label,
    )
    # A non-null share has a non-zero pooled count; its float is held non-zero too.
    pooled_delta = summary["pooled_delta"]
    _require(
        [
            (
                math.isclose(sum(strata[v]["delta_contribution"] for v in LABELS), pooled_delta, abs_tol=1e-12),
                "by-judgment contributions do not sum to pooled_delta",
            ),
            *(
                (
                    part["share_of_pooled_delta"] is None
                    or (
                        _is_real(part["share_of_pooled_delta"])
                        and pooled_delta != 0
                        and math.isclose(
                            part["share_of_pooled_delta"], part["delta_contribution"] / pooled_delta, rel_tol=1e-9
                        )
                    ),
                    f"{name} share_of_pooled_delta {part['share_of_pooled_delta']!r} is not "
                    "delta_contribution / pooled_delta",
                )
                for name, part in parts.items()
            ),
        ],
        label,
    )


def _named_checks(named: Mapping[str, Any], effects: Sequence[Mapping[str, Any]]) -> List[Tuple[bool, str]]:
    n_named = sum(row["named_preempted"]["n_cells"] for row in effects)
    contribution = sum(row["named_preempted"]["delta_contribution"] for row in effects) / len(effects)
    rate = named.get("in_context_no_rate")
    return [
        (
            named["n_named_cells"] == n_named,
            f"named_preempted n_named_cells {named['n_named_cells']!r} is not the rules' sum {n_named}",
        ),
        (
            abs(named["delta_contribution"] - contribution) <= 1e-12,
            f"named_preempted delta_contribution {named['delta_contribution']!r} is not the rules' mean {contribution!r}",
        ),
        (
            "in_context_no_rate" in named
            and (rate is None) == (n_named == 0)
            and (rate is None or (_is_real(rate) and 0 <= rate <= 1)),
            f"named_preempted in_context_no_rate {named.get('in_context_no_rate', '(missing)')!r} is not a rate in "
            f"[0, 1] that is null exactly where no cell is named ({n_named} are)",
        ),
    ]


def _agreement_checks(name: str, record: Mapping[str, Any]) -> List[Tuple[bool, str]]:
    """Check a typed agreement block over a label matrix: its shares first, since the summary divides by their squared
    total, then its figures against the matrix."""
    matrix = record["matrix"]
    shares = [matrix[a][b] for a in LABELS for b in LABELS]
    if not (min(shares) >= 0 and math.isclose(sum(shares), 1.0, abs_tol=1e-9)):
        return [(False, f"{name} matrix is not non-negative shares summing to 1")]
    summary = transition_summary(matrix)
    return [
        (
            abs(record[key] - summary[key]) <= 1e-12,
            f"{name} {key} {record[key]!r} is not its matrix's {summary[key]!r}",
        )
        for key in AGREEMENT_TYPES["a real number"]
    ]


def _validate_flips(label: str, flips: Any, effects: Sequence[Mapping[str, Any]]) -> None:
    where = f"{label} flips"
    flips = typed(flips, {"a string": ("method",), **AGREEMENT_TYPES}, where)
    references = flips.get("draw_to_draw")
    _require([(isinstance(references, Mapping), "draw_to_draw is not an object")], where)
    parts = {"flips": flips, **{f"draw_to_draw {side}": references.get(side) for side in (ALONE, IN_CONTEXT)}}
    for name, part in parts.items():
        typed(part, AGREEMENT_TYPES, f"{label} {name}")
        _require(
            [(_is_label_matrix(part.get("matrix")), f"matrix is not a {{label: {{label: number}}}} matrix over {LABELS}")],
            f"{label} {name}",
        )
    mean = {a: {b: sum(row["transitions"][a][b] for row in effects) / len(effects) for b in LABELS} for a in LABELS}
    _require(
        [
            (flips["method"] == TRANSITION_METHOD, f"method {flips['method']!r}, not {TRANSITION_METHOD!r}"),
            (
                references.get("method") == AGREEMENT_METHOD,
                f"draw_to_draw method {references.get('method')!r}, not {AGREEMENT_METHOD!r}",
            ),
            (
                all(abs(flips["matrix"][a][b] - mean[a][b]) <= 1e-9 for a in LABELS for b in LABELS),
                "matrix is not the mean of the rules' transitions",
            ),
            *(check for name, part in parts.items() for check in _agreement_checks(name, part)),
        ],
        where,
    )
