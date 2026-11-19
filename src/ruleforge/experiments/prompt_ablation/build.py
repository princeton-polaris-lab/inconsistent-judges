from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


from ruleforge.evaluate.estimates import (
    SHARED_DESIGN_FIELDS,
    CampaignEstimatesError,
    estimates_source,
    is_complete,
    load_estimates,
    recorded_stack_mismatch_refusal,
)
from ruleforge.experiments import driver, goldens
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, campaign_label, display_order
from ruleforge.evaluate.multiplicity import ALPHA
from ruleforge.experiments.prompt_ablation.analysis import (
    BH_METHOD,
    FORCED_B,
    PROMPT_BARS,
    BAR_POPULATION,
    HOLM_MC_MARGIN_NOTE,
    HOLM_METHOD,
    SAME_VS_SEPARATE_KEYS,
    SLOT_STYLES,
    TABLE_ADJUSTMENT_FAMILY,
    TABLE_CONTRASTS,
    TABLE_METRIC,
    TABLE_SCHEMA_VERSION,
    TABLE_SOURCE,
    add_table_multiplicity,
    rate_bar,
    table_row,
    validate_table,
)
from ruleforge.experiments.prompt_ablation.figure import draw
from ruleforge.experiments.paper_outputs import paper_output_refusal, paper_paths
from ruleforge.evaluate.inference import RepeatedSamplingError
from ruleforge.core import record_names
from ruleforge.core.record_names import SAME_CALL_RATE_KEY, SEPARATE_VS_SAME_CALL_KEY
from ruleforge.core.paths import repository_root
from ruleforge.experiments.prompt_ablation.arms import build_payload as build_arm_payload
from ruleforge.experiments.prompt_ablation.arms import (
    add_arm_map_var_argument,
    arm_map_var_refusal,
    paper_table_refusal,
    parse_variables,
    validate_cross_stack,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.artifacts import write_text_atomically

logger = logging.getLogger(__name__)

METRIC = TABLE_METRIC
POPULATION = BAR_POPULATION
#: The figure's bar set in drawn order: the three styles scored separate-call and the forced arm scored same-call.
DRAWN_BARS = PROMPT_BARS

#: The slot of each prompt style a table of campaign roots shows.
STYLE_SLOTS: Dict[str, str] = {style: slot for slot, style in SLOT_STYLES.items()}

#: ``{label: {slot: condition summary}}``: one model's conditions, joined across roots.
ConditionsByModel = Dict[str, Dict[str, Dict[str, Any]]]
#: ``{label: {population: {bar: {mean, ci_low, ci_high, n, degenerate}}}}``, what the panels read.
BarBlock = Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]


REFUSALS = (CampaignEstimatesError, RepeatedSamplingError, SamplingDesignError)

CampaignTableError = CampaignEstimatesError


def _model_name(styles: Mapping[str, Mapping[str, Any]]) -> str:
    return next(iter(styles.values()))["model_name"]


def public_condition_id(condition_id: str) -> str:
    """Return a ``<model>/<style>`` condition id as the golden names it, its style segment by the style's public name."""
    return record_names.public_condition_id(condition_id, condition_id.split("/", 1)[-1])


def conditions_by_model(payloads: Iterable[Mapping[str, Any]]) -> ConditionsByModel:
    """Group each root's conditions by golden label and slot, refusing an arm that is not one of the slots' styles."""
    grouped: ConditionsByModel = {}
    for payload in payloads:
        for condition_id, row in payload["conditions"].items():
            public_id = public_condition_id(condition_id)
            model_name, style = public_id.split("/", 1)
            if style not in STYLE_SLOTS:
                raise CampaignTableError(f"{public_id}: unknown prompt style {style!r}: a table's styles are its "
                                         f"slots' ({', '.join(STYLE_SLOTS)})")
            grouped.setdefault(campaign_label(model_name), {})[STYLE_SLOTS[style]] = dict(
                row, condition_id=condition_id, public_id=public_id, prompt_style=style)
    return grouped


def same_call_by_model(payloads: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Return ``{label: {condition_id, same-call rate, same-call comparison}}`` from each forced arm, its two statistics
    under their recorded keys."""
    out: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        for condition_id, row in (payload.get("forced_conditions") or {}).items():
            model_name, _ = condition_id.split("/", 1)
            out[campaign_label(model_name)] = {
                "condition_id": public_condition_id(condition_id),
                SAME_CALL_RATE_KEY: row[SAME_CALL_RATE_KEY],
                SEPARATE_VS_SAME_CALL_KEY: row[SEPARATE_VS_SAME_CALL_KEY],
            }
    return out


def _contrast_lookup(payloads: Iterable[Mapping[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    lookup: Dict[tuple, Dict[str, Any]] = {}
    for payload in payloads:
        for family in (payload.get("style_families") or {}).values():
            for contrast in family["contrasts"]:
                lookup[(contrast["a"], contrast["b"])] = contrast
    return lookup


def build_table(
    payloads: Sequence[Mapping[str, Any]],
    grouped: ConditionsByModel,
    *,
    alpha: float = ALPHA,
    allow_incomplete: bool = False,
) -> List[Dict[str, Any]]:
    """Return one row per (model, contrast), paper order, both families over all rows; a missing contrast is refused
    unless this is a scratch build, which logs it."""
    lookup = _contrast_lookup(payloads)
    rows: List[Dict[str, Any]] = []
    for label in display_order(grouped):
        styles = grouped[label]
        model_name = _model_name(styles)
        for baseline, treatment in TABLE_CONTRASTS:
            contrast = None
            if baseline in styles and treatment in styles:
                contrast = lookup.get((styles[baseline]["condition_id"], styles[treatment]["condition_id"]))
            if contrast is None:
                message = f"{label}: no {baseline} vs {treatment} contrast in the estimates"
                if not allow_incomplete:
                    raise CampaignTableError(
                        f"{message}; every model needs all of {TABLE_CONTRASTS} paired within one root"
                    )
                logger.warning("%s; row omitted (scratch build)", message)
                continue
            rows.append(table_row(label, model_name, baseline, treatment, contrast,
                                   test_method=contrast["test_method"]))
    return add_table_multiplicity(rows, alpha=alpha)


def build_bars(
    grouped: ConditionsByModel,
    same_call: Mapping[str, Mapping[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> BarBlock:
    """Return the figure block: per label, one population, the drawn bars; a model missing a bar is refused unless this
    is a scratch build."""
    bar_block: BarBlock = {}
    for label, styles in grouped.items():
        bars = {style: rate_bar(summary) for style, summary in styles.items()}
        if label in same_call:
            bars[FORCED_B] = rate_bar(same_call[label][SAME_CALL_RATE_KEY])
        missing = [bar for bar in DRAWN_BARS if bar not in bars]
        if missing:
            if not allow_incomplete:
                raise CampaignTableError(f"{label}: no {missing} bar(s); every model needs all of {DRAWN_BARS}")
            logger.warning("%s: missing %s; left off the figure (scratch build)", label, missing)
            continue
        bar_block[label] = {POPULATION: bars}
    return bar_block


def table_rates(grouped: ConditionsByModel) -> Dict[str, Any]:
    return {
        label: {
            "rates": {style: float(summary["estimate"]) for style, summary in sorted(styles.items())},
            "n_common": int(next(iter(styles.values()))["n_scenarios"]),
        }
        for label, styles in grouped.items()
    }


def diagnostics(
    payloads: Sequence[Mapping[str, Any]],
    grouped: ConditionsByModel,
    same_call: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "conditions": {
            summary["public_id"]: {
                "label": label,
                "prompt_style": summary["prompt_style"],
                "variance_components": summary["variance_components"],
                "attempt_statistics": summary["attempt_statistics"],
                "condition_fingerprint": summary["condition_fingerprint"],
                "grid_digest": summary["grid_digest"],
            }
            for label, styles in grouped.items()
            for summary in styles.values()
        },
        "separate_vs_same_call": {
            label: {key: entry[SEPARATE_VS_SAME_CALL_KEY][key] for key in SAME_VS_SEPARATE_KEYS}
            for label, entry in same_call.items()
        },
        "excluded_conditions": {
            public_condition_id(cid): entry
            for payload in payloads
            for cid, entry in (payload.get("excluded_conditions") or {}).items()
        },
    }


def _interval_basis(grouped: ConditionsByModel) -> str:
    bases = {summary["interval_basis"] for styles in grouped.values() for summary in styles.values()}
    if len(bases) != 1:
        raise CampaignTableError(f"the conditions carry {len(bases)} interval bases {sorted(bases)}; expected one")
    return bases.pop()


def provenance(
    payloads: Sequence[Mapping[str, Any]],
    grouped: ConditionsByModel,
    *,
    alpha: float,
    complete: bool,
) -> Dict[str, Any]:
    first = payloads[0]
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "source": TABLE_SOURCE,
        "complete": complete,
        "campaigns": [
            {**estimates_source(payload), "conditions": sorted(map(public_condition_id, payload["conditions"]))}
            for payload in payloads
        ],
        "models": display_order(grouped),
        "model_labels": {_model_name(styles): label for label, styles in grouped.items()},
        **{field: first.get(field) for field in SHARED_DESIGN_FIELDS},
        "alpha": alpha,
        "metric": METRIC,
        "pairing": "scenario_id over one frozen manifest shared by every root",
        "multiplicity": (
            f"{BH_METHOD} in p_bh and {HOLM_METHOD} in p_holm, each recomputed over "
            "every contrast in this table across all roots; the per-root p_holm in the "
            f"estimates files is a per-campaign family and is not reused; {HOLM_MC_MARGIN_NOTE}"
        ),
        "bar_settings": {
            "population": POPULATION,
            "interval_basis": _interval_basis(grouped),
            "inference_method": first.get("inference_method"),
            "bootstrap": first.get("bootstrap"),
            "n_scenarios": first.get("n_scenarios"),
            "n_decoder_draws": first.get("n_decoder_draws"),
            "bars": list(DRAWN_BARS),
        },
    }


def build_payload(
    paths: Sequence[Path],
    *,
    alpha: float = ALPHA,
    allow_incomplete: bool = False,
    allow_stack_mismatch: bool = False,
) -> Dict[str, Any]:
    payloads = load_estimates(paths, allow_incomplete=allow_incomplete, allow_stack_mismatch=allow_stack_mismatch)
    grouped = conditions_by_model(payloads)
    same_call = same_call_by_model(payloads)
    complete = all(is_complete(payload) for payload in payloads)
    return {
        "schema_version": TABLE_SCHEMA_VERSION,
        "table": build_table(payloads, grouped, alpha=alpha, allow_incomplete=allow_incomplete),
        "rates": table_rates(grouped),
        "bars": build_bars(grouped, same_call, allow_incomplete=allow_incomplete),
        "diagnostics": diagnostics(payloads, grouped, same_call),
        "provenance": provenance(payloads, grouped, alpha=alpha, complete=complete),
    }


def write_csv(table: Sequence[Mapping[str, Any]], path: Path, *, alpha: float = ALPHA) -> None:
    """Write the table as CSV."""
    table = validate_table(table, alpha=alpha)
    validate_cross_stack(table)
    # An arm-map table adds each row's two prompt styles and its cross-stack flag after the standard columns.
    arms = any("cross_stack" in row for row in table)
    columns = [
        "model", "prompt_a", "prompt_b", "n_scenarios", "n_draws",
        "rate_a", "rate_b", "difference", "diff_ci_low", "diff_ci_high",
        "randomization_p", "p_holm", "significant_holm", "p_bh", "significant_bh",
        "adjustment_method", "adjustment_family", "adjustment_family_n",
        "metric", "inference_method",
        *(ARM_CSV_COLUMNS if arms else []),
    ]
    rows = [
        dict(zip(columns, (
            row["label"], row["baseline"], row["treatment"],
            row["n_scenarios"], row["n_decoder_draws"],
            f"{row['baseline_rate']:.6f}", f"{row['treatment_rate']:.6f}",
            f"{row['difference']:.6f}", f"{row['ci_low']:.6f}", f"{row['ci_high']:.6f}",
            f"{row['p_value']:.6e}", f"{row['p_holm']:.6e}", row["significant_holm"],
            f"{row['p_bh']:.6e}", row["significant_bh"],
            HOLM_METHOD, TABLE_ADJUSTMENT_FAMILY, row["adjustment_family_n"],
            METRIC, row["inference_method"],
            *((row["baseline"], row["treatment"], row.get("cross_stack") is not None) if arms else ()),
        )))
        for row in table
    ]
    goldens.write_csv(rows, columns, path)


#: The arm-map table's trailing CSV columns: each arm's slot as the paper prints it, and whether the row crosses stacks.
ARM_CSV_COLUMNS = ["style_a", "style_b", "cross_stack"]
#: The LaTeX mark of a row whose two arms ran on different inference stacks.
CROSS_STACK_MARK = r"$^{\dagger}$"


def _cross_stack_note(table: Sequence[Mapping[str, Any]]) -> str:
    pairs = sorted({(row["cross_stack"]["baseline_stack"], row["cross_stack"]["treatment_stack"])
                    for row in table if row.get("cross_stack") is not None})
    if not pairs:
        return ""
    crossed = "; ".join(f"{_latex_escape(a)} against {_latex_escape(b)}" for a, b in pairs)
    return (
        rf" {CROSS_STACK_MARK}The two arms were judged on different inference stacks ({crossed}), an exception "
        r"the table's arm map declares for the naive arm."
    )


def _latex_escape(text: str) -> str:
    return (
        str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
        .replace("_", r"\_").replace("#", r"\#")
    )


#: Three decimals cannot show anything below 0.001, so smaller values print as a bound.
_LATEX_P_FLOOR = 1e-3


def _format_latex_p(p: float, *, significant: bool = False) -> str:
    text = r"\textless 0.001" if p < _LATEX_P_FLOOR else f"{p:.3f}"
    # Text-mode bold: \textbf over math would leave the digits upright.
    return rf"\textbf{{{text}}}" if significant else text


def write_latex(
    table: Sequence[Mapping[str, Any]], path: Path, *, alpha: float = ALPHA, ci_level: float = 0.95
) -> None:
    """Write a compact paper table: the difference with its interval and both families, at the golden's ``ci_level``."""
    table = validate_table(table, alpha=alpha)
    validate_cross_stack(table)
    level = f"{100 * ci_level:g}"
    n_scenarios = table[0]["n_scenarios"]
    n_draws = table[0]["n_decoder_draws"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Paired prompt-style comparisons under the {METRIC} inconsistency rate.}}",
        r"\label{tab:prompt-style-comparisons}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Comparison & Inconsistency A & Inconsistency B & "
        rf"Difference [{level}\% CI] & Randomization $p$ & Holm-adjusted $p$ & BH-adjusted $p$ \\",
        r"\midrule",
    ]
    for row in table:
        model = _latex_escape(MODEL_DISPLAY_LABELS.get(row["label"], row["label"]))
        mark = CROSS_STACK_MARK if row.get("cross_stack") is not None else ""
        lines.append(
            f"{model} & {_latex_escape(row['baseline_label'])} $\\rightarrow$ "
            f"{_latex_escape(row['treatment_label'])}{mark} & {row['baseline_rate']:.3f} & "
            f"{row['treatment_rate']:.3f} & {row['difference']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] & {_format_latex_p(row['p_value'])} & "
            f"{_format_latex_p(row['p_holm'], significant=bool(row['significant_holm']))} & "
            f"{_format_latex_p(row['p_bh'], significant=bool(row['significant_bh']))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}",
            rf"\footnotesize\emph{{Note.}} Every cell rests on the same $N={n_scenarios}$ frozen "
            rf"scenarios judged with $D={n_draws}$ accepted temperature-1 draws each. Differences "
            rf"are baseline minus treatment (positive favours the treatment prompt) with {level}\% "
            r"scenario-cluster bootstrap intervals; $p$ from two-sided scenario-clustered paired "
            rf"randomization tests. Holm and Benjamini--Hochberg adjustments are over all "
            rf"{len(table)} contrasts in this table ($\alpha={alpha:g}$); bold values are significant "
            r"after that column's adjustment, and the published CSV and the site print the Holm "
            rf"column. Metric: {METRIC} inconsistency rate." + _cross_stack_note(table),
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    write_text_atomically(path, "\n".join(lines))


def _summary(payload: Mapping[str, Any]) -> str:
    lines = []
    for row in payload["table"]:
        star = "*" if row["significant_holm"] else " "
        cross = " (cross-stack)" if row.get("cross_stack") is not None else ""
        lines.append(
            f"  {row['label']:<13} {row['baseline_label']:<10} vs {row['treatment_label']:<10} "
            f"{row['baseline_rate']:.3f} -> {row['treatment_rate']:.3f}  diff {row['difference']:+.3f} "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  p={row['p_value']:.1e}  "
            f"Holm={row['p_holm']:.4f}{star}{cross}"
        )
    excluded = payload["diagnostics"]["excluded_conditions"]
    lines.append(
        f"  {len(payload['table'])} rows over {len(payload['bars'])} model(s); "
        f"alpha={payload['provenance']['alpha']:g}; complete={payload['provenance']['complete']}"
        + (f"; excluded: {sorted(excluded)}" if excluded else "")
    )
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    repo = repository_root()
    parser = argparse.ArgumentParser(
        description="Build the prompt-ablation golden, table and figures from estimates or an arm map, or redraw them.")
    parser.add_argument("--estimates", type=Path, action="append", default=[],
                        help="a campaign root's aggregate/estimates.json (repeat per root)")
    parser.add_argument("--arm-map", type=Path, default=None,
                        help="build from an arm map (each judge's three arms by root and condition) "
                        "instead of --estimates")
    add_arm_map_var_argument(parser, "the paper's arm map configs/paper/prompt_ablation_arms.yaml takes "
                             "pop=pop-<first 8 hex of the population's scenario manifest>")
    driver.add_from_golden(parser)
    driver.add_golden(parser)
    driver.add_csv(parser, "judge and contrast")
    parser.add_argument("--latex", type=driver.output_path, default=None, help="where to write the LaTeX table")
    driver.add_figure(parser, "--figure_styles", "the prompt-styles panel (.png or .pdf)", repeatable=False)
    driver.add_figure(parser, "--figure_methods", "the scoring-methods panel (.png or .pdf)", repeatable=False)
    driver.add_dpi(parser)
    parser.add_argument("--alpha", type=float, default=None,
                        help=f"significance level for a fresh build (default {ALPHA}); a re-render "
                        "always uses the golden's own")
    driver.add_allow_incomplete(parser, "build from an incomplete campaign (scratch previews only; the golden records "
                                "complete=false)")
    driver.add_allow_stack_mismatch(parser)
    args = driver.parse(parser, argv)

    if args.arm_map is not None and (args.estimates or args.from_golden is not None):
        parser.error("--arm-map builds from the roots the arm map names; do not also pass --estimates or --from_golden")
    if arm_map_var_refusal(args.arm_map, args.arm_map_var):
        parser.error(arm_map_var_refusal(args.arm_map, args.arm_map_var))
    if args.arm_map is None:
        driver.one_source(parser, args)
    driver.require_drawing([args.figure_styles, args.figure_methods])
    alpha = ALPHA if args.alpha is None else args.alpha  # a fresh build's; a re-render keeps its golden's

    def read(path: Path) -> Dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != TABLE_SCHEMA_VERSION:
            parser.error(f"{path} has schema_version {payload.get('schema_version')!r}, not {TABLE_SCHEMA_VERSION}; "
                         "this driver renders only the campaign golden")
        if not payload.get("provenance", {}).get("complete") and not args.allow_incomplete:
            parser.error(f"{path} was built from an incomplete campaign (provenance.complete is false); a scratch "
                         "golden is not re-rendered into the paper's figures without --allow-incomplete")
        refusal = recorded_stack_mismatch_refusal(payload.get("provenance") or {}, str(path))
        if refusal is not None and not args.allow_stack_mismatch:
            parser.error(refusal)
        return payload

    def build() -> Dict[str, Any]:
        if args.arm_map is not None:
            return build_arm_payload(args.arm_map, alpha=alpha, allow_incomplete=args.allow_incomplete,
                                     allow_stack_mismatch=args.allow_stack_mismatch,
                                     variables=parse_variables(args.arm_map_var))
        return build_payload(args.estimates, alpha=alpha, allow_incomplete=args.allow_incomplete,
                             allow_stack_mismatch=args.allow_stack_mismatch)

    payload = driver.golden_or_build(parser, args, read, build, REFUSALS)
    if args.from_golden is not None:
        try:
            alpha = float(payload["provenance"]["alpha"])
        except (KeyError, TypeError, ValueError):
            parser.error(f"{args.from_golden} records no provenance.alpha; not a golden this driver wrote")
        if args.alpha is not None and args.alpha != alpha:
            parser.error(f"--alpha {args.alpha} differs from the golden's {alpha}; rebuild from --estimates instead")
        payload["table"] = validate_table(payload["table"], alpha=alpha)

    try:
        validate_cross_stack(payload["table"], payload["provenance"])
    except CampaignEstimatesError as error:
        parser.error(str(error))
    # Into the paper's files only an arm map's table goes, and of those only the table the templates name.
    outputs = [args.golden, args.csv, args.latex, args.figure_styles, args.figure_methods]
    refusal = paper_output_refusal(outputs, "arm_map" in payload["provenance"], repo_root=repo)
    if refusal is None and paper_paths(outputs, repo_root=repo):
        refusal = paper_table_refusal(payload)
    if refusal is not None:
        parser.error(refusal)
    # The golden first, so a figure that fails to write sits beside this run's golden, not an older one.
    driver.write_golden(args.golden, payload)
    if args.csv is not None:
        write_csv(payload["table"], args.csv, alpha=alpha)
    if args.latex is not None:
        write_latex(payload["table"], args.latex, alpha=alpha,
                    ci_level=float(payload["provenance"]["bootstrap"]["ci_level"]))
    for part, out in (("styles", args.figure_styles), ("methods", args.figure_methods)):
        driver.draw_each([out], lambda out, part=part: draw(payload["bars"], out, POPULATION, part, dpi=args.dpi))
    print(_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
