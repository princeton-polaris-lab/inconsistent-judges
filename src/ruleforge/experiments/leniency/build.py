from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Mapping

from ruleforge.evaluate.estimates import (
    CampaignEstimatesError,
    recorded_stack_mismatch_refusal,
)
from ruleforge.experiments import driver
from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS, display_order
from ruleforge.experiments.paper_outputs import paper_output_refusal
from ruleforge.evaluate.inference import RepeatedSamplingError
from ruleforge.core.paths import repository_root
from ruleforge.experiments.leniency.analysis import CSV_COLUMNS, build_payload, csv_rows
from ruleforge.experiments.leniency.schema import LABELS
from ruleforge.experiments.leniency.golden import read_golden, validate_payload
from ruleforge.experiments.leniency.figure import (
    direction_significance_counts,
    draw,
    draw_flips,
    main_text_rule_order,
    style_note,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.prompts.styles import FORCED, FORCED_JSON_STYLES

DEFAULT_RULES = Path("data") / "rules" / "modelspec" / "openai_rules_all.jsonl"

REFUSALS = (CampaignEstimatesError, RepeatedSamplingError, SamplingDesignError)


def load_golden(path: Path, *, allow_incomplete: bool, allow_stack_mismatch: bool = False) -> dict:
    """Return an installed golden, validated; a scratch one only with ``allow_incomplete``, one with stack mismatches
    only with ``allow_stack_mismatch``."""
    payload = read_golden(path, "golden")
    validate_payload(payload)
    if not payload["provenance"]["complete"] and not allow_incomplete:
        raise CampaignEstimatesError(
            f"{path} was built from an incomplete campaign (provenance.complete is false); a "
            "scratch golden is not re-rendered into the paper's figures without --allow-incomplete"
        )
    refusal = recorded_stack_mismatch_refusal(payload["provenance"], str(path))
    if refusal is not None and not allow_stack_mismatch:
        raise CampaignEstimatesError(refusal)
    return payload


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def summary(payload: Mapping[str, Any]) -> str:
    """Return, per model, the headline numbers, the flips agreement and the decompositions."""
    lines = []
    for label in display_order(payload["models"]):
        block = payload["models"][label]
        head, flips = block["summary"], block["flips"]
        references = flips["draw_to_draw"]
        verdict = block["by_verdict"]["strata"]
        named = block["named_preempted"]
        lines.append(
            f"  {MODEL_DISPLAY_LABELS.get(label, label):<13} pooled delta {head['pooled_delta']:+.4f} "
            f"({head['pooled_rate_alone']:.4f} -> {head['pooled_rate_in_context']:.4f}); "
            f"lower {head['n_rules_lower']} / higher {head['n_rules_higher']} / tied {head['n_rules_tied']}; "
            f"Holm {head['n_holm_lower']} down, {head['n_holm_higher']} up"
        )
        lines.append(
            f"  {'':<13} flips unchanged {flips['unchanged_fraction']:.4f}, floor "
            f"{flips['chance_agreement']:.4f}, kappa {flips['kappa']:.4f}; draws agree alone "
            f"{references['alone']['unchanged_fraction']:.4f}, in full "
            f"{references['in_context']['unchanged_fraction']:.4f}"
        )
        # In label order: a re-read golden has sort_keys order.
        contributions = ", ".join(
            f"full={stratum} {verdict[stratum]['delta_contribution']:+.4f}" for stratum in LABELS
        )
        lines.append(
            f"  {'':<13} by judgment {contributions}; "
            + (
                "named: not applicable (the prompt asks for no preempted_rules)"
                if named is None
                else f"named {named['n_named_cells']} cells, {named['delta_contribution']:+.4f} of delta, "
                f"judged 'no' in context on {_percent(named['in_context_no_rate'])}"
            )
        )
        effects = block["effects"]
        margin = [f"R{row['rule']}" for row in effects if row["holm_mc_margin"]]
        margin_significant = sum(row["holm_mc_margin"] and row["significant_holm"] for row in effects)
        few = [f"R{row['rule']}" for row in effects if row["few_informative"]]
        lines.append(
            f"  {'':<13} p at the permutation floor {sum(row['p_value_is_floor'] for row in effects)}; "
            f"Holm within Monte Carlo margin {margin} ({margin_significant} of them significant); "
            f"few informative {few}"
        )
    counts = direction_significance_counts(payload)
    note = style_note(payload)
    if note:
        lines.append(f"  prompt style: {note} ({payload['settings']['prompt_style']})")
    lines.append(f"  compact rules: {['R%d' % n for n in main_text_rule_order(payload)]}")
    lines.append(f"  direction x Holm counts: {counts}")
    lines.append(f"  complete={payload['provenance']['complete']}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    repo = repository_root()
    parser = argparse.ArgumentParser(
        description="Build the leniency golden, CSV and figures from campaign estimates, or redraw them from a golden.")
    parser.add_argument("--estimates", type=Path, action="append", default=[],
                        help="a campaign root's aggregate/estimates.json (repeat per root)")
    parser.add_argument("--rules", type=Path, default=repo / DEFAULT_RULES, help="rules file (default: %(default)s)")
    parser.add_argument("--prompt-style", choices=FORCED_JSON_STYLES, default=None,
                        help=f"the forced style whose conditions the golden holds (default: the paper's, {FORCED}); "
                        "with --from_golden, a check that the golden holds it")
    driver.add_from_golden(parser)
    driver.add_golden(parser)
    driver.add_csv(parser, "model and rule")
    driver.add_figure(parser, help="compact main-text forest plot; repeatable (.png/.pdf)")
    driver.add_figure(parser, "--figure_all", "all-rules appendix forest plot; repeatable (.png/.pdf)")
    driver.add_figure(parser, "--figure_flips", "flips matrices; repeatable (.png/.pdf)")
    driver.add_dpi(parser, 200)
    driver.add_allow_incomplete(parser, "build from (or re-render) an incomplete campaign; scratch previews only")
    driver.add_allow_stack_mismatch(parser)
    args = driver.parse(parser, argv)
    driver.one_source(parser, args)
    driver.require_drawing([*args.figure, *args.figure_all, *args.figure_flips])

    def read(path: Path) -> dict:
        payload = load_golden(path, allow_incomplete=args.allow_incomplete,
                              allow_stack_mismatch=args.allow_stack_mismatch)
        held = payload["settings"]["prompt_style"]
        if args.prompt_style not in (None, held):
            raise CampaignEstimatesError(f"{path} holds {held!r} conditions, not --prompt-style {args.prompt_style!r}")
        return payload

    payload = driver.golden_or_build(
        parser, args, read,
        lambda: build_payload(args.estimates, args.rules, allow_incomplete=args.allow_incomplete,
                              allow_stack_mismatch=args.allow_stack_mismatch,
                              prompt_style=args.prompt_style or FORCED),
        REFUSALS)
    outputs = [args.golden, args.csv, *args.figure, *args.figure_all, *args.figure_flips]
    refusal = paper_output_refusal(outputs, payload["settings"]["prompt_style"] == FORCED, repo_root=repo)
    if refusal is not None:
        parser.error(refusal)
    # The golden first, so a figure that fails to write sits beside this run's golden, not an older one.
    driver.write_golden(args.golden, payload)
    driver.write_csv(args.csv, lambda: csv_rows(payload), CSV_COLUMNS)
    driver.draw_each(args.figure, lambda out: draw(payload, out, dpi=args.dpi, compact=True))
    driver.draw_each(args.figure_all, lambda out: draw(payload, out, dpi=args.dpi, compact=False))
    driver.draw_each(args.figure_flips, lambda out: draw_flips(payload, out, dpi=args.dpi))
    print(summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
