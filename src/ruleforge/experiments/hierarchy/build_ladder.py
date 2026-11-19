from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from ruleforge.experiments import driver, goldens
from ruleforge.experiments.hierarchy.analysis import FLAT, HIERARCHY, HierarchyError
from ruleforge.experiments.hierarchy.build import REFUSALS, add_rules_levels_arguments, load_pairs, refuse_missing_files
from ruleforge.experiments.hierarchy.ladder import CSV_COLUMNS, SCHEMA_VERSION, build_payload, csv_rows, summary_markdown


def load_golden(path: Path | str) -> Dict[str, Any]:
    """Return the ladder golden at ``path``, refusing one of another schema."""
    return goldens.read_analysis_golden(path, "hierarchy_ladder", SCHEMA_VERSION, error=HierarchyError)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the inconsistency rate by hierarchy level (its golden, CSV, summary and figure) from the "
        "hierarchy pair map, or redraw them from a golden.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path, default=None,
                        help="the hierarchy pair map: each judge's flat and hierarchy arms")
    driver.add_from_golden(source)
    add_rules_levels_arguments(parser)
    driver.add_golden(parser)
    driver.add_csv(parser, "arm and rung")
    driver.add_summary(parser)
    driver.add_figure(parser)
    parser.add_argument("--arm", choices=(FLAT, HIERARCHY), default=None,
                        help="draw only the arms of this name; the golden, CSV and summary keep every arm")
    driver.add_dpi(parser)
    driver.add_allow_incomplete(parser)
    args = driver.parse(parser, argv)
    driver.refuse_beside_golden(parser, args, ("allow_incomplete", "golden"))
    driver.require_drawing(args.figure)

    def build() -> Dict[str, Any]:
        refuse_missing_files(parser, args)
        return build_payload(pairs=load_pairs(args.pairs), rules_path=args.rules, levels_path=args.levels,
                             allow_incomplete=args.allow_incomplete)

    payload = driver.as_stored(driver.golden_or_build(parser, args, load_golden, build, REFUSALS))
    from ruleforge.experiments.hierarchy.ladder_figure import draw

    drawn = payload if args.arm is None else {**payload, "arms": [arm for arm in payload["arms"]
                                                                  if arm["arm"] == args.arm]}
    text = summary_markdown(payload)
    # The golden first, so a figure that fails to write sits beside this run's golden, not an older one.
    driver.write_golden(args.golden, payload)
    driver.write_csv(args.csv, lambda: csv_rows(payload), CSV_COLUMNS)
    driver.write_text(args.summary, text)
    driver.draw_each(args.figure, lambda out: draw(drawn, out, dpi=args.dpi))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
