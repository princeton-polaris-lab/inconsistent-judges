from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.record_names import SHARED_BANK_ESTIMATES_FNAME
from ruleforge.experiments import driver, goldens
from ruleforge.experiments.ruleset_size.repeated_draw_baseline import (
    ANALYSIS,
    CSV_COLUMNS,
    SCHEMA_VERSION,
    RepeatedDrawBaselineError,
    build_payload,
    csv_rows,
    summary,
)

#: Refusals the build turns into usage errors; any other error keeps its traceback.
REFUSALS = (OSError, RuleForgeError)


def load_golden(path: Path | str) -> Dict[str, Any]:
    """Return the repeated-draw baseline's golden at ``path``, refusing one of another schema."""
    return goldens.read_analysis_golden(path, ANALYSIS, SCHEMA_VERSION, error=RepeatedDrawBaselineError)


def _usable_cpus() -> int:
    # The CPUs this process may run on where the platform says so (Linux), else every CPU.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the repeated-draw baseline of the ruleset-size curves (its golden, CSV, summary and figure) "
        "from the campaign estimates and their roots, or redraw them from a golden.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--estimates", type=Path, action="append", default=None,
                        help=f"a campaign root's aggregate/{SHARED_BANK_ESTIMATES_FNAME} (repeat per root)")
    driver.add_from_golden(source)
    driver.add_golden(parser)
    driver.add_csv(parser, "judge and size")
    driver.add_figure(parser, help="the figure (.png or .pdf)", repeatable=False)
    driver.add_dpi(parser)
    args = driver.parse(parser, argv)
    driver.refuse_beside_golden(parser, args, ("golden",))
    driver.require_drawing([args.figure])

    payload = driver.as_stored(driver.golden_or_build(
        parser, args, load_golden, lambda: build_payload(args.estimates, workers=_usable_cpus()), REFUSALS))
    from ruleforge.experiments.ruleset_size.repeated_draw_baseline_figure import draw

    # The golden first, so a figure that fails to write sits beside this run's golden, not an older one.
    driver.write_golden(args.golden, payload)
    driver.write_csv(args.csv, lambda: csv_rows(payload), CSV_COLUMNS)
    driver.draw_each([args.figure], lambda out: draw(payload, out, dpi=args.dpi))
    print(summary(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
