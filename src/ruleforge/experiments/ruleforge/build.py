from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruleforge.experiments.driver import output_path
from ruleforge.experiments.ruleforge.comparison import build_from_campaign
from ruleforge.core.artifacts import write_text_atomically

DEFAULT_RULESETS = {
    "initial": "Initial",
    "seed5_winner": "Seed 5 (winner)",
    "seed9_runner_up": "Seed 9 (runner-up)",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the RuleForge golden from one complete repeated-decoding ruleset campaign.")
    parser.add_argument("--estimates", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=output_path)
    parser.add_argument("--ruleset", action="append", metavar="TAG=LABEL")
    parser.add_argument("--figure-a", type=output_path)
    parser.add_argument("--figure-b", type=output_path)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--allow-stack-mismatch", action="store_true",
                        help="build from an aggregate that reports inference-stack mismatches")
    args = parser.parse_args(argv)
    labels = {}
    for item in args.ruleset or []:
        tag, sep, label = item.partition("=")
        if not sep or not tag or not label or tag in labels:
            parser.error("--ruleset requires unique TAG=LABEL entries")
        labels[tag] = label
    stats = build_from_campaign(
        args.estimates, labels or DEFAULT_RULESETS, allow_stack_mismatch=args.allow_stack_mismatch
    )
    write_text_atomically(
        args.golden, json.dumps(stats, indent=2, allow_nan=False) + "\n"
    )
    if args.figure_a or args.figure_b:
        from ruleforge.experiments.ruleforge.plot import draw

        for panel, path in (("a", args.figure_a), ("b", args.figure_b)):
            if path:
                draw(stats, path, dpi=args.dpi, panel=panel, allow_stack_mismatch=args.allow_stack_mismatch)


if __name__ == "__main__":
    main()
