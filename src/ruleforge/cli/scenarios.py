from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ruleforge.cli import records
from ruleforge.cli.common import subcommand


def register(sub: argparse._SubParsersAction) -> None:
    """Add ``scenarios concat``, ``subtract`` and ``partition``."""
    parser = sub.add_parser("scenarios", help="frozen scenario populations: concatenate, subtract, partition",
                            allow_abbrev=False)
    verbs = parser.add_subparsers(dest="verb", required=True)
    concat = subcommand(verbs, "concat", run_concat, "concatenate frozen populations into one, with a sidecar")
    concat.add_argument("--source", required=True, type=Path, action="append",
                        help="a population, in the output's order (repeatable; at least two)")
    concat.add_argument("--expect-source-sha256", action="append", default=[],
                        help="each --source's approved SHA-256, in the same order (one per source, or none)")
    concat.add_argument("--out", required=True, type=Path)
    subtract = subcommand(verbs, "subtract", run_subtract,
                          "write a frozen population's rows that populations carved from it do not hold, with a sidecar")
    subtract.add_argument("--source", required=True, type=Path)
    subtract.add_argument("--expect-source-sha256", default=None,
                          help="refuse a source whose SHA-256 differs from this approved digest")
    subtract.add_argument("--exclude", required=True, type=Path, action="append",
                          help="a population carved from the source, whose rows are left out (repeatable)")
    subtract.add_argument("--out", required=True, type=Path)
    partition = subcommand(verbs, "partition", run_partition,
                           "cut a frozen population into contiguous blocks, each with a sidecar")
    partition.add_argument("--source", required=True, type=Path)
    partition.add_argument("--block-size", required=True, type=int)
    partition.add_argument("--out-dir", required=True, type=Path,
                           help="the directory the blocks and sidecars go to")
    partition.add_argument("--expect-source-sha256", default=None,
                           help="refuse a source whose SHA-256 differs from this approved digest")


WAY_OUT = "give the output a place of its own, apart from its sidecar and its run records"


def run_concat(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run :func:`ruleforge.scenarios.subset.build_concatenation` and print where each source landed."""
    from ruleforge.scenarios.subset import build_concatenation, sidecar_path

    out = records.absolute_path(args.out)
    records.check_outputs([out], beside=[sidecar_path(out)], way_out=WAY_OUT)
    record = records.open_record(out, "scenarios concat", argv, None)
    with records.logging_to(record, verbose=args.verbose):
        payload = build_concatenation(args.source, out, expected_source_sha256s=args.expect_source_sha256 or None)
    for entry in payload["sources"]:
        start, stop = entry["row_range"]
        print(f"{entry['name']} (sha256 {entry['sha256']}): rows [{start}, {stop})")
    print(f"{out}: {payload['population_count']} rows, sha256 {payload['output_sha256']}")
    return 0


def run_subtract(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run :func:`ruleforge.scenarios.subset.build_difference` and print what each excluded population removed."""
    from ruleforge.scenarios.subset import build_difference, sidecar_path

    out = records.absolute_path(args.out)
    records.check_outputs([out], beside=[sidecar_path(out)], way_out=WAY_OUT)
    record = records.open_record(out, "scenarios subtract", argv, None)
    with records.logging_to(record, verbose=args.verbose):
        payload = build_difference(args.source, out, exclude=args.exclude,
                                   expected_source_sha256=args.expect_source_sha256)
    print(f"source: {args.source} ({payload['population_count']} rows, sha256 {payload['source_sha256']})")
    for entry in payload["excluded"]:
        print(f"less {entry['name']} (sha256 {entry['sha256']}): {entry['removed']} of its {entry['count']} rows")
    print(f"{out}: {payload['count']} rows, sha256 {payload['output_sha256']}")
    return 0


def run_partition(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Run :func:`ruleforge.scenarios.subset.build_blocks` and print each block's rows and digest."""
    from ruleforge.scenarios.subset import build_blocks

    out_dir = records.absolute_path(args.out_dir)
    records.check_outputs([out_dir], way_out=WAY_OUT)
    record = records.open_record(out_dir, "scenarios partition", argv, None)
    with records.logging_to(record, verbose=args.verbose):
        payloads = build_blocks(args.source, out_dir, block_size=args.block_size,
                                expected_source_sha256=args.expect_source_sha256)
    print(f"source: {args.source} ({payloads[0]['population_count']} rows, sha256 {payloads[0]['source_sha256']})")
    for payload in payloads:
        start, stop = payload["row_range"]
        print(f"{out_dir / payload['output_name']}: rows [{start}, {stop}) sha256 {payload['output_sha256']}")
    return 0
