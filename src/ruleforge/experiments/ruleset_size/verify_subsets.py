from __future__ import annotations

import argparse
from pathlib import Path

from ruleforge.rulesets.subsets import (
    build_manifest,
    diff_manifest,
    load_manifest,
    strategy_root,
)
from ruleforge.core.paths import repository_root

DEFAULT_MANIFEST_DIR = Path("data") / "rules" / "subset_manifest"
DEFAULT_OUTPUT = Path("data") / "rules" / "ruleset_size_subsets"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check regenerated ruleset-size subsets against the committed checksum manifest.")
    parser.add_argument("--manifest-dir", type=Path, default=None, help=f"default: the checkout's {DEFAULT_MANIFEST_DIR}")
    parser.add_argument("--output-root", type=Path, default=None, help=f"default: the checkout's {DEFAULT_OUTPUT}")
    args = parser.parse_args(argv)
    repo = repository_root()

    meta, expected = load_manifest(args.manifest_dir if args.manifest_dir is not None else repo / DEFAULT_MANIFEST_DIR)
    root = strategy_root(args.output_root if args.output_root is not None else repo / DEFAULT_OUTPUT,
                         meta["sampling_strategy"])
    if not root.exists():
        print(f"No subsets at {root}; run `ruleforge reproduce ruleset-size subsets` first.")
        return 2

    differences = diff_manifest(expected, build_manifest(root))
    if not any(differences.values()):
        print(f"All {len(expected)} subset files match the manifest.")
        return 0

    for kind, paths in differences.items():
        if paths:
            print(f"{kind}: {len(paths)}")
            for path in paths[:10]:
                print(f"  {path}")
            if len(paths) > 10:
                print(f"  ... and {len(paths) - 10} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
