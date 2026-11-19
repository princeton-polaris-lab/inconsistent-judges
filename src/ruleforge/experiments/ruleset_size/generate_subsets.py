from __future__ import annotations

import argparse
import sys
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ruleforge.rulesets.subsets import (
    file_digest,
    generate_subset_files,
    load_manifest,
    seed_dir,
    strategy_root,
)
from ruleforge.core.paths import one_place, repository_root

DEFAULT_MANIFEST_DIR = Path("data") / "rules" / "subset_manifest"
DEFAULT_OUTPUT = Path("data") / "rules" / "ruleset_size_subsets"
#: Checkout directories the subsets are never written into.
COMMITTED = ("results",)


def output_place(given: str | Path, repo: Path) -> Path:
    """Return the output root as one resolved place, refusing a committed directory or a protected root."""
    from ruleforge.evaluate.api import refuse_protected

    out = one_place(given)
    for name in COMMITTED:
        if out.resolve().is_relative_to((repo / name).resolve()):
            raise ValueError(f"{given} lies in the checkout's {name}/, which holds the paper's committed files: write "
                             "the subsets into a directory of their own")
    refuse_protected(out, "the subsets' output root", below=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the ruleset-size subsets byte for byte from the manifest's ruleset and seeds.",
        allow_abbrev=False)
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help=f"the directory of the manifest.json to read (default: the checkout's {DEFAULT_MANIFEST_DIR})")
    parser.add_argument("--output-root", type=Path, default=None, help=f"default: the checkout's {DEFAULT_OUTPUT}")
    args = parser.parse_args(argv)
    repo = repository_root()
    manifest_dir = args.manifest_dir if args.manifest_dir is not None else repo / DEFAULT_MANIFEST_DIR
    try:
        output_root = output_place(args.output_root if args.output_root is not None else repo / DEFAULT_OUTPUT, repo)
    except ValueError as error:
        parser.error(f"--output-root {error}")

    meta, _ = load_manifest(manifest_dir)
    strategy = meta["sampling_strategy"]
    ruleset = repo / meta["source_ruleset"]["path"]

    # Check both inputs before writing ~200 MB: a changed ruleset or Python version would otherwise read as a
    # sampler change in the verifier.
    actual = file_digest(ruleset)
    if actual != meta["source_ruleset"]["sha256"]:
        raise SystemExit(
            f"{ruleset} does not match the manifest (sha256 {actual}, expected "
            f"{meta['source_ruleset']['sha256']}); the subsets embed its records "
            "verbatim, so regeneration would not reproduce them."
        )
    required = meta.get("reproducibility", {}).get("python_requires")
    if required:
        try:
            specifier = SpecifierSet(required)
        except InvalidSpecifier as error:
            raise SystemExit(f"manifest's python_requires {required!r} is no version specifier: {error}") from error
        current = ".".join(map(str, sys.version_info[:3]))
        if not specifier.contains(current):
            # random.sample is not stable across Python versions.
            raise SystemExit(
                f"manifest requires Python {required}, running {current}; the "
                "sampler is not guaranteed to reproduce across versions."
            )

    root = strategy_root(output_root, strategy)
    total = 0
    for seed in meta["seeds"]:
        written = generate_subset_files(
            ruleset, seed_dir(output_root, strategy, seed), strategy, seed
        )
        total += len(written)
        print(f"seed {seed}: {len(written)} files")

    print(f"\n{total} files under {root}")
    print("Verify with: uv run ruleforge reproduce ruleset-size subsets --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
