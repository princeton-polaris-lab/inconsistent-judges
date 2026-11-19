from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ruleforge.core.artifacts import ArtifactSchemaError
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.paths import repository_root
from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.evaluate.inference import RepeatedSamplingError
from ruleforge.experiments import driver, goldens
from ruleforge.experiments.hierarchy.analysis import (
    CSV_COLUMNS,
    FLAT,
    HIERARCHY,
    SCHEMA_VERSION,
    HierarchyError,
    JudgePair,
    build_payload,
    csv_rows,
    summary_markdown,
)

DEFAULT_RULES = Path("data") / "rules" / "modelspec" / "openai_rules_all.jsonl"
DEFAULT_LEVELS = Path("data") / "rules" / "modelspec" / "openai_rules_all_full.jsonl"
PAIRS_VERSION = 1
PAIRS_KEYS = ("version", "judges")
ARM_KEYS = ("root", "condition")
REFUSALS = (CampaignEstimatesError, RepeatedSamplingError, SamplingDesignError, ArtifactSchemaError)


def load_pairs(path: Path | str) -> List[JudgePair]:
    """Return the judges' pairs a pair map names, in its order (``{flat: {root, condition}, hierarchy: {...}}`` per
    judge); anything else is refused."""
    path = Path(path)
    where = f"pair map {path.name}"
    try:
        top = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HierarchyError(f"{where}: unreadable: {error}") from error
    if not (isinstance(top, dict) and sorted(top) == sorted(PAIRS_KEYS) and top["version"] == PAIRS_VERSION
            and isinstance(top["judges"], list) and top["judges"]):
        raise HierarchyError(f"{where}: must hold exactly {list(PAIRS_KEYS)}, version {PAIRS_VERSION} and a "
                                  "non-empty list of judges")
    pairs = []
    for index, judge in enumerate(top["judges"]):
        arms = []
        for side in (FLAT, HIERARCHY):
            arm = judge.get(side) if isinstance(judge, dict) and sorted(judge) == sorted((FLAT, HIERARCHY)) else None
            if not (isinstance(arm, dict) and sorted(arm) == sorted(ARM_KEYS)
                    and all(isinstance(arm[key], str) and arm[key] for key in ARM_KEYS)):
                raise HierarchyError(f"{where}: judge {index} must name its {FLAT} and {HIERARCHY} arms, each "
                                          f"exactly {list(ARM_KEYS)} as non-empty strings")
            arms.append((arm["root"], arm["condition"]))
        pairs.append(JudgePair(flat=arms[0], hierarchy=arms[1]))
    return pairs


def load_golden(path: Path | str) -> Dict[str, Any]:
    """Return the golden at ``path``, refusing one of another schema."""
    return goldens.read_analysis_golden(path, "hierarchy", SCHEMA_VERSION, error=HierarchyError)


def add_rules_levels_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--rules`` and ``--levels``: the ruleset the arms were judged against and each rule's level."""
    repo = repository_root()
    parser.add_argument("--rules", type=Path, default=repo / DEFAULT_RULES,
                        help="the ruleset the arms were judged against (default: %(default)s)")
    parser.add_argument("--levels", type=Path, default=repo / DEFAULT_LEVELS,
                        help="each rule's level (rule_type), row by row: the file the hierarchy arms read "
                        "(default: %(default)s)")


def refuse_missing_files(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit with a usage error naming ``--rules`` or ``--levels`` when its file does not exist."""
    missing = [f"--{name} {path}" for name, path in (("rules", args.rules), ("levels", args.levels))
               if not path.is_file()]
    if missing:
        parser.error(f"no such file: {', '.join(missing)}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the hierarchy golden, its CSV, summary and figure, or redraw them from a golden.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path, default=None,
                        help="a pair map: each judge's flat and hierarchy arms, each a root and a public condition id "
                        "(the judges are one Holm family)")
    source.add_argument("--judge", nargs=4, action="append", default=None,
                        metavar=("FLAT_ROOT", "FLAT_CONDITION", "HIERARCHY_ROOT", "HIERARCHY_CONDITION"),
                        help="one judge's flat and hierarchy arms, each an artifact:// root and a public condition id "
                        "(repeatable: the judges are one Holm family)")
    driver.add_from_golden(source, "re-render from an existing golden, reading no root")
    add_rules_levels_arguments(parser)
    driver.add_golden(parser)
    driver.add_csv(parser, "judge and quantity")
    driver.add_summary(parser, "where to write the summary (markdown, with the figure's caption notes)")
    driver.add_figure(parser)
    driver.add_dpi(parser, 200)
    driver.add_allow_incomplete(parser)
    args = driver.parse(parser, argv)
    driver.refuse_beside_golden(parser, args, ("allow_incomplete", "golden"))
    driver.require_drawing(args.figure)

    def build() -> Dict[str, Any]:
        refuse_missing_files(parser, args)
        pairs = load_pairs(args.pairs) if args.pairs is not None else [
            JudgePair(flat=(flat_root, flat_cid), hierarchy=(hier_root, hier_cid))
            for flat_root, flat_cid, hier_root, hier_cid in args.judge]
        return build_payload(pairs=pairs, rules_path=args.rules, levels_path=args.levels,
                             allow_incomplete=args.allow_incomplete)

    payload = driver.as_stored(driver.golden_or_build(parser, args, load_golden, build, REFUSALS))
    from ruleforge.experiments.hierarchy.figure import caption_notes, draw

    text = summary_markdown(payload) + "\n" + caption_notes(payload)
    # The golden first, so a figure that fails to write sits beside this run's golden, not an older one.
    driver.write_golden(args.golden, payload)
    driver.write_csv(args.csv, lambda: csv_rows(payload), CSV_COLUMNS)
    driver.write_text(args.summary, text)
    driver.draw_each(args.figure, lambda out: draw(payload, out, dpi=args.dpi))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
