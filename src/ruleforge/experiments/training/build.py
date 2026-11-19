from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.experiments import driver
from ruleforge.experiments.training.campaign_readout import TrainingReadoutError, policy_pairs
from ruleforge.experiments.training.family_readout import SCHEMA_VERSION, SOURCE, FamilyArm, build_readout, summary_markdown
from ruleforge.experiments.training.figure import ARM_STYLES, PANELS, draw
from ruleforge.experiments.training.paper_families import PAPER_OUTPUTS, drawn_names, paper_refusal
from ruleforge.experiments.paper_outputs import PAPER_OUTPUT_DIRS, paper_paths
from ruleforge.evaluate.inference import RepeatedSamplingError
from ruleforge.core.paths import repository_root
from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.paths import resolve
from ruleforge.core.errors import SamplingDesignError

STEM = "training_family"
SUFFIXES = ("pdf", "png")
REFUSALS = (TrainingReadoutError, CampaignEstimatesError, RepeatedSamplingError, SamplingDesignError, ArtifactRootError)
NO_SCRATCH_BUILD = (" (this readout reads complete, final roots only and has no --allow-incomplete: finish or "
                    "re-analyse the root)")
FIGURE_FLAGS = {panel: f"figure_{panel}" for panel in PANELS}
#: The flags a paper family's own arguments supply (``--family`` refuses each beside it).
FAMILY_FLAGS = ("--arm", "--reference", "--rule-set", "--label", "--merge-manifest", "--parity", "--allow-code-mismatch")
RERENDER_FLAGS = ("from_golden", "dpi", *FIGURE_FLAGS.values())


def output_paths(out_dir: Path, *, paper: str | None = None) -> dict[str, Path]:
    """Return every file a build writes, by name: the readout and summary in ``out_dir``, the panels there too or,
    for a ``paper`` family, in the checkout's figure files."""
    def panel_path(panel: str, suffix: str) -> Path:
        if paper is None:
            return out_dir / f"training_{panel}.{suffix}"
        return repository_root() / f"{PAPER_OUTPUTS[paper].figures[panel]}.{suffix}"

    return {
        "json": out_dir / f"{STEM}.json",
        "md": out_dir / f"{STEM}.md",
        **{f"{panel}.{suffix}": panel_path(panel, suffix) for panel in PANELS for suffix in SUFFIXES},
    }


def rerender(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Draw the panels the ``--figure-*`` flags name from an installed golden."""
    driver.refuse_beside_golden(parser, args, [name for name in vars(args) if name not in RERENDER_FLAGS])
    figures = {panel: getattr(args, name) for panel, name in FIGURE_FLAGS.items() if getattr(args, name) is not None}
    if not figures:
        parser.error("--from_golden draws the panel --figure-a names: pass it")
    golden = json.loads(args.from_golden.read_text(encoding="utf-8"))
    if not isinstance(golden, dict):
        parser.error(f"{args.from_golden} is no training family readout (a JSON {type(golden).__name__}, not an object)")
    if golden.get("source") != SOURCE or golden.get("schema_version") != SCHEMA_VERSION:
        parser.error(f"{args.from_golden} is no training family readout (source {golden.get('source')!r}, "
                     f"schema_version {golden.get('schema_version')!r})")
    root = repository_root()
    if strays := [f"--figure-{panel} {path}" for panel, path in figures.items() if paper_paths([path], repo_root=root)]:
        parser.error(f"{', '.join(strays)} would be among the paper's files ({', '.join(PAPER_OUTPUT_DIRS)}): a "
                     "re-render draws its panels elsewhere")
    names = drawn_names(golden)
    for panel, path in figures.items():
        driver.draw_each([path], lambda out, panel=panel: draw(golden, out, panel, dpi=args.dpi, names=names))
    print(summary_markdown(golden), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the training readout and panel for a family of arms, or redraw the panel from a golden.")
    parser.add_argument("--arm", nargs=3, action="append", metavar=("LABEL", "ROOT", "CONDITION"),
                        help="one arm: its policy label, its campaign root (artifact://) and its condition id "
                        "(repeatable, in the family's order)")
    parser.add_argument("--reference", help="the reference arm's label (the base model)")
    parser.add_argument("--rule-set", type=Path, help="the ruleset the arms judged, which the same-call rate is reread against")
    parser.add_argument("--label", action="append", metavar="POLICY=TEXT", help="a policy's display label")
    parser.add_argument("--parity", action="append", metavar="POLICY=REPORT",
                        help="a member's parity B report, one per arm (prefill, decode): a path, or an artifact:// "
                        "(read under the artifact root) or repo:// URI (ruleforge.core.paths.resolve); none for Gemma 4 "
                        "31B-it")
    parser.add_argument("--merge-manifest", action="append", metavar="POLICY=URI",
                        help="where a member's merge manifest is read (artifact://) instead of the weights directory "
                        "its draws recorded: for a root copied into this artifact root (checked by SHA-256)")
    parser.add_argument("--allow-code-mismatch", action="store_true",
                        help="read roots that ran different code, or no recorded clean commit (recorded)")
    parser.add_argument("--family", choices=sorted(PAPER_OUTPUTS), default=None,
                        help="build a paper family's readout from its roots renamed for --population: its arms, "
                        "labels, ruleset, merge manifests, parity and code-state override, in place of those flags")
    parser.add_argument("--population", default=None, metavar="POP",
                        help="with --family: pop-<first 8 hex of the population's scenario manifest>")
    parser.add_argument("--out-dir", type=driver.output_dir,
                        help="where to write the readout (and, but for --paper, the panels)")
    parser.add_argument("--paper", nargs="?", const="gemma", choices=sorted(PAPER_OUTPUTS), default=None,
                        help="a paper family (ruleforge.experiments.training.paper_families.PAPER_OUTPUTS; alone: the paper's training figure, "
                        "GEMMA_FAMILY): its panels into its files in this checkout's figures/, refused for any other "
                        "family")
    parser.add_argument("--force", action="store_true", help="overwrite an earlier run's files")
    driver.add_from_golden(parser, "re-render the panel --figure-a names from an installed golden")
    driver.add_figure(parser, "--figure-a", "--from_golden: where to draw panel a", repeatable=False)
    driver.add_dpi(parser, 200)
    args = driver.parse(parser, argv)

    if args.from_golden is not None:
        return rerender(parser, args)
    if (args.family is None) != (args.population is None):
        parser.error("--family and --population go together")
    if args.family is not None:
        named = [flag for flag in (*FAMILY_FLAGS, "--paper") if getattr(args, flag[2:].replace("-", "_"))]
        if named:
            parser.error(f"--family gives the family's own arguments; do not also pass {', '.join(named)}")
        try:
            family = PAPER_OUTPUTS[args.family].family.on_population(args.population)
        except ValueError as error:
            parser.error(str(error))
        parser.parse_args(family.arguments(), namespace=args)
    if given := [f"--{name.replace('_', '-')}" for name in FIGURE_FLAGS.values() if getattr(args, name) is not None]:
        parser.error(f"{', '.join(given)} name a re-render's panel (--from_golden); a build writes its own")
    if not (args.arm and args.reference and args.rule_set and args.out_dir):
        parser.error("a build takes --arm (one per arm), --reference, --rule-set and --out-dir; or pass --from_golden")
    if len(args.arm) > len(ARM_STYLES):
        parser.error(f"{len(args.arm)} arms; the panels draw at most {len(ARM_STYLES)}")
    if not args.rule_set.is_file():
        parser.error(f"no such file: --rule-set {args.rule_set}")
    paths = output_paths(args.out_dir, paper=args.paper)
    scratch = [path for name, path in paths.items() if not args.paper or name in ("json", "md")]
    hits = paper_paths(scratch, repo_root=repository_root())
    if hits:
        parser.error(f"{', '.join(hits)} would be among the paper's files ({', '.join(PAPER_OUTPUT_DIRS)}); write the "
                     "family readout elsewhere and copy it over the golden once two rebuilds agree")
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing and not args.force:
        parser.error(f"{existing} exist already; pass --force to overwrite them")
    driver.require_drawing(paths[f"{panel}.{suffix}"] for panel in PANELS for suffix in SUFFIXES)

    try:
        parity = {policy: [resolve(path) for path in reports]
                  for policy, reports in policy_pairs(args.parity, "--parity", repeat=True).items()}
    except (ArtifactRootError, ValueError) as error:
        parser.error(str(error))
    if missing := [str(path) for paths in parity.values() for path in paths if not path.is_file()]:
        parser.error(f"no such file: --parity {', '.join(missing)}")
    try:
        readout = build_readout(
            [FamilyArm(label, root, condition) for label, root, condition in args.arm],
            reference=args.reference,
            rule_set_path=args.rule_set,
            labels={policy: text for policy, (text,) in policy_pairs(args.label, "--label").items()},
            parity=parity,
            merge_manifests={policy: uri for policy, (uri,) in
                             policy_pairs(args.merge_manifest, "--merge-manifest").items()},
            allow_code_mismatch=args.allow_code_mismatch,
        )
    except REFUSALS as error:
        parser.error(str(error) + (NO_SCRATCH_BUILD if "--allow-incomplete" in str(error) else ""))
    refusal = paper_refusal(readout, paper=args.paper) if args.paper else None
    if refusal is not None:
        parser.error(f"--paper: this readout is {refusal}")

    # The readout first, so a panel that fails to write sits beside this run's readout, not an older one.
    driver.write_golden(paths["json"], readout, allow_nan=False)
    text = summary_markdown(readout)
    driver.write_text(paths["md"], text)
    names = drawn_names(readout)
    for panel in PANELS:
        driver.draw_each([paths[f"{panel}.{suffix}"] for suffix in SUFFIXES],
                         lambda out, panel=panel: draw(readout, out, panel, dpi=args.dpi, names=names))
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
