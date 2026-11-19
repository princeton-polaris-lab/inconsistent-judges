from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.paths import repository_root
from ruleforge.evaluate.estimates import estimates_source
from ruleforge.experiments import driver
from ruleforge.experiments.goldens import read_json_golden
from ruleforge.experiments.paper_outputs import PAPER_OUTPUT_DIRS, paper_paths
from ruleforge.experiments.campaign_roots import Root, arm_outcomes, arm_stack, open_roots, registered_condition
from ruleforge.experiments.training.family_readout import (TrainingFamilyError, check_design, check_engine,
                                                           draw_provenances, identity_differences)
from ruleforge.experiments.training.paper_families import PAPER_OUTPUTS

if TYPE_CHECKING:
    from ruleforge.evaluate.identity import ConditionSpec
    from ruleforge.experiments.training.family_readout import FamilyArm
    from ruleforge.experiments.training.paper_families import PaperFamily

REFUSALS = (ValueError, ArtifactRootError)
BUILD_FLAGS = ("family", "population", "golden", "force")


@dataclass(frozen=True)
class FamilyArms:
    """A family's base arm and its one trained arm as their roots hold them, by arm label: each arm's registered
    condition and accepted bundle outcomes (``{(scenario_id, draw): row}``)."""

    family: PaperFamily
    roots: Mapping[str, Root]
    specs: Mapping[str, ConditionSpec]
    cells: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]]
    n_draws: int

    @property
    def reference(self) -> FamilyArm:
        return self.family.arms[0]

    @property
    def member(self) -> FamilyArm:
        return self.family.arms[1]

    def labels(self) -> dict[str, str]:
        """Return each arm's display label, by arm label."""
        return {arm.label: self.family.labels.get(arm.label, arm.label) for arm in self.family.arms}

    def provenance(self) -> dict[str, Any]:
        """Return what a golden records of where its arms come from."""
        return {
            "arms": {arm.label: {"root": arm.root, "campaign_id": self.roots[arm.root].payload["campaign_id"],
                                 "condition_id": self.specs[arm.label].public_id,
                                 "condition_fingerprint": self.specs[arm.label].fingerprint,
                                 "prompt_style": self.specs[arm.label].prompt_style}
                     for arm in self.family.arms},
            "campaigns": [{**estimates_source(root.payload), "root": uri} for uri, root in self.roots.items()],
            "scenario_manifest_sha256": self.roots[self.reference.root].payload["scenario_manifest_sha256"],
            "n_decoder_draws": self.n_draws,
        }


def read_family_arms(family: PaperFamily, where: str, *, error: type[Exception]) -> FamilyArms:
    """Return ``family``'s two arms from their final roots, refused as ``error`` unless each arm's outcomes rebuild its
    estimates' grid and the two arms, of two or more decoder draws, differ only in weights and policy labels, on one
    design and engine as the family readout checks them."""
    if len(family.arms) != 2:
        raise error(f"{where}: {len(family.arms)} arms; this reads a base arm and one trained arm")
    roots = open_roots({arm.root: arm.root for arm in family.arms}, where, allow_incomplete=False,
                          allow_stack_mismatch=False)
    n_draws = int(roots[family.arms[0].root].payload["n_decoder_draws"])
    if n_draws < 2:
        raise error(f"{where}: {n_draws} decoder draw; the draw-to-draw rates need two or more")
    specs = {arm.label: registered_condition(roots[arm.root], arm.condition_id, f"{where} arm {arm.label}")
             for arm in family.arms}
    reference, member = family.arms
    if differing := identity_differences(specs[reference.label], specs[member.label]):
        raise error(f"{where}: arm {member.label} differs from the reference {reference.label} in {differing}, not "
                    "only in the weights it serves and its policy labels")
    try:
        check_design(roots, where)
        check_engine({arm.label: arm_stack(roots[arm.root], specs[arm.label], where) for arm in family.arms},
                     {arm.label: draw_provenances(roots[arm.root].path / specs[arm.label].condition_id, n_draws)
                      for arm in family.arms}, where)
    except TrainingFamilyError as refusal:
        raise error(str(refusal)) from refusal
    cells = {arm.label: arm_outcomes(roots[arm.root], specs[arm.label], n_draws, f"{where} arm {arm.label}")[0]
             for arm in family.arms}
    return FamilyArms(family, roots, specs, cells, n_draws)


def arm_names(golden: Mapping[str, Any], paper: str) -> tuple[str, str]:
    """Return the family ``paper``'s base and trained arm names in a family golden, as its panels print them."""
    family, drawn = golden["families"][paper], PAPER_OUTPUTS[paper].drawn
    return tuple(drawn.get(family[role], family["labels"][family[role]]) for role in ("reference", "member"))


def read_family_golden(path: Path, *, source: str, schema_version: int, noun: str,
                       error: type[Exception]) -> dict[str, Any]:
    """Return the golden at ``path``, refusing one of another source or schema version as ``error``."""
    golden = read_json_golden(path, "golden", error=error)
    if not isinstance(golden, dict) or golden.get("source") != source or golden.get("schema_version") != schema_version:
        raise error(f"{path} is no {noun} golden (source {source!r}, schema_version {schema_version})")
    return golden


def _build(parser: argparse.ArgumentParser, args: argparse.Namespace, render_flags: Sequence[str],
           build_golden: Callable[[Sequence[str], str], dict[str, Any]]) -> int:
    if given := driver.passed(args, render_flags):
        parser.error(f"{', '.join(given)} draw a golden's panels (--from_golden); a build writes the golden")
    if not (args.family and args.population and args.golden):
        parser.error("a build takes --family (one or more), --population and --golden; or pass --from_golden")
    if hits := paper_paths([args.golden], repo_root=repository_root()):
        parser.error(f"{hits[0]} would be among the paper's files ({', '.join(PAPER_OUTPUT_DIRS)}); write the golden "
                     "elsewhere and copy it over the golden once two rebuilds agree")
    if args.golden.exists() and not args.force:
        parser.error(f"{args.golden} exists already; pass --force to overwrite it")
    try:
        golden = build_golden(list(dict.fromkeys(args.family)), args.population)
    except REFUSALS as error:
        parser.error(str(error))
    driver.write_golden(args.golden, golden, allow_nan=False)
    return 0


def _render(parser: argparse.ArgumentParser, args: argparse.Namespace, panels: Sequence[str],
            read_golden: Callable[[Path], dict[str, Any]], figure_module: str) -> int:
    driver.refuse_beside_golden(parser, args, BUILD_FLAGS)
    figures = {panel: path for panel in panels if (path := getattr(args, f"figure_{panel}")) is not None}
    if not (args.paper and figures):
        flags = " or ".join(f"--figure-{panel}" for panel in panels)
        parser.error(f"--from_golden draws one family's panels: pass --paper and {flags}")
    try:
        golden = read_golden(args.from_golden)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.paper not in golden["families"]:
        parser.error(f"{args.from_golden} holds no family {args.paper!r} (it holds {sorted(golden['families'])})")
    draw = importlib.import_module(figure_module).draw
    for panel, path in figures.items():
        driver.draw_each([path], lambda out, panel=panel: draw(golden, args.paper, panel, out, dpi=args.dpi))
    return 0


def family_golden_main(argv: list[str] | None, *, noun: str,
                       build_golden: Callable[[Sequence[str], str], dict[str, Any]],
                       read_golden: Callable[[Path], dict[str, Any]], figure_module: str,
                       panels: Mapping[str, str]) -> int:
    """Build a training-family golden, or draw one family's panels from it (``--from_golden``) with
    ``figure_module``'s ``draw``; ``panels`` maps each panel to what its ``--figure-<panel>`` flag draws."""
    parser = argparse.ArgumentParser(
        description=f"Build the {noun} golden of the training families, or draw its panels.")
    parser.add_argument("--family", action="append", choices=sorted(PAPER_OUTPUTS),
                        help="a paper family to read (repeatable; paper_families.PAPER_OUTPUTS)")
    parser.add_argument("--population", metavar="POP",
                        help="pop-<first 8 hex of the population's scenario manifest>: the families' roots renamed for it")
    driver.add_golden(parser, "where a build writes the golden (outside results/)")
    parser.add_argument("--force", action="store_true", help="overwrite an earlier build's golden")
    driver.add_from_golden(parser, "draw a family's panels from an installed golden")
    parser.add_argument("--paper", choices=sorted(PAPER_OUTPUTS), help="--from_golden: the family whose panels to draw")
    for panel, drawn in panels.items():
        parser.add_argument(f"--figure-{panel}", type=driver.output_path, default=None,
                            help=f"--from_golden: where to draw {drawn}")
    driver.add_dpi(parser, 200)
    args = driver.parse(parser, argv)
    if args.from_golden is not None:
        return _render(parser, args, list(panels), read_golden, figure_module)
    return _build(parser, args, ("paper", *[f"figure_{panel}" for panel in panels]), build_golden)
