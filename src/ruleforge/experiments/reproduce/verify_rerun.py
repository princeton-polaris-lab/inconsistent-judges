from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.paths import is_artifact_uri, one_place, repository_root, resolve, to_uri
from ruleforge.experiments import table
from ruleforge.experiments.reproduce.rerun import RerunError, select, verify_rerun
from ruleforge.core.artifacts import write_text_atomically

#: Directories of committed paper files a report never lands in.
COMMITTED = ("results",)


def rerun_tree(given: str, repo_root: Path, goldens: Sequence[str], *, what: str = "--rerun-root") -> Path:
    """Return the rerun tree ``given`` names, refused unless it is a directory other than the checkout and holds none of
    its ``goldens``: the committed goldens checked against themselves pass whatever was rebuilt."""
    try:
        tree = resolve(given) if is_artifact_uri(given) else one_place(given, kind="directory")
    except (ArtifactRootError, ValueError) as error:
        raise RerunError(f"{what} {given}: {error}; give the tree a rebuild's --out or a rerun wrote "
                         "(DIR/results/...)") from error
    if not tree.is_dir():
        raise RerunError(f"{what} {given} ({tree}) is no directory: give the tree a rebuild's --out or a rerun wrote")
    own = [relative for relative in goldens if (tree / relative).exists() and (repo_root / relative).exists()
           and (tree / relative).samefile(repo_root / relative)]
    if tree.resolve() == Path(repo_root).resolve() or own:
        held = f", or holds its {', '.join(own)}" if own else ""
        raise RerunError(f"{what} {given} ({tree}) is this checkout{held}: its committed goldens checked against "
                         "themselves pass whatever was rebuilt; give the tree the rebuild or rerun wrote")
    return tree


def report_place(given: str | Path, *, repo_root: Path, tree: Path, goldens: Sequence[str],
                 what: str = "--json") -> Path:
    """Return the report path ``given`` names, refused where writing it would replace what the check reads or the paper
    commits."""
    from ruleforge.evaluate.api import refuse_protected

    try:
        report = one_place(given)
    except ValueError as error:
        raise RerunError(f"{what} {error}") from error
    resolved = report.resolve()
    checkouts = [Path(repo_root)]
    try:
        checkouts.append(repository_root())
    except ArtifactRootError:  # an installed package: no checkout of its own
        pass
    for checkout in checkouts:
        for name in COMMITTED:
            if resolved.is_relative_to((checkout / name).resolve()):
                raise RerunError(f"{what} {given} lies in the checkout's {name}/, which holds the paper's committed "
                                 "files: give the report a place of its own")
    if resolved.is_relative_to((tree / "results").resolve()):
        raise RerunError(f"{what} {given} lies in the tree's results/, which holds the goldens it checks: give the "
                         "report a place of its own")
    for path in (base / relative for base in (Path(repo_root), tree) for relative in goldens):
        if resolved == path.resolve() or resolved.is_relative_to(path.resolve()):
            raise RerunError(f"{what} {given} is {path} ({path.resolve()}), which the check reads: give the report a "
                             "place of its own")
    try:
        refuse_protected(report, what)
    except ValueError as error:
        raise RerunError(str(error)) from error
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge a fresh rerun against the committed goldens' numbers.")
    parser.add_argument(
        "--rerun-root",
        required=True,
        help="tree holding the rerun's goldens, as a path or an artifact:// URI",
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument(
        "--only",
        nargs="+",
        default=[],
        metavar="NAME",
        help="check only these goldens",
    )
    parser.add_argument("--json", type=Path, help="write the full report here")
    return parser


def _root_reference(root: Path) -> str:
    try:
        return to_uri(root)
    except (ArtifactRootError, ValueError):
        return str(root)


#: Findings of each kind the terminal prints; `--json` carries every one.
MAX_LINES = 20


def _finding_lines(prefix: str, findings: List[Dict[str, Any]]) -> List[str]:
    lines = [
        f"  {prefix} {finding['golden']}: {finding['check']} at "
        f"{finding['path'] or '.'}: {finding['detail']}"
        for finding in findings[:MAX_LINES]
    ]
    if len(findings) > MAX_LINES:
        lines.append(
            f"  ... and {len(findings) - MAX_LINES} more; --json has all of them"
        )
    return lines


def _render(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for golden in report["goldens"]:
        if golden["status"] == "skipped":
            lines.append(f"  skip {golden['name']}: {golden['reason']}")
            continue
        counts = golden["checks_run"]
        found = {check: sum(1 for finding in golden["findings"] if finding["check"] == check)
                 for check in ("overlap", "sign", "flag")}
        largest = "; ".join(f"{item['path']} {item['was']:.6g} -> {item['now']:.6g}"
                            for item in golden["largest_differences"]) or "none"
        lines += [
            f"  {golden['name']}",
            f"    intervals compared       {counts['containment']}",
            f"    outside the interval     {counts['missed']} ({counts['expected_misses']:.1f} expected by chance"
            + (f"; expectation unavailable for {counts['without_sd']} with no stored standard deviation)"
               if counts["without_sd"] else ")"),
            f"    intervals apart          {found['overlap']} of {counts['overlap']}",
            f"    directions flipped       {found['sign']}",
            f"    flags flipped            {found['flag']} of {counts['flag']}",
            f"    values moved             {golden['differences']}",
            f"    largest differences      {largest}",
        ]
    # A new sample moves numbers by chance: those findings are reported, and only structural ones fail the check.
    lines += _finding_lines("note", [finding for finding in report["advisory"]
                                     if finding["check"] not in ("containment", "overlap", "sign", "flag", "drift")])
    lines += _finding_lines("FAIL", report["fatal"])
    return lines


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        goldens = select(table.goldens(), args.only)
        places = [golden.path for golden in goldens]
        rerun_root = rerun_tree(args.rerun_root, args.repo_root, places)
        destination = None if args.json is None else report_place(
            args.json, repo_root=args.repo_root, tree=rerun_root, goldens=places)
        report = verify_rerun(goldens, repo_root=args.repo_root, rerun_root=rerun_root)
    except (RerunError, ArtifactRootError, ValueError, OSError) as error:
        # ArtifactRootError is a RuntimeError, so it is named here to exit 2: the check could not be run.
        print(f"cannot verify this rerun: {error}", file=sys.stderr)
        return 2

    report = {**report, "rerun_root": _root_reference(rerun_root)}
    print(f"verifying {report['rerun_root']} against the goldens of {args.repo_root}")
    for line in _render(report):
        print(line)

    if destination is not None:
        write_text_atomically(destination, json.dumps(report, indent=2) + "\n")
        print(f"wrote {destination}")

    if report["ok"]:
        print(f"OK: {len(report['goldens'])} golden(s), every structural check passed; numbers that moved are above")
        return 0
    first = report["fatal"][0]
    print(
        f"FAILED: {len(report['fatal'])} check(s) failed; first is "
        f"{first['golden']} {first['check']} at {first['path'] or '.'}: "
        f"{first['detail']}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
