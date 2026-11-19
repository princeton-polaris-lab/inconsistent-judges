from __future__ import annotations

import argparse
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional, Sequence

from ruleforge.cli import records
from ruleforge.cli.common import CliError, subcommand
from ruleforge.cli.reproduce import ALIASES
from ruleforge.core.optional import require
from ruleforge.env import artifact_root, env_value

LOG = logging.getLogger(__name__)
DATASET = "ai-law-society-lab/inconsistent-judges"
#: Each experiment's folder of the dataset where the two names differ: leniency reads prompt-ablation's roots.
FOLDERS = {**ALIASES, "leniency": "prompt-ablation"}


def _selection(args: argparse.Namespace) -> list[str]:
    if args.names and args.all:
        raise CliError(f"data {args.verb}: name the experiments or roots (EXPERIMENT|ROOT ...), or pass --all for "
                       "every one -- not both")
    return [FOLDERS.get(name, name) for name in args.names]


def _repository(args: argparse.Namespace) -> tuple[str, str]:
    given, kind = (args.model, "model") if args.model is not None else (args.dataset or DATASET, "dataset")
    local = _given(given, f"the {kind} repository", args.verb)
    return (str(local), kind) if local.is_dir() else (given, kind)


def _refuse_campaign(place: Path, what: str, verb: str = "fetch") -> tuple[Path, Path]:
    from ruleforge import hub

    return hub.refuse_campaign(place, f"data {verb}: {what}")


def _given(place: Optional[Path | str], what: str, verb: str) -> Optional[Path]:
    return None if place is None else records.one_place(place, f"data {verb}: {what}")


def _artifact_root(place: Path, verb: str) -> Path:
    from ruleforge import hub
    from ruleforge.evaluate.api import _spellings

    return hub.check_root(_spellings(place, f"data {verb}: the artifact root")[0])


def refuse_campaign_writes(root: Path, entries: Sequence[Any], download_dir: Optional[Path] = None) -> Optional[Path]:
    """Refuse a fetch of ``entries`` into ``root`` any write of which could land in a campaign root."""
    from ruleforge import hub

    _refuse_campaign(root, "the artifact root")
    held = None if download_dir is None else _refuse_campaign(download_dir, "the download directory")
    if held is not None:
        hub.refuse_in_destinations(held[0], root, entries, "data fetch: the download directory")
    for entry in entries:
        if hub.check_destination(root, entry.dest) == "present":  # refuses a symlink or a file on the way
            continue
        directory = root
        for part in PurePosixPath(entry.dest).parts[:-1]:
            if not (directory / part).is_dir():
                break
            directory = directory / part
        _refuse_campaign(directory, f"{entry.dest}'s directory")
    return None if held is None else held[0]


def _guard(verb: str) -> Callable[[Path, str], None]:
    def guard(place: Path, what: str) -> None:
        _refuse_campaign(place, what, verb)

    return guard


def run_fetch(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Place the repository's named entries under the artifact root and print what was placed, as JSON."""
    only = _selection(args)
    repository, repo_type = _repository(args)
    if args.root is None and env_value("ARTIFACT_ROOT") is None:
        raise CliError("data fetch: no artifact root: pass --root DIR, or set RULEFORGE_ARTIFACT_ROOT")
    root = _artifact_root(artifact_root() if args.root is None else _given(args.root, "the artifact root", "fetch"),
                          "fetch")
    local = Path(repository).is_dir()
    require("data fetch", "zstandard", *(() if local else ("huggingface_hub",)))
    from ruleforge import hub

    guard = _guard("fetch")
    scratch = None if local else hub.temporary_parent(root, guard)
    commit = hub.resolve_commit(repository, repo_type, args.revision)
    with hub.reference(repository, repo_type, commit, only, scratch, guard, root=root) as (_, selected):
        download_dir = refuse_campaign_writes(
            root, selected, None if local else _given(args.download_dir, "the download directory", "fetch"))
    record = None if args.dry_run else records.open_record(root, "data fetch", argv, None, artifact_root=True)
    with records.logging_to(record, verbose=args.verbose):
        if record is not None:
            LOG.info(f"data fetch: its run's record is {record}")
        LOG.info(f"reading {repository} ({repo_type}" + (f", commit {commit})" if commit else ", a directory)"))
        report = hub.fetch(repository, root, only=only, dry_run=args.dry_run, repo_type=repo_type, revision=commit,
                           download_dir=download_dir, scratch_dir=scratch, guard=guard, echo=LOG.info)
    print(records.describe({"repository": repository, "repo_type": repo_type, "revision": args.revision,
                            "commit": commit, **report}))
    return 0


def run_verify(args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Check the repository's named entries under ``--root`` and print what was checked, as JSON."""
    only = _selection(args)
    repository, repo_type = _repository(args)
    if not Path(repository).is_dir():
        require("data verify", "huggingface_hub")
    from ruleforge import hub

    commit = hub.resolve_commit(repository, repo_type, args.revision)
    with records.logging_to(None, verbose=args.verbose):
        report = hub.verify(repository, _given(args.root, "the artifact root", "verify"), only=only,
                            repo_type=repo_type, revision=commit, guard=_guard("verify"), echo=LOG.info)
    print(records.describe({"repository": repository, "repo_type": repo_type, "revision": args.revision,
                            "commit": commit, **report}))
    return 0


def _repository_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("names", nargs="*", metavar="EXPERIMENT|ROOT",
                        help="an experiment (leniency, ruleset-size, prompt-ablation, training, ruleforge, hierarchy), "
                             "populations, or a campaign root, a root with its experiment's other files (default: every "
                             "entry)")
    parser.add_argument("--all", action="store_true", help="every entry of the repository (the default, said)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", metavar="DIR|REPO",
                        help=f"the dataset repository: a Hub id, or a local directory laid out as one (default: "
                             f"{DATASET})")
    source.add_argument("--model", metavar="DIR|REPO", help="the model repository of the GRPO adapters")
    parser.add_argument("--revision", metavar="REV", help="a branch, tag or commit of a Hub repository (default: main); "
                                                          "read at the commit it names when the command starts")


def register(sub: argparse._SubParsersAction) -> None:
    """Add ``data fetch`` and ``data verify`` to the command line."""
    parser = sub.add_parser("data", help="the paper's data: fetch it from the Hub into an artifact root, verify it "
                                         "there", allow_abbrev=False)
    verbs = parser.add_subparsers(dest="verb", required=True)
    text = ("place the paper's data under an artifact root: every file checked against the repository's SHA256SUMS, "
            "each archive member by member, nothing written over what is there")
    fetch = subcommand(verbs, "fetch", run_fetch, text)
    _repository_options(fetch)
    fetch.add_argument("--root", type=Path, metavar="DIR", help="the artifact root to fill (default: "
                                                                "$RULEFORGE_ARTIFACT_ROOT)")
    fetch.add_argument("--download-dir", type=Path, metavar="DIR",
                       help="where a Hub download is held until it is placed (default: a hidden directory under the "
                            "root, removed after)")
    fetch.add_argument("--dry-run", action="store_true", help="check and list; write nothing")
    text = ("check the paper's data under an artifact root against the repository: each entry there, file by file; "
            "writes nothing")
    check = subcommand(verbs, "verify", run_verify, text)
    _repository_options(check)
    check.add_argument("--root", type=Path, required=True, metavar="DIR", help="the artifact root the data was fetched into")
