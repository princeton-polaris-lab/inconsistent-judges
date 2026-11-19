from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator, Optional, Sequence

from ruleforge.cli import records
from ruleforge.cli.common import CliError, subcommand
from ruleforge.core import paths
from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.optional import require
from ruleforge.env import PREFIX, env_value
from ruleforge.experiments import table
from ruleforge.experiments.paper_outputs import PAPER_OUTPUT_DIRS
from ruleforge.experiments.table import Step

logger = logging.getLogger(__name__)

#: The experiments, in the paper's order, and the other spelling two take.
EXPERIMENTS = tuple(row.name for row in table.TABLE)
ALIASES = {"prompt_ablation": "prompt-ablation", "ruleset_size": "ruleset-size"}
ALL = "all"
ARTIFACT_ROOT = "ARTIFACT_ROOT"
LEVELS = ("render", "rebuild", "verify", "subsets")
LEVEL_OPTIONS = {"rerun_dir": "verify", "exact": "verify", "json": "verify", "check": "subsets"}
VERIFY_PROGRAM = "ruleforge.experiments.reproduce.verify_rerun"
SUBSETS_EXPERIMENT = "ruleset-size"
SUBSETS_PROGRAMS = {False: "ruleforge.experiments.ruleset_size.generate_subsets",
                    True: "ruleforge.experiments.ruleset_size.verify_subsets"}
SUBSETS_OUT = Path("data") / "rules" / "ruleset_size_subsets"
OUT = "{out}"
PINNED_ENVIRONMENT = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "SOURCE_DATE_EPOCH": "0"}
LOG_TAIL = 20
LOCK_FNAME = "reproduce.lock"


def register(sub: argparse._SubParsersAction) -> None:
    """Add ``ruleforge reproduce`` to ``sub``'s commands."""
    text = ("the paper's figures from its goldens (render), its goldens from the published campaign roots (rebuild), "
            "a rerun's goldens checked (verify), the ruleset-size rule subsets (subsets)")
    parser = subcommand(sub, "reproduce", run_reproduce, text)
    parser.add_argument("experiment", choices=(*EXPERIMENTS, *ALIASES, ALL), metavar="EXPERIMENT",
                        help=f"{', '.join(EXPERIMENTS)} ({' and '.join(ALIASES)} too), or {ALL}")
    parser.add_argument("level", choices=LEVELS, help="render | rebuild | verify | subsets (ruleset-size's)")
    parser.add_argument("--out", type=Path, help="render: where the figures go; rebuild: where results/, logs/ and "
                                                 "the readouts go (a new directory, or one holding none of them); "
                                                 "subsets: where the subsets go (default: data/rules/ruleset_size_subsets)")
    parser.add_argument("--rerun", dest="rerun_dir", metavar="DIR",
                        help="verify: the tree holding the rerun's goldens (DIR/results/...), a path or "
                             "artifact:// URI")
    parser.add_argument("--exact", action="store_true",
                        help="verify: each golden equal to the committed one as parsed JSON, not within "
                             "its tolerance")
    parser.add_argument("--json", type=Path, metavar="FILE", help="verify: write every finding here")
    parser.add_argument("--check", action="store_true",
                        help="subsets: check the subsets against the committed checksums; write nothing")


def checkout() -> Path:
    """Return this package's source checkout, or refuse."""
    try:
        return paths.repository_root()
    except ArtifactRootError as error:
        raise CliError("ruleforge reproduce reads a source checkout of ruleforge -- the goldens and the producers' "
                       "committed inputs -- and this one is installed without it: clone the repository and run the "
                       "command there") from error


def experiments_of(name: str) -> tuple[str, ...]:
    """The experiments ``name`` (an experiment, its other spelling, or ``all``) stands for."""
    return EXPERIMENTS if name == ALL else (ALIASES.get(name, name),)


def entries(level: str, experiments: Sequence[str]) -> list[Step]:
    """``level``'s steps (a render's plots, or a rebuild's builds) of ``experiments``, in the table's order."""
    rows = [row for row in table.TABLE if row.name in experiments]
    return [step for row in rows for step in (row.plots if level == "render" else row.builds)]


def paper_lines(experiments: Sequence[str]) -> list[str]:
    """Each of ``experiments``' line naming the paper's figures and tables it reproduces, in the table's order."""
    return [f"{row.name}: {', '.join(row.paper)}" for row in table.TABLE if row.name in experiments and row.paper]


def _placed(word: str, out: Path) -> Path:
    return out / word[len(OUT) + 1:]


def _resolved(word: str, out: Path) -> str:
    if word.startswith(f"{OUT}/"):
        return str(_placed(word, out))
    if paths.is_artifact_uri(word) or paths.is_repository_uri(word):
        return str(paths.resolve(word))
    return word


def command(entry: Step, out: Path) -> list[str]:
    """``entry``'s command line as it runs: this interpreter, then its words resolved."""
    return [sys.executable, *(_resolved(word, out) for word in entry.words[1:])]


def inputs_of(entry: Step) -> list[str]:
    """The ``artifact://`` and ``repo://`` words of ``entry``'s command line, then the inputs it declares."""
    named = [word for word in entry.words if paths.is_artifact_uri(word) or paths.is_repository_uri(word)]
    return [*named, *entry.inputs]


def outputs_of(entry: Step, out: Path) -> list[Path]:
    """Every place below ``out`` ``entry`` writes: its ``{out}`` words and the golden its copy makes."""
    placed = [word for word in entry.words if word.startswith(f"{OUT}/")]
    placed += [_copy_target(entry)] if entry.copied_from else []
    return [_placed(word, out) for word in placed]


def _copy_target(entry: Step) -> str:
    return f"{OUT}/{entry.golden.path}"


def check_inputs(selected: Sequence[Step]) -> None:
    """Refuse a run one of whose inputs is not there, before any entry runs: each named, with what was looked for."""
    missing = []
    for entry in selected:
        for word in inputs_of(entry):
            try:
                path = paths.resolve(word)
            except (ArtifactRootError, ValueError) as error:
                raise CliError(f"{entry.id}: {word}: {error}") from error
            if not path.exists():
                missing.append(f"{entry.id}: {word} ({path})")
    if missing:
        raise CliError("inputs not found (fetch the experiment's data into the artifact root first): "
                       + "; ".join(missing))


def environment(level: str, repo: Path) -> dict[str, str]:
    """Return the environment an entry of ``level`` runs under."""
    pinned = dict(PINNED_ENVIRONMENT)
    if level == "render" and os.environ.get("SOURCE_DATE_EPOCH"):
        pinned["SOURCE_DATE_EPOCH"] = os.environ["SOURCE_DATE_EPOCH"]
    pinned["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])
    return {**os.environ, **pinned}


def log_path(entry: Step, out: Path) -> Path:
    """Where a rebuild writes ``entry``'s output: ``<out>/logs/<title, spaces as underscores>.log``."""
    return out / "logs" / f"{entry.title.replace(' ', '_')}.log"


def _run(entry: Step, out: Path, repo: Path, level: str, log: Optional[IO[str]], lock: int) -> int:
    for target in outputs_of(entry, out):
        target.parent.mkdir(parents=True, exist_ok=True)
    argv = command(entry, out)
    logger.info("%s: %s", entry.id, shlex.join(argv))
    return subprocess.run(argv, cwd=repo, env=environment(level, repo), stdout=log,
                          stderr=None if log is None else subprocess.STDOUT, check=False, pass_fds=(lock,)).returncode


def render(selected: Sequence[Step], out: Path, repo: Path, lock: int) -> int:
    """Draw every entry into ``out``; 1 at the first that fails."""
    for entry in selected:
        print(f"== {entry.id}", flush=True)
        if _run(entry, out, repo, "render", None, lock) != 0:
            print(f"ruleforge reproduce: {entry.id} failed; its output is above", file=sys.stderr)
            return 1
    print(f"wrote {out}")
    return 0


def rebuild(selected: Sequence[Step], out: Path, repo: Path, lock: int) -> int:
    """Run every build entry into ``out``; 1 at the first that fails."""
    for entry in selected:
        if entry.check_only:
            print(f"== {entry.title}: not rebuilt and not checked (a rebuild regenerates nothing it verifies); once "
                  f"regenerated, check with {shlex.join(entry.words)}")
            continue
        print(f"== {entry.title}", flush=True)
        log, start = log_path(entry, out), time.monotonic()
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = log.open("x", encoding="utf-8")  # refuse_existing found none: one there now is another's
        except FileExistsError:
            print(f"ruleforge reproduce: {log} was written during this run, by another of its entries or by a process "
                  f"that does not hold {out}'s lock; {entry.title} was not run", file=sys.stderr)
            return 1
        with handle:
            status = _run(entry, out, repo, "rebuild", handle, lock)
        if status != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL:]
            print("\n".join(tail), file=sys.stderr)
            print(f"ruleforge reproduce: {entry.title} failed (its log: {log})", file=sys.stderr)
            return 1
        if entry.copied_from:
            shutil.copyfile(_placed(entry.copied_from, out), _placed(_copy_target(entry), out))
        print(f"   {int(time.monotonic() - start)} s", flush=True)
    print(f"rebuilt into {out} (compare with the checkout's results/)")
    return 0


def refuse_existing(selected: Sequence[Step], out: Path) -> None:
    """Refuse a rebuild into a directory that holds one of its outputs (a golden, a readout's directory, a log)."""
    existing = sorted({str(path) for entry in selected if not entry.check_only
                       for path in (*outputs_of(entry, out), log_path(entry, out)) if os.path.lexists(path)})
    if existing:
        raise CliError(f"{', '.join(existing)} exist(s) already: a rebuild writes into a new directory, or one "
                       "holding none of its outputs")


def refuse_committed(out: Path, repo: Path, level: str) -> None:
    """Refuse an ``out`` at or below one of the checkout's directories of the paper's committed files
    (:data:`~ruleforge.experiments.paper_outputs.PAPER_OUTPUT_DIRS`), whatever the ``level``."""
    for name in PAPER_OUTPUT_DIRS:
        if out.resolve().is_relative_to((repo / name).resolve()):
            raise CliError(f"{out} lies in the checkout's {name}/, which holds the paper's committed files: a {level} "
                           "writes into a directory of its own")


@contextmanager
def claimed(out: Path) -> Iterator[int]:
    """Hold ``out``'s lock for the block, or refuse at once while another process holds it."""
    runs = records.check_place(out)
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / LOCK_FNAME
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
    except OSError as error:
        raise CliError(f"{path}, {out}'s lock, cannot be opened as a file of its own ({error})") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CliError(f"another process holds {path}: a ruleforge reproduce is writing {out} (or a producer it "
                           "started still is); give another --out, or run this once that one ends") from error
        yield descriptor
    finally:
        os.close(descriptor)  # the lock with it, once no producer holds it either


def program(words: Sequence[str], repo: Path, level: str, lock: Optional[int] = None) -> int:
    """Run ``python -m`` a program of the checkout's, as an entry runs, and return its exit status."""
    argv = [sys.executable, "-m", *words]
    logger.info("%s", shlex.join(argv))
    return subprocess.run(argv, cwd=repo, env=environment(level, repo), check=False,
                          pass_fds=() if lock is None else (lock,)).returncode


def place(given: str | os.PathLike[str], *, existing: bool = False) -> Path:
    """Return ``--out`` ``given`` as one absolute place."""
    try:
        return paths.one_place(given, kind="directory" if existing else None)
    except ValueError as error:
        raise CliError(f"--out {error}") from error


def exact(goldens: Sequence[str], repo: Path, tree: Path) -> int:
    """Compare each golden path under ``tree`` against the checkout's, as parsed JSON."""
    unequal = identical = 0
    for golden in goldens:
        rebuilt, committed = tree / golden, repo / golden
        present = rebuilt.is_file()
        equal = present and json.loads(rebuilt.read_bytes()) == json.loads(committed.read_bytes())
        same = present and rebuilt.read_bytes() == committed.read_bytes()
        print(f"{'EQUAL' if equal else 'DIFFERS' if present else 'MISSING':8}{golden}{'' if same else '  (bytes differ)'}")
        unequal, identical = unequal + (not equal), identical + same
    print(f"bytes: {identical} of {len(goldens)} identical")
    print(f"ALL {len(goldens)} EQUAL" if not unequal else f"{unequal} OF {len(goldens)} DIFFER")
    return 1 if unequal else 0


def verify(args: argparse.Namespace, experiments: Sequence[str], repo: Path) -> int:
    """Check the goldens the experiments' builds write in ``--rerun DIR``: within tolerances, or equal with ``--exact``."""
    if args.rerun_dir is None:
        raise CliError("verify checks a rerun's goldens: give the tree that holds them, --rerun DIR (DIR/results/...)")
    if args.exact and args.json is not None:
        raise CliError("--json writes the tolerance checks' findings: verify --exact prints its comparison alone")
    from ruleforge.experiments.reproduce import verify_rerun
    from ruleforge.experiments.reproduce.rerun import RerunError

    goldens = [step.golden for step in entries("rebuild", experiments) if step.golden is not None]
    places = [golden.path for golden in goldens]
    try:
        tree = verify_rerun.rerun_tree(args.rerun_dir, repo, places, what="--rerun")
        report = None if args.json is None else verify_rerun.report_place(
            args.json, repo_root=repo, tree=tree, goldens=places)
    except RerunError as error:
        raise CliError(str(error)) from error
    if args.exact:
        return exact(places, repo, tree)
    word = args.rerun_dir if paths.is_artifact_uri(args.rerun_dir) else str(tree)
    words = [VERIFY_PROGRAM, "--rerun-root", word, "--repo-root", str(repo), "--only", *[golden.name for golden in goldens]]
    return program(words + ([] if report is None else ["--json", str(report)]), repo, "verify")


def subsets(args: argparse.Namespace, argv: Sequence[str], repo: Path) -> int:
    """Generate the ruleset-size rule subsets into ``--out``, or check them with ``--check``."""
    if args.experiment != SUBSETS_EXPERIMENT:
        raise CliError(f"subsets makes the ruleset-size rule subsets: it is {SUBSETS_EXPERIMENT}'s "
                       f"(ruleforge reproduce {SUBSETS_EXPERIMENT} subsets), not {args.experiment}'s")
    if args.check:
        out = repo / SUBSETS_OUT if args.out is None else place(args.out, existing=True)
        return program([SUBSETS_PROGRAMS[True], "--output-root", str(out)], repo, "subsets")
    from ruleforge.experiments.ruleset_size.generate_subsets import output_place

    try:  # the program's own check of its output root (never in results/), made before the lock is taken
        out = output_place(repo / SUBSETS_OUT if args.out is None else args.out, repo)
    except ValueError as error:
        raise CliError(f"--out {error}") from error
    with claimed(out) as lock:
        record = records.open_record(out, "reproduce subsets", argv, None)
        with records.logging_to(record, verbose=args.verbose):
            return program([SUBSETS_PROGRAMS[False], "--output-root", str(out)], repo, "subsets", lock)


def pin_artifact_root() -> None:
    """Make a relative artifact root absolute in this process's environment: the programs started here run in the
    checkout, where a relative root names another place."""
    value = env_value(ARTIFACT_ROOT)
    if value is None or Path(value).expanduser().is_absolute():
        return
    try:
        absolute = str(paths.one_place(value))
    except ValueError as error:
        raise CliError(f"{PREFIX}{ARTIFACT_ROOT}: {error}") from error
    os.environ[PREFIX + ARTIFACT_ROOT] = absolute


def run_reproduce(args: argparse.Namespace, argv: Sequence[str]) -> int:
    pin_artifact_root()
    repo = checkout()
    print(f"ruleforge reproduce: the checkout {repo}", file=sys.stderr)
    experiments = experiments_of(args.experiment)
    for name, level in LEVEL_OPTIONS.items():
        if getattr(args, name) not in (None, False) and args.level != level:
            option = "--rerun" if name == "rerun_dir" else f"--{name}"
            raise CliError(f"{option} is {level}'s option: {args.level} takes none")
    if args.level == "verify":
        if args.out is not None:
            raise CliError("verify reads a tree (--rerun DIR) and writes none: it takes no --out")
        return verify(args, experiments, repo)
    if args.level == "subsets":
        return subsets(args, argv, repo)
    selected = entries(args.level, experiments)
    if args.out is None:
        raise CliError(f"{args.level} writes into --out DIR: give one")
    if args.level == "render":
        require("drawing the figures", "matplotlib")
    out = records.one_place(str(args.out), "--out")
    refuse_committed(out, repo, args.level)
    check_inputs(selected)
    if args.level == "rebuild":
        refuse_existing(selected, out)  # before anything is written, and again once no other run can write
    with claimed(out) as lock:
        if args.level == "rebuild":
            refuse_existing(selected, out)
        record = records.open_record(out, f"reproduce {args.level}", argv, None)
        out.mkdir(parents=True, exist_ok=True)
        with records.logging_to(record, verbose=args.verbose):
            for line in paper_lines(experiments):
                print(line, flush=True)
            return (render if args.level == "render" else rebuild)(selected, out, repo, lock)
