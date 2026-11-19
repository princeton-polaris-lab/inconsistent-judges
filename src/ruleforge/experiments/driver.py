from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar

from ruleforge.core.artifacts import write_text_atomically
from ruleforge.core.optional import require
from ruleforge.experiments import goldens

logger = logging.getLogger(__name__)

T = TypeVar("T")
#: Where a program's arguments go: its parser, or a group of it.
Arguments = Any


def output_path(value: str, *, below: bool = False) -> Path:
    """Return an output flag's path, refused (``refuse_protected``) where a write could land in a campaign root or a
    published run: a producer writes no campaign's records. ``below``: a directory, refused too for a link below it."""
    from ruleforge.evaluate.api import CampaignError, refuse_protected

    try:
        refuse_protected(value, "the output", below=below)
    except CampaignError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return Path(value)


def output_dir(value: str) -> Path:
    """Return an output directory flag's path, refused as ``output_path`` refuses a file's, and for a link below it."""
    return output_path(value, below=True)


def add_from_golden(target: Arguments, help: str = "re-render from an existing golden", *,
                    required: bool = False) -> None:
    """Add ``--from_golden``: draw the outputs from an existing golden instead of building it."""
    target.add_argument("--from_golden", type=Path, default=None, required=required, help=help)


def add_golden(target: Arguments, help: str = "where to write the golden (nothing by default)") -> None:
    target.add_argument("--golden", type=output_path, default=None, help=help)


def add_csv(target: Arguments, rows: str) -> None:
    """Add ``--csv``; ``rows`` says what one row of it is."""
    target.add_argument("--csv", type=output_path, default=None, help=f"where to write the CSV (one row per {rows})")


def add_summary(target: Arguments, help: str = "where to write the Markdown summary") -> None:
    target.add_argument("--summary", type=output_path, default=None, help=help)


def add_figure(target: Arguments, flag: str = "--figure", help: str = "the figure; repeatable (.png/.pdf)", *,
               repeatable: bool = True) -> None:
    """Add a figure's flag: a list of paths when ``repeatable``, else one path or None."""
    if repeatable:
        target.add_argument(flag, type=output_path, action="append", default=[], help=help)
    else:
        target.add_argument(flag, type=output_path, default=None, help=help)


def add_dpi(target: Arguments, default: Optional[int] = None) -> None:
    """Add ``--dpi``; with no default a .png is drawn at the figure's own resolution."""
    note = "default: %(default)s" if default is not None else "committed PNGs use 200"
    target.add_argument("--dpi", type=int, default=default, help=f"raster resolution for .png outputs ({note})")


def add_allow_incomplete(target: Arguments,
                         help: str = "read incomplete or non-final campaigns; scratch previews only") -> None:
    target.add_argument("--allow-incomplete", action="store_true", help=help)


def add_allow_stack_mismatch(target: Arguments, help: str = "build from (or re-render) estimates that report "
                             "inference-stack mismatches; the golden names them") -> None:
    target.add_argument("--allow-stack-mismatch", action="store_true", help=help)


def parse(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Return ``argv`` parsed and start the program's INFO log, without fontTools' line for every glyph it subsets."""
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    return args


def passed(args: argparse.Namespace, names: Iterable[str]) -> list[str]:
    """Return the flags of the argparse ``names`` that ``args`` was given, as a command line spells them."""
    return [f"--{name.replace('_', '-')}" for name in names if getattr(args, name)]


def refuse_beside_golden(parser: argparse.ArgumentParser, args: argparse.Namespace, names: Iterable[str]) -> None:
    """Refuse ``--from_golden`` beside any of a build's flags (argparse ``names``)."""
    if args.from_golden is not None and (given := passed(args, names)):
        parser.error(f"--from_golden re-renders an existing golden; do not also pass {', '.join(given)}")


def one_source(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a command line that gives both ``--from_golden`` and a build's ``--estimates``, or neither."""
    refuse_beside_golden(parser, args, ["estimates"])
    if args.from_golden is None and not args.estimates:
        parser.error("pass --estimates at least once, or --from_golden")


def golden_or_build(parser: argparse.ArgumentParser, args: argparse.Namespace, read: Callable[[Path], T],
                    build: Callable[[], T], refusals: tuple[type[Exception], ...]) -> T:
    """Return the golden ``--from_golden`` names as ``read`` reads it, or else ``build()``; one of ``refusals`` ends the
    program as a usage error that names it."""
    try:
        return read(args.from_golden) if args.from_golden is not None else build()
    except refusals as error:
        parser.error(str(error))


def as_stored(payload: T) -> T:
    """Return ``payload`` as its golden stores it, which is what a re-render reads."""
    return json.loads(json.dumps(payload, sort_keys=True))


def golden_text(payload: Any, *, sort_keys: bool = True, allow_nan: bool = True) -> str:
    """Return a golden's text: indented JSON ending in a newline."""
    return json.dumps(payload, indent=2, sort_keys=sort_keys, allow_nan=allow_nan) + "\n"


def write_golden(path: Optional[Path], payload: Any, *, sort_keys: bool = True, allow_nan: bool = True) -> None:
    """Write ``payload`` as the golden at ``path`` (:func:`golden_text`); nothing for no path."""
    if path is not None:
        write_text_atomically(path, golden_text(payload, sort_keys=sort_keys, allow_nan=allow_nan))
        logger.info("wrote golden %s", path)


def write_csv(path: Optional[Path], rows: Callable[[], Iterable[Mapping[str, Any]]], columns: Sequence[str]) -> None:
    """Write ``rows()`` as CSV at ``path``; nothing for no path."""
    if path is not None:
        goldens.write_csv(rows(), columns, path)
        logger.info("wrote %s", path)


def write_text(path: Optional[Path], text: str) -> None:
    if path is not None:
        write_text_atomically(path, text)
        logger.info("wrote %s", path)


def require_drawing(paths: Iterable[Optional[Path]]) -> None:
    """Refuse, before any file is written, a run that will draw any of ``paths`` without the plotting library."""
    if any(path is not None for path in paths):
        require("drawing the figures", "matplotlib")


def draw_each(paths: Iterable[Optional[Path]], draw: Callable[[Path], None]) -> None:
    """Draw each of ``paths`` with ``draw``; a None path is skipped."""
    for path in paths:
        if path is not None:
            draw(path)
            logger.info("wrote %s", path)

