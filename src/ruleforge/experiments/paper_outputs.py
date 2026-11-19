from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from ruleforge.experiments.labels import SLOTS, STYLE_LABELS

__all__ = [
    "PAPER_OUTPUT_DIRS",
    "SLOT_NAMES",
    "paper_output_refusal",
    "paper_paths",
]

SLOT_NAMES = {STYLE_LABELS[slot]: slot for slot in SLOTS}

PAPER_OUTPUT_DIRS = ("results",)


def paper_paths(
    paths: Iterable[Optional[Path]], *, repo_root: Path, dirs: Sequence[str] = PAPER_OUTPUT_DIRS
) -> List[str]:
    """Return the ``paths`` that lie under one of ``dirs`` of ``repo_root``; None paths are skipped."""
    paper_dirs = [(Path(repo_root) / name).resolve() for name in dirs]
    return sorted(
        str(path)
        for path in paths
        if path is not None and any(Path(path).resolve().is_relative_to(directory) for directory in paper_dirs)
    )


def paper_output_refusal(paths: Iterable[Optional[Path]], is_paper: bool, *, repo_root: Path) -> Optional[str]:
    """Return why outputs that are not the paper's may not be written to ``paths``, or None when they may."""
    if is_paper:
        return None
    hits = paper_paths(paths, repo_root=repo_root)
    if not hits:
        return None
    return f"{', '.join(hits)} hold(s) the paper's files; this output is not one of them. Write it elsewhere"
