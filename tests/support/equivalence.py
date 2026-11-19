from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable, Union

from tests.support.trees import tree_files

VOLATILE_KEYS = frozenset({"created_at", "updated_at", "recorded_at", "started_at", "completed_at", "pid"})


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable(item) for key, item in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def read(path: Path) -> Any:
    """``path`` as what it records: a JSON or JSONL file parsed, its volatile keys left out."""
    if path.suffix == ".json":
        return _stable(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".jsonl":
        return [_stable(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            return sorted((member.name, hashlib.sha256(archive.extractfile(member).read()).hexdigest())
                          for member in archive.getmembers() if member.isfile())
    return path.read_bytes()


def tree(root: Union[Path, str], *, leave_out: Iterable[str] = ()) -> dict[str, Any]:
    """Every file under ``root`` by its relative path, read (:func:`read`)."""
    root = Path(root)
    skipped = tuple(leave_out)
    return {str(path.relative_to(root)): read(path) for path in tree_files(root)
            if not str(path.relative_to(root)).startswith(skipped)}


def unnamed(text: str, *names: str) -> str:
    """``text`` with each program name of ``names`` replaced by one word."""
    for name in names:
        text = text.replace(name, "PROGRAM")
    return text
