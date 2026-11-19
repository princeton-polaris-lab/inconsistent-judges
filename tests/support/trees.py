from __future__ import annotations

import time
from pathlib import Path
from typing import Union

#: The name an NFS client gives a file removed while open (``.nfs`` and a hexadecimal id).
NFS_SILLY_RENAME_PREFIX = ".nfs"


def is_silly_rename(path: Union[Path, str]) -> bool:
    """Whether ``path`` names an NFS client's silly rename (:data:`NFS_SILLY_RENAME_PREFIX`)."""
    return Path(path).name.startswith(NFS_SILLY_RENAME_PREFIX)


def tree_files(root: Union[Path, str]) -> list[Path]:
    """Every file under ``root``, sorted, without an NFS client's silly renames (never stat'ed: one may be gone)."""
    return [path for path in sorted(Path(root).rglob("*")) if not is_silly_rename(path) and path.is_file()]


def settled(root: Union[Path, str], *, timeout: float = 30.0) -> Path:
    """``root``, once the NFS client has deleted every silly rename under it (:func:`is_silly_rename`)."""
    root = Path(root)
    deadline = time.monotonic() + timeout
    while True:
        left = [path for path in root.rglob(f"{NFS_SILLY_RENAME_PREFIX}*") if is_silly_rename(path)]
        if not left:
            return root
        if time.monotonic() > deadline:
            raise AssertionError(f"{left}: still under {root} after {timeout} s -- a descriptor left open")
        time.sleep(0.05)
