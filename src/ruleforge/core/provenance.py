from __future__ import annotations

import functools
import hashlib
import io
import os
import subprocess
import tarfile
import uuid
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from ruleforge.core.digests import sha256_file
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.paths import checkout_root, repository_root as repo_root

SOURCE_PATHS = ("src", "configs", "pyproject.toml", "uv.lock")
SNAPSHOT_DIRNAME = "code_snapshots"
_EXCLUDED_PARTS = ("__pycache__",)
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")
DISTRIBUTION = "ruleforge"


def _git(args: list[str], repo: Path | None) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=str(repo) if repo else None, capture_output=True, text=True,
                                check=False)
    except (OSError, ValueError):
        return None
    return result.stdout if result.returncode == 0 else None


def git_revision(repo: Path | None = None) -> str | None:
    """Return the current commit, or None outside a git checkout."""
    output = _git(["rev-parse", "HEAD"], repo)
    return output.strip() or None if output is not None else None


def git_is_dirty(repo: Path | None = None) -> bool:
    """Return whether the working tree differs from HEAD; True when git cannot answer."""
    output = _git(["status", "--porcelain"], repo)
    return bool(output.strip()) if output is not None else True


def distribution() -> Optional[metadata.Distribution]:
    """Return the installed distribution, skipping a stale ``*.egg-info`` beside the checkout's package, or None."""
    beside = Path(__file__).resolve().parents[2]
    for found in metadata.distributions(name=DISTRIBUTION):
        stale = found.read_text("METADATA") is None and Path(str(found.locate_file(""))).resolve() == beside
        if not stale:
            return found
    return None


def package_version() -> Optional[str]:
    """Return the installed distribution's version, or None when none is installed."""
    found = distribution()
    return None if found is None else found.version


class CheckoutMismatch(SamplingDesignError):
    """Raised when this process imported ruleforge from another checkout than the entry point it runs."""


def require_checkout(expected: Path | str) -> Path:
    """Return ``expected`` once this process imports ruleforge from it (a pickled task may run in another checkout's
    environment)."""
    imported, named = checkout_root(), Path(expected).resolve()
    if imported != named:
        raise CheckoutMismatch(
            f"this process imported ruleforge from {imported or Path(__file__).resolve().parents[1]}, not from {named}, "
            "the checkout whose entry point it runs: "
            "its interpreter's environment is bound to another checkout (an editable install elsewhere, or "
            "PYTHONPATH). Rebuild the environment from this checkout or launch from the checkout it is bound to"
        )
    return named


def _git_ls_files(root: Path) -> Optional[list[str]]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS],
            cwd=str(root), capture_output=True, text=True, check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return [path for path in result.stdout.split("\0") if path]


def installed_source_files(dist: metadata.Distribution) -> list[tuple[str, Path]]:
    """Return an installed distribution's package files as ``(relative path, path)``, sorted; metadata and bytecode left
    out."""
    found = {}
    for entry in dist.files or ():
        relative = PurePosixPath(str(entry))
        if (relative.parts[0] == ".." or relative.parts[0].endswith(".dist-info") or len(relative.parts) < 2
                or any(part in _EXCLUDED_PARTS for part in relative.parts) or relative.suffix in _EXCLUDED_SUFFIXES):
            continue
        found[relative.as_posix()] = Path(dist.locate_file(entry))
    return sorted(found.items())


def source_files(root: Optional[Path] = None) -> Optional[list[Path]]:
    """Return the checkout's source files as sorted relative paths, or None without git."""
    base = Path(root) if root is not None else repo_root()
    listed = _git_ls_files(base)
    if listed is None:
        return None
    kept = []
    for relative in sorted(set(listed)):
        path = Path(relative)
        if any(part in _EXCLUDED_PARTS for part in path.parts) or path.suffix in _EXCLUDED_SUFFIXES:
            continue
        if (base / path).is_file():
            kept.append(path)
    return kept


def code_state(root: Optional[Path] = None) -> dict[str, Any]:
    """Return the commit, dirtiness and source digest of a checkout (the imported one, hashed once per process, by
    default)."""
    if root is None:
        return dict(_imported_code_state())
    return _code_state(Path(root))


@functools.lru_cache(maxsize=1)
def _imported_code_state() -> dict[str, Any]:
    checkout = checkout_root()
    return _code_state(checkout) if checkout is not None else installed_code_state(distribution())


def _digest(files: list[tuple[str, Path]]) -> str:
    hasher = hashlib.sha256()
    for relative, path in files:
        hasher.update(f"{relative}\0{sha256_file(path)}\n".encode("utf-8"))
    return hasher.hexdigest()


def installed_code_state(dist: Optional[metadata.Distribution]) -> dict[str, Any]:
    """Return the code state of an installed distribution: no commit, dirty, and a digest of its own files."""
    files = None if dist is None else installed_source_files(dist)
    return {"code_revision": None, "code_dirty": True, "code_state_sha256": _digest(files) if files else None}


def _code_state(base: Path) -> dict[str, Any]:
    files = source_files(base)
    digest = None if files is None else _digest([(path.as_posix(), base / path) for path in files])
    return {
        "code_revision": git_revision(base),
        "code_dirty": git_is_dirty(base),
        "code_state_sha256": digest,
    }


def snapshot_path(campaign_root: Path | str, code_state_sha256: str) -> Path:
    return Path(campaign_root) / SNAPSHOT_DIRNAME / f"{code_state_sha256[:16]}.tar.gz"


def write_code_snapshot(
    campaign_root: Path | str, state: dict[str, Any], root: Optional[Path] = None
) -> Optional[Path]:
    """Archive the sources behind ``state`` under the campaign root, one tarball per digest; return its path or None."""
    digest = state.get("code_state_sha256")
    if not digest:
        return None
    destination = snapshot_path(campaign_root, digest)
    if destination.exists():
        return destination
    if root is None and checkout_root() is None:
        installed = distribution()
        files = [] if installed is None else installed_source_files(installed)
    else:
        base = Path(root) if root is not None else repo_root()
        files = [(path.as_posix(), base / path) for path in source_files(base) or []]
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative, path in files:
            archive.add(path, arcname=relative, recursive=False)
    # Jobs of one launch share a digest and may write at once; a per-process temp name keeps them apart.
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(buffer.getvalue())
    if destination.exists():
        temporary.unlink()
        return destination
    os.replace(temporary, destination)
    return destination
