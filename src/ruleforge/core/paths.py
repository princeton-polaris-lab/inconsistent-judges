from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Callable, Iterator

from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root

ARTIFACT_ROOT_ENV = "RULEFORGE_ARTIFACT_ROOT"
ARTIFACT_SCHEME = "artifact://"
REPOSITORY_SCHEME = "repo://"


def is_artifact_uri(value: str) -> bool:
    """Return whether ``value`` is an ``artifact://`` reference."""
    return value.startswith(ARTIFACT_SCHEME)


def is_repository_uri(value: str) -> bool:
    """Return whether ``value`` is a ``repo://`` reference."""
    return value.startswith(REPOSITORY_SCHEME)


def checkout_root() -> Path | None:
    """Return the source checkout this package runs from (``src/ruleforge`` below a ``pyproject.toml``), or None when it
    runs installed."""
    here = Path(__file__).resolve()
    root = here.parents[3]
    return root if here.parents[2].name == "src" and (root / "pyproject.toml").is_file() else None


def repository_root() -> Path:
    """Return the source checkout this package runs from; ``repo://`` resolves only from a checkout."""
    root = checkout_root()
    if root is None:
        raise ArtifactRootError("cannot resolve repo:// input outside a ruleforge source checkout")
    return root


def to_uri(path: Path | str, env: dict[str, str] | None = None) -> str:
    """Return the ``artifact://`` form of a path under the artifact root; raise for one outside it."""
    root = artifact_root(env)
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"{resolved} is not under {ARTIFACT_ROOT_ENV} ({root}), so it has "
            "no artifact:// form. A committed file must not record it."
        ) from error
    return ARTIFACT_SCHEME + relative.as_posix()


def artifact_uri_or_path(path: Path | str, env: dict[str, str] | None = None) -> str:
    """Return a path's ``artifact://`` form where it has one, else its resolved absolute path."""
    try:
        return to_uri(path, env)
    except (ArtifactRootError, ValueError):
        return str(Path(path).expanduser().resolve())


def resolve(uri: str, env: dict[str, str] | None = None) -> Path:
    """Resolve an ``artifact://`` or ``repo://`` URI to a path inside its root; a plain path is returned
    ``~``-expanded."""
    if is_repository_uri(uri):
        relative = uri[len(REPOSITORY_SCHEME):]
        if not relative or relative.startswith("/"):
            raise ValueError(
                f"{uri!r} must name a relative path inside the repository"
            )
        if ".." in Path(relative).parts:
            raise ValueError(f"{uri!r} escapes the repository root via '..'")
        root = repository_root().resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(
                f"{uri!r} resolves to {target}, outside the repository root "
                f"({root})"
            )
        return target
    if not is_artifact_uri(uri):
        return Path(uri).expanduser()
    relative = uri[len(ARTIFACT_SCHEME):]
    if relative.startswith("/"):
        raise ValueError(
            f"{uri!r} has an absolute path after the scheme; artifact:// "
            "references are relative to the root by construction"
        )
    if ".." in Path(relative).parts:
        raise ValueError(
            f"{uri!r} escapes the artifact root via '..'. A committed "
            "reference must stay inside the tree it names."
        )
    root = artifact_root(env).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(
            f"{uri!r} resolves to {target}, outside the artifact root "
            f"({root}). A committed reference must stay inside the tree it "
            "names, symlinks included."
        )
    return target


PLACE_KINDS: dict[str, Callable[[str], bool]] = {"directory": os.path.isdir, "file": os.path.isfile}


def second_place(text: str, written: Path, base: str | os.PathLike[str] | None = None) -> Path | None:
    """Return where the kernel takes the spelling ``text`` (relative to ``base``, by default the working directory)
    when that differs from ``written`` (a ``..`` after a symlink)."""
    reached = Path(os.path.realpath(os.path.join(os.getcwd() if base is None else base, os.path.expanduser(text))))
    return reached if Path(os.path.realpath(written)) != reached else None


def made_on_the_way(text: str | os.PathLike[str]) -> Path | None:
    """Return the missing directory that ``mkdir(parents=True)`` on ``text`` would create and then leave by a ``..``, or
    None."""
    spelled = Path(os.path.join(os.getcwd(), os.fspath(text)))
    for index, part in enumerate(spelled.parts):
        before = Path(*spelled.parts[:index])
        if part == ".." and not os.path.lexists(before):
            return Path(os.path.realpath(before))
    return None


def one_place(given: str | os.PathLike[str], *, kind: str | None = None,
              base: str | os.PathLike[str] | None = None) -> Path:
    """Return ``given`` as one absolute normalized path from ``base`` (default: the working directory), refusing a
    spelling the kernel would take elsewhere; ``kind`` (``"directory"`` or ``"file"``) also requires such an entry along
    the path."""
    text = os.fspath(given)
    walked = os.path.join(os.getcwd() if base is None else base, os.path.expanduser(text))
    written = Path(os.path.normpath(walked))
    reached = second_place(text, written, base)
    if reached is not None:
        raise ValueError(f"{text} names two places: {written} as written out, and {reached} where the kernel takes "
                         "it (a '..' after a symlink, taken from the link's target); give the place without that '..'")
    if kind is not None and not PLACE_KINDS[kind](walked):
        raise ValueError(f"{text} is no {kind} there (as the kernel walks the path given, each '..' taken where it "
                         "stands)")
    return written


# ${oc.env:VAR} without a default; a default would be a literal absolute path.
_OC_ENV_REF = re.compile(r"\$\{oc\.env:\s*\w+\s*\}")

# A closing tag such as "</think>", spared before matching.
_CLOSING_TAG = re.compile(r"</[A-Za-z][\w-]*>")

# A "/"-rooted token anywhere in a string; the lookbehind spares URL schemes, "${...}/", "../" and glob stars.
_LITERAL_ABSOLUTE = re.compile(r"(?<![\w}:/.*])/[\w.-]")


def _is_absolute_string(value: str) -> bool:
    if value.startswith("/") and len(value) > 1:
        return True
    return _LITERAL_ABSOLUTE.search(_CLOSING_TAG.sub(" ", _OC_ENV_REF.sub("ENV", value))) is not None


def absolute_strings(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(key path, value)`` for every literal absolute path among a payload's string values."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from absolute_strings(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from absolute_strings(value, f"{path}[{index}]")
    elif isinstance(node, str):
        if _is_absolute_string(node):
            yield path, node


def walk_strictly(
    top: Path, descend: Callable[[Path], bool] | None = None, *, missing_ok: bool = False
) -> Iterator[tuple[Path, os.stat_result]]:
    """Yield each entry under ``top`` with its ``lstat`` through real directories only, raising every ``OSError``;
    ``missing_ok`` skips one that vanishes."""
    gone = (FileNotFoundError, NotADirectoryError) if missing_ok else ()
    pending = [Path(top)]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                names = [entry.name for entry in entries]
        except gone:
            continue
        for name in names:
            path = directory / name
            try:
                info = path.lstat()
            except gone:
                continue
            if stat.S_ISDIR(info.st_mode) and (descend is None or descend(path)):
                pending.append(path)
            yield path, info
