from __future__ import annotations

import errno
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Collection, Iterator


class _Entry:
    """One listed entry, its ``is_dir()`` failing if its name is among ``fails``."""

    def __init__(self, entry: os.DirEntry, fails: frozenset[str]) -> None:
        self._entry, self._fails = entry, fails

    def __getattr__(self, name: str) -> Any:
        return getattr(self._entry, name)

    def __fspath__(self) -> str:
        return self._entry.path

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        if self._entry.name in self._fails:
            raise PermissionError(errno.EACCES, "Permission denied", self._entry.path)
        return self._entry.is_dir(follow_symlinks=follow_symlinks)


class _Listing:
    """A directory's listing, used as ``os.scandir``'s is: a context manager and an iterator."""

    def __init__(self, listing: Any, fails: frozenset[str]) -> None:
        self._listing, self._fails = listing, fails

    def __enter__(self) -> "_Listing":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._listing.close()

    def __iter__(self) -> "_Listing":
        return self

    def __next__(self) -> _Entry:
        return _Entry(next(self._listing), self._fails)

    def close(self) -> None:
        self._listing.close()


def failing_scandir(is_dir_fails: Collection[str] = ()) -> Callable[..., Any]:
    """An ``os.scandir`` whose named entries fail ``is_dir()``."""
    real, fails = os.scandir, frozenset(is_dir_fails)

    def scandir(path: Any = ".") -> _Listing:
        return _Listing(real(path), fails)

    return scandir


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def vanishing_scandir(gone_after_listing: Collection[str] = (),
                      gone_before_listing: Collection[Path] = ()) -> Callable[..., Any]:
    """An ``os.scandir`` under which named entries vanish after, or just before, their directory is listed."""
    real, after, before = os.scandir, frozenset(gone_after_listing), frozenset(Path(path) for path in gone_before_listing)

    class Listing:
        def __init__(self, path: Any) -> None:
            with real(path) as entries:
                self._entries = list(entries)
            for entry in self._entries:
                if entry.name in after:
                    _remove(Path(entry.path))

        def __enter__(self) -> list[os.DirEntry]:
            return self._entries

        def __exit__(self, *exc_info: object) -> None:
            return None

    def scandir(path: Any = ".") -> Any:
        if isinstance(path, int):  # a listing by descriptor: shutil.rmtree's own, removing an entry here
            return real(path)
        if Path(os.fspath(path)) in before and os.path.lexists(path):
            _remove(Path(os.fspath(path)))
        return Listing(path)

    return scandir


@contextmanager
def concurrent_writer(directory: Path) -> Iterator[list[int]]:
    """For the block, a thread writing ``directory`` as a job of a campaign does while another's guard walks it."""
    from ruleforge.core.artifacts import write_text_atomically

    stop, rounds = threading.Event(), [0]

    def write() -> None:
        while not stop.is_set():
            write_text_atomically(directory / f"state{rounds[0] % 3}.json", "{}")
            lock = directory / ".draw.lock"
            lock.mkdir()
            (lock / "holder").write_text("1")
            shutil.rmtree(lock)
            rounds[0] += 1

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        yield rounds
    finally:
        stop.set()
        writer.join()
