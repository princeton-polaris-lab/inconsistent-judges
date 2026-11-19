from __future__ import annotations

from typing import Iterable, Tuple


def require(checks: Iterable[Tuple[bool, str]], where: str, *, error: type[Exception]) -> None:
    """Raise ``error`` once, naming every failed check, or return."""
    failed = [message for ok, message in checks if not ok]
    if failed:
        raise error(f"{where}: " + "; ".join(failed))
