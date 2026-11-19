from __future__ import annotations

import re
from typing import Any

from ruleforge.core.errors import SamplingDesignError

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_identifier(name: str, value: Any, *, allow_slashes: bool = False) -> str:
    """Return ``value`` if every ``/``-separated segment (one, unless ``allow_slashes``) is a plain identifier."""
    segments = value.split("/") if isinstance(value, str) and allow_slashes else [value]
    if not all(isinstance(seg, str) and _IDENTIFIER.match(seg) for seg in segments):
        shape = "<segment>/<segment>" if allow_slashes else "<segment>"
        raise SamplingDesignError(
            f"{name} must be {shape} with each segment matching "
            f"{_IDENTIFIER.pattern!r}, got {value!r}"
        )
    return value
