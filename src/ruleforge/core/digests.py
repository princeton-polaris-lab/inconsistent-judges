from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ruleforge.core.artifacts import read_or_derive_prompt_response_hash
from ruleforge.core.errors import ArtifactReadError

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON encoding every digest is taken over: sorted keys, no whitespace, no escapes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Streaming SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path | str) -> dict[str, Any]:
    """A reference file's name and full SHA-256, as a golden's provenance records it."""
    return {"name": Path(path).name, "sha256": sha256_file(path)}


def request_namespace(block_id: str) -> str:
    """Return a short provider custom-id namespace (16 hex characters) derived from a block id."""
    return "ns" + hashlib.sha256(block_id.encode("utf-8")).hexdigest()[:16]


def row_hash(row: Mapping[str, Any]) -> str:
    """Return a row's join key, deriving it from prompt/response when the row does not store one."""
    digest = read_or_derive_prompt_response_hash(row)
    if digest is None:
        raise ArtifactReadError(
            "row has neither prompt_response_hash nor string prompt/response"
        )
    return digest
