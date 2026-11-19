from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ruleforge.core.errors import RuleForgeError

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactSchemaError(ValueError, RuleForgeError):
    """Raised when an artifact record cannot be normalized safely."""


def read_or_derive_prompt_response_hash(
    record: "Mapping[str, Any]",
) -> "str | None":
    """Return a record's stored join key, or one derived from its top-level prompt/response; None if it has neither."""
    stored = record.get("prompt_response_hash")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    prompt, response = record.get("prompt"), record.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None
    return compute_prompt_response_hash(prompt, response)


def compute_prompt_response_hash(prompt: str, response: str) -> str:
    """Return the SHA-256 join key of a prompt/response pair."""
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompt and response must both be strings")
    payload = (
        prompt.encode("utf-8", errors="surrogatepass")
        + b"\x1e"
        + response.encode("utf-8", errors="surrogatepass")
    )
    return hashlib.sha256(payload).hexdigest()


def compute_scenario_id(scenario: Mapping[str, Any]) -> str:
    """Return a scenario's deterministic id: its prompt/response hash when it has one, else a hash of its JSON."""
    prompt = scenario.get("prompt")
    response = scenario.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        return compute_prompt_response_hash(prompt, response)

    canonical = json.dumps(
        scenario,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8", errors="surrogatepass")).hexdigest()


def prompt_sha256(entry: Mapping[str, str]) -> str:
    """A judge prompt's digest: SHA-256 of its system and user turns joined by a record separator."""
    payload = entry.get("system_prompt", "") + "\x1e" + entry.get("user_prompt", "")
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


def make_evaluation_record(
    scenario: Mapping[str, Any],
    evaluations: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical evaluation envelope."""
    scenario_copy = deepcopy(dict(scenario))
    scenario_id = compute_scenario_id(scenario_copy)
    record = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario": scenario_copy,
        "evaluations": deepcopy(dict(evaluations)),
    }
    if isinstance(scenario_copy.get("prompt"), str) and isinstance(
        scenario_copy.get("response"), str
    ):
        record["prompt_response_hash"] = scenario_id
    return record


def extract_prompt_response(record: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return a record's ``(prompt, response)`` from the top level or its ``scenario`` (None without one), refusing two
    that disagree."""
    pairs = []
    for source in (record, record.get("scenario")):
        if not isinstance(source, Mapping):
            continue
        prompt = source.get("prompt")
        response = source.get("response")
        if isinstance(prompt, str) and isinstance(response, str):
            pairs.append((prompt, response))

    if len(pairs) == 2 and pairs[0] != pairs[1]:
        raise ArtifactSchemaError(
            "conflicting nested scenario and top-level prompt/response"
        )
    return pairs[0] if pairs else None


def _jsonl_records(lines: Iterable[str], source: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactSchemaError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ArtifactSchemaError(
                f"{source}:{line_number}: expected a JSON object"
            )
        yield record


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield object records from JSONL, rejecting malformed nonblank lines."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        yield from _jsonl_records(handle, source)


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load strict JSONL object records."""
    return list(iter_jsonl_records(path))


def jsonl_records_of(data: bytes, source: str | Path) -> list[dict[str, Any]]:
    """Parse strict JSONL object records from a file's bytes; ``source`` names the file in errors."""
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8") as handle:
        return list(_jsonl_records(handle, Path(source)))


def load_jsonl_hashed(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Return a file's strict JSONL records and the SHA-256 of the bytes they were parsed from (one read)."""
    data = Path(path).read_bytes()
    return jsonl_records_of(data, path), hashlib.sha256(data).hexdigest()


def umask() -> int:
    """Return this process's umask, which only setting it reveals; 077 stands meanwhile, so nothing gets wider."""
    mask = os.umask(0o077)
    os.umask(mask)
    return mask


def write_text_atomically(path: str | Path, text: str) -> None:
    """Write text through a temporary file and a rename, so a reader never sees a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(text)
        temporary_path.chmod(0o666 & ~umask())  # a new file's mode, not the temporary file's 0600
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def link_new(target: str | Path, data: bytes) -> None:
    """Write ``data`` to ``target`` only if nothing is there yet; of two racing writers one wins
    (``FileExistsError``)."""
    target = Path(target)
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    partial = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        partial.chmod(0o666 & ~umask())  # a new file's mode, not the temporary file's 0600
        os.link(partial, target)
    finally:
        partial.unlink()


def write_jsonl_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Atomically write object records as JSONL."""
    serialized: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ArtifactSchemaError(f"record {index} must be a mapping")
        try:
            serialized.append(json.dumps(dict(record), ensure_ascii=False) + "\n")
        except (TypeError, ValueError) as exc:
            raise ArtifactSchemaError(
                f"record {index} is not JSON serializable: {exc}"
            ) from exc
    write_text_atomically(path, "".join(serialized))
