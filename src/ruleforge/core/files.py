from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from ruleforge.core.artifacts import iter_jsonl_records, write_text_atomically


def process_jsonl_file(input_file: str | Path) -> Iterator[dict[str, Any]]:
    """Yield a JSONL file's object rows, blank lines passed over; a line that is no JSON object is refused, by number."""
    return iter_jsonl_records(input_file)


def json_sidecar_text(payload: Any) -> str:
    """Return a JSON sidecar's text in the shared indentation."""
    return json.dumps(payload, indent=2) + "\n"


def write_json_sidecar(path: Path | str, payload: Any) -> Path:
    """Write a JSON sidecar atomically."""
    destination = Path(path)
    write_text_atomically(destination, json_sidecar_text(payload))
    return destination
