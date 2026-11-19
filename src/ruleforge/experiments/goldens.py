from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from ruleforge.core.artifacts import write_text_atomically


def read_json_golden(path: Path | str, where: str, *, error: type[Exception]) -> Any:
    """Return a stored golden's JSON; refused as ``error``, naming the file, when it cannot be read as JSON."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        json.dumps(value, ensure_ascii=False).encode("utf-8")
        return value
    except (OSError, ValueError, RecursionError) as failure:
        raise error(f"{where}: {path} cannot be read as JSON ({type(failure).__name__}: {failure})") from failure


def read_analysis_golden(path: Path | str, analysis: str, schema_version: int, *,
                         error: type[Exception]) -> Dict[str, Any]:
    """Return the golden at ``path``, refused as ``error`` unless it is an ``analysis`` payload of ``schema_version``."""
    payload = read_json_golden(path, "golden", error=error)
    if not (isinstance(payload, dict)
            and (payload.get("analysis"), payload.get("schema_version")) == (analysis, schema_version)):
        raise error(f"{path} is not a {analysis} payload of schema {schema_version}")
    return payload


def write_csv(rows: Iterable[Mapping[str, Any]], columns: Sequence[str], path: Path) -> None:
    """Write ``rows`` as CSV with LF line ends; a boolean is spelled ``true`` or ``false``."""
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: str(value).lower() if isinstance(value, bool) else value for key, value in row.items()})
    write_text_atomically(path, handle.getvalue())


#: Each interval CSV column's suffix and the interval key it holds.
INTERVAL_COLUMNS = (("", "estimate"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high"))


def interval_record(summary: Mapping[str, Any]) -> Dict[str, float]:
    """Return a rate's estimate, interval bounds and bootstrap sd, as floats."""
    return {key: float(summary[key]) for key in ("estimate", "ci_low", "ci_high", "sd")}


def format_interval(entry: Optional[Mapping[str, Any]], signed: bool = False) -> str:
    """Return ``estimate [low, high]`` to four places, or ``undefined`` when there is no estimate."""
    if entry is None or entry.get("estimate") is None:
        return "undefined"
    sign = "+" if signed else ""
    return f"{entry['estimate']:{sign}.4f} [{entry['ci_low']:{sign}.4f}, {entry['ci_high']:{sign}.4f}]"
