from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _looks_like_a_torn_record(output_file: Path) -> bool:
    with output_file.open("rb") as handle:
        head = handle.read(64)
    return head.lstrip().startswith(b"{")


def _parses(raw: bytes) -> bool:
    try:
        json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return True


def count_complete_jsonl_records(output_file: Path, repair: bool = False) -> int:
    """Count the complete leading JSON records of a checkpoint (0 for a missing file). ``repair`` completes the newline
    of a last record that parses, truncates a last line that does not, and refuses damage that more lines follow."""
    if not output_file.exists():
        return 0

    valid = 0
    safe_offset = 0
    with output_file.open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n") or not _parses(raw):
                break
            valid += 1
            safe_offset += len(raw)

    if repair:
        total_size = output_file.stat().st_size
        if valid == 0 and total_size > 0 and not _looks_like_a_torn_record(output_file):
            # Content that cannot be a partial record is more likely the wrong file than a checkpoint.
            raise ValueError(
                f"'{output_file}' has {total_size} byte(s) that do not begin a "
                "JSON record; refusing to truncate it. Check the path, or delete "
                "the file if it really is a failed run's output."
            )
        with output_file.open("rb") as handle:
            handle.seek(safe_offset)
            tail = handle.read()
        if b"\n" in tail[:-1]:
            # A write cut short damages only the last line; truncating here would drop the records after it.
            raise ValueError(
                f"line {valid + 1} of '{output_file}' is not a JSON record, and more lines follow it; refusing to "
                "truncate the file there. Repair that line by hand."
            )
        if tail and not tail.endswith(b"\n") and _parses(tail):
            with output_file.open("ab") as handle:
                handle.write(b"\n")
            logger.warning("Repaired '%s': completed the newline of its last record (an interrupted write).",
                           output_file)
            return valid + 1
        if safe_offset != total_size:
            with output_file.open("ab") as handle:
                handle.truncate(safe_offset)
            logger.warning(
                "Repaired '%s': dropped %d trailing byte(s) after %d complete "
                "record(s) (partial write from an interrupted run).",
                output_file,
                total_size - safe_offset,
                valid,
            )
    return valid


def prepare_resume_skip(
    output_path: Path,
    resume: bool,
    label: str = "output",
) -> int:
    """Return how many records of ``output_path`` to skip; an existing file without ``resume`` raises ``ValueError``."""
    if output_path.exists() and not resume:
        raise ValueError(
            f"Output file '{output_path}' already exists. "
            "Aborting to avoid appending to existing file."
        )

    skip_count = count_complete_jsonl_records(output_path, repair=resume)
    if skip_count:
        logger.info(
            "Resume: %s '%s' already has %d complete record(s); skipping those.",
            label,
            output_path,
            skip_count,
        )
    return skip_count
