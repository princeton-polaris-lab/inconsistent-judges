from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import re

logger = logging.getLogger(__name__)


def parse_scenarios_to_jsonl(scenario_pattern: re.Pattern, scenario_str: str, output_path: Path) -> None:
    """Parse scenarios from text with a named-group pattern and write one JSONL row per match."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {k: (v or "").strip() for k, v in match.groupdict().items()}
        for match in scenario_pattern.finditer(scenario_str)
    ]
    save_data_to_jsonl(entries, output_path)
    logger.info(f"Saved {len(entries)} scenarios to {output_path}")


def read_string_from_file(file_path: Path) -> str:
    """Return a file's text."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def save_string_to_file(string: str, file_path: Path) -> None:
    """Write a string to a file, replacing what it held."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(string)
    logger.info("Saved string to %s", file_path)


def save_data_to_jsonl(data: Any, file_path: Path) -> None:
    """Write data as JSONL: one line per item of a list, else one line."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    logger.info(f"Saved data to {file_path}")

