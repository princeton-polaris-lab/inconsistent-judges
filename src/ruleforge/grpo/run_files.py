from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.artifacts import write_text_atomically
from ruleforge.core.record_names import PROVENANCE_FNAME, STOPPED_KIND

#: A finished save: both adapter files and the trainer's provenance, written last.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
SAVE_FILES = (*ADAPTER_FILES, PROVENANCE_FNAME)
#: The early-stop marker: this file in the run's output directory ends the run after its next save.
STOP_FNAME = "STOP.json"
#: The trainer's record that it honoured the marker.
STOPPED_FNAME = "STOPPED.json"


class RunFileError(ValueError, RuleForgeError):
    """A run file that is not what the contract says it is."""


def _read(path: Path, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RunFileError(f"{path} is not JSON ({error})") from error
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise RunFileError(f"{path} is not a {kind!r} record")
    return payload


def checkpoint_name(step: int) -> str:
    return f"checkpoint-{int(step)}"


def checkpoint_steps(root: Path | str) -> list[int]:
    """Return the steps of the `checkpoint-N` directories under `root`."""
    steps = []
    for path in Path(root).glob("checkpoint-*"):
        suffix = path.name.split("-", 1)[1]
        if path.is_dir() and suffix.isdigit():
            steps.append(int(suffix))
    return sorted(steps)


def stop_marker_sha256(run_dir: Path | str) -> str:
    return hashlib.sha256((Path(run_dir) / STOP_FNAME).read_bytes()).hexdigest()


def read_stopped(run_dir: Path | str) -> Optional[dict[str, Any]]:
    """Return the trainer's record of the early stop it honoured, or None."""
    path = Path(run_dir) / STOPPED_FNAME
    return _read(path, STOPPED_KIND) if path.is_file() else None


def write_stopped(run_dir: Path | str, *, step: int, marker_sha256: Optional[str], at: str) -> Path:
    """Write the trainer's record that it stopped after saving `step` and return its path."""
    path = Path(run_dir) / STOPPED_FNAME
    payload = {"kind": STOPPED_KIND, "step": int(step), "marker_sha256": marker_sha256, "at": at}
    write_text_atomically(path, json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path
