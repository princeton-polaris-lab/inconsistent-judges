from __future__ import annotations

import json

import pytest

from ruleforge.grpo import run_files as rf


def test_the_early_stop_files_read_as_their_contract_says(tmp_path):
    assert rf.read_stopped(tmp_path) is None
    (tmp_path / rf.STOP_FNAME).write_text("{}")
    assert len(rf.stop_marker_sha256(tmp_path)) == 64
    rf.write_stopped(tmp_path, step=300, marker_sha256="m" * 64, at="now")
    assert rf.read_stopped(tmp_path) == {"kind": rf.STOPPED_KIND, "step": 300, "marker_sha256": "m" * 64, "at": "now"}
    (tmp_path / rf.STOPPED_FNAME).write_text("{")
    with pytest.raises(rf.RunFileError, match="is not JSON"):
        rf.read_stopped(tmp_path)
    (tmp_path / rf.STOPPED_FNAME).write_text(json.dumps({"kind": "other"}))
    with pytest.raises(rf.RunFileError, match="is not a 'smrc-grpo-early-stopped' record"):
        rf.read_stopped(tmp_path)


def test_checkpoint_steps_are_the_numbered_directories(tmp_path):
    for name in ("checkpoint-200", "checkpoint-100", "checkpoint-x"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-300").write_text("a file")
    assert rf.checkpoint_steps(tmp_path) == [100, 200] and rf.checkpoint_name(100) == "checkpoint-100"
