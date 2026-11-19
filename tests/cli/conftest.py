from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge.cli import main
from tests.test_sampling_runner import RULES, write_inputs


@pytest.fixture(autouse=True)
def _workdir(tmp_path, monkeypatch):
    """Each test runs from an empty directory of its own, under its own artifact root, with no root variables set."""
    for name in ("ARTIFACT_ROOT", "MODEL_ROOT", "BATCH_ROOT", "CLUSTER_DIR"):
        for prefix in ("RULEFORGE_",):
            monkeypatch.delenv(prefix + name, raising=False)
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def inputs(tmp_path) -> tuple[Path, Path]:
    """Three scenarios and a two-rule ruleset with a meta-rule, in a directory of their own."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    scenarios = write_inputs(directory)
    rules = directory / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES), encoding="utf-8")
    return scenarios, rules


@pytest.fixture
def cli(capsys):
    """Run ``ruleforge ARGV`` here: its exit status, its stdout (parsed as JSON when it is) and its stderr."""

    def run(*argv: str):
        status = main([str(item) for item in argv])
        captured = capsys.readouterr()
        try:
            return status, json.loads(captured.out), captured.err
        except ValueError:
            return status, captured.out, captured.err

    return run
