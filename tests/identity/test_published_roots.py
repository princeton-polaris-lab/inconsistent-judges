from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.identity import generate_fixtures
from tests.identity import published as pub

pytestmark = pytest.mark.cluster

RESULTS = pub.REPO_ROOT / "results"


def _artifact_root() -> Path:
    from ruleforge.core.errors import ArtifactRootError
    from ruleforge.env import artifact_root

    try:
        return artifact_root()
    except ArtifactRootError as error:
        pytest.skip(str(error))


def rebuild_dir() -> Path | None:
    """The rebuilt goldens' directory: ``RULEFORGE_REBUILD_DIR``, or None when unset."""
    value = os.environ.get("RULEFORGE_REBUILD_DIR")
    return Path(value) if value else None


def estimates_entries(value: Any, path: str = "") -> Iterator[tuple[str, Any, Any, Any]]:
    """``(where, estimates_sha256, created_at, retry_scope)`` of every recorded estimates file in a golden."""
    if isinstance(value, dict):
        if "estimates_sha256" in value:
            yield path, value["estimates_sha256"], value.get("created_at"), value.get("retry_scope")
        for key, item in value.items():
            yield from estimates_entries(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from estimates_entries(item, f"{path}[{index}]")


def test_the_identity_fixtures_are_what_the_published_roots_hold():
    root = _artifact_root()
    # The table and batch checks read the roots row by row from the fixtures: an empty fixture would compare nothing.
    assert all(pub.read_fixture(name) for name in (*generate_fixtures.TABLE_KINDS, pub.BATCH))
    assert generate_fixtures.check(root) == []


def test_a_golden_rebuild_reproduces_the_committed_goldens():
    rebuilt = rebuild_dir()
    if rebuilt is None:
        pytest.skip("no golden rebuild to compare (RULEFORGE_REBUILD_DIR unset)")
    differing = []
    for golden in pub.GOLDENS:
        ours = json.loads((rebuilt / "results" / golden).read_text(encoding="utf-8"))
        committed = json.loads((RESULTS / golden).read_text(encoding="utf-8"))
        if list(estimates_entries(ours)) != list(estimates_entries(committed)):
            differing.append(f"{golden}: an estimates file's sha256, created_at or retry scope")
        elif ours != committed:
            differing.append(golden)
    assert differing == []


def test_the_rebuild_directory_is_named_by_one_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_REBUILD_DIR", str(tmp_path))
    assert rebuild_dir() == tmp_path


def test_every_golden_is_committed_and_records_its_estimates():
    for golden in pub.GOLDENS:
        assert (RESULTS / golden).is_file(), golden
    assert any(list(estimates_entries(json.loads((RESULTS / golden).read_text(encoding="utf-8"))))
               for golden in pub.GOLDENS)
