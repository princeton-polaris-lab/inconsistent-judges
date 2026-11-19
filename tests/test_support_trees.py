from __future__ import annotations

import pytest

from tests.support.trees import is_silly_rename, settled, tree_files


def test_a_listing_names_every_file_but_a_silly_rename(tmp_path):
    """A held draw lock removed at a run's end stays, for a moment, as ``.nfs<id>`` beside the draw."""
    (tmp_path / "condition" / "draw-000").mkdir(parents=True)
    (tmp_path / "condition" / "draw-000" / "attempts.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "condition" / ".draw-001.lock").write_text("", encoding="utf-8")
    (tmp_path / "condition" / ".nfs85ee40066a880d4000017d62").write_text("", encoding="utf-8")
    assert is_silly_rename(tmp_path / "condition" / ".nfs85ee40066a880d4000017d62")
    assert not is_silly_rename("condition/.draw-001.lock")
    assert tree_files(tmp_path) == tree_files(str(tmp_path)) == [
        tmp_path / "condition" / ".draw-001.lock", tmp_path / "condition" / "draw-000" / "attempts.jsonl"]


def test_a_settled_tree_holds_no_silly_rename_and_one_left_fails(tmp_path):
    (tmp_path / "condition").mkdir()
    (tmp_path / "condition" / "draw-000.json").write_text("{}")
    assert settled(tmp_path) == tmp_path
    (tmp_path / "condition" / ".nfs000000000123").write_text("")
    with pytest.raises(AssertionError, match="a descriptor left open"):
        settled(tmp_path, timeout=0.1)
