from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from ruleforge.core.artifacts import (
    extract_prompt_response,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactSchemaError,
    compute_prompt_response_hash,
    compute_scenario_id,
    iter_jsonl_records,
    jsonl_records_of,
    link_new,
    load_jsonl_hashed,
    load_jsonl_records,
    make_evaluation_record,
    read_or_derive_prompt_response_hash,
    write_jsonl_records,
    write_text_atomically,
)


def test_prompt_response_hash_matches_feature_branch_vector():
    assert compute_prompt_response_hash("prompt", "response") == (
        "2d9bafef7710c996c0a3a9052f8cb3a56909f90c78d0e7ebff24d04340dfc583"
    )
    assert compute_scenario_id({"prompt": "prompt", "response": "response"}) == (
        compute_prompt_response_hash("prompt", "response")
    )


@pytest.mark.parametrize("prompt,response", [(1, "response"), ("prompt", 1)])
def test_prompt_response_hash_requires_strings(prompt, response):
    with pytest.raises(TypeError, match="must both be strings"):
        compute_prompt_response_hash(prompt, response)


def test_generic_scenario_id_is_order_independent():
    assert compute_scenario_id({"question": "Q", "facts": ["A", "B"]}) == (
        compute_scenario_id({"facts": ["A", "B"], "question": "Q"})
    )
    assert compute_scenario_id({"prompt": "P"}) != compute_scenario_id({"prompt": "Q"})


def test_make_evaluation_record_is_versioned_and_detached():
    scenario = {"prompt": "P", "response": "R", "nested": {"value": 1}}
    evaluations = {"judge": {"full_ruleset": {"type": "yes"}}}

    record = make_evaluation_record(scenario, evaluations)
    scenario["nested"]["value"] = 2
    evaluations["judge"]["full_ruleset"]["type"] = "no"

    expected_hash = compute_prompt_response_hash("P", "R")
    assert record == {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scenario_id": expected_hash,
        "prompt_response_hash": expected_hash,
        "scenario": {"prompt": "P", "response": "R", "nested": {"value": 1}},
        "evaluations": {"judge": {"full_ruleset": {"type": "yes"}}},
    }


def test_an_evaluation_record_without_a_prompt_response_pair_has_no_pair_hash():
    record = make_evaluation_record({"question": "Q"}, {})
    assert record == {"schema_version": ARTIFACT_SCHEMA_VERSION, "scenario_id": compute_scenario_id({"question": "Q"}),
                      "scenario": {"question": "Q"}, "evaluations": {}}


def test_jsonl_round_trip_creates_parent_and_ignores_blank_lines(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    records = [{"value": 1}, {"value": "naïve"}]
    write_jsonl_records(path, records)
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert load_jsonl_records(path) == records
    assert list(iter_jsonl_records(path)) == records


@pytest.mark.parametrize(
    "contents,match",
    [
        ("{bad json}\n", "invalid JSON"),
        (json.dumps(["not", "an", "object"]) + "\n", "expected a JSON object"),
    ],
)
def test_jsonl_reader_rejects_invalid_records(tmp_path, contents, match):
    path = tmp_path / "bad.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ArtifactSchemaError, match=match):
        load_jsonl_records(path)
    with pytest.raises(ArtifactSchemaError, match=f"{path}:1: {match}"):
        load_jsonl_hashed(path)


def replaced_once_read(patch: pytest.MonkeyPatch, replacements: Mapping[Path, bytes]) -> None:
    """Replace each file of ``replacements`` by its bytes right after its first read."""
    pending, real = dict(replacements), Path.open

    def open_then_replace(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None) -> Any:
        if "r" not in mode or self not in pending:
            return real(self, mode, buffering, encoding, errors, newline)
        with real(self, "rb") as handle:
            data = handle.read()
        self.write_bytes(pending.pop(self))
        raw = io.BytesIO(data)
        return raw if "b" in mode else io.TextIOWrapper(raw, encoding=encoding, errors=errors, newline=newline)

    patch.setattr(Path, "open", open_then_replace)


def test_a_hashed_load_parses_and_hashes_one_read(tmp_path, monkeypatch):
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(b'{"value": 1}\r\n\n{"value": "na\xc3\xafve"}\n')
    before = rows.read_bytes()
    with monkeypatch.context() as patch:
        replaced_once_read(patch, {rows: b'{"value": 2}\n'})
        loaded, rows_sha256 = load_jsonl_hashed(rows)
    assert loaded == [{"value": 1}, {"value": "naïve"}] and rows_sha256 == hashlib.sha256(before).hexdigest()
    assert rows.read_bytes() == b'{"value": 2}\n'
    rows.write_bytes(before)
    assert jsonl_records_of(before, rows) == load_jsonl_records(rows) == loaded


def test_jsonl_writer_rejects_non_mapping_without_touching_destination(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("original\n", encoding="utf-8")

    with pytest.raises(ArtifactSchemaError, match="record 2"):
        write_jsonl_records(path, [{"valid": True}, "invalid"])

    assert path.read_text(encoding="utf-8") == "original\n"


def test_jsonl_writer_rejects_unserializable_without_touching_destination(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("original\n", encoding="utf-8")

    with pytest.raises(ArtifactSchemaError, match="record 1 is not JSON serializable"):
        write_jsonl_records(path, [{"bad": object()}])

    assert path.read_text(encoding="utf-8") == "original\n"


def test_jsonl_writer_cleans_temporary_file_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "bad.jsonl"
    path.write_text("original\n", encoding="utf-8")

    def fail_replace(self, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(type(path), "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_jsonl_records(path, [{"valid": True}])

    assert path.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"prompt": "a", "response": "b"}, ("a", "b")),
        ({"scenario": {"prompt": "a", "response": "b"}}, ("a", "b")),
        ({"scenario": "not a mapping"}, None),
        ({"prompt": "a"}, None),
        ({}, None),
        ({"prompt": 1, "response": "b"}, None),
    ],
)
def test_extract_prompt_response_handles_both_shapes(record, expected):
    assert extract_prompt_response(record) == expected


def test_agreeing_top_level_and_nested_pairs_are_accepted():
    record = {"prompt": "p", "response": "r", "scenario": {"prompt": "p", "response": "r"}}
    assert extract_prompt_response(record) == ("p", "r")


def test_conflicting_top_level_and_nested_pairs_are_rejected():
    record = {"prompt": "STALE", "response": "r", "scenario": {"prompt": "FRESH", "response": "r"}}
    with pytest.raises(ArtifactSchemaError, match="conflicting nested scenario"):
        extract_prompt_response(record)


def test_text_is_written_through_a_rename(tmp_path):
    path = tmp_path / "summary.json"
    write_text_atomically(path, '{"status": "complete"}\n')
    assert path.read_text() == '{"status": "complete"}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]


def test_an_existing_file_is_replaced_not_appended(tmp_path):
    path = tmp_path / "summary.json"
    write_text_atomically(path, "first")
    write_text_atomically(path, "second")
    assert path.read_text() == "second"


def test_a_missing_parent_directory_is_created(tmp_path):
    path = tmp_path / "run" / "nested" / "summary.json"
    write_text_atomically(path, "ok")
    assert path.read_text() == "ok"


def test_a_failed_write_leaves_no_partial_file_and_no_temporary(tmp_path, monkeypatch):
    path = tmp_path / "summary.json"
    write_text_atomically(path, "the good one")

    def _explode(self, target):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", _explode)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomically(path, "the bad one")

    assert path.read_text() == "the good one"
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]


def test_link_new_writes_a_file_once_whole_and_never_over_another(tmp_path):
    path = tmp_path / "record.json"
    link_new(path, b"first")
    assert path.read_bytes() == b"first"
    with pytest.raises(FileExistsError):
        link_new(path, b"second")
    assert path.read_bytes() == b"first"
    with pytest.raises(TypeError):
        link_new(tmp_path / "other.json", "not bytes")
    assert [p.name for p in tmp_path.iterdir()] == ["record.json"]


@pytest.mark.parametrize("mask, mode", [(0o022, 0o644), (0o002, 0o664), (0o077, 0o600)])
def test_the_atomic_writers_give_a_new_file_the_umask_s_mode_as_open_does(tmp_path, mask, mode):
    before = os.umask(mask)
    try:
        write_text_atomically(tmp_path / "text.json", "{}")
        link_new(tmp_path / "linked.json", b"{}")
        with open(tmp_path / "opened.json", "w") as handle:
            handle.write("{}")
    finally:
        os.umask(before)
    assert {path.name: path.stat().st_mode & 0o777 for path in tmp_path.iterdir()} == {
        "text.json": mode, "linked.json": mode, "opened.json": mode}


def test_a_stored_hash_wins_over_deriving_one():
    assert read_or_derive_prompt_response_hash(
        {"prompt": "p", "response": "r", "prompt_response_hash": "stored"}
    ) == "stored"


def test_a_hash_is_derived_from_top_level_prompt_and_response():
    assert read_or_derive_prompt_response_hash({"prompt": "p", "response": "r"}) == (
        compute_prompt_response_hash("p", "r")
    )


def test_a_blank_stored_hash_is_ignored():
    assert read_or_derive_prompt_response_hash(
        {"prompt": "p", "response": "r", "prompt_response_hash": "  "}
    ) == compute_prompt_response_hash("p", "r")


def test_none_is_returned_rather_than_raising():
    assert read_or_derive_prompt_response_hash({"prompt": "p"}) is None
    assert read_or_derive_prompt_response_hash({}) is None


def test_a_nested_scenario_envelope_does_not_hash():
    assert read_or_derive_prompt_response_hash(
        {"scenario": {"prompt": "p", "response": "r"}}
    ) is None


def test_a_padded_stored_hash_is_stripped():
    """A padded hash looks right in a log line and keys a different bucket."""
    digest = compute_prompt_response_hash("p", "r")
    assert read_or_derive_prompt_response_hash(
        {"prompt_response_hash": f"  {digest}\n"}
    ) == digest
