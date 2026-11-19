from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest
import zstandard

from ruleforge import hub
from ruleforge.experiments import table
from ruleforge.core.constants import PUBLISHED_COPY_FNAME, PUBLISHED_LEDGER_FNAME
from ruleforge.evaluate.ledger import LEDGER_FNAME
from ruleforge.core.record_names import IDENTITY_KEYS
from ruleforge.evaluate.identity import PublishedCopyError
from ruleforge.evaluate.aggregate import analyze_campaign, refuse_stripped
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.hub import (
    ARCHIVE,
    INDEX,
    SUMS,
    CompactRule,
    Entry,
    HubError,
    fetch,
    load_index,
    load_inventory,
    main_pack,
    members_digest,
    pack,
    read_sums,
)

REPO = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def _files(top: Path) -> dict[str, bytes]:
    return {path.relative_to(top).as_posix(): path.read_bytes() for path in sorted(top.rglob("*")) if path.is_file()}


def _ledger_text(campaign_id: str, condition_id: str, draw: int) -> str:
    """One draw's attempt ledger as the runner writes it: one scenario rejected once, then accepted."""
    from ruleforge.evaluate.ledger import AttemptRecord, ComponentRecord

    def row(attempt: int, valid: bool) -> str:
        full = ComponentRecord("full_ruleset", valid, None if valid else "invalid_label", "yes" if valid else None,
                               "stop", 5, True, "c" * 64)
        record = AttemptRecord(campaign_id, condition_id, "s1", draw, attempt, f"b{attempt}",
                               "accepted" if valid else "rejected", "local_deterministic", (full,),
                               f"2026-10-0{attempt + 1}T00:00:00Z", f"2026-10-0{attempt + 1}T00:00:01Z")
        return json.dumps(record.to_dict(), ensure_ascii=False) + "\n"

    return row(0, False) + row(1, True)


#: Each kept ledger of :data:`ARTIFACTS`: its campaign, condition and draw, and its root's profile.
LEDGERS = {"sampling/r1/m/forced-grpo/draw-000/attempts.jsonl": ("r1", "m/forced-grpo", 0, "ruleset-size"),
           "sampling/grpo-x/m/figure5/base/draw-001/attempts.jsonl": ("grpo-x", "m/figure5/base", 1, "training-gemma")}
ARTIFACTS = {
    "sampling/r1/aggregate/estimates.json": '{"rate": 0.5}',
    "sampling/r1/sampling_manifest.json": '{"campaign_id": "r1"}',
    "sampling/grpo-x/sampling_manifest.json": '{"campaign_id": "grpo-x"}',
    "sampling/r1/m/forced-grpo/draw-000/attempts.jsonl": _ledger_text("r1", "m/forced-grpo", 0),
    "sampling/r1/m/forced-grpo/draw-000/raw_outputs.jsonl": "dropped\n",
    "sampling/r1/m/naive-rules-above/draw-000/attempts.jsonl": "dropped attempts\n",
    "sampling/r1/m/naive-rules-above/draw-000/evaluations.jsonl": "dropped\n",
    "sampling/r1/m/naive-rules-above/draw-000/bundle_outcomes.jsonl": '{"scenario_id": "s1"}\n',
    "sampling/grpo-x/m/figure5/base/draw-000/evaluations.jsonl": '{"scenario_id": "s1", "scenario": {"source": "w"}}\n',
    "sampling/grpo-x/m/figure5/base/draw-001/evaluations.jsonl": "dropped\n",
    "sampling/grpo-x/m/figure5/base/draw-000/judge_outputs.jsonl": "dropped\n",
    "sampling/grpo-x/m/figure5/base/draw-001/attempts.jsonl": _ledger_text("grpo-x", "m/figure5/base", 1),
    "selection/run/merge_manifest.json": '{"weights": "abc"}',
    "anchors/run/confirm/condition.json": '{"k": 2}',
    "anchors/run/confirm/draw-000/raw_outputs.jsonl": "an input keeps everything\n",
}
KEPT = sorted(set(ARTIFACTS) - {"sampling/r1/m/forced-grpo/draw-000/raw_outputs.jsonl",
                                 "sampling/r1/m/naive-rules-above/draw-000/attempts.jsonl",
                                 "sampling/r1/m/naive-rules-above/draw-000/evaluations.jsonl",
                                 "sampling/grpo-x/m/figure5/base/draw-001/evaluations.jsonl",
                                 "sampling/grpo-x/m/figure5/base/draw-000/judge_outputs.jsonl"})
#: Each packed root's published-copy record, which a fetch lands beside its files.
MARKED = {f"sampling/{root}/{PUBLISHED_COPY_FNAME}": hub._published_copy(root, [
    relative.removeprefix(f"sampling/{root}/") for relative in LEDGERS if relative.startswith(f"sampling/{root}/")])
    for root in ("r1", "grpo-x")}
#: What a fetch of :data:`INVENTORY`'s package lands: the kept files, each kept ledger as its published form, the
#: roots' published-copy records.
FETCHED = sorted({relative.replace(LEDGER_FNAME, PUBLISHED_LEDGER_FNAME) if relative in LEDGERS else relative
                  for relative in KEPT} | set(MARKED))


def _fetched(artifacts: Path) -> dict[str, bytes]:
    """:data:`FETCHED` with its bytes: a kept ledger's are its strict replay's published form."""
    from ruleforge.evaluate.ledger import published_form_text

    out = dict(MARKED)
    for relative in KEPT:
        if relative in LEDGERS:
            campaign, condition, draw, profile = LEDGERS[relative]
            out[relative.replace(LEDGER_FNAME, PUBLISHED_LEDGER_FNAME)] = published_form_text(
                (artifacts / relative).parent, campaign_id=campaign, condition_id=condition, decoder_draw_id=draw,
                parts=hub.PROFILES[profile].ledger)
        else:
            out[relative] = (artifacts / relative).read_bytes()
    return out

INVENTORY = {
    "format": 1,
    "compact": {"attempts_conditions": ["forced*"]},
    "experiments": {
        "ruleset-size": {"profile": "ruleset-size", "roots": {"r1": None}},
        "training": {
            "profile": "training-gemma",
            "draw0_evaluations": True,
            "all_attempts": ["grpo-x"],
            "roots": {"grpo-x": None},
            "files": [
                {"path": "selection/merge_manifest.json", "kind": "selection",
                 "dest": "selection/run/merge_manifest.json",
                 "sha256": _sha(ARTIFACTS["selection/run/merge_manifest.json"].encode())},
                {"path": "inputs/anchors.tar.zst", "kind": "input", "format": ARCHIVE, "prefix": "confirm",
                 "dest": "anchors/run/confirm"},
            ],
        },
    },
}


@pytest.fixture
def artifacts(tmp_path) -> Path:
    root = tmp_path / "artifacts"
    for relative, text in ARTIFACTS.items():
        _write(root / relative, text)
    return root


@pytest.fixture
def repository(tmp_path, artifacts) -> Path:
    card = _write(tmp_path / "card.md", "# the card\n")
    pack(artifacts, INVENTORY, tmp_path / "repo", card=card, level=3)
    return tmp_path / "repo"


@pytest.fixture
def empty_root(tmp_path) -> Path:
    root = tmp_path / "fresh"
    root.mkdir()
    return root


def _rewrite_sums(repo: Path) -> None:
    listing = hub._listing(repo) - {SUMS}
    (repo / SUMS).write_text(hub.write_sums({path: hub.sha256_file(repo / path) for path in listing}))


def _set_index(repo: Path, edit) -> None:
    payload = json.loads((repo / INDEX).read_text())
    edit(payload)
    (repo / INDEX).write_text(json.dumps(payload))
    _rewrite_sums(repo)


def _leftovers(root: Path) -> list[str]:
    return sorted(path.name for path in root.rglob(".*fetching-*"))


def test_a_packed_root_fetches_back_byte_for_byte_under_the_compact_rule(repository, artifacts, empty_root):
    report = fetch(repository, empty_root, echo=lambda _: None)
    assert _files(empty_root) == _fetched(artifacts)
    assert sorted(report["placed"]) == ["anchors/run/confirm", "sampling/grpo-x", "sampling/r1",
                                        "selection/run/merge_manifest.json"]
    assert report["bytes_written"] == sum(map(len, _fetched(artifacts).values()))
    index = json.loads((repository / INDEX).read_text())
    assert index["format"] == 1 and index["documents"] == ["README.md"]
    assert "draw-000/evaluations.jsonl in training" in index["compact_rule"]
    assert "every attempts.jsonl of grpo-x" in index["compact_rule"]
    assert {entry["path"]: entry["kind"] for entry in index["files"]} == {
        "ruleset-size/roots/r1.tar.zst": "root", "training/roots/grpo-x.tar.zst": "root",
        "training/selection/merge_manifest.json": "selection", "training/inputs/anchors.tar.zst": "input"}
    assert _leftovers(empty_root) == []


def test_packing_is_deterministic(tmp_path, artifacts, repository):
    again = tmp_path / "again"
    pack(artifacts, INVENTORY, again, card=tmp_path / "card.md", level=3)
    assert _files(again) == _files(repository)


def test_a_root_already_there_is_refused_and_an_identical_file_is_left_as_it_is(repository, empty_root):
    fetch(repository, empty_root, only=["training"], echo=lambda _: None)
    with pytest.raises(HubError, match="sampling/grpo-x exists under .*: nothing is fetched over a root"):
        fetch(repository, empty_root, only=["training"], echo=lambda _: None)
    report = fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)
    assert report["placed"] == ["sampling/r1"]
    shutil.rmtree(empty_root / "sampling")
    shutil.rmtree(empty_root / "anchors")
    report = fetch(repository, empty_root, only=["training"], echo=lambda _: None)
    assert report["present"] == ["selection/run/merge_manifest.json"] and "sampling/grpo-x" in report["placed"]


def test_a_file_already_there_with_other_bytes_is_refused(repository, empty_root):
    _write(empty_root / "selection/run/merge_manifest.json", "other")
    with pytest.raises(HubError, match="exists"):
        fetch(repository, empty_root, only=["training"], echo=lambda _: None)
    assert not (empty_root / "sampling").exists()


def test_a_dry_run_lists_and_writes_nothing(repository, empty_root):
    lines: list[str] = []
    report = fetch(repository, empty_root, dry_run=True, echo=lines.append)
    assert report["dry_run"] and report["placed"] == [] and list(empty_root.iterdir()) == []
    assert lines[-1] == "dry run: 4 entries checked, nothing written" and any("sampling/r1" in line for line in lines)


def test_only_names_an_experiment_or_a_root_with_its_experiment_s_other_files(repository, empty_root):
    report = fetch(repository, empty_root, only=["grpo-x"], echo=lambda _: None)
    assert sorted(report["placed"]) == ["anchors/run/confirm", "sampling/grpo-x", "selection/run/merge_manifest.json"]
    with pytest.raises(HubError, match="--only nope: neither an experiment"):
        fetch(repository, empty_root, only=["nope"], echo=lambda _: None)


#: One experiment with two roots and an input archive both share.
TWO_ROOTS = {"format": 1, "experiments": {"training": {"profile": "training-gemma", "roots": {"r1": None, "grpo-x": None},
                                                    "files": [INVENTORY["experiments"]["training"]["files"][1]]}}}


@pytest.fixture
def two_roots(tmp_path, artifacts) -> Path:
    pack(artifacts, TWO_ROOTS, tmp_path / "two", level=1, echo=lambda _: None)
    return tmp_path / "two"


def test_two_roots_of_one_experiment_fetch_one_after_the_other(two_roots, empty_root):
    first = fetch(two_roots, empty_root, only=["r1"], echo=lambda _: None)
    assert sorted(first["placed"]) == ["anchors/run/confirm", "sampling/r1"]
    (empty_root / "anchors/run/confirm/empty").mkdir()
    lines: list[str] = []
    second = fetch(two_roots, empty_root, only=["grpo-x"], echo=lines.append)
    assert second["placed"] == ["sampling/grpo-x"] and second["present"] == ["anchors/run/confirm"]
    assert any(line.startswith("present  input") and "anchors/run/confirm" in line for line in lines)
    assert (empty_root / "anchors/run/confirm/condition.json").read_text() == '{"k": 2}'
    with pytest.raises(HubError, match="sampling/r1 exists under .*: nothing is fetched over a root"):
        fetch(two_roots, empty_root, only=["r1"], echo=lambda _: None)


def _special(directory: Path) -> None:
    os.mkfifo(directory / "fifo")


def _replaced_by(kind: str):
    def replace(directory: Path) -> None:
        shutil.rmtree(directory)
        if kind == "file":
            _write(directory, "a file")
        else:
            directory.symlink_to(directory.parent, target_is_directory=True)
    return replace


@pytest.mark.parametrize("tamper", [
    lambda directory: _write(directory / "condition.json", '{"k": 3}'),
    lambda directory: (directory / "condition.json").unlink(),
    lambda directory: _write(directory / "notes.txt", "mine"),
    lambda directory: _write(directory / "sub/notes.txt", "mine"),
    lambda directory: (directory / "link").symlink_to(directory / "condition.json"),
    _special, _replaced_by("file"), _replaced_by("symlink"),
], ids=["other-bytes", "a-member-missing", "a-file-more", "a-nested-file-more", "a-symlink", "a-special-file",
        "a-file", "a-symlink-to-a-directory"])
def test_an_input_already_there_that_is_not_its_archive_s_files_is_refused(two_roots, empty_root, tamper):
    fetch(two_roots, empty_root, only=["r1"], echo=lambda _: None)
    tamper(empty_root / "anchors/run/confirm")
    for dry_run in (True, False):
        with pytest.raises(HubError, match="anchors/run/confirm exists under .* is not this entry's members: nothing "
                                           "is fetched over it"):
            fetch(two_roots, empty_root, only=["grpo-x"], dry_run=dry_run, echo=lambda _: None)
    assert not (empty_root / "sampling/grpo-x").exists() and _leftovers(empty_root) == []


def test_an_input_already_there_is_checked_against_a_manifest_held_to_the_index(two_roots, empty_root):
    fetch(two_roots, empty_root, only=["r1"], echo=lambda _: None)
    _write(two_roots / "training/inputs/anchors.files.sha256", "")
    with pytest.raises(HubError, match="anchors.files.sha256: not the bytes SHA256SUMS lists"):
        fetch(two_roots, empty_root, only=["grpo-x"], echo=lambda _: None)


def test_a_file_with_other_bytes_than_listed_is_refused(repository, empty_root):
    archive = repository / "ruleset-size/roots/r1.tar.zst"
    archive.write_bytes(archive.read_bytes() + b"\0")
    with pytest.raises(HubError, match="r1.tar.zst: not the bytes SHA256SUMS lists"):
        fetch(repository, empty_root, echo=lambda _: None)
    assert not (empty_root / "sampling/r1").exists()


def test_a_members_manifest_with_other_bytes_than_listed_is_refused(repository, empty_root):
    _write(repository / "ruleset-size/roots/r1.files.sha256", "")
    with pytest.raises(HubError, match="r1.files.sha256: not the bytes"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)


def test_a_members_manifest_that_is_not_the_one_the_index_names_is_refused(repository, empty_root):
    _write(repository / "ruleset-size/roots/r1.files.sha256", f"{'0' * 64}  sampling/r1/x\n")

    def edit(payload):
        [entry] = [entry for entry in payload["files"] if entry["path"] == "ruleset-size/roots/r1.tar.zst"]
        entry["members_sha256"] = hub.sha256_file(repository / "ruleset-size/roots/r1.files.sha256")
    _set_index(repository, edit)
    with pytest.raises(HubError, match="not the members index.json names"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)


def test_a_manifest_naming_a_member_outside_the_prefix_is_refused(repository, empty_root):
    _write(repository / "ruleset-size/roots/r1.files.sha256", f"{'0' * 64}  sampling/other/x\n")

    def edit(payload):
        [entry] = [entry for entry in payload["files"] if entry["path"] == "ruleset-size/roots/r1.tar.zst"]
        entry["members_sha256"] = hub.sha256_file(repository / "ruleset-size/roots/r1.files.sha256")
        entry["members_digest"] = members_digest({"sampling/other/x": "0" * 64})
    _set_index(repository, edit)
    with pytest.raises(HubError, match="outside the archive's prefix"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)


def test_an_unlisted_or_missing_file_is_refused(repository, empty_root):
    _write(repository / "extra.txt", "x")
    with pytest.raises(HubError, match=r"files not in SHA256SUMS \['extra.txt'\]"):
        fetch(repository, empty_root, echo=lambda _: None)
    (repository / "extra.txt").unlink()
    (repository / "README.md").unlink()
    with pytest.raises(HubError, match=r"missing \['README.md'\]"):
        fetch(repository, empty_root, echo=lambda _: None)


def test_the_snapshot_s_own_files_are_not_the_repository_s(repository, empty_root):
    _write(repository / ".gitattributes", "*.zst filter=lfs\n")
    _write(repository / ".cache/huggingface/download/x.metadata", "x")
    fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)


def test_a_symlink_in_the_repository_is_refused(repository, empty_root):
    (repository / "link").symlink_to(repository / "README.md")
    with pytest.raises(HubError, match="link is a symlink"):
        fetch(repository, empty_root, echo=lambda _: None)


def test_an_index_that_is_not_the_one_sha256sums_lists_is_refused(repository, empty_root):
    (repository / INDEX).write_text((repository / INDEX).read_text() + " ")
    with pytest.raises(HubError, match="index.json is not the one SHA256SUMS lists"):
        fetch(repository, empty_root, echo=lambda _: None)


@pytest.mark.parametrize(("text", "message"), [
    ("nonsense\n", "SHA256SUMS:1: not a sha256sum line"),
    (f"{'a' * 64}  ../up\n", "not a relative path"),
    (f"{'a' * 64}  x\n{'b' * 64} *x\n", "x is listed twice"),
])
def test_a_malformed_listing_is_refused(text, message):
    with pytest.raises(HubError, match=message):
        read_sums(text)


def test_a_listing_reads_both_sha256sum_modes_and_drops_a_leading_dot():
    assert read_sums(f"{'a' * 64}  ./x/y\n\n{'b' * 64} *z\n") == {"x/y": "a" * 64, "z": "b" * 64}


@pytest.mark.parametrize("path", ["", "/abs", "a/../b", "a//b", "./a", "a\\b", "a\0b", ".", 7, None])
def test_an_unsafe_path_is_refused(path):
    with pytest.raises(HubError):
        hub.safe_path(path, "x")


def _index(**changes):
    entry = {"path": "g/f.json", "kind": "selection", "group": "g", "dest": "sel/f.json", "sha256": "a" * 64,
             "bytes": 3}
    archive = {"path": "g/roots/r.tar.zst", "kind": "root", "group": "g", "dest": "sampling/r", "sha256": "b" * 64,
               "bytes": 5, "format": ARCHIVE, "prefix": "sampling/r", "members": "g/roots/r.files.sha256",
               "members_sha256": "c" * 64, "members_digest": "d" * 64, "root": "r"}
    payload = {"format": 1, "files": [entry, archive], "documents": ["README.md"]}
    sums = {INDEX: "e" * 64, "README.md": "f" * 64, "g/f.json": "a" * 64, "g/roots/r.tar.zst": "b" * 64,
            "g/roots/r.files.sha256": "c" * 64}
    for key, value in changes.items():
        if key == "sums":
            sums = value(sums)
        elif key in ("index_format", "files", "documents"):
            payload[key.removeprefix("index_")] = value
        elif key.startswith("archive_"):
            archive[key.removeprefix("archive_")] = value
        else:
            entry[key] = value
    return payload, sums


def test_a_well_formed_index_loads():
    index = load_index(*_index())
    assert [entry.path for entry in index.entries] == ["g/f.json", "g/roots/r.tar.zst"]
    assert index.listed() == {INDEX, "README.md", "g/f.json", "g/roots/r.tar.zst", "g/roots/r.files.sha256"}
    assert index.entries[1].archive and index.entries[1].to_json()["root"] == "r"


def _drop(key):
    def edit(sums):
        return {path: value for path, value in sums.items() if path != key}
    return edit


@pytest.mark.parametrize(("changes", "message"), [
    ({"index_format": 2}, "not an index of format 1"),
    ({"files": {}}, "files and documents are lists"),
    ({"files": ["x"]}, r"files\[0\]: not an object"),
    ({"extra": 1}, "unknown keys"),
    ({"files": [{"path": "x"}]}, "missing"),
    ({"kind": "other"}, "kind 'other'"),
    ({"group": ""}, "no group"),
    ({"sha256": "A" * 64}, "no sha256"),
    ({"bytes": True}, "bytes must be a count"),
    ({"bytes": -1}, "bytes must be a count"),
    ({"format": "zip"}, "format 'zip'"),
    ({"archive_prefix": None}, "an archive names its prefix"),
    ({"archive_members_digest": "x"}, "SHA-256 digests"),
    ({"prefix": "p"}, "a plain file has no prefix"),
    ({"omit": ["a/**"]}, "a plain file has no prefix, members or omit list"),
    ({"archive_omit": "a/**"}, "omit is a list of globs"),
    ({"archive_omit": ["a/**", "/b"]}, "not a relative path"),
    ({"archive_dest": "sampling/s"}, "a root is an archive of sampling"),
    ({"root": "r"}, "only a root names a root"),
    ({"path": "g/roots/r.tar.zst"}, "listed more than once"),
    ({"path": "README.md", "sums": _drop("g/f.json")}, "listed more than once"),
    ({"dest": "sampling/r/inner"}, "overlap"),
    ({"sums": lambda sums: {**sums, SUMS: "0" * 64}}, "SHA256SUMS lists itself"),
    ({"sums": _drop("README.md")}, "disagree"),
    ({"sums": lambda sums: {**sums, "g/f.json": "0" * 64}}, "g/f.json: index.json and SHA256SUMS give two"),
    ({"sums": lambda sums: {**sums, "g/roots/r.files.sha256": "0" * 64}}, "r.files.sha256: index.json and"),
])
def test_a_malformed_index_is_refused(changes, message):
    with pytest.raises(HubError, match=message):
        load_index(*_index(**changes))


def test_a_destination_inside_another_is_refused_whatever_sorts_between_them():
    """``sampling/r-x`` sorts between ``sampling/r`` and ``sampling/r/f.json`` ('-' before '/')."""
    payload, sums = _index(dest="sampling/r/f.json")
    sibling = {**payload["files"][1], "path": "g/roots/r-x.tar.zst", "members": "g/roots/r-x.files.sha256",
               "dest": "sampling/r-x", "prefix": "sampling/r-x", "root": "r-x"}
    payload["files"].append(sibling)
    sums = {**sums, "g/roots/r-x.tar.zst": "b" * 64, "g/roots/r-x.files.sha256": "c" * 64}
    with pytest.raises(HubError, match=r"\[\('sampling/r', 'sampling/r/f.json'\)\] overlap"):
        load_index(payload, sums)
    assert hub.overlapping(["a", "a-b", "a/b/c", "a/b", "d", "d"]) == [
        ("a", "a/b"), ("a", "a/b/c"), ("a/b", "a/b/c"), ("d", "d")]
    assert hub.overlapping(["a", "a-b", "ab"]) == []


def test_a_dest_and_a_document_are_paths_inside_the_repository():
    with pytest.raises(HubError, match="dest"):
        load_index(*_index(dest="/abs"))
    with pytest.raises(HubError, match="documents"):
        load_index(*_index(documents=["../card"]))


def test_a_root_that_is_not_a_directory_is_refused(repository, tmp_path):
    with pytest.raises(HubError, match="is not a directory"):
        fetch(repository, tmp_path / "missing", echo=lambda _: None)


def test_a_destination_under_a_symlink_or_a_file_is_refused(repository, empty_root, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (empty_root / "sampling").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(HubError, match="sampling/r1: sampling is a symlink or not a directory"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)
    (empty_root / "sampling").unlink()
    _write(empty_root / "sampling", "a file")
    with pytest.raises(HubError, match="symlink or not a directory"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)
    assert list(elsewhere.iterdir()) == []


def test_a_destination_that_is_a_symlink_is_refused(repository, empty_root, tmp_path):
    target = _write(tmp_path / "target.json", ARTIFACTS["selection/run/merge_manifest.json"])
    (empty_root / "selection/run").mkdir(parents=True)
    (empty_root / "selection/run/merge_manifest.json").symlink_to(target)
    with pytest.raises(HubError, match="exists"):
        fetch(repository, empty_root, only=["training"], echo=lambda _: None)


def test_an_ancestor_turned_symlink_after_the_plan_is_refused_at_placement(repository, empty_root, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def swap(line: str) -> None:
        if not (empty_root / "sampling").is_symlink():
            (empty_root / "sampling").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(HubError, match="sampling/r1: sampling is a symlink or not a directory"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=swap)
    assert list(elsewhere.iterdir()) == [] and _leftovers(empty_root) == []


@pytest.mark.parametrize(("only", "dest"), [(["ruleset-size"], "sampling/r1"),
                                            (["training"], "selection/run/merge_manifest.json")])
def test_a_destination_made_after_the_plan_is_refused_at_placement(repository, empty_root, only, dest):
    def make(line: str) -> None:
        if not os.path.lexists(empty_root / dest):
            (empty_root / dest).mkdir(parents=True)
    with pytest.raises(HubError, match=f"{dest}: appeared while it was fetched"):
        fetch(repository, empty_root, only=only, echo=make)
    assert list((empty_root / dest).iterdir()) == [] and _leftovers(empty_root) == []


def test_placed_files_and_roots_take_the_umask_not_the_staging_mode(repository, empty_root):
    mask = os.umask(0o027)
    try:
        fetch(repository, empty_root, echo=lambda _: None)
    finally:
        os.umask(mask)
    assert (empty_root / "selection/run/merge_manifest.json").stat().st_mode & 0o777 == 0o640
    assert (empty_root / "sampling/r1").stat().st_mode & 0o777 == 0o750


def test_a_file_that_appears_while_it_is_copied_is_refused(repository, empty_root, monkeypatch):
    real = hub._copy_hashing

    def racing(source, handle):
        result = real(source, handle)
        _write(empty_root / "selection/run/merge_manifest.json", "raced")
        return result
    monkeypatch.setattr(hub, "_copy_hashing", racing)
    with pytest.raises(HubError, match="appeared while it was copied"):
        hub.place_file(repository / "training/selection/merge_manifest.json", empty_root,
                       hub.Entry("training/selection/merge_manifest.json", "selection", "training",
                                 "selection/run/merge_manifest.json",
                                 _sha(ARTIFACTS["selection/run/merge_manifest.json"].encode()), 18))
    assert (empty_root / "selection/run/merge_manifest.json").read_text() == "raced"
    assert _leftovers(empty_root) == []
    (empty_root / "selection/run/merge_manifest.json").unlink()
    monkeypatch.setattr(hub, "_still_absent", lambda *args: None)
    with pytest.raises(HubError, match="appeared while it was copied"):
        hub.place_file(repository / "training/selection/merge_manifest.json", empty_root,
                       hub.Entry("training/selection/merge_manifest.json", "selection", "training",
                                 "selection/run/merge_manifest.json",
                                 _sha(ARTIFACTS["selection/run/merge_manifest.json"].encode()), 18))
    assert (empty_root / "selection/run/merge_manifest.json").read_text() == "raced"
    assert _leftovers(empty_root) == []


@pytest.mark.parametrize("only", [["ruleset-size"], ["training"]])
def test_a_guard_refusing_before_the_move_leaves_nothing_of_the_entry(repository, empty_root, only):
    seen: list[tuple[Path, str]] = []

    def guard(place: Path, what: str) -> None:
        seen.append((place, what))
        if len(seen) == 2:
            raise HubError("refused by the guard")
    with pytest.raises(HubError, match="refused by the guard"):
        fetch(repository, empty_root, only=only, guard=guard, echo=lambda _: None)
    [entry] = [entry for entry in load_index(json.loads((repository / INDEX).read_text()),
                                             hub.read_sums((repository / SUMS).read_text())).entries
               if entry.path == {"ruleset-size": "ruleset-size/roots/r1.tar.zst", "training": "training/inputs/anchors.tar.zst"}[only[0]]]
    assert seen == [(empty_root / entry.dest, f"the destination of {entry.path}")] * 2
    assert not (empty_root / entry.dest).exists() and _leftovers(empty_root) == []


def test_a_guard_refusing_before_a_file_is_linked_leaves_nothing(repository, empty_root):
    entry = hub.Entry("training/selection/merge_manifest.json", "selection", "training", "selection/run/merge_manifest.json",
                      _sha(ARTIFACTS["selection/run/merge_manifest.json"].encode()), 18)
    seen: list[Path] = []

    def guard(place: Path, what: str) -> None:
        seen.append(place)
        if len(seen) == 2:
            raise HubError("refused by the guard")
    with pytest.raises(HubError, match="refused by the guard"):
        hub.place_file(repository / entry.path, empty_root, entry, guard)
    assert seen == [empty_root / entry.dest] * 2
    assert not (empty_root / entry.dest).exists() and _leftovers(empty_root) == []


def test_a_file_that_changes_while_it_is_copied_is_refused(repository, empty_root):
    entry = hub.Entry("training/selection/merge_manifest.json", "selection", "training", "selection/run/merge_manifest.json",
                      "0" * 64, 18)
    with pytest.raises(HubError, match="changed while it was copied"):
        hub.place_file(repository / entry.path, empty_root, entry)
    assert not (empty_root / entry.dest).exists() and _leftovers(empty_root) == []


def _tarinfo(name: str, data: bytes = b"", kind: bytes = tarfile.REGTYPE, link: str = "") -> tuple:
    info = tarfile.TarInfo(name)
    info.type, info.size, info.linkname, info.mtime = kind, len(data) if kind == tarfile.REGTYPE else 0, link, 1
    return info, data


def _tar_zst(archive: Path, members: list[tuple]) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if info.type == tarfile.REGTYPE else None)
    archive.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))
    return archive


def _crafted(tmp_path: Path, members: list[tuple], manifest: dict[str, str] | None = None) -> Path:
    repo = tmp_path / "crafted"
    _tar_zst(repo / "x/roots/r.tar.zst", members)
    if manifest is None:
        manifest = {info.name: _sha(data) for info, data in members if info.type == tarfile.REGTYPE}
    (repo / "x/roots/r.files.sha256").write_text(hub.write_sums(manifest))
    built = hub._Built(repo)
    built.add(hub._archive_entry(repo, "x/roots/r.tar.zst", "root", "x", "sampling/r", "sampling/r", manifest, "r"))
    built.finish("rule", None)
    return repo


GOOD_MEMBER = _tarinfo("sampling/r/a.json", b"{}")
#: Archives no fetch extracts and no staging accepts, whatever their manifest: (members, manifest, message).
UNSAFE = [
    ([_tarinfo("/sampling/r/abs", b"x")], {}, "not a relative path"),
    ([_tarinfo("sampling/r/../../up", b"x")], {}, "not a relative path"),
    ([_tarinfo("sampling/other/a", b"x")], {}, "outside the archive's prefix"),
    ([GOOD_MEMBER, _tarinfo("sampling/r/link", kind=tarfile.SYMTYPE, link="/etc/passwd")], None,
     "not a regular file or a directory"),
    ([GOOD_MEMBER, _tarinfo("sampling/r/hard", kind=tarfile.LNKTYPE, link="sampling/r/a.json")], None,
     "not a regular file or a directory"),
    ([GOOD_MEMBER, _tarinfo("sampling/r/fifo", kind=tarfile.FIFOTYPE)], None, "not a regular file or a directory"),
    ([GOOD_MEMBER, _tarinfo("sampling/r/dev", kind=tarfile.CHRTYPE)], None, "not a regular file or a directory"),
    ([GOOD_MEMBER, GOOD_MEMBER], {"sampling/r/a.json": _sha(b"{}")}, "appears twice"),
    ([GOOD_MEMBER, _tarinfo("sampling/r/a.json", kind=tarfile.DIRTYPE)], None, "appears twice"),
    ([_tarinfo("sampling/r/a", b"x"), _tarinfo("sampling/r/a/b", b"y")], None, "collides with another member"),
    ([_tarinfo("sampling/r/a/b", b"y"), _tarinfo("sampling/r/a", b"x")], None, "collides with another member"),
    ([_tarinfo("sampling/r/a", b"x"), _tarinfo("sampling/r/a/d", kind=tarfile.DIRTYPE)], None,
     "collides with another member"),
    ([_tarinfo("sampling/r", b"x")], None, "collides with another member"),
]


@pytest.mark.parametrize(("members", "manifest", "message"), UNSAFE + [
    ([GOOD_MEMBER], {"sampling/r/a.json": _sha(b"{}"), "sampling/r/b.json": "0" * 64}, r"missing \['sampling/r/b.json'\]"),
    ([GOOD_MEMBER, _tarinfo("sampling/r/extra", b"x")], {"sampling/r/a.json": _sha(b"{}")},
     r"not listed \['sampling/r/extra'\]"),
    ([GOOD_MEMBER], {"sampling/r/a.json": "0" * 64}, r"other bytes \['sampling/r/a.json'\]"),
])
def test_an_archive_that_is_not_its_manifest_or_not_safe_is_refused_and_leaves_nothing(tmp_path, empty_root, members,
                                                                                      manifest, message):
    repo = _crafted(tmp_path, members, manifest)
    with pytest.raises(HubError, match=message):
        fetch(repo, empty_root, echo=lambda _: None)
    assert not (empty_root / "sampling/r").exists() and _leftovers(empty_root) == []
    assert not (tmp_path / "etc").exists() and not (empty_root.parent / "up").exists()


@pytest.mark.parametrize(("members", "manifest", "message"), UNSAFE)
def test_reading_an_archive_through_refuses_what_a_fetch_refuses(tmp_path, members, manifest, message):
    archive = _tar_zst(tmp_path / "r.tar.zst", members)
    with pytest.raises(HubError, match=message):
        hub._archive_members(archive, "sampling/r", "r.tar.zst")


def test_directories_are_extracted_and_the_prefix_s_own_entry_is_accepted(tmp_path, empty_root):
    members = [_tarinfo("sampling/r", kind=tarfile.DIRTYPE), _tarinfo("sampling/r/d", kind=tarfile.DIRTYPE),
               _tarinfo("sampling/r/d/e.json", b"[]"), _tarinfo("sampling/r/empty", kind=tarfile.DIRTYPE)]
    fetch(_crafted(tmp_path, members), empty_root, echo=lambda _: None)
    assert (empty_root / "sampling/r/d/e.json").read_bytes() == b"[]" and (empty_root / "sampling/r/empty").is_dir()


def test_a_root_that_appears_while_it_is_extracted_is_refused(tmp_path, empty_root, monkeypatch):
    real = hub._copy_hashing

    def racing(source, handle):
        (empty_root / "sampling/r").mkdir(parents=True, exist_ok=True)
        return real(source, handle)
    monkeypatch.setattr(hub, "_copy_hashing", racing)
    with pytest.raises(HubError, match="appeared while it was extracted"):
        fetch(_crafted(tmp_path, [GOOD_MEMBER]), empty_root, echo=lambda _: None)
    assert _leftovers(empty_root) == []


@pytest.mark.parametrize(("relative", "kept"), [
    ("aggregate/estimates.json", True), ("m/forced-grpo/draw-000/attempts.jsonl", True),
    ("m/forced/draw-003/attempts.jsonl", True), ("m/naive-rules-above/draw-000/attempts.jsonl", False),
    ("attempts.jsonl", False), ("m/attempts.jsonl", False), ("m/forced-grpo/draw-000/raw_outputs.jsonl", False),
    ("m/x/draw-000/judge_outputs.jsonl", False), ("m/x/draw-000/smrc_per_scenario.jsonl", False),
    ("m/x/draw-000/evaluations.jsonl", False), ("m/x/draw-000/bundle_outcomes.jsonl", True),
    # the run records: Hydra's run directories (logs, submitit records, configs) and the code snapshots
    ("hydra/j/s/run_campaign.log", False), ("hydra/j/.submitit/7/7_0_log.out", False),
    ("hydra/j/s/.hydra/config.yaml", False), ("hydra/preflight/c/.hydra/overrides.yaml", False),
    ("code_snapshots/0123456789abcdef.tar.gz", False), ("m/x/run.log", True), ("m/hydra/x.json", True),
])
def test_the_compact_rule(relative, kept):
    assert CompactRule().keeps(relative) is kept
    assert CompactRule(all_attempts=True).keeps(relative) is (kept or relative.endswith("attempts.jsonl"))
    if relative.endswith("draw-000/evaluations.jsonl"):
        assert CompactRule(draw0_evaluations=True).keeps(relative)
    assert not CompactRule(draw0_evaluations=True).keeps("evaluations.jsonl")


STYLES = sorted(IDENTITY_KEYS.items())


@pytest.mark.parametrize(("public", "recorded"), STYLES)
def test_the_compact_rule_reads_a_recorded_condition_directory_by_its_public_name(public, recorded):
    for relative in ("m/{}/draw-000/attempts.jsonl", "m/{}/draw-003/raw_outputs.jsonl", "m/{}/condition.json",
                     "hydra/{}/x/run_campaign.log", "m/{}/draw-000/bundle_outcomes.jsonl"):
        assert CompactRule().keeps(relative.format(recorded)) is CompactRule().keeps(relative.format(public))
    assert CompactRule().keeps(f"m/{recorded}/draw-000/attempts.jsonl") is public.startswith("forced")


#: The files of each condition directory and Hydra tree of :func:`_styled_root`'s roots, ``{}`` the directory.
STYLED = ("m/{}/condition.json", "m/{}/draw-000/attempts.jsonl", "m/{}/draw-000/raw_outputs.jsonl",
          "m/{}/draw-000/bundle_outcomes.jsonl", "hydra/m/{}/x/.hydra/config.yaml", "hydra/m/{}/x/run_campaign.log")
#: An omit list naming public directories: a style's condition directories and Hydra trees, another's directories.
STYLED_OMIT = ["m/naive/**", "hydra/m/naive/**", "m/naive-rules-above/**"]


def _styled_root(top: Path, name) -> Path:
    """A campaign root with :data:`STYLED`'s files for each style, its directories named ``name(public style)``."""
    _write(top / "sampling_manifest.json", json.dumps({"campaign_id": top.name}))
    for public, _ in STYLES:
        for template in STYLED:
            relative = template.format(name(public))
            _write(top / relative, _ledger_text(top.name, f"m/{name(public)}", 0) if relative.endswith(LEDGER_FNAME)
                   else json.dumps({"scenario_id": f"{public} {Path(template).name}"}) + "\n")
    return top


def _styled_kept(public: str, template: str) -> bool:
    omitted = public == "naive" or (public == "naive-rules-above" and template.startswith("m/"))
    return not (omitted or template.startswith("hydra/") or template.endswith("raw_outputs.jsonl")
                or (template.endswith("attempts.jsonl") and not public.startswith("forced")))


def test_packing_keeps_the_same_files_of_a_recorded_root_and_a_public_one(tmp_path):
    artifacts = tmp_path / "artifacts"
    names = {"recorded": IDENTITY_KEYS.__getitem__, "public": lambda public: public}
    for root, name in names.items():
        _styled_root(artifacts / "sampling" / root, name)
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": dict.fromkeys(names),
                                                    "omit": dict.fromkeys(names, STYLED_OMIT)}}}
    lines: list[str] = []
    pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lines.append)
    for root, name in names.items():
        members = hub.read_sums((tmp_path / f"repo/x/roots/{root}.files.sha256").read_text())
        assert set(members) == {f"sampling/{root}/{template.format(name(public))}".replace(
            LEDGER_FNAME, PUBLISHED_LEDGER_FNAME) for public, _ in STYLES for template in STYLED
            if _styled_kept(public, template)} | {f"sampling/{root}/{name}" for name in (PUBLISHED_COPY_FNAME,
                                                                                     "sampling_manifest.json")}
    assert lines == [f"{root}: 10 file(s) omitted (m/naive/**: 4, hydra/m/naive/**: 2, m/naive-rules-above/**: 4)"
                     for root in names]


def test_packing_leaves_out_symlinks_and_lists_them_and_refuses_a_special_file_and_a_missing_tree(tmp_path,
                                                                                                    artifacts):
    for number in range(4):
        (artifacts / f"sampling/r1/hydra/.submitit/link{number}.sh").parent.mkdir(parents=True, exist_ok=True)
        (artifacts / f"sampling/r1/hydra/.submitit/link{number}.sh").symlink_to(artifacts / "sampling/r1/aggregate")
    lines: list[str] = []
    index = pack(artifacts, INVENTORY, tmp_path / "out", level=1, echo=lines.append)
    assert lines == ["r1: 4 symlink(s) left out (hydra/.submitit/link0.sh, hydra/.submitit/link1.sh, "
                     "hydra/.submitit/link2.sh, ...)"]
    [r1] = [entry for entry in index.entries if entry.root == "r1"]
    assert all("link" not in name for name in read_sums((tmp_path / "out" / r1.members).read_text()))
    assert hub.pack_tree(artifacts / "sampling/r1", "p", tmp_path / "b.tar.zst", tmp_path / "b.sha256", level=1)
    os.mkfifo(artifacts / "sampling/r1/fifo")
    with pytest.raises(HubError, match="not a regular file or a directory"):
        pack(artifacts, INVENTORY, tmp_path / "again", level=1)
    with pytest.raises(HubError, match="is not a directory"):
        hub.pack_tree(tmp_path / "missing", "p", tmp_path / "a.tar.zst", tmp_path / "a.sha256")


def test_packing_refuses_a_file_that_changed_while_it_was_packed_and_leaves_no_archive(tmp_path, artifacts,
                                                                                      monkeypatch):
    real = hub.sha256_file
    monkeypatch.setattr(hub, "sha256_file", lambda path: "0" * 64 if path.name == "estimates.json" else real(path))
    archive, manifest = tmp_path / "a.tar.zst", tmp_path / "a.sha256"
    with pytest.raises(HubError, match="a file changed while it was packed"):
        hub.pack_tree(artifacts / "sampling/r1", "p", archive, manifest, level=1)
    assert not archive.exists() and not manifest.exists()
    monkeypatch.setattr(hub, "sha256_file", real)
    members = hub.pack_tree(artifacts / "sampling/r1", "p", archive, manifest, level=1)
    assert members == hub._archive_members(archive, "p", "a") == read_sums(manifest.read_text())
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        hub.pack_tree(artifacts / "sampling/r1", "p", archive, manifest, level=1)
    assert archive.read_bytes() == before and manifest.exists()


def test_packing_refuses_a_file_that_shrank_and_a_name_its_manifest_cannot_hold(tmp_path, artifacts, monkeypatch):
    real = hub._tree_files

    def grown(top, symlinks):
        return [(name, os.stat_result((*info[:6], info.st_size + 5, *info[7:]))) for name, info in real(top, symlinks)]

    archive, manifest = tmp_path / "a.tar.zst", tmp_path / "a.sha256"
    monkeypatch.setattr(hub, "_tree_files", grown)
    with pytest.raises(OSError):
        hub.pack_tree(artifacts / "sampling/r1", "p", archive, manifest, level=1)
    assert not archive.exists() and not manifest.exists()
    monkeypatch.setattr(hub, "_tree_files", real)
    monkeypatch.setattr(hub, "read_sums", lambda text, what=hub.SUMS: {})
    with pytest.raises(HubError, match="a member name its manifest cannot hold"):
        hub.pack_tree(artifacts / "sampling/r1", "p", archive, manifest, level=1)
    assert not archive.exists() and not manifest.exists()
    manifest.write_text("kept\n")
    with pytest.raises(HubError, match="a member name its manifest cannot hold"):
        hub.pack_tree(artifacts / "sampling/r1", "p", archive, manifest, level=1)
    assert not archive.exists() and manifest.read_text() == "kept\n"


def _pinned_inventory(repository: Path, **wrong: str) -> dict:
    inventory = json.loads(json.dumps(INVENTORY))
    recorded = {entry["root"]: entry["members_digest"]
                for entry in json.loads((repository / INDEX).read_text())["files"] if entry["kind"] == "root"}
    for experiment in inventory["experiments"].values():
        experiment["roots"] = {root: wrong.get(root, recorded[root]) for root in experiment["roots"]}
    return inventory


def test_packing_holds_each_root_to_its_pinned_members(tmp_path, artifacts, repository):
    pack(artifacts, _pinned_inventory(repository), tmp_path / "pinned", level=1, echo=lambda _: None)
    with pytest.raises(HubError, match="r1.tar.zst: .* is not the 000000000000... the inventory pins"):
        pack(artifacts, _pinned_inventory(repository, r1="0" * 64), tmp_path / "wrong", level=1)


def test_packing_refuses_an_input_that_is_not_the_one_the_inventory_pins(tmp_path, artifacts):
    _write(artifacts / "selection/run/merge_manifest.json", "edited")
    with pytest.raises(HubError, match="is not the .* the inventory pins"):
        pack(artifacts, INVENTORY, tmp_path / "out", level=1)


def test_packing_only_some_experiments_or_roots(tmp_path, artifacts):
    index = pack(artifacts, INVENTORY, tmp_path / "out", only=["ruleset-size"], level=1)
    assert [entry.path for entry in index.entries] == ["ruleset-size/roots/r1.tar.zst"] and index.documents == ()
    index = pack(artifacts, INVENTORY, tmp_path / "root", only=["grpo-x"], level=1)
    assert [entry.path for entry in index.entries] == ["training/roots/grpo-x.tar.zst"]
    with pytest.raises(HubError, match="--only nope: neither an experiment nor a root"):
        pack(artifacts, INVENTORY, tmp_path / "nope", only=["nope", "ruleset-size"], level=1)
    assert not (tmp_path / "nope").exists()


@pytest.fixture
def packed(tmp_path, repository) -> Path:
    """A packing output laid out as a release's: flat roots/, sources, an input archive, a file and an adapter."""
    out = tmp_path / "packed"
    for path in (repository / "ruleset-size/roots").iterdir():
        _write(out / "roots" / path.name, path.read_bytes())
    for path in (repository / "training/roots").iterdir():
        _write(out / "roots" / path.name, path.read_bytes())
    _write(out / "selection/run/merge_manifest.json", ARTIFACTS["selection/run/merge_manifest.json"])
    _write(out / "inputs/anchors.tar.zst", (repository / "training/inputs/anchors.tar.zst").read_bytes())
    _write(tmp_path / "rows/manifest.json", "{}")
    _write(out / "models/adapter/adapter_config.json", '{"r": 8}')
    return out


def _repacked(artifacts: Path, packed: Path, manifest: Path) -> None:
    (packed / "roots/r1.tar.zst").unlink()
    manifest.unlink(missing_ok=True)
    hub.pack_tree(artifacts / "sampling/r1", "sampling/r1", packed / "roots/r1.tar.zst", manifest, level=1)


OMITTED = {
    "sampling/r1/m/arm-x/condition.json": "{}",
    "sampling/r1/m/arm-x/draw-000/bundle_outcomes.jsonl": "left out\n",
    "sampling/r1/hydra/m/arm-x/2026-01-01/.hydra/config.yaml": "left out\n",
    "anchors/run/confirm/trainer.log": "a run log\n",
    "anchors/run/confirm/.submitit/1/1_0_log.out": "a submitit log\n",
}


def _omitting(**omit) -> dict:
    """INVENTORY with the omit lists ``r1`` (of r1) and ``anchors`` (of the input archive)."""
    inventory = json.loads(json.dumps(INVENTORY))
    inventory["experiments"]["ruleset-size"]["omit"] = {"r1": omit.get("r1", ["m/arm-x/**", "hydra/m/arm-x/**"])}
    inventory["experiments"]["training"]["files"][1]["omit"] = omit.get("anchors", ["**/trainer.log", "**/.submitit/**"])
    return inventory


@pytest.fixture
def omitted(tmp_path, artifacts) -> tuple[Path, list[str]]:
    """The artifact root with :data:`OMITTED` added, packed under :func:`_omitting`."""
    for relative, text in OMITTED.items():
        _write(artifacts / relative, text)
    lines: list[str] = []
    pack(artifacts, _omitting(), tmp_path / "omitted", level=1, echo=lines.append)
    return tmp_path / "omitted", lines


def test_an_omit_list_leaves_its_files_out_of_the_archive_and_the_index_declares_it(omitted, artifacts, empty_root):
    repository, lines = omitted
    assert lines == ["r1: 3 file(s) omitted (m/arm-x/**: 2, hydra/m/arm-x/**: 1)",
                     "anchors/run/confirm: 2 file(s) omitted (**/trainer.log: 1, **/.submitit/**: 1)"]
    declared = {entry["path"]: entry.get("omit") for entry in json.loads((repository / INDEX).read_text())["files"]}
    assert declared == {"ruleset-size/roots/r1.tar.zst": ["m/arm-x/**", "hydra/m/arm-x/**"],
                        "training/roots/grpo-x.tar.zst": None, "training/selection/merge_manifest.json": None,
                        "training/inputs/anchors.tar.zst": ["**/trainer.log", "**/.submitit/**"]}
    fetch(repository, empty_root, echo=lambda _: None)
    assert _files(empty_root) == _fetched(artifacts)
    assert load_index(json.loads((repository / INDEX).read_text()),
                      read_sums((repository / SUMS).read_text())).entries[0].omit == ("m/arm-x/**",
                                                                                     "hydra/m/arm-x/**")


@pytest.mark.parametrize(("omit", "unused"), [({"r1": ["m/arm-x/**", "m/absent/**"]}, "r1: .*'m/absent/\\*\\*'"),
                                               ({"anchors": ["**/trainer.log", "**/*.err"]},
                                                "anchors/run/confirm: .*'\\*\\*/\\*.err'")])
def test_an_omit_glob_that_leaves_out_nothing_is_refused_and_nothing_is_left(tmp_path, artifacts, omit, unused):
    for relative, text in OMITTED.items():
        _write(artifacts / relative, text)
    with pytest.raises(HubError, match=f"{unused}\\] leave out no file: it names what the tree does not hold"):
        pack(artifacts, _omitting(**omit), tmp_path / "out", level=1, echo=lambda _: None)
    assert not (tmp_path / "out").exists()


def test_an_omit_list_counts_each_file_every_glob_matches():
    omit = hub.Omit(["m/a/**", "**/x.log"])
    assert [omit.leaves_out(path) for path in ("m/a/x.log", "m/a/b", "m/ab/c", "x.log", "m/b/y.log")] == [
        True, True, False, True, False]
    assert omit.counts == {"m/a/**": 2, "**/x.log": 2}
    assert omit.keeping(lambda relative: relative != "kept-not")("m/b/c") and not omit.keeping(bool)("m/a/c")
    lines: list[str] = []
    hub.Omit().check("t", lines.append)
    assert lines == []


def test_a_fetched_root_without_a_condition_its_estimates_name_is_never_aggregated(tmp_path, artifacts, empty_root):
    """An omit list ships r1 without the arm-x condition while its estimates, the whole campaign's, still name it."""
    for relative, text in OMITTED.items():
        _write(artifacts / relative, text)
    _write(artifacts / "sampling/r1/m/naive-rules-above/condition.json", "{}")
    _write(artifacts / "sampling/r1/aggregate/estimates.json",
           json.dumps({"conditions": {"m/naive-rules-above": {}, "m/arm-x": {}}, "excluded_conditions": {}}))
    refuse_stripped(artifacts / "sampling/r1")
    pack(artifacts, _omitting(), tmp_path / "repo", level=1, echo=lambda _: None)
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    root = empty_root / "sampling/r1"
    assert (root / "m/naive-rules-above/condition.json").is_file() and not (root / "m/arm-x").exists()
    before = _files(empty_root)
    for aggregate in (refuse_stripped, analyze_campaign):
        with pytest.raises(PublishedCopyError, match=r"sampling/r1 is a published copy of a campaign root "
                                                     r"\(published_copy.json\).*not each draw's completion record"):
            aggregate(root)
    assert _files(empty_root) == before
    (root / PUBLISHED_COPY_FNAME).unlink()  # a copy that does not say it is one is refused for what it lacks
    for aggregate in (refuse_stripped, analyze_campaign):
        with pytest.raises(CampaignRunError, match=r"does not hold \['m/arm-x'\], which its aggregate/estimates.json "
                                                   "names"):
            aggregate(root)


@pytest.fixture
def hub_snapshot(monkeypatch, repository):
    """``snapshot_download`` served from the local repository: each call's files and the snapshot's metadata."""
    calls = []

    def snapshot_download(*, repo_id, repo_type, revision, local_dir, allow_patterns, token):
        calls.append((repo_id, repo_type, revision, sorted(allow_patterns), token))
        for pattern in allow_patterns:
            _write(Path(local_dir) / pattern, (repository / pattern).read_bytes())
        _write(Path(local_dir) / ".cache/huggingface/x.metadata", "x")
    def list_repo_files(repo_id, *, repo_type, revision, token):
        calls.append(("list", repo_id, repo_type, revision, token))
        return sorted(hub._listing(repository) | {".gitattributes"})
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", list_repo_files)
    return calls


def test_a_hub_repository_is_downloaded_file_by_file_and_the_download_removed(hub_snapshot, empty_root, tmp_path):
    report = fetch("org/data", empty_root, only=["ruleset-size"], revision="v1", download_dir=tmp_path / "dl",
                   echo=lambda _: None)
    assert report["placed"] == ["sampling/r1"]
    assert hub_snapshot == [("org/data", "dataset", "v1", [SUMS, INDEX], False),
                            ("list", "org/data", "dataset", "v1", False),
                            ("org/data", "dataset", "v1", ["README.md", "ruleset-size/roots/r1.files.sha256",
                                                           "ruleset-size/roots/r1.tar.zst"], False)]
    assert list((tmp_path / "dl").iterdir()) == []


def test_a_hub_dry_run_downloads_the_index_alone_outside_the_root(hub_snapshot, empty_root):
    fetch("org/data", empty_root, dry_run=True, echo=lambda _: None)
    assert [call[3] for call in hub_snapshot if call[0] != "list"] == [[SUMS, INDEX]]
    assert list(empty_root.iterdir()) == []


def test_a_guard_is_called_with_each_place_before_it_is_written(hub_snapshot, empty_root, tmp_path, monkeypatch):
    calls: list[tuple[Path, str, bool]] = []
    targets: list[Path] = []
    real = hub._download
    monkeypatch.setattr(hub, "_download", lambda repo, kind, revision, target, files: (
        targets.append(target.parent), real(repo, kind, revision, target, files)))

    def guard(place: Path, what: str) -> None:
        calls.append((place, what, os.path.lexists(place)))
    report = fetch("org/data", empty_root, only=["ruleset-size"], download_dir=tmp_path / "dl", guard=guard,
                   echo=lambda _: None)
    assert report["placed"] == ["sampling/r1"] and set(targets) == {tmp_path / "dl"}
    held = (tmp_path / "dl", "the directory the download is held in", False)
    destination = (empty_root / "sampling/r1", "the destination of ruleset-size/roots/r1.tar.zst", False)
    assert calls[:1] + calls[2:] == [held, destination, destination]
    assert calls[1][0].parent == tmp_path / "dl" and calls[1][1:] == ("the directory the download is held in", True)
    calls.clear()
    targets.clear()
    (tmp_path / "scratch").mkdir()
    fetch("org/data", empty_root, only=["training"], dry_run=True, download_dir=tmp_path / "unused",
          scratch_dir=tmp_path / "scratch", guard=guard, echo=lambda _: None)
    assert calls == [(tmp_path / "scratch", "the directory the download is held in", True)]
    assert targets == [tmp_path / "scratch"] and list((tmp_path / "scratch").iterdir()) == []
    assert not (tmp_path / "unused").exists()


def test_a_hub_fetch_downloads_the_manifest_of_an_input_already_there_before_it_plans(hub_snapshot, repository,
                                                                                      empty_root):
    fetch(repository, empty_root, only=["training"], echo=lambda _: None)
    shutil.rmtree(empty_root / "sampling")
    hub_snapshot.clear()
    report = fetch("org/data", empty_root, only=["grpo-x"], dry_run=True, echo=lambda _: None)
    assert report["present"] == ["anchors/run/confirm", "selection/run/merge_manifest.json"]
    assert not (empty_root / "sampling").exists() and not list(empty_root.glob(".hub-download-*"))
    report = fetch("org/data", empty_root, only=["grpo-x"], echo=lambda _: None)
    assert report["placed"] == ["sampling/grpo-x"]
    manifest = ["training/inputs/anchors.files.sha256"]
    assert [call[3] for call in hub_snapshot if call[0] != "list"] == [
        [SUMS, INDEX], manifest, [SUMS, INDEX], manifest,
        ["README.md", "training/roots/grpo-x.files.sha256", "training/roots/grpo-x.tar.zst"]]


def test_a_hub_repository_with_a_file_more_or_less_than_its_listing_is_refused(hub_snapshot, repository, empty_root):
    _write(repository / "extra.txt", "x")
    with pytest.raises(HubError, match=r"files not in SHA256SUMS \['extra.txt'\]"):
        fetch("org/data", empty_root, echo=lambda _: None)
    (repository / "extra.txt").unlink()
    (repository / "README.md").unlink()
    with pytest.raises(HubError, match=r"missing \['README.md'\]"):
        fetch("org/data", empty_root, echo=lambda _: None)
    assert list(empty_root.iterdir()) == []


def test_a_document_with_other_bytes_than_listed_is_refused(repository, empty_root):
    _write(repository / "README.md", "# another card\n")
    fetch(repository, empty_root, dry_run=True, echo=lambda _: None)
    with pytest.raises(HubError, match="README.md: not the bytes SHA256SUMS lists"):
        fetch(repository, empty_root, only=["ruleset-size"], echo=lambda _: None)
    assert list(empty_root.iterdir()) == []


def test_the_pack_command_line_reads_the_checkout_s_inventory_by_default(tmp_path, monkeypatch, capsys):
    seen = []
    real = hub.load_inventory
    monkeypatch.setattr(hub, "load_inventory", lambda path: seen.append(Path(path)) or real(path))
    (tmp_path / "out").mkdir()
    assert main_pack(["pack", "--out", str(tmp_path / "out")]) == 2
    assert seen == [REPO / "data" / "hub_inventory.yaml"] and "exists: the layout is written" in capsys.readouterr().err


def test_the_pack_command_line_refuses_where_the_package_finds_no_inventory(tmp_path, monkeypatch, capsys):
    from ruleforge import env

    monkeypatch.setattr(env, "PACKAGE_DIR", tmp_path / "site" / "ruleforge")
    monkeypatch.setattr("ruleforge.core.paths.repository_root", lambda: tmp_path / "nowhere")
    assert main_pack(["pack", "--out", str(tmp_path / "out")]) == 2
    assert "refused: no packaged data/hub_inventory.yaml" in capsys.readouterr().err
    inventory = _write(tmp_path / "inventory.yaml", json.dumps({"format": 2}))
    assert main_pack(["pack", "--inventory", str(inventory), "--out", str(tmp_path / "out")]) == 2
    assert "not an inventory of format 1" in capsys.readouterr().err


def test_a_hub_refusal_is_the_library_s_error_and_still_a_runtime_error():
    from ruleforge.core.errors import RuleForgeError

    assert HubError.__bases__ == (RuntimeError, RuleForgeError)
    with pytest.raises(RuntimeError, match="^the message$"):
        raise HubError("the message")


@pytest.mark.parametrize(("inventory", "message"), [
    ({"format": 2}, "not an inventory"),
    ({"format": 1, "experiments": {}}, "no experiments"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": ["r"]}}}, "each root names its members_digest"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"../r": None}}}}, "not a relative path"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": "A" * 64}}}}, "not a SHA-256 digest"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": 7}}}}, "not a SHA-256 digest"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}}, "y": {"roots": {"r": None}}}}, "r is listed twice"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "raw_outputs": ["s"]}}}, r"\['s'\] are not its roots"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None, "r-x": None}, "files": [
        {"path": "f", "dest": "sampling/r/f.json", "kind": "selection"}]}}}, r"\('sampling/r', 'sampling/r/f.json'\)"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}}}, "models": [
        {"dir": "m", "dest": "sampling/r/adapters", "files": {"a.json": "0" * 64}}]},
     r"\('sampling/r', 'sampling/r/adapters/a.json'\)"),
    ({"format": 1, "experiments": {"x": {}}, "models": [{"dir": "m", "dest": "d", "files": {"../a": "0" * 64}}]},
     "model file"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "/d"}]}}}, "not a relative path"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "d", "kind": "root"}]}}},
     "kind is selection or input"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "d", "kind": "input",
                                                    "format": ARCHIVE}]}}}, "an archive, and only one"),
    ({"format": 1, "experiments": {"x": {}}, "models": [{"dir": "m", "dest": "/abs"}]}, "not a relative path"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "d", "kind": "input", "format": ARCHIVE,
                                                    "prefix": "p", "sha256": "0" * 64}]}}}, "an archive is pinned by"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "d", "kind": "input",
                                                    "members_digest": "0" * 64}]}}}, "a plain file by sha256"),
    # omit lists: a root's, by its name, and an input archive's own
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "omit": ["r"]}}}, "each root names its omit list"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "omit": {"s": ["a/**"]}}}},
     r"x omit: \['s'\] are not its roots"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "omit": {"r": []}}}}, "r: omit is a list of globs"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "omit": {"r": "a/**"}}}}, "omit is a list of globs"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "omit": {"r": ["../a"]}}}}, "not a relative path"),
    ({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r": None}, "omit": {"r": ["a/**", "a/**"]}}}},
     "omit names a glob twice"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "d", "kind": "input", "omit": ["*.log"],
                                                    "sha256": "0" * 64}]}}}, r"\(and only an archive has an omit"),
    ({"format": 1, "experiments": {"x": {"files": [{"path": "f", "dest": "d", "kind": "input", "format": ARCHIVE,
                                                    "prefix": "p", "omit": []}]}}}, "f: omit is a list of globs"),
])
def test_a_malformed_inventory_is_refused(tmp_path, inventory, message):
    with pytest.raises(HubError, match=message):
        load_inventory(_write(tmp_path / "inventory.yaml", json.dumps(inventory)))


def test_the_committed_inventory_names_every_paper_root_once_and_pins_every_file():
    inventory = load_inventory(REPO / "data" / "hub_inventory.yaml")
    roots = [root for experiment in inventory["experiments"].values() for root in experiment.get("roots", {})]
    assert len(roots) == len(set(roots)) == 24
    assert all(pinned for experiment in inventory["experiments"].values()
               for pinned in experiment.get("roots", {}).values())
    assert set(inventory["experiments"]) == {"populations", "ruleset-size", "prompt-ablation", "ruleforge", "training",
                                              "hierarchy"}
    for experiment in inventory["experiments"].values():
        for item in experiment.get("files", []):
            assert item.get("members_digest" if item.get("format") == ARCHIVE else "sha256"), item["path"]
    [sweep] = inventory["experiments"]["ruleforge"]["files"]
    # the paper's launch alone, without its run records or the selection written beside it
    assert sweep["omit"][:5] == ["**/trainer.log", "**/.submitit/**", "**/.hydra/**", "**/multirun.yaml",
                                 "selection.json"]
    assert all(glob.startswith("2026-07-") and glob.endswith("/**") and "23-45-41" not in glob
               for glob in sweep["omit"][5:]) and len(sweep["omit"]) == 10
    # Every root leaves out what no rebuild reads (the draws' summaries and implicated rules, the root's own gate, the
    # conditions' stack records, the campaign markers); each experiment's roots, the kinds its rebuild does not read.
    stack, runs, blocks = "**/condition_stack.json", ["**/runs.jsonl", "**/draw_provenance.json"], ["**/request_blocks.jsonl"]
    assert hub._every_root(inventory) == ("**/smrc_summary.json", "**/implicated_rules.json", "completion.json", stack,
                                          ".ruleforge-*")
    omits = {root: globs for experiment in inventory["experiments"].values()
             for root, globs in experiment.get("omit", {}).items()}
    assert set(omits) == set(roots)
    bank = [*runs, *blocks, "*/subsets/*/k57/draw-*/*", "sampling_manifest.json", "scenario_manifest.json",
            "aggregate/figure2_estimates.json"]
    sizes = inventory["experiments"]["ruleset-size"]["roots"]
    assert {root: omits[root] for root in sizes} == {
        root: [*bank, "rebase.json" if "glimmer" in root else "label_replay.json"] if root.startswith("fig2-")
        else [*runs, *blocks, "sampling_manifest.json"] for root in sizes}
    # a local judge's draws keep their run logs and provenance (its stack), an API judge's their request blocks
    api = ("luna", "haiku")
    for name in ("prompt-ablation", "hierarchy"):
        for root in inventory["experiments"][name]["roots"]:
            globs = [glob for glob in omits[root] if glob not in ("rebase.json", "gpt-5.6-luna/good/**",
                                                                  "gpt-5.6-luna/forced-grpo/**")]
            assert globs == (runs if any(judge in root for judge in api) else blocks), root
    # GPT-5.6 Luna's retired prompt's arm, named by its recorded key, and its GRPO-prompt forced arm
    assert omits["final-2026-09-luna-pop-a22af097-rescored-lenient_last_line"][:2] == [
        "gpt-5.6-luna/good/**", "gpt-5.6-luna/forced-grpo/**"]
    # GPT-5.6 Luna's re-scored arms date their requests from their attempts too (their request blocks were rebased)
    assert {"final-2026-09-luna-pop-a22af097-rescored-lenient_last_line",
            "prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line"} <= set(
        inventory["experiments"]["prompt-ablation"]["all_attempts"])
    # the earlier checkpoints' arms in the two base roots of the training experiment
    assert inventory["experiments"]["training"]["omit"] == {
        "fig5-gemma-2026-10-07-pop-a22af097": ["gemma-4-31B-it/figure5/grpo/**", *blocks],
        "fig5-gemma-ckpt600-2026-10-09-pop-a22af097": blocks,
        "fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097": ["Qwen3-8B/figure5/grpo/**", *blocks],
        "fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097": blocks}
    assert [item.get("omit") for item in inventory["experiments"]["training"]["files"]
            if item.get("format") == ARCHIVE] == [[stack, "draw-*/**"], [stack, "draw-*/**"]]
    assert omits["fig4-2026-09-pop-a22af097-rescored-lenient_last_line"] == [*runs, *blocks, "label_replay.json"]
    assert not inventory["experiments"]["populations"].get("omit")
    assert [model["dir"] for model in inventory["models"]] == ["gemma4-31b-it/checkpoint-600",
                                                                "qwen3-8b/checkpoint-300"]


#: The development set's digest, as both training runs' rows manifests record it among the files their rows were held
#: apart from (``artifact://populations/grpo_v2_dev.jsonl``).
DEVELOPMENT_SHA256 = "cf8b3591dd38b7d5c5065af705f412872d4698b5f434959b4f071bc3bd89b778"


def test_the_inventory_publishes_the_evaluation_scenarios_and_the_development_set_as_plain_files():
    inventory = load_inventory(REPO / "data" / "hub_inventory.yaml")
    [evaluation] = inventory["experiments"]["populations"]["files"]
    assert (evaluation["path"], evaluation["kind"], evaluation["dest"]) == (
        "frozen_500_r3.jsonl", "input", "populations/frozen_500_r3.jsonl")
    assert evaluation["sha256"] == hub.sha256_file(REPO / "data" / "populations" / "frozen_500_r3.jsonl")
    [development] = [item for item in inventory["experiments"]["training"]["files"]
                     if item["path"] == "inputs/development.jsonl"]
    assert (development["kind"], development["dest"], development["sha256"]) == (
        "input", "populations/grpo_v2_dev.jsonl", DEVELOPMENT_SHA256)
    assert "format" not in evaluation and "format" not in development


def test_fetching_the_hierarchy_experiment_selects_its_four_roots_and_names_the_golden_s_other_three():
    inventory = load_inventory(REPO / "data" / "hub_inventory.yaml")
    index = hub.Index(tuple(Entry(f"{group}/roots/{root}.tar.zst", "root", group, f"sampling/{root}", "0" * 64, 1,
                                  format=ARCHIVE, prefix=f"sampling/{root}", root=root)
                            for group, experiment in inventory["experiments"].items()
                            for root in experiment.get("roots", {})), (), "")
    hierarchy = sorted(inventory["experiments"]["hierarchy"]["roots"])
    assert len(hierarchy) == 4 and sorted(entry.root for entry in index.select(["hierarchy"])) == hierarchy
    (row,) = [row for row in table.TABLE if row.name == "hierarchy"]
    read = sorted(uri.removeprefix("artifact://sampling/") for uri in row.runs)
    flat = sorted(set(read) - set(hierarchy))
    assert len(flat) == 3 and {entry.group for entry in index.select(flat)} == {"prompt-ablation"}
    assert {entry.root for entry in index.select(["hierarchy", *flat])} == set(read)


def test_an_entry_serialises_only_what_it_holds():
    assert Entry("p", "selection", "g", "d", "a" * 64, 1).to_json() == {
        "path": "p", "kind": "selection", "group": "g", "dest": "d", "sha256": "a" * 64, "bytes": 1, "format": "file"}
    assert os.path.basename(hub._members_path("x/roots/r.tar.zst")) == "r.files.sha256"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root lists a mode-0 directory")
def test_packing_refuses_a_directory_it_cannot_list(tmp_path, artifacts):
    locked = artifacts / "sampling/r1/m"
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            pack(artifacts, INVENTORY, tmp_path / "out", level=1)
    finally:
        locked.chmod(0o700)


def test_an_archive_s_members_are_its_regular_files(tmp_path):
    repo = _crafted(tmp_path, [_tarinfo("sampling/r", kind=tarfile.DIRTYPE), GOOD_MEMBER])
    assert hub._archive_members(repo / "x/roots/r.tar.zst", "sampling/r", "r") == {"sampling/r/a.json": _sha(b"{}")}


def test_multi_threaded_packing_is_deterministic_whatever_the_thread_count(tmp_path, artifacts):
    archives = []
    for threads in (2, 3):
        archive = tmp_path / f"{threads}.tar.zst"
        hub.pack_tree(artifacts / "sampling/r1", "sampling/r1", archive, tmp_path / f"{threads}.sha256", level=3,
                      threads=threads)
        archives.append(archive.read_bytes())
    assert archives[0] == archives[1]


def test_a_root_listed_for_its_raw_outputs_keeps_them():
    rule = hub._rule({"experiments": {}}, {"raw_outputs": ["r"], "all_attempts": ["r"]}, "r")
    assert rule.keeps("m/naive-rules-above/draw-002/raw_outputs.jsonl") and rule.keeps("m/naive-rules-above/draw-002/attempts.jsonl")
    assert not rule.keeps("m/naive-rules-above/draw-002/judge_outputs.jsonl")
    text = hub.compact_text({"experiments": {"g": {"raw_outputs": ["r"]}}})
    assert ("except attempts.jsonl of the conditions forced* (each condition directory matched by its public style "
            "name), raw_outputs.jsonl of r; ") in text
    assert text.endswith("in every root, input archive and file, as the path's last segment (name), 'host' (host), "
                         "an artifact:// URI from the path's first artifact-root directory (location), the path from "
                         "its data/ segment (repo), the published SHA-256 of a file the package rewrites (digest) and "
                         "the adapter's base model as the Hub names it (hub)")


def test_every_file_is_checked_before_the_first_is_placed(repository, empty_root):
    manifest = repository / "training/selection/merge_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(HubError, match="merge_manifest.json: not the bytes"):
        fetch(repository, empty_root, echo=lambda _: None)
    assert list(empty_root.iterdir()) == []


def _quiet(_: str) -> None:
    pass


def test_verify_takes_a_root_with_a_tilde_or_a_relative_one_as_the_command_line_does(repository, tmp_path,
                                                                                     monkeypatch):
    home = tmp_path / "home"
    (home / "artifacts").mkdir(parents=True)
    fetch(repository, home / "artifacts", echo=_quiet)
    monkeypatch.setenv("HOME", str(home))
    report = hub.verify(repository, "~/artifacts", echo=_quiet)
    assert report["root"] == str(home / "artifacts") and len(report["verified"]) == 4 and report["absent"] == []
    monkeypatch.chdir(home)
    assert hub.verify(repository, "artifacts", echo=_quiet)["root"] == str(home / "artifacts")
    assert not (home / "~").exists()


def test_verify_refuses_a_root_that_names_two_places_without_speaking_of_writes(repository, tmp_path, monkeypatch):
    (tmp_path / "elsewhere/child").mkdir(parents=True)
    (tmp_path / "safe/root").mkdir(parents=True)
    (tmp_path / "safe/link").symlink_to(tmp_path / "elsewhere/child", target_is_directory=True)
    monkeypatch.chdir(tmp_path / "safe")
    for given in (tmp_path / "safe/link/../root", "link/../root"):
        with pytest.raises(HubError, match="names two places") as refused:
            hub.verify(repository, given, echo=_quiet)
        message = str(refused.value)
        assert f"{tmp_path / 'safe/root'} as written out" in message and "where it is read" in message
        assert "write" not in message.split("names two places")[1]


def test_verify_of_a_hub_repository_holds_its_reference_in_the_temporary_directory_and_guards_each_place(
        hub_snapshot, repository, empty_root, tmp_path, monkeypatch):
    fetch(repository, empty_root, echo=_quiet)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    calls: list[tuple[Path, str, bool]] = []
    report = hub.verify("org/data", empty_root, revision="c1", echo=_quiet,
                        guard=lambda place, what: calls.append((place, what, os.path.lexists(place))))
    assert len(report["verified"]) == 4 and report["files"] == len(KEPT) + len(MARKED)
    assert calls[:2] == [(scratch, hub.TEMPORARY, True)] * 2 and len(calls) == 3
    assert calls[2][0].parent == scratch and calls[2][1:] == ("the directory the repository's reference is held in",
                                                              True)
    manifests = ["ruleset-size/roots/r1.files.sha256", "training/inputs/anchors.files.sha256", "training/roots/grpo-x.files.sha256"]
    assert hub_snapshot == [("org/data", "dataset", "c1", [SUMS, INDEX], False),
                            ("org/data", "dataset", "c1", manifests, False)]
    hub_snapshot.clear()
    assert hub.verify("org/data", empty_root, only=["ruleset-size"], echo=_quiet)["verified"] == ["sampling/r1"]
    assert [call[3] for call in hub_snapshot] == [[SUMS, INDEX], ["ruleset-size/roots/r1.files.sha256"]]
    assert list(scratch.iterdir()) == []


def test_a_reference_given_no_scratch_directory_is_held_in_the_temporary_directory(hub_snapshot, tmp_path,
                                                                                  monkeypatch):
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    with hub.reference("org/data", "dataset", None, ["ruleset-size"], None) as (base, selected):
        assert base.parent == tmp_path / "scratch" and [entry.dest for entry in selected] == ["sampling/r1"]
    assert not base.exists()


def test_the_temporary_directory_is_found_without_writing_into_any_candidate(tmp_path, monkeypatch):
    missing, usable, later = tmp_path / "missing", tmp_path / "usable", tmp_path / "later"
    a_file = _write(tmp_path / "file", "x")
    for path in (usable, later):
        path.mkdir()
        os.utime(path, ns=(10**18, 10**18))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(tempfile, "_candidate_tempdir_list", lambda: [str(path) for path in
                                                                      (missing, a_file, usable, later)])
    seen: list[Path] = []
    assert hub.temporary_parent(tmp_path / "root", lambda place, what: seen.append(place)) == usable
    assert seen == [missing, a_file, usable] and usable.stat().st_mtime_ns == 10**18 and list(usable.iterdir()) == []
    monkeypatch.setattr(tempfile, "_candidate_tempdir_list", lambda: [str(missing), str(a_file)])
    with pytest.raises(HubError, match=f"none of {missing}, {a_file} is a directory this process may write"):
        hub.temporary_parent(tmp_path / "root")


@pytest.mark.parametrize(("temporary", "named"), [("root", "is the artifact root"),
                                                  ("root/tmp", "is inside the artifact root")])
def test_a_temporary_directory_in_the_artifact_root_is_refused_before_it_is_tried(tmp_path, monkeypatch, temporary,
                                                                                 named):
    (tmp_path / "root").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / temporary))
    with pytest.raises(HubError, match=named):
        hub.temporary_parent(tmp_path / "root")
    assert list((tmp_path / "root").iterdir()) == []


def _unpackable(repository: Path, path: str) -> None:
    """Flip a zstd magic byte of ``path``'s archive (a later byte may be one no member needs) and relist its bytes."""
    archive = repository / path
    raw = bytearray(archive.read_bytes())
    raw[0] ^= 0xFF
    archive.write_bytes(bytes(raw))

    def relist(payload: dict) -> None:
        [entry] = [entry for entry in payload["files"] if entry["path"] == path]
        entry.update(sha256=hub.sha256_file(archive), bytes=archive.stat().st_size)
    _set_index(repository, relist)


def test_an_archive_that_does_not_unpack_is_a_refusal_of_the_fetch(repository, empty_root):
    _unpackable(repository, "ruleset-size/roots/r1.tar.zst")
    with pytest.raises(HubError) as refused:
        fetch(repository, empty_root, only=["ruleset-size"], echo=_quiet)
    err = str(refused.value)
    assert err.startswith(f"{repository}: ruleset-size/roots/r1.tar.zst, an archive with the bytes its "
                          "SHA256SUMS lists, cannot be unpacked (") and "the repository is damaged" in err
    assert not (empty_root / "sampling/r1").exists() and _leftovers(empty_root) == []


def _run(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *argv], capture_output=True, text=True, cwd=cwd,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})


def _no_probe(monkeypatch) -> None:
    """``tempfile.tempdir`` unset, as in a fresh process, and :func:`tempfile.gettempdir`."""
    def probe(*_):
        raise AssertionError("tempfile's probe ran")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(tempfile, "gettempdir", probe)
    monkeypatch.setattr(tempfile, "_get_default_tempdir", probe)


def _campaign_tmp(tmp_path: Path) -> tuple[Path, Path]:
    campaign = _write(tmp_path / "campaign/sampling_manifest.json", "{}").parent
    (campaign / "tmp").mkdir()
    os.utime(campaign / "tmp", ns=(10**18, 10**18))
    return campaign, campaign / "tmp"


def _untouched(campaign: Path) -> bool:
    return (sorted(path.name for path in campaign.iterdir()) == ["sampling_manifest.json", "tmp"]
            and list((campaign / "tmp").iterdir()) == [] and (campaign / "tmp").stat().st_mtime_ns == 10**18)


def test_a_library_verify_without_a_guard_refuses_a_temporary_directory_in_a_campaign_root(
        hub_snapshot, repository, empty_root, tmp_path, monkeypatch):
    fetch(repository, empty_root, echo=_quiet)
    campaign, tmp = _campaign_tmp(tmp_path)
    _no_probe(monkeypatch)
    monkeypatch.setenv("TMPDIR", str(tmp))
    with pytest.raises(HubError, match=re.escape(f"the temporary directory (TMPDIR) {tmp} is in the campaign root "
                                                 f"{campaign}, which nothing but its own campaign writes")):
        hub.verify("org/data", empty_root, echo=_quiet)
    assert _untouched(campaign) and hub_snapshot == []


def test_a_reference_given_no_scratch_directory_finds_it_without_the_probe_and_holds_it_to_the_campaign_rule(
        hub_snapshot, tmp_path, monkeypatch):
    _no_probe(monkeypatch)
    (tmp_path / "usable").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "usable"))
    with hub.reference("org/data", "dataset", None, ["ruleset-size"], None) as (base, _):
        assert base.parent == tmp_path / "usable"
    assert list((tmp_path / "usable").iterdir()) == []
    campaign, tmp = _campaign_tmp(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp))
    seen: list[Path] = []
    with pytest.raises(HubError, match=re.escape(f"{tmp} is in the campaign root {campaign}")):
        with hub.reference("org/data", "dataset", None, (), None, lambda place, what: seen.append(place)):
            pass  # pragma: no cover - refused before the block
    assert seen == [tmp] and _untouched(campaign) and len(hub_snapshot) == 1


def test_a_hub_dry_run_holds_its_download_where_temporary_parent_finds_it(hub_snapshot, empty_root, tmp_path,
                                                                         monkeypatch):
    _no_probe(monkeypatch)
    targets: list[Path] = []
    real = hub._download
    monkeypatch.setattr(hub, "_download", lambda repo, kind, revision, target, files: (
        targets.append(target.parent), real(repo, kind, revision, target, files)))
    (tmp_path / "usable").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "usable"))
    fetch("org/data", empty_root, dry_run=True, echo=_quiet)
    assert targets == [tmp_path / "usable"] and list((tmp_path / "usable").iterdir()) == []
    (empty_root / "tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(empty_root / "tmp"))
    with pytest.raises(HubError, match="is inside the artifact root"):
        fetch("org/data", empty_root, dry_run=True, echo=_quiet)
    assert list((empty_root / "tmp").iterdir()) == [] and len(targets) == 1


def _root_with_sub(tmp_path: Path, root: Path) -> None:
    """``root/sub`` (its mtime pinned), a symlink ``link`` beside the root to it, and ``root-link`` to the root."""
    (root / "sub").mkdir()
    os.utime(root / "sub", ns=(10**18, 10**18))
    (tmp_path / "link").symlink_to(root / "sub")
    (tmp_path / "root-link").symlink_to(root)


def _root_as_it_was(root: Path) -> bool:
    return ([path.name for path in root.iterdir()] == ["sub"] and list((root / "sub").iterdir()) == []
            and (root / "sub").stat().st_mtime_ns == 10**18)


def _no_directory_made(monkeypatch) -> list:
    made: list = []
    real = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda *args, **kwargs: (made.append(kwargs.get("dir")),
                                                                      real(*args, **kwargs))[1])
    return made


_UNDER_ROOT = [("fresh", "fresh", "is the artifact root"), ("fresh/sub", "fresh", "is inside the artifact root"),
               ("link", "fresh", "is inside the artifact root"), ("fresh", "root-link", "is the artifact root")]


@pytest.mark.parametrize(("scratch", "named_root", "named"), _UNDER_ROOT)
def test_a_dry_run_s_scratch_dir_under_the_root_is_refused_before_anything_is_made(
        hub_snapshot, empty_root, tmp_path, monkeypatch, scratch, named_root, named):
    _root_with_sub(tmp_path, empty_root)
    made = _no_directory_made(monkeypatch)
    with pytest.raises(HubError, match=re.escape(f"the directory the download is held in {tmp_path / scratch} {named} "
                                                 f"{tmp_path / named_root}: a dry run writes nothing under the root")):
        fetch("org/data", tmp_path / named_root, dry_run=True, scratch_dir=tmp_path / scratch, echo=_quiet)
    assert made == [] and hub_snapshot == [] and _root_as_it_was(empty_root)


@pytest.mark.parametrize(("scratch", "named_root", "named"), _UNDER_ROOT)
def test_a_reference_s_scratch_under_the_root_it_is_given_is_refused_before_anything_is_made(
        hub_snapshot, empty_root, tmp_path, monkeypatch, scratch, named_root, named):
    _root_with_sub(tmp_path, empty_root)
    made = _no_directory_made(monkeypatch)
    with pytest.raises(HubError, match=re.escape(f"the temporary directory (TMPDIR) {tmp_path / scratch} {named} "
                                                 f"{tmp_path / named_root}: the repository's reference is downloaded")):
        with hub.reference("org/data", "dataset", None, (), tmp_path / scratch, root=tmp_path / named_root):
            pass  # pragma: no cover - refused before the block
    _no_probe(monkeypatch)
    monkeypatch.setenv("TMPDIR", str(tmp_path / scratch))
    with pytest.raises(HubError, match=re.escape(f"{tmp_path / scratch} {named} {tmp_path / named_root}")):
        with hub.reference("org/data", "dataset", None, (), None, root=tmp_path / named_root):
            pass  # pragma: no cover - refused before the block
    assert made == [] and hub_snapshot == [] and _root_as_it_was(empty_root)
    (tmp_path / "outside").mkdir()
    with hub.reference("org/data", "dataset", None, ["ruleset-size"], tmp_path / "outside", root=empty_root) as (base, _):
        assert base.parent == tmp_path / "outside"
    assert made == [tmp_path / "outside"] and list((tmp_path / "outside").iterdir()) == []


@pytest.mark.parametrize("through_a_link", [False, True])
def test_a_download_directory_in_an_entry_s_destination_is_refused_and_what_it_made_removed(
        hub_snapshot, empty_root, tmp_path, through_a_link):
    (tmp_path / "root-link").symlink_to(empty_root)
    given = (tmp_path / "root-link" if through_a_link else empty_root) / "sampling/r1/dl"
    with pytest.raises(HubError, match=re.escape(f"the directory the download is held in {given} is inside "
                                                 f"{empty_root / 'sampling/r1'}, where ruleset-size/roots/r1.tar.zst is "
                                                 "placed: choose a place outside every entry's destination")):
        fetch("org/data", empty_root, only=["ruleset-size"], download_dir=given, echo=_quiet)
    assert list(empty_root.iterdir()) == []
    assert [call[3] for call in hub_snapshot if call[0] != "list"] == [[SUMS, INDEX]]
    assert fetch("org/data", empty_root, only=["ruleset-size"], echo=_quiet)["placed"] == ["sampling/r1"]


def test_a_refused_fetch_removes_the_download_directories_it_made_and_left_empty(hub_snapshot, empty_root, tmp_path,
                                                                                monkeypatch):
    with pytest.raises(HubError, match="neither an experiment"):
        fetch("org/data", empty_root, only=["nothing"], download_dir=tmp_path / "new/dl", echo=_quiet)
    assert not (tmp_path / "new").exists()
    real = hub._download

    def beside(repo, kind, revision, target, files):
        real(repo, kind, revision, target, files)
        _write(tmp_path / "new/mid/other", "x")
    monkeypatch.setattr(hub, "_download", beside)
    with pytest.raises(HubError, match="neither an experiment"):
        fetch("org/data", empty_root, only=["nothing"], download_dir=tmp_path / "new/mid/dl", echo=_quiet)
    assert _files(tmp_path / "new") == {"mid/other": b"x"} and not (tmp_path / "new/mid/dl").exists()
    monkeypatch.setattr(hub, "_download", real)
    (tmp_path / "there").mkdir()
    with pytest.raises(HubError, match="neither an experiment"):
        fetch("org/data", empty_root, only=["nothing"], download_dir=tmp_path / "there", echo=_quiet)
    assert (tmp_path / "there").is_dir() and list((tmp_path / "there").iterdir()) == []
    fetch("org/data", empty_root, only=["ruleset-size"], download_dir=tmp_path / "kept/dl", echo=_quiet)
    assert list((tmp_path / "kept/dl").iterdir()) == []


@pytest.mark.parametrize("marker", ["sampling_manifest.json", ".ruleforge-campaign"])
def test_a_library_fetch_without_a_guard_writes_into_no_campaign_root(repository, tmp_path, marker):
    campaign = _write(tmp_path / "campaign" / marker, "{}").parent
    (campaign / "artifacts").mkdir()
    with pytest.raises(HubError, match=re.escape(f"the destination of ruleset-size/roots/r1.tar.zst "
                                                 f"{campaign / 'artifacts/sampling/r1'} is in the campaign root "
                                                 f"{campaign}")):
        fetch(repository, campaign / "artifacts", only=["ruleset-size"], echo=_quiet)
    assert list((campaign / "artifacts").iterdir()) == []


def test_a_campaign_root_made_during_a_hub_download_is_refused_before_the_next_download(hub_snapshot, empty_root,
                                                                                      tmp_path, monkeypatch):
    (tmp_path / "dl").mkdir()
    real = hub._download

    def racing(repo, kind, revision, target, files):
        real(repo, kind, revision, target, files)
        _write(tmp_path / "dl/sampling_manifest.json", "{}")
    monkeypatch.setattr(hub, "_download", racing)
    with pytest.raises(HubError, match=re.escape(f"the directory the download is held in {tmp_path / 'dl'}/"
                                                 ".hub-download-") + r"\S+ is in the campaign root"):
        fetch("org/data", empty_root, download_dir=tmp_path / "dl", echo=_quiet)
    assert [call[3] for call in hub_snapshot if call[0] != "list"] == [[SUMS, INDEX]]
    assert sorted(path.name for path in (tmp_path / "dl").iterdir()) == ["sampling_manifest.json"]
    assert list(empty_root.iterdir()) == []


def test_the_campaign_rule_s_refusals_are_the_module_s_own(hub_snapshot, empty_root, tmp_path):
    with pytest.raises(HubError, match="names two places"):
        hub.refuse_campaign("~/somewhere", "the place")
    assert hub.refuse_campaign(tmp_path, "the place") == (tmp_path, Path(os.path.realpath(tmp_path)))
    with pytest.raises(HubError, match="^the directory the download is held in ~/dl names two places"):
        fetch("org/data", empty_root, download_dir="~/dl", echo=_quiet)
    assert not hub_snapshot


def test_packing_checks_every_source_before_it_writes(tmp_path, artifacts):
    shutil.rmtree(artifacts / "sampling/grpo-x")
    shutil.rmtree(artifacts / "anchors")
    (artifacts / "selection/run/merge_manifest.json").unlink()
    with pytest.raises(HubError, match=re.escape(
            f"{artifacts} does not hold what the selection packs: sampling/grpo-x (a directory); "
            "selection/run/merge_manifest.json (a file); anchors/run/confirm (a directory); pack from a root that "
            "holds them, or name with --only what this one holds; nothing was written")):
        pack(artifacts, INVENTORY, tmp_path / "out", level=1)
    assert not (tmp_path / "out").exists()
    assert [entry.path for entry in pack(artifacts, INVENTORY, tmp_path / "out", only=["ruleset-size"], level=1).entries] \
        == ["ruleset-size/roots/r1.tar.zst"]


def _full_disk(path: Path) -> OSError:
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(path))


def _failing_once(monkeypatch, target, name: str, fails) -> None:
    """Make ``target.name`` raise, the first time ``fails(*args, **kwargs)`` returns an error, instead of running."""
    real, armed = getattr(target, name), [True]

    def failing(*args, **kwargs):
        error = fails(*args, **kwargs) if armed else None
        if error is not None:
            armed.clear()
            raise error
        return real(*args, **kwargs)
    monkeypatch.setattr(target, name, failing)


@pytest.mark.parametrize(("failing", "error"), [("mkdtemp", OSError), ("mkdtemp", KeyboardInterrupt),
                                                ("sampling/r1/dl", OSError), ("sampling/r1", OSError)])
def test_a_download_directory_that_cannot_be_made_leaves_nothing_a_later_fetch_is_refused_by(
        hub_snapshot, empty_root, monkeypatch, failing, error):
    given = empty_root / "sampling/r1/dl"
    raised = _full_disk(given) if error is OSError else error()
    if failing == "mkdtemp":
        _failing_once(monkeypatch, tempfile, "mkdtemp",
                      lambda *_, prefix=None, **__: raised if prefix == ".hub-download-" else None)
    else:  # its first attempt fails as the kernel's does, with no parent yet; the one after its parent is made, full
        _failing_once(monkeypatch, os, "mkdir", lambda path, *_, **__: (
            raised if Path(path) == empty_root / failing and Path(path).parent.is_dir() else None))
    with pytest.raises(error):
        fetch("org/data", empty_root, only=["ruleset-size"], download_dir=given, echo=_quiet)
    assert list(empty_root.iterdir()) == [] and hub_snapshot == []
    assert fetch("org/data", empty_root, only=["ruleset-size"], echo=_quiet)["placed"] == ["sampling/r1"]


def test_a_reference_whose_directory_cannot_be_made_leaves_nothing(hub_snapshot, tmp_path, monkeypatch):
    (tmp_path / "scratch").mkdir()
    _failing_once(monkeypatch, tempfile, "mkdtemp", lambda *_, prefix=None, **__: (
        _full_disk(tmp_path / "scratch") if prefix == "ruleforge-hub-reference-" else None))
    with pytest.raises(OSError, match="No space left"):
        with hub.reference("org/data", "dataset", None, (), tmp_path / "scratch"):
            pass  # pragma: no cover - refused before the block
    assert list((tmp_path / "scratch").iterdir()) == [] and hub_snapshot == []
    with hub.reference("org/data", "dataset", None, (), tmp_path / "scratch") as (base, _):
        assert base.parent == tmp_path / "scratch"
    assert list((tmp_path / "scratch").iterdir()) == []


def _interrupted_once(monkeypatch) -> None:
    _failing_once(monkeypatch, hub, "_pinned", lambda *_: KeyboardInterrupt())


def test_a_refused_or_interrupted_pack_leaves_nothing_and_runs_again_into_the_same_place(
        tmp_path, artifacts, repository, monkeypatch, capsys):
    out = tmp_path / "new/deeper/out"
    with pytest.raises(HubError, match="the inventory pins"):
        pack(artifacts, _pinned_inventory(repository, r1="0" * 64), out, level=1)
    assert not (tmp_path / "new").exists()
    _interrupted_once(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        pack(artifacts, INVENTORY, out, level=1)
    assert not (tmp_path / "new").exists()
    wrong = _write(tmp_path / "wrong.yaml", json.dumps(_pinned_inventory(repository, r1="0" * 64)))
    assert main_pack(["pack", "--artifact-root", str(artifacts), "--inventory", str(wrong), "--out", str(out),
                      "--level", "1"]) == 2
    assert "the inventory pins" in capsys.readouterr().err and not (tmp_path / "new").exists()
    assert len(pack(artifacts, INVENTORY, out, level=1).entries) == 4
    _write(tmp_path / "there/kept", "x")
    with pytest.raises(FileExistsError):
        pack(artifacts, INVENTORY, tmp_path / "there", level=1)
    assert _files(tmp_path / "there") == {"kept": b"x"}


def test_an_out_another_run_made_after_this_one_saw_it_absent_is_left_as_that_run_made_it(tmp_path, artifacts,
                                                                                         monkeypatch):
    out, mkdir = tmp_path / "new/out", Path.mkdir

    def racing(self, *args, **kwargs):
        if self == out and not out.exists():  # the other build's mkdir lands first
            mkdir(out, parents=True)
            _write(out / "theirs", "x")
        return mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", racing)
    with pytest.raises(FileExistsError):
        pack(artifacts, INVENTORY, out, level=1)
    assert _files(out) == {"theirs": b"x"}


def test_the_pack_command_line_builds_the_layout_and_refuses_without_zstandard(artifacts, tmp_path, capsys, monkeypatch):
    import yaml

    inventory = _write(tmp_path / "inventory.yaml", yaml.safe_dump(INVENTORY))
    argv = ["pack", "--artifact-root", str(artifacts), "--inventory", str(inventory), "--level", "1"]
    assert main_pack([*argv, "--out", str(tmp_path / "layout")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("dataset: ") and "entries" in out and (tmp_path / "layout").is_dir()
    assert main_pack([*argv, "--out", str(tmp_path / "layout")]) == 2
    assert "refused: " in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "zstandard", None)
    assert main_pack([*argv, "--out", str(tmp_path / "again")]) == 2
    assert "needs zstandard" in capsys.readouterr().err and not (tmp_path / "again").exists()


#: Two adapters' files as an artifact root holds them where a model fetch places them.
ADAPTERS = {"selection/run-a/adapters/checkpoint-6/adapter_config.json": '{"r": 32}',
            "selection/run-a/adapters/checkpoint-6/adapter_model.safetensors": "weights a",
            "selection/run-b/adapters/checkpoint-3/adapter_model.safetensors": "weights b"}
MODELS = [{"dir": "a/checkpoint-6", "dest": "selection/run-a/adapters/checkpoint-6",
           "files": {name: _sha(ADAPTERS[f"selection/run-a/adapters/checkpoint-6/{name}"].encode())
                     for name in ("adapter_config.json", "adapter_model.safetensors")}},
          {"dir": "b/checkpoint-3", "dest": "selection/run-b/adapters/checkpoint-3",
           "files": {"adapter_model.safetensors": _sha(b"weights b")}}]


@pytest.fixture
def adapters(tmp_path) -> Path:
    root = tmp_path / "adapters"
    for relative, text in ADAPTERS.items():
        _write(root / relative, text)
    return root


def test_the_model_repository_is_packed_from_where_a_fetch_places_its_files_and_fetches_back(tmp_path, adapters,
                                                                                            empty_root):
    card = _write(tmp_path / "card.md", "# the model card\n")
    index = hub.pack_models(adapters, {"models": MODELS}, tmp_path / "model", card=card)
    assert [(entry.path, entry.kind, entry.group, entry.dest) for entry in index.entries] == [
        ("a/checkpoint-6/adapter_config.json", "adapter", "a/checkpoint-6",
         "selection/run-a/adapters/checkpoint-6/adapter_config.json"),
        ("a/checkpoint-6/adapter_model.safetensors", "adapter", "a/checkpoint-6",
         "selection/run-a/adapters/checkpoint-6/adapter_model.safetensors"),
        ("b/checkpoint-3/adapter_model.safetensors", "adapter", "b/checkpoint-3",
         "selection/run-b/adapters/checkpoint-3/adapter_model.safetensors")]
    assert index.documents == ("README.md",) and index.compact_rule == ""
    assert (tmp_path / "model/README.md").read_text() == "# the model card\n"
    report = fetch(tmp_path / "model", empty_root, repo_type="model", echo=lambda _: None)
    assert sorted(report["placed"]) == sorted(ADAPTERS)
    assert _files(empty_root) == {relative: text.encode() for relative, text in ADAPTERS.items()}


def test_a_model_pack_holds_each_file_to_its_pin_and_refuses_a_missing_one(tmp_path, adapters):
    wrong = [{**MODELS[0], "files": {**MODELS[0]["files"], "adapter_config.json": "0" * 64}}]
    with pytest.raises(HubError, match="a/checkpoint-6/adapter_config.json: .* is not the 000000000000... the "
                                       "inventory pins"):
        hub.pack_models(adapters, {"models": wrong}, tmp_path / "model")
    assert not (tmp_path / "model").exists()
    (adapters / "selection/run-b/adapters/checkpoint-3/adapter_model.safetensors").unlink()
    with pytest.raises(HubError, match="does not hold the adapters' files: selection/run-b/adapters/checkpoint-3/"
                                       "adapter_model.safetensors; nothing was written"):
        hub.pack_models(adapters, {"models": MODELS}, tmp_path / "model")
    with pytest.raises(HubError, match="the inventory names no model"):
        hub.pack_models(adapters, {"experiments": {}}, tmp_path / "model")
    assert not (tmp_path / "model").exists()


def test_the_pack_command_line_packs_the_model_repository_without_zstandard(adapters, tmp_path, capsys,
                                                                           monkeypatch):
    import yaml

    inventory = _write(tmp_path / "inventory.yaml", yaml.safe_dump({**INVENTORY, "models": MODELS}))
    argv = ["pack", "--model", "--artifact-root", str(adapters), "--inventory", str(inventory)]
    monkeypatch.setitem(sys.modules, "zstandard", None)
    assert main_pack([*argv, "--out", str(tmp_path / "model")]) == 0
    assert capsys.readouterr().out.startswith("model: 3 entries") and (tmp_path / "model" / INDEX).is_file()
    assert main_pack([*argv, "--only", "training", "--out", str(tmp_path / "other")]) == 2
    assert "--model packs every adapter" in capsys.readouterr().err and not (tmp_path / "other").exists()


#: A runtime record as a judge's run records it: the engine's weights directory and the host the job ran on.
RUNTIME = {"engine_args": {"attention_config": {"backend": "TRITON_ATTN"}, "dtype": "float32",
                           "model": "/srv/someone/models/judge-7b"},
           "hostname": "node-7.cluster.example", "vllm_version": "0.28.0"}
PUBLIC_RUNTIME = {**RUNTIME, "engine_args": {**RUNTIME["engine_args"], "model": "judge-7b"},
                  "hostname": hub.HOST_PLACEHOLDER}
#: :data:`RUNTIME` as a campaign root publishes it: what the stack checks read (no host, no placement argument).
PROJECTED_RUNTIME = {"engine_args": {"attention_config": {"backend": "TRITON_ATTN"}, "dtype": "float32"},
                     "vllm_version": "0.28.0"}
RECORDED_DIGEST, PUBLISHED_DIGEST = "1" * 64, "2" * 64
PUBLICATION = hub.Publication(digests={RECORDED_DIGEST: PUBLISHED_DIGEST}, base_model="org/judge-7b")
#: Each rule: a recorded value, and the value its published copy holds.
RULE_VALUES = {"name": ("/srv/someone/models/judge-7b", "judge-7b"),
               "host": ("node-7.cluster.example", hub.HOST_PLACEHOLDER),
               "location": ("/srv/someone/roots/one/sampling/r1/m/judge-7b", "artifact://sampling/r1/m/judge-7b"),
               "repo": ("/srv/someone/repos/checkout/data/rules/judge-7b.jsonl", "data/rules/judge-7b.jsonl"),
               "digest": (RECORDED_DIGEST, PUBLISHED_DIGEST),
               "hub": ("/srv/someone/models/judge-7b", "org/judge-7b")}


def _written(name: str, record) -> bytes:
    """A record's bytes as the code writes it: a JSON sidecar, or one sorted JSON line per row."""
    from ruleforge.core.files import json_sidecar_text

    if name.endswith(".jsonl"):
        return "".join(json.dumps(row, sort_keys=True) + "\n" for row in record).encode("utf-8")
    return json_sidecar_text(record).encode("utf-8")


def _holding(path: str, value) -> dict:
    """A record holding ``value`` at ``path`` (``a.b[].c``), one item in each list."""
    key, _, rest = path.partition(".")
    inner = value if not rest else _holding(rest, value)
    return {key[:-2]: [inner]} if key.endswith("[]") else {key: inner}


def test_the_published_fields_name_the_files_with_paths_or_hosts_and_no_field_of_a_stack_s_identity():
    from ruleforge.evaluate.identity import DRAW_PROVENANCE_FNAME, RUNS_FNAME
    from ruleforge.evaluate.stack import CONDITION_STACK_FNAME, stack_digest

    names = {name for name, _, _ in hub.PUBLISHED_FIELDS}
    assert {RUNS_FNAME, DRAW_PROVENANCE_FNAME, CONDITION_STACK_FNAME, "rebase.json",
            "grpo_data_manifest.json", "merge_manifest.json", "adapter_config.json"} <= names
    assert {rule for _, _, rule in hub.PUBLISHED_FIELDS} == set(RULE_VALUES)
    assert stack_digest(RUNTIME) == stack_digest(PUBLIC_RUNTIME)


@pytest.mark.parametrize(("name", "path", "rule"), hub.PUBLISHED_FIELDS)
def test_a_file_publishes_each_field_its_table_names_once_and_for_all(name, path, rule):
    recorded, public = RULE_VALUES[rule]
    record = {"kept": "/srv/someone/elsewhere", **_holding(path, recorded)}
    rows = [record, {"ledger_rows": 3}] if name.endswith(".jsonl") else record
    published = hub.published_record(name, _written(name, rows), publication=PUBLICATION)
    expected = {"kept": "/srv/someone/elsewhere", **_holding(path, public)}
    assert published == _written(name, [expected, {"ledger_rows": 3}] if name.endswith(".jsonl") else expected)
    assert hub.published_record(name, published, publication=PUBLICATION) == published


@pytest.mark.parametrize("name", sorted({name for name, _, _ in hub.PUBLISHED_FIELDS}))
def test_a_record_with_none_of_the_fields_to_change_is_copied_byte_for_byte(name):
    for data in (b'{"a":1,  "runtime":{"pid":3, "engine_args": null}}\n',
                 b'{"runtime": {"hostname": "host", "engine_args": {"model": null}}}',
                 _written(name, [{"b": 2}] if name.endswith(".jsonl") else {"b": 2})):
        assert hub.published_record(name, data, publication=PUBLICATION) == data


@pytest.mark.parametrize(("name", "path", "rule"), hub.PUBLISHED_FIELDS)
def test_a_record_not_written_as_the_code_writes_it_or_not_json_is_refused_rather_than_rewritten(name, path, rule):
    data = json.dumps(_holding(path, RULE_VALUES[rule][0]), separators=(",", ":")).encode("utf-8") + b"\n"
    with pytest.raises(HubError, match="r1/m/x.* is not written as the code writes it"):
        hub.published_record(name, data, "r1/m/x", PUBLICATION)
    with pytest.raises(HubError, match="r1/m/x.* is not JSON"):
        hub.published_record(name, f'{{"{path.split(".")[-1]}": '.encode(), "r1/m/x", PUBLICATION)


def test_a_file_is_put_back_in_the_form_its_recorded_bytes_are_written_in():
    manifest = {"base": {"path": "/srv/someone/models/judge-7b"}, "kind": "lora_merge"}
    for form in (lambda record: json.dumps(record, indent=1) + "\n", lambda record: json.dumps(record, indent=2)):
        published = hub.published_record("merge_manifest.json", form(manifest).encode())
        assert published == form({**manifest, "base": {"path": "judge-7b"}}).encode()
    runs = json.dumps({"z": 1, "runtime": {"hostname": "node-7"}}) + "\n"  # one plain JSON line per row
    assert hub.published_record("runs.jsonl", runs.encode()) == (
        json.dumps({"z": 1, "runtime": {"hostname": "host"}}) + "\n").encode()


def test_the_location_and_repo_rules_keep_what_is_no_absolute_path_and_name_what_lies_outside_their_trees():
    keep = hub.Publication()
    for value in ("artifact://sampling/r1", "sampling/r1", None, 3):
        assert hub._location(value, keep) == value and hub._repo(value, keep) == value
    assert hub._location("/srv/someone/models/judge-7b", keep) == "judge-7b"
    assert hub._repo("/srv/someone/notes/rules.jsonl", keep) == "rules.jsonl"
    with pytest.raises(HubError, match="the inventory names none"):
        hub.published_record("adapter_config.json", _written("adapter_config.json", {
            "base_model_name_or_path": "/srv/someone/models/judge-7b"}))


def test_a_packed_root_publishes_every_run_record_one_a_record_of_its_root_holds_by_digest_too(tmp_path, empty_root):
    artifacts = tmp_path / "artifacts"
    provenance = _written("draw_provenance.json", {"decoder_draw_id": 0, "runtime": RUNTIME})
    replay = json.dumps({"draw_provenance_sha256": _sha(provenance)})
    for root in ("r1", "r2"):
        _write(artifacts / f"sampling/{root}/m/forced/draw-000/draw_provenance.json", provenance)
        _write(artifacts / f"sampling/{root}/m/forced/draw-000/runs.jsonl", _written("runs.jsonl", [{"runtime": RUNTIME}]))
        _write(artifacts / f"sampling/{root}/aggregate/estimates.json", '{"rate": 0.5}')
    # r2's re-score record holds its draw's provenance by digest, as recorded
    _write(artifacts / "sampling/r2/label_replay.json", replay)
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None, "r2": None}}}}
    index = pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    assert "runtime.hostname of runs.jsonl (host)" in index.compact_rule
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    for root in ("r1", "r2"):
        runs = json.loads((empty_root / f"sampling/{root}/m/forced/draw-000/runs.jsonl").read_text())
        assert runs["runtime"] == PROJECTED_RUNTIME
        # a draw's provenance keeps the fields a rebuild reads (PROJECTIONS): of its runtime, what the stack checks read
        assert json.loads((empty_root / f"sampling/{root}/m/forced/draw-000/draw_provenance.json").read_text()) == {
            "runtime": PROJECTED_RUNTIME}
    assert (empty_root / "sampling/r2/label_replay.json").read_text() == replay
    assert (empty_root / "sampling/r1/aggregate/estimates.json").read_text() == '{"rate": 0.5}'
    assert (artifacts / "sampling/r1/m/forced/draw-000/draw_provenance.json").read_bytes() == provenance


def test_an_input_archive_publishes_its_run_records_too(tmp_path, artifacts, empty_root):
    _write(artifacts / "anchors/run/confirm/draw-000/runs.jsonl", _written("runs.jsonl", [{"runtime": RUNTIME}]))
    pack(artifacts, INVENTORY, tmp_path / "repo", level=1, echo=lambda _: None)
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    runs = json.loads((empty_root / "anchors/run/confirm/draw-000/runs.jsonl").read_text())
    assert runs["runtime"] == PUBLIC_RUNTIME
    assert (empty_root / "anchors/run/confirm/condition.json").read_text() == ARTIFACTS["anchors/run/confirm/condition.json"]


def test_a_plain_file_is_published_and_a_run_record_names_its_published_digest(tmp_path, empty_root):
    artifacts = tmp_path / "artifacts"
    manifest = (json.dumps({"base": {"path": "/srv/someone/models/judge-7b"}}, indent=1) + "\n").encode()
    _write(artifacts / "selection/s/merged/checkpoint-3/merge_manifest.json", manifest)
    weights = {"manifest_sha256": _sha(manifest), "path": "/srv/someone/roots/one/selection/s/merged/checkpoint-3"}
    _write(artifacts / "sampling/r1/m/forced/draw-000/draw_provenance.json", _written("draw_provenance.json", {
        "model_weights": weights, "runtime": RUNTIME}))
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}, "files": [
        {"path": "selection/s/merge_manifest.json", "kind": "selection",
         "dest": "selection/s/merged/checkpoint-3/merge_manifest.json"}]}}}
    pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    published = (empty_root / "selection/s/merged/checkpoint-3/merge_manifest.json").read_bytes()
    assert json.loads(published) == {"base": {"path": "judge-7b"}}
    provenance = json.loads((empty_root / "sampling/r1/m/forced/draw-000/draw_provenance.json").read_text())
    assert provenance["model_weights"] == {"manifest_sha256": _sha(published),
                                           "path": "artifact://selection/s/merged/checkpoint-3"}
    assert (artifacts / "selection/s/merged/checkpoint-3/merge_manifest.json").read_bytes() == manifest


def test_the_model_pack_names_an_adapter_s_base_model_by_its_hub_id(tmp_path):
    artifacts = tmp_path / "artifacts"
    config = json.dumps({"base_model_name_or_path": "/srv/someone/models/judge-7b", "r": 8}, indent=2)
    _write(artifacts / "selection/s/adapters/checkpoint-3/adapter_config.json", config)
    public = json.dumps({"base_model_name_or_path": "org/judge-7b", "r": 8}, indent=2)
    model = {"dir": "a/checkpoint-3", "dest": "selection/s/adapters/checkpoint-3",
             "files": {"adapter_config.json": _sha(public.encode())}}
    hub.pack_models(artifacts, {**INVENTORY, "models": [{**model, "base_model": "org/judge-7b"}]}, tmp_path / "model")
    assert (tmp_path / "model/a/checkpoint-3/adapter_config.json").read_text() == public
    with pytest.raises(HubError, match="the inventory names none"):
        hub.pack_models(artifacts, {**INVENTORY, "models": [model]}, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_a_run_record_that_is_not_json_is_refused(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/m/forced/draw-000/runs.jsonl", b'{"runtime": {"hostname": \n')
    with pytest.raises(HubError, match="r1/m/forced/draw-000/runs.jsonl line 1 is not JSON"):
        pack(artifacts, {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}, tmp_path / "repo", level=1,
             echo=lambda _: None)
    assert not (tmp_path / "repo").exists()


def test_a_local_path_in_a_field_no_rule_covers_is_refused_by_file_line_and_field(tmp_path):
    artifacts = tmp_path / "artifacts"
    planted = {"runtime": {"hostname": "node-7.cluster.example",
                           "engine_args": {"model": "/srv/someone/models/judge-7b",
                                           "tokenizer": f"{HOME}someone/tokenizers/judge-7b"}}}
    _write(artifacts / "sampling/r1/m/forced/draw-000/runs.jsonl", json.dumps(planted, sort_keys=True) + "\n")
    with pytest.raises(HubError) as refused:
        pack(artifacts, {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}, tmp_path / "repo", level=1,
             echo=lambda _: None)
    assert "r1/m/forced/draw-000/runs.jsonl line 1: runtime.engine_args.tokenizer name an absolute or local path" in str(refused.value)
    assert f"{HOME}someone" not in str(refused.value) and not (tmp_path / "repo").exists()


def test_scheduler_fields_no_rule_knows_in_a_draw_provenance_are_refused_by_field_never_by_value(tmp_path, empty_root):
    from ruleforge.core.files import json_sidecar_text

    # Planted independently of PUBLISHED_FIELDS (the round-7 adversarial review's example): the table rewrites the
    # model and the host, and knows nothing of a scheduler block.
    artifacts = tmp_path / "artifacts"
    record = {"runtime": {"engine_args": {"model": "/srv/base/judge"}, "hostname": "synthetic-node"},
              "scheduler": {"working_directory": f"{HOME}synthetic-account/work", "account": "synthetic-cluster-account"}}
    planted = json_sidecar_text(record)
    _write(artifacts / "anchors/run/confirm/draw-000/draw_provenance.json", planted)
    inputs = {"format": 1, "experiments": {"x": {"files": [
        {"path": "inputs/anchors.tar.zst", "kind": "input", "format": ARCHIVE, "prefix": "confirm",
         "dest": "anchors/run/confirm"}]}}}
    with pytest.raises(HubError) as refused:
        pack(artifacts, inputs, tmp_path / "repo", level=1, private=["synthetic-cluster-account"], echo=lambda _: None)
    message = str(refused.value)
    assert ("confirm/draw-000/draw_provenance.json: scheduler.account, scheduler.working_directory name an absolute "
            "or local path or a private name") in message
    assert "synthetic-account" not in message and HOME not in message and not (tmp_path / "repo").exists()
    with pytest.raises(HubError, match=r"draw_provenance.json: scheduler.working_directory name an absolute or local path"):
        pack(artifacts, inputs, tmp_path / "unnamed", level=1, echo=lambda _: None)
    # In a campaign root the draw's provenance keeps only the fields a rebuild reads: no scheduler block is published.
    _write(artifacts / "sampling/r1/m/forced/draw-000/draw_provenance.json", planted)
    roots = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}
    pack(artifacts, roots, tmp_path / "root", level=1, private=["synthetic-cluster-account"], echo=lambda _: None)
    fetch(tmp_path / "root", empty_root, echo=lambda _: None)
    published = json.loads((empty_root / "sampling/r1/m/forced/draw-000/draw_provenance.json").read_text())
    assert published == {"runtime": {"engine_args": {}}}


def test_a_private_name_in_any_json_file_is_refused_and_text_fields_are_left_alone(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/m/forced/condition.json", json.dumps({"notes": {"ran_on": "g9999"}}))
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}
    with pytest.raises(HubError) as refused:
        pack(artifacts, inventory, tmp_path / "repo", level=1, private=["g9999"], echo=lambda _: None)
    assert "r1/m/forced/condition.json: notes.ran_on name an absolute or local path or a private name" in str(refused.value)
    assert "g9999" not in str(refused.value)
    scenario = {"prompt": f"my notes are in {HOME}someone/notes.txt", "response": f"see {USERS}someone/x", "id": "s1"}
    _write(artifacts / "inputs/scenarios.jsonl", json.dumps(scenario) + "\n")
    files = [{"path": "inputs/scenarios.jsonl", "kind": "input", "dest": "inputs/scenarios.jsonl"}]
    index = pack(artifacts, {"format": 1, "experiments": {"x": {"files": files}}}, tmp_path / "text", level=1,
                 echo=lambda _: None)
    assert [entry.path for entry in index.entries] == ["x/inputs/scenarios.jsonl"]
    _write(artifacts / "inputs/scenarios.jsonl", json.dumps({**scenario, "source": f"{HOME}someone/raw.jsonl"}) + "\n")
    with pytest.raises(HubError, match="inputs/scenarios.jsonl line 1: source name an absolute or local path"):
        pack(artifacts, {"format": 1, "experiments": {"x": {"files": files}}}, tmp_path / "meta", level=1,
             echo=lambda _: None)


def test_the_pack_command_line_reads_the_private_names_from_a_file(tmp_path, capsys):
    import yaml

    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/m/forced/condition.json", json.dumps({"account": "acct-17"}))
    inventory = _write(tmp_path / "inventory.yaml", yaml.safe_dump({"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {
        "r1": None}}}}))
    names = _write(tmp_path / "private.txt", "acct-17\n\n")
    argv = ["pack", "--artifact-root", str(artifacts), "--inventory", str(inventory), "--level", "1"]
    assert main_pack([*argv, "--private", str(names), "--out", str(tmp_path / "repo")]) == 2
    assert "condition.json: account name an absolute or local path or a private name" in capsys.readouterr().err
    assert main_pack([*argv, "--out", str(tmp_path / "plain")]) == 0


#: Where a local path begins, as the packer's check reads it (its LOCAL_ROOTS).
HOME, USERS = hub.LOCAL_ROOTS


#: Each projected kind: a recorded file's records, and the records its published copy keeps.
PROJECTED = {
    "bundle_outcomes.jsonl": (
        [{"scenario_id": "s1", "rule_judgments": {"Rule 1": "yes"}, "contradiction": 0.0, "penalty": 0.0,
          "in_context_penalty": None, "in_context_contradiction": None}],
        [{"scenario_id": "s1", "rule_judgments": {"Rule 1": "yes"}, "contradiction": 0.0}]),
    # the draw's ledger (_ledger_text) rejects s1's attempt 0 and accepts its attempt 1, in block b1
    "raw_outputs.jsonl": (
        [{"scenario_id": "s1", "semantic_attempt_id": 1, "request_block_id": "b1", "component_id": "Rule 1",
          "raw_response": "no", "completion": "no"},
         {"scenario_id": "s1", "semantic_attempt_id": 0, "request_block_id": "b0", "component_id": "full_ruleset",
          "raw_response": "maybe", "completion": "maybe"},
         {"scenario_id": "s1", "semantic_attempt_id": 1, "request_block_id": "b1", "component_id": "full_ruleset",
          "raw_response": "<think>x</think>yes", "completion": "yes", "finish_reason": "stop", "error": None,
          "raw_output_sha256": "3" * 64}],
        [{"scenario_id": "s1", "semantic_attempt_id": 1, "request_block_id": "b1", "component_id": "full_ruleset",
          "raw_response": "<think>x</think>yes", "completion": "yes"}]),
    "evaluations.jsonl": (
        [{"scenario_id": "s1", "scenario": {"prompt": "p", "response": "r", "source": "wildchat"},
          "evaluations": {"judge": {"verdict": "yes"}}, "prompt_response_hash": "4" * 64}],
        [{"scenario_id": "s1", "scenario": {"source": "wildchat"}}]),
    "completion.json": (
        {"state": "complete", "decoder_draw_id": 0, "grid": {"rows": 2}, "updated_at": "2026-10-01T00:00:00Z"},
        {"state": "complete", "decoder_draw_id": 0}),
    "draw_provenance.json": (
        {"runtime": RUNTIME, "component_ids": ["full_ruleset"], "code_revision": "abc", "tp_size": 2,
         "model_type": "local"},
        {"runtime": PROJECTED_RUNTIME, "tp_size": 2, "model_type": "local"}),
    "runs.jsonl": (
        [{"runtime": RUNTIME, "started_at": "2026-10-01T00:00:00Z", "ledger_rows": 4, "code_revision": "abc"}],
        [{"runtime": PROJECTED_RUNTIME, "code_revision": "abc"}]),
}


def test_a_campaign_root_publishes_only_the_rows_and_fields_a_rebuild_reads(tmp_path, empty_root):
    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/sampling_manifest.json", '{"campaign_id": "r1"}')
    _write(artifacts / "sampling/r1/m/forced/draw-000/attempts.jsonl", _ledger_text("r1", "m/forced", 0))
    for name, (recorded, _) in PROJECTED.items():
        _write(artifacts / f"sampling/r1/m/forced/draw-000/{name}", _written(name, recorded))
        _write(artifacts / f"anchors/run/confirm/draw-000/{name}", _written(name, recorded))
    inventory = {"format": 1, "experiments": {"x": {
        "profile": "training-gemma", "roots": {"r1": None}, "raw_outputs": ["r1"], "draw0_evaluations": True, "files": [
            {"path": "inputs/anchors.tar.zst", "kind": "input", "format": ARCHIVE, "prefix": "confirm",
             "dest": "anchors/run/confirm"}]}}}
    blocks = [{"request_block_id": "b", "scenario_ids": ["s1", "s2"], "recorded_at": "2026-10-01T00:00:00Z"}]
    _write(artifacts / "sampling/r2/m/naive/draw-000/request_blocks.jsonl", _written("request_blocks.jsonl", blocks))
    inventory["experiments"]["x"]["roots"]["r2"] = None
    inventory["experiments"]["x"]["profiles"] = {"r2": "hierarchy-api"}
    index = pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    assert ("a training-gemma root (r1) keeps of each bundle_outcomes.jsonl scenario_id, decoder_draw_id, contradiction"
            in index.compact_rule)
    assert ("of each raw_outputs.jsonl scenario_id, semantic_attempt_id, request_block_id, component_id, raw_response, "
            "completion of the rows whose component_id is full_ruleset") in index.compact_rule
    assert ("of each raw_outputs.jsonl it keeps only the rows of the draw's accepted full-ruleset calls, by that "
            "replay") in index.compact_rule
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    assert (empty_root / "sampling/r2/m/naive/draw-000/request_blocks.jsonl").read_bytes() == _written(
        "request_blocks.jsonl", [{"recorded_at": "2026-10-01T00:00:00Z"}])
    projections = hub.PROFILES["training-gemma"].projections
    for name, (recorded, kept) in PROJECTED.items():
        published = (empty_root / f"sampling/r1/m/forced/draw-000/{name}").read_bytes()
        assert published == _written(name, kept), name  # in the form the record is written in
        assert hub.published_record(name, published, projection=projections[name]) == published  # idempotent
        # an input archive keeps every row and field
        anchor = (empty_root / f"anchors/run/confirm/draw-000/{name}").read_bytes()
        assert anchor == hub.published_record(name, _written(name, recorded)), name
    unescaped = (json.dumps({"scenario_id": "s1", "scenario": {"source": "w", "prompt": "café"}}, ensure_ascii=False)
                 + "\n").encode()
    assert hub.published_record("evaluations.jsonl", unescaped, projection=projections["evaluations.jsonl"]) == (
        json.dumps({"scenario_id": "s1", "scenario": {"source": "w"}}) + "\n").encode()
    plain = (json.dumps({"scenario_id": "s1", "penalty": 0.0}) + "\n").encode()
    assert hub.published_record("bundle_outcomes.jsonl", plain, projection=projections["bundle_outcomes.jsonl"]) == (
        json.dumps({"scenario_id": "s1"}) + "\n").encode()
    with pytest.raises(HubError, match="line 1 is not a JSON object"):
        hub.published_record("bundle_outcomes.jsonl", b"[1]\n", projection=projections["bundle_outcomes.jsonl"])
    with pytest.raises(HubError, match="applies to a JSONL file's rows only"):
        hub.published_record("x.json", b'{"a": 1}', projection=hub.Projection(("a",), row=(("a", "2"),)))


def test_every_campaign_root_leaves_out_what_the_compact_omit_list_names(tmp_path, empty_root):
    artifacts = tmp_path / "artifacts"
    for root in ("r1", "r2"):
        _write(artifacts / f"sampling/{root}/m/forced/draw-000/smrc_summary.json", "{}")
        _write(artifacts / f"sampling/{root}/m/forced/condition.json", "{}")
    _write(artifacts / "sampling/r2/completion.json", "{}")
    inventory = {"format": 1, "compact": {"omit": ["**/smrc_summary.json", "completion.json"]},
                 "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None, "r2": None}}}}
    lines: list[str] = []
    index = pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lines.append)
    assert "every root also leaves out **/smrc_summary.json, completion.json" in index.compact_rule
    assert "every campaign root: 3 file(s) omitted (**/smrc_summary.json: 2, completion.json: 1)" in lines
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    assert sorted(path.relative_to(empty_root).as_posix() for path in empty_root.rglob("*") if path.is_file()) == [
        "sampling/r1/m/forced/condition.json", f"sampling/r1/{PUBLISHED_COPY_FNAME}",
        "sampling/r2/m/forced/condition.json", f"sampling/r2/{PUBLISHED_COPY_FNAME}"]
    # A selection need not hold what every list names; the whole inventory must.
    pack(artifacts, inventory, tmp_path / "only", level=1, only=["r1"], echo=lambda _: None)
    unused = {**inventory, "compact": {"omit": ["**/smrc_summary.json", "completion.json", "**/judge_outputs.jsonl"]}}
    with pytest.raises(HubError, match=r"every campaign root: its omit list's \['\*\*/judge_outputs.jsonl'\] leave out"):
        pack(artifacts, unused, tmp_path / "unused", level=1, echo=lambda _: None)
    bad = _write(tmp_path / "bad.yaml", "format: 1\ncompact: {omit: []}\nexperiments: {x: {roots: {r1: null}}}\n")
    with pytest.raises(HubError, match="compact: omit is a list of globs"):
        load_inventory(bad)


def test_a_root_that_already_holds_a_published_copy_record_is_refused(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write(artifacts / f"sampling/r1/{PUBLISHED_COPY_FNAME}", "{}")
    with pytest.raises(HubError, match=r"r1 already holds \['published_copy.json'\], which its published copy writes"):
        pack(artifacts, {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}, tmp_path / "repo", level=1,
             echo=lambda _: None)
    assert not (tmp_path / "repo").exists()


def test_a_root_publishes_each_kept_ledger_as_its_strict_replay_s_form_and_an_input_keeps_its_ledger(tmp_path,
                                                                                                     empty_root):
    from ruleforge.evaluate.ledger import DrawLedger

    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/sampling_manifest.json", '{"campaign_id": "c1"}')
    _write(artifacts / "sampling/r1/m/forced/draw-002/attempts.jsonl", _ledger_text("c1", "m/forced", 2))
    _write(artifacts / "anchors/run/confirm/draw-000/attempts.jsonl", _ledger_text("a", "confirm", 0))
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}, "files": [
        {"path": "inputs/anchors.tar.zst", "kind": "input", "format": ARCHIVE, "prefix": "confirm",
         "dest": "anchors/run/confirm"}]}}}
    index = pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    assert f"a root publishes each attempts.jsonl it keeps as {PUBLISHED_LEDGER_FNAME}" in index.compact_rule
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    draw = empty_root / "sampling/r1/m/forced/draw-002"
    assert sorted(path.name for path in draw.iterdir()) == [PUBLISHED_LEDGER_FNAME]
    form = json.loads((draw / PUBLISHED_LEDGER_FNAME).read_text())
    assert [row["full_ruleset"]["semantic_attempt_id"] for row in form["accepted"]] == [1]  # a training root's parts
    assert "started_at" not in form and form["has_attempts"]  # a local training root reads no request dates
    state = DrawLedger(draw, campaign_id="c1", condition_id="m/forced", decoder_draw_id=2).accepted()
    recorded = DrawLedger(artifacts / "sampling/r1/m/forced/draw-002", campaign_id="c1", condition_id="m/forced",
                          decoder_draw_id=2).load()
    assert state.accepted_components("s1")["full_ruleset"][0].semantic_attempt_id == (
        recorded.accepted_components("s1")["full_ruleset"][0].semantic_attempt_id) == 1
    assert (empty_root / "anchors/run/confirm/draw-000/attempts.jsonl").read_text() == _ledger_text("a", "confirm", 0)


def test_a_root_ledger_the_packer_cannot_place_or_replay_is_refused(tmp_path):
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}, "all_attempts": ["r1"]}}}
    cases = (({"m/forced/draw-000/attempts.jsonl": _ledger_text("c1", "m/forced", 0)},
              "holds draw records but no sampling_manifest.json naming its campaign"),
             ({"sampling_manifest.json": '{"campaign_id": "c1"}', "m/draw-x/attempts.jsonl": _ledger_text("c1", "m", 0)},
              "r1/m/draw-x/attempts.jsonl: a ledger or raw outputs file lies in a condition's draw directory"),
             ({"sampling_manifest.json": '{"campaign_id": "c2"}',
               "m/forced/draw-000/attempts.jsonl": _ledger_text("c1", "m/forced", 0)},
              r"belongs to \('c1', 'm/forced', 0\), not \('c2', 'm/forced', 0\)"),
             ({"sampling_manifest.json": '{"campaign_id": "c1"}',
               "m/forced/draw-000/attempts.jsonl": _ledger_text("c1", "m/forced", 0),
               f"m/forced/draw-001/{PUBLISHED_LEDGER_FNAME}": "{}"},
              r"already holds \['m/forced/draw-001/accepted_attempts.json'\], which its published copy writes"))
    for number, (files, message) in enumerate(cases):
        artifacts = tmp_path / f"artifacts{number}"
        for relative, text in files.items():
            _write(artifacts / "sampling/r1" / relative, text)
        with pytest.raises((HubError, hub.RuleForgeError), match=message):
            pack(artifacts, inventory, tmp_path / f"repo{number}", level=1, echo=lambda _: None)
        assert not (tmp_path / f"repo{number}").exists()
    tree = _write(tmp_path / "tree/m/draw-000/attempts.jsonl", "x\n").parent.parent.parent
    with pytest.raises(HubError, match="a renamed file is packed as its published bytes"):
        hub.pack_tree(tree, "t", tmp_path / "t.tar.zst", tmp_path / "t.files.sha256", level=1,
                      renames=hub.LEDGER_PUBLISHED_AS)


def test_a_local_path_in_a_list_or_in_a_file_that_is_not_json_is_refused_by_place(tmp_path):
    artifacts = tmp_path / "artifacts"
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}
    _write(artifacts / "sampling/r1/m/forced/condition.json", json.dumps({"extra": {"paths": [f"{HOME}someone/x"]}}))
    with pytest.raises(HubError, match=r"condition.json: extra.paths\[\] name an absolute or local path"):
        pack(artifacts, inventory, tmp_path / "list", level=1, echo=lambda _: None)
    # a kind no profile projects is held to the local-name check as it is; a projected one is refused as not JSON first
    _write(artifacts / "sampling/r1/m/forced/condition.json", "{}")
    _write(artifacts / "sampling/r1/scenario_manifest.json", f'{{"source_name": "{HOME}someone/x"')
    with pytest.raises(HubError, match="scenario_manifest.json is not JSON .*, so its fields cannot be held to the "
                                       "local-name"):
        pack(artifacts, inventory, tmp_path / "torn", level=1, echo=lambda _: None)
    (artifacts / "sampling/r1/scenario_manifest.json").unlink()
    _write(artifacts / "sampling/r1/m/forced/condition.json", f'{{"extra": "{HOME}someone/x"')
    with pytest.raises(HubError, match="condition.json is not JSON .*: its published copy cannot be made"):
        pack(artifacts, inventory, tmp_path / "torn2", level=1, echo=lambda _: None)


def test_a_root_s_raw_outputs_need_their_draw_s_ledger(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/sampling_manifest.json", '{"campaign_id": "r1"}')
    _write(artifacts / "sampling/r1/m/forced/draw-000/raw_outputs.jsonl", '{"scenario_id": "s1"}\n')
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}, "raw_outputs": ["r1"]}}}
    with pytest.raises(HubError, match="published by its ledger's replay, and the draw holds no attempts.jsonl"):
        pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    assert hub.published_record("raw_outputs.jsonl", b'[1]\n{"a": 1}\n', keep=lambda row: isinstance(row, dict)) == (
        b'{"a": 1}\n')


def test_a_root_file_of_a_kind_no_rule_names_is_refused_and_an_unlisted_nested_field_is_left_out(tmp_path, empty_root):
    artifacts = tmp_path / "artifacts"
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}}
    _write(artifacts / "sampling/r1/m/forced/draw-000/new_metadata.json", '{"note": "a new kind"}')
    with pytest.raises(HubError, match=r"r1/m/forced/draw-000/new_metadata.json: a file kind the package's rules "
                                       r"neither publish \(hub.PUBLISHED_KINDS\) nor leave out"):
        pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    (artifacts / "sampling/r1/m/forced/draw-000/new_metadata.json").unlink()
    runtime = {**RUNTIME, "unread_field": "planted", "engine_args": {**RUNTIME["engine_args"], "max_num_seqs": 8}}
    _write(artifacts / "sampling/r1/m/forced/draw-000/draw_provenance.json", _written("draw_provenance.json", {
        "runtime": runtime, "model_type": "local", "unread_top_field": 1}))
    pack(artifacts, inventory, tmp_path / "repo", level=1, echo=lambda _: None)
    fetch(tmp_path / "repo", empty_root, echo=lambda _: None)
    published = json.loads((empty_root / "sampling/r1/m/forced/draw-000/draw_provenance.json").read_text())
    assert published == {"runtime": PROJECTED_RUNTIME, "model_type": "local"}
    with pytest.raises(HubError, match="r1/x: runtime.engine_args holds an object"):
        hub.published_record("p.json", json.dumps({"runtime": {"engine_args": {"a": 1}}}).encode(), "r1/x",
                             projection=hub.Projection(("runtime.engine_args",)))
    with pytest.raises(HubError, match="r1/x: runtime is not an object, and its projection names fields of it"):
        hub.published_record("p.json", json.dumps({"runtime": [1]}).encode(), "r1/x",
                             projection=hub.Projection(("runtime.vllm_version",)))
    assert hub.published_record("p.json", json.dumps({"runtime": None}).encode(), "r1/x",
                                projection=hub.Projection(("runtime.vllm_version",))) == b'{"runtime": null}'


def test_a_root_s_conditions_manifest_and_rebase_record_keep_what_its_kind_s_readers_read(tmp_path):
    manifest = {"schema_version": 2, "campaign_id": "c", "campaign_mode": "final", "max_semantic_attempts": 20,
                "generation_seed_base": 16, "retry_scope": "component", "scenarios_per_request_block": 50,
                "expected_decoder_draws": 5, "scenario_manifest_sha256": "a" * 64, "temperature": 1.0,
                "bootstrap": {"seed": 1}, "created_at": "2026-10-27T00:00:00+00:00", "conditions": {"m/x": {}}}
    hop = {"parent": {"root": "artifact://sampling/p", "campaign_id": "p", "scenario_manifest_sha256": "b" * 64,
                      "scenario_source_sha256": "c" * 64, "sampling_manifest_sha256": "d" * 64},
           "target": {"population": "artifact://populations/x.jsonl", "count": 2}, "removed_ids": ["s1"],
           "added_ids": ["s2"]}
    rebase = {**hop, "chain": [hop], "target": {**hop["target"], "scenario_manifest_sha256": "a" * 64},
              "created_at": "2026-10-27T00:00:00+00:00", "totals": {"copied": 1}}
    condition = {"schema_version": 1, "condition_id": "m/x", "temperature": 1.0, "extra": {"policy": "base"},
                 "fingerprint": "f" * 64}
    published = {}
    for profile in ("training-gemma", "prompt-ablation-local"):
        artifacts = tmp_path / profile / "artifacts"
        _write(artifacts / "sampling/r1/sampling_manifest.json", json.dumps(manifest))
        _write(artifacts / "sampling/r1/rebase.json", json.dumps(rebase))
        _write(artifacts / "sampling/r1/m/x/condition.json", json.dumps(condition))
        pack(artifacts, {"format": 1, "experiments": {"x": {"profile": profile, "roots": {"r1": None}}}},
             tmp_path / profile / "repo", level=1, echo=lambda _: None)
        (tmp_path / profile / "fetched").mkdir()
        fetch(tmp_path / profile / "repo", tmp_path / profile / "fetched", echo=lambda _: None)
        published[profile] = {name: json.loads((tmp_path / profile / "fetched/sampling/r1" / name).read_text())
                              for name in ("sampling_manifest.json", "rebase.json", "m/x/condition.json")}
    # the fingerprint covers every field of a condition but its envelope's schema version
    assert published["training-gemma"]["m/x/condition.json"] == {key: value for key, value in condition.items()
                                                                 if key != "schema_version"}
    # a training family's design check compares every key but the run's, so only the conditions and the time go
    assert published["training-gemma"]["sampling_manifest.json"] == {
        key: value for key, value in manifest.items() if key not in ("conditions", "created_at")}
    assert published["prompt-ablation-local"]["sampling_manifest.json"] == {
        key: manifest[key] for key in ("schema_version", "campaign_id", "campaign_mode", "max_semantic_attempts",
                                       "generation_seed_base", "retry_scope", "scenarios_per_request_block",
                                       "expected_decoder_draws", "scenario_manifest_sha256")}
    parent = {key: value for key, value in hop["parent"].items() if key != "root"}
    assert published["training-gemma"]["rebase.json"] == {
        "parent": {key: parent[key] for key in ("campaign_id", "scenario_manifest_sha256", "sampling_manifest_sha256")},
        "target": {"scenario_manifest_sha256": "a" * 64}, "removed_ids": ["s1"], "added_ids": ["s2"],
        "chain": [{"parent": parent, "target": hop["target"], "removed_ids": ["s1"], "added_ids": ["s2"]}]}
    # a kind whose readers read no rebase record keeps a root's whole
    assert published["prompt-ablation-local"]["rebase.json"] == rebase


def test_a_projection_names_the_fields_of_a_list_s_objects_and_a_star_never_reaches_into_a_list():
    record = {"chain": [{"parent": {"campaign_id": "a", "root": "artifact://x"}, "target": {"count": 2}},
                        {"parent": {"campaign_id": "b", "root": "artifact://y"}, "target": {"count": 3}}],
              "removed_ids": ["s1"], "note": "unread"}
    projection = hub.Projection(("chain[].parent.campaign_id", "chain[].target", "removed_ids"),
                                whole=("chain[].target",))
    published = json.loads(hub.published_record("r.json", json.dumps(record).encode(), "r1/r", projection=projection))
    assert published == {"chain": [{"parent": {"campaign_id": "a"}, "target": {"count": 2}},
                                   {"parent": {"campaign_id": "b"}, "target": {"count": 3}}], "removed_ids": ["s1"]}
    with pytest.raises(HubError, match="r1/r: chain is not a list of objects, and its projection names their fields"):
        hub.published_record("r.json", json.dumps({"chain": {"parent": {}}}).encode(), "r1/r", projection=projection)
    # ``*`` names a key: a list of objects under it is refused as an object its projection does not keep whole
    with pytest.raises(HubError, match="r1/r: chain holds an object"):
        hub.published_record("r.json", json.dumps(record).encode(), "r1/r", projection=hub.Projection(("*",)))


def test_an_absolute_path_whatever_its_prefix_is_refused_in_metadata_and_left_alone_in_text(tmp_path):
    # planted independently of the packer's own list of local roots
    for value in ("/srv/models/judge-7b", "/mnt/scratch/run", "/tmp/stage/x", "~/notes", "~user/notes",
                  "C:\\Users\\x", "D:/runs/x"):
        artifacts = tmp_path / f"artifacts-{abs(hash(value))}"
        _write(artifacts / "sampling/r1/m/forced/condition.json", json.dumps({"extra": {"source": value}}))
        with pytest.raises(HubError) as refused:
            pack(artifacts, {"format": 1, "experiments": {"x": {"profile": "training-gemma", "roots": {"r1": None}}}},
                 tmp_path / f"repo-{abs(hash(value))}", level=1, echo=lambda _: None)
        assert ("r1/m/forced/condition.json: extra.source name an absolute or local path" in str(refused.value)
                and value not in str(refused.value)), value
    for value in ("https://huggingface.co/org/model", "artifact://sampling/r1/x", "data/rules/x.jsonl", "1/2/3",
                  "responses", "C:x"):
        hub.refuse_private("condition.json", json.dumps({"endpoint": value}).encode(), "r1/condition.json")
    text = {"prompt": "my files are under /srv/x/ and C:\\y", "response": "see /tmp/z/ or ~/w"}
    hub.refuse_private("frozen_500_r3.jsonl", (json.dumps(text) + "\n").encode(), "populations/frozen_500_r3.jsonl")
    rewards = {"0": {"per_scenario_rewards": [{"scenario": {"prompt": "/srv/x/", "response": "/mnt/y/"}}]}}
    hub.refuse_private("reward_all.jsonl", (json.dumps(rewards) + "\n").encode(), "sweep/x/reward_all.jsonl")


def test_a_whole_value_that_is_an_absolute_path_of_one_component_is_refused_unless_its_field_is_exempt(
        tmp_path, monkeypatch):
    # the re-check's input (a training-qwen root's condition) and the home and drive forms, through a real pack
    inventory = {"format": 1, "experiments": {"x": {"profile": "training-qwen", "roots": {"r1": None}}}}
    for number, value in enumerate(("/srv", "/mnt", "/tmp", "/opt", "/srv/", "/", "~", "~user", "~user/", "C:", "C:\\",
                                    "D:/", " /srv ")):
        artifacts = tmp_path / f"artifacts-{number}"
        _write(artifacts / "sampling/r1/m/x/condition.json", json.dumps({"extra": {"review_unknown": value}}))
        with pytest.raises(HubError) as refused:
            pack(artifacts, inventory, tmp_path / f"repo-{number}", level=1, echo=lambda _: None)
        assert ("r1/m/x/condition.json: extra.review_unknown name an absolute or local path" in str(refused.value)
                and repr(value) not in str(refused.value)), value
    # a value of that shape that is no path is exempt by its field, in the table of text fields, not by its value
    monkeypatch.setitem(hub.TEXT_FIELDS, "condition.json", ("endpoint",))
    artifacts = tmp_path / "artifacts-endpoint"
    _write(artifacts / "sampling/r1/m/x/condition.json", json.dumps({"endpoint": "/v1", "extra": {}}))
    pack(artifacts, inventory, tmp_path / "repo-endpoint", level=1, echo=lambda _: None)
    (tmp_path / "fetched").mkdir()
    fetch(tmp_path / "repo-endpoint", tmp_path / "fetched", echo=lambda _: None)
    assert json.loads((tmp_path / "fetched/sampling/r1/m/x/condition.json").read_text())["endpoint"] == "/v1"
    _write(artifacts / "sampling/r1/m/x/condition.json", json.dumps({"endpoint": "/v1", "extra": {"route": "/v1"}}))
    with pytest.raises(HubError, match="condition.json: extra.route name an absolute or local path"):
        pack(artifacts, inventory, tmp_path / "repo-route", level=1, echo=lambda _: None)


def test_a_root_the_inventory_names_no_profile_for_is_refused(tmp_path):
    artifacts = tmp_path / "artifacts"
    _write(artifacts / "sampling/r1/m/forced/condition.json", "{}")
    with pytest.raises(HubError, match="r1: the inventory names no profile of .* so what its published copy keeps is"):
        pack(artifacts, {"format": 1, "experiments": {"x": {"roots": {"r1": None}}}}, tmp_path / "repo", level=1,
             echo=lambda _: None)
    bad = _write(tmp_path / "bad.yaml", "format: 1\nexperiments: {x: {profile: nothing, roots: {r1: null}}}\n")
    with pytest.raises(HubError, match=r"x: \['r1'\] name no profile of"):
        load_inventory(bad)
