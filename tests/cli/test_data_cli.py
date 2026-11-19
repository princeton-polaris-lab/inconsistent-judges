from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import yaml

from ruleforge import hub
from ruleforge.cli import data
from tests.test_repro_hub import ARTIFACTS, FETCHED as LANDED, INVENTORY, _files, _write

#: What a fetch of the test repository lands.
FETCHED = set(LANDED)

REPO = Path(__file__).resolve().parents[2]
NAMED = {**INVENTORY, "experiments": {"prompt-ablation": INVENTORY["experiments"]["ruleset-size"],
                                      "training": INVENTORY["experiments"]["training"]}}
PLACED = ["anchors/run/confirm", "sampling/grpo-x", "sampling/r1", "selection/run/merge_manifest.json"]


@pytest.fixture
def repository(tmp_path) -> Path:
    """A dataset repository packed from the synthetic artifact root."""
    artifacts = tmp_path / "source"
    for relative, text in ARTIFACTS.items():
        _write(artifacts / relative, text)
    hub.pack(artifacts, NAMED, tmp_path / "repo", card=_write(tmp_path / "card.md", "# the card\n"), level=1,
             echo=lambda _: None)
    return tmp_path / "repo"


@pytest.fixture
def model(tmp_path) -> Path:
    """A model repository of one adapter's two files."""
    repo = tmp_path / "model"
    built = hub._Built(repo)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        _write(repo / "m/checkpoint-1" / name, f"the {name}")
        built.add(hub._file_entry(repo, f"m/checkpoint-1/{name}", "adapter", "m/checkpoint-1",
                                  f"selection/run/adapters/c1/{name}"))
    built.finish("", None)
    return repo


@pytest.fixture
def root(tmp_path) -> Path:
    path = tmp_path / "root"
    path.mkdir()
    return path


def _tree(top: Path) -> dict[str, tuple[bytes, int]]:
    return {path.relative_to(top).as_posix(): (path.read_bytes() if path.is_file() else b"", path.stat().st_mtime_ns)
            for path in sorted(top.rglob("*"))}


def test_fetch_places_every_entry_and_prints_what_it_placed(cli, repository, root):
    status, report, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", root)
    assert status == 0 and sorted(report["placed"]) == PLACED and report["repository"] == str(repository)
    assert report["repo_type"] == "dataset" and report["root"] == str(root) and not report["dry_run"]
    assert set(_files(root)) == FETCHED
    assert "placed   sampling/r1" in err and "placed 4 entries under" in err


def test_fetch_takes_experiments_and_roots_by_name_and_an_alias(cli, repository, root):
    status, report, _ = cli("data", "fetch", "prompt_ablation", "--dataset", repository, "--root", root)
    assert status == 0 and report["placed"] == ["sampling/r1"]
    status, report, _ = cli("data", "fetch", "grpo-x", "--dataset", repository, "--root", root)
    assert status == 0 and sorted(report["placed"]) == ["anchors/run/confirm", "sampling/grpo-x",
                                                        "selection/run/merge_manifest.json"]


def test_fetch_fills_the_configured_artifact_root_by_default(cli, repository, tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    status, report, _ = cli("data", "fetch", "training", "--dataset", repository)
    assert status == 0 and report["root"] == str(tmp_path / "artifacts")
    assert (tmp_path / "artifacts/sampling/grpo-x/sampling_manifest.json").is_file()


def test_fetch_without_an_artifact_root_is_refused(cli, repository, monkeypatch):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    status, _, err = cli("data", "fetch", "--all", "--dataset", repository)
    assert status == 2 and "no artifact root: pass --root DIR, or set RULEFORGE_ARTIFACT_ROOT" in err


@pytest.mark.parametrize("verb", ["fetch", "verify"])
@pytest.mark.parametrize(("names", "message"), [
    (("ruleset-size", "--all"), "name the experiments or roots (EXPERIMENT|ROOT ...), or pass --all for every one -- not both"),
    (("dropof", "r9"), "dropof, r9: neither an experiment of "),
])
def test_a_selection_is_names_or_all(cli, repository, root, verb, names, message):
    status, out, err = cli("data", verb, *names, "--dataset", repository, "--root", root)
    assert status == 2 and message in err and out == "" and "Traceback" not in err and "--only" not in err
    assert list(root.iterdir()) == [] and not root.with_name("root.runs").exists()


def test_nothing_named_is_every_entry_as_hub_fetch_py_took_it(cli, repository, root):
    status, report, _ = cli("data", "fetch", "--dataset", repository, "--root", root)
    assert status == 0 and sorted(report["placed"]) == PLACED and set(_files(root)) == FETCHED
    status, report, _ = cli("data", "verify", "--dataset", repository, "--root", root)
    assert status == 0 and sorted(report["verified"]) == PLACED and report["absent"] == []


def test_a_second_fetch_into_the_same_root_is_refused(cli, repository, root):
    assert cli("data", "fetch", "--all", "--dataset", repository, "--root", root)[0] == 0
    before = _tree(root)
    status, out, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", root)
    assert status == 1 and out == "" and "nothing is fetched over a root" in err and "Traceback" not in err
    assert _tree(root) == before


def test_a_tampered_archive_is_refused_and_nothing_is_placed(cli, repository, root):
    archive = repository / "prompt-ablation/roots/r1.tar.zst"
    archive.write_bytes(archive.read_bytes()[:-1] + bytes([archive.read_bytes()[-1] ^ 1]))
    status, _, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", root)
    assert status == 1 and "r1.tar.zst: not the bytes SHA256SUMS lists" in err
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("damage", ["a-flipped-byte", "a-truncation"])
def test_an_archive_that_does_not_unpack_is_refused_though_sha256sums_lists_its_bytes(cli, repository, root, damage):
    """A damaged upload: the archive's bytes changed and the repository's listings rewritten to them."""
    archive = repository / "prompt-ablation/roots/r1.tar.zst"
    raw = bytearray(archive.read_bytes())
    if damage == "a-flipped-byte":
        raw[0] ^= 0xFF  # the zstd magic: a flip further in can land on a byte no member depends on
    archive.write_bytes(bytes(raw if damage == "a-flipped-byte" else raw[: len(raw) // 2]))
    payload = json.loads((repository / hub.INDEX).read_text())
    [entry] = [entry for entry in payload["files"] if entry["path"] == "prompt-ablation/roots/r1.tar.zst"]
    entry.update(sha256=hub.sha256_file(archive), bytes=archive.stat().st_size)
    (repository / hub.INDEX).write_text(json.dumps(payload))
    (repository / hub.SUMS).write_text(hub.write_sums({path: hub.sha256_file(repository / path)
                                                      for path in hub._listing(repository) - {hub.SUMS}}))
    status, out, err = cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", root)
    assert status == 1 and out == "" and "Traceback" not in err
    assert "prompt-ablation/roots/r1.tar.zst, an archive with the bytes its SHA256SUMS lists, cannot be unpacked" in err
    assert "the repository is damaged" in err
    assert not (root / "sampling/r1").exists() and not list(root.rglob(".*fetching-*"))


def test_a_dry_run_writes_nothing(cli, repository, root):
    status, report, err = cli("data", "fetch", "--all", "--dry-run", "--dataset", repository, "--root", root)
    assert status == 0 and report["dry_run"] and report["placed"] == [] and list(root.iterdir()) == []
    assert "dry run: 4 entries checked, nothing written" in err and not root.with_name("root.runs").exists()


def _record(root: Path) -> Path:
    [record] = sorted(root.with_name(f"{root.name}.runs").iterdir())
    return record


def test_fetch_records_its_run_beside_the_root(cli, repository, root):
    argv = ["data", "fetch", "--all", "--dataset", str(repository), "--root", str(root)]
    status, report, _ = cli(*argv)
    assert status == 0 and report["revision"] is None and report["commit"] is None
    record = _record(root)
    assert record.name.endswith("-data-fetch") and not list(root.glob("*.runs"))
    argv_record = json.loads((record / "argv.json").read_text())
    assert argv_record["argv"] == argv and argv_record["subcommand"] == "data fetch" and "package_version" in argv_record
    log = (record / "log.txt").read_text()
    assert f"reading {repository} (dataset, a directory)" in log and "placed   sampling/r1" in log
    assert "placed 4 entries under" in log


def test_a_refused_fetch_keeps_its_record_with_why_it_stopped(cli, repository, root):
    assert cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", root)[0] == 0
    status, out, err = cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", root)
    assert status == 1 and out == "" and "Traceback" not in err
    records = sorted(root.with_name("root.runs").iterdir())
    assert len(records) == 2
    log = (records[-1] / "log.txt").read_text()
    assert f"data fetch: its run's record is {records[-1]}" in log
    assert "the run ended on HubError: sampling/r1 exists under" in log and "data fetch stopped" not in log
    assert err.count("sampling/r1 exists under") == 1


def test_a_refused_dry_run_makes_no_record(cli, repository, root):
    assert cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", root)[0] == 0
    status, _, err = cli("data", "fetch", "prompt-ablation", "--dry-run", "--dataset", repository, "--root", root)
    assert status == 2 and "nothing is fetched over a root" in err and "its run's record" not in err
    assert len(list(root.with_name("root.runs").iterdir())) == 1


def _campaign_root(path: Path, name: str = "sampling_manifest.json") -> Path:
    _write(path / name, "{}")
    return path


@pytest.mark.parametrize("place", ["the-root", "a-root-inside-one", "a-marked-root", "a-destination-s-directory"])
def test_fetch_writes_into_no_campaign_root(cli, repository, tmp_path, place):
    root = {"the-root": lambda: _campaign_root(tmp_path / "published"),
            "a-root-inside-one": lambda: _campaign_root(tmp_path / "published") / "below",
            "a-marked-root": lambda: _campaign_root(tmp_path / "published", ".ruleforge-campaign"),
            "a-destination-s-directory": lambda: _campaign_root(tmp_path / "root" / "anchors" / "run").parents[1]}[place]()
    root.mkdir(parents=True, exist_ok=True)
    before = _tree(tmp_path)
    status, out, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", root)
    assert status == 2 and out == "" and "Traceback" not in err
    named = {"a-root-inside-one": f"the artifact root {root} is in the campaign root {tmp_path / 'published'}",
             "a-destination-s-directory": f"anchors/run/confirm's directory {root / 'anchors/run'} is the campaign root"}
    assert named.get(place, f"the artifact root {root} is the campaign root {root}") in err
    assert "which nothing but its own campaign writes" in err and _tree(tmp_path) == before


def test_a_root_that_is_a_symlink_is_held_to_what_it_resolves_to(cli, repository, root, tmp_path):
    """An artifact root may be a link (``ruleforge.core.paths`` resolves one)."""
    published = _campaign_root(tmp_path / "published")
    (tmp_path / "into").symlink_to(published, target_is_directory=True)
    status, _, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", tmp_path / "into")
    assert status == 2 and f"the artifact root {tmp_path / 'into'} is the campaign root {published}" in err
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]
    assert not (tmp_path / "into.runs").exists()
    (tmp_path / "link").symlink_to(root, target_is_directory=True)
    status, report, _ = cli("data", "fetch", "--all", "--dataset", repository, "--root", tmp_path / "link")
    assert status == 0 and report["root"] == str(tmp_path / "link") and set(_files(root)) == FETCHED
    assert len(list((tmp_path / "link.runs").iterdir())) == 1


@pytest.mark.parametrize("verb", ["fetch", "verify"])
def test_an_artifact_root_that_names_two_places_is_refused(cli, repository, root, tmp_path, verb):
    """``--root <safe>/link/../root``: written out, ``<safe>/root``."""
    (tmp_path / "safe").mkdir()
    (tmp_path / "elsewhere/child").mkdir(parents=True)
    (tmp_path / "safe/link").symlink_to(tmp_path / "elsewhere/child", target_is_directory=True)
    (tmp_path / "safe/root").mkdir()
    given = f"{tmp_path}/safe/link/../root"
    status, out, err = cli("data", verb, "--all", "--dataset", repository, "--root", given)
    assert status == 2 and out == "" and "Traceback" not in err
    assert f"the artifact root {given} names two places: {tmp_path / 'safe/root'} as written out, and " in err
    assert f"{Path(os.path.realpath(tmp_path / 'elsewhere')) / 'root'} where the kernel takes it" in err
    assert list((tmp_path / "safe/root").iterdir()) == [] and list((tmp_path / "elsewhere").iterdir()) == [
        tmp_path / "elsewhere/child"] and not (tmp_path / "safe/root.runs").exists()


def test_fetch_refuses_a_destination_under_a_symlink_before_it_writes(cli, repository, root, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (root / "sampling").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    status, _, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", root)
    assert status == 2 and "sampling is a symlink or not a directory" in err
    assert not root.with_name("root.runs").exists() and list((tmp_path / "elsewhere").iterdir()) == []


@pytest.mark.parametrize(("only", "ancestor", "entry", "dest"), [
    ("prompt-ablation", "sampling", "prompt-ablation/roots/r1.tar.zst", "sampling/r1"),
    ("training", "selection/run", "training/selection/merge_manifest.json", "selection/run/merge_manifest.json"),
], ids=["an-archive", "a-file"])
def test_a_campaign_root_that_appears_while_an_entry_is_staged_is_refused_before_it_is_moved(
        cli, repository, root, monkeypatch, only, ancestor, entry, dest):
    """Each destination is checked again as it is written: a campaign root made above it after the plan."""
    real = hub._copy_hashing

    def racing(source, handle):
        if (root / ancestor).is_dir() and not (root / ancestor / "sampling_manifest.json").exists():
            _campaign_root(root / ancestor)
        return real(source, handle)
    monkeypatch.setattr(hub, "_copy_hashing", racing)
    status, out, err = cli("data", "fetch", only, "--dataset", repository, "--root", root)
    assert status == 1 and out == "" and "Traceback" not in err
    assert (f"the destination of {entry} {root / dest} is in the campaign root {root / ancestor}, which nothing but "
            "its own campaign writes") in err
    assert sorted(path.name for path in (root / ancestor).iterdir()) == ["sampling_manifest.json"]
    assert "the run ended on HubError: data fetch: the destination of" in (_record(root) / "log.txt").read_text()


def test_fetch_fills_an_artifact_root_beside_its_campaign_roots_and_on_the_pinned_roots_path(cli, repository,
                                                                                              tmp_path, monkeypatch):
    from ruleforge.evaluate import api

    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/r1": ("leniency",),
                                                       "artifact://sampling/grpo-x": ("training",)})
    with pytest.raises(api.CampaignError, match="a published run of"):
        api.refuse_protected(artifacts, "the artifact root")
    assert cli("data", "fetch", "prompt-ablation", "--dataset", repository)[0] == 0
    _campaign_root(artifacts / "sampling/r1")
    status, report, _ = cli("data", "fetch", "training", "--dataset", repository)
    assert status == 0 and report["root"] == str(artifacts) and "sampling/grpo-x" in report["placed"]
    assert len(list((tmp_path / "artifacts.runs").iterdir())) == 2


@pytest.mark.parametrize("verb", ["fetch", "verify"])
@pytest.mark.parametrize("field", ["members_sha256", "members_digest", "root"])
def test_an_index_field_of_the_wrong_type_is_a_damaged_repository(cli, repository, root, verb, field):
    """An ``index.json`` ``SHA256SUMS`` lists, with a digest that is no string or a root that is no name."""
    payload = json.loads((repository / hub.INDEX).read_text())
    [entry] = [entry for entry in payload["files"] if entry["path"] == "prompt-ablation/roots/r1.tar.zst"]
    if field == "root":
        entry.update(root=["r1"], dest="sampling/['r1']", prefix="sampling/['r1']")
    else:
        entry[field] = 5
    (repository / hub.INDEX).write_text(json.dumps(payload))
    (repository / hub.SUMS).write_text(hub.write_sums({path: hub.sha256_file(repository / path)
                                                      for path in hub._listing(repository) - {hub.SUMS}}))
    status, out, err = cli("data", verb, "prompt-ablation", "--dataset", repository, "--root", root)
    assert status == 2 and out == "" and "Traceback" not in err
    assert "its index.json, the one SHA256SUMS lists, is not an index" in err and "the repository is damaged" in err
    assert list(root.iterdir()) == [] and not root.with_name("root.runs").exists()


def test_fetch_and_verify_read_a_model_repository(cli, model, root):
    status, report, _ = cli("data", "fetch", "m/checkpoint-1", "--model", model, "--root", root)
    assert status == 0 and report["repo_type"] == "model" and len(report["placed"]) == 2
    assert (root / "selection/run/adapters/c1/adapter_config.json").read_text() == "the adapter_config.json"
    status, report, _ = cli("data", "verify", "--all", "--model", model, "--root", root)
    assert status == 0 and report["files"] == 2 and report["absent"] == []


def test_a_repository_is_a_dataset_or_a_model_not_both(cli, repository, model, root):
    with pytest.raises(SystemExit) as exited:
        cli("data", "fetch", "--all", "--dataset", repository, "--model", model, "--root", root)
    assert exited.value.code == 2


def test_fetch_without_the_hub_extra_is_refused_naming_it(cli, repository, root, monkeypatch):
    monkeypatch.setitem(sys.modules, "zstandard", None)
    status, _, err = cli("data", "fetch", "--all", "--dataset", repository, "--root", root)
    assert status == 2 and ("data fetch needs zstandard, which this environment lacks: add --group dev to uv sync, or "
                            "pip install -e '.[hub]' from the checkout") in err
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    status, _, err = cli("data", "fetch", "--all", "--root", root)
    assert status == 2 and "data fetch needs zstandard, huggingface_hub, which" in err
    assert list(root.iterdir()) == []


@pytest.fixture
def fetched(cli, repository, root) -> Path:
    assert cli("data", "fetch", "--all", "--dataset", repository, "--root", root)[0] == 0
    return root


def test_verify_checks_every_entry_and_writes_nothing(cli, repository, fetched):
    before = _tree(fetched)
    status, report, err = cli("data", "verify", "--all", "--dataset", repository, "--root", fetched)
    assert status == 0 and sorted(report["verified"]) == PLACED and report["absent"] == []
    assert report["files"] == len(FETCHED) and report["root"] == str(fetched)
    assert "verified sampling/r1" in err and f"verified 4 entries under {fetched} ({len(FETCHED)} files)" in err
    assert _tree(fetched) == before


def test_verify_checks_the_part_a_partial_fetch_placed(cli, repository, root):
    assert cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", root)[0] == 0
    status, report, err = cli("data", "verify", "--all", "--dataset", repository, "--root", root)
    assert status == 0 and report["verified"] == ["sampling/r1"] and len(report["absent"]) == 3
    assert "absent   sampling/grpo-x" in err and "; 3 not there" in err
    status, _, err = cli("data", "verify", "training", "--dataset", repository, "--root", root)
    assert status == 2 and "none of the 3 entries named is under" in err


def test_verify_with_nothing_there_is_refused(cli, repository, root):
    status, _, err = cli("data", "verify", "--all", "--dataset", repository, "--root", root)
    assert status == 2 and "none of the 4 entries named is under" in err and "nothing to verify" in err


@pytest.mark.parametrize(("tamper", "named"), [
    (lambda root: _write(root / "sampling/r1/aggregate/estimates.json", '{"rate": 0.6}'),
     "sampling/r1/aggregate/estimates.json: other bytes than prompt-ablation/roots/r1.tar.zst's"),
    (lambda root: _write(root / "sampling/r1/m/naive/draw-000/attempts.jsonl", "a dropped file, back\n"),
     "sampling/r1/m/naive/draw-000/attempts.jsonl: not a file of prompt-ablation/roots/r1.tar.zst"),
    (lambda root: (root / "sampling/grpo-x/sampling_manifest.json").unlink(),
     "sampling/grpo-x/sampling_manifest.json: missing"),
    (lambda root: (root / "sampling/r1/link").symlink_to(root / "sampling/r1/aggregate"), "sampling/r1/link: a symlink"),
    (lambda root: _write(root / "selection/run/merge_manifest.json", '{"weights": "abd"}'),
     "selection/run/merge_manifest.json: other bytes than training/selection/merge_manifest.json's"),
    (lambda root: ((root / "selection/run/merge_manifest.json").unlink(),
                   (root / "selection/run/merge_manifest.json").mkdir()),
     "selection/run/merge_manifest.json: not a regular file"),
    (lambda root: (os.rename(root / "anchors/run/confirm", root / "moved"), _write(root / "anchors/run/confirm", "x")),
     "anchors/run/confirm: not a directory"),
], ids=["a-changed-byte", "an-extra-file", "a-missing-file", "a-symlink", "a-changed-file", "a-directory-for-a-file",
        "a-file-for-a-directory"])
def test_verify_fails_naming_each_file_that_is_not_the_repository_s(cli, repository, fetched, tamper, named):
    tamper(fetched)
    status, out, err = cli("data", "verify", "--all", "--dataset", repository, "--root", fetched)
    assert status == 2 and out == "" and "Traceback" not in err
    assert "1 file(s) under" in err and named in err and "differs " in err


def test_verify_refuses_a_special_file_under_a_destination(cli, repository, fetched):
    os.mkfifo(fetched / "sampling/r1/fifo")
    status, _, err = cli("data", "verify", "prompt-ablation", "--dataset", repository, "--root", fetched)
    assert status == 2 and "fifo: not a regular file or a directory" in err


def test_verify_names_the_first_files_wrong_and_counts_the_rest(cli, repository, fetched):
    for number in range(hub.SHOWN + 3):
        _write(fetched / f"sampling/r1/extra/{number:02d}.json", "{}")
    status, _, err = cli("data", "verify", "prompt-ablation", "--dataset", repository, "--root", fetched)
    assert status == 2 and f"{hub.SHOWN + 3} file(s) under" in err and "; and 3 more" in err
    assert "extra/19.json" in err and "extra/20.json" not in err


def test_verify_refuses_a_destination_under_a_symlink(cli, repository, root, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (root / "sampling").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    status, _, err = cli("data", "verify", "--all", "--dataset", repository, "--root", root)
    assert status == 2 and "sampling is a symlink or not a directory" in err


def test_verify_reads_an_index_only_as_sha256sums_lists_it(cli, repository, fetched):
    (repository / hub.INDEX).write_text((repository / hub.INDEX).read_text() + " ")
    status, _, err = cli("data", "verify", "--all", "--dataset", repository, "--root", fetched)
    assert status == 2 and "index.json is not the one SHA256SUMS lists" in err


def test_verify_holds_each_members_manifest_to_sha256sums(cli, repository, fetched):
    _write(repository / "prompt-ablation/roots/r1.files.sha256", "")
    status, _, err = cli("data", "verify", "--all", "--dataset", repository, "--root", fetched)
    assert status == 2 and "r1.files.sha256: not the bytes SHA256SUMS lists" in err


def test_verify_takes_its_root_by_flag_alone(cli, repository):
    with pytest.raises(SystemExit) as exited:
        cli("data", "verify", "--all", "--dataset", repository)
    assert exited.value.code == 2


COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub_client(monkeypatch):
    """The Hub client's reads served from a local repository, each download recorded."""
    import types

    import huggingface_hub

    class Client:
        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.revisions: list[tuple] = []
            self.served = Path()

        def repo_info(self, repo_id, *, repo_type, revision, token):
            self.revisions.append((repo_id, repo_type, revision, token))
            return types.SimpleNamespace(sha=COMMIT)

        def serve(self, repository: Path) -> "Client":
            self.served = repository
            return self

        def snapshot_download(self, *, repo_id, repo_type, revision, local_dir, allow_patterns, token):
            self.calls.append((repo_id, repo_type, revision, sorted(allow_patterns), token, Path(local_dir)))
            for pattern in allow_patterns:
                _write(Path(local_dir) / pattern, (self.served / pattern).read_bytes())
            _write(Path(local_dir) / ".cache/huggingface/x.metadata", "x")

        def list_repo_files(self, repo_id, *, repo_type, revision, token):
            return sorted(hub._listing(self.served) | {".gitattributes"})

    client = Client()
    monkeypatch.setattr(huggingface_hub, "repo_info", client.repo_info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", client.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "list_repo_files", client.list_repo_files)
    return client


def test_fetch_reads_the_paper_s_dataset_on_the_hub_by_default(cli, hub_client, repository, root):
    hub_client.serve(repository)
    status, report, _ = cli("data", "fetch", "prompt-ablation", "--revision", "v1", "--root", root)
    assert status == 0 and report["repository"] == data.DATASET and report["placed"] == ["sampling/r1"]
    assert report["revision"] == "v1" and report["commit"] == COMMIT
    assert hub_client.revisions == [(data.DATASET, "dataset", "v1", False)]
    assert [call[:5] for call in hub_client.calls] == [
        (data.DATASET, "dataset", COMMIT, [hub.SUMS, hub.INDEX], False),
        (data.DATASET, "dataset", COMMIT, [hub.SUMS, hub.INDEX], False),
        (data.DATASET, "dataset", COMMIT, ["README.md", "prompt-ablation/roots/r1.files.sha256",
                                           "prompt-ablation/roots/r1.tar.zst"], False)]
    reference, download = hub_client.calls[0][5], hub_client.calls[1][5]
    assert not reference.is_relative_to(root) and not reference.exists()
    assert download.parent == root and not download.exists()
    assert sorted(path.name for path in root.iterdir()) == ["sampling"]
    assert f"reading {data.DATASET} (dataset, commit {COMMIT})" in (_record(root) / "log.txt").read_text()


def test_a_download_directory_in_a_campaign_root_is_refused(cli, hub_client, repository, root, tmp_path):
    hub_client.serve(repository)
    published = _campaign_root(tmp_path / "published")
    status, _, err = cli("data", "fetch", "--all", "--root", root, "--download-dir", published / "downloads")
    assert status == 2 and f"the download directory {published / 'downloads'} is in the campaign root {published}" in err
    assert list(root.iterdir()) == [] and sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]
    assert len(hub_client.calls) == 1


def test_a_download_directory_that_names_two_places_is_refused(cli, hub_client, repository, root, tmp_path):
    """``--download-dir <safe>/link/../downloads``, the link into a campaign root."""
    hub_client.serve(repository)
    campaign = _campaign_root(tmp_path / "campaign")
    (campaign / "child").mkdir()
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe/link").symlink_to(campaign / "child", target_is_directory=True)
    given = f"{tmp_path}/safe/link/../downloads"
    status, out, err = cli("data", "fetch", "--all", "--root", root, "--download-dir", given)
    assert status == 2 and out == "" and "Traceback" not in err
    assert (f"data fetch: the download directory {given} names two places: {tmp_path / 'safe/downloads'} as written "
            f"out, and {Path(os.path.realpath(campaign)) / 'downloads'} where the kernel takes it") in err
    assert sorted(path.name for path in campaign.iterdir()) == ["child", "sampling_manifest.json"]
    assert list(root.iterdir()) == [] and not (tmp_path / "safe/downloads").exists()
    assert not root.with_name("root.runs").exists() and len(hub_client.calls) == 1


def test_the_download_is_held_where_the_download_directory_was_checked(cli, hub_client, repository, root, tmp_path,
                                                                       monkeypatch):
    hub_client.serve(repository)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    status, report, _ = cli("data", "fetch", "prompt-ablation", "--root", root, "--download-dir", "~/dl")
    assert status == 0 and report["placed"] == ["sampling/r1"]
    assert hub_client.calls[1][5].parent == tmp_path / "home/dl" and not (Path.cwd() / "~").exists()


def test_a_root_given_with_a_tilde_is_the_home_directory_s(cli, repository, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home/artifacts").mkdir(parents=True)
    status, report, _ = cli("data", "fetch", "--dataset", repository, "--root", "~/artifacts")
    assert status == 0 and sorted(report["placed"]) == PLACED and set(_files(tmp_path / "home/artifacts")) == FETCHED
    status, report, _ = cli("data", "verify", "--dataset", repository, "--root", "~/artifacts")
    assert status == 0 and sorted(report["verified"]) == PLACED and not (Path.cwd() / "~").exists()


@pytest.mark.parametrize("flag", ["--dataset", "--model"])
def test_a_local_repository_given_with_a_tilde_or_relative_is_that_directory_not_a_hub_id(
        flag, cli, repository, model, root, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    source = repository if flag == "--dataset" else model
    for given in (f"~/{source.name}", os.path.relpath(source)):
        assert cli("data", "verify", flag, given, "--root", root)[0] == 2
        status, report, err = cli("data", "fetch", flag, given, "--root", root, "--dry-run")
        assert status == 0 and report["repository"] == str(source) and report["commit"] is None, err
    from ruleforge.cli import data

    for given in ("org/name", "~/no-such-repo"):
        parsed = types.SimpleNamespace(dataset=given, model=None, verb="fetch")
        assert data._repository(parsed) == (given, "dataset")


@pytest.mark.parametrize(("flag", "verb"), [("--dataset", "fetch"), ("--dataset", "verify"), ("--model", "fetch"),
                                            ("--model", "verify"), ("--root", "fetch"), ("--root", "verify"),
                                            ("--download-dir", "fetch")])
@pytest.mark.parametrize("spelled", ["relative", "absolute"])
def test_a_place_whose_dotdot_follows_a_symlink_is_refused_not_taken_lexically(
        flag, verb, spelled, cli, hub_client, repository, model, tmp_path, monkeypatch):
    """``link/../X`` from ``here/``, ``here/link`` a link to ``releases/v2/subdir``: written out, ``here/X``."""
    work, release = tmp_path / "here", tmp_path / "releases" / "v2"
    (release / "subdir").mkdir(parents=True)
    work.mkdir()
    (work / "link").symlink_to(release / "subdir", target_is_directory=True)
    name = {"--dataset": "dataset", "--model": "model", "--root": "root", "--download-dir": "downloads"}[flag]
    if flag in ("--dataset", "--model"):
        source = repository if flag == "--dataset" else model
        for top in (work, release):
            shutil.copytree(source, top / name)
    if flag != "--download-dir":
        (work / "root").mkdir()
        (release / "root").mkdir(exist_ok=True)
    hub_client.serve(repository)
    monkeypatch.chdir(work)
    given = f"link/../{name}" if spelled == "relative" else f"{work}/link/../{name}"
    source_flags = {"--dataset": [], "--model": []}.get(flag, ["--dataset", str(repository)])
    if flag == "--download-dir":
        source_flags = []  # the paper's dataset, served from the Hub: a download is made
    argv = ["data", verb, *source_flags, flag, given]
    if flag != "--root":
        argv += ["--root", str(work / ("root" if flag != "--download-dir" else "artifacts"))]
        (work / "artifacts").mkdir(exist_ok=True)
    before = {top: _tree(top) for top in (work, tmp_path / "releases")}
    status, out, err = cli(*argv)
    what = {"--dataset": "the dataset repository", "--model": "the model repository", "--root": "the artifact root",
            "--download-dir": "the download directory"}[flag]
    assert status == 2 and out == "" and "Traceback" not in err
    assert (f"data {verb}: {what} {given} names two places: {work / name} as written out, and "
            f"{Path(os.path.realpath(release)) / name} where the kernel takes it") in err, err
    assert {top: _tree(top) for top in (work, tmp_path / "releases")} == before
    assert not list(work.glob("*.runs")) and len(hub_client.calls) <= 1


def test_a_download_directory_in_an_entry_s_destination_is_refused(cli, hub_client, repository, root):
    hub_client.serve(repository)
    status, out, err = cli("data", "fetch", "prompt-ablation", "--root", root,
                           "--download-dir", root / "sampling/r1/dl")
    assert status == 2 and out == "" and "Traceback" not in err
    assert (f"the download directory {root / 'sampling/r1/dl'} is inside {root / 'sampling/r1'}, where "
            "prompt-ablation/roots/r1.tar.zst is placed: choose a place outside every entry's destination") in err
    assert list(root.iterdir()) == [] and len(hub_client.calls) == 1


# A fetched root is a published copy of a campaign root: a temporary directory in it is refused as one.
@pytest.mark.parametrize(("verb", "place", "named"), [
    (("verify",), lambda root, tmp_path: (root / "sampling/r1", root), "which nothing but its own campaign writes"),
    (("fetch", "training"), lambda root, tmp_path: (root, root), "is the artifact root"),
    (("fetch", "training", "--dry-run"), lambda root, tmp_path: (root / "tmp", root), "is inside the artifact root"),
    (("verify",), lambda root, tmp_path: (root / "sampling/r1", tmp_path / "link"),
     "which nothing but its own campaign writes"),
    (("verify",), lambda root, tmp_path: (_campaign_root(tmp_path / "published") / "tmp", root),
     "is in the campaign root"),
    (("fetch", "training"), lambda root, tmp_path: (_campaign_root(tmp_path / "published"), root), "is the campaign root"),
], ids=["verify-in-an-entry", "fetch-at-the-root", "dry-run-in-the-root", "verify-in-the-root-a-link-names",
        "verify-in-a-campaign-root", "fetch-at-a-campaign-root"])
def test_a_temporary_directory_in_the_artifact_root_or_a_campaign_root_is_refused(cli, hub_client, repository, root,
                                                                                tmp_path, monkeypatch, verb, place,
                                                                                named):
    assert cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", root)[0] == 0
    (tmp_path / "link").symlink_to(root, target_is_directory=True)
    hub_client.serve(repository)
    temporary, given = place(root, tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    before = _tree(tmp_path)
    status, out, err = cli("data", *verb, "--root", given)
    assert status == 2 and out == "" and "Traceback" not in err
    assert "the temporary directory" in err and named in err and str(temporary) in err
    assert _tree(tmp_path) == before and hub_client.calls == []


def test_a_hub_repository_s_reference_and_a_dry_run_s_download_are_held_in_the_temporary_directory(
        cli, hub_client, repository, root, tmp_path, monkeypatch):
    hub_client.serve(repository)
    (tmp_path / "scratch").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "scratch"))
    status, report, _ = cli("data", "fetch", "--all", "--dry-run", "--root", root)
    assert status == 0 and report["dry_run"] and list(root.iterdir()) == []
    assert [call[5].parent for call in hub_client.calls] == [tmp_path / "scratch"] * 2
    assert list((tmp_path / "scratch").iterdir()) == []


LONG_AGO = 10**18


@pytest.mark.parametrize("verb", [("fetch", "training"), ("verify",)], ids=["fetch", "verify"])
def test_a_tmpdir_in_a_campaign_root_is_refused_before_anything_is_tried_or_written_there(cli, hub_client, repository,
                                                                                          root, tmp_path, monkeypatch,
                                                                                          verb):
    hub_client.serve(repository)
    campaign = _campaign_root(tmp_path / "published")
    (campaign / "tmp").mkdir()
    for path in (campaign, campaign / "tmp"):
        os.utime(path, ns=(LONG_AGO, LONG_AGO))
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(campaign / "tmp"))
    before = {path: (sorted(os.listdir(path)), path.stat().st_mtime_ns) for path in (campaign, campaign / "tmp")}
    status, out, err = cli("data", *verb, "--root", root)
    assert status == 2 and out == "" and "Traceback" not in err
    assert (f"data {verb[0]}: the temporary directory (TMPDIR) {campaign / 'tmp'} is in the campaign root "
            f"{campaign}") in err
    assert {path: (sorted(os.listdir(path)), path.stat().st_mtime_ns) for path in before} == before
    assert hub_client.calls == [] and list(root.iterdir()) == []


@pytest.mark.parametrize("verb", [("fetch", "training"), ("verify",)], ids=["fetch", "verify"])
def test_a_campaign_root_made_in_the_temporary_directory_during_the_commit_lookup_is_refused_before_the_reference_is(
        cli, hub_client, repository, root, tmp_path, monkeypatch, verb):
    import huggingface_hub

    hub_client.serve(repository)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def lookup(*args, **kwargs):
        _campaign_root(scratch)
        return hub_client.repo_info(*args, **kwargs)
    monkeypatch.setattr(huggingface_hub, "repo_info", lookup)
    status, out, err = cli("data", *verb, "--root", root)
    assert status == 2 and out == "" and "Traceback" not in err
    assert f"data {verb[0]}: the temporary directory (TMPDIR) {scratch} is the campaign root {scratch}" in err
    assert sorted(path.name for path in scratch.iterdir()) == ["sampling_manifest.json"]
    assert hub_client.revisions and hub_client.calls == [] and list(root.iterdir()) == []


def test_a_campaign_root_made_while_verify_s_reference_downloads_is_refused_before_its_manifests_are(
        cli, hub_client, repository, fetched, tmp_path, monkeypatch):
    import huggingface_hub

    hub_client.serve(repository)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def download(**kwargs):
        hub_client.snapshot_download(**kwargs)
        _campaign_root(scratch)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    status, out, err = cli("data", "verify", "--all", "--root", fetched)
    assert status == 2 and out == "" and "Traceback" not in err
    assert "data verify: the directory the repository's reference is held in" in err
    assert f"is in the campaign root {scratch}, which nothing but its own campaign writes" in err
    assert [call[3] for call in hub_client.calls] == [[hub.SUMS, hub.INDEX]]
    assert sorted(path.name for path in scratch.iterdir()) == ["sampling_manifest.json"]


def test_a_campaign_root_that_appears_during_the_download_is_refused_at_placement(cli, hub_client, repository, root,
                                                                                monkeypatch):
    import huggingface_hub

    hub_client.serve(repository)

    def download(**kwargs):
        hub_client.snapshot_download(**kwargs)
        if "prompt-ablation/roots/r1.tar.zst" in kwargs["allow_patterns"]:
            _campaign_root(root / "sampling")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    status, out, err = cli("data", "fetch", "prompt-ablation", "--root", root)
    assert status == 1 and out == "" and "Traceback" not in err
    assert (f"the destination of prompt-ablation/roots/r1.tar.zst {root / 'sampling/r1'} is in the campaign root "
            f"{root / 'sampling'}") in err
    assert sorted(path.name for path in (root / "sampling").iterdir()) == ["sampling_manifest.json"]
    assert sorted(path.name for path in root.iterdir()) == ["sampling"]


def test_a_hub_that_names_no_commit_is_refused(cli, hub_client, root, monkeypatch):
    import types

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "repo_info", lambda *args, **kwargs: types.SimpleNamespace(sha=None))
    status, _, err = cli("data", "fetch", "--all", "--root", root)
    assert status == 2 and "the Hub names no commit for revision main" in err and hub_client.calls == []


def test_verify_downloads_the_reference_alone_into_a_temporary_directory(cli, hub_client, repository, fetched):
    hub_client.serve(repository)
    before = _tree(fetched)
    status, report, _ = cli("data", "verify", "--all", "--root", fetched)
    assert status == 0 and report["repository"] == data.DATASET and sorted(report["verified"]) == PLACED
    downloaded = [call[3] for call in hub_client.calls]
    assert {call[2] for call in hub_client.calls} == {COMMIT} and report["commit"] == COMMIT
    assert downloaded == [[hub.SUMS, hub.INDEX], ["prompt-ablation/roots/r1.files.sha256",
                                                   "training/inputs/anchors.files.sha256", "training/roots/grpo-x.files.sha256"]]
    reference = hub_client.calls[0][5]
    assert not reference.exists() and not reference.is_relative_to(fetched) and _tree(fetched) == before


def test_verify_of_plain_files_on_the_hub_downloads_no_manifest(cli, hub_client, model, root):
    assert cli("data", "fetch", "--all", "--model", model, "--root", root)[0] == 0
    hub_client.serve(model)
    status, report, _ = cli("data", "verify", "--all", "--model", "org/adapters", "--root", root)
    assert status == 0 and report["repo_type"] == "model" and report["files"] == 2
    assert [call[:4] for call in hub_client.calls] == [("org/adapters", "model", COMMIT, [hub.SUMS, hub.INDEX])]
    assert hub_client.revisions == [("org/adapters", "model", None, False)] and report["revision"] is None


def test_a_hub_that_refuses_is_a_refusal(cli, monkeypatch, root):
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    def refused(**kwargs):
        response = httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co/api/datasets/x"))
        raise RepositoryNotFoundError("401 Client Error: Repository Not Found", response=response)

    monkeypatch.setattr(huggingface_hub, "repo_info", lambda *args, **kwargs: refused())
    monkeypatch.setattr(huggingface_hub, "snapshot_download", refused)
    for verb in ("fetch", "verify"):
        status, out, err = cli("data", verb, "--all", "--root", root)
        assert status == 2 and out == "" and "RepositoryNotFoundError" in err and "Traceback" not in err
    assert list(root.iterdir()) == []


def test_verify_of_a_hub_repository_without_the_hub_extra_is_refused_naming_it(cli, monkeypatch, root):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    status, _, err = cli("data", "verify", "--all", "--root", root)
    assert status == 2 and "data verify needs huggingface_hub, which this environment lacks" in err


def test_the_command_line_and_a_local_verify_need_no_hub_extra(cli, repository, fetched, tmp_path):
    probe = ("import contextlib, sys\nsys.modules['zstandard'] = sys.modules['huggingface_hub'] = None\n"
             "from ruleforge.cli import main\n"
             "for argv in (['--help'], ['data', '--help'], ['data', 'fetch', '--help']):\n"
             "    try:\n        with contextlib.redirect_stdout(sys.stderr):\n            main(argv)\n"
             "    except SystemExit as exited:\n        assert exited.code == 0, argv\n"
             "sys.exit(main(sys.argv[1:]))\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    run = [sys.executable, "-c", probe, "data"]
    verified = subprocess.run([*run, "verify", "--all", "--dataset", str(repository), "--root", str(fetched)],
                              capture_output=True, text=True, env=env, cwd=tmp_path)
    assert verified.returncode == 0, verified.stderr
    assert sorted(json.loads(verified.stdout)["verified"]) == PLACED and "usage: ruleforge data fetch" in verified.stderr
    (tmp_path / "empty").mkdir()
    fetched_again = subprocess.run([*run, "fetch", "--all", "--dataset", str(repository), "--root",
                                    str(tmp_path / "empty")], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert fetched_again.returncode == 2 and "pip install -e '.[hub]'" in fetched_again.stderr
    assert "Traceback" not in fetched_again.stderr and list((tmp_path / "empty").iterdir()) == []


def test_the_default_dataset_is_the_one_the_inventory_names():
    inventory = yaml.safe_load((REPO / "data/hub_inventory.yaml").read_text(encoding="utf-8"))
    assert data.DATASET == inventory["dataset"]
    assert set(data.ALIASES.values()) <= set(inventory["experiments"])


CARDS = REPO / "data/hub_cards"


def _card_commands(text: str) -> list[list[str]]:
    lines = re.findall(r"^uv run ruleforge (.+)$", text, re.MULTILINE)
    return [shlex.split(re.sub(r"\[[^]]*\]", "", line), comments=True) for line in lines]


def test_the_dataset_card_names_every_folder_and_fetches_verifies_and_rebuilds_with_the_command_line():
    from ruleforge.cli.app import build_parser

    text = (CARDS / "dataset.md").read_text(encoding="utf-8")
    inventory = hub.load_inventory(REPO / "data/hub_inventory.yaml")
    assert all(f"| `{name}/` |" in text for name in inventory["experiments"])
    commands = _card_commands(text)
    assert [argv[:2] for argv in commands] == [["data", "fetch"], ["data", "fetch"], ["data", "verify"],
                                                ["reproduce", "all"]]
    assert not [word for argv in commands for word in argv if word.startswith("[") or word.endswith("]")]
    for argv in commands:
        build_parser().parse_args(argv)
    assert "scripts/" not in text and re.search(r"\.(?:py|sh)\b", text) is None


def test_the_model_card_names_no_script():
    text = (CARDS / "model.md").read_text(encoding="utf-8")
    assert "scripts/" not in text and _card_commands(text) == [] and "ai-law-society-lab/inconsistent-judges" in text


def test_each_experiment_fetches_one_folder_of_the_dataset():
    from ruleforge.cli.reproduce import EXPERIMENTS

    inventory = yaml.safe_load((REPO / "data" / "hub_inventory.yaml").read_text(encoding="utf-8"))
    folders = [data.FOLDERS.get(name, name) for name in EXPERIMENTS]
    assert set(folders) <= set(inventory["experiments"])
    assert data.FOLDERS["leniency"] == "prompt-ablation"
    assert [name for name in EXPERIMENTS if data.FOLDERS.get(name, name) != name] == ["leniency"]


def test_fetching_leniency_places_prompt_ablation_s_folder(cli, repository, tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "other").mkdir()
    status, leniency, error = cli("data", "fetch", "leniency", "--dataset", repository, "--root", tmp_path / "one")
    assert status == 0, error
    status, ablation, error = cli("data", "fetch", "prompt-ablation", "--dataset", repository, "--root", tmp_path / "other")
    assert status == 0, error
    assert leniency["placed"] == ablation["placed"] and leniency["placed"]
