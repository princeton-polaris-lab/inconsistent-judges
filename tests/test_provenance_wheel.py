from __future__ import annotations

import hashlib
import tarfile
from importlib import metadata
from pathlib import Path

import pytest

from ruleforge.core import provenance


def _installed(site: Path, version: str = "9.9") -> metadata.Distribution:
    """A distribution as a wheel installs one: its modules, its bytecode, a console script and its metadata."""
    package = site / "ruleforge"
    (package / "core" / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "core" / "provenance.py").write_text("x = 1\n")
    (package / "core" / "__pycache__" / "provenance.cpython-312.pyc").write_bytes(b"\0")
    (package / "configs").mkdir()
    (package / "configs" / "evaluate.yaml").write_text("a: 1\n")
    info = site / f"ruleforge-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: ruleforge\nVersion: {version}\n")
    (site.parent.parent / "bin").mkdir(parents=True)
    (site.parent.parent / "bin" / "ruleforge").write_text("#!/bin/sh\n")
    (site / "ruleforge.pth").write_text("")
    listed = ["ruleforge/__init__.py", "ruleforge/core/provenance.py",
              "ruleforge/core/__pycache__/provenance.cpython-312.pyc", "ruleforge/configs/evaluate.yaml",
              "ruleforge/gone.py", f"ruleforge-{version}.dist-info/METADATA", f"ruleforge-{version}.dist-info/RECORD",
              "../../bin/ruleforge", "ruleforge.pth"]
    (info / "RECORD").write_text("".join(f"{name},,\n" for name in listed))
    return metadata.Distribution.at(info)


def test_an_installed_package_s_sources_are_its_own_files(tmp_path):
    dist = _installed(tmp_path / "lib" / "site")
    assert [relative for relative, _ in provenance.installed_source_files(dist)] == [
        "ruleforge/__init__.py", "ruleforge/configs/evaluate.yaml", "ruleforge/core/provenance.py"]


def test_its_code_state_has_no_commit_counts_as_dirty_and_digests_its_files(tmp_path):
    dist = _installed(tmp_path / "lib" / "site")
    state = provenance.installed_code_state(dist)
    lines = "".join(f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}\n"
                    for relative, path in provenance.installed_source_files(dist))
    assert state == {"code_revision": None, "code_dirty": True,
                     "code_state_sha256": hashlib.sha256(lines.encode()).hexdigest()}
    assert provenance.installed_code_state(None) == {"code_revision": None, "code_dirty": True,
                                                     "code_state_sha256": None}


def test_editing_one_byte_of_an_installed_module_changes_the_digest_and_the_snapshot(tmp_path, monkeypatch):
    dist = _installed(tmp_path / "lib" / "site")
    monkeypatch.setattr(provenance, "checkout_root", lambda: None)
    monkeypatch.setattr(provenance, "distribution", lambda: dist)
    provenance._imported_code_state.cache_clear()
    try:
        before = provenance.code_state()
        first = provenance.write_code_snapshot(tmp_path / "campaign", before)
        module = tmp_path / "lib" / "site" / "ruleforge" / "core" / "provenance.py"
        module.write_text("x = 2\n")
        provenance._imported_code_state.cache_clear()
        after = provenance.code_state()
        second = provenance.write_code_snapshot(tmp_path / "campaign", after)
    finally:
        provenance._imported_code_state.cache_clear()
    assert before["code_state_sha256"] != after["code_state_sha256"] and first != second
    with tarfile.open(first) as archive:
        assert archive.getnames() == ["ruleforge/__init__.py", "ruleforge/configs/evaluate.yaml",
                                      "ruleforge/core/provenance.py"]
        assert archive.extractfile("ruleforge/core/provenance.py").read() == b"x = 1\n"
    with tarfile.open(second) as archive:
        assert archive.extractfile("ruleforge/core/provenance.py").read() == b"x = 2\n"


def test_with_nothing_installed_there_is_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "checkout_root", lambda: None)
    monkeypatch.setattr(provenance, "distribution", lambda: None)
    path = provenance.write_code_snapshot(tmp_path / "campaign", {"code_state_sha256": "a" * 64})
    with tarfile.open(path) as archive:
        assert archive.getnames() == []


def test_the_version_is_the_installed_distribution_s(monkeypatch):
    assert provenance.package_version() == metadata.version("ruleforge")
    assert provenance.distribution().metadata["Name"] == "ruleforge"

    monkeypatch.setattr(provenance.metadata, "distributions", lambda **kwargs: iter(()))
    assert provenance.package_version() is None and provenance.distribution() is None


def test_a_stale_egg_info_beside_the_package_never_names_the_version(tmp_path, monkeypatch):
    """A stale ``src/ruleforge.egg-info`` ahead on ``sys.path`` is passed over for the installed ``*.dist-info``."""
    src, site = tmp_path / "src", tmp_path / "site-packages"
    (src / "ruleforge" / "core").mkdir(parents=True)
    (src / "ruleforge.egg-info").mkdir()
    (src / "ruleforge.egg-info" / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: ruleforge\nVersion: 0.0.1\n")
    (site / "ruleforge-1.0.0.dist-info").mkdir(parents=True)
    (site / "ruleforge-1.0.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ruleforge\nVersion: 1.0.0\n")
    found = [metadata.PathDistribution(src / "ruleforge.egg-info"),
             metadata.PathDistribution(site / "ruleforge-1.0.0.dist-info")]
    monkeypatch.setattr(provenance.metadata, "distributions", lambda **kwargs: iter(found))
    monkeypatch.setattr(provenance, "__file__", str(src / "ruleforge" / "core" / "provenance.py"))
    assert provenance.package_version() == "1.0.0"
    monkeypatch.setattr(provenance.metadata, "distributions", lambda **kwargs: iter(found[:1]))
    assert provenance.package_version() is None


def test_this_checkout_is_found_as_one():
    assert provenance.checkout_root() == provenance.repo_root()
    assert (provenance.checkout_root() / "pyproject.toml").is_file()


@pytest.mark.parametrize("field", ["package_version"])
def test_the_version_is_a_run_field_never_compared(field):
    from ruleforge.evaluate.identity import MANIFEST_RUN_FIELDS
    from ruleforge.evaluate.runner import PROVENANCE_RUN_FIELDS

    assert field in MANIFEST_RUN_FIELDS and field in PROVENANCE_RUN_FIELDS
