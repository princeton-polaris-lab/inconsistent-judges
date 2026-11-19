from __future__ import annotations

import io
import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from ruleforge.core import provenance
from ruleforge.core.provenance import (
    SNAPSHOT_DIRNAME,
    CheckoutMismatch,
    code_state,
    repo_root,
    require_checkout,
    snapshot_path,
    source_files,
    write_code_snapshot,
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A committed checkout with sources in each directory the code state covers."""
    root = tmp_path / "repo"
    for relative, text in {
        "src/pkg/a.py": "print('a')\n",
        "configs/c.yaml": "x: 1\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "version = 1\n",
        "docs/ignored_by_dir.md": "not a source dir\n",
        "scripts/run.sh": "not a source dir either\n",
        ".gitignore": "*.log\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def test_a_clean_tree_reports_head_and_a_content_digest(tree):
    state = code_state(tree)
    assert state["code_revision"] == git(tree, "rev-parse", "HEAD")
    assert state["code_dirty"] is False
    assert len(state["code_state_sha256"]) == 64
    assert code_state(tree) == state
    assert source_files(tree) == [Path("configs/c.yaml"), Path("pyproject.toml"), Path("src/pkg/a.py"),
                                  Path("uv.lock")]


def test_a_changed_lock_file_changes_the_digest(tree):
    before = code_state(tree)["code_state_sha256"]
    (tree / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    assert code_state(tree)["code_state_sha256"] != before


def test_untracked_sources_count_and_caches_and_ignored_files_do_not(tree):
    clean = code_state(tree)
    (tree / "src/pkg/__pycache__").mkdir()
    (tree / "src/pkg/__pycache__/a.cpython-312.pyc").write_bytes(b"\x00")
    (tree / "src/pkg/stale.pyc").write_bytes(b"\x00")
    (tree / "src/debug.log").write_text("ignored\n", encoding="utf-8")
    assert source_files(tree) == source_files(tree) and Path("src/pkg/stale.pyc") not in source_files(tree)
    assert code_state(tree)["code_state_sha256"] == clean["code_state_sha256"]

    (tree / "src/pkg/new_module.py").write_text("print('new')\n", encoding="utf-8")
    dirty = code_state(tree)
    assert dirty["code_dirty"] is True
    assert dirty["code_revision"] == clean["code_revision"]
    assert dirty["code_state_sha256"] != clean["code_state_sha256"]
    assert Path("src/pkg/new_module.py") in source_files(tree)


def test_the_digest_follows_content_not_mtime_and_a_deleted_file_drops_out(tree):
    before = code_state(tree)["code_state_sha256"]
    path = tree / "src/pkg/a.py"
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")  # touch only
    assert code_state(tree)["code_state_sha256"] == before
    path.unlink()
    assert Path("src/pkg/a.py") not in source_files(tree)
    assert code_state(tree)["code_state_sha256"] != before


def test_a_snapshot_archives_the_sources_once_per_digest(tree, tmp_path):
    campaign_root = tmp_path / "campaign"
    state = code_state(tree)
    archive = write_code_snapshot(campaign_root, state, root=tree)
    assert archive == snapshot_path(campaign_root, state["code_state_sha256"])
    assert archive.parent.name == SNAPSHOT_DIRNAME
    with tarfile.open(archive, "r:gz") as tar:
        names = sorted(tar.getnames())
        content = tar.extractfile("src/pkg/a.py").read()
    assert names == ["configs/c.yaml", "pyproject.toml", "src/pkg/a.py", "uv.lock"]
    assert content == b"print('a')\n"

    first_bytes = archive.read_bytes()
    assert write_code_snapshot(campaign_root, state, root=tree) == archive
    assert archive.read_bytes() == first_bytes
    (tree / "src/pkg/a.py").write_text("print('b')\n", encoding="utf-8")
    second = write_code_snapshot(campaign_root, code_state(tree), root=tree)
    assert second != archive and second.exists() and archive.exists()


def test_without_git_the_state_is_unknown_and_nothing_is_snapshotted(tmp_path, monkeypatch):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    state = code_state(plain)
    assert state == {"code_revision": None, "code_dirty": True, "code_state_sha256": None}
    assert write_code_snapshot(tmp_path / "campaign", state, root=plain) is None

    def boom(*args, **kwargs):
        raise OSError("no git binary")

    monkeypatch.setattr(provenance.subprocess, "run", boom)
    assert source_files(plain) is None


def test_the_default_root_is_this_checkout():
    root = repo_root()
    assert (root / "src" / "ruleforge" / "core" / "provenance.py").is_file()
    files = source_files()
    assert files is None or Path("src/ruleforge/core/provenance.py") in files


def test_a_process_runs_only_the_checkout_whose_entry_point_it_runs(tmp_path):
    assert require_checkout(repo_root()) == repo_root()
    link = tmp_path / "link"
    link.symlink_to(repo_root())
    assert require_checkout(link) == repo_root()
    other = tmp_path / "other-checkout"
    other.mkdir()
    with pytest.raises(CheckoutMismatch,
                       match=re.escape(f"imported ruleforge from {repo_root()}, not from {other.resolve()}, the checkout")):
        require_checkout(other)


@pytest.mark.parametrize("module, after", [
    ("ruleforge.launch.programs.run_campaign", "HydraConfig was not set"),
    ("ruleforge.launch.programs.optimize", "is not an OmegaConf config object"),
])
def test_a_submitit_job_of_a_program_runs_only_the_checkout_that_launched_it(tmp_path, module, after):
    import cloudpickle

    root = repo_root()
    program = runpy.run_module(module, run_name=f"entry_{module.rsplit('.', 1)[1]}")  # no importable name: by value
    pickled = tmp_path / "main.pkl"
    pickled.write_bytes(cloudpickle.dumps(program["run"]))  # what the launcher submits
    other = tmp_path / "other-checkout"
    shutil.copytree(root / "src" / "ruleforge", other / "src" / "ruleforge", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(root / "pyproject.toml", other / "pyproject.toml")
    job = "import pickle, sys; pickle.loads(open(sys.argv[1], 'rb').read())(None)"

    def run_job(src: Path) -> subprocess.CompletedProcess:
        # PYTHONPATH stands in for an environment's binding: it puts src first, as an editable install's .pth would.
        return subprocess.run([sys.executable, "-c", job, str(pickled)], cwd=tmp_path, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(src)})

    refused = run_job(other / "src")
    assert refused.returncode != 0 and "CheckoutMismatch" in refused.stderr
    assert f"imported ruleforge from {other.resolve()}, not from {root}" in refused.stderr
    passed = run_job(root / "src")  # past the check, the job fails only for want of a Hydra run (no config)
    assert passed.returncode != 0 and "CheckoutMismatch" not in passed.stderr
    assert after in passed.stderr  # the task function's next line: the check returned


def test_a_snapshot_with_no_listable_sources_is_an_empty_archive(tree, tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "source_files", lambda root=None: None)
    archive = write_code_snapshot(tmp_path / "campaign", {"code_state_sha256": "f" * 64}, root=tree)
    with tarfile.open(archive, "r:gz") as tar:
        assert tar.getnames() == []
    buffer = io.BytesIO(archive.read_bytes())
    assert tarfile.is_tarfile(buffer) or archive.stat().st_size > 0


def test_a_snapshot_finished_by_another_process_is_kept_and_no_temp_file_remains(tmp_path, monkeypatch):
    from ruleforge.core import provenance

    tree = tmp_path / "tree"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "a.py").write_text("x = 1\n")
    (tree / "pyproject.toml").write_text("[project]\nname='t'\n")
    campaign_root = tmp_path / "campaign"
    state = {"code_state_sha256": "e" * 64}
    destination = provenance.snapshot_path(campaign_root, state["code_state_sha256"])
    real_source_files = provenance.source_files

    def other_process_wins_first(base):
        files = real_source_files(base)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"the other job's archive")
        return files

    monkeypatch.setattr(provenance, "source_files", other_process_wins_first)
    assert provenance.write_code_snapshot(campaign_root, state, root=tree) == destination
    assert destination.read_bytes() == b"the other job's archive"
    assert list(destination.parent.glob("*.tmp")) == []


@pytest.fixture
def fresh_memo():
    """An empty per-process memo of the imported checkout's state, before and after."""
    provenance._imported_code_state.cache_clear()
    yield
    provenance._imported_code_state.cache_clear()


def test_the_imported_checkout_is_hashed_once_per_process_and_an_explicit_root_every_time(tree, fresh_memo, monkeypatch):
    hashed = []
    real = provenance._code_state

    def counting(base):
        hashed.append(base)
        return real(base)

    monkeypatch.setattr(provenance, "_code_state", counting)
    first = code_state()
    first["code_state_sha256"] = "mutated by a caller"
    second = code_state()
    assert hashed == [repo_root()]  # a job planning 561 conditions hashes the tree once
    assert second["code_state_sha256"] != "mutated by a caller" and second == real(repo_root())
    before = code_state(tree)
    (tree / "src/pkg/a.py").write_text("print('edited')\n", encoding="utf-8")
    assert code_state(tree)["code_state_sha256"] != before["code_state_sha256"]
    assert hashed == [repo_root(), tree, tree]


def test_git_helpers_return_sane_types_outside_a_repo(tmp_path):
    assert provenance.git_revision(tmp_path) is None or isinstance(provenance.git_revision(tmp_path), str)
    assert isinstance(provenance.git_is_dirty(tmp_path), bool)


def test_both_git_helpers_degrade_the_same_way_without_git(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", absent)
    assert provenance.git_revision() is None
    assert provenance.git_is_dirty() is True


def test_a_failing_git_command_is_unknown_not_clean(monkeypatch):
    """`git status` exits 128 on a corrupt index while `rev-parse` exits 0."""
    monkeypatch.setattr(provenance.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], returncode=128, stdout="", stderr="not a repository"))
    assert provenance.git_revision() is None
    assert provenance.git_is_dirty() is True
