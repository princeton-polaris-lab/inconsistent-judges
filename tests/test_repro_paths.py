from __future__ import annotations

import os
from pathlib import Path

import pytest

from ruleforge.core.paths import (
    ARTIFACT_ROOT_ENV,
    REPOSITORY_SCHEME,
    absolute_strings,
    artifact_uri_or_path,
    is_artifact_uri,
    is_repository_uri,
    repository_root,
    resolve,
    to_uri,
    walk_strictly,
)
from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root
from tests.fake_scandir import failing_scandir, vanishing_scandir


@pytest.fixture
def root(tmp_path):
    (tmp_path / "ruleset-size" / "gemma").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def env(root):
    return {ARTIFACT_ROOT_ENV: str(root)}


def test_an_unset_root_raises_rather_than_defaulting(root):
    with pytest.raises(ArtifactRootError, match="is unset"):
        artifact_root({})


def test_an_empty_root_is_treated_as_unset():
    with pytest.raises(ArtifactRootError, match="is unset"):
        artifact_root({ARTIFACT_ROOT_ENV: ""})


def test_a_root_that_is_not_a_directory_raises(tmp_path):
    missing = tmp_path / "not-mounted"
    with pytest.raises(ArtifactRootError, match="not a directory"):
        artifact_root({ARTIFACT_ROOT_ENV: str(missing)})


def test_a_file_as_root_raises(tmp_path):
    handle = tmp_path / "f"
    handle.write_text("x")
    with pytest.raises(ArtifactRootError, match="not a directory"):
        artifact_root({ARTIFACT_ROOT_ENV: str(handle)})


def test_artifact_root_reads_the_process_environment_by_default(root, monkeypatch):
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(root))
    assert artifact_root() == root


def test_a_user_home_root_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert artifact_root({ARTIFACT_ROOT_ENV: "~"}) == tmp_path


def test_a_path_under_the_root_round_trips(root, env):
    target = root / "ruleset-size" / "gemma"
    uri = to_uri(target, env)
    assert uri == "artifact://ruleset-size/gemma"
    assert resolve(uri, env) == target


def test_a_path_outside_the_root_has_no_uri_form(root, env, tmp_path):
    outside = tmp_path.parent / "elsewhere"
    with pytest.raises(ValueError, match="not under"):
        to_uri(outside, env)


def test_a_provenance_record_takes_the_uri_where_one_exists_else_the_absolute_path(root, env, tmp_path, monkeypatch):
    assert artifact_uri_or_path(root / "ruleset-size" / "gemma", env) == "artifact://ruleset-size/gemma"
    outside = tmp_path.parent / "elsewhere"
    assert artifact_uri_or_path(outside, env) == str(outside.resolve())
    assert artifact_uri_or_path(root / "ruleset-size", {}) == str((root / "ruleset-size").resolve())
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv("HOME", str(tmp_path.parent))
    assert artifact_uri_or_path("elsewhere", env) == artifact_uri_or_path("~/elsewhere", env) == str(outside.resolve())


def test_is_artifact_uri_distinguishes_the_two_forms():
    assert is_artifact_uri("artifact://a/b") is True
    assert is_artifact_uri("/abs/a/b") is False


def test_a_plain_path_resolves_unchanged_without_a_root():
    assert resolve("/tmp/x", {}) == Path("/tmp/x")


def test_a_repository_uri_resolves_without_an_artifact_root():
    target = resolve("repo://results", {})
    assert target == repository_root() / "results"
    assert target.is_dir()


def test_repository_uri_detection_is_separate_from_artifacts():
    assert is_repository_uri(f"{REPOSITORY_SCHEME}results") is True
    assert is_artifact_uri(f"{REPOSITORY_SCHEME}results") is False


@pytest.mark.parametrize(
    "uri", ["repo://", "repo:///etc/passwd", "repo://../../etc"]
)
def test_a_repository_uri_cannot_escape_the_checkout(uri):
    with pytest.raises(ValueError):
        resolve(uri, {})


def test_a_uri_requires_the_root_even_when_a_plain_path_would_not():
    with pytest.raises(ArtifactRootError):
        resolve("artifact://a", {})


def test_a_uri_cannot_escape_the_root(env):
    with pytest.raises(ValueError, match="escapes the artifact root"):
        resolve("artifact://../../etc/passwd", env)


def test_a_uri_cannot_carry_an_absolute_path(env):
    with pytest.raises(ValueError, match="absolute path after the scheme"):
        resolve("artifact:///etc/passwd", env)


def test_the_root_itself_round_trips(root, env):
    assert resolve(to_uri(root, env), env) == root


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-0 directory")
def test_a_directory_the_walk_cannot_list_raises_and_is_never_left_out(tmp_path):
    """rglob passes over a directory it cannot list: a walk would describe the tree without it."""
    _write(tmp_path / "x.jsonl", b"one")
    _write(tmp_path / "sub" / "y.json", b"two")
    (tmp_path / "sub").chmod(0)
    try:
        with pytest.raises(PermissionError):
            list(walk_strictly(tmp_path))
    finally:
        (tmp_path / "sub").chmod(0o700)


def test_an_entry_whose_type_cannot_be_read_is_still_walked_into(tmp_path, monkeypatch):
    _write(tmp_path / "x.jsonl", b"one")
    _write(tmp_path / "sub" / "deeper" / "y.json", b"two")
    monkeypatch.setattr(os, "scandir", failing_scandir(is_dir_fails={"sub", "deeper"}))
    walked = sorted(path.relative_to(tmp_path).as_posix() for path, _ in walk_strictly(tmp_path))
    assert walked == ["sub", "sub/deeper", "sub/deeper/y.json", "x.jsonl"]


def test_the_walk_never_lists_a_directory_it_is_told_not_to_descend(tmp_path, monkeypatch):
    _write(tmp_path / "keep" / "a.json", b"1")
    _write(tmp_path / "draw-000" / "b.json", b"2")
    listed, real = [], os.scandir

    def recording(path="."):
        listed.append(Path(path))
        return real(path)

    monkeypatch.setattr(os, "scandir", recording)
    walked = walk_strictly(tmp_path, descend=lambda directory: not directory.name.startswith("draw-"))
    assert sorted(path.relative_to(tmp_path).as_posix() for path, _ in walked) == ["draw-000", "keep", "keep/a.json"]
    assert sorted(listed) == [tmp_path, tmp_path / "keep"]


def test_an_entry_gone_while_the_strict_walk_reaches_it_is_passed_over_only_when_asked(tmp_path, monkeypatch):
    def plant() -> None:
        _write(tmp_path / "keep" / "a.json", b"1")
        _write(tmp_path / "state.json.x.tmp", b"2")
        _write(tmp_path / "gone" / "b.json", b"3")

    plant()
    monkeypatch.setattr(os, "scandir", vanishing_scandir({"state.json.x.tmp"}, {tmp_path / "gone"}))
    walked = sorted(path.relative_to(tmp_path).as_posix() for path, _ in walk_strictly(tmp_path, missing_ok=True))
    assert walked == ["gone", "keep", "keep/a.json"]
    for after, before in (({"state.json.x.tmp"}, ()), ((), {tmp_path / "gone"})):
        plant()
        monkeypatch.setattr(os, "scandir", vanishing_scandir(after, before))
        with pytest.raises(FileNotFoundError):
            list(walk_strictly(tmp_path))


def test_a_symlinked_subdirectory_cannot_read_outside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside)
    env = {ARTIFACT_ROOT_ENV: str(root)}
    with pytest.raises(ValueError, match="outside the artifact root"):
        resolve("artifact://escape", env)


def test_a_symlink_staying_inside_the_root_resolves(tmp_path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "link").symlink_to(root / "real")
    env = {ARTIFACT_ROOT_ENV: str(root)}
    assert resolve("artifact://link", env) == (root / "real").resolve()


@pytest.mark.parametrize(
    "value",
    [
        "/home/user/weights",
        "--download-dir=/scratch/x",     # embedded, invisible to startswith
        "/.cache/huggingface/hub",
        "/-tmp/x",
        "//scratch/shared/weights",      # double slash: startswith catches it
        "/ home/user",                   # space after the slash: ditto
        "${oc.env:ROOT,/scratch/dflt}",  # default-bearing env form is banned
        "--cache-dir=/.cache/hub",       # embedded + dot: needs the widened
        "--tmp=/-scratch/x",             # regex class, startswith can't see it
        "after </think> at /scratch/x",  # a path beside a closing tag is still one
        "<code> /scratch/x",             # and so is one after an opening tag
        "</think>/scratch/x",            # and one straight after a closing tag
        "x</scratch/a",                  # a path straight after '<' is no tag
        "cat </scratch/x.jsonl",         # shell input redirection
        "</home/user/models>",
        "see </tmp/x>",
    ],
)
def test_the_absolute_string_predicate_catches_both_shape_families(value):
    assert list(absolute_strings({"k": value})) == [("k", value)]


@pytest.mark.parametrize(
    "value",
    [
        "https://host/path",
        "${oc.env:ROOT}/weights",        # sanctioned env reference
        "../../data/rules/x.jsonl",      # a relative form (task configs name the checkout's data/...)
        "artifact://ruleset-size/gemma",      # the committed reference form
        "repo://results/final_eval",     # committed compact provenance
        "/",                             # a bare slash is not a path
        "plain text with no path",
        "kappa_*/viol: seed-cluster",    # glob-star prose in a provenance note
        "inconsistency_graded/inconsistency_binary: two-way",    # metric-name prose, slash mid-word
        "its answer after its last </think>, and",
        "</think>",
        "</channel-end>",                # a tag name with a dash
    ],
)
def test_portable_strings_are_not_flagged(value):
    assert list(absolute_strings({"k": value})) == []


def test_the_walker_reports_the_key_path_through_nesting():
    payload = {"a": {"b": ["ok", {"c": "/abs/path"}]}}
    assert list(absolute_strings(payload)) == [("a.b[1].c", "/abs/path")]


def test_a_repository_symlink_cannot_read_outside_the_checkout(tmp_path, monkeypatch):
    from ruleforge.core import paths

    root = tmp_path / "checkout"
    (root / "results").mkdir(parents=True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (root / "results" / "link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(paths, "repository_root", lambda: root)
    with pytest.raises(ValueError, match="outside the repository root"):
        resolve(f"{REPOSITORY_SCHEME}results/link/golden.json")


def test_repository_inputs_need_a_source_checkout(tmp_path, monkeypatch):
    from ruleforge.core import paths

    installed = tmp_path / "site-packages" / "ruleforge" / "repro" / "paths.py"
    installed.parent.mkdir(parents=True)
    monkeypatch.setattr(paths, "__file__", str(installed))
    with pytest.raises(ArtifactRootError, match="outside a ruleforge source checkout"):
        repository_root()


def test_a_package_installed_inside_a_clone_runs_from_no_checkout(tmp_path, monkeypatch):
    from ruleforge.core import paths, provenance

    clone = tmp_path / "clone"
    (clone / "src" / "ruleforge").mkdir(parents=True)
    (clone / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    installed = clone / ".venv" / "lib" / "python3.12" / "site-packages" / "ruleforge" / "core" / "paths.py"
    installed.parent.mkdir(parents=True)
    monkeypatch.setattr(paths, "__file__", str(installed))
    assert paths.checkout_root() is None and provenance.checkout_root() is None
    for where in (repository_root, provenance.repo_root):
        with pytest.raises(ArtifactRootError, match="outside a ruleforge source checkout"):
            where()
    with pytest.raises(provenance.CheckoutMismatch, match=f"not from {clone}"):
        provenance.require_checkout(clone)
    monkeypatch.setattr(paths, "__file__", str(clone / "src" / "ruleforge" / "core" / "paths.py"))
    assert repository_root() == provenance.repo_root() == provenance.checkout_root() == clone


def test_second_place_is_where_the_kernel_takes_a_spelling_when_that_is_not_the_place_written_out(tmp_path,
                                                                                                  monkeypatch):
    from ruleforge.core.paths import second_place

    (tmp_path / "work" / "data").mkdir(parents=True)
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "work" / "link").symlink_to(tmp_path / "elsewhere" / "child")
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.setenv("HOME", str(tmp_path / "work"))
    reached = Path(os.path.realpath(tmp_path)) / "elsewhere" / "x"
    assert second_place("link/../x", tmp_path / "work" / "x") == reached
    assert second_place(str(tmp_path / "work" / "link" / ".." / "x"), tmp_path / "work" / "x") == reached
    assert second_place("~/data", tmp_path / "work" / "data") is None
    assert second_place("data", tmp_path / "work" / "data") is None


def test_made_on_the_way_is_the_first_missing_directory_a_dotdot_leaves(tmp_path, monkeypatch):
    from ruleforge.core.paths import made_on_the_way

    (tmp_path / "work" / "there").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "work" / "link").symlink_to(tmp_path / "elsewhere")
    monkeypatch.chdir(tmp_path / "work")
    work, real = tmp_path / "work", Path(os.path.realpath(tmp_path))
    assert made_on_the_way("new/../x") == real / "work" / "new"
    assert made_on_the_way(str(work / "there" / "new" / ".." / ".." / "x")) == real / "work" / "there" / "new"
    assert made_on_the_way("link/new/../x") == real / "elsewhere" / "new"
    assert made_on_the_way("there/../b/c/../d") == real / "work" / "b" / "c"
    assert made_on_the_way("a/../b/../x") == real / "work" / "a"
    assert made_on_the_way("there/../x/y") is None
    assert made_on_the_way("../x") is None and made_on_the_way("/..") is None
    assert made_on_the_way("new/./x") is None and made_on_the_way(Path("new")) is None
    (work / "dangling").symlink_to(tmp_path / "missing")
    assert made_on_the_way("dangling/../x") is None  # the link is there: the kernel follows it (and fails)


def test_one_place_takes_a_spelling_as_written_and_refuses_one_the_kernel_takes_elsewhere(tmp_path, monkeypatch):
    """``link/../x`` is ``./x`` written out and ``<link's target's parent>/x`` to the kernel: refused."""
    from ruleforge.core.paths import one_place

    (tmp_path / "work" / "data").mkdir(parents=True)
    (tmp_path / "work" / "data" / "rows.jsonl").write_text("{}\n")
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "work" / "link").symlink_to(tmp_path / "elsewhere" / "child")
    monkeypatch.chdir(tmp_path / "work")
    monkeypatch.setenv("HOME", str(tmp_path / "work"))
    assert one_place("data/./x/..") == tmp_path / "work" / "data"
    assert one_place("~/data") == one_place(Path("data")) == tmp_path / "work" / "data"
    assert one_place("missing/../new") == tmp_path / "work" / "new"
    assert one_place("data/rows.jsonl", kind="file") == tmp_path / "work" / "data" / "rows.jsonl"
    assert one_place("link", kind="directory") == tmp_path / "work" / "link"
    with pytest.raises(ValueError, match="names two places"):
        one_place("link/../child")
    for given, kind in [("missing/../data", "directory"), ("missing/../data/rows.jsonl", "file"),
                        ("data", "file"), ("data/rows.jsonl", "directory")]:
        with pytest.raises(ValueError, match=f"is no {kind} there"):
            one_place(given, kind=kind)
