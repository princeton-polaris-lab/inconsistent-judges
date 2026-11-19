from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from ruleforge.core.paths import absolute_strings
from tests.conftest import REPO_ROOT
from tests.support.leak_scan import Hit, Policy, Scanner, ThirdParty

THIRD_PARTY = (
    "data/populations/*.jsonl",
    "data/rules/modelspec/openai_rules_all*.jsonl",
    "data/scenarios/*/modelspec/*.jsonl",
    "tests/fixtures/rules/openai_subset_prolific.jsonl",
)
_LUNA_REQUEST_LINES = "tests/fixtures/batch/luna_ruleset_size_request_lines.json"
_LUNA_REQUEST_LINES_SHA256 = "365ed07b64847199e7d9a9052e42c85162923328b229682af4824294823f6717"
NOT_A_PATH = {
    **{(_LUNA_REQUEST_LINES, key): (_LUNA_REQUEST_LINES_SHA256,
                                    "the Responses API endpoint, a URL path inside the pinned request bytes")
       for key in ("endpoint", "cases[0].line", "cases[1].line", "cases[2].line", "cases[3].line")},
}
#: Top-level directories a walk skips: run outputs and the paper's site, not code.
WALK_SKIPS = frozenset({"experiments", "site"})
#: Files a test or coverage run writes at the root.
GENERATED = frozenset({"coverage.json"})


def walked_json(root: Path) -> list[str]:
    """``*.json`` / ``*.jsonl`` under ``root``, repository-relative, as a tree with no git index holds them."""
    found = []
    for directory, subdirectories, files in os.walk(root):
        top = Path(directory) == root
        subdirectories[:] = sorted(
            name for name in subdirectories
            if not name.startswith(".") and name != "__pycache__" and not name.endswith(".egg-info")
            and not (top and name in WALK_SKIPS))
        for name in files:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            if name.endswith((".json", ".jsonl")) and relative not in GENERATED:
                found.append(relative)
    return sorted(found)


def committed_json(root: Path) -> list[str]:
    """The committed ``*.json`` / ``*.jsonl``."""
    if not (root / ".git").exists():
        return walked_json(root)
    listed = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                             "*.json", "*.jsonl"], cwd=root, capture_output=True,
                            check=True).stdout.decode("utf-8").split("\0")
    return sorted(path for path in listed if path and (root / path).is_file())


def _sha256(relative: str) -> str:
    return hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()


def is_third_party(relative: str) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in THIRD_PARTY)


def _payloads(relative: str):
    """``(key prefix, payload)`` per JSON value: the file's one, or one per JSONL line."""
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    if relative.endswith(".json"):
        yield "", json.loads(text)
    else:
        for number, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                yield f"line {number}:", json.loads(line)


def _keys(node, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key)
            yield from _keys(value, here)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _keys(value, f"{path}[{index}]")


def absolute_paths(payload):
    """``(key path, string)`` for every literal absolute path in a payload, as a value or as a mapping key."""
    yield from absolute_strings(payload)
    for key_path, key in _keys(payload):
        for _, value in absolute_strings(key):
            yield f"{key_path} (key)", value


@pytest.fixture(scope="module")
def files() -> list[str]:
    """The committed JSON files this module checks: every one, with no exemption."""
    return committed_json(REPO_ROOT)


def test_the_walk_reaches_the_goldens_and_the_fixtures(files):
    from ruleforge.experiments import table

    goldens = {golden.path for golden in table.goldens()}
    assert goldens and goldens <= set(files)
    assert any(path.startswith("tests/fixtures/") for path in files)


def test_no_first_party_json_holds_an_absolute_path(files):
    offenders, excused = {}, set()
    for relative in files:
        if is_third_party(relative):
            continue
        for prefix, payload in _payloads(relative):
            for key, value in absolute_paths(payload):
                pinned = NOT_A_PATH.get((relative, key))
                if pinned is not None and pinned[0] == _sha256(relative):
                    excused.add((relative, key))
                else:
                    offenders.setdefault(relative, []).append(f"{prefix}{key}: {value[:120]}")
    assert offenders == {}, (
        "absolute paths in committed JSON: rewrite them as artifact:// URIs (ruleforge.core.paths.to_uri) or "
        f"repository-relative paths: {offenders}")
    assert excused == set(NOT_A_PATH), f"stale or lapsed NOT_A_PATH entries: {sorted(set(NOT_A_PATH) - excused)}"


def test_an_absolute_path_is_found_as_a_key_as_well_as_a_value():
    payload = {"inputs": {"/tmp/run/evaluations.jsonl": "abc", "rel/x.jsonl": "d"}, "out": ["/scratch/y"],
               "rows": [{"n": {"/home/user/z": 1}}]}
    assert sorted(absolute_paths(payload)) == [
        ("inputs./tmp/run/evaluations.jsonl (key)", "/tmp/run/evaluations.jsonl"), ("out[0]", "/scratch/y"),
        ("rows[0].n./home/user/z (key)", "/home/user/z")]


def _scan(files: list[str], policy: Policy) -> list[Hit]:
    scanner = Scanner(policy)
    for relative in files:
        scanner.scan_file(relative, (REPO_ROOT / relative).read_bytes())
    return scanner.result().hits


def test_committed_json_passes_the_scans_identifier_free_rules(files):
    third_party = tuple(ThirdParty(path, _sha256(path), "declared in THIRD_PARTY") for path in files
                        if is_third_party(path))
    hits = _scan(files, Policy(strict=(), third_party=third_party))
    assert hits == [], [hit.render() for hit in hits]


def test_a_walk_skips_what_is_not_committed_code(tmp_path):
    for relative in ("data/a.json", "results/b.jsonl", "c.json", "coverage.json", "notes.txt",
                     "experiments/run/out.json", "site/page.json", ".venv/lib/pkg.json",
                     "src/pkg/__pycache__/x.json", "src/pkg.egg-info/y.json",
                     "data/experiments/kept.json"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text("{}", encoding="utf-8")
    assert walked_json(tmp_path) == ["c.json", "data/a.json", "data/experiments/kept.json", "results/b.jsonl"]
    assert committed_json(tmp_path) == walked_json(tmp_path)  # no .git: the walk
