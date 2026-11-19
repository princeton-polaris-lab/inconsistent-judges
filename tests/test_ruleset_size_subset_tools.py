from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from ruleforge.experiments.ruleset_size import generate_subsets, verify_subsets
from ruleforge.rulesets.subsets import build_manifest, file_digest, strategy_root
from tests.support.subsets import render_manifest

N_RULES = 12
REPO_ROOT = Path(__file__).resolve().parents[1]


def _checkout(tmp_path: Path, **meta_overrides) -> Path:
    repo = tmp_path / "repo"
    rules = repo / "data" / "rules" / "toy.jsonl"
    rules.parent.mkdir(parents=True)
    rules.write_text("".join(json.dumps({"rule_id": n, "text": f"Rule {n}"}) + "\n" for n in range(1, N_RULES + 1)))
    manifest_dir = repo / "data" / "rules" / "subset_manifest"
    manifest_dir.mkdir(parents=True)
    meta = {"sampling_strategy": "with_replacement", "seeds": [0, 1], "files_manifest": "manifest.sha256",
            "source_ruleset": {"path": "data/rules/toy.jsonl", "sha256": file_digest(rules)}, **meta_overrides}
    (manifest_dir / "manifest.json").write_text(json.dumps(meta))
    (manifest_dir / "manifest.sha256").write_text("")
    return repo


@pytest.fixture
def repo(tmp_path, monkeypatch):
    repo = _checkout(tmp_path)
    for module in (generate_subsets, verify_subsets):
        monkeypatch.setattr(module, "repository_root", lambda: repo)
    return repo


def _generate_and_pin(repo: Path) -> Path:
    assert generate_subsets.main([]) == 0
    root = strategy_root(repo / generate_subsets.DEFAULT_OUTPUT, "with_replacement")
    (repo / verify_subsets.DEFAULT_MANIFEST_DIR / "manifest.sha256").write_text(render_manifest(build_manifest(root)))
    return root


def test_the_generator_writes_every_seed_s_subsets_under_the_checkout(repo, capsys):
    root = _generate_and_pin(repo)
    assert sorted(path.name for path in (root / "seed_0").iterdir()) == [
        f"subset_size_{size:02d}.jsonl" for size in range(2, N_RULES + 1)]
    out = capsys.readouterr().out
    assert f"seed 0: {N_RULES - 1} files" in out and f"{2 * (N_RULES - 1)} files under {root}" in out


def test_the_verifier_accepts_the_regenerated_subsets(repo, capsys):
    _generate_and_pin(repo)
    capsys.readouterr()
    assert verify_subsets.main([]) == 0
    assert f"All {2 * (N_RULES - 1)} subset files match the manifest." in capsys.readouterr().out


def test_the_verifier_lists_changed_files_and_counts_the_rest(repo, capsys):
    root = _generate_and_pin(repo)
    for path in sorted((root / "seed_0").iterdir()):
        path.write_text("tampered\n")
    capsys.readouterr()
    assert verify_subsets.main([]) == 1
    out = capsys.readouterr().out
    assert f"changed: {N_RULES - 1}" in out and "... and 1 more" in out


def test_the_verifier_lists_a_short_difference_whole(repo, capsys):
    root = _generate_and_pin(repo)
    (root / "seed_1" / "subset_size_02.jsonl").unlink()
    capsys.readouterr()
    assert verify_subsets.main([]) == 1
    out = capsys.readouterr().out
    assert "missing: 1" in out and "and" not in out.split("missing: 1")[1]


def test_the_verifier_has_nothing_to_check_before_the_generator_ran(repo, capsys):
    assert verify_subsets.main([]) == 2
    assert "run `ruleforge reproduce ruleset-size subsets` first" in capsys.readouterr().out


def test_the_generator_names_a_manifest_directory_as_the_verifier_does(repo, capsys):
    with pytest.raises(SystemExit):
        generate_subsets.main(["--manifest", str(repo / "data" / "rules" / "subset_manifest" / "other.json")])
    assert "unrecognized arguments: --manifest" in capsys.readouterr().err


def test_explicit_places_override_the_checkout_s(repo, tmp_path, capsys):
    out = tmp_path / "elsewhere"
    manifest = repo / "data" / "rules" / "subset_manifest" / "manifest.json"
    assert generate_subsets.main(["--manifest-dir", str(manifest.parent), "--output-root", str(out)]) == 0
    root = strategy_root(out, "with_replacement")
    (manifest.parent / "manifest.sha256").write_text(render_manifest(build_manifest(root)))
    assert verify_subsets.main(["--manifest-dir", str(manifest.parent), "--output-root", str(out)]) == 0
    assert not (repo / generate_subsets.DEFAULT_OUTPUT).exists()


def test_the_generator_never_writes_into_the_checkout_s_committed_directories(repo, tmp_path, capsys):
    (tmp_path / "linked").symlink_to(repo / "results")
    (repo / "results" / "elsewhere").mkdir(parents=True)
    (tmp_path / "hop").symlink_to(repo / "results" / "elsewhere")
    for given, message in [(repo / "results" / "subsets", "the checkout's results/"),
                           (tmp_path / "linked" / "subsets", "the checkout's results/"),
                           (f"{tmp_path / 'hop'}/../subsets", "names two places")]:
        with pytest.raises(SystemExit) as refused:
            generate_subsets.main(["--output-root", str(given)])
        assert refused.value.code == 2 and message in capsys.readouterr().err, given
    assert sorted(path.name for path in (repo / "results").rglob("*")) == ["elsewhere"]


def test_a_ruleset_that_does_not_match_the_manifest_is_refused_before_writing(repo):
    (repo / "data" / "rules" / "toy.jsonl").write_text('{"rule_id": 1}\n{"rule_id": 2}\n')
    with pytest.raises(SystemExit, match="does not match the manifest"):
        generate_subsets.main([])
    assert not (repo / generate_subsets.DEFAULT_OUTPUT).exists()


def test_a_manifest_requiring_another_python_is_refused(tmp_path, monkeypatch):
    repo = _checkout(tmp_path, reproducibility={"python_requires": ">=2.7,<2.8"})
    monkeypatch.setattr(generate_subsets, "repository_root", lambda: repo)
    with pytest.raises(SystemExit, match=f"running {sys.version_info.major}.{sys.version_info.minor}"):
        generate_subsets.main([])
    assert not (repo / generate_subsets.DEFAULT_OUTPUT).exists()


def test_the_committed_manifest_refuses_another_python_and_takes_its_own(tmp_path, monkeypatch):
    committed = json.loads((REPO_ROOT / generate_subsets.DEFAULT_MANIFEST_DIR / "manifest.json").read_text(
        encoding="utf-8"))
    required = committed["reproducibility"]["python_requires"]
    repo = _checkout(tmp_path, reproducibility={"python_requires": required})
    monkeypatch.setattr(generate_subsets, "repository_root", lambda: repo)
    for version in [(3, 11, 9), (3, 13, 1)]:
        monkeypatch.setattr(generate_subsets.sys, "version_info", version)
        with pytest.raises(SystemExit, match=f"requires Python {re.escape(required)}, running {re.escape('.'.join(map(str, version)))}"):
            generate_subsets.main([])
    monkeypatch.setattr(generate_subsets.sys, "version_info", (3, 12, 13))
    assert generate_subsets.main([]) == 0


def test_a_manifest_whose_python_requirement_is_no_specifier_is_refused(tmp_path, monkeypatch):
    repo = _checkout(tmp_path, reproducibility={"python_requires": "three-twelve"})
    monkeypatch.setattr(generate_subsets, "repository_root", lambda: repo)
    with pytest.raises(SystemExit, match="'three-twelve' is no version specifier"):
        generate_subsets.main([])


def test_a_manifest_requiring_this_python_generates(tmp_path, monkeypatch):
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    repo = _checkout(tmp_path, reproducibility={"python_requires": f"=={current}.*"})
    monkeypatch.setattr(generate_subsets, "repository_root", lambda: repo)
    assert generate_subsets.main([]) == 0


def test_the_generator_alone_never_writes_through_a_link_below_its_output_root(repo, tmp_path, capsys):
    out = tmp_path / "subsets"
    target = repo / "data" / "rules" / "subset_manifest" / "manifest.json"
    before = target.read_bytes()
    linked = strategy_root(out, "with_replacement") / "seed_0" / "subset_size_02.jsonl"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(target)
    with pytest.raises(SystemExit) as refused:
        generate_subsets.main(["--output-root", str(out)])
    assert refused.value.code == 2 and "symlink" in capsys.readouterr().err
    assert target.read_bytes() == before and sorted(path.name for path in out.rglob("*")) == [
        "seed_0", "subset_size_02.jsonl", "with_replacement"]
