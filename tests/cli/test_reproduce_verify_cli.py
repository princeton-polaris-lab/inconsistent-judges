from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import ruleforge.evaluate.api  # noqa: F401 -- imported before any test patches paths.repository_root (below)
from ruleforge.cli import reproduce
from ruleforge.core import paths
from ruleforge.experiments import table
from ruleforge.experiments.table import Experiment, Golden, Step
from tests.conftest import REPO_ROOT
from tests.identity import published

#: Writes its argument list and the lock it holds to <output root>/probe.json, as the generation's stand-in.
PROBE = """\
import json, os, sys
out = sys.argv[sys.argv.index("--output-root") + 1]
os.makedirs(out, exist_ok=True)
json.dump({"argv": sys.argv[1:]}, open(os.path.join(out, "probe.json"), "w"))
"""


def _copies(tmp_path: Path, names) -> Path:
    tree = tmp_path / "tree"
    places = {golden.name: golden.path for golden in table.goldens()}
    for name in names:
        (tree / places[name]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / places[name], tree / places[name])
    return tree


def test_verify_checks_the_goldens_the_experiment_s_builds_write(cli, tmp_path):
    names = [golden.name for golden in table.goldens()]
    assert len(names) == len(published.GOLDENS)
    tree = _copies(tmp_path, names)
    status, _, _ = cli("reproduce", "all", "verify", "--rerun", tree, "--json", tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert status == 0 and report["ok"] and [golden["name"] for golden in report["goldens"]] == names
    status, _, _ = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--json", tmp_path / "ruleset-size.json")
    ruleset_size = json.loads((tmp_path / "ruleset-size.json").read_text())
    assert status == 0 and [golden["name"] for golden in ruleset_size["goldens"]] == [
        "ruleset_size_curves", "ruleset_size_repeated_draw_baseline"]


def test_verify_fails_a_serving_dtype_changed_in_every_stack_description_and_passes_a_version_change(cli, tmp_path):
    tree = _copies(tmp_path, ["training_gemma", "training_qwen", "grpo_judgment_changes", "grpo_frozen_anchors"])
    gemma = tree / "results" / "training" / "gemma.json"
    committed = gemma.read_text()
    gemma.write_text(committed.replace('engine_args.dtype=\\"float32\\"', 'engine_args.dtype=\\"bfloat16\\"'))
    status, _, _ = cli("reproduce", "training", "verify", "--rerun", tree, "--json", tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert status == 1 and sorted((finding["check"], finding["path"]) for finding in report["fatal"]) == [
        ("design", "provenance.arms.base.stack"), ("design", "provenance.arms.final.stack"),
        ("design", "provenance.engine.stack")]
    gemma.write_text(committed.replace("vllm 0.28.0+cu129", "vllm 0.29.1+cu130"))
    assert cli("reproduce", "training", "verify", "--rerun", tree)[0] == 0


def test_verify_reads_a_tree_by_its_artifact_uri_and_exits_1_when_a_golden_is_not_there(cli, tmp_path):
    tree = _copies(tmp_path / "artifacts", ["ruleforge_final_eval"])
    assert tree == tmp_path / "artifacts" / "tree"
    assert cli("reproduce", "ruleforge", "verify", "--rerun", "artifact://tree")[0] == 0
    status, _, _ = cli("reproduce", "ruleset-size", "verify", "--rerun", "artifact://tree", "--json", "report.json")
    report = json.loads((Path.cwd() / "report.json").read_text())
    assert status == 1 and not report["ok"] and {finding["check"] for finding in report["fatal"]} == {"missing"}
    status, printed, _ = cli("reproduce", "ruleforge", "verify", "--rerun", "artifact://tree", "--exact")
    assert status == 0 and printed.splitlines()[-1] == "ALL 1 EQUAL"


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """A toy checkout: one golden, ruleset-size's two builds (one writing it, one verifying), and no other experiment's."""
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    (repo / "results" / "golden.json").write_text('{"rate": 0.5, "n": 2}')
    builds = (Step("b", "python -c pass", golden=Golden("g", "results/golden.json", 0.0)),
              Step("v", "python -c pass", check_only=True))
    monkeypatch.setattr(table, "TABLE", (Experiment("ruleset-size", builds=builds),))
    monkeypatch.setattr(paths, "repository_root", lambda: repo)
    return repo


def test_verify_exact_compares_each_golden_as_parsed_json_and_reports_its_bytes(cli, toy, tmp_path):
    tree = tmp_path / "tree"
    (tree / "results").mkdir(parents=True)
    (tree / "results" / "golden.json").write_text('{"n": 2, "rate": 0.5}')
    status, printed, _ = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--exact")
    assert status == 0 and printed.splitlines() == ["EQUAL   results/golden.json  (bytes differ)",
                                                    "bytes: 0 of 1 identical", "ALL 1 EQUAL"]
    shutil.copyfile(toy / "results" / "golden.json", tree / "results" / "golden.json")
    status, printed, _ = cli("reproduce", "all", "verify", "--rerun", tree, "--exact")
    assert status == 0 and printed.splitlines()[:2] == ["EQUAL   results/golden.json", "bytes: 1 of 1 identical"]
    (tree / "results" / "golden.json").write_text('{"rate": 0.25, "n": 2}')
    status, printed, _ = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--exact")
    assert status == 1 and printed.splitlines()[0].startswith("DIFFERS results/golden.json")
    assert printed.splitlines()[-1] == "1 OF 1 DIFFER"
    (tree / "results" / "golden.json").unlink()
    status, printed, _ = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--exact")
    assert status == 1 and printed.splitlines()[0].startswith("MISSING results/golden.json")


def test_verify_refuses_the_checkout_itself_whatever_its_spelling(cli, toy, monkeypatch):
    for given in (toy, toy / "results" / ".."):
        status, printed, error = cli("reproduce", "ruleset-size", "verify", "--rerun", given, "--exact")
        assert status == 2 and "is this checkout" in error and printed == ""
    monkeypatch.chdir(toy)
    for given in (".", "results/.."):
        status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", given)
        assert status == 2 and "is this checkout" in error
    status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", "missing/..", "--exact")
    assert status == 2 and "is no directory" in error  # the kernel walks missing/: nothing there


def test_verify_refuses_the_real_checkout_as_its_own_rerun(cli):
    status, printed, error = cli("reproduce", "all", "verify", "--rerun", REPO_ROOT, "--exact")
    assert status == 2 and "is this checkout" in error and "EQUAL" not in printed


def test_verify_refuses_a_tree_holding_the_checkout_s_goldens_as_its_own(cli, toy, tmp_path):
    (tmp_path / "linked").mkdir()
    (tmp_path / "linked" / "results").symlink_to(toy / "results")
    status, printed, error = cli("reproduce", "ruleset-size", "verify", "--rerun", tmp_path / "linked", "--exact")
    assert status == 2 and "holds its results/golden.json" in error and printed == ""


def test_verify_refuses_a_tree_that_is_not_there_or_names_two_places(cli, toy, tmp_path):
    status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", tmp_path / "absent", "--exact")
    assert status == 2 and "is no directory" in error
    (tmp_path / "file").write_text("x")
    assert "is no directory" in cli("reproduce", "ruleset-size", "verify", "--rerun", tmp_path / "file")[2]
    status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", "artifact://absent", "--exact")
    assert status == 2 and "is no directory" in error
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "work" / "link").symlink_to(tmp_path / "elsewhere" / "child")
    status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", "link/..", "--exact")
    assert status == 2 and "names two places" in error


def test_verify_refuses_an_artifact_uri_it_cannot_resolve(cli, toy, monkeypatch):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", "artifact://tree", "--exact")
    assert status == 2 and "--rerun artifact://tree" in error


def test_verify_s_report_never_lands_on_a_golden_or_the_tree_s_goldens(cli, toy, tmp_path):
    tree = tmp_path / "tree"
    (tree / "results").mkdir(parents=True)
    (tree / "results" / "golden.json").write_text('{"n": 2, "rate": 0.5}')
    before = {path: path.read_bytes() for path in (toy / "results").iterdir()}
    for report, where in [(toy / "results" / "golden.json", "the checkout's results/"),
                          (tree / "results" / "golden.json", "the tree's results/"),
                          (tree / "results" / "report.json", "the tree's results/")]:
        status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--json", report)
        assert status == 2 and where in error, (report, error)
    assert {path: path.read_bytes() for path in (toy / "results").iterdir()} == before
    assert (tree / "results" / "golden.json").read_text() == '{"n": 2, "rate": 0.5}'
    assert not (tree / "results" / "report.json").exists()


@pytest.mark.parametrize("argv, message", [
    (["ruleset-size", "verify"], "--rerun DIR"),
    (["ruleset-size", "verify", "--rerun", "t", "--exact", "--json", "r.json"], "--exact prints its comparison alone"),
    (["ruleset-size", "verify", "--rerun", "t", "--out", "o"], "it takes no --out"),
    (["ruleset-size", "render", "--out", "o", "--rerun", "t"], "--rerun is verify's option: render takes none"),
    (["ruleset-size", "rebuild", "--out", "o", "--exact"], "--exact is verify's option: rebuild takes none"),
    (["ruleset-size", "rebuild", "--json", "r.json"], "--json is verify's option: rebuild takes none"),
    (["ruleset-size", "verify", "--rerun", "t", "--check"], "--check is subsets's option: verify takes none"),
    (["training", "subsets"], "it is ruleset-size's (ruleforge reproduce ruleset-size subsets), not training's"),
])
def test_each_level_refuses_another_level_s_options(cli, toy, argv, message):
    status, _, error = cli("reproduce", *argv)
    assert status == 2 and message in error


def test_subsets_check_runs_the_checker_in_the_checkout_and_writes_nothing(cli, tmp_path):
    (tmp_path / "empty").mkdir()
    status, _, _ = cli("reproduce", "ruleset-size", "subsets", "--check", "--out", tmp_path / "empty")
    assert status == 2
    assert list((tmp_path / "empty").iterdir()) == [] and not (tmp_path / "empty.runs").exists()


def test_subsets_check_refuses_a_place_the_kernel_cannot_walk_to(cli, tmp_path, monkeypatch):
    """``missing/../present`` is ``present`` written out, but names nothing: checked as ``present`` it would pass."""
    (tmp_path / "present").mkdir()
    monkeypatch.chdir(tmp_path)
    for given in ("missing/../present", "absent"):
        status, _, error = cli("reproduce", "ruleset-size", "subsets", "--check", "--out", given)
        assert status == 2 and f"--out {given} is no directory there" in error
    assert not (tmp_path / "missing").exists() and not (tmp_path / "present.runs").exists()


def test_subsets_generates_into_out_holding_its_lock_and_recording_its_run(cli, toy, tmp_path, monkeypatch):
    (toy / "src").mkdir()
    (toy / "src" / "probe_subsets.py").write_text(PROBE)
    monkeypatch.setitem(reproduce.SUBSETS_PROGRAMS, False, "probe_subsets")
    status, _, _ = cli("reproduce", "ruleset-size", "subsets", "--out", "made")
    out = Path.cwd() / "made"
    assert status == 0 and json.loads((out / "probe.json").read_text()) == {"argv": ["--output-root", str(out)]}
    runs = Path.cwd() / "made.runs"
    assert (runs / reproduce.LOCK_FNAME).is_file()
    [record] = [path for path in runs.iterdir() if path.is_dir()]
    assert record.name.endswith("-reproduce-subsets")
    status, _, _ = cli("reproduce", "ruleset-size", "subsets")  # the default: the checkout's data/rules/ruleset_size_subsets
    assert status == 0 and (toy / reproduce.SUBSETS_OUT / "probe.json").is_file()


def test_subsets_refuses_an_out_among_the_committed_goldens(cli, toy):
    status, _, error = cli("reproduce", "ruleset-size", "subsets", "--out", toy / "results" / "subsets")
    assert status == 2 and "results/" in error and not (toy / "results" / "subsets").exists()


def test_a_report_or_subsets_place_spelling_two_places_is_refused(cli, toy, tmp_path):
    tree = tmp_path / "tree"
    (tree / "results").mkdir(parents=True)
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "work" / "link").symlink_to(tmp_path / "elsewhere" / "child")
    status, _, error = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--json", "link/../report.json")
    assert status == 2 and "names two places" in error and not (tmp_path / "elsewhere" / "report.json").exists()
    for check in ([], ["--check"]):
        status, _, error = cli("reproduce", "ruleset-size", "subsets", "--out", "link/../subsets", *check)
        assert status == 2 and "names two places" in error and not (tmp_path / "elsewhere" / "subsets").exists()


def test_a_render_or_rebuild_out_spelling_two_places_is_refused(cli, tmp_path):
    (tmp_path / "elsewhere" / "child").mkdir(parents=True)
    (tmp_path / "work" / "link").symlink_to(tmp_path / "elsewhere" / "child")
    for level in ("render", "rebuild"):
        status, _, error = cli("reproduce", "ruleset-size", level, "--out", "link/../out")
        assert status == 2 and "--out link/../out names two places" in error
    assert not (tmp_path / "elsewhere" / "out").exists() and not (tmp_path / "work" / "out").exists()
    assert not (tmp_path / "elsewhere" / "out.runs").exists() and not (tmp_path / "work" / "out.runs").exists()


def test_verify_s_report_never_replaces_a_golden_the_tree_reaches_through_a_link(cli, toy, tmp_path):
    rebuilt = tmp_path / "rebuilt.json"
    rebuilt.write_text('{"n": 2, "rate": 0.5}')
    tree = tmp_path / "tree"
    (tree / "results").mkdir(parents=True)
    (tree / "results" / "golden.json").symlink_to(rebuilt)
    status, printed, error = cli("reproduce", "ruleset-size", "verify", "--rerun", tree, "--json", rebuilt)
    assert status == 2 and printed == "" and f"is {tree / 'results' / 'golden.json'} ({rebuilt.resolve()})" in error
    assert "which the check reads" in error and rebuilt.read_text() == '{"n": 2, "rate": 0.5}'
