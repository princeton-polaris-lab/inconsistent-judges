from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

from ruleforge.cli import app, jobs
from ruleforge.launch import cli as launch_cli
from ruleforge.launch.backends import LocalBackend
from ruleforge.launch.records import LaunchRecord, records_root

REPO = Path(__file__).resolve().parents[2]


def _record(launch_id: str, *, backend: str = "local", exit_code=0, merges=()) -> LaunchRecord:
    spec = launch_cli.build_job({"backend": "local", "output": ["sampling/c"]}, ["true"])[1]
    record = LaunchRecord.create(records_root() / launch_id, {
        "launch_id": launch_id, "backend": backend, "verb": "job", "code": {"commit": "a" * 40}, "merges": list(merges),
        "options": {"gpu": None, "gpus": None, "pod": None, "allow_dirty": False},
        "jobs": [{**spec.to_json(), "request": LocalBackend(REPO).render([spec])[0]}]})
    record.add_run({"job_id": "job", "exit_code": exit_code} if backend == "local" else {"job_id": "job", "run_id": "r"})
    record.finish()
    return record


def test_status_wait_pull_and_stop_follow_a_launch_as_the_launcher_does(cli, monkeypatch):
    _record("L1")
    status, out, _ = cli("jobs", "status", "L1")
    assert status == 0 and out.splitlines()[0] == "== L1 (local): 1 succeeded"
    status, out, _ = cli("jobs", "status", "L1", "--json")
    assert status == 0 and out["states"] == {"job": {"state": "succeeded", "exit_code": 0}}
    assert out["backend"] == "local" and out["launch_state"] == "submitted"
    status, out, _ = cli("jobs", "wait", "L1", "--json", "--timeout-min", "1", "--pull")
    assert status == 0 and out["wait"] == 0 and out["pull"] == {"pulled": ["sampling/c"], "merged": []}
    status, out, _ = cli("jobs", "pull", "L1")
    assert status == 0 and out.strip() == "== pulled 1 path(s)"
    status, out, _ = cli("jobs", "stop", "L1", "--reason", "enough")
    assert status == 0 and "nothing (every named run has ended)" in out
    _record("L2", exit_code=3, merges=[{"campaign_root": "sampling/c", "condition_id": "m/x"}])
    status, out, _ = cli("jobs", "wait", "L2")
    assert status == 1 and "job  failed" in out
    status, out, _ = cli("jobs", "pull", "L2", "--json")
    assert status == 1 and out["held"] == ["sampling/c"] and out["merge_skipped"][0]["condition_id"] == "m/x"


def test_a_refusal_after_a_pull_opened_its_run_record_exits_1_and_shows_its_traceback_under_verbose(cli):
    from ruleforge.cli import records
    from ruleforge.launch.environment import LaunchError

    spec = launch_cli.build_job({"backend": "local", "output": ["sampling/c"]}, ["true"])[1]
    record = LaunchRecord.create(records_root() / "L1", {
        "launch_id": "L1", "backend": "local", "verb": "job", "code": {"commit": "a" * 40},
        "merges": [{"campaign_root": "sampling/c", "condition_id": "m/x"}], "options": {"allow_dirty": False},
        "jobs": [{**spec.to_json(), "request": LocalBackend(REPO).render([spec])[0]}]})
    record.add_run({"job_id": "job"})  # still running: no exit status yet
    status, _, err = cli("jobs", "pull", "L1")
    assert status == 1 and "job not ended" in err and "Traceback" not in err
    [made] = records.runs_dir(records_root().parent / "sampling" / "c").iterdir()
    assert (made / records.LOG_FNAME).read_text().splitlines()[-1].endswith("the run ended on LaunchError: L1: job not "
                                                                             "ended (1 submitting): wait first")
    with pytest.raises(LaunchError, match="wait first"):
        app.main(["-v", "jobs", "pull", "L1"])


def test_a_launcher_refusal_the_command_line_would_take_for_a_failure_still_ends_in_one_line(cli, monkeypatch):
    _record("L1")

    def unreadable(*args):
        raise KeyError("jobs")

    monkeypatch.setitem(launch_cli.CONTROL_HANDLERS, "status", unreadable)
    status, _, err = cli("jobs", "status", "L1")
    assert status == 2 and err == "ruleforge jobs: CliError: KeyError: 'jobs'\n"


def test_a_refusal_exits_2_with_its_reason_and_no_traceback(cli):
    _record("L1")
    status, _, err = cli("jobs", "status", "nope")
    assert status == 2 and err.startswith("ruleforge jobs: LaunchError: no launch record 'nope'")
    assert "Traceback" not in err
    status, _, err = cli("jobs", "stop", "L1", "--after-save", "100")
    assert status == 2 and "L1 is not a training launch (ruleforge jobs launch train)" in err
    status, _, err = cli("jobs", "stop", "L1", "--timeout-min", "5")
    assert status == 2 and "--timeout-min bounds the wait of --after-save" in err
    (records_root() / "L1" / "launch.json").write_text("{}")
    status, _, err = cli("jobs", "status", "L1")
    assert status == 2 and "KeyError: 'backend'" in err
    with pytest.raises(SystemExit):
        app.main(["jobs", "status", "L1", "--", "x=1"])


def test_jobs_needs_submitit_and_says_so(cli, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "submitit" else real(name))
    status, _, err = cli("jobs", "status", "L1")
    assert status == 2 and err == ("ruleforge jobs: MissingDependency: launching and following jobs needs submitit, which "
                                   "this environment lacks: add --group dev to uv sync, or pip install -e '.[slurm]' "
                                   "from the checkout\n")


def test_the_jobs_parser_is_the_launcher_s_control_verbs():
    parser = app.build_parser()
    args = parser.parse_args(["jobs", "wait", "L1", "--timeout-min", "2", "--pull", "--json"])
    assert (args.command, args.verb, args.launch, args.timeout_min, args.pull, args.json) == (
        "jobs", "wait", "L1", 2.0, True, True)
    assert args.handler is jobs.run_jobs and set(launch_cli.CONTROL_HANDLERS) == {"status", "wait", "pull", "stop"}
    args = parser.parse_args(["jobs", "stop", "L1", "--after-save", "300", "--reason", "r"])
    assert (args.after_save, args.reason, args.timeout_min) == (300, "r", None)
    with pytest.raises(SystemExit):
        parser.parse_args(["jobs", "wai", "L1"])  # no abbreviation


def _tree(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() if path.is_file() else b""
            for path in sorted(directory.rglob("*"))}


def test_a_pull_into_a_pinned_or_enclosed_campaign_root_is_refused_before_anything_is_written(cli, monkeypatch):
    from ruleforge.evaluate import api

    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/pub": ("leniency",)})
    artifacts = records_root().parent
    published = artifacts / "sampling" / "pub"
    (published / "m" / "x" / "draw-001").mkdir(parents=True)
    (published / "sampling_manifest.json").write_text("{}")
    (published / "completion.json").write_text('{"state": "complete"}\n')
    _record("P1", merges=[{"campaign_root": "sampling/pub", "condition_id": "m/x"}])
    before = _tree(artifacts)
    for argv in (("pull", "P1"), ("wait", "P1", "--pull", "--timeout-min", "1")):  # the shard merge itself, unpatched
        status, _, err = cli("jobs", *argv)
        assert status == 2 and err.startswith(f"ruleforge jobs: CampaignError: {published} ")
        assert "a published run of leniency" in err and "Traceback" not in err
    assert _tree(artifacts) == before
    merged = []
    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards",
                        lambda *args, **kwargs: merged.append(args) or {})
    status, out, _ = cli("jobs", "wait", "P1", "--json", "--timeout-min", "1")
    assert status == 0 and "pull" not in out
    outer = records_root().parent / "sampling" / "outer"
    outer.mkdir(parents=True)
    (outer / "sampling_manifest.json").write_text("{}")
    _record("P2", merges=[{"campaign_root": "sampling/outer/inner", "condition_id": "m/x"}])
    status, _, err = cli("jobs", "pull", "P2")
    assert status == 2 and f"is inside the campaign root {outer}" in err and merged == []
    _record("P3", merges=[{"campaign_root": "sampling/fresh", "condition_id": "m/x"}])
    status, out, _ = cli("jobs", "pull", "P3", "--json")
    assert status == 0 and out["merged"] == [{"campaign_root": "sampling/fresh", "condition_id": "m/x"}]
    assert len(merged) == 1
    assert sorted(path.name for path in (records_root().parent / "sampling").iterdir()) == ["fresh.runs", "outer",
                                                                                            "pub"]


def _merging(merged: list):
    def merge(root, condition_id, **kwargs):
        logging.getLogger("ruleforge.evaluate.shard_merge").info("merged %s into %s", condition_id, root)
        merged.append((root, condition_id))
        return {}

    return merge


def _runs(root: Path) -> list[Path]:
    return sorted(root.parent.glob(f"{root.name}.runs/*"))


def test_a_pull_that_merges_records_its_run_beside_each_campaign_root_it_writes(cli, monkeypatch, capsys):
    merged = []
    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards", _merging(merged))
    artifacts = records_root().parent
    first, second = artifacts / "sampling" / "a", artifacts / "sampling" / "b"
    _record("M1", merges=[{"campaign_root": "sampling/a", "condition_id": "m/x"},
                          {"campaign_root": "sampling/a", "condition_id": "m/y"},
                          {"campaign_root": "sampling/b", "condition_id": "m/x"}])
    status, out, err = cli("jobs", "pull", "M1", "--json")
    assert status == 0 and len(out["merged"]) == 3 and len(merged) == 3
    assert [len(_runs(first)), len(_runs(second))] == [1, 1]
    for root in (first, second):
        [run] = _runs(root)
        assert run.name.endswith("-jobs-pull")
        argv = json.loads((run / "argv.json").read_text())
        assert argv["argv"] == ["jobs", "pull", "M1", "--json"] and argv["subcommand"] == "jobs pull"
        assert {"code", "cwd", "package_version", "python", "started_at"} <= argv.keys()
        assert (run / "resolved_config.yaml").read_text().startswith("# no config")
        log = (run / "log.txt").read_text()
        assert f"merged m/x into {first}" in log and f"merged m/x into {second}" in log
    assert err.count(f"merged m/x into {first}") == 1
    status, out, _ = cli("jobs", "wait", "M1", "--pull", "--json", "--timeout-min", "1")
    assert status == 0 and len(out["pull"]["merged"]) == 3 and len(merged) == 6
    assert [path.name.rsplit("-", 2)[-2:] for path in _runs(first)] == [["jobs", "pull"], ["jobs", "wait"]]
    for argv in (("status", "M1"), ("wait", "M1"), ("stop", "M1"), ("pull", "L0")):
        if argv[1] == "L0":
            _record("L0")
        assert cli("jobs", *argv)[0] == 0
    assert [len(_runs(first)), len(_runs(second))] == [2, 2]
    assert sorted(path.name for path in records_root().iterdir()) == ["L0", "M1"]
    assert sorted(path.name for path in (artifacts / "sampling").iterdir()) == ["a.runs", "b.runs"]


def test_a_run_ended_by_an_exception_ends_every_record_s_log_with_it(tmp_path):
    directories = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    for directory in directories:
        directory.mkdir()
    with pytest.raises(RuntimeError), jobs.logging_to(directories):
        logging.getLogger("ruleforge.cli").info("started")
        raise RuntimeError("the backend went away")
    for directory in directories:
        log = (directory / "log.txt").read_text().splitlines()
        assert "started" in log[0] and log[-1].endswith("the run ended on RuntimeError: the backend went away")
    assert not [handler for handler in logging.getLogger().handlers
                if Path(getattr(handler, "baseFilename", "/")).parent in directories]


def test_a_pull_whose_run_records_cannot_be_made_where_they_go_is_refused_before_anything_is_written(cli, monkeypatch,
                                                                                                   tmp_path):
    """A campaign root's ``.runs`` that is a symlink would carry the record wherever it points."""
    merged = []
    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards", _merging(merged))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    runs = records_root().parent / "sampling" / "a.runs"
    runs.parent.mkdir(parents=True)
    runs.symlink_to(elsewhere, target_is_directory=True)
    record = _record("R1", merges=[{"campaign_root": "sampling/a", "condition_id": "m/x"}])
    before = record.path.read_text()
    status, _, err = cli("jobs", "pull", "R1")
    assert status == 2 and f"its run records' directory {runs} is a symlink" in err
    assert list(elsewhere.iterdir()) == [] and merged == [] and record.path.read_text() == before


def test_a_verb_that_writes_a_launch_s_record_refuses_one_reached_through_a_symlink(cli, tmp_path):
    record = _record("S1")
    link = tmp_path / "linked"
    link.symlink_to(record.directory)
    status, out, _ = cli("jobs", "status", str(link), "--json")
    assert status == 0 and out["launch_id"] == "S1"
    for verb in ("pull", "stop"):
        status, _, err = cli("jobs", verb, str(link))
        assert status == 2 and f"the launch S1's record directory {link} is a symlink" in err
    assert not (record.directory / "stop.json").exists() and "pulled" not in record.refresh().data
