from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from ruleforge.launch import records
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.records import LaunchRecord


def _job(job_id: str, *, timeout: int = 60, resumable: bool = False, outputs=()) -> dict:
    return {"job_id": job_id, "outputs": list(outputs),
            "resources": {"kind": "cpu", "timeout_min": timeout, "resumable": resumable}}


def _record(tmp_path, jobs=None, **extra) -> LaunchRecord:
    data = {"launch_id": "L1", "backend": "fake", "jobs": jobs or [_job("a"), _job("b")], **extra}
    return LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L1", data)


class FakeBackend:
    """Answers ``states`` from a list of per-poll states (the last repeats), and records pulls and stops."""

    def __init__(self, polls, fail=()):
        self.polls = list(polls)
        self.fail = set(fail)
        self.calls = 0
        self.pulled: list[str] = []
        self.stopped: list[str] = []

    def states(self, record):
        self.calls += 1
        return {job: {"state": state} for job, state in self.polls[min(self.calls, len(self.polls)) - 1].items()}

    def pull(self, record, path):
        if path in self.fail:
            raise LaunchError(f"{path}: holds nothing")
        self.pulled.append(path)

    def stop(self, record, run):
        if run["job_id"] in self.fail:
            raise OSError(f"scancel for {run['job_id']}: no such file")
        self.stopped.append(run["job_id"])
        return {"requested": True}


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_a_record_is_written_before_any_submission_and_after_each(tmp_path):
    record = _record(tmp_path)
    saved = json.loads(record.path.read_text())
    assert saved["state"] == "submitting" and saved["runs"] == [] and saved["error"] is None
    assert saved["schema_version"] == records.SCHEMA_VERSION and saved["created_at"]
    record.add_run({"job_id": "a", "slurm_job": "7"})
    assert json.loads(record.path.read_text())["runs"] == [{"job_id": "a", "slurm_job": "7"}]
    assert record.run_for("a") == {"job_id": "a", "slurm_job": "7"} and record.run_for("b") is None
    record.add_runs([{"job_id": "b", "slurm_job": "8_0"}, {"job_id": "c", "slurm_job": "8_1"}])
    record.update_run("a", exit_code=0)
    assert json.loads(record.path.read_text())["runs"] == [{"job_id": "a", "slurm_job": "7", "exit_code": 0},
                                                           {"job_id": "b", "slurm_job": "8_0"},
                                                           {"job_id": "c", "slurm_job": "8_1"}]
    with pytest.raises(LaunchError, match="records no run of d"):
        record.update_run("d", exit_code=0)
    record.finish(RuntimeError("sbatch died"))
    saved = json.loads(record.path.read_text())
    assert saved["state"] == "failed" and saved["error"] == "RuntimeError: sbatch died" and saved["finished_at"]
    record.finish()
    assert json.loads(record.path.read_text())["state"] == "submitted"
    with pytest.raises(LaunchError, match="a launch id is never reused"):
        _record(tmp_path)


def test_a_launch_id_a_record_holds_is_drawn_again_and_one_without_a_root_is_the_first_draw(tmp_path, monkeypatch):
    drawn = iter(["T-c-0001", "T-c-0001", "T-c-0002"])
    monkeypatch.setattr(records, "new_run_id", lambda commit: next(drawn))
    env = {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path)}
    assert records.new_launch_id("c", env=env) == "T-c-0001"
    LaunchRecord.create(records.records_root(env) / "T-c-0001", {"launch_id": "T-c-0001"})
    assert records.new_launch_id("c", env=env) == "T-c-0002"
    monkeypatch.setattr(records, "new_run_id", lambda commit: "T-c-0001")
    assert records.new_launch_id("c", env={}) == "T-c-0001"


def test_a_record_is_found_by_id_directory_or_file(tmp_path):
    record = _record(tmp_path)
    env = {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path)}
    for reference in ("L1", record.directory, record.path, str(record.directory)):
        assert LaunchRecord.load(reference, env=env).launch_id == "L1"
    for missing in ("L2", "not a name", tmp_path / "nowhere"):
        with pytest.raises(LaunchError, match="no launch record"):
            LaunchRecord.load(missing, env=env)
    assert records.records_root(env) == tmp_path / ".ruleforge-launch"


def test_a_record_under_the_recorded_directory_is_found_by_id_and_seen_as_a_live_writer(tmp_path):
    from ruleforge.core.record_names import RECORDED_LAUNCHES_DIRNAME

    env = {"RULEFORGE_ARTIFACT_ROOT": str(tmp_path)}
    LaunchRecord.create(tmp_path / RECORDED_LAUNCHES_DIRNAME / "OLD", {
        "launch_id": "OLD", "backend": "slurm", "jobs": [_job("r1")], "train": {"output_dir": "training/r1"}})
    assert LaunchRecord.load("OLD", env=env).launch_id == "OLD"
    root = records.records_root(env)
    assert [record.launch_id for record in records.recorded_launches(root)] == ["OLD"]
    live = records.live_writers(root, {"training/r1"}, lambda other: {"r1": {"state": "running"}})
    assert [(entry["launch_id"], entry["paths"]) for entry in live] == [("OLD", ["training/r1"])]


def test_the_paths_to_pull_are_every_jobs_outputs_once(tmp_path):
    record = _record(tmp_path, [_job("a", outputs=["x", "y"]), _job("b", outputs=["y", "z"])])
    assert record.pulls() == ["x", "y", "z"]


def test_a_job_without_a_run_is_submitting_then_never_launched_and_an_unreported_run_is_unknown(tmp_path):
    record = _record(tmp_path, [_job("a"), _job("b"), _job("c")])
    record.add_run({"job_id": "a"})
    record.add_run({"job_id": "b", "refused": "the checkout moved"})
    found = {"a": {"state": "running", "detail": 1}}
    assert records.run_states(record, found)["c"] == {"state": "submitting"}
    record.finish(RuntimeError("died"))
    states = records.run_states(record, found)
    assert states == {"a": {"state": "running", "detail": 1}, "b": {"state": "unknown", "refused": "the checkout moved"},
                      "c": {"state": "not-launched"}}
    assert records.summary(states) == "1 not-launched, 1 running, 1 unknown"
    assert records.succeeded({"state": "succeeded"}) and not records.succeeded({"state": "succeeded", "refused": "x"})


def test_a_wait_allows_the_longest_job_its_requeued_walls_and_a_margin(tmp_path):
    record = _record(tmp_path, [_job("a", timeout=60), _job("b", timeout=100, resumable=True)])
    assert records.default_wait_minutes(record) == 100 * (1 + records.TIMEOUT_REQUEUES) + records.WAIT_MARGIN_MIN
    record.data["jobs"] = []
    assert records.default_wait_minutes(record) == records.WAIT_MARGIN_MIN


@pytest.mark.parametrize("polls, status", [
    ([{"a": "pending", "b": "running"}, {"a": "running", "b": "running"}, {"a": "succeeded", "b": "succeeded"}], 0),
    ([{"a": "running", "b": "succeeded"}, {"a": "failed", "b": "succeeded"}], 1),
    ([{"a": "stopped", "b": "succeeded"}], 1),
    ([{"a": "submitting", "b": "succeeded"}, {"a": "not-launched", "b": "succeeded"}], 1),
    ([{}], 1),
])
def test_wait_exits_0_when_every_run_succeeded_and_1_when_one_did_not(tmp_path, polls, status):
    record, backend, clock, said = _record(tmp_path), FakeBackend(polls), Clock(), []
    code, states = records.wait(record, backend, poll_seconds=5, sleep=clock.sleep, clock=clock, echo=said.append)
    assert code == status and backend.calls == len(polls) and clock.sleeps == [5] * (len(polls) - 1)
    assert states == {job: {"state": state} for job, state in polls[-1].items()}
    assert len(said) == len({records.summary({k: {"state": v} for k, v in poll.items()}) for poll in polls})


def test_wait_exits_3_at_its_deadline_and_echoes_only_changes(tmp_path):
    record, clock, said = _record(tmp_path), Clock(), []
    backend = FakeBackend([{"a": "running", "b": "unknown"}])
    code, states = records.wait(record, backend, timeout_min=1, poll_seconds=20, sleep=clock.sleep, clock=clock,
                                echo=said.append)
    assert code == records.WAIT_TIMED_OUT and clock.now == 60 and backend.calls == 4
    assert said == ["== L1: 1 running, 1 unknown"]


def test_wait_never_counts_a_refused_run_as_a_success(tmp_path):
    class Refused(FakeBackend):
        def states(self, record):
            return {"a": {"state": "succeeded", "refused": "the checkout moved"}, "b": {"state": "succeeded"}}

    clock = Clock()
    assert records.wait(_record(tmp_path), Refused([]), sleep=clock.sleep, clock=clock)[0] == records.WAIT_FAILED


class FromRecord(FakeBackend):
    """A backend whose states come from the record it is handed: each run's ``state`` as written."""

    def states(self, record):
        self.calls += 1
        return records.run_states(record, {run["job_id"]: {"state": run["state"]} for run in record.runs})


def test_wait_pull_and_stop_read_the_record_as_the_launch_left_it_not_as_it_was_loaded(tmp_path):
    launching = _record(tmp_path, [_job("a", outputs=["x"]), _job("b")])
    followed = LaunchRecord.load(launching.directory)
    clock, backend = Clock(), FromRecord([])

    def sleep(seconds):
        clock.sleep(seconds)
        launching.add_run({"job_id": "a", "state": "succeeded"})
        launching.add_run({"job_id": "b", "state": "succeeded"})
        launching.finish()

    code, states = records.wait(followed, backend, poll_seconds=5, sleep=sleep, clock=clock)
    assert code == records.WAIT_SUCCEEDED and backend.calls == 2 and clock.sleeps == [5]
    assert states == {"a": {"state": "succeeded"}, "b": {"state": "succeeded"}}
    assert followed.data["state"] == "submitted" and len(followed.runs) == 2

    stale = LaunchRecord.load(launching.directory)
    stale.data.update(state="submitting", runs=[])
    assert records.pull(stale, backend)["pulled"] == ["x"] and len(stale.runs) == 2
    launching.data["runs"][1]["state"] = "running"
    launching.save()
    stale.data.update(state="submitting", runs=[])
    entry = records.stop(stale, backend)
    assert backend.stopped == ["b"] and "submitting" not in entry


def test_pull_brings_back_only_what_succeeded_and_decides_each_merge_by_its_own_runs(tmp_path, monkeypatch):
    """Two sharded conditions, each shard job writing its own shard root, and a job no merge names."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    x = {"campaign_root": "sampling/c", "condition_id": "m/x", "num_shards": 2, "job_ids": ["x0", "x1"]}
    y = {"campaign_root": "sampling/c", "condition_id": "m/y", "num_shards": 1, "job_ids": ["y0"]}
    x0, x1, y0 = "sampling/c.shards/m/x/shard-0-of-2", "sampling/c.shards/m/x/shard-1-of-2", \
        "sampling/c.shards/m/y/shard-0-of-1"
    record = _record(tmp_path, [_job("x0", outputs=[x0]), _job("x1", outputs=[x1]), _job("y0", outputs=[y0]),
                                _job("z", outputs=["logs/z"])], merges=[x, y])
    merged = []

    def merge(root, condition, *, num_shards):
        merged.append((root, condition, num_shards))
        return {}

    failed = FakeBackend([{"x0": "succeeded", "x1": "failed", "y0": "succeeded", "z": "failed"}])
    report = records.pull(record, failed, merge=merge)
    assert failed.pulled == [x0, y0]
    assert report["pulled"] == [x0, y0] and report["held"] == [x1, "logs/z"]
    assert merged == [(tmp_path / "sampling/c", "m/y", 1)] and report["merged"] == [y]
    [skipped] = report["merge_skipped"]
    assert skipped == {**x, "reason": "1 failed of its 2 shard run(s) here, and no other launch's pull brought their "
                                      "shard roots back: the pull of the relaunch that finishes them merges it"}
    assert "merge_errors" not in report and "pull_errors" not in report

    saved = json.loads(record.path.read_text())
    assert saved["pulled"] == [x0, y0] and saved["merged"] == [{key: y[key] for key in ("campaign_root",
                                                                                        "condition_id", "num_shards")}]

    (tmp_path / x1).mkdir(parents=True)
    report = records.pull(record, FakeBackend([{"x0": "succeeded", "x1": "failed", "y0": "succeeded",
                                                "z": "failed"}]), merge=merge)
    assert report["held"] == [x1, "logs/z"] and report["merge_skipped"][0]["condition_id"] == "m/x"
    assert [condition for _, condition, _ in merged[1:]] == ["m/y"]
    del merged[1:]

    relaunch = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L2", {"launch_id": "L2", "jobs": []})
    relaunch.data["pulled"] = [x1]
    relaunch.save()
    report = records.pull(record, FakeBackend([{"x0": "succeeded", "x1": "failed", "y0": "succeeded",
                                                "z": "failed"}]), merge=merge)
    assert report["pulled"] == [x0, y0] and report["held"] == ["logs/z"] and "merge_skipped" not in report
    assert report["merged"] == [x, y] and [condition for _, condition, _ in merged[1:]] == ["m/x", "m/y"]
    assert len(json.loads(record.path.read_text())["merged"]) == 2

    done = FakeBackend([{"x0": "succeeded", "x1": "succeeded", "y0": "succeeded", "z": "succeeded"}])
    report = records.pull(record, done, merge=merge)
    assert report["merged"] == [x, y] and "held" not in report and "merge_skipped" not in report
    assert done.pulled == [x0, x1, y0, "logs/z"]

    # The default merge is the campaign's own (ruleforge.evaluate.shard_merge), imported only when a merge is due.
    import ruleforge.evaluate.shard_merge as shard_merge

    monkeypatch.setattr(shard_merge, "merge_condition_shards", merge)
    records.pull(record, done)
    assert len(merged) == 7


def test_a_condition_another_launch_merged_is_not_merged_again_nor_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    x = {"campaign_root": "sampling/c", "condition_id": "m/x", "num_shards": 2, "job_ids": ["x0", "x1"]}
    first = _record(tmp_path, [_job("x0"), _job("x1")], merges=[x])
    relaunch = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L2", {
        "launch_id": "L2", "jobs": [_job("x1")], "merges": [{**x, "job_ids": ["x1"]}]})
    merged = []

    def merge(root, condition, *, num_shards):
        merged.append(condition)

    report = records.pull(first, FakeBackend([{"x0": "succeeded", "x1": "failed"}]), merge=merge)
    assert report["merge_skipped"] and merged == []
    report = records.pull(relaunch, FakeBackend([{"x1": "succeeded"}]), merge=merge)
    assert report["merged"] == [{**x, "job_ids": ["x1"]}] and merged == ["m/x"]
    report = records.pull(first, FakeBackend([{"x0": "succeeded", "x1": "failed"}]), merge=merge)
    assert report["merged_elsewhere"] == [{**x, "launch_id": "L2"}] and "merge_skipped" not in report
    assert merged == ["m/x"] and report["merged"] == []
    first.data["merges"] = [{**x, "num_shards": 4}]
    first.save()
    assert records.pull(first, FakeBackend([{"x0": "succeeded", "x1": "failed"}]), merge=merge)["merge_skipped"]


def test_an_earlier_launch_s_pull_or_merge_of_the_same_path_is_of_an_older_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    x = {"campaign_root": "sampling/c", "condition_id": "m/x", "num_shards": 2, "job_ids": ["x0", "x1"]}
    old = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L0", {
        "launch_id": "L0", "jobs": [], "merged": [{key: x[key] for key in ("campaign_root", "condition_id",
                                                                         "num_shards")}],
        "pulled": ["training/r"]})
    old.data["created_at"] = "2000-01-01T00:00:00+00:00"
    old.save()
    record = _record(tmp_path, [_job("x0", outputs=["s0"]), _job("x1", outputs=["s1"]),
                                _job("t", outputs=["training/r"])], merges=[x])
    (tmp_path / "training" / "r").mkdir(parents=True)
    merged = []
    report = records.pull(record, FakeBackend([{"x0": "succeeded", "x1": "succeeded", "t": "stopped"}]),
                          merge=lambda root, condition, *, num_shards: merged.append(condition))
    assert merged == ["m/x"] and "merged_elsewhere" not in report and report["held"] == ["training/r"]


def _lock_taken(path) -> bool:
    script = ("import fcntl, sys; h = open(sys.argv[1], 'a')\n"
              "try: fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
              "except BlockingIOError: sys.exit(3)\n")
    return subprocess.run([sys.executable, "-c", script, str(path)], check=False).returncode == 3


def test_locked_holds_each_lock_beside_its_directory_for_the_block(tmp_path):
    records_dir, root = tmp_path / ".ruleforge-launch", tmp_path / "sampling" / "c1"
    locks = [records.lock_path(records_dir), records.lock_path(records.writers_dir(root))]
    assert [lock.name for lock in locks] == [".ruleforge-launch.lock", ".c1.writers.lock"]
    assert locks[1].parent == tmp_path / "sampling" and records.writers_dir(root).name == ".c1.writers"
    with records.locked(locks):
        assert all(_lock_taken(lock) for lock in locks)
    assert not any(_lock_taken(lock) for lock in locks) and all(lock.is_file() for lock in locks)
    with pytest.raises(RuntimeError, match="inside"), records.locked(locks[:1]):
        assert _lock_taken(locks[0])
        raise RuntimeError("inside")
    assert not _lock_taken(locks[0])


def test_locked_never_takes_a_lock_through_a_symlink_in_its_place(tmp_path):
    """A lock file is opened without following a symlink in its place (``O_NOFOLLOW``)."""
    published = tmp_path / "published"
    published.mkdir()
    lock = records.lock_path(records.writers_dir(tmp_path / "sampling" / "c1"))
    lock.parent.mkdir()
    lock.symlink_to(published / "x.lock")
    with pytest.raises(LaunchError, match="is a symlink: taking it would create or open the file it points at"):
        with records.locked([lock]):
            pytest.fail("the lock was taken")
    assert list(published.iterdir()) == []
    lock.unlink()
    lock.mkdir()
    with pytest.raises(IsADirectoryError):
        with records.locked([lock]):
            pytest.fail("the lock was taken")


def test_a_claim_place_is_held_as_the_records_place_is(tmp_path, monkeypatch):
    from ruleforge.evaluate.api import CampaignError

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    root, published = tmp_path / "sampling" / "c1", tmp_path / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    root.parent.mkdir()
    records.check_claim_places([root])
    directory = records.writers_dir(root)
    for place, what in ((records.lock_path(directory), "claims lock"), (directory, "claims directory")):
        place.symlink_to(published / "x", target_is_directory=place == directory)
        with pytest.raises(CampaignError, match=f"the campaign root's {what} .* is a symlink"):
            records.check_claim_places([root])
        place.unlink()
    inside = published / "sampling" / "c1"
    with pytest.raises(CampaignError, match="is inside the campaign root"):
        records.check_claim_places([inside])
    assert sorted(path.name for path in tmp_path.rglob("*")) == ["published", "sampling", "sampling_manifest.json"]


def test_a_claim_beside_a_root_names_the_launch_and_outlives_its_record(tmp_path):
    record = _record(tmp_path, [_job("a")], campaign={"conditions": []})
    roots = [tmp_path / "sampling" / "c1", tmp_path / "sampling" / "c2"]
    records.claim_roots(record, roots)
    assert [sorted(path.name for path in records.writers_dir(root).iterdir()) for root in roots] == [["L1.json"]] * 2
    claim = json.loads((records.writers_dir(roots[0]) / "L1.json").read_text(encoding="utf-8"))
    assert claim == {"launch_id": "L1", "record": str(record.directory), "data": record.data}
    record.add_run({"job_id": "a", "run_id": "r"})
    [claimed] = records.claimed_launches(roots[:1])
    assert claimed.directory == record.directory and claimed.runs == [{"job_id": "a", "run_id": "r"}]
    assert [other.launch_id for other in records.claimed_launches(roots)] == ["L1", "L1"]
    assert [other.launch_id for other in records.launches_writing(tmp_path / ".ruleforge-launch", roots)] == ["L1"]
    assert records.launches_writing(tmp_path / ".ruleforge-launch", []) [0].directory == record.directory
    assert records.claimed_launches([tmp_path / "sampling" / "none"]) == []

    def shared_of(other):
        return {job["job_id"]: "c1" for job in other.jobs}

    def states_of(other):
        return records.run_states(other, {"a": {"state": "running"}} if other.runs else {})

    live = records.live_launches(records.launches_writing(tmp_path / ".ruleforge-launch", roots), shared_of, states_of)
    assert live == [{"launch_id": "L1", "shared": {"a": "c1"}, "states": {"a": {"state": "running"}},
                     "record_gone": False}]
    shutil.rmtree(record.directory)
    [gone] = records.claimed_launches(roots[:1])
    assert gone.data["state"] == "submitting" and gone.runs == [] and not gone.path.is_file()
    assert records.live_launches([gone], shared_of, states_of) == [
        {"launch_id": "L1", "shared": {"a": "c1"}, "states": {"a": {"state": "submitting"}}, "record_gone": True}]
    assert records.live_launches([gone], shared_of, states_of, ignore=["L1"]) == []
    assert records.recorded_launches(tmp_path / ".ruleforge-launch") == []


def test_a_live_writer_is_a_launch_whose_job_writing_the_path_has_not_ended(tmp_path):
    train = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "T1", {
        "launch_id": "T1", "backend": "slurm", "jobs": [_job("r1")],
        "train": {"output_dir": "training/r1"}})
    LaunchRecord.create(tmp_path / ".ruleforge-launch" / "C1", {
        "launch_id": "C1", "backend": "slurm", "jobs": [_job("s0", outputs=["a.shards/s0"]),
                                                        _job("s1", outputs=["a.shards/s1"])]})
    assert records.job_writes(train.data) == {"r1": {"training/r1"}}
    states = {"T1": {"r1": {"state": "running"}}, "C1": {"s0": {"state": "failed"}, "s1": {"state": "running"}}}

    def states_of(record):
        return states[record.launch_id]

    root = tmp_path / ".ruleforge-launch"
    assert records.live_writers(root, {"training/r1", "a.shards/s0", "elsewhere"}, states_of) == [
        {"launch_id": "T1", "paths": ["training/r1"], "states": {"r1": {"state": "running"}}}]
    assert records.live_writers(root, {"a.shards/s1"}, states_of)[0]["launch_id"] == "C1"
    assert records.live_writers(root, {"training/r1"}, states_of, ignore=["T1"]) == []
    assert records.live_writers(root, set(), states_of) == []

    def unreachable(record):
        raise LaunchError("sacct is down")

    [entry] = records.live_writers(root, {"training/r1"}, unreachable)
    assert entry["states"] == {"r1": {"state": "unknown", "error": "sacct is down"}}
    assert records.live_writers(root, {"training/r1"}, lambda record: {})[0]["states"] == {
        "r1": {"state": "unknown"}}


def test_a_merge_naming_no_jobs_waits_on_every_run_and_one_naming_an_unknown_job_never_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))  # the pull checks its merges' roots under it first
    record = _record(tmp_path, [_job("a", outputs=["sampling/c.shards"]), _job("b")],
                     merges=[{"campaign_root": "sampling/c", "condition_id": "m/f/p"}])

    def merge(*args, **kwargs):
        pytest.fail("merged")

    report = records.pull(record, FakeBackend([{"a": "succeeded", "b": "failed"}]), merge=merge)
    assert report["pulled"] == ["sampling/c.shards"] and "held" not in report
    assert report["merge_skipped"][0]["reason"].startswith("1 failed of its 2 shard run(s) here")
    record.data["merges"][0]["job_ids"] = ["a", "gone"]
    record.save()
    report = records.pull(record, FakeBackend([{"a": "succeeded", "b": "succeeded"}]), merge=merge)
    assert report["merge_skipped"][0]["reason"].startswith("1 unknown of its 2 shard run(s) here")


def test_a_merge_that_refuses_is_reported_and_the_others_still_run(tmp_path, monkeypatch):
    from ruleforge.evaluate.sharding import ShardError

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    merges = [{"campaign_root": "sampling/c", "condition_id": condition, "num_shards": 1, "job_ids": [job]}
              for condition, job in (("m/x", "a"), ("m/y", "b"))]
    record = _record(tmp_path, merges=merges)
    merged = []

    def merge(root, condition, *, num_shards):
        if condition == "m/x":
            raise ShardError("sampling/c.shards/m/x lacks shard(s) [0] of 1")
        merged.append(condition)
        return {}

    report = records.pull(record, FakeBackend([{"a": "succeeded", "b": "succeeded"}]), merge=merge)
    assert merged == ["m/y"] and report["merged"] == [merges[1]]
    assert report["merge_errors"] == [{**merges[0], "error": "ShardError: sampling/c.shards/m/x lacks shard(s) [0] "
                                                             "of 1"}]


def test_pull_with_nothing_to_pull_asks_nothing(tmp_path):
    record, backend = _record(tmp_path, [_job("a")]), FakeBackend([])
    assert records.pull(record, backend) == {"pulled": [], "merged": []} and backend.calls == 0


def test_pull_waits_for_this_launchs_runs_to_end(tmp_path):
    record = _record(tmp_path, [_job("a", outputs=["x"]), _job("b")])
    backend = FakeBackend([{"a": "succeeded", "b": "running"}])
    with pytest.raises(LaunchError, match="L1: b not ended \\(1 running, 1 succeeded\\): wait first"):
        records.pull(record, backend)
    assert backend.pulled == []


def test_pull_takes_each_path_on_its_own_and_skips_only_the_merge_whose_path_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    merges = [{"campaign_root": "sampling/c", "condition_id": condition, "job_ids": [job]}
              for condition, job in (("m/x", "a"), ("m/y", "b"))]
    record = _record(tmp_path, [_job("a", outputs=["sampling/missing", "sampling/partial"]),
                                _job("b", outputs=["sampling/fine"])], merges=merges)
    backend = FakeBackend([{"a": "succeeded", "b": "succeeded"}], fail=["sampling/missing"])
    merged = []
    report = records.pull(record, backend, merge=lambda root, condition, **kwargs: merged.append(condition))
    assert backend.pulled == ["sampling/partial", "sampling/fine"]
    assert report["pulled"] == ["sampling/partial", "sampling/fine"]
    assert report["pull_errors"] == {"sampling/missing": "sampling/missing: holds nothing"}
    assert report["merge_skipped"] == [{**merges[0], "reason": "1 of its path(s) were not pulled: a merge refuses "
                                                               "missing shards"}]
    assert merged == ["m/y"] and report["merged"] == [merges[1]]
    record.data["merges"] = []
    record.save()
    assert "merge_skipped" not in records.pull(record, backend)


def test_the_later_records_are_the_other_launches_made_at_or_after_one(tmp_path):
    first = _record(tmp_path)
    earlier = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L0", {"launch_id": "L0", "jobs": []})
    earlier.data["created_at"] = "2000-01-01T00:00:00+00:00"
    earlier.save()
    later = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L2", {"launch_id": "L2", "jobs": []})
    later.data["created_at"] = "2999-01-01T00:00:00+00:00"
    later.save()
    same = LaunchRecord.create(tmp_path / ".ruleforge-launch" / "L3", {"launch_id": "L3", "jobs": []})
    same.data["created_at"] = first.data["created_at"]
    same.save()
    assert [record.launch_id for record in records.later_records(first)] == ["L3", "L2"]


def test_stop_stops_the_runs_that_have_not_ended_and_logs_each_request(tmp_path):
    record = _record(tmp_path, [_job("a"), _job("b"), _job("c"), _job("d")])
    for job in ("a", "b", "c", "d"):
        record.add_run({"job_id": job})
    record.check_not_stopped()
    backend = FakeBackend([{"a": "running", "b": "succeeded", "c": "pending", "d": "unknown"}], fail=["c"])
    entry = records.stop(record, backend, reason="save landed", details={"after_save": 200})
    assert entry["after_save"] == 200
    assert backend.stopped == ["a", "d"] and entry["stopped"] == {"a": {"requested": True}, "d": {"requested": True}}
    assert entry["stop_errors"] == {"c": "scancel for c: no such file"} and "submitting" not in entry
    records.stop(record, FakeBackend([{"a": "stopped", "b": "succeeded", "c": "stopped", "d": "stopped"}]))
    log = json.loads((record.directory / records.STOP_FNAME).read_text())
    assert [item["reason"] for item in log] == ["save landed", None] and log[1]["stopped"] == {}
    assert "stop_errors" not in log[1]


def test_a_stop_while_the_launch_submits_names_the_jobs_it_could_not_stop_and_ends_the_submission(tmp_path):
    record = _record(tmp_path, [_job("a"), _job("b")])
    record.add_run({"job_id": "a"})
    backend = FakeBackend([{"a": "running", "b": "submitting"}])
    entry = records.stop(record, backend)
    assert backend.stopped == ["a"] and entry["submitting"] == ["b"]
    with pytest.raises(LaunchError, match="a stop of L1 was requested while it was submitting"):
        record.check_not_stopped()
