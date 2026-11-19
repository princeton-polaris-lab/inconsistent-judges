from __future__ import annotations

import fcntl
import json
import os
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Optional, Protocol, Sequence

from ruleforge.launch.environment import new_run_id, utc_now
from ruleforge.launch.guards import NAME
from ruleforge.launch.jobs import TIMEOUT_REQUEUES
from ruleforge.launch.environment import LaunchError
from ruleforge.core.errors import ArtifactRootError
from ruleforge.env import artifact_root
from ruleforge.core.artifacts import write_text_atomically
from ruleforge.core.record_names import RECORDED_LAUNCHES_DIRNAME

RECORDS_DIRNAME = ".ruleforge-launch"
RECORD_FNAME = "launch.json"
STOP_FNAME = "stop.json"
#: The lock a launch holds from its check of the other launches to its own record.
LOCK_SUFFIX = ".lock"
#: Beside a campaign root, `.<root name>.writers/`: the launches that write it.
WRITERS_SUFFIX = ".writers"
SCHEMA_VERSION = 1

PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED, UNKNOWN, SUBMITTING, NOT_LAUNCHED = (
    "pending", "running", "succeeded", "failed", "stopped", "unknown", "submitting", "not-launched")
ENDED = (SUCCEEDED, FAILED, STOPPED, NOT_LAUNCHED)
WAIT_SUCCEEDED, WAIT_FAILED, WAIT_TIMED_OUT = 0, 1, 3
#: Minutes a wait allows past the longest job's own limit.
WAIT_MARGIN_MIN = 30
POLL_SECONDS = 30.0


class RunControl(Protocol):
    """What these functions ask of a backend."""

    def states(self, record: "LaunchRecord") -> dict[str, dict[str, Any]]: ...

    def pull(self, record: "LaunchRecord", path: str) -> None: ...

    def stop(self, record: "LaunchRecord", run: Mapping[str, Any]) -> dict[str, Any]: ...


def records_root(env: Optional[dict[str, str]] = None) -> Path:
    return artifact_root(env) / RECORDS_DIRNAME


def writers_dir(campaign_root: Path | str) -> Path:
    """Return `.<root name>.writers` beside the campaign root, resolved."""
    root = Path(campaign_root).resolve()
    return root.with_name(f".{root.name}{WRITERS_SUFFIX}")


def lock_path(directory: Path) -> Path:
    """Return `<directory>.lock` beside the directory a launch's check and claim concern."""
    return Path(directory).with_name(f"{Path(directory).name}{LOCK_SUFFIX}")


@dataclass(frozen=True)
class RecordsPlace:
    """A place every launch writes under the artifact root: its path, its inventory name and what a refusal calls it.
    """

    path: Path
    name: str
    what: str


def records_places() -> list[RecordsPlace]:
    """Return the places every launch writes under the artifact root: the records directory and its lock."""
    directory = records_root()
    return [RecordsPlace(lock_path(directory), "records lock", "the launch records' lock"),
            RecordsPlace(directory, "records directory", "the launch records' directory")]


def check_records_place() -> None:
    """Refuse a launch whose record would land on a symlink, a campaign root or a pinned root's path; nothing is
    written."""
    from ruleforge.evaluate.api import refuse_protected

    for place in records_places():
        refuse_protected(place.path, place.what)


def check_claim_places(roots: Iterable[Path]) -> None:
    """Refuse a launch whose claims beside the campaign `roots` would land where nothing may; nothing is written."""
    from ruleforge.evaluate.api import refuse_protected

    for root in roots:
        directory = writers_dir(root)
        refuse_protected(lock_path(directory), "the campaign root's claims lock")
        refuse_protected(directory, "the campaign root's claims directory")


def _no_follow(path: Path, flags: int) -> int:
    return os.open(path, flags | os.O_NOFOLLOW, 0o666)


@contextmanager
def locked(paths: Iterable[Path]) -> Iterator[None]:
    """Hold `flock` on each lock file for the block, taken in path order; a lock file through a symlink is refused."""
    handles = []
    try:
        for path in sorted({Path(path).parent.resolve() / Path(path).name for path in paths}):
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                handle = open(path, "a", encoding="utf-8", opener=_no_follow)
            except OSError:
                if path.is_symlink():
                    raise LaunchError(f"the lock {path} is a symlink: taking it would create or open the file it "
                                      "points at; remove it") from None
                raise
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def new_launch_id(commit: str, *, env: Optional[dict[str, str]] = None) -> str:
    """Return a launch id no record under the artifact root holds yet."""
    try:
        root = records_root(env)
    except ArtifactRootError:
        return new_run_id(commit)
    launch_id = new_run_id(commit)
    while (root / launch_id).exists():
        launch_id = new_run_id(commit)
    return launch_id


class LaunchRecord:
    """One launch's record, in its directory."""

    def __init__(self, directory: Path, data: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.data = data

    @property
    def path(self) -> Path:
        return self.directory / RECORD_FNAME

    @classmethod
    def create(cls, directory: Path, data: Mapping[str, Any]) -> "LaunchRecord":
        """Create the record, written before anything is submitted; refused where one exists."""
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError as error:
            raise LaunchError(f"{directory} exists: a launch id is never reused") from error
        record = cls(directory, {"schema_version": SCHEMA_VERSION, **data, "created_at": utc_now(),
                                 "state": "submitting", "runs": [], "error": None})
        record.save()
        return record

    @classmethod
    def load(cls, reference: str | Path, *, env: Optional[dict[str, str]] = None) -> "LaunchRecord":
        """Return the record a launch id, a launch directory or its `launch.json` names."""
        path = Path(reference).expanduser()
        directory = path.parent if path.name == RECORD_FNAME else path
        if not (directory / RECORD_FNAME).is_file() and NAME.fullmatch(str(reference)):
            root = records_root(env)
            directory = next((place / str(reference) for place in (root, root.with_name(RECORDED_LAUNCHES_DIRNAME))
                              if (place / str(reference) / RECORD_FNAME).is_file()), root / str(reference))
        if (directory / RECORD_FNAME).is_file():
            return cls(directory, {}).refresh()
        raise LaunchError(f"no launch record {str(reference)!r}: give a launch id (under $RULEFORGE_ARTIFACT_ROOT/"
                          f"{RECORDS_DIRNAME}/) or a launch directory")

    def save(self) -> None:
        write_text_atomically(self.path, json.dumps(self.data, indent=1, sort_keys=True) + "\n")

    def refresh(self) -> "LaunchRecord":
        """Re-read `launch.json`: another process may have named runs or finished since."""
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        return self

    @property
    def launch_id(self) -> str:
        return self.data["launch_id"]

    @property
    def jobs(self) -> list[dict[str, Any]]:
        return self.data["jobs"]

    @property
    def runs(self) -> list[dict[str, Any]]:
        return self.data["runs"]

    def run_for(self, job_id: str) -> Optional[dict[str, Any]]:
        return next((run for run in self.runs if run["job_id"] == job_id), None)

    def add_run(self, run: Mapping[str, Any]) -> None:
        """Record one submitted run at once."""
        self.add_runs([run])

    def add_runs(self, runs: Iterable[Mapping[str, Any]]) -> None:
        """Record several runs in one write."""
        self.runs.extend(dict(run) for run in runs)
        self.save()

    def update_run(self, job_id: str, **fields: Any) -> None:
        """Add fields to a recorded run at once."""
        self.update_runs({job_id: fields})

    def update_runs(self, updates: Mapping[str, Mapping[str, Any]]) -> None:
        """Add fields to several recorded runs in one write; a job with no recorded run is refused."""
        runs = {job_id: self.run_for(job_id) for job_id in updates}
        missing = [job_id for job_id, run in runs.items() if run is None]
        if missing:
            raise LaunchError(f"{self.launch_id} records no run of {', '.join(missing)}")
        for job_id, fields in updates.items():
            runs[job_id].update(fields)
        self.save()

    def finish(self, error: Optional[BaseException] = None) -> None:
        self.data.update({"state": "submitted" if error is None else "failed", "finished_at": utc_now(),
                          "error": None if error is None else f"{type(error).__name__}: {error}"})
        self.save()

    def pulls(self) -> list[str]:
        """Return the artifact paths the jobs write, in job order, each once."""
        return list(dict.fromkeys(path for job in self.jobs for path in job["outputs"]))

    def check_not_stopped(self) -> None:
        """Refuse to submit once a stop was requested."""
        if (self.directory / STOP_FNAME).is_file():
            raise LaunchError(f"a stop of {self.launch_id} was requested while it was submitting: no later job "
                              "was submitted")


def run_states(record: LaunchRecord, found: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return every job's state in job order: what the backend `found`, `unknown` for a run it did not report, and
    `submitting` or `not-launched` for a job with no run."""
    states = {}
    for job in record.jobs:
        run = record.run_for(job["job_id"])
        if run is None:
            states[job["job_id"]] = {"state": SUBMITTING if record.data.get("state") == "submitting" else NOT_LAUNCHED}
            continue
        states[job["job_id"]] = dict(found.get(job["job_id"]) or {"state": UNKNOWN})
        if run.get("refused"):
            states[job["job_id"]]["refused"] = run["refused"]
    return states


def succeeded(entry: Mapping[str, Any]) -> bool:
    """Return whether a run succeeded and the launch took it (a refused run never counts)."""
    return entry["state"] == SUCCEEDED and not entry.get("refused")


def summary(states: Mapping[str, Mapping[str, Any]]) -> str:
    """Return a count of the runs in each state, like `2 running, 1 succeeded`."""
    counts = Counter(entry["state"] for entry in states.values())
    return ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))


def default_wait_minutes(record: LaunchRecord) -> int:
    """Return how long a wait allows: the longest job's limit plus `WAIT_MARGIN_MIN`."""
    walls = [job["resources"]["timeout_min"] * (1 + TIMEOUT_REQUEUES if job["resources"]["resumable"] else 1)
             for job in record.jobs]
    return max(walls, default=0) + WAIT_MARGIN_MIN


def wait(
    record: LaunchRecord, backend: RunControl, *, timeout_min: Optional[float] = None, poll_seconds: float = POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
    echo: Callable[[str], None] = lambda line: None,
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Poll the runs' states until every run has ended or `timeout_min` passes; return the exit status and the last
    states."""
    minutes = default_wait_minutes(record) if timeout_min is None else timeout_min
    deadline = clock() + minutes * 60
    said = None
    while True:
        states = backend.states(record.refresh())
        line = summary(states)
        if line != said:
            echo(f"== {record.launch_id}: {line}")
            said = line
        if all(entry["state"] in ENDED for entry in states.values()):
            every = bool(states) and all(succeeded(entry) for entry in states.values())
            return (WAIT_SUCCEEDED if every else WAIT_FAILED), states
        if clock() >= deadline:
            return WAIT_TIMED_OUT, states
        sleep(poll_seconds)


_PER_ITEM_ERRORS = (LaunchError, OSError)
_MERGE_ERRORS = (ValueError, OSError)


def _merge_runs(entry: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {job_id: states.get(job_id, {"state": UNKNOWN}) for job_id in entry.get("job_ids") or states}


def _here(path: str) -> bool:
    return (artifact_root() / path).exists()


def _brought_back(paths: Sequence[str], pulled_elsewhere: Collection[str]) -> bool:
    return bool(paths) and all(path in pulled_elsewhere and _here(path) for path in paths)


def merge_key(entry: Mapping[str, Any]) -> tuple[str, str, Optional[int]]:
    """Return what a merge merges: a condition of a campaign root, cut into so many shards."""
    return entry["campaign_root"], entry["condition_id"], entry.get("num_shards")


def other_records(record: LaunchRecord) -> list[LaunchRecord]:
    """Return the other launches recorded beside `record`."""
    paths = sorted(record.directory.parent.glob(f"*/{RECORD_FNAME}"))
    return [LaunchRecord(path.parent, {}).refresh() for path in paths if path.parent != record.directory]


def check_writes(record: LaunchRecord, *, merges: bool = False) -> list[Path]:
    """Refuse a verb that would write where nothing may (the record directory; with `merges`, each campaign root the
    shard merges write) and return those roots. Nothing is written."""
    from ruleforge.evaluate.api import refuse_protected

    refuse_protected(record.directory, f"the launch {record.launch_id}'s record directory")
    return check_merge_roots(record.data.get("merges") or []) if merges else []


def check_merge_roots(merges: Iterable[Mapping[str, Any]]) -> list[Path]:
    """Return the campaign roots the shard `merges` write, each refused first when it lies in another campaign root or on
    a pinned root's path."""
    from ruleforge.evaluate.api import refuse_protected
    from ruleforge.evaluate.campaign import check_new_condition_id

    roots: dict[Path, None] = {}
    for entry in merges:
        root = artifact_root() / entry["campaign_root"]
        refuse_protected(root, f"the campaign root its merge of {entry['condition_id']} writes", campaign_root=True)
        check_new_condition_id(root, entry["condition_id"])
        roots[root] = None
    return list(roots)


def pull(
    record: LaunchRecord, backend: RunControl, *, merge: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Once every run has ended, take the succeeded runs' outputs and run each shard merge whose shard roots are here
    (not one a later launch merged), adding what was taken and merged to the record."""
    record.refresh()
    paths, merges = record.pulls(), record.data.get("merges") or []
    check_writes(record, merges=True)
    report: dict[str, Any] = {"pulled": [], "merged": []}
    if not paths and not merges:
        return report
    states = backend.states(record)
    live = [job_id for job_id, entry in states.items() if entry["state"] not in ENDED]
    if live:
        raise LaunchError(f"{record.launch_id}: {', '.join(live)} not ended ({summary(states)}): wait first")
    outputs = {job["job_id"]: job["outputs"] for job in record.jobs}
    unsucceeded = {path for job_id, entry in states.items() if not succeeded(entry)
                   for path in outputs.get(job_id, ())}
    others = later_records(record)
    pulled_elsewhere = {path for other in others for path in other.data.get("pulled") or ()}
    merged_elsewhere = {merge_key(entry): other.launch_id for other in others
                        for entry in other.data.get("merged") or ()}
    errors = {}
    for path in paths:
        if path in unsucceeded:
            continue
        try:
            backend.pull(record, path)
        except _PER_ITEM_ERRORS as error:
            errors[path] = str(error)
        else:
            report["pulled"].append(path)
    if errors:
        report["pull_errors"] = errors
    held = [path for path in paths if path in unsucceeded and not _brought_back([path], pulled_elsewhere)]
    if held:
        report["held"] = held
    skipped, due, elsewhere = [], [], []
    for entry in merges:
        if merge_key(entry) in merged_elsewhere:
            elsewhere.append({**entry, "launch_id": merged_elsewhere[merge_key(entry)]})
            continue
        runs = _merge_runs(entry, states)
        missed = {job_id: state for job_id, state in runs.items()
                  if not succeeded(state) and not _brought_back(outputs.get(job_id, ()), pulled_elsewhere)}
        unpulled = {path for job_id in runs for path in outputs.get(job_id, ())} & errors.keys()
        if missed:
            skipped.append({**entry, "reason": f"{summary(missed)} of its {len(runs)} shard run(s) here, and no "
                                               "other launch's pull brought their shard roots back: the pull of "
                                               "the relaunch that finishes them merges it"})
        elif unpulled:
            skipped.append({**entry, "reason": f"{len(unpulled)} of its path(s) were not pulled: a merge refuses "
                                               "missing shards"})
        else:
            due.append(entry)
    if skipped:
        report["merge_skipped"] = skipped
    if elsewhere:
        report["merged_elsewhere"] = elsewhere
    if due and merge is None:
        from ruleforge.evaluate.shard_merge import merge_condition_shards as merge
    root, refused = artifact_root() if due else None, []
    for entry in due:
        try:
            merge(root / entry["campaign_root"], entry["condition_id"], num_shards=entry.get("num_shards"))
        except _MERGE_ERRORS as error:
            refused.append({**entry, "error": f"{type(error).__name__}: {error}"})
        else:
            report["merged"].append(entry)
    if refused:
        report["merge_errors"] = refused
    _note_pull(record, report)
    return report


def _note_pull(record: LaunchRecord, report: Mapping[str, Any]) -> None:
    pulled = list(dict.fromkeys([*record.data.get("pulled", ()), *report["pulled"]]))
    known = {merge_key(entry) for entry in record.data.get("merged", ())}
    merged = [*record.data.get("merged", ()), *(
        {"campaign_root": entry["campaign_root"], "condition_id": entry["condition_id"],
         "num_shards": entry.get("num_shards")} for entry in report["merged"] if merge_key(entry) not in known)]
    if pulled != record.data.get("pulled", []) or merged != record.data.get("merged", []):
        record.data.update(pulled=pulled, merged=merged)
        record.save()


def job_writes(data: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return what each job of a launch writes that no other launch's run may write while it runs."""
    run = data.get("train")
    return {job["job_id"]: {*job.get("outputs", ()), *((run["output_dir"],) if run else ())}
            for job in data.get("jobs", ())}


def recorded_launches(root: Path) -> list[LaunchRecord]:
    """Return the launches recorded under `root` and under its recorded name beside it, by record path."""
    root = Path(root)
    return [LaunchRecord(path.parent, {}).refresh()
            for directory in (root, root.with_name(RECORDED_LAUNCHES_DIRNAME))
            for path in sorted(directory.glob(f"*/{RECORD_FNAME}"))]


def claim_roots(record: LaunchRecord, roots: Iterable[Path]) -> None:
    """Name `record` beside each campaign root it writes, under the root's lock, so a launch under another artifact root
    finds it."""
    text = json.dumps({"launch_id": record.launch_id, "record": str(record.directory), "data": record.data},
                      indent=1, sort_keys=True) + "\n"
    for root in roots:
        write_text_atomically(writers_dir(root) / f"{record.launch_id}.json", text)


def claimed_launches(roots: Iterable[Path]) -> list[LaunchRecord]:
    """Return the launches that claimed any of `roots`, each from its record where it still exists."""
    found = []
    for root in roots:
        for path in sorted(writers_dir(root).glob("*.json")):
            claim = json.loads(path.read_text(encoding="utf-8"))
            record = LaunchRecord(Path(claim["record"]), claim["data"])
            found.append(record.refresh() if record.path.is_file() else record)
    return found


def launches_writing(root: Path, roots: Iterable[Path]) -> list[LaunchRecord]:
    """Return the launches recorded under `root` and those that claimed any of `roots`, each once."""
    found: dict[str, LaunchRecord] = {}
    for record in [*recorded_launches(root), *claimed_launches(roots)]:
        found.setdefault(str(record.directory), record)
    return list(found.values())


def live_launches(
    launches: Iterable[LaunchRecord], shared_of: Callable[[LaunchRecord], Mapping[str, Any]],
    states_of: Callable[[LaunchRecord], Mapping[str, Mapping[str, Any]]], *, ignore: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Return the `launches` (but `ignore`'s) with a job `shared_of` names whose run has not ended: each launch's id,
    what its live jobs share, their states and whether its record is gone."""
    found = []
    for other in launches:
        shared = dict(shared_of(other))
        if not shared or other.launch_id in ignore:
            continue
        try:
            states = states_of(other)
        except _PER_ITEM_ERRORS as error:
            states = {job_id: {"state": UNKNOWN, "error": str(error)} for job_id in shared}
        live = {job_id: states.get(job_id, {"state": UNKNOWN}) for job_id in shared
                if states.get(job_id, {"state": UNKNOWN})["state"] not in ENDED}
        if live:
            found.append({"launch_id": other.launch_id, "shared": {job: shared[job] for job in live}, "states": live,
                          "record_gone": not other.path.is_file()})
    return found


def live_writers(
    root: Path, paths: Collection[str], states_of: Callable[[LaunchRecord], Mapping[str, Mapping[str, Any]]], *,
    ignore: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Return the launches recorded under `root` (but `ignore`'s) with a live run writing one of `paths`."""
    wanted = set(paths)

    def shared_of(other: LaunchRecord) -> dict[str, set[str]]:
        return {job_id: written & wanted for job_id, written in job_writes(other.data).items() if written & wanted}

    return [{"launch_id": entry["launch_id"], "paths": sorted(set().union(*entry["shared"].values())),
             "states": entry["states"]}
            for entry in live_launches(recorded_launches(root), shared_of, states_of, ignore=ignore)]


def later_records(record: LaunchRecord) -> list[LaunchRecord]:
    """Return the other launches recorded beside `record` and made at or after it, oldest first."""
    return sorted((other for other in other_records(record) if other.data["created_at"] >= record.data["created_at"]),
                  key=lambda other: other.data["created_at"])


def stop(record: LaunchRecord, backend: RunControl, *, reason: Optional[str] = None,
         details: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Stop every named run that has not ended and log the request in `stop.json`; the launch submits no later job."""
    check_writes(record)
    states = backend.states(record.refresh())
    stopped, errors = {}, {}
    for job_id, entry in states.items():
        if entry["state"] in (*ENDED, SUBMITTING):
            continue
        try:
            stopped[job_id] = backend.stop(record, record.run_for(job_id))
        except _PER_ITEM_ERRORS as error:
            errors[job_id] = str(error)
    submitting = [job_id for job_id, entry in states.items() if entry["state"] == SUBMITTING]
    entry = {"requested_at": utc_now(), "reason": reason, **(details or {}), "states": states, "stopped": stopped,
             **({"stop_errors": errors} if errors else {}), **({"submitting": submitting} if submitting else {})}
    log = record.directory / STOP_FNAME
    history = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else []
    write_text_atomically(log, json.dumps([*history, entry], indent=1, sort_keys=True) + "\n")
    return entry
