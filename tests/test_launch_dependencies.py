from __future__ import annotations

import pytest

from ruleforge.launch.backends import SlurmBackend
from ruleforge.launch.environment import LaunchError
from tests.test_launch_backends import _executors, _record, _repo, _spec


def test_a_dependent_job_is_rendered_after_its_jobs_by_their_ids(tmp_path):
    backend = SlurmBackend(_repo(tmp_path))
    first, second, last = backend.render([_spec("a"), _spec("b"), _spec("z", depends_on=("a", "b"))])
    assert "dependency" not in first["submitit"]["slurm_additional_parameters"]
    assert last["submitit"]["slurm_additional_parameters"]["dependency"] == "afterok:{a}:{b}"


def test_a_dependent_job_is_submitted_after_its_jobs_with_their_slurm_ids(tmp_path):
    log, factory = _executors()
    backend = SlurmBackend(_repo(tmp_path), executor=factory, echo=lambda line: None)
    record = _record(tmp_path, backend, [_spec("a"), _spec("b"), _spec("z", depends_on=("a", "b"))])
    backend.submit(record)
    assert [run["job_id"] for run in record.runs] == ["a", "b", "z"]
    (_, members, batched, _), _, (_, analysis, single, _) = log["jobs"]
    assert batched and "dependency" not in members["slurm_additional_parameters"] and not single
    assert analysis["slurm_additional_parameters"]["dependency"] == "afterok:100_0:100_1"
    rendered = record.jobs[2]["request"]["submitit"]["slurm_additional_parameters"]
    assert rendered["dependency"] == "afterok:{a}:{b}"


def test_a_job_depending_on_one_the_launch_does_not_submit_is_refused(tmp_path):
    with pytest.raises(LaunchError, match="z depends on y, which this launch does not submit"):
        SlurmBackend(_repo(tmp_path)).render([_spec("a"), _spec("z", depends_on=("y",))])


def test_jobs_depending_on_one_another_are_refused_before_any_is_submitted(tmp_path):
    log, factory = _executors()
    backend = SlurmBackend(_repo(tmp_path), executor=factory, echo=lambda line: None)
    record = _record(tmp_path, backend, [_spec("a", depends_on=("b",)), _spec("b", depends_on=("a",))])
    with pytest.raises(LaunchError, match="depend on one another in a cycle"):
        backend.submit(record)
    assert log["jobs"] == []
