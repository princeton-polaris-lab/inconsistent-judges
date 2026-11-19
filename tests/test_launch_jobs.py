from __future__ import annotations

import pytest

from ruleforge.launch import environment, jobs
from ruleforge.launch.jobs import JobSpec, Resources
from ruleforge.launch.environment import LaunchError

SLURM = {"partition": "gpu-partition", "account": "lab-account"}


def _spec(**overrides) -> JobSpec:
    values = {"job_id": "j", "kind": "job", "step": 0, "argv": ("python", "x.py"),
              "resources": Resources("gpu", cpus=8, gpus=2, mem_gb=96, timeout_min=60)}
    values.update(overrides)
    return JobSpec(**values)


def test_the_retired_runners_job_renderers_are_gone():
    for gone in ("modal_command", "runpod_command", "pod_command", "pod_assignments", "MODAL_SCRIPT"):
        assert not hasattr(jobs, gone)


def test_a_spec_s_new_fields_default_so_older_specs_and_plans_read_the_same():
    spec = JobSpec("merge-50", "merge", 50, ("python", "m.py"), Resources("cpu", 8, 0, 80, 60, "normal"),
                   ("a",), ("out",))
    assert (spec.workdir, spec.env, spec.resources.resumable) == (".", (), False)
    payload = spec.to_json()
    assert payload["groups"] == [] and payload["env"] == [] and payload["workdir"] == "."
    assert list(payload) == ["job_id", "kind", "step", "argv", "resources", "depends_on", "outputs", "notes",
                             "workdir", "groups", "env"]
    assert payload["depends_on"] == ["a"] and payload["resources"]["resumable"] is False
    assert Resources(**payload["resources"]) == spec.resources


@pytest.mark.parametrize("env, message", [
    (("NOEQUALS",), "expected NAME=VALUE"),
    (("1BAD=x",), "not an environment variable name"),
    (("RULEFORGE_ARTIFACT_ROOT=/x",), "the launcher owns that variable"),
    (("SLURM_JOB_ID=1",), "the launcher owns that variable"),
    (("HF_TOKEN=x",), "looks like a credential"),
    (("A=1", "A=2"), "names A more than once"),
])
def test_a_job_s_exports_are_checked(env, message):
    with pytest.raises(LaunchError, match=message):
        _spec(env=env)


def test_a_workdir_stays_below_the_checkout():
    assert _spec(workdir="scripts/sampling").workdir == "scripts/sampling"
    for bad in ("../x", "/abs"):
        with pytest.raises(ValueError, match="must name a path below"):
            _spec(workdir=bad)


def test_exports_are_literal_shell_lines():
    assert environment.export_lines(("A=1", "B=two words", "C=$HOME")) == [
        "export A=1", "export B='two words'", "export C='$HOME'"]


def test_submitit_requeues_only_a_resumable_job_and_leaves_a_null_account_to_the_cluster():
    resumable = _spec(resources=Resources("gpu", 8, 1, 96, 60, "jit", resumable=True))
    parameters = jobs.submitit_parameters(resumable, {"partition": "p", "account": None})
    assert parameters["slurm_additional_parameters"] == {"qos": "jit", "requeue": True}
    assert parameters["gpus_per_node"] == 1 and parameters["slurm_partition"] == "p"
    assert jobs.submitit_parameters(_spec(), SLURM)["slurm_additional_parameters"] == {
        "account": "lab-account", "no_requeue": True}
    assert jobs.submitit_parameters(_spec(), {**SLURM, "exclude": ["n1", "n2"]})["slurm_additional_parameters"] == {
        "account": "lab-account", "no_requeue": True, "exclude": "n1,n2"}
    assert "exclude" not in jobs.submitit_parameters(_spec(), {**SLURM, "exclude": []})["slurm_additional_parameters"]


def test_a_local_action_is_no_scheduler_job():
    with pytest.raises(LaunchError, match="local action"):
        jobs.submitit_parameters(_spec(resources=Resources(jobs.LOCAL)), SLURM)


def test_an_axis_is_a_list_of_values_whose_whole_number_ranges_are_spread():
    assert jobs.parse_axis("judge_model.seed=1..10") == ("judge_model.seed", tuple(map(str, range(1, 11))))
    assert jobs.parse_axis("n=-1..1,7,x") == ("n", ("-1", "0", "1", "7", "x"))
    assert jobs.parse_axis("model@judge_model=a,b") == ("model@judge_model", ("a", "b"))
    assert jobs.parse_axis("+x.y=3..3") == ("+x.y", ("3",))
    assert jobs.parse_axis("x=1.5,a..b,..,1..2.5") == ("x", ("1.5", "a..b", "..", "1..2.5"))


@pytest.mark.parametrize("text, message", [
    ("seed", "expected KEY=VALUES"),
    ("=1", "expected KEY=VALUES"),
    ("-m=1", "expected KEY=VALUES"),
    ("a b=1", "expected KEY=VALUES"),
    ("x=", "'' is not a value"),
    ("x=1,,2", "'' is not a value"),
    ("x=3..1", "'3..1' is not a value"),
    ("x=1..3,2", "gives 2 more than once"),
])
def test_an_axis_that_is_not_one_is_refused(text, message):
    with pytest.raises(LaunchError, match=message):
        jobs.parse_axis(text)


def test_a_job_fans_out_over_the_product_of_its_axes_the_first_outermost():
    spec = _spec(argv=("python", "t.py", "hydra.run.dir=s/{member}", "keep={other}", "${x}"), env=("M={member}",),
                 outputs=("o/{member}",))
    fanned = jobs.members(spec, ["a=1..2", "b=x,y,z"])
    assert [job.job_id for job in fanned] == [f"j-{index}" for index in range(6)]
    assert [job.argv[-2:] for job in fanned] == [(f"a={a}", f"b={b}") for a in "12" for b in "xyz"]
    assert fanned[4].argv[:5] == ("python", "t.py", "hydra.run.dir=s/4", "keep={other}", "${x}")
    assert fanned[4].env == ("M=4",) and fanned[4].outputs == ("o/4",)
    assert {job.resources for job in fanned} == {spec.resources}
    [alone] = jobs.members(spec)
    assert alone.job_id == "j" and alone.argv == ("python", "t.py", "hydra.run.dir=s/0", "keep={other}", "${x}")
    for axes in (["a=1", "b=2", "c=3"], ["a=1", "a=2"]):
        with pytest.raises(LaunchError, match="at most 2 axes, each on its own key"):
            jobs.members(spec, axes)


def test_members_that_would_write_one_output_are_refused():
    spec = _spec(outputs=("o/{member}", "shared/log.txt"))
    with pytest.raises(LaunchError, match="every member would write shared/log.txt: put {member} in its path"):
        jobs.members(spec, ["a=1..2"])
    assert jobs.members(spec)[0].outputs == ("o/0", "shared/log.txt")
    assert [job.outputs for job in jobs.members(_spec(outputs=("o/{member}",)), ["a=1..2"])] == [("o/0",), ("o/1",)]
