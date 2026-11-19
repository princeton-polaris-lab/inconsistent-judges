from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.launch.environment import LAUNCHER_CONFIG
from ruleforge.launch import cli, control, records
from ruleforge.launch.backends import LocalBackend
from ruleforge.launch.cli import main
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.records import LaunchRecord
from tests.support.git import _git, make_repo
from tests.test_launch_backends import Completed, _executors

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clean checkout with the launcher config and a spec directory; the artifact root beside it."""
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu-partition")
    monkeypatch.delenv("RULEFORGE_SLURM_ACCOUNT", raising=False)
    checkout = make_repo(tmp_path)
    config = checkout / LAUNCHER_CONFIG
    config.parent.mkdir(parents=True)
    shutil.copy(REPO / LAUNCHER_CONFIG, config)
    (checkout / control.SPECS_DIR).mkdir(parents=True)
    (checkout / "work" / "sampling").mkdir(parents=True)
    (checkout / "work" / "sampling" / "keep").write_text("")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "launcher")
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    (tmp_path / "artifacts").mkdir()
    return checkout


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").strip()


def _spec(repo: Path, name: str, text: str) -> Path:
    """A committed spec: an uncommitted one would make the checkout dirty."""
    path = repo / control.SPECS_DIR / f"{name}.yaml"
    path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", name)
    return path


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _records(tmp_path: Path) -> list[Path]:
    root = tmp_path / "artifacts" / ".ruleforge-launch"
    return sorted(root.iterdir()) if root.is_dir() else []


SPEC = """\
backend: slurm
gpu-count: 1
timeout-min: 90
env: [HF_HUB_OFFLINE=1, "RUN={campaign_id}"]
output: ["sampling/{campaign_id}"]
workdir: work/sampling
resumable: true
vars:
  campaign_id: null
  stage: dev
args:
  - python
  - run_campaign.py
  - campaign_id={campaign_id}
  - grpo_dev.stage={stage}
  - "hydra.run.dir=${{campaign_root}}/{launch_id}"
  - "+training.policies.x={{adapter_path: ${oc.env:RULEFORGE_ARTIFACT_ROOT}/a}}"
"""


def test_a_spec_fills_only_its_declared_names_and_leaves_hydra_alone(repo, tmp_path, capsys):
    _spec(repo, "arm", SPEC)
    assert main(["job", "--spec", "arm", "--var", "campaign_id=c1", "--dry-run", "--json"], repo=repo) == 0
    printed = _json(capsys)
    [job] = printed["jobs"]
    launch_id = printed["launch_id"]
    assert job["argv"] == ["python", "run_campaign.py", "campaign_id=c1", "grpo_dev.stage=dev",
                           "hydra.run.dir=${{campaign_root}}/" + launch_id,
                           "+training.policies.x={{adapter_path: ${oc.env:RULEFORGE_ARTIFACT_ROOT}/a}}"]
    assert job["job_id"] == "arm" and job["env"] == ["HF_HUB_OFFLINE=1", "RUN=c1"] and job["groups"] == []
    assert job["outputs"] == ["sampling/c1"] and job["workdir"] == "work/sampling"
    assert job["resources"] == {"kind": "gpu", "cpus": 4, "gpus": 1, "mem_gb": 24, "timeout_min": 90, "qos": None,
                                "resumable": True}
    assert printed["spec"]["path"] == "configs/launch/arm.yaml" and printed["spec"]["vars"] == {
        "campaign_id": "c1", "stage": "dev"}
    assert len(printed["spec"]["sha256"]) == 64 and printed["dry_run"] is True and _records(tmp_path) == []


@pytest.mark.parametrize("text, argv, message", [
    ("backend: slurm\nbogus: 1\n", [], "unknown key\\(s\\) bogus"),
    ("backend: slurm\ndry-run: true\n", [], "unknown key\\(s\\) dry-run"),
    ("backend: slurm\nvars: {a: null}\n", [], "needs --var for a"),
    ("backend: slurm\nvars: [a]\n", ["--var", "a=1", "--var", "b=2"], "--var b: not declared"),
    ("backend: slurm\nvars: 3\n", [], "vars is a mapping"),
    ("backend: slurm\nvars: {launch_id: x}\n", [], "not a built-in \\(launch_id, member, block, artifact_root, "
                                                  "model_root\\)"),
    ("backend: slurm\nvars: {artifact_root: x}\n", [], "vars.artifact_root: .* not a built-in"),
    ("backend: slurm\nvars: {member: x}\n", [], "vars.member: .* not a built-in"),
    ("backend: local\naxis: [seed]\nargs: [x]\n", [], "--axis 'seed': expected KEY=VALUES"),
    ("backend: local\naxis: [a=1, b=2, c=3]\nargs: [x]\n", [], "at most 2 axes"),
    ("backend: slurm\nvars: {9x: y}\n", [], "vars.9x"),
    ("- a list\n", [], "a mapping of options"),
    ("backend: slurm\nargs: python\n", [], "args is a list"),
    ("backend: slurm\nargs: [[nested]]\n", [], "args: \\['nested'\\] is not text"),
    # A per-invocation waiver is not a spec's to give.
    ("backend: local\nallow-dirty: true\nargs: [x]\n", [], "unknown key\\(s\\) allow-dirty"),
    ("backend: local\nallow-active-runs: true\nargs: [x]\n", [], "unknown key\\(s\\) allow-active-runs"),
    ("backend: local\nno-wait: true\nargs: [x]\n", [], "unknown key\\(s\\) no-wait"),
    ("backend: local\nargs: [true]\n", [], "args: True is not text: quote it"),
    ("backend: local\nvars: {x: false}\nargs: [x]\n", [], "vars.x: False is not text"),
    ("backend: slurm\ncpus: true\nargs: [x]\n", [], "cpus: True is not text"),
    ("backend: slurm\ntimeout-min: 1:30\nargs: [x]\n", [], "timeout-min: '1:30' is not a int"),
    ("backend: slurm\ngroup: [serve]\nargs: [x]\n", [], "unknown key\\(s\\) group"),
    ("backend: slurm\ngpu: H100\nargs: [x]\n", [], "unknown key\\(s\\) gpu"),
    ("backend: slurm\npod: p1\nargs: [x]\n", [], "unknown key\\(s\\) pod"),
    ("backend: slurm\nimage-like: r.json\nargs: [x]\n", [], "unknown key\\(s\\) image-like"),
    ("backend: slurm\nprofile: site\nargs: [x]\n", [], "unknown key\\(s\\) profile"),
    ("backend: nomad\nargs: [x]\n", [], "backend: 'nomad' is not one of"),
    ("backend: slurm\ncpus: many\nargs: [x]\n", [], "cpus: 'many' is not a int"),
    ("backend: slurm\nresumable: yes please\nargs: [x]\n", [], "resumable: true or false"),
    ("backend: slurm\nvars: {a: {b: c}}\n", [], "vars.a"),
])
def test_a_spec_is_refused_where_it_is_not_what_the_verb_takes(repo, capsys, text, argv, message):
    _spec(repo, "bad", text)
    assert main(["job", "--dry-run", "--spec", "bad", *argv], repo=repo) == 2
    assert re.search(message, capsys.readouterr().err)


def test_a_spec_keeps_the_text_written_where_yaml_1_1_would_not(repo, capsys):
    _spec(repo, "texts", "backend: slurm\nname: no\ncpus: 010\ngpu-count: 0\nvars: {x: 010, when: 2026-09-26}\n"
                         "args: [python, x.py, --lr, 1.0e-5, --wandb, off, --n, 12, --id, '{x}', 0x1f, 1_000, '{when}']\n")
    assert main(["job", "--spec", "texts", "--dry-run", "--json"], repo=repo) == 0
    [job] = _json(capsys)["jobs"]
    assert job["argv"] == ["python", "x.py", "--lr", "1.0e-5", "--wandb", "off", "--n", "12", "--id", "010", "0x1f",
                           "1_000", "2026-09-26"]
    assert job["job_id"] == "no" and job["resources"]["cpus"] == 10 and job["resources"]["gpus"] == 0


def test_a_preset_is_its_spec_with_some_values_given_and_its_record_names_both(repo, tmp_path, capsys):
    _spec(repo, "arm", SPEC)
    preset = _spec(repo, "arm_final", "# one stage's values\npreset: arm\nvars:\n  stage: final\n")
    assert main(["job", "--spec", "arm_final", "--var", "campaign_id=c1", "--dry-run", "--json"], repo=repo) == 0
    printed = _json(capsys)
    assert main(["job", "--spec", "arm", "--name", "arm_final", "--var", "campaign_id=c1", "--var", "stage=final",
                 "--dry-run", "--json"], repo=repo) == 0
    direct = _json(capsys)

    def submitted(launch: dict) -> str:
        return json.dumps(launch["jobs"], sort_keys=True).replace(launch["launch_id"], "L")

    assert submitted(printed) == submitted(direct) and printed["options"] == direct["options"]
    [job] = printed["jobs"]
    assert job["job_id"] == "arm_final" and job["argv"][3] == "grpo_dev.stage=final"
    assert printed["spec"]["path"] == "configs/launch/arm_final.yaml"
    assert printed["spec"]["vars"] == {"campaign_id": "c1", "stage": "final"}
    import hashlib

    assert printed["spec"]["sha256"] == hashlib.sha256(preset.read_bytes()).hexdigest()
    assert printed["spec"]["preset_of"] == {
        "path": "configs/launch/arm.yaml",
        "sha256": hashlib.sha256((repo / control.SPECS_DIR / "arm.yaml").read_bytes()).hexdigest()}
    assert "preset_of" not in direct["spec"]
    assert main(["job", "--spec", "arm_final", "--var", "campaign_id=c1", "--var", "stage=dev", "--dry-run",
                 "--json"], repo=repo) == 0
    assert _json(capsys)["jobs"][0]["argv"][3] == "grpo_dev.stage=dev"


@pytest.mark.parametrize("text, message", [
    ("preset: arm\nbackend: local\n", "a preset gives its spec \\(arm\\) and values of the spec's vars, nothing "
                                       "else \\(not backend\\)"),
    ("preset: arm\nvars: {other: x}\n", "a preset gives values of .*arm.yaml's vars \\(campaign_id, stage\\); it "
                                         "declares no other"),
    ("preset: arm\nvars: {stage: null}\n", "; stage given no value"),
    ("preset: arm_dev\n", "its spec .*arm_dev.yaml is a preset itself: name the spec it presets"),
    ("preset: missing\n", "no launch spec 'missing'"),
    ("preset: [arm]\n", "preset: \\['arm'\\] is not text"),
    ("preset: arm\nvars: {stage: true}\n", "vars.stage: True is not text"),
])
def test_a_preset_is_refused_where_it_is_more_than_its_spec_s_values(repo, capsys, text, message):
    _spec(repo, "arm", SPEC)
    _spec(repo, "arm_dev", "preset: arm\nvars: {stage: dev}\n")
    _spec(repo, "bad", text)
    assert main(["job", "--dry-run", "--spec", "bad", "--var", "campaign_id=c1"], repo=repo) == 2
    assert re.search(message, capsys.readouterr().err)


def test_specs_are_found_by_path_or_name(repo, tmp_path, capsys):
    outside = tmp_path / "elsewhere.yml"
    outside.write_text("backend: local\nargs: ['true']\n")
    _spec(repo, "named", "backend: local\nargs: ['true']\n")
    named = repo / control.SPECS_DIR / "named.yaml"
    assert cli.resolve_spec("named", repo) == cli.resolve_spec("named.yaml", repo) == named
    assert cli.resolve_spec(str(outside), repo) == outside
    with pytest.raises(LaunchError, match="no launch spec 'missing'"):
        cli.resolve_spec("missing", repo)
    assert main(["job", "--spec", str(outside), "--dry-run", "--json"], repo=repo) == 0
    assert _json(capsys)["spec"]["path"] == str(outside.resolve())
    _spec(repo, "empty", "")
    assert main(["job", "--spec", "empty", "--backend", "local", "--dry-run", "--json", "--", "true"], repo=repo) == 0
    assert _json(capsys)["jobs"][0]["job_id"] == "empty"


def test_variables_are_names_given_once():
    assert cli.parse_vars(["a=1", "b=x=y"]) == {"a": "1", "b": "x=y"}
    for bad, message in ((["a"], "expected NAME=VALUE"), (["1a=2"], "expected NAME=VALUE"),
                         (["a=1", "a=2"], "gives a twice")):
        with pytest.raises(LaunchError, match=message):
            cli.parse_vars(bad)
    assert cli.fill("${x} {x} {{x}} {y}", {"x": "1"}) == "${x} 1 {1} {y}"


@pytest.mark.parametrize("argv, message", [
    (["--", "true"], "--backend is required"),
    (["--backend", "local", "--qos", "jit", "--", "true"], "--qos applies to slurm only, not local"),
    (["--backend", "local", "--pull", "--", "true"], "--pull comes with --wait"),
    (["--backend", "local"], "no command"),
    (["--backend", "local", "--", ], "no command"),
    (["--backend", "local", "--cpus", "0", "--", "true"], "the rest at least 1"),
    (["--backend", "slurm", "--gpu-count", "-1", "--", "true"], "GPUs at least 0"),
    (["--backend", "local", "--resumable", "--", "true"], "--resumable applies to slurm only, not local"),
    (["--backend", "local", "--gpu-count", "1", "--", "true"], "--gpu-count applies to slurm only"),
    (["--backend", "local", "--timeout-min", "5", "--", "true"], "--timeout-min applies to slurm only"),
    (["--backend", "local", "--cpus", "8", "--", "true"], "--cpus applies to slurm only, not local"),
    (["--backend", "local", "--name", "a b", "--", "true"], "--name 'a b'"),
    (["--backend", "local", "--var", "a=1", "--", "true"], "give --spec"),
    (["--backend", "local", "--env", "HF_TOKEN=x", "--", "true"], "looks like a credential"),
    (["--backend", "local", "--workdir", "../up", "--", "true"], "below the root"),
    (["--backend", "local", "--expect-commit", "bbbbbbb", "--", "true"], "not --expect-commit bbbbbbb"),
])
def test_the_job_verb_refuses_before_it_spends(repo, capsys, argv, message):
    assert main(["job", "--dry-run", *argv], repo=repo) == 2
    assert message in capsys.readouterr().err


def test_the_cluster_comes_from_the_environment(repo, capsys, monkeypatch):
    def dry_run() -> dict:
        assert main(["job", "--backend", "slurm", "--dry-run", "--json", "--", "true"], repo=repo) == 0
        return _json(capsys)["jobs"][0]["request"]["submitit"]

    parameters = dry_run()
    assert parameters["slurm_partition"] == "gpu-partition" and "account" not in parameters[
        "slurm_additional_parameters"]
    monkeypatch.setenv("RULEFORGE_SLURM_ACCOUNT", "myaccount")
    assert dry_run()["slurm_additional_parameters"]["account"] == "myaccount"
    monkeypatch.delenv("RULEFORGE_SLURM_PARTITION")
    assert main(["job", "--backend", "slurm", "--dry-run", "--", "true"], repo=repo) == 2
    assert "RULEFORGE_SLURM_PARTITION" in capsys.readouterr().err


def test_a_dirty_checkout_is_refused_unless_allowed(repo, capsys):
    (repo / "src" / "pkg" / "a.py").write_text("A = 2\n")
    assert main(["job", "--backend", "local", "--dry-run", "--", "true"], repo=repo) == 2
    assert "uncommitted changes" in capsys.readouterr().err
    assert main(["job", "--backend", "local", "--dry-run", "--allow-dirty", "--json", "--", "true"], repo=repo) == 0
    assert _json(capsys)["code"]["dirty"] is True


def test_outputs_are_artifact_paths(repo, tmp_path, capsys):
    absolute = tmp_path / "artifacts" / "sampling" / "x"
    assert main(["job", "--backend", "local", "--output", "artifact://sampling/y", "--output", str(absolute),
                 "--dry-run", "--json", "--", "true"], repo=repo) == 0
    [job] = _json(capsys)["jobs"]
    assert job["outputs"] == ["sampling/y", "sampling/x"] and job["groups"] == []
    assert main(["job", "--backend", "local", "--output", "/elsewhere/x", "--dry-run", "--", "true"], repo=repo) == 2


def test_a_dry_run_prints_each_backends_request(repo, capsys):
    assert main(["job", "--backend", "slurm", "--expect-commit", _head(repo)[:10], "--dry-run", "--", "true"],
                repo=repo) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "== job" and out[1] == "true" and json.loads(out[2])["slurm_additional_parameters"]["qos"] == (
        "normal") and out[-1] == "== dry run: nothing was submitted"
    assert main(["job", "--backend", "local", "--dry-run", "--json", "--", "true"], repo=repo) == 0
    printed = _json(capsys)
    assert printed["preview"] is None and printed["jobs"][0]["request"]["argv"][:2] == ["bash", "-c"]
    assert printed["options"] == {"allow_dirty": False}


def test_a_local_job_waits_and_reports_in_json(repo, tmp_path, capsys):
    assert main(["job", "--backend", "local", "--wait", "--json", "--workdir", "work/sampling", "--",
                 "bash", "-c", "echo made; pwd"], repo=repo) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["wait"] == 0 and result["states"] == {"job": {"state": "succeeded", "exit_code": 0}}
    assert "made" in captured.err and "== launch" in captured.err
    [directory] = _records(tmp_path)
    record = LaunchRecord.load(directory)
    assert record.data["state"] == "submitted" and record.data["code"]["commit"] == _head(repo)
    assert result["record"] == str(directory) and result["runs"] == record.runs
    assert main(["job", "--backend", "local", "--wait", "--", "bash", "-c", "exit 5"], repo=repo) == 1
    assert "1 failed" in capsys.readouterr().out
    assert main(["job", "--backend", "local", "--", "true"], repo=repo) == 0
    assert "follow it with python -m ruleforge.launch.cli status" in capsys.readouterr().out
    assert main(["job", "--backend", "local", "--no-wait", "--json", "--", "bash", "-c", "exit 5"], repo=repo) == 1
    assert _json(capsys)["states"]["job"] == {"state": "failed", "exit_code": 5}


def test_launches_from_one_commit_in_one_second_never_share_an_id(repo, tmp_path, capsys, monkeypatch):
    second = f"20261011T185130Z-{_head(repo)[:8]}"
    drawn = iter([f"{second}-2d84", f"{second}-2d84", f"{second}-7ccb"])
    monkeypatch.setattr(records, "new_run_id", lambda commit: next(drawn))
    for _ in range(2):
        assert main(["job", "--backend", "local", "--json", "--", "true"], repo=repo) == 0
        assert _json(capsys)["states"]["job"]["state"] == "succeeded"
    assert [path.name for path in _records(tmp_path)] == [f"{second}-2d84", f"{second}-7ccb"]


def test_the_command_goes_after_the_options_and_a_double_dash(repo, capsys):
    # An option typed after the command's first word would be the command's: it is refused instead.
    with pytest.raises(SystemExit) as refused:
        main(["job", "--backend", "local", "python", "x.py", "--dry-run"], repo=repo)
    assert refused.value.code == 2 and "unrecognized arguments: python x.py" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["status", "L1", "--", "x"], repo=repo)
    assert "status takes no command after --" in capsys.readouterr().err
    assert main(["job", "--backend", "local", "--dry-run", "--json", "--", "echo", "--", "--dry-run"], repo=repo) == 0
    assert _json(capsys)["jobs"][0]["argv"] == ["echo", "--", "--dry-run"]


def test_a_slurm_job_is_followed_stopped_and_waited_on_by_its_record(repo, tmp_path, capsys):
    log, executor = _executors()
    assert main(["job", "--backend", "slurm", "--resumable", "--json", "--", "python", "x.py"], repo=repo,
                executor=executor) == 0
    launch_id = _json(capsys)["launch_id"]
    assert log["jobs"][0][1]["slurm_additional_parameters"]["requeue"] is True
    running = Completed({"sacct": [(0, "100|RUNNING\n", "")], "scancel": [(0, "", "")]})
    assert main(["status", launch_id], repo=repo, run=running) == 0
    out = capsys.readouterr().out
    assert "1 running" in out and "job  running       slurm_job=100" in out
    assert main(["stop", launch_id, "--reason", "enough", "--json"], repo=repo, run=running) == 0
    assert _json(capsys)["stopped"] == {"job": {"slurm_job": "100", "requested": True}}
    assert running.calls[-1] == ["scancel", "100"]
    assert main(["wait", launch_id, "--timeout-min", "0"], repo=repo, run=running) == 3
    assert "still running after the wait's limit" in capsys.readouterr().out
    stopped = Completed({"sacct": [(0, "100|CANCELLED by 1\n", "")]})
    assert main(["wait", launch_id, "--pull", "--json"], repo=repo, run=stopped) == 1
    result = _json(capsys)
    assert result["states"]["job"]["state"] == "stopped" and result["pull"] == {"pulled": [], "merged": []}
    assert main(["stop", launch_id], repo=repo, run=stopped) == 0
    assert "nothing (every named run has ended)" in capsys.readouterr().out
    failing = Completed({"sacct": [(0, "100|RUNNING\n", "")], "scancel": [(1, "", "scancel: error: denied")]})
    assert main(["stop", launch_id], repo=repo, run=failing) == 1
    assert "not stopped: job: scancel 100 failed (1): scancel: error: denied" in capsys.readouterr().out
    assert main(["status", "20990101T000000Z-none"], repo=repo) == 2
    assert "no launch record" in capsys.readouterr().err


def test_a_launch_that_dies_part_way_says_so_and_keeps_its_record(repo, tmp_path, capsys):
    log, executor = _executors(fail_at=1)
    assert main(["job", "--backend", "slurm", "--", "true"], repo=repo, executor=executor) == 2
    captured = capsys.readouterr()
    assert "stopped part-way; its record names the 1 run(s) it started or was starting" in captured.out
    assert "submitting failed: RuntimeError: sbatch: error: Batch job submission failed" in captured.err
    [directory] = _records(tmp_path)
    saved = json.loads((directory / "launch.json").read_text())
    assert saved["state"] == "failed" and "submission failed" in saved["error"]


@pytest.mark.parametrize("program", [["-m", "ruleforge", "jobs", "launch"], ["-m", "ruleforge.launch.cli"]])
def test_the_launcher_s_programs_run_main(repo_root, program):
    result = subprocess.run([sys.executable, *program, "job", "--help"], capture_output=True, text=True, check=False,
                            cwd=repo_root)
    assert result.returncode == 0 and "--expect-commit" in result.stdout


def test_a_launch_still_submitting_is_waited_on_and_its_stop_ends_the_submission(repo, tmp_path, capsys, monkeypatch):
    backend = LocalBackend(repo)
    spec = cli.build_job({"backend": "local"}, ["true"])[1]
    record = LaunchRecord.create(tmp_path / "artifacts" / ".ruleforge-launch" / "L1", {
        "launch_id": "L1", "backend": "local", "options": {}, "code": {},
        "jobs": [{**spec.to_json(), "request": backend.render([spec])[0]}]})
    assert main(["status", "L1"], repo=repo) == 0
    out = capsys.readouterr().out
    assert "1 submitting" in out and "== the launch is submitting" in out
    assert main(["wait", "L1", "--timeout-min", "0"], repo=repo) == 3
    capsys.readouterr()
    assert main(["stop", "L1", "--json"], repo=repo) == 1
    assert _json(capsys)["submitting"] == ["job"]
    assert main(["stop", "L1"], repo=repo) == 1
    assert "still being submitted: job; the launch submits no later job" in capsys.readouterr().out
    with pytest.raises(LaunchError, match="was requested while it was submitting"):
        backend.submit(record)

    def the_launch_goes_on(seconds):
        record.add_run({"job_id": "job", "exit_code": 0, "log": "job.log"})
        record.finish()

    monkeypatch.setattr(cli, "wait", functools.partial(records.wait, poll_seconds=0, sleep=the_launch_goes_on))
    assert main(["wait", "L1", "--json"], repo=repo) == 0
    assert _json(capsys)["states"] == {"job": {"state": "succeeded", "exit_code": 0}}


def test_a_spec_s_wait_is_turned_off_on_the_command_line(repo, capsys):
    _spec(repo, "waits", "backend: slurm\nwait: true\nargs: ['true']\n")
    run = Completed()
    assert main(["job", "--spec", "waits", "--no-wait", "--json"], repo=repo, run=run,
                executor=_executors()[1]) == 0
    assert "wait" not in _json(capsys) and run.calls == []


def test_a_launch_of_no_job_is_refused(repo):
    with pytest.raises(LaunchError, match="nothing to launch"):
        cli.launch(LocalBackend(repo), [], verb="job", launch_id="L1", code={}, spec_record=None,
                   out=cli.Output(False), refuse=dict)


FANOUT = """\
backend: slurm
gpu-count: 2
env: ["GROUP={launch_id}"]
axis: ["seed=1..3"]
vars: {task: a/b}
args: [python, train.py, "task={task}", "hydra.run.dir=../sweep/${x}/{launch_id}/{member}", "hydra.job.num={member}"]
"""


def test_a_spec_s_axis_fans_the_job_out_and_the_command_line_s_replaces_it(repo, capsys):
    _spec(repo, "fan", FANOUT)
    assert main(["job", "--spec", "fan", "--dry-run", "--json"], repo=repo) == 0
    printed = _json(capsys)
    launch_id = printed["launch_id"]
    assert [job["job_id"] for job in printed["jobs"]] == ["fan-0", "fan-1", "fan-2"]
    assert printed["jobs"][2]["argv"] == ["python", "train.py", "task=a/b",
                                          "hydra.run.dir=../sweep/${x}/" + launch_id + "/2", "hydra.job.num=2",
                                          "seed=3"]
    assert {tuple(job["env"]) for job in printed["jobs"]} == {(f"GROUP={launch_id}",)}
    assert main(["job", "--spec", "fan", "--axis", "seed=7", "--axis", "style=a,b", "--dry-run", "--json", "--",
                 "extra={member}"], repo=repo) == 0
    assert [job["argv"][-3:] for job in _json(capsys)["jobs"]] == [["extra=0", "seed=7", "style=a"],
                                                                    ["extra=1", "seed=7", "style=b"]]


def test_a_slurm_fan_out_is_one_array_of_its_members(repo, capsys):
    _spec(repo, "fan", FANOUT)
    log, executor = _executors()
    assert main(["job", "--spec", "fan", "--json"], repo=repo, executor=executor) == 0
    printed = _json(capsys)
    assert [run["slurm_job"] for run in printed["runs"]] == ["100_0", "100_1", "100_2"]
    assert all(batched for _, _, batched, _ in log["jobs"]) and log["jobs"][0][1]["name"] == printed["launch_id"]
    assert [command.argv[-1] for command, *_ in log["jobs"]] == ["seed=1", "seed=2", "seed=3"]


def test_the_command_line_overrides_the_spec_and_extends_its_args(repo, capsys):
    _spec(repo, "arm", SPEC)
    assert main(["job", "--spec", str(repo / control.SPECS_DIR / "arm.yaml"), "--var", "campaign_id=c1",
                 "--var", "stage=final", "--gpu-count", "2", "--name", "mine", "--timeout-min", "5",
                 "--no-resumable", "--dry-run", "--json", "--", "extra={campaign_id}", "keep={other}"], repo=repo) == 0
    printed = _json(capsys)
    [job] = printed["jobs"]
    assert job["job_id"] == "mine" and job["resources"]["gpus"] == 2 and job["resources"]["timeout_min"] == 5
    assert job["resources"]["resumable"] is False and job["request"]["max_num_timeout"] == 0
    assert job["argv"][3:4] == ["grpo_dev.stage=final"] and job["argv"][-2:] == ["extra=c1", "keep={other}"]


def test_a_missing_file_is_a_refusal_not_a_traceback(repo, capsys):
    (repo / LAUNCHER_CONFIG).unlink()
    assert main(["job", "--backend", "slurm", "--allow-dirty", "--dry-run", "--", "true"], repo=repo) == 2
    assert "python -m ruleforge.launch.cli job: FileNotFoundError: " in capsys.readouterr().err


def test_the_retired_cloud_options_are_gone(repo, capsys):
    for verb, option in (("job", ["--gpu", "H100"]), ("job", ["--gpus", "0"]), ("job", ["--pod", "p1"]),
                         ("job", ["--group", "serve"]), ("job", ["--image-like", "r.json"]),
                         ("job", ["--allow-active-runs"]), ("campaign", ["--parallel", "2"]),
                         ("campaign", ["--preflight", "remote"]), ("train", ["--push", "grpo_data/b"]),
                         ("pull", ["L1", "--allow-active-runs"]), ("job", ["--backend", "modal"])):
        with pytest.raises(SystemExit) as refused:
            main([verb, *option, "--", "true"] if verb != "pull" else [verb, *option], repo=repo)
        assert refused.value.code == 2, (verb, option)
    capsys.readouterr()


def test_a_slurm_job_s_output_is_taken_after_its_wait_and_holds_another_launch_back(repo, tmp_path, capsys):
    _, executor = _executors()
    argv = ["job", "--backend", "slurm", "--output", "training/r/merged/checkpoint-100", "--json", "--", "true"]
    done = Completed({"sacct": [(0, "100|COMPLETED\n", "")]})
    assert main([*argv[:-2], "--wait", "--pull", *argv[-2:]], repo=repo, executor=executor, run=done) == 0
    result = _json(capsys)
    assert result["pull"] == {"pulled": ["training/r/merged/checkpoint-100"], "merged": []}
    running = Completed({"sacct": [(0, "100|RUNNING\n101|RUNNING\n", "")]})
    assert main(argv, repo=repo, executor=executor, run=done) == 0
    _json(capsys)
    assert main(argv, repo=repo, executor=executor, run=running) == 2
    assert "writes training/r/merged/checkpoint-100 (1 running): its run has not ended" in capsys.readouterr().err


def test_the_pull_verb_says_what_it_took_held_and_merged(repo, tmp_path, capsys, monkeypatch):
    _, executor = _executors()
    assert main(["job", "--backend", "slurm", "--output", "sampling/c", "--json", "--", "true"], repo=repo,
                executor=executor) == 0
    launch_id = _json(capsys)["launch_id"]
    directory = tmp_path / "artifacts" / ".ruleforge-launch" / launch_id
    record = LaunchRecord.load(directory)
    record.data["merges"] = [{"campaign_root": "sampling/c", "condition_id": "m/f/p"}]
    record.save()
    unread = Completed({"sacct": [(1, "", "sacct: error: slurmdbd down")], "squeue": [(1, "", "")]})
    assert main(["status", launch_id], repo=repo, run=unread) == 0
    assert "job  unknown       slurm_job=100" in capsys.readouterr().out
    failed = Completed({"sacct": [(0, "100|FAILED\n", "")]})
    assert main(["pull", launch_id], repo=repo, run=failed) == 1
    assert capsys.readouterr().out.splitlines() == [
        "== pulled 0 path(s)",
        "== held: its run did not succeed and no later launch's pull took it (its relaunch's pull does): sampling/c",
        "== merge of m/f/p into sampling/c skipped: 1 failed of its 1 shard run(s) here, and no other launch's pull "
        "brought their shard roots back: the pull of the relaunch that finishes them merges it"]
    assert main(["pull", launch_id], repo=repo, run=Completed({"sacct": [(0, "100|RUNNING\n", "")]})) == 2
    assert f"{launch_id}: job not ended (1 running): wait first" in capsys.readouterr().err
    merged = []
    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards",
                        lambda root, condition, num_shards=None: merged.append(condition))
    assert main(["pull", str(directory)], repo=repo, run=Completed({"sacct": [(0, "100|COMPLETED\n", "")]})) == 0
    assert "merged 1" in capsys.readouterr().out and merged == ["m/f/p"]


def _local_record(artifacts: Path, launch_id: str, *, merges=()) -> LaunchRecord:
    spec = cli.build_job({"backend": "local", "output": ["sampling/c"]}, ["true"])[1]
    record = LaunchRecord.create(artifacts / ".ruleforge-launch" / launch_id, {
        "launch_id": launch_id, "backend": "local", "verb": "job", "code": {"commit": "a" * 40},
        "merges": list(merges), "options": {"allow_dirty": False},
        "jobs": [{**spec.to_json(), "request": LocalBackend(REPO).render([spec])[0]}]})
    record.add_run({"job_id": "job", "exit_code": 0})
    record.finish()
    return record


def _tree(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() if path.is_file() else b""
            for path in sorted(directory.rglob("*"))}


def test_a_pull_into_a_pinned_campaign_root_is_refused_before_anything_is_written(repo, tmp_path, capsys):
    from ruleforge.evaluate.api import pinned_roots

    pinned = sorted(pinned_roots())[0]
    relative = pinned.removeprefix("artifact://")
    artifacts = tmp_path / "artifacts"
    root = artifacts / relative
    (root / "m" / "x" / "draw-001").mkdir(parents=True)
    (root / "sampling_manifest.json").write_text("{}")
    (root / "completion.json").write_text('{"state": "complete"}\n')
    _local_record(artifacts, "P1", merges=[{"campaign_root": relative, "condition_id": "m/x"}])
    before = _tree(artifacts)
    for argv in (["pull", "P1"], ["wait", "P1", "--pull", "--timeout-min", "1"]):
        assert main(argv, repo=repo) == 2
        err = capsys.readouterr().err
        assert err.startswith(f"python -m ruleforge.launch.cli {argv[0]}: {root} ({pinned}) is {pinned}, or on its path"), err
        assert "a published run of" in err and "Traceback" not in err
    env = {key: value for key, value in os.environ.items() if key != "RULEFORGE_ARTIFACT_ROOT"}
    result = subprocess.run([sys.executable, "-m", "ruleforge.launch.cli", "pull", "P1"], capture_output=True,
                            text=True, check=False, env={**env, "RULEFORGE_ARTIFACT_ROOT": str(artifacts)}, cwd=REPO)
    refusals = [line for line in result.stderr.splitlines() if line.startswith("python -m ruleforge.launch.cli pull: ")]  # past warnings
    assert result.returncode == 2 and refusals[0].startswith(f"python -m ruleforge.launch.cli pull: {root} ({pinned})"), result.stderr
    assert "Traceback" not in result.stderr
    assert _tree(artifacts) == before
    assert main(["status", "P1"], repo=repo) == 0 and "1 succeeded" in capsys.readouterr().out


def test_a_verb_that_writes_a_launch_s_record_refuses_one_reached_through_a_symlink(repo, tmp_path, capsys):
    record = _local_record(tmp_path / "artifacts", "S1")
    link = tmp_path / "linked"
    link.symlink_to(record.directory)
    assert main(["status", str(link)], repo=repo) == 0
    capsys.readouterr()
    for argv in (["pull", str(link)], ["stop", str(link)], ["stop", str(link), "--after-save", "100"],
                 ["wait", str(link), "--pull"]):
        assert main(argv, repo=repo) == 2
        assert f"the launch S1's record directory {link} is a symlink" in capsys.readouterr().err
    assert not (record.directory / "stop.json").exists() and "pulled" not in record.refresh().data


@pytest.mark.parametrize("planted, refusal", [
    ("records", r"the launch records' directory .*\.ruleforge-launch is a symlink"),
    ("lock", r"the launch records' lock .*\.ruleforge-launch\.lock is a symlink"),
    ("root", r"the launch records' lock .* is inside the campaign root"),
])
def test_a_launch_whose_record_would_land_in_a_campaign_root_is_refused_before_anything_is_written(
        repo, tmp_path, monkeypatch, capsys, planted, refusal):
    published = tmp_path / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    artifacts = tmp_path / "artifacts"
    if planted == "records":
        (artifacts / ".ruleforge-launch").symlink_to(published, target_is_directory=True)
    elif planted == "lock":
        (artifacts / ".ruleforge-launch.lock").symlink_to(published / "x.lock")
    else:
        (published / "artifacts").mkdir()
        monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(published / "artifacts"))
    before = sorted(published.rglob("*"))
    log, executor = _executors()
    argv = ["job", "--backend", "slurm", "--output", "sampling/c", "--", "true"]
    assert main(argv, repo=repo, executor=executor) == 2
    assert re.search(refusal, capsys.readouterr().err) and not log["jobs"]
    assert sorted(published.rglob("*")) == before
    assert main([*argv[:-2], "--dry-run", "--", "true"], repo=repo, executor=executor) == 0


def test_the_pull_lines_name_every_outcome():
    entry = {"campaign_root": "sampling/c", "condition_id": "m/x"}
    assert cli._pull_lines(None) == []
    assert cli._pull_lines({"pulled": ["a"], "merged": [entry], "pull_errors": {"b": "unreadable"}, "held": ["c"],
                            "merged_elsewhere": [{**entry, "launch_id": "L2"}],
                            "merge_skipped": [{**entry, "reason": "why"}],
                            "merge_errors": [{**entry, "error": "ShardError: no"}]}) == [
        "== pulled 1 path(s); merged 1", "== not pulled: b: unreadable",
        "== held: its run did not succeed and no later launch's pull took it (its relaunch's pull does): c",
        "== merge of m/x into sampling/c: merged already by L2", "== merge of m/x into sampling/c skipped: why",
        "== merge of m/x into sampling/c refused: ShardError: no"]


def test_a_launch_that_fails_after_its_record_prints_its_runs_in_its_json(repo, tmp_path, capsys):
    _, executor = _executors(fail_at=1)
    assert main(["job", "--backend", "slurm", "--json", "--", "true"], repo=repo, executor=executor) == 2
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["runs"] == [{"job_id": "job", "slurm_job": None}]
    assert printed["error"] == "RuntimeError: sbatch: error: Batch job submission failed"
    assert "python -m ruleforge.launch.cli job: submitting failed: RuntimeError" in captured.err


def test_a_wait_that_fails_after_the_submission_still_prints_the_launch(repo, capsys, monkeypatch):
    def unreadable(*args, **kwargs):
        raise OSError("sacct went away")

    monkeypatch.setattr(cli, "wait", unreadable)
    assert main(["job", "--backend", "local", "--json", "--", "true"], repo=repo) == 2
    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["runs"][0]["exit_code"] == 0 and printed["error"] == "OSError: sacct went away"
    assert "python -m ruleforge.launch.cli job: OSError: sacct went away" in captured.err


def test_a_run_an_old_launch_refused_never_succeeds(repo, tmp_path, capsys):
    spec = cli.build_job({"backend": "local"}, ["true"])[1]
    record = LaunchRecord.create(tmp_path / "artifacts" / ".ruleforge-launch" / "L1", {
        "launch_id": "L1", "backend": "local", "options": {}, "code": {},
        "jobs": [{**spec.to_json(), "request": LocalBackend(repo).render([spec])[0]}]})
    record.add_run({"job_id": "job", "exit_code": 0, "refused": "the checkout moved"})
    record.finish()
    assert main(["status", "L1"], repo=repo) == 0
    assert "job  succeeded     exit_code=0  (refused: the checkout moved)" in capsys.readouterr().out
    assert main(["wait", "L1", "--json"], repo=repo) == 1
    assert _json(capsys)["states"]["job"]["refused"] == "the checkout moved"


PROFILE = """partition: p
account: acct
qos: {gpu: g, cpu: c}
gpus_per_node: 4
exclude: [n1, n2]
defaults: {timeout_min: 30, cpus: 2, mem_gb: 8, mem_gb_per_gpu: 16, signal_delay_s: 90}
setup: ["export A=1", "export MAX_JOBS=${oc.env:MAX_JOBS,8}"]
"""


@pytest.fixture
def profile(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "clusters"
    directory.mkdir()
    (directory / "site.yaml").write_text(PROFILE)
    monkeypatch.setenv("RULEFORGE_CLUSTER_DIR", str(directory))
    monkeypatch.delenv("MAX_JOBS", raising=False)
    return directory / "site.yaml"


def _submitit(capsys) -> dict:
    printed = _json(capsys)
    return {"printed": printed, **printed["jobs"][0]["request"]["submitit"]}


def test_a_profile_is_the_cluster_and_its_qos_the_default_after_the_command_line_and_the_spec(repo, profile, capsys,
                                                                                            monkeypatch):
    monkeypatch.delenv("RULEFORGE_SLURM_PARTITION")
    gpu = ["job", "--backend", "slurm", "--gpu-count", "1", "--profile", "site", "--dry-run", "--json", "--", "true"]
    assert main(gpu, repo=repo) == 0
    request = _submitit(capsys)
    assert request["slurm_partition"] == "p" and request["slurm_setup"] == ["export A=1", "export MAX_JOBS=8"]
    assert request["slurm_signal_delay_s"] == 90 and request["cpus_per_task"] == 2 and request["mem_gb"] == 16
    assert request["timeout_min"] == 30
    assert request["slurm_additional_parameters"] == {"account": "acct", "qos": "g", "no_requeue": True,
                                                      "exclude": "n1,n2"}
    assert request["printed"]["profile"] == str(profile)
    assert main([*gpu[:-2], "--qos", "jit", *gpu[-2:]], repo=repo) == 0
    assert _submitit(capsys)["slurm_additional_parameters"]["qos"] == "jit"
    _spec(repo, "named", "backend: slurm\ngpu-count: 1\nqos: compact\nargs: ['true']\n")
    assert main(["job", "--spec", "named", "--profile", "site", "--dry-run", "--json"], repo=repo) == 0
    assert _submitit(capsys)["slurm_additional_parameters"]["qos"] == "compact"
    assert main(["job", "--backend", "slurm", "--profile", str(profile), "--dry-run", "--json", "--", "true"],
                repo=repo) == 0
    assert _submitit(capsys)["slurm_additional_parameters"]["qos"] == "c"
    (profile.parent / "bare.yaml").write_text(PROFILE.replace("qos: {gpu: g, cpu: c}\n", ""))
    assert main(["job", "--backend", "slurm", "--profile", "bare", "--dry-run", "--json", "--", "true"],
                repo=repo) == 0
    request = _submitit(capsys)
    assert "qos" not in request["slurm_additional_parameters"] and (request["cpus_per_task"], request["mem_gb"]) == (2, 8)
    assert main(["job", "--backend", "slurm", "--profile", "bare", "--cpus", "3", "--mem-gb", "5", "--timeout-min",
                 "7", "--dry-run", "--json", "--", "true"], repo=repo) == 0
    request = _submitit(capsys)
    assert (request["cpus_per_task"], request["mem_gb"], request["timeout_min"]) == (3, 5, 7)


def test_a_profile_applies_to_slurm_alone_and_must_be_one(repo, profile, capsys):
    assert main(["job", "--backend", "local", "--profile", "site", "--dry-run", "--", "true"], repo=repo) == 2
    assert "--profile applies to slurm only, not local" in capsys.readouterr().err
    assert main(["job", "--backend", "slurm", "--profile", "nowhere", "--dry-run", "--", "true"], repo=repo) == 2
    assert "no profile nowhere.yaml in" in capsys.readouterr().err
    assert main(["job", "--backend", "slurm", "--gpu-count", "5", "--profile", "site", "--dry-run", "--", "true"],
                repo=repo) == 2
    assert "job asks for 5 GPUs; a node of site has 4" in capsys.readouterr().err


def test_a_profile_is_read_once_into_the_settings_with_the_qos_it_gives(repo, profile):
    args = cli.build_parser().parse_args(["train", "--run-name", "r", "--profile", "site", "--gpu-count", "2"])
    args.args = []
    settings = cli.job_settings(args, args.options_parser, "L", repo)[0]
    assert settings["profile"].path == profile and settings["qos"] == "g"
    args = cli.build_parser().parse_args(["job", "--backend", "slurm"])
    args.args = ["true"]
    assert "profile" not in cli.job_settings(args, args.options_parser, "L", repo)[0]


def test_the_launcher_runs_as_a_program(repo_root):
    result = subprocess.run([sys.executable, "-m", "ruleforge.launch.cli", "status", "--help"], capture_output=True,
                            text=True, cwd=repo_root, env={**os.environ, "PYTHONPATH": str(repo_root / "src")})
    assert result.returncode == 0 and result.stdout.startswith("usage: python -m ruleforge.launch.cli status")


def test_the_verbs_parsers_name_the_backends_the_launcher_has():
    from ruleforge.launch import backends

    assert list(control.LAUNCH_BACKENDS) == sorted(backends.BACKENDS)
    assert set(control.TRAIN_BACKENDS) < set(backends.BACKENDS) and control.PREFLIGHT_WHERE == ("local", "none")
