from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from ruleforge.cli import slurm
from ruleforge.cli.common import CliError
from ruleforge.core.optional import MissingDependency
from ruleforge.evaluate import CampaignError
from ruleforge.env import package_path
from ruleforge.launch.environment import LaunchError
from tests.test_launch_backends import _executors

PROFILE = """partition: p
account: acct
qos: {gpu: g, cpu: c}
gpus_per_node: 4
defaults: {timeout_min: 30, cpus: 2, mem_gb: 8, mem_gb_per_gpu: 16, signal_delay_s: null}
setup: ["export A=1"]
"""


@pytest.fixture
def profiles(tmp_path):
    directory = tmp_path / "clusters"
    directory.mkdir()
    (directory / "site.yaml").write_text(PROFILE)
    return directory


def test_a_profile_is_found_by_file_flag_variable_then_the_package(profiles, monkeypatch, tmp_path):
    assert slurm.find_profile(str(profiles / "site.yaml")).partition == "p"
    assert slurm.find_profile("site", profiles).account == "acct"
    monkeypatch.setenv("RULEFORGE_CLUSTER_DIR", str(profiles))
    assert slurm.find_profile("site").qos_cpu == "c"
    assert slurm.find_profile("example").path == package_path("configs/cluster/example.yaml")
    dirs = slurm.profile_dirs(tmp_path)
    assert dirs[0] == tmp_path and dirs[1] == profiles and dirs[-1] == package_path("configs/cluster")
    with pytest.raises(LaunchError, match="no profile nowhere.yaml"):
        slurm.find_profile("nowhere")
    with pytest.raises(LaunchError, match="no such profile file"):
        slurm.find_profile(str(tmp_path / "absent.yaml"))


def test_a_checkout_s_profiles_are_looked_in_and_an_installed_one_s_are_not(monkeypatch):
    from ruleforge.core import paths
    from ruleforge.core.errors import ArtifactRootError

    assert Path("configs/cluster") in [Path(*path.parts[-2:]) for path in slurm.profile_dirs(None)]

    def outside():
        raise ArtifactRootError("installed")

    monkeypatch.setattr(paths, "repository_root", outside)
    assert slurm.profile_dirs(None) == [package_path("configs/cluster")]


@pytest.mark.parametrize("text", ["partition: p\n", "- a\n", "partition: [\n", "gpus_per_node: x\n"])
def test_a_file_that_is_no_profile_is_refused_naming_the_example(tmp_path, text):
    (tmp_path / "bad.yaml").write_text(text)
    with pytest.raises(LaunchError, match="(?s)not a cluster profile.*example.yaml"):
        slurm.find_profile(str(tmp_path / "bad.yaml"))


def test_a_job_is_sized_from_the_request_over_the_profile(profiles):
    profile = slurm.find_profile("site", profiles)
    gpu = slurm.job("g", ["evaluate"], profile, slurm.Request(gpus=2, resumable=True))
    assert (gpu.resources.kind, gpu.resources.gpus, gpu.resources.mem_gb, gpu.resources.qos) == ("gpu", 2, 32, "g")
    assert gpu.argv == ("ruleforge", "evaluate") and gpu.resources.resumable
    cpu = slurm.job("c", ["analyze", "x"], profile, slurm.Request(timeout_min=5, cpus=1, mem_gb=3), depends_on=["g"])
    assert (cpu.resources.kind, cpu.resources.cpus, cpu.resources.mem_gb, cpu.resources.timeout_min) == (
        "cpu", 1, 3, 5)
    assert cpu.resources.qos == "c" and cpu.depends_on == ("g",)
    with pytest.raises(CliError, match="asks for 5 GPUs; a node of site has 4"):
        slurm.job("x", ["evaluate"], profile, slurm.Request(gpus=5))


def test_the_profile_backend_is_the_profile_s_cluster(profiles, tmp_path):
    profile = slurm.find_profile("site", profiles)
    backend = slurm.backend(profile, tmp_path)
    spec = slurm.job("c", ["analyze", "x"], profile, slurm.Request())
    assert backend.prepare(spec) is spec
    [request] = backend.render([spec])
    assert request["submitit"]["slurm_partition"] == "p" and request["submitit"]["slurm_setup"] == ["export A=1"]
    assert request["submitit"]["slurm_additional_parameters"] == {"account": "acct", "qos": "c", "no_requeue": True}
    assert "slurm_signal_delay_s" not in request["submitit"] and request["cwd"] == str(tmp_path)


def test_a_profile_s_excluded_nodes_are_every_job_s_exclude(profiles, tmp_path):
    (profiles / "fenced.yaml").write_text(PROFILE + "exclude: [n1, n2]\n")
    profile = slurm.find_profile("fenced", profiles)
    [request] = slurm.backend(profile, tmp_path).render([slurm.job("g", ["evaluate"], profile, slurm.Request(gpus=1))])
    assert request["submitit"]["slurm_additional_parameters"]["exclude"] == "n1,n2"


def test_a_launch_is_recorded_under_the_artifact_root_and_submitted(profiles, tmp_path, monkeypatch):
    profile = slurm.find_profile("site", profiles)
    log, factory = _executors()
    specs = [slurm.job("a", ["evaluate"], profile, slurm.Request()),
             slurm.job("b", ["analyze", "r"], profile, slurm.Request(), depends_on=["a"])]
    assert slurm.submit(specs, profile, verb="evaluate", dry_run=False, executor=factory) == 0
    [record] = (tmp_path / "artifacts" / ".ruleforge-launch").iterdir()
    assert (record / "launch.json").is_file() and len(log["jobs"]) == 2
    assert log["jobs"][1][1]["slurm_additional_parameters"]["dependency"] == "afterok:100"


def test_a_launch_needs_an_artifact_root_unless_it_is_a_dry_run(profiles, monkeypatch, capsys):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    profile = slurm.find_profile("site", profiles)
    specs = [slurm.job("a", ["evaluate"], profile, slurm.Request())]
    with pytest.raises(CliError, match="set RULEFORGE_ARTIFACT_ROOT, or pass --dry-run"):
        slurm.submit(specs, profile, verb="evaluate", dry_run=False)
    assert slurm.submit(specs, profile, verb="evaluate", dry_run=True) == 0
    assert "== dry run: nothing was submitted" in capsys.readouterr().out
    with pytest.raises(CliError, match="does not submit"):
        slurm.submit([slurm.job("a", ["x"], profile, slurm.Request(), depends_on=["z"])], profile, verb="x",
                     dry_run=True)


@pytest.mark.parametrize("argv, expected", [
    (["evaluate", "--slurm", "p", "--rules", "r"], ["evaluate", "--rules", "r"]),
    (["evaluate", "--slurm=p", "--dry-run", "--gpus", "2"], ["evaluate"]),
    (["evaluate", "--", "--slurm", "kept"], ["evaluate", "--", "--slurm", "kept"]),
])
def test_a_job_runs_the_command_without_what_only_the_submission_reads(argv, expected):
    assert slurm.passthrough_argv(argv, drop={"--slurm": 1, "--dry-run": 0, "--gpus": 1}) == expected


def test_the_options_a_job_drops_are_the_command_line_s_slurm_options_with_their_arity():
    from ruleforge.cli import app

    parser = argparse.ArgumentParser()
    app._slurm_options(parser)
    [group] = [group for group in parser._action_groups if group.title.startswith("SLURM")]
    assert all(len(action.option_strings) == 1 for action in group._group_actions)
    assert {action.option_strings[0]: 0 if isinstance(action, argparse._StoreTrueAction) else 1
            for action in group._group_actions} == slurm.SUBMISSION_OPTIONS


def test_a_job_s_request_is_its_gpus_and_the_command_s_slurm_options():
    args = argparse.Namespace(timeout_min=5, cpus=1, mem_gb=3)
    assert slurm.request(args, 2) == slurm.Request(gpus=2, timeout_min=5, cpus=1, mem_gb=3)
    assert slurm.request(args, 0, resumable=True) == slurm.Request(timeout_min=5, cpus=1, mem_gb=3, resumable=True)


def test_a_job_s_kind_is_its_command_past_a_leading_option(profiles):
    profile = slurm.find_profile("site", profiles)
    assert slurm.job("e", ["-v", "evaluate", "--rules", "r"], profile, slurm.Request()).kind == "evaluate"


def test_slurm_without_submitit_is_refused_naming_the_extra(profiles, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "submitit" else real(name))
    profile = slurm.find_profile("site", profiles)
    with pytest.raises(MissingDependency, match=r"^--slurm needs submitit, .*: add --group dev to uv sync, or pip install -e '\.\[slurm\]'"):
        slurm.submit([slurm.job("a", ["evaluate"], profile, slurm.Request())], profile, verb="evaluate", dry_run=True)


@pytest.mark.parametrize("planted, refusal", [
    ("records", r"the launch records' directory .*\.ruleforge-launch is a symlink"),
    ("lock", r"the launch records' lock .*\.ruleforge-launch\.lock is a symlink"),
    ("root", r"the launch records' lock .* is inside the campaign root"),
])
def test_a_launch_whose_record_would_land_in_a_campaign_root_is_refused_before_anything_is_written(
        profiles, tmp_path, monkeypatch, planted, refusal):
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
    profile = slurm.find_profile("site", profiles)
    log, factory = _executors()
    with pytest.raises(CampaignError, match=refusal):
        slurm.submit([slurm.job("a", ["evaluate"], profile, slurm.Request())], profile, verb="evaluate",
                     dry_run=False, executor=factory)
    assert not log["jobs"] and sorted(published.rglob("*")) == before


def test_the_launch_records_places_are_the_launcher_s_spelled_once(tmp_path, monkeypatch):
    from ruleforge.launch.records import records_places

    places = records_places()
    artifacts = tmp_path / "artifacts"
    assert [(place.path, place.name, place.what) for place in places] == [
        (artifacts / ".ruleforge-launch.lock", "records lock", "the launch records' lock"),
        (artifacts / ".ruleforge-launch", "records directory", "the launch records' directory")]
    found = slurm.launch_places()
    assert [(place.what, place.path, place.own) for place in found] == [(place.name, place.path, False)
                                                                        for place in places]
    (artifacts / ".ruleforge-launch").symlink_to(tmp_path, target_is_directory=True)
    found[0].protect()
    with pytest.raises(CampaignError, match=r"the launch records' directory \S*/\.ruleforge-launch is a symlink"):
        found[1].protect()
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    with pytest.raises(CliError, match="--slurm records its launch under the artifact root"):
        slurm.launch_places()


@pytest.mark.parametrize("command", [
    ["evaluate", "--judge", "fake", "--rules", "r.jsonl", "--scenarios", "s.jsonl"],
    ["optimize", "--rules", "r.jsonl"],
    ["grpo", "train", "--model", "m"],
    ["grpo", "rows"],
])
@pytest.mark.parametrize("given", [["--gpus", "2", "--json"], ["--cluster-dir", "c"], ["--dry-run", "--cpus", "4"]])
def test_a_slurm_option_without_slurm_is_refused_before_anything_runs(cli, command, given, tmp_path):
    status, out, err = cli(*command, *given)
    assert status == 2 and out == ""
    flags = " and ".join(flag for flag in given if flag.startswith("--") and flag != "--dry-run")
    assert f"{flags} {'is' if ' and ' not in flags else 'are'} for --slurm PROFILE" in err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artifacts", "work"]
    assert list((tmp_path / "work").iterdir()) == list((tmp_path / "artifacts").iterdir()) == []


@pytest.mark.parametrize("command", [
    ["evaluate", "--judge", "fake", "--rules", "{rules}", "--scenarios", "{scenarios}", "--draws", "2", "--out", "runs/e"],
    ["optimize", "--judge", "fake", "--rules", "{rules}", "--seeds-good", "{scenarios}", "--seeds-bad", "{scenarios}",
     "--out", "runs/o"],
    ["grpo", "train", "--model", "m", "--run-name", "r"],
    ["grpo", "rows", "--anchors", "a", "--scenarios", "{scenarios}", "--rules", "{rules}", "--model", "m", "--out",
     "runs/rows"],
])
def test_a_dry_run_here_prints_what_would_run_and_writes_nothing(cli, inputs, command, tmp_path):
    scenarios, rules = inputs
    status, out, err = cli(*[item.format(rules=rules, scenarios=scenarios) for item in command], "--dry-run")
    assert status == 0, err
    assert out.startswith("== dry run: nothing was run or written; the run would write ")
    assert "model_name: " in out or "model:" in out
    assert sorted(path.name for path in tmp_path.iterdir()) == ["artifacts", "inputs", "work"]
    assert list((tmp_path / "work").iterdir()) == list((tmp_path / "artifacts").iterdir()) == []
