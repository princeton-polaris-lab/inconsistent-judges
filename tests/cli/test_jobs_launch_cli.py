from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.cli import app, jobs
from ruleforge.launch import cli as launch_cli
from ruleforge.launch import control
from ruleforge.core.record_names import RECORDED_LAUNCHES_DIRNAME
from ruleforge.launch.records import LaunchRecord, records_root
from tests.conftest import requires_git_worktree
from tests.test_launch_backends import Completed, FakePopen, _executors
from tests.test_launch_guards import CONFIG_PATH_FLAGS, OUTSIDE_NAMES

pytestmark = requires_git_worktree

REPO = Path(__file__).resolve().parents[2]
SPECS = sorted(path.stem for path in (REPO / control.SPECS_DIR).glob("*.yaml"))
#: sacctmgr on a cluster where jit preempts normal and nothing preempts compact (a training launch asks it).
QOS = Completed({"sacctmgr": [(0, "normal|\njit|normal\ncompact|\n", "")]})
#: The stand-in population's one row (its digest is the test's ``POPULATION_SHA256``).
ROW = '{"prompt": "p", "response": "r"}\n'
POPULATION_SHA256 = hashlib.sha256(ROW.encode()).hexdigest()
SHA = "ab" * 32
LAUNCHES = {
    "ruleforge_modelspec": ["job", "--var", "run_name=r"],
    "ruleforge_smoke": ["job"],
    "merge_fp32": ["job", "--var", "run=r", "--var", "step=100"],
    "grpo_rows_gemma": ["job", "--var", "anchors=anchors/a", "--var", "scenarios=populations/p.jsonl", "--var",
                        "require_protected=populations/p.jsonl", "--var", "dev_nonyes=populations/p.jsonl", "--var",
                        "out=grpo_data/b"],
    "ruleset_size_open_models": ["job", "--var", "campaign_id=ruleset-size-x", "--var", "seeds=[0,2,4,6,8]"],
    "ruleset_size_batch_judge": ["job", "--var", "campaign_id=ruleset-size-x", "--var", "shares=[0],[3]"],
    "ruleforge_rulesets": ["campaign", "--var", "campaign_id=ruleforge-x"],
    "prompt_ablation_naive": ["campaign", "--var", "campaign_id=final-x"],
    "batch_judge": ["campaign", "--var", "campaign_id=b-x", "--var", "model=gpt-5.6-luna-batch", "--var",
                    "population=populations/p.jsonl", "--var", f"population_sha256={POPULATION_SHA256}"],
    "prompt_ablation": ["campaign", "--var", "campaign_id=revised-x"],
    "hierarchy": ["campaign", "--var", "campaign_id=h-x", "--var", "models=gpt-oss-tp-2", "--shards", "3"],
    "gemma_anchors": ["campaign", "--var", "stage=bank", "--var", "campaign_prefix=ga-x", "--blocks",
                      "populations/blocks"],
    "grpo_anchors": ["campaign", "--var", "judge=qwen3-8b", "--var", "weights=Qwen3-8B", "--var", "strict_think=true",
                     "--var", "stage=mixscreen", "--var", "campaign_prefix=q-x", "--var", "request_block=250",
                     "--blocks", "populations/blocks"],
    "gemma_references": ["campaign", "--var", "stage=references_bank", "--var", "campaign_prefix=gr-x", "--var",
                         "population=populations/p.jsonl", "--var", f"population_sha256={POPULATION_SHA256}"],
    "training_arm_gemma": ["campaign", "--var", "campaign_id=f5g-x", "--var",
                          "checkpoint=training/r/merged/checkpoint-200", "--var", f"checkpoint_sha256={SHA}"],
    "training_arms": ["campaign", "--var", "serving=merged_fp32", "--var", "gpus=1", "--var", "mem_gb=96", "--var",
                     "campaign_id=f5-x", "--var", "checkpoint=training/r/merged/checkpoint-100", "--var",
                     f"checkpoint_sha256={SHA}"],
    "grpo_train_run": ["train", "--run-name", "r1", "--var", "recipe=training/grpo/qwen3_8b_mix", "--var",
                       "rows=grpo_data/x", "--var", "condition=sampling/c/condition.json", "--resume", "latest"],
    "grpo_train_gemma_run": ["train", "--run-name", "r2", "--var", "rows=grpo_data/x", "--var",
                             "condition=sampling/c/condition.json", "--resume", "latest"],
}


@pytest.fixture
def site(tmp_path, monkeypatch):
    """The cluster a launch reads from the environment, the weights root, the population and the specs' stand-ins."""
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu-partition")
    monkeypatch.delenv("RULEFORGE_SLURM_ACCOUNT", raising=False)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    populations = tmp_path / "artifacts" / "populations"
    (populations / "blocks").mkdir(parents=True)
    shutil.copy(REPO / "data" / "populations" / "frozen_500_r3.jsonl", populations)
    (populations / "p.jsonl").write_text(ROW)
    (populations / "blocks" / "b-000.jsonl").write_text(ROW)
    return tmp_path / "artifacts"


@pytest.fixture
def launcher(monkeypatch):
    """Both command lines' launches against ``checkout`` with fake SLURM answers and processes."""
    real = launch_cli.run_verb
    given = {"repo": REPO, "run": QOS}

    def run_verb(args, handler, *, program, repo=None, follow=launch_cli.PROGRAM, **_):
        return real(args, handler, program=program, follow=follow, **given)

    monkeypatch.setattr(launch_cli, "run_verb", run_verb)
    return given


def _both(capsys, argv, *, slurm_executors=False, given=None):
    found = []
    for entry in ("launch.py", "ruleforge"):
        if slurm_executors:
            given["executor"] = _executors()[1]
        status = launch_cli.main(list(argv)) if entry == "launch.py" else app.main(["jobs", "launch", *argv])
        captured = capsys.readouterr()
        found.append((status, captured.out, captured.err))
    return found


def _same(printed: str, launch_id: str) -> str:
    return printed.replace(launch_id, "LAUNCH")


def _without_entry(launch: dict) -> dict:
    kept = {key: value for key, value in launch.items()
            if key not in ("command_line", "launch_id", "created_at", "finished_at")}
    return json.loads(_same(json.dumps(kept, sort_keys=True), launch["launch_id"]))


@pytest.mark.parametrize("name", SPECS)
def test_every_committed_spec_renders_alike_through_both_command_lines(name, site, launcher, capsys, monkeypatch):
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(site.parent / "scratch" / "batch"))  # a Batch judge's paid state
    argv = [*LAUNCHES[name][:1], "--spec", name, *LAUNCHES[name][1:], "--dry-run", "--json", "--allow-dirty"]
    (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, argv)
    assert status == jobs_status == 0, err + jobs_err
    printed, jobs_printed = json.loads(out), json.loads(jobs_out)
    assert printed["command_line"] == [*launch_cli.ENTRY_POINT, *argv]
    assert jobs_printed["command_line"] == ["ruleforge", "jobs", "launch", *argv]
    assert _without_entry(printed) == _without_entry(jobs_printed) and printed["jobs"]
    for job, jobs_job in zip(printed["jobs"], jobs_printed["jobs"], strict=True):
        assert _same(json.dumps(job["request"], sort_keys=True), printed["launch_id"]) == _same(
            json.dumps(jobs_job["request"], sort_keys=True), jobs_printed["launch_id"])
    assert _same(err, printed["launch_id"]) == _same(jobs_err, jobs_printed["launch_id"])
    assert not records_root().exists()


def test_the_spec_table_names_every_committed_spec():
    assert set(LAUNCHES) == set(SPECS)


def _options(parser):
    return [(action.option_strings, action.dest, action.nargs, action.const, action.default, action.type,
             action.choices, action.required, type(action).__name__) for action in parser._actions
            if action.dest != "help"]


@pytest.mark.parametrize("verb", list(control.LAUNCH_VERBS))
def test_a_launch_verb_takes_every_flag_of_launch_py_s_with_its_meaning(verb):
    [launch_verbs] = [action for action in launch_cli.build_parser()._actions if action.dest == "verb"]
    [jobs_verbs] = [action for action in app.build_parser()._actions if action.dest == "command"]
    [jobs_verb] = [action for action in jobs_verbs.choices["jobs"]._actions if action.dest == "verb"]
    [launch_kinds] = [action for action in jobs_verb.choices["launch"]._actions if action.dest == "verb"]
    ours, theirs = launch_kinds.choices[verb], launch_verbs.choices[verb]
    assert _options(ours) == _options(theirs) and launch_cli.spec_options(ours).keys() == \
        launch_cli.spec_options(theirs).keys()
    assert not ours.allow_abbrev and theirs.allow_abbrev
    assert ours.epilog == theirs.epilog.replace(launch_cli.PROGRAM, "ruleforge jobs launch")
    argv = {"job": ["--spec", "s", "--var", "a=1", "--backend", "slurm", "--gpu-count", "2", "--axis", "k=1..3",
                    "--no-resumable", "--env", "A=1", "--output", "o", "--wait", "--json"],
            "campaign": ["--campaign-id", "c", "--shards", "4", "--only", "1,2", "--pack-draws=on", "--blocks", "b",
                         "--root-template", "r/{block}", "--preflight", "none", "--no-shards"],
            "train": ["--run-name", "r", "--resume", "latest", "--expect", "a=1", "--completions-per-step", "32",
                      "--preflight-only", "--ignore-launch", "L0"]}[verb]
    parsed = app.build_parser().parse_args(["jobs", "launch", verb, *argv])
    expected = vars(launch_cli.build_parser().parse_args([verb, *argv]))
    assert {key: value for key, value in vars(parsed).items() if key not in (*jobs.COMMAND_KEYS, "handler",
                                                                              "options_parser")} == {
        key: value for key, value in expected.items() if key not in ("handler", "options_parser")}
    assert parsed.handler is jobs.run_launch and parsed.takes_overrides
    with pytest.raises(SystemExit):
        app.build_parser().parse_args(["jobs", "launch", verb, "--spe", "s"])  # no abbreviation


def test_a_launch_through_either_command_line_writes_one_record_but_for_the_command_line(site, launcher, capsys,
                                                                                         tmp_path):
    launcher["popen"] = FakePopen([(0, ["ran"]), (0, ["ran"])])
    for argv, slurm in ((["job", "--backend", "local", "--allow-dirty", "--json", "--name", "j", "--", "true"], False),
                        (["job", "--backend", "slurm", "--gpu-count", "1", "--allow-dirty", "--", "python", "x.py"],
                         True)):
        (status, out, _), (jobs_status, jobs_out, _) = _both(capsys, argv, slurm_executors=slurm, given=launcher)
        assert status == jobs_status == 0
        ids = [json.loads(out)["launch_id"], json.loads(jobs_out)["launch_id"]] if not slurm else [
            re.search(r"== launch (\S+)", text).group(1) for text in (out, jobs_out)]
        if slurm:
            assert f"follow it with python -m ruleforge.launch.cli status {ids[0]}" in out
            assert f"follow it with ruleforge jobs status {ids[1]}" in jobs_out
        record, jobs_record = (LaunchRecord.load(launch_id).data for launch_id in ids)
        assert record["command_line"] == [*launch_cli.ENTRY_POINT, *argv]
        assert jobs_record["command_line"] == ["ruleforge", "jobs", "launch", *argv]
        assert _without_entry(record) == _without_entry(jobs_record) and record["runs"]


@pytest.mark.parametrize(("argv", "reason"), [
    (["job", "--spec", "nope"], "no launch spec 'nope': a path, or a name under configs/launch/"),
    (["job", "--spec", "ruleforge_modelspec", "--dry-run"], "configs/launch/ruleforge_modelspec.yaml needs --var for "
                                                           "run_name"),
    (["job", "--var", "a=1", "--backend", "local", "--", "true"], "--var fills a spec's placeholders: give --spec"),
    (["job", "--spec", "ruleforge_smoke", "--var", "bad"], "--var 'bad': expected NAME=VALUE"),
    (["job", "--backend", "local", "--qos", "normal", "--allow-dirty", "--", "true"],
     "--qos applies to slurm only, not local"),
    (["job", "--backend", "slurm", "--pull", "--allow-dirty", "--", "true"], "--pull comes with --wait"),
    (["job", "--backend", "slurm", "--allow-dirty", "--dry-run"], "no command: put it after --"),
    (["campaign", "--spec", "prompt_ablation_naive", "--var", "campaign_id=x", "--only", "1", "--allow-dirty", "--dry-run"],
     "--only selects shards: give --shards"),
    (["campaign", "--backend", "slurm", "--preflight", "none", "--preflight-only", "--allow-dirty", "--dry-run"],
     "--preflight-only with --preflight none would check nothing"),
    (["train", "--spec", "grpo_train_run", "--run-name", "r", "--dry-run"], "needs --var for"),
    (["train", "--backend", "slurm", "--run-name", "r", "--allow-dirty", "--dry-run"],
     "a training run takes GPUs: give --gpu-count"),
])
def test_a_refusal_is_both_command_lines_refusal(site, launcher, capsys, argv, reason):
    (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, argv)
    assert status == jobs_status == 2 and out == jobs_out == ""
    assert err.startswith(f"python -m ruleforge.launch.cli {argv[0]}: ") and jobs_err.startswith(f"ruleforge jobs launch {argv[0]}: ")
    assert reason in err and err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1] and "Traceback" not in jobs_err
    assert not records_root().exists()


def test_an_option_unknown_to_a_verb_or_a_command_where_none_goes_is_refused_by_both(launcher, capsys):
    for argv in (["train", "--name", "x", "--run-name", "r"], ["status", "L1", "--", "x"]):
        with pytest.raises(SystemExit) as raised:
            launch_cli.main(argv)
        with pytest.raises(SystemExit) as jobs_raised:
            app.main(["jobs", *(["launch"] if argv[0] == "train" else []), *argv])
        assert raised.value.code == jobs_raised.value.code == 2
    capsys.readouterr()


def test_jobs_launch_needs_submitit_and_says_so(cli, monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *args: None if name == "submitit" else real(name))
    status, _, err = cli("jobs", "launch", "job", "--backend", "local", "--", "true")
    assert status == 2 and "needs submitit, which this environment lacks: add --group dev to uv sync, or pip install -e '.[slurm]'" in err
    assert not records_root().exists()


def test_launch_py_s_module_entry_names_itself(site, launcher, capsys):
    assert launch_cli.main(["job", "--backend", "local", "--dry-run", "--json", "--allow-dirty", "--", "true"]) == 0
    assert json.loads(capsys.readouterr().out)["command_line"][:4] == ["python", "-m", "ruleforge.launch.cli", "job"]
    assert launch_cli.ENTRY_POINT == ("python", "-m", "ruleforge.launch.cli") and launch_cli.PROGRAM == " ".join(
        launch_cli.ENTRY_POINT)
    source = Path(launch_cli.__file__).read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":\n    sys.exit(main())' in source


BATCH = ["campaign", "--spec", "batch_judge", *LAUNCHES["batch_judge"][1:], "--allow-dirty"]


@pytest.mark.parametrize(("planted", "refusal"), [
    ("names it", r"batch_root of \S+'s batch judge \S*/published is inside \S*/published, a campaign root this "
                 r"library did not create"),
    ("links it", r"batch_root of \S+'s batch judge \S*/scratch/batch is a symlink: a write would follow it to "
                 r"\S*/published"),
    ("links on its way", r"batch_root of \S+'s batch judge \S*/scratch/link/batch is inside \S*/scratch/link, a "
                         r"campaign root this library did not create"),
    ("pinned", r"\S*/artifacts/pub/batch \(artifact://pub/batch\) is artifact://pub, or on its path"),
    ("a link below the artifact root", r"batch_root of \S+'s batch judge \S*/artifacts/batches/b: "
                                       r"\S*/artifacts/batches, on its way, is a symlink"),
    ("in this launch's root", r"batch_root of \S+'s batch judge \S*/sampling/b-x/elsewhere lies in the campaign root "
                              r"\S*/sampling/b-x outside its batch/"),
    ("holding this launch's root", r"batch_root of \S+'s batch judge \S*/sampling is, lies in or holds "
                                   r"\S*/sampling/b-x, \S+'s campaign root or shard root in this launch"),
    ("relative", r"batch_root of \S+'s batch judge 'batches' is relative"),
    ("home", r"batch_root of \S+'s batch judge '~/batches' is relative"),
])
def test_a_batch_judge_s_batch_root_where_nothing_may_be_written_is_refused_by_both(
        planted, refusal, site, launcher, capsys, tmp_path, monkeypatch):
    """A batch judge's job writes its paid batches' state."""
    from ruleforge.evaluate import api

    published, elsewhere, scratch = tmp_path / "published", tmp_path / "elsewhere", tmp_path / "scratch"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    elsewhere.mkdir()
    scratch.mkdir()
    batch_root = {"names it": published, "links it": scratch / "batch", "links on its way": scratch / "link" / "batch",
                  "pinned": site / "pub" / "batch", "a link below the artifact root": site / "batches" / "b",
                  "in this launch's root": site / "sampling" / "b-x" / "elsewhere",
                  "holding this launch's root": site / "sampling", "relative": "batches",
                  "home": "~/batches"}[planted]
    (scratch / "batch").symlink_to(published, target_is_directory=True)
    (scratch / "link").symlink_to(published, target_is_directory=True)
    (site / "batches").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://pub": ("leniency",)})
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(batch_root))
    before = [_tree_digest(published), _tree_digest(elsewhere), _tree_digest(site)]
    launcher["popen"] = popen = FakePopen([])
    (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, BATCH)
    assert status == jobs_status == 2 and out == jobs_out == ""
    assert err.startswith("python -m ruleforge.launch.cli campaign: ") and jobs_err.startswith("ruleforge jobs launch campaign: ")
    assert re.search(refusal, err) and err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1]
    assert popen.calls == [] and not records_root().exists()
    assert [_tree_digest(published), _tree_digest(elsewhere), _tree_digest(site)] == before
    (status, _, err), (jobs_status, _, jobs_err) = _both(capsys, [*BATCH, "--dry-run"])
    assert status == jobs_status == 0, err + jobs_err


def test_a_batch_judge_s_fresh_batch_root_launches_and_its_export_is_refused(site, launcher, capsys, tmp_path,
                                                                            monkeypatch):
    for batch_root in (tmp_path / "scratch" / "batch", site / "sampling" / "b-x" / "batch"):
        monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(batch_root))
        for argv in ([*BATCH, "--dry-run", "--json"], [*BATCH, "--preflight-only"]):
            launcher["popen"] = popen = FakePopen([(0, ["preflight ok"])] * 6)
            (status, _, err), (jobs_status, _, jobs_err) = _both(capsys, argv)
            assert status == jobs_status == 0, err + jobs_err
            assert len(popen.calls) == (6 if "--preflight-only" in argv else 0)
        assert not (tmp_path / "scratch").exists() and not batch_root.exists()
    exported = [*BATCH, "--env", f"RULEFORGE_BATCH_ROOT={tmp_path / 'published'}", "--dry-run"]
    (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, exported)
    assert status == jobs_status == 2 and out == jobs_out == ""
    assert "--env RULEFORGE_BATCH_ROOT: a campaign job's config reads it" in err
    assert err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1]


@pytest.mark.parametrize("override", [None, "judge_model.batch_root=${hydra:runtime.cwd}/batch"])
def test_a_batch_root_that_resolves_only_in_the_job_is_refused_by_both(override, site, launcher, capsys,
                                                                        monkeypatch):
    from hydra.core.hydra_config import HydraConfig

    monkeypatch.setattr(HydraConfig.instance(), "cfg", None)  # no Hydra run in this process: as the launch's
    if override is None:
        monkeypatch.delenv("RULEFORGE_BATCH_ROOT", raising=False)
    else:
        monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(site / "unused"))
    launcher["popen"] = popen = FakePopen([])
    for dry in ((), ("--dry-run",)):
        argv = [*BATCH, *dry, *(("--", override) if override else ())]
        (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, argv)
        assert status == jobs_status == 2 and out == jobs_out == ""
        assert "the batch judge's batch_root does not resolve where the launch runs" in err, err
        assert (("set RULEFORGE_BATCH_ROOT (the variable it reads) where the launch runs" in err) is (override is None)
                and ("would resolve only in the job" in err) is (override is not None)), err
        assert err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1]
    assert popen.calls == [] and not records_root().exists()


@pytest.mark.parametrize("override, exported, refusal", [
    ('judge_model.batch_root=${oc.env:"JOB_BATCH_ROOT"}', "JOB_BATCH_ROOT", "--env JOB_BATCH_ROOT: a campaign job's "
     "config reads it"),
    ("judge_model.batch_root=\"${oc.select:judge_model.nothing,${oc.env: 'JOB_BATCH_ROOT' }}\"", "JOB_BATCH_ROOT",
     "--env JOB_BATCH_ROOT: a campaign job's config reads it"),
    ("judge_model.batch_root='${oc.env:${judge_model.model_type}}'", "OTHER_NAME",
     "--env OTHER_NAME: a campaign job's config reads a variable whose name it computes"),
    ("judge_model.batch_root='${${judge_model.model_type}:JOB_BATCH_ROOT}'", "JOB_BATCH_ROOT",
     "--env JOB_BATCH_ROOT: a campaign job's config reads a variable whose name it computes"),  # a resolver's name
])
def test_an_export_of_a_variable_the_job_reads_however_it_is_spelled_is_refused_by_both(
        override, exported, refusal, site, launcher, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("JOB_BATCH_ROOT", str(tmp_path / "scratch" / "batch"))
    published = tmp_path / "published"
    launcher["popen"] = popen = FakePopen([])
    for dry in ((), ("--dry-run",)):
        argv = [*BATCH, "--env", f"{exported}={published}", *dry, "--", override]
        (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, argv)
        assert status == jobs_status == 2 and out == jobs_out == ""
        assert refusal in err and err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1]
    assert popen.calls == [] and not records_root().exists()


def test_an_export_that_chooses_the_job_s_judge_config_is_refused_by_both(site, launcher, capsys, monkeypatch):
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(site.parent / "scratch" / "batch"))
    monkeypatch.setenv("REVIEW_MODEL", "gpt-5.6-luna-batch")
    chosen = [item if item != "model=gpt-5.6-luna-batch" else "model=${oc.env:REVIEW_MODEL}" for item in BATCH]
    assert chosen != BATCH
    before = _tree_digest(site)
    launcher["popen"] = popen = FakePopen([])
    for dry in ((), ("--dry-run",)):
        argv = [*chosen, "--env", "REVIEW_MODEL=claude-haiku-5-5-batch", *dry]
        (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, argv)
        assert status == jobs_status == 2 and out == jobs_out == ""
        assert ("--env REVIEW_MODEL: a campaign job's config chooses its files by the environment" in err
                and "compose model/claude-haiku-5-5-batch at judge_model, model/claude-haiku-5-5-batch at "
                    "agent_model where the launch composed, and checked the places of, model/gpt-5.6-luna-batch at "
                    "judge_model, model/gpt-5.6-luna-batch at agent_model;" in err), err
        assert err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1]
    assert popen.calls == [] and not records_root().exists() and _tree_digest(site) == before
    (status, _, err), (jobs_status, _, jobs_err) = _both(
        capsys, [*chosen, "--env", "REVIEW_MODEL=gpt-5.6-luna-batch", "--dry-run"])
    assert status == jobs_status == 0, err + jobs_err


@pytest.mark.parametrize("verb", [["campaign", "--backend", "local", "--campaign-id", "c", "--new-root"],
                                  ["train", "--backend", "slurm", "--gpu-count", "1", "--run-name", "r"]])
@pytest.mark.parametrize("hydra", [*OUTSIDE_NAMES, *CONFIG_PATH_FLAGS])
def test_a_config_from_outside_the_configs_is_refused_by_both(hydra, verb, site, launcher, capsys, tmp_path):
    escape = tmp_path / "abs" / "dir"
    escape.mkdir(parents=True)
    (escape / "escape.yaml").write_text(f"# @package _global_\nhydra:\n  job:\n    name: ../../{tmp_path}/abs/x\n")
    hydra = [item.replace("//abs/dir/escape", f"/{escape}/escape").replace("//abs/x", f"/{escape}/escape")
             for item in hydra]
    before = _tree_digest(tmp_path / "abs")
    launcher["popen"] = popen = FakePopen([])
    argv = [*verb, "--allow-dirty", "--", *hydra]
    (status, out, err), (jobs_status, jobs_out, jobs_err) = _both(capsys, argv, slurm_executors=True, given=launcher)
    assert status == jobs_status == 2 and out == jobs_out == ""
    assert ("Hydra reads a name led by '/' as a path" in err or "a job's config is one of configs/, named by "
            "--config-name" in err), err
    assert err.split(": ", 1)[1] == jobs_err.split(": ", 1)[1]
    assert popen.calls == [] and not records_root().exists() and _tree_digest(tmp_path / "abs") == before


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        digest.update(str(path.relative_to(directory)).encode())
        if path.is_file() and not path.is_symlink():
            digest.update(path.read_bytes())
    return digest.hexdigest()


RERUN_ROOT = os.environ.get("RULEFORGE_RERUN_ROOT")
FIRST_RERUN_COMMIT = "4bb93d742677696ba9bd46a7aacd0de7e1f3d529"


@pytest.mark.cluster
@pytest.mark.skipif(not RERUN_ROOT, reason="the rerun's root is the cluster's: set RULEFORGE_RERUN_ROOT")
def test_every_real_rerun_record_still_reads_with_jobs_status(cli):
    """Every launch record of the rerun's root reads with ``ruleforge jobs status``, which writes nothing there."""
    records = Path(RERUN_ROOT) / RECORDED_LAUNCHES_DIRNAME
    before = _tree_digest(records)
    directories = sorted(path.parent for path in records.glob("*/launch.json"))
    launched = {directory: json.loads((directory / "launch.json").read_text()) for directory in directories}
    first_rerun = [directory for directory, record in launched.items() if record["code"]["commit"] == FIRST_RERUN_COMMIT]
    assert first_rerun and all("command_line" not in launched[directory] for directory in first_rerun)
    for directory in directories:
        status, out, err = cli("jobs", "status", str(directory), "--json")
        assert status == 0, err
        assert out["launch_id"] == directory.name and out["states"]
    assert _tree_digest(records) == before
