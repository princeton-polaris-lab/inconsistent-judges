from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ruleforge.core.record_names import RECORDED_BLOCK_KEYS
from ruleforge.launch import campaign
from ruleforge.launch.campaign import (
    PREFLIGHT_RUN_DIR,
    SHARD_PREFLIGHT_RUN_DIR,
    CampaignOptions,
    Place,
    place,
    resolve_condition,
    selected_shards,
)
from ruleforge.launch.cli import main
from ruleforge.launch.guards import CAMPAIGN_MARKER, hydra_arguments
from ruleforge.launch.environment import LaunchError
from ruleforge.launch.records import LaunchRecord, claimed_launches, lock_path, records_root, writers_dir
from ruleforge.evaluate.sharding import ShardError, shard_root
from tests.conftest import requires_git_worktree
from tests.support.git import _git, make_repo
from tests.support.launch_records import launch_site, model_config_now, model_words_now, multirun_site
from tests.test_launch_backends import Completed, FakeExecutor, FakePopen, _executors

REPO = Path(__file__).resolve().parents[1]
STYLES = ("naive-rules-above", "encouraged")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A clean checkout holding the repository's configs (so conditions compose as they do here)."""
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", "gpu-partition")
    monkeypatch.delenv("RULEFORGE_SLURM_ACCOUNT", raising=False)
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    checkout = make_repo(tmp_path)
    shutil.copytree(REPO / "configs", checkout / "configs", dirs_exist_ok=True)
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-q", "-m", "configs")
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    (tmp_path / "artifacts").mkdir()
    return checkout


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def dry(repo: Path, capsys, *argv: str, **kwargs) -> dict:
    """The campaign's dry run, as JSON (its options before the ``--`` of ``argv``'s Hydra arguments)."""
    split = argv.index("--") if "--" in argv else len(argv)
    command = ["campaign", *argv[:split], "--dry-run", "--json", *argv[split:]]
    assert main(command, repo=repo, **kwargs) == 0, capsys.readouterr().err
    return _json(capsys)


def given(job: dict) -> list[str]:
    """A campaign job's Hydra arguments (its command past its program, ``campaign.PROGRAM``)."""
    assert job["argv"][:3] == ["python", "-m", "ruleforge.launch.programs.run_campaign"] == list(campaign.PROGRAM)
    return job["argv"][3:]


def population(tmp_path: Path, relative: str = "pops/p.jsonl", text: str = '{"prompt": "p", "response": "r"}\n'
               ) -> tuple[Path, str]:
    path = tmp_path / "artifacts" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


SLURM = ("--backend", "slurm", "--gpu-count", "1")


def test_a_campaign_id_names_its_root_and_nothing_else_is_passed(repo, tmp_path, capsys):
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--", "sampling.num_decoder_draws=3")
    [job] = printed["jobs"]
    root = tmp_path / "artifacts" / "sampling" / "c1"
    assert given(job) == ["sampling.num_decoder_draws=3", "campaign_id=c1", f"campaign_root={root}"]
    assert job["job_id"] == "campaign" and job["workdir"] == "." and job["outputs"] == []
    assert printed["campaign"]["conditions"] == [{"job_id": "campaign", "campaign_id": "c1",
                                                  "campaign_root": "sampling/c1",
                                                  "condition_id": "gemma-4-31B-it/naive-rules-above", "shards": None,
                                                  "selection": []}]
    assert printed["campaign"]["preflights"] == [["python", "-m", "ruleforge.launch.programs.preflight_campaign",
                                                  *given(job), PREFLIGHT_RUN_DIR]]
    assert printed["merges"] == [] and printed["campaign"]["preflight"] == "local"
    assert main(["campaign", *SLURM, "--dry-run"], repo=repo) == 2
    assert re.search(r"does not compose: MissingMandatoryValue.*campaign_id", capsys.readouterr().err)


def test_the_campaign_marker_is_the_manifest_a_campaign_root_holds():
    from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME

    assert CAMPAIGN_MARKER == SAMPLING_MANIFEST_FNAME  # the launcher keeps its own copy of the name


@pytest.mark.parametrize("spelled", ["roots/rebased", "{artifacts}/roots/rebased", "{elsewhere}/rebased"])
def test_a_root_named_must_hold_a_campaign_unless_a_new_one_is_meant(repo, tmp_path, capsys, spelled):
    root = Path(spelled.format(artifacts=tmp_path / "artifacts", elsewhere=tmp_path / "elsewhere"))
    if not root.is_absolute():
        root = tmp_path / "artifacts" / root
    text = spelled.format(artifacts=tmp_path / "artifacts", elsewhere=tmp_path / "elsewhere")
    argv = [*SLURM, "--campaign-id", "c1", "--campaign-root", text]
    assert main(["campaign", *argv, "--dry-run"], repo=repo) == 2
    err = capsys.readouterr().err
    assert f"--campaign-root {root} holds no campaign (no sampling_manifest.json)" in err and "--new-root" in err
    assert not root.exists()
    [job] = dry(repo, capsys, *argv, "--new-root")["jobs"]
    assert given(job)[-1] == f"campaign_root={root}"
    root.mkdir(parents=True)
    (root / "sampling_manifest.json").write_text("{}\n")
    printed = dry(repo, capsys, *argv)
    assert given(printed["jobs"][0])[-1] == f"campaign_root={root}"
    recorded = printed["campaign"]["conditions"][0]["campaign_root"]
    assert recorded == (str(root) if "elsewhere" in spelled else "roots/rebased")


def test_a_population_comes_with_the_digest_it_carries(repo, tmp_path, capsys):
    path, digest = population(tmp_path)
    for argv, message in (
        (["--population", "pops/p.jsonl"], "--population needs --expect-population-sha256"),
        (["--expect-population-sha256", digest], "without --population would check nothing"),
        (["--population", "pops/absent.jsonl", "--expect-population-sha256", digest], "is not a file"),
        (["--population", "pops/p.jsonl", "--expect-population-sha256", "ab" * 32], f"its digest {digest} is not"),
    ):
        assert main(["campaign", *SLURM, "--campaign-id", "c1", "--dry-run", *argv], repo=repo) == 2
        assert message in capsys.readouterr().err
    for spelled in ("pops/p.jsonl", str(path)):
        [job] = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--population", spelled,
                    "--expect-population-sha256", digest)["jobs"]
        assert given(job)[-2:] == [f"scenario_set_path={path}", f"approved_scenario_sha256={digest}"]
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--population", "pops/p.jsonl",
                 "--expect-population-sha256", digest, "--dry-run", "--", "scenario_set_path=x"], repo=repo) == 2
    assert "'scenario_set_path=x' sets what --population sets" in capsys.readouterr().err


@pytest.mark.parametrize(("switch", "packed"), [
    ("--pack-draws", True), ("--pack-draws=1", True), ("--pack-draws=true", True), ("--pack-draws=yes", True),
    ("--pack-draws=on", True), ("--pack-draws=0", False), ("--pack-draws=false", False), ("--pack-draws=no", False),
    ("--pack-draws=off", False), ("--pack-draws=", False), ("--no-pack-draws", False),
])
def test_pack_draws_is_a_switch(repo, capsys, switch, packed):
    [job] = dry(repo, capsys, *SLURM, "--campaign-id", "c1", switch)["jobs"]
    assert [arg for arg in given(job) if arg.startswith("execution.")] == (["execution.pack_draws=true"] if packed
                                                                           else [])


@pytest.mark.parametrize("switch", ["--pack-draws=2", "--pack-draws=TRUE", "--pack-draws=flase", "--pack-draws= 1",
                                    "--no-pack-draws=1"])
def test_a_pack_draws_that_is_not_a_switch_is_refused(repo, capsys, switch):
    with pytest.raises(SystemExit) as refused:
        main(["campaign", *SLURM, "--campaign-id", "c1", switch, "--dry-run"], repo=repo)
    assert refused.value.code == 2
    assert re.search(r"is not a switch: 1\|true\|yes\|on or 0\|false\|no\|off|takes no value",
                     capsys.readouterr().err)


def test_the_packing_bounds_tune_only_a_packing_that_is_on(repo, capsys):
    [job] = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--pack-draws", "--pack-max-prompts", "2048",
                "--pack-max-draws", "300")["jobs"]
    assert [arg for arg in given(job) if arg.startswith("execution.")] == [
        "execution.pack_draws=true", "execution.pack_max_prompts=2048", "execution.pack_max_draws=300"]
    for argv in (["--pack-max-prompts", "8"], ["--pack-max-draws", "8", "--pack-draws=off"]):
        assert main(["campaign", *SLURM, "--campaign-id", "c1", "--dry-run", *argv], repo=repo) == 2
        assert "which only --pack-draws turns on" in capsys.readouterr().err
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--pack-draws", "--dry-run", "--",
                 "execution.pack_draws=false"], repo=repo) == 2
    assert "sets what --pack-draws sets" in capsys.readouterr().err


@pytest.mark.parametrize(("argument", "message"), [
    ("campaign_id=other", "reaches 'campaign_id'"), ("campaign_root=/elsewhere", "reaches 'campaign_root'"),
    ("+campaign_root=/x", "reaches 'campaign_root'"), ("execution.shard_index=1", "reaches 'execution.shard_index'"),
    ("+job_root=/x", "reaches 'job_root'"), ("hydra.run.dir=/x", "reaches 'hydra.run.dir'"),
    ("-m", "no -m"), ("eval_cfg.reasoning_prompt_style=a,b", "is a sweep"),
    ("hydra.output_subdir=/published/root/.hydra", "gives Hydra's 'hydra.output_subdir' a value: of Hydra's own"),
    ("hydra.output_subdir=../../../published", "gives Hydra's 'hydra.output_subdir' a value"),
    ("hydra.job_logging.handlers.file.filename=/published/x.log",
     "gives Hydra's 'hydra.job_logging.handlers.file.filename' a value"),
    ("hydra.job.env_set.RULEFORGE_ARTIFACT_ROOT=/published", "gives Hydra's 'hydra.job.env_set.RULEFORGE_ARTIFACT_ROOT' a value"),
    ("hydra.job.name=../../../../published/provenance", "gives Hydra's 'hydra.job.name' a value"),
    ("hydra.searchpath=[file:///elsewhere]", "gives Hydra's 'hydra.searchpath' a value"),
])
def test_what_the_verb_sets_is_refused_in_the_arguments(repo, capsys, argument, message):
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--dry-run", "--", argument], repo=repo) == 2
    assert message in capsys.readouterr().err


def test_what_the_verb_sets_is_refused_on_an_axis_too(repo, capsys):
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--dry-run", "--axis", "campaign_id=a,b"], repo=repo) == 2
    assert "reaches 'campaign_id'" in capsys.readouterr().err


def test_axes_and_shards_fan_out_into_conditions_and_their_shards(repo, tmp_path, capsys):
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--shards", "3",
                  "--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}", "--", "sampling.num_decoder_draws=2")
    jobs = printed["jobs"]
    assert [job["job_id"] for job in jobs] == [f"campaign-{c}-shard-{i}-of-3" for c in range(2) for i in range(3)]
    root = tmp_path / "artifacts" / "sampling" / "c1"
    assert given(jobs[4]) == ["sampling.num_decoder_draws=2", "eval_cfg.reasoning_prompt_style=encouraged",
                              "campaign_id=c1", f"campaign_root={root}", "execution.num_shards=3",
                              "execution.shard_index=1", "job_root=${campaign_root}.shards/${condition_id}/shard-1-of-3"]
    # One array: the jobs share their parameters. On SLURM the shard roots need no pull; the merges run at pull.
    assert all(job["outputs"] == [] for job in jobs)
    assert printed["merges"] == [{"campaign_root": "sampling/c1", "condition_id": f"gemma-4-31B-it/{style}",
                                  "num_shards": 3, "job_ids": [f"campaign-{c}-shard-{i}-of-3" for i in range(3)]}
                                 for c, style in enumerate(STYLES)]
    assert [condition["selection"] for condition in printed["campaign"]["conditions"]] == [
        ["--axis", f"eval_cfg.reasoning_prompt_style={style}"] for style in STYLES]
    preflights = printed["campaign"]["preflights"]
    assert len(preflights) == 2 and preflights[1][-4:] == [
        "execution.num_shards=3", "execution.shard_index=0",
        "job_root=${campaign_root}.shards/${condition_id}/shard-0-of-3", SHARD_PREFLIGHT_RUN_DIR]
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--shards", "3", "--only", "2,1",
                  "--axis", "eval_cfg.reasoning_prompt_style=encouraged")
    assert [job["job_id"] for job in printed["jobs"]] == [f"campaign-0-shard-{i}-of-3" for i in (1, 2)]
    assert all("eval_cfg.reasoning_prompt_style=encouraged" in given(job) for job in printed["jobs"])
    assert "execution.shard_index=1" in printed["campaign"]["preflights"][0]
    assert printed["merges"] == [{"campaign_root": "sampling/c1", "condition_id": "gemma-4-31B-it/encouraged",
                                  "num_shards": 3, "job_ids": [f"campaign-0-shard-{i}-of-3" for i in (1, 2)]}]


def test_a_paired_axis_sets_both_keys_to_each_value(repo, capsys):
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1",
                  "--axis", "model@judge_model+model@agent_model=gpt-oss-tp-2,olmo3-32b-tp-2",
                  "--axis", "eval_cfg.reasoning_prompt_style=naive-rules-above,encouraged")
    assert [condition["condition_id"] for condition in printed["campaign"]["conditions"]] == [
        "gpt-oss-120b/naive-rules-above", "gpt-oss-120b/encouraged", "Olmo-3-32B-Think/naive-rules-above", "Olmo-3-32B-Think/encouraged"]
    assert given(printed["jobs"][2])[:3] == ["model@judge_model=olmo3-32b-tp-2", "model@agent_model=olmo3-32b-tp-2",
                                             "eval_cfg.reasoning_prompt_style=naive-rules-above"]


def blocks(tmp_path: Path, directory: str = "pops/x.blocks-2", count: int = 2) -> list[tuple[Path, str]]:
    found = []
    for index in range(count):
        found.append(population(tmp_path, f"{directory}/x.block-{index:03d}.jsonl", f'{{"prompt": "{index}"}}\n'))
    (tmp_path / "artifacts" / directory / "x.block-000.manifest.json").write_text("{}\n")  # a sidecar: not a block
    return found


def test_conditions_sharing_a_root_write_it_directly_unless_sharded(repo, capsys):
    axis = ["--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}"]
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", *axis)
    assert [job["job_id"] for job in printed["jobs"]] == ["campaign-0", "campaign-1"] and printed["merges"] == []
    assert all(job["outputs"] == [] for job in printed["jobs"])
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--shards", "1", *axis)
    assert [job["job_id"] for job in printed["jobs"]] == ["campaign-0-shard-0-of-1", "campaign-1-shard-0-of-1"]
    assert given(printed["jobs"][1])[-3:] == ["execution.num_shards=1", "execution.shard_index=0",
                                              "job_root=${campaign_root}.shards/${condition_id}/shard-0-of-1"]
    assert printed["merges"] == [{"campaign_root": "sampling/c1", "condition_id": f"gemma-4-31B-it/{style}",
                                  "num_shards": 1, "job_ids": [f"campaign-{c}-shard-0-of-1"]}
                                 for c, style in enumerate(STYLES)]


def test_each_block_is_a_condition_in_a_root_of_its_own(repo, tmp_path, capsys):
    (first, first_digest), (second, second_digest) = blocks(tmp_path)
    extra, extra_digest = population(tmp_path, "pops/y.jsonl")
    argv = [*SLURM, "--blocks", "pops/x.blocks-2", "--blocks", str(extra), "--root-template", "sampling/screen/{block}",
            "--", "--config-name", "sampling_training_anchors", "grpo_anchors.stage=screen",
            "grpo_anchors.campaign_prefix=p"]
    printed = dry(repo, capsys, *argv)
    jobs = printed["jobs"]
    assert [job["job_id"] for job in jobs] == ["campaign-x.block-000", "campaign-x.block-001", "campaign-y"]
    artifacts = tmp_path / "artifacts"
    assert given(jobs[1])[-3:] == [f"campaign_root={artifacts}/sampling/screen/x.block-001",
                                   f"scenario_set_path={artifacts}/pops/x.blocks-2/x.block-001.jsonl",
                                   f"approved_scenario_sha256={second_digest}"]
    assert given(jobs[2])[-1] == f"approved_scenario_sha256={extra_digest}"
    assert all(job["outputs"] == [] for job in jobs) and printed["merges"] == []
    assert [condition["selection"] for condition in printed["campaign"]["conditions"]] == [
        ["--blocks", "pops/x.blocks-2/x.block-000.jsonl"], ["--blocks", "pops/x.blocks-2/x.block-001.jsonl"],
        ["--blocks", "pops/y.jsonl"]]
    assert {c["condition_id"] for c in printed["campaign"]["conditions"]} == {"Qwen3-8B/grpo-v2-anchors/screen"}
    assert {c["campaign_id"] for c in printed["campaign"]["conditions"]} == {"p-screen"}


def test_blocks_crossed_with_an_axis_write_each_block_s_root_one_condition_a_job(repo, tmp_path, capsys):
    blocks(tmp_path)
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--blocks", "pops/x.blocks-2", "--root-template",
                  "sampling/screen/{block}", "--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}")
    assert [job["job_id"] for job in printed["jobs"]] == [f"campaign-{c}-x.block-{b:03d}" for c in range(2)
                                                          for b in range(2)]
    assert sorted((c["campaign_root"], c["condition_id"]) for c in printed["campaign"]["conditions"]) == sorted(
        (f"sampling/screen/x.block-{b:03d}", f"gemma-4-31B-it/{style}") for style in STYLES for b in range(2))
    assert printed["merges"] == []


def test_the_dry_run_prints_one_preflight_a_condition_and_a_run_a_shard(repo, capsys):
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--shards", "4", "--dry-run"], repo=repo) == 0
    captured = capsys.readouterr()
    assert captured.out.count("== preflight campaign (local): ") == 1
    assert captured.out.count("ruleforge.launch.programs.run_campaign") == 4
    assert "dry run: nothing was submitted" in captured.out


def test_a_campaign_plan_composes_its_one_condition(repo):
    from ruleforge.launch.jobs import JobSpec, Resources

    template = JobSpec("campaign", "job", 0, ("python", campaign.PROGRAM[-1]), Resources("gpu", cpus=4, gpus=1, mem_gb=24,
                                                                                  timeout_min=60))
    options = CampaignOptions(campaign_id="c1")
    [condition] = campaign.plan(repo, template, [], (), options, exists=lambda root: False)
    assert condition.condition_id == "gemma-4-31B-it/naive-rules-above"


@pytest.mark.parametrize(("argv", "message"), [
    (["--blocks", "pops/x.blocks-2"], "--blocks needs --root-template"),
    (["--blocks", "pops/x.blocks-2", "--root-template", "sampling/screen"], "--blocks needs --root-template"),
    (["--root-template", "sampling/{block}"], "--root-template names each block's root: give --blocks"),
    (["--blocks", "pops/x.blocks-2", "--root-template", "s/{block}", "--campaign-root", "r"], "each block has a root"),
    (["--blocks", "pops/x.blocks-2", "--root-template", "s/{block}", "--population", "pops/y.jsonl",
      "--expect-population-sha256", "ab"], "each block is its own population"),
    (["--blocks", "pops/x.blocks-2", "--blocks", "pops/x.blocks-2/x.block-000.jsonl", "--root-template", "s/{block}"],
     "names x.block-000 twice"),
    (["--blocks", "pops/empty", "--root-template", "s/{block}"], "no \\*.jsonl block in it"),
    (["--blocks", "pops/absent", "--root-template", "s/{block}"], "no such file or directory"),
    (["--blocks", "pops/x.blocks-2", "--root-template", "s/{block}", "--", "scenario_set_path=x"],
     "sets what --blocks sets"),
])
def test_blocks_are_refused_where_they_would_not_be_one_root_each(repo, tmp_path, capsys, argv, message):
    blocks(tmp_path)
    (tmp_path / "artifacts" / "pops" / "empty").mkdir()
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--dry-run", *argv], repo=repo) == 2
    assert re.search(message, capsys.readouterr().err)


@pytest.mark.parametrize(("argv", "message"), [
    (["--axis", "sampling.num_decoder_draws=2,3"],
     "two jobs of the campaign would write condition gemma-4-31B-it/naive-rules-above"),
    (["--axis", "eval_cfg.reasoning_prompt_style=../x"], "condition_id .* must name a path below the root"),
    (["--axis", "eval_cfg.reasoning_prompt_style=a//b"], "is not a plain relative path"),
    (["--", "--config-name", "nonesuch"], "does not compose: MissingConfigException"),
    (["--", "--config-dir", "elsewhere"], "--config-dir: a job's config is one of configs/"),
    (["--shards", "0"], "--shards 0: at least 1"),
    (["--only", "1"], "--only selects shards: give --shards"),
    (["--shards", "2", "--only", "a"], "comma-separated shard indices"),
    (["--shards", "2", "--only", "2"], r"shard indices in \[0, 2\)"),
    (["--shards", "2", "--only", ","], r"shard indices in \[0, 2\)"),
    (["--expect-commit", "abc"], "7 to 40 hex digits"),
    (["--expect-commit", "bbbbbbb"], "not --expect-commit bbbbbbb"),
    (["--campaign-id", "a b"], "--campaign-id 'a b'"),
    (["--axis", "training.policy;rm=base"], "expected KEY=VALUES"),
    (["--axis", "eval_cfg.reasoning_prompt_style=a,a"], "gives a more than once"),
    # --only relaunches one condition's shards: of two, the other's running shards would launch again.
    (["--shards", "2", "--only", "1", "--axis", "eval_cfg.reasoning_prompt_style=a,b"],
     "--only relaunches shards of one condition, and this campaign has 2"),
    (["--preflight", "none", "--preflight-only"], "--preflight-only with --preflight none would check nothing"),
    (["--campaign-root", "${oc.env:RULEFORGE_ARTIFACT_ROOT}/sampling/x", "--new-root"], "not an interpolation"),
    (["--population", "$RULEFORGE_ARTIFACT_ROOT/p.jsonl", "--expect-population-sha256", "ab"], "not an interpolation"),
])
def test_a_campaign_it_cannot_run_as_asked_is_refused_before_anything_runs(repo, capsys, argv, message):
    popen = FakePopen([])
    assert main(["campaign", *SLURM, "--campaign-id", "c1", *argv], repo=repo, popen=popen) == 2
    assert re.search(message, capsys.readouterr().err) and popen.calls == []


def test_a_root_a_config_names_relative_is_refused(repo, capsys):
    assert main(["campaign", *SLURM, "--dry-run", "--", "--config-name", "sampling_training_anchors",
                 "grpo_anchors.stage=screen", "grpo_anchors.campaign_prefix=p",
                 "grpo_anchors.campaign_root=relative/root"], repo=repo) == 2
    assert "campaign's campaign_root 'relative/root' is relative" in capsys.readouterr().err


def test_a_local_judge_is_refused_a_job_with_no_gpu(repo, capsys):
    assert main(["campaign", "--backend", "slurm", "--campaign-id", "c1", "--dry-run"], repo=repo) == 2
    assert "campaign's judge is a local model, served by vLLM on the job's GPUs: give --gpu-count" in \
        capsys.readouterr().err
    # A judge behind an API needs no GPU, and a job on this machine has the GPUs it sees.
    [job] = dry(repo, capsys, "--backend", "slurm", "--campaign-id", "c1", "--", "model@judge_model=gpt-5.6-luna")[
        "jobs"]
    assert job["resources"]["gpus"] == 0
    assert main(["campaign", "--backend", "local", "--campaign-id", "c1", "--dry-run"], repo=repo) == 0


@pytest.mark.parametrize(("key", "value"), [("new-root", "true"), ("preflight", "none"), ("preflight-only", "true"),
                                            ("only", "'1'")])
def test_a_spec_cannot_waive_a_campaign_s_checks(repo, tmp_path, capsys, key, value):
    spec = tmp_path / "waiver.yaml"
    spec.write_text(f"backend: slurm\ngpu-count: 1\ncampaign-id: c1\n{key}: {value}\n")
    assert main(["campaign", "--spec", str(spec), "--dry-run"], repo=repo) == 2
    assert f"unknown key(s) {key}" in capsys.readouterr().err


class Preflights(FakePopen):
    """A preflight (or runner) subprocess per call: exits ``codes[i]`` (0 by default), printing ``lines``."""

    def __init__(self, codes=(), lines=()):
        self.codes = list(codes)
        self.lines = list(lines)
        super().__init__([])

    def __call__(self, argv, **kwargs):
        index = len(self.calls)
        self.launches.append((self.codes[index] if index < len(self.codes) else 0,
                              self.lines[index] if index < len(self.lines) else []))
        return super().__call__(argv, **kwargs)


def test_each_condition_is_preflighted_once_here_before_one_array_is_submitted(repo, tmp_path, capsys):
    log, executor = _executors()
    popen = Preflights()
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--shards", "3", "--resumable", "--json",
            "--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}"]
    assert main(argv, repo=repo, popen=popen, executor=executor) == 0
    assert len(popen.calls) == 2, "one preflight per condition, not per shard"
    for (command, cwd), style in zip(popen.calls, STYLES):
        assert command[:2] == ["bash", "-c"] and "export TIKTOKEN_RS_CACHE_DIR=$HOME/.cache/tiktoken" in command[2]
        assert command[4:7] == list(campaign.PREFLIGHT_PROGRAM) and cwd == str(repo)
        assert f"eval_cfg.reasoning_prompt_style={style}" in command and "execution.shard_index=0" in command
    assert len(log["jobs"]) == 6 and all(batched for _, _, batched, _ in log["jobs"])
    result = _json(capsys)
    record = LaunchRecord.load(result["record"])
    assert record.data["verb"] == "campaign" and len(record.data["merges"]) == 2
    assert record.data["campaign"]["preflight"] == "local"


def test_a_refused_preflight_submits_nothing_and_writes_no_record(repo, tmp_path, capsys):
    log, executor = _executors()
    popen = Preflights(codes=[0, 1])
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--axis", "eval_cfg.reasoning_prompt_style=a,b,c"],
                repo=repo, popen=popen, executor=executor) == 1
    assert len(popen.calls) == 2 and log["jobs"] == []
    assert "the preflight of campaign-1 refused it (exit 1): nothing was submitted" in capsys.readouterr().out
    assert not (tmp_path / "artifacts" / ".ruleforge-launch").exists()


def test_preflight_only_stops_after_the_preflights_and_none_skips_them(repo, tmp_path, capsys):
    log, executor = _executors()
    popen = Preflights()
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--preflight-only",
                 "--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}"],
                repo=repo, popen=popen, executor=executor) == 0
    assert len(popen.calls) == 2 and log["jobs"] == []
    assert "2 condition(s) passed their preflight; nothing was submitted" in capsys.readouterr().out
    assert not (tmp_path / "artifacts" / ".ruleforge-launch").exists()
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--preflight", "none"], repo=repo, popen=popen,
                executor=executor) == 0
    assert len(popen.calls) == 2 and len(log["jobs"]) == 1


def relaunch_options(out: str, job_id: str) -> list[str]:
    """The options a launch's hint names to relaunch one condition's missed jobs alone."""
    [line] = [line for line in out.splitlines() if line.startswith(f"== {job_id} (")]
    return shlex.split(re.search(r"narrowed to this condition: (.*) in place of its --axis", line).group(1))


def merging(monkeypatch) -> list[str]:
    merged = []

    def merge(root, condition_id, *, num_shards=None):
        missing = [index for index in range(num_shards)
                   if not shard_root(root, condition_id, index, num_shards).is_dir()]
        if missing:
            raise ShardError(f"lacks shard(s) {missing} of {num_shards}")
        merged.append(condition_id)

    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards", merge)
    return merged


def test_on_slurm_a_failed_shard_is_relaunched_in_place_and_the_relaunch_s_pull_merges(repo, tmp_path, capsys,
                                                                                         monkeypatch):
    merged = merging(monkeypatch)
    _, executor = _executors()
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--shards", "2", "--preflight", "none"]
    assert main([*argv, "--json"], repo=repo, executor=executor) == 0
    first = LaunchRecord.load(_json(capsys)["launch_id"])
    done, failed = (run["slurm_job"] for run in first.runs)
    assert main(["wait", first.launch_id, "--pull"], repo=repo,
                run=Completed({"sacct": [(0, f"{done}|COMPLETED\n{failed}|FAILED\n", "")]})) == 1
    out = capsys.readouterr().out
    assert "launch shard(s) 1 again with the same command and --shards 2 --only 1, from commit" in out
    assert (f"== pull {first.launch_id} once its runs have ended, and each relaunch once its runs and "
            f"{first.launch_id}'s have") in out
    assert "merge of gemma-4-31B-it/naive-rules-above into sampling/c1 skipped: 1 failed of its 2 shard run(s) here" in out
    assert "left on the volume" not in out
    for index in range(2):
        shard_root(tmp_path / "artifacts" / "sampling" / "c1", "gemma-4-31B-it/naive-rules-above", index, 2).mkdir(parents=True)
    assert main([*argv, "--only", "1", "--json"], repo=repo, executor=executor,
                run=Completed({"sacct": [(0, f"{done}|COMPLETED\n{failed}|FAILED\n", "")]})) == 0
    relaunch = LaunchRecord.load(_json(capsys)["launch_id"])
    [run] = relaunch.runs
    assert main(["wait", relaunch.launch_id, "--pull"], repo=repo,
                run=Completed({"sacct": [(0, f"{run['slurm_job']}|COMPLETED\n", "")]})) == 0
    assert merged == ["gemma-4-31B-it/naive-rules-above"] and "merged 1" in capsys.readouterr().out
    assert main(["pull", first.launch_id], repo=repo,
                run=Completed({"sacct": [(0, f"{done}|COMPLETED\n{failed}|FAILED\n", "")]})) == 0
    assert f"sampling/c1: merged already by {relaunch.launch_id}" in capsys.readouterr().out
    assert merged == ["gemma-4-31B-it/naive-rules-above"]
    # Unsharded on SLURM, nothing is pulled or merged: the hint names no pull.
    assert main(["campaign", *SLURM, "--campaign-id", "c2", "--preflight", "none", "--json"], repo=repo,
                executor=executor) == 0
    plain = LaunchRecord.load(_json(capsys)["launch_id"])
    [run] = plain.runs
    assert main(["wait", plain.launch_id], repo=repo,
                run=Completed({"sacct": [(0, f"{run['slurm_job']}|FAILED\n", "")]})) == 1
    out = capsys.readouterr().out
    assert "launch it again with the same command, from commit" in out and "== pull" not in out


def test_a_failed_condition_s_shard_is_relaunched_alone_and_a_later_relaunch_is_named_not_launched_again(
        repo, capsys, monkeypatch):
    merging(monkeypatch)
    _, executor = _executors()
    common = ["campaign", *SLURM, "--campaign-id", "c1", "--preflight", "none", "--shards", "2"]
    axis = ["--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}"]
    assert main([*common, *axis, "--json"], repo=repo, executor=executor) == 0
    first = LaunchRecord.load(_json(capsys)["launch_id"])
    jobs = [run["slurm_job"] for run in first.runs]
    states = "".join(f"{job}|{'FAILED' if index == 3 else 'COMPLETED'}\n" for index, job in enumerate(jobs))
    assert main(["wait", first.launch_id], repo=repo, run=Completed({"sacct": [(0, states, "")]})) == 1
    out = capsys.readouterr().out
    assert "campaign-0 (" not in out
    options = relaunch_options(out, "campaign-1")
    assert options == ["--axis", "eval_cfg.reasoning_prompt_style=encouraged", "--shards", "2", "--only", "1"]
    assert main([*common, *options, "--json"], repo=repo, executor=executor,
                run=Completed({"sacct": [(0, states, "")]})) == 0
    relaunch = LaunchRecord.load(_json(capsys)["launch_id"])
    assert [job["job_id"] for job in relaunch.jobs] == ["campaign-0-shard-1-of-2"]
    assert main(["wait", first.launch_id], repo=repo, run=Completed({"sacct": [(0, states, "")]})) == 1
    out = capsys.readouterr().out
    assert (f"== campaign-1 (gemma-4-31B-it/encouraged): campaign-1-shard-1-of-2 launched again by "
            f"{relaunch.launch_id}: follow that launch") in out and "launch shard(s)" not in out


def test_a_block_that_failed_is_named_alone_for_a_relaunch_with_no_merge_to_follow(repo, tmp_path, capsys):
    blocks(tmp_path)
    _, executor = _executors()
    argv = ["campaign", *SLURM, "--preflight", "none", "--blocks", "pops/x.blocks-2", "--root-template",
            "sampling/screen/{block}", "--json", "--", "--config-name", "sampling_training_anchors",
            "grpo_anchors.stage=screen", "grpo_anchors.campaign_prefix=p"]
    assert main(argv, repo=repo, executor=executor) == 0
    record = LaunchRecord.load(_json(capsys)["launch_id"])
    done, failed = (run["slurm_job"] for run in record.runs)
    assert main(["wait", record.launch_id], repo=repo,
                run=Completed({"sacct": [(0, f"{done}|COMPLETED\n{failed}|FAILED\n", "")]})) == 1
    out = capsys.readouterr().out
    assert relaunch_options(out, "campaign-x.block-001") == ["--blocks", "pops/x.blocks-2/x.block-001.jsonl"]
    assert "a relaunch resumes from its ledger\n" in out and "== pull" not in out and "campaign-x.block-000 (" not in out


class StopsAfterTheFirst(FakePopen):
    """A local child process that requests the launch's stop once the first job starts."""

    def __call__(self, argv, **kwargs):
        process = super().__call__(argv, **kwargs)
        [record] = records_root().iterdir()
        (record / "stop.json").write_text("[]")
        return process


def test_a_campaign_stopped_part_way_names_the_conditions_it_did_not_launch(repo, capsys):
    argv = ["campaign", "--backend", "local", "--campaign-id", "c1", "--preflight", "none", "--axis",
            f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}", "--", "model@judge_model=gpt-5.6-luna"]
    assert main(argv, repo=repo, popen=StopsAfterTheFirst([(0, ["ran"])])) == 2
    captured = capsys.readouterr()
    assert "a stop of" in captured.err and "no later job was submitted" in captured.err
    assert relaunch_options(captured.out, "campaign-1") == ["--axis", "eval_cfg.reasoning_prompt_style=encouraged"]
    assert "campaign-0 (" not in captured.out


def test_shared_jobs_are_the_same_root_condition_and_shard_of_another_launch_s_plan():
    mine = {"job_id": "campaign-1", "campaign_root": "/a/sampling/c", "condition_id": "m/naive-rules-above", "shards": 2}
    other = {"jobs": [{"job_id": f"relaunch-{shard}"} for shard in ("shard-0-of-2", "shard-1-of-2")],
             "runs": [{"job_id": "relaunch-shard-1-of-2"}, {"job_id": "relaunch-shard-0-of-2", "refused": "x"}],
             "campaign": {"conditions": [
                 {"job_id": "relaunch", "campaign_root": "/a/sampling/c", "condition_id": "m/naive-rules-above", "shards": 2},
                 {"job_id": "relaunch", "campaign_root": "/a/sampling/c", "condition_id": "m/encouraged", "shards": 2},
                 {"job_id": "whole", "campaign_root": "/a/sampling/c", "condition_id": "m/naive-rules-above", "shards": None}]}}
    assert campaign.shared_jobs(other, mine) == {"campaign-1-shard-0-of-2": "relaunch-shard-0-of-2",
                                                 "campaign-1-shard-1-of-2": "relaunch-shard-1-of-2"}
    assert campaign.relaunched_jobs(other, mine) == {"campaign-1-shard-1-of-2"}
    plain = {**mine, "shards": None}
    assert campaign.shared_jobs({**other, "jobs": [{"job_id": "whole"}]}, plain) == {"campaign-1": "whole"}
    assert campaign.shared_jobs(other, plain) == {} and campaign.relaunched_jobs(other, plain) == set()
    assert campaign.shared_jobs({"jobs": [], "runs": []}, mine) == {}
    relative = {**mine, "campaign_root": "sampling/c"}
    assert campaign.shared_jobs(other, relative) == {}
    assert campaign.shared_jobs(other, relative, is_root=lambda root: campaign.local_root(root, Path("/b")) == Path(
        "/a/sampling/c")) == campaign.shared_jobs(other, mine)
    assert campaign.local_root("sampling/c", Path("/b")) == Path("/b/sampling/c").resolve()
    assert campaign.local_root("/a/sampling/c", Path("/b")) == Path("/a/sampling/c").resolve()


def test_a_campaign_job_is_refused_while_another_launch_s_run_of_it_has_not_ended(repo, tmp_path, capsys):
    _, executor = _executors()
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--preflight", "none"]
    assert main([*argv, "--json"], repo=repo, executor=executor) == 0
    first = LaunchRecord.load(_json(capsys)["launch_id"])
    [run] = first.runs
    running = Completed({"sacct": [(0, f"{run['slurm_job']}|RUNNING\n", "")]})
    assert main([*argv, "--json"], repo=repo, executor=executor, run=running) == 2
    err = capsys.readouterr().err
    assert (f"{first.launch_id} runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 running): its run has not ended, and "
            "two runs would write one root at once" in err) and "--ignore-launch <its id>" in err
    assert len(list((tmp_path / "artifacts" / ".ruleforge-launch").iterdir())) == 1
    assert main([*argv, "--preflight-only"], repo=repo, executor=executor, run=running) == 2
    assert main([*argv, "--dry-run"], repo=repo, executor=executor, run=running) == 0
    assert main(["campaign", *SLURM, "--campaign-id", "c2", "--preflight", "none"], repo=repo, executor=executor,
                run=running) == 0
    capsys.readouterr()
    assert main([*argv, "--ignore-launch", first.launch_id, "--json"], repo=repo, executor=executor,
                run=running) == 0
    ignoring = LaunchRecord.load(_json(capsys)["launch_id"])
    assert ignoring.data["ignored_launches"] == [first.launch_id]
    [second] = ignoring.runs
    ended = Completed({"sacct": [(0, f"{run['slurm_job']}|COMPLETED\n{second['slurm_job']}|COMPLETED\n", "")]})
    capsys.readouterr()
    assert main([*argv, "--json"], repo=repo, executor=executor, run=ended) == 0
    [third] = LaunchRecord.load(_json(capsys)["launch_id"]).runs
    ended = Completed({"sacct": [(0, "".join(f"{job['slurm_job']}|COMPLETED\n" for job in (run, second, third)), "")]})
    sharded = ["campaign", *SLURM, "--campaign-id", "c3", "--shards", "2", "--preflight", "none"]
    capsys.readouterr()
    assert main([*sharded, "--json"], repo=repo, executor=executor) == 0
    zero, one = (run["slurm_job"] for run in LaunchRecord.load(_json(capsys)["launch_id"]).runs)
    states = Completed({"sacct": [(0, f"{zero}|RUNNING\n{one}|FAILED\n", "")]})
    assert main([*sharded, "--only", "1"], repo=repo, executor=executor, run=states) == 0
    assert main([*sharded, "--only", "0"], repo=repo, executor=executor, run=states) == 2
    assert "(1 running): its run has not ended" in capsys.readouterr().err
    dead = LaunchRecord.create(tmp_path / "artifacts" / ".ruleforge-launch" / "LDEAD", {
        **{key: first.data[key] for key in ("options", "jobs", "campaign")}, "launch_id": "LDEAD", "backend": "local"})
    assert dead.data["state"] == "submitting" and dead.runs == []
    assert main(argv, repo=repo, executor=executor, run=ended) == 2
    assert "LDEAD runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 submitting)" in capsys.readouterr().err
    assert main([*argv, "--ignore-launch", "LDEAD", "--json"], repo=repo, executor=executor, run=ended) == 0
    [fourth] = LaunchRecord.load(_json(capsys)["launch_id"]).runs
    ended = Completed({"sacct": [(0, "".join(f"{job['slurm_job']}|COMPLETED\n"
                                             for job in (run, second, third, fourth)), "")]})
    job_id = first.jobs[0]["job_id"]
    dead.add_run({"job_id": job_id, "log": "logs/x.log"})
    assert main(argv, repo=repo, executor=executor, run=ended) == 2
    assert "LDEAD runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 submitting)" in capsys.readouterr().err
    dead.finish(BrokenPipeError("the terminal went away"))
    assert main(argv, repo=repo, executor=executor, run=ended) == 2
    assert "LDEAD runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 unknown)" in capsys.readouterr().err
    dead.update_run(job_id, error="BrokenPipeError: the terminal went away")
    assert main(argv, repo=repo, executor=executor, run=ended) == 2
    assert "LDEAD runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 unknown)" in capsys.readouterr().err
    dead.update_run(job_id, exit_code=None)
    assert main(argv, repo=repo, executor=executor, run=ended) == 0


def test_a_campaign_whose_slurm_submission_raised_after_sbatch_stays_live_until_a_look_by_hand(repo, capsys):
    log = {"submitted": 0, "jobs": [], "executors": []}

    def raising(folder, max_num_timeout):
        executor = FakeExecutor(log, folder, max_num_timeout)
        submit = executor.submit

        def queued_then_raises(command):
            submit(command)
            raise OSError("[Errno 116] Stale file handle")

        executor.submit = queued_then_raises
        return executor

    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--preflight", "none"]
    assert main(argv, repo=repo, executor=raising) == 2
    captured = capsys.readouterr()
    [directory] = sorted(records_root().iterdir())
    first = LaunchRecord.load(directory)
    [job_id] = [job["job_id"] for job in first.jobs]
    assert log["submitted"] == 1 and first.runs == [{"job_id": job_id, "slurm_job": None}]
    assert f"== {job_id}: the launch failed after sbatch was asked to queue it, and no job id was read" in captured.out
    assert (f"launched no job for certain: launch again with the same command once that look found none of it "
            f"queued or running, with --ignore-launch {first.launch_id}, from commit") in captured.out
    _, executor = _executors()
    assert main(argv, repo=repo, executor=executor, run=Completed()) == 2
    assert f"{first.launch_id} runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 unknown)" in capsys.readouterr().err
    assert main([*argv, "--ignore-launch", first.launch_id], repo=repo, executor=executor, run=Completed()) == 0


class RivalDuringPreflight(FakePopen):
    """A local preflight during which another launch of the same campaign records and submits; ``rival`` is its id."""

    def __init__(self, repo: Path, argv: list[str], executor):
        super().__init__([(0, ["preflight ok"])])
        self.repo, self.argv, self.executor, self.rival = repo, argv, executor, None

    def __call__(self, argv, **kwargs):
        assert main(self.argv, repo=self.repo, executor=self.executor) == 0
        self.rival = sorted(records_root().iterdir())[-1].name
        return super().__call__(argv, **kwargs)


def test_a_campaign_launch_claims_its_roots_and_is_refused_by_a_rival_that_recorded_during_its_preflight(
        repo, tmp_path, capsys):
    _, executor = _executors()
    argv = ["campaign", *SLURM, "--campaign-id", "c1"]
    rival = RivalDuringPreflight(repo, [*argv, "--preflight", "none", "--json"], executor)
    unknown = Completed({"sacct": [(0, "", "")]})  # the rival's run: not reported, so not ended
    assert main([*argv, "--preflight", "local"], repo=repo, executor=executor, popen=rival, run=unknown) == 2
    err = capsys.readouterr().err
    assert f"{rival.rival} runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 unknown): its run has not ended" in err
    [record] = [LaunchRecord.load(path) for path in records_root().iterdir()]
    assert record.launch_id == rival.rival and len(rival.calls) == 1
    root = tmp_path / "artifacts" / "sampling" / "c1"
    [claim] = sorted(writers_dir(root).iterdir())
    assert claim.name == f"{rival.rival}.json"
    written = json.loads(claim.read_text(encoding="utf-8"))
    assert written["record"] == str(record.directory) and written["data"]["state"] == "submitting"
    assert written["data"]["runs"] == [] and written["data"]["campaign"] == record.data["campaign"]
    assert lock_path(records_root()).is_file() and lock_path(writers_dir(root)).is_file()
    assert sorted(path.name for path in (tmp_path / "artifacts").iterdir()) == [".ruleforge-launch", ".ruleforge-launch.lock",
                                                                                  "sampling"]
    assert sorted(path.name for path in root.parent.iterdir()) == [".c1.writers", ".c1.writers.lock"]
    [run] = record.runs
    ended = Completed({"sacct": [(0, f"{run['slurm_job']}|COMPLETED\n", "")]})
    assert main([*argv, "--preflight", "none", "--json"], repo=repo, executor=executor, run=ended) == 0
    second = _json(capsys)["launch_id"]
    assert sorted(path.name for path in writers_dir(root).iterdir()) == sorted([f"{rival.rival}.json", f"{second}.json"])
    assert [claimed.launch_id for claimed in claimed_launches([root])] == sorted([rival.rival, second])


def test_a_launch_of_one_campaign_root_under_another_artifact_root_is_refused_by_the_claim_beside_the_root(
        repo, tmp_path, capsys, monkeypatch):
    _, executor = _executors()
    elsewhere = tmp_path / "elsewhere"
    root = elsewhere / "sampling" / "c1"
    root.mkdir(parents=True)
    (root / CAMPAIGN_MARKER).write_text("{}")
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--campaign-root", str(root), "--preflight", "none",
                 "--json"], repo=repo, executor=executor) == 0
    first = LaunchRecord.load(_json(capsys)["launch_id"])
    assert first.data["campaign"]["conditions"][0]["campaign_root"] == str(root)
    [run] = first.runs
    running = Completed({"sacct": [(0, f"{run['slurm_job']}|RUNNING\n", "")]})
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(elsewhere))
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--preflight", "none"]
    assert main(argv, repo=repo, executor=executor, run=running) == 2
    assert f"{first.launch_id} runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 running)" in capsys.readouterr().err
    assert not (elsewhere / ".ruleforge-launch").exists()
    assert main(["campaign", *SLURM, "--campaign-id", "c2", "--preflight", "none"], repo=repo, executor=executor,
                run=running) == 0
    shutil.rmtree(first.directory)
    assert main(argv, repo=repo, executor=executor, run=running) == 2
    assert (f"{first.launch_id} runs gemma-4-31B-it/naive-rules-above in sampling/c1 (1 submitting; its record is gone)"
            in capsys.readouterr().err)
    assert main([*argv, "--ignore-launch", first.launch_id, "--json"], repo=repo, executor=executor,
                run=running) == 0
    second = LaunchRecord.load(_json(capsys)["launch_id"])
    assert second.data["ignored_launches"] == [first.launch_id]
    assert second.data["campaign"]["conditions"][0]["campaign_root"] == "sampling/c1"
    assert [claimed.launch_id for claimed in claimed_launches([root])] == sorted([first.launch_id, second.launch_id])
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    [run] = second.runs
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--campaign-root", str(root), "--preflight", "none"],
                repo=repo, executor=executor, run=Completed({"sacct": [(0, f"{run['slurm_job']}|PENDING\n", "")]})) == 2
    assert f"{second.launch_id} runs gemma-4-31B-it/naive-rules-above in {root} (1 pending)" in capsys.readouterr().err


@requires_git_worktree
def test_jobs_launch_runs_the_campaign_verb(repo, monkeypatch, capsys):
    from ruleforge.cli import main as cli_main

    assert cli_main(["jobs", "launch", "campaign", "--backend", "slurm", "--gpu-count", "1", "--campaign-id", "c1",
                     "--allow-dirty", "--dry-run"]) == 0
    assert "python -m ruleforge.launch.programs.run_campaign campaign_id=c1 campaign_root=" in capsys.readouterr().out


def test_the_options_places_and_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    assert CampaignOptions.from_settings({"shards": 3, "blocks": ["a"], "backend": "slurm", "new_root": None}) == \
        CampaignOptions(shards=3, blocks=("a",))
    assert place("a/b", what="x") == Place(tmp_path / "a" / "b", "a/b")
    assert place(str(tmp_path / "c"), what="x").named == str(tmp_path / "c")
    outside = place("/elsewhere/d", what="x")
    assert outside == Place(Path("/elsewhere/d"), None) and outside.recorded == "/elsewhere/d"
    assert outside.named == "/elsewhere/d"
    assert selected_shards(None, None) == [0] and selected_shards(None, 3) == [0, 1, 2]
    assert selected_shards("2, 0,2", 3) == [0, 2]
    with pytest.raises(LaunchError, match="--shards 0: at least 1"):
        selected_shards(None, 0)


def test_a_condition_is_resolved_as_hydra_composes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    assert resolve_condition(REPO, ["--config-name", "sampling_ruleforge", "campaign_id=f",
                                    "ruleforge.ruleset=seed5_winner"]) == {
        "campaign_id": "f", "campaign_root": f"{tmp_path}/sampling/f",
        "condition_id": "gemma-4-31B-it/figure4/seed5_winner", "judge_model_type": "local", "batch_root": None}
    assert hydra_arguments(["-cn", "x", "a=1", "--config-dir=d"]) == ("x", ["a=1"], ["--config-dir"])


def test_an_export_that_chooses_the_judge_s_config_is_refused_where_it_changes_the_condition(tmp_path, monkeypatch):
    """``model@judge_model=${oc.env:REVIEW_MODEL}`` with ``--env REVIEW_MODEL=qwen3-8b``."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    arguments = ["campaign_id=probe", "model@judge_model=${oc.env:REVIEW_MODEL}"]
    monkeypatch.setenv("REVIEW_MODEL", "gemma4-31b-tp-2")
    assert resolve_condition(REPO, arguments)["condition_id"] == "gemma-4-31B-it/naive-rules-above"
    with pytest.raises(LaunchError, match=r"--env REVIEW_MODEL: a campaign job's config chooses its files by the "
                                          r"environment .* compose model/qwen3-8b at judge_model where the launch "
                                          r"composed, and checked the places of, model/gemma4-31b-tp-2 at "
                                          r"judge_model;"):
        resolve_condition(REPO, arguments, env=["REVIEW_MODEL=qwen3-8b"])
    monkeypatch.setenv("REVIEW_MODEL", "qwen3-8b")
    assert resolve_condition(REPO, arguments, env=["REVIEW_MODEL=qwen3-8b"])["condition_id"] == "Qwen3-8B/naive-rules-above"


def run_job(job: dict, modelspec, seen: set) -> None:
    """Run one rendered campaign job in-process."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from ruleforge.evaluate.campaign import build_plan, execution_settings, run_plan
    from tests.test_sampling_shards import SeededBackend

    config_name, overrides, _ = hydra_arguments(given(job))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name=config_name or "sampling_campaign", overrides=overrides)
    backend = SeededBackend(seen=seen)
    task_spec = modelspec(reasoning_prompt_style=str(cfg.eval_cfg.reasoning_prompt_style), evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    run_plan(plan, backend=backend, task_spec=task_spec, execution=execution_settings(cfg))


def test_two_conditions_run_as_one_shard_each_merge_into_the_unsharded_root(repo, tmp_path, capsys, monkeypatch,
                                                                            modelspec):
    from ruleforge.evaluate import aggregate as aggregate_module
    from ruleforge.evaluate import campaign as campaign_module
    from ruleforge.evaluate import runner as runner_module
    from ruleforge.evaluate import shard_merge as merge_module
    from ruleforge.evaluate.aggregate import analyze_campaign
    from ruleforge.evaluate.ledger import LEDGER_FNAME
    from tests.test_sampling_runner import RULES
    from tests.test_sampling_shards import NOW, STACK, assert_same_campaign

    for module in (runner_module, campaign_module, merge_module, aggregate_module):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    monkeypatch.setattr(runner_module, "_package_version", STACK.get)
    scenarios = "".join(json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(10))
    _, digest = population(tmp_path, "pops/scenarios.jsonl", scenarios)
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES))
    options = ["--gpu-count", "1", "--campaign-id", "camp", "--new-root", "--population", "pops/scenarios.jsonl",
               "--expect-population-sha256", digest, "--axis", f"eval_cfg.reasoning_prompt_style={','.join(STYLES)}"]
    arguments = ["--", f"rule_set_path={rules}", "sampling.campaign_mode=pilot", "sampling.num_decoder_draws=2",
                 "sampling.scenarios_per_request_block=3", "sampling.bootstrap.n_resamples=150"]
    # Unsharded, as SLURM runs it: each condition writes the root.
    plain = dry(repo, capsys, "--backend", "slurm", "--campaign-root", "plain", *options, *arguments)
    for job in plain["jobs"]:
        run_job(job, modelspec, set())
    _, executor = _executors()
    argv = ["campaign", "--backend", "slurm", "--campaign-root", "sharded", "--shards", "1", "--preflight", "none",
            *options, "--json", *arguments]
    assert main(argv, repo=repo, executor=executor) == 0
    launch_id = _json(capsys)["launch_id"]
    record = LaunchRecord.load(launch_id)
    assert [job["job_id"] for job in record.jobs] == ["campaign-0-shard-0-of-1", "campaign-1-shard-0-of-1"]
    for job in record.jobs:
        run_job(job, modelspec, set())
    states = "".join(f"{run['slurm_job']}|COMPLETED\n" for run in record.runs)
    assert main(["pull", launch_id, "--json"], repo=repo, run=Completed({"sacct": [(0, states, "")]})) == 0
    assert len(_json(capsys)["merged"]) == 2
    unsharded, merged = tmp_path / "artifacts" / "plain", tmp_path / "artifacts" / "sharded"
    statuses = {json.loads(line)["status"] for style in STYLES for draw in ("draw-000", "draw-001")
                for line in (unsharded / "gemma-4-31B-it" / style / draw / LEDGER_FNAME).read_text().splitlines()}
    assert {"rejected", "transport_failed", "accepted"} <= statuses, "the fake must exercise both retries"
    assert json.loads((merged / "completion.json").read_text())["n_conditions"] == 2
    assert_same_campaign(unsharded, merged)
    assert analyze_campaign(unsharded) == analyze_campaign(merged)


@requires_git_worktree  # the job writes code snapshots, which need git's file list
def test_the_places_check_roots_holds_are_where_a_campaign_job_writes(repo, tmp_path, capsys, monkeypatch, modelspec):
    from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME
    from ruleforge.evaluate.identity import COMPLETION_FNAME, SAMPLING_MANIFEST_FNAME
    from ruleforge.evaluate import runner as runner_module
    from ruleforge.core.provenance import SNAPSHOT_DIRNAME
    from tests.test_sampling_runner import RULES
    from tests.test_sampling_shards import STACK

    from ruleforge.evaluate.configured import MARKER_FNAME

    assert campaign.JOB_FILES == (SCENARIO_MANIFEST_FNAME, SAMPLING_MANIFEST_FNAME, COMPLETION_FNAME, MARKER_FNAME)
    monkeypatch.setattr(runner_module, "_package_version", STACK.get)
    scenarios = "".join(json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(6))
    _, digest = population(tmp_path, "pops/scenarios.jsonl", scenarios)
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES))
    options = ["--backend", "slurm", "--gpu-count", "1", "--campaign-id", "camp", "--new-root", "--population",
               "pops/scenarios.jsonl", "--expect-population-sha256", digest]
    arguments = ["--", f"rule_set_path={rules}", "sampling.campaign_mode=pilot", "sampling.num_decoder_draws=2",
                 "sampling.scenarios_per_request_block=3", "sampling.bootstrap.n_resamples=150"]
    artifacts = tmp_path / "artifacts"
    before = set(artifacts.iterdir())

    def held(path: Path, places: list[Path]) -> bool:
        return any(path == place or place in path.parents or place.is_relative_to(path) for place in places)

    [job] = dry(repo, capsys, "--campaign-root", "plain", *options, *arguments)["jobs"]
    run_job(job, modelspec, set())
    root = artifacts / "plain"
    places = [*campaign.job_destinations(root, CONDITION), root / "hydra" / "preflight" / CONDITION]
    written = sorted(root.rglob("*"))
    assert [path for path in written if not (path.parent == root and path.name in campaign.JOB_FILES)
            and not held(path, places)] == []
    assert sorted(path.name for path in root.iterdir() if path.is_file()) == sorted(campaign.JOB_FILES)
    assert (root / SNAPSHOT_DIRNAME).is_dir() and (root / CONDITION / "draw-001").is_dir()
    for job in dry(repo, capsys, "--campaign-root", "sharded", "--shards", "2", *options, *arguments)["jobs"]:
        run_job(job, modelspec, set())
    shards = [shard_root(artifacts / "sharded", CONDITION, index, 2) for index in range(2)]
    assert all((shard / SNAPSHOT_DIRNAME).is_dir() and (shard / CONDITION).is_dir() for shard in shards)
    assert [path for path in sorted((artifacts / "sharded.shards").rglob("*")) if not held(path, shards)] == []
    assert set(artifacts.iterdir()) - before == {root, artifacts / "sharded.shards"}


def swept(call: list[str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    """A multirun call's conditions (each sweep's values spread) and its launcher overrides."""
    from hydra.core.override_parser.overrides_parser import OverridesParser

    config_name, overrides, _ = hydra_arguments(call[call.index("-m") + 1:])
    task, launcher = ([[f"--config-name={config_name}"]] if config_name else []), {}
    for argument in overrides:
        override = OverridesParser.create().parse_override(argument)
        key = override.get_key_element()
        if key.startswith(("hydra", "~hydra", "+hydra")):
            launcher[key.lstrip("~+")] = argument.split("=", 1)[1] if "=" in argument else None
        elif override.is_sweep_override():
            task.append([f"{key}={value}" for value in override.sweep_string_iterator()])
        else:
            task.append([argument])
    return [dict(item.split("=", 1) for item in point) for point in itertools.product(*task)], launcher


def _flags(job: dict) -> dict[str, str]:
    arguments = given(job)
    config_name, overrides, _ = hydra_arguments(arguments)
    return {**({"--config-name": config_name} if config_name else {}),
            **dict(arg.split("=", 1) for arg in overrides if not arg.startswith("campaign_root="))}


def _last_json(text: str) -> dict:
    decoder, found, index = json.JSONDecoder(), None, text.find("{")
    while index != -1:
        try:
            found, end = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        index = text.find("{", end)
    return found


PAPER_SPECS = [("ruleforge_rulesets", "fig4-2026-09-pop-a22af097"), ("prompt_ablation_naive", "final-2026-09-pop-a22af097")]
PAPER_CONDITIONS = {"ruleforge_rulesets": 3, "prompt_ablation_naive": 3}
#: The fields the campaign's preflight prints that condition.json records, by their name there.
IDENTITY = {"condition_id": "condition_id", "judge_model": "model_name", "prompt_style": "prompt_style",
            "evaluation_mode": "evaluation_mode", "forced_json_requirement": "forced_json_requirement",
            "sampling_truncation": "sampling_truncation"}
#: The override a paper multirun sets the style by, which the records name by its recorded key.
STYLE_OVERRIDE = "eval_cfg.reasoning_prompt_style"
LABEL_RULE_OVERRIDE = "sampling.label_rule"
#: What they call a config block, and its name now.
RECORDED_BLOCKS = {recorded: public for public, recorded in RECORDED_BLOCK_KEYS.items()}


def _key_now(key: str) -> str:
    head, dot, rest = key.partition(".")
    return f"{RECORDED_BLOCKS.get(head, head)}{dot}{rest}"


@pytest.mark.cluster
@pytest.mark.parametrize(("spec", "paper_root"), PAPER_SPECS)
def test_a_paper_spec_composes_its_paper_root_s_conditions_and_never_extends_it(spec, paper_root, tmp_path,
                                                                                monkeypatch, capsys):
    from ruleforge.core.errors import SamplingDesignError
    from ruleforge.core.record_names import public_style
    from ruleforge.evaluate.campaign import (
        check_condition_id, check_registered_condition, read_condition, refuse_retired_root,
    )
    from ruleforge.evaluate.identity import LABEL_RULE
    from ruleforge.evaluate.runner import CampaignRunError
    from ruleforge.launch.guards import compose
    from tests.test_launch_specs import _as_recorded, _condition_spec

    real = os.environ.get("RULEFORGE_ARTIFACT_ROOT")
    root = Path(real or "/nonexistent") / "sampling" / paper_root
    if not root.is_dir():
        pytest.skip(f"no paper root {root}")
    published = root.with_name(f"{paper_root}-rescored-{LABEL_RULE}")
    rebase = json.loads((root / "rebase.json").read_text())
    start, target = rebase["chain"][0]["parent"], rebase["target"]
    campaign_id = start["campaign_id"]
    assert rebase["parent"]["campaign_id"] == campaign_id
    population = {start["scenario_source_sha256"]: target["source_sha256"]}
    monkeypatch.setenv("RULEFORGE_SLURM_PARTITION", os.environ.get("RULEFORGE_SLURM_PARTITION") or "any")
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", os.environ.get("RULEFORGE_MODEL_ROOT") or str(Path.home() / "models"))
    prompt = ("--var", "prompt=naive-rules-above") if spec == "ruleforge_rulesets" else ()
    printed = dry(REPO, capsys, "--spec", spec, "--var", f"campaign_id={campaign_id}", *prompt,
                  "--campaign-root", f"sampling/{paper_root}", "--allow-dirty")
    paper = {}
    for path in (Path(real) / start["root"].removeprefix("artifact://") / "hydra").glob("*/multirun.yaml"):
        multirun = yaml.safe_load(path.read_text())
        overrides, site = multirun["hydra"]["overrides"]["task"], multirun_site(multirun)
        for condition in swept(["-m", *overrides])[0]:
            condition = {_key_now(key): value for key, value in condition.items()}
            if "approved_scenario_sha256" in condition:
                condition["approved_scenario_sha256"] = population[condition["approved_scenario_sha256"]]
                condition["scenario_set_path"] = str(Path(real) / target["population"].removeprefix("artifact://"))
            condition[STYLE_OVERRIDE] = public_style(condition.get(STYLE_OVERRIDE, "legacy"))
            condition.pop(LABEL_RULE_OVERRIDE, None)
            for key in ("model@judge_model", "model@agent_model"):
                if key in condition:
                    condition[key] = model_config_now(condition[key], site)
            paper[json.dumps(sorted(condition.items()))] = path
    count = PAPER_CONDITIONS[spec]
    assert len(printed["campaign"]["conditions"]) == len(printed["jobs"]) == len(
        printed["campaign"]["preflights"]) == count
    rendered = []
    for job in printed["jobs"]:
        flags = _flags(job)
        flags.pop("--config-name", None)
        rendered.append(json.dumps(sorted(flags.items())))
    launched = {key for key in paper if dict(json.loads(key))[STYLE_OVERRIDE] == "naive-rules-above"}
    assert set(rendered) == launched and len(set(rendered)) == count, "each paper condition rendered once"
    # A paper root holding a retired style's arm refuses every preflight into it for that first.
    try:
        refuse_retired_root(root)
        retired = None
    except SamplingDesignError:
        retired = "registers a condition of a retired style"
    refused = 0
    for index, (job, condition, preflight) in enumerate(zip(printed["jobs"], printed["campaign"]["conditions"],
                                                            printed["campaign"]["preflights"])):
        composed = _condition_spec(compose(REPO, given(job), default_config=campaign.DEFAULT_CONFIG))
        assert composed.condition_id == condition["condition_id"] and composed.label_rule == LABEL_RULE
        shipped = read_condition(published / composed.identity_id / "condition.json")
        assert dataclasses.replace(composed, condition_id=shipped.condition_id) == shipped
        assert composed.fingerprint == shipped.fingerprint
        recorded_path = root / composed.identity_id / "condition.json"
        recorded, read = json.loads(recorded_path.read_text()), read_condition(recorded_path)
        as_run = _as_recorded(composed, recorded["label_rule"])
        assert as_run.fingerprint == recorded["fingerprint"] and composed.identity_id == recorded["condition_id"]
        assert dataclasses.replace(as_run, condition_id=read.condition_id) == read
        assert composed.prompt_style == public_style(recorded["prompt_style"])
        argv = [arg for arg in preflight if not arg.startswith("hydra.run.dir=")]
        environment = {**os.environ, "TIKTOKEN_RS_CACHE_DIR": os.environ.get("TIKTOKEN_RS_CACHE_DIR")
                       or str(Path.home() / ".cache" / "tiktoken")}
        result = subprocess.run([sys.executable, *argv[1:], f"hydra.run.dir={tmp_path}/{index}"],
                                cwd=REPO, capture_output=True, text=True, env=environment)
        refused += 1
        if composed.condition_id != composed.identity_id:
            assert result.returncode != 0 and (retired or "is a rebased root") in result.stderr, result.stderr[-3000:]
            with pytest.raises(CampaignRunError, match=r"is hashed over \(its identity id\): the two would share"):
                check_condition_id(root, composed)
            continue
        assert result.returncode != 0 and (
            retired or "describes a different campaign (label_rule differ)") in result.stderr, result.stderr[-3000:]
        with pytest.raises(CampaignRunError, match="already registers condition .* with a different identity"):
            check_registered_condition(root, composed)
        assert check_registered_condition(root, as_run) == recorded_path
    assert refused == count


def test_a_campaign_on_a_profile_records_it_and_asks_for_its_cluster(repo, tmp_path, capsys, monkeypatch):
    profile = tmp_path / "site.yaml"
    profile.write_text("partition: p\naccount: a\nqos: {gpu: g, cpu: c}\ngpus_per_node: 8\nexclude: [n1]\n"
                       "defaults: {timeout_min: 30, cpus: 2, mem_gb: 8, mem_gb_per_gpu: 16}\nsetup: []\n")
    printed = dry(repo, capsys, *SLURM, "--campaign-id", "c1", "--shards", "2", "--profile", str(profile))
    assert printed["profile"] == str(profile) and printed["campaign"]["preflight"] == "local"
    for job in printed["jobs"]:
        assert job["request"]["submitit"]["slurm_partition"] == "p"
        assert job["request"]["submitit"]["slurm_additional_parameters"] == {"account": "a", "qos": "g",
                                                                             "no_requeue": True, "exclude": "n1"}


@pytest.mark.parametrize("options", [
    ["--shards", "1", "--preflight", "none", "--wait", "--pull"],  # sharded, merged by the launch's own pull
    ["--shards", "2"],  # sharded, merged by a later pull
    ["--no-shards"],  # one job writing the root itself
    [],  # unsharded: no --shards
    ["--preflight-only"],  # nothing submitted, but the preflight makes its Hydra directory in the root
])
def test_a_launch_that_would_write_a_pinned_campaign_root_is_refused_before_it_spends(repo, tmp_path, capsys,
                                                                                       monkeypatch, options):
    from ruleforge.evaluate import api

    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/c1": ("leniency",)})
    argv = ["campaign", "--backend", "local", "--campaign-id", "c1", *options, "--", "model@judge_model=gpt-5.6-luna"]
    popen = FakePopen([(0, ["ran"])])
    assert main(argv, repo=repo, popen=popen) == 2
    err = capsys.readouterr().err
    root = tmp_path / "artifacts" / "sampling" / "c1"
    assert f"python -m ruleforge.launch.cli campaign: {root} (artifact://sampling/c1) is artifact://sampling/c1, or on its path" in err
    assert popen.calls == [] and list((tmp_path / "artifacts").iterdir()) == []
    assert main([*argv[:-2], "--dry-run", *argv[-2:]], repo=repo, popen=popen) == 0
    assert popen.calls == [] and list((tmp_path / "artifacts").iterdir()) == []


@pytest.mark.parametrize("planted, refusal", [
    ("claims", r"the campaign root's claims directory \S*/sampling/\.c1\.writers is a symlink"),
    ("claims lock", r"the campaign root's claims lock \S*/sampling/\.c1\.writers\.lock is a symlink"),
    ("shards", r"the shard root of gemma-4-31B-it/naive-rules-above \S*/sampling/c1\.shards is a symlink"),
])
def test_a_campaign_launch_whose_writes_beside_its_root_would_follow_a_symlink_is_refused_before_anything_runs(
        repo, tmp_path, capsys, planted, refusal):
    published = tmp_path / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    sampling = tmp_path / "artifacts" / "sampling"
    sampling.mkdir()
    if planted == "claims":
        (sampling / ".c1.writers").symlink_to(published, target_is_directory=True)
    elif planted == "claims lock":
        (sampling / ".c1.writers.lock").symlink_to(published / "x.lock")
    else:
        (sampling / "c1.shards").symlink_to(published, target_is_directory=True)
    log, executor = _executors()
    popen = FakePopen([(0, ["preflight ok"])])
    assert main(["campaign", *SLURM, "--campaign-id", "c1", "--shards", "2"], repo=repo, popen=popen,
                executor=executor) == 2
    assert re.search(refusal, capsys.readouterr().err)
    assert popen.calls == [] and log["jobs"] == [] and not records_root().exists()
    assert sorted(path.name for path in published.iterdir()) == ["sampling_manifest.json"]


def _digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        digest.update(str(path.relative_to(directory)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


#: The condition the campaign verb composes from the configs alone, and a Hydra run directory's name below it.
CONDITION, STAMP = "gemma-4-31B-it/naive-rules-above", "2026-01-01_00-00-00"


@pytest.mark.parametrize("planted, options, refusal", [
    # alias -> real, alias.shards -> a published stand-in: the jobs write <alias>.shards, the record names real
    ("alias", ["--campaign-root", "sampling/alias", "--shards", "2"],
     r"the campaign root of gemma-4-31B-it/naive-rules-above \S*/sampling/alias, as its jobs spell it, is not "
     r"\S*/sampling/real, where the launch's record and a pull name it: it is reached through a symlink"),
    ("alias", ["--campaign-root", "sampling/alias"], r"\S*/sampling/alias, as its jobs spell it, is not"),
    # a hydra symlink under a valid root: the unsharded job's and preflight's Hydra directories
    ("hydra", ["--campaign-root", "sampling/c1"],
     r"a directory gemma-4-31B-it/naive-rules-above's jobs write \S*/sampling/c1/hydra/preflight/gemma-4-31B-it/naive-rules-above: "
     r"\S*/sampling/c1/hydra, on its way, is a symlink"),
    # ... and below each Hydra directory, where Hydra makes <time>/.hydra/ at a name the launch cannot see beforehand
    ("below job", ["--campaign-root", "sampling/c1"],
     r"jobs write \S*/sampling/c1/hydra/gemma-4-31B-it/naive-rules-above: \S*/naive-rules-above/2026-01-01_00-00-00, below it, is a "
     r"symlink"),
    ("below preflight", ["--campaign-root", "sampling/c1"],
     r"jobs write \S*/sampling/c1/hydra/preflight/gemma-4-31B-it/naive-rules-above: \S*/2026-01-01_00-00-00/\.hydra, below it, "
     r"is a symlink"),
    ("snapshots", ["--campaign-root", "sampling/c1"],
     r"jobs write \S*/sampling/c1/code_snapshots: \S*/sampling/c1/code_snapshots, on its way, is a symlink"),
    ("below snapshots", ["--campaign-root", "sampling/c1"],
     r"jobs write \S*/sampling/c1/code_snapshots: \S*/code_snapshots/0123456789abcdef\.tar\.gz, below it, is a "
     r"symlink"),
    ("model", ["--campaign-root", "sampling/c1"],
     r"jobs write \S*/sampling/c1/gemma-4-31B-it/naive-rules-above: \S*/sampling/c1/gemma-4-31B-it, on its way, is a symlink"),
    ("below condition", ["--campaign-root", "sampling/c1"],
     r"jobs write \S*/sampling/c1/gemma-4-31B-it/naive-rules-above: \S*/naive-rules-above/draw-000, below it, is a symlink"),
    ("manifest", ["--campaign-root", "sampling/c1"],
     r"\S*/sampling/c1/scenario_manifest\.json, a file gemma-4-31B-it/naive-rules-above's job writes in its root, is a symlink"),
    ("below shard preflight", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"jobs write \S*/c1\.shards/gemma-4-31B-it/naive-rules-above/preflight: \S*/preflight/2026-01-01_00-00-00/\.hydra, below "
     r"it, is a symlink"),
    ("below shard hydra", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"jobs write \S*/c1\.shards/gemma-4-31B-it/naive-rules-above/shard-0-of-2: \S*/2026-01-01_00-00-00/\.hydra, below it, is "
     r"a symlink"),
    ("preflight", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"\S*/c1\.shards/gemma-4-31B-it/naive-rules-above/preflight, on its way, is a symlink"),
    ("shard hydra", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"jobs write \S*/naive-rules-above/shard-1-of-2: \S*/c1\.shards/gemma-4-31B-it/naive-rules-above/shard-1-of-2/hydra, below it, is a "
     r"symlink"),
    ("shard snapshots", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"jobs write \S*/naive-rules-above/shard-0-of-2: \S*/shard-0-of-2/code_snapshots, below it, is a symlink"),
    ("shard manifest", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"jobs write \S*/naive-rules-above/shard-1-of-2: \S*/shard-1-of-2/sampling_manifest\.json, below it, is a symlink"),
    ("shard", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"\S*/c1\.shards/gemma-4-31B-it/naive-rules-above/shard-0-of-2, on its way, is a symlink"),
    ("condition", ["--campaign-root", "sampling/c1", "--shards", "2"],
     r"\S*/c1\.shards/gemma-4-31B-it, on its way, is a symlink"),
])
def test_a_campaign_whose_jobs_would_write_through_a_symlink_is_refused_before_anything_runs(
        repo, tmp_path, capsys, planted, options, refusal):
    sampling = tmp_path / "artifacts" / "sampling"
    published = tmp_path / "published"
    (published / "inner").mkdir(parents=True)
    (published / "sampling_manifest.json").write_text("{}")
    root = sampling / ("real" if planted == "alias" else "c1")
    root.mkdir(parents=True)
    (root / CAMPAIGN_MARKER).write_text("{}")  # a root of a campaign: --campaign-root resumes it
    shards = sampling / "c1.shards" / CONDITION
    links = {"alias": [(sampling / "alias", root), (sampling / "alias.shards", published)],
             "hydra": [(root / "hydra", published)],
             "preflight": [(shards / "preflight", published)],
             "shard hydra": [(shards / "shard-1-of-2" / "hydra", published)],
             "shard snapshots": [(shards / "shard-0-of-2" / "code_snapshots", published)],
             "shard manifest": [(shards / "shard-1-of-2" / "sampling_manifest.json",
                                 published / "sampling_manifest.json")],
             "shard": [(shards / "shard-0-of-2", published)],
             "condition": [(sampling / "c1.shards" / "gemma-4-31B-it", published)],
             "below job": [(root / "hydra" / CONDITION / STAMP, published)],
             "below preflight": [(root / "hydra" / "preflight" / CONDITION / STAMP / ".hydra", published)],
             "snapshots": [(root / "code_snapshots", published)],
             "below snapshots": [(root / "code_snapshots" / "0123456789abcdef.tar.gz",
                                  published / "sampling_manifest.json")],
             "model": [(root / "gemma-4-31B-it", published)],
             "below condition": [(root / CONDITION / "draw-000", published)],
             "manifest": [(root / "scenario_manifest.json", published / "sampling_manifest.json")],
             "below shard preflight": [(shards / "preflight" / STAMP / ".hydra", published)],
             "below shard hydra": [(shards / "shard-0-of-2" / "hydra" / CONDITION / STAMP / ".hydra",
                                    published)]}[planted]
    for link, target in links:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=target.is_dir())
    before = _digest(published)
    log, executor = _executors()
    popen = FakePopen([(0, ["preflight ok"])])
    argv = ["campaign", *SLURM, "--campaign-id", "c1", *options]
    assert main(argv, repo=repo, popen=popen, executor=executor) == 2
    assert re.search(refusal, capsys.readouterr().err)
    assert popen.calls == [] and log["jobs"] == [] and not records_root().exists() and _digest(published) == before
    assert main([*argv, "--dry-run"], repo=repo) == 0


def jobs_launch(monkeypatch, argv: list[str], **given) -> int:
    """``ruleforge jobs launch ARGV`` with the launcher's backend handed ``given`` (the tests' fakes)."""
    from ruleforge.cli import app
    from ruleforge.launch import cli as launch_cli

    real = launch_cli.run_verb

    def run_verb(args, handler, *, program, repo=None, follow=launch_cli.PROGRAM, **_):
        return real(args, handler, program=program, follow=follow, **given)

    monkeypatch.setattr(launch_cli, "run_verb", run_verb)
    return app.main(["jobs", "launch", *argv])


@pytest.mark.parametrize("planted, options, refusal", [
    # a campaign made at <root>/hydra: a launch at <root> would write its Hydra directories into it
    ("hydra/sampling_manifest.json", [],
     r"\S*/sampling/c1/hydra/preflight/gemma-4-31B-it/naive-rules-above: \S*/sampling/c1/hydra, on its way, is another "
     r"campaign's root"),
    ("gemma-4-31B-it/.ruleforge-campaign", [],
     r"\S*/sampling/c1/gemma-4-31B-it/naive-rules-above: \S*/sampling/c1/gemma-4-31B-it, on its way, is another campaign's root"),
    (f"{CONDITION}/sampling_manifest.json", [],
     r"\S*/sampling/c1/gemma-4-31B-it/naive-rules-above: \S*/sampling/c1/gemma-4-31B-it/naive-rules-above, on its way, is another "
     r"campaign's root"),
    (f"{CONDITION}/draw-000/sampling_manifest.json", [],
     r"\S*/sampling/c1/gemma-4-31B-it/naive-rules-above: \S*/naive-rules-above/draw-000, below it, is another campaign's root"),
    ("code_snapshots/old/sampling_manifest.json", [],
     r"\S*/sampling/c1/code_snapshots: \S*/code_snapshots/old, below it, is another campaign's root"),
    (f"hydra/{CONDITION}/{STAMP}/sampling_manifest.json", [],
     r"\S*/hydra/gemma-4-31B-it/naive-rules-above: \S*/naive-rules-above/2026-01-01_00-00-00, below it, is another campaign's root"),
    ("../c1.shards/gemma-4-31B-it/sampling_manifest.json", ["--shards", "2"],
     r"\S*/c1\.shards/gemma-4-31B-it, on its way, is another campaign's root"),
    (f"../c1.shards/{CONDITION}/shard-0-of-2/hydra/sampling_manifest.json", ["--shards", "2"],
     r"jobs write \S*/naive-rules-above/shard-0-of-2: \S*/shard-0-of-2/hydra, below it, is another campaign's root"),
    (f"../c1.shards/{CONDITION}/preflight/sampling_manifest.json", ["--shards", "2"],
     r"\S*/c1\.shards/gemma-4-31B-it/naive-rules-above/preflight, on its way, is another campaign's root"),
])
def test_a_campaign_whose_jobs_would_write_into_another_campaign_s_root_is_refused_before_anything_runs(
        repo, tmp_path, capsys, monkeypatch, planted, options, refusal):
    root = tmp_path / "artifacts" / "sampling" / "c1"
    root.mkdir(parents=True)
    (root / CAMPAIGN_MARKER).write_text("{}")  # a root of a campaign: --campaign-root resumes it
    marker = (root / planted).resolve()
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    before = _digest(marker.parent)
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--campaign-root", "sampling/c1", *options]
    for entry in ("launch.py", "ruleforge"):
        log, executor = _executors()
        popen = FakePopen([(0, ["preflight ok"])])
        given = {"repo": repo, "popen": popen, "executor": executor}
        assert (main(argv, **given) if entry == "launch.py" else jobs_launch(monkeypatch, argv, **given)) == 2
        assert re.search(refusal, capsys.readouterr().err)
        assert popen.calls == [] and log["jobs"] == [] and not records_root().exists()
    assert _digest(marker.parent) == before


def test_a_sharded_campaign_s_own_shard_roots_are_no_other_campaign_s(repo, tmp_path, capsys):
    root = tmp_path / "artifacts" / "sampling" / "c1"
    root.mkdir(parents=True)
    (root / CAMPAIGN_MARKER).write_text("{}")
    for index in range(2):
        shard = shard_root(root, CONDITION, index, 2)
        shard.mkdir(parents=True)
        (shard / CAMPAIGN_MARKER).write_text("{}")
    log, executor = _executors()
    popen = FakePopen([(0, ["preflight ok"])])
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--campaign-root", "sampling/c1", "--shards", "2", "--json"]
    assert main(argv, repo=repo, popen=popen, executor=executor) == 0, capsys.readouterr().err
    assert len(log["jobs"]) == 2


@pytest.mark.parametrize("options", [[], ["--shards", "2"]])
def test_a_campaign_whose_log_file_hydra_would_name_outside_its_run_directory_is_refused_before_its_preflight(
        repo, tmp_path, capsys, options):
    """Hydra's default log handler writes ``${hydra.runtime.output_dir}/${hydra.job.name}.log``."""
    published = tmp_path / "published"
    published.mkdir()
    (published / "sampling_manifest.json").write_text("{}")
    before = _digest(published)
    log, executor = _executors()
    popen = FakePopen([(0, ["preflight ok"])])
    escape = "../" * 12 + str(published).lstrip("/") + "/provenance"
    argv = ["campaign", *SLURM, "--campaign-id", "c1", "--new-root", *options, "--", f"hydra.job.name={escape}"]
    assert main(argv, repo=repo, popen=popen, executor=executor) == 2
    assert "gives Hydra's 'hydra.job.name' a value" in capsys.readouterr().err
    assert popen.calls == [] and log["jobs"] == [] and not records_root().exists() and _digest(published) == before


def test_a_condition_planned_without_its_jobs_spelling_is_checked_as_its_record_names_it(repo, tmp_path):
    template = campaign.JobSpec("t", "campaign", 0, ("python", campaign.PROGRAM[-1]), campaign.PREFLIGHT_RESOURCES)
    campaign.check_roots([campaign.Condition(template, "c1", "sampling/c1", "m/x")])
    (tmp_path / "artifacts" / "sampling" / "c1").mkdir(parents=True)
    (tmp_path / "artifacts" / "sampling" / "c1" / "hydra").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(LaunchError, match="c1/hydra, on its way, is a symlink"):
        campaign.check_roots([campaign.Condition(template, "c1", "sampling/c1", "m/x")])


def test_a_launch_s_batch_roots_meet_the_batch_root_rule_by_their_own_condition_s_root(repo, tmp_path):
    template = campaign.JobSpec("t", "campaign", 0, ("python", campaign.PROGRAM[-1]), campaign.PREFLIGHT_RESOURCES)
    first, second = (campaign.Condition(template, name, f"sampling/{name}", "m/x") for name in ("c1", "c2"))
    shared = campaign.Condition(template, "c1", "sampling/c1", "m/y")
    roots = tmp_path / "artifacts" / "sampling"
    what = "the batch_root"
    campaign.check_batch_root(str(roots / "c1" / "batch"), first, [first, shared, second], what=what)
    campaign.check_batch_root(str(roots / "c1" / "batch" / "x"), shared, [first, shared, second], what=what)
    with pytest.raises(LaunchError, match=r"the batch_root \S*/sampling/c1/batch is, lies in or holds \S*/sampling/c1, "
                                          r"m/x's campaign root"):
        campaign.check_batch_root(str(roots / "c1" / "batch"), second, [first, second], what=what)
    from ruleforge.evaluate.api import CampaignError

    for inside in (roots / "c1", roots / "c1" / "m"):
        with pytest.raises(CampaignError, match="outside its batch/"):
            campaign.check_batch_root(str(inside), first, [first, second], what=what)
    nested = campaign.Condition(template, "n", "sampling/c1/batch/n", "m/n")
    for transport in (roots / "c1" / "batch" / "n" / "transport", roots / "c1" / "batch" / "n.shards" / "t"):
        with pytest.raises(LaunchError, match=r"lies in or holds \S*/sampling/c1/batch/n(\.shards)?, m/n's "
                                              "campaign root or shard root"):
            campaign.check_batch_root(str(transport), first, [first, nested], what=what)
    assert not roots.exists()


def test_a_batch_root_that_names_two_places_is_refused_before_a_check_takes_one(repo, tmp_path):
    from ruleforge.evaluate.api import CampaignError

    template = campaign.JobSpec("t", "campaign", 0, ("python", campaign.PROGRAM[-1]), campaign.PREFLIGHT_RESOURCES)
    first = campaign.Condition(template, "c1", "sampling/c1", "m/x")
    safe, target = tmp_path / "safe", tmp_path / "elsewhere" / "child"
    target.mkdir(parents=True)
    safe.mkdir()
    (safe / "link").symlink_to(target, target_is_directory=True)
    (safe / "batch").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(CampaignError, match=rf"the batch_root {re.escape(str(safe))}/link/\.\./batch names two "
                                            rf"places: {re.escape(str(safe / 'batch'))} as written out"):
        campaign.check_batch_root(f"{safe}/link/../batch", first, [first], what="the batch_root")


def test_a_launch_s_campaign_roots_may_not_nest_before_their_markers_exist(repo, tmp_path):
    template = campaign.JobSpec("t", "campaign", 0, ("python", campaign.PROGRAM[-1]), campaign.PREFLIGHT_RESOURCES)

    def condition(cid: str, root: str) -> campaign.Condition:
        return campaign.Condition(template, cid, root, f"m/{cid}")

    campaign.check_roots([condition("a", "sampling/a"), condition("b", "sampling/b"), condition("a2", "sampling/a")])
    for outer, inner in (("sampling/a", "sampling/a/batch/b"), ("sampling/a", "sampling/a.shards/x"),
                         ("sampling/a/b", "sampling/a")):
        with pytest.raises(LaunchError, match=r"the campaign root or shard root of m/(a|b), is or lies in \S+, the "
                                              r"campaign root or shard root of m/(a|b) in this launch"):
            campaign.check_roots([condition("a", outer), condition("b", inner)])
    artifacts = tmp_path / "artifacts"
    (artifacts / "real").mkdir(parents=True)
    (artifacts / "alias").symlink_to(artifacts / "real", target_is_directory=True)
    with pytest.raises(LaunchError, match=r"real/sub, the campaign root or shard root of m/b, is or lies in \S+/real, "
                                          "the campaign root or shard root of m/a"):
        campaign.check_roots([condition("a", "real"), condition("b", "alias/sub")])
    assert not any((artifacts / "real").iterdir())


def test_a_merge_that_refuses_fails_a_wait_that_succeeded(repo, capsys, monkeypatch):
    def refuses(root, condition_id, *, num_shards=None):
        raise ShardError("lacks shard(s) [0] of 1")

    monkeypatch.setattr("ruleforge.evaluate.shard_merge.merge_condition_shards", refuses)
    argv = ["campaign", "--backend", "local", "--campaign-id", "c1", "--shards", "1", "--preflight", "none",
            "--wait", "--pull", "--", "model@judge_model=gpt-5.6-luna"]
    assert main(argv, repo=repo, popen=FakePopen([(0, ["ran"])])) == 1
    assert "refused: ShardError: lacks shard(s) [0] of 1" in capsys.readouterr().out


def test_a_recorded_model_config_name_drops_only_its_own_record_s_site():
    """A recorded name is today's committed config when its one added suffix is the partition its record names; a
    precision, a typo or another record's site leaves it as recorded, unequal to today's name."""
    configs = {"gemma4-31b-tp-2", "qwen3-8b", "qwen3-8b-fp32"}
    assert [model_config_now(name, "somesite", configs) for name in (
        "gemma4-31b-tp-2-somesite", "gemma4-31b-tp-2", "qwen3-8b-fp32", "qwen3-8b-fp32-somesite")] == [
        "gemma4-31b-tp-2", "gemma4-31b-tp-2", "qwen3-8b-fp32", "qwen3-8b-fp32"]
    for wrong, site in (("gemma4-31b-tp-2-fp32", "somesite"), ("qwen3-8b-fp64", "somesite"),
                        ("gemma4-31b-tp-2-typo", "somesite"), ("qwen3-8b-othersite", "somesite"),
                        ("qwen3-8b-somesite", None), ("unknown-somesite", "somesite")):
        assert model_config_now(wrong, site, configs) == wrong, (wrong, site)
    assert model_words_now(["model@judge_model=qwen3-8b-somesite", "+model@agent_model=qwen3-8b-fp64", "run_name=a-b"],
                           "somesite", configs) == [
        "model@judge_model=qwen3-8b", "+model@agent_model=qwen3-8b-fp64", "run_name=a-b"]
    assert model_config_now("gemma4-31b-tp-2-somesite", "somesite") == "gemma4-31b-tp-2"  # the checkout's own configs
    job = {"request": {"submitit": {"slurm_partition": "somesite"}}}
    assert (launch_site(job), launch_site({"request": {}}), launch_site({})) == ("somesite", None, None)
    assert (multirun_site({"hydra": {"launcher": {"partition": "somesite"}}}), multirun_site({"hydra": {}})) == (
        "somesite", None)
