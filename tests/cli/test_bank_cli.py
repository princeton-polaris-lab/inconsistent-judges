from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import ruleforge.evaluate.bank as bank
from ruleforge.cli import records
from ruleforge.cli.common import FAKE_NOTE
from ruleforge.cli.judging import judge_node
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.runner import CampaignRunError
from tests.conftest import requires_git_worktree
from tests.support.equivalence import read, tree
from tests.test_sampling_ruleset_size import write_subsets
from tests.test_sampling_runner import write_inputs

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "sampling_ruleset_size.yaml"
MARKER = (".ruleforge-campaign",)
CONFIGURED_MARKER = ".ruleforge-configured"


@pytest.fixture
def design(tmp_path, monkeypatch):
    """A four-rule ruleset's subsets, three scenarios, the fake judge's config and the overrides both runs take."""
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    base = tmp_path / "design"
    base.mkdir()
    rule_path, subset_root, manifest_dir = write_subsets(base)
    scenarios = write_inputs(base)
    (base / "model").mkdir()
    (base / "model" / "fake.yaml").write_text(OmegaConf.to_yaml(judge_node("fake", cwd=base)), encoding="utf-8")
    return base, [
        "campaign_id=ruleset-size", f"scenario_set_path={scenarios}", "approved_scenario_sha256=null",
        f"rule_set_path={rule_path}", "sampling.campaign_mode=pilot", "sampling.num_decoder_draws=2",
        "sampling.scenarios_per_request_block=null", "sampling.bootstrap.n_resamples=150",
        f"ruleset_size.subset_root={subset_root}", f"ruleset_size.subset_manifest_dir={manifest_dir}",
        f"ruleset_size.approved_subset_manifest_sha256={sha256_file(manifest_dir / 'manifest.sha256')}",
        "ruleset_size.subset_seeds=[0,1]", "ruleset_size.subset_sizes=[2,3,4]"]


def compose_driver(*overrides: str):
    """The config Hydra composes for ``overrides`` from the checkout's ``configs/sampling_ruleset_size.yaml``."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        return compose(config_name="sampling_ruleset_size", overrides=list(overrides))


def script(base: Path, overrides, root: Path, *extra: str) -> subprocess.CompletedProcess:
    """The ruleset-size campaign run into ``root`` in this process, its plan printed; a refusal is raised."""
    cfg = compose_driver(f"hydra.searchpath=[file://{base}]", "model@judge_model=fake", f"campaign_root={root}",
                         *overrides, *extra)
    design, planned, report = bank.plan_job(cfg)
    if not cfg.ruleset_size.get("preflight_only", False):
        assert not bank.run_planned(cfg, design, planned, report)
    return subprocess.CompletedProcess([], 0, stdout=json.dumps(report, indent=2) + "\n", stderr="")


def analysed(root: Path, **options) -> str:
    """The analysis printed for ``root`` (and written into it): each model's curve, then the campaign's summary."""
    payload = bank.analyze_shared_bank_campaign(root, **options)
    lines = []
    for model, entry in payload["models"].items():
        first, last, rise = entry["points"][0], entry["points"][-1], entry["contrasts"]["net_rise"]
        lines.append(f"{model}: K={first['size']} {first['estimate']:.3f} [{first['ci_low']:.3f}, {first['ci_high']:.3f}]"
                     f" -> K={last['size']} {last['estimate']:.3f} [{last['ci_low']:.3f}, {last['ci_high']:.3f}];"
                     f" net rise {rise['estimate']:+.3f} [{rise['ci_low']:+.3f}, {rise['ci_high']:+.3f}];"
                     f" {entry['adjacent_rises']} of {len(entry['points']) - 1} adjacent steps rise")
    keys = ("campaign_id", "state", "n_scenarios", "n_decoder_draws", "n_subset_seeds")
    return "\n".join([*lines, json.dumps({key: payload[key] for key in keys})]) + "\n"


def report_of(stdout: str) -> dict:
    """The plan a job prints first (the script's log lines may follow it)."""
    return json.JSONDecoder().raw_decode(stdout[stdout.index("{"):])[0]


def test_bank_run_judges_the_job_the_script_judges(cli, design, tmp_path):
    base, overrides = design
    scripted = script(base, overrides, tmp_path / "script-root")
    status, out, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                         *overrides)
    assert status == 0 and out == report_of(scripted.stdout)
    assert out["conditions_this_job"] == 7 and out["runs_bank"] is True
    assert tree(tmp_path / "root", leave_out=MARKER) == tree(tmp_path / "script-root",
                                                            leave_out=("hydra", CONFIGURED_MARKER))
    assert not (tmp_path / "root" / CONFIGURED_MARKER).exists()
    assert (tmp_path / "root" / ".ruleforge-campaign").is_file()
    assert next(records.runs_dir(tmp_path / "root").iterdir()).name.endswith("-bank-run")
    before = tree(tmp_path / "root")
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root",
                       "--yes-spend", "--", *overrides)
    assert status == 0 and tree(tmp_path / "root") == before


@pytest.mark.parametrize("change", ["+judge_model.max_new_tokens=77", "eval_cfg.reasoning_prompt_style=forced-grpo"])
def test_a_job_run_again_under_another_judge_configuration_is_refused_not_skipped(cli, design, tmp_path, change):
    base, overrides = design
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                       *overrides)
    assert status == 0
    before = tree(tmp_path / "root")
    status, out, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                             *overrides, change)
    assert status == 1 and "already registers condition" in error and "different identity" in error
    assert tree(tmp_path / "root") == before


def _locked(root: Path) -> bool:
    import fcntl
    import os

    from ruleforge.evaluate.api import LOCK_FNAME

    descriptor = os.open(root / LOCK_FNAME, os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)
    return False


def test_a_job_registers_its_conditions_and_leaves_its_gate_under_the_root_s_lock(cli, design, tmp_path,
                                                                                 monkeypatch):
    import ruleforge.evaluate.campaign as campaign_module

    root = tmp_path / "root"
    gates, registrations = [], []
    completion, register = campaign_module.campaign_completion, campaign_module.register_condition

    def gate(*args, **kwargs):
        gates.append(_locked(root))
        return completion(*args, **kwargs)

    def registered(*args, **kwargs):
        registrations.append(_locked(root))
        return register(*args, **kwargs)

    monkeypatch.setattr(campaign_module, "campaign_completion", gate)
    monkeypatch.setattr(campaign_module, "register_condition", registered)
    base, overrides = design
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", root, "--", *overrides)
    # One gate, the job's last write: no condition's run writes one of its own.
    assert status == 0 and len(registrations) == 7 and all(registrations) and gates == [True]
    assert json.loads((root / "completion.json").read_text(encoding="utf-8"))["state"] == "complete"


def _files(root: Path, leave_out: tuple[str, ...] = ()) -> dict[str, bytes]:
    return {name: path.read_bytes() for path in sorted(root.rglob("*"))
            if path.is_file() and not (name := path.relative_to(root).as_posix()).startswith(leave_out)}


@pytest.mark.parametrize(("change", "field"), [("+judge_model.max_new_tokens=77", "max_new_tokens"),
                                               ("eval_cfg.reasoning_prompt_style=forced-grpo", "prompt_style")])
def test_a_seed_share_under_another_judge_configuration_is_refused_at_registration(cli, design, tmp_path, change,
                                                                                  field):
    base, overrides = design
    root = tmp_path / "root"
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", root, "--", *overrides,
                       "ruleset_size.jobs_seeds=[0]")
    assert status == 0
    before = _files(root)
    status, out, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", root, "--", *overrides,
                             "ruleset_size.jobs_seeds=[1]", change)
    assert status == 1 and "more than one judge configuration" in error and f"({field}: " in error
    assert _files(root) == before
    scripted = tmp_path / "script-root"
    script(base, overrides, scripted, "ruleset_size.jobs_seeds=[0]")
    before = _files(scripted, ("hydra",))
    with pytest.raises(CampaignRunError) as refused:
        script(base, overrides, scripted, "ruleset_size.jobs_seeds=[1]", change)
    assert "more than one judge configuration" in str(refused.value) and f"({field}: " in str(refused.value)
    assert _files(scripted, ("hydra",)) == before


def test_bank_analyze_refuses_a_root_whose_model_holds_two_judge_configurations(cli, design, tmp_path):
    """A registry mixed by hand."""
    from ruleforge.evaluate.identity import ConditionSpec

    base, overrides = design
    script(base, overrides, tmp_path / "script-root")
    path = tmp_path / "script-root" / "fake-1" / "subsets" / "s01" / "k02" / "condition.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompt_style"] = "forced-grpo"
    payload["fingerprint"] = ConditionSpec.from_dict(
        {key: value for key, value in payload.items() if key not in ("schema_version", "fingerprint")}).fingerprint
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = _files(tmp_path / "script-root")
    status, _, error = cli("bank", "analyze", tmp_path / "script-root", "--out", tmp_path / "analysis.json")
    assert status == 1 and "more than one judge configuration (prompt_style: " in error
    assert _files(tmp_path / "script-root") == before and not (tmp_path / "analysis.json").exists()


@pytest.mark.parametrize("packed", [False, True], ids=["sequential", "packed"])
@pytest.mark.parametrize("fails", [False, True], ids=["completes", "a-condition-raises"])
def test_every_write_to_a_file_the_root_s_jobs_share_is_made_under_the_root_s_lock(design, tmp_path, monkeypatch,
                                                                                    packed, fails):
    import os

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import ruleforge.evaluate.campaign as campaign_module
    import ruleforge.judges.backend as backend_module
    from ruleforge.evaluate.api import Campaign
    from ruleforge.evaluate.identity import SamplingConfig
    from ruleforge.evaluate.stack import CONDITION_STACK_FNAME
    from ruleforge.scenarios import ScenarioSet
    from tests.test_sampling_ruleset_size import SCRIPT
    from tests.test_sampling_runner import FakeBackend

    def fake(model_cfg):
        backend = FakeBackend(style="naive", script=SCRIPT)
        backend.max_new_tokens, backend.llm = 1024, None
        return backend

    monkeypatch.setattr(backend_module, "ModelBackend", fake)
    base, overrides = design
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="sampling_ruleset_size", overrides=[
            f"campaign_root={tmp_path / 'root'}", f"execution.pack_draws={str(packed).lower()}", *overrides])
    design_, planned, report = bank.plan_job(cfg)
    campaign = Campaign.create(tmp_path / "root", campaign_id="ruleset-size", scenarios=ScenarioSet.load(cfg.scenario_set_path),
                               sampling=SamplingConfig.from_mapping(OmegaConf.to_container(cfg.sampling, resolve=True)))
    root = campaign.root
    writes, real = [], os.replace

    def replace(source, destination, *args, **kwargs):
        target = Path(os.path.realpath(os.path.dirname(destination))) / os.path.basename(destination)
        if target.is_relative_to(root):
            relative = target.relative_to(root)
            if not any(part.startswith("draw-") for part in relative.parts):
                writes.append((relative.as_posix(), _locked(root)))
        return real(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    if fails:
        name = "run_plans_packed" if packed else "run_plan"
        original, calls = getattr(campaign_module, name), []

        def raising(*args, **kwargs):
            calls.append(True)
            if packed or len(calls) == 2:
                original(*args, **kwargs)
                raise RuntimeError("a condition raised")
            return original(*args, **kwargs)

        monkeypatch.setattr(campaign_module, name, raising)
        with pytest.raises(RuntimeError, match="a condition raised"):
            bank.run_job(campaign, cfg, design_, planned, report)
    else:
        assert bank.run_job(campaign, cfg, design_, planned, report) == {}
    unlocked = [name for name, locked in writes if not locked]
    names = [name for name, _ in writes]
    assert writes and unlocked == []
    assert names.count("completion.json") == 1 and names[-1] == "completion.json"
    bank_id = planned[0].condition_id
    assert {"figure2_design.json", f"{bank_id}/condition.json", f"{bank_id}/{CONDITION_STACK_FNAME}"} <= set(names)
    gate = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    assert gate["state"] == "complete" and gate["n_conditions"] == (2 if fails and not packed else 7)


def test_preflight_only_prints_the_plan_and_writes_nothing(cli, design, tmp_path):
    base, overrides = design
    scripted = script(base, overrides, tmp_path / "script-root", "ruleset_size.preflight_only=true")
    status, out, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root",
                         "--campaign-id", "ruleset-size", "--preflight-only", "--", *overrides[1:])
    assert status == 0 and out == report_of(scripted.stdout)
    assert not (tmp_path / "root").exists() and not records.runs_dir(tmp_path / "root").exists()


def test_a_root_this_library_did_not_create_is_refused_before_anything_is_written(cli, design, tmp_path):
    base, overrides = design
    script(base, overrides, tmp_path / "script-root")
    before = tree(tmp_path / "script-root")
    for extra in ((), ("--preflight-only",)):
        status, out, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out",
                                 tmp_path / "script-root", *extra, "--", *overrides)
        assert status == 2 and "a campaign root this library did not create" in error and out == ""
    status, _, error = cli("bank", "analyze", tmp_path / "script-root")
    assert status == 2 and "holds no .ruleforge-campaign" in error and "--out FILE" in error
    assert tree(tmp_path / "script-root") == before and not records.runs_dir(tmp_path / "script-root").exists()


def test_bank_analyze_out_analyses_a_root_a_script_made_and_writes_nothing_into_it(cli, design, tmp_path):
    base, overrides = design
    script(base, overrides, tmp_path / "script-root")
    copy = tmp_path / "copy"
    shutil.copytree(tmp_path / "script-root", copy)
    printed_by_script = analysed(copy, n_resamples=120)
    before = tree(tmp_path / "script-root")
    out = tmp_path / "analysis" / "figure2_estimates.json"
    status, printed, _ = cli("bank", "analyze", tmp_path / "script-root", "--out", out, "--n-resamples", "120")
    assert status == 0 and printed == printed_by_script
    assert read(out) == read(copy / "aggregate" / bank.SHARED_BANK_ESTIMATES_FNAME)
    assert tree(tmp_path / "script-root") == before
    assert next(records.runs_dir(out).iterdir()).name.endswith("-bank-analyze")
    status, _, error = cli("bank", "analyze", tmp_path / "script-root", "--out", out)
    assert status == 1 and "AnalysisExists" in error
    status, _, _ = cli("bank", "analyze", tmp_path / "script-root", "--out", out, "--replace", "--seed", "3")
    assert status == 0 and json.loads(out.read_text(encoding="utf-8"))["bootstrap"]["seed"] == 3
    inside = tmp_path / "script-root" / "estimates.json"
    status, _, error = cli("bank", "analyze", tmp_path / "script-root", "--out", inside)
    assert status == 2 and "inside the campaign root" in error and tree(tmp_path / "script-root") == before
    with pytest.raises(bank.CampaignError, match="inside the campaign root"):
        bank.analyze_into(tmp_path / "script-root", inside)


def test_the_config_s_preflight_only_is_the_flag_s(cli, design, tmp_path):
    base, overrides = design
    status, out, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                         *overrides, "ruleset_size.preflight_only=true")
    assert status == 0 and out["conditions_this_job"] == 7
    assert not (tmp_path / "root").exists() and not records.runs_dir(tmp_path / "root").exists()


def test_the_checkout_s_config_is_read_as_its_jobs_read_it(design, tmp_path):
    from ruleforge.cli.app import build_parser
    from ruleforge.cli.evaluate import bank_config

    args = build_parser().parse_args(["bank", "run", "--config", str(CONFIG), "--judge", "fake", "--out",
                                      str(tmp_path / "root"), "--campaign-id", "ruleset-size"])
    args.overrides = []
    cfg = bank_config(args, tmp_path)
    assert Path(cfg.rule_set_path) == REPO / "data/rules/modelspec/openai_rules_all_full.jsonl"
    assert Path(cfg.ruleset_size.subset_root) == REPO / "data/rules/ruleset_size_subsets"
    assert Path(cfg.ruleset_size.subset_manifest_dir) == REPO / "data" / "rules" / "subset_manifest"
    assert Path(cfg.rule_set_path).is_file() and Path(cfg.campaign_root) == tmp_path / "root"


@pytest.mark.cluster
def test_the_checkout_s_config_plans_the_script_s_job(tmp_path, monkeypatch):
    if not (REPO / "data" / "rules" / "ruleset_size_subsets").is_dir():
        pytest.skip("needs the regenerated ruleset-size subsets (ruleforge reproduce ruleset-size subsets)")
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    population = REPO / "data" / "populations" / "frozen_500_r3.jsonl"
    common = ["model@agent_model=gemma4-31b-tp-2", "judge_model.tp_size=1",
              f"scenario_set_path={population}"]
    for share in ("[0]", "[5]"):
        with monkeypatch.context() as here:
            here.chdir(REPO)
            scripted = bank.plan_job(compose_driver("campaign_id=fig2-2026-09",
                                                    f"campaign_root={tmp_path / 'script-root'}",
                                                    "model@judge_model=gemma4-31b-tp-2", *common,
                                                    f"ruleset_size.jobs_seeds={share}"))[2]
        command = subprocess.run([sys.executable, "-m", "ruleforge", "bank", "run", "--config", str(CONFIG), "--judge",
                                  "gemma4-31b-tp-2", "--out", str(tmp_path / "root"), "--campaign-id",
                                  "fig2-2026-09", "--preflight-only", "--", *common, f"ruleset_size.jobs_seeds={share}"],
                                 cwd=tmp_path, capture_output=True, text=True, check=True)
        assert report_of(command.stdout) == json.loads(json.dumps(scripted))
    assert not (tmp_path / "root").exists() and not (tmp_path / "script-root").exists()


def test_bank_analyze_writes_the_script_s_estimates_into_a_root_of_its_own(cli, design, tmp_path, capsys):
    base, overrides = design
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                       *overrides)
    copy = tmp_path / "copy"
    shutil.copytree(tmp_path / "root", copy)
    printed_by_script = analysed(copy, n_resamples=120)
    status, out, error = cli("bank", "analyze", tmp_path / "root", "--n-resamples", "120")
    assert status == 0 and out == printed_by_script and error.splitlines()[-1] == FAKE_NOTE
    assert "fake-1: K=2" in out and "net rise" in out
    estimates = Path("aggregate") / bank.SHARED_BANK_ESTIMATES_FNAME
    assert tree(tmp_path / "root", leave_out=MARKER) == tree(copy, leave_out=MARKER)
    assert (tmp_path / "root" / estimates).is_file()
    status, _, error = cli("bank", "analyze", tmp_path / "root")
    assert status == 1 and "AnalysisExists" in error
    status, _, _ = cli("bank", "analyze", tmp_path / "root", "--replace", "--seed", "3", "--ci-level", "0.9",
                       "--allow-stack-mismatch")
    payload = json.loads((tmp_path / "root" / estimates).read_text(encoding="utf-8"))
    assert status == 0 and payload["bootstrap"]["seed"] == 3 and payload["bootstrap"]["ci_level"] == 0.9


@requires_git_worktree  # the run writes code snapshots, which need git's file list
def test_a_job_s_writes_are_checked_as_a_campaign_s_run_checks_them(cli, design, tmp_path, monkeypatch):
    base, overrides = design
    original = bank.run_conditions
    monkeypatch.setattr(bank, "run_conditions", lambda *args, **kwargs: {})
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                       *overrides)
    assert status == 0 and (tmp_path / "root" / ".ruleforge-campaign").is_file()
    monkeypatch.setattr(bank, "run_conditions", original)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "root" / "fake-1").symlink_to(elsewhere)
    status, _, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                           *overrides)
    assert status == 1 and "fake-1 is a symlink" in error and list(elsewhere.iterdir()) == []
    (tmp_path / "root" / "fake-1").unlink()
    snapshots = tmp_path / "root" / "code_snapshots"
    shutil.rmtree(snapshots)
    snapshots.symlink_to(elsewhere)
    status, _, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                           *overrides)
    assert status == 2 and "code_snapshots is a symlink" in error and list(elsewhere.iterdir()) == []


def test_a_job_s_guard_refuses_a_write_outside_its_root_and_a_root_not_the_job_s(tmp_path):
    from ruleforge.evaluate.api import Campaign
    from ruleforge.scenarios import ScenarioSet
    from tests.test_sampling_runner import write_inputs

    scenarios = write_inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=ScenarioSet.load(scenarios))
    guard = bank._job_write_guard(campaign, tmp_path / "root")
    guard(tmp_path / "root" / "m" / "bank")
    with pytest.raises(bank.CampaignError, match="outside the campaign root"):
        guard(tmp_path / "other")
    cfg = OmegaConf.create({"campaign_root": str(tmp_path / "another"), "judge_model": {}})
    with pytest.raises(bank.CampaignError, match="is not the campaign's root"):
        bank.run_job(campaign, cfg, None, [], {})
    foreign = tmp_path / "published"
    foreign.mkdir()
    (foreign / "sampling_manifest.json").write_text("{}", encoding="utf-8")
    cfg = OmegaConf.create({"campaign_root": str(tmp_path / "root"), "judge_model": {"batch_root": str(foreign)}})
    with pytest.raises(bank.CampaignError, match="a campaign root this library did not"):
        bank.run_job(campaign, cfg, None, [], {})


def test_a_batch_work_directory_refused_stops_the_job(tmp_path, monkeypatch):
    from ruleforge.evaluate.api import Campaign
    from ruleforge.scenarios import ScenarioSet
    from ruleforge.judges.batch_api import BatchDestinationRefused
    from tests.test_sampling_runner import write_inputs

    scenarios = write_inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=ScenarioSet.load(scenarios))
    monkeypatch.setattr(bank, "freeze_design", lambda campaign, design: None)

    def refused(*args, **kwargs):
        raise BatchDestinationRefused("a work directory is a symlink")

    monkeypatch.setattr(bank, "run_planned", refused)
    cfg = OmegaConf.create({"campaign_root": str(tmp_path / "root"), "judge_model": {}})
    with pytest.raises(bank.CampaignError, match="campaign 'c': a work directory is a symlink"):
        bank.run_job(campaign, cfg, None, [], {})


def test_a_condition_left_incomplete_fails_the_job(cli, design, tmp_path, monkeypatch):
    base, overrides = design
    monkeypatch.setattr(bank, "run_conditions", lambda *args, **kwargs: {"fake-1/bank": {0: "incomplete"}})
    status, _, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                           *overrides)
    assert status == 1 and "1 condition(s) did not complete" in error and "Nothing is aggregated" in error


def test_a_local_engine_is_released_after_the_job(design, tmp_path, monkeypatch):
    from hydra import compose
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import ruleforge.judges.backend as backend_module
    import ruleforge.judges.models as models_module

    released = []

    class Engine:
        llm = "an engine"
        close = backend_module.ModelBackend.close  # the release run_planned calls: cleanup_vllm, never raising

        def __init__(self, cfg):
            pass

    monkeypatch.setattr(backend_module, "ModelBackend", Engine)
    monkeypatch.setattr(models_module, "cleanup_vllm", lambda llm, raise_on_error: released.append(llm))
    import ruleforge.evaluate.campaign as campaign_module

    monkeypatch.setattr(bank, "run_conditions", lambda *args, **kwargs: {})
    monkeypatch.setattr(campaign_module, "campaign_completion",
                        lambda root: {"campaign_id": "ruleset-size", "state": "incomplete", "n_conditions": 0})
    base, overrides = design
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="sampling_ruleset_size", overrides=[
            f"hydra.searchpath=[file://{base}]", "model@judge_model=fake", f"campaign_root={tmp_path / 'root'}",
            *overrides])
    design_, planned, report = bank.plan_job(cfg)
    assert bank.run_planned(cfg, design_, planned, report) == {} and released == ["an engine"]
    assert bank.incomplete_error({"c": {0: "x"}}).startswith("1 condition(s) did not complete")


def test_a_job_that_fails_raises_its_own_failure_when_its_gate_cannot_be_published(design, tmp_path, monkeypatch,
                                                                                    caplog):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import ruleforge.judges.backend as backend_module

    class Engine:
        llm = None
        close = backend_module.ModelBackend.close

        def __init__(self, cfg):
            pass

    def failing(cfg, planned, *, writing, **kwargs):
        with writing():
            pass
        raise RuntimeError("a condition raised")

    def refused(root, writing):
        raise bank.CampaignError("the root is no longer the campaign's")

    monkeypatch.setattr(backend_module, "ModelBackend", Engine)
    monkeypatch.setattr(bank, "run_conditions", failing)
    monkeypatch.setattr(bank, "publish_gate", refused)
    base, overrides = design
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="sampling_ruleset_size", overrides=[
            f"hydra.searchpath=[file://{base}]", "model@judge_model=fake", f"campaign_root={tmp_path / 'root'}",
            *overrides])
    design_, planned, report = bank.plan_job(cfg)
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="a condition raised"):
        bank.run_planned(cfg, design_, planned, report)
    assert "was not published after the job failed: the root is no longer the campaign's" in caplog.text


def test_preflight_only_takes_the_config_s_own_judge(cli, design, tmp_path):
    """No ``--judge``: the config's model config judges (a local engine, whose weights a plan does not read)."""
    base, overrides = design
    status, out, _ = cli("bank", "run", "--config", CONFIG, "--out", tmp_path / "root", "--preflight-only", "--",
                         *overrides)
    assert status == 0 and out["judge_model"] == "gemma-4-31B-it" and not (tmp_path / "root").exists()


def test_a_condition_out_of_attempts_is_reported_and_the_rest_still_judged(design, tmp_path, modelspec):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from tests.test_sampling_runner import FakeBackend

    base, overrides = design
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose(config_name="sampling_ruleset_size", overrides=[
            f"campaign_root={tmp_path / 'root'}", "sampling.max_semantic_attempts=1", *overrides])
    design_, planned, report = bank.plan_job(cfg)
    (tmp_path / "root").mkdir()
    bank.write_design(tmp_path / "root", design_)
    backend = FakeBackend(style="naive", script={(0, "full_ruleset"): ["not a label"]})
    backend.max_new_tokens = 1024
    incomplete = bank.run_conditions(cfg, planned, identity=bank.job_identity(cfg, report), backend=backend,
                                     task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0))
    assert sorted(incomplete) == sorted(item.condition_id for item in planned if "subsets" in item.condition_id)
    assert set(incomplete[planned[1].condition_id].values()) == {"incomplete_exhausted"}


DRIVER_NAMES = {
    "ruleforge.rulesets.subsets": ("load_manifest",),
    "ruleforge.evaluate.campaign": ("build_plan", "campaign_completion", "execution_settings",
                                    "expected_requests_per_draw", "preflight", "run_plan", "run_plans_packed"),
    "ruleforge.evaluate.identity": ("COMPLETION_FNAME", "draw_dirname"),
    "ruleforge.evaluate.bank": ("SharedBankDesign", "SharedBankDesignError", "PlannedCondition", "plan_conditions",
                                "verify_subsets", "write_design", "design_from_cfg", "job_seeds", "runs_bank",
                                "condition_cfg", "condition_complete", "job_draws", "plan_job", "incomplete_states",
                                "job_identity", "run_conditions", "run_packed_conditions"),
    "ruleforge.core.provenance": ("require_checkout",),
    "ruleforge.evaluate.runner": ("STATE_COMPLETE",),
    "ruleforge.tasks.spec": ("build_task_spec",),
    "ruleforge.core.artifacts": ("load_jsonl_records",),
    "ruleforge.judges.backend": ("ModelBackend",),
    "ruleforge.optimize.job_logging": ("ensure_job_logging",),
}


def test_every_name_the_driver_script_held_is_at_its_library_home():
    import importlib

    missing = [f"{module}.{name}" for module, names in DRIVER_NAMES.items() for name in names
               if not hasattr(importlib.import_module(module), name)]
    assert missing == [] and sum(len(names) for names in DRIVER_NAMES.values()) == 33


@pytest.mark.parametrize("place, message", [
    ("root.runs/batches", "lie one inside the other"),
    (".root.lock", "the campaign lock"),
    ("root/completion.json", "a record of the campaign's own"),
])
@pytest.mark.parametrize("preflight", [True, False], ids=["preflight", "run"])
def test_a_batch_root_where_the_job_writes_is_refused_before_anything_is_written(cli, design, tmp_path, place,
                                                                                  message, preflight):
    base, overrides = design
    root = tmp_path / "root"
    status, out, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", root, "--campaign-id",
                             "ruleset-size", *(("--preflight-only",) if preflight else ()), "--", *overrides[1:],
                             f"+judge_model.batch_root={tmp_path / place}")
    assert status == 2 and message in error and out == "", error
    assert not root.exists() and not records.runs_dir(root).exists()


@pytest.mark.parametrize(("flags", "extra", "confirmed"), [
    ((), (), False), (("--yes-spend",), (), True), ((), ("+confirm_spend=true",), True)],
    ids=["unconfirmed", "flag", "override"])
def test_a_bank_run_confirms_its_spend_by_flag_or_override(cli, design, tmp_path, monkeypatch, flags, extra, confirmed):
    from ruleforge.evaluate import bank
    from ruleforge.judges import batch_api

    seen = []

    def run_job(*_args):
        seen.append(batch_api.spend_is_confirmed())
        return []

    monkeypatch.setattr(bank, "run_job", run_job)
    _, overrides = design
    status, _, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", *flags, "--",
                           *overrides, *extra)
    assert status == 0, error
    assert seen == [confirmed]


@pytest.mark.parametrize("flags", [(), ("--yes-spend",)], ids=["override", "with the flag"])
def test_a_bank_run_refuses_a_spend_setting_that_is_not_a_boolean(cli, design, tmp_path, flags):
    _, overrides = design
    status, _, error = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", *flags, "--",
                           *overrides, "+confirm_spend='false'")
    assert status == 2 and "confirm_spend is true or false, got 'false'" in error
    assert not (tmp_path / "root").exists()


def _saved_config(root: Path) -> dict:
    """The resolved config a ``bank run`` into ``root`` saved with its run record."""
    import yaml

    [run] = [path for path in records.runs_dir(root).iterdir() if path.is_dir()]
    return yaml.safe_load((run / records.CONFIG_FNAME).read_text(encoding="utf-8"))


def test_a_saved_config_with_the_recorded_design_key_preflights_and_resumes_its_root(cli, design, tmp_path):
    """A config saved before the design block took its current key: the job runs from it as from today's."""
    import yaml

    from ruleforge.core.record_names import recorded_block_key

    base, overrides = design
    status, _, _ = cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "root", "--",
                       *overrides, "ruleset_size.jobs_seeds=[0]")
    assert status == 0
    saved = _saved_config(tmp_path / "root")
    block = {**saved.pop("ruleset_size"), "jobs_seeds": None}
    today, recorded = tmp_path / "today.yaml", tmp_path / "recorded.yaml"
    today.write_text(yaml.safe_dump({**saved, "ruleset_size": block}), encoding="utf-8")
    recorded.write_text(yaml.safe_dump({**saved, recorded_block_key("ruleset_size"): block}), encoding="utf-8")
    status, planned, _ = cli("bank", "run", "--config", today, "--preflight-only")
    assert status == 0 and cli("bank", "run", "--config", recorded, "--preflight-only")[:2] == (0, planned)
    assert planned["conditions_this_job"] == 7
    assert cli("bank", "run", "--config", recorded)[0] == 0
    assert cli("bank", "run", "--config", CONFIG, "--judge", "fake", "--out", tmp_path / "whole", "--",
               *overrides)[0] == 0
    assert analysed(tmp_path / "root") == analysed(tmp_path / "whole")
    both = tmp_path / "both.yaml"
    both.write_text(yaml.safe_dump({**saved, "ruleset_size": block, recorded_block_key("ruleset_size"): block}),
                    encoding="utf-8")
    status, _, error = cli("bank", "run", "--config", both, "--preflight-only")
    assert status == 2 and "one experiment block under its recorded and its current key" in error
