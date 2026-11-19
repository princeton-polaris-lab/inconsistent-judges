from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate import Campaign, ConditionSettings, configured
from ruleforge.evaluate import bank
from ruleforge.evaluate.api import CampaignError, refuse_work_directory
from ruleforge.evaluate.campaign import (
    build_plan,
    checked_at_entry,
    load_conditions,
    preflight,
    refuse_retired_root,
    register_condition,
    run_plan,
    run_plans_packed,
)
from ruleforge.experiments import driver
from ruleforge.experiments.prompt_ablation import templates
from ruleforge.evaluate.identity import recorded_fingerprint
from ruleforge.evaluate.packing import ExecutionSettings
from ruleforge.evaluate.shard_merge import merge_condition_shards
from ruleforge.judges import JudgeSpec
from tests.cli.test_bank_cli import compose_driver, design, script  # noqa: F401 (design is a fixture)
from tests.evaluate.test_campaign_api import PILOT, inputs  # noqa: F401 (inputs is a fixture)
from tests.test_sampling_campaign import backend_for, compose_campaign
from tests.test_sampling_shards import CONDITION, SeededBackend, compose_cfg, run_shards

REFUSED = r"registers a condition of a retired style \('good'\)"
#: A refusal, from the protection of a campaign root (CampaignError) or from the root's own check.
REFUSALS = (SamplingDesignError, CampaignError)
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def plant_retired(root: Path, active: Path, condition_id: str = "judge/good") -> Path:
    """A registration of the retired style ``good`` in ``root``, whole: its fingerprint, id and directory its own."""
    fields = {**json.loads(active.read_text(encoding="utf-8")), "condition_id": condition_id, "prompt_style": "good"}
    path = root / condition_id / "condition.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**fields, "fingerprint": recorded_fingerprint(fields)}), encoding="utf-8")
    return path


def contents(root: Path) -> dict[str, bytes | None]:
    """Every file under ``root`` by its bytes and every directory (None), but the lock files a refused writer may
    leave."""
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(root.rglob("*")) if not path.name.endswith(".lock")}


@pytest.mark.parametrize("condition_id", [None, "judge/additional"], ids=["resume", "extension"])
def test_a_configured_job_into_such_a_root_is_refused_before_a_call_or_a_write(tmp_path, modelspec, condition_id):
    cfg = compose_campaign(tmp_path)
    backend, task = backend_for("naive-rules-above"), modelspec(reasoning_prompt_style="naive-rules-above",
                                                                 evaluation_temperature=1.0)
    plan = build_plan(cfg, backend=backend, task_spec=task)
    root = plan.campaign_root
    plant_retired(root, plan.condition_root / "condition.json")
    before = contents(root)
    if condition_id is not None:
        cfg.condition_id = condition_id
    for step in (lambda: preflight(cfg), lambda: build_plan(cfg, backend=backend, task_spec=task),
                 lambda: register_condition(root, plan.condition),
                 lambda: run_plan(plan, backend=backend, task_spec=task),
                 lambda: run_plan(plan, backend=backend, task_spec=task, gate=False),
                 lambda: run_plans_packed([plan], backend=backend, task_spec=task, execution=ExecutionSettings())):
        with pytest.raises(REFUSALS, match=REFUSED):
            step()
    assert backend.blocks == [] and contents(root) == before


@pytest.mark.parametrize("program", ["run_campaign", "preflight_campaign"])
def test_a_configured_job_program_into_such_a_root_leaves_its_tree_byte_for_byte(tmp_path, modelspec, program):
    """The job's program refuses before Hydra makes its run directory, its log or its saved configs in the root."""
    from omegaconf import OmegaConf

    cfg = compose_campaign(tmp_path)
    plan = build_plan(cfg, backend=backend_for("naive-rules-above"),
                      task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    root = plan.campaign_root
    plant_retired(root, plan.condition_root / "condition.json")
    OmegaConf.save(cfg, tmp_path / "job.yaml")
    before = contents(root)
    paths = [str(REPO / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    ran = subprocess.run([sys.executable, "-m", f"ruleforge.launch.programs.{program}", "--config-path", str(tmp_path),
                          "--config-name", "job", f"hydra.run.dir={root}/hydra/job"], cwd=REPO, capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(paths)})
    assert ran.returncode != 0 and "registers a condition of a retired style ('good')" in ran.stderr, ran.stderr[-2000:]
    assert contents(root) == before


def test_a_registration_run_or_rebinding_of_such_a_library_root_is_refused(tmp_path, inputs):  # noqa: F811
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="demo", scenarios=scenarios, sampling=PILOT)
    condition = campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1),
                                   settings=ConditionSettings(prompt="naive"))
    plant_retired(campaign.root, campaign.root / condition.condition_id / "condition.json")
    before = contents(campaign.root)
    for step in (lambda: campaign.condition("fake-2/naive", rules=rules, judge=JudgeSpec.fake(2),
                                            settings=ConditionSettings(prompt="naive")),
                 lambda: campaign.run("fake-1/naive"), lambda: campaign.check_shard_merge("fake-1/naive"),
                 lambda: refuse_work_directory(campaign.root / "batch" / "key", "the batch work directory"),
                 lambda: Campaign.create(campaign.root, campaign_id="demo", scenarios=scenarios, sampling=PILOT)):
        with pytest.raises(CampaignError, match=REFUSED):
            step()
    assert contents(campaign.root) == before and not (campaign.root / "fake-2").exists()


def test_a_shard_merge_into_such_a_root_copies_nothing(tmp_path, modelspec):
    root = tmp_path / "root"
    run_shards(tmp_path, modelspec, root, 2)
    task = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    plan = build_plan(compose_cfg(tmp_path, root), backend=SeededBackend(), task_spec=task)
    plant_retired(root, plan.condition_root / "condition.json")
    before = contents(root)
    for step in (lambda: merge_condition_shards(root, CONDITION), lambda: configured.merge_shards(root, CONDITION),
                 lambda: configured.check_merge(root, CONDITION)):
        with pytest.raises(REFUSALS, match=REFUSED):
            step()
    assert contents(root) == before


def test_a_bank_job_into_such_a_root_is_refused_before_a_write(design, tmp_path):  # noqa: F811
    base, overrides = design
    root = tmp_path / "bank-root"
    script(base, overrides, root)
    cfg = compose_driver(f"hydra.searchpath=[file://{base}]", "model@judge_model=fake", f"campaign_root={root}",
                         *overrides)
    active = next(root.glob("**/condition.json"))
    plant_retired(root, active)
    before = contents(root)
    with pytest.raises(SamplingDesignError, match=REFUSED):
        bank.plan_job(cfg)
    with pytest.raises(SamplingDesignError, match=REFUSED):
        bank.run_planned(cfg, bank.design_from_cfg(cfg), [], {})
    assert contents(root) == before


def test_a_job_s_writes_rest_on_its_entry_check(tmp_path, modelspec):
    """``checked_at_entry`` refuses a root at a job's entry and is not repeated within the job: a registration planted
    while it runs is not looked for, and the next entry refuses the root."""
    plan = build_plan(compose_campaign(tmp_path), backend=backend_for("naive-rules-above"),
                      task_spec=modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0))
    root = plan.campaign_root
    with checked_at_entry(root):
        plant_retired(root, plan.condition_root / "condition.json")
        refuse_retired_root(root)
    with pytest.raises(SamplingDesignError, match=REFUSED):
        refuse_retired_root(root)
    with pytest.raises(SamplingDesignError, match=REFUSED), checked_at_entry(root):
        pass  # never entered: the entry check refuses


def test_a_registration_below_another_s_draw_directory_is_read_and_a_retired_one_there_refused(tmp_path, modelspec):
    """A configured root accepts an id below another condition's draw directory: the loader reads it, and the check
    finds a retired registration there."""
    cfg = compose_campaign(tmp_path, condition_id="judge/parent")
    task = modelspec(reasoning_prompt_style="naive-rules-above", evaluation_temperature=1.0)
    parent = build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=task)
    (parent.condition_root / "draw-000").mkdir(exist_ok=True)
    cfg.condition_id = "judge/parent/draw-000/child"
    build_plan(cfg, backend=backend_for("naive-rules-above"), task_spec=task)
    root = parent.campaign_root
    assert sorted(load_conditions(root)) == ["judge/parent", "judge/parent/draw-000/child"]
    plant_retired(root, parent.condition_root / "condition.json", "judge/parent/draw-000/good")
    for check in (lambda: refuse_retired_root(root), lambda: load_conditions(root)):
        with pytest.raises(SamplingDesignError, match=REFUSED):
            check()


def test_a_producer_writes_no_output_into_a_campaign_root(tmp_path, inputs):  # noqa: F811
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="demo", scenarios=scenarios, sampling=PILOT)
    active = campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1),
                                settings=ConditionSettings(prompt="naive"))
    plant_retired(campaign.root, campaign.root / active.condition_id / "condition.json")
    before = contents(campaign.root)
    for place in (campaign.root / "templates", campaign.root / "aggregate" / "estimates.json", campaign.root):
        with pytest.raises(SystemExit):
            templates.main(["--output-dir", str(place)])
        with pytest.raises(driver.argparse.ArgumentTypeError, match="campaign root"):
            driver.output_path(str(place))
    assert contents(campaign.root) == before
    assert driver.output_path(str(tmp_path / "out" / "golden.json")) == tmp_path / "out" / "golden.json"
    assert templates.main(["--output-dir", str(tmp_path / "templates")]) == 0
    assert sorted(path.name for path in (tmp_path / "templates").iterdir())
    # A directory outside the root whose file is a link into it is refused before anything is written.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "naive-rules-above.txt").symlink_to(campaign.root / "sampling_manifest.json")
    with pytest.raises(SystemExit):
        templates.main(["--output-dir", str(outside)])
    with pytest.raises(driver.argparse.ArgumentTypeError, match="symlink"):
        driver.output_dir(str(outside))
    assert contents(campaign.root) == before and sorted(path.name for path in outside.iterdir()) == [
        "naive-rules-above.txt"]
