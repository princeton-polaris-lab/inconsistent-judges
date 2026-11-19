from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ruleforge.evaluate import Campaign, CampaignError, SamplingSettings
from ruleforge.evaluate.campaign import build_plan, preflight, run_plan
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.judges import openai_batch
from ruleforge.judges.backend import ModelBackend
from ruleforge.judges.batch_api import BatchExecutionError, guard_work_directories, work_directory
from ruleforge.tasks.spec import build_task_spec
from tests.evaluate.helpers import inputs, tree
from tests.test_sampling_campaign import compose_campaign

STOPPED = "stopped before any spend"


@pytest.fixture(autouse=True)
def _job(monkeypatch, tmp_path):
    """The job's environment, and a Batch transport that prepares its work directory and stops before submitting."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)

    def prepare_then_stop(self, requests):
        self.prepare(requests)
        raise BatchExecutionError(STOPPED)

    monkeypatch.setattr(openai_batch.OpenAIBatchRunner, "submit", prepare_then_stop)


@pytest.fixture
def published(tmp_path) -> Path:
    """A published root's stand-in: a campaign root this library did not create, with a file of its own."""
    root = tmp_path / "published"
    (root / "aggregate").mkdir(parents=True)
    (root / SAMPLING_MANIFEST_FNAME).write_text("{}")
    (root / "aggregate" / "estimates.json").write_text("{}")
    return root


def run_job(tmp_path: Path, batch_root: str):
    """A configured campaign job of the Batch judge with ``batch_root`` as given, run as a campaign job runs it."""
    cfg = compose_campaign(tmp_path, **{"model@judge_model": "gpt-5.6-luna-batch"})
    cfg.judge_model.batch_root = batch_root
    preflight(cfg)
    backend = ModelBackend(cfg.judge_model)
    task_spec = build_task_spec(cfg)
    plan = build_plan(cfg, backend=backend, task_spec=task_spec)
    return run_plan(plan, backend=backend, task_spec=task_spec)


def test_a_symlink_already_at_a_work_directory_is_refused_before_anything_is_written(tmp_path, published):
    batches = tmp_path / "batches"
    with pytest.raises(BatchExecutionError, match=STOPPED):
        run_job(tmp_path, str(batches))
    (workdir,) = batches.iterdir()
    shutil.rmtree(workdir)
    workdir.symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    with pytest.raises(CampaignError, match=f"the batch work directory {workdir} is a symlink"):
        run_job(tmp_path, str(batches))
    assert tree(published) == before


def test_a_batch_root_that_resolves_only_in_the_job_is_held_where_it_resolves(tmp_path, published, monkeypatch):
    """``${hydra:runtime.cwd}/...`` resolves only in the job (the launch refuses it unresolved)."""
    monkeypatch.setattr(HydraConfig.instance(), "cfg", OmegaConf.create({"hydra": {"runtime": {"cwd": str(published)}}}))
    before = tree(published)
    with pytest.raises(CampaignError, match=f"batch_root {published / 'batch'} is inside {published}, a campaign root "
                                            "this library did not create"):
        run_job(tmp_path, "${hydra:runtime.cwd}/batch")
    assert tree(published) == before and not (published / "batch").exists()


def test_a_quoted_env_read_under_the_job_s_export_is_held_where_it_resolves(tmp_path, published, monkeypatch):
    """A quoted ``${oc.env:"NAME"}`` read (the launch refuses its export)."""
    monkeypatch.setenv("JOB_BATCH_ROOT", str(published / "batch"))
    before = tree(published)
    with pytest.raises(CampaignError, match="a campaign root this library did not create"):
        run_job(tmp_path, '${oc.env:"JOB_BATCH_ROOT"}')
    assert tree(published) == before and not (published / "batch").exists()


def test_the_job_s_own_root_s_batch_directory_is_its_one_place_in_the_root(tmp_path):
    root = tmp_path / "root"
    with pytest.raises(BatchExecutionError, match=STOPPED):
        run_job(tmp_path, str(root / "batch"))
    assert len(list((root / "batch").iterdir())) == 1
    with pytest.raises(CampaignError, match=f"lies in the campaign root {root} outside its batch/"):
        run_job(tmp_path, str(root / "elsewhere"))
    assert not (root / "elsewhere").exists()


def test_a_library_campaign_s_own_guard_is_left_in_place(tmp_path, published):
    from ruleforge.evaluate.campaign import guarding_batches

    seen = []
    with guard_work_directories(lambda root, directory: seen.append(directory)), guarding_batches([published]):
        assert work_directory(published, "x") == published / "x"
    assert seen == [published / "x"]
    with pytest.raises(CampaignError, match="did not create"), guarding_batches([tmp_path / "root"]):
        work_directory(published, "x")
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=inputs(tmp_path)[0],
                               sampling=SamplingSettings(draws=2))
    with guarding_batches([campaign.root]):
        assert work_directory(campaign.root / "batch", "x") == campaign.root / "batch" / "x"
