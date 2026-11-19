from __future__ import annotations

import json
import shutil

import pytest

from ruleforge.evaluate import Campaign, CampaignError, SamplingSettings
from ruleforge.judges import JudgeSpec
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.judges import batch_api, openai_batch
from ruleforge.judges.batch_api import BatchExecutionError, BatchRequest, guard_work_directories, work_directory
from tests.evaluate.helpers import configured_root, inputs, tree

TWO = SamplingSettings(draws=2)


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def luna_batch(batch_root=None) -> JudgeSpec:
    return JudgeSpec.from_config("gpt-5.6-luna-batch", batch_root=None if batch_root is None else str(batch_root))


@pytest.fixture
def prepared_once(tmp_path, monkeypatch):
    """A campaign whose batch judge prepared its first block's work directory, then stopped before any submission."""
    scenarios, rules = inputs(tmp_path)
    published = configured_root(tmp_path / "published")

    def prepare_then_stop(self, requests):
        self.prepare(requests)
        raise BatchExecutionError("stopped before any spend")

    monkeypatch.setattr(openai_batch.OpenAIBatchRunner, "submit", prepare_then_stop)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=TWO)
    campaign.condition("luna/naive", rules=rules, judge=luna_batch())
    with pytest.raises(BatchExecutionError, match="stopped before any spend"):
        campaign.run(confirm_spend=True)
    (workdir,) = (campaign.root / "batch").iterdir()
    assert (workdir / "state.json").is_file()
    return campaign, scenarios, rules, published, workdir


def _again(campaign, scenarios, rules, judge=None):
    rebound = Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=TWO)
    rebound.condition("luna/naive", rules=rules, judge=judge or luna_batch())
    return rebound


def test_a_symlink_planted_where_a_work_directory_goes_is_refused_before_anything_is_written(prepared_once):
    campaign, scenarios, rules, published, workdir = prepared_once
    shutil.rmtree(workdir)
    workdir.symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    with pytest.raises(CampaignError, match=f"the batch work directory {workdir} is a symlink"):
        _again(campaign, scenarios, rules).run(confirm_spend=True)
    assert tree(published) == before


def test_a_symlink_below_a_work_directory_is_refused(prepared_once):
    campaign, scenarios, rules, published, workdir = prepared_once
    shutil.rmtree(workdir / "inputs")
    (workdir / "inputs").symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    with pytest.raises(CampaignError, match=f"{workdir / 'inputs'}, below the batch work directory .* is a symlink"):
        _again(campaign, scenarios, rules).run(confirm_spend=True)
    assert tree(published) == before


def test_a_work_directory_holding_a_campaign_root_is_refused(prepared_once):
    campaign, scenarios, rules, _, workdir = prepared_once
    (workdir / SAMPLING_MANIFEST_FNAME).write_text("{}")
    with pytest.raises(CampaignError, match="holds a campaign root of its own"):
        _again(campaign, scenarios, rules).run(confirm_spend=True)


def test_a_batch_root_outside_the_campaign_is_held_to_the_same(tmp_path, monkeypatch, prepared_once):
    campaign, scenarios, rules, published, workdir = prepared_once
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / workdir.name).symlink_to(published / "aggregate", target_is_directory=True)
    rebound = _again(campaign, scenarios, rules, judge=luna_batch(outside))
    before = tree(published)
    with pytest.raises(CampaignError, match=f"the batch work directory {outside / workdir.name} is a symlink"):
        rebound.run(confirm_spend=True, relocate_batch_root=True)
    assert tree(published) == before


def test_a_resumed_run_derives_the_same_work_directory(prepared_once):
    campaign, scenarios, rules, _, workdir = prepared_once
    state = json.loads((workdir / "state.json").read_text())
    with pytest.raises(BatchExecutionError, match="stopped before any spend"):
        _again(campaign, scenarios, rules).run(confirm_spend=True)
    assert [path.name for path in (campaign.root / "batch").iterdir()] == [workdir.name]
    assert json.loads((workdir / "state.json").read_text()) == state


@pytest.mark.parametrize("change", ["reasoning_summary", "batch_root"])
def test_a_judge_started_with_other_request_settings_or_batch_root_than_recorded_is_refused(tmp_path, monkeypatch,
                                                                                            change):
    scenarios, rules = inputs(tmp_path)
    submitted = []
    monkeypatch.setattr(openai_batch.OpenAIBatchRunner, "submit", lambda self, requests: submitted.append(requests))
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=TWO)
    campaign.condition("luna/naive", rules=rules, judge=luna_batch())
    other = {"reasoning_summary": "detailed", "batch_root": str(campaign.root / "elsewhere")}[change]
    real = JudgeSpec.start
    monkeypatch.setattr(JudgeSpec, "start", lambda self: real(self.with_options(**{change: other})))
    with pytest.raises(CampaignError, match=r"the judge started for condition 'luna/naive' sends request settings "
                                            r".* and its execution record holds"):
        campaign.run(confirm_spend=True)
    assert not submitted and not (campaign.root / "luna" / "naive" / "draw-000").exists()
    assert campaign.execution_record("luna/naive")["request_settings"] == {"reasoning_summary": "auto"}


def test_the_guard_runs_only_inside_its_block_and_leaves_the_address_unchanged(tmp_path):
    seen = []
    request = BatchRequest(custom_id="x", body={"model": "m", "input": "p"})
    name = batch_api.fingerprint([request], "/v1/responses")
    with guard_work_directories(lambda root, directory: seen.append((root, directory))):
        assert work_directory(tmp_path, name) == tmp_path / name
    assert work_directory(tmp_path, name) == tmp_path / name and seen == [(tmp_path, tmp_path / name)]
