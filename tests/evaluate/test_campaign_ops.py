from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from ruleforge.evaluate import (
    Approvals,
    Campaign,
    CampaignError,
    ConditionSettings,
    ExecutionSettings,
    ReadOnlyCampaign,
    SamplingSettings,
    bank,
    evaluate,
)
from ruleforge.evaluate import api
from ruleforge.evaluate.results import Contrast, Kappa, contrasts_of
from ruleforge.judges import JudgeError, JudgeSpec
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME
from ruleforge.evaluate.identity import (
    SHARED_BANK_DESIGN_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    SAMPLING_MANIFEST_FNAME,
    BootstrapSettings,
    SamplingConfig,
)
from ruleforge.evaluate.discrimination import SINGLE_STRATUM
from ruleforge.evaluate.bank import SHARED_BANK_ESTIMATES_FNAME, SharedBankDesignError
from ruleforge.evaluate.runner import STATE_COMPLETE, CampaignRunError, draw_lock_path
from ruleforge.evaluate.sharding import campaign_lock_path
from ruleforge.scenarios import ScenarioSet
from ruleforge.judges.batch_api import BatchSpendNotConfirmed, require_spend_confirmation
from tests.conftest import requires_git_worktree
from tests.evaluate.helpers import (
    Rewritten,
    configured_root,
    count_starts,
    inputs,
    rewrite_answers,
    tree,
    write_rules,
)
from tests.test_sampling_runner import RULES

TWO = SamplingSettings(draws=2)
REPO_ROOT = Path(__file__).resolve().parents[2]
#: The artifact root holding the published roots, read before any test points RULEFORGE_ARTIFACT_ROOT elsewhere.
PUBLISHED = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT") or "/nonexistent")


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


@pytest.fixture
def made(tmp_path):
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=TWO)
    return campaign, scenarios, rules


def test_sampling_settings_are_the_toolkit_s_defaults_and_validate_as_a_config():
    config = SamplingSettings().config()
    assert (config.num_decoder_draws, config.max_semantic_attempts, config.retry_scope, config.label_rule,
            config.campaign_mode) == (5, 30, "component", "lenient_last_line", "pilot")
    assert SamplingSettings.from_config(config) == SamplingSettings()
    for bad, match in (({"draws": 1}, "num_decoder_draws must be at least 2"),
                       ({"approved_scenario_count": 0}, "approved_scenario_count must be a positive integer"),
                       ({"max_transport_retries": True}, "max_transport_retries must be a positive integer")):
        with pytest.raises(CampaignError, match=match):
            SamplingSettings(**bad)


def test_a_population_other_than_the_approved_count_is_refused(tmp_path):
    scenarios, _ = inputs(tmp_path)
    with pytest.raises(CampaignError, match="approves 4 scenarios; scenarios.jsonl holds 3"):
        Campaign.create(tmp_path / "r", campaign_id="c", scenarios=scenarios,
                        sampling=SamplingSettings(draws=2, approved_scenario_count=4))
    assert not (tmp_path / "r").exists()


def test_a_final_campaign_judges_the_count_it_approves(tmp_path):
    """``approved_scenario_count`` replaces the paper's 500 for a final campaign that states one."""
    scenarios, _ = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "r", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(campaign_mode="final", approved_scenario_count=3))
    assert campaign.config.campaign_mode == "final" and (campaign.root / api.MARKER_FNAME).is_file()
    with pytest.raises(CampaignError, match="a final campaign judges 500 scenarios unless it approves another count"):
        Campaign.create(tmp_path / "s", campaign_id="c", scenarios=scenarios,
                        sampling=SamplingSettings(campaign_mode="final"))
    assert not (tmp_path / "s").exists()


def test_approvals_pin_the_ruleset_and_the_levels_and_reach_the_judge_preflight(made, tmp_path):
    campaign, _, rules = made
    with pytest.raises(CampaignError, match="not the approved 0+$"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1),
                           settings=ConditionSettings(approvals=Approvals(rule_set_sha256="0" * 64)))
    with pytest.raises(CampaignError, match="reads each rule's level from rule_levels"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1),
                           settings=ConditionSettings(approvals=Approvals(rule_levels_sha256="0" * 64)))
    levels = tmp_path / "levels.jsonl"
    levels.write_text("".join(json.dumps({**rule, "rule_type": "user"}) + "\n" for rule in RULES))
    with pytest.raises(CampaignError, match="rule levels SHA-256 .* does not match approved 0+"):
        campaign.condition("fake-1/h", rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(
            prompt="naive-hierarchy", rule_context="superiors", rule_levels=levels,
            approvals=Approvals(rule_levels_sha256="0" * 64)))
    digest = hashlib.sha256(levels.read_bytes()).hexdigest()
    assert campaign.condition("fake-1/h", rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(
        prompt="naive-hierarchy", rule_context="superiors", rule_levels=levels,
        approvals=Approvals(rule_levels_sha256=digest))).spec.rule_levels_sha256 == digest
    with pytest.raises(CampaignError, match="approved_adapter_sha256=.* but judge_model.adapter_path is unset"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1),
                           settings=ConditionSettings(approvals=Approvals(adapter_sha256="a" * 64)))
    assert campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(
        approvals=Approvals(rule_set_sha256=rules.sha256))).spec.rule_set_sha256 == rules.sha256


def test_an_auto_name_its_judge_cannot_carry_is_refused(made):
    campaign, _, rules = made
    with pytest.raises(CampaignError, match="condition gpt test/naive: condition_id must be"):
        campaign.condition(rules=rules, judge=JudgeSpec.openai("gpt test", fixed_temperature=1.0))


def test_run_judges_the_draws_not_complete_and_starts_nothing_when_none_are(made, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    started = count_starts(monkeypatch)
    first = campaign.run("fake-1/naive-rules-above", draw_ids=[1])
    assert list(first["fake-1/naive-rules-above"]) == [1] and len(started) == 1
    every = campaign.run()
    assert list(every["fake-1/naive-rules-above"]) == [0] and len(started) == 2
    assert campaign.run() == {"fake-1/naive-rules-above": {}} and len(started) == 2
    assert set(campaign.result("fake-1/naive-rules-above").draws.values()) == {STATE_COMPLETE}


def test_a_registered_condition_runs_itself(made):
    campaign, _, rules = made
    condition = campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    assert list(condition.run(draw_ids=(1,))) == [1] and list(condition.run()) == [0] and condition.run() == {}
    unbound = api.Condition(condition_id="x", spec=condition.spec, judge=condition.judge, rules=rules,
                            settings=condition.settings, task_locator="modelspec")
    with pytest.raises(CampaignError, match="bound to no campaign"):
        unbound.run()


def test_a_condition_over_scenarios_lacking_a_field_its_task_reads_is_refused_before_any_write(tmp_path):
    path = tmp_path / "no_response.jsonl"
    path.write_text("".join(json.dumps({"prompt": f"prompt {i}"}) + "\n" for i in range(3)))
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=ScenarioSet.load(path), sampling=TWO)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=r"condition fake-1/naive-rules-above \(.*no_response.jsonl\): scenario row 0 "
                                            r"lacks \['response'\]"):
        campaign.condition("fake-1/naive-rules-above", rules=write_rules(tmp_path / "rules.jsonl"), judge=JudgeSpec.fake(1))
    assert tree(campaign.root) == before


def test_a_task_spec_that_cannot_be_built_starts_no_judge(made, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    started = count_starts(monkeypatch)

    def unbuildable(task, settings):
        raise RuntimeError("no task spec")

    monkeypatch.setattr(api, "_task_spec", unbuildable)
    with pytest.raises(RuntimeError, match="no task spec"):
        campaign.run()
    assert started == [] and not (campaign.root / "fake-1/naive-rules-above" / "draw-000").exists()


def test_analyze_refuses_a_campaign_lock_planted_beside_the_root(made, tmp_path):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    target = tmp_path / "published" / "aggregate" / "planted.json"
    target.parent.mkdir(parents=True)
    campaign_lock_path(campaign.root).unlink()
    campaign_lock_path(campaign.root).symlink_to(target)
    with pytest.raises(CampaignError, match="is not a file of its own"):
        campaign.analyze()
    assert not target.exists() and not (campaign.root / AGGREGATE_DIRNAME).exists()


@pytest.mark.parametrize("draw_ids", [(2,), (0, 0), (True,), (-1,)])
def test_run_takes_a_selection_of_the_campaign_s_draws(made, draw_ids):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match="must be distinct ids below 2"):
        campaign.run("fake-1/naive-rules-above", draw_ids=draw_ids)


def test_run_refuses_what_a_campaign_of_its_own_cannot_run(made, monkeypatch):
    campaign, _, rules = made
    with pytest.raises(CampaignError, match="no condition is bound"):
        campaign.run()
    with pytest.raises(CampaignError, match="no judge is bound to 'x'"):
        campaign.run("x")
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match="judges a condition in one process"):
        campaign.run(execution=ExecutionSettings(num_shards=2, shard_index=0))
    with pytest.raises(CampaignError, match="pack_draws needs a 'local' judge"):
        campaign.run(execution=ExecutionSettings(pack_draws=True))
    with pytest.raises(ReadOnlyCampaign):
        Campaign.open(campaign.root).run()
    manifest = campaign.root / SAMPLING_MANIFEST_FNAME
    original = manifest.read_text()
    manifest.write_text(json.dumps({**json.loads(original), MANIFEST_AMENDMENTS_KEY: [{}]}))
    with pytest.raises(CampaignError, match=r"records a cap amendment, which only a configured campaign's jobs run "
                                            "under") as refusal:
        campaign.run()
    assert "v1.0" not in str(refusal.value)
    manifest.write_text(original)


def test_run_resumes_nothing_recorded_for_another_identity(made):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run(draw_ids=[0])
    gate = campaign.root / "fake-1" / "naive-rules-above" / "draw-000" / "completion.json"
    gate.write_text(json.dumps({**json.loads(gate.read_text()), "condition_fingerprint": "0" * 64}))
    with pytest.raises(CampaignError, match=r"draws \[0\] were recorded for another fingerprint"):
        campaign.run()


def _seed_base_edited(root: Path, tmp_path: Path) -> None:
    manifest = root / SAMPLING_MANIFEST_FNAME
    manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "generation_seed_base": 99}))


def _population_swapped(root: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    ScenarioSet.load(other).manifest.write(root / SCENARIO_MANIFEST_FNAME)


def _sampling_manifest_removed(root: Path, tmp_path: Path) -> None:
    (root / SAMPLING_MANIFEST_FNAME).unlink()


FROZEN_EDITS = pytest.mark.parametrize("edit, match", [
    (_seed_base_edited, r"no longer records the campaign this process holds \(.*describes a different campaign "
                        r"\(generation_seed_base differ"),
    (_population_swapped, r"froze another population \(.*\) than the one this campaign holds"),
    (_sampling_manifest_removed, f"holds no {SAMPLING_MANIFEST_FNAME}: its records no longer describe"),
], ids=["seed-base", "population", "manifest-removed"])


@FROZEN_EDITS
def test_a_held_campaign_whose_frozen_records_changed_writes_nothing(made, tmp_path, monkeypatch, edit, match):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    edit(campaign.root, tmp_path)
    before = tree(campaign.root)
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started"))
    for operation in (campaign.run,
                      lambda: campaign.condition("fake-2/naive-rules-above", rules=rules, judge=JudgeSpec.fake(2)),
                      campaign.analyze):
        with pytest.raises(CampaignError, match=match):
            operation()
    assert tree(campaign.root) == before


def test_frozen_records_changed_while_the_judge_starts_are_refused_before_any_draw(made, tmp_path, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start

    def start(self):
        _seed_base_edited(campaign.root, tmp_path)
        return real(self)

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match="describes a different campaign"):
        campaign.run()
    assert not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()


def test_frozen_records_changed_during_a_run_are_refused_at_its_next_write(made, tmp_path, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    rewrite_answers(monkeypatch, lambda prompt, outcome: _seed_base_edited(campaign.root, tmp_path))
    with pytest.raises(CampaignError, match="describes a different campaign"):
        campaign.run()
    attempts = campaign.root / "fake-1" / "naive-rules-above" / "draw-000" / "attempts.jsonl"
    assert not attempts.exists() or attempts.read_text() == ""


def test_a_condition_registered_anew_under_a_held_binding_is_refused(made, monkeypatch):
    campaign, scenarios, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    registered = campaign.root / "fake-1" / "naive-rules-above" / "condition.json"
    registered.unlink()
    other = write_rules(campaign.root.parent / "other_rules.jsonl", RULES[:1])
    Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=TWO).condition(
        "fake-1/naive-rules-above", rules=other, judge=JudgeSpec.fake(1))
    started = count_starts(monkeypatch)
    with pytest.raises(CampaignError, match="already registers condition 'fake-1/naive-rules-above' with a different identity; "
                                            "two runs .*; bind the condition the root registers"):
        campaign.run()
    assert started == [] and not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()
    real = JudgeSpec.start
    rebound = campaign.condition("fake-2/naive-rules-above", rules=rules, judge=JudgeSpec.fake(2))

    def start(self):
        (campaign.root / "fake-2" / "naive-rules-above" / "condition.json").unlink()
        return real(self)

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match="is gone: condition 'fake-2/naive-rules-above' is no longer registered"):
        rebound.run()
    assert not (campaign.root / "fake-2" / "naive-rules-above" / "draw-000").exists()


def _hold(path: Path) -> int:
    """Hold ``path``'s ``flock`` through a descriptor of its own, as another process would."""
    import fcntl

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def test_a_condition_another_process_is_judging_is_refused_at_once(made, monkeypatch):
    """Two runs of one condition would each repair and append to its draws' ledgers."""
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.condition("fake-2/naive-rules-above", rules=rules, judge=JudgeSpec.fake(2))
    started = count_starts(monkeypatch)
    descriptor = _hold(campaign.root / "fake-1" / "naive-rules-above" / api.RUN_LOCK_FNAME)
    try:
        with pytest.raises(CampaignError, match=r"another process holds .*fake-1/naive-rules-above/\.ruleforge-run\.lock, "
                                                "condition 'fake-1/naive-rules-above''s run lock: another process is judging"):
            campaign.run("fake-1/naive-rules-above")
        assert started == [] and not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()
        assert list(campaign.run("fake-2/naive-rules-above")["fake-2/naive-rules-above"]) == [0, 1]
    finally:
        os.close(descriptor)
    assert list(campaign.run("fake-1/naive-rules-above")["fake-1/naive-rules-above"]) == [0, 1]


def test_draws_another_run_completed_meanwhile_are_not_judged_again(made, monkeypatch):
    campaign, scenarios, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    other = Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=TWO)
    other.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = api.Campaign._pending
    looks = []

    def pending(self, condition_id, draws):
        found = real(self, condition_id, draws)
        if self is campaign and not looks:
            looks.append(found)
            other.run()
        return found

    monkeypatch.setattr(api.Campaign, "_pending", pending)
    started = count_starts(monkeypatch)
    # The one judge started is the other process's.
    assert campaign.run() == {"fake-1/naive-rules-above": {}} and looks == [(0, 1)] and len(started) == 1
    assert set(campaign.result("fake-1/naive-rules-above").draws.values()) == {STATE_COMPLETE}


def test_the_campaign_gate_a_run_leaves_is_computed_again_under_the_root_s_lock(made, monkeypatch):
    from ruleforge.evaluate.campaign import campaign_completion

    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    gate = campaign.root / "completion.json"
    monkeypatch.setattr("ruleforge.evaluate.campaign.campaign_completion",
                        lambda root: gate.write_text(json.dumps({"stale": True})))
    campaign.run()
    written = json.loads(gate.read_text())
    expected = campaign_completion(campaign.root, write=False)
    assert written.pop("updated_at") and expected.pop("updated_at")
    assert written == expected and written["state"] == STATE_COMPLETE


def _root_locked(root: Path) -> bool:
    """Whether something holds ``root``'s write lock (``flock`` is per open file, and this opens another)."""
    import fcntl

    descriptor = os.open(root / api.LOCK_FNAME, os.O_WRONLY | os.O_CREAT)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def test_every_write_a_run_makes_to_a_file_the_root_s_writers_share_is_made_under_the_root_s_lock(made,
                                                                                                    monkeypatch):
    import ruleforge.evaluate.campaign as campaign_module
    from ruleforge.evaluate.runner import DrawRunner

    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    writes = []
    gate, snapshot, stack = api.campaign_completion, api.write_code_snapshot, DrawRunner._record_condition_stack

    def published(root, *args, write=True, **kwargs):
        if write:
            writes.append(("gate", _root_locked(campaign.root)))
        return gate(root, *args, write=write, **kwargs)

    def archived(*args, **kwargs):
        writes.append(("snapshot", _root_locked(campaign.root)))
        return snapshot(*args, **kwargs)

    def recorded(self):
        writes.append(("stack", _root_locked(campaign.root)))
        return stack(self)

    monkeypatch.setattr(api, "campaign_completion", published)
    monkeypatch.setattr(campaign_module, "campaign_completion",
                        lambda *args, **kwargs: pytest.fail("run_plan wrote the campaign gate"))
    monkeypatch.setattr(api, "snapshot_path", lambda root, digest: Path(root) / "no-such-snapshot")
    monkeypatch.setattr(api, "code_state", lambda: {"code_state_sha256": "0" * 64})
    monkeypatch.setattr(api, "write_code_snapshot", archived)
    monkeypatch.setattr(DrawRunner, "_record_condition_stack", recorded)
    campaign.run()
    assert writes == [("snapshot", True), ("stack", True), ("stack", True), ("gate", True)]
    assert json.loads((campaign.root / "completion.json").read_text())["state"] == STATE_COMPLETE


def test_a_run_that_fails_after_judging_publishes_the_gate_under_the_lock_and_raises_its_own_failure(
        made, monkeypatch, caplog):
    """A run refused before its first draw -- its judge did not start -- leaves the gate as it found it."""
    from ruleforge.evaluate.campaign import campaign_completion

    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    gate = campaign.root / "completion.json"
    found = gate.read_bytes() if gate.exists() else None
    start, publish, real, compute = JudgeSpec.start, api.Campaign._publish_gate, api.run_plan, api.campaign_completion

    def stopped(self):
        raise _Stop

    monkeypatch.setattr(JudgeSpec, "start", stopped)
    monkeypatch.setattr(api.Campaign, "_publish_gate", lambda self: pytest.fail("a gate was published"))
    with pytest.raises(_Stop):
        campaign.run()
    assert (gate.read_bytes() if gate.exists() else None) == found
    monkeypatch.setattr(JudgeSpec, "start", start)
    monkeypatch.setattr(api.Campaign, "_publish_gate", publish)
    published = []

    def locked(root, *args, write=True, **kwargs):
        if write:
            published.append(_root_locked(campaign.root))
        return compute(root, *args, write=write, **kwargs)

    def judged_then_failed(plan, **kwargs):
        assert kwargs["gate"] is False
        real(plan, **kwargs)
        raise _Stop

    monkeypatch.setattr(api, "campaign_completion", locked)
    monkeypatch.setattr(api, "run_plan", judged_then_failed)
    with pytest.raises(_Stop):
        campaign.run(draw_ids=(0,))
    written = json.loads(gate.read_text())
    expected = campaign_completion(campaign.root, write=False)
    assert written.pop("updated_at") and expected.pop("updated_at")
    assert written == expected and written["conditions"]["fake-1/naive-rules-above"]["draws"]["0"]["state"] == STATE_COMPLETE
    assert published == [True]
    before = gate.read_bytes()

    def refused(self):
        raise CampaignError("the root is no longer the campaign's")

    monkeypatch.setattr(api.Campaign, "_publish_gate", refused)
    with caplog.at_level("WARNING"), pytest.raises(_Stop):
        campaign.run(draw_ids=(1,))
    assert ("was not published after the run of condition 'fake-1/naive-rules-above' failed: the root is no longer the "
            "campaign's") in caplog.text and gate.read_bytes() == before


class _Stop(Exception):
    pass


def _confirmed() -> bool:
    try:
        require_spend_confirmation()
    except BatchSpendNotConfirmed:
        return False
    return True


def test_a_judge_that_spends_money_runs_only_when_the_spend_is_confirmed(made, monkeypatch):
    campaign, _, rules = made
    campaign.condition("gpt/naive-rules-above", rules=rules, judge=JudgeSpec.openai("gpt-test", fixed_temperature=1.0))
    seen = []

    def start(self):
        seen.append(_confirmed())
        raise _Stop

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match=r"spends money on every call; pass confirm_spend=True \(--yes-spend\)"):
        campaign.run()
    assert seen == []
    with pytest.raises(_Stop):
        campaign.run(confirm_spend=True)
    assert seen == [True] and not _confirmed()


def test_overlapping_confirmed_runs_in_two_threads_leave_each_other_s_confirmation_alone(made, monkeypatch):
    import threading

    campaign, scenarios, rules = made
    campaign.condition("gpt/naive-rules-above", rules=rules, judge=JudgeSpec.openai("gpt-test", fixed_temperature=1.0))
    other = Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=TWO)
    other.condition("gpt-2/naive-rules-above", rules=rules, judge=JudgeSpec.openai("gpt-test-2", fixed_temperature=1.0))
    inside, first_done, seen, raised = threading.Event(), threading.Event(), {}, []

    def second():
        try:
            other.run(confirm_spend=True)
        except _Stop:
            raised.append("gpt-test-2")

    thread = threading.Thread(target=second)

    def start(self):
        if self.model_name == "gpt-test":
            thread.start()
            assert inside.wait(timeout=60)
        else:
            inside.set()
            assert first_done.wait(timeout=60)
        seen[self.model_name] = _confirmed()
        raise _Stop

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(_Stop):
        campaign.run(confirm_spend=True)
    first_done.set()
    thread.join(timeout=60)
    assert raised == ["gpt-test-2"] and seen == {"gpt-test": True, "gpt-test-2": True}
    assert not _confirmed()


@requires_git_worktree  # the run writes code snapshots, which need git's file list
def test_a_run_refuses_a_condition_directory_or_snapshots_carried_elsewhere(made, tmp_path, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    draw = campaign.root / "fake-1" / "naive-rules-above" / "draw-000"
    draw.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(CampaignError, match=f"{draw}, below condition 'fake-1/naive-rules-above''s directory .* is a symlink"):
        campaign.run()
    draw.unlink()
    snapshots = campaign.root / "code_snapshots"
    for item in snapshots.iterdir():
        item.rename(elsewhere / item.name)
        (snapshots / item.name).symlink_to(elsewhere / item.name)
    with pytest.raises(CampaignError, match="below the code snapshots' directory"):
        campaign.run()
    assert not any(path.name.startswith("draw") for path in elsewhere.iterdir())


def test_a_draw_a_configured_script_is_running_is_refused_by_a_run_before_its_ledger_is_touched(made):
    """A configured script's run (``run_plan``) and a campaign's run take the same lock, the draw's own."""
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    assert list(campaign.run(draw_ids=(0,))["fake-1/naive-rules-above"]) == [0]
    draw = campaign.root / "fake-1" / "naive-rules-above" / "draw-000"
    raw = draw / "raw_outputs.jsonl"
    # The script's block in flight: its raw rows appended, its ledger rows not yet, the draw's gate reopened.
    row = json.loads(raw.read_text().splitlines()[-1])
    with raw.open("a") as handle:
        handle.write(json.dumps({**row, "request_block_id": row["request_block_id"] + "-in-flight"}) + "\n")
    (draw / "completion.json").unlink()
    before = tree(draw)
    descriptor = _hold(draw_lock_path(draw))
    try:
        with pytest.raises(CampaignRunError, match=r"another process holds .*fake-1/naive-rules-above/\.draw-000\.lock: it is "
                                                   "running .*draw-000, and two runs of one draw"):
            campaign.run(draw_ids=(0,))
        assert tree(draw) == before
        assert list(campaign.run(draw_ids=(1,))["fake-1/naive-rules-above"]) == [1]
    finally:
        os.close(descriptor)
    assert not draw_lock_path(campaign.root / "fake-1" / "naive-rules-above" / "draw-001").exists()


def test_a_ruleset_changed_in_memory_or_built_by_hand_is_refused_before_any_write(made):
    campaign, _, rules = made
    before = tree(campaign.root)
    changed = Ruleset.load(rules.path)
    changed.rules[0].fields["rule"] = "another rule"
    with pytest.raises(CampaignError, match=r"holds rules other than its file's .* under the file's SHA-256"):
        campaign.condition("fake-1/naive-rules-above", rules=changed, judge=JudgeSpec.fake(1))
    forged = dataclasses.replace(rules, sha256="0" * 64)
    with pytest.raises(CampaignError, match=r"is not the file it names .*not the expected 0{64}"):
        campaign.condition("fake-1/naive-rules-above", rules=forged, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match=r"holds rules other than its file's"):
        evaluate(changed, campaign.scenarios.path, "fake", out=campaign.root, campaign_id="c", sampling=TWO)
    assert tree(campaign.root) == before


def test_a_population_changed_in_memory_or_built_by_hand_is_refused_before_a_root_is_made(tmp_path):
    scenarios, _ = inputs(tmp_path)
    changed = ScenarioSet.load(scenarios.path)
    changed.rows[0]["context"] = "added"
    with pytest.raises(CampaignError, match="holds rows other than its file's"):
        Campaign.create(tmp_path / "root", campaign_id="c", scenarios=changed, sampling=TWO)
    reordered = dataclasses.replace(scenarios, manifest=dataclasses.replace(
        scenarios.manifest, scenario_ids=scenarios.ids[::-1]))
    with pytest.raises(CampaignError, match="holds rows other than its file's"):
        Campaign.create(tmp_path / "root", campaign_id="c", scenarios=reordered, sampling=TWO)
    forged = dataclasses.replace(scenarios, manifest=dataclasses.replace(scenarios.manifest, source_sha256="0" * 64))
    with pytest.raises(CampaignError, match="is not the file it names"):
        Campaign.create(tmp_path / "root", campaign_id="c", scenarios=forged, sampling=TWO)
    assert not (tmp_path / "root").exists()


def test_rules_and_scenarios_changed_in_memory_after_binding_never_reach_a_run(made, monkeypatch):
    campaign, scenarios, rules = made
    condition = campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    rules.rules[0].fields["rule"] = "another rule"
    scenarios.rows[0]["context"] = "added"
    assert condition.rules.rows() == RULES and "context" not in campaign.scenarios.rows[0]
    started = count_starts(monkeypatch)
    condition.rules.rules[0].fields["rule"] = "another rule"
    with pytest.raises(CampaignError, match=r"condition 'fake-1/naive-rules-above''s rules \(rules.jsonl\) changed in memory "
                                            "since they were bound"):
        campaign.run()
    condition.rules.rules[0].fields["rule"] = RULES[0]["rule"]
    campaign.scenarios.rows[0]["context"] = "added"
    with pytest.raises(CampaignError, match="the scenarios of campaign 'c' changed in memory since they were bound"):
        campaign.run()
    assert started == [] and not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()
    del campaign.scenarios.rows[0]["context"]
    assert list(campaign.run()["fake-1/naive-rules-above"]) == [0, 1]
    judged = json.loads((campaign.root / "fake-1" / "naive-rules-above" / "draw-000" / "attempts.jsonl").read_text()
                        .splitlines()[0])
    assert judged["scenario_id"] in campaign.scenario_manifest.scenario_ids


def test_rules_changed_while_the_judge_starts_are_refused_before_any_draw(made, monkeypatch):
    campaign, _, rules = made
    condition = campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start

    def start(self):
        condition.rules.rules[1].fields["rule"] = "changed meanwhile"
        return real(self)

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match="changed in memory since they were bound"):
        campaign.run()
    assert not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()


@pytest.mark.parametrize("other_id, other_population", [("other", False), ("c", True)])
def test_a_held_campaign_whose_root_was_replaced_reads_nothing_of_it(made, tmp_path, other_id, other_population):
    campaign, scenarios, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    campaign.analyze()
    opened = Campaign.open(campaign.root)
    population = scenarios
    if other_population:
        (tmp_path / "other_scenarios.jsonl").write_text("".join(
            json.dumps({"prompt": f"other {i}", "response": f"answer {i}"}) + "\n" for i in range(3)))
        population = ScenarioSet.load(tmp_path / "other_scenarios.jsonl")
    other = Campaign.create(tmp_path / "other", campaign_id=other_id, scenarios=population, sampling=TWO)
    other.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    other.run()
    other.analyze()
    shutil.move(campaign.root, tmp_path / "moved")
    shutil.move(other.root, campaign.root)
    for held in (campaign, opened):
        for read in (held.status, held.result, lambda: held.result("fake-1/naive-rules-above"),
                     lambda: held.kappa("fake-1/naive-rules-above"), lambda: held.execution_record("fake-1/naive-rules-above")):
            with pytest.raises(CampaignError, match=rf"records campaign '{other_id}' over population .* not the "
                                                    r"campaign this object holds \('c' over"):
                read()
    assert Campaign.open(campaign.root).result("fake-1/naive-rules-above").campaign_id == other_id


@pytest.mark.parametrize("analysed", [True, False])
def test_a_root_replaced_while_a_result_is_read_is_refused(made, tmp_path, monkeypatch, analysed):
    campaign, scenarios, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    campaign.analyze()
    other = Campaign.create(tmp_path / "other", campaign_id="other", scenarios=scenarios, sampling=TWO)
    if analysed:
        other.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
        other.run()
        other.analyze()
    real = api.Campaign._estimates_file

    def estimates_file(self):
        if self is campaign:
            shutil.move(campaign.root, tmp_path / "moved")
            shutil.move(other.root, campaign.root)
        return real(self)

    monkeypatch.setattr(api.Campaign, "_estimates_file", estimates_file)
    with pytest.raises(CampaignError, match="records campaign 'other'"):
        campaign.result()


@pytest.mark.parametrize("fails", [False, True], ids=["read", "read refused"])
@pytest.mark.parametrize("read", ["status", "kappa"])
def test_a_root_replaced_while_its_gate_or_kappa_is_read_is_refused(made, tmp_path, monkeypatch, read, fails):
    campaign, scenarios, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    other = Campaign.create(tmp_path / "other", campaign_id="other", scenarios=scenarios, sampling=TWO)
    other.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    other.run()
    target = "campaign_completion" if read == "status" else "load_condition_outcomes"
    real = getattr(api, target)

    def meanwhile(*args, **kwargs):
        if campaign.root.exists() and not (tmp_path / "moved").exists():
            shutil.move(campaign.root, tmp_path / "moved")
            shutil.move(other.root, campaign.root)
            if fails:
                raise CampaignRunError("the read failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(api, target, meanwhile)
    with pytest.raises(CampaignError, match="records campaign 'other'"):
        campaign.status() if read == "status" else campaign.kappa("fake-1/naive-rules-above")


def test_a_judge_option_read_from_the_environment_is_read_once_when_bound(made, monkeypatch):
    campaign, _, rules = made
    judge = JudgeSpec.fake(1, max_new_tokens="${oc.env:RULEFORGE_TEST_CAP}")
    monkeypatch.delenv("RULEFORGE_TEST_CAP", raising=False)
    with pytest.raises(CampaignError, match="the judge's option 'max_new_tokens' does not resolve"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=judge)
    assert not (campaign.root / "fake-1").exists()
    monkeypatch.setenv("RULEFORGE_TEST_CAP", "64")
    condition = campaign.condition("fake-1/naive-rules-above", rules=rules, judge=judge)
    assert condition.spec.max_new_tokens == 64 and condition.judge.options.written()["max_new_tokens"] == "64"
    monkeypatch.setenv("RULEFORGE_TEST_CAP", "128")
    started = count_starts(monkeypatch)
    assert list(campaign.run()["fake-1/naive-rules-above"]) == [0, 1]
    assert [spec.identity().max_new_tokens for spec in started] == [64]


@pytest.mark.parametrize("escaped", [r"\${model_name}", r"\${oc.env:RULEFORGE_TEST_UNSET}"],
                         ids=["another key", "unresolvable"])
def test_a_value_that_reads_as_an_interpolation_again_is_refused_when_bound(made, monkeypatch, escaped):
    campaign, _, rules = made
    monkeypatch.delenv("RULEFORGE_TEST_ESCAPED", raising=False)
    monkeypatch.delenv("RULEFORGE_TEST_UNSET", raising=False)
    judge = JudgeSpec.fake(1, num_tries="${oc.env:RULEFORGE_TEST_ESCAPED}")
    monkeypatch.setenv("RULEFORGE_TEST_ESCAPED", escaped)
    with pytest.raises(CampaignError, match="do not read back as the values they resolved to"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=judge)
    assert not (campaign.root / "fake-1").exists()


def test_a_started_judge_that_is_not_the_one_registered_is_refused_before_any_draw(made, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start
    monkeypatch.setattr(JudgeSpec, "start", lambda self: real(self.with_options(max_new_tokens=128)))
    with pytest.raises(CampaignError, match=r"is not the judge it was registered with \(\{'max_new_tokens': "
                                            r"\(\d+, 128\)\}"):
        campaign.run()
    assert not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()


@pytest.mark.parametrize("field, value", [("max_new_tokens", 128), ("seed_capability", "unsupported")])
def test_a_started_judge_holding_other_generation_fields_than_its_identity_is_refused_before_any_draw(
        made, monkeypatch, field, value):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start

    def start(self):
        judge = real(self)
        setattr(judge, field, value)
        return judge

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match=rf"does not generate as its identity says \(\{{'{field}': \([^,]+, "
                                            rf"{value!r}\)\}}"):
        campaign.run()
    assert not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()


class _NoSpec:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        if name == "spec":
            raise AttributeError("it carries no spec")
        return getattr(self.inner, name)


@pytest.mark.parametrize("unreadable, reason", [("identity", "no identity"), ("spec", "it carries no spec")])
def test_a_started_judge_whose_identity_or_spec_cannot_be_read_is_refused(made, monkeypatch, unreadable, reason):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start

    def start(self):
        judge = real(self)
        if unreadable == "spec":
            return _NoSpec(judge)
        judge.identity = lambda: (_ for _ in ()).throw(JudgeError("no identity"))
        return judge

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match="the identity or request settings of the judge started cannot be read "
                                            rf"\({reason}\)"):
        campaign.run()
    assert not (campaign.root / "fake-1" / "naive-rules-above" / "draw-000").exists()


def test_a_root_whose_sampling_manifest_is_gone_is_refused_by_a_read(made):
    campaign, _, _ = made
    (campaign.root / SAMPLING_MANIFEST_FNAME).unlink()
    with pytest.raises(CampaignError, match="no longer records the campaign this object holds .*cannot read sampling "
                                            "manifest"):
        campaign.status()


def test_a_run_writes_a_snapshot_of_a_new_code_state(made, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    state = {"code_revision": "r", "code_dirty": True, "code_state_sha256": "f" * 64}
    monkeypatch.setattr(api, "code_state", lambda: dict(state))
    written = []
    monkeypatch.setattr(api, "write_code_snapshot", lambda root, code: written.append(code["code_state_sha256"]))
    campaign.run(draw_ids=[0])
    assert written == ["f" * 64]
    monkeypatch.setattr(api, "snapshot_path", lambda root, digest: campaign.root / SAMPLING_MANIFEST_FNAME)
    campaign.run(draw_ids=[1])
    assert written == ["f" * 64]


def test_analyze_refuses_an_existing_analysis_a_ruleset_size_root_and_a_root_of_collections_before_any_write(made):
    campaign, _, rules = made
    full = campaign.condition(rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(mode="full"))
    campaign.run()
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="registers no 'both' condition.*bank.analyze"):
        campaign.analyze()
    assert tree(campaign.root) == before and campaign.result().conditions[full.condition_id].estimates is None
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    campaign.analyze()
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=r"holds this root's analysis \(sha256 [0-9a-f]{64}\).*replace=True"):
        campaign.analyze()
    (campaign.root / SHARED_BANK_DESIGN_FNAME).write_text("{}")
    with pytest.raises(CampaignError, match="is a shared-bank campaign"):
        campaign.analyze(replace=True)
    (campaign.root / SHARED_BANK_DESIGN_FNAME).unlink()
    assert tree(campaign.root) == before
    with pytest.raises(ReadOnlyCampaign):
        Campaign.open(campaign.root).analyze(replace=True)


def test_analyze_refuses_an_analysis_directory_carried_elsewhere(made, tmp_path):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (campaign.root / AGGREGATE_DIRNAME).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(CampaignError, match="aggregate is a symlink"):
        campaign.analyze()
    assert not any(elsewhere.iterdir())


def test_result_without_an_analysis_of_a_both_condition_is_refused(made):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    assert campaign.result("fake-1/naive-rules-above").estimates is None
    with pytest.raises(CampaignError, match="holds no analysis"):
        campaign.result()
    with pytest.raises(CampaignError, match="registers no condition 'nope'"):
        campaign.result("nope")


def test_an_unreadable_analysis_is_refused(made):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    (campaign.root / AGGREGATE_DIRNAME).mkdir()
    (campaign.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).write_text("{")
    with pytest.raises(CampaignError, match="cannot be read as an analysis"):
        campaign.result("fake-1/naive-rules-above")


def test_a_style_family_s_contrast_and_a_forced_condition_s_same_call_are_returned_as_recorded(made):
    campaign, _, rules = made
    for style in ("naive-rules-above", "forced-grpo"):
        campaign.condition(f"fake-1/{style}", rules=rules, judge=JudgeSpec.fake(1),
                           settings=ConditionSettings(prompt=style))
    campaign.run()
    result = campaign.analyze(n_resamples=200)
    (contrast,) = result.contrasts
    assert isinstance(contrast, Contrast) and (contrast.a, contrast.b) == ("fake-1/naive-rules-above", "fake-1/forced-grpo")
    assert contrast.family == "style_contrasts" and contrast.family_size == 1 and contrast.p_holm is not None
    row = result.payload["style_families"][contrast.group]["contrasts"][0]
    assert contrast.estimate.value == row["estimate"] and contrast.stack_mismatch is None
    forced = result.conditions["fake-1/forced-grpo"].estimates
    assert forced.same_call.value == result.payload["forced_conditions"]["fake-1/forced-grpo"]["method_b_rate"][
        "estimate"]
    assert len(forced.per_rule) == 2 and result.conditions["fake-1/naive-rules-above"].estimates.per_rule == ()
    assert contrasts_of({}) == ()


def test_an_analysis_under_the_stack_override_discloses_it_per_condition(made):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    path = campaign.analyze(allow_stack_mismatch=True).path
    payload = json.loads(path.read_text())
    payload["stack_mismatches"] = [{"scope": "condition", "conditions": ["fake-1/naive-rules-above"], "reason": "r"},
                                   {"scope": "condition", "conditions": ["other"], "reason": "s"}]
    path.write_text(json.dumps(payload))
    result = Campaign.open(campaign.root).result()
    estimates = result.conditions["fake-1/naive-rules-above"].estimates
    assert estimates.overrides == {"allow_stack_mismatch": True}
    assert [record["reason"] for record in estimates.stack_mismatches] == ["r"] and len(result.stack_mismatches) == 2


def test_kappa_on_an_unstratified_condition_is_one_stratum_and_computed_without_a_write(made):
    campaign, scenarios, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    before = tree(campaign.root)
    value = Campaign.open(campaign.root).kappa("fake-1/naive-rules-above", n_resamples=200)
    assert isinstance(value, Kappa) and value.strata == {"field": None, "counts": {SINGLE_STRATUM: 3}}
    assert 0.0 <= value.p_observed <= 1.0 and value.to_dict()["strata"] == value.strata
    assert tree(campaign.root) == before


def test_kappa_is_refused_on_an_incomplete_condition_a_collection_or_an_unknown_one(made):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    full = campaign.condition(rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(mode="full"))
    campaign.run("fake-1/naive-rules-above", draw_ids=[0])
    with pytest.raises(CampaignError, match=r"is not complete \(draws \{1: 'in_progress'\}\)"):
        campaign.kappa("fake-1/naive-rules-above")
    with pytest.raises(CampaignError, match="is a 'full' condition"):
        campaign.kappa(full.condition_id)
    with pytest.raises(CampaignError, match="registers no condition 'nope'"):
        campaign.kappa("nope")


def test_kappa_on_a_partly_stratified_population_is_refused(tmp_path):
    rows = [{"prompt": f"prompt {i}", "response": f"response {i}", **({"source": "s"} if i else {})} for i in range(3)]
    path = tmp_path / "scenarios.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rules = write_rules(tmp_path / "rules.jsonl")
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=ScenarioSet.load(path), sampling=TWO)
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run()
    with pytest.raises(CampaignError, match="1 of 3 scenarios carry no 'source'"):
        campaign.kappa("fake-1/naive-rules-above")


def test_kappa_where_both_streams_always_say_yes_is_undefined(made, monkeypatch):
    campaign, _, rules = made
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    rewrite_answers(monkeypatch, lambda prompt, outcome: "yes")
    campaign.run()
    value = campaign.kappa("fake-1/naive-rules-above", n_resamples=200)
    assert (value.value, value.ci, value.sd, value.p_chance) == (None, None, None, 1.0)
    assert value.n_undefined_replicates == 200 and value.to_dict()["ci"] is None


def test_a_marker_less_root_is_read_by_every_read_path_and_written_by_none(tmp_path, capsys):
    built = configured_root(tmp_path / "built")
    copy = tmp_path / "copy"
    shutil.copytree(built, copy)
    scenarios = ScenarioSet.load(tmp_path / "built" / "scenarios.jsonl")
    rules = write_rules(tmp_path / "rules.jsonl")
    before = tree(copy)
    config = Campaign.open(copy).config
    with pytest.raises(CampaignError, match="did not create"):
        Campaign.create(copy, campaign_id="camp", scenarios=scenarios, sampling=config)
    with pytest.raises(CampaignError, match="did not create"):
        evaluate(rules, scenarios, JudgeSpec.fake(3), out=copy, campaign_id="camp",
                 sampling=SamplingSettings.from_config(config), condition_id="fake-3/naive-rules-above")
    with pytest.raises(ReadOnlyCampaign):
        Campaign.open(copy).analyze()
    with pytest.raises(CampaignError, match="holds no .ruleforge-campaign"):
        bank.analyze(copy)
    opened = Campaign.open(copy)
    assert opened.status()["state"] == STATE_COMPLETE
    result = opened.result()
    assert result.conditions["fake-3/naive-rules-above"].estimates.rate.value == json.loads(
        (copy / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_text())["conditions"]["fake-3/naive-rules-above"]["estimate"]
    assert opened.kappa("fake-3/naive-rules-above", n_resamples=200).strata["field"] is None
    assert tree(copy) == before and not (copy / api.LOCK_FNAME).exists()
    assert not (tmp_path / f".{copy.name}.lock").exists()


@pytest.mark.cluster
@pytest.mark.parametrize("name", ["fig5-gemma-2026-10-07-pop-a22af097", "fig5-gemma-2026-10-07"])
def test_a_copy_of_a_published_root_is_read_and_never_written(tmp_path, capsys, name):
    source = PUBLISHED / "sampling" / name
    if not source.is_dir():  # pragma: no cover - the published roots are on the cluster
        pytest.skip(f"{source} is not on this machine")
    copy = tmp_path / source.name
    files = ["sampling_manifest.json", "scenario_manifest.json", "completion.json", "aggregate/estimates.json"]
    condition_files = sorted(source.glob("**/condition.json"))
    assert condition_files, f"{source} holds no condition"
    for condition in condition_files:
        files.append(str(condition.relative_to(source)))
        for draw in sorted(condition.parent.glob("draw-*")):
            names = ("completion.json", "bundle_outcomes.jsonl", *(("evaluations.jsonl",) * (draw.name == "draw-000")))
            files.extend(str((draw / name).relative_to(source)) for name in names)
    for relative in files:
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, copy / relative)
    before = tree(copy)
    opened = Campaign.open(copy)
    assert opened.status()["state"] == STATE_COMPLETE
    conditions = opened.result().conditions
    (cid, collected), *_ = conditions.items()
    assert collected.estimates.estimates_sha256 == hashlib.sha256(
        (source / "aggregate" / "estimates.json").read_bytes()).hexdigest()
    assert json.loads((source / "sampling_manifest.json").read_text())["retry_scope"] == "component"
    assert {result.estimates.validity.retry_scope for result in conditions.values()} == {"component"}
    assert opened.kappa(cid, n_resamples=200).p_observed > 0
    population = ScenarioSet.load(REPO_ROOT / "data" / "populations" / "frozen_500_r3.jsonl")
    with pytest.raises(CampaignError, match=r"sampling sets \['label_rule'\], which a campaign of this library"):
        Campaign.create(copy, campaign_id=opened.campaign_id, scenarios=population, sampling=opened.config)
    with pytest.raises(CampaignError, match="did not create"):
        Campaign.create(copy, campaign_id=opened.campaign_id, scenarios=population,
                        sampling=SamplingSettings.from_config(opened.config))
    with pytest.raises(CampaignError, match="did not create"):
        evaluate(REPO_ROOT / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl", population, "fake",
                 out=copy, campaign_id=opened.campaign_id, sampling=SamplingSettings.from_config(opened.config))
    with pytest.raises(ReadOnlyCampaign):
        opened.analyze()
    with pytest.raises(CampaignError, match="holds no .ruleforge-campaign"):
        bank.analyze(copy)
    assert tree(copy) == before


def test_the_bank_s_conditions_are_planned_run_and_analysed_in_a_root_of_its_own(tmp_path):
    from tests.test_sampling_ruleset_size import make_design, write_subsets

    rule_path, subset_root, manifest_dir = write_subsets(tmp_path)
    design = make_design(rule_path, manifest_dir)
    scenarios, _ = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "ruleset-size", campaign_id="ruleset-size", scenarios=scenarios, sampling=SamplingSettings(
        draws=2, bootstrap=BootstrapSettings(n_resamples=150)))
    with pytest.raises(ReadOnlyCampaign):
        bank.freeze_design(Campaign.open(campaign.root), design)
    assert bank.freeze_design(campaign, design) == campaign.root / SHARED_BANK_DESIGN_FNAME
    assert bank.freeze_design(campaign, design) == campaign.root / SHARED_BANK_DESIGN_FNAME
    with pytest.raises(SharedBankDesignError, match="freezes a different shared-bank design"):
        bank.freeze_design(campaign, make_design(rule_path, manifest_dir, seeds=(0, 1, 2)))
    from ruleforge.rulesets import Ruleset

    for planned in bank.plan_conditions(design, "fake-1", subset_root=subset_root, bank_rule_set_path=rule_path):
        campaign.condition(planned.condition_id, rules=Ruleset.load(planned.rule_set_path), judge=JudgeSpec.fake(1),
                           settings=ConditionSettings(mode=planned.evaluation_mode, extra=planned.extra))
    campaign.run()
    with pytest.raises(CampaignError, match="is a shared-bank campaign"):
        campaign.analyze()
    payload = bank.analyze(campaign.root)
    assert list(payload["models"]) == ["fake-1"] and payload["n_scenarios"] == 3
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="holds this root's analysis"):
        bank.analyze(campaign.root)
    assert tree(campaign.root) == before
    again = bank.analyze(campaign.root, replace=True, n_resamples=150)
    assert again["models"]["fake-1"]["points"] == payload["models"]["fake-1"]["points"]
    assert (campaign.root / AGGREGATE_DIRNAME / SHARED_BANK_ESTIMATES_FNAME).is_file()


def test_a_pinned_root_is_never_created_or_written(tmp_path, monkeypatch):
    scenarios, rules = inputs(tmp_path)
    artifacts = tmp_path / "artifacts"
    (artifacts / "sampling").mkdir(parents=True)
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/pinned": ("leniency",)})
    for place in (artifacts / "sampling" / "pinned", artifacts / "sampling" / "pinned" / "inner", artifacts):
        with pytest.raises(CampaignError, match="a published run of leniency: it is never written"):
            Campaign.create(place, campaign_id="c", scenarios=scenarios, sampling=TWO)
    campaign = Campaign.create(artifacts / "sampling" / "mine", campaign_id="c", scenarios=scenarios, sampling=TWO)
    monkeypatch.setattr(api, "pinned_roots", lambda: {"artifact://sampling/mine": ("training",)})
    with pytest.raises(CampaignError, match="a published run of training"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    assert not (campaign.root / "fake-1").exists()


def _replace_root(campaign, tmp_path, *, with_marker: bool, published=None):
    campaign.root.rename(tmp_path / "moved")
    if with_marker:
        (tmp_path / "other").mkdir()
        scenarios, _ = inputs(tmp_path / "other")
        Campaign.create(campaign.root, campaign_id="other", scenarios=scenarios, sampling=TWO)
    else:
        shutil.copytree(published or configured_root(tmp_path / "published"), campaign.root)
    return tree(campaign.root)


@pytest.mark.parametrize("with_marker", [False, True], ids=["published-copy", "another-campaign"])
def test_a_root_replaced_under_a_held_campaign_is_never_written(tmp_path, monkeypatch, with_marker):
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=TWO)
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    before = _replace_root(campaign, tmp_path, with_marker=with_marker)
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started"))
    match = "replaced since" if with_marker else "holds no .ruleforge-campaign|replaced since"
    for operation in (lambda: campaign.condition("fake-2/naive-rules-above", rules=rules, judge=JudgeSpec.fake(2)),
                      lambda: campaign.start("fake-1/naive-rules-above"), lambda: campaign.run(),
                      lambda: campaign.analyze()):
        with pytest.raises(CampaignError, match=match):
            operation()
    assert tree(campaign.root) == before


@pytest.mark.parametrize("with_marker", [False, True], ids=["published-copy", "another-campaign"])
def test_a_root_replaced_while_the_judge_starts_is_never_written(tmp_path, monkeypatch, with_marker):
    """The judge's start (a local engine's build) can take minutes: the root is checked again once it returns."""
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=TWO)
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start
    before = []
    # Built before the run: a configured build inside it would itself be held to the run's checks.
    published = None if with_marker else configured_root(tmp_path / "published")

    def start(self):
        judge = real(self)
        before.append(_replace_root(campaign, tmp_path, with_marker=with_marker, published=published))
        return judge

    monkeypatch.setattr(JudgeSpec, "start", start)
    with pytest.raises(CampaignError, match="replaced since|holds no .ruleforge-campaign"):
        campaign.run()
    assert tree(campaign.root) == before[0]


@pytest.mark.parametrize("with_marker", [False, True], ids=["published-copy", "another-campaign"])
def test_a_root_replaced_during_a_run_is_refused_at_the_next_write(tmp_path, monkeypatch, with_marker):
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=TWO)
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    real = JudgeSpec.start
    before = []
    published = None if with_marker else configured_root(tmp_path / "published")

    def answer(prompt, outcome):
        if not before:
            before.append(_replace_root(campaign, tmp_path, with_marker=with_marker, published=published))

    monkeypatch.setattr(JudgeSpec, "start", lambda self: Rewritten(real(self), answer))
    with pytest.raises(CampaignError, match="replaced since|holds no .ruleforge-campaign"):
        campaign.run()
    assert tree(campaign.root) == before[0]
    moved = tmp_path / "moved" / "fake-1" / "naive-rules-above" / "draw-000"
    assert (moved / "request_blocks.jsonl").is_file() and not (moved / "attempts.jsonl").exists()


def test_a_root_whose_marker_was_rewritten_for_another_campaign_is_refused(made):
    campaign, _, rules = made
    marker = campaign.root / api.MARKER_FNAME
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "campaign_id": "other"}))
    with pytest.raises(CampaignError, match="holds campaign 'other', not 'c'"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    marker.unlink()
    with pytest.raises(CampaignError, match="holds no .ruleforge-campaign"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))


def test_a_root_that_can_no_longer_be_read_is_refused(made, monkeypatch):
    campaign, _, rules = made
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if Path(path) == campaign.root:
            raise PermissionError("denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(api.os, "stat", stat)
    with pytest.raises(CampaignError, match="cannot be read .*denied"):
        campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))


def test_a_campaign_created_from_a_config_keeps_it(tmp_path):
    scenarios, _ = inputs(tmp_path)
    config = SamplingConfig(campaign_mode="pilot", num_decoder_draws=3, max_semantic_attempts=4)
    campaign = Campaign.create(tmp_path / "r", campaign_id="c", scenarios=scenarios, sampling=config)
    assert campaign.config == config and campaign.sampling == SamplingSettings.from_config(config)
    assert campaign.scenarios == scenarios and campaign.scenarios is not scenarios
    assert campaign.scenarios.rows[0] is not scenarios.rows[0] and Campaign.open(campaign.root).scenarios is None


def test_an_outside_task_runs_through_its_own_renderer_and_its_locator_is_named(made, monkeypatch):
    from tests.test_tasks import DemoTask

    campaign, scenarios, rules = made
    condition = campaign.condition(rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(task=DemoTask))
    assert campaign.execution_record(condition.condition_id)["task_locator"] == "tests.test_tasks:DemoTask"
    prompts = []
    rewrite_answers(monkeypatch, lambda prompt, outcome: prompts.append(prompt["system_prompt"]))
    campaign.run()
    assert prompts and all(text.startswith("# Demo task\n") for text in prompts)
    command = api._run_command(campaign.root, condition, "fake:1", scenarios.path)
    assert command.endswith("--task tests.test_tasks:DemoTask")


def test_a_refusal_names_the_judge_and_levels_it_was_called_with(made, tmp_path):
    campaign, scenarios, rules = made
    condition = campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    command = api._run_command(campaign.root, condition, api._judge_flag(JudgeSpec.fake(1), JudgeSpec.fake(1)),
                               scenarios.path)
    assert command.endswith(f"--judge fake:1 --rules {rules.path} --scenarios {scenarios.path}")
    levels = tmp_path / "levels.jsonl"
    levels.write_text("".join(json.dumps({**rule, "rule_type": "user"}) + "\n" for rule in RULES))
    condition = campaign.condition("fake-1/h", rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(
        prompt="naive-hierarchy", rule_context="superiors", rule_levels=levels))
    command = api._run_command(campaign.root, condition, api._judge_flag(JudgeSpec.fake(1), JudgeSpec.fake(1)),
                               scenarios.path)
    assert f"--judge fake:1 --rules {rules.path} --scenarios {scenarios.path} --rule-levels {levels}" in command
    assert api._judge_flag("my-config", JudgeSpec.fake(1)) == "my-config"
    assert api._judge_flag(None, JudgeSpec.local(tmp_path / "weights")) == f"local:{tmp_path / 'weights'}"
    assert api._judge_flag(None, JudgeSpec.openai("gpt-x")) == "openai:gpt-x"
    assert api._judge_flag(None, JudgeSpec.from_config("gpt-5.6-luna-batch", batch_root=str(tmp_path))) is None


def test_analyze_refuses_a_screening_campaign_s_root(tmp_path):
    """A screening campaign (one draw over a candidate population) is never an estimate."""
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(campaign_mode="screening", draws=1))
    campaign.condition("fake-1/naive-rules-above", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignRunError, match="is a screening campaign: one draw over a candidate population"):
        campaign.analyze()
