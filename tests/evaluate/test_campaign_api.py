from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ruleforge.evaluate import (
    EXECUTION_FNAME,
    LOCK_FNAME,
    MARKER_FNAME,
    Campaign,
    CampaignError,
    ConditionSettings,
    ReadOnlyCampaign,
)
from ruleforge.judges import Capabilities, JudgeIdentity, JudgeSpec, Provider, extension
from ruleforge.judges import providers
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.campaign import CONDITION_FNAME, SCENARIO_MANIFEST_FNAME, condition_payload
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME, SamplingConfig
from ruleforge.core.provenance import SNAPSHOT_DIRNAME
from ruleforge.scenarios import ScenarioSet
from ruleforge.judges.backend import ModelBackend
from tests.support.trees import tree_files
from tests.test_sampling_runner import RULES, write_inputs

PILOT = SamplingConfig(campaign_mode="pilot", num_decoder_draws=2)


@pytest.fixture
def inputs(tmp_path):
    scenarios = ScenarioSet.load(write_inputs(tmp_path))
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES), encoding="utf-8")
    return scenarios, Ruleset.load(rules)


def tree(root: Path) -> dict[str, str]:
    """Every file under ``root`` by relative path, with its content's digest."""
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tree_files(root)}


def luna_batch(batch_root, **changes) -> JudgeSpec:
    return JudgeSpec.from_config("gpt-5.6-luna-batch", batch_root=None if batch_root is None else str(batch_root),
                                 **changes)


def test_create_freezes_and_marks_a_new_root_and_rebinds_it_unchanged(tmp_path, inputs):
    scenarios, _ = inputs
    root = tmp_path / "runs" / "quick"
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=PILOT)
    assert campaign.root == root and campaign.writable and campaign.config == PILOT
    marker = json.loads((root / MARKER_FNAME).read_text())
    assert (marker["kind"], marker["campaign_id"], marker["scenario_manifest_sha256"]) == (
        "ruleforge-campaign", "quick", scenarios.manifest.manifest_sha256)
    assert (root / SCENARIO_MANIFEST_FNAME).is_file() and (root / SAMPLING_MANIFEST_FNAME).is_file()
    before = tree(root)
    again = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=PILOT)
    assert tree(root) == before and again.campaign_id == "quick"


def test_an_empty_directory_is_a_new_root(tmp_path, inputs):
    (tmp_path / "empty").mkdir()
    Campaign.create(tmp_path / "empty", campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert (tmp_path / "empty" / MARKER_FNAME).is_file()


def test_a_root_without_the_marker_is_refused_and_left_untouched(tmp_path, inputs):
    foreign = tmp_path / "published"
    Campaign.create(foreign, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    (foreign / MARKER_FNAME).unlink()
    before = tree(foreign)
    with pytest.raises(CampaignError, match="did not create"):
        Campaign.create(foreign, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert tree(foreign) == before


def test_a_root_holding_another_population_config_or_campaign_is_refused(tmp_path, inputs):
    scenarios, _ = inputs
    root = tmp_path / "root"
    Campaign.create(root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    before = tree(root)
    other_path = tmp_path / "other.jsonl"
    other_path.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    with pytest.raises(CampaignError, match="another population"):
        Campaign.create(root, campaign_id="c", scenarios=ScenarioSet.load(other_path), sampling=PILOT)
    with pytest.raises(CampaignError, match="describes a different campaign .*expected_decoder_draws"):
        Campaign.create(root, campaign_id="c", scenarios=scenarios,
                        sampling=SamplingConfig(campaign_mode="pilot", num_decoder_draws=3))
    with pytest.raises(CampaignError, match="holds campaign 'c', not 'd'"):
        Campaign.create(root, campaign_id="d", scenarios=scenarios, sampling=PILOT)
    assert tree(root) == before


def test_a_root_inside_another_campaign_root_is_refused(tmp_path, inputs):
    outer = tmp_path / "outer"
    Campaign.create(outer, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    (outer / MARKER_FNAME).unlink()
    before = tree(outer)
    with pytest.raises(CampaignError, match="is inside the campaign root"):
        Campaign.create(outer / "nested" / "deeper", campaign_id="d", scenarios=inputs[0], sampling=PILOT)
    assert tree(outer) == before and not (outer / "nested").exists()


def test_a_frozen_population_changed_under_its_marker_is_refused(tmp_path, inputs):
    root = tmp_path / "root"
    Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    ScenarioSet.load(other).manifest.write(root / SCENARIO_MANIFEST_FNAME)
    with pytest.raises(CampaignError, match="froze another population"):
        Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)


def test_an_interrupted_creation_is_completed_by_rebinding(tmp_path, inputs):
    root = tmp_path / "root"
    Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    (root / SAMPLING_MANIFEST_FNAME).unlink()
    (root / SCENARIO_MANIFEST_FNAME).unlink()
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    with pytest.raises(CampaignError, match="was created over another population"):
        Campaign.create(root, campaign_id="c", scenarios=ScenarioSet.load(other), sampling=PILOT)
    Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert (root / SAMPLING_MANIFEST_FNAME).is_file() and (root / SCENARIO_MANIFEST_FNAME).is_file()


@pytest.mark.parametrize("make, match", [
    (lambda root: root.write_text("x"), "is not a directory"),
    (lambda root: (root.mkdir(), (root / MARKER_FNAME).write_text("[]")), r"is not a campaign marker \(a JSON object"),
    (lambda root: (root.mkdir(), (root / MARKER_FNAME).write_text("{")), "cannot be read as a campaign marker"),
    (lambda root: (root.mkdir(), (root / MARKER_FNAME).write_text('{"kind": "x"}')),
     "is not a ruleforge-campaign marker$"),
])
def test_a_root_that_is_no_campaign_of_ours_is_refused(tmp_path, inputs, make, match):
    root = tmp_path / "root"
    make(root)
    with pytest.raises(CampaignError, match=match):
        Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)


def test_a_final_campaign_judges_its_approved_count(tmp_path, inputs):
    with pytest.raises(CampaignError, match="a final campaign judges 500 scenarios"):
        Campaign.create(tmp_path / "r", campaign_id="c", scenarios=inputs[0], sampling=SamplingConfig())


def test_open_reads_any_root_and_writes_nothing(tmp_path, inputs):
    scenarios, rules = inputs
    root = tmp_path / "root"
    Campaign.create(root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    (root / MARKER_FNAME).unlink()
    before = tree(root)
    opened = Campaign.open(root)
    assert (opened.campaign_id, opened.config, opened.writable) == ("c", PILOT, False)
    with pytest.raises(ReadOnlyCampaign):
        opened.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(ReadOnlyCampaign):
        opened.start("fake-1/naive")
    assert tree(root) == before
    with pytest.raises(CampaignError, match="is no campaign root"):
        Campaign.open(tmp_path)


def test_a_condition_is_registered_with_its_execution_record_and_rebound_without_a_write(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    condition = campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1),
                                   settings=ConditionSettings(prompt="naive"))
    directory = campaign.root / "fake-1" / "naive"
    assert json.loads((directory / EXECUTION_FNAME).read_text()) == {
        "batch_root": None, "task_locator": "modelspec", "request_settings": {}}
    assert json.loads((directory / CONDITION_FNAME).read_text()) == condition_payload(condition.spec)
    assert (condition.spec.prompt_style, condition.spec.model_type, condition.spec.sampling_truncation) == (
        "naive", "fake", "provider_default")
    before = tree(campaign.root)
    assert campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1)).spec == condition.spec
    assert tree(campaign.root) == before
    started = campaign.start("fake-1/naive")
    assert isinstance(started, ModelBackend) and started.model_name == "fake-1"
    with pytest.raises(CampaignError, match="different identity"):
        campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(2))


@pytest.mark.parametrize("settings, match", [
    (ConditionSettings(prompt="nope"), "no prompt style|nope"),
    (ConditionSettings(task="tests.test_tasks:ArchivedDemoTask", prompt="json"),
     "'json' of task 'demo-archived@1.0' is recognised only to read the records that ran it"),
    (ConditionSettings(mode="individual"), "mode 'individual' is not one of"),
    (ConditionSettings(rule_context="nearby"), "rule context 'nearby' is not one of"),
    (ConditionSettings(prompt="naive-hierarchy", rule_context="superiors"), "reads each rule's level from rule_levels"),
    (ConditionSettings(rule_context="superiors"), "belongs to the hierarchy prompt"),
    (ConditionSettings(rule_levels="levels.jsonl"), "reads each rule's level from rule_levels"),
])
def test_a_condition_its_task_or_campaign_would_refuse_is_refused_before_any_write(tmp_path, inputs, settings, match):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=match):
        campaign.condition("fake-1/x", rules=rules, judge=JudgeSpec.fake(1), settings=settings)
    assert tree(campaign.root) == before


def test_a_superiors_condition_names_its_levels_file(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    levels = tmp_path / "levels.jsonl"
    levels.write_text("".join(json.dumps({**rule, "rule_type": "user"}) + "\n" for rule in RULES))
    bad = tmp_path / "bad_levels.jsonl"
    bad.write_text(json.dumps({"id": "Rule 1", "rule": "other", "rule_type": "user"}) + "\n")
    settings = ConditionSettings(prompt="naive-hierarchy", rule_context="superiors", rule_levels=levels)
    condition = campaign.condition("fake-1/h", rules=rules, judge=JudgeSpec.fake(1), settings=settings)
    assert condition.spec.rule_levels_sha256 == hashlib.sha256(levels.read_bytes()).hexdigest()
    assert condition.levels.by_id == {rule["id"]: "user" for rule in RULES}
    with pytest.raises(CampaignError, match="levels file bad_levels.jsonl"):
        campaign.condition("fake-1/h2", rules=rules, judge=JudgeSpec.fake(1),
                           settings=ConditionSettings(prompt="naive-hierarchy", rule_context="superiors",
                                                      rule_levels=bad))


def test_a_ruleset_file_changed_while_its_levels_are_read_is_refused(tmp_path, inputs, monkeypatch):
    """The levels are read with the ruleset file, which must still be the bytes bound."""
    from ruleforge.evaluate import api

    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    levels = tmp_path / "levels.jsonl"
    levels.write_text("".join(json.dumps({**rule, "rule_type": "user"}) + "\n" for rule in RULES))
    real = api.load_rule_levels

    def load_rule_levels(rules_path, levels_path):
        # Other bytes, rows the levels still align with (an added field): what the condition did not record.
        Path(rules_path).write_text("".join(json.dumps({**rule, "note": "edited"}) + "\n" for rule in RULES))
        return real(rules_path, levels_path)

    monkeypatch.setattr(api, "load_rule_levels", load_rule_levels)
    with pytest.raises(CampaignError, match=r"rules.jsonl changed while its levels were read"):
        campaign.condition("fake-1/h", rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(
            prompt="naive-hierarchy", rule_context="superiors", rule_levels=levels))
    assert not (campaign.root / "fake-1").exists()


def test_an_unsaved_ruleset_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    unsaved = Ruleset(rules=rules.rules, name="r.jsonl", sha256=rules.sha256)
    with pytest.raises(CampaignError, match=r"write its rules to a JSONL file.*\(Ruleset.load\(path\)\)"):
        campaign.condition("fake-1/naive", rules=unsaved, judge=JudgeSpec.fake(1))


def test_the_fake_judge_is_refused_in_a_final_campaign(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"prompt": f"p{i}", "response": "r"}) + "\n" for i in range(500)))
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(rule) + "\n" for rule in RULES))
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=ScenarioSet.load(rows),
                               sampling=SamplingConfig())
    with pytest.raises(CampaignError, match="never a 'final' one"):
        campaign.condition("fake-1/naive", rules=Ruleset.load(rules), judge=JudgeSpec.fake(1))


def test_a_judge_that_reports_no_per_request_errors_is_refused(tmp_path, inputs, monkeypatch):
    class Quiet(Provider):
        def identity(self, spec, *, verify_weights=True):
            return JudgeIdentity(model_name=spec.model_name, model_type=spec.model_type,
                                 seed_capability="unsupported", max_new_tokens=8, extension=extension("q", "1", {}))

        def capabilities(self, spec):
            return Capabilities(seed="unsupported")

        def start(self, spec):  # pragma: no cover - never reached
            raise AssertionError

    monkeypatch.setattr(providers, "_REGISTERED", {"quiet": Quiet()})
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    with pytest.raises(CampaignError, match="per_request_errors"):
        campaign.condition("q/naive", rules=rules, judge=JudgeSpec.from_config({"model_type": "quiet",
                                                                                  "model_name": "q"}))


def test_a_condition_registered_without_its_execution_record_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    (campaign.root / "fake-1" / "naive" / EXECUTION_FNAME).unlink()
    rebound = Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    with pytest.raises(CampaignError, match="was registered without its execution.json"):
        rebound.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match="no condition 'fake-1/naive' registered through"):
        rebound.execution_record("fake-1/naive")


def test_an_execution_record_that_is_not_an_object_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    (campaign.root / "fake-1" / "naive" / EXECUTION_FNAME).write_text("[]")
    with pytest.raises(CampaignError, match="is not an execution record"):
        campaign.start("fake-1/naive")


def test_start_needs_a_judge_bound_in_this_process(tmp_path, inputs):
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    with pytest.raises(CampaignError, match="no judge is bound"):
        campaign.start("fake-1/naive")


def test_a_batch_judge_keeps_its_state_under_an_absolute_root(tmp_path, inputs, monkeypatch):
    scenarios, rules = inputs
    monkeypatch.chdir(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("luna/naive", rules=rules, judge=luna_batch("relative/batch"))
    record = campaign.execution_record("luna/naive")
    assert record == {"batch_root": str(tmp_path / "relative" / "batch"), "task_locator": "modelspec",
                      "request_settings": {"reasoning_summary": "auto"}}
    campaign.condition("luna/default", rules=rules, judge=luna_batch(None))
    assert campaign.execution_record("luna/default")["batch_root"] == str(campaign.root / "batch")


def test_a_batch_root_is_bound_where_its_transport_writes_and_two_places_in_one_spelling_are_refused(
        tmp_path, inputs, monkeypatch):
    from ruleforge.evaluate.api import bound_batch_root

    scenarios, rules = inputs
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "root"
    assert bound_batch_root(root, None) == bound_batch_root(root, "") == str(root / "batch")
    assert bound_batch_root(root, "~/b") == str(tmp_path / "home" / "b")
    assert bound_batch_root(root, "rel/./b/") == str(tmp_path / "rel" / "b")
    (tmp_path / "elsewhere" / "inner").mkdir(parents=True)
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe" / "link").symlink_to(tmp_path / "elsewhere" / "inner", target_is_directory=True)
    spelled = f"{tmp_path / 'safe'}/link/../batch"
    with pytest.raises(CampaignError, match=rf"names two places: {tmp_path / 'safe' / 'batch'} as written out, and "
                                            rf"{tmp_path / 'elsewhere' / 'batch'}"):
        bound_batch_root(root, spelled)
    campaign = Campaign.create(root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="names two places"):
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(spelled))
    assert tree(campaign.root) == before and not (campaign.root / "luna").exists()


def test_a_dotdot_after_a_directory_not_there_yet_names_two_places_for_every_writer_s_check(tmp_path, monkeypatch):
    from ruleforge.evaluate.api import bound_batch_root, refuse_pinned, refuse_protected, refuse_work_directory

    (tmp_path / "safe" / "there").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    for spelled in (f"{tmp_path / 'safe'}/new/../batch", "safe/new/../batch"):
        refusal = (rf"names two places: {re.escape(str(tmp_path / 'safe' / 'batch'))} as written out, and "
                   rf"{re.escape(os.path.realpath(tmp_path / 'safe' / 'new'))}, which a write handed it as given "
                   r"makes on the way when it makes its parents \(mkdir -p\), before it takes the '\.\.' after it "
                   r"\(new is not there yet\)")
        for check in (lambda: refuse_protected(spelled, "the place"), lambda: refuse_pinned(spelled, "the place"),
                      lambda: refuse_work_directory(spelled, "the place"),
                      lambda: bound_batch_root(tmp_path / "root", spelled)):
            with pytest.raises(CampaignError, match=refusal):
                check()
    assert sorted(tmp_path.rglob("*")) == before
    refuse_protected("safe/there/../batch", "the place")
    assert bound_batch_root(tmp_path / "root", "safe/there/../batch") == str(tmp_path / "safe" / "batch")


@pytest.mark.parametrize("below", [False, True])
def test_a_batch_root_at_the_campaign_lock_is_refused_before_anything_is_registered(tmp_path, inputs, below):
    from ruleforge.evaluate.sharding import campaign_lock_path

    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    lock = campaign_lock_path(campaign.root)
    assert lock.is_file() and lock.read_bytes() == b""
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=f"is, or lies in, {lock}, the campaign lock"):
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(lock / "x" if below else lock))
    assert tree(campaign.root) == before and lock.is_file() and lock.read_bytes() == b""
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path))


def test_an_unresolvable_batch_root_is_refused_naming_it(tmp_path, inputs, monkeypatch):
    scenarios, rules = inputs
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT", raising=False)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    refusal = (r"(?s)batch_root does not resolve \(.*RULEFORGE_BATCH_ROOT.*\); set the variable it reads, or give "
               rf"batch_root as null for {re.escape(str(campaign.root))}/batch$")
    written = Path(__file__).resolve().parents[2] / "configs" / "model" / "gpt-5.6-luna-batch.yaml"
    unresolved = OmegaConf.load(written)  # a mapping is taken as written: its batch_root reads the variable alone
    with pytest.raises(CampaignError, match=refusal):
        campaign.condition("luna/naive", rules=rules, judge=JudgeSpec.from_config(unresolved))
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(None))


def test_a_relocated_batch_root_is_refused_then_accepted_when_asked(tmp_path, inputs, monkeypatch):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "a"))
    started = []
    real_start = JudgeSpec.start
    monkeypatch.setattr(JudgeSpec, "start", lambda self: started.append(self) or real_start(self))

    moved = Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    moved.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "b"))
    with pytest.raises(CampaignError, match="relocate_batch_root=True"):
        moved.start("luna/naive")
    assert started == [] and moved.execution_record("luna/naive")["batch_root"] == str(tmp_path / "a")
    judge = moved.start("luna/naive", relocate_batch_root=True)
    assert judge.model_type == "openai_batch" and len(started) == 1
    assert moved.execution_record("luna/naive")["batch_root"] == str(tmp_path / "b")
    assert moved.start("luna/naive").model_type == "openai_batch"


def test_a_reasoning_summary_change_is_refused_even_with_relocation(tmp_path, inputs, monkeypatch):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "a"))
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started"))
    changed = Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    # Not identity, so the condition rebinds; but it changes every request's bytes.
    changed.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "a", reasoning_summary="detailed"))
    for relocate in (False, True):
        with pytest.raises(CampaignError, match="request settings .*'auto'.*'detailed'"):
            changed.start("luna/naive", relocate_batch_root=relocate)
    assert changed.execution_record("luna/naive")["request_settings"] == {"reasoning_summary": "auto"}


def test_a_root_aliased_into_a_published_root_is_refused(tmp_path, inputs):
    published = tmp_path / "published"
    Campaign.create(published, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    (published / MARKER_FNAME).unlink()
    (published / "aggregate").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    with pytest.raises(CampaignError, match=f"is inside the campaign root {published}"):
        Campaign.create(alias / "new", campaign_id="d", scenarios=inputs[0], sampling=PILOT)
    assert tree(published) == before and not (published / "aggregate" / "new").exists()


def test_a_root_reached_through_a_symlink_is_located_where_it_resolves(tmp_path, inputs):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    campaign = Campaign.create(tmp_path / "link" / "root", campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert campaign.root == tmp_path / "real" / "root" and (tmp_path / "real" / "root" / MARKER_FNAME).is_file()
    again = Campaign.create(tmp_path / "real" / "root", campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert again.root == campaign.root


def _published(tmp_path, inputs) -> Path:
    published = tmp_path / "published"
    Campaign.create(published, campaign_id="p", scenarios=inputs[0], sampling=PILOT)
    (published / MARKER_FNAME).unlink()
    (published / "aggregate").mkdir()
    return published


@pytest.mark.parametrize("condition_id", ["linked/new", "linked"])
def test_a_condition_directory_linked_into_a_published_root_is_refused(tmp_path, inputs, condition_id):
    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    (campaign.root / "linked").symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    with pytest.raises(CampaignError, match=f"{campaign.root / 'linked'} is a symlink"):
        campaign.condition(condition_id, rules=rules, judge=JudgeSpec.fake(1))
    assert tree(published) == before and not any((published / "aggregate").iterdir())


def test_a_condition_directory_that_is_a_campaign_root_of_its_own_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    nested = campaign.root / "nested"
    nested.mkdir()
    (nested / SAMPLING_MANIFEST_FNAME).write_text("{}")
    with pytest.raises(CampaignError, match=f"{nested} is a campaign root of its own inside {campaign.root}"):
        campaign.condition("nested/naive", rules=rules, judge=JudgeSpec.fake(1))
    assert [item.name for item in nested.iterdir()] == [SAMPLING_MANIFEST_FNAME]


def test_a_record_replaced_by_a_link_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    directory = campaign.root / "fake-1" / "naive"
    directory.mkdir(parents=True)
    for name in (EXECUTION_FNAME, CONDITION_FNAME):
        (directory / name).symlink_to(published / "aggregate" / name)
        with pytest.raises(CampaignError, match=f"{directory / name} is a symlink"):
            campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
        (directory / name).unlink()
    (campaign.root / MARKER_FNAME).rename(tmp_path / "marker")
    (campaign.root / MARKER_FNAME).symlink_to(tmp_path / "marker")
    with pytest.raises(CampaignError, match=f"{campaign.root / MARKER_FNAME} is a symlink"):
        Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    assert not any((published / "aggregate").iterdir())


def _planted_marker(tmp_path: Path, root: Path, scenarios: ScenarioSet, campaign_id: str) -> Path:
    target = tmp_path / "planted-marker.json"
    target.write_text(json.dumps({"kind": "ruleforge-campaign", "schema_version": 1, "campaign_id": campaign_id,
                                  "scenario_manifest_sha256": scenarios.manifest.manifest_sha256,
                                  "created_at": "2026-01-01T00:00:00+00:00"}))
    (root / MARKER_FNAME).symlink_to(target)
    return target


def test_a_foreign_root_marked_by_a_link_is_refused_before_its_lock_file_is_made(tmp_path, inputs):
    scenarios, _ = inputs
    published = _published(tmp_path, inputs)
    (published / LOCK_FNAME).unlink()
    _planted_marker(tmp_path, published, scenarios, "p")
    before = tree(published)
    with pytest.raises(CampaignError, match=f"{published / MARKER_FNAME} is a symlink"):
        Campaign.create(published, campaign_id="p", scenarios=scenarios, sampling=PILOT)
    assert tree(published) == before and not os.path.lexists(published / LOCK_FNAME)


def test_a_batch_root_inside_a_foreign_root_marked_by_a_link_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    _planted_marker(tmp_path, published, scenarios, "p")
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    place = published / "aggregate" / "batch"
    with pytest.raises(CampaignError, match=f"batch_root {place} is inside {published}, a campaign root this "
                                            "library did not create"):
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(place))
    assert not (campaign.root / "luna").exists()


def test_a_root_inside_a_directory_holding_a_dangling_manifest_link_is_refused(tmp_path, inputs):
    """A link at a parent's sampling manifest counts as the manifest, whether or not its target exists."""
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / SAMPLING_MANIFEST_FNAME).symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(CampaignError, match=f"is inside the campaign root {outer}"):
        Campaign.create(outer / "inner", campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert not (outer / "inner").exists()


def test_an_interrupted_creation_writes_no_snapshot_through_a_link(tmp_path, inputs):
    published = _published(tmp_path, inputs)
    root = tmp_path / "root"
    Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    (root / SAMPLING_MANIFEST_FNAME).unlink()
    if (root / SNAPSHOT_DIRNAME).exists():
        (root / SNAPSHOT_DIRNAME).rename(tmp_path / "snapshots")
    (root / SNAPSHOT_DIRNAME).symlink_to(published / "aggregate", target_is_directory=True)
    with pytest.raises(CampaignError, match=f"{root / SNAPSHOT_DIRNAME} is a symlink"):
        Campaign.create(root, campaign_id="c", scenarios=inputs[0], sampling=PILOT)
    assert not any((published / "aggregate").iterdir()) and not (root / SAMPLING_MANIFEST_FNAME).exists()


def test_a_lock_file_replaced_by_a_link_creates_nothing_where_it_points(tmp_path, inputs, monkeypatch):
    """Refused by name; and, should a link appear after that check, opened without following it."""
    from ruleforge.evaluate import api

    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    lock = campaign.root / LOCK_FNAME
    lock.unlink()
    lock.symlink_to(published / "aggregate" / "lock.json")
    with pytest.raises(CampaignError, match=f"{lock} is a symlink"):
        campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match=f"{lock} is a symlink"):
        Campaign.create(campaign.root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    monkeypatch.setattr(api, "_destination", lambda root, relative: root / relative)
    with pytest.raises(CampaignError, match=f"{lock} cannot be opened as this root's lock"):
        campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    assert not any((published / "aggregate").iterdir())


def test_a_root_moved_away_and_replaced_by_a_link_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.root.rename(tmp_path / "moved")
    campaign.root.symlink_to(published / "aggregate", target_is_directory=True)
    with pytest.raises(CampaignError, match=f"{campaign.root} now resolves to {published / 'aggregate'}"):
        campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    assert not any((published / "aggregate").iterdir())


def test_a_batch_root_inside_a_root_this_library_did_not_create_is_refused(tmp_path, inputs):
    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    (tmp_path / "alias").symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    for place in (published / "aggregate" / "batch", tmp_path / "alias" / "batch"):
        with pytest.raises(CampaignError, match=f"batch_root {place} is inside {published}, a campaign root this "
                                                "library did not create"):
            campaign.condition("luna/naive", rules=rules, judge=luna_batch(place))
    (campaign.root / "batch").symlink_to(published / "aggregate", target_is_directory=True)
    with pytest.raises(CampaignError, match=f"{campaign.root / 'batch'} is a symlink"):
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(None))
    assert tree(published) == before and not (campaign.root / "luna").exists()


def test_the_batch_root_rule_gives_a_judge_its_root_s_batch_directory_alone(tmp_path, inputs):
    from ruleforge.evaluate.api import refuse_batch_root

    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    other = Campaign.create(tmp_path / "other", campaign_id="o", scenarios=scenarios, sampling=PILOT)
    for place in (campaign.root / "batch", campaign.root / "batch" / "x", tmp_path / "outside" / "batch"):
        refuse_batch_root(campaign.root, place)
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(place))
        assert not os.path.lexists(place)
    before = tree(campaign.root)
    for place, refusal in ((campaign.root, "outside its batch/"), (campaign.root / "batches", "outside its batch/"),
                           (campaign.root / "luna" / "batch", "outside its batch/"),
                           (other.root / "batch", f"is, or is inside, {other.root}, another campaign's root"),
                           (other.root, f"is, or is inside, {other.root}, another campaign's root")):
        with pytest.raises(CampaignError, match=re.escape(refusal)):
            refuse_batch_root(campaign.root, place, what="the job's batch root")
        with pytest.raises(CampaignError, match=re.escape(refusal)):
            campaign.condition("luna/other", rules=rules, judge=luna_batch(place))
    assert tree(campaign.root) == before and not (campaign.root / "luna" / "other").exists()


def test_a_search_only_directory_holding_a_linked_draw_is_refused_before_anything_is_written(tmp_path, inputs):
    scenarios, rules = inputs
    published = _published(tmp_path, inputs)
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    condition = campaign.root / "fake-1" / "naive"
    (condition / "draw-000").symlink_to(published / "aggregate", target_is_directory=True)
    before = tree(published)
    condition.chmod(0o300)
    try:
        with pytest.raises(CampaignError, match=f"condition 'fake-1/naive''s directory {condition} cannot be walked "
                                                "whole"):
            campaign.run()
    finally:
        condition.chmod(0o700)
    assert tree(published) == before


def test_an_entry_another_job_removes_while_a_place_is_walked_is_no_refusal(tmp_path, monkeypatch):
    from ruleforge.evaluate.api import refuse_protected
    from tests.fake_scandir import vanishing_scandir

    draw = tmp_path / "sel" / "campaigns" / "p-stage1" / "m" / "draw-000"
    draw.mkdir(parents=True)
    (draw / ".state4.json.abc.tmp").write_text("{}")
    (draw.parent / ".draw-000.lock").mkdir()
    monkeypatch.setattr(os, "scandir", vanishing_scandir({".state4.json.abc.tmp"}, {draw.parent / ".draw-000.lock"}))
    refuse_protected(tmp_path / "sel", "the selection root", below=True)
    (draw / "outputs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(CampaignError, match=r"draw-000/outputs, below the selection root \S+, is a symlink"):
        refuse_protected(tmp_path / "sel", "the selection root", below=True)


def test_a_job_writing_the_tree_a_guard_walks_never_makes_it_refuse(tmp_path):
    from ruleforge.evaluate.api import _refuse_links, refuse_protected
    from tests.fake_scandir import concurrent_writer

    draw = tmp_path / "sel" / "campaigns" / "p-stage1" / "m" / "draw-000"
    draw.mkdir(parents=True)
    with concurrent_writer(draw) as rounds:
        for _ in range(300):
            _refuse_links(draw.parent, "condition 'm''s directory")
            refuse_protected(tmp_path / "sel", "the selection root", below=True)
        assert rounds[0] > 0
    (draw / "outputs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(CampaignError, match=r"draw-000/outputs, below condition 'm''s directory \S+, is a symlink"):
        _refuse_links(draw.parent, "condition 'm''s directory")


def test_a_place_that_cannot_be_checked_is_refused_and_one_below_a_file_is_none(tmp_path):
    from ruleforge.evaluate.api import _refuse_links, refuse_protected

    closed = tmp_path / "closed"
    (closed / "out").mkdir(parents=True)
    closed.chmod(0o600)
    try:
        with pytest.raises(CampaignError, match=f"the output {closed / 'out'} cannot be checked"):
            _refuse_links(closed / "out", "the output")
    finally:
        closed.chmod(0o700)
    (tmp_path / "file").write_text("x")
    refuse_protected(tmp_path / "file", "the output", below=True)
    refuse_protected(tmp_path / "file" / "below", "the output", below=True)


def test_start_refuses_a_destination_carried_elsewhere_since_registration(tmp_path, inputs, monkeypatch):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "mine", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "later" / "batch"))
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("started"))
    (campaign.root / "fake-1").rename(tmp_path / "elsewhere")
    (campaign.root / "fake-1").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(CampaignError, match=f"{campaign.root / 'fake-1'} is a symlink"):
        campaign.start("fake-1/naive")
    (tmp_path / "later").mkdir()
    (tmp_path / "later" / SAMPLING_MANIFEST_FNAME).write_text("{}")
    with pytest.raises(CampaignError, match="a campaign root this library did not create"):
        campaign.start("luna/naive", relocate_batch_root=True)


def test_an_unfinished_registration_is_completed_as_itself_and_never_replaced(tmp_path, inputs):
    """``execution.json`` written, ``condition.json`` not (a registration killed between them)."""
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "a"))
    directory = campaign.root / "luna" / "naive"
    recorded = (directory / EXECUTION_FNAME).read_bytes()
    (directory / CONDITION_FNAME).unlink()
    with pytest.raises(CampaignError, match="records an unfinished registration of 'luna/naive'"):
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "b"))
    assert (directory / EXECUTION_FNAME).read_bytes() == recorded and not (directory / CONDITION_FNAME).exists()
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(tmp_path / "a"))
    assert (directory / EXECUTION_FNAME).read_bytes() == recorded and (directory / CONDITION_FNAME).is_file()


HOLD_LOCK = """
import fcntl, sys
with open(sys.argv[1], "a") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    print("held", flush=True)
    sys.stdin.read()
"""


def test_a_registration_waits_for_the_process_holding_the_root_s_lock(tmp_path, inputs):
    import subprocess
    import sys
    import threading

    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    holder = subprocess.Popen([sys.executable, "-c", HOLD_LOCK, str(campaign.root / LOCK_FNAME)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        done = []
        worker = threading.Thread(target=lambda: done.append(
            campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))))
        worker.start()
        worker.join(timeout=1.0)
        assert worker.is_alive() and not (campaign.root / "fake-1" / "naive").exists()
    finally:
        holder.communicate("")
    worker.join(timeout=30)
    assert not worker.is_alive() and done and (campaign.root / "fake-1" / "naive" / CONDITION_FNAME).is_file()


def _fp32_merge(monkeypatch):
    monkeypatch.setattr("ruleforge.evaluate.campaign.judge_model_weights", lambda judge: SimpleNamespace(
        sha256="w" * 64, path=str(judge.model_path), saved_dtype="float32"))


@pytest.mark.parametrize("judge, sampling, prepare, match", [
    (lambda tmp_path: JudgeSpec.local(tmp_path / "merged"), PILOT, _fp32_merge, "holds float32 weights"),
    (lambda tmp_path: JudgeSpec.from_config({"model_type": "openai", "model_name": "m",
                                             "default_truncation": {"top_p": 0.9}}),
     PILOT, None, "only a local judge samples with a truncation"),
    (lambda tmp_path: JudgeSpec.from_config({"model_type": "openai", "model_name": "m",
                                             "adapter_path": str(tmp_path / "adapter")}),
     PILOT, None, "only a local judge serves a LoRA adapter"),
    (lambda tmp_path: JudgeSpec.fake(1), SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, strict_think=True),
     None, "strict_think needs a local thinking judge"),
], ids=["fp32-merge-without-dtype", "default-truncation-on-a-provider", "adapter-on-a-provider",
        "strict-think-without-think-tags"])
def test_a_judge_the_configured_preflight_refuses_is_refused_before_any_write(tmp_path, inputs, monkeypatch, judge,
                                                                             sampling, prepare, match):
    scenarios, rules = inputs
    if prepare is not None:
        prepare(monkeypatch)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=sampling)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=match):
        campaign.condition("j/naive", rules=rules, judge=judge(tmp_path))
    assert tree(campaign.root) == before


def test_condition_settings_are_hashable():
    assert hash(ConditionSettings(extra={"arm": "a"})) == hash(ConditionSettings(extra={"arm": "b"}))
    assert ConditionSettings(extra={"arm": "a"}) != ConditionSettings(extra={"arm": "b"})


@pytest.mark.parametrize("place", [
    "completion.json", ".ruleforge-campaign.lock", ".ruleforge-campaign", "sampling_manifest.json",
    "scenario_manifest.json", "aggregate/estimates.json", "code_snapshots", "code_snapshots/x",
    "luna/naive/.ruleforge-run.lock", "luna/naive/condition.json", "luna/naive/execution.json",
    "luna/naive/condition_stack.json", "luna/naive/draw-000", "luna/naive/.draw-001.lock/x",
])
def test_a_batch_root_on_a_record_of_its_campaign_is_refused_before_anything_is_written(tmp_path, inputs, place):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="a record of the campaign's own"):
        campaign.condition("luna/naive", rules=rules, judge=luna_batch(campaign.root / place))
    assert tree(campaign.root) == before and not (campaign.root / "luna").exists()
    draft = Campaign.draft(tmp_path / "other", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    with pytest.raises(CampaignError, match="a record of the campaign's own"):
        draft.register([("luna/naive", rules, luna_batch(draft.root / place), ConditionSettings())])
    assert not (tmp_path / "other").exists()
    campaign.condition("luna/naive", rules=rules, judge=luna_batch(campaign.root / "batch" / "x"))


@pytest.mark.parametrize("second", ["a/condition.json", "a/execution.json", "a/condition_stack.json", "a/draw-001",
                                    "aggregate", "completion.json", "batch/x"])
def test_a_condition_id_on_another_s_record_is_refused_before_any_member_is_written(tmp_path, inputs, second):
    """``a`` and ``a/condition.json``, or a draw's directory or a root's record."""
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="a record of the campaign's own"):
        campaign.register([("a", rules, JudgeSpec.fake(1), ConditionSettings()),
                           (second, rules, JudgeSpec.fake(2), ConditionSettings())])
    assert tree(campaign.root) == before
    campaign.condition("a", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match="a record of the campaign's own"):
        campaign.condition(second, rules=rules, judge=JudgeSpec.fake(2))


def test_a_condition_id_below_a_file_of_the_root_is_refused_never_an_os_error(tmp_path, inputs):
    scenarios, rules = inputs
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=PILOT)
    (campaign.root / "notes.txt").write_text("a note")
    before = tree(campaign.root)
    for condition_id, message in (("notes.txt/a", "below .*notes.txt, a file of the root"),
                                  ("notes.txt", "would be .*notes.txt, a file of the root")):
        with pytest.raises(CampaignError, match=message):
            campaign.condition(condition_id, rules=rules, judge=JudgeSpec.fake(1))
    assert tree(campaign.root) == before


HOLD_CAMPAIGN_LOCK = """
import sys
from ruleforge.evaluate.sharding import campaign_lock
with campaign_lock(sys.argv[1]):
    print("held", flush=True)
    sys.stdin.read()
"""


def test_create_is_refused_while_a_shard_merge_holds_the_campaign_lock_of_its_unmarked_root(tmp_path, inputs):
    import subprocess
    import sys

    scenarios, rules = inputs
    root = tmp_path / "root"
    holder = subprocess.Popen([sys.executable, "-c", HOLD_CAMPAIGN_LOCK, str(root)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")})
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(CampaignError, match="cannot be created or rebound now: another process holds"):
            Campaign.create(root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
        assert not (root / MARKER_FNAME).exists() and not (root / SAMPLING_MANIFEST_FNAME).exists()
    finally:
        holder.communicate("")
    campaign = Campaign.create(root, campaign_id="c", scenarios=scenarios, sampling=PILOT)
    assert (campaign.root / MARKER_FNAME).is_file()


def test_a_sampling_config_that_sets_what_the_library_does_not_carry_is_refused(tmp_path, inputs):
    with pytest.raises(CampaignError, match=r"sampling sets \['label_rule'\], which a campaign of this library does not "
                                            r"carry \(label_rule='lenient'\)"):
        Campaign.create(tmp_path / "r", campaign_id="c", scenarios=inputs[0],
                        sampling=SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, label_rule="lenient"))
