from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager

import pytest

from ruleforge.evaluate import Campaign, CampaignError, ConditionSettings, ReadOnlyCampaign, SamplingSettings
from ruleforge.evaluate import api
from ruleforge.judges import JudgeSpec
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME
from ruleforge.evaluate.campaign import load_conditions
from tests.evaluate.helpers import inputs, rewrite_answers, tree, write_rules
from tests.test_sampling_runner import RULES


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))


@pytest.fixture
def made(tmp_path):
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(draws=2))
    return campaign, scenarios, rules


def members(rules, *styles, mode="both"):
    return [(None, rules, JudgeSpec.fake(1), ConditionSettings(prompt=style, mode=mode)) for style in styles]


def test_a_draft_names_conditions_as_its_campaign_would_and_writes_nothing(tmp_path):
    scenarios, rules = inputs(tmp_path)
    draft = Campaign.draft(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    named = draft.prepare(rules=rules, judge=JudgeSpec.fake(1), segments="/prompt.naive")
    assert named.condition_id.startswith("fake-1/naive/prompt.naive-") and not (tmp_path / "root").exists()
    conditions = draft.register(members(rules, "naive", "encouraged"))
    assert [condition.condition_id[:-9] for condition in conditions] == ["fake-1/naive", "fake-1/encouraged"]
    assert not (tmp_path / "root").exists()
    with pytest.raises(ReadOnlyCampaign):
        draft.condition(rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(ReadOnlyCampaign):
        draft.sweep(members(rules, "naive"))
    made = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    assert made.prepare(rules=rules, judge=JudgeSpec.fake(1), segments="/prompt.naive").spec == named.spec


def test_register_names_every_member_before_it_writes_any(made):
    campaign, scenarios, rules = made
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="more than once"):
        campaign.register(members(rules, "naive", "naive"))
    with pytest.raises(CampaignError):
        campaign.register([*members(rules, "naive"), (None, rules, JudgeSpec.fake(1),
                                                       ConditionSettings(prompt="no_such_style"))])
    assert tree(campaign.root) == before
    conditions = campaign.register(members(rules, "naive", "encouraged"))
    assert sorted(load_conditions(campaign.root)) == sorted(condition.condition_id for condition in conditions)


def test_a_sweep_runs_every_member_then_analyses_the_root_once(made):
    campaign, scenarios, rules = made
    result = campaign.sweep(members(rules, "naive", "encouraged"))
    assert sorted(result.conditions) == sorted(load_conditions(campaign.root))
    assert all(item.estimates is not None for item in result.conditions.values())
    before = tree(campaign.root)
    assert campaign.sweep(members(rules, "naive", "encouraged")).payload == result.payload
    assert tree(campaign.root) == before


def test_a_sweep_of_no_member_is_refused(made):
    with pytest.raises(CampaignError, match="a sweep of no member"):
        made[0].sweep([])


def test_a_collection_sweep_is_not_analysed(made):
    campaign, scenarios, rules = made
    result = campaign.sweep(members(rules, "naive", "encouraged", mode="full"))
    assert result.path is None and all(item.estimates is None for item in result.conditions.values())
    assert not (campaign.root / AGGREGATE_DIRNAME).exists()


def test_a_sweep_beside_an_analysis_joins_no_condition_and_reads_none_incomplete(made):
    campaign, scenarios, rules = made
    campaign.sweep(members(rules, "naive"))
    with pytest.raises(CampaignError, match="not registered there: a condition joins an analysed root explicitly"):
        campaign.sweep(members(rules, "naive", "encouraged"))
    late = campaign.condition("late", rules=rules, judge=JudgeSpec.fake(2))
    with pytest.raises(CampaignError, match=r"conditions \['late'\] are not complete beside the analysis"):
        campaign.sweep([("late", rules, late.judge, late.settings)])


def test_a_member_out_of_attempts_is_refused_before_the_analysis(tmp_path, monkeypatch):
    scenarios, rules = inputs(tmp_path)
    campaign = Campaign.create(tmp_path / "root", campaign_id="c", scenarios=scenarios,
                               sampling=SamplingSettings(draws=2, max_attempts=1))
    rewrite_answers(monkeypatch, lambda prompt, outcome: "maybe" if "prompt 1\n" in prompt["system_prompt"] else None)
    with pytest.raises(CampaignError, match="out of attempts"):
        campaign.sweep(members(rules, "naive", "encouraged"))
    assert not (campaign.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).exists()


def test_a_reopened_root_writes_its_analysis_and_judges_nothing(made):
    campaign, scenarios, rules = made
    campaign.condition("c1", rules=rules, judge=JudgeSpec.fake(1)).run()
    reopened = Campaign.reopen(campaign.root)
    assert (reopened.campaign_id, reopened.writable, reopened.scenarios) == ("c", True, None)
    assert set(reopened.analyze().conditions) == {"c1"}
    with pytest.raises(CampaignError, match="held without its population's rows"):
        reopened.condition("c2", rules=rules, judge=JudgeSpec.fake(2))
    with pytest.raises(CampaignError, match="held without its population's rows"):
        reopened.run("c1")


def test_reopen_refuses_a_root_this_library_did_not_create_or_one_it_cannot_read(made, tmp_path):
    campaign, scenarios, rules = made
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with pytest.raises(CampaignError, match="did not create"):
        Campaign.reopen(foreign)
    (campaign.root / "scenario_manifest.json").unlink()
    with pytest.raises(CampaignError, match="does not hold a whole campaign's records"):
        Campaign.reopen(campaign.root)


def test_reopen_refuses_a_pinned_root(made, monkeypatch):
    campaign, scenarios, rules = made
    monkeypatch.setattr(api, "_refuse_pinned", lambda location: (_ for _ in ()).throw(CampaignError("pinned")))
    with pytest.raises(CampaignError, match="pinned"):
        Campaign.reopen(campaign.root)


def test_an_open_campaign_names_no_condition(made):
    campaign, scenarios, rules = made
    campaign.condition("c1", rules=rules, judge=JudgeSpec.fake(1))
    with pytest.raises(CampaignError, match="held without its population's rows"):
        Campaign.open(campaign.root).prepare(rules=rules, judge=JudgeSpec.fake(1))


def test_two_members_of_one_identity_are_refused_before_either_is_written(made):
    campaign, scenarios, rules = made
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=r"members 'a' and 'b' are one condition"):
        campaign.register([("a", rules, JudgeSpec.fake(1), ConditionSettings(prompt="naive")),
                           ("b", rules, JudgeSpec.fake(1), ConditionSettings(prompt="naive"))])
    assert tree(campaign.root) == before


def test_a_member_the_root_refuses_registers_none_of_the_others(made):
    campaign, scenarios, rules = made
    campaign.condition("b", rules=rules, judge=JudgeSpec.fake(1))
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="different identity"):
        campaign.register([("a", rules, JudgeSpec.fake(2), ConditionSettings()),
                           ("b", rules, JudgeSpec.fake(3), ConditionSettings())])
    assert tree(campaign.root) == before and set(load_conditions(campaign.root)) == {"b"}
    (campaign.root / "c").mkdir()
    (campaign.root / "c" / api.EXECUTION_FNAME).write_text('{"batch_root": "/elsewhere"}')
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="unfinished registration"):
        campaign.register([("a", rules, JudgeSpec.fake(2), ConditionSettings()),
                           ("c", rules, JudgeSpec.fake(3), ConditionSettings())])
    assert tree(campaign.root) == before


def test_a_shared_bank_root_registers_one_judge_configuration_per_model(made):
    from ruleforge.rulesets.subsets import WITH_REPLACEMENT
    from ruleforge.evaluate import bank

    campaign, scenarios, rules = made
    bank.freeze_design(campaign, bank.SharedBankDesign(subset_seeds=(0, 1), subset_sizes=(2, 3),
                                                    sampling_strategy=WITH_REPLACEMENT,
                                                    subset_manifest_sha256="0" * 64, source_ruleset_sha256="0" * 64))
    before = tree(campaign.root)
    pair = [("fake-1/subsets/s00/k02", rules, JudgeSpec.fake(1), ConditionSettings(prompt="naive")),
            ("fake-1/subsets/s01/k02", rules, JudgeSpec.fake(1), ConditionSettings(prompt="encouraged"))]
    mixed = (r"more than one judge configuration \(prompt_style: 'naive' \(fake-1/subsets/s00/k02\) vs 'encouraged' "
             r"\(fake-1/subsets/s01/k02\)")
    with pytest.raises(CampaignError, match=mixed):
        campaign.register(pair)
    draft = Campaign.draft(campaign.root, campaign_id="c", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    with pytest.raises(CampaignError, match=mixed):
        draft.register(pair)
    assert tree(campaign.root) == before
    campaign.condition(pair[0][0], rules=rules, judge=JudgeSpec.fake(1), settings=pair[0][3])
    with pytest.raises(CampaignError, match="more than one judge configuration"):
        campaign.condition(pair[1][0], rules=rules, judge=JudgeSpec.fake(1), settings=pair[1][3])
    assert set(load_conditions(campaign.root)) == {pair[0][0]} and not (campaign.root / pair[1][0]).exists()


def test_a_sweep_holds_every_member_to_its_execution_record_before_any_is_judged(made, monkeypatch):
    campaign, scenarios, rules = made
    campaign.condition("b", rules=rules, judge=JudgeSpec.fake(1))
    record = campaign.root / "b" / api.EXECUTION_FNAME
    record.write_text(record.read_text().replace('"request_settings": {}', '"request_settings": {"x": 1}'))
    monkeypatch.setattr(api, "run_plan", lambda *args, **kwargs: pytest.fail("a member was judged"))
    with pytest.raises(CampaignError, match="was recorded with request settings"):
        campaign.sweep([("a", rules, JudgeSpec.fake(2), ConditionSettings()),
                        ("b", rules, JudgeSpec.fake(1), ConditionSettings())])
    assert not list(campaign.root.glob("*/draw-*"))


def test_a_sweep_reads_an_analysis_as_evaluate_does_never_one_under_other_settings(made, monkeypatch):
    campaign, scenarios, rules = made
    swept = members(rules, "naive", "encouraged")
    campaign.sweep(swept)
    campaign.analyze(replace=True, ci_level=0.5, n_resamples=200)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=r"(?s)analysed under other settings.*bootstrap.*result\(\)"):
        campaign.sweep(swept)
    assert tree(campaign.root) == before
    campaign.analyze(replace=True)
    assert campaign.sweep(swept).path is not None
    payload = {"excluded_conditions": {"x": "dropped"}}
    monkeypatch.setattr(api.Campaign, "_estimates_file", lambda self: (payload, None))
    with pytest.raises(CampaignError, match=r"records excluded conditions or the fallback.*result\(\)"):
        campaign.sweep(swept)


def test_a_draft_checks_what_the_root_registers_and_registers_its_own_conditions_alone(made, tmp_path):
    campaign, scenarios, rules = made
    campaign.condition("b", rules=rules, judge=JudgeSpec.fake(1))
    draft = Campaign.draft(campaign.root, campaign_id="c", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match="different identity"):
        draft.register([("b", rules, JudgeSpec.fake(2), ConditionSettings())])
    prepared = draft.prepare("a", rules=rules, judge=JudgeSpec.fake(2))
    assert draft.register([prepared]) == (prepared,) and tree(campaign.root) == before
    with pytest.raises(CampaignError, match="prepared by another campaign object"):
        campaign.register([prepared])


def test_a_collection_only_sweep_never_returns_an_analysis_under_other_settings(made):
    campaign, scenarios, rules = made
    full = members(rules, "encouraged", mode="full")
    campaign.sweep([*members(rules, "naive"), *full])
    campaign.analyze(replace=True, ci_level=0.5, n_resamples=200)
    before = tree(campaign.root)
    with pytest.raises(CampaignError, match=r"(?s)analysed under other settings.*bootstrap.*result\(\)"):
        campaign.sweep(full)
    assert tree(campaign.root) == before
    campaign.analyze(replace=True)
    result = campaign.sweep(full)
    assert result.path is not None and sorted(item.mode for item in result.conditions.values()) == ["both", "full"]


def test_an_accepted_analysis_names_a_condition_and_is_none_on_a_root_without_one(made):
    campaign, scenarios, rules = made
    assert campaign.accepted_analysis(["anything"]) is None
    with pytest.raises(CampaignError, match="a read of no condition"):
        campaign.accepted_analysis([])


def test_an_analysis_another_process_wrote_while_a_sweep_ran_is_read_as_one_found(made, monkeypatch):
    campaign, scenarios, rules = made
    real = api.Campaign.analyze
    written = []

    def analyze(self, **kwargs):
        if not written:
            real(Campaign.reopen(self.root))
            written.append((self.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes())
        return real(self, **kwargs)

    monkeypatch.setattr(api.Campaign, "analyze", analyze)
    result = campaign.sweep(members(rules, "naive", "encouraged"))
    assert (campaign.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes() == written[0]
    assert result.path is not None and all(item.estimates is not None for item in result.conditions.values())


@pytest.fixture
def analysed_while_waiting(monkeypatch):
    """The root analysed by another process while this one waits for the root's lock."""
    real = api.Campaign._writing
    state = {"armed": False, "landed": False}

    @contextmanager
    def writing(self):
        if state["armed"] and not state["landed"]:
            state["landed"] = True
            Campaign.reopen(self.root).analyze()
        with real(self):
            yield

    monkeypatch.setattr(api.Campaign, "_writing", writing)
    return state


def test_a_registration_that_waited_for_the_lock_while_the_root_was_analysed_is_refused(made,
                                                                                         analysed_while_waiting):
    campaign, scenarios, rules = made
    campaign.condition("a", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run("a")
    analysed_while_waiting["armed"] = True
    with pytest.raises(CampaignError, match=r"conditions \['b'\] are not registered there: a condition joins"):
        campaign.register([("b", rules, JudgeSpec.fake(2), ConditionSettings())])
    assert analysed_while_waiting["landed"] and set(load_conditions(campaign.root)) == {"a"}
    assert not (campaign.root / "b").exists()


def test_an_evaluate_that_waited_for_the_lock_while_the_root_was_analysed_registers_nothing(made, tmp_path,
                                                                                            analysed_while_waiting):
    campaign, scenarios, rules = made
    campaign.condition("a", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run("a")
    analysed_while_waiting["armed"] = True
    other = write_rules(tmp_path / "other_rules.jsonl", [*RULES, {"id": "Rule 3", "rule": "rule 3 body"}])
    with pytest.raises(CampaignError, match="is not registered there: a condition is added beside an analysis"):
        api.evaluate(other, scenarios, "fake", out=campaign.root, campaign_id="c", draws=2)
    assert analysed_while_waiting["landed"] and set(load_conditions(campaign.root)) == {"a"}


@pytest.fixture
def replaced_after_check(monkeypatch):
    """Another process replaces the root's analysis."""
    real = api.Campaign.status
    state = {"armed": False}

    def status(self, **kwargs):
        if state["armed"]:
            state["armed"] = False
            state["read"] = (self.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes()
            Campaign.reopen(self.root).analyze(replace=True, ci_level=0.5, n_resamples=200)
        return real(self, **kwargs)

    monkeypatch.setattr(api.Campaign, "status", status)
    return state


def built_from(result, data: bytes) -> bool:
    """Whether ``result`` is built from the analysis ``data`` alone."""
    digest = hashlib.sha256(data).hexdigest()
    conditions = result.conditions.values() if hasattr(result, "conditions") else [result]
    return (getattr(result, "payload", json.loads(data)) == json.loads(data)
            and all(item.estimates.estimates_sha256 == digest for item in conditions if item.mode == "both"))


def test_a_read_returns_the_analysis_its_check_accepted_never_one_written_after(made, replaced_after_check):
    campaign, scenarios, rules = made
    swept = members(rules, "naive", "encouraged")
    ids = list(campaign.sweep(swept).conditions)
    estimates = campaign.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    for read in (lambda: campaign.accepted_analysis(ids), lambda: campaign.sweep(swept)):
        campaign.analyze(replace=True)
        replaced_after_check["armed"] = True
        result = read()
        assert built_from(result, replaced_after_check["read"]) and result.payload["bootstrap"]["ci_level"] != 0.5
        assert json.loads(estimates.read_bytes())["bootstrap"]["ci_level"] == 0.5


def test_an_analysis_returns_what_it_wrote_never_one_written_after_it_let_go_of_the_root(made, replaced_after_check):
    campaign, scenarios, rules = made
    campaign.sweep(members(rules, "naive"))
    replaced_after_check["armed"] = True
    result = campaign.analyze(replace=True, ci_level=0.8, n_resamples=200)
    assert built_from(result, replaced_after_check["read"]) and result.payload["bootstrap"]["ci_level"] == 0.8
    assert json.loads((campaign.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes())["bootstrap"][
        "ci_level"] == 0.5


@pytest.fixture
def registered_after_gate(monkeypatch, made):
    """Another process registers a condition the moment this one has read the campaign gate."""
    campaign, scenarios, rules = made
    real = api.Campaign.status
    state = {"armed": False}

    def status(self, **kwargs):
        gate = real(self, **kwargs)
        if state["armed"]:
            state["armed"] = False
            other = Campaign.create(self.root, campaign_id="c", scenarios=scenarios, sampling=SamplingSettings(draws=2))
            other.condition("late", rules=rules, judge=JudgeSpec.fake(9), settings=ConditionSettings(mode="full"))
        return gate

    monkeypatch.setattr(api.Campaign, "status", status)
    return state


@pytest.mark.parametrize("read", ["result", "accepted", "collected", "analyze", "evaluate"])
def test_a_condition_registered_between_the_gate_s_read_and_the_registry_s_is_left_out_never_a_key_error(
        made, registered_after_gate, read):
    campaign, scenarios, rules = made
    mode = "full" if read == "collected" else "both"
    ids = list(campaign.sweep(members(rules, "naive", mode=mode)).conditions)
    reads = {"result": campaign.result, "accepted": lambda: campaign.accepted_analysis(ids),
             "collected": lambda: campaign._accepted(ids, collected=True),
             "analyze": lambda: campaign.analyze(replace=True),
             "evaluate": lambda: api.evaluate(rules, scenarios, "fake", out=campaign.root, campaign_id="c", draws=2)}
    registered_after_gate["armed"] = True
    result = reads[read]()
    assert not registered_after_gate["armed"] and "late" in load_conditions(campaign.root)
    assert (result.condition_id == ids[0]) if read == "evaluate" else (list(result.conditions) == ids)


def test_a_condition_removed_while_the_root_is_read_is_refused_never_a_key_error(made, monkeypatch):
    campaign, scenarios, rules = made
    ids = list(campaign.sweep(members(rules, "naive", "encouraged", mode="full")).conditions)
    real = api.Campaign.status

    def status(self, **kwargs):
        gate = real(self, **kwargs)
        shutil.rmtree(self.root / ids[1])
        return gate

    monkeypatch.setattr(api.Campaign, "status", status)
    with pytest.raises(CampaignError, match=rf"conditions \['{ids[1]}'\] were removed or registered again"):
        campaign.result()
