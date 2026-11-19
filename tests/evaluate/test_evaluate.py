from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path

import numpy as np
import pytest

from ruleforge.evaluate import (
    Campaign,
    CampaignError,
    CampaignResult,
    ConditionSettings,
    EvalResult,
    SamplingSettings,
    evaluate,
)
from ruleforge.cli.app import main
from ruleforge.evaluate import api
from ruleforge.judges import JudgeSpec
from ruleforge.judges.base import Capabilities
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, ESTIMATES_FNAME
from ruleforge.evaluate.identity import (
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_FULL,
    LABEL_RULE,
    RETRY_SCOPE_COMPONENT,
    RETRY_SCOPE_SCENARIO_BUNDLE,
    SCENARIO_EXCLUSION_KEY,
    BootstrapSettings,
)
from ruleforge.evaluate.sharding import campaign_lock_path
from ruleforge.evaluate.runner import STATE_COMPLETE, STATE_INCOMPLETE_EXHAUSTED
from tests.evaluate.helpers import (
    cluster_bootstrap,
    count_starts,
    inputs,
    recorded_cells,
    rewrite_answers,
    tree,
    write_rules,
)
from tests.test_sampling_runner import RULES


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)


@pytest.fixture
def setup(tmp_path):
    scenarios, rules = inputs(tmp_path)
    return scenarios, rules, tmp_path / "runs" / "quick"


def quick(setup, **kwargs):
    scenarios, rules, root = setup
    return evaluate(rules, scenarios, kwargs.pop("judge", "fake"), out=root, campaign_id="quick",
                    **{"draws": 2, **kwargs})


def forbid_running(monkeypatch):
    """Fail the test if anything starts a judge or runs a plan."""
    monkeypatch.setattr(JudgeSpec, "start", lambda self: pytest.fail("a judge was started"))
    monkeypatch.setattr(api, "run_plan", lambda *args, **kwargs: pytest.fail("run_plan was called"))


def test_evaluate_runs_a_fake_condition_end_to_end_and_its_estimate_is_the_hand_computed_one(setup):
    result = quick(setup)
    root = setup[2]
    assert isinstance(result, EvalResult) and result.root == root and result.campaign_id == "quick"
    assert result.condition_id.startswith("fake-1/naive-") and result.mode == "both"
    assert result.draws == {0: STATE_COMPLETE, 1: STATE_COMPLETE} and len(result.outcomes) == 2
    estimates = result.estimates
    assert (estimates.n_scenarios, estimates.draws, estimates.excluded_scenario) == (3, 2, None)
    assert (estimates.validity.label_rule, estimates.validity.retry_scope, estimates.validity.max_attempts) == (
        LABEL_RULE, RETRY_SCOPE_COMPONENT, 30)
    assert estimates.estimates_path == root / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    assert estimates.estimates_sha256 == hashlib.sha256(estimates.estimates_path.read_bytes()).hexdigest()
    assert estimates.attempts["attempts_histogram"] == {"1": 6} and estimates.same_call is None

    cells = recorded_cells(root / result.condition_id, 2, fake_seed=1)
    ids = setup[0].ids
    values = np.array([[cells[(sid, draw)] for draw in range(2)] for sid in ids])
    assert estimates.rate.value == pytest.approx(values.mean())
    assert estimates.rate.ci == pytest.approx(cluster_bootstrap(values, n_resamples=10_000, ci_level=0.95, seed=16))
    assert estimates.rate.basis == "scenario_cluster_bootstrap"


def test_a_repeated_evaluate_reads_the_complete_root_without_a_judge_a_run_or_a_write(setup, monkeypatch):
    first = quick(setup)
    before = tree(setup[2])
    forbid_running(monkeypatch)
    again = quick(setup)
    assert again == first and tree(setup[2]) == before
    assert Campaign.open(setup[2]).result(first.condition_id) == first


@pytest.mark.parametrize("other", ["campaign", "population"])
def test_an_analysis_another_root_wrote_is_refused_by_every_reader(setup, tmp_path, other):
    scenarios, rules, root = setup
    mine = quick(setup)
    source = scenarios.path
    if other == "population":
        source = tmp_path / "two.jsonl"
        source.write_text("".join(scenarios.path.read_text().splitlines(keepends=True)[:2]))
    foreign = evaluate(rules, source, "fake", out=tmp_path / "runs" / "other", campaign_id="other", draws=2)
    estimates = root / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    estimates.write_bytes(foreign.estimates.estimates_path.read_bytes())
    field = "campaign_id 'other'" if other == "campaign" else "scenario_manifest_sha256"
    for read in (lambda: Campaign.open(root).result(mine.condition_id), lambda: Campaign.open(root).result(),
                 lambda: quick(setup)):
        with pytest.raises(CampaignError, match=rf"{field}.*ruleforge analyze {root} --replace"):
            read()


def test_an_evaluate_differing_only_in_its_bootstrap_is_refused_naming_the_analysis_that_uses_it(setup):
    """The bootstrap enters a default campaign id, not the judgments: the refusal names how to analyse with it."""
    scenarios, rules, root = setup
    first = evaluate(rules, scenarios, "fake", out=root, draws=2)
    command = f"ruleforge analyze {root} --n-resamples 2000 --ci-level 0.9 --seed 17 --replace"
    with pytest.raises(CampaignError, match=re.escape(f"`{command}`")):
        evaluate(rules, scenarios, "fake", out=root, draws=2,
                 bootstrap=BootstrapSettings(n_resamples=2000, ci_level=0.9, seed=17))
    assert main(shlex.split(command)[1:]) == 0
    again = Campaign.open(root).result(first.condition_id)
    assert again.estimates.rate.value == first.estimates.rate.value and again.estimates.rate.ci != first.estimates.rate.ci
    assert Campaign.open(root).result().payload["bootstrap"] == {"n_resamples": 2000, "ci_level": 0.9, "seed": 17}


def test_a_root_of_another_campaign_without_its_sampling_record_is_refused_as_such(setup):
    scenarios, rules, root = setup
    first = evaluate(rules, scenarios, "fake", out=root, draws=2)
    (root / "sampling_manifest.json").unlink()
    with pytest.raises(CampaignError, match=rf"holds campaign '{first.campaign_id}', not 'other'; create this one in "
                                            r"a root of its own$"):
        Campaign.create(root, campaign_id="other", scenarios=scenarios, sampling=SamplingSettings(draws=2))


def test_an_analysis_another_process_wrote_meanwhile_is_read_as_one_found(setup, monkeypatch):
    real = api.Campaign.analyze
    calls = []

    def analyze(self, **kwargs):
        if not calls:
            calls.append("other")
            real(Campaign.create(self.root, campaign_id=self.campaign_id, scenarios=self.scenarios,
                                 sampling=self.sampling), **kwargs)
            written = (self.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes()
            calls.append(written)
        return real(self, **kwargs)

    monkeypatch.setattr(api.Campaign, "analyze", analyze)
    result = quick(setup)
    assert calls[0] == "other" and (setup[2] / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes() == calls[1]
    assert result.estimates is not None and result == Campaign.open(setup[2]).result(result.condition_id)


def test_a_read_returns_the_analysis_its_check_accepted_never_one_written_after(setup, monkeypatch):
    first = quick(setup)
    real = api.Campaign.status
    read = []

    def status(self, **kwargs):
        if not read:
            read.append((setup[2] / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes())
            Campaign.reopen(self.root).analyze(replace=True, ci_level=0.5, n_resamples=200)
        return real(self, **kwargs)

    monkeypatch.setattr(api.Campaign, "status", status)
    again = quick(setup)
    assert again == first and again.estimates.estimates_sha256 == hashlib.sha256(read[0]).hexdigest()
    assert json.loads((setup[2] / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_bytes())["bootstrap"]["ci_level"] == 0.5


def test_an_analysis_removed_between_its_refusal_and_its_read_is_refused(setup, monkeypatch):
    def analyze(self, **kwargs):
        raise api.AnalysisExists("another process analysed the root")

    monkeypatch.setattr(api.Campaign, "analyze", analyze)
    with pytest.raises(CampaignError, match="was removed while this call read it"):
        quick(setup)


@pytest.mark.parametrize("incomplete", [False, True])
def test_an_analysis_another_process_wrote_while_a_collection_ran_is_read_as_one_found(setup, monkeypatch,
                                                                                       incomplete):
    scenarios, rules, root = setup
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run("fake-1/naive")
    if incomplete:
        campaign.condition("fake-2/naive", rules=rules, judge=JudgeSpec.fake(2))
    real = api.Campaign.run

    def run(self, *args, **kwargs):
        found = real(self, *args, **kwargs)
        Campaign.reopen(self.root).analyze(allow_incomplete=incomplete)
        return found

    monkeypatch.setattr(api.Campaign, "run", run)
    if incomplete:
        with pytest.raises(CampaignError, match="records excluded conditions or the fallback"):
            quick(setup, mode=EVALUATION_MODE_FULL)
        return
    result = quick(setup, mode=EVALUATION_MODE_FULL)
    assert result.mode == EVALUATION_MODE_FULL and result.estimates is None
    assert result == Campaign.open(root).result(result.condition_id)


def test_the_retry_scope_is_the_condition_s_even_where_the_estimates_file_names_none(setup, tmp_path):
    first = quick(setup)
    path = first.estimates.estimates_path
    payload = json.loads(path.read_text())
    assert payload.pop("retry_scope") == RETRY_SCOPE_COMPONENT
    path.write_text(json.dumps(payload))
    read = Campaign.open(setup[2]).result(first.condition_id)
    assert read.estimates.validity.retry_scope == RETRY_SCOPE_COMPONENT
    assert read.estimates.estimates_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    scenarios, rules, _ = setup
    bundle = evaluate(rules, scenarios, "fake", out=tmp_path / "bundle", campaign_id="bundle", draws=2,
                      retry_scope=RETRY_SCOPE_SCENARIO_BUNDLE)
    assert "retry_scope" not in json.loads(bundle.estimates.estimates_path.read_text())
    assert bundle.estimates.validity.retry_scope == RETRY_SCOPE_SCENARIO_BUNDLE


def test_a_root_s_draw_count_is_fixed_when_it_is_created(setup):
    quick(setup)
    before = tree(setup[2])
    with pytest.raises(CampaignError, match="describes a different campaign .*expected_decoder_draws"):
        quick(setup, sampling=SamplingSettings(draws=4), draws=4)
    assert tree(setup[2]) == before


def test_evaluate_runs_only_the_draws_a_partly_run_root_is_missing(setup, monkeypatch):
    scenarios, rules, root = setup
    sampling = SamplingSettings(draws=4)
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=sampling)
    condition = campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    campaign.run("fake-1/naive", draw_ids=(0, 1))
    assert {draw: state for draw, state in campaign.result("fake-1/naive").draws.items()} == {
        0: STATE_COMPLETE, 1: STATE_COMPLETE, 2: "in_progress", 3: "in_progress"}
    started = count_starts(monkeypatch)
    plans = []
    real = api.run_plan
    monkeypatch.setattr(api, "run_plan", lambda plan, **kwargs: plans.append(plan) or real(plan, **kwargs))
    result = evaluate(rules, scenarios, JudgeSpec.fake(1), out=root, campaign_id="quick", sampling=sampling,
                      condition_id="fake-1/naive")
    assert [plan.decoder_draw_ids for plan in plans] == [(2, 3)] and len(started) == 1
    assert result.fingerprint == condition.spec.fingerprint and set(result.draws.values()) == {STATE_COMPLETE}
    assert result.estimates.draws == 4


def test_an_unsaved_ruleset_is_refused_before_any_root_is_made(setup):
    scenarios, rules, root = setup
    unsaved = Ruleset(rules=rules.rules, name="r.jsonl", sha256=rules.sha256)
    with pytest.raises(CampaignError, match=r"write its rules to a JSONL file.*\(Ruleset.load\(path\)\)"):
        evaluate(unsaved, scenarios, "fake", out=root, draws=2)
    assert not root.exists()


@pytest.fixture
def billed(monkeypatch):
    """Every judge bills its calls, the fake one among them."""
    capabilities = Capabilities(seed="local_deterministic", batch=False, per_request_errors=True, spends_money=True)
    monkeypatch.setattr(JudgeSpec, "capabilities", lambda self: capabilities)


SPEND = r"spends money on every call; pass confirm_spend=True \(--yes-spend\)"


def test_an_unconfirmed_paid_judge_is_refused_before_anything_is_written(setup, billed, monkeypatch):
    forbid_running(monkeypatch)
    with pytest.raises(CampaignError, match=SPEND):
        quick(setup)
    assert not setup[2].parent.exists()


def test_an_unconfirmed_paid_judge_with_draws_to_judge_in_a_root_writes_nothing_there(setup, billed, monkeypatch):
    """A condition the root registers with no draw judged, and one it does not register yet."""
    scenarios, rules, root = setup
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    for lock in [*root.parent.glob(".*.lock"), *root.rglob("*.lock")]:
        lock.unlink()
    before = tree(root.parent)
    forbid_running(monkeypatch)
    for condition_id in ("fake-1/naive", "fake-1/other"):
        with pytest.raises(CampaignError, match=SPEND):
            quick(setup, condition_id=condition_id)
    assert tree(root.parent) == before


def test_an_unconfirmed_paid_judge_meets_a_draw_recorded_for_another_condition_as_a_run_does(setup, billed,
                                                                                               monkeypatch):
    scenarios, rules, root = setup
    settings = dict(out=root, campaign_id="quick", draws=2, mode=EVALUATION_MODE_FULL)
    collected = evaluate(rules, scenarios, "fake", **settings, confirm_spend=True)
    gate = root / collected.condition_id / "draw-000" / "completion.json"
    gate.write_text(json.dumps({**json.loads(gate.read_text()), "condition_fingerprint": "0" * 64}))
    for lock in [*root.parent.glob(".*.lock"), *root.rglob("*.lock")]:
        lock.unlink()
    before = tree(root.parent)
    forbid_running(monkeypatch)
    with pytest.raises(CampaignError, match=r"draws \[0\] were recorded for another fingerprint"):
        evaluate(rules, scenarios, "fake", **settings)
    assert tree(root.parent) == before


@pytest.mark.parametrize("ran", [False, True], ids=["pending", "complete"])
def test_an_unconfirmed_paid_judge_asking_more_draws_than_its_root_froze_meets_the_root_s_refusal(setup, billed, ran,
                                                                                               monkeypatch):
    scenarios, rules, root = setup
    settings = dict(out=root, campaign_id="quick", condition_id="fake-1/full", mode=EVALUATION_MODE_FULL)
    if ran:
        evaluate(rules, scenarios, "fake", draws=2, confirm_spend=True, **settings)
    else:
        Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2)).condition(
            "fake-1/full", rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(mode=EVALUATION_MODE_FULL))
    for lock in [*root.parent.glob(".*.lock"), *root.rglob("*.lock")]:
        lock.unlink()
    before = tree(root.parent)
    forbid_running(monkeypatch)
    for draws in (3, 4):
        with pytest.raises(CampaignError, match="expected_decoder_draws"):
            evaluate(rules, scenarios, "fake", draws=draws, **settings)
    assert tree(root.parent) == before


def test_a_foreign_root_is_refused_before_an_unconfirmed_paid_judge_s_condition_is_named(setup, billed):
    root = setup[2]
    root.mkdir(parents=True)
    (root / "notes.txt").write_text("not a campaign")
    for judge in ("fake", JudgeSpec.fake(1)):
        with pytest.raises(CampaignError, match="holds a campaign root this library did not create"):
            quick(setup, judge=judge, prompt="no-such-prompt")
    assert sorted(path.name for path in root.parent.iterdir()) == ["quick"]


def test_an_unconfirmed_paid_judge_reads_a_root_that_leaves_it_nothing_to_judge(setup, billed, monkeypatch):
    """An analysed root, and a collection's root, which no analysis follows, once every draw is complete."""
    scenarios, rules, root = setup
    collection = (rules, scenarios, "fake")
    settings = dict(out=root.with_name("full"), campaign_id="quick", draws=2, mode=EVALUATION_MODE_FULL)
    analysed, collected = quick(setup, confirm_spend=True), evaluate(*collection, **settings, confirm_spend=True)
    forbid_running(monkeypatch)
    assert quick(setup) == analysed and evaluate(*collection, **settings) == collected
    with pytest.raises(CampaignError, match="has been analysed .* is not registered there"):
        quick(setup, prompt="forced")


def test_a_population_lacking_a_field_its_task_reads_is_refused_before_any_root_is_made(setup, tmp_path):
    """A Model Spec row with no ``response`` would be rendered with ``None`` in its place and judged."""
    _, rules, root = setup
    path = tmp_path / "no_response.jsonl"
    path.write_text("".join(json.dumps({"prompt": f"prompt {i}"}) + "\n" for i in range(3)))
    with pytest.raises(CampaignError, match=r"the scenarios \(.*no_response.jsonl\): scenario row 0 lacks "
                                            r"\['response'\], which task 'modelspec' reads"):
        evaluate(rules, path, "fake", out=root, campaign_id="quick", draws=2)
    assert not root.exists()


def _legal_rows(path: Path, **extra) -> Path:
    path.write_text("".join(json.dumps({"facts_of_case": f"facts {i}", "legal_question": f"question {i}",
                                        **{key: f"{value} {i}" for key, value in extra.items()}}) + "\n"
                            for i in range(3)))
    return path


@pytest.mark.parametrize("mode", [EVALUATION_MODE_FULL, EVALUATION_MODE_BANK])
def test_the_legal_task_s_scenarios_are_collected_without_a_prompt_and_response(tmp_path, monkeypatch, mode):
    rules = write_rules(tmp_path / "rules.jsonl")
    prompts = []
    rewrite_answers(monkeypatch, lambda prompt, outcome: prompts.append(prompt["system_prompt"]))
    result = evaluate(rules, _legal_rows(tmp_path / "legal.jsonl"), "fake", out=tmp_path / "legal",
                      campaign_id="legal", draws=2, task="legal", mode=mode)
    assert set(result.draws.values()) == {STATE_COMPLETE} and result.estimates is None and len(result.outcomes) == 2
    assert prompts and all("# Legal Question\nquestion" in text for text in prompts)


def test_the_legal_task_s_scenarios_are_refused_in_a_both_condition_before_any_write(tmp_path, monkeypatch):
    """A ``both`` draw's per-scenario summary is keyed by the prompt and response's hash."""
    rules = write_rules(tmp_path / "rules.jsonl")
    started = count_starts(monkeypatch)
    with pytest.raises(CampaignError, match=r"scenario row 0 has no string 'prompt' and 'response', which a 'both' "
                                            r"draw's per-scenario summary is keyed by .*task 'legal'.*'full' or "
                                            r"'individual_bank' condition"):
        evaluate(rules, _legal_rows(tmp_path / "legal.jsonl"), "fake", out=tmp_path / "legal", campaign_id="legal",
                 draws=2, task="legal")
    assert started == [] and not (tmp_path / "legal").exists()
    (tmp_path / "modelspec").mkdir()
    with pytest.raises(CampaignError, match=r"lacks \['facts_of_case', 'legal_question'\], which task 'legal'"):
        evaluate(rules, inputs(tmp_path / "modelspec")[0], "fake", out=tmp_path / "wrong", campaign_id="w",
                 draws=2, task="legal")
    keyed = _legal_rows(tmp_path / "keyed.jsonl", prompt="prompt", response="response")
    result = evaluate(rules, keyed, "fake", out=tmp_path / "keyed", campaign_id="keyed", draws=2, task="legal")
    assert set(result.draws.values()) == {STATE_COMPLETE} and result.estimates.n_scenarios == 3


def test_an_unknown_mode_is_refused_before_any_root_is_made(setup):
    with pytest.raises(CampaignError, match=r"mode 'bogus' is not one of"):
        quick(setup, mode="bogus")
    assert not setup[2].exists()


def test_a_campaign_lock_planted_beside_the_root_is_refused_before_the_judge_runs(setup, tmp_path, monkeypatch):
    """The analysis's campaign lock is the one file it writes outside its root."""
    from tests.evaluate.helpers import configured_root

    published = configured_root(tmp_path / "published")
    before = tree(published)
    scenarios, rules, root = setup
    root.parent.mkdir(parents=True)
    target = published / AGGREGATE_DIRNAME / "planted.json"
    campaign_lock_path(root).symlink_to(target)
    started = count_starts(monkeypatch)
    with pytest.raises(CampaignError, match="the campaign lock an analysis of .* is not a file of its own"):
        quick(setup)
    assert started == [] and not target.exists() and tree(published) == before
    campaign_lock_path(root).unlink()
    campaign_lock_path(root).mkdir()
    with pytest.raises(CampaignError, match="is not a file of its own"):
        quick(setup)
    campaign_lock_path(root).rmdir()
    assert quick(setup).estimates is not None


def test_evaluate_takes_paths_names_and_settings_by_keyword(setup, tmp_path, monkeypatch):
    scenarios, rules, _ = setup
    monkeypatch.setenv(api.ARTIFACT_ROOT_ENV, str(tmp_path / "artifacts"))
    result = evaluate(rules.path, scenarios.path, "fake:2", draws=2, prompt="naive", mode=EVALUATION_MODE_FULL)
    expected_id = api.default_campaign_id(scenarios.manifest, SamplingSettings(draws=2).config())
    assert result.root == tmp_path / "artifacts" / "sampling" / expected_id and result.campaign_id == expected_id
    assert result.condition_id.startswith("fake-2/naive/mode.full-") and result.estimates is None
    with pytest.raises(TypeError, match=r"unknown settings \['drawz'\]"):
        evaluate(rules, scenarios, "fake", drawz=2)


def test_the_default_root_is_under_the_working_directory_without_an_artifact_root(monkeypatch):
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)
    assert api.default_root("eval-x") == Path("ruleforge-runs") / "eval-x"


def test_the_default_root_is_under_the_artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    assert api.default_root("eval-x") == tmp_path / "artifacts" / "sampling" / "eval-x"


def test_the_artifact_root_s_variable_is_one_constant():
    import ruleforge.evaluate
    from ruleforge.core import paths

    assert api.ARTIFACT_ROOT_ENV is paths.ARTIFACT_ROOT_ENV is ruleforge.evaluate.ARTIFACT_ROOT_ENV


def test_a_collection_condition_is_collected_once_and_read_with_no_estimates(setup, monkeypatch):
    started = count_starts(monkeypatch)
    first = quick(setup, mode=EVALUATION_MODE_FULL)
    again = quick(setup, mode=EVALUATION_MODE_FULL)
    assert first.estimates is None and again == first and len(started) == 1
    assert not (setup[2] / AGGREGATE_DIRNAME).exists()
    assert Campaign.open(setup[2]).result(first.condition_id) == first
    whole = Campaign.open(setup[2]).result()
    assert isinstance(whole, CampaignResult) and whole.path is None and whole.contrasts == ()
    assert whole.conditions == {first.condition_id: first}


def test_a_root_with_a_both_and_a_full_condition_analyses_the_both_one_alone(setup, monkeypatch):
    scenarios, rules, root = setup
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    both = campaign.condition(rules=rules, judge=JudgeSpec.fake(1))
    full = campaign.condition(rules=rules, judge=JudgeSpec.fake(1), settings=ConditionSettings(mode="full"))
    campaign.run()
    analysed = campaign.analyze()
    payload = json.loads((root / AGGREGATE_DIRNAME / ESTIMATES_FNAME).read_text())
    assert list(payload["conditions"]) == [both.condition_id] and payload["excluded_conditions"] == {}
    assert campaign.result(full.condition_id).estimates is None
    assert set(analysed.conditions) == {both.condition_id, full.condition_id}
    assert analysed.conditions[both.condition_id].estimates is not None
    forbid_running(monkeypatch)
    before = tree(root)
    read = quick(setup, mode="full")
    assert read == campaign.result(full.condition_id) and tree(root) == before
    assert quick(setup) == campaign.result(both.condition_id) and tree(root) == before


def test_an_analysis_that_excluded_an_incomplete_condition_is_read_as_written_and_refused_by_evaluate(setup):
    scenarios, rules, root = setup
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    done = campaign.condition("fake-1/naive", rules=rules, judge=JudgeSpec.fake(1))
    partial = campaign.condition("fake-2/naive", rules=rules, judge=JudgeSpec.fake(2))
    campaign.run(done.condition_id)
    campaign.run(partial.condition_id, draw_ids=(0,))
    result = campaign.analyze(allow_incomplete=True)
    assert set(result.excluded_conditions) == {partial.condition_id}
    assert result.conditions[done.condition_id].estimates is not None
    excluded = result.conditions[partial.condition_id]
    assert excluded.estimates is None and excluded.excluded["draw_states"] == {"0": STATE_COMPLETE,
                                                                                "1": "in_progress"}
    before = tree(root)
    with pytest.raises(CampaignError, match=r"records excluded conditions or the fallback.*\.result\('fake-1/naive'\)"):
        evaluate(rules, scenarios, "fake", out=root, campaign_id="quick", condition_id="fake-1/naive", draws=2)
    assert tree(root) == before


def _invalid_for_scenario(index):
    return lambda prompt, outcome: "maybe" if f"prompt {index}\n" in prompt["system_prompt"] else None


def test_an_exhausted_draw_is_refused_naming_the_fallback_which_analyze_alone_applies(setup, monkeypatch):
    scenarios, rules, root = setup
    rewrite_answers(monkeypatch, _invalid_for_scenario(1))
    with pytest.raises(CampaignError, match=r"out of attempts .*`ruleforge analyze .* --exclude-exhausted fake-1 "
                                            rf"{scenarios.ids[1]}` .*\.result\('fake-1/naive'\)"):
        quick(setup, max_attempts=1, condition_id="fake-1/naive")
    campaign = Campaign.open(root)
    assert set(campaign.result("fake-1/naive").draws.values()) == {STATE_INCOMPLETE_EXHAUSTED}
    with pytest.raises(CampaignError, match="out of attempts"):
        quick(setup, max_attempts=1, condition_id="fake-1/naive")
    owner = Campaign.create(root, campaign_id="quick", scenarios=scenarios,
                            sampling=SamplingSettings(draws=2, max_attempts=1))
    analysed = owner.analyze(exclude_exhausted=("fake-1", scenarios.ids[1]))
    estimates = analysed.conditions["fake-1/naive"].estimates
    assert (estimates.n_scenarios, estimates.excluded_scenario) == (2, ("fake-1", scenarios.ids[1]))
    assert analysed.scenario_exclusion["scenario_id"] == scenarios.ids[1]
    assert SCENARIO_EXCLUSION_KEY in analysed.payload
    with pytest.raises(CampaignError, match="records excluded conditions or the fallback"):
        quick(setup, max_attempts=1, condition_id="fake-1/naive")


def test_a_draw_gated_out_of_attempts_with_a_scenario_pending_is_resumed(setup, monkeypatch):
    scenarios, rules, root = setup
    killed = []

    def answer(prompt, outcome):
        if "prompt 2\n" in prompt["system_prompt"] and not killed:
            killed.append(1)
            raise RuntimeError("killed mid-pass")
        return _invalid_for_scenario(1)(prompt, outcome)

    rewrite_answers(monkeypatch, answer)
    with pytest.raises(RuntimeError, match="killed mid-pass"):
        quick(setup, max_attempts=1, block_size=1, condition_id="fake-1/naive")
    gate_path = root / "fake-1/naive" / "draw-000" / "completion.json"
    gate = json.loads(gate_path.read_text())
    assert (gate["pending"], gate["accepted"]) == (1, 1)
    gate_path.write_text(json.dumps({**gate, "state": STATE_INCOMPLETE_EXHAUSTED}))

    with pytest.raises(CampaignError, match="out of attempts"):
        quick(setup, max_attempts=1, block_size=1, condition_id="fake-1/naive")
    gate = json.loads(gate_path.read_text())
    assert (gate["state"], gate["pending"], gate["accepted"]) == (STATE_INCOMPLETE_EXHAUSTED, 0, 2)


def test_an_exhausted_draw_beside_an_analysis_names_the_fallback_with_replace(setup, monkeypatch):
    scenarios, rules, root = setup
    quick(setup, max_attempts=1, condition_id="fake-1/naive")
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios,
                               sampling=SamplingSettings(draws=2, max_attempts=1))
    campaign.condition("fake-2/naive", rules=rules, judge=JudgeSpec.fake(2))
    rewrite_answers(monkeypatch, _invalid_for_scenario(0))
    campaign.run("fake-2/naive")
    with pytest.raises(CampaignError, match=rf"--exclude-exhausted fake-2 {scenarios.ids[0]} --replace`"):
        quick(setup, max_attempts=1, condition_id="fake-1/naive")


def test_a_forced_condition_out_of_attempts_is_told_to_collect_again_not_to_take_the_fallback(setup, monkeypatch):
    """The fallback does not serve a forced-style arm (its same-call rate reads every draw whole)."""
    rewrite_answers(monkeypatch, _invalid_for_scenario(1))
    with pytest.raises(CampaignError, match=r"the fallback does not serve judge 'fake-1''s arms \['fake-1/forced'\] "
                                            r"\(forced-style or policy arms.*more attempts \(max_attempts\)"):
        quick(setup, prompt="forced", max_attempts=1, condition_id="fake-1/forced")


def test_a_collection_condition_out_of_attempts_has_no_fallback(setup, monkeypatch):
    rewrite_answers(monkeypatch, _invalid_for_scenario(1))
    with pytest.raises(CampaignError, match="in a 'full' condition, which no analysis's fallback covers"):
        quick(setup, max_attempts=1, mode="full")


def test_an_analysis_under_another_bootstrap_is_refused_naming_replace(setup):
    scenarios, rules, root = setup
    first = quick(setup)
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    campaign.condition(first.condition_id, rules=rules, judge=JudgeSpec.fake(1))
    campaign.analyze(n_resamples=200, replace=True)
    with pytest.raises(CampaignError, match=r"other settings .*bootstrap \{'n_resamples': 200.*`ruleforge analyze "
                                            r".* --replace`"):
        quick(setup)


def test_an_analysis_missing_a_registered_condition_is_refused_naming_replace(setup):
    scenarios, rules, root = setup
    first = quick(setup)
    path = root / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    payload = json.loads(path.read_text())
    payload["conditions"][first.condition_id]["condition_fingerprint"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(CampaignError, match=rf"condition '{first.condition_id}' \(another fingerprint\)"):
        quick(setup)
    assert Campaign.open(root).result(first.condition_id).estimates is None
    payload["conditions"] = {}
    path.write_text(json.dumps(payload))
    with pytest.raises(CampaignError, match=r"\(absent\)"):
        quick(setup)
    payload["n_decoder_draws"] = 3
    path.write_text(json.dumps(payload))
    with pytest.raises(CampaignError, match=r"n_decoder_draws 3 \(the manifest's 2\)"):
        quick(setup)


def test_a_condition_registered_after_the_analysis_is_refused_until_it_is_run_and_the_root_reanalysed(setup):
    scenarios, rules, root = setup
    first = quick(setup)
    other = write_rules(root.parent / "other_rules.jsonl", [*RULES, {"id": "Rule 3", "rule": "rule 3 body"}])
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    late = campaign.condition(rules=other, judge=JudgeSpec.fake(1))
    assert late.condition_id.startswith("fake-1/naive-") and late.condition_id != first.condition_id
    with pytest.raises(CampaignError, match=r"in neither its conditions nor its excluded ones.*campaign run .*"
                                            r"analyze\(replace=True\)"):
        Campaign.open(root).result(late.condition_id)
    with pytest.raises(CampaignError, match="in neither its conditions"):
        Campaign.open(root).result()
    assert Campaign.open(root).result(first.condition_id) == first
    third = write_rules(root.parent / "third_rules.jsonl", [RULES[0], RULES[2]])
    before = tree(root)
    with pytest.raises(CampaignError, match=r"is not registered there.*`ruleforge campaign add .*`ruleforge campaign "
                                            r"run .* --judge fake --rules .*third_rules.jsonl --scenarios .*"):
        evaluate(third, scenarios, "fake", out=root, campaign_id="quick", draws=2)
    assert tree(root) == before
    with pytest.raises(CampaignError, match=rf"conditions \['{late.condition_id}'\] are not complete beside the "
                                            r"analysis.*campaign run .*other_rules.jsonl"):
        evaluate(other, scenarios, JudgeSpec.fake(1), out=root, campaign_id="quick", draws=2)
    campaign.run(late.condition_id)
    with pytest.raises(CampaignError, match="holds this root's analysis"):
        campaign.analyze()
    reanalysed = campaign.analyze(replace=True)
    assert set(reanalysed.conditions) == {first.condition_id, late.condition_id}
    assert evaluate(other, scenarios, "fake", out=root, campaign_id="quick", draws=2).estimates is not None


def test_a_joining_judge_no_command_line_rebuilds_is_given_no_resume_command(setup):
    scenarios, _, root = setup
    quick(setup)
    third = write_rules(root.parent / "third_rules.jsonl", [RULES[0], RULES[2]])
    with pytest.raises(CampaignError, match=r"is not registered there.*run it \(`ruleforge campaign run \S+ "
                                            r"fake-1/naive-\w+ \.\.\.`\)"):
        evaluate(third, scenarios, JudgeSpec.fake(1, max_new_tokens=64), out=root, campaign_id="quick", draws=2)


def test_several_incomplete_conditions_beside_an_analysis_are_each_named(setup):
    scenarios, rules, root = setup
    quick(setup)
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    for cid in ("fake-2/a", "fake-2/b"):
        campaign.condition(cid, rules=rules, judge=JudgeSpec.fake(2))
    with pytest.raises(CampaignError, match=r"\['fake-2/a', 'fake-2/b'\] are not complete .*`ruleforge campaign run "
                                            r".* CID \.\.\.` for each"):
        quick(setup)


def test_a_second_ruleset_registers_beside_the_first_under_its_own_name(setup):
    scenarios, rules, root = setup
    campaign = Campaign.create(root, campaign_id="quick", scenarios=scenarios, sampling=SamplingSettings(draws=2))
    first = campaign.condition(rules=rules, judge=JudgeSpec.fake(1))
    other = write_rules(root.parent / "other_rules.jsonl", [RULES[1], RULES[0]])
    second = campaign.condition(rules=other, judge=JudgeSpec.fake(1))
    assert first.condition_id == f"fake-1/naive-{first.spec.fingerprint[:8]}"
    assert second.condition_id == f"fake-1/naive-{second.spec.fingerprint[:8]}" != first.condition_id
    assert {path.parent.name for path in (root / "fake-1").glob("*/condition.json")} == {
        first.condition_id.split("/")[1], second.condition_id.split("/")[1]}
    assert campaign.condition(rules=other, judge=JudgeSpec.fake(1)).condition_id == second.condition_id


def test_a_root_this_library_did_not_create_is_refused_and_left_untouched(setup, tmp_path):
    from tests.evaluate.helpers import configured_root

    published = configured_root(tmp_path / "published")
    scenarios, rules, _ = setup
    before = tree(published)
    with pytest.raises(CampaignError, match="did not create"):
        evaluate(rules, scenarios, "fake", out=published, campaign_id="camp", draws=2)
    assert tree(published) == before


def test_bootstrap_settings_reach_the_analysis(setup):
    result = quick(setup, bootstrap=BootstrapSettings(n_resamples=200, seed=3))
    assert json.loads(result.estimates.estimates_path.read_text())["bootstrap"] == {
        "n_resamples": 200, "ci_level": 0.95, "seed": 3}
