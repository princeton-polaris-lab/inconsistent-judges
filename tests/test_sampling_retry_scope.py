from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import random
import re

import pytest

from ruleforge.evaluate.scoring import calculate_inconsistency
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.record_names import identity_condition_id
from ruleforge.evaluate.identity import derive_generation_seed
from ruleforge.evaluate.ledger import STATUS_ACCEPTED, STATUS_REJECTED, STATUS_TRANSPORT_FAILED, DrawLedger
from ruleforge.evaluate.identity import BUNDLE_OUTCOMES_FNAME
from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.evaluate.runner import (
    EVALUATIONS_FNAME,
    REQUEST_BLOCKS_FNAME,
    STATE_COMPLETE,
    STATE_INCOMPLETE_EXHAUSTED,
    CampaignRunError,
    DrawRunner,
    DrawSpec,
)
from ruleforge.judges.backend import GenerationOutcome
from tests.test_sampling_runner import (
    COMPONENTS,
    RULES,
    FakeBackend,
    build_runner,
    make_spec,
    read_jsonl,
    sent_block,
    sid_of,
    write_inputs,
)

COMPONENT = {"retry_scope": "component"}


@pytest.fixture
def inputs(tmp_path):
    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


def rows_for(runner, inputs, index):
    ledger = DrawLedger(runner.draw_dir, campaign_id="camp", condition_id=runner.spec.condition.condition_id,
                        decoder_draw_id=runner.spec.decoder_draw_id)
    return [r for r in ledger.load().records if r.scenario_id == sid_of(inputs, index)]


def attempt_of(block_id):
    return int(re.search(r"/attempt-(\d+)/", block_id).group(1))


def test_component_scope_regenerates_only_the_invalid_components(tmp_path, inputs, modelspec):
    backend = FakeBackend(script={(1, "Rule 1"): ["maybe", "yes"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, **COMPONENT)
    completion = runner.run()

    assert completion["state"] == STATE_COMPLETE and completion["rejected_attempts"] == 1
    assert backend.calls[(1, "Rule 1")] == 2
    assert all(n == 1 for key, n in backend.calls.items() if key != (1, "Rule 1"))
    assert [(attempt_of(b[0]), b[1]) for b in backend.blocks] == [(0, 9), (1, 1)]
    assert backend.blocks[1][2] == [derive_generation_seed(
        generation_seed_base=16, campaign_id="camp", condition_id=identity_condition_id("fake/naive", "naive"),
        scenario_id=sid_of(inputs, 1),
        decoder_draw_id=0, semantic_attempt_id=1, component_id="Rule 1",
    )]
    rows = rows_for(runner, inputs, 1)
    assert [(r.status, r.semantic_attempt_id, [c.component_id for c in r.components]) for r in rows] == [
        (STATUS_REJECTED, 0, list(COMPONENTS)), (STATUS_ACCEPTED, 1, ["Rule 1"]),
    ]
    assert {r.retry_scope for r in rows} == {"component"}
    assert completion["attempts_histogram"] == {"1": 2, "2": 1}
    assert completion["component_attempts_histogram"] == {"1": 8, "2": 1}

    # The views read the assembled bundle: the full set and Rule 2 from attempt 0, Rule 1 from attempt 1.
    outcome = read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)[1]
    assert outcome["semantic_attempt_id"] == 1 and outcome["rule_judgments"] == {"Rule 1": "yes", "Rule 2": "no"}
    assert outcome["full_set_judgment"] == "no" and outcome["contradiction"] == 1.0
    evaluation = read_jsonl(runner.draw_dir / EVALUATIONS_FNAME)[1]
    assert evaluation["evaluations"]["fake-judge"] == {
        "full_ruleset": {"No_Interpretation": "no"}, "Rule 1": {"No_Interpretation": "yes"},
        "Rule 2": {"No_Interpretation": "no"},
    }
    traced = read_jsonl(runner.draw_dir / JUDGE_OUTPUTS_FNAME)[1]["judge_outputs"]["fake-judge"]
    rule1 = traced["Rule 1"]["No_Interpretation"]
    assert [a["status"] for a in rule1["attempts"]] == [STATUS_REJECTED, STATUS_ACCEPTED]
    assert rule1["selected_attempt"] == 2 and rule1["judgment"] == "yes"
    full = traced["full_ruleset"]["No_Interpretation"]
    assert [a["status"] for a in full["attempts"]] == [STATUS_REJECTED]
    assert full["selected_attempt"] == 1 and full["attempts"][0]["candidates"][0]["accepted"] is True


def test_a_transport_failure_regenerates_every_unsettled_component_at_the_same_attempt(tmp_path, inputs, modelspec):
    failed = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")
    backend = FakeBackend(script={(2, "full_ruleset"): ["maybe", failed, "yes"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, **COMPONENT)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    rows = rows_for(runner, inputs, 2)
    assert [(r.status, r.semantic_attempt_id, len(r.components)) for r in rows] == [
        (STATUS_REJECTED, 0, 3), (STATUS_TRANSPORT_FAILED, 1, 1), (STATUS_ACCEPTED, 1, 1),
    ]
    assert completion["attempts_histogram"] == {"1": 2, "2": 1}


def test_a_component_lost_in_transport_inside_a_rejected_row_waits_for_the_next_attempt(tmp_path, inputs, modelspec):
    """The rejected full-set answer consumes attempt 0, so Rule 1 retries with attempt 1's seed."""
    failed = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")
    backend = FakeBackend(script={(1, "full_ruleset"): ["maybe", "no"], (1, "Rule 1"): [failed, "yes"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, **COMPONENT)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert (completion["rejected_attempts"], completion["transport_failed_attempts"]) == (1, 0)
    rows = rows_for(runner, inputs, 1)
    assert [(r.status, r.semantic_attempt_id, [c.component_id for c in r.components]) for r in rows] == [
        (STATUS_REJECTED, 0, list(COMPONENTS)), (STATUS_ACCEPTED, 1, ["full_ruleset", "Rule 1"]),
    ]
    assert rows[0].components[1].failure_reason == "transport_error"
    assert rows[1].components[1].generation_seed == derive_generation_seed(
        generation_seed_base=16, campaign_id="camp", condition_id=identity_condition_id("fake/naive", "naive"),
        scenario_id=sid_of(inputs, 1),
        decoder_draw_id=0, semantic_attempt_id=1, component_id="Rule 1",
    )
    # Rule 1 counts as generated at attempt 2 though it answered once.
    assert completion["component_attempts_histogram"] == {"1": 7, "2": 2}


def test_a_persistently_invalid_component_exhausts_its_bundle_alone(tmp_path, inputs, modelspec):
    backend = FakeBackend(script={(2, "Rule 2"): ["hmm"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, **COMPONENT)
    completion = runner.run()
    assert completion["state"] == STATE_INCOMPLETE_EXHAUSTED and completion["exhausted"] == [sid_of(inputs, 2)]
    assert (backend.calls[(2, "full_ruleset")], backend.calls[(2, "Rule 1")], backend.calls[(2, "Rule 2")]) == (1, 1, 5)
    assert completion["invalid_components"] == {"invalid_label": 5}
    assert completion["component_attempts_histogram"] == {"1": 6}  # two accepted bundles, three components each
    assert not (runner.draw_dir / BUNDLE_OUTCOMES_FNAME).exists()


def test_a_block_torn_mid_append_is_resubmitted_as_it_was_planned(tmp_path, inputs, modelspec):
    backend = FakeBackend(script={(0, "Rule 1"): ["maybe", "maybe", "no"], (1, "Rule 2"): ["maybe", "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, **COMPONENT)
    runner.run()
    recorded = len(runner.ledger.load().records)
    (block,) = [b for b in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME) if b["semantic_attempt_id"] == 1]
    assert block["scenario_ids"] == [sid_of(inputs, 0), sid_of(inputs, 1)]
    original = next(b for b in backend.blocks if b[0] == sent_block(block["request_block_id"]))
    assert original[1] == 2  # scenario 0's Rule 1 and scenario 1's Rule 2
    lines = runner.ledger.ledger_path.read_text().splitlines(keepends=True)
    torn = [i for i, line in enumerate(lines) if json.loads(line)["scenario_id"] == sid_of(inputs, 1)][-1]
    runner.ledger.ledger_path.write_text("".join(lines[:torn] + lines[torn + 1:]))
    (runner.draw_dir / BUNDLE_OUTCOMES_FNAME).unlink()

    submitted = len(backend.blocks)
    resumed = build_runner(tmp_path, inputs, modelspec, backend, **COMPONENT)  # the same judge, its answers continuing
    assert resumed.run()["state"] == STATE_COMPLETE
    assert backend.blocks[submitted:] == [original]
    assert len(resumed.ledger.load().records) == recorded


def test_the_runner_refuses_a_condition_whose_retry_scope_or_strict_think_is_not_the_campaigns(tmp_path, inputs, modelspec):
    manifest, rows = inputs
    spec = make_spec(manifest, **COMPONENT)
    for condition in (dataclasses.replace(spec.condition, retry_scope="scenario_bundle"),
                      dataclasses.replace(spec.condition, strict_think=True)):
        with pytest.raises(CampaignRunError, match="but the campaign declares 'component' and False"):
            DrawRunner(backend=FakeBackend(), task_spec=modelspec(), spec=DrawSpec(**{**spec.__dict__, "condition": condition}),
                       draw_dir=tmp_path / "d", scenario_rows=rows, rules=RULES)


def test_a_draw_recorded_under_one_scope_is_not_resumed_under_the_other(tmp_path, inputs, modelspec):
    build_runner(tmp_path, inputs, modelspec, FakeBackend(script={(0, "Rule 1"): ["maybe", "no"]})).run()
    with pytest.raises(CampaignRunError, match="different draw or condition"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend(), **COMPONENT).run()


def test_strict_think_reaches_the_validator(tmp_path, inputs, modelspec):
    unclosed = GenerationOutcome(text="yes", raw_response="<think>so yes", finish_reason="stop")
    backend = FakeBackend(script={(0, "Rule 2"): [unclosed, "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend, strict_think=True, **COMPONENT)
    assert runner.schema.strict_think is True
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["invalid_components"] == {"unclosed_think": 1}
    assert read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)[0]["rule_judgments"]["Rule 2"] == "no"


DISTRIBUTIONS = {
    "full_ruleset": {"yes": 0.30, "no": 0.35, "unsure": 0.05, "maybe": 0.30},
    "Rule 1": {"yes": 0.20, "no": 0.40, "unsure": 0.10, "maybe": 0.30},
    "Rule 2": {"yes": 0.10, "no": 0.45, "unsure": 0.05, "maybe": 0.40},
}
LABELS = ("yes", "no", "unsure")
MC_SCENARIOS = 500
MC_DRAWS = 4
MC_ATTEMPTS = 60
#: The bound on each compared frequency, in binomial standard errors.
Z = 4.0


class MonteCarloBackend(FakeBackend):
    """A seeded local judge: each answer drawn from its component's distribution by its own seed."""

    def __init__(self, *, correlated=False):
        super().__init__()
        self.correlated = correlated
        self.prompts = 0

    def _answer(self, component_id, seed, scenario_index, request_block_id):
        u = random.Random(seed).random()
        if self.correlated:
            key = f"{scenario_index}|{attempt_of(request_block_id)}".encode()
            bad = hashlib.sha256(key).digest()[0] < 192
            table = {"maybe": 0.60, "yes": 0.40} if bad else {"maybe": 0.02, "no": 0.93, "unsure": 0.05}
        else:
            table = DISTRIBUTIONS[component_id]
        for label, p in table.items():
            if u < p:
                return label
            u -= p
        return label  # pragma: no cover - float slack at u ~ 1

    def generate_once(self, prompts, *, temperature, max_new_tokens=None, seeds=None, request_block_id="", truncation=None):
        assert seeds is not None and len(seeds) == len(prompts)
        self.prompts += len(prompts)
        outcomes = []
        for entry, seed in zip(prompts, seeds):
            index, component_id = self.locate(entry)
            text = self._answer(component_id, seed, index, request_block_id)
            outcomes.append(GenerationOutcome(text=text, raw_response=f"<think>x</think>{text}", finish_reason="stop"))
        return outcomes


@pytest.fixture(scope="module")
def mc_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mc")
    path = directory / "scenarios.jsonl"
    path.write_text("".join(
        json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n" for i in range(MC_SCENARIOS)
    ), encoding="utf-8")
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


def accepted_bundles(root, inputs, modelspec, scope, *, correlated=False, draws=MC_DRAWS):
    """Every accepted bundle of ``draws`` draws under ``scope``: (full label, rule labels), and the prompts sent."""
    backend = MonteCarloBackend(correlated=correlated)
    bundles = []
    for draw in range(draws):
        runner = build_runner(root / scope, inputs, modelspec, backend, draw=draw, retry_scope=scope,
                              max_semantic_attempts=MC_ATTEMPTS)
        assert runner.run()["state"] == STATE_COMPLETE
        bundles += [(row["full_set_judgment"], row["rule_judgments"])
                    for row in read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)]
    return bundles, backend.prompts


def frequencies(bundles):
    """Accepted label shares per component, and the flat inconsistency rate."""
    n = len(bundles)
    shares = {("full_ruleset", label): sum(full == label for full, _ in bundles) / n for label in LABELS}
    for rule in ("Rule 1", "Rule 2"):
        shares.update({(rule, label): sum(rules[rule] == label for _, rules in bundles) / n for label in LABELS})
    shares["inconsistency"] = sum(calculate_inconsistency(full, list(rules.values()))["contradiction"] for full, rules in bundles) / n
    return shares


def expected_frequencies():
    """The accepted distribution the claim predicts: each component conditioned on validity, independently."""
    conditional = {
        cid: {label: table[label] / sum(table[x] for x in LABELS) for label in LABELS}
        for cid, table in DISTRIBUTIONS.items()
    }
    shares = {(cid, label): p for cid, table in conditional.items() for label, p in table.items()}
    shares["inconsistency"] = sum(
        conditional["full_ruleset"][f] * conditional["Rule 1"][a] * conditional["Rule 2"][b]
        * calculate_inconsistency(f, [a, b])["contradiction"]
        for f, a, b in itertools.product(LABELS, repeat=3)
    )
    return shares


def test_component_and_bundle_retry_accept_the_same_distribution(tmp_path, mc_inputs, modelspec):
    """Component and bundle retry accept the same distribution on the real runner, over a seeded synthetic judge."""
    bundle, bundle_prompts = accepted_bundles(tmp_path, mc_inputs, modelspec, "scenario_bundle")
    component, component_prompts = accepted_bundles(tmp_path, mc_inputs, modelspec, "component")
    n = len(bundle)
    assert n == len(component) == MC_SCENARIOS * MC_DRAWS
    expected = expected_frequencies()
    observed = {"scenario_bundle": frequencies(bundle), "component": frequencies(component)}
    for key, p in expected.items():
        se = math.sqrt(p * (1 - p) / n)
        for scope, shares in observed.items():
            assert abs(shares[key] - p) <= Z * se, (scope, key, shares[key], p)
        assert abs(observed["scenario_bundle"][key] - observed["component"][key]) <= Z * math.sqrt(2) * se, key
    # Component scope reached the same sample with far fewer requests (~1.4 vs ~3.4 per component).
    assert component_prompts < 0.6 * bundle_prompts


def test_correlated_components_break_the_equivalence_and_the_test_sees_it(tmp_path, mc_inputs, modelspec):
    """The negative control: the equivalence needs independent requests."""
    draws = 2
    bundle, _ = accepted_bundles(tmp_path, mc_inputs, modelspec, "scenario_bundle", correlated=True, draws=draws)
    component, _ = accepted_bundles(tmp_path, mc_inputs, modelspec, "component", correlated=True, draws=draws)
    gap = frequencies(component)[("Rule 1", "yes")] - frequencies(bundle)[("Rule 1", "yes")]
    p = expected_frequencies()[("Rule 1", "yes")]
    tolerance = Z * math.sqrt(2) * math.sqrt(p * (1 - p) / (MC_SCENARIOS * MC_DRAWS))
    assert gap > 3 * tolerance, (gap, tolerance)
