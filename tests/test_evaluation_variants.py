from __future__ import annotations

import json
import logging

import pytest

from ruleforge.rulesets.variants import (
    build_rule_variants,
    write_implicated_sidecar,
)
from ruleforge.optimize.judging import run_judgments

RULES = [
    {"id": "Rule 1", "rule": "be helpful unless it would violate Rule 2"},
    {"id": "Rule 2", "rule": "do no harm"},
    {"id": "Meta-Rule 1", "rule": "Rule 2 beats Rule 1"},
]


class StubBackend:
    def __init__(self, model_type="local"):
        self.model_name = "stub"
        self.model_type = model_type
        self.seed = 16
        self.calls = 0

    def generate_with_validation(self, prompts, **_kwargs):
        self.calls += 1
        return ["no" for _ in prompts]


@pytest.fixture
def inputs(tmp_path):
    rules = tmp_path / "rules.jsonl"
    rules.write_text("".join(json.dumps(r) + "\n" for r in RULES), encoding="utf-8")
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        "".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) + "\n" for i in range(2)),
        encoding="utf-8",
    )
    return rules, scenarios


def test_build_rule_variants_skips_meta_rules_and_adds_implicated_context():
    isolated, isolated_map = build_rule_variants(RULES, "isolated")
    assert [v[0] for v in isolated] == ["Rule 1", "Rule 2"]
    assert all(v[2] is None for v in isolated) and isolated_map is None
    implicated, implication_map = build_rule_variants(RULES, "IMPLICATED")
    assert implication_map is not None
    assert implicated[0][2]
    assert implicated[0][1] == "Rule 1: be helpful unless it would violate Rule 2"


def test_build_rule_variants_refuses_an_unknown_context():
    with pytest.raises(ValueError, match="individual_rule_context"):
        build_rule_variants(RULES, "telepathic")


def test_write_implicated_sidecar_writes_nothing_for_the_isolated_variant(tmp_path):
    assert write_implicated_sidecar(tmp_path, "isolated", None) is None
    _, implication_map = build_rule_variants(RULES, "implicated")
    path = write_implicated_sidecar(tmp_path / "run", "implicated", implication_map)
    assert path is not None and json.loads(path.read_text())["implicated_rules"] == implication_map


def _run(inputs, tmp_path, modelspec, backend=None, **overrides):
    rules, scenarios = inputs
    kwargs = dict(
        scenario_set_path=scenarios,
        rule_set_path=rules,
        backend=backend or StubBackend(),
        task_spec=modelspec(),
        judge_types=["No_Interpretation"],
        evaluation_mode="both",
        output_path=tmp_path / "out" / "evaluations.jsonl",
    )
    kwargs.update(overrides)
    return run_judgments(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evaluation_mode": "some"}, "Invalid evaluation_mode"),
        ({"individual_rule_context": "telepathic"}, "Invalid individual_rule_context"),
        ({"judge_types": []}, "judge_types must be non-empty"),
    ],
)
def test_run_judgments_refuses_bad_arguments(inputs, tmp_path, modelspec, overrides, message):
    with pytest.raises(ValueError, match=message):
        _run(inputs, tmp_path, modelspec, **overrides)


def test_run_judgments_refuses_a_raw_path_that_aliases_the_output(inputs, tmp_path, modelspec):
    out = tmp_path / "out" / "evaluations.jsonl"
    with pytest.raises(ValueError, match="must differ"):
        _run(inputs, tmp_path, modelspec, output_path=out, raw_output_path=out)


def test_run_judgments_refuses_mismatched_resume_cursors(inputs, tmp_path, modelspec):
    out = tmp_path / "out" / "evaluations.jsonl"
    raw = tmp_path / "out" / "judge_outputs.jsonl"
    out.parent.mkdir()
    out.write_text('{"a": 1}\n', encoding="utf-8")
    raw.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot resume mismatched artifacts"):
        _run(inputs, tmp_path, modelspec, output_path=out, raw_output_path=raw, resume=True)


def test_run_judgments_refuses_an_empty_rule_set(inputs, tmp_path, modelspec):
    rules, scenarios = inputs
    rules.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No rules found"):
        _run(inputs, tmp_path, modelspec)


def test_a_batch_provider_with_a_small_flush_is_warned(inputs, tmp_path, modelspec, caplog):
    with caplog.at_level(logging.WARNING, logger="ruleforge.optimize.judging"):
        _run(inputs, tmp_path, modelspec, backend=StubBackend(model_type="openai_batch"), save_batch_size=10)
    assert "sequential completion windows" in caplog.text


@pytest.mark.parametrize("mode, keys", [("full", {"full_ruleset"}), ("individual", {"Rule 1", "Rule 2"})])
def test_single_sided_modes_judge_only_their_side(inputs, tmp_path, modelspec, mode, keys):
    _run(inputs, tmp_path, modelspec, evaluation_mode=mode)
    rows = [json.loads(line) for line in (tmp_path / "out" / "evaluations.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert set(rows[0]["evaluations"]["stub"]) == keys


def test_zero_scenarios_still_write_an_empty_output(inputs, tmp_path, modelspec, caplog):
    rules, scenarios = inputs
    scenarios.write_text("", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="ruleforge.optimize.judging"):
        _run(inputs, tmp_path, modelspec)
    out = tmp_path / "out" / "evaluations.jsonl"
    assert out.exists() and out.read_text() == ""
    assert "No scenarios evaluated" in caplog.text


def test_a_flush_per_scenario_writes_every_row(inputs, tmp_path, modelspec):
    backend = StubBackend()
    _run(inputs, tmp_path, modelspec, backend=backend, save_batch_size=1)
    rows = (tmp_path / "out" / "evaluations.jsonl").read_text().splitlines()
    assert len(rows) == 2 and backend.calls == 2


def test_individual_mode_over_meta_rules_only_queues_nothing(inputs, tmp_path, modelspec):
    rules, _ = inputs
    rules.write_text(json.dumps({"id": "Meta-Rule 1", "rule": "meta"}) + "\n", encoding="utf-8")
    backend = StubBackend()
    _run(inputs, tmp_path, modelspec, backend=backend, evaluation_mode="individual")
    assert backend.calls == 0


def test_a_resumed_run_judges_only_the_scenarios_after_its_complete_rows(inputs, tmp_path, modelspec):
    out = tmp_path / "out" / "evaluations.jsonl"
    _run(inputs, tmp_path, modelspec)
    first, second = out.read_text(encoding="utf-8").splitlines()
    out.write_text(first + "\n", encoding="utf-8")  # the run stopped after one flushed row
    backend = StubBackend()
    _run(inputs, tmp_path, modelspec, backend=backend, resume=True, save_batch_size=1)
    assert backend.calls == 1
    assert out.read_text(encoding="utf-8").splitlines() == [first, second]
