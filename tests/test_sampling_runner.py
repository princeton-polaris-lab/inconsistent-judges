from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ruleforge.core.record_names import identity_condition_id
from ruleforge.evaluate.identity import (
    ConditionSpec,
    SamplingConfig,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import request_namespace
from ruleforge.evaluate.ledger import LEDGER_FNAME, DrawLedger, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_TRANSPORT_FAILED
from ruleforge.evaluate.identity import (
    BUNDLE_OUTCOMES_FNAME,
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    RUNS_FNAME,
)
from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.evaluate.runner import (
    EVALUATIONS_FNAME,
    PER_SCENARIO_FNAME,
    REQUEST_BLOCKS_FNAME,
    STATE_COMPLETE,
    STATE_INCOMPLETE_EXHAUSTED,
    STATE_IN_PROGRESS,
    SUMMARY_FNAME,
    CampaignRunError,
    DrawRunner,
    DrawSpec,
    TransportExhausted,
    draw_lock_path,
    draw_run_lock,
)
from ruleforge.evaluate.stack import (
    CONDITION_STACK_FNAME,
    JUDGED_ON_KEY,
    condition_signatures,
    judgment_runtimes,
    judgment_signature,
    mixed_conditions,
    recorded_runtime,
    stack_digest,
)
from ruleforge.judges import Capabilities
from ruleforge.judges.backend import BATCH_MODEL_TYPES, GenerationOutcome
from ruleforge.evaluate import runner as runner_module
from ruleforge.judges.batch_api import (
    BatchExecutionError,
    BatchSpendNotConfirmed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RULES = [
    {"id": "Rule 1", "rule": "rule 1 body"},
    {"id": "Rule 2", "rule": "rule 2 body"},
    {"id": "Meta-Rule 1", "rule": "meta body"},
]
N_SCENARIOS = 3
COMPONENTS = ("full_ruleset", "Rule 1", "Rule 2")


def sent_block(block_id: str) -> str:
    """The id a judge is sent for the ``fake/naive`` condition's request block ``block_id``."""
    return block_id.replace("camp/fake/naive/", f"camp/{identity_condition_id('fake/naive', 'naive')}/", 1)


_PROMPT_RE = re.compile(r"# Original Prompt Given to the AI\n(.+?)\n", re.S)
_RULESET_RE = re.compile(r"# Rule Set Used for Evaluation\n(.+?)\n\n", re.S)


def forced(full="no", rules=None):
    return json.dumps(
        {
            "rule_judgments": {"Rule 1": "no", "Rule 2": "no"} if rules is None else rules,
            "preempted_rules": [],
            "full_set_judgment": full,
        }
    )


class FakeBackend:
    """Answers from ``script[(scenario_index, component_id)]`` call by call."""

    def __init__(self, *, style="naive", script=None, seed_capability="local_deterministic",
                 model_type="local", temperature=1.0, fixed_temperature=None,
                 fail_blocks=(), per_request_errors=True, request_settings=None):
        self.model_name = "fake-judge"
        self.model_type = model_type
        self.seed_capability = seed_capability
        self.capabilities = Capabilities(
            seed=seed_capability, batch=model_type in BATCH_MODEL_TYPES, per_request_errors=per_request_errors,
            spends_money=model_type not in ("local", "fake"))
        self.temperature = temperature
        self.fixed_temperature = fixed_temperature
        self.style = style
        self.script = {key: list(value) for key, value in (script or {}).items()}
        self.calls: dict[tuple[int, str], int] = {}
        self.blocks: list[tuple[str, int, list]] = []
        self.truncations: list = []
        self.fail_blocks = set(fail_blocks)
        #: What its spec sends among its request settings (``JudgeSpec.request_settings``): none unless a test says.
        self.request_settings = dict(request_settings or {})

    @property
    def spec(self):
        """A stand-in for the judge's ``JudgeSpec``: its request settings, as the runner reads them."""
        return SimpleNamespace(request_settings=lambda: dict(self.request_settings))

    def resolved_temperature(self, requested):
        return self.fixed_temperature if self.fixed_temperature is not None else requested

    def close(self):
        """As ``ModelBackend.close`` for a backend without a local engine: nothing to stop."""

    def _default(self, component_id):
        if self.style == "forced":
            return json.dumps({key: value for key, value in json.loads(forced("no")).items() if key != "preempted_rules"})
        return forced("no") if self.style == "forced-grpo" else "no"

    @staticmethod
    def locate(entry):
        scenario_index = int(_PROMPT_RE.search(entry["system_prompt"]).group(1).split()[-1])
        ruleset = _RULESET_RE.search(entry["system_prompt"]).group(1)
        component_id = "full_ruleset" if "\n" in ruleset else ruleset.split(":")[0]
        return scenario_index, component_id

    def generate_once(self, prompts, *, temperature, max_new_tokens=None, seeds=None, request_block_id="", truncation=None):
        assert temperature == 1.0
        assert seeds is None or len(seeds) == len(prompts)
        self.blocks.append((request_block_id, len(prompts), list(seeds) if seeds else None))
        self.truncations.append(truncation)
        if request_block_id in self.fail_blocks:
            raise BatchExecutionError(f"batch for {request_block_id} died")
        outcomes = []
        for entry in prompts:
            key = self.locate(entry)
            n = self.calls.get(key, 0)
            self.calls[key] = n + 1
            entries = self.script.get(key)
            value = entries[min(n, len(entries) - 1)] if entries else self._default(key[1])
            if isinstance(value, Exception):
                raise value
            if isinstance(value, str):
                value = GenerationOutcome(text=value, raw_response=f"<think>{key}</think>{value}", finish_reason="stop")
            outcomes.append(value)
        return outcomes


def restamp_stacks(draw_dir: Path) -> None:
    """Rewrite the stack each of a draw's ledger rows names (``judged_on``) to its provenance's."""
    ledger = Path(draw_dir) / LEDGER_FNAME
    stack = stack_digest(recorded_runtime(draw_dir))
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    ledger.write_text("".join(json.dumps({**row, JUDGED_ON_KEY: stack} if JUDGED_ON_KEY in row else row,
                                         ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_inputs(tmp_path: Path):
    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        "".join(
            json.dumps({"prompt": f"prompt {i}", "response": f"response {i}"}) + "\n"
            for i in range(N_SCENARIOS)
        ),
        encoding="utf-8",
    )
    return scenarios


def make_spec(manifest, *, style="naive", draw=0, seed_capability="local_deterministic", **config_overrides):
    config = SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, **config_overrides)
    condition = ConditionSpec(
        condition_id=f"fake/{style}",
        model_name="fake-judge",
        model_type="local",
        prompt_style=style,
        individual_rule_context="isolated",
        judge_type="No_Interpretation",
        evaluation_mode="both",
        rule_set_name="rules.jsonl",
        rule_set_sha256="a" * 64,
        temperature=1.0,
        prompt_shape_version=2,
        sampling_truncation={"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0},
        forced_json_requirement=config.forced_json_requirement,
        retry_scope=config.retry_scope,
        strict_think=config.strict_think,
        seed_capability=seed_capability,
    )
    return DrawSpec(
        campaign_id="camp", condition=condition, decoder_draw_id=draw, config=config,
        scenario_manifest=manifest, judge_type="No_Interpretation", max_transport_retries=2,
    )


@pytest.fixture
def inputs(tmp_path):
    path = write_inputs(tmp_path)
    manifest = ScenarioManifest.build(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return manifest, rows


def build_runner(tmp_path, inputs, modelspec, backend, *, style="naive", draw=0, **overrides):
    manifest, rows = inputs
    spec = make_spec(manifest, style=style, draw=draw, seed_capability=backend.seed_capability, **overrides)
    clock = iter(f"2026-09-08T00:00:{i:02d}+00:00" for i in range(10_000))
    return DrawRunner(
        backend=backend,
        task_spec=modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0),
        spec=spec,
        draw_dir=tmp_path / "camp" / "fake" / style / f"draw-{draw:03d}",
        scenario_rows=rows,
        rules=RULES,
        clock=lambda: next(clock),
        code_revision="rev",
    )


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sid_of(inputs, index):
    return inputs[0].scenario_ids[index]


def test_a_judge_that_reports_no_per_request_errors_is_refused(tmp_path, inputs, modelspec):
    with pytest.raises(CampaignRunError, match="cannot run a stochastic campaign"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend(per_request_errors=False))


def test_a_sync_provider_reporting_per_request_errors_runs_once_its_spend_is_confirmed(
        tmp_path, inputs, modelspec, monkeypatch):
    backend = FakeBackend(model_type="openai", seed_capability="unsupported")
    runner = build_runner(tmp_path, inputs, modelspec, backend, scenarios_per_request_block=2)
    with pytest.raises(BatchSpendNotConfirmed, match="--yes-spend"):
        runner.run()
    assert backend.blocks == []
    calls = []
    monkeypatch.setattr(runner_module, "require_spend_confirmation", lambda: calls.append(1))
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    assert runner.run()["state"] == "complete"
    assert calls == [1] and len(backend.blocks) > 1


@pytest.mark.parametrize("model_type, seed_capability", [("local", "local_deterministic"),
                                                          ("openai_batch", "unsupported")])
def test_a_free_judge_or_a_batch_transport_is_not_asked_by_the_runner(
        tmp_path, inputs, modelspec, monkeypatch, model_type, seed_capability):
    monkeypatch.setattr(runner_module, "require_spend_confirmation",
                        lambda: pytest.fail("the runner asked a spend confirmation"))
    runner = build_runner(tmp_path, inputs, modelspec,
                          FakeBackend(model_type=model_type, seed_capability=seed_capability))
    assert runner.run()["state"] == "complete"


def test_the_fake_judge_is_refused_in_a_final_campaign(tmp_path, inputs, modelspec):
    manifest, rows = inputs
    spec = make_spec(manifest)
    spec = dataclasses.replace(spec, config=dataclasses.replace(spec.config, campaign_mode="final"))
    with pytest.raises(CampaignRunError, match="never a 'final' one"):
        DrawRunner(backend=FakeBackend(model_type="fake"), task_spec=modelspec(), spec=spec, draw_dir=tmp_path / "d",
                   scenario_rows=rows, rules=RULES)


def test_a_pinned_temperature_other_than_one_is_refused(tmp_path, inputs, modelspec):
    with pytest.raises(CampaignRunError, match="would sample at temperature 0.1"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend(fixed_temperature=0.1))


def test_a_seed_capability_mismatch_is_refused(tmp_path, inputs, modelspec):
    manifest, rows = inputs
    spec = make_spec(manifest, seed_capability="unsupported")
    with pytest.raises(CampaignRunError, match="seed capability"):
        DrawRunner(backend=FakeBackend(), task_spec=modelspec(), spec=spec, draw_dir=tmp_path / "d",
                   scenario_rows=rows, rules=RULES)


def test_scenario_rows_must_be_the_manifest_population(tmp_path, inputs, modelspec):
    manifest, rows = inputs
    spec = make_spec(manifest)
    with pytest.raises(CampaignRunError, match="not the frozen manifest"):
        DrawRunner(backend=FakeBackend(), task_spec=modelspec(), spec=spec, draw_dir=tmp_path / "d",
                   scenario_rows=rows[:-1], rules=RULES)
    with pytest.raises(CampaignRunError, match="not the frozen manifest"):
        DrawRunner(backend=FakeBackend(), task_spec=modelspec(), spec=spec, draw_dir=tmp_path / "d",
                   scenario_rows=rows + rows[:1], rules=RULES)


def test_a_clean_draw_completes_and_materialises_every_view(tmp_path, inputs, modelspec):
    s1 = 1
    backend = FakeBackend(script={(s1, "full_ruleset"): ["No"], (s1, "Rule 2"): ["YES"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()

    assert completion["state"] == STATE_COMPLETE
    assert completion["accepted"] == 3 and completion["grid"]["ok"]
    assert completion["attempts_histogram"] == {"1": 3}
    assert completion["invalid_components"] == {} and completion["rejected_attempts"] == 0
    assert completion["components_per_bundle"] == 3
    assert json.loads((runner.draw_dir / COMPLETION_FNAME).read_text()) == completion

    assert len(backend.blocks) == 1
    block_id, n_prompts, seeds = backend.blocks[0]
    assert block_id == sent_block("camp/fake/naive/draw-000/attempt-0/block-0000")
    assert n_prompts == 9 and len(set(seeds)) == 9
    blocks = read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)
    assert blocks[0]["scenario_ids"] == list(inputs[0].scenario_ids)

    # Bundle-level Y: scenario 1 said no overall while Rule 2 said yes.
    outcomes = read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)
    assert [o["scenario_id"] for o in outcomes] == list(inputs[0].scenario_ids)
    assert [o["contradiction"] for o in outcomes] == [0.0, 1.0, 0.0]
    assert outcomes[1]["rule_judgments"] == {"Rule 1": "no", "Rule 2": "yes"}
    assert outcomes[1]["full_set_judgment"] == "no" and outcomes[1]["penalty"] == 1.0
    assert outcomes[1]["in_context_contradiction"] is None

    evaluations = read_jsonl(runner.draw_dir / EVALUATIONS_FNAME)
    assert [row["scenario_id"] for row in evaluations] == list(inputs[0].scenario_ids)
    assert evaluations[1]["evaluations"]["fake-judge"] == {
        "full_ruleset": {"No_Interpretation": "no"},
        "Rule 1": {"No_Interpretation": "no"},
        "Rule 2": {"No_Interpretation": "yes"},
    }
    assert evaluations[1]["decoder_draw_id"] == 0 and evaluations[1]["semantic_attempt_id"] == 0
    per_scenario = read_jsonl(runner.draw_dir / PER_SCENARIO_FNAME)
    assert [row["contradiction"] for row in per_scenario] == [0.0, 1.0, 0.0]
    summary = json.loads((runner.draw_dir / SUMMARY_FNAME).read_text())
    assert summary["status"] == "complete" and summary["n_scenarios"] == 3
    assert summary["average_flat_contradiction"] == pytest.approx(1 / 3)
    assert summary["decoder_draw_id"] == 0

    traced = read_jsonl(runner.draw_dir / JUDGE_OUTPUTS_FNAME)
    trace = traced[1]["judge_outputs"]["fake-judge"]["Rule 2"]["No_Interpretation"]
    assert trace["judgment"] == "yes" and trace["selected_attempt"] == 1
    assert trace["attempts"][0]["candidates"][0]["accepted"] is True
    assert trace["attempts"][0]["candidates"][0]["raw_response"].startswith("<think>")
    assert trace["prompt"]["system_prompt"].startswith("# Task")

    provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance["condition_fingerprint"] == runner.spec.condition.fingerprint
    assert provenance["resolved_temperature"] == 1.0 and provenance["seeds_sent"] is True
    assert provenance["component_ids"] == list(COMPONENTS)


def test_a_maximally_inconsistent_bundle_is_accepted_first_time(tmp_path, inputs, modelspec):
    backend = FakeBackend(script={(0, "full_ruleset"): ["no"], (0, "Rule 1"): ["yes"], (0, "Rule 2"): ["yes"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert all(count == 1 for count in backend.calls.values())
    outcomes = read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)
    assert outcomes[0]["contradiction"] == 1.0


def test_a_not_sure_answer_is_rejected_rather_than_read_as_no(tmp_path, inputs, modelspec):
    backend = FakeBackend(script={(0, "Rule 1"): ["Not sure.", "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert completion["invalid_components"] == {"invalid_label": 1}
    assert completion["attempts_histogram"] == {"1": 2, "2": 1}


def test_one_malformed_component_regenerates_the_whole_bundle_and_nothing_else(
    tmp_path, inputs, modelspec
):
    backend = FakeBackend(script={(1, "Rule 1"): ["maybe", "yes"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()

    assert completion["state"] == STATE_COMPLETE
    assert completion["rejected_attempts"] == 1
    assert {key: n for key, n in backend.calls.items() if key[0] == 1} == {(1, c): 2 for c in COMPONENTS}
    assert all(n == 1 for key, n in backend.calls.items() if key[0] != 1)
    assert [(b[0].rsplit("/", 2)[-2], b[1]) for b in backend.blocks] == [("attempt-0", 9), ("attempt-1", 3)]

    ledger = DrawLedger(runner.draw_dir, campaign_id="camp", condition_id="fake/naive", decoder_draw_id=0)
    rows = [r for r in ledger.load().records if r.scenario_id == sid_of(inputs, 1)]
    assert [(r.status, r.semantic_attempt_id) for r in rows] == [(STATUS_REJECTED, 0), (STATUS_ACCEPTED, 1)]
    assert rows[0].failure_reason == "component_invalid"
    assert [c.valid for c in rows[0].components] == [True, False, True]
    traced = read_jsonl(runner.draw_dir / JUDGE_OUTPUTS_FNAME)
    trace = traced[1]["judge_outputs"]["fake-judge"]["Rule 1"]["No_Interpretation"]
    assert [a["status"] for a in trace["attempts"]] == [STATUS_REJECTED, STATUS_ACCEPTED]
    assert trace["attempts"][0]["candidates"][0]["completion"] == "maybe"
    assert trace["selected_attempt"] == 2 and trace["judgment"] == "yes"
    outcomes = read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)
    assert outcomes[1]["semantic_attempt_id"] == 1 and outcomes[1]["contradiction"] == 1.0


def test_five_failed_attempts_leave_the_draw_incomplete_and_unmaterialised(
    tmp_path, inputs, modelspec
):
    backend = FakeBackend(script={(2, "full_ruleset"): ["hmm"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()

    assert completion["state"] == STATE_INCOMPLETE_EXHAUSTED
    assert completion["exhausted"] == [sid_of(inputs, 2)]
    assert completion["accepted"] == 2 and completion["pending"] == 0
    assert completion["rejected_attempts"] == 5
    assert completion["invalid_components"] == {"invalid_label": 5}
    assert not completion["grid"]["ok"] and completion["grid"]["missing"] == [[sid_of(inputs, 2), 0]]
    assert backend.calls[(2, "full_ruleset")] == 5
    assert not (runner.draw_dir / EVALUATIONS_FNAME).exists()
    assert not (runner.draw_dir / BUNDLE_OUTCOMES_FNAME).exists()
    with pytest.raises(CampaignRunError, match="only a complete draw"):
        runner.materialize(runner.ledger.load())


def test_a_draw_stopped_with_one_scenario_exhausted_and_one_pending_stays_in_progress(tmp_path, inputs, modelspec):
    dying = "camp/fake/naive/draw-000/attempt-0/block-0002"
    backend = FakeBackend(script={(0, "full_ruleset"): ["hmm"]}, fail_blocks={sent_block(dying)})
    settings = {"scenarios_per_request_block": 1, "max_semantic_attempts": 1}
    with pytest.raises(BatchExecutionError):
        build_runner(tmp_path, inputs, modelspec, backend, **settings).run()
    gate = json.loads((tmp_path / "camp" / "fake" / "naive" / "draw-000" / COMPLETION_FNAME).read_text())
    assert (gate["state"], gate["exhausted"], gate["pending"]) == (STATE_IN_PROGRESS, [sid_of(inputs, 0)], 1)

    backend.fail_blocks.clear()
    completion = build_runner(tmp_path, inputs, modelspec, backend, **settings).run()
    assert (completion["state"], completion["exhausted"], completion["pending"], completion["accepted"]) == (
        STATE_INCOMPLETE_EXHAUSTED, [sid_of(inputs, 0)], 0, 2)


def test_a_draw_whose_views_failed_to_write_is_gated_complete_only_once_they_all_are(
        tmp_path, inputs, modelspec, monkeypatch):
    real = runner_module.write_draw_summary

    def full_disk(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(runner_module, "write_draw_summary", full_disk)
    gate = tmp_path / "camp" / "fake" / "naive" / "draw-000" / COMPLETION_FNAME
    for _ in range(2):
        with pytest.raises(OSError, match="No space left"):
            build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()
        assert (gate.parent / BUNDLE_OUTCOMES_FNAME).is_file() and not (gate.parent / SUMMARY_FNAME).exists()
        assert json.loads(gate.read_text())["state"] == STATE_IN_PROGRESS

    monkeypatch.setattr(runner_module, "write_draw_summary", real)
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()["state"] == STATE_COMPLETE
    assert (gate.parent / SUMMARY_FNAME).is_file() and (gate.parent / PER_SCENARIO_FNAME).is_file()


def test_a_transport_failure_is_retried_without_consuming_an_attempt(tmp_path, inputs, modelspec):
    failed = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")
    backend = FakeBackend(script={(2, "full_ruleset"): [failed, "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()

    assert completion["state"] == STATE_COMPLETE
    assert completion["transport_failed_attempts"] == 1 and completion["rejected_attempts"] == 0
    assert completion["attempts_histogram"] == {"1": 3}
    assert completion["invalid_components"] == {"transport_error": 1}
    assert [b[0].rsplit("/", 2)[-2:] for b in backend.blocks] == [
        ["attempt-0", "block-0000"], ["attempt-0", "block-0001"],
    ]
    ledger = DrawLedger(runner.draw_dir, campaign_id="camp", condition_id="fake/naive", decoder_draw_id=0)
    rows = [r for r in ledger.load().records if r.scenario_id == sid_of(inputs, 2)]
    assert [(r.status, r.semantic_attempt_id) for r in rows] == [(STATUS_TRANSPORT_FAILED, 0), (STATUS_ACCEPTED, 0)]
    raw = [row for row in ledger.iter_raw() if row["request_block_id"].endswith("block-0000") and row["scenario_id"] == sid_of(inputs, 2)]
    assert sorted(row["component_id"] for row in raw) == ["Rule 1", "Rule 2"]


def test_repeated_transport_failures_stop_the_run_but_leave_it_resumable(tmp_path, inputs, modelspec):
    failed = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")
    backend = FakeBackend(script={(0, "Rule 2"): [failed, failed, failed, "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    with pytest.raises(TransportExhausted, match="failed in transport 2 time"):
        runner.run()
    completion = json.loads((runner.draw_dir / COMPLETION_FNAME).read_text())
    assert completion["state"] == STATE_IN_PROGRESS
    assert completion["accepted"] == 2 and completion["pending"] == 1

    resumed = build_runner(tmp_path, inputs, modelspec, backend)
    completion = resumed.run()
    assert completion["state"] == STATE_COMPLETE
    assert completion["attempts_histogram"] == {"1": 3}
    assert completion["transport_failed_attempts"] == 3
    assert backend.calls[(0, "Rule 2")] == 4


def test_the_transport_cap_counts_a_scenario_s_failures_in_this_run_across_its_attempts(tmp_path, inputs, modelspec):
    failed = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")
    backend = FakeBackend(script={(0, "Rule 2"): [failed, "maybe", failed]})
    with pytest.raises(TransportExhausted, match=r"failed in transport 2 time\(s\) in this run \(first: ") as stopped:
        build_runner(tmp_path, inputs, modelspec, backend).run()
    assert "at attempt" not in str(stopped.value)
    rows = read_jsonl(tmp_path / "camp" / "fake" / "naive" / "draw-000" / LEDGER_FNAME)
    assert [(row["status"], row["semantic_attempt_id"]) for row in rows if row["scenario_id"] == sid_of(inputs, 0)] == [
        (STATUS_TRANSPORT_FAILED, 0), (STATUS_REJECTED, 0), (STATUS_TRANSPORT_FAILED, 1)]


def test_a_crash_mid_pass_resumes_by_resubmitting_the_recorded_block(tmp_path, inputs, modelspec):
    dying = "camp/fake/naive/draw-000/attempt-0/block-0001"
    backend = FakeBackend(fail_blocks={sent_block(dying)})
    runner = build_runner(tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1)
    with pytest.raises(BatchExecutionError):
        runner.run()
    assert [b["block_index"] for b in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)] == [0, 1, 2]
    assert len(runner.ledger.load().records) == 1

    backend.fail_blocks.clear()
    resumed = build_runner(tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1)
    completion = resumed.run()
    assert completion["state"] == STATE_COMPLETE
    submitted = [b[0] for b in backend.blocks]
    assert submitted == [sent_block(block) for block in (
        "camp/fake/naive/draw-000/attempt-0/block-0000",
        dying,
        "camp/fake/naive/draw-000/attempt-0/block-0001",
        "camp/fake/naive/draw-000/attempt-0/block-0002",
    )]
    assert backend.calls[(0, "full_ruleset")] == 1
    assert [b["block_index"] for b in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)] == [0, 1, 2]


def test_a_recorded_block_that_disagrees_with_the_ledger_is_refused(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.run()
    # Forge a block record for attempt 0 naming an already-accepted scenario.
    with (runner.draw_dir / REQUEST_BLOCKS_FNAME).open("a") as handle:
        handle.write(json.dumps({
            "request_block_id": "camp/fake/naive/draw-000/attempt-0/block-0007",
            "block_index": 7, "semantic_attempt_id": 0,
            "scenario_ids": [sid_of(inputs, 0)], "recorded_at": "t",
        }) + "\n")
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()["state"] == STATE_COMPLETE
    fresh = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    state = fresh.ledger.load()
    with pytest.raises(CampaignRunError, match="no longer pending"):
        fresh._plan_blocks(state, 0, [sid_of(inputs, 1)])


def test_a_resumed_complete_draw_asks_nothing_and_rematerialises(tmp_path, inputs, modelspec):
    first = FakeBackend()
    build_runner(tmp_path, inputs, modelspec, first).run()
    second = FakeBackend()
    completion = build_runner(tmp_path, inputs, modelspec, second).run()
    assert completion["state"] == STATE_COMPLETE and second.calls == {}


def test_a_draw_directory_of_another_condition_is_refused(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.run()
    other = build_runner(tmp_path, inputs, modelspec, FakeBackend(), forced_json_requirement="label_only")
    with pytest.raises(CampaignRunError, match="different draw or condition"):
        other.run()


def _hold(path: Path) -> int:
    """Hold ``path``'s ``flock`` through a descriptor of its own, as another process would."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _files(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()}


def test_a_draw_another_process_is_running_is_refused_before_its_ledger_is_repaired(tmp_path, inputs, modelspec):
    first = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    assert first.run()["state"] == STATE_COMPLETE
    lock = draw_lock_path(first.draw_dir)
    assert lock == first.draw_dir.parent / ".draw-000.lock" and not lock.exists()
    raw = first.draw_dir / "raw_outputs.jsonl"
    row = json.loads(raw.read_text().splitlines()[-1])
    with raw.open("a") as handle:  # another run's block in flight: its raw rows appended, its ledger rows not yet
        handle.write(json.dumps({**row, "request_block_id": row["request_block_id"] + "-in-flight"}) + "\n")
    before = _files(first.draw_dir)
    descriptor = _hold(lock)
    try:
        backend = FakeBackend()
        with pytest.raises(CampaignRunError, match=r"another process holds .*\.draw-000\.lock: it is running "
                                                   r".*draw-000, and two runs of one draw would each repair"):
            build_runner(tmp_path, inputs, modelspec, backend).run()
        assert _files(first.draw_dir) == before and backend.blocks == []
        assert build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=1).run()["state"] == STATE_COMPLETE
        assert not draw_lock_path(first.draw_dir.with_name("draw-001")).exists()
    finally:
        os.close(descriptor)
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()["state"] == STATE_COMPLETE
    assert not lock.exists()


def test_a_draw_lock_replaced_by_a_link_creates_nothing_where_it_points(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    target = tmp_path / "elsewhere" / "lock"
    target.parent.mkdir()
    lock = draw_lock_path(runner.draw_dir)
    lock.parent.mkdir(parents=True)
    lock.symlink_to(target)
    with pytest.raises(CampaignRunError, match=r"\.draw-000\.lock, the run lock of .*draw-000, cannot be opened as a "
                                               "file of its own"):
        runner.run()
    assert not target.exists() and not runner.draw_dir.exists()


def test_a_lock_that_fails_once_taken_is_released_and_its_error_raised(tmp_path, monkeypatch):
    draw = tmp_path / "condition" / "draw-000"
    real = os.fstat

    def fstat(descriptor):
        raise OSError(116, "Stale file handle")

    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(OSError, match="Stale file handle"):
        with draw_run_lock(draw):
            pass
    monkeypatch.setattr(os, "fstat", real)
    with draw_run_lock(draw) as held:
        assert held.exists()
    assert not held.exists()


@pytest.mark.parametrize("meanwhile", ["removed", "replaced"])
def test_a_lock_its_holder_removed_after_this_run_opened_it_is_opened_again(tmp_path, monkeypatch, meanwhile):
    draw = tmp_path / "condition" / "draw-000"
    path = draw_lock_path(draw)
    real = fcntl.flock
    calls = []

    def flock(descriptor, operation):
        if not calls:  # the previous holder ends between this run's open and its lock
            path.unlink()
            if meanwhile == "replaced":
                path.write_text("")
        calls.append(operation)
        return real(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", flock)
    with draw_run_lock(draw) as held:
        assert held == path and path.exists()
        monkeypatch.setattr(fcntl, "flock", real)
        with pytest.raises(CampaignRunError, match="another process holds"):
            with draw_run_lock(draw):
                pass
    assert calls == [fcntl.LOCK_EX | fcntl.LOCK_NB] * 2 and not path.exists()


def test_a_lock_is_removed_only_where_it_was_taken_and_while_it_is_the_file_held(tmp_path):
    root = tmp_path / "root"
    draw = root / "condition" / "draw-000"
    first = draw_run_lock(draw)
    path = first.__enter__()
    shutil.move(root, tmp_path / "moved")
    second = draw_run_lock(draw)
    assert second.__enter__() == path
    first.__exit__(None, None, None)
    assert path.exists() and not draw_lock_path(tmp_path / "moved" / "condition" / "draw-000").exists()
    with pytest.raises(CampaignRunError, match="another process holds"):
        with draw_run_lock(draw):
            pass
    second.__exit__(None, None, None)
    assert not path.exists()


def test_a_lock_file_replaced_where_it_was_taken_is_left_to_its_new_holder(tmp_path):
    draw = tmp_path / "condition" / "draw-000"
    first = draw_run_lock(draw)
    path = first.__enter__()
    path.unlink()
    with draw_run_lock(draw) as again:
        first.__exit__(None, None, None)
        assert again == path and path.exists()
    assert not path.exists()


def test_a_lock_whose_directory_cannot_be_opened_is_refused(tmp_path, monkeypatch):
    draw = tmp_path / "condition" / "draw-000"
    real = os.open

    def refusing(path, flags, *args, **kwargs):
        if flags & os.O_DIRECTORY:
            raise PermissionError(13, "Permission denied")
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", refusing)
    with pytest.raises(CampaignRunError, match="where the run lock of .* goes, cannot be opened as a directory"):
        with draw_run_lock(draw):
            pass
    monkeypatch.setattr(os, "open", real)
    assert not draw_lock_path(draw).exists()


def test_the_forced_style_yields_in_context_outcomes_from_the_same_response(tmp_path, inputs, modelspec):
    backend = FakeBackend(
        style="forced-grpo",
        script={
            (0, "full_ruleset"): [forced("no", {"Rule 1": "yes", "Rule 2": "no"})],
            (0, "Rule 1"): [forced("no", {"Rule 1": "no"})],
        },
    )
    runner = build_runner(tmp_path, inputs, modelspec, backend, style="forced-grpo")
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    outcomes = read_jsonl(runner.draw_dir / BUNDLE_OUTCOMES_FNAME)
    # Separate calls: full no, rules no/no -> consistent.
    assert outcomes[0]["contradiction"] == 0.0
    # Same call: full no, its own Rule 1 entry yes -> inconsistent.
    assert outcomes[0]["in_context_rule_judgments"] == {"Rule 1": "yes", "Rule 2": "no"}
    assert outcomes[0]["in_context_contradiction"] == 1.0
    assert outcomes[0]["in_context_penalty"] == 1.0


def test_a_forced_full_set_missing_a_rule_is_rejected(tmp_path, inputs, modelspec):
    backend = FakeBackend(style="forced-grpo", script={(1, "full_ruleset"): [forced("no", {"Rule 1": "no"}), forced("no")]})
    completion = build_runner(tmp_path, inputs, modelspec, backend, style="forced-grpo").run()
    assert completion["state"] == STATE_COMPLETE
    assert completion["invalid_components"] == {"missing_rule_ids": 1}


def test_a_seedless_provider_sends_no_seeds_and_records_that(tmp_path, inputs, modelspec):
    backend = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert backend.blocks[0][2] is None
    record = runner.ledger.load().accepted[sid_of(inputs, 0)]
    assert all(c.generation_seed is None and c.seed_sent is False for c in record.components)
    provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance["seeds_sent"] is False and provenance["seed_capability"] == "unsupported"
    assert request_namespace(backend.blocks[0][0]).startswith("ns")


def test_the_trace_refuses_a_prompt_that_no_longer_matches(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.run()
    runner.rule_set_string += "\nRule 9: added later"
    with pytest.raises(CampaignRunError, match="no longer matches"):
        runner.materialize(runner.ledger.load())


def test_a_second_draw_lives_beside_the_first_and_retains_identical_text(tmp_path, inputs, modelspec):
    first = build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=0)
    second = build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=1)
    assert first.run()["state"] == STATE_COMPLETE
    assert second.run()["state"] == STATE_COMPLETE
    a = read_jsonl(first.draw_dir / EVALUATIONS_FNAME)
    b = read_jsonl(second.draw_dir / EVALUATIONS_FNAME)
    assert [r["evaluations"] for r in a] == [r["evaluations"] for r in b]
    assert [r["decoder_draw_id"] for r in b] == [1, 1, 1]
    assert first.draw_dir != second.draw_dir


def test_the_default_clock_is_utc_iso8601():
    from ruleforge.evaluate.runner import utc_now

    stamp = utc_now()
    assert stamp.endswith("+00:00") and "T" in stamp


def test_a_returned_empty_answer_is_rejected_and_consumes_an_attempt(tmp_path, inputs, modelspec):
    empty = GenerationOutcome(text=None, raw_response="", finish_reason="stop")
    backend = FakeBackend(script={(1, "Rule 1"): [empty, empty, "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert completion["rejected_attempts"] == 2 and completion["transport_failed_attempts"] == 0
    assert completion["invalid_components"] == {"empty_response": 2}
    assert completion["attempts_histogram"] == {"1": 2, "3": 1}


def test_a_whole_block_failure_is_retried_on_the_same_block_id(tmp_path, inputs, modelspec):
    class FlakyBackend(FakeBackend):
        def __init__(self, fail_first):
            super().__init__()
            self.fail_first = fail_first
            self.seen = []

        def generate_once(self, prompts, *, request_block_id="", **kwargs):
            self.seen.append(request_block_id)
            if self.fail_first > 0:
                self.fail_first -= 1
                return [
                    GenerationOutcome(text=None, raw_response=None, finish_reason=None,
                                      error="RuntimeError: engine died", block_failed=True)
                    for _ in prompts
                ]
            return super().generate_once(prompts, request_block_id=request_block_id, **kwargs)

    backend = FlakyBackend(fail_first=1)
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert backend.seen == [sent_block("camp/fake/naive/draw-000/attempt-0/block-0000")] * 2
    assert completion["transport_failed_attempts"] == 0
    assert len(read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)) == 1

    dead = FlakyBackend(fail_first=99)
    other = build_runner(tmp_path / "other", inputs, modelspec, dead)
    with pytest.raises(TransportExhausted, match="raised 2 time"):
        other.run()
    assert not other.ledger.ledger_path.exists()
    assert json.loads((other.draw_dir / COMPLETION_FNAME).read_text())["state"] == STATE_IN_PROGRESS


def test_a_torn_ledger_append_resubmits_the_block_whole_and_records_the_rest(
    tmp_path, inputs, modelspec
):
    first = FakeBackend()
    runner = build_runner(tmp_path, inputs, modelspec, first)
    runner.run()
    lines = runner.ledger.ledger_path.read_text().splitlines(keepends=True)
    runner.ledger.ledger_path.write_text(lines[0] + lines[1][:40])
    for name in (BUNDLE_OUTCOMES_FNAME, EVALUATIONS_FNAME, JUDGE_OUTPUTS_FNAME, PER_SCENARIO_FNAME, SUMMARY_FNAME):
        (runner.draw_dir / name).unlink()

    second = FakeBackend()
    resumed = build_runner(tmp_path, inputs, modelspec, second)
    completion = resumed.run()
    assert completion["state"] == STATE_COMPLETE
    assert [b[0] for b in second.blocks] == [sent_block("camp/fake/naive/draw-000/attempt-0/block-0000")]
    assert second.blocks[0][1] == 9
    assert [b["block_index"] for b in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)] == [0]
    assert len(resumed.ledger.load().records) == 3
    assert second.calls[(0, "full_ruleset")] == 1


def test_the_gate_says_complete_only_once_the_views_exist(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE and completion["materialized"] is True
    state = runner.ledger.load()
    assert runner.completion(state, materialized=False)["state"] == STATE_IN_PROGRESS
    (runner.draw_dir / BUNDLE_OUTCOMES_FNAME).unlink()
    assert runner.completion(state)["state"] == STATE_IN_PROGRESS
    assert runner.completion(state)["materialized"] is False


def test_materialize_refuses_before_writing_when_a_prompt_digest_differs(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.run()
    for name in (BUNDLE_OUTCOMES_FNAME, EVALUATIONS_FNAME, JUDGE_OUTPUTS_FNAME):
        (runner.draw_dir / name).unlink()
    runner.rule_set_string += "\nRule 9: added later"
    with pytest.raises(CampaignRunError, match="no longer matches"):
        runner.materialize(runner.ledger.load())
    assert not (runner.draw_dir / BUNDLE_OUTCOMES_FNAME).exists()
    assert not (runner.draw_dir / EVALUATIONS_FNAME).exists()


def test_each_run_is_logged_and_an_edited_tree_may_resume(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    assert runner.run()["state"] == STATE_COMPLETE
    runs = read_jsonl(runner.draw_dir / RUNS_FNAME)
    assert len(runs) == 1
    assert runs[0]["code_revision"] == "rev"
    assert runs[0]["code_dirty"] is None
    assert runs[0]["max_semantic_attempts"] == runner.spec.config.max_semantic_attempts
    assert runs[0]["started_at"].startswith("2026-09-08T")
    assert {"hostname", "pid", "slurm_job_id", "python_version", "vllm_version", "torch_version", "gpu"} <= set(runs[0]["runtime"])

    provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance["sampling_truncation"] == {"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
    assert provenance["code_state_sha256"] is None
    assert provenance["tp_size"] is None and provenance["attention_backend"] is None  # the fake has no engine

    again = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    again.code_revision, again.code_dirty, again.code_state_sha256 = "rev2", True, "e" * 64
    assert again.run()["state"] == STATE_COMPLETE
    runs = read_jsonl(again.draw_dir / RUNS_FNAME)
    assert [row["code_revision"] for row in runs] == ["rev", "rev2"]
    assert runs[1]["code_dirty"] is True and runs[1]["code_state_sha256"] == "e" * 64
    held = len((again.draw_dir / LEDGER_FNAME).read_text().splitlines())
    assert [row["ledger_rows"] for row in runs] == [0, held] and held > 0
    assert json.loads((again.draw_dir / DRAW_PROVENANCE_FNAME).read_text())["code_revision"] == "rev"


def test_each_row_names_its_run_s_stack_and_no_run_logs_itself_behind_torn_bytes(tmp_path, inputs, modelspec, caplog):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.run()
    ledger, runs = runner.draw_dir / LEDGER_FNAME, runner.draw_dir / RUNS_FNAME
    [logged] = read_jsonl(runs)
    assert {row[JUDGED_ON_KEY] for row in read_jsonl(ledger)} == {stack_digest(logged["runtime"])}
    provider = build_runner(tmp_path / "p", inputs, modelspec, FakeBackend(model_type="openai_batch"))
    provider.run()
    assert not any(JUDGED_ON_KEY in row for row in read_jsonl(provider.draw_dir / LEDGER_FNAME))

    complete = runs.read_bytes()
    runs.write_bytes(complete + b'{"started_at": "2026-')
    build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()
    repaired = runs.read_bytes()
    assert repaired.startswith(complete) and len(read_jsonl(runs)) == 2
    assert "its runs.jsonl ends in a torn row (21 byte(s) after its 1 complete row(s)); repaired" in caplog.text
    runs.write_bytes(repaired + b'{"started_at": "2026-' + complete)
    with pytest.raises(CampaignRunError, match=r"holds a malformed row after its 2 complete row\(s\), and 0 line"
                                               r"\(s\) after it: .*refusing to log a run after it. Repair the run log"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()
    runs.write_bytes(repaired)
    rows = read_jsonl(ledger)
    ledger.write_text("".join(json.dumps({**row, JUDGED_ON_KEY: "f" * 64}) + "\n" for row in rows))
    with pytest.raises(CampaignRunError, match=rf"{len(rows)} row\(s\) of its attempts.jsonl were judged on a stack "
                                               r"\(judged_on ffffffffffff\).*refusing to log a run beside it"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()
    assert runs.read_bytes() == repaired


def test_a_draw_with_pending_work_resumes_at_another_placement(tmp_path, inputs, modelspec, monkeypatch):
    one_gpu = FakeBackend()
    one_gpu.engine_args = {"model": "/a/x", "tensor_parallel_size": 1, "max_num_seqs": 256,
                           "disable_log_stats": True, "attention_config": {"backend": "FLASH_ATTN"}}
    _interrupted(tmp_path, inputs, modelspec, monkeypatch, backend=one_gpu)
    two_gpus = FakeBackend()
    two_gpus.engine_args = {**one_gpu.engine_args, "model": "/b/x", "tensor_parallel_size": 2,
                            "max_num_seqs": 512, "disable_log_stats": False}
    two_gpus.tp_size, two_gpus.attention_backend = 2, "FLASH_ATTN"
    resumed = build_runner(tmp_path, inputs, modelspec, two_gpus)
    assert resumed.run()["state"] == STATE_COMPLETE
    assert all("stack_change" not in row for row in read_jsonl(resumed.draw_dir / RUNS_FNAME))
    assert len(condition_signatures(resumed.condition_dir)) == 1


def test_runtime_provenance_reads_the_gpu_only_when_torch_is_loaded(monkeypatch):
    import sys

    from ruleforge.evaluate.runner import _package_version, runtime_provenance

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("RULEFORGE_MODAL_RUN_ID", raising=False)
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    monkeypatch.setenv("VLLM_FLOAT32_MATMUL_PRECISION", "high")
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    info = runtime_provenance()
    assert info["gpu"] is None and info["slurm_job_id"] is None and info["pid"] > 0
    assert info["modal_run_id"] is None and info["modal_task_id"] is None
    assert info["precision_env"] == {
        "VLLM_FLOAT32_MATMUL_PRECISION": "high", "TRITON_F32_DEFAULT": "unset", "VLLM_BATCH_INVARIANT": "0",
    }
    assert {"tokenizers_version", "triton_version", "transformers_version"} <= set(info)

    on = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: "H100"))
    monkeypatch.setitem(sys.modules, "torch", on)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    for name in ("RULEFORGE_MODAL_RUN_ID", "MODAL_TASK_ID", "RULEFORGE_RUNPOD_RUN_ID", "RUNPOD_POD_ID"):
        monkeypatch.setenv(name, "20260925T062638Z-3dc730f4-4a21")
    info = runtime_provenance()
    assert info["gpu"] == "H100" and info["slurm_job_id"] == "123"
    assert [info[name] for name in ("modal_run_id", "modal_task_id", "runpod_run_id", "runpod_pod_id")] == [None] * 4

    off = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: "x"))
    monkeypatch.setitem(sys.modules, "torch", off)
    assert runtime_provenance()["gpu"] is None
    assert _package_version("definitely-not-an-installed-package") is None
    assert _package_version("numpy")


def test_the_draw_passes_its_truncation_to_the_judge_and_records_it(tmp_path, inputs, modelspec):
    import dataclasses

    shipped = {"top_k": 64, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0}
    backend = FakeBackend()
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    condition = dataclasses.replace(runner.spec.condition, sampling_truncation=tuple(sorted(shipped.items())))
    runner.spec = DrawSpec(**{**runner.spec.__dict__, "condition": condition})
    assert runner.run()["state"] == STATE_COMPLETE
    assert backend.truncations and all(t == shipped for t in backend.truncations)
    provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance["sampling_truncation"] == shipped
    again = build_runner(tmp_path / "m", inputs, modelspec, FakeBackend())
    again.spec = DrawSpec(**{**again.spec.__dict__, "condition": dataclasses.replace(again.spec.condition, sampling_truncation=dict(shipped))})
    assert again.run()["state"] == STATE_COMPLETE and again.backend.truncations[0] == shipped

    # A provider judge gets no truncation argument and records provider_default.
    provider = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
    other = build_runner(tmp_path / "p", inputs, modelspec, provider)
    other.spec = DrawSpec(**{**other.spec.__dict__, "condition": dataclasses.replace(other.spec.condition, sampling_truncation="provider_default")})
    assert other.run()["state"] == STATE_COMPLETE
    assert provider.truncations and all(t is None for t in provider.truncations)
    assert json.loads((other.draw_dir / DRAW_PROVENANCE_FNAME).read_text())["sampling_truncation"] == "provider_default"


def test_a_local_condition_without_a_resolved_truncation_is_refused(tmp_path, inputs, modelspec):
    import dataclasses

    manifest, rows = inputs
    spec = make_spec(manifest)
    bare = DrawSpec(**{**spec.__dict__, "condition": dataclasses.replace(spec.condition, sampling_truncation=None)})
    with pytest.raises(CampaignRunError, match="no resolved sampling truncation"):
        DrawRunner(
            backend=FakeBackend(), task_spec=modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0),
            spec=bare, draw_dir=tmp_path / "d", scenario_rows=rows, rules=RULES,
        )


def _revised(tmp_path, rows, *, replace=None, keep=None):
    revised = [dict(row) for row in rows]
    if replace is not None:
        revised[replace] = {"prompt": "prompt 9", "response": "response 9"}
    if keep is not None:
        revised = revised[:keep]
    path = tmp_path / f"revised_{replace}_{keep}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in revised), encoding="utf-8")
    return ScenarioManifest.build(path), revised


def test_a_ledger_from_another_population_is_refused_before_any_generation(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    assert runner.run()["state"] == STATE_COMPLETE
    manifest, rows = _revised(tmp_path, inputs[1], replace=1)
    backend = FakeBackend()
    moved = build_runner(tmp_path, (manifest, rows), modelspec, backend)
    before = {p.name: p.read_bytes() for p in runner.draw_dir.iterdir()}
    with pytest.raises(CampaignRunError, match=r"1 scenario\(s\) outside the frozen manifest.* Resume it under the "
                                               "population it froze, or create a new campaign root for the revised "
                                               "population and run it there$") as refusal:
        moved.run()
    assert "rebase" not in str(refusal.value)
    assert backend.blocks == [] and {p.name: p.read_bytes() for p in runner.draw_dir.iterdir()} == before

    fresh = build_runner(tmp_path / "fresh", (manifest, rows), modelspec, FakeBackend())
    fresh.draw_dir.mkdir(parents=True)
    (fresh.draw_dir / REQUEST_BLOCKS_FNAME).write_text(json.dumps({
        "request_block_id": "b", "block_index": 0, "semantic_attempt_id": 0,
        "scenario_ids": [sid_of(inputs, 1)], "recorded_at": "t",
    }) + "\n")
    with pytest.raises(CampaignRunError, match="outside the frozen manifest"):
        fresh.run()


def test_the_population_check_leaves_a_torn_block_record_for_the_pass_that_appends(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.draw_dir.mkdir(parents=True)
    torn = json.dumps({
        "request_block_id": "camp/fake/naive/draw-000/attempt-0/block-0000", "block_index": 0,
        "semantic_attempt_id": 0, "scenario_ids": [sid_of(inputs, 0)], "recorded_at": "t",
    }) + "\n" + '{"request_block_id": "camp/fake/leg'
    (runner.draw_dir / REQUEST_BLOCKS_FNAME).write_text(torn)
    runner.prepare()
    assert (runner.draw_dir / REQUEST_BLOCKS_FNAME).read_text() == torn
    assert runner.run()["state"] == STATE_COMPLETE


def test_pending_counts_only_manifest_scenarios(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    runner.run()
    state = runner.ledger.load()
    fewer = build_runner(tmp_path, _revised(tmp_path, inputs[1], keep=2), modelspec, FakeBackend())
    gate = fewer.completion(state, materialized=False)
    # Three accepted rows against a two-scenario manifest: never -1 pending.
    assert gate["pending"] == 0 and gate["grid"]["unexpected"] == [[sid_of(inputs, 2), 0]]
    swapped = build_runner(tmp_path, _revised(tmp_path, inputs[1], replace=2), modelspec, FakeBackend())
    assert swapped.completion(state, materialized=False)["pending"] == 1


def _on_stack(monkeypatch, *, precision_env=None, **versions):
    import ruleforge.evaluate.runner as runner_module

    real = runner_module.runtime_provenance

    def _runtime(**kwargs):
        runtime = {**real(**kwargs), **{f"{name}_version": value for name, value in versions.items()}}
        return runtime if precision_env is None else {**runtime, "precision_env": precision_env}

    monkeypatch.setattr(runner_module, "runtime_provenance", _runtime)


OLD_STACK = {"vllm": "0.19.0", "torch": "2.10.0+cu128", "transformers": "5.5.0"}
NEW_VLLM = {**OLD_STACK, "vllm": "0.28.0"}
_FAILED = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")


def _interrupted(tmp_path, inputs, modelspec, monkeypatch, *, draw=0, backend=None, **stack):
    _on_stack(monkeypatch, **(stack or OLD_STACK))
    backend = backend or FakeBackend()
    backend.script = {(0, "Rule 2"): [_FAILED, _FAILED, "no"]}
    runner = build_runner(tmp_path, inputs, modelspec, backend, draw=draw)
    with pytest.raises(TransportExhausted):
        runner.run()
    state = runner.ledger.load()
    assert state.records and runner.completion(state)["pending"] == 1
    return runner


def _condition_record(runner):
    path = runner.draw_dir.parent / CONDITION_STACK_FNAME
    return json.loads(path.read_text()) if path.exists() else None


def _stack(record):
    return {name: record["stack"][name] for name in OLD_STACK}


def test_a_draw_with_attempts_and_pending_work_is_not_resumed_on_another_vllm_release(tmp_path, inputs, modelspec, monkeypatch):
    first = _interrupted(tmp_path, inputs, modelspec, monkeypatch)
    _on_stack(monkeypatch, **NEW_VLLM)
    resumed = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    with pytest.raises(CampaignRunError) as error:
        resumed.run()
    message = str(error.value)
    assert "holds 4 attempt(s) recorded on another inference stack or engine (vllm 0.19.0 -> 0.28.0)" in message
    assert "1 pending bundle(s)" in message and "execution.allow_stack_change=true" in message
    assert len(read_jsonl(first.draw_dir / RUNS_FNAME)) == 1
    assert len(resumed.ledger.load().records) == 4


@pytest.mark.parametrize("package, now", [
    ("vllm", "0.19.1"), ("torch", "2.13.0+cu129"), ("transformers", "5.17.0"),
    ("torch", "2.10.0+cu129"),  # the same release, another CUDA build
])
def test_a_patch_release_or_another_torch_or_transformers_is_refused_too(tmp_path, inputs, modelspec, monkeypatch, package, now):
    _interrupted(tmp_path, inputs, modelspec, monkeypatch)
    _on_stack(monkeypatch, **{**OLD_STACK, package: now})
    with pytest.raises(CampaignRunError, match=re.escape(f"{package} {OLD_STACK[package]} -> {now}")):
        build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()


def test_another_precision_setting_is_refused_once_recorded(tmp_path, inputs, modelspec, monkeypatch):
    from ruleforge.judges.backend import precision_env

    library_defaults = precision_env({})
    _interrupted(tmp_path, inputs, modelspec, monkeypatch, **OLD_STACK, precision_env=library_defaults)
    _on_stack(monkeypatch, **OLD_STACK, precision_env={**library_defaults, "TRITON_F32_DEFAULT": "tf32"})
    with pytest.raises(CampaignRunError, match="TRITON_F32_DEFAULT unset -> tf32"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()


def test_an_allowed_stack_change_resumes_and_is_logged(tmp_path, inputs, modelspec, monkeypatch, caplog):
    _interrupted(tmp_path, inputs, modelspec, monkeypatch)
    _on_stack(monkeypatch, **{**OLD_STACK, "vllm": "0.28.0+cu129", "torch": "2.13.0+cu129"})
    resumed = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    resumed.allow_stack_change = True
    with caplog.at_level("WARNING"):
        assert resumed.run()["state"] == STATE_COMPLETE
    runs = read_jsonl(resumed.draw_dir / RUNS_FNAME)
    # the first run, the resuming run, then the change it logged before recording its first attempt
    assert ["stack_change" in row for row in runs] == [False, False, True]
    assert runs[2]["stack_change"] == {"vllm": ["0.19.0", "0.28.0+cu129"], "torch": ["2.10.0+cu128", "2.13.0+cu129"]}
    assert runs[2]["runtime"]["vllm_version"] == "0.28.0+cu129" and "recorded_at" in runs[2]
    assert [runtime["vllm_version"] for runtime in judgment_runtimes(resumed.draw_dir)] == ["0.19.0", "0.28.0+cu129"]
    assert "as allowed" in caplog.text
    assert json.loads((resumed.draw_dir / DRAW_PROVENANCE_FNAME).read_text())["runtime"]["vllm_version"] == "0.19.0"
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()["state"] == STATE_COMPLETE


def test_a_complete_draw_is_not_refused_on_another_stack(tmp_path, inputs, modelspec, monkeypatch):
    _on_stack(monkeypatch, **OLD_STACK)
    build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()
    _on_stack(monkeypatch, **NEW_VLLM)
    backend = FakeBackend()
    again = build_runner(tmp_path, inputs, modelspec, backend)
    assert again.run()["state"] == STATE_COMPLETE and backend.calls == {}
    assert all("stack_change" not in row for row in read_jsonl(again.draw_dir / RUNS_FNAME))


def test_a_draw_with_provenance_but_no_attempt_starts_on_the_new_stack(tmp_path, inputs, modelspec, monkeypatch, caplog):
    _on_stack(monkeypatch, **OLD_STACK)
    stopped = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    stopped.prepare()
    assert _condition_record(stopped)["stack"]["vllm"] == "0.19.0"
    _on_stack(monkeypatch, **NEW_VLLM)
    with caplog.at_level("INFO"):
        assert build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()["state"] == STATE_COMPLETE
    assert "re-stamped" in caplog.text
    provenance = json.loads((stopped.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance["runtime"]["vllm_version"] == "0.28.0"
    record = _condition_record(stopped)
    assert _stack(record) == NEW_VLLM
    assert record["source"] == "draw-000: first run"
    _interrupted(tmp_path, inputs, modelspec, monkeypatch, draw=1, **NEW_VLLM)
    written = (stopped.draw_dir.parent / "draw-001" / DRAW_PROVENANCE_FNAME).read_text()
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=1).run()["state"] == STATE_COMPLETE
    assert (stopped.draw_dir.parent / "draw-001" / DRAW_PROVENANCE_FNAME).read_text() == written


def test_a_new_draw_of_a_condition_judged_on_another_stack_is_refused(tmp_path, inputs, modelspec, monkeypatch):
    _on_stack(monkeypatch, **OLD_STACK)
    first = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    first.run()
    record = _condition_record(first)
    assert record["condition_id"] == "fake/naive" and record["source"] == "draw-000: first run"
    assert _stack(record) == OLD_STACK
    assert record["runtime"]["vllm_version"] == "0.19.0"

    _on_stack(monkeypatch, **NEW_VLLM)
    second = build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=1)
    with pytest.raises(CampaignRunError) as error:
        second.run()
    message = str(error.value)
    assert ("condition 'fake/naive' holds judgments recorded on another inference stack or engine "
            "(vllm 0.19.0 -> 0.28.0") in message
    assert "refusing to start its draw 1" in message and "execution.allow_stack_change=true" in message
    assert not second.draw_dir.exists()

    second.allow_stack_change = True
    assert second.run()["state"] == STATE_COMPLETE
    assert read_jsonl(second.draw_dir / RUNS_FNAME)[1]["stack_change"] == {"vllm": ["0.19.0", "0.28.0"]}
    assert _condition_record(first) == record


def test_a_new_draw_with_other_engine_arguments_is_refused_but_not_at_another_placement(
    tmp_path, inputs, modelspec, monkeypatch
):
    _on_stack(monkeypatch, **OLD_STACK)
    backend = FakeBackend()
    backend.engine_args = {"model": "/a/Qwen3-8B", "tensor_parallel_size": 1, "max_num_seqs": 256,
                           "max_model_len": 40960, "enable_prefix_caching": False}
    build_runner(tmp_path, inputs, modelspec, backend).run()
    moved = FakeBackend()
    moved.engine_args = {**backend.engine_args, "model": "/b/Qwen3-8B", "tensor_parallel_size": 2, "max_num_seqs": 512}
    assert build_runner(tmp_path, inputs, modelspec, moved, draw=1).run()["state"] == STATE_COMPLETE
    other = FakeBackend()
    other.engine_args = {**backend.engine_args, "max_model_len": 8192}
    with pytest.raises(CampaignRunError, match=r"engine_args.max_model_len 40960 -> 8192"):
        build_runner(tmp_path, inputs, modelspec, other, draw=2).run()
    allowed = build_runner(tmp_path, inputs, modelspec, other, draw=2)
    allowed.allow_stack_change = True
    assert allowed.run()["state"] == STATE_COMPLETE
    assert read_jsonl(allowed.draw_dir / RUNS_FNAME)[1]["stack_change"] == {"engine_args.max_model_len": [40960, 8192]}
    assert len(condition_signatures(allowed.condition_dir)) == 2


def _second_draw(tmp_path, inputs, modelspec, monkeypatch, *, first, then):
    _on_stack(monkeypatch, **OLD_STACK)
    placed = {"model": "/a/Qwen3-8B", "tensor_parallel_size": 1}
    backend = FakeBackend()
    backend.engine_args = {**placed, **first}
    build_runner(tmp_path, inputs, modelspec, backend).run()
    other = FakeBackend()
    other.engine_args = {**placed, **then}
    return build_runner(tmp_path, inputs, modelspec, other, draw=1)


def test_an_engine_argument_recorded_as_none_is_unset_to_the_runner_and_the_analysis(
    tmp_path, inputs, modelspec, monkeypatch
):
    second = _second_draw(tmp_path, inputs, modelspec, monkeypatch, first={"enable_prefix_caching": None}, then={})
    assert second.run()["state"] == STATE_COMPLETE
    assert len(condition_signatures(second.condition_dir)) == 1


@pytest.mark.parametrize("first, then, refused", [
    # canonical JSON, the digest's own predicate: 4.0 is not 4, nor 0 False
    ({"max_model_len": 4}, {"max_model_len": 4.0}, "engine_args.max_model_len 4 -> 4.0"),
    ({"enforce_eager": False}, {"enforce_eager": 0}, "engine_args.enforce_eager False -> 0"),
])
def test_an_engine_argument_the_runner_refuses_mixes_the_condition_for_the_analysis(
    tmp_path, inputs, modelspec, monkeypatch, first, then, refused
):
    second = _second_draw(tmp_path, inputs, modelspec, monkeypatch, first=first, then=then)
    with pytest.raises(CampaignRunError, match=re.escape(f"({refused}; from draw-000: first run)")):
        second.run()
    second.allow_stack_change = True
    assert second.run()["state"] == STATE_COMPLETE
    args_by_digest = {}
    signatures = condition_signatures(second.condition_dir, args_by_digest=args_by_digest)
    [record] = mixed_conditions({"c": signatures}, args_by_digest=args_by_digest)
    assert record["reason"].endswith(f"; {refused})")


def test_a_provider_judge_records_a_precision_setting_vllm_would_refuse(tmp_path, inputs, modelspec, monkeypatch):
    refused = {"VLLM_BATCH_INVARIANT": "true", "VLLM_FLOAT32_MATMUL_PRECISION": "fastest"}
    for name, value in refused.items():
        monkeypatch.setenv(name, value)
    provider = build_runner(tmp_path / "p", inputs, modelspec, FakeBackend(model_type="openai_batch"))
    provider.prepare()  # provenance, no attempt: the job is killed here
    assert not provider.ledger.load().records
    resumed = build_runner(tmp_path / "p", inputs, modelspec, FakeBackend(model_type="openai_batch"))
    assert resumed.run()["state"] == STATE_COMPLETE
    runtime = json.loads((resumed.draw_dir / DRAW_PROVENANCE_FNAME).read_text())["runtime"]
    assert runtime["precision_env"] == {**refused, "TRITON_F32_DEFAULT": "unset"}
    with pytest.raises(ValueError, match="VLLM_FLOAT32_MATMUL_PRECISION='fastest' is not one of"):
        build_runner(tmp_path / "l", inputs, modelspec, FakeBackend()).run()


def test_a_model_config_differing_only_in_its_engine_stats_resumes_the_other_s_draws(
    tmp_path, inputs, modelspec, monkeypatch
):
    """``disable_log_stats`` says how an engine runs, not what it computes (placement)."""
    from omegaconf import OmegaConf

    from ruleforge.judges.backend import local_engine_settings
    from ruleforge.judges.models import local_engine_kwargs

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/models")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    config = OmegaConf.load(REPO_ROOT / "configs" / "model" / "gemma4-31b-tp-2.yaml")
    backends = []
    for disabled in (True, False):
        backend = FakeBackend()
        settings = local_engine_settings(OmegaConf.merge(config, {"disable_log_stats": disabled}), engine_fields=())
        backend.engine_args = local_engine_kwargs(**settings)
        backends.append(backend)
    without, with_stats = (backend.engine_args for backend in backends)
    assert {key for key in without if without[key] != with_stats[key]} == {"disable_log_stats"}
    _interrupted(tmp_path, inputs, modelspec, monkeypatch, backend=backends[0])
    resumed = build_runner(tmp_path, inputs, modelspec, backends[1])
    assert resumed.run()["state"] == STATE_COMPLETE
    assert all("stack_change" not in row for row in read_jsonl(resumed.draw_dir / RUNS_FNAME))


def test_an_allowed_change_that_recorded_no_attempt_leaves_no_stack_behind(tmp_path, inputs, modelspec, monkeypatch):
    first = _interrupted(tmp_path, inputs, modelspec, monkeypatch)
    _on_stack(monkeypatch, **NEW_VLLM)
    preempted = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    preempted.allow_stack_change = True
    preempted.prepare()  # the job is killed here
    assert all("stack_change" not in row for row in read_jsonl(first.draw_dir / RUNS_FNAME))
    _on_stack(monkeypatch, **OLD_STACK)
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()["state"] == STATE_COMPLETE
    assert [runtime["vllm_version"] for runtime in judgment_runtimes(first.draw_dir)] == ["0.19.0"]
    assert condition_signatures(first.condition_dir) == [judgment_signature(judgment_runtimes(first.draw_dir)[0])]


def test_a_condition_recorded_before_the_stack_record_existed_takes_its_lowest_judged_draws(
    tmp_path, inputs, modelspec, monkeypatch
):
    _on_stack(monkeypatch, **OLD_STACK)
    build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=1).run()
    stopped = build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=0)
    stopped.prepare()
    (stopped.draw_dir.parent / CONDITION_STACK_FNAME).unlink()
    _on_stack(monkeypatch, **NEW_VLLM)
    with pytest.raises(CampaignRunError, match=r"from draw-001/draw_provenance.json\)"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=0).run()
    _on_stack(monkeypatch, **OLD_STACK)
    assert build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=0).run()["state"] == STATE_COMPLETE
    record = _condition_record(stopped)
    assert record["source"] == "draw-001/draw_provenance.json" and record["stack"]["vllm"] == "0.19.0"


def test_a_condition_record_no_judgment_stands_behind_is_replaced(tmp_path, inputs, modelspec, monkeypatch):
    _on_stack(monkeypatch, **OLD_STACK)
    stopped = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    stopped.prepare()
    assert _condition_record(stopped)["stack"]["vllm"] == "0.19.0"
    before = _condition_record(stopped)
    build_runner(tmp_path, inputs, modelspec, FakeBackend()).prepare()
    assert _condition_record(stopped) == before
    _on_stack(monkeypatch, **NEW_VLLM)
    build_runner(tmp_path, inputs, modelspec, FakeBackend(), draw=1).prepare()
    assert _condition_record(stopped)["stack"]["vllm"] == "0.28.0"


def test_an_unrecorded_field_is_no_stack_change_but_a_local_build_label_is(tmp_path, inputs, modelspec, monkeypatch):
    # a first run recorded without transformers_version
    _interrupted(tmp_path, inputs, modelspec, monkeypatch, vllm="0.19.0", torch="2.10.0", transformers=None)
    _on_stack(monkeypatch, vllm="0.19.0", torch="2.10.0+cu128", transformers="5.17.0")
    with pytest.raises(CampaignRunError, match=r"torch 2.10.0 -> 2.10.0\+cu128\)"):
        build_runner(tmp_path, inputs, modelspec, FakeBackend()).run()
    _on_stack(monkeypatch, vllm="0.19.0", torch="2.10.0", transformers="5.17.0")
    resumed = build_runner(tmp_path, inputs, modelspec, FakeBackend())
    assert resumed.run()["state"] == STATE_COMPLETE
    assert all("stack_change" not in row for row in read_jsonl(resumed.draw_dir / RUNS_FNAME))


def test_a_provider_judges_draws_are_not_checked_for_the_local_stack(tmp_path, inputs, modelspec, monkeypatch):
    provider = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
    first = _interrupted(tmp_path, inputs, modelspec, monkeypatch, backend=provider)
    _on_stack(monkeypatch, **NEW_VLLM)
    assert build_runner(tmp_path, inputs, modelspec, provider).run()["state"] == STATE_COMPLETE
    assert build_runner(tmp_path, inputs, modelspec, provider, draw=1).run()["state"] == STATE_COMPLETE
    assert _condition_record(first) is None


def test_the_runtime_records_transformers_and_the_engine_arguments(tmp_path, inputs, modelspec):
    backend = FakeBackend()
    backend.engine_args = {"model": Path("/models/x"), "enable_lora": False, "max_model_len": 4096}
    runner = build_runner(tmp_path, inputs, modelspec, backend)
    assert runner.run()["state"] == STATE_COMPLETE
    runtime = read_jsonl(runner.draw_dir / RUNS_FNAME)[0]["runtime"]
    assert "transformers_version" in runtime
    assert runtime["engine_args"] == {"model": "/models/x", "enable_lora": False, "max_model_len": 4096}
    provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance["runtime"]["engine_args"]["max_model_len"] == 4096
    assert _condition_record(runner)["runtime"]["engine_args"]["max_model_len"] == 4096
    backend.engine_args = {"model": "/elsewhere/x"}
    assert build_runner(tmp_path, inputs, modelspec, backend).run()["state"] == STATE_COMPLETE
