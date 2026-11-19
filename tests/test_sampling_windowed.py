from __future__ import annotations

import json

import pytest

from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.ledger import DrawLedger
from ruleforge.evaluate.runner import REQUEST_BLOCKS_FNAME, BlockWork, DrawRunner, STATE_COMPLETE, TransportExhausted
from ruleforge.judges.backend import GenerationOutcome
from ruleforge.judges.batch_api import BatchExecutionError

from tests.test_sampling_runner import (
    FakeBackend,
    N_SCENARIOS,
    RULES,
    make_spec,
    sent_block,
    write_inputs,
)


class WindowFakeBackend(FakeBackend):
    """A fake ``openai_batch`` judge that windows: submit/poll/estimate, no network."""

    def __init__(self, *, max_enqueued_tokens, tokens_per_block=100, polls_needed=None,
                 script=None, style="naive", submit_failures_before_ok=0, submit_fatal=False,
                 poll_failures_before_ok=0, poll_fatal=False, batch_timeout_seconds=None):
        super().__init__(model_type="openai_batch", seed_capability="unsupported",
                         script=script, style=style)
        self.max_enqueued_tokens = max_enqueued_tokens
        self.batch_poll_seconds = 0.0
        self.batch_timeout_seconds = batch_timeout_seconds
        self._poll_request_block = lambda *a, **k: None
        self.tokens_per_block = tokens_per_block
        self.polls_needed = polls_needed or {}
        self.submit_failures_before_ok = submit_failures_before_ok
        self.submit_fatal = submit_fatal
        self.poll_failures_before_ok = poll_failures_before_ok
        self.poll_fatal = poll_fatal
        self.blocks_state: dict[str, dict] = {}
        self.submit_order: list[str] = []
        self.submit_counts: dict[str, int] = {}
        self.poll_raises: dict[str, int] = {}
        self.enqueued = 0
        self.peak_enqueued = 0
        self.poll_order: list[str] = []

    def _answer(self, entry):
        key = self.locate(entry)
        n = self.calls.get(key, 0)
        self.calls[key] = n + 1
        entries = self.script.get(key)
        value = entries[min(n, len(entries) - 1)] if entries else self._default(key[1])
        if isinstance(value, str):
            value = GenerationOutcome(
                text=value, raw_response=f"<think>{key}</think>{value}", finish_reason="stop"
            )
        return value

    def estimate_block_tokens(self, prompts, *, temperature, seeds=None, request_block_id=""):
        assert temperature == 1.0
        return self.tokens_per_block

    def submit_block(self, prompts, *, temperature, seeds=None, request_block_id=""):
        assert temperature == 1.0
        assert seeds is None
        self.submit_counts[request_block_id] = self.submit_counts.get(request_block_id, 0) + 1
        if self.submit_fatal:
            raise BatchExecutionError("a batch shard ended failed")
        if self.submit_counts[request_block_id] <= self.submit_failures_before_ok:
            raise RuntimeError("transient submit failure")
        if request_block_id in self.blocks_state:
            return
        self.blocks_state[request_block_id] = {"prompts": list(prompts), "polls": 0, "done": False}
        self.submit_order.append(request_block_id)
        self.enqueued += self.tokens_per_block
        self.peak_enqueued = max(self.peak_enqueued, self.enqueued)

    def poll_block(self, prompts, *, temperature, seeds=None, request_block_id=""):
        if self.poll_fatal:
            raise BatchExecutionError("a batch shard ended failed")
        raises = self.poll_raises.get(request_block_id, 0)
        if raises < self.poll_failures_before_ok:
            self.poll_raises[request_block_id] = raises + 1
            raise RuntimeError("transient poll failure")
        rec = self.blocks_state[request_block_id]
        rec["polls"] += 1
        index = self.submit_order.index(request_block_id)
        if rec["polls"] < self.polls_needed.get(index, 1):
            return None
        if not rec["done"]:
            rec["done"] = True
            self.enqueued -= self.tokens_per_block
            self.poll_order.append(request_block_id)
        return [self._answer(entry) for entry in rec["prompts"]]


def _inputs(tmp_path):
    path = write_inputs(tmp_path)
    manifest = ScenarioManifest.build(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return manifest, rows


def build_windowed_runner(tmp_path, inputs, modelspec, backend, *, style="naive",
                          sleep=None, monotonic=None, **overrides):
    manifest, rows = inputs
    spec = make_spec(manifest, style=style, seed_capability=backend.seed_capability, **overrides)
    clock = iter(f"2026-09-11T00:00:{i:02d}+00:00" for i in range(100_000))
    kwargs = {}
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    return DrawRunner(
        backend=backend,
        task_spec=modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0),
        spec=spec,
        draw_dir=tmp_path / "camp" / "luna" / style / "draw-000",
        scenario_rows=rows,
        rules=RULES,
        clock=lambda: next(clock),
        code_revision="rev",
        sleep=sleep or (lambda _seconds: None),
        **kwargs,
    )


@pytest.fixture
def inputs(tmp_path):
    return _inputs(tmp_path)


def _accepted_judgments(draw_dir, manifest, campaign_id="camp", condition_id="fake/naive"):
    state = DrawLedger(
        draw_dir, campaign_id=campaign_id, condition_id=condition_id, decoder_draw_id=0
    ).load(repair=True)
    return {
        sid: state.accepted[sid].components[0].judgment
        for sid in manifest.scenario_ids
        if state.is_accepted(sid)
    }


def test_the_window_never_exceeds_the_budget_and_saturates(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(max_enqueued_tokens=200, tokens_per_block=100)
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1
    )
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert backend.peak_enqueued == 200
    assert backend.peak_enqueued <= backend.max_enqueued_tokens
    assert len(backend.submit_order) == N_SCENARIOS
    assert all(count == 1 for count in backend.submit_counts.values())


def test_a_block_larger_than_the_whole_budget_still_runs_alone(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(max_enqueued_tokens=50, tokens_per_block=100)
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1
    )
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert backend.peak_enqueued == 100


def test_out_of_order_completion_records_every_block(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(
        max_enqueued_tokens=1000, tokens_per_block=100, polls_needed={0: 2, 1: 1}
    )
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1
    )
    completion = runner.run()
    assert completion["state"] == STATE_COMPLETE
    assert backend.poll_order[0] == backend.submit_order[1]
    manifest, _ = inputs
    assert len(_accepted_judgments(runner.draw_dir, manifest)) == N_SCENARIOS


def test_windowed_and_sequential_reach_identical_results(tmp_path, modelspec):
    script = {
        (0, "Rule 1"): ["yes"],
        (1, "full_ruleset"): ["not-a-label", "no"],
    }

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    seq_inputs = _inputs(seq_dir)
    seq_backend = FakeBackend(model_type="openai_batch", seed_capability="unsupported", script=dict(script))
    seq_runner = build_windowed_runner(seq_dir, seq_inputs, modelspec, seq_backend)
    seq_completion = seq_runner.run()

    win_dir = tmp_path / "win"
    win_dir.mkdir()
    win_inputs = _inputs(win_dir)
    win_backend = WindowFakeBackend(max_enqueued_tokens=1000, tokens_per_block=100, script=dict(script))
    win_runner = build_windowed_runner(win_dir, win_inputs, modelspec, win_backend)
    win_completion = win_runner.run()

    assert seq_completion["state"] == win_completion["state"] == STATE_COMPLETE
    assert _accepted_judgments(seq_runner.draw_dir, seq_inputs[0]) == _accepted_judgments(
        win_runner.draw_dir, win_inputs[0]
    )


class SeededWindowBackend(WindowFakeBackend):
    """A windowed judge that takes seeds, logging what each block is sent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_capability = "local_deterministic"
        self.sent: dict[tuple[str, str], tuple[list, list]] = {}

    def _log(self, method, prompts, seeds, request_block_id):
        assert seeds is not None and len(seeds) == len(prompts)
        self.sent.setdefault((method, request_block_id), (list(prompts), list(seeds)))

    def estimate_block_tokens(self, prompts, *, temperature, seeds=None, request_block_id=""):
        self._log("estimate", prompts, seeds, request_block_id)
        return super().estimate_block_tokens(prompts, temperature=temperature, request_block_id=request_block_id)

    def submit_block(self, prompts, *, temperature, seeds=None, request_block_id=""):
        self._log("submit", prompts, seeds, request_block_id)
        return super().submit_block(prompts, temperature=temperature, request_block_id=request_block_id)

    def poll_block(self, prompts, *, temperature, seeds=None, request_block_id=""):
        self._log("poll", prompts, seeds, request_block_id)
        return super().poll_block(prompts, temperature=temperature, request_block_id=request_block_id)


def test_the_window_sends_exactly_what_build_request_builds(tmp_path, inputs, modelspec):
    script = {(1, "full_ruleset"): ["not-a-label", "no"]}
    backend = SeededWindowBackend(max_enqueued_tokens=250, tokens_per_block=100, script=script)
    runner = build_windowed_runner(tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1)
    assert runner.run()["state"] == STATE_COMPLETE
    rows = [json.loads(line) for line in (runner.draw_dir / REQUEST_BLOCKS_FNAME).read_text().splitlines()]
    assert {row["semantic_attempt_id"] for row in rows} == {0, 1}
    for row in rows:
        members = tuple(row["scenario_ids"])
        request = runner.build_request(BlockWork(row["request_block_id"], row["semantic_attempt_id"], members, members))
        expected = (list(request.prompts), request.seed_list())
        for method in ("estimate", "submit", "poll"):
            assert backend.sent[(method, sent_block(row["request_block_id"]))] == expected
    ledger = [json.loads(line) for line in (runner.draw_dir / "attempts.jsonl").read_text().splitlines()]
    sent = {seed for _, seeds in backend.sent.values() for seed in seeds}
    assert {c["generation_seed"] for row in ledger for c in row["components"]} == sent


def test_a_completed_windowed_draw_resumes_without_resubmitting(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(max_enqueued_tokens=1000, tokens_per_block=100)
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1
    )
    assert runner.run()["state"] == STATE_COMPLETE
    first_pass_submits = dict(backend.submit_counts)

    resumed_backend = WindowFakeBackend(max_enqueued_tokens=1000, tokens_per_block=100)
    resumed = build_windowed_runner(
        tmp_path, inputs, modelspec, resumed_backend, scenarios_per_request_block=1
    )
    assert resumed.run()["state"] == STATE_COMPLETE
    assert resumed_backend.submit_counts == {}
    assert first_pass_submits


def test_the_window_sleeps_between_polls_while_a_block_is_in_flight(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(max_enqueued_tokens=1000, tokens_per_block=100, polls_needed={0: 4})
    sleeps = []
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS,
        sleep=lambda seconds: sleeps.append(seconds),
    )
    assert runner.run()["state"] == STATE_COMPLETE
    assert len(sleeps) >= 2


def test_a_transient_submit_failure_is_retried_then_succeeds(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(
        max_enqueued_tokens=1000, tokens_per_block=100, submit_failures_before_ok=1
    )
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS
    )
    assert runner.run()["state"] == STATE_COMPLETE
    assert max(backend.submit_counts.values()) == 2


def test_repeated_submit_failures_stop_the_draw_but_leave_it_resumable(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(
        max_enqueued_tokens=1000, tokens_per_block=100, submit_failures_before_ok=99
    )
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS
    )
    with pytest.raises(TransportExhausted, match="failed to submit"):
        runner.run()


def test_a_fatal_submit_error_surfaces_immediately(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(max_enqueued_tokens=1000, tokens_per_block=100, submit_fatal=True)
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS
    )
    with pytest.raises(BatchExecutionError):
        runner.run()


def test_a_budget_without_wired_windowing_stays_on_the_sequential_path(tmp_path, inputs, modelspec):
    backend = FakeBackend(model_type="openai_batch", seed_capability="unsupported")
    backend.max_enqueued_tokens = 5000  # a budget, but no _poll_request_block wired
    runner = build_windowed_runner(tmp_path, inputs, modelspec, backend)
    assert runner._windowed() is False
    assert runner.run()["state"] == STATE_COMPLETE


def test_a_transient_poll_failure_is_retried_then_succeeds(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(
        max_enqueued_tokens=1000, tokens_per_block=100, poll_failures_before_ok=1
    )
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS
    )
    assert runner.run()["state"] == STATE_COMPLETE
    assert all(raised == 1 for raised in backend.poll_raises.values())


def test_repeated_poll_failures_stop_the_draw_but_leave_it_resumable(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(
        max_enqueued_tokens=1000, tokens_per_block=100, poll_failures_before_ok=99
    )
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS
    )
    with pytest.raises(TransportExhausted, match="failed to poll"):
        runner.run()


def test_a_fatal_poll_error_surfaces_immediately(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(max_enqueued_tokens=1000, tokens_per_block=100, poll_fatal=True)
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS
    )
    with pytest.raises(BatchExecutionError):
        runner.run()


def test_the_batch_timeout_gives_up_and_stays_resumable(tmp_path, inputs, modelspec):
    backend = WindowFakeBackend(
        max_enqueued_tokens=1000, tokens_per_block=100, polls_needed={0: 99},
        batch_timeout_seconds=10,
    )
    ticks = iter([0.0, 0.0, 0.0, 100.0])  # deadline set at 0; the third wait is past it
    runner = build_windowed_runner(
        tmp_path, inputs, modelspec, backend, scenarios_per_request_block=N_SCENARIOS,
        monotonic=lambda: next(ticks, 100.0),
    )
    with pytest.raises(BatchExecutionError, match="did not finish within"):
        runner.run()
