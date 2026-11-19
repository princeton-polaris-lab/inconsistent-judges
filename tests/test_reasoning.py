from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from ruleforge.evaluate.ledger import RAW_OUTPUTS_FNAME
from ruleforge.evaluate.validation import EMPTY_RESPONSE, ComponentSchema
from ruleforge.evaluate.outputs import RawOutput
from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.evaluate.identity import DRAW_PROVENANCE_FNAME
from ruleforge.evaluate.runner import (
    PROVENANCE_RUN_FIELDS,
    REQUEST_BLOCKS_FNAME,
    REQUEST_SETTINGS_KEY,
    STATE_COMPLETE,
    UNRECORDED_REQUEST_SETTINGS,
    CampaignRunError,
)
from ruleforge.judges import JudgeSpec, anthropic_batch, batch_api, models, openai_batch
from ruleforge.judges.anthropic_batch import (
    DISPLAY_SUMMARIZED,
    THINKING_DISPLAYS,
    build_request_params,
    display_setting,
    outcome_from_message,
)
from ruleforge.judges.backend import STRIP_FAILED_KEY, GenerationOutcome, ModelBackend
from ruleforge.judges.base import JudgeError
from ruleforge.judges.batch_api import (
    CHAT_COMPLETIONS_ENDPOINT,
    REASONING_KEY,
    REASONING_SUMMARIES_KEY,
    RESPONSES_ENDPOINT,
    BatchExecutionError,
    BatchOutcome,
    responses_summaries_requested,
    summary_fields,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from tests.test_anthropic_batch import FakeAnthropic, message, succeeded, text, thinking
from tests.test_openai_batch import FakeOpenAI, success_record
from tests.test_sampling_runner import FakeBackend, build_runner, read_jsonl, sent_block, sid_of, write_inputs

HAIKU = "claude-haiku-5-5"
PROMPTS = [{"system_prompt": f"judge {i}", "user_prompt": ""} for i in range(3)]
PROMPT = PROMPTS[:1]
SCHEMA = ComponentSchema.from_rules([{"id": "Rule 1", "rule": "be brief"}], prompt_style="naive",
                                    forced_json_requirement="label_only")
BLOCK = "camp/cond/draw-000/attempt-0/block-0000"


def summarized(summary: str) -> Dict[str, Any]:
    """A thinking block as the summarized display returns it."""
    return {"type": "thinking", "thinking": summary, "signature": "sig"}


@pytest.fixture(autouse=True)
def _no_spend_no_key_no_gate(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                 *anthropic_batch.REFUSED_SDK_ENVS):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def confirmed(monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)


@pytest.fixture
def inputs(tmp_path):
    """The runner tests' scenarios and their manifest."""
    path = write_inputs(tmp_path)
    return ScenarioManifest.build(path), [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("endpoint, effort, summary, expected", [
    (RESPONSES_ENDPOINT, "medium", "auto", True),
    (RESPONSES_ENDPOINT, None, "auto", False),  # the body carries a summary only beside an effort
    (RESPONSES_ENDPOINT, "medium", None, False),
    (CHAT_COMPLETIONS_ENDPOINT, "medium", "auto", False),  # chat completions has no summary block
])
def test_a_request_asks_for_a_summary_only_where_its_body_carries_one(endpoint, effort, summary, expected):
    assert responses_summaries_requested(endpoint, effort, summary) is expected
    if endpoint == RESPONSES_ENDPOINT:
        body = batch_api.build_responses_body("p", "", model="m", temperature=1.0, reasoning_effort=effort,
                                              reasoning_summary=summary)
        assert ("summary" in (body.get("reasoning") or {})) is expected


def test_a_detail_carries_summaries_when_asked_and_the_reasoning_text_when_one_came_back():
    outcome = BatchOutcome(text="no", reasoning_summaries=("first", "second"))
    assert summary_fields(outcome, True) == {REASONING_SUMMARIES_KEY: ["first", "second"]}
    assert summary_fields(BatchOutcome(text="no"), True) == {REASONING_SUMMARIES_KEY: []}
    assert summary_fields(outcome, False) == {}
    assert summary_fields(BatchOutcome(text="no", reasoning="all of it"), False) == {REASONING_KEY: "all of it"}
    assert batch_api.reasoning_fields({"completion": "x", REASONING_KEY: "r", "other": 1}) == {REASONING_KEY: "r"}


def _record(body):
    return {"custom_id": "c", "response": {"status_code": 200, "body": body}}


def test_a_chat_body_s_reasoning_is_kept_apart_from_its_answer():
    body = {"choices": [{"message": {"content": " yes ", "reasoning": "I checked rule 2"}, "finish_reason": "stop"}]}
    outcome, _ = batch_api.outcome_from_record(_record(body), CHAT_COMPLETIONS_ENDPOINT)
    assert outcome == BatchOutcome(text="yes", finish_reason="stop", reasoning="I checked rule 2")


def test_a_responses_body_s_reasoning_text_rides_beside_its_summaries():
    body = {"status": "completed", "output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "short"}],
         "content": [{"type": "reasoning_text", "text": "long"}, {"type": "other", "text": "x"}]},
        {"type": "message", "content": [{"type": "output_text", "text": "no"}]}]}
    outcome, _ = batch_api.outcome_from_record(_record(body), RESPONSES_ENDPOINT)
    assert (outcome.text, outcome.reasoning_summaries, outcome.reasoning) == ("no", ("short",), "long")


@pytest.mark.parametrize("message_fields, expected", [
    ({"reasoning": "thought it through"}, "thought it through"),
    ({"reasoning": "  ", "reasoning_content": "from a vllm server"}, "from a vllm server"),
    ({"reasoning_details": [{"type": "reasoning.summary", "summary": "in short"},
                            {"type": "reasoning.encrypted", "data": "opaque"},
                            {"type": "reasoning.text", "text": "at length"},
                            {"type": ["unhashable"]}, "junk"]}, "in short\nat length"),
    ({"reasoning_details": "not a list"}, None),
    ({}, None),
])
def test_a_message_s_reasoning_is_read_from_the_provider_s_fields(message_fields, expected):
    assert batch_api.message_reasoning({"content": "yes", **message_fields}) == expected
    assert batch_api.message_reasoning(SimpleNamespace(content="yes", **message_fields)) == expected


@pytest.mark.parametrize("body", [{}, {"choices": "junk"}, {"choices": [None]}, {"choices": [{"message": "x"}]},
                                  {"output": {"type": "reasoning"}},
                                  {"output": [{"type": "reasoning", "content": "junk"}]}])
def test_a_body_of_another_shape_carries_no_reasoning_and_never_raises(body):
    assert batch_api.chat_reasoning(body) is None
    assert batch_api.responses_reasoning(body) is None
    assert batch_api.reasoning_summaries(body) == []


@pytest.mark.parametrize("value, expected", [
    (None, None), ("summarized", "summarized"), (" Summarized ", "summarized"), ("omitted", "omitted")])
def test_a_config_s_display_is_read_as_thinking_display_carries_it(value, expected):
    assert display_setting(value) == expected
    assert THINKING_DISPLAYS == ("summarized", "omitted")


@pytest.mark.parametrize("value", ["updates", "raw", ""])
def test_a_display_the_transport_does_not_send_is_refused(value):
    with pytest.raises(ValueError, match="unknown thinking display"):
        display_setting(value)


def test_the_summarized_display_changes_the_request_s_thinking_and_nothing_else():
    plain = build_request_params("sys", "", model=HAIKU, max_tokens=16384, effort="medium")
    shown = build_request_params("sys", "", model=HAIKU, max_tokens=16384, effort="medium",
                                 display=DISPLAY_SUMMARIZED)
    assert plain["thinking"] == {"type": "adaptive"}
    assert shown["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert {k: v for k, v in shown.items() if k != "thinking"} == {k: v for k, v in plain.items() if k != "thinking"}
    assert anthropic_batch.THINKING == {"type": "adaptive"}


def test_a_summarized_message_keeps_its_answer_and_its_summaries_apart():
    shown = outcome_from_message(message(summarized("weighed rule 1"), thinking(), summarized("then rule 2"),
                                         text("no")))
    omitted = outcome_from_message(message(thinking(), text("no")))
    assert (shown.text, shown.finish_reason, shown.error) == (omitted.text, omitted.finish_reason, omitted.error)
    assert (shown.text, shown.reasoning_summaries) == ("no", ("weighed rule 1", "then rule 2"))
    assert omitted.reasoning_summaries == ()


def test_a_refusal_a_textless_answer_and_a_truncated_one_keep_their_summaries():
    refused = outcome_from_message(message(summarized("considered it"), text("x"), stop="refusal",
                                           stop_details={"category": "bio"}))
    assert (refused.text, refused.finish_reason, refused.reasoning_summaries) == (
        None, "refusal:bio", ("considered it",))
    silent = outcome_from_message(message(summarized("ran out"), stop="max_tokens"))
    assert (silent.text, silent.finish_reason, silent.reasoning_summaries) == (None, "max_tokens", ("ran out",))


def _haiku_responder(cid: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Prompt 0 answers with a summary, prompt 1 with a blank thinking block, prompt 2 refuses after one."""
    if cid.endswith("p000000-c00"):
        return succeeded(cid, summarized(f"reasoning::{cid}"), text("no"))
    if cid.endswith("p000001-c00"):
        return succeeded(cid, thinking(), text("yes"))
    return succeeded(cid, summarized("declined"), stop="refusal", stop_details={"category": "cyber"})


def test_the_block_callable_asks_for_summaries_and_hands_each_on_beside_its_answer(tmp_path, confirmed):
    client = FakeAnthropic(responder=_haiku_responder)
    block = anthropic_batch.make_block_generate(model_name=HAIKU, root=tmp_path / "batch", reasoning_effort="medium",
                                                thinking_display="summarized", client_factory=lambda: client)
    details = block(PROMPTS, 1.0, 16384, None, BLOCK)
    [lines] = client.create_calls
    assert {json.dumps(line["params"]["thinking"], sort_keys=True) for line in lines} == {
        '{"display": "summarized", "type": "adaptive"}'}
    namespace = details[0]["provider_request_ref"].rsplit("-p", 1)[0]
    assert [(d["raw_response"], d["completion"], d["finish_reason"], d[REASONING_SUMMARIES_KEY]) for d in details] == [
        ("no", "no", "end_turn", [f"reasoning::{namespace}-p000000-c00"]),
        ("yes", "yes", "end_turn", []),
        (None, None, "refusal:cyber", ["declined"]),
    ]
    sidecar = Path(details[0]["request_block_ref"]) / "reasoning_summaries.jsonl"
    assert [json.loads(line)["custom_id"].rsplit("-", 2)[1] for line in sidecar.read_text().splitlines()] == [
        "p000000", "p000002"]


def test_the_answers_and_the_usage_are_the_omitted_display_s(tmp_path, confirmed):
    runs = {}
    for display in (None, "summarized"):
        client = FakeAnthropic(responder=_haiku_responder)
        block = anthropic_batch.make_block_generate(
            model_name=HAIKU, root=tmp_path / str(display), reasoning_effort="medium", thinking_display=display,
            client_factory=lambda: client)
        details = block(PROMPTS, 1.0, 16384, None, BLOCK)
        usage = json.loads((Path(details[0]["request_block_ref"]) / "usage.json").read_text())
        runs[display] = ([(d["completion"], d["finish_reason"]) for d in details], usage)
        assert (REASONING_SUMMARIES_KEY in details[0]) is (display is not None)
    assert runs[None] == runs["summarized"]


@pytest.mark.parametrize("display, expected", [("summarized", ["why"]), ("omitted", None), (None, None)])
def test_the_detailed_batch_callable_carries_the_summaries_beside_its_text(tmp_path, confirmed, display, expected):
    client = FakeAnthropic(responder=lambda cid, params: succeeded(cid, summarized("why"), text("yes")))
    _, detailed = anthropic_batch.make_batch_generate(model_name=HAIKU, root=tmp_path, thinking_display=display,
                                                      client_factory=lambda: client)
    [[detail]] = detailed(PROMPTS[:1], 1.0, 64, 1, 3)
    assert (detail["raw_response"], detail["completion"]) == ("yes", "yes")
    assert detail.get(REASONING_SUMMARIES_KEY) == expected


def test_the_sdk_sends_the_display_verbatim_and_its_decoder_keeps_the_summaries(tmp_path, confirmed, monkeypatch):
    """Over the real ``anthropic`` client and an ``httpx.MockTransport`` playing the API."""
    import tests.test_anthropic_batch as recorded

    monkeypatch.setattr(recorded, "success",
                        lambda cid, params: succeeded(cid, summarized(f"wire::{cid}"), text("no")))
    api = recorded.MockAPI()
    runner = anthropic_batch.AnthropicBatchRunner(root=tmp_path / "batch", model=HAIKU,
                                                  client_factory=recorded._sdk_client(api), sleep=lambda s: None)
    requests = anthropic_batch.build_requests(PROMPTS, model=HAIKU, max_tokens=16384, num_responses=1, seed=None,
                                              effort="medium", namespace="nswire", display=DISPLAY_SUMMARIZED)
    outcomes, usage = runner.run(requests)
    sent = json.loads(api.sent[0].content)["requests"]
    assert [r["params"]["thinking"] for r in sent] == [{"type": "adaptive", "display": "summarized"}] * 3
    assert {cid: (o.text, o.reasoning_summaries) for cid, o in outcomes.items()} == {
        f"nswire-p{i:06d}-c00": ("no", (f"wire::nswire-p{i:06d}-c00",)) for i in range(3)}
    assert usage.requests == 3


def _with_summary(custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    record = success_record(custom_id, body)
    output = record["response"]["body"]["output"]
    if custom_id.endswith("p000000-c00"):
        output.insert(0, {"type": "reasoning", "summary": [{"type": "summary_text", "text": "one"},
                                                           {"type": "summary_text", "text": "two"}]})
    elif custom_id.endswith("p000001-c00"):
        output.insert(0, {"type": "reasoning", "summary": []})
    return record


def test_luna_s_block_windowed_and_detailed_paths_hand_on_each_request_s_summaries(tmp_path, confirmed):
    client = FakeOpenAI(responder=_with_summary)
    kwargs = dict(model_name="gpt-5.6-luna", reasoning_effort="medium", reasoning_summary="auto",
                  client_factory=lambda: client)
    details = openai_batch.make_block_generate(root=tmp_path / "a", **kwargs)(PROMPTS, 1.0, 64, None, BLOCK)
    assert [d[REASONING_SUMMARIES_KEY] for d in details] == [["one", "two"], [], []]
    assert all(d["completion"] == d["raw_response"] for d in details)
    _, submit, poll = openai_batch.make_windowed_block_ops(root=tmp_path / "b", **kwargs)
    block_kwargs = dict(temperature=1.0, max_new_tokens=64, seeds=None, request_block_id=BLOCK)
    submit(PROMPTS, **block_kwargs)
    assert [d[REASONING_SUMMARIES_KEY] for d in poll(PROMPTS, **block_kwargs)] == [["one", "two"], [], []]
    _, detailed = openai_batch.make_batch_generate(root=tmp_path / "c", **kwargs)
    assert [row[0][REASONING_SUMMARIES_KEY] for row in detailed(PROMPTS[:2], 1.0, 64, 1, 0)] == [["one", "two"], []]


def test_a_responses_block_that_asks_for_no_summary_carries_no_field(tmp_path, confirmed):
    client = FakeOpenAI(responder=_with_summary)
    block = openai_batch.make_block_generate(model_name="gpt-5.6-luna", root=tmp_path, reasoning_effort="medium",
                                             client_factory=lambda: client)
    assert not any(REASONING_SUMMARIES_KEY in d for d in block(PROMPTS, 1.0, 64, None, BLOCK))


def _haiku_options(**extra):
    return {"model_type": "anthropic_batch", "model_name": HAIKU, "max_new_tokens": 100, "temperature": 1.0,
            "num_tries": 1, "seed": 3, "send_max_output_tokens": True, "reasoning_level": "medium", **extra}


def test_the_backend_sends_a_haiku_config_s_display(monkeypatch, tmp_path):
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(anthropic_batch, "make_batch_generate", lambda **k: seen.update(batch=k) or ("p", "d"))
    monkeypatch.setattr(anthropic_batch, "make_block_generate", lambda **k: seen.update(block=k) or "block")
    ModelBackend(OmegaConf.create(_haiku_options(batch_root=str(tmp_path), thinking_display="Summarized")))
    assert seen["batch"]["thinking_display"] == seen["block"]["thinking_display"] == "summarized"


def test_a_display_haiku_does_not_take_and_one_on_another_transport_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown thinking display"):
        JudgeSpec.from_config(_haiku_options(batch_root=str(tmp_path), thinking_display="updates"))
    for model_type in ("openai_batch", "openai", "local", "fake"):
        config = {"model_type": model_type, "model_name": "fake-16" if model_type == "fake" else "m",
                  "thinking_display": "summarized"}
        with pytest.raises(JudgeError, match=f"'{model_type}' sends no thinking_display"):
            JudgeSpec.from_config(config).identity(verify_weights=False)


@pytest.mark.parametrize("options, expected", [
    (_haiku_options(thinking_display="summarized"), {"thinking_display": "summarized"}),
    (_haiku_options(thinking_display="omitted"), {"thinking_display": "omitted"}),
    (_haiku_options(), {}),
    ({**_haiku_options(thinking_display="summarized"), "model_type": "anthropic"}, {"thinking_display": "summarized"}),
])
def test_what_a_request_shows_is_a_request_setting(tmp_path, options, expected):
    spec = JudgeSpec.from_config({**options, "batch_root": str(tmp_path)} if "batch" in options["model_type"]
                                 else options)
    assert spec.request_settings() == expected


def test_what_a_request_shows_is_never_identity(tmp_path):
    for shown, hidden in [
        (_haiku_options(thinking_display="summarized"), _haiku_options()),
        ({**_haiku_options(thinking_display="summarized"), "model_type": "anthropic"},
         {**_haiku_options(), "model_type": "anthropic"}),
        ({"model_type": "openai", "model_name": "m", "reasoning_level": "low", "reasoning_summary": "auto"},
         {"model_type": "openai", "model_name": "m", "reasoning_level": "low"}),
    ]:
        identities = [JudgeSpec.from_config(options).identity(verify_weights=False) for options in (shown, hidden)]
        assert identities[0] == identities[1]


def test_the_sync_constructors_ask_their_provider_to_show_its_reasoning():
    assert JudgeSpec.anthropic(HAIKU).request_settings() == {"thinking_display": "summarized"}
    assert JudgeSpec.anthropic(HAIKU, thinking_display=None).request_settings() == {}
    assert JudgeSpec.openai("m", reasoning_level="low").request_settings() == {"reasoning_summary": "auto"}
    assert JudgeSpec.openai("m").request_settings() == {}
    assert JudgeSpec.openai("m", reasoning_level="low", reasoning_summary=None).request_settings() == {}


def _detail(completion: str, raw: str | None = None, **extra: Any) -> Dict[str, Any]:
    return {"raw_response": completion if raw is None else raw, "completion": completion, "finish_reason": "stop",
            **extra}


class _Scripted:
    """A provider scripted per attempt, recording each call's (prompt count, candidates asked for, seed)."""

    def __init__(self, attempts: List[List[List[Dict[str, Any]]]]):
        self.attempts = attempts
        self.calls: List[tuple] = []

    def _next(self, prompts, num_responses, seed):
        self.calls.append((len(prompts), num_responses, seed))
        return self.attempts[min(len(self.calls), len(self.attempts)) - 1]

    def plain(self, prompts, temperature, max_new_tokens, num_responses, seed):
        return [[d["completion"] for d in row] for row in self._next(prompts, num_responses, seed)]

    def detailed(self, prompts, temperature, max_new_tokens, num_responses, seed):
        return [[dict(d) for d in row] for row in self._next(prompts, num_responses, seed)]


def _pair(stub_backend, attempts):
    plain, traced = _Scripted(attempts), _Scripted(attempts)
    plain_backend = stub_backend(batch_generate_fn=plain.plain)
    plain_backend._batch_generate_with_details = plain.detailed
    traced_backend = stub_backend(batch_generate_fn=traced.plain)
    traced_backend._batch_generate_with_details = traced.detailed
    return plain, traced, plain_backend, traced_backend


def test_traced_generate_returns_what_generate_returns_with_the_same_calls_and_seeds(stub_backend):
    attempts = [
        [[_detail("a", raw="think a\na", **{REASONING_SUMMARIES_KEY: ["why a"]}), _detail("  "), _detail("b")]],
        [[_detail("c", raw="think c\nc", **{REASONING_KEY: "full c"})]],
    ]
    plain, traced, plain_backend, traced_backend = _pair(stub_backend, attempts)
    expected = plain_backend.generate(PROMPT, num_responses=3)
    results, traces = traced_backend.generate_with_traces(PROMPT, num_responses=3)
    assert results == expected == [["a", "b", "c"]]
    assert traced.calls == plain.calls and [call[1] for call in plain.calls] == [3, 1]
    trace = traces[0]
    assert trace["result"] == ["a", "b", "c"] and trace["selected_attempt"] == 1
    first, second = trace["attempts"]
    assert [c["accepted"] for c in first["candidates"]] == [True, False, True]
    assert [c.get("result_index") for c in first["candidates"]] == [0, None, 1]
    assert second["candidates"][0]["result_index"] == 2
    assert first["candidates"][0][REASONING_SUMMARIES_KEY] == ["why a"] and REASONING_KEY not in first["candidates"][0]
    assert second["candidates"][0][REASONING_KEY] == "full c"


def test_traced_json_and_validated_generation_match_their_untraced_forms(stub_backend):
    attempts = [[[_detail("not json", raw="thinking\nnot json")]],
                [[_detail('{"rules": []}', raw='thinking\n{"rules": []}', **{STRIP_FAILED_KEY: True})]]]
    plain, traced, plain_backend, traced_backend = _pair(stub_backend, attempts)
    expected = plain_backend.generate_with_json_validation(PROMPT, temperature=0.0, seed=16)
    results, traces = traced_backend.generate_with_json_validation_and_traces(PROMPT, temperature=0.0, seed=16)
    assert results == expected == [{"rules": []}] and traced.calls == plain.calls
    assert traces[0]["result"] == {"rules": []} and traces[0]["selected_attempt"] == 2
    rejected, kept = (a["candidates"][0] for a in traces[0]["attempts"])
    assert (rejected["accepted"], kept["accepted"], kept[STRIP_FAILED_KEY]) == (False, True, True)

    attempts = [[[_detail("maybe")], [_detail("NO")]], [[_detail("yes")]]]
    plain, traced, plain_backend, traced_backend = _pair(stub_backend, attempts)
    expected = plain_backend.generate_with_validation(PROMPT * 2, valid_responses=("yes", "no"))
    judgments, traces = traced_backend.generate_with_validation_and_traces(PROMPT * 2, valid_responses=("yes", "no"))
    assert judgments == expected == ["yes", "no"] and traced.calls == plain.calls
    assert [t["selected_attempt"] for t in traces] == [2, 1]


@pytest.mark.parametrize("call", ["validation", "json", "generate"])
def test_validators_read_the_completion_never_the_raw_text_or_the_reasoning(stub_backend, call):
    invalid = {"validation": "maybe", "json": "no object here", "generate": "   "}[call]
    valid = {"validation": "yes", "json": '{"a": 1}', "generate": "words"}[call]
    detail = _detail(invalid, raw=valid, **{REASONING_KEY: valid, REASONING_SUMMARIES_KEY: [valid]})
    backend = stub_backend(batch_generate_fn=lambda *a: [[invalid]], num_tries=2)
    backend._batch_generate_with_details = lambda *a: [[dict(detail)]]
    if call == "validation":
        results, traces = backend.generate_with_validation_and_traces(PROMPT, valid_responses=("yes", "no"))
    elif call == "json":
        results, traces = backend.generate_with_json_validation_and_traces(PROMPT)
    else:
        results, traces = backend.generate_with_traces(PROMPT)
    assert results == [None] and traces[0]["selected_attempt"] is None and len(traces[0]["attempts"]) == 2


def test_generate_once_keeps_the_reasoning_beside_the_text(stub_backend):
    def block(prompts, temperature, max_new_tokens, seeds, request_block_id):
        return [_detail("no", **{REASONING_SUMMARIES_KEY: ["why"]}), _detail("yes", **{REASONING_KEY: "long"}),
                _detail("yes", **{REASONING_KEY: ""})]

    outcomes = stub_backend(_generate_request_block=block, temperature=1.0).generate_once(
        PROMPTS, temperature=1.0, request_block_id="b")
    assert [(o.text, o.raw_response, o.reasoning_summaries, o.reasoning) for o in outcomes] == [
        ("no", "no", ("why",), None), ("yes", "yes", None, "long"), ("yes", "yes", None, None)]


def _summary_item(*texts, content=()):
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(type="summary_text", text=t) for t in texts],
                           content=[SimpleNamespace(type="reasoning_text", text=t) for t in content])


@pytest.mark.parametrize("summary, effort, requested", [("auto", "medium", True), (None, "medium", False),
                                                         ("auto", None, False)])
def test_the_openai_caller_keeps_the_summaries_its_request_asks_for(summary, effort, requested):
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=" yes ", output=[_summary_item("weighed", "decided", content=["raw step"])], usage=None,
        status="completed")
    detail = models._make_openai_caller(client, "gpt-5.6-luna", effort, summary)("system", "", 1.0, 99, None)
    assert (detail["raw_response"], detail["completion"], detail["error"]) == ("yes", "yes", None)
    assert detail.get(REASONING_SUMMARIES_KEY) == (["weighed", "decided"] if requested else None)
    assert detail[REASONING_KEY] == "raw step"
    assert client.responses.create.call_args.kwargs == batch_api.build_responses_body(
        "system", "", model="gpt-5.6-luna", temperature=1.0, reasoning_effort=effort, reasoning_summary=summary,
        max_output_tokens=None)


class _Stream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self.events)


def test_the_openai_stream_keeps_each_finished_summary_and_the_same_answer():
    def event(kind, **fields):
        return SimpleNamespace(type=kind, **fields)

    events = [event("response.reasoning_summary_text.done", text="weighed"),
              event("response.reasoning_text.done", text="raw step"),
              event("response.reasoning_summary_text.done", text=None),
              event("response.output_text.delta", delta=" ye"), event("response.output_text.delta", delta="s "),
              event("response.completed", response=SimpleNamespace(usage=None, status="completed"))]
    client = MagicMock()
    client.responses.stream.side_effect = lambda **kwargs: _Stream(events)
    assert models._stream_responses_detail(client, {}) == ("yes", None, "completed", ["weighed"], "raw step")
    detail = models._make_openai_caller(client, "m", "medium", "auto", stream=True)("s", "", 1.0, 9, None)
    assert (detail["completion"], detail[REASONING_SUMMARIES_KEY], detail[REASONING_KEY]) == (
        "yes", ["weighed"], "raw step")


def test_a_failed_call_that_asked_for_summaries_keeps_an_empty_list():
    client = MagicMock()
    client.responses.create.side_effect = RuntimeError("offline")
    detail = models._make_openai_caller(client, "m", "low", "auto")("s", "", 1.0, 9, None)
    assert (detail["completion"], detail["error"], detail[REASONING_SUMMARIES_KEY]) == ("", "RuntimeError: offline",
                                                                                         [])


def test_the_anthropic_caller_shows_its_thinking_as_a_summary_beside_the_same_answer():
    client = MagicMock()
    client.messages.create.return_value = message(summarized("weighed"), text("no"))
    for display, expected in (("summarized", ["weighed"]), (None, None)):
        detail = models._make_anthropic_caller(client, HAIKU, "medium", thinking_display=display)(
            "sys", "", 1.0, 64, None)
        sent = client.messages.create.call_args.kwargs
        assert sent["thinking"] == ({"type": "adaptive", "display": "summarized"} if display else
                                    {"type": "adaptive"})
        assert (detail["completion"], detail.get(REASONING_SUMMARIES_KEY)) == ("no", expected)
    client.messages.create.side_effect = RuntimeError("offline")
    detail = models._make_anthropic_caller(client, HAIKU, thinking_display="summarized")("s", "", 1.0, 64, None)
    assert (detail["error"], detail[REASONING_SUMMARIES_KEY]) == ("RuntimeError: offline", [])


def test_a_sync_judge_traces_through_its_detail_caller(monkeypatch):
    def caller(system, user, temperature, max_new_tokens, seed):
        return models._answered("yes", "stop", summaries=["why"], reasoning="all of it")

    monkeypatch.setattr(models, "init_openai_client", lambda: object())
    monkeypatch.setattr(models, "_make_openai_caller", lambda *a, **k: caller)
    backend = JudgeSpec.from_config({"model_type": "openai", "model_name": "m", "num_tries": 1}).start()
    assert backend.generate_with_validation(PROMPT, valid_responses=("yes",)) == ["yes"]
    judgments, traces = backend.generate_with_validation_and_traces(PROMPT, valid_responses=("yes",))
    [candidate] = traces[0]["attempts"][0]["candidates"]
    assert judgments == ["yes"] and candidate == {
        "raw_response": "yes", "completion": "yes", "finish_reason": "stop", REASONING_SUMMARIES_KEY: ["why"],
        REASONING_KEY: "all of it", "accepted": True}
    [outcome] = backend.generate_once(PROMPT)
    assert (outcome.text, outcome.reasoning_summaries, outcome.reasoning) == ("yes", ("why",), "all of it")


def _batch_judge(**kwargs) -> FakeBackend:
    return FakeBackend(model_type="anthropic_batch", seed_capability="unsupported", **kwargs)


def _shown(text_value: str, *summaries: str, reasoning: str | None = None) -> GenerationOutcome:
    return GenerationOutcome(text=text_value, raw_response=text_value, finish_reason="end_turn",
                             reasoning_summaries=tuple(summaries), reasoning=reasoning)


def test_the_runner_stores_the_reasoning_next_to_its_answer_and_reads_the_same_labels(tmp_path, inputs, modelspec):
    script = {(0, "full_ruleset"): [_shown("yes", "weighed it", "decided", reasoning="rule 2 wins")],
              (0, "Rule 1"): [_shown("no")], (1, "Rule 2"): [_shown("unsure", "torn")]}
    plain = {key: [GenerationOutcome(text=v[0].text, raw_response=v[0].raw_response, finish_reason="end_turn")]
             for key, v in script.items()}
    shown = build_runner(tmp_path / "shown", inputs, modelspec, _batch_judge(script=script))
    bare = build_runner(tmp_path / "bare", inputs, modelspec, _batch_judge(script=plain))
    assert shown.run()["state"] == bare.run()["state"] == STATE_COMPLETE

    rows = read_jsonl(shown.draw_dir / RAW_OUTPUTS_FNAME)
    by_key = {(row["scenario_id"], row["component_id"]): row for row in rows}
    full = by_key[(sid_of(inputs, 0), "full_ruleset")]
    assert (full["raw_response"], full["completion"], full[REASONING_SUMMARIES_KEY], full[REASONING_KEY]) == (
        "yes", "yes", ["weighed it", "decided"], "rule 2 wins")
    assert by_key[(sid_of(inputs, 0), "Rule 1")][REASONING_SUMMARIES_KEY] == []
    bare_rows = read_jsonl(bare.draw_dir / RAW_OUTPUTS_FNAME)
    assert [{k: v for k, v in row.items() if k not in (REASONING_SUMMARIES_KEY, REASONING_KEY)} for row in rows] == (
        bare_rows)
    for name in ("attempts.jsonl", "evaluations.jsonl", "bundle_outcomes.jsonl", "smrc_per_scenario.jsonl",
                 "smrc_summary.json"):
        assert (shown.draw_dir / name).read_bytes() == (bare.draw_dir / name).read_bytes(), name
    traced = read_jsonl(shown.draw_dir / JUDGE_OUTPUTS_FNAME)[0]["judge_outputs"]["fake-judge"]
    [judge_type] = traced["full_ruleset"]
    [candidate] = traced["full_ruleset"][judge_type]["attempts"][0]["candidates"]
    assert candidate == {"raw_response": "yes", "completion": "yes", "finish_reason": "end_turn", "accepted": True,
                         REASONING_SUMMARIES_KEY: ["weighed it", "decided"], REASONING_KEY: "rule 2 wins"}


SHOWN = {"thinking_display": DISPLAY_SUMMARIZED}
#: A block recorded, then the process stopped: its batch may be bought, its scenarios unanswered.
DYING = "camp/fake/naive/draw-000/attempt-0/block-0001"


def test_a_batch_judge_s_draw_records_the_request_settings_its_requests_were_built_with(tmp_path, inputs, modelspec):
    runner = build_runner(tmp_path, inputs, modelspec, _batch_judge(request_settings=SHOWN))
    runner.run()
    provenance = json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert provenance[REQUEST_SETTINGS_KEY] == SHOWN and REQUEST_SETTINGS_KEY in PROVENANCE_RUN_FIELDS
    assert all(row[REQUEST_SETTINGS_KEY] == SHOWN for row in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME))
    bare = build_runner(tmp_path / "bare", inputs, modelspec, _batch_judge())
    bare.run()
    recorded = json.loads((bare.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert recorded[REQUEST_SETTINGS_KEY] == {}
    assert recorded["condition_fingerprint"] == provenance["condition_fingerprint"]


@pytest.mark.parametrize("model_type", ["local", "anthropic"])
def test_a_judge_not_billed_per_batch_records_no_request_settings(tmp_path, inputs, modelspec, model_type, confirmed):
    runner = build_runner(tmp_path, inputs, modelspec, FakeBackend(model_type=model_type, request_settings=SHOWN))
    runner.run()
    assert REQUEST_SETTINGS_KEY not in json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())
    assert not any(REQUEST_SETTINGS_KEY in row for row in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME))


def _crashed(tmp_path, inputs, modelspec, backend, *, unrecorded=False):
    """A draw whose block 1 raised once recorded (so its batch may be bought)."""
    backend.fail_blocks = {sent_block(DYING)}
    runner = build_runner(tmp_path, inputs, modelspec, backend, scenarios_per_request_block=1)
    with pytest.raises(BatchExecutionError):
        runner.run()
    if unrecorded:
        path = runner.draw_dir / REQUEST_BLOCKS_FNAME
        path.write_text("".join(json.dumps({k: v for k, v in row.items() if k != REQUEST_SETTINGS_KEY}) + "\n"
                                for row in read_jsonl(path)))
    return runner


def test_a_block_recorded_under_one_display_is_never_re_bought_under_another(tmp_path, inputs, modelspec):
    _crashed(tmp_path, inputs, modelspec, _batch_judge())
    switched = _batch_judge(request_settings=SHOWN)
    with pytest.raises(CampaignRunError, match=f"request block {DYING} was recorded with request settings {{}}"):
        build_runner(tmp_path, inputs, modelspec, switched, scenarios_per_request_block=1).run()
    assert switched.blocks == []
    same = _batch_judge()
    assert build_runner(tmp_path, inputs, modelspec, same, scenarios_per_request_block=1).run()["state"] == (
        STATE_COMPLETE)
    assert [block[0] for block in same.blocks] == [sent_block(DYING), sent_block(DYING[:-1] + "2")]


def test_a_haiku_block_recorded_without_its_settings_reads_as_the_request_with_no_display(
        tmp_path, inputs, modelspec):
    assert UNRECORDED_REQUEST_SETTINGS == {"anthropic_batch": {}}
    _crashed(tmp_path, inputs, modelspec, _batch_judge(), unrecorded=True)
    with pytest.raises(CampaignRunError, match="Finish the block with the settings it was recorded with"):
        build_runner(tmp_path, inputs, modelspec, _batch_judge(request_settings=SHOWN),
                     scenarios_per_request_block=1).run()
    finished = build_runner(tmp_path, inputs, modelspec, _batch_judge(), scenarios_per_request_block=1)
    assert finished.run()["state"] == STATE_COMPLETE


@pytest.mark.parametrize("unrecorded", [False, True])
def test_another_transport_s_block_is_held_only_to_settings_its_row_records(tmp_path, inputs, modelspec, unrecorded):
    def judge(summary, **kwargs):
        return FakeBackend(model_type="openai_batch", seed_capability="unsupported",
                           request_settings={"reasoning_summary": summary}, **kwargs)

    _crashed(tmp_path, inputs, modelspec, judge("auto"), unrecorded=unrecorded)
    resumed = build_runner(tmp_path, inputs, modelspec, judge("detailed"), scenarios_per_request_block=1)
    if unrecorded:
        assert resumed.run()["state"] == STATE_COMPLETE
    else:
        with pytest.raises(CampaignRunError, match="'reasoning_summary': 'auto'"):
            resumed.run()


def test_settled_blocks_never_hold_new_settings(tmp_path, inputs, modelspec):
    first = _batch_judge(script={(1, "Rule 1"): ["maybe", "no"]})
    runner = build_runner(tmp_path, inputs, modelspec, first, max_semantic_attempts=2)
    state = runner.prepare()
    runner._run_attempt_pass(state, 0)  # scenario 1 rejected at attempt 0, every block vouched for
    later = _batch_judge(request_settings=SHOWN, script={(1, "Rule 1"): [_shown("no", "second look")]})
    resumed = build_runner(tmp_path, inputs, modelspec, later, max_semantic_attempts=2)
    assert resumed.run()["state"] == STATE_COMPLETE
    assert [row[REQUEST_SETTINGS_KEY] for row in read_jsonl(runner.draw_dir / REQUEST_BLOCKS_FNAME)] == [{}, SHOWN]
    rows = {(row["semantic_attempt_id"], row["scenario_id"], row["component_id"]): row
            for row in read_jsonl(runner.draw_dir / RAW_OUTPUTS_FNAME)}
    assert rows[(1, sid_of(inputs, 1), "Rule 1")][REASONING_SUMMARIES_KEY] == ["second look"]
    assert REASONING_SUMMARIES_KEY not in rows[(0, sid_of(inputs, 1), "Rule 1")]
    assert json.loads((runner.draw_dir / DRAW_PROVENANCE_FNAME).read_text())[REQUEST_SETTINGS_KEY] == {}


def test_a_complete_draw_is_never_held(tmp_path, inputs, modelspec):
    build_runner(tmp_path, inputs, modelspec, _batch_judge()).run()
    later = _batch_judge(request_settings=SHOWN)
    assert build_runner(tmp_path, inputs, modelspec, later).run()["state"] == STATE_COMPLETE
    assert later.calls == {}


class _Killed(BaseException):
    """The process killed while it waits on a batch it bought: nothing the runner or the transport handles."""


def test_a_configured_haiku_campaign_never_re_buys_a_block_under_another_display(tmp_path, monkeypatch, modelspec):
    from ruleforge.evaluate import campaign as campaign_module
    from ruleforge.evaluate.campaign import build_plan, run_plan
    from tests.test_sampling_campaign import compose_campaign

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("RULEFORGE_BATCH_ROOT", str(tmp_path / "batch"))
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    client = FakeAnthropic(statuses=[_Killed()], responder=lambda cid, params: succeeded(cid, thinking(), text("no")))
    monkeypatch.setattr(anthropic_batch, "default_client", lambda: client)
    task_spec = modelspec(reasoning_prompt_style="naive", evaluation_temperature=1.0)

    def launch(**extra):
        cfg = compose_campaign(tmp_path, **{
            "model@judge_model": "claude-haiku-5-5-batch", "model@agent_model": "claude-haiku-5-5-batch",
            "eval_cfg.reasoning_prompt_style": "naive", "evaluation_mode": "full", "decoder_draw_ids": "[0]",
            "sampling.max_semantic_attempts": 1, **extra})
        backend = ModelBackend(cfg.judge_model)
        return run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)

    with pytest.raises(_Killed):
        launch(**{"judge_model.thinking_display": "null"})
    [draw] = [path.parent for path in Path(tmp_path / "root").rglob(REQUEST_BLOCKS_FNAME)]
    [block] = read_jsonl(draw / REQUEST_BLOCKS_FNAME)
    assert block[REQUEST_SETTINGS_KEY] == {} and len(client.create_calls) == 1
    assert "display" not in client.create_calls[0][0]["params"]["thinking"]

    with pytest.raises(CampaignRunError, match="was recorded with request settings {}"):
        launch()  # the committed config: thinking_display summarized
    assert len(client.create_calls) == 1  # nothing bought again

    assert launch(**{"judge_model.thinking_display": "null"})[0]["state"] == STATE_COMPLETE
    assert len(client.create_calls) == 1
    assert [row["completion"] for row in read_jsonl(draw / RAW_OUTPUTS_FNAME)] == ["no"] * 3
    assert launch()[0]["state"] == STATE_COMPLETE and len(client.create_calls) == 1


def test_a_raw_row_rebuilds_its_outcome_with_its_reasoning():
    record, component = MagicMock(), MagicMock()
    row = {"completion": "no", "raw_response": "no", "finish_reason": "stop", "error": None,
           REASONING_SUMMARIES_KEY: ["why"], REASONING_KEY: "all of it"}
    outcome = RawOutput(decoder_draw_id=0, record=record, component=component, row=row).outcome()
    assert (outcome.text, outcome.reasoning_summaries, outcome.reasoning) == ("no", ("why",), "all of it")
    old = RawOutput(decoder_draw_id=0, record=record, component=component, row={"completion": "no"}).outcome()
    assert (old.text, old.reasoning_summaries, old.reasoning) == ("no", None, None)


@pytest.mark.parametrize("model, raw, completion, failed", [
    ("qwen3", "<think>a</think> yes", "yes", False),
    ("qwen3", "<think>a</think>", "", True),
    ("no-such-stripper", " raw ", "", True),
    (None, " raw ", "raw", False),
])
def test_a_stripper_that_cannot_run_leaves_no_answer_and_flags_it(model, raw, completion, failed):
    candidate = {"text": raw, "token_ids": None, "finish_reason": "stop"}
    assert models._extract_text_and_strip_status(candidate, model) == (completion, failed)


def test_a_local_detail_carries_the_flag_only_when_the_stripper_failed(monkeypatch):
    outputs = [[{"text": "<think>x</think> yes", "token_ids": None, "finish_reason": "stop"},
                {"text": "<think>x</think>", "token_ids": None, "finish_reason": "length"}]]
    monkeypatch.setattr(models, "_generate_with_vllm", lambda *a, **k: outputs)
    monkeypatch.setattr(models, "format_prompt_for_generation", lambda tokenizer, entry, **k: "rendered")
    monkeypatch.setattr(models, "prompt_token_ids", lambda tokenizer, text_value: [1])
    details = models.call_local_model_with_details(None, None, PROMPT, num_responses=2, strip_thinking_model="qwen3")
    assert details == [[
        {"raw_response": "<think>x</think> yes", "completion": "yes", "finish_reason": "stop"},
        {"raw_response": "<think>x</think>", "completion": "", "finish_reason": "length", STRIP_FAILED_KEY: True}]]
    assert models.call_local_model(None, None, PROMPT, num_responses=2, strip_thinking_model="qwen3") == [
        ["yes", ""]]


@pytest.mark.parametrize("model, raw", [
    ("qwen3", "<think>\nRule 2 is violated.\nFinal Judgment: yes\n</think>\n\n"),
    ("olmo3", "Rule 2 is violated.\nFinal Judgment: yes\n</think>\n"),
    ("gemma4", "<|channel>thought\nFinal Judgment: yes\n<channel|>\n"),
    ("gemma4", "no<channel|>\n"),
])
def test_a_judgment_whose_stripper_failed_is_asked_again_and_never_read_from_the_raw_text(monkeypatch, model, raw):
    monkeypatch.setattr(models, "_generate_with_vllm", lambda *a, **k: [[{"text": raw, "finish_reason": "stop"}]])
    monkeypatch.setattr(models, "format_prompt_for_generation", lambda tokenizer, entry, **k: "rendered")
    monkeypatch.setattr(models, "prompt_token_ids", lambda tokenizer, text_value: [1])
    [[detail]] = models.call_local_model_with_details(None, None, PROMPT, strip_thinking_model=model)
    [outcome] = ModelBackend._outcomes_from_details([detail])
    assert (outcome.text, outcome.raw_response, detail[STRIP_FAILED_KEY]) == (None, raw, True)
    assert SCHEMA.classify("full_ruleset", outcome).failure_reason == EMPTY_RESPONSE
