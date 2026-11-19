from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from ruleforge.judges import JudgeError, JudgeSpec, batch_api, models
from ruleforge.judges.backend import (
    SEED_CAPABILITIES,
    SEED_LOCAL_DETERMINISTIC,
    SEED_PROVIDER_BEST_EFFORT,
    SEED_UNSUPPORTED,
    GenerationOutcome,
    ModelBackend,
)
from ruleforge.judges.batch_api import (
    CHAT_COMPLETIONS_ENDPOINT,
    RESPONSES_ENDPOINT,
    BatchExecutionError,
    BatchOutcome,
    BatchSpendNotConfirmed,
)
from ruleforge.core.digests import request_namespace
from tests.test_openai_batch import FakeOpenAI, success_record

PROMPTS = [{"system_prompt": f"prompt {i}", "user_prompt": ""} for i in range(3)]


@pytest.fixture
def confirmed(monkeypatch):
    monkeypatch.setattr("ruleforge.judges.batch_api.spend_is_confirmed", lambda: True)


def test_generate_once_prefers_the_request_block_callable(stub_backend):
    seen = {}

    def block(prompts, temperature, max_new_tokens, seeds, request_block_id):
        seen.update(
            prompts=prompts, temperature=temperature, max_new_tokens=max_new_tokens,
            seeds=seeds, request_block_id=request_block_id,
        )
        return [
            {
                "raw_response": f"raw {i}", "completion": f"text {i}",
                "finish_reason": "stop", "error": None,
                "provider_request_ref": f"ref-{i}", "request_block_ref": "/work/dir",
            }
            for i in range(len(prompts))
        ]

    backend = stub_backend(
        batch_generate_fn=lambda *a: pytest.fail("block path must be used"),
        _generate_request_block=block,
        temperature=1.0,
    )
    outcomes = backend.generate_once(
        PROMPTS, temperature=1.0, seeds=[1, None, 3], request_block_id="c/k/draw-000/attempt-0/block-0000"
    )
    assert seen["seeds"] == [1, None, 3]
    assert seen["request_block_id"] == "c/k/draw-000/attempt-0/block-0000"
    assert seen["max_new_tokens"] == 1024
    assert outcomes[1] == GenerationOutcome(
        text="text 1", raw_response="raw 1", finish_reason="stop", error=None,
        provider_request_ref="ref-1", request_block_ref="/work/dir",
    )


def test_generate_once_uses_the_detailed_local_callable_with_per_prompt_seeds(
    stub_backend,
):
    seen = {}

    def details(prompts, temperature, max_new_tokens, num_responses, seed):
        seen.update(seed=seed, num_responses=num_responses, temperature=temperature)
        return [
            [{"raw_response": "<think>x</think>yes", "completion": "yes", "finish_reason": "stop"}],
            [],
            [{"raw_response": "", "completion": "", "finish_reason": "length"}],
        ]

    backend = stub_backend(
        batch_generate_fn=lambda *a: pytest.fail("detailed path must be used"),
        _batch_generate_with_details=details,
        fixed_temperature=1.0,
    )
    outcomes = backend.generate_once(PROMPTS, temperature=0.1, seeds=[7, 8, 9])
    assert seen == {"seed": [7, 8, 9], "num_responses": 1, "temperature": 1.0}
    assert outcomes[0].text == "yes" and outcomes[0].raw_response == "<think>x</think>yes"
    assert outcomes[1] == GenerationOutcome(text=None, raw_response=None, finish_reason=None)
    assert outcomes[1].returned is False
    assert outcomes[2].text is None and outcomes[2].raw_response == ""
    assert outcomes[2].finish_reason == "length" and outcomes[2].returned is True


def test_generate_once_falls_back_to_plain_text_generation(stub_backend):
    calls = []

    def plain(prompts, temperature, max_new_tokens, num_responses, seed):
        calls.append(seed)
        return [["a"], [], [""]]

    backend = stub_backend(batch_generate_fn=plain)
    outcomes = backend.generate_once(PROMPTS, temperature=1.0)
    assert calls == [None]
    assert [o.text for o in outcomes] == ["a", None, None]
    assert outcomes[0].raw_response == "a"
    assert outcomes[0].error is None and outcomes[0].finish_reason is None
    assert [o.raw_response for o in outcomes] == ["a", None, ""]
    assert [o.returned for o in outcomes] == [True, False, True]
    assert not any(o.block_failed for o in outcomes)


def test_generate_once_refuses_misaligned_seeds(stub_backend):
    backend = stub_backend(batch_generate_fn=lambda *a: [["a"]] * 3)
    with pytest.raises(ValueError, match="seeds must align"):
        backend.generate_once(PROMPTS, seeds=[1, 2])


def test_generate_once_reports_a_transport_failure_without_raising(stub_backend, caplog):
    def broken(*_args):
        raise ConnectionError("socket closed")

    backend = stub_backend(batch_generate_fn=broken)
    with caplog.at_level("WARNING", logger="ruleforge.judges.backend"):
        outcomes = backend.generate_once(PROMPTS, request_block_id="blk")
    assert [o.error for o in outcomes] == ["ConnectionError: socket closed"] * 3
    assert all(o.text is None and o.block_failed and not o.returned for o in outcomes)
    assert "socket closed" in caplog.text and "blk" not in caplog.text


@pytest.mark.parametrize("error", [BatchSpendNotConfirmed("no"), BatchExecutionError("dead")])
def test_generate_once_lets_a_fatal_provider_error_propagate(stub_backend, error):
    def fatal(*_args):
        raise error

    backend = stub_backend(batch_generate_fn=fatal)
    with pytest.raises(type(error)):
        backend.generate_once(PROMPTS)


def test_generate_once_refuses_a_short_result_list(stub_backend):
    backend = stub_backend(batch_generate_fn=lambda *a: [["a"]])
    with pytest.raises(RuntimeError, match="returned 1 results for 3 prompts"):
        backend.generate_once(PROMPTS)


def test_generate_once_rejects_malformed_prompts(stub_backend):
    backend = stub_backend(batch_generate_fn=lambda *a: [])
    with pytest.raises(ValueError, match="nonblank"):
        backend.generate_once([{"system_prompt": "", "user_prompt": ""}])


def test_resolved_temperature_lets_a_pinned_provider_win(stub_backend):
    assert stub_backend(temperature=0.3).resolved_temperature(None) == 0.3
    assert stub_backend(temperature=0.3).resolved_temperature(1.0) == 1.0
    assert stub_backend(fixed_temperature=1.0).resolved_temperature(0.1) == 1.0


def _config(model_type, **extra):
    return OmegaConf.create(
        {
            "model_type": model_type,
            "model_name": "test-model",
            "max_new_tokens": 100,
            "temperature": 1.0,
            "num_tries": 1,
            "seed": 3,
            **extra,
        }
    )


def test_a_local_backend_is_locally_deterministic(monkeypatch):
    monkeypatch.setattr(models, "init_local_model", lambda **kwargs: (object(), object()))
    monkeypatch.setattr(models, "engine_argument_names", frozenset)  # no vLLM here
    monkeypatch.setattr("ruleforge.judges.providers.require", lambda purpose, *modules: None)  # vLLM stubbed here
    backend = ModelBackend(
        _config(
            "local", model_path="/m", tp_size=1, reasoning_parser=None,
            speculative_method=None, num_speculative_tokens=None, max_model_len=None,
            attention_backend=None, enable_thinking=False,
        )
    )
    assert backend.seed_capability == SEED_LOCAL_DETERMINISTIC
    assert backend._generate_request_block is None


@pytest.mark.parametrize(
    "model_type, init_name, caller_name, expected",
    [
        ("openai", "init_openai_client", "_make_openai_caller", SEED_UNSUPPORTED),
    ],
)
def test_sync_providers_declare_whether_their_wire_shape_carries_a_seed(
    monkeypatch, model_type, init_name, caller_name, expected
):
    monkeypatch.setattr(models, init_name, lambda: object())
    monkeypatch.setattr(models, caller_name, lambda *a, **k: (lambda *args: {"completion": "ok"}))
    backend = ModelBackend(_config(model_type))
    assert backend.seed_capability == expected
    assert backend.seed_capability in SEED_CAPABILITIES


@pytest.mark.parametrize("model_type", ["openai_batch"])
@pytest.mark.parametrize(
    "endpoint, expected",
    [("responses", SEED_UNSUPPORTED), ("chat_completions", SEED_PROVIDER_BEST_EFFORT)],
)
def test_batch_providers_expose_a_block_callable_and_declare_the_endpoints_seed(
    monkeypatch, tmp_path, model_type, endpoint, expected
):
    import ruleforge.judges.openai_batch as openai_batch

    seen = {}

    def fake_block(**kwargs):
        seen.update(kwargs)
        return "block"

    monkeypatch.setattr(openai_batch, "make_batch_generate", lambda **k: ("plain", "detailed"))
    monkeypatch.setattr(openai_batch, "make_block_generate", fake_block)

    backend = ModelBackend(
        _config(model_type, batch_root=str(tmp_path), batch_endpoint=endpoint)
    )
    assert backend.seed_capability == expected
    assert backend._generate_request_block == "block"
    assert seen["endpoint"] == batch_api.resolve_endpoint(endpoint)
    assert seen["root"] == tmp_path


def test_estimate_block_tokens_delegates_to_the_windowed_op(stub_backend):
    seen = {}

    def estimate(prompts, *, temperature, max_new_tokens, seeds, request_block_id):
        seen.update(temperature=temperature, seeds=seeds, request_block_id=request_block_id)
        return 4242

    backend = stub_backend(
        max_enqueued_tokens=1000, _estimate_request_block=estimate,
        temperature=1.0, fixed_temperature=1.0,
    )
    tokens = backend.estimate_block_tokens(PROMPTS, temperature=1.0, seeds=None, request_block_id="b")
    assert tokens == 4242
    assert seen["temperature"] == 1.0 and seen["seeds"] is None and seen["request_block_id"] == "b"


def test_submit_block_delegates_to_the_windowed_op(stub_backend):
    calls = []
    backend = stub_backend(
        max_enqueued_tokens=1000,
        _submit_request_block=lambda prompts, **kwargs: calls.append(kwargs),
        temperature=1.0, fixed_temperature=1.0,
    )
    backend.submit_block(PROMPTS, temperature=1.0, seeds=None, request_block_id="b")
    assert calls and calls[0]["request_block_id"] == "b"


def test_poll_block_returns_none_until_done_then_converts_to_outcomes(stub_backend):
    state = {"done": False}

    def poll(prompts, *, temperature, max_new_tokens, seeds, request_block_id):
        if not state["done"]:
            return None
        return [
            {
                "raw_response": f"raw {i}", "completion": f"text {i}",
                "finish_reason": "stop", "error": None,
                "provider_request_ref": f"ref-{i}", "request_block_ref": "/wd",
            }
            for i in range(len(prompts))
        ]

    backend = stub_backend(
        max_enqueued_tokens=1000, _poll_request_block=poll,
        temperature=1.0, fixed_temperature=1.0,
    )
    assert backend.poll_block(PROMPTS, temperature=1.0, seeds=None, request_block_id="b") is None
    state["done"] = True
    outcomes = backend.poll_block(PROMPTS, temperature=1.0, seeds=None, request_block_id="b")
    assert [o.text for o in outcomes] == ["text 0", "text 1", "text 2"]
    assert outcomes[1].provider_request_ref == "ref-1"


def test_poll_block_refuses_a_mismatched_result_count(stub_backend):
    backend = stub_backend(
        max_enqueued_tokens=1000,
        _poll_request_block=lambda prompts, **kwargs: [{"completion": "only one"}],
        temperature=1.0, fixed_temperature=1.0,
    )
    with pytest.raises(RuntimeError, match="results for"):
        backend.poll_block(PROMPTS, temperature=1.0, seeds=None, request_block_id="b")


def test_a_budget_wires_the_windowed_ops_for_openai_batch(monkeypatch, tmp_path):
    import ruleforge.judges.openai_batch as openai_batch

    monkeypatch.setattr(openai_batch, "make_batch_generate", lambda **k: ("plain", "detailed"))
    monkeypatch.setattr(openai_batch, "make_block_generate", lambda **k: "block")
    monkeypatch.setattr(openai_batch, "make_windowed_block_ops", lambda **k: ("est", "sub", "poll"))
    backend = ModelBackend(
        _config("openai_batch", batch_root=str(tmp_path), batch_endpoint="responses",
                max_enqueued_tokens=250000)
    )
    assert backend.max_enqueued_tokens == 250000
    assert (
        backend._estimate_request_block,
        backend._submit_request_block,
        backend._poll_request_block,
    ) == ("est", "sub", "poll")


def test_a_budget_without_windowing_support_warns_and_stays_sequential(monkeypatch, tmp_path, caplog):
    import ruleforge.judges.openai_batch as openai_batch

    monkeypatch.setattr(openai_batch, "make_batch_generate", lambda **k: ("plain", "detailed"))
    monkeypatch.setattr(openai_batch, "make_block_generate", lambda **k: "block")
    monkeypatch.delattr(openai_batch, "make_windowed_block_ops", raising=False)
    with caplog.at_level("WARNING"):
        backend = ModelBackend(
            _config("openai_batch", batch_root=str(tmp_path), batch_endpoint="responses",
                    max_enqueued_tokens=250000)
        )
    assert backend.max_enqueued_tokens is None
    assert not hasattr(backend, "_submit_request_block")
    assert "no windowed" in caplog.text


def test_no_budget_leaves_a_batch_backend_on_the_sequential_path(monkeypatch, tmp_path):
    import ruleforge.judges.openai_batch as openai_batch

    monkeypatch.setattr(openai_batch, "make_batch_generate", lambda **k: ("plain", "detailed"))
    monkeypatch.setattr(openai_batch, "make_block_generate", lambda **k: "block")
    backend = ModelBackend(
        _config("openai_batch", batch_root=str(tmp_path), batch_endpoint="responses")
    )
    assert backend.max_enqueued_tokens is None
    assert not hasattr(backend, "_submit_request_block")


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params=None, **kwargs):
        self.calls.append((list(prompts), sampling_params, kwargs))
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"out {i}", token_ids=[i], finish_reason="stop")])
            for i, _ in enumerate(prompts)
        ]


@pytest.fixture
def fake_vllm(monkeypatch):
    module = SimpleNamespace(SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


def test_vllm_builds_one_sampling_params_per_prompt_for_a_seed_list(fake_vllm):
    llm = _FakeLLM()
    outputs = models._generate_with_vllm(
        llm, ["a", "b"], temperature=1.0, max_new_tokens=8, num_responses=1, seed=[5, None]
    )
    _, params, _ = llm.calls[0]
    assert [p.kwargs["seed"] for p in params] == [5, None]
    assert all(p.kwargs["temperature"] == 1.0 and p.kwargs["n"] == 1 for p in params)
    assert [o[0]["text"] for o in outputs] == ["out 0", "out 1"]


def test_vllm_keeps_the_shared_seed_path_for_a_scalar(fake_vllm):
    llm = _FakeLLM()
    models._generate_with_vllm(
        llm, ["a", "b"], temperature=0.1, max_new_tokens=8, num_responses=2, seed=9
    )
    _, params, _ = llm.calls[0]
    assert isinstance(params, _FakeSamplingParams) and params.kwargs["seed"] == 9


def test_vllm_refuses_a_misaligned_seed_list(fake_vllm):
    with pytest.raises(ValueError, match="per-prompt seeds must align"):
        models._generate_with_vllm(
            _FakeLLM(), ["a", "b"], temperature=1.0, max_new_tokens=8,
            num_responses=1, seed=[1],
        )


def test_api_generate_batch_takes_one_seed_per_prompt():
    seen = []

    def call(system, user, temperature, max_tokens, seed):
        seen.append(seed)
        return {"raw_response": "ok", "completion": "ok", "finish_reason": "stop", "error": None}

    prompts = [{"system_prompt": "a"}, {"system_prompt": "b"}, {"system_prompt": "c"}]
    models._api_generate_batch(call, prompts, 1.0, 10, num_responses=2, seed=[4, None, 6])
    assert seen == [4, 5, None, None, 6, 7]
    with pytest.raises(ValueError, match="per-prompt seeds must align"):
        models._api_generate_batch(call, prompts, 1.0, 10, num_responses=1, seed=[1])


def test_custom_id_uses_the_namespace_instead_of_the_seed():
    assert batch_api.batch_custom_id(3, 1, 16) == "s16-p000003-c01"
    assert batch_api.batch_custom_id(3, 1, 16, "nsabc") == "nsabc-p000003-c01"


def test_build_batch_requests_gives_each_prompt_its_own_chat_seed():
    prompts = [{"system_prompt": "a"}, {"system_prompt": "b"}]
    requests = batch_api.build_batch_requests(
        prompts, endpoint=CHAT_COMPLETIONS_ENDPOINT, model="m", temperature=1.0,
        num_responses=2, seed=None, seeds=[10, None], namespace="nsx",
    )
    assert [r.custom_id for r in requests] == [
        "nsx-p000000-c00", "nsx-p000000-c01", "nsx-p000001-c00", "nsx-p000001-c01",
    ]
    assert [r.body.get("seed") for r in requests] == [10, 11, None, None]


def test_build_batch_requests_never_seeds_a_responses_body_and_checks_alignment():
    prompts = [{"system_prompt": "a"}]
    requests = batch_api.build_batch_requests(
        prompts, endpoint=RESPONSES_ENDPOINT, model="m", temperature=1.0,
        num_responses=1, seed=None, seeds=[10], namespace="nsx",
    )
    assert "seed" not in requests[0].body
    with pytest.raises(ValueError, match="seeds must align"):
        batch_api.build_batch_requests(
            prompts, endpoint=RESPONSES_ENDPOINT, model="m", temperature=1.0,
            num_responses=1, seed=None, seeds=[1, 2],
        )


def test_reshape_outcomes_looks_up_namespaced_ids():
    outcomes = {
        batch_api.batch_custom_id(0, 0, None, "nsq"): BatchOutcome(text="a"),
        batch_api.batch_custom_id(1, 0, None, "nsq"): BatchOutcome(text="b"),
    }
    grid = batch_api.reshape_outcomes(
        outcomes, num_prompts=2, num_responses=1, seed=None, namespace="nsq"
    )
    assert [[o.text for o in row] for row in grid] == [["a"], ["b"]]


def _block(tmp_path, client, **kwargs):
    from ruleforge.judges.openai_batch import make_block_generate

    return make_block_generate(
        model_name="test-model", root=tmp_path, client_factory=lambda: client,
        poll_seconds=1.0, **kwargs,
    )


def test_openai_block_returns_one_detail_per_prompt_with_errors_kept_apart(
    tmp_path, confirmed
):
    def flaky(custom_id, body):
        if custom_id.endswith("p000001-c00"):
            return None
        if custom_id.endswith("p000002-c00"):
            return {"custom_id": custom_id, "error": {"message": "overloaded"}}
        return success_record(custom_id, body)

    client = FakeOpenAI(responder=flaky)
    tracker = SimpleNamespace(added=[], add=lambda usage: tracker.added.append(usage))
    generate_block = _block(tmp_path, client, tracker=tracker)
    block_id = "c/k/draw-000/attempt-0/block-0000"
    details = generate_block(PROMPTS, 1.0, 64, None, block_id)

    namespace = request_namespace(block_id)
    assert details[0]["completion"] == f"answer::{namespace}-p000000-c00"
    assert details[0]["raw_response"] == details[0]["completion"]
    assert details[0]["error"] is None
    assert details[0]["provider_request_ref"] == f"{namespace}-p000000-c00"
    assert details[0]["request_block_ref"].startswith(str(tmp_path))
    assert details[1]["completion"] is None and details[1]["error"] is None
    assert details[2]["completion"] is None and "overloaded" in details[2]["error"]
    assert len(tracker.added) == 1
    uploaded = b"".join(client.uploads.values()).decode("utf-8")
    assert '"seed"' not in uploaded
    assert len(client.create_calls) == 1


def test_openai_block_reuses_the_batch_for_the_same_block_and_buys_a_new_attempt(
    tmp_path, confirmed
):
    client = FakeOpenAI()
    generate_block = _block(tmp_path, client)
    first = generate_block(PROMPTS, 1.0, 64, None, "c/k/draw-000/attempt-0/block-0000")
    again = generate_block(PROMPTS, 1.0, 64, None, "c/k/draw-000/attempt-0/block-0000")
    assert len(client.create_calls) == 1
    assert first == again
    generate_block(PROMPTS, 1.0, 64, None, "c/k/draw-000/attempt-1/block-0000")
    generate_block(PROMPTS, 1.0, 64, None, "c/k/draw-001/attempt-0/block-0000")
    assert len(client.create_calls) == 3


def test_openai_block_warns_once_that_responses_drops_seeds(tmp_path, confirmed, caplog):
    generate_block = _block(tmp_path, FakeOpenAI())
    with caplog.at_level("WARNING", logger="ruleforge.judges.models"):
        generate_block(PROMPTS, 1.0, 64, [1, 2, 3], "c/k/draw-000/attempt-0/block-0000")
        generate_block(PROMPTS, 1.0, 64, [4, 5, 6], "c/k/draw-000/attempt-1/block-0000")
    assert caplog.text.count("no seed parameter") == 1


def test_openai_block_sends_per_prompt_seeds_on_chat(tmp_path, confirmed):
    def chat_record(custom_id, body):
        return {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": f"seed={body.get('seed')}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            },
        }

    client = FakeOpenAI(responder=chat_record)
    generate_block = _block(tmp_path, client, endpoint=CHAT_COMPLETIONS_ENDPOINT, send_max_output_tokens=True)
    details = generate_block(PROMPTS, 1.0, 64, [11, None, 13], "c/k/draw-000/attempt-0/block-0000")
    assert [d["completion"] for d in details] == ["seed=11", "seed=None", "seed=13"]


def test_vllm_layers_the_shipped_truncation_over_the_full_distribution(fake_vllm):
    llm = _FakeLLM()
    models._generate_with_vllm(
        llm, ["a", "b"], temperature=1.0, max_new_tokens=8, num_responses=1, seed=[1, 2],
        truncation={"top_k": 64, "top_p": 0.95},
    )
    _, params, _ = llm.calls[0]
    for p in params:
        assert (p.kwargs["top_k"], p.kwargs["top_p"], p.kwargs["min_p"], p.kwargs["repetition_penalty"]) == (64, 0.95, 0.0, 1.0)
    models._generate_with_vllm(llm, ["a"], temperature=1.0, max_new_tokens=8, num_responses=1, seed=[3])
    _, params, _ = llm.calls[1]
    assert {k: params[0].kwargs[k] for k in models.UNTRUNCATED_SAMPLING} == models.UNTRUNCATED_SAMPLING


def test_generate_once_forwards_the_truncation_to_a_local_judge_only(stub_backend):
    seen = []

    def details(prompts, temperature, max_tokens, n, seeds, **kwargs):
        seen.append(kwargs)
        return [[{"raw_response": "no", "completion": "no", "finish_reason": "stop"}] for _ in prompts]

    local = stub_backend(model_type="local", temperature=1.0, _batch_generate_with_details=details)
    local.generate_once([{"user_prompt": "q"}], temperature=1.0, truncation={"top_k": 64})
    assert seen == [{"truncation": {"top_k": 64}}]
    seen.clear()
    provider = stub_backend(model_type="openai", temperature=1.0, _batch_generate_with_details=details)
    provider.generate_once([{"user_prompt": "q"}], temperature=1.0, truncation={"top_k": 64})
    assert seen == [{}]


@pytest.mark.parametrize("model_type", ["local", "openai", "openai_batch", "anthropic", "anthropic_batch"])
def test_a_built_in_provider_refuses_a_base_url_it_would_ignore(model_type, tmp_path):
    config = {"model_type": model_type, "model_name": "test-model", "base_url": "https://router.example/v1"}
    with pytest.raises(JudgeError, match=r"takes no base_url.*remove base_url from its config") as refused:
        JudgeSpec.from_config(config, batch_root=str(tmp_path))
    assert "router.example" not in str(refused.value)


@pytest.mark.parametrize("model_type", ["openai_batch", "anthropic_batch"])
@pytest.mark.parametrize("root", [{}, {"batch_root": "batches"}], ids=["no_batch_root", "batch_root"])
def test_a_batch_backend_built_directly_refuses_a_base_url_it_would_ignore(model_type, root):
    config = OmegaConf.create({"model_type": model_type, "model_name": "test-model", **root,
                               "base_url": "https://other.example/v1"})
    with pytest.raises(JudgeError, match=r"takes no base_url") as refused:
        ModelBackend(config)
    assert "other.example" not in str(refused.value)


def test_a_null_base_url_is_no_base_url(tmp_path):
    config = {"model_type": "openai", "model_name": "test-model"}
    assert JudgeSpec.from_config(config, base_url=None).options["base_url"] is None
