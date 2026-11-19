from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from ruleforge.judges.models import (
    _api_generate_batch,
    _api_generate_batch_with_details,
    _extract_text_and_strip_status,
    call_local_model_with_details,
    parse_json_from_response,
)
from tests.identity import published as pub


def _make_backend(
    batch_generate_fn,
    base_delay: float = 0.0,
    num_tries: int = 3,
    seed: int = 42,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
) -> Any:
    from ruleforge.judges.backend import ModelBackend

    backend = object.__new__(ModelBackend)
    # Kept in sync with the stub_backend fixture in conftest.py.
    backend.fixed_temperature = None
    backend.max_retry_seconds = None
    backend.model_type = "test"
    backend.model_name = "test-model"
    backend.max_new_tokens = max_new_tokens
    backend.seed = seed
    backend.temperature = temperature
    backend.num_tries = num_tries
    backend._batch_generate = batch_generate_fn
    backend._base_delay = base_delay
    return backend


class TestExtractTextFromCandidate:
    def test_plain_text_no_stripping(self):
        candidate = {"text": "hello world"}
        assert (
            _extract_text_and_strip_status(candidate, strip_thinking_model=None)[0]
            == "hello world"
        )

    def test_strips_whitespace(self):
        candidate = {"text": "  hello  "}
        assert (
            _extract_text_and_strip_status(candidate, strip_thinking_model=None)[0]
            == "hello"
        )

    def test_empty_text(self):
        candidate = {"text": ""}
        assert _extract_text_and_strip_status(candidate, strip_thinking_model=None)[0] == ""

    def test_missing_text_key(self):
        candidate = {"token_ids": [1, 2, 3]}
        assert _extract_text_and_strip_status(candidate, strip_thinking_model=None)[0] == ""

    def test_unsupported_strip_model_leaves_no_answer(self):
        candidate = {"text": "<think>reasoning</think>answer"}
        result = _extract_text_and_strip_status(
            candidate, strip_thinking_model="nonexistent_model"
        )[0]
        assert result == ""

    def test_think_tag_stripping(self):
        candidate = {"text": "<think>internal thought</think>the actual answer"}
        result = _extract_text_and_strip_status(candidate, strip_thinking_model="qwen3")[0]
        assert result == "the actual answer"


class TestLocalGenerationDetails:
    def test_preserves_raw_reasoning_before_thinking_strip(self, monkeypatch):
        class Tokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return "formatted prompt"

            def encode(self, text, add_special_tokens=True):
                assert (text, add_special_tokens) == ("formatted prompt", False)
                return [7, 8]

        raw = "<think>reasoning trace</think>yes"
        sent = []

        def generate(_llm, prompts, **_kwargs):
            sent.extend(prompts)
            return [[{"text": raw, "token_ids": [1, 2, 3], "finish_reason": "stop"}]]

        monkeypatch.setattr("ruleforge.judges.models._generate_with_vllm", generate)

        details = call_local_model_with_details(
            object(),
            Tokenizer(),
            [{"system_prompt": "judge", "user_prompt": ""}],
            strip_thinking_model="qwen3",
        )

        assert details == [
            [
                {
                    "raw_response": raw,
                    "completion": "yes",
                    "finish_reason": "stop",
                }
            ]
        ]
        assert sent == [{"prompt_token_ids": [7, 8]}]


class TestApiGenerateBatch:
    def test_single_prompt_single_response(self):
        def fake_call(sys, usr, temp, max_tok, seed):
            return {"completion": f"response to: {usr}"}

        prompts = [{"system_prompt": "sys", "user_prompt": "hello"}]
        result = _api_generate_batch(
            fake_call,
            prompts,
            temperature=0.0,
            max_new_tokens=100,
            num_responses=1,
            seed=42,
        )

        assert len(result) == 1
        assert result[0] == ["response to: hello"]

    def test_multiple_prompts(self):
        def fake_call(sys, usr, temp, max_tok, seed):
            return {"completion": f"re:{usr}"}

        prompts = [
            {"system_prompt": "", "user_prompt": "a"},
            {"system_prompt": "", "user_prompt": "b"},
        ]
        result = _api_generate_batch(
            fake_call,
            prompts,
            temperature=0.0,
            max_new_tokens=100,
            num_responses=1,
            seed=1,
        )

        assert len(result) == 2
        assert result[0] == ["re:a"]
        assert result[1] == ["re:b"]

    def test_num_responses_multiple(self):
        call_count = [0]

        def fake_call(sys, usr, temp, max_tok, seed):
            call_count[0] += 1
            return {"completion": f"resp_{call_count[0]}"}

        prompts = [{"system_prompt": "", "user_prompt": "q"}]
        result = _api_generate_batch(
            fake_call,
            prompts,
            temperature=0.0,
            max_new_tokens=100,
            num_responses=3,
            seed=10,
        )

        assert len(result) == 1
        assert len(result[0]) == 3
        assert call_count[0] == 3

    def test_a_failed_or_blank_call_becomes_empty_string(self):
        answers = iter([{"completion": "", "error": "RuntimeError: offline"}, {"completion": None}])

        def fake_call(sys, usr, temp, max_tok, seed):
            return next(answers)

        prompts = [{"system_prompt": "", "user_prompt": "q"}, {"system_prompt": "", "user_prompt": "r"}]
        result = _api_generate_batch(
            fake_call,
            prompts,
            temperature=0.0,
            max_new_tokens=100,
            num_responses=1,
            seed=1,
        )

        assert result == [[""], [""]]

    def test_the_detail_path_keeps_each_call_s_detail(self):
        details = iter([{"completion": "yes", "error": None}, {"completion": "", "error": "RuntimeError: offline"}])

        def fake_call(sys, usr, temp, max_tok, seed):
            return next(details)

        prompts = [{"system_prompt": "s", "user_prompt": ""}]
        result = _api_generate_batch_with_details(fake_call, prompts, 1.0, 10, 2, None)
        assert result == [[{"completion": "yes", "error": None}, {"completion": "", "error": "RuntimeError: offline"}]]

    def test_seed_varies_per_response(self):
        seeds_seen = []

        def fake_call(sys, usr, temp, max_tok, seed):
            seeds_seen.append(seed)
            return {"completion": "ok"}

        prompts = [{"system_prompt": "", "user_prompt": "q"}]
        _api_generate_batch(
            fake_call,
            prompts,
            temperature=0.0,
            max_new_tokens=100,
            num_responses=3,
            seed=10,
        )

        assert seeds_seen == [10, 11, 12]


class TestParseJsonFromResponse:
    def test_plain_json(self):
        result = parse_json_from_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        text = '```json\n{"rules": [{"id": "R1"}]}\n```'
        result = parse_json_from_response(text)
        assert result == {"rules": [{"id": "R1"}]}

    def test_json_with_surrounding_text(self):
        text = 'Here is the result: {"score": 3} end'
        result = parse_json_from_response(text)
        assert result == {"score": 3}

    def test_empty_string(self):
        assert parse_json_from_response("") is None

    def test_no_json(self):
        assert parse_json_from_response("just some text with no json") is None

    def test_invalid_json(self):
        assert parse_json_from_response("{invalid json}") is None


class TestRetryGenerate:
    def test_success_on_first_attempt(self):
        call_count = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            call_count[0] += 1
            return [["response"]]

        backend = _make_backend(fake_batch)
        result = backend.generate([{"system_prompt": "", "user_prompt": "hi"}])

        assert result == [["response"]]
        assert call_count[0] == 1

    def test_retry_on_empty_response(self):
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                return [[""]]
            return [["got it"]]

        backend = _make_backend(fake_batch, num_tries=3)
        result = backend.generate([{"system_prompt": "", "user_prompt": "hi"}])

        assert result == [["got it"]]
        assert attempt[0] == 2

    def test_only_failed_prompts_retried(self):
        batch_sizes = []

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            batch_sizes.append(len(prompts))
            results = []
            for p in prompts:
                if "good" in p["user_prompt"]:
                    results.append(["success"])
                else:
                    results.append([""])
            return results

        backend = _make_backend(fake_batch, num_tries=2)
        prompts = [
            {"system_prompt": "", "user_prompt": "good"},
            {"system_prompt": "", "user_prompt": "bad"},
        ]
        result = backend.generate(prompts)

        assert batch_sizes == [2, 1]  # first: both, second: only failed
        assert result[0] == ["success"]
        assert result[1] is None

    def test_exhaustion_returns_none(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [[""] for _ in prompts]

        backend = _make_backend(fake_batch, num_tries=2)
        result = backend.generate([{"system_prompt": "", "user_prompt": "q"}])

        assert result == [None]

    def test_num_responses_multiple_all_succeed(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [["a", "b", "c"] for _ in prompts]

        backend = _make_backend(fake_batch)
        result = backend.generate(
            [{"system_prompt": "", "user_prompt": "q"}],
            num_responses=3,
        )

        assert result == [["a", "b", "c"]]

    def test_num_responses_accumulate_across_retries(self):
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                return [["", "", ""]]  # 0 valid → retry, still need 3
            if attempt[0] == 2:
                return [["ok", "", "x"]]  # 2 valid → retry, still need 1
            return [["last", "", ""]]  # 1 valid → done

        backend = _make_backend(fake_batch, num_tries=5)
        result = backend.generate(
            [{"system_prompt": "", "user_prompt": "q"}],
            num_responses=3,
        )

        assert attempt[0] == 3
        assert result == [["ok", "x", "last"]]

    def test_num_responses_partial_on_exhaustion(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [["only", "", ""]]  # 1 valid each attempt, never reaches 3

        backend = _make_backend(fake_batch, num_tries=2)
        result = backend.generate(
            [{"system_prompt": "", "user_prompt": "q"}],
            num_responses=3,
        )

        assert result == [["only", "only"]]  # 2 attempts × 1 valid = 2 collected

    def test_deterministic_seeds(self):
        seeds_seen = []

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            seeds_seen.append(seed)
            return [[""] for _ in prompts]

        backend = _make_backend(fake_batch, num_tries=3, seed=42)
        backend.generate([{"system_prompt": "", "user_prompt": "q"}])

        first_run = list(seeds_seen)
        seeds_seen.clear()

        backend2 = _make_backend(fake_batch, num_tries=3, seed=42)
        backend2.generate([{"system_prompt": "", "user_prompt": "q"}])

        assert first_run == seeds_seen
        assert len(set(first_run)) == len(first_run)

    def test_exception_in_batch_generate_retries(self):
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("GPU error")
            return [["recovered"]]

        backend = _make_backend(fake_batch, num_tries=3)
        result = backend.generate([{"system_prompt": "", "user_prompt": "q"}])

        assert result == [["recovered"]]
        assert attempt[0] == 2


class TestRetryWithValidation:
    def test_valid_response_first_attempt(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [["yes"] for _ in prompts]

        backend = _make_backend(fake_batch)
        result = backend.generate_with_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
            valid_responses=("yes", "no", "unsure"),
        )

        assert result == ["yes"]

    def test_retry_on_invalid_response(self):
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                return [["maybe"]]
            return [["yes"]]

        backend = _make_backend(fake_batch, num_tries=3)
        result = backend.generate_with_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
            valid_responses=("yes", "no"),
        )

        assert result == ["yes"]
        assert attempt[0] == 2

    def test_exhaustion_returns_none(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [["invalid"] for _ in prompts]

        backend = _make_backend(fake_batch, num_tries=2)
        result = backend.generate_with_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
            valid_responses=("yes", "no"),
        )

        assert result == [None]

    def test_partial_batch_retry(self):
        batch_sizes = []
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            batch_sizes.append(len(prompts))
            attempt[0] += 1
            if attempt[0] == 1:
                return [["yes"], ["invalid"], ["no"]]
            return [["unsure"]]

        backend = _make_backend(fake_batch, num_tries=3)
        prompts = [
            {"system_prompt": "", "user_prompt": "a"},
            {"system_prompt": "", "user_prompt": "b"},
            {"system_prompt": "", "user_prompt": "c"},
        ]
        result = backend.generate_with_validation(
            prompts, valid_responses=("yes", "no", "unsure")
        )

        assert batch_sizes == [3, 1]
        assert result == ["yes", "unsure", "no"]

    def test_case_insensitive_validation(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [["YES"]]

        backend = _make_backend(fake_batch)
        result = backend.generate_with_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
            valid_responses=("yes", "no"),
        )

        assert result == ["yes"]

    def test_traced_validation_keeps_rejected_and_selected_raw_attempts(self):
        attempt = [0]

        def fake_details(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                return [
                    [
                        {
                            "raw_response": "reasoning then maybe",
                            "completion": "maybe",
                            "finish_reason": "stop",
                        }
                    ]
                ]
            return [
                [
                    {
                        "raw_response": "reasoning then yes",
                        "completion": "yes",
                        "finish_reason": "stop",
                    }
                ]
            ]

        backend = _make_backend(lambda *_args: [[]], num_tries=3)
        backend._batch_generate_with_details = fake_details
        judgments, traces = backend.generate_with_validation_and_traces(
            [{"system_prompt": "judge", "user_prompt": ""}],
            valid_responses=("yes", "no"),
        )

        assert judgments == ["yes"]
        assert traces[0]["judgment"] == "yes"
        assert traces[0]["selected_attempt"] == 2
        assert [
            item["candidates"][0]["raw_response"] for item in traces[0]["attempts"]
        ] == ["reasoning then maybe", "reasoning then yes"]
        assert [
            item["candidates"][0]["accepted"] for item in traces[0]["attempts"]
        ] == [False, True]

    def test_json_validation_success(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [['{"rules": [{"id": "R1"}]}']]

        backend = _make_backend(fake_batch)
        result = backend.generate_with_json_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
        )

        assert result == [{"rules": [{"id": "R1"}]}]

    def test_json_validation_retry_on_invalid(self):
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                return [["not json at all"]]
            return [['{"valid": true}']]

        backend = _make_backend(fake_batch, num_tries=3)
        result = backend.generate_with_json_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
        )

        assert result == [{"valid": True}]
        assert attempt[0] == 2

    def test_json_validation_exhaustion_returns_none(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [["still not json"]]

        backend = _make_backend(fake_batch, num_tries=2)
        result = backend.generate_with_json_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
        )

        assert result == [None]

    def test_json_in_code_block_accepted(self):
        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            return [['```json\n{"key": "val"}\n```']]

        backend = _make_backend(fake_batch)
        result = backend.generate_with_json_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
        )

        assert result == [{"key": "val"}]

    def test_exception_in_batch_retries(self):
        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] == 1:
                raise RuntimeError("API timeout")
            return [["yes"]]

        backend = _make_backend(fake_batch, num_tries=3)
        result = backend.generate_with_validation(
            [{"system_prompt": "", "user_prompt": "q"}],
            valid_responses=("yes", "no"),
        )

        assert result == ["yes"]
        assert attempt[0] == 2


class TestBackoff:
    def test_no_backoff_when_base_delay_zero(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(
            "ruleforge.judges.backend.time.sleep", lambda d: sleep_calls.append(d)
        )

        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] < 3:
                return [[""]]
            return [["ok"]]

        backend = _make_backend(fake_batch, base_delay=0.0, num_tries=5)
        backend.generate([{"system_prompt": "", "user_prompt": "q"}])

        assert sleep_calls == []

    def test_exponential_backoff_for_api(self, monkeypatch):
        sleep_calls = []
        monkeypatch.setattr(
            "ruleforge.judges.backend.time.sleep", lambda d: sleep_calls.append(d)
        )

        attempt = [0]

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            attempt[0] += 1
            if attempt[0] < 4:
                return [[""]]
            return [["ok"]]

        backend = _make_backend(fake_batch, base_delay=1.0, num_tries=5)
        backend.generate([{"system_prompt": "", "user_prompt": "q"}])

        # Backoff: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0, 1.0 * 2^2 = 4.0
        assert sleep_calls == [1.0, 2.0, 4.0]


class TestParameterResolution:
    def test_defaults_from_backend(self):
        call_args = {}

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            call_args.update(temp=temp, max_tok=max_tok, seed=seed)
            return [["ok"]]

        backend = _make_backend(
            fake_batch,
            temperature=0.7,
            max_new_tokens=2048,
            seed=99,
            num_tries=1,
        )
        backend.generate([{"system_prompt": "", "user_prompt": "q"}])

        # seed is derived from RNG seeded with 99, not 99 directly
        assert call_args["temp"] == 0.7
        assert call_args["max_tok"] == 2048

    def test_overrides_take_precedence(self):
        call_args = {}

        def fake_batch(prompts, temp, max_tok, num_resp, seed):
            call_args.update(temp=temp, max_tok=max_tok)
            return [["ok"]]

        backend = _make_backend(
            fake_batch,
            temperature=0.7,
            max_new_tokens=2048,
            seed=99,
            num_tries=1,
        )
        backend.generate(
            [{"system_prompt": "", "user_prompt": "q"}],
            temperature=0.1,
            max_new_tokens=512,
        )

        assert call_args["temp"] == 0.1
        assert call_args["max_tok"] == 512


def test_olmo3_is_registered_with_the_think_tag_stripper():
    from ruleforge.judges.models import _MODEL_STRIP_FN, _strip_think_tags

    assert _MODEL_STRIP_FN["olmo3"] is _strip_think_tags
    candidate = {"text": "<think>weighing the rule</think>no"}
    assert _extract_text_and_strip_status(candidate, strip_thinking_model="olmo3")[0] == "no"


def test_model_default_truncation_reads_only_the_explicit_truncation_keys(tmp_path):
    """Mirrors vLLM: only keys generation_config.json states are applied."""
    from ruleforge.judges.models import TRUNCATION_KEYS, UNTRUNCATED_SAMPLING, model_default_truncation

    assert model_default_truncation(tmp_path) == {}
    (tmp_path / "generation_config.json").write_text(
        '{"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 5, "do_sample": true}', encoding="utf-8"
    )
    assert model_default_truncation(tmp_path) == {"top_p": 0.95}
    (tmp_path / "generation_config.json").write_text(
        '{"temperature": 1.0, "top_k": 64, "top_p": 0.95, "repetition_penalty": 1.1, "min_p": 0.05}', encoding="utf-8"
    )
    assert model_default_truncation(tmp_path) == {"top_k": 64, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.1}
    assert set(TRUNCATION_KEYS) == set(UNTRUNCATED_SAMPLING)


# The Qwen3.5-9B card's thinking-mode values, as a model config spells them.
CARD_TRUNCATION = {"top_k": 20, "top_p": 0.95, "min_p": 0.0}


def test_default_truncation_fills_only_the_keys_the_weights_do_not_ship(tmp_path, caplog):
    """Resolution order: UNTRUNCATED_SAMPLING <- default_truncation <- generation_config.json."""
    import logging

    from ruleforge.judges.models import model_default_truncation

    assert model_default_truncation(tmp_path, CARD_TRUNCATION) == CARD_TRUNCATION
    assert model_default_truncation(tmp_path, None) == model_default_truncation(tmp_path) == {}
    assert model_default_truncation(tmp_path, {}) == {}

    (tmp_path / "generation_config.json").write_text('{"top_k": 20, "top_p": 0.8}', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="ruleforge.judges.models"):
        resolved = model_default_truncation(tmp_path, CARD_TRUNCATION)
    assert resolved == {"top_k": 20, "top_p": 0.8, "min_p": 0.0}
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "top_p=0.8" in warnings[0] and "top_p=0.95" in warnings[0]
    assert "top_k" not in warnings[0]
    assert model_default_truncation(tmp_path) == {"top_k": 20, "top_p": 0.8}


def test_default_truncation_is_canonicalised_to_the_sampler_types():
    from omegaconf import OmegaConf

    from ruleforge.judges.models import validate_default_truncation

    canonical = validate_default_truncation({"top_k": 20.0, "top_p": 1, "min_p": 0, "repetition_penalty": 1})
    assert canonical == {"top_k": 20, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0}
    assert [type(canonical[k]) for k in ("top_k", "top_p", "min_p", "repetition_penalty")] == [int, float, float, float]
    assert validate_default_truncation(OmegaConf.create(CARD_TRUNCATION)) == CARD_TRUNCATION
    assert validate_default_truncation(None) == {}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({**CARD_TRUNCATION, "presence_penalty": 1.5}, r"\['presence_penalty'\] are not sampler truncation keys"),
        ({"temperature": 0.6, "frequency_penalty": 0.1}, r"\['frequency_penalty', 'temperature'\]"),
        ([20, 0.95], "must be a mapping"),
        ({"top_k": 20.5}, "top_k must be an integer"),
        ({"top_p": "0.95"}, "top_p must be a finite number"),
        ({"top_p": None}, "top_p must be a finite number"),
        ({"min_p": True}, "min_p must be a finite number"),
        ({"top_p": float("nan")}, "top_p must be a finite number"),
        ({"top_k": float("inf")}, "top_k must be a finite number"),
    ],
)
def test_default_truncation_refuses_what_the_sampler_would_not_apply(tmp_path, values, message):
    from ruleforge.judges.models import model_default_truncation, validate_default_truncation

    with pytest.raises(ValueError, match=message):
        validate_default_truncation(values)
    with pytest.raises(ValueError, match=message):
        model_default_truncation(tmp_path, values)


def test_a_rendered_prompt_is_sent_with_the_template_s_one_beginning_of_text_token(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from transformers import AutoTokenizer

    from ruleforge.judges.models import (
        GLIMMER_SPECIAL_IDS,
        _ThinkingTokenizer,
        format_prompt_for_generation,
        prompt_token_ids,
    )
    from tests.tiny_glimmer import save_tiny_glimmer

    tokenizer = _ThinkingTokenizer(AutoTokenizer.from_pretrained(str(save_tiny_glimmer(tmp_path / "glimmer"))), True)
    bos = tokenizer.bos_token_id
    rendered = format_prompt_for_generation(tokenizer, {"system_prompt": "judge the rule set", "user_prompt": ""})
    assert rendered.startswith("<|begin_of_text|><|start|>system<|message|>judge the rule set\n\nReasoning")
    assert list(tokenizer.encode(rendered))[:2] == [bos, bos]  # what a text prompt would have been
    ids = prompt_token_ids(tokenizer, rendered)
    assert ids[0] == bos and ids[1] != bos and ids.count(bos) == 1

    special = GLIMMER_SPECIAL_IDS
    glimmer_ids = [5, special["<|message|>"], 6, special["<|eom|>"], special["<|start|>"], 8, 9,
                   special["<|message|>"], 10, special["<|eot|>"]]

    class LLM:
        def __init__(self):
            self.sent = []

        def generate(self, prompts, sampling_params=None, **_kwargs):
            self.sent.extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(
                text=" to=self<|message|>r<|eom|><|start|>assistant to=user<|message|>no", token_ids=glimmer_ids,
                finish_reason="stop")]) for _ in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: kwargs))
    llm = LLM()
    details = call_local_model_with_details(
        llm, tokenizer, [{"system_prompt": "judge the rule set", "user_prompt": ""}],
        temperature=1.0, seed=[3], strip_thinking_model="muse_glimmer")
    assert llm.sent == [{"prompt_token_ids": ids}]
    assert details[0][0]["completion"] == "no"


REPO = Path(__file__).resolve().parents[1]
GLIMMER = "Muse-Glimmer-30B"
#: The judges every local model config's weights directory names: the four open judges' and Glimmer's.
LOCAL_JUDGE_DIRECTORIES = {"gemma-4-31B-it", "Olmo-3-32B-Think", "gpt-oss-120b", "Qwen3-8B", GLIMMER}


def _local_renderings() -> list[tuple[str, bool, str | None]]:
    import yaml

    renderings = set()
    for config in sorted((REPO / "configs" / "model").glob("*.yaml")):
        model = yaml.safe_load(config.read_text(encoding="utf-8"))
        if model.get("model_type") == "local":
            renderings.add((Path(str(model["model_path"])).name, bool(model.get("enable_thinking")),
                            model.get("gpt_oss_reasoning_level")))
    return sorted(renderings, key=lambda rendering: (rendering[0], rendering[1], str(rendering[2])))


LOCAL_RENDERINGS = _local_renderings()


def test_each_local_judge_renders_its_prompts_one_way():
    assert sorted(name for name, _, _ in LOCAL_RENDERINGS) == sorted(LOCAL_JUDGE_DIRECTORIES)


def _population_prompts() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """``{style: (prompts, judged components)}`` for every style, as a paper-population campaign renders them."""
    import contextlib
    from types import SimpleNamespace

    from ruleforge.core.artifacts import load_jsonl_records
    from ruleforge.evaluate.campaign import campaign_rule_levels
    from ruleforge.evaluate.validation import ComponentSchema
    from ruleforge.grpo.grpo_data import CampaignPrompts, condition_task_spec
    from ruleforge.launch.guards import compose
    from ruleforge.prompts.styles import HIERARCHY_STYLES, VALID_REASONING_PROMPT_STYLES
    from ruleforge.tasks.spec import build_task_spec

    rules = load_jsonl_records(REPO / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl")
    isolated = ComponentSchema.from_rules(rules, prompt_style="naive", forced_json_requirement="complete_rule_map",
                                          evaluation_mode="both").component_ids
    found = {}
    for style in VALID_REASONING_PROMPT_STYLES:
        if style not in HIERARCHY_STYLES:
            condition = SimpleNamespace(judge_type="No_Interpretation", prompt_style=style,
                                        individual_rule_context="isolated")
            found[style] = (CampaignPrompts.build(task_spec=condition_task_spec(condition), rules=rules,
                                                  judge_type="No_Interpretation"), isolated)
            continue
        cfg = compose(REPO, ["+hierarchy=model_spec", f"eval_cfg.reasoning_prompt_style={style}"],
                      default_config="sampling_campaign")
        with contextlib.chdir(REPO):  # the checkout's top, where a campaign reads the paths its config names
            levels, task_spec = campaign_rule_levels(cfg), build_task_spec(cfg)
            assert load_jsonl_records(Path(str(cfg.rule_set_path))) == rules
        assert (str(cfg.interpretation), task_spec.individual_rule_context) == ("No_Interpretation", "superiors")
        components = ComponentSchema.from_rules(
            rules, prompt_style=style, forced_json_requirement=str(cfg.sampling.forced_json_requirement),
            evaluation_mode=str(cfg.evaluation_mode)).component_ids
        found[style] = (CampaignPrompts.build(task_spec=task_spec, rules=rules, judge_type=str(cfg.interpretation),
                                              levels=levels.by_id), components)
    return found


def _population_prompt_entries(prompts: dict[str, tuple[Any, tuple[str, ...]]]):
    """``(style, prompt entry)`` for every prompt of ``prompts`` over each scenario of the frozen population."""
    from ruleforge.core.artifacts import load_jsonl_records

    rows = load_jsonl_records(REPO / "data" / "populations" / "frozen_500_r3.jsonl")
    for style, (built, components) in prompts.items():
        for row in rows:
            for component in components:
                yield style, built.entry(row, component)


@pytest.mark.cluster
@pytest.mark.parametrize("rendering", LOCAL_RENDERINGS, ids=lambda rendering: rendering[0])
def test_every_local_judge_s_prompts_are_the_ids_a_text_prompt_had_but_glimmer_s(rendering):
    import itertools
    import os
    from collections import Counter

    from transformers import AutoTokenizer

    from ruleforge.judges.models import _ThinkingTokenizer, format_prompt_for_generation, prompt_token_ids
    from ruleforge.prompts.styles import HIERARCHY_STYLES, NAIVE_HIERARCHY

    name, enable_thinking, reasoning_level = rendering
    directory = Path(os.environ.get("RULEFORGE_MODEL_ROOT", "/nonexistent")) / name
    if not directory.is_dir():
        pytest.skip(f"needs {name} under RULEFORGE_MODEL_ROOT")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True)
    renderer = _ThinkingTokenizer(tokenizer, enable_thinking)
    styles = _population_prompts()
    bos, prompts, differing = tokenizer.bos_token_id, Counter(), 0
    for chunk in itertools.batched(_population_prompt_entries(styles), 1000):
        rendered = [format_prompt_for_generation(renderer, entry, gpt_oss_reasoning_level=reasoning_level)
                    for _, entry in chunk]
        encoded = tokenizer(rendered, add_special_tokens=True)["input_ids"]
        for (style, _), text, as_text in zip(chunk, rendered, encoded):
            sent, as_text = prompt_token_ids(renderer, text), list(as_text)
            if name == GLIMMER:
                assert as_text == [bos, *sent] and sent[:1] == [bos] != sent[1:2], text[:200]
            differing += sent != as_text
            prompts[style] += 1
    isolated = [style for style in styles if style not in HIERARCHY_STYLES]
    assert len(isolated) == 5 and tuple(HIERARCHY_STYLES) == (NAIVE_HIERARCHY,)
    hierarchy = 500 * len(styles[NAIVE_HIERARCHY][1])
    assert hierarchy == 500 * 58
    assert prompts == {**dict.fromkeys(isolated, 500 * 58), NAIVE_HIERARCHY: hierarchy}
    assert differing == (prompts.total() if name == GLIMMER else 0)


@lru_cache(maxsize=None)
def _population() -> dict[str, dict[str, Any]]:
    from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records

    return {compute_scenario_id(row): dict(row)
            for row in iter_jsonl_records(REPO / "data" / "populations" / "frozen_500_r3.jsonl")}


def _subset_rule_files(seed: int, work: Path) -> dict[str, Path]:
    from ruleforge.rulesets.subsets import file_digest, generate_subset_files, load_manifest

    subsets = REPO / "data" / "rules" / "subset_manifest"
    meta, entries = load_manifest(subsets)
    directory = work / f"seed_{seed}"
    written = generate_subset_files(REPO / meta["source_ruleset"]["path"], directory, meta["sampling_strategy"], seed)
    digests = {f"seed_{seed}/{path.name}": file_digest(path) for path in written}
    assert digests == {path: digest for path, digest in entries.items() if path.startswith(f"seed_{seed}/")}
    return {digest: directory / Path(path).name for path, digest in digests.items()}


def _glimmer_bos_once(entries: list[dict[str, str]]) -> int:
    import os

    from transformers import AutoTokenizer

    from ruleforge.judges.models import _ThinkingTokenizer, format_prompt_for_generation, prompt_token_ids

    tokenizer = AutoTokenizer.from_pretrained(str(Path(os.environ["RULEFORGE_MODEL_ROOT"]) / GLIMMER),
                                              trust_remote_code=True)
    renderer = _ThinkingTokenizer(tokenizer, True)
    bos, once = tokenizer.bos_token_id, 0
    for entry in entries:
        sent = prompt_token_ids(renderer, format_prompt_for_generation(renderer, entry, gpt_oss_reasoning_level=None))
        once += sent[:1] == [bos] and bos not in sent[1:]
    return once


@pytest.mark.cluster
@pytest.mark.parametrize(("root", "index"), pub.GLIMMER_SHARDS, ids=pub.GLIMMER_SHARD_IDS)
def test_every_prompt_glimmer_s_published_roots_sent_is_rendered_again_and_sent_with_one_bos(root, index, tmp_path):
    import os

    from ruleforge.core.artifacts import iter_jsonl_records, prompt_sha256
    from ruleforge.core.errors import ArtifactRootError
    from ruleforge.env import artifact_root
    from tests.support.prompts import prompt_renderer
    from ruleforge.rulesets.levels import load_rule_levels

    if not (Path(os.environ.get("RULEFORGE_MODEL_ROOT", "/nonexistent")) / GLIMMER).is_dir():
        pytest.skip(f"needs {GLIMMER} under RULEFORGE_MODEL_ROOT")
    try:
        base = artifact_root() / "sampling" / root
    except ArtifactRootError as error:
        pytest.skip(str(error))
    committed, subsets = pub.repo_rule_files(), {}
    by_digest = {digest: path for (_, digest), path in committed.items()}
    renderers: dict[str, Any] = {}
    digests: dict[tuple[str, str, str], str] = {}
    first: dict[str, dict[str, str]] = {}
    checked, differing = 0, []
    for condition, draw in pub.shard_draws(base, index, pub.GLIMMER_ROOTS[root]):
        cid = condition.condition_id
        if cid not in renderers:
            rules = committed.get((condition.rule_set_name, condition.rule_set_sha256))
            if rules is None:
                seed = int(dict(condition.extra)["subset_seed"])
                if seed not in subsets:
                    subsets[seed] = _subset_rule_files(seed, tmp_path)
                rules = subsets[seed][condition.rule_set_sha256]
                assert rules.name == condition.rule_set_name, cid
            levels = (None if condition.rule_levels_sha256 is None
                      else load_rule_levels(rules, by_digest[condition.rule_levels_sha256]))
            renderers[cid] = prompt_renderer(condition, list(iter_jsonl_records(rules)), levels=levels)
        for row in pub.ledger_rows(draw / "attempts.jsonl"):
            for component in row["components"]:
                key = (cid, row["scenario_id"], component["component_id"])
                if key not in digests:
                    entry = renderers[cid](_population()[row["scenario_id"]], component["component_id"])
                    first.setdefault(cid, entry)
                    digests[key] = prompt_sha256(entry)
                checked += 1
                if digests[key] != component["prompt_sha256"]:
                    differing.append(f"{cid}/{draw.name}: {row['scenario_id'][:12]}/{component['component_id']}")
    assert checked and differing == []
    assert _glimmer_bos_once(list(first.values())) == len(first)


def test_a_chat_template_that_rejects_a_system_role_gets_one_merged_user_turn():
    from ruleforge.judges.models import format_prompt_for_generation

    class Tokenizer:
        def apply_chat_template(self, chat, **kwargs):
            if chat[0]["role"] == "system":
                raise ValueError("system role not supported")
            return "|".join(turn["content"] for turn in chat)

    assert format_prompt_for_generation(Tokenizer(), {"system_prompt": "judge", "user_prompt": "the rule"}) == "judge\nthe rule"
