from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from ruleforge.judges import batch_api
from ruleforge.judges.batch_api import (
    CHAT_COMPLETIONS_ENDPOINT,
    RESPONSES_ENDPOINT,
    BatchAlreadyClaimed,
    BatchOutcome,
    BatchRequest,
    BatchSpendNotConfirmed,
    BatchUsage,
    build_batch_requests,
    claim_for_creation,
)
from tests.support.trees import is_silly_rename

REPO_ROOT = Path(__file__).resolve().parents[1]
LUNA_REQUEST_LINES = REPO_ROOT / "tests" / "fixtures" / "batch" / "luna_ruleset_size_request_lines.json"

EMPTY_USAGE = {
    "input_tokens": 0,
    "cached_tokens": 0,
    "cache_write_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
    "requests": 0,
}


def _usage(**overrides) -> dict:
    """A ``BatchUsage.to_dict()`` snapshot with only the named buckets set."""
    return {**EMPTY_USAGE, **overrides}


def test_batch_usage_counts_a_body_that_reports_no_usage_as_a_request() -> None:
    usage = BatchUsage()
    usage.add_body({})
    usage.add_body({"usage": None})
    usage.add_body({"usage": {}})

    assert usage.to_dict() == _usage(requests=3)


def test_batch_usage_reads_the_responses_style_token_names() -> None:
    """``/v1/responses`` reports input_/output_tokens with *_tokens_details."""
    usage = BatchUsage()
    usage.add_body(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 40,
                "input_tokens_details": {
                    "cached_tokens": 60,
                    "cache_write_tokens": 10,
                },
                "output_tokens_details": {"reasoning_tokens": 25},
            }
        }
    )

    assert usage.to_dict() == _usage(
        input_tokens=100,
        cached_tokens=60,
        cache_write_tokens=10,
        output_tokens=40,
        reasoning_tokens=25,
        requests=1,
    )


def test_batch_usage_reads_the_chat_style_token_names() -> None:
    """``/v1/chat/completions`` reports prompt_/completion_tokens instead."""
    usage = BatchUsage()
    usage.add_body(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {
                    "cached_tokens": 60,
                    "cache_write_tokens": 10,
                },
                "completion_tokens_details": {"reasoning_tokens": 25},
            }
        }
    )

    assert usage.to_dict() == _usage(
        input_tokens=100,
        cached_tokens=60,
        cache_write_tokens=10,
        output_tokens=40,
        reasoning_tokens=25,
        requests=1,
    )


def test_batch_usage_counts_non_numeric_token_values_as_zero() -> None:
    usage = BatchUsage()
    usage.add_body(
        {
            "usage": {
                "input_tokens": "100",
                "output_tokens": None,
                "input_tokens_details": {
                    "cached_tokens": "60",
                    "cache_write_tokens": [10],
                },
                "output_tokens_details": {"reasoning_tokens": "many"},
            }
        }
    )

    assert usage.to_dict() == _usage(requests=1)


def test_batch_usage_falls_back_to_the_chat_name_when_the_first_is_unusable() -> None:
    usage = BatchUsage()
    usage.add_body({"usage": {"input_tokens": "junk", "prompt_tokens": 7}})
    usage.add_body({"usage": {"input_tokens": 5, "prompt_tokens": 7}})

    assert usage.to_dict() == _usage(input_tokens=12, requests=2)


def test_batch_usage_accumulates_across_bodies_of_mixed_shapes() -> None:
    usage = BatchUsage()
    usage.add_body(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 4},
            }
        }
    )
    usage.add_body(
        {
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 8},
            }
        }
    )
    usage.add_body({"usage": {"input_tokens": 30, "output_tokens": 3}})

    assert usage.to_dict() == _usage(
        input_tokens=60,
        cached_tokens=4,
        output_tokens=6,
        reasoning_tokens=8,
        requests=3,
    )


def test_batch_usage_to_dict_carries_every_billed_bucket() -> None:
    usage = BatchUsage(
        input_tokens=1,
        cached_tokens=2,
        cache_write_tokens=3,
        output_tokens=4,
        reasoning_tokens=5,
        requests=6,
    )

    assert usage.to_dict() == {
        "input_tokens": 1,
        "cached_tokens": 2,
        "cache_write_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 5,
        "requests": 6,
    }


def test_build_responses_body_sends_a_system_only_prompt_as_developer() -> None:
    """Every evaluation prompt in this repo is system-only on this endpoint."""
    assert batch_api.build_responses_body(
        "judge this", "", model="gpt-5.6-luna", temperature=1.0
    ) == {
        "model": "gpt-5.6-luna",
        "input": [{"role": "developer", "content": "judge this"}],
        "temperature": 1.0,
    }


def test_build_responses_body_sends_a_user_only_prompt_as_user() -> None:
    assert batch_api.build_responses_body(
        "", "hello", model="m", temperature=0.0
    ) == {
        "model": "m",
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
    }


def test_build_responses_body_keeps_both_turns_verbatim_in_order() -> None:
    assert batch_api.build_responses_body(
        "system\n", " user \n", model="m", temperature=0.5
    ) == {
        "model": "m",
        "input": [
            {"role": "developer", "content": "system\n"},
            {"role": "user", "content": " user \n"},
        ],
        "temperature": 0.5,
    }


def test_build_responses_body_drops_whitespace_only_turns() -> None:
    assert batch_api.build_responses_body(
        "  \n\t ", "\n", model="m", temperature=1.0
    ) == {"model": "m", "input": [], "temperature": 1.0}


def test_build_responses_body_omits_reasoning_and_ceiling_when_unset() -> None:
    body = batch_api.build_responses_body("s", "", model="m", temperature=1.0)

    assert "reasoning" not in body
    assert "max_output_tokens" not in body


def test_build_responses_body_omits_the_summary_when_only_effort_is_given() -> None:
    assert batch_api.build_responses_body(
        "s", "", model="m", temperature=1.0, reasoning_effort="medium"
    ) == {
        "model": "m",
        "input": [{"role": "developer", "content": "s"}],
        "temperature": 1.0,
        "reasoning": {"effort": "medium"},
    }


def test_build_responses_body_nests_the_summary_under_reasoning() -> None:
    assert batch_api.build_responses_body(
        "s",
        "",
        model="m",
        temperature=1.0,
        reasoning_effort="medium",
        reasoning_summary="auto",
        max_output_tokens=2048,
    ) == {
        "model": "m",
        "input": [{"role": "developer", "content": "s"}],
        "temperature": 1.0,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "max_output_tokens": 2048,
    }


def test_build_responses_body_coerces_the_output_ceiling_to_an_int() -> None:
    body = batch_api.build_responses_body(
        "s", "", model="m", temperature=1.0, max_output_tokens=2048.0
    )

    assert body["max_output_tokens"] == 2048
    assert isinstance(body["max_output_tokens"], int)


def test_build_chat_body_sends_a_system_only_prompt_as_system() -> None:
    assert batch_api.build_chat_body(
        "judge this", "", model="m", temperature=0.7
    ) == {
        "model": "m",
        "messages": [{"role": "system", "content": "judge this"}],
        "temperature": 0.7,
    }


def test_build_chat_body_sends_a_user_only_prompt_as_user() -> None:
    assert batch_api.build_chat_body("", "hello", model="m", temperature=0.0) == {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
    }


def test_build_chat_body_drops_whitespace_only_turns() -> None:
    assert batch_api.build_chat_body(" \t", "  \n ", model="m", temperature=1.0) == {
        "model": "m",
        "messages": [],
        "temperature": 1.0,
    }


def test_build_chat_body_carries_effort_ceiling_and_seed_when_given() -> None:
    assert batch_api.build_chat_body(
        "s",
        "u",
        model="m",
        temperature=1.0,
        reasoning_effort="high",
        max_output_tokens=512.0,
        seed=7,
    ) == {
        "model": "m",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
        "temperature": 1.0,
        "reasoning_effort": "high",
        "max_completion_tokens": 512,
        "seed": 7,
    }


def test_build_chat_body_omits_effort_ceiling_and_seed_when_unset() -> None:
    body = batch_api.build_chat_body("s", "", model="m", temperature=1.0)

    assert body == {
        "model": "m",
        "messages": [{"role": "system", "content": "s"}],
        "temperature": 1.0,
    }


def test_the_two_endpoints_disagree_on_the_system_role_name() -> None:
    responses = batch_api.build_responses_body("s", "u", model="m", temperature=1.0)
    chat = batch_api.build_chat_body("s", "u", model="m", temperature=1.0)

    assert [turn["role"] for turn in responses["input"]] == ["developer", "user"]
    assert [turn["role"] for turn in chat["messages"]] == ["system", "user"]
    # The turn list also lives under a different key on each endpoint.
    assert "messages" not in responses
    assert "input" not in chat


def test_body_builders_registry_maps_each_endpoint_to_its_builder() -> None:
    assert batch_api.BODY_BUILDERS == {
        RESPONSES_ENDPOINT: batch_api.build_responses_body,
        CHAT_COMPLETIONS_ENDPOINT: batch_api.build_chat_body,
    }


@pytest.mark.parametrize(
    "prompt_index,candidate_index,seed,expected",
    [
        (0, 0, None, "snone-p000000-c00"),
        (7, 3, 42, "s42-p000007-c03"),
        (0, 0, 0, "s0-p000000-c00"),
        (123456, 99, 24, "s24-p123456-c99"),
    ],
)
def test_batch_custom_id_renders_seed_prompt_and_candidate(
    prompt_index: int, candidate_index: int, seed, expected: str
) -> None:
    """The id is how results are joined back to prompts, and it is zero-padded."""
    assert batch_api.batch_custom_id(prompt_index, candidate_index, seed) == expected


def test_build_batch_requests_rejects_an_unknown_endpoint() -> None:
    with pytest.raises(ValueError, match="unsupported batch endpoint 'v1/nope'"):
        batch_api.build_batch_requests(
            [{"system_prompt": "s", "user_prompt": ""}],
            endpoint="v1/nope",
            model="m",
            temperature=1.0,
            num_responses=1,
            seed=0,
        )


def test_build_batch_requests_gives_every_candidate_a_distinct_id() -> None:
    requests = batch_api.build_batch_requests(
        [{"system_prompt": "s0"}, {"system_prompt": "s1", "user_prompt": "u1"}],
        endpoint=RESPONSES_ENDPOINT,
        model="m",
        temperature=1.0,
        num_responses=3,
        seed=5,
    )

    ids = [request.custom_id for request in requests]
    assert ids == [
        "s5-p000000-c00",
        "s5-p000000-c01",
        "s5-p000000-c02",
        "s5-p000001-c00",
        "s5-p000001-c01",
        "s5-p000001-c02",
    ]
    assert len(set(ids)) == len(ids)
    assert requests[0].body["input"] == [{"role": "developer", "content": "s0"}]
    assert requests[3].body["input"] == [
        {"role": "developer", "content": "s1"},
        {"role": "user", "content": "u1"},
    ]


def test_build_batch_requests_offsets_the_seed_per_candidate_on_chat() -> None:
    requests = batch_api.build_batch_requests(
        [{"system_prompt": "s", "user_prompt": ""}],
        endpoint=CHAT_COMPLETIONS_ENDPOINT,
        model="m",
        temperature=1.0,
        num_responses=3,
        seed=100,
    )

    assert [request.body["seed"] for request in requests] == [100, 101, 102]
    assert [request.custom_id for request in requests] == [
        "s100-p000000-c00",
        "s100-p000000-c01",
        "s100-p000000-c02",
    ]


def test_build_batch_requests_omits_the_seed_on_the_chat_path_when_none() -> None:
    requests = batch_api.build_batch_requests(
        [{"system_prompt": "s", "user_prompt": ""}],
        endpoint=CHAT_COMPLETIONS_ENDPOINT,
        model="m",
        temperature=1.0,
        num_responses=2,
        seed=None,
    )

    assert all("seed" not in request.body for request in requests)
    assert [request.custom_id for request in requests] == [
        "snone-p000000-c00",
        "snone-p000000-c01",
    ]


def test_build_batch_requests_never_puts_a_seed_in_a_responses_body() -> None:
    """``/v1/responses`` has no ``seed`` field; sending one is a 400."""
    requests = batch_api.build_batch_requests(
        [{"system_prompt": "s", "user_prompt": ""}],
        endpoint=RESPONSES_ENDPOINT,
        model="m",
        temperature=1.0,
        num_responses=2,
        seed=9,
        reasoning_effort="medium",
        reasoning_summary="auto",
        max_output_tokens=64,
    )

    assert all("seed" not in request.body for request in requests)
    assert requests[0].body == {
        "model": "m",
        "input": [{"role": "developer", "content": "s"}],
        "temperature": 1.0,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "max_output_tokens": 64,
    }
    assert all(request.custom_id.startswith("s9-") for request in requests)


def test_build_batch_requests_returns_nothing_for_no_prompts() -> None:
    assert (
        batch_api.build_batch_requests(
            [],
            endpoint=RESPONSES_ENDPOINT,
            model="m",
            temperature=1.0,
            num_responses=2,
            seed=0,
        )
        == []
    )


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-output-key"),
        pytest.param({"output": None}, id="null-output"),
        pytest.param({"output": []}, id="empty-output"),
        pytest.param({"output": ["not-a-mapping"]}, id="non-mapping-item"),
        pytest.param({"output": [{"type": "reasoning"}]}, id="wrong-item-type"),
        pytest.param({"output": [{"type": "message"}]}, id="no-content-key"),
        pytest.param({"output": [{"type": "message", "content": None}]}, id="null"),
        pytest.param(
            {"output": [{"type": "message", "content": ["nope"]}]},
            id="non-mapping-content",
        ),
        pytest.param(
            {"output": [{"type": "message", "content": [{"type": "refusal"}]}]},
            id="wrong-content-type",
        ),
        pytest.param(
            {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": 7}]}
                ]
            },
            id="non-str-text",
        ),
        pytest.param(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "  \n "}],
                    }
                ]
            },
            id="whitespace-only",
        ),
    ],
)
def test_responses_text_returns_none_rather_than_an_empty_string(body) -> None:
    assert batch_api.responses_text(body) is None


def test_responses_text_concatenates_every_output_text_part() -> None:
    body = {
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hm"}]},
            "not-a-mapping",
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": " {\"Rule 1\": "},
                    {"type": "refusal", "text": "IGNORED"},
                    {"type": "output_text", "text": "\"yes\"} \n"},
                    {"type": "output_text", "text": 12345},
                    "not-a-mapping",
                ],
            },
        ]
    }

    assert batch_api.responses_text(body) == '{"Rule 1": "yes"}'


def test_responses_text_does_not_lowercase_the_answer() -> None:
    body = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"Rule 1": "Yes", "Meta-Rule 2": "UNSURE"}',
                    }
                ],
            }
        ]
    }

    assert (
        batch_api.responses_text(body)
        == '{"Rule 1": "Yes", "Meta-Rule 2": "UNSURE"}'
    )


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-choices-key"),
        pytest.param({"choices": None}, id="null-choices"),
        pytest.param({"choices": []}, id="empty-choices"),
        pytest.param({"choices": ["not-a-mapping"]}, id="non-mapping-choice"),
        pytest.param({"choices": [{}]}, id="no-message-key"),
        pytest.param({"choices": [{"message": "not-a-mapping"}]}, id="non-mapping-msg"),
        pytest.param({"choices": [{"message": {}}]}, id="no-content-key"),
        pytest.param({"choices": [{"message": {"content": None}}]}, id="null-content"),
        pytest.param({"choices": [{"message": {"content": 7}}]}, id="non-str-content"),
        pytest.param(
            {"choices": [{"message": {"content": "  \n"}}]}, id="whitespace-only"
        ),
    ],
)
def test_chat_text_returns_none_rather_than_an_empty_string(body) -> None:
    assert batch_api.chat_text(body) is None


def test_chat_text_returns_the_first_choice_that_has_text() -> None:
    body = {
        "choices": [
            "not-a-mapping",
            {"message": None},
            {"message": {"content": "   "}},
            {"message": {"content": "  Rule 1: Yes  "}},
            {"message": {"content": "later"}},
        ]
    }

    assert batch_api.chat_text(body) == "Rule 1: Yes"


@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param({"status": "completed"}, "completed", id="completed"),
        pytest.param(
            {"status": "incomplete", "incomplete_details": {"reason": "max_tokens"}},
            "incomplete:max_tokens",
            id="incomplete-with-reason",
        ),
        pytest.param(
            {"status": "incomplete", "incomplete_details": {}},
            "incomplete",
            id="incomplete-empty-details",
        ),
        pytest.param(
            {"status": "incomplete"}, "incomplete", id="incomplete-no-details"
        ),
        pytest.param(
            {"status": "incomplete", "incomplete_details": {"reason": 7}},
            "incomplete",
            id="incomplete-non-str-reason",
        ),
        pytest.param({"status": 500}, None, id="non-str-status"),
        pytest.param({}, None, id="missing-status"),
    ],
)
def test_responses_finish_reason_distinguishes_truncated_from_brief(
    body, expected
) -> None:
    assert batch_api.responses_finish_reason(body) == expected


@pytest.mark.parametrize(
    "body,expected",
    [
        pytest.param(
            {"choices": [{"finish_reason": "length"}]}, "length", id="present"
        ),
        pytest.param({"choices": [{}]}, None, id="absent"),
        pytest.param({"choices": [{"finish_reason": 7}]}, None, id="non-str"),
        pytest.param({"choices": ["not-a-mapping"]}, None, id="non-mapping-choice"),
        pytest.param({}, None, id="no-choices"),
        pytest.param(
            {"choices": ["x", {}, {"finish_reason": "stop"}]}, "stop", id="skips-junk"
        ),
    ],
)
def test_chat_finish_reason_reads_the_first_choice_that_declares_one(
    body, expected
) -> None:
    assert batch_api.chat_finish_reason(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="no-output"),
        pytest.param({"output": None}, id="null-output"),
        pytest.param({"output": ["not-a-mapping"]}, id="non-mapping-item"),
        pytest.param({"output": [{"type": "message"}]}, id="wrong-item-type"),
        pytest.param({"output": [{"type": "reasoning"}]}, id="no-summary-key"),
        pytest.param({"output": [{"type": "reasoning", "summary": None}]}, id="null"),
        pytest.param(
            {"output": [{"type": "reasoning", "summary": ["not-a-mapping"]}]},
            id="non-mapping-summary",
        ),
        pytest.param(
            {"output": [{"type": "reasoning", "summary": [{"type": "other"}]}]},
            id="wrong-summary-type",
        ),
        pytest.param(
            {
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": 7}],
                    }
                ]
            },
            id="non-str-text",
        ),
    ],
)
def test_reasoning_summaries_is_empty_rather_than_failing_when_absent(body) -> None:
    assert batch_api.reasoning_summaries(body) == []


def test_reasoning_summaries_collects_every_summary_text_in_order() -> None:
    body = {
        "output": [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "first"},
                    {"type": "other", "text": "IGNORED"},
                    {"type": "summary_text", "text": "second"},
                ],
            },
            {"type": "message", "content": []},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "third"}],
            },
        ]
    }

    assert batch_api.reasoning_summaries(body) == ["first", "second", "third"]


def test_extractor_registries_cover_both_endpoints() -> None:
    assert batch_api.TEXT_EXTRACTORS == {
        RESPONSES_ENDPOINT: batch_api.responses_text,
        CHAT_COMPLETIONS_ENDPOINT: batch_api.chat_text,
    }
    assert batch_api.FINISH_REASON_EXTRACTORS == {
        RESPONSES_ENDPOINT: batch_api.responses_finish_reason,
        CHAT_COMPLETIONS_ENDPOINT: batch_api.chat_finish_reason,
    }


def test_outcome_from_record_unwraps_a_successful_responses_result() -> None:
    response_body = {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "Yes"}]},
        ],
        "usage": {"input_tokens": 3},
    }
    record = {
        "custom_id": "s0-p000000-c00",
        "response": {"status_code": 200, "body": response_body},
        "error": None,
    }

    outcome, billable = batch_api.outcome_from_record(record, RESPONSES_ENDPOINT)

    assert outcome == BatchOutcome(
        text="Yes", finish_reason="completed", reasoning_summaries=("why",)
    )
    assert billable is response_body


def test_outcome_from_record_unwraps_a_successful_chat_result() -> None:
    record = {
        "custom_id": "s0-p000000-c00",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {"message": {"content": "no"}, "finish_reason": "stop"},
                ]
            },
        },
    }

    outcome, billable = batch_api.outcome_from_record(record, CHAT_COMPLETIONS_ENDPOINT)

    assert outcome == BatchOutcome(text="no", finish_reason="stop")
    assert billable is not None


@pytest.mark.parametrize("status", ["failed", "cancelled", "queued"])
def test_outcome_from_record_refuses_a_responses_body_that_did_not_answer_and_bills_it(status) -> None:
    response_body = {
        "status": status,
        "error": {"code": "server_error", "message": "boom"},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "yes"}]}],
        "usage": {"input_tokens": 3},
    }
    record = {"custom_id": "s0-p000000-c00", "response": {"status_code": 200, "body": response_body}}

    outcome, billable = batch_api.outcome_from_record(record, RESPONSES_ENDPOINT)

    assert outcome == BatchOutcome(error=f"the response's status is {status!r} (server_error: boom)")
    assert billable is response_body


def test_outcome_from_record_reports_a_record_level_error_and_bills_nothing() -> None:
    record = {
        "custom_id": "s0-p000000-c00",
        "error": {"message": "rate_limit_exceeded"},
        "response": {"status_code": 200, "body": {"usage": {"input_tokens": 999}}},
    }

    outcome, billable = batch_api.outcome_from_record(record, RESPONSES_ENDPOINT)

    assert outcome.text is None
    assert "rate_limit_exceeded" in outcome.error
    assert billable is None


def test_outcome_from_record_treats_a_non_200_status_as_a_failure() -> None:
    record = {
        "custom_id": "s0-p000000-c00",
        "response": {"status_code": "500", "body": {"usage": {"input_tokens": 999}}},
    }

    outcome, billable = batch_api.outcome_from_record(record, RESPONSES_ENDPOINT)

    assert outcome == BatchOutcome(error="status 500")
    assert billable is None


def test_outcome_from_record_accepts_a_result_with_no_status_code() -> None:
    """OpenRouter returns inline results without a status_code on success."""
    outcome, billable = batch_api.outcome_from_record(
        {"custom_id": "s0-p000000-c00", "response": {}}, RESPONSES_ENDPOINT
    )

    assert outcome == BatchOutcome()
    assert billable == {}
    assert billable is not None


def test_sha256_bytes_matches_the_standard_digest() -> None:
    assert batch_api.sha256_bytes(b"") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert batch_api.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_request_line_serializes_exactly_the_batch_file_bytes() -> None:
    """Key order, separators and the trailing newline are all pinned."""
    request = BatchRequest(
        custom_id="s0-p000000-c00",
        body={"model": "m", "input": [], "temperature": 1.0},
    )

    assert batch_api.request_line(request, RESPONSES_ENDPOINT) == (
        b'{"custom_id":"s0-p000000-c00","method":"POST","url":"/v1/responses",'
        b'"body":{"model":"m","input":[],"temperature":1.0}}\n'
    )


def test_request_line_writes_non_ascii_prompts_as_utf8_not_escapes() -> None:
    """``ensure_ascii=False`` is part of the frozen serialization."""
    request = BatchRequest(custom_id="c", body={"input": "café"})

    line = batch_api.request_line(request, RESPONSES_ENDPOINT)

    assert "café".encode("utf-8") in line
    assert b"\\u00e9" not in line


@pytest.mark.parametrize("case", json.loads(LUNA_REQUEST_LINES.read_text(encoding="utf-8"))["cases"],
                         ids=lambda case: case["name"])
def test_request_line_is_byte_identical_to_the_published_ruleset_size_campaign_s(case: dict) -> None:
    """The anti-drift pin against the published Luna ruleset-size campaign."""
    fixture = json.loads(LUNA_REQUEST_LINES.read_text(encoding="utf-8"))
    assert fixture["endpoint"] == RESPONSES_ENDPOINT
    body = batch_api.build_responses_body(
        case["prompt"],
        "",
        model=fixture["model"],
        temperature=fixture["temperature"],
        reasoning_effort=fixture["reasoning_effort"],
        reasoning_summary=fixture["reasoning_summary"],
        max_output_tokens=None,
    )
    shared_line = batch_api.request_line(
        BatchRequest(custom_id=case["custom_id"], body=body),
        fixture["endpoint"],
    )

    assert shared_line == case["line"].encode("utf-8")
    assert shared_line.endswith(b"\n") and json.loads(shared_line)["custom_id"] == case["custom_id"]


def test_fingerprint_is_stable_for_the_same_requests() -> None:
    requests = [
        BatchRequest(custom_id="a", body={"model": "m", "input": []}),
        BatchRequest(custom_id="b", body={"model": "m", "input": []}),
    ]
    twin = [
        BatchRequest(custom_id="a", body={"model": "m", "input": []}),
        BatchRequest(custom_id="b", body={"model": "m", "input": []}),
    ]

    digest = batch_api.fingerprint(requests, RESPONSES_ENDPOINT)

    assert digest == batch_api.fingerprint(twin, RESPONSES_ENDPOINT)
    assert len(digest) == 32


def test_fingerprint_changes_when_a_body_the_endpoint_or_the_order_changes() -> None:
    base = [
        BatchRequest(custom_id="a", body={"model": "m", "temperature": 1.0}),
        BatchRequest(custom_id="b", body={"model": "m", "temperature": 1.0}),
    ]
    changed_body = [
        BatchRequest(custom_id="a", body={"model": "m", "temperature": 0.0}),
        base[1],
    ]
    reordered = [base[1], base[0]]

    digest = batch_api.fingerprint(base, RESPONSES_ENDPOINT)

    assert batch_api.fingerprint(changed_body, RESPONSES_ENDPOINT) != digest
    assert batch_api.fingerprint(base, CHAT_COMPLETIONS_ENDPOINT) != digest
    assert batch_api.fingerprint(reordered, RESPONSES_ENDPOINT) != digest


def test_write_json_atomic_writes_overwrites_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "state.json"

    batch_api.write_json_atomic(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").startswith('{\n  "a": 1')

    batch_api.write_json_atomic(path, {"batch_id": "batch_abc"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"batch_id": "batch_abc"}

    assert list(path.parent.glob("*.tmp")) == []
    assert [entry.name for entry in path.parent.iterdir()] == ["state.json"]


def test_require_spend_confirmation_refuses_when_the_env_var_is_unset(
    monkeypatch,
) -> None:
    with pytest.raises(BatchSpendNotConfirmed) as excinfo:
        batch_api.require_spend_confirmation()

    assert "--yes-spend" in str(excinfo.value)
    assert "spends money" in str(excinfo.value)


def test_a_spend_confirmed_in_a_context_holds_there_alone_and_leaves_the_environment_alone(monkeypatch) -> None:
    import threading

    elsewhere = []

    def other_thread() -> None:
        try:
            batch_api.require_spend_confirmation()
        except BatchSpendNotConfirmed:
            elsewhere.append("refused")

    with batch_api.spend_confirmed():
        assert batch_api.require_spend_confirmation() is None
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
    assert elsewhere == ["refused"]
    with pytest.raises(BatchSpendNotConfirmed, match="--yes-spend"):
        batch_api.require_spend_confirmation()


def test_write_json_once_writes_a_state_whole_and_never_replaces_one(tmp_path) -> None:
    path = tmp_path / "work" / "state.json"
    assert batch_api.write_json_once(path, {"parts": {}}) is True
    assert batch_api.write_json_once(path, {"parts": {"replaced": True}}) is False
    assert json.loads(path.read_text()) == {"parts": {}}
    assert sorted(item.name for item in path.parent.iterdir()) == ["state.json"]


def test_write_json_once_without_hard_links_creates_exclusively_and_still_never_replaces(tmp_path, monkeypatch) -> None:
    def no_links(source, destination):
        raise PermissionError("hard links not supported")

    monkeypatch.setattr(batch_api.os, "link", no_links)
    path = tmp_path / "state.json"
    assert batch_api.write_json_once(path, {"parts": {}}) is True
    assert batch_api.write_json_once(path, {"parts": {"replaced": True}}) is False
    assert json.loads(path.read_text()) == {"parts": {}}
    assert sorted(item.name for item in tmp_path.iterdir()) == ["state.json"]


def test_private_temporary_is_a_new_name_beside_the_file_that_no_reader_globs(tmp_path) -> None:
    destination = tmp_path / "outputs" / "part-0000.jsonl"
    first, second = batch_api.private_temporary(destination), batch_api.private_temporary(destination)
    assert first != second and first.parent == second.parent == destination.parent
    assert first.name.startswith(f"part-0000.jsonl.{os.getpid()}.") and first.suffix == ".tmp"
    assert not first.match("part-*.jsonl") and not second.match("part-*.jsonl")


def test_write_bytes_atomic_replaces_the_file_in_one_rename(tmp_path) -> None:
    path = tmp_path / "part-0000.jsonl"
    path.write_bytes(b"old\n")
    with path.open("rb") as reader:
        batch_api.write_bytes_atomic(path, b"new\n")
        assert reader.read() == b"old\n"
    left = sorted(item.name for item in tmp_path.iterdir() if not is_silly_rename(item))
    assert path.read_bytes() == b"new\n" and left == [path.name]


@pytest.mark.parametrize("ids, prefix", [(["nsabc-p000000-c00", "nsabc-p000001-c00"], "nsabc"),
                                         (["s5-p000000-c00"], "s5"), (["a-p000000-c00", "b-p000001-c00"], None),
                                         ([], None)])
def test_custom_id_prefix_is_the_inverse_of_batch_custom_id_s_prefix(ids, prefix) -> None:
    assert batch_api.custom_id_prefix(ids) == prefix
    assert batch_api.custom_id_prefix([batch_api.batch_custom_id(3, 1, None, "nsq")]) == "nsq"


@pytest.mark.parametrize("reason, truncated", [("max_tokens", True), ("MAX_TOKENS", True), ("length", True),
                                               ("incomplete:max_output_tokens", True), ("end_turn", False),
                                               ("refusal:cyber", False), ("stop", False), (None, False)])
def test_every_provider_s_spelling_of_the_output_ceiling_reads_as_truncated(reason, truncated) -> None:
    assert batch_api.is_truncated(reason) is truncated


def test_write_reasoning_summaries_writes_nothing_when_there_are_none(
    tmp_path: Path,
) -> None:
    """``summary="auto"`` often returns nothing, which is not a failure."""
    outcomes = {
        "s0-p000000-c00": BatchOutcome(text="yes"),
        "s0-p000000-c01": BatchOutcome(text="no", reasoning_summaries=()),
    }

    assert batch_api.write_reasoning_summaries(tmp_path, outcomes) is None
    assert list(tmp_path.iterdir()) == []


def test_write_reasoning_summaries_writes_sorted_jsonl_beside_the_batch(
    tmp_path: Path,
) -> None:
    outcomes = {
        "s0-p000001-c00": BatchOutcome(text="no", reasoning_summaries=("second",)),
        "s0-p000000-c00": BatchOutcome(
            text="yes", reasoning_summaries=("first", "also")
        ),
        "s0-p000002-c00": BatchOutcome(text="unsure"),
    }

    path = batch_api.write_reasoning_summaries(tmp_path, outcomes)

    assert path == tmp_path / "reasoning_summaries.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"custom_id": "s0-p000000-c00", "summaries": ["first", "also"]},
        {"custom_id": "s0-p000001-c00", "summaries": ["second"]},
    ]


def test_reshape_outcomes_rebuilds_the_prompts_by_candidates_grid() -> None:
    """The caller pairs ``grid[j]`` with ``prompts[pending_idx[j]]``."""
    outcomes = {
        batch_api.batch_custom_id(0, 0, 5): BatchOutcome(text="a"),
        batch_api.batch_custom_id(0, 1, 5): BatchOutcome(text="b"),
        batch_api.batch_custom_id(1, 0, 5): BatchOutcome(text="c"),
        batch_api.batch_custom_id(1, 1, 5): BatchOutcome(text="d"),
    }

    grid = batch_api.reshape_outcomes(
        outcomes, num_prompts=2, num_responses=2, seed=5
    )

    assert [[outcome.text for outcome in row] for row in grid] == [
        ["a", "b"],
        ["c", "d"],
    ]


def test_reshape_outcomes_looks_cells_up_by_id_not_by_arrival_order() -> None:
    """Batch results come back in arbitrary order; positions must not be assumed."""
    shuffled = {
        batch_api.batch_custom_id(1, 1, None): BatchOutcome(text="p1c1"),
        batch_api.batch_custom_id(0, 1, None): BatchOutcome(text="p0c1"),
        batch_api.batch_custom_id(1, 0, None): BatchOutcome(text="p1c0"),
        batch_api.batch_custom_id(0, 0, None): BatchOutcome(text="p0c0"),
    }

    grid = batch_api.reshape_outcomes(
        shuffled, num_prompts=2, num_responses=2, seed=None
    )

    assert [[outcome.text for outcome in row] for row in grid] == [
        ["p0c0", "p0c1"],
        ["p1c0", "p1c1"],
    ]


def test_reshape_outcomes_fills_a_missing_result_with_an_empty_outcome() -> None:
    outcomes = {batch_api.batch_custom_id(0, 0, 0): BatchOutcome(text="only")}

    grid = batch_api.reshape_outcomes(outcomes, num_prompts=2, num_responses=2, seed=0)

    assert grid[0][0].text == "only"
    assert grid[0][1] == BatchOutcome()
    assert grid[1] == [BatchOutcome(), BatchOutcome()]
    assert all(len(row) == 2 for row in grid)


def test_log_batch_completion_warns_that_truncated_text_is_partial(caplog) -> None:
    grid = [
        [
            BatchOutcome(text="yes", finish_reason="completed"),
            BatchOutcome(
                text='{"Rule 1"', finish_reason="incomplete:max_output_tokens"
            ),
        ],
        [
            BatchOutcome(text=None, finish_reason=None),
            BatchOutcome(text="no", finish_reason="incomplete"),
        ],
    ]

    with caplog.at_level(logging.INFO, logger="ruleforge.judges.batch_api"):
        batch_api.log_batch_completion(grid, "openai")

    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "2 openai response(s) stopped before finishing" in warnings[0].getMessage()
    assert "openai batch complete: 3/4 returned text" in caplog.text


def test_log_batch_completion_stays_quiet_when_nothing_was_truncated(caplog) -> None:
    grid = [[BatchOutcome(text="yes", finish_reason="completed")], [BatchOutcome()]]

    with caplog.at_level(logging.INFO, logger="ruleforge.judges.batch_api"):
        batch_api.log_batch_completion(grid, "openai")

    assert [rec for rec in caplog.records if rec.levelno >= logging.WARNING] == []
    assert "openai batch complete: 1/2 returned text" in caplog.text


@pytest.mark.parametrize(
    "name,expected",
    [
        ("responses", RESPONSES_ENDPOINT),
        ("chat_completions", CHAT_COMPLETIONS_ENDPOINT),
        (RESPONSES_ENDPOINT, RESPONSES_ENDPOINT),
        (CHAT_COMPLETIONS_ENDPOINT, CHAT_COMPLETIONS_ENDPOINT),
    ],
)
def test_resolve_endpoint_accepts_both_an_alias_and_a_url_path(
    name: str, expected: str
) -> None:
    """Configs name the endpoint; programmatic callers may pass the path."""
    assert batch_api.resolve_endpoint(name) == expected


def test_resolve_endpoint_rejects_an_unknown_name_and_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError) as excinfo:
        batch_api.resolve_endpoint("/v1/completions")

    message = str(excinfo.value)
    assert "unknown batch endpoint '/v1/completions'" in message
    assert "responses" in message and "chat_completions" in message


def test_terminal_statuses_are_the_four_both_providers_publish() -> None:
    assert batch_api.TERMINAL_STATUSES == {
        "completed",
        "failed",
        "expired",
        "cancelled",
    }


def test_a_second_process_is_refused_the_creation_claim(tmp_path):
    with claim_for_creation(tmp_path):
        with pytest.raises(BatchAlreadyClaimed, match="creating.lock"):
            with claim_for_creation(tmp_path):
                pass

    with claim_for_creation(tmp_path):
        pass


def test_the_claim_is_released_even_when_the_body_raises(tmp_path):
    with pytest.raises(RuntimeError):
        with claim_for_creation(tmp_path):
            raise RuntimeError("upload died")
    assert not (tmp_path / "creating.lock").exists()


def test_asking_for_a_summary_the_endpoint_cannot_produce_is_announced(caplog):
    with caplog.at_level(logging.WARNING):
        build_batch_requests(
            [{"system_prompt": "s", "user_prompt": ""}],
            endpoint=CHAT_COMPLETIONS_ENDPOINT,
            model="m",
            temperature=1.0,
            num_responses=1,
            seed=None,
            reasoning_summary="auto",
        )
    assert "no reasoning summaries will be saved" in caplog.text


def test_no_such_warning_when_the_endpoint_can_carry_it(caplog):
    with caplog.at_level(logging.WARNING):
        build_batch_requests(
            [{"system_prompt": "s", "user_prompt": ""}],
            endpoint=RESPONSES_ENDPOINT,
            model="m",
            temperature=1.0,
            num_responses=1,
            seed=None,
            reasoning_summary="auto",
        )
    assert "reasoning summaries" not in caplog.text


def test_the_refusal_names_both_ways_to_confirm():
    with pytest.raises(BatchSpendNotConfirmed) as caught:
        batch_api.require_spend_confirmation()
    assert "--yes-spend" in str(caught.value) and "+confirm_spend=true" in str(caught.value)


@pytest.mark.parametrize(("config", "confirmed"), [({}, False), ({"confirm_spend": False}, False),
                                                   ({"confirm_spend": True}, True)])
def test_a_spend_setting_is_read_as_a_boolean(config, confirmed):
    assert batch_api.config_confirms_spend(config) is confirmed


@pytest.mark.parametrize("value", ["false", "0", "no", "true", 1, None])
def test_a_spend_setting_that_is_not_a_boolean_never_confirms(value):
    with pytest.raises(batch_api.BatchSpendNotConfirmed, match="confirm_spend is true or false"):
        batch_api.config_confirms_spend({"confirm_spend": value})
