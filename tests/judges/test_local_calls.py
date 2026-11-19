from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from ruleforge.judges import models


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LLM:
    """An engine answering each prompt with ``texts`` (one candidate each), or with nothing at all."""

    def __init__(self, texts=None):
        self.texts, self.calls = texts, []

    def generate(self, prompts, sampling_params=None, **kwargs):
        self.calls.append((list(prompts), sampling_params, kwargs))
        if self.texts is None:
            return []
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[1], finish_reason="stop")])
                for text in self.texts]


class _Tokenizer:
    """A chat template that writes each turn as ``role:content`` and appends the template's keyword arguments."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, *, tokenize, add_generation_prompt, **kwargs):
        self.calls.append(kwargs)
        return "|".join(f"{turn['role']}:{turn['content']}" for turn in chat) + "".join(
            f"|{key}={value}" for key, value in sorted(kwargs.items()))

    def encode(self, text, add_special_tokens=True):
        """Each character's code point, after a beginning-of-text id (0) when special tokens are added."""
        return [0] * add_special_tokens + [ord(char) for char in text]


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=_SamplingParams))


def test_an_engine_that_returns_nothing_gives_each_prompt_no_candidates(fake_vllm):
    llm = _LLM()
    outputs = models._generate_with_vllm(llm, ["a", "b"], temperature=1.0, max_new_tokens=8, num_responses=1,
                                         seed=None)
    assert outputs == [[], []]
    assert llm.calls[0][0] == ["a", "b"]


def test_a_gpt_oss_prompt_carries_its_reasoning_level_to_the_chat_template():
    tokenizer = _Tokenizer()
    text = models.format_prompt_for_generation(tokenizer, {"system_prompt": "s", "user_prompt": "u"},
                                               gpt_oss_reasoning_level="high")
    assert text == "system:s|user:u|reasoning_effort=high"
    assert tokenizer.calls == [{"reasoning_effort": "high"}]


def test_the_local_call_returns_the_stripped_completions_of_the_detailed_call(fake_vllm):
    llm, tokenizer = _LLM(["<think>why</think> yes", "<think>so</think>no "]), _Tokenizer()
    batch = [{"system_prompt": "s1", "user_prompt": ""}, {"system_prompt": "s2", "user_prompt": "u2"}]
    detailed = models.call_local_model_with_details(llm, tokenizer, batch, temperature=1.0, max_new_tokens=8,
                                                    seed=[1, 2], strip_thinking_model="qwen3")
    assert [[c["raw_response"] for c in prompt] for prompt in detailed] == [["<think>why</think> yes"],
                                                                         ["<think>so</think>no "]]
    assert models.call_local_model(llm, tokenizer, batch, temperature=1.0, max_new_tokens=8, seed=[1, 2],
                                   strip_thinking_model="qwen3") == [["yes"], ["no"]]
    assert llm.calls[0][0] == llm.calls[1][0] == [{"prompt_token_ids": [ord(char) for char in text]}
                                                  for text in ("system:s1|user:", "system:s2|user:u2")]
    assert [params.kwargs["seed"] for params in llm.calls[1][1]] == [1, 2]


def test_an_empty_prompt_batch_is_refused_before_anything_is_rendered():
    with pytest.raises(ValueError, match="prompt_batch must not be empty"):
        models.call_local_model_with_details(_LLM(), _Tokenizer(), [])
