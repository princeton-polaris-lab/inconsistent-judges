from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ruleforge.judges.models as models
from ruleforge.judges.models import (
    _MODEL_STRIP_FN,
    _extract_text_and_strip_status,
    _strip_gemma4_thinking,
    _strip_harmony_thinking,
    _strip_muse_glimmer_thinking,
    _strip_think_tags,
    harmony_encoding,
)
from tests.identity import published as pub


def test_think_tags_are_removed_and_the_answer_survives():
    assert _strip_think_tags({"text": "<think>deliberating</think>yes"}) == "yes"


def test_everything_before_an_orphaned_closing_tag_is_dropped():
    """A model that emits `</think>` without an opener is the common failure."""
    assert _strip_think_tags({"text": "reasoning with no opener</think>no"}) == "no"


def test_the_orphan_rule_can_be_turned_off():
    assert (
        _strip_think_tags({"text": "prefix</think>no"}, strip_before_orphaned_closing=False)
        == "prefix</think>no"
    )


def test_the_last_orphaned_closing_tag_wins():
    assert _strip_think_tags({"text": "a</think>b</think>unsure"}) == "unsure"


def test_a_missing_text_field_raises_rather_than_returning_nothing():
    with pytest.raises(ValueError, match="Missing 'text' field"):
        _strip_think_tags({"token_ids": [1, 2]})
    with pytest.raises(ValueError, match="Missing 'text' field"):
        _strip_think_tags({"text": None})


def test_thinking_with_no_answer_after_it_raises():
    with pytest.raises(ValueError, match="No content remaining"):
        _strip_think_tags({"text": "<think>all reasoning, no verdict</think>   "})


def test_gemma4_thought_blocks_are_removed():
    assert (
        _strip_gemma4_thinking({"text": "<|channel>thought reasoning<channel|>yes"})
        == "yes"
    )


def test_gemma4_drops_everything_before_an_orphaned_closing_marker():
    """Gemma 4 omits the opening marker often enough that this is the norm."""
    assert _strip_gemma4_thinking({"text": "reasoning<channel|>no"}) == "no"


def test_gemma4_takes_the_last_closing_marker():
    assert _strip_gemma4_thinking({"text": "a<channel|>b<channel|>unsure"}) == "unsure"


def test_gemma4_leaves_a_clean_answer_alone():
    assert _strip_gemma4_thinking({"text": "  yes  "}) == "yes"


def test_gemma4_raises_on_a_missing_text_field():
    with pytest.raises(ValueError, match="Missing 'text' field"):
        _strip_gemma4_thinking({"text": 17})


def test_gemma4_raises_when_nothing_remains():
    with pytest.raises(ValueError, match="No content remaining"):
        _strip_gemma4_thinking({"text": "<|channel>thought only<channel|>  "})


GLIMMER_CASES = json.loads((Path(__file__).resolve().parent / "fixtures" / "judges" / "glimmer_framing_cases.json")
                           .read_text(encoding="utf-8"))["cases"]
#: Muse Glimmer's framing, as its outputs carry it (vLLM keeps the special tokens and drops the <|eot|> it stops on).
SELF, USER, EOM, EOT, START = "to=self<|message|>", "to=user<|message|>", "<|eom|>", "<|eot|>", "<|start|>assistant"
#: The markers no extracted answer may hold (``to=self`` aside: an answer can say it in its body).
GLIMMER_MARKERS = ("to=user", "<|message|>", "<|start|>", "<|eom|>", "<|eot|>")
#: Glimmer's framing tokens as its vocabulary numbers them.
IDS = models.GLIMMER_SPECIAL_IDS


def _ids(text, *, stop=True):
    ids = [IDS[token] if token in IDS else 7 for token in re.split(r"(<\|start\|>|<\|message\|>|<\|eom\|>|<\|eot\|>)", text)
           if token]
    return ids + [IDS["<|eot|>"]] * stop


def test_the_glimmer_cases_read_answers_and_empty_answers_alike():
    answers = [case["answer"] for case in GLIMMER_CASES]
    assert len(answers) == 30 and answers.count("") == 16


@pytest.mark.parametrize("case", GLIMMER_CASES, ids=lambda case: case["case"])
def test_glimmer_reads_the_body_of_its_last_answer_message_or_nothing(case, caplog):
    text, answer = case["text"], case["answer"]
    with caplog.at_level("WARNING"):
        assert _strip_muse_glimmer_thinking({"text": text}) == answer
        assert _extract_text_and_strip_status({"text": text}, "muse_glimmer")[0] == answer
        assert _extract_text_and_strip_status({"text": text, "token_ids": _ids(text)}, "muse_glimmer")[0] == answer
        assert _strip_muse_glimmer_thinking({"text": text, "token_ids": tuple(_ids(text, stop=False))}) == answer
    assert "Failed to strip" not in caplog.text
    assert not any(marker in answer for marker in GLIMMER_MARKERS)


@pytest.mark.parametrize("framing, written, rest, stopped", [
    # a header the reasoning wrote as ordinary tokens, then a real answer: the text cannot tell them apart
    (f" {SELF}I will write ", f"{START} {USER}", f"yes{EOM}{START} {USER}no", True),
    # an <|eom|> the answer wrote as text
    (f" {SELF}r{EOM}{START} {USER}yes", EOM, "", True),
    # an <|eot|> the answer wrote as text, and the generation cut off (no stop id)
    (f" {SELF}r{EOM}{START} {USER}yes", EOT, "", False),
])
def test_glimmer_reads_nothing_when_its_framing_in_the_text_is_not_what_it_emitted(framing, written, rest, stopped):
    text = framing + written + rest
    ids = _ids(framing, stop=False) + [7, 7, 7] + _ids(rest, stop=stopped)
    assert _strip_muse_glimmer_thinking({"text": text}) != ""
    assert _strip_muse_glimmer_thinking({"text": text, "token_ids": ids}) == ""


@pytest.mark.parametrize("output", [{"token_ids": [1]}, {"text": None}, {"text": 17},
                                    {"text": f" {USER}no", "token_ids": 17}, {"text": f" {USER}no", "token_ids": "ab"},
                                    {"text": f" {USER}no", "token_ids": [IDS["<|message|>"], None]}])
def test_glimmer_without_text_or_with_ids_that_are_not_ids_is_the_empty_answer(output):
    assert _strip_muse_glimmer_thinking(output) == ""


def test_glimmer_s_answer_span_is_its_last_answer_body_in_the_text():
    text = f" {SELF}r{EOM}{START} {USER}yes{EOM}{START} {USER} no "
    start, end = models.muse_glimmer_answer_span(text)
    assert text[start:end] == " no " and text[:start].endswith(USER)
    assert models.muse_glimmer_answer_span(f" {SELF}r{EOM}") is None


def test_glimmer_s_empty_answer_is_invalid_under_the_label_reader():
    from ruleforge.evaluate.validation import ComponentSchema
    from ruleforge.judges.backend import GenerationOutcome

    rules = [{"id": "Rule 1", "text": "Be kind."}]
    raw = f" {SELF}Final answer:\n\nyes{EOM}"
    completion = _extract_text_and_strip_status({"text": raw}, "muse_glimmer")[0]
    assert completion == ""
    schema = ComponentSchema.from_rules(rules, prompt_style="naive", forced_json_requirement="complete_rule_map",
                                        evaluation_mode="both")
    verdict = schema.classify("full_ruleset", GenerationOutcome(text=completion or None, raw_response=raw,
                                                                 finish_reason="stop"))
    assert not verdict.valid and verdict.judgment is None


@pytest.mark.cluster
def test_glimmer_s_framing_ids_are_the_weights_directory_s():
    tokenizer = Path(os.environ.get("RULEFORGE_MODEL_ROOT", "/nonexistent")) / "Muse-Glimmer-30B" / "tokenizer.json"
    if not tokenizer.is_file():
        pytest.skip("needs Muse-Glimmer-30B under RULEFORGE_MODEL_ROOT")
    added = {token["content"]: (token["id"], token["special"])
             for token in json.loads(tokenizer.read_text(encoding="utf-8"))["added_tokens"]}
    assert {token: added[token] for token in IDS} == {token: (token_id, True) for token, token_id in IDS.items()}


def published_root(root: str) -> Path:
    """``<artifact root>/sampling/<root>``; the test skipped without an artifact root."""
    from ruleforge.core.errors import ArtifactRootError
    from ruleforge.env import artifact_root

    try:
        return artifact_root() / "sampling" / root
    except ArtifactRootError as error:
        pytest.skip(str(error))


def raw_key(row, component_id: str) -> tuple:
    return row["scenario_id"], row["semantic_attempt_id"], row["request_block_id"], component_id


@pytest.mark.cluster
@pytest.mark.parametrize(("root", "index"), pub.GLIMMER_SHARDS, ids=pub.GLIMMER_SHARD_IDS)
def test_glimmer_s_strip_reads_every_recorded_answer_as_the_run_recorded_it(root, index):
    from ruleforge.evaluate.ledger import raw_output_sha256
    from ruleforge.evaluate.validation import EMPTY_RESPONSE

    read, unread, misread, id_only = 0, [], [], []
    for condition, draw in pub.shard_draws(published_root(root), index, pub.GLIMMER_ROOTS[root]):
        where = f"{condition.condition_id}/{draw.name}"
        components = {raw_key(row, component["component_id"]): component
                      for row in pub.ledger_rows(draw / "attempts.jsonl") for component in row["components"]
                      if component["raw_output_sha256"] is not None}
        seen = set()
        for raw in pub.ledger_rows(draw / "raw_outputs.jsonl"):
            key = raw_key(raw, raw["component_id"])
            component = components.get(key)
            if (component is None or key in seen or raw_output_sha256(raw["raw_response"]) != raw["raw_output_sha256"]
                    or raw["raw_output_sha256"] != component["raw_output_sha256"]):
                unread.append(f"{where}: {key}")
                continue
            seen.add(key)
            read += 1
            answer = _extract_text_and_strip_status({"text": raw["raw_response"]}, "muse_glimmer")[0] or None
            if answer != raw["completion"]:
                refused = raw["completion"] is None and component["failure_reason"] == EMPTY_RESPONSE
                (id_only if refused else misread).append(f"{where}: {key}")
        unread += [f"{where}: no raw row for {key}" for key in sorted(components.keys() - seen)]
    assert read
    assert unread == []
    assert misread == []
    assert id_only == []


class _Encoding:
    """A stand-in harmony encoding: parses token ids into the messages it is given."""

    def __init__(self, messages=None, error=None):
        self.messages, self.error, self.calls = messages or [], error, []

    def parse_messages_from_completion_tokens(self, token_ids, role):
        self.calls.append((token_ids, role))
        if self.error is not None:
            raise self.error
        return self.messages


def _message(channel, text):
    return SimpleNamespace(channel=channel, content=[SimpleNamespace(text=text)])


@pytest.fixture
def harmony(monkeypatch):
    """A stand-in openai_harmony module, nothing loaded yet; ``loads`` counts the vocab loads."""
    state = SimpleNamespace(loads=0, encoding=_Encoding(), error=None)

    def load_harmony_encoding(name):
        assert name == "HARMONY_GPT_OSS"
        state.loads += 1
        if state.error is not None:
            raise state.error
        return state.encoding

    module = SimpleNamespace(HarmonyEncodingName=SimpleNamespace(HARMONY_GPT_OSS="HARMONY_GPT_OSS"),
                             Role=SimpleNamespace(ASSISTANT="assistant"), load_harmony_encoding=load_harmony_encoding)
    monkeypatch.setattr(models, "_openai_harmony", module)
    monkeypatch.setattr(models, "_harmony_loaded", None)
    return state


def test_importing_the_models_module_loads_no_harmony_vocab():
    """Loading the vocab downloads it into $TIKTOKEN_RS_CACHE_DIR when that cache is empty."""
    probe = (
        "import sys, types; harmony = types.ModuleType('openai_harmony'); "
        "harmony.HarmonyEncodingName = types.SimpleNamespace(HARMONY_GPT_OSS='HARMONY_GPT_OSS'); "
        "harmony.load_harmony_encoding = lambda name: sys.exit('the vocab was loaded at import'); "
        "sys.modules['openai_harmony'] = harmony; import ruleforge.judges.models as m; "
        "print(m._openai_harmony is harmony, m._harmony_loaded)"
    )
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True None"


def test_the_harmony_vocab_is_loaded_once_on_first_use(harmony):
    assert harmony.loads == 0
    assert harmony_encoding() is harmony.encoding
    assert harmony_encoding() is harmony.encoding
    assert harmony.loads == 1


def test_a_vocab_that_cannot_be_loaded_is_raised_not_read_as_raw_text(harmony):
    """A RuntimeError (openai_harmony.HarmonyError is one) is not a stripper's ValueError."""
    harmony.error = RuntimeError("failed to download or load vocab file")
    with pytest.raises(RuntimeError, match="vocab file"):
        _extract_text_and_strip_status({"text": "yes", "token_ids": [1]}, "openai_gptoss")
    harmony.error = None
    assert harmony_encoding() is harmony.encoding
    assert harmony.loads == 2


def test_without_openai_harmony_the_gpt_oss_stripper_leaves_no_answer(monkeypatch, caplog):
    monkeypatch.setattr(models, "_openai_harmony", None)
    with pytest.raises(NotImplementedError, match="openai_harmony is not installed"):
        harmony_encoding()
    with caplog.at_level("WARNING"):
        assert _extract_text_and_strip_status({"text": " yes ", "token_ids": [1]}, "openai_gptoss") == ("", True)
    assert "openai_harmony is not installed" in caplog.text


def test_the_last_final_channel_message_is_the_answer(harmony):
    harmony.encoding.messages = [_message("analysis", "thinking"), _message("final", "no"),
                                 _message("commentary", "aside"), _message("final", "yes")]
    assert _strip_harmony_thinking({"token_ids": [5, 6]}) == "yes"
    assert harmony.encoding.calls == [([5, 6], "assistant")]
    assert _MODEL_STRIP_FN["openai_gptoss"] is _strip_harmony_thinking


@pytest.mark.parametrize("output", [{"text": "yes"}, {"token_ids": []}, {"token_ids": (1, 2)}])
def test_the_gpt_oss_stripper_needs_token_ids(harmony, output):
    with pytest.raises(ValueError, match="Missing token_ids"):
        _strip_harmony_thinking(output)


def test_token_ids_the_encoding_cannot_parse_raise_a_value_error(harmony):
    harmony.encoding.error = KeyError("unknown token")
    with pytest.raises(ValueError, match="Unable to decode assistant messages") as caught:
        _strip_harmony_thinking({"token_ids": [1]})
    assert isinstance(caught.value.__cause__, KeyError)


def test_no_final_channel_message_raises(harmony):
    harmony.encoding.messages = [_message("analysis", "thinking only")]
    with pytest.raises(ValueError, match="No final assistant messages"):
        _strip_harmony_thinking({"token_ids": [1]})


def test_every_registered_stripper_is_callable_by_its_config_name():
    assert set(_MODEL_STRIP_FN) == {
        "openai_gptoss",
        "qwen3",
        "gemma4",
        "olmo3",
        "muse_glimmer",
    }


def test_olmo3_shares_the_think_tag_stripper_and_survives_an_orphaned_close():
    """Olmo-3 Think can emit its reasoning, then ``</think>`` and the label, with no opening tag."""
    assert _MODEL_STRIP_FN["olmo3"] is _MODEL_STRIP_FN["qwen3"]
    assert _extract_text_and_strip_status({"text": "long reasoning</think>\n\nno"}, "olmo3")[0] == "no"
    assert _extract_text_and_strip_status({"text": "<think>r</think>unsure"}, "olmo3")[0] == "unsure"


def test_an_unknown_stripper_name_warns_and_leaves_no_answer(caplog):
    with caplog.at_level("WARNING"):
        assert _extract_text_and_strip_status({"text": " yes "}, "not-a-real-model") == ("", True)
    assert "No thinking-token stripper" in caplog.text


def test_a_failing_stripper_warns_and_leaves_no_answer_rather_than_raising(caplog):
    with caplog.at_level("WARNING"):
        assert _extract_text_and_strip_status({"text": "<think>x</think>"}, "qwen3") == ("", True)
    assert "Failed to strip thinking tokens" in caplog.text


def test_no_stripper_configured_means_the_raw_text_stripped_of_whitespace():
    assert _extract_text_and_strip_status({"text": "  no  "}, None)[0] == "no"


def test_a_nonstring_text_with_no_stripper_becomes_empty():
    assert _extract_text_and_strip_status({"text": None}, None)[0] == ""


@pytest.mark.parametrize("model, raw", [
    ("qwen3", "<think>\nFinal Judgment: yes\nbut wait"),
    ("qwen3", "<think>a</think>\n<think>\nFinal Judgment: no"),
    ("gemma4", "<|channel>thought\nDraft\nFinal Judgment: no\nhmm"),
    ("gemma4", "<|channel>thought\nx<channel|>\nno\n<|channel>thought\nFinal Judgment: yes"),
    ("gemma4", '<|channel>thought\n{"full_set_judgment": "yes"}'),
])
def test_an_answer_whose_thought_never_closes_has_no_answer(model, raw):
    assert _extract_text_and_strip_status({"text": raw}, model) == ("", True)


@pytest.mark.parametrize("model, raw, answer", [
    ("qwen3", "<think>a</think>\n<think>b</think>\nno", "no"),
    ("olmo3", "reasoning</think>\nno", "no"),
    ("gemma4", "<|channel>thought\nx<channel|>no", "no"),
])
def test_a_closed_thought_leaves_its_answer(model, raw, answer):
    assert _extract_text_and_strip_status({"text": raw}, model) == (answer, False)


FORCED_YES = '{"rule_judgments": {"Rule 1": "yes"}, "full_set_judgment": "yes"}'


# Each style with each validation it runs under: strict_think reads a </think> thought, so a campaign refuses it for Gemma.
STYLE_VALIDATIONS = [("qwen3", "<think>", "</think>", strict) for strict in (False, True)] + [
    ("olmo3", "<think>", "</think>", strict) for strict in (False, True)] + [
    ("gemma4", "<|channel>thought", "<channel|>", False)]


@pytest.mark.parametrize("style, opening, closing, strict_think", STYLE_VALIDATIONS)
@pytest.mark.parametrize("shape", ["prose, then an open thought", "a closed thought, then an open one"])
def test_no_answer_is_read_from_a_thought_that_never_closes_in_any_style(style, opening, closing, shape, strict_think):
    from ruleforge.evaluate.validation import EMPTY_RESPONSE, UNCLOSED_THINK, ComponentSchema
    from ruleforge.judges.backend import ModelBackend, _judged_text

    first = "Let me reason carefully.\n" if shape.startswith("prose") else f"{opening}draft{closing}\n"
    raw = f"{first}{opening}\n{FORCED_YES}"
    completion, strip_failed = _extract_text_and_strip_status({"text": raw}, style)
    detail = {"raw_response": raw, "completion": completion, "finish_reason": "stop"}
    [outcome] = ModelBackend._outcomes_from_details([detail])
    schema = ComponentSchema.from_rules([{"id": "Rule 1", "rule": "be brief"}], prompt_style="forced",
                                        forced_json_requirement="complete_rule_map", strict_think=strict_think)
    verdict = schema.classify("full_ruleset", outcome)
    assert (verdict.valid, verdict.judgment, _judged_text([detail])) == (False, None, "")
    assert verdict.failure_reason in ((UNCLOSED_THINK,) if strict_think or not strip_failed else (EMPTY_RESPONSE,))
