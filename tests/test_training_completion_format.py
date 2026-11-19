from __future__ import annotations

import dataclasses
import os
import types
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from ruleforge.judges import completion_format as cf  # noqa: E402
from ruleforge.grpo.rewards import QWEN3_FINISH_TOKEN_IDS, QWEN3_THINK_END_TOKEN_ID  # noqa: E402
from tests.tiny_gemma import save_tiny_gemma  # noqa: E402

MODEL_ROOT = Path(os.environ.get("RULEFORGE_MODEL_ROOT") or Path.home() / "models")
QWEN3_DIR, GEMMA_DIR = MODEL_ROOT / "Qwen3-8B", MODEL_ROOT / "gemma-4-31B-it"


@pytest.fixture(scope="module")
def gemma_dir(tmp_path_factory):
    return save_tiny_gemma(tmp_path_factory.mktemp("models") / "gemma-4-31B-it", weights=False)


def _qwen_like(directory: Path) -> Path:
    """A word-level Qwen3 stand-in."""
    import transformers
    from tokenizers import Tokenizer as RustTokenizer
    from tokenizers import models, pre_tokenizers

    specials = ["<|endoftext|>", "<|im_end|>"]
    vocab = {token: index for index, token in enumerate(specials + ["<think>", "no", "yes", "[UNK]", "</think>"])}
    tok = RustTokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok.add_special_tokens(specials)
    tok.add_tokens(["<think>", "</think>"])
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token="[UNK]",
    ).save_pretrained(str(directory))
    return directory


def test_the_families_are_keyed_by_model_type():
    assert cf.thinking_family(types.SimpleNamespace(model_type="qwen3")) is cf.THINKING_FAMILIES["qwen3"]
    assert cf.thinking_family(types.SimpleNamespace(model_type="gemma4")).end_of_turn == "<turn|>"
    with pytest.raises(cf.CompletionFormatError, match="no thinking family for model_type 'llama'"):
        cf.thinking_family(types.SimpleNamespace(model_type="llama"))


def test_the_qwen3_family_reads_the_decoded_text_and_keeps_the_tokenizers_eos(tmp_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(_qwen_like(tmp_path / "qwen")))
    family = cf.THINKING_FAMILIES["qwen3"]
    assert cf.prepare_tokenizer(tokenizer, family) is tokenizer and tokenizer.eos_token == "<|im_end|>"
    fmt = cf.completion_format(tokenizer, family, stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id])
    assert fmt.finish_token_ids == (tokenizer.eos_token_id, tokenizer.pad_token_id)
    assert fmt.think_end_token_id == tokenizer.convert_tokens_to_ids("</think>")
    assert (fmt.decode, fmt.think_open, fmt.other_stop_token_ids) == (None, "<think>", ())
    assert fmt.record()["split"] == "text"


def test_the_gemma4_family_ends_the_turn_at_turn_and_splits_the_ids_at_the_channel(gemma_dir):
    tokenizer, fmt = cf.load_training_format(gemma_dir)
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in ("<turn|>", "<pad>", "<eos>", "<channel|>",
                                                                      "<|tool_response>")}
    # the training tokenizer's eos is the end of turn; the checkpoint's other stop ids finish nothing
    assert tokenizer.eos_token == "<turn|>" and fmt.finish_token_ids == (ids["<turn|>"], ids["<pad>"])
    assert fmt.other_stop_token_ids == tuple(sorted((ids["<eos>"], ids["<|tool_response>"])))
    assert (fmt.family, fmt.think_end_token_id, fmt.think_open) == ("gemma4", ids["<channel|>"], "thought")
    # TRL's decode drops the channel markers: the format decodes the ids it splits
    text = "<|channel>thought\nweigh<channel|>answer"
    assert fmt.decode(tokenizer.encode(text, add_special_tokens=False)) == "thought\nweighanswer"
    assert fmt.record() == {
        "family": "gemma4", "finish_token_ids": list(fmt.finish_token_ids), "think_end_token_id": ids["<channel|>"],
        "think_open": "thought", "split": "ids", "other_stop_token_ids": list(fmt.other_stop_token_ids),
    }
    assert cf.prepare_tokenizer(tokenizer, cf.THINKING_FAMILIES["gemma4"]).eos_token == "<turn|>"


def test_a_tokenizer_the_family_does_not_describe_is_refused(gemma_dir, tmp_path):
    gemma = cf.THINKING_FAMILIES["gemma4"]
    raw = transformers.AutoTokenizer.from_pretrained(str(gemma_dir))
    with pytest.raises(cf.CompletionFormatError, match="prepare it first"):
        cf.completion_format(raw, gemma)
    tokenizer = cf.prepare_tokenizer(raw, gemma)
    with pytest.raises(cf.CompletionFormatError, match="declares that it keeps it"):
        cf.completion_format(tokenizer, dataclasses.replace(gemma, close_decoded=True))
    with pytest.raises(cf.CompletionFormatError, match="'</think>' is not one token"):
        cf.completion_format(tokenizer, dataclasses.replace(gemma, think_close="</think>"))
    with pytest.raises(cf.CompletionFormatError, match="not among the ids generation stops on"):
        cf.completion_format(tokenizer, gemma, stop_token_ids=[tokenizer.pad_token_id])
    with pytest.raises(cf.CompletionFormatError, match="'<end_of_turn>' is not one token"):
        cf.prepare_tokenizer(transformers.AutoTokenizer.from_pretrained(str(gemma_dir)),
                             dataclasses.replace(gemma, end_of_turn="<end_of_turn>"))
    qwen = transformers.AutoTokenizer.from_pretrained(str(_qwen_like(tmp_path / "qwen")))
    with pytest.raises(cf.CompletionFormatError, match="declares that it drops it"):
        cf.completion_format(qwen, dataclasses.replace(cf.THINKING_FAMILIES["qwen3"], close_decoded=False))
    unnamed = types.SimpleNamespace(eos_token_id=None, pad_token_id=0)
    with pytest.raises(cf.CompletionFormatError, match="names no eos"):
        cf.completion_format(unnamed, gemma)


@pytest.mark.skipif(not (QWEN3_DIR / "tokenizer.json").is_file(), reason="the Qwen3-8B checkpoint is not on this machine")
def test_the_qwen3_8b_format_is_exactly_the_rewards_constants():
    tokenizer, fmt = cf.load_training_format(QWEN3_DIR)
    assert fmt.finish_token_ids == QWEN3_FINISH_TOKEN_IDS == (151645, 151643)
    assert fmt.think_end_token_id == QWEN3_THINK_END_TOKEN_ID == 151668
    assert (fmt.decode, fmt.think_open, fmt.other_stop_token_ids, tokenizer.eos_token) == (
        None, "<think>", (), "<|im_end|>")


@pytest.mark.skipif(not (GEMMA_DIR / "tokenizer.json").is_file(), reason="the Gemma 4 checkpoint is not on this machine")
def test_the_gemma_4_31b_format_is_its_tokenizers():
    tokenizer, fmt = cf.load_training_format(GEMMA_DIR)
    # <turn|> 106 ends a turn, <pad> 0 pads; generation also stops on <eos> 1 and <|tool_response> 50
    assert (fmt.finish_token_ids, fmt.other_stop_token_ids, fmt.think_end_token_id) == ((106, 0), (1, 50), 101)
    assert tokenizer.eos_token == "<turn|>" and fmt.decode([100, 45518, 107, 101]) == "thought\n"


def test_a_directorys_thinking_family_is_read_from_its_config(tmp_path):
    import json

    from ruleforge.judges.completion_format import THINKING_FAMILIES, CompletionFormatError, directory_family

    (tmp_path / "q").mkdir()
    (tmp_path / "q" / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert directory_family(tmp_path / "q") is THINKING_FAMILIES["qwen3"]
    fixture = Path(__file__).resolve().parent / "fixtures" / "grpo" / "gemma4_31b"
    assert directory_family(fixture) is THINKING_FAMILIES["gemma4"]
    with pytest.raises(CompletionFormatError, match="holds no config.json"):
        directory_family(tmp_path)
    (tmp_path / "o").mkdir()
    (tmp_path / "o" / "config.json").write_text(json.dumps({"model_type": "olmo3"}))
    with pytest.raises(CompletionFormatError, match="no thinking family for model_type 'olmo3'"):
        directory_family(tmp_path / "o")
    assert directory_family(tmp_path, required=False) is None and directory_family(tmp_path / "o", required=False) is None
    assert directory_family(tmp_path / "q", required=False) is THINKING_FAMILIES["qwen3"]
