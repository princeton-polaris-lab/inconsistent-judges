from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Gemma 4's special tokens, as its tokenizer_config names them (pad, eos, bos, unk first).
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>",
            "<|tool_response>"]
CHAT_TEMPLATE = (
    "{{ bos_token }}<|turn>system\n{%- if enable_thinking is defined and enable_thinking %}<|think|>\n{% endif -%}"
    "{{ messages[0]['content'] }}<turn|>\n{%- for m in messages[1:] %}<|turn>{{ m['role'] }}\n{{ m['content'] }}"
    "<turn|>\n{%- endfor %}{%- if add_generation_prompt %}<|turn>model\n"
    "{% if not (enable_thinking is defined and enable_thinking) %}<|channel>thought\n<channel|>{% endif %}{% endif %}"
)
#: A processor config of Gemma 4's shape (``gemma-4-31B-it`` ships one; the merge copies it).
PROCESSOR_CONFIG = {
    "processor_class": "Gemma4Processor", "image_seq_length": 4,
    "image_processor": {"image_processor_type": "Gemma4ImageProcessor", "patch_size": 4, "pooling_kernel_size": 1},
}
#: The model's weights directory name: a campaign condition names its judge's by it.
MODEL_NAME = "gemma-4-31B-it"


def text_config(**changes: Any) -> dict[str, Any]:
    config = dict(
        vocab_size=272, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention"], sliding_window=8, max_position_embeddings=256,
        final_logit_softcapping=30.0, tie_word_embeddings=True, pad_token_id=0, eos_token_id=1, bos_token_id=2,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=272,
    )
    config.update(changes)
    return config


def gemma_config(**text_changes: Any) -> Any:
    import transformers

    vision = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=8, global_head_dim=8, patch_size=4, pooling_kernel_size=1,
                  position_embedding_size=64)
    return transformers.Gemma4Config(
        text_config=text_config(**text_changes), vision_config=vision, audio_config=None, tie_word_embeddings=True,
    )


def save_tokenizer(directory: Path) -> Path:
    """A byte-level BPE tokenizer (every text round-trips) with :data:`SPECIALS` as special tokens."""
    import transformers
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {symbol: index for index, symbol in enumerate(sorted(ByteLevel.alphabet()))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in SPECIALS])
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
        chat_template=CHAT_TEMPLATE,
    ).save_pretrained(str(directory))
    return directory


def token_id(directory: Path, token: str) -> int:
    from transformers import AutoTokenizer

    return int(AutoTokenizer.from_pretrained(str(directory)).convert_tokens_to_ids(token))


def save_tiny_gemma(directory: Path, *, seed: int = 0, weights: bool = True, **text_changes: Any) -> Path:
    """The tiny checkpoint directory."""
    import transformers

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_tokenizer(directory)
    ids = {token: token_id(directory, token) for token in ("<pad>", "<eos>", "<bos>", "<turn|>", "<|tool_response>")}
    config = gemma_config(**text_changes)
    text = config.get_text_config()
    text.pad_token_id, text.eos_token_id, text.bos_token_id = ids["<pad>"], ids["<eos>"], ids["<bos>"]
    config.eos_token_id = [ids["<eos>"], ids["<turn|>"]]
    if weights:
        import torch

        torch.manual_seed(seed)
        transformers.Gemma4ForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(str(directory))
    else:
        config.save_pretrained(str(directory))
    (directory / "generation_config.json").write_text(json.dumps({
        "bos_token_id": ids["<bos>"], "do_sample": True, "eos_token_id": [ids["<eos>"], ids["<turn|>"],
                                                                            ids["<|tool_response>"]],
        "pad_token_id": ids["<pad>"], "temperature": 1.0, "top_k": 64, "top_p": 0.95,
    }))
    (directory / "processor_config.json").write_text(json.dumps(PROCESSOR_CONFIG, indent=2))
    return directory
