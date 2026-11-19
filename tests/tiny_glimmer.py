from __future__ import annotations

from pathlib import Path

SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start|>", "<|message|>", "<|eom|>", "<|eot|>",
            "<|finetune_right_pad|>"]
WORDS = ["[UNK]", "system", "user", "assistant", "to", "=", "self", "Reasoning", "strength", ":", "high", ".", "#",
         "Valid", "recipients", '"', ",", "yes", "no", "unsure", "judge", "the", "rule", "set"]
CHAT_TEMPLATE = (
    "{{- bos_token -}}{%- for m in messages -%}"
    "{%- if m['role'] == 'system' -%}<|start|>system<|message|>{{- m['content'] }}"
    "\n\nReasoning strength: high.\n\n# Valid recipients: \"self\", \"user\".<|eot|>"
    "{%- else -%}<|start|>{{- m['role'] -}}<|message|>{{- m['content'] -}}<|eot|>{%- endif -%}"
    "{%- endfor -%}{%- if add_generation_prompt -%}<|start|>assistant{%- endif -%}"
)


def save_tiny_glimmer(directory: Path) -> Path:
    """Write the tiny tokenizer into ``directory`` (created) and return it."""
    import transformers
    from tokenizers import Tokenizer as RustTokenizer
    from tokenizers import models, pre_tokenizers, processors

    vocab = {token: index for index, token in enumerate(SPECIALS + WORDS)}
    tok = RustTokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok.add_special_tokens(SPECIALS)
    tok.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A", pair="<|begin_of_text|> $A <|begin_of_text|> $B",
        special_tokens=[("<|begin_of_text|>", vocab["<|begin_of_text|>"])])
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>",
        pad_token="<|finetune_right_pad|>", unk_token="[UNK]")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(str(directory))
    return directory
