from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ruleforge.core.errors import RuleForgeError

logger = logging.getLogger(__name__)

TOKENIZER_JSON_FNAME = "tokenizer.json"

# Combining marks, accents, emoji, CJK, digits, contractions and whitespace: every regex branch a class could differ on.
FIDELITY_PROBE = (
    "สวัสดีครับ ยินดีต้อนรับ | नमस्ते, आप कैसे हैं? | مَرْحَبًا بِكُمْ مُحَمَّد | "
    "שָׁלוֹם | নমস্কার | வணக்கம் | မင်္ဂလာပါ | ជំរាបសួរ | Tiếng Việt | "
    "café naïve é ñ | ❤️ 👍🏽 👨‍👩‍👧 | 你好，世界 こんにちは 안녕하세요 | "
    "It's 2026-09-24: x=3.14; they'll've\n\n\t  indented\r\n"
)


class TokenizerMismatch(RuntimeError, RuleForgeError):
    """Raised when a loaded tokenizer encodes text differently from the checkpoint's ``tokenizer.json``."""


def has_tokenizer_json(model_path: str | Path) -> bool:
    """Return whether the checkpoint ships a ``tokenizer.json``."""
    return (Path(model_path) / TOKENIZER_JSON_FNAME).is_file()


def reference_token_ids(model_path: str | Path, text: str = FIDELITY_PROBE) -> list[int] | None:
    """Return ``text`` encoded by the checkpoint's own ``tokenizer.json``, or None without one."""
    if not has_tokenizer_json(model_path):
        return None
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(model_path) / TOKENIZER_JSON_FNAME))
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def check_tokenizer_fidelity(tokenizer: Any, model_path: str | Path, *, where: str, text: str = FIDELITY_PROBE) -> None:
    """Refuse ``tokenizer`` unless it encodes ``text`` as ``tokenizer.json`` does; ``where`` names it in the error."""
    expected = reference_token_ids(model_path, text)
    if expected is None:
        logger.info("%s: no %s to check the %s tokenizer against", model_path, TOKENIZER_JSON_FNAME, where)
        return
    actual = list(tokenizer.encode(text, add_special_tokens=False))
    if actual != expected:
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise TokenizerMismatch(
            f"the {where} tokenizer ({type(tokenizer).__name__}) encodes the fidelity probe as "
            f"{len(actual)} tokens, {Path(model_path) / TOKENIZER_JSON_FNAME} as {len(expected)} "
            f"(first difference at token {first}): it is not the tokenizer the checkpoint ships, so "
            "prompts would reach the model as other token ids than it was trained on. Known cause: "
            "transformers >= 5.6 loads Qwen3.5-class checkpoints (model_type qwen3_5) through the "
            "Qwen2Tokenizer their tokenizer_config.json names, without their \\p{M} pre-tokenizer "
            "classes (transformers #49066); such a judge is blocked on this stack until that is fixed"
        )
