from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

from ruleforge.core.errors import RuleForgeError, SamplingDesignError

GENERATION_CONFIG_FNAME = "generation_config.json"


class CompletionFormatError(ValueError, RuleForgeError):
    """Raised for a model whose completions cannot be read by its declared thinking family."""


class GrpoDataError(SamplingDesignError):
    """Raised when the training data cannot be built from these inputs."""


@dataclass(frozen=True)
class ThinkingFamily:
    """How one family of thinking models writes a completion."""

    name: str
    think_open: str  # what the decoded reasoning starts with
    think_close: str  # the one token that closes the reasoning
    close_decoded: bool  # whether decoding with skip_special_tokens keeps think_close in the text
    raw_think_open: str  # what a raw completion (special tokens kept) opens its reasoning with
    end_of_turn: Optional[str] = None  # made the training tokenizer's eos when set


# By config.json's model_type.
THINKING_FAMILIES: dict[str, ThinkingFamily] = {
    "qwen3": ThinkingFamily(
        name="qwen3", think_open="<think>", think_close="</think>", close_decoded=True, raw_think_open="<think>",
    ),
    "gemma4": ThinkingFamily(
        name="gemma4", think_open="thought", think_close="<channel|>", close_decoded=False,
        raw_think_open="<|channel>thought", end_of_turn="<turn|>",
    ),
}


def thinking_family(model_config: Any) -> ThinkingFamily:
    """Return the thinking family of a checkpoint's config; refuse a ``model_type`` no family declares."""
    model_type = getattr(model_config, "model_type", None)
    if model_type not in THINKING_FAMILIES:
        raise CompletionFormatError(
            f"no thinking family for model_type {model_type!r}; the reward reads completions of "
            f"{sorted(THINKING_FAMILIES)} (ruleforge.judges.completion_format.THINKING_FAMILIES)"
        )
    return THINKING_FAMILIES[model_type]


def directory_family(model_dir: Path | str, *, required: bool = True) -> Optional[ThinkingFamily]:
    """Return the thinking family of a checkpoint directory from its ``config.json``; None instead of raising unless
    ``required``."""
    path = Path(model_dir) / "config.json"
    model_type = json.loads(path.read_text(encoding="utf-8")).get("model_type") if path.is_file() else None
    if not required:
        return THINKING_FAMILIES.get(model_type)
    if not path.is_file():
        raise CompletionFormatError(f"{model_dir} holds no config.json: its thinking family cannot be read")
    return thinking_family(SimpleNamespace(model_type=model_type))


def single_token_id(tokenizer: Any, token: str) -> int:
    """Return the id of ``token``, which must be exactly one token of the tokenizer."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    encoded = list(tokenizer.encode(token, add_special_tokens=False))
    if token_id is None or token_id == tokenizer.unk_token_id or encoded != [token_id]:
        raise CompletionFormatError(
            f"{token!r} is not one token of this tokenizer (id {token_id!r}, encodes to {encoded}): the reward "
            "reads the completion at that token"
        )
    return int(token_id)


def prepare_tokenizer(tokenizer: Any, family: ThinkingFamily) -> Any:
    """Make the tokenizer's eos the family's end of turn, where one is declared; changes it in place and returns it."""
    if family.end_of_turn is not None and tokenizer.eos_token != family.end_of_turn:
        single_token_id(tokenizer, family.end_of_turn)
        tokenizer.eos_token = family.end_of_turn
    return tokenizer


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True)


@dataclass(frozen=True)
class CompletionFormat:
    """The ids a completion is read by, resolved from the training tokenizer."""

    family: str
    finish_token_ids: tuple[int, ...]  # the tokenizer's eos and pad
    think_end_token_id: int
    think_open: str
    decode: Optional[Callable[[Sequence[int]], str]] = None  # None when the closing marker survives decoding
    other_stop_token_ids: tuple[int, ...] = ()

    def record(self) -> dict[str, Any]:
        """Return what a checkpoint's provenance records of it."""
        return {
            "family": self.family,
            "finish_token_ids": list(self.finish_token_ids),
            "think_end_token_id": self.think_end_token_id,
            "think_open": self.think_open,
            "split": "text" if self.decode is None else "ids",
            "other_stop_token_ids": list(self.other_stop_token_ids),
        }


def completion_format(
    tokenizer: Any, family: ThinkingFamily, *, stop_token_ids: Sequence[int] = ()
) -> CompletionFormat:
    """Return the ids ``family``'s completions are read by under a prepared ``tokenizer``, whose eos must be among
    ``stop_token_ids`` (the checkpoint's generation stops) when given."""
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if eos is None or pad is None:
        raise CompletionFormatError(f"the tokenizer names no eos ({eos!r}) or no pad ({pad!r}) token")
    if family.end_of_turn is not None and tokenizer.eos_token != family.end_of_turn:
        raise CompletionFormatError(
            f"the tokenizer's eos is {tokenizer.eos_token!r}, not {family.name}'s end of turn "
            f"{family.end_of_turn!r}: prepare it first (prepare_tokenizer)"
        )
    think_end = single_token_id(tokenizer, family.think_close)
    decoded = _decode(tokenizer, [think_end])
    if (decoded == family.think_close) != family.close_decoded:
        raise CompletionFormatError(
            f"TRL's decode turns {family.think_close!r} into {decoded!r}; {family.name} declares that it "
            f"{'keeps' if family.close_decoded else 'drops'} it"
        )
    stops = [int(token) for token in stop_token_ids]
    if stops and int(eos) not in stops:
        raise CompletionFormatError(
            f"the tokenizer's eos {eos} is not among the ids generation stops on ({stops}): no completion "
            "would end on it"
        )
    return CompletionFormat(
        family=family.name,
        finish_token_ids=(int(eos), int(pad)),
        think_end_token_id=think_end,
        think_open=family.think_open,
        decode=None if family.close_decoded else functools.partial(_decode, tokenizer),
        other_stop_token_ids=tuple(sorted(set(stops) - {int(eos), int(pad)})),
    )


def load_training_format(model_dir: Any) -> tuple[Any, CompletionFormat]:
    """Return a checkpoint directory's prepared training tokenizer and its ``CompletionFormat``."""
    from transformers import AutoConfig, AutoTokenizer

    family = thinking_family(AutoConfig.from_pretrained(str(model_dir)))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return tokenizer, training_format(tokenizer, family, model_dir)


def stop_token_ids(directory: Path) -> list[int]:
    """Return the ids generation stops on, from the checkpoint's ``generation_config.json``."""
    path = directory / GENERATION_CONFIG_FNAME
    if not path.is_file():
        raise GrpoDataError(
            f"{directory} has no {GENERATION_CONFIG_FNAME}; the replayed completions end on the tokenizer's eos, "
            "which must be one of the ids the checkpoint stops generating on"
        )
    ids = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
    return [int(value) for value in (ids if isinstance(ids, list) else [] if ids is None else [ids])]


def training_format(tokenizer: Any, family: ThinkingFamily, model_dir: Any) -> CompletionFormat:
    """Prepare ``tokenizer`` for ``family`` in place and return the ``CompletionFormat`` it reads completions by."""
    prepare_tokenizer(tokenizer, family)
    return completion_format(tokenizer, family, stop_token_ids=stop_token_ids(Path(model_dir)))
