from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch

from ruleforge.core.errors import RuleForgeError

#: The value a masked logit takes: finite, so no NaN can arise.
TRUNCATED_LOGIT = -1.0e4
#: Rows of logits the keep mask handles at once.
DEFAULT_CHUNK_ROWS = 256


class TruncationError(ValueError, RuleForgeError):
    """A truncation the mask cannot reproduce, or that disagrees with what the sampler uses."""


@dataclass(frozen=True)
class Truncation:
    """The sampler truncation the rollouts use; `top_k` 0 or less disables top-k, `top_p` 1.0 disables top-p."""

    temperature: float
    top_k: int
    top_p: float
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise TruncationError(f"temperature must be positive and finite, got {self.temperature!r}")
        if isinstance(self.top_k, bool) or int(self.top_k) != self.top_k:
            raise TruncationError(f"top_k must be an integer, got {self.top_k!r}")
        object.__setattr__(self, "top_k", max(int(self.top_k), 0))
        if not (0.0 < float(self.top_p) <= 1.0):
            raise TruncationError(f"top_p must be in (0, 1], got {self.top_p!r}")
        if float(self.min_p) != 0.0:
            raise TruncationError(
                f"min_p {self.min_p!r}: vLLM applies min-p before top-k and top-p, and the HF-side mask does "
                "not mirror it; the training decoder has min_p 0"
            )
        if float(self.repetition_penalty) != 1.0:
            raise TruncationError(
                f"repetition_penalty {self.repetition_penalty!r}: a penalty depends on the tokens generated so "
                "far and the HF-side mask does not mirror it; the training decoder has none"
            )
        for name in ("temperature", "top_p", "min_p", "repetition_penalty"):
            object.__setattr__(self, name, float(getattr(self, name)))

    @classmethod
    def from_sampling(cls, sampling: Mapping[str, Any], *, temperature: float) -> "Truncation":
        """Return the recorded `sampling_truncation` at `temperature`."""
        keys = ("top_k", "top_p", "min_p", "repetition_penalty")
        missing = [key for key in keys if key not in sampling]
        if missing:
            raise TruncationError(f"the sampling truncation {dict(sampling)} lacks {missing}")
        return cls(temperature=temperature, **{key: sampling[key] for key in keys})

    @property
    def truncates(self) -> bool:
        """Return whether the mask changes anything."""
        return self.top_k > 0 or self.top_p < 1.0

    @property
    def outside_logp(self) -> float:
        """Return the log-prob below which a token is outside the kept set."""
        return TRUNCATED_LOGIT / (2.0 * self.temperature)

    def as_sampling(self) -> dict[str, float | int]:
        """Return the four sampler keys, as a campaign records `sampling_truncation`."""
        return {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }


def _first(candidates: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    return candidates & (torch.cumsum(candidates, dim=-1, dtype=torch.int32) <= count)


def _keep_rows(scaled: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    vocab = scaled.size(-1)
    if 0 < top_k < vocab:
        values = scaled.topk(top_k, dim=-1).values
        above = scaled > values[:, -1:]
        keep = above | _first(scaled == values[:, -1:], top_k - above.sum(dim=-1, keepdim=True))
    else:
        values = None
        keep = torch.ones_like(scaled, dtype=torch.bool)
    if top_p >= 1.0:
        return keep
    if values is None:
        values = scaled.sort(dim=-1, descending=True).values
    # renormalize over the kept set, then keep the count whose exclusive cumulative mass is below top_p
    lse = torch.logsumexp(scaled.masked_fill(~keep, -math.inf), dim=-1, keepdim=True)
    probs = (values - lse).exp()
    exclusive = probs.cumsum(dim=-1) - probs
    count = (exclusive < top_p).sum(dim=-1, keepdim=True).clamp(min=1)
    threshold = values.gather(-1, count - 1)
    above = keep & (scaled > threshold)
    return above | _first(keep & (scaled == threshold), count - above.sum(dim=-1, keepdim=True))


@torch.no_grad()
def keep_mask(logits: torch.Tensor, truncation: Truncation, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> torch.Tensor:
    """Return a boolean `logits`-shaped mask of the tokens the truncation keeps at each position."""
    if chunk_rows < 1:
        raise TruncationError(f"chunk_rows must be positive, got {chunk_rows}")
    flat = logits.detach().reshape(-1, logits.size(-1))
    out = torch.empty(flat.shape, dtype=torch.bool, device=flat.device)
    for start in range(0, flat.size(0), chunk_rows):
        rows = flat[start : start + chunk_rows].float() / truncation.temperature
        out[start : start + chunk_rows] = _keep_rows(rows, truncation.top_k, truncation.top_p)
    return out.view(logits.shape)


def mask_logits(logits: torch.Tensor, truncation: Truncation, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> torch.Tensor:
    """Return `logits` with every token the truncation drops set to `TRUNCATED_LOGIT` (differentiable)."""
    if not truncation.truncates:
        return logits
    keep = keep_mask(logits, truncation, chunk_rows=chunk_rows)
    return logits.masked_fill(~keep, TRUNCATED_LOGIT)


def outside_kept_share(logps: torch.Tensor, mask: torch.Tensor, truncation: Truncation) -> tuple[int, int]:
    """Return `(tokens outside the kept set, tokens)` among the masked positions."""
    selected = mask.bool()
    outside = (logps < truncation.outside_logp) & selected
    return int(outside.sum()), int(selected.sum())


def zero_outside_logps(logps: torch.Tensor, truncation: Truncation) -> torch.Tensor:
    """Return `logps` with each token outside the kept set set to 0, so a later pass that keeps the token gets a finite
    PPO ratio."""
    return torch.where(logps < truncation.outside_logp, torch.zeros_like(logps), logps)


class TruncationMask:
    """A forward hook that masks a module's output logits, active only inside `active`; attached to the LM head, or to
    the model itself when it soft-caps its final logits."""

    def __init__(self, truncation: Truncation, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> None:
        self.truncation = truncation
        self.chunk_rows = chunk_rows
        self._depth = 0
        self._handle: Any = None

    def attach(self, module: torch.nn.Module) -> "TruncationMask":
        if self._handle is not None:
            raise TruncationError("the truncation mask is already attached")
        self._handle = module.register_forward_hook(self._hook)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def is_active(self) -> bool:
        return self._depth > 0

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def _hook(self, module: torch.nn.Module, args: Any, output: Any) -> Any:
        if not self.is_active:
            return None
        if isinstance(output, torch.Tensor):
            return mask_logits(output, self.truncation, chunk_rows=self.chunk_rows)
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise TruncationError(
                f"the masked module returned {type(output).__name__}: not a logits tensor, nor an output with logits"
            )
        output["logits"] = mask_logits(logits, self.truncation, chunk_rows=self.chunk_rows)
        return output


def logits_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module whose output holds the logits TRL takes log-probabilities of: the LM head, or the model itself
    when it soft-caps its final logits."""
    config = model.config
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    if getattr(text, "final_logit_softcapping", None) is None:
        return model.get_output_embeddings()
    get_base = getattr(model, "get_base_model", None)
    return model if get_base is None else get_base()
