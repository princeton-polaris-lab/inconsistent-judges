from __future__ import annotations

import functools
import hashlib
import json
import math
import weakref
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
import torch.utils.data

from ruleforge.core.record_names import ANCHOR_REPLAY_SEED_LABEL


def validate_anchor_num_workers(num_workers: int) -> int:
    """Return `num_workers`, refusing a count that could fork after CUDA initialisation."""
    if num_workers < 0:
        raise ValueError(
            f"anchor.dataloader_num_workers must be non-negative, got {num_workers}."
        )
    if num_workers > 0:
        raise ValueError(
            "Anchor replay requires dataloader_num_workers=0 to avoid "
            "fork-after-CUDA hazards."
        )
    return num_workers


@dataclass(frozen=True)
class AnchorAuxConfig:
    """Settings of the dense auxiliary loss."""

    loss_weight: float
    kind: str = "kl"
    mix_every_n_steps: int = 1
    per_device_batch_size: int = 1
    dataloader_num_workers: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"kl", "ce"}:
            raise ValueError(f"kind must be 'kl' or 'ce', got {self.kind!r}")
        if self.loss_weight < 0:
            raise ValueError("loss_weight must be non-negative")
        if self.mix_every_n_steps <= 0:
            raise ValueError("mix_every_n_steps must be positive")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be positive")
        validate_anchor_num_workers(self.dataloader_num_workers)


def _shift(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, T, V], got {tuple(logits.shape)}")
    if labels.dim() != 2:
        raise ValueError(f"labels must be [B, T], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits/labels shape mismatch: {tuple(logits.shape[:2])} "
            f"vs {tuple(labels.shape)}"
        )
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def _at_least_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.promote_types(tensor.dtype, torch.float32))


def anchor_kl_loss(
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the summed forward KL over the supervised tokens and their count."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if policy_logits.shape != base_logits.shape:
        raise ValueError(
            "policy/base logits shape mismatch: "
            f"{tuple(policy_logits.shape)} vs {tuple(base_logits.shape)}"
        )

    shifted_policy, shifted_labels = _shift(policy_logits, labels)
    shifted_base, _ = _shift(base_logits, labels)
    mask = shifted_labels != -100
    token_count = mask.sum().to(dtype=torch.float32)
    if not bool(mask.any()):
        return policy_logits.sum() * 0.0, token_count

    # at least fp32: a KL of order 1e-3 between two bf16 log-softmaxes is mostly rounding
    policy = _at_least_fp32(shifted_policy[mask])
    base = _at_least_fp32(shifted_base[mask].detach())
    log_policy = F.log_softmax(policy / temperature, dim=-1)
    log_base = F.log_softmax(base / temperature, dim=-1)
    kl_per_token = (log_base.exp() * (log_base - log_policy)).sum(dim=-1)
    return kl_per_token.sum(), token_count


def anchor_ce_loss(
    policy_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the summed masked cross entropy over the supervised tokens and their count."""
    shifted_logits, shifted_labels = _shift(policy_logits, labels)
    loss_sum = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction="sum",
        ignore_index=-100,
    )
    token_count = shifted_labels.ne(-100).sum().to(dtype=torch.float32)
    return loss_sum, token_count


def anchor_aux_loss(
    *,
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor | None,
    labels: torch.Tensor,
    config: AnchorAuxConfig,
    temperature: float = 1.0,
    grad_accum_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the weighted loss and the detached per-token raw loss."""
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if config.kind == "kl":
        if base_logits is None:
            raise ValueError("KL anchor loss requires base_logits")
        loss_sum, token_count = anchor_kl_loss(
            policy_logits,
            base_logits,
            labels,
            temperature=temperature,
        )
    else:
        loss_sum, token_count = anchor_ce_loss(policy_logits, labels)

    raw_loss = loss_sum / token_count.clamp(min=1.0)
    weighted_loss = raw_loss * float(config.loss_weight) / float(grad_accum_steps)
    return weighted_loss, raw_loss.detach()


#: Columns a replay row feeds the model; `weight` is optional (default 1).
REPLAY_TOKEN_COLUMNS = ("input_ids", "labels")
REPLAY_MASK_COLUMN = "attention_mask"
REPLAY_WEIGHT_COLUMN = "weight"
IGNORE_INDEX = -100


def _replay_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    missing = [name for name in REPLAY_TOKEN_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"replay row {index} lacks {missing}; a replay row is token ids and labels")
    input_ids = [int(token) for token in row["input_ids"]]
    labels = [int(token) for token in row["labels"]]
    mask = [int(value) for value in row.get(REPLAY_MASK_COLUMN) or [1] * len(input_ids)]
    if not input_ids or len(labels) != len(input_ids) or len(mask) != len(input_ids):
        raise ValueError(
            f"replay row {index}: input_ids ({len(input_ids)}), labels ({len(labels)}) and attention_mask "
            f"({len(mask)}) must be non-empty and of one length"
        )
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError(
            f"replay row {index} supervises no token after its first: it would add nothing to the anchor "
            "but its share of the batch"
        )
    weight = float(row.get(REPLAY_WEIGHT_COLUMN, 1.0))
    if not (math.isfinite(weight) and weight > 0):
        raise ValueError(f"replay row {index} has weight {weight!r}; a weight is positive and finite")
    return {"input_ids": input_ids, "attention_mask": mask, "labels": labels, REPLAY_WEIGHT_COLUMN: weight}


def collate_replay_rows(rows: Sequence[Mapping[str, Any]], *, pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad replay rows into a batch; weights come back as a float vector."""
    length = max(len(row["input_ids"]) for row in rows)

    def padded(name: str, value: int) -> torch.Tensor:
        return torch.tensor([row[name] + [value] * (length - len(row[name])) for row in rows], dtype=torch.long)

    return {
        "input_ids": padded("input_ids", pad_token_id),
        "attention_mask": padded("attention_mask", 0),
        "labels": padded("labels", IGNORE_INDEX),
        REPLAY_WEIGHT_COLUMN: torch.tensor([row[REPLAY_WEIGHT_COLUMN] for row in rows], dtype=torch.float32),
    }


def answer_segment_labels(input_ids: torch.Tensor, labels: torch.Tensor, think_end_token_id: int) -> torch.Tensor:
    """Return `labels` restricted to the tokens after each row's last `</think>` (all ignored for a row without one).
    """
    positions = torch.arange(input_ids.size(-1), device=input_ids.device).expand_as(input_ids)
    marked = torch.where(input_ids == think_end_token_id, positions, torch.full_like(positions, -1))
    last = marked.max(dim=-1, keepdim=True).values
    keep = (last >= 0) & (positions > last)
    return torch.where(keep, labels, torch.full_like(labels, IGNORE_INDEX))


def weighted_anchor_loss(
    *,
    policy_logits: torch.Tensor,
    base_logits: torch.Tensor | None,
    labels: torch.Tensor,
    weights: torch.Tensor,
    config: AnchorAuxConfig,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(weighted loss, raw loss)` of a replay batch: `mean_i(w_i * loss_weight * raw_i)` and `mean_i(raw_i)`.
    """
    if weights.dim() != 1 or weights.size(0) != labels.size(0):
        raise ValueError(f"weights must be one per row: {tuple(weights.shape)} for {labels.size(0)} rows")
    weighted_rows, raw_rows = [], []
    for row in range(labels.size(0)):
        weighted, raw = anchor_aux_loss(
            policy_logits=policy_logits[row : row + 1],
            base_logits=None if base_logits is None else base_logits[row : row + 1],
            labels=labels[row : row + 1],
            config=config,
            temperature=temperature,
        )
        weighted_rows.append(weighted * weights[row].to(weighted.dtype))
        raw_rows.append(raw)
    return torch.stack(weighted_rows).mean(), torch.stack(raw_rows).mean()


def replay_seed(seed: int, start_step: int, cycle: int) -> int:
    """Return the generator seed of one pass over the replay rows."""
    digest = hashlib.sha256(f"{ANCHOR_REPLAY_SEED_LABEL}\0{seed}\0{start_step}\0{cycle}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class ReplayRows(torch.utils.data.Dataset):
    """The replay rows, each checked when indexed and weighted to a mean of 1, a file's rows left on disk with only
    their byte offsets held."""

    def __init__(
        self, read: Callable[[int], Mapping[str, Any]], weights: array, *, source: str, sha256: Optional[str] = None
    ) -> None:
        if len(weights) == 0:
            raise ValueError("the anchor replay set is empty; an anchored run needs rows to replay")
        self._read = read
        self._size = len(weights)
        self.weight_scale = len(weights) / math.fsum(weights)
        self.source = source
        self.sha256 = sha256

    @classmethod
    def from_records(cls, rows: Sequence[Mapping[str, Any]]) -> "ReplayRows":
        rows = list(rows)
        weights = array("d", (_replay_row(row, index)[REPLAY_WEIGHT_COLUMN] for index, row in enumerate(rows)))
        return cls(rows.__getitem__, weights, source="records")

    @classmethod
    def from_jsonl(cls, path: Path | str, *, expected_sha256: Optional[str] = None) -> "ReplayRows":
        """Index a JSONL file of replay rows; `expected_sha256` must match when given."""
        path = Path(path)
        handle = path.open("rb")
        try:
            offsets, weights, digest, offset = array("q"), array("d"), hashlib.sha256(), 0
            for line in handle:
                digest.update(line)
                if line.strip():
                    row = _parse_replay_line(line, path, len(offsets))
                    weights.append(_replay_row(row, len(offsets))[REPLAY_WEIGHT_COLUMN])
                    offsets.append(offset)
                offset += len(line)
            if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
                raise ValueError(
                    f"{path} has sha256 {digest.hexdigest()}, not the {expected_sha256} its marker was checked "
                    "for: the file changed in between"
                )

            def read(index: int) -> dict[str, Any]:
                handle.seek(offsets[index])
                return _parse_replay_line(handle.readline(), path, index)

            rows = cls(read, weights, source=str(path), sha256=digest.hexdigest())
        except BaseException:
            handle.close()
            raise
        weakref.finalize(rows, handle.close)
        return rows

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = _replay_row(self._read(index), index)
        row[REPLAY_WEIGHT_COLUMN] *= self.weight_scale
        return row

    def summary(self) -> dict[str, Any]:
        return {"source": self.source, "sha256": self.sha256, "rows": self._size, "weight_scale": self.weight_scale}


def _parse_replay_line(line: bytes, path: Path, index: int) -> Mapping[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: replay row {index} is not JSON ({error.msg})") from error
    if not isinstance(row, dict):
        raise ValueError(f"{path}: replay row {index} is not a JSON object")
    return row


class AnchorReplayManager:
    """The anchor's replay rows, served one batch per anchor micro-step: a plain random sampler, sharded across ranks,
    reseeded from the run seed, the start step and the pass number (no sampler state is checkpointed)."""

    def __init__(
        self,
        rows: ReplayRows | Sequence[Mapping[str, Any]],
        *,
        config: AnchorAuxConfig,
        pad_token_id: int,
        seed: int,
        num_processes: int = 1,
        process_index: int = 0,
        rows_served: int = 0,
    ) -> None:
        if not isinstance(rows, ReplayRows):
            rows = ReplayRows.from_records(rows)
        if not 0 <= process_index < num_processes:
            raise ValueError(f"process {process_index} is not one of {num_processes}")
        self.weight_scale = rows.weight_scale
        self.config = config
        self.seed = int(seed)
        self.pad_token_id = int(pad_token_id)
        self.num_processes = num_processes
        self.process_index = process_index
        self.dataset = rows
        self.generator = torch.Generator()
        self.sampler = torch.utils.data.RandomSampler(self.dataset, generator=self.generator)
        batches = torch.utils.data.BatchSampler(self.sampler, batch_size=config.per_device_batch_size, drop_last=False)
        if num_processes > 1:
            from accelerate.data_loader import BatchSamplerShard

            batches = BatchSamplerShard(batches, num_processes=num_processes, process_index=process_index)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=batches,
            collate_fn=functools.partial(collate_replay_rows, pad_token_id=self.pad_token_id),
            num_workers=config.dataloader_num_workers,
        )
        self.start_step: int | None = None
        self.cycle = 0
        self.rows_served = int(rows_served)
        self._iterator: Any = None

    @property
    def started(self) -> bool:
        return self.start_step is not None

    def start(self, step: int) -> None:
        """(Re)start the stream at optimizer step `step`."""
        self.start_step = int(step)
        self.cycle = 0
        self._new_pass()

    def _new_pass(self) -> None:
        self.generator.manual_seed(replay_seed(self.seed, self.start_step, self.cycle))
        self._iterator = iter(self.dataloader)

    def next_batch(self) -> dict[str, torch.Tensor]:
        """Return the next batch of this rank's shard, starting a new pass when one ends."""
        if not self.started:
            raise RuntimeError("the anchor replay stream was not started; call start(step) first")
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.cycle += 1
            self._new_pass()
            batch = next(self._iterator)
        self.rows_served += int(batch["input_ids"].size(0))
        return batch

    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.dataset),
            "weight_scale": self.weight_scale,
            "seed": self.seed,
            "start_step": self.start_step,
            "cycle": self.cycle,
            "rows_served": self.rows_served,
            "num_processes": self.num_processes,
            "per_device_batch_size": self.config.per_device_batch_size,
            "sampler": type(self.sampler).__name__,
        }
