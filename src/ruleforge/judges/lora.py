from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# vLLM's default; a floor, since a lower-rank adapter loads under a higher ceiling.
DEFAULT_MAX_LORA_RANK = 16

ADAPTER_CONFIG_FNAME = "adapter_config.json"
# The weights files vLLM looks for in an adapter directory, in the order it tries them.
VLLM_ADAPTER_WEIGHTS_FNAMES = ("adapter_model.safetensors", "adapter_model.bin", "adapter_model.pt")


def resolve_lora_adapter_metadata(adapter_path: Path) -> tuple[Path, str, str]:
    """Return ``(adapter_dir, adapter_name, adapter_id)`` for an adapter file under
    ``<run_id>/<stage>/<adapter_dir>/<file>``. ``adapter_id`` is a readable label for logs, not the registry key."""
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise ValueError(f"Adapter file path does not exist: {adapter_path}")
    if not adapter_path.is_file():
        raise ValueError(
            "Adapter path must point to an adapter file (for example, "
            f".../final_adapter/adapter_model.safetensors). Received: {adapter_path}"
        )

    # By depth on the resolved path: ``.parent`` stops at the root, so a shallow path would name an unrelated ancestor.
    resolved = adapter_path.resolve()
    if len(resolved.parents) < 4:
        raise ValueError(
            "Unable to derive adapter ID: expected the training layout "
            "<run_id>/<stage>/<adapter_dir>/<file>, so at least two parent "
            f"directories above the adapter directory. Received: {adapter_path}"
        )
    adapter_dir = resolved.parent
    adapter_id = adapter_dir.parent.parent.name
    return adapter_dir, adapter_dir.name, adapter_id


def resolve_lora_max_rank(
    adapter_path: Path,
    floor: int = DEFAULT_MAX_LORA_RANK,
) -> int:
    """Return the ``max_lora_rank`` vLLM needs for this adapter: its config's ``r``, or ``floor`` when unreadable."""
    config_file = Path(adapter_path).parent / ADAPTER_CONFIG_FNAME
    try:
        with config_file.open(encoding="utf-8") as handle:
            rank = int(json.load(handle)["r"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Could not read LoRA rank from %s (%s); falling back to %d.",
            config_file,
            exc,
            floor,
        )
        return floor
    return max(floor, rank)


def stable_adapter_int_id(key: str) -> int:
    """Map a string key to a deterministic positive 31-bit integer (zero means "no adapter" to vLLM)."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1) or 1


def build_lora_request(adapter_path: Path) -> tuple[Any, dict[str, Any]]:
    """Build a vLLM ``LoRARequest`` (its int id keyed on the resolved adapter directory) plus metadata for the run
    record. vLLM compares requests by ``lora_name`` alone, so one engine must not serve two same-named adapters."""
    from vllm.lora.request import LoRARequest

    adapter_dir, adapter_name, adapter_id = resolve_lora_adapter_metadata(adapter_path)
    adapter_int_id = stable_adapter_int_id(str(adapter_dir))
    metadata = {
        "path": str(adapter_dir),
        "name": adapter_name,
        "id": adapter_id,
        "vllm_int_id": adapter_int_id,
        "max_lora_rank": resolve_lora_max_rank(adapter_path),
    }
    return LoRARequest(adapter_name, adapter_int_id, str(adapter_dir)), metadata


@dataclass(frozen=True)
class LoraServing:
    """The adapter one local engine serves: its request (None for none) and its record."""

    request: Any = None
    metadata: Optional[dict[str, Any]] = None

    @property
    def enable_lora(self) -> bool:
        """Return whether an adapter is served."""
        return self.request is not None

    @property
    def max_lora_rank(self) -> Optional[int]:
        """Return the served adapter's rank, or None without an adapter."""
        return None if self.metadata is None else int(self.metadata["max_lora_rank"])

    def engine_settings(self) -> dict[str, Any]:
        """Return the engine's LoRA arguments: ``enable_lora`` and ``max_lora_rank``."""
        return {"enable_lora": self.enable_lora, "max_lora_rank": self.max_lora_rank}


def lora_serving(adapter_path: Optional[Path]) -> LoraServing:
    """Return the adapter serving for ``adapter_path`` (none for None), resolved before the engine is built."""
    if adapter_path is None:
        return LoraServing()
    request, metadata = build_lora_request(Path(adapter_path))
    return LoraServing(request=request, metadata=metadata)
