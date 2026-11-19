from __future__ import annotations

import contextlib
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ruleforge.core.digests import _HEX64, sha256_file
from ruleforge.core.errors import RuleForgeError, SamplingDesignError
from ruleforge.core.record_names import WEIGHTS_DIGEST_VERSION
from ruleforge.judges.lora import ADAPTER_CONFIG_FNAME, VLLM_ADAPTER_WEIGHTS_FNAMES

MANIFEST_FNAME = "merge_manifest.json"
# Files a merge copies from the base verbatim when present: decoder defaults, tokenizer and processor configs.
SERVING_FILES = (
    "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "chat_template.json", "tokenizer.model",
    "processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json",
)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")
_READ_CHUNK = 16 * 1024 * 1024


class ParityError(ValueError, RuleForgeError):
    """Raised for inputs that cannot be compared, or an adapter that cannot be served."""


class MergeError(ValueError, RuleForgeError):
    """Raised for an adapter that does not match its declaration, or a merge that did not do what it must."""


def safetensors_layout(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    """Return a safetensors file's data offset and header (name -> {dtype, shape, data_offsets}), reading no data."""
    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ParityError(f"{path} is not a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return 8 + length, header


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Return a safetensors header (tensor name -> {dtype, shape, data_offsets}), reading no data."""
    return safetensors_layout(path)[1]


def _tensor_index(directory: Path) -> dict[str, tuple[Path, int, int, str, list[int]]]:
    files = sorted(Path(directory).glob("*.safetensors"))
    if not files:
        raise MergeError(f"{directory} holds no .safetensors file")
    index: dict[str, tuple[Path, int, int, str, list[int]]] = {}
    for path in files:
        start, header = safetensors_layout(path)
        for name, meta in header.items():
            if name in index:
                raise MergeError(f"tensor {name!r} is in both {index[name][0].name} and {path.name}")
            begin, end = meta["data_offsets"]
            index[name] = (path, start + begin, end - begin, str(meta["dtype"]), list(meta["shape"]))
    return index


def weights_digest(directory: Path | str) -> str:
    """Return a SHA-256 over every tensor's name, dtype, shape and bytes in name order; the shard layout does not
    enter."""
    index = _tensor_index(Path(directory))
    digest = hashlib.sha256(f"{WEIGHTS_DIGEST_VERSION}\n".encode("utf-8"))
    with contextlib.ExitStack() as stack:
        handles: dict[Path, Any] = {}
        for name in sorted(index):
            path, offset, size, dtype, shape = index[name]
            digest.update(json.dumps([name, dtype, shape, size]).encode("utf-8") + b"\n")
            if path not in handles:
                handles[path] = stack.enter_context(path.open("rb"))
            handle = handles[path]
            handle.seek(offset)
            remaining = size
            while remaining:
                chunk = handle.read(min(_READ_CHUNK, remaining))
                if not chunk:
                    raise MergeError(f"{path} ends inside tensor {name!r}")
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdapterIdentity:
    """A LoRA adapter as a campaign names it: ``label`` and the two digests are identity, the rest provenance."""

    path: str
    label: str
    sha256: str
    config_sha256: str
    rank: int
    base_model: Optional[str]


def adapter_identity(adapter_path: Path | str) -> AdapterIdentity:
    """Hash an adapter weights file and the ``adapter_config.json`` beside it; refuse a file vLLM would not serve."""
    path = Path(adapter_path)
    if not path.is_file():
        raise SamplingDesignError(
            f"adapter_path {path} must be an existing adapter weights FILE "
            "(e.g. .../checkpoint-125/adapter_model.safetensors), not a directory or a missing path"
        )
    resolved = path.resolve()
    served = next((name for name in VLLM_ADAPTER_WEIGHTS_FNAMES if (resolved.parent / name).is_file()), None)
    if resolved.name != served:
        raise SamplingDesignError(
            f"adapter_path {path} is not the weights file vLLM serves from {resolved.parent}: it "
            f"loads the first of {', '.join(VLLM_ADAPTER_WEIGHTS_FNAMES)} there ({served})"
        )
    config_path = resolved.parent / ADAPTER_CONFIG_FNAME
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SamplingDesignError(f"cannot read {config_path}: {error}") from error
    rank = config.get("r") if isinstance(config, Mapping) else None
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise SamplingDesignError(f"{config_path} carries no positive integer rank 'r' (got {rank!r})")
    base_model = config.get("base_model_name_or_path")
    return AdapterIdentity(
        path=str(resolved),
        label=f"{resolved.parent.name}/{resolved.name}",
        sha256=sha256_file(resolved),
        config_sha256=sha256_file(config_path),
        rank=rank,
        base_model=None if base_model is None else str(base_model),
    )


@dataclass(frozen=True)
class ModelWeightsIdentity:
    """Merged weights as a campaign names them: ``sha256`` is identity, the rest provenance."""

    path: str
    sha256: str
    digest_version: str
    manifest_sha256: str
    kind: Optional[str]
    saved_dtype: Optional[str]


# Where transformers finds further chat templates beside a checkpoint's own.
_EXTRA_CHAT_TEMPLATES_DIR = "additional_chat_templates"


def _derivation_manifest(directory: Path) -> Optional[tuple[Path, Mapping[str, Any]]]:
    manifest_path = directory / MANIFEST_FNAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SamplingDesignError(f"cannot read {manifest_path}: {error}") from error
    if not (isinstance(manifest, Mapping) and isinstance(manifest.get("result"), Mapping)):
        raise SamplingDesignError(f"{manifest_path} is not a derivation manifest (a JSON object with a result block)")
    recorded = manifest["result"].get("weights_sha256")
    version = manifest.get("weights_digest")
    if version != WEIGHTS_DIGEST_VERSION or not (isinstance(recorded, str) and _HEX64.match(recorded)):
        raise SamplingDesignError(
            f"{manifest_path} records no {WEIGHTS_DIGEST_VERSION} weights digest (weights_digest {version!r}, "
            f"result.weights_sha256 {recorded!r}); this build names derived weights by that digest only"
        )
    return manifest_path, manifest


def _weights_identity(directory: Path, manifest_path: Path, manifest: Mapping[str, Any]) -> ModelWeightsIdentity:
    merge = manifest.get("merge")
    saved = merge.get("saved_dtype") if isinstance(merge, Mapping) else None
    kind = manifest.get("kind")
    return ModelWeightsIdentity(
        path=str(directory),
        sha256=manifest["result"]["weights_sha256"],
        digest_version=manifest["weights_digest"],
        manifest_sha256=sha256_file(manifest_path),
        kind=None if kind is None else str(kind),
        saved_dtype=None if saved is None else str(saved),
    )


def model_weights_identity(model_path: Path | str) -> Optional[ModelWeightsIdentity]:
    """Return a weights directory's identity by content when it carries a merge manifest, else None; the recomputed
    weights digest and the files the engine reads beside them must be the recorded ones."""
    directory = Path(model_path).resolve()
    found = _derivation_manifest(directory)
    if found is None:
        return None
    manifest_path, manifest = found
    recorded = manifest["result"]["weights_sha256"]
    try:
        actual = weights_digest(directory)
    except (OSError, ValueError) as error:
        raise SamplingDesignError(f"cannot digest the weights in {directory}: {error}") from error
    if actual != recorded:
        raise SamplingDesignError(
            f"the weights in {directory} have digest {actual}, not the {recorded} its {MANIFEST_FNAME} records: a "
            "shard was changed or replaced after the merge, or the manifest is another merge's"
        )
    copied = manifest["result"].get("copied_files") or {}
    if not isinstance(copied, Mapping):
        raise SamplingDesignError(f"{manifest_path}: result.copied_files must map file names to SHA-256 digests")
    files = {"config.json": manifest["result"].get("config_sha256"), **copied}
    changed = sorted(
        name for name, digest in files.items()
        if digest is not None and not ((directory / name).is_file() and sha256_file(directory / name) == digest)
    )
    if changed:
        raise SamplingDesignError(
            f"{directory}: {changed} are not the files its {MANIFEST_FNAME} records (missing or changed after the "
            "merge); the engine would read them beside the weights"
        )
    if "copied_files" in manifest["result"]:
        unrecorded = sorted(
            name for name in (*SERVING_FILES, _EXTRA_CHAT_TEMPLATES_DIR)
            if name not in copied and (directory / name).exists()
        )
        if unrecorded:
            raise SamplingDesignError(
                f"{directory}: {unrecorded} are not among the files its {MANIFEST_FNAME} records (added after the "
                "merge); the engine or its tokenizer would read them, rendering or decoding otherwise than the base"
            )
    return _weights_identity(directory, manifest_path, manifest)


def recorded_model_weights(model_path: Path | str) -> Optional[ModelWeightsIdentity]:
    """Return what a weights directory's merge manifest records without recomputing the digest; None without one."""
    directory = Path(model_path).resolve()
    found = _derivation_manifest(directory)
    return None if found is None else _weights_identity(directory, *found)
