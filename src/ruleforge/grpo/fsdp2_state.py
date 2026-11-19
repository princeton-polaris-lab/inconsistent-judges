from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from ruleforge.core.errors import TracebackError

#: Under a checkpoint: the DCP directory of the sharded resume state, and its record.
STATE_DIRNAME = "fsdp2_state"
RECORD_FNAME = "state.json"
STATE_FORMAT = 1
ADAPTER, OPTIMIZER, SCHEDULER = "adapter", "optimizer", "scheduler"
FSDP1_REFUSAL = (
    "a run under FSDP1 does not resume from a checkpoint: transformers 5.17 restores its adapter through "
    "accelerate 1.13's adapter-only load_fsdp_model, which brings back rank 0's shard alone; only "
    "FSDP2 has a sharded resume state"
)


class Fsdp2StateError(RuntimeError, TracebackError):
    """A resume state that cannot be restored exactly, or a run that cannot load or save it."""


def configured_fsdp_version(fsdp_config: Optional[Mapping[str, Any]]) -> int:
    """Return the FSDP version an `fsdp_config` runs under, as transformers reads it (2 when unset)."""
    config = dict(fsdp_config or {})
    for key in list(config):
        if key.startswith("fsdp_"):
            config[key[len("fsdp_"):]] = config.pop(key)
    return int(config.get("version", 2))


def fsdp_version(args: Any) -> Optional[int]:
    """Return the FSDP version a TrainingArguments runs under, None without FSDP."""
    if not getattr(args, "fsdp", None):
        return None
    return configured_fsdp_version(getattr(args, "fsdp_config", None))


def rank0_model(model_path: Path | str, *, rank: int, dtype: str, attn_implementation: str) -> Any:
    """Return the checkpoint's model at `dtype`: loaded on rank 0, built on the meta device on every other rank with real
    buffers and the tied embedding tied."""
    import torch
    import transformers
    from accelerate import init_empty_weights

    from ruleforge.grpo.lora_parity import checkpoint_class

    config, model_class = checkpoint_class(Path(model_path))
    kwargs = {"dtype": getattr(torch, dtype), "attn_implementation": attn_implementation}
    if rank == 0:
        return model_class.from_pretrained(str(model_path), device_map=None, **kwargs)
    with init_empty_weights(include_buffers=False):
        model = model_class._from_config(config, **kwargs)
    model.tie_weights()  # the meta build leaves a tied head untied
    if (Path(model_path) / "generation_config.json").is_file():
        model.generation_config = transformers.GenerationConfig.from_pretrained(str(model_path))
    return model


def shard_from_rank0(accelerator: Any, model: Any) -> Any:
    """Shard `model` under the accelerator's FSDP2 plugin and fill every rank's shards from rank 0's values."""
    import torch.distributed as dist
    from accelerate.utils.fsdp_utils import fsdp2_prepare_model
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    full = model.state_dict() if dist.get_rank() == 0 else {}
    model = fsdp2_prepare_model(accelerator, model)
    set_model_state_dict(model, full, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
    del full
    model.tie_weights()
    return model


def state_dir(checkpoint: Path | str) -> Path:
    return Path(checkpoint) / STATE_DIRNAME


def adapter_state(model: Any) -> dict[str, Any]:
    """Return the trained parameters' sharded state dict (the adapter only); refused unless it names every trained
    parameter."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    trained = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {name: tensor for name, tensor in get_model_state_dict(model).items() if name in trained}
    if len(state) != len(trained):
        missing = sorted(trained - set(state))
        raise Fsdp2StateError(f"the sharded state dict names {len(state)} of the {len(trained)} trained parameters "
                              f"(not {', '.join(missing[:3])}): a wrapper renamed them, so the resume state would "
                              "not hold the adapter")
    return state


def _local(tensor: Any) -> Any:
    local = getattr(tensor, "to_local", None)
    return local() if local is not None else tensor


def _tensor_digest(digest: Any, name: str, tensor: Any) -> None:
    import torch

    local = _local(tensor).detach().to("cpu").contiguous()
    digest.update(f"{name}|{local.dtype}|{tuple(local.shape)}|".encode())
    digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())


def _digest(tree: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                walk(f"{prefix}/{key}", value[key])
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{prefix}/{index}", item)
        elif isinstance(value, torch.Tensor):
            _tensor_digest(digest, prefix, value)
        else:
            digest.update(f"{prefix}={json.dumps(value, default=str)};".encode())

    walk("", tree)
    return digest.hexdigest()


def _optimizer_state(model: Any, optimizer: Any) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    return get_optimizer_state_dict(model, optimizer)


def rank_digests(model: Any, optimizer: Any, scheduler: Any) -> dict[str, str]:
    """Return this rank's digests of its adapter shards, optimizer state and scheduler state."""
    return {ADAPTER: _digest(adapter_state(model)), OPTIMIZER: _digest(_optimizer_state(model, optimizer)),
            SCHEDULER: _digest(scheduler.state_dict())}


def save_state(checkpoint: Path | str, *, model: Any, optimizer: Any, scheduler: Any, step: int) -> Path:
    """Collectively write the sharded resume state under `checkpoint`, then on rank 0 its record with every rank's
    digests."""
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp

    directory = state_dir(checkpoint)
    dcp.save({ADAPTER: adapter_state(model), OPTIMIZER: _optimizer_state(model, optimizer),
              SCHEDULER: scheduler.state_dict()}, checkpoint_id=str(directory))
    digests: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(digests, rank_digests(model, optimizer, scheduler))
    write_record(directory, {"format": STATE_FORMAT, "world_size": dist.get_world_size(), "step": int(step),
                             "ranks": digests}, rank=dist.get_rank())
    dist.barrier()
    return directory


def write_record(directory: Path, record: Mapping[str, Any], *, rank: int) -> None:
    """Write the resume state's record on rank 0 alone."""
    from ruleforge.core.artifacts import write_text_atomically

    if rank == 0:
        write_text_atomically(Path(directory) / RECORD_FNAME, json.dumps(record, indent=1, sort_keys=True) + "\n")


def read_record(checkpoint: Path | str) -> Optional[dict[str, Any]]:
    """Return the resume state's record, None when the checkpoint holds none."""
    path = state_dir(checkpoint) / RECORD_FNAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _state_refusal(record: Optional[Mapping[str, Any]]) -> Optional[str]:
    if record is None:
        return (f"it holds no sharded resume state ({STATE_DIRNAME}/{RECORD_FNAME}: saved without FSDP, or "
                "with save_only_model), so its optimizer and every rank's adapter shard cannot be restored exactly")
    if record.get("format") != STATE_FORMAT:
        return f"its resume state is format {record.get('format')}, not {STATE_FORMAT}"
    return None


def resume_refusal(checkpoint: Path | str, fsdp_config: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return why a run under `fsdp_config` cannot resume exactly from `checkpoint` (FSDP1, no sharded state, another
    format), or None when it can."""
    if configured_fsdp_version(fsdp_config) != 2:
        return FSDP1_REFUSAL
    return _state_refusal(read_record(checkpoint))


def _agree(problem: Optional[str]) -> None:
    # Every rank raises together: a rank raising alone leaves the others blocked at the next collective.
    import torch.distributed as dist

    problems: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(problems, problem)
    found = [f"rank {rank}: {text}" for rank, text in enumerate(problems) if text]
    if found:
        raise Fsdp2StateError("; ".join(found))


def _checked_record(checkpoint: Path | str) -> dict[str, Any]:
    import torch.distributed as dist

    record = read_record(checkpoint)
    why = _state_refusal(record)
    if why is None and record["world_size"] != dist.get_world_size():
        why = f"it was saved on {record['world_size']} ranks, and this run has {dist.get_world_size()}"
    _agree(None if why is None else f"{checkpoint}: {why}")
    return record


def _check_digest(record: Mapping[str, Any], key: str, found: str) -> Optional[str]:
    import torch.distributed as dist

    saved = record["ranks"][dist.get_rank()][key]
    return None if found == saved else f"its restored {key} digests to {found[:16]}, not the saved {saved[:16]}"


def load_adapter(checkpoint: Path | str, model: Any, *, adapter_file: Path | str,
                 expect_adapter_sha256: Optional[str]) -> None:
    """Collectively restore the adapter's shards from the checkpoint's resume state, refused unless every rank's shards
    digest to what it saved and the gathered adapter equals `adapter_file`."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    record = _checked_record(checkpoint)
    state = {ADAPTER: adapter_state(model)}
    dcp.load(state, checkpoint_id=str(state_dir(checkpoint)))
    set_model_state_dict(model, state[ADAPTER], options=StateDictOptions(strict=False))
    _agree(_check_digest(record, ADAPTER, _digest(adapter_state(model))))
    _agree(adapter_file_problem(adapter_file, expect_adapter_sha256))
    _agree(adapter_mismatch(model, adapter_file))


def adapter_file_problem(adapter_file: Path | str, expect_sha256: Optional[str]) -> Optional[str]:
    """Return why `adapter_file` cannot be the checkpoint's adapter (missing, or another SHA-256), or None."""
    from ruleforge.core.digests import sha256_file

    path = Path(adapter_file)
    if not path.is_file():
        return f"{path} does not exist"
    if expect_sha256 is not None and sha256_file(path) != expect_sha256:
        return f"{path} is not the adapter the provenance records ({expect_sha256[:16]})"
    return None


def adapter_mismatch(model: Any, adapter_file: Path | str) -> Optional[str]:
    """Return, collectively, how the gathered adapter differs from `adapter_file`, or None when they agree."""
    import safetensors.torch
    import torch

    path = Path(adapter_file)
    saved = safetensors.torch.load_file(str(path))
    trained = {name.replace(".default.", "."): parameter for name, parameter in model.named_parameters()
               if parameter.requires_grad}
    wrong = []
    for key, parameter in trained.items():
        whole = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        if key not in saved or not torch.equal(whole.detach().to("cpu"), saved[key]):
            wrong.append(key)
    missing = sorted(set(saved) - set(trained))
    if wrong or missing:
        return (f"the restored adapter is not {path.name}: {len(wrong)} tensor(s) differ or are absent from it "
                f"({', '.join(wrong[:3])}), {len(missing)} of its tensor(s) name no trained parameter "
                f"({', '.join(missing[:3])})")
    return None


def load_optimizer(checkpoint: Path | str, *, model: Any, optimizer: Any, scheduler: Any) -> None:
    """Collectively restore the optimizer's and scheduler's state from the checkpoint after `load_adapter`, refused
    unless each rank's restored state digests to what it saved."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    record = _checked_record(checkpoint)
    state = {OPTIMIZER: _optimizer_state(model, optimizer), SCHEDULER: scheduler.state_dict()}
    dcp.load(state, checkpoint_id=str(state_dir(checkpoint)))
    set_optimizer_state_dict(model, optimizer, state[OPTIMIZER])
    scheduler.load_state_dict(state[SCHEDULER])
    found = rank_digests(model, optimizer, scheduler)
    problems = [_check_digest(record, key, found[key]) for key in (ADAPTER, OPTIMIZER, SCHEDULER)]
    _agree(next((problem for problem in problems if problem), None))
