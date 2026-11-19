from __future__ import annotations

import copy
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "training" / "grpo" / "gemma4_31b.yaml"
RANKS = 2


def _engine(pushed: dict, config) -> dict:
    """The pushed tensors as vLLM's Gemma 4 holds them at tensor-parallel size 1."""
    engine = {"language_model.model.norm.weight": pushed["model.language_model.norm.weight"].to(torch.bfloat16)}
    for layer in range(config.get_text_config().num_hidden_layers):
        hf, vllm = f"model.language_model.layers.{layer}", f"language_model.model.layers.{layer}"
        q, k = pushed[f"{hf}.self_attn.q_proj.weight"], pushed[f"{hf}.self_attn.k_proj.weight"]
        v = pushed.get(f"{hf}.self_attn.v_proj.weight", k)
        engine[f"{vllm}.self_attn.qkv_proj.weight"] = torch.cat([q, k, v]).to(torch.bfloat16)
        engine[f"{vllm}.mlp.gate_up_proj.weight"] = torch.cat(
            [pushed[f"{hf}.mlp.gate_proj.weight"], pushed[f"{hf}.mlp.up_proj.weight"]]).to(torch.bfloat16)
        engine[f"{vllm}.mlp.down_proj.weight"] = pushed[f"{hf}.mlp.down_proj.weight"].to(torch.bfloat16)
    return engine


def _layout(rank: int, model_dir: str, fsdp_config: dict, out_dir: str) -> None:
    import peft
    from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin
    from accelerate.utils.fsdp_utils import fsdp2_prepare_model
    from torch.distributed.tensor import DTensor
    from transformers.distributed.fsdp import is_fsdp_enabled
    from trl.distributed import DistributedBackend
    from trl.generation.vllm_generation import VLLMGeneration

    from ruleforge.grpo import trainer as tn
    from ruleforge.grpo.sync_probe import SyncProbe

    # the plugin transformers builds from the config's fsdp_config: every rank loads the checkpoint itself
    arguments = types.SimpleNamespace(fsdp=True, fsdp_config=dict(fsdp_config), gradient_checkpointing=False)
    plugin = FullyShardedDataParallelPlugin(**transformers.TrainingArguments._process_fsdp_args(arguments))
    assert not plugin.cpu_ram_efficient_loading and not is_fsdp_enabled()
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    preset = tn.lora_preset("Gemma4ForConditionalGeneration")
    model = peft.get_peft_model(model, preset.lora_config())
    # the same factors on every rank (FSDP2 keeps each rank's own shard of whatever that rank initialised)
    torch.manual_seed(7)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.normal_(0.0, 0.5)
    adapter = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name}
    reference = copy.deepcopy(model)
    reference.merge_adapter()
    merged = {name.removeprefix("base_model.model.").replace(".base_layer", ""): parameter.detach()
              for name, parameter in reference.named_parameters() if "lora_" not in name}

    accelerator = types.SimpleNamespace(
        state=types.SimpleNamespace(fsdp_plugin=plugin, deepspeed_plugin=None), mixed_precision="no",
        is_main_process=rank == 0, device=torch.device("cpu"), num_processes=RANKS, process_index=rank,
        is_fsdp2=True, distributed_type=DistributedType.FSDP, wait_for_everyone=torch.distributed.barrier)
    model = fsdp2_prepare_model(accelerator, model)
    state = model.state_dict()
    plain = sorted(name for name, tensor in state.items() if not isinstance(tensor, DTensor))
    assert plain == [f"base_model.model.model.language_model.layers.{layer}.layer_scalar" for layer in (0, 1)]
    assert all(torch.equal(state[name], torch.ones(1)) for name in plain)

    generation = object.__new__(VLLMGeneration)
    generation.model, generation.accelerator = model, accelerator
    generation._dist = DistributedBackend(accelerator)
    assert generation._dist.fsdp_version == 2
    original = generation._iter_fsdp2_params
    tn.install_fsdp2_sync(generation)
    config = model.get_base_model().config
    probe = SyncProbe.for_modules(tn.default_probe_modules(2, preset.decoder_prefix), config, max_rows=64,
                                  plain=["model.language_model.norm"])
    probe.snapshot(model)
    pushed = {name: tensor.detach().clone() for name, tensor in generation._iter_named_params()}
    assert set(pushed) == set(merged) | {"lm_head.weight"}  # the tied head: vLLM skips it
    for name, tensor in merged.items():
        torch.testing.assert_close(pushed[name], tensor, msg=name)
    result = probe.check(model, _engine(pushed, config), step=0)
    assert result.ok and not result.vacuous

    _checkpoint(rank, model, adapter, plugin, accelerator, out_dir)

    # TRL's own walk: full_tensor() on the buffer, at the first sync
    generation._iter_fsdp2_params = original
    with pytest.raises(AttributeError, match="full_tensor"):
        list(generation._iter_named_params())


def _checkpoint(rank: int, model, adapter: dict, plugin, accelerator, out_dir: str) -> None:
    """The checkpoint's adapter file is the whole adapter; the FSDP resume path restores rank 0's rows alone."""
    import logging

    import accelerate.utils.fsdp_utils as fsdp_utils
    import safetensors.torch
    from accelerate import Accelerator
    from accelerate.utils import load_fsdp_model, save_fsdp_model
    from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs

    # accelerate's own logger asks for an AcceleratorState, which the stand-in accelerator does not start
    fsdp_utils.logger = logging.getLogger(__name__)
    # Trainer.save_model under FULL_STATE_DICT: the full state dict (on rank 0), then the PEFT model's save
    state = Accelerator.get_state_dict(accelerator, model)
    if rank == 0:
        model.save_pretrained(out_dir, state_dict=state)
    torch.distributed.barrier()
    saved = safetensors.torch.load_file(str(Path(out_dir) / "adapter_model.safetensors"))
    assert set(saved) == {name.replace(".default", "") for name in adapter}
    for name, tensor in adapter.items():
        assert torch.equal(saved[name.replace(".default", "")], tensor), name

    # the resume: Trainer._save_optimizer_and_scheduler's adapter save, a fresh adapter, _load_from_checkpoint's load
    kwargs = get_fsdp_ckpt_kwargs()
    assert kwargs == {"adapter_only": True}
    save_fsdp_model(plugin, accelerator, model, out_dir, **kwargs)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.zero_()
    load_fsdp_model(plugin, accelerator, model, out_dir, **kwargs)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            first, *others = parameter.detach().full_tensor().chunk(RANKS)  # FSDP2 shards dim 0 as torch.chunk
            assert torch.equal(first, adapter[name].chunk(RANKS)[0]), name
            assert all(not other.any() for other in others), name


def _rank(rank: int, init_file: str, model_dir: str, fsdp_config: dict, out_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=RANKS)
    try:
        _layout(rank, model_dir, fsdp_config, out_dir)
    finally:
        dist.destroy_process_group()


def test_gemmas_fsdp2_layout_loads_shards_syncs_and_saves_on_two_ranks(tmp_path):
    from omegaconf import OmegaConf

    from tests.tiny_gemma import MODEL_NAME, save_tiny_gemma

    fsdp_config = OmegaConf.to_container(OmegaConf.load(CONFIG).grpo.fsdp_config)
    assert fsdp_config["cpu_ram_efficient_loading"] is False
    model_dir = save_tiny_gemma(tmp_path / MODEL_NAME)
    torch.multiprocessing.start_processes(
        _rank, args=(str(tmp_path / "rendezvous"), str(model_dir), fsdp_config, str(tmp_path / "checkpoint")),
        nprocs=RANKS, start_method="spawn")


#: The resume's steps: trained before the save, then the one compared.
TRAINED_STEPS = 3


def stand_in(rank: int, fsdp_config: dict):
    """The FSDP2 plugin transformers builds from ``fsdp_config``, as ``fsdp2_prepare_model`` reads it."""
    from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin

    arguments = types.SimpleNamespace(fsdp=True, fsdp_config=dict(fsdp_config), gradient_checkpointing=False)
    plugin = FullyShardedDataParallelPlugin(**transformers.TrainingArguments._process_fsdp_args(arguments))
    return types.SimpleNamespace(
        state=types.SimpleNamespace(fsdp_plugin=plugin, deepspeed_plugin=None), mixed_precision="no",
        is_main_process=rank == 0, device=torch.device("cpu"), num_processes=RANKS, process_index=rank,
        is_fsdp2=True, distributed_type=DistributedType.FSDP, wait_for_everyone=torch.distributed.barrier)


def wrapped(model, seed: int):
    """``model`` under the Gemma preset's LoRA, drawn at ``seed``, B drawn too (PEFT's is zero) where it is real."""
    import peft

    from ruleforge.grpo import trainer as tn

    torch.manual_seed(seed)
    model = peft.get_peft_model(model, tn.lora_preset("Gemma4ForConditionalGeneration").lora_config())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name and not parameter.is_meta:
                parameter.normal_(0.0, 0.5)
    return model


class TrainerBase:
    """The part of HF's Trainer the FSDP2 overrides reach."""

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        trained = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(trained, lr=1e-2, weight_decay=0.1, fused=True)
        self.lr_scheduler = transformers.get_scheduler("constant_with_warmup", self.optimizer, num_warmup_steps=2,
                                                       num_training_steps=max_steps)
        return self.model, train_dataloader


def fsdp2_trainer(rank: int, model_dir: str, fsdp_config: dict, *, seed: int, adapter_sha256=None):
    """The anchored trainer's FSDP2 half on :class:`TrainerBase`."""
    from ruleforge.grpo import fsdp2_state as fs
    from ruleforge.grpo import trainer as tn

    cls = tn.make_anchored_grpo_trainer_class(TrainerBase)
    trainer = cls.__new__(cls)
    trainer.model = trainer.model_wrapped = wrapped(
        fs.rank0_model(model_dir, rank=rank, dtype="float32", attn_implementation="sdpa"), seed)
    trainer.accelerator = stand_in(rank, fsdp_config)
    trainer.is_fsdp_enabled, trainer.rank0_master, trainer.resumed_adapter_sha256 = True, True, adapter_sha256
    trainer.state = types.SimpleNamespace(global_step=0)
    trainer._prepare_for_training(10, None, None)
    return trainer


def train_step(model, optimizer, scheduler, step: int) -> float:
    """One optimizer step on a fixed batch for ``step`` (backward, clip at 0.2, step)."""
    ids = torch.randint(10, 200, (2, 12), generator=torch.Generator().manual_seed(100 + step))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 0.2)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def local_state(model, optimizer=None) -> dict:
    """This rank's shards of every state-dict entry (buffers whole), and of the optimizer's state."""
    def local(tensor):
        return (tensor.to_local() if hasattr(tensor, "to_local") else tensor).detach().clone()

    found = {name: local(tensor) for name, tensor in model.state_dict().items()}
    for index, state in enumerate((optimizer.state if optimizer is not None else {}).values()):
        found.update({f"optimizer.{index}.{key}": local(value) for key, value in state.items()})
    return found


def assert_same(first: dict, second: dict) -> None:
    assert set(first) == set(second), set(first) ^ set(second)
    differing = [name for name in first if not torch.equal(first[name], second[name])]
    assert not differing, differing[:5]


def tied(model) -> bool:
    base = model.get_base_model()
    return base.lm_head.weight is base.model.language_model.embed_tokens.weight


def save_adapter_file(rank: int, trainer, directory: Path) -> str:
    """The checkpoint's adapter file as ``save_model`` writes it under ``FULL_STATE_DICT``; returns its SHA-256."""
    import logging

    import accelerate.utils.fsdp_utils as fsdp_utils
    from accelerate import Accelerator

    from ruleforge.core.digests import sha256_file

    fsdp_utils.logger = logging.getLogger(__name__)  # accelerate's own logger asks for an AcceleratorState
    state = Accelerator.get_state_dict(trainer.accelerator, trainer.model)
    if rank == 0:
        trainer.model.save_pretrained(str(directory), state_dict=state)
    torch.distributed.barrier()
    return sha256_file(directory / "adapter_model.safetensors")


def _one_rank_load(rank: int, model_dir: str, fsdp_config: dict) -> None:
    """Loading the master on rank 0 alone and broadcasting gives the every-rank load's shards, buffers and tie."""
    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

    every = wrapped(transformers.Gemma4ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32), 7)
    every = fsdp2_prepare_model(stand_in(rank, fsdp_config), every)
    trainer = fsdp2_trainer(rank, model_dir, fsdp_config, seed=7 + rank)
    assert trainer.rank0_master is False and tied(trainer.model) and tied(every)
    assert not any(tensor.is_meta for tensor in trainer.model.state_dict().values())
    assert_same(local_state(every), local_state(trainer.model))
    optimizer = torch.optim.AdamW([p for p in every.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1,
                                  fused=True)
    scheduler = transformers.get_scheduler("constant_with_warmup", optimizer, num_warmup_steps=2, num_training_steps=10)
    assert train_step(every, optimizer, scheduler, 0) == train_step(trainer.model, trainer.optimizer,
                                                                    trainer.lr_scheduler, 0)
    assert_same(local_state(every, optimizer), local_state(trainer.model, trainer.optimizer))


def _resume(rank: int, model_dir: str, fsdp_config: dict, out_dir: str) -> None:
    """N steps, the trainer's save, one more step."""
    import json
    import shutil

    from ruleforge.grpo import fsdp2_state as fs

    checkpoint = Path(out_dir) / "checkpoint-3"
    run = fsdp2_trainer(rank, model_dir, fsdp_config, seed=7)
    for step in range(TRAINED_STEPS):
        train_step(run.model, run.optimizer, run.lr_scheduler, step)
    run.state.global_step = TRAINED_STEPS
    adapter_sha256 = save_adapter_file(rank, run, checkpoint)
    run._save_optimizer_and_scheduler(str(checkpoint))
    record = fs.read_record(checkpoint)
    assert (record["format"], record["world_size"], record["step"], len(record["ranks"])) == (1, RANKS, 3, RANKS)
    saved = local_state(run.model, run.optimizer)
    uninterrupted = train_step(run.model, run.optimizer, run.lr_scheduler, TRAINED_STEPS)
    expected, rate = local_state(run.model, run.optimizer), run.lr_scheduler.get_last_lr()

    resumed = fsdp2_trainer(rank, model_dir, fsdp_config, seed=99, adapter_sha256=adapter_sha256)
    fresh = local_state(resumed.model)
    assert any(not torch.equal(fresh[name], saved[name]) for name in fresh if "lora_" in name)
    resumed._load_from_checkpoint(str(checkpoint), resumed.model_wrapped)
    resumed._load_optimizer_and_scheduler(str(checkpoint))
    assert_same(saved, local_state(resumed.model, resumed.optimizer))
    assert train_step(resumed.model, resumed.optimizer, resumed.lr_scheduler, TRAINED_STEPS) == uninterrupted
    assert_same(expected, local_state(resumed.model, resumed.optimizer))
    assert resumed.lr_scheduler.get_last_lr() == rate

    # another rank's shards contradict: an adapter file changed after the save, then a record's digest
    bad = Path(out_dir) / "checkpoint-bad"
    if rank == 0:
        shutil.copytree(checkpoint, bad)
        import safetensors.torch

        tensors = safetensors.torch.load_file(str(bad / "adapter_model.safetensors"))
        name = sorted(tensors)[0]
        tensors[name][-1] += 1.0  # the last row: the other rank's shard of a dim-0 split
        safetensors.torch.save_file(tensors, str(bad / "adapter_model.safetensors"))
    torch.distributed.barrier()
    refused = fsdp2_trainer(rank, model_dir, fsdp_config, seed=5)
    with pytest.raises(fs.Fsdp2StateError, match="the restored adapter is not adapter_model.safetensors"):
        refused._load_from_checkpoint(str(bad), refused.model_wrapped)
    torch.distributed.barrier()
    if rank == 0:
        shutil.copyfile(checkpoint / "adapter_model.safetensors", bad / "adapter_model.safetensors")
        path = fs.state_dir(bad) / fs.RECORD_FNAME
        record = json.loads(path.read_text())
        record["ranks"][1]["optimizer"] = "0" * 64
        path.write_text(json.dumps(record))
    torch.distributed.barrier()
    refused._load_from_checkpoint(str(bad), refused.model_wrapped)
    with pytest.raises(fs.Fsdp2StateError, match="^rank 1: its restored optimizer digests to .*, not the saved 0+$"):
        refused._load_optimizer_and_scheduler(str(bad))


def _load_and_resume_rank(rank: int, init_file: str, model_dir: str, fsdp_config: dict, out_dir: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=RANKS)
    try:
        _one_rank_load(rank, model_dir, fsdp_config)
        _resume(rank, model_dir, fsdp_config, out_dir)
    finally:
        dist.destroy_process_group()


def test_the_master_from_rank_0_and_a_sharded_resume_are_exact_on_two_ranks(tmp_path):
    from omegaconf import OmegaConf

    from tests.tiny_gemma import MODEL_NAME, save_tiny_gemma

    fsdp_config = OmegaConf.to_container(OmegaConf.load(CONFIG).grpo.fsdp_config)
    assert fsdp_config["version"] == 2
    model_dir = save_tiny_gemma(tmp_path / MODEL_NAME)
    torch.multiprocessing.start_processes(
        _load_and_resume_rank, args=(str(tmp_path / "rendezvous"), str(model_dir), fsdp_config, str(tmp_path / "out")),
        nprocs=RANKS, start_method="spawn")
