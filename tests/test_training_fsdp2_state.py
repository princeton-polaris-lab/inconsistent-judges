from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

from ruleforge.grpo import fsdp2_state as fs  # noqa: E402
from tests.test_training_fsdp2 import CONFIG, TrainerBase, local_state, train_step, wrapped  # noqa: E402

LORA = "lora_"

pytestmark = pytest.mark.torch


def _run_fsdp_config() -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(CONFIG).grpo.fsdp_config)


@pytest.fixture(scope="module")
def gemma_dir(tmp_path_factory) -> Path:
    from tests.tiny_gemma import MODEL_NAME, save_tiny_gemma

    return save_tiny_gemma(tmp_path_factory.mktemp("models") / MODEL_NAME)


@pytest.fixture(scope="module")
def one_rank(tmp_path_factory):
    """A process group of one gloo rank in this process, gone after the module."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{tmp_path_factory.mktemp('pg') / 'rendezvous'}", rank=0,
                            world_size=1)
    yield
    dist.destroy_process_group()


def _accelerator(monkeypatch):
    from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin

    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "false")
    arguments = types.SimpleNamespace(fsdp=True, fsdp_config=_run_fsdp_config(), gradient_checkpointing=False)
    plugin = FullyShardedDataParallelPlugin(**transformers.TrainingArguments._process_fsdp_args(arguments))
    return types.SimpleNamespace(
        state=types.SimpleNamespace(fsdp_plugin=plugin), mixed_precision="no", is_main_process=True,
        device=torch.device("cpu"), num_processes=1, process_index=0, distributed_type=DistributedType.FSDP)


def _sharded(gemma_dir, monkeypatch, seed=7):
    """A tiny Gemma loaded on rank 0, wrapped at ``seed`` and sharded, with the stand-in trainer's optimizer."""
    run = types.SimpleNamespace(model=wrapped(fs.rank0_model(gemma_dir, rank=0, dtype="float32",
                                                             attn_implementation="sdpa"), seed))
    run.model = fs.shard_from_rank0(_accelerator(monkeypatch), run.model)
    TrainerBase._prepare_for_training(run, 10, None, None)
    return run


def test_the_fsdp_version_a_run_shards_under_is_the_one_transformers_reads(monkeypatch):
    """transformers 5.17's reading (its plugin's version)."""
    monkeypatch.setenv("FSDP_CPU_RAM_EFFICIENT_LOADING", "false")  # _process_fsdp_args sets it
    assert fs.fsdp_version(types.SimpleNamespace(fsdp=None)) is None
    assert fs.fsdp_version(types.SimpleNamespace(fsdp="full_shard", fsdp_config=None)) == 2
    assert fs.fsdp_version(types.SimpleNamespace(fsdp=["full_shard"], fsdp_config={"version": 1})) == 1
    for config in ({}, {"version": 1}, {"fsdp_version": 1}, {"version": 2, "fsdp_version": 1}, _run_fsdp_config()):
        arguments = types.SimpleNamespace(fsdp=True, fsdp_config=dict(config), gradient_checkpointing=False)
        plugin = transformers.TrainingArguments._process_fsdp_args(arguments)
        assert fs.configured_fsdp_version(config) == plugin["fsdp_version"], config


def test_rank_0_loads_the_checkpoint_and_the_other_ranks_build_its_tree_on_meta(gemma_dir, tmp_path):
    loaded = fs.rank0_model(gemma_dir, rank=0, dtype="float32", attn_implementation="sdpa")
    built = fs.rank0_model(gemma_dir, rank=1, dtype="float32", attn_implementation="sdpa")
    assert type(built) is type(loaded) and built.name_or_path == loaded.name_or_path == str(gemma_dir)
    assert list(built.state_dict()) == list(loaded.state_dict())
    assert all(parameter.is_meta for parameter in built.parameters())
    assert not any(parameter.is_meta for parameter in loaded.parameters())
    assert all(not buffer.is_meta for buffer in built.buffers()) and all(
        parameter.dtype == torch.float32 for parameter in loaded.parameters())
    # a non-persistent buffer (the rotary tables) is computed on every rank, equal to rank 0's
    persistent = set(loaded.state_dict())
    for (name, buffer), (_, other) in zip(built.named_buffers(), loaded.named_buffers()):
        if name not in persistent:
            assert torch.equal(buffer, other), name
    for model in (built, loaded):
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert built.generation_config.to_dict() == loaded.generation_config.to_dict()
    assert built.config.get_text_config()._attn_implementation == "sdpa"
    bare = shutil.copytree(gemma_dir, tmp_path / "bare")
    (bare / "generation_config.json").unlink()
    assert fs.rank0_model(bare, rank=1, dtype="bfloat16", attn_implementation="eager").generation_config.top_k != 64


def test_the_master_is_sharded_from_rank_0_with_its_values_buffers_and_tie(gemma_dir, one_rank, monkeypatch):
    from torch.distributed.tensor import DTensor

    reference = wrapped(transformers.Gemma4ForConditionalGeneration.from_pretrained(gemma_dir, dtype=torch.float32), 7)
    expected = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}
    run = _sharded(gemma_dir, monkeypatch)
    state = run.model.state_dict()
    assert set(state) == set(expected)
    for name, tensor in state.items():
        assert isinstance(tensor, DTensor) or "layer_scalar" in name or "embed_scale" in name, name
        assert torch.equal(tensor.full_tensor() if isinstance(tensor, DTensor) else tensor, expected[name]), name
    base = run.model.get_base_model()
    assert base.lm_head.weight is base.model.language_model.embed_tokens.weight


def test_the_resume_state_saves_records_and_restores_exactly(gemma_dir, one_rank, monkeypatch, tmp_path):
    run = _sharded(gemma_dir, monkeypatch)
    for step in range(2):
        train_step(run.model, run.optimizer, run.lr_scheduler, step)
    checkpoint = tmp_path / "checkpoint-2"
    assert fs.read_record(checkpoint) is None
    assert fs.save_state(checkpoint, model=run.model, optimizer=run.optimizer, scheduler=run.lr_scheduler,
                         step=2) == fs.state_dir(checkpoint)
    record = fs.read_record(checkpoint)
    assert record == {"format": 1, "world_size": 1, "step": 2,
                      "ranks": [fs.rank_digests(run.model, run.optimizer, run.lr_scheduler)]}
    assert set(record["ranks"][0]) == {"adapter", "optimizer", "scheduler"}
    fs.write_record(tmp_path / "elsewhere", {"format": 1}, rank=1)
    assert not (tmp_path / "elsewhere").exists()
    run.model.save_pretrained(str(checkpoint), state_dict={name: tensor.full_tensor() for name, tensor in
                                                           run.model.state_dict().items() if "lora_" in name})
    adapter_file = checkpoint / "adapter_model.safetensors"
    from ruleforge.core.digests import sha256_file

    sha256 = sha256_file(adapter_file)
    saved = local_state(run.model, run.optimizer)
    fresh = _sharded(gemma_dir, monkeypatch, seed=99)
    fs.load_adapter(checkpoint, fresh.model, adapter_file=adapter_file, expect_adapter_sha256=sha256)
    fs.load_optimizer(checkpoint, model=fresh.model, optimizer=fresh.optimizer, scheduler=fresh.lr_scheduler)
    restored = local_state(fresh.model, fresh.optimizer)
    assert set(restored) == set(saved) and all(torch.equal(restored[name], saved[name]) for name in saved)
    assert fresh.lr_scheduler.state_dict() == run.lr_scheduler.state_dict()
    assert fs.resume_refusal(checkpoint, {"version": 2}) is None
    fs.load_adapter(checkpoint, fresh.model, adapter_file=adapter_file, expect_adapter_sha256=None)


def test_a_resume_state_that_cannot_be_restored_exactly_is_refused(gemma_dir, one_rank, monkeypatch, tmp_path):
    run = _sharded(gemma_dir, monkeypatch)
    train_step(run.model, run.optimizer, run.lr_scheduler, 0)
    checkpoint = tmp_path / "checkpoint-1"
    fs.save_state(checkpoint, model=run.model, optimizer=run.optimizer, scheduler=run.lr_scheduler, step=1)
    adapter = {name.replace(".default.", "."): tensor.full_tensor() for name, tensor in run.model.state_dict().items()
               if "lora_" in name}
    import safetensors.torch

    adapter_file = checkpoint / "adapter_model.safetensors"
    safetensors.torch.save_file(adapter, str(adapter_file))
    # at plan time: FSDP1 (however spelled), no state, another format; an unset version is FSDP2
    assert fs.resume_refusal(checkpoint, {"version": 1}) == fs.resume_refusal(checkpoint, {"fsdp_version": 1}) == (
        fs.FSDP1_REFUSAL)
    assert fs.resume_refusal(checkpoint, {}) is None and fs.resume_refusal(checkpoint, None) is None
    assert "holds no sharded resume state" in fs.resume_refusal(tmp_path, {"version": 2})
    record_path = fs.state_dir(checkpoint) / fs.RECORD_FNAME
    record = json.loads(record_path.read_text())
    with pytest.raises(fs.Fsdp2StateError, match="checkpoint-1: it was saved on 2 ranks, and this run has 1"):
        record_path.write_text(json.dumps({**record, "world_size": 2}))
        fs.load_adapter(checkpoint, run.model, adapter_file=adapter_file, expect_adapter_sha256=None)
    with pytest.raises(fs.Fsdp2StateError, match="its resume state is format 7, not 1"):
        record_path.write_text(json.dumps({**record, "format": 7}))
        fs.load_optimizer(checkpoint, model=run.model, optimizer=run.optimizer, scheduler=run.lr_scheduler)
    for key in ("adapter", "optimizer", "scheduler"):
        record_path.write_text(json.dumps({**record, "ranks": [{**record["ranks"][0], key: "0" * 64}]}))
        with pytest.raises(fs.Fsdp2StateError, match=f"^rank 0: its restored {key} digests to .*, not the saved 0+$"):
            fs.load_optimizer(checkpoint, model=run.model, optimizer=run.optimizer, scheduler=run.lr_scheduler)
        if key == "adapter":
            with pytest.raises(fs.Fsdp2StateError, match="^rank 0: its restored adapter digests to"):
                fs.load_adapter(checkpoint, run.model, adapter_file=adapter_file, expect_adapter_sha256=None)
    record_path.write_text(json.dumps(record))
    fresh = _sharded(gemma_dir, monkeypatch, seed=99)
    with pytest.raises(fs.Fsdp2StateError, match="nowhere.safetensors does not exist"):
        fs.load_adapter(checkpoint, fresh.model, adapter_file=tmp_path / "nowhere.safetensors",
                        expect_adapter_sha256=None)
    with pytest.raises(fs.Fsdp2StateError, match="is not the adapter the provenance records \\(0{16}\\)"):
        fs.load_adapter(checkpoint, fresh.model, adapter_file=adapter_file, expect_adapter_sha256="0" * 64)
    name = sorted(adapter)[0]
    safetensors.torch.save_file({**adapter, name: adapter[name] + 1, "extra.lora_A.weight": adapter[name]},
                                str(adapter_file))
    with pytest.raises(fs.Fsdp2StateError, match="1 tensor\\(s\\) differ or are absent from it .*1 of its tensor\\(s\\) "
                                                  "name no trained parameter \\(extra.lora_A.weight\\)"):
        fs.load_adapter(checkpoint, fresh.model, adapter_file=adapter_file, expect_adapter_sha256=None)
    # a wrapper that renames the module tree (activation checkpointing) would leave the adapter out of the state
    from torch.distributed.checkpoint import state_dict as dcp_state_dict

    real = dcp_state_dict.get_model_state_dict
    monkeypatch.setattr(dcp_state_dict, "get_model_state_dict", lambda model: {
        f"_checkpoint_wrapped_module.{name}": tensor for name, tensor in real(model).items()})
    with pytest.raises(fs.Fsdp2StateError, match="names 0 of the \\d+ trained parameters .*a wrapper renamed them"):
        fs.save_state(tmp_path / "checkpoint-2", model=run.model, optimizer=run.optimizer, scheduler=run.lr_scheduler,
                      step=2)


def test_the_adapter_file_check_reads_an_unsharded_adapter_too(gemma_dir, tmp_path):
    model = wrapped(transformers.Gemma4ForConditionalGeneration.from_pretrained(gemma_dir, dtype=torch.float32), 3)
    model.save_pretrained(str(tmp_path))
    assert fs.adapter_mismatch(model, tmp_path / "adapter_model.safetensors") is None
    with torch.no_grad():
        next(parameter for name, parameter in model.named_parameters() if "lora_B" in name).add_(1.0)
    assert "1 tensor(s) differ" in fs.adapter_mismatch(model, tmp_path / "adapter_model.safetensors")


def test_a_warm_start_loads_another_runs_saved_adapter_into_a_fresh_runs_shards(gemma_dir, one_rank, monkeypatch,
                                                                               tmp_path):
    from ruleforge.grpo import trainer as tn
    from ruleforge.core.digests import sha256_file

    run = _sharded(gemma_dir, monkeypatch)
    for step in range(2):
        train_step(run.model, run.optimizer, run.lr_scheduler, step)
    checkpoint = tmp_path / "checkpoint-2"
    fs.save_state(checkpoint, model=run.model, optimizer=run.optimizer, scheduler=run.lr_scheduler, step=2)
    saved = {name: tensor.full_tensor() for name, tensor in run.model.state_dict().items() if LORA in name}
    run.model.save_pretrained(str(checkpoint), state_dict=saved)
    init = {"checkpoint": str(checkpoint), "run": "round-1", "step": 2,
            "adapter_sha256": sha256_file(checkpoint / "adapter_model.safetensors")}

    def new_run(resume):
        cls = tn.make_anchored_grpo_trainer_class(TrainerBase)
        trainer = cls.__new__(cls)
        trainer.model = trainer.model_wrapped = wrapped(fs.rank0_model(gemma_dir, rank=0, dtype="float32",
                                                                       attn_implementation="sdpa"), 99)
        trainer.accelerator = _accelerator(monkeypatch)
        trainer.is_fsdp_enabled, trainer.rank0_master, trainer.init_adapter = True, True, init
        trainer._prepare_for_training(10, None, resume)
        return trainer

    def adapter(trainer):
        return {name: tensor.full_tensor() for name, tensor in trainer.model.state_dict().items() if LORA in name}

    warm = new_run(None)
    loaded = adapter(warm)
    assert set(loaded) == set(saved) and all(torch.equal(loaded[name], saved[name]) for name in saved)
    assert warm.warm_start_pending and not warm.optimizer.state and warm.lr_scheduler.get_last_lr() == [0.0]
    for step in range(2):  # the warmup's first rate is 0, its second is not
        train_step(warm.model, warm.optimizer, warm.lr_scheduler, step)
    assert warm.optimizer.state and any(not torch.equal(tensor, saved[name]) for name, tensor in adapter(warm).items())
    resumed = adapter(new_run("out/checkpoint-100"))
    assert any(not torch.equal(resumed[name], saved[name]) for name in saved)
