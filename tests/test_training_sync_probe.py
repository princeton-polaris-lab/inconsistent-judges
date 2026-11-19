from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from ruleforge.grpo import sync_probe as sp  # noqa: E402

pytestmark = pytest.mark.torch

TINY_QWEN3 = dict(
    vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=3, num_attention_heads=4,
    num_key_value_heads=2, head_dim=4, max_position_embeddings=64, tie_word_embeddings=False,
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PROBED = ["model.layers.0.self_attn.k_proj", "model.layers.1.mlp.up_proj", "model.layers.2.mlp.down_proj"]


def _model(seed: int = 0, b_std: float = 0.5):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(**TINY_QWEN3)
    base = transformers.Qwen3ForCausalLM(config).to(torch.bfloat16).float()  # bf16 weights, fp32 master
    model = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, lora_dropout=0.0))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0.0, b_std)
    return config, model


def _weight(model, path: str) -> torch.Tensor:
    return model.get_submodule(f"base_model.model.{path}").get_base_layer().weight


def _sync(model, config, *, plant=None) -> dict[str, torch.Tensor]:
    """TRL's colocated sync into a vLLM-packed bf16 engine: merge in place, cast, unmerge."""
    model.merge_adapter()
    engine = {}
    for layer in range(config.num_hidden_layers):
        attn, mlp = f"model.layers.{layer}.self_attn", f"model.layers.{layer}.mlp"
        merged = {leaf: _weight(model, f"{attn}.{leaf}").detach().clone() for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")}
        merged |= {leaf: _weight(model, f"{mlp}.{leaf}").detach().clone() for leaf in ("gate_proj", "up_proj", "down_proj")}
        engine[f"{attn}.qkv_proj.weight"] = torch.cat([merged["q_proj"], merged["k_proj"], merged["v_proj"]]).to(torch.bfloat16)
        engine[f"{attn}.o_proj.weight"] = merged["o_proj"].to(torch.bfloat16)
        engine[f"{mlp}.gate_up_proj.weight"] = torch.cat([merged["gate_proj"], merged["up_proj"]]).to(torch.bfloat16)
        engine[f"{mlp}.down_proj.weight"] = merged["down_proj"].to(torch.bfloat16)
    engine["model.norm.weight"] = model.get_submodule("base_model.model.model.norm").weight.detach().to(torch.bfloat16)
    model.unmerge_adapter()
    if plant is not None:
        plant(engine, model)
    return engine


def _probe(config, max_rows: int = 512) -> sp.SyncProbe:
    return sp.SyncProbe.for_modules(PROBED, config, max_rows=max_rows)


def _synced(probe, model, config, *, step=0, plant=None) -> sp.ProbeResult:
    probe.snapshot(model)
    return probe.check(model, _sync(model, config, plant=plant), step=step)


def test_the_packed_layout_is_vllms():
    config, _ = _model()
    q, kv, inter = 4 * 4, 2 * 4, 24
    assert sp.shard_sizes(config) == {"q_proj": q, "k_proj": kv, "v_proj": kv, "gate_proj": inter, "up_proj": inter}
    layer = "model.layers.1"
    assert sp.packed_layout(f"{layer}.self_attn.q_proj", config) == (f"{layer}.self_attn.qkv_proj.weight", 0)
    assert sp.packed_layout(f"{layer}.self_attn.k_proj", config) == (f"{layer}.self_attn.qkv_proj.weight", q)
    assert sp.packed_layout(f"{layer}.self_attn.v_proj", config) == (f"{layer}.self_attn.qkv_proj.weight", q + kv)
    assert sp.packed_layout(f"{layer}.mlp.up_proj", config) == (f"{layer}.mlp.gate_up_proj.weight", inter)
    assert sp.packed_layout(f"{layer}.mlp.down_proj", config) == (f"{layer}.mlp.down_proj.weight", 0)
    # a config without head_dim or GQA derives both
    older = types.SimpleNamespace(hidden_size=32, num_attention_heads=4, intermediate_size=10)
    assert sp.shard_sizes(older)["k_proj"] == 32


def test_the_probe_passes_on_a_correct_bf16_merge_and_moved_elements_differ_from_w():
    config, model = _model()
    probe = _probe(config)
    result = _synced(probe, model, config, step=3)
    assert result.ok and not result.vacuous and result.engine_dtype == "bfloat16"
    k_proj, up_proj, down_proj = result.points
    assert (k_proj.row_offset, up_proj.row_offset) == (16, 24)
    assert k_proj.elements == 8 * 16 and up_proj.elements == 24 * 16 and down_proj.elements == 16 * 24
    assert all(point.moved > 0 and point.mismatched == 0 and point.stale == 0 for point in result.points)
    assert 0 < result.retention <= 1.5 and probe.history == [result]
    assert result.to_dict()["points"][0]["module"] == PROBED[0]


def test_the_probe_stays_exact_over_many_merge_unmerge_cycles():
    """The unmerge leaves the fp32 master a few ulps off; the snapshot keeps the expectation exact."""
    config, model = _model(seed=1, b_std=0.05)
    probe = _probe(config)
    for step in range(12):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_A" in name:
                    parameter.add_(torch.randn_like(parameter) * 0.3)
        assert _synced(probe, model, config, step=step).ok


def test_the_probe_fails_on_a_planted_stale_weight():
    config, model = _model()

    def stale_k(engine, model):
        shard = engine["model.layers.0.self_attn.qkv_proj.weight"]
        shard[16:24] = _weight(model, "model.layers.0.self_attn.k_proj").detach().to(torch.bfloat16)

    probe = _probe(config)
    with pytest.raises(sp.SyncProbeError, match=r"k_proj .*equal to the base W") as raised:
        _synced(probe, model, config, step=7, plant=stale_k)
    failed = probe.history[-1]
    assert not failed.ok and failed.points[0].stale == failed.points[0].mismatched == failed.points[0].moved > 0
    assert failed.points[1].ok and "step 7" in str(raised.value)


def test_the_probe_fails_on_one_wrong_element_and_on_a_misplaced_shard():
    config, model = _model()

    def one_element(engine, model):
        engine["model.layers.1.mlp.gate_up_proj.weight"][30, 3] += 1.0

    with pytest.raises(sp.SyncProbeError, match=r"up_proj .*1 of 384 elements differ"):
        _synced(_probe(config), model, config, plant=one_element)

    def swapped(engine, model):
        shard = engine["model.layers.0.self_attn.qkv_proj.weight"]
        engine["model.layers.0.self_attn.qkv_proj.weight"] = torch.cat([shard[:16], shard[24:], shard[16:24]])

    with pytest.raises(sp.SyncProbeError, match="k_proj"):
        _synced(_probe(config), model, config, plant=swapped)

    def dropped_up(engine, model):
        engine["model.layers.1.mlp.gate_up_proj.weight"] = engine["model.layers.1.mlp.gate_up_proj.weight"][:30]

    with pytest.raises(sp.SyncProbeError, match="packed layout is wrong"):
        _synced(_probe(config), model, config, plant=dropped_up)


def test_a_zero_adapter_probe_is_vacuous_but_still_checks_equality():
    config, model = _model(b_std=0.0)
    result = _synced(_probe(config), model, config)
    assert result.ok and result.vacuous and result.retention is None
    assert all(point.retention is None and point.moved == 0 for point in result.points)


def test_rows_are_capped_by_the_shard_and_by_max_rows():
    config, model = _model()
    probe = _probe(config, max_rows=5)
    assert [point.rows for point in probe.points] == [5, 5, 5]
    result = _synced(probe, model, config)
    assert result.ok and result.points[0].elements == 5 * 16
    assert sp.ProbePoint.resolve("model.layers.0.self_attn.k_proj", config, max_rows=100).rows == 8
    with pytest.raises(sp.SyncProbeError, match="at least one row"):
        sp.ProbePoint.resolve(PROBED[0], config, max_rows=0)


def test_the_probe_refuses_what_it_cannot_check():
    config, model = _model()
    with pytest.raises(sp.SyncProbeError, match="at least one module"):
        sp.SyncProbe([])
    probe = _probe(config)
    with pytest.raises(sp.SyncProbeError, match="without a snapshot"):
        probe.check(model, {}, step=0)
    probe.snapshot(model)
    with pytest.raises(sp.SyncProbeError, match="no parameter"):
        probe.check(model, {}, step=0)
    with pytest.raises(sp.SyncProbeError, match="not a LoRA-wrapped module"):
        sp.SyncProbe.for_modules(["model.layers.0.input_layernorm"], config, max_rows=4).snapshot(model)
    bare = transformers.Qwen3ForCausalLM(config)
    with pytest.raises(sp.SyncProbeError, match="not a LoRA-wrapped module"):
        sp.SyncProbe.for_modules(["model.layers.0.self_attn.k_proj"], config, max_rows=4).snapshot(bare)


def test_a_plain_point_holds_the_engine_to_the_unwrapped_weight_exactly():
    config, model = _model()
    with torch.no_grad():
        model.get_submodule("base_model.model.model.norm").weight.normal_(1.0, 0.1)
    probe = sp.SyncProbe.for_modules(PROBED, config, max_rows=512, plain=["model.norm"])
    result = _synced(probe, model, config, step=4)
    norm = result.points[-1]
    assert result.ok and not result.vacuous and (norm.module, norm.lora, norm.elements) == ("model.norm", False, 16)
    assert norm.moved == norm.stale == 0 and norm.retention is None
    assert result.retention == pytest.approx(sum(point.retention for point in result.points[:3]) / 3)

    def discarded(engine, model):  # the sleep's freed memory, never refilled
        engine["model.norm.weight"].fill_(float("nan"))

    with pytest.raises(sp.SyncProbeError, match=r"model.norm \(model.norm.weight rows 0\+, plain\): 16 of 16"):
        _synced(probe, model, config, plant=discarded)
    assert sp.SyncProbe.for_modules([], config, max_rows=8, plain=["model.norm"]).points[0].rows == 8
    for module in ("model.layers.0.self_attn.k_proj", "model.no_such_norm", "model.layers.0.self_attn"):
        with pytest.raises(sp.SyncProbeError, match="not a plain"):
            sp.SyncProbe.for_modules([], config, max_rows=4, plain=[module]).snapshot(model)
    plain_only = sp.SyncProbe.for_modules([], config, max_rows=4, plain=["model.norm"])
    assert _synced(plain_only, model, config).vacuous


def test_the_delta_is_the_sum_over_the_active_adapters():
    calls = []

    class Layer:
        lora_A = {"default": None, "second": None}
        active_adapters = ["default", "missing", "second"]

        def get_delta_weight(self, name):
            calls.append(name)
            return torch.full((2, 2), 1.0 if name == "default" else 2.0)

        def get_base_layer(self):
            return types.SimpleNamespace(weight=torch.zeros(2, 2))

    assert torch.equal(sp._delta(Layer()), torch.full((2, 2), 3.0)) and calls == ["default", "second"]
    Layer.active_adapters = ["missing"]
    with pytest.raises(sp.SyncProbeError, match="no active adapter"):
        sp._delta(Layer())


def _result(step: int, *retentions) -> sp.ProbeResult:
    points = [sp.PointResult(f"m{i}", "p", 0, 10, 1, 0, 0, value, None) for i, value in enumerate(retentions)]
    return sp.ProbeResult(step=step, engine_dtype="bfloat16", points=points)


def test_the_sync_alarm_fires_once_at_the_first_sync_at_or_after_its_step():
    alarm = sp.RetentionAlarm(step=100, threshold=0.5)
    assert alarm.observe(_result(99, 0.1)) is None and not alarm.checked
    fired = alarm.observe(_result(100, 0.2, 0.4, None))
    assert fired["step"] == 100 and fired["retention"] == pytest.approx(0.3) and alarm.fired is fired
    assert fired["per_module"] == {"m0": 0.2, "m1": 0.4, "m2": None} and "you choose" in fired["action"]
    assert alarm.observe(_result(125, 0.1)) is None
    quiet = sp.RetentionAlarm(step=100, threshold=0.5)
    assert quiet.observe(_result(101, 0.7)) is None and quiet.checked and quiet.fired is None
    assert quiet.observe(_result(102, 0.1)) is None
    unknown = sp.RetentionAlarm()
    assert unknown.observe(_result(100, None)) is None and unknown.checked


def test_a_probed_sync_snapshots_syncs_checks_and_reports():
    config, model = _model()
    probe = _probe(config)
    seen, engine = [], {}

    def sync():
        engine.update(_sync(model, config))

    run = sp.probed_sync(sync, probe=probe, model=model, engine_params=lambda: engine, step=lambda: 42,
                         on_result=seen.append)
    run()
    assert len(seen) == 1 and seen[0].ok and seen[0].step == 42


def _disk(model, config) -> dict[str, torch.Tensor]:
    """The engine as the checkpoint on disk fills it: the base weights, as if the adapter were zero."""
    saved = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_B" in name}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in saved:
                parameter.zero_()
        engine = _sync(model, config)
        for name, parameter in model.named_parameters():
            if name in saved:
                parameter.copy_(saved[name])
    return engine


def test_the_engine_is_checked_unchanged_right_before_it_samples():
    config, model = _model()
    probe = _probe(config)
    seen, checks, engine = [], [], {}

    def sync():
        engine.update(_sync(model, config))

    sync = sp.probed_sync(sync, probe=probe, model=model, engine_params=lambda: engine, step=lambda: 7,
                          on_result=seen.append)
    generate = sp.checked_generate(lambda prompts, **kwargs: ("sampled", prompts, kwargs), probe=probe,
                                   engine_params=lambda: engine, step=lambda: 7, on_check=lambda: checks.append(1))
    with pytest.raises(sp.SyncProbeError, match="samples at step 7 with no accepted sync probe"):
        generate([1])
    sync()
    assert generate([1], use_tqdm=False) == ("sampled", [1], {"use_tqdm": False})
    assert probe.generation_checks == 1 == len(checks) and seen[0].ok and not seen[0].vacuous
    engine.update(_disk(model, config))  # wake_up + collective_rpc("reload_weights"), after the probed sync
    with pytest.raises(sp.SyncProbeError, match=r"changed between the sync probed at step 7 and sampling at step 7 "
                                                r".*k_proj \(model.layers.0.self_attn.qkv_proj.weight rows 16\+\): "
                                                r"(\d+) of 128 elements, \1 of them now equal to the base W"):
        generate([1])
    assert probe.generation_checks == 1
    sync()
    engine["model.layers.1.mlp.gate_up_proj.weight"].fill_(float("nan"))  # a sleep's discarded memory
    with pytest.raises(sp.SyncProbeError, match=r"up_proj .*384 of 384 elements, 0 of them"):
        generate([1])
    with pytest.raises(sp.SyncProbeError, match="no parameter"):
        probe.check_unchanged({}, step=8)
    probe.snapshot(model)
    with pytest.raises(sp.SyncProbeError, match="no parameter"):
        probe.check(model, {}, step=8)
    with pytest.raises(sp.SyncProbeError, match="no accepted sync probe"):
        probe.check_unchanged(engine, step=8)


def test_the_packed_shards_are_vllms_qwen3_mapping():
    """The installed vLLM's own mapper stacks q, k, v and gate, up in the order the probe reads (vllm group)."""
    qwen3 = pytest.importorskip("vllm.model_executor.models.qwen3", reason="needs the vllm group")
    model = qwen3.Qwen3ForCausalLM
    stacked = model.hf_to_vllm_mapper.orig_to_new_stacked
    assert {leaf: stacked[f".{leaf}"][0].lstrip(".") for leaf in sp.PACKED_SHARDS} == {
        leaf: packed for leaf, (packed, _) in sp.PACKED_SHARDS.items()}
    for packed, leaves in model.packed_modules_mapping.items():
        assert [sp.PACKED_SHARDS[leaf] for leaf in leaves] == [(packed, index) for index in range(len(leaves))]


def _shard_rows(tensor: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    size = tensor.size(0) // tp
    return tensor[rank * size : (rank + 1) * size]


def _shard_columns(tensor: torch.Tensor, tp: int, rank: int) -> torch.Tensor:
    size = tensor.size(1) // tp
    return tensor[:, rank * size : (rank + 1) * size]


def _sync_tp(model, config, tp: int, rank: int) -> dict[str, torch.Tensor]:
    """One rank's engine at tensor-parallel size ``tp``, as vLLM shards it."""
    whole = _sync(model, config)
    q, kv = 4 * 4, 2 * 4
    engine = {}
    for layer in range(config.num_hidden_layers):
        attn, mlp = f"model.layers.{layer}.self_attn", f"model.layers.{layer}.mlp"
        qkv = whole[f"{attn}.qkv_proj.weight"]
        engine[f"{attn}.qkv_proj.weight"] = torch.cat(
            [_shard_rows(part, tp, rank) for part in (qkv[:q], qkv[q : q + kv], qkv[q + kv :])])
        gate_up = whole[f"{mlp}.gate_up_proj.weight"]
        engine[f"{mlp}.gate_up_proj.weight"] = torch.cat(
            [_shard_rows(part, tp, rank) for part in (gate_up[:24], gate_up[24:])])
        engine[f"{attn}.o_proj.weight"] = _shard_columns(whole[f"{attn}.o_proj.weight"], tp, rank)
        engine[f"{mlp}.down_proj.weight"] = _shard_columns(whole[f"{mlp}.down_proj.weight"], tp, rank)
    engine["model.norm.weight"] = whole["model.norm.weight"]
    return engine


def test_each_tensor_parallel_rank_checks_its_own_engine_shard():
    config, model = _model()
    for rank in (0, 1):
        probe = sp.SyncProbe.for_modules(PROBED, config, max_rows=512, plain=["model.norm"], tp_size=2, tp_rank=rank)
        k_proj, up_proj, down_proj, norm = probe.points
        assert (k_proj.row_offset, k_proj.rows, k_proj.hf_row) == (8, 4, 4 * rank)
        assert (up_proj.row_offset, up_proj.rows, up_proj.hf_row) == (12, 12, 12 * rank)
        assert down_proj.hf_columns == (12 * rank, 12 * (rank + 1)) and norm.hf_columns is None
        probe.snapshot(model)
        result = probe.check(model, _sync_tp(model, config, 2, rank), step=rank)
        assert result.ok and not result.vacuous
        assert [point.elements for point in result.points] == [4 * 16, 12 * 16, 16 * 12, 16]
    probe = sp.SyncProbe.for_modules(PROBED, config, max_rows=512, tp_size=2, tp_rank=1)
    probe.snapshot(model)
    with pytest.raises(sp.SyncProbeError, match="k_proj"):
        probe.check(model, _sync_tp(model, config, 2, 0), step=0)


def test_a_tensor_parallel_layout_the_probe_cannot_place_is_refused():
    config, _ = _model()
    with pytest.raises(sp.SyncProbeError, match="rank 2 is not one of 2"):
        sp.ProbePoint.resolve(PROBED[0], config, max_rows=4, tp_size=2, tp_rank=2)
    with pytest.raises(sp.SyncProbeError, match="8 rows do not split over tensor-parallel size 3"):
        sp.ProbePoint.resolve(PROBED[0], config, max_rows=4, tp_size=3)
    with pytest.raises(sp.SyncProbeError, match="24 columns do not split over tensor-parallel size 5"):
        sp.ProbePoint.resolve(PROBED[2], config, max_rows=4, tp_size=5)
    with pytest.raises(sp.SyncProbeError, match="how vLLM shards a q_norm"):
        sp.ProbePoint.resolve("model.layers.0.self_attn.q_norm", config, max_rows=4, tp_size=2)
    # o_proj's input is the attention output: heads x head_dim
    assert sp.ProbePoint.resolve("model.layers.0.self_attn.o_proj", config, max_rows=4, tp_size=2,
                                 tp_rank=1).hf_columns == (8, 16)


def test_a_sharded_master_is_gathered_whole_for_the_check():
    whole = torch.arange(6.0)
    assert sp._whole(types.SimpleNamespace(full_tensor=lambda: whole)) is whole
    assert sp._whole(whole) is whole
    # a delta is laid out as its weight first (the merge's reduction), then gathered
    laid_out = []
    weight = types.SimpleNamespace(device_mesh="mesh", placements=("Shard(0)",))
    delta = types.SimpleNamespace(
        full_tensor=lambda: pytest.fail("gathered before its layout"),
        redistribute=lambda mesh, placements: laid_out.append((mesh, placements)) or types.SimpleNamespace(
            full_tensor=lambda: whole))
    assert sp._whole(delta, like=weight) is whole and laid_out == [("mesh", ("Shard(0)",))]
    assert sp._whole(types.SimpleNamespace(full_tensor=lambda: whole), like=torch.zeros(1)) is whole


def _gemma_model(b_std: float = 0.5, **text_changes):
    from tests.tiny_gemma import gemma_config

    torch.manual_seed(3)
    config = gemma_config(**text_changes)
    base = transformers.Gemma4ForConditionalGeneration(config).to(torch.bfloat16).float()
    targets = r"model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    model = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=targets, lora_dropout=0.0))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0.0, b_std)
    return config, model


def _gemma_sync(model, config) -> dict[str, torch.Tensor]:
    """TRL's sync into vLLM 0.28's Gemma 4."""
    model.merge_adapter()
    engine = {}
    for layer in range(2):
        hf, vllm = f"model.language_model.layers.{layer}", f"language_model.model.layers.{layer}"
        attn = {leaf: _weight(model, f"{hf}.self_attn.{leaf}").detach().clone() for leaf in ("q_proj", "k_proj")}
        attn["v_proj"] = (_weight(model, f"{hf}.self_attn.v_proj").detach().clone() if layer == 0
                          else attn["k_proj"])
        engine[f"{vllm}.self_attn.qkv_proj.weight"] = torch.cat(
            [attn["q_proj"], attn["k_proj"], attn["v_proj"]]).to(torch.bfloat16)
        mlp = {leaf: _weight(model, f"{hf}.mlp.{leaf}").detach().clone() for leaf in ("gate_proj", "up_proj",
                                                                                     "down_proj")}
        engine[f"{vllm}.mlp.gate_up_proj.weight"] = torch.cat([mlp["gate_proj"], mlp["up_proj"]]).to(torch.bfloat16)
        engine[f"{vllm}.mlp.down_proj.weight"] = mlp["down_proj"].to(torch.bfloat16)
    norm = model.get_submodule("base_model.model.model.language_model.norm").weight.detach()
    engine["language_model.model.norm.weight"] = norm.to(torch.bfloat16)
    model.unmerge_adapter()
    return engine


def test_the_probe_reads_gemma_4s_engine_names_and_its_full_attention_layers_own_shape():
    config, model = _gemma_model()
    assert sp.engine_name("model.language_model.layers.0.mlp.up_proj.weight") == (
        "language_model.model.layers.0.mlp.up_proj.weight")
    assert sp.engine_name("model.layers.0.mlp.up_proj.weight") == "model.layers.0.mlp.up_proj.weight"
    # the sliding layer's heads are 8 wide, the full-attention layer's 16 (global_head_dim), one KV head each
    assert sp.shard_sizes(config, 0)["q_proj"] == 16 and sp.shard_sizes(config, 1)["q_proj"] == 32
    modules = ["model.language_model.layers.0.self_attn.k_proj", "model.language_model.layers.1.self_attn.k_proj",
               "model.language_model.layers.1.mlp.up_proj", "model.language_model.layers.1.mlp.down_proj"]
    probe = sp.SyncProbe.for_modules(modules, config, max_rows=512, plain=["model.language_model.norm"])
    assert [(point.engine_param, point.row_offset) for point in probe.points[:2]] == [
        ("language_model.model.layers.0.self_attn.qkv_proj.weight", 16),
        ("language_model.model.layers.1.self_attn.qkv_proj.weight", 32),
    ]
    probe.snapshot(model)
    result = probe.check(model, _gemma_sync(model, config), step=1)
    assert result.ok and not result.vacuous and result.points[1].elements == 16 * 16


def test_the_engine_prefixes_are_vllms_gemma_4_mapping():
    gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm", reason="needs the vllm group")
    mapper = gemma4_mm.Gemma4ForConditionalGeneration.hf_to_vllm_mapper
    for name in ("model.language_model.layers.3.self_attn.k_proj.weight", "model.language_model.norm.weight"):
        assert mapper._map_name(name) == sp.engine_name(name)


GEMMA_PROBED = ["model.language_model.layers.0.self_attn.k_proj", "model.language_model.layers.1.self_attn.k_proj",
                "model.language_model.layers.1.mlp.up_proj", "model.language_model.layers.1.mlp.down_proj"]


def _gemma_sync_tp(whole, tp: int, rank: int) -> dict[str, torch.Tensor]:
    """One rank's engine of the tiny Gemma (2 query heads, 2 KV heads a layer) at tensor-parallel size ``tp``."""
    engine = {"language_model.model.norm.weight": whole["language_model.model.norm.weight"]}
    for layer, head_dim in ((0, 8), (1, 16)):
        prefix = f"language_model.model.layers.{layer}"
        qkv, rows = whole[f"{prefix}.self_attn.qkv_proj.weight"], 2 * head_dim
        engine[f"{prefix}.self_attn.qkv_proj.weight"] = torch.cat(
            [_shard_rows(part, tp, rank) for part in (qkv[:rows], qkv[rows : 2 * rows], qkv[2 * rows :])])
        gate_up = whole[f"{prefix}.mlp.gate_up_proj.weight"]
        engine[f"{prefix}.mlp.gate_up_proj.weight"] = torch.cat(
            [_shard_rows(part, tp, rank) for part in (gate_up[:32], gate_up[32:])])
        engine[f"{prefix}.mlp.down_proj.weight"] = _shard_columns(whole[f"{prefix}.mlp.down_proj.weight"], tp, rank)
    return engine


def test_gemma_4s_per_layer_shapes_split_over_tensor_parallel_ranks():
    """Each layer's own head size and KV heads (the sliding and the K=V full-attention layer) at TP 2."""
    config, model = _gemma_model(num_key_value_heads=2, num_global_key_value_heads=2)
    whole = _gemma_sync(model, config)
    for rank in (0, 1):
        probe = sp.SyncProbe.for_modules(GEMMA_PROBED, config, max_rows=512, plain=["model.language_model.norm"],
                                         tp_size=2, tp_rank=rank)
        assert [(point.row_offset, point.rows, point.hf_row) for point in probe.points[:2]] == [
            (8, 8, 8 * rank), (16, 16, 16 * rank)]  # q's rows at the layer's head size, then this rank's K rows
        probe.snapshot(model)
        result = probe.check(model, _gemma_sync_tp(whole, 2, rank), step=rank)
        assert result.ok and not result.vacuous
    probe = sp.SyncProbe.for_modules(GEMMA_PROBED, config, max_rows=512, tp_size=2, tp_rank=1)
    probe.snapshot(model)
    with pytest.raises(sp.SyncProbeError, match="layers.0.self_attn.k_proj"):
        probe.check(model, _gemma_sync_tp(whole, 2, 0), step=0)


def test_kv_heads_the_ranks_would_replicate_are_refused():
    """vLLM replicates KV heads fewer than the ranks (Gemma 4 31B-it's 4 global ones at TP 8)."""
    from tests.tiny_gemma import gemma_config

    config = gemma_config()  # one KV head a layer
    assert sp.attention_heads(config, 1) == {"q_proj": 2, "k_proj": 1, "v_proj": 1}
    for layer in (0, 1):
        with pytest.raises(sp.SyncProbeError, match="1 heads do not split over tensor-parallel size 2"):
            sp.ProbePoint.resolve(f"model.language_model.layers.{layer}.self_attn.k_proj", config, max_rows=4,
                                  tp_size=2)
    assert sp.ProbePoint.resolve("model.language_model.layers.1.self_attn.q_proj", config, max_rows=64, tp_size=2,
                                 tp_rank=1).hf_row == 16
