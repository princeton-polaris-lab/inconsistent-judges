from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
from safetensors.torch import load_file, save_file  # noqa: E402

from ruleforge.grpo import lora_parity as lp  # noqa: E402
from ruleforge.grpo import lora_targets as lt  # noqa: E402
from ruleforge.grpo import merging as mg  # noqa: E402

pytestmark = pytest.mark.torch

TINY_QWEN3 = dict(
    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
)
BASE_NAME = "Qwen3-tiny"
RANK, ALPHA = 4, 8
EXPECT = mg.MergeExpectation(rank=RANK, alpha=ALPHA, base_model=BASE_NAME)
TOKENIZER_BYTES = {name: json.dumps({"file": name}).encode() for name in ("tokenizer.json", "tokenizer_config.json",
                                                                          "vocab.json", "merges.txt")}
GENERATION_CONFIG = b'{"do_sample": true, "top_k": 20, "top_p": 0.95, "temperature": 0.6, "eos_token_id": [1, 2]}\n'


def _save_base(path: Path, *, seed: int = 0, config: dict | None = None) -> Path:
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**{**TINY_QWEN3, **(config or {})}))
    model.to(torch.bfloat16).save_pretrained(str(path))
    for name, data in TOKENIZER_BYTES.items():
        (path / name).write_bytes(data)
    (path / "generation_config.json").write_bytes(GENERATION_CONFIG)
    return path


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny random Qwen3ForCausalLM saved in bf16, with tokenizer files and a generation config."""
    return _save_base(tmp_path_factory.mktemp("bases") / BASE_NAME)


@pytest.fixture(scope="module")
def adapter_dir(base_dir, tmp_path_factory):
    """A PEFT-keyed random adapter over the 7 projections, in fp32, strong enough to survive a bf16 cast."""
    out = tmp_path_factory.mktemp("adapters") / "checkpoint-25"
    lp.make_random_adapter(base_dir, out, rank=RANK, alpha=ALPHA, seed=3, b_std=0.5)
    return out


def _copy_adapter(source: Path, dest: Path, *, config: dict | None = None, tensors=None) -> Path:
    dest.mkdir(parents=True)
    stored = load_file(str(source / lp.ADAPTER_WEIGHTS_FNAME))
    save_file(tensors(stored) if tensors else stored, str(dest / lp.ADAPTER_WEIGHTS_FNAME), metadata={"format": "pt"})
    settings = json.loads((source / lp.ADAPTER_CONFIG_FNAME).read_text())
    (dest / lp.ADAPTER_CONFIG_FNAME).write_text(json.dumps({**settings, **(config or {})}))
    return dest


def _stored(directory: Path) -> dict[str, "torch.Tensor"]:
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


def test_the_weights_digest_follows_the_tensors_not_the_shards_or_headers(base_dir, tmp_path):
    tensors = _stored(base_dir)
    names = sorted(tensors)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    save_file(tensors, str(one / "model.safetensors"), metadata={"format": "pt"})
    save_file({k: tensors[k] for k in names[::2]}, str(two / "a.safetensors"), metadata={"note": "x"})
    save_file({k: tensors[k] for k in names[1::2]}, str(two / "b.safetensors"))
    digest = mg.weights_digest(one)
    assert mg.weights_digest(two) == digest == mg.weights_digest(base_dir)
    assert len(digest) == 64
    # one changed value, one renamed tensor or one other dtype: another digest
    changed = dict(tensors)
    changed[names[0]] = changed[names[0]].clone()
    changed[names[0]].view(-1)[0] += 1
    for variant in (changed, {**{k: v for k, v in tensors.items() if k != names[0]}, "renamed": tensors[names[0]]},
                    {**tensors, names[0]: tensors[names[0]].float()}):
        other = tmp_path / f"v{len(list(tmp_path.iterdir()))}"
        other.mkdir()
        save_file(variant, str(other / "model.safetensors"))
        assert mg.weights_digest(other) != digest


def test_the_weights_digest_refuses_what_it_cannot_read_as_one_set_of_tensors(tmp_path):
    with pytest.raises(mg.MergeError, match="no .safetensors file"):
        mg.weights_digest(tmp_path)
    tensor = {"w": torch.ones(4)}
    save_file(tensor, str(tmp_path / "a.safetensors"))
    save_file(tensor, str(tmp_path / "b.safetensors"))
    with pytest.raises(mg.MergeError, match="'w' is in both a.safetensors and b.safetensors"):
        mg.weights_digest(tmp_path)
    (tmp_path / "b.safetensors").unlink()
    data = (tmp_path / "a.safetensors").read_bytes()
    (tmp_path / "a.safetensors").write_bytes(data[:-3])
    with pytest.raises(mg.MergeError, match="ends inside tensor 'w'"):
        mg.weights_digest(tmp_path)


def test_the_declared_adapter_passes_the_check(base_dir, adapter_dir):
    check = mg.check_adapter(adapter_dir, base_dir, EXPECT)
    # 3 layers x 7 projections, two factors each (Qwen3-8B: 36 x 7 = 252 modules, 504 tensors)
    assert len(check.modules) == 21 and check.tensors == 42
    assert check.modules["model.layers.0.mlp.down_proj"] == (16, 32)  # (out, in)
    assert (check.rank, check.alpha) == (RANK, 8.0)


def test_the_expectation_is_declared_not_empty():
    with pytest.raises(mg.MergeError, match="rank must be positive"):
        mg.MergeExpectation(rank=0, alpha=1, base_model="m")
    with pytest.raises(mg.MergeError, match="names no target module"):
        mg.MergeExpectation(rank=1, alpha=1, base_model="m", target_modules=())
    assert mg.MergeExpectation(rank=1, alpha=2, base_model="m", target_modules=["q_proj"]).target_modules == ("q_proj",)


def test_targets_must_name_linear_projections_of_the_base(base_dir):
    with pytest.raises(mg.MergeError, match="match no module"):
        mg.expected_modules(base_dir, ("nope",))
    with pytest.raises(mg.MergeError, match="not a linear projection"):
        mg.expected_modules(base_dir, ("input_layernorm",))


def _problems(adapter: Path, base: Path, expect: mg.MergeExpectation = EXPECT) -> str:
    with pytest.raises(mg.MergeError) as error:
        mg.check_adapter(adapter, base, expect)
    return str(error.value)


def test_a_non_fp32_adapter_is_refused(base_dir, adapter_dir, tmp_path):
    half = _copy_adapter(adapter_dir, tmp_path / "a", tensors=lambda t: {k: v.to(torch.bfloat16) for k, v in t.items()})
    assert "tensors in ['BF16'], not F32" in _problems(half, base_dir)


def test_an_adapter_for_another_base_is_refused(base_dir, adapter_dir, tmp_path):
    named = _copy_adapter(adapter_dir, tmp_path / "named", config={"base_model_name_or_path": "Qwen/Qwen3-4B"})
    assert "names base 'Qwen/Qwen3-4B', not 'Qwen3-tiny'" in _problems(named, base_dir)
    unnamed = _copy_adapter(adapter_dir, tmp_path / "unnamed", config={"base_model_name_or_path": None})
    assert "names base None" in _problems(unnamed, base_dir)
    wider = _save_base(tmp_path / "Qwen3-wide", config={"intermediate_size": 48})
    message = _problems(adapter_dir, wider)
    assert "the base directory" in message and "is not 'Qwen3-tiny'" in message
    assert "have factors of another shape" in message and "mlp.down_proj" in message


def test_rank_alpha_and_the_module_set_must_be_the_declared_ones(base_dir, adapter_dir, tmp_path):
    message = _problems(adapter_dir, base_dir, mg.MergeExpectation(rank=8, alpha=16, base_model=BASE_NAME))
    assert "r is 4, not the expected 8" in message and "lora_alpha is 8, not the expected 16.0" in message
    assert "factors of another shape" in message
    narrow = mg.MergeExpectation(rank=RANK, alpha=ALPHA, base_model=BASE_NAME, target_modules=("q_proj", "v_proj"))
    assert "15 adapted module(s) are not among the expected targets" in _problems(adapter_dir, base_dir, narrow)
    dropped = _copy_adapter(adapter_dir, tmp_path / "dropped",
                            tensors=lambda t: {k: v for k, v in t.items() if ".layers.2.mlp.up_proj." not in k})
    assert "1 expected module(s) are not adapted (e.g. model.layers.2.mlp.up_proj)" in _problems(dropped, base_dir)
    half = _copy_adapter(adapter_dir, tmp_path / "half",
                         tensors=lambda t: {k: v for k, v in t.items() if not k.endswith("layers.1.self_attn.q_proj.lora_B.weight")})
    message = _problems(half, base_dir)
    assert "model.layers.1.self_attn.q_proj: has lora_A only" in message and "another shape" not in message


def test_anything_but_plain_lora_factors_is_refused(base_dir, adapter_dir, tmp_path):
    extra = _copy_adapter(adapter_dir, tmp_path / "extra",
                          tensors=lambda t: {**t, "base_model.model.lm_head.weight": torch.zeros(64, 16)})
    message = _problems(extra, base_dir)
    assert "1 tensor(s) are not LoRA factors (e.g. base_model.model.lm_head.weight)" in message
    flagged = _copy_adapter(adapter_dir, tmp_path / "flagged", config={
        "use_dora": True, "use_rslora": True, "rank_pattern": {"q_proj": 8}, "modules_to_save": ["lm_head"],
    })
    message = _problems(flagged, base_dir)
    for key in ("use_dora is True", "use_rslora is True", "rank_pattern is set", "modules_to_save is set"):
        assert key in message
    for name in (lp.ADAPTER_CONFIG_FNAME, lp.ADAPTER_WEIGHTS_FNAME):
        missing = _copy_adapter(adapter_dir, tmp_path / f"no-{name}")
        (missing / name).unlink()
        assert f"has no {name}" in _problems(missing, base_dir)


@pytest.mark.parametrize("init", ["pissa", "pissa_niter_4", "olora", "loftq", "corda", "lora_ga", "orthogonal", "eva",
                                  "mica", 1])
def test_an_init_other_than_the_plain_ones_is_refused(base_dir, adapter_dir, tmp_path, init):
    adapter = _copy_adapter(adapter_dir, tmp_path / "a", config={"init_lora_weights": init})
    assert f"init_lora_weights is {init!r}, not one of [True, False, 'gaussian']" in _problems(adapter, base_dir)
    for plain in (False, "gaussian"):
        mg.check_adapter(_copy_adapter(adapter_dir, tmp_path / f"plain-{plain}", config={"init_lora_weights": plain}),
                         base_dir, EXPECT)


def test_a_base_weight_peft_rewrote_on_load_is_refused_whatever_the_config_says(base_dir, adapter_dir, tmp_path,
                                                                              monkeypatch):
    """PiSSA's init replaces W by its SVD residual when PEFT loads the adapter."""
    pissa = _copy_adapter(adapter_dir, tmp_path / "pissa", config={"init_lora_weights": "pissa"})
    monkeypatch.setattr(mg, "_PLAIN_INITS", (*mg._PLAIN_INITS, "pissa"))
    with pytest.raises(mg.MergeError, match=r"21 wrapped layer\(s\) no longer hold the base's weight"):
        mg.merge_adapter(base_dir, pissa, tmp_path / "out", expect=EXPECT, threads=1)
    assert not (tmp_path / "out").exists()
    assert mg._rewritten_base_weights({"model.nope": None}, base_dir) == ["model.nope (no model.nope.weight in the base)"]


def test_what_peft_builds_is_rechecked_layer_by_layer(base_dir):
    """rsLoRA scales by alpha/sqrt(r) and DoRA merges another way: the config check is backed by PEFT's layers."""
    from peft import LoraConfig, get_peft_model

    check = mg.AdapterCheck(modules={}, tensors=0, rank=RANK, alpha=float(ALPHA))

    def layers(**settings):
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**TINY_QWEN3))
        peft_model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["q_proj"], **settings))
        return {name: module for name, module in peft_model.base_model.model.named_modules()
                if name.endswith("q_proj") and hasattr(module, "lora_A")}

    assert mg._variant_problems(layers(), check) == []
    assert "scaling 4.0, not alpha/r = 2.0" in mg._variant_problems(layers(use_rslora=True), check)[0]
    assert "variant DoraLinearVariant" in mg._variant_problems(layers(use_dora=True), check)[0]
    plain = layers()
    for layer in plain.values():
        layer.fan_in_fan_out = True
    assert "transposed factors or a LoRA bias" in mg._variant_problems(plain, check)[0]


def _independent_merge(base_dir: Path, adapter_dir: Path) -> dict[str, "torch.Tensor"]:
    """W + (B.A).alpha/r for every adapted weight, from the files, in fp32; every other tensor upcast."""
    merged = {name: tensor.float() for name, tensor in _stored(base_dir).items()}
    factors = load_file(str(adapter_dir / lp.ADAPTER_WEIGHTS_FNAME))
    for key, a in factors.items():
        if key.endswith(".lora_A.weight"):
            path = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
            b = factors[key.replace("lora_A", "lora_B")]
            merged[f"{path}.weight"] = merged[f"{path}.weight"] + (b @ a) * (ALPHA / RANK)
    return merged


@pytest.fixture(scope="module")
def merged(base_dir, adapter_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("merges") / "checkpoint-25"
    return out, mg.merge_adapter(base_dir, adapter_dir, out, expect=EXPECT, threads=1, training_run="grpo-test")


def test_the_fp32_merge_is_w_plus_ba_times_alpha_over_r(base_dir, adapter_dir, merged):
    out, _ = merged
    saved, expected = _stored(out), _independent_merge(base_dir, adapter_dir)
    assert set(saved) == set(expected)
    assert not any("lora_" in name for name in saved)
    for name, tensor in saved.items():
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, expected[name]), name
    name = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(saved[name], _stored(base_dir)[name].float())


def test_the_merged_directory_keeps_the_class_and_serves_as_the_base_does(base_dir, merged):
    out, _ = merged
    config, base_config = (json.loads((d / "config.json").read_text()) for d in (out, base_dir))
    assert config["architectures"] == base_config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["model_type"] == "qwen3" and config["dtype"] == "float32"
    assert (out / "generation_config.json").read_bytes() == GENERATION_CONFIG
    for name, data in TOKENIZER_BYTES.items():
        assert (out / name).read_bytes() == data
    assert not out.with_name(out.name + ".partial").exists()
    loaded = transformers.Qwen3ForCausalLM.from_pretrained(str(out))
    assert loaded.dtype == torch.float32


def test_the_manifest_records_what_was_merged_how_and_what_came_out(base_dir, adapter_dir, merged):
    out, manifest = merged
    assert json.loads((out / mg.MANIFEST_FNAME).read_text()) == manifest
    assert manifest["base"]["weights_sha256"] == mg.weights_digest(base_dir)
    assert manifest["result"]["weights_sha256"] == mg.weights_digest(out)
    identity = lp.ADAPTER_WEIGHTS_FNAME
    from ruleforge.judges.weights import adapter_identity

    campaign = adapter_identity(adapter_dir / identity)
    assert (manifest["adapter"]["sha256"], manifest["adapter"]["config_sha256"]) == (campaign.sha256, campaign.config_sha256)
    assert (manifest["adapter"]["modules"], manifest["adapter"]["tensors"]) == (21, 42)
    assert manifest["training"] == {"run": "grpo-test", "step": None}
    merge = manifest["merge"]
    assert (merge["merge_dtype"], merge["saved_dtype"], merge["rounding"], merge["threads"]) == ("float32", "float32", None, 1)
    assert merge["fp32_matmul_precision"] == "ieee"
    assert merge["cpu_capability"] and set(merge["versions"]) == {"torch", "transformers", "peft", "safetensors"}
    assert set(merge["code"]) == {"code_revision", "code_dirty", "code_state_sha256"}
    result = manifest["result"]
    assert (result["modules_merged"], result["tensors"]) == (21, 36)
    from ruleforge.core.digests import sha256_file

    assert result["config_sha256"] == sha256_file(out / "config.json")
    assert set(result["copied_files"]) == {"generation_config.json", *TOKENIZER_BYTES}
    telemetry = result["bf16_cast"]
    assert set(telemetry["per_projection"]) == set(lp.DEFAULT_TARGET_MODULES)
    assert telemetry["overall"]["modules"] == 21 and 0.9 < telemetry["overall"]["retention"] < 1.1
    assert manifest["expectation"] == {"rank": RANK, "alpha": 8.0, "base_model": BASE_NAME,
                                       "target_modules": list(lp.DEFAULT_TARGET_MODULES)}


def test_the_campaign_names_a_merged_directory_by_the_digest_its_manifest_records(merged):
    from ruleforge.judges.weights import model_weights_identity

    out, manifest = merged
    identity = model_weights_identity(out)
    assert identity.sha256 == manifest["result"]["weights_sha256"] == mg.weights_digest(out)
    assert (identity.kind, identity.saved_dtype, identity.digest_version) == (
        "lora_merge", "float32", mg.WEIGHTS_DIGEST_VERSION
    )


def test_two_merges_on_one_machine_give_one_digest(base_dir, adapter_dir, merged, tmp_path, monkeypatch):
    # the second in a process that lowered oneDNN's fp32 matmuls to bf16: the merge still runs in IEEE fp32
    matmul = torch.backends.mkldnn.matmul
    monkeypatch.setattr(matmul, "fp32_precision", "bf16")
    again = mg.merge_adapter(base_dir, adapter_dir, tmp_path / "again", expect=EXPECT, threads=1)
    assert again["result"]["weights_sha256"] == merged[1]["result"]["weights_sha256"]
    assert again["merge"]["fp32_matmul_precision"] == "ieee" and matmul.fp32_precision == "bf16"
    assert again["created_at"] and again["training"]["run"] is None


def test_the_fallbacks_bf16_cast_is_the_rounded_fp32_merge(base_dir, adapter_dir, tmp_path):
    out = tmp_path / "bf16"
    manifest = mg.merge_adapter(base_dir, adapter_dir, out, expect=EXPECT, threads=1, cast_bf16=True)
    expected = _independent_merge(base_dir, adapter_dir)
    for name, tensor in _stored(out).items():
        assert tensor.dtype == torch.bfloat16
        assert torch.equal(tensor, expected[name].to(torch.bfloat16)), name  # round to nearest even
    assert json.loads((out / "config.json").read_text())["dtype"] == "bfloat16"
    assert (manifest["merge"]["saved_dtype"], manifest["merge"]["rounding"]) == ("bfloat16", mg.BF16_ROUNDING)


def test_the_bf16_cast_telemetry_counts_what_rounds_away():
    tally = mg.RetentionTally()
    weight = torch.ones(4, 4)
    tally.add("q_proj", weight, torch.full((4, 4), 1e-4))  # far under half an ulp of 1 in bf16: all of it lost
    tally.add("q_proj", weight, torch.full((4, 4), 0.5))  # on the bf16 grid: all of it kept
    tally.add("v_proj", weight, torch.zeros(4, 4))
    summary = tally.summary()
    q = summary["per_projection"]["q_proj"]
    assert q["modules"] == 2 and q["nonzero_elements"] == 32 and q["rounded_away"] == 0.5
    assert q["retention"] == pytest.approx(0.25 / (0.25 + 1e-8))
    assert summary["per_projection"]["v_proj"] == {"modules": 1, "retention": None, "rounded_away": None,
                                                   "nonzero_elements": 0}
    assert summary["overall"]["modules"] == 3 and summary["overall"]["rounded_away"] == 0.5
    assert mg.RetentionTally().summary()["overall"] == {"modules": 0, "retention": None, "rounded_away": None,
                                                        "nonzero_elements": 0}


def test_a_merge_refuses_before_loading_what_it_could_not_serve(base_dir, adapter_dir, merged, tmp_path):
    out, _ = merged
    with pytest.raises(mg.MergeError, match="exists: a merge never overwrites"):
        mg.merge_adapter(base_dir, adapter_dir, out, expect=EXPECT, threads=1)
    (tmp_path / "x.partial").mkdir()
    with pytest.raises(mg.MergeError, match="x.partial exists"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "x", expect=EXPECT, threads=1)
    with pytest.raises(mg.MergeError, match="is not the expected 00"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "y", expect=EXPECT, threads=1, expect_base_sha256="00")
    with pytest.raises(mg.MergeError, match="threads must be positive"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "z", expect=EXPECT, threads=0)
    bare = tmp_path / BASE_NAME
    bare.mkdir()
    for path in base_dir.iterdir():
        if path.name not in TOKENIZER_BYTES:
            (bare / path.name).write_bytes(path.read_bytes())
    with pytest.raises(mg.MergeError, match="could not be served"):
        mg.merge_adapter(bare, adapter_dir, tmp_path / "w", expect=EXPECT, threads=1)


def test_a_base_without_a_generation_config_gets_none_and_the_step_comes_from_the_trainer(adapter_dir, tmp_path):
    base = _save_base(tmp_path / "bases" / BASE_NAME)
    (base / "generation_config.json").unlink()
    adapter = _copy_adapter(adapter_dir, tmp_path / "checkpoint-50")
    (adapter / mg.TRAINER_STATE_FNAME).write_text(json.dumps({"global_step": 50}))
    manifest = mg.merge_adapter(base, adapter, tmp_path / "out", expect=EXPECT, threads=2)
    assert not (tmp_path / "out" / "generation_config.json").exists()
    assert manifest["training"]["step"] == 50 and manifest["merge"]["threads"] == 2
    assert "generation_config.json" not in manifest["result"]["copied_files"]
    assert mg.training_step(tmp_path) is None
    (adapter / mg.TRAINER_STATE_FNAME).write_text("{}")
    assert mg.training_step(adapter) is None


def test_a_non_finite_base_value_is_refused_after_the_merge(adapter_dir, tmp_path):
    base = _save_base(tmp_path / "bases" / BASE_NAME)
    tensors = load_file(str(base / "model.safetensors"))
    tensors["model.norm.weight"] = tensors["model.norm.weight"].clone()
    tensors["model.norm.weight"][0] = float("inf")
    save_file(tensors, str(base / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(mg.MergeError, match=r"non-finite values after the merge in \['model.norm.weight'\]"):
        mg.merge_adapter(base, adapter_dir, tmp_path / "out", expect=EXPECT, threads=1)


def test_what_peft_did_not_load_or_merge_is_refused(base_dir, adapter_dir, tmp_path, monkeypatch):
    import peft

    real_state = peft.get_peft_model_state_dict

    def zeroed(model):  # a factor PEFT could not place stays at its init
        state = real_state(model)
        key = sorted(state)[0]
        return {**state, key: torch.zeros_like(state[key])}

    monkeypatch.setattr(peft, "get_peft_model_state_dict", zeroed)
    with pytest.raises(mg.MergeError, match="did not load the adapter's factors as stored"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "a", expect=EXPECT, threads=1)
    monkeypatch.setattr(peft, "get_peft_model_state_dict", lambda model: {**real_state(model), "extra": torch.ones(1)})
    with pytest.raises(mg.MergeError, match=r"as stored: \['extra'\]"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "b", expect=EXPECT, threads=1)
    monkeypatch.undo()
    # a check that expects other modules than PEFT wraps
    real_check = mg.check_adapter
    monkeypatch.setattr(mg, "check_adapter", lambda *a: types.SimpleNamespace(
        **{**real_check(*a).__dict__, "modules": {"model.layers.0.self_attn.q_proj": (16, 16)}}))
    with pytest.raises(mg.MergeError, match="PEFT wrapped 21 module"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "c", expect=EXPECT, threads=1)
    monkeypatch.undo()
    # a variant PEFT built that the config check let through
    monkeypatch.setattr(mg, "_variant_problems", lambda layers, check: ["x: variant"])
    with pytest.raises(mg.MergeError, match="not W \\+ \\(B.A\\).alpha/r"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "d", expect=EXPECT, threads=1)
    monkeypatch.undo()
    monkeypatch.setattr(peft.PeftModel, "merge_and_unload", lambda self, **kw: self, raising=False)
    with pytest.raises(mg.MergeError, match="LoRA modules are left after the merge"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "e", expect=EXPECT, threads=1)


def test_a_saved_config_that_changed_the_class_or_the_dtype_is_refused(base_dir, adapter_dir, tmp_path, monkeypatch):
    real = transformers.PreTrainedModel.save_pretrained

    def renamed(self, directory, **kwargs):
        real(self, directory, **kwargs)
        path = Path(directory) / "config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "architectures": ["Other"]}))

    monkeypatch.setattr(transformers.PreTrainedModel, "save_pretrained", renamed)
    with pytest.raises(mg.MergeError, match=r"not the base's config in float32: \{'architectures': \(\['Other'\]"):
        mg.merge_adapter(base_dir, adapter_dir, tmp_path / "out", expect=EXPECT, threads=1)


def test_the_config_is_compared_as_transformers_parses_it(base_dir, tmp_path):
    """Re-serialization (rope_theta -> rope_parameters) is no drift; a changed field is."""
    saved = tmp_path / "saved"
    saved.mkdir()
    config = json.loads((base_dir / "config.json").read_text())
    (saved / "config.json").write_text(json.dumps({**config, "dtype": "float32"}))
    assert mg.config_drift(base_dir, saved) == {}
    (saved / "config.json").write_text(json.dumps({**config, "max_position_embeddings": 32}))
    assert mg.config_drift(base_dir, saved) == {"max_position_embeddings": (32, 64)}


def test_the_cpu_model_is_read_where_the_machine_says_it(tmp_path, monkeypatch):
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nmodel name\t: Test CPU @ 1GHz\n")
    monkeypatch.setattr(mg, "CPUINFO_PATH", info)
    assert mg.cpu_model() == "Test CPU @ 1GHz"
    info.write_text("processor\t: 0\n")
    monkeypatch.setattr(mg.platform, "processor", lambda: "x86_64")
    assert mg.cpu_model() == "x86_64"
    monkeypatch.setattr(mg, "CPUINFO_PATH", tmp_path / "absent")
    monkeypatch.setattr(mg.platform, "processor", lambda: "")
    assert mg.cpu_model() is None


def test_cli_merges_and_prints_the_summary(base_dir, adapter_dir, tmp_path, capsys):
    out = tmp_path / "out"
    argv = ["--base", str(base_dir), "--adapter", str(adapter_dir), "--out", str(out), "--threads", "1",
            "--expect-rank", str(RANK), "--expect-alpha", str(ALPHA), "--expect-base-model", BASE_NAME,
            "--expect-targets", ",".join(lp.DEFAULT_TARGET_MODULES), "--cast-bf16", "--training-run", "r",
            "--training-step", "7"]
    assert mg.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((out / mg.MANIFEST_FNAME).read_text())
    assert summary["weights_sha256"] == manifest["result"]["weights_sha256"] and summary["saved_dtype"] == "bfloat16"
    assert summary["modules_merged"] == 21 and summary["bf16_cast"] == manifest["result"]["bf16_cast"]["overall"]
    assert manifest["training"] == {"run": "r", "step": 7}
    with pytest.raises(SystemExit):
        mg.main(["--base", str(base_dir), "--adapter", str(adapter_dir), "--out", str(tmp_path / "o"), "--threads", "1"])


class _RenderTokenizer:
    def __init__(self, offset=0):
        self.offset = offset

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True, **kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in chat)

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) + self.offset for c in text]}


def _condition(**overrides):
    from ruleforge.evaluate.identity import ConditionSpec

    fields = dict(
        condition_id="base", model_name="qwen3-8b", model_type="local", prompt_style="forced-grpo",
        individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
        rule_set_name="openai_rules_all.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
        forced_json_requirement="label_only", model_path=BASE_NAME, serving_dtype="float32",
    )
    return ConditionSpec(**{**fields, **overrides})


@pytest.fixture(scope="module")
def two_merges(base_dir, adapter_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("m0")
    for name in ("first", "second"):
        mg.merge_adapter(base_dir, adapter_dir, root / name, expect=EXPECT, threads=1)
    return root / "first", root / "second"


def test_m0_plumbing_passes_a_faithful_merge_and_names_each_check(base_dir, two_merges, monkeypatch):
    first, second = two_merges
    digest = json.loads((first / mg.MANIFEST_FNAME).read_text())["result"]["weights_sha256"]
    monkeypatch.setattr(lp, "_load_tokenizer", lambda path: _RenderTokenizer())
    prompts = [{"system_prompt": "S", "user_prompt": ""}]
    merged = _condition(condition_id="m0", model_path="first", model_weights_sha256=digest)
    report = lp.plumbing(base_dir, first, second_merge=second, prompts=prompts, conditions=(_condition(), merged))
    assert report["passed"], report
    checks = {check["name"]: check for check in report["checks"]}
    assert list(checks) == ["weights", "second_merge", "delta_survives_bf16", "truncation", "rendering", "identity"]
    assert checks["second_merge"]["threads"] == [1, 1]
    assert checks["truncation"]["base"] == checks["truncation"]["merged"] == {"top_k": 20, "top_p": 0.95}
    assert checks["identity"]["differs"] == ["condition_id", "model_path", "model_weights_sha256"]
    assert [c["name"] for c in lp.plumbing(base_dir, first)["checks"]] == ["weights", "delta_survives_bf16", "truncation"]


def test_m0_plumbing_fails_what_m0_is_meant_to_catch(base_dir, adapter_dir, two_merges, monkeypatch, tmp_path):
    first, _ = two_merges
    digest = json.loads((first / mg.MANIFEST_FNAME).read_text())["result"]["weights_sha256"]
    # a synthetic delta too small to survive bf16
    assert not lp.plumbing(base_dir, first, max_rounded_share=-1.0)["passed"]
    tokenizers = iter([_RenderTokenizer(), _RenderTokenizer(offset=1)])
    monkeypatch.setattr(lp, "_load_tokenizer", lambda path: next(tokenizers))
    rendering = lp.plumbing(base_dir, first, prompts=[{"system_prompt": "S", "user_prompt": ""}])
    assert {c["name"]: c["passed"] for c in rendering["checks"]}["rendering"] is False
    # the identity moves in another field, or does not name the merged digest
    moved = _condition(condition_id="m0", model_path="first", model_weights_sha256=digest, max_new_tokens=5)
    identity = lp.plumbing(base_dir, first, conditions=(_condition(), moved))["checks"][-1]
    assert identity["passed"] is False and identity["unexpected"] == ["max_new_tokens"]
    unnamed = lp.plumbing(base_dir, first, conditions=(_condition(), _condition(condition_id="m0", model_path="x")))
    assert unnamed["checks"][-1]["passed"] is False
    other = tmp_path / "other"
    doubled = _copy_adapter(adapter_dir, tmp_path / "a2", tensors=lambda t: {k: v * 2 for k, v in t.items()})
    mg.merge_adapter(base_dir, doubled, other, expect=EXPECT, threads=1)
    assert not lp.plumbing(base_dir, first, second_merge=other)["passed"]
    monkeypatch.setattr(lp, "_model_default_truncation", lambda path: {"top_k": 20 if path == base_dir else 40})
    assert not lp.plumbing(base_dir, first)["passed"]
    with pytest.raises(lp.ParityError, match="not a merged directory"):
        lp.plumbing(base_dir, base_dir)


def test_cli_plumbing_reads_the_conditions_from_a_campaign_root(base_dir, two_merges, tmp_path, capsys, monkeypatch):
    from ruleforge.evaluate.campaign import register_condition

    first, second = two_merges
    digest = json.loads((first / mg.MANIFEST_FNAME).read_text())["result"]["weights_sha256"]
    root = tmp_path / "campaign"
    register_condition(root, _condition())
    register_condition(root, _condition(condition_id="m0", model_path="first", model_weights_sha256=digest))
    args = ["plumbing", "--base", str(base_dir), "--merged", str(first), "--second-merge", str(second),
            "--conditions-root", str(root), "--base-condition", "base", "--merged-condition", "m0",
            "--report", str(tmp_path / "p.json")]
    assert lp.main(args) == 0
    assert json.loads((tmp_path / "p.json").read_text())["passed"] is True
    assert '"identity": true' in capsys.readouterr().out
    assert lp.main([*args, "--may-differ", "condition_id"]) == 1
    with pytest.raises(lp.ParityError, match="go together"):
        lp.main(args[:-4] + ["--report", str(tmp_path / "p.json")])
    with pytest.raises(lp.ParityError, match="registers no condition"):
        lp.main([*args[:-2], "--merged-condition", "absent", "--report", str(tmp_path / "p.json")])
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"q": "S"}) + "\n")
    rendered = ["plumbing", "--base", str(base_dir), "--merged", str(first), "--prompts", str(prompts),
                "--prompt-field", "q", "--report", str(tmp_path / "r.json")]
    monkeypatch.setattr(lp, "_load_tokenizer", lambda path: _RenderTokenizer())
    assert lp.main(rendered) == 0


GEMMA_SCOPE = "model.language_model."


@pytest.fixture(scope="module")
def gemma_base(tmp_path_factory):
    """The tiny Gemma 4 checkpoint (``tests/tiny_gemma.py``): a vision tower, tied embeddings, a K=V layer."""
    from tests.tiny_gemma import MODEL_NAME, save_tiny_gemma

    return save_tiny_gemma(tmp_path_factory.mktemp("bases") / MODEL_NAME)


def _gemma_adapter(base: Path, out: Path, *, b_std: float) -> Path:
    """A random adapter over the language model's targets alone; its config names them by the scope's pattern."""
    from peft import LoraConfig

    scoped = [name for name in lp.checkpoint_module_names(base) if lt.in_scope(name, lp.DEFAULT_TARGET_MODULES, GEMMA_SCOPE)]
    report = lp.make_random_adapter(base, out, target_modules=scoped, rank=RANK, alpha=ALPHA, seed=5, b_std=b_std)
    config = LoraConfig.from_pretrained(str(out))
    config.target_modules = lt.peft_target_modules(lp.DEFAULT_TARGET_MODULES, GEMMA_SCOPE)
    config.save_pretrained(str(out))
    assert report.ok and len(report.module_paths) == 13 and all(path.startswith(GEMMA_SCOPE)
                                                                for path in report.module_paths)
    return out


def test_gemmas_targets_are_its_language_models_and_unscoped_they_are_refused(gemma_base, tmp_path):
    adapter = _gemma_adapter(gemma_base, tmp_path / "adapter", b_std=0.5)
    scoped = mg.MergeExpectation(rank=RANK, alpha=ALPHA, base_model=gemma_base.name, module_scope=GEMMA_SCOPE)
    check = mg.check_adapter(adapter, gemma_base, scoped)
    assert len(check.modules) == 13 and "model.language_model.layers.1.self_attn.v_proj" not in check.modules
    # the vision tower names modules q_proj that are not linear projections: an unscoped expectation is refused
    unscoped = mg.MergeExpectation(rank=RANK, alpha=ALPHA, base_model=gemma_base.name)
    with pytest.raises(mg.MergeError, match="vision_tower.*not a linear projection"):
        mg.check_adapter(adapter, gemma_base, unscoped)


def test_a_gemma_merge_is_w_plus_ba_on_the_language_model_and_keeps_the_tied_head(gemma_base, tmp_path):
    adapter = _gemma_adapter(gemma_base, tmp_path / "adapter", b_std=0.5)
    expect = mg.MergeExpectation(rank=RANK, alpha=ALPHA, base_model=gemma_base.name, module_scope=GEMMA_SCOPE)
    out = tmp_path / "merged"
    manifest = mg.merge_adapter(gemma_base, adapter, out, expect=expect, threads=1)
    saved, expected = _stored(out), _independent_merge(gemma_base, adapter)
    assert set(saved) == set(expected) and "lm_head.weight" not in saved  # tied: the embedding alone is stored
    for name, tensor in saved.items():
        assert tensor.dtype == torch.float32 and torch.equal(tensor, expected[name]), name
    assert (manifest["result"]["modules_merged"], manifest["expectation"]["module_scope"]) == (13, GEMMA_SCOPE)
    config = json.loads((out / "config.json").read_text())
    assert config["architectures"] == ["Gemma4ForConditionalGeneration"] and config["tie_word_embeddings"] is True
    assert config["dtype"] == "float32"
    loaded = transformers.Gemma4ForConditionalGeneration.from_pretrained(str(out))
    assert loaded.lm_head.weight is loaded.model.language_model.embed_tokens.weight
    assert (out / "processor_config.json").read_bytes() == (gemma_base / "processor_config.json").read_bytes()
    assert "processor_config.json" in manifest["result"]["copied_files"]
    from ruleforge.judges.weights import model_weights_identity

    assert model_weights_identity(out) is not None


def test_a_zero_gemma_adapter_merges_to_the_base_exactly(gemma_base, tmp_path):
    adapter = _gemma_adapter(gemma_base, tmp_path / "zero", b_std=0.0)
    expect = mg.MergeExpectation(rank=RANK, alpha=ALPHA, base_model=gemma_base.name, module_scope=GEMMA_SCOPE)
    mg.merge_adapter(gemma_base, adapter, tmp_path / "merged", expect=expect, threads=1)
    base = _stored(gemma_base)
    saved = _stored(tmp_path / "merged")
    assert set(saved) == set(base) and all(torch.equal(saved[name], base[name].float()) for name in base)


def test_a_composite_configs_sub_configs_may_change_only_their_dtype(gemma_base, tmp_path):
    out = tmp_path / "saved"
    out.mkdir()
    config = json.loads((gemma_base / "config.json").read_text())
    config["text_config"]["dtype"] = config["vision_config"]["dtype"] = "float32"
    (out / "config.json").write_text(json.dumps(config))
    assert mg.config_drift(gemma_base, out) == {}
    config["text_config"]["final_logit_softcapping"] = 50.0
    (out / "config.json").write_text(json.dumps(config))
    assert set(mg.config_drift(gemma_base, out)) == {"text_config"}


def test_the_merge_cli_scopes_the_expectation(gemma_base, tmp_path, capsys):
    _gemma_adapter(gemma_base, tmp_path / "a", b_std=0.5)
    assert json.loads((tmp_path / "a" / lp.ADAPTER_CONFIG_FNAME).read_text())["target_modules"] == (
        lt.peft_target_modules(lp.DEFAULT_TARGET_MODULES, GEMMA_SCOPE))
    assert mg.main(["--base", str(gemma_base), "--adapter", str(tmp_path / "a"), "--out", str(tmp_path / "m"),
                    "--threads", "1", "--expect-rank", "4", "--expect-alpha", "8", "--expect-base-model",
                    gemma_base.name, "--expect-scope", GEMMA_SCOPE]) == 0
    assert json.loads(capsys.readouterr().out)["modules_merged"] == 13


def test_scoped_targets_are_one_full_match_pattern_below_the_scope():
    import re

    pattern = lt.peft_target_modules(("q_proj", "down_proj"), GEMMA_SCOPE)
    assert re.fullmatch(pattern, "model.language_model.layers.3.self_attn.q_proj")
    assert not re.fullmatch(pattern, "model.vision_tower.encoder.layers.0.self_attn.q_proj")
    assert not re.fullmatch(pattern, "model.language_model.layers.3.self_attn.q_proj.base_layer")
    assert lt.peft_target_modules(("q_proj",)) == ["q_proj"]
    assert lt.in_scope("model.layers.0.q_proj", ("q_proj",)) and not lt.in_scope("model.layers.0.q_proj", ("q_proj",),
                                                                                    GEMMA_SCOPE)
