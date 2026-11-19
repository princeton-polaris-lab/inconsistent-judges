from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from ruleforge.grpo import lora_parity as lp  # noqa: E402
from ruleforge.core.artifacts import load_jsonl_records, write_jsonl_records  # noqa: E402
from tests.support.parity import preset_thresholds  # noqa: E402

pytestmark = pytest.mark.torch

TINY_LLAMA = dict(
    vocab_size=48, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
)
TINY_QWEN3 = dict(
    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
)
QWEN3_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "qwen3-8b.yaml"
QWEN3_FP32_CONFIG = QWEN3_CONFIG.with_name("qwen3-8b-fp32.yaml")


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    """A saved tiny random Llama: config.json names LlamaForCausalLM."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny_llama")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY_LLAMA)).save_pretrained(str(path))
    return path


@pytest.fixture(scope="module")
def qwen3_dir(tmp_path_factory):
    """A tiny Qwen3ForCausalLM config (the GRPO base model's class); no weights are needed."""
    path = tmp_path_factory.mktemp("tiny_qwen3")
    config = transformers.Qwen3Config(**TINY_QWEN3)
    config.architectures = ["Qwen3ForCausalLM"]
    config.save_pretrained(str(path))
    return path


def _adapter_dir(root: Path) -> Path:
    """Deep enough for build_lora_request's <run>/<stage>/<adapter>/<file> layout."""
    return root / "run" / "stage" / "adapter"


#: What a score file of an adapter arm records of the adapter it applies.
ADAPTER = {"adapter_sha256": "a" * 64, "adapter_config_sha256": "c" * 64}


def _scores(logprobs, argmax=None) -> lp.Scores:
    return lp.Scores({arm: dict(rows) for arm, rows in logprobs.items()}, dict(argmax or {}))


def test_sequences_round_trip_and_refuse_bad_files(tmp_path):
    seqs = [lp.FrozenSequence("a", (1, 2), (3,), "stop"), lp.FrozenSequence("b", (4,), (5, 6), None)]
    path = tmp_path / "s.jsonl"
    lp.write_sequences(path, seqs)
    path.write_text(path.read_text() + "\n")
    assert lp.read_sequences(path) == seqs
    assert seqs[0].token_ids == (1, 2, 3)
    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(lp.ParityError, match="no sequences"):
        lp.read_sequences(tmp_path / "empty.jsonl")
    lp.write_sequences(tmp_path / "dup.jsonl", [seqs[0], seqs[0]])
    with pytest.raises(lp.ParityError, match="repeats"):
        lp.read_sequences(tmp_path / "dup.jsonl")
    with pytest.raises(lp.ParityError, match="fewer than two"):
        lp.FrozenSequence("c", (1,), ())


def test_scores_round_trip_keep_nan_and_argmax_and_merge_refuses_a_shared_arm(tmp_path):
    path = tmp_path / "scores.json"
    sequence = lp.FrozenSequence("a", (1, 2), (3,))
    lp.write_scores(path, _scores({"hf_base": {"a": [-1.0, math.nan]}}, {"hf_base": {"a": [3, None]}}), {"k": "v"},
                    [sequence])
    loaded = lp.read_scores(path)
    assert loaded.logprobs["hf_base"]["a"][0] == -1.0 and math.isnan(loaded.logprobs["hf_base"]["a"][1])
    assert loaded.argmax == {"hf_base": {"a": [3, None]}}
    written = {"k": "v", "sequences_sha256": {"a": lp.sequence_sha256(sequence)}}
    assert json.loads(path.read_text())["meta"] == written
    assert loaded.meta == {"hf_base": written}
    path.write_text(json.dumps({"meta": {"engine_dtype": "float32"}, "arms": {"vllm_base": {"a": [-2.0]}}}))
    assert lp.read_scores(path).argmax == {}
    merged = loaded.merge(lp.read_scores(path))
    assert set(merged.logprobs) == {"hf_base", "vllm_base"}
    assert merged.meta == {"hf_base": written, "vllm_base": {"engine_dtype": "float32"}}
    with pytest.raises(lp.ParityError, match=r"arm\(s\) \['hf_base'\] are in two score files"):
        merged.merge(loaded)
    both = _scores({"x": {"a": [1.0], "b": [2.0]}}, {"x": {"a": [1], "b": [2]}}).only(["b"])
    assert (both.logprobs, both.argmax) == ({"x": {"b": [2.0]}}, {"x": {"b": [2]}})
    assert merged.only(["a"]).meta == merged.meta
    path.write_text(json.dumps({"arms": {"vllm_base": {"a": [-2.0]}}}))
    assert lp.read_scores(path).meta == {"vllm_base": {}}


def test_prompts_in_both_shapes(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"system_prompt": "S", "user_prompt": "U"}) + "\n\n" + json.dumps({"user_prompt": "u"}) + "\n")
    assert lp.read_prompts(path) == [{"system_prompt": "S", "user_prompt": "U"}, {"system_prompt": "", "user_prompt": "u"}]
    field = tmp_path / "f.jsonl"
    field.write_text(json.dumps({"prompt": "P"}) + "\n")
    assert lp.read_prompts(field, "prompt") == [{"system_prompt": "P", "user_prompt": ""}]
    with pytest.raises(lp.ParityError, match="non-empty 'other'"):
        lp.read_prompts(field, "other")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"system_prompt": ""}) + "\n")
    with pytest.raises(lp.ParityError, match="neither"):
        lp.read_prompts(bad)
    (tmp_path / "none.jsonl").write_text("\n")
    with pytest.raises(lp.ParityError, match="no prompts"):
        lp.read_prompts(tmp_path / "none.jsonl")


def _write_adapter(path: Path, shapes: dict[str, tuple[int, ...]], config: dict | None = None) -> Path:
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.zeros(v) for k, v in shapes.items()}, str(path / lp.ADAPTER_WEIGHTS_FNAME))
    (path / lp.ADAPTER_CONFIG_FNAME).write_text(json.dumps(config if config is not None else {"r": 4, "lora_alpha": 8}))
    return path


def test_the_safetensors_header_is_read_without_the_data(tmp_path):
    adapter = _write_adapter(tmp_path / "a", {"x.lora_A.weight": (2, 3)})
    assert lp.read_safetensors_header(adapter / lp.ADAPTER_WEIGHTS_FNAME) == {
        "x.lora_A.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}
    }
    (tmp_path / "short.safetensors").write_bytes(b"abc")
    with pytest.raises(lp.ParityError, match="not a safetensors"):
        lp.read_safetensors_header(tmp_path / "short.safetensors")


def test_a_clean_adapter_passes_and_reports_its_rank(tmp_path):
    prefix = "base_model.model.model.layers.0."
    adapter = _write_adapter(tmp_path / "a", {
        prefix + "self_attn.q_proj.lora_A.weight": (4, 16),
        prefix + "self_attn.q_proj.lora_B.weight": (16, 4),
        prefix + "mlp.up_proj.lora_A.weight": (4, 16),
        prefix + "mlp.up_proj.lora_B.weight": (32, 4),
        "base_model.model.lm_head.modules_to_save.weight": (8, 16),
    }, {"r": 4, "lora_alpha": 8, "target_modules": "q_proj"})
    names = ["model.layers.0.mlp.up_proj", "model.layers.0.self_attn.q_proj"]
    report = lp.inspect_adapter(adapter, module_names=names)
    assert report.ok, report.problems
    assert (report.rank, report.alpha, report.max_rank) == (4, 8.0, 4)
    assert report.target_modules == ("q_proj",)
    assert report.module_paths == tuple(names)
    assert report.targeted_suffixes == ("q_proj", "up_proj")
    assert report.extra_keys == ("base_model.model.lm_head.modules_to_save.weight",)


def test_every_way_an_adapter_cannot_be_served_is_named(tmp_path):
    adapter = _write_adapter(tmp_path / "a", {
        "base_model.model.m.a.lora_A.weight": (4, 8),
        # ranks disagree
        "base_model.model.m.b.lora_A.weight": (4, 8),
        "base_model.model.m.b.lora_B.weight": (8, 2),
        # keyed outside the checkpoint's tree
        "model.language_model.layers.0.self_attn.q_proj.lora_A.weight": (4, 8),
        "model.language_model.layers.0.self_attn.q_proj.lora_B.weight": (8, 4),
    }, {"r": 4})
    report = lp.inspect_adapter(adapter, module_names=["m.a", "m.b"])
    joined = "\n".join(report.problems)
    assert not report.ok and len(report.problems) == 3
    assert "m.a: has lora_A only" in joined
    assert "lora_A rank 4 != lora_B rank 2" in joined
    assert "not in the checkpoint's module tree" in joined and "model.language_model.layers.0.self_attn.q_proj" in joined
    assert report.alpha is None and report.target_modules == ()
    empty = _write_adapter(tmp_path / "e", {"x.weight": (1,)}, {})
    assert lp.inspect_adapter(empty).problems == ("no LoRA factors (keys ending .lora_A.weight / .lora_B.weight)",)
    assert lp.inspect_adapter(empty).rank is None


def test_a_random_adapter_is_keyed_as_peft_keys_it_and_is_seeded(llama_dir, tmp_path):
    first = lp.make_random_adapter(llama_dir, tmp_path / "one", rank=2, alpha=4, seed=3)
    again = lp.make_random_adapter(llama_dir, tmp_path / "two", rank=2, alpha=4, seed=3)
    assert first.ok and first.max_rank == 2 and len(first.module_paths) == 14
    assert first.targeted_suffixes == tuple(sorted(lp.DEFAULT_TARGET_MODULES))
    weights = (tmp_path / "one" / lp.ADAPTER_WEIGHTS_FNAME).read_bytes()
    assert weights == (tmp_path / "two" / lp.ADAPTER_WEIGHTS_FNAME).read_bytes()
    config = json.loads((tmp_path / "one" / lp.ADAPTER_CONFIG_FNAME).read_text())
    assert (config["r"], config["lora_alpha"], config["base_model_name_or_path"]) == (2, 4, str(llama_dir))
    from safetensors.torch import load_file

    tensors = load_file(str(tmp_path / "one" / lp.ADAPTER_WEIGHTS_FNAME))
    b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    assert b.abs().sum() > 0, "B = 0 would make the adapter a no-op and the check vacuous"
    assert a.abs().max() <= 1 / math.sqrt(16)
    assert set(again.module_paths) == set(first.module_paths)


def test_a_target_that_matches_nothing_is_refused(llama_dir, tmp_path):
    with pytest.raises(lp.ParityError, match="in_proj_qkv"):
        lp.make_random_adapter(llama_dir, tmp_path / "x", target_modules=("q_proj", "in_proj_qkv"))


def test_the_default_adapter_covers_qwen3s_attention_and_mlp_and_a_misplaced_one_adapts_nothing(qwen3_dir, tmp_path):
    report = lp.make_random_adapter(qwen3_dir, tmp_path / "adapter", rank=4, alpha=8)
    assert report.ok, report.problems
    # 7 targets x 3 layers, every one in the text stack
    assert len(report.module_paths) == 21
    assert all(p.startswith("model.layers.") for p in report.module_paths)
    names = lp.checkpoint_module_names(qwen3_dir)
    assert "model.layers.0.self_attn.q_proj" in names
    moved = tmp_path / "moved"
    moved.mkdir()
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(tmp_path / "adapter" / lp.ADAPTER_WEIGHTS_FNAME))
    save_file({k.replace("model.model.", "model.model.language_model."): v for k, v in tensors.items()},
              str(moved / lp.ADAPTER_WEIGHTS_FNAME))
    (moved / lp.ADAPTER_CONFIG_FNAME).write_text((tmp_path / "adapter" / lp.ADAPTER_CONFIG_FNAME).read_text())
    refused = lp.inspect_adapter(moved, module_names=names)
    assert any("21 adapted module(s)" in p and "adapt nothing" in p for p in refused.problems)


def test_a_config_without_architectures_is_refused(tmp_path):
    transformers.LlamaConfig(**TINY_LLAMA).save_pretrained(str(tmp_path))
    config = json.loads((tmp_path / "config.json").read_text())
    config.pop("architectures", None)
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(lp.ParityError, match="names no architectures"):
        lp.checkpoint_module_names(tmp_path)


def test_target_scores_match_a_direct_log_softmax_and_argmax():
    logits = torch.randn(5, 7)
    ids = [3, 1, 4, 1, 5]
    values, argmax = lp.scores_of_targets(logits, ids, chunk=2)
    assert values == pytest.approx([torch.log_softmax(logits[i], -1)[ids[i + 1]].item() for i in range(4)], abs=1e-6)
    assert argmax == [int(logits[i].argmax()) for i in range(4)]


def test_hf_arms_differ_by_the_adapter_and_the_base_arm_is_the_plain_model(llama_dir, tmp_path):
    adapter = _adapter_dir(tmp_path)
    lp.make_random_adapter(llama_dir, adapter, rank=2, alpha=4, seed=1, b_std=0.5)
    model = lp.load_hf_model(llama_dir, dtype="float32", device="cpu", attn_implementation="eager")
    seqs = [lp.FrozenSequence("a", (1, 2, 3), (4, 5)), lp.FrozenSequence("b", (7, 8), (9,))]
    plain = {s.seq_id: lp.hf_sequence_scores(model, s.token_ids, device="cpu") for s in seqs}
    # without an adapter: the plain model alone, as hf_base (before PEFT injects the adapter in place)
    alone = lp.score_hf(model, None, seqs, device="cpu")
    assert set(alone.logprobs) == set(alone.argmax) == {"hf_base"}
    assert alone.logprobs["hf_base"]["b"] == plain["b"][0] and alone.argmax["hf_base"]["b"] == plain["b"][1]
    scores = lp.score_hf(model, adapter, seqs, device="cpu")
    assert set(scores.logprobs) == set(scores.argmax) == {"hf_base", "hf_lora"}
    assert scores.logprobs["hf_base"]["a"] == pytest.approx(plain["a"][0], abs=1e-5)
    assert scores.argmax["hf_base"]["a"] == plain["a"][1]
    assert len(scores.logprobs["hf_lora"]["a"]) == len(scores.argmax["hf_lora"]["a"]) == 4
    assert max(abs(x - y) for x, y in zip(scores.logprobs["hf_lora"]["a"], scores.logprobs["hf_base"]["a"])) > 1e-4


def test_the_hf_forward_runs_without_autocast(llama_dir):
    """An enclosing bf16 autocast (a training forward's) does not reach the reference arms."""
    model = lp.load_hf_model(llama_dir, dtype="float32", device="cpu", attn_implementation="eager")
    ids = (1, 2, 3, 4, 5, 6)
    plain = lp.hf_sequence_scores(model, ids, device="cpu")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert lp.hf_sequence_scores(model, ids, device="cpu") == plain
        with torch.no_grad():
            autocast_logits = model(input_ids=torch.tensor([ids])).logits[0]
    assert lp.scores_of_targets(autocast_logits, ids)[0] != plain[0]


def test_exact_float32_turns_tf32_off():
    before = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32)
    try:
        torch.set_float32_matmul_precision("high")
        assert lp.exact_float32() == {"float32_matmul_precision": "highest", "cudnn_allow_tf32": False}
    finally:
        torch.set_float32_matmul_precision(before[0])
        torch.backends.cudnn.allow_tf32 = before[1]


class _Logprob:
    def __init__(self, value, rank=None):
        self.logprob, self.rank = value, rank


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=_FakeSamplingParams))


def _entry(token, value, *, top=0):
    if token == top:
        return {token: _Logprob(value, 1)}
    return {token: _Logprob(value, 2), top: _Logprob(-0.01, 1)}


class _ScoringLLM:
    """Prompt logprob of token t at position i: -(t + i)/10, plus ``shift`` under a LoRA request; argmax token 0."""

    def __init__(self, drop=None, shift=1.0):
        self.calls = []
        self.drop = drop
        self.shift = shift

    def generate(self, prompts, sampling_params=None, use_tqdm=True, lora_request=None):
        self.calls.append((sampling_params.kwargs, lora_request))
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            shift = self.shift if lora_request is not None else 0.0
            entries = [None] + [_entry(t, -(t + i) / 10 + shift) for i, t in enumerate(ids) if i > 0]
            if self.drop == "entry":
                entries[1] = None
            elif self.drop == "length":
                entries = entries[:-1]
            elif self.drop == "rank":
                entries[1] = {ids[1]: _Logprob(-1.0)}
            outputs.append(types.SimpleNamespace(prompt_logprobs=entries))
        return outputs[:-1] if self.drop == "outputs" else outputs


def test_servings_of_one_engine_with_and_without_the_request(fake_vllm):
    llm = _ScoringLLM()
    seqs = [lp.FrozenSequence("a", (1, 2), (0,))]
    scores = lp.score_servings(llm, [("vllm_base", None), ("vllm_lora", "REQ")], seqs)
    assert scores.logprobs["vllm_base"]["a"] == pytest.approx([-0.3, -0.2])
    assert scores.logprobs["vllm_lora"]["a"] == pytest.approx([0.7, 0.8])
    assert scores.argmax["vllm_base"]["a"] == [0, 0]
    assert [call[1] for call in llm.calls] == [None, "REQ"]
    assert llm.calls[0][0] == {"max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}
    assert lp.score_servings(llm, [], seqs).logprobs == {}
    assert lp.score_servings(_ScoringLLM(drop="rank"), [("x", None)], seqs).argmax["x"]["a"] == [None, 0]


@pytest.mark.parametrize("drop, match", [
    ("entry", "no logprob for token 2"), ("length", "no logprob per token"), ("outputs", "1 sequences"),
])
def test_a_short_vllm_answer_is_refused(fake_vllm, drop, match):
    seqs = [lp.FrozenSequence("a", (1, 2), (3,))]
    if drop == "outputs":
        seqs.append(lp.FrozenSequence("b", (1, 2), (3,)))
        match = "1 outputs for 2 sequences"
    with pytest.raises(lp.ParityError, match=match):
        lp.vllm_prompt_scores(_ScoringLLM(drop=drop), seqs)


class _Tokenizer:
    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True, **kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in chat)


def test_sampling_renders_like_the_campaign_and_freezes_the_ids(fake_vllm):
    seen = {}

    class _SamplingLLM:
        def generate(self, prompts, sampling_params=None, use_tqdm=True):
            seen["prompts"], seen["params"] = prompts, [p.kwargs for p in sampling_params]
            return [
                types.SimpleNamespace(prompt_token_ids=[1, 2], outputs=[types.SimpleNamespace(token_ids=[3, 4], finish_reason="stop")]),
                types.SimpleNamespace(prompt_token_ids=[5], outputs=[types.SimpleNamespace(token_ids=[6], finish_reason="length")]),
            ]

    prompts = [{"system_prompt": "S", "user_prompt": ""}, {"system_prompt": "T", "user_prompt": "u"}]
    seqs = lp.sample_sequences(_SamplingLLM(), _Tokenizer(), prompts, max_tokens=9, seed=40, truncation={"top_k": 20})
    assert seen["prompts"] == ["system:S|user:", "system:T|user:u"]
    assert [p["seed"] for p in seen["params"]] == [40, 41]
    assert seen["params"][0] == {
        "temperature": 1.0, "max_tokens": 9, "seed": 40, "skip_special_tokens": False,
        "top_p": 1.0, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
    }
    assert seqs == [lp.FrozenSequence("seq-0000", (1, 2), (3, 4), "stop"), lp.FrozenSequence("seq-0001", (5,), (6,), "length")]


class _DecodingLLM:
    def __init__(self, *, short=None):
        self.calls = []
        self.short = short

    def generate(self, prompts, sampling_params=None, use_tqdm=True, lora_request=None):
        self.calls.append(([p.kwargs for p in sampling_params], lora_request, len(prompts)))
        shift = 0.5 if lora_request is not None else 0.0
        outputs = []
        for index, prompt in enumerate(prompts):
            prompt_ids = [1, 2, 3 + len(prompt)]
            completion = [9, 8 + index]
            prompt_lp = [None] + [_entry(t, -0.1 * t) for t in prompt_ids[1:]]
            sample_lp = [_entry(t, -0.01 * t + shift, top=9) for t in completion]
            if self.short == "sample":
                sample_lp = sample_lp[:-1]
            outputs.append(types.SimpleNamespace(
                prompt_token_ids=prompt_ids, prompt_logprobs=prompt_lp,
                outputs=[types.SimpleNamespace(token_ids=completion, logprobs=sample_lp, finish_reason="length")],
            ))
        return outputs[:-1] if self.short == "outputs" else outputs


def test_the_decode_arm_samples_in_small_batches_with_or_without_the_adapter(fake_vllm):
    llm = _DecodingLLM()
    prompts = [{"system_prompt": f"S{i}", "user_prompt": ""} for i in range(5)]
    seqs, scores = lp.sample_decode(
        llm, _Tokenizer(), prompts, lora_request="REQ", max_tokens=7, seed=100, batch_size=2, truncation={"top_k": 20},
    )
    # three calls of at most two prompts, every one under the adapter
    assert [(call[1], call[2]) for call in llm.calls] == [("REQ", 2), ("REQ", 2), ("REQ", 1)]
    params = [kw for call in llm.calls for kw in call[0]]
    assert [kw["seed"] for kw in params] == [100, 101, 102, 103, 104]
    assert params[0] == {
        "temperature": 1.0, "max_tokens": 7, "seed": 100, "skip_special_tokens": False, "top_p": 1.0,
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "logprobs": 1, "prompt_logprobs": 1,
    }
    assert [seq.seq_id for seq in seqs] == [f"dec-{i:04d}" for i in range(5)]
    # the fake engine's third prompt token is 3 + the rendered prompt's length ("system:S0|user:")
    assert seqs[0] == lp.FrozenSequence("dec-0000", (1, 2, 18), (9, 8), "length")
    # prompt positions from the request's prefill, completion positions from its decode
    assert scores.logprobs["_"]["dec-0000"] == pytest.approx([-0.2, -1.8, 0.41, 0.42])
    assert scores.argmax["_"]["dec-0000"] == [0, 0, 9, 9]
    assert len(scores.logprobs["_"]["dec-0004"]) == len(seqs[4].token_ids) - 1
    reference = _DecodingLLM()
    refs, _ = lp.sample_decode(reference, _Tokenizer(), prompts[:1], max_tokens=7, seed=100, batch_size=2, prefix="ref")
    assert reference.calls[0][1] is None and refs[0].seq_id == "ref-0000"


@pytest.mark.parametrize("short, match", [("sample", "no logprob per token"), ("outputs", "1 outputs for 2 prompts")])
def test_a_short_decode_answer_is_refused(fake_vllm, short, match):
    prompts = [{"system_prompt": "S", "user_prompt": ""}] * 2
    with pytest.raises(lp.ParityError, match=match):
        lp.sample_decode(_DecodingLLM(short=short), _Tokenizer(), prompts, lora_request="REQ", max_tokens=2, seed=0, batch_size=2)
    with pytest.raises(lp.ParityError, match="batch_size must be positive"):
        lp.sample_decode(_DecodingLLM(), _Tokenizer(), prompts, max_tokens=2, seed=0, batch_size=0)


def _arms(n_tokens, *, noise=0.01, effect=0.2, vllm_effect=None, served="vllm_lora"):
    vllm_effect = effect if vllm_effect is None else vllm_effect
    hf_base = [-1.0 - 0.001 * i for i in range(n_tokens)]
    vllm_base = [x + noise for x in hf_base]
    return _scores({
        "hf_base": {"s": hf_base},
        "hf_lora": {"s": [x - effect for x in hf_base]},
        "vllm_base": {"s": vllm_base},
        served: {"s": [x - vllm_effect for x in vllm_base]},
    })


def _seq(n_tokens, prompt=None):
    prompt = n_tokens // 2 if prompt is None else prompt
    return [lp.FrozenSequence("s", tuple(range(prompt)), tuple(range(n_tokens + 1 - prompt)))]


def test_a_faithful_adapter_passes_every_bucket():
    thresholds = lp.ParityThresholds(bucket_edges=(100,), min_bucket_tokens=10)
    report = lp.compare(_arms(300), _seq(300), thresholds)
    assert report.passed, report.failures
    assert (report.mode, report.served_arm) == ("lora", "vllm_lora")
    assert set(report.buckets) == {"all", "[0,100)", "[100,inf)"}
    stats = report.buckets["all"]
    assert stats.tokens == 300 and stats.effect_ratio == pytest.approx(1.0)
    assert stats.noise_base == pytest.approx(0.01) == stats.noise and math.isnan(stats.decode_noise)
    assert stats.effect_noise_ratio == pytest.approx(20.0)
    assert stats.serving_limit == pytest.approx(1.5 * 0.01)
    assert stats.effect_window == pytest.approx(((2 / 0.15) * 0.01, (0.015 + 0.01) / 0.15))
    assert stats.rho == pytest.approx(1.0)
    assert report.segments["prompt"].tokens + report.segments["completion"].tokens == 300
    assert report.to_json()["buckets"]["all"]["tokens"] == 300
    assert math.isnan(stats.greedy_served) and any("greedy agreement not measured" in n for n in report.notes)


def test_the_silent_no_op_fails_the_paired_gate_and_the_opt_in_effect_ratio():
    report = lp.compare(_arms(100, vllm_effect=0.0), _seq(100))
    assert not report.passed
    assert report.thresholds["effect_ratio"] is None and report.buckets["all"].effect_ratio == 0.0
    assert not any("outside [" in f for f in report.failures)
    assert any("paired delta" in f and "no-op signature" in f for f in report.failures)
    assert any("exceeds" in f for f in report.failures)
    assert report.buckets["all"].rho == pytest.approx(0.0)
    hunted = lp.compare(_arms(100, vllm_effect=0.0), _seq(100), lp.ParityThresholds(effect_ratio=(0.9, 1.1)))
    assert any("effect ratio vLLM/HF 0.000 outside [0.9, 1.1]" in f and "no-op signature" in f for f in hunted.failures)
    assert lp.compare(_arms(100), _seq(100), lp.ParityThresholds(effect_ratio=(0.9, 1.1))).passed


def test_a_long_context_only_failure_is_not_averaged_away():
    thresholds = lp.ParityThresholds(bucket_edges=(100,), min_bucket_tokens=10)
    arms = _arms(1000)
    arms.logprobs["vllm_lora"]["s"] = [v - 0.4 if i >= 900 else v for i, v in enumerate(arms.logprobs["vllm_lora"]["s"])]
    report = lp.compare(arms, _seq(1000), thresholds)
    assert not report.passed
    assert all(f.startswith("[100,inf)") or f.startswith("all") for f in report.failures)
    assert any(f.startswith("[100,inf)") for f in report.failures)


def test_nan_fails_and_is_counted_but_the_finite_tokens_are_still_compared():
    arms = _arms(100)
    arms.logprobs["vllm_lora"]["s"][50] = math.nan
    report = lp.compare(arms, _seq(100))
    assert report.nonfinite == {"hf_base": 0, "hf_lora": 0, "vllm_base": 0, "vllm_lora": 1}
    assert report.buckets["all"].tokens == 99
    assert any("non-finite" in f for f in report.failures)


def test_a_weak_adapter_proves_nothing_and_fails():
    report = lp.compare(_arms(100, effect=0.0), _seq(100))
    assert any("too weak to test" in f for f in report.failures)
    assert math.isnan(report.buckets["all"].effect_ratio) and math.isnan(report.buckets["all"].rho)


def test_identical_engines_pass_under_the_absolute_tolerance_and_small_buckets_are_not_gated():
    arms = _arms(60, noise=0.0)
    arms.logprobs["vllm_lora"]["s"] = [v + 5e-4 for v in arms.logprobs["vllm_lora"]["s"]]
    report = lp.compare(arms, _seq(60), lp.ParityThresholds(bucket_edges=(50,), min_bucket_tokens=20))
    assert report.passed, report.failures
    assert report.buckets["[50,inf)"].tokens == 11
    assert report.buckets["all"].effect_noise_ratio == math.inf


def test_an_all_nan_arm_leaves_nothing_to_compare():
    arms = _arms(10)
    arms.logprobs["vllm_base"]["s"] = [math.nan] * 10
    report = lp.compare(arms, _seq(10))
    assert "all: no finite tokens to compare" in report.failures
    assert report.buckets["all"].tokens == 0 and math.isnan(report.buckets["all"].noise_base)


def test_arms_that_do_not_line_up_are_refused():
    arms = _arms(10)
    with pytest.raises(lp.ParityError, match="no scores for arm"):
        lp.compare(_scores({k: v for k, v in arms.logprobs.items() if k != "hf_lora"}), _seq(10))
    with pytest.raises(lp.ParityError, match="exactly one served arm"):
        lp.compare(_scores({k: v for k, v in arms.logprobs.items() if k != "vllm_lora"}), _seq(10))
    with pytest.raises(lp.ParityError, match="exactly one served arm"):
        lp.compare(_scores({**arms.logprobs, "vllm_merged": arms.logprobs["vllm_lora"]}), _seq(10))
    arms.logprobs["vllm_base"]["s"] = arms.logprobs["vllm_base"]["s"][:-1]
    with pytest.raises(lp.ParityError, match="vllm_base has no 10 logprobs"):
        lp.compare(arms, _seq(10))


def _delta_arms(n_tokens=400, *, seed=0, noise=0.02, attn=0.4, mlp=0.15, independent=False,
                served=lambda d_attn, d_mlp, other: d_attn + d_mlp, arm="vllm_lora", n_seqs=1):
    """Arms whose adapter delta is two module families' sum, per token, with random signs."""
    rng = random.Random(seed)

    def signed(scale):
        return rng.choice((-1.0, 1.0)) * rng.uniform(0.5 * scale, 1.5 * scale)

    names = [f"s{i}" for i in range(n_seqs)]
    arms = {name: {seq: [] for seq in names} for name in ("hf_base", "hf_lora", "vllm_base", arm)}
    for index in range(n_tokens):
        seq = names[index % n_seqs]
        hb = -rng.uniform(0.1, 3.0)
        d_attn, d_mlp = signed(attn), signed(mlp)
        other = signed(attn) + signed(mlp)
        vb = hb + signed(noise)
        arms["hf_base"][seq].append(hb)
        arms["hf_lora"][seq].append(hb + d_attn + d_mlp)
        arms["vllm_base"][seq].append(vb)
        own = hb + signed(noise) if independent else vb + signed(0.002)
        arms[arm][seq].append(own + served(d_attn, d_mlp, other))
    return _scores(arms)


def _delta_seqs(n_tokens=400, n_seqs=1):
    per = n_tokens // n_seqs
    return [lp.FrozenSequence(f"s{i}", tuple(range(per // 2)), tuple(range(per + 1 - per // 2))) for i in range(n_seqs)]


def test_a_faithfully_served_delta_passes_the_paired_gate():
    report = lp.compare(_delta_arms(), _delta_seqs(), lp.ParityThresholds(bucket_edges=(200,)))
    assert report.passed, report.failures
    stats = report.buckets["all"]
    assert stats.paired_delta_ratio < 0.02 and 0.9 < stats.effect_ratio < 1.1
    assert stats.effect_noise_ratio > 2 / 0.15
    independent = lp.compare(_delta_arms(independent=True), _delta_seqs(), lp.ParityThresholds(bucket_edges=(200,)))
    assert independent.passed, independent.failures


@pytest.mark.parametrize("served", [
    pytest.param(lambda d_attn, d_mlp, other: -(d_attn + d_mlp), id="negated"),
    pytest.param(lambda d_attn, d_mlp, other: other, id="unrelated"),
    pytest.param(lambda d_attn, d_mlp, other: d_attn, id="mlp-family-dropped"),
])
def test_a_wrong_delta_fails_the_paired_gate_overall_and_per_bucket(served):
    report = lp.compare(_delta_arms(served=served), _delta_seqs(), lp.ParityThresholds(bucket_edges=(200,)))
    assert not report.passed
    for label in ("all", "[0,200)", "[200,inf)"):
        assert any(f.startswith(label) and "paired delta" in f for f in report.failures), (label, report.failures)
    assert report.buckets["all"].paired_delta_ratio > 0.15


def test_the_paired_gate_catches_what_the_effect_ratio_passes():
    """An unrelated delta of the same magnitude: vLLM moves by as much as HF, so the unpaired effect ratio passes."""
    report = lp.compare(_delta_arms(served=lambda d_attn, d_mlp, other: other), _delta_seqs(),
                        lp.ParityThresholds(bucket_edges=(), effect_ratio=(0.9, 1.1)))
    assert not report.passed
    assert 0.9 <= report.buckets["all"].effect_ratio <= 1.1
    assert not any("effect ratio vLLM/HF" in f for f in report.failures)
    assert any("paired delta" in f and "no-op" not in f for f in report.failures)
    assert abs(report.buckets["all"].rho) < 0.2


def test_independent_rounding_puts_the_paired_floor_at_sqrt2_times_the_noise():
    """Each vLLM pass rounds on its own: a faithful adapter's paired statistic is ~sqrt(2) x the base arms' noise."""
    def arms(scale):
        rng = random.Random(7)
        rows = {name: {"s": []} for name in ("hf_base", "hf_lora", "vllm_base", "vllm_merged")}
        for _ in range(4000):
            hb, delta = -rng.uniform(0.1, 3.0), rng.choice((-1.0, 1.0)) * scale
            rows["hf_base"]["s"].append(hb)
            rows["hf_lora"]["s"].append(hb + delta)
            rows["vllm_base"]["s"].append(hb + rng.gauss(0.0, 0.01))
            rows["vllm_merged"]["s"].append(hb + delta + rng.gauss(0.0, 0.01))
        return _scores(rows)

    seqs = _seq(4000)
    noise = 0.01 * math.sqrt(2 / math.pi)
    weak = lp.compare(arms(5 * noise), seqs, lp.ParityThresholds(bucket_edges=()))
    stats = weak.buckets["all"]
    assert stats.paired_delta / stats.noise_base == pytest.approx(math.sqrt(2), rel=0.05)
    assert stats.effect_noise_ratio == pytest.approx(5, rel=0.05)
    assert stats.paired_delta_ratio > 0.15
    assert len(weak.failures) == 2 and all("too weak to test" in f for f in weak.failures)
    strong = lp.compare(arms(25 * noise), seqs, lp.ParityThresholds(bucket_edges=()))
    assert strong.passed, strong.failures
    assert strong.mode == "merged" and strong.buckets["all"].rho == pytest.approx(1.0, abs=0.01)


def test_the_minimum_effect_is_held_to_the_paired_gates_floor():
    assert lp.ParityThresholds().min_effect_noise_ratio == pytest.approx(2 / 0.15)
    assert lp.ParityThresholds(paired_delta=0.2).min_effect_noise_ratio == pytest.approx(10.0)
    assert lp.ParityThresholds(min_effect_noise_ratio=9.5).min_effect_noise_ratio == 9.5
    with pytest.raises(lp.ParityError, match=r"under sqrt\(2\)/paired_delta = 9.43"):
        lp.ParityThresholds(min_effect_noise_ratio=3.0)
    with pytest.raises(lp.ParityError, match="paired_delta must be positive"):
        lp.ParityThresholds(paired_delta=0.0)
    with pytest.raises(lp.ParityError, match="rho_range"):
        lp.ParityThresholds(rho_range=(1.01, 0.99))
    with pytest.raises(lp.ParityError, match="effect_ratio"):
        lp.ParityThresholds(effect_ratio=(1.1, 0.9))


def test_an_effect_inside_the_engine_noise_proves_nothing_or_is_inconclusive():
    arms = _delta_arms(noise=0.3, attn=0.4, mlp=0.1)
    report = lp.compare(arms, _delta_seqs(), lp.ParityThresholds(bucket_edges=()))
    assert len(report.failures) == 2
    assert all("the gates would compare engine noise" in f for f in report.failures)
    lenient = lp.compare(arms, _delta_seqs(), lp.ParityThresholds(bucket_edges=(), require_effect=False))
    assert lenient.passed and len(lenient.inconclusive) == 2
    arms.logprobs["vllm_lora"]["s0"] = [v + 1.0 for v in arms.logprobs["vllm_lora"]["s0"]]
    served_wrong = lp.compare(arms, _delta_seqs(), lp.ParityThresholds(bucket_edges=(), require_effect=False))
    assert any("exceeds" in f for f in served_wrong.failures)


def test_a_degenerate_completion_fails_and_is_located():
    run = (7,) * 40
    seqs = [lp.FrozenSequence("s", tuple(range(10)), tuple(range(20)) + run + (1, 2))]
    n = len(seqs[0].token_ids) - 1
    report = lp.compare(_arms(n), seqs)
    assert report.degenerate == {"s": {"token": 7, "run": 40, "start": 20}}
    assert any("repeat one token 32+ times" in f and "#52568" in f for f in report.failures)
    assert lp.compare(_arms(n), seqs, lp.ParityThresholds(degenerate_run=0)).passed
    assert lp.compare(_arms(n), seqs, lp.ParityThresholds(degenerate_run=41)).passed


def test_longest_run():
    assert lp.longest_run(()) == (None, 0, 0)
    assert lp.longest_run((5,)) == (5, 1, 0)
    assert lp.longest_run((1, 2, 2, 3, 3, 3, 2)) == (3, 3, 3)
    assert lp.longest_run((4, 4, 1)) == (4, 2, 0)


def test_the_completion_segment_gates_only_completion_positions():
    seqs = _seq(300, prompt=200)
    arms = _arms(300)
    arms.logprobs["vllm_lora"]["s"] = [v - 0.5 if i < 150 else v for i, v in enumerate(arms.logprobs["vllm_lora"]["s"])]
    assert not lp.compare(arms, seqs).passed
    report = lp.compare(arms, seqs, segment="completion")
    assert report.passed, report.failures
    assert report.segment == "completion" and report.buckets["all"].tokens == 101
    assert report.segments["prompt"].tokens == 199
    arms.logprobs["vllm_lora"]["s"][0] = math.nan
    assert any("non-finite" in f for f in lp.compare(arms, seqs, segment="completion").failures)
    with pytest.raises(lp.ParityError, match="segment must be one of"):
        lp.compare(arms, seqs, segment="decode")


def test_rho_is_one_for_a_copy_half_for_half_and_its_interval_is_over_sequences():
    faithful = lp.compare(_delta_arms(arm="vllm_merged", n_seqs=20), _delta_seqs(n_seqs=20), lp.ParityThresholds(bucket_edges=()))
    stats = faithful.buckets["all"]
    assert faithful.mode == "merged" and faithful.served_arm == "vllm_merged"
    assert stats.rho == pytest.approx(1.0, abs=0.01) and stats.correlation > 0.99
    assert stats.rho_ci[0] <= stats.rho <= stats.rho_ci[1] and stats.rho_halfwidth < 0.01
    half = lp.compare(_delta_arms(arm="vllm_merged", n_seqs=20, served=lambda a, m, o: 0.5 * (a + m)),
                      _delta_seqs(n_seqs=20), lp.ParityThresholds(bucket_edges=()))
    assert half.buckets["all"].rho == pytest.approx(0.5, abs=0.01)


def test_rho_is_gated_where_it_is_precise_and_reported_where_it_is_not():
    gate = dict(bucket_edges=(10,), rho_range=(0.99, 1.01), rho_precision=0.01, paired_delta=1.0, effect_ratio=(0.5, 1.5))
    seqs = _delta_seqs(n_seqs=20)
    assert lp.compare(_delta_arms(arm="vllm_merged", n_seqs=20), seqs, lp.ParityThresholds(**gate)).passed
    scaled = _delta_arms(arm="vllm_merged", n_seqs=20, served=lambda a, m, o: 0.95 * (a + m))
    report = lp.compare(scaled, seqs, lp.ParityThresholds(**gate))
    assert report.failures and all(f.startswith("all") and "retention rho" in f for f in report.failures)
    per_bucket = lp.compare(scaled, seqs, lp.ParityThresholds(**gate, rho_per_bucket=True))
    assert {f.split(":")[0] for f in per_bucket.failures if "retention rho" in f} == {"all", "[0,10)", "[10,inf)"}
    assert report.rho_gated == {"all": True} and per_bucket.rho_gated == {"all": True, "[0,10)": True, "[10,inf)": True}
    imprecise = lp.compare(scaled, seqs, lp.ParityThresholds(**{**gate, "rho_precision": 1e-6}))
    assert imprecise.rho_gated == {"all": True} and not any("not gated" in note for note in imprecise.notes)
    assert [f.split(" (")[0] for f in imprecise.failures] == ["all: retention rho 0.9499"]
    assert lp.compare(scaled, seqs, lp.ParityThresholds(**{**gate, "rho_precision": None})).failures
    noisy, noisy_seqs = _paired_arms(served_noise=0.3)
    straddling = lp.compare(noisy, noisy_seqs, lp.ParityThresholds(**LOOSE, rho_range=(0.99, 1.01), rho_precision=0.01))
    stats = straddling.buckets["all"]
    assert stats.rho_halfwidth > 0.01 and stats.rho_ci[0] < 0.99 and stats.rho_ci[1] > 1.01
    assert straddling.passed and straddling.rho_gated == {"all": False}, straddling.failures
    assert any(note.startswith("all: rho ") and "straddles a bound of [0.99, 1.01]" in note for note in straddling.notes)
    # and one wholly inside it passes: 0.9993-1.0004 decides 0.99-1.01 at any precision
    inside = lp.compare(_delta_arms(arm="vllm_merged", n_seqs=20), seqs, lp.ParityThresholds(**{**gate, "rho_precision": 1e-6}))
    assert inside.passed and inside.rho_gated == {"all": True} and inside.buckets["all"].rho_halfwidth > 1e-6
    assert lp.compare(scaled, seqs, lp.ParityThresholds(bucket_edges=(10,), paired_delta=1.0)).rho_gated == {}


def test_a_requested_rho_gate_that_a_weak_effect_skipped_is_reported_as_not_gated():
    gate = dict(bucket_edges=(), rho_range=(0.99, 1.01), rho_precision=0.01, paired_delta=1.0, min_effect=1e9)
    arms, seqs = _delta_arms(arm="vllm_merged", n_seqs=20), _delta_seqs(n_seqs=20)
    inconclusive = lp.compare(arms, seqs, lp.ParityThresholds(**gate, require_effect=False))
    assert inconclusive.passed and inconclusive.inconclusive and inconclusive.rho_gated == {"all": False}
    failed = lp.compare(arms, seqs, lp.ParityThresholds(**gate))
    assert not failed.passed and failed.rho_gated == {"all": False}
    assert lp.compare(arms, seqs, lp.ParityThresholds(bucket_edges=(), min_effect=1e9, require_effect=False)).rho_gated == {}


def test_rho_without_a_resample_that_moves_has_no_interval():
    tokens = [lp._Token(1, -1.0, -0.5, -1.0, -0.5, None, None)]
    thresholds = next(
        t for seed in range(50)
        if math.isnan(lp._rho(tokens, 2, t := lp.ParityThresholds(rho_resamples=1, rho_seed=seed))[1][0])
    )
    rho, interval, halfwidth = lp._rho(tokens, 2, thresholds)
    assert rho == 1.0 and math.isnan(interval[1]) and math.isnan(halfwidth)
    assert math.isnan(lp._correlation(tokens))
    assert math.isnan(lp._correlation(tokens * 2))


def test_greedy_agreement_is_gated_where_both_arms_recorded_their_argmax():
    arms = _arms(200, served="vllm_merged")
    argmax = {arm: {"s": [i % 7 for i in range(200)]} for arm in arms.logprobs}
    agreed = lp.compare(_scores(arms.logprobs, argmax), _seq(200), lp.ParityThresholds(bucket_edges=()))
    assert agreed.passed and agreed.buckets["all"].greedy_served == 1.0 == agreed.buckets["all"].greedy_base
    argmax["vllm_merged"]["s"] = [(i % 7) + (i % 10 == 0) for i in range(200)]  # 10% disagree
    argmax["hf_base"]["s"][0] = None  # unknown positions are left out
    report = lp.compare(_scores(arms.logprobs, argmax), _seq(200), lp.ParityThresholds(bucket_edges=()))
    assert report.buckets["all"].greedy_served == pytest.approx(0.9)
    assert any("agree on the most likely token at 0.900" in f for f in report.failures)
    assert lp.compare(_scores(arms.logprobs, argmax), _seq(200), lp.ParityThresholds(bucket_edges=(), greedy_agreement=None)).passed


def _reference(n_prompt=10, n_completion=120, gap=0.05):
    seq = lp.FrozenSequence("ref-0000", tuple(range(n_prompt)), tuple(range(n_completion)))
    prefill = [-1.0] * (len(seq.token_ids) - 1)
    decoded = [v + (gap if i >= n_prompt - 1 else 0.0) for i, v in enumerate(prefill)]
    decoded[-1] = math.nan
    return _scores({"vllm_base": {seq.seq_id: prefill}, "vllm_base_decode": {seq.seq_id: decoded}}), [seq]


def test_the_decode_floor_is_the_base_decode_paths_gap_on_completion_positions():
    scores, seqs = _reference()
    floor = lp.decode_noise_floor(scores, seqs, lp.ParityThresholds(bucket_edges=(64,)))
    assert floor.buckets["all"] == (119, pytest.approx(0.05))
    assert floor.buckets["[0,64)"][0] + floor.buckets["[64,inf)"][0] == 119
    assert floor.meta is None
    scores.meta[lp.BASE_DECODE_ARM] = {"decode_batch": 2}
    assert lp.decode_noise_floor(scores, seqs).meta == {"decode_batch": 2}
    with pytest.raises(lp.ParityError, match="no vllm_base_decode scores"):
        lp.decode_noise_floor(_scores({"vllm_base": scores.logprobs["vllm_base"]}), seqs)


def test_the_decode_arm_allows_for_the_decode_floor():
    seqs = _seq(300, prompt=200)
    arms = _arms(300, effect=0.6)
    arms.logprobs["vllm_lora"]["s"] = [v + 0.02 for v in arms.logprobs["vllm_lora"]["s"]]
    assert any("exceeds" in f for f in lp.compare(arms, seqs, segment="completion").failures)
    thresholds = lp.ParityThresholds(bucket_edges=(250,), min_bucket_tokens=40)
    floor = lp.DecodeFloor({"all": (119, 0.02), "[0,250)": (60, 0.02), "[250,inf)": (5, 0.9)})
    report = lp.compare(arms, seqs, thresholds, segment="completion", decode_floor=floor)
    assert report.passed, report.failures
    assert report.buckets["all"].noise == pytest.approx(0.03) and report.buckets["all"].decode_noise == 0.02
    assert report.buckets["[250,inf)"].decode_noise == 0.02
    with pytest.raises(lp.ParityError, match="decode floor applies"):
        lp.compare(arms, seqs, decode_floor=floor)
    empty = lp.compare(arms, seqs, segment="completion", decode_floor=lp.DecodeFloor({"all": (0, math.nan)}))
    assert any("no decode floor was added" in note for note in empty.notes)


def _recorded(arms, meta):
    return lp.Scores(arms.logprobs, arms.argmax, {arm: dict(value) for arm, value in meta.items()})


FP32_HF = {"hf_base": {"dtype": "float32"}, "hf_lora": {"dtype": "float32"}}


def test_the_inputs_each_file_recorded_are_copied_into_the_report():
    arms = _arms(100, served="vllm_merged")
    meta = {**FP32_HF, "vllm_base": {"engine_dtype": "float32"}, "vllm_merged": {"engine_dtype": "float32"}}
    report = lp.compare(_recorded(arms, meta), _seq(100))
    assert report.passed, report.failures
    assert report.inputs == {
        "engine_dtype": {"vllm_base": "float32", "vllm_merged": "float32"},
        **{name: {"vllm_base": None, "vllm_merged": None} for name in lp.ENGINE_FIELDS if name != "engine_dtype"},
        "hf_dtype": {"hf_base": "float32", "hf_lora": "float32"},
    }
    assert report.to_json()["inputs"] == report.inputs
    assert lp.compare(arms, _seq(100)).inputs == {**{name: {} for name in lp.ENGINE_FIELDS}, "hf_dtype": {}}


def test_vllm_files_served_in_different_dtypes_are_refused():
    arms = _arms(100, served="vllm_merged")
    meta = {**FP32_HF, "vllm_base": {"engine_dtype": "bfloat16"}, "vllm_merged": {"engine_dtype": "float32"}}
    with pytest.raises(lp.ParityError, match=r"different engine_dtype \{'vllm_base': 'bfloat16', 'vllm_merged': 'float32'\}"):
        lp.compare(_recorded(arms, meta), _seq(100))
    meta["vllm_base"] = {}
    with pytest.raises(lp.ParityError, match="different engine_dtype"):
        lp.compare(_recorded(arms, meta), _seq(100))


@pytest.mark.parametrize("field, first, second", [
    ("precision_env", {"TRITON_F32_DEFAULT": "unset"}, {"TRITON_F32_DEFAULT": "tf32x3"}),
    ("precision_env", {"VLLM_BATCH_INVARIANT": "0"}, {"VLLM_BATCH_INVARIANT": "1"}),
    ("attention_backend", "TRITON_ATTN", "FLEX_ATTENTION"),
    ("vllm_version", "0.28.0+cu129", "0.28.1+cu129"),
    ("max_num_batched_tokens", 4096, 8192),
])
def test_vllm_files_from_engines_configured_otherwise_are_refused(field, first, second):
    arms = _arms(100, served="vllm_merged")
    same = {"engine_dtype": "float32", field: first}
    assert lp.compare(_recorded(arms, {**FP32_HF, "vllm_base": same, "vllm_merged": same}), _seq(100)).passed
    other = {**same, field: second}
    with pytest.raises(lp.ParityError, match=f"different {field}"):
        lp.compare(_recorded(arms, {**FP32_HF, "vllm_base": same, "vllm_merged": other}), _seq(100))


def test_hf_arms_must_be_float32_unless_allowed_on_the_record():
    arms = _arms(100)
    meta = {"hf_base": {"dtype": "bfloat16"}, "hf_lora": {"dtype": "bfloat16"}}
    with pytest.raises(lp.ParityError, match=r"HF arm\(s\) scored in .*bfloat16.*not float32"):
        lp.compare(_recorded(arms, meta), _seq(100))
    with pytest.raises(lp.ParityError, match="not float32"):
        lp.compare(_recorded(arms, {"hf_base": {}, "hf_lora": {"dtype": "float32"}}), _seq(100))
    allowed = lp.compare(_recorded(arms, meta), _seq(100), lp.ParityThresholds(require_hf_float32=False))
    assert allowed.passed and allowed.inputs["hf_dtype"] == {"hf_base": "bfloat16", "hf_lora": "bfloat16"}
    assert allowed.thresholds["require_hf_float32"] is False


def test_the_decode_arm_and_its_reference_share_one_decode_batch():
    seqs = _seq(300, prompt=200)
    arms = _arms(300, effect=0.6)
    served = {**FP32_HF, "vllm_base": {"decode_batch": 2}, "vllm_lora": {"decode_batch": 2}}
    floor = lp.DecodeFloor({"all": (119, 0.0)}, {"decode_batch": 2})
    report = lp.compare(_recorded(arms, served), seqs, segment="completion", decode_floor=floor)
    assert report.passed, report.failures
    assert report.inputs["decode_batch"] == {"decode_arm": 2, "reference": 2}
    with pytest.raises(lp.ParityError, match="decode batch 2, its reference at 4"):
        lp.compare(_recorded(arms, served), seqs, segment="completion",
                   decode_floor=lp.DecodeFloor({"all": (119, 0.0)}, {"decode_batch": 4}))
    # the reference's engine dtype joins the vLLM files'
    with pytest.raises(lp.ParityError, match="'decode_reference': 'float32'"):
        lp.compare(_recorded(arms, served), seqs, segment="completion",
                   decode_floor=lp.DecodeFloor({"all": (119, 0.0)}, {"decode_batch": 2, "engine_dtype": "float32"}))
    unrecorded = lp.compare(arms, seqs, segment="completion", decode_floor=floor)
    assert unrecorded.inputs["decode_batch"] == {"decode_arm": None, "reference": 2}
    assert lp.compare(_recorded(arms, served), seqs, segment="completion",
                      decode_floor=lp.DecodeFloor({"all": (119, 0.0)})).inputs["decode_batch"] == {
        "decode_arm": 2, "reference": None}


def test_a_model_config_must_be_local(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/root")
    assert str(lp.load_model_config(QWEN3_CONFIG).model_path) == "/root/Qwen3-8B"
    api = tmp_path / "api.yaml"
    api.write_text("model_type: openai\n")
    with pytest.raises(lp.ParityError, match="not a local model config"):
        lp.load_model_config(api)


@pytest.fixture
def captured_engine(monkeypatch):
    import ruleforge.judges.lora as lora_module
    import ruleforge.judges.models as models_module

    seen = {}

    def _init(**kwargs):
        seen.update(kwargs)
        return "LLM", "TOK"

    monkeypatch.setattr(models_module, "init_local_model", _init)
    monkeypatch.setattr(models_module, "engine_argument_names", frozenset)

    def metadata_of(path):
        return {"name": "adapter", "max_lora_rank": lora_module.resolve_lora_max_rank(path)}

    monkeypatch.setattr(lora_module, "build_lora_request", lambda path: (("REQ", path), metadata_of(path)))
    monkeypatch.setattr(lp, "require", lambda purpose, *modules: None)  # vLLM stubbed here
    return seen


def test_the_engine_without_vllm_is_refused_in_one_line_before_any_part_of_it_is_reached(tmp_path, monkeypatch):
    import ruleforge.judges.lora as lora_module
    import ruleforge.judges.models as models_module
    from ruleforge.core.optional import MissingDependency

    def reached(*args, **kwargs):
        raise AssertionError("a part of the vLLM stack was reached")

    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/root")
    monkeypatch.setitem(sys.modules, "vllm", None)
    for name in ("engine_argument_names", "init_local_model"):
        monkeypatch.setattr(models_module, name, reached)
    monkeypatch.setattr(lora_module, "build_lora_request", reached)
    adapter = _write_adapter(_adapter_dir(tmp_path), {"x.lora_A.weight": (32, 4)}, {"r": 32})
    with pytest.raises(MissingDependency) as refused:
        lp.build_vllm_engine(lp.load_model_config(QWEN3_CONFIG), adapter)
    assert str(refused.value) == ("starting the parity engine needs vllm, which this environment lacks: add --group "
                                  "vllm to uv sync, or pip install -e '.[vllm]' from the checkout")


def test_the_engine_is_the_campaigns_with_the_adapter_or_the_merged_weights(tmp_path, monkeypatch, captured_engine):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/root")
    seen = captured_engine
    adapter = _write_adapter(_adapter_dir(tmp_path), {"x.lora_A.weight": (32, 4)}, {"r": 32})
    llm, tok, request, metadata = lp.build_vllm_engine(
        lp.load_model_config(QWEN3_CONFIG), adapter, engine_args={"max_num_batched_tokens": 4096}
    )
    assert (llm, tok, metadata) == ("LLM", "TOK", {"name": "adapter", "max_lora_rank": 32})
    assert request == ("REQ", adapter / lp.ADAPTER_WEIGHTS_FNAME)
    # the rank ceiling is the adapter's own r (build_lora_request's metadata), read before the engine exists
    assert seen["enable_lora"] is True and seen["max_lora_rank"] == 32
    assert seen["model_path"] == "/root/Qwen3-8B" and seen["attention_backend"] == "FLASH_ATTN"
    assert seen["engine_args"] == {"max_num_batched_tokens": 4096}
    assert seen["dtype"] is None  # vLLM's auto
    # merged: the merged weights on the base's engine configuration, no LoRA, in fp32
    _, _, request, metadata = lp.build_vllm_engine(
        lp.load_model_config(QWEN3_CONFIG), model_path=tmp_path / "merged", dtype="float32"
    )
    assert (request, metadata) == (None, None)
    assert seen["enable_lora"] is False and seen["max_lora_rank"] is None
    assert seen["model_path"] == str(tmp_path / "merged") and seen["engine_args"] == {}
    assert seen["dtype"] == "float32"
    # processed logprobs would compare the sampler, not the model
    with pytest.raises(lp.ParityError, match="raw logprobs"):
        lp.build_vllm_engine(lp.load_model_config(QWEN3_CONFIG), adapter, engine_args={"logprobs_mode": "processed_logprobs"})


def test_the_fp32_config_serves_float32_unless_the_tool_asks_for_another_dtype(monkeypatch, captured_engine):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", "/root")
    lp.build_vllm_engine(lp.load_model_config(QWEN3_FP32_CONFIG), model_path=Path("/m/merged"))
    assert (captured_engine["dtype"], captured_engine["attention_backend"]) == ("float32", "TRITON_ATTN")
    lp.build_vllm_engine(lp.load_model_config(QWEN3_FP32_CONFIG), dtype="bfloat16")
    assert captured_engine["dtype"] == "bfloat16"


def test_an_engine_is_expected_in_the_dtype_its_config_declares(tmp_path, monkeypatch, fake_vllm):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    argv = ["vllm", "--model-config", str(QWEN3_FP32_CONFIG), "--arm", "vllm_base", "--prompts",
            str(_prompts(tmp_path)), "--sequences", str(tmp_path / "s.jsonl"), "--out", str(tmp_path / "b.json")]
    built = _patch_engine(monkeypatch, _EngineLLM(dtype=torch.bfloat16), _Tokenizer())
    with pytest.raises(lp.ParityError, match="serves the model in bfloat16, not float32"):
        lp.main(argv)
    assert built[-1]["dtype"] is None  # no --dtype: the config's key is the engine's (build_vllm_engine)
    # --expect-dtype still says otherwise
    assert lp.main([*argv, "--expect-dtype", "bfloat16"]) == 0


def test_the_engine_dtype_is_read_off_the_engine():
    assert lp.engine_dtype(types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=torch.float32))) == "float32"
    assert lp.engine_dtype(object()) is None


def test_the_engine_record_names_what_changes_the_engines_arithmetic(monkeypatch):
    monkeypatch.setenv("TRITON_F32_DEFAULT", "tf32x3")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)  # the rest at their defaults, whatever the shell set
    monkeypatch.delenv("VLLM_FLOAT32_MATMUL_PRECISION", raising=False)
    monkeypatch.setattr(lp, "_package_version", lambda name: {"vllm": "0.28.0+cu129"}[name])
    scheduler = types.SimpleNamespace(max_num_batched_tokens=4096)
    cache = types.SimpleNamespace(cache_dtype="auto")
    llm = types.SimpleNamespace(
        model_config=types.SimpleNamespace(dtype=torch.float32),
        llm_engine=types.SimpleNamespace(vllm_config=types.SimpleNamespace(scheduler_config=scheduler, cache_config=cache)),
    )
    record = lp.engine_record(llm, "TRITON_ATTN")
    assert set(record) == set(lp.ENGINE_FIELDS)
    assert record == {
        "engine_dtype": "float32", "kv_cache_dtype": "float32", "attention_backend": "TRITON_ATTN",
        "vllm_version": "0.28.0+cu129", "max_num_batched_tokens": 4096,
        "precision_env": {"VLLM_FLOAT32_MATMUL_PRECISION": "highest", "TRITON_F32_DEFAULT": "tf32x3",
                          "VLLM_BATCH_INVARIANT": "0"},
    }
    cache.cache_dtype = "fp8_e4m3"
    assert lp.kv_cache_dtype(llm) == "fp8_e4m3"
    bare = lp.engine_record(object(), None)
    assert (bare["engine_dtype"], bare["kv_cache_dtype"], bare["attention_backend"], bare["max_num_batched_tokens"]) == (
        None, None, None, None)
    monkeypatch.undo()
    assert lp._package_version("no-such-package-anywhere") is None


def test_engine_args_parse_json_values_and_refuse_a_bare_word():
    assert lp._parse_engine_args(["a=4096", "b=0.8", "c=false", "d=FLASH_ATTN"]) == {
        "a": 4096, "b": 0.8, "c": False, "d": "FLASH_ATTN"
    }
    with pytest.raises(lp.ParityError, match="KEY=VALUE"):
        lp._parse_engine_args(["oops"])


def test_cli_make_adapter_and_inspect(llama_dir, tmp_path, capsys):
    out = tmp_path / "adapter"
    code = lp.main(["make-adapter", "--model-path", str(llama_dir), "--out", str(out),
                    "--targets", "q_proj,v_proj", "--rank", "2", "--alpha", "4"])
    assert code == 0 and json.loads(capsys.readouterr().out)["max_rank"] == 2
    assert lp.main(["inspect", "--adapter", str(out), "--model-path", str(llama_dir)]) == 0
    capsys.readouterr()
    bad = _write_adapter(tmp_path / "bad", {"x.lora_A.weight": (2, 2)})
    assert lp.main(["inspect", "--adapter", str(bad)]) == 1


def test_cli_make_adapter_reports_a_problem_with_exit_1(llama_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lp, "make_random_adapter", lambda *a, **k: lp.AdapterReport(
        None, None, 0, (), (), (), (), ("broken",)))
    assert lp.main(["make-adapter", "--model-path", str(llama_dir), "--out", str(tmp_path)]) == 1


class _EngineLLM(_ScoringLLM):
    def __init__(self, dtype=None, kv_cache_dtype="auto", **kwargs):
        super().__init__(**kwargs)
        if dtype is not None:
            self.model_config = types.SimpleNamespace(dtype=dtype)
            cache = types.SimpleNamespace(cache_dtype=kv_cache_dtype)
            self.llm_engine = types.SimpleNamespace(vllm_config=types.SimpleNamespace(cache_config=cache))

    def generate(self, prompts, sampling_params=None, use_tqdm=True, lora_request=None):
        if isinstance(prompts[0], dict):
            return super().generate(prompts, sampling_params, use_tqdm, lora_request)
        if sampling_params[0].kwargs.get("logprobs") is not None:
            return _DecodingLLM().generate(prompts, sampling_params, use_tqdm, lora_request)
        return [types.SimpleNamespace(prompt_token_ids=[1, 2], outputs=[types.SimpleNamespace(token_ids=[3], finish_reason="stop")])
                for _ in prompts]


def _patch_engine(monkeypatch, llm, tokenizer=None):
    built = []

    def _build(cfg, adapter=None, *, model_path=None, dtype=None, engine_args=None):
        built.append({"adapter": adapter, "model_path": model_path, "dtype": dtype, "engine_args": engine_args})
        return llm, tokenizer, None if adapter is None else "REQ", None if adapter is None else {"name": "a"}

    monkeypatch.setattr(lp, "build_vllm_engine", _build)
    return built


def _prompts(tmp_path, n=3):
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps({"system_prompt": f"S{i}"}) + "\n" for i in range(n)))
    return path


def test_cli_vllm_lora_mode_runs_prefill_decode_and_the_decode_reference(tmp_path, monkeypatch, fake_vllm, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    (tmp_path / lp.ADAPTER_WEIGHTS_FNAME).write_bytes(b"factors")  # the adapter the arms record by digest
    (tmp_path / lp.ADAPTER_CONFIG_FNAME).write_text("{}")
    built = _patch_engine(monkeypatch, _EngineLLM(dtype=torch.bfloat16), _Tokenizer())
    prompts = _prompts(tmp_path)
    out, dec, dec_out = tmp_path / "v.json", tmp_path / "d.jsonl", tmp_path / "vd.json"
    ref, ref_out = tmp_path / "r.jsonl", tmp_path / "vr.json"
    argv = ["vllm", "--model-config", str(QWEN3_CONFIG), "--adapter", str(tmp_path), "--prompts", str(prompts),
            "--sequences", str(tmp_path / "s.jsonl"), "--out", str(out), "--engine-arg", "max_num_batched_tokens=2048",
            "--decode-sequences", str(dec), "--decode-out", str(dec_out), "--decode-reference", str(ref),
            "--decode-reference-out", str(ref_out), "--decode-count", "2", "--decode-batch", "1",
            "--decode-max-tokens", "5", "--expect-dtype", "bfloat16"]
    assert lp.main(argv) == 0
    assert built == [{"adapter": tmp_path, "model_path": None, "dtype": None,
                      "engine_args": {"max_num_batched_tokens": 2048}}]
    summary = json.loads(capsys.readouterr().out)
    assert summary["engine_dtype"] == "bfloat16"
    assert set(summary["seconds"]) == {"sample", "prefill", "decode", "decode_reference"}
    assert summary["decode"] == {"sequences": 2, "completion_tokens": 4, "sampled": True, "out": str(dec_out)}
    assert summary["decode_reference"] == {"sequences": 2, "out": str(ref_out)}
    assert lp.read_sequences(tmp_path / "s.jsonl") == [lp.FrozenSequence(f"seq-{i:04d}", (1, 2), (3,), "stop") for i in range(3)]
    assert set(lp.read_scores(out).logprobs) == {"vllm_base", "vllm_lora"}
    prefill_meta = json.loads(out.read_text())["meta"]
    assert prefill_meta["arm"] == "prefill" and prefill_meta["servings"] == ["vllm_base", "vllm_lora"]
    assert set(lp.ENGINE_FIELDS) <= set(prefill_meta) and prefill_meta["attention_backend"] == "FLASH_ATTN"
    assert prefill_meta["elapsed_seconds"] >= 0 and json.loads(dec_out.read_text())["meta"]["elapsed_seconds"] >= 0
    decoded = lp.read_scores(dec_out)
    # vllm_base: the no-adapter prefill over the decoded ids; vllm_lora: the decode arm's own logprobs
    assert decoded.logprobs["vllm_base"]["dec-0000"] == pytest.approx([-0.3, -2.0, -1.2, -1.2])
    assert decoded.logprobs["vllm_lora"]["dec-0000"] == pytest.approx([-0.2, -1.8, 0.41, 0.42])
    assert json.loads(dec_out.read_text())["meta"]["sampled"] is True
    reference = lp.read_scores(ref_out)
    assert set(reference.logprobs) == {"vllm_base", "vllm_base_decode"}
    assert [seq.seq_id for seq in lp.read_sequences(ref)] == ["ref-0000", "ref-0001"]
    # a second run would re-label prefill scores as the decode arm's
    with pytest.raises(lp.ParityError, match="remove it to sample again"):
        lp.main(argv)
    dec.unlink()
    with pytest.raises(lp.ParityError, match="not float32"):
        lp.main([*argv[:-1], "float32"])
    # --dtype asks the engine for a dtype and, unless --expect-dtype says otherwise, expects it
    with pytest.raises(lp.ParityError, match="serves the model in bfloat16, not float32"):
        lp.main([*argv[:-2], "--dtype", "float32"])
    assert built[-1]["dtype"] == "float32"
    _patch_engine(monkeypatch, _EngineLLM(dtype=torch.bfloat16, kv_cache_dtype="fp8"), _Tokenizer())
    with pytest.raises(lp.ParityError, match="keeps its KV cache in fp8, not bfloat16"):
        lp.main(argv)


def test_cli_vllm_merged_mode_two_engines_then_compare(tmp_path, monkeypatch, fake_vllm, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    prompts = _prompts(tmp_path)
    seqs, dec = tmp_path / "s.jsonl", tmp_path / "d.jsonl"
    common = ["vllm", "--model-config", str(QWEN3_CONFIG), "--prompts", str(prompts), "--sequences", str(seqs),
              "--decode-count", "2", "--decode-batch", "2", "--truncation", "none"]
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "merge_manifest.json").write_text(json.dumps({"adapter": {"sha256": "a" * 64, "config_sha256": "c" * 64}}))
    built = _patch_engine(monkeypatch, _EngineLLM(shift=0.0), _Tokenizer())
    assert lp.main([*common, "--arm", "vllm_merged", "--model-path", str(tmp_path / "merged"),
                    "--out", str(tmp_path / "m.json"), "--decode-sequences", str(dec),
                    "--decode-out", str(tmp_path / "md.json")]) == 0
    assert built[-1]["model_path"] == tmp_path / "merged" and built[-1]["adapter"] is None
    assert set(lp.read_scores(tmp_path / "m.json").logprobs) == {"vllm_merged"}
    assert set(lp.read_scores(tmp_path / "md.json").logprobs) == {"vllm_merged"}
    with pytest.raises(lp.ParityError, match="sampled from vllm_base"):
        lp.main([*common, "--arm", "vllm_merged", "--out", str(tmp_path / "x.json"),
                 "--decode-reference", str(tmp_path / "r.jsonl"), "--decode-reference-out", str(tmp_path / "vr.json")])
    assert lp.main([*common, "--arm", "vllm_base", "--out", str(tmp_path / "b.json"), "--decode-sequences", str(dec),
                    "--decode-out", str(tmp_path / "bd.json"), "--decode-reference", str(tmp_path / "r.jsonl"),
                    "--decode-reference-out", str(tmp_path / "vr.json")]) == 0
    capsys.readouterr()
    assert json.loads((tmp_path / "bd.json").read_text())["meta"]["sampled"] is False
    base, merged = lp.read_scores(tmp_path / "b.json"), lp.read_scores(tmp_path / "m.json")
    lp.write_scores(tmp_path / "h.json", lp.Scores({"hf_base": base.logprobs["vllm_base"],
                                                   "hf_lora": merged.logprobs["vllm_merged"]}),
                    {"dtype": "float32", **ADAPTER}, lp.read_sequences(seqs))
    assert lp.read_scores(tmp_path / "m.json").meta["vllm_merged"]["adapter_sha256"] == "a" * 64
    gate = ["compare", "--sequences", str(seqs), "--hf", str(tmp_path / "h.json"), "--vllm", str(tmp_path / "b.json"),
            "--vllm", str(tmp_path / "m.json"), "--report", str(tmp_path / "rep.json"), "--allow-weak-effect"]
    assert lp.main(gate) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "merged" and printed["passed"] is True and printed["inconclusive"]
    lp.write_scores(tmp_path / "hd.json", lp.Scores({
        "hf_base": lp.read_scores(tmp_path / "bd.json").logprobs["vllm_base"],
        "hf_lora": lp.read_scores(tmp_path / "md.json").logprobs["vllm_merged"],
    }), {"dtype": "float32", **ADAPTER}, lp.read_sequences(dec))
    decode_gate = ["compare", "--segment", "completion", "--sequences", str(dec), "--hf", str(tmp_path / "hd.json"),
                   "--vllm", str(tmp_path / "bd.json"), "--vllm", str(tmp_path / "md.json"),
                   "--decode-reference-sequences", str(tmp_path / "r.jsonl"), "--decode-reference", str(tmp_path / "vr.json"),
                   "--report", str(tmp_path / "rd.json"), "--allow-weak-effect", "--degenerate-run", "0"]
    assert lp.main(decode_gate) == 0
    report = json.loads((tmp_path / "rd.json").read_text())
    assert report["segment"] == "completion" and report["buckets"]["all"]["decode_noise"] >= 0
    assert report["inputs"]["decode_batch"] == {"decode_arm": 2, "reference": 2}
    assert report["inputs"]["hf_dtype"] == {"hf_base": "float32", "hf_lora": "float32"}


def test_cli_vllm_refuses_what_it_cannot_run(tmp_path, monkeypatch, fake_vllm):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    _patch_engine(monkeypatch, _EngineLLM(), _Tokenizer())
    base = ["vllm", "--model-config", str(QWEN3_CONFIG), "--out", str(tmp_path / "v.json")]
    seqs = tmp_path / "s.jsonl"
    lp.write_sequences(seqs, [lp.FrozenSequence("a", (1, 2), (3,))])
    with pytest.raises(lp.ParityError, match="no --prompts"):
        lp.main([*base, "--arm", "vllm_base", "--sequences", str(tmp_path / "absent.jsonl")])
    with pytest.raises(lp.ParityError, match="sampled by the served arm's engine"):
        lp.main([*base, "--arm", "vllm_base", "--sequences", str(seqs), "--decode-sequences", str(tmp_path / "d.jsonl"),
                 "--decode-out", str(tmp_path / "vd.json")])
    with pytest.raises(lp.ParityError, match="the decode arms sample from --prompts"):
        lp.main([*base, "--arm", "vllm_base", "--sequences", str(seqs), "--decode-reference", str(tmp_path / "r.jsonl"),
                 "--decode-reference-out", str(tmp_path / "vr.json")])
    with pytest.raises(lp.ParityError, match="go together"):
        lp.main([*base, "--arm", "vllm_base", "--sequences", str(seqs), "--decode-out", str(tmp_path / "vd.json")])
    with pytest.raises(SystemExit):
        lp.main([*base, "--adapter", str(tmp_path), "--arm", "vllm_base", "--sequences", str(seqs)])


def test_cli_vllm_reuses_frozen_sequences_and_samples_the_model_default(tmp_path, monkeypatch, fake_vllm, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    (tmp_path / lp.ADAPTER_WEIGHTS_FNAME).write_bytes(b"factors")  # the adapter the arms record by digest
    (tmp_path / lp.ADAPTER_CONFIG_FNAME).write_text("{}")
    weights = tmp_path / "Qwen3-8B"
    weights.mkdir()
    (weights / "generation_config.json").write_text(json.dumps({"top_k": 20, "top_p": 0.95}))
    seen = {}
    monkeypatch.setattr(lp, "sample_sequences", lambda *a, **k: seen.update(k) or [lp.FrozenSequence("a", (1,), (2,))])
    _patch_engine(monkeypatch, _ScoringLLM())
    base = ["vllm", "--model-config", str(QWEN3_CONFIG), "--adapter", str(tmp_path), "--out", str(tmp_path / "v.json")]
    assert lp.main([*base, "--prompts", str(_prompts(tmp_path)), "--sequences", str(tmp_path / "s.jsonl")]) == 0
    assert seen["truncation"]["top_k"] == 20 and seen["truncation"]["top_p"] == 0.95
    assert seen["gpt_oss_reasoning_level"] is None
    assert lp.main([*base, "--sequences", str(tmp_path / "s.jsonl"), "--truncation", "none"]) == 0
    assert lp.read_scores(tmp_path / "v.json").logprobs["vllm_lora"]["a"] == pytest.approx([0.7])


def test_cli_hf_scores_both_arms_with_one_adapter_wrap_in_exact_float32(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    (tmp_path / lp.ADAPTER_WEIGHTS_FNAME).write_bytes(b"factors")  # the adapter the arms record by digest
    (tmp_path / lp.ADAPTER_CONFIG_FNAME).write_text("{}")
    prefill, decode = tmp_path / "s.jsonl", tmp_path / "d.jsonl"
    lp.write_sequences(prefill, [lp.FrozenSequence("seq-0000", (1, 2), (3,))])
    lp.write_sequences(decode, [lp.FrozenSequence("dec-0000", (1,), (4, 5))])
    scored, loaded = [], {}
    monkeypatch.setattr(lp, "load_hf_model", lambda path, **kw: loaded.update(path=path, **kw) or "MODEL")
    monkeypatch.setattr(lp, "exact_float32", lambda: {"float32_matmul_precision": "highest"})
    monkeypatch.setattr(lp, "wrap_adapter", lambda model, adapter: ("PEFT", model, adapter))

    def _score(model, adapter, sequences, device, arm="hf_base", autocast_arm=False, wrapped=None):
        assert wrapped == (None if adapter is None else ("PEFT", model, adapter))
        scored.append([seq.seq_id for seq in sequences])
        rows = {seq.seq_id: [0.0] * (len(seq.token_ids) - 1) for seq in sequences}
        arms = lp.HF_ARMS + ((lp.HF_LORA_AC_ARM,) if autocast_arm else ()) if adapter else (arm,)
        return lp.Scores({name: dict(rows) for name in arms})

    monkeypatch.setattr(lp, "score_hf", _score)
    base = ["hf", "--model-config", str(QWEN3_CONFIG), "--adapter", str(tmp_path), "--sequences", str(prefill),
            "--device", "cpu", "--out", str(tmp_path / "h.json")]
    # float32 is the default: the reference compare expects
    assert lp.main([*base, "--decode-sequences", str(decode), "--decode-out", str(tmp_path / "hd.json")]) == 0
    assert scored == [["seq-0000", "dec-0000"]]
    assert loaded == {"path": tmp_path / "Qwen3-8B", "dtype": "float32", "device": "cpu", "attn_implementation": "sdpa"}
    assert set(lp.read_scores(tmp_path / "h.json").logprobs["hf_lora"]) == {"seq-0000"}
    assert set(lp.read_scores(tmp_path / "hd.json").logprobs["hf_base"]) == {"dec-0000"}
    meta = json.loads((tmp_path / "h.json").read_text())["meta"]
    assert meta == {"adapter": str(tmp_path), "adapter_sha256": hashlib.sha256(b"factors").hexdigest(),
                    "adapter_config_sha256": hashlib.sha256(b"{}").hexdigest(),
                    "model_path": str(tmp_path / "Qwen3-8B"), "dtype": "float32", "autocast": False,
                    "float32_matmul_precision": "highest",
                    "sequences_sha256": {"seq-0000": lp.sequence_sha256(lp.read_sequences(prefill)[0])}}
    assert lp.main([*base, "--dtype", "bfloat16"]) == 0
    assert "float32_matmul_precision" not in json.loads((tmp_path / "h.json").read_text())["meta"]
    assert lp.main([arg for arg in base if arg != str(tmp_path) and arg != "--adapter"] + ["--dtype", "bfloat16"]) == 0
    assert json.loads((tmp_path / "h.json").read_text())["meta"]["adapter"] is None
    assert json.loads((tmp_path / "h.json").read_text())["meta"]["adapter_sha256"] is None
    with pytest.raises(lp.ParityError, match="share a sequence id"):
        lp.main([*base, "--decode-sequences", str(prefill), "--decode-out", str(tmp_path / "hd.json")])


def test_compare_refuses_scores_of_other_tokens_or_another_adapter(tmp_path, capsys):
    seqs, sequences, arms = tmp_path / "s.jsonl", _seq(100), _arms(100)
    lp.write_sequences(seqs, sequences)

    def write(name, keys, scored=sequences, adapter="a" * 64):
        lp.write_scores(tmp_path / name, lp.Scores({key: arms.logprobs[key] for key in keys}),
                        {"dtype": "float32", "adapter_sha256": adapter}, scored)

    write("h.json", lp.HF_ARMS)
    write("v.json", ("vllm_base", "vllm_lora"))
    gate = ["compare", "--sequences", str(seqs), "--hf", str(tmp_path / "h.json"), "--vllm", str(tmp_path / "v.json"),
            "--report", str(tmp_path / "r.json")]
    assert lp.main(gate) == 0
    recorded = json.loads((tmp_path / "v.json").read_text())["meta"]["sequences_sha256"]
    assert recorded == {"s": lp.sequence_sha256(sequences[0])}
    moved = [lp.FrozenSequence(seq.seq_id, seq.prompt_ids[:-1], seq.prompt_ids[-1:] + seq.completion_ids)
             for seq in sequences]
    write("v.json", ("vllm_base", "vllm_lora"), scored=moved)
    with pytest.raises(lp.ParityError, match="vllm_base's score file records other tokens, or none, for 1 of these "
                                             "sequences"):
        lp.main(gate)
    write("v.json", ("vllm_base", "vllm_lora"), adapter="b" * 64)
    with pytest.raises(lp.ParityError, match="every adapter arm scores the one adapter"):
        lp.main(gate)
    write("v.json", ("vllm_base", "vllm_lora"), adapter=None)
    with pytest.raises(lp.ParityError, match="every adapter arm scores the one adapter"):
        lp.main(gate)


def test_score_files_written_before_the_digests_compare_as_before_and_say_they_are_unbound(tmp_path, capsys):
    seqs, arms = tmp_path / "s.jsonl", _arms(100)
    lp.write_sequences(seqs, _seq(100))
    for name, keys, meta in (("h.json", lp.HF_ARMS, {"dtype": "float32", "adapter": "checkpoint-300"}),
                             ("v.json", ("vllm_base", "vllm_lora"), {"engine_dtype": "float32", "adapter": None})):
        (tmp_path / name).write_text(json.dumps({"meta": meta, "arms": {key: arms.logprobs[key] for key in keys}}))
    gate = ["compare", "--sequences", str(seqs), "--hf", str(tmp_path / "h.json"), "--vllm", str(tmp_path / "v.json"),
            "--report", str(tmp_path / "r.json")]
    assert lp.main(gate) == 0
    notes = json.loads((tmp_path / "r.json").read_text())["notes"]
    assert [note for note in notes if "not bound" in note] == [
        *(f"{arm}: its score file records no sequence digests, so its scores are not bound to these sequences"
          for arm in ("hf_base", "hf_lora", "vllm_base", "vllm_lora")),
        *(f"hf_lora, vllm_lora: no {key} recorded, so these scores are not bound to an adapter"
          for key in ("adapter_sha256", "adapter_config_sha256"))]
    assert json.loads(capsys.readouterr().out)["notes"] == notes
    lp.write_scores(tmp_path / "v.json", lp.Scores({key: arms.logprobs[key] for key in ("vllm_base", "vllm_lora")}),
                    {"adapter_sha256": None}, _seq(100))
    with pytest.raises(lp.ParityError, match=r"record adapter_sha256 \{'vllm_lora': None\}"):
        lp.main(gate)


def test_compare_and_the_decode_floor_called_directly_check_what_their_scores_recorded():
    sequences, arms = _seq(100), _arms(100)
    moved = [lp.FrozenSequence(seq.seq_id, seq.prompt_ids[:-1], seq.prompt_ids[-1:] + seq.completion_ids)
             for seq in sequences]

    def bound(scores, scored, adapters=None):
        meta = {"dtype": "float32", lp.SEQUENCES_KEY: {seq.seq_id: lp.sequence_sha256(seq) for seq in scored}}
        return lp.Scores(scores.logprobs, scores.argmax,
                         {arm: {**meta, **(adapters or {}).get(arm, ADAPTER)} for arm in scores.logprobs})

    report = lp.compare(bound(arms, sequences), sequences)
    assert report.passed and not [note for note in report.notes if "not bound" in note]
    with pytest.raises(lp.ParityError, match="records other tokens, or none, for 1 of these sequences"):
        lp.compare(bound(arms, moved), sequences)
    with pytest.raises(lp.ParityError, match="every adapter arm scores the one adapter"):
        lp.compare(bound(arms, sequences, {"vllm_lora": {"adapter_sha256": "b" * 64}}), sequences)
    assert any("not bound" in note for note in lp.compare(arms, sequences).notes)
    reference = lp.Scores({lp.BASE_DECODE_ARM: arms.logprobs["vllm_base"], lp.BASE_ARM: arms.logprobs["vllm_base"]})
    assert lp.decode_noise_floor(bound(reference, sequences), sequences).notes == ()
    assert lp.decode_noise_floor(reference, sequences).notes
    with pytest.raises(lp.ParityError, match="records other tokens"):
        lp.decode_noise_floor(bound(reference, moved), sequences)


def test_an_arm_s_adapter_is_its_weights_file_or_the_one_its_merge_records(tmp_path):
    adapter = tmp_path / "checkpoint-300"
    adapter.mkdir()
    (adapter / lp.ADAPTER_WEIGHTS_FNAME).write_bytes(b"factors")
    (adapter / lp.ADAPTER_CONFIG_FNAME).write_text('{"r": 32, "lora_alpha": 64}')
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "merge_manifest.json").write_text(json.dumps({"adapter": {"sha256": "a" * 64, "config_sha256": "c" * 64}}))
    assert lp.scored_adapter(adapter, tmp_path / "base") == {
        "adapter_sha256": hashlib.sha256(b"factors").hexdigest(),
        "adapter_config_sha256": hashlib.sha256(b'{"r": 32, "lora_alpha": 64}').hexdigest()}
    assert lp.scored_adapter(None, merged) == {"adapter_sha256": "a" * 64, "adapter_config_sha256": "c" * 64}
    assert lp.scored_adapter(None, tmp_path / "base") == {"adapter_sha256": None, "adapter_config_sha256": None}


def test_adapters_of_one_weights_file_and_another_configuration_are_two_adapters(tmp_path):
    """Equal factors at another lora_alpha scale the delta otherwise."""
    sequences, arms = _seq(100), _arms(100)
    meta = {"dtype": "float32", lp.SEQUENCES_KEY: {seq.seq_id: lp.sequence_sha256(seq) for seq in sequences},
            **ADAPTER}
    configs = {"hf_lora": "c" * 64, "vllm_lora": "d" * 64}
    scores = lp.Scores(arms.logprobs, arms.argmax,
                       {arm: {**meta, "adapter_config_sha256": configs.get(arm)} for arm in arms.logprobs})
    with pytest.raises(lp.ParityError, match=r"record adapter_config_sha256 \{'hf_lora': 'c+', 'vllm_lora': 'd+'\}"):
        lp.compare(scores, sequences)


def test_cli_compare_takes_every_threshold(tmp_path, capsys):
    seqs = tmp_path / "s.jsonl"
    lp.write_sequences(seqs, _seq(100))
    arms = _arms(100)
    lp.write_scores(tmp_path / "h.json", lp.Scores({k: arms.logprobs[k] for k in lp.HF_ARMS}),
                    {"dtype": "bfloat16", **ADAPTER}, _seq(100))
    lp.write_scores(tmp_path / "v.json", lp.Scores({k: arms.logprobs[k] for k in ("vllm_base", "vllm_lora")}), ADAPTER,
                    _seq(100))
    report = tmp_path / "r.json"
    gate = ["compare", "--sequences", str(seqs), "--hf", str(tmp_path / "h.json"), "--vllm", str(tmp_path / "v.json"),
            "--report", str(report)]
    # HF arms scored in bf16 are compared only when allowed
    with pytest.raises(lp.ParityError, match="not float32"):
        lp.main(gate)
    gate.append("--allow-hf-dtype")
    assert lp.main(gate) == 0 and json.loads(report.read_text())["passed"] is True
    printed = json.loads(capsys.readouterr().out)
    assert printed["inputs"]["hf_dtype"] == {"hf_base": "bfloat16", "hf_lora": "bfloat16"}
    assert lp.main([*gate, "--bucket-edges", "50", "--min-bucket-tokens", "10", "--effect-ratio", "0.8", "1.2",
                    "--paired-delta", "0.2", "--min-effect-noise-ratio", "8", "--rho-range", "0.99", "1.01",
                    "--rho-precision", "0.5", "--rho-per-bucket", "--rho-resamples", "50", "--rho-seed", "3",
                    "--noise-factor", "2", "--abs-tolerance", "0.01", "--min-effect", "0.001",
                    "--greedy-agreement", "0.9"]) == 0
    thresholds = json.loads(report.read_text())["thresholds"]
    assert thresholds["bucket_edges"] == [50] and thresholds["rho_range"] == [0.99, 1.01]
    assert thresholds["min_effect_noise_ratio"] == 8 and thresholds["rho_per_bucket"] is True
    assert thresholds["effect_ratio"] == [0.8, 1.2] and thresholds["require_hf_float32"] is False
    lp.write_scores(tmp_path / "v.json", lp.Scores({k: _arms(100, vllm_effect=0.0).logprobs[k] for k in ("vllm_base", "vllm_lora")}),
                    ADAPTER, _seq(100))
    assert lp.main(gate) == 1
    with pytest.raises(lp.ParityError, match="go together"):
        lp.main([*gate, "--decode-reference", str(tmp_path / "v.json")])


def test_cli_compare_exits_3_when_a_requested_rho_range_could_not_be_applied(tmp_path, capsys):
    seqs = tmp_path / "s.jsonl"
    sequences = _delta_seqs(n_seqs=20)
    lp.write_sequences(seqs, sequences)
    arms = _delta_arms(arm="vllm_merged", n_seqs=20)
    lp.write_scores(tmp_path / "h.json", lp.Scores({k: arms.logprobs[k] for k in lp.HF_ARMS}),
                    {"dtype": "float32", **ADAPTER}, sequences)
    lp.write_scores(tmp_path / "v.json", lp.Scores({k: arms.logprobs[k] for k in ("vllm_base", "vllm_merged")}), ADAPTER,
                    sequences)
    gate = ["compare", "--sequences", str(seqs), "--hf", str(tmp_path / "h.json"), "--vllm", str(tmp_path / "v.json"),
            "--report", str(tmp_path / "r.json"), "--bucket-edges", "", "--paired-delta", "1.0", "--rho-range", "0.99", "1.01"]
    assert lp.main([*gate, "--rho-precision", "0.01"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rho_gated"] == {"all": True}
    assert set(printed["effect"]) == {"noise", "decode_noise", "serving_limit", "effect_hf", "effect_noise_ratio",
                                      "effect_window"}
    report = json.loads((tmp_path / "r.json").read_text())["buckets"]["all"]
    assert printed["effect"]["serving_limit"] == report["serving_limit"] and len(printed["effect"]["effect_window"]) == 2
    assert lp.main([*gate, "--rho-precision", "1e-9"]) == 0
    capsys.readouterr()
    straddle = [*gate, "--rho-precision", "1e-9", "--rho-range", "1.0", "1.01"]
    assert lp.main(straddle) == lp.RHO_NOT_GATED_EXIT == 3
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"] is True and printed["rho_gated"] == {"all": False}
    assert json.loads((tmp_path / "r.json").read_text())["rho_gated"] == {"all": False}
    assert lp.main([*straddle, "--allow-imprecise-rho"]) == 0
    capsys.readouterr()
    weak = [*gate, "--rho-precision", "0.01", "--min-effect", "1e9", "--allow-weak-effect"]
    assert lp.main(weak) == lp.RHO_NOT_GATED_EXIT
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"] is True and printed["inconclusive"] and printed["rho_gated"] == {"all": False}
    assert lp.main([*weak, "--allow-imprecise-rho"]) == 0


def _paired_arms(n_seqs=40, per_seq=100, *, seed=0, served_noise=0.0, hf_noise=0.0, served_scale=1.0,
                 merged_scale=None, merged_noise=0.0, autocast_noise=None, served_arm="vllm_merged", effect_scale=1.0,
                 per_sequence_scale=None):
    """Arms whose trained effect d is random per token: hl = hb + d (+ noise in dH), served = vb + scale * d."""
    rng = random.Random(seed)
    names = [f"s{i:02d}" for i in range(n_seqs)]
    arms = {"hf_base": {}, "hf_lora": {}, "vllm_base": {}, served_arm: {}}
    if merged_scale is not None:
        arms[lp.HF_MERGED_ARM] = {}
    if autocast_noise is not None:
        arms[lp.HF_LORA_AC_ARM] = {}
    for name in names:
        for arm in arms:
            arms[arm][name] = []
        factor = 1.0 if per_sequence_scale is None else rng.uniform(*per_sequence_scale)
        for _ in range(per_seq):
            hb = -rng.uniform(0.1, 3.0)
            d = rng.choice((-1.0, 1.0)) * rng.uniform(0.2, 0.6) * effect_scale
            vb = hb + rng.gauss(0.0, 0.002)
            hl = hb + d + rng.gauss(0.0, hf_noise)
            arms["hf_base"][name].append(hb)
            arms["hf_lora"][name].append(hl)
            arms["vllm_base"][name].append(vb)
            arms[served_arm][name].append(vb + factor * served_scale * d + rng.gauss(0.0, served_noise))
            if merged_scale is not None:
                arms[lp.HF_MERGED_ARM][name].append(
                    hb + factor * merged_scale * d + (rng.gauss(0.0, merged_noise) if merged_noise else 0.0))
            if autocast_noise is not None:
                arms[lp.HF_LORA_AC_ARM][name].append(hl + rng.gauss(0.0, autocast_noise))
    seqs = [lp.FrozenSequence(name, tuple(range(per_seq // 2)), tuple(range(per_seq + 1 - per_seq // 2))) for name in names]
    return _scores(arms), seqs


LOOSE = dict(bucket_edges=(), paired_delta=10.0, noise_factor=1e6, greedy_agreement=None)


def test_rho_is_zero_when_the_served_arm_carries_no_effect():
    assert lp.compare(*_paired_arms(served_scale=0.0), lp.ParityThresholds(**LOOSE)).buckets["all"].rho == 0.0
    stats = lp.compare(*_paired_arms(served_scale=0.0, served_noise=0.05), lp.ParityThresholds(**LOOSE)).buckets["all"]
    assert abs(stats.rho) < 0.02 and stats.rho_ci[0] < 0.0 < stats.rho_ci[1]


def test_rho_is_unbiased_under_independent_noise_in_the_served_delta():
    rhos = []
    for seed in range(8):
        scores, seqs = _paired_arms(seed=seed, served_noise=0.3)
        stats = lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE)).buckets["all"]
        assert stats.rho_ci[0] < 1.0 < stats.rho_ci[1]
        rhos.append(stats.rho)
    clean = lp.compare(*_paired_arms(), lp.ParityThresholds(**LOOSE)).buckets["all"]
    assert clean.rho_halfwidth < stats.rho_halfwidth
    assert sum(rhos) / len(rhos) == pytest.approx(1.0, abs=0.01)


def test_rho_is_attenuated_by_noise_in_the_hf_delta():
    """Noise in dH (the reference's own rounding) biases rho towards 0 by var(d) / (var(d) + sigma^2)."""
    sigma = 0.3
    energy = (0.6 ** 3 - 0.2 ** 3) / (3 * 0.4)  # E[d^2] for |d| uniform on [0.2, 0.6]
    scores, seqs = _paired_arms(n_seqs=60, hf_noise=sigma)
    rho = lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE)).buckets["all"].rho
    assert rho == pytest.approx(energy / (energy + sigma ** 2), abs=0.03)
    assert rho < 0.75


def test_rho_hf_tests_the_merge_alone_and_the_serving_gate_reads_hf_merged():
    thresholds = lp.ParityThresholds(bucket_edges=(), greedy_agreement=None, paired_delta=1.0)
    scores, seqs = _paired_arms(merged_scale=1.0)
    report = lp.compare(scores, seqs, thresholds)
    stats = report.buckets["all"]
    assert report.passed, report.failures
    assert report.serving_reference == lp.HF_MERGED_ARM
    assert stats.rho_hf == pytest.approx(1.0) and stats.rho_hf_halfwidth < 1e-9
    assert stats.merged_gap == pytest.approx(stats.noise_base, rel=0.2)
    half = lp.compare(*_paired_arms(merged_scale=0.5), thresholds).buckets["all"]
    assert half.rho_hf == pytest.approx(0.5)
    rounded, seqs = _paired_arms(merged_scale=0.8, served_scale=0.8)
    loose = lp.ParityThresholds(bucket_edges=(), greedy_agreement=None, paired_delta=10.0)
    assert lp.compare(rounded, seqs, loose).passed
    without = lp.Scores({k: v for k, v in rounded.logprobs.items() if k != lp.HF_MERGED_ARM})
    failed = lp.compare(without, seqs, loose)
    assert failed.serving_reference == "hf_lora"
    assert any("|hf_lora - vllm_merged|" in f for f in failed.failures)
    drifted, seqs = _paired_arms(merged_scale=1.0, served_scale=0.8)
    assert any("|hf_merged - vllm_merged|" in f for f in lp.compare(drifted, seqs, loose).failures)


def test_the_autocast_gap_is_reported_and_gates_nothing():
    scores, seqs = _paired_arms(autocast_noise=0.05)
    report = lp.compare(scores, seqs, lp.ParityThresholds(bucket_edges=(), greedy_agreement=None, paired_delta=1.0))
    stats = report.buckets["all"]
    assert report.passed, report.failures
    assert stats.autocast_gap == pytest.approx(0.05 * math.sqrt(2 / math.pi), rel=0.1)
    assert stats.autocast_gap_p99 > stats.autocast_gap
    assert math.isnan(stats.rho_hf) and math.isnan(stats.merged_gap)


def test_a_planted_nan_in_any_arm_fails_the_finite_gate():
    for arm in ("hf_merged", "hf_lora_ac", "vllm_merged", "hf_base"):
        scores, seqs = _paired_arms(merged_scale=1.0, autocast_noise=0.01)
        scores.logprobs[arm]["s03"][7] = math.nan
        report = lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE))
        assert not report.passed and report.nonfinite[arm] == 1
        assert any("non-finite logprobs" in f for f in report.failures)
        assert report.buckets["all"].tokens == 40 * 100 - 1  # the other positions are still compared


def test_hf_merged_is_the_merged_modes_reference_scored_as_the_served_weights():
    scores, seqs = _paired_arms(merged_scale=1.0, served_arm="vllm_lora")
    with pytest.raises(lp.ParityError, match="merged mode's reference"):
        lp.compare(scores, seqs)
    scores, seqs = _paired_arms(merged_scale=1.0)
    served = {"engine_dtype": "float32", "model_path": "/m/merged"}
    scores.meta = {"vllm_merged": served, lp.HF_MERGED_ARM: {"dtype": "bfloat16", "model_path": "/m/merged"}}
    with pytest.raises(lp.ParityError, match="dtype they are served in"):
        lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE))
    scores.meta[lp.HF_MERGED_ARM] = {"dtype": "float32", "model_path": "/m/other"}
    with pytest.raises(lp.ParityError, match="one set of weights on two engines"):
        lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE))
    scores.meta[lp.HF_MERGED_ARM] = {"dtype": "float32", "model_path": "/m/merged"}
    report = lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE))
    assert report.inputs["hf_merged"] == {"dtype": "float32", "model_path": "/m/merged"}
    scores.meta[lp.HF_MERGED_ARM] = {"dtype": "float32"}
    assert lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE)).inputs["hf_merged"]["model_path"] is None
    # hf_lora_ac is held to the reference's float32 like the other HF arms (its fp32 model, under autocast)
    scores, seqs = _paired_arms(autocast_noise=0.01)
    scores.meta = {lp.HF_LORA_AC_ARM: {"dtype": "bfloat16"}}
    with pytest.raises(lp.ParityError, match="not float32"):
        lp.compare(scores, seqs, lp.ParityThresholds(**LOOSE))


def test_the_presets_are_the_designs_gates_and_the_fallback_threshold_only_reports():
    assert preset_thresholds("b").rho_range == (0.99, math.inf) and preset_thresholds("b").rho_precision == 0.01
    assert preset_thresholds("b").rho_hf_range == (0.99, 1.01)
    # rho_hf is gated in fp32 only: under the fallback it is the bf16 rounding's loss; M0's rho is two-sided
    assert {name for name in lp.PARITY_PRESETS if preset_thresholds(name).rho_hf_range} == {"b"}
    m0 = preset_thresholds("m0")
    assert (m0.rho_range, m0.rho_per_bucket) == ((0.99, 1.01), True)
    assert preset_thresholds("m0_fallback").rho_range == (0.95, 1.05)
    assert preset_thresholds("b", rho_precision=None).rho_precision is None
    # the fallback's 0.90 is reported, never gated
    scores, seqs = _paired_arms(served_scale=0.8)
    report = lp.compare(scores, seqs, preset_thresholds("b_fallback", **LOOSE))
    assert report.passed and report.rho_threshold_met == {"all": False} and report.rho_gated == {}
    passing = lp.compare(*_paired_arms(), preset_thresholds("b_fallback", **LOOSE))
    assert passing.rho_threshold_met == {"all": True}
    over, seqs = _paired_arms(served_scale=1.05)
    failed = lp.compare(over, seqs, preset_thresholds("m0", bucket_edges=(40,), paired_delta=1.0, greedy_agreement=None))
    assert {f.split(":")[0] for f in failed.failures if "retention rho" in f} == {"all", "[0,40)", "[40,inf)"}


def test_preset_b_leaves_a_weak_effect_to_the_other_gates_and_still_gates_a_precise_rho():
    faithful, seqs = _paired_arms(effect_scale=0.02, served_noise=0.001, merged_scale=1.0)
    b = preset_thresholds("b", bucket_edges=())
    report = lp.compare(faithful, seqs, b)
    stats = report.buckets["all"]
    assert stats.effect_noise_ratio < b.min_effect_noise_ratio  # too weak for the paired gate: 5x the noise
    assert report.passed and report.inconclusive and report.rho_gated == {"all": True}, report.failures
    assert report.rho_hf_gated == {"all": True} and stats.rho_hf == pytest.approx(1.0)
    assert any("too weak to test" in f for f in lp.compare(faithful, seqs, preset_thresholds("m0", bucket_edges=())).failures)
    # rho 0.8, measured precisely at 6x the noise: a retention failure, weak effect or not
    lost, seqs = _paired_arms(effect_scale=0.024, served_noise=0.001, served_scale=0.8, merged_scale=1.0)
    report = lp.compare(lost, seqs, b)
    assert report.buckets["all"].rho_halfwidth <= 0.01 and report.rho_gated == {"all": True}
    assert any("retention rho 0.8" in f for f in report.failures) and report.inconclusive
    report = lp.compare(lost, seqs, preset_thresholds("b", bucket_edges=(), min_effect=1.0))
    assert report.rho_gated == report.rho_hf_gated == {"all": False}
    assert any("rho not gated" in n for n in report.notes) and any("rho_hf not gated" in n for n in report.notes)
    assert not any("retention rho" in f for f in report.failures) and any("exceeds" in f for f in report.failures)


def test_preset_b_fails_a_merge_that_over_applies_the_delta_at_a_weak_effect():
    b = preset_thresholds("b", bucket_edges=())
    for effect_scale in (0.02, 0.05):  # about 5x and 12x the noise, under the paired gate's 13.3x
        for scale in (1.5, 3.0, 0.5):
            scores, seqs = _paired_arms(effect_scale=effect_scale, served_noise=0.001, served_scale=scale,
                                        merged_scale=scale)
            report = lp.compare(scores, seqs, b)
            assert report.inconclusive and not report.passed and report.rho_hf_gated == {"all": True}
            assert report.buckets["all"].rho_hf == pytest.approx(scale)
            assert any(f.startswith(f"all: rho_hf {scale:.4f}") and "the merge mis-scales" in f for f in report.failures)
            assert not any("exceeds" in f for f in report.failures)
            without = lp.compare(scores, seqs, preset_thresholds("b", bucket_edges=(), rho_hf_range=None))
            assert without.passed == (scale > 1)
        correct, seqs = _paired_arms(effect_scale=effect_scale, served_noise=0.001, merged_scale=1.0)
        assert lp.compare(correct, seqs, b).passed
    missing = lp.Scores({k: v for k, v in scores.logprobs.items() if k != lp.HF_MERGED_ARM})
    with pytest.raises(lp.ParityError, match="scores have no hf_merged"):
        lp.compare(missing, seqs, b)
    noisy, seqs = _paired_arms(effect_scale=0.02, served_noise=0.001, merged_scale=1.0, merged_noise=0.05)
    loose = dict(noise_factor=1e6, greedy_agreement=None)
    report = lp.compare(noisy, seqs, preset_thresholds("b", bucket_edges=(40,), rho_per_bucket=True, **loose))
    assert report.rho_hf_gated == {"all": True, "[0,40)": True, "[40,inf)": True} and not report.passed
    assert report.rho_gated == {"all": True, "[0,40)": True, "[40,inf)": True}
    assert {f.split(":")[0] for f in report.failures} == {"all", "[0,40)", "[40,inf)"}
    assert all(" rho_hf " in f and "wider than 0.01" in f for f in report.failures), report.failures
    assert report.buckets["all"].rho_hf_halfwidth > 0.01
    alone = lp.compare(*_paired_arms(merged_scale=1.5, served_scale=1.5), lp.ParityThresholds(
        bucket_edges=(), rho_hf_range=(0.99, 1.01), paired_delta=10.0, greedy_agreement=None))
    assert alone.rho_gated == {} and alone.rho_hf_gated == {"all": True}
    assert [f.split(" ")[1] for f in alone.failures] == ["rho_hf"]
    with pytest.raises(lp.ParityError, match="rho_hf_range"):
        lp.ParityThresholds(rho_hf_range=(1.01, 0.99))


def test_preset_b_fails_a_merge_whose_mis_scale_varies_by_prompt_at_a_weak_effect():
    b = preset_thresholds("b", bucket_edges=())
    for shares, rho_decided in (((1.0, 1.4), "passes"), ((0.6, 1.0), "fails")):
        scores, seqs = _paired_arms(n_seqs=64, effect_scale=0.02, served_noise=0.001, merged_scale=1.0,
                                    per_sequence_scale=shares)
        report = lp.compare(scores, seqs, b)
        stats = report.buckets["all"]
        assert stats.effect_noise_ratio < b.min_effect_noise_ratio and report.inconclusive
        assert not report.passed and report.rho_hf_gated == report.rho_gated == {"all": True}
        assert stats.rho_hf_halfwidth > 0.01 and stats.rho_halfwidth > 0.01
        rho_hf_failures = [f for f in report.failures if f.startswith("all: rho_hf ")]
        assert len(rho_hf_failures) == 1 and "wider than 0.01" in rho_hf_failures[0]
        assert not any("exceeds" in f for f in report.failures)
        rho_failures = [f for f in report.failures if "retention rho" in f]
        assert (stats.rho_ci[1] < 0.99 and len(rho_failures) == 1) if rho_decided == "fails" else (
            stats.rho_ci[0] > 0.99 and not rho_failures)
    correct, seqs = _paired_arms(n_seqs=64, effect_scale=0.02, served_noise=0.001, merged_scale=1.0)
    report = lp.compare(correct, seqs, b)
    assert report.passed and report.buckets["all"].rho_hf_halfwidth < 1e-9, report.failures


def test_preset_b_fallback_does_not_fail_the_expected_bf16_rounding():
    """Under the fallback the bf16 merge keeps only part of the delta (retention 0.64-0.89 early on)."""
    rounded, seqs = _paired_arms(served_scale=0.8, merged_scale=0.8)
    strict = lp.compare(rounded, seqs, lp.ParityThresholds(bucket_edges=(), greedy_agreement=None))
    assert any("paired delta" in f for f in strict.failures)
    report = lp.compare(rounded, seqs, preset_thresholds("b_fallback", bucket_edges=(), greedy_agreement=None))
    assert report.passed, report.failures
    assert report.rho_threshold_met == {"all": False} and report.serving_reference == lp.HF_MERGED_ARM
    broken, seqs = _paired_arms(served_scale=0.5, merged_scale=0.8)
    assert any("exceeds" in f for f in lp.compare(broken, seqs, preset_thresholds(
        "b_fallback", bucket_edges=(), greedy_agreement=None)).failures)


def test_every_presets_thresholds_are_strict_json():
    for name in lp.PARITY_PRESETS:
        recorded = lp.thresholds_to_json(preset_thresholds(name))
        assert json.loads(json.dumps(recorded, allow_nan=False)) == recorded
    assert lp.thresholds_to_json(preset_thresholds("b"))["rho_range"] == [0.99, "inf"]
    assert lp.thresholds_to_json(preset_thresholds("b_fallback"))["paired_delta"] == "inf"


def test_cli_compare_applies_the_whole_preset_and_an_explicit_flag_over_it(tmp_path, capsys):
    def write(scores, sequences):
        lp.write_sequences(seqs, sequences)
        for name, arms, meta in (("h", lp.HF_ARMS, {"dtype": "float32"}), ("m", (lp.HF_MERGED_ARM,), {"dtype": "float32"}),
                                 ("v", ("vllm_base", "vllm_merged"), {"engine_dtype": "float32"})):
            lp.write_scores(tmp_path / f"{name}.json", lp.Scores({k: scores.logprobs[k] for k in arms}),
                            {**meta, **ADAPTER}, sequences)

    seqs = tmp_path / "s.jsonl"
    write(*_paired_arms(effect_scale=0.02, served_noise=0.001, merged_scale=1.0))
    gate = ["compare", "--preset", "b", "--sequences", str(seqs), "--hf", str(tmp_path / "h.json"),
            "--hf", str(tmp_path / "m.json"), "--vllm", str(tmp_path / "v.json"), "--report", str(tmp_path / "r.json"),
            "--bucket-edges", ""]
    assert lp.main(gate) == 0
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["thresholds"]["require_effect"] is False and report["thresholds"]["rho_range"] == [0.99, "inf"]
    assert report["thresholds"]["rho_hf_range"] == [0.99, 1.01] and report["rho_hf_gated"] == {"all": True}
    assert report["gates"] == {"preset": "b", "thresholds": report["thresholds"]} and report["inconclusive"]
    assert json.loads(capsys.readouterr().out)["rho_hf_gated"] == {"all": True}
    assert lp.main([*gate, "--require-effect"]) == 1
    assert json.loads((tmp_path / "r.json").read_text())["thresholds"]["require_effect"] is True
    capsys.readouterr()
    assert lp.main([*gate, "--min-effect", "1e9"]) == lp.RHO_NOT_GATED_EXIT
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"] is True and printed["rho_gated"] == printed["rho_hf_gated"] == {"all": False}
    assert lp.main([*gate, "--min-effect", "1e9", "--allow-imprecise-rho"]) == lp.RHO_NOT_GATED_EXIT
    capsys.readouterr()
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["gates"] == {"preset": "b", "thresholds": report["thresholds"], "allow_imprecise_rho": True}
    write(*_paired_arms(effect_scale=0.02, served_noise=0.001, merged_scale=1.0, merged_noise=0.05))
    assert lp.main([*gate, "--noise-factor", "1e6"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"] is False and printed["rho_gated"] == printed["rho_hf_gated"] == {"all": True}
    assert [f.split(" (")[0] for f in printed["failures"]] == [f"all: rho_hf {printed['rho_hf']:.4f}"]
    assert lp.main([*gate, "--noise-factor", "1e6", "--allow-imprecise-rho"]) == 1
    assert lp.main([*gate, "--noise-factor", "1e6", "--rho-hf-range", "0.5", "1.5", "--rho-precision", "0.5"]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["thresholds"]["rho_hf_range"] == [0.5, 1.5]


def test_the_decode_arms_scores_line_up_token_by_token_with_teacher_forcing(fake_vllm):
    prompts = [{"system_prompt": f"S{i}", "user_prompt": ""} for i in range(4)]
    seqs, served = lp.sample_decode(_DecodingLLM(), _Tokenizer(), prompts, lora_request="REQ", max_tokens=2,
                                    seed=0, batch_size=2)

    def teacher_forced(seq, shift):
        """What the fake engine's function gives at each position of token_ids[1:]."""
        prompt = [-0.1 * t for t in seq.prompt_ids[1:]]
        return prompt + [-0.01 * t + shift for t in seq.completion_ids]

    arms = {
        "hf_base": {seq.seq_id: teacher_forced(seq, 0.0) for seq in seqs},
        "hf_lora": {seq.seq_id: teacher_forced(seq, 0.5) for seq in seqs},
        "vllm_base": {seq.seq_id: teacher_forced(seq, 0.0) for seq in seqs},
        "vllm_lora": served.logprobs["_"],
    }
    thresholds = lp.ParityThresholds(bucket_edges=(), greedy_agreement=None, min_effect_noise_ratio=None,
                                     abs_tolerance=1e-9, min_bucket_tokens=1)
    aligned = lp.compare(_scores(arms), seqs, thresholds, segment="completion")
    assert aligned.passed, aligned.failures
    assert aligned.buckets["all"].lora_gap == pytest.approx(0.0, abs=1e-12)
    shifted = {seq_id: values[:-2] + values[-1:] + values[-2:-1] for seq_id, values in arms["vllm_lora"].items()}
    report = lp.compare(_scores({**arms, "vllm_lora": shifted}), seqs, thresholds, segment="completion")
    assert not report.passed and any("|hf_lora - vllm_lora|" in f for f in report.failures)


def test_the_autocast_arm_is_the_peft_model_under_bf16_and_a_plain_model_takes_its_arm(llama_dir, tmp_path):
    adapter = lp.make_random_adapter(llama_dir, tmp_path / "a", rank=4, alpha=8.0, seed=1, b_std=0.05)
    assert adapter.ok
    seqs = [lp.FrozenSequence("a", (1, 2, 3), (4, 5, 6))]
    model = lp.load_hf_model(llama_dir, dtype="float32", device="cpu")
    scores = lp.score_hf(model, tmp_path / "a", seqs, device="cpu", autocast_arm=True)
    assert set(scores.logprobs) == {"hf_lora", "hf_base", lp.HF_LORA_AC_ARM}
    lora, autocast = scores.logprobs["hf_lora"]["a"], scores.logprobs[lp.HF_LORA_AC_ARM]["a"]
    gaps = [abs(a - b) for a, b in zip(lora, autocast)]
    assert 0 < max(gaps) < 0.1  # bf16 autocast moves the logprobs a little
    plain = lp.load_hf_model(llama_dir, dtype="float32", device="cpu")
    merged = lp.score_hf(plain, None, seqs, device="cpu", arm=lp.HF_MERGED_ARM)
    assert set(merged.logprobs) == {lp.HF_MERGED_ARM}
    with pytest.raises(lp.ParityError, match="plain model is scored as one of"):
        lp.score_hf(plain, None, seqs, device="cpu", arm="hf_lora")
    with pytest.raises(lp.ParityError, match="without an autocast arm"):
        lp.score_hf(plain, None, seqs, device="cpu", autocast_arm=True)


def test_cli_hf_scores_a_merged_directory_and_the_autocast_arm(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    (tmp_path / lp.ADAPTER_WEIGHTS_FNAME).write_bytes(b"factors")  # the adapter the arms record by digest
    (tmp_path / lp.ADAPTER_CONFIG_FNAME).write_text("{}")
    seqs = tmp_path / "s.jsonl"
    lp.write_sequences(seqs, [lp.FrozenSequence("seq-0000", (1, 2), (3,))])
    calls = []
    monkeypatch.setattr(lp, "load_hf_model", lambda path, **kw: calls.append(path) or "MODEL")
    monkeypatch.setattr(lp, "exact_float32", lambda: {})
    monkeypatch.setattr(lp, "wrap_adapter", lambda model, adapter: "PEFT")

    def _score(model, adapter, sequences, device, arm="hf_base", autocast_arm=False, wrapped=None):
        calls.append((arm, autocast_arm))
        rows = {seq.seq_id: [0.0] * (len(seq.token_ids) - 1) for seq in sequences}
        return lp.Scores({arm if adapter is None else "hf_lora": rows})

    monkeypatch.setattr(lp, "score_hf", _score)
    merged = tmp_path / "merged"
    base = ["hf", "--model-config", str(QWEN3_FP32_CONFIG), "--sequences", str(seqs), "--device", "cpu",
            "--out", str(tmp_path / "h.json")]
    assert lp.main([*base, "--model-path", str(merged), "--arm", "hf_merged"]) == 0
    assert calls == [merged, ("hf_merged", False)]
    meta = json.loads((tmp_path / "h.json").read_text())["meta"]
    assert meta["model_path"] == str(merged) and "autocast_arms" not in meta
    assert lp.main([*base, "--adapter", str(tmp_path), "--autocast-arm"]) == 0
    assert json.loads((tmp_path / "h.json").read_text())["meta"]["autocast_arms"] == {"hf_lora_ac": "bfloat16"}
    with pytest.raises(lp.ParityError, match="needs --adapter"):
        lp.main([*base, "--autocast-arm"])
    with pytest.raises(lp.ParityError, match="plain model's"):
        lp.main([*base, "--adapter", str(tmp_path), "--arm", "hf_merged"])


def test_eos_ids_come_from_the_generation_config_else_the_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 8]}))
    assert lp.eos_token_ids(tmp_path) == (7, 8)
    (tmp_path / "generation_config.json").write_text(json.dumps({"top_k": 20}))
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 5}))
    assert lp.eos_token_ids(tmp_path) == (5,)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(lp.ParityError, match="names no eos_token_id"):
        lp.eos_token_ids(tmp_path)


class _IdTokenizer(_Tokenizer):
    """Renders as the fake chat template, tokenizes one id per character, decodes ids as letters."""

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) % 50 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is False
        return "".join(chr(97 + i % 26) for i in ids)


def test_both_engines_generate_greedily_from_one_set_of_prompt_ids(fake_vllm, llama_dir):
    ids = lp.prompt_token_ids(_IdTokenizer(), [{"system_prompt": "S", "user_prompt": ""}])
    assert ids == [[ord(c) % 50 for c in "system:S|user:"]]

    class _GreedyLLM:
        def generate(self, prompts, sampling_params=None, use_tqdm=True):
            self.params = sampling_params.kwargs
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[3, 4, 2], finish_reason="stop")])
                    for _ in prompts]

    llm = _GreedyLLM()
    assert lp.vllm_greedy(llm, [[1, 2], [3]], max_tokens=5) == [([3, 4, 2], "stop"), ([3, 4, 2], "stop")]
    assert llm.params == {"temperature": 0.0, "max_tokens": 5, "skip_special_tokens": False}

    class _Short(_GreedyLLM):
        def generate(self, prompts, **kwargs):
            return super().generate(prompts, **kwargs)[:-1]

    with pytest.raises(lp.ParityError, match="1 outputs for 2 prompts"):
        lp.vllm_greedy(_Short(), [[1], [2]], max_tokens=5)
    model = lp.load_hf_model(llama_dir, dtype="float32", device="cpu")
    [(first, reason)] = lp.hf_greedy(model, [[1, 2, 3]], max_tokens=3, eos_ids=(47,), device="cpu")
    assert len(first) == 3 and 47 not in first and reason == "length"
    [(stopped, reason)] = lp.hf_greedy(model, [[1, 2, 3]], max_tokens=3, eos_ids=(first[0],), device="cpu")
    assert (stopped, reason) == ([first[0]], "stop")
    rows = lp.greedy_records([[1, 2]], [([3, 4, 2, 2], "stop")], _IdTokenizer(), eos_ids=(2,))
    assert rows[0]["completion_ids"] == [3, 4] and rows[0]["text"] == "de" and rows[0]["id"] == "greedy-0000"
    assert rows[0]["prompt_sha256"] == lp.greedy_records([[1, 2]], [([], "length")], _IdTokenizer(), (2,))[0]["prompt_sha256"]


def _rules(tmp_path):
    path = tmp_path / "rules.jsonl"
    path.write_text("".join(json.dumps({"id": f"Rule {i}", "rule": f"r{i}"}) + "\n" for i in (1, 2)))
    return path


def _greedy_row(index, text, finish="stop", ids=None):
    return {"id": f"greedy-{index:04d}", "prompt_sha256": f"p{index}", "completion_ids": ids or [index],
            "text": text, "finish_reason": finish}


def test_the_greedy_verdict_agreement_parses_each_output_as_the_campaign_does(tmp_path):
    schema = lp.verdict_schema(_rules(tmp_path), prompt_style="forced-grpo", forced_json_requirement="label_only",
                               strict_think=True)
    yes = '<think>t</think>{"full_set_judgment": "yes"}'
    no = '<think>t</think>{"full_set_judgment": "no"}'
    assert lp.greedy_outcome(_greedy_row(0, yes), schema) == "yes"
    assert lp.greedy_outcome(_greedy_row(0, "<think>never closed"), schema) == "invalid:unclosed_think"
    assert lp.greedy_outcome(_greedy_row(0, "<think>cut", "length"), schema) == "invalid:truncated"
    assert lp.greedy_outcome(_greedy_row(0, "<think>t</think>"), schema) == "invalid:empty_response"
    first = [_greedy_row(i, yes, ids=[1]) for i in range(38)] + [_greedy_row(38, yes), _greedy_row(39, "<think>x", "length")]
    second = [_greedy_row(i, yes, ids=[1]) for i in range(38)] + [_greedy_row(38, no), _greedy_row(39, "<think>y", "length")]
    result = lp.greedy_agreement(first, second, schema)
    assert result["agreement"] == pytest.approx(39 / 40) and result["passed"]  # two truncations agree
    assert result["both_valid"] == 39 and result["valid_agreement"] == pytest.approx(38 / 39)
    assert result["both_invalid_share"] == pytest.approx(1 / 40)
    assert result["identical_completions"] == pytest.approx(1.0)
    assert result["disagreements"] == {"greedy-0038": ["yes", "no"]}
    assert not lp.greedy_agreement(first, second, schema, minimum=0.98)["passed"]
    cut = [_greedy_row(i, f"<think>{i}", "length") for i in range(70)]
    first = cut + [_greedy_row(70 + i, yes) for i in range(30)]
    second = [_greedy_row(i, f"<think>other {i}", "length") for i in range(70)]
    second += [_greedy_row(70 + i, yes if i < 25 else no) for i in range(30)]
    diluted = lp.greedy_agreement(first, second, schema)
    assert diluted["agreement"] == pytest.approx(0.95) and diluted["valid_agreement"] == pytest.approx(25 / 30)
    assert diluted["passed"] is False and diluted["both_invalid_share"] == pytest.approx(0.7)
    assert lp.greedy_agreement(cut, cut, schema)["passed"] is False
    with pytest.raises(lp.ParityError, match="same prompts"):
        lp.greedy_agreement(first, second[:-1], schema)
    with pytest.raises(lp.ParityError, match="same prompts"):
        lp.greedy_agreement([], [], schema)


def test_cli_greedy_agreement_exits_1_under_the_minimum(tmp_path, capsys):
    rules = _rules(tmp_path)
    yes, no = '<think>t</think>{"full_set_judgment": "yes"}', '<think>t</think>{"full_set_judgment": "no"}'
    write_jsonl_records(tmp_path / "a.jsonl", [_greedy_row(0, yes), _greedy_row(1, yes)])
    write_jsonl_records(tmp_path / "b.jsonl", [_greedy_row(0, yes), _greedy_row(1, no)])
    args = ["greedy-agreement", "--first", str(tmp_path / "a.jsonl"), "--second", str(tmp_path / "b.jsonl"),
            "--rules", str(rules), "--report", str(tmp_path / "g.json")]
    assert lp.main(args) == 1
    assert json.loads((tmp_path / "g.json").read_text())["agreement"] == 0.5
    assert lp.main([*args, "--min-agreement", "0.5", "--no-strict-think"]) == 0
    assert '"agreement": 0.5' in capsys.readouterr().out
    report = json.loads((tmp_path / "g.json").read_text())
    assert report["greedy_verdict"] == {**lp.GREEDY_VERDICT_GATE, "min_agreement": 0.5, "strict_think": False}


def test_cli_vllm_and_hf_write_greedy_rows(tmp_path, monkeypatch, fake_vllm, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    weights = tmp_path / "Qwen3-8B"
    weights.mkdir()
    (weights / "generation_config.json").write_text(json.dumps({"eos_token_id": [2]}))
    prompts = _prompts(tmp_path, n=2)
    seqs = tmp_path / "s.jsonl"
    lp.write_sequences(seqs, [lp.FrozenSequence("seq-0000", (1, 2), (3,))])
    monkeypatch.setattr(lp, "prompt_token_ids", lambda tokenizer, prompts, level=None: [[1, 2]] * len(prompts))
    greedy_from = []
    monkeypatch.setattr(lp, "vllm_greedy", lambda llm, ids, max_tokens, **kw: greedy_from.append(("vllm", kw)) or [
        ([5, 2], "stop")] * len(ids))
    monkeypatch.setattr(lp, "hf_greedy", lambda model, ids, **kw: greedy_from.append(("hf", model)) or [
        ([6, 2], "stop")] * len(ids))
    monkeypatch.setattr(lp, "_load_tokenizer", lambda path: _IdTokenizer())
    _patch_engine(monkeypatch, _ScoringLLM(), _IdTokenizer())
    assert lp.main(["vllm", "--model-config", str(QWEN3_CONFIG), "--arm", "vllm_base", "--sequences", str(seqs),
                    "--out", str(tmp_path / "v.json"), "--greedy-prompts", str(prompts),
                    "--greedy-out", str(tmp_path / "gv.jsonl")]) == 0
    assert [row["completion_ids"] for row in load_jsonl_records(tmp_path / "gv.jsonl")] == [[5], [5]]
    monkeypatch.setattr(lp, "load_hf_model", lambda path, **kw: "MODEL")
    monkeypatch.setattr(lp, "score_hf", lambda model, adapter, sequences, device, arm, autocast_arm, wrapped: lp.Scores(
        {arm: {seq.seq_id: [0.0] * (len(seq.token_ids) - 1) for seq in sequences}}))
    hf = ["hf", "--model-config", str(QWEN3_FP32_CONFIG), "--sequences", str(seqs), "--device", "cpu",
          "--out", str(tmp_path / "h.json"), "--greedy-prompts", str(prompts), "--greedy-out", str(tmp_path / "gh.jsonl")]
    assert lp.main(hf) == 0
    assert [row["text"] for row in load_jsonl_records(tmp_path / "gh.jsonl")] == ["g", "g"]
    assert greedy_from == [("vllm", {}), ("hf", "MODEL")]
    with pytest.raises(lp.ParityError, match="go together"):
        lp.main(hf[:-2])
    # lora mode: the adapter's greedy outputs, HF-PEFT's in a greedy-only run in the served dtype
    monkeypatch.setattr(lp, "wrap_adapter", lambda model, adapter: ("PEFT", adapter))
    greedy_only = ["hf", "--model-config", str(QWEN3_CONFIG), "--dtype", "bfloat16", "--adapter", str(tmp_path),
                   "--device", "cpu", "--greedy-prompts", str(prompts), "--greedy-out", str(tmp_path / "ga.jsonl")]
    assert lp.main(greedy_only) == 0
    assert greedy_from[-1] == ("hf", ("PEFT", tmp_path)) and len(load_jsonl_records(tmp_path / "ga.jsonl")) == 2
    with pytest.raises(lp.ParityError, match="nothing to run"):
        lp.main(greedy_only[:-4])
    with pytest.raises(lp.ParityError, match="the decode arm is scored beside the prefill one"):
        lp.main([*greedy_only, "--decode-sequences", str(seqs), "--decode-out", str(tmp_path / "hd.json")])


def test_the_vllm_greedy_arm_encodes_through_the_engines_thinking_tokenizer(tmp_path, monkeypatch, fake_vllm):
    from ruleforge.judges.models import _ThinkingTokenizer

    class _Recording(_IdTokenizer):
        def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True, **kwargs):
            rendered.append(kwargs)
            return super().apply_chat_template(chat, tokenize, add_generation_prompt, **kwargs)

    rendered: list = []
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    weights = tmp_path / "Qwen3-8B"
    weights.mkdir()
    (weights / "generation_config.json").write_text(json.dumps({"eos_token_id": [2]}))
    seqs = tmp_path / "s.jsonl"
    lp.write_sequences(seqs, [lp.FrozenSequence("seq-0000", (1, 2), (3,))])
    generated_from = []
    monkeypatch.setattr(lp, "vllm_greedy", lambda llm, ids, max_tokens, **kw: generated_from.append(ids) or [
        ([5, 2], "stop")] * len(ids))
    _patch_engine(monkeypatch, _ScoringLLM(), _ThinkingTokenizer(_Recording(), True))
    assert lp.main(["vllm", "--model-config", str(QWEN3_CONFIG), "--arm", "vllm_base", "--sequences", str(seqs),
                    "--out", str(tmp_path / "v.json"), "--greedy-prompts", str(_prompts(tmp_path, n=2)),
                    "--greedy-out", str(tmp_path / "gv.jsonl")]) == 0
    assert generated_from == [[[ord(c) % 50 for c in f"system:S{i}|user:"] for i in range(2)]]
    assert rendered == [{"enable_thinking": True}] * 2
    assert [row["completion_ids"] for row in load_jsonl_records(tmp_path / "gv.jsonl")] == [[5], [5]]


def test_the_plumbing_tokenizer_is_the_directorys_own(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: seen.append(path) or "TOK")
    assert lp._load_tokenizer(tmp_path) == "TOK" and seen == [str(tmp_path)]


def test_the_lora_presets_hold_rho_two_sided_with_no_merge_to_test():
    b, m0 = preset_thresholds("b_lora", bucket_edges=()), preset_thresholds("m0_lora")
    assert (b.rho_range, b.rho_precision, b.require_effect, b.rho_hf_range) == ((0.95, 1.05), 0.01, False, None)
    assert (m0.rho_range, m0.rho_per_bucket, m0.require_effect) == ((0.95, 1.05), True, True)
    # lora mode's arms: vllm_base and vllm_lora on one engine, against HF-PEFT
    faithful, seqs = _paired_arms(served_arm=lp.LORA_ARM, served_noise=0.002)
    report = lp.compare(faithful, seqs, b)
    assert report.mode == "lora" and report.passed and report.rho_gated == {"all": True}, report.failures
    for scale in (1.5, 0.5, 0.0):
        scores, seqs = _paired_arms(served_arm=lp.LORA_ARM, served_noise=0.002, served_scale=scale)
        failures = lp.compare(scores, seqs, preset_thresholds("b_lora", **LOOSE)).failures
        assert any("retention rho" in f for f in failures), (scale, failures)
    weak, seqs = _paired_arms(served_arm=lp.LORA_ARM, effect_scale=0.02, served_noise=0.001)
    assert lp.compare(weak, seqs, b).inconclusive
    assert any("too weak to test" in f for f in lp.compare(weak, seqs, preset_thresholds("m0_lora")).failures)


def test_vllm_generates_greedily_from_the_served_adapter(fake_vllm):
    class _GreedyLLM:
        def generate(self, prompts, sampling_params=None, use_tqdm=True, **kwargs):
            self.kwargs = kwargs
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[3, 2], finish_reason="stop")])
                    for _ in prompts]

    llm = _GreedyLLM()
    lp.vllm_greedy(llm, [[1]], max_tokens=5, lora_request="REQ")
    assert llm.kwargs == {"lora_request": "REQ"}
    lp.vllm_greedy(llm, [[1]], max_tokens=5)
    assert llm.kwargs == {}


def test_cli_vllm_lora_mode_generates_greedily_with_the_adapter(tmp_path, monkeypatch, fake_vllm, capsys):
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path))
    (tmp_path / lp.ADAPTER_WEIGHTS_FNAME).write_bytes(b"factors")  # the adapter the arms record by digest
    (tmp_path / lp.ADAPTER_CONFIG_FNAME).write_text("{}")
    weights = tmp_path / "Qwen3-8B"
    weights.mkdir()
    (weights / "generation_config.json").write_text(json.dumps({"eos_token_id": [2]}))
    _patch_engine(monkeypatch, _ScoringLLM(), _IdTokenizer())
    seen = []
    monkeypatch.setattr(lp, "vllm_greedy", lambda llm, ids, max_tokens, **kw: seen.append(kw) or [([5, 2], "stop")] * len(ids))
    monkeypatch.setattr(lp, "prompt_token_ids", lambda tokenizer, prompts, level=None: [[1, 2]] * len(prompts))
    seqs = tmp_path / "s.jsonl"
    lp.write_sequences(seqs, [lp.FrozenSequence("seq-0000", (1, 2), (3,))])
    assert lp.main(["vllm", "--model-config", str(QWEN3_CONFIG), "--adapter", str(tmp_path), "--sequences", str(seqs),
                    "--out", str(tmp_path / "v.json"), "--greedy-prompts", str(_prompts(tmp_path, n=2)),
                    "--greedy-out", str(tmp_path / "gv.jsonl")]) == 0
    assert seen == [{"lora_request": "REQ"}] and len(load_jsonl_records(tmp_path / "gv.jsonl")) == 2
    assert '"arm": "vllm_lora"' in capsys.readouterr().out


def test_hf_greedy_runs_on_the_peft_model_with_its_adapter(llama_dir, tmp_path):
    adapter = tmp_path / "a"
    assert lp.make_random_adapter(llama_dir, adapter, rank=4, alpha=8.0, seed=1, b_std=2.0).ok
    plain = lp.load_hf_model(llama_dir, dtype="float32", device="cpu")
    [(base, _)] = lp.hf_greedy(plain, [[1, 2, 3]], max_tokens=6, eos_ids=(47,), device="cpu")
    wrapped = lp.wrap_adapter(lp.load_hf_model(llama_dir, dtype="float32", device="cpu"), adapter)
    [(tuned, _)] = lp.hf_greedy(wrapped, [[1, 2, 3]], max_tokens=6, eos_ids=(47,), device="cpu")
    assert len(tuned) == 6 and tuned != base
