from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import tomllib
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="needs the grpo group (CPU torch is enough)")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from ruleforge.grpo import trainer as tn  # noqa: E402
from ruleforge.grpo import grpo_data as gd  # noqa: E402
from ruleforge.grpo.kl_anchor import AnchorAuxConfig  # noqa: E402
from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path  # noqa: E402
from ruleforge.grpo.sync_probe import SyncProbe, SyncProbeError  # noqa: E402
from ruleforge.grpo.truncation import Truncation  # noqa: E402
from ruleforge.core.digests import canonical_digest, sha256_file  # noqa: E402
from tests.conftest import requires_git_worktree  # noqa: E402

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "grpo"
RULES = REPO_ROOT / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"
RULES_SHA256 = "a99062f00a9ff616cba4d548e9cd59460dc4ee52fc46e3de4a4eb2579a260228"
SAMPLING_TRUNCATION = {"top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0}

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
WORDS = ["<think>", "</think>", "system", "user", "assistant", "judge", "the", "scenario", "rule", "yes", "no",
         "unsure", "[UNK]"]
VOCAB = {token: index for index, token in enumerate(SPECIALS + WORDS)}
PAD, EOS, THINK, THINK_END = VOCAB["<|endoftext|>"], VOCAB["<|im_end|>"], VOCAB["<think>"], VOCAB["</think>"]
CHAT_TEMPLATE = (
    "{%- for m in messages -%}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|> assistant {% endif -%}"
)
TINY_QWEN3 = dict(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=8, max_position_embeddings=256, tie_word_embeddings=False,
    eos_token_id=EOS, pad_token_id=PAD, bos_token_id=PAD,
)


def _save_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(SPECIALS)
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token="[UNK]",
        chat_template=CHAT_TEMPLATE,
    ).save_pretrained(str(directory))


def _save_tiny_qwen3(directory: Path, *, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**TINY_QWEN3))
    with torch.no_grad():
        # a strong end-of-turn direction, so that the policy (and the fake engine) finish some completions
        model.lm_head.weight[EOS] *= 30.0
    model.to(torch.bfloat16).save_pretrained(str(directory))
    (directory / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 0.6, "top_k": 20, "top_p": 0.95, "eos_token_id": [EOS, PAD],
         "pad_token_id": PAD}
    ))
    _save_tokenizer(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    return _save_tiny_qwen3(tmp_path_factory.mktemp("models") / "Qwen3-tiny")


def _grpo_row(index: int) -> dict:
    return {
        "prompt": [{"role": "system", "content": "judge the scenario rule" + " the" * (index % 3)},
                   {"role": "user", "content": ""}],
        "q_yes": 0.7, "q_unsure": 0.1, "q_no": 0.2, "anchor_class": "c", "topped_up": False,
        "base_yes_rules": ["Rule 1"], "row_id": f"s{index}",
    }


def _replay_row(index: int) -> dict:
    prompt = [VOCAB["<|im_start|>"], VOCAB["system"], VOCAB["rule"], EOS, VOCAB["<|im_start|>"], VOCAB["assistant"]]
    completion = [THINK, VOCAB["judge"], VOCAB["the"], THINK_END, VOCAB["yes" if index % 2 else "no"], EOS]
    return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion,
            "weight": 0.5 if index % 3 else 0.25, "scenario_id": f"s{index}", "rule_id": f"Rule {index}"}


def _write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    disjoint_marker_path(path).write_text(json.dumps({"kind": "prompt_disjoint", "disjoint": True}))
    return path


def marker_double(dataset: Path, marker_path: Path) -> dict:
    """Stands in for check_prompt_marker: the marker must say it is a prompt-disjoint one."""
    marker = json.loads(Path(marker_path).read_text())
    if marker.get("kind") != "prompt_disjoint":
        raise LeakageError(f"{marker_path} is not a prompt-disjoint marker")
    return marker


class RewardDouble:
    """Stands in for GrpoReward: the three reward functions by name, weighted [1, λ_i, λ_c]."""

    finish_token_ids = (EOS, PAD)  # the tiny tokenizer's eos and pad, as QWEN3_FINISH_TOKEN_IDS are Qwen3's

    def __init__(self, rules, *, truncated, num_generations, lambda_int, lambda_comp, completion_format=None):
        self.rules, self.truncated, self.num_generations = rules, truncated, num_generations
        self.lambda_int, self.lambda_comp = lambda_int, lambda_comp
        self.completion_format = completion_format

    @property
    def reward_funcs(self):
        return [self.flat_soft_reward, self.internal_consistency_reward, self.map_completeness_reward]

    @property
    def reward_weights(self):
        return [1.0, self.lambda_int, self.lambda_comp]

    def flat_soft_reward(self, completions, completion_ids, **columns):
        assert set(columns) >= {"q_yes", "anchor_class", "base_yes_rules", "log_metric", "trainer_state"}
        columns["log_metric"]("consistency/double_rows", float(len(completions)))
        return [float(sum(ids) % 7) / 7 for ids in completion_ids]

    def internal_consistency_reward(self, completions, completion_ids, **columns):
        return [float(ids[-1] == EOS) for ids in completion_ids]

    def map_completeness_reward(self, completions, completion_ids, **columns):
        return [float(len(ids) % 2) for ids in completion_ids]


def _condition(model_dir: Path, **changes) -> dict:
    return {"model_path": model_dir.name, "adapter_path": None, "rule_set_sha256": RULES_SHA256,
            "temperature": 1.0, "sampling_truncation": dict(SAMPLING_TRUNCATION), **changes}


def _compose(overrides):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="training/grpo/base", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _built_on(model: Path, train: Path, replay: Path) -> Path:
    from ruleforge.judges.completion_format import load_training_format
    from ruleforge.grpo.grpo_data import GRPO_ROWS_FNAME, MANIFEST_FNAME, REPLAY_ROWS_FNAME, tokenizer_record

    tokenizer, fmt = load_training_format(model)
    manifest = train.parent / MANIFEST_FNAME
    manifest.write_text(json.dumps({
        "outputs": {GRPO_ROWS_FNAME: {"sha256": sha256_file(train)}, REPLAY_ROWS_FNAME: {"sha256": sha256_file(replay)}},
        "tokenizer": tokenizer_record(model, tokenizer, fmt),
        "rendering": {"chat_template_kwargs": {"enable_thinking": True}}}))
    return manifest


@pytest.fixture
def run_files(tiny_model, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    condition = data / "condition.json"
    condition.write_text(json.dumps(_condition(tiny_model)))
    files = {
        "train": _write_jsonl(data / "grpo_train.jsonl", [_grpo_row(i) for i in range(8)]),
        "replay": _write_jsonl(data / "replay.jsonl", [_replay_row(i) for i in range(6)]),
        "condition": condition,
        "root": tmp_path / "artifacts",
    }
    files["manifest"] = _built_on(tiny_model, files["train"], files["replay"])
    return files


def _config(run_files, tiny_model, monkeypatch, *extra) -> dict:
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(run_files["root"]))
    return _compose([
        "run_name=dry-run",
        f"model.path={tiny_model}",
        f"data.train={run_files['train']}",
        f"data.replay={run_files['replay']}",
        f"data.anchor_condition={run_files['condition']}",
        f"data.rules={RULES}",
        "data.prompt_token_cap=64",
        "reward.truncated=zero",
        "sync_probe.max_rows=64",
        "retention_alarm.step=1",
        "retention_alarm.threshold=2.0",
        "grpo.num_generations=2",
        "grpo.per_device_train_batch_size=2",
        "grpo.gradient_accumulation_steps=2",
        "grpo.max_completion_length=12",
        "grpo.max_steps=3",
        "grpo.save_steps=1",
        "grpo.learning_rate=0.05",
        "grpo.warmup_steps=0",
        *(() if any(item.startswith("grpo.report_to=") for item in extra) else ("grpo.report_to=[]",)),
        *extra,
    ])


def _plan(run_files, tiny_model, monkeypatch, *extra, **kwargs) -> tn.GrpoPlan:
    cfg = _config(run_files, tiny_model, monkeypatch, *extra)
    return tn.plan_grpo(cfg, reward_factory=kwargs.get("reward_factory", RewardDouble),
                        marker_checker=kwargs.get("marker_checker", marker_double))


CPU = {"use_vllm": False, "use_cpu": True}
#: bf16 autocast on a CPU runner needs TrainingArguments.use_cpu.
ON_CPU = ("+grpo.use_cpu=true",)


def preset_modules(config, preset):
    """Return `(wrapped module paths, trainable parameters)` of the preset on `config`'s class, on the meta device."""
    from peft import get_peft_model
    from peft.tuners.lora import LoraLayer

    model_class = getattr(transformers, config.architectures[0])
    with torch.device("meta"):
        model = get_peft_model(model_class(config), preset.lora_config())
    wrapped = [
        name.removeprefix(tn._PEFT_PREFIX) for name, module in model.named_modules() if isinstance(module, LoraLayer)
    ]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return wrapped, trainable


def test_the_qwen3_preset_wraps_252_modules_on_the_meta_device():
    """Qwen3-8B's own config (a copy of config.json): 7 projections x 36 layers, 87.3M parameters at r 32."""
    config = transformers.AutoConfig.from_pretrained(str(FIXTURES / "qwen3_8b"))
    preset = tn.lora_preset(config.architectures[0])
    wrapped, trainable = preset_modules(config, preset)
    assert len(wrapped) == tn.expected_module_count(config, preset) == 252
    assert trainable == 87_293_952
    assert {name.rpartition(".")[2] for name in wrapped} == set(preset.target_modules)
    assert (preset.r, preset.alpha, preset.dropout, preset.master_dtype) == (32, 64, 0.0, "float32")
    lora = preset.lora_config()
    assert (lora.r, lora.lora_alpha, lora.bias, set(lora.target_modules)) == (32, 64, "none", set(preset.target_modules))
    assert preset.model_init_kwargs("sdpa") == {"dtype": "float32", "attn_implementation": "sdpa"}
    with pytest.raises(tn.TrainerConfigError, match="no LoRA preset"):
        tn.lora_preset("Qwen3_5ForConditionalGeneration")


def test_the_truncation_is_what_the_model_ships_checked_against_the_anchor_condition(tiny_model):
    assert tn.effective_truncation(tiny_model) == SAMPLING_TRUNCATION  # never the shipped temperature 0.6
    truncation = tn.check_anchor_condition(_condition(tiny_model), model_path=tiny_model, rules_sha256=RULES_SHA256)
    assert truncation == Truncation(temperature=1.0, top_k=20, top_p=0.95)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"sampling_truncation": {**SAMPLING_TRUNCATION, "top_p": 1.0}}, "sampling_truncation"),
        ({"sampling_truncation": "provider_default"}, "sampling_truncation"),
        ({"model_path": "Qwen3-32B"}, "served 'Qwen3-32B'"),
        ({"adapter_path": "a/adapter"}, "adapter"),
        ({"rule_set_sha256": "0" * 64}, "ruleset digest"),
        ({"temperature": None}, "temperature"),
        ({"temperature": True}, "temperature"),
    ],
)
def test_an_anchor_condition_from_another_decoder_model_or_ruleset_is_refused(tiny_model, changes, match):
    with pytest.raises(tn.TrainerConfigError, match=match):
        tn.check_anchor_condition(_condition(tiny_model, **changes), model_path=tiny_model, rules_sha256=RULES_SHA256)


def _rewards(**kwargs):
    settings = dict(rules=[], truncated="zero", num_generations=8, lambda_int=0.3, lambda_comp=0.1,
                    factory=RewardDouble)
    return tn.build_reward_set(**{**settings, **kwargs})


def test_the_reward_set_is_the_three_section_6_functions_weighted_one_lambda_i_lambda_c():
    reward_set = _rewards()
    assert [func.__name__ for func in reward_set.reward_funcs] == list(tn.REWARD_FUNC_NAMES)
    assert reward_set.num_generations == 8 and reward_set.truncated == "zero"

    class Renamed(RewardDouble):
        @property
        def reward_funcs(self):
            return [self.map_completeness_reward, self.flat_soft_reward, self.internal_consistency_reward]

    class Reweighted(RewardDouble):
        @property
        def reward_weights(self):
            return [1.0, 0.1, 0.3]

    with pytest.raises(tn.TrainerConfigError, match="reward functions are"):
        _rewards(factory=Renamed)
    with pytest.raises(tn.TrainerConfigError, match="reward weights are"):
        _rewards(factory=Reweighted)


def test_the_plan_line_names_the_reward_form():
    assert tn.reward_form(_rewards()) == (
        "sum (1 flat_soft_reward + 0.3 internal_consistency_reward + 0.1 map_completeness_reward)")


def test_the_reward_seam_imports_item_10s_module_and_refuses_without_it(monkeypatch):
    monkeypatch.setattr(tn, "REWARD_FACTORY", "tests.test_training_trainer:RewardDouble")
    assert isinstance(_rewards(factory=None), RewardDouble)
    monkeypatch.setattr(tn, "REWARD_FACTORY", "ruleforge.grpo.no_such_rewards:GrpoReward")
    with pytest.raises(tn.TrainerConfigError, match="ruleforge.grpo.rewards must be importable before a run can start"):
        _rewards(factory=None)
    with pytest.raises(tn.TrainerConfigError, match="module:attribute"):
        tn.import_object("ruleforge.grpo.trainer")


def test_the_marker_is_required(tmp_path, monkeypatch):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text('{"prompt": 1}\n')
    with pytest.raises(LeakageError, match="carries no disjointness marker"):
        tn.require_disjoint_marker(dataset, checker=marker_double)
    disjoint_marker_path(dataset).write_text(json.dumps({"kind": "pair_hash"}))
    with pytest.raises(LeakageError, match="not a prompt-disjoint marker"):
        tn.require_disjoint_marker(dataset, checker=marker_double)
    disjoint_marker_path(dataset).write_text(json.dumps({"kind": "prompt_disjoint"}))
    record = tn.require_disjoint_marker(dataset, checker=marker_double)
    assert record["dataset_sha256"] == sha256_file(dataset) and record["marker_kind"] == "prompt_disjoint"
    monkeypatch.setattr(tn, "MARKER_CHECKER", "ruleforge.grpo.no_such_exclusion:check_prompt_marker")
    with pytest.raises(LeakageError, match="the exclusion module must be importable to validate the marker"):
        tn.require_disjoint_marker(dataset)
    monkeypatch.setattr(tn, "MARKER_CHECKER", "tests.test_training_trainer:marker_double")
    assert tn.require_disjoint_marker(dataset)["marker"] == str(disjoint_marker_path(dataset))


def test_unscreened_rows_are_taken_only_by_the_flag_which_the_record_then_carries(tmp_path):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text('{"prompt": 1}\n')
    disjoint_marker_path(dataset).write_text(json.dumps({"kind": "prompt_disjoint",
                                                         "screen": "containment-only, unscreened"}))
    seen = []

    def checker(dataset, marker_path, **kwargs):
        seen.append(kwargs)
        if not kwargs.get("allow_unscreened"):
            raise LeakageError(f"{marker_path} attests the second pass alone")
        return json.loads(Path(marker_path).read_text())

    with pytest.raises(LeakageError, match="second pass alone"):
        tn.require_disjoint_marker(dataset, checker=checker)
    record = tn.require_disjoint_marker(dataset, checker=checker, allow_unscreened=True)
    assert seen == [{}, {"allow_unscreened": True}]  # as the published runs' checker was called, then with the flag
    assert record["allow_unscreened_rows"] is True and record["screen"] == "containment-only, unscreened"
    assert {"allow_unscreened_rows", "screen"}.isdisjoint(tn.require_disjoint_marker(dataset, checker=marker_double))


@pytest.mark.parametrize("override", ["no", "off", '"false"', "1"])
def test_a_value_meant_as_off_does_not_take_unscreened_rows(run_files, tiny_model, monkeypatch, override):
    with pytest.raises(tn.TrainerConfigError, match="data.allow_unscreened_rows is true or false"):
        _plan(run_files, tiny_model, monkeypatch, f"data.allow_unscreened_rows={override}")


@pytest.mark.parametrize(
    "prompt",
    [
        "judge the scenario",
        [{"role": "user", "content": "judge the scenario"}],
        [{"role": "system", "content": "judge"}, {"role": "user", "content": "x"}],
        [{"role": "system", "content": "  "}, {"role": "user", "content": ""}],
        [{"role": "system", "content": "judge"}, "user"],
        [{"role": "system", "content": "a"}, {"role": "user", "content": ""}, {"role": "user", "content": ""}],
    ],
)
def test_a_prompt_that_is_not_the_campaign_rendering_is_refused(prompt):
    tn.check_prompt_shape([_grpo_row(0)])
    with pytest.raises(tn.TrainerConfigError, match="system: <prompt>, user"):
        tn.check_prompt_shape([_grpo_row(0), {**_grpo_row(1), "prompt": prompt}])


def test_prompt_lengths_are_counted_as_trl_renders_them(tiny_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_model))
    counts = tn.prompt_token_counts([_grpo_row(0), _grpo_row(2)], tokenizer, {"enable_thinking": True})
    assert counts == [12, 14]  # <|im_start|> system judge the scenario rule <|im_end|> ... <|im_start|> assistant
    tn.check_prompt_lengths(counts, 14)
    with pytest.raises(tn.TrainerConfigError, match="1 GRPO prompt"):
        tn.check_prompt_lengths(counts, 13)


def _kwargs(grpo, **kwargs):
    settings = dict(truncation=Truncation(temperature=1.0, top_k=20, top_p=0.95), reward_weights=[1.0, 0.3, 0.1],
                    model_init_kwargs={"dtype": "float32"}, output_dir="/out", vllm_length_margin=5120, anchored=True)
    return tn.grpo_config_kwargs(grpo, **{**settings, **kwargs})


def _base_grpo(monkeypatch) -> dict:
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", "/artifacts")
    return _compose(["run_name=x", "model.path=/m"])["grpo"]


def test_every_field_the_run_depends_on_is_explicit_and_the_derived_ones_are_derived(monkeypatch):
    grpo = _base_grpo(monkeypatch)
    kwargs, forced = _kwargs(grpo)
    assert forced == {}
    assert {k: kwargs[k] for k in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty")} == {
        "temperature": 1.0, **SAMPLING_TRUNCATION}
    assert kwargs["generation_kwargs"] is None and kwargs["reward_weights"] == [1.0, 0.3, 0.1]
    assert kwargs["vllm_max_model_length"] == 6144 + 5120 and kwargs["output_dir"] == "/out"
    assert kwargs["run_name"] == "out" and kwargs["report_to"] == ["wandb"]
    missing = {k: v for k, v in grpo.items() if k != "use_bias_correction_kl"}
    with pytest.raises(tn.TrainerConfigError, match=r"\['use_bias_correction_kl'\] unset"):
        _kwargs(missing)
    with pytest.raises(tn.TrainerConfigError, match="grpo.top_p is derived"):
        _kwargs({**grpo, "top_p": 1.0})
    with pytest.raises(tn.TrainerConfigError, match="grpo.run_name is derived"):
        _kwargs({**grpo, "run_name": "another"})


def test_the_anchor_forces_beta_zero_and_masked_truncations(monkeypatch):
    """The anchor replaces TRL's KL; truncated completions carry no gradient."""
    grpo = {**_base_grpo(monkeypatch), "beta": 0.04, "mask_truncated_completions": False}
    kwargs, forced = _kwargs(grpo)
    assert (kwargs["beta"], kwargs["mask_truncated_completions"]) == (0.0, True)
    assert forced == {"beta": [0.04, 0.0], "mask_truncated_completions": [False, True]}
    unanchored, none_forced = _kwargs(grpo, anchored=False)
    assert unanchored["beta"] == 0.04 and none_forced == {}


def test_fsdps_rank_0_loading_is_refused(monkeypatch):
    grpo = {**_base_grpo(monkeypatch), "fsdp": True, "fsdp_config": {"version": 2, "cpu_ram_efficient_loading": True}}
    for key in ("cpu_ram_efficient_loading", "fsdp_cpu_ram_efficient_loading"):  # transformers strips "fsdp_"
        with pytest.raises(tn.TrainerConfigError, match="cpu_ram_efficient_loading.*persistent buffer"):
            _kwargs({**grpo, "fsdp_config": {"version": 2, key: "true"}})
    for setting in (False, "false"):
        _kwargs({**grpo, "fsdp_config": {"version": 2, "cpu_ram_efficient_loading": setting}})
    _kwargs({**grpo, "fsdp": None})


def test_the_adaptive_entropy_controller_is_refused_under_fsdp2(monkeypatch):
    grpo = {**_base_grpo(monkeypatch), "fsdp": True, "use_adaptive_entropy": True}
    for config in ({"version": 2}, {}, {"fsdp_version": 2}):
        with pytest.raises(tn.TrainerConfigError, match="use_adaptive_entropy under FSDP2"):
            _kwargs({**grpo, "fsdp_config": config})
    _kwargs({**grpo, "fsdp_config": {"version": 1}})
    _kwargs({**grpo, "fsdp": None})


def test_the_design_values_reach_grpo_config(monkeypatch):
    kwargs, _ = _kwargs(_base_grpo(monkeypatch))
    args = tn.build_grpo_config({**kwargs, "use_cpu": True, "output_dir": "/tmp/unused"})
    assert (args.learning_rate, args.lr_scheduler_type.value, args.warmup_steps, args.max_steps) == (
        2e-5, "constant_with_warmup", 10, 600)
    assert (args.beta, args.loss_type, args.scale_rewards, args.multi_objective_aggregation) == (
        0.0, "dapo", "batch", "sum_then_normalize")
    assert (args.vllm_importance_sampling_mode, args.vllm_importance_sampling_clip_max) == ("token_truncate", 2.0)
    assert (args.num_generations, args.per_device_train_batch_size, args.steps_per_generation) == (8, 2, 8)
    assert (args.epsilon, args.epsilon_high, args.delta, args.top_entropy_quantile) == (0.2, None, None, 1.0)
    assert args.off_policy_mask_threshold is None and args.vllm_importance_sampling_clip_min is None
    assert (args.max_completion_length, args.vllm_max_model_length, args.vllm_gpu_memory_utilization) == (
        6144, 11264, 0.30)
    assert (args.top_k, args.top_p, args.min_p, args.temperature) == (20, 0.95, 0.0, 1.0)
    assert args.mask_truncated_completions and args.vllm_enable_sleep_mode and not args.use_bias_correction_kl
    assert args.model_init_kwargs == {"dtype": "float32"} and args.save_total_limit is None
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert args.chat_template_kwargs == {"enable_thinking": True} and args.reward_weights == [1.0, 0.3, 0.1]


def test_each_rank_generates_whole_groups():
    tn.check_whole_groups(2, 8, 8)
    with pytest.raises(tn.TrainerConfigError, match="straddle two ranks"):
        tn.check_whole_groups(2, 3, 8)


SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_JOB_NUM_NODES", "SLURM_PROCID", "SLURM_LOCALID",
              "SLURM_JOB_NODELIST", "SUBMITIT_LOCAL_JOB_ID", "_TEST_CLUSTER_", "MASTER_PORT")


def test_a_two_task_submitit_job_becomes_two_torch_ranks(monkeypatch):
    for key in SLURM_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert tn.export_submitit_distributed_env({"WORLD_SIZE": "2"}) is None  # accelerate or torchrun did it
    assert tn.export_submitit_distributed_env({}) is None  # not a submitit job
    monkeypatch.setenv("_TEST_CLUSTER_", "slurm")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert tn.export_submitit_distributed_env({}) is None
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "vp-node[07-08]")
    environ: dict = {}
    values = tn.export_submitit_distributed_env(environ)
    assert values == environ
    assert {k: environ[k] for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR")} == {
        "RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "MASTER_ADDR": "vp-node07"}
    assert 20000 <= int(environ["MASTER_PORT"]) <= 60000
    assert tn.export_submitit_distributed_env({"MASTER_PORT": "29555"})["MASTER_PORT"] == "29555"


def test_the_stack_names_the_libraries_and_the_lock(tmp_path):
    stack = tn.stack_versions()
    assert set(stack) == set(tn.STACK_PACKAGES) | {"uv_lock_sha256"}
    assert stack["trl"] and stack["uv_lock_sha256"] == sha256_file(REPO_ROOT / "uv.lock")
    assert tn.stack_versions(tmp_path)["uv_lock_sha256"] is None


def test_a_checkpoint_resumes_only_under_its_identity(tmp_path):
    preset, truncation = tn.lora_preset("Qwen3ForCausalLM"), Truncation(temperature=1.0, top_k=20, top_p=0.95)
    config = {"grpo": {"max_steps": 600}, "resume_from_checkpoint": None, "run_name": "a"}
    identity = tn.run_identity(config=config, digests={"train": "d"}, preset=preset, truncation=truncation,
                               stack={"trl": "1.13.0"})
    moved = {**config, "resume_from_checkpoint": "latest", "run_name": "b", "hydra": {"x": 1},
             "wandb": {"project": "ruleforge", "group": "g"}}
    assert tn.run_identity(config=moved, digests={"train": "d"}, preset=preset, truncation=truncation,
                           stack={"trl": "1.13.0"}) == identity
    out = tmp_path / "run"
    assert tn.resolve_resume(None, out) is None and tn.resolve_resume("latest", out) is None
    checkpoint = out / "checkpoint-25"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}")
    tn.write_provenance(checkpoint, {"run": {"identity": identity}})
    with pytest.raises(tn.TrainerConfigError, match="already holds checkpoint-25"):
        tn.resolve_resume(None, out)
    assert tn.resolve_resume("latest", out) == checkpoint == tn.resolve_resume(str(checkpoint), out)
    with pytest.raises(tn.TrainerConfigError, match="not a checkpoint this trainer wrote"):
        tn.resolve_resume(str(out), out)
    # a job killed mid-save leaves a newer directory without provenance: latest skips it, a fresh run refuses
    partial = out / "checkpoint-50"
    partial.mkdir()
    (partial / tn.ADAPTER_WEIGHTS_FNAME).write_bytes(b"half")
    (out / "checkpoint-notes").mkdir()
    (out / "checkpoint-75.json").write_text("{}")
    assert [path.name for path in tn.checkpoint_dirs(out)] == ["checkpoint-50", "checkpoint-25"]
    assert tn.resolve_resume("latest", out) == checkpoint
    with pytest.raises(tn.TrainerConfigError, match="already holds checkpoint-50"):
        tn.resolve_resume(None, out)
    (checkpoint / tn.PROVENANCE_FNAME).unlink()
    assert tn.resolve_resume("latest", out) is None
    tn.write_provenance(checkpoint, {"step": 25, "run": {"identity": identity}})
    assert tn.check_resume(checkpoint, identity, max_steps=26)["run"]["identity"] == identity
    other = {**identity, "config_sha256": "0" * 64, "stack": {"trl": "1.14.0"}}
    named = r"another config_sha256, stack; a resume continues the same run or none \(recorded -> this run: stack.trl "
    with pytest.raises(tn.TrainerConfigError, match=named + r"'1.13.0' -> '1.14.0'\)"):
        tn.check_resume(checkpoint, other, max_steps=26)
    # the config's digest alone, this run's config not given: the refusal names the digest only
    with pytest.raises(tn.TrainerConfigError, match=r"another config_sha256; a resume continues the same run or "
                                                    r"none\. Resume it under the code, lock and config"):
        tn.check_resume(checkpoint, {**identity, "config_sha256": "0" * 64}, max_steps=26)
    # a checkpoint at (or past) max_steps is a finished run: HF would train one step past it
    for max_steps in (25, 24):
        with pytest.raises(tn.TrainerConfigError, match=f"at step 25, already at max_steps {max_steps}: the run is "
                                                        "finished"):
            tn.check_resume(checkpoint, identity, max_steps=max_steps)
    assert tn.adapter_digest(checkpoint) is None
    (checkpoint / tn.ADAPTER_WEIGHTS_FNAME).write_bytes(b"weights")
    # the ranks: the preflight runs as one process, so the run checks them once its GRPOConfig knows them
    two_ranks = types.SimpleNamespace(world_size=2, generation_batch_size=32)
    tn.check_resume_geometry({"batch_geometry": tn.batch_geometry(two_ranks)}, two_ranks)
    one_rank = types.SimpleNamespace(world_size=1, generation_batch_size=16)
    with pytest.raises(tn.TrainerConfigError, match="'world_size': 2.*this run is .*'world_size': 1"):
        tn.check_resume_geometry({"batch_geometry": tn.batch_geometry(two_ranks)}, one_rank)
    with pytest.raises(tn.TrainerConfigError, match="trained at None"):
        tn.check_resume_geometry({}, one_rank)
    assert tn.adapter_digest(checkpoint) == sha256_file(checkpoint / tn.ADAPTER_WEIGHTS_FNAME)


#: What W&B reads from the environment for a run, as the trainer sets it (and conftest's WANDB_MODE).
WANDB_VARIABLES = ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_DIR",
                   "WANDB_SYMLINK", "WANDB_MODE")


def _isolated_wandb_env(monkeypatch) -> None:
    """No W&B variable set but conftest's ``WANDB_MODE=disabled``; whatever the test exports is undone after it."""
    for key in WANDB_VARIABLES:
        monkeypatch.setenv(key, "disabled")  # recorded first: the value before the test, or its absence
        if key != "WANDB_MODE":
            monkeypatch.delenv(key)


class _FakeWandb:
    """``wandb`` as HF's WandbCallback and the trainer's marker call it, so that no test creates a W&B run."""

    class Config(dict):
        def update(self, values, allow_val_change=False):
            super().update(values)

    def __init__(self, mode: str = "online", refuse: bool = False) -> None:
        self.inits: list[dict] = []
        self.logged: list[dict] = []
        self.run = None
        self.config = self.Config()
        self.mode, self.refuse = mode, refuse

    def init(self, **kwargs):
        env = {key: os.environ.get(key) for key in WANDB_VARIABLES}
        self.inits.append({"kwargs": kwargs, "env": env})
        run_id = kwargs.get("id", env["WANDB_RUN_ID"])
        if self.refuse:
            logs = Path(env["WANDB_DIR"]) / "wandb" / f"run-20261005_000000-{run_id}" / "logs"
            logs.mkdir(parents=True)
            (logs / "debug.log").write_text(f"run {run_id} already exists\n")
            raise RuntimeError(f"run {run_id} already exists (resume='never')")
        self.run = types.SimpleNamespace(
            summary={}, _label=lambda **labels: None, id=run_id, project=kwargs.get("project"),
            entity=env["WANDB_ENTITY"] or "key-default", offline=self.mode == "offline",
            disabled=self.mode == "disabled")
        return self.run

    def define_metric(self, *args, **kwargs):
        pass

    def termwarn(self, *args, **kwargs):
        pass

    def log(self, values):
        self.logged.append(dict(values))


def test_a_wandb_run_id_is_the_run_name_where_wandb_takes_it_else_a_stable_sanitized_one():
    import wandb

    assert tn.wandb_run_id("gemma-grpo-mix-2026-10-04") == "gemma-grpo-mix-2026-10-04"
    assert tn.wandb_run_id("a" * 64) == "a" * 64
    odd = ["grpo run:a/b", "a" * 65, "ü", "a;b", "a,b", "a#b?"]
    ids = [tn.wandb_run_id(name) for name in odd]
    assert ids == [tn.wandb_run_id(name) for name in odd] and len(set(ids)) == len(odd)

    def digest(name):
        return hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]

    assert ids[:3] == [f"grpo-run-a-b-{digest(odd[0])}", f"{'a' * 55}-{digest(odd[1])}", digest("ü")]
    for run_id in ids:
        assert len(run_id) <= tn.WANDB_ID_MAX and tn.WANDB_ID_KEPT.fullmatch(run_id)
        assert wandb.Settings(run_id=run_id).run_id == run_id
    with pytest.raises(wandb.errors.UsageError):
        wandb.Settings(run_id="a/b")


def test_a_run_reporting_to_wandb_names_its_project_entity_group_id_and_directory(monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", "/artifacts")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    cfg = _compose(["run_name=gemma-grpo-mix-2026-10-04", "model.path=/m"])
    out = Path(cfg["output_dir"])

    def env(settings=cfg["wandb"], report_to=cfg["grpo"]["report_to"], resume="never"):
        return tn.wandb_env(settings, output_dir=out, report_to=report_to, resume=resume)

    assert env() == {"WANDB_PROJECT": "ruleforge", "WANDB_RUN_GROUP": "grpo", "WANDB_RUN_ID": "gemma-grpo-mix-2026-10-04",
                     "WANDB_RESUME": "never", "WANDB_DIR": "/artifacts/training/gemma-grpo-mix-2026-10-04",
                     "WANDB_SYMLINK": "false"}
    assert env(resume="allow") == {**env(), "WANDB_RESUME": "allow"}
    assert env({**cfg["wandb"], "entity": "a-team"})["WANDB_ENTITY"] == "a-team"
    assert env(report_to="all") == env() == env(report_to="wandb")
    for report_to in ([], "none", ["none"], None):
        assert env(report_to=report_to) == {}
    with pytest.raises(tn.TrainerConfigError, match="wandb block sets no group"):
        env({"project": "ruleforge"})
    with pytest.raises(tn.TrainerConfigError, match="wandb block sets no project, group"):
        env(None)
    with pytest.raises(tn.TrainerConfigError, match="W&B refuses the run's settings .*cannot contain"):
        env({**cfg["wandb"], "project": "ruleforge:v2"})


def test_a_launch_appends_to_its_wandb_run_only_on_evidence_that_the_run_opened_it(tmp_path):
    import wandb

    def resume(output_dir, resume_from=None):
        return tn.wandb_resume(output_dir, resume_from, project="ruleforge", entity=None)

    out = tmp_path / "training" / "r"
    assert resume(out) == "never"  # nothing there yet
    (out / "hydra" / "launch-1").mkdir(parents=True)  # a preflight's and Hydra's files: still fresh
    (out / "STAGE.json").write_text("{}")
    assert resume(out) == "never"
    logs = out / "wandb" / "run-20261005_000000-r" / "logs"
    logs.mkdir(parents=True)
    (logs / "debug.log").write_text("run r already exists\n")
    (out / "code_snapshots").mkdir()
    assert resume(out) == "never"
    assert resume(out, tmp_path / "training" / "other" / "checkpoint-100") == "never"
    marker = out / tn.WANDB_RUN_FNAME
    for text in (json.dumps({"kind": tn.WANDB_RUN_KIND, "id": "other", "project": "ruleforge"}),
                 json.dumps({"id": "r", "project": "ruleforge"}), "[]", "{"):
        marker.write_text(text)
        assert resume(out) == "never"
    marker.write_text(json.dumps({"kind": tn.WANDB_RUN_KIND, "id": "r", "project": "ruleforge", "entity": "a-team"}))
    assert resume(out) == "allow"
    requeued = tmp_path / "training" / "q"
    (requeued / "checkpoint-200").mkdir(parents=True)
    assert resume(requeued) == resume(requeued, requeued / "checkpoint-200") == "allow"
    assert resume(tmp_path / "moved" / "q", requeued / "checkpoint-200") == "allow"
    for value in ("never", "allow"):  # what W&B's init reads from the environment the trainer exports
        settings = wandb.Settings()
        settings.update_from_env_vars({"WANDB_RUN_ID": "r", "WANDB_RESUME": value})
        assert (settings.run_id, settings.resume) == ("r", value)


def test_a_marker_counts_only_for_its_own_wandb_project_and_entity(tmp_path):
    out = tmp_path / "training" / "r"
    out.mkdir(parents=True)
    marker = out / tn.WANDB_RUN_FNAME
    marker.write_text(json.dumps({"kind": tn.WANDB_RUN_KIND, "id": "r", "project": "ruleforge-old", "entity": "a-team"}))
    for project, entity in (("ruleforge", None), ("ruleforge", "a-team"), (None, None)):
        assert tn.wandb_marker_id(out, project=project, entity=entity) is None
        assert tn.wandb_resume(out, None, project=project, entity=entity) == "never"
    assert tn.wandb_marker_id(out, project="ruleforge-old", entity=None) == "r"
    assert tn.wandb_resume(out, None, project="ruleforge-old", entity="a-team") == "allow"
    assert tn.wandb_resume(out, None, project="ruleforge-old", entity="b-team") == "never"
    marker.write_text(json.dumps({"kind": tn.WANDB_RUN_KIND, "id": "r", "entity": "a-team"}))
    assert tn.wandb_resume(out, None, project="ruleforge", entity=None) == "never"
    (out / "checkpoint-100").mkdir()
    assert tn.wandb_resume(out, None, project="ruleforge", entity="b-team") == "allow"


def test_rank_0_records_the_wandb_run_only_once_wandb_has_accepted_it(tmp_path):
    from transformers import TrainerState
    from wandb.sdk.wandb_run import Run

    assert all(isinstance(getattr(Run, name), property) for name in ("id", "project", "entity", "offline", "disabled"))
    out = tmp_path / "r"
    marker = out / tn.WANDB_RUN_FNAME
    callback = tn.anchored_grpo_trainer_class().wandb_run_marker(out)
    zero, other, control = TrainerState(is_world_process_zero=True), TrainerState(is_world_process_zero=False), None
    fake = callback._wandb = _FakeWandb()
    callback.on_train_begin(None, zero, control)  # no run open: HF's callback opened none
    for mode in ("offline", "disabled"):  # a run W&B never saw proves nothing about the id there
        fake.mode = mode
        fake.init(project="ruleforge", id="r")
        callback.on_train_begin(None, zero, control)
    assert not out.exists()
    fake.mode = "online"
    fake.init(project="ruleforge", id="r")
    callback.on_train_begin(None, other, control)
    assert not out.exists()
    callback.on_train_begin(None, zero, control)
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record == {"kind": tn.WANDB_RUN_KIND, "id": "r", "project": "ruleforge", "entity": "key-default",
                      "opened_at": record["opened_at"]} and record["opened_at"].endswith("+00:00")
    assert [path.name for path in out.iterdir()] == [tn.WANDB_RUN_FNAME]
    assert tn.wandb_marker_id(out, project="ruleforge", entity=None) == "r"
    assert tn.wandb_resume(out, None, project="ruleforge", entity="key-default") == "allow"
    assert tn.write_wandb_marker(out, None, at="t") is None


def test_tensorboard_is_refused_and_gone_from_the_stack(monkeypatch):
    grpo = _base_grpo(monkeypatch)
    for report_to in (["tensorboard"], "tensorboard", ["wandb", "tensorboard"]):
        with pytest.raises(tn.TrainerConfigError, match="names tensorboard, which is not supported"):
            _kwargs({**grpo, "report_to": report_to})
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    named = [*project["project"]["dependencies"],
             *(item for group in project["dependency-groups"].values() for item in group if isinstance(item, str))]
    assert named and not [item for item in named if item.lower().startswith("tensorboard")]
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"] for package in lock["package"]}
    assert "trl" in locked and not locked & {"tensorboard", "tensorboard-data-server", "tensorboardx"}


def test_every_rank_exports_the_settings_and_only_rank_0_creates_the_wandb_run():
    from transformers import TrainerControl, TrainerState
    from transformers.integrations import WandbCallback

    settings = {"WANDB_PROJECT": "ruleforge", "WANDB_RUN_ID": "r", "WANDB_RESUME": "allow"}
    environ = {"WANDB_MODE": "offline"}
    assert tn.export_wandb_env(environ, settings, rank=0) == settings
    assert environ == {**settings, "WANDB_MODE": "offline"}
    other: dict = {}
    assert tn.export_wandb_env(other, settings, rank=1) == {**settings, "WANDB_MODE": "disabled"} == other
    untouched: dict = {}
    assert tn.export_wandb_env(untouched, {}, rank=1) == {} == untouched  # a run that does not report to W&B
    # HF's own callback: the world's process zero creates the run and logs to it, any other rank neither
    args = types.SimpleNamespace(run_name="r", output_dir="/out/r", to_dict=lambda: {"seed": 0})
    for zero in (True, False):
        fake = _FakeWandb()
        callback = WandbCallback()
        callback._wandb = fake
        state = TrainerState(is_world_process_zero=zero, global_step=1)
        callback.on_train_begin(args, state, TrainerControl(), model=None)
        callback.on_log(args, state, TrainerControl(), logs={"consistency/yes_rate_favoured_no/num": 0.0})
        assert (len(fake.inits), len(fake.logged)) == ((1, 1) if zero else (0, 0))
    assert fake.inits == [] and callback._initialized


def test_a_run_and_its_resume_report_to_one_wandb_run_from_rank_0(run_files, tiny_model, monkeypatch):
    from transformers.integrations import WandbCallback

    _isolated_wandb_env(monkeypatch)
    wandb_on = "grpo.report_to=[wandb]"
    assert _plan(run_files, tiny_model, monkeypatch, wandb_on, "resume_from_checkpoint=latest").wandb_env[
        "WANDB_RESUME"] == "never"
    plan = _plan(run_files, tiny_model, monkeypatch, wandb_on)
    expected = {"WANDB_PROJECT": "ruleforge", "WANDB_RUN_GROUP": "grpo", "WANDB_RUN_ID": "dry-run",
                "WANDB_RESUME": "never", "WANDB_DIR": str(plan.output_dir), "WANDB_SYMLINK": "false"}
    assert plan.wandb_env == expected and plan.grpo_kwargs["run_name"] == "dry-run"
    assert tn.run_grpo(_config(run_files, tiny_model, monkeypatch, wandb_on, "preflight_only=true"),
                       reward_factory=RewardDouble, marker_checker=marker_double) is None
    assert not {key: os.environ[key] for key in expected if key in os.environ}
    trainer = tn.build_trainer(plan, grpo_overrides=CPU)
    assert {key: os.environ[key] for key in expected} == expected
    callbacks = trainer.callback_handler.callbacks
    [callback] = [item for item in callbacks if isinstance(item, WandbCallback)]
    [marker] = [item for item in callbacks if isinstance(item, trainer.wandb_run_marker)]
    assert callbacks.index(callback) < callbacks.index(marker)  # the run opens before rank 0 records it
    # W&B refuses the run (its id has another run's W&B run): wandb.init wrote wandb/run-*, then failed
    fake = callback._wandb = marker._wandb = _FakeWandb(refuse=True)
    with pytest.raises(RuntimeError, match="already exists"):
        trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)
    assert any((plan.output_dir / "wandb").glob("run-*-dry-run/logs/debug.log"))
    assert not (plan.output_dir / tn.WANDB_RUN_FNAME).exists()
    for again in ((), ("resume_from_checkpoint=latest",)):
        assert _plan(run_files, tiny_model, monkeypatch, wandb_on, *again).wandb_env == expected
    fake.refuse, callback._initialized = False, False
    trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)
    opened = fake.inits[-1]
    assert opened["kwargs"] == {"project": "ruleforge", "name": "dry-run"}
    assert opened["env"] == {**expected, "WANDB_ENTITY": None, "WANDB_MODE": "disabled"}  # conftest's mode
    assert tn.wandb_marker_id(plan.output_dir, project="ruleforge", entity=None) == "dry-run"
    appended = {**expected, "WANDB_RESUME": "allow"}
    for again in ((), ("resume_from_checkpoint=latest",)):
        relaunched = _plan(run_files, tiny_model, monkeypatch, wandb_on, *again)
        assert relaunched.resume_from is None and relaunched.wandb_env == appended
    checkpoint = plan.output_dir / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    tn.write_provenance(checkpoint, {"step": 1, "run": {"identity": plan.provenance["identity"],
                                                        "config": plan.provenance["config"]}})
    (plan.output_dir / tn.WANDB_RUN_FNAME).unlink()
    for resume in ("latest", str(checkpoint)):
        resumed = _plan(run_files, tiny_model, monkeypatch, wandb_on, f"resume_from_checkpoint={resume}")
        assert resumed.resume_from == checkpoint and resumed.wandb_env == appended
    for again in ((), (f"resume_from_checkpoint={checkpoint}",)):
        other = _plan(run_files, tiny_model, monkeypatch, wandb_on, "run_name=dry-run-2", *again)
        assert (other.wandb_env["WANDB_RUN_ID"], other.wandb_env["WANDB_RESUME"]) == ("dry-run-2", "never")


class _TableWandb:
    """``wandb`` as TRL 1.13's ``GRPOTrainer.log`` calls it for the completions table (its module's own ``wandb``)."""

    def __init__(self) -> None:
        self.run = types.SimpleNamespace(id="dry-run")
        self.logged: list[dict] = []

    def Table(self, dataframe):  # wandb's name
        return ("table", dataframe.to_dict("records"))

    def log(self, values):
        self.logged.append(dict(values))


def _logged_step(trainer, step: int) -> None:
    """One optimizer step's log, as TRL's training loop calls it: two completions of one prompt in TRL's buffers."""
    trainer.state.global_step = step
    trainer._logs["prompt"].extend(["p", "p"])
    trainer._logs["completion"].extend([f"c{step}-0", f"c{step}-1"])
    trainer._logs["rewards"]["smrc"].extend([1.0, 0.0])
    trainer._logs["advantages"].extend([0.5, -0.5])
    trainer.log({"loss": 0.25})


def test_the_completions_table_reaches_wandb_only_when_the_run_asks_and_the_parquet_is_always_written(
        run_files, tiny_model, monkeypatch):
    import pandas as pd
    import trl.trainer.grpo_trainer as grpo_trainer
    from transformers.integrations import WandbCallback

    _isolated_wandb_env(monkeypatch)
    wandb_on = "grpo.report_to=[wandb]"
    assert _plan(run_files, tiny_model, monkeypatch, wandb_on).wandb_completions_table is False
    with pytest.raises(tn.TrainerConfigError, match="log_completions_table is true or false"):
        _plan(run_files, tiny_model, monkeypatch, wandb_on, "wandb.log_completions_table=sometimes")
    plan = _plan(run_files, tiny_model, monkeypatch, wandb_on, "wandb.log_completions_table=true")
    assert plan.wandb_completions_table is True and plan.grpo_kwargs["log_completions"] is True
    trainer = tn.build_trainer(plan, grpo_overrides=CPU)
    assert trainer.wandb_completions_table is True
    [callback] = [item for item in trainer.callback_handler.callbacks if isinstance(item, WandbCallback)]
    metrics = callback._wandb = _FakeWandb()
    for item in trainer.callback_handler.callbacks:
        if isinstance(item, trainer.wandb_run_marker):
            item._wandb = metrics
    trainer.callback_handler.on_train_begin(trainer.args, trainer.state, trainer.control)
    table = _TableWandb()
    monkeypatch.setattr(grpo_trainer, "wandb", table)
    completions = plan.output_dir / "completions"

    _logged_step(trainer, 1)
    assert [list(item) for item in table.logged] == [["completions"]]
    assert [row["completion"] for row in table.logged[0]["completions"][1]] == ["c1-0", "c1-1"]
    trainer.wandb_completions_table = False
    _logged_step(trainer, 2)
    assert len(table.logged) == 1
    for step in (1, 2):
        rows = pd.read_parquet(completions / f"completions_{step:05d}.parquet").to_dict("records")
        assert [(row["step"], row["completion"]) for row in rows][-2:] == [(step, f"c{step}-0"), (step, f"c{step}-1")]
    assert [entry["train/global_step"] for entry in metrics.logged] == [1, 2]
    assert trainer.args.report_to == ["wandb"]
    trainer.args.report_to = []  # a trainer that reports to no W&B run: TRL's log as it is
    _logged_step(trainer, 3)
    assert (completions / "completions_00003.parquet").is_file() and len(table.logged) == 1
    assert trainer.args.report_to == [] and len(metrics.logged) == 3
    trainer.args.report_to = ["wandb"]

    def refused(*args, **kwargs):
        raise RuntimeError("a callback failed")

    monkeypatch.setattr(trainer.callback_handler, "on_log", refused)
    with pytest.raises(RuntimeError, match="a callback failed"):
        _logged_step(trainer, 4)
    assert trainer.args.report_to == ["wandb"]  # TRL's again, after an exception inside the log


def test_the_wandb_line_names_the_run_and_its_mode_and_warns_unless_online(run_files, tiny_model, monkeypatch,
                                                                          caplog):
    config = _config(run_files, tiny_model, monkeypatch, "grpo.report_to=[wandb]", "preflight_only=true")
    out = config["output_dir"]
    for mode in ("disabled", "offline", None):
        if mode is None:
            monkeypatch.delenv("WANDB_MODE")  # nothing below opens a run: a preflight stops before the trainer
        else:
            monkeypatch.setenv("WANDB_MODE", mode)
        caplog.clear()
        with caplog.at_level("INFO", logger=tn.__name__):
            assert tn.run_grpo(config, reward_factory=RewardDouble, marker_checker=marker_double) is None
        lines = {record.getMessage(): record.levelname for record in caplog.records if "W&B" in record.getMessage()}
        named = (f"W&B: run dry-run in project ruleforge, group grpo, mode {mode or 'online'} (resume never; local "
                 f"files under {out}/wandb)")
        warned = (f"W&B mode is {mode} (WANDB_MODE), not online: offline keeps the run under {out}/wandb until "
                  "`wandb sync`; disabled logs nothing")
        assert lines == ({named: "INFO"} if mode is None else {named: "INFO", warned: "WARNING"})


EARLIER_LOCK = "150c06cb864a86be48f3c877cd3d753a6e6253cdbc5a8bf121bb385834eb6913"


def test_a_checkpoint_of_an_older_reporter_is_refused_naming_its_reporter_and_its_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    base = ["run_name=r", "model.path=/m"]
    old, new = _compose([*base, "grpo.report_to=[tensorboard]"]), _compose(base)
    del old["wandb"]
    preset, truncation = tn.lora_preset("Qwen3ForCausalLM"), Truncation(temperature=1.0, top_k=20, top_p=0.95)
    stack = {"trl": "1.13.0", "uv_lock_sha256": "f" * 64}

    def identity(config, lock=stack):
        return tn.run_identity(config=config, digests={"train": "d"}, preset=preset, truncation=truncation, stack=lock)

    assert identity({**new, "wandb": {"project": "elsewhere"}}) == identity(new)
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    tn.write_provenance(checkpoint, {"step": 100, "run": {
        "identity": identity(old, {**stack, "uv_lock_sha256": EARLIER_LOCK}), "config": old}})
    with pytest.raises(tn.TrainerConfigError) as refused:
        tn.check_resume(checkpoint, identity(new), max_steps=600, config=new)
    assert str(refused.value) == (
        f"{checkpoint} was trained under another config_sha256, stack; a resume continues the same run or none "
        "(recorded -> this run: stack.uv_lock_sha256 150c06cb864a... -> ffffffffffff...; config grpo.report_to "
        "['tensorboard'] -> ['wandb']). Resume it under the code, lock and config that trained it, or train under "
        "another run_name")
    many = {**new, "grpo": {**new["grpo"], **{f"extra_{index}": index for index in range(9)}}}
    with pytest.raises(tn.TrainerConfigError, match=r"config grpo.extra_0 unset -> 0; .*; 3 more\)"):
        tn.check_resume(checkpoint, identity(many), max_steps=600, config=many)


def test_a_checkpoint_trained_before_the_program_moved_resumes_under_the_rules_path_from_the_checkout_s_top(
        tmp_path, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    new = _compose(["run_name=r", "model.path=/m"])
    assert new["data"]["rules"] == "data/rules/modelspec/openai_rules_all.jsonl"
    old = {**new, "data": {**new["data"], "rules": "../../" + new["data"]["rules"]}}
    preset, truncation = tn.lora_preset("Qwen3ForCausalLM"), Truncation(temperature=1.0, top_k=20, top_p=0.95)

    def identity(config):
        return tn.run_identity(config=config, digests={"rules_sha256": "r"}, preset=preset, truncation=truncation,
                               stack={"trl": "1.13.0"})

    assert identity(old)["config_sha256"] != identity(new)["config_sha256"]
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    tn.write_provenance(checkpoint, {"step": 100, "run": {"identity": identity(old), "config": old}})
    assert tn.check_resume(checkpoint, identity(new), max_steps=600, config=new)["step"] == 100
    assert tn.respelled_only(new, old) and tn.respelled_only(old, old)
    for other in ({**new, "data": {**new["data"], "rules": "data/rules/modelspec/other.jsonl"}},
                  {**new, "grpo": {**new["grpo"], "max_steps": 700}}):
        with pytest.raises(tn.TrainerConfigError, match="was trained under another config_sha256; .*config (data.rules|"
                                                        "grpo.max_steps)"):
            tn.check_resume(checkpoint, identity(other), max_steps=600, config=other)
    # the digest alone differing, this run's config not given (or not recorded): nothing to read the spelling from
    with pytest.raises(tn.TrainerConfigError, match="was trained under another config_sha256"):
        tn.check_resume(checkpoint, identity(new), max_steps=600)
    # another identity entry differing besides the digest: refused, whatever the spelling
    with pytest.raises(tn.TrainerConfigError, match="another config_sha256, stack"):
        tn.check_resume(checkpoint, {**identity(new), "stack": {"trl": "1.14.0"}}, max_steps=600, config=new)
    bare = {key: value for key, value in new.items() if key != "data"}
    assert tn.respelled_only(bare, bare) and not tn.respelled_only({**bare, "data": {"rules": None}}, new)


def test_a_checkpoint_recorded_with_a_retired_key_resumes_without_it(tmp_path, monkeypatch):
    """A published run's config and identity hold a `prereg` key and digest."""
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    new = _compose(["run_name=r", "model.path=/m"])
    assert "prereg" not in new
    old = {**new, "prereg": {"path": "/s/prereg/prereg.json", "sha256": "a" * 64}}
    preset, truncation = tn.lora_preset("Qwen3ForCausalLM"), Truncation(temperature=1.0, top_k=20, top_p=0.95)

    def identity(config, **digests):
        return tn.run_identity(config=config, digests={"rules_sha256": "r", **digests}, preset=preset,
                               truncation=truncation, stack={"trl": "1.13.0"})

    recorded = identity(old, prereg_sha256="a" * 64)
    assert recorded["config_sha256"] == canonical_digest(tn.resume_free(old))
    assert recorded["config_sha256"] != identity(new)["config_sha256"]
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    tn.write_provenance(checkpoint, {"step": 100, "run": {"identity": recorded, "config": old}})
    assert tn.check_resume(checkpoint, identity(new), max_steps=600, config=new)["step"] == 100
    assert tn.respelled_only(old, new) and tn.respelled_only(new, old)
    other = {**new, "grpo": {**new["grpo"], "max_steps": 700}}
    with pytest.raises(tn.TrainerConfigError) as refused:
        tn.check_resume(checkpoint, identity(other), max_steps=600, config=other)
    assert "config grpo.max_steps 600 -> 700" in str(refused.value) and "prereg" not in str(refused.value)
    # the digest alone differing and no config to read: nothing says it differs by the `prereg` key alone
    with pytest.raises(tn.TrainerConfigError, match="another config_sha256"):
        tn.check_resume(checkpoint, identity(new), max_steps=600)
    # only the `prereg` digest is dropped: any other digest still binds the resume
    with pytest.raises(tn.TrainerConfigError, match="another config_sha256, digests"):
        tn.check_resume(checkpoint, identity(new, rules_sha256="x"), max_steps=600, config=new)
    # a record whose digests are absent compares as it is
    bare = tmp_path / "checkpoint-200"
    bare.mkdir()
    tn.write_provenance(bare, {"step": 200, "run": {"identity": {"config_sha256": "c"}}})
    assert tn.check_resume(bare, {"config_sha256": "c"}, max_steps=600)["step"] == 200


def test_a_same_stack_resume_through_run_grpo_takes_a_save_recorded_with_the_retired_key(run_files, tiny_model,
                                                                                         monkeypatch):
    monkeypatch.setattr("trl.trainer.grpo_trainer.VLLMGeneration", _fake_generation_class())
    with monkeypatch.context() as patch:
        _with_callbacks(patch, _StopAt(2))
        trainer = tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU), reward_factory=RewardDouble,
                              marker_checker=marker_double)
    checkpoint = Path(trainer.args.output_dir) / "checkpoint-2"
    record = json.loads((checkpoint / tn.PROVENANCE_FNAME).read_text())
    run = record["run"]
    run["config"]["prereg"] = {"path": "/s/prereg/prereg.json", "sha256": "a" * 64}
    run["identity"]["digests"]["prereg_sha256"] = "a" * 64
    run["identity"]["config_sha256"] = canonical_digest(tn.resume_free(run["config"]))
    (checkpoint / tn.PROVENANCE_FNAME).write_text(json.dumps(record))
    resumed = tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU, "resume_from_checkpoint=latest"),
                          reward_factory=RewardDouble, marker_checker=marker_double)
    assert resumed.state.global_step == 3 and resumed.anchor_manager.start_step == 2
    final = json.loads((Path(trainer.args.output_dir) / "checkpoint-3" / tn.PROVENANCE_FNAME).read_text())["run"]
    assert "prereg" not in final["config"] and "prereg_sha256" not in final["identity"]["digests"]


def test_the_default_probe_reads_a_packed_qkv_shard_a_packed_gate_up_shard_and_an_unpacked_module():
    assert tn.default_probe_modules(36) == [
        "model.layers.0.self_attn.k_proj", "model.layers.18.mlp.up_proj", "model.layers.35.mlp.down_proj"]


def test_the_plan_resolves_and_checks_everything_without_a_gpu(run_files, tiny_model, monkeypatch):
    plan = _plan(run_files, tiny_model, monkeypatch, "grpo.beta=0.04")
    assert plan.truncation == Truncation(temperature=1.0, top_k=20, top_p=0.95)
    assert plan.forced == {"beta": [0.04, 0.0]} and plan.grpo_kwargs["beta"] == 0.0
    assert len(plan.train_rows) == 8 and len(plan.replay) == 6 and plan.resume_from is None
    assert plan.provenance["replay"] == {"source": str(run_files["replay"]), "sha256": sha256_file(run_files["replay"]),
                                         "rows": 6, "weight_scale": pytest.approx(6 / 2.5)}
    assert "prereg_sha256" not in plan.provenance["identity"]["digests"]
    assert plan.probe_plain_modules == ["model.norm"]
    assert plan.anchor.accum == 4 and plan.anchor.config.loss_weight == 1.0
    assert plan.probe_modules == tn.default_probe_modules(2)
    assert isinstance(plan.reward_set, RewardDouble) and plan.reward_set.num_generations == 2
    markers = plan.provenance["identity"]["digests"]["markers"]
    assert set(markers) == {"train", "replay"} and markers["train"]["dataset_sha256"] == sha256_file(run_files["train"])
    assert plan.provenance["prompt_tokens"] == {"rows": 8, "max": 14, "cap": 64}
    assert plan.provenance["preset_expected_modules"] == 14
    assert plan.grpo_kwargs["vllm_max_model_length"] == 12 + 5120
    assert tn.run_grpo(_config(run_files, tiny_model, monkeypatch, "preflight_only=true"),
                       reward_factory=RewardDouble, marker_checker=marker_double) is None


def test_the_plan_refuses_a_bad_run(run_files, tiny_model, monkeypatch):
    with pytest.raises(tn.TrainerConfigError, match="not the pinned"):
        _plan(run_files, tiny_model, monkeypatch, "data.rules_sha256=" + "0" * 64)
    with pytest.raises(tn.TrainerConfigError, match="exceed the 12-token cap"):
        _plan(run_files, tiny_model, monkeypatch, "data.prompt_token_cap=12")
    with pytest.raises(tn.TrainerConfigError, match="anchor_accum must be positive"):
        _plan(run_files, tiny_model, monkeypatch, "anchor.anchor_accum=0")
    with pytest.raises(tn.TrainerConfigError, match="positive loss_weight, got 0.0"):
        _plan(run_files, tiny_model, monkeypatch, "anchor.loss_weight=0")
    with pytest.raises(tn.TrainerConfigError, match="straddle"):
        _plan(run_files, tiny_model, monkeypatch, "grpo.num_generations=3")
    with pytest.raises(tn.TrainerConfigError, match="exceeds vllm_length_margin 32"):
        _plan(run_files, tiny_model, monkeypatch, "vllm_length_margin=32")
    with pytest.raises(tn.TrainerConfigError, match="the reward ends a completion at"):
        _plan(run_files, tiny_model, monkeypatch, reward_factory=type("Other", (RewardDouble,), {"finish_token_ids": (EOS,)}))
    checked = tn.require_disjoint_marker

    def then_changed(dataset, **kwargs):  # the replay file changes between its marker check and its index
        record = checked(dataset, **kwargs)
        if Path(dataset) == run_files["replay"]:  # (a changed train file is refused by its build manifest's digest)
            Path(dataset).write_text(Path(dataset).read_text() + "\n")
        return record

    with monkeypatch.context() as patch:
        patch.setattr(tn, "require_disjoint_marker", then_changed)
        with pytest.raises(ValueError, match="replay.jsonl has sha256 .* the file changed in between"):
            _plan(run_files, tiny_model, monkeypatch)
    disjoint_marker_path(run_files["replay"]).unlink()
    with pytest.raises(LeakageError, match="replay.jsonl carries no disjointness marker"):
        _plan(run_files, tiny_model, monkeypatch)
    plan = _plan(run_files, tiny_model, monkeypatch, "anchor.enabled=false",
                 "sync_probe.modules=[model.layers.1.self_attn.q_proj]")
    assert plan.anchor is None and plan.replay is None and plan.forced == {}
    assert plan.probe_modules == ["model.layers.1.self_attn.q_proj"]
    assert set(plan.provenance["rows_model"]) == {"train"}


def test_the_plan_refuses_rows_built_on_another_model(run_files, tiny_model, monkeypatch):
    plan = _plan(run_files, tiny_model, monkeypatch)
    rows_model = plan.provenance["rows_model"]
    assert {name: entry["output"] for name, entry in rows_model.items()} == {
        "train": "grpo_rows.jsonl", "replay": "replay_rows.jsonl"}
    assert rows_model["train"]["built_on"] == str(tiny_model) and "rows_model" not in plan.provenance["identity"]
    manifest = json.loads(run_files["manifest"].read_text())
    other = {**manifest, "tokenizer": {**manifest["tokenizer"], "name": "gemma-4-31B-it", "model": "/m/gemma-4-31B-it"}}
    run_files["manifest"].write_text(json.dumps(other))
    with pytest.raises(tn.TrainerConfigError, match="was built on /m/gemma-4-31B-it, not on Qwen3-tiny"):
        _plan(run_files, tiny_model, monkeypatch)
    replay_only = {**manifest, "outputs": {**manifest["outputs"], "replay_rows.jsonl": {"sha256": "0" * 64}}}
    run_files["manifest"].write_text(json.dumps(replay_only))
    with pytest.raises(tn.TrainerConfigError, match="replay.jsonl is neither the grpo_rows.jsonl nor"):
        _plan(run_files, tiny_model, monkeypatch)
    assert _plan(run_files, tiny_model, monkeypatch, "anchor.enabled=false").replay is None
    run_files["manifest"].unlink()
    with pytest.raises(tn.TrainerConfigError, match="has no grpo_data_manifest.json beside it"):
        _plan(run_files, tiny_model, monkeypatch)


def test_the_rows_marker_must_cover_the_files_their_build_was_held_to(run_files, tiny_model, monkeypatch):
    pinned, required, added = "p" * 64, "r" * 64, "a" * 64
    manifest = json.loads(run_files["manifest"].read_text())
    run_files["manifest"].write_text(json.dumps({
        **manifest, "pinned_protected": [{"name": "frozen", "sha256": pinned}],
        "required_protected": [{"path": "dev.jsonl", "sha256": required}],
        "protected_extra": [{"path": "dev_nonyes.jsonl", "sha256": added}]}))
    assert gd.recorded_protection(run_files["train"]) == {"required_protected": [pinned, required],
                                                          "required_carved": [added]}
    seen = []

    def checker(dataset, marker_path, **requirements):
        seen.append((Path(dataset).name, requirements))
        marker = json.loads(Path(marker_path).read_text())
        covered = {entry["sha256"] for entry in marker.get("protected", ())}
        read = covered | {entry["sha256"] for entry in marker.get("carved", ())}
        if not (set(requirements.get("required_protected", ())) <= covered
                and set(requirements.get("required_carved", ())) <= read):
            raise LeakageError(f"{marker_path} was not checked against the required protected file(s)")
        return marker

    with pytest.raises(LeakageError, match="grpo_train.jsonl.disjoint.json was not checked against the required"):
        _plan(run_files, tiny_model, monkeypatch, marker_checker=checker)
    for dataset in (run_files["train"], run_files["replay"]):
        disjoint_marker_path(dataset).write_text(json.dumps({
            "kind": "prompt_disjoint", "protected": [{"sha256": pinned}, {"sha256": required}],
            "carved": [{"sha256": added}]}))
    _plan(run_files, tiny_model, monkeypatch, marker_checker=checker)
    requirements = {"required_protected": [pinned, required], "required_carved": [added]}
    assert seen[-2:] == [("grpo_train.jsonl", requirements), ("replay.jsonl", requirements)]
    run_files["manifest"].unlink()
    assert gd.recorded_protection(run_files["train"]) == {}


def test_the_run_is_bound_to_the_base_weights_it_trains_and_its_rows_were_built_on(run_files, tiny_model, monkeypatch):
    from ruleforge.judges.weights import weights_digest

    plan = _plan(run_files, tiny_model, monkeypatch)
    identity = plan.provenance["identity"]
    assert identity["digests"]["model_weights_sha256"] == weights_digest(tiny_model)
    other = json.loads(json.dumps(identity))
    other["digests"]["model_weights_sha256"] = "0" * 64
    checkpoint = run_files["root"] / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    tn.write_provenance(checkpoint, {"step": 1, "run": {"identity": other}})
    with pytest.raises(tn.TrainerConfigError, match="digests.model_weights_sha256 000000000000... -> "):
        tn.check_resume(checkpoint, identity, max_steps=3)
    manifest = json.loads(run_files["manifest"].read_text())
    run_files["manifest"].write_text(json.dumps({**manifest, "base_weights_sha256": "0" * 64}))
    with pytest.raises(tn.TrainerConfigError, match="was built on base weights 000000000000"):
        _plan(run_files, tiny_model, monkeypatch)


def test_a_base_whose_weights_are_not_digested_is_refused_before_training(run_files, tiny_model, monkeypatch, tmp_path):
    base = tmp_path / "bin" / tiny_model.name
    shutil.copytree(tiny_model, base)
    (base / "model.safetensors").rename(base / "pytorch_model.bin")
    with pytest.raises(tn.TrainerConfigError, match=r"holds weights in a format .*pytorch_model.bin"):
        _plan(run_files, base, monkeypatch)
    (base / "pytorch_model.bin").unlink()
    with pytest.raises(tn.TrainerConfigError, match="holds no safetensors weights: a run is bound to its base weights"):
        _plan(run_files, base, monkeypatch)


def test_a_checkpoint_recorded_before_the_weights_digest_resumes_with_its_base_weights_unchecked(
        run_files, tiny_model, monkeypatch, caplog):
    identity = _plan(run_files, tiny_model, monkeypatch).provenance["identity"]
    older = json.loads(json.dumps(identity))
    del older["digests"]["model_weights_sha256"]
    checkpoint = run_files["root"] / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    tn.write_provenance(checkpoint, {"step": 1, "run": {"identity": older}})
    with caplog.at_level("WARNING", logger=tn.logger.name):
        assert tn.check_resume(checkpoint, identity, max_steps=3)["run"]["identity"] == older
    assert "records no base weights digest: the base weights are unchecked for this resume" in caplog.text
    older["digests"]["rules_sha256"] = "0" * 64
    tn.write_provenance(checkpoint, {"step": 1, "run": {"identity": older}})
    with pytest.raises(tn.TrainerConfigError, match="digests.rules_sha256"):
        tn.check_resume(checkpoint, identity, max_steps=3)


def test_the_plan_renders_the_prompts_as_the_rows_build_verified_them(run_files, tiny_model, monkeypatch):
    """The rows build checked TRL's rendering against the campaign's under the chat-template arguments it records."""
    plan = _plan(run_files, tiny_model, monkeypatch)
    assert plan.provenance["rows_model"]["train"]["chat_template_kwargs"] == {"enable_thinking": True}
    with pytest.raises(tn.TrainerConfigError, match=r"chat_template_kwargs \{'enable_thinking': False\}, and the rows "
                                                     r"were built and checked under \{'enable_thinking': True\}"):
        _plan(run_files, tiny_model, monkeypatch, "grpo.chat_template_kwargs.enable_thinking=false")
    manifest = json.loads(run_files["manifest"].read_text())
    run_files["manifest"].write_text(json.dumps({key: value for key, value in manifest.items() if key != "rendering"}))
    with pytest.raises(tn.TrainerConfigError, match=r"rows were built and checked under None"):
        _plan(run_files, tiny_model, monkeypatch)


def test_a_recorded_condition_is_found_under_the_artifact_root_it_was_fetched_into(tmp_path, monkeypatch):
    relative = "sampling/camp/block-000/M/grpo-v2-anchors/confirm/condition.json"
    built = {"anchor_condition": f"/elsewhere/builder/root/{relative}"}
    root = tmp_path / "root"
    fetched = root / relative
    fetched.parent.mkdir(parents=True)
    fetched.write_text('{"k": 2}')
    given = tmp_path / "given.json"
    given.write_text('{"k": 2}')
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT", raising=False)
    assert tn.recorded_condition(built["anchor_condition"]) == []
    with pytest.raises(tn.TrainerConfigError, match=r"at that path or under \$RULEFORGE_ARTIFACT_ROOT"):
        tn.check_built_condition(built, given)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(root))
    assert tn.recorded_condition(built["anchor_condition"]) == [fetched]
    tn.check_built_condition(built, given)
    tn.check_built_condition(built, fetched)
    given.write_text('{"k": 3}')
    with pytest.raises(tn.TrainerConfigError, match="is not the condition the rows were built from"):
        tn.check_built_condition(built, given)
    fetched.unlink()
    assert tn.recorded_condition(built["anchor_condition"]) == []
    assert tn.recorded_condition(str(given)) == [given]
    tn.check_built_condition({"anchor_condition": None}, given)


def test_the_plan_binds_the_rows_to_their_condition(run_files, tiny_model, monkeypatch):
    manifest = json.loads(run_files["manifest"].read_text())
    run_files["manifest"].write_text(json.dumps({**manifest, "anchor_condition": str(run_files["condition"])}))
    plan = _plan(run_files, tiny_model, monkeypatch)
    assert plan.provenance["rows_model"]["train"]["anchor_condition"] == str(run_files["condition"])
    other = run_files["condition"].with_name("other_condition.json")
    other.write_text("{}")
    run_files["manifest"].write_text(json.dumps({**manifest, "anchor_condition": str(other)}))
    with pytest.raises(tn.TrainerConfigError, match="is not the condition the rows were built from"):
        _plan(run_files, tiny_model, monkeypatch)
    run_files["manifest"].write_text(json.dumps(manifest))


class _StubBase:
    """A GRPOTrainer stand-in for the anchor's dispatch: training_step returns 1."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        return torch.tensor(1.0)


def _stub_trainer(**state):
    cls = tn.make_anchored_grpo_trainer_class(_StubBase)
    trainer = cls.__new__(cls)
    trainer.anchored = True
    trainer.anchor = tn.AnchorSettings(config=AnchorAuxConfig(loss_weight=1.0), accum=3, temperature=1.0)
    trainer.state = types.SimpleNamespace(global_step=state.get("global_step", 5))
    trainer.accelerator = types.SimpleNamespace(sync_gradients=state.get("sync", True))
    trainer.anchor_manager = types.SimpleNamespace(started=False, start=lambda step: trainer.calls.append(("start", step)))
    trainer.calls = []
    recorded = []
    trainer._record_anchor = lambda weighted, sums: recorded.append((weighted, dict(sums)))
    trainer.recorded = recorded
    return trainer


def test_the_anchor_runs_anchor_accum_micro_steps_on_the_sync_step_only():
    trainer = _stub_trainer()

    def micro_step(model):
        trainer.calls.append("micro")
        return torch.tensor(0.2), torch.tensor(0.6), (1.5, 3.0)

    trainer._anchor_micro_step = micro_step
    loss = trainer.training_step(None, None)
    assert trainer.calls == [("start", 5), "micro", "micro", "micro"]
    assert float(loss) == pytest.approx(1.0 + 3 * 0.2)
    assert trainer.recorded[0][0] == pytest.approx(0.6)
    assert trainer.recorded[0][1] == pytest.approx({"raw": 0.6, "answer_kl": 4.5, "answer_tokens": 9.0})
    for idle in (_stub_trainer(sync=False), _stub_trainer()):
        idle._anchor_micro_step = micro_step
        idle.anchored = not idle.accelerator.sync_gradients
        idle.calls = []
        assert float(idle.training_step(None, None)) == 1.0 and idle.calls == []


def test_the_micro_step_divides_by_anchor_accum_before_its_backward():
    trainer = _stub_trainer()
    trainer.anchor = tn.AnchorSettings(config=AnchorAuxConfig(loss_weight=2.0), accum=4, temperature=1.0)
    trainer.think_end_token_id = THINK_END
    rows = [_replay_row(1)]
    trainer.anchor_manager = types.SimpleNamespace(next_batch=lambda: {
        "input_ids": torch.tensor([rows[0]["input_ids"]]), "attention_mask": torch.ones(1, 12, dtype=torch.long),
        "labels": torch.tensor([rows[0]["labels"]]), "weight": torch.tensor([1.0])})
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    torch.manual_seed(0)
    policy, base = torch.randn(1, 12, 32, requires_grad=True), torch.randn(1, 12, 32)
    trainer._anchor_logits = lambda model, batch, adapter: policy if adapter else base
    backward = []
    trainer._anchor_backward = backward.append
    scaled, raw, (answer_kl, answer_tokens) = trainer._anchor_micro_step(None)
    assert backward[0].item() == pytest.approx(2.0 * float(raw) / 4) and scaled.item() == backward[0].item()
    assert answer_tokens == 2.0 and answer_kl > 0  # the verdict and the end of turn, after </think>
    trainer.anchor = tn.AnchorSettings(config=AnchorAuxConfig(loss_weight=2.0, kind="ce"), accum=4, temperature=1.0)
    assert trainer._anchor_micro_step(None)[2] == (0.0, 0.0)


class _FullStub:
    """A GRPOTrainer stand-in for the hooks the dry run cannot reach on one CPU process."""

    def __init__(self, **kwargs):
        self.saved = []
        self.loaded = []

    batch = {"completion_mask": torch.ones(1, 2)}

    def _generate_and_score_completions(self, inputs):
        return self.batch

    def _save_checkpoint(self, model, trial):
        self.saved.append(trial)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        self.loaded.append((resume_from_checkpoint, model))

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        self.loaded.append(("prepared", self.model))
        return self.model, train_dataloader

    def _save_optimizer_and_scheduler(self, output_dir):
        self.saved.append(("optimizer", output_dir))

    def _load_optimizer_and_scheduler(self, checkpoint):
        self.loaded.append(("optimizer", checkpoint))


def _full_stub(**attributes):
    from collections import defaultdict

    cls = tn.make_anchored_grpo_trainer_class(_FullStub)
    trainer = cls.__new__(cls)
    _FullStub.__init__(trainer)
    trainer._metrics = {"train": defaultdict(list)}
    trainer.telemetry = {"logp_floor_step1": None, "sync_probe_last": None, "retention": [], "alarms": [],
                              "retention_alarm_checked_at": None}
    trainer.args = types.SimpleNamespace(should_save=False, output_dir="/nonexistent")
    for name, value in attributes.items():
        setattr(trainer, name, value)
    return trainer


def test_the_hooks_off_the_main_path():
    from ruleforge.grpo.sync_probe import PointResult, ProbeResult, RetentionAlarm

    trainer = _full_stub(retention_alarm=RetentionAlarm(step=0, threshold=0.9))
    result = ProbeResult(step=3, engine_dtype="bfloat16", points=[
        PointResult("a", "p", 0, 4, 2, 0, 0, 0.5, 0.5), PointResult("b", "q", 0, 4, 0, 0, 0, None, None)])
    trainer._on_probe_result(result)
    assert trainer._metrics["train"]["sync/retention/a"] == [0.5] and "sync/retention/b" not in trainer._metrics["train"]
    assert trainer.telemetry["alarms"][0]["step"] == 3 and trainer.telemetry["retention"] == [[3, 0.5]]
    assert trainer.telemetry["retention_alarm_checked_at"] == 3
    trainer._on_probe_result(dataclasses.replace(result, step=4))
    assert len(trainer.telemetry["alarms"]) == 1 and trainer.telemetry["retention_alarm_checked_at"] == 3
    # evaluation-mode generations are not training telemetry, and a batch without vLLM log-probs records nothing
    trainer.model = types.SimpleNamespace(training=False)
    assert trainer._generate_and_score_completions([]) is _FullStub.batch
    trainer._record_truncation({"old_per_token_logps": None, "completion_mask": torch.ones(1, 2)})
    assert "truncation/outside_kept_share" not in trainer._metrics["train"]
    trainer.anchor = tn.AnchorSettings(config=AnchorAuxConfig(loss_weight=1.0, kind="ce"), accum=1, temperature=1.0)
    trainer.anchor_manager = types.SimpleNamespace(rows_served=4)
    trainer.accelerator = types.SimpleNamespace(device=torch.device("cpu"), gather=lambda tensor: tensor)
    trainer._record_anchor(0.25, {"raw": 0.5, "answer_kl": 0.0, "answer_tokens": 0.0})
    assert trainer._metrics["train"]["anchor/ce"] == [0.5] and "anchor/kl_answer" not in trainer._metrics["train"]
    trainer._save_checkpoint(None, "trial")
    assert trainer.saved == ["trial"]


def test_old_log_probs_outside_the_kept_set_are_zeroed_after_trls_is_ratio():
    trainer = _full_stub(truncation=Truncation(temperature=1.0, top_k=20, top_p=0.95),
                         model=types.SimpleNamespace(training=False))
    trainer.batch = {"completion_mask": torch.ones(1, 3), "old_per_token_logps": torch.tensor([[-0.5, -1e4, -2.0]])}
    output = trainer._generate_and_score_completions([])
    assert output["old_per_token_logps"].tolist() == [[-0.5, 0.0, -2.0]]


def test_the_sync_alarm_file_holds_each_alarm_once_across_a_resume(tmp_path):
    from ruleforge.grpo.sync_probe import PointResult, ProbeResult, RetentionAlarm

    result = ProbeResult(step=100, engine_dtype="bfloat16", points=[PointResult("a", "p", 0, 4, 2, 0, 0, 0.2, 0.8)])
    for _ in range(2):  # the run, then its resume from checkpoint-100 (whose telemetry predates the alarm)
        trainer = _full_stub(retention_alarm=RetentionAlarm(step=100, threshold=0.5))
        trainer.args = types.SimpleNamespace(should_save=True, output_dir=str(tmp_path / "run"))
        trainer._on_probe_result(result)
        assert len(trainer.telemetry["alarms"]) == 1
    assert len((tmp_path / "run" / tn.ALARMS_FNAME).read_text().splitlines()) == 1
    other_step = {**trainer.telemetry["alarms"][0], "step": 125}
    assert tn.append_alarm(tmp_path / "run" / tn.ALARMS_FNAME, other_step)
    assert not tn.append_alarm(tmp_path / "run" / tn.ALARMS_FNAME, other_step)


@pytest.fixture
def cpu_plan(run_files, tiny_model, monkeypatch):
    return _plan(run_files, tiny_model, monkeypatch)


def test_the_kl_is_zero_at_a_zero_adapter(cpu_plan):
    """At init lora_B is 0: policy and base logits agree bit for bit, so the KL and its gradient are 0."""
    trainer = tn.build_trainer(cpu_plan, grpo_overrides=CPU)
    assert type(trainer.anchor_manager.sampler).__name__ == "RandomSampler"
    trainer.anchor_manager.start(0)
    trainer.model.zero_grad(set_to_none=True)
    scaled, raw, answer = trainer._anchor_micro_step(trainer.model)
    assert float(raw) == 0.0 and float(scaled) == 0.0 and answer[0] == 0.0 and answer[1] > 0
    grads = [p.grad for name, p in trainer.model.named_parameters() if "lora_" in name]
    assert grads and all(grad is not None and torch.count_nonzero(grad) == 0 for grad in grads)


def test_the_trainer_refuses_an_unanchored_objective_or_other_sampling(cpu_plan, monkeypatch):
    cls = tn.anchored_grpo_trainer_class()
    args = tn.build_grpo_config({**cpu_plan.grpo_kwargs, **CPU, "beta": 0.04})
    with pytest.raises(tn.TrainerConfigError, match="beta 0 and mask_truncated_completions True"):
        cls(model=str(cpu_plan.model_path), reward_funcs=cpu_plan.reward_set.reward_funcs, args=args,
            truncation=cpu_plan.truncation, anchor_rows=cpu_plan.replay, anchor=cpu_plan.anchor)
    with pytest.raises(tn.TrainerConfigError, match="come together"):
        cls(args=args, truncation=cpu_plan.truncation, anchor=cpu_plan.anchor)
    liger = types.SimpleNamespace(beta=0.0, mask_truncated_completions=True, use_liger_kernel=True)
    with pytest.raises(tn.TrainerConfigError, match="use_liger_kernel"):
        cls(args=liger, truncation=cpu_plan.truncation)
    with pytest.raises(tn.TrainerConfigError, match="positive loss_weight"):
        tn.AnchorSettings(config=AnchorAuxConfig(loss_weight=0.0), accum=4, temperature=1.0)
    other = Truncation(temperature=1.0, top_k=10, top_p=0.95)
    from datasets import Dataset

    common = dict(model=str(cpu_plan.model_path), reward_funcs=cpu_plan.reward_set.reward_funcs,
                  args=tn.build_grpo_config({**cpu_plan.grpo_kwargs, **CPU}),
                  train_dataset=Dataset.from_list(cpu_plan.train_rows), processing_class=cpu_plan.tokenizer,
                  peft_config=cpu_plan.preset.lora_config())
    with pytest.raises(tn.TrainerConfigError, match="the mask applies"):
        cls(**common, truncation=other)
    with monkeypatch.context() as patch:
        patch.setattr(cpu_plan.tokenizer, "response_template", "{{ content }}", raising=False)
        with pytest.raises(tn.TrainerConfigError, match="response template"):
            cls(**common, truncation=cpu_plan.truncation)
    trainer = cls(**common, truncation=cpu_plan.truncation)
    assert trainer.anchor_manager is None and not trainer.anchored
    with pytest.raises(tn.TrainerConfigError, match="wrapped 14 modules, not the expected 252"):
        trainer.check_lora_coverage(252)
    trainer.vllm_mode = "server"
    with pytest.raises(tn.TrainerConfigError, match="in-process engine.*this run is server"):
        trainer.install_sync_probe(SyncProbe.for_modules(tn.default_probe_modules(2), trainer.model.config,
                                                         max_rows=4))


class _FakeEngineModel(torch.nn.Module):
    """The in-process vLLM model: packed bf16 parameters, filled by ``load_weights`` as vLLM's mapper does."""

    def __init__(self, config) -> None:
        super().__init__()
        q, kv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
        self.offsets = {"q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", q), "v_proj": ("qkv_proj", q + kv),
                        "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", config.intermediate_size)}
        self.params = torch.nn.ParameterDict()
        self.loaded: list[str] = []
        for layer in range(config.num_hidden_layers):
            for name, rows, cols in (("self_attn.qkv_proj", q + 2 * kv, config.hidden_size),
                                     ("self_attn.o_proj", config.hidden_size, q),
                                     ("mlp.gate_up_proj", 2 * config.intermediate_size, config.hidden_size),
                                     ("mlp.down_proj", config.hidden_size, config.intermediate_size)):
                key = f"model.layers.{layer}.{name}.weight".replace(".", "__")
                self.params[key] = torch.nn.Parameter(torch.zeros(rows, cols, dtype=torch.bfloat16),
                                                      requires_grad=False)
        self.params["model__norm__weight"] = torch.nn.Parameter(
            torch.zeros(config.hidden_size, dtype=torch.bfloat16), requires_grad=False)

    def named_parameters(self, *args, **kwargs):
        for key, parameter in self.params.items():
            yield key.replace("__", "."), parameter

    def load_weights(self, weights):
        for name, tensor in weights:
            self.loaded.append(name)
            parent, _, leaf = name.removesuffix(".weight").rpartition(".")
            packed, offset = self.offsets.get(leaf, (leaf, 0))
            key = f"{parent}.{packed}.weight".replace(".", "__")
            if key in self.params:
                # vLLM's loader: param.data.copy_ of the fp32 tensor, rounding to nearest even
                self.params[key].data[offset : offset + tensor.shape[0]].copy_(tensor)

    def discard(self):
        """A level-2 sleep: the weights are gone until the next push."""
        for parameter in self.params.values():
            parameter.data.fill_(float("nan"))


def _fake_generation_class(lost_push: bool = False, reload_after_sync: bool = False):
    """TRL's VLLMGeneration with its engine and its sampler replaced; its constructor and sync are TRL's."""
    from safetensors.torch import load_file
    from trl.generation.vllm_generation import VLLMGeneration

    from ruleforge.grpo.truncation import mask_logits

    class FakeGeneration(VLLMGeneration):
        instances: list = []

        def _init_vllm(self):
            self.engine = _FakeEngineModel(self.model.config)
            self.llm = types.SimpleNamespace(
                llm_engine=types.SimpleNamespace(model_executor=types.SimpleNamespace(
                    driver_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=self.engine)))),
                reset_prefix_cache=lambda: None,
                wake_up=lambda tags=None: None,
                sleep=lambda level=1: self.engine.discard(),
                collective_rpc=self._collective_rpc,
                generate=self._sample,
            )
            if lost_push:
                push = self.engine.load_weights

                def reload_from_disk(weights):
                    for name, tensor in weights:
                        module = self.model.get_submodule(f"base_model.model.{name.removesuffix('.weight')}")
                        if hasattr(module, "get_delta_weight"):
                            tensor = tensor - module.get_delta_weight("default")
                        push([(name, tensor)])

                self.engine.load_weights = reload_from_disk
            if self.enable_sleep_mode:
                self.llm.sleep(level=2)
            self._llm_weights_sleeping = self.enable_sleep_mode
            FakeGeneration.instances.append(self)

        def _collective_rpc(self, method):
            assert method == "reload_weights"
            self.engine.load_weights(load_file(str(Path(self.model.name_or_path) / "model.safetensors")).items())

        def generate(self, prompts, images, num_generations, profiler=None):
            if reload_after_sync:  # TRL 1.7 (vllm_generation.py:535-541): no re-push; the checkpoint instead
                self.llm.wake_up(tags=["weights"])
                self.llm.collective_rpc("reload_weights")
            elif self.enable_sleep_mode and self._llm_weights_sleeping:  # TRL 1.13 (:587-590): a re-push
                self.sync_weights()
            truncation = Truncation(temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                                    min_p=self.min_p, repetition_penalty=self.repetition_penalty)
            completions, logprobs = self.llm.generate(prompts, sampling_params=truncation, use_tqdm=False)
            if self.enable_sleep_mode:
                self.llm.sleep(level=2)
                self._llm_weights_sleeping = True
            return prompts, completions, logprobs, None

        def _sample(self, prompts, sampling_params, use_tqdm=False):
            completions, logprobs = [], []
            with torch.no_grad():
                for prompt in prompts:
                    ids, lps = list(prompt), []
                    for _ in range(self.max_completion_length):
                        logits = self.model(input_ids=torch.tensor([ids])).logits[0, -1].float()
                        logp = torch.log_softmax(mask_logits(logits, sampling_params), dim=-1)
                        token = int(torch.multinomial(logp.exp(), 1))
                        ids.append(token)
                        lps.append([float(logp[token])])
                        if token == EOS:
                            break
                    completions.append(ids[len(prompt):])
                    logprobs.append(lps)
            return completions, logprobs

    return FakeGeneration


class _StopAt(transformers.TrainerCallback):
    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


class _Logs(transformers.TrainerCallback):
    def __init__(self) -> None:
        self.logs: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logs.append(dict(logs or {}))


def _with_callbacks(monkeypatch, *callbacks):
    build = tn.build_trainer

    def build_with_callbacks(plan, **kwargs):
        trainer = build(plan, **kwargs)
        for callback in callbacks:
            trainer.add_callback(callback)
        return trainer

    monkeypatch.setattr(tn, "build_trainer", build_with_callbacks)


@requires_git_worktree  # the checkpoint records the checkout's code state and snapshot
def test_the_cpu_dry_run_trains_probes_checkpoints_and_resumes(run_files, tiny_model, monkeypatch):
    generation_class = _fake_generation_class()
    monkeypatch.setattr("trl.trainer.grpo_trainer.VLLMGeneration", generation_class)
    logs = _Logs()
    with monkeypatch.context() as patch:
        _with_callbacks(patch, _StopAt(2), logs)
        trainer = tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU), reward_factory=RewardDouble,
                              marker_checker=marker_double)
    assert trainer.state.global_step == 2 and trainer.use_vllm

    # TRL handed the engine the truncation and no other sampling kwargs
    generation = generation_class.instances[0]
    assert (generation.temperature, generation.top_k, generation.top_p, generation.min_p,
            generation.repetition_penalty, generation.generation_kwargs) == (1.0, 20, 0.95, 0.0, 1.0, {})
    assert "model.layers.0.self_attn.k_proj.weight" in generation.engine.loaded
    history = trainer.sync_probe.history
    assert len(history) == 2 and all(result.ok for result in history) and history[0].vacuous
    assert not history[1].vacuous and history[1].step == 1
    assert [(point.module, point.lora, point.ok) for point in history[1].points][-1] == ("model.norm", False, True)
    seen = {key for entry in logs.logs for key in entry}
    for key in ("anchor/kl", "anchor/kl_answer", "anchor/loss_weighted", "truncation/outside_kept_share",
                "truncation/logp_difference_kept_mean", "truncation/logp_difference_kept_signed_mean",
                "sync/probe_ok", "sync/retention",
                "rewards/flat_soft_reward/mean", "rewards/map_completeness_reward/mean", "consistency/double_rows",
                "sampling/importance_sampling_ratio/mean"):
        assert key in seen, key
    assert logs.logs[0]["anchor/kl"] == 0.0
    assert all(0.0 <= entry.get("truncation/outside_kept_share", 0.0) <= 1.0 for entry in logs.logs)

    output_dir = Path(trainer.args.output_dir)
    checkpoint = output_dir / "checkpoint-2"
    record = json.loads((checkpoint / tn.PROVENANCE_FNAME).read_text())
    assert record["step"] == 2 and record["adapter_sha256"] == sha256_file(checkpoint / tn.ADAPTER_WEIGHTS_FNAME)
    assert record["grpo_config"]["top_k"] == 20 and record["grpo_config"]["beta"] == 0.0
    assert record["grpo_config"]["vllm_importance_sampling_mode"] == "token_truncate"
    assert record["run"]["identity"]["digests"]["rules_sha256"] == RULES_SHA256
    assert record["run"]["code"]["code_state_sha256"] and Path(record["run"]["code_snapshot"]).is_file()
    telemetry = record["telemetry"]
    floor = telemetry["logp_floor_step1"]
    assert 0 < floor["logp_difference_kept_mean"] < 0.05 and floor["outside_kept_share"] == 0.0
    assert floor["logp_difference_kept_mean"] == pytest.approx(floor["trl_sampling_logp_difference_mean"])
    assert abs(floor["logp_difference_kept_signed_mean"]) <= floor["logp_difference_kept_mean"]
    assert record["run"]["rollout_engine_seeds"] == [0]  # TRL's process_index // TP, not grpo.seed
    assert "prereg_sha256" not in record["run"]["identity"]["digests"]
    assert telemetry["sync_probe_last"]["ok"] and telemetry["retention"]
    assert telemetry["generation_checks"] == 2
    assert record["batch_geometry"] == {"world_size": 1, "generation_batch_size": 2 * 1 * 2}
    assert telemetry["alarms"][0]["alarm"] == "retention"  # threshold 2.0: forced to fire, once
    assert len((output_dir / tn.ALARMS_FNAME).read_text().splitlines()) == 1
    assert record["anchor"]["sampler"] == "RandomSampler" and record["anchor"]["rows_served"] == 2 * 4

    resumed = tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU, "resume_from_checkpoint=latest"),
                          reward_factory=RewardDouble, marker_checker=marker_double)
    assert resumed.state.global_step == 3
    first_after = resumed.sync_probe.history[0]
    assert first_after.ok and first_after.step == 2 and not first_after.vacuous
    assert resumed.anchor_manager.start_step == 2
    final = json.loads((output_dir / "checkpoint-3" / tn.PROVENANCE_FNAME).read_text())
    assert final["telemetry"]["logp_floor_step1"] == floor
    assert len(final["telemetry"]["alarms"]) == 1
    assert final["anchor"]["rows_served"] == 3 * 4
    assert final["telemetry"]["retention_alarm_checked_at"] == 1 == telemetry["retention_alarm_checked_at"]
    assert final["telemetry"]["generation_checks"] == 3 and resumed.sync_probe.generation_checks == 1
    with pytest.raises(tn.TrainerConfigError, match="already holds checkpoint-3"):
        _plan(run_files, tiny_model, monkeypatch, *ON_CPU)
    # the run is finished (checkpoint-3 is at max_steps 3): latest refuses, in the plan, before the model loads
    loads = []
    with monkeypatch.context() as patch:
        patch.setattr(tn, "build_trainer", lambda plan, **kwargs: loads.append(plan))
        with pytest.raises(tn.TrainerConfigError, match="checkpoint-3 is at step 3, already at max_steps 3"):
            tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU, "resume_from_checkpoint=latest"),
                        reward_factory=RewardDouble, marker_checker=marker_double)
    assert loads == [] and [path.name for path in tn.checkpoint_dirs(output_dir)][0] == "checkpoint-3"
    earlier = output_dir / "checkpoint-2"
    before = json.loads((earlier / tn.PROVENANCE_FNAME).read_text())
    before["batch_geometry"] = {"world_size": 2, "generation_batch_size": 8}
    tn.write_provenance(earlier, before)
    with pytest.raises(tn.TrainerConfigError, match="resume on the ranks it recorded"):
        tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU, f"resume_from_checkpoint={earlier}"),
                    reward_factory=RewardDouble, marker_checker=marker_double)


def test_a_push_the_engine_does_not_keep_aborts_the_run(run_files, tiny_model, monkeypatch):
    monkeypatch.setattr("trl.trainer.grpo_trainer.VLLMGeneration", _fake_generation_class(lost_push=True))
    with pytest.raises(SyncProbeError, match=r"sync probe failed after the sync at step 1: .*equal to the base W"):
        tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU), reward_factory=RewardDouble,
                    marker_checker=marker_double)


def test_a_reload_after_the_sync_aborts_the_run(run_files, tiny_model, monkeypatch):
    monkeypatch.setattr("trl.trainer.grpo_trainer.VLLMGeneration", _fake_generation_class(reload_after_sync=True))
    with pytest.raises(SyncProbeError, match=r"changed between the sync probed at step 1 and sampling at step 1 "
                                             r".*k_proj .*: 128 of 128 elements, 128 of them now equal to the base W"):
        tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *ON_CPU), reward_factory=RewardDouble,
                    marker_checker=marker_double)


def test_a_stop_marker_ends_the_run_after_the_next_save_and_the_run_never_resumes(run_files, tiny_model, monkeypatch):
    from ruleforge.grpo import run_files as rf

    monkeypatch.setattr("trl.trainer.grpo_trainer.VLLMGeneration", _fake_generation_class())
    config = _config(run_files, tiny_model, monkeypatch, *ON_CPU)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / rf.STOP_FNAME).write_text(json.dumps({"fired_at": 300, "rule_sha256": "p" * 64}), encoding="utf-8")
    trainer = tn.run_grpo(config, reward_factory=RewardDouble, marker_checker=marker_double)
    assert trainer.state.global_step == 1  # max_steps 3, save_steps 1: it stopped after the first save
    assert [path.name for path in tn.checkpoint_dirs(output_dir)] == ["checkpoint-1"]
    record = json.loads((output_dir / "checkpoint-1" / tn.PROVENANCE_FNAME).read_text())
    digest = rf.stop_marker_sha256(output_dir)
    assert record["early_stop"] == {"marker_sha256": digest, "stopped_after_step": 1}
    stopped = rf.read_stopped(output_dir)
    assert (stopped["step"], stopped["marker_sha256"]) == (1, digest) and stopped["at"]
    with pytest.raises(tn.TrainerConfigError, match="ended at its early stop after step 1"):
        _plan(run_files, tiny_model, monkeypatch, *ON_CPU, "resume_from_checkpoint=latest")


def test_every_rank_stops_when_any_rank_sees_the_marker(tmp_path):
    trainer = _full_stub(accelerator=types.SimpleNamespace(device=torch.device("cpu"),
                                                           gather=lambda flag: torch.tensor([flag.item(), 1.0])),
                         state=types.SimpleNamespace(global_step=200))
    assert trainer.early_stop_requested(tmp_path) == {"marker_sha256": None, "stopped_after_step": 200}
    alone = _full_stub(accelerator=types.SimpleNamespace(device=torch.device("cpu"), gather=lambda flag: flag),
                       state=types.SimpleNamespace(global_step=200))
    assert alone.early_stop_requested(tmp_path) is None


def test_a_run_under_fsdp_resumes_only_from_its_sharded_resume_state(run_files, tiny_model, monkeypatch):
    from ruleforge.grpo import fsdp2_state

    latest = "resume_from_checkpoint=latest"
    fsdp1, fsdp2 = (latest, "+grpo.fsdp=true", "+grpo.fsdp_config={version: 1}"), (
        latest, "+grpo.fsdp=true", "+grpo.fsdp_config={version: 2}")
    # FSDP2 as transformers 5.17 reads it too: a version left unset, or spelled with its fsdp_ prefix
    unset, prefixed = (latest, "+grpo.fsdp=true"), (latest, "+grpo.fsdp=true", "+grpo.fsdp_config={fsdp_version: 2}")
    plans = {name: _plan(run_files, tiny_model, monkeypatch, *extra)
             for name, extra in (("plain", (latest,)), ("fsdp1", fsdp1), ("fsdp2", fsdp2), ("unset", unset),
                                 ("prefixed", prefixed))}
    assert all(plan.resume_from is None for plan in plans.values())
    checkpoint = plans["fsdp2"].output_dir / "checkpoint-1"
    checkpoint.mkdir(parents=True)

    def saved_under(name):
        tn.write_provenance(checkpoint, {"step": 1, "run": {"identity": plans[name].provenance["identity"]}})

    saved_under("fsdp2")
    with pytest.raises(tn.TrainerConfigError, match="checkpoint-1: it holds no sharded resume state .*saved without "
                                                    "FSDP"):
        _plan(run_files, tiny_model, monkeypatch, *fsdp2)
    record = fsdp2_state.state_dir(checkpoint) / fsdp2_state.RECORD_FNAME
    record.parent.mkdir()
    record.write_text(json.dumps({"format": 2}))
    with pytest.raises(tn.TrainerConfigError, match="its resume state is format 2, not 1"):
        _plan(run_files, tiny_model, monkeypatch, *fsdp2)
    record.write_text(json.dumps({"format": 1, "world_size": 2, "step": 1, "ranks": []}))
    assert _plan(run_files, tiny_model, monkeypatch, *fsdp2).resume_from == checkpoint
    for name, extra in (("unset", unset), ("prefixed", prefixed)):
        saved_under(name)
        assert _plan(run_files, tiny_model, monkeypatch, *extra).resume_from == checkpoint
    saved_under("fsdp1")
    with pytest.raises(tn.TrainerConfigError, match="checkpoint-1: a run under FSDP1 does not resume.*rank 0's shard"):
        _plan(run_files, tiny_model, monkeypatch, *fsdp1)
    saved_under("plain")
    assert _plan(run_files, tiny_model, monkeypatch, latest).resume_from == checkpoint


def _fsdp_stub(version, **attributes):
    plugin = types.SimpleNamespace(fsdp_version=version)
    return _full_stub(**{
        "is_fsdp_enabled": version is not None, "model": "model", "model_wrapped": "model", "rank0_master": False,
        "accelerator": types.SimpleNamespace(state=types.SimpleNamespace(fsdp_plugin=plugin)),
        "resumed_adapter_sha256": "a" * 64, "optimizer": "optimizer", "lr_scheduler": "scheduler",
        "state": types.SimpleNamespace(global_step=7), **attributes})


def test_the_fsdp2_overrides_shard_from_rank_0_and_save_and_load_the_sharded_state(monkeypatch):
    from ruleforge.grpo import fsdp2_state

    calls = []
    monkeypatch.setattr(fsdp2_state, "shard_from_rank0", lambda accelerator, model: calls.append("shard") or "sharded")
    monkeypatch.setattr(fsdp2_state, "save_state", lambda path, **kw: calls.append(("save", path, kw)))
    monkeypatch.setattr(fsdp2_state, "load_adapter", lambda path, model, **kw: calls.append(("adapter", path, model, kw)))
    monkeypatch.setattr(fsdp2_state, "load_optimizer", lambda path, **kw: calls.append(("optimizer", path, kw)))
    plain, fsdp1, fsdp2 = _fsdp_stub(None), _fsdp_stub(1), _fsdp_stub(2, rank0_master=True)
    assert (plain._fsdp_version(), fsdp1._fsdp_version(), fsdp2._fsdp_version()) == (None, 1, 2)
    # the master from rank 0: sharded once, then HF's preparation over the shards
    assert fsdp2._prepare_for_training(10, "loader", None) == ("sharded", "loader")
    assert fsdp2._prepare_for_training(10, "loader", None) == ("sharded", "loader")
    assert calls == ["shard"] and fsdp2.model == fsdp2.model_wrapped == "sharded" and not fsdp2.rank0_master
    assert fsdp2.loaded == [("prepared", "sharded"), ("prepared", "sharded")]
    plain._prepare_for_training(10, None, None)
    assert plain.loaded == [("prepared", "model")] and calls == ["shard"]
    # the save: the sharded state under FSDP2, HF's otherwise
    for stub in (plain, fsdp1, fsdp2):
        stub._save_optimizer_and_scheduler("out/checkpoint-7")
    assert plain.saved == fsdp1.saved == [("optimizer", "out/checkpoint-7")] and fsdp2.saved == []
    assert calls[1] == ("save", "out/checkpoint-7", {"model": "sharded", "optimizer": "optimizer",
                                                      "scheduler": "scheduler", "step": 7})
    fsdp2._load_from_checkpoint("out/checkpoint-7", "wrapped")
    fsdp2._load_optimizer_and_scheduler("out/checkpoint-7")
    fsdp2._load_optimizer_and_scheduler(None)
    assert calls[2:] == [("adapter", "out/checkpoint-7", "sharded", {
        "adapter_file": Path("out/checkpoint-7/adapter_model.safetensors"), "expect_adapter_sha256": "a" * 64}),
        ("optimizer", "out/checkpoint-7", {"model": "sharded", "optimizer": "optimizer", "scheduler": "scheduler"})]
    assert fsdp2.loaded[-1] == ("optimizer", None)
    with pytest.raises(tn.TrainerConfigError, match="checkpoint-1: a run under FSDP1 does not resume"):
        fsdp1._load_from_checkpoint("checkpoint-1", "wrapped")
    fsdp1._load_optimizer_and_scheduler("checkpoint-1")
    plain._load_from_checkpoint("checkpoint-1")
    plain._load_optimizer_and_scheduler("checkpoint-1")
    assert fsdp1.loaded == [("optimizer", "checkpoint-1")]
    assert plain.loaded[1:] == [("checkpoint-1", None), ("optimizer", "checkpoint-1")]
    assert len(calls) == 4


def test_the_trainer_builds_from_the_path_or_under_fsdp2_from_rank_0(monkeypatch):
    """TRL loads the checkpoint's path on every rank (without FSDP, or under FSDP1)."""
    from ruleforge.grpo import fsdp2_state

    plan = types.SimpleNamespace(model_path=Path("/models/gemma"), grpo_kwargs={
        "model_init_kwargs": {"dtype": "float32", "attn_implementation": "sdpa"}})
    monkeypatch.setattr(fsdp2_state, "rank0_model", lambda path, **kw: ("rank0_model", path, kw))
    assert tn.trainer_model(plan, types.SimpleNamespace(fsdp=None)) == "/models/gemma"
    assert tn.trainer_model(plan, types.SimpleNamespace(fsdp="full_shard", fsdp_config={"version": 1})) == (
        "/models/gemma")
    for config in ({"version": 2}, {}):
        fsdp2 = types.SimpleNamespace(fsdp="full_shard", fsdp_config=config, process_index=3)
        assert tn.trainer_model(plan, fsdp2) == ("rank0_model", Path("/models/gemma"), {
            "rank": 3, "dtype": "float32", "attn_implementation": "sdpa"})


#: A run under FSDP2, as gemma4_31b.yaml's (the plan reads the version: transformers 5.17's).
FSDP2 = ("+grpo.fsdp=true", "+grpo.fsdp_config={version: 2}")


def _init_checkpoint(run_files, tiny_model, monkeypatch, *, world_size: int = 4) -> Path:
    round1 = _plan(run_files, tiny_model, monkeypatch, "run_name=round-1", *FSDP2)
    checkpoint = round1.output_dir / "checkpoint-200"
    checkpoint.mkdir(parents=True)
    (checkpoint / tn.ADAPTER_WEIGHTS_FNAME).write_bytes(b"round 1's adapter at step 200")
    (checkpoint / tn.ADAPTER_CONFIG_FNAME).write_text(json.dumps({"base_model_name_or_path": str(tiny_model)}))
    record = tn.fsdp2_state.state_dir(checkpoint) / tn.fsdp2_state.RECORD_FNAME
    record.parent.mkdir()
    record.write_text(json.dumps({"format": 1, "world_size": world_size, "step": 200, "ranks": []}))
    tn.write_provenance(checkpoint, {"schema_version": 1, "step": 200, "adapter_sha256": tn.adapter_digest(checkpoint),
                                     "run": round1.provenance})
    return checkpoint


def test_a_warm_start_is_bound_into_the_runs_identity_and_provenance(run_files, tiny_model, monkeypatch):
    checkpoint = _init_checkpoint(run_files, tiny_model, monkeypatch)
    cold = _plan(run_files, tiny_model, monkeypatch, *FSDP2)
    warm = _plan(run_files, tiny_model, monkeypatch, *FSDP2, f"model.init_adapter={checkpoint}")
    init = {"checkpoint": str(checkpoint), "run": "round-1", "step": 200,
            "adapter_sha256": sha256_file(checkpoint / tn.ADAPTER_WEIGHTS_FNAME),
            "provenance_sha256": sha256_file(checkpoint / tn.PROVENANCE_FNAME)}
    assert warm.init_adapter == warm.provenance["init_adapter"] == init
    assert warm.provenance["identity"]["digests"] == {**cold.provenance["identity"]["digests"], "init_adapter": init}
    assert "init_adapter" not in cold.provenance["identity"]["digests"]
    assert cold.init_adapter is None and cold.provenance["init_adapter"] is None
    assert warm.provenance["identity"]["config_sha256"] != cold.provenance["identity"]["config_sha256"]
    with monkeypatch.context() as patch:
        records = []
        patch.setattr(tn.logger, "info", lambda message, *args: records.append(message % args))
        assert tn.run_grpo(_config(run_files, tiny_model, monkeypatch, *FSDP2, f"model.init_adapter={checkpoint}",
                                   "preflight_only=true"),
                           reward_factory=RewardDouble, marker_checker=marker_double) is None
    assert any(f"init_adapter {checkpoint} (run round-1, step 200, adapter {init['adapter_sha256'][:12]}; loaded "
               "before step 0)" in line for line in records)


def _warm_checkpoint_1(warm: tn.GrpoPlan) -> Path:
    checkpoint = warm.output_dir / "checkpoint-1"
    record = tn.fsdp2_state.state_dir(checkpoint) / tn.fsdp2_state.RECORD_FNAME
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"format": 1, "world_size": 1, "step": 1, "ranks": []}))
    tn.write_provenance(checkpoint, {"step": 1, "run": warm.provenance})
    return checkpoint


def test_a_warm_runs_requeue_resumes_its_own_checkpoint_under_the_same_init_alone(run_files, tiny_model, monkeypatch):
    checkpoint = _init_checkpoint(run_files, tiny_model, monkeypatch)
    warm = (*FSDP2, f"model.init_adapter={checkpoint}")
    own = _warm_checkpoint_1(_plan(run_files, tiny_model, monkeypatch, *warm))
    resumed = _plan(run_files, tiny_model, monkeypatch, *warm, "resume_from_checkpoint=latest")
    assert resumed.resume_from == own and resumed.resumed_record["step"] == 1
    assert resumed.init_adapter == resumed.provenance["identity"]["digests"]["init_adapter"]
    assert resumed.init_adapter["checkpoint"] == str(checkpoint)
    with pytest.raises(tn.TrainerConfigError, match=r"another config_sha256, digests.*digests\.init_adapter"):
        _plan(run_files, tiny_model, monkeypatch, *FSDP2, "resume_from_checkpoint=latest")
    other = checkpoint.parent / "checkpoint-300"
    shutil.copytree(checkpoint, other)
    tn.write_provenance(other, {**json.loads((checkpoint / tn.PROVENANCE_FNAME).read_text()), "step": 300})
    with pytest.raises(tn.TrainerConfigError, match=r"digests\.init_adapter\.checkpoint .*digests\.init_adapter\.step "
                                                    "200 -> 300"):
        _plan(run_files, tiny_model, monkeypatch, *FSDP2, f"model.init_adapter={other}",
              "resume_from_checkpoint=latest")


def test_a_cold_starts_config_digest_is_the_one_before_the_init_adapter_key(run_files, tiny_model, monkeypatch):
    config = _plan(run_files, tiny_model, monkeypatch).provenance["config"]
    assert config["model"]["init_adapter"] is None
    before = {**config, "model": {key: value for key, value in config["model"].items() if key != "init_adapter"}}
    assert tn.resume_free(config) == tn.resume_free(before) and "init_adapter" not in tn.resume_free(config)["model"]
    warm = {**config, "model": {**config["model"], "init_adapter": "/r1/checkpoint-200"}}
    assert tn.resume_free(warm)["model"]["init_adapter"] == "/r1/checkpoint-200"
    # base.yaml's false data.allow_unscreened_rows likewise; a run that set it keeps it in its digest.
    assert config["data"]["allow_unscreened_rows"] is False
    assert "allow_unscreened_rows" not in tn.resume_free(config)["data"]
    taken = {**config, "data": {**config["data"], "allow_unscreened_rows": True}}
    assert tn.resume_free(taken)["data"]["allow_unscreened_rows"] is True


def test_a_warm_start_may_change_the_reward(run_files, tiny_model, monkeypatch):
    checkpoint = _init_checkpoint(run_files, tiny_model, monkeypatch)
    recorded = json.loads((checkpoint / tn.PROVENANCE_FNAME).read_text())["run"]["config"]["reward"]
    assert recorded["lambda_int"] == 0.3
    same = _plan(run_files, tiny_model, monkeypatch, *FSDP2, f"model.init_adapter={checkpoint}")
    changed = _plan(run_files, tiny_model, monkeypatch, *FSDP2, f"model.init_adapter={checkpoint}",
                    "reward.lambda_int=0.2")
    assert changed.init_adapter == same.init_adapter and changed.init_adapter["step"] == 200
    assert changed.grpo_kwargs["reward_weights"] == [1.0, 0.2, 0.1]
    assert changed.provenance["identity"]["digests"] == same.provenance["identity"]["digests"]
    assert changed.provenance["identity"]["config_sha256"] != same.provenance["identity"]["config_sha256"]
    with pytest.raises(tn.TrainerConfigError, match=r"the reward block names reward\.lambda_itn, which no reward "
                                                    r"reads: its keys are truncated, lambda_int, "):
        _plan(run_files, tiny_model, monkeypatch, "+reward.lambda_itn=0.2")


def test_a_warm_start_from_another_model_or_reading_or_without_its_sharded_state_is_refused(run_files, tiny_model,
                                                                                            monkeypatch):
    checkpoint = _init_checkpoint(run_files, tiny_model, monkeypatch)
    plan = _plan(run_files, tiny_model, monkeypatch, *FSDP2)
    provenance = json.loads((checkpoint / tn.PROVENANCE_FNAME).read_text())
    identity = provenance["run"]["identity"]

    def check(path=checkpoint, **grpo):
        return tn.check_init_adapter(path, model_path=tiny_model, preset=plan.preset,
                                     digests=plan.provenance["identity"]["digests"], truncation=plan.truncation,
                                     grpo={"fsdp": True, "fsdp_config": {"version": 2}, **grpo})

    def recorded(**changes):
        tn.write_provenance(checkpoint, {**provenance, "run": {**provenance["run"],
                                                               "identity": {**identity, **changes}}})

    assert check()["step"] == 200
    with pytest.raises(tn.TrainerConfigError, match=f"model.init_adapter {checkpoint.parent} is not a checkpoint this "
                                                    "trainer wrote"):
        check(checkpoint.parent)
    with pytest.raises(tn.TrainerConfigError, match="is not the adapter its provenance records"):
        tn.write_provenance(checkpoint, {**provenance, "adapter_sha256": "0" * 64})
        check()
    (checkpoint / tn.ADAPTER_WEIGHTS_FNAME).rename(checkpoint / "moved")
    with pytest.raises(tn.TrainerConfigError, match=r"adapter_model.safetensors \(None\) is not the adapter"):
        tn.write_provenance(checkpoint, provenance)
        check()
    (checkpoint / "moved").rename(checkpoint / tn.ADAPTER_WEIGHTS_FNAME)
    for name in tn.INIT_ADAPTER_DIGESTS:
        recorded(digests={**identity["digests"], name: "0" * 64})
        with pytest.raises(tn.TrainerConfigError, match=f"another model or reading .*{name} 000000000000"):
            check()
    recorded(preset={**identity["preset"], "master_dtype": "bfloat16"})
    with pytest.raises(tn.TrainerConfigError, match="preset.master_dtype 'bfloat16' -> 'float32'"):
        check()
    recorded(truncation={**identity["truncation"], "top_k": 64})
    with pytest.raises(tn.TrainerConfigError, match="truncation.top_k 64 -> 20"):
        check()
    recorded(digests={key: value for key, value in identity["digests"].items() if key != "rules_sha256"})
    with pytest.raises(tn.TrainerConfigError, match="rules_sha256 unset -> "):
        check()
    tn.write_provenance(checkpoint, provenance)
    config = checkpoint / tn.ADAPTER_CONFIG_FNAME
    config.write_text(json.dumps({"base_model_name_or_path": "/models/gemma-4-31B-it"}))
    with pytest.raises(tn.TrainerConfigError, match="names base '/models/gemma-4-31B-it', not 'Qwen3-tiny'"):
        check()
    config.unlink()
    with pytest.raises(tn.TrainerConfigError, match="names base None"):
        check()
    config.write_text(json.dumps({"base_model_name_or_path": str(tiny_model)}))
    with pytest.raises(tn.TrainerConfigError, match="this run is not under FSDP"):
        check(fsdp=False)
    with pytest.raises(tn.TrainerConfigError, match="a run under FSDP1 does not resume"):
        check(fsdp_config={"version": 1})
    shutil.rmtree(tn.fsdp2_state.state_dir(checkpoint))
    with pytest.raises(tn.TrainerConfigError, match="it holds no sharded resume state"):
        check()
    tn.write_provenance(checkpoint, {**provenance, "adapter_sha256": "0" * 64})
    with pytest.raises(tn.TrainerConfigError, match="is not the adapter its provenance records.*holds no sharded"):
        _plan(run_files, tiny_model, monkeypatch, *FSDP2, f"model.init_adapter={checkpoint}")


def test_the_init_adapter_loads_on_a_fresh_run_after_hfs_preparation_and_moves_the_first_sync(monkeypatch):
    from ruleforge.grpo import fsdp2_state
    from ruleforge.grpo.sync_probe import PointResult, ProbeResult, RetentionAlarm

    calls = []
    monkeypatch.setattr(fsdp2_state, "load_adapter", lambda path, model, **kw: calls.append((path, model, kw)))
    init = {"checkpoint": "/r1/checkpoint-200", "run": "r1", "step": 200, "adapter_sha256": "c" * 64}
    warm = _fsdp_stub(2, init_adapter=init, retention_alarm=RetentionAlarm(step=10_000, threshold=0.0))
    assert warm._prepare_for_training(10, "loader", None) == ("model", "loader")
    assert warm.loaded == [("prepared", "model")]
    assert calls == [(Path("/r1/checkpoint-200"), "model", {
        "adapter_file": Path("/r1/checkpoint-200/adapter_model.safetensors"), "expect_adapter_sha256": "c" * 64})]
    assert warm.warm_start_pending
    resumed = _fsdp_stub(2, init_adapter=init)
    resumed._prepare_for_training(10, "loader", "out/checkpoint-100")
    _fsdp_stub(2)._prepare_for_training(10, "loader", None)
    assert len(calls) == 1 and not resumed.warm_start_pending
    with pytest.raises(tn.TrainerConfigError, match="a warm start loads the checkpoint's FSDP2 sharded state, and "
                                                    "this trainer runs under FSDP 1"):
        _fsdp_stub(1, init_adapter=init)._prepare_for_training(10, "loader", None)
    moved = ProbeResult(step=0, engine_dtype="bfloat16", points=[PointResult("a", "p", 0, 4, 2, 0, 0, 0.6, 0.6)])
    vacuous = ProbeResult(step=0, engine_dtype="bfloat16", points=[PointResult("a", "p", 0, 4, 0, 0, 0, None, None)])
    with pytest.raises(tn.SyncProbeError, match="the first sync after the warm start from /r1/checkpoint-200 "
                                                r"\(step 0\) moved no probed element"):
        warm._on_probe_result(vacuous)
    warm._on_probe_result(moved)
    assert not warm.warm_start_pending and warm.telemetry["retention"] == [[0, 0.6]]
    warm._on_probe_result(vacuous)  # later syncs (a vacuous probe still requires equality)


def test_a_warm_start_is_checked_on_its_ranks_before_the_model_loads(run_files, tiny_model, monkeypatch):
    four = _init_checkpoint(run_files, tiny_model, monkeypatch, world_size=4)
    plan = _plan(run_files, tiny_model, monkeypatch)
    init = {"checkpoint": str(four), "run": "round-1", "step": 200, "adapter_sha256": "c" * 64}
    tn.check_init_geometry(init, types.SimpleNamespace(world_size=4))
    with pytest.raises(tn.TrainerConfigError, match="was saved on 4 ranks, and this run has 1"):
        tn.build_trainer(dataclasses.replace(plan, init_adapter=init), grpo_overrides=CPU)
    with pytest.raises(tn.TrainerConfigError, match="was saved on None ranks"):
        tn.check_init_geometry({**init, "checkpoint": str(four.parent)}, types.SimpleNamespace(world_size=1))
    record = tn.fsdp2_state.state_dir(four) / tn.fsdp2_state.RECORD_FNAME
    record.write_text(json.dumps({**json.loads(record.read_text()), "world_size": 1}))
    trainer = tn.build_trainer(dataclasses.replace(plan, init_adapter=init), grpo_overrides=CPU)
    assert trainer.init_adapter == init and not trainer.warm_start_pending
    assert tn.build_trainer(plan, grpo_overrides=CPU).init_adapter is None


def test_the_gemma4_preset_wraps_the_language_models_410_modules_on_the_meta_device():
    """Gemma 4 31B-it's own config (a copy of config.json)."""
    config = transformers.AutoConfig.from_pretrained(str(FIXTURES / "gemma4_31b"))
    preset = tn.lora_preset(config.architectures[0])
    wrapped, trainable = preset_modules(config, preset)
    assert len(wrapped) == tn.expected_module_count(config, preset) == 410 and trainable == 244_858_880
    assert all(name.startswith("model.language_model.layers.") for name in wrapped)
    assert sum(name.endswith(".v_proj") for name in wrapped) == 50
    assert (preset.r, preset.alpha, preset.master_dtype, preset.decoder_prefix) == (32, 64, "float32",
                                                                                  "model.language_model.")
    assert preset.identity()["module_scope"] == "model.language_model."
    qwen = tn.lora_preset("Qwen3ForCausalLM")
    assert qwen.identity() == {key: value for key, value in dataclasses.asdict(qwen).items() if key != "module_scope"}
    assert tn.default_probe_modules(60, preset.decoder_prefix) == [
        "model.language_model.layers.0.self_attn.k_proj", "model.language_model.layers.30.mlp.up_proj",
        "model.language_model.layers.59.mlp.down_proj"]


def test_a_run_declares_its_master_dtype_or_takes_the_presets(caplog):
    config = transformers.AutoConfig.from_pretrained(str(FIXTURES / "gemma4_31b"))
    preset = tn.lora_preset(config.architectures[0])
    assert tn.resolve_preset(config, None) is preset and tn.resolve_preset(config, "float32") is preset
    with caplog.at_level("WARNING"):
        bf16 = tn.resolve_preset(config, "bfloat16")
    assert bf16.master_dtype == "bfloat16" and bf16.identity()["master_dtype"] == "bfloat16"
    assert "drifts the base weights" in caplog.text
    with pytest.raises(tn.TrainerConfigError, match="model.master_dtype must be one of"):
        tn.resolve_preset(config, "float16")


def test_a_colocated_engines_tensor_parallel_rank_is_its_place_in_its_group():
    assert [tn.probe_tp_rank(rank, 2) for rank in range(4)] == [0, 1, 0, 1]
    assert [tn.probe_tp_rank(rank, 1) for rank in range(2)] == [0, 0]


@pytest.fixture(scope="module")
def gemma_model(tmp_path_factory) -> Path:
    from tests.tiny_gemma import save_tiny_gemma

    return save_tiny_gemma(tmp_path_factory.mktemp("models") / "gemma-4-31B-it")


def _gemma_replay_row(tokenizer, index: int) -> dict:
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": "judge rule 1"}, {"role": "user", "content": ""}],
                                           add_generation_prompt=True, tokenize=True, return_dict=True,
                                           enable_thinking=True)["input_ids"]
    completion = tokenizer.encode(f"<|channel>thought\nweigh<channel|>{'yes' if index % 2 else 'no'}",
                                  add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<turn|>")]
    return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion, "weight": 1.0}


@pytest.fixture
def gemma_plan(gemma_model, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    condition = data / "condition.json"
    condition.write_text(json.dumps(_condition(gemma_model, sampling_truncation={
        "top_k": 64, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0})))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(gemma_model))
    train = _write_jsonl(data / "grpo_train.jsonl", [_grpo_row(i) for i in range(8)])
    replay = _write_jsonl(data / "replay.jsonl", [_gemma_replay_row(tokenizer, i) for i in range(4)])
    _built_on(gemma_model, train, replay)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="training/grpo/gemma4_31b", overrides=[
            "run_name=gemma-dry", f"model.path={gemma_model}", f"data.train={train}", f"data.replay={replay}",
            f"data.anchor_condition={condition}", f"data.rules={RULES}",
            "data.prompt_token_cap=128", "vllm_length_margin=256", "reward.truncated=zero", "sync_probe.max_rows=8",
            "grpo.num_generations=2", "grpo.per_device_train_batch_size=2", "grpo.gradient_accumulation_steps=2",
            "grpo.max_completion_length=12", "grpo.report_to=[]", "grpo.vllm_tensor_parallel_size=1",
            "grpo.fsdp=null", "model.master_dtype=float32",
        ])
    return tn.plan_grpo(OmegaConf.to_container(cfg, resolve=True), marker_checker=marker_double)


def test_a_gemma_plan_reads_completions_by_its_tokenizers_ids(gemma_plan, gemma_model):
    from ruleforge.grpo.rewards import GrpoReward

    tokenizer = gemma_plan.tokenizer
    turn, pad, channel = (tokenizer.convert_tokens_to_ids(token) for token in ("<turn|>", "<pad>", "<channel|>"))
    fmt = gemma_plan.completion_format
    assert (fmt.family, fmt.finish_token_ids, fmt.think_end_token_id) == ("gemma4", (turn, pad), channel)
    # the real reward set, reading Gemma's ids; TRL's finished set is the same pair
    assert isinstance(gemma_plan.reward_set, GrpoReward) and tokenizer.eos_token_id == turn
    assert gemma_plan.reward_set.finish_token_ids == (turn, pad) and gemma_plan.reward_set.decode is not None
    assert gemma_plan.provenance["completion_format"] == fmt.record()
    assert gemma_plan.reward_set.favoured_no_rate is True
    assert set(gemma_plan.provenance["rows_model"]) == {"train", "replay"}
    assert gemma_plan.provenance["rows_model"]["replay"]["completion_format"] == fmt.record()
    assert gemma_plan.preset.module_scope == "model.language_model."
    assert gemma_plan.provenance["preset_expected_modules"] == 13  # 2 layers x 7, less the full layer's v_proj
    assert gemma_plan.provenance["identity"]["preset"]["module_scope"] == "model.language_model."
    assert gemma_plan.probe_modules == tn.default_probe_modules(2, "model.language_model.")
    assert gemma_plan.probe_plain_modules == ["model.language_model.norm"]
    assert (gemma_plan.truncation.top_k, gemma_plan.truncation.top_p) == (64, 0.95)
    assert gemma_plan.grpo_kwargs["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["Gemma4TextDecoderLayer"]


def test_a_template_that_closes_the_reasoning_itself_is_refused(gemma_model):
    from ruleforge.judges.completion_format import load_training_format

    tokenizer, fmt = load_training_format(gemma_model)
    rows, think_end = [_grpo_row(0), _grpo_row(1)], fmt.think_end_token_id
    on = tn.prompt_token_counts(rows, tokenizer, {"enable_thinking": True}, think_end_token_id=think_end)
    with pytest.raises(tn.TrainerConfigError, match="prompt 0, rendered with .*'enable_thinking': False.*already holds"):
        tn.prompt_token_counts(rows, tokenizer, {"enable_thinking": False}, think_end_token_id=think_end)
    off = tn.prompt_token_counts(rows, tokenizer, {"enable_thinking": False})
    assert len(off) == len(on) == 2 and off != on


def test_a_gemma_trainer_masks_after_the_cap_and_anchors_its_answer_segment(gemma_plan):
    trainer = tn.build_trainer(gemma_plan, grpo_overrides=CPU)
    base = trainer.model.get_base_model()
    assert type(base).__name__ == "Gemma4ForConditionalGeneration"
    assert trainer.think_end_token_id == gemma_plan.completion_format.think_end_token_id
    assert trainer.check_lora_coverage(13) == 13
    # TRL's log-probs are taken on the capped logits, masked: the hook sits on the model, not its head
    assert trainer.truncation_mask._handle.id in base._forward_hooks
    from ruleforge.grpo.truncation import mask_logits

    ids = torch.tensor([[5, 6, 7]])
    with torch.no_grad():  # a greedy continuation: every token in the kept set
        for _ in range(3):
            ids = torch.cat([ids, base(input_ids=ids).logits[:, -1:].argmax(-1)], dim=1)
    logps = trainer._get_per_token_logps_and_entropies(trainer.model, ids, torch.ones_like(ids), 3)[0]
    with torch.no_grad():
        capped = base(input_ids=ids).logits[:, -4:-1].float()
    targets = ids[:, -3:, None]
    expected = torch.log_softmax(mask_logits(capped, gemma_plan.truncation), dim=-1).gather(-1, targets)[..., 0]
    untruncated = torch.log_softmax(capped, dim=-1).gather(-1, targets)[..., 0]
    torch.testing.assert_close(logps.float(), expected, rtol=0, atol=2e-2)
    assert bool((expected > untruncated + 1e-3).all())  # renormalized over the kept set of the capped logits
    # the anchor's base pass (adapter off) equals the policy at a zero adapter, and its answer segment is found
    trainer.anchor_manager.start(0)
    scaled, raw, answer = trainer._anchor_micro_step(trainer.model)
    assert float(raw) == 0.0 and answer[1] > 0


def test_an_anchored_trainer_takes_the_models_think_end_id(gemma_plan):
    args = tn.build_grpo_config({**gemma_plan.grpo_kwargs, **CPU})
    with pytest.raises(tn.TrainerConfigError, match="anchored run takes the model's think-end id"):
        tn.anchored_grpo_trainer_class()(args=args, truncation=gemma_plan.truncation, anchor_rows=gemma_plan.replay,
                                         anchor=gemma_plan.anchor)


def test_the_fsdp2_sync_walk_pushes_what_the_unsharded_sync_pushes(gemma_model):
    import peft
    from trl.generation.vllm_generation import VLLMGeneration

    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(str(gemma_model), dtype=torch.float32)
    model = peft.get_peft_model(model, tn.lora_preset("Gemma4ForConditionalGeneration").lora_config())
    generation = object.__new__(VLLMGeneration)
    generation.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    tn.install_fsdp2_sync(generation)
    pushed = dict(generation._iter_fsdp2_params(model))
    unsharded = {name.removeprefix("base_model.model.").replace(".base_layer", ""): parameter
                 for name, parameter in model.named_parameters() if "lora_" not in name}
    # the state dict also names the tied head, which TRL's walk pushes too and vLLM's Gemma 4 skips when tied
    assert set(pushed) == set(unsharded) | {"lm_head.weight"}
    assert "model.language_model.layers.0.self_attn.q_proj.weight" in pushed
    assert all(torch.equal(pushed[name], unsharded[name]) for name in unsharded)
    assert torch.equal(pushed["lm_head.weight"], unsharded["model.language_model.embed_tokens.weight"])
    buffers = [name for name, _ in model.named_buffers() if name.endswith("layer_scalar")]
    assert len(buffers) == 2 and not any(name.endswith("layer_scalar") for name in pushed)
    # a module-to-save's original copy is left out, its trained copy pushed under the module's own name
    saved = torch.nn.Module()
    saved.head = torch.nn.Module()
    saved.head.original_module = torch.nn.Linear(1, 1, bias=False)
    saved.head.modules_to_save = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 1, bias=False)})
    [(name, tensor)] = tn.fsdp2_engine_tensors(generation, saved)
    assert name == "head.weight" and torch.equal(tensor, saved.head.modules_to_save["default"].weight)


def test_each_step_records_its_time_and_each_ranks_memory(monkeypatch):
    trainer = _full_stub(accelerator=types.SimpleNamespace(
        device=torch.device("cpu"), gather=lambda tensor: torch.cat([tensor, tensor + 1.0])))
    trainer.telemetry["resources"] = {}
    trainer._device_used_gb = 0.0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer._record_resources(3.0)
    metrics = trainer._metrics["train"]
    assert metrics["resources/step_seconds"] == [4.0]  # the slowest rank's: max(3, 4) seconds
    assert metrics["resources/device_used_gb/rank1"] == [1.0] and metrics["resources/torch_reserved_gb/rank0"] == [0.0]
    gb = tn.GB
    assert gb == 1e9
    free = iter([70 * gb, 20 * gb])
    reset = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (next(free), 80 * gb))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 30 * gb)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: reset.append(True))
    trainer._on_generation_check = tn.make_anchored_grpo_trainer_class(_FullStub)._on_generation_check.__get__(trainer)
    trainer.telemetry["generation_checks"] = 0
    trainer._on_generation_check()  # the engine awake: 60 GB used
    trainer._record_resources(5.0)  # the step's end: 10 GB used; the maximum is kept
    assert metrics["resources/device_used_gb/rank0"][-1] == 60.0 and metrics["resources/torch_reserved_gb/rank0"][-1] == 30.0
    assert trainer.telemetry["resources"] == {
        "step_seconds": [4.0, 6.0], "device_used_gb_max": [60.0, 61.0], "torch_reserved_gb_max": [30.0, 31.0]}
    assert reset == [True] and trainer._device_used_gb == 0.0


def test_the_step_timer_hands_each_step_to_the_trainer():
    recorded = []
    callback = tn.make_anchored_grpo_trainer_class(_FullStub).resource_steps(
        types.SimpleNamespace(_record_resources=recorded.append))
    callback.on_step_end(None, None, None)
    callback.on_step_begin(None, None, None)
    callback.on_step_end(None, None, None)
    assert len(recorded) == 1 and recorded[0] >= 0.0 and callback.started is None
