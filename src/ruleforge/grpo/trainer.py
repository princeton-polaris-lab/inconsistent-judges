from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Optional, Protocol, Sequence

from ruleforge.grpo import fsdp2_state
from ruleforge.grpo.kl_anchor import (
    AnchorAuxConfig,
    AnchorReplayManager,
    ReplayRows,
    anchor_kl_loss,
    answer_segment_labels,
    weighted_anchor_loss,
)
from ruleforge.judges.completion_format import CompletionFormat, load_training_format
from ruleforge.grpo.primitives import should_run_anchor_step
from ruleforge.grpo.lora_parity import (
    _PEFT_PREFIX,
    ADAPTER_CONFIG_FNAME,
    ADAPTER_WEIGHTS_FNAME,
    DEFAULT_TARGET_MODULES,
    _package_version,
    checkpoint_class,
)
from ruleforge.grpo.lora_targets import peft_target_modules
from ruleforge.grpo.leakage import LeakageError, disjoint_marker_path
from ruleforge.grpo.run_files import (
    PROVENANCE_FNAME,
    STOP_FNAME,
    read_stopped,
    stop_marker_sha256,
    write_stopped,
)
from ruleforge.grpo.sync_probe import (
    ProbeResult,
    RetentionAlarm,
    SyncProbe,
    SyncProbeError,
    _whole,
    checked_generate,
    probed_sync,
)
from ruleforge.grpo.truncation import (
    Truncation,
    TruncationMask,
    logits_module,
    outside_kept_share,
    zero_outside_logps,
)
from ruleforge.core.errors import ArtifactRootError, RuleForgeError
from ruleforge.core.record_names import WANDB_RUN_KIND
from ruleforge.env import artifact_root
from ruleforge.core.digests import canonical_digest, sha256_file
from ruleforge.core.artifacts import load_jsonl_records, write_text_atomically
from ruleforge.judges.models import UNTRUNCATED_SAMPLING, model_default_truncation

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = 1
#: Retention alarms, one JSON object per line, under the output directory.
ALARMS_FNAME = "alarms.jsonl"
#: The resource telemetry's unit: decimal gigabytes.
GB = 1e9

REWARD_FUNC_NAMES = ("flat_soft_reward", "internal_consistency_reward", "map_completeness_reward")
#: The reward set (`module:attribute`), called as `build_reward_set` documents.
REWARD_FACTORY = "ruleforge.grpo.rewards:GrpoReward"
#: The marker check (`module:attribute`): `(dataset, marker_path) -> marker`, raising on refusal.
MARKER_CHECKER = "ruleforge.grpo.exclusion:check_prompt_marker"

#: GRPOConfig fields the run depends on: each must be set in the config.
REQUIRED_GRPO_FIELDS = (
    "learning_rate", "lr_scheduler_type", "warmup_steps", "optim", "weight_decay", "max_grad_norm",
    "beta", "loss_type", "scale_rewards", "multi_objective_aggregation", "epsilon", "num_iterations",
    "importance_sampling_level", "num_generations", "per_device_train_batch_size",
    "gradient_accumulation_steps", "max_completion_length", "mask_truncated_completions", "use_vllm",
    "vllm_mode", "vllm_tensor_parallel_size", "vllm_gpu_memory_utilization", "vllm_enable_sleep_mode",
    "vllm_importance_sampling_correction", "vllm_importance_sampling_mode",
    "vllm_importance_sampling_clip_max", "gradient_checkpointing", "gradient_checkpointing_kwargs", "bf16",
    "max_steps", "save_strategy", "save_steps", "save_total_limit", "logging_steps", "seed",
    "shuffle_dataset", "remove_unused_columns", "chat_template_kwargs", "use_bias_correction_kl",
    "steps_per_generation", "epsilon_high", "delta", "vllm_importance_sampling_clip_min", "top_entropy_quantile",
    "off_policy_mask_threshold",
)
#: GRPOConfig fields the trainer derives; a config that sets one is refused.
DERIVED_GRPO_FIELDS = (
    "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "generation_kwargs", "reward_weights",
    "model_init_kwargs", "output_dir", "vllm_max_model_length", "run_name",
)
STACK_PACKAGES = ("torch", "transformers", "trl", "peft", "accelerate", "vllm", "tokenizers", "datasets")
#: Configuration keys that may change across a resume.
RESUME_FREE_KEYS = (
    "resume_from_checkpoint", "preflight_only", "output_root", "output_dir", "run_name", "hydra", "wandb",
)
#: Config keys and identity digests only a recorded run holds; they do not bear on training.
RETIRED_KEYS = ("prereg",)
RETIRED_DIGESTS = ("prereg_sha256",)
#: Config paths a resume may spell otherwise, naming the same file (the content is pinned apart).
RESPELLED_PATHS = (("data", "rules"),)
RETIRED_SCRIPT_PREFIX = "../../"
#: The characters W&B keeps in a run id as written, at most `WANDB_ID_MAX` of them.
WANDB_ID_KEPT = re.compile(r"[A-Za-z0-9._-]+")
WANDB_ID_MAX = 64
#: Rank 0's record that W&B accepted the run: the evidence a later launch needs to append to it.
WANDB_RUN_FNAME = "wandb_run.json"
#: How many differences a refused resume names.
RESUME_DIFFERENCES_SHOWN = 8
#: The identity digest of the base's weights; a recorded checkpoint that lacks it resumes without it.
WEIGHTS_DIGEST = "model_weights_sha256"
#: The identity digests a warm start's checkpoint must share with the run.
INIT_ADAPTER_DIGESTS = ("model_config_sha256", "generation_config_sha256", "rules_sha256", "anchor_condition_sha256")
_UNSET = object()


class TrainerConfigError(ValueError, RuleForgeError):
    """The run is not configured as the trainer requires; nothing was started."""


@dataclass(frozen=True)
class LoraPreset:
    """The adapter and master precision for one architecture; `module_scope` limits the targets to the modules below a
    module-path prefix (a multimodal checkpoint's language model)."""

    architecture: str
    r: int
    alpha: int
    dropout: float = 0.0
    bias: str = "none"
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    master_dtype: str = "float32"
    module_scope: Optional[str] = None

    def lora_config(self) -> Any:
        from peft import LoraConfig

        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type="CAUSAL_LM",
            target_modules=peft_target_modules(self.target_modules, self.module_scope),
        )

    @property
    def decoder_prefix(self) -> str:
        """Return where the decoder layers' modules live."""
        return self.module_scope or "model."

    def identity(self) -> dict[str, Any]:
        """Return the preset as a checkpoint's identity records it (without `module_scope` where unset)."""
        record = dataclasses.asdict(self)
        if record["module_scope"] is None:
            del record["module_scope"]
        return record

    def model_init_kwargs(self, attn_implementation: str) -> dict[str, Any]:
        """Return TRL's `model_init_kwargs`: `dtype` and the attention kernel."""
        return {"dtype": self.master_dtype, "attn_implementation": attn_implementation}


#: By `config.json`'s `architectures[0]`.
LORA_PRESETS: dict[str, LoraPreset] = {
    "Qwen3ForCausalLM": LoraPreset(architecture="Qwen3ForCausalLM", r=32, alpha=64),
    "Gemma4ForConditionalGeneration": LoraPreset(
        architecture="Gemma4ForConditionalGeneration", r=32, alpha=64, module_scope="model.language_model.",
    ),
}
#: The master dtypes a run may declare; float32 unless declared.
MASTER_DTYPES = ("float32", "bfloat16")


def lora_preset(architecture: str) -> LoraPreset:
    if architecture not in LORA_PRESETS:
        raise TrainerConfigError(f"no LoRA preset for {architecture!r}; presets exist for {sorted(LORA_PRESETS)}")
    return LORA_PRESETS[architecture]


def expected_module_count(config: Any, preset: LoraPreset) -> int:
    """Return the expected wrapped module count: one per target per decoder layer, less the `v_proj` a layer does not
    have."""
    text = config.get_text_config()
    layers = int(text.num_hidden_layers)
    count = len(preset.target_modules) * layers
    if "v_proj" in preset.target_modules and getattr(text, "attention_k_eq_v", False):
        count -= sum(1 for kind in text.layer_types[:layers] if kind == "full_attention")
    return count


def effective_truncation(model_path: Path | str) -> dict[str, float | int]:
    """Return the sampler truncation a campaign would record for this model directory."""
    return {**UNTRUNCATED_SAMPLING, **model_default_truncation(str(model_path))}


def _same_numbers(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return set(first) == set(second) and all(float(first[key]) == float(second[key]) for key in first)


def check_anchor_condition(
    condition: Mapping[str, Any], *, model_path: Path | str, rules_sha256: str
) -> Truncation:
    """Return the rollout truncation, checked against the train anchor campaign's recorded condition: the same model
    directory with no adapter, this ruleset's digest and an equal `sampling_truncation`."""
    effective = effective_truncation(model_path)
    recorded = condition.get("sampling_truncation")
    problems = []
    if not isinstance(recorded, Mapping) or not _same_numbers(recorded, effective):
        problems.append(f"it recorded sampling_truncation {recorded!r}, the model ships {effective}")
    if Path(str(condition.get("model_path", ""))).name != Path(model_path).name:
        problems.append(f"it served {condition.get('model_path')!r}, not {Path(model_path).name!r}")
    if condition.get("adapter_path") is not None:
        problems.append(f"it served adapter {condition.get('adapter_path')!r}; the anchors are the base's")
    if condition.get("rule_set_sha256") != rules_sha256:
        problems.append(f"its ruleset digest {condition.get('rule_set_sha256')!r} is not {rules_sha256!r}")
    temperature = condition.get("temperature")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        problems.append(f"it records no numeric temperature ({temperature!r})")
    if problems:
        raise TrainerConfigError("the train anchor condition does not match this run: " + "; ".join(problems))
    return Truncation.from_sampling(effective, temperature=float(temperature))


class RewardSet(Protocol):
    """What the trainer needs of the reward set: its TRL reward functions, their weights and the token ids that end a
    completion."""

    @property
    def reward_funcs(self) -> list[Callable[..., list[Optional[float]]]]: ...

    @property
    def reward_weights(self) -> list[float]: ...

    @property
    def finish_token_ids(self) -> Sequence[int]: ...


def import_object(spec: str) -> Any:
    """Import `module:attribute`."""
    module, _, attribute = spec.partition(":")
    if not module or not attribute:
        raise TrainerConfigError(f"{spec!r} is not module:attribute")
    return getattr(importlib.import_module(module), attribute)


def build_reward_set(
    *,
    rules: Sequence[Mapping[str, Any]],
    truncated: str,
    num_generations: int,
    lambda_int: float,
    lambda_comp: float,
    factory: Optional[Callable[..., RewardSet]] = None,
    completion_format: Optional[CompletionFormat] = None,
    favoured_no_rate: bool = False,
) -> RewardSet:
    """Build the reward set from `factory` with the ruleset, the truncated policy, `num_generations`, the two lambdas
    and, when given, the completion format and the favoured-no rate switch."""
    if factory is None:
        try:
            factory = import_object(REWARD_FACTORY)
        except (ImportError, AttributeError) as error:
            raise TrainerConfigError(
                f"the reward set {REWARD_FACTORY} is not importable ({error}); "
                "ruleforge.grpo.rewards must be importable before a run can start"
            ) from error
    extra = {**({} if completion_format is None else {"completion_format": completion_format}),
             **({"favoured_no_rate": True} if favoured_no_rate else {})}
    reward_set = factory(
        rules, truncated=truncated, num_generations=num_generations, lambda_int=lambda_int, lambda_comp=lambda_comp,
        **extra,
    )
    check_reward_set(reward_set, lambda_int=lambda_int, lambda_comp=lambda_comp)
    return reward_set


REWARD_KEYS = ("truncated", "lambda_int", "lambda_comp", "favoured_no_rate")


def configured_reward_set(
    reward: Mapping[str, Any],
    *,
    rules: Sequence[Mapping[str, Any]],
    num_generations: int,
    factory: Optional[Callable[..., RewardSet]] = None,
    completion_format: Optional[CompletionFormat] = None,
) -> RewardSet:
    """Return `build_reward_set` from a run's `reward` block; a key outside `REWARD_KEYS` is refused."""
    unknown = sorted(set(reward) - set(REWARD_KEYS))
    if unknown:
        raise TrainerConfigError(f"the reward block names {', '.join(f'reward.{key}' for key in unknown)}, which no "
                                 f"reward reads: its keys are {', '.join(REWARD_KEYS)}")
    return build_reward_set(
        rules=rules,
        truncated=str(reward["truncated"]),
        num_generations=num_generations,
        lambda_int=float(reward["lambda_int"]),
        lambda_comp=float(reward["lambda_comp"]),
        factory=factory,
        completion_format=completion_format,
        favoured_no_rate=bool(reward.get("favoured_no_rate", False)),
    )


def reward_form(reward_set: RewardSet) -> str:
    """Return what the plan line names of the reward: its form and each function at its weight."""
    names = _func_names(reward_set)
    terms = " + ".join(f"{float(weight):g} {name}" for weight, name in zip(reward_set.reward_weights, names))
    return f"sum ({terms})"


def _func_names(reward_set: RewardSet) -> tuple[str, ...]:
    return tuple(getattr(func, "__name__", type(func).__name__) for func in reward_set.reward_funcs)


def rows_built_on(rows: Path | str, model_path: Path, *, sha256: Optional[str] = None,
                  weights_sha256: Optional[str] = None) -> dict[str, Any]:
    """Return `check_rows_model`'s comparison, refusing rows whose build manifest records another model than the one
    trained, or rows without a manifest."""
    from ruleforge.judges.completion_format import GrpoDataError
    from ruleforge.grpo.grpo_data import check_rows_model

    try:
        return check_rows_model(rows, model_path, sha256=sha256, weights_sha256=weights_sha256)
    except GrpoDataError as error:
        raise TrainerConfigError(str(error)) from error


def recorded_condition(named: str) -> list[Path]:
    """Return where the anchor condition a rows build records is now: its recorded path, else the same artifact-relative
    path under `$RULEFORGE_ARTIFACT_ROOT`, else nowhere."""
    if Path(named).is_file():
        return [Path(named)]
    try:
        root = artifact_root()
    except ArtifactRootError:
        return []
    parts = PurePosixPath(named).parts
    return [root.joinpath(*parts[start:]) for start, part in enumerate(parts)
            if part == "sampling" and root.joinpath(*parts[start:]).is_file()]


def check_built_condition(built: Mapping[str, Any], condition_path: Path) -> None:
    """Refuse an anchor condition that is not the one the rows build names, compared by content."""
    named = built.get("anchor_condition")
    if named is None:
        return
    given = sha256_file(condition_path)
    if not any(sha256_file(path) == given for path in recorded_condition(named)):
        raise TrainerConfigError(f"data.anchor_condition {condition_path} is not the condition the rows were built "
                                 f"from ({named}, their manifest's anchor_condition, at that path or under "
                                 "$RULEFORGE_ARTIFACT_ROOT by its sampling/ path): train under that one")


def check_reward_set(reward_set: RewardSet, *, lambda_int: float, lambda_comp: float) -> None:
    """Check the reward set holds the three functions in order, weighted `[1, λ_i, λ_c]`."""
    names = _func_names(reward_set)
    if names != REWARD_FUNC_NAMES:
        raise TrainerConfigError(f"the reward functions are {list(names)}, not {list(REWARD_FUNC_NAMES)}")
    weights = [float(weight) for weight in reward_set.reward_weights]
    expected_weights = [1.0, float(lambda_int), float(lambda_comp)]
    if weights != expected_weights:
        raise TrainerConfigError(f"the reward weights are {weights}, not {expected_weights}")


def check_finish_tokens(reward_set: RewardSet, tokenizer: Any) -> None:
    """Check the reward's finish tokens are TRL's (the tokenizer's eos and pad), so 'truncated' means one thing."""
    trl_finish = {int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)}
    if {int(token) for token in reward_set.finish_token_ids} != trl_finish:
        raise TrainerConfigError(
            f"the reward ends a completion at {sorted(reward_set.finish_token_ids)}, TRL at the tokenizer's eos "
            f"and pad {sorted(trl_finish)}: a truncated completion must be one for both"
        )


def require_disjoint_marker(
    dataset: Path | str, *, checker: Optional[Callable[..., Mapping[str, Any]]] = None,
    allow_unscreened: bool = False, **requirements: Sequence[str],
) -> dict[str, Any]:
    """Return the prompt-disjointness marker beside `dataset`, validated for its bytes and against the files
    `requirements` name (`required_protected`, `required_carved`: digests); refuse otherwise, and refuse a marker of the
    second pass alone unless `allow_unscreened`."""
    dataset = Path(dataset)
    marker_path = disjoint_marker_path(dataset)
    if not marker_path.is_file():
        raise LeakageError(
            f"{dataset} carries no disjointness marker ({marker_path.name}); training data must be shown "
            "prompt-disjoint from the protected sets before a run may read it"
        )
    if checker is None:
        try:
            checker = import_object(MARKER_CHECKER)
        except (ImportError, AttributeError) as error:
            raise LeakageError(
                f"the marker check {MARKER_CHECKER} is not importable ({error}); the exclusion module "
                "must be importable to validate the marker, and the run refuses without it"
            ) from error
    options = {**({"allow_unscreened": True} if allow_unscreened else {}),
               **{key: list(values) for key, values in requirements.items()}}
    marker = checker(dataset, marker_path, **options)
    record = {
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "marker": str(marker_path),
        "marker_sha256": sha256_file(marker_path),
        "marker_kind": marker.get("kind"),
    }
    if allow_unscreened:
        record["allow_unscreened_rows"] = True
        record["screen"] = marker.get("screen", "full")
    return record


def check_prompt_shape(rows: Sequence[Mapping[str, Any]]) -> None:
    """Check every GRPO prompt is `[system: P, user: ""]`."""
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        ok = (
            isinstance(prompt, list)
            and len(prompt) == 2
            and all(isinstance(message, Mapping) for message in prompt)
            and prompt[0].get("role") == "system"
            and isinstance(prompt[0].get("content"), str)
            and prompt[0]["content"].strip() != ""
            and prompt[1].get("role") == "user"
            and prompt[1].get("content") == ""
        )
        if not ok:
            raise TrainerConfigError(
                f"GRPO row {index}'s prompt is not [system: <prompt>, user: \"\"], the rendering every "
                "campaign condition uses"
            )


def prompt_token_counts(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    chat_template_kwargs: Mapping[str, Any],
    *,
    think_end_token_id: Optional[int] = None,
) -> list[int]:
    """Return each prompt's length in tokens as TRL renders it; a rendered prompt holding `think_end_token_id` is
    refused."""
    counts = []
    for index, row in enumerate(rows):
        ids = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True, **chat_template_kwargs
        )["input_ids"]
        if think_end_token_id is not None and think_end_token_id in ids:
            raise TrainerConfigError(
                f"GRPO prompt {index}, rendered with chat_template_kwargs {dict(chat_template_kwargs)}, already holds "
                f"the think-end id {think_end_token_id}: the template closes the reasoning itself (thinking off?), so "
                "every completion would be read as unclosed_think"
            )
        counts.append(len(ids))
    return counts


def check_prompt_lengths(counts: Sequence[int], cap: int) -> None:
    long = [index for index, count in enumerate(counts) if count > cap]
    if long:
        raise TrainerConfigError(
            f"{len(long)} GRPO prompt(s) exceed the {cap}-token cap (e.g. row {long[0]}, {counts[long[0]]} tokens); "
            "the data build filters to the cap"
        )


def grpo_config_kwargs(
    grpo: Mapping[str, Any],
    *,
    truncation: Truncation,
    reward_weights: Sequence[float],
    model_init_kwargs: Mapping[str, Any],
    output_dir: Path | str,
    vllm_length_margin: int,
    anchored: bool,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Return `(GRPOConfig keyword arguments, fields forced by the anchor)`."""
    derived = sorted(set(grpo) & set(DERIVED_GRPO_FIELDS))
    if derived:
        raise TrainerConfigError(f"grpo.{derived[0]} is derived by the trainer and may not be set ({derived})")
    missing = [name for name in REQUIRED_GRPO_FIELDS if name not in grpo]
    if missing:
        raise TrainerConfigError(
            f"grpo leaves {missing} unset; every field GRPO training depends on is set explicitly (TRL 1.13's "
            "defaults differ from 1.9.2's)"
        )
    if "tensorboard" in report_targets(grpo.get("report_to")):
        raise TrainerConfigError(
            "grpo.report_to names tensorboard, which is not supported: the grpo group does not "
            "install it, and HF's TensorBoardCallback would fail on every rank once the model had loaded; report to "
            "wandb"
        )
    fsdp_config = grpo.get("fsdp_config") or {}
    # transformers reads a key with or without its fsdp_ prefix
    loading = fsdp_config.get("cpu_ram_efficient_loading", fsdp_config.get("fsdp_cpu_ram_efficient_loading", False))
    fsdp2 = bool(grpo.get("fsdp")) and fsdp2_state.configured_fsdp_version(fsdp_config) == 2
    if fsdp2 and grpo.get("use_adaptive_entropy"):
        raise TrainerConfigError(
            "grpo.use_adaptive_entropy under FSDP2: the sharded resume state restores the adapter, the "
            "optimizer and the scheduler in place of HF's, and TRL 1.13 restores its entropy controller "
            "(entropy_ctrl_state.json) in the HF load that path replaces, so a resumed run would restart the "
            "coefficient with no error"
        )
    if grpo.get("fsdp") and str(loading).lower() == "true":
        raise TrainerConfigError(
            "grpo.fsdp_config.cpu_ram_efficient_loading: accelerate 1.13 then loads rank 0's weights into the sharded "
            "model as if every state-dict entry were a DTensor, and a persistent buffer (Gemma 4's layer_scalar) is "
            "not; nor does it save host memory at load under transformers 5.17, where every rank reads the whole "
            "checkpoint. Under FSDP2 the trainer loads the master on rank 0 alone itself"
        )
    kwargs = dict(grpo)
    forced: dict[str, list[Any]] = {}
    if anchored:
        # the anchor replaces TRL's KL, and truncated completions carry no gradient
        for name, value in (("beta", 0.0), ("mask_truncated_completions", True)):
            if kwargs[name] != value:
                forced[name] = [kwargs[name], value]
                logger.warning("anchored run: forcing grpo.%s from %r to %r", name, kwargs[name], value)
                kwargs[name] = value
    kwargs.update(
        temperature=truncation.temperature,
        top_k=truncation.top_k,
        top_p=truncation.top_p,
        min_p=truncation.min_p,
        repetition_penalty=truncation.repetition_penalty,
        generation_kwargs=None,
        reward_weights=[float(weight) for weight in reward_weights],
        model_init_kwargs=dict(model_init_kwargs),
        output_dir=str(output_dir),
        vllm_max_model_length=int(kwargs["max_completion_length"]) + int(vllm_length_margin),
        run_name=Path(output_dir).name,
    )
    return kwargs, forced


def check_whole_groups(per_device_train_batch_size: int, steps_per_generation: int, num_generations: int) -> None:
    """Check each rank's generation batch is whole groups of `num_generations`."""
    local = per_device_train_batch_size * steps_per_generation
    if local % num_generations:
        raise TrainerConfigError(
            f"each rank generates {per_device_train_batch_size} x {steps_per_generation} = {local} completions, "
            f"not a multiple of num_generations {num_generations}: a group would straddle two ranks"
        )


def build_grpo_config(kwargs: Mapping[str, Any]) -> Any:
    from trl import GRPOConfig

    args = GRPOConfig(**kwargs)
    check_whole_groups(args.per_device_train_batch_size, args.steps_per_generation, args.num_generations)
    return args


def export_submitit_distributed_env(environ: MutableMapping[str, str]) -> Optional[dict[str, str]]:
    """Export torch.distributed's env:// variables from a multi-task submitit job; None when not needed."""
    if "WORLD_SIZE" in environ:
        return None
    try:
        from submitit import JobEnvironment

        job = JobEnvironment()
    except RuntimeError:
        return None
    if job.num_tasks <= 1:
        return None
    port = environ.get("MASTER_PORT") or str(random.Random(str(job.job_id)).randint(20000, 60000))
    values = {
        "RANK": str(job.global_rank),
        "LOCAL_RANK": str(job.local_rank),
        "WORLD_SIZE": str(job.num_tasks),
        "LOCAL_WORLD_SIZE": str(job.num_tasks // job.num_nodes),
        "MASTER_ADDR": job.hostnames[0],
        "MASTER_PORT": port,
    }
    environ.update(values)
    return values


def report_targets(report_to: Any) -> list[str]:
    """Return `grpo.report_to` as the list of integrations HF reads it as."""
    if report_to is None:
        return []
    targets = [report_to] if isinstance(report_to, str) else [str(name) for name in report_to]
    return [] if targets == ["none"] else targets


def wandb_run_id(run: str) -> str:
    """Return the W&B run id of the training run `run`: the name where W&B takes it as written, else its kept characters
    plus a digest suffix."""
    if WANDB_ID_KEPT.fullmatch(run) and len(run) <= WANDB_ID_MAX:
        return run
    digest = hashlib.sha256(run.encode("utf-8")).hexdigest()[:8]
    kept = "-".join(WANDB_ID_KEPT.findall(run))[: WANDB_ID_MAX - len(digest) - 1]
    return f"{kept}-{digest}" if kept else digest


def wandb_resume(
    output_dir: Path, resume_from: Optional[Path], *, project: Optional[str], entity: Optional[str],
) -> str:
    """Return W&B's `resume` for this launch: `allow` only on evidence that a launch of this run opened the W&B run (a
    checkpoint, rank 0's marker, a resumed checkpoint of the same name), else `never`."""
    run_id = wandb_run_id(output_dir.name)
    opened = (
        bool(checkpoint_dirs(output_dir))
        or (resume_from is not None and wandb_run_id(resume_from.absolute().parent.name) == run_id)
        or wandb_marker_id(output_dir, project=project, entity=entity) == run_id
    )
    return "allow" if opened else "never"


def write_wandb_marker(output_dir: Path, run: Any, *, at: str) -> Optional[Path]:
    """Record in `output_dir` that W&B accepted the run (id, project, entity, time); return the marker's path, or None
    when no run was open."""
    if run is None or run.offline or run.disabled:
        return None
    path = Path(output_dir) / WANDB_RUN_FNAME
    record = {"kind": WANDB_RUN_KIND, "id": run.id, "project": run.project, "entity": run.entity, "opened_at": at}
    write_text_atomically(path, json.dumps(record, indent=1, sort_keys=True) + "\n")
    return path


def wandb_marker_id(output_dir: Path, *, project: Optional[str], entity: Optional[str]) -> Optional[str]:
    """Return the W&B run id rank 0's marker names when it was written for a run in `project` (and `entity`), else None.
    """
    try:
        marker = json.loads((output_dir / WANDB_RUN_FNAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(marker, dict) or marker.get("kind") != WANDB_RUN_KIND:
        return None
    if project is None or marker.get("project") != project or entity not in (None, marker.get("entity")):
        return None
    return marker.get("id")


def wandb_env(
    settings: Optional[Mapping[str, Any]], *, output_dir: Path | str, report_to: Any, resume: str,
) -> dict[str, str]:
    """Return the environment W&B reads for the run (project, entity, group, run name and id, resume mode, files under
    `output_dir/wandb`); empty for a run that does not report to W&B."""
    targets = report_targets(report_to)
    if "wandb" not in targets and "all" not in targets:
        return {}
    block = dict(settings or {})
    missing = [key for key in ("project", "group") if not block.get(key)]
    if missing:
        raise TrainerConfigError(
            f"grpo.report_to names wandb, and the config's wandb block sets no {', '.join(missing)}"
        )
    env = {
        "WANDB_PROJECT": str(block["project"]),
        "WANDB_RUN_GROUP": str(block["group"]),
        "WANDB_RUN_ID": wandb_run_id(Path(output_dir).name),
        "WANDB_RESUME": resume,
        "WANDB_DIR": str(output_dir),
        "WANDB_SYMLINK": "false",
    }
    if block.get("entity") is not None:
        env["WANDB_ENTITY"] = str(block["entity"])
    import wandb

    try:
        wandb.Settings().update_from_env_vars(env)
    except (ValueError, wandb.errors.UsageError) as error:
        raise TrainerConfigError(f"W&B refuses the run's settings {env}: {error}") from None
    return env


def export_wandb_env(environ: MutableMapping[str, str], settings: Mapping[str, str], *, rank: int) -> dict[str, str]:
    """Export the run's W&B settings into `environ` on every rank, with W&B disabled on every rank but 0; return what was
    exported."""
    if not settings:
        return {}
    values = {**settings, **({} if rank == 0 else {"WANDB_MODE": "disabled"})}
    environ.update(values)
    return values


def wandb_completions_table(settings: Optional[Mapping[str, Any]]) -> bool:
    """Return whether the run uploads TRL's per-step completions table to W&B (`wandb.log_completions_table`, off unless
    set)."""
    value = (settings or {}).get("log_completions_table", False)
    if not isinstance(value, bool):
        raise TrainerConfigError(f"wandb.log_completions_table is true or false, got {value!r}")
    return value


def stack_versions(repo_root: Optional[Path] = None) -> dict[str, Optional[str]]:
    """Return the installed versions of `STACK_PACKAGES` and the `uv.lock` digest."""
    from ruleforge.core.provenance import repo_root as default_root

    lock = Path(repo_root or default_root()) / "uv.lock"
    versions = {name: _package_version(name) for name in STACK_PACKAGES}
    versions["uv_lock_sha256"] = sha256_file(lock) if lock.is_file() else None
    return versions


def resume_free(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return `config` without the keys a resume may change and without a key at its inert value
    (a null `init_adapter`; `reward.lambda_prec` and `reward.lambda_just` at 0; a false
    `data.allow_unscreened_rows`)."""
    free = {key: value for key, value in config.items() if key not in RESUME_FREE_KEYS}
    for block, key, inert in (("model", "init_adapter", None), ("reward", "lambda_prec", 0),
                              ("reward", "lambda_just", 0), ("data", "allow_unscreened_rows", False)):
        section = free.get(block)
        if isinstance(section, Mapping) and key in section and section[key] == inert:
            free[block] = {name: value for name, value in section.items() if name != key}
    return free


def without_retired(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return `config` without the keys only a recorded run holds."""
    return {key: value for key, value in config.items() if key not in RETIRED_KEYS}


def respelled_only(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return whether two configs are one run's but for how `RESPELLED_PATHS` spell their files."""
    def from_top(config: Mapping[str, Any]) -> Any:
        free = without_retired(json.loads(json.dumps(resume_free(config), default=str)))
        for block, key in RESPELLED_PATHS:
            section = free.get(block)
            value = section.get(key) if isinstance(section, dict) else None
            if isinstance(value, str) and value.startswith(RETIRED_SCRIPT_PREFIX):
                section[key] = value[len(RETIRED_SCRIPT_PREFIX):]
        return free

    return from_top(recorded) == from_top(current)


def run_identity(
    *, config: Mapping[str, Any], digests: Mapping[str, Any], preset: LoraPreset, truncation: Truncation,
    stack: Mapping[str, Any],
) -> dict[str, Any]:
    """Return what a checkpoint may only be resumed under, in its JSON form."""
    return json.loads(json.dumps({
        "config_sha256": canonical_digest(resume_free(config)),
        "digests": dict(digests),
        "preset": preset.identity(),
        "truncation": dataclasses.asdict(truncation),
        "stack": dict(stack),
    }))


def checkpoint_dirs(output_dir: Path) -> list[Path]:
    """Return the `checkpoint-<step>` directories under `output_dir`, newest first."""
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    prefix = f"{PREFIX_CHECKPOINT_DIR}-"
    found = [
        path for path in (output_dir.iterdir() if output_dir.is_dir() else ())
        if path.is_dir() and path.name.startswith(prefix) and path.name[len(prefix):].isdigit()
    ]
    return sorted(found, key=lambda path: int(path.name[len(prefix):]), reverse=True)


def resolve_resume(value: Optional[str], output_dir: Path) -> Optional[Path]:
    """Return the checkpoint to resume from: None, the newest complete checkpoint for `latest` (an incomplete newer one
    is skipped with a warning), or a directory."""
    existing = checkpoint_dirs(output_dir)
    if value is None:
        if existing:
            raise TrainerConfigError(
                f"{output_dir} already holds {existing[0].name}; a fresh run would mix with it. Set "
                "resume_from_checkpoint=latest (or a checkpoint), or another run_name"
            )
        return None
    if value == "latest":
        for checkpoint in existing:
            if (checkpoint / PROVENANCE_FNAME).is_file():
                return checkpoint
            logger.warning("skipping %s: it has no %s (a save that did not finish)", checkpoint, PROVENANCE_FNAME)
        return None
    path = Path(value)
    if not (path / PROVENANCE_FNAME).is_file():
        raise TrainerConfigError(f"{path} is not a checkpoint this trainer wrote (no {PROVENANCE_FNAME})")
    return path


def batch_geometry(args: Any) -> dict[str, int]:
    """Return the ranks and the global generation batch of a GRPOConfig."""
    return {"world_size": int(args.world_size), "generation_batch_size": int(args.generation_batch_size)}


def check_resume_geometry(record: Mapping[str, Any], args: Any) -> None:
    """Refuse a resume on another number of ranks than the checkpoint trained on."""
    recorded, current = record.get("batch_geometry"), batch_geometry(args)
    if recorded != current:
        raise TrainerConfigError(
            f"the checkpoint was trained at {recorded}, this run is {current}: a resume on another number of "
            "ranks would re-train or skip prompts; resume on the ranks it recorded"
        )


def _shown(value: Any) -> str:
    if value is _UNSET:
        return "unset"
    return f"{value[:12]}..." if isinstance(value, str) and len(value) > 16 else repr(value)


def identity_differences(recorded: Any, current: Any, path: str = "") -> list[str]:
    """Return where two identities (or configs) differ: each dotted key with its recorded value, then this run's."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        return [found for key in sorted(set(recorded) | set(current), key=str)
                for found in identity_differences(recorded.get(key, _UNSET), current.get(key, _UNSET),
                                                  f"{path}.{key}" if path else str(key))]
    return [] if recorded == current else [f"{path} {_shown(recorded)} -> {_shown(current)}"]


def check_resume(
    checkpoint: Path, identity: Mapping[str, Any], *, max_steps: int, config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the checkpoint's provenance when it was written under `identity` before `max_steps`; refuse otherwise,
    naming what differs."""
    record = json.loads((checkpoint / PROVENANCE_FNAME).read_text(encoding="utf-8"))
    run = record.get("run", {})
    recorded = dict(run.get("identity", {}))
    if isinstance(recorded.get("digests"), dict):
        recorded["digests"] = {key: value for key, value in recorded["digests"].items() if key not in RETIRED_DIGESTS}
        if WEIGHTS_DIGEST not in recorded["digests"] and isinstance(identity.get("digests"), Mapping):
            identity = {**identity, "digests": {key: value for key, value in identity["digests"].items()
                                                if key != WEIGHTS_DIGEST}}
            logger.warning("%s records no base weights digest: the base weights are unchecked for this resume",
                           checkpoint)
    differing = sorted(key for key in set(recorded) | set(identity) if recorded.get(key) != identity.get(key))
    if (differing == ["config_sha256"] and config is not None and isinstance(run.get("config"), dict)
            and respelled_only(run["config"], config)):
        differing = []
    if differing:
        found = [item for key in differing if key != "config_sha256"
                 for item in identity_differences(recorded.get(key, _UNSET), identity.get(key, _UNSET), key)]
        if "config_sha256" in differing and config is not None and isinstance(run.get("config"), dict):
            current = json.loads(json.dumps(resume_free(config), default=str))
            found += [f"config {item}" for item in identity_differences(without_retired(resume_free(run["config"])),
                                                                        without_retired(current))]
        more = len(found) - RESUME_DIFFERENCES_SHOWN
        shown = "; ".join(found[:RESUME_DIFFERENCES_SHOWN]) + (f"; {more} more" if more > 0 else "")
        raise TrainerConfigError(
            f"{checkpoint} was trained under another {', '.join(differing)}; a resume continues the same run "
            f"or none{f' (recorded -> this run: {shown})' if found else ''}. Resume it under the code, lock and "
            "config that trained it, or train under another run_name"
        )
    if int(record["step"]) >= max_steps:
        raise TrainerConfigError(
            f"{checkpoint} is at step {record['step']}, already at max_steps {max_steps}: the run is finished, and a "
            "resume from it would train past max_steps. Nothing to resume; train under another run_name for a new run"
        )
    return record


def adapter_digest(checkpoint: Path) -> Optional[str]:
    path = Path(checkpoint) / ADAPTER_WEIGHTS_FNAME
    return sha256_file(path) if path.is_file() else None


def check_init_adapter(
    checkpoint: Path | str, *, model_path: Path, preset: LoraPreset, digests: Mapping[str, Any],
    truncation: Truncation, grpo: Mapping[str, Any],
) -> dict[str, Any]:
    """Return what the run records of a warm start's checkpoint, refused unless its provenance, adapter, preset, model,
    ruleset, anchor condition and truncation are this run's and its resume state can be restored."""
    path = Path(checkpoint)
    provenance_path = path / PROVENANCE_FNAME
    if not provenance_path.is_file():
        raise TrainerConfigError(f"model.init_adapter {path} is not a checkpoint this trainer wrote (no "
                                 f"{PROVENANCE_FNAME})")
    record = json.loads(provenance_path.read_text(encoding="utf-8"))
    run = record.get("run") or {}
    recorded = run.get("identity") or {}
    problems = []
    adapter = adapter_digest(path)
    if adapter is None or adapter != record.get("adapter_sha256"):
        problems.append(f"its {ADAPTER_WEIGHTS_FNAME} ({_shown(adapter)}) is not the adapter its provenance records "
                        f"({_shown(record.get('adapter_sha256'))})")
    found = identity_differences(
        {"preset": recorded.get("preset", _UNSET), "truncation": recorded.get("truncation", _UNSET),
         **{key: (recorded.get("digests") or {}).get(key, _UNSET) for key in INIT_ADAPTER_DIGESTS}},
        json.loads(json.dumps({"preset": preset.identity(), "truncation": dataclasses.asdict(truncation),
                               **{key: digests[key] for key in INIT_ADAPTER_DIGESTS}})))
    if found:
        problems.append(f"it trained under another model or reading (recorded -> this run: {'; '.join(found)})")
    config_path = path / ADAPTER_CONFIG_FNAME
    named = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path") if (
        config_path.is_file()) else None
    if named is None or Path(str(named)).name != model_path.name:
        problems.append(f"its {ADAPTER_CONFIG_FNAME} names base {named!r}, not {model_path.name!r}")
    if not grpo.get("fsdp"):
        problems.append("this run is not under FSDP, and the adapter is loaded from the checkpoint's FSDP2 sharded "
                        "state (grpo.fsdp)")
    else:
        refusal = fsdp2_state.resume_refusal(path, grpo.get("fsdp_config") or {})
        if refusal is not None:
            problems.append(refusal)
    if problems:
        raise TrainerConfigError(f"model.init_adapter {path}: " + "; ".join(problems))
    return {"checkpoint": str(path), "run": (run.get("config") or {}).get("run_name"), "step": int(record["step"]),
            "adapter_sha256": adapter, "provenance_sha256": sha256_file(provenance_path)}


def check_init_geometry(init: Mapping[str, Any], args: Any) -> None:
    """Refuse a warm start on another number of ranks than the init checkpoint's shards were saved on."""
    saved = (fsdp2_state.read_record(init["checkpoint"]) or {}).get("world_size")
    if saved != int(args.world_size):
        raise TrainerConfigError(
            f"model.init_adapter {init['checkpoint']} was saved on {saved} ranks, and this run has {args.world_size}: "
            "its sharded adapter loads on the ranks it was saved on")


def write_provenance(checkpoint: Path, record: Mapping[str, Any]) -> Path:
    path = Path(checkpoint) / PROVENANCE_FNAME
    write_text_atomically(path, json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    return path


def append_alarm(path: Path, alarm: Mapping[str, Any]) -> bool:
    """Append `alarm` to the alarms file unless it already holds that alarm at that step; return whether it wrote."""
    key = (alarm.get("alarm"), alarm.get("step"))
    if path.is_file():
        held = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any((entry.get("alarm"), entry.get("step")) == key for entry in held):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(alarm) + "\n")
    return True


@dataclass
class AnchorSettings:
    """An enabled KL anchor: β 0, masked truncations, the replay stream."""

    config: AnchorAuxConfig
    accum: int
    temperature: float

    def __post_init__(self) -> None:
        if self.config.loss_weight <= 0:
            raise TrainerConfigError(
                f"an enabled anchor has a positive loss_weight, got {self.config.loss_weight}: at 0 the run would "
                "drop TRL's KL (β 0) and add no anchor, training with no KL at all; set anchor.enabled=false"
            )
        if self.accum < 1:
            raise TrainerConfigError(f"anchor.anchor_accum must be positive, got {self.accum}")


@dataclass
class GrpoPlan:
    """A checked run, ready to build."""

    output_dir: Path
    model_path: Path
    preset: LoraPreset
    truncation: Truncation
    tokenizer: Any
    train_rows: list[dict[str, Any]]
    replay: Optional[ReplayRows]
    reward_set: RewardSet
    grpo_kwargs: dict[str, Any]
    forced: dict[str, list[Any]]
    anchor: Optional[AnchorSettings]
    probe_modules: list[str]
    probe_plain_modules: list[str]
    probe_max_rows: int
    alarm: RetentionAlarm
    resume_from: Optional[Path]
    completion_format: CompletionFormat
    provenance: dict[str, Any] = field(default_factory=dict)
    resumed_record: Optional[dict[str, Any]] = None
    #: The W&B settings every rank exports before its trainer exists.
    wandb_env: dict[str, str] = field(default_factory=dict)
    wandb_completions_table: bool = False
    #: The warm start's checkpoint, checked; None for a cold start.
    init_adapter: Optional[dict[str, Any]] = None


def default_probe_modules(num_layers: int, prefix: str = "model.") -> list[str]:
    """Return the default probe modules: one packed q/k/v shard, one packed gate/up shard and one unpacked module."""
    return [
        f"{prefix}layers.0.self_attn.k_proj",
        f"{prefix}layers.{num_layers // 2}.mlp.up_proj",
        f"{prefix}layers.{num_layers - 1}.mlp.down_proj",
    ]


def resolve_preset(model_config: Any, master_dtype: Optional[str]) -> LoraPreset:
    """Return the architecture's preset, at `master_dtype` when the run declares one."""
    preset = lora_preset(model_config.architectures[0])
    if master_dtype is None or master_dtype == preset.master_dtype:
        return preset
    if master_dtype not in MASTER_DTYPES:
        raise TrainerConfigError(f"model.master_dtype must be one of {MASTER_DTYPES}, got {master_dtype!r}")
    logger.warning(
        "model.master_dtype=%s: the sync merges and unmerges the adapter into a %s master in place, which drifts "
        "the base weights over many syncs; for a short mechanics run only", master_dtype, master_dtype,
    )
    return dataclasses.replace(preset, master_dtype=master_dtype)


def plan_grpo(
    cfg: Mapping[str, Any],
    *,
    reward_factory: Optional[Callable[..., RewardSet]] = None,
    marker_checker: Optional[Callable[[Path, Path], Mapping[str, Any]]] = None,
) -> GrpoPlan:
    """Resolve and check a run from its resolved config; nothing is trained and no GPU is used."""
    config = dict(cfg)
    output_dir = Path(config["output_dir"])
    stopped = read_stopped(output_dir)
    if stopped is not None:
        raise TrainerConfigError(
            f"{output_dir} ended at its early stop after step {stopped['step']} ({output_dir / STOP_FNAME}): a run "
            "that stopped so is never resumed; train under another run_name"
        )
    model_path = Path(config["model"]["path"])
    if not model_path.is_dir():
        raise TrainerConfigError(f"model.path {model_path} is not a directory: set RULEFORGE_MODEL_ROOT to where the "
                                 "base model's weights are")
    model_config, _ = checkpoint_class(model_path)
    preset = resolve_preset(model_config, config["model"].get("master_dtype"))

    data = config["data"]
    rules_path = Path(data["rules"])
    rules_sha256 = sha256_file(rules_path)
    if rules_sha256 != data["rules_sha256"]:
        raise TrainerConfigError(f"{rules_path} has sha256 {rules_sha256}, not the pinned {data['rules_sha256']}")
    condition_path = Path(data["anchor_condition"])
    truncation = check_anchor_condition(
        json.loads(condition_path.read_text(encoding="utf-8")), model_path=model_path, rules_sha256=rules_sha256
    )

    allow_unscreened = data.get("allow_unscreened_rows", False)
    if not isinstance(allow_unscreened, bool):
        raise TrainerConfigError(f"data.allow_unscreened_rows is true or false, got {allow_unscreened!r}")
    from ruleforge.grpo.grpo_data import base_weights_sha256, recorded_protection
    from ruleforge.judges.completion_format import GrpoDataError

    # the marker covers the files the rows build was held to (its manifest's pinned, required and added files)
    markers = {"train": require_disjoint_marker(data["train"], checker=marker_checker,
                                                allow_unscreened=allow_unscreened,
                                                **recorded_protection(data["train"]))}
    # the base's weights by content, read once: the rows replay them, and a resume is held to them
    try:
        weights_sha256 = base_weights_sha256(model_path)
    except GrpoDataError as error:
        raise TrainerConfigError(str(error)) from error
    if weights_sha256 is None:
        raise TrainerConfigError(f"{model_path} holds no safetensors weights: a run is bound to its base weights by "
                                 "their digest")
    # the replay rows are the base's raw token ids, which another model's would silently corrupt
    rows_model = {"train": rows_built_on(data["train"], model_path, sha256=markers["train"]["dataset_sha256"],
                                         weights_sha256=weights_sha256)}
    check_built_condition(rows_model["train"], condition_path)
    train_rows = load_jsonl_records(data["train"])
    check_prompt_shape(train_rows)
    grpo = dict(config["grpo"])
    template_kwargs = dict(grpo.get("chat_template_kwargs") or {})
    built_kwargs = rows_model["train"]["chat_template_kwargs"]
    if template_kwargs != built_kwargs:
        raise TrainerConfigError(
            f"grpo renders the prompts with chat_template_kwargs {template_kwargs}, and the rows were built and checked "
            f"under {built_kwargs} (their manifest's rendering): train under the arguments the rows were rendered with"
        )
    tokenizer, completion_format = load_training_format(model_path)
    counts = prompt_token_counts(
        train_rows, tokenizer, template_kwargs,
        think_end_token_id=completion_format.think_end_token_id,
    )
    cap, margin = int(data["prompt_token_cap"]), int(config["vllm_length_margin"])
    check_prompt_lengths(counts, cap)
    if cap > margin:
        raise TrainerConfigError(
            f"the {cap}-token prompt cap exceeds vllm_length_margin {margin}: vLLM would stop a long prompt's "
            "completion at max_model_len before max_completion_length, and TRL would mask it as truncated"
        )

    anchor_cfg = config["anchor"]
    anchor: Optional[AnchorSettings] = None
    replay: Optional[ReplayRows] = None
    if anchor_cfg["enabled"]:
        anchor = AnchorSettings(
            config=AnchorAuxConfig(
                loss_weight=float(anchor_cfg["loss_weight"]),
                kind=str(anchor_cfg["kind"]),
                mix_every_n_steps=int(anchor_cfg["mix_every_n_steps"]),
                per_device_batch_size=int(anchor_cfg["per_device_batch_size"]),
            ),
            accum=int(anchor_cfg["anchor_accum"]),
            temperature=float(anchor_cfg["temperature"]),
        )
        markers["replay"] = require_disjoint_marker(data["replay"], checker=marker_checker,
                                                    allow_unscreened=allow_unscreened,
                                                    **recorded_protection(data["replay"]))
        replay = ReplayRows.from_jsonl(data["replay"], expected_sha256=markers["replay"]["dataset_sha256"])
        rows_model["replay"] = rows_built_on(data["replay"], model_path, sha256=markers["replay"]["dataset_sha256"],
                                             weights_sha256=weights_sha256)

    reward_set = configured_reward_set(config["reward"], rules=load_jsonl_records(rules_path),
                                       num_generations=int(grpo["num_generations"]), factory=reward_factory,
                                       completion_format=completion_format)
    check_finish_tokens(reward_set, tokenizer)
    grpo_kwargs, forced = grpo_config_kwargs(
        grpo,
        truncation=truncation,
        reward_weights=reward_set.reward_weights,
        model_init_kwargs=preset.model_init_kwargs(str(config["model"]["attn_implementation"])),
        output_dir=output_dir,
        vllm_length_margin=margin,
        anchored=anchor is not None,
    )
    check_whole_groups(
        int(grpo_kwargs["per_device_train_batch_size"]),
        int(grpo_kwargs["steps_per_generation"]),
        int(grpo_kwargs["num_generations"]),
    )
    probe_cfg = config["sync_probe"]
    probe_modules = list(probe_cfg.get("modules") or default_probe_modules(
        int(model_config.get_text_config().num_hidden_layers), preset.decoder_prefix
    ))
    digests = {
        WEIGHTS_DIGEST: weights_sha256,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "generation_config_sha256": sha256_file(model_path / "generation_config.json"),
        "rules_sha256": rules_sha256,
        "anchor_condition_sha256": sha256_file(condition_path),
        "markers": markers,
    }
    init_path, init = config["model"].get("init_adapter"), None
    if init_path is not None:  # a cold start's identity leaves the key out, as resume_free's digest does
        init = check_init_adapter(init_path, model_path=model_path, preset=preset, digests=digests,
                                  truncation=truncation, grpo=grpo)
        digests["init_adapter"] = init
    stack = stack_versions()
    identity = run_identity(config=config, digests=digests, preset=preset, truncation=truncation, stack=stack)
    resume_from = resolve_resume(config.get("resume_from_checkpoint"), output_dir)
    resumed = (
        None if resume_from is None
        else check_resume(resume_from, identity, max_steps=int(grpo_kwargs["max_steps"]), config=config)
    )
    refusal = None if resumed is None or not grpo.get("fsdp") else fsdp2_state.resume_refusal(
        resume_from, grpo.get("fsdp_config") or {})
    if refusal is not None:
        raise TrainerConfigError(f"{resume_from}: {refusal}")
    wandb_block = config.get("wandb") or {}
    wandb_settings = wandb_env(wandb_block, output_dir=output_dir, report_to=grpo_kwargs.get("report_to"),
                               resume=wandb_resume(output_dir, resume_from, project=wandb_block.get("project"),
                                                   entity=wandb_block.get("entity")))

    from ruleforge.core.provenance import code_state

    provenance = {
        "identity": identity,
        "config": config,
        "code": code_state(),
        "forced": forced,
        "prompt_tokens": {"rows": len(counts), "max": max(counts), "cap": cap},
        "preset_expected_modules": expected_module_count(model_config, preset),
        "replay": None if replay is None else replay.summary(),
        "completion_format": completion_format.record(),
        "rows_model": rows_model,
        "init_adapter": init,
    }
    return GrpoPlan(
        output_dir=output_dir,
        model_path=model_path,
        preset=preset,
        truncation=truncation,
        tokenizer=tokenizer,
        train_rows=train_rows,
        replay=replay,
        reward_set=reward_set,
        grpo_kwargs=grpo_kwargs,
        forced=forced,
        anchor=anchor,
        probe_modules=probe_modules,
        probe_plain_modules=list(probe_cfg.get("plain_modules") or []),
        probe_max_rows=int(probe_cfg["max_rows"]),
        alarm=RetentionAlarm(step=int(config["retention_alarm"]["step"]),
                             threshold=float(config["retention_alarm"]["threshold"])),
        resume_from=resume_from,
        provenance=provenance,
        resumed_record=resumed,
        completion_format=completion_format,
        wandb_env=wandb_settings,
        wandb_completions_table=wandb_completions_table(wandb_block),
        init_adapter=init,
    )


def colocated_engine_model(vllm_generation: Any) -> Any:
    """Return the in-process vLLM model TRL's colocated sync writes through."""
    return vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model


def fsdp2_engine_tensors(vllm_generation: Any, module: Any) -> Iterator[tuple[str, Any]]:
    """Yield TRL's FSDP2 sync walk for a model with persistent buffers: no buffer is pushed, and a sharded parameter is
    gathered whole."""
    from accelerate.utils import is_peft_model

    buffers = {name for name, _ in module.named_buffers()}
    peft = is_peft_model(module)
    for key, tensor in module.state_dict().items():
        name = key.removeprefix(_PEFT_PREFIX).replace(".base_layer", "")
        if key in buffers or (peft and module.prefix in name) or "original_module" in name:
            continue
        name = vllm_generation._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
        yield name, _whole(tensor.to(vllm_generation.accelerator.device))


def install_fsdp2_sync(vllm_generation: Any) -> None:
    """Make `fsdp2_engine_tensors` the generation backend's FSDP2 walk."""
    vllm_generation._iter_fsdp2_params = functools.partial(fsdp2_engine_tensors, vllm_generation)


def make_anchored_grpo_trainer_class(base_cls: type) -> type:
    """Return `AnchoredGRPOTrainer` built on `base_cls`, `trl.GRPOTrainer` in a run."""
    import torch
    from transformers import TrainerCallback

    class ResourceSteps(TrainerCallback):
        """Times each optimizer step and hands it to the trainer's resource record."""

        def __init__(self, trainer: Any) -> None:
            self.trainer = trainer
            self.started: Optional[float] = None

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.started = time.monotonic()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.started is not None:
                self.trainer._record_resources(time.monotonic() - self.started)
                self.started = None

    class WandbRunMarker(TrainerCallback):
        """Rank 0's record that W&B accepted the run, written at the start of training after HF's WandbCallback has
        opened it."""

        def __init__(self, output_dir: Path) -> None:
            import wandb

            self.output_dir = Path(output_dir)
            self._wandb = wandb

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            from ruleforge.evaluate.runner import utc_now

            if state.is_world_process_zero:
                write_wandb_marker(self.output_dir, self._wandb.run, at=utc_now())

    class AnchoredGRPOTrainer(base_cls):  # type: ignore[misc, valid-type]
        """TRL's GRPO with the KL anchor, the truncation mask, the sync probe and provenance."""

        resource_steps = ResourceSteps
        wandb_run_marker = WandbRunMarker
        #: The warm start's checkpoint, and whether its first sync is still to come.
        init_adapter: Optional[dict[str, Any]] = None
        warm_start_pending = False

        def __init__(
            self,
            *args: Any,
            truncation: Truncation,
            anchor_rows: Optional[ReplayRows | Sequence[Mapping[str, Any]]] = None,
            anchor: Optional[AnchorSettings] = None,
            sync_probe: Optional[SyncProbe] = None,
            retention_alarm: Optional[RetentionAlarm] = None,
            provenance: Optional[Mapping[str, Any]] = None,
            resumed_record: Optional[Mapping[str, Any]] = None,
            think_end_token_id: Optional[int] = None,
            rank0_master: bool = False,
            wandb_completions_table: bool = False,
            init_adapter: Optional[Mapping[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            grpo_args = kwargs.get("args")
            if (anchor is None) != (anchor_rows is None):
                raise TrainerConfigError("the anchor and its replay rows come together: an anchored run needs both")
            self.anchored = anchor is not None
            if self.anchored and (grpo_args.beta != 0.0 or not grpo_args.mask_truncated_completions):
                raise TrainerConfigError(
                    f"an anchored run needs beta 0 and mask_truncated_completions True, got beta "
                    f"{grpo_args.beta} and mask_truncated_completions {grpo_args.mask_truncated_completions}"
                )
            if self.anchored and think_end_token_id is None:
                raise TrainerConfigError(
                    "an anchored run takes the model's think-end id (its plan's completion format): the anchor's "
                    "answer-segment KL is read after it"
                )
            if getattr(grpo_args, "use_liger_kernel", False):
                raise TrainerConfigError(
                    "use_liger_kernel: Liger's fused linear loss never calls the LM head's forward, so the "
                    "truncation mask (a hook on it) would silently not apply"
                )
            super().__init__(*args, **kwargs)
            self.wandb_completions_table = wandb_completions_table
            self.rank0_master = rank0_master
            self.resumed_adapter_sha256 = (resumed_record or {}).get("adapter_sha256")
            self.init_adapter = None if init_adapter is None else dict(init_adapter)
            self.truncation = truncation
            self._check_sampling()
            self.truncation_mask = TruncationMask(truncation).attach(logits_module(self.model))
            self.anchor = anchor
            self.anchor_manager: Optional[AnchorReplayManager] = None
            self.think_end_token_id: Optional[int] = None
            if self.anchored:
                self.anchor_manager = AnchorReplayManager(
                    anchor_rows,
                    config=anchor.config,
                    pad_token_id=self._tokenizer.pad_token_id,
                    seed=self.args.seed,
                    num_processes=self.accelerator.num_processes,
                    process_index=self.accelerator.process_index,
                    rows_served=int(((resumed_record or {}).get("anchor") or {}).get("rows_served", 0)),
                )
                self.think_end_token_id = think_end_token_id
            recorded = dict((resumed_record or {}).get("telemetry", {}))
            self.run_provenance = dict(provenance or {})
            if self.use_vllm and self.vllm_mode == "colocate":
                # TRL seeds each colocated engine with process_index // TP, whatever grpo.seed is
                tp = int(self.args.vllm_tensor_parallel_size)
                self.run_provenance["rollout_engine_seeds"] = [
                    rank // tp for rank in range(self.accelerator.num_processes)]
            self.telemetry: dict[str, Any] = {
                "logp_floor_step1": recorded.get("logp_floor_step1"),
                "sync_probe_last": None,
                "retention": list(recorded.get("retention", [])),
                "alarms": list(recorded.get("alarms", [])),
                "retention_alarm_checked_at": recorded.get("retention_alarm_checked_at"),
                "generation_checks": int(recorded.get("generation_checks", 0)),
                "resources": dict(recorded.get("resources") or {}),
            }
            self._device_used_gb = 0.0
            self.add_callback(self.resource_steps(self))
            self.retention_alarm = retention_alarm or RetentionAlarm()
            self.retention_alarm.checked = (
                self.retention_alarm.checked or self.telemetry["retention_alarm_checked_at"] is not None
            )
            if self.use_vllm:
                install_fsdp2_sync(self.vllm_generation)
            self.sync_probe = sync_probe
            if self.use_vllm and sync_probe is not None:
                self.install_sync_probe(sync_probe)

        def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
            """TRL's `log`, with the W&B completions table left out unless the run asks for it."""
            targets = self.args.report_to
            if self.wandb_completions_table or not isinstance(targets, list) or "wandb" not in targets:
                super().log(logs, start_time)
                return
            self.args.report_to = [name for name in targets if name != "wandb"]
            try:
                super().log(logs, start_time)
            finally:
                self.args.report_to = targets

        def _check_sampling(self) -> None:
            sampler = {
                "temperature": float(self.temperature),
                "top_k": int(self.top_k or 0),
                "top_p": float(self.top_p),
                "min_p": float(self.min_p or 0.0),
                "repetition_penalty": float(self.repetition_penalty),
            }
            expected = {"temperature": self.truncation.temperature, **self.truncation.as_sampling()}
            if sampler != expected or self.args.generation_kwargs:
                raise TrainerConfigError(
                    f"TRL samples with {sampler} and generation_kwargs {self.args.generation_kwargs!r}; the mask "
                    f"applies {expected} and no other kwargs are allowed"
                )
            template = getattr(self._tokenizer, "response_template", None)
            schema = getattr(self._tokenizer, "response_schema", None)
            if template is not None or schema is not None:
                raise TrainerConfigError(
                    "the tokenizer carries a response template or schema, so TRL would parse completions into "
                    "messages and the reward would not see the raw text after </think>"
                )

        def check_lora_coverage(self, expected: int) -> int:
            """Check the wrapped model has exactly `expected` LoRA modules and return the count."""
            from peft.tuners.lora import LoraLayer

            count = sum(isinstance(module, LoraLayer) for module in self.model.modules())
            if count != expected:
                raise TrainerConfigError(f"the preset wrapped {count} modules, not the expected {expected}")
            return count

        def install_sync_probe(self, probe: SyncProbe) -> None:
            """Wrap the generation backend's `sync_weights` so every sync is probed, and the engine's `llm.generate` so
            the engine is checked unchanged before it samples."""
            if self.vllm_mode != "colocate":
                raise TrainerConfigError(
                    f"the sync probe reads the in-process engine's shards (colocate); this run is {self.vllm_mode}"
                )
            generation = self.vllm_generation

            def engine_params() -> dict[str, Any]:
                return dict(colocated_engine_model(generation).named_parameters())

            def step() -> int:
                return int(self.state.global_step)

            generation.sync_weights = probed_sync(
                generation.sync_weights, probe=probe, model=generation.model, engine_params=engine_params,
                step=step, on_result=self._on_probe_result,
            )
            checked = checked_generate(
                generation.llm.generate, probe=probe, engine_params=engine_params, step=step,
                on_check=self._on_generation_check,
            )

            def generate(*args: Any, **kwargs: Any) -> Any:
                outputs = checked(*args, **kwargs)
                self._sample_device_memory()
                return outputs

            generation.llm.generate = generate

        def _on_generation_check(self) -> None:
            self.telemetry["generation_checks"] += 1
            self._sample_device_memory()

        def _sample_device_memory(self) -> None:
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                self._device_used_gb = max(self._device_used_gb, (total - free) / GB)

        def _record_resources(self, seconds: float) -> None:
            self._sample_device_memory()
            cuda = torch.cuda.is_available()
            reserved = torch.cuda.max_memory_reserved() / GB if cuda else 0.0
            local = torch.tensor([seconds, self._device_used_gb, reserved], dtype=torch.float64,
                                 device=self.accelerator.device)
            ranks = self.accelerator.gather(local.unsqueeze(0)).tolist()
            step_seconds = max(row[0] for row in ranks)
            metrics = self._metrics["train"]
            metrics["resources/step_seconds"].append(step_seconds)
            record = self.telemetry["resources"]
            record.setdefault("step_seconds", []).append(step_seconds)
            for index, name in ((1, "device_used_gb"), (2, "torch_reserved_gb")):
                values = [row[index] for row in ranks]
                for rank, value in enumerate(values):
                    metrics[f"resources/{name}/rank{rank}"].append(value)
                held = record.get(f"{name}_max") or [0.0] * len(values)
                record[f"{name}_max"] = [max(old, new) for old, new in zip(held, values)]
            self._device_used_gb = 0.0
            if cuda:
                torch.cuda.reset_peak_memory_stats()

        def _on_probe_result(self, result: ProbeResult) -> None:
            if self.warm_start_pending:
                if result.vacuous:
                    raise SyncProbeError(
                        f"the first sync after the warm start from {self.init_adapter['checkpoint']} (step "
                        f"{result.step}) moved no probed element: the engine would sample the base, not the init "
                        "adapter")
                self.warm_start_pending = False
            self.telemetry["sync_probe_last"] = result.to_dict()
            metrics = self._metrics["train"]
            metrics["sync/probe_ok"].append(1.0)
            metrics["sync/moved_elements"].append(float(sum(point.moved for point in result.points)))
            if result.retention is not None:
                self.telemetry["retention"].append([result.step, result.retention])
                metrics["sync/retention"].append(result.retention)
                for point in result.points:
                    if point.retention is not None:
                        metrics[f"sync/retention/{point.module}"].append(point.retention)
            alarm = self.retention_alarm.observe(result)
            if self.retention_alarm.checked and self.telemetry["retention_alarm_checked_at"] is None:
                self.telemetry["retention_alarm_checked_at"] = result.step
            if alarm is not None:
                self.telemetry["alarms"].append(alarm)
                logger.warning("RETENTION ALARM: %s", json.dumps(alarm))
                if self.args.should_save:
                    append_alarm(Path(self.args.output_dir) / ALARMS_FNAME, alarm)

        def _get_per_token_logps_and_entropies(self, *args: Any, **kwargs: Any) -> Any:
            with self.truncation_mask.active():
                return super()._get_per_token_logps_and_entropies(*args, **kwargs)

        def _generate_and_score_completions(self, inputs: Any) -> Any:
            output = super()._generate_and_score_completions(inputs)
            if self.model.training:
                self._record_truncation(output)
            if output.get("old_per_token_logps") is not None:
                # TRL has taken the vLLM IS ratio from them; zeroed, an outside token's PPO ratio stays finite
                output["old_per_token_logps"] = zero_outside_logps(output["old_per_token_logps"], self.truncation)
            return output

        def _record_truncation(self, output: Mapping[str, Any]) -> None:
            old = output.get("old_per_token_logps")
            sampled = output.get("sampling_per_token_logps")
            if old is None or sampled is None:
                return
            mask = output["completion_mask"].bool()
            outside, total = outside_kept_share(old, mask, self.truncation)
            kept = mask & (old >= self.truncation.outside_logp) & torch.isfinite(sampled)
            # signed as well: a kept set renormalized over other tokens than vLLM's shifts every log-prob one way
            signed = (old - sampled)[kept]
            device = self.accelerator.device
            local = torch.tensor(
                [float(outside), float(total), float(signed.abs().sum()), float(signed.sum()), float(signed.numel())],
                dtype=torch.float64, device=device,
            )
            pooled = self.accelerator.gather(local.unsqueeze(0)).sum(dim=0).tolist()
            share = pooled[0] / pooled[1] if pooled[1] else 0.0
            mean_difference = pooled[2] / pooled[4] if pooled[4] else 0.0
            mean_signed = pooled[3] / pooled[4] if pooled[4] else 0.0
            self._metrics["train"]["truncation/outside_kept_share"].append(share)
            self._metrics["train"]["truncation/logp_difference_kept_mean"].append(mean_difference)
            self._metrics["train"]["truncation/logp_difference_kept_signed_mean"].append(mean_signed)
            if self.telemetry["logp_floor_step1"] is None and int(self.state.global_step) == 0:
                trl_mean = self._metrics["train"].get("sampling/sampling_logp_difference/mean") or [None]
                self.telemetry["logp_floor_step1"] = {
                    "logp_difference_kept_mean": mean_difference,
                    "logp_difference_kept_signed_mean": mean_signed,
                    "outside_kept_share": share,
                    "trl_sampling_logp_difference_mean": trl_mean[-1],
                }

        def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> Any:
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not self.anchored or not should_run_anchor_step(
                global_step=int(self.state.global_step),
                mix_every_n_steps=self.anchor.config.mix_every_n_steps,
                sync_gradients=bool(self.accelerator.sync_gradients),
            ):
                return loss
            if not self.anchor_manager.started:
                self.anchor_manager.start(int(self.state.global_step))
            total = None
            sums = {"raw": 0.0, "answer_kl": 0.0, "answer_tokens": 0.0}
            for _ in range(self.anchor.accum):
                scaled, raw, answer = self._anchor_micro_step(model)
                total = scaled if total is None else total + scaled
                sums["raw"] += float(raw) / self.anchor.accum
                sums["answer_kl"] += answer[0]
                sums["answer_tokens"] += answer[1]
            self._record_anchor(float(total), sums)
            return loss + total

        def _anchor_micro_step(self, model: Any) -> tuple[Any, Any, tuple[float, float]]:
            batch = {name: value.to(self.accelerator.device) for name, value in self.anchor_manager.next_batch().items()}
            base = self._anchor_logits(model, batch, adapter=False) if self.anchor.config.kind == "kl" else None
            policy = self._anchor_logits(model, batch, adapter=True)
            weighted, raw = weighted_anchor_loss(
                policy_logits=policy,
                base_logits=base,
                labels=batch["labels"],
                weights=batch["weight"],
                config=self.anchor.config,
                temperature=self.anchor.temperature,
            )
            scaled = weighted / self.anchor.accum
            if self.anchor.config.kind == "kl" and float(raw) == 0.0:
                # KL = 0 is its minimum: the backward there is rounding, which Adam would scale to a full step. Zeroed,
                # not skipped, so every rank runs the same backward passes.
                scaled = scaled * 0.0
            self._anchor_backward(scaled)
            answer = (0.0, 0.0)
            if base is not None and self.think_end_token_id is not None:
                with torch.no_grad():
                    labels = answer_segment_labels(batch["input_ids"], batch["labels"], self.think_end_token_id)
                    kl_sum, count = anchor_kl_loss(policy.detach(), base, labels, temperature=self.anchor.temperature)
                answer = (float(kl_sum), float(count))
            return scaled.detach(), raw, answer

        def _anchor_logits(self, model: Any, batch: Mapping[str, Any], *, adapter: bool) -> Any:
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
            with self.compute_loss_context_manager():
                if adapter:
                    return model(**inputs).logits
                from trl.models.utils import disable_gradient_checkpointing
                from trl.trainer.utils import use_adapter

                with torch.no_grad(), disable_gradient_checkpointing(
                    self.model, self.args.gradient_checkpointing_kwargs
                ), use_adapter(self.accelerator.unwrap_model(self.model), adapter_name=None):
                    return self.model(**inputs).logits

        def _anchor_backward(self, loss: Any) -> None:
            self.accelerator.backward(loss)

        def _record_anchor(self, weighted: float, sums: Mapping[str, float]) -> None:
            local = torch.tensor(
                [weighted, sums["raw"], sums["answer_kl"], sums["answer_tokens"]],
                dtype=torch.float64, device=self.accelerator.device,
            )
            gathered = self.accelerator.gather(local.unsqueeze(0))
            means = gathered.mean(dim=0).tolist()
            totals = gathered.sum(dim=0).tolist()
            metrics = self._metrics["train"]
            metrics["anchor/loss_weighted"].append(means[0])
            metrics["anchor/kl" if self.anchor.config.kind == "kl" else "anchor/ce"].append(means[1])
            if totals[3]:
                metrics["anchor/kl_answer"].append(totals[2] / totals[3])
            metrics["anchor/rows_served"].append(float(self.anchor_manager.rows_served))

        def _fsdp_version(self) -> Optional[int]:
            if not getattr(self, "is_fsdp_enabled", False):
                return None
            return int(getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1))

        def _prepare_for_training(self, max_steps: Any, train_dataloader: Any, resume_from_checkpoint: Any) -> Any:
            """HF's preparation, with the master sharded and filled from rank 0 first when only rank 0 loaded it."""
            if self.rank0_master:
                self.model = self.model_wrapped = fsdp2_state.shard_from_rank0(self.accelerator, self.model)
                self.rank0_master = False
            prepared = super()._prepare_for_training(max_steps, train_dataloader, resume_from_checkpoint)
            if resume_from_checkpoint is None and self.init_adapter is not None:
                self._load_init_adapter()
            return prepared

        def _load_init_adapter(self) -> None:
            checkpoint = Path(self.init_adapter["checkpoint"])
            if self._fsdp_version() != 2:
                raise TrainerConfigError(f"model.init_adapter {checkpoint}: a warm start loads the checkpoint's FSDP2 "
                                         f"sharded state, and this trainer runs under FSDP {self._fsdp_version()}")
            fsdp2_state.load_adapter(checkpoint, self.model, adapter_file=checkpoint / ADAPTER_WEIGHTS_FNAME,
                                     expect_adapter_sha256=self.init_adapter["adapter_sha256"])
            self.warm_start_pending = True
            logger.info("warm start: loaded %s's adapter (run %s, step %s, adapter %s)", checkpoint,
                        self.init_adapter.get("run"), self.init_adapter.get("step"),
                        self.init_adapter["adapter_sha256"][:12])

        def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
            """Save the sharded resume state under FSDP2 in place of transformers' adapter-only one; HF's otherwise."""
            if self._fsdp_version() != 2:
                super()._save_optimizer_and_scheduler(output_dir)
                return
            fsdp2_state.save_state(output_dir, model=self.model, optimizer=self.optimizer,
                                   scheduler=self.lr_scheduler, step=int(self.state.global_step))

        def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any = None) -> None:
            """HF's checkpoint load; under FSDP2 the adapter from the sharded resume state, checked; FSDP1 refused."""
            version = self._fsdp_version()
            if version is None:
                super()._load_from_checkpoint(resume_from_checkpoint, model)
                return
            if version != 2:
                raise TrainerConfigError(f"{resume_from_checkpoint}: {fsdp2_state.FSDP1_REFUSAL}")
            fsdp2_state.load_adapter(resume_from_checkpoint, self.model,
                                     adapter_file=Path(resume_from_checkpoint) / ADAPTER_WEIGHTS_FNAME,
                                     expect_adapter_sha256=self.resumed_adapter_sha256)

        def _load_optimizer_and_scheduler(self, checkpoint: Optional[str]) -> None:
            """HF's; under FSDP2 the optimizer and scheduler from the sharded resume state."""
            if checkpoint is None or self._fsdp_version() != 2:
                super()._load_optimizer_and_scheduler(checkpoint)
                return
            fsdp2_state.load_optimizer(checkpoint, model=self.model, optimizer=self.optimizer,
                                       scheduler=self.lr_scheduler)

        def checkpoint_record(self, checkpoint: Path) -> dict[str, Any]:
            return {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "step": int(self.state.global_step),
                "adapter_sha256": adapter_digest(checkpoint),
                "grpo_config": json.loads(self.args.to_json_string()),
                "batch_geometry": batch_geometry(self.args),
                "run": self.run_provenance,
                "telemetry": self.telemetry,
                "anchor": None if self.anchor_manager is None else self.anchor_manager.summary(),
            }

        def early_stop_requested(self, output_dir: Path) -> Optional[dict[str, Any]]:
            """Return the early-stop marker as every rank agrees on it (requested when any rank sees it), or None."""
            seen = (output_dir / STOP_FNAME).is_file()
            flag = torch.tensor([1.0 if seen else 0.0], device=self.accelerator.device)
            if not float(self.accelerator.gather(flag).max()) > 0:
                return None
            digest = stop_marker_sha256(output_dir) if (output_dir / STOP_FNAME).is_file() else None
            return {"marker_sha256": digest, "stopped_after_step": int(self.state.global_step)}

        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            from ruleforge.evaluate.runner import utc_now

            super()._save_checkpoint(model, trial)
            output_dir = Path(self.args.output_dir)
            stop = self.early_stop_requested(output_dir)
            if self.args.should_save:
                checkpoint = Path(self._get_output_dir(trial=trial)) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                record = self.checkpoint_record(checkpoint)
                if stop is not None:
                    record["early_stop"] = stop
                # written last: a checkpoint without it is one whose save did not finish
                write_provenance(checkpoint, record)
                if stop is not None:
                    write_stopped(output_dir, step=stop["stopped_after_step"], marker_sha256=stop["marker_sha256"],
                                  at=utc_now())
            if stop is not None:
                logger.warning("early stop: %s requested it; training ends after the save at step %d",
                               output_dir / STOP_FNAME, stop["stopped_after_step"])
                self.control.should_training_stop = True

    return AnchoredGRPOTrainer


@functools.lru_cache(maxsize=1)
def anchored_grpo_trainer_class() -> type:
    """Return the trainer on `trl.GRPOTrainer`."""
    from trl import GRPOTrainer

    return make_anchored_grpo_trainer_class(GRPOTrainer)


def probe_tp_rank(process_index: int, tp_size: int) -> int:
    """Return the tensor-parallel rank of a process's colocated engine (TRL groups consecutive ranks)."""
    return int(process_index) % int(tp_size)


def trainer_model(plan: GrpoPlan, args: Any) -> Any:
    """Return what the trainer builds its model from: the checkpoint's path, or under FSDP2 the master loaded on rank 0
    alone."""
    if fsdp2_state.fsdp_version(args) != 2:
        return str(plan.model_path)
    init = plan.grpo_kwargs["model_init_kwargs"]
    return fsdp2_state.rank0_model(plan.model_path, rank=int(args.process_index), dtype=init["dtype"],
                                   attn_implementation=init["attn_implementation"])


def build_trainer(plan: GrpoPlan, *, grpo_overrides: Optional[Mapping[str, Any]] = None) -> Any:
    """Return the trainer for a plan; `grpo_overrides` replaces GRPOConfig fields."""
    from datasets import Dataset

    args = build_grpo_config({**plan.grpo_kwargs, **(grpo_overrides or {})})
    if plan.resumed_record is not None:
        check_resume_geometry(plan.resumed_record, args)
    elif plan.init_adapter is not None:
        check_init_geometry(plan.init_adapter, args)
    model_config, _ = checkpoint_class(plan.model_path)
    tp_size = int(args.vllm_tensor_parallel_size)
    probe = SyncProbe.for_modules(
        plan.probe_modules, model_config, max_rows=plan.probe_max_rows, plain=plan.probe_plain_modules,
        tp_size=tp_size, tp_rank=probe_tp_rank(args.process_index, tp_size),
    )
    export_wandb_env(os.environ, plan.wandb_env, rank=int(args.process_index))
    model = trainer_model(plan, args)
    trainer = anchored_grpo_trainer_class()(
        model=model,
        reward_funcs=list(plan.reward_set.reward_funcs),
        args=args,
        train_dataset=Dataset.from_list(plan.train_rows),
        processing_class=plan.tokenizer,
        peft_config=plan.preset.lora_config(),
        truncation=plan.truncation,
        anchor_rows=plan.replay,
        anchor=plan.anchor,
        sync_probe=probe,
        retention_alarm=plan.alarm,
        provenance=plan.provenance,
        resumed_record=plan.resumed_record,
        think_end_token_id=plan.completion_format.think_end_token_id,
        rank0_master=not isinstance(model, str),
        wandb_completions_table=plan.wandb_completions_table,
        init_adapter=plan.init_adapter,
    )
    if plan.wandb_env:
        trainer.add_callback(trainer.wandb_run_marker(plan.output_dir))
    trainer.check_lora_coverage(plan.provenance["preset_expected_modules"])
    return trainer


def finalize_plan(plan: GrpoPlan) -> None:
    """Archive the run's sources under the output directory and record the archive's path."""
    from ruleforge.core.provenance import write_code_snapshot

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = write_code_snapshot(plan.output_dir, plan.provenance["code"])
    plan.provenance["code_snapshot"] = None if snapshot is None else str(snapshot)


def run_grpo(
    cfg: Mapping[str, Any],
    *,
    reward_factory: Optional[Callable[..., RewardSet]] = None,
    marker_checker: Optional[Callable[[Path, Path], Mapping[str, Any]]] = None,
) -> Optional[Any]:
    """Plan, then (unless `preflight_only`) build and train; return the trainer, or None after a preflight."""
    plan = plan_grpo(cfg, reward_factory=reward_factory, marker_checker=marker_checker)
    init = plan.init_adapter
    init_text = None if init is None else (
        f"{init['checkpoint']} (run {init['run']}, step {init['step']}, adapter {init['adapter_sha256'][:12]}; "
        f"{'loaded before step 0' if plan.resume_from is None else 'not loaded: a resume'})")
    logger.info(
        "GRPO plan: %d train rows, %s replay rows, truncation %s, forced %s, reward %s, init_adapter %s, resume %s",
        len(plan.train_rows), None if plan.replay is None else len(plan.replay),
        plan.truncation.as_sampling(), plan.forced, reward_form(plan.reward_set), init_text, plan.resume_from,
    )
    if plan.wandb_env:
        mode = os.environ.get("WANDB_MODE", "online")
        logger.info("W&B: run %s in project %s, group %s, mode %s (resume %s; local files under %s/wandb)",
                    plan.wandb_env["WANDB_RUN_ID"], plan.wandb_env["WANDB_PROJECT"],
                    plan.wandb_env["WANDB_RUN_GROUP"], mode, plan.wandb_env["WANDB_RESUME"],
                    plan.wandb_env["WANDB_DIR"])
        if mode != "online":
            logger.warning("W&B mode is %s (WANDB_MODE), not online: offline keeps the run under %s/wandb until "
                           "`wandb sync`; disabled logs nothing", mode, plan.wandb_env["WANDB_DIR"])
    if cfg.get("preflight_only"):
        return None
    finalize_plan(plan)
    trainer = build_trainer(plan)
    trainer.train(resume_from_checkpoint=None if plan.resume_from is None else str(plan.resume_from))
    return trainer
