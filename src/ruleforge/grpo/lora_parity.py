from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ruleforge.core.artifacts import load_jsonl_records, write_jsonl_records
from ruleforge.core.digests import canonical_digest, sha256_file
from ruleforge.core.optional import require
from ruleforge.judges.engine_state import engine_dtype, kv_cache_dtype, vllm_config
from ruleforge.judges.weights import MANIFEST_FNAME as MERGE_MANIFEST_FNAME
from ruleforge.judges.weights import ParityError, read_safetensors_header

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

HF_ARMS = ("hf_base", "hf_lora")
#: HF_lora under bf16 autocast, the forward training ran (descriptive: the autocast gap).
HF_LORA_AC_ARM = "hf_lora_ac"
#: The merged directory in HF, in its serving dtype: the served arm's weights on HF's arithmetic.
HF_MERGED_ARM = "hf_merged"
#: The HF arms :func:`compare` reads when a score file holds them.
OPTIONAL_HF_ARMS = (HF_MERGED_ARM, HF_LORA_AC_ARM)
#: The autocast dtype of ``hf_lora_ac`` (training's forward).
AUTOCAST_DTYPE = "bfloat16"
BASE_ARM = "vllm_base"
LORA_ARM = "vllm_lora"
MERGED_ARM = "vllm_merged"
SERVED_ARMS = {"lora": LORA_ARM, "merged": MERGED_ARM}
#: The base decode reference's decode-path logprobs (its prefill ones are ``vllm_base``).
BASE_DECODE_ARM = "vllm_base_decode"

ADAPTER_CONFIG_FNAME = "adapter_config.json"
ADAPTER_WEIGHTS_FNAME = "adapter_model.safetensors"
_PEFT_PREFIX = "base_model.model."
_LORA_SUFFIXES = {".lora_A.weight": "A", ".lora_B.weight": "B"}

#: The projections LoRA usually targets in a Llama-style decoder (Llama, Qwen2/Qwen3, Mistral): attention and MLP.
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

#: mean|X - Y| / mean|X| for two i.i.d. Gaussian noises: a faithful adapter's paired statistic over the base arms'
#: noise when the two vLLM passes round independently.
PAIRED_NOISE_FACTOR = math.sqrt(2.0)


@dataclass(frozen=True)
class FrozenSequence:
    """One token sequence every arm is teacher-forced over."""

    seq_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.prompt_ids) + len(self.completion_ids) < 2:
            raise ParityError(f"sequence {self.seq_id!r} has fewer than two tokens: nothing to score")

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.completion_ids


def write_sequences(path: Path, sequences: Sequence[FrozenSequence]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for seq in sequences:
            handle.write(json.dumps({
                "id": seq.seq_id,
                "prompt_ids": list(seq.prompt_ids),
                "completion_ids": list(seq.completion_ids),
                "finish_reason": seq.finish_reason,
            }) + "\n")


def read_sequences(path: Path) -> list[FrozenSequence]:
    sequences = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequences.append(FrozenSequence(
            seq_id=str(row["id"]),
            prompt_ids=tuple(int(t) for t in row["prompt_ids"]),
            completion_ids=tuple(int(t) for t in row["completion_ids"]),
            finish_reason=row.get("finish_reason"),
        ))
    ids = [seq.seq_id for seq in sequences]
    if not ids:
        raise ParityError(f"{path} holds no sequences")
    if len(set(ids)) != len(ids):
        raise ParityError(f"{path} repeats a sequence id")
    return sequences


#: ``{arm: {sequence id: one value per scored position}}``.
ArmValues = dict[str, dict[str, list[Any]]]


@dataclass
class Scores:
    """Per-arm logprobs of tokens 1..L-1 and, where recorded, each position's argmax token."""

    logprobs: ArmValues = field(default_factory=dict)
    argmax: ArmValues = field(default_factory=dict)
    meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    def merge(self, other: "Scores") -> "Scores":
        """Both files' arms; an arm in both is refused."""
        shared = sorted(set(self.logprobs) & set(other.logprobs))
        if shared:
            raise ParityError(f"arm(s) {shared} are in two score files")
        return Scores(
            {**self.logprobs, **other.logprobs}, {**self.argmax, **other.argmax}, {**self.meta, **other.meta}
        )

    def only(self, seq_ids: Iterable[str]) -> "Scores":
        keep = set(seq_ids)
        return Scores(
            {arm: {k: v for k, v in rows.items() if k in keep} for arm, rows in self.logprobs.items()},
            {arm: {k: v for k, v in rows.items() if k in keep} for arm, rows in self.argmax.items()},
            dict(self.meta),
        )


SEQUENCES_KEY = "sequences_sha256"
ADAPTER_KEY = "adapter_sha256"
ADAPTER_CONFIG_KEY = "adapter_config_sha256"
ADAPTER_KEYS = (ADAPTER_KEY, ADAPTER_CONFIG_KEY)
#: The arms that apply the trained adapter, as a LoRA or merged into the weights.
ADAPTER_ARMS = ("hf_lora", HF_LORA_AC_ARM, HF_MERGED_ARM, LORA_ARM, MERGED_ARM)


def sequence_sha256(seq: FrozenSequence) -> str:
    """Return the digest of a sequence's id, prompt ids and completion ids, so a moved split is another sequence."""
    return canonical_digest([seq.seq_id, list(seq.prompt_ids), list(seq.completion_ids)])


def scored_adapter(adapter: Path | None, weights: Path | str) -> dict[str, str | None]:
    """Return the digests of the adapter an arm applies, its weights file and its configuration (the scale is
    lora_alpha / r): an adapter directory's own, or those a merged directory's manifest records; None for weights no
    merge made."""
    if adapter is not None:
        return {ADAPTER_KEY: sha256_file(Path(adapter) / ADAPTER_WEIGHTS_FNAME),
                ADAPTER_CONFIG_KEY: sha256_file(Path(adapter) / ADAPTER_CONFIG_FNAME)}
    manifest = Path(str(weights)) / MERGE_MANIFEST_FNAME
    merged = json.loads(manifest.read_text(encoding="utf-8"))["adapter"] if manifest.is_file() else {}
    return {ADAPTER_KEY: merged.get("sha256"), ADAPTER_CONFIG_KEY: merged.get("config_sha256")}


def check_scored(scores: Scores, sequences: Sequence[FrozenSequence]) -> list[str]:
    """Refuse score files whose recorded sequences are not `sequences` (other tokens or another split), or adapter arms
    that record different adapters; return notes naming the scores a file without these digests leaves unbound."""
    expected = {seq.seq_id: sequence_sha256(seq) for seq in sequences}
    notes = []
    for arm in sorted(scores.logprobs):
        recorded = scores.meta.get(arm, {}).get(SEQUENCES_KEY)
        if recorded is None:
            notes.append(f"{arm}: its score file records no sequence digests, so its scores are not bound to these "
                         "sequences")
            continue
        other = sorted(sid for sid, digest in expected.items() if recorded.get(sid) != digest)
        if other:
            raise ParityError(f"{arm}'s score file records other tokens, or none, for {len(other)} of these sequences "
                              f"(first {other[0]!r}): score every arm on these sequences")
    arms = [arm for arm in ADAPTER_ARMS if arm in scores.logprobs]
    for key in ADAPTER_KEYS:
        recorded = {arm: scores.meta[arm][key] for arm in arms if key in scores.meta.get(arm, {})}
        if unbound := [arm for arm in arms if arm not in recorded]:
            notes.append(f"{', '.join(unbound)}: no {key} recorded, so these scores are not bound to an adapter")
        if None in recorded.values() or len(set(recorded.values())) > 1:
            raise ParityError(f"the adapter arms record {key} {recorded}: every adapter arm scores the one adapter, "
                              "recorded by digest")
    return notes


def write_scores(path: Path, scores: Scores, meta: Mapping[str, Any], sequences: Sequence[FrozenSequence]) -> None:
    """Write a score file of `sequences`, recording each one's digest; NaN is kept as NaN."""
    payload = {
        "meta": {**meta, SEQUENCES_KEY: {seq.seq_id: sequence_sha256(seq) for seq in sequences}},
        "arms": {arm: {k: list(v) for k, v in rows.items()} for arm, rows in scores.logprobs.items()},
        "argmax": {arm: {k: list(v) for k, v in rows.items()} for arm, rows in scores.argmax.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def read_scores(path: Path) -> Scores:
    """Return a score file; every arm carries the file's `meta`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = dict(payload.get("meta") or {})
    return Scores(
        {arm: {k: [float(x) for x in v] for k, v in rows.items()} for arm, rows in payload["arms"].items()},
        {
            arm: {k: [None if x is None else int(x) for x in v] for k, v in rows.items()}
            for arm, rows in (payload.get("argmax") or {}).items()
        },
        {arm: dict(meta) for arm in payload["arms"]},
    )


def read_prompts(path: Path, prompt_field: str | None = None) -> list[dict[str, str]]:
    """Return prompt entries in the shape `format_prompt_for_generation` renders."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if prompt_field is not None:
            text = row.get(prompt_field)
            if not isinstance(text, str) or not text.strip():
                raise ParityError(f"a row of {path} has no non-empty {prompt_field!r}")
            entries.append({"system_prompt": text, "user_prompt": ""})
            continue
        system, user = row.get("system_prompt", ""), row.get("user_prompt", "")
        if not (isinstance(system, str) and isinstance(user, str)) or not (system or user):
            raise ParityError(f"a row of {path} has neither system_prompt nor user_prompt text")
        entries.append({"system_prompt": system, "user_prompt": user})
    if not entries:
        raise ParityError(f"{path} holds no prompts")
    return entries


@dataclass(frozen=True)
class AdapterReport:
    """What an adapter adapts, and why it cannot be served as-is (``problems``)."""

    rank: int | None
    alpha: float | None
    max_rank: int
    target_modules: tuple[str, ...]
    module_paths: tuple[str, ...]
    targeted_suffixes: tuple[str, ...]
    extra_keys: tuple[str, ...]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def inspect_adapter(adapter_dir: Path, *, module_names: Iterable[str] | None = None) -> AdapterReport:
    """Check an adapter directory without loading a model; `module_names`, every module path of the checkpoint's own
    class, turn a silent no-op into a refusal."""
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / ADAPTER_CONFIG_FNAME).read_text(encoding="utf-8"))
    header = read_safetensors_header(adapter_dir / ADAPTER_WEIGHTS_FNAME)
    factors: dict[str, dict[str, list[int]]] = {}
    extra: list[str] = []
    for key, meta in header.items():
        for suffix, factor in _LORA_SUFFIXES.items():
            if key.endswith(suffix):
                path = key[: -len(suffix)]
                path = path[len(_PEFT_PREFIX):] if path.startswith(_PEFT_PREFIX) else path
                factors.setdefault(path, {})[factor] = list(meta["shape"])
                break
        else:
            extra.append(key)

    problems: list[str] = []
    if not factors:
        problems.append("no LoRA factors (keys ending .lora_A.weight / .lora_B.weight)")
    ranks = set()
    for path, found in sorted(factors.items()):
        if set(found) != {"A", "B"}:
            problems.append(f"{path}: has lora_{''.join(sorted(found))} only")
            continue
        ranks.add(found["A"][0])
        if found["B"][1] != found["A"][0]:
            problems.append(f"{path}: lora_A rank {found['A'][0]} != lora_B rank {found['B'][1]}")
    if module_names is not None:
        known = set(module_names)
        missing = sorted(path for path in factors if path not in known)
        if missing:
            problems.append(
                f"{len(missing)} adapted module(s) are not in the checkpoint's module tree, "
                f"so they adapt nothing (e.g. {missing[0]}); an adapter trained on another "
                "class of the checkpoint (a text-only Auto class on a multimodal one) keys them so"
            )
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    targets = config.get("target_modules") or ()
    if isinstance(targets, str):
        targets = (targets,)
    return AdapterReport(
        rank=None if rank is None else int(rank),
        alpha=None if alpha is None else float(alpha),
        max_rank=max(ranks, default=0),
        target_modules=tuple(sorted(targets)),
        module_paths=tuple(sorted(factors)),
        targeted_suffixes=tuple(sorted({path.rpartition(".")[2] for path in factors})),
        extra_keys=tuple(sorted(extra)),
        problems=tuple(problems),
    )


def checkpoint_class(model_path: Path) -> tuple[Any, type]:
    """Return the checkpoint's config and its own class, `architectures[0]` of config.json, never a text-only Auto
    class."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(str(model_path))
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise ParityError(f"{model_path}/config.json names no architectures")
    return config, getattr(transformers, architectures[0])


def _meta_model(model_path: Path):
    import torch

    config, model_class = checkpoint_class(model_path)
    with torch.device("meta"):
        return model_class(config)


def checkpoint_module_names(model_path: Path) -> list[str]:
    """Return every module path of the checkpoint's own class, built on the meta device."""
    return [name for name, _ in _meta_model(model_path).named_modules()]


def make_random_adapter(
    model_path: Path,
    out_dir: Path,
    *,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    rank: int = 8,
    alpha: float = 16.0,
    seed: int = 0,
    b_std: float = 2e-3,
) -> AdapterReport:
    """Write a PEFT adapter with random, non-zero factors for `model_path`, since a freshly initialised LoRA (B = 0)
    would pass any parity check vacuously."""
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file

    model = _meta_model(model_path)
    module_names = [name for name, _ in model.named_modules()]
    unmatched = [t for t in target_modules if not any(n == t or n.endswith("." + t) for n in module_names)]
    if unmatched:
        raise ParityError(f"target modules {unmatched} match no module of {model_path}")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(target_modules),
        base_model_name_or_path=str(model_path),
    )
    peft_model = get_peft_model(model, config)
    shapes = {key: tuple(value.shape) for key, value in get_peft_model_state_dict(peft_model).items()}
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for key in sorted(shapes):
        shape = shapes[key]
        if key.endswith(".lora_A.weight"):
            bound = 1.0 / math.sqrt(shape[1])
            value = (torch.rand(shape, generator=generator) * 2.0 - 1.0) * bound
        else:
            value = torch.randn(shape, generator=generator) * b_std
        tensors[key] = value.contiguous()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.peft_config["default"].save_pretrained(str(out_dir))
    save_file(tensors, str(out_dir / ADAPTER_WEIGHTS_FNAME), metadata={"format": "pt"})
    return inspect_adapter(out_dir, module_names=module_names)


def scores_of_targets(
    logits: "torch.Tensor", token_ids: Sequence[int], *, chunk: int = 1024
) -> tuple[list[float], list[int]]:
    """Return log p(token_ids[i] | token_ids[:i]) and the argmax token, for i = 1..L-1, from [L, V] logits."""
    import torch

    targets = torch.as_tensor(list(token_ids[1:]), device=logits.device)
    values: list[float] = []
    argmax: list[int] = []
    for start in range(0, targets.shape[0], chunk):
        block = logits[start : start + chunk].float()
        rows = targets[start : start + chunk]
        logprobs = torch.log_softmax(block[: rows.shape[0]], dim=-1)
        values.extend(logprobs.gather(-1, rows[:, None])[:, 0].tolist())
        argmax.extend(logprobs.argmax(dim=-1).tolist())
    return values, argmax


def hf_sequence_scores(
    model: Any, token_ids: Sequence[int], *, device: str, chunk: int = 1024, autocast_dtype: str | None = None,
) -> tuple[list[float], list[int]]:
    """Return teacher-forced logprobs and argmax of one sequence under a HF (or PEFT) model, autocast off unless
    `autocast_dtype` names one."""
    import torch

    ids = torch.as_tensor([list(token_ids)], device=device)
    autocast = {"enabled": False} if autocast_dtype is None else {"dtype": getattr(torch, autocast_dtype)}
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, **autocast):
        logits = model(input_ids=ids, use_cache=False).logits[0]
    return scores_of_targets(logits, token_ids, chunk=chunk)


def exact_float32() -> dict[str, Any]:
    """Keep float32 matmuls and convolutions in float32 (no TF32); what is now in force."""
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def load_hf_model(model_path: Path, *, dtype: str = "bfloat16", device: str = "cuda", attn_implementation: str = "sdpa"):
    """Return the checkpoint's own class with its weights, in eval mode."""
    import torch

    _, model_class = checkpoint_class(model_path)
    model = model_class.from_pretrained(
        str(model_path), dtype=getattr(torch, dtype), attn_implementation=attn_implementation
    )
    return model.to(device).eval()


#: The arm a plain model (no adapter) is scored as: the base, or a merged directory.
PLAIN_HF_ARMS = ("hf_base", HF_MERGED_ARM)


def wrap_adapter(model: Any, adapter_dir: Path) -> Any:
    """Return `model` wrapped by `PeftModel` with the adapter, in eval mode; once per model, since a second wrap would
    inject the adapter twice."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir)).eval()


def score_hf(
    model: Any, adapter_dir: Path | None, sequences: Sequence[FrozenSequence], *, device: str,
    arm: str = "hf_base", autocast_arm: bool = False, wrapped: Any = None,
) -> Scores:
    """Return `hf_lora` and `hf_base` (the same PeftModel, adapter disabled), or a plain model alone, without an
    adapter, scored as `arm`."""
    if adapter_dir is None:
        if arm not in PLAIN_HF_ARMS or autocast_arm:
            raise ParityError(f"a plain model is scored as one of {PLAIN_HF_ARMS}, without an autocast arm")
        arms = [(arm, model, contextlib.nullcontext, None)]
    else:
        peft_model = wrapped if wrapped is not None else wrap_adapter(model, adapter_dir)
        arms = [
            ("hf_lora", peft_model, contextlib.nullcontext, None),
            ("hf_base", peft_model, peft_model.disable_adapter, None),
        ]
        if autocast_arm:
            arms.append((HF_LORA_AC_ARM, peft_model, contextlib.nullcontext, AUTOCAST_DTYPE))
    scores = Scores()
    for name, scorer, context, autocast in arms:
        with context():
            scored = {
                seq.seq_id: hf_sequence_scores(scorer, seq.token_ids, device=device, autocast_dtype=autocast)
                for seq in sequences
            }
        scores.logprobs[name] = {k: v[0] for k, v in scored.items()}
        scores.argmax[name] = {k: v[1] for k, v in scored.items()}
    return scores


def _position_scores(
    entries: Any, token_ids: Sequence[int], first: int, seq_id: str
) -> tuple[list[float], list[int | None]]:
    """vLLM's logprob of `token_ids[i]` and its rank-1 token, from `entries[i]`, i = first..len-1."""
    if entries is None or len(entries) != len(token_ids):
        raise ParityError(f"{seq_id}: vLLM returned no logprob per token")
    values: list[float] = []
    argmax: list[int | None] = []
    for position in range(first, len(token_ids)):
        entry = entries[position]
        token = token_ids[position]
        if entry is None or token not in entry:
            raise ParityError(f"{seq_id}: no logprob for token {token} at position {position}")
        values.append(float(entry[token].logprob))
        argmax.append(next((int(t) for t, lp in entry.items() if getattr(lp, "rank", None) == 1), None))
    return values, argmax


def vllm_prompt_scores(llm: Any, sequences: Sequence[FrozenSequence], *, lora_request: Any | None = None) -> Scores:
    """Return teacher-forced logprobs and argmax from vLLM, each sequence sent whole as a prompt; arm `_`."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    prompts = [{"prompt_token_ids": list(seq.token_ids)} for seq in sequences]
    outputs = llm.generate(
        prompts,
        sampling_params=params,
        use_tqdm=False,
        **({"lora_request": lora_request} if lora_request is not None else {}),
    )
    if len(outputs) != len(sequences):
        raise ParityError(f"vLLM returned {len(outputs)} outputs for {len(sequences)} sequences")
    scored = {
        seq.seq_id: _position_scores(output.prompt_logprobs, seq.token_ids, 1, seq.seq_id)
        for seq, output in zip(sequences, outputs)
    }
    return Scores({"_": {k: v[0] for k, v in scored.items()}}, {"_": {k: v[1] for k, v in scored.items()}})


def _named(scores: Scores, arm: str) -> Scores:
    (logprobs,), (argmax,) = scores.logprobs.values(), scores.argmax.values()
    return Scores({arm: logprobs}, {arm: argmax})


def score_servings(llm: Any, servings: Sequence[tuple[str, Any]], sequences: Sequence[FrozenSequence]) -> Scores:
    """Return the scores of each `(arm, lora_request or None)` serving of one engine over `sequences`."""
    scores = Scores()
    for arm, request in servings:
        scores = scores.merge(_named(vllm_prompt_scores(llm, sequences, lora_request=request), arm))
    return scores


def _render(tokenizer: Any, prompts: Sequence[Mapping[str, str]], gpt_oss_reasoning_level: str | None) -> list[str]:
    from ruleforge.judges.models import format_prompt_for_generation

    return [
        format_prompt_for_generation(tokenizer, dict(entry), gpt_oss_reasoning_level=gpt_oss_reasoning_level)
        for entry in prompts
    ]


def _sampling_params(*, max_tokens: int, seed: int, truncation: Mapping[str, float | int] | None, **extra: Any) -> Any:
    from vllm import SamplingParams

    from ruleforge.judges.models import UNTRUNCATED_SAMPLING

    effective = {**UNTRUNCATED_SAMPLING, **dict(truncation or {})}
    return SamplingParams(temperature=1.0, max_tokens=max_tokens, seed=seed, skip_special_tokens=False, **effective, **extra)


def sample_sequences(
    llm: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, str]],
    *,
    max_tokens: int,
    seed: int,
    truncation: Mapping[str, float | int] | None = None,
    gpt_oss_reasoning_level: str | None = None,
) -> list[FrozenSequence]:
    """Sample one completion per prompt, freezing its ids: the prefill arm's sequences."""
    formatted = _render(tokenizer, prompts, gpt_oss_reasoning_level)
    params = [
        _sampling_params(max_tokens=max_tokens, seed=seed + index, truncation=truncation)
        for index in range(len(formatted))
    ]
    outputs = llm.generate(formatted, sampling_params=params, use_tqdm=False)
    return [
        FrozenSequence(
            seq_id=f"seq-{index:04d}",
            prompt_ids=tuple(output.prompt_token_ids),
            completion_ids=tuple(output.outputs[0].token_ids),
            finish_reason=output.outputs[0].finish_reason,
        )
        for index, output in enumerate(outputs)
    ]


def sample_decode(
    llm: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, str]],
    *,
    lora_request: Any | None = None,
    max_tokens: int,
    seed: int,
    batch_size: int,
    truncation: Mapping[str, float | int] | None = None,
    gpt_oss_reasoning_level: str | None = None,
    prefix: str = "dec",
) -> tuple[list[FrozenSequence], Scores]:
    """Sample on the decode path, keeping vLLM's logprob and argmax of every token; arm `_`."""
    if batch_size < 1:
        raise ParityError(f"batch_size must be positive, got {batch_size}")
    formatted = _render(tokenizer, prompts, gpt_oss_reasoning_level)
    sequences: list[FrozenSequence] = []
    logprobs: dict[str, list[float]] = {}
    argmax: dict[str, list[int | None]] = {}
    for start in range(0, len(formatted), batch_size):
        chunk = formatted[start : start + batch_size]
        params = [
            _sampling_params(max_tokens=max_tokens, seed=seed + start + offset, truncation=truncation,
                             logprobs=1, prompt_logprobs=1)
            for offset in range(len(chunk))
        ]
        outputs = llm.generate(
            chunk, sampling_params=params, use_tqdm=False,
            **({"lora_request": lora_request} if lora_request is not None else {}),
        )
        if len(outputs) != len(chunk):
            raise ParityError(f"vLLM returned {len(outputs)} outputs for {len(chunk)} prompts")
        for offset, output in enumerate(outputs):
            completion = output.outputs[0]
            seq = FrozenSequence(
                seq_id=f"{prefix}-{start + offset:04d}",
                prompt_ids=tuple(output.prompt_token_ids),
                completion_ids=tuple(completion.token_ids),
                finish_reason=completion.finish_reason,
            )
            prompt_part = _position_scores(output.prompt_logprobs, seq.prompt_ids, 1, seq.seq_id)
            decode_part = _position_scores(completion.logprobs, seq.completion_ids, 0, seq.seq_id)
            logprobs[seq.seq_id] = prompt_part[0] + decode_part[0]
            argmax[seq.seq_id] = prompt_part[1] + decode_part[1]
            sequences.append(seq)
    return sequences, Scores({"_": logprobs}, {"_": argmax})


SEGMENTS = ("all", "completion")


@dataclass(frozen=True)
class ParityThresholds:
    """The gate settings `compare` applies; `PARITY_PRESETS` holds each check's."""

    noise_factor: float = 1.5
    abs_tolerance: float = 1e-3
    effect_ratio: tuple[float, float] | None = None
    min_effect: float = 1e-3
    paired_delta: float = 0.15
    min_effect_noise_ratio: float | None = None
    require_effect: bool = True
    rho_range: tuple[float, float] | None = None
    rho_hf_range: tuple[float, float] | None = None
    rho_precision: float | None = None
    rho_per_bucket: bool = False
    rho_report_threshold: float | None = None
    rho_ci_level: float = 0.95
    rho_resamples: int = 2000
    rho_seed: int = 0
    greedy_agreement: float | None = 0.95
    bucket_edges: tuple[int, ...] = (512, 2048, 4096)
    min_bucket_tokens: int = 50
    degenerate_run: int = 32
    require_hf_float32: bool = True

    def __post_init__(self) -> None:
        if not self.paired_delta > 0:
            raise ParityError(f"paired_delta must be positive, got {self.paired_delta}")
        floor = PAIRED_NOISE_FACTOR / self.paired_delta
        ratio = 2.0 / self.paired_delta if self.min_effect_noise_ratio is None else float(self.min_effect_noise_ratio)
        if ratio < floor:
            raise ParityError(
                f"min_effect_noise_ratio {ratio:g} is under sqrt(2)/paired_delta = {floor:.3g}: a faithful "
                "adapter's paired statistic sits near sqrt(2) x the noise, so the paired gate would fail "
                "it where the minimum effect calls it testable"
            )
        object.__setattr__(self, "min_effect_noise_ratio", ratio)
        for name in ("effect_ratio", "rho_range", "rho_hf_range"):
            bounds = getattr(self, name)
            if bounds is not None and not bounds[0] <= bounds[1]:
                raise ParityError(f"{name} {bounds} is empty")


@dataclass
class BucketStats:
    tokens: int = 0
    noise_base: float = math.nan
    decode_noise: float = math.nan
    noise: float = math.nan
    lora_gap: float = math.nan
    effect_hf: float = math.nan
    effect_vllm: float = math.nan
    effect_noise_ratio: float = math.nan
    #: The serving gate's limit on ``lora_gap``: max(noise_factor * noise, abs_tolerance).
    serving_limit: float = math.nan
    #: HF effects the gates call testable but a faithful adapter can still fail
    #: the paired gate at (a false alarm): [the minimum testable effect,
    #: (serving_limit + noise_base) / paired_delta]; empty when low > high.
    effect_window: tuple[float, float] = (math.nan, math.nan)
    effect_ratio: float = math.nan
    paired_delta: float = math.nan
    paired_delta_ratio: float = math.nan
    rho: float = math.nan
    rho_ci: tuple[float, float] = (math.nan, math.nan)
    rho_halfwidth: float = math.nan
    correlation: float = math.nan
    greedy_served: float = math.nan
    greedy_base: float = math.nan
    #: With ``hf_merged`` scored: mean |hf_merged - served|, the serving gate's statistic (the
    #: same weights on both sides), and rho with ``hf_merged - hf_base`` as the served delta.
    merged_gap: float = math.nan
    rho_hf: float = math.nan
    rho_hf_ci: tuple[float, float] = (math.nan, math.nan)
    rho_hf_halfwidth: float = math.nan
    #: With ``hf_lora_ac`` scored: mean and p99 of |hf_lora_ac - hf_lora| (descriptive).
    autocast_gap: float = math.nan
    autocast_gap_p99: float = math.nan


@dataclass
class ParityReport:
    passed: bool
    failures: list[str]
    thresholds: dict[str, Any]
    mode: str
    served_arm: str
    sequences: int
    nonfinite: dict[str, int]
    segment: str = "all"
    buckets: dict[str, BucketStats] = field(default_factory=dict)
    segments: dict[str, BucketStats] = field(default_factory=dict)
    degenerate: dict[str, dict[str, int]] = field(default_factory=dict)
    inconclusive: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: What the score files recorded of how each arm was scored (:func:`score_inputs`).
    inputs: dict[str, Any] = field(default_factory=dict)
    #: ``{bucket: whether rho was gated}`` for each bucket where a rho range
    #: applied: False where its interval was wider than ``rho_precision`` and
    #: straddled a bound of the range, or the HF effect was under ``min_effect``.
    rho_gated: dict[str, bool] = field(default_factory=dict)
    #: The HF arm the serving gate compares the served arm with: ``hf_merged`` when scored, else ``hf_lora``.
    serving_reference: str = "hf_lora"
    #: ``{"all": whether the overall rho meets rho_report_threshold}`` when one is set (gates nothing).
    rho_threshold_met: dict[str, bool] = field(default_factory=dict)
    #: ``rho_gated``'s record for rho_hf, where ``rho_hf_range`` applied: False only where the HF effect
    #: was under ``min_effect`` (a wide interval fails rho_hf, :func:`_rho_failures`).
    rho_hf_gated: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Token:
    """One scored position: its sequence, the arms' logprobs and argmax agreements (None: unknown)."""

    seq: int
    hb: float
    hl: float
    vb: float
    vs: float
    served_agrees: bool | None
    base_agrees: bool | None
    hm: float | None = None
    hl_ac: float | None = None


def _bucket_label(position: int, edges: Sequence[int]) -> str:
    low = 0
    for edge in edges:
        if position < edge:
            return f"[{low},{edge})"
        low = edge
    return f"[{low},inf)"


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def _share(flags: Iterable[bool | None]) -> float:
    known = [flag for flag in flags if flag is not None]
    return sum(known) / len(known) if known else math.nan


def _served_delta(token: _Token) -> float:
    """dM: the served arm's effect, ``served - vllm_base``."""
    return token.vs - token.vb


def _merged_delta(token: _Token) -> float:
    """The merge's effect on HF's arithmetic, ``hf_merged - hf_base`` (rho_hf's dM)."""
    return token.hm - token.hb


def _rho(
    tokens: Sequence[_Token], n_sequences: int, thresholds: ParityThresholds, served: Any = _served_delta,
) -> tuple[float, tuple[float, float], float]:
    """rho = sum dM*dH / sum dH^2, and its percentile bootstrap interval over sequences."""
    import numpy as np

    num, den = np.zeros(n_sequences), np.zeros(n_sequences)
    for token in tokens:
        d_h = token.hl - token.hb
        num[token.seq] += served(token) * d_h
        den[token.seq] += d_h * d_h
    if den.sum() <= 0:
        return math.nan, (math.nan, math.nan), math.nan
    rho = float(num.sum() / den.sum())
    rng = np.random.default_rng(thresholds.rho_seed)
    picks = rng.integers(0, n_sequences, size=(thresholds.rho_resamples, n_sequences))
    sums = den[picks].sum(axis=1)
    if not (sums > 0).any():
        return rho, (math.nan, math.nan), math.nan
    resampled = num[picks].sum(axis=1)[sums > 0] / sums[sums > 0]
    tail = (1.0 - thresholds.rho_ci_level) / 2.0 * 100.0
    low, high = (float(x) for x in np.percentile(resampled, [tail, 100.0 - tail]))
    return rho, (low, high), (high - low) / 2.0


def _correlation(tokens: Sequence[_Token]) -> float:
    import numpy as np

    if len(tokens) < 2:
        return math.nan
    d_m = np.array([t.vs - t.vb for t in tokens])
    d_h = np.array([t.hl - t.hb for t in tokens])
    if d_m.std() == 0 or d_h.std() == 0:
        return math.nan
    return float(np.corrcoef(d_m, d_h)[0, 1])


def _stats(tokens: Sequence[_Token], n_sequences: int, thresholds: ParityThresholds, decode_noise: float = math.nan) -> BucketStats:
    if not tokens:
        return BucketStats()
    noise_base = _mean(abs(t.hb - t.vb) for t in tokens)
    noise = noise_base + (0.0 if math.isnan(decode_noise) else decode_noise)
    effect_hf = _mean(abs(t.hl - t.hb) for t in tokens)
    effect_vllm = _mean(abs(t.vs - t.vb) for t in tokens)
    paired = _mean(abs((t.hl - t.hb) - (t.vs - t.vb)) for t in tokens)
    rho, rho_ci, halfwidth = _rho(tokens, n_sequences, thresholds)
    serving_limit = max(thresholds.noise_factor * noise, thresholds.abs_tolerance)
    optional: dict[str, Any] = {}
    if tokens[0].hm is not None:  # an optional arm is scored at every position or none (compare's arms)
        rho_hf, rho_hf_ci, rho_hf_halfwidth = _rho(tokens, n_sequences, thresholds, _merged_delta)
        optional = {
            "merged_gap": _mean(abs(t.hm - t.vs) for t in tokens),
            "rho_hf": rho_hf, "rho_hf_ci": rho_hf_ci, "rho_hf_halfwidth": rho_hf_halfwidth,
        }
    if tokens[0].hl_ac is not None:
        import numpy as np

        gaps = [abs(t.hl_ac - t.hl) for t in tokens]
        optional.update(autocast_gap=_mean(gaps), autocast_gap_p99=float(np.percentile(gaps, 99)))
    return BucketStats(
        tokens=len(tokens),
        noise_base=noise_base,
        decode_noise=decode_noise,
        noise=noise,
        lora_gap=_mean(abs(t.hl - t.vs) for t in tokens),
        effect_hf=effect_hf,
        effect_vllm=effect_vllm,
        effect_noise_ratio=effect_hf / noise_base if noise_base > 0 else math.inf,
        serving_limit=serving_limit,
        # the triangle bound: paired <= lora_gap + noise_base
        effect_window=(
            max(thresholds.min_effect, thresholds.min_effect_noise_ratio * noise),
            (serving_limit + noise_base) / thresholds.paired_delta,
        ),
        effect_ratio=effect_vllm / effect_hf if effect_hf > 0 else math.nan,
        paired_delta=paired,
        paired_delta_ratio=paired / effect_hf if effect_hf > 0 else math.nan,
        rho=rho,
        rho_ci=rho_ci,
        rho_halfwidth=halfwidth,
        correlation=_correlation(tokens),
        greedy_served=_share(t.served_agrees for t in tokens),
        greedy_base=_share(t.base_agrees for t in tokens),
        **optional,
    )


def longest_run(token_ids: Sequence[int]) -> tuple[int | None, int, int]:
    """Return (token, length, start) of the longest run of one repeated token; (None, 0, 0) when empty."""
    best = (None, 0, 0)
    run_start = 0
    for index in range(1, len(token_ids) + 1):
        if index == len(token_ids) or token_ids[index] != token_ids[run_start]:
            if index - run_start > best[1]:
                best = (token_ids[run_start], index - run_start, run_start)
            run_start = index
    return best


def _effect_measurable(label: str, stats: BucketStats, thresholds: ParityThresholds) -> str | None:
    if not stats.effect_hf >= thresholds.min_effect:
        return (
            f"{label}: the adapter moves HF by {stats.effect_hf:.2e} nats/token, under "
            f"{thresholds.min_effect:.0e}: too weak to test (or a no-op under PEFT too)"
        )
    if not stats.effect_hf >= thresholds.min_effect_noise_ratio * stats.noise:
        return (
            f"{label}: the adapter moves HF by {stats.effect_hf:.2e} nats/token, under "
            f"{thresholds.min_effect_noise_ratio:.3g} x the noise {stats.noise:.2e}: too weak to test, "
            "the gates would compare engine noise, not the adapter"
        )
    return None


def _bucket_failures(
    label: str, stats: BucketStats, thresholds: ParityThresholds, served: str, report: ParityReport
) -> list[str]:
    """Why one bucket (or ``all``) fails; empty when it passes. Notes inconclusive buckets on ``report``."""
    if stats.tokens == 0:
        return [f"{label}: no finite tokens to compare"]
    failures = []
    reference = report.serving_reference
    gap = stats.merged_gap if reference == HF_MERGED_ARM else stats.lora_gap
    if not gap <= stats.serving_limit:
        failures.append(
            f"{label}: |{reference} - {served}| {gap:.2e} exceeds {stats.serving_limit:.2e} "
            f"({thresholds.noise_factor} x the noise {stats.noise:.2e})"
        )
    if thresholds.greedy_agreement is not None and not math.isnan(stats.greedy_served):
        if not stats.greedy_served >= thresholds.greedy_agreement:
            failures.append(
                f"{label}: {reference} and {served} agree on the most likely token at {stats.greedy_served:.3f} of "
                f"positions, under {thresholds.greedy_agreement} (the base arms: {stats.greedy_base:.3f})"
            )
    weak = _effect_measurable(label, stats, thresholds)
    if weak is not None:
        if thresholds.require_effect:
            failures.append(weak)
        else:
            report.inconclusive.append(weak)
    else:
        failures.extend(_effect_failures(label, stats, thresholds, served))
    if label == "all" or thresholds.rho_per_bucket:
        for name in RETENTION_STATISTICS:
            if getattr(thresholds, f"{name}_range") is not None:
                failures.extend(_rho_failures(label, stats, thresholds, served, report, name))
    return failures


def _effect_failures(label: str, stats: BucketStats, thresholds: ParityThresholds, served: str) -> list[str]:
    """The gates that need an effect the paired statistic can resolve: the effect ratio and the paired delta."""
    failures = []
    no_op = " -- the silent no-op signature (vLLM #48019/#49354)" if stats.effect_ratio < 0.1 else ""
    if thresholds.effect_ratio is not None:
        low, high = thresholds.effect_ratio
        if not low <= stats.effect_ratio <= high:
            failures.append(f"{label}: effect ratio vLLM/HF {stats.effect_ratio:.3f} outside [{low}, {high}]{no_op}")
    if not stats.paired_delta_ratio <= thresholds.paired_delta:
        failures.append(
            f"{label}: paired delta mean|(hl-hb)-(vs-vb)| is {stats.paired_delta_ratio:.3f} of the HF "
            f"effect, over {thresholds.paired_delta}: {served} applies another delta than HF "
            f"(a wrong sign, an unrelated delta, a module family dropped); effect ratio "
            f"{stats.effect_ratio:.3f}{no_op}"
        )
    return failures


#: The retention statistics a range gates: rho (the served arm's, ``rho_range``) and rho_hf (the merge's,
#: ``rho_hf_range``), each a :class:`BucketStats` field with ``_ci`` and ``_halfwidth`` beside it.
RETENTION_STATISTICS = ("rho", "rho_hf")


def _rho_failures(
    label: str, stats: BucketStats, thresholds: ParityThresholds, served: str, report: ParityReport, name: str = "rho",
) -> list[str]:
    """The retention gate: `name` (`RETENTION_STATISTICS`) within its range; not gated where the HF effect is under
    `min_effect`."""
    gated = report.rho_gated if name == "rho" else report.rho_hf_gated
    if not stats.effect_hf >= thresholds.min_effect:
        gated[label] = False
        report.notes.append(f"{label}: {name} not gated, the HF effect {stats.effect_hf:.2e} is under "
                            f"{thresholds.min_effect:.0e}: nothing to regress on")
        return []
    value, (ci_low, ci_high) = getattr(stats, name), getattr(stats, f"{name}_ci")
    halfwidth = getattr(stats, f"{name}_halfwidth")
    low, high = getattr(thresholds, f"{name}_range")
    precise = thresholds.rho_precision is None or halfwidth <= thresholds.rho_precision
    interval = f"(95% interval {ci_low:.4f}-{ci_high:.4f}, half-width {halfwidth:.4f})"
    if name == "rho_hf":
        gated[label] = True
        if not precise:
            return [
                f"{label}: rho_hf {value:.4f} {interval}: the interval is wider than {thresholds.rho_precision}, "
                f"where a correct fp32 merge's is not: {HF_MERGED_ARM}'s effect is not one fixed multiple of the "
                "trained one -- the merge mis-scales, drops or misplaces a part whose share varies by prompt"
            ]
        if low <= value <= high:
            return []
        return [
            f"{label}: rho_hf {value:.4f} {interval} outside [{low}, {high}]: {HF_MERGED_ARM} does not apply the "
            f"trained delta -- the merge mis-scales, drops or misplaces it, which the serving gate (the same "
            f"weights in {served}) cannot see"
        ]
    excluded = ci_high < low or ci_low > high
    gated[label] = precise or excluded or low <= ci_low <= ci_high <= high
    if not gated[label]:
        report.notes.append(
            f"{label}: rho {value:.4f} not gated, its interval {ci_low:.4f}-{ci_high:.4f} (half-width "
            f"{halfwidth:.4f}, over {thresholds.rho_precision}) straddles a bound of [{low}, {high}]"
        )
        return []
    if excluded or (precise and not low <= value <= high):
        return [f"{label}: retention rho {value:.4f} {interval} outside [{low}, {high}]: {served} does not carry "
                "the trained adapter's effect"]
    return []


def _check_arm(scores: Scores, arm: str, sequences: Sequence[FrozenSequence]) -> None:
    for seq in sequences:
        values = scores.logprobs[arm].get(seq.seq_id)
        if values is None or len(values) != len(seq.token_ids) - 1:
            raise ParityError(f"{arm} has no {len(seq.token_ids) - 1} logprobs for {seq.seq_id}")


def _agrees(scores: Scores, a: str, b: str, seq_id: str, offset: int) -> bool | None:
    first, second = scores.argmax.get(a, {}).get(seq_id), scores.argmax.get(b, {}).get(seq_id)
    if first is None or second is None or first[offset] is None or second[offset] is None:
        return None
    return first[offset] == second[offset]


def served_arm(scores: Scores) -> tuple[str, str]:
    """Return `(mode, arm)` of the one served arm the scores hold."""
    present = [(mode, arm) for mode, arm in SERVED_ARMS.items() if arm in scores.logprobs]
    if len(present) != 1:
        raise ParityError(f"the scores must hold exactly one served arm of {sorted(SERVED_ARMS.values())}")
    return present[0]


@dataclass(frozen=True)
class DecodeFloor:
    """The decode path's noise floor and how its reference was sampled."""

    buckets: Mapping[str, tuple[int, float]]
    meta: Mapping[str, Any] | None = None
    #: `check_scored`'s notes on the reference's scores.
    notes: tuple[str, ...] = ()


def decode_noise_floor(
    reference: Scores, sequences: Sequence[FrozenSequence], thresholds: ParityThresholds = ParityThresholds()
) -> DecodeFloor:
    """Return the decode path's noise, mean |vb_decode - vb_prefill| over the reference's completion positions, per
    bucket and `all`; the reference's scores must be of `sequences` (`check_scored`)."""
    notes = check_scored(reference, sequences)
    for arm in (BASE_DECODE_ARM, BASE_ARM):
        if arm not in reference.logprobs:
            raise ParityError(f"the decode reference has no {arm} scores")
        _check_arm(reference, arm, sequences)
    per_bucket: dict[str, list[float]] = {"all": []}
    for seq in sequences:
        decoded, prefilled = reference.logprobs[BASE_DECODE_ARM][seq.seq_id], reference.logprobs[BASE_ARM][seq.seq_id]
        for offset in range(len(seq.prompt_ids) - 1, len(seq.token_ids) - 1):
            gap = abs(decoded[offset] - prefilled[offset])
            if math.isfinite(gap):
                per_bucket["all"].append(gap)
                per_bucket.setdefault(_bucket_label(offset + 1, thresholds.bucket_edges), []).append(gap)
    return DecodeFloor(
        {label: (len(gaps), _mean(gaps)) for label, gaps in per_bucket.items()},
        reference.meta.get(BASE_DECODE_ARM), tuple(notes),
    )


#: The HF arms' dtype the gate expects (``ParityThresholds.require_hf_float32``).
HF_REFERENCE_DTYPE = "float32"


#: What a vLLM score file records of the engine that scored it (:func:`engine_record`),
#: each of which must agree across the vLLM arms compared and the decode reference.
ENGINE_FIELDS = (
    "engine_dtype", "kv_cache_dtype", "precision_env", "attention_backend", "vllm_version", "max_num_batched_tokens",
)


def score_inputs(
    scores: Scores, served: str, thresholds: ParityThresholds, decode_floor: DecodeFloor | None = None
) -> dict[str, Any]:
    """Return what the score files recorded of how the compared arms were scored; unlike inputs are refused."""
    files = {arm: scores.meta[arm] for arm in (BASE_ARM, served) if arm in scores.meta}
    reference = None if decode_floor is None else decode_floor.meta
    if reference is not None:
        files["decode_reference"] = reference
    inputs: dict[str, Any] = {}
    for name in ENGINE_FIELDS:
        recorded = {arm: meta.get(name) for arm, meta in files.items()}
        if len({json.dumps(value, sort_keys=True) for value in recorded.values()}) > 1:
            raise ParityError(
                f"the vLLM score files were scored on engines with different {name} {recorded}: their "
                "difference would be read as the adapter's effect; score every vLLM arm on one engine "
                "configuration"
            )
        inputs[name] = recorded
    hf = {arm: scores.meta[arm].get("dtype") for arm in (*HF_ARMS, HF_LORA_AC_ARM) if arm in scores.meta}
    other = {arm: dtype for arm, dtype in hf.items() if dtype != HF_REFERENCE_DTYPE}
    if other and thresholds.require_hf_float32:
        raise ParityError(
            f"HF arm(s) scored in {other}, not {HF_REFERENCE_DTYPE}: the reference's own rounding noise "
            "would bias rho towards 0; score them with --dtype float32, or allow another dtype "
            "(--allow-hf-dtype) on the record"
        )
    inputs["hf_dtype"] = hf
    if HF_MERGED_ARM in scores.meta and served in scores.meta:
        merged, engine = scores.meta[HF_MERGED_ARM], scores.meta[served]
        if merged.get("dtype") != engine.get("engine_dtype"):
            raise ParityError(
                f"{HF_MERGED_ARM} was scored in {merged.get('dtype')}, {served}'s engine serves "
                f"{engine.get('engine_dtype')}: HF_merged loads the merged weights in the dtype they are served in"
            )
        paths = [meta.get("model_path") for meta in (merged, engine)]
        if None not in paths and Path(paths[0]).resolve() != Path(paths[1]).resolve():
            raise ParityError(
                f"{HF_MERGED_ARM} scored {paths[0]}, {served} served {paths[1]}: the serving gate compares one "
                "set of weights on two engines"
            )
        inputs["hf_merged"] = {"dtype": merged.get("dtype"), "model_path": paths[0]}
    if decode_floor is not None:
        batches = {
            "decode_arm": scores.meta[served].get("decode_batch") if served in scores.meta else None,
            "reference": None if reference is None else reference.get("decode_batch"),
        }
        if served in scores.meta and reference is not None and batches["decode_arm"] != batches["reference"]:
            raise ParityError(
                f"the decode arm was sampled at decode batch {batches['decode_arm']}, its reference at "
                f"{batches['reference']}: the floor is the decode path's noise at its own batch; sample both "
                "at one --decode-batch"
            )
        inputs["decode_batch"] = batches
    return inputs


def compare(
    scores: Scores,
    sequences: Sequence[FrozenSequence],
    thresholds: ParityThresholds = ParityThresholds(),
    *,
    segment: str = "all",
    decode_floor: DecodeFloor | None = None,
) -> ParityReport:
    """Return the report of the HF arms gated against the vLLM arms over `sequences`."""
    if segment not in SEGMENTS:
        raise ParityError(f"segment must be one of {SEGMENTS}, got {segment!r}")
    if decode_floor is not None and segment != "completion":
        raise ParityError("a decode floor applies to the decode arm's completion positions (segment='completion')")
    missing = [arm for arm in (*HF_ARMS, BASE_ARM) if arm not in scores.logprobs]
    if missing:
        raise ParityError(f"no scores for arm(s) {missing}")
    mode, served = served_arm(scores)
    optional = tuple(arm for arm in OPTIONAL_HF_ARMS if arm in scores.logprobs)
    if HF_MERGED_ARM in optional and mode != "merged":
        raise ParityError(f"{HF_MERGED_ARM} is the merged mode's reference; the scores serve {served}")
    if thresholds.rho_hf_range is not None and HF_MERGED_ARM not in optional:
        raise ParityError(
            f"the gates hold rho_hf within {thresholds.rho_hf_range} (parity B) and the scores have no "
            f"{HF_MERGED_ARM}: score the merged directory in HF (hf --arm {HF_MERGED_ARM} --model-path $MERGED); "
            "without it a merge that over-applies the delta passes wherever the paired gate is skipped"
        )
    inputs = score_inputs(scores, served, thresholds, decode_floor)
    notes = [*(() if decode_floor is None else decode_floor.notes), *check_scored(scores, sequences)]
    arms = (*HF_ARMS, BASE_ARM, served, *optional)
    for arm in arms:
        _check_arm(scores, arm, sequences)
    reference = HF_MERGED_ARM if HF_MERGED_ARM in optional else "hf_lora"

    nonfinite = {arm: sum(1 for seq in sequences for v in scores.logprobs[arm][seq.seq_id] if not math.isfinite(v)) for arm in arms}
    per_bucket: dict[str, list[_Token]] = {}
    per_segment: dict[str, list[_Token]] = {"prompt": [], "completion": []}
    gated: list[_Token] = []
    degenerate: dict[str, dict[str, int]] = {}
    for index, seq in enumerate(sequences):
        columns = [scores.logprobs[arm][seq.seq_id] for arm in arms]
        for offset, row in enumerate(zip(*columns)):
            if not all(math.isfinite(v) for v in row):
                continue
            extra = dict(zip(optional, row[4:]))
            token = _Token(
                index, *row[:4],
                served_agrees=_agrees(scores, reference, served, seq.seq_id, offset),
                base_agrees=_agrees(scores, "hf_base", BASE_ARM, seq.seq_id, offset),
                hm=extra.get(HF_MERGED_ARM),
                hl_ac=extra.get(HF_LORA_AC_ARM),
            )
            position = offset + 1
            part = "prompt" if position < len(seq.prompt_ids) else "completion"
            per_segment[part].append(token)
            if segment == "completion" and part == "prompt":
                continue
            gated.append(token)
            per_bucket.setdefault(_bucket_label(position, thresholds.bucket_edges), []).append(token)
        run_token, run, start = longest_run(seq.completion_ids)
        if thresholds.degenerate_run and run >= thresholds.degenerate_run:
            degenerate[seq.seq_id] = {"token": run_token, "run": run, "start": start}

    def floor(label: str) -> float:
        if decode_floor is None:
            return math.nan
        buckets = decode_floor.buckets
        tokens, mean = buckets.get(label, (0, math.nan))
        return mean if tokens >= thresholds.min_bucket_tokens or label == "all" else buckets["all"][1]

    n = len(sequences)
    report = ParityReport(
        passed=False,
        failures=[],
        thresholds=thresholds_to_json(thresholds),
        mode=mode,
        served_arm=served,
        sequences=n,
        nonfinite=nonfinite,
        segment=segment,
        buckets={
            "all": _stats(gated, n, thresholds, floor("all")),
            **{
                label: _stats(tokens, n, thresholds, floor(label))
                for label, tokens in sorted(per_bucket.items(), key=lambda kv: int(kv[0][1:].split(",")[0]))
            },
        },
        segments={label: _stats(tokens, n, thresholds) for label, tokens in per_segment.items()},
        degenerate=degenerate,
        notes=notes,
        inputs=inputs,
        serving_reference=reference,
    )
    failures = report.failures
    if any(nonfinite.values()):
        failures.append(f"non-finite logprobs {nonfinite} (NaN logits, e.g. vLLM #48590 on Hopper)")
    if thresholds.rho_report_threshold is not None:
        report.rho_threshold_met["all"] = bool(report.buckets["all"].rho >= thresholds.rho_report_threshold)
    if degenerate:
        first = next(iter(degenerate.items()))
        failures.append(
            f"{len(degenerate)} completion(s) repeat one token {thresholds.degenerate_run}+ times in a row "
            f"(first: {first[0]}, token {first[1]['token']} x{first[1]['run']} from {first[1]['start']}): "
            "the degenerate decode of NaN logits (e.g. vLLM #52568)"
        )
    if thresholds.greedy_agreement is not None and math.isnan(report.buckets["all"].greedy_served):
        report.notes.append(f"greedy agreement not measured: no argmax recorded for {reference} and {served}")
    if decode_floor is not None and not decode_floor.buckets["all"][0]:
        report.notes.append("the decode reference has no completion token: no decode floor was added")
    for label, stats in report.buckets.items():
        if label != "all" and stats.tokens < thresholds.min_bucket_tokens:
            continue
        failures.extend(_bucket_failures(label, stats, thresholds, served, report))
    report.passed = not failures
    return report


#: The parity gates, by check and by serving: merged in fp32 (``_fallback``: the bf16 fallback), or LoRA on the bf16
#: engine (``_lora``). ``ParityThresholds`` fields over its defaults; the CLI's explicit flags win.
PARITY_PRESETS: dict[str, dict[str, Any]] = {
    # rho >= 0.99 where its 95% interval's half-width is at most 0.01; an effect too small to test is inconclusive.
    # rho_hf (HF fp32 on both sides) catches a merge that over- or under-applies the delta; a wider interval fails.
    "b": {"rho_range": (0.99, math.inf), "rho_hf_range": (0.99, 1.01), "rho_precision": 0.01, "require_effect": False},
    # 0.90 is a reporting threshold only: bf16 rounding lowers retention without any engine fault.
    "b_fallback": {"rho_report_threshold": 0.90, "require_effect": False, "paired_delta": math.inf},
    # The synthetic adapter's effect is large by construction, so an effect too weak to test fails.
    "m0": {"rho_range": (0.99, 1.01), "rho_precision": 0.01, "rho_per_bucket": True},
    "m0_fallback": {"rho_range": (0.95, 1.05), "rho_precision": 0.01, "rho_per_bucket": True},
    # LoRA serving has no merge and no rho_hf: rho is two-sided, so a no-op (low) and a mis-scaled adapter (high) both
    # fail; 0.95-1.05 is the bf16 engine's window, its noise entering dM.
    "b_lora": {"rho_range": (0.95, 1.05), "rho_precision": 0.01, "require_effect": False},
    "m0_lora": {"rho_range": (0.95, 1.05), "rho_precision": 0.01, "rho_per_bucket": True},
}


def thresholds_to_json(thresholds: ParityThresholds) -> dict[str, Any]:
    """Return every gate setting as strict JSON: tuples as lists, an infinite bound as the string `"inf"`/`"-inf"`."""

    def encode(value: Any) -> Any:
        if isinstance(value, float) and math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return [encode(v) for v in value] if isinstance(value, (tuple, list)) else value

    return {name: encode(value) for name, value in asdict(thresholds).items()}


#: The greedy agreement gate's default: HF_merged and vLLM_merged agree on >= 0.95 of prompts.
GREEDY_VERDICT_AGREEMENT = 0.95
#: The greedy judgment gate's default; flags override it: the minimum agreement, and how a greedy output is parsed to a
#: judgment, by the campaign's validator on the forced prompt.
GREEDY_VERDICT_GATE: dict[str, Any] = {
    "min_agreement": GREEDY_VERDICT_AGREEMENT, "prompt_style": "forced-grpo",
    "forced_json_requirement": "label_only", "strict_think": True,
}


def eos_token_ids(model_path: Path) -> tuple[int, ...]:
    """Return the ids that end a generation: `generation_config.json`'s `eos_token_id`, else `config.json`'s."""
    for name in ("generation_config.json", "config.json"):
        path = Path(model_path) / name
        value = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id") if path.is_file() else None
        if value is not None:
            return tuple(int(v) for v in (value if isinstance(value, list) else [value]))
    raise ParityError(f"{model_path} names no eos_token_id: a greedy generation could not say how it ended")


def prompt_token_ids(
    tokenizer: Any, prompts: Sequence[Mapping[str, str]], gpt_oss_reasoning_level: str | None = None,
) -> list[list[int]]:
    """Return the prompts rendered as the campaign renders them, tokenized once."""
    return [
        list(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in _render(tokenizer, prompts, gpt_oss_reasoning_level)
    ]


def vllm_greedy(
    llm: Any, prompt_ids: Sequence[Sequence[int]], *, max_tokens: int, lora_request: Any | None = None,
) -> list[tuple[list[int], str | None]]:
    """Return greedy (temperature 0) completions from vLLM: `(completion ids, finish reason)` per prompt."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, skip_special_tokens=False)
    outputs = llm.generate([{"prompt_token_ids": list(ids)} for ids in prompt_ids], sampling_params=params,
                           use_tqdm=False, **({"lora_request": lora_request} if lora_request is not None else {}))
    if len(outputs) != len(prompt_ids):
        raise ParityError(f"vLLM returned {len(outputs)} outputs for {len(prompt_ids)} prompts")
    return [(list(output.outputs[0].token_ids), output.outputs[0].finish_reason) for output in outputs]


def hf_greedy(
    model: Any, prompt_ids: Sequence[Sequence[int]], *, max_tokens: int, eos_ids: Sequence[int], device: str,
) -> list[tuple[list[int], str]]:
    """Return greedy completions from a HF model, one prompt at a time, autocast off."""
    import torch
    from transformers import GenerationConfig

    config = GenerationConfig(
        do_sample=False, max_new_tokens=max_tokens, eos_token_id=list(eos_ids), pad_token_id=int(eos_ids[0]),
    )
    results = []
    for ids in prompt_ids:
        input_ids = torch.as_tensor([list(ids)], device=device)
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            output = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), generation_config=config,
            )
        completion = output[0, len(ids):].tolist()
        results.append((completion, "stop" if completion and completion[-1] in eos_ids else "length"))
    return results


def greedy_records(
    prompt_ids: Sequence[Sequence[int]], generated: Sequence[tuple[Sequence[int], str | None]], tokenizer: Any,
    eos_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Return one row per prompt: its digest, the completion ids and text without the ending eos, the finish reason."""
    rows = []
    for index, (ids, (completion, finish)) in enumerate(zip(prompt_ids, generated, strict=True)):
        kept = list(completion)
        while kept and kept[-1] in eos_ids:
            kept.pop()
        rows.append({
            "id": f"greedy-{index:04d}", "prompt_sha256": canonical_digest(list(ids)), "completion_ids": kept,
            "text": tokenizer.decode(kept, skip_special_tokens=False), "finish_reason": finish,
        })
    return rows


def verdict_schema(rules_path: Path, *, prompt_style: str, forced_json_requirement: str, strict_think: bool) -> Any:
    """Return the campaign's validator for the full-set component over a ruleset file."""
    from ruleforge.evaluate.validation import ComponentSchema

    return ComponentSchema.from_rules(
        load_jsonl_records(rules_path), prompt_style=prompt_style,
        forced_json_requirement=forced_json_requirement, strict_think=strict_think,
    )


def greedy_outcome(row: Mapping[str, Any], schema: Any) -> str:
    """Return the full-set judgment a greedy row parses to under `schema`, or `invalid:<reason>`."""
    from ruleforge.evaluate.identity import FULL_RULESET_COMPONENT
    from ruleforge.evaluate.validation import strict_think_answer
    from ruleforge.judges.backend import GenerationOutcome

    raw = str(row["text"])
    answer = strict_think_answer(raw).strip()
    verdict = schema.classify(
        FULL_RULESET_COMPONENT,
        GenerationOutcome(text=answer or None, raw_response=raw, finish_reason=row["finish_reason"]),
    )
    return verdict.judgment if verdict.valid else f"invalid:{verdict.failure_reason}"


def greedy_agreement(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], schema: Any, *,
    minimum: float = GREEDY_VERDICT_AGREEMENT,
) -> dict[str, Any]:
    """Return the gate on two engines' greedy full-set outcomes: they, and their judgments on the prompts valid on both,
    agree on at least `minimum` of the prompts."""
    keys = [[(row["id"], row["prompt_sha256"]) for row in rows] for rows in (first, second)]
    if keys[0] != keys[1] or not keys[0]:
        raise ParityError("the two greedy files do not hold the same prompts in the same order")
    outcomes = [(greedy_outcome(a, schema), greedy_outcome(b, schema)) for a, b in zip(first, second)]
    agree = [a == b for a, b in outcomes]
    invalid = [(a.startswith("invalid:"), b.startswith("invalid:")) for a, b in outcomes]
    valid = [same for same, (bad_a, bad_b) in zip(agree, invalid) if not bad_a and not bad_b]
    agreement, valid_agreement = sum(agree) / len(agree), _share(valid)
    return {
        "passed": agreement >= minimum and valid_agreement >= minimum, "agreement": agreement, "minimum": minimum,
        "prompts": len(agree), "both_valid": len(valid), "valid_agreement": valid_agreement,
        "both_invalid_share": _share(bad_a and bad_b for bad_a, bad_b in invalid),
        "identical_completions": _share(a["completion_ids"] == b["completion_ids"] for a, b in zip(first, second)),
        "disagreements": {row["id"]: list(pair) for row, pair, same in zip(first, outcomes, agree) if not same},
    }


#: M0: a bf16 cast of the synthetic merge may round away at most this share of the delta's non-zero elements.
M0_MAX_ROUNDED_SHARE = 0.01
#: ConditionSpec fields a merged directory's condition may differ in from the base's: the identity changes with the
#: weights digest and the serving dtype, and nothing else.
IDENTITY_MAY_DIFFER = ("condition_id", "extra", "model_path", "model_weights_sha256", "serving_dtype")


def _check(name: str, passed: bool, **detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **detail}


def _load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path))


def _model_default_truncation(model_path: Path) -> Mapping[str, Any]:
    from ruleforge.judges.models import model_default_truncation

    return model_default_truncation(model_path)


def identity_difference(base: Any, merged: Any) -> list[str]:
    """Return the `ConditionSpec` fields two conditions differ in (a field one of them omits counts)."""
    first, second = base.to_dict(), merged.to_dict()
    return sorted(key for key in first.keys() | second.keys() if first.get(key) != second.get(key))


def _merged_digest(directory: Path) -> tuple[str, dict[str, Any]]:
    from ruleforge.judges.weights import MANIFEST_FNAME
    from ruleforge.judges.weights import model_weights_identity

    if model_weights_identity(directory) is None:
        raise ParityError(f"{directory} has no {MANIFEST_FNAME}: not a merged directory")
    manifest = json.loads((Path(directory) / MANIFEST_FNAME).read_text(encoding="utf-8"))
    return manifest["result"]["weights_sha256"], manifest


def plumbing(
    base_dir: Path,
    merged_dir: Path,
    *,
    second_merge: Path | None = None,
    prompts: Sequence[Mapping[str, str]] | None = None,
    conditions: tuple[Any, Any] | None = None,
    max_rounded_share: float = M0_MAX_ROUNDED_SHARE,
    may_differ: Sequence[str] = IDENTITY_MAY_DIFFER,
) -> dict[str, Any]:
    """Return M0's plumbing checks on a merged directory; `passed` when every check that ran passed."""
    digest, manifest = _merged_digest(merged_dir)
    checks = [_check("weights", True, weights_sha256=digest)]
    if second_merge is not None:
        again, other = _merged_digest(second_merge)
        threads = [m["merge"].get("threads") for m in (manifest, other)]
        checks.append(_check("second_merge", again == digest, weights_sha256=again, threads=threads))
    rounded = (manifest["result"].get("bf16_cast") or {}).get("overall", {}).get("rounded_away")
    checks.append(_check(
        "delta_survives_bf16", rounded is not None and rounded <= max_rounded_share,
        rounded_away=rounded, limit=max_rounded_share,
    ))
    truncations = [dict(_model_default_truncation(path)) for path in (base_dir, merged_dir)]
    checks.append(_check("truncation", truncations[0] == truncations[1], base=truncations[0], merged=truncations[1]))
    if prompts:
        ids = [prompt_token_ids(_load_tokenizer(path), prompts) for path in (base_dir, merged_dir)]
        differing = [index for index, (a, b) in enumerate(zip(*ids)) if a != b]
        checks.append(_check("rendering", not differing, prompts=len(prompts), differing=differing))
    if conditions is not None:
        base, merged = conditions
        differs = identity_difference(base, merged)
        unexpected = sorted(set(differs) - set(may_differ))
        named = base.model_weights_sha256 is None and merged.model_weights_sha256 == digest
        checks.append(_check("identity", not unexpected and named, differs=differs, unexpected=unexpected,
                             merged_weights_sha256=merged.model_weights_sha256))
    return {"passed": all(check["passed"] for check in checks), "checks": checks, "merged_dir": str(merged_dir)}


def load_model_config(path: Path):
    """Return a `configs/model/*.yaml` local model config, env interpolations live."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(path))
    if str(cfg.get("model_type")) != "local":
        raise ParityError(f"{path} is not a local model config (model_type {cfg.get('model_type')!r})")
    return cfg


#: vLLM's default: returned logprobs are the raw model's, what HF teacher-forcing computes.
RAW_LOGPROBS_MODE = "raw_logprobs"


def build_vllm_engine(
    model_cfg: Any,
    adapter_dir: Path | None = None,
    *,
    model_path: Path | None = None,
    dtype: str | None = None,
    engine_args: Mapping[str, Any] | None = None,
):
    """Return the campaign's engine for `model_cfg`: `(llm, tokenizer, lora_request, adapter metadata)`."""
    from ruleforge.judges.backend import local_engine_settings
    from ruleforge.judges.lora import lora_serving
    from ruleforge.judges.models import init_local_model

    require("starting the parity engine", "vllm")  # before its engine arguments are read off vLLM
    settings = local_engine_settings(model_cfg)
    settings["engine_args"] = {**settings["engine_args"], **dict(engine_args or {})}
    if model_path is not None:
        settings["model_path"] = str(model_path)
    if dtype is not None:
        settings["dtype"] = dtype
    mode = settings["engine_args"].get("logprobs_mode", RAW_LOGPROBS_MODE)
    if mode != RAW_LOGPROBS_MODE:
        raise ParityError(f"logprobs_mode {mode!r}: the parity arms compare raw logprobs ({RAW_LOGPROBS_MODE})")
    serving = lora_serving(None if adapter_dir is None else Path(adapter_dir) / ADAPTER_WEIGHTS_FNAME)
    llm, tokenizer = init_local_model(**settings, **serving.engine_settings())
    return llm, tokenizer, serving.request, serving.metadata


def check_engine_dtype(engine: Mapping[str, Any], expected: str | None) -> None:
    """Refuse an engine whose model or KV-cache dtype is not `expected` (None: any)."""
    if expected is None:
        return
    model, cache = engine.get("engine_dtype"), engine.get("kv_cache_dtype")
    if model != expected:
        raise ParityError(f"the engine serves the model in {model}, not {expected}")
    if cache != expected:
        raise ParityError(f"the engine keeps its KV cache in {cache}, not {expected}")


def _package_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def engine_record(llm: Any, attention_backend: Any) -> dict[str, Any]:
    """Return how an engine computes, as its score files record it: every one of `ENGINE_FIELDS`."""
    from ruleforge.judges.backend import precision_env

    scheduler = getattr(vllm_config(llm), "scheduler_config", None)
    return {
        "engine_dtype": engine_dtype(llm),
        "kv_cache_dtype": kv_cache_dtype(llm),
        "precision_env": precision_env(),
        "attention_backend": None if attention_backend is None else str(attention_backend),
        "vllm_version": _package_version("vllm"),
        "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None),
    }


def _parse_engine_args(pairs: Sequence[str]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ParityError(f"--engine-arg {pair!r} is not KEY=VALUE")
        try:
            args[key] = json.loads(raw)
        except json.JSONDecodeError:
            args[key] = raw
    return args


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments of a command that starts one vLLM engine."""
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="serve this weights directory (e.g. a merged checkpoint; "
                        "default: the config's)")
    parser.add_argument("--dtype", help="serve the model in this dtype (e.g. float32; default: the config's dtype "
                        "key, else vLLM's auto)")
    parser.add_argument("--expect-dtype",
                        help="refuse an engine that resolved another model or KV-cache dtype (default: the dtype "
                        "served, --dtype or the config's, when either names one)")
    parser.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="an engine argument over the config's (repeatable)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truncation", choices=("none", "model_default"), default="model_default")


@dataclass(frozen=True)
class StartedEngine:
    """An engine `start_engine` built and checked, with what its commands sample by."""

    llm: Any
    tokenizer: Any
    lora_request: Any
    #: The served adapter's metadata (``lora_serving``), as the score files record it.
    adapter: Any
    #: :func:`engine_record`.
    record: dict[str, Any]
    weights: Any
    truncation: Mapping[str, float | int] | None


def start_engine(args: argparse.Namespace, model_cfg: Any, adapter: Path | None = None) -> StartedEngine:
    """Start the engine `add_engine_arguments` describe, the one start every vLLM command shares."""
    from ruleforge.judges.backend import canonical_serving_dtype

    llm, tokenizer, lora_request, metadata = build_vllm_engine(
        model_cfg, adapter, model_path=args.model_path, dtype=args.dtype, engine_args=_parse_engine_args(args.engine_arg),
    )
    record = engine_record(llm, model_cfg.get("attention_backend"))
    check_engine_dtype(record, args.expect_dtype or canonical_serving_dtype(args.dtype or model_cfg.get("dtype")))
    weights = args.model_path or model_cfg.model_path
    truncation = None
    if args.truncation == "model_default":
        from ruleforge.judges.models import model_default_truncation

        truncation = model_default_truncation(weights, model_cfg.get("default_truncation"))
    return StartedEngine(llm, tokenizer, lora_request, metadata, record, weights, truncation)


#: `compare`'s exit status when the gates passed but a requested rho range was not applied.
RHO_NOT_GATED_EXIT = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora_parity",
        description="Compare HF-PEFT with vLLM: does a served adapter compute what it trained?")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-adapter", help="write a seeded random adapter (CPU, no weights loaded)")
    make.add_argument("--model-path", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--targets", default=",".join(DEFAULT_TARGET_MODULES))
    make.add_argument("--rank", type=int, default=8)
    make.add_argument("--alpha", type=float, default=16.0)
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--b-std", type=float, default=2e-3)

    inspect = sub.add_parser("inspect", help="static adapter checks (CPU)")
    inspect.add_argument("--adapter", type=Path, required=True)
    inspect.add_argument("--model-path", type=Path, help="check module paths against this checkpoint's own class")

    vllm = sub.add_parser("vllm", help="freeze sequences (unless given) and score one engine's vLLM arms (GPU)")
    add_engine_arguments(vllm)
    serving = vllm.add_mutually_exclusive_group(required=True)
    serving.add_argument("--adapter", type=Path, help="lora mode: vllm_base and vllm_lora on one LoRA engine")
    serving.add_argument("--arm", choices=(BASE_ARM, MERGED_ARM), help="merged mode: this engine's one arm")
    vllm.add_argument("--prompts", type=Path, help="JSONL prompts to sample sequences from")
    vllm.add_argument("--prompt-field", help="use this string field as the system prompt, user turn empty")
    vllm.add_argument("--sequences", type=Path, required=True, help="read if it exists, else sampled and written")
    vllm.add_argument("--max-tokens", type=int, default=2048)
    vllm.add_argument("--out", type=Path, required=True)
    vllm.add_argument("--decode-sequences", type=Path,
                      help="the decode arm's sequences: scored if the file exists, else sampled from the served arm")
    vllm.add_argument("--decode-out", type=Path, help="the decode arm's vLLM scores")
    vllm.add_argument("--decode-reference", type=Path, help="the base decode reference's sequences (sampled)")
    vllm.add_argument("--decode-reference-out", type=Path, help="its vLLM scores, decode and prefill")
    vllm.add_argument("--decode-count", type=int, default=4, help="prompts (the first N) the decode arms sample")
    vllm.add_argument("--decode-batch", type=int, default=2, help="prompts per decode call: the decode batch size")
    vllm.add_argument("--decode-max-tokens", type=int, help="default: --max-tokens")
    _add_greedy_arguments(vllm)

    hf = sub.add_parser("hf", help="score the two HF arms, or a plain model alone without --adapter (GPU)")
    hf.add_argument("--model-config", type=Path, required=True)
    hf.add_argument("--model-path", type=Path, help="load this weights directory (e.g. a merged checkpoint; "
                    "default: the config's)")
    hf.add_argument("--adapter", type=Path, help="score hf_lora and hf_base; without it, the plain model as --arm")
    hf.add_argument("--arm", choices=PLAIN_HF_ARMS, default="hf_base",
                    help="the plain model's arm (without --adapter): hf_merged for a merged directory")
    hf.add_argument("--autocast-arm", action="store_true",
                    help=f"with --adapter: also score the PeftModel under {AUTOCAST_DTYPE} autocast as {HF_LORA_AC_ARM}")
    hf.add_argument("--sequences", type=Path, help="the sequences to score (with --out; optional for a greedy-only run)")
    hf.add_argument("--dtype", default=HF_REFERENCE_DTYPE,
                    help="default float32 (TF32 off), the reference compare expects; another is on the record")
    hf.add_argument("--device", default="cuda")
    hf.add_argument("--attn-implementation", default="sdpa")
    hf.add_argument("--out", type=Path, help="the HF score file (with --sequences)")
    hf.add_argument("--decode-sequences", type=Path, help="also score the decode arm's sequences (one model load)")
    hf.add_argument("--decode-out", type=Path)
    hf.add_argument("--prompt-field", help="the greedy prompts' field: the system prompt, user turn empty")
    _add_greedy_arguments(hf)

    gate = sub.add_parser("compare", help="gate HF against vLLM (CPU); exit 1 on failure")
    gate.add_argument("--preset", choices=sorted(PARITY_PRESETS),
                      help="the gates of a check (parity B, M0; _fallback: the bf16 fallback); an explicit flag wins "
                      "over the preset")
    gate.add_argument("--check", choices=("b", "m0"), help="which check this is, recorded in the report's gates")
    gate.add_argument("--sequences", type=Path, required=True)
    gate.add_argument("--hf", type=Path, action="append", required=True, help="an HF score file (repeatable)")
    gate.add_argument("--vllm", type=Path, action="append", required=True,
                      help="a vLLM score file (repeatable: merged mode has one per engine)")
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--segment", choices=SEGMENTS, default="all", help="completion: the decode arm")
    gate.add_argument("--decode-reference-sequences", type=Path, help="the base decode reference (with --segment completion)")
    gate.add_argument("--decode-reference", type=Path, help="its scores: the decode floor")
    gate.add_argument("--allow-imprecise-rho", action="store_true",
                      help="exit 0 when a requested rho range could not be gated, for its interval's width or an "
                      "effect under --min-effect (reported as rho_gated; the waiver is recorded in the report's "
                      f"gates); an ungated rho_hf still exits {RHO_NOT_GATED_EXIT}")
    defaults = ParityThresholds()
    flags = gate.add_argument_group("gates", "each defaults to the preset's value, else ParityThresholds'")
    flags.add_argument("--noise-factor", type=float, help=f"default {defaults.noise_factor}")
    flags.add_argument("--abs-tolerance", type=float, help=f"default {defaults.abs_tolerance}")
    flags.add_argument("--effect-ratio", type=float, nargs=2,
                       help="gate the unpaired effect ratio within [LOW, HIGH] (default: report only; "
                       "for hunting a LoRA-mode no-op)")
    flags.add_argument("--min-effect", type=float, help=f"default {defaults.min_effect}")
    flags.add_argument("--paired-delta", type=float, help=f"default {defaults.paired_delta}")
    flags.add_argument("--min-effect-noise-ratio", type=float, help="default: 2 / paired delta")
    weak = flags.add_mutually_exclusive_group()
    weak.add_argument("--allow-weak-effect", dest="require_effect", action="store_const", const=False,
                      help="an effect too weak for the paired gate is inconclusive, not a failure")
    weak.add_argument("--require-effect", dest="require_effect", action="store_const", const=True,
                      help="an effect too weak for the paired gate fails (the default without a preset)")
    flags.add_argument("--rho-range", type=float, nargs=2, help="gate rho within [LOW, HIGH] (default: report only)")
    flags.add_argument("--rho-hf-range", type=float, nargs=2,
                       help="gate rho_hf (the merge alone) within [LOW, HIGH], failing an interval wider than "
                       "--rho-precision too, and require hf_merged (default: report only; preset b: 0.99 1.01)")
    flags.add_argument("--rho-precision", type=float,
                       help="gate rho's point estimate only where its interval half-width is at most this (an "
                       "interval wholly outside the range fails, one wholly inside passes, at any width); where it "
                       "is wider and straddles a bound (or the HF effect is under --min-effect), exit "
                       f"{RHO_NOT_GATED_EXIT} unless --allow-imprecise-rho. rho_hf fails a wider interval")
    flags.add_argument("--rho-per-bucket", action="store_true", default=None)
    flags.add_argument("--rho-report-threshold", type=float,
                       help="report whether the overall rho meets this (the bf16 fallback's 0.90); gates nothing")
    flags.add_argument("--rho-resamples", type=int, help=f"default {defaults.rho_resamples}")
    flags.add_argument("--rho-seed", type=int, help=f"default {defaults.rho_seed}")
    flags.add_argument("--greedy-agreement", type=float, help=f"default {defaults.greedy_agreement}")
    flags.add_argument("--bucket-edges", help=f"default {','.join(str(e) for e in defaults.bucket_edges)}")
    flags.add_argument("--min-bucket-tokens", type=int, help=f"default {defaults.min_bucket_tokens}")
    flags.add_argument("--degenerate-run", type=int, help=f"default {defaults.degenerate_run}")
    flags.add_argument("--allow-hf-dtype", dest="require_hf_float32", action="store_const", const=False,
                       help="compare HF arms scored in another dtype than float32 (recorded in the report)")

    greedy = sub.add_parser("greedy-agreement",
                            help="the greedy full-set judgment agreement of two engines (CPU); exit 1 under the "
                                 "minimum")
    greedy.add_argument("--first", type=Path, required=True, help="greedy rows (e.g. hf_merged's)")
    greedy.add_argument("--second", type=Path, required=True, help="greedy rows over the same prompts (vllm_merged's)")
    greedy.add_argument("--rules", type=Path, required=True, help="the ruleset JSONL the prompts carry")
    greedy.add_argument("--prompt-style", help=f"default {GREEDY_VERDICT_GATE['prompt_style']}")
    greedy.add_argument("--forced-json-requirement", help=f"default {GREEDY_VERDICT_GATE['forced_json_requirement']}")
    greedy.add_argument("--strict-think", action=argparse.BooleanOptionalAction,
                        help=f"default {GREEDY_VERDICT_GATE['strict_think']}")
    greedy.add_argument("--min-agreement", type=float, help=f"default {GREEDY_VERDICT_GATE['min_agreement']}")
    greedy.add_argument("--report", type=Path, required=True)

    plumb = sub.add_parser("plumbing", help="M0's plumbing checks on a merged directory (CPU); exit 1 on failure")
    plumb.add_argument("--base", type=Path, required=True, help="the base weights directory")
    plumb.add_argument("--merged", type=Path, required=True, help="the merged directory")
    plumb.add_argument("--second-merge", type=Path, help="a second merge of the same adapter on the same node type")
    plumb.add_argument("--prompts", type=Path, help="JSONL prompts both tokenizers must render to the same ids")
    plumb.add_argument("--prompt-field", help="use this string field as the system prompt, user turn empty")
    plumb.add_argument("--conditions-root", type=Path,
                       help="a campaign root registering the base's and the merged directory's conditions")
    plumb.add_argument("--base-condition", help="the base's condition id in --conditions-root")
    plumb.add_argument("--merged-condition", help="the merged directory's condition id in --conditions-root")
    plumb.add_argument("--max-rounded-share", type=float, default=M0_MAX_ROUNDED_SHARE)
    plumb.add_argument("--may-differ", default=",".join(IDENTITY_MAY_DIFFER),
                       help="ConditionSpec fields the two conditions may differ in")
    plumb.add_argument("--report", type=Path, required=True)
    return parser


def _add_greedy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--greedy-prompts", type=Path, help="JSONL prompts to generate greedily from")
    parser.add_argument("--greedy-out", type=Path, help="the greedy rows (greedy-agreement compares two)")
    parser.add_argument("--greedy-max-tokens", type=int, default=4096)


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=1, default=str))


@contextlib.contextmanager
def _timed(seconds: dict[str, float], name: str):
    start = time.perf_counter()
    yield
    seconds[name] = round(time.perf_counter() - start, 3)


def _paired(args: argparse.Namespace, first: str, second: str) -> bool:
    a, b = getattr(args, first), getattr(args, second)
    if (a is None) != (b is None):
        raise ParityError(f"--{first.replace('_', '-')} and --{second.replace('_', '-')} go together")
    return a is not None


def _run_vllm(args: argparse.Namespace) -> int:
    model_cfg = load_model_config(args.model_config)
    decode = _paired(args, "decode_out", "decode_sequences")
    reference = _paired(args, "decode_reference_out", "decode_reference")
    lora = args.adapter is not None
    serves_base = lora or args.arm == BASE_ARM
    served_arm_name = LORA_ARM if lora else (MERGED_ARM if args.arm == MERGED_ARM else None)
    # the served arm's decode scores come from sampling it; every other arm scores those sequences through prefill
    samples_decode = decode and served_arm_name is not None
    if samples_decode and args.decode_sequences.exists():
        raise ParityError(
            f"{args.decode_sequences} exists, but {served_arm_name}'s decode scores come from sampling it: "
            "remove it to sample again"
        )
    if decode and not samples_decode and not args.decode_sequences.exists():
        raise ParityError(
            f"{args.decode_sequences} does not exist: the decode arm is sampled by the served arm's engine "
            f"({MERGED_ARM}), which runs first"
        )
    if reference and not serves_base:
        raise ParityError(f"the decode reference is sampled from {BASE_ARM}")
    if (samples_decode or reference) and args.prompts is None:
        raise ParityError("the decode arms sample from --prompts")
    started = start_engine(args, model_cfg, args.adapter)
    llm, tokenizer, engine = started.llm, started.tokenizer, started.record
    dtype = engine["engine_dtype"]
    servings = [(BASE_ARM, None), (LORA_ARM, started.lora_request)] if lora else [(args.arm, None)]
    sampling = {
        "seed": args.seed, "truncation": started.truncation,
        "gpt_oss_reasoning_level": model_cfg.get("gpt_oss_reasoning_level"),
    }
    seconds: dict[str, float] = {}
    if args.sequences.exists():
        sequences = read_sequences(args.sequences)
    else:
        if args.prompts is None:
            raise ParityError(f"{args.sequences} does not exist and no --prompts to sample it from")
        with _timed(seconds, "sample"):
            sequences = sample_sequences(
                llm, tokenizer, read_prompts(args.prompts, args.prompt_field), max_tokens=args.max_tokens, **sampling,
            )
        write_sequences(args.sequences, sequences)
    meta = {"adapter": started.adapter, "model_path": str(started.weights),
            **scored_adapter(args.adapter, started.weights), **engine,
            "servings": [a for a, _ in servings]}
    with _timed(seconds, "prefill"):
        prefill_scores = score_servings(llm, servings, sequences)
    write_scores(args.out, prefill_scores, {**meta, "arm": "prefill", "elapsed_seconds": seconds["prefill"]}, sequences)
    summary: dict[str, Any] = {
        "sequences": len(sequences), "tokens": sum(len(s.token_ids) for s in sequences), "out": str(args.out),
        "engine_dtype": dtype, "seconds": seconds,
    }
    decode_settings = {"max_tokens": args.decode_max_tokens or args.max_tokens, "batch_size": args.decode_batch, **sampling}
    prompts = read_prompts(args.prompts, args.prompt_field)[: args.decode_count] if args.prompts else []
    if decode:
        others = [(arm, request) for arm, request in servings if arm != served_arm_name]
        with _timed(seconds, "decode"):
            if samples_decode:
                request = dict(servings)[served_arm_name]
                decoded, served_scores = sample_decode(llm, tokenizer, prompts, lora_request=request, **decode_settings)
                scores = _named(served_scores, served_arm_name).merge(score_servings(llm, others, decoded))
            else:
                decoded = read_sequences(args.decode_sequences)
                scores = score_servings(llm, others, decoded)
        if samples_decode:
            write_sequences(args.decode_sequences, decoded)
        write_scores(args.decode_out, scores, {**meta, "arm": "decode", "decode_batch": args.decode_batch,
                                               "sampled": samples_decode, "elapsed_seconds": seconds["decode"]}, decoded)
        summary["decode"] = {
            "sequences": len(decoded), "completion_tokens": sum(len(s.completion_ids) for s in decoded),
            "sampled": samples_decode, "out": str(args.decode_out),
        }
    if reference:
        with _timed(seconds, "decode_reference"):
            sampled, decode_scores = sample_decode(llm, tokenizer, prompts, prefix="ref", **decode_settings)
            scores = _named(decode_scores, BASE_DECODE_ARM).merge(score_servings(llm, [(BASE_ARM, None)], sampled))
        write_sequences(args.decode_reference, sampled)
        write_scores(args.decode_reference_out, scores, {
            **meta, "arm": "decode_reference", "decode_batch": args.decode_batch,
            "elapsed_seconds": seconds["decode_reference"],
        }, sampled)
        summary["decode_reference"] = {"sequences": len(sampled), "out": str(args.decode_reference_out)}
    if _paired(args, "greedy_prompts", "greedy_out"):
        ids = prompt_token_ids(tokenizer, read_prompts(args.greedy_prompts, args.prompt_field),
                               model_cfg.get("gpt_oss_reasoning_level"))
        adapter = {"lora_request": started.lora_request} if lora else {}
        with _timed(seconds, "greedy"):
            generated = vllm_greedy(llm, ids, max_tokens=args.greedy_max_tokens, **adapter)
        write_jsonl_records(args.greedy_out, greedy_records(ids, generated, tokenizer, eos_token_ids(Path(str(started.weights)))))
        summary["greedy"] = {"prompts": len(ids), "out": str(args.greedy_out), "arm": served_arm_name or args.arm}
    _print(summary)
    return 0


def _run_hf(args: argparse.Namespace) -> int:
    model_cfg = load_model_config(args.model_config)
    score = _paired(args, "sequences", "out")
    decode = _paired(args, "decode_out", "decode_sequences")
    greedy = _paired(args, "greedy_prompts", "greedy_out")
    if not (score or greedy):
        raise ParityError("nothing to run: --sequences/--out (scoring), --greedy-prompts/--greedy-out, or both")
    if decode and not score:
        raise ParityError("the decode arm is scored beside the prefill one: --decode-* need --sequences/--out")
    if args.adapter is not None and args.arm != "hf_base":
        raise ParityError("--arm is the plain model's (without --adapter)")
    if args.autocast_arm and args.adapter is None:
        raise ParityError(f"{HF_LORA_AC_ARM} is the adapter's arm under autocast: --autocast-arm needs --adapter")
    sets = [(read_sequences(args.sequences), args.out)] if score else []
    if decode:
        sets.append((read_sequences(args.decode_sequences), args.decode_out))
    ids = [seq.seq_id for sequences, _ in sets for seq in sequences]
    if len(set(ids)) != len(ids):
        raise ParityError("the prefill and decode sequence files share a sequence id")
    numerics = exact_float32() if args.dtype == "float32" else {}
    weights = Path(str(args.model_path or model_cfg.model_path))
    model = load_hf_model(weights, dtype=args.dtype, device=args.device, attn_implementation=args.attn_implementation)
    # one PeftModel for scoring and greedy generation: wrapping twice would inject the adapter twice
    wrapped = None if args.adapter is None else wrap_adapter(model, args.adapter)
    summary: dict[str, Any] = {"sequences": len(ids), "out": [str(out) for _, out in sets]}
    if score:
        scores = score_hf(
            model, args.adapter, [seq for sequences, _ in sets for seq in sequences], device=args.device,
            arm=args.arm, autocast_arm=args.autocast_arm, wrapped=wrapped,
        )
        adapter = None if args.adapter is None else str(args.adapter)
        meta = {"adapter": adapter, **scored_adapter(args.adapter, weights), "model_path": str(weights),
                "dtype": args.dtype, "autocast": False, **numerics}
        if args.autocast_arm:
            meta["autocast_arms"] = {HF_LORA_AC_ARM: AUTOCAST_DTYPE}
        for sequences, out in sets:
            write_scores(out, scores.only(seq.seq_id for seq in sequences), meta, sequences)
    if greedy:
        tokenizer = _load_tokenizer(weights)
        prompt_ids = prompt_token_ids(tokenizer, read_prompts(args.greedy_prompts, args.prompt_field),
                                      model_cfg.get("gpt_oss_reasoning_level"))
        eos_ids = eos_token_ids(weights)
        generated = hf_greedy(wrapped if wrapped is not None else model, prompt_ids, max_tokens=args.greedy_max_tokens,
                              eos_ids=eos_ids, device=args.device)
        write_jsonl_records(args.greedy_out, greedy_records(prompt_ids, generated, tokenizer, eos_ids))
        summary["greedy"] = {"prompts": len(prompt_ids), "out": str(args.greedy_out), "dtype": args.dtype,
                             "arm": "hf_lora" if wrapped is not None else args.arm}
    _print(summary)
    return 0


def _read_all(paths: Sequence[Path]) -> Scores:
    scores = Scores()
    for path in paths:
        scores = scores.merge(read_scores(path))
    return scores


#: The `all` bucket's fields `compare` prints beside the verdict.
_EFFECT_SUMMARY = ("noise", "decode_noise", "serving_limit", "effect_hf", "effect_noise_ratio", "effect_window")


#: `compare`'s gate flags: `ParityThresholds` fields.
_GATE_FLAGS = {
    "noise_factor": None, "abs_tolerance": None, "effect_ratio": tuple, "min_effect": None, "paired_delta": None,
    "min_effect_noise_ratio": None, "require_effect": None, "rho_range": tuple, "rho_hf_range": tuple,
    "rho_precision": None, "rho_per_bucket": None, "rho_report_threshold": None, "rho_resamples": None,
    "rho_seed": None,
    "greedy_agreement": None, "bucket_edges": lambda text: tuple(int(e) for e in text.split(",") if e),
    "min_bucket_tokens": None, "degenerate_run": None, "require_hf_float32": None,
}


def compare_thresholds(args: argparse.Namespace) -> tuple[ParityThresholds, dict[str, Any]]:
    """Return the gates `compare` applies, the preset's under the explicit flags, and what its report records of them."""
    explicit = {name: (convert or (lambda v: v))(getattr(args, name))
                for name, convert in _GATE_FLAGS.items() if getattr(args, name) is not None}
    thresholds = ParityThresholds(**{**PARITY_PRESETS.get(args.preset, {}), **explicit})
    gates = {"preset": args.preset, **({} if args.check is None else {"check": args.check}),
             "thresholds": thresholds_to_json(thresholds)}
    return thresholds, gates


def _run_compare(args: argparse.Namespace) -> int:
    thresholds, gates = compare_thresholds(args)
    floor = None
    if _paired(args, "decode_reference", "decode_reference_sequences"):
        floor = decode_noise_floor(read_scores(args.decode_reference), read_sequences(args.decode_reference_sequences), thresholds)
    report = compare(_read_all(args.hf).merge(_read_all(args.vllm)), read_sequences(args.sequences), thresholds,
                     segment=args.segment, decode_floor=floor)
    if args.allow_imprecise_rho:
        gates["allow_imprecise_rho"] = True
    args.report.write_text(json.dumps({**report.to_json(), "gates": gates}, indent=1) + "\n", encoding="utf-8")
    overall = report.buckets["all"]
    _print({
        "passed": report.passed, "mode": report.mode, "failures": report.failures,
        "inconclusive": report.inconclusive, "notes": report.notes, "inputs": report.inputs,
        "rho": overall.rho, "rho_ci": overall.rho_ci, "rho_gated": report.rho_gated,
        "rho_hf": overall.rho_hf, "rho_hf_gated": report.rho_hf_gated, "serving_reference": report.serving_reference,
        "autocast_gap": [overall.autocast_gap, overall.autocast_gap_p99],
        **({"rho_threshold_met": report.rho_threshold_met} if report.rho_threshold_met else {}),
        "effect": {name: getattr(overall, name) for name in _EFFECT_SUMMARY},
        "report": str(args.report),
    })
    if not report.passed:
        return 1
    # --allow-imprecise-rho waives rho only; rho_hf goes ungated only under a too-weak effect, never a pass
    rho_ungated = not all(report.rho_gated.values()) and not args.allow_imprecise_rho
    if rho_ungated or not all(report.rho_hf_gated.values()):
        return RHO_NOT_GATED_EXIT
    return 0


def _run_greedy_agreement(args: argparse.Namespace) -> int:
    gate = {**GREEDY_VERDICT_GATE,
            **{key: getattr(args, key) for key in GREEDY_VERDICT_GATE if getattr(args, key) is not None}}
    schema = verdict_schema(
        args.rules, prompt_style=gate["prompt_style"], forced_json_requirement=gate["forced_json_requirement"],
        strict_think=gate["strict_think"],
    )
    result = greedy_agreement(load_jsonl_records(args.first), load_jsonl_records(args.second), schema,
                              minimum=gate["min_agreement"])
    result.update(inputs={"first": str(args.first), "second": str(args.second), "rules": str(args.rules)},
                  greedy_verdict=gate)
    args.report.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    _print({key: result[key] for key in ("passed", "agreement", "minimum", "prompts", "valid_agreement")})
    return 0 if result["passed"] else 1


def _run_plumbing(args: argparse.Namespace) -> int:
    conditions = None
    wanted = (args.conditions_root, args.base_condition, args.merged_condition)
    if any(value is not None for value in wanted):
        if None in wanted:
            raise ParityError("--conditions-root, --base-condition and --merged-condition go together")
        from ruleforge.evaluate.campaign import load_conditions

        registered = load_conditions(args.conditions_root)
        missing = [cid for cid in wanted[1:] if cid not in registered]
        if missing:
            raise ParityError(f"{args.conditions_root} registers no condition {missing}")
        conditions = (registered[args.base_condition], registered[args.merged_condition])
    result = plumbing(
        args.base, args.merged, second_merge=args.second_merge,
        prompts=None if args.prompts is None else read_prompts(args.prompts, args.prompt_field),
        conditions=conditions, max_rounded_share=args.max_rounded_share,
        may_differ=tuple(name for name in args.may_differ.split(",") if name),
    )
    args.report.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    _print({"passed": result["passed"], "checks": {check["name"]: check["passed"] for check in result["checks"]}})
    return 0 if result["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "make-adapter":
        report = make_random_adapter(
            args.model_path, args.out, target_modules=[t for t in args.targets.split(",") if t],
            rank=args.rank, alpha=args.alpha, seed=args.seed, b_std=args.b_std,
        )
        _print(asdict(report))
        return 0 if report.ok else 1
    if args.command == "inspect":
        names = checkpoint_module_names(args.model_path) if args.model_path else None
        report = inspect_adapter(args.adapter, module_names=names)
        _print(asdict(report))
        return 0 if report.ok else 1
    if args.command == "vllm":
        return _run_vllm(args)
    if args.command == "hf":
        return _run_hf(args)
    if args.command == "greedy-agreement":
        return _run_greedy_agreement(args)
    if args.command == "plumbing":
        return _run_plumbing(args)
    return _run_compare(args)
