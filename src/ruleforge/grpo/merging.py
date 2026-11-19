from __future__ import annotations

import argparse
import contextlib
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ruleforge.grpo.lora_parity import (
    _PEFT_PREFIX,
    ADAPTER_CONFIG_FNAME,
    ADAPTER_WEIGHTS_FNAME,
    DEFAULT_TARGET_MODULES,
    _meta_model,
    _package_version,
    checkpoint_class,
    inspect_adapter,
)
from ruleforge.judges.weights import read_safetensors_header
from ruleforge.grpo.lora_targets import in_scope
from ruleforge.judges.weights import (
    MANIFEST_FNAME,
    SERVING_FILES,
    TOKENIZER_FILES,
    WEIGHTS_DIGEST_VERSION,
    MergeError,
    _tensor_index,
    weights_digest,
)

MANIFEST_SCHEMA_VERSION = 1
TRAINER_STATE_FNAME = "trainer_state.json"
CPUINFO_PATH = Path("/proc/cpuinfo")
#: The shard size the merged directory is saved in; the digest does not depend on it.
MAX_SHARD_SIZE = "5GB"
#: The staging directory a merge writes before its rename.
PARTIAL_SUFFIX = ".partial"
_ADAPTER_NAME = "default"
#: LoRA config keys under which a merge is not W + (B.A).alpha/r, each with its plain value.
_PLAIN_LORA = {
    "peft_type": "LORA", "use_dora": False, "use_rslora": False, "use_qalora": False, "fan_in_fan_out": False,
    "bias": "none", "lora_bias": False,
}
#: LoRA config keys that must be empty.
_EMPTY_LORA = (
    "rank_pattern", "alpha_pattern", "modules_to_save", "trainable_token_indices", "layer_replication",
    "target_parameters", "alora_invocation_tokens",
)
#: The `init_lora_weights` values the merge takes: PEFT's plain inits, which set only A and B.
_PLAIN_INITS = (True, False, "gaussian")
#: oneDNN's fp32 matmul precision during the merge.
MATMUL_FP32_PRECISION = "ieee"
BF16_ROUNDING = "round to nearest even (torch Tensor.to(torch.bfloat16))"


@dataclass(frozen=True)
class MergeExpectation:
    """What the adapter must be: its rank and alpha, the base it names (by the last path component) and the module leaves
    it adapts, below `module_scope` when given."""

    rank: int
    alpha: float
    base_model: str
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    module_scope: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise MergeError(f"the expected rank must be positive, got {self.rank}")
        if not self.target_modules:
            raise MergeError("the expectation names no target module")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "target_modules", tuple(self.target_modules))


@dataclass(frozen=True)
class AdapterCheck:
    """An adapter that passed `check_adapter`: the modules it adapts, `{path: (out, in)}`."""

    modules: dict[str, tuple[int, int]]
    tensors: int
    rank: int
    alpha: float


def _name_of(value: Any) -> str:
    return Path(str(value).rstrip("/")).name


def expected_modules(
    base_dir: Path, target_modules: Sequence[str], module_scope: str | None = None
) -> dict[str, tuple[int, int]]:
    """Return `{module path: (out_features, in_features)}` of every module of the base's own class a target names."""
    modules = {}
    for name, module in _meta_model(Path(base_dir)).named_modules():
        if not in_scope(name, target_modules, module_scope):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() != 2:
            raise MergeError(f"target {name!r} of {base_dir} is not a linear projection (no 2-D weight)")
        modules[name] = (int(weight.shape[0]), int(weight.shape[1]))
    if not modules:
        raise MergeError(f"the targets {list(target_modules)} match no module of {base_dir}")
    return modules


def _factor_shape(header: Mapping[str, Mapping[str, Any]], path: str, factor: str) -> list[int] | None:
    for key in (f"{_PEFT_PREFIX}{path}.lora_{factor}.weight", f"{path}.lora_{factor}.weight"):
        if key in header:
            return list(header[key]["shape"])
    return None


def check_adapter(adapter_dir: Path, base_dir: Path, expect: MergeExpectation) -> AdapterCheck:
    """Check an adapter against its expectation and the base's module tree without reading weights; every problem is
    listed in one `MergeError`."""
    adapter_dir, base_dir = Path(adapter_dir), Path(base_dir)
    absent = [name for name in (ADAPTER_CONFIG_FNAME, ADAPTER_WEIGHTS_FNAME) if not (adapter_dir / name).is_file()]
    if absent:
        raise MergeError(f"{adapter_dir} has no {' or '.join(absent)}")
    config = json.loads((adapter_dir / ADAPTER_CONFIG_FNAME).read_text(encoding="utf-8"))
    header = read_safetensors_header(adapter_dir / ADAPTER_WEIGHTS_FNAME)
    report = inspect_adapter(adapter_dir)
    expected = expected_modules(base_dir, expect.target_modules, expect.module_scope)
    problems = list(report.problems)
    if report.extra_keys:
        problems.append(
            f"{len(report.extra_keys)} tensor(s) are not LoRA factors (e.g. {report.extra_keys[0]}): "
            "the merge would drop them"
        )
    other = sorted({meta["dtype"] for meta in header.values()} - {"F32"})
    if other:
        problems.append(f"tensors in {other}, not F32: the merge starts from the fp32 factors training kept")
    adapted = set(report.module_paths)
    missing, unexpected = sorted(set(expected) - adapted), sorted(adapted - set(expected))
    if missing:
        problems.append(f"{len(missing)} expected module(s) are not adapted (e.g. {missing[0]})")
    if unexpected:
        problems.append(
            f"{len(unexpected)} adapted module(s) are not among the expected targets "
            f"{list(expect.target_modules)} (e.g. {unexpected[0]})"
        )
    misshapen = []
    for path in sorted(adapted & set(expected)):
        out_features, in_features = expected[path]
        shapes = {factor: _factor_shape(header, path, factor) for factor in ("A", "B")}
        if None not in shapes.values() and shapes != {"A": [expect.rank, in_features], "B": [out_features, expect.rank]}:
            misshapen.append(
                f"{path}: lora_A {shapes['A']} and lora_B {shapes['B']}, not [{expect.rank}, {in_features}] and "
                f"[{out_features}, {expect.rank}]"
            )
    if misshapen:
        problems.append(
            f"{len(misshapen)} module(s) have factors of another shape (another rank, or an adapter for "
            f"another base), e.g. {misshapen[0]}"
        )
    if config.get("r") != expect.rank:
        problems.append(f"adapter_config.json r is {config.get('r')!r}, not the expected {expect.rank}")
    if config.get("lora_alpha") != expect.alpha:
        problems.append(f"adapter_config.json lora_alpha is {config.get('lora_alpha')!r}, not the expected {expect.alpha}")
    for key, plain in _PLAIN_LORA.items():
        if config.get(key, plain) != plain:
            problems.append(f"adapter_config.json {key} is {config[key]!r}: the merge would not be W + (B.A).alpha/r")
    for key in _EMPTY_LORA:
        if config.get(key):
            problems.append(f"adapter_config.json {key} is set ({config[key]!r}): the merge would not be W + (B.A).alpha/r")
    init = config.get("init_lora_weights", True)
    if not isinstance(init, (bool, str)) or init not in _PLAIN_INITS:
        problems.append(
            f"adapter_config.json init_lora_weights is {init!r}, not one of {list(_PLAIN_INITS)}: the merge takes "
            "only PEFT's plain inits (others rewrite W at load or cannot load: PiSSA, OLoRA, LoftQ, CorDA; were "
            "trained against a W they rewrote: LoRA-GA; or are not what GRPO training uses), so that it is "
            "W + (B.A).alpha/r of this base"
        )
    named = config.get("base_model_name_or_path")
    if named is None or _name_of(named) != expect.base_model:
        problems.append(f"the adapter names base {named!r}, not {expect.base_model!r}")
    if _name_of(base_dir) != expect.base_model:
        problems.append(f"the base directory {base_dir} is not {expect.base_model!r}")
    if problems:
        raise MergeError(f"{adapter_dir} does not match its expectation: " + "; ".join(problems))
    return AdapterCheck(modules=expected, tensors=len(header), rank=expect.rank, alpha=expect.alpha)


class RetentionTally:
    """Telemetry of what a bf16 cast of the fp32 merge keeps of each module's delta: retention
    `<bf16(W + dW) - W, dW> / |dW|^2` and the share of non-zero elements that round away, overall and per projection."""

    #: Per projection: modules, sum <kept, dW>, sum dW^2, elements rounded away, non-zero elements.
    FIELDS = ("modules", "kept", "energy", "rounded", "nonzero")

    def __init__(self) -> None:
        self.sums: dict[str, dict[str, float]] = {}

    def add(self, projection: str, weight: "Any", delta: "Any") -> None:
        import torch

        merged = weight + delta
        cast = merged.to(torch.bfloat16)
        kept = cast.float() - weight
        nonzero = delta != 0
        rounded = (cast == weight.to(torch.bfloat16)) & nonzero
        sums = self.sums.setdefault(projection, dict.fromkeys(self.FIELDS, 0))
        sums["modules"] += 1
        sums["kept"] += float(torch.sum(kept * delta, dtype=torch.float64))
        sums["energy"] += float(torch.sum(delta * delta, dtype=torch.float64))
        sums["rounded"] += int(rounded.sum())
        sums["nonzero"] += int(nonzero.sum())

    @staticmethod
    def _summary(modules: float, kept: float, energy: float, rounded: float, nonzero: float) -> dict[str, Any]:
        return {
            "modules": int(modules),
            "retention": kept / energy if energy > 0 else None,
            "rounded_away": rounded / nonzero if nonzero else None,
            "nonzero_elements": int(nonzero),
        }

    def summary(self) -> dict[str, Any]:
        totals = {field: sum(sums[field] for sums in self.sums.values()) for field in self.FIELDS}
        return {
            "overall": self._summary(**totals),
            "per_projection": {name: self._summary(**sums) for name, sums in sorted(self.sums.items())},
        }


def cpu_model() -> str | None:
    """Return the CPU's model name, or None."""
    try:
        text = CPUINFO_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def training_step(adapter_dir: Path) -> int | None:
    """Return the optimizer step a trainer checkpoint was saved at, if `trainer_state.json` says."""
    path = Path(adapter_dir) / TRAINER_STATE_FNAME
    if not path.is_file():
        return None
    step = json.loads(path.read_text(encoding="utf-8")).get("global_step")
    return None if step is None else int(step)


#: Config fields a merge may change.
_CONFIG_MAY_DIFFER = ("_name_or_path", "dtype", "transformers_version")


def _sub_configs_without_may_differ(fields: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for name, value in fields.items():
        if isinstance(value, Mapping) and "model_type" in value:
            value = {key: item for key, item in value.items() if key not in _CONFIG_MAY_DIFFER}
        cleaned[name] = value
    return cleaned


def config_drift(base_dir: Path, saved_dir: Path) -> dict[str, tuple[Any, Any]]:
    """Return `{field: (saved, base)}` for every config field that differs, compared as parsed."""
    from transformers import AutoConfig

    saved, base = (
        _sub_configs_without_may_differ(AutoConfig.from_pretrained(str(directory)).to_dict())
        for directory in (saved_dir, base_dir)
    )
    fields = sorted((set(saved) | set(base)) - set(_CONFIG_MAY_DIFFER))
    return {field: (saved.get(field), base.get(field)) for field in fields if saved.get(field) != base.get(field)}


def _copy_serving_files(base_dir: Path, out_dir: Path) -> dict[str, str]:
    from ruleforge.core.digests import sha256_file

    copied = {}
    for name in SERVING_FILES:
        source = base_dir / name
        if source.is_file():
            shutil.copyfile(source, out_dir / name)
            copied[name] = sha256_file(out_dir / name)
    if "generation_config.json" not in copied:
        (out_dir / "generation_config.json").unlink(missing_ok=True)
    return copied


@contextlib.contextmanager
def _cpu_numerics(count: int):
    import torch

    matmul = torch.backends.mkldnn.matmul
    previous = torch.get_num_threads(), matmul.fp32_precision
    torch.set_num_threads(count)
    matmul.fp32_precision = MATMUL_FP32_PRECISION
    try:
        yield torch.get_num_threads(), matmul.fp32_precision
    finally:
        torch.set_num_threads(previous[0])
        matmul.fp32_precision = previous[1]


def _rewritten_base_weights(layers: Mapping[str, Any], base_dir: Path) -> list[str]:
    import torch
    from safetensors import safe_open

    index = _tensor_index(base_dir)
    rewritten = []
    with contextlib.ExitStack() as stack:
        handles: dict[Path, Any] = {}
        for path, layer in sorted(layers.items()):
            name = f"{path}.weight"
            if name not in index:
                rewritten.append(f"{path} (no {name} in the base)")
                continue
            file = index[name][0]
            if file not in handles:
                handles[file] = stack.enter_context(safe_open(str(file), framework="pt"))
            if not torch.equal(layer.get_base_layer().weight, handles[file].get_tensor(name).to(torch.float32)):
                rewritten.append(path)
    return rewritten


def _variant_problems(layers: Mapping[str, Any], check: AdapterCheck) -> list[str]:
    scaling = check.alpha / check.rank
    problems = []
    for path, layer in sorted(layers.items()):
        variant = getattr(layer, "lora_variant", {}).get(_ADAPTER_NAME)
        if variant is not None:
            problems.append(f"{path}: variant {type(variant).__name__}")
        elif layer.scaling.get(_ADAPTER_NAME) != scaling:
            problems.append(f"{path}: scaling {layer.scaling.get(_ADAPTER_NAME)}, not alpha/r = {scaling}")
        elif getattr(layer, "fan_in_fan_out", False) or getattr(layer, "lora_bias", {}).get(_ADAPTER_NAME):
            problems.append(f"{path}: transposed factors or a LoRA bias")
    return problems


def _merged_model(base_dir: Path, adapter_dir: Path, check: AdapterCheck, tally: RetentionTally):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file

    _, model_class = checkpoint_class(base_dir)
    model = model_class.from_pretrained(str(base_dir), dtype=torch.float32).eval()
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir)).eval()
    layers = {
        name: module for name, module in peft_model.base_model.model.named_modules()
        if isinstance(module, BaseTunerLayer)
    }
    if set(layers) != set(check.modules):
        raise MergeError(
            f"PEFT wrapped {len(layers)} module(s), not the {len(check.modules)} checked: "
            f"{sorted(set(layers) ^ set(check.modules))[:3]}"
        )
    # PEFT leaves an adapter key it cannot place at B = 0: merged, it would silently be the base there
    stored = load_file(str(adapter_dir / ADAPTER_WEIGHTS_FNAME))
    loaded = get_peft_model_state_dict(peft_model)
    unloaded = sorted(k for k in stored if k not in loaded or not torch.equal(loaded[k], stored[k]))
    if unloaded or set(loaded) != set(stored):
        raise MergeError(
            f"PEFT did not load the adapter's factors as stored: {unloaded[:3] or sorted(set(loaded) - set(stored))[:3]}"
        )
    problems = _variant_problems(layers, check)
    if problems:
        raise MergeError(f"PEFT built layers whose merge is not W + (B.A).alpha/r: {problems[:3]}")
    rewritten = _rewritten_base_weights(layers, base_dir)
    if rewritten:
        raise MergeError(
            f"{len(rewritten)} wrapped layer(s) no longer hold the base's weight after PEFT loaded the adapter "
            f"(an init that rewrites W?): {rewritten[:3]}"
        )
    with torch.no_grad():
        for path, layer in sorted(layers.items()):
            tally.add(path.rpartition(".")[2], layer.get_base_layer().weight, layer.get_delta_weight(_ADAPTER_NAME))
    merged = peft_model.merge_and_unload(safe_merge=True)
    left = [name for name, module in merged.named_modules() if isinstance(module, BaseTunerLayer)]
    left += [name for name, _ in merged.named_parameters() if "lora_" in name]
    if left:
        raise MergeError(f"LoRA modules are left after the merge: {left[:3]}")
    nonfinite = [
        name for name, tensor in merged.state_dict().items()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ]
    if nonfinite:
        raise MergeError(f"non-finite values after the merge in {nonfinite[:3]}")
    return merged


def merge_adapter(
    base_dir: Path | str,
    adapter_dir: Path | str,
    out_dir: Path | str,
    *,
    expect: MergeExpectation,
    threads: int,
    cast_bf16: bool = False,
    expect_base_sha256: str | None = None,
    training_run: str | None = None,
    training_step_override: int | None = None,
) -> dict[str, Any]:
    """Merge `adapter_dir` into `base_dir`, save it at `out_dir` and return the manifest; `expect_base_sha256` pins the
    base's weights."""
    import torch
    from transformers import GenerationConfig

    from ruleforge.judges.weights import adapter_identity
    from ruleforge.core.digests import sha256_file
    from ruleforge.core.provenance import code_state

    if threads < 1:
        raise MergeError(f"threads must be positive, got {threads}")
    base_dir, adapter_dir, out_dir = Path(base_dir), Path(adapter_dir), Path(out_dir)
    partial = out_dir.with_name(out_dir.name + PARTIAL_SUFFIX)
    for path in (out_dir, partial):
        if path.exists():
            raise MergeError(f"{path} exists: a merge never overwrites (remove it to merge again)")
    if not any((base_dir / name).is_file() for name in TOKENIZER_FILES):
        raise MergeError(f"{base_dir} has none of {list(TOKENIZER_FILES)}: the merged directory could not be served")
    check = check_adapter(adapter_dir, base_dir, expect)
    identity = adapter_identity(adapter_dir / ADAPTER_WEIGHTS_FNAME)
    base_sha256 = weights_digest(base_dir)
    if expect_base_sha256 is not None and base_sha256 != expect_base_sha256:
        raise MergeError(f"{base_dir} weights digest {base_sha256} is not the expected {expect_base_sha256}")
    base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    saved_dtype = "bfloat16" if cast_bf16 else "float32"
    tally = RetentionTally()
    started = time.perf_counter()
    with _cpu_numerics(threads) as (used_threads, matmul_precision):
        merged = _merged_model(base_dir, adapter_dir, check, tally)
        if cast_bf16:
            merged = merged.to(torch.bfloat16)
        parameters = sum(parameter.numel() for parameter in merged.parameters())
        # the base's generation_config.json is copied verbatim after the save; a default stands in for the one save_pretrained validates strictly
        merged.generation_config = GenerationConfig()
        merged.save_pretrained(str(partial), max_shard_size=MAX_SHARD_SIZE)
    del merged
    copied = _copy_serving_files(base_dir, partial)
    saved_config = json.loads((partial / "config.json").read_text(encoding="utf-8"))
    changed = config_drift(base_dir, partial)
    if changed or saved_config.get("dtype") != saved_dtype:
        raise MergeError(
            f"the saved config.json is not the base's config in {saved_dtype}: {changed or saved_config.get('dtype')}"
        )
    expectation = {**asdict(expect), "target_modules": list(expect.target_modules)}
    if expect.module_scope is None:
        del expectation["module_scope"]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "lora_merge",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights_digest": WEIGHTS_DIGEST_VERSION,
        "base": {
            "path": str(base_dir.resolve()),
            "weights_sha256": base_sha256,
            "architectures": base_config.get("architectures"),
            "model_type": base_config.get("model_type"),
        },
        "adapter": {
            "path": identity.path,
            "label": identity.label,
            "sha256": identity.sha256,
            "config_sha256": identity.config_sha256,
            "rank": check.rank,
            "alpha": check.alpha,
            "base_model": identity.base_model,
            "modules": len(check.modules),
            "tensors": check.tensors,
        },
        "training": {
            "run": training_run,
            "step": training_step(adapter_dir) if training_step_override is None else int(training_step_override),
        },
        "expectation": expectation,
        "merge": {
            "method": "peft merge_and_unload(safe_merge=True)",
            "device": "cpu",
            "merge_dtype": "float32",
            "saved_dtype": saved_dtype,
            "rounding": BF16_ROUNDING if cast_bf16 else None,
            "threads": used_threads,
            "fp32_matmul_precision": matmul_precision,
            "cpu_model": cpu_model(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "versions": {name: _package_version(name) for name in ("torch", "transformers", "peft", "safetensors")},
            "code": code_state(),
            "seconds": round(time.perf_counter() - started, 3),
        },
        "result": {
            "weights_sha256": weights_digest(partial),
            "config_sha256": sha256_file(partial / "config.json"),
            "tensors": len(_tensor_index(partial)),
            "parameters": parameters,
            "modules_merged": len(check.modules),
            "copied_files": copied,
            "bf16_cast": tally.summary(),
        },
    }
    (partial / MANIFEST_FNAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    partial.rename(out_dir)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge_checkpoint",
        description="Merge a LoRA adapter into its base in fp32, with a manifest of every digest.")
    parser.add_argument("--base", type=Path, required=True, help="the base weights directory")
    parser.add_argument("--adapter", type=Path, required=True, help="the adapter directory (adapter_config.json + weights)")
    parser.add_argument("--out", type=Path, required=True, help="the merged directory (must not exist)")
    parser.add_argument("--threads", type=int, required=True, help="torch's CPU thread count (recorded)")
    parser.add_argument("--expect-rank", type=int, required=True)
    parser.add_argument("--expect-alpha", type=float, required=True)
    parser.add_argument("--expect-base-model", required=True,
                        help="the base's name, as the adapter and the base directory must carry it (e.g. Qwen3-8B)")
    parser.add_argument("--expect-targets", default=",".join(DEFAULT_TARGET_MODULES),
                        help="the module leaves the adapter must adapt, every one of them the base has")
    parser.add_argument("--expect-scope",
                        help="only the targets below this module path (Gemma 4: model.language_model.)")
    parser.add_argument("--expect-base-sha256", help="refuse a base whose weights digest differs")
    parser.add_argument("--cast-bf16", action="store_true",
                        help="save in bf16 (the evaluated merge is fp32)")
    parser.add_argument("--training-run", help="the training run the adapter comes from (provenance)")
    parser.add_argument("--training-step", type=int, help="its step (default: the checkpoint's trainer_state.json)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    expect = MergeExpectation(
        rank=args.expect_rank, alpha=args.expect_alpha, base_model=args.expect_base_model,
        target_modules=tuple(t for t in args.expect_targets.split(",") if t), module_scope=args.expect_scope,
    )
    manifest = merge_adapter(
        args.base, args.adapter, args.out, expect=expect, threads=args.threads, cast_bf16=args.cast_bf16,
        expect_base_sha256=args.expect_base_sha256, training_run=args.training_run,
        training_step_override=args.training_step,
    )
    summary: Mapping[str, Any] = {
        "out": str(args.out),
        "weights_sha256": manifest["result"]["weights_sha256"],
        "saved_dtype": manifest["merge"]["saved_dtype"],
        "modules_merged": manifest["result"]["modules_merged"],
        "bf16_cast": manifest["result"]["bf16_cast"]["overall"],
        "seconds": manifest["merge"]["seconds"],
    }
    print(json.dumps(summary, indent=1))
    return 0
