from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from ruleforge.core.errors import TracebackError
from ruleforge.grpo.lora_parity import _PEFT_PREFIX
from ruleforge.grpo.merging import RetentionTally

#: HF projection leaf -> (the vLLM packed leaf, its shard index) at tensor-parallel size 1.
PACKED_SHARDS: dict[str, tuple[str, int]] = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}


#: HF projection leaves vLLM shards by input column under tensor parallelism.
ROW_PARALLEL = ("o_proj", "down_proj")
#: HF module-path prefix -> the prefix the in-process engine names it under, where they differ.
ENGINE_PREFIXES = {"model.language_model.": "language_model.model."}
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class SyncProbeError(RuntimeError, TracebackError):
    """The rollout engine does not hold the weights the trainer pushed."""


def _text_config(config: Any) -> Any:
    get = getattr(config, "get_text_config", None)
    return config if get is None else get()


def attention_heads(config: Any, layer: Optional[int] = None) -> dict[str, int]:
    """Return the heads of each packed attention projection of `layer`."""
    text = _text_config(config)
    kv_heads = getattr(_layer_shape(text, layer), "num_key_value_heads", None) or text.num_attention_heads
    return {"q_proj": int(text.num_attention_heads), "k_proj": int(kv_heads), "v_proj": int(kv_heads)}


def _layer_shape(text: Any, layer: Optional[int]) -> Any:
    per_layer = getattr(text, "per_layer_config", None)
    return text if per_layer is None or layer is None else per_layer[layer]


def shard_sizes(config: Any, layer: Optional[int] = None) -> dict[str, int]:
    """Return the rows of each packable projection from a text config; `layer` reads that layer's own attention shape
    where the config keeps one per layer."""
    text = _text_config(config)
    head_dim = getattr(_layer_shape(text, layer), "head_dim", None) or text.hidden_size // text.num_attention_heads
    kv_heads = attention_heads(config, layer)["k_proj"]
    return {
        "q_proj": text.num_attention_heads * head_dim,
        "k_proj": kv_heads * head_dim,
        "v_proj": kv_heads * head_dim,
        "gate_proj": text.intermediate_size,
        "up_proj": text.intermediate_size,
    }


def engine_name(name: str) -> str:
    """Return an HF parameter or module path as the in-process engine names it."""
    for hf_prefix, engine_prefix in ENGINE_PREFIXES.items():
        if name.startswith(hf_prefix):
            return engine_prefix + name[len(hf_prefix):]
    return name


def _layer_of(module: str) -> Optional[int]:
    found = _LAYER.search(module)
    return None if found is None else int(found.group(1))


def packed_layout(module: str, config: Any, *, tp_size: int = 1) -> tuple[str, int]:
    """Return `(engine parameter name, row offset)` of an HF module's weight in one rank's engine."""
    parent, _, leaf = module.rpartition(".")
    if leaf not in PACKED_SHARDS:
        return engine_name(f"{module}.weight"), 0
    packed, index = PACKED_SHARDS[leaf]
    sizes = shard_sizes(config, _layer_of(module))
    siblings = [name for name, (target, _) in PACKED_SHARDS.items() if target == packed]
    offset = sum(sizes[name] // tp_size for name in siblings[:index])
    return engine_name(f"{parent}.{packed}.weight"), offset


@dataclass(frozen=True)
class ProbePoint:
    """One probed module: its HF path, where its rows sit in the engine, how many rows are checked and whether an adapter
    wraps it."""

    module: str
    engine_param: str
    row_offset: int
    rows: int
    lora: bool = True
    hf_row: int = 0
    hf_columns: Optional[tuple[int, int]] = None

    @classmethod
    def resolve(
        cls, module: str, config: Any, *, max_rows: int, lora: bool = True, tp_size: int = 1, tp_rank: int = 0,
    ) -> "ProbePoint":
        if max_rows < 1:
            raise SyncProbeError(f"a probe checks at least one row, got max_rows {max_rows}")
        if not 0 <= tp_rank < tp_size:
            raise SyncProbeError(f"tensor-parallel rank {tp_rank} is not one of {tp_size}")
        leaf = module.rpartition(".")[2]
        layer = _layer_of(module)
        total = shard_sizes(config, layer)[leaf] if leaf in PACKED_SHARDS else None
        engine_param, offset = packed_layout(module, config, tp_size=tp_size)
        if total is not None:
            if total % tp_size:
                raise SyncProbeError(f"{module}'s {total} rows do not split over tensor-parallel size {tp_size}")
            heads = attention_heads(config, layer).get(leaf)
            if heads is not None and heads % tp_size:
                raise SyncProbeError(
                    f"{module}'s {heads} heads do not split over tensor-parallel size {tp_size}: vLLM replicates "
                    "KV heads fewer than the ranks, which the probe does not model; probe another layer or module"
                )
            local = total // tp_size
            return cls(module, engine_param, offset, min(max_rows, local), lora, hf_row=tp_rank * local)
        columns = None
        if tp_size > 1 and leaf in ROW_PARALLEL:
            width = _row_parallel_width(config, leaf, layer)
            if width % tp_size:
                raise SyncProbeError(f"{module}'s {width} columns do not split over tensor-parallel size {tp_size}")
            columns = (tp_rank * width // tp_size, (tp_rank + 1) * width // tp_size)
        elif tp_size > 1 and lora:
            raise SyncProbeError(
                f"{module}: how vLLM shards a {leaf} over tensor parallelism is not declared; probe a packed "
                f"projection ({sorted(PACKED_SHARDS)}) or a row-parallel one ({list(ROW_PARALLEL)})"
            )
        return cls(module, engine_param, offset, max_rows, lora, hf_columns=columns)

    def region(self, weight: Any) -> Any:
        """Return the checked region of the full HF weight (or delta)."""
        rows = weight[self.hf_row : self.hf_row + self.rows]
        return rows if self.hf_columns is None else rows[:, self.hf_columns[0] : self.hf_columns[1]]


def _row_parallel_width(config: Any, leaf: str, layer: Optional[int]) -> int:
    if leaf == "down_proj":
        return int(_text_config(config).intermediate_size)
    return shard_sizes(config, layer)["q_proj"]


@dataclass
class PointResult:
    """One module's comparison after one sync."""

    module: str
    engine_param: str
    row_offset: int
    elements: int
    moved: int
    mismatched: int
    stale: int
    retention: Optional[float]
    rounded_away: Optional[float]
    lora: bool = True

    @property
    def ok(self) -> bool:
        return self.mismatched == 0


@dataclass
class ProbeResult:
    """Every probed module after one sync."""

    step: int
    engine_dtype: str
    points: list[PointResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(point.ok for point in self.points)

    @property
    def vacuous(self) -> bool:
        """Return whether no LoRA-probed element moved (equality cannot tell a stale engine from a fresh one)."""
        return all(point.moved == 0 for point in self.points if point.lora)

    @property
    def retention(self) -> Optional[float]:
        """Return the mean retention over the probed modules with a non-zero delta."""
        values = [point.retention for point in self.points if point.retention is not None]
        return sum(values) / len(values) if values else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "engine_dtype": self.engine_dtype,
            "ok": self.ok,
            "vacuous": self.vacuous,
            "retention": self.retention,
            "points": [asdict(point) for point in self.points],
        }

    def describe_failure(self) -> str:
        bad = [point for point in self.points if not point.ok]
        parts = [
            f"{point.module} ({point.engine_param} rows {point.row_offset}+): {point.mismatched} of "
            f"{point.elements} elements differ from cast(W + dW), {point.stale} of them equal to the base W "
            f"({point.moved} elements moved)"
            if point.lora else
            f"{point.module} ({point.engine_param} rows {point.row_offset}+, plain): {point.mismatched} of "
            f"{point.elements} elements differ from cast(W): not re-pushed after the engine's sleep?"
            for point in bad
        ]
        return f"sync probe failed after the sync at step {self.step}: " + "; ".join(parts)


def _submodule(model: torch.nn.Module, module: str) -> Optional[torch.nn.Module]:
    for name in (_PEFT_PREFIX + module, module):
        try:
            return model.get_submodule(name)
        except AttributeError:
            continue
    return None


def _is_lora(layer: Any) -> bool:
    return hasattr(layer, "get_delta_weight") and hasattr(layer, "get_base_layer")


def _lora_module(model: torch.nn.Module, module: str) -> Any:
    layer = _submodule(model, module)
    if layer is None or not _is_lora(layer):
        raise SyncProbeError(f"{module} is not a LoRA-wrapped module of the trained model")
    return layer


def _whole(tensor: torch.Tensor, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    full = getattr(tensor, "full_tensor", None)
    if full is None:
        return tensor
    if like is not None and hasattr(like, "placements"):
        return tensor.redistribute(like.device_mesh, like.placements).full_tensor()
    return full()


def _probed_weight(model: torch.nn.Module, point: ProbePoint) -> torch.Tensor:
    if point.lora:
        return _whole(_lora_module(model, point.module).get_base_layer().weight)
    layer = _submodule(model, point.module)
    if layer is None or _is_lora(layer) or not isinstance(getattr(layer, "weight", None), torch.Tensor):
        raise SyncProbeError(f"{point.module} is not a plain (unwrapped) module with a weight")
    return _whole(layer.weight)


def _delta(layer: Any) -> torch.Tensor:
    adapters = [name for name in layer.active_adapters if name in layer.lora_A]
    if not adapters:
        raise SyncProbeError("the LoRA layer has no active adapter to merge")
    delta = layer.get_delta_weight(adapters[0])
    for name in adapters[1:]:
        delta = delta + layer.get_delta_weight(name)
    return _whole(delta, like=layer.get_base_layer().weight)


class SyncProbe:
    """Compare the engine's weights with the trainer's merge after every sync."""

    def __init__(self, points: Sequence[ProbePoint]) -> None:
        if not points:
            raise SyncProbeError("a sync probe needs at least one module to probe")
        self.points = list(points)
        self._snapshots: Optional[dict[str, torch.Tensor]] = None
        #: The last accepted check: its step and, per module, the engine rows it read and cast(W).
        self._accepted: Optional[tuple[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]] = None
        self.history: list[ProbeResult] = []
        self.generation_checks = 0

    @classmethod
    def for_modules(
        cls, modules: Sequence[str], config: Any, *, max_rows: int, plain: Sequence[str] = (), tp_size: int = 1,
        tp_rank: int = 0,
    ) -> "SyncProbe":
        """Return a probe over LoRA-wrapped `modules` and `plain` ones on the engine of rank `tp_rank` of `tp_size`.
        """
        where = {"tp_size": tp_size, "tp_rank": tp_rank}
        return cls(
            [ProbePoint.resolve(module, config, max_rows=max_rows, **where) for module in modules]
            + [ProbePoint.resolve(module, config, max_rows=max_rows, lora=False, **where) for module in plain]
        )

    @torch.no_grad()
    def snapshot(self, model: torch.nn.Module) -> None:
        """Copy the probed region of each base weight, before the sync merges into it."""
        self._snapshots = {}
        for point in self.points:
            self._snapshots[point.module] = point.region(_probed_weight(model, point)).detach().clone()

    @torch.no_grad()
    def check(self, model: torch.nn.Module, engine_params: Mapping[str, torch.Tensor], *, step: int) -> ProbeResult:
        """Compare the engine with `cast(W + dW)`; raise `SyncProbeError` on any mismatch."""
        if self._snapshots is None:
            raise SyncProbeError("check() without a snapshot() before the sync")
        snapshots, self._snapshots, self._accepted = self._snapshots, None, None
        self._require_params(engine_params)
        dtype = engine_params[self.points[0].engine_param].dtype
        result = ProbeResult(step=step, engine_dtype=str(dtype).removeprefix("torch."))
        accepted: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for point in self.points:
            engine = engine_params[point.engine_param]
            base = snapshots[point.module]
            layer = _lora_module(model, point.module) if point.lora else None
            delta = None if layer is None else _delta(layer)
            expected = (base if delta is None else base + point.region(delta)).to(engine.dtype)
            held = engine[point.row_offset : point.row_offset + point.rows].to(expected.device)
            if held.shape != expected.shape:
                raise SyncProbeError(
                    f"{point.engine_param} rows {point.row_offset}..{point.row_offset + point.rows} have shape "
                    f"{tuple(held.shape)}, not {point.module}'s {tuple(expected.shape)}: the packed layout is wrong"
                )
            unmoved = base.to(engine.dtype)
            accepted[point.module] = (held.clone(), unmoved)
            moved = expected != unmoved
            mismatch = held != expected
            summary = {"retention": None, "rounded_away": None}
            if layer is not None:
                tally = RetentionTally()
                weight = _whole(layer.get_base_layer().weight)
                tally.add(point.module, weight.detach().float(), delta.detach().float())
                summary = tally.summary()["overall"]
            result.points.append(PointResult(
                module=point.module,
                engine_param=point.engine_param,
                row_offset=point.row_offset,
                elements=expected.numel(),
                moved=int(moved.sum()),
                mismatched=int(mismatch.sum()),
                stale=int((mismatch & (held == unmoved)).sum()),
                retention=summary["retention"],
                rounded_away=summary["rounded_away"],
                lora=point.lora,
            ))
        self.history.append(result)
        if not result.ok:
            raise SyncProbeError(result.describe_failure())
        self._accepted = (step, accepted)
        return result

    @torch.no_grad()
    def check_unchanged(self, engine_params: Mapping[str, torch.Tensor], *, step: int) -> None:
        """Require the engine to hold the rows the last accepted `check` read, exactly."""
        if self._accepted is None:
            raise SyncProbeError(f"the engine samples at step {step} with no accepted sync probe before it")
        self._require_params(engine_params)
        synced_step, accepted = self._accepted
        changed = []
        for point in self.points:
            held, unmoved = accepted[point.module]
            now = engine_params[point.engine_param][point.row_offset : point.row_offset + point.rows].to(held.device)
            differ = now != held
            if bool(differ.any()):
                changed.append(
                    f"{point.module} ({point.engine_param} rows {point.row_offset}+): {int(differ.sum())} of "
                    f"{held.numel()} elements, {int((differ & (now == unmoved)).sum())} of them now equal to the base W"
                )
        if changed:
            raise SyncProbeError(
                f"the engine's weights changed between the sync probed at step {synced_step} and sampling at step "
                f"{step} (reloaded or discarded after the sync?): " + "; ".join(changed)
            )
        self.generation_checks += 1

    def _require_params(self, engine_params: Mapping[str, torch.Tensor]) -> None:
        absent = [point.engine_param for point in self.points if point.engine_param not in engine_params]
        if absent:
            raise SyncProbeError(f"the engine has no parameter(s) {absent}: the packed layout or the names are wrong")


@dataclass
class RetentionAlarm:
    """The retention still below `threshold` at the first sync at or after `step` (fires once)."""

    step: int = 100
    threshold: float = 0.5
    fired: Optional[dict[str, Any]] = None
    checked: bool = False

    def observe(self, result: ProbeResult) -> Optional[dict[str, Any]]:
        """Return the alarm record when this result triggers it, else None."""
        if self.checked or result.step < self.step:
            return None
        self.checked = True
        retention = result.retention
        if retention is None or retention >= self.threshold:
            return None
        self.fired = {
            "alarm": "retention",
            "step": result.step,
            "retention": retention,
            "threshold": self.threshold,
            "checked_from_step": self.step,
            "per_module": {point.module: point.retention for point in result.points},
            "action": "none automatic: you choose the response",
        }
        return self.fired


def probed_sync(
    sync: Callable[[], None],
    *,
    probe: SyncProbe,
    model: torch.nn.Module,
    engine_params: Callable[[], Mapping[str, torch.Tensor]],
    step: Callable[[], int],
    on_result: Callable[[ProbeResult], None],
) -> Callable[[], None]:
    """Return `sync` wrapped so that every call is snapshotted before and probed after."""

    def run() -> None:
        probe.snapshot(model)
        sync()
        on_result(probe.check(model, engine_params(), step=step()))

    return run


def checked_generate(
    generate: Callable[..., Any],
    *,
    probe: SyncProbe,
    engine_params: Callable[[], Mapping[str, torch.Tensor]],
    step: Callable[[], int],
    on_check: Callable[[], None],
) -> Callable[..., Any]:
    """Return `generate` wrapped so that the engine is checked unchanged since the last probed sync before every call.
    """

    def run(*args: Any, **kwargs: Any) -> Any:
        probe.check_unchanged(engine_params(), step=step())
        on_check()
        return generate(*args, **kwargs)

    return run
