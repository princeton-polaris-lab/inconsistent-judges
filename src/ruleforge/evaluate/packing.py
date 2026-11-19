from __future__ import annotations

import dataclasses
import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ruleforge.evaluate.ledger import LedgerState
from ruleforge.evaluate.runner import (
    BlockRequest,
    BlockWork,
    CampaignRunError,
    DrawRunner,
    TransportExhausted,
    draw_run_lock,
)

logger = logging.getLogger(__name__)

# Bounds what one call holds in memory and what a crash mid-call throws away.
DEFAULT_PACK_MAX_PROMPTS = 4096
PACKABLE_MODEL_TYPE = "local"


def _positive_int(name: str, value: Any) -> int:
    # bool is an int subclass.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CampaignRunError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class ExecutionSettings:
    """How a job schedules its generation; never what it generates, and never part of a campaign's identity."""

    pack_draws: bool = False
    pack_max_prompts: int = DEFAULT_PACK_MAX_PROMPTS
    pack_max_draws: Optional[int] = None
    allow_stack_change: bool = False
    num_shards: Optional[int] = None
    shard_index: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ("pack_draws", "allow_stack_change"):
            if not isinstance(getattr(self, name), bool):
                raise CampaignRunError(f"execution.{name} must be true or false, got {getattr(self, name)!r}")
        _positive_int("execution.pack_max_prompts", self.pack_max_prompts)
        if self.pack_max_draws is not None:
            _positive_int("execution.pack_max_draws", self.pack_max_draws)
        if (self.num_shards is None) != (self.shard_index is None):
            raise CampaignRunError("execution.num_shards and execution.shard_index are set together or not at all")
        if self.sharded:
            _positive_int("execution.num_shards", self.num_shards)
            index = self.shard_index
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.num_shards:
                raise CampaignRunError(
                    f"execution.shard_index must be an integer in [0, {self.num_shards}), got {index!r}"
                )

    @property
    def sharded(self) -> bool:
        """Whether this job runs one shard of its condition's units."""
        return self.num_shards is not None

    @classmethod
    def from_mapping(cls, mapping: Optional[Mapping[str, Any]]) -> "ExecutionSettings":
        """Build from a config's ``execution:`` block (None: the defaults), refusing unknown keys."""
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise CampaignRunError("execution must be a mapping")
        unknown = sorted(set(mapping) - _EXECUTION_KEYS)
        if unknown:
            raise CampaignRunError(f"unknown execution keys: {unknown}")
        return cls(**dict(mapping))


_EXECUTION_KEYS = frozenset(item.name for item in dataclasses.fields(ExecutionSettings))


def pack_blocks(sizes: Sequence[int], max_prompts: int) -> list[list[int]]:
    """Group block indices, in order, into calls of at most ``max_prompts`` prompts; a larger block is submitted
    alone."""
    _positive_int("max_prompts", max_prompts)
    calls: list[list[int]] = []
    current: list[int] = []
    total = 0
    for index, size in enumerate(sizes):
        if current and total + size > max_prompts:
            calls.append(current)
            current, total = [], 0
        current.append(index)
        total += size
    if current:
        calls.append(current)
    return calls


def _named_adapter(runner: DrawRunner) -> tuple[Optional[str], Optional[str]]:
    condition = runner.spec.condition
    return condition.adapter_sha256, condition.adapter_config_sha256


def _adapter_text(runner: DrawRunner) -> str:
    weights, config = _named_adapter(runner)
    return f"(weights SHA-256 {weights}, config SHA-256 {config})"


def check_packable(runners: Sequence[DrawRunner]) -> None:
    """Refuse runners one packed call cannot serve: one backend, temperature, truncation and adapter; no batch judge."""
    first = runners[0]
    backend = first.backend
    if backend.model_type != PACKABLE_MODEL_TYPE:
        raise CampaignRunError(
            f"packed generation serves a {PACKABLE_MODEL_TYPE!r} judge only, not "
            f"{backend.model_type!r}: a batch provider addresses its paid batch by request "
            "block, so packing blocks of several draws would change what a resume re-addresses"
        )
    if first._windowed():
        raise CampaignRunError("packed generation cannot drive a windowed batch judge")
    for runner in runners[1:]:
        if runner.backend is not backend:
            raise CampaignRunError(
                f"{runner.draw_dir}: every packed draw must share one backend instance"
            )
        if runner.spec.config.temperature != first.spec.config.temperature:
            raise CampaignRunError(
                f"{runner.draw_dir}: temperature {runner.spec.config.temperature} is not "
                f"{first.spec.config.temperature}; one packed call samples at one temperature"
            )
        if runner.sampler_truncation() != first.sampler_truncation():
            raise CampaignRunError(
                f"{runner.draw_dir}: sampling truncation {runner.sampler_truncation()} is not "
                f"{first.sampler_truncation()}; one packed call samples with one truncation"
            )
        if _named_adapter(runner) != _named_adapter(first):
            raise CampaignRunError(
                f"{runner.draw_dir}: its condition names adapter {_adapter_text(runner)} but "
                f"{first.draw_dir}'s names {_adapter_text(first)}; one backend serves one "
                "adapter, so their draws cannot share a packed call"
            )
    directories = [runner.draw_dir.resolve() for runner in runners]
    if len(set(directories)) != len(directories):
        raise CampaignRunError("a draw directory is named twice; one ledger has one writer")


@dataclass
class _Draw:
    runner: DrawRunner
    state: LedgerState


def run_packed(
    runners: Sequence[DrawRunner],
    *,
    max_prompts_per_call: int = DEFAULT_PACK_MAX_PROMPTS,
) -> list[dict[str, Any]]:
    """Bring every runner's draw to a terminal state and return completions in runner order. Raises
    ``TransportExhausted``, leaving every draw resumable, when a call keeps failing or a transport cap is reached."""
    runners = list(runners)
    if not runners:
        return []
    _positive_int("max_prompts_per_call", max_prompts_per_call)
    check_packable(runners)
    execution = {"pack_draws": True, "pack_max_prompts": max_prompts_per_call, "packed_draws": len(runners)}
    with ExitStack() as held:
        for runner in runners:
            held.enter_context(draw_run_lock(runner.draw_dir))
        draws = [_Draw(runner, runner.prepare(execution=execution)) for runner in runners]
        max_attempts = max(runner.spec.config.max_semantic_attempts for runner in runners)
        retries = min(runner.spec.max_transport_retries for runner in runners)
        for attempt in range(max_attempts):
            while True:
                work = [
                    (draw, block)
                    for draw in draws
                    if attempt < draw.runner.spec.config.max_semantic_attempts
                    for block in draw.runner.plan_pass(draw.state, attempt)
                ]
                if not work:
                    break
                sizes = [draw.runner.prompt_count(block) for draw, block in work]
                calls = pack_blocks(sizes, max_prompts_per_call)
                logger.info(
                    "Attempt %d: %d block(s) of %d draw(s), %d prompt(s), in %d packed call(s)",
                    attempt, len(work), len({id(draw) for draw, _ in work}), sum(sizes), len(calls),
                )
                for indices in calls:
                    _run_call([work[i] for i in indices], retries=retries)
        return [draw.runner.finish(draw.state) for draw in draws]


def _run_call(work: Sequence[tuple[_Draw, BlockWork]], *, retries: int) -> None:
    requests: list[tuple[_Draw, BlockRequest]] = [
        (draw, draw.runner.build_request(block)) for draw, block in work
    ]
    first = requests[0][0].runner
    prompts = [prompt for _, request in requests for prompt in request.prompts]
    seeds: Optional[list[int]] = None
    if first.send_seeds:
        seeds = [seed for _, request in requests for seed in request.seeds]
    label = f"packed:{len(requests)}-blocks:{requests[0][1].work.block_id}"
    # Which prompts share a batch matters since vLLM is not batch-invariant, so the job log records it.
    logger.info(
        "Packed call %s: %d prompt(s) from block(s) %s",
        label, len(prompts), ", ".join(request.work.block_id for _, request in requests),
    )
    failures = 0
    while True:
        started = [draw.runner.clock() for draw, _ in requests]
        outcomes = first.backend.generate_once(
            prompts,
            temperature=first.spec.config.temperature,
            seeds=seeds,
            request_block_id=label,
            truncation=first.sampler_truncation(),
        )
        completed = [draw.runner.clock() for draw, _ in requests]
        if len(outcomes) != len(prompts):
            raise CampaignRunError(
                f"{label}: the judge returned {len(outcomes)} outcomes for {len(prompts)} prompts"
            )
        if not all(outcome.block_failed for outcome in outcomes):
            break
        # The whole call raised: resubmit the same packing until the cap says the judge is down.
        failures += 1
        logger.warning("Packed call %s raised before any result: %s", label, outcomes[0].error)
        if failures >= retries:
            raise TransportExhausted(
                f"packed call {label} raised {failures} time(s) in a row; every draw "
                "stays resumable, fix the judge and rerun"
            )
    position = 0
    for (draw, request), started_at, completed_at in zip(requests, started, completed):
        end = position + len(request.prompts)
        draw.runner.record(
            draw.state, request, outcomes[position:end],
            started_at=started_at, completed_at=completed_at,
        )
        position = end
