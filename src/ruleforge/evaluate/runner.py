from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import socket
import sys
import time
import weakref
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping, Optional, Sequence

from ruleforge.evaluate.report import build_per_scenario_rows, summarize_rows
from ruleforge.rulesets.variants import (
    build_rule_variants,
    rule_set_text,
    write_implicated_sidecar,
)
from ruleforge.evaluate.scoring import PHI_MAP, calculate_inconsistency, max_rule_phi
from ruleforge.evaluate.identity import (
    BUNDLE_OUTCOMES_FNAME,
    CAMPAIGN_MODE_FINAL,
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_BOTH,
    EVALUATION_MODE_FULL,
    FULL_RULESET_COMPONENT,
    RETRY_SCOPE_COMPONENT,
    RUN_LOG_ROWS_KEY,
    RUNS_FNAME,
    ConditionSpec,
    SamplingConfig,
    check_exact_grid,
    derive_generation_seed,
    request_block_id,
    run_log_damage,
    run_log_problem,
    run_log_rows,
)
from ruleforge.scenarios.manifest import (
    SAMPLING_SCHEMA_VERSION,
    ScenarioManifest,
)
from ruleforge.judges.weights import (
    ModelWeightsIdentity,
    adapter_identity,
    model_weights_identity,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.record_names import PER_SCENARIO_FNAME, SUMMARY_FNAME
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.ledger import (
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_TRANSPORT_FAILED,
    AttemptRecord,
    ComponentRecord,
    DrawLedger,
    LedgerState,
    raw_output_sha256,
)
from ruleforge.evaluate.sharding import DrawShard, block_members, shard_block_id
from ruleforge.evaluate.stack import (
    ENGINE_ARGS_KEY,
    LEDGER_ROWS_KEY,
    LOCAL_MODEL_TYPE,
    PRECISION_ENV_KEY,
    STACK_CHANGE_KEY,
    STACK_PACKAGES,
    condition_stack,
    describe_changes,
    read_condition_stack,
    recorded_runtime,
    runtime_changes,
    stack_digest,
    unlogged_stacks,
    write_condition_stack,
)
from ruleforge.evaluate.validation import TRANSPORT_REASONS, ComponentSchema
from ruleforge.tasks.spec import BaseTaskSpec, task_identity_of
from ruleforge.tasks.base import TaskError
from ruleforge.core.artifacts import (
    compute_scenario_id,
    make_evaluation_record,
    prompt_sha256,
    write_jsonl_records,
)
from ruleforge.judges.backend import (
    SEED_UNSUPPORTED,
    GenerationOutcome,
    ModelBackend,
    canonical_serving_dtype,
    precision_env,
)
from ruleforge.judges.batch_api import (
    DEFAULT_POLL_SECONDS,
    FATAL_PROVIDER_ERRORS,
    REASONING_KEY,
    REASONING_SUMMARIES_KEY,
    BatchExecutionError,
    reasoning_fields,
    require_spend_confirmation,
)
from ruleforge.core.constants import JUDGE_OUTPUTS_FNAME
from ruleforge.core.resume import count_complete_jsonl_records

logger = logging.getLogger(__name__)

REQUEST_BLOCKS_FNAME = "request_blocks.jsonl"
# A batch judge's request settings (what it asks the provider to show of its reasoning): in no identity, but in a paid
# batch's address, so a resumed block is held to those recorded.
REQUEST_SETTINGS_KEY = "request_settings"
# Settings of a block recorded without that key, where its transport fixed them.
UNRECORDED_REQUEST_SETTINGS: dict[str, dict[str, Any]] = {"anthropic_batch": {}}
# Provenance fields that describe the process rather than the draw; a resume from another host may differ in these only.
PROVENANCE_RUN_FIELDS = (
    "code_revision", "code_dirty", "code_state_sha256", "package_version", "runtime", "tp_size", "attention_backend",
    "adapter", "model_weights",
    REQUEST_SETTINGS_KEY,
)
# Runtime key for what a local engine resolved at start, checked against the condition when the runner starts.
RESOLVED_ENGINE_KEY = "resolved_engine"

# Digested once per engine: a file changed after loading does not change what the engine serves.
_SERVED_WEIGHTS: "weakref.WeakKeyDictionary[Any, Optional[ModelWeightsIdentity]]" = weakref.WeakKeyDictionary()


def served_model_weights(backend: Any) -> Optional[ModelWeightsIdentity]:
    """Return the derived weights a local backend serves, by content, or None for upstream weights or a provider."""
    path = getattr(backend, "model_path", None)
    if backend.model_type != LOCAL_MODEL_TYPE or path is None:
        return None
    if backend not in _SERVED_WEIGHTS:
        _SERVED_WEIGHTS[backend] = model_weights_identity(path)
    return _SERVED_WEIGHTS[backend]


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


# Informational fields in recorded provenance, null in new records.
RETIRED_RUNNER_FIELDS = ("modal_run_id", "modal_task_id", "runpod_run_id", "runpod_pod_id")


def runtime_provenance(*, local_engine: bool = True) -> dict[str, Any]:
    """Return where and with which runtime this process runs; informational, never identity."""
    torch = sys.modules.get("torch")
    gpu: Optional[str] = None
    if torch is not None:
        try:
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:  # pragma: no cover - driver-specific failures
            gpu = None
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        **dict.fromkeys(RETIRED_RUNNER_FIELDS),
        "python_version": platform.python_version(),
        **{f"{name}_version": _package_version(name) for name in STACK_PACKAGES},
        "gpu": gpu,
        PRECISION_ENV_KEY: precision_env(strict=local_engine),
    }


EVALUATIONS_FNAME = "evaluations.jsonl"

STATE_IN_PROGRESS = "in_progress"
STATE_COMPLETE = "complete"
STATE_INCOMPLETE_EXHAUSTED = "incomplete_exhausted"
STATE_INVALID_MANIFEST = "invalid_manifest"
BUNDLE_FAILURE_COMPONENT_INVALID = "component_invalid"
BUNDLE_FAILURE_TRANSPORT = "transport"

DEFAULT_MAX_TRANSPORT_RETRIES = 3

# Runs pilots, never a final campaign.
FAKE_MODEL_TYPE = "fake"


class CampaignRunError(SamplingDesignError):
    """Raised when a draw cannot be run or finished as configured."""


def draw_ended(gate: Mapping[str, Any]) -> bool:
    """Whether a draw's gate (or a campaign gate's entry for the draw) says it ended: complete, or out of attempts with
    nothing pending."""
    state = gate.get("state")
    return state == STATE_COMPLETE or (state == STATE_INCOMPLETE_EXHAUSTED and gate.get("pending") == 0)


def refuse_fake_in_final(model_type: str, campaign_mode: str) -> None:
    """Refuse the ``fake`` judge in a ``final`` campaign: a final campaign's numbers are a model's."""
    if model_type == FAKE_MODEL_TYPE and campaign_mode == CAMPAIGN_MODE_FINAL:
        raise CampaignRunError(
            "the fake judge answers by a stated function of its seed and the prompt, not as a model: it runs "
            f"pilot and screening campaigns, never a {CAMPAIGN_MODE_FINAL!r} one"
        )


def bundle_status(components: Sequence[ComponentRecord]) -> tuple[str, Optional[str]]:
    """Return an attempted bundle's status and failure reason, from its components' validity."""
    if all(component.valid for component in components):
        return STATUS_ACCEPTED, None
    if any(not component.valid and component.failure_reason not in TRANSPORT_REASONS for component in components):
        return STATUS_REJECTED, BUNDLE_FAILURE_COMPONENT_INVALID
    return STATUS_TRANSPORT_FAILED, BUNDLE_FAILURE_TRANSPORT


class TransportExhausted(CampaignRunError):
    """Raised when a bundle keeps failing in transport; the draw stays resumable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_RUN_WRITE_GUARD: ContextVar[Optional[Callable[[Path], None]]] = ContextVar("run_write_guard", default=None)


def check_run_write(directory: Path) -> None:
    """Run the active write guard, if any, on ``directory`` before the runner writes there."""
    guard = _RUN_WRITE_GUARD.get()
    if guard is not None:
        guard(Path(directory))


@contextmanager
def guard_run_writes(guard: Callable[[Path], None]) -> Iterator[None]:
    """Run ``guard(directory)`` before each of a run's writes inside the block; what is written does not change."""
    token = _RUN_WRITE_GUARD.set(guard)
    try:
        yield
    finally:
        _RUN_WRITE_GUARD.reset(token)


_SHARED_WRITE_HOLD: ContextVar[Optional[Callable[[], ContextManager[Any]]]] = ContextVar("shared_write_hold",
                                                                                         default=None)


@contextmanager
def shared_write() -> Iterator[None]:
    """Hold what the caller set around a draw's read and write of a file its condition's other draws share."""
    hold = _SHARED_WRITE_HOLD.get()
    with nullcontext() if hold is None else hold():
        yield


@contextmanager
def hold_shared_writes(hold: Callable[[], ContextManager[Any]]) -> Iterator[None]:
    """Hold ``hold()`` around each write the runner makes to a file a condition's draws share, inside the block."""
    token = _SHARED_WRITE_HOLD.set(hold)
    try:
        yield
    finally:
        _SHARED_WRITE_HOLD.reset(token)


DRAW_LOCK_SUFFIX = ".lock"


def draw_lock_path(draw_dir: Path | str) -> Path:
    """Return the run lock of the draw at ``draw_dir``: ``.draw-NNN.lock`` in its condition's directory."""
    draw = Path(draw_dir)
    return draw.with_name(f".{draw.name}{DRAW_LOCK_SUFFIX}")


def _inode(path: Path | str, directory: Optional[int] = None) -> Optional[tuple[int, int]]:
    try:
        found = os.stat(path, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return found.st_dev, found.st_ino


@contextmanager
def draw_run_lock(draw_dir: Path | str) -> Iterator[Path]:
    """Hold the draw's exclusive run lock, or refuse at once while another run holds it (two would both repair the
    ledger and append the same blocks); a killed run's empty, unheld lock file is taken by the next run."""
    draw = Path(draw_dir)
    path = draw_lock_path(draw)
    check_run_write(draw)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError as error:
            raise CampaignRunError(f"{path.parent}, where the run lock of {draw} goes, cannot be opened as a "
                                   f"directory ({error})") from error
        try:
            descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666,
                                 dir_fd=directory)
        except OSError as error:
            os.close(directory)
            raise CampaignRunError(f"{path}, the run lock of {draw}, cannot be opened as a file of its own ({error}); "
                                   "a symlink there is never followed: remove it") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = os.fstat(descriptor)
            current = _inode(path)
        except BaseException as error:
            os.close(descriptor)
            os.close(directory)
            if isinstance(error, BlockingIOError):
                raise CampaignRunError(
                    f"another process holds {path}: it is running {draw}, and two runs of one draw would each repair "
                    "and append to its ledger; run this one again once that run ends (what it completes is not "
                    "judged again)") from error
            raise
        if current == (held.st_dev, held.st_ino):
            break
        # removed or replaced by the run that held it since it was opened: open it again
        os.close(descriptor)
        os.close(directory)
    try:
        yield path
    finally:
        try:
            # Removed only while it is the file held, never another run's lock now at this path.
            if _inode(path.name, directory) == (held.st_dev, held.st_ino):
                os.unlink(path.name, dir_fd=directory)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            os.close(directory)


@dataclass(frozen=True)
class _PlannedComponent:

    scenario_id: str
    component_id: str
    seed: int
    prompt_sha256: str


@dataclass(frozen=True)
class BlockWork:
    """One request block of an attempt pass, planned and recorded but not yet answered: a re-submission re-runs all of
    ``members`` and records only ``record_for``; ``components`` aligns with ``members`` (None: all)."""

    block_id: str
    semantic_attempt_id: int
    members: tuple[str, ...]
    record_for: tuple[str, ...]
    components: Optional[tuple[tuple[str, ...], ...]] = None


@dataclass(frozen=True)
class BlockRequest:
    """A block's prompts, built only when it is about to be submitted, and what each prompt is."""

    work: BlockWork
    prompts: tuple[dict[str, str], ...]
    planned: tuple[_PlannedComponent, ...]
    # None for a backend that takes none.
    seeds: Optional[tuple[int, ...]]

    def seed_list(self) -> Optional[list[int]]:
        """The seeds as a judge call takes them, or None."""
        return None if self.seeds is None else list(self.seeds)


@dataclass
class _WindowBlock:

    request: BlockRequest
    est_tokens: int
    started_at: Optional[str] = None

    @property
    def block_id(self) -> str:
        return self.request.work.block_id


@dataclass(frozen=True)
class DrawSpec:
    """Everything that identifies one draw of one condition."""

    campaign_id: str
    condition: ConditionSpec
    decoder_draw_id: int
    config: SamplingConfig
    scenario_manifest: ScenarioManifest
    judge_type: str
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES
    # The blocks of this draw one shard runs; None runs the whole draw.
    shard: Optional[DrawShard] = None
    # ``{scenario id: cap}`` of bundles a cap amendment holds at the old cap.
    held_caps: Mapping[str, int] = field(default_factory=dict)

    def attempt_cap(self, scenario_id: str) -> int:
        """Return the attempts a scenario's bundle may use in this draw."""
        return self.held_caps.get(scenario_id, self.config.max_semantic_attempts)

    def scope_ids(self) -> tuple[str, ...]:
        """The manifest scenarios this draw's runner brings to a terminal state, in manifest order."""
        ids = self.scenario_manifest.scenario_ids
        if self.shard is None:
            return tuple(ids)
        return self.shard.scenario_ids(ids, self.config.scenarios_per_request_block)


def pending_count(state: LedgerState, spec: DrawSpec, scope_ids: Sequence[str]) -> int:
    """Return the scenarios of ``scope_ids`` with no accepted bundle and an attempt still to spend."""
    return sum(
        1
        for sid in scope_ids
        if not state.is_accepted(sid) and state.next_attempt(sid) < spec.attempt_cap(sid)
    )


def ledger_counts(state: LedgerState, *, component_scope: bool) -> dict[str, Any]:
    """Return the fields of a draw's gate its ledger rows alone decide."""
    histogram: dict[str, int] = {}
    for record in state.accepted.values():
        key = str(record.semantic_attempt_id + 1)
        histogram[key] = histogram.get(key, 0) + 1
    invalid: dict[str, int] = {}
    for record in state.records:
        for component in record.components:
            if not component.valid:
                invalid[component.failure_reason] = invalid.get(component.failure_reason, 0) + 1
    counts: dict[str, Any] = {
        "accepted": len(state.accepted),
        "attempts_histogram": dict(sorted(histogram.items())),
        "rejected_attempts": sum(1 for r in state.records if r.status == STATUS_REJECTED),
        "transport_failed_attempts": sum(1 for r in state.records if r.status == STATUS_TRANSPORT_FAILED),
        "invalid_components": dict(sorted(invalid.items())),
    }
    if component_scope:
        per_component: dict[str, int] = {}
        for sid in state.accepted:
            for record, _ in state.accepted_components(sid).values():
                key = str(record.semantic_attempt_id + 1)
                per_component[key] = per_component.get(key, 0) + 1
        counts["component_attempts_histogram"] = dict(sorted(per_component.items()))
    return counts


def draw_completion(
    state: LedgerState,
    *,
    spec: DrawSpec,
    scope_ids: Sequence[str],
    components_per_bundle: int,
    materialized: bool,
    updated_at: str,
    runs_logged: int,
) -> dict[str, Any]:
    """Return a draw's ``completion.json`` payload from its ledger state; ``complete`` needs the exact grid and the
    derived views."""
    manifest = spec.scenario_manifest
    grid = check_exact_grid(scope_ids, 1, [(sid, 0) for sid in state.accepted])
    exhausted = sorted(
        sid
        for sid in scope_ids
        if not state.is_accepted(sid) and state.next_attempt(sid) >= spec.attempt_cap(sid)
    )
    counts = ledger_counts(state, component_scope=spec.config.retry_scope == RETRY_SCOPE_COMPONENT)
    extra: dict[str, Any] = {}
    if "component_attempts_histogram" in counts:
        extra["component_attempts_histogram"] = counts["component_attempts_histogram"]
    if spec.shard is not None:
        extra["shard"] = {**spec.shard.to_dict(), "scenarios": len(scope_ids)}
    extra[RUN_LOG_ROWS_KEY] = runs_logged
    # Scope scenarios only: a foreign row must not drive this negative.
    pending = pending_count(state, spec, scope_ids)
    if grid.ok and materialized:
        run_state = STATE_COMPLETE
    elif exhausted and not pending:
        run_state = STATE_INCOMPLETE_EXHAUSTED
    else:
        run_state = STATE_IN_PROGRESS
    return {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "campaign_id": spec.campaign_id,
        "condition_id": spec.condition.condition_id,
        "condition_fingerprint": spec.condition.fingerprint,
        "decoder_draw_id": spec.decoder_draw_id,
        "state": run_state,
        "materialized": bool(grid.ok and materialized),
        "scenario_manifest_sha256": manifest.manifest_sha256,
        "expected_scenarios": manifest.record_count,
        "components_per_bundle": components_per_bundle,
        "grid": grid.to_dict(),
        "accepted": counts["accepted"],
        "pending": pending,
        "exhausted": exhausted,
        "attempts_histogram": counts["attempts_histogram"],
        "rejected_attempts": counts["rejected_attempts"],
        "transport_failed_attempts": counts["transport_failed_attempts"],
        "invalid_components": counts["invalid_components"],
        **extra,
        "updated_at": updated_at,
    }


def write_draw_summary(
    draw_dir: Path,
    evaluation_rows: Sequence[Mapping[str, Any]],
    *,
    model_name: str,
    judge_type: str,
    evaluation_mode: str,
    decoder_draw_id: int,
    condition_id: str,
    components_per_bundle: int,
) -> Optional[Path]:
    """Write a complete draw's summary and, under ``both``, its per-scenario file; return that path or None."""
    if evaluation_mode != EVALUATION_MODE_BOTH:
        write_json_sidecar(
            draw_dir / SUMMARY_FNAME,
            {
                "decoder_draw_id": decoder_draw_id,
                "condition_id": condition_id,
                "evaluation_mode": evaluation_mode,
                "n_accepted": len(evaluation_rows),
                "components_per_bundle": components_per_bundle,
            },
        )
        return None
    drop_counts: dict[str, int] = {}
    rows = build_per_scenario_rows(
        evaluation_rows,
        model_name,
        judge_type,
        ruleset_key=FULL_RULESET_COMPONENT,
        require_not_applicable=False,
        applicability_judge_type=judge_type,
        drop_counts=drop_counts,
    )
    if len(rows) != len(evaluation_rows):  # pragma: no cover - every accepted bundle carries canonical labels
        raise CampaignRunError(f"scored {len(rows)} of {len(evaluation_rows)} accepted bundles")
    path = draw_dir / PER_SCENARIO_FNAME
    write_jsonl_records(path, rows)
    summary = summarize_rows(rows, n_input_scenarios=len(evaluation_rows), drop_counts=drop_counts)
    summary["decoder_draw_id"] = decoder_draw_id
    summary["condition_id"] = condition_id
    write_json_sidecar(draw_dir / SUMMARY_FNAME, summary)
    return path


def bundle_outcome(schema: ComponentSchema, record: AttemptRecord) -> dict[str, Any]:
    """Return one accepted bundle's inconsistency, its full-set label against its strongest rule label; a ``full``
    bundle carries only its label and rule ids, an ``individual_bank`` bundle its rule judgments."""
    by_component = {c.component_id: c for c in record.components}
    mode = schema.evaluation_mode
    payload: dict[str, Any] = {
        "scenario_id": record.scenario_id,
        "decoder_draw_id": record.decoder_draw_id,
        "semantic_attempt_id": record.semantic_attempt_id,
        "full_set_judgment": None,
        "full_set_phi": None,
        "rule_judgments": None,
        "max_rule_phi": None,
        "contradiction": None,
        "penalty": None,
        "in_context_rule_judgments": None,
        "in_context_contradiction": None,
        "in_context_penalty": None,
    }
    if mode != EVALUATION_MODE_FULL:
        rule_judgments = {
            rule_id: by_component[rule_id].judgment for rule_id in schema.expected_rule_ids
        }
        payload["rule_judgments"] = rule_judgments
        payload["max_rule_phi"] = max_rule_phi(list(rule_judgments.values()))
    if mode == EVALUATION_MODE_BANK:
        return payload
    full = by_component[FULL_RULESET_COMPONENT]
    payload["full_set_judgment"] = full.judgment
    payload["full_set_phi"] = PHI_MAP[full.judgment]
    if mode == EVALUATION_MODE_BOTH:
        scored = calculate_inconsistency(full.judgment, list(payload["rule_judgments"].values()))
        payload["contradiction"] = scored["contradiction"]
        payload["penalty"] = scored["penalty"]
    else:
        payload["ruleset_rule_ids"] = list(schema.expected_rule_ids)
    if full.rule_judgments is not None:
        in_context = calculate_inconsistency(full.judgment, list(full.rule_judgments.values()))
        payload["in_context_rule_judgments"] = dict(full.rule_judgments)
        payload["in_context_contradiction"] = in_context["contradiction"]
        payload["in_context_penalty"] = in_context["penalty"]
    return payload


@dataclass(frozen=True)
class DrawViews:
    """Writes a complete draw's derived views from its ledger."""

    spec: DrawSpec
    schema: ComponentSchema
    model_name: str
    scenarios: Mapping[str, Mapping[str, Any]]
    bundle_prompts: Callable[[str], Sequence[tuple[str, dict[str, str]]]]
    individual_rule_context: str
    implication_map: Optional[Mapping[str, Any]] = None

    def evaluation_record(self, record: AttemptRecord) -> dict[str, Any]:
        """Return an ``evaluations.jsonl`` row for one accepted bundle."""
        judgments = {
            component.component_id: {self.spec.judge_type: component.judgment}
            for component in record.components
        }
        row = make_evaluation_record(
            self.scenarios[record.scenario_id], {self.model_name: judgments}
        )
        row["decoder_draw_id"] = record.decoder_draw_id
        row["semantic_attempt_id"] = record.semantic_attempt_id
        return row

    def paths(self, draw_dir: Path) -> dict[str, Path]:
        """The views :meth:`write` writes into ``draw_dir``, by name."""
        paths = {
            "bundle_outcomes": draw_dir / BUNDLE_OUTCOMES_FNAME,
            "evaluations": draw_dir / EVALUATIONS_FNAME,
            "judge_outputs": draw_dir / JUDGE_OUTPUTS_FNAME,
            "per_scenario": draw_dir / PER_SCENARIO_FNAME,
            "summary": draw_dir / SUMMARY_FNAME,
        }
        if self.schema.evaluation_mode != EVALUATION_MODE_BOTH:
            del paths["per_scenario"]
        return paths

    def write(self, state: LedgerState, draw_dir: Path, raw_rows: Iterable[Mapping[str, Any]]) -> dict[str, Path]:
        """Write every derived view of a complete draw into ``draw_dir``, in manifest order, after everything that can
        refuse."""
        ordered = [state.accepted_bundle(sid) for sid in self.spec.scope_ids()]
        traced_rows = list(self._traced_rows(state, ordered, raw_rows))
        paths = self.paths(draw_dir)
        write_jsonl_records(paths["bundle_outcomes"], (bundle_outcome(self.schema, r) for r in ordered))
        evaluation_rows = [self.evaluation_record(record) for record in ordered]
        write_jsonl_records(paths["evaluations"], evaluation_rows)
        write_implicated_sidecar(draw_dir, self.individual_rule_context, self.implication_map)
        write_jsonl_records(paths["judge_outputs"], traced_rows)
        write_draw_summary(
            draw_dir,
            evaluation_rows,
            model_name=self.model_name,
            judge_type=self.spec.judge_type,
            evaluation_mode=self.schema.evaluation_mode,
            decoder_draw_id=self.spec.decoder_draw_id,
            condition_id=self.spec.condition.condition_id,
            components_per_bundle=len(self.schema.component_ids),
        )
        return paths

    def _traced_rows(
        self, state: LedgerState, ordered: Sequence[AttemptRecord], raw_rows: Iterable[Mapping[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        # Prompts are regenerated and checked against recorded digests, so a trace shows only the prompt sent.
        raw_by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for row in raw_rows:
            raw_by_key[
                (row["scenario_id"], int(row["semantic_attempt_id"]),
                 row["request_block_id"], row["component_id"])
            ] = row
        history: dict[str, list[AttemptRecord]] = {}
        for record in state.records:
            history.setdefault(record.scenario_id, []).append(record)
        model = self.model_name
        for accepted in ordered:
            sid = accepted.scenario_id
            chosen = state.accepted_components(sid)
            prompts = dict(self.bundle_prompts(sid))
            outputs: dict[str, Any] = {}
            for component_id, entry in prompts.items():
                digest = prompt_sha256(entry)
                attempts = []
                selected = None
                judgment = None
                for number, record in enumerate(history[sid], start=1):
                    component = next((c for c in record.components if c.component_id == component_id), None)
                    if component is None:
                        continue
                    if component.prompt_sha256 != digest:
                        raise CampaignRunError(
                            f"prompt for {sid[:12]}/{component_id} no longer matches the "
                            "one recorded at generation time; refusing to write a trace "
                            "that misattributes the judge's answer"
                        )
                    raw = raw_by_key.get(
                        (sid, record.semantic_attempt_id, record.request_block_id, component_id)
                    )
                    accepted_here = chosen[component_id][0] is record
                    if accepted_here:
                        selected, judgment = number, component.judgment
                    attempts.append(
                        {
                            "attempt": number,
                            "seed": component.generation_seed,
                            "semantic_attempt_id": record.semantic_attempt_id,
                            "request_block_id": record.request_block_id,
                            "status": record.status,
                            "candidates": [] if raw is None else [
                                {
                                    "raw_response": raw["raw_response"],
                                    "completion": raw["completion"],
                                    "finish_reason": raw["finish_reason"],
                                    "accepted": accepted_here,
                                    **reasoning_fields(raw),
                                }
                            ],
                        }
                    )
                outputs[component_id] = {
                    self.spec.judge_type: {
                        "prompt": dict(entry),
                        "attempts": attempts,
                        "selected_attempt": selected,
                        "judgment": judgment,
                    }
                }
            row = make_evaluation_record(self.scenarios[sid], {})
            row.pop("evaluations")
            row["judge_outputs"] = {model: outputs}
            row["decoder_draw_id"] = accepted.decoder_draw_id
            yield row


class DrawRunner:
    """Run, resume and materialise one draw directory."""

    def __init__(
        self,
        *,
        backend: ModelBackend,
        task_spec: BaseTaskSpec,
        spec: DrawSpec,
        draw_dir: Path | str,
        scenario_rows: Sequence[Mapping[str, Any]],
        rules: Sequence[Mapping[str, Any]],
        rule_levels: Optional[Mapping[str, str]] = None,
        clock: Optional[Callable[[], str]] = None,
        code_revision: Optional[str] = None,
        code_dirty: Optional[bool] = None,
        code_state_sha256: Optional[str] = None,
        package_version: Optional[str] = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        allow_stack_change: bool = False,
    ) -> None:
        self.backend = backend
        self.allow_stack_change = bool(allow_stack_change)
        self._spend_confirmed = False
        self._stack_change: dict[str, list[Any]] = {}
        # The stack this run's rows are recorded as judged on; None for a provider judge.
        self._judged_on: Optional[str] = None
        self.task_spec = task_spec
        self.spec = spec
        self.draw_dir = Path(draw_dir)
        self.clock = clock if clock is not None else utc_now
        self._sleep = sleep
        self._monotonic = monotonic
        self.code_revision = code_revision
        self.code_dirty = code_dirty
        self.code_state_sha256 = code_state_sha256
        self.package_version = package_version
        config = spec.config

        if not backend.capabilities.per_request_errors:
            raise CampaignRunError(
                f"model_type {backend.model_type!r} cannot run a stochastic campaign: "
                "its generate path does not report a transport failure as one "
                "(capabilities.per_request_errors), so it would be charged to a "
                "semantic attempt as an empty answer."
            )
        refuse_fake_in_final(backend.model_type, config.campaign_mode)
        resolved = backend.resolved_temperature(config.temperature)
        if backend.model_type == "local" and not isinstance(
            spec.condition.sampling_truncation, (tuple, Mapping)
        ):
            raise CampaignRunError(
                f"condition {spec.condition.condition_id!r} is served by a local judge but "
                "its identity carries no resolved sampling truncation; build the plan "
                "through ruleforge.evaluate.campaign so the shipped truncation is recorded"
            )
        if resolved != config.temperature:
            raise CampaignRunError(
                f"{backend.model_name} would sample at temperature {resolved}, not the "
                f"campaign's {config.temperature}; fix the model config's fixed_temperature"
            )
        if backend.seed_capability != spec.condition.seed_capability:
            raise CampaignRunError(
                f"condition {spec.condition.condition_id!r} declares seed capability "
                f"{spec.condition.seed_capability!r} but the backend reports "
                f"{backend.seed_capability!r}"
            )
        served = self._served_adapter()
        unnamed = served is not None and spec.condition.adapter_sha256 is None
        unserved = served is None and spec.condition.adapter_path is not None
        if unnamed or unserved:
            raise CampaignRunError(
                f"condition {spec.condition.condition_id!r} names adapter "
                f"{spec.condition.adapter_path!r} (SHA-256 {spec.condition.adapter_sha256}) but "
                f"the backend serves {served}; build the plan through ruleforge.evaluate.campaign so "
                "the served adapter is the condition's, named by content"
            )
        self._check_served_weights()
        self._check_serving_dtype()
        declared = (spec.condition.retry_scope, spec.condition.strict_think)
        if declared != (config.retry_scope, config.strict_think):
            raise CampaignRunError(
                f"condition {spec.condition.condition_id!r} declares retry_scope {declared[0]!r} and strict_think "
                f"{declared[1]} but the campaign declares {config.retry_scope!r} and {config.strict_think}; build "
                "the plan through ruleforge.evaluate.campaign"
            )

        self.scenarios: dict[str, dict[str, Any]] = {}
        for row in scenario_rows:
            self.scenarios[compute_scenario_id(row)] = dict(row)
        manifest_ids = set(spec.scenario_manifest.scenario_ids)
        if set(self.scenarios) != manifest_ids or len(self.scenarios) != len(scenario_rows):
            raise CampaignRunError(
                "the scenario rows handed to the runner are not the frozen manifest's "
                f"population ({len(self.scenarios)} distinct rows, {len(manifest_ids)} "
                "manifest ids)"
            )
        self.scope_ids = spec.scope_ids()

        self.rules = [dict(rule) for rule in rules]
        self.rule_set_string = rule_set_text(self.rules, task_spec.individual_rule_context, rule_levels)
        variants, self.implication_map = build_rule_variants(
            self.rules, task_spec.individual_rule_context, rule_levels
        )
        self.variants: dict[str, tuple[str, Optional[list[str]]]] = {
            rule_id: (text, context) for rule_id, text, context in variants
        }
        self._check_task_identity(task_spec)
        self.schema = ComponentSchema.for_condition(
            spec.condition, self.rules, forced_json_requirement=config.forced_json_requirement
        )
        if set(self.variants) != set(self.schema.expected_rule_ids):  # pragma: no cover - both derive from the same rule list
            raise CampaignRunError("rule variants and validation schema disagree")
        self.send_seeds = backend.seed_capability != SEED_UNSUPPORTED
        # Transport failures seen by this process: a fresh run starts with a fresh budget; the ledger keeps the
        # cumulative count.
        self._transport_retries: dict[str, int] = {}
        self.ledger = DrawLedger(
            self.draw_dir,
            campaign_id=spec.campaign_id,
            condition_id=spec.condition.condition_id,
            decoder_draw_id=spec.decoder_draw_id,
        )


    def component_prompt(self, scenario_id: str, component_id: str) -> dict[str, str]:
        """The prompt of one component of one scenario's bundle."""
        if component_id == FULL_RULESET_COMPONENT:
            rule_text, context = self.rule_set_string, None
        else:
            rule_text, context = self.variants[component_id]
        system_prompt = self.task_spec.build_eval_prompt(
            rule_text=rule_text,
            scenario=self.scenarios[scenario_id],
            judge_type=self.spec.judge_type,
            context_rules=context,
        )
        return {"system_prompt": system_prompt, "user_prompt": ""}

    def bundle_prompts(self, scenario_id: str) -> list[tuple[str, dict[str, str]]]:
        """Every component prompt of one scenario, full set first."""
        return [
            (component_id, self.component_prompt(scenario_id, component_id))
            for component_id in self.schema.component_ids
        ]

    def component_seed(self, scenario_id: str, attempt: int, component_id: str) -> int:
        """The seed one component is sent: derived over the condition's identity id (``ConditionSpec.identity_id``)."""
        return derive_generation_seed(
            generation_seed_base=self.spec.config.generation_seed_base,
            campaign_id=self.spec.campaign_id,
            condition_id=self.spec.condition.identity_id,
            scenario_id=scenario_id,
            decoder_draw_id=self.spec.decoder_draw_id,
            semantic_attempt_id=attempt,
            component_id=component_id,
        )

    def member_components(self, work: BlockWork) -> tuple[tuple[str, ...], ...]:
        """The components each member of ``work`` submits, aligned with its members."""
        if work.components is None:
            return (self.schema.component_ids,) * len(work.members)
        return work.components

    def prompt_count(self, work: BlockWork) -> int:
        """How many prompts ``work`` submits, without building them."""
        return sum(len(components) for components in self.member_components(work))

    def build_request(self, work: BlockWork) -> BlockRequest:
        """Every component prompt of a block, paired with its planned record and seed."""
        prompts, planned = self._build_block_prompts(
            work.members, work.semantic_attempt_id, self.member_components(work)
        )
        return BlockRequest(
            work=work,
            prompts=tuple(prompts),
            planned=tuple(planned),
            seeds=tuple(item.seed for item in planned) if self.send_seeds else None,
        )


    def sampling_truncation(self) -> Any:
        """Return the condition's truncation as a plain dict (local), a string (provider) or None."""
        value = self.spec.condition.sampling_truncation
        if isinstance(value, (list, tuple)):
            return {str(key): item for key, item in value}
        return value

    def sampler_truncation(self) -> Optional[dict[str, Any]]:
        """What ``generate_once`` is handed: a local judge's mapping, else None."""
        truncation = self.sampling_truncation()
        return truncation if isinstance(truncation, dict) else None

    def _check_task_identity(self, task_spec: BaseTaskSpec) -> None:
        condition = self.spec.condition
        try:
            rendered = task_identity_of(task_spec, self.spec.judge_type)
        except TaskError as error:
            raise CampaignRunError(f"condition {condition.condition_id!r}: {error}") from error
        differ = [key for key in rendered if getattr(condition, key) != rendered[key]]
        if differ:
            raise CampaignRunError(
                f"condition {condition.condition_id!r} records "
                + ", ".join(f"{key} {getattr(condition, key)!r}" for key in differ)
                + " but the task spec renders "
                + ", ".join(f"{key} {rendered[key]!r}" for key in differ)
                + "; build the plan through ruleforge.evaluate.campaign"
            )

    def _served_adapter(self) -> Optional[Path]:
        if self.backend.model_type != "local":
            return None
        return getattr(self.backend, "adapter_path", None)

    def adapter_provenance(self) -> Optional[dict[str, Any]]:
        """Return the served adapter's provenance, re-hashed now and refused unless it is the condition's."""
        served = self._served_adapter()
        if served is None:
            return None
        identity = adapter_identity(served)
        condition = self.spec.condition
        if (identity.sha256, identity.config_sha256) != (condition.adapter_sha256, condition.adapter_config_sha256):
            raise CampaignRunError(
                f"the backend serves adapter {identity.path} (SHA-256 {identity.sha256}) but "
                f"condition {condition.condition_id!r} names {condition.adapter_path} "
                f"(SHA-256 {condition.adapter_sha256}); refusing to add judgments to it"
            )
        return {
            "path": identity.path,
            "sha256": identity.sha256,
            "config_sha256": identity.config_sha256,
            "rank": identity.rank,
            "base_model": identity.base_model,
            "backend_metadata": getattr(self.backend, "adapter_metadata", None),
        }

    def _check_served_weights(self) -> None:
        served = served_model_weights(self.backend)
        condition = self.spec.condition
        named = condition.model_weights_sha256
        if served is not None:
            if served.sha256 == named:
                return
            what = f"{served.path} (weights SHA-256 {served.sha256})"
        else:
            path = getattr(self.backend, "model_path", None)
            local = self.backend.model_type == LOCAL_MODEL_TYPE and path is not None
            if named is None and (not local or Path(path).name == condition.model_path):
                return
            what = "no derived weights" if named is not None else f"the upstream weights directory {path}"
        raise CampaignRunError(
            f"condition {condition.condition_id!r} names model weights {condition.model_path!r} (weights SHA-256 "
            f"{named}) but the backend serves {what}; refusing to add judgments to it. Build the plan through "
            "ruleforge.evaluate.campaign so the served weights are the condition's, named by content"
        )

    def _check_serving_dtype(self) -> None:
        # An fp32 checkpoint under ``auto`` would be served in bf16 with no error.
        condition = self.spec.condition
        declared = condition.serving_dtype
        requested = canonical_serving_dtype((getattr(self.backend, "engine_args", None) or {}).get("dtype"))
        if requested != declared:
            raise CampaignRunError(
                f"condition {condition.condition_id!r} declares serving dtype {declared or 'auto'} but the backend's "
                f"engine was built for {requested or 'auto'}; build the plan through ruleforge.evaluate.campaign"
            )
        if declared is None:
            return
        resolved = getattr(self.backend, "engine_resolved", None) or {}
        found = (resolved.get("engine_dtype"), resolved.get("kv_cache_dtype"))
        if found != (declared, declared):
            raise CampaignRunError(
                f"condition {condition.condition_id!r} declares serving dtype {declared}, but the engine resolved "
                f"{found[0]} for the model and {found[1]} for its KV cache; refusing to add judgments to it"
            )

    def model_weights_provenance(self) -> Optional[dict[str, Any]]:
        """The served derived weights (checked when the runner started), or None: where from, and their manifest."""
        served = served_model_weights(self.backend)
        if served is None:
            return None
        return {
            "path": served.path,
            "sha256": served.sha256,
            "digest_version": served.digest_version,
            "manifest_sha256": served.manifest_sha256,
            "kind": served.kind,
            "saved_dtype": served.saved_dtype,
        }

    def provenance(self) -> dict[str, Any]:
        adapter = self.adapter_provenance()
        weights = self.model_weights_provenance()
        return {
            "schema_version": SAMPLING_SCHEMA_VERSION,
            "campaign_id": self.spec.campaign_id,
            "condition_id": self.spec.condition.condition_id,
            "condition_fingerprint": self.spec.condition.fingerprint,
            "decoder_draw_id": self.spec.decoder_draw_id,
            "scenario_manifest_sha256": self.spec.scenario_manifest.manifest_sha256,
            "expected_scenarios": self.spec.scenario_manifest.record_count,
            "component_ids": list(self.schema.component_ids),
            "judge_type": self.spec.judge_type,
            "temperature": self.spec.config.temperature,
            "resolved_temperature": self.backend.resolved_temperature(
                self.spec.config.temperature
            ),
            "seed_capability": self.backend.seed_capability,
            "seeds_sent": self.send_seeds,
            "model_name": self.backend.model_name,
            "model_type": self.backend.model_type,
            "sampling_truncation": self.sampling_truncation(),
            "tp_size": getattr(self.backend, "tp_size", None),
            "attention_backend": getattr(self.backend, "attention_backend", None),
            "code_revision": self.code_revision,
            "code_dirty": self.code_dirty,
            "code_state_sha256": self.code_state_sha256,
            **({"package_version": self.package_version} if self.package_version is not None else {}),
            "runtime": self._runtime(),
            **({"adapter": adapter} if adapter is not None else {}),
            **({"model_weights": weights} if weights is not None else {}),
            **self._request_settings_fields(),
        }

    def _request_settings_fields(self) -> dict[str, Any]:
        if not self.backend.capabilities.batch:
            return {}
        return {REQUEST_SETTINGS_KEY: dict(self.backend.spec.request_settings())}

    def _request_settings_guard(self, state: LedgerState) -> None:
        """Refuse to re-submit a recorded block under other request settings: its requests would address, and buy,
        another batch."""
        current = self._request_settings_fields().get(REQUEST_SETTINGS_KEY)
        if current is None or not self._pending_count(state):
            return
        unrecorded = UNRECORDED_REQUEST_SETTINGS.get(self.backend.model_type)
        for row in self._recorded_blocks(repair=False):
            block_id = str(row["request_block_id"])
            if all((str(sid), block_id) in state.vouched for sid in row["scenario_ids"]):
                continue
            recorded = row.get(REQUEST_SETTINGS_KEY, unrecorded)
            if recorded is not None and recorded != current:
                raise CampaignRunError(
                    f"{self.draw_dir}: request block {block_id} was recorded with request settings {recorded} and "
                    f"has scenarios its ledger does not vouch for, so its batch may be bought; this judge sends "
                    f"{current}, whose requests would address and buy another batch. Finish the block with the "
                    "settings it was recorded with, then change them"
                )

    def _runtime(self) -> dict[str, Any]:
        runtime = runtime_provenance(local_engine=self.backend.model_type == LOCAL_MODEL_TYPE)
        engine_args = getattr(self.backend, "engine_args", None)
        if engine_args is not None:
            runtime[ENGINE_ARGS_KEY] = json.loads(json.dumps(engine_args, sort_keys=True, default=str))
        resolved = getattr(self.backend, "engine_resolved", None)
        if resolved is not None:
            runtime[RESOLVED_ENGINE_KEY] = dict(resolved)
        return runtime

    def _append_run_row(self, row: Mapping[str, Any]) -> None:
        # Durable before anything that follows it, so a crash never keeps the gate or ledger rows without it.
        with (self.draw_dir / RUNS_FNAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _record_run(self, state: LedgerState, execution: Optional[Mapping[str, Any]] = None) -> None:
        row = {
            "started_at": self.clock(),
            "code_revision": self.code_revision,
            "code_dirty": self.code_dirty,
            "code_state_sha256": self.code_state_sha256,
            **({"package_version": self.package_version} if self.package_version is not None else {}),
            "max_semantic_attempts": self.spec.config.max_semantic_attempts,
            "runtime": self._runtime(),
            LEDGER_ROWS_KEY: len(state.records),
        }
        if execution is not None:
            row["execution"] = dict(execution)
        if self.spec.shard is not None:
            row["shard"] = self.spec.shard.to_dict()
        self._judged_on = stack_digest(row["runtime"]) if self.backend.model_type == LOCAL_MODEL_TYPE else None
        self._append_run_row(row)

    def _log_stack_change(self) -> None:
        # Logged before the first attempt record: a row logged at run start could name a stack that judged nothing.
        if not self._stack_change:
            return
        self._append_run_row({
            "recorded_at": self.clock(),
            "runtime": self._runtime(),
            STACK_CHANGE_KEY: self._stack_change,
        })
        self._stack_change = {}

    @property
    def condition_dir(self) -> Path:
        """The condition's directory: every draw directory sits directly in it."""
        return self.draw_dir.parent

    def _stack_guard(self, state: LedgerState) -> dict[str, list[Any]]:
        # Only a local judge with pending work is checked. Returns the change when ``execution.allow_stack_change``
        # allows it.
        pending = self._pending_count(state)
        if self.backend.model_type != LOCAL_MODEL_TYPE or not pending:
            return {}
        current = self._runtime()
        if state.records:
            changed = runtime_changes(recorded_runtime(self.draw_dir), current)
        else:
            record = condition_stack(self.condition_dir)
            changed = runtime_changes(record["runtime"], current) if record else {}
        if not changed:
            return {}
        if not self.allow_stack_change:
            if state.records:
                message = (
                    f"{self.draw_dir} holds {len(state.records)} attempt(s) recorded on another "
                    f"inference stack or engine ({describe_changes(changed)}) and {pending} pending bundle(s); "
                    "refusing to add attempts that would mix two stacks' judgments in one draw. Finish "
                    "it on the stack it was recorded on"
                )
            else:
                message = (
                    f"condition {self.spec.condition.condition_id!r} holds judgments recorded on "
                    f"another inference stack or engine ({describe_changes(changed)}; from {record['source']}); "
                    f"refusing to start its draw {self.spec.decoder_draw_id} on this one, which would "
                    "pool two stacks' draws in one condition. Run it on the condition's stack"
                )
            raise CampaignRunError(
                f"{message}, start a new campaign root, or set execution.allow_stack_change=true "
                "to add them anyway (logged in runs.jsonl)"
            )
        logger.warning("%s: adding attempts on another stack (%s), as allowed", self.draw_dir, describe_changes(changed))
        return {name: list(pair) for name, pair in changed.items()}

    def _ensure_provenance(self, state: LedgerState) -> None:
        path = self.draw_dir / DRAW_PROVENANCE_FNAME
        current = self.provenance()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            differing = sorted(
                key
                for key in set(existing) | set(current)
                if key not in PROVENANCE_RUN_FIELDS and existing.get(key) != current.get(key)
            )
            if differing:
                raise CampaignRunError(
                    f"{path} was written for a different draw or condition "
                    f"({', '.join(differing)} differ); refusing to add attempts to it"
                )
            if state.records or not runtime_changes(existing.get("runtime") or {}, current["runtime"]):
                return
            logger.info("%s: no attempt recorded yet; provenance re-stamped on this run's stack", path)
        write_json_sidecar(path, current)

    def _record_condition_stack(self) -> None:
        if self.backend.model_type != LOCAL_MODEL_TYPE:
            return
        record = read_condition_stack(self.condition_dir)
        judged = condition_stack(self.condition_dir)
        if judged is not None:
            if record is None:
                write_condition_stack(
                    self.condition_dir, runtime=judged["runtime"], source=judged["source"],
                    condition_id=self.spec.condition.condition_id,
                    condition_fingerprint=self.spec.condition.fingerprint, recorded_at=self.clock(),
                )
            return
        current = self._runtime()
        if record is not None and not runtime_changes(record.get("runtime") or {}, current):
            return
        write_condition_stack(
            self.condition_dir, runtime=current, source=f"{self.draw_dir.name}: first run",
            condition_id=self.spec.condition.condition_id,
            condition_fingerprint=self.spec.condition.fingerprint, recorded_at=self.clock(),
        )


    def run(self) -> dict[str, Any]:
        """Bring the draw to a terminal state under its run lock and return its completion record."""
        with draw_run_lock(self.draw_dir):
            state = self.prepare()
            for attempt in range(self.spec.config.max_semantic_attempts):
                self._run_attempt_pass(state, attempt)
            return self.finish(state)

    def prepare(self, *, execution: Optional[Mapping[str, Any]] = None) -> LedgerState:
        """Load and repair the ledger, check it and stamp this run, before any generation."""
        check_run_write(self.draw_dir)
        state = self.ledger.load(repair=True)
        self._refuse_foreign_scenarios(state)
        damage = run_log_damage(self.draw_dir)
        if damage and not damage[1]:
            raise CampaignRunError(f"{self.draw_dir}: {damage[0]}; refusing to log a run after it. Repair the run log "
                                   "by hand")
        cut = run_log_problem(self.draw_dir) or unlogged_stacks(self.draw_dir, state.records)
        if cut:
            # A run gone from the log, whose attempts would be read as another run's; logging after it would hide that.
            raise CampaignRunError(f"{self.draw_dir}: {cut}; refusing to log a run beside it. Restore the run log")
        if damage:
            # A torn trailing row: appending behind it would fuse this run's row with it.
            logger.warning("%s: %s; repaired before this run logs itself", self.draw_dir, damage[0])
            count_complete_jsonl_records(self.draw_dir / RUNS_FNAME, repair=True)
        self._stack_change = self._stack_guard(state)
        self._request_settings_guard(state)
        self._ensure_provenance(state)
        with shared_write():
            self._record_condition_stack()
        self._record_run(state, execution)
        self._write_completion(state)
        return state

    def finish(self, state: LedgerState) -> dict[str, Any]:
        """Materialise a complete draw, then write and return its gate."""
        check_run_write(self.draw_dir)
        if self.completion(state, materialized=True)["state"] == STATE_COMPLETE:
            # Views first, gate last: a digest check that fails in materialize() must leave the gate in_progress.
            self.materialize(state)
        return self._write_completion(state)

    def _refuse_foreign_scenarios(self, state: LedgerState) -> None:
        # Such a draw would spend inference and never gate complete.
        manifest_ids = set(self.spec.scenario_manifest.scenario_ids)
        foreign = {record.scenario_id for record in state.records} - manifest_ids
        for block in self._recorded_blocks(repair=False):
            foreign.update(str(sid) for sid in block["scenario_ids"] if sid not in manifest_ids)
        if foreign:
            raise CampaignRunError(
                f"{self.draw_dir} holds ledger rows or request blocks for {len(foreign)} "
                f"scenario(s) outside the frozen manifest (first: {sorted(foreign)[0][:12]}); "
                "it was run under another population. Resume it under the population it froze, or create a new "
                "campaign root for the revised population and run it there"
            )
        outside = {record.scenario_id for record in state.records} - set(self.scope_ids)
        if outside:
            raise CampaignRunError(
                f"{self.draw_dir} holds ledger rows for {len(outside)} scenario(s) outside "
                f"{self.spec.shard.label}'s request blocks {list(self.spec.shard.blocks)} (first: "
                f"{sorted(outside)[0][:12]}); a shard root holds its own units only, so it was written by "
                "another shard layout. Run each layout in its own root (ruleforge.evaluate.sharding.shard_root)"
            )

    def _pending(self, state: LedgerState, attempt: int) -> list[str]:
        return [
            sid
            for sid in self.scope_ids
            if not state.is_accepted(sid) and state.next_attempt(sid) == attempt < self.spec.attempt_cap(sid)
        ]

    def plan_pass(self, state: LedgerState, attempt: int) -> list[BlockWork]:
        """Return the blocks still to answer at ``attempt``, recorded first; refuses when a bundle's transport retries
        run out."""
        pending = self._pending(state, attempt)
        if not pending:
            return []
        stuck = [
            sid
            for sid in pending
            if self._transport_retries.get(sid, 0) >= self.spec.max_transport_retries
        ]
        if stuck:
            raise TransportExhausted(
                f"{len(stuck)} bundle(s) failed in transport "
                f"{self.spec.max_transport_retries} time(s) in this run "
                f"(first: {stuck[0][:12]}); the draw stays resumable, fix the "
                "provider and rerun"
            )
        return self._plan_blocks(state, attempt, pending)

    def _run_attempt_pass(self, state: LedgerState, attempt: int) -> None:
        while True:
            plan = self.plan_pass(state, attempt)
            if not plan:
                return
            if self._windowed():
                # A batch judge with an enqueued-token budget: only the submission schedule differs.
                self._drive_window(state, plan)
            else:
                for work in plan:
                    # Re-submitted under the same id, re-entering a paid batch, until the cap says the provider is down.
                    failures = 0
                    while not self._run_block(state, work):
                        failures += 1
                        if failures >= self.spec.max_transport_retries:
                            raise TransportExhausted(
                                f"request block {work.block_id} raised {failures} time(s) in a "
                                "row; the draw stays resumable, fix the provider and rerun"
                            )

    def _windowed(self) -> bool:
        # The gate is the private ``_poll_request_block``: every backend carries the public ``poll_block``.
        return getattr(self.backend, "max_enqueued_tokens", None) is not None and callable(
            getattr(self.backend, "_poll_request_block", None)
        )

    def _window_poll_seconds(self) -> float:
        return float(getattr(self.backend, "batch_poll_seconds", DEFAULT_POLL_SECONDS))

    def _drive_window(self, state: LedgerState, plan: Sequence[BlockWork]) -> None:
        # Submits while the enqueued-token estimate leaves room (always at least one block), polls without blocking on
        # any one, and records each as it completes.
        budget = self.backend.max_enqueued_tokens
        timeout = getattr(self.backend, "batch_timeout_seconds", None)
        deadline = None if timeout is None else self._monotonic() + timeout
        todo: list[_WindowBlock] = []
        for work in plan:
            request = self.build_request(work)
            est = self.backend.estimate_block_tokens(
                list(request.prompts),
                temperature=self.spec.config.temperature,
                seeds=request.seed_list(),
                request_block_id=self._identity_block(work.block_id),
            )
            todo.append(_WindowBlock(request, est))
        in_flight: list[_WindowBlock] = []
        enqueued = 0
        submit_failures: dict[str, int] = {}
        poll_failures: dict[str, int] = {}
        while todo or in_flight:
            submitted_any = False
            for block in list(todo):
                if enqueued + block.est_tokens <= budget or not in_flight:
                    if self._submit_window_block(block, submit_failures):
                        block.started_at = self.clock()
                        enqueued += block.est_tokens
                        in_flight.append(block)
                        todo.remove(block)
                        submitted_any = True
            completed_any = False
            for block in list(in_flight):
                outcomes = self._poll_window_block(block, poll_failures)
                if outcomes is None:
                    continue
                self.record(
                    state, block.request, outcomes,
                    started_at=block.started_at, completed_at=self.clock(),
                )
                in_flight.remove(block)
                enqueued -= block.est_tokens
                completed_any = True
            if (todo or in_flight) and not submitted_any and not completed_any:
                self._check_window_deadline(deadline, timeout, in_flight)
                self._sleep(self._window_poll_seconds())

    def _check_window_deadline(
        self, deadline: Optional[float], timeout: Optional[float], in_flight: Sequence[_WindowBlock]
    ) -> None:
        if deadline is not None and self._monotonic() >= deadline:
            raise BatchExecutionError(
                f"windowed batch did not finish within {timeout}s; "
                f"{len(in_flight)} block(s) are still running and resumable. "
                "Rerun to resume -- the work is already paid for."
            )

    def _poll_window_block(
        self, block: _WindowBlock, poll_failures: dict[str, int]
    ) -> Optional[list[GenerationOutcome]]:
        try:
            outcomes = self.backend.poll_block(
                list(block.request.prompts),
                temperature=self.spec.config.temperature,
                seeds=block.request.seed_list(),
                request_block_id=self._identity_block(block.block_id),
            )
        except FATAL_PROVIDER_ERRORS:
            raise
        except Exception as exc:
            self._window_transport_failure(poll_failures, block.block_id, "poll", exc)
            return None
        if outcomes is not None:
            poll_failures.pop(block.block_id, None)
        return outcomes

    def _submit_window_block(
        self, block: _WindowBlock, submit_failures: dict[str, int]
    ) -> bool:
        try:
            self.backend.submit_block(
                list(block.request.prompts),
                temperature=self.spec.config.temperature,
                seeds=block.request.seed_list(),
                request_block_id=self._identity_block(block.block_id),
            )
        except FATAL_PROVIDER_ERRORS:
            raise
        except Exception as exc:
            self._window_transport_failure(submit_failures, block.block_id, "submit", exc)
            return False
        submit_failures.pop(block.block_id, None)
        return True

    def _window_transport_failure(
        self, failures: dict[str, int], block_id: str, verb: str, exc: Exception
    ) -> None:
        count = failures.get(block_id, 0) + 1
        failures[block_id] = count
        logger.warning(
            "Windowed %s of block %s failed in transport (%d): %s",
            verb, block_id, count, exc,
        )
        if count >= self.spec.max_transport_retries:
            raise TransportExhausted(
                f"request block {block_id} failed to {verb} {count} time(s) "
                "in a row; the draw stays resumable, fix the provider and rerun"
            ) from exc


    @property
    def blocks_path(self) -> Path:
        return self.draw_dir / REQUEST_BLOCKS_FNAME

    def _recorded_blocks(self, *, repair: bool = True) -> list[dict[str, Any]]:
        n_rows = count_complete_jsonl_records(self.blocks_path, repair=repair)
        if not n_rows:
            return []
        with self.blocks_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for _, line in zip(range(n_rows), handle)]

    def _record_block(
        self, block_id: str, block_index: int, attempt: int, members: Sequence[str]
    ) -> None:
        self.draw_dir.mkdir(parents=True, exist_ok=True)
        with self.blocks_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "request_block_id": block_id,
                        "block_index": block_index,
                        "semantic_attempt_id": attempt,
                        "scenario_ids": list(members),
                        "recorded_at": self.clock(),
                        **self._request_settings_fields(),
                    }
                )
                + "\n"
            )
            handle.flush()

    def _plan_blocks(
        self, state: LedgerState, attempt: int, pending: Sequence[str]
    ) -> list[BlockWork]:
        # Recorded blocks keep their membership and order, which re-addresses a paid batch; new scenarios get new block
        # indices.
        check_run_write(self.draw_dir)
        recorded = self._recorded_blocks()
        pending_set = set(pending)
        plans: list[BlockWork] = []
        taken: set[str] = set()
        next_index = 0
        shard = self.spec.shard
        for record in recorded:
            if record["semantic_attempt_id"] != attempt:
                continue
            next_index = max(next_index, int(record["block_index"]) + 1)
            block_id = str(record["request_block_id"])
            members = [str(sid) for sid in record["scenario_ids"]]
            missing = [sid for sid in members if (sid, block_id) not in state.vouched]
            if not missing:
                continue
            if not set(missing) <= pending_set or set(missing) & taken:
                raise CampaignRunError(
                    f"recorded block {block_id} names scenarios that are no longer "
                    "pending at this attempt; the ledger and the block record disagree"
                )
            # Whole membership, so a paid batch's address is the recorded one; only the un-vouched bundles are recorded.
            plans.append(BlockWork(
                block_id, attempt, tuple(members), tuple(missing), self._block_components(state, members, attempt)
            ))
            taken.update(missing)
        remaining = [sid for sid in pending if sid not in taken]
        if shard is not None and attempt == 0:
            for block_index, members in self._unplanned_units(recorded, set(remaining)):
                block_id = self._block_id(attempt, block_index)
                self._record_block(block_id, block_index, attempt, members)
                plans.append(BlockWork(
                    block_id, attempt, tuple(members), tuple(members), self._block_components(state, members, attempt)
                ))
                taken.update(members)
                next_index = max(next_index, block_index + 1)
            remaining = [sid for sid in remaining if sid not in taken]
        size = self.spec.config.scenarios_per_request_block or max(len(remaining), 1)
        for start in range(0, len(remaining), size):
            members = remaining[start : start + size]
            block_id = self._block_id(attempt, next_index)
            if shard is not None:
                # Pools only this shard's scenarios; the merge re-plans it.
                block_id = shard_block_id(block_id, shard.index, shard.count)
            self._record_block(block_id, next_index, attempt, members)
            plans.append(BlockWork(
                block_id, attempt, tuple(members), tuple(members), self._block_components(state, members, attempt)
            ))
            next_index += 1
        return plans

    def _identity_block(self, block_id: str) -> str:
        return self.spec.condition.identity_block_id(self.spec.campaign_id, block_id)

    def _block_id(self, attempt: int, block_index: int) -> str:
        return request_block_id(
            campaign_id=self.spec.campaign_id,
            condition_id=self.spec.condition.condition_id,
            decoder_draw_id=self.spec.decoder_draw_id,
            semantic_attempt_id=attempt,
            block_index=block_index,
        )

    def _unplanned_units(
        self, recorded: Sequence[Mapping[str, Any]], remaining: set[str]
    ) -> list[tuple[int, list[str]]]:
        planned = {
            int(record["block_index"])
            for record in recorded
            if record["semantic_attempt_id"] == 0
            and record["request_block_id"] == self._block_id(0, int(record["block_index"]))
        }
        per_block = self.spec.config.scenarios_per_request_block
        ids = self.spec.scenario_manifest.scenario_ids
        units = [
            (block, [sid for sid in block_members(ids, per_block, block) if sid in remaining])
            for block in self.spec.shard.blocks
            if block not in planned
        ]
        return [(block, members) for block, members in units if members]

    def _block_components(
        self, state: LedgerState, members: Sequence[str], attempt: int
    ) -> Optional[tuple[tuple[str, ...], ...]]:
        if self.spec.config.retry_scope != RETRY_SCOPE_COMPONENT:
            return None
        return tuple(state.components_at(sid, attempt, self.schema.component_ids) for sid in members)

    def _confirm_spend(self) -> None:
        capabilities = self.backend.capabilities
        if capabilities.spends_money and not capabilities.batch and not self._spend_confirmed:
            require_spend_confirmation()
            self._spend_confirmed = True

    def _run_block(self, state: LedgerState, work: BlockWork) -> bool:
        started_at = self.clock()
        request = self.build_request(work)
        self._confirm_spend()
        outcomes = self.backend.generate_once(
            list(request.prompts),
            temperature=self.spec.config.temperature,
            seeds=request.seed_list(),
            request_block_id=self._identity_block(work.block_id),
            truncation=self.sampler_truncation(),
        )
        completed_at = self.clock()
        if all(outcome.block_failed for outcome in outcomes):
            logger.warning(
                "Request block %s raised before any result: %s", work.block_id, outcomes[0].error
            )
            return False
        self.record(state, request, outcomes, started_at=started_at, completed_at=completed_at)
        return True

    def record(
        self,
        state: LedgerState,
        request: BlockRequest,
        outcomes: Sequence[GenerationOutcome],
        *,
        started_at: str,
        completed_at: str,
    ) -> None:
        """Record one answered block: its ledger rows, then the draw's gate; a call that raised whole is the caller's to
        retry."""
        if len(outcomes) != len(request.prompts):
            raise CampaignRunError(
                f"request block {request.work.block_id}: {len(outcomes)} outcomes for "
                f"{len(request.prompts)} prompts"
            )
        check_run_write(self.draw_dir)
        work = request.work
        self._record_outcomes(
            state, work.semantic_attempt_id, work.block_id, work.members, work.record_for,
            self.member_components(work), request.planned, outcomes, started_at, completed_at,
        )

    def _build_block_prompts(
        self, members: Sequence[str], attempt: int, member_components: Sequence[Sequence[str]]
    ) -> tuple[list[dict[str, str]], list[_PlannedComponent]]:
        prompts: list[dict[str, str]] = []
        planned: list[_PlannedComponent] = []
        for sid, component_ids in zip(members, member_components):
            for component_id in component_ids:
                entry = self.component_prompt(sid, component_id)
                prompts.append(entry)
                planned.append(
                    _PlannedComponent(
                        sid, component_id,
                        self.component_seed(sid, attempt, component_id),
                        prompt_sha256(entry),
                    )
                )
        return prompts, planned

    def _record_outcomes(
        self,
        state: LedgerState,
        attempt: int,
        block_id: str,
        members: Sequence[str],
        record_for: Sequence[str],
        member_components: Sequence[Sequence[str]],
        planned: Sequence[_PlannedComponent],
        outcomes: Sequence[GenerationOutcome],
        started_at: str,
        completed_at: str,
    ) -> None:
        wanted = set(record_for)
        records: list[AttemptRecord] = []
        raw_rows: list[dict[str, Any]] = []
        position = 0
        for sid, component_ids in zip(members, member_components):
            if sid not in wanted:
                position += len(component_ids)
                continue
            components: list[ComponentRecord] = []
            for component_id in component_ids:
                item, outcome = planned[position], outcomes[position]
                position += 1
                verdict = self.schema.classify(component_id, outcome)
                if outcome.returned:
                    raw_rows.append(
                        {
                            "scenario_id": sid,
                            "decoder_draw_id": self.spec.decoder_draw_id,
                            "semantic_attempt_id": attempt,
                            "request_block_id": block_id,
                            "component_id": component_id,
                            "raw_response": outcome.raw_response,
                            "completion": outcome.text,
                            "finish_reason": outcome.finish_reason,
                            "error": outcome.error,
                            "raw_output_sha256": raw_output_sha256(outcome.raw_response),
                            **({REASONING_SUMMARIES_KEY: list(outcome.reasoning_summaries)}
                               if outcome.reasoning_summaries is not None else {}),
                            **({REASONING_KEY: outcome.reasoning} if outcome.reasoning is not None else {}),
                        }
                    )
                components.append(
                    ComponentRecord(
                        component_id=component_id,
                        valid=verdict.valid,
                        failure_reason=verdict.failure_reason,
                        judgment=verdict.judgment,
                        finish_reason=outcome.finish_reason,
                        generation_seed=item.seed if self.send_seeds else None,
                        seed_sent=self.send_seeds,
                        raw_output_sha256=raw_output_sha256(outcome.raw_response),
                        provider_request_ref=outcome.provider_request_ref,
                        request_block_ref=outcome.request_block_ref,
                        rule_judgments=verdict.rule_judgments,
                        preempted_rules=verdict.preempted_rules,
                        prompt_sha256=item.prompt_sha256,
                    )
                )
            status, failure = bundle_status(components)
            records.append(
                AttemptRecord(
                    campaign_id=self.spec.campaign_id,
                    condition_id=self.spec.condition.condition_id,
                    scenario_id=sid,
                    decoder_draw_id=self.spec.decoder_draw_id,
                    semantic_attempt_id=attempt,
                    request_block_id=block_id,
                    status=status,
                    seed_capability=self.backend.seed_capability,
                    components=tuple(components),
                    started_at=started_at,
                    completed_at=completed_at,
                    failure_reason=failure,
                    retry_scope=self.spec.config.retry_scope,
                    judged_on=self._judged_on,
                )
            )
        self._log_stack_change()
        self.ledger.append(records, raw_rows)
        for record in records:
            state.add(record, where=block_id)
            if record.status == STATUS_TRANSPORT_FAILED:
                self._transport_retries[record.scenario_id] = (
                    self._transport_retries.get(record.scenario_id, 0) + 1
                )
        counts = {status: sum(1 for r in records if r.status == status) for status in
                  (STATUS_ACCEPTED, STATUS_REJECTED, STATUS_TRANSPORT_FAILED)}
        logger.info("Block %s: %s", block_id, counts)
        self._write_completion(state)


    def _pending_count(self, state: LedgerState) -> int:
        return pending_count(state, self.spec, self.scope_ids)

    def completion(
        self, state: LedgerState, *, materialized: Optional[bool] = None
    ) -> dict[str, Any]:
        """Return the strict gate for this draw; ``complete`` also requires every derived view."""
        if materialized is None:
            materialized = all(path.exists() for path in self.views().paths(self.draw_dir).values())
        return draw_completion(
            state,
            spec=self.spec,
            scope_ids=self.scope_ids,
            components_per_bundle=len(self.schema.component_ids),
            materialized=materialized,
            updated_at=self.clock(),
            runs_logged=run_log_rows(self.draw_dir),
        )

    def _write_completion(self, state: LedgerState) -> dict[str, Any]:
        payload = self.completion(state)
        write_json_sidecar(self.draw_dir / COMPLETION_FNAME, payload)
        return payload


    def views(self) -> DrawViews:
        """This draw's view writer (:class:`DrawViews`), over the runner's prompts, scenarios and schema."""
        return DrawViews(
            spec=self.spec,
            schema=self.schema,
            model_name=self.backend.model_name,
            scenarios=self.scenarios,
            bundle_prompts=self.bundle_prompts,
            individual_rule_context=self.task_spec.individual_rule_context,
            implication_map=self.implication_map,
        )

    def materialize(self, state: LedgerState) -> dict[str, Path]:
        """Write every derived view of a complete draw, in manifest order (:meth:`DrawViews.write`)."""
        if self.completion(state, materialized=True)["state"] != STATE_COMPLETE:
            raise CampaignRunError("only a complete draw is materialised")
        return self.views().write(state, self.draw_dir, self.ledger.iter_raw())
