from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.rulesets.rules import is_meta_rule_id as _is_meta_rule_id
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_SCREENING,
    COMPLETION_FNAME,
    PROVIDER_DEFAULT_TRUNCATION,
    RETRY_SCOPE_COMPONENT,
    TRUNCATION_NONE,
    ConditionSpec,
    SamplingConfig,
    MANIFEST_AMENDMENTS_KEY,
    REPLAY_RECORD_FNAME,
    SAMPLING_MANIFEST_FNAME,
    build_sampling_manifest,
    check_sampling_manifest,
    draw_dirname,
    ensure_sampling_manifest,
    load_sampling_manifest,
    recorded_fingerprint,
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_BOTH,
    EVALUATION_MODE_FULL,
    EVALUATION_MODES,
    SHARED_BANK_DESIGN_FNAME,
)
from ruleforge.scenarios.manifest import (
    SAMPLING_SCHEMA_VERSION,
    ScenarioManifest,
)
from ruleforge.core import record_names
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import (
    canonical_json,
    sha256_file,
)
from ruleforge.core.identifiers import validate_identifier
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate import configured
from ruleforge.evaluate.amendment import AMENDED_FIELD, CapAmendment, cap_amendment
from ruleforge.evaluate.ledger import DrawLedger
from ruleforge.evaluate.packing import ExecutionSettings, PACKABLE_MODEL_TYPE, run_packed
from ruleforge.core.provenance import code_state, package_version, write_code_snapshot
from ruleforge.evaluate.sharding import (
    SHARD_RECORD_FNAME,
    ShardAssignment,
    ShardError,
    blocks_per_draw,
    code_state_key,
    describe_code_state,
    job_root_override,
    layout_code_states,
    recorded_code_states,
    shard_dirname,
    shard_records,
    shard_root,
    shards_root,
)
from ruleforge.evaluate.runner import (
    DEFAULT_MAX_TRANSPORT_RETRIES,
    STATE_COMPLETE,
    STATE_IN_PROGRESS,
    STATE_INCOMPLETE_EXHAUSTED,
    STATE_INVALID_MANIFEST,
    CampaignRunError,
    DrawRunner,
    DrawSpec,
    check_run_write,
    ledger_counts,
    refuse_fake_in_final,
    utc_now,
)
from ruleforge.tasks.spec import BaseTaskSpec, task_identity_of
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT
from ruleforge.tasks.base import DEFAULT_TASK, Task, TaskError, TaskSettings
from ruleforge.tasks.registry import load_task
from ruleforge.core.artifacts import iter_jsonl_records
from ruleforge.judges.base import JudgeError, JudgeIdentity
from ruleforge.judges.providers import backend_identity, judge_adapter, judge_model_weights, judge_serving_dtype
from ruleforge.judges.backend import ModelBackend
from ruleforge.judges.models import (
    THINK_TAG_MODELS,
    UNTRUNCATED_SAMPLING,
    model_default_truncation,
    validate_default_truncation,
)
from ruleforge.judges.models import PROMPT_SHAPE_VERSION
from ruleforge.judges.batch_api import BatchDestinationRefused, guard_work_directories, work_directories_guarded
from ruleforge.rulesets.levels import RuleLevels, RuleLevelsError, load_rule_levels
from ruleforge.rulesets.variants import warn_unexpanded_ranges

logger = logging.getLogger(__name__)


def _is_meta(rule: Mapping[str, Any]) -> bool:
    return _is_meta_rule_id(str(rule.get("id", "")))


SCENARIO_MANIFEST_FNAME = "scenario_manifest.json"
CONDITION_FNAME = "condition.json"
# A rebased root's record of its parent: such a root resumes only the conditions it registers.
REBASE_FNAME = "rebase.json"


@dataclass(frozen=True)
class CampaignPlan:
    """Everything one job needs to run its condition's draws."""

    campaign_id: str
    campaign_root: Path
    config: SamplingConfig
    scenario_manifest: ScenarioManifest
    scenario_rows: tuple[dict[str, Any], ...]
    condition: ConditionSpec
    rules: tuple[dict[str, Any], ...]
    judge_type: str
    decoder_draw_ids: tuple[int, ...]
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES
    code_state: Mapping[str, Any] = field(default_factory=dict)
    # The units this job runs when it is one shard; ``campaign_root`` is then the shard's own root.
    shard: Optional[ShardAssignment] = None
    # Aligned with ``rules``; a ``superiors`` condition's only.
    rule_levels: Optional[RuleLevels] = None
    # ``{draw: {scenario id: old cap}}``: bundles a cap amendment holds at the old cap.
    held_caps: Mapping[int, Mapping[str, int]] = field(default_factory=dict)

    @property
    def condition_root(self) -> Path:
        return self.campaign_root / self.condition.condition_id


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignRunError(message)


def campaign_task(cfg: DictConfig) -> Task:
    """The task a campaign config judges (``task_spec``; the Model Spec task when unset), refused when none is found."""
    try:
        return load_task(str(cfg.get("task_spec", DEFAULT_TASK)))
    except TaskError as error:
        raise CampaignRunError(f"task_spec: {error}") from error


def validate_campaign_cfg(cfg: DictConfig) -> None:
    """Refuse a config that is not a valid campaign; the task approves the prompt style, rule context and judge type."""
    _require(int(cfg.num_tries) == 1, (
        f"num_tries={cfg.num_tries}: per-prompt semantic retry cannot be "
        "composed with scenario-bundle retry; set num_tries: 1"
    ))
    _require(float(cfg.eval_cfg.evaluation_temperature) == 1.0, (
        f"eval_cfg.evaluation_temperature={cfg.eval_cfg.evaluation_temperature}: a "
        "stochastic campaign judges at temperature 1.0"
    ))
    task = campaign_task(cfg)
    style = campaign_style(cfg, task)
    _require(style in task.styles, (
        f"eval_cfg.reasoning_prompt_style={style!r} must be one of {tuple(task.styles)}"
    ))
    refusal = task.new_run_refusal(style)
    _require(refusal is None, f"eval_cfg.reasoning_prompt_style: {refusal}")
    _require(str(cfg.evaluation_mode) in EVALUATION_MODES, (
        f"evaluation_mode={cfg.evaluation_mode!r} must be one of {EVALUATION_MODES}: "
        "'both' is the full-set judgment plus every individual-rule judgment; "
        "'full' and 'individual_bank' are the shared-bank shapes"
    ))
    judge_types = [str(item) for item in cfg.eval_cfg.judge_types]
    _require(judge_types == [str(cfg.interpretation)], (
        f"eval_cfg.judge_types={judge_types} must be exactly [{cfg.interpretation!r}]"
    ))
    context = str(cfg.eval_cfg.get("individual_rule_context", "isolated")).lower()
    try:
        task.check(TaskSettings(style, context, str(cfg.interpretation)))
    except TaskError as error:
        key = "interpretation" if error.setting == "judge_type" else "eval_cfg"
        raise CampaignRunError(f"{key}: {error}") from error
    levels_path, approved = cfg.get("rule_levels_path"), cfg.get("approved_rule_levels_sha256")
    _require((context == HIERARCHY_RULE_CONTEXT) == (levels_path is not None), (
        f"rule_levels_path={levels_path}: a {HIERARCHY_RULE_CONTEXT!r} condition reads each rule's level from one, "
        f"and no other condition reads one (individual_rule_context={context!r})"
    ))
    _require(approved is None or levels_path is not None, (
        f"approved_rule_levels_sha256={approved} but no rule_levels_path: the approval would pin nothing"
    ))
    _require(bool(cfg.get("save_raw_outputs", True)), (
        "save_raw_outputs cannot be false: a campaign always stores every raw attempt"
    ))


def campaign_style(cfg: DictConfig, task: Task) -> str:
    """Return the prompt style a campaign config names, the task's default when unset."""
    style = str(cfg.eval_cfg.get("reasoning_prompt_style", task.default_style))
    return style if style in task.styles else style.lower()


def campaign_rule_levels(cfg: DictConfig) -> Optional[RuleLevels]:
    """Return each rule's level from ``rule_levels_path``, aligned row by row with ``rule_set_path``, or None without
    one."""
    levels_path = cfg.get("rule_levels_path")
    if levels_path is None:
        return None
    _require(Path(str(levels_path)).is_file(), f"rule_levels_path {levels_path} does not exist")
    try:
        levels = load_rule_levels(Path(str(cfg.rule_set_path)), Path(str(levels_path)))
    except RuleLevelsError as error:
        raise CampaignRunError(str(error)) from error
    approved = cfg.get("approved_rule_levels_sha256")
    _require(approved is None or levels.levels_file["sha256"] == str(approved), (
        f"rule levels SHA-256 {levels.levels_file['sha256']} does not match approved {approved}"
    ))
    return levels


def resolve_sampling_truncation(config: SamplingConfig, model_type: str, source: Any) -> Any:
    """Return this condition's effective truncation as identity: a local judge's ``generation_config.json`` over
    ``default_truncation`` under ``model_default``, the full distribution under ``none``, else ``provider_default``."""
    if model_type != "local":
        return PROVIDER_DEFAULT_TRUNCATION
    if config.truncation == TRUNCATION_NONE:
        return dict(UNTRUNCATED_SAMPLING)
    model_path, default = (
        source if isinstance(source, tuple) else (str(source.model_path), source.get("default_truncation"))
    )
    return {
        **UNTRUNCATED_SAMPLING,
        **model_default_truncation(model_path, default),
    }


def _preflight_default_truncation(cfg: DictConfig) -> None:
    values = cfg.judge_model.get("default_truncation")
    if not values:
        return
    _require(str(cfg.judge_model.model_type) == "local", (
        f"judge_model.default_truncation is set for a {cfg.judge_model.model_type!r} judge; "
        "only a local judge samples with a truncation this repo sets"
    ))
    try:
        validate_default_truncation(values)
    except ValueError as error:
        raise CampaignRunError(f"judge_model {cfg.judge_model.model_name}: {error}") from error


def _preflight_adapter(cfg: DictConfig) -> None:
    approved = cfg.get("approved_adapter_sha256")
    if not cfg.judge_model.get("adapter_path"):
        _require(approved is None, (
            f"approved_adapter_sha256={approved} but judge_model.adapter_path is unset: "
            "this run would judge with the base model"
        ))
        return
    _require(str(cfg.judge_model.model_type) == "local", (
        f"judge_model.adapter_path is set for a {cfg.judge_model.model_type!r} judge; only a "
        "local judge serves a LoRA adapter"
    ))
    adapter = judge_adapter(cfg.judge_model)
    if approved is not None:
        _require(adapter.sha256 == str(approved), (
            f"adapter SHA-256 {adapter.sha256} ({adapter.label}) does not match approved {approved}"
        ))


def _preflight_strict_think(cfg: DictConfig, config: SamplingConfig) -> None:
    if not config.strict_think:
        return
    judge = cfg.judge_model
    strip = judge.get("strip_thinking_model")
    _require(
        str(judge.model_type) == "local" and judge.get("enable_thinking") is True and strip in THINK_TAG_MODELS, (
            f"sampling.strict_think needs a local thinking judge whose thinking closes with </think> "
            f"(enable_thinking true, strip_thinking_model one of {sorted(THINK_TAG_MODELS)}); judge_model "
            f"{judge.model_name} is {judge.model_type!r} with enable_thinking {judge.get('enable_thinking')!r} and "
            f"strip_thinking_model {strip!r}"
        ),
    )


def _preflight_weights(cfg: DictConfig) -> None:
    # An fp32 merge with no declared serving dtype is refused: vLLM's ``auto`` would silently serve it in bf16.
    approved = cfg.get("approved_model_weights_sha256")
    judge = cfg.judge_model
    if str(judge.model_type) != "local":
        _require(approved is None and judge.get("dtype") is None, (
            f"approved_model_weights_sha256={approved} or dtype={judge.get('dtype')} is set for a "
            f"{judge.model_type!r} judge; only a local judge serves weights this repo names or a dtype it sets"
        ))
        return
    try:
        serving = judge_serving_dtype(judge)
    except JudgeError as error:
        raise CampaignRunError(str(error)) from error
    weights = judge_model_weights(judge)
    if weights is None:
        _require(approved is None, (
            f"approved_model_weights_sha256={approved} but {judge.model_path} carries no derivation manifest: "
            "its weights are named by directory name, so the approval would pin nothing the condition records"
        ))
        return
    _require(approved is None or weights.sha256 == str(approved), (
        f"model weights SHA-256 {weights.sha256} ({weights.path}) does not match approved {approved}"
    ))
    _require(serving is not None or weights.saved_dtype != "float32", (
        f"{weights.path} holds float32 weights and judge_model {judge.model_name} declares no dtype: vLLM's auto "
        "would serve them in bf16 on Hopper, silently; declare dtype (the fp32 engine is "
        "configs/model/qwen3-8b-fp32.yaml)"
    ))


def condition_spec_for(
    *,
    condition_id: str,
    config: SamplingConfig,
    judge: Optional[JudgeIdentity] = None,
    task: Optional[Mapping[str, Any]] = None,
    ruleset: Any = None,
    backend: Optional[ModelBackend] = None,
    task_spec: Optional[BaseTaskSpec] = None,
    judge_type: Optional[str] = None,
    rule_set_path: Optional[Path] = None,
    model_cfg: Any = None,
    extra: Mapping[str, Any] | None = None,
    evaluation_mode: str = EVALUATION_MODE_BOTH,
    rule_levels_sha256: Optional[str] = None,
) -> ConditionSpec:
    """Return the condition identity a judge, a task and a rule set define. Pass them as identities, or a constructed
    ``backend`` with ``model_cfg``, ``task_spec``, ``judge_type`` and ``rule_set_path``."""
    if judge is None:
        task = condition_task_identity(task_spec, str(judge_type))
        try:
            judge = backend_identity(backend, model_cfg)
        except JudgeError as error:
            raise CampaignRunError(str(error)) from error
        rule_set_name, rule_set_sha256 = Path(rule_set_path).name, sha256_file(rule_set_path)
    else:
        rule_set_name, rule_set_sha256 = ruleset.name, ruleset.sha256
    adapter = judge.adapter
    return ConditionSpec(
        condition_id=condition_id,
        model_name=judge.model_name,
        model_type=judge.model_type,
        prompt_style=task["prompt_style"],
        individual_rule_context=task["individual_rule_context"],
        judge_type=task["judge_type"],
        evaluation_mode=evaluation_mode,
        rule_set_name=rule_set_name,
        rule_set_sha256=rule_set_sha256,
        temperature=config.temperature,
        prompt_shape_version=PROMPT_SHAPE_VERSION,
        forced_json_requirement=config.forced_json_requirement,
        label_rule=config.label_rule,
        retry_scope=config.retry_scope,
        strict_think=config.strict_think,
        endpoint=judge.endpoint,
        reasoning_effort=judge.reasoning_effort,
        max_new_tokens=judge.max_new_tokens,
        send_max_output_tokens=judge.send_max_output_tokens,
        seed_capability=judge.seed_capability,
        model_path=judge.model_path,
        adapter_path=None if adapter is None else adapter.label,
        adapter_sha256=None if adapter is None else adapter.sha256,
        adapter_config_sha256=None if adapter is None else adapter.config_sha256,
        model_weights_sha256=judge.model_weights_sha256,
        serving_dtype=judge.serving_dtype,
        enable_thinking=judge.enable_thinking,
        sampling_truncation=resolve_sampling_truncation(config, judge.model_type, judge.truncation_source),
        rule_levels_sha256=rule_levels_sha256,
        task=task["task"],
        response_contract=task["response_contract"],
        judge_extension=judge.extension,
        extra=tuple((extra or {}).items()),
    )


def condition_task_identity(task_spec: Any, judge_type: str) -> dict[str, Any]:
    """Return what a condition records of its task."""
    try:
        return task_identity_of(task_spec, judge_type)
    except TaskError as error:
        raise CampaignRunError(f"task spec {getattr(task_spec, 'task_spec_name', '?')}: {error}") from error


def condition_extra(cfg: DictConfig) -> dict[str, Any]:
    """Return ``cfg.condition_extra`` resolved; an undeclared value (``???``) refuses the campaign at preflight."""
    node = cfg.get("condition_extra")
    if not node:
        return {}
    try:
        return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    except OmegaConfBaseException as error:
        message = str(error).splitlines()[0]
        raise CampaignRunError(
            f"condition_extra does not resolve ({message}): it is condition identity, so "
            "every value must be declared before any condition of the campaign runs"
        ) from error


def execution_settings(cfg: DictConfig) -> ExecutionSettings:
    """Return the job's validated ``execution:`` block; packing is refused for a judge that is not local."""
    block = cfg.get("execution")
    if isinstance(block, DictConfig):
        block = OmegaConf.to_container(block, resolve=True)
    settings = ExecutionSettings.from_mapping(block)
    model_type = cfg.judge_model.get("model_type")
    _require(not settings.pack_draws or model_type == PACKABLE_MODEL_TYPE, (
        f"execution.pack_draws needs a {PACKABLE_MODEL_TYPE!r} judge; {model_type!r} keeps "
        "its per-block submissions"
    ))
    _require(not settings.sharded or model_type == PACKABLE_MODEL_TYPE, (
        f"execution.num_shards needs a {PACKABLE_MODEL_TYPE!r} judge: a batch provider addresses its paid "
        f"batch by request block, and a shard's retry blocks are re-planned when the shards merge; "
        f"{model_type!r} runs its condition as one job"
    ))
    _require(not (settings.sharded and settings.allow_stack_change), (
        "execution.allow_stack_change with execution.num_shards: the shard merge refuses a condition judged "
        "on two inference stacks, so a shard finishes on the stack its condition started on"
    ))
    return settings


# Spells Hydra's run and sweep directories in ``configs/sampling_campaign.yaml``.
UNPINNED_RESOLVER = "rf.unpinned"


def unpinned(place: Any) -> str:
    """Return ``place`` as a string once it is on no pinned root; registered with OmegaConf on import; nothing is
    written."""
    from ruleforge.evaluate.api import refuse_pinned

    refuse_pinned(str(place), "the job root Hydra makes its run directory under")
    return str(place)


OmegaConf.register_new_resolver(UNPINNED_RESOLVER, unpinned, use_cache=False, replace=True)


def job_root(cfg: DictConfig, settings: ExecutionSettings) -> Path:
    """Return where this job writes: the campaign root, or one shard's root; ``job_root`` must name exactly that."""
    campaign_root = Path(str(cfg.campaign_root))
    configured = cfg.get("job_root")
    given = campaign_root if configured is None else Path(str(configured))
    if not settings.sharded:
        _require(given == campaign_root, (
            f"job_root={given} is not the campaign root {campaign_root}: a job writes another root only as "
            "one shard of its condition (execution.num_shards and execution.shard_index)"
        ))
        return campaign_root
    expected = shard_root(campaign_root, str(cfg.condition_id), settings.shard_index, settings.num_shards)
    _require(given == expected, (
        f"shard {settings.shard_index} of {settings.num_shards} writes its own root {expected}, so its job_root "
        f"(and Hydra's run directory under it) must be that root, not {given}: pass "
        f"'{job_root_override(settings.shard_index, settings.num_shards)}'"
    ))
    return expected


def check_job_roots(cfg: DictConfig, settings: Optional[ExecutionSettings] = None, *,
                    workdir: Optional[str] = None) -> tuple[Path, Path]:
    """Return the campaign root and the job's own root, each held to the places a configured job may write; nothing is
    written."""
    from ruleforge.evaluate.api import refuse_protected

    settings = execution_settings(cfg) if settings is None else settings
    campaign_root = Path(str(cfg.campaign_root))
    job = job_root(cfg, settings)
    if workdir is not None:
        campaign_root, job = Path(workdir, campaign_root), Path(workdir, job)
    refuse_protected(campaign_root, "the campaign root a configured job writes", campaign_root=True)
    if job != campaign_root:
        refuse_protected(job, "the shard root a configured job writes", campaign_root=True)
    return campaign_root, job


def requested_draw_ids(cfg: DictConfig, config: SamplingConfig) -> tuple[int, ...]:
    """``decoder_draw_ids`` checked against the campaign's draws; every draw when null."""
    requested = cfg.get("decoder_draw_ids")
    if requested is None:
        return tuple(range(config.num_decoder_draws))
    draw_ids = tuple(int(item) for item in requested)
    bad = [d for d in draw_ids if not 0 <= d < config.num_decoder_draws]
    _require(not bad and len(set(draw_ids)) == len(draw_ids), (
        f"decoder_draw_ids={list(draw_ids)} must be distinct ids below "
        f"{config.num_decoder_draws}"
    ))
    return draw_ids


def shard_assignment(
    cfg: DictConfig, settings: ExecutionSettings, config: SamplingConfig, manifest: ScenarioManifest
) -> Optional[ShardAssignment]:
    """Return this job's units when it runs one shard, or None; a screening campaign is never sharded."""
    if not settings.sharded:
        return None
    _require(config.campaign_mode != CAMPAIGN_MODE_SCREENING, (
        "execution.num_shards shards a final or pilot campaign's draws; a screening campaign runs its condition "
        "as one job"
    ))
    n_blocks = blocks_per_draw(manifest.record_count, config.scenarios_per_request_block)
    draw_ids = requested_draw_ids(cfg, config)
    try:
        return ShardAssignment.build(settings.shard_index, settings.num_shards, draw_ids, n_blocks)
    except ShardError as error:
        raise CampaignRunError(str(error)) from error


_SHARD_PLAN_KEYS = ("campaign_id", "condition_id", "shard_index", "num_shards", "plan_draw_ids", "blocks_per_draw",
                    "units")


def _check_shard_record(path: Path, payload: Mapping[str, Any], keys: Sequence[str]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8"))
    differing = [key for key in keys if existing.get(key) != payload.get(key)]
    _require(not differing, (
        f"{path} records another plan than this job's ({', '.join(differing)} differ; this job runs units "
        f"{payload['units']}): one shard root belongs to one shard of one plan. Relaunch it with the "
        "arguments it was launched with (the same decoder_draw_ids and population), or shard this plan "
        "under another shard count"
    ))


def write_shard_record(root: Path, payload: Mapping[str, Any]) -> Path:
    """Write a shard root's ``shard.json`` once; refuse a root recorded for other units."""
    path = Path(root) / SHARD_RECORD_FNAME
    if path.exists():
        _check_shard_record(path, payload, tuple(payload))
        return path
    return write_json_sidecar(path, dict(payload))


def rebase_record(campaign_root: Path | str) -> Optional[dict[str, Any]]:
    """Return a rebased root's ``rebase.json``, or None; refused unless it records this root's campaign and
    population."""
    root = Path(campaign_root)
    path = root / REBASE_FNAME
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        found = (record["parent"]["campaign_id"], record["target"]["scenario_manifest_sha256"])
        own = (load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)["campaign_id"],
               ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME).manifest_sha256)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise CampaignRunError(f"{path} cannot be read as a rebase's record ({error!r})") from error
    _require(found == own, (
        f"{path} records a rebase of campaign {found[0]!r} onto scenario manifest {str(found[1])[:12]}, not this root's "
        f"{own[0]!r} over {own[1][:12]}: it is no record of this root"
    ))
    return record


def carried_exhausted(campaign_root: Path | str) -> set[tuple[str, int, str]]:
    """Return ``(condition id, decoder draw, scenario id)`` of each bundle a rebase carried over exhausted."""
    record = rebase_record(campaign_root)
    carried: set[tuple[str, int, str]] = set()
    try:
        for relative, scenario_ids in ({} if record is None else record.get("carried_exhausted") or {}).items():
            condition_id, draw = str(relative).rsplit("/", 1)
            carried.update((condition_id, int(draw.removeprefix("draw-")), str(sid)) for sid in scenario_ids)
    except (AttributeError, TypeError, ValueError) as error:
        raise CampaignRunError(f"{Path(campaign_root) / REBASE_FNAME} records a malformed carried_exhausted "
                               f"({error!r})") from error
    return carried


def campaign_amendment(campaign_root: Path) -> Optional[CapAmendment]:
    """Return the campaign's cap amendment as every root records it, or None, reading only; refused while roots disagree
    (record the same request again to complete it) or a carried bundle is neither held nor named."""
    paths = [campaign_root / SAMPLING_MANIFEST_FNAME,
             *(record.parent / SAMPLING_MANIFEST_FNAME for record in shard_records(campaign_root))]
    manifests = {path: load_sampling_manifest(path) for path in paths if path.is_file()}
    held = [manifest for manifest in manifests.values() if MANIFEST_AMENDMENTS_KEY in manifest]
    if not held:
        return None
    lacking = [str(path.parent) for path, manifest in manifests.items() if MANIFEST_AMENDMENTS_KEY not in manifest]
    different = len({canonical_json(manifest[MANIFEST_AMENDMENTS_KEY]) for manifest in held}) > 1
    _require(not lacking and not different, (
        f"the campaign's cap amendment is recorded in {len(held)} of its {len(manifests)} roots"
        + (f", not in {', '.join(lacking)}: a record interrupted part-way. Before any job of the campaign runs, "
           "complete it by recording the interrupted request again, its cap and bundles the record's" if lacking else
           ", and they record different ones. Resolve them before any job of the campaign runs")
    ))
    amendment = cap_amendment(held[0])
    carried, kept = carried_exhausted(campaign_root), set(amendment.held)
    named = {(bundle.condition_id, bundle.decoder_draw_id, bundle.scenario_id) for bundle in amendment.bundles}
    problems = [f"it holds {sorted(kept - carried)[:3]} at the old cap, which {REBASE_FNAME} records no rebase "
                "carrying over exhausted"] if kept - carried else []
    if carried - kept - named:
        problems.append(f"it neither holds at the old cap nor names {sorted(carried - kept - named)[:3]}, which the "
                        "rebase carried over exhausted: a resume of its draw would give it the raised cap and judge it "
                        "again, unrecorded")
    if named & kept:
        problems.append(f"it both holds and names {sorted(named & kept)[:3]}")
    _require(not problems, (
        f"the campaign's cap amendment (ledger {amendment.ledger}) is not a record of this root: "
        + "; ".join(problems) + ". No job of the campaign runs on it. While no draw has "
        "resumed under the raised cap, it is put right by recording it again: keep a copy of the record (its cap, "
        "bundles and decision are the request's), in every root holding it "
        f"({', '.join(str(path.parent) for path in manifests)}) restore {AMENDED_FIELD} {amendment.old_cap} and "
        f"remove {MANIFEST_AMENDMENTS_KEY}, then record it again"
    ))
    return amendment


def _preflight_amended(
    cfg: DictConfig,
    settings: ExecutionSettings,
    config: SamplingConfig,
    amendment: Optional[CapAmendment],
    assignment: Optional[ShardAssignment],
) -> None:
    # Refuses work a cap amendment does not cover (its surveyed conditions, draws and bundles) before the judge loads.
    if amendment is None:
        return
    condition_id = str(cfg.condition_id)
    where = (f"the campaign's cap was raised from {amendment.old_cap} to {amendment.new_cap} by an amendment "
             f"(recorded as {amendment.ledger}) covering the conditions and draws it surveyed")
    _require(condition_id in amendment.conditions, (
        f"{where} ({', '.join(amendment.conditions)}): {condition_id} is not one of them, and would run every bundle "
        "under the raised cap. Run it in a fresh campaign root"
    ))
    root = job_root(cfg, settings)
    if assignment is not None:
        # A root whose job died between its manifest and its shard record holds no draw and no record.
        _require((root / SHARD_RECORD_FNAME).is_file(), (
            f"{where}: {root} is a shard root started after it, which would judge again draws the campaign has "
            "finished, under the raised cap, and whose manifest would lack the record the merge compares. Relaunch "
            "only shard roots the amendment recorded"
        ))
        draws, held_by = assignment.draws, assignment.label
    else:
        draws, held_by = requested_draw_ids(cfg, config), None
        missing = [draw_dirname(draw) for draw in draws if not (root / condition_id / draw_dirname(draw)).is_dir()]
        _require(not missing, (
            f"{where}: {root / condition_id} does not hold {missing}, which this job would start under the raised "
            "cap. Resume only draws the campaign root holds (decoder_draw_ids), or run them in a fresh campaign root"
        ))
        # Readers hold an amended derived or merged draw file for file, so a run row logged in it would leave it
        # unreadable.
        named = [bundle for draw in draws for bundle in amendment.bundles
                 if (bundle.condition_id, bundle.decoder_draw_id) == (condition_id, draw)]
        _require(not (named and (root / REPLAY_RECORD_FNAME).is_file()), (
            f"{where}: {root} is a label replay's derived root ({REPLAY_RECORD_FNAME}), whose amended "
            "draws its readers hold to the files its re-score wrote, by digest (amendment._derived_outcomes): this "
            f"job's runs.jsonl row in the draw of {', '.join(bundle.describe() for bundle in named)} would leave it "
            "unreadable. A derived root is never resumed: run the work in a fresh campaign root"
        ))
        merged = [bundle.describe() for bundle in named if bundle.shard is not None]
        _require(not merged, (
            f"{where}: {', '.join(merged)} resumed in shard roots and merged into {root / condition_id}, whose "
            "readers hold such a draw to the merge of its shard roots, file for file (shard_merge.check_merged_draw): "
            "this job's runs.jsonl row in it would leave it unreadable. Leave those draws out (decoder_draw_ids): "
            "they hold nothing left to judge"
        ))
    resumed = [bundle.describe() for draw in draws for bundle in amendment.bundles_of(condition_id, draw, held_by)]
    if not resumed:
        return
    current = code_state_key(code_state())
    commit = amendment.code_state[0]
    _require(current == amendment.code_state, (
        f"{where}: this job resumes {', '.join(resumed)}, whose new attempts only the amendment's code state, "
        f"{describe_code_state(amendment.code_state)}, may run, and it runs {describe_code_state(current)}; the "
        "merge and the analysis refuse a run of any other in the draw, and this job's runs.jsonl row would make "
        f"that permanent. Launch from commit {commit} with a clean tree (ruleforge jobs launch campaign ... "
        "--expect-commit "
        f"{commit})"
    ))
    _require(not settings.allow_stack_change, (
        f"{where}: this job resumes {', '.join(resumed)} with execution.allow_stack_change, and the analysis refuses "
        "an amended draw whose run log holds a stack change. Resume it on the stack its attempts were recorded on"
    ))


def _preflight_shard(
    cfg: DictConfig, settings: ExecutionSettings, assignment: ShardAssignment, campaign_root: Path,
    amendment: Optional[CapAmendment],
) -> None:
    # Refuses what would make a shard's work unmergeable before the judge loads and a permanent ``runs.jsonl`` row:
    # another plan's record, a draw the campaign root holds, or a second code state.
    condition_id = str(cfg.condition_id)
    record = job_root(cfg, settings) / SHARD_RECORD_FNAME
    if record.exists():
        _check_shard_record(record, assignment.record(
            campaign_id=str(cfg.campaign_id), condition_id=condition_id, condition_fingerprint="",
        ), _SHARD_PLAN_KEYS)
    held = [draw for draw in assignment.draw_ids if (campaign_root / condition_id / draw_dirname(draw)).exists()]
    lacking = [draw for draw in assignment.draw_ids if draw not in held]
    _require(not held, (
        f"{campaign_root / condition_id} already holds draw(s) {held} of this shard plan: an unsharded job "
        "resumes them there, but shards would judge them again and the merge refuses a draw the root holds "
        "with other content. "
        + (f"Shard only the draws it lacks (decoder_draw_ids={lacking})" if lacking else
           "Every planned draw is there already; nothing is left to shard")
    ))
    current = code_state_key(code_state())
    shards_dir = shards_root(campaign_root) / condition_id
    if amendment is not None:
        # A cap amendment lets its own code state in beside the campaign's, in the draws it names alone.
        found = layout_code_states(((shard_dirname(index, settings.num_shards),
                                     shard_root(campaign_root, condition_id, index, settings.num_shards))
                                    for index in range(settings.num_shards)), condition_id)
        found.setdefault(current, []).extend((assignment.label, draw_dirname(draw)) for draw in assignment.draws)
        refusal = amendment.code_state_refusal(found, condition_id) if len(found) > 1 else None
        own = " or ".join(sorted({str(state[0]) for state in found if state in amendment.prior_code_states}))
        hint = ("Relaunch only the shards holding the amended draws (--only), every one of whose draws the amendment "
                "names" if current == amendment.code_state else
                f"Relaunch the shards it does not name from the campaign's own code state (ruleforge jobs launch "
                "campaign ... "
                f"--expect-commit {own or '<its commit>'}), the ones it names from its own (--expect-commit "
                f"{amendment.code_state[0]})")
        _require(refusal is None, (
            f"this job runs {describe_code_state(current)}, and the campaign's cap amendment allows its layout the "
            f"campaign's own code state and, only in the draws it names, its own: {refusal}. {hint}"
        ))
        return
    for index in range(settings.num_shards):
        root = shard_root(campaign_root, condition_id, index, settings.num_shards)
        for state, draws in recorded_code_states(root, condition_id).items():
            _require(state == current, (
                f"{shard_dirname(index, settings.num_shards)} ({', '.join(draws)}) ran {describe_code_state(state)}, "
                f"and this job runs {describe_code_state(current)}: the merge refuses a condition whose shards ran "
                f"two code states, and this job's runs.jsonl rows would make that permanent. Launch from commit "
                f"{state[0]} with a clean tree (ruleforge jobs launch campaign ... --expect-commit {state[0]}), or "
                f"move {shards_dir} aside to run the whole layout again on this one"
            ))


def preflight_judge(cfg: DictConfig, config: SamplingConfig) -> None:
    """Refuse a judge that cannot serve its config: adapter, weights, dtype, ``default_truncation``,
    ``strict_think``."""
    _preflight_adapter(cfg)
    _preflight_weights(cfg)
    _preflight_default_truncation(cfg)
    _preflight_strict_think(cfg, config)


def preflight(cfg: DictConfig) -> tuple[SamplingConfig, ScenarioManifest]:
    """Refuse a campaign config without loading a judge model, so a bad job fails in seconds; writes nothing."""
    validate_campaign_cfg(cfg)
    settings = execution_settings(cfg)
    campaign_root, job = check_job_roots(cfg, settings)
    check_new_condition_id(campaign_root, str(cfg.condition_id))
    condition_extra(cfg)
    config = SamplingConfig.from_mapping(
        OmegaConf.to_container(cfg.sampling, resolve=True)
    )
    refuse_fake_in_final(str(cfg.judge_model.get("model_type")), config.campaign_mode)
    validate_identifier("campaign_id", str(cfg.campaign_id))
    scenario_path = Path(str(cfg.scenario_set_path))
    approved = cfg.get("approved_scenario_sha256")
    if approved is not None:
        actual = sha256_file(scenario_path)
        _require(actual == str(approved), (
            f"scenario SHA-256 {actual} does not match approved {approved}"
        ))
    manifest = ScenarioManifest.build(
        scenario_path, expected_count=config.expected_scenario_count
    )
    rule_set_path = Path(str(cfg.rule_set_path))
    _require(rule_set_path.is_file(), f"rule_set_path {rule_set_path} does not exist")
    approved_rules = cfg.get("approved_rule_set_sha256")
    if approved_rules is not None:
        actual = sha256_file(rule_set_path)
        _require(actual == str(approved_rules), (
            f"rule set SHA-256 {actual} does not match approved {approved_rules}"
        ))
    campaign_rule_levels(cfg)
    preflight_judge(cfg, config)
    assignment = shard_assignment(cfg, settings, config, manifest)
    # A root that already froze a population or manifest must agree with this config now, before a GPU loads the judge.
    roots = [campaign_root]
    _require(not (campaign_root / REBASE_FNAME).is_file()
             or (campaign_root / str(cfg.condition_id) / CONDITION_FNAME).is_file(), (
        f"{campaign_root} is a rebased root ({REBASE_FNAME}), which resumes the conditions it registers; "
        f"{cfg.condition_id} is not one of them. Check the launch's variables, or run it in a root of its own"
    ))
    amendment = campaign_amendment(campaign_root)
    _preflight_amended(cfg, settings, config, amendment, assignment)
    if assignment is not None:
        roots.append(job)
        check_new_condition_id(job, str(cfg.condition_id))
        _preflight_shard(cfg, settings, assignment, campaign_root, amendment)
    for root in roots:
        frozen_path = root / SCENARIO_MANIFEST_FNAME
        if frozen_path.exists():
            frozen = ScenarioManifest.load(frozen_path)
            _require(frozen.manifest_sha256 == manifest.manifest_sha256, (
                f"{frozen_path} froze a different population "
                f"({frozen.manifest_sha256[:12]}...) than {scenario_path.name} "
                f"({manifest.manifest_sha256[:12]}...)"
            ))
        check_sampling_manifest(
            root / SAMPLING_MANIFEST_FNAME,
            campaign_manifest(str(cfg.campaign_id), config, manifest),
        )
        # A configured root's marker is left before the manifests, so a root may hold it alone.
        if os.path.lexists(root / configured.MARKER_FNAME):
            configured.check_marker(root, campaign_id=str(cfg.campaign_id),
                                    scenario_manifest_sha256=manifest.manifest_sha256)
    return config, manifest


def campaign_manifest(
    campaign_id: str,
    config: SamplingConfig,
    manifest: ScenarioManifest,
    code: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return the campaign manifest a job with this config freezes over ``manifest``; ``code`` None records it as
    unknown."""
    code = code or {}
    return build_sampling_manifest(
        campaign_id=campaign_id,
        config=config,
        scenario_manifest=manifest,
        conditions=[],
        created_at=utc_now(),
        code_revision=code.get("code_revision"),
        code_dirty=code.get("code_dirty"),
        code_state_sha256=code.get("code_state_sha256"),
        package_version=package_version(),
    )


def config_condition(
    cfg: DictConfig,
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
    config: SamplingConfig,
    levels: Optional[RuleLevels],
) -> ConditionSpec:
    """Return the condition a campaign config registers; nothing is written."""
    return condition_spec_for(
        condition_id=str(cfg.condition_id),
        backend=backend,
        task_spec=task_spec,
        config=config,
        judge_type=str(cfg.interpretation),
        rule_set_path=Path(str(cfg.rule_set_path)),
        model_cfg=cfg.judge_model,
        extra=condition_extra(cfg),
        evaluation_mode=str(cfg.evaluation_mode),
        rule_levels_sha256=None if levels is None else levels.levels_file["sha256"],
    )


def build_plan(
    cfg: DictConfig,
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
) -> CampaignPlan:
    """Validate the config, freeze the population, register the condition under the job's root, and return the plan."""
    config, manifest = preflight(cfg)
    settings = execution_settings(cfg)
    campaign_id = str(cfg.campaign_id)
    campaign_root = job_root(cfg, settings)
    scenario_path = Path(str(cfg.scenario_set_path))
    levels = campaign_rule_levels(cfg)
    condition = config_condition(cfg, backend=backend, task_spec=task_spec, config=config, levels=levels)
    shard = shard_assignment(cfg, settings, config, manifest)
    if shard is not None:
        # The merge registers the condition in the campaign root: refuse another identity under this label now.
        check_registered_condition(Path(str(cfg.campaign_root)), condition)
    check_registered_condition(campaign_root, condition)
    campaign_root.mkdir(parents=True, exist_ok=True)
    configured.mark(campaign_root, campaign_id=campaign_id, scenario_manifest_sha256=manifest.manifest_sha256)
    manifest_path = campaign_root / SCENARIO_MANIFEST_FNAME
    if manifest_path.exists():
        frozen = ScenarioManifest.load(manifest_path)
        _require(frozen.manifest_sha256 == manifest.manifest_sha256, (
            f"{manifest_path} froze a different population "
            f"({frozen.manifest_sha256[:12]}...) than {scenario_path.name} "
            f"({manifest.manifest_sha256[:12]}...)"
        ))
    else:
        manifest.write(manifest_path)
    code = code_state()
    ensure_sampling_manifest(
        campaign_root / SAMPLING_MANIFEST_FNAME, campaign_manifest(campaign_id, config, manifest, code)
    )
    write_code_snapshot(campaign_root, code)
    rule_set_path = Path(str(cfg.rule_set_path))
    rules = tuple(iter_jsonl_records(rule_set_path))
    warn_unexpanded_ranges(list(rules), condition.individual_rule_context)
    judge_type = str(cfg.interpretation)
    register_condition(campaign_root, condition)
    draw_ids = requested_draw_ids(cfg, config)
    if shard is not None:
        write_shard_record(campaign_root, shard.record(
            campaign_id=campaign_id, condition_id=condition.condition_id, condition_fingerprint=condition.fingerprint,
        ))
        logger.info("Shard %s of %s: units %s", shard.label, condition.condition_id, list(shard.units))
        draw_ids = shard.draws
    amendment = campaign_amendment(Path(str(cfg.campaign_root)))
    held_caps = {} if amendment is None else {
        draw: caps for draw in draw_ids if (caps := amendment.held_caps(condition.condition_id, draw))}
    plan = CampaignPlan(
        campaign_id=campaign_id,
        campaign_root=campaign_root,
        config=config,
        scenario_manifest=manifest,
        scenario_rows=tuple(iter_jsonl_records(scenario_path)),
        condition=condition,
        rules=rules,
        judge_type=judge_type,
        decoder_draw_ids=draw_ids,
        max_transport_retries=int(cfg.get("max_transport_retries", DEFAULT_MAX_TRANSPORT_RETRIES)),
        code_state=code,
        shard=shard,
        rule_levels=levels,
        held_caps=held_caps,
    )
    logger.info(
        "Condition %s: %d scenarios, %d requests per draw",
        condition.condition_id,
        manifest.record_count,
        expected_requests_per_draw(manifest, rules, condition.evaluation_mode),
    )
    return plan


def non_meta_rule_count(rules: Sequence[Mapping[str, Any]]) -> int:
    """How many rules of a ruleset are judged individually (meta-rules are not)."""
    return sum(1 for rule in rules if not _is_meta(rule))


def expected_requests_per_draw(
    manifest: ScenarioManifest,
    rules: Sequence[Mapping[str, Any]],
    evaluation_mode: str = EVALUATION_MODE_BOTH,
) -> int:
    """Return the per-draw request count: ``N * (1 + non-meta rules)``, ``N`` for full-only, ``N * non-meta rules`` for
    a bank."""
    if evaluation_mode not in EVALUATION_MODES:
        raise SamplingDesignError(f"unknown evaluation_mode {evaluation_mode!r}")
    per_bundle = {
        EVALUATION_MODE_BOTH: 1 + non_meta_rule_count(rules),
        EVALUATION_MODE_FULL: 1,
        EVALUATION_MODE_BANK: non_meta_rule_count(rules),
    }[evaluation_mode]
    return manifest.record_count * per_bundle


def condition_payload(condition: ConditionSpec) -> dict[str, Any]:
    """A condition's ``condition.json`` payload."""
    return {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        **condition.to_dict(),
        "fingerprint": condition.fingerprint,
    }


def registers(payload: Any, condition: ConditionSpec, schema_version: Any = SAMPLING_SCHEMA_VERSION) -> bool:
    """Return whether a ``condition.json`` payload registers ``condition``, compared in memory and by fingerprint."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != schema_version:
        return False
    try:
        registered = condition_of(payload, "a registration")
    except (ValueError, TypeError):
        return False
    return registered == condition and registered.fingerprint == condition.fingerprint


def check_new_condition_id(campaign_root: Path, condition_id: str) -> None:
    """Refuse a new condition id holding a recorded or retired key as a segment."""
    root = Path(campaign_root)
    if not (root / condition_id / CONDITION_FNAME).exists():
        try:
            record_names.refuse_record_keys(condition_id)
        except SamplingDesignError as error:
            raise CampaignRunError(f"{root}: {error}") from error


def check_condition_id(campaign_root: Path, condition: ConditionSpec) -> None:
    """Refuse an id no registration may take: a new one holding a recorded key, or one sharing another condition's
    identity id."""
    root = Path(campaign_root)
    check_new_condition_id(root, condition.condition_id)
    identity = condition.identity_id
    if identity != condition.condition_id and (root / identity / CONDITION_FNAME).exists():
        raise CampaignRunError(
            f"{root / identity / CONDITION_FNAME} registers {identity!r}, the id condition {condition.condition_id!r} "
            "is hashed over (its identity id): the two would share generation seeds, provider request namespaces and "
            "draw grids. A root holding a condition the published records name so is read, never extended under the "
            f"public names; run {condition.condition_id!r} in a root of its own")


def check_registered_condition(campaign_root: Path, condition: ConditionSpec) -> Path:
    """Return ``<condition_id>/condition.json`` under ``campaign_root``, refusing another identity or an unusable id."""
    path = Path(campaign_root) / condition.condition_id / CONDITION_FNAME
    if path.exists() and not registers(json.loads(path.read_text(encoding="utf-8")), condition):
        raise CampaignRunError(
            f"{path} already registers condition {condition.condition_id!r} with a "
            "different identity; two runs cannot share a label and differ in the "
            "prompt, model, rule set or validation rule"
        )
    check_condition_id(campaign_root, condition)
    check_judge_configuration(campaign_root, [condition])
    return path


def check_judge_configuration(campaign_root: Path, conditions: Sequence[ConditionSpec]) -> None:
    """In a shared-bank root, refuse ``conditions`` unless each model holds one judge configuration."""
    if (Path(campaign_root) / SHARED_BANK_DESIGN_FNAME).is_file():
        from ruleforge.evaluate.bank import check_model_configuration

        check_model_configuration(Path(campaign_root), *conditions)


def register_condition(campaign_root: Path, condition: ConditionSpec) -> Path:
    """Write ``<condition_id>/condition.json`` once; refuse a different identity, and a root holding a retired style's
    condition."""
    refuse_retired_root(campaign_root)
    path = check_registered_condition(campaign_root, condition)
    if path.exists():
        return path
    return write_json_sidecar(path, condition_payload(condition))


def condition_of(payload: Mapping[str, Any], where: Any) -> ConditionSpec:
    """Return the condition a ``condition.json`` payload registers, refused when its fingerprint does not match."""
    fields = dict(payload)
    fields.pop("schema_version", None)
    recorded = fields.pop("fingerprint", None)
    condition = ConditionSpec.from_dict(fields)
    if condition.fingerprint != recorded:
        raise SamplingDesignError(
            f"{where}: recorded fingerprint {recorded!r} does not match the "
            "condition it describes"
        )
    return condition


def read_condition(path: Path) -> ConditionSpec:
    """Return the condition a ``condition.json`` registers, refused when its fingerprint does not match."""
    return condition_of(json.loads(Path(path).read_text(encoding="utf-8")), path)


def _in_its_directory(path: Path, campaign_root: Path, condition_id: str) -> None:
    if path.resolve() != (Path(campaign_root) / condition_id / CONDITION_FNAME).resolve():
        raise SamplingDesignError(f"{path} registers {condition_id!r} outside its own directory")


def registers_retired(payload: Mapping[str, Any], path: Path, campaign_root: Path) -> bool:
    """Return whether a ``condition.json`` payload at ``path`` registers a condition of a retired style, refused unless
    its recorded fingerprint, its condition id and its directory are that condition's."""
    if not record_names.is_retired(payload.get("prompt_style"), payload.get("task")):
        return False
    if recorded_fingerprint(payload) != payload.get("fingerprint"):
        raise SamplingDesignError(
            f"{path}: recorded fingerprint {payload.get('fingerprint')!r} does not match the condition it describes")
    _in_its_directory(path, campaign_root,
                      validate_identifier("condition_id", payload.get("condition_id"), allow_slashes=True))
    return True


def _retired_refusal(path: Path, payload: Mapping[str, Any]) -> SamplingDesignError:
    return SamplingDesignError(
        f"{path} registers a condition of a retired style ({payload['prompt_style']!r}): this release reads such a "
        "root's recorded estimates alone and writes nothing into it; a run, registration, merge or analysis beside the "
        "condition would leave it out and change every family its analysis records")


def registration_paths(campaign_root: Path | str) -> list[Path]:
    """Return every ``condition.json`` under ``campaign_root``, sorted: every layout a writer accepts, a registration
    below another's draw directory among them."""
    return sorted(Path(campaign_root).glob(f"**/{CONDITION_FNAME}"))


# The roots a job checked at its entry (checked_at_entry), by resolved path.
_CHECKED_AT_ENTRY: ContextVar[frozenset[str]] = ContextVar("checked_at_entry", default=frozenset())


def refuse_retired_root(campaign_root: Path | str) -> None:
    """Refuse ``campaign_root`` while it registers a condition of a retired style (``registers_retired``). This release
    writes no such condition: only data recorded before it holds one. So two things hold, and no more. No analysis,
    aggregate, gate or bill is computed from such a root: ``load_conditions`` refuses it, and only the experiments'
    read of its recorded estimates (``skip_retired``) leaves the condition out. A campaign job or command pointed at
    it is refused at its entry, before Hydra writes, before its first backend call and before any registration: the
    campaign-root protection (``refuse_protected``), the configured plan and run, registration, the library's writes
    and runs, the shard merges, batch collection and the bank job call this first. A registration planted in the root
    while a job runs is not looked for: within ``checked_at_entry`` of the root the check is not repeated."""
    root = Path(campaign_root)
    if os.path.realpath(root) in _CHECKED_AT_ENTRY.get():
        return
    # A link that resolves to no file registers nothing; the writes' own checks refuse it.
    for path in registration_paths(root):
        if path.is_file() and registers_retired(payload := json.loads(path.read_text(encoding="utf-8")), path, root):
            raise _retired_refusal(path, payload)


@contextmanager
def checked_at_entry(campaign_root: Path | str, *, refused: bool = False) -> Iterator[None]:
    """Refuse ``campaign_root`` once, at a job's entry (unless the caller has just, ``refused``), and let the job's many
    writes into it rest on that check."""
    if not refused:
        refuse_retired_root(campaign_root)
    token = _CHECKED_AT_ENTRY.set(_CHECKED_AT_ENTRY.get() | {os.path.realpath(campaign_root)})
    try:
        yield
    finally:
        _CHECKED_AT_ENTRY.reset(token)


def load_conditions(campaign_root: Path, *, skip_retired: bool = False) -> dict[str, ConditionSpec]:
    """Every registered condition, keyed by id, refusing a fingerprint mismatch and a condition recorded under a
    retired style's key. ``skip_retired`` leaves the latter out instead, for a reader of the root's recorded estimates
    alone: an analysis, extension or re-score of the root without it would change the families it records."""
    conditions: dict[str, ConditionSpec] = {}
    for path in registration_paths(campaign_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if registers_retired(payload, path, campaign_root):
            if not skip_retired:
                raise _retired_refusal(path, payload)
            logger.info("%s: a condition of a retired style, not read", path)
            continue
        condition = condition_of(payload, path)
        _in_its_directory(path, campaign_root, condition.condition_id)
        conditions[condition.condition_id] = condition
    return conditions


def draw_runner(
    plan: CampaignPlan,
    draw: int,
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
    allow_stack_change: bool = False,
) -> DrawRunner:
    """Return the runner of one requested draw of the plan's condition."""
    spec = DrawSpec(
        campaign_id=plan.campaign_id,
        condition=plan.condition,
        decoder_draw_id=draw,
        config=plan.config,
        scenario_manifest=plan.scenario_manifest,
        judge_type=plan.judge_type,
        max_transport_retries=plan.max_transport_retries,
        shard=None if plan.shard is None else plan.shard.draw_shard(draw),
        held_caps=dict(plan.held_caps.get(draw, {})),
    )
    return DrawRunner(
        backend=backend,
        task_spec=task_spec,
        spec=spec,
        draw_dir=plan.condition_root / draw_dirname(draw),
        scenario_rows=plan.scenario_rows,
        rules=plan.rules,
        rule_levels=None if plan.rule_levels is None else plan.rule_levels.by_id,
        code_revision=plan.code_state.get("code_revision"),
        code_dirty=plan.code_state.get("code_dirty"),
        code_state_sha256=plan.code_state.get("code_state_sha256"),
        package_version=package_version(),
        allow_stack_change=allow_stack_change,
    )


@contextmanager
def guarding_batches(roots: Sequence[Path]) -> Iterator[None]:
    """Check every batch work directory a judge's transport derives inside the block, unless the caller already holds a
    guard."""
    if work_directories_guarded():
        yield
        return
    from ruleforge.evaluate.api import CampaignError, batch_work_guard

    checks = [batch_work_guard(root) for root in dict.fromkeys(roots)]

    def guard(batch_root: Path, directory: Path) -> None:
        for check in checks:
            check(batch_root, directory)

    try:
        with guard_work_directories(guard):
            yield
    except BatchDestinationRefused as error:
        raise CampaignError(f"a batch transport's write was refused before it was made: {error}") from error


def run_plan(
    plan: CampaignPlan,
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
    execution: Optional[ExecutionSettings] = None,
    gate: bool = True,
) -> dict[int, dict[str, Any]]:
    """Run every requested draw of the plan's condition and return their completions (``execution.pack_draws``: packed
    calls; a shard or ``gate=False`` writes no campaign gate)."""
    refuse_retired_root(plan.campaign_root)
    with guarding_batches([plan.campaign_root]):
        return _run_plan(plan, backend=backend, task_spec=task_spec, execution=execution, gate=gate)


def _run_plan(
    plan: CampaignPlan,
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
    execution: Optional[ExecutionSettings],
    gate: bool,
) -> dict[int, dict[str, Any]]:
    if execution is not None and execution.pack_draws:
        completions = run_plans_packed(
            [plan], backend=backend, task_spec=task_spec, execution=execution
        )[plan.condition.condition_id]
    else:
        completions = {}
        allow = execution is not None and execution.allow_stack_change
        for draw in plan.decoder_draw_ids:
            completions[draw] = draw_runner(
                plan, draw, backend=backend, task_spec=task_spec, allow_stack_change=allow
            ).run()
            logger.info(
                "Draw %d of %s: %s", draw, plan.condition.condition_id, completions[draw]["state"]
            )
    if gate and plan.shard is None:
        check_run_write(plan.campaign_root)
        campaign_completion(plan.campaign_root)
    return completions


def run_plans_packed(
    plans: Sequence[CampaignPlan],
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
    execution: ExecutionSettings,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Return ``{condition_id: {draw: completion}}``, running draws in chunks of ``execution.pack_max_draws`` through
    the packed driver."""
    for root in dict.fromkeys(plan.campaign_root for plan in plans):
        refuse_retired_root(root)
    with guarding_batches([plan.campaign_root for plan in plans]):
        return _run_plans_packed(plans, backend=backend, task_spec=task_spec, execution=execution)


def _run_plans_packed(
    plans: Sequence[CampaignPlan],
    *,
    backend: ModelBackend,
    task_spec: BaseTaskSpec,
    execution: ExecutionSettings,
) -> dict[str, dict[int, dict[str, Any]]]:
    pairs = [(plan, draw) for plan in plans for draw in plan.decoder_draw_ids]
    size = execution.pack_max_draws or max(len(pairs), 1)
    completions: dict[str, dict[int, dict[str, Any]]] = {
        plan.condition.condition_id: {} for plan in plans
    }
    for start in range(0, len(pairs), size):
        chunk = pairs[start : start + size]
        runners = [
            draw_runner(
                plan, draw, backend=backend, task_spec=task_spec,
                allow_stack_change=execution.allow_stack_change,
            )
            for plan, draw in chunk
        ]
        logger.info(
            "Packed run: draws %d-%d of %d", start + 1, start + len(chunk), len(pairs)
        )
        results = run_packed(runners, max_prompts_per_call=execution.pack_max_prompts)
        for (plan, draw), completion in zip(chunk, results):
            completions[plan.condition.condition_id][draw] = completion
            logger.info(
                "Draw %d of %s: %s", draw, plan.condition.condition_id, completion["state"]
            )
    return completions


def _gate_behind_ledger(
    path: Path, record: Mapping[str, Any], *, manifest: Mapping[str, Any], condition_id: str, draw: int
) -> list[str]:
    state = DrawLedger(
        path.parent, campaign_id=manifest["campaign_id"], condition_id=condition_id, decoder_draw_id=draw
    ).load()
    counts = ledger_counts(state, component_scope=manifest.get("retry_scope") == RETRY_SCOPE_COMPONENT)
    return [
        f"{name}: gate {record.get(name)!r}, ledger {value!r}"
        for name, value in counts.items()
        if record.get(name) != value
    ]


def campaign_completion(
    campaign_root: Path | str, *, write: bool = True, check_ledgers: bool = False
) -> dict[str, Any]:
    """Return the campaign gate over every registered condition and draw, written to ``completion.json`` unless
    ``write`` is false; ``check_ledgers`` refuses a draw whose ledger disagrees."""
    root = Path(campaign_root)
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    conditions = load_conditions(root)
    expected_draws = int(manifest["expected_decoder_draws"])
    per_condition: dict[str, Any] = {}
    states: set[str] = set()
    behind: list[str] = []
    for condition_id, condition in conditions.items():
        draws: dict[str, Any] = {}
        for draw in range(expected_draws):
            path = root / condition_id / draw_dirname(draw) / COMPLETION_FNAME
            if not path.exists():
                draws[str(draw)] = {"state": STATE_IN_PROGRESS, "recorded": False}
                states.add(STATE_IN_PROGRESS)
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            state = str(record.get("state"))
            if (
                record.get("condition_fingerprint") != condition.fingerprint
                or record.get("scenario_manifest_sha256") != manifest["scenario_manifest_sha256"]
                or record.get("campaign_id") != manifest["campaign_id"]
            ):
                state = STATE_INVALID_MANIFEST
            elif check_ledgers:
                fields = _gate_behind_ledger(path, record, manifest=manifest, condition_id=condition_id, draw=draw)
                if fields:
                    behind.append(f"{path} ({'; '.join(fields)})")
            entry = {
                "state": state,
                "recorded": True,
                "accepted": record.get("accepted"),
                "pending": record.get("pending"),
                "attempts_histogram": record.get("attempts_histogram"),
                "rejected_attempts": record.get("rejected_attempts"),
                "transport_failed_attempts": record.get("transport_failed_attempts"),
                "invalid_components": record.get("invalid_components"),
                "exhausted": record.get("exhausted"),
            }
            if "component_attempts_histogram" in record:
                entry["component_attempts_histogram"] = record["component_attempts_histogram"]
            draws[str(draw)] = entry
            states.add(state)
        per_condition[condition_id] = {
            "fingerprint": condition.fingerprint,
            "draws": draws,
        }
    if behind:
        raise CampaignRunError(
            f"{root}: {len(behind)} draw gate(s) behind their ledgers -- a run ended between a block's append and the "
            f"gate's write; resume the draw (it rewrites its gate from the ledger before it sends anything) and read "
            f"again: {', '.join(behind)}"
        )
    if not conditions:
        state = STATE_IN_PROGRESS
    elif STATE_INVALID_MANIFEST in states:
        state = STATE_INVALID_MANIFEST
    elif STATE_INCOMPLETE_EXHAUSTED in states:
        state = STATE_INCOMPLETE_EXHAUSTED
    elif states == {STATE_COMPLETE}:
        state = STATE_COMPLETE
    else:
        state = STATE_IN_PROGRESS
    payload = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "state": state,
        "scenario_manifest_sha256": manifest["scenario_manifest_sha256"],
        "expected_scenarios": manifest["expected_scenarios"],
        "expected_decoder_draws": expected_draws,
        "n_conditions": len(conditions),
        "conditions": per_condition,
        "updated_at": utc_now(),
    }
    if write:
        write_json_sidecar(root / COMPLETION_FNAME, payload)
    return payload


def require_complete_campaign(campaign_root: Path | str, *, write: bool = True) -> dict[str, Any]:
    """Return the campaign gate recomputed now, or raise naming what is not complete."""
    payload = campaign_completion(campaign_root, write=write)
    if payload["state"] != STATE_COMPLETE:
        blocking = {
            cid: {d: r["state"] for d, r in entry["draws"].items() if r["state"] != STATE_COMPLETE}
            for cid, entry in payload["conditions"].items()
        }
        raise CampaignRunError(
            f"campaign {payload['campaign_id']!r} is {payload['state']!r}, not complete; "
            f"nothing is aggregated: {json.dumps({k: v for k, v in blocking.items() if v})}"
        )
    return payload
