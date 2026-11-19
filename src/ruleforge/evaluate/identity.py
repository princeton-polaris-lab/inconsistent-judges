from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ruleforge.evaluate.bootstrap import (
    DEFAULT_CI_LEVEL,
    DEFAULT_RESAMPLES,
    validate_bootstrap_args,
)
from ruleforge.judges import backend as _backend
from ruleforge.prompts.styles import (
    FORCED_JSON_STYLES,
    HIERARCHY_RULE_CONTEXT,
    PREEMPTION_LISTING_STYLES,
    hierarchy_refusal,
)
from ruleforge.tasks.base import (
    BUILTIN_TASKS,
    DEFAULT_TASK,
    FORCED_JSON,
    LABEL,
    RESPONSE_CONTRACTS,
    is_task_identifier,
)
from ruleforge.tasks.registry import BUILTIN_TASK_CLASSES
from ruleforge.core.constants import FULL_RULESET_KEY, PUBLISHED_COPY_FNAME
from ruleforge.core.resume import count_complete_jsonl_records
from ruleforge.core.digests import (
    _HEX64,
    canonical_digest,
    canonical_json,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.identifiers import validate_identifier
from ruleforge.core import record_names
from ruleforge.core.files import write_json_sidecar
from ruleforge.scenarios.manifest import SAMPLING_SCHEMA_VERSION, ScenarioManifest

# Part of the hashed tuple, so two versions can never collide.
SEED_SCHEMA_VERSION = 1

# Bumped when what counts as a valid component changes; in the condition identity, as the estimand depends on it.
VALIDATION_SCHEMA_VERSION = 3

# The recorded name of the one label reader, ``ruleforge.prompts.parse.read_label``. A label condition recorded under
# another reader is refused wherever its answers would be read.
LABEL_RULE = "lenient_last_line"

FINAL_SCENARIO_COUNT = 500
REQUIRED_TEMPERATURE = 1.0
MIN_DECODER_DRAWS = 2
DEFAULT_DECODER_DRAWS = 5
DEFAULT_MAX_SEMANTIC_ATTEMPTS = 5

# What a failed validation regenerates. ``scenario_bundle``: every component of a rejected bundle. ``component``: only
# those that failed, keeping the valid ones. For independent requests both accept the same distribution.
RETRY_SCOPE_SCENARIO_BUNDLE = "scenario_bundle"
RETRY_SCOPE_COMPONENT = "component"
RETRY_SCOPES = (RETRY_SCOPE_SCENARIO_BUNDLE, RETRY_SCOPE_COMPONENT)


def retry_scope_record(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return ``{"retry_scope": ...}`` for a scope that is not the bundle default, else nothing."""
    scope = manifest.get("retry_scope")
    return {} if scope in (None, RETRY_SCOPE_SCENARIO_BUNDLE) else {"retry_scope": scope}

# How temperature 1 is read for a local judge: with the truncation the model ships, or the full distribution.
TRUNCATION_MODEL_DEFAULT = "model_default"
TRUNCATION_NONE = "none"
TRUNCATION_POLICIES = (TRUNCATION_MODEL_DEFAULT, TRUNCATION_NONE)
PROVIDER_DEFAULT_TRUNCATION = "provider_default"

CAMPAIGN_MODE_PILOT = "pilot"
CAMPAIGN_MODE_FINAL = "final"
# One draw over a candidate population, to learn which scenarios every judge can answer; never analysed.
CAMPAIGN_MODE_SCREENING = "screening"
CAMPAIGN_MODES = (CAMPAIGN_MODE_PILOT, CAMPAIGN_MODE_FINAL, CAMPAIGN_MODE_SCREENING)

# ``unsupported`` sends no seed; ``decoder_draw_id`` plus a fresh call identifies a replicate.
SEED_UNSUPPORTED = _backend.SEED_UNSUPPORTED
SEED_LOCAL_DETERMINISTIC = _backend.SEED_LOCAL_DETERMINISTIC
SEED_PROVIDER_BEST_EFFORT = _backend.SEED_PROVIDER_BEST_EFFORT
SEED_CAPABILITIES = _backend.SEED_CAPABILITIES

# ``complete_rule_map`` also requires every non-meta rule exactly once with a valid label, so analyses of the same
# response form exact grids.
FORCED_JSON_COMPLETE_RULE_MAP = "complete_rule_map"
FORCED_JSON_LABEL_ONLY = "label_only"
FORCED_JSON_REQUIREMENTS = (FORCED_JSON_COMPLETE_RULE_MAP, FORCED_JSON_LABEL_ONLY)

FULL_RULESET_COMPONENT = FULL_RULESET_KEY

KIND_FULL_SET = "full_set"
KIND_RULE = "rule"
COMPONENT_KINDS = (KIND_FULL_SET, KIND_RULE)


def component_kind(component_id: str) -> str:
    """``full_set`` for the whole-ruleset judgment, ``rule`` for an individual rule's."""
    return KIND_FULL_SET if component_id == FULL_RULESET_COMPONENT else KIND_RULE


# What one bundle contains: ``both`` the full-set and every individual-rule judgment, ``full`` the full-set judgment
# alone (joined to its draw's bank), ``individual_bank`` every individual-rule judgment.
EVALUATION_MODE_BOTH = "both"
EVALUATION_MODE_FULL = "full"
EVALUATION_MODE_BANK = "individual_bank"
EVALUATION_MODES = (EVALUATION_MODE_BOTH, EVALUATION_MODE_FULL, EVALUATION_MODE_BANK)

SHARED_BANK_DESIGN_FNAME = record_names.SHARED_BANK_DESIGN_FNAME

_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "schema_version",
        "campaign_mode",
        "temperature",
        "num_decoder_draws",
        "max_semantic_attempts",
        "retry_scope",
        "require_complete",
        "generation_seed_base",
        "scenarios_per_request_block",
        "truncation",
        "forced_json_requirement",
        "strict_think",
        "bootstrap",
    }
)
_BOOTSTRAP_KEYS = frozenset({"n_resamples", "ci_level", "seed"})


COMPLETION_FNAME = "completion.json"
BUNDLE_OUTCOMES_FNAME = "bundle_outcomes.jsonl"


class PublishedCopyError(SamplingDesignError):
    """A command that reads more of a campaign root than its published copy holds was given the copy."""


def refuse_published(root: Path | str, command: str, reads: str) -> None:
    """Refuse ``command``, which reads ``reads``, on a published copy of a campaign root (one holding
    :data:`PUBLISHED_COPY_FNAME`): the copy holds only the files, rows and fields the paper's rebuilds read."""
    if (Path(root) / PUBLISHED_COPY_FNAME).is_file():
        raise PublishedCopyError(
            f"{root} is a published copy of a campaign root ({PUBLISHED_COPY_FNAME}): it holds only the files, rows and "
            f"fields the paper's rebuilds read, not {reads}, which {command} reads; run {command} on a campaign root "
            "you ran")
DRAW_PROVENANCE_FNAME = "draw_provenance.json"
# One line per process that ran the draw.
RUNS_FNAME = "runs.jsonl"
# A derived root's record of the root it was re-scored from.
REPLAY_RECORD_FNAME = "label_replay.json"
SAMPLING_MANIFEST_FNAME = "sampling_manifest.json"
# ``completion.json`` key: the complete rows the run log held when the gate was written. A log holding fewer lost a run,
# whose ledger rows would be credited to the run before it. A consistency check, not an authentication.
RUN_LOG_ROWS_KEY = "run_log_rows"


def run_log_rows(draw_dir: Path | str) -> int:
    """The complete rows of a draw's run log (0 without one), as its gate counts them (:data:`RUN_LOG_ROWS_KEY`)."""
    return count_complete_jsonl_records(Path(draw_dir) / RUNS_FNAME)


def run_log_problem(draw_dir: Path | str, *, required: bool = False) -> Optional[str]:
    """Return what is wrong with a draw's run log against its gate, or None; ``required`` also demands a gate that
    counts it."""
    gate = Path(draw_dir) / COMPLETION_FNAME
    counted = json.loads(gate.read_text(encoding="utf-8")).get(RUN_LOG_ROWS_KEY) if gate.is_file() else None
    if counted is None:
        return (f"its {COMPLETION_FNAME} counts no {RUNS_FNAME} rows, so the run log cannot be held to the rows it "
                "describes" if required else None)
    held = run_log_rows(draw_dir)
    if type(counted) is int and held >= counted:
        return None
    return (f"its {RUNS_FNAME} holds {held} complete row(s), and its {COMPLETION_FNAME} counted {counted!r} when it "
            "was written: rows were cut from the run log since, and the ledger rows of a run cut from it would read "
            "as the run's before it, on that run's stack and code state")


def run_log_damage(draw_dir: Path | str) -> Optional[tuple[str, bool]]:
    """Return what follows a run log's complete rows and whether it is a torn trailing row alone, or None."""
    path = Path(draw_dir) / RUNS_FNAME
    if not path.is_file():
        return None
    lines = path.read_bytes().split(b"\n")
    held, terminated, trailing = run_log_rows(draw_dir), len(lines) - 1, lines[-1]
    if held < terminated:
        return (f"its {RUNS_FNAME} holds a malformed row after its {held} complete row(s), and "
                f"{terminated - held - 1 + bool(trailing)} line(s) after it: no reader reads past it (rows logged "
                "behind torn bytes are lost to every reader)", False)
    if trailing:
        return f"its {RUNS_FNAME} ends in a torn row ({len(trailing)} byte(s) after its {held} complete row(s))", True
    return None


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SamplingDesignError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise SamplingDesignError(f"{name} must be at least {minimum}, got {value}")
    return value


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise SamplingDesignError(f"{name} must be true or false, got {value!r}")


def _require_recorded_label_rule(name: str, value: Any) -> None:
    if not (isinstance(value, str) and value):
        raise SamplingDesignError(f"{name} must be a label reader's recorded name, got {value!r}")


@dataclass(frozen=True)
class BootstrapSettings:
    """Post-processing resampling settings; they enter a default campaign id (so its seeds) and nothing else."""

    n_resamples: int = DEFAULT_RESAMPLES
    ci_level: float = DEFAULT_CI_LEVEL
    seed: int = 16

    def __post_init__(self) -> None:
        _require_int("bootstrap.n_resamples", self.n_resamples, minimum=1)
        _require_int("bootstrap.seed", self.seed, minimum=0)
        try:
            validate_bootstrap_args(self.n_resamples, float(self.ci_level))
        except ValueError as error:
            raise SamplingDesignError(f"bootstrap: {error}") from error


@dataclass(frozen=True)
class SamplingConfig:
    """The ``sampling:`` block of a campaign config, validated: a value that describes another experiment is refused."""

    campaign_mode: str = CAMPAIGN_MODE_FINAL
    schema_version: int = SAMPLING_SCHEMA_VERSION
    temperature: float = REQUIRED_TEMPERATURE
    num_decoder_draws: int = DEFAULT_DECODER_DRAWS
    max_semantic_attempts: int = DEFAULT_MAX_SEMANTIC_ATTEMPTS
    retry_scope: str = RETRY_SCOPE_SCENARIO_BUNDLE
    require_complete: bool = True
    generation_seed_base: int = 16
    # ``None`` submits every pending scenario as one block, so a resumed pass re-addresses the paid batch it bought.
    scenarios_per_request_block: Optional[int] = None
    forced_json_requirement: str = FORCED_JSON_COMPLETE_RULE_MAP
    truncation: str = TRUNCATION_MODEL_DEFAULT
    # A manifest's value, never a config's.
    label_rule: str = LABEL_RULE
    strict_think: bool = False
    bootstrap: BootstrapSettings = field(default_factory=BootstrapSettings)

    def __post_init__(self) -> None:
        _require_bool("sampling.strict_think", self.strict_think)
        _require_recorded_label_rule("sampling.label_rule", self.label_rule)
        if self.truncation not in TRUNCATION_POLICIES:
            raise SamplingDesignError(
                f"sampling.truncation must be one of {TRUNCATION_POLICIES}, got "
                f"{self.truncation!r}"
            )
        if self.schema_version != SAMPLING_SCHEMA_VERSION:
            raise SamplingDesignError(
                f"sampling.schema_version {self.schema_version!r} is not "
                f"{SAMPLING_SCHEMA_VERSION}"
            )
        if self.campaign_mode not in CAMPAIGN_MODES:
            raise SamplingDesignError(
                f"sampling.campaign_mode must be one of {CAMPAIGN_MODES}, got "
                f"{self.campaign_mode!r}"
            )
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise SamplingDesignError(
                f"sampling.temperature must be a number, got {self.temperature!r}"
            )
        if float(self.temperature) != REQUIRED_TEMPERATURE:
            raise SamplingDesignError(
                f"sampling.temperature must be exactly {REQUIRED_TEMPERATURE} for a "
                f"stochastic campaign, got {self.temperature!r}"
            )
        # So ``1`` and ``1.0`` yield one manifest and one fingerprint.
        object.__setattr__(self, "temperature", float(self.temperature))
        _require_int(
            "sampling.num_decoder_draws", self.num_decoder_draws,
            minimum=1 if self.campaign_mode == CAMPAIGN_MODE_SCREENING else MIN_DECODER_DRAWS,
        )
        _require_int("sampling.max_semantic_attempts", self.max_semantic_attempts,
                     minimum=1)
        if self.retry_scope not in RETRY_SCOPES:
            raise SamplingDesignError(
                f"sampling.retry_scope must be one of {RETRY_SCOPES}; "
                f"{self.retry_scope!r} is not an implemented policy"
            )
        if self.require_complete is not True:
            raise SamplingDesignError(
                "sampling.require_complete must be true: an incomplete grid is "
                "never aggregated, so a config saying otherwise is misleading"
            )
        _require_int("sampling.generation_seed_base", self.generation_seed_base,
                     minimum=0)
        if self.scenarios_per_request_block is not None:
            _require_int(
                "sampling.scenarios_per_request_block",
                self.scenarios_per_request_block,
                minimum=1,
            )
        if self.forced_json_requirement not in FORCED_JSON_REQUIREMENTS:
            raise SamplingDesignError(
                "sampling.forced_json_requirement must be one of "
                f"{FORCED_JSON_REQUIREMENTS}, got {self.forced_json_requirement!r}"
            )
        if not isinstance(self.bootstrap, BootstrapSettings):
            raise SamplingDesignError("sampling.bootstrap must be BootstrapSettings")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SamplingConfig":
        """Build from a config block, refusing unknown keys."""
        if not isinstance(mapping, Mapping):
            raise SamplingDesignError("sampling block must be a mapping")
        unknown = sorted(set(mapping) - _CONFIG_KEYS)
        if unknown:
            raise SamplingDesignError(f"unknown sampling keys: {unknown}")
        if mapping.get("enabled", True) is not True:
            raise SamplingDesignError(
                "sampling.enabled must be true to build a campaign from this block"
            )
        raw_bootstrap = mapping.get("bootstrap", {})
        if not isinstance(raw_bootstrap, Mapping):
            raise SamplingDesignError("sampling.bootstrap must be a mapping")
        unknown_bootstrap = sorted(set(raw_bootstrap) - _BOOTSTRAP_KEYS)
        if unknown_bootstrap:
            raise SamplingDesignError(
                f"unknown sampling.bootstrap keys: {unknown_bootstrap}"
            )
        kwargs = {
            key: value
            for key, value in mapping.items()
            if key not in ("enabled", "bootstrap")
        }
        return cls(bootstrap=BootstrapSettings(**dict(raw_bootstrap)), **kwargs)

    @property
    def expected_scenario_count(self) -> Optional[int]:
        """Exactly 500 in a final campaign; a pilot declares its own count."""
        return FINAL_SCENARIO_COUNT if self.campaign_mode == CAMPAIGN_MODE_FINAL else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Left out of the payload at these values, keeping the fingerprints of conditions that never set them.
_OMITTED_AT_DEFAULT: dict[str, Any] = {
    "adapter_sha256": None,
    "adapter_config_sha256": None,
    "model_weights_sha256": None,
    "serving_dtype": None,
    "retry_scope": RETRY_SCOPE_SCENARIO_BUNDLE,
    "strict_think": False,
    "rule_levels_sha256": None,
    "task": DEFAULT_TASK,
    "response_contract": None,
    "judge_extension": None,
}

BUILTIN_MODEL_TYPES = _backend.BUILTIN_MODEL_TYPES


@dataclass(frozen=True)
class ConditionSpec:
    """Everything that makes two judgments comparable as the same condition: ``fingerprint`` digests every field but
    ``condition_id``, the operator's label; a draw recorded under another fingerprint is refused."""

    condition_id: str
    model_name: str
    model_type: str
    prompt_style: str
    individual_rule_context: str
    judge_type: str
    evaluation_mode: str
    rule_set_name: str
    rule_set_sha256: str
    temperature: float
    prompt_shape_version: int
    forced_json_requirement: str
    validation_schema_version: int = VALIDATION_SCHEMA_VERSION
    label_rule: str = LABEL_RULE
    retry_scope: str = RETRY_SCOPE_SCENARIO_BUNDLE
    strict_think: bool = False
    endpoint: Optional[str] = None
    reasoning_effort: Optional[str] = None
    max_new_tokens: Optional[int] = None
    send_max_output_tokens: Optional[bool] = None
    seed_capability: str = SEED_UNSUPPORTED
    # ``model_name`` is a free-text label, so these keep two checkpoints under one label apart.
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    adapter_sha256: Optional[str] = None
    adapter_config_sha256: Optional[str] = None
    # Derived weights by content; None for upstream weights, named by their directory.
    model_weights_sha256: Optional[str] = None
    # ``float32``, ``bfloat16``, ``float16``; None for vLLM's ``auto``.
    serving_dtype: Optional[str] = None
    enable_thinking: Optional[bool] = None
    # For a local judge the resolved truncation values as sorted pairs, for a provider ``provider_default``.
    sampling_truncation: Any = None
    rule_levels_sha256: Optional[str] = None
    # The task's name, or ``name@version``.
    task: str = DEFAULT_TASK
    # ``label`` or ``forced_json`` for a task that is not built in; None otherwise.
    response_contract: Optional[str] = None
    judge_extension: Optional[str] = None
    # Sorted pairs, so the digest is order-free.
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        validate_identifier("condition_id", self.condition_id, allow_slashes=True)
        if self.evaluation_mode not in EVALUATION_MODES:
            raise SamplingDesignError(
                f"evaluation_mode must be one of {EVALUATION_MODES}, got "
                f"{self.evaluation_mode!r}"
            )
        if isinstance(self.sampling_truncation, Mapping):
            object.__setattr__(
                self, "sampling_truncation",
                tuple(sorted((str(k), v) for k, v in self.sampling_truncation.items())),
            )
        elif isinstance(self.sampling_truncation, (list, tuple)):
            object.__setattr__(
                self, "sampling_truncation",
                tuple(sorted((str(k), v) for k, v in self.sampling_truncation)),
            )
        elif self.sampling_truncation is not None and not isinstance(self.sampling_truncation, str):
            raise SamplingDesignError(
                "sampling_truncation must be a mapping, a string or None, got "
                f"{type(self.sampling_truncation).__name__}"
            )
        if self.seed_capability not in SEED_CAPABILITIES:
            raise SamplingDesignError(
                f"seed_capability must be one of {SEED_CAPABILITIES}, got "
                f"{self.seed_capability!r}"
            )
        if self.forced_json_requirement not in FORCED_JSON_REQUIREMENTS:
            raise SamplingDesignError(
                f"forced_json_requirement must be one of {FORCED_JSON_REQUIREMENTS}, "
                f"got {self.forced_json_requirement!r}"
            )
        _require_recorded_label_rule("label_rule", self.label_rule)
        if self.retry_scope not in RETRY_SCOPES:
            raise SamplingDesignError(f"retry_scope must be one of {RETRY_SCOPES}, got {self.retry_scope!r}")
        _require_bool("strict_think", self.strict_think)
        self._validate_task()
        self._validate_judge()
        self._validate_adapter()
        self._validate_weights()
        self._validate_levels()
        if float(self.temperature) != REQUIRED_TEMPERATURE:
            raise SamplingDesignError(
                f"condition {self.condition_id!r} declares temperature "
                f"{self.temperature!r}; a stochastic campaign runs at "
                f"{REQUIRED_TEMPERATURE}"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(
            self, "extra", tuple(sorted((str(k), v) for k, v in self.extra))
        )

    def _validate_task(self) -> None:
        if not is_task_identifier(self.task):
            raise SamplingDesignError(f"task must be a task name or name@version, got {self.task!r}")
        if self.task in BUILTIN_TASKS and self.prompt_style not in BUILTIN_TASK_CLASSES[self.task].styles:
            public = record_names.renamed_to(self.prompt_style)
            pointer = (f"; {self.prompt_style!r} is the key the published records name the style {public!r} by: "
                       "read a recorded condition through ConditionSpec.from_dict, which maps it" if public else
                       f"; {self.prompt_style!r} is a retired style's key: its experiment is not part of the release"
                       if self.prompt_style in record_names.RETIRED_KEYS else "")
            raise SamplingDesignError(
                f"condition {self.condition_id!r}: task {self.task!r} has no prompt style {self.prompt_style!r} (its "
                f"styles: {list(BUILTIN_TASK_CLASSES[self.task].styles)}){pointer}")
        segments = record_names.record_key_segments(self.condition_id, self.prompt_style, self.task)
        if segments:
            raise SamplingDesignError(
                f"condition {self.condition_id!r}: its id holds {list(segments)}, the key"
                f"{'s' if len(segments) > 1 else ''} the published records name a style by, as a segment, and not as "
                "its own style's; name the condition otherwise (a condition's id names its style by the style's "
                "public name)")
        if self.response_contract is not None and self.response_contract not in RESPONSE_CONTRACTS:
            raise SamplingDesignError(
                f"response_contract must be one of {RESPONSE_CONTRACTS} or None, got {self.response_contract!r}")
        if (self.response_contract is None) != (self.task in BUILTIN_TASKS):
            raise SamplingDesignError(
                f"condition {self.condition_id!r}: task {self.task!r} with response_contract "
                f"{self.response_contract!r}; a task that is not built in ({list(BUILTIN_TASKS)}) records its "
                "style's response contract, and a built-in one records none")

    def _validate_judge(self) -> None:
        value = self.judge_extension
        if value is not None and not (isinstance(value, str) and value):
            raise SamplingDesignError(f"judge_extension must be a non-empty string or None, got {value!r}")
        if (self.judge_extension is None) != (self.model_type in BUILTIN_MODEL_TYPES):
            raise SamplingDesignError(
                f"condition {self.condition_id!r}: model_type {self.model_type!r} with judge_extension "
                f"{self.judge_extension!r}; a judge of a model type the backend does not serve "
                f"({list(BUILTIN_MODEL_TYPES)}) records its options as its extension, and a built-in one records "
                "none")

    @property
    def contract(self) -> str:
        """How this condition's answers are read: the recorded contract, else the built-in style's."""
        if self.response_contract is not None:
            return self.response_contract
        return FORCED_JSON if self.prompt_style in FORCED_JSON_STYLES else LABEL

    @property
    def lists_preemption(self) -> bool:
        """Whether a forced answer's ``preempted_rules`` is read: under the built-in style that asks for it alone."""
        return self.response_contract is None and self.prompt_style in PREEMPTION_LISTING_STYLES

    def _validate_adapter(self) -> None:
        digests = (self.adapter_sha256, self.adapter_config_sha256)
        if any(d is not None and not (isinstance(d, str) and _HEX64.match(d)) for d in digests):
            raise SamplingDesignError(
                f"adapter digests must be SHA-256 hex digests, got {digests!r}"
            )
        if (self.adapter_sha256 is None) != (self.adapter_config_sha256 is None):
            raise SamplingDesignError(
                "adapter_sha256 and adapter_config_sha256 are recorded together or not at all"
            )
        if self.adapter_sha256 is not None and self.adapter_path is None:
            raise SamplingDesignError(
                f"condition {self.condition_id!r} carries adapter digests but no adapter_path"
            )

    def _validate_weights(self) -> None:
        digest = self.model_weights_sha256
        if digest is not None and not (isinstance(digest, str) and _HEX64.match(digest)):
            raise SamplingDesignError(f"model_weights_sha256 must be a SHA-256 hex digest, got {digest!r}")
        if digest is not None and self.model_path is None:
            raise SamplingDesignError(
                f"condition {self.condition_id!r} carries a model weights digest but no model_path"
            )
        if self.serving_dtype is not None and self.serving_dtype not in _backend.SERVING_DTYPES:
            raise SamplingDesignError(
                f"serving_dtype must be one of {_backend.SERVING_DTYPES} (None: vLLM's auto), got "
                f"{self.serving_dtype!r}"
            )

    def _validate_levels(self) -> None:
        digest = self.rule_levels_sha256
        if digest is not None and not (isinstance(digest, str) and _HEX64.match(digest)):
            raise SamplingDesignError(f"rule_levels_sha256 must be a SHA-256 hex digest, got {digest!r}")
        if (self.individual_rule_context == HIERARCHY_RULE_CONTEXT) != (digest is not None):
            raise SamplingDesignError(
                f"condition {self.condition_id!r}: individual_rule_context {self.individual_rule_context!r} with "
                f"rule_levels_sha256 {digest!r}; a {HIERARCHY_RULE_CONTEXT!r} condition names its levels file by "
                "content, and no other condition reads one"
            )
        refusal = hierarchy_refusal(self.prompt_style, self.individual_rule_context)
        if refusal is not None and self.response_contract is None:
            raise SamplingDesignError(f"condition {self.condition_id!r}: {refusal}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extra"] = {key: value for key, value in self.extra}
        if isinstance(self.sampling_truncation, tuple):
            payload["sampling_truncation"] = dict(self.sampling_truncation)
        for key, default in _OMITTED_AT_DEFAULT.items():
            if payload[key] == default:
                del payload[key]
        return payload

    @property
    def fingerprint(self) -> str:
        """Return the fingerprint of ``to_dict``: see :func:`recorded_fingerprint`."""
        return recorded_fingerprint(self.to_dict())

    @property
    def identity_id(self) -> str:
        """Return the id this condition's seeds, request block ids and draw grids are hashed over."""
        return record_names.identity_condition_id(self.condition_id, self.prompt_style, self.task)

    @property
    def public_id(self) -> str:
        """Return the id a result names this condition by, with recorded style keys replaced by public names."""
        return record_names.public_condition_id(self.condition_id, self.prompt_style, self.task)

    def identity_block_id(self, campaign_id: str, block_id: str) -> str:
        """Return the id a provider's request namespace is taken over for a request block."""
        return record_names.identity_block_id(block_id, campaign_id=campaign_id, condition_id=self.condition_id,
                                               prompt_style=self.prompt_style, task=self.task)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConditionSpec":
        """Return the condition ``payload`` describes, reading a recorded style key under its public name."""
        data = dict(payload)
        extra = data.pop("extra", {}) or {}
        style = data.get("prompt_style")
        public = record_names.public_style(style, data.get("task", DEFAULT_TASK)) if isinstance(style, str) else style
        if public != style:
            if public in str(data.get("condition_id")).split("/"):
                raise SamplingDesignError(
                    f"condition {data.get('condition_id')!r} is recorded under the style key {style!r} and holds that "
                    f"style's public name {public!r} as a segment: its identity id would not be the id it was "
                    "recorded under")
            data["prompt_style"] = public
        try:
            return cls(extra=tuple(dict(extra).items()), **data)
        except TypeError as error:
            raise SamplingDesignError(f"malformed condition spec: {error}") from error


def recorded_fingerprint(fields: Mapping[str, Any]) -> str:
    """Return the canonical digest of a condition's fields without ``condition_id`` and the envelope (``schema_version``,
    ``fingerprint``), with the style under its identity key: a condition's fingerprint, derived from its recorded fields
    alone for one this release does not construct."""
    payload = {key: value for key, value in fields.items()
               if key not in ("condition_id", "schema_version", "fingerprint")}
    payload["prompt_style"] = record_names.identity_key(payload.get("prompt_style"), payload.get("task"))
    return canonical_digest(payload)


def derive_generation_seed(
    *,
    generation_seed_base: int,
    campaign_id: str,
    condition_id: str,
    scenario_id: str,
    decoder_draw_id: int,
    semantic_attempt_id: int,
    component_id: str,
) -> int:
    """Return a stable 31-bit seed from the canonical, versioned identity tuple; every axis changes it."""
    payload = [
        SEED_SCHEMA_VERSION,
        int(generation_seed_base),
        str(campaign_id),
        str(condition_id),
        str(scenario_id),
        int(decoder_draw_id),
        int(semantic_attempt_id),
        str(component_id),
    ]
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def draw_dirname(decoder_draw_id: int) -> str:
    """``draw-000`` style directory name for one decoder draw."""
    return f"draw-{_require_int('decoder_draw_id', decoder_draw_id, minimum=0):03d}"


def request_block_id(
    *,
    campaign_id: str,
    condition_id: str,
    decoder_draw_id: int,
    semantic_attempt_id: int,
    block_index: int,
) -> str:
    """Return the identity of one request block, so a resumed pass re-addresses it and a new attempt or draw never
    can."""
    return (
        f"{campaign_id}/{condition_id}/{draw_dirname(decoder_draw_id)}/"
        f"attempt-{_require_int('semantic_attempt_id', semantic_attempt_id, minimum=0)}/"
        f"block-{_require_int('block_index', block_index, minimum=0):04d}"
    )


@dataclass(frozen=True)
class GridCheck:
    """What an exact ``scenario x decoder_draw`` grid check found."""

    expected_keys: int
    observed_keys: int
    missing: tuple[tuple[str, int], ...]
    duplicates: tuple[tuple[str, int], ...]
    unexpected: tuple[tuple[str, int], ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicates or self.unexpected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "expected_keys": self.expected_keys,
            "observed_keys": self.observed_keys,
            "missing": [list(key) for key in self.missing],
            "duplicates": [list(key) for key in self.duplicates],
            "unexpected": [list(key) for key in self.unexpected],
        }


def check_exact_grid(
    expected_scenario_ids: Sequence[str],
    expected_decoder_draws: int,
    observed_keys: Iterable[tuple[str, int]],
) -> GridCheck:
    """Compare accepted ``(scenario_id, decoder_draw_id)`` keys to the design; a repeated or extra key is a failure."""
    _require_int("expected_decoder_draws", expected_decoder_draws, minimum=1)
    expected = {
        (sid, draw)
        for sid in expected_scenario_ids
        for draw in range(expected_decoder_draws)
    }
    counts: dict[tuple[str, int], int] = {}
    for key in observed_keys:
        normalized = (str(key[0]), int(key[1]))
        counts[normalized] = counts.get(normalized, 0) + 1
    missing = sorted(expected - set(counts))
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    unexpected = sorted(set(counts) - expected)
    return GridCheck(
        expected_keys=len(expected),
        observed_keys=sum(counts.values()),
        missing=tuple(missing),
        duplicates=tuple(duplicates),
        unexpected=tuple(unexpected),
    )


# Manifest fields that describe the run, which may differ on resume.
MANIFEST_RUN_FIELDS = ("created_at", "code_revision", "code_dirty", "code_state_sha256", "package_version")

# A manifest's record of a design field changed after freezing; the field itself carries the amended value.
MANIFEST_AMENDMENTS_KEY = "amendments"

# An analysis's record of the fallback: one exhausted scenario dropped from every arm of one judge.
SCENARIO_EXCLUSION_KEY = "scenario_exclusion"


_CONDITION_CONFIG_FIELDS = ("forced_json_requirement", "retry_scope", "strict_think")

# Recorded only when on.
_MANIFEST_FLAGS = ("strict_think",)


def build_sampling_manifest(
    *,
    campaign_id: str,
    config: SamplingConfig,
    scenario_manifest: ScenarioManifest,
    conditions: Sequence[ConditionSpec],
    created_at: str,
    code_revision: Optional[str],
    code_dirty: Optional[bool] = None,
    code_state_sha256: Optional[str] = None,
    package_version: Optional[str] = None,
) -> dict[str, Any]:
    """Return the campaign-level record of what was asked for: per-condition settings, the scenario digest and the seed
    scheme."""
    validate_identifier("campaign_id", campaign_id)
    labels = [condition.condition_id for condition in conditions]
    if len(set(labels)) != len(labels):
        raise SamplingDesignError(f"duplicate condition ids: {sorted(labels)}")
    for condition in conditions:
        for name in _CONDITION_CONFIG_FIELDS:
            if getattr(condition, name) != getattr(config, name):
                raise SamplingDesignError(
                    f"condition {condition.condition_id!r} declares "
                    f"{name}={getattr(condition, name)!r} "
                    f"but the campaign declares {getattr(config, name)!r}"
                )
    return {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "seed_schema_version": SEED_SCHEMA_VERSION,
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "campaign_mode": config.campaign_mode,
        "scenario_manifest_sha256": scenario_manifest.manifest_sha256,
        "scenario_source_sha256": scenario_manifest.source_sha256,
        "expected_scenarios": scenario_manifest.record_count,
        "expected_decoder_draws": config.num_decoder_draws,
        "temperature": config.temperature,
        "max_semantic_attempts": config.max_semantic_attempts,
        "retry_scope": config.retry_scope,
        "generation_seed_base": config.generation_seed_base,
        "scenarios_per_request_block": config.scenarios_per_request_block,
        "forced_json_requirement": config.forced_json_requirement,
        "truncation": config.truncation,
        "label_rule": config.label_rule,
        **{name: True for name in _MANIFEST_FLAGS if getattr(config, name)},
        "bootstrap": asdict(config.bootstrap),
        "conditions": [
            {**condition.to_dict(), "fingerprint": condition.fingerprint}
            for condition in conditions
        ],
        "code_revision": code_revision,
        "code_dirty": code_dirty,
        "code_state_sha256": code_state_sha256,
        **({"package_version": package_version} if package_version is not None else {}),
        "created_at": created_at,
    }


def sampling_config_from_manifest(manifest: Mapping[str, Any]) -> SamplingConfig:
    """Return the ``SamplingConfig`` a campaign manifest was built from; a manifest short of a field is refused."""
    try:
        bootstrap = BootstrapSettings(**dict(manifest["bootstrap"]))
        return SamplingConfig(
            campaign_mode=manifest["campaign_mode"],
            schema_version=manifest["schema_version"],
            temperature=manifest["temperature"],
            num_decoder_draws=manifest["expected_decoder_draws"],
            max_semantic_attempts=manifest["max_semantic_attempts"],
            retry_scope=manifest["retry_scope"],
            generation_seed_base=manifest["generation_seed_base"],
            scenarios_per_request_block=manifest["scenarios_per_request_block"],
            forced_json_requirement=manifest["forced_json_requirement"],
            truncation=manifest["truncation"],
            label_rule=manifest["label_rule"],
            **{name: manifest.get(name, False) for name in _MANIFEST_FLAGS},
            bootstrap=bootstrap,
        )
    except (KeyError, TypeError) as error:
        raise SamplingDesignError(
            f"sampling manifest does not record a whole campaign config: {error!r}"
        ) from error


def differing_manifest_fields(existing: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    """The experiment fields two campaign manifests disagree on (run fields ignored)."""
    return sorted(
        key
        for key in set(existing) | set(manifest)
        if key not in MANIFEST_RUN_FIELDS and existing.get(key) != manifest.get(key)
    )


def check_sampling_manifest(path: Path | str, manifest: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Return None when nothing is frozen, the frozen manifest when it describes this campaign, and raise otherwise;
    writes nothing."""
    destination = Path(path)
    if not destination.exists():
        return None
    try:
        existing = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SamplingDesignError(
            f"{destination} is unreadable ({error}); refusing to overwrite the only "
            "record of what this campaign asked for"
        ) from error
    if not isinstance(existing, Mapping):
        raise SamplingDesignError(f"{destination} is not a JSON object")
    if existing.get("schema_version") != SAMPLING_SCHEMA_VERSION:
        raise SamplingDesignError(
            f"{destination} was written by schema_version "
            f"{existing.get('schema_version')!r}; this build writes "
            f"{SAMPLING_SCHEMA_VERSION}"
        )
    amendments = existing.get(MANIFEST_AMENDMENTS_KEY) or []
    differing = differing_manifest_fields(
        {key: value for key, value in existing.items() if key != MANIFEST_AMENDMENTS_KEY}, manifest
    )
    if differing:
        amended = sorted({str(record.get("field")) for record in amendments} & set(differing))
        raise SamplingDesignError(
            f"{destination} describes a different campaign ({', '.join(differing)} "
            "differ); use a fresh campaign root rather than relabelling its draws"
            + (f". Amended after it was frozen ({MANIFEST_AMENDMENTS_KEY}): "
               + ", ".join(f"{field}={existing.get(field)!r}" for field in amended)
               + "; a job resumes it stating those values" if amended else "")
        )
    return dict(existing)


def ensure_sampling_manifest(path: Path | str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Write the manifest once; on a resume, refuse one that describes another campaign."""
    existing = check_sampling_manifest(path, manifest)
    if existing is None:
        write_json_sidecar(path, manifest)
        return dict(manifest)
    return existing


def load_sampling_manifest(path: Path | str) -> dict[str, Any]:
    """Return a campaign manifest, refusing another schema version."""
    return read_sampling_manifest(path)[0]


def read_sampling_manifest(path: Path | str) -> tuple[dict[str, Any], str]:
    """Return a campaign manifest and the SHA-256 of the bytes it was parsed from."""
    source = Path(path)
    try:
        data = source.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SamplingDesignError(f"cannot read sampling manifest {source}: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SAMPLING_SCHEMA_VERSION:
        raise SamplingDesignError(
            f"{source} is not a schema_version {SAMPLING_SCHEMA_VERSION} sampling manifest"
        )
    return dict(payload), hashlib.sha256(data).hexdigest()
