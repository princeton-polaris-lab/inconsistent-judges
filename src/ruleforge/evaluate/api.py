from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import stat
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, TypeVar, Union

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ruleforge.experiments.table import published_runs
from ruleforge.evaluate.inference import load_condition_outcomes
from ruleforge.evaluate.results import CampaignResult, EvalResult, Kappa, contrasts_of, estimates_of
from ruleforge.judges import Judge, JudgeError, JudgeSpec, resolve
from ruleforge.judges.backend import COMPATIBLE_MODEL_TYPE
from ruleforge.core.paths import (
    ARTIFACT_ROOT_ENV,
    ARTIFACT_SCHEME,
    made_on_the_way,
    to_uri,
    walk_strictly,
)
from ruleforge.env import env_value
from ruleforge.core.errors import ArtifactRootError
from ruleforge.rulesets import Ruleset, RulesetError
from ruleforge.evaluate.aggregate import (
    AGGREGATE_DIRNAME,
    ESTIMATES_FNAME,
    analyze_campaign,
    attempt_statistics,
    fallback_unserved,
    refuse_stripped,
)
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    CampaignPlan,
    campaign_completion,
    campaign_manifest,
    check_judge_configuration,
    check_new_condition_id,
    check_registered_condition,
    condition_spec_for,
    expected_requests_per_draw,
    load_conditions,
    preflight_judge,
    checked_at_entry,
    refuse_retired_root,
    register_condition,
    run_plan,
)
from ruleforge.evaluate.configured import MARKER_FNAME as CONFIGURED_MARKER_FNAME
from ruleforge.evaluate.identity import (
    BUNDLE_OUTCOMES_FNAME,
    CAMPAIGN_MODE_PILOT,
    COMPLETION_FNAME,
    EVALUATION_MODE_BOTH,
    EVALUATION_MODES,
    SHARED_BANK_DESIGN_FNAME,
    FORCED_JSON_COMPLETE_RULE_MAP,
    MANIFEST_AMENDMENTS_KEY,
    RETRY_SCOPE_COMPONENT,
    SAMPLING_MANIFEST_FNAME,
    SCENARIO_EXCLUSION_KEY,
    TRUNCATION_MODEL_DEFAULT,
    BootstrapSettings,
    ConditionSpec,
    SamplingConfig,
    check_sampling_manifest,
    draw_dirname,
    ensure_sampling_manifest,
    load_sampling_manifest,
    sampling_config_from_manifest,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import canonical_json
from ruleforge.core.identifiers import validate_identifier
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.discrimination import (
    KAPPA_STRATUM_FIELD,
    DiscriminationError,
    VerdictGrids,
    kappa_summary,
    read_strata,
)
from ruleforge.evaluate.packing import PACKABLE_MODEL_TYPE, ExecutionSettings
from ruleforge.core.provenance import SNAPSHOT_DIRNAME, code_state, snapshot_path, write_code_snapshot
from ruleforge.evaluate.runner import (
    DEFAULT_MAX_TRANSPORT_RETRIES,
    STATE_COMPLETE,
    STATE_INCOMPLETE_EXHAUSTED,
    STATE_INVALID_MANIFEST,
    CampaignRunError,
    draw_ended,
    guard_run_writes,
    hold_shared_writes,
    refuse_fake_in_final,
    utc_now,
)
from ruleforge.evaluate.sharding import campaign_lock, campaign_lock_path
from ruleforge.evaluate.stack import CONDITION_STACK_FNAME, STACK_MISMATCHES_KEY
from ruleforge.scenarios import ScenarioSet
from ruleforge.tasks.spec import BaseTaskSpec, TaskSpecAdapter
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT
from ruleforge.tasks.spec import build_task_spec
from ruleforge.tasks import DEFAULT_TASK, Task, TaskError, TaskSettings, load_task, task_locator
from ruleforge.judges.batch_api import BatchDestinationRefused, guard_work_directories, spend_confirmed
from ruleforge.rulesets.levels import RuleLevels, RuleLevelsError, load_rule_levels

_TASK_IDENTITY_FIELDS = ("task", "prompt_style", "individual_rule_context", "judge_type", "response_contract")
# A started judge also holds these itself; each must match its identity.
_LIVE_IDENTITY_FIELDS = ("model_name", "model_type", "seed_capability", "max_new_tokens")
# Marks a root this library created; a root without it is never written.
MARKER_FNAME = ".ruleforge-campaign"
MARKER_KIND = "ruleforge-campaign"
MARKER_SCHEMA_VERSION = 1
# What makes a directory a campaign root to every guard of a place a write may land.
ROOT_MARKERS = (SAMPLING_MANIFEST_FNAME, MARKER_FNAME, CONFIGURED_MARKER_FNAME)
LOCK_FNAME = ".ruleforge-campaign.lock"
RUN_LOCK_FNAME = ".ruleforge-run.lock"
EXECUTION_FNAME = "execution.json"
DEFAULT_BATCH_DIRNAME = "batch"
# The root's own records at its top; no batch state or condition lies at or below one.
ROOT_RECORDS = (LOCK_FNAME, MARKER_FNAME, SAMPLING_MANIFEST_FNAME, SCENARIO_MANIFEST_FNAME, COMPLETION_FNAME,
                AGGREGATE_DIRNAME, SNAPSHOT_DIRNAME, SHARED_BANK_DESIGN_FNAME)
CONDITION_RECORDS = (CONDITION_FNAME, EXECUTION_FNAME, RUN_LOCK_FNAME, CONDITION_STACK_FNAME)
DRAW_RECORD = re.compile(r"\.?draw-\d{3,}(\.lock)?")
# Under the working directory: ``evaluate``'s roots when no ``out`` is given and no artifact root is set.
DEFAULT_RUNS_DIRNAME = "ruleforge-runs"
_Read = TypeVar("_Read")

logger = logging.getLogger(__name__)


class CampaignError(CampaignRunError):
    """A campaign operation refused."""


class ReadOnlyCampaign(CampaignError):
    """A write asked of a campaign opened for reading (``Campaign.open``)."""


class AnalysisExists(CampaignError):
    """An analysis asked of a root that holds one, without ``replace``."""


@dataclass(frozen=True)
class SamplingSettings:
    """A campaign's sampling, one per root."""

    draws: int = 5
    max_attempts: int = 30
    retry_scope: str = RETRY_SCOPE_COMPONENT
    forced_json: str = FORCED_JSON_COMPLETE_RULE_MAP
    truncation: str = TRUNCATION_MODEL_DEFAULT
    strict_think: bool = False
    block_size: Optional[int] = None
    seed_base: int = 16
    bootstrap: BootstrapSettings = field(default_factory=BootstrapSettings)
    campaign_mode: str = CAMPAIGN_MODE_PILOT
    approved_scenario_count: Optional[int] = None
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES

    def __post_init__(self) -> None:
        for name in ("approved_scenario_count", "max_transport_retries"):
            value = getattr(self, name)
            if value is None and name == "approved_scenario_count":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CampaignError(f"sampling.{name} must be a positive integer, got {value!r}")
        self.config()

    def config(self) -> SamplingConfig:
        """The ``SamplingConfig`` these settings describe (validated as a configured campaign's)."""
        try:
            return SamplingConfig(
                campaign_mode=self.campaign_mode, num_decoder_draws=self.draws,
                max_semantic_attempts=self.max_attempts, retry_scope=self.retry_scope,
                generation_seed_base=self.seed_base, scenarios_per_request_block=self.block_size,
                forced_json_requirement=self.forced_json, truncation=self.truncation, strict_think=self.strict_think, bootstrap=self.bootstrap)
        except SamplingDesignError as error:
            raise CampaignError(str(error)) from error

    @classmethod
    def from_config(cls, config: SamplingConfig, *,
                    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES) -> "SamplingSettings":
        """The settings a ``SamplingConfig`` (a manifest's, a configured campaign's) describes."""
        return cls(draws=config.num_decoder_draws, max_attempts=config.max_semantic_attempts,
                   retry_scope=config.retry_scope,
                   forced_json=config.forced_json_requirement, truncation=config.truncation,
                   strict_think=config.strict_think, block_size=config.scenarios_per_request_block,
                   seed_base=config.generation_seed_base, bootstrap=config.bootstrap,
                   campaign_mode=config.campaign_mode, max_transport_retries=max_transport_retries)


@dataclass(frozen=True)
class Approvals:
    """The content a condition's inputs must have, pinned before it registers; None pins nothing."""

    rule_set_sha256: Optional[str] = None
    rule_levels_sha256: Optional[str] = None
    adapter_sha256: Optional[str] = None
    model_weights_sha256: Optional[str] = None


@dataclass(frozen=True)
class ConditionSettings:
    """One condition's task settings: task, prompt style, interpretation, rule context, evaluation mode, extras and
    approvals."""

    task: Union[str, Task, type] = DEFAULT_TASK
    prompt: Optional[str] = None
    interpretation: str = "No_Interpretation"
    rule_context: str = "isolated"
    rule_levels: Optional[Union[Path, str]] = None
    mode: str = EVALUATION_MODE_BOTH
    extra: Mapping[str, Any] = field(default_factory=dict, hash=False)
    approvals: Approvals = field(default_factory=Approvals)


@dataclass(frozen=True)
class Condition:
    """A condition registered or rebound in this process: its identity, and the judge, rules and settings bound to
    it."""

    condition_id: str
    spec: ConditionSpec
    judge: JudgeSpec
    rules: Ruleset
    settings: ConditionSettings
    task_locator: str
    task: Optional[Task] = None
    levels: Optional[RuleLevels] = None
    campaign: Optional["Campaign"] = field(default=None, repr=False, compare=False)
    _rule_rows: "_Rows" = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rule_rows", _Rows.of(self.rules.rows()))

    def judged_rules(self) -> tuple[dict[str, Any], ...]:
        """Return the rules a run judges: those bound, refused when ``rules`` has changed since."""
        return self._rule_rows.judged(self.rules.rows(), f"condition {self.condition_id!r}'s rules "
                                                         f"({self.rules.name})")

    def run(self, **kwargs: Any) -> dict[int, dict[str, Any]]:
        """Run this condition alone; return its completions by draw."""
        if self.campaign is None:
            raise CampaignError(f"condition {self.condition_id!r} is bound to no campaign; register it with "
                                "Campaign.condition")
        return self.campaign.run(self.condition_id, **kwargs)[self.condition_id]


def _style(task: Task, prompt: Optional[str]) -> str:
    return task.default_style if prompt is None else prompt


def _batch_root(judge: JudgeSpec) -> Optional[str]:
    return judge.options.batch_root if judge.capabilities().batch else None


def _task_spec(task: Task, settings: TaskSettings) -> BaseTaskSpec:
    cfg = OmegaConf.create({
        "task_spec": task.name if task.builtin else task_locator(task),
        "rule_amendment": {"initial_rule_set_path": "unused.jsonl"},
        # Every scenario-seed path a built-in task spec reads at construction.
        "scenario_generation": {"good_seed_scenario_path": "unused.jsonl", "bad_seed_scenario_path": "unused.jsonl",
                                "seed_scenario_path": "unused.jsonl"},
        "agent_model": {},
        "judge_model": {},
        "eval_cfg": {"judge_types": [settings.judge_type], "reasoning_prompt_style": settings.prompt_style,
                     "individual_rule_context": settings.rule_context, "evaluation_temperature": 1.0},
    })
    return build_task_spec(cfg) if task.builtin else TaskSpecAdapter(task, cfg)


def _keyed(mode: str) -> bool:
    # A ``both`` draw's per-scenario summary joins on string ``prompt`` and ``response``.
    if mode not in EVALUATION_MODES:
        raise CampaignError(f"mode {mode!r} is not one of {EVALUATION_MODES}")
    return mode == EVALUATION_MODE_BOTH


def _task_of(spec: Union[str, Task, type], scenarios: ScenarioSet, where: str, *, keyed: bool) -> Task:
    try:
        task = load_task(spec)
        task.check_scenarios(scenarios)
    except TaskError as error:
        raise CampaignError(f"{where} ({scenarios.path}): {error}") from error
    unkeyed = next((index for index, row in enumerate(scenarios.rows) if not all(
        isinstance(row.get(name), str) for name in ("prompt", "response"))), None) if keyed else None
    if unkeyed is not None:
        raise CampaignError(
            f"{where} ({scenarios.path}): scenario row {unkeyed} has no string 'prompt' and 'response', which a "
            f"{EVALUATION_MODE_BOTH!r} draw's per-scenario summary is keyed by (their hash), so no such draw over "
            f"these rows could complete; task {task.identifier!r}'s scenarios without them run in a 'full' or "
            "'individual_bank' condition (mode=...)")
    return task


@dataclass(frozen=True)
class _Rows:
    """Rows held as JSON text, detached from every caller, so nothing changes them in place."""

    text: str

    @classmethod
    def of(cls, rows: Sequence[Mapping[str, Any]]) -> "_Rows":
        return cls(json.dumps([dict(row) for row in rows], ensure_ascii=False))

    def judged(self, held: Sequence[Mapping[str, Any]], what: str) -> tuple[dict[str, Any], ...]:
        """Return a fresh copy of the rows, refused if ``held`` no longer matches them."""
        if _Rows.of(held) != self:
            raise CampaignError(f"{what} changed in memory since they were bound: a run judges the rows whose file "
                                "the campaign recorded; load them again from the file and bind them anew")
        return tuple(json.loads(self.text))


def _detached_rules(rules: Ruleset) -> Ruleset:
    where = f"ruleset {rules.name!r} ({rules.path})"
    try:
        fresh = Ruleset.load(rules.path, name=rules.name, expect_sha256=rules.sha256)
    except RulesetError as error:
        raise CampaignError(f"{where} is not the file it names ({error}); a condition records the file's bytes, so "
                            "load the ruleset again from the file (Ruleset.load)") from error
    if fresh != rules:
        raise CampaignError(f"{where} holds rules other than its file's (changed in memory since it was loaded, or "
                            f"built by hand), under the file's SHA-256 {rules.sha256}: a condition records the file, "
                            "so its rules are judged from it; write the rules to a file and load it (Ruleset.load)")
    return fresh


def _detached_scenarios(scenarios: ScenarioSet) -> ScenarioSet:
    where = f"scenario set {scenarios.path}"
    try:
        fresh = ScenarioSet.load(scenarios.path, expect_sha256=scenarios.sha256)
    except (OSError, ValueError) as error:
        raise CampaignError(f"{where} is not the file it names ({error}); load it again from the file "
                            "(ScenarioSet.load)") from error
    if fresh.manifest != scenarios.manifest or fresh.rows != tuple(scenarios.rows):
        raise CampaignError(f"{where} holds rows other than its file's (changed in memory since it was loaded, or "
                            "built by hand): a campaign freezes the file's manifest and judges its rows; load it "
                            "again from the file (ScenarioSet.load)")
    return fresh


def _sampling_settings(sampling: Union[SamplingSettings, SamplingConfig]) -> SamplingSettings:
    if isinstance(sampling, SamplingSettings):
        return sampling
    settings = SamplingSettings.from_config(sampling)
    carried = settings.config().to_dict()
    dropped = sorted(name for name, value in sampling.to_dict().items() if carried[name] != value)
    if dropped:
        raise CampaignError(
            f"sampling sets {dropped}, which a campaign of this library does not carry "
            f"({', '.join(f'{name}={getattr(sampling, name)!r}' for name in dropped)}); run such a "
            "campaign as a configured campaign (ruleforge jobs launch campaign, whose jobs run "
            "python -m ruleforge.launch.programs.run_campaign)")
    return settings


@contextmanager
def _file_lock(root: Path, relative: Union[Path, str], *, holder: str, wait: bool = True,
               busy: str = "") -> Iterator[None]:
    # ``flock``: released however its holder ends and held across nodes on NFS. The file stays: unlinking a held lock
    # file would let a third process lock a new one.
    path = _destination(root, relative)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
    except OSError as error:
        raise CampaignError(f"{path} cannot be opened as {holder} ({error}); a write of this campaign's records "
                            "goes to its own files alone") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError(f"another process holds {path}, {holder}: {busy}") from error
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def _root_lock(root: Path) -> Iterator[None]:
    with _file_lock(root, LOCK_FNAME, holder="this root's lock"):
        yield


def _own_marker(directory: Path) -> bool:
    marker = directory / MARKER_FNAME
    return marker.is_file() and not marker.is_symlink()


def _is_campaign_root(directory: Path) -> bool:
    return any(os.path.lexists(directory / name) for name in ROOT_MARKERS)


def _destination(root: Path, relative: Union[Path, str]) -> Path:
    """Return ``root / relative`` once it stays in the root: no symlink or nested campaign root on the way; nothing is
    written."""
    resolved = os.path.realpath(root)
    if Path(resolved) != root:
        raise CampaignError(f"{root} now resolves to {resolved}: a write of this campaign's records would go there, "
                            "not to the root it created; restore the root, or create the campaign again")
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise CampaignError(f"{current} is a symlink: a write of this campaign's records would follow it out of "
                                f"{root}; replace it with a directory or file of the root's own")
        if _is_campaign_root(current):
            raise CampaignError(f"{current} is a campaign root of its own inside {root}: this campaign's records "
                                "are never written into another root")
    return current


def _refuse_links(directory: Path, what: str, claimed: Sequence[Union[Path, str]] = ()) -> None:
    """Refuse ``directory`` when it or anything below is a symlink or a campaign root, other than ``claimed`` places."""
    try:
        info = os.lstat(directory)
    except (FileNotFoundError, NotADirectoryError):
        return
    except OSError as error:
        raise CampaignError(f"{what} {directory} cannot be checked ({error}): a write there could follow a link no "
                            "check saw; make it readable") from error
    if stat.S_ISLNK(info.st_mode):
        raise CampaignError(f"{what} {directory} is a symlink: a write would follow it out; replace it with a "
                            "directory of its own")
    if not stat.S_ISDIR(info.st_mode):
        return
    exempt = {directory / place for place in claimed}

    def refuse_root(here: Path) -> None:
        if here not in exempt and _is_campaign_root(here):
            raise CampaignError(f"{what} {directory} holds a campaign root of its own ({here}), which is never "
                                "written into")

    refuse_root(directory)
    try:
        for path, entry in walk_strictly(directory, missing_ok=True):
            if stat.S_ISLNK(entry.st_mode):
                raise CampaignError(f"{path}, below {what} {directory}, is a symlink: a write would follow it out; "
                                    "replace it with a file or directory of its own")
            if stat.S_ISDIR(entry.st_mode):
                refuse_root(path)
    except OSError as error:
        raise CampaignError(f"{what} {directory} cannot be walked whole ({error}): an entry below it that cannot be "
                            "listed or examined may be, or hold, a link a write would follow; make it readable"
                            ) from error


def _spellings(place: Union[Path, str], what: str) -> tuple[Path, Path]:
    """Return ``place`` as written and as a write reaches it, refusing a spelling that names two places: a ``..`` after
    a symlink or a missing directory, or a leading ``~``."""
    text = os.fspath(place)
    expanded = os.path.expanduser(text)
    if expanded != text:
        raise CampaignError(f"{what} {place} names two places: {os.path.abspath(expanded)} with its '~' expanded, "
                            f"and {os.path.abspath(text)} where a write handed it as given lands (nothing expands it "
                            "there); give the place with its '~' expanded")
    given = Path(text)
    written = Path(os.path.abspath(given))
    resolved = Path(os.path.realpath(given))
    if Path(os.path.realpath(written)) != resolved:
        raise CampaignError(f"{what} {place} names two places: {written} as written out, and {resolved} where a write "
                            "handed it as given lands (a '..' after a symlink, which the kernel takes from the link's "
                            "target); give the place without that '..'")
    made = made_on_the_way(text)
    if made is not None:
        raise CampaignError(f"{what} {place} names two places: {written} as written out, and {made}, which a write "
                            "handed it as given makes on the way when it makes its parents (mkdir -p), before it takes "
                            f"the '..' after it ({made.name} is not there yet); give the place without that '..'")
    return written, resolved


def bound_batch_root(root: Path, given: Optional[str]) -> str:
    """Return where a batch judge bound to the campaign keeps its paid state: its ``batch_root`` made absolute, or
    ``root/batch``."""
    if not given:
        return os.path.abspath(root / DEFAULT_BATCH_DIRNAME)
    place = Path(given).expanduser()
    _spellings(place, "the batch judge's batch_root")
    return os.path.abspath(place)


def refuse_batch_root(root: Union[Path, str], place: Union[Path, str], *,
                      what: str = "the batch judge's batch_root") -> None:
    """Refuse ``place`` as the batch root of a judge bound to the campaign at ``root``."""
    _refuse_foreign(Path(os.path.realpath(root)), os.fspath(place), what)


def _refuse_foreign(root: Path, place: str, what: str) -> None:
    """Refuse a batch root unless it is the root's own ``batch/``, or outside every campaign root."""
    written, resolved = _spellings(place, what)
    lock = campaign_lock_path(root)
    if any(candidate.is_relative_to(lock) for candidate in (written, resolved)):
        raise CampaignError(f"{what} {place} is, or lies in, {lock}, the campaign lock beside {root}, which its creation "
                            "and its analysis take: the transport would make a directory where the lock is opened, and "
                            "the analysis of what it paid for would fail; choose another place")
    own = root / DEFAULT_BATCH_DIRNAME
    for candidate in (written, resolved):
        if candidate.is_relative_to(root):
            reserved = _reserved_record(candidate.relative_to(root))
            if reserved is not None:
                raise CampaignError(f"{what} {place} is, or lies in, {root / reserved}, a record of the campaign's own: "
                                    "the transport would make its directories there, and the campaign's write of that "
                                    "record would fail after the paid judging; choose another place")
            if not candidate.is_relative_to(own):
                raise CampaignError(f"{what} {place} lies in the campaign root {root} outside its "
                                    f"{DEFAULT_BATCH_DIRNAME}/, the one place in it a batch judge's paid state goes: "
                                    f"its directories would sit among the campaign's own; give {own} (or a place in "
                                    "it), or a place outside every campaign root")
            _destination(root, candidate.relative_to(root))
            return
    directory = _foreign_root(written, resolved)
    if directory is not None:
        raise CampaignError(f"{what} {place} is inside {directory}, a campaign root this library did not "
                            "create (no marker), which it never writes; choose a place outside it")
    other = next((parent for parent in dict.fromkeys((written, *written.parents, resolved, *resolved.parents))
                  if _is_campaign_root(parent)), None)
    if other is not None:
        raise CampaignError(f"{what} {place} is, or is inside, {other}, another campaign's root, which only that "
                            "campaign writes; choose a place outside it")


def _refuse_work_place(root: Optional[Path], batch_root: Path, directory: Path) -> None:
    if root is None:
        refuse_protected(batch_root, "the batch judge's batch_root")
    else:
        _refuse_foreign(root, os.fspath(batch_root), "the batch judge's batch_root")
    refuse_pinned(directory, "the batch work directory")
    _refuse_links(Path(os.path.abspath(directory)), "the batch work directory")


def refuse_pinned(place: Union[Path, str], what: str) -> None:
    """Refuse ``place`` on a pinned root's path; nothing is written."""
    for location in dict.fromkeys(_spellings(place, what)):
        _refuse_pinned(location)


def batch_work_guard(root: Optional[Union[Path, str]] = None) -> Callable[[Path, Path], None]:
    """Return the check for every batch work directory, for a judge no ``Campaign`` holds; a refusal is fatal, never
    retried."""
    bound = None if root is None else Path(os.path.realpath(root))

    def guard(batch_root: Path, directory: Path) -> None:
        try:
            _refuse_work_place(bound, batch_root, directory)
        except CampaignError as error:
            raise BatchDestinationRefused(str(error)) from error

    return guard


def _refuse_condition_place(root: Path, condition_id: str) -> None:
    relative = Path(condition_id)
    reserved = _reserved_record(relative)
    if reserved is None and relative.parts and relative.parts[0] == DEFAULT_BATCH_DIRNAME:
        reserved = Path(DEFAULT_BATCH_DIRNAME)
    if reserved is not None:
        raise CampaignError(f"condition {condition_id!r}'s directory would be, or lie in, {root / reserved}, a record "
                            "of the campaign's own (its root's, a condition's or a draw's) or its batch root; give the "
                            "condition another id")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if os.path.lexists(current) and not current.is_dir():
            raise CampaignError(f"condition {condition_id!r}'s directory would lie below {current}, a file of the "
                                "root, where nothing can be made; give the condition another id")
    target = root / relative
    if os.path.lexists(target) and not target.is_dir():
        raise CampaignError(f"condition {condition_id!r}'s directory would be {target}, a file of the root; give the "
                            "condition another id")


def _reserved_record(relative: Path) -> Optional[Path]:
    parts = relative.parts
    if parts and parts[0] in ROOT_RECORDS:
        return Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        if part in CONDITION_RECORDS or DRAW_RECORD.fullmatch(part):
            return Path(*parts[:index + 1])
    return None


def _foreign_root(written: Path, resolved: Path) -> Optional[Path]:
    for directory in dict.fromkeys((written, *written.parents, resolved, *resolved.parents)):
        if os.path.lexists(directory / SAMPLING_MANIFEST_FNAME) and not _own_marker(directory):
            return directory
    return None


def _enclosing_root(location: Path, written: Path) -> Optional[Path]:
    parents = dict.fromkeys((*location.parents, *written.parents))
    return next((parent for parent in parents if _is_campaign_root(parent)), None)


def refuse_protected(place: Union[Path, str], what: str, *, campaign_root: bool = False,
                     below: bool = False, claimed: Sequence[Union[Path, str]] = ()) -> None:
    """Refuse ``place`` when a write there could land in a root nothing may write (a two-place spelling, a symlink, a
    campaign or pinned root, or a ``campaign_root`` holding a retired style's condition), with ``below`` all under it
    but ``claimed`` directories; nothing is written."""
    written, resolved = _spellings(place, what)
    if written.is_symlink() and not campaign_root:
        raise CampaignError(f"{what} {written} is a symlink: a write would follow it to {resolved}; replace it with a "
                            "directory of its own")
    for location in dict.fromkeys((written, resolved)):
        _refuse_pinned(location)
        if campaign_root:
            _refuse_retired(location)
    enclosing = _enclosing_root(resolved, written)
    if enclosing is not None:
        raise CampaignError(f"{what} {written} is inside the campaign root {enclosing}, which nothing but its own "
                            "campaign writes; choose a place outside it")
    if not campaign_root and (_is_campaign_root(written) or _is_campaign_root(resolved)):
        raise CampaignError(f"{what} {written} is a campaign root, which nothing but its own campaign writes; choose "
                            "another place")
    if below:
        _refuse_links(written, what, claimed)


def refuse_work_directory(place: Union[Path, str], what: str) -> None:
    """Refuse a batch work directory that is a symlink, is pinned, lies in a campaign root but its ``batch/`` or in one
    holding a retired style's condition, or holds a link below."""
    written, resolved = _spellings(place, what)
    if written.is_symlink():
        raise CampaignError(f"{what} {written} is a symlink: a write would follow it to {resolved}")
    for location in dict.fromkeys((written, resolved)):
        _refuse_pinned(location)
    for root in dict.fromkeys(Path(os.path.realpath(parent)) for parent in (*written.parents, *resolved.parents)
                              if _is_campaign_root(parent)):
        _refuse_foreign(root, os.fspath(written), what)
        _refuse_retired(root)
    _refuse_links(written, what)


@contextmanager
def _campaign_locked(root: Path) -> Iterator[None]:
    _refuse_linked_campaign_lock(root)
    held = ExitStack()
    try:
        held.enter_context(campaign_lock(root))
    except SamplingDesignError as error:
        raise CampaignError(f"{root} cannot be created or rebound now: {error}") from error
    with held:
        yield


def _refuse_linked_campaign_lock(root: Path) -> None:
    lock = campaign_lock_path(root)
    if lock.is_symlink() or (os.path.lexists(lock) and not lock.is_file()):
        raise CampaignError(f"{lock}, the campaign lock an analysis of {root} takes beside it, is not a file of its "
                            "own (a symlink, or another kind of file): an analysis would create it wherever it "
                            "points. Remove it")


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"{path} cannot be read as {what} ({error})") from error
    if not isinstance(payload, dict):
        raise CampaignError(f"{path} is not {what} (a JSON object)")
    return payload


def pinned_roots() -> Mapping[str, tuple[str, ...]]:
    """Return each published run no code may write to, with its experiments: the runs the goldens read."""
    return published_runs()


def _refuse_pinned(location: Path) -> None:
    try:
        uri = to_uri(location)
    except (ArtifactRootError, ValueError):
        return
    whole = uri == ARTIFACT_SCHEME + "."
    for pinned, experiments in pinned_roots().items():
        if whole or uri == pinned or uri.startswith(pinned + "/") or pinned.startswith(uri + "/"):
            raise CampaignError(f"{location} ({uri}) is {pinned}, or on its path, a published run of "
                                f"{', '.join(experiments)}: it is never written; choose another root")


def _check_frozen(root: Path, campaign_id: str, manifest: ScenarioManifest, config: SamplingConfig) -> None:
    try:
        frozen = ScenarioManifest.load(_destination(root, SCENARIO_MANIFEST_FNAME))
        recorded = check_sampling_manifest(_destination(root, SAMPLING_MANIFEST_FNAME),
                                           campaign_manifest(campaign_id, config, manifest))
    except SamplingDesignError as error:
        raise CampaignError(f"{root} no longer records the campaign this process holds ({error}); create it again "
                            "(Campaign.create) to bind what it records") from error
    if recorded is None or frozen.manifest_sha256 != manifest.manifest_sha256:
        found = (f"holds no {SAMPLING_MANIFEST_FNAME}" if recorded is None else
                 f"froze another population ({frozen.manifest_sha256[:12]}...) than the one this campaign holds "
                 f"({manifest.manifest_sha256[:12]}...)")
        raise CampaignError(f"{root} {found}: its records no longer describe the campaign this process holds; "
                            "create it again (Campaign.create) to bind what it records")


def _bootstrap_only(root: Path, marker: Mapping[str, Any], manifest: ScenarioManifest, config: SamplingConfig) -> str:
    """The way on when ``root`` holds this population and ``config`` but for its bootstrap, which a default campaign id
    digests and only the analysis reads; else nothing."""
    path = root / SAMPLING_MANIFEST_FNAME
    if not path.is_file() or marker.get("scenario_manifest_sha256") != manifest.manifest_sha256:
        return ""
    held = sampling_config_from_manifest(load_sampling_manifest(path))
    if held.bootstrap == config.bootstrap or dataclasses.replace(held, bootstrap=config.bootstrap) != config:
        return ""
    bootstrap = config.bootstrap
    command = (f"ruleforge analyze {shlex.quote(str(root))} --n-resamples {bootstrap.n_resamples} --ci-level "
               f"{bootstrap.ci_level} --seed {bootstrap.seed} --replace")
    return (f", or, as the two differ only in their bootstrap, which changes the analysis alone, analyse this root "
            f"with it (`{command}`) and read it with Campaign.open({str(root)!r}).result()")


def _check_owner(root: Path, campaign_id: str, manifest: ScenarioManifest, config: SamplingConfig,
                 identity: Optional[tuple[int, int]] = None) -> None:
    marker = _destination(root, MARKER_FNAME)
    if identity is not None:
        try:
            found = os.stat(root)
        except OSError as error:
            raise CampaignError(f"{root} cannot be read ({error}); this campaign writes only the root it "
                                "created") from error
        if (found.st_dev, found.st_ino) != identity:
            raise CampaignError(f"{root} is no longer the directory this campaign created (it was replaced since): "
                                "a write would land in whatever took its place; create the campaign again")
    if not marker.is_file():
        raise CampaignError(f"{root} holds no {MARKER_FNAME}: it is a root this library did not create (every "
                            "published root among them), read with Campaign.open and never written")
    Campaign._check_marker(marker, campaign_id, manifest, config)
    _check_frozen(root, campaign_id, manifest, config)
    _refuse_pinned(root)


def _refuse_retired(root: Path) -> None:
    try:
        refuse_retired_root(root)
    except SamplingDesignError as error:
        raise CampaignError(str(error)) from error


@contextmanager
def _owner_writing(root: Path, campaign_id: str, manifest: ScenarioManifest, config: SamplingConfig,
                   identity: Optional[tuple[int, int]] = None) -> Iterator[None]:
    _check_owner(root, campaign_id, manifest, config, identity)
    _refuse_retired(root)
    with _root_lock(root):
        _check_owner(root, campaign_id, manifest, config, identity)
        yield


def _work_directory_guard(campaign: "Campaign") -> Callable[[Path, Path], None]:
    def guard(batch_root: Path, directory: Path) -> None:
        try:
            campaign._owned()
            _refuse_work_place(campaign.root, batch_root, directory)
        except CampaignError as error:
            raise BatchDestinationRefused(str(error)) from error

    return guard


def _run_write_guard(campaign: "Campaign", condition: "Condition") -> Callable[[Path], None]:
    def guard(directory: Path) -> None:
        campaign._owned()
        relative = directory.relative_to(campaign.root)
        _destination(campaign.root, relative if relative.parts else Path(COMPLETION_FNAME))
        campaign._run_destinations(condition)

    return guard


class Campaign:
    """One campaign root: created and written by this process (:meth:`create`), or read (:meth:`open`)."""

    def __init__(self, root: Path, campaign_id: str, config: SamplingConfig, scenarios: ScenarioManifest, *,
                 writable: bool, rows: Optional[ScenarioSet] = None, sampling: Optional[SamplingSettings] = None,
                 identity: Optional[tuple[int, int]] = None) -> None:
        self.root = root
        self.campaign_id = campaign_id
        self.config = config
        self.scenario_manifest = scenarios
        self.writable = writable
        # None on a campaign opened rather than created.
        self.scenarios = rows
        self._scenario_rows = None if rows is None else _Rows.of(rows.rows)
        self.sampling = sampling if sampling is not None else SamplingSettings.from_config(config)
        self._identity = identity
        self._bound: dict[str, Condition] = {}


    @classmethod
    def create(cls, root: Path | str, *, campaign_id: str, scenarios: ScenarioSet,
               sampling: Union[SamplingSettings, SamplingConfig] = SamplingSettings()) -> "Campaign":
        """Return a new campaign at ``root`` over ``scenarios`` and ``sampling`` (an empty or absent root is created and
        marked), or this one again for a marked root of this campaign, population and config; any other root, or one a
        provenance sidecar pins, is refused (``open`` reads those)."""
        settings, config, scenarios, location = cls._creating(root, campaign_id=campaign_id, scenarios=scenarios,
                                                              sampling=sampling)
        location.mkdir(parents=True, exist_ok=True)
        # Both locks, in the order shard merges, amendments and analyses take them.
        with _root_lock(location), _campaign_locked(location):
            cls._check_ours(location, campaign_id, scenarios.manifest, config)
            marker, *_ = (_destination(location, name) for name in (
                MARKER_FNAME, SCENARIO_MANIFEST_FNAME, SAMPLING_MANIFEST_FNAME, SNAPSHOT_DIRNAME))
            if not marker.is_file():
                write_json_sidecar(marker, {"kind": MARKER_KIND, "schema_version": MARKER_SCHEMA_VERSION,
                                            "campaign_id": campaign_id,
                                            "scenario_manifest_sha256": scenarios.manifest.manifest_sha256,
                                            "created_at": utc_now()})
            cls._freeze(location, campaign_id, scenarios.manifest, config)
            found = os.stat(location)
        return cls(location, campaign_id, config, scenarios.manifest, writable=True, rows=scenarios,
                   sampling=settings, identity=(found.st_dev, found.st_ino))

    @classmethod
    def check_create(cls, root: Path | str, *, campaign_id: str, scenarios: ScenarioSet,
                     sampling: Union[SamplingSettings, SamplingConfig] = SamplingSettings()) -> None:
        """Make every refusal ``create`` makes, the frozen records' among them, with nothing written."""
        _, config, scenarios, location = cls._creating(root, campaign_id=campaign_id, scenarios=scenarios,
                                                       sampling=sampling)
        cls._frozen(location, campaign_id, scenarios.manifest, config)

    @classmethod
    def _creating(cls, root: Path | str, *, campaign_id: str, scenarios: ScenarioSet,
                  sampling: Union[SamplingSettings, SamplingConfig]) -> tuple[SamplingSettings, SamplingConfig,
                                                                             ScenarioSet, Path]:
        settings = _sampling_settings(sampling)
        config = settings.config()
        validate_identifier("campaign_id", campaign_id)
        scenarios = _detached_scenarios(scenarios)
        given = Path(root).expanduser()
        location = given.resolve()
        approved = settings.approved_scenario_count
        expected = config.expected_scenario_count if approved is None else approved
        if expected is not None and len(scenarios) != expected:
            source = (f"a {config.campaign_mode} campaign judges {expected} scenarios unless it approves another "
                      "count (approved_scenario_count)") if approved is None else (
                f"the campaign approves {approved} scenarios")
            raise CampaignError(f"{source}; {scenarios.path.name} holds {len(scenarios)}")
        if location.exists() and not location.is_dir():
            raise CampaignError(f"{location} is not a directory")
        _refuse_pinned(location)
        _refuse_linked_campaign_lock(location)
        enclosing = _enclosing_root(location, Path(os.path.abspath(given)))
        if enclosing is not None:
            raise CampaignError(f"{location} is inside the campaign root {enclosing}, which a new campaign would write "
                                "into; create it in a directory of its own")
        # Checked before the root is made or locked, so a foreign root gets no lock file, and again under the lock.
        cls._check_ours(location, campaign_id, scenarios.manifest, config)
        return settings, config, scenarios, location

    @classmethod
    def _check_ours(cls, location: Path, campaign_id: str, manifest: ScenarioManifest, config: SamplingConfig) -> None:
        marker = _destination(location, MARKER_FNAME)
        if marker.is_file():
            cls._check_marker(marker, campaign_id, manifest, config)
            _refuse_retired(location)
        elif location.exists() and any(item.name != LOCK_FNAME for item in location.iterdir()):
            raise CampaignError(
                f"{location} holds a campaign root this library did not create (no {MARKER_FNAME}): a root made "
                "outside it, every published one among them, is read with Campaign.open and never written. "
                "Create the campaign in a new root")

    @staticmethod
    def _check_marker(marker: Path, campaign_id: str, manifest: ScenarioManifest, config: SamplingConfig) -> None:
        payload = _read_json(marker, "a campaign marker")
        if payload.get("kind") != MARKER_KIND:
            raise CampaignError(f"{marker} is not a {MARKER_KIND} marker")
        if payload.get("campaign_id") != campaign_id:
            raise CampaignError(f"{marker.parent} holds campaign {payload.get('campaign_id')!r}, not {campaign_id!r}; "
                                "create this one in a root of its own" + _bootstrap_only(marker.parent, payload,
                                                                                      manifest, config))
        if payload.get("scenario_manifest_sha256") != manifest.manifest_sha256:
            raise CampaignError(f"{marker.parent} was created over another population "
                                f"({str(payload.get('scenario_manifest_sha256'))[:12]}...) than this one "
                                f"({manifest.manifest_sha256[:12]}...)")

    @staticmethod
    def _frozen(root: Path, campaign_id: str, manifest: ScenarioManifest,
                sampling: SamplingConfig) -> Optional[dict[str, Any]]:
        """Refuse a root that froze another population or campaign, and return its sampling manifest (None when it froze
        none); nothing is written."""
        frozen_path = root / SCENARIO_MANIFEST_FNAME
        if frozen_path.exists():
            frozen = ScenarioManifest.load(frozen_path)
            if frozen.manifest_sha256 != manifest.manifest_sha256:
                raise CampaignError(f"{root} froze another population ({frozen.manifest_sha256[:12]}...) than this "
                                    f"one ({manifest.manifest_sha256[:12]}...)")
        try:
            return check_sampling_manifest(root / SAMPLING_MANIFEST_FNAME,
                                           campaign_manifest(campaign_id, sampling, manifest))
        except SamplingDesignError as error:
            raise CampaignError(str(error)) from error

    @staticmethod
    def _freeze(root: Path, campaign_id: str, manifest: ScenarioManifest, sampling: SamplingConfig) -> None:
        existing = Campaign._frozen(root, campaign_id, manifest, sampling)
        frozen_path = root / SCENARIO_MANIFEST_FNAME
        if not frozen_path.exists():
            manifest.write(frozen_path)
        if existing is None:
            code = code_state()
            ensure_sampling_manifest(root / SAMPLING_MANIFEST_FNAME,
                                     campaign_manifest(campaign_id, sampling, manifest, code))
            write_code_snapshot(root, code)

    @classmethod
    def open(cls, root: Path | str) -> "Campaign":
        """A campaign root, read: any root, marked or not; nothing is written, and every write is refused."""
        location = Path(os.path.abspath(Path(root).expanduser()))
        manifest_path = location / SAMPLING_MANIFEST_FNAME
        if not manifest_path.is_file():
            raise CampaignError(f"{location} holds no {SAMPLING_MANIFEST_FNAME}: it is no campaign root")
        manifest = load_sampling_manifest(manifest_path)
        return cls(location, str(manifest["campaign_id"]), sampling_config_from_manifest(manifest),
                   ScenarioManifest.load(location / SCENARIO_MANIFEST_FNAME), writable=False)

    @classmethod
    def draft(cls, root: Path | str, *, campaign_id: str, scenarios: ScenarioSet,
              sampling: Union[SamplingSettings, SamplingConfig] = SamplingSettings()) -> "Campaign":
        """Return the campaign ``create`` would make, made nowhere: every write is refused; conditions can still be
        named and checked."""
        settings = _sampling_settings(sampling)
        config = settings.config()
        validate_identifier("campaign_id", campaign_id)
        scenarios = _detached_scenarios(scenarios)
        return cls(Path(root).expanduser().resolve(), campaign_id, config, scenarios.manifest, writable=False,
                   rows=scenarios, sampling=settings)

    @classmethod
    def reopen(cls, root: Path | str) -> "Campaign":
        """Reopen a root this library created to write its analysis; it registers and judges nothing."""
        location = Path(root).expanduser().resolve()
        if not _own_marker(location):
            raise CampaignError(f"{location} holds no {MARKER_FNAME} of its own: a root this library did not create "
                                "(every published one among them) is read with Campaign.open and never written (a "
                                "configured root this version made, its .ruleforge-configured, is analysed and merged "
                                "into by `ruleforge analyze` and `ruleforge campaign merge-shards`)")
        _refuse_pinned(location)
        try:
            manifest = load_sampling_manifest(location / SAMPLING_MANIFEST_FNAME)
            config = sampling_config_from_manifest(manifest)
            frozen = ScenarioManifest.load(location / SCENARIO_MANIFEST_FNAME)
        except (OSError, SamplingDesignError) as error:
            raise CampaignError(f"{location} does not hold a whole campaign's records ({error})") from error
        found = os.stat(location)
        campaign = cls(location, str(manifest.get("campaign_id")), config, frozen, writable=True,
                       identity=(found.st_dev, found.st_ino))
        campaign._owned()
        return campaign

    def _writable(self) -> None:
        if not self.writable:
            raise ReadOnlyCampaign(f"{self.root} was opened for reading (Campaign.open); a campaign is written only "
                                   "through Campaign.create")

    def _owned(self) -> None:
        self._writable()
        _check_owner(self.root, self.campaign_id, self.scenario_manifest, self.config, self._identity)

    @contextmanager
    def _writing(self) -> Iterator[None]:
        self._writable()
        with _owner_writing(self.root, self.campaign_id, self.scenario_manifest, self.config, self._identity):
            yield

    @contextmanager
    def _reading(self) -> Iterator[None]:
        self._recorded()
        try:
            yield
        except Exception:
            self._recorded()
            raise
        self._recorded()

    def _recorded(self) -> None:
        path = self.root / SAMPLING_MANIFEST_FNAME
        try:
            manifest = load_sampling_manifest(path)
        except SamplingDesignError as error:
            raise CampaignError(f"{self.root} no longer records the campaign this object holds ({error}); open it "
                                "again (Campaign.open) to read what it records") from error
        found = (manifest.get("campaign_id"), manifest.get("scenario_manifest_sha256"))
        held = (self.campaign_id, self.scenario_manifest.manifest_sha256)
        if found != held:
            raise CampaignError(
                f"{path} records campaign {found[0]!r} over population {str(found[1])[:12]}..., not the campaign this "
                f"object holds ({held[0]!r} over {held[1][:12]}...): the root was replaced since it was created or "
                "opened, and its records would be returned under the wrong campaign. Open it again (Campaign.open) "
                "to read what it records")


    def condition(self, condition_id: Optional[str] = None, *, rules: Ruleset, judge: JudgeSpec,
                  settings: ConditionSettings = ConditionSettings()) -> Condition:
        """Register ``condition_id`` (or rebind it when this root holds it with this identity) and bind the judge, rules
        and settings; with no id it is named ``<model>/<style>[/mode.<mode>]-<fingerprint[:8]>``."""
        self._writable()
        prepared = self._prepare(condition_id, rules=rules, judge=judge, settings=settings)
        self._register(prepared)
        return prepared

    def prepare(self, condition_id: Optional[str] = None, *, rules: Ruleset, judge: JudgeSpec,
                settings: ConditionSettings = ConditionSettings(), segments: str = "") -> Condition:
        """Return the condition ``condition`` would register, with every refusal made and nothing written."""
        return self._prepare(condition_id, rules=rules, judge=judge, settings=settings, segments=segments)

    def _holding_rows(self, what: str) -> None:
        if self.scenarios is None:
            raise CampaignError(f"{self.root} is held without its population's rows (Campaign.open, Campaign.reopen): "
                                f"{what} takes Campaign.create, which reads the population again")

    def _prepare(self, condition_id: Optional[str], *, rules: Ruleset, judge: JudgeSpec,
                 settings: ConditionSettings, segments: str = "") -> Condition:
        self._holding_rows("naming a condition")
        if rules.path is None:
            raise _unsaved(rules)
        rules = _detached_rules(rules)
        where = condition_id or "(named by its fingerprint)"
        task = _task_of(settings.task, self.scenarios, f"condition {where}", keyed=_keyed(settings.mode))
        try:
            task_settings = TaskSettings(_style(task, settings.prompt), settings.rule_context,
                                         settings.interpretation)
            task_identity = task.identity(task_settings)
        except TaskError as error:
            raise CampaignError(f"condition {where}: {error}") from error
        refusal = task.new_run_refusal(task_settings.prompt_style)
        if refusal is not None:
            raise CampaignError(f"condition {where}: {refusal}")
        approvals = settings.approvals
        if approvals.rule_set_sha256 is not None and rules.sha256 != approvals.rule_set_sha256:
            raise CampaignError(f"ruleset {rules.name} has SHA-256 {rules.sha256}, not the approved "
                                f"{approvals.rule_set_sha256}")
        levels = self._levels(rules, settings)
        try:
            refuse_fake_in_final(judge.model_type, self.config.campaign_mode)
        except CampaignRunError as error:
            raise CampaignError(str(error)) from error
        bound = self._bound_judge(judge)
        if not bound.capabilities().per_request_errors:
            raise CampaignError(f"model_type {judge.model_type!r} does not report a transport failure as one "
                                "(capabilities.per_request_errors); a campaign would charge it to an attempt")
        self._preflight(where, bound, approvals)
        # The preflight has just recomputed derived weights' digest, so the identity reads it as recorded.
        identity = bound.identity(verify_weights=False)

        def spec_for(cid: str) -> ConditionSpec:
            try:
                return condition_spec_for(
                    condition_id=cid, config=self.config, judge=identity, task=task_identity, ruleset=rules,
                    evaluation_mode=settings.mode,
                    rule_levels_sha256=None if levels is None else levels.levels_file["sha256"],
                    extra=dict(settings.extra))
            except SamplingDesignError as error:
                raise CampaignError(f"condition {cid}: {error}") from error

        if condition_id is None:
            base = f"{identity.model_name}/{task_settings.prompt_style}"
            if settings.mode != EVALUATION_MODE_BOTH:
                base += f"/mode.{settings.mode}"
            base += segments
            # The fingerprint leaves the id out, so a draft under the base names the condition.
            condition_id = f"{base}-{spec_for(base).fingerprint[:8]}"
        return Condition(condition_id=condition_id, spec=spec_for(condition_id), judge=bound, rules=rules,
                         settings=settings, task_locator=task_locator(settings.task), task=task, levels=levels,
                         campaign=self)

    def _register(self, condition: Condition, *, check: Optional[Callable[[], None]] = None) -> None:
        with self._writing():
            if check is not None:
                check()
            self._write_registration(condition, self._registration(condition))
        self._bound[condition.condition_id] = condition

    def _joining_analysis(self, condition_ids: Sequence[str]) -> list[str]:
        if not self._estimates_path().exists():
            return []
        return sorted(set(condition_ids) - set(load_conditions(self.root)))

    def _registration(self, condition: Condition) -> Optional[tuple[Path, dict[str, Any]]]:
        condition_id, spec, bound = condition.condition_id, condition.spec, condition.judge
        record = {"batch_root": _batch_root(bound), "task_locator": condition.task_locator,
                  "request_settings": bound.request_settings()}
        execution = self._condition_destinations(condition_id, record["batch_root"])
        try:
            registered = check_registered_condition(self.root, spec)
        except CampaignRunError as error:
            raise CampaignError(str(error)) from error
        if registered.exists():
            if not execution.is_file():
                raise CampaignError(f"{registered} was registered without its {EXECUTION_FNAME} (not through "
                                    "Campaign.condition), so where its paid state lives is not recorded")
            return None
        if execution.is_file():
            # A registration interrupted between its two records: completed when it is this one, never replaced when it
            # is another's.
            recorded = _read_json(execution, "an execution record")
            if recorded != record:
                raise CampaignError(
                    f"{execution} records an unfinished registration of {condition_id!r} ({recorded}), not this one "
                    f"({record}); register this condition under another id, or remove {execution} if that "
                    "registration was abandoned")
        return execution, record

    def _one_configuration(self, conditions: Sequence[Condition]) -> None:
        try:
            check_judge_configuration(self.root, [condition.spec for condition in conditions])
        except CampaignRunError as error:
            raise CampaignError(str(error)) from error

    def _write_registration(self, condition: Condition, pending: Optional[tuple[Path, dict[str, Any]]]) -> None:
        if pending is None:
            return
        execution, record = pending
        if not execution.is_file():
            write_json_sidecar(execution, record)
        register_condition(self.root, condition.spec)

    def _condition_destinations(self, condition_id: str, batch_root: Optional[str]) -> Path:
        _refuse_condition_place(self.root, condition_id)
        execution = _destination(self.root, Path(condition_id) / EXECUTION_FNAME)
        _destination(self.root, Path(condition_id) / CONDITION_FNAME)
        if batch_root is not None:
            _refuse_foreign(self.root, batch_root, "the batch judge's batch_root")
        return execution

    def _preflight(self, condition_id: str, judge: JudgeSpec, approvals: Approvals) -> None:
        pins = {"approved_adapter_sha256": approvals.adapter_sha256,
                "approved_model_weights_sha256": approvals.model_weights_sha256}
        try:
            preflight_judge(OmegaConf.create({"judge_model": judge.options.written(), **pins}), self.config)
        except (CampaignRunError, SamplingDesignError, OmegaConfBaseException) as error:
            raise CampaignError(f"condition {condition_id}: {error}") from error

    def _levels(self, rules: Ruleset, settings: ConditionSettings) -> Optional[RuleLevels]:
        superiors = settings.rule_context == HIERARCHY_RULE_CONTEXT
        approved = settings.approvals.rule_levels_sha256
        if superiors != (settings.rule_levels is not None) or (approved is not None and not superiors):
            raise CampaignError(f"a {HIERARCHY_RULE_CONTEXT!r} condition reads each rule's level from rule_levels, "
                                f"and no other condition reads one (rule_context={settings.rule_context!r})")
        if settings.rule_levels is None:
            return None
        try:
            levels = load_rule_levels(rules.path, Path(settings.rule_levels))
        except RuleLevelsError as error:
            raise CampaignError(str(error)) from error
        if levels.ruleset["sha256"] != rules.sha256:
            raise CampaignError(f"{rules.path} changed while its levels were read (SHA-256 {levels.ruleset['sha256']}, "
                                f"not the ruleset's {rules.sha256}); bind the condition again")
        if approved is not None and levels.levels_file["sha256"] != approved:
            raise CampaignError(f"rule levels SHA-256 {levels.levels_file['sha256']} does not match approved "
                                f"{approved}")
        return levels

    def _bound_judge(self, judge: JudgeSpec) -> JudgeSpec:
        """Return ``judge`` with every option resolved once and detached, and its ``batch_root`` as its transport writes
        it."""
        batch = judge.capabilities().batch
        resolved: dict[str, Any] = {}
        for key in judge.options:
            try:
                resolved[key] = judge.options[key]
            except OmegaConfBaseException as error:
                if batch and key == "batch_root":
                    raise CampaignError(f"the batch judge's batch_root does not resolve ({error}); set the "
                                        "variable it reads, or give batch_root as null for "
                                        f"{self.root / DEFAULT_BATCH_DIRNAME}") from error
                raise CampaignError(f"the judge's option {key!r} does not resolve ({error}): a condition records its "
                                    "judge as bound, so every option is read once, when it is bound; set it") from error
        if batch:
            resolved["batch_root"] = bound_batch_root(self.root, resolved.get("batch_root"))
        bound = judge.with_options(**resolved)
        try:
            detached = {key: bound.options[key] for key in bound.options} == resolved
        except OmegaConfBaseException:
            detached = False
        if not detached:
            # A resolved value that reads as an interpolation again.
            raise CampaignError(f"the judge's options {sorted(resolved)} do not read back as the values they resolved "
                                "to; a condition records its judge as bound, so its options must be plain values")
        return bound

    def execution_record(self, condition_id: str) -> dict[str, Any]:
        """Return a registered condition's ``execution.json``."""
        path = self.root / condition_id / EXECUTION_FNAME
        with self._reading():
            if not (self.root / condition_id / CONDITION_FNAME).is_file() or not path.is_file():
                raise CampaignError(f"{self.root} holds no condition {condition_id!r} registered through "
                                    "Campaign.condition")
            return _read_json(path, "an execution record")

    def _bound_condition(self, condition_id: str) -> Condition:
        if condition_id not in self._bound:
            raise CampaignError(f"no judge is bound to {condition_id!r} in this process; bind one with "
                                "Campaign.condition first")
        return self._bound[condition_id]

    def start(self, condition_id: str, *, relocate_batch_root: bool = False) -> Judge:
        """Start the judge bound to ``condition_id``; differing request settings are refused, and a moved ``batch_root``
        unless ``relocate_batch_root``."""
        self._writable()
        judge = self._bound_condition(condition_id).judge
        batch_root = _batch_root(judge)
        with self._writing():
            execution = self._condition_destinations(condition_id, batch_root)
            record = self._held_to_record(condition_id, judge, relocate_batch_root=relocate_batch_root)
            if batch_root != record.get("batch_root"):
                write_json_sidecar(execution, {**record, "batch_root": batch_root})
        return judge.start()

    def _held_to_record(self, condition_id: str, judge: JudgeSpec, *, relocate_batch_root: bool) -> dict[str, Any]:
        settings = judge.request_settings()
        batch_root = _batch_root(judge)
        record = self.execution_record(condition_id)
        if settings != record.get("request_settings"):
            raise CampaignError(
                f"condition {condition_id!r} was recorded with request settings {record.get('request_settings')}, "
                f"and this judge sends {settings}: they change the request bytes, so its blocks would buy other "
                "batches under the same block ids. Bind the judge it was recorded with, or register a new condition")
        if batch_root != record.get("batch_root") and not relocate_batch_root:
            raise CampaignError(
                f"condition {condition_id!r} keeps its paid batch state under {record.get('batch_root')}, and this "
                f"judge's batch_root is {batch_root}: the batches bought there would be bought again. Bind the judge "
                "with the recorded batch_root, or move the state there and pass relocate_batch_root=True")
        return record


    def status(self, *, check_ledgers: bool = False) -> dict[str, Any]:
        """Return the campaign gate, computed and never written; ``check_ledgers`` also holds each draw's gate to its
        ledger."""
        with self._reading():
            return campaign_completion(self.root, write=False, check_ledgers=check_ledgers)

    def _snapshot(self) -> tuple[dict[str, ConditionSpec], dict[str, Any]]:
        gate = self.status()
        found = load_conditions(self.root)
        changed = sorted(cid for cid, entry in gate["conditions"].items()
                         if cid not in found or found[cid].fingerprint != entry["fingerprint"])
        if changed:
            raise CampaignError(f"{self.root}: conditions {changed} were removed or registered again under another "
                                "identity while the root was read; read it again")
        return {cid: found[cid] for cid in gate["conditions"]}, gate

    def check_shard_merge(self, condition_id: str) -> None:
        """Refuse a merge of a condition's shard roots unless every place it writes is the root's own; nothing is
        written."""
        self._owned()
        _refuse_retired(self.root)
        try:
            check_new_condition_id(self.root, condition_id)
        except CampaignRunError as error:
            raise CampaignError(str(error)) from error
        for relative in (COMPLETION_FNAME, SAMPLING_MANIFEST_FNAME, SCENARIO_MANIFEST_FNAME, SNAPSHOT_DIRNAME,
                         condition_id, Path(condition_id) / CONDITION_FNAME):
            _destination(self.root, relative)
        _refuse_links(self.root / SNAPSHOT_DIRNAME, "the code snapshots' directory")
        _refuse_links(self.root / condition_id, f"condition {condition_id!r}'s directory")
        _refuse_linked_campaign_lock(self.root)

    def merge_shards(self, condition_id: str, *, shards_dir: Optional[Union[Path, str]] = None,
                     num_shards: Optional[int] = None) -> dict[str, Any]:
        """Merge a condition's shard roots into this root under its write lock and return what was merged."""
        from ruleforge.evaluate.shard_merge import merge_condition_shards

        self.check_shard_merge(condition_id)
        with self._writing():
            self.check_shard_merge(condition_id)
            return merge_condition_shards(self.root, condition_id, shards_dir=shards_dir, num_shards=num_shards,
                                          writing=nullcontext)

    def _refuse_exhausted_draws(self, condition_id: str, *, analysed: bool) -> None:
        registered, gate = self._snapshot()
        _refuse_exhausted(self.root, condition_id, gate, registered, analysed=analysed)

    def _requested(self, draw_ids: Optional[Sequence[int]]) -> tuple[int, ...]:
        if draw_ids is None:
            return tuple(range(self.config.num_decoder_draws))
        draws = tuple(draw_ids)
        bad = [draw for draw in draws if isinstance(draw, bool) or not isinstance(draw, int)
               or not 0 <= draw < self.config.num_decoder_draws]
        if bad or len(set(draws)) != len(draws):
            raise CampaignError(f"draw_ids={list(draws)} must be distinct ids below {self.config.num_decoder_draws}, "
                                "the campaign's draws (a selection of draws, never a count)")
        return draws

    def run(self, condition_id: Optional[str] = None, *, draw_ids: Optional[Sequence[int]] = None,
            execution: ExecutionSettings = ExecutionSettings(), confirm_spend: bool = False,
            relocate_batch_root: bool = False) -> dict[str, dict[int, dict[str, Any]]]:
        """Judge the requested draws that are not complete and return each condition's completions by draw
        (``draw_ids``: all declared; ``condition_id``: all bound here); a paid judge needs ``confirm_spend=True``, and a
        condition another process runs is refused at once."""
        self._owned()
        _refuse_retired(self.root)
        self._holding_rows("a run")
        ids = list(self._bound) if condition_id is None else [condition_id]
        if not ids:
            raise CampaignError("no condition is bound in this process; register or rebind one with "
                                "Campaign.condition first")
        draws = self._requested(draw_ids)
        with checked_at_entry(self.root, refused=True):
            return {cid: self._run_condition(self._bound_condition(cid), draws, execution=execution,
                                             confirm_spend=confirm_spend, relocate_batch_root=relocate_batch_root)
                    for cid in ids}

    def _pending(self, condition_id: str, draws: tuple[int, ...]) -> tuple[int, ...]:
        """Return the requested draws that have not ended (``draw_ended``), all of them while the root registers no such
        condition; a draw recorded for another identity is refused."""
        entry = (self.status()["conditions"].get(condition_id)
                 if (self.root / SAMPLING_MANIFEST_FNAME).is_file() else None)
        if entry is None:
            return draws
        gates = {int(draw): record for draw, record in entry["draws"].items()}
        invalid = sorted(draw for draw in draws if gates[draw]["state"] == STATE_INVALID_MANIFEST)
        if invalid:
            raise CampaignError(f"condition {condition_id!r}: draws {invalid} were recorded for another fingerprint, "
                                "population or campaign (invalid_manifest); nothing of them is resumed")
        return tuple(draw for draw in draws if not draw_ended(gates[draw]))

    def _refuse_running(self, condition: Condition, execution: ExecutionSettings) -> None:
        if execution.sharded:
            raise CampaignError("execution.num_shards shards a configured campaign's condition over jobs (ruleforge "
                                "jobs launch campaign); Campaign.run judges a condition in one process")
        if execution.pack_draws and condition.judge.model_type != PACKABLE_MODEL_TYPE:
            raise CampaignError(f"execution.pack_draws needs a {PACKABLE_MODEL_TYPE!r} judge; "
                                f"{condition.judge.model_type!r} keeps its per-block submissions")
        if MANIFEST_AMENDMENTS_KEY in load_sampling_manifest(self.root / SAMPLING_MANIFEST_FNAME):
            raise CampaignError(f"{self.root} records a cap amendment, which only a configured campaign's jobs run "
                                "under (ruleforge jobs launch campaign); Campaign.run judges no amended root")
        self._run_destinations(condition)
        condition.judged_rules()
        self._judged_scenarios()

    @staticmethod
    def _refuse_unconfirmed(condition: Condition, confirm_spend: bool) -> None:
        judge_spec = condition.judge
        if judge_spec.capabilities().spends_money and not confirm_spend:
            raise CampaignError(f"condition {condition.condition_id!r}: judge {judge_spec.model_name!r} "
                                f"({judge_spec.model_type}) spends money on every call; pass confirm_spend=True "
                                "(--yes-spend) to run it")

    def preflight(self, condition_id: str, *, rules: Ruleset, judge: JudgeSpec,
                  settings: ConditionSettings = ConditionSettings(), draw_ids: Optional[Sequence[int]] = None,
                  execution: ExecutionSettings = ExecutionSettings(), confirm_spend: bool = False,
                  relocate_batch_root: bool = False) -> dict[str, Any]:
        """Return what ``condition`` then ``run`` would do, with every refusal made before a judge starts and nothing
        written."""
        self._holding_rows("a preflight")
        condition = self._prepare(condition_id, rules=rules, judge=judge, settings=settings)
        try:
            registered = check_registered_condition(self.root, condition.spec)
        except CampaignRunError as error:
            raise CampaignError(str(error)) from error
        if not registered.is_file():
            raise CampaignError(f"{self.root} registers no condition {condition_id!r}; register it first "
                                "(Campaign.condition, `ruleforge campaign add`)")
        self._held_to_record(condition_id, condition.judge, relocate_batch_root=relocate_batch_root)
        self._refuse_running(condition, execution)
        draws = self._requested(draw_ids)
        states = {int(draw): record["state"]
                  for draw, record in self.status()["conditions"][condition_id]["draws"].items()}
        pending = self._pending(condition_id, draws)
        if pending:
            self._refuse_unconfirmed(condition, confirm_spend)
        per_draw = expected_requests_per_draw(self.scenario_manifest, condition.judged_rules(),
                                              condition.spec.evaluation_mode)
        return {"root": str(self.root), "campaign_id": self.campaign_id, "condition_id": condition_id,
                "fingerprint": condition.spec.fingerprint, "identity": condition.spec.to_dict(),
                "draws": {str(draw): states[draw] for draw in draws}, "pending": list(pending),
                "requests_per_draw": per_draw, "requests": per_draw * len(pending)}

    def _run_condition(self, condition: Condition, draws: tuple[int, ...], *, execution: ExecutionSettings,
                       confirm_spend: bool, relocate_batch_root: bool) -> dict[int, dict[str, Any]]:
        cid = condition.condition_id
        self._refuse_running(condition, execution)
        if not self._pending(cid, draws):
            return {}
        self._refuse_unconfirmed(condition, confirm_spend)
        # Built before the judge starts, so a task spec that cannot be built starts no engine.
        task_spec = _task_spec(condition.task, TaskSettings(
            condition.spec.prompt_style, condition.spec.individual_rule_context, condition.spec.judge_type))
        # One run of a condition at a time: two would each repair the draws' ledgers and append the same blocks twice.
        self._owned()
        with _file_lock(self.root, Path(cid) / RUN_LOCK_FNAME, holder=f"condition {cid!r}'s run lock", wait=False,
                        busy=f"another process is judging condition {cid!r} in {self.root}, and two runs of one "
                             "condition would each write its draws' ledgers; run it again once that run ends (the "
                             "draws it completes are not judged again)"):
            pending = self._pending(cid, draws)
            if not pending:
                return {}
            judging: list[bool] = []
            try:
                completions = self._judge(condition, pending, task_spec, execution=execution,
                                          confirm_spend=confirm_spend, relocate_batch_root=relocate_batch_root,
                                          judging=judging)
            except BaseException:
                # A run that judged anything publishes the gate even when it fails, then raises its own failure.
                if judging:
                    try:
                        self._publish_gate()
                    except Exception as error:
                        logger.warning("The campaign gate of %s was not published after the run of condition %r "
                                       "failed: %s", self.root, cid, error)
                raise
            self._publish_gate()
        return completions

    def _publish_gate(self) -> dict[str, Any]:
        with self._writing():
            _destination(self.root, COMPLETION_FNAME)
            return campaign_completion(self.root)

    def _judge(self, condition: Condition, pending: tuple[int, ...], task_spec: BaseTaskSpec, *,
               execution: ExecutionSettings, confirm_spend: bool, relocate_batch_root: bool,
               judging: list[bool]) -> dict[int, dict[str, Any]]:
        cid = condition.condition_id
        code = code_state()
        # Confirmed for this run's calls alone, never in the process environment, which other threads read.
        confirmed = spend_confirmed() if confirm_spend else nullcontext()
        with confirmed, guard_work_directories(_work_directory_guard(self)), \
                guard_run_writes(_run_write_guard(self, condition)), hold_shared_writes(self._writing):
            judge = self.start(cid, relocate_batch_root=relocate_batch_root)
            try:
                # Again once the judge has started: an engine build can take minutes, and the records may have changed.
                self._owned()
                self._run_destinations(condition)
                self._check_started(condition, judge)
                digest = code.get("code_state_sha256")
                if digest and not snapshot_path(self.root, digest).exists():
                    with self._writing():
                        write_code_snapshot(self.root, code)
                plan = CampaignPlan(
                    campaign_id=self.campaign_id, campaign_root=self.root, config=self.config,
                    scenario_manifest=self.scenario_manifest, scenario_rows=self._judged_scenarios(),
                    condition=condition.spec, rules=condition.judged_rules(),
                    judge_type=condition.spec.judge_type, decoder_draw_ids=pending,
                    max_transport_retries=self.sampling.max_transport_retries, code_state=code,
                    rule_levels=condition.levels)
                judging.append(True)
                return run_plan(plan, backend=judge, task_spec=task_spec, execution=execution, gate=False)
            except BatchDestinationRefused as error:
                raise CampaignError(f"condition {cid!r}: {error}") from error
            finally:
                judge.close()

    def _check_started(self, condition: Condition, judge: Judge) -> None:
        """Refuse a started judge that is not the one ``condition`` records, before any draw is written."""
        spec = condition.spec
        try:
            identity = judge.identity()
            started = condition_spec_for(
                condition_id=spec.condition_id, config=self.config, judge=identity,
                task={name: getattr(spec, name) for name in _TASK_IDENTITY_FIELDS}, ruleset=condition.rules,
                evaluation_mode=spec.evaluation_mode, rule_levels_sha256=spec.rule_levels_sha256,
                extra=dict(spec.extra))
            sends = (judge.spec.request_settings(), _batch_root(judge.spec))
        except (JudgeError, SamplingDesignError, CampaignRunError, AttributeError) as error:
            raise CampaignError(f"condition {spec.condition_id!r}: the identity or request settings of the judge "
                                f"started cannot be read ({error}); nothing of it is judged") from error
        if started.fingerprint != spec.fingerprint:
            differ = {item.name: (getattr(spec, item.name), getattr(started, item.name))
                      for item in dataclasses.fields(ConditionSpec)
                      if getattr(started, item.name) != getattr(spec, item.name)}
            raise CampaignError(
                f"the judge started for condition {spec.condition_id!r} is not the judge it was registered with "
                f"({differ}, each field as registered and as started): its draws would be recorded under the "
                "registered identity, so nothing of it is judged; bind the judge again (Campaign.condition)")
        held = {name: (getattr(identity, name), getattr(judge, name)) for name in _LIVE_IDENTITY_FIELDS
                if hasattr(judge, name) and getattr(judge, name) != getattr(identity, name)}
        if held:
            raise CampaignError(
                f"the judge started for condition {spec.condition_id!r} does not generate as its identity says "
                f"({held}, each field as its identity reports it and as the judge holds it): its draws would be "
                "made and recorded with what it holds under the registered identity, so nothing of it is judged; "
                "bind the judge again (Campaign.condition)")
        record = self.execution_record(spec.condition_id)
        recorded = (record.get("request_settings"), record.get("batch_root"))
        if sends != recorded:
            raise CampaignError(
                f"the judge started for condition {spec.condition_id!r} sends request settings {sends[0]} and keeps "
                f"its batch state under {sends[1]}, and its execution record holds {recorded[0]} and {recorded[1]}: "
                "its blocks would be bought under the recorded block ids with other request bytes, or bought again "
                "elsewhere, so nothing of it is judged; bind the judge again (Campaign.condition)")

    def _judged_scenarios(self) -> tuple[dict[str, Any], ...]:
        return self._scenario_rows.judged(self.scenarios.rows, f"the scenarios of campaign {self.campaign_id!r}")

    def _run_destinations(self, condition: Condition) -> None:
        cid = condition.condition_id
        for relative in (COMPLETION_FNAME, SNAPSHOT_DIRNAME, cid):
            _destination(self.root, relative)
        _refuse_links(self.root / SNAPSHOT_DIRNAME, "the code snapshots' directory")
        _refuse_links(self.root / cid, f"condition {cid!r}'s directory")
        registered = _destination(self.root, Path(cid) / CONDITION_FNAME)
        if not registered.is_file():
            raise CampaignError(f"{registered} is gone: condition {cid!r} is no longer registered in {self.root}; "
                                "register it again (Campaign.condition)")
        try:
            check_registered_condition(self.root, condition.spec)
        except CampaignRunError as error:
            raise CampaignError(f"{error}; bind the condition the root registers (Campaign.condition)") from error


    def register(self, members: Sequence[Union[Sequence[Any], Condition]]) -> tuple[Condition, ...]:
        """Register several conditions as one: all are checked first, so a refusal of any registers none. Each member is
        ``(condition_id, rules, judge, settings)`` or a prepared ``Condition``. On a draft nothing is written."""
        prepared = [member if isinstance(member, Condition) else
                    self._prepare(member[0], rules=member[1], judge=member[2], settings=member[3])
                    for member in members]
        foreign = sorted(condition.condition_id for condition in prepared if condition.campaign is not self)
        if foreign:
            raise CampaignError(f"conditions {foreign} were prepared by another campaign object; register them here "
                                "by their arguments (condition_id, rules, judge, settings)")
        ids = [condition.condition_id for condition in prepared]
        repeated = sorted({cid for cid in ids if ids.count(cid) > 1})
        if repeated:
            raise CampaignError(f"members name conditions {repeated} more than once: each member of a sweep is a "
                                "condition of its own (give each its own id, or let its fingerprint name it)")
        identities: dict[str, str] = {}
        for condition in prepared:
            other = identities.setdefault(condition.spec.fingerprint, condition.condition_id)
            if other != condition.condition_id:
                raise CampaignError(
                    f"members {other!r} and {condition.condition_id!r} are one condition (fingerprint "
                    f"{condition.spec.fingerprint[:12]}...): they judge the same rules, judge, task settings, mode and "
                    "extra, so they would be one condition under two ids (two spellings of one value: a prompt alias "
                    "and its style, a judge named two ways); give each member a value of its own")

        def refuse_joining() -> None:
            unregistered = self._joining_analysis(ids)
            if unregistered:
                raise CampaignError(
                    f"{self.root} has been analysed ({self._estimates_path()}), and conditions {unregistered} are not "
                    f"registered there: a condition joins an analysed root explicitly. Register it (`ruleforge "
                    f"campaign add {self.root} ...`, Campaign.condition), run it (`ruleforge campaign run {self.root} "
                    f"CID ...`), then analyse the root again (`ruleforge analyze {self.root} --replace`)")

        refuse_joining()
        if not self.writable:
            for condition in prepared:
                self._registration(condition)
            self._one_configuration(prepared)
            return tuple(prepared)
        with self._writing():
            refuse_joining()
            pending = [self._registration(condition) for condition in prepared]
            self._one_configuration(prepared)
            for condition, found in zip(prepared, pending):
                self._write_registration(condition, found)
        self._bound.update((condition.condition_id, condition) for condition in prepared)
        return tuple(prepared)

    def sweep(self, members: Sequence[Sequence[Any]], *, execution: ExecutionSettings = ExecutionSettings(),
              confirm_spend: bool = False, relocate_batch_root: bool = False) -> CampaignResult:
        """Register the conditions, judge each in turn, analyse the root once and return its result; a root holding an
        accepted analysis is read with nothing judged, and a draw out of attempts is refused."""
        self._writable()
        if not members:
            raise CampaignError("a sweep of no member: give it one condition at least")
        conditions = self.register(members)
        ids = [condition.condition_id for condition in conditions]
        accepted = self.accepted_analysis(ids)
        if accepted is not None:
            return accepted
        for condition in conditions:
            self._held_to_record(condition.condition_id, condition.judge, relocate_batch_root=relocate_batch_root)
        for condition in conditions:
            self.run(condition.condition_id, execution=execution, confirm_spend=confirm_spend,
                     relocate_batch_root=relocate_batch_root)
        self._refuse_exhausted_draws(conditions[0].condition_id, analysed=False)
        if any(condition.spec.evaluation_mode == EVALUATION_MODE_BOTH for condition in conditions):
            try:
                return self.analyze()
            except AnalysisExists:
                pass
        return self._accepted(ids, collected=True)

    def accepted_analysis(self, condition_ids: Sequence[str]) -> Optional[CampaignResult]:
        """Return the root's result when ``condition_ids`` accept its analysis, None without one; refused for recorded
        exclusions, an exhausted draw, an incomplete condition or differing settings."""
        return self._accepted(condition_ids, collected=False)

    def _accepted(self, condition_ids: Sequence[str], *, collected: bool) -> Optional[CampaignResult]:
        if not condition_ids:
            raise CampaignError("a read of no condition: name one at least")
        payload, digest = self._estimates_file()
        if payload is None and not collected:
            return None

        def read() -> CampaignResult:
            registered, gate = self._snapshot()
            if payload is not None:
                both = any(spec.evaluation_mode == EVALUATION_MODE_BOTH for spec in registered.values())
                _accept_analysis(self, payload, self.sampling, list(condition_ids), gate=gate, registered=registered,
                                 both=both, whole=True)
            return self._campaign_result(registered, gate, payload, digest)

        return self._read_records(read)


    def _estimates_path(self) -> Path:
        return self.root / AGGREGATE_DIRNAME / ESTIMATES_FNAME

    def _refuse_replacing(self, path: Path, replace: bool) -> None:
        if path.exists() and not replace:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            raise AnalysisExists(f"{path} holds this root's analysis (sha256 {digest}); a new analysis replaces it "
                                "only when asked (replace=True, --replace), and changes the bytes anything built "
                                "from it pinned")

    def analyze(self, *, allow_incomplete: bool = False, allow_stack_mismatch: bool = False,
                rule_set: Optional[Union[Ruleset, Path, str]] = None,
                exclude_exhausted: Optional[tuple[str, str]] = None, n_resamples: Optional[int] = None,
                ci_level: Optional[float] = None, seed: Optional[int] = None, n_permutations: Optional[int] = None,
                replace: bool = False) -> CampaignResult:
        """Write the root's analysis of its ``both`` conditions to ``aggregate/estimates.json`` and return it, strict
        unless ``allow_incomplete`` or ``exclude_exhausted``, replacing a file only under ``replace``; ``n_resamples``,
        ``ci_level`` and ``seed`` override the manifest's bootstrap."""
        self._owned()
        refuse_stripped(self.root)
        if (self.root / SHARED_BANK_DESIGN_FNAME).exists():
            raise CampaignError(f"{self.root} is a shared-bank campaign ({SHARED_BANK_DESIGN_FNAME}); its bank and subset "
                                "conditions are analysed by ruleforge.evaluate.bank.analyze")
        if not any(spec.evaluation_mode == EVALUATION_MODE_BOTH for spec in load_conditions(self.root).values()):
            raise CampaignError(f"{self.root} registers no {EVALUATION_MODE_BOTH!r} condition: its analysis estimates "
                                "inconsistency, which a 'full' or 'individual_bank' condition (a collection) does "
                                "not carry alone; read those with result(), or analyse a shared bank with "
                                "ruleforge.evaluate.bank.analyze")
        target = self._estimates_path()
        self._refuse_replacing(target, replace)
        _refuse_linked_campaign_lock(self.root)
        # Read under the lock it was written under, so another process's analysis is not returned as this call's.
        with self._writing():
            self._refuse_replacing(target, replace)
            for relative in (COMPLETION_FNAME, Path(AGGREGATE_DIRNAME) / ESTIMATES_FNAME):
                _destination(self.root, relative)
            _refuse_links(self.root / AGGREGATE_DIRNAME, "the analysis's directory")
            _refuse_linked_campaign_lock(self.root)
            analyze_campaign(
                self.root, n_resamples=n_resamples, ci_level=ci_level, seed=seed,
                **({} if n_permutations is None else {"n_permutations": n_permutations}),
                allow_incomplete_conditions=allow_incomplete, allow_stack_mismatch=allow_stack_mismatch,
                rule_set_path=None if rule_set is None else getattr(rule_set, "path", rule_set),
                exclude_exhausted=None if exclude_exhausted is None else tuple(exclude_exhausted),
                evaluation_modes=(EVALUATION_MODE_BOTH,))
            payload, digest = self._estimates_file()
        return self._read_records(lambda: self._campaign_result(*self._snapshot(), payload, digest))

    def kappa(self, condition_id: str, *, strata: str = KAPPA_STRATUM_FIELD, n_resamples: Optional[int] = None,
              ci_level: Optional[float] = None, seed: Optional[int] = None) -> Kappa:
        """Return a complete ``both`` condition's stratified Cohen's kappa between full-set judgments and per-rule
        maxima, chance taken within the scenario field ``strata``; None at chance agreement 1, never written."""
        with self._reading():
            return self._kappa(condition_id, strata=strata, n_resamples=n_resamples, ci_level=ci_level, seed=seed)

    def _kappa(self, condition_id: str, *, strata: str, n_resamples: Optional[int], ci_level: Optional[float],
               seed: Optional[int]) -> Kappa:
        registered, gate = self._snapshot()
        spec = registered.get(condition_id)
        if spec is None:
            raise CampaignError(f"{self.root} registers no condition {condition_id!r}")
        if spec.evaluation_mode != EVALUATION_MODE_BOTH:
            raise CampaignError(f"condition {condition_id!r} is a {spec.evaluation_mode!r} condition; kappa reads a "
                                f"{EVALUATION_MODE_BOTH!r} condition's full-set judgments beside its per-rule maxima")
        draws = gate["conditions"][condition_id]["draws"]
        incomplete = {int(draw): record["state"] for draw, record in draws.items()
                      if record["state"] != STATE_COMPLETE}
        if incomplete:
            raise CampaignError(f"condition {condition_id!r} is not complete (draws {incomplete}); kappa is computed "
                                "over every declared draw")
        manifest = load_sampling_manifest(self.root / SAMPLING_MANIFEST_FNAME)
        population = ScenarioManifest.load(self.root / SCENARIO_MANIFEST_FNAME)
        ids = population.scenario_ids
        n_draws = int(manifest["expected_decoder_draws"])
        cells, _ = load_condition_outcomes(self.root / condition_id, expected_scenario_ids=ids,
                                           expected_draws=n_draws,
                                           scenario_manifest_sha256=population.manifest_sha256)
        grids = VerdictGrids.from_cells(condition_id, cells, scenario_ids=ids, n_draws=n_draws,
                                        identity_id=spec.identity_id)
        try:
            groups = read_strata([self.root / condition_id], ids, strata)
        except DiscriminationError as error:
            raise CampaignError(f"condition {condition_id!r}: {error}") from error
        bootstrap = manifest["bootstrap"]
        return Kappa.from_summary(kappa_summary(
            grids, groups, n_resamples=int(bootstrap["n_resamples"] if n_resamples is None else n_resamples),
            ci_level=float(bootstrap["ci_level"] if ci_level is None else ci_level),
            seed=int(bootstrap["seed"] if seed is None else seed)))

    def _estimates_file(self) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        path = self._estimates_path()
        if not path.is_file():
            return None, None
        data = path.read_bytes()
        try:
            payload = json.loads(data)
        except ValueError as error:
            raise CampaignError(f"{path} cannot be read as an analysis ({error})") from error
        return payload, hashlib.sha256(data).hexdigest()

    def result(self, condition_id: Optional[str] = None) -> Union[EvalResult, CampaignResult]:
        """Return the recorded analysis as written (``result(cid)``: the collected condition with its entry;
        ``result()``: the whole file), computing nothing; refused when the root or analysis records another campaign."""
        with self._reading():
            registered, gate = self._snapshot()
            payload, digest = self._estimates_file()
            foreign = [] if payload is None else _foreign_analysis(
                payload, load_sampling_manifest(self.root / SAMPLING_MANIFEST_FNAME))
            if foreign:
                raise CampaignError(f"{self._estimates_path()} is not this root's analysis ({'; '.join(foreign)}); "
                                    f"analyse the root again (`ruleforge analyze {self.root} --replace`, "
                                    "analyze(replace=True))")
            if condition_id is not None:
                return self._collected(condition_id, registered, gate, payload, digest)
            return self._campaign_result(registered, gate, payload, digest)

    def _read_records(self, read: Callable[[], _Read]) -> _Read:
        with self._reading():
            return read()

    def _campaign_result(self, registered: Mapping[str, ConditionSpec], gate: Mapping[str, Any],
                         payload: Optional[dict[str, Any]], digest: Optional[str]) -> CampaignResult:
        path = self._estimates_path()
        if payload is None and any(spec.evaluation_mode == EVALUATION_MODE_BOTH for spec in registered.values()):
            raise CampaignError(f"{self.root} holds no analysis ({path}); write it with analyze() "
                                "(`ruleforge analyze ROOT`) once its conditions are complete")
        conditions = {cid: self._collected(cid, registered, gate, payload, digest) for cid in registered}
        if payload is None:
            return CampaignResult(path=None, payload={}, conditions=conditions, contrasts=(), excluded_conditions={},
                                  scenario_exclusion=None, stack_mismatches=())
        return CampaignResult(
            path=path, payload=payload, conditions=conditions, contrasts=contrasts_of(payload),
            excluded_conditions=dict(payload.get("excluded_conditions") or {}),
            scenario_exclusion=payload.get(SCENARIO_EXCLUSION_KEY),
            stack_mismatches=tuple(dict(record) for record in payload.get(STACK_MISMATCHES_KEY) or ()))

    def _collected(self, condition_id: str, registered: Mapping[str, ConditionSpec], gate: Mapping[str, Any],
                   payload: Optional[Mapping[str, Any]], digest: Optional[str]) -> EvalResult:
        spec = registered.get(condition_id)
        if spec is None:
            raise CampaignError(f"{self.root} registers no condition {condition_id!r}")
        entries = gate["conditions"][condition_id]["draws"]
        estimates = excluded = None
        if spec.evaluation_mode == EVALUATION_MODE_BOTH and payload is not None:
            path = self._estimates_path()
            if condition_id in payload.get("conditions", {}):
                if payload["conditions"][condition_id].get("condition_fingerprint") == spec.fingerprint:
                    estimates = estimates_of(spec, payload, path=path, sha256=str(digest))
            elif condition_id in (payload.get("excluded_conditions") or {}):
                excluded = dict(payload["excluded_conditions"][condition_id])
            else:
                raise CampaignError(
                    f"{path} names condition {condition_id!r} in neither its conditions nor its excluded ones: it was "
                    f"registered after the analysis. Run it (Campaign.run, `ruleforge campaign run {self.root} "
                    f"{condition_id} ...`), then analyse the root again (analyze(replace=True), `ruleforge analyze "
                    f"{self.root} --replace`)")
        outcomes = tuple(path for draw in sorted(int(key) for key in entries)
                         if (path := self.root / condition_id / draw_dirname(draw) / BUNDLE_OUTCOMES_FNAME).is_file())
        return EvalResult(
            root=self.root, campaign_id=self.campaign_id, condition_id=condition_id, fingerprint=spec.fingerprint,
            mode=spec.evaluation_mode, draws={int(draw): record["state"] for draw, record in entries.items()},
            attempts=attempt_statistics([record for record in entries.values() if record.get("recorded")]),
            outcomes=outcomes, estimates=estimates, excluded=excluded)


def refuse_unconfirmed_spend(condition: Condition, *, draw_ids: Optional[Sequence[int]] = None,
                             analysis_read: bool = False) -> None:
    """Refuse, before anything is written, a drafted run of a billed judge with no spend confirmed that has draws to
    judge (``Campaign._pending``), once the caller has made the root's refusals (``Campaign.check_create``). With
    ``analysis_read`` a root holding an analysis is read, not judged, as ``evaluate`` and a sweep read it."""
    draft = condition.campaign
    draft._registration(condition)  # what registering it refuses, as the run registers it first
    if analysis_read and draft._estimates_path().is_file():
        return
    if draft._pending(condition.condition_id, draft._requested(draw_ids)):
        Campaign._refuse_unconfirmed(condition, confirm_spend=False)


def _unsaved(ruleset: Ruleset) -> CampaignError:
    return CampaignError(f"ruleset {ruleset.name!r} has no file: a condition names its ruleset by the file's name and "
                         "bytes; write its rules to a JSONL file, one {\"id\": ..., \"rule\": ...} object per line, "
                         "and load that (Ruleset.load(path))")


def default_campaign_id(manifest: ScenarioManifest, config: SamplingConfig) -> str:
    """Return ``eval-<12 hex>``, a digest of the population and config; the id enters every generation seed."""
    digest = hashlib.sha256((manifest.manifest_sha256 + canonical_json(config.to_dict())).encode("utf-8"))
    return f"eval-{digest.hexdigest()[:12]}"


def default_root(campaign_id: str) -> Path:
    """Return ``<artifact root>/sampling/<campaign id>``, or ``./ruleforge-runs/<campaign id>`` when no artifact root is
    set."""
    base = env_value("ARTIFACT_ROOT")
    return Path(base) / "sampling" / campaign_id if base else Path(DEFAULT_RUNS_DIRNAME) / campaign_id


_SAMPLING_FIELDS = frozenset(item.name for item in dataclasses.fields(SamplingSettings))
_CONDITION_FIELDS = frozenset(item.name for item in dataclasses.fields(ConditionSettings))


def _with_overrides(sampling: SamplingSettings, condition: ConditionSettings,
                    overrides: Mapping[str, Any]) -> tuple[SamplingSettings, ConditionSettings]:
    unknown = sorted(set(overrides) - _SAMPLING_FIELDS - _CONDITION_FIELDS)
    if unknown:
        raise TypeError(f"evaluate() got unknown settings {unknown}: a keyword names a field of SamplingSettings "
                        f"({sorted(_SAMPLING_FIELDS)}) or ConditionSettings ({sorted(_CONDITION_FIELDS)})")
    return (dataclasses.replace(sampling, **{key: value for key, value in overrides.items()
                                             if key in _SAMPLING_FIELDS}),
            dataclasses.replace(condition, **{key: value for key, value in overrides.items()
                                              if key in _CONDITION_FIELDS}))


def _judge_flag(given: Any, judge: JudgeSpec) -> Optional[str]:
    """Return a ``--judge`` value that rebuilds ``judge`` whole (what it sends, not only what it records), or None."""
    if isinstance(given, str):
        return given
    candidates = (_shorthand(judge), judge.model_name)
    return next((flag for flag in candidates if flag is not None and _rebuilds(flag, judge)), None)


def _shorthand(judge: JudgeSpec) -> Optional[str]:
    if judge.model_type == "fake":
        return f"fake:{judge.options['seed']}"
    if judge.model_type == "local":
        return f"local:{judge.options['model_path']}"
    if judge.model_type in ("openai", "anthropic"):
        return f"{judge.model_type}:{judge.model_name}"
    if judge.model_type == COMPATIBLE_MODEL_TYPE:
        return f"{judge.model_type}:{judge.model_name}@{judge.options['base_url']}"
    return None


def _rebuilds(flag: str, judge: JudgeSpec) -> bool:
    try:
        return resolve(flag) == judge
    except (ValueError, TypeError, OSError, OmegaConfBaseException):
        return False


def _run_command(root: Path, condition: Condition, judge_flag: str, scenarios: Path) -> str:
    parts = ["ruleforge", "campaign", "run", str(root), condition.condition_id, "--judge", judge_flag,
             "--rules", str(condition.rules.path), "--scenarios", str(scenarios)]
    if condition.settings.rule_levels is not None:
        parts += ["--rule-levels", str(condition.settings.rule_levels)]
    if condition.task_locator != DEFAULT_TASK:
        parts += ["--task", condition.task_locator]
    return " ".join(shlex.quote(part) for part in parts)


def _refuse_exhausted(root: Path, condition_id: str, gate: Mapping[str, Any],
                      registered: Mapping[str, ConditionSpec], *, analysed: bool) -> None:
    found = [(cid, int(draw), list(record.get("exhausted") or ()))
             for cid, entry in gate["conditions"].items()
             for draw, record in entry["draws"].items()
             if record["state"] == STATE_INCOMPLETE_EXHAUSTED and draw_ended(record)]
    if not found:
        return
    listed = "; ".join(f"{cid} draw {draw}: {', '.join(sid[:12] for sid in sids)}" for cid, draw, sids in found)
    cid, _, scenarios = found[0]
    spec = registered[cid]
    if spec.evaluation_mode != EVALUATION_MODE_BOTH:
        raise CampaignError(f"{root}: draws are out of attempts ({listed}), in a {spec.evaluation_mode!r} condition, "
                            "which no analysis's fallback covers; collect it again in a campaign of its own with "
                            "more attempts (max_attempts)")
    both = {name: entry for name, entry in registered.items() if entry.evaluation_mode == EVALUATION_MODE_BOTH}
    unserved = fallback_unserved(both, spec.model_name)
    if unserved:
        raise CampaignError(f"{root}: draws are out of attempts ({listed}), and the fallback does not serve judge "
                            f"{spec.model_name!r}'s arms {unserved} (forced-style or policy arms, whose readings take "
                            "every draw whole); collect the condition again in a campaign of its own with more "
                            "attempts (max_attempts)")
    command = f"ruleforge analyze {root} --exclude-exhausted {shlex.quote(spec.model_name)} {scenarios[0]}"
    raise CampaignError(
        f"{root}: draws are out of attempts ({listed}), so the strict gate never passes. The fallback drops one "
        f"exhausted scenario from every arm of its judge: `{command}{' --replace' if analysed else ''}` "
        f"(Campaign.analyze(exclude_exhausted=({spec.model_name!r}, {scenarios[0]!r}))), then read it with "
        f"Campaign.open({str(root)!r}).result({condition_id!r})")


def _foreign_analysis(payload: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    return [f"{key} {payload.get(key)!r} (the manifest's {manifest[source]!r})"
            for key, source in (("campaign_id", "campaign_id"),
                                ("scenario_manifest_sha256", "scenario_manifest_sha256"),
                                ("n_decoder_draws", "expected_decoder_draws"),
                                ("max_semantic_attempts", "max_semantic_attempts"))
            if payload.get(key) != manifest[source]]


def _settings_mismatch(payload: Mapping[str, Any], manifest: Mapping[str, Any], sampling: SamplingSettings,
                       registered: Mapping[str, ConditionSpec]) -> list[str]:
    differ = _foreign_analysis(payload, manifest)
    requested = dataclasses.asdict(sampling.bootstrap)
    if payload.get("bootstrap") != manifest["bootstrap"] or payload.get("bootstrap") != requested:
        differ.append(f"bootstrap {payload.get('bootstrap')} (the manifest's {manifest['bootstrap']}, the request's "
                      f"{requested})")
    for cid, spec in registered.items():
        if spec.evaluation_mode != EVALUATION_MODE_BOTH:
            continue
        entry = (payload.get("conditions") or {}).get(cid)
        if entry is None or entry.get("condition_fingerprint") != spec.fingerprint:
            differ.append(f"condition {cid!r} ({'absent' if entry is None else 'another fingerprint'})")
    return differ


def evaluate(rules: Union[Ruleset, Path, str], scenarios: Union[ScenarioSet, Path, str], judge: Any, *,
             out: Optional[Union[Path, str]] = None, sampling: SamplingSettings = SamplingSettings(),
             condition: ConditionSettings = ConditionSettings(), campaign_id: Optional[str] = None,
             condition_id: Optional[str] = None, confirm_spend: bool = False,
             execution: ExecutionSettings = ExecutionSettings(), relocate_batch_root: bool = False,
             **overrides: Any) -> EvalResult:
    """Evaluate one ruleset over one scenario set with one judge (a ``JudgeSpec`` or a name ``ruleforge.judges.resolve``
    reads) as a one-condition campaign, ``overrides`` setting ``sampling`` and ``condition`` fields; an analysed root is
    read, one without analysis judged first, others refused."""
    sampling, condition = _with_overrides(sampling, condition, overrides)
    ruleset = rules if isinstance(rules, Ruleset) else Ruleset.load(rules)
    if ruleset.path is None:
        raise _unsaved(ruleset)
    population = scenarios if isinstance(scenarios, ScenarioSet) else ScenarioSet.load(scenarios)
    _task_of(condition.task, population, "the scenarios", keyed=_keyed(condition.mode))
    spec = resolve(judge)
    campaign_id = campaign_id or default_campaign_id(population.manifest, sampling.config())
    root = out if out is not None else default_root(campaign_id)
    if spec.capabilities().spends_money and not confirm_spend:
        Campaign.check_create(root, campaign_id=campaign_id, scenarios=population, sampling=sampling)
        draft = Campaign.draft(root, campaign_id=campaign_id, scenarios=population, sampling=sampling)
        refuse_unconfirmed_spend(draft.prepare(condition_id, rules=ruleset, judge=spec, settings=condition),
                                 analysis_read=True)
    campaign = Campaign.create(root, campaign_id=campaign_id, scenarios=population, sampling=sampling)
    root = campaign.root
    prepared = campaign._prepare(condition_id, rules=ruleset, judge=spec, settings=condition)
    cid = prepared.condition_id
    flag = _judge_flag(judge, spec)
    resume = None if flag is None else _run_command(root, prepared, flag, population.path)
    run_it = f"`{resume}`" if resume is not None else f"`ruleforge campaign run {root} {cid} ...`"

    def refuse_joining() -> None:
        if campaign._joining_analysis([cid]):
            raise CampaignError(
                f"{root} has been analysed ({campaign._estimates_path()}), and condition {cid!r} is not registered "
                f"there: a condition is added beside an analysis explicitly. Register it (`ruleforge campaign add "
                f"{root} ...`, Campaign.condition), run it ({run_it}), analyse the root again (`ruleforge analyze "
                f"{root} --replace`), then read it with Campaign.open({str(root)!r}).result({cid!r})")

    refuse_joining()  # again under the root's lock, where an analysis written meanwhile is seen
    campaign._register(prepared, check=refuse_joining)
    payload, digest = campaign._estimates_file()
    if payload is None:
        both = prepared.spec.evaluation_mode == EVALUATION_MODE_BOTH
        campaign._refuse_exhausted_draws(cid, analysed=False)
        if both:
            _refuse_linked_campaign_lock(root)
        campaign.run(cid, execution=execution, confirm_spend=confirm_spend, relocate_batch_root=relocate_batch_root)
        campaign._refuse_exhausted_draws(cid, analysed=False)
        if both:
            try:
                analysed = campaign.analyze()
            except AnalysisExists:
                # Another process analysed the root since this call looked.
                payload, digest = campaign._estimates_file()
            else:
                return analysed.conditions[cid]
        else:
            # An analysis another process wrote during a collection is still accepted or refused as one found here.
            payload, digest = campaign._estimates_file()
            if payload is None:
                return campaign._read_records(lambda: campaign._collected(cid, *campaign._snapshot(), None, None))
    return _read_analysed(campaign, prepared, payload, digest, sampling, resume)


def _read_analysed(campaign: Campaign, prepared: Condition, payload: Optional[dict[str, Any]], digest: Optional[str],
                   sampling: SamplingSettings, resume: Optional[str]) -> EvalResult:
    cid = prepared.condition_id
    if payload is None:
        raise CampaignError(f"{campaign._estimates_path()} was removed while this call read it; run it again")
    def read() -> EvalResult:
        registered, gate = campaign._snapshot()
        _accept_analysis(campaign, payload, sampling, [cid], gate=gate, registered=registered,
                         both=prepared.spec.evaluation_mode == EVALUATION_MODE_BOTH, resume=resume)
        return campaign._collected(cid, registered, gate, payload, digest)

    return campaign._read_records(read)


def _accept_analysis(campaign: Campaign, payload: Mapping[str, Any], sampling: SamplingSettings,
                     condition_ids: Sequence[str], *, gate: Mapping[str, Any],
                     registered: Mapping[str, ConditionSpec], both: bool, whole: bool = False,
                     resume: Optional[str] = None) -> None:
    root, cid = campaign.root, condition_ids[0]
    read = f"Campaign.open({str(root)!r}).result({'' if whole else repr(cid)})"
    if payload.get("excluded_conditions") or SCENARIO_EXCLUSION_KEY in payload:
        raise CampaignError(f"{campaign._estimates_path()} records excluded conditions or the fallback: an analysis "
                            f"with exclusions is read as written, with its disclosures, by {read}")
    _refuse_exhausted(root, cid, gate, registered, analysed=True)
    incomplete = sorted(name for name, entry in gate["conditions"].items()
                        if any(record["state"] != STATE_COMPLETE for record in entry["draws"].values()))
    if incomplete:
        commands = (f"`{resume}`" if resume is not None and incomplete == [cid]
                    else f"`ruleforge campaign run {root} CID ...` for each")
        raise CampaignError(
            f"{root} has been analysed, and conditions {incomplete} are not complete beside the analysis. Run them "
            f"({commands}), analyse the root again (`ruleforge analyze {root} --replace`), then read it with {read}")
    if both:
        differ = _settings_mismatch(payload, load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME), sampling,
                                    registered)
        if differ:
            raise CampaignError(f"{campaign._estimates_path()} was analysed under other settings than this root's and "
                                f"this request's ({'; '.join(differ)}); analyse it again (`ruleforge analyze {root} "
                                f"--replace`, analyze(replace=True)), then read it with {read}")


__all__ = [
    "ARTIFACT_ROOT_ENV", "AnalysisExists", "Approvals", "Campaign", "CampaignError", "Condition", "ConditionSettings",
    "DEFAULT_BATCH_DIRNAME", "EXECUTION_FNAME", "LOCK_FNAME", "MARKER_FNAME", "RUN_LOCK_FNAME",
    "ReadOnlyCampaign", "SamplingSettings", "batch_work_guard", "bound_batch_root", "default_campaign_id",
    "default_root", "evaluate", "pinned_roots", "refuse_batch_root", "refuse_pinned", "refuse_protected",
    "refuse_unconfirmed_spend", "refuse_work_directory",
]
