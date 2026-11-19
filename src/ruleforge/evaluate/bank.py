from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Mapping, Optional, Sequence

import numpy as np

from ruleforge.evaluate.inference import (
    CURVE_INFERENCE_METHOD,
    CURVE_INTERVAL_BASIS,
    RepeatedSamplingError,
    bank_shared_curve_bootstrap,
    load_condition_outcomes,
)
from ruleforge.rulesets.subsets import (
    VALID_SAMPLING_STRATEGIES,
    file_digest,
    load_manifest,
    seed_dir,
)
from ruleforge.evaluate.api import (
    AnalysisExists,
    Campaign,
    CampaignError,
    _destination,
    _owner_writing,
    _refuse_foreign,
    _refuse_links,
    _work_directory_guard,
    refuse_protected,
)
from ruleforge.evaluate.scoring import PHI_MAP, max_rule_phi
from ruleforge.evaluate.aggregate import AGGREGATE_DIRNAME, attempt_statistics
from ruleforge.evaluate.amendment import AmendmentError, cap_amendment
from ruleforge.evaluate.amendment import disclosure as amendment_disclosure
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    campaign_completion,
    load_conditions,
    non_meta_rule_count,
    read_condition,
    require_complete_campaign,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_SCREENING,
    COMPLETION_FNAME,
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_FULL,
    SHARED_BANK_DESIGN_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    ConditionSpec,
    differing_manifest_fields,
    load_sampling_manifest,
    refuse_published,
    retry_scope_record,
)
from ruleforge.scenarios.manifest import (
    SAMPLING_SCHEMA_VERSION,
    ScenarioManifest,
)
from ruleforge.core import record_names
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import (
    canonical_digest,
    sha256_file,
)
from ruleforge.core.files import write_json_sidecar
from ruleforge.core.paths import artifact_uri_or_path, repository_root
from ruleforge.core.provenance import SNAPSHOT_DIRNAME
from ruleforge.evaluate.runner import STATE_COMPLETE, CampaignRunError, utc_now
from ruleforge.evaluate.stack import (
    LOCAL_MODEL_TYPE,
    EngineArgsIndex,
    Signature,
    condition_signatures,
    describe_mismatches,
    override_record,
    shared_stack_conflicts,
    stacks_record,
)
from ruleforge.core.artifacts import iter_jsonl_records, load_jsonl_records

logger = logging.getLogger(__name__)

DESIGN_SHARED_BANK = "shared_individual_bank_v1"
DESIGNS = (DESIGN_SHARED_BANK,)
DESIGN_CONFIG_KEY = "ruleset_size"
SHARED_BANK_ESTIMATES_FNAME = record_names.SHARED_BANK_ESTIMATES_FNAME
BANK_SEGMENT = "bank"
SUBSETS_SEGMENT = "subsets"
ROLE_BANK = "bank"
ROLE_SUBSET = "subset"
_SUBSET_ID = re.compile(rf"^(?P<model>.+)/{SUBSETS_SEGMENT}/s(?P<seed>\d{{2,}})/k(?P<size>\d{{2,}})$")
_BANK_ID = re.compile(rf"^(?P<model>.+)/{BANK_SEGMENT}$")

# The contrasts each curve carries: the net rise, the upper range against the middle, and the last step.
UPPER_SIZE = 56
MIDDLE_SIZES = tuple(range(20, 31))
# A model's sizes at which every subset seed judged one ruleset, so its seeds are replicate draws of it.
ONE_RULESET_SIZES_KEY = "one_ruleset_sizes"
# An analysis's record of the root its banks and the sizes its own design lacks were read from.
BANK_ROOT_KEY = "bank_root"
SUBSET_MANIFESTS = Path("data") / "rules"


class SharedBankDesignError(SamplingDesignError):
    """The shared-bank design cannot be built or read."""


@dataclass(frozen=True)
class SharedBankDesign:
    """What a shared-bank campaign was asked to generate, frozen at the root."""

    subset_seeds: tuple[int, ...]
    subset_sizes: tuple[int, ...]
    sampling_strategy: str
    subset_manifest_sha256: str
    source_ruleset_sha256: str
    design: str = DESIGN_SHARED_BANK

    def __post_init__(self) -> None:
        if self.design not in DESIGNS:
            raise SharedBankDesignError(f"design must be one of {DESIGNS}, got {self.design!r}")
        if self.sampling_strategy not in VALID_SAMPLING_STRATEGIES:
            raise SharedBankDesignError(
                f"sampling_strategy must be one of {VALID_SAMPLING_STRATEGIES}, got "
                f"{self.sampling_strategy!r}"
            )
        seeds = tuple(int(seed) for seed in self.subset_seeds)
        sizes = tuple(int(size) for size in self.subset_sizes)
        if len(seeds) < 2 or len(set(seeds)) != len(seeds) or min(seeds) < 0:
            raise SharedBankDesignError(
                f"subset_seeds must be at least two distinct non-negative seeds, got {seeds}"
            )
        # One size is a design too: it adds that size's subsets to a curve whose other sizes another root holds.
        if not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 2:
            raise SharedBankDesignError(
                f"subset_sizes must be one or more distinct sizes of at least 2, got {sizes}"
            )
        object.__setattr__(self, "subset_seeds", seeds)
        object.__setattr__(self, "subset_sizes", tuple(sorted(sizes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "subset_seeds": list(self.subset_seeds),
            "subset_sizes": list(self.subset_sizes),
            "sampling_strategy": self.sampling_strategy,
            "subset_manifest_sha256": self.subset_manifest_sha256,
            "source_ruleset_sha256": self.source_ruleset_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SharedBankDesign":
        try:
            return cls(**{key: payload[key] for key in cls.__dataclass_fields__})
        except KeyError as error:
            raise SharedBankDesignError(f"malformed shared-bank design: missing {error}") from error

    @property
    def n_conditions_per_model(self) -> int:
        return 1 + len(self.subset_seeds) * len(self.subset_sizes)


def write_design(campaign_root: Path | str, design: SharedBankDesign) -> Path:
    """Freeze the design at the root; refuse a different one already there."""
    path = Path(campaign_root) / SHARED_BANK_DESIGN_FNAME
    payload = {"schema_version": SAMPLING_SCHEMA_VERSION, **design.to_dict()}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SharedBankDesignError(
                f"{path} freezes a different shared-bank design; a campaign root holds one "
                "design, so run a different design under a different campaign_id"
            )
        return path
    return write_json_sidecar(path, payload)


def load_design(campaign_root: Path | str) -> SharedBankDesign:
    path = Path(campaign_root) / SHARED_BANK_DESIGN_FNAME
    if not path.exists():
        raise SharedBankDesignError(f"{path} is missing: not a shared-bank campaign root")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SAMPLING_SCHEMA_VERSION:
        raise SharedBankDesignError(
            f"{path} has schema_version {payload.get('schema_version')!r}, not "
            f"{SAMPLING_SCHEMA_VERSION}"
        )
    return SharedBankDesign.from_dict(payload)


def bank_condition_id(model_name: str) -> str:
    return f"{model_name}/{BANK_SEGMENT}"


def subset_condition_id(model_name: str, seed: int, size: int) -> str:
    return f"{model_name}/{SUBSETS_SEGMENT}/s{int(seed):02d}/k{int(size):02d}"


def parse_condition_id(condition_id: str) -> tuple[str, str, Optional[int], Optional[int]]:
    """``(model, role, seed, size)`` for a bank or subset id; anything else is refused."""
    match = _SUBSET_ID.match(condition_id)
    if match:
        return match["model"], ROLE_SUBSET, int(match["seed"]), int(match["size"])
    match = _BANK_ID.match(condition_id)
    if match:
        return match["model"], ROLE_BANK, None, None
    raise SharedBankDesignError(
        f"{condition_id!r} is neither a bank (<model>/bank) nor a subset "
        f"(<model>/subsets/sNN/kNN) condition"
    )


def subset_path(subset_root: Path | str, sampling_strategy: str, seed: int, size: int) -> Path:
    return seed_dir(Path(subset_root), sampling_strategy, int(seed)) / f"subset_size_{int(size):02d}.jsonl"


def design_manifest(manifest_dir: Path | str, design: SharedBankDesign) -> dict[str, str]:
    """Return the digests the subset manifest in ``manifest_dir`` pins, refused unless it is ``design``'s: its checksum
    file, strategy and source ruleset."""
    manifest_path = Path(manifest_dir) / "manifest.sha256"
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != design.subset_manifest_sha256:
        raise SharedBankDesignError(
            f"{manifest_path} digests to {actual_manifest[:12]}..., not the design's "
            f"{design.subset_manifest_sha256[:12]}..."
        )
    meta, pinned = load_manifest(manifest_dir)
    source = meta.get("source_ruleset", {})
    if source.get("sha256") != design.source_ruleset_sha256:
        raise SharedBankDesignError(
            f"the subset manifest's source ruleset {str(source.get('sha256'))[:12]}... is not "
            f"the design's {design.source_ruleset_sha256[:12]}..."
        )
    if meta.get("sampling_strategy") != design.sampling_strategy:
        raise SharedBankDesignError(
            f"the subset manifest is {meta.get('sampling_strategy')!r}, the design "
            f"{design.sampling_strategy!r}"
        )
    return pinned


def verify_subsets(
    subset_root: Path | str,
    manifest_dir: Path | str,
    design: SharedBankDesign,
) -> dict[tuple[int, int], str]:
    """Return ``{(seed, size): digest}``, every subset file checked against the committed manifest."""
    pinned = design_manifest(manifest_dir, design)
    digests: dict[tuple[int, int], str] = {}
    for seed in design.subset_seeds:
        for size in design.subset_sizes:
            path = subset_path(subset_root, design.sampling_strategy, seed, size)
            relative = subset_key(seed, size)
            if relative not in pinned:
                raise SharedBankDesignError(f"{relative} is not in the committed subset manifest")
            if not path.is_file():
                raise SharedBankDesignError(
                    f"{path} is missing; regenerate the subsets with "
                    "ruleforge reproduce ruleset-size subsets and verify them (--check)"
                )
            digest = file_digest(path)
            if digest != pinned[relative]:
                raise SharedBankDesignError(
                    f"{path} digests to {digest[:12]}..., the manifest pins "
                    f"{pinned[relative][:12]}...: a regenerated subset that differs is a "
                    "different dataset"
                )
            rules = list(iter_jsonl_records(path))
            n_meta = len(rules) - non_meta_rule_count(rules)
            if len(rules) != size or n_meta:
                raise SharedBankDesignError(
                    f"{path} must hold exactly {size} non-meta rules, found {len(rules)} rules "
                    f"({n_meta} meta)"
                )
            digests[(seed, size)] = digest
    return digests


def subset_key(seed: int, size: int) -> str:
    """A subset file's key in its manifest."""
    return f"seed_{int(seed)}/subset_size_{int(size):02d}.jsonl"


def committed_subsets(design: SharedBankDesign) -> dict[str, str]:
    """Return the digests the checkout's subset manifest that ``design`` names pins, found by its checksum file's digest
    and checked as ``design_manifest`` checks it."""
    for path in sorted((repository_root() / SUBSET_MANIFESTS).glob("*/manifest.sha256")):
        if sha256_file(path) == design.subset_manifest_sha256:
            return design_manifest(path.parent, design)
    raise SharedBankDesignError(f"no subset manifest under {SUBSET_MANIFESTS} digests to "
                                f"{design.subset_manifest_sha256[:12]}...; the design's subsets cannot be checked")


@dataclass(frozen=True)
class PlannedCondition:
    """One condition the driver will run: its id, ruleset, shape and identity extras."""

    condition_id: str
    rule_set_path: Path
    evaluation_mode: str
    extra: dict[str, Any]


def plan_conditions(
    design: SharedBankDesign,
    model_name: str,
    *,
    subset_root: Path | str,
    bank_rule_set_path: Path | str,
    seeds: Optional[Sequence[int]] = None,
    include_bank: bool = True,
) -> list[PlannedCondition]:
    """Return one model's conditions, bank first, then seed-major so that a campaign stopped early holds complete
    seeds."""
    chosen = list(design.subset_seeds) if seeds is None else [int(seed) for seed in seeds]
    unknown = [seed for seed in chosen if seed not in design.subset_seeds]
    if unknown or len(set(chosen)) != len(chosen):
        raise SharedBankDesignError(
            f"seeds {chosen} must be distinct members of the design's {list(design.subset_seeds)}"
        )
    bank_path = Path(bank_rule_set_path)
    if sha256_file(bank_path) != design.source_ruleset_sha256:
        raise SharedBankDesignError(
            f"the bank ruleset {bank_path} is not the subsets' source ruleset "
            f"({design.source_ruleset_sha256[:12]}...); bank rule ids and subset rule ids "
            "must be one namespace"
        )
    planned: list[PlannedCondition] = []
    if include_bank:
        planned.append(
            PlannedCondition(
                condition_id=bank_condition_id(model_name),
                rule_set_path=bank_path,
                evaluation_mode=EVALUATION_MODE_BANK,
                extra={"design": design.design, "role": ROLE_BANK},
            )
        )
    for seed in chosen:
        for size in design.subset_sizes:
            planned.append(
                PlannedCondition(
                    condition_id=subset_condition_id(model_name, seed, size),
                    rule_set_path=subset_path(subset_root, design.sampling_strategy, seed, size),
                    evaluation_mode=EVALUATION_MODE_FULL,
                    extra={
                        "design": design.design,
                        "role": ROLE_SUBSET,
                        "subset_seed": int(seed),
                        "subset_size": int(size),
                        "bank_condition_id": bank_condition_id(model_name),
                    },
                )
            )
    return planned


def expected_condition_ids(design: SharedBankDesign, model_name: str) -> set[str]:
    return {bank_condition_id(model_name)} | {
        subset_condition_id(model_name, seed, size)
        for seed in design.subset_seeds
        for size in design.subset_sizes
    }


# Fields in which one model's conditions in a shared-bank root may differ; every other field is the judge configuration
# each inconsistency joins across a subset and the bank.
ROLE_FIELDS = ("condition_id", "evaluation_mode", "rule_set_name", "rule_set_sha256")
ROLE_EXTRAS = ("role", "subset_seed", "subset_size", "bank_condition_id")


def judge_configuration(condition: ConditionSpec) -> dict[str, Any]:
    """Return what every condition of ``condition``'s model shares: its identity less ``ROLE_FIELDS`` and
    ``ROLE_EXTRAS``."""
    found = {field.name: getattr(condition, field.name) for field in dataclasses.fields(condition)
             if field.name not in ROLE_FIELDS}
    found["extra"] = tuple(pair for pair in condition.extra if pair[0] not in ROLE_EXTRAS)
    return found


def configuration_conflicts(conditions: Sequence[ConditionSpec]) -> dict[str, dict[str, str]]:
    """Return ``{field: {value: first condition holding it}}`` for each judge-configuration field in which
    ``conditions`` differ."""
    seen: dict[str, dict[str, str]] = {}
    for condition in conditions:
        for name, value in judge_configuration(condition).items():
            seen.setdefault(name, {}).setdefault(repr(value), condition.condition_id)
    return {name: values for name, values in seen.items() if len(values) > 1}


def refuse_mixed_configuration(model: str, conditions: Sequence[ConditionSpec], *, what: str,
                               error: type[Exception]) -> None:
    """Refuse ``conditions`` unless they hold one judge configuration, naming each field that differs."""
    conflicts = configuration_conflicts(conditions)
    if conflicts:
        detail = "; ".join(f"{name}: " + " vs ".join(f"{value} ({cid})" for value, cid in values.items())
                           for name, values in conflicts.items())
        raise error(f"{model}: {what} hold more than one judge configuration ({detail}). Each inconsistency joins a "
                    "subset condition's judgment to the bank's of the same draw, so a model's conditions in one "
                    "shared-bank root differ in their id, bundle shape, ruleset and role alone; judge another "
                    "configuration into a root of its own")


@functools.lru_cache(maxsize=8192)
def _read_registration(path: str, stamp: tuple[int, int, int, int]) -> ConditionSpec:
    return read_condition(Path(path))


def _registered(path: Path) -> Optional[ConditionSpec]:
    try:
        found = os.stat(path)
    except FileNotFoundError:
        return None
    return _read_registration(str(path), (found.st_dev, found.st_ino, found.st_mtime_ns, found.st_size))


def check_model_configuration(root: Path, *conditions: ConditionSpec) -> None:
    """Refuse registering ``conditions`` into a shared-bank root unless each model holds one judge configuration."""
    by_model: dict[str, list[ConditionSpec]] = {}
    for condition in conditions:
        try:
            model = parse_condition_id(condition.condition_id)[0]
        except SharedBankDesignError:
            continue
        by_model.setdefault(model, []).append(condition)
    if not by_model:
        return
    design = load_design(root)
    for model, members in by_model.items():
        ids = sorted(expected_condition_ids(design, model))
        registered = [found for cid in ids if (found := _registered(root / cid / CONDITION_FNAME)) is not None]
        named = ", ".join(repr(condition.condition_id) for condition in members)
        refuse_mixed_configuration(model, [*members, *registered], error=CampaignRunError,
                                   what=f"condition(s) {named} and those registered in {root}")


def _check_condition(condition: ConditionSpec, design: SharedBankDesign) -> tuple[str, str, Optional[int], Optional[int]]:
    model, role, seed, size = parse_condition_id(condition.condition_id)
    extra = dict(condition.extra)
    expected_mode = EVALUATION_MODE_BANK if role == ROLE_BANK else EVALUATION_MODE_FULL
    if condition.evaluation_mode != expected_mode:
        raise SharedBankDesignError(
            f"{condition.condition_id} is registered as {condition.evaluation_mode!r}, not "
            f"{expected_mode!r}"
        )
    if extra.get("design") != design.design or extra.get("role") != role:
        raise SharedBankDesignError(
            f"{condition.condition_id}: identity extras {extra} do not name design "
            f"{design.design!r} and role {role!r}"
        )
    if role == ROLE_SUBSET and (extra.get("subset_seed"), extra.get("subset_size")) != (seed, size):
        raise SharedBankDesignError(
            f"{condition.condition_id}: identity extras {extra} disagree with the id's seed/size"
        )
    if role == ROLE_BANK and condition.rule_set_sha256 != design.source_ruleset_sha256:
        raise SharedBankDesignError(
            f"{condition.condition_id}: bank ruleset {condition.rule_set_sha256[:12]}... is not "
            f"the design's source ruleset"
        )
    return model, role, seed, size


def _bank_cells(
    cells: Mapping[tuple[str, int], Mapping[str, Any]], condition_id: str
) -> tuple[dict[tuple[str, int], dict[str, str]], tuple[str, ...]]:
    rule_ids: Optional[tuple[str, ...]] = None
    out: dict[tuple[str, int], dict[str, str]] = {}
    for key, row in cells.items():
        judgments = row.get("rule_judgments")
        if not isinstance(judgments, Mapping) or not judgments:
            raise RepeatedSamplingError(f"{condition_id}: {key[0][:12]}/draw {key[1]} carries no rule judgments")
        ids = tuple(judgments)
        if rule_ids is None:
            rule_ids = ids
        elif set(ids) != set(rule_ids):
            raise RepeatedSamplingError(f"{condition_id}: {key[0][:12]}/draw {key[1]} judges a different rule set")
        out[key] = {str(rule_id): str(label) for rule_id, label in judgments.items()}
    return out, rule_ids or ()


def _subset_judgments(
    cells: Mapping[tuple[str, int], Mapping[str, Any]],
    bank: Mapping[tuple[str, int], Mapping[str, str]],
    *,
    condition_id: str,
    size: int,
    scenario_ids: Sequence[str],
    n_draws: int,
) -> Iterator[tuple[int, int, str, list[str]]]:
    """Yield ``(scenario index, draw, full-set label, the bank's labels of the subset's own rules)``, refusing a row
    that cannot be joined to the bank of its draw."""
    rule_ids: Optional[list[str]] = None
    for i, sid in enumerate(scenario_ids):
        for draw in range(n_draws):
            row = cells[(sid, draw)]
            ids = row.get("ruleset_rule_ids")
            label = row.get("full_set_judgment")
            if not isinstance(ids, list) or label is None:
                raise RepeatedSamplingError(
                    f"{condition_id}: {sid[:12]}/draw {draw} is not a full-only outcome"
                )
            row_ids = [str(rule_id) for rule_id in ids]
            if rule_ids is None:
                rule_ids = row_ids
                if len(rule_ids) != size or len(set(rule_ids)) != size:
                    raise RepeatedSamplingError(
                        f"{condition_id}: the subset names {len(rule_ids)} rules, not {size}"
                    )
            elif row_ids != rule_ids:
                raise RepeatedSamplingError(f"{condition_id}: {sid[:12]}/draw {draw} judged a different subset")
            judgments = bank[(sid, draw)]
            try:
                labels = [judgments[rule_id] for rule_id in rule_ids]
            except KeyError as error:
                raise RepeatedSamplingError(
                    f"{condition_id}: the bank of draw {draw} has no judgment for rule {error}"
                ) from None
            yield i, draw, str(label), labels


@dataclass(frozen=True)
class ModelPhi:
    """One model's bank grid digest, rule ids and provenance, its full-set and separate-call maximum phi arrays
    ``(sizes, scenarios, subset_seeds, draws)``, its :func:`max_rule_median` array ``(sizes, scenarios,
    subset_seeds)``, and its subset conditions' draw records."""

    bank_digest: str
    bank_rule_ids: tuple[str, ...]
    bank_provenance: dict[str, Any]
    full: np.ndarray
    separate: np.ndarray
    max_rule_median: np.ndarray
    subset_draws: list[dict[str, Any]]

    @property
    def inconsistency(self) -> np.ndarray:
        """Return 1.0 where a draw's full-set phi is not its separate-call maximum's, else 0.0."""
        return (self.full != self.separate).astype(float)


def max_rule_median(rule_phi: np.ndarray) -> np.ndarray:
    """Return the strongest rule's median phi over draws of a ``(..., draws, rules)`` array, the upper median when
    the draws are even."""
    return np.sort(rule_phi, axis=-2)[..., rule_phi.shape[-2] // 2, :].max(axis=-1)


def model_phi(
    dirs: Mapping[str, Path],
    entry: Mapping[str, Any],
    design: SharedBankDesign,
    *,
    scenario_ids: Sequence[str],
    n_draws: int,
    scenario_manifest_sha256: str,
) -> ModelPhi:
    """Return one model's phi arrays from its conditions, each read from its directory in ``dirs`` and each subset draw
    joined to the bank's draw of the same index, refusing a cell no row fills."""
    load = dict(
        expected_scenario_ids=scenario_ids,
        expected_draws=n_draws,
        scenario_manifest_sha256=scenario_manifest_sha256,
    )
    bank_id = entry["bank"].condition_id
    bank_raw, bank_prov = load_condition_outcomes(dirs[bank_id], **load)
    bank, bank_rule_ids = _bank_cells(bank_raw, bank_id)
    shape = (len(design.subset_sizes), len(scenario_ids), len(design.subset_seeds), n_draws)
    full, separate = np.full(shape, np.nan), np.full(shape, np.nan)
    rule_median = np.full(shape[:3], np.nan)
    subset_draws: list[dict[str, Any]] = []
    for j, subset_seed in enumerate(design.subset_seeds):
        for k, size in enumerate(design.subset_sizes):
            condition_id = entry["subsets"][(subset_seed, size)].condition_id
            cells, prov = load_condition_outcomes(dirs[condition_id], **load)
            subset_draws.extend(prov["draws"])
            rule_phi = np.full((len(scenario_ids), n_draws, size), np.nan)
            for i, draw, label, labels in _subset_judgments(cells, bank, condition_id=condition_id, size=size,
                                                            scenario_ids=scenario_ids, n_draws=n_draws):
                full[k, i, j, draw] = PHI_MAP[label]
                separate[k, i, j, draw] = max_rule_phi(labels)
                rule_phi[i, draw] = [PHI_MAP[value] for value in labels]
            # An unfilled cell would compare unequal to itself and so count as inconsistent; refuse it instead.
            missing = np.argwhere(~(np.isfinite(full[k, :, j]) & np.isfinite(separate[k, :, j])))
            if missing.size:
                i, draw = missing[0]
                raise RepeatedSamplingError(f"{condition_id}: {scenario_ids[i][:12]}/draw {draw} is missing")
            rule_median[k, :, j] = max_rule_median(rule_phi)
    bank_digest = canonical_digest([[sid, draw, bank[(sid, draw)]] for sid in scenario_ids for draw in range(n_draws)])
    return ModelPhi(bank_digest, bank_rule_ids, bank_prov, full, separate, rule_median, subset_draws)


def subset_grid_digest(values: np.ndarray) -> str:
    """Return the digest of a model's ``(sizes, scenarios, subset_seeds, draws)`` inconsistency grid."""
    return canonical_digest(values.tolist())


def default_contrasts(sizes: Sequence[int]) -> list[dict[str, Any]]:
    """Return the standard contrasts (net rise, upper-vs-middle, last step) where the design carries their sizes."""
    sizes = sorted(int(size) for size in sizes)
    if len(sizes) < 2:
        return []
    contrasts = [{"name": "net_rise", "minuend": [sizes[-1]], "subtrahend": [sizes[0]]}]
    if UPPER_SIZE in sizes and all(size in sizes for size in MIDDLE_SIZES):
        contrasts.append({"name": "upper_vs_middle", "minuend": [UPPER_SIZE], "subtrahend": list(MIDDLE_SIZES)})
    contrasts.append({"name": "last_step", "minuend": [sizes[-1]], "subtrahend": [sizes[-2]]})
    return contrasts


def group_by_model(
    registered: Mapping[str, ConditionSpec], design: SharedBankDesign
) -> dict[str, dict[str, Any]]:
    """Return ``{model: {"bank": spec, "subsets": {(seed, size): spec}}}``, refusing ids off the design or mixed judge
    configurations."""
    by_model: dict[str, dict[str, Any]] = {}
    for condition in registered.values():
        model, role, subset_seed, size = _check_condition(condition, design)
        entry = by_model.setdefault(model, {"bank": None, "subsets": {}})
        if role == ROLE_BANK:
            entry["bank"] = condition
        else:
            entry["subsets"][(subset_seed, size)] = condition
    for model, entry in by_model.items():
        present = {cid for cid in registered if cid.startswith(model + "/")}
        expected = expected_condition_ids(design, model)
        if present != expected:
            raise SharedBankDesignError(
                f"{model}: {len(expected - present)} design condition(s) missing and "
                f"{len(present - expected)} extra; the design is {len(expected)} conditions per model"
            )
        refuse_mixed_configuration(model, [entry["bank"], *entry["subsets"].values()], error=SharedBankDesignError,
                                   what="its conditions in this root")
    return by_model


def one_ruleset_sizes(entry: Mapping[str, Any], design: SharedBankDesign) -> list[int]:
    """Return the sizes at which every subset seed of one model judged the same ruleset."""
    return [size for size in design.subset_sizes
            if len({entry["subsets"][(seed, size)].rule_set_sha256 for seed in design.subset_seeds}) == 1]


@dataclass(frozen=True)
class RootConditions:
    """What a shared-bank analysis reads: its design, its conditions by id, the directory each is read from and the
    sizes a bank root supplies (none for one root)."""

    design: SharedBankDesign
    registered: dict[str, ConditionSpec]
    dirs: dict[str, Path]
    bank_sizes: tuple[int, ...] = ()


def read_conditions(root: Path, bank_root: Optional[Path] = None) -> RootConditions:
    """Return ``root``'s design and conditions; with ``bank_root``, banks and the sizes ``root``'s design lacks come
    from that root, refused unless both designs share seeds, source ruleset and subset files."""
    design, registered = load_design(root), load_conditions(root)
    dirs = {condition_id: root / condition_id for condition_id in registered}
    if bank_root is None:
        return RootConditions(design, registered, dirs)
    bank_design = load_design(bank_root)
    differing = [name for name in ("design", "subset_seeds", "source_ruleset_sha256")
                 if getattr(bank_design, name) != getattr(design, name)]
    if differing:
        raise SharedBankDesignError(f"{bank_root}'s design differs from {root}'s in {', '.join(differing)}; a bank "
                                    "root joins only a design over the same seeds and ruleset")
    parsed = {condition_id: parse_condition_id(condition_id) for condition_id in registered}
    own_bank = sorted(condition_id for condition_id, (_, role, _, _) in parsed.items() if role == ROLE_BANK)
    if own_bank:
        raise SharedBankDesignError(f"{root} registers a bank of its own ({own_bank[0]}); with a bank root, the bank "
                                    "comes from that root alone")
    outside = sorted(condition_id for condition_id, (_, role, _, size) in parsed.items()
                     if role == ROLE_SUBSET and size not in design.subset_sizes)
    if outside:
        raise SharedBankDesignError(f"{root} registers {outside[0]}, a size its own design does not name; with a bank "
                                    "root, every size the design lacks comes from that root alone")
    models = {model for model, _, _, _ in parsed.values()}
    supplied = tuple(size for size in bank_design.subset_sizes if size not in design.subset_sizes)
    taken: dict[str, ConditionSpec] = {}
    for condition_id, condition in load_conditions(bank_root).items():
        model, role, _, size = parse_condition_id(condition_id)
        if model in models and (role == ROLE_BANK or size in supplied):
            taken[condition_id] = condition
    joined = dataclasses.replace(design, subset_sizes=(*design.subset_sizes, *supplied))
    dirs = {**{condition_id: bank_root / condition_id for condition_id in taken}, **dirs}
    # Each subset, from either root, must be the joined design's file for its seed and size.
    pinned = committed_subsets(joined)
    for condition_id, condition in {**taken, **registered}.items():
        _, role, seed, size = parse_condition_id(condition_id)
        if role == ROLE_SUBSET and pinned.get(subset_key(seed, size)) != condition.rule_set_sha256:
            raise SharedBankDesignError(f"{dirs[condition_id]} judged a ruleset that is not the "
                                        f"{joined.sampling_strategy} design's subset {subset_key(seed, size)}")
    return RootConditions(joined, {**taken, **registered}, dirs, supplied)


def bank_root_record(bank_root: Path, manifest: Mapping[str, Any], found: RootConditions) -> dict[str, Any]:
    """Return what an analysis records of the bank root it joined, refused unless both campaigns agree on every sampling
    setting in force after their amendments (the retry cap too; only the amendment records may differ) and every
    condition taken from it is complete."""
    bank_manifest = load_sampling_manifest(bank_root / SAMPLING_MANIFEST_FNAME)
    for side in (bank_manifest, manifest):
        try:
            cap_amendment(side)
        except AmendmentError as error:
            kinds = [record.get("kind") if isinstance(record, Mapping) else record
                     for record in side.get(MANIFEST_AMENDMENTS_KEY) or ()]
            raise CampaignRunError(f"campaign {side['campaign_id']!r} records the amendment(s) {kinds} ({error}); two "
                                   "campaigns join only where an amendment raised the retry cap and nothing else") from error
    # A manifest holds each setting as in force after its amendments; only the amendment records may differ.
    differing = [name for name in differing_manifest_fields(bank_manifest, manifest)
                 if name not in ("campaign_id", MANIFEST_AMENDMENTS_KEY)]
    if differing:
        detail = "; ".join(f"{name} {bank_manifest.get(name)!r} vs {manifest.get(name)!r}" for name in differing)
        raise CampaignRunError(f"the bank root {bank_root} ran other sampling settings than campaign "
                               f"{manifest['campaign_id']!r} ({detail}); its banks join only the same settings")
    gate = campaign_completion(bank_root, write=False)
    taken = sorted(condition_id for condition_id, place in found.dirs.items() if place == bank_root / condition_id)
    blocking = {condition_id: incomplete_states(gate["conditions"][condition_id]["draws"]) for condition_id in taken}
    blocking = {condition_id: states for condition_id, states in blocking.items() if states}
    if blocking:
        raise CampaignRunError(f"the bank root {bank_root} is not complete for the conditions this analysis takes "
                               f"from it; nothing is aggregated: {json.dumps(blocking)}")
    amendment = amendment_disclosure(bank_root, bank_manifest)
    return {
        "campaign_id": bank_manifest["campaign_id"],
        "root": artifact_uri_or_path(bank_root),
        "subset_sizes": list(found.bank_sizes),
        "max_semantic_attempts": bank_manifest["max_semantic_attempts"],
        **({MANIFEST_AMENDMENTS_KEY: [amendment]} if amendment is not None else {}),
    }


def model_stacks(
    dirs: Mapping[str, Path], entry: Mapping[str, Any], design: SharedBankDesign, *, args_by_digest: EngineArgsIndex
) -> dict[str, list[Signature]]:
    """Return ``{condition_id: its judgments' stacks}`` for one model, bank first; a provider judge's are none."""
    members = [entry["bank"]] + [
        entry["subsets"][(seed, size)] for seed in design.subset_seeds for size in design.subset_sizes
    ]
    return {
        condition.condition_id: (
            condition_signatures(dirs[condition.condition_id], args_by_digest=args_by_digest)
            if condition.model_type == LOCAL_MODEL_TYPE
            else []
        )
        for condition in members
    }


def _model_estimates(
    dirs: Mapping[str, Path],
    entry: Mapping[str, Any],
    *,
    stacks: Mapping[str, Sequence[Signature]],
    allow_stack_mismatch: bool,
    design: SharedBankDesign,
    scenario_ids: Sequence[str],
    n_draws: int,
    scenario_manifest_sha256: str,
    contrasts: Sequence[Mapping[str, Any]],
    n_resamples: int,
    ci_level: float,
    seed: int,
    joined: bool,
) -> dict[str, Any]:
    bank_condition = entry["bank"]
    phi = model_phi(dirs, entry, design, scenario_ids=scenario_ids, n_draws=n_draws,
                    scenario_manifest_sha256=scenario_manifest_sha256)
    values = phi.inconsistency
    curve = bank_shared_curve_bootstrap(
        values, sizes=design.subset_sizes, contrasts=contrasts,
        n_resamples=n_resamples, ci_level=ci_level, seed=seed,
    )
    distinct: list[Signature] = []
    for signatures in stacks.values():
        for signature in signatures:
            if signature not in distinct:
                distinct.append(signature)
    return {
        "model_name": bank_condition.model_name,
        "judge_type": bank_condition.judge_type,
        **stacks_record(allow_stack_mismatch, distinct),
        "bank": {
            "condition_id": bank_condition.condition_id,
            "condition_fingerprint": phi.bank_provenance["condition_fingerprint"],
            "n_rules": len(phi.bank_rule_ids),
            "grid_digest": phi.bank_digest,
            "attempt_statistics": attempt_statistics(phi.bank_provenance["draws"]),
        },
        "subsets": {
            "n_conditions": len(entry["subsets"]),
            "grid_digest": subset_grid_digest(values),
            "attempt_statistics": attempt_statistics(phi.subset_draws),
        },
        # Only a joined analysis records them; the builders read one root's from its conditions.
        **({ONE_RULESET_SIZES_KEY: one_ruleset_sizes(entry, design)} if joined else {}),
        "points": curve["points"],
        "contrasts": curve["contrasts"],
        "adjacent_rises": int(
            sum(
                later["estimate"] > earlier["estimate"]
                for earlier, later in zip(curve["points"], curve["points"][1:])
            )
        ),
    }


def analyze_shared_bank_campaign(
    campaign_root: Path | str,
    *,
    n_resamples: Optional[int] = None,
    ci_level: Optional[float] = None,
    seed: Optional[int] = None,
    allow_stack_mismatch: bool = False,
    bank_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compute and write a complete campaign's shared-bank estimates under ``aggregate/``."""
    root = Path(campaign_root)
    payload = shared_bank_estimates(root, n_resamples=n_resamples, ci_level=ci_level, seed=seed,
                                allow_stack_mismatch=allow_stack_mismatch, write_gate=True, bank_root=bank_root)
    destination = write_json_sidecar(root / AGGREGATE_DIRNAME / SHARED_BANK_ESTIMATES_FNAME, payload)
    logger.info("Wrote %s", destination)
    return payload


def shared_bank_estimates(
    campaign_root: Path | str,
    *,
    n_resamples: Optional[int] = None,
    ci_level: Optional[float] = None,
    seed: Optional[int] = None,
    allow_stack_mismatch: bool = False,
    write_gate: bool = False,
    bank_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return the estimates for a complete campaign under the strict gate; a stack mismatch is refused unless
    ``allow_stack_mismatch``. With ``bank_root``, each model's bank and missing sizes come from that root
    (``read_conditions``), which must agree on every sampling setting in force after amendments
    (``bank_root_record``)."""
    root = Path(campaign_root)
    for place in (root, *(() if bank_root is None else (Path(bank_root),))):
        refuse_published(place, "bank analyze", "its manifests and each draw's completion record in full")
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    if manifest.get("campaign_mode") == CAMPAIGN_MODE_SCREENING:
        raise CampaignRunError(f"campaign {manifest['campaign_id']!r} is a screening campaign")
    scenario_manifest = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME)
    # Identities first: a wrong registration is a design error, not an incomplete run.
    banks = None if bank_root is None else Path(os.path.realpath(Path(bank_root).expanduser()))
    found = read_conditions(root, banks)
    design = found.design
    by_model = group_by_model(found.registered, design)
    gate = require_complete_campaign(root, write=write_gate)
    # A joined analysis names both roots, so its estimates are read the same wherever they are written.
    joined = {} if banks is None else {"root": artifact_uri_or_path(root),
                                       BANK_ROOT_KEY: bank_root_record(banks, manifest, found)}
    args_by_digest: EngineArgsIndex = {}
    stacks = {
        model: model_stacks(found.dirs, entry, design, args_by_digest=args_by_digest)
        for model, entry in by_model.items()
    }
    mismatches = [
        record
        for model, members in stacks.items()
        for record in shared_stack_conflicts(members, scope=model, args_by_digest=args_by_digest)
    ]
    if mismatches and not allow_stack_mismatch:
        raise CampaignRunError(
            f"campaign {manifest['campaign_id']!r}: a model's bank and subset conditions do not share one "
            f"inference stack ({describe_mismatches(mismatches)}); every inconsistency joins a subset's "
            "judgment to the bank's of the same draw. Re-run them on one stack, or pass "
            "allow_stack_mismatch to report the curve marked as such"
        )
    bootstrap = manifest["bootstrap"]
    amendment = amendment_disclosure(root, manifest)
    settings = dict(
        design=design,
        scenario_ids=scenario_manifest.scenario_ids,
        n_draws=int(manifest["expected_decoder_draws"]),
        scenario_manifest_sha256=scenario_manifest.manifest_sha256,
        contrasts=default_contrasts(design.subset_sizes),
        n_resamples=int(bootstrap["n_resamples"] if n_resamples is None else n_resamples),
        ci_level=float(bootstrap["ci_level"] if ci_level is None else ci_level),
        seed=int(bootstrap["seed"] if seed is None else seed),
    )
    payload = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "state": gate["state"],
        **override_record(allow_stack_mismatch, mismatches),
        "design": design.to_dict(),
        **joined,
        "inference_method": CURVE_INFERENCE_METHOD,
        "interval_basis": CURVE_INTERVAL_BASIS,
        "n_scenarios": len(settings["scenario_ids"]),
        "n_decoder_draws": settings["n_draws"],
        "n_subset_seeds": len(design.subset_seeds),
        "subset_sizes": list(design.subset_sizes),
        "temperature": manifest["temperature"],
        "max_semantic_attempts": manifest["max_semantic_attempts"],
        **({MANIFEST_AMENDMENTS_KEY: [amendment]} if amendment is not None else {}),
        "truncation": manifest.get("truncation"),
        "label_rule": manifest.get("label_rule"),
        **retry_scope_record(manifest),
        "scenario_manifest_sha256": scenario_manifest.manifest_sha256,
        "bootstrap": {key: settings[key] for key in ("n_resamples", "ci_level", "seed")},
        "contrasts": settings["contrasts"],
        "models": {
            model: _model_estimates(
                found.dirs, entry, stacks=stacks[model], allow_stack_mismatch=allow_stack_mismatch,
                joined=banks is not None, **settings
            )
            for model, entry in by_model.items()
        },
        "created_at": utc_now(),
    }
    return payload


def freeze_design(campaign: Campaign, design: SharedBankDesign) -> Path:
    """Freeze ``design`` into ``campaign``'s root as its design file, or find it there; another design is refused."""
    with campaign._writing():
        _destination(campaign.root, SHARED_BANK_DESIGN_FNAME)
        return write_design(campaign.root, design)


def analyze(root: Path | str, *, n_resamples: Optional[int] = None, ci_level: Optional[float] = None,
            seed: Optional[int] = None, allow_stack_mismatch: bool = False, replace: bool = False,
            bank_root: Path | str | None = None) -> dict[str, Any]:
    """Write a shared-bank root's analysis (this library's root; new, or under ``replace``) and return it;
    ``bank_root``, which supplies banks and missing sizes, is only read."""
    location = Path(os.path.realpath(Path(root).expanduser()))
    campaign = Campaign.open(location)
    target = location / AGGREGATE_DIRNAME / SHARED_BANK_ESTIMATES_FNAME
    with _owner_writing(location, campaign.campaign_id, campaign.scenario_manifest, campaign.config):
        if target.exists() and not replace:
            raise AnalysisExists(f"{target} holds this root's analysis; a new analysis replaces it only when asked "
                                "(replace=True, --replace)")
        for relative in (COMPLETION_FNAME, Path(AGGREGATE_DIRNAME) / SHARED_BANK_ESTIMATES_FNAME):
            _destination(location, relative)
        _refuse_links(location / AGGREGATE_DIRNAME, "the analysis's directory")
        return analyze_shared_bank_campaign(location, n_resamples=n_resamples, ci_level=ci_level, seed=seed,
                                            allow_stack_mismatch=allow_stack_mismatch, bank_root=bank_root)


def analyze_into(root: Path | str, out: Path | str, *, n_resamples: Optional[int] = None,
                 ci_level: Optional[float] = None, seed: Optional[int] = None, allow_stack_mismatch: bool = False,
                 replace: bool = False, bank_root: Path | str | None = None) -> dict[str, Any]:
    """Write any shared-bank root's analysis to ``out``, outside it, and return it; neither the root nor
    ``bank_root`` is written."""
    location = Path(os.path.realpath(Path(root).expanduser()))
    Campaign.open(location)
    target = Path(out)
    refuse_protected(target, "the analysis's file")
    if target.exists() and not replace:
        raise AnalysisExists(f"{target} holds an analysis; a new analysis replaces it only when asked (replace=True, "
                             "--replace)")
    payload = shared_bank_estimates(location, n_resamples=n_resamples, ci_level=ci_level, seed=seed,
                                    allow_stack_mismatch=allow_stack_mismatch, bank_root=bank_root)
    logger.info("Wrote %s", write_json_sidecar(target, payload))
    return payload


_SIZE_IN_NAME = re.compile(r"subset_size_(\d+)\.jsonl$")


def design_block(cfg: Any) -> Any:
    """Return the config's ruleset-size design block."""
    return getattr(cfg, DESIGN_CONFIG_KEY)


def design_from_cfg(cfg: Any) -> SharedBankDesign:
    """Return the design the config declares; sizes default to every size the manifest pins."""
    block = design_block(cfg)
    manifest_dir = Path(str(block.subset_manifest_dir))
    meta, pinned = load_manifest(manifest_dir)
    if block.get("subset_sizes") is None:
        matches = (_SIZE_IN_NAME.search(name) for name in pinned)
        sizes = sorted({int(match.group(1)) for match in matches if match})
    else:
        sizes = [int(size) for size in block.subset_sizes]
    return SharedBankDesign(
        design=str(block.design),
        subset_seeds=tuple(int(seed) for seed in block.subset_seeds),
        subset_sizes=tuple(sizes),
        sampling_strategy=str(block.sampling_strategy),
        subset_manifest_sha256=str(block.approved_subset_manifest_sha256),
        source_ruleset_sha256=str(meta["source_ruleset"]["sha256"]),
    )


def job_seeds(cfg: Any, design: SharedBankDesign) -> list[int]:
    """Return this job's share of the design's seeds."""
    chosen = design_block(cfg).get("jobs_seeds")
    return list(design.subset_seeds) if chosen is None else [int(seed) for seed in chosen]


def runs_bank(cfg: Any, design: SharedBankDesign, seeds: Sequence[int]) -> bool:
    """Whether this job generates the bank: ``auto`` gives it to the share holding the first seed."""
    setting = design_block(cfg).get("run_bank", "auto")
    if isinstance(setting, bool):
        return setting
    if str(setting).lower() != "auto":
        raise SharedBankDesignError(f"{DESIGN_CONFIG_KEY}.run_bank must be auto, true or false, got {setting!r}")
    return min(design.subset_seeds) in seeds


def condition_cfg(cfg: Any, planned: PlannedCondition) -> Any:
    """The base config with one condition's id, ruleset, shape and extras."""
    from omegaconf import OmegaConf

    copy = cfg.copy()
    OmegaConf.set_struct(copy, False)
    copy.condition_id = planned.condition_id
    copy.rule_set_path = str(planned.rule_set_path)
    copy.evaluation_mode = planned.evaluation_mode
    copy.condition_extra = dict(planned.extra)
    return copy


def condition_complete(
    campaign_root: Path,
    condition: ConditionSpec,
    n_draws: int,
    *,
    scenario_manifest_sha256: str,
    campaign_id: str,
) -> bool:
    """Return whether every declared draw of ``condition`` is gated complete for this job's population, campaign and
    fingerprint, refusing a root that registers the id under another identity."""
    from ruleforge.evaluate.campaign import check_registered_condition
    from ruleforge.evaluate.identity import draw_dirname

    if not check_registered_condition(campaign_root, condition).is_file():
        return False
    for draw in range(n_draws):
        path = campaign_root / condition.condition_id / draw_dirname(draw) / COMPLETION_FNAME
        if not path.exists():
            return False
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("state") != STATE_COMPLETE or not record.get("materialized"):
            return False
        if record.get("scenario_manifest_sha256") != scenario_manifest_sha256:
            return False
        if record.get("campaign_id") != campaign_id:
            return False
        if record.get("condition_fingerprint") != condition.fingerprint:
            return False
    return True


def job_conditions(cfg: Any, planned: Sequence[PlannedCondition], *, backend: Any,
                   task_spec: Any) -> dict[str, ConditionSpec]:
    """Return each planned condition's identity by id; nothing is written."""
    from omegaconf import OmegaConf

    from ruleforge.evaluate.campaign import campaign_rule_levels, config_condition
    from ruleforge.evaluate.identity import SamplingConfig

    config = SamplingConfig.from_mapping(OmegaConf.to_container(cfg.sampling, resolve=True))
    found = {}
    for item in planned:
        ccfg = condition_cfg(cfg, item)
        found[item.condition_id] = config_condition(ccfg, backend=backend, task_spec=task_spec, config=config,
                                                    levels=campaign_rule_levels(ccfg))
    return found


def incomplete_conditions(cfg: Any, planned: Sequence[PlannedCondition], *, identity: Mapping[str, str],
                          backend: Any, task_spec: Any) -> tuple[list[PlannedCondition], dict[str, ConditionSpec]]:
    """Return the planned conditions not complete for this job and each one's identity, refusing mismatches before
    judging."""
    root = Path(str(cfg.campaign_root))
    n_draws = int(cfg.sampling.num_decoder_draws)
    conditions = job_conditions(cfg, planned, backend=backend, task_spec=task_spec)
    todo = [item for item in planned if not condition_complete(root, conditions[item.condition_id], n_draws, **identity)]
    return todo, conditions


def _register(cfg: Any, item: PlannedCondition, condition: ConditionSpec, *, backend: Any, task_spec: Any,
              writing: Callable[[], ContextManager[Any]]) -> Any:
    from ruleforge.evaluate.campaign import build_plan, check_registered_condition
    from ruleforge.evaluate.runner import check_run_write

    root = Path(str(cfg.campaign_root))
    with writing():
        check_run_write(root / item.condition_id)
        check_registered_condition(root, condition)
        return build_plan(condition_cfg(cfg, item), backend=backend, task_spec=task_spec)


def job_draws(cfg: Any, n_draws: int) -> list[int]:
    """Return the decoder draws this job runs, checked at preflight."""
    chosen = cfg.get("decoder_draw_ids")
    if chosen is None:
        return list(range(n_draws))
    draws = [int(draw) for draw in chosen]
    if not draws or len(set(draws)) != len(draws) or any(not 0 <= draw < n_draws for draw in draws):
        raise SharedBankDesignError(f"decoder_draw_ids={draws} must be distinct draw ids below {n_draws}")
    return draws


def plan_job(cfg: Any) -> tuple[SharedBankDesign, list[PlannedCondition], dict[str, Any]]:
    """Validate everything a job needs before a judge is loaded, and describe it."""
    from ruleforge.evaluate.campaign import execution_settings, expected_requests_per_draw, preflight

    if execution_settings(cfg).sharded:
        raise SharedBankDesignError(
            "execution.num_shards shards one condition (a configured campaign's job, `ruleforge jobs launch campaign "
            "--shards`); a shared-bank job "
            "runs many conditions and writes its campaign root's design and gate itself, so it is never sharded"
        )
    config, manifest = preflight(cfg)
    draws = job_draws(cfg, config.num_decoder_draws)
    design = design_from_cfg(cfg)
    block = design_block(cfg)
    verify_subsets(str(block.subset_root), str(block.subset_manifest_dir), design)
    seeds = job_seeds(cfg, design)
    include_bank = runs_bank(cfg, design, seeds)
    planned = plan_conditions(
        design,
        str(cfg.judge_model.model_name),
        subset_root=str(block.subset_root),
        bank_rule_set_path=Path(str(cfg.rule_set_path)),
        seeds=seeds,
        include_bank=include_bank,
    )
    rules = load_jsonl_records(Path(str(cfg.rule_set_path)))
    # A subset condition's count is N whatever its ruleset.
    requests_per_draw = sum(
        expected_requests_per_draw(manifest, rules, item.evaluation_mode) for item in planned
    )
    report = {
        "campaign_id": str(cfg.campaign_id),
        "scenario_manifest_sha256": manifest.manifest_sha256,
        "judge_model": str(cfg.judge_model.model_name),
        "design": design.to_dict(),
        "job_seeds": seeds,
        "runs_bank": include_bank,
        "conditions_this_job": len(planned),
        "conditions_per_model": design.n_conditions_per_model,
        "scenarios": manifest.record_count,
        "num_decoder_draws": config.num_decoder_draws,
        "decoder_draw_ids": draws,
        "requests_per_draw_this_job": requests_per_draw,
        "requests_total_this_job": requests_per_draw * len(draws),
    }
    return design, planned, report


def incomplete_states(completions: Mapping[int, Mapping[str, Any]]) -> dict[int, str]:
    """The draws of one condition that did not gate complete, with their states."""
    return {draw: record["state"] for draw, record in completions.items() if record["state"] != STATE_COMPLETE}


def job_identity(cfg: Any, report: Mapping[str, Any]) -> dict[str, str]:
    """What an on-disk draw gate must name to let this job skip its condition."""
    return {"scenario_manifest_sha256": report["scenario_manifest_sha256"], "campaign_id": str(cfg.campaign_id)}


def run_conditions(
    cfg: Any,
    planned: Sequence[PlannedCondition],
    *,
    identity: Mapping[str, str],
    backend: Any,
    task_spec: Any,
    writing: Callable[[], ContextManager[Any]] = nullcontext,
) -> dict[str, dict[int, str]]:
    """Run the planned conditions not complete, one after another; return the incomplete ones with their draw states."""
    from ruleforge.evaluate.campaign import execution_settings, run_plan

    todo, conditions = incomplete_conditions(cfg, planned, identity=identity, backend=backend, task_spec=task_spec)
    incomplete: dict[str, dict[int, str]] = {}
    ran, skipped = 0, len(planned) - len(todo)
    for item in todo:
        plan = _register(cfg, item, conditions[item.condition_id], backend=backend, task_spec=task_spec,
                         writing=writing)
        completions = run_plan(plan, backend=backend, task_spec=task_spec, execution=execution_settings(cfg),
                               gate=False)
        ran += 1
        bad = incomplete_states(completions)
        if bad:
            # An exhausted draw voids this arm; the job goes on with the other conditions.
            incomplete[item.condition_id] = bad
        logger.info(
            "%d of %d conditions done this job (%d skipped as complete)",
            ran + skipped, len(planned), skipped,
        )
    return incomplete


def run_packed_conditions(
    cfg: Any,
    planned: Sequence[PlannedCondition],
    *,
    identity: Mapping[str, str],
    backend: Any,
    task_spec: Any,
    writing: Callable[[], ContextManager[Any]] = nullcontext,
) -> dict[str, dict[int, str]]:
    """Plan every incomplete condition, then run all their draws packed; return the incomplete ones."""
    from dataclasses import replace

    from ruleforge.evaluate.campaign import execution_settings, run_plans_packed

    todo, conditions = incomplete_conditions(cfg, planned, identity=identity, backend=backend, task_spec=task_spec)
    logger.info("Packing the draws of %d condition(s) (%d skipped as complete)", len(todo), len(planned) - len(todo))
    rows: dict[str, tuple[dict[str, Any], ...]] = {}
    plans = []
    for item in todo:
        plan = _register(cfg, item, conditions[item.condition_id], backend=backend, task_spec=task_spec,
                         writing=writing)
        shared = rows.setdefault(plan.scenario_manifest.manifest_sha256, plan.scenario_rows)
        plans.append(replace(plan, scenario_rows=shared))
    completions = run_plans_packed(plans, backend=backend, task_spec=task_spec, execution=execution_settings(cfg))
    return {cid: bad for cid, done in completions.items() if (bad := incomplete_states(done))}


def publish_gate(root: Path, writing: Callable[[], ContextManager[Any]] = nullcontext) -> dict[str, Any]:
    """Compute ``root``'s campaign gate from disk and write it, under ``writing``."""
    from ruleforge.evaluate.campaign import campaign_completion
    from ruleforge.evaluate.runner import check_run_write

    with writing():
        check_run_write(root)
        return campaign_completion(root)


def run_planned(cfg: Any, design: SharedBankDesign, planned: Sequence[PlannedCondition],
                report: Mapping[str, Any], *,
                writing: Callable[[], ContextManager[Any]] = nullcontext) -> dict[str, dict[int, str]]:
    """Judge a planned job into its campaign root with one judge load and return the conditions left incomplete with
    their draw states; shared files are written under ``writing``, the gate once on the way out."""
    from ruleforge.evaluate.campaign import checked_at_entry
    from ruleforge.evaluate.runner import check_run_write

    root = Path(str(cfg.campaign_root))
    check_run_write(root)
    with checked_at_entry(root):  # the job's entry: its many registrations and runs rest on this check
        return _run_planned(cfg, root, design, planned, report, writing=writing)


def _run_planned(cfg: Any, root: Path, design: SharedBankDesign, planned: Sequence[PlannedCondition],
                 report: Mapping[str, Any], *, writing: Callable[[], ContextManager[Any]]) -> dict[str, dict[int, str]]:
    from ruleforge.evaluate.campaign import execution_settings
    from ruleforge.tasks.spec import build_task_spec
    from ruleforge.judges.backend import ModelBackend

    root.mkdir(parents=True, exist_ok=True)
    with writing():
        write_design(root, design)
    run = run_packed_conditions if execution_settings(cfg).pack_draws else run_conditions
    task_spec = build_task_spec(cfg)
    backend = ModelBackend(cfg.judge_model)
    began: list[bool] = []

    def registering() -> ContextManager[Any]:
        began.append(True)
        return writing()

    try:
        try:
            incomplete = run(cfg, planned, identity=job_identity(cfg, report), backend=backend, task_spec=task_spec,
                             writing=registering)
        finally:
            backend.close()
    except BaseException:
        if began:
            try:
                publish_gate(root, writing)
            except Exception as error:
                logger.warning("The campaign gate of %s was not published after the job failed: %s", root, error)
        raise
    gate = publish_gate(root, writing)
    logger.info("Campaign %s gate: %s (%d conditions registered)", gate["campaign_id"], gate["state"],
                gate["n_conditions"])
    return incomplete


def _job_write_guard(campaign: Campaign, given: Path) -> Any:
    spellings = dict.fromkeys((campaign.root, Path(os.path.abspath(given))))

    def guard(directory: Path) -> None:
        campaign._owned()
        relative = next((Path(directory).relative_to(base) for base in spellings
                         if Path(directory).is_relative_to(base)), None)
        if relative is None:
            raise CampaignError(f"{directory} is outside the campaign root {campaign.root}, which is the one place "
                                "a job of it writes")
        for name in (COMPLETION_FNAME, SCENARIO_MANIFEST_FNAME, SAMPLING_MANIFEST_FNAME, SHARED_BANK_DESIGN_FNAME,
                     SNAPSHOT_DIRNAME):
            _destination(campaign.root, name)
        _refuse_links(campaign.root / SNAPSHOT_DIRNAME, "the code snapshots' directory")
        if relative.parts:
            _destination(campaign.root, relative)
            _refuse_links(campaign.root / relative, "the directory a job writes")

    return guard


def run_job(campaign: Campaign, cfg: Any, design: SharedBankDesign, planned: Sequence[PlannedCondition],
            report: Mapping[str, Any]) -> dict[str, dict[int, str]]:
    """Run ``run_planned`` into ``campaign``'s root under the campaign's guards and write lock."""
    from ruleforge.evaluate.runner import guard_run_writes, hold_shared_writes
    from ruleforge.judges.batch_api import BatchDestinationRefused, guard_work_directories

    given = Path(str(cfg.campaign_root))
    if Path(os.path.realpath(given)) != campaign.root:
        raise CampaignError(f"the job's campaign_root {given} is not the campaign's root {campaign.root}")
    batch_root = cfg.judge_model.get("batch_root")
    if batch_root:
        _refuse_foreign(campaign.root, str(batch_root), "the batch judge's batch_root")
    freeze_design(campaign, design)
    try:
        with guard_work_directories(_work_directory_guard(campaign)), \
                guard_run_writes(_job_write_guard(campaign, given)), hold_shared_writes(campaign._writing):
            return run_planned(cfg, design, planned, report, writing=campaign._writing)
    except BatchDestinationRefused as error:
        raise CampaignError(f"campaign {campaign.campaign_id!r}: {error}") from error


def incomplete_error(incomplete: Mapping[str, Mapping[int, str]]) -> str:
    """What a job that left conditions incomplete says."""
    return (f"{len(incomplete)} condition(s) did not complete: {json.dumps(incomplete)}. "
            "Nothing is aggregated until they do.")


__all__ = ["SharedBankDesign", "PlannedCondition", "analyze", "freeze_design", "plan_conditions"]
