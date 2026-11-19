from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ruleforge.core.record_names import SAME_CALL_RATE_KEY
from ruleforge.evaluate.identity import SCENARIO_EXCLUSION_KEY, VALIDATION_SCHEMA_VERSION, ConditionSpec
from ruleforge.evaluate.stack import ALLOW_STACK_MISMATCH_KEY, STACK_MISMATCHES_KEY

# Attempt statistics count every manifest scenario, including those a fallback dropped from the rate.
ATTEMPT_POPULATION = "every manifest scenario"
# Each group is ``{family: {"members": [...], "contrasts": [row, ...]}}``.
CONTRAST_GROUPS = ("style_families", "ruleset_families", "policy_families")


@dataclass(frozen=True)
class Estimate:
    """One statistic and its interval, with its replicates' ``sd`` and, where reported, the two-stage interval."""

    value: float
    ci: tuple[float, float]
    sd: float
    basis: str
    two_stage: Optional[tuple[float, float]] = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "Estimate":
        """Build one from a bootstrap summary as an analysis writes one."""
        two_stage = row.get("two_stage")
        return cls(value=row["estimate"], ci=(row["ci_low"], row["ci_high"]), sd=row["sd"],
                   basis=row["interval_basis"],
                   two_stage=None if two_stage is None else (two_stage["ci_low"], two_stage["ci_high"]))


@dataclass(frozen=True)
class Kappa:
    """A condition's stratified Cohen's kappa between full-set judgments and per-rule maxima; ``value``, ``ci`` and
    ``sd`` are None where it is undefined (chance agreement 1)."""

    value: Optional[float]
    ci: Optional[tuple[float, float]]
    sd: Optional[float]
    p_observed: float
    p_chance: float
    n_undefined_replicates: int
    strata: Mapping[str, Any]

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> "Kappa":
        """From ``ruleforge.evaluate.discrimination.kappa_summary``'s output."""
        defined = summary["ci_low"] is not None
        return cls(value=summary["estimate"], ci=(summary["ci_low"], summary["ci_high"]) if defined else None,
                   sd=summary["sd"], p_observed=summary["p_observed"], p_chance=summary["p_chance"],
                   n_undefined_replicates=summary["n_undefined_replicates"], strata=summary["strata"])

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "ci": None if self.ci is None else list(self.ci), "sd": self.sd,
                "p_observed": self.p_observed, "p_chance": self.p_chance,
                "n_undefined_replicates": self.n_undefined_replicates, "strata": dict(self.strata)}


@dataclass(frozen=True)
class ValidityPolicy:
    """What counted as a valid answer, read from the condition's identity: the rate is conditional on it."""

    label_rule: str
    forced_json: str
    retry_scope: str
    strict_think: bool
    max_attempts: int
    validation_schema_version: int = VALIDATION_SCHEMA_VERSION


@dataclass(frozen=True)
class Contrast:
    """One paired contrast row: ``rate(a) - rate(b)``, its interval, the randomization ``p_value`` and ``p_holm``."""

    family: str
    group: str
    a: str
    b: str
    estimate: Estimate
    p_value: Optional[float]
    p_holm: Optional[float]
    family_size: Optional[int]
    stack_mismatch: Optional[str] = None

    @classmethod
    def from_row(cls, group: str, row: Mapping[str, Any]) -> "Contrast":
        return cls(family=str(row.get("family", "")), group=group, a=row["a"], b=row["b"],
                   estimate=Estimate.from_row(row), p_value=row.get("p_value"), p_holm=row.get("p_holm"),
                   family_size=row.get("family_size"), stack_mismatch=row.get("stack_mismatch"))


@dataclass(frozen=True)
class Estimates:
    """One analysed ``both`` condition: its entry of ``aggregate/estimates.json``, as written. Every rate is over
    ``n_scenarios`` x ``draws`` accepted draws, conditional on ``validity``."""

    n_scenarios: int
    draws: int
    rate: Estimate
    variance: Mapping[str, Any]
    validity: ValidityPolicy
    excluded_scenario: Optional[tuple[str, str]]
    attempts: Mapping[str, Any]
    attempt_population: str
    stack_mismatches: tuple[Mapping[str, Any], ...]
    overrides: Mapping[str, Any]
    same_call: Optional[Estimate]
    per_rule: tuple[Mapping[str, Any], ...]
    estimates_path: Path
    estimates_sha256: str
    created_at: str


@dataclass(frozen=True)
class EvalResult:
    """A collected condition: its draws' states, attempt statistics, outcome files and any estimates."""

    root: Path
    campaign_id: str
    condition_id: str
    fingerprint: str
    mode: str
    draws: Mapping[int, str]
    attempts: Mapping[str, Any]
    outcomes: tuple[Path, ...]
    estimates: Optional[Estimates] = None
    excluded: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CampaignResult:
    """A campaign root's recorded analysis: the file (or None), every condition, the contrasts and the disclosures."""

    path: Optional[Path]
    payload: Mapping[str, Any]
    conditions: Mapping[str, EvalResult]
    contrasts: tuple[Contrast, ...]
    excluded_conditions: Mapping[str, Any]
    scenario_exclusion: Optional[Mapping[str, Any]]
    stack_mismatches: tuple[Mapping[str, Any], ...]


def estimates_of(condition: ConditionSpec, payload: Mapping[str, Any], *, path: Path, sha256: str) -> Estimates:
    """The :class:`Estimates` of ``condition`` from an estimates file's ``payload`` (its entry must be there)."""
    cid = condition.condition_id
    entry = payload["conditions"][cid]
    exclusion = payload.get(SCENARIO_EXCLUSION_KEY)
    excluded = None
    if exclusion is not None and cid in exclusion.get("arms", {}):
        excluded = (str(exclusion["judge"]), str(exclusion["scenario_id"]))
    forced = (payload.get("forced_conditions") or {}).get(cid)
    overrides = {key: payload[key] for key in (ALLOW_STACK_MISMATCH_KEY, "amendments") if key in payload}
    mismatches = tuple(dict(record) for record in payload.get(STACK_MISMATCHES_KEY) or ()
                       if cid in (record.get("conditions") or ()))
    return Estimates(
        n_scenarios=int(entry["n_scenarios"]),
        draws=int(entry["n_decoder_draws"]),
        rate=Estimate.from_row(entry),
        variance=dict(entry.get("variance_components") or {}),
        validity=ValidityPolicy(
            label_rule=condition.label_rule, forced_json=condition.forced_json_requirement,
            retry_scope=condition.retry_scope,
            strict_think=condition.strict_think, max_attempts=int(payload["max_semantic_attempts"])),
        excluded_scenario=excluded,
        attempts=dict(entry.get("attempt_statistics") or {}),
        attempt_population=ATTEMPT_POPULATION,
        stack_mismatches=mismatches,
        overrides=overrides,
        same_call=None if forced is None else Estimate.from_row(forced[SAME_CALL_RATE_KEY]),
        per_rule=() if forced is None else tuple(dict(row) for row in forced["rule_context"]),
        estimates_path=path,
        estimates_sha256=sha256,
        created_at=str(payload["created_at"]),
    )


def contrasts_of(payload: Mapping[str, Any]) -> tuple[Contrast, ...]:
    """Every paired contrast row an estimates file records, family by family."""
    return tuple(Contrast.from_row(group, row)
                 for key in CONTRAST_GROUPS
                 for group, entry in (payload.get(key) or {}).items()
                 for row in entry.get("contrasts") or ())


__all__ = [
    "ATTEMPT_POPULATION", "CampaignResult", "Contrast", "Estimate", "Estimates", "EvalResult", "Kappa",
    "ValidityPolicy", "contrasts_of", "estimates_of",
]
