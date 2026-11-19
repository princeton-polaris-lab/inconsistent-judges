from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ruleforge.core.errors import RuleForgeError
from ruleforge.experiments.goldens import read_json_golden
from ruleforge.experiments import table
from ruleforge.experiments.table import Golden

#: Sibling key suffixes that spell an interval's two bounds.
_BOUND_SUFFIXES = (("_low", "_high"), ("_lo", "_hi"))

#: Estimate keys tried when the bounds key carries no stem of its own.
_BARE_ESTIMATE_KEYS = ("mean", "estimate", "value", "change", "difference", "delta")

#: Booleans that only say a value lies near a threshold; a flip is noise, and the drift count still reports it.
_NOISE_FLAGS = frozenset({"holm_mc_margin"})

# Every leaf of a golden is compared strictly (a different, missing or added one fails the check) but for two groups.
#: Statistical results, reported as drift: these keys, a "Rule N" count, an interval list, everything under attempt
#: statistics, and the claims of ``RESULT_TEXT`` when their builder derives them.
RESULT_KEYS = frozenset({
    "baseline_rate", "between_scenario", "both_consistent", "both_inconsistent", "chance_agreement", "change_rate",
    "ci_high", "ci_low", "decoder_share_of_variance", "delta", "delta_contribution", "difference", "estimate",
    "first_attempt_acceptance_share", "fixed", "icc", "infinite_draws", "invalid_attempt_rate", "kappa",
    "max_change_rate", "mean", "min_change_rate", "p", "p_bh", "p_chance", "p_holm", "p_observed", "p_randomization",
    "p_value", "phi_alone", "phi_delta", "phi_in_context", "pooled_delta", "pooled_phi_alone", "pooled_phi_delta",
    "pooled_phi_in_context", "pooled_rate_alone", "pooled_rate_in_context", "rate", "rate_a", "rate_alone", "rate_b",
    "rate_in_context", "rate_member", "rate_reference", "regressed", "rho", "sd", "se_analytic",
    "share_of_gain_explained", "share_of_pooled_delta", "share_of_responses", "share_of_under_flags", "treatment_rate",
    "unchanged_fraction", "within_draw", "yes_rate_member", "yes_rate_reference", "naive", "encouraged", "forced",
    "1", "2", "5", "10", "20",
    "significant", "significant_bh", "significant_holm", "holm_mc_margin", "improvement", "excludes_zero", "degenerate",
    "few_informative", "flagged", "p_value_is_floor", "passed", "mismatched", "zero_seed_variance",
    "adjacent_rises", "changed", "unchanged", "yes", "no", "unsure", "count", "n_holm_higher", "n_holm_lower",
    "n_informative_scenarios", "n_responses", "n_rules_higher", "n_rules_lower", "n_rules_tied",
    "n_undefined_replicates", "n_complete_maps", "pooled_yes_difference", "yes_difference",
})
RESULT_SUBTREES = ("attempt_statistics", "incomplete_reasons")
RESULT_TEXT = frozenset({"claim"})
#: Run environment, reported in one notice and never fatal: a leaf, never a list entry, whose own key is named here or
#: whose place ``ENVIRONMENT_PLACES`` spells. A design or identity value (a retry cap, a condition, a level named root)
#: and a stack description (``STACK_DESCRIPTIONS``) keep their group.
ENVIRONMENT_KEYS = frozenset({
    "created_at", "recorded_at", "decided_on", "first_request_block", "last_request_block",
    "baseline_root", "treatment_root", "campaign_root", "uri",
    "campaign_id", "parent_campaign_id",
    "sha256", "base_weights_sha256", "estimates_sha256", "merge_manifest_sha256", "model_weights_sha256",
    "parent_sampling_manifest_sha256", "parent_scenario_manifest_sha256", "report_sha256", "rule_levels_sha256",
    "rule_set_sha256", "sampling_manifest_sha256", "scenario_manifest_sha256", "scenario_source_sha256",
    "source_ruleset_sha256", "source_sha256", "subset_manifest_sha256", "condition_fingerprint", "grid_digest",
    "grid_digest_a", "grid_digest_b", "grid_digest_alone", "grid_digest_in_context",
    "code_revision", "code_dirty", "code_state_sha256", "code_state_described",
    "vllm_version",
})
_KEYS = r"[^\[\]]+"  # one key or more under a named parent, a dotted model name included
#: Keys that are the run environment only at these places.
ENVIRONMENT_PLACES = (
    r"(?:campaigns|arms)\[\d+\]\.root",
    r"bootstrap\.root",
    r"judges\[\d+\]\.(?:flat|hierarchy)\.root",
    rf"(?:provenance\.arms|diagnostics\.conditions)\.{_KEYS}\.root",
    r"(?:campaign|campaigns\[\d+\])\.estimates",
    r"campaign\.(?:root|sizes_from\.\d+\.root)",
    rf"parity_b\.{_KEYS}\.(?:decode|prefill)\.report",
    rf"base_weights\.members\.{_KEYS}\.merge_manifest",
    r"grid_digests\.(?:bank|subsets)",
)
#: Where an inference stack is described: compared strictly, but for the versions ``unversioned`` masks, which alone
#: moving is the run environment.
STACK_DESCRIPTIONS = (
    r"stack\.(?:description|short)",
    r"engine\.(?:short|stack)",
    rf"provenance\.arms\.{_KEYS}\.stack",
    r"cross_stack\.(?:baseline_stack|treatment_stack|conflict)",
)
#: Every name a version follows in the committed goldens' stack descriptions, as the stack records spell it.
VERSIONED_NAMES = ("vllm", "vLLM", "torch", "transformers", "tokenizers", "triton")
VERSION_MASK = "<version>"


def _place(patterns: Sequence[str]) -> re.Pattern[str]:
    return re.compile(r"(?:^|[.\]])(?:" + "|".join(patterns) + r")$")


_ENVIRONMENT_PLACE, _STACK_DESCRIPTION = _place(ENVIRONMENT_PLACES), _place(STACK_DESCRIPTIONS)
_VERSION = re.compile(r"(?<![\w.])(" + "|".join(VERSIONED_NAMES) + r") [^\s,]+")


def unversioned(text: str) -> str:
    """``text`` with the version after each of ``VERSIONED_NAMES`` replaced by ``VERSION_MASK``: what a stack
    description says besides the versions it ran on."""
    return _VERSION.sub(rf"\1 {VERSION_MASK}", text)


STRICT, RESULT, SOURCE = "strict", "result", "environment"


def _under(path: str, names: Sequence[str]) -> bool:
    parts = set(re.split(r"[.\[\]]+", path))
    return any(name in parts for name in names)


def _source(key: str, path: str = "") -> bool:
    return not path.endswith("]") and (key in ENVIRONMENT_KEYS or _ENVIRONMENT_PLACE.search(path) is not None)


def role(key: str, value: Any, path: str = "") -> str:
    """Return how a leaf under ``key`` at ``path`` is compared: ``environment`` (one notice), ``result`` (reported) or
    ``strict``; by its name and place, never by its type."""
    if _source(key, path):
        return SOURCE
    if (key in RESULT_KEYS or key.startswith("Rule ") or key == "ci" or key.endswith("_ci")
            or _under(path, RESULT_SUBTREES)):
        return RESULT
    return RESULT if key in RESULT_TEXT and isinstance(value, str) else STRICT


class RerunError(RuntimeError, RuleForgeError):
    """The rerun could not be checked at all: an unreadable file or an unknown golden name."""


@dataclass(frozen=True)
class Finding:
    """One thing the rerun did or did not preserve."""

    golden: str
    check: str
    path: str
    detail: str
    fatal: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return f"{self.golden}: {self.check} at {self.path or '.'}: {self.detail}"


@dataclass(frozen=True)
class Interval:
    """An estimate and the interval a golden commits it to; `key` identifies the interval inside its node."""

    key: str
    estimate_key: str | None
    estimate: float | None
    low: float
    high: float
    keys: tuple[str, ...]
    sd: float | None = None

    @property
    def name(self) -> str:
        return self.estimate_key or self.key

    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def overlaps(self, other: "Interval") -> bool:
        return self.low <= other.high and other.low <= self.high

    def miss_chance(self) -> float:
        """Return the chance that an independent estimate of the same size falls outside this interval (their difference
        has sd * sqrt(2)); asked only of an interval with an estimate and a standard deviation."""
        if self.sd <= 0:
            return 0.0
        spread = 2 * self.sd  # sqrt(2) for the difference, sqrt(2) inside erf
        return 0.5 * (1 + math.erf((self.low - self.estimate) / spread)) + 0.5 * (
            1 - math.erf((self.high - self.estimate) / spread))


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bounds(value: Any) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) == 2 and all(_numeric(v) for v in value):
        return float(value[0]), float(value[1])
    return None


def _estimate_for(
    node: Mapping[str, Any], stem: str
) -> tuple[str | None, float | None]:
    """Return the sibling key holding the point estimate for a bounds key, trying the longest stem first."""
    candidates: List[str] = []
    while stem:
        candidates += [stem, f"{stem}_mean"]
        stem = stem.rpartition("_")[0]
    candidates += list(_BARE_ESTIMATE_KEYS)
    for key in candidates:
        if _numeric(node.get(key)):
            return key, float(node[key])
    return None, None


def _sd_for(node: Mapping[str, Any], stem: str) -> float | None:
    for key in (f"{stem}_sd", "sd"):
        if _numeric(node.get(key)):
            return float(node[key])
    return None


def intervals(node: Mapping[str, Any]) -> List[Interval]:
    """Return every estimate-plus-interval this node commits to: a `<stem>_ci` (or `ci`) pair, or sibling bound keys
    such as `ci_low`/`ci_high`."""
    found: List[Interval] = []
    for key in sorted(node):
        if key != "ci" and not key.endswith("_ci"):
            continue
        bounds = _bounds(node[key])
        if bounds is None:
            continue
        stem = key[: -len("_ci")] if key != "ci" else ""
        estimate_key, estimate = _estimate_for(node, stem)
        keys = (key,) if estimate_key is None else (key, estimate_key)
        found.append(
            Interval(key, estimate_key, estimate, bounds[0], bounds[1], keys, _sd_for(node, stem))
        )

    for key in sorted(node):
        for low_suffix, high_suffix in _BOUND_SUFFIXES:
            if not key.endswith(low_suffix):
                continue
            high_key = key[: -len(low_suffix)] + high_suffix
            if not (_numeric(node.get(key)) and _numeric(node.get(high_key))):
                continue
            stem = key[: -len(low_suffix)]
            estimate_key, estimate = _estimate_for(node, stem)
            keys = (key, high_key)
            if estimate_key is not None:
                keys += (estimate_key,)
            found.append(
                Interval(
                    key,
                    estimate_key,
                    estimate,
                    float(node[key]),
                    float(node[high_key]),
                    keys,
                    _sd_for(node, stem.removesuffix("_ci")),
                )
            )
    return found


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _interval_findings(
    golden: str,
    path: str,
    committed: Interval,
    rerun: Interval,
    counts: Dict[str, int],
) -> List[Finding]:
    where = _join(path, committed.name)
    findings: List[Finding] = []

    if committed.estimate is not None and rerun.estimate is None:
        # A rerun without point estimates fails: containment is the strongest check.
        findings.append(
            Finding(
                golden,
                "missing",
                where,
                "the committed golden records a point estimate here and the "
                "rerun does not, so the containment check cannot run",
                fatal=True,
            )
        )

    if rerun.estimate is not None and committed.estimate is not None:
        counts["containment"] += 1
        if committed.sd is None:
            counts["without_sd"] += 1
        else:
            counts["expected_misses"] += committed.miss_chance()
        if not committed.contains(rerun.estimate):
            counts["missed"] += 1
            findings.append(
                Finding(
                    golden,
                    "containment",
                    where,
                    f"rerun estimate {rerun.estimate:.6g} is outside the "
                    f"committed interval [{committed.low:.6g}, "
                    f"{committed.high:.6g}]",
                    fatal=False,
                )
            )
        # A direction counts only when the committed interval excludes zero.
        if committed.excludes_zero():
            counts["sign"] += 1
            if committed.estimate * rerun.estimate <= 0.0:
                findings.append(
                    Finding(
                        golden,
                        "sign",
                        where,
                        f"committed {committed.estimate:.6g} (interval "
                        f"excludes zero) but rerun {rerun.estimate:.6g}",
                        fatal=False,
                    )
                )

    counts["overlap"] += 1
    if not committed.overlaps(rerun):
        findings.append(
            Finding(
                golden,
                "overlap",
                where,
                f"committed [{committed.low:.6g}, {committed.high:.6g}] and "
                f"rerun [{rerun.low:.6g}, {rerun.high:.6g}] do not overlap",
                fatal=False,
            )
        )
    return findings


def _node_findings(
    golden: str,
    path: str,
    committed: Mapping[str, Any],
    rerun: Mapping[str, Any],
    tolerance: float,
    counts: Dict[str, int],
) -> List[Finding]:
    findings: List[Finding] = []
    committed_intervals = intervals(committed)
    rerun_intervals = {interval.key: interval for interval in intervals(rerun)}
    consumed = {key for interval in committed_intervals for key in interval.keys}

    for interval in committed_intervals:
        counterpart = rerun_intervals.get(interval.key)
        if counterpart is None:
            findings.append(
                Finding(
                    golden,
                    "missing",
                    _join(path, interval.name),
                    "the committed golden carries an interval here and the "
                    "rerun does not, so the claim cannot be checked",
                    fatal=True,
                )
            )
            continue
        findings += _interval_findings(golden, path, interval, counterpart, counts)

    for key in sorted(set(committed) & set(rerun) - consumed - _NOISE_FLAGS):
        if key in RESULT_TEXT and isinstance(committed[key], str):
            findings += _derived_findings(golden, _join(path, key), key, committed[key], rerun)
        elif not isinstance(committed[key], (Mapping, list)):
            findings += _scalar_findings(golden, _join(path, key), key, committed[key], rerun[key], tolerance, counts)
    return findings


def _derived(node: Mapping[str, Any]) -> Any:
    from ruleforge.experiments.training.campaign_readout import claim

    try:
        return claim(node["flat"], node["kappa"])
    except (KeyError, TypeError):
        return None


def _derived_findings(golden: str, where: str, key: str, before: str, rerun: Mapping[str, Any]) -> List[Finding]:
    """Text derived from results is drift only when it is what the builder derives from the rerun's own values."""
    after = rerun[key]
    if after != _derived(rerun):
        return [Finding(golden, "design", where, f"rerun {after!r} is not what the builder derives from the rerun's "
                        "own values", fatal=True)]
    return [] if after == before else [Finding(golden, "derived", where, f"committed {before!r}, rerun {after!r}: "
                                               "derived from results that moved", fatal=False)]


def _scalar_findings(golden: str, where: str, key: str, before: Any, after: Any, tolerance: float,
                     counts: Dict[str, Any]) -> List[Finding]:
    kind = role(key, before, where)
    if kind == SOURCE:
        counts["environment"] += before != after
        return []
    # Text derived from results is checked against its builder where its row stands (``_derived_findings``).
    if kind == STRICT or isinstance(before, str):
        counts["design"] += 1
        same = before == after and isinstance(after, bool) == isinstance(before, bool) and type(after) in (
            type(before), *((int, float) if _numeric(before) else ()))
        if not same and isinstance(before, str) and isinstance(after, str) and _STACK_DESCRIPTION.search(where) \
                and unversioned(before) == unversioned(after):
            # Only the package versions a stack description names moved: where the run happened.
            counts["environment"] += 1
            return []
        return [] if same else [Finding(golden, "design", where, f"committed {before!r}, rerun {after!r}: the rerun "
                                        f"compared something else ({key})", fatal=True)]
    if before is None:
        # A result with nothing recorded (an undefined estimate): reported when the rerun records one.
        return [] if after is None else [Finding(golden, "result", where, f"committed None, rerun {after!r}",
                                                 fatal=False)]
    return _leaf_findings(golden, where, before, after, tolerance, counts)


def _leaf_findings(
    golden: str,
    where: str,
    before: Any,
    after: Any,
    tolerance: float,
    counts: Dict[str, int],
) -> List[Finding]:
    """The checks one committed number or boolean supports: lost, flipped or changed in direction."""
    if (_numeric(before) and not _numeric(after)) or (isinstance(before, bool) and not isinstance(after, bool)):
        # Null, a string, a container or no finite number where a claim stood: a lost claim, not numeric drift.
        return [
            Finding(
                golden,
                "missing",
                where,
                f"committed {before!r}, rerun {after!r}: the rerun holds no "
                f"{'boolean' if isinstance(before, bool) else 'finite number'} "
                "here, so the claim cannot be checked",
                fatal=True,
            )
        ]
    if isinstance(before, bool):
        # A committed boolean is a claim: a flip is fatal.
        counts["flag"] += 1
        if before != after:
            return [
                Finding(
                    golden,
                    "flag",
                    where,
                    f"committed {before}, rerun {after}: a boolean claim "
                    "flipped",
                    fatal=False,
                )
            ]
        return []
    counts["bare_sign"] += 1
    # Both sides must clear the tolerance: a null contrast may wobble across zero.
    if before * after < 0.0 and min(abs(before), abs(after)) > tolerance:
        return [
            Finding(
                golden,
                "sign",
                where,
                f"committed {before:.6g}, rerun {after:.6g}: both exceed "
                f"the tolerance {tolerance:g}, so this is a "
                "direction change rather than noise",
                fatal=False,
            )
        ]
    return []


def _carries_checkable(node: Any, path: str = "", key: str = "") -> bool:
    """Return whether the walk would compare a leaf under this node at ``path`` whose role is not the run environment;
    a container is judged by its leaves, never by its own key."""
    if isinstance(node, Mapping):
        return any(_carries_checkable(value, _join(path, name), name) for name, value in node.items())
    if isinstance(node, list):
        return any(_carries_checkable(value, f"{path}[{index}]", key) for index, value in enumerate(node))
    return role(key, node, path) != SOURCE


def _walk_pair(
    golden: str,
    path: str,
    committed: Any,
    rerun: Any,
    tolerance: float,
    counts: Dict[str, int],
    out: List[Finding],
    key: str = "",
) -> None:
    """Recurse the two structures in lockstep, checking each mapping node and each list entry by the key it sits under;
    a length mismatch on a list that carries anything checked is fatal."""
    if isinstance(committed, Mapping) and isinstance(rerun, Mapping):
        out += _node_findings(golden, path, committed, rerun, tolerance, counts)
        # A dropped key that carries claims is fatal. Keys only the rerun carries are `compare`'s business, and keys in
        # this node's own intervals are reported by `_node_findings`.
        consumed = {
            key for interval in intervals(committed) for key in interval.keys
        }
        for key in sorted(set(committed) - set(rerun) - consumed - _NOISE_FLAGS):
            where = _join(path, key)
            if not _carries_checkable(committed[key], where, key):
                counts["environment"] += 1
            elif _under(where, RESULT_SUBTREES):
                # A new sample may hold other failure reasons or attempt counts: reported, never fatal.
                out.append(Finding(golden, "result", where, "the rerun records no count here", fatal=False))
            else:
                out.append(
                    Finding(
                        golden,
                        "missing",
                        _join(path, key),
                        "the committed golden carries checkable claims under "
                        "this key and the rerun does not carry the key at "
                        "all, so every claim check under it never ran",
                        fatal=True,
                    )
                )
        out += [Finding(golden, "extra", _join(path, key), "the rerun carries a key the committed golden does not: a "
                        "judge, condition, size or field it did not compare",
                        fatal=not _under(_join(path, key), RESULT_SUBTREES))
                for key in sorted(set(rerun) - set(committed) - _NOISE_FLAGS)
                if _carries_checkable(rerun[key], _join(path, key), key)]
        # An interval's own keys were checked above; where a run happened is not compared.
        for name in sorted(set(committed) & set(rerun) - consumed):
            if isinstance(committed[name], (Mapping, list)):
                _walk_pair(golden, _join(path, name), committed[name], rerun[name], tolerance, counts, out, name)
        return
    if isinstance(committed, list) and isinstance(rerun, list):
        if len(committed) == len(rerun):
            for index, (before, after) in enumerate(zip(committed, rerun)):
                where = f"{path}[{index}]"
                if isinstance(before, (Mapping, list)):
                    _walk_pair(golden, where, before, after, tolerance, counts, out, key)
                else:
                    out += _scalar_findings(golden, where, key, before, after, tolerance, counts)
        elif _carries_checkable(committed, path, key) or _carries_checkable(rerun, path, key):
            out.append(
                Finding(
                    golden,
                    "missing",
                    f"{path}[]",
                    f"the committed list has {len(committed)} entries and the "
                    f"rerun {len(rerun)}, so the claim checks under it never "
                    "ran; a curve or table that gained or lost a row is a "
                    "structural change, not numeric drift",
                    fatal=True,
                )
            )
        return
    if isinstance(committed, (Mapping, list)) and _carries_checkable(committed, path, key):
        # A claim-bearing subtree the rerun holds as null, a scalar or the
        # other container: every check under it would silently never run.
        out.append(
            Finding(
                golden,
                "missing",
                path,
                "the committed golden carries checkable claims under this "
                f"{'mapping' if isinstance(committed, Mapping) else 'list'} "
                f"and the rerun holds {type(rerun).__name__} here, so every "
                "claim check under it never ran",
                fatal=True,
            )
        )


def claim_findings(
    golden: str,
    committed: Any,
    rerun: Any,
    tolerance: float,
) -> tuple[List[Finding], Dict[str, int]]:
    """Every containment, overlap and sign check the two structures support."""
    counts: Dict[str, Any] = {"containment": 0, "missed": 0, "expected_misses": 0.0, "without_sd": 0, "overlap": 0,
                              "sign": 0, "bare_sign": 0, "flag": 0, "design": 0, "environment": 0}
    findings: List[Finding] = []
    _walk_pair(golden, "", committed, rerun, tolerance, counts, findings)
    return findings, counts


def _leaves_equal(before: Any, after: Any, tolerance: float) -> bool:
    if isinstance(before, bool) or isinstance(after, bool):
        return before is after
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if isinstance(before, int) and isinstance(after, int):
            return before == after
        if before == after:
            return True
        # Two NaNs are equal; a NaN against a number is not.
        if before != before or after != after:
            return before != before and after != after
        return abs(float(before) - float(after)) <= tolerance
    return before == after


def _walk(before: Any, after: Any, tolerance: float, path: str, out: Dict[str, Any]) -> None:
    def key(suffix: Any) -> str:
        return f"{path}{suffix}" if path else str(suffix)

    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for name in sorted(set(before) | set(after)):
            child = key(f".{name}" if path else name)
            if name not in before:
                out[child] = {"was": None, "now": after[name], "kind": "added"}
            elif name not in after:
                out[child] = {"was": before[name], "now": None, "kind": "removed"}
            else:
                _walk(before[name], after[name], tolerance, child, out)
        return
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            out[path or "."] = {"was": len(before), "now": len(after), "kind": "length"}
            return
        for index, (b, a) in enumerate(zip(before, after)):
            _walk(b, a, tolerance, key(f"[{index}]"), out)
        return
    if not _leaves_equal(before, after, tolerance):
        out[path or "."] = {"was": before, "now": after, "kind": "changed"}


def differences(before: Any, after: Any, tolerance: float = 0.0) -> Dict[str, Any]:
    """Return what moved between two goldens, keyed by path; numbers within ``tolerance`` match."""
    found: Dict[str, Any] = {}
    _walk(before, after, tolerance, "", found)
    return found


def read_golden(path: Path) -> Any:
    """Read a golden from disk."""
    return read_json_golden(Path(path), "golden", error=RerunError)


def verify_golden(golden: Golden, committed: Any, rerun: Any) -> Dict[str, Any]:
    """Check one golden's claims against one rerun of it."""
    tolerance = golden.rerun_tolerance
    findings: List[Finding] = []
    moved = differences(committed, rerun, tolerance)
    if moved:
        paths = sorted(moved)
        findings.append(
            Finding(
                golden.name, "drift", "",
                f"{len(paths)} value(s) differ beyond tolerance {tolerance:g} (e.g. {', '.join(paths[:3])})",
                fatal=False,
            )
        )
    claims, counts = claim_findings(golden.name, committed, rerun, tolerance)
    findings += claims
    if counts["environment"]:
        findings.append(Finding(golden.name, "environment", "", f"{counts['environment']} value(s) of where, when or "
                                "on what the run happened differ", fatal=False))
    numeric = [(abs(float(change["now"]) - float(change["was"])), path, change) for path, change in moved.items()
               if change["kind"] == "changed" and _numeric(change["was"]) and _numeric(change["now"])]
    return {
        "name": golden.name,
        "path": golden.path,
        "status": "checked",
        "tolerance": tolerance,
        "differences": len(moved),
        "difference_paths": sorted(moved)[:5],
        "largest_differences": [{"path": path, "was": change["was"], "now": change["now"]}
                                for _, path, change in sorted(numeric, key=lambda item: -item[0])[:3]],
        "checks_run": counts,
        "findings": [finding.as_dict() for finding in findings],
    }


def select(goldens: Sequence[Golden], only: Sequence[str] = ()) -> List[Golden]:
    """Return the goldens to check, in the given order; an unknown ``only`` name raises."""
    if not only:
        return list(goldens)
    known = {golden.name for golden in goldens}
    unknown = sorted(set(only) - known)
    if unknown:
        raise RerunError(f"no golden named {', '.join(unknown)}; known names are {', '.join(sorted(known))}")
    return [golden for golden in goldens if golden.name in set(only)]


def extra_files(goldens: Sequence[Golden], rerun_root: Path) -> List[Finding]:
    """A file of the rerun tree that is no golden, where the selected goldens live: all of ``results/`` when every
    golden is selected, else their own directories."""
    known = {golden.path for golden in (*table.goldens(), *goldens)}
    every = {golden.path for golden in table.goldens()} <= {golden.path for golden in goldens}
    places = {Path(rerun_root) / "results"} if every else {(Path(rerun_root) / golden.path).parent for golden in goldens}
    found = sorted({path.relative_to(rerun_root).as_posix() for place in places if place.is_dir()
                    for path in place.rglob("*") if path.is_file()} - known)
    return [Finding("results", "extra", path, "the rerun tree holds a file no golden is: a result this table does not "
                    "have", fatal=True) for path in found]


def verify_rerun(goldens: Sequence[Golden], repo_root: Path, rerun_root: Path) -> Dict[str, Any]:
    """Judge a rerun tree against every golden in ``goldens``."""
    reports: List[Dict[str, Any]] = []
    for golden in goldens:
        committed_path, rerun_path = Path(repo_root) / golden.path, Path(rerun_root) / golden.path
        if not committed_path.is_file():
            raise RerunError(f"{golden.path} is in the experiment table but not committed at {committed_path}")
        if not rerun_path.is_file():
            missing = Finding(golden.name, "missing", golden.path,
                              f"the rerun tree has no {golden.path}, so its claims were not checked", fatal=True)
            reports.append({"name": golden.name, "path": golden.path, "status": "skipped",
                            "reason": f"no rerun artifact at {rerun_path}", "findings": [missing.as_dict()]})
            continue
        reports.append(verify_golden(golden, read_golden(committed_path), read_golden(rerun_path)))

    findings = [finding for report in reports for finding in report["findings"]]
    findings += [finding.as_dict() for finding in extra_files(goldens, rerun_root)]
    fatal = [finding for finding in findings if finding["fatal"]]
    return {
        "tool": "verify_rerun",
        "goldens": reports,
        "fatal": fatal,
        "advisory": [finding for finding in findings if not finding["fatal"]],
        "ok": not fatal,
    }
