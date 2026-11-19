from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ruleforge.evaluate.inference import (
    DEFAULT_PERMUTATIONS,
    INFERENCE_METHOD,
    DrawGrid,
    clustered_paired_randomization_p,
    hierarchical_bootstrap,
    holm_adjust,
    load_condition_outcomes,
    outcome_grid,
    read_draw_outcomes,
    rule_grids,
)
from ruleforge.evaluate.amendment import disclosure as amendment_disclosure
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    campaign_completion,
    load_conditions,
    require_complete_campaign,
)
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_SCREENING,
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    SHARED_BANK_DESIGN_FNAME,
    FORCED_JSON_COMPLETE_RULE_MAP,
    MANIFEST_AMENDMENTS_KEY,
    SCENARIO_EXCLUSION_KEY,
    ConditionSpec,
    draw_dirname,
    load_sampling_manifest,
    refuse_published,
    retry_scope_record,
)
from ruleforge.scenarios.manifest import (
    SAMPLING_SCHEMA_VERSION,
    ScenarioManifest,
)
from ruleforge.core.files import write_json_sidecar
from ruleforge.core.record_names import SAME_CALL_KEY, SAME_CALL_RATE_KEY, SEPARATE_CALL_KEY, SEPARATE_VS_SAME_CALL_KEY
from ruleforge.evaluate.discrimination import (
    KAPPA_ORIENTATION,
    KAPPA_TEST_METHOD,
    VerdictGrids,
    arm_statistics,
    cross_arm_decomposition,
    kappa_contrast,
    load_rules,
    same_call_statistics,
    read_strata,
    reclassify_forced_maps,
    stratified_differences,
)
from ruleforge.evaluate.ledger import DrawLedger
from ruleforge.evaluate.validation import ComponentSchema
from ruleforge.evaluate.runner import (
    STATE_COMPLETE,
    STATE_INCOMPLETE_EXHAUSTED,
    CampaignRunError,
    bundle_outcome,
    draw_ended,
    utc_now,
)
from ruleforge.evaluate.sharding import campaign_lock
from ruleforge.evaluate.stack import (
    LOCAL_MODEL_TYPE,
    EngineArgsIndex,
    Signature,
    condition_signatures,
    contrast_conflict,
    describe_mismatches,
    mismatch_record,
    mixed_conditions,
    override_record,
    stacks_record,
)
from ruleforge.prompts.styles import VALID_REASONING_PROMPT_STYLES
from ruleforge.tasks.base import FORCED_JSON

logger = logging.getLogger(__name__)

AGGREGATE_DIRNAME = "aggregate"
ESTIMATES_FNAME = "estimates.json"
# The order a style family lists its members, and so each contrast's ``a`` and ``b``.
_STYLE_ORDER = {style: index for index, style in enumerate(VALID_REASONING_PROMPT_STYLES)}


def contrast_row(
    grids: Mapping[str, DrawGrid],
    a: str,
    b: str,
    *,
    shared: bool,
    n_resamples: int,
    ci_level: float,
    seed: int,
    n_permutations: int,
) -> dict[str, Any]:
    """Return one paired contrast ``rate(a) - rate(b)``: interval, decomposition, p."""
    boot = hierarchical_bootstrap(
        {a: grids[a], b: grids[b]},
        contrasts=[(a, b)],
        shared_inner=[[a, b]] if shared else [],
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
    )
    test = clustered_paired_randomization_p(
        grids[a], grids[b], n_permutations=n_permutations, seed=seed
    )
    row = dict(boot["contrasts"][0])
    row.update(
        {
            "rate_a": grids[a].rate,
            "rate_b": grids[b].rate,
            "n_scenarios": grids[a].n_scenarios,
            "n_decoder_draws_a": grids[a].n_draws,
            "n_decoder_draws_b": grids[b].n_draws,
            "p_value": test["p_value"],
            "n_permutations": test["n_permutations"],
            "inference_method": INFERENCE_METHOD,
            "grid_digest_a": grids[a].digest,
            "grid_digest_b": grids[b].digest,
            "bootstrap_seed": seed,
        }
    )
    return row


def _all_pairs(members: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]


def _stack_conflicts(
    pairs: Mapping[str, Sequence[tuple[str, str]]],
    stacks: Mapping[str, Sequence[Signature]],
    *,
    label: str,
    allow: bool,
) -> dict[tuple[str, str], str]:
    conflicts = {
        (a, b): reason
        for members in pairs.values()
        for a, b in members
        if (reason := contrast_conflict(stacks.get(a, ()), stacks.get(b, ()))) is not None
    }
    if conflicts and not allow:
        listed = "; ".join(f"{a} vs {b}: {reason}" for (a, b), reason in conflicts.items())
        raise CampaignRunError(
            f"{label}: {len(conflicts)} contrast(s) would compare judgments from different inference "
            f"stacks or serving settings ({listed}). Re-run a member on the other's stack and serving "
            "settings (dtype, attention backend, context window), or pass allow_stack_mismatch to report them "
            "marked as such"
        )
    return conflicts


def _family_contrasts(
    grids: Mapping[str, DrawGrid],
    families: Mapping[str, Sequence[str]],
    *,
    label: str,
    pairs: Optional[Mapping[str, Sequence[tuple[str, str]]]] = None,
    stacks: Mapping[str, Sequence[Signature]],
    allow_stack_mismatch: bool = False,
    **common: Any,
) -> dict[str, Any]:
    """Return one contrast row per pair of each family; all rows of one call form one Holm family."""
    groups: dict[str, Any] = {
        family: {"members": list(members), "contrasts": []} for family, members in families.items()
    }
    if pairs is None:
        pairs = {family: _all_pairs(members) for family, members in families.items()}
    mismatched = _stack_conflicts(pairs, stacks, label=label, allow=allow_stack_mismatch)
    rows: list[dict[str, Any]] = []
    for family in families:
        for a, b in pairs[family]:
            row = contrast_row(grids, a, b, shared=False, **common)
            row["family"] = label
            row["family_group"] = family
            if (a, b) in mismatched:
                row["stack_mismatch"] = mismatched[(a, b)]
            rows.append(row)
    for row in holm_adjust(rows):
        row["family_size"] = len(rows)
        groups[row["family_group"]]["contrasts"].append(row)
    return groups


def _families(
    conditions: Mapping[str, ConditionSpec],
    varying: Sequence[str],
    extra_varying: Sequence[str] = (),
) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for condition_id, condition in conditions.items():
        payload = condition.to_dict()
        for key in ("condition_id", *varying):
            payload.pop(key, None)
        for key in extra_varying:
            payload["extra"].pop(key, None)
        groups.setdefault(json.dumps(payload, sort_keys=True), []).append(condition_id)
    return list(groups.values())


def style_families(conditions: Mapping[str, ConditionSpec]) -> dict[str, list[str]]:
    """Return conditions that differ only in prompt style, grouped and ordered naive -> encouraged -> forced."""
    return {
        "+".join(sorted(members)): sorted(
            members, key=lambda cid: _STYLE_ORDER.get(conditions[cid].prompt_style, 99)
        )
        for members in _families(conditions, ("prompt_style", "forced_json_requirement", "response_contract"))
    }


def ruleset_families(conditions: Mapping[str, ConditionSpec]) -> dict[str, list[str]]:
    """Return conditions that differ only in their ruleset, ordered by id; two conditions over one ruleset are
    refused."""
    families: dict[str, list[str]] = {}
    for members in _families(conditions, ("rule_set_name", "rule_set_sha256"), ("ruleset",)):
        digests = [conditions[cid].rule_set_sha256 for cid in members]
        if len(set(digests)) < 2:
            continue
        if len(set(digests)) != len(digests):
            raise CampaignRunError(
                f"conditions {sorted(members)} differ only in ruleset yet two of them judge the same one"
            )
        ordered = sorted(members)
        families["+".join(ordered)] = ordered
    return families


def missing_rulesets(conditions: Mapping[str, ConditionSpec], members: Sequence[str]) -> list[str]:
    """Return the rulesets a family declares that none of its analysed members judged."""
    extras = [dict(conditions[cid].extra) for cid in members]
    declared = extras[0].get("ruleset_family")
    if declared is None:
        return []
    present = {extra.get("ruleset") for extra in extras}
    return [label for label in str(declared).split("+") if label not in present]


# What members of a policy family may differ in besides ``extra["policy"]``: which weights the judge serves.
POLICY_FIELDS = (
    "model_path", "model_weights_sha256", "adapter_path", "adapter_sha256", "adapter_config_sha256",
)
# Positive when the member policy is less inconsistent than the reference.
POLICY_ORIENTATION = "rate(reference) - rate(member)"


def served_weights(condition: ConditionSpec) -> tuple[Optional[str], Optional[str]]:
    """Return the weights a condition's judge serves, by digest where the identity has one."""
    return condition.model_weights_sha256 or condition.model_path, condition.adapter_sha256


@dataclass(frozen=True)
class PolicyFamily:
    """One declared policy family: ``members`` are condition ids, ``policies`` maps each present member's label to its
    id."""

    members: tuple[str, ...]
    policies: Mapping[str, str]
    declared: tuple[str, ...]
    reference: str

    @property
    def missing(self) -> list[str]:
        return [label for label in self.declared if label not in self.policies]

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """``(reference, member)`` for every other present member -- never member pairs."""
        reference = self.policies.get(self.reference)
        if reference is None:
            return []
        return [(reference, cid) for cid in self.members if cid != reference]


def policy_families(conditions: Mapping[str, ConditionSpec]) -> dict[str, PolicyFamily]:
    """Return conditions that differ only in the judge's weights and ``extra["policy"]``. Refuses an undeclared or
    repeated label, a reference outside the declaration, and two members serving the same weights."""
    declaring = {cid: spec for cid, spec in conditions.items() if "policy" in dict(spec.extra)}
    families: dict[str, PolicyFamily] = {}
    for group in _families(declaring, POLICY_FIELDS, ("policy",)):
        extra = dict(declaring[group[0]].extra)
        declared_value, reference = extra.get("policy_family"), extra.get("policy_reference")
        if declared_value is None or reference is None:
            raise CampaignRunError(
                f"conditions {sorted(group)} declare a policy but not policy_family and "
                "policy_reference; a policy contrast needs its declared members and reference"
            )
        declared = tuple(str(declared_value).split("+"))
        if len(set(declared)) != len(declared) or str(reference) not in declared:
            raise CampaignRunError(
                f"policy_family {declared_value!r} must name distinct policies including the "
                f"reference {reference!r}"
            )
        labels = {cid: str(dict(declaring[cid].extra)["policy"]) for cid in group}
        undeclared = any(label not in declared for label in labels.values())
        if undeclared or len(set(labels.values())) != len(labels):
            raise CampaignRunError(
                f"conditions {sorted(group)} carry policies {sorted(labels.values())}: each must be "
                f"a distinct member of the declared family {declared_value!r}"
            )
        weights = [served_weights(declaring[cid]) for cid in group]
        if len(set(weights)) != len(weights):
            raise CampaignRunError(
                f"conditions {sorted(group)} differ only in policy yet two of them serve the same "
                "weights (one merged checkpoint or adapter, or both the base model)"
            )
        by_label = {label: cid for cid, label in labels.items()}
        policies = {label: by_label[label] for label in declared if label in by_label}
        members = tuple(policies.values())
        families["+".join(members)] = PolicyFamily(
            members=members,
            policies=policies,
            declared=declared,
            reference=str(reference),
        )
    return families


def first_attempt_acceptance_share(statistics: Mapping[str, Any]) -> Optional[float]:
    """Return the share of accepted bundles accepted on the first attempt, or None when nothing was accepted."""
    histogram = statistics["attempts_histogram"]
    accepted = sum(histogram.values())
    return histogram.get("1", 0) / accepted if accepted else None


def policy_contrasts(
    registered: Mapping[str, ConditionSpec],
    grids: Mapping[str, DrawGrid],
    *,
    statistics: Mapping[str, Mapping[str, Any]],
    excluded: Mapping[str, Mapping[str, Any]],
    campaign_id: str,
    allow_incomplete: bool,
    stacks: Mapping[str, Sequence[Signature]],
    allow_stack_mismatch: bool = False,
    **common: Any,
) -> dict[str, Any]:
    # Every family's ``reference - member`` rows form one Holm family across the root; a family short of a member is
    # refused, or reported without contrasts under ``allow_incomplete``.
    families = policy_families(registered)
    missing = {name: family.missing for name, family in families.items() if family.missing}
    if missing and not allow_incomplete:
        raise CampaignRunError(
            f"campaign {campaign_id!r}: policy families are missing declared policies "
            f"{missing}; their Holm family is not the declared one yet"
        )
    dropped = {
        name: {label: cid for label, cid in family.policies.items() if cid in excluded}
        for name, family in families.items()
    }
    dropped = {name: members for name, members in dropped.items() if members}
    member_statistics = {
        **statistics, **{cid: entry["attempt_statistics"] for cid, entry in excluded.items()}
    }
    complete = {
        name: family for name, family in families.items() if name not in missing and name not in dropped
    }
    groups = _family_contrasts(
        grids,
        {name: family.members for name, family in complete.items()},
        label="policy_contrasts",
        pairs={name: family.pairs for name, family in complete.items()},
        stacks=stacks,
        allow_stack_mismatch=allow_stack_mismatch,
        **common,
    )
    for name, family in families.items():
        group = groups.setdefault(name, {"members": list(family.members), "contrasts": []})
        label_of = {cid: label for label, cid in family.policies.items()}
        for row in group["contrasts"]:
            row["policy_a"], row["policy_b"] = label_of[row["a"]], label_of[row["b"]]
        group.update(
            {
                "orientation": POLICY_ORIENTATION,
                "reference_policy": family.reference,
                "declared_policies": list(family.declared),
                "policies": dict(family.policies),
                "adapter_sha256": {
                    label: registered[cid].adapter_sha256 for label, cid in family.policies.items()
                },
                "model_weights_sha256": {
                    label: registered[cid].model_weights_sha256 for label, cid in family.policies.items()
                },
                "attempt_statistics": {
                    label: member_statistics[cid] for label, cid in family.policies.items()
                },
                "first_attempt_acceptance_share": {
                    label: first_attempt_acceptance_share(member_statistics[cid])
                    for label, cid in family.policies.items()
                },
            }
        )
        if name in missing:
            group["missing_policies"] = missing[name]
        if name in dropped:
            group["excluded_policies"] = dropped[name]
    return groups


def _policy_verdicts(
    condition_id: str, cells: Mapping[tuple[str, int], Mapping[str, Any]], grid: DrawGrid, *, n_draws: int
) -> VerdictGrids:
    verdicts = VerdictGrids.from_cells(condition_id, cells, scenario_ids=grid.scenario_ids, n_draws=n_draws,
                                       identity_id=grid.identity_id)
    if not np.array_equal(grid.values, (verdicts.full.values != verdicts.rule_max.values).astype(float)):
        raise CampaignRunError(
            f"{condition_id}: its bundle outcomes' contradiction is not phi(full set) != max phi(rules) in every "
            "cell; kappa and the flat rate would describe different outcomes"
        )
    return verdicts


def policy_discrimination(
    groups: dict[str, Any],
    registered: Mapping[str, ConditionSpec],
    grids: Mapping[str, DrawGrid],
    cells: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    *,
    condition_dirs: Mapping[str, Path],
    campaign_ids: Mapping[str, str],
    n_draws: int,
    rule_set_path: Optional[Path],
    stacks: Mapping[str, Sequence[Signature]],
    allow_stack_mismatch: bool,
    n_resamples: int,
    ci_level: float,
    seed: int,
    n_permutations: int,
) -> list[dict[str, Any]]:
    # Adds per-arm statistics to each policy family and, where all members are analysed, kappa contrasts: a second Holm
    # family across the root.
    bootstrap = dict(n_resamples=n_resamples, ci_level=ci_level, seed=seed)
    families = policy_families(registered)
    forced = sorted({
        registered[cid].rule_set_sha256
        for family in families.values() for cid in family.policies.values()
        if cid in cells and registered[cid].contract == FORCED_JSON
    })
    if forced and rule_set_path is None:
        campaigns = ", ".join(repr(campaign) for campaign in sorted(set(campaign_ids.values())))
        raise CampaignRunError(
            f"campaign {campaigns}: a policy family of forced-prompt conditions reports the same-call rate per arm, "
            "reclassified from the stored raw outputs against the ruleset; pass rule_set_path (the file whose "
            "digest the conditions record)"
        )
    rules = {digest: load_rules(rule_set_path, digest) for digest in forced}
    rows: list[dict[str, Any]] = []
    complete: dict[str, Any] = {}
    for name, family in families.items():
        group = groups[name]
        group.update({
            "kappa_orientation": KAPPA_ORIENTATION, "strata": None, "arms": {},
            "discrimination_contrasts": [], "decompositions": {}, "stratified_differences": {},
        })
        present = {label: cid for label, cid in family.policies.items() if cid in cells}
        if not present:
            continue
        verdicts = {
            cid: _policy_verdicts(cid, cells[cid], grids[cid], n_draws=n_draws) for cid in present.values()
        }
        strata = read_strata(
            [condition_dirs[cid] for cid in present.values()], grids[next(iter(present.values()))].scenario_ids
        )
        arms = group["arms"]
        for label, cid in present.items():
            arms[label] = arm_statistics(verdicts[cid], strata, **bootstrap)
            spec = registered[cid]
            if spec.contract == FORCED_JSON:
                outcomes = reclassify_forced_maps(
                    condition_dirs[cid], spec, campaign_id=campaign_ids[cid], scenario_ids=verdicts[cid].full.scenario_ids,
                    n_draws=n_draws, rules=rules[spec.rule_set_sha256], cells=cells[cid],
                )
                arms[label][SAME_CALL_KEY] = same_call_statistics(
                    outcomes, scenario_ids=verdicts[cid].full.scenario_ids, n_draws=n_draws, **bootstrap
                )
        group["strata"] = strata.to_dict()
        if "missing_policies" in group or "excluded_policies" in group:
            continue
        complete[name] = family
        label_of = {cid: label for label, cid in family.policies.items()}
        for a, b in family.pairs:
            row = kappa_contrast(verdicts[a], verdicts[b], strata, n_permutations=n_permutations, **bootstrap)
            row.update({"family": "discrimination_contrasts", "family_group": name,
                        "policy_a": label_of[a], "policy_b": label_of[b]})
            rows.append(row)
            group["decompositions"][label_of[b]] = cross_arm_decomposition(verdicts[a], verdicts[b], **bootstrap)
            group["stratified_differences"][label_of[b]] = stratified_differences(
                verdicts[a], verdicts[b], **bootstrap
            )
    mismatched = _stack_conflicts(
        {name: family.pairs for name, family in complete.items()}, stacks,
        label="discrimination_contrasts", allow=allow_stack_mismatch,
    )
    adjusted = holm_adjust(rows, test_method=KAPPA_TEST_METHOD)
    for row in adjusted:
        row["family_size"] = len(adjusted)
        if (row["a"], row["b"]) in mismatched:
            row["stack_mismatch"] = mismatched[(row["a"], row["b"])]
        groups[row["family_group"]]["discrimination_contrasts"].append(row)
    return adjusted


def attempt_statistics(draws: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the pooled retry histogram and invalid-component counts over a condition's draws."""
    histogram: dict[str, int] = {}
    per_component: dict[str, int] = {}
    invalid: dict[str, int] = {}
    rejected = transport = 0
    component_scope = False
    for record in draws:
        for key, count in (record.get("attempts_histogram") or {}).items():
            histogram[key] = histogram.get(key, 0) + int(count)
        if "component_attempts_histogram" in record:
            component_scope = True
            for key, count in record["component_attempts_histogram"].items():
                per_component[key] = per_component.get(key, 0) + int(count)
        for key, count in (record.get("invalid_components") or {}).items():
            invalid[key] = invalid.get(key, 0) + int(count)
        rejected += int(record.get("rejected_attempts") or 0)
        transport += int(record.get("transport_failed_attempts") or 0)
    accepted = sum(histogram.values())
    return {
        "attempts_histogram": dict(sorted(histogram.items())),
        **({"component_attempts_histogram": dict(sorted(per_component.items()))} if component_scope else {}),
        "rejected_attempts": rejected,
        "transport_failed_attempts": transport,
        "invalid_components": dict(sorted(invalid.items())),
        "invalid_attempt_rate": rejected / (accepted + rejected) if accepted + rejected else None,
    }


EXCLUSION_RULE = (
    "The fallback: one scenario, out of attempts in a draw of an arm of one judge, dropped from every arm "
    "of that judge in every draw, so the judge's arms stay paired over the same scenarios; every other condition "
    "keeps the whole manifest, and no other draw may be incomplete. One bootstrap per scenario set: the other "
    "conditions' rate intervals are their own bootstrap's, apart from the judge's arms (point estimates, contrasts "
    "and p-values are unaffected). The arms' attempt statistics are their draw gates', so they count the dropped scenario's attempts "
    "too (its fate in each draw is recorded beside them)"
)


def fallback_unserved(registered: Mapping[str, ConditionSpec], judge: str) -> list[str]:
    """Return ``judge``'s arms the fallback does not serve: forced-style arms and policy-family members read every draw
    whole."""
    policy = {cid for family in policy_families(registered).values() for cid in family.members}
    return [cid for cid, spec in registered.items()
            if spec.model_name == judge and (spec.contract == FORCED_JSON or cid in policy)]


def _exclusion_arms(
    registered: Mapping[str, ConditionSpec], gate: Mapping[str, Any], judge: str, scenario_id: str,
    scenario_ids: Sequence[str],
) -> list[str]:
    arms = [cid for cid, spec in registered.items() if spec.model_name == judge]
    unsupported = fallback_unserved(registered, judge)
    draws = {(cid, draw): record for cid in arms for draw, record in gate["conditions"][cid]["draws"].items()}
    other = sorted(f"{cid} draw {draw} ({record['state']})" for (cid, draw), record in draws.items()
                   if record["state"] != STATE_COMPLETE
                   and not (draw_ended(record) and record.get("exhausted") == [scenario_id]))
    exhausted = [key for key, record in draws.items() if record["state"] == STATE_INCOMPLETE_EXHAUSTED]
    problems = [message for failed, message in (
        (not arms, f"no condition of judge {judge!r} is registered"),
        (scenario_id not in scenario_ids, f"scenario {scenario_id[:12]} is not the manifest's"),
        (bool(unsupported), f"{unsupported} are forced-style or policy arms, which it does not serve"),
        (bool(other), f"{other} are incomplete otherwise than by that scenario alone"),
        (not other and not exhausted, f"scenario {scenario_id[:12]} is out of attempts in no draw of {arms}: the "
                                      "fallback never drops a scenario that is not exhausted"),
    ) if failed]
    if problems:
        raise CampaignRunError(f"the fallback (drop scenario {scenario_id[:12]} from the arms of judge {judge!r}): "
                               + "; ".join(problems))
    return arms


def exclusion_outcomes(
    condition_root: Path | str,
    condition: ConditionSpec,
    *,
    campaign_id: str,
    scenario_ids: Sequence[str],
    excluded: str,
    expected_draws: int,
    scenario_manifest_sha256: str,
    max_semantic_attempts: int,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Return one arm's bundle outcomes over ``scenario_ids`` by (scenario, draw) with their provenance, refusing a draw
    incomplete beyond the excluded scenario."""
    root = Path(condition_root)
    names = sorted(path.name for path in root.glob("draw-*") if path.is_dir())
    if names != [draw_dirname(draw) for draw in range(expected_draws)]:
        raise CampaignRunError(f"{root}: expected {expected_draws} draw directories, found {names}")
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    completions: list[dict[str, Any]] = []
    fates: dict[str, dict[str, Any]] = {}
    for draw in range(expected_draws):
        draw_dir = root / draw_dirname(draw)
        gate = json.loads((draw_dir / COMPLETION_FNAME).read_text(encoding="utf-8"))
        identity = (gate.get("decoder_draw_id"), gate.get("scenario_manifest_sha256"), gate.get("condition_fingerprint"))
        if identity != (draw, scenario_manifest_sha256, condition.fingerprint):
            raise CampaignRunError(f"{draw_dir}: its gate is another draw's, population's or condition's")
        if gate.get("state") == STATE_COMPLETE:
            rows, _ = read_draw_outcomes(draw_dir, draw, expected_scenario_ids=[*scenario_ids, excluded],
                                         scenario_manifest_sha256=scenario_manifest_sha256)
            fates[str(draw)] = {"state": STATE_COMPLETE, "accepted_at_attempt": rows.pop(excluded)["semantic_attempt_id"]}
        elif draw_ended(gate) and gate.get("exhausted") == [excluded]:
            rows, attempts = _exhausted_draw_outcomes(draw_dir, draw, condition, campaign_id=campaign_id,
                                                      excluded=excluded, scenario_ids=scenario_ids,
                                                      scenario_manifest_sha256=scenario_manifest_sha256,
                                                      max_semantic_attempts=max_semantic_attempts)
            fates[str(draw)] = {"state": STATE_INCOMPLETE_EXHAUSTED, "attempts": attempts}
        else:
            raise CampaignRunError(f"{draw_dir}: {gate.get('state')!r}, not complete or exhausted at "
                                   f"{excluded[:12]} alone")
        cells.update(((sid, draw), row) for sid, row in rows.items())
        completions.append(gate)
    provenance = {"condition_fingerprint": condition.fingerprint, "n_draws": expected_draws, "draws": completions,
                  "excluded_scenario": fates}
    return cells, provenance


def _exhausted_draw_outcomes(
    draw_dir: Path, draw: int, condition: ConditionSpec, *, campaign_id: str, excluded: str,
    scenario_ids: Sequence[str], scenario_manifest_sha256: str, max_semantic_attempts: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    state = DrawLedger(draw_dir, campaign_id=campaign_id, condition_id=condition.condition_id,
                       decoder_draw_id=draw).load()
    used = state.next_attempt(excluded)
    if state.is_accepted(excluded) or used < max_semantic_attempts:
        raise CampaignRunError(
            f"{draw_dir}: its gate says {excluded[:12]} is out of attempts, but its ledger has it "
            + ("accepted" if state.is_accepted(excluded) else f"at {used} of the campaign's {max_semantic_attempts}")
            + ": a gate written under another cap (an amendment raised it); resume the draw under the "
            "campaign's cap before the fallback reads it"
        )
    provenance = json.loads((draw_dir / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
    identity = (provenance.get("condition_fingerprint"), provenance.get("decoder_draw_id"),
                provenance.get("scenario_manifest_sha256"))
    if identity != (condition.fingerprint, draw, scenario_manifest_sha256):
        raise CampaignRunError(f"{draw_dir}: its provenance is another draw's, population's or condition's")
    components = list(provenance["component_ids"])
    schema = ComponentSchema.for_recorded_judgments(condition, components)
    rows = {}
    for sid in state.accepted:
        bundle = state.accepted_bundle(sid)
        if [component.component_id for component in bundle.components] != components:
            raise CampaignRunError(f"{draw_dir}: {sid[:12]}'s accepted bundle holds other components than its "
                                   "provenance names")
        rows[sid] = bundle_outcome(schema, bundle)
    if sorted(rows) != sorted(scenario_ids) or any(int(row["decoder_draw_id"]) != draw for row in rows.values()):
        raise CampaignRunError(f"{draw_dir}: its outcomes are not one per scenario of the manifest but "
                               f"{excluded[:12]}")
    return rows, used


def root_order(identity_id: str) -> tuple[str, ...]:
    """Return where a condition sorts among a root's: bootstrap resamples are drawn in this order, whatever the
    condition names."""
    return (PurePosixPath(identity_id) / CONDITION_FNAME).parts


def _covered_conditions(campaign_root: Path | str) -> tuple[str, ...]:
    path = Path(campaign_root) / AGGREGATE_DIRNAME / ESTIMATES_FNAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(payload, dict):
        return ()
    parts = (payload.get("conditions"), payload.get("excluded_conditions", {}))
    return tuple(cid for part in parts for cid in part) if all(isinstance(part, dict) for part in parts) else ()


def refuse_stripped(campaign_root: Path | str) -> None:
    """Refuse to aggregate a published copy, or a root missing a condition its recorded analysis covers: the bootstrap
    and Holm families would change."""
    root = Path(campaign_root)
    refuse_published(root, "an analysis", "each draw's completion record in full (its attempt counts)")
    absent = sorted(cid for cid in _covered_conditions(root) if not (root / cid / CONDITION_FNAME).is_file())
    if absent:
        raise CampaignRunError(
            f"{root} does not hold {absent}, which its {AGGREGATE_DIRNAME}/{ESTIMATES_FNAME} names (the analysis it "
            "holds estimates them or lists them as excluded): it is a copy that leaves their directories out (a "
            "published root as shipped leaves out its conditions that are not in the paper), so an aggregate of it "
            "would change every interval its estimates record (they are the full tree's). Read the estimates it "
            "holds, or aggregate a copy holding every condition they name"
        )


def analyze_campaign(
    campaign_root: Path | str,
    *,
    n_resamples: Optional[int] = None,
    ci_level: Optional[float] = None,
    seed: Optional[int] = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rule_ids: Optional[Sequence[str]] = None,
    allow_incomplete_conditions: bool = False,
    allow_stack_mismatch: bool = False,
    rule_set_path: Optional[Path | str] = None,
    exclude_exhausted: Optional[tuple[str, str]] = None,
    evaluation_modes: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Compute and write ``aggregate/estimates.json`` for a complete campaign under its lock; ``rule_set_path`` is the
    ruleset policy families' forced-prompt members judged, the other options ``analyze``'s."""
    refuse_stripped(campaign_root)
    with campaign_lock(campaign_root):
        return _analyze_campaign(
            Path(campaign_root), n_resamples=n_resamples, ci_level=ci_level, seed=seed,
            n_permutations=n_permutations, rule_ids=rule_ids,
            allow_incomplete_conditions=allow_incomplete_conditions, allow_stack_mismatch=allow_stack_mismatch,
            rule_set_path=rule_set_path, exclude_exhausted=exclude_exhausted, evaluation_modes=evaluation_modes,
        )


def _analyze_campaign(
    root: Path,
    *,
    n_resamples: Optional[int],
    ci_level: Optional[float],
    seed: Optional[int],
    n_permutations: int,
    rule_ids: Optional[Sequence[str]],
    allow_incomplete_conditions: bool,
    allow_stack_mismatch: bool,
    rule_set_path: Optional[Path | str],
    exclude_exhausted: Optional[tuple[str, str]],
    evaluation_modes: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    refuse_stripped(root)
    if (root / SHARED_BANK_DESIGN_FNAME).exists():
        raise CampaignRunError(
            f"{root} is a shared-bank campaign (it carries {SHARED_BANK_DESIGN_FNAME}); its bank and "
            "subset conditions are not prompt-style arms -- analyse it with "
            "ruleforge.evaluate.bank.analyze_shared_bank_campaign"
        )
    manifest = load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME)
    if manifest.get("campaign_mode") == CAMPAIGN_MODE_SCREENING:
        raise CampaignRunError(
            f"campaign {manifest['campaign_id']!r} is a screening campaign: one draw over a "
            "candidate population, run to freeze a manifest, never an estimate"
        )
    if allow_incomplete_conditions or exclude_exhausted is not None:
        gate = campaign_completion(root)
    else:
        gate = require_complete_campaign(root)
    scenario_manifest = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME)
    # In identity order (``root_order``), which the bootstrap draws each condition's inner resample in.
    registered = dict(sorted(load_conditions(root).items(), key=lambda item: root_order(item[1].identity_id)))
    if evaluation_modes is not None:
        registered = {cid: spec for cid, spec in registered.items() if spec.evaluation_mode in evaluation_modes}
    ids = scenario_manifest.scenario_ids
    arms = [] if exclude_exhausted is None else _exclusion_arms(registered, gate, *exclude_exhausted, ids)
    excluded: dict[str, Any] = {}
    for condition_id, entry in gate["conditions"].items():
        if condition_id not in registered:
            continue
        states = {draw: record["state"] for draw, record in entry["draws"].items()}
        if condition_id not in arms and any(state != STATE_COMPLETE for state in states.values()):
            excluded[condition_id] = {
                "draw_states": states,
                "attempt_statistics": attempt_statistics(
                    [record for record in entry["draws"].values() if record.get("recorded")]
                ),
            }
    if excluded and not allow_incomplete_conditions:
        raise CampaignRunError(
            f"campaign {manifest['campaign_id']!r}: the fallback drops one scenario from one judge's arms, and "
            f"{sorted(excluded)} are incomplete too: {json.dumps({cid: e['draw_states'] for cid, e in excluded.items()})}"
        )
    conditions = {cid: spec for cid, spec in registered.items() if cid not in excluded}
    if not conditions:
        raise CampaignRunError(
            f"campaign {manifest['campaign_id']!r} has no complete condition to analyse"
        )
    # A provider judge's stack depends on none of this process's packages.
    args_by_digest: EngineArgsIndex = {}
    stacks = {
        condition_id: (
            condition_signatures(root / condition_id, args_by_digest=args_by_digest)
            if condition.model_type == LOCAL_MODEL_TYPE
            else []
        )
        for condition_id, condition in conditions.items()
    }
    mixed = mixed_conditions(stacks, args_by_digest=args_by_digest)
    if mixed and not allow_stack_mismatch:
        raise CampaignRunError(
            f"campaign {manifest['campaign_id']!r}: {len(mixed)} condition(s) hold judgments from several "
            f"inference stacks or engine configurations ({describe_mismatches(mixed)}); a rate over them "
            "would pool two engines' judgments. Re-run them on one stack and engine configuration, or pass "
            "allow_stack_mismatch to report them marked as such"
        )
    bootstrap = manifest["bootstrap"]
    n_resamples = int(bootstrap["n_resamples"] if n_resamples is None else n_resamples)
    ci_level = float(bootstrap["ci_level"] if ci_level is None else ci_level)
    seed = int(bootstrap["seed"] if seed is None else seed)
    n_draws = int(manifest["expected_decoder_draws"])
    kept = tuple(sid for sid in ids if exclude_exhausted is None or sid != exclude_exhausted[1])
    ids_of = {condition_id: kept if condition_id in arms else ids for condition_id in conditions}

    cells: dict[str, dict] = {}
    provenance: dict[str, Any] = {}
    grids: dict[str, DrawGrid] = {}
    for condition_id in conditions:
        if condition_id in arms:
            cells[condition_id], provenance[condition_id] = exclusion_outcomes(
                root / condition_id, conditions[condition_id], campaign_id=manifest["campaign_id"],
                scenario_ids=kept, excluded=exclude_exhausted[1], expected_draws=n_draws,
                scenario_manifest_sha256=scenario_manifest.manifest_sha256,
                max_semantic_attempts=int(manifest["max_semantic_attempts"]),
            )
        else:
            cells[condition_id], provenance[condition_id] = load_condition_outcomes(
                root / condition_id,
                expected_scenario_ids=ids,
                expected_draws=n_draws,
                scenario_manifest_sha256=scenario_manifest.manifest_sha256,
            )
        grids[condition_id] = outcome_grid(
            condition_id, cells[condition_id], scenario_ids=ids_of[condition_id], n_draws=n_draws,
            identity_id=conditions[condition_id].identity_id,
        )

    statistics = {
        condition_id: attempt_statistics(provenance[condition_id]["draws"])
        for condition_id in conditions
    }
    # One bootstrap per scenario set: one for the whole manifest unless the fallback dropped a scenario from a judge's
    # arms.
    rates: dict[str, Any] = {}
    for scenario_set in dict.fromkeys(ids_of.values()):
        rates.update(hierarchical_bootstrap(
            {cid: grid for cid, grid in grids.items() if ids_of[cid] == scenario_set},
            n_resamples=n_resamples, ci_level=ci_level, seed=seed,
        )["conditions"])
    common = dict(n_resamples=n_resamples, ci_level=ci_level, seed=seed, n_permutations=n_permutations)
    stack_checks = dict(stacks=stacks, allow_stack_mismatch=allow_stack_mismatch)

    # Every style contrast of the campaign is one multiplicity family.
    style_groups = _family_contrasts(
        grids, style_families(conditions), label="style_contrasts", **stack_checks, **common
    )
    ruleset_members = ruleset_families(conditions)
    partial = {
        family: labels
        for family, members in ruleset_members.items()
        if (labels := missing_rulesets(conditions, members))
    }
    if partial and not allow_incomplete_conditions:
        raise CampaignRunError(
            f"campaign {manifest['campaign_id']!r}: ruleset families are missing declared "
            f"rulesets {partial}; their Holm family is not the declared one yet"
        )
    ruleset_groups = _family_contrasts(
        grids,
        {family: members for family, members in ruleset_members.items() if family not in partial},
        label="ruleset_contrasts",
        **stack_checks,
        **common,
    )
    for family, labels in partial.items():
        ruleset_groups[family] = {
            "members": ruleset_members[family], "contrasts": [], "missing_rulesets": labels,
        }
    policy_groups = policy_contrasts(
        registered, grids, statistics=statistics, excluded=excluded,
        campaign_id=manifest["campaign_id"], allow_incomplete=allow_incomplete_conditions,
        **stack_checks, **common,
    )
    discrimination_rows = policy_discrimination(
        policy_groups, registered, grids, cells, condition_dirs={cid: root / cid for cid in cells},
        campaign_ids={cid: manifest["campaign_id"] for cid in cells}, n_draws=n_draws,
        rule_set_path=None if rule_set_path is None else Path(rule_set_path), **stack_checks, **common,
    )

    forced: dict[str, Any] = {}
    for condition_id, condition in conditions.items():
        if (
            condition.contract != FORCED_JSON
            or condition.forced_json_requirement != FORCED_JSON_COMPLETE_RULE_MAP
        ):
            continue
        same_call = outcome_grid(
            condition_id, cells[condition_id], scenario_ids=ids, n_draws=n_draws,
            quantity="in_context_contradiction", identity_id=condition.identity_id,
        )
        methods = contrast_row(
            {SEPARATE_CALL_KEY: grids[condition_id], SAME_CALL_KEY: same_call},
            SEPARATE_CALL_KEY, SAME_CALL_KEY, shared=True, **common,
        )
        wanted = list(rule_ids) if rule_ids is not None else sorted(
            next(iter(cells[condition_id].values()))["rule_judgments"]
        )
        rule_rows = []
        for rule_id in wanted:
            alone = rule_grids(
                condition_id, cells[condition_id], scenario_ids=ids, n_draws=n_draws, rule_id=rule_id,
                identity_id=condition.identity_id,
            )
            in_context = rule_grids(
                condition_id, cells[condition_id], scenario_ids=ids, n_draws=n_draws,
                rule_id=rule_id, field="in_context_rule_judgments", identity_id=condition.identity_id,
            )
            row = contrast_row(
                {"in_context": in_context, "alone": alone}, "in_context", "alone",
                shared=True, **common,
            )
            row["rule_id"] = rule_id
            row["family"] = f"rule_context:{condition_id}"
            row["family_size"] = len(wanted)
            rule_rows.append(row)
        forced[condition_id] = {
            SEPARATE_VS_SAME_CALL_KEY: methods,
            SAME_CALL_RATE_KEY: hierarchical_bootstrap(
                {SAME_CALL_KEY: same_call}, n_resamples=n_resamples, ci_level=ci_level, seed=seed
            )["conditions"][SAME_CALL_KEY],
            "rule_context": holm_adjust(rule_rows),
        }

    contrast_rows = [
        row
        for groups in (style_groups, ruleset_groups, policy_groups)
        for group in groups.values()
        for row in group["contrasts"]
    ]
    mismatches = mixed + [
        mismatch_record(row["family"], [row["a"], row["b"]], row["stack_mismatch"])
        for row in contrast_rows + discrimination_rows
        if "stack_mismatch" in row
    ]
    amendment = amendment_disclosure(root, manifest)
    payload = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "campaign_id": manifest["campaign_id"],
        "state": gate["state"],
        **override_record(allow_stack_mismatch, mismatches),
        "inference_method": INFERENCE_METHOD,
        "n_scenarios": len(ids),
        "n_decoder_draws": n_draws,
        "temperature": manifest["temperature"],
        "max_semantic_attempts": manifest["max_semantic_attempts"],
        **retry_scope_record(manifest),
        **({MANIFEST_AMENDMENTS_KEY: [amendment]} if amendment is not None else {}),
        **({SCENARIO_EXCLUSION_KEY: {
            "judge": exclude_exhausted[0], "scenario_id": exclude_exhausted[1], "rule": EXCLUSION_RULE,
            "arms": {cid: {"n_scenarios": len(kept), "n_decoder_draws": n_draws,
                           "excluded_scenario": provenance[cid]["excluded_scenario"]} for cid in arms},
            "n_scenarios_other_conditions": len(ids),
        }} if arms else {}),
        "forced_json_requirement": manifest["forced_json_requirement"],
        "truncation": manifest.get("truncation"),
        "scenario_manifest_sha256": scenario_manifest.manifest_sha256,
        "bootstrap": {"n_resamples": n_resamples, "ci_level": ci_level, "seed": seed},
        "randomization_test": {"n_permutations": n_permutations, "seed": seed},
        "conditions": {
            condition_id: {
                **rates[condition_id],
                "condition_fingerprint": provenance[condition_id]["condition_fingerprint"],
                "prompt_style": condition.prompt_style,
                "model_name": condition.model_name,
                "rule_set_name": condition.rule_set_name,
                "rule_set_sha256": condition.rule_set_sha256,
                "extra": dict(condition.extra),
                "sampling_truncation": condition.to_dict()["sampling_truncation"],
                "attempt_statistics": statistics[condition_id],
                **stacks_record(allow_stack_mismatch, stacks[condition_id]),
            }
            for condition_id, condition in conditions.items()
        },
        "style_families": style_groups,
        "ruleset_families": ruleset_groups,
        "policy_families": policy_groups,
        "forced_conditions": forced,
        "excluded_conditions": excluded,
        "created_at": utc_now(),
    }
    destination = write_json_sidecar(root / AGGREGATE_DIRNAME / ESTIMATES_FNAME, payload)
    logger.info("Wrote %s", destination)
    return payload
