from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ruleforge.experiments import arm_records as rp
from ruleforge.hub import published_record
from ruleforge.experiments.campaign_roots import (Root, arm_outcomes, arm_stack, open_roots, public_id_of,
                                                registered_condition)
from ruleforge.evaluate.estimates import (
    amendment_sentence,
    campaign_settings_differences,
    estimates_source,
    frozen_settings,
)
from ruleforge.experiments.checks import require
from ruleforge.experiments.goldens import format_interval
from ruleforge.experiments.training.campaign_readout import TrainingReadoutError, design_provenance, family_golden
from ruleforge.evaluate.inference import DrawGrid, hierarchical_bootstrap
from ruleforge.core.digests import file_record
from ruleforge.core.errors import ArtifactRootError
from ruleforge.core.paths import artifact_uri_or_path, is_artifact_uri, resolve
from ruleforge.evaluate.aggregate import POLICY_FIELDS, policy_contrasts, policy_discrimination, policy_families, served_weights
from ruleforge.evaluate.campaign import CONDITION_FNAME, SCENARIO_MANIFEST_FNAME, rebase_record
from ruleforge.evaluate.identity import (
    DRAW_PROVENANCE_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    MANIFEST_RUN_FIELDS,
    ConditionSpec,
    draw_dirname,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.evaluate.discrimination import YES_PHI, VerdictGrids
from ruleforge.evaluate.chain import CHAIN_KEY, rebase_chain
from ruleforge.evaluate.sharding import CodeState, code_state_key, describe_code_state, recorded_code_states
from ruleforge.evaluate.stack import contrast_conflict

__all__ = [
    "CAMPAIGN_RUN_FIELDS",
    "EXTRA_EXEMPT",
    "FamilyArm",
    "TrainingFamilyError",
    "build_readout",
    "identity_differences",
    "summary_markdown",
    "yes_shift",
]

SCHEMA_VERSION = 1
SOURCE = "training_family"
#: The ``extra`` keys a family's arms may differ in beside the weights they serve.
EXTRA_EXEMPT = ("policy", "policy_family")
#: Sampling-manifest keys that belong to a run, not the design; every other key must agree across the roots.
CAMPAIGN_RUN_FIELDS = MANIFEST_RUN_FIELDS + ("campaign_id", "campaign_mode", "conditions")
YES_SHIFT_QUANTITY = "yes_shift_expected_inconsistency"
YES_SHIFT_DEFINITION = (
    "Descriptive, no test and in no Holm family. On the reference's cells (scenario x decoder draw): F its full-set "
    "judgment, M its per-rule max (phi: no 0, unsure 0.5, yes 1), a cell inconsistent iff phi(F) != M (the separate-call "
    "rate's stored contradiction); y_ref and y_mem the two arms' full-set yes shares. p = (y_mem - y_ref) / (1 - y_ref) is "
    "the share of the reference's non-yes judgments that would have to turn yes to reach the member's yes share. The "
    "shifted rate is the reference's expected inconsistency had each of its non-yes judgments turned yes with "
    "probability p, blind to the rules: the mean over its cells of I(phi(F) != M) where F is yes, else "
    "p * I(M != yes) + (1 - p) * I(phi(F) != M) (analytic, no simulation). The share of the gain it explains is "
    "(rate_ref - shifted) / (rate_ref - rate_mem). Not applicable when the member's yes share is not above the "
    "reference's, or its rate is not below. The interval is a scenario-cluster bootstrap of the shifted rate at the "
    "campaigns' settings with p held at its point value (p's own sampling variation is not propagated)."
)


class TrainingFamilyError(TrainingReadoutError):
    """The arms named cannot be read as one family."""


_require = partial(require, error=TrainingFamilyError)


@dataclass(frozen=True)
class FamilyArm:
    """One arm as named on the command line: its policy label, its root (``artifact://``) and its condition id."""

    label: str
    root: str
    condition_id: str


def _identity(spec: ConditionSpec) -> dict[str, Any]:
    payload = spec.to_dict()
    for key in ("condition_id", *POLICY_FIELDS):
        payload.pop(key, None)
    payload["extra"] = {key: value for key, value in payload["extra"].items() if key not in EXTRA_EXEMPT}
    return payload


def identity_differences(a: ConditionSpec, b: ConditionSpec) -> list[str]:
    """Return the sorted identity fields two conditions differ in, beyond what a family's arms may."""
    first, other = _identity(a), _identity(b)
    fields = [key for key in set(first) | set(other) if key != "extra" and first.get(key) != other.get(key)]
    extra = [f"extra.{key}" for key in set(first["extra"]) | set(other["extra"])
             if first["extra"].get(key) != other["extra"].get(key)]
    return sorted(fields + extra)


def _check_command_line(arms: Sequence[FamilyArm], reference: str, labels: Mapping[str, str],
                        merge_manifests: Mapping[str, str], where: str) -> None:
    names = [arm.label for arm in arms]
    named = [(arm.root, arm.condition_id) for arm in arms]
    members = [name for name in names if name != reference]
    _require(
        [
            (len(arms) >= 2, "a family is its reference and at least one member"),
            (not (twice := sorted({name for name in names if names.count(name) > 1})),
             f"labels {twice} name more than one arm: a label is one policy"),
            (not (again := sorted({f"{root} {cid}" for root, cid in named if named.count((root, cid)) > 1})),
             f"{again} named more than once: an arm is one policy"),
            (reference in names, f"the reference {reference!r} is not one of the family's labels {names}"),
            (reference not in names or names[0] == reference,
             f"the reference {reference!r} is not the first arm named ({names}): the panels draw the reference first"),
            (not (bad := sorted({arm.root for arm in arms if not is_artifact_uri(arm.root)})),
             f"roots {bad} are not artifact:// URIs"),
            (set(labels) <= set(names), f"display labels for {sorted(set(labels) - set(names))}, which are not arms"),
            (set(merge_manifests) <= set(members),
             f"merge manifests for {sorted(set(merge_manifests) - set(members))}, which are not members"),
            (not (plain := sorted(uri for uri in merge_manifests.values() if not is_artifact_uri(uri))),
             f"merge manifests {plain} are not artifact:// URIs"),
        ],
        where,
    )


def _check_declarations(specs: Mapping[str, ConditionSpec], reference: str, where: str) -> None:
    declared = {label: dict(spec.extra) for label, spec in specs.items()}
    _require(
        [
            *[(extra.get("policy") == label, f"arm {label} ({specs[label].condition_id}) declared policy "
                                             f"{extra.get('policy')!r}, not {label!r}")
              for label, extra in declared.items()],
            *[(extra.get("policy_reference") == reference, f"arm {label} declared the reference "
                                                           f"{extra.get('policy_reference')!r}, not {reference!r}")
              for label, extra in declared.items()],
        ],
        where,
    )


def _check_identities(specs: Mapping[str, ConditionSpec], reference: str, where: str) -> None:
    _require(
        [
            (not (differing := identity_differences(specs[reference], spec)),
             f"arm {label} differs from the reference {reference} in {differing}, not only in the weights it serves "
             "and its policy labels")
            for label, spec in specs.items() if label != reference
        ],
        where,
    )


def check_design(roots: Mapping[str, Root], where: str) -> None:
    """Check that the roots' sampling manifests share every design key, each at its campaign's frozen settings."""
    manifests = {uri: root.manifest for uri, root in roots.items()}
    keys = sorted({key for manifest in manifests.values() for key in manifest} - set(CAMPAIGN_RUN_FIELDS))
    differing = campaign_settings_differences(manifests, fields=keys)
    _require([(not differing, f"the roots' designs differ in {'; '.join(differing)}")], where)


def _code_states(roots: Mapping[str, Root], allow_code_mismatch: bool, where: str) -> dict[str, CodeState]:
    states = {uri: code_state_key(root.manifest) for uri, root in roots.items()}
    unclean = [uri for uri, (revision, dirty, _) in states.items() if revision is None or dirty is not False]
    problems = [problem for ok, problem in (
        (len(set(states.values())) == 1, "the roots ran different code"),
        (not unclean, f"roots {unclean} record no clean commit (a dirty or unknown tree, or no revision)"),
    ) if not ok]
    _require(
        [(not problems or allow_code_mismatch,
          "; ".join(problems) + " (" + "; ".join(f"{uri} {describe_code_state(state)}" for uri, state in states.items())
          + "); pass allow_code_mismatch to read them anyway, recorded")],
        where,
    )
    return states


#: What the readout keeps of a chain hop's parent: identities only, never its root path.
HOP_PARENT_KEYS = ("campaign_id", "scenario_manifest_sha256", "scenario_source_sha256", "sampling_manifest_sha256")


def _readout_hop(hop: Mapping[str, Any]) -> dict[str, Any]:
    return {"parent": {key: hop["parent"][key] for key in HOP_PARENT_KEYS}, "target": dict(hop["target"]),
            "removed_ids": list(hop["removed_ids"]), "added_ids": list(hop["added_ids"])}


def _rebase(root: Root, condition_ids: Mapping[str, str]) -> Optional[dict[str, Any]]:
    """Return a rebased root's record (its parent, the scenarios swapped, its draws' logged code states), or None; a
    logged code state means a run started on the draw, not that it judged a row there."""
    record = rebase_record(root.path)
    if record is None:
        return None
    return {
        "parent_campaign_id": record["parent"]["campaign_id"],
        "parent_scenario_manifest_sha256": record["parent"]["scenario_manifest_sha256"],
        "parent_sampling_manifest_sha256": record["parent"]["sampling_manifest_sha256"],
        "removed_ids": list(record["removed_ids"]),
        "added_ids": list(record["added_ids"]),
        **({CHAIN_KEY: [_readout_hop(hop) for hop in rebase_chain(record)]} if CHAIN_KEY in record else {}),
        "run_code_states": {
            public_id: [dict(zip(("code_revision", "code_dirty", "code_state_sha256"), state), draws=draws)
                        for state, draws in recorded_code_states(root.path, condition_id).items()]
            for public_id, condition_id in condition_ids.items()
        },
    }


def _rebase_line(uri: str, rebase: Mapping[str, Any]) -> str:
    def ids(values: Sequence[str]) -> str:
        return ", ".join(f"{value[:12]}..." for value in values)

    logged = "; ".join(
        f"{condition_id}: " + ", ".join(
            f"{describe_code_state(code_state_key(state))} "
            f"({', '.join(state['draws'])})" for state in states)
        for condition_id, states in rebase["run_code_states"].items())
    return (f"{uri} is a rebase of campaign {rebase['parent_campaign_id']}'s root from its population "
            f"(scenario manifest {rebase['parent_scenario_manifest_sha256'][:12]}...): scenario(s) "
            f"{ids(rebase['removed_ids'])} removed and {ids(rebase['added_ids'])} added in their place, the added "
            "judged by a resume under the parent's campaign id, at the seeds it gives them; every other judgment is "
            "the parent's. Its sampling manifest's code state is the rebase's; its draws' run logs record these code "
            "states, each logged by a run that started on the draw, whether it judged a row there or found nothing "
            f"left to judge: {logged}.")


def draw_provenances(condition_dir: Path, n_draws: int) -> list[dict[str, Any]]:
    return [json.loads((condition_dir / draw_dirname(draw) / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))
            for draw in range(n_draws)]


def check_engine(stacks: Mapping[str, Mapping[str, Any]], records: Mapping[str, Sequence[Mapping[str, Any]]],
                  where: str) -> dict[str, Any]:
    """Check that every arm ran on one inference stack at one tensor-parallel size; return the engine record."""
    labels = list(stacks)
    conflicts = [
        f"{a} vs {b}: {reason}"
        for i, a in enumerate(labels) for b in labels[i + 1:]
        if (reason := contrast_conflict(stacks[a]["signatures"], stacks[b]["signatures"])) is not None
    ]
    sizes = {label: sorted({record.get("tp_size") for record in records[label]}, key=str) for label in labels}
    _require(
        [
            (not conflicts, "the arms' judgments come from different inference stacks or serving settings ("
                            + "; ".join(conflicts) + "): a family's arms run on one engine"),
            (len({size for values in sizes.values() for size in values}) == 1,
             f"the arms ran at tensor-parallel sizes {sizes}: a family's arms run on one engine"),
        ],
        where,
    )
    return {"stack": stacks[labels[0]]["description"], "short": stacks[labels[0]]["short"],
            "tp_size": sizes[labels[0]][0]}


def _member_merge(label: str, spec: ConditionSpec, provenances: Sequence[Mapping[str, Any]], where: str,
                  located: Optional[str] = None) -> dict[str, Any]:
    """Return a member's merged checkpoint and the base its merge manifest records; the manifest (read where the draws
    recorded it, or at ``located``) must be the file the draws recorded, by sha256."""
    from ruleforge.judges.weights import MANIFEST_FNAME

    at = f"{where}: member {label} ({spec.condition_id})"
    _require([(spec.model_weights_sha256 is not None and spec.adapter_sha256 is None,
               "it serves no merged checkpoint named by content: the family readout pairs merged checkpoints by the "
               "base their merges record (a LoRA-served arm is not a family member)")], at)
    records = {json.dumps(record.get("model_weights"), sort_keys=True) for record in provenances}
    record = json.loads(next(iter(records)))
    _require([(len(records) == 1 and isinstance(record, dict) and record.get("sha256") == spec.model_weights_sha256,
               f"its draws record served weights {sorted(records)}, not one merged checkpoint of digest "
               f"{spec.model_weights_sha256}")], at)
    try:
        path = Path(record["path"]) / MANIFEST_FNAME if located is None else resolve(located)
    except (ArtifactRootError, ValueError) as error:
        raise TrainingFamilyError(f"{at}: its merge manifest {located} does not resolve: {error}") from error
    _require([(path.is_file(), f"its merge manifest {path} is not readable here")], at)
    data = path.read_bytes()
    manifest = json.loads(data)
    base = manifest.get("base") if isinstance(manifest.get("base"), dict) else {}
    digest = base.get("weights_sha256")
    _require(
        [
            (hashlib.sha256(data).hexdigest() == record.get("manifest_sha256"),
             f"{path} is not the merge manifest its draws recorded ({record.get('manifest_sha256')})"),
            ((manifest.get("result") or {}).get("weights_sha256") == spec.model_weights_sha256,
             f"{path} records weights {(manifest.get('result') or {}).get('weights_sha256')}, not the arm's"),
            (isinstance(digest, str) and len(digest) == 64 and isinstance(base.get("path"), str),
             f"{path} records no base weights digest and directory (base {base})"),
        ],
        at,
    )
    # the digest of the manifest as the Hub publishes it: the same from a recorded root and from a fetched one
    published = hashlib.sha256(published_record(path.name, data)).hexdigest()
    return {"model_weights_sha256": spec.model_weights_sha256, "merge_manifest": artifact_uri_or_path(path),
            "merge_manifest_sha256": published, "base_weights_sha256": digest,
            "base_directory": Path(base["path"]).name}


def _check_weights(specs: Mapping[str, ConditionSpec], records: Mapping[str, Sequence[Mapping[str, Any]]],
                   reference: str, where: str, merge_manifests: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Check that no two arms share weights, every member's merge records one base, and the reference is that base
    (by content, or by directory name); return the base-weights record."""
    served = {label: served_weights(spec) for label, spec in specs.items()}
    shared = sorted(label for label, weights in served.items() if list(served.values()).count(weights) > 1)
    _require([(not shared, f"arms {shared} serve the same weights: a contrast between them compares nothing")], where)
    members = {label: _member_merge(label, spec, records[label], where, (merge_manifests or {}).get(label))
               for label, spec in specs.items() if label != reference}
    bases = sorted({merge["base_weights_sha256"] for merge in members.values()})
    names = sorted({merge["base_directory"] for merge in members.values()})
    _require([(len(bases) == 1, "the members' merges record different base weights ("
                                + ", ".join(f"{label} {merge['base_weights_sha256']}" for label, merge in members.items())
                                + "): their contrasts with one reference would not share a base")], where)
    spec = specs[reference]
    by_content = spec.model_weights_sha256 is not None
    _require(
        [(spec.model_weights_sha256 == bases[0], f"the reference {reference} serves weights "
                                                 f"{spec.model_weights_sha256}, not the base {bases[0]} the members' "
                                                 "merges record")]
        if by_content else
        [(names == [spec.model_path], f"the members' merges name the base directory {names}, not the reference "
                                      f"{reference}'s {spec.model_path!r}")],
        where,
    )
    return {
        "sha256": bases[0],
        "reference": {"label": reference, "model_path": spec.model_path,
                      "model_weights_sha256": spec.model_weights_sha256,
                      "checked": "content" if by_content else "directory_name"},
        "members": members,
    }


def yes_shift(reference: VerdictGrids, member: VerdictGrids, *, n_resamples: int, ci_level: float,
              seed: int) -> dict[str, Any]:
    """Return the descriptive yes-shift baseline of one member against the reference (:data:`YES_SHIFT_DEFINITION`)."""
    full, rule_max = reference.full.values, reference.rule_max.values
    inconsistent = (full != rule_max).astype(float)
    ids = reference.full.scenario_ids
    rate_reference = DrawGrid(reference.condition_id, "contradiction", ids, inconsistent).rate
    rate_member = DrawGrid(member.condition_id, "contradiction", member.full.scenario_ids,
                           (member.full.values != member.rule_max.values).astype(float)).rate
    yes_reference = float(reference.shares("full", YES_PHI).mean())
    yes_member = float(member.shares("full", YES_PHI).mean())
    record = {"descriptive": True, "yes_rate_reference": yes_reference, "yes_rate_member": yes_member,
              "rate_reference": rate_reference, "rate_member": rate_member}
    reasons = [reason for ok, reason in (
        (yes_member > yes_reference, "the member's full-set yes share is not above the reference's"),
        (rate_member < rate_reference, "the member is not less inconsistent than the reference"),
    ) if not ok]
    if reasons:
        return {**record, "applicable": False, "reason": "; ".join(reasons), "p": None, "shifted_rate": None,
                "share_of_gain_explained": None}
    p = (yes_member - yes_reference) / (1.0 - yes_reference)
    expected = np.where(full == YES_PHI, inconsistent, p * (rule_max != YES_PHI) + (1.0 - p) * inconsistent)
    grid = DrawGrid(reference.condition_id, YES_SHIFT_QUANTITY, ids, expected, reference.full.identity_id)
    summary = hierarchical_bootstrap({YES_SHIFT_QUANTITY: grid}, n_resamples=n_resamples, ci_level=ci_level,
                                     seed=seed)["conditions"][YES_SHIFT_QUANTITY]
    shifted = summary["estimate"]
    return {
        **record,
        "applicable": True,
        "p": p,
        "shifted_rate": {**{key: summary[key] for key in ("estimate", "ci_low", "ci_high", "sd", "interval_basis")},
                         "p_held_fixed": True},
        "share_of_gain_explained": (rate_reference - shifted) / (rate_reference - rate_member),
    }


def _declaring(spec: ConditionSpec, label: str, family: str, reference: str) -> ConditionSpec:
    extra = {**dict(spec.extra), "policy": label, "policy_family": family, "policy_reference": reference}
    return dataclasses.replace(spec, extra=tuple(extra.items()))


def _root_order(identity_id: str) -> tuple[str, ...]:
    """Return where a condition sorts among a root's: the order the joint bootstrap draws in."""
    return (PurePosixPath(identity_id) / CONDITION_FNAME).parts


def _base_line(reference: str, members: Sequence[str], weights: Mapping[str, Any]) -> str:
    base = weights["reference"]
    merges = ", ".join(f"{label} {weights['members'][label]['model_weights_sha256'][:12]}..." for label in members)
    head = (f"Each member serves a merged checkpoint named by its weights digest ({merges}), whose merge manifest -- "
            f"the one its draws recorded -- records base weights {weights['sha256'][:12]}...")
    if base["checked"] == "content":
        return f"{head}; the reference {reference} serves those weights, named by content."
    return (f"{head} from a directory named {base['model_path']}. The reference {reference} serves the upstream "
            f"directory {base['model_path']}, named by that directory alone (its condition identity carries no "
            "content digest): this readout checks the name, not that the reference's files are those weights.")


def _amendment_lines(roots: Mapping[str, Root]) -> list[str]:
    return [f"{uri} took a retry ladder, read at its campaign's frozen cap beside the other roots: "
            + amendment_sentence(root.payload["campaign_id"], amendment)
            for uri, root in roots.items() for amendment in root.payload.get(MANIFEST_AMENDMENTS_KEY) or ()]


def _family_lines(arms: Sequence[FamilyArm], reference: str, specs: Mapping[str, ConditionSpec],
                  roots: Mapping[str, Root], display: Mapping[str, str], weights: Mapping[str, Any],
                  engine: Mapping[str, Any], states: Mapping[str, CodeState],
                  rebases: Mapping[str, Mapping[str, Any]]) -> list[str]:
    members = [arm.label for arm in arms if arm.label != reference]
    root_of = {arm.label: arm.root for arm in arms}
    spec, first = specs[reference], next(iter(roots.values()))
    manifest, payload = frozen_settings(first.manifest), first.payload
    elsewhere = [label for label in members if root_of[label] != root_of[reference]]
    analysed = {uri: sorted(public_id_of(root, cid) for cid in root.payload["conditions"])
                for uri, root in roots.items()}
    joint = any(set(conditions) != {specs[arm.label].public_id for arm in arms} for conditions in analysed.values())
    bootstrap = payload["bootstrap"]
    lines = [
        f"The family is named on the command line: reference {reference} ({display[reference]}), members "
        + ", ".join(f"{label} ({display[label]})" for label in members)
        + ". Its arms come from these campaign roots: "
        + "; ".join(f"{uri} (campaign {root.payload['campaign_id']}, {describe_code_state(states[uri])}: "
                    + ", ".join(label for label in root_of if root_of[label] == uri) + ")"
                    for uri, root in roots.items())
        + ".",
        *[_rebase_line(uri, rebase) for uri, rebase in rebases.items()],
        f"Checked across the arms: one condition identity but the weights served and the policy labels (judge "
        f"{spec.model_name}, prompt {spec.prompt_style}, label rule {spec.label_rule}, {spec.forced_json_requirement} "
        f"validity with strict_think {'on' if spec.strict_think else 'off'}, temperature {spec.temperature}, truncation "
        f"{manifest.get('truncation')}, token cap {spec.max_new_tokens}, ruleset {spec.rule_set_name} "
        f"({spec.rule_set_sha256[:12]}...), evaluation mode {spec.evaluation_mode}, judge type {spec.judge_type}, rule "
        f"context {spec.individual_rule_context}, serving dtype {spec.serving_dtype}); one scenario manifest "
        f"({payload['scenario_manifest_sha256'][:12]}...) of {payload['n_scenarios']} scenarios; one design "
        f"({payload['n_decoder_draws']} decoder draws, {manifest.get('max_semantic_attempts')} semantic attempts, "
        f"{manifest.get('retry_scope')} retry, bootstrap {bootstrap['n_resamples']:,} at seed {bootstrap['seed']}, "
        f"randomization {payload['randomization_test']['n_permutations']:,} permutations); one engine "
        f"({engine['short']}, TP {engine['tp_size']}); every root complete and final; no two arms on one set of "
        "weights.",
        _base_line(reference, members, weights),
        *_amendment_lines(roots),
    ]
    if elsewhere:
        lines.append(f"The reference arm {reference} comes from {root_of[reference]}; paired with it by (scenario, "
                     f"decoder draw index) over the one scenario manifest, from another root: {', '.join(elsewhere)}.")
    lines.append(f"Holm runs within each family (flat inconsistency, the kappa difference) over the members named "
                 f"here ({', '.join(members)}): a family of {len(members)}, over which each member's Holm p is taken.")
    if joint:
        lines.append(
            "Each arm's flat-rate interval is the joint scenario-cluster bootstrap over the arms named here, as one "
            "root holding them would give it. Past the bootstrap's first chunk of replicates it depends on which "
            "conditions are analysed together, so it can differ in its last digits from the interval its own root's "
            "estimates give, which analysed that root's conditions together ("
            + "; ".join(f"{uri}: {', '.join(conditions)}" for uri, conditions in analysed.items())
            + "). The point estimates, and each contrast's estimate, interval and unadjusted p, do not depend on it."
        )
    lines.append("The yes-shift baseline beside each member is descriptive: no test, in no family; its interval "
                 "holds p at its point value.")
    return lines


def build_readout(
    arms: Sequence[FamilyArm],
    *,
    reference: str,
    rule_set_path: Path | str,
    labels: Optional[Mapping[str, str]] = None,
    parity: Optional[Mapping[str, Path | str | Sequence[Path | str]]] = None,
    merge_manifests: Optional[Mapping[str, str]] = None,
    allow_code_mismatch: bool = False,
) -> dict[str, Any]:
    """Return the readout of the family ``arms`` name (reference first): the arms checked, their rates, contrasts,
    yes-shift baselines, disclosure and provenance; ``parity`` and ``merge_manifests`` are per member."""
    where = "training family"
    labels, parity = dict(labels or {}), dict(parity or {})
    merge_manifests = dict(merge_manifests or {})
    _check_command_line(arms, reference, labels, merge_manifests, where)
    roots = open_roots({arm.root: arm.root for arm in arms}, where, allow_incomplete=False,
                          allow_stack_mismatch=False)
    specs = {arm.label: registered_condition(roots[arm.root], arm.condition_id, f"{where} arm {arm.label}")
             for arm in arms}
    # From here each arm is read under its registered id; the readout names it by its public id.
    arms = [dataclasses.replace(arm, condition_id=specs[arm.label].condition_id) for arm in arms]
    _check_declarations(specs, reference, where)
    _check_identities(specs, reference, where)
    check_design(roots, where)
    states = _code_states(roots, allow_code_mismatch, where)
    rebases = {uri: rebase for uri, root in roots.items()
               if (rebase := _rebase(root, {specs[arm.label].public_id: arm.condition_id
                                            for arm in arms if arm.root == uri})) is not None}
    first = next(iter(roots.values()))
    n_draws = int(first.payload["n_decoder_draws"])
    dirs = {arm.label: roots[arm.root].path / arm.condition_id for arm in arms}
    records = {label: draw_provenances(directory, n_draws) for label, directory in dirs.items()}
    stacks = {arm.label: arm_stack(roots[arm.root], specs[arm.label], where) for arm in arms}
    engine = check_engine(stacks, records, where)
    weights = _check_weights(specs, records, reference, where, merge_manifests)

    # Laid out and analysed as one root holding the arms would be.
    settings = rp.bootstrap_settings(first.payload)
    bootstrap = {key: settings[key] for key in ("n_resamples", "ci_level", "seed")}
    family_name = "+".join(arm.label for arm in arms)
    ordered = sorted(arms, key=lambda arm: _root_order(specs[arm.label].identity_id))
    declaring = {arm.condition_id: _declaring(specs[arm.label], arm.label, family_name, reference) for arm in ordered}
    cells, grids = {}, {}
    for arm in ordered:
        cells[arm.condition_id], grids[arm.condition_id] = arm_outcomes(
            roots[arm.root], specs[arm.label], n_draws, f"{where} arm {arm.label}")
    rates = hierarchical_bootstrap(grids, **bootstrap)
    signatures = {arm.condition_id: stacks[arm.label]["signatures"] for arm in arms}
    groups = policy_contrasts(
        declaring, grids, campaign_id=where, allow_incomplete=False, stacks=signatures, excluded={},
        statistics={arm.condition_id: roots[arm.root].payload["conditions"][arm.condition_id]["attempt_statistics"]
                    for arm in arms},
        **settings,
    )
    policy_discrimination(
        groups, declaring, grids, cells, condition_dirs={arm.condition_id: dirs[arm.label] for arm in arms},
        campaign_ids={arm.condition_id: roots[arm.root].payload["campaign_id"] for arm in arms}, n_draws=n_draws,
        rule_set_path=Path(rule_set_path), stacks=signatures,
        allow_stack_mismatch=False, **settings,
    )
    ((name, family),) = policy_families(declaring).items()
    rows = {cid: {**rates["conditions"][cid], "condition_fingerprint": specs[label].fingerprint}
            for label, cid in family.policies.items()}
    scenarios = ScenarioManifest.load(first.path / SCENARIO_MANIFEST_FNAME)
    strict_think = bool(first.manifest.get("strict_think", False))
    result, _ = family_golden(
        rows, groups[name], family, {arm.condition_id: specs[arm.label] for arm in arms},
        {arm.condition_id: dirs[arm.label] for arm in arms}, scenarios, n_draws=n_draws, strict_think=strict_think,
        parity=parity, labels=labels, allow_incomplete=False, where=where,
    )
    display = {label: arm["label"] for label, arm in result["arms"].items()}
    result["disclosure"]["lines"] = _family_lines(
        arms, reference, specs, roots, display, weights, engine, states, rebases,
    ) + result["disclosure"]["lines"]
    verdicts = {cid: VerdictGrids.from_cells(cid, cells[cid], scenario_ids=scenarios.scenario_ids, n_draws=n_draws,
                                             identity_id=grids[cid].identity_id)
                for cid in cells}
    # The yes-shift reads inconsistency as phi(F) != M; that must equal each arm's stored contradiction.
    _require(
        [
            (np.array_equal((verdicts[cid].full.values != verdicts[cid].rule_max.values).astype(float),
                            grids[cid].values),
             f"arm {label}'s stored contradiction is not phi(F) != M in every cell: the yes-shift baseline would not "
             "be read against its flat rate")
            for label, cid in family.policies.items()
        ],
        where,
    )
    result.update({
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "yes_shift_definition": YES_SHIFT_DEFINITION,
        "yes_shift": {
            label: yes_shift(verdicts[family.policies[reference]], verdicts[cid], **bootstrap)
            for label, cid in family.policies.items() if label != reference
        },
    })
    result["provenance"] = {
        "complete": True,
        "family": {"declared_on": "the command line", "labels": [arm.label for arm in arms], "reference": reference,
                   "members": [arm.label for arm in arms if arm.label != reference],
                   "holm_family_size": len(arms) - 1},
        "arms": {
            arm.label: {"root": arm.root, "campaign_id": roots[arm.root].payload["campaign_id"],
                        "condition_id": specs[arm.label].public_id,
                        "condition_fingerprint": specs[arm.label].fingerprint,
                        "model_path": specs[arm.label].model_path,
                        "model_weights_sha256": specs[arm.label].model_weights_sha256,
                        "declared": {key: dict(specs[arm.label].extra).get(key)
                                     for key in ("policy", "policy_family", "policy_reference")},
                        "stack": stacks[arm.label]["description"]}
            for arm in arms
        },
        "campaigns": [
            {**estimates_source(root.payload), "root": uri,
             **dict(zip(("code_revision", "code_dirty", "code_state_sha256"), states[uri])),
             "conditions": [specs[arm.label].public_id for arm in arms if arm.root == uri],
             **({"rebase": rebases[uri]} if uri in rebases else {})}
            for uri, root in roots.items()
        ],
        "allow_code_mismatch": allow_code_mismatch,
        "engine": engine,
        "base_weights": weights,
        "rule_set": file_record(rule_set_path),
        **design_provenance(frozen_settings(first.manifest), first.payload, scenarios, n_draws=n_draws,
                            strict_think=strict_think, reference_spec=specs[reference]),
    }
    return result


def summary_markdown(readout: Mapping[str, Any]) -> str:
    """Return the readout's Markdown summary: arms, contrasts, yes-shift baselines and disclosure."""
    family = readout["provenance"]["family"]
    arms = readout["arms"]
    lines = [
        "# Training readout over a named policy family",
        "",
        f"Reference {arms[family['reference']]['label']}; members "
        + ", ".join(f"{arms[label]['label']} ({label})" for label in family["members"])
        + f". Holm within each family over {family['holm_family_size']} member(s).",
        "",
        "## Arms",
        "",
        "| Arm | condition | weights served | flat [95% CI] | kappa [95% CI] | yes rate | always-yes floor | M = yes | "
        "M != yes | complete maps | first-attempt share |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in readout["policies"]:
        arm, served = arms[label], readout["provenance"]["arms"][label]
        strata = arm["m_stratified"]
        same_call = arm.get("same_call") or {}
        maps = f"{same_call['n_complete_maps']}/{same_call['n_outputs']}" if same_call else "-"
        digest = served["model_weights_sha256"]
        weights = served["model_path"] + (f" ({digest[:12]}...)" if digest else " (by directory)")
        lines.append(
            f"| {arm['label']} | {arm['condition_id']} | {weights} | {format_interval(arm['flat'])} | "
            f"{format_interval(arm['kappa'], True)} | "
            f"{arm['yes_rate']['estimate']:.4f} | {arm['always_yes_floor']['estimate']:.4f} | "
            f"{format_interval(strata['M=yes'])} (n={strata['M=yes']['n_draws']}) | "
            f"{format_interval(strata['M!=yes'])} (n={strata['M!=yes']['n_draws']}) | {maps} | "
            f"{arm['first_attempt_acceptance_share']:.4f} |"
        )
    lines += [
        "",
        "## Contrasts (reference against each member)",
        "",
        "| Member | flat: reference - member | p | p (Holm) | kappa: member - reference | p | p (Holm) | claim | "
        "harm flag |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label in family["members"]:
        comparison = readout["comparisons"][label]
        flat, kappa = comparison["flat"], comparison["kappa"]
        lines.append(
            f"| {arms[label]['label']} | {format_interval(flat, True)} | {flat['p_value']:.4g} | {flat['p_holm']:.4g}"
            f"{' *' if flat['significant_holm'] else ''} | {format_interval(kappa, True)} | {kappa['p_value']:.4g} | "
            f"{kappa['p_holm']:.4g}{' *' if kappa['significant_holm'] else ''} | {comparison['claim']} | "
            f"{'FLAGGED' if comparison['harm_flag']['flagged'] else 'no'} |"
        )
    lines += [
        "",
        "## Yes-shift baseline (descriptive, no test)",
        "",
        readout["yes_shift_definition"],
        "",
        "| Member | yes share: reference -> member | p | shifted reference rate [95% CI, p fixed] | member rate | "
        "share of the gain explained |",
        "|---|---|---|---|---|---|",
    ]
    for label in family["members"]:
        shift = readout["yes_shift"][label]
        yes = f"{shift['yes_rate_reference']:.4f} -> {shift['yes_rate_member']:.4f}"
        if shift["applicable"]:
            lines.append(f"| {arms[label]['label']} | {yes} | {shift['p']:.4f} | {format_interval(shift['shifted_rate'])} | "
                         f"{shift['rate_member']:.4f} | {shift['share_of_gain_explained']:.4f} |")
        else:
            lines.append(f"| {arms[label]['label']} | {yes} | - | not applicable: {shift['reason']} | "
                         f"{shift['rate_member']:.4f} | - |")
    lines += ["", f"## Disclosure ({readout['disclosure']['regime']})", ""]
    lines += [f"- {line}" for line in readout["disclosure"]["lines"]]
    return "\n".join(lines) + "\n"
