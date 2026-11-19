from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ruleforge.evaluate.inference import load_condition_outcomes
from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.rulesets.rules import is_meta_rule_id
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.campaign import (
    CONDITION_FNAME,
    SCENARIO_MANIFEST_FNAME,
    campaign_completion,
    load_conditions,
)
from ruleforge.evaluate.identity import (
    CAMPAIGN_MODE_PILOT,
    EVALUATION_MODE_BANK,
    EVALUATION_MODE_FULL,
    ConditionSpec,
    check_exact_grid,
    draw_dirname,
    load_sampling_manifest,
)
from ruleforge.core import record_names
from ruleforge.core.errors import SamplingDesignError
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import sha256_file
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.ledger import AttemptRecord, ComponentRecord, DrawLedger, LedgerState, raw_output_sha256
from ruleforge.core.provenance import code_state
from ruleforge.evaluate.runner import STATE_COMPLETE, utc_now
from ruleforge.evaluate.stack import (
    CONDITION_SCOPE,
    LOCAL_MODEL_TYPE,
    EngineArgsIndex,
    Signature,
    condition_signatures,
    describe_mismatches,
    describe_signatures,
    shared_stack_conflicts,
)
from ruleforge.scenarios.subset import SubsetError, check_slices, read_slice_sidecar, sidecar_path
from ruleforge.prompts.parse import LABELS
from ruleforge.tasks.base import FORCED_JSON
from ruleforge.core.artifacts import (
    compute_scenario_id,
    load_jsonl_records,
    write_jsonl_records,
    write_text_atomically,
)

#: 1: every input is one campaign root; 2: an input spans several roots (a list of records).
ANCHORS_SCHEMA_VERSION = 1
ANCHORS_SCHEMA_VERSION_ROOTS = 2
#: 3: a staged set; its manifest holds `stages` and no `bank` or `top_up`.
ANCHORS_SCHEMA_VERSION_STAGED = 3
ANCHORS_SCHEMA_VERSIONS = (ANCHORS_SCHEMA_VERSION, ANCHORS_SCHEMA_VERSION_ROOTS, ANCHORS_SCHEMA_VERSION_STAGED)
ANCHORS_FNAME = "anchors.jsonl"
REPLAY_INDEX_FNAME = "replay_index.jsonl"
ANCHORS_MANIFEST_FNAME = "anchors_manifest.json"
ROUTE_SIDECAR_SUFFIX = ".route.json"

ESTIMATOR = "product_form_v1"
#: The judgments in tie-break order: the earlier wins, so a tie is never read as yes.
TIE_ORDER = ("no", "unsure", "yes")
#: Which draws a row's q̂ and replay rows come from.
SOURCE_BANK = "bank"
SOURCE_TOP_UP = "top_up"
MIXED_SELECTION_KIND = "mixed_screen_selection"
STAGE_CONFIRM = "confirm"
STAGE_YES_SIDE = "yes_side"
STAGE_CALIBRATION = "calibration"
CLASS_A = "a"
CLASS_B = "b"
CLASS_C = "c"
ANCHOR_CLASSES = (CLASS_A, CLASS_B, CLASS_C)

#: Which engine the anchors ran on: train on bf16, dev on the engine the checkpoints are served on.
SIDE_TRAIN = "train"
SIDE_DEV = "dev"
SIDES = (SIDE_TRAIN, SIDE_DEV)
#: ConditionSpec fields never compared against the reference base arm.
PROJECTION_EXCLUDED = ("condition_id", "evaluation_mode", "extra")
#: Excluded on the train side only: the train anchors run on the bf16 engine.
TRAIN_SIDE_EXCLUDED = ("serving_dtype",)
#: Not compared for a forced condition: its answer is JSON, which the label reader never reads.
FORCED_INERT = ("label_rule",)


class AnchorError(SamplingDesignError):
    """An anchor set cannot be built from these inputs."""


@dataclass(frozen=True)
class AnchorEstimate:
    """q̂ and its ingredients for one scenario, every probability an exact `Fraction`."""

    k: int
    #: rule id -> label -> p̂(label), in rule order.
    marginals: dict[str, dict[str, Fraction]]
    #: judgment -> q̂(judgment).
    q: dict[str, Fraction]
    #: judgment -> the share of the draws whose per-rule maximum is that judgment.
    q_freq: dict[str, Fraction]


def estimate(draws: Sequence[Mapping[str, str]], rule_ids: Sequence[str]) -> AnchorEstimate:
    """Return the product-form q̂ of one scenario from its per-rule draws; a draw short of a rule is refused."""
    expected = set(rule_ids)
    if not draws:
        raise AnchorError("an estimate needs at least one draw")
    for number, draw in enumerate(draws):
        if set(draw) != expected:
            raise AnchorError(
                f"draw {number} carries {len(draw)} rule label(s), not the {len(rule_ids)} rules "
                f"of the ruleset (missing {sorted(expected - set(draw))[:3]}, "
                f"unexpected {sorted(set(draw) - expected)[:3]})"
            )
        bad = sorted(rule for rule, label in draw.items() if label not in LABELS)
        if bad:
            raise AnchorError(f"draw {number} carries labels outside {LABELS} for {bad[:3]}")
    k = len(draws)
    marginals = {
        rule: {label: Fraction(sum(1 for draw in draws if draw[rule] == label), k) for label in TIE_ORDER}
        for rule in rule_ids
    }
    no_yes = Fraction(1)
    all_no = Fraction(1)
    for rule in rule_ids:
        no_yes *= 1 - marginals[rule]["yes"]
        all_no *= marginals[rule]["no"]
    q = {"no": all_no, "unsure": no_yes - all_no, "yes": 1 - no_yes}
    maxima = [max((draw[rule] for rule in rule_ids), key=PHI_MAP.__getitem__) for draw in draws]
    q_freq = {label: Fraction(maxima.count(label), k) for label in TIE_ORDER}
    return AnchorEstimate(k=k, marginals=marginals, q=q, q_freq=q_freq)


def mode_of(q: Mapping[str, Any]) -> str:
    """Return the judgment with the largest q̂; a tie goes to the earlier judgment of `TIE_ORDER`."""
    return max(TIE_ORDER, key=lambda verdict: (q[verdict], -TIE_ORDER.index(verdict)))


def routes_to_top_up(q: Mapping[str, Any]) -> bool:
    """Return whether a row is re-estimated from fresh draws: q̂_yes < 1."""
    return q["yes"] < 1


def anchor_class(q: Mapping[str, Any], full_set_labels: Sequence[str]) -> str:
    """Return the row's class: `a` (mode not yes, or q̂_yes < 1/2), `b` (a base full-set draw disagrees with the mode) or
    `c` (the rest)."""
    mode = mode_of(q)
    if mode != "yes" or q["yes"] < Fraction(1, 2):
        return CLASS_A
    if any(label != mode for label in full_set_labels):
        return CLASS_B
    return CLASS_C


def majority_labels(marginals: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Return each rule's most frequent label, ties by `TIE_ORDER`; for monitoring only."""
    return {rule: mode_of(shares) for rule, shares in marginals.items()}


def read_condition_file(path: Path | str) -> ConditionSpec:
    """Return a registered `condition.json`, its recorded fingerprint checked."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnchorError(f"cannot read the reference condition {source}: {error}") from error
    if not isinstance(payload, Mapping) or "fingerprint" not in payload:
        raise AnchorError(f"{source} is not a registered condition.json (no fingerprint)")
    data = dict(payload)
    data.pop("schema_version", None)
    recorded = data.pop("fingerprint")
    condition = ConditionSpec.from_dict(data)
    if condition.fingerprint != recorded:
        raise AnchorError(f"{source}: recorded fingerprint {recorded!r} does not match the condition it describes")
    return condition


def projection_excluded(side: str) -> tuple[str, ...]:
    """Return the ConditionSpec fields not compared for anchors of this side."""
    if side not in SIDES:
        raise AnchorError(f"side must be one of {SIDES}, got {side!r}")
    return PROJECTION_EXCLUDED + (TRAIN_SIDE_EXCLUDED if side == SIDE_TRAIN else ())


def condition_projection(condition: ConditionSpec, *, side: str) -> dict[str, Any]:
    """Return the identity payload an anchor condition shares with the reference base arm."""
    excluded = projection_excluded(side) + (FORCED_INERT if condition.contract == FORCED_JSON else ())
    return {key: value for key, value in condition.to_dict().items() if key not in excluded}


def check_reference(reference: ConditionSpec) -> None:
    """Refuse a reference that is not a base arm (upstream weights, no adapter)."""
    payload = reference.to_dict()
    if payload.get("adapter_path") is not None or payload.get("model_weights_sha256") is not None:
        raise AnchorError(
            f"reference {reference.condition_id!r} serves an adapter or derived weights; the anchors are the "
            "base model's, so the reference must be the base judge's arm"
        )


def check_condition_projection(condition: ConditionSpec, reference: ConditionSpec, *, side: str) -> None:
    """Refuse an anchor condition whose projection differs from the reference's, naming every field."""
    mine = condition_projection(condition, side=side)
    theirs = condition_projection(reference, side=side)
    differing = sorted(key for key in set(mine) | set(theirs) if mine.get(key) != theirs.get(key))
    if differing:
        detail = "; ".join(f"{key}: {mine.get(key)!r} vs {theirs.get(key)!r}" for key in differing)
        raise AnchorError(
            f"condition {condition.condition_id!r} is not the base judge's per-rule condition "
            f"{reference.condition_id!r} ({side} side): {detail}"
        )


def condition_key(condition: ConditionSpec) -> str:
    """Return the digest two conditions are held one condition by: the fingerprint, with a forced condition's inert
    fields at their ``ConditionSpec`` defaults."""
    if condition.contract != FORCED_JSON:
        return condition.fingerprint
    inert = {field.name: field.default for field in fields(ConditionSpec) if field.name in FORCED_INERT}
    return replace(condition, **inert).fingerprint


def recorded_condition_key(record: Mapping[str, Any]) -> Optional[str]:
    """Return the `condition_key` of the condition a manifest record names, or None when it names no file."""
    try:
        condition = read_condition_file(Path(record["root"]) / record["condition_id"] / CONDITION_FNAME)
    except (KeyError, TypeError, SamplingDesignError):
        return None
    return condition_key(condition) if condition.fingerprint == record.get("fingerprint") else None


def one_recorded_condition(records: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether manifest records of campaign roots name one condition."""
    if len({record["fingerprint"] for record in records}) <= 1:
        return True
    keys = {recorded_condition_key(record) for record in records}
    return None not in keys and len(keys) == 1


@dataclass(frozen=True)
class CampaignCondition:
    """One complete condition of a pilot campaign root, as the anchors read it."""

    root: Path
    manifest: dict[str, Any]
    scenario_manifest: ScenarioManifest
    condition: ConditionSpec

    @property
    def condition_id(self) -> str:
        return self.condition.condition_id

    @property
    def n_draws(self) -> int:
        return int(self.manifest["expected_decoder_draws"])

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return self.scenario_manifest.scenario_ids

    def outcome_cells(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Return the materialised `bundle_outcomes.jsonl` rows by (scenario, draw), checked against the manifests."""
        cells, _ = load_condition_outcomes(
            self.root / self.condition_id,
            expected_scenario_ids=self.scenario_ids,
            expected_draws=self.n_draws,
            scenario_manifest_sha256=self.scenario_manifest.manifest_sha256,
        )
        return cells

    def record(self) -> dict[str, Any]:
        """Return what the anchor manifest records about this input."""
        return {
            "root": str(self.root),
            "campaign_id": self.manifest["campaign_id"],
            "condition_id": self.condition_id,
            "fingerprint": self.condition.fingerprint,
            "evaluation_mode": self.condition.evaluation_mode,
            "scenario_manifest_sha256": self.scenario_manifest.manifest_sha256,
            "scenario_source_sha256": self.scenario_manifest.source_sha256,
            "n_scenarios": len(self.scenario_ids),
            "decoder_draws": self.n_draws,
            "generation_seed_base": self.manifest["generation_seed_base"],
        }


def named_condition(conditions: Mapping[str, ConditionSpec], condition_id: str) -> ConditionSpec:
    """Return the condition `condition_id` names among a root's registered `conditions`, by registered id or public id;
    none or two is refused."""
    named = [spec for spec in conditions.values() if condition_id in (spec.condition_id, spec.public_id)]
    if len(named) == 1:
        return named[0]
    return record_names.locate_condition(conditions, condition_id)


def open_condition(root: Path | str, condition_id: Optional[str], *, evaluation_mode: str) -> CampaignCondition:
    """Return, reading only, a condition of `root` in `evaluation_mode` whose every draw is complete; `condition_id`
    None picks the root's only condition in that mode."""
    # absolute, as the manifest and every replay row record it
    base = Path(root).absolute()
    manifest = load_sampling_manifest(base / SAMPLING_MANIFEST_FNAME)
    if manifest["campaign_mode"] != CAMPAIGN_MODE_PILOT:
        raise AnchorError(
            f"{base} is a {manifest['campaign_mode']!r} campaign; anchors come from pilot campaigns "
            "over the train and dev populations"
        )
    scenario_manifest = ScenarioManifest.load(base / SCENARIO_MANIFEST_FNAME)
    if scenario_manifest.manifest_sha256 != manifest["scenario_manifest_sha256"]:
        raise AnchorError(f"{base}: the scenario manifest is not the one the campaign manifest froze")
    conditions = load_conditions(base)
    if condition_id is None:
        matching = sorted(cid for cid, spec in conditions.items() if spec.evaluation_mode == evaluation_mode)
        if len(matching) != 1:
            raise AnchorError(
                f"{base} holds {len(matching)} {evaluation_mode!r} condition(s) {matching}; name the one to read"
            )
        condition_id = matching[0]
    try:
        condition = named_condition(conditions, condition_id)
    except SamplingDesignError as error:
        raise AnchorError(f"{base} registers no condition {condition_id!r}: {error}") from error
    condition_id = condition.condition_id
    if condition.evaluation_mode != evaluation_mode:
        raise AnchorError(
            f"{base}/{condition_id} is registered as {condition.evaluation_mode!r}, not {evaluation_mode!r}"
        )
    gate = campaign_completion(base, write=False)
    states = {draw: entry["state"] for draw, entry in gate["conditions"][condition_id]["draws"].items()}
    if any(state != STATE_COMPLETE for state in states.values()):
        raise AnchorError(
            f"{base}/{condition_id}: draws {states} are not all complete; an anchor is never "
            "estimated from an incomplete draw"
        )
    return CampaignCondition(base, manifest, scenario_manifest, condition)


@dataclass(frozen=True)
class CampaignInput:
    """One anchor input (the bank, the top-up or the full-set draws) over its campaign roots, which are one condition
    over pairwise disjoint scenarios."""

    name: str
    members: tuple[CampaignCondition, ...]

    @property
    def condition(self) -> ConditionSpec:
        return self.members[0].condition

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sid for member in self.members for sid in member.scenario_ids)

    @property
    def seed_bases(self) -> set[int]:
        return {int(member.manifest["generation_seed_base"]) for member in self.members}

    def roots(self) -> dict[str, str]:
        """Return `{scenario_id: the root that judged it}`."""
        return {sid: str(member.root) for member in self.members for sid in member.scenario_ids}

    def read_bank(self, rule_ids: Sequence[str], *, verify_raw: bool = True) -> dict[str, list[dict[str, RuleJudgment]]]:
        """Return `read_bank` over every root."""
        return {
            sid: draws
            for member in self.members
            for sid, draws in read_bank(member, rule_ids, verify_raw=verify_raw).items()
        }

    def read_full_set(self, rule_ids: Sequence[str]) -> dict[str, list[str]]:
        """Return `read_full_set` over every root."""
        return {sid: labels for member in self.members for sid, labels in read_full_set(member, rule_ids).items()}

    def record(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return what the anchor manifest records: one record for a single root, a list for several."""
        records = [member.record() for member in self.members]
        return records[0] if len(records) == 1 else records


def one_of(versions: Sequence[int]) -> str:
    """Return `1, 2 or 3`: the schema versions a reader takes, as its refusal names them."""
    names = [str(version) for version in versions]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"


def schema_version(*inputs: Optional[CampaignInput]) -> int:
    """Return the anchor set's `schema_version`: 1 unless an input spans several roots."""
    several = any(source is not None and len(source.members) > 1 for source in inputs)
    return ANCHORS_SCHEMA_VERSION_ROOTS if several else ANCHORS_SCHEMA_VERSION


def _root_paths(roots: Path | str | Sequence[Path | str]) -> list[Path]:
    paths = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(root) for root in roots]
    if not paths:
        raise AnchorError("no campaign root given")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise AnchorError(f"a campaign root is named twice among {[str(path) for path in paths]}")
    return paths


def open_input(
    name: str, roots: Path | str | Sequence[Path | str], condition_id: Optional[str], *, evaluation_mode: str
) -> CampaignInput:
    """Open input `name` over one root or several; a root named twice, roots that are not one condition, and roots
    sharing a scenario are refused."""
    members = tuple(open_condition(root, condition_id, evaluation_mode=evaluation_mode) for root in _root_paths(roots))
    first = members[0]
    seen: dict[str, Path] = {}
    for member in members:
        if condition_key(member.condition) != condition_key(first.condition):
            raise AnchorError(
                f"the {name} roots are not one condition: {member.root}/{member.condition_id} and "
                f"{first.root}/{first.condition_id} differ in fingerprint"
            )
        if member.n_draws != first.n_draws:
            raise AnchorError(
                f"the {name} roots drew different numbers of draws ({first.root}: {first.n_draws}, "
                f"{member.root}: {member.n_draws}); one input is one design"
            )
        shared = [sid for sid in member.scenario_ids if sid in seen]
        if shared:
            raise AnchorError(
                f"the {name} roots {seen[shared[0]]} and {member.root} share {len(shared)} scenario(s) (first "
                f"{shared[0][:12]}); the blocks of one input are disjoint"
            )
        seen.update((sid, member.root) for sid in member.scenario_ids)
    return CampaignInput(name, members)


@dataclass(frozen=True)
class RuleJudgment:
    """One accepted per-rule judgment and the ledger key of the raw output behind it."""

    label: str
    semantic_attempt_id: int
    request_block_id: str
    raw_output_sha256: str
    prompt_sha256: str


def _accepted_components(state: LedgerState, sid: str, where: str) -> list[tuple[AttemptRecord, ComponentRecord]]:
    record = state.accepted[sid]
    carried = [component.component_id for component in record.components]
    if len(carried) != len(set(carried)):
        raise AnchorError(f"{where}: an accepted bundle names a rule twice")
    return list(state.accepted_components(sid).values())


def _rule_judgments(
    components: Sequence[tuple[AttemptRecord, ComponentRecord]], rule_ids: Sequence[str], where: str
) -> dict[str, RuleJudgment]:
    carried = {component.component_id for _, component in components}
    if carried != set(rule_ids):
        raise AnchorError(
            f"{where}: the accepted bundle carries {len(carried)} rule label(s), not the "
            f"{len(rule_ids)} rules of the ruleset; an anchor draw needs every rule"
        )
    judgments: dict[str, RuleJudgment] = {}
    for record, component in components:
        digests = (component.raw_output_sha256, component.prompt_sha256)
        if component.judgment not in LABELS or not all(isinstance(digest, str) for digest in digests):
            raise AnchorError(
                f"{where}: {component.component_id} has no valid label, or no raw-output or prompt digest"
            )
        judgments[component.component_id] = RuleJudgment(
            label=component.judgment,
            semantic_attempt_id=record.semantic_attempt_id,
            request_block_id=record.request_block_id,
            raw_output_sha256=component.raw_output_sha256,
            prompt_sha256=component.prompt_sha256,
        )
    return {rule: judgments[rule] for rule in rule_ids}


def _verify_raw(ledger: DrawLedger, wanted: Mapping[tuple[str, int, str, str], str], where: str) -> None:
    found: set[tuple[str, int, str, str]] = set()
    for row in ledger.iter_raw():
        key = (
            str(row["scenario_id"]), int(row["semantic_attempt_id"]),
            str(row["request_block_id"]), str(row["component_id"]),
        )
        digest = wanted.get(key)
        if digest is None:
            continue
        if raw_output_sha256(row.get("raw_response")) != digest or row.get("raw_output_sha256") != digest:
            raise AnchorError(
                f"{where}: the raw output of {key[0][:12]}/{key[3]} does not hash to the digest its "
                "ledger row vouches for"
            )
        found.add(key)
    missing = sorted(set(wanted) - found)
    if missing:
        raise AnchorError(
            f"{where}: {len(missing)} accepted judgment(s) have no raw output on disk "
            f"(first {missing[0][0][:12]}/{missing[0][3]}); the replay rows could not be rebuilt"
        )


def read_bank(
    source: CampaignCondition, rule_ids: Sequence[str], *, verify_raw: bool = True
) -> dict[str, list[dict[str, RuleJudgment]]]:
    """Return `{scenario_id: [draw 0's {rule: judgment}, ...]}` of a complete bank condition, read from the ledgers and
    checked against the materialised outcomes; with `verify_raw`, every raw output is read back and checked against
    its digest."""
    cells = source.outcome_cells()
    bank: dict[str, list[dict[str, RuleJudgment]]] = {sid: [] for sid in source.scenario_ids}
    for draw in range(source.n_draws):
        draw_dir = source.root / source.condition_id / draw_dirname(draw)
        where = str(draw_dir)
        ledger = DrawLedger(
            draw_dir,
            campaign_id=source.manifest["campaign_id"],
            condition_id=source.condition_id,
            decoder_draw_id=draw,
        )
        state = ledger.load()
        grid = check_exact_grid(source.scenario_ids, 1, [(sid, 0) for sid in state.accepted])
        if not grid.ok:
            raise AnchorError(f"{where}: the ledger's accepted bundles are not the manifest's scenarios")
        wanted: dict[tuple[str, int, str, str], str] = {}
        for sid in source.scenario_ids:
            label = f"{where} {sid[:12]}"
            judgments = _rule_judgments(_accepted_components(state, sid, label), rule_ids, label)
            view = cells[(sid, draw)].get("rule_judgments")
            if view != {rule: judgment.label for rule, judgment in judgments.items()}:
                raise AnchorError(f"{where} {sid[:12]}: bundle_outcomes.jsonl disagrees with the ledger")
            bank[sid].append(judgments)
            for rule, judgment in judgments.items():
                wanted[(sid, judgment.semantic_attempt_id, judgment.request_block_id, rule)] = (
                    judgment.raw_output_sha256
                )
        if verify_raw:
            _verify_raw(ledger, wanted, where)
    return bank


def read_full_set(source: CampaignCondition, rule_ids: Sequence[str]) -> dict[str, list[str]]:
    """Return `{scenario_id: [draw 0's full-set judgment, ...]}` of a complete full-only condition."""
    cells = source.outcome_cells()
    verdicts: dict[str, list[str]] = {}
    for sid in source.scenario_ids:
        for draw in range(source.n_draws):
            row = cells[(sid, draw)]
            if row.get("full_set_judgment") not in LABELS or row.get("ruleset_rule_ids") != list(rule_ids):
                raise AnchorError(
                    f"{source.root}/{source.condition_id}: {sid[:12]}/draw {draw} is not a full-set judgment "
                    "over the ruleset's rules"
                )
            verdicts.setdefault(sid, []).append(row["full_set_judgment"])
    return verdicts


def labels_of(draws: Sequence[Mapping[str, RuleJudgment]]) -> list[dict[str, str]]:
    """Return a scenario's draws as `{rule: label}`, one per draw."""
    return [{rule: judgment.label for rule, judgment in draw.items()} for draw in draws]


def stack_check(inputs: Mapping[str, CampaignInput]) -> None:
    """Refuse inputs whose judgments come from different inference stacks."""
    index: EngineArgsIndex = {}
    per_root: dict[tuple[str, Path], list[Signature]] = {}
    stacks: dict[str, list[Signature]] = {}
    for label, source in inputs.items():
        pooled: list[Signature] = []
        for member in source.members:
            signatures = (
                condition_signatures(member.root / member.condition_id, args_by_digest=index)
                if member.condition.model_type == LOCAL_MODEL_TYPE else []
            )
            per_root[(label, member.root)] = signatures
            for signature in signatures:
                if signature not in pooled:
                    pooled.append(signature)
        stacks[label] = pooled
    conflicts = shared_stack_conflicts(stacks, scope="anchors", args_by_digest=index)
    if conflicts:
        by_root = []
        for record in conflicts:
            if record["scope"] != CONDITION_SCOPE:
                continue
            label = record["conditions"][0]
            groups: dict[str, list[str]] = {}
            for member in inputs[label].members:
                groups.setdefault(describe_signatures(per_root[(label, member.root)]), []).append(str(member.root))
            by_root.append(f"{label} roots: " + "; ".join(
                f"{', '.join(roots)} on {description}" for description, roots in groups.items()
            ))
        detail = f" [{' | '.join(by_root)}]" if by_root else ""
        raise AnchorError(
            f"the anchor inputs' judgments come from different inference stacks ({describe_mismatches(conflicts)})"
            f"{detail}; one anchor set is estimated on one engine"
        )


@dataclass(frozen=True)
class Ruleset:
    """The ruleset every anchor condition must have judged, by digest."""

    path: Path
    sha256: str
    rule_ids: tuple[str, ...]

    @classmethod
    def load(cls, path: Path | str) -> "Ruleset":
        source = Path(path)
        rule_ids = tuple(
            str(rule["id"]).strip() for rule in load_jsonl_records(source) if not is_meta_rule_id(str(rule.get("id", "")))
        )
        if not rule_ids or len(set(rule_ids)) != len(rule_ids):
            raise AnchorError(f"{source} holds no non-meta rule, or names one twice")
        return cls(source, sha256_file(source), rule_ids)

    def check(self, source: CampaignCondition) -> None:
        if source.condition.rule_set_sha256 != self.sha256:
            raise AnchorError(
                f"{source.root}/{source.condition_id} judged ruleset {source.condition.rule_set_sha256[:12]}..., "
                f"not {self.path.name} ({self.sha256[:12]}...)"
            )

    def record(self) -> dict[str, Any]:
        return {"path": str(self.path.absolute()), "sha256": self.sha256, "n_rules": len(self.rule_ids)}


def _checked(source: CampaignInput, reference: ConditionSpec, rules: Ruleset, side: str) -> CampaignInput:
    for member in source.members:
        check_condition_projection(member.condition, reference, side=side)
        rules.check(member)
    return source


def population_coverage(
    source: CampaignInput,
    population: Optional[Path | str],
    blocks: Optional[Path | str],
    *,
    flags: tuple[str, str] = ("--population", "--blocks"),
) -> tuple[list[str], Optional[dict[str, Any]]]:
    """Return the scenario order of the anchors and the record of the population the input covers (None when none is
    declared); with `blocks`, the roots' sidecars must tile `population`."""
    if blocks is None:
        if len(source.members) > 1:
            raise AnchorError(
                f"{len(source.members)} {source.name} roots need the population they partition and the directory of "
                f"its blocks ({', '.join(flags)}): each root's block sidecar proves it a slice of the population, "
                "and together they must tile it"
            )
        if population is None:
            return list(source.scenario_ids), None
        member = source.members[0]
        population_path = Path(population)
        if sha256_file(population_path) != member.scenario_manifest.source_sha256:
            raise AnchorError(f"{population_path} is not the population the {source.name} {member.root} froze")
        return list(source.scenario_ids), {"name": population_path.name, "sha256": sha256_file(population_path)}
    if population is None:
        raise AnchorError("--blocks names the blocks of a population; declare the population too (--population)")
    population_path = Path(population)
    sidecars = []
    try:
        for member in source.members:
            sidecar = read_slice_sidecar(sidecar_path(Path(blocks) / member.scenario_manifest.source_name))
            if sidecar["output_sha256"] != member.scenario_manifest.source_sha256:
                raise AnchorError(
                    f"{member.root} froze {member.scenario_manifest.source_name} with digest "
                    f"{member.scenario_manifest.source_sha256[:12]}..., not the block its sidecar in {blocks} describes"
                )
            sidecars.append(sidecar)
        order = check_slices(population_path, sidecars)
    except SubsetError as error:
        raise AnchorError(f"the {source.name} roots are not blocks of {population_path.name}: {error}") from error
    if set(order) != set(source.scenario_ids):
        raise AnchorError(
            f"the {source.name} roots froze {len(set(source.scenario_ids) - set(order))} scenario id(s) that are not "
            f"{population_path.name}'s, which has {len(set(order) - set(source.scenario_ids))} they lack"
        )
    record = {
        "name": population_path.name,
        "sha256": sha256_file(population_path),
        "blocks": [
            {"name": sidecar["output_name"], "row_range": sidecar["row_range"], "sha256": sidecar["output_sha256"]}
            for sidecar in sorted(sidecars, key=lambda sidecar: sidecar["row_range"])
        ],
    }
    return order, record


def routed_scenarios(bank: CampaignInput, rules: Ruleset) -> set[str]:
    """Return the bank's scenarios whose q̂_yes < 1, over every root."""
    judgments = bank.read_bank(rules.rule_ids, verify_raw=False)
    return {sid for sid, draws in judgments.items() if routes_to_top_up(estimate(labels_of(draws), rules.rule_ids).q)}


def route_top_up(
    *,
    bank: Path | str | Sequence[Path | str],
    reference: Path | str,
    rules: Path | str,
    side: str,
    population: Path | str,
    out: Path | str,
    bank_condition: Optional[str] = None,
    blocks: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Write the top-up campaign's population (the bank rows with q̂_yes < 1, copied verbatim from `population` in its
    order) with a `<out>.route.json` sidecar, writing nothing when no row routes."""
    reference_spec = read_condition_file(reference)
    check_reference(reference_spec)
    ruleset = Ruleset.load(rules)
    source = _checked(
        open_input("bank", bank, bank_condition, evaluation_mode=EVALUATION_MODE_BANK), reference_spec, ruleset, side
    )
    order, covered = population_coverage(source, population, blocks)
    stack_check({"bank": source})
    wanted = routed_scenarios(source, ruleset)
    summary: dict[str, Any] = {
        "bank": source.record(),
        "rule": "q_yes < 1 over the bank's draws: no rule labelled yes in every draw",
        "routed": len(wanted),
        "out": None,
    }
    if not wanted:
        return summary
    population_path = Path(population)
    lines = [line for line in population_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    kept = [line for line in lines if compute_scenario_id(json.loads(line)) in wanted]
    if len(kept) != len(wanted):
        raise AnchorError(
            f"{population_path.name} holds {len(kept)} of the {len(wanted)} routed scenario(s); the routed population "
            "would be partial"
        )
    text = "".join(line + "\n" for line in kept)
    out_path = Path(out)
    if out_path.exists() and out_path.read_bytes() != text.encode("utf-8"):
        raise AnchorError(f"{out_path} exists with different content; refusing to overwrite a routed population")
    write_text_atomically(out_path, text)
    summary["out"] = str(out_path.absolute())
    write_json_sidecar(
        Path(str(out_path) + ROUTE_SIDECAR_SUFFIX),
        {
            "schema_version": schema_version(source),
            **summary,
            "reference_fingerprint": reference_spec.fingerprint,
            "rules": ruleset.record(),
            "population": covered,
            "scenario_ids": [sid for sid in order if sid in wanted],
            "output_sha256": sha256_file(out_path),
        },
    )
    return summary


def _float_map(values: Mapping[str, Fraction]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def anchor_row(
    sid: str,
    estimate_: AnchorEstimate,
    *,
    topped_up: bool,
    full_set_labels: Sequence[str],
) -> dict[str, Any]:
    """Return one `anchors.jsonl` row."""
    return {
        "scenario_id": sid,
        "k": estimate_.k,
        "topped_up": topped_up,
        "q_yes": float(estimate_.q["yes"]),
        "q_unsure": float(estimate_.q["unsure"]),
        "q_no": float(estimate_.q["no"]),
        "q_freq": _float_map(estimate_.q_freq),
        "mode": mode_of(estimate_.q),
        "anchor_class": anchor_class(estimate_.q, full_set_labels),
        "full_set_labels": list(full_set_labels),
        "rule_marginals": {rule: _float_map(shares) for rule, shares in estimate_.marginals.items()},
        "base_rule_labels": majority_labels(estimate_.marginals),
        "base_yes_rules": [rule for rule, shares in estimate_.marginals.items() if shares["yes"] > 0],
    }


def replay_rows(
    sid: str, draws: Sequence[Mapping[str, RuleJudgment]], *, source: str, root: str
) -> list[dict[str, Any]]:
    """Return the replay index rows of one scenario: every rule of every draw behind its q̂, at weight 1/K."""
    weight = 1.0 / len(draws)
    rows = []
    for rule in draws[0]:
        for draw, judgments in enumerate(draws):
            judgment = judgments[rule]
            rows.append(
                {
                    "scenario_id": sid,
                    "rule_id": rule,
                    "source": source,
                    "root": root,
                    "decoder_draw_id": draw,
                    "semantic_attempt_id": judgment.semantic_attempt_id,
                    "request_block_id": judgment.request_block_id,
                    "raw_output_sha256": judgment.raw_output_sha256,
                    "prompt_sha256": judgment.prompt_sha256,
                    "label": judgment.label,
                    "weight": weight,
                }
            )
    return rows


def _count(values: Sequence[str], keys: Sequence[str]) -> dict[str, int]:
    return {key: sum(1 for value in values if value == key) for key in keys}


def build_anchor_set(
    *,
    bank: Path | str | Sequence[Path | str],
    full_set: Path | str | Sequence[Path | str],
    reference: Path | str,
    rules: Path | str,
    side: str,
    out: Path | str,
    top_up: Optional[Path | str | Sequence[Path | str]] = None,
    bank_condition: Optional[str] = None,
    top_up_condition: Optional[str] = None,
    full_set_condition: Optional[str] = None,
    allow_untopped: bool = False,
    population: Optional[Path | str] = None,
    blocks: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Build `anchors.jsonl`, `replay_index.jsonl` and their manifest under `out` and return the manifest, refusing a
    train-side build with routed rows and no top-up unless `allow_untopped`."""
    out_dir = Path(out)
    partial = out_dir.with_name(out_dir.name + ".partial")
    for path in (out_dir, partial):
        if path.exists():
            raise AnchorError(f"{path} exists; an anchor set is written once, to a new directory")
    reference_spec = read_condition_file(reference)
    check_reference(reference_spec)
    ruleset = Ruleset.load(rules)
    bank_source = _checked(
        open_input("bank", bank, bank_condition, evaluation_mode=EVALUATION_MODE_BANK), reference_spec, ruleset, side
    )
    full_source = _checked(
        open_input("full_set", full_set, full_set_condition, evaluation_mode=EVALUATION_MODE_FULL),
        reference_spec, ruleset, side,
    )
    order, covered = population_coverage(bank_source, population, blocks)
    if set(full_source.scenario_ids) != set(bank_source.scenario_ids):
        raise AnchorError("the full-set draws do not cover exactly the bank's scenarios")
    inputs = {"bank": bank_source, "full_set": full_source}
    top_source: Optional[CampaignInput] = None
    if top_up is not None:
        top_source = open_input("top_up", top_up, top_up_condition, evaluation_mode=EVALUATION_MODE_BANK)
        if condition_key(top_source.condition) != condition_key(bank_source.condition):
            raise AnchorError("the top-up condition is not the bank's condition (their fingerprints differ)")
        if top_source.seed_bases & bank_source.seed_bases:
            raise AnchorError("the top-up campaign must draw with another generation_seed_base than the bank")
        inputs["top_up"] = top_source
    stack_check(inputs)

    bank_draws = bank_source.read_bank(ruleset.rule_ids)
    estimates = {sid: estimate(labels_of(bank_draws[sid]), ruleset.rule_ids) for sid in order}
    routed = [sid for sid in order if routes_to_top_up(estimates[sid].q)]
    if side == SIDE_TRAIN and top_source is None and routed and not allow_untopped:
        raise AnchorError(
            f"{len(routed)} train row(s) route to the top-up (q_yes < 1) and no top-up campaign was given; "
            "the top-up re-estimates them from fresh draws. Run the top-up, or declare allow_untopped "
            "(--allow-untopped) to keep their bank estimates"
        )
    top_draws: dict[str, list[dict[str, RuleJudgment]]] = {}
    top_roots: dict[str, str] = {}
    if top_source is not None:
        if set(top_source.scenario_ids) != set(routed):
            raise AnchorError(
                f"the top-up campaign covers {len(top_source.scenario_ids)} scenario(s), not the {len(routed)} the "
                "bank routes (q_yes < 1); re-route the bank and run the top-up over that population"
            )
        top_draws = top_source.read_bank(ruleset.rule_ids)
        top_roots = top_source.roots()
    full_labels = full_source.read_full_set(ruleset.rule_ids)
    bank_roots = bank_source.roots()

    rows: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    for sid in order:
        topped_up = sid in top_draws
        draws = top_draws[sid] if topped_up else bank_draws[sid]
        row_estimate = estimate(labels_of(draws), ruleset.rule_ids) if topped_up else estimates[sid]
        rows.append(anchor_row(sid, row_estimate, topped_up=topped_up, full_set_labels=full_labels[sid]))
        replay.extend(replay_rows(
            sid, draws, source=SOURCE_TOP_UP if topped_up else SOURCE_BANK,
            root=top_roots[sid] if topped_up else bank_roots[sid],
        ))

    partial.mkdir(parents=True)
    write_jsonl_records(partial / ANCHORS_FNAME, rows)
    write_jsonl_records(partial / REPLAY_INDEX_FNAME, replay)
    manifest = {
        "schema_version": schema_version(bank_source, top_source, full_source),
        "estimator": ESTIMATOR,
        "tie_order": list(TIE_ORDER),
        "side": side,
        "allow_untopped": allow_untopped,
        "projection_excluded": list(projection_excluded(side)),
        "reference": _reference_record(reference, reference_spec),
        "rules": ruleset.record(),
        "bank": bank_source.record(),
        "top_up": None if top_source is None else top_source.record(),
        "full_set": full_source.record(),
        # a single root's manifest has no population key
        **({} if covered is None else {"population": covered}),
        "counts": {
            "scenarios": len(rows),
            "routed": len(routed),
            "topped_up": len(top_draws),
            "classes": _count([row["anchor_class"] for row in rows], ANCHOR_CLASSES),
            "modes": _count([row["mode"] for row in rows], TIE_ORDER),
            "replay_rows": len(replay),
        },
        "outputs": {
            name: {"sha256": sha256_file(partial / name), "rows": count}
            for name, count in ((ANCHORS_FNAME, len(rows)), (REPLAY_INDEX_FNAME, len(replay)))
        },
        **code_state(),
        "created_at": utc_now(),
    }
    write_json_sidecar(partial / ANCHORS_MANIFEST_FNAME, manifest)
    partial.rename(out_dir)
    return manifest


def _reference_record(reference: Path | str, spec: ConditionSpec) -> dict[str, Any]:
    return {
        "path": str(Path(reference).absolute()),
        "sha256": sha256_file(reference),
        "condition_id": spec.condition_id,
        "fingerprint": spec.fingerprint,
    }


@dataclass(frozen=True)
class AnchorSet:
    """A built anchor set, read back with its outputs' digests checked."""

    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    replay: list[dict[str, Any]]


def read_anchor_set(directory: Path | str) -> AnchorSet:
    """Read an anchor set, refusing one whose files are not the ones its manifest records."""
    base = Path(directory)
    manifest = json.loads((base / ANCHORS_MANIFEST_FNAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in ANCHORS_SCHEMA_VERSIONS:
        raise AnchorError(f"{base} is not a schema_version {one_of(ANCHORS_SCHEMA_VERSIONS)} anchor set")
    loaded = {}
    for name, recorded in manifest["outputs"].items():
        if sha256_file(base / name) != recorded["sha256"]:
            raise AnchorError(f"{base / name} is not the file its manifest records")
        loaded[name] = load_jsonl_records(base / name)
    return AnchorSet(manifest, loaded[ANCHORS_FNAME], loaded[REPLAY_INDEX_FNAME])


def input_records(manifest: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    """Return the campaign root records an anchor set read for input `name` (or a stage), one each; no top-up is `[]`.
    """
    stages = manifest.get("stages") or {}
    value = stages[name] if name in stages else manifest[name]
    if value is None:
        return []
    return [dict(record) for record in value] if isinstance(value, list) else [dict(value)]


def replay_source(manifest: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest's record of the root whose ledger holds a replay row's raw output."""
    matches = [record for record in input_records(manifest, row["source"]) if record["root"] == row["root"]]
    if not matches:
        raise AnchorError(f"the manifest names no {row['source']} root {row['root']!r} for this replay row")
    return matches[0]


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--bank", type=Path, nargs="+", required=True,
        help="the per-rule bank's campaign root, or one root per block of its population (with --blocks)",
    )
    parser.add_argument("--bank-condition", help="its individual_bank condition id (default: the only one)")
    parser.add_argument(
        "--blocks", type=Path,
        help="the directory of the blocks' sidecars (ruleforge scenarios partition): each bank root is one block of "
        "--population, and together they must tile it (required with several bank roots)",
    )
    parser.add_argument(
        "--reference", type=Path, required=True,
        help="the base judge's condition.json, which every anchor condition must match",
    )
    parser.add_argument("--rules", type=Path, required=True, help="the ruleset every condition judged")
    parser.add_argument(
        "--side", choices=SIDES, required=True,
        help="train (the bf16 engine: serving_dtype not compared) or dev (the serving engine: compared)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_anchors",
        description="Build anchor sets: the reward target and the KL anchor's replay index, from per-rule banks.")
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("route", help="write the top-up population: the bank rows with q_yes < 1")
    _add_common(route)
    route.add_argument(
        "--population", type=Path, required=True,
        help="the population file the bank froze, or, with --blocks, the one its blocks tile",
    )
    route.add_argument("--out", type=Path, required=True, help="the routed population (JSONL) to write")
    build = commands.add_parser("build", help="build the anchor set")
    _add_common(build)
    top = build.add_mutually_exclusive_group(required=True)
    top.add_argument(
        "--top-up", type=Path, nargs="+",
        help="the top-up campaign root (fresh draws of the routed rows), or one root per block of them",
    )
    top.add_argument("--no-top-up", action="store_true", help="no top-up: q-hat from the bank for every row (dev)")
    build.add_argument(
        "--allow-untopped", action="store_true",
        help="train side: build with --no-top-up even when rows route (recorded)",
    )
    build.add_argument("--top-up-condition", help="its individual_bank condition id (default: the only one)")
    build.add_argument(
        "--full-set", type=Path, nargs="+", required=True,
        help="the base full-set draws' campaign root, or one root per block",
    )
    build.add_argument("--full-set-condition", help="its full condition id (default: the only one)")
    build.add_argument(
        "--population", type=Path,
        help="the population the bank covers: the file a single root froze, or the one --blocks tile",
    )
    build.add_argument("--out", type=Path, required=True, help="the anchor set directory to create")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    common = dict(
        bank=args.bank, bank_condition=args.bank_condition, reference=args.reference, rules=args.rules,
        side=args.side, out=args.out, population=args.population, blocks=args.blocks,
    )
    if args.command == "route":
        summary = route_top_up(**common)
    else:
        manifest = build_anchor_set(
            top_up=args.top_up, top_up_condition=args.top_up_condition, full_set=args.full_set,
            full_set_condition=args.full_set_condition, allow_untopped=args.allow_untopped, **common,
        )
        summary = {"out": str(args.out), **manifest["counts"]}
    print(json.dumps(summary, indent=1))
    return 0


def build(
    *,
    bank: Path | str | Sequence[Path | str],
    full_set: Path | str | Sequence[Path | str],
    reference: Path | str,
    rules: Path | str,
    side: str,
    out: Path | str,
    top_up: Optional[Path | str | Sequence[Path | str]] = None,
    bank_condition: Optional[str] = None,
    top_up_condition: Optional[str] = None,
    full_set_condition: Optional[str] = None,
    allow_untopped: bool = False,
    population: Optional[Path | str] = None,
    blocks: Optional[Path | str] = None,
) -> Path:
    """Build an anchor set (`build_anchor_set`) and return its directory."""
    build_anchor_set(bank=bank, full_set=full_set, reference=reference, rules=rules, side=side, out=out,
                     top_up=top_up, bank_condition=bank_condition, top_up_condition=top_up_condition,
                     full_set_condition=full_set_condition, allow_untopped=allow_untopped, population=population,
                     blocks=blocks)
    return Path(out)


def route(
    *,
    bank: Path | str | Sequence[Path | str],
    reference: Path | str,
    rules: Path | str,
    side: str,
    population: Path | str,
    out: Path | str,
    bank_condition: Optional[str] = None,
    blocks: Optional[Path | str] = None,
) -> Path:
    """Write the top-up campaign's population (`route_top_up`) and return it."""
    route_top_up(bank=bank, reference=reference, rules=rules, side=side, population=population, out=out,
                 bank_condition=bank_condition, blocks=blocks)
    return Path(out)


if __name__ == "__main__":
    raise SystemExit(main())
