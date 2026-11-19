from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ruleforge.evaluate.amendment import (
    AMENDED_FIELD,
    APPENDED_FILES,
    FIXED_FILES,
    AmendedBundle,
    AmendmentError,
    CapAmendment,
    bundle_prompt_digest,
    cap_amendment,
    check_seeds,
    component_prompts,
    file_snapshot,
)
from ruleforge.evaluate.campaign import (
    REBASE_FNAME,
    SCENARIO_MANIFEST_FNAME,
    carried_exhausted,
    load_conditions,
)
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    DRAW_PROVENANCE_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    ConditionSpec,
    differing_manifest_fields,
    draw_dirname,
    load_sampling_manifest,
    sampling_config_from_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.digests import canonical_json
from ruleforge.core.files import write_json_sidecar
from ruleforge.evaluate.ledger import DrawLedger, LedgerState, raw_row_key
from ruleforge.core.provenance import code_state as current_code_state
from ruleforge.evaluate.runner import STATE_COMPLETE, STATE_INCOMPLETE_EXHAUSTED, utc_now
from ruleforge.core.artifacts import prompt_sha256
from ruleforge.evaluate.sharding import (
    CodeState,
    ShardAssignment,
    campaign_lock,
    code_state_key,
    describe_code_state,
    recorded_code_states,
    shard_records,
    shard_root,
    shards_root,
)
from ruleforge.core.artifacts import compute_scenario_id, iter_jsonl_records
from ruleforge.core.resume import count_complete_jsonl_records
from ruleforge.rulesets.levels import RuleLevelsError, load_rule_levels
from tests.support.prompts import prompt_renderer, rule_sets_by_digest

#: A bundle: (condition id, decoder draw, scenario id).
Bundle = tuple[str, int, str]


def draw_snapshot(draw_dir: Path) -> dict[str, dict[str, Any]]:
    """Return the files of an amended draw a resume may append to or must leave alone, as they are now."""
    return {name: file_snapshot(Path(draw_dir) / name) for name in (*APPENDED_FILES, *FIXED_FILES)}


@dataclass(frozen=True)
class Root:
    """One root of a campaign: the campaign root itself (``shard`` None), or one shard root of one condition."""

    path: Path
    manifest: Mapping[str, Any]
    shard: Optional[str] = None
    condition_id: Optional[str] = None
    assignment: Optional[ShardAssignment] = None

    @property
    def label(self) -> str:
        return "the campaign root" if self.shard is None else f"{self.condition_id}/{self.shard}"


def campaign_roots(campaign_root: Path | str) -> list[Root]:
    """The campaign root (when it holds a sampling manifest) and every shard root beside it, in path order."""
    root = Path(campaign_root)
    roots = [Root(root, load_sampling_manifest(root / SAMPLING_MANIFEST_FNAME))] if (
        root / SAMPLING_MANIFEST_FNAME).is_file() else []
    for record_path in shard_records(root):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assignment = ShardAssignment.from_record(record)
        condition_id = str(record.get("condition_id"))
        if record_path.parent != shard_root(root, condition_id, assignment.index, assignment.count):
            raise AmendmentError(f"{record_path} records shard {assignment.label} of {condition_id}, not its directory")
        roots.append(Root(record_path.parent, load_sampling_manifest(record_path.parent / SAMPLING_MANIFEST_FNAME),
                          assignment.label, condition_id, assignment))
    if not roots:
        raise AmendmentError(f"{root} holds no campaign: no sampling manifest, and no shard root beside it")
    return roots


@dataclass
class _Survey:
    campaign_id: str
    old_cap: int
    generation_seed_base: int
    scenario_manifest_sha256: str
    conditions: dict[str, ConditionSpec] = field(default_factory=dict)
    accepted: dict[Bundle, int] = field(default_factory=dict)
    exhausted: dict[Bundle, list[tuple[Root, LedgerState]]] = field(default_factory=dict)
    code_states: set[CodeState] = field(default_factory=set)
    draws_read: int = 0
    held: set[Bundle] = field(default_factory=set)
    #: Carried-over bundles named for the raise (``--raise-carried``: a kept scenario's ladder).
    raised: set[Bundle] = field(default_factory=set)


def _whole_rows(path: Path) -> bool:
    data = path.read_bytes()
    return (not data or data.endswith(b"\n")) and count_complete_jsonl_records(path) == data.count(b"\n")


def _survey(roots: Sequence[Root], named: set[Bundle], raise_carried: bool = False) -> _Survey:
    """Read every finished draw of every root."""
    first = roots[0].manifest
    for root in roots[1:]:
        differing = differing_manifest_fields(first, root.manifest)
        if differing:
            raise AmendmentError(f"{root.label}'s sampling manifest differs from {roots[0].label}'s ({', '.join(differing)})")
    config = sampling_config_from_manifest(first)
    old_cap, draws = config.max_semantic_attempts, range(config.num_decoder_draws)
    ids = ScenarioManifest.load(roots[0].path / SCENARIO_MANIFEST_FNAME).scenario_ids
    survey = _Survey(campaign_id=str(first["campaign_id"]), old_cap=old_cap,
                     generation_seed_base=config.generation_seed_base,
                     scenario_manifest_sha256=str(first["scenario_manifest_sha256"]))
    covered: dict[str, set[tuple[int, str]]] = {}
    refusals: list[str] = []
    for root in roots:
        if ScenarioManifest.load(root.path / SCENARIO_MANIFEST_FNAME).manifest_sha256 != survey.scenario_manifest_sha256:
            raise AmendmentError(f"{root.label} froze another population than its sampling manifest names")
        for condition_id, spec in load_conditions(root.path).items():
            if survey.conditions.setdefault(condition_id, spec).fingerprint != spec.fingerprint:
                raise AmendmentError(f"{root.label} registers {condition_id} with another identity than another "
                                     "root does")
            survey.code_states.update(recorded_code_states(root.path, condition_id))
            for draw in (draws if root.assignment is None else root.assignment.draws):
                draw_dir = root.path / condition_id / draw_dirname(draw)
                if root.assignment is None and not draw_dir.is_dir():
                    continue  # a draw its shard roots hold, or none: the coverage says which
                scope = ids if root.assignment is None else root.assignment.draw_shard(draw).scenario_ids(
                    ids, config.scenarios_per_request_block)
                refusals += _read_draw(survey, root, condition_id, draw, draw_dir, scope)
                covered.setdefault(condition_id, set()).update((draw, sid) for sid in scope)
                survey.draws_read += 1
    for condition_id in survey.conditions:
        missing = len(draws) * len(ids) - len(covered.get(condition_id, ()))
        if missing:
            refusals.append(f"{condition_id}: {missing} (scenario, draw) bundle(s) are in no finished draw of any root")
    exhausted = set(survey.exhausted)
    carried = next((carried_exhausted(root.path) for root in roots if root.shard is None), set())
    survey.held = (exhausted & carried) - named
    survey.raised = named & carried
    refusals += [f"{_describe(bundle)} reached the old cap ({old_cap} attempts) but is not named: name every bundle "
                 "the cap is raised for" for bundle in sorted(exhausted - named - carried)]
    if not raise_carried:
        refusals += [f"{_describe(bundle)} is a retained scenario a rebase carried over exhausted ({REBASE_FNAME} "
                     "carried_exhausted): it keeps the old cap, so name the added scenarios' bundles "
                     "alone -- or name it for the raise (--raise-carried: a kept scenario's ladder)"
                     for bundle in sorted(named & carried)]
    refusals += [f"{_describe(bundle)} is one {REBASE_FNAME} records carried over exhausted, but no finished draw "
                 "holds it exhausted at the old cap: the record does not describe the root" for bundle in
                 sorted(carried - exhausted)]
    refusals += [f"{_describe(bundle)} is named but not exhausted (" + (
        f"accepted at attempt {survey.accepted[bundle]}" if bundle in survey.accepted else "in no finished draw") + ")"
        for bundle in sorted(named - exhausted)]
    refusals += [f"{_describe(bundle)} is held by {len(held)} roots ({', '.join(root.label for root, _ in held)}): "
                 "a merged draw beside its shard's -- move the campaign root's draw aside (it is the shards')"
                 for bundle, held in sorted(survey.exhausted.items()) if len(held) > 1]
    refusals += [f"the roots disagree on {_describe(bundle)}: exhausted in "
                 f"{', '.join(root.label for root, _ in survey.exhausted[bundle])}, accepted at attempt "
                 f"{survey.accepted[bundle]} in another" for bundle in sorted(exhausted & set(survey.accepted))]
    refusals += _unnamed_draws(survey, named & exhausted)
    if refusals:
        raise AmendmentError("the amendment would change more than the named bundles, or cannot: " + "; ".join(refusals))
    return survey


def _unnamed_draws(survey: _Survey, named: set[Bundle]) -> list[str]:
    held: dict[str, tuple[Root, set[int]]] = {}
    for condition_id, draw, scenario_id in sorted(named):
        for root, _ in survey.exhausted[(condition_id, draw, scenario_id)]:
            if root.assignment is not None:
                held.setdefault(root.label, (root, set()))[1].add(draw)
    return [f"{label} holds {[draw_dirname(draw) for draw in sorted(set(root.assignment.draws) - draws)]} beside "
            f"its named bundles' {[draw_dirname(draw) for draw in sorted(draws)]}: its resume would run the "
            "amendment's code state in every draw of the shard, which the amendment allows only in the draws it names"
            for label, (root, draws) in sorted(held.items()) if set(root.assignment.draws) - draws]


def _describe(bundle: Bundle) -> str:
    condition_id, draw, scenario_id = bundle
    return f"{condition_id} {draw_dirname(draw)} scenario {scenario_id[:12]}"


def _read_draw(
    survey: _Survey, root: Root, condition_id: str, draw: int, draw_dir: Path, scope: Sequence[str]
) -> list[str]:
    where = f"{root.label} {condition_id} {draw_dirname(draw)}"
    if not (draw_dir / COMPLETION_FNAME).is_file():
        return [f"{where} has not run"]
    state = DrawLedger(draw_dir, campaign_id=survey.campaign_id, condition_id=condition_id, decoder_draw_id=draw).load()
    gate = json.loads((draw_dir / COMPLETION_FNAME).read_text(encoding="utf-8"))
    refusals = []
    exhausted = []
    for sid in scope:
        bundle = (condition_id, draw, sid)
        if state.is_accepted(sid):
            attempt = state.accepted[sid].semantic_attempt_id
            if attempt >= survey.old_cap:
                refusals.append(f"{where}: {sid[:12]} was accepted at attempt {attempt}, beyond the old cap")
            if survey.accepted.setdefault(bundle, attempt) != attempt:
                refusals.append(f"{where}: the roots disagree on {_describe(bundle)}")
        elif state.next_attempt(sid) == survey.old_cap:
            exhausted.append(sid)
            survey.exhausted.setdefault(bundle, []).append((root, state))
        else:
            refusals.append(f"{where}: {sid[:12]} has used {state.next_attempt(sid)} of {survey.old_cap} attempts and "
                            "is not accepted: the draw has not finished")
    expected = (STATE_INCOMPLETE_EXHAUSTED, sorted(exhausted), 0) if exhausted else (STATE_COMPLETE, [], 0)
    if (gate.get("state"), gate.get("exhausted"), gate.get("pending")) != expected:
        refusals.append(f"{where}: its gate says {gate.get('state')!r} with exhausted {gate.get('exhausted')} and "
                        f"{gate.get('pending')} pending; its ledger says {expected[0]!r} with {len(exhausted)} exhausted")
    return refusals


def _check_bundle(
    survey: _Survey, bundle: Bundle, root: Root, state: LedgerState, *, rule_sets: Mapping[str, Path],
    levels: Optional[Path], rows: Mapping[str, Mapping[str, Any]],
) -> AmendedBundle:
    """A named bundle's draw whole and vouched; returns the bundle as the record holds it."""
    condition_id, draw, sid = bundle
    draw_dir = root.path / condition_id / draw_dirname(draw)
    where = f"{root.label} {condition_id} {draw_dirname(draw)}"
    torn = [name for name in APPENDED_FILES if not _whole_rows(draw_dir / name)]
    vouched = {(record.scenario_id, record.semantic_attempt_id, record.request_block_id) for record in state.records}
    ledger = DrawLedger(draw_dir, campaign_id=survey.campaign_id, condition_id=condition_id, decoder_draw_id=draw)
    orphans = 0 if torn else sum(1 for row in ledger.iter_raw() if raw_row_key(row) not in vouched)
    if torn or orphans:
        raise AmendmentError(f"{where}: torn or broken rows in {torn}, and {orphans} raw output row(s) no ledger row "
                             "vouches for, which a resume would repair: a resume must only append")
    spec = survey.conditions[condition_id]
    rules = rule_sets.get(spec.rule_set_sha256)
    if rules is None:
        raise AmendmentError(f"{where}: no ruleset given ({', '.join(sorted(map(str, rule_sets.values())))}) is the "
                             f"one {condition_id} judged ({spec.rule_set_name}): give it (--rules)")
    loaded = None
    if spec.rule_levels_sha256 is not None:
        try:
            loaded = None if levels is None else load_rule_levels(rules, levels)
        except RuleLevelsError as error:
            raise AmendmentError(f"{where}: {error}") from error
        if loaded is None or loaded.levels_file["sha256"] != spec.rule_levels_sha256:
            raise AmendmentError(f"{where}: {condition_id} judged by the levels file {spec.rule_levels_sha256[:12]}...; "
                                 "give that file (--levels)")
    render = prompt_renderer(spec, list(iter_jsonl_records(rules)), levels=loaded)
    records = [record for record in state.records if record.scenario_id == sid]
    prompts = component_prompts(records, sid, where)
    components = json.loads((draw_dir / DRAW_PROVENANCE_FNAME).read_text(encoding="utf-8"))["component_ids"]
    differing = sorted(cid for cid in components if prompt_sha256(render(rows[sid], cid)) != prompts.get(cid))
    if list(prompts) != components or differing:
        raise AmendmentError(
            f"{where}: scenario {sid[:12]}'s prompts as this code renders them are not the ones its attempts were "
            f"sent ({len(differing)} of {len(components)} components differ, first {differing[:3]})"
        )
    check_seeds(records, generation_seed_base=survey.generation_seed_base, where=where, identity_id=spec.identity_id)
    return AmendedBundle(condition_id=condition_id, decoder_draw_id=draw, scenario_id=sid, shard=root.shard,
                         attempts=survey.old_cap, prompts_sha256=bundle_prompt_digest(prompts),
                         files=draw_snapshot(draw_dir))


def _histograms(survey: _Survey) -> dict[str, dict[str, Any]]:
    """Per condition: its bundles, how many attempts each accepted one took, and the highest attempt accepted."""
    attempts: dict[str, Counter] = {condition_id: Counter() for condition_id in survey.conditions}
    for (condition_id, _, _), attempt in survey.accepted.items():
        attempts[condition_id][attempt] += 1
    return {
        condition_id: {
            "accepted": sum(counts.values()),
            "exhausted": sum(1 for bundle in survey.exhausted if bundle[0] == condition_id),
            "max_accepted_attempt": max(counts, default=None),
            "attempts_histogram": {str(attempt + 1): counts[attempt] for attempt in sorted(counts)},
        }
        for condition_id, counts in sorted(attempts.items())
    }


def _write(roots: Sequence[Root], record: Mapping[str, Any]) -> list[str]:
    named = {(bundle["condition_id"], bundle["shard"]) for bundle in record["bundles"]}
    written = []
    for root in sorted(roots, key=lambda root: (root.shard is not None, (root.condition_id, root.shard) not in named)):
        write_json_sidecar(root.path / SAMPLING_MANIFEST_FNAME,
                           {**root.manifest, record["field"]: record["new_cap"], MANIFEST_AMENDMENTS_KEY: [dict(record)]})
        written.append(str(root.path / SAMPLING_MANIFEST_FNAME))
    return written


def _at_old_cap(manifest: Mapping[str, Any], amendment: CapAmendment) -> dict[str, Any]:
    return {**{key: value for key, value in manifest.items() if key != MANIFEST_AMENDMENTS_KEY},
            AMENDED_FIELD: amendment.old_cap}


def _check_as_found(roots: Sequence[Root], amendment: CapAmendment, named: set[Bundle], raise_carried: bool) -> None:
    """An interrupted record's campaign, before the record is completed: still the one the record found."""
    carried = next((carried_exhausted(root.path) for root in roots if root.shard is None), set())
    try:
        survey = _survey([replace(root, manifest=_at_old_cap(root.manifest, amendment)) for root in roots], named,
                         raise_carried or bool(named & carried))
    except AmendmentError as error:
        problems = [str(error)]
    else:
        problems = [] if set(survey.conditions) == set(amendment.conditions) else [
            f"its roots register {sorted(survey.conditions)}, the record {list(amendment.conditions)}"]
        for bundle in amendment.bundles:
            [(root, _)] = survey.exhausted[(bundle.condition_id, bundle.decoder_draw_id, bundle.scenario_id)]
            draw_dir = root.path / bundle.condition_id / draw_dirname(bundle.decoder_draw_id)
            if root.shard != bundle.shard or draw_snapshot(draw_dir) != bundle.files:
                problems.append(f"{bundle.describe()}: its draw is not the bytes the record snapshotted")
    if problems:
        holding = ", ".join(root.label for root in roots if MANIFEST_AMENDMENTS_KEY in root.manifest)
        raise AmendmentError(
            "the record was interrupted, and the campaign is no longer the one it found: " + "; ".join(problems)
            + f". It cannot be completed: in the roots holding it ({holding}) restore {AMENDED_FIELD} "
            f"{amendment.old_cap} and remove {MANIFEST_AMENDMENTS_KEY}, then record the amendment again"
        )


def _recorded(roots: Sequence[Root], *, record: bool, new_cap: int, named: set[Bundle],
              raise_carried: bool = False) -> dict[str, Any]:
    """A campaign whose roots record an amendment already: left alone when every root holds the one record."""
    held = [root for root in roots if MANIFEST_AMENDMENTS_KEY in root.manifest]
    texts = {canonical_json(root.manifest[MANIFEST_AMENDMENTS_KEY]) for root in held}
    if len(texts) > 1:
        raise AmendmentError(f"the campaign's roots record different amendments ({', '.join(root.label for root in held)})")
    amendment = cap_amendment(held[0].manifest)
    recorded = {(bundle.condition_id, bundle.decoder_draw_id, bundle.scenario_id) for bundle in amendment.bundles}
    if (amendment.new_cap, recorded) != (int(new_cap), named):
        raise AmendmentError(
            f"the campaign records an amendment already (cap {amendment.old_cap} -> {amendment.new_cap} for "
            f"{', '.join(map(_describe, sorted(recorded)))}), and this request differs (cap {new_cap} for "
            f"{', '.join(map(_describe, sorted(named)))}): this build reads one cap amendment per campaign"
        )
    reference = _at_old_cap(held[0].manifest, amendment)
    lacking = [root for root in roots if root not in held]
    for root in lacking:
        differing = differing_manifest_fields(reference, root.manifest)
        if differing:
            raise AmendmentError(f"{root.label} lacks the amendment and differs from the amended roots ({differing})")
    if lacking:
        _check_as_found(roots, amendment, named, raise_carried)
    written = _write(lacking, amendment.to_dict()) if record and lacking else []
    return {"state": "completed" if written else ("interrupted" if lacking else "recorded already"),
            "record": amendment.to_dict(), "lacking": [root.label for root in lacking if not written],
            "written": written}


def amend_attempt_cap(
    campaign_root: Path | str,
    *,
    new_cap: int,
    bundles: Sequence[Bundle],
    decided_by: str,
    decided_on: str,
    reason: str,
    ledger: str,
    rules_path: Path | str | Sequence[Path | str],
    population_path: Path | str,
    levels_path: Optional[Path | str] = None,
    record: bool = False,
    code: Optional[Mapping[str, Any]] = None,
    now: Optional[str] = None,
    raise_carried: bool = False,
) -> dict[str, Any]:
    """Survey the campaign and check the named ``bundles``."""
    if record:
        from ruleforge.evaluate.api import refuse_protected

        refuse_protected(campaign_root, "the campaign root a cap amendment writes", campaign_root=True)
        refuse_protected(shards_root(campaign_root), "the shard roots a cap amendment writes", campaign_root=True)
    with campaign_lock(campaign_root) if record else nullcontext():
        roots = campaign_roots(campaign_root)
        named = {(str(condition_id), int(draw), str(scenario_id)) for condition_id, draw, scenario_id in bundles}
        if not named:
            raise AmendmentError("name the bundles the cap is raised for (--bundle CONDITION DRAW SCENARIO)")
        if any(MANIFEST_AMENDMENTS_KEY in root.manifest for root in roots):
            return {"campaign_root": str(campaign_root),
                    **_recorded(roots, record=record, new_cap=new_cap, named=named, raise_carried=raise_carried)}
        survey = _survey(roots, named, raise_carried)
        if new_cap <= survey.old_cap:
            raise AmendmentError(f"the new cap {new_cap} does not raise the campaign's {survey.old_cap}")
        code = dict(current_code_state() if code is None else code)
        if record and (code.get("code_dirty") is not False or not code.get("code_state_sha256")):
            raise AmendmentError(f"recording from {describe_code_state(code_state_key(code))} (dirty or unknown): "
                                 "the resume runs from the code that recorded the amendment, so record from a clean "
                                 "checkout")
        population = Path(population_path)
        if ScenarioManifest.build(population).manifest_sha256 != survey.scenario_manifest_sha256:
            raise AmendmentError(f"{population} is not the population the campaign froze")
        rows = {compute_scenario_id(row): row for row in iter_jsonl_records(population)}
        rule_sets = rule_sets_by_digest([rules_path] if isinstance(rules_path, (str, Path)) else rules_path)
        amended = tuple(
            _check_bundle(survey, bundle, root, state, rule_sets=rule_sets,
                          levels=None if levels_path is None else Path(levels_path), rows=rows)
            for bundle in sorted(named) for root, state in survey.exhausted[bundle]
        )
        histograms = _histograms(survey)
        checks = {
            "other_bundles": f"every other (condition, scenario, draw) of the {len(roots)} roots was accepted below "
                             f"the old cap ({survey.draws_read} draws read)",
            "max_accepted_attempt": {cid: entry["max_accepted_attempt"] for cid, entry in histograms.items()},
            "prompts": "the recording code renders every component of each amended bundle as every one of its "
                       "attempts was sent (the ledger's prompt digests)",
            "seeds": "each recorded seed is derive_generation_seed's for its campaign, condition, scenario, draw, "
                     "attempt and component; the cap is not among them",
            **({"held": f"{len(survey.held)} bundle(s) a rebase carried over exhausted keep the old cap: a job "
                        "resuming its draw gives it no attempt beyond it"} if survey.held else {}),
            **({"raised_carried": f"{len(survey.raised)} bundle(s) a rebase carried over exhausted are named for the "
                                  "raise (--raise-carried: a kept scenario's ladder), amended like an "
                                  "added scenario's and not held"} if survey.raised else {}),
        }
        amendment = CapAmendment(
            old_cap=survey.old_cap, new_cap=int(new_cap), recorded_at=now or utc_now(), decided_by=decided_by,
            decided_on=decided_on, reason=reason, ledger=ledger, code_state=code_state_key(code),
            prior_code_states=tuple(sorted(survey.code_states, key=str)), bundles=amended,
            conditions=tuple(sorted(survey.conditions)), checks=checks, held=tuple(sorted(survey.held)),
        )
        written = _write(roots, amendment.to_dict()) if record else []
        return {
            "campaign_root": str(campaign_root),
            "state": "recorded" if record else "dry run: nothing written",
            "roots": [root.label for root in roots],
            "old_cap": survey.old_cap,
            "new_cap": int(new_cap),
            "conditions": histograms,
            "amended": [bundle.describe() for bundle in amended],
            "held": [_describe(bundle) for bundle in sorted(survey.held)],
            # the carried-over bundles named for the raise, amended and not held
            "raised_carried": [_describe(bundle) for bundle in sorted(survey.raised)],
            "code_states": {"prior": [describe_code_state(state) for state in amendment.prior_code_states],
                            "resume": describe_code_state(amendment.code_state)},
            "record": amendment.to_dict(),
            "written": written,
        }

