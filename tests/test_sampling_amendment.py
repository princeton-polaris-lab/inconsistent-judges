from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.evaluate import aggregate as aggregate_module
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate import runner as runner_module
from ruleforge.evaluate import shard_merge as merge_module
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.amendment import (
    AmendmentError,
    CapAmendment,
    cap_amendment,
    check_resumed_draw,
)
from ruleforge.core.record_names import identity_condition_id
from tests.support import cap_amendment as cap_module
from tests.support.cap_amendment import amend_attempt_cap, campaign_roots
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.campaign import load_conditions, preflight
from ruleforge.core.constants import PUBLISHED_COPY_FNAME
from ruleforge.evaluate.identity import (
    MANIFEST_AMENDMENTS_KEY,
    RUN_LOG_ROWS_KEY,
    RUNS_FNAME,
    PublishedCopyError,
    derive_generation_seed,
    differing_manifest_fields,
    draw_dirname,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.ledger import LEDGER_FNAME, RAW_OUTPUTS_FNAME
from ruleforge.core.provenance import SNAPSHOT_DIRNAME
from ruleforge.evaluate.identity import COMPLETION_FNAME
from ruleforge.evaluate.runner import REQUEST_BLOCKS_FNAME, STATE_COMPLETE, CampaignRunError
from ruleforge.evaluate.shard_merge import merge_condition_shards
from ruleforge.evaluate.sharding import ShardError, campaign_lock_path, shard_root
from ruleforge.judges.backend import GenerationOutcome
from ruleforge.judges.batch_api import BatchExecutionError
from tests.test_launch_campaign import repo  # noqa: F401 - a fixture
from tests.test_sampling_shards import (
    CONDITION,
    NOW,
    STACK,
    SeededBackend,
    assert_same_file,
    compose_cfg,
    edit_json,
    files_of,
    read_jsonl,
    shard_cfg,
    task_spec_for,
    write_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_CAP, NEW_CAP, ACCEPTED_AT = 4, 8, 6
OLD = {"code_revision": "a" * 40, "code_dirty": False, "code_state_sha256": "1" * 64}
NEW = {"code_revision": "b" * 40, "code_dirty": False, "code_state_sha256": "2" * 64}
#: The stubborn bundle: draw 1, scenario 7 -- request block 2 of draw 1, held by shard 2 of 3 alone.
TARGET_DRAW, TARGET_INDEX = 1, 7
DECISION = {"decided_by": "a maintainer", "decided_on": "2026-10-10", "reason": "retry the exhausted bundle",
            "ledger": "note 132"}


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    for module in (runner_module, campaign_module, merge_module, aggregate_module, cap_module):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    monkeypatch.setattr(runner_module, "_package_version", STACK.get)
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    use_code(monkeypatch, OLD)


def use_code(monkeypatch, state):
    """The code state every job (and its preflight) runs from."""
    monkeypatch.setattr(campaign_module, "code_state", lambda: dict(state))


def without_cap_amendments(patch) -> None:
    """The preflight as code without cap amendments runs it: no amendment read, manifests compared field by field."""

    def check(path, manifest):
        path = Path(path)
        differing = differing_manifest_fields(json.loads(path.read_text()), manifest) if path.exists() else []
        if differing:
            raise SamplingDesignError(f"{path} describes a different campaign ({', '.join(differing)} differ)")

    patch.setattr(campaign_module, "campaign_amendment", lambda campaign_root: None)
    patch.setattr(campaign_module, "check_sampling_manifest", check)


class StubbornBackend(SeededBackend):
    """The shard tests' seeded judge, but every seed in ``stubborn`` is answered invalid (``maybe``)."""

    def __init__(self, stubborn, **kwargs):
        super().__init__(**kwargs)
        self.stubborn = set(stubborn)

    def generate_once(self, prompts, *, seeds=None, **kwargs):
        outcomes = super().generate_once(prompts, seeds=seeds, **kwargs)
        return [GenerationOutcome(text="maybe", raw_response=f"<think>{seed}</think>maybe", finish_reason="stop")
                if seed in self.stubborn else outcome for seed, outcome in zip(seeds, outcomes)]


class LineSeparatorBackend(StubbornBackend):
    """The stubborn judge, each raw response ending in characters ``str.splitlines()`` splits on."""

    def generate_once(self, prompts, *, seeds=None, **kwargs):
        return [outcome if outcome.raw_response is None else dataclasses.replace(
                    outcome, raw_response=f"{outcome.raw_response}\u2028\u2029\x85tail")
                for outcome in super().generate_once(prompts, seeds=seeds, **kwargs)]


def scenario_ids(tmp_path) -> list[str]:
    from ruleforge.scenarios.manifest import ScenarioManifest

    scenarios, _ = write_inputs(tmp_path)
    return list(ScenarioManifest.build(scenarios).scenario_ids)


def _identity(condition: str) -> str:
    return identity_condition_id(condition, condition.rsplit("/", 1)[-1])


def stubborn_seeds(tmp_path, *targets, attempts=ACCEPTED_AT, condition=CONDITION):
    """The full-set seeds of each ``(draw, scenario index)`` at its first ``attempts`` attempts."""
    ids = scenario_ids(tmp_path)
    return {derive_generation_seed(generation_seed_base=16, campaign_id="camp", condition_id=_identity(condition),
                                   scenario_id=ids[index], decoder_draw_id=draw, semantic_attempt_id=attempt,
                                   component_id="full_ruleset")
            for draw, index in targets for attempt in range(attempts)}


def run(cfg, modelspec, backend):
    plan = campaign_module.build_plan(cfg, backend=backend, task_spec=task_spec_for(modelspec))
    return campaign_module.run_plan(plan, backend=backend, task_spec=task_spec_for(modelspec),
                                    execution=campaign_module.execution_settings(cfg))


def run_campaign(tmp_path, modelspec, root, cap, *, sharded, stubborn, only=None, extra=(), backend=StubbornBackend):
    """The condition under ``cap``: unsharded, or as three shards (``only`` some of them)."""
    extra = (f"sampling.max_semantic_attempts={cap}", *extra)
    if not sharded:
        return run(compose_cfg(tmp_path, root, *extra), modelspec, backend(stubborn))
    for index in (range(3) if only is None else only):
        run(shard_cfg(tmp_path, root, index, 3, *extra), modelspec, backend(stubborn))


def amend(tmp_path, root, *bundles, record=True, code=NEW, new_cap=NEW_CAP, **overrides):
    scenarios, rules = write_inputs(tmp_path)
    ids = scenario_ids(tmp_path)
    named = [(CONDITION, draw, ids[index]) for draw, index in (bundles or ((TARGET_DRAW, TARGET_INDEX),))]
    options = {"rules_path": rules, "population_path": scenarios, **DECISION, **overrides}
    return amend_attempt_cap(root, new_cap=new_cap, bundles=named, record=record, code=code, **options)


def held(root: Path, draw: int = TARGET_DRAW, index: int = 2) -> Path:
    """Shard ``index`` of 3's directory of ``draw``."""
    return shard_root(root, CONDITION, index, 3) / CONDITION / f"draw-{draw:03d}"


@pytest.mark.parametrize("sharded", [True, False])
def test_an_amended_and_resumed_campaign_is_the_one_run_with_the_higher_cap_from_the_start(
        tmp_path, modelspec, monkeypatch, sharded):
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX))
    ids = scenario_ids(tmp_path)
    reference, amended = tmp_path / "from-the-start", tmp_path / "amended"
    run_campaign(tmp_path, modelspec, reference, NEW_CAP, sharded=sharded, stubborn=stubborn)
    run_campaign(tmp_path, modelspec, amended, OLD_CAP, sharded=sharded, stubborn=stubborn)
    target = held(amended) if sharded else amended / CONDITION / f"draw-{TARGET_DRAW:03d}"
    gate = json.loads((target / COMPLETION_FNAME).read_text())
    assert (gate["state"], gate["exhausted"], gate["pending"]) == ("incomplete_exhausted", [ids[TARGET_INDEX]], 0)
    before = {name: path.read_bytes() for name, path in files_of(tmp_path).items() if "amended" in name}
    report = amend(tmp_path, amended, record=False)
    assert {name: path.read_bytes() for name, path in files_of(tmp_path).items() if "amended" in name} == before
    assert report["state"] == "dry run: nothing written" and report["written"] == []
    assert report["amended"] == [f"{CONDITION} draw-001 scenario {ids[TARGET_INDEX][:12]} "
                                 f"({'shard-2-of-3' if sharded else 'the campaign root'})"]
    assert report["conditions"][CONDITION]["exhausted"] == 1
    assert report["conditions"][CONDITION]["max_accepted_attempt"] < OLD_CAP
    report = amend(tmp_path, amended)
    assert report["state"] == "recorded" and len(report["written"]) == (3 if sharded else 1)
    for root in campaign_roots(amended):
        assert root.manifest["max_semantic_attempts"] == NEW_CAP
        assert root.manifest[MANIFEST_AMENDMENTS_KEY] == [report["record"]]
    use_code(monkeypatch, NEW)
    if sharded:
        run_campaign(tmp_path, modelspec, amended, NEW_CAP, sharded=True, stubborn=stubborn, only=(2,))
        merged = merge_condition_shards(amended, CONDITION)
        assert merged["draws"]["1"]["amended"] == {"shard-2-of-3": {
            "appended_attempt_rows": ACCEPTED_AT + 1 - OLD_CAP,
            "bundles": {ids[TARGET_INDEX]: {"outcome": "accepted", "accepted_at_attempt": ACCEPTED_AT,
                                            "attempts": ACCEPTED_AT + 1}},
            "resumed_by": "commit bbbbbbbbbbbb, sources 222222222222"}}
        assert "amended" not in merged["draws"]["0"]
        use_code(monkeypatch, OLD)
        merge_condition_shards(reference, CONDITION)
    else:
        run_campaign(tmp_path, modelspec, amended, NEW_CAP, sharded=False, stubborn=stubborn)
    ours, theirs = files_of(reference), files_of(amended)
    assert set(ours) == set(theirs)
    for name, path in ours.items():
        if name.startswith(SNAPSHOT_DIRNAME) or name.endswith((RUNS_FNAME, SAMPLING_MANIFEST_FNAME)):
            continue
        assert_same_file(path, theirs[name], name)
    for name in ours:
        if name.endswith(SAMPLING_MANIFEST_FNAME):
            mine = json.loads(theirs[name].read_text())
            assert mine.pop(MANIFEST_AMENDMENTS_KEY) == [report["record"]]
            assert {k: v for k, v in mine.items() if not k.startswith("code_")} == {
                k: v for k, v in json.loads(ours[name].read_text()).items() if not k.startswith("code_")}
    rows = read_jsonl(amended / CONDITION / "draw-001" / LEDGER_FNAME)
    assert [row["semantic_attempt_id"] for row in rows if row["scenario_id"] == ids[TARGET_INDEX]] == list(
        range(ACCEPTED_AT + 1))
    others = {row["status"] for draw in ("draw-000", "draw-001") for row in read_jsonl(
        amended / CONDITION / draw / LEDGER_FNAME) if row["scenario_id"] != ids[TARGET_INDEX]}
    assert others == {"accepted", "rejected", "transport_failed"}, "the other bundles retry too, below the old cap"
    estimates, expected = analyze_campaign(amended), analyze_campaign(reference)
    disclosure = estimates.pop(MANIFEST_AMENDMENTS_KEY)
    assert estimates == expected
    [entry] = disclosure
    assert (entry["old_cap"], entry["new_cap"], entry["decided_by"], entry["ledger"]) == (
        OLD_CAP, NEW_CAP, "a maintainer", "note 132")
    assert entry["bundles"] == [{
        "condition_id": CONDITION, "decoder_draw_id": TARGET_DRAW, "scenario_id": ids[TARGET_INDEX],
        "held_by": "shard-2-of-3" if sharded else "campaign root", "attempts_at_old_cap": OLD_CAP,
        "outcome": "accepted", "accepted_at_attempt": ACCEPTED_AT, "attempts": ACCEPTED_AT + 1}]
    states = entry["code_states"][CONDITION]
    assert sorted(states) == ["commit aaaaaaaaaaaa, sources 111111111111", "commit bbbbbbbbbbbb, sources 222222222222"]
    assert states["commit bbbbbbbbbbbb, sources 222222222222"] == (["draw-001"] if sharded else ["draw-000", "draw-001"])


@pytest.fixture
def exhausted(tmp_path, modelspec):
    """The condition's three shards at the old cap, its stubborn bundle exhausted; not merged, not amended."""
    root = tmp_path / "campaign"
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True, stubborn=stubborn_seeds(
        tmp_path, (TARGET_DRAW, TARGET_INDEX)))
    return root


def edit_rows(path: Path, edit, *, where=lambda row: True) -> None:
    """Rewrite a JSONL file with ``edit(row)`` applied to each row ``where`` holds."""
    rows = read_jsonl(path)
    path.write_text("".join(json.dumps(edit(row) if where(row) else row) + "\n" for row in rows))


def test_the_new_cap_must_raise_the_old_and_the_bundles_be_named(tmp_path, exhausted):
    for cap in (OLD_CAP, OLD_CAP - 1):
        with pytest.raises(AmendmentError, match=f"the new cap {cap} does not raise the campaign's {OLD_CAP}"):
            amend(tmp_path, exhausted, new_cap=cap)
    with pytest.raises(AmendmentError, match="name the bundles the cap is raised for"):
        amend_attempt_cap(exhausted, new_cap=NEW_CAP, bundles=[], rules_path="r", population_path="p", **DECISION)
    with pytest.raises(AmendmentError, match="holds no campaign"):
        amend(tmp_path, tmp_path / "nowhere")


def test_every_exhausted_bundle_is_named_and_every_named_one_exhausted(tmp_path, modelspec):
    root = tmp_path / "campaign"
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True,
                 stubborn=stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX), (0, 1)))
    ids = scenario_ids(tmp_path)
    with pytest.raises(AmendmentError, match=rf"draw-000 scenario {ids[1][:12]} reached the old cap \(4 attempts\) "
                                             "but is not named"):
        amend(tmp_path, root)
    accepted = (0, 2)
    with pytest.raises(AmendmentError, match=rf"draw-000 scenario {ids[2][:12]} is named but not exhausted \("
                                             r"accepted at attempt \d\)"):
        amend(tmp_path, root, (TARGET_DRAW, TARGET_INDEX), (0, 1), accepted)
    with pytest.raises(AmendmentError, match=r"draw-005 .* is named but not exhausted \(in no finished draw\)"):
        amend(tmp_path, root, (TARGET_DRAW, TARGET_INDEX), (0, 1), (5, 2))
    report = amend(tmp_path, root, (TARGET_DRAW, TARGET_INDEX), (0, 1), record=False)
    assert len(report["record"]["bundles"]) == 2 and report["conditions"][CONDITION]["exhausted"] == 2


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        # A draw that has not finished: a bundle with attempts left, its gate (and a shard's draw) missing.
        (lambda root, ids: edit_rows(held(root) / LEDGER_FNAME, lambda row: {**row, "status": "transport_failed"},
                                     where=lambda row: row["semantic_attempt_id"] == OLD_CAP - 1),
         r"has used 3 of 4 attempts and is not accepted: the draw has not finished"),
        (lambda root, ids: (held(root, 0, 0) / COMPLETION_FNAME).unlink(), r"shard-0-of-3 .* draw-000 has not run"),
        (lambda root, ids: shutil.rmtree(shard_root(root, CONDITION, 0, 3)),
         r"bundle\(s\) are in no finished draw of any root"),
        (lambda root, ids: edit_json(held(root) / COMPLETION_FNAME, state=STATE_COMPLETE, exhausted=[]),
         "its gate says 'complete' with exhausted"),
        (lambda root, ids: edit_json(shard_root(root, CONDITION, 1, 3) / SAMPLING_MANIFEST_FNAME,
                                     generation_seed_base=17), r"sampling manifest differs .*generation_seed_base"),
        (lambda root, ids: reorder(shard_root(root, CONDITION, 1, 3)), "froze another population"),
        (lambda root, ids: reidentify(shard_root(root, CONDITION, 1, 3) / CONDITION / "condition.json"),
         "registers gemma-4-31B-it/naive-rules-above with another identity"),
        (lambda root, ids: shard_root(root, CONDITION, 2, 3).rename(shard_root(root, CONDITION, 2, 3).with_name(
            "shard-9-of-3")), "records shard shard-2-of-3 of gemma-4-31B-it/naive-rules-above, not its directory"),
    ],
)
def test_a_campaign_that_is_not_finished_or_not_one_campaign_is_refused(tmp_path, exhausted, tamper, message):
    tamper(exhausted, scenario_ids(tmp_path))
    with pytest.raises(AmendmentError, match=message):
        amend(tmp_path, exhausted, record=False)


def reorder(root: Path) -> None:
    from tests.test_sampling_shards import reorder_population

    reorder_population(root / "scenario_manifest.json")


def reidentify(path: Path) -> None:
    from ruleforge.evaluate.identity import ConditionSpec

    payload = json.loads(path.read_text())
    payload.pop("fingerprint")
    payload.pop("schema_version")
    spec = ConditionSpec.from_dict({**payload, "judge_type": "Narrow"})
    path.write_text(json.dumps({"schema_version": 2, **spec.to_dict(), "fingerprint": spec.fingerprint}))


def test_a_bundle_accepted_at_or_beyond_the_cap_is_refused(tmp_path, modelspec):
    # A campaign that ran with cap 8 (the stubborn bundle accepted at attempt 6) whose manifests say 4.
    root = tmp_path / "campaign"
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=True, stubborn=stubborn_seeds(
        tmp_path, (TARGET_DRAW, TARGET_INDEX)))
    for each in campaign_roots(root):
        edit_json(each.path / SAMPLING_MANIFEST_FNAME, max_semantic_attempts=OLD_CAP)
    with pytest.raises(AmendmentError, match="was accepted at attempt 6, beyond the old cap"):
        amend(tmp_path, root, record=False)


def test_roots_that_disagree_on_a_bundle_or_hold_one_twice_are_refused(tmp_path, exhausted):
    # The merge brings the exhausted draw into the campaign root too: the bundle is held twice.
    merge_condition_shards(exhausted, CONDITION)
    with pytest.raises(AmendmentError, match=r"is held by 2 roots \(the campaign root, gemma-4-31B-it/naive-rules-above/"
                                             r"shard-2-of-3\): a merged draw beside its shard's"):
        amend(tmp_path, exhausted, record=False)
    ledger = exhausted / CONDITION / "draw-000" / LEDGER_FNAME
    rows = read_jsonl(ledger)
    retried = next(row["scenario_id"] for row in rows if row["status"] == "rejected")
    kept = [row for row in rows if not (row["scenario_id"] == retried and row["status"] != "accepted")]
    for row in kept:
        if row["scenario_id"] == retried:
            row["semantic_attempt_id"] = 0
    ledger.write_text("".join(json.dumps(row) + "\n" for row in kept))
    with pytest.raises(AmendmentError, match=rf"the roots disagree on {CONDITION} draw-000 scenario {retried[:12]}"):
        amend(tmp_path, exhausted, record=False)
    # A bundle exhausted in its shard root and accepted below the cap in the campaign root's copy of its draw.
    merged = exhausted / CONDITION / "draw-001"
    donor = next(row for row in read_jsonl(merged / LEDGER_FNAME) if row["status"] == "accepted")
    last_row(merged / LEDGER_FNAME, lambda row: {**row, "status": "accepted", "failure_reason": None,
                                                 "components": donor["components"]})
    edit_json(merged / COMPLETION_FNAME, state=STATE_COMPLETE, exhausted=[], pending=0)
    sid = scenario_ids(tmp_path)[TARGET_INDEX]
    with pytest.raises(AmendmentError, match=rf"the roots disagree on {CONDITION} draw-001 scenario {sid[:12]}: exhausted "
                                             rf"in {CONDITION}/shard-2-of-3, accepted at attempt {OLD_CAP - 1} in another"):
        amend(tmp_path, exhausted, record=False)


def test_a_named_bundle_s_shard_holding_a_draw_the_amendment_does_not_name_is_refused(tmp_path, modelspec):
    # Shard 1 of 3 holds the last block of draw 0 and the first two of draw 1: its resume would run draw 0 too.
    root = tmp_path / "campaign"
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True, stubborn=stubborn_seeds(tmp_path, (TARGET_DRAW, 1)))
    with pytest.raises(AmendmentError, match=rf"{CONDITION}/shard-1-of-3 holds \['draw-000'\] beside its named bundles' "
                                             r"\['draw-001'\]: its resume would run the amendment's code state in every "
                                             "draw of the shard"):
        amend(tmp_path, root, (TARGET_DRAW, 1), record=False)


def target_rows(root: Path):
    """Edit every ledger row of the stubborn bundle in its shard's draw."""
    sid = None

    def edit(change):
        nonlocal sid
        rows = read_jsonl(held(root) / LEDGER_FNAME)
        sid = rows[-1]["scenario_id"]  # the exhausted bundle's attempts come last
        edit_rows(held(root) / LEDGER_FNAME, change, where=lambda row: row["scenario_id"] == sid)
    return edit


def tampered_component(**changes):
    def edit(row):
        row["components"][1].update(changes)
        return row
    return edit


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda root, tmp: target_rows(root)(tampered_component(prompt_sha256="f" * 64)),
         r"prompts as this code renders them are not the ones its attempts were sent \(1 of 3 components differ, "
         r"first \['Rule 1'\]\)"),
        (lambda root, tmp: edit_rows(held(root) / LEDGER_FNAME, tampered_component(prompt_sha256="f" * 64),
                                     where=lambda row: row["semantic_attempt_id"] == OLD_CAP - 1),
         r"records no prompt digest, or more than one, for component\(s\) \['Rule 1'\]"),
        (lambda root, tmp: target_rows(root)(tampered_component(generation_seed=7)),
         r"/Rule 1 at attempt 0 records seed 7, not the \d+ the runner derives for it"),
        (lambda root, tmp: (held(root) / RAW_OUTPUTS_FNAME).open("a").write('{"torn": '),
         r"torn or broken rows in \['raw_outputs.jsonl'\], and 0 raw output row"),
        (lambda root, tmp: (held(root) / RAW_OUTPUTS_FNAME).open("a").write(json.dumps({
            "scenario_id": "x", "semantic_attempt_id": 0, "request_block_id": "orphan"}) + "\n"),
         r"torn or broken rows in \[\], and 1 raw output row\(s\) no ledger row vouches for"),
        (lambda root, tmp: (tmp / "rules.jsonl").write_text((tmp / "rules.jsonl").read_text() + "\n"),
         r"no ruleset given \(\S+rules.jsonl\) is the one gemma-4-31B-it/naive-rules-above judged"),
    ],
)
def test_the_named_bundle_s_draw_prompts_and_seeds_are_held_to_its_ledger(tmp_path, exhausted, tamper, message):
    amend(tmp_path, exhausted, record=False)
    bundle = (CONDITION, TARGET_DRAW, scenario_ids(tmp_path)[TARGET_INDEX])
    tamper(exhausted, tmp_path)
    with pytest.raises(AmendmentError, match=message):
        amend_attempt_cap(exhausted, new_cap=NEW_CAP, bundles=[bundle], rules_path=tmp_path / "rules.jsonl",
                          population_path=tmp_path / "scenarios.jsonl", code=NEW, **DECISION)


def test_each_condition_s_ruleset_is_the_one_given_with_its_digest(tmp_path, exhausted):
    others = tmp_path / "others"
    (others / "subsets").mkdir(parents=True)
    (others / "another.jsonl").write_text('{"rule": "another"}\n')
    (others / "subsets" / "copy.jsonl").write_bytes((tmp_path / "rules.jsonl").read_bytes())
    for given in ([others / "another.jsonl", tmp_path / "rules.jsonl"], [others], str(others)):
        assert amend(tmp_path, exhausted, record=False, rules_path=given)["state"] == "dry run: nothing written"
    with pytest.raises(AmendmentError, match=r"no ruleset given \(\S+another.jsonl\) is the one "
                                             r"gemma-4-31B-it/naive-rules-above judged \(rules.jsonl\): give it \(--rules\)"):
        amend(tmp_path, exhausted, record=False, rules_path=[others / "another.jsonl"])


def test_the_population_must_be_the_campaign_s_and_a_record_needs_a_clean_code_state(tmp_path, exhausted):
    other = tmp_path / "other.jsonl"
    other.write_text('{"prompt": "p", "response": "r"}\n')
    with pytest.raises(AmendmentError, match="other.jsonl is not the population the campaign froze"):
        amend(tmp_path, exhausted, population_path=other)
    for code in ({**NEW, "code_dirty": True}, {**NEW, "code_dirty": None}, {**NEW, "code_state_sha256": None}):
        with pytest.raises(AmendmentError, match="record from a clean checkout"):
            amend(tmp_path, exhausted, code=code)
        assert amend(tmp_path, exhausted, code=code, record=False)["state"] == "dry run: nothing written"
    assert amend_attempt_cap(exhausted, new_cap=NEW_CAP, bundles=[(CONDITION, TARGET_DRAW, scenario_ids(tmp_path)[
        TARGET_INDEX])], rules_path=tmp_path / "rules.jsonl", population_path=tmp_path / "scenarios.jsonl",
        **DECISION)["code_states"]["prior"] == ["commit aaaaaaaaaaaa, sources 111111111111"]


def test_a_recorded_amendment_is_left_alone_an_interrupted_one_completed(tmp_path, exhausted):
    shard = shard_root(exhausted, CONDITION, 1, 3) / SAMPLING_MANIFEST_FNAME
    original = shard.read_text()
    first = amend(tmp_path, exhausted)
    again = amend(tmp_path, exhausted, reason="another reason")
    assert (again["state"], again["record"], again["written"]) == ("recorded already", first["record"], [])
    shard.write_text(original)  # a record interrupted before it reached shard 1
    interrupted = amend(tmp_path, exhausted, record=False)
    assert (interrupted["state"], interrupted["lacking"]) == ("interrupted", [f"{CONDITION}/shard-1-of-3"])
    completed = amend(tmp_path, exhausted)
    assert (completed["state"], completed["written"]) == ("completed", [str(shard)])
    assert json.loads(shard.read_text())[MANIFEST_AMENDMENTS_KEY] == [first["record"]]
    with pytest.raises(AmendmentError, match=r"records an amendment already \(cap 4 -> 8 for gemma-4-31B-it/naive-rules-above "
                                             r"draw-001 scenario [0-9a-f]{12}\), and this request differs \(cap 9 for"):
        amend(tmp_path, exhausted, new_cap=9)
    with pytest.raises(AmendmentError, match="this request differs"):
        amend(tmp_path, exhausted, (TARGET_DRAW, TARGET_INDEX), (0, 2))
    edit_json(shard, **{MANIFEST_AMENDMENTS_KEY: [{**first["record"], "reason": "edited"}]})
    with pytest.raises(AmendmentError, match="the campaign's roots record different amendments"):
        amend(tmp_path, exhausted)
    shard.write_text(original.replace('"generation_seed_base": 16', '"generation_seed_base": 17'))
    with pytest.raises(AmendmentError, match=r"lacks the amendment and differs from the amended roots \(\['gen"):
        amend(tmp_path, exhausted)


def locked_out(root: Path) -> str:
    """The refusal of a process started while another holds the campaign's lock."""
    return (rf"another process holds {re.escape(str(campaign_lock_path(root)))}, the campaign's lock: a shard merge, "
            "a cap amendment's record or an analysis of this campaign is running")


def test_two_records_or_a_record_and_a_merge_or_an_analysis_never_interleave(tmp_path, exhausted, monkeypatch):
    lock = campaign_lock_path(exhausted)
    assert lock == campaign_lock_path(tmp_path / "." / "campaign") == tmp_path.resolve() / ".campaign.lock"
    amend(tmp_path, exhausted, record=False)
    assert not lock.exists()
    real, refused = cap_module.write_json_sidecar, []

    def write(path, payload):
        while len(refused) < 3:  # at the first record's first write: a second record, a merge, an analysis
            attempt = (lambda: amend(tmp_path, exhausted, now="2026-10-11T00:00:00Z"),
                       lambda: merge_condition_shards(exhausted, CONDITION),
                       lambda: analyze_campaign(exhausted))[len(refused)]
            with pytest.raises(SamplingDesignError, match=locked_out(exhausted)):
                attempt()
            refused.append(attempt)
        return real(path, payload)

    monkeypatch.setattr(cap_module, "write_json_sidecar", write)
    first = amend(tmp_path, exhausted)
    assert (first["state"], len(refused), lock.exists()) == ("recorded", 3, True)
    assert [root.manifest[MANIFEST_AMENDMENTS_KEY] for root in campaign_roots(exhausted)] == [[first["record"]]] * 3
    again = amend(tmp_path, exhausted, now="2026-10-11T00:00:00Z")
    assert (again["state"], again["record"]) == ("recorded already", first["record"])


def test_an_analysis_and_a_merge_or_a_record_never_interleave(tmp_path, monkeypatch, resumed):
    real_merge_gate, real_gate, refused = merge_module.campaign_completion, aggregate_module.require_complete_campaign, []

    def merge_gate(root, **kwargs):
        with pytest.raises(SamplingDesignError, match=locked_out(resumed)):
            analyze_campaign(resumed)
        refused.append("analysis")
        return real_merge_gate(root, **kwargs)

    monkeypatch.setattr(merge_module, "campaign_completion", merge_gate)
    merge_condition_shards(resumed, CONDITION)
    monkeypatch.setattr(merge_module, "campaign_completion", real_merge_gate)

    def gate(root):
        for name, attempt in (("merge", lambda: merge_condition_shards(resumed, CONDITION)),
                              ("record", lambda: amend(tmp_path, resumed, now="2026-10-11T00:00:00Z"))):
            with pytest.raises(SamplingDesignError, match=locked_out(resumed)):
                attempt()
            refused.append(name)
        return real_gate(root)

    monkeypatch.setattr(aggregate_module, "require_complete_campaign", gate)
    estimates = analyze_campaign(resumed)
    assert refused == ["analysis", "merge", "record"]
    written = json.loads((resumed / COMPLETION_FNAME).read_text())
    assert (written["state"], estimates["state"], estimates["max_semantic_attempts"]) == (STATE_COMPLETE,) * 2 + (
        NEW_CAP,)
    assert [entry["bundles"][0]["outcome"] for entry in estimates[MANIFEST_AMENDMENTS_KEY]] == ["accepted"]
    assert merge_condition_shards(resumed, CONDITION)["draws"]["1"]["state"] == "already merged"
    monkeypatch.setattr(aggregate_module, "require_complete_campaign", real_gate)
    assert analyze_campaign(resumed)["conditions"] == estimates["conditions"]


def unrecord(path: Path) -> None:
    """A root's manifest with the record removed by hand, the cap restored."""
    manifest = json.loads(path.read_text())
    manifest.pop(MANIFEST_AMENDMENTS_KEY)
    path.write_text(json.dumps({**manifest, "max_semantic_attempts": OLD_CAP}))


def interrupt_after(patch, writes: int) -> list[str]:
    """Make the manifest writes fail after the first ``writes``; returns the directories written, in order."""
    real, written = cap_module.write_json_sidecar, []

    def write(path, payload):
        if len(written) == writes:
            raise OSError("Disk quota exceeded")
        written.append(Path(path).parent.name)
        return real(path, payload)

    patch.setattr(cap_module, "write_json_sidecar", write)
    return written


def test_an_interrupted_record_runs_no_job_and_is_completed_only_over_the_campaign_it_found(
        tmp_path, modelspec, monkeypatch):
    root = tmp_path / "campaign"
    bundles = ((TARGET_DRAW, TARGET_INDEX), (0, 1))  # held by shards 2 and 0, which hold no other draw
    stubborn = stubborn_seeds(tmp_path, *bundles)
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True, stubborn=stubborn)
    with monkeypatch.context() as patch:
        written = interrupt_after(patch, 1)
        with pytest.raises(OSError, match="Disk quota exceeded"):
            amend(tmp_path, root, *bundles)
    assert written == ["shard-0-of-3"]
    for code in (OLD, NEW):
        use_code(monkeypatch, code)
        for index in (0, 1, 2):
            with pytest.raises(CampaignRunError, match=r"recorded in 1 of its 3 roots, not in \S+/shard-1-of-3, \S+/"
                                                       "shard-2-of-3: a record interrupted part-way. Before any job of "
                                                       "the campaign runs, complete it by recording the interrupted "
                                                       "request again, its cap and bundles the record's$"
                                                       ) as refusal:
                preflight(shard_cfg(tmp_path, root, index, 3, f"sampling.max_semantic_attempts={OLD_CAP}"))
            assert "v1.0" not in str(refusal.value) and "scripts/" not in str(refusal.value)
    with monkeypatch.context() as old_code:
        without_cap_amendments(old_code)
        use_code(old_code, OLD)
        with pytest.raises(SamplingDesignError, match=r"describes a different campaign \(amendments, "
                                                      r"max_semantic_attempts differ\)"):
            preflight(shard_cfg(tmp_path, root, 0, 3, f"sampling.max_semantic_attempts={OLD_CAP}"))
        before = len(read_jsonl(held(root) / RUNS_FNAME))
        run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True, stubborn=stubborn, only=(2,))
        assert len(read_jsonl(held(root) / RUNS_FNAME)) == before + 1
    holding = shard_root(root, CONDITION, 0, 3) / SAMPLING_MANIFEST_FNAME
    for record in (False, True):
        with pytest.raises(AmendmentError, match=r"the record was interrupted, and the campaign is no longer the one "
                                                 rf"it found: {re.escape(CONDITION)} draw-001 scenario \w{{12}} \(shard-"
                                                 r"2-of-3\): its draw is not the bytes the record snapshotted. It cannot "
                                                 rf"be completed: in the roots holding it \({re.escape(CONDITION)}/shard-"
                                                 r"0-of-3\) restore max_semantic_attempts 4 and remove amendments, then "
                                                 "record the amendment again"):
            amend(tmp_path, root, *bundles, record=record)
    unrecord(holding)
    assert amend(tmp_path, root, *bundles)["state"] == "recorded"
    use_code(monkeypatch, NEW)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=True, stubborn=stubborn, only=(0, 2))
    merged = merge_condition_shards(root, CONDITION)
    assert [entry["outcome"] for draw in ("0", "1") for shard in merged["draws"][draw]["amended"].values()
            for entry in shard["bundles"].values()] == ["accepted", "accepted"]


def test_an_interrupted_record_is_completed_only_over_the_campaign_it_found(tmp_path, exhausted):
    record = amend(tmp_path, exhausted)["record"]
    unrecord(shard_root(exhausted, CONDITION, 1, 3) / SAMPLING_MANIFEST_FNAME)
    gate = held(exhausted, 0, 0) / COMPLETION_FNAME
    saved = gate.read_bytes()
    gate.unlink()
    with pytest.raises(AmendmentError, match=r"no longer the one it found: the amendment would change more than the "
                                             r"named bundles, or cannot: \S+/shard-0-of-3 \S+ draw-000 has not run\. It "
                                             rf"cannot be completed: in the roots holding it \({re.escape(CONDITION)}/"
                                             rf"shard-0-of-3, {re.escape(CONDITION)}/shard-2-of-3\) restore"):
        amend(tmp_path, exhausted, record=False)
    gate.write_bytes(saved)
    for index in (0, 2):
        edit_json(shard_root(exhausted, CONDITION, index, 3) / SAMPLING_MANIFEST_FNAME,
                  **{MANIFEST_AMENDMENTS_KEY: [{**record, "conditions": ["gemma-4-31B-it/other"]}]})
    with pytest.raises(AmendmentError, match=r"its roots register \['gemma-4-31B-it/naive-rules-above'\], the record "
                                             r"\['gemma-4-31B-it/other'\]"):
        amend(tmp_path, exhausted)


def test_the_record_reads_back_and_a_manifest_holds_one_cap_amendment(tmp_path, exhausted):
    record = amend(tmp_path, exhausted)["record"]
    amendment = CapAmendment.from_dict(record)
    assert amendment.to_dict() == record and amendment.code_state == ("b" * 40, False, "2" * 64)
    assert cap_amendment({MANIFEST_AMENDMENTS_KEY: [record]}) == amendment and cap_amendment({}) is None
    for records in ([record, record], record, []):
        with pytest.raises(AmendmentError, match="this build reads one cap amendment, no more"):
            cap_amendment({MANIFEST_AMENDMENTS_KEY: records})
    for bad in ({**record, "kind": "other"}, {**record, "schema_version": 2}, "text"):
        with pytest.raises(AmendmentError, match="another kind or version"):
            CapAmendment.from_dict(bad)
    for bad in ({k: v for k, v in record.items() if k != "reason"}, {**record, "old_cap": "x"},
                {**record, "bundles": [{"condition_id": CONDITION}]}):
        with pytest.raises(AmendmentError, match="malformed semantic_attempt_cap amendment"):
            CapAmendment.from_dict(bad)


def test_a_hierarchy_condition_s_prompts_are_rendered_from_its_levels_file(tmp_path, modelspec):
    from ruleforge.core.digests import sha256_file
    from tests.test_sampling_hierarchy import LEVELS, write_levels

    condition = "gemma-4-31B-it/naive-hierarchy"
    levels = write_levels(tmp_path / "levels.jsonl")
    overrides = ("eval_cfg.reasoning_prompt_style=naive-hierarchy", "eval_cfg.individual_rule_context=superiors",
                 f"+rule_levels_path={levels}", f"+approved_rule_levels_sha256={sha256_file(levels)}")
    root = tmp_path / "hierarchy"
    cfg = compose_cfg(tmp_path, root, f"sampling.max_semantic_attempts={OLD_CAP}", *overrides)
    task_spec = modelspec(reasoning_prompt_style="naive-hierarchy", evaluation_temperature=1.0,
                          individual_rule_context="superiors")
    backend = StubbornBackend(stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX), condition=condition))
    plan = campaign_module.build_plan(cfg, backend=backend, task_spec=task_spec)
    campaign_module.run_plan(plan, backend=backend, task_spec=task_spec)
    bundle = (condition, TARGET_DRAW, scenario_ids(tmp_path)[TARGET_INDEX])

    def amend_hierarchy(levels_path):
        return amend_attempt_cap(root, new_cap=NEW_CAP, bundles=[bundle], rules_path=tmp_path / "rules.jsonl",
                                 population_path=tmp_path / "scenarios.jsonl", levels_path=levels_path, code=NEW,
                                 **DECISION)

    assert amend_hierarchy(levels)["amended"] == [f"{condition} draw-001 scenario {bundle[2][:12]} (the campaign root)"]
    other = write_levels(tmp_path / "other_levels.jsonl", {**LEVELS, "Rule 2": "developer"})
    for given in (None, other):
        with pytest.raises(AmendmentError, match=r"judged by the levels file [0-9a-f]{12}\.\.\.; give that file"):
            amend_hierarchy(given)
    misaligned = tmp_path / "misaligned.jsonl"
    misaligned.write_text(levels.read_text().replace("rule 2 body", "another body"))
    with pytest.raises(AmendmentError, match=r"naive-hierarchy draw-001: levels file misaligned.jsonl .* the rule text "
                                             "differs"):
        amend_hierarchy(misaligned)


@pytest.fixture
def amended(tmp_path, exhausted):
    """The exhausted campaign, its cap amended to 8 by the new code state; not resumed."""
    amend(tmp_path, exhausted)
    return exhausted


@pytest.fixture
def resumed(tmp_path, modelspec, monkeypatch, amended):
    """The amended campaign, its stubborn bundle resumed by shard 2 from the new code; not merged."""
    use_code(monkeypatch, NEW)
    run_campaign(tmp_path, modelspec, amended, NEW_CAP, sharded=True, only=(2,),
                 stubborn=stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX)))
    return amended


#: A code state neither the campaign's nor the amendment's.
THIRD = {**NEW, "code_state_sha256": "3" * 64}
#: How the preflight names the code states.
DESCRIBED = {"old": "commit aaaaaaaaaaaa, sources 111111111111", "third": "commit bbbbbbbbbbbb, sources 333333333333"}
#: The preflight's refusal of a shard the amendment does not name, run from :data:`THIRD`.
THIRD_REFUSED = (r"this job runs commit bbbbbbbbbbbb, sources 333333333333, and the campaign's cap amendment allows its "
                 r"layout the campaign's own code state and, only in the draws it names, its own: commit bbbbbbbbbbbb, "
                 r"sources 333333333333 is no code state the campaign ran before it; 2 code states besides the "
                 r"amendment's .* Relaunch the shards it does not name from the campaign's own code state \(ruleforge "
                 r"jobs launch campaign ... --expect-commit a{40}\), the ones it names from its own "
                 r"\(--expect-commit b{40}\)")


def resumes_from_another_code_state(where: str, code: str) -> str:
    """The preflight's refusal of a job resuming the stubborn bundle's draw (held ``where``) from ``code``."""
    return (rf"this job resumes {re.escape(CONDITION)} draw-001 scenario \w{{12}} \({where}\), whose new attempts "
            r"only the amendment's code state, commit bbbbbbbbbbbb, sources 222222222222, may run, and it runs "
            rf"{DESCRIBED[code]}; the merge and the analysis refuse .* --expect-commit bbbbbbbbbbbb")


def test_the_preflight_takes_the_amendment_s_code_state_only_for_shards_it_resumes(tmp_path, monkeypatch, amended):
    cap = f"sampling.max_semantic_attempts={NEW_CAP}"
    use_code(monkeypatch, NEW)
    with pytest.raises(SamplingDesignError, match=r"max_semantic_attempts differ\); use a fresh campaign root .*"
                                                  r"Amended after it was frozen \(amendments\): max_semantic_attempts=8; "
                                                  "a job resumes it stating those values"):
        preflight(shard_cfg(tmp_path, amended, 2, 3, f"sampling.max_semantic_attempts={OLD_CAP}"))
    for code, state in (("old", OLD), ("third", THIRD)):
        use_code(monkeypatch, state)
        with pytest.raises(CampaignRunError, match=resumes_from_another_code_state("shard-2-of-3", code)):
            preflight(shard_cfg(tmp_path, amended, 2, 3, cap))
    with pytest.raises(CampaignRunError, match=THIRD_REFUSED):
        preflight(shard_cfg(tmp_path, amended, 0, 3, cap))
    use_code(monkeypatch, NEW)
    preflight(shard_cfg(tmp_path, amended, 2, 3, cap))
    for index in (0, 1):
        with pytest.raises(CampaignRunError, match=r"its own, commit bbbbbbbbbbbb, sources 222222222222, ran "
                                                   rf"shard-{index}-of-3 draw-000.*which it does not name"):
            preflight(shard_cfg(tmp_path, amended, index, 3, cap))
    use_code(monkeypatch, OLD)
    preflight(shard_cfg(tmp_path, amended, 0, 3, cap))


def test_after_the_named_shard_resumed_the_others_run_on_from_the_campaign_s_code_state(
        tmp_path, modelspec, monkeypatch, resumed):
    cap = f"sampling.max_semantic_attempts={NEW_CAP}"
    use_code(monkeypatch, THIRD)
    with pytest.raises(CampaignRunError, match=THIRD_REFUSED):
        preflight(shard_cfg(tmp_path, resumed, 0, 3, cap))
    use_code(monkeypatch, NEW)
    with pytest.raises(CampaignRunError, match=r"its own, commit bbbbbbbbbbbb, sources 222222222222, ran "
                                               r"shard-0-of-3 draw-000.*which it does not name\. Relaunch only the "
                                               r"shards holding the amended draws \(--only\)"):
        preflight(shard_cfg(tmp_path, resumed, 0, 3, cap))
    preflight(shard_cfg(tmp_path, resumed, 2, 3, cap))
    use_code(monkeypatch, OLD)
    with pytest.raises(CampaignRunError, match=resumes_from_another_code_state("shard-2-of-3", "old")):
        preflight(shard_cfg(tmp_path, resumed, 2, 3, cap))
    for index in (0, 1):
        preflight(shard_cfg(tmp_path, resumed, index, 3, cap))
    run_campaign(tmp_path, modelspec, resumed, NEW_CAP, sharded=True, only=(0,), stubborn=())
    rows = read_jsonl(held(resumed, 0, 0) / RUNS_FNAME)
    assert [(row["code_state_sha256"], row["max_semantic_attempts"]) for row in rows] == [("1" * 64, OLD_CAP),
                                                                                           ("1" * 64, NEW_CAP)]
    merged = merge_condition_shards(resumed, CONDITION)
    assert merged["condition_state"] == {"0": STATE_COMPLETE, "1": STATE_COMPLETE}
    assert merged["draws"]["1"]["amended"]["shard-2-of-3"]["resumed_by"] == "commit bbbbbbbbbbbb, sources 222222222222"


def test_code_predating_the_amendment_runs_no_job_of_an_amended_campaign(tmp_path, monkeypatch, amended):
    without_cap_amendments(monkeypatch)
    for index in range(3):
        for cap, differing in ((OLD_CAP, "amendments, max_semantic_attempts"), (NEW_CAP, "amendments")):
            with pytest.raises(SamplingDesignError, match=rf"describes a different campaign \({differing} differ\)"):
                preflight(shard_cfg(tmp_path, amended, index, 3, f"sampling.max_semantic_attempts={cap}"))


def test_an_unsharded_amended_draw_is_resumed_by_the_amendment_s_code_state_alone(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "unsharded"
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=False, stubborn=stubborn_seeds(
        tmp_path, (TARGET_DRAW, TARGET_INDEX)))
    amend(tmp_path, root)
    cap = f"sampling.max_semantic_attempts={NEW_CAP}"
    for code, state in (("old", OLD), ("third", THIRD)):
        use_code(monkeypatch, state)
        for draws in ((), ("decoder_draw_ids=[1]",)):
            with pytest.raises(CampaignRunError, match=resumes_from_another_code_state("the campaign root", code)):
                preflight(compose_cfg(tmp_path, root, cap, *draws))
        preflight(compose_cfg(tmp_path, root, cap, "decoder_draw_ids=[0]"))
    use_code(monkeypatch, NEW)
    preflight(compose_cfg(tmp_path, root, cap))
    with pytest.raises(CampaignRunError, match=r"draw-001 scenario \w{12} \(the campaign root\) with "
                                               "execution.allow_stack_change, and the analysis refuses"):
        preflight(compose_cfg(tmp_path, root, cap, "execution.allow_stack_change=true"))


def test_work_the_amendment_does_not_cover_is_refused_before_its_judge_loads(tmp_path, monkeypatch, resumed):
    cap = f"sampling.max_semantic_attempts={NEW_CAP}"
    for code in (NEW, OLD):
        use_code(monkeypatch, code)
        for cfg in (shard_cfg(tmp_path, resumed, 0, 3, cap, "eval_cfg.reasoning_prompt_style=encouraged"),
                    compose_cfg(tmp_path, resumed, cap, "eval_cfg.reasoning_prompt_style=encouraged")):
            with pytest.raises(CampaignRunError, match=r"raised from 4 to 8 by an amendment \(recorded as note 132\) covering the "
                                                       rf"conditions and draws it surveyed \({CONDITION}\): "
                                                       "gemma-4-31B-it/encouraged is not one of them"):
                preflight(cfg)
    with pytest.raises(CampaignRunError, match="shard-0-of-2 is a shard root started after it"):
        preflight(shard_cfg(tmp_path, resumed, 0, 2, cap))
    stray = shard_root(resumed, CONDITION, 1, 2)
    stray.mkdir(parents=True)
    shutil.copy(shard_root(resumed, CONDITION, 0, 3) / SAMPLING_MANIFEST_FNAME, stray / SAMPLING_MANIFEST_FNAME)
    with pytest.raises(CampaignRunError, match="shard-1-of-2 is a shard root started after it"):
        preflight(shard_cfg(tmp_path, resumed, 1, 2, cap))
    shutil.rmtree(stray)
    with pytest.raises(CampaignRunError, match=r"does not hold \['draw-000', 'draw-001'\], which this job would start"):
        preflight(compose_cfg(tmp_path, resumed, cap))
    merge_condition_shards(resumed, CONDITION)
    for draws in ((), ("decoder_draw_ids=[1]",)):
        with pytest.raises(CampaignRunError, match=r"draw-001 scenario .* \(shard-2-of-3\) resumed in shard roots and "
                                                   r"merged into .* Leave those draws out \(decoder_draw_ids\)"):
            preflight(compose_cfg(tmp_path, resumed, cap, *draws))
    preflight(compose_cfg(tmp_path, resumed, cap, "decoder_draw_ids=[0]"))


def test_an_amended_draw_not_yet_resumed_is_not_merged(amended):
    with pytest.raises(AmendmentError, match=r"still pending under the amended cap: resume the draw"):
        merge_condition_shards(amended, CONDITION)


def runs_row(root: Path, index: int, draw: int, **changes) -> None:
    """Append to a shard draw's run log a copy of its first row, changed."""
    path = held(root, draw, index) / RUNS_FNAME
    with path.open("a") as handle:
        handle.write(json.dumps({**read_jsonl(path)[0], **changes}) + "\n")


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda root: runs_row(root, 0, 0, code_revision="b" * 40, code_state_sha256="2" * 64),
         r"does not allow this mix: its own, commit bbbbbbbbbbbb, sources 222222222222, ran shard-0-of-3 draw-000"),
        (lambda root: runs_row(root, 0, 0, code_state_sha256="3" * 64),
         r"commit aaaaaaaaaaaa, sources 333333333333 is no code state the campaign ran before it; 2 code states"),
    ],
)
def test_the_merge_takes_the_amendment_s_code_state_only_in_the_draws_it_names(resumed, tamper, message):
    tamper(resumed)
    with pytest.raises(ShardError, match=message):
        merge_condition_shards(resumed, CONDITION)


def append(path: Path, row) -> None:
    with path.open("a") as handle:
        handle.write(row if isinstance(row, str) else json.dumps(row) + "\n")


def last_row(path: Path, change) -> None:
    rows = read_jsonl(path)
    rows[-1] = change(rows[-1])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def on_torch(version: str):
    """A run row's change: its runtime another torch's."""
    return lambda row: {**row, "runtime": {**row["runtime"], "torch_version": version}}


def recorded_prompts(root: Path, digest: str) -> None:
    """Every root's amendment record, its bundle's prompt digest changed."""
    for each in campaign_roots(root):
        record = each.manifest[MANIFEST_AMENDMENTS_KEY][0]
        record["bundles"][0]["prompts_sha256"] = digest
        edit_json(each.path / SAMPLING_MANIFEST_FNAME, **{MANIFEST_AMENDMENTS_KEY: [record]})


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda draw, root: (draw / LEDGER_FNAME).write_text((draw / LEDGER_FNAME).read_text().replace(
            NOW, NOW.replace("00:00:00", "00:00:01"), 1)), "attempts.jsonl no longer begins with the"),
        (lambda draw, root: append(draw / "draw_provenance.json", " "), r"\['draw_provenance.json'\] changed since"),
        (lambda draw, root: append(draw / RAW_OUTPUTS_FNAME, '{"torn"'), "raw_outputs.jsonl ends in a torn row"),
        (lambda draw, root: append(draw / RAW_OUTPUTS_FNAME, '{"broken"\n'), "raw_outputs.jsonl gained a row that is not "
                                                                            "JSON"),
        (lambda draw, root: last_row(draw / LEDGER_FNAME, tampered_component(prompt_sha256="f" * 64)),
         r"/Rule 1 at attempt 6 was sent another prompt than its earlier attempts"),
        (lambda draw, root: last_row(draw / LEDGER_FNAME, tampered_component(generation_seed=7)),
         "at attempt 6 records seed 7, not the"),
        (lambda draw, root: append(draw / RAW_OUTPUTS_FNAME, {"scenario_id": "other", "semantic_attempt_id": 5}),
         "raw_outputs.jsonl gained a row of other at attempt 5"),
        (lambda draw, root: append(draw / REQUEST_BLOCKS_FNAME, {"request_block_id": "b", "semantic_attempt_id": 5,
                                                           "scenario_ids": ["other"]}),
         "request_blocks.jsonl gained block b"),
        (lambda draw, root: append(draw / RUNS_FNAME, {"recorded_at": NOW, "runtime": {},
                                                 "stack_change": {"vllm_version": ["0.28.0", "0.29.0"]}}),
         r"logs a stack change \(vllm_version 0.28.0 -> 0.29.0\)"),
        (lambda draw, root: last_row(draw / RUNS_FNAME, on_torch("2.10.0")),
         r"logs a run on another stack \(torch 2.13.0 -> 2.10.0\) and no stack change, and the rows it appended are "
         "its"),
        (lambda draw, root: last_row(draw / RUNS_FNAME, lambda row: on_torch("2.10.0")(
            {key: value for key, value in row.items() if key != "ledger_rows"})),
         r"and the rows it appended cannot be told from the others'"),
        (lambda draw, root: last_row(draw / RUNS_FNAME, lambda row: {**row, "ledger_rows": row["ledger_rows"] - 1}),
         r"runs.jsonl: its runs' ledger boundaries \[\d+\] are not its ledger's"),
        (lambda draw, root: last_row(draw / RUNS_FNAME, lambda row: {"recorded_at": NOW, "runtime": row["runtime"],
                                                                     "stack_change": {"torch": ["2.13.0", "2.10.0"]}}),
         "runs.jsonl: a stack change logged before any run"),
        # The campaign's own code state, which the merge's code-state check lets in: not a resume's run.
        (lambda draw, root: last_row(draw / RUNS_FNAME, lambda row: {**row, **OLD}),
         "logs a run of commit aaaaaaaaaaaa, sources 111111111111 stating 8 attempts"),
        (lambda draw, root: last_row(draw / RUNS_FNAME, lambda row: {**row, "max_semantic_attempts": 9}),
         "stating 9 attempts"),
        (lambda draw, root: recorded_prompts(root, "e" * 64),
         "earlier attempts were sent other prompts than"),
    ],
)
def test_a_resumed_draw_holds_only_the_amended_bundle_s_new_attempts(resumed, tamper, message):
    tamper(held(resumed), resumed)
    with pytest.raises(AmendmentError, match=message):
        merge_condition_shards(resumed, CONDITION)


def test_a_resumed_draw_whose_run_log_lost_a_whole_run_is_refused(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "campaign"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX))
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=False, stubborn=stubborn)
    amend(tmp_path, root)
    use_code(monkeypatch, NEW)
    with pytest.raises(BatchExecutionError):
        run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=False, stubborn=stubborn,
                     backend=lambda seeds: StubbornBackend(seeds, fail_after=1))
    draw = root / CONDITION / f"draw-{TARGET_DRAW:03d}"
    first = (draw / RUNS_FNAME).read_bytes()
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=False, stubborn=stubborn)
    manifest = json.loads((root / SAMPLING_MANIFEST_FNAME).read_text())
    from ruleforge.evaluate.amendment import disclosure

    [entry] = disclosure(root, manifest)["bundles"]
    assert (entry["outcome"], entry["accepted_at_attempt"]) == ("accepted", ACCEPTED_AT)
    logged, written = (draw / RUNS_FNAME).read_bytes(), (draw / COMPLETION_FNAME).read_bytes()
    assert logged.startswith(first) and logged != first
    (draw / RUNS_FNAME).write_bytes(first)
    with pytest.raises(AmendmentError, match=rf"its runs.jsonl holds {len(first.splitlines())} complete row\(s\), and "
                                             rf"its completion.json counted {len(logged.splitlines())}"):
        disclosure(root, manifest)
    (draw / RUNS_FNAME).write_bytes(logged)
    (draw / COMPLETION_FNAME).write_text(json.dumps({key: value for key, value in json.loads(written).items()
                                                     if key != RUN_LOG_ROWS_KEY}))
    with pytest.raises(AmendmentError, match="its completion.json counts no runs.jsonl rows"):
        disclosure(root, manifest)
    (draw / COMPLETION_FNAME).write_bytes(written)
    held = (draw / LEDGER_FNAME).read_bytes()
    last_row(draw / LEDGER_FNAME, lambda row: {**row, "judged_on": "e" * 64})
    with pytest.raises(AmendmentError, match=r"1 row\(s\) of its attempts.jsonl were judged on a stack \(judged_on "
                                             r"eeeeeeeeeeee\) that neither"):
        disclosure(root, manifest)
    (draw / LEDGER_FNAME).write_bytes(held)
    assert disclosure(root, manifest)["bundles"] == [entry]


def test_a_provider_judge_s_resume_is_held_to_no_stack_as_the_runner_holds_it(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "provider"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX))

    def provider(seeds):
        return StubbornBackend(seeds, model_type="openai_batch")

    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=False, stubborn=stubborn, backend=provider)
    amend(tmp_path, root)
    use_code(monkeypatch, NEW)
    monkeypatch.setattr(runner_module, "_package_version", {**STACK, "torch": "2.10.0"}.get)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=False, stubborn=stubborn, backend=provider)
    assert read_jsonl(root / CONDITION / "draw-001" / RUNS_FNAME)[-1]["runtime"]["torch_version"] == "2.10.0"
    [entry] = analyze_campaign(root)[MANIFEST_AMENDMENTS_KEY][0]["bundles"]
    assert (entry["outcome"], entry["accepted_at_attempt"]) == ("accepted", ACCEPTED_AT)


@pytest.mark.parametrize("sharded", [True, False])
def test_a_finished_amended_draw_run_again_on_another_stack_is_read_as_the_runner_ran_it(
        tmp_path, modelspec, monkeypatch, sharded):
    root = tmp_path / "campaign"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX))
    only, draw = ((2,), held(root)) if sharded else (None, root / CONDITION / "draw-001")
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=sharded, stubborn=stubborn)
    amend(tmp_path, root)
    use_code(monkeypatch, NEW)
    other = {**STACK, "torch": "2.10.0"}.get
    monkeypatch.setattr(runner_module, "_package_version", other)
    logged = read_jsonl(draw / RUNS_FNAME)
    with pytest.raises(CampaignRunError, match=r"recorded on another inference stack or engine \(torch 2.13.0 -> "
                                               r"2.10.0\) and 1 pending bundle"):
        run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=sharded, stubborn=stubborn, only=only)
    assert read_jsonl(draw / RUNS_FNAME) == logged
    monkeypatch.setattr(runner_module, "_package_version", STACK.get)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=sharded, stubborn=stubborn, only=only)
    before = None if sharded else analyze_campaign(root)
    monkeypatch.setattr(runner_module, "_package_version", other)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=sharded, stubborn=stubborn, only=only)
    assert read_jsonl(draw / RUNS_FNAME)[-1]["runtime"]["torch_version"] == "2.10.0"
    if sharded:
        merged = merge_condition_shards(root, CONDITION)["draws"]["1"]["amended"]["shard-2-of-3"]
        assert merged["appended_attempt_rows"] == ACCEPTED_AT + 1 - OLD_CAP
    estimates = analyze_campaign(root)
    assert before is None or estimates == before
    [entry] = estimates[MANIFEST_AMENDMENTS_KEY][0]["bundles"]
    assert (entry["outcome"], entry["accepted_at_attempt"]) == ("accepted", ACCEPTED_AT)


def test_a_bundle_still_exhausted_at_the_new_cap_is_merged_so_and_disclosed(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "campaign"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX), attempts=NEW_CAP)
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True, stubborn=stubborn)
    amend(tmp_path, root)
    use_code(monkeypatch, NEW)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=True, stubborn=stubborn, only=(2,))
    report = merge_condition_shards(root, CONDITION)
    sid = scenario_ids(tmp_path)[TARGET_INDEX]
    assert report["condition_state"] == {"0": STATE_COMPLETE, "1": "incomplete_exhausted"}
    assert report["draws"]["1"]["amended"]["shard-2-of-3"]["bundles"] == {sid: {"outcome": "exhausted",
                                                                               "attempts": NEW_CAP}}
    from ruleforge.evaluate.amendment import disclosure

    manifest = json.loads((root / SAMPLING_MANIFEST_FNAME).read_text())
    assert [(entry["outcome"], entry["attempts"]) for entry in disclosure(root, manifest)["bundles"]] == [
        ("exhausted", NEW_CAP)]
    ledger = held(root) / LEDGER_FNAME
    append(ledger, {**read_jsonl(ledger)[-1], "semantic_attempt_id": NEW_CAP, "request_block_id": "extra"})
    with pytest.raises(AmendmentError, match=f"attempts.jsonl gained a row of {sid[:12]} at attempt {NEW_CAP}"):
        merge_condition_shards(root, CONDITION)


def test_a_resumed_raw_output_may_hold_the_line_separators_json_leaves_raw(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "campaign"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX))
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=True, stubborn=stubborn, backend=LineSeparatorBackend)
    amend(tmp_path, root)
    use_code(monkeypatch, NEW)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=True, stubborn=stubborn, only=(2,),
                 backend=LineSeparatorBackend)
    assert "\u2028\u2029\x85tail" in (held(root) / RAW_OUTPUTS_FNAME).read_text(encoding="utf-8")
    report = merge_condition_shards(root, CONDITION)
    sid = scenario_ids(tmp_path)[TARGET_INDEX]
    assert report["draws"]["1"]["amended"]["shard-2-of-3"]["bundles"][sid]["outcome"] == "accepted"


def test_the_fallback_never_reads_a_draw_whose_gate_predates_the_amended_cap(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "unsharded"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX), attempts=NEW_CAP)
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=False, stubborn=stubborn)
    exclusion = (load_conditions(root)[CONDITION].model_name, scenario_ids(tmp_path)[TARGET_INDEX])
    amend(tmp_path, root)
    with pytest.raises(CampaignRunError, match=r"its ledger has it at 4 of the campaign's 8: a gate written under "
                                               "another cap"):
        analyze_campaign(root, exclude_exhausted=exclusion)
    use_code(monkeypatch, NEW)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=False, stubborn=stubborn)
    payload = analyze_campaign(root, exclude_exhausted=exclusion)
    assert payload["scenario_exclusion"]["arms"][CONDITION]["excluded_scenario"][str(TARGET_DRAW)] == {
        "state": "incomplete_exhausted", "attempts": NEW_CAP}
    assert [(entry["outcome"], entry["attempts"]) for entry in payload[MANIFEST_AMENDMENTS_KEY][0]["bundles"]] == [
        ("exhausted", NEW_CAP)]


def test_a_draw_holding_two_amended_bundles_is_held_once_and_discloses_both(tmp_path, modelspec, monkeypatch):
    from ruleforge.evaluate import amendment as amendment_module

    root = tmp_path / "unsharded"
    named = ((TARGET_DRAW, TARGET_INDEX), (TARGET_DRAW, TARGET_INDEX + 1))
    stubborn = stubborn_seeds(tmp_path, *named)
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=False, stubborn=stubborn)
    amend(tmp_path, root, *named)
    use_code(monkeypatch, NEW)
    run_campaign(tmp_path, modelspec, root, NEW_CAP, sharded=False, stubborn=stubborn)
    held, real = [], amendment_module.check_root_draw

    def counted(*args, **kwargs):
        held.append(kwargs["decoder_draw_id"])
        return real(*args, **kwargs)

    monkeypatch.setattr(amendment_module, "check_root_draw", counted)
    entries = amendment_module.disclosure(root, json.loads((root / SAMPLING_MANIFEST_FNAME).read_text()))["bundles"]
    ids = scenario_ids(tmp_path)
    assert [(entry["scenario_id"], entry["outcome"], entry["accepted_at_attempt"]) for entry in entries] == [
        (ids[index], "accepted", ACCEPTED_AT) for _, index in named]
    assert held == [TARGET_DRAW]


def test_the_disclosure_of_a_root_not_merged_or_not_resumed(tmp_path, modelspec, exhausted):
    from ruleforge.evaluate.amendment import disclosure

    record = amend(tmp_path, exhausted)["record"]
    manifest = {"campaign_id": "camp", "generation_seed_base": 16, MANIFEST_AMENDMENTS_KEY: [record]}
    [entry] = disclosure(exhausted, manifest)["bundles"]
    assert (entry["outcome"], entry["held_by"]) == ("not in this root", "shard-2-of-3")
    assert disclosure(exhausted, {"campaign_id": "camp"}) is None
    root = tmp_path / "unsharded"
    run_campaign(tmp_path, modelspec, root, OLD_CAP, sharded=False, stubborn=stubborn_seeds(
        tmp_path, (TARGET_DRAW, TARGET_INDEX)))
    amend(tmp_path, root)
    manifest = json.loads((root / SAMPLING_MANIFEST_FNAME).read_text())
    [entry] = disclosure(root, manifest)["bundles"]
    assert (entry["outcome"], entry["attempts"], entry["held_by"]) == ("pending", OLD_CAP, "campaign root")
    with pytest.raises(AmendmentError, match="still pending under the amended cap"):
        check_resumed_draw(root / CONDITION / "draw-001", cap_amendment(manifest), campaign_id="camp",
                           condition_id=CONDITION, decoder_draw_id=1, shard=None, generation_seed_base=16)
    assert check_resumed_draw(root / CONDITION / "draw-000", cap_amendment(manifest), campaign_id="camp",
                              condition_id=CONDITION, decoder_draw_id=0, shard=None, generation_seed_base=16) is None
    # A root holding the amended draw's completion record but not its ledger cannot say its outcome; nor can a
    # published copy, which says what it is.
    draw = root / CONDITION / draw_dirname(TARGET_DRAW)
    assert (draw / COMPLETION_FNAME).is_file()
    ledger = (draw / LEDGER_FNAME).read_bytes()
    (draw / LEDGER_FNAME).unlink()
    with pytest.raises(AmendmentError, match=rf"{draw_dirname(TARGET_DRAW)} holds a completion.json but no "
                                             "attempts.jsonl, which the amended bundle's outcome is read from"):
        disclosure(root, manifest)
    (draw / LEDGER_FNAME).write_bytes(ledger)
    (root / PUBLISHED_COPY_FNAME).write_text("{}")
    with pytest.raises(PublishedCopyError, match="not the amended draws' attempt ledgers, which a cap amendment's "
                                                 "disclosure reads"):
        disclosure(root, manifest)


def test_a_root_holding_some_draws_and_shards_the_rest_is_surveyed_whole(tmp_path, modelspec, monkeypatch):
    root = tmp_path / "mixed"
    stubborn = stubborn_seeds(tmp_path, (TARGET_DRAW, TARGET_INDEX))
    run(compose_cfg(tmp_path, root, f"sampling.max_semantic_attempts={OLD_CAP}", "decoder_draw_ids=[0]"), modelspec,
        StubbornBackend(stubborn))
    for index in range(2):
        run(shard_cfg(tmp_path, root, index, 2, f"sampling.max_semantic_attempts={OLD_CAP}", "decoder_draw_ids=[1]"),
            modelspec, StubbornBackend(stubborn))
    report = amend(tmp_path, root, record=False)
    assert report["roots"] == ["the campaign root", f"{CONDITION}/shard-0-of-2", f"{CONDITION}/shard-1-of-2"]
    assert report["amended"] == [f"{CONDITION} draw-001 scenario {scenario_ids(tmp_path)[TARGET_INDEX][:12]} "
                                 "(shard-1-of-2)"]
    with monkeypatch.context() as patch:
        written = interrupt_after(patch, 1)
        with pytest.raises(OSError):
            amend(tmp_path, root)
        without_cap_amendments(patch)
        with pytest.raises(SamplingDesignError, match=r"mixed/sampling_manifest.json describes a different campaign"):
            preflight(shard_cfg(tmp_path, root, 1, 2, f"sampling.max_semantic_attempts={OLD_CAP}",
                                "decoder_draw_ids=[1]"))
    assert written == ["mixed"]
    completed = amend(tmp_path, root)
    assert completed["state"] == "completed" and [Path(path).parent.name for path in completed["written"]] == [
        "shard-1-of-2", "shard-0-of-2"]


def test_the_relaunch_plans_exactly_the_amended_shards_stating_the_new_cap(request, tmp_path, capsys):
    from tests.test_launch_campaign import dry, given, population

    checkout = request.getfixturevalue("repo")  # a clean checkout of this repository's configs (the launch tests')
    _, digest = population(tmp_path, "populations/frozen_500_r3.jsonl")
    printed = dry(checkout, capsys, "--spec", "prompt_ablation", "--var", "campaign_id=prompt-ablation-v2-2026-10",
                  "--var", "models=gemma4-31b-tp-2", "--var", "styles=forced",
                  "--var", "attempts=50", "--shards", "8", "--only", "2,3,5", "--timeout-min", "960",
                  "--expect-population-sha256", digest)
    jobs = printed["jobs"]
    assert [job["job_id"].rsplit("-shard-", 1)[1] for job in jobs] == ["2-of-8", "3-of-8", "5-of-8"]
    for job, index in zip(jobs, (2, 3, 5)):
        arguments = given(job)
        assert [arguments.count("sampling.max_semantic_attempts=50"), f"execution.shard_index={index}" in arguments,
                "eval_cfg.reasoning_prompt_style=forced" in arguments] == [1, True, True]
        resources = job["resources"]
        assert (resources["gpus"], resources["qos"], resources["timeout_min"], resources["resumable"]) == (
            1, "normal", 960, True)
    [merge] = printed["merges"]
    assert (merge["condition_id"], merge["num_shards"]) == ("gemma-4-31B-it/forced", 8)
