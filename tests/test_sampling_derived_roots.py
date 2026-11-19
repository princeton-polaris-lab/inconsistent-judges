from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from ruleforge.evaluate import aggregate as aggregate_module
from ruleforge.evaluate import campaign as campaign_module
from tests.support import cap_amendment as cap_module
from tests.support.derived_root import derive
from tests.support import rebase as rebase_module
from ruleforge.evaluate import runner as runner_module
from ruleforge.evaluate import shard_merge as merge_module
from ruleforge.core.record_names import identity_condition_id
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME
from ruleforge.evaluate.campaign import build_plan, run_plan
from ruleforge.evaluate.amendment import AMENDED_FIELD, AmendmentError, cap_amendment, disclosure
from tests.support.cap_amendment import amend_attempt_cap
from ruleforge.evaluate.identity import (
    COMPLETION_FNAME,
    MANIFEST_AMENDMENTS_KEY,
    RUN_LOG_ROWS_KEY,
    derive_generation_seed,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.identity import REPLAY_RECORD_FNAME
from ruleforge.evaluate.ledger import LEDGER_FNAME
from ruleforge.evaluate.campaign import REBASE_FNAME
from tests.support.rebase import (
    execute_rebase,
    plan_rebase,
)
from tests.support.rebase_checks import check_added, check_retained
from ruleforge.evaluate.runner import STATE_COMPLETE, CampaignRunError
from ruleforge.evaluate.sharding import job_root_override, shards_root
from ruleforge.core.artifacts import compute_scenario_id
from ruleforge.judges.backend import GenerationOutcome
from tests import test_sampling_amendment as amendment_tests
from tests.support.trees import settled
from tests.test_sampling_amendment import DECISION, NEW, OLD, StubbornBackend, use_code
from tests.test_sampling_shards import (
    read_jsonl,
    CONDITION,
    NOW,
    PER_BLOCK,
    STACK,
    SeededBackend,
    write_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
#: The scenario the target removes (block 2 of each draw: 6, 7, 8) and the one it adds in its place.
REMOVED_INDEX, ADDED = 7, {"prompt": "prompt 10", "response": "response 10"}
CAP, RAISED = 4, 8
N_DRAWS = 2
REBASE_CODE = {"code_revision": "c" * 40, "code_dirty": False, "code_state_sha256": "3" * 64}


@pytest.fixture(autouse=True)
def _environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    for module in (runner_module, campaign_module, merge_module, aggregate_module, cap_module):
        monkeypatch.setattr(module, "utc_now", lambda: NOW)
    monkeypatch.setattr(runner_module, "_package_version", STACK.get)
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)
    monkeypatch.setattr(rebase_module, "code_state", lambda: dict(REBASE_CODE))
    monkeypatch.setattr(rebase_module, "write_code_snapshot", lambda root, state: None)
    use_code(monkeypatch, OLD)


def ids(tmp_path) -> list[str]:
    scenarios, _ = write_inputs(tmp_path)
    return [compute_scenario_id(json.loads(line)) for line in scenarios.read_text().splitlines()]


def added_id() -> str:
    return compute_scenario_id(ADDED)


def revised(tmp_path, index=REMOVED_INDEX) -> Path:
    """The test population with row ``index`` swapped for a scenario no parent judged (``revise`` of one row)."""
    return revise(tmp_path, write_inputs(tmp_path)[0], {index: ADDED}, f"revised-{index}")


def cfg(tmp_path, root, *extra, population=None, style="naive-rules-above", cap=CAP):
    """The shard tests' campaign (ten scenarios, blocks of three, two draws), over ``population`` if given."""
    scenarios, rules = write_inputs(tmp_path)
    overrides = {
        "campaign_id": "camp", "campaign_root": str(root), "scenario_set_path": str(population or scenarios),
        "approved_scenario_sha256": sha256_file(population) if population else "null", "rule_set_path": str(rules),
        "sampling.campaign_mode": "pilot", "sampling.num_decoder_draws": str(N_DRAWS),
        "sampling.scenarios_per_request_block": str(PER_BLOCK), "sampling.bootstrap.n_resamples": "150",
        "sampling.max_semantic_attempts": str(cap), "eval_cfg.reasoning_prompt_style": style,
    }
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        return compose(config_name="sampling_campaign",
                       overrides=[f"{key}={value}" for key, value in overrides.items()] + list(extra))


def run(config, modelspec, backend, style="naive-rules-above"):
    """Run the campaign ``config`` composes."""
    task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0)
    plan = build_plan(config, backend=backend, task_spec=task_spec)
    completions = run_plan(plan, backend=backend, task_spec=task_spec,
                           execution=campaign_module.execution_settings(config))
    settled(config.campaign_root)
    return completions


def _identity(condition: str) -> str:
    return identity_condition_id(condition, condition.rsplit("/", 1)[-1])


def stubborn(tmp_path, *targets, attempts=RAISED, condition=CONDITION):
    """The full-set seeds of each ``(draw, scenario index)`` at its first ``attempts`` attempts: answered invalid."""
    scenario = ids(tmp_path)
    return {derive_generation_seed(generation_seed_base=16, campaign_id="camp", condition_id=_identity(condition),
                                   scenario_id=scenario[index], decoder_draw_id=draw, semantic_attempt_id=attempt,
                                   component_id="full_ruleset")
            for draw, index in targets for attempt in range(attempts)}


def added_seeds(condition=CONDITION, components=("full_ruleset", "Rule 1", "Rule 2"), cap=RAISED) -> set:
    """Every seed the added scenario's attempts may carry: only these may reach the judge on a resume."""
    return seeds_of([added_id()], attempts=range(cap), components=components, condition=condition)


def resume(tmp_path, modelspec, new_root, population, *extra, style="naive-rules-above", cap=CAP, backend=None):
    """Resume the rebased root through the normal entry point, restating the parent's campaign."""
    backend = backend or SeededBackend()
    completions = run(cfg(tmp_path, new_root, *extra, population=population, style=style, cap=cap), modelspec,
                      backend, style)
    return backend, completions


def judged_seeds(backend) -> set:
    return {seed for _, _, seeds in backend.blocks for seed in seeds}


@pytest.fixture
def parent(tmp_path, modelspec):
    root = tmp_path / "parent"
    run(cfg(tmp_path, root), modelspec, SeededBackend())
    return root


COMPONENTS = ("full_ruleset", "Rule 1", "Rule 2")


def revise(tmp_path, source: Path, swaps, name: str) -> Path:
    """``source`` with each row of ``swaps`` (index -> scenario) replaced: a hop's target population."""
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for index, row in swaps.items():
        rows[index] = dict(row)
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def seeds_of(scenario_ids, *, draws=range(N_DRAWS), attempts=range(RAISED), components=COMPONENTS,
             condition=CONDITION) -> set:
    """Every seed the given scenarios' attempts may carry (``added_seeds`` over several scenarios)."""
    return {derive_generation_seed(generation_seed_base=16, campaign_id="camp", condition_id=_identity(condition),
                                   scenario_id=sid, decoder_draw_id=draw, semantic_attempt_id=attempt,
                                   component_id=component)
            for sid in scenario_ids for draw in draws for attempt in attempts for component in components}


@pytest.fixture
def amended_rescore(tmp_path, modelspec, monkeypatch):
    """A rebased raw root amended from cap 4 to 8 and resumed, then copied as a re-score: the source and the copy."""
    parent = tmp_path / "parent"
    run(cfg(tmp_path, parent), modelspec, SeededBackend())
    population = revised(tmp_path)
    source = tmp_path / "new"
    execute_rebase(plan_rebase(parent, population), source)
    hard = {derive_generation_seed(generation_seed_base=16, campaign_id="camp", condition_id=_identity(CONDITION),
                                   scenario_id=added_id(), decoder_draw_id=0, semantic_attempt_id=attempt,
                                   component_id="full_ruleset") for attempt in range(6)}
    resume(tmp_path, modelspec, source, population, backend=StubbornBackend(hard))
    rules = tmp_path / "rules.jsonl"
    amend_attempt_cap(source, new_cap=RAISED, bundles=[(CONDITION, 0, added_id())], record=True, code=NEW,
                      rules_path=rules, population_path=population, **DECISION)
    use_code(monkeypatch, NEW)
    resume(tmp_path, modelspec, source, population, cap=RAISED, backend=StubbornBackend(hard))
    derived = tmp_path / "rescored"
    derive(source, derived)
    return source, derived


def _rewritten(path: Path, edit) -> bytes:
    saved = path.read_bytes()
    payload = json.loads(saved)
    edit(payload)
    path.write_text(json.dumps(payload))
    return saved


def test_no_job_logs_a_run_in_an_amended_draw_of_a_derived_root(tmp_path, amended_rescore):
    _, derived = amended_rescore
    population = revised(tmp_path)
    for draws in ((), ("decoder_draw_ids=[0]",)):
        with pytest.raises(CampaignRunError, match=r"is a label replay's derived root \(label_replay.json\).* A derived "
                                                   r"root is never resumed"):
            campaign_module.preflight(cfg(tmp_path, derived, *draws, population=population, cap=RAISED))
    campaign_module.preflight(cfg(tmp_path, derived, "decoder_draw_ids=[1]", population=population, cap=RAISED))


def test_a_derived_root_is_read_as_one_only_when_its_replay_record_describes_it(tmp_path, amended_rescore):
    source, derived = amended_rescore
    manifest = json.loads((derived / SAMPLING_MANIFEST_FNAME).read_text())
    raw_manifest = json.loads((source / SAMPLING_MANIFEST_FNAME).read_text())
    expected = disclosure(derived, manifest)
    assert [(entry["outcome"], entry["accepted_at_attempt"]) for entry in expected["bundles"]] == [("accepted", 6)]
    record_path, provenance = derived / REPLAY_RECORD_FNAME, source / CONDITION / "draw-000" / "draw_provenance.json"
    saved_provenance = _rewritten(provenance, lambda payload: payload.update(note="edited"))
    with pytest.raises(AmendmentError, match=r"\['draw_provenance.json'\] changed since the amendment"):
        disclosure(source, raw_manifest)
    marker = source / REPLAY_RECORD_FNAME
    recorded = json.loads(record_path.read_text())
    relabelled = {**recorded, "label_rule": "lenient"}
    refingerprinted = json.loads(record_path.read_text())
    refingerprinted["conditions"][CONDITION]["fingerprint"] = "0" * 64
    for text, message in [
        ("", "is no label replay's record"),
        (json.dumps(relabelled), "records a re-score of campaign 'camp' under 'lenient', not this root's 'camp' under "
                                 "'lenient_last_line'"),
        (json.dumps(refingerprinted), "names another condition than the re-score wrote; its "
                                      "\\['draw_provenance.json'\\] are not the files the re-score wrote"),
        (record_path.read_text(), "its \\['draw_provenance.json'\\] are not the files the re-score wrote"),
    ]:
        marker.write_text(text)
        with pytest.raises(AmendmentError, match=message):
            disclosure(source, raw_manifest)
    marker.unlink()
    with pytest.raises(AmendmentError, match=r"\['draw_provenance.json'\] changed since the amendment"):
        disclosure(derived, manifest)
    provenance.write_bytes(saved_provenance)
    assert disclosure(derived, manifest) == expected
    derived_draw, source_draw = derived / CONDITION / "draw-000", source / CONDITION / "draw-000"
    for path, edit, message in [
        (derived_draw / LEDGER_FNAME, None, r"its \['attempts.jsonl'\] are not the files the re-score wrote"),
        # A runtime field of the derived draw's provenance edited: the files the code states are read beside.
        (derived_draw / "draw_provenance.json", lambda payload: payload["runtime"].update(edited=True),
         r"its \['draw_provenance.json'\] are not the files the re-score wrote"),
        (source / CONDITION / "draw-000" / LEDGER_FNAME, None, "is not the one the re-score read"),
        (source / SAMPLING_MANIFEST_FNAME, lambda payload: payload[MANIFEST_AMENDMENTS_KEY][0].update(reason="other"),
         "records another cap amendment than this root"),
        (record_path, lambda payload: payload["conditions"][CONDITION]["draws"]["0"].pop("replayed_draw"),
         "records no re-score of it to hold it to"),
        (record_path, lambda payload: payload["conditions"][CONDITION].update(from_shards=True),
         "names no draw of a campaign root as its source"),
        (record_path, lambda payload: payload["conditions"][CONDITION]["draws"]["0"]["source_ledgers"][0].update(
            path=str(source_draw.with_name("draw-001"))), "names no draw of a campaign root as its source"),
        (record_path, lambda payload: payload["conditions"][CONDITION].update(source=str(tmp_path / "gone")) or
         payload["conditions"][CONDITION]["draws"]["0"]["source_ledgers"][0].update(
             path=str(tmp_path / "gone" / CONDITION / "draw-000")), "its source draw cannot be read"),
    ]:
        if edit is None:
            saved = path.read_bytes()
            path.write_bytes(saved + saved.splitlines(keepends=True)[-1])
        else:
            saved = _rewritten(path, edit)
        with pytest.raises(AmendmentError, match=message):
            disclosure(derived, manifest)
        path.write_bytes(saved)
    before_resume = cap_amendment(manifest).bundles[0].files["runs.jsonl"]["bytes"]
    for path, message, roots in [
        (derived_draw / "runs.jsonl", r"its \['runs.jsonl'\] are not the files the re-score wrote", [derived]),
        (source_draw / "runs.jsonl", r"attempts.jsonl gained 3 row\(s\), and runs.jsonl logs no run since the "
                                     "amendment was recorded", [source, derived]),
    ]:
        saved = path.read_bytes()
        assert len(saved) > before_resume
        path.write_bytes(saved[:before_resume])
        for root in roots:
            with pytest.raises(AmendmentError, match=message):
                disclosure(root, json.loads((root / SAMPLING_MANIFEST_FNAME).read_text()))
        path.write_bytes(saved)
    saved = (derived_draw / "runs.jsonl").read_bytes()
    (derived_draw / "runs.jsonl").unlink()
    with pytest.raises(AmendmentError, match="its files cannot be read to hold them to the re-score"):
        disclosure(derived, manifest)
    (derived_draw / "runs.jsonl").write_bytes(saved)
    assert disclosure(derived, manifest) == expected and disclosure(source, raw_manifest) is not None


@pytest.fixture
def merged_amended(tmp_path, modelspec, monkeypatch):
    """The amendment tests' sharded campaign (three shards) amended from 4 to 8, resumed and merged."""
    root, extra = tmp_path / "campaign", ()
    stubborn = amendment_tests.stubborn_seeds(tmp_path, (1, REMOVED_INDEX))
    amendment_tests.run_campaign(tmp_path, modelspec, root, CAP, sharded=True, stubborn=stubborn, extra=extra)
    amendment_tests.amend(tmp_path, root)
    use_code(monkeypatch, NEW)
    amendment_tests.run_campaign(tmp_path, modelspec, root, RAISED, sharded=True, stubborn=stubborn, only=(2,),
                                 extra=extra)
    merge_module.merge_condition_shards(root, CONDITION)
    return root


def test_a_merged_shard_draw_of_an_amended_root_is_held_to_its_shard_roots_whenever_it_is_read(
        tmp_path, merged_amended):
    root = merged_amended
    scenarios, rules = write_inputs(tmp_path)
    manifest = json.loads((root / SAMPLING_MANIFEST_FNAME).read_text())
    expected = disclosure(root, manifest)
    assert [(entry["held_by"], entry["outcome"], entry["accepted_at_attempt"]) for entry in expected["bundles"]] == [
        ("shard-2-of-3", "accepted", 6)]
    assert sorted(expected["code_states"][CONDITION]) == ["commit aaaaaaaaaaaa, sources 111111111111",
                                                          "commit bbbbbbbbbbbb, sources 222222222222"]
    runs = root / CONDITION / "draw-001" / "runs.jsonl"
    gate = runs.with_name(COMPLETION_FNAME)
    merged, merged_gate = runs.read_bytes(), gate.read_bytes()
    stale = b"".join(line for line in merged.splitlines(keepends=True) if b'"code_revision": "bbbb' not in line)
    assert stale != merged
    runs.write_bytes(stale)
    with pytest.raises(AmendmentError, match=r"merged from shard roots, it cannot be held to them .*: its runs.jsonl "
                                             rf"holds {len(stale.splitlines())} complete row\(s\), and its "
                                             rf"completion.json counted {len(merged.splitlines())}"):
        disclosure(root, manifest)
    gate.write_text(json.dumps({**json.loads(merged_gate), RUN_LOG_ROWS_KEY: len(stale.splitlines())}))
    with pytest.raises(AmendmentError, match="merged from shard roots, it cannot be held to them .* is not the merge "
                                             "of its shard roots"):
        disclosure(root, manifest)
    runs.write_bytes(merged)
    gate.write_bytes(merged_gate)
    derived = tmp_path / "rescored"
    derive(root, derived)
    derived_manifest = json.loads((derived / SAMPLING_MANIFEST_FNAME).read_text())
    assert disclosure(derived, derived_manifest)["bundles"] == expected["bundles"]
    runs.write_bytes(stale)
    gate.write_text(json.dumps({**json.loads(merged_gate), RUN_LOG_ROWS_KEY: len(stale.splitlines())}))
    with pytest.raises(AmendmentError, match="is not the merge of its shard roots"):
        disclosure(derived, derived_manifest)
    runs.write_bytes(merged)
    gate.write_bytes(merged_gate)
    shard_runs = amendment_tests.held(root) / "runs.jsonl"
    shard_gate = shard_runs.with_name(COMPLETION_FNAME)
    saved, saved_gate = shard_runs.read_bytes(), shard_gate.read_bytes()
    cut = saved[:cap_amendment(manifest).bundles[0].files["runs.jsonl"]["bytes"]]
    shard_runs.write_bytes(cut)
    for recount, message in ((False, "its runs.jsonl holds .* complete row\\(s\\), and its completion.json counted"),
                             (True, "logs no run since the amendment was recorded")):
        if recount:
            shard_gate.write_text(json.dumps({**json.loads(saved_gate), RUN_LOG_ROWS_KEY: len(cut.splitlines())}))
        for each, payload in ((root, manifest), (derived, derived_manifest)):
            with pytest.raises(AmendmentError, match=message):
                disclosure(each, payload)
    shard_runs.write_bytes(saved)
    shard_gate.write_bytes(saved_gate)
    assert disclosure(derived, derived_manifest)["bundles"] == expected["bundles"]
    shutil.move(shards_root(root), tmp_path / "aside")
    with pytest.raises(AmendmentError, match="holds no shard roots"):
        disclosure(root, manifest)
    shutil.move(tmp_path / "aside", shards_root(root))
    with pytest.raises(SamplingDesignError, match="no shard root in .* holds draw 5"):
        merge_module.check_merged_draw(root, CONDITION, 5, num_shards=3)
    amendment = cap_amendment(manifest)
    for label, message in (("shard-two", "names no shard root"), ("shard-0-of-3", "holds no amended bundle")):
        named = {**manifest, MANIFEST_AMENDMENTS_KEY: [dataclasses.replace(amendment, bundles=(dataclasses.replace(
            amendment.bundles[0], shard=label),)).to_dict()]}
        with pytest.raises(AmendmentError, match=message):
            disclosure(root, named)


def test_a_re_score_from_shard_roots_is_held_to_them_whenever_its_root_is_read(tmp_path, merged_amended):
    root = merged_amended
    scenarios, rules = write_inputs(tmp_path)
    derived = tmp_path / "rescored"
    derive(root, derived, shards=shards_root(root) / CONDITION)
    manifest = json.loads((derived / SAMPLING_MANIFEST_FNAME).read_text())
    expected = disclosure(derived, manifest)
    assert [(entry["outcome"], entry["accepted_at_attempt"]) for entry in expected["bundles"]] == [("accepted", 6)]
    shard_draw = amendment_tests.held(root)
    record_path = derived / REPLAY_RECORD_FNAME
    for path, edit, message in [
        (shard_draw / "runs.jsonl", lambda data: data[:cap_amendment(manifest).bundles[0].files["runs.jsonl"]["bytes"]],
         "logs no run since the amendment was recorded"),
        (shard_draw / LEDGER_FNAME, lambda data: data + data.splitlines(keepends=True)[-1],
         "its source shard draws in .* are not the ones the re-score read"),
        (shard_draw.parents[2] / SAMPLING_MANIFEST_FNAME, lambda data: data.replace(b"retry the exhausted", b"other"),
         "its source shard roots in .* record another cap amendment than this root"),
        (record_path, lambda data: json.dumps(_dropped_source(json.loads(data), shard_draw)).encode(),
         "names no draw of the shard roots holding its amended bundles as its source"),
    ]:
        saved = path.read_bytes()
        path.write_bytes(edit(saved))
        with pytest.raises(AmendmentError, match=message):
            disclosure(derived, manifest)
        path.write_bytes(saved)
    assert disclosure(derived, manifest) == expected


def _dropped_source(record: dict, shard_draw: Path) -> dict:
    entry = record["conditions"][CONDITION]["draws"]["1"]
    entry["source_ledgers"] = [source for source in entry["source_ledgers"]
                               if Path(source["path"]).resolve() != shard_draw.resolve()]
    return record


def test_a_rebase_record_binds_to_its_own_root(tmp_path, parent):
    new_root = tmp_path / "new"
    execute_rebase(plan_rebase(parent, revised(tmp_path)), new_root)
    assert campaign_module.rebase_record(parent) is None and campaign_module.carried_exhausted(parent) == set()
    assert campaign_module.rebase_record(new_root)["removed_ids"] == [ids(tmp_path)[REMOVED_INDEX]]
    assert campaign_module.carried_exhausted(new_root) == set()
    stray = parent / REBASE_FNAME
    stray.write_bytes((new_root / REBASE_FNAME).read_bytes())
    with pytest.raises(SamplingDesignError, match="not this root's 'camp' over .*: it is no record of this root"):
        campaign_module.carried_exhausted(parent)
    stray.write_text("[]")
    with pytest.raises(SamplingDesignError, match="cannot be read as a rebase's record"):
        campaign_module.rebase_record(parent)
    stray.unlink()
    _rewritten(new_root / REBASE_FNAME, lambda payload: payload.update(carried_exhausted={"no-draw": ["x"]}))
    with pytest.raises(SamplingDesignError, match="records a malformed carried_exhausted"):
        campaign_module.carried_exhausted(new_root)


ENCOURAGED = "gemma-4-31B-it/encouraged"


@pytest.fixture
def sharded_source(tmp_path, modelspec):
    """``naive`` in the root."""
    root = tmp_path / "parent"
    run(cfg(tmp_path, root), modelspec, SeededBackend())
    seeds = stubborn(tmp_path, (0, 2), condition=ENCOURAGED)
    for index in range(3):
        config = cfg(tmp_path, root, "execution.num_shards=3", f"execution.shard_index={index}",
                     job_root_override(index, 3), style="encouraged")
        run(config, modelspec, StubbornBackend(seeds), "encouraged")
    return root, shards_root(root) / ENCOURAGED


class UnreadableBackend(SeededBackend):
    """The seeded judge, every seed in ``unreadable`` answered ``maybe``, which the label reader reads as no label."""

    def __init__(self, unreadable, **kwargs):
        super().__init__(**kwargs)
        self.unreadable = set(unreadable)

    def generate_once(self, prompts, *, seeds=None, **kwargs):
        outcomes = super().generate_once(prompts, seeds=seeds, **kwargs)
        return [GenerationOutcome(text="maybe", raw_response=f"<think>{seed}</think>maybe", finish_reason="stop")
                if seed in self.unreadable else outcome for seed, outcome in zip(seeds, outcomes)]


@pytest.mark.parametrize("same_draw", [False, True], ids=["another-draw", "the-held-bundle-s-draw"])
def test_a_carried_bundle_is_held_at_the_old_cap_while_the_added_one_is_amended(
        tmp_path, modelspec, monkeypatch, same_draw):
    kept, r_draw = ids(tmp_path)[4], int(same_draw)
    parent = tmp_path / "parent"
    run(cfg(tmp_path, parent), modelspec, UnreadableBackend(stubborn(tmp_path, (1, 4), attempts=CAP)))
    population = revised(tmp_path, index=2)  # keeps scenario 4, exhausted in draw 1
    new_root = tmp_path / "new"
    execute_rebase(plan_rebase(parent, population, carry_exhausted=[kept]), new_root)
    hard = {derive_generation_seed(generation_seed_base=16, campaign_id="camp", condition_id=_identity(CONDITION),
                                   scenario_id=added_id(), decoder_draw_id=r_draw, semantic_attempt_id=attempt,
                                   component_id="full_ruleset") for attempt in range(6)}
    _, completions = resume(tmp_path, modelspec, new_root, population, backend=StubbornBackend(hard))
    assert completions[r_draw]["exhausted"] == sorted({added_id(), *([kept] if same_draw else [])})
    assert completions[1]["exhausted"] == sorted({kept, *([added_id()] if same_draw else [])})
    rules = tmp_path / "rules.jsonl"
    decision = {"rules_path": rules, "population_path": population, "code": NEW, **DECISION}
    with pytest.raises(SamplingDesignError, match="is a retained scenario a rebase carried over exhausted"):
        amend_attempt_cap(new_root, new_cap=RAISED, bundles=[(CONDITION, r_draw, added_id()), (CONDITION, 1, kept)],
                          **decision)
    rebase_path = new_root / REBASE_FNAME
    saved_rebase = rebase_path.read_bytes()
    listed = json.loads(saved_rebase)
    listed["carried_exhausted"][f"{CONDITION}/draw-000"] = [ids(tmp_path)[5]]  # accepted in draw 0
    rebase_path.write_text(json.dumps(listed))
    with pytest.raises(SamplingDesignError, match="records carried over exhausted, but no finished draw holds it"):
        amend_attempt_cap(new_root, new_cap=RAISED, bundles=[(CONDITION, r_draw, added_id())], **decision)
    rebase_path.write_bytes(saved_rebase)
    report = amend_attempt_cap(new_root, new_cap=RAISED, bundles=[(CONDITION, r_draw, added_id())], record=True,
                               **decision)
    assert report["held"] == [f"{CONDITION} draw-001 scenario {kept[:12]}"] and "held" in report["record"]["checks"]
    assert report["record"]["held"] == [{"condition_id": CONDITION, "decoder_draw_id": 1, "scenario_id": kept}]
    path = new_root / SAMPLING_MANIFEST_FNAME
    manifest = json.loads(path.read_bytes())
    manifest[MANIFEST_AMENDMENTS_KEY][0]["held"][0]["decoder_draw_id"] = 0
    path.write_text(json.dumps(manifest))
    with pytest.raises(SamplingDesignError, match=r"it holds \[\('[^']+', 0, '[0-9a-f]+'\)\] at the old cap, which "
                                                  r".* No job of the campaign runs on it\. While no draw has resumed "
                                                  r"under the raised cap, it is put right by recording it again: keep "
                                                  r"a copy of the record \(its cap, bundles and decision are the "
                                                  r"request's\), in every root holding it "
                                                  f"\\({re.escape(str(new_root))}\\) restore "
                                                  f"max_semantic_attempts {CAP} and remove {MANIFEST_AMENDMENTS_KEY}, "
                                                  "then record it again$") as refusal:
        campaign_module.campaign_amendment(new_root)
    assert "v1.0" not in str(refusal.value) and "scripts/" not in str(refusal.value)
    del manifest[MANIFEST_AMENDMENTS_KEY]
    path.write_text(json.dumps({**manifest, AMENDED_FIELD: CAP}))
    again = amend_attempt_cap(new_root, new_cap=RAISED, bundles=[(CONDITION, r_draw, added_id())], record=True,
                              **decision)
    assert again["state"] == report["state"] and again["record"]["held"] == report["record"]["held"]
    assert campaign_module.campaign_amendment(new_root).held == cap_amendment(json.loads(path.read_bytes())).held
    use_code(monkeypatch, NEW)
    packed = ("execution.pack_draws=true",) if same_draw else ()
    backend, completions = resume(tmp_path, modelspec, new_root, population, *packed, cap=RAISED,
                                  backend=StubbornBackend(hard))
    assert not same_draw or all(label.startswith("packed:") for label, _, _ in backend.blocks)
    assert judged_seeds(backend) and judged_seeds(backend) <= added_seeds(cap=RAISED)
    assert completions[1]["exhausted"] == [kept] and completions[1]["pending"] == 0
    assert completions[1]["state"] == "incomplete_exhausted"
    assert same_draw or completions[0]["state"] == STATE_COMPLETE
    assert not any(row["scenario_id"] == kept for row in read_jsonl(new_root / CONDITION / "draw-001" / LEDGER_FNAME)
                   if row["semantic_attempt_id"] >= CAP)
    assert check_retained(parent, new_root)["problems"] == []
    outcomes = check_added(parent, new_root, population=population, rule_sets=[rules])["outcomes"][added_id()]
    assert outcomes[f"{CONDITION} draw-00{r_draw}"] == "accepted at attempt 6"
    path = new_root / SAMPLING_MANIFEST_FNAME
    saved = path.read_bytes()
    manifest = json.loads(saved)
    manifest[MANIFEST_AMENDMENTS_KEY][0]["held"][0]["decoder_draw_id"] = 0
    path.write_text(json.dumps(manifest))
    unkept = (f"the rebased root's cap amendment neither holds at the old cap nor names bundle(s) ['{CONDITION}/draw-001 "
              f"{kept[:12]}'] that the parent's draws hold exhausted: a resume would judge a carried-over bundle again, "
              "unrecorded")
    assert check_retained(parent, new_root)["problems"] == [
        f"the rebased root's cap amendment holds bundle(s) ['{CONDITION}/draw-000 {kept[:12]}'] at the old cap that "
        "the parent's draws do not hold exhausted: only a carried-over bundle is held", unkept]
    with pytest.raises(SamplingDesignError, match=r"it holds \[\('[^']+', 0, '[0-9a-f]+'\)\] at the old cap, which "):
        campaign_module.campaign_amendment(new_root)
    del manifest[MANIFEST_AMENDMENTS_KEY][0]["held"]
    path.write_text(json.dumps(manifest))
    assert check_retained(parent, new_root)["problems"] == [unkept]
    with pytest.raises(SamplingDesignError, match=r"it neither holds at the old cap nor names \[\('[^']+', 1, "
                                                  r"'[0-9a-f]+'\)\], which the rebase carried over exhausted: a resume "
                                                  "of its draw would"):
        campaign_module.campaign_amendment(new_root)
    refused = StubbornBackend(hard)
    with pytest.raises(SamplingDesignError, match="which the rebase carried over exhausted"):
        resume(tmp_path, modelspec, new_root, population, *packed, cap=RAISED, backend=refused)
    assert not refused.blocks
    manifest[MANIFEST_AMENDMENTS_KEY][0]["held"] = [{"condition_id": CONDITION, "decoder_draw_id": 1,
                                                     "scenario_id": kept}]
    manifest[MANIFEST_AMENDMENTS_KEY][0]["bundles"].append(
        {**manifest[MANIFEST_AMENDMENTS_KEY][0]["bundles"][0], "decoder_draw_id": 1, "scenario_id": kept})
    path.write_text(json.dumps(manifest))
    with pytest.raises(SamplingDesignError, match=r"it both holds and names \[\('[^']+', 1, '[0-9a-f]+'\)\]"):
        campaign_module.campaign_amendment(new_root)
    path.write_bytes(saved)
    derived = tmp_path / "rescored"
    derive(new_root, derived)
    derived_manifest = json.loads((derived / SAMPLING_MANIFEST_FNAME).read_text())
    disclosed = disclosure(derived, derived_manifest)
    assert [(entry["decoder_draw_id"], entry["outcome"], entry["accepted_at_attempt"])
            for entry in disclosed["bundles"]] == [(r_draw, "accepted", 6)]
    assert derived_manifest["max_semantic_attempts"] == RAISED
    provenance = new_root / CONDITION / f"draw-00{r_draw}" / "draw_provenance.json"
    provenance.write_text(json.dumps({**json.loads(provenance.read_text()), "note": "edited"}))
    with pytest.raises(SamplingDesignError, match=r"\['draw_provenance.json'\] changed since the amendment"):
        disclosure(derived, derived_manifest)
