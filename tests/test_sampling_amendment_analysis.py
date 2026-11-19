from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.cli import main
from ruleforge.experiments import campaign_roots as cr
from ruleforge.evaluate.estimates import CampaignEstimatesError, estimates_source, load_estimates
from ruleforge.evaluate.inference import (
    RepeatedSamplingError,
    hierarchical_bootstrap,
    load_condition_outcomes,
    outcome_grid,
)
from ruleforge.evaluate.aggregate import exclusion_outcomes
from ruleforge.evaluate.amendment import AmendedBundle, CapAmendment
from ruleforge.evaluate.identity import MANIFEST_AMENDMENTS_KEY, SCENARIO_EXCLUSION_KEY
from ruleforge.core.files import write_json_sidecar
from ruleforge.core.record_names import identity_condition_id
from ruleforge.evaluate.runner import CampaignRunError
from ruleforge.evaluate.scoring import calculate_inconsistency
from ruleforge.evaluate.sharding import campaign_lock
from tests.test_leniency_campaign import N, PERMUTATIONS, S, artifact_root, write_rules
from tests.test_sampling_shards import edit_json, read_jsonl
from tests.support.configured import mark_configured
from tests.test_prompt_ablation_arms import GEMMA, NEW, OLMO, UNPRIVILEGED, read_only, write_condition

REPO = Path(__file__).resolve().parents[1]
OPEN = "fb-open"
URI = {OPEN: f"artifact://sampling/{OPEN}"}
#: Each judge's two arms: the naive prompt and its revision, both label styles (the separate-call rate).
FLAT_STYLE, REVISED_STYLE = "naive-rules-above", "naive"
#: The scenario Gemma's revised arm exhausts, by index, in draw 1.
EXHAUSTED_INDEX, EXHAUSTED_DRAW = 5, 1
ATTEMPTS = 20


def exhaust(root, condition_id, draw, scenario_id, attempts=ATTEMPTS):
    """Leave ``condition_id``'s draw as an exhausted bundle leaves it."""
    draw_dir = root / condition_id / f"draw-{draw:03d}"
    rows = read_jsonl(draw_dir / "attempts.jsonl")
    accepted = next(row for row in rows if row["scenario_id"] == scenario_id)
    invalid = {**accepted["components"][0], "valid": False, "failure_reason": "invalid_label", "judgment": None}
    rejected = [{**accepted, "semantic_attempt_id": attempt, "status": "rejected", "failure_reason": "component_invalid",
                 "request_block_id": f"retry-{attempt}", "components": [invalid]} for attempt in range(attempts)]
    kept = [row for row in rows if row["scenario_id"] != scenario_id]
    (draw_dir / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in kept + rejected))
    (draw_dir / "bundle_outcomes.jsonl").unlink()
    gate = json.loads((draw_dir / "completion.json").read_text())
    gate.update(state="incomplete_exhausted", materialized=False, exhausted=[scenario_id], pending=0, accepted=N - 1)
    (draw_dir / "completion.json").write_text(json.dumps(gate))
    provenance = json.loads((draw_dir / "draw_provenance.json").read_text())
    provenance["component_ids"] = [component["component_id"] for component in accepted["components"]]
    provenance.update({key: gate[key] for key in ("condition_fingerprint", "decoder_draw_id", "scenario_manifest_sha256")})
    (draw_dir / "draw_provenance.json").write_text(json.dumps(provenance))


def ids_of(root):
    return json.loads((root / "scenario_manifest.json").read_text())["scenario_ids"]


def rescore(root):
    """Each bundle's ``contradiction`` as the campaign scores it (``calculate_inconsistency``)."""
    for path in sorted(root.glob("*/*/draw-*/bundle_outcomes.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["contradiction"] = calculate_inconsistency(row["full_set_judgment"], list(row["rule_judgments"].values()))[
                "contradiction"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def grids(root, condition_ids, scenario_ids):
    """Each condition's grid from its complete draws, over ``scenario_ids``."""
    out = {}
    for cid in condition_ids:
        cells, _ = load_condition_outcomes(root / cid, expected_scenario_ids=ids_of(root), expected_draws=S)
        out[cid] = outcome_grid(cid, cells, scenario_ids=ids_of(root), n_draws=S,
                                identity_id=identity_condition_id(cid, cid.rsplit("/", 1)[-1])).restricted_to(scenario_ids)
    return out


GEMMA_ARMS = (f"{GEMMA}/{FLAT_STYLE}", f"{GEMMA}/{REVISED_STYLE}")
OLMO_ARMS = (f"{OLMO}/{FLAT_STYLE}", f"{OLMO}/{REVISED_STYLE}")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """The open root, Gemma's revised arm exhausted at one scenario; the grids its complete draws had."""
    base = tmp_path_factory.mktemp("fallback")
    with artifact_root(base):
        write_rules(base / "rules.jsonl")
        for model, seeds in ((GEMMA, (3, 14)), (OLMO, (5, 9))):
            write_condition(base, OPEN, model=model, style=FLAT_STYLE, seed=seeds[0], runtime=NEW)
            write_condition(base, OPEN, model=model, style=REVISED_STYLE, seed=seeds[1], runtime=NEW)
        root = base / "sampling" / OPEN
        rescore(root)
        ids = ids_of(root)
        excluded = ids[EXHAUSTED_INDEX]
        kept = [sid for sid in ids if sid != excluded]
        complete = {**grids(root, GEMMA_ARMS, kept), **grids(root, OLMO_ARMS, ids)}
        shutil.copytree(root / GEMMA_ARMS[1] / f"draw-{EXHAUSTED_DRAW:03d}", base / "complete-draw")
        exhaust(root, GEMMA_ARMS[1], EXHAUSTED_DRAW, excluded)
        from ruleforge.evaluate.aggregate import analyze_campaign

        payload = analyze_campaign(root, n_permutations=PERMUTATIONS, exclude_exhausted=(GEMMA, excluded))
    return {"base": base, "root": root, "excluded": excluded, "kept": kept, "grids": complete, "payload": payload}


@pytest.fixture
def env(study, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(study["base"]))
    return study


def test_the_fallback_drops_the_scenario_from_both_of_the_judge_s_arms_and_from_no_other(study):
    payload, kept, excluded = study["payload"], study["kept"], study["excluded"]
    assert payload["state"] == "incomplete_exhausted" and payload["excluded_conditions"] == {}
    # Gemma's two arms over N - 1 scenarios, Olmo's over N: each set its own bootstrap, the aggregate's.
    for arms, scenarios in ((GEMMA_ARMS, kept), (OLMO_ARMS, ids_of(study["root"]))):
        expected = hierarchical_bootstrap({cid: study["grids"][cid] for cid in arms}, n_resamples=200, ci_level=0.95,
                                          seed=16)["conditions"]
        for cid in arms:
            row = payload["conditions"][cid]
            assert {key: row[key] for key in expected[cid]} == expected[cid]
            assert row["n_scenarios"] == len(scenarios)
    record = payload[SCENARIO_EXCLUSION_KEY]
    assert (record["judge"], record["scenario_id"], record["n_scenarios_other_conditions"]) == (GEMMA, excluded, N)
    assert sorted(record["arms"]) == sorted(GEMMA_ARMS)
    fates = record["arms"][GEMMA_ARMS[1]]["excluded_scenario"]
    assert fates[str(EXHAUSTED_DRAW)] == {"state": "incomplete_exhausted", "attempts": ATTEMPTS}
    assert fates["0"] == {"state": "complete", "accepted_at_attempt": 0}
    assert record["arms"][GEMMA_ARMS[0]]["n_scenarios"] == N - 1
    assert payload["max_semantic_attempts"] == 20 and MANIFEST_AMENDMENTS_KEY not in payload


def copy_root(study, tmp_path):
    shutil.copytree(study["base"], tmp_path / "base")
    return tmp_path / "base", tmp_path / "base" / "sampling" / OPEN


def edit_gate(root, condition_id, draw, **changes):
    edit_json(root / condition_id / f"draw-{draw:03d}" / "completion.json", **changes)


def recover(base, root):
    """Gemma's exhausted draw of the revised arm made complete: the bundle accepted, as a resume that recovers it."""
    draw_dir = root / GEMMA_ARMS[1] / f"draw-{EXHAUSTED_DRAW:03d}"
    shutil.rmtree(draw_dir)
    shutil.copytree(base / "complete-draw", draw_dir)


def retry_more(root, condition_id, draw, scenario_id, until):
    """An exhausted bundle's draw given more rejected attempts, up to ``until`` (a resume under a raised cap)."""
    path = root / condition_id / f"draw-{draw:03d}" / "attempts.jsonl"
    last = [row for row in read_jsonl(path) if row["scenario_id"] == scenario_id][-1]
    with path.open("a") as handle:
        for attempt in range(last["semantic_attempt_id"] + 1, until):
            handle.write(json.dumps({**last, "semantic_attempt_id": attempt, "request_block_id": f"retry-{attempt}"})
                         + "\n")


def drop_rows(path, where):
    path.write_text("".join(json.dumps(row) + "\n" for row in read_jsonl(path) if not where(row)))


def accept_last_attempt(draws, excluded):
    """The exhausted draw's ledger accepting the excluded bundle's last attempt, its gate left saying exhausted."""
    path = draws / f"draw-{EXHAUSTED_DRAW:03d}" / "attempts.jsonl"
    rows = read_jsonl(path)
    donor = next(row for row in rows if row["status"] == "accepted")
    rows[-1] = {**rows[-1], "status": "accepted", "failure_reason": None, "components": donor["components"]}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize(
    ("judge", "scenario", "tamper", "message"),
    [
        (GEMMA, "other", None, r"scenario other is not the manifest's"),
        ("nobody", None, None, r"no condition of judge 'nobody' is registered"),
        (GEMMA, 0, None, r"\['gemma-4-31B-it/naive draw 1 \(incomplete_exhausted\)'\] are incomplete "
                         "otherwise than by that scenario alone"),
        (GEMMA, None, lambda root: edit_gate(root, GEMMA_ARMS[0], 2, state="in_progress"),
         r"\['gemma-4-31B-it/naive-rules-above draw 2 \(in_progress\)'\] are incomplete otherwise than by that scenario alone"),
        (GEMMA, None, lambda root: edit_gate(root, OLMO_ARMS[0], 2, state="in_progress"),
         r"and \['Olmo-3-32B-Think/naive-rules-above'\] are incomplete too"),
    ],
)
def test_the_fallback_is_refused_for_a_scenario_not_exhausted_or_any_other_incompleteness(
        study, tmp_path, judge, scenario, tamper, message):
    from ruleforge.evaluate.aggregate import analyze_campaign

    _, root = copy_root(study, tmp_path)
    if tamper is not None:
        tamper(root)
    sid = study["excluded"] if scenario is None else (ids_of(root)[scenario] if isinstance(scenario, int) else scenario)
    with pytest.raises(CampaignRunError, match=message):
        analyze_campaign(root, n_permutations=PERMUTATIONS, exclude_exhausted=(judge, sid))


def test_the_fallback_never_drops_a_scenario_no_draw_exhausted(tmp_path):
    from ruleforge.evaluate.aggregate import analyze_campaign

    base = tmp_path / "complete"
    base.mkdir()
    with artifact_root(base):
        write_rules(base / "rules.jsonl")
        write_condition(base, OPEN, model=GEMMA, style=FLAT_STYLE, seed=3, runtime=NEW)
        root = base / "sampling" / OPEN
        rescore(root)
        with pytest.raises(CampaignRunError, match=r"is out of attempts in no draw of \['gemma-4-31B-it/naive-rules-above'\]: "
                                                   "the fallback never drops a scenario that is not exhausted"):
            analyze_campaign(root, n_permutations=PERMUTATIONS, exclude_exhausted=(GEMMA, ids_of(root)[0]))


def test_the_fallback_leaves_another_incomplete_condition_to_allow_incomplete_conditions(study, tmp_path):
    from ruleforge.evaluate.aggregate import analyze_campaign

    _, root = copy_root(study, tmp_path)
    edit_gate(root, OLMO_ARMS[0], 2, state="in_progress")
    payload = analyze_campaign(root, n_permutations=PERMUTATIONS, exclude_exhausted=(GEMMA, study["excluded"]),
                               allow_incomplete_conditions=True)
    assert sorted(payload["conditions"]) == sorted([*GEMMA_ARMS, OLMO_ARMS[1]])
    assert list(payload["excluded_conditions"]) == [OLMO_ARMS[0]]


def test_the_fallback_serves_separate_call_arms_only(study, tmp_path):
    from ruleforge.evaluate.aggregate import analyze_campaign

    base, root = copy_root(study, tmp_path)
    with artifact_root(base):
        write_condition(base, OPEN, model=GEMMA, style="forced-grpo", seed=8, runtime=NEW)
    with pytest.raises(CampaignRunError, match=r"\['gemma-4-31B-it/forced-grpo'\] are forced-style or policy arms"):
        analyze_campaign(root, n_permutations=PERMUTATIONS, exclude_exhausted=(GEMMA, study["excluded"]))


def duplicate_row(path):
    rows = read_jsonl(path)
    path.write_text("".join(json.dumps(row) + "\n" for row in [*rows, {**rows[1], "contradiction": 0.0}]))


EXHAUSTED_DIR, COMPLETE_DIR = f"draw-{EXHAUSTED_DRAW:03d}", "draw-000"


@pytest.mark.parametrize(
    ("tamper", "cap", "error", "message"),
    [
        (lambda draws, sid: (draws / "draw-009").mkdir(), ATTEMPTS, CampaignRunError, "expected 3 draw directories"),
        (lambda draws, sid: edit_gate(draws, "", 0, decoder_draw_id=2), ATTEMPTS, CampaignRunError,
         "its gate is another draw's"),
        (lambda draws, sid: edit_gate(draws, "", 0, state="in_progress"), ATTEMPTS, CampaignRunError,
         r"'in_progress', not complete or exhausted"),
        # A complete draw is read as the strict reader reads it: a repeated row, the dropped scenario's row missing.
        (lambda draws, sid: duplicate_row(draws / COMPLETE_DIR / "bundle_outcomes.jsonl"), ATTEMPTS,
         RepeatedSamplingError, "appears twice"),
        (lambda draws, sid: drop_rows(draws / COMPLETE_DIR / "bundle_outcomes.jsonl", lambda row: row["scenario_id"] == sid),
         ATTEMPTS, RepeatedSamplingError, rf"outcomes cover {N - 1} scenarios, the manifest {N} \(1 missing"),
        # An exhausted draw: its ledger under the cap as it stands, its provenance, its accepted bundles.
        (lambda draws, sid: None, 50, CampaignRunError,
         r"its gate says .* is out of attempts, but its ledger has it at 20 of the campaign's 50: a gate written "
         "under another cap"),
        (lambda draws, sid: accept_last_attempt(draws, sid), ATTEMPTS, CampaignRunError, "its ledger has it accepted"),
        (lambda draws, sid: edit_json(draws / EXHAUSTED_DIR / "draw_provenance.json", decoder_draw_id=0), ATTEMPTS,
         CampaignRunError, "its provenance is another draw's"),
        (lambda draws, sid: edit_json(draws / EXHAUSTED_DIR / "draw_provenance.json", component_ids=["full_ruleset"]),
         ATTEMPTS, CampaignRunError, "accepted bundle holds other components than its provenance names"),
        (lambda draws, sid: drop_rows(draws / EXHAUSTED_DIR / "attempts.jsonl",
                                      lambda row: row["status"] == "accepted" and row["scenario_id"] != sid),
         ATTEMPTS, CampaignRunError, "its outcomes are not one per scenario of the manifest but"),
    ],
)
def test_an_arm_s_outcomes_without_the_scenario_are_refused_as_the_strict_reader_refuses(
        study, tmp_path, tamper, cap, error, message):
    _, root = copy_root(study, tmp_path)
    from ruleforge.evaluate.campaign import load_conditions

    spec = load_conditions(root)[GEMMA_ARMS[1]]
    tamper(root / GEMMA_ARMS[1], study["excluded"])
    manifest = json.loads((root / "scenario_manifest.json").read_text())["manifest_sha256"]
    with pytest.raises(error, match=message):
        exclusion_outcomes(root / GEMMA_ARMS[1], spec, campaign_id=OPEN, scenario_ids=study["kept"],
                           excluded=study["excluded"], expected_draws=S, scenario_manifest_sha256=manifest,
                           max_semantic_attempts=cap)


def test_the_estimates_of_a_fallback_are_refused_but_by_a_scratch_build(study):
    path = study["root"] / "aggregate" / "estimates.json"
    with pytest.raises(CampaignEstimatesError, match=rf"drop scenario {study['excluded'][:12]} from the arms of judge "
                                                     rf"'{GEMMA}' .*no paper build reads: .*Campaign\.open\(root\)"
                                                     r"\.result\(\).*--allow-incomplete"):
        load_estimates([path], allow_incomplete=False)
    [payload] = load_estimates([path], allow_incomplete=True)
    assert payload["state"] == "incomplete_exhausted"
    with artifact_root(study["base"]):
        source = estimates_source(payload)
    assert source[SCENARIO_EXCLUSION_KEY] == payload[SCENARIO_EXCLUSION_KEY] and MANIFEST_AMENDMENTS_KEY not in source


def test_estimates_source_names_the_retry_scope_its_estimates_record(study):
    payload = {"campaign_id": "c", "_path": study["root"] / "aggregate" / "estimates.json", "_sha256": "a" * 64}
    with artifact_root(study["base"]):
        assert "retry_scope" not in estimates_source(payload)
        assert estimates_source({**payload, "retry_scope": "component"})["retry_scope"] == "component"
        assert "retry_scope" not in estimates_source({**payload, "retry_scope": "scenario_bundle"})


def amendment_record(scenario_id):
    """A cap amendment of the root's Gemma revised arm, its bundle resumed in a shard (so not held here)."""
    bundle = AmendedBundle(condition_id=GEMMA_ARMS[1], decoder_draw_id=2, scenario_id=scenario_id, shard="shard-0-of-1",
                           attempts=20, prompts_sha256="0" * 64, files={})
    return CapAmendment(old_cap=20, new_cap=50, recorded_at="2026-10-10T00:00:00+00:00", decided_by="a maintainer",
                        decided_on="2026-10-10", reason="retry", ledger="note 132",
                        code_state=("b" * 40, False, "2" * 64), prior_code_states=(("a" * 40, False, "1" * 64),),
                        bundles=(bundle,), conditions=(*GEMMA_ARMS, *OLMO_ARMS), checks={"prompts": "match"}).to_dict()


def held_by_its_shard(root, condition_id, draw, *, num_shards):
    """``shard_merge.check_merged_draw`` for a synthetic root that keeps no shard roots."""
    from ruleforge.evaluate.amendment import _outcome
    from ruleforge.evaluate.ledger import DrawLedger

    state = DrawLedger(Path(root) / condition_id / f"draw-{draw:03d}", campaign_id=OPEN, condition_id=condition_id,
                       decoder_draw_id=draw).load()
    return {"amended": {f"shard-0-of-{num_shards}": {"bundles": {sid: _outcome(state, sid, 50)
                                                                 for sid in state.accepted}}}}


def test_an_amended_root_s_analyses_disclose_the_amendment(study, tmp_path, monkeypatch, capsys):
    from ruleforge.evaluate import shard_merge
    from ruleforge.evaluate.aggregate import analyze_campaign

    monkeypatch.setattr(shard_merge, "check_merged_draw", held_by_its_shard)
    base, root = copy_root(study, tmp_path)
    manifest = json.loads((root / "sampling_manifest.json").read_text())
    sid = ids_of(root)[2]
    record = amendment_record(sid)
    # the whole config a campaign job's manifest records, so the command reads the root as any configured root
    whole = {"retry_scope": "scenario_bundle", "scenarios_per_request_block": None, "label_rule": "bare"}
    (root / "sampling_manifest.json").write_text(json.dumps({**manifest, **whole, "max_semantic_attempts": 50,
                                                             "generation_seed_base": 16,
                                                             MANIFEST_AMENDMENTS_KEY: [record]}))
    runs = root / GEMMA_ARMS[1] / "draw-002" / "runs.jsonl"
    runs.write_text(json.dumps({"code_revision": "b" * 40, "code_dirty": False, "code_state_sha256": "2" * 64}) + "\n")
    # Under the raised cap the fallback reads a draw exhausted at 50, not at 20 (a resume's 30 more attempts).
    retry_more(root, GEMMA_ARMS[1], EXHAUSTED_DRAW, study["excluded"], 50)
    mark_configured(root)  # a root the configured runner made: `ruleforge analyze` writes its analysis again
    assert main(["analyze", str(root), "--n-permutations", str(PERMUTATIONS), "--exclude-exhausted", GEMMA,
                 study["excluded"], "--replace"]) == 0
    assert json.loads(capsys.readouterr().out)["scenario_exclusion"]["judge"] == GEMMA
    [disclosed] = json.loads((root / "aggregate" / "estimates.json").read_text())[MANIFEST_AMENDMENTS_KEY]
    assert (disclosed["old_cap"], disclosed["new_cap"], disclosed["code_state_described"]) == (
        20, 50, "commit bbbbbbbbbbbb, sources 222222222222")
    assert disclosed["bundles"] == [{"condition_id": GEMMA_ARMS[1], "decoder_draw_id": 2, "scenario_id": sid,
                                     "held_by": "shard-0-of-1", "attempts_at_old_cap": 20, "outcome": "accepted",
                                     "accepted_at_attempt": 0, "attempts": 1}]
    assert disclosed["code_states"] == {GEMMA_ARMS[1]: {"commit bbbbbbbbbbbb, sources 222222222222": ["draw-002"]}}
    payload = analyze_campaign(root, n_permutations=PERMUTATIONS, exclude_exhausted=(GEMMA, study["excluded"]))
    fates = payload[SCENARIO_EXCLUSION_KEY]["arms"][GEMMA_ARMS[1]]["excluded_scenario"]
    assert fates[str(EXHAUSTED_DRAW)] == {"state": "incomplete_exhausted", "attempts": 50}
    fates[str(EXHAUSTED_DRAW)]["attempts"] = ATTEMPTS
    assert {key: value for key, value in payload.items() if key not in (MANIFEST_AMENDMENTS_KEY, "created_at")} == {
        **{key: value for key, value in study["payload"].items() if key != "created_at"}, "max_semantic_attempts": 50}


def complete_copy(study, tmp_path):
    """A copy of the open root with Gemma's exhausted draw recovered, analysed again without the fallback."""
    from ruleforge.evaluate.aggregate import analyze_campaign

    base, root = copy_root(study, tmp_path)
    recover(base, root)
    with artifact_root(base):
        analyze_campaign(root, n_permutations=PERMUTATIONS)
    return base, root


def open_arms():
    """The open root as the prompt ablation's arm reader opens it (``open_roots``)."""
    return cr.open_roots({OPEN: URI[OPEN]}, "arm reader", allow_incomplete=False, allow_stack_mismatch=False)


def test_prompt_table_s_arm_reader_refuses_estimates_written_under_the_fallback(env):
    with pytest.raises(CampaignEstimatesError, match=rf"drop scenario {env['excluded'][:12]} from the arms of judge "
                                                     rf"'{GEMMA}'"):
        open_arms()


@pytest.mark.parametrize("amended", [True, False])
def test_estimates_older_than_their_root_s_cap_are_refused(study, tmp_path, monkeypatch, amended):
    """Estimates written at cap 20, then the root's cap raised (by an amendment, or edited), not re-analysed."""
    base, root = complete_copy(study, tmp_path)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(base))
    [opened] = open_arms().values()
    assert opened.scenario_ids == tuple(ids_of(root))
    record = {MANIFEST_AMENDMENTS_KEY: [amendment_record(ids_of(root)[2])]} if amended else {}
    edit_json(root / "sampling_manifest.json", max_semantic_attempts=50, **record)
    with pytest.raises(cr.CampaignRootError, match=r"its estimates were analysed at max_semantic_attempts 20, its sampling "
                                             r"manifest records 50 now.*re-run analyze_campaign on it\), or "
                                             rf"{re.escape(cr.CHANGED_WHILE_READ)}"):
        open_arms()


def test_the_estimates_digest_recorded_is_of_the_bytes_parsed(study, tmp_path):
    base, root = copy_root(study, tmp_path)
    path = root / "aggregate" / "estimates.json"
    read = hashlib.sha256(path.read_bytes()).hexdigest()
    [payload] = load_estimates([path], allow_incomplete=True)
    write_json_sidecar(path, {**json.loads(path.read_text()), "created_at": "2026-10-11T00:00:00Z"})
    with artifact_root(base):
        source = estimates_source(payload)
    assert (source["estimates_sha256"], source["created_at"]) == (read, study["payload"]["created_at"])
    assert read != hashlib.sha256(path.read_bytes()).hexdigest()


@UNPRIVILEGED
def test_the_readers_read_a_read_only_root_as_a_writable_one(study, tmp_path, monkeypatch):
    base, root = copy_root(study, tmp_path)
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(base))

    def read():
        [payload] = load_estimates([root / "aggregate" / "estimates.json"], allow_incomplete=True)
        return json.dumps(estimates_source(payload), sort_keys=True)

    files = sorted(base.rglob("*"))
    writable = read()
    with read_only(base / "sampling"):
        assert read() == writable
    assert sorted(base.rglob("*")) == files
    with campaign_lock(root):
        assert read() == writable
