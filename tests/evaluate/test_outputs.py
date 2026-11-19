from __future__ import annotations

import json
import shutil

import pytest

from ruleforge.core.digests import sha256_file
from ruleforge.core.files import write_json_sidecar
from ruleforge.core.paths import artifact_uri_or_path
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate.campaign import build_plan, load_conditions, run_plan
from ruleforge.evaluate.outputs import CampaignRoot, TokenReadoutError, draw_outputs, load_rules, raw_rows
from ruleforge.judges.backend import GenerationOutcome
from ruleforge.scenarios.manifest import ScenarioManifest
from tests.test_sampling_campaign import backend_for, compose_campaign
from tests.test_sampling_runner import forced, read_jsonl

FORCED = "gemma-4-31B-it/forced-grpo"
TRUNCATED = GenerationOutcome(text=None, raw_response="<think> cut", finish_reason="length")
UNCLOSED = GenerationOutcome(text=forced("no", {"Rule 1": "no"}), raw_response=forced("no", {"Rule 1": "no"}),
                             finish_reason="stop")
TRANSPORT = GenerationOutcome(text=None, raw_response=None, finish_reason=None, error="HTTP 500")
SCRIPT = {
    (0, "full_ruleset"): ["not json", forced("yes", {"Rule 1": "yes", "Rule 2": "no"})],
    (1, "full_ruleset"): [TRUNCATED, forced("no")],
    (2, "full_ruleset"): [UNCLOSED],
    (2, "Rule 1"): [TRANSPORT, forced("no")],
}


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(campaign_module, "code_state", lambda: {})
    monkeypatch.setattr(campaign_module, "write_code_snapshot", lambda root, state: None)


@pytest.fixture
def forced_root(tmp_path, modelspec):
    cfg = compose_campaign(tmp_path, **{"eval_cfg.reasoning_prompt_style": "forced-grpo", "evaluation_mode": "both",
                                        "sampling.forced_json_requirement": "label_only"})
    backend = backend_for("forced-grpo", script=SCRIPT)
    task_spec = modelspec(reasoning_prompt_style="forced-grpo", evaluation_temperature=1.0)
    run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    return tmp_path / "root"


def _raw_path(root):
    return root / FORCED / "draw-000" / "raw_outputs.jsonl"


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_a_conditions_outputs_carry_their_ledger_rows(forced_root):
    campaign = CampaignRoot.open(forced_root)
    assert (campaign.campaign_id, campaign.n_draws) == ("camp", 2)
    outputs, orphans = campaign.outputs(FORCED)
    first = [o for o in outputs if o.first_attempt and o.kind == "full_set"]
    manifest_ids = set(campaign.scenario_manifest.scenario_ids)
    assert orphans == 0 and len(outputs) == 26 and len(first) == 6
    assert {(o.scenario_id, o.decoder_draw_id) for o in first} == {(sid, d) for sid in manifest_ids for d in (0, 1)}
    assert all(o.component_id == "full_ruleset" and o.record.status in ("accepted", "rejected") for o in first)
    truncated = [o for o in first if o.finish_reason == "length"]
    assert [(o.raw_response, o.outcome().text) for o in truncated] == [("<think> cut", None)]
    assert all(o.raw_response == (o.row.get("raw_response") or "") for o in outputs)
    with pytest.raises(TokenReadoutError, match="registers no condition"):
        campaign.outputs("gemma-4-31B-it/naive")


def test_the_campaign_record_names_every_input_by_digest(forced_root, tmp_path):
    record = CampaignRoot.open(forced_root).record()
    draw = forced_root / FORCED / "draw-000"
    assert record["root"] == artifact_uri_or_path(forced_root) and record["state"] == "complete"
    assert record["conditions"][FORCED]["draws"]["0"] == {
        "state": "complete",
        "attempts.jsonl": sha256_file(draw / "attempts.jsonl"),
        "raw_outputs.jsonl": sha256_file(draw / "raw_outputs.jsonl"),
    }
    assert record["sampling_manifest_sha256"] == sha256_file(forced_root / "sampling_manifest.json")
    # The content digest ignores where the root lives, and follows every ledger byte; a draw not run has none.
    before = record["content_sha256"]
    for name in ("attempts.jsonl", "raw_outputs.jsonl"):
        copy = tmp_path / f"copy-{name}"
        shutil.copytree(forced_root, copy)
        assert CampaignRoot.open(copy).record()["content_sha256"] == before
        edited = copy / FORCED / "draw-001" / name
        edited.write_text(edited.read_text() + "\n")
        assert CampaignRoot.open(copy).record()["content_sha256"] != before, name
    unrun = tmp_path / "unrun"
    shutil.copytree(forced_root, unrun)
    shutil.rmtree(unrun / FORCED / "draw-001")
    assert CampaignRoot.open(unrun).record()["conditions"][FORCED]["draws"]["1"]["attempts.jsonl"] is None


def test_the_campaign_manifest_is_recorded_by_the_digest_of_the_bytes_read(forced_root):
    path = forced_root / "sampling_manifest.json"
    read = sha256_file(path)
    campaign = CampaignRoot.open(forced_root)
    write_json_sidecar(path, {**json.loads(path.read_text()), "max_semantic_attempts": 50})
    assert campaign.record()["sampling_manifest_sha256"] == read != sha256_file(path)


def test_an_unvouched_raw_row_is_counted_and_left_out(forced_root):
    rows = read_jsonl(_raw_path(forced_root))
    _write(_raw_path(forced_root), [*rows, {**rows[0], "semantic_attempt_id": 9}])
    outputs, orphans = CampaignRoot.open(forced_root).outputs(FORCED)
    assert orphans == 1 and len(outputs) == 26


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda row: {**row, "raw_response": row["raw_response"] + " "}, "does not hash to the digest"),
        (lambda row: {**row, "raw_output_sha256": "0" * 64}, "does not hash to the digest"),
        (lambda row: {**row, "component_id": "Rule 9"}, "no component of its ledger row"),
        # finish_reason is outside the digest, and it sets the stop token and the truncation split.
        (lambda row: {**row, "finish_reason": "length" if row["finish_reason"] != "length" else "stop"},
         "finish reason"),
    ],
)
def test_a_raw_row_the_ledger_does_not_vouch_for_as_written_is_refused(forced_root, edit, message):
    rows = read_jsonl(_raw_path(forced_root))
    rows[0] = edit(rows[0])
    _write(_raw_path(forced_root), rows)
    with pytest.raises(TokenReadoutError, match=message):
        CampaignRoot.open(forced_root).outputs(FORCED)


def test_an_output_the_ledger_vouches_for_without_its_raw_row_is_refused(forced_root):
    rows = read_jsonl(_raw_path(forced_root))
    _write(_raw_path(forced_root), rows[1:])
    with pytest.raises(TokenReadoutError, match="1 output\\(s\\) its ledger vouches for have no raw row"):
        CampaignRoot.open(forced_root).outputs(FORCED)
    _raw_path(forced_root).unlink()
    assert list(raw_rows(_raw_path(forced_root))) == []
    with pytest.raises(TokenReadoutError, match="have no raw row"):
        CampaignRoot.open(forced_root).outputs(FORCED)


def test_a_torn_final_raw_line_is_an_orphan_and_a_malformed_line_before_it_is_refused(forced_root):
    text = _raw_path(forced_root).read_text()
    # A blank line is skipped; a final line cut short is an append the job died in.
    _raw_path(forced_root).write_text(text + "\n" + '{"scenario_id": "ab')
    campaign = CampaignRoot.open(forced_root)
    [(ledger, state, outputs, orphans)] = [draw for draw in campaign.draws(FORCED) if draw[0].decoder_draw_id == 0]
    assert orphans == 1 and draw_outputs(ledger, state) == (outputs, 1)
    _raw_path(forced_root).write_text('{"scenario_id": "ab\n' + text)
    with pytest.raises(TokenReadoutError, match="line 1 is not a JSON row"):
        campaign.outputs(FORCED)


def test_a_second_raw_row_for_one_output_is_refused(forced_root):
    rows = read_jsonl(_raw_path(forced_root))
    _write(_raw_path(forced_root), [*rows, rows[-1]])
    with pytest.raises(TokenReadoutError, match="a second raw row"):
        CampaignRoot.open(forced_root).outputs(FORCED)


def test_the_ruleset_must_be_the_conditions_and_the_scenario_manifest_the_campaigns(forced_root, tmp_path):
    condition = load_conditions(forced_root)[FORCED]
    assert load_rules(tmp_path / "rules.jsonl", condition) == read_jsonl(tmp_path / "rules.jsonl")
    other = tmp_path / "other_rules.jsonl"
    other.write_text(json.dumps({"id": "Rule 1", "rule": "another"}) + "\n")
    with pytest.raises(TokenReadoutError, match="judged ruleset"):
        load_rules(other, condition)
    elsewhere = tmp_path / "elsewhere.jsonl"
    elsewhere.write_text(json.dumps({"prompt": "p", "response": "r"}) + "\n")
    ScenarioManifest.build(elsewhere).write(forced_root / "scenario_manifest.json")
    with pytest.raises(TokenReadoutError, match="not the one the campaign manifest froze"):
        CampaignRoot.open(forced_root)
