from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from ruleforge.core.digests import request_namespace
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.record_names import identity_key, locate_condition, public_condition_id
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import CONDITION_FNAME, build_plan, load_conditions, run_plan
from ruleforge.evaluate.identity import SAMPLING_MANIFEST_FNAME, ConditionSpec, derive_generation_seed
from ruleforge.evaluate.ledger import LEDGER_FNAME
from ruleforge.prompts.styles import ENCOURAGED, FORCED, FORCED_JSON_STYLES, NAIVE_RULES_ABOVE
from tests.support.recorded_root import Recording, _Mapper, record, recorded_condition, recorded_payload
from tests.test_sampling_campaign import backend_for, compose_campaign
from tests.test_sampling_runner import N_SCENARIOS

#: The paper's three prompts, the prompt ablation's arms.
STYLES = (NAIVE_RULES_ABOVE, ENCOURAGED, FORCED)
#: The judge the campaign composes (``configs/sampling_campaign.yaml``'s default model).
MODEL = "gemma-4-31B-it"
#: What the aggregate is computed with, the same for the run and for its recorded copy.
ANALYSIS = {"n_permutations": 100, "n_resamples": 100}
ESTIMATES = Path("aggregate") / "estimates.json"


@pytest.fixture(autouse=True)
def _artifact_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))


def _answer(style: str, verdict: str, rule_1: str) -> str:
    """A judge's answer under ``style``: a bare label, or the forced prompt's rule map and verdict."""
    if style in FORCED_JSON_STYLES:
        return json.dumps({"rule_judgments": {"Rule 1": rule_1, "Rule 2": "no"}, "full_set_judgment": verdict})
    return verdict


def _script(style: str, offset: int) -> dict:
    """Answers that vary over scenarios, draws and arms, so no rate, interval or contrast is constant."""
    script = {}
    for index in range(N_SCENARIOS):
        verdicts = ["no" if (index + draw + offset) % 2 == 0 else "yes" for draw in range(2)]
        rules = ["yes" if (index + draw + offset) % 3 else "no" for draw in range(2)]
        script[(index, "full_ruleset")] = [_answer(style, v, r) for v, r in zip(verdicts, rules)]
        script[(index, "Rule 1")] = [_answer(style, r, r) for r in rules]
    return script


@pytest.fixture
def roots(tmp_path, modelspec) -> tuple[Path, Path, Recording]:
    """The three arms run and aggregated under the public names, and their recorded copy."""
    for offset, style in enumerate(STYLES):
        cfg = compose_campaign(tmp_path, **{"eval_cfg.reasoning_prompt_style": style})
        backend = backend_for(style, script=_script(style, offset))
        task_spec = modelspec(reasoning_prompt_style=style, evaluation_temperature=1.0)
        run_plan(build_plan(cfg, backend=backend, task_spec=task_spec), backend=backend, task_spec=task_spec)
    public = tmp_path / "root"
    analyze_campaign(public, **ANALYSIS)
    recorded = tmp_path / "recorded"
    shutil.copytree(public, recorded)
    return public, recorded, record(recorded)


def _public_id(style: str) -> str:
    return f"{MODEL}/{style}"


def _ledger(root: Path, condition_id: str, draw: str) -> list[dict]:
    return [json.loads(line) for line in (root / condition_id / draw / LEDGER_FNAME).read_text().splitlines()]


def test_the_record_names_each_arm_and_style_by_its_recorded_key(roots):
    public, recorded, recording = roots
    assert recording.ids == {_public_id(style): f"{MODEL}/{identity_key(style)}" for style in STYLES}
    assert recording.styles == {style: identity_key(style) for style in STYLES}
    assert set(recording.ids.values()).isdisjoint(recording.ids)
    for public_id, recorded_id in recording.ids.items():
        assert not (recorded / public_id).exists()
        assert sorted(path.name for path in (recorded / recorded_id).iterdir()) == sorted(
            path.name for path in (public / public_id).iterdir())
    for name in ("scenario_manifest.json", SAMPLING_MANIFEST_FNAME):
        assert (recorded / name).read_bytes() == (public / name).read_bytes()
    for path in recorded.rglob("*.json*"):
        text = path.read_text(encoding="utf-8")
        assert not any(f'"{public_id}' in text or f"/{public_id}/" in text for public_id in recording.ids), path
        assert not any(f'"prompt_style": "{style}"' in text for style in STYLES), path


def test_a_recorded_root_loads_under_its_recorded_ids_with_the_public_styles(roots):
    public, recorded, recording = roots
    ran, read = load_conditions(public), load_conditions(recorded)
    assert set(read) == set(recording.ids.values()) and set(ran) == set(recording.ids)
    for public_id, recorded_id in recording.ids.items():
        twin, spec = ran[public_id], read[recorded_id]
        assert dataclasses.replace(spec, condition_id=public_id) == twin
        assert spec.prompt_style in STYLES and spec.prompt_style == twin.prompt_style
        assert spec.fingerprint == twin.fingerprint
        assert spec.identity_id == recorded_id == twin.identity_id
        assert spec.public_id == public_id == twin.public_id
        assert public_condition_id(recorded_id, spec.prompt_style) == public_id
        payload = json.loads((recorded / recorded_id / CONDITION_FNAME).read_text())
        assert payload == recorded_payload(twin) and recorded_condition(payload) == spec


def test_every_seed_and_request_namespace_is_the_one_the_run_derived(roots):
    public, recorded, recording = roots
    manifest = json.loads((public / SAMPLING_MANIFEST_FNAME).read_text())
    read = load_conditions(recorded)
    seeds = 0
    for public_id, recorded_id in recording.ids.items():
        spec = read[recorded_id]
        for draw in sorted(path.name for path in (recorded / recorded_id).glob("draw-*")):
            rows, ran = _ledger(recorded, recorded_id, draw), _ledger(public, public_id, draw)
            assert len(rows) == len(ran) > 0
            for row, twin in zip(rows, ran):
                assert row["condition_id"] == recorded_id and twin["condition_id"] == public_id
                assert row["request_block_id"] == twin["request_block_id"].replace(
                    f"/{public_id}/", f"/{recorded_id}/", 1)
                assert request_namespace(spec.identity_block_id(manifest["campaign_id"], row["request_block_id"])) == (
                    request_namespace(dataclasses.replace(spec, condition_id=public_id).identity_block_id(
                        manifest["campaign_id"], twin["request_block_id"])))
                for component, other in zip(row["components"], twin["components"], strict=True):
                    assert component["generation_seed"] == other["generation_seed"] == derive_generation_seed(
                        generation_seed_base=manifest["generation_seed_base"], campaign_id=row["campaign_id"],
                        condition_id=spec.identity_id, scenario_id=row["scenario_id"],
                        decoder_draw_id=row["decoder_draw_id"], semantic_attempt_id=row["semantic_attempt_id"],
                        component_id=component["component_id"])
                    seeds += 1
    assert seeds


def _aggregates(roots) -> tuple[dict, dict, dict]:
    """The run's estimates, the recorded copy's aggregated again, and the copy's as recorded; no ``created_at``."""
    public, recorded, recording = roots
    as_recorded = json.loads((recorded / ESTIMATES).read_text())
    again = analyze_campaign(recorded, **ANALYSIS)
    assert again == json.loads((recorded / ESTIMATES).read_text())
    ran = _Mapper(recording.ids, {}).value(json.loads((public / ESTIMATES).read_text()))
    for payload in (ran, again, as_recorded):
        payload.pop("created_at")
    return ran, again, as_recorded


def _without_two_stage(value):
    if isinstance(value, dict):
        return {key: _without_two_stage(item) for key, item in value.items() if key != "two_stage"}
    if isinstance(value, list):
        return [_without_two_stage(item) for item in value]
    return value


def test_the_recorded_copy_aggregates_again_to_its_estimates_but_for_the_style_names(roots):
    ran, again, as_recorded = _aggregates(roots)
    assert _without_two_stage(again) == _without_two_stage(ran)
    _, _, recording = roots
    assert _without_two_stage(_Mapper({}, recording.styles).value(again)) == _without_two_stage(as_recorded)
    assert {entry["prompt_style"] for entry in again["conditions"].values()} == set(STYLES)
    assert {entry["grid_digest"] for entry in again["conditions"].values()} == {
        entry["grid_digest"] for entry in ran["conditions"].values()}
    assert len({entry["estimate"] for entry in again["conditions"].values()}) > 1
    assert any(entry["two_stage"]["sd"] > 0 for entry in again["conditions"].values()), "the arms vary"


def test_a_public_run_s_two_stage_intervals_are_the_recorded_ones(roots):
    ran, again, _ = _aggregates(roots)
    assert again == ran


def test_a_configuration_s_public_id_finds_the_recorded_condition_and_an_output_names_it_publicly(roots):
    """The prompt ablation's arm map names each arm by its public id (``gemma-4-31B-it/naive``)."""
    public, recorded, recording = roots
    read, ran = load_conditions(recorded), load_conditions(public)
    for public_id, recorded_id in recording.ids.items():
        assert locate_condition(read, public_id) is read[recorded_id]
        assert locate_condition(read.values(), public_id).condition_id == recorded_id
        assert locate_condition(ran, public_id) is ran[public_id]
    assert sorted(spec.public_id for spec in read.values()) == sorted(ran)
    with pytest.raises(SamplingDesignError, match="no registered condition has the public id"):
        locate_condition(read, f"{MODEL}/forced-grpo")
    with pytest.raises(SamplingDesignError, match="no registered condition has the public id"):
        locate_condition(read, recording.ids[_public_id(NAIVE_RULES_ABOVE)])


def test_the_record_maps_whole_ids_resorts_what_was_written_sorted_and_maps_only_style_fields():
    mapper = _Mapper({"m/forced": "m/z0", "m/naive": "m/a0"}, {"naive": "a0"})
    assert mapper.string("camp/m/naive/draw-000/attempt-0/block-0000") == "camp/m/a0/draw-000/attempt-0/block-0000"
    assert mapper.string("rule_context:m/naive") == "rule_context:m/a0"
    assert mapper.string("artifact://sampling/r/m/naive/draw-000") == "artifact://sampling/r/m/a0/draw-000"
    assert [mapper.string(text) for text in ("m/naive-hierarchy", "xm/naive", "m/naives", "naive")] == [
        "m/naive-hierarchy", "xm/naive", "m/naives", "naive"]
    assert mapper.string("m/forced+m/naive") == "m/a0+m/z0"
    assert mapper.string("m/naive+m/forced") == "m/a0+m/z0"
    assert mapper.string("m/naive+x/forced") == "m/a0+x/forced"
    value = mapper.value({"m/forced": {"prompt_style": "forced"}, "m/naive": {"prompt_style": "naive",
                                                                             "label": "naive"},
                          "members": ["m/naive", "m/forced"]})
    assert list(value) == ["m/a0", "m/z0", "members"]
    assert value["m/a0"] == {"prompt_style": "a0", "label": "naive"} and value["m/z0"] == {"prompt_style": "forced"}
    assert value["members"] == ["m/a0", "m/z0"]
    assert list(mapper.value({"m/naive": 1, "m/forced": 2})) == ["m/a0", "m/z0"]
    assert list(mapper.value({"x": 1, "m/naive": 2})) == ["x", "m/a0"]
    assert _Mapper({}, {}).string("m/naive") == "m/naive"
    with pytest.raises(ValueError, match="two keys of one object are recorded as one"):
        _Mapper({"m/a": "m/b"}, {}).value({"m/a": 1, "m/b": 2})


def test_the_record_refuses_a_file_it_cannot_write_back_and_a_style_it_cannot_tell_apart(roots, tmp_path):
    public, _, _ = roots
    copy = tmp_path / "unreadable"
    shutil.copytree(public, copy)
    (copy / "notes.json").write_text('{"a":  1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="notes.json: no encoding of record"):
        record(copy)
    assert (copy / _public_id(NAIVE_RULES_ABOVE)).is_dir(), "nothing is changed before a refusal"
    mixed = tmp_path / "mixed"
    shutil.copytree(public, mixed)
    spec = load_conditions(public)[_public_id(NAIVE_RULES_ABOVE)]
    other = ConditionSpec.from_dict({**spec.to_dict(), "condition_id": "theirs/naive", "task": "demo@1",
                                     "response_contract": "label"})
    path = mixed / other.condition_id / CONDITION_FNAME
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 2, **other.to_dict(), "fingerprint": other.fingerprint}, indent=2)
                    + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is recorded both as"):
        record(mixed)
