from __future__ import annotations

import json
from pathlib import Path

import pytest

from ruleforge.evaluate import identity as design
from ruleforge.scenarios.manifest import (
    SAMPLING_SCHEMA_VERSION,
    ScenarioManifest,
)
from ruleforge.evaluate.identity import (
    BootstrapSettings,
    ConditionSpec,
    GridCheck,
    SamplingConfig,
    build_sampling_manifest,
    check_exact_grid,
    derive_generation_seed,
    draw_dirname,
    ensure_sampling_manifest,
    load_sampling_manifest,
    request_block_id,
)
from ruleforge.core.errors import SamplingDesignError
from ruleforge.core.digests import (
    canonical_digest,
    canonical_json,
    request_namespace,
    sha256_file,
)
from ruleforge.core.identifiers import validate_identifier
from ruleforge.core.artifacts import ArtifactSchemaError, compute_scenario_id


def write_scenarios(path: Path, n: int) -> list[dict]:
    rows = [{"prompt": f"prompt {i}", "response": f"response {i}"} for i in range(n)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows


def make_condition(**overrides) -> ConditionSpec:
    fields = dict(
        condition_id="gemma/forced-grpo",
        model_name="gemma-4-31B-it",
        model_type="local",
        prompt_style="forced-grpo",
        individual_rule_context="isolated",
        judge_type="No_Interpretation",
        evaluation_mode="both",
        rule_set_name="openai_rules_all.jsonl",
        rule_set_sha256="a" * 64,
        temperature=1.0,
        prompt_shape_version=2,
        forced_json_requirement="complete_rule_map",
    )
    fields.update(overrides)
    return ConditionSpec(**fields)


def test_canonical_json_is_order_free_and_unescaped():
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert canonical_digest({"b": 1, "a": "é"}) == canonical_digest({"a": "é", "b": 1})


def test_sha256_file_streams_the_bytes(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize("value", ["", "-lead", "has space", "a/b", 7, None, "x" * 129])
def test_validate_identifier_refuses_what_cannot_be_a_path_segment(value):
    with pytest.raises(SamplingDesignError, match="each segment matching"):
        validate_identifier("campaign_id", value)


def test_validate_identifier_returns_a_valid_id():
    assert validate_identifier("campaign_id", "ruleset-size.pilot_2026-09") == "ruleset-size.pilot_2026-09"


@pytest.mark.parametrize("value", ["gemma/naive", "a/b/c", "solo"])
def test_a_condition_id_may_carry_path_segments(value):
    assert validate_identifier("condition_id", value, allow_slashes=True) == value


@pytest.mark.parametrize("value", ["/lead", "a//b", "a/../b", "a/", 3])
def test_a_condition_id_segment_must_still_be_an_identifier(value):
    with pytest.raises(SamplingDesignError, match="condition_id"):
        validate_identifier("condition_id", value, allow_slashes=True)


def test_defaults_are_the_contracts_planning_values():
    config = SamplingConfig()
    assert config.temperature == 1.0
    assert config.num_decoder_draws == 5
    assert config.max_semantic_attempts == 5
    assert config.retry_scope == "scenario_bundle"
    assert config.bootstrap.n_resamples == 10_000
    assert config.bootstrap.ci_level == 0.95
    assert config.expected_scenario_count == 500
    assert SamplingConfig(campaign_mode="pilot").expected_scenario_count is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 3}, "schema_version"),
        ({"campaign_mode": "smoke"}, "campaign_mode"),
        ({"temperature": True}, "must be a number"),
        ({"temperature": "1.0"}, "must be a number"),
        ({"temperature": 0.7}, "exactly 1.0"),
        ({"num_decoder_draws": 1}, "num_decoder_draws must be at least 2"),
        ({"num_decoder_draws": True}, "num_decoder_draws must be an integer"),
        ({"max_semantic_attempts": 0}, "max_semantic_attempts"),
        ({"retry_scope": "per_rule"}, "retry_scope"),
        ({"strict_think": 1}, "strict_think must be true or false"),
        ({"require_complete": False}, "require_complete"),
        ({"generation_seed_base": -1}, "generation_seed_base"),
        ({"scenarios_per_request_block": 0}, "scenarios_per_request_block"),
        ({"forced_json_requirement": "lenient"}, "forced_json_requirement"),
        ({"bootstrap": {"seed": 1}}, "must be BootstrapSettings"),
    ],
)
def test_config_refuses_a_different_experiment(overrides, message):
    with pytest.raises(SamplingDesignError, match=message):
        SamplingConfig(**overrides)


def test_config_accepts_the_component_scope_and_strict_think():
    config = SamplingConfig(campaign_mode="pilot", retry_scope="component", strict_think=True)
    assert (config.retry_scope, config.strict_think) == ("component", True)
    assert SamplingConfig().strict_think is False
    assert SamplingConfig.from_mapping(
        {"campaign_mode": "pilot", "retry_scope": "component", "strict_think": True}
    ) == config


def test_config_accepts_a_pilot_with_request_blocks_and_int_temperature():
    config = SamplingConfig(
        campaign_mode="pilot", scenarios_per_request_block=25, temperature=1
    )
    assert config.scenarios_per_request_block == 25
    assert config.to_dict()["bootstrap"] == {
        "n_resamples": 10_000, "ci_level": 0.95, "seed": 16,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": 0}, "n_resamples must be at least 1"),
        ({"n_resamples": 50}, "n_resamples must be at least 100"),
        ({"ci_level": 1.5}, "ci_level"),
        ({"seed": -3}, "bootstrap.seed"),
    ],
)
def test_bootstrap_settings_are_validated(kwargs, message):
    with pytest.raises(SamplingDesignError, match=message):
        BootstrapSettings(**kwargs)


def test_from_mapping_reads_a_config_block():
    config = SamplingConfig.from_mapping(
        {
            "enabled": True,
            "campaign_mode": "pilot",
            "num_decoder_draws": 3,
            "bootstrap": {"n_resamples": 200, "seed": 7},
        }
    )
    assert config.num_decoder_draws == 3
    assert config.bootstrap == BootstrapSettings(n_resamples=200, seed=7)
    assert SamplingConfig.from_mapping({}) == SamplingConfig()


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (["not", "a", "mapping"], "must be a mapping"),
        ({"num_decoder_draw": 5}, r"unknown sampling keys: \['num_decoder_draw'\]"),
        ({"enabled": False}, "enabled must be true"),
        ({"bootstrap": 5}, "bootstrap must be a mapping"),
        ({"bootstrap": {"resamples": 5}}, "unknown sampling.bootstrap keys"),
    ],
)
def test_from_mapping_refuses_typos_and_disabled_blocks(block, message):
    with pytest.raises(SamplingDesignError, match=message):
        SamplingConfig.from_mapping(block)


def test_build_freezes_ordered_ids_and_digests(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    rows = write_scenarios(path, 3)
    manifest = ScenarioManifest.build(path)
    assert manifest.scenario_ids == tuple(compute_scenario_id(row) for row in rows)
    assert manifest.record_count == 3
    assert manifest.source_name == "scenarios.jsonl"
    assert manifest.source_sha256 == sha256_file(path)
    assert manifest.manifest_sha256 == ScenarioManifest.digest(
        manifest.scenario_ids, manifest.source_sha256, SAMPLING_SCHEMA_VERSION)


def test_build_refuses_a_count_other_than_the_declared_one(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(path, 3)
    with pytest.raises(SamplingDesignError, match="requires exactly 500"):
        ScenarioManifest.build(path, expected_count=500)
    assert ScenarioManifest.build(path, expected_count=3).record_count == 3


def test_build_refuses_an_unparseable_line_rather_than_skipping_it(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    path.write_text('{"prompt": "p", "response": "r"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ArtifactSchemaError):
        ScenarioManifest.build(path)


def test_build_refuses_a_duplicate_scenario(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    row = json.dumps({"prompt": "p", "response": "r"}) + "\n"
    path.write_text(row * 2, encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="duplicate scenario id"):
        ScenarioManifest.build(path)


def test_build_refuses_an_empty_file(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="no scenarios"):
        ScenarioManifest.build(path)


def _manifest_with(tmp_path, **overrides) -> ScenarioManifest:
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(path, 3)
    base = ScenarioManifest.build(path).to_dict()
    base.update(overrides)
    return ScenarioManifest(
        scenario_ids=tuple(base["scenario_ids"]),
        source_name=base["source_name"],
        source_sha256=base["source_sha256"],
        record_count=base["record_count"],
        manifest_sha256=base["manifest_sha256"],
        schema_version=base["schema_version"],
    )


def test_validate_refuses_another_schema_version(tmp_path):
    with pytest.raises(SamplingDesignError, match="schema_version"):
        _manifest_with(tmp_path, schema_version=SAMPLING_SCHEMA_VERSION + 1).validate()


def test_validate_refuses_a_malformed_id(tmp_path):
    manifest = _manifest_with(tmp_path)
    ids = ("not-a-hash",) + manifest.scenario_ids[1:]
    with pytest.raises(SamplingDesignError, match="malformed scenario id"):
        ScenarioManifest(
            scenario_ids=ids,
            source_name=manifest.source_name,
            source_sha256=manifest.source_sha256,
            record_count=3,
            manifest_sha256=manifest.manifest_sha256,
        ).validate()


def test_validate_refuses_a_record_count_that_disagrees_with_the_ids(tmp_path):
    with pytest.raises(SamplingDesignError, match="record_count"):
        _manifest_with(tmp_path, record_count=4).validate()


def test_validate_refuses_a_reordered_manifest(tmp_path):
    manifest = _manifest_with(tmp_path)
    reordered = tuple(reversed(manifest.scenario_ids))
    with pytest.raises(SamplingDesignError, match="digest mismatch"):
        ScenarioManifest(
            scenario_ids=reordered,
            source_name=manifest.source_name,
            source_sha256=manifest.source_sha256,
            record_count=3,
            manifest_sha256=manifest.manifest_sha256,
        ).validate()


def test_verify_source_accepts_the_frozen_file_and_refuses_another(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(path, 3)
    manifest = ScenarioManifest.build(path)
    manifest.verify_source(path)
    other = tmp_path / "other.jsonl"
    write_scenarios(other, 4)
    with pytest.raises(SamplingDesignError, match="the manifest froze"):
        manifest.verify_source(other)


def test_write_and_load_round_trip(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(path, 2)
    manifest = ScenarioManifest.build(path)
    target = manifest.write(tmp_path / "scenario_manifest.json")
    assert ScenarioManifest.load(target) == manifest
    payload = json.loads(target.read_text())
    assert payload["scenario_ids"] == list(manifest.scenario_ids)


def test_load_refuses_missing_invalid_and_non_object_files(tmp_path):
    with pytest.raises(SamplingDesignError, match="cannot read"):
        ScenarioManifest.load(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="cannot read"):
        ScenarioManifest.load(bad)
    listy = tmp_path / "list.json"
    listy.write_text("[]", encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="not a JSON object"):
        ScenarioManifest.load(listy)


def test_from_dict_refuses_a_malformed_payload(tmp_path):
    manifest = _manifest_with(tmp_path)
    payload = manifest.to_dict()
    del payload["source_sha256"]
    with pytest.raises(SamplingDesignError, match="malformed scenario manifest"):
        ScenarioManifest.from_dict(payload)
    payload = manifest.to_dict()
    payload["record_count"] = "three"
    with pytest.raises(SamplingDesignError, match="malformed scenario manifest"):
        ScenarioManifest.from_dict(payload)
    payload = manifest.to_dict()
    del payload["schema_version"]
    with pytest.raises(SamplingDesignError, match="schema_version"):
        ScenarioManifest.from_dict(payload)


def test_condition_fingerprint_ignores_the_label_and_covers_every_axis():
    base = make_condition()
    relabelled = make_condition(condition_id="other")
    assert base.fingerprint == relabelled.fingerprint
    changed = {
        "model_name": "other-model",
        "model_type": "openai_batch",
        "prompt_style": "naive",
        "individual_rule_context": "implicated",
        "judge_type": "Narrow",
        "evaluation_mode": "full",
        "rule_set_name": "subset.jsonl",
        "rule_set_sha256": "b" * 64,
        "prompt_shape_version": 1,
        "forced_json_requirement": "label_only",
        "validation_schema_version": 4,
        "label_rule": "lenient",
        "endpoint": "/v1/responses",
        "reasoning_effort": "medium",
        "max_new_tokens": 4096,
        "send_max_output_tokens": False,
        "seed_capability": "local_deterministic",
        "retry_scope": "component",
        "strict_think": True,
        "extra": (("ruleset_label", "best_seed"),),
    }
    seen = {base.fingerprint}
    for key, value in changed.items():
        fingerprint = make_condition(**{key: value}).fingerprint
        assert fingerprint not in seen, key
        seen.add(fingerprint)


def test_the_retry_scope_and_strict_think_enter_the_identity_only_off_their_defaults():
    base = make_condition(label_rule="bare")
    payload = base.to_dict()
    assert "retry_scope" not in payload and "strict_think" not in payload
    assert make_condition(label_rule="bare", retry_scope="scenario_bundle", strict_think=False).fingerprint == (
        base.fingerprint)
    # this condition's fingerprint without either field
    assert base.fingerprint == "87eff761352e71efe05f5bc8b993d294ce78732d9d6ea91338840f8dcc279239"
    changed = make_condition(label_rule="bare", retry_scope="component", strict_think=True)
    assert changed.to_dict()["retry_scope"] == "component" and changed.to_dict()["strict_think"] is True
    assert ConditionSpec.from_dict(changed.to_dict()) == changed
    assert ConditionSpec.from_dict(payload) == base


def test_condition_extra_is_sorted_and_round_trips():
    spec = make_condition(extra=(("k", 57), ("design", "literal_bundle")))
    assert spec.extra == (("design", "literal_bundle"), ("k", 57))
    payload = spec.to_dict()
    assert payload["extra"] == {"design": "literal_bundle", "k": 57}
    assert ConditionSpec.from_dict(payload) == spec
    payload["extra"] = None
    assert ConditionSpec.from_dict(payload).extra == ()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"condition_id": "bad id"}, "condition_id"),
        ({"seed_capability": "maybe"}, "seed_capability"),
        ({"forced_json_requirement": "loose"}, "forced_json_requirement"),
        ({"temperature": 0.1}, "declares temperature"),
        ({"retry_scope": "rule"}, "retry_scope must be one of"),
        ({"strict_think": "true"}, "strict_think must be true or false"),
    ],
)
def test_condition_refuses_bad_axes(overrides, message):
    with pytest.raises(SamplingDesignError, match=message):
        make_condition(**overrides)


def test_condition_from_dict_refuses_an_unknown_field():
    payload = make_condition().to_dict()
    payload["decoder_seed"] = 3
    with pytest.raises(SamplingDesignError, match="malformed condition spec"):
        ConditionSpec.from_dict(payload)


SEED_TUPLE = dict(
    generation_seed_base=16,
    campaign_id="c",
    condition_id="k",
    scenario_id="s",
    decoder_draw_id=0,
    semantic_attempt_id=0,
    component_id="full_ruleset",
)


def test_generation_seed_is_process_independent():
    assert derive_generation_seed(**SEED_TUPLE) == 177013527
    assert derive_generation_seed(**SEED_TUPLE) < 2**31


@pytest.mark.parametrize(
    "axis, value",
    [
        ("generation_seed_base", 17),
        ("campaign_id", "c2"),
        ("condition_id", "k2"),
        ("scenario_id", "s2"),
        ("decoder_draw_id", 1),
        ("semantic_attempt_id", 1),
        ("component_id", "Rule 1"),
    ],
)
def test_every_identity_axis_changes_the_seed(axis, value):
    assert derive_generation_seed(**{**SEED_TUPLE, axis: value}) != derive_generation_seed(
        **SEED_TUPLE
    )


def test_draw_dirname_and_request_block_id_render_the_axes():
    assert draw_dirname(3) == "draw-003"
    block = request_block_id(
        campaign_id="ruleset-size",
        condition_id="gemma/naive",
        decoder_draw_id=2,
        semantic_attempt_id=1,
        block_index=4,
    )
    assert block == "ruleset-size/gemma/naive/draw-002/attempt-1/block-0004"
    with pytest.raises(SamplingDesignError, match="decoder_draw_id"):
        draw_dirname(-1)
    with pytest.raises(SamplingDesignError, match="semantic_attempt_id"):
        request_block_id(
            campaign_id="c", condition_id="k", decoder_draw_id=0,
            semantic_attempt_id=-1, block_index=0,
        )


def test_request_namespace_is_short_and_distinct_per_block():
    a = request_namespace("c/k/draw-000/attempt-0/block-0000")
    b = request_namespace("c/k/draw-000/attempt-1/block-0000")
    c = request_namespace("c/k/draw-001/attempt-0/block-0000")
    assert len({a, b, c}) == 3
    assert a.startswith("ns") and len(a) == 18
    assert all(ch in "0123456789abcdef" for ch in a[2:])


def test_a_complete_grid_passes():
    check = check_exact_grid(["a", "b"], 2, [("a", 0), ("a", 1), ("b", 0), ("b", 1)])
    assert check.ok
    assert check.expected_keys == 4 and check.observed_keys == 4
    assert check.to_dict() == {
        "ok": True, "expected_keys": 4, "observed_keys": 4,
        "missing": [], "duplicates": [], "unexpected": [],
    }


def test_missing_duplicate_and_unexpected_keys_each_fail():
    check = check_exact_grid(
        ["a", "b"], 2, [("a", 0), ("a", 0), ("b", 0), ("b", 1), ("zz", 0)]
    )
    assert not check.ok
    assert check.missing == (("a", 1),)
    assert check.duplicates == (("a", 0),)
    assert check.unexpected == (("zz", 0),)
    assert check.observed_keys == 5
    assert isinstance(check, GridCheck)


def test_grid_check_refuses_a_design_with_no_draws():
    with pytest.raises(SamplingDesignError, match="expected_decoder_draws"):
        check_exact_grid(["a"], 0, [])


@pytest.fixture
def scenario_manifest(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    write_scenarios(path, 3)
    return ScenarioManifest.build(path)


def test_build_sampling_manifest_records_every_import_decision_field(scenario_manifest):
    config = SamplingConfig(campaign_mode="pilot", num_decoder_draws=3)
    condition = make_condition(endpoint="/v1/responses", reasoning_effort="medium")
    manifest = build_sampling_manifest(
        campaign_id="ruleset-size-pilot",
        config=config,
        scenario_manifest=scenario_manifest,
        conditions=[condition],
        created_at="2026-09-08T00:00:00Z",
        code_revision="abc123",
    )
    assert manifest["schema_version"] == SAMPLING_SCHEMA_VERSION
    assert manifest["scenario_manifest_sha256"] == scenario_manifest.manifest_sha256
    assert manifest["expected_scenarios"] == 3
    assert manifest["expected_decoder_draws"] == 3
    assert manifest["temperature"] == 1.0
    assert manifest["max_semantic_attempts"] == 5
    assert manifest["retry_scope"] == "scenario_bundle"
    assert manifest["forced_json_requirement"] == "complete_rule_map"
    assert manifest["seed_schema_version"] == design.SEED_SCHEMA_VERSION
    recorded = manifest["conditions"][0]
    assert recorded["fingerprint"] == condition.fingerprint
    for key in (
        "model_name", "endpoint", "reasoning_effort", "max_new_tokens",
        "send_max_output_tokens", "prompt_shape_version", "rule_set_sha256",
        "validation_schema_version", "seed_capability", "temperature",
    ):
        assert key in recorded


def test_build_sampling_manifest_refuses_bad_condition_sets(scenario_manifest):
    config = SamplingConfig(campaign_mode="pilot")
    common = dict(
        config=config, scenario_manifest=scenario_manifest,
        created_at="t", code_revision=None,
    )
    with pytest.raises(SamplingDesignError, match="campaign_id"):
        build_sampling_manifest(campaign_id="bad id", conditions=[make_condition()], **common)
    assert build_sampling_manifest(campaign_id="c", conditions=[], **common)["conditions"] == []
    with pytest.raises(SamplingDesignError, match="duplicate condition ids"):
        build_sampling_manifest(
            campaign_id="c", conditions=[make_condition(), make_condition()], **common
        )
    with pytest.raises(SamplingDesignError, match="forced_json_requirement"):
        build_sampling_manifest(
            campaign_id="c",
            conditions=[make_condition(forced_json_requirement="label_only")],
            **common,
        )
    for field, value in (("retry_scope", "component"), ("strict_think", True)):
        with pytest.raises(SamplingDesignError, match=f"declares {field}={value!r} but the campaign declares"):
            build_sampling_manifest(campaign_id="c", conditions=[make_condition(**{field: value})], **common)


def test_the_manifest_records_strict_think_only_when_on(scenario_manifest):
    common = dict(scenario_manifest=scenario_manifest, created_at="t", code_revision=None)
    off = build_sampling_manifest(campaign_id="c", config=SamplingConfig(campaign_mode="pilot"), conditions=[], **common)
    assert "strict_think" not in off and off["retry_scope"] == "scenario_bundle"
    config = SamplingConfig(campaign_mode="pilot", retry_scope="component", strict_think=True)
    condition = make_condition(retry_scope="component", strict_think=True)
    on = build_sampling_manifest(campaign_id="c", config=config, conditions=[condition], **common)
    assert (on["retry_scope"], on["strict_think"]) == ("component", True)
    assert on["conditions"][0]["fingerprint"] == condition.fingerprint
    assert design.sampling_config_from_manifest(on) == config
    assert design.sampling_config_from_manifest(off) == SamplingConfig(campaign_mode="pilot")
    assert design.differing_manifest_fields(off, on) == ["conditions", "retry_scope", "strict_think"]


def _manifest(scenario_manifest, **overrides):
    manifest = build_sampling_manifest(
        campaign_id="c",
        config=SamplingConfig(campaign_mode="pilot"),
        scenario_manifest=scenario_manifest,
        conditions=[make_condition()],
        created_at="2026-09-08",
        code_revision="rev1",
    )
    manifest.update(overrides)
    return manifest


def test_ensure_sampling_manifest_writes_once_and_tolerates_run_fields(
    tmp_path, scenario_manifest
):
    path = tmp_path / "sampling_manifest.json"
    first = _manifest(scenario_manifest)
    assert ensure_sampling_manifest(path, first) == first
    resumed = _manifest(scenario_manifest, created_at="later", code_revision="rev2")
    assert ensure_sampling_manifest(path, resumed) == first
    assert load_sampling_manifest(path) == first


def test_ensure_sampling_manifest_refuses_a_different_campaign(tmp_path, scenario_manifest):
    path = tmp_path / "sampling_manifest.json"
    ensure_sampling_manifest(path, _manifest(scenario_manifest))
    with pytest.raises(SamplingDesignError, match="expected_decoder_draws differ"):
        ensure_sampling_manifest(
            path, _manifest(scenario_manifest, expected_decoder_draws=7)
        )


def test_ensure_sampling_manifest_refuses_unreadable_or_foreign_files(
    tmp_path, scenario_manifest
):
    manifest = _manifest(scenario_manifest)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="unreadable"):
        ensure_sampling_manifest(broken, manifest)
    listy = tmp_path / "list.json"
    listy.write_text("[]", encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="not a JSON object"):
        ensure_sampling_manifest(listy, manifest)
    old = tmp_path / "old.json"
    old.write_text(json.dumps({**manifest, "schema_version": 0}), encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="schema_version"):
        ensure_sampling_manifest(old, manifest)


def test_load_sampling_manifest_refuses_missing_and_wrong_files(tmp_path):
    with pytest.raises(SamplingDesignError, match="cannot read"):
        load_sampling_manifest(tmp_path / "absent.json")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"schema_version": 9}), encoding="utf-8")
    with pytest.raises(SamplingDesignError, match="not a schema_version"):
        load_sampling_manifest(wrong)


def test_an_integer_temperature_is_stored_as_a_float_everywhere():
    assert SamplingConfig(temperature=1).to_dict()["temperature"] == 1.0
    assert isinstance(SamplingConfig(temperature=1).temperature, float)
    assert make_condition(temperature=1).fingerprint == make_condition(temperature=1.0).fingerprint
    assert make_condition(temperature=1).to_dict()["temperature"] == 1.0


def test_local_identity_fields_enter_the_fingerprint():
    base = make_condition().fingerprint
    assert make_condition(model_path="gemma-4-31B-it").fingerprint != base
    assert make_condition(adapter_path="final_adapter").fingerprint != base
    assert make_condition(enable_thinking=False).fingerprint != base


def test_check_sampling_manifest_compares_without_writing(tmp_path, scenario_manifest):
    from ruleforge.evaluate.identity import check_sampling_manifest

    path = tmp_path / "sampling_manifest.json"
    first = _manifest(scenario_manifest, code_dirty=True, code_state_sha256="s" * 64)
    assert check_sampling_manifest(path, first) is None
    assert not path.exists()
    ensure_sampling_manifest(path, first)
    resumed = _manifest(scenario_manifest, created_at="later", code_revision="rev2",
                        code_dirty=False, code_state_sha256=None)
    assert check_sampling_manifest(path, resumed) == first
    with pytest.raises(SamplingDesignError, match="different campaign"):
        check_sampling_manifest(path, _manifest(scenario_manifest, temperature=0.7))
    assert load_sampling_manifest(path) == first


def test_build_sampling_manifest_records_the_code_state(scenario_manifest):
    manifest = build_sampling_manifest(
        campaign_id="camp", config=SamplingConfig(campaign_mode="pilot", num_decoder_draws=2),
        scenario_manifest=scenario_manifest, conditions=[], created_at="t",
        code_revision="abc", code_dirty=True, code_state_sha256="d" * 64,
    )
    assert (manifest["code_revision"], manifest["code_dirty"], manifest["code_state_sha256"]) == (
        "abc", True, "d" * 64
    )


def test_the_truncation_policy_is_validated_and_part_of_the_identity(scenario_manifest):
    from ruleforge.evaluate.identity import TRUNCATION_MODEL_DEFAULT, TRUNCATION_NONE

    assert SamplingConfig(campaign_mode="pilot", num_decoder_draws=2).truncation == TRUNCATION_MODEL_DEFAULT
    assert SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, truncation=TRUNCATION_NONE).truncation == "none"
    with pytest.raises(SamplingDesignError, match="sampling.truncation"):
        SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, truncation="nucleus")
    with pytest.raises(SamplingDesignError, match="unknown sampling keys"):
        SamplingConfig.from_mapping({"campaign_mode": "pilot", "num_decoder_draws": 2, "truncate": "none"})
    manifest = build_sampling_manifest(
        campaign_id="camp", config=SamplingConfig(campaign_mode="pilot", num_decoder_draws=2, truncation="none"),
        scenario_manifest=scenario_manifest, conditions=[], created_at="t", code_revision=None,
    )
    assert manifest["truncation"] == "none"


def test_condition_identity_distinguishes_sampling_truncation():
    base = dict(
        condition_id="m/naive", model_name="m", model_type="local", prompt_style="naive",
        individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
        rule_set_name="r.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
        forced_json_requirement="complete_rule_map",
    )
    full = ConditionSpec(**base, sampling_truncation={"top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0})
    shipped = ConditionSpec(**base, sampling_truncation={"top_k": 64, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0})
    reordered = ConditionSpec(**base, sampling_truncation=[("top_p", 0.95), ("repetition_penalty", 1.0), ("min_p", 0.0), ("top_k", 64)])
    provider = ConditionSpec(**base, sampling_truncation="provider_default")
    unset = ConditionSpec(**base)
    assert len({full.fingerprint, shipped.fingerprint, provider.fingerprint, unset.fingerprint}) == 4
    assert shipped.fingerprint == reordered.fingerprint
    assert shipped.to_dict()["sampling_truncation"] == {"top_k": 64, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0}
    assert ConditionSpec.from_dict(shipped.to_dict()).fingerprint == shipped.fingerprint
    assert ConditionSpec.from_dict(provider.to_dict()).sampling_truncation == "provider_default"
    with pytest.raises(SamplingDesignError, match="sampling_truncation"):
        ConditionSpec(**base, sampling_truncation=0.95)


def test_a_manifest_reads_back_as_the_config_it_was_built_from(scenario_manifest):
    config = SamplingConfig(
        campaign_mode="pilot", num_decoder_draws=3, max_semantic_attempts=20, label_rule="lenient",
        scenarios_per_request_block=50, truncation="none", forced_json_requirement="label_only",
        generation_seed_base=7, bootstrap=BootstrapSettings(n_resamples=200, ci_level=0.9, seed=3),
    )
    manifest = build_sampling_manifest(
        campaign_id="c", config=config, scenario_manifest=scenario_manifest, conditions=[],
        created_at="t", code_revision=None,
    )
    assert design.sampling_config_from_manifest(manifest) == config
    rebuilt = build_sampling_manifest(
        campaign_id="c", config=design.sampling_config_from_manifest(manifest),
        scenario_manifest=scenario_manifest, conditions=[], created_at="later", code_revision="abc",
    )
    assert design.differing_manifest_fields(manifest, rebuilt) == []
    assert design.differing_manifest_fields(manifest, {**rebuilt, "label_rule": "bare", "extra": 1}) == ["extra", "label_rule"]
    before_the_axis = {key: value for key, value in manifest.items() if key != "label_rule"}
    with pytest.raises(SamplingDesignError, match="whole campaign config"):
        design.sampling_config_from_manifest(before_the_axis)
    with pytest.raises(SamplingDesignError, match="whole campaign config"):
        design.sampling_config_from_manifest({**manifest, "bootstrap": {"n_resamples": 10, "unknown": 1}})
