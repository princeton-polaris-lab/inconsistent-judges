from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from ruleforge.core import record_names as names
from ruleforge.core.digests import canonical_digest, request_namespace
from ruleforge.core.errors import SamplingDesignError
from ruleforge.evaluate.cost import block_namespace
from tests.conftest import REPO_ROOT
from ruleforge.evaluate.discrimination import VerdictGrids
from ruleforge.evaluate.identity import ConditionSpec, derive_generation_seed, request_block_id
from ruleforge.evaluate.inference import DrawGrid, RepeatedSamplingError, grid_from_cells, outcome_grid, rule_grids
from ruleforge.prompts import styles
from ruleforge.tasks import BUILTIN_TASKS, FORCED_JSON, LABEL
from ruleforge.tasks.registry import BUILTIN_TASK_CLASSES
from tests.identity import published as pub

REPO = Path(__file__).resolve().parents[1]
TABLE = REPO / "src" / "ruleforge" / "core" / names.TABLE_FNAME
CONDITION_ROWS = pub.rows_of(pub.read_fixture(pub.CONDITIONS), "condition")
SEED_ROWS = pub.read_fixture(pub.SEEDS)
#: The paper's styles, which every regeneration of the fixtures keeps conditions of.
PAPER_STYLES = {styles.NAIVE_RULES_ABOVE, styles.ENCOURAGED, styles.FORCED, styles.FORCED_GRPO}
KEY = names.IDENTITY_KEYS  # public style -> recorded key


def _public(payload):
    style = payload["prompt_style"]
    public = names.public_style(style)
    renamed = dict(payload, prompt_style=public)
    renamed["condition_id"] = "/".join(public if segment == style else segment
                                       for segment in payload["condition_id"].split("/"))
    return renamed


def _spec(payload) -> ConditionSpec:
    return ConditionSpec.from_dict({key: value for key, value in payload.items()
                                    if key not in ("schema_version", "fingerprint")})


def _root(row) -> str:
    return row["source"]["path"].split("/")[1]


CONDITIONS = {(row["root"], row["payload"]["condition_id"]): row["payload"] for row in CONDITION_ROWS}


def _conditions(row) -> tuple[ConditionSpec, ConditionSpec]:
    payload = CONDITIONS[(_root(row), row["condition_id"])]
    return _spec(payload), _spec(_public(payload))


def spec(**overrides) -> ConditionSpec:
    """A condition of ``overrides``' style, the hierarchy style's in its one rule context with a levels file."""
    fields = dict(condition_id="judge/naive-rules-above", model_name="judge", model_type="local", prompt_style="naive-rules-above",
                  individual_rule_context="isolated", judge_type="No_Interpretation", evaluation_mode="both",
                  rule_set_name="rules.jsonl", rule_set_sha256="a" * 64, temperature=1.0, prompt_shape_version=2,
                  forced_json_requirement="complete_rule_map")
    if overrides.get("prompt_style") in styles.HIERARCHY_STYLES:
        fields.update(individual_rule_context=styles.HIERARCHY_RULE_CONTEXT, rule_levels_sha256="b" * 64)
    return ConditionSpec(**{**fields, **overrides})


def test_the_table_is_the_json_beside_the_module_and_names_every_built_in_style_once():
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    assert names.read_table(table)["keys"] == dict(KEY)
    assert [row["public"] for row in table["styles"]] == list(names.ORDER) == list(KEY)
    assert set(KEY) == set(styles.VALID_REASONING_PROMPT_STYLES)
    for task in BUILTIN_TASK_CLASSES.values():
        assert set(task.styles) <= set(KEY)
    assert names.RECORD_KEYS == frozenset(KEY.values()) == frozenset(names.PUBLIC_NAMES)
    assert len(names.RECORD_KEYS) == len(KEY) and names.RECORD_KEYS.isdisjoint(KEY)
    assert names.RETIRED_KEYS and names.RETIRED_KEYS.isdisjoint({*KEY, *names.RECORD_KEYS})
    assert names.IDENTITY_TASKS == BUILTIN_TASKS == tuple(BUILTIN_TASK_CLASSES)


def test_the_styles_are_the_public_names_in_the_records_order_and_the_record_only_ones_the_table_s():
    """The enumeration order orders every style family's members, so each contrast's a/b orientation."""
    assert styles.VALID_REASONING_PROMPT_STYLES == names.ORDER == (
        "naive-rules-above", "forced-grpo", "naive", "encouraged", "forced", "naive-hierarchy")
    assert names.RECORD_ONLY == ()
    assert styles.REVISED_STYLES == ("naive", "encouraged", "forced")
    assert styles.FORCED_JSON_STYLES == ("forced-grpo", "forced")
    assert styles.PREEMPTION_LISTING_STYLES == ("forced-grpo",)
    assert dict(BUILTIN_TASK_CLASSES["modelspec"].styles) == {
        style: FORCED_JSON if style in styles.FORCED_JSON_STYLES else LABEL
        for style in styles.VALID_REASONING_PROMPT_STYLES}
    assert dict(BUILTIN_TASK_CLASSES["legal"].styles) == {"naive": LABEL}


@pytest.mark.parametrize("change", [
    lambda table: table.pop("styles"),
    lambda table: table.update(styles=None),
    lambda table: table.update(schema="ruleforge.core.record_names/2"),
    lambda table: table.update(tasks=[]),
    lambda table: table.update(tasks=[""]),
    lambda table: table["styles"][0].update(record_only="yes"),
    lambda table: table["styles"][0].update(public=table["styles"][1]["public"]),
    lambda table: table["styles"][0].update(public=table["styles"][1]["recorded"]),
    lambda table: table["retired"].append(table["styles"][0]["recorded"]),
    lambda table: table["retired"].append(3),
    lambda table: table.update(tasks="modelspec"),  # a string is no list of names, though its letters are names
    lambda table: table.update(retired="z"),
    lambda table: table.pop("task_keys"),
    lambda table: table.update(task_keys=None),
    lambda table: table.update(task_keys={"legal": "z"}),
    lambda table: table.update(task_keys={"my-task": {"naive": table["styles"][0]["recorded"]}}),
    lambda table: table.update(task_keys={"legal": {"my-style": table["styles"][0]["recorded"]}}),
    lambda table: table.update(task_keys={"legal": {"naive": "my-key"}}),
])
def test_a_malformed_table_is_refused(change):
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    change(table)
    with pytest.raises(SamplingDesignError, match=f"{names.TABLE_FNAME} is malformed"):
        names.read_table(table)


def test_a_built_in_task_s_style_is_hashed_under_its_key_and_read_back_by_its_public_name():
    for style, key in KEY.items():
        for task in (None, *BUILTIN_TASKS):
            if style not in names.TASK_KEYS.get(task, {}) and key not in names.TASK_KEYS.get(task, {}).values():
                assert names.identity_key(style, task) == key
                assert names.public_style(key, task) == style
                assert names.public_style(style, task) == style
        assert names.identity_key(style, "my-task") == style and names.public_style(key, "my-task") == key
        assert names.identity_key(key) == key
        assert names.renamed_to(key) == style and names.renamed_to(style) is None
        assert names.public_segment(key) == style and names.public_segment(style) == style
    for retired in names.RETIRED_KEYS:
        assert names.public_style(retired) == names.identity_key(retired) == names.public_segment(retired) == retired
        assert names.renamed_to(retired) is None
        # A built-in task's condition recorded under a retired key is not read; another task's style is its own.
        assert names.is_retired(retired) and names.is_retired(retired, "legal") and not names.is_retired(retired, "own")
    assert not any(names.is_retired(style) for style in (*names.RECORD_KEYS, *names.ORDER, None))
    assert names.identity_key("my-style") == names.public_style("my-style") == "my-style"
    assert names.public_segment("bank") == "bank"


def test_the_legal_task_s_naive_prompt_is_hashed_under_its_own_key():
    paper_key = KEY["naive-rules-above"]
    assert names.TASK_KEYS == {"legal": {"naive": paper_key}}
    assert names.identity_key("naive", "legal") == paper_key
    assert names.public_style(paper_key, "legal") == names.public_style("naive", "legal") == "naive"
    assert names.identity_key("naive") == names.identity_key("naive", "modelspec") == KEY["naive"]
    condition = spec(condition_id="judge/naive", prompt_style="naive", task="legal")
    assert condition.identity_id == f"judge/{paper_key}" and condition.public_id == "judge/naive"
    assert condition.fingerprint == "9da8fe063802068117255c811c00abb1f9a9d39d879f68ab01a157ff61c7f175"
    payload = dict(condition.to_dict(), condition_id=f"judge/{paper_key}", prompt_style=paper_key)
    recorded = ConditionSpec.from_dict(payload)
    assert recorded.prompt_style == "naive" and recorded.fingerprint == condition.fingerprint


def test_an_identity_id_maps_the_segments_equal_to_its_own_style_alone():
    for style, key in KEY.items():
        other = next(name for name in KEY if name != style)
        assert names.identity_condition_id(f"judge/{style}", style) == f"judge/{key}"
        assert names.identity_condition_id(f"{style}/x/{style}/prompt.{style}", style) == \
            f"{key}/x/{key}/prompt.{style}"
        assert names.identity_condition_id(f"judge/{other}", style) == f"judge/{other}"
        assert names.identity_condition_id(f"judge/{style}", style, "my-task") == f"judge/{style}"
        assert names.identity_condition_id("judge/bank", style) == "judge/bank"
        assert names.identity_condition_id(f"judge/{key}", style) == f"judge/{key}"
    assert names.identity_condition_id("judge/my-style", "my-style") == "judge/my-style"


def test_a_public_id_maps_the_segments_equal_to_its_own_style_s_key_alone_and_inverts_the_identity_id():
    for style, key in KEY.items():
        other = KEY[next(name for name in KEY if name != style)]
        for given in (style, key):
            assert names.public_condition_id(f"judge/{key}", given) == f"judge/{style}"
            assert names.public_condition_id(f"{key}/x/{key}/prompt.{key}", given) == f"{style}/x/{style}/prompt.{key}"
            assert names.public_condition_id(f"judge/{style}", given) == f"judge/{style}"
            assert names.public_condition_id(f"judge/{other}", given) == f"judge/{other}"
            assert names.public_condition_id("judge/figure5/base", given) == "judge/figure5/base"
        assert names.public_condition_id(f"judge/{key}", key, "my-task") == f"judge/{key}"
        public = f"judge/{style}"
        assert names.public_condition_id(names.identity_condition_id(public, style), style) == public
    assert names.public_condition_id("judge/my-style", "my-style", "my-task") == "judge/my-style"


def test_a_block_s_identity_id_maps_its_condition_id_and_keeps_its_draw_attempt_block_and_shard():
    parts = dict(decoder_draw_id=2, semantic_attempt_id=1, block_index=7)
    for style, key in KEY.items():
        block = request_block_id(campaign_id="camp", condition_id=f"judge/{style}", **parts)
        expected = request_block_id(campaign_id="camp", condition_id=f"judge/{key}", **parts)
        for suffix in ("", "@shard-01-of-02"):
            assert names.identity_block_id(block + suffix, campaign_id="camp", condition_id=f"judge/{style}",
                                           prompt_style=style) == expected + suffix
            assert names.identity_block_id(expected + suffix, campaign_id="camp", condition_id=f"judge/{key}",
                                           prompt_style=style) == expected + suffix
    block = request_block_id(campaign_id="camp", condition_id="judge/naive-rules-above", **parts)
    for campaign, condition in (("other", "judge/naive-rules-above"), ("camp", "judge")):
        with pytest.raises(SamplingDesignError, match="is not a block of condition"):
            names.identity_block_id(block, campaign_id=campaign, condition_id=condition, prompt_style="naive-rules-above")


def test_a_condition_id_s_recorded_key_segments_but_its_own_style_s_are_found():
    key, other = KEY["naive-rules-above"], KEY["forced"]
    assert names.record_key_segments(f"{key}/judge/{key}") == (key, key)
    assert names.record_key_segments("judge/naive-rules-above/prompt.forced") == ()
    for style in ("naive-rules-above", key):
        assert names.record_key_segments(f"{key}/judge/{other}", style) == (other,)
        assert names.record_key_segments(f"{key}/judge/{other}", style, "legal") == (other,)
    assert names.record_key_segments(f"judge/{key}", "my-style", "my-task") == (key,)
    assert names.record_key_segments(f"judge/{key}", key, "my-task") == (key,)


def test_a_new_condition_id_holding_a_recorded_or_retired_key_is_refused_naming_each():
    retired = sorted(names.RETIRED_KEYS)[0]
    for key, public in names.PUBLIC_NAMES.items():
        with pytest.raises(SamplingDesignError, match=rf"'judge/{key}' holds '{key}' \(the records' key of "
                                                      rf"'{public}'\) as a segment: .* public name"):
            names.refuse_record_keys(f"judge/{key}")
    with pytest.raises(SamplingDesignError, match=rf"holds '{KEY['forced']}' .*, '{retired}' \(a retired style's key\)"):
        names.refuse_record_keys(f"{KEY['forced']}/{retired}")
    for allowed in ("judge/naive-rules-above", "judge/figure5/base", f"judge/{KEY['naive-rules-above']}x", f"judge/prompt.{KEY['naive-rules-above']}"):
        names.refuse_record_keys(allowed)


def test_one_registered_condition_is_located_by_its_public_id_in_a_recorded_or_a_new_root():
    recorded = spec(condition_id=f"judge/{KEY['encouraged']}", prompt_style="encouraged")
    new = spec(condition_id="other/encouraged", prompt_style="encouraged")
    bank = spec(condition_id="judge/bank")
    pool = {condition.condition_id: condition for condition in (recorded, new, bank)}
    assert names.locate_condition(pool, "judge/encouraged") is recorded
    assert names.locate_condition(list(pool.values()), "other/encouraged") is new
    assert names.locate_condition(iter([bank]), "judge/bank") is bank
    with pytest.raises(SamplingDesignError, match=rf"no registered condition has the public id 'judge/forced' "
                                                  rf"\(registered: \['judge/bank', 'judge/{KEY['encouraged']}', "
                                                  r"'other/encouraged'\]\)"):
        names.locate_condition(pool, "judge/forced")
    with pytest.raises(SamplingDesignError, match="no registered condition has the public id"):
        names.locate_condition(pool, recorded.condition_id)
    twin = spec(condition_id="judge/encouraged", prompt_style="encouraged")
    with pytest.raises(SamplingDesignError, match=rf"2 registered conditions have the public id 'judge/encouraged' "
                                                  rf"\(\['judge/encouraged', 'judge/{KEY['encouraged']}'\]\)"):
        names.locate_condition([recorded, twin], "judge/encouraged")


def test_a_built_in_task_s_fingerprint_hashes_its_style_s_key_and_a_task_of_one_s_own_s_its_style():
    for style, key in KEY.items():
        condition = spec(condition_id=f"judge/{style}", prompt_style=style)
        payload = condition.to_dict()
        assert payload["prompt_style"] == style and payload["condition_id"] == f"judge/{style}"
        del payload["condition_id"]
        assert condition.fingerprint == canonical_digest({**payload, "prompt_style": key})
        assert condition.identity_id == f"judge/{key}" and condition.public_id == f"judge/{style}"
        assert condition.identity_block_id("camp", f"camp/judge/{style}/draw-000/attempt-0/block-0000") == \
            f"camp/judge/{key}/draw-000/attempt-0/block-0000"
    custom = spec(condition_id="judge/my-style", prompt_style="my-style", task="my-task", response_contract="label")
    payload = custom.to_dict()
    del payload["condition_id"]
    assert custom.fingerprint == canonical_digest(payload)
    assert custom.identity_id == custom.public_id == "judge/my-style"


def test_a_built_in_task_s_condition_named_by_a_recorded_or_retired_key_is_refused_and_pointed_at_the_read_seam():
    for style, key in KEY.items():
        with pytest.raises(SamplingDesignError, match=rf"no prompt style '{key}'.*'{key}' is the key the published "
                                                      rf"records name the style '{style}' by: read a recorded "
                                                      r"condition through ConditionSpec.from_dict"):
            spec(condition_id="judge/x", prompt_style=key)
    for retired in names.RETIRED_KEYS:
        with pytest.raises(SamplingDesignError, match=rf"'{retired}' is a retired style's key: its experiment is not "
                                                      "part of the release"):
            spec(condition_id="judge/x", prompt_style=retired)
    with pytest.raises(SamplingDesignError, match=r"no prompt style 'my-style' \(its styles: \[[^]]*\]\)$"):
        spec(condition_id="judge/x", prompt_style="my-style")
    with pytest.raises(SamplingDesignError, match="task 'legal' has no prompt style 'forced'"):
        spec(condition_id="judge/forced", prompt_style="forced", task="legal")


def test_a_condition_id_holds_a_recorded_key_only_as_its_own_style_s():
    for style, key in KEY.items():
        recorded = spec(condition_id=f"judge/{key}", prompt_style=style)
        assert recorded.identity_id == f"judge/{key}" and recorded.public_id == f"judge/{style}"
        other = KEY[next(name for name in KEY if name != style)]
        with pytest.raises(SamplingDesignError, match=rf"its id holds \['{other}'\], the key the published records "
                                                      "name a style by, as a segment, and not as its own style's"):
            spec(condition_id=f"judge/{other}", prompt_style=style)
    key = KEY["forced"]
    with pytest.raises(SamplingDesignError, match=rf"its id holds \['{key}'\], the key "):
        spec(condition_id=f"judge/{key}", task="my-task", prompt_style=key, response_contract="label")
    both = (KEY["naive-rules-above"], KEY["forced"], KEY["encouraged"])
    with pytest.raises(SamplingDesignError, match=rf"its id holds \['{both[1]}', '{both[2]}'\], the keys "):
        spec(condition_id="/".join(both))
    assert spec(condition_id=f"judge/{key}x").condition_id == f"judge/{key}x"


def test_a_recorded_condition_is_read_under_its_public_style_with_its_recorded_id_and_identity():
    for style in KEY:
        for task in ("modelspec", "legal"):
            if style not in BUILTIN_TASK_CLASSES[task].styles:
                continue
            key = names.identity_key(style, task)
            public = spec(condition_id=f"judge/{style}", prompt_style=style, task=task)
            payload = dict(public.to_dict(), condition_id=f"judge/{key}", prompt_style=key)
            recorded = ConditionSpec.from_dict(payload)
            assert recorded.prompt_style == style and recorded.condition_id == f"judge/{key}"
            assert recorded.fingerprint == public.fingerprint
            assert recorded.identity_id == public.identity_id == f"judge/{key}"
            assert recorded.public_id == public.public_id == f"judge/{style}"
            assert ConditionSpec.from_dict(recorded.to_dict()) == recorded
            assert ConditionSpec.from_dict(public.to_dict()) == public
            styleless = ConditionSpec.from_dict(dict(payload, condition_id="judge/figure5/base"))
            assert styleless.fingerprint == public.fingerprint and styleless.identity_id == "judge/figure5/base"
    key = KEY["naive-rules-above"]
    custom = dict(spec(condition_id="judge/x", prompt_style="x", task="my-task", response_contract="label").to_dict(),
                  prompt_style=key)
    assert ConditionSpec.from_dict(custom).prompt_style == key


def test_a_recorded_condition_whose_id_holds_its_style_s_public_name_is_refused():
    payload = dict(spec(condition_id="judge/x").to_dict(), prompt_style=KEY["forced"],
                   condition_id="judge/forced/archive")
    with pytest.raises(SamplingDesignError, match=rf"'judge/forced/archive' is recorded under the style key "
                                                  rf"'{KEY['forced']}' and holds that style's public name 'forced'"):
        ConditionSpec.from_dict(payload)
    assert ConditionSpec.from_dict(dict(payload, condition_id="judge/naive-rules-above/archive")).prompt_style == "forced"


def test_a_recorded_condition_s_contract_and_preemption_listing_are_its_public_style_s():
    for style in KEY:
        public = spec(condition_id=f"judge/{style}", prompt_style=style)
        recorded = ConditionSpec.from_dict(dict(public.to_dict(), condition_id=f"judge/{KEY[style]}",
                                                prompt_style=KEY[style]))
        assert recorded.contract == public.contract == (FORCED_JSON if style in styles.FORCED_JSON_STYLES else LABEL)
        assert recorded.lists_preemption == public.lists_preemption == (style in styles.PREEMPTION_LISTING_STYLES)


def test_a_payload_whose_style_is_no_name_reaches_the_guard_unmapped():
    with pytest.raises(SamplingDesignError, match="has no prompt style None"):
        ConditionSpec.from_dict(dict(spec().to_dict(), prompt_style=None))
    with pytest.raises(SamplingDesignError, match="malformed condition spec"):
        ConditionSpec.from_dict(dict(spec().to_dict(), unknown=1))


def test_every_published_condition_is_read_with_its_recorded_fingerprint_and_its_recorded_id_as_its_identity():
    assert len(CONDITION_ROWS) >= 3000
    seen = set()
    for row in CONDITION_ROWS:
        payload = row["payload"]
        recorded, public = _spec(payload), _spec(_public(payload))
        assert recorded.fingerprint == public.fingerprint == payload["fingerprint"], row["path"]
        assert recorded.identity_id == public.identity_id == payload["condition_id"], row["path"]
        assert recorded.public_id == public.public_id == public.condition_id, row["path"]
        assert recorded.prompt_style == public.prompt_style == names.public_style(payload["prompt_style"])
        seen.add(recorded.prompt_style)
    assert PAPER_STYLES <= seen


def test_every_published_generation_seed_re_derives_from_its_condition_read_as_recorded_and_under_its_public_names():
    seeds = pub.rows_of(SEED_ROWS, "seed")
    assert len(seeds) == 90
    renamed = 0
    for row in seeds:
        for condition in _conditions(row):
            renamed += condition.condition_id != condition.identity_id
            assert derive_generation_seed(
                generation_seed_base=row["generation_seed_base"], campaign_id=row["campaign_id"],
                condition_id=condition.identity_id, scenario_id=row["scenario_id"],
                decoder_draw_id=row["decoder_draw_id"], semantic_attempt_id=row["semantic_attempt_id"],
                component_id=row["component_id"],
            ) == row["generation_seed"], row["source"]
    assert renamed


def test_every_published_request_block_s_provider_namespace_re_derives_from_its_recorded_and_public_block_id():
    blocks = pub.rows_of(SEED_ROWS, "block")
    assert len(blocks) == 19
    renamed = 0
    for row in blocks:
        campaign = row["campaign_id"]
        namespace = row["provider_request_ref"].split("-", 1)[0]
        index = int(re.fullmatch(r".*/block-(\d{4})", row["request_block_id"]).group(1))
        for condition in _conditions(row):
            block = request_block_id(campaign_id=campaign, condition_id=condition.condition_id,
                                     decoder_draw_id=row["decoder_draw_id"],
                                     semantic_attempt_id=row["semantic_attempt_id"], block_index=index)
            renamed += block != condition.identity_block_id(campaign, block)
            assert request_namespace(condition.identity_block_id(campaign, block)) == namespace, row["source"]
            assert block_namespace(campaign, condition, block) == namespace
    assert renamed


def _cells(values):
    return {(f"s{index}", draw): {"contradiction": value, "full_set_phi": value, "max_rule_phi": 1.0 - value,
                                  "rule_judgments": {"r1": "yes" if value else "no"}}
            for index, row in enumerate(values) for draw, value in enumerate(row)}


def test_a_grid_s_digest_hashes_its_identity_id_and_every_message_names_its_condition_id():
    values = [[1.0, 0.0], [0.0, 0.0]]
    ids = ("s0", "s1")
    recorded = DrawGrid("judge/key", "contradiction", ids, np.array(values))
    grid = DrawGrid("judge/naive-rules-above", "contradiction", ids, np.array(values), identity_id="judge/key")
    assert grid.digest == recorded.digest != DrawGrid("judge/naive-rules-above", "contradiction", ids, np.array(values)).digest
    assert grid.restricted_to(["s1", "s0"]).digest == recorded.restricted_to(["s1", "s0"]).digest
    with pytest.raises(RepeatedSamplingError, match="judge/naive-rules-above: 1 requested scenario"):
        grid.restricted_to(["s2"])
    cells = _cells(values)
    common = dict(scenario_ids=ids, n_draws=2)
    built = outcome_grid("judge/naive-rules-above", cells, identity_id="judge/key", **common)
    assert (built.condition_id, built.digest) == ("judge/naive-rules-above", recorded.digest)
    assert grid_from_cells("judge/naive-rules-above", "contradiction", ids, 2,
                           {key: row["contradiction"] for key, row in cells.items()},
                           identity_id="judge/key").digest == recorded.digest
    assert rule_grids("judge/naive-rules-above", cells, rule_id="r1", identity_id="judge/key", **common).digest == \
        rule_grids("judge/key", cells, rule_id="r1", **common).digest
    verdicts = VerdictGrids.from_cells("judge/naive-rules-above", cells, identity_id="judge/key", **common)
    assert verdicts.condition_id == "judge/naive-rules-above"
    assert verdicts.digests == VerdictGrids.from_cells("judge/key", cells, **common).digests
    assert "judge/naive-rules-above" in repr(grid) and "judge/key" not in repr(grid)


def test_an_amended_draw_s_seeds_are_derived_over_its_registered_condition_s_identity_id(tmp_path):
    from ruleforge.evaluate.amendment import AmendmentError, _identity_id
    from ruleforge.evaluate.campaign import register_condition

    condition = spec(condition_id="judge/naive-rules-above")
    draw_dir = tmp_path / "judge" / "naive-rules-above" / "draw-000"
    assert _identity_id(draw_dir, "judge/naive-rules-above") == "judge/naive-rules-above"
    register_condition(tmp_path, condition)
    assert _identity_id(draw_dir, "judge/naive-rules-above") == condition.identity_id != "judge/naive-rules-above"
    other = tmp_path / "judge" / "naive-rules-above" / "draw-001"
    with pytest.raises(AmendmentError, match="registers 'judge/naive-rules-above', not the draw's 'judge/x'"):
        _identity_id(other, "judge/x")


def test_a_block_that_fails_in_transport_is_logged_by_its_public_id_alone(tmp_path, monkeypatch, caplog):
    from ruleforge.evaluate import evaluate
    from ruleforge.evaluate.runner import STATE_COMPLETE
    from ruleforge.judges import JudgeSpec
    from tests.evaluate.helpers import inputs

    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    scenarios, rules = inputs(tmp_path)
    sent, start = [], JudgeSpec.start

    def started(self):
        judge = start(self)
        generate_once, call = judge.generate_once, judge._batch_generate_with_details

        def recording(prompts, **kwargs):
            sent.append(kwargs["request_block_id"])
            return generate_once(prompts, **kwargs)

        def raising_once(*args, **kwargs):
            if len(sent) == 1:
                raise ConnectionError("connection reset")
            return call(*args, **kwargs)

        judge.generate_once, judge._batch_generate_with_details = recording, raising_once
        return judge

    monkeypatch.setattr(JudgeSpec, "start", started)
    with caplog.at_level("WARNING"):
        result = evaluate(rules, scenarios, JudgeSpec.fake(1), out=tmp_path / "root", campaign_id="quick", draws=2,
                          condition_id="fake-1/naive")
    public = "quick/fake-1/naive/draw-000/attempt-0/block-0000"
    identity = f"quick/{names.identity_condition_id('fake-1/naive', 'naive')}/draw-000/attempt-0/block-0000"
    assert sent[:2] == [identity, identity] and identity != public and set(result.draws.values()) == {STATE_COMPLETE}
    messages = [record.getMessage() for record in caplog.records]
    assert any(public in message and "ConnectionError: connection reset" in message for message in messages)
    assert not [message for message in messages
                if identity in message or set(re.split(r"[^\w-]+", message)) & names.RECORD_KEYS]


def test_the_recorded_strings_of_the_published_data_sit_in_the_one_module():
    allowed = {"core/record_names.py", "core/record_names.json", "optimize/scoring.py", "tasks/modelspec.py"}
    source = REPO_ROOT / "src" / "ruleforge"
    holding = {str(path.relative_to(source)) for path in source.rglob("*.py") if "smrc" in path.read_text(encoding="utf-8")}
    assert holding <= allowed


def test_a_saved_config_s_recorded_block_keys_read_as_today_s_with_their_references():
    recorded = {public: names.recorded_block_key(public) for public in ("ruleset_size", "ruleforge", "training")}
    assert recorded == dict(names.RECORDED_BLOCK_KEYS)
    saved = {"run_name": "x", recorded["training"]: {"policy": "base", "policies": {"base": {"model_path": "/m"}}},
             "condition_id": "judge/" + recorded["training"] + "/${" + recorded["training"] + ".policy}",
             "judge_model": {"model_path": "${" + recorded["training"] + ".policies.${" + recorded["training"]
                             + ".policy}.model_path}"},
             "whole": ["${" + recorded["training"] + "}", 3], "other": {"figure": recorded["training"]}}
    assert names.public_blocks(saved) == {
        "run_name": "x", "training": {"policy": "base", "policies": {"base": {"model_path": "/m"}}},
        "condition_id": "judge/" + recorded["training"] + "/${training.policy}",
        "judge_model": {"model_path": "${training.policies.${training.policy}.model_path}"},
        "whole": ["${training}", 3], "other": {"figure": recorded["training"]}}
    today = {"ruleset_size": {"subset_seeds": [0]}, "n": 1}
    assert names.public_blocks(today) == today and names.public_blocks(today) is not today


def test_a_config_holding_one_block_under_both_keys_is_refused():
    both = {"ruleset_size": {"subset_seeds": [0]}, names.recorded_block_key("ruleset_size"): {"subset_seeds": [1]}}
    with pytest.raises(SamplingDesignError, match="one experiment block under its recorded and its current key"):
        names.public_blocks(both)


def test_a_block_with_no_recorded_key_is_refused():
    with pytest.raises(SamplingDesignError, match="'hierarchy' is no experiment block with a recorded key"):
        names.recorded_block_key("hierarchy")
