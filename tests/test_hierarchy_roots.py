from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ruleforge.core.artifacts import compute_scenario_id, load_jsonl_records, prompt_sha256
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.campaign import load_conditions, registers_retired
from tests.support.prompts import prompt_renderer
from ruleforge.evaluate.identity import ConditionSpec
from ruleforge.experiments.hierarchy.build import load_pairs
from ruleforge.experiments import campaign_roots as cr
from ruleforge.prompts.styles import HIERARCHY_RULE_CONTEXT, NAIVE_HIERARCHY
from ruleforge.rulesets.levels import load_rule_levels

pytestmark = pytest.mark.cluster

REPO = Path(__file__).resolve().parents[1]
PAIRS = REPO / "configs" / "paper" / "hierarchy_pairs.yaml"
RULES = REPO / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"
LEVELS = REPO / "data" / "rules" / "modelspec" / "openai_rules_all_full.jsonl"
POPULATION = REPO / "data" / "populations" / "frozen_500_r3.jsonl"
#: How many scenarios of each hierarchy condition's first draw are rendered again (every component of each).
RENDERED_SCENARIOS = 3


@pytest.fixture(scope="module")
def roots():
    """The roots the pair map names, by URI, as the artifact root holds them (skipped without them)."""
    base = Path(os.environ.get("RULEFORGE_ARTIFACT_ROOT", "/nonexistent"))
    uris = sorted({uri for pair in load_pairs(PAIRS) for uri, _ in (pair.flat, pair.hierarchy)})
    found = {uri: base / uri.removeprefix("artifact://") for uri in uris}
    if not all(path.is_dir() for path in found.values()):
        pytest.skip("needs the hierarchy appendix's roots under the artifact root")
    return found


def test_every_condition_of_the_roots_rebuilds_to_its_recorded_identity(roots):
    named = {(uri, condition) for pair in load_pairs(PAIRS) for uri, condition in (pair.flat, pair.hierarchy)}
    located = set()
    for uri, root in roots.items():
        for path in sorted(root.glob("*/*/condition.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if registers_retired(payload, path, root):  # held to its recorded fields alone
                continue
            recorded = payload.pop("fingerprint")
            payload.pop("schema_version")
            spec = ConditionSpec.from_dict(payload)
            assert (spec.fingerprint, spec.identity_id) == (recorded, payload["condition_id"]), path
            assert spec.public_id == payload["condition_id"].replace(payload["prompt_style"], spec.prompt_style)
            assert (spec.rule_levels_sha256 is not None) == (spec.prompt_style == NAIVE_HIERARCHY)
            if spec.rule_levels_sha256 is not None:
                assert (spec.rule_levels_sha256, spec.individual_rule_context) == (
                    sha256_file(LEVELS), HIERARCHY_RULE_CONTEXT)
            located.add((uri, spec.public_id))
    assert named <= located


def test_each_arm_s_grid_rebuilt_from_its_draws_is_its_estimates_own(roots, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(next(iter(roots.values())).parents[1]))
    opened = cr.open_roots({uri: uri for uri in roots}, "hierarchy roots", allow_incomplete=False,
                           allow_stack_mismatch=False)
    for pair in load_pairs(PAIRS):
        for uri, condition in (pair.flat, pair.hierarchy):
            root = opened[uri]
            spec = cr.registered_condition(root, condition, condition)
            _, grid = cr.arm_outcomes(root, spec, int(root.payload["n_decoder_draws"]), condition)
            assert grid.digest == root.payload["conditions"][spec.condition_id]["grid_digest"]


def test_the_release_renders_the_prompts_each_hierarchy_condition_recorded(roots):
    rules = load_jsonl_records(RULES)
    levels = load_rule_levels(RULES, LEVELS)
    population = {compute_scenario_id(row): row for row in load_jsonl_records(POPULATION)}
    checked = 0
    for root in roots.values():
        for condition_id, spec in sorted(load_conditions(root, skip_retired=True).items()):
            if spec.prompt_style != NAIVE_HIERARCHY:
                continue
            render = prompt_renderer(spec, rules, levels=levels)
            rows = [row for row in load_jsonl_records(root / condition_id / "draw-000" / "attempts.jsonl")
                    if row["semantic_attempt_id"] == 0]
            for row in rows[:RENDERED_SCENARIOS]:
                for component in row["components"]:
                    entry = render(population[row["scenario_id"]], component["component_id"])
                    assert prompt_sha256(entry) == component["prompt_sha256"], (condition_id, component["component_id"])
                    checked += 1
    assert checked == len(load_pairs(PAIRS)) * RENDERED_SCENARIOS * 58  # each pair's hierarchy arm
