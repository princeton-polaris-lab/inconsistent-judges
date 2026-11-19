from __future__ import annotations

import json
import os
import warnings

import pytest

from ruleforge.judges import JudgeSpec
from ruleforge.rulesets import Ruleset
from ruleforge.evaluate.campaign import condition_payload, condition_spec_for
from ruleforge.evaluate.identity import sampling_config_from_manifest
from ruleforge.tasks import ModelSpecTask, TaskSettings
from tests.identity import published as pub
from tests.identity.judge_descriptors import (
    RECORDED_ROOT_VARIABLE,
    DESCRIPTORS,
    descriptor_of,
    judge_key,
    live_weights,
    model_config_problems,
    recorded_config,
    recorded_digests,
)
from tests.support.recorded_root import recorded_condition, recorded_payload

UNAVAILABLE_CHECKPOINTS = frozenset({"44cd554da8309c815cacbd071b0cefcbc5ffbe722529b461d19f2a01be935bfb"})


def _held_to_the_pinned_exception(unavailable: list[tuple[str, object]], what: str) -> None:
    """Fail on any unavailable weights but the pinned checkpoint; report the pinned one, never pass it silently."""
    unexpected = [name for name, digest in unavailable if digest not in UNAVAILABLE_CHECKPOINTS]
    assert unexpected == [], f"{what} whose weights this machine does not hold: {unexpected}"
    if unavailable:
        warnings.warn(f"{what} not checked over live weights (the pinned checkpoint, held nowhere here): "
                      f"{[name for name, _ in unavailable]}", stacklevel=2)


@pytest.fixture(scope="module")
def published():
    rows = pub.read_fixture(pub.CONDITIONS)
    manifests = {row["root"]: row["payload"] for row in pub.rows_of(rows, "sampling_manifest")}
    return pub.rows_of(rows, "condition"), manifests


@pytest.fixture(scope="module")
def descriptors():
    return {judge_key(descriptor["key"]): descriptor for descriptor in json.loads(DESCRIPTORS.read_text("utf-8"))}


@pytest.fixture
def frozen_identities(tmp_path, monkeypatch, descriptors):
    """Each descriptor's judge identity, a local judge's read from a weights directory of the descriptor's name."""
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "unused"))
    monkeypatch.delenv("RULEFORGE_BATCH_ROOT", raising=False)
    identities = {}
    for index, (key, descriptor) in enumerate(sorted(descriptors.items(), key=str)):
        overrides = {}
        if descriptor["config"]["model_type"] == "local":
            directory = tmp_path / str(index) / descriptor["key"]["model_path"]
            directory.mkdir(parents=True)
            for name, text in descriptor["files"].items():
                (directory / name).write_text(text, encoding="utf-8")
            overrides["model_path"] = str(directory)
        identities[key] = JudgeSpec.from_config(descriptor["config"], **overrides).identity(verify_weights=False)
    return identities


def rebuilt(payload, manifest, identity):
    """``condition_spec_for`` over the judge's identity, the Model Spec task's and the campaign's config."""
    recorded = recorded_condition(payload)
    task = ModelSpecTask().identity(TaskSettings(recorded.prompt_style, recorded.individual_rule_context,
                                                 recorded.judge_type))
    return condition_spec_for(
        condition_id=recorded.condition_id, config=sampling_config_from_manifest(manifest), judge=identity,
        task=task, ruleset=Ruleset(rules=(), name=recorded.rule_set_name, sha256=recorded.rule_set_sha256),
        evaluation_mode=recorded.evaluation_mode, rule_levels_sha256=recorded.rule_levels_sha256,
        extra=payload.get("extra") or {})


def _differs(row, spec) -> bool:
    payload = row["payload"]
    written = (condition_payload if row["root"] in pub.PUBLIC_STYLE_ROOTS else recorded_payload)(spec)
    return (spec != recorded_condition(payload) or written != payload
            or spec.fingerprint != payload["fingerprint"] or spec.identity_id != payload["condition_id"])


def test_every_published_judge_has_one_descriptor(published, descriptors):
    conditions, _ = published
    assert {judge_key(row["payload"]) for row in conditions} == set(descriptors)
    assert len(conditions) == 3463
    text = DESCRIPTORS.read_text("utf-8")
    assert "/home/" not in text and "/scratch/" not in text and "/smrc/" not in text


def test_every_published_condition_is_rebuilt_from_its_judge_and_task_identities(published, frozen_identities):
    conditions, manifests = published
    identities = frozen_identities
    differing = []
    for row in conditions:
        payload = row["payload"]
        spec = rebuilt(payload, manifests[row["root"]], identities[judge_key(payload)])
        if _differs(row, spec):
            differing.append(f"{row['root']}/{payload['condition_id']}")
    assert differing == []


@pytest.mark.cluster
def test_every_published_local_condition_is_rebuilt_over_its_live_weights(published, descriptors):
    from ruleforge.env import artifact_root

    if not os.environ.get("RULEFORGE_MODEL_ROOT"):
        pytest.skip("needs RULEFORGE_MODEL_ROOT")
    root = artifact_root()
    conditions, manifests = published
    identities: dict[tuple, object] = {}
    local = [row for row in conditions if row["payload"]["model_type"] == "local"]
    assert local, "the identity fixture holds no local condition"
    checked, unavailable, differing = 0, [], []
    for row in local:
        payload = row["payload"]
        live = live_weights(root, row["root"], payload)
        if live is None:
            unavailable.append((f"{row['root']}/{payload['condition_id']}", payload.get("model_weights_sha256")))
            continue
        key = (judge_key(payload), str(live))
        if key not in identities:
            config = {**descriptors[judge_key(payload)]["config"], "model_path": str(live)}
            identities[key] = JudgeSpec.from_config(config).identity(verify_weights=True)
        spec = rebuilt(payload, manifests[row["root"]], identities[key])
        checked += 1
        if _differs(row, spec):
            differing.append(f"{row['root']}/{payload['condition_id']}")
    assert differing == []
    assert checked + len(unavailable) == len(local)
    _held_to_the_pinned_exception(unavailable, "local conditions")


@pytest.mark.cluster
def test_every_descriptor_is_what_its_recorded_source_gives(published, monkeypatch):
    from ruleforge.env import artifact_root

    if not os.environ.get("RULEFORGE_MODEL_ROOT"):
        pytest.skip("needs RULEFORGE_MODEL_ROOT")
    monkeypatch.setenv(RECORDED_ROOT_VARIABLE, os.environ["RULEFORGE_MODEL_ROOT"])
    root = artifact_root()
    digests = recorded_digests(published[0])
    unavailable, differing, departures = [], [], []
    recorded = json.loads(DESCRIPTORS.read_text("utf-8"))
    assert recorded, f"{DESCRIPTORS} holds no descriptor"
    for descriptor in recorded:
        cfg = recorded_config(root, descriptor["source"])
        assert cfg is not None, f"{descriptor['source']} is not there"
        again = descriptor_of(root, descriptor["source"], cfg, digests)
        if again is None:
            unavailable.append((descriptor["source"], descriptor["key"]["model_weights_sha256"]))
        elif again != descriptor:
            differing.append(descriptor["source"])
        departures += model_config_problems(root, descriptor, published[0])
    assert differing == [] and departures == []
    _held_to_the_pinned_exception(unavailable, "descriptors")
