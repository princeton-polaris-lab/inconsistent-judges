from __future__ import annotations

import json
import shutil
from collections import Counter

import pytest

from ruleforge.rulesets.subsets import build_manifest, diff_manifest, generate_subset_files, load_manifest
from ruleforge.evaluate.campaign import CONDITION_FNAME, condition_payload, load_conditions
from ruleforge.evaluate.identity import (
    MANIFEST_AMENDMENTS_KEY,
    SAMPLING_MANIFEST_FNAME,
    ConditionSpec,
    build_sampling_manifest,
    check_sampling_manifest,
    sampling_config_from_manifest,
)
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.files import json_sidecar_text
from ruleforge.core.record_names import public_style
from tests.identity import published as pub
from tests.support.recorded_root import recorded_payload

ROWS = pub.read_fixture(pub.CONDITIONS)
CONDITIONS = pub.rows_of(ROWS, "condition")
MANIFESTS = pub.rows_of(ROWS, "sampling_manifest")
SCENARIO_MANIFESTS = {row["sha256"]: row["payload"] for row in pub.rows_of(ROWS, "scenario_manifest")}
#: Each subset manifest by the roots it holds: the shuffled full set's for pub.SHUFFLED_FULL_ROOTS.
SUBSET_DIRS = {False: pub.REPO_ROOT / "data" / "rules" / "subset_manifest",
               True: pub.REPO_ROOT / "data" / "rules" / "subset_manifest_shuffled_full"}

CONDITIONS_PER_ROOT = {
    "fig2-2026-09-luna-pop-a22af097-rescored-lenient_last_line": 561,
    "fig2-2026-09-pop-a22af097-rescored-lenient_last_line": 1683,
    "fig2-2026-10-glimmer-pop-a22af097": 561,
    "fig2-2026-10-haiku-sum-rescored-lenient_last_line": 561,
    "fig4-2026-09-pop-a22af097-rescored-lenient_last_line": 3,
    "fig5-gemma-2026-10-07-pop-a22af097": 2, "fig5-gemma-ckpt600-2026-10-09-pop-a22af097": 1,
    "fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097": 2,
    "fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097": 1,
    "final-2026-09-luna-pop-a22af097-rescored-lenient_last_line": 2,
    "final-2026-09-pop-a22af097-rescored-lenient_last_line": 3,
    "hierarchy-aware-2026-10-glimmer": 1, "hierarchy-aware-2026-10-haiku-sum": 1,
    "hierarchy-aware-2026-10-luna-sum": 1, "hierarchy-aware-2026-10-pop-a22af097": 6,
    "k57-shuffled-2026-11": 30, "k57-shuffled-2026-11-glimmer": 10,
    "k57-shuffled-2026-11-haiku": 10, "k57-shuffled-2026-11-luna": 10, "prompt-ablation-v2-2026-10-glimmer-pop-a22af097": 3,
    "prompt-ablation-v2-2026-10-haiku-sum": 3,
    "prompt-ablation-v2-2026-10-luna-pop-a22af097-rescored-lenient_last_line": 2,
    "prompt-ablation-v2-2026-10-pop-a22af097-rescored-lenient_last_line": 2,
    "prompt-ablation-v2-open-2026-10-pop-a22af097-rescored-lenient_last_line": 4,
}
SUBSET_CONDITIONS = 3420


def _envelope(payload):
    """A ``condition.json`` payload split into the condition's fields and its envelope (version, fingerprint)."""
    fields = dict(payload)
    return fields, fields.pop("schema_version"), fields.pop("fingerprint")


def _is_subset(payload) -> bool:
    return (payload.get("extra") or {}).get("role") == "subset"


def test_the_fixture_holds_every_condition_and_manifest_of_the_published_roots():
    header = pub.rows_of(ROWS, "header")
    assert len(header) == 1 and tuple(header[0]["roots"]) == pub.ROOTS
    assert dict(Counter(row["root"] for row in CONDITIONS)) == CONDITIONS_PER_ROOT
    assert sum(CONDITIONS_PER_ROOT.values()) == 3463
    assert [row["root"] for row in MANIFESTS] == list(pub.ROOTS)
    assert {row["scenario_manifest_sha256"] for row in MANIFESTS} <= set(SCENARIO_MANIFESTS)


def test_every_published_condition_file_is_its_condition_written_again_byte_for_byte():
    failures = []
    for row in CONDITIONS:
        fields, schema_version, fingerprint = _envelope(row["payload"])
        spec = ConditionSpec.from_dict(fields)
        rewritten = (condition_payload if row["root"] in pub.PUBLIC_STYLE_ROOTS else recorded_payload)(spec)
        problems = [
            name for name, ok in (
                ("to_dict", spec.to_dict() == {**fields, "prompt_style": public_style(fields["prompt_style"])}),
                ("fingerprint", spec.fingerprint == fingerprint),
                ("identity_id", spec.identity_id == fields["condition_id"]),
                ("envelope", rewritten == row["payload"] and rewritten["schema_version"] == schema_version),
                ("bytes", pub.sha256_bytes(json_sidecar_text(rewritten).encode("utf-8")) == row["sha256"]),
                ("registration", condition_payload(spec) == {**rewritten, "prompt_style": spec.prompt_style}),
                ("directory", row["path"] == f"{spec.condition_id}/{CONDITION_FNAME}"),
            ) if not ok
        ]
        if problems:
            failures.append(f"{row['root']}/{row['path']}: {problems}")
    assert failures == []


def test_the_campaign_loader_reads_every_published_root_as_recorded(tmp_path):
    for root in pub.ROOTS:
        base = tmp_path / root
        expected = {}
        for row in CONDITIONS:
            if row["root"] != root:
                continue
            path = base / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_sidecar_text(row["payload"]), encoding="utf-8")
            expected[row["payload"]["condition_id"]] = (row["payload"]["fingerprint"],
                                                        public_style(row["payload"]["prompt_style"]))
        loaded = load_conditions(base)
        assert {cid: (spec.fingerprint, spec.prompt_style) for cid, spec in loaded.items()} == expected, root


def test_every_campaign_manifest_is_its_config_built_again():
    for row in MANIFESTS:
        payload = row["payload"]
        scenario = ScenarioManifest.from_dict(SCENARIO_MANIFESTS[row["scenario_manifest_sha256"]])
        assert pub.sha256_bytes(json_sidecar_text(scenario.to_dict()).encode("utf-8")) == row[
            "scenario_manifest_sha256"]
        conditions = [ConditionSpec.from_dict(_envelope({"schema_version": None, **entry})[0])
                      for entry in payload["conditions"]]
        rebuilt = build_sampling_manifest(
            campaign_id=payload["campaign_id"], config=sampling_config_from_manifest(payload),
            scenario_manifest=scenario, conditions=conditions, created_at=payload["created_at"],
            code_revision=payload["code_revision"], code_dirty=payload["code_dirty"],
            code_state_sha256=payload["code_state_sha256"],
        )
        recorded = {key: value for key, value in payload.items() if key != MANIFEST_AMENDMENTS_KEY}
        assert rebuilt == recorded, row["root"]
        if MANIFEST_AMENDMENTS_KEY not in payload:
            assert pub.sha256_bytes(json_sidecar_text(rebuilt).encode("utf-8")) == row["sha256"], row["root"]


def test_the_resume_check_accepts_every_campaign_manifest_as_its_own(tmp_path):
    for row in MANIFESTS:
        path = tmp_path / row["root"] / SAMPLING_MANIFEST_FNAME
        path.parent.mkdir()
        path.write_text(json_sidecar_text(row["payload"]), encoding="utf-8")
        # A resumed job's config states an amended field at its amended value, never the amendments' record.
        stated = {key: value for key, value in row["payload"].items() if key != MANIFEST_AMENDMENTS_KEY}
        assert check_sampling_manifest(path, stated) == row["payload"]


def test_every_non_subset_condition_names_a_ruleset_the_repository_commits():
    committed = pub.repo_rule_files()
    by_digest = {sha for _, sha in committed}
    named = [row["payload"] for row in CONDITIONS if not _is_subset(row["payload"])]
    assert len(named) == 3463 - SUBSET_CONDITIONS
    missing = sorted({(c["rule_set_name"], c["rule_set_sha256"]) for c in named} - set(committed))
    assert missing == []
    levels = {c.get("rule_levels_sha256") for c in named} - {None}
    assert levels <= by_digest
    assert "99eac16d5fc4c8cd028b091b6cab23badcd1573bd33d9765f0ca436e91f7130a" in by_digest  # the paper's levels


@pytest.fixture(scope="session")
def regenerated_subsets(tmp_path_factory):
    """``{shuffled full set: {(seed, file name): (digest, rules)}}``, each manifest's files regenerated and checked."""
    found = {}
    for shuffled, manifest_dir in SUBSET_DIRS.items():
        meta, entries = load_manifest(manifest_dir)
        source = pub.REPO_ROOT / meta["source_ruleset"]["path"]
        work = tmp_path_factory.mktemp("subsets")
        regenerated, digests = {}, {}
        for seed in meta["seeds"]:
            directory = work / f"seed_{seed}"
            for path in generate_subset_files(source, directory, meta["sampling_strategy"], seed):
                lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                regenerated[(seed, path.name)] = (pub.sha256_bytes(path.read_bytes()), len(lines))
            digests.update(build_manifest(work))
            shutil.rmtree(directory)
        assert len(entries) == 1400
        assert diff_manifest(entries, digests) == {"missing": [], "unexpected": [], "changed": []}
        found[shuffled] = regenerated
    return found


def test_every_subset_condition_names_a_regenerated_subset_file(regenerated_subsets):
    subsets = [row for row in CONDITIONS if _is_subset(row["payload"])]
    assert len(subsets) == SUBSET_CONDITIONS
    failures = []
    for row in subsets:
        condition, extra = row["payload"], row["payload"]["extra"]
        name = f"subset_size_{extra['subset_size']:02d}.jsonl"
        files = regenerated_subsets[row["root"] in pub.SHUFFLED_FULL_ROOTS]
        found = files.get((extra["subset_seed"], name))
        if condition["rule_set_name"] != name or found != (condition["rule_set_sha256"], extra["subset_size"]):
            failures.append(condition["condition_id"])
    assert failures == []


def test_the_fixture_is_its_rows_written_again():
    for name in pub.FIXTURE_NAMES:
        assert pub.fixture_text(pub.read_fixture(name)) == pub.committed_text(name), name


def test_no_fixture_records_a_machine_path():
    for name in pub.FIXTURE_NAMES:
        text = json.dumps(pub.read_fixture(name))
        assert "/home/" not in text and "/scratch/" not in text, name
