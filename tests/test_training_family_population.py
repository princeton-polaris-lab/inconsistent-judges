from __future__ import annotations

import json
import importlib
import shutil

import pytest

from ruleforge.experiments.training import paper_families
from ruleforge.experiments.training.family_readout import TrainingFamilyError
from ruleforge.experiments.training.paper_families import PaperFamily
from ruleforge.evaluate.amendment import AmendedBundle, CapAmendment
from ruleforge.evaluate.estimates import public_amendment
from ruleforge.evaluate.identity import MANIFEST_AMENDMENTS_KEY
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.experiments.training.family_readout import FamilyArm
from tests.support.training_campaign import _edit, _roots  # noqa: F401 -- an autouse fixture
from tests import test_training_family_readout as family_tests
from tests.test_training_family_readout import LABELS, PREFIX, estimates_of, read, run_script

#: Root A (base and grpo) and root B (final alone, its reference in A), analysed: the family tests' fixture.
two_roots = family_tests.two_roots


def _ladder(root, *, old_cap, new_cap=40):
    bundle = AmendedBundle(condition_id=f"{PREFIX}/final", decoder_draw_id=1, scenario_id="e" * 64, shard=None,
                           attempts=old_cap, prompts_sha256="0" * 64, files={})
    amendment = CapAmendment(old_cap=old_cap, new_cap=new_cap, recorded_at="2026-10-14T00:00:00+00:00",
                             decided_by="a maintainer", decided_on="2026-10-14", reason="an added scenario exhausted",
                             ledger="note 143", code_state=("b" * 40, False, "2" * 64), prior_code_states=(),
                             bundles=(bundle,), conditions=(f"{PREFIX}/final",), checks={"prompts": "match"})
    disclosure = {**amendment.summary(), "bundles": [{**{k: v for k, v in bundle.to_dict().items()
                                                         if k in ("condition_id", "decoder_draw_id", "scenario_id")},
                                                      "held_by": "campaign root", "attempts_at_old_cap": old_cap,
                                                      "outcome": "accepted", "accepted_at_attempt": 21,
                                                      "attempts": 22}],
                  "code_states": {f"{PREFIX}/final": {"commit bbbbbbbbbbbb, sources 222222222222": ["draw-001"]}}}
    _edit(root / "sampling_manifest.json", lambda m: m.update(
        max_semantic_attempts=new_cap, **{MANIFEST_AMENDMENTS_KEY: [amendment.to_dict()]}))
    _edit(estimates_of(root), lambda e: e.update(max_semantic_attempts=new_cap,
                                                 **{MANIFEST_AMENDMENTS_KEY: [disclosure]}))
    return disclosure


def test_a_root_that_took_the_ladder_is_read_at_its_frozen_cap_and_its_amendment_disclosed(tmp_path, two_roots):
    before = read(tmp_path, two_roots)
    frozen = json.loads((two_roots["final"] / "sampling_manifest.json").read_text())["max_semantic_attempts"]
    disclosure = _ladder(two_roots["final"], old_cap=frozen)
    after = read(tmp_path, two_roots)
    assert {key: after[key] for key in ("arms", "comparisons", "yes_shift")} == {
        key: before[key] for key in ("arms", "comparisons", "yes_shift")}
    assert after["provenance"]["max_semantic_attempts"] == before["provenance"]["max_semantic_attempts"] == frozen
    laddered = after["provenance"]["campaigns"][1]
    assert laddered["root"] == "artifact://B/root" and laddered[MANIFEST_AMENDMENTS_KEY] == [public_amendment(disclosure)]
    assert laddered["max_semantic_attempts"] == 40 and MANIFEST_AMENDMENTS_KEY not in after["provenance"]["campaigns"][0]
    added = [line for line in after["disclosure"]["lines"] if line not in before["disclosure"]["lines"]]
    assert added == [
        f"artifact://B/root took a retry ladder, read at its campaign's frozen cap beside the other roots: Campaign "
        f"training-B: max_semantic_attempts raised from {frozen} to 40 for bundles exhausted at {frozen}. "
        f"{PREFIX}/final draw 1 scenario "
        f"{'e' * 12}: accepted after 22 attempts. Code states: {PREFIX}/final: commit bbbbbbbbbbbb, sources "
        "222222222222 (draw-001)."]
    _ladder(two_roots["final"], old_cap=frozen + 1)
    with pytest.raises(TrainingFamilyError, match=rf"designs differ in max_semantic_attempts \(artifact://A/root "
                                                 rf"{frozen}, artifact://B/root {frozen + 1}\)"):
        read(tmp_path, two_roots)


def _rebase_record(root, *, chain):
    """A rebase record for ``root``: a single rebase (schema 2) or, with ``chain``, a two-hop chain (schema 3)."""
    target = ScenarioManifest.load(root / "scenario_manifest.json").manifest_sha256
    parent = {"root": "/elsewhere/training-B", "campaign_id": "training-B", "scenario_manifest_sha256": "1" * 64,
              "scenario_source_sha256": "2" * 64, "sampling_manifest_sha256": "3" * 64}
    record = {"schema_version": 3 if chain else 2, "kind": "population_rebase", "parent": parent,
              "target": {"scenario_manifest_sha256": target, "source_sha256": "4" * 64},
              "removed_ids": ["5" * 64, "6" * 64], "added_ids": ["7" * 64, "8" * 64]}
    if chain:
        first = {"parent": {**parent, "scenario_manifest_sha256": "9" * 64, "root": "/elsewhere/older"},
                 "target": {"scenario_manifest_sha256": "1" * 64, "source_sha256": "2" * 64},
                 "removed_ids": ["a" * 64], "added_ids": ["5" * 64]}
        last = {"parent": parent, "target": dict(record["target"]), "removed_ids": record["removed_ids"],
                "added_ids": record["added_ids"]}
        record["chain"] = [first, last]
    (root / "rebase.json").write_text(json.dumps(record))


@pytest.mark.parametrize("chain", [False, True])
def test_a_rebase_record_of_either_schema_is_read_by_its_own_hop(tmp_path, two_roots, chain):
    _rebase_record(two_roots["final"], chain=chain)
    readout = read(tmp_path, two_roots)
    rebase = readout["provenance"]["campaigns"][1]["rebase"]
    assert (rebase["parent_campaign_id"], rebase["parent_scenario_manifest_sha256"]) == ("training-B", "1" * 64)
    assert (rebase["removed_ids"], rebase["added_ids"]) == (["5" * 64, "6" * 64], ["7" * 64, "8" * 64])
    assert any(line.startswith("artifact://B/root is a rebase of campaign training-B's root")
               for line in readout["disclosure"]["lines"])


POP = "pop-0123abcd"


@pytest.fixture
def renamed(tmp_path, two_roots, monkeypatch):
    """The two roots copied to their rebased names, and the family switch standing for their unrenamed roots."""
    for name in ("A", "B"):
        shutil.copytree(tmp_path / name / "root", tmp_path / name / f"root-{POP}")
    family = PaperFamily(
        arms=(FamilyArm("base", "artifact://A/root", f"{PREFIX}/base"),
              FamilyArm("final", "artifact://B/root", f"{PREFIX}/final")),
        labels={"base": LABELS["base"], "final": LABELS["final"]}, rule_set=str(tmp_path / "A" / "rules.jsonl"),
        merge_manifests={})
    monkeypatch.setattr(paper_families, "GEMMA_FAMILY", family)
    return family


def test_the_command_line_reads_a_paper_family_from_its_roots_renamed_for_a_population(tmp_path, renamed, capsys):
    out = tmp_path / "out"
    code, printed, _ = run_script(["--family", "gemma", "--population", POP, "--out-dir", str(out), "--dpi", "40"],
                                  capsys)
    assert code == 0
    written = json.loads((out / "training_family.json").read_text())
    assert [written["provenance"]["arms"][label]["root"] for label in ("base", "final")] == [
        f"artifact://A/root-{POP}", f"artifact://B/root-{POP}"]
    assert written["arms"]["base"]["label"] == LABELS["base"]


def test_family_reads_the_family_s_own_words_and_refuses_every_flag_they_can_hold():
    flags = importlib.import_module("ruleforge.experiments.training.build").FAMILY_FLAGS
    for output in paper_families.PAPER_OUTPUTS.values():
        family = output.family.on_population(POP)
        assert {word for word in family.arguments() if word.startswith("--")} <= set(flags)
    assert set(flags) == {"--arm", "--reference", "--rule-set", "--label", "--merge-manifest", "--parity",
                          "--allow-code-mismatch"}


@pytest.mark.parametrize(("argv", "message"), [
    (["--family", "gemma"], "--family and --population go together"),
    (["--population", POP], "--family and --population go together"),
    (["--family", "gemma", "--population", "pop-XYZ"], "population 'pop-XYZ' is not pop-<first 8 hex"),
    (["--family", "gemma", "--population", POP, "--reference", "base"],
     "--family gives the family's own arguments; do not also pass --reference"),
    (["--family", "gemma", "--population", POP, "--paper"], "do not also pass --paper"),
    (["--family", "gemma", "--population", POP, "--allow-code-mismatch"], "do not also pass --allow-code-mismatch"),
    (["--family", "gemma", "--population", POP, "--parity", "final=artifact://x"], "do not also pass --parity"),
])
def test_the_command_line_refuses_a_family_with_other_arguments_or_no_population(tmp_path, renamed, capsys, argv,
                                                                                 message):
    code, _, err = run_script([*argv, "--out-dir", str(tmp_path / "out")], capsys)
    assert code == 2 and message in err
    assert not (tmp_path / "out").exists()


def as_recorded(root, prefix, style):
    from ruleforge.core import record_names

    key = record_names.identity_key(style)
    recorded = prefix[: -len(style)] + key
    (root / prefix).rename(root / recorded)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            text = path.read_text(encoding="utf-8")
            if f"{prefix}/" in text:
                path.write_text(text.replace(f"{prefix}/", f"{recorded}/"), encoding="utf-8")
    for condition in (root / recorded).glob("*/condition.json"):
        _edit(condition, lambda payload: payload.update(prompt_style=key))
    return recorded


def _where_built_aside(document, base):
    def strip(value):
        if isinstance(value, dict):
            return {key: strip(inner) for key, inner in value.items()
                    if key not in ("estimates_sha256", "created_at", "merge_manifest_sha256")
                    and not (key == "sha256" and isinstance(value.get("uri"), str))}
        if isinstance(value, list):
            return [strip(inner) for inner in value]
        return value

    return strip(json.loads(json.dumps(document).replace(str(base), "BASE")))


def test_a_recorded_family_whose_ids_carry_the_style_reads_as_its_public_spelling(tmp_path, modelspec, monkeypatch):
    from ruleforge.core import record_names
    from ruleforge.experiments.training import campaign_readout
    from tests.support.training_campaign import GEMMA, run_training_campaign

    style = "forced-grpo"
    prefix = f"{GEMMA['name']}/{style}"
    record_key = record_names.identity_key(style)
    readouts, goldens = {}, {}
    for recorded in (False, True):
        base = tmp_path / ("recorded" if recorded else "public")
        monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(base))
        roots = {}
        for name, policies, family in (("A", ("base", "grpo"), None), ("B", ("final",), "base+final")):
            (base / name).mkdir(parents=True)
            root = run_training_campaign(base / name, modelspec, policies, judge={**GEMMA, "prefix": prefix},
                                        campaign_id=f"training-{name}", family=family)
            if recorded:
                assert as_recorded(root, prefix, style) == f"{GEMMA['name']}/{record_key}"
            family_tests.analysed(root, **({"allow_incomplete_conditions": True} if name == "B" else {}))
            roots.update({policy: root for policy in policies})
        if recorded:
            names = json.loads(estimates_of(roots["base"]).read_text())["conditions"]
            assert sorted(names) == [f"{GEMMA['name']}/{record_key}/{policy}" for policy in ("base", "grpo")]
        arms = [FamilyArm(label, family_tests.uri(roots[label], base), f"{prefix}/{label}")
                for label in ("base", "grpo", "final")]
        readout = read(base, roots, arms=arms)
        golden = campaign_readout.build_from_campaign(estimates_of(roots["base"]), parity={}, allow_incomplete=True)
        readouts[recorded], goldens[recorded] = _where_built_aside(readout, base), _where_built_aside(golden, base)
    assert readouts[True] == readouts[False]
    assert goldens[True] == goldens[False]
    provenance = readouts[True]["provenance"]
    assert {label: arm["condition_id"] for label, arm in provenance["arms"].items()} == {
        label: f"{prefix}/{label}" for label in ("base", "grpo", "final")}
    assert [campaign["conditions"] for campaign in provenance["campaigns"]] == [
        [f"{prefix}/base", f"{prefix}/grpo"], [f"{prefix}/final"]]
    assert readouts[True]["comparisons"]["grpo"]["flat"]["a"] == f"{prefix}/base"
    assert {arm["condition_id"] for arm in goldens[True]["arms"].values()} == {f"{prefix}/base", f"{prefix}/grpo"}
    assert f"/{record_key}/" not in json.dumps(readouts[True]) and f"/{record_key}/" not in json.dumps(goldens[True])
