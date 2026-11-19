from __future__ import annotations

import copy
import dataclasses
import functools
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from ruleforge.experiments import table
from ruleforge.experiments.training import paper_families
from ruleforge.experiments.training.family_readout import HOP_PARENT_KEYS, FamilyArm, TrainingFamilyError
from ruleforge.experiments.training.paper_families import (
    QWEN_FAMILY,
    QWEN_FIGURES,
    QWEN_GOLDEN,
    GEMMA_FAMILY,
    GEMMA_FIGURES,
    PAPER_OUTPUTS,
    PaperFamily,
    paper_refusal,
)
from ruleforge.core.paths import absolute_strings
from ruleforge.evaluate import campaign as campaign_module
from ruleforge.evaluate import identity as design_module
from tests.support import rebase as rebase_module
from ruleforge.evaluate import runner as runner_module
from ruleforge.evaluate.aggregate import analyze_campaign
from tests.support.rebase import execute_rebase, plan_rebase
from ruleforge.evaluate.chain import rebase_hop
from tests.support.rebase_checks import check_retained
from tests.support.training_campaign import (
    N_SCENARIOS,
    QWEN,
    RUNTIME,
    _roots,  # noqa: F401 - an autouse fixture
    assert_panel_a_draws_the_flat_rates_alone,
    mark_final,
    run_training_campaign,
    source_of,
    write_parity,
)
from tests.test_training_family_readout import N_PERMUTATIONS, N_RESAMPLES, read
from tests.test_training_gemma_frozen import FLOOR_P, PROGRAM
from tests.test_training_paper_families import DPI, MERGE, PANEL_FILES, copy_into_root, run

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ARTIFACT_ROOT = os.environ.get("RULEFORGE_ARTIFACT_ROOT")
PREFIX = QWEN["prefix"]
LABELS = {"base": "Qwen3-8B", "final": "GRPO (checkpoint-300)"}
PARITY = tuple(f"parity/final/{arm}_report.json" for arm in ("prefill", "decode"))
CODE = {name: {"code_revision": letter * 40, "code_dirty": False, "code_state_sha256": digit * 64}
        for name, letter, digit in (("A", "a", "1"), ("B", "b", "2"), ("rebase", "c", "3"), ("resume", "d", "4"),
                                    ("retry", "e", "5"))}
#: The scenario the revised population replaces (a wildchat row) and the one it adds in its place.
REMOVED_INDEX = 7
ADDED = {"prompt": f"prompt {N_SCENARIOS}", "response": f"response {N_SCENARIOS}", "source": source_of(REMOVED_INDEX)}
APPENDIX = "qwen"
HEAD = ("not the appendix's Qwen3-8B family (ruleforge.experiments.training.paper_families.QWEN_FAMILY): ")


def stand_in(tmp_path: Path, a: str = "artifact://A/root", b: str = "artifact://B/root", **changes) -> PaperFamily:
    """The synthetic stand-in for :data:`QWEN_FAMILY`."""
    return dataclasses.replace(PaperFamily(
        arms=(FamilyArm("base", a, f"{PREFIX}/base"), FamilyArm("final", b, f"{PREFIX}/final")),
        labels=dict(LABELS), rule_set=str(tmp_path / "A" / "rules.jsonl"),
        merge_manifests={"final": f"artifact://{MERGE}"},
        parity={"final": tuple(f"artifact://{path}" for path in PARITY)}, allow_code_mismatch=True,
    ), **changes)


def qwen_root(directory: Path, modelspec, policies, code: str, monkeypatch, **kwargs) -> Path:
    """A synthetic training root of ``policies`` on Qwen3-8B's fp32 engine, its runs at ``CODE[code]``."""
    directory.mkdir(exist_ok=True)
    monkeypatch.setattr(campaign_module, "code_state", lambda: dict(CODE[code]))
    return run_training_campaign(directory, modelspec, policies, judge=QWEN, campaign_id=f"training-{directory.name}",
                                **kwargs)


def finalised(root: Path, rules: Path, **flags) -> Path:
    """``root`` analysed and marked final, as a 500-scenario root would be."""
    analyze_campaign(root, n_resamples=N_RESAMPLES, n_permutations=N_PERMUTATIONS, rule_set_path=rules, **flags)
    mark_final(root)
    return root


def write_parities(tmp_path: Path) -> list[Path]:
    """The stand-in member's prefill and decode reports, passing the gate, at :data:`PARITY`."""
    decode = {"rho": 0.9998, "rho_ci": [0.9990, 1.0007], "rho_halfwidth": 0.00085}
    return [write_parity(tmp_path, "final"), write_parity(tmp_path, "final", "decode", buckets={"all": decode})]


@pytest.fixture
def parents(tmp_path, modelspec, monkeypatch):
    """Root A (base and grpo) and root B (final alone), each run at its own commit and not yet analysed."""
    monkeypatch.setattr(runner_module, "_package_version", lambda name: RUNTIME[f"{name}_version"])
    roots = {"A": qwen_root(tmp_path / "A", modelspec, ("base", "grpo"), "A", monkeypatch),
             "B": qwen_root(tmp_path / "B", modelspec, ("final",), "B", monkeypatch, family="base+final")}
    write_parities(tmp_path)
    return roots


def read_family(tmp_path: Path, roots, family: PaperFamily, **kwargs):
    """``family``'s readout, the parity reports resolved under the artifact root (as the build script reads them)."""
    from ruleforge.core.paths import resolve

    return read(tmp_path, {"base": roots["A"]}, arms=list(family.arms), labels=dict(family.labels),
                merge_manifests=dict(family.merge_manifests),
                parity={label: [resolve(uri) for uri in uris] for label, uris in family.parity.items()},
                allow_code_mismatch=family.allow_code_mismatch, **kwargs)


@pytest.fixture
def driver(monkeypatch, tmp_path):
    """The script as a module, over a scratch checkout, the synthetic family and reproducibly dated PDFs."""
    module = importlib.import_module("ruleforge.experiments.training.build")
    monkeypatch.setattr(module, "repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paper_families, "QWEN_FAMILY", stand_in(tmp_path))
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    return module


def _shifted(readout: dict, path: str, by) -> dict:
    edited = copy.deepcopy(readout)
    *parents, last = path.split(".")
    node = edited
    for key in parents:
        node = node[key]
    node[last] = node[last] + by if isinstance(by, float) else by
    return edited


def appendix_panels(tmp_path: Path, figures=QWEN_FIGURES) -> list[Path]:
    return [tmp_path / "repo" / f"{figures[panel]}.{suffix}" for panel in "a" for suffix in ("pdf", "png")]


def test_the_appendix_switch_is_qwen3_8b_s_base_against_checkpoint_300_on_the_paper_population():
    assert [(arm.label, arm.root, arm.condition_id) for arm in QWEN_FAMILY.arms] == [
        ("base", "artifact://sampling/fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097",
         "Qwen3-8B/figure5/base"),
        ("pick", "artifact://sampling/fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097", "Qwen3-8B/figure5/pick"),
    ]
    assert QWEN_FAMILY.labels == {"base": "Qwen3-8B", "pick": "GRPO (checkpoint-300)"}
    assert (REPO_ROOT / QWEN_FAMILY.rule_set).is_file()
    selection = "artifact://selection/qwen-grpo-mix-2026-10-11"
    assert QWEN_FAMILY.merge_manifests == {
        "pick": f"{selection}/modal_root/finetuning/qwen-grpo-mix-2026-10-11/merged/checkpoint-300/merge_manifest.json"}
    assert QWEN_FAMILY.parity == {"pick": (
        f"{selection}/selection/checkpoints/checkpoint-300/parity/prefill_report.json",
        f"{selection}/selection/checkpoints/checkpoint-300/parity/decode_report.json")}
    assert QWEN_FAMILY.allow_code_mismatch is True
    assert QWEN_GOLDEN == "results/training/qwen.json"
    assert QWEN_FIGURES == {"a": "figures/grpo_qwen_inconsistency"}
    assert {name: (outputs.family, outputs.golden, outputs.figures) for name, outputs in PAPER_OUTPUTS.items()} == {
        "gemma": (GEMMA_FAMILY, "results/training/gemma.json", GEMMA_FIGURES),
        APPENDIX: (QWEN_FAMILY, QWEN_GOLDEN, QWEN_FIGURES)}
    assert PAPER_OUTPUTS["gemma"].head == ("not the paper's Gemma family "
                                             "(ruleforge.experiments.training.paper_families.GEMMA_FAMILY): ")
    assert PAPER_OUTPUTS[APPENDIX].head == HEAD
    assert GEMMA_FAMILY.parity == {} and GEMMA_FAMILY.allow_code_mismatch is False


def test_the_arguments_name_the_parity_reports_and_the_code_override_after_the_paper_s_flags():
    arguments = QWEN_FAMILY.arguments()
    assert arguments[arguments.index("--merge-manifest"):] == [
        "--merge-manifest", f"pick={QWEN_FAMILY.merge_manifests['pick']}",
        "--parity", f"pick={QWEN_FAMILY.parity['pick'][0]}",
        "--parity", f"pick={QWEN_FAMILY.parity['pick'][1]}", "--allow-code-mismatch"]
    assert "--parity" not in GEMMA_FAMILY.arguments() and "--allow-code-mismatch" not in GEMMA_FAMILY.arguments()


@pytest.fixture
def family_roots(tmp_path, parents):
    """The two parents analysed and final: root B with its reference missing there (allowed: it is in root A)."""
    finalised(parents["A"], tmp_path / "A" / "rules.jsonl")
    finalised(parents["B"], tmp_path / "B" / "rules.jsonl", allow_incomplete_conditions=True)
    return parents


def test_only_a_readout_of_exactly_the_appendix_family_passes_its_gate(tmp_path, family_roots):
    family = stand_in(tmp_path)
    with pytest.raises(TrainingFamilyError, match="the roots ran different code"):
        read_family(tmp_path, family_roots, dataclasses.replace(family, allow_code_mismatch=False))
    readout = read_family(tmp_path, family_roots, family)
    assert paper_refusal(readout, family, paper=APPENDIX) is None
    assert {label: sorted(arm["report"] for arm in arms.values())
            for label, arms in readout["disclosure"]["parity_b"].items()} == {
        "final": sorted(f"artifact://{path}" for path in PARITY)}
    assert paper_refusal(readout, paper=APPENDIX).startswith(HEAD + "its arms are not the appendix's ['base', 'pick']")
    assert paper_refusal(readout).startswith("not the paper's Gemma family (")

    parity = readout["disclosure"]["parity_b"]["final"]
    for edited, message in (
        (_shifted(readout, "provenance.allow_code_mismatch", False), "it was read without allow_code_mismatch"),
        (_shifted(readout, "disclosure.parity_b", {}),
         "no parity report for final (the appendix's Qwen3-8B family has one)"),
        (_shifted(readout, "disclosure.parity_b", None), "no parity report for final"),
        (_shifted(readout, "disclosure.parity_b.final", {"prefill": parity["prefill"]}),
         "final's parity report is not the appendix's"),
        (_shifted(readout, "disclosure.parity_b.final", ["a report"]), "final's parity report is not the appendix's"),
        (_shifted(readout, "disclosure.parity_b.final.decode", {**parity["decode"], "report": "artifact://elsewhere.json"}),
         "final's parity report is not the appendix's"),
        (_shifted(readout, "disclosure.parity_b.final.decode", "a report"), "final's parity report is not the appendix's"),
    ):
        refusal = paper_refusal(edited, family, paper=APPENDIX)
        assert refusal is not None and refusal.startswith(HEAD) and message in refusal, (message, refusal)
    bare = dataclasses.replace(family, parity={}, allow_code_mismatch=False)
    refusal = paper_refusal(readout, bare)
    assert "parity report given for final (the paper's Gemma family has none)" in refusal
    assert "it was read with allow_code_mismatch" in refusal


def test_paper_builds_the_family_it_names_into_its_own_files(tmp_path, family_roots, driver, capsys):
    family = stand_in(tmp_path)
    out = tmp_path / "out"
    argv = [*family.arguments(), "--paper", APPENDIX, "--out-dir", out, "--dpi", DPI]
    code, printed, err = run(driver, argv, capsys)
    assert code == 0, err
    assert printed == (out / "training_family.md").read_text()
    assert all(path.is_file() and path.stat().st_size > 0 for path in appendix_panels(tmp_path))
    assert not any(path.exists() for path in appendix_panels(tmp_path, GEMMA_FIGURES))
    readout = json.loads((out / "training_family.json").read_text())
    assert paper_refusal(readout, paper=APPENDIX) is None
    assert sorted(arm["report"] for arm in readout["disclosure"]["parity_b"]["final"].values()) == sorted(
        f"artifact://{path}" for path in PARITY)
    code, _, err = run(driver, [*family.arguments(), "--paper", APPENDIX, "--out-dir", tmp_path / "again"], capsys)
    assert code == 2 and "grpo_qwen_inconsistency.pdf" in err and "pass --force" in err
    code, _, err = run(driver, [*family.arguments(), "--paper", "--out-dir", tmp_path / "gemma"], capsys)
    assert code == 2 and "--paper: this readout is not the paper's Gemma family (" in err
    assert not (tmp_path / "gemma").exists()
    code, _, err = run(driver, [*family.arguments(), "--paper", "figure6", "--out-dir", tmp_path / "x"], capsys)
    assert code == 2 and "invalid choice: 'figure6'" in err
    missing = [word.replace("prefill_report", "missing_report") for word in family.arguments()]
    code, _, err = run(driver, [*missing, "--out-dir", tmp_path / "x"], capsys)
    assert code == 2 and "missing_report.json" in err and not (tmp_path / "x").exists()
    escaping = [word.replace("artifact://", "artifact://../") if "prefill_report" in word else word
                for word in family.arguments()]
    code, _, err = run(driver, [*escaping, "--out-dir", tmp_path / "x"], capsys)
    assert code == 2 and "escapes the artifact root" in err and not (tmp_path / "x").exists()


def test_from_golden_redraws_the_appendix_panel_and_nothing_into_results(tmp_path, family_roots, driver, capsys):
    family = stand_in(tmp_path)
    built = tmp_path / "built"
    assert run(driver, [*family.arguments(), "--out-dir", built, "--dpi", DPI], capsys)[0] == 0
    golden = built / "training_family.json"
    appendix = [word for panel, path in zip("a", appendix_panels(tmp_path)[::2])
                for word in (f"--figure-{panel}", path)]
    code, printed, _ = run(driver, ["--from_golden", golden, *appendix, "--dpi", DPI], capsys)
    assert code == 0 and printed == (built / "training_family.md").read_text()
    for name, path in zip(PANEL_FILES[::2], appendix_panels(tmp_path)[::2]):
        assert path.read_bytes() == (built / name).read_bytes(), name
    path = tmp_path / "repo" / "results" / "training" / "grpo_qwen_inconsistency.pdf"
    code, _, err = run(driver, ["--from_golden", golden, "--figure-a", path], capsys)
    assert code == 2 and f"--figure-a {path} would be among the paper's files (results)" in err and not path.exists()
    drawn = {path: path.read_bytes() for path in appendix_panels(tmp_path)[::2]}
    (listed := tmp_path / "list.json").write_text("[1]")
    code, _, err = run(driver, ["--from_golden", listed, *appendix], capsys)
    assert code == 2 and "list.json is no training family readout (a JSON list, not an object)" in err
    assert {path: path.read_bytes() for path in drawn} == drawn


def revised(tmp_path: Path) -> Path:
    """The synthetic population with row :data:`REMOVED_INDEX` swapped for :data:`ADDED` (its stratum kept)."""
    rows = [json.loads(line) for line in (tmp_path / "A" / "training_population.jsonl").read_text().splitlines()]
    rows[REMOVED_INDEX] = dict(ADDED)
    path = tmp_path / "revised.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_the_appendix_runbook_on_synthetic_roots(tmp_path, tmp_path_factory, parents, modelspec, driver, capsys,
                                                 monkeypatch):
    paper_root = tmp_path_factory.mktemp("paper_root")
    population = revised(tmp_path)
    monkeypatch.setattr(rebase_module, "code_state", lambda: dict(CODE["rebase"]))
    monkeypatch.setattr(rebase_module, "write_code_snapshot", lambda root, state: None)
    monkeypatch.setattr(design_module, "FINAL_SCENARIO_COUNT", N_SCENARIOS)
    finalised(parents["A"], tmp_path / "A" / "rules.jsonl")
    finalised(parents["B"], tmp_path / "B" / "rules.jsonl", allow_incomplete_conditions=True)
    before = read_family(tmp_path, parents, stand_in(tmp_path))
    rebased = {name: paper_root / "sampling" / f"{name}-pop" for name in "AB"}
    for name, parent in parents.items():
        execute_rebase(plan_rebase(parent, population), rebased[name])
        assert all(path.stat().st_nlink == 1 for path in rebased[name].rglob("*") if path.is_file())
    qwen_root(tmp_path / "A", modelspec, ("base", "grpo"), "resume", monkeypatch, population=population,
              campaign_root=rebased["A"], campaign_mode="final")
    qwen_root(tmp_path / "B", modelspec, ("final",), "resume", monkeypatch, family="base+final",
              population=population, campaign_root=rebased["B"], campaign_mode="final")
    draws = sorted((rebased["B"] / PREFIX / "final").glob("draw-*"))
    ledgers = [(draw / "attempts.jsonl").read_bytes() for draw in draws]
    qwen_root(tmp_path / "B", modelspec, ("final",), "retry", monkeypatch, family="base+final",
              population=population, campaign_root=rebased["B"], campaign_mode="final")
    assert [(draw / "attempts.jsonl").read_bytes() for draw in draws] == ledgers
    for draw, ledger in zip(draws, ledgers):
        # its run row's ledger boundary is the ledger's end: it appended nothing (stack.run_spans)
        last = json.loads((draw / "runs.jsonl").read_text().splitlines()[-1])
        assert (last["code_revision"], last["ledger_rows"]) == (CODE["retry"]["code_revision"], ledger.count(b"\n"))
    for name in "AB":
        assert check_retained(parents[name], rebased[name])["problems"] == []
    record_path = rebased["B"] / "rebase.json"
    written_b = json.loads(record_path.read_text())
    written_b["parent"]["root"] = written_b["chain"][0]["parent"]["root"] = "/scratch/elsewhere/B/root"
    record_path.write_text(json.dumps(written_b))
    assert check_retained(parents["B"], rebased["B"])["problems"] == []
    finalised(rebased["A"], tmp_path / "A" / "rules.jsonl")
    finalised(rebased["B"], tmp_path / "B" / "rules.jsonl", allow_incomplete_conditions=True)
    copy_into_root(tmp_path, paper_root, [MERGE, *PARITY])
    shutil.move(str(tmp_path / "B" / "grpo"), str(tmp_path / "weights_moved_away"))
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(paper_root))
    family = stand_in(tmp_path, "artifact://sampling/A-pop", "artifact://sampling/B-pop")
    monkeypatch.setattr(paper_families, "QWEN_FAMILY", family)
    out = tmp_path / "paper_readout"
    code, _, err = run(driver, [*family.arguments(), "--paper", APPENDIX, "--out-dir", out, "--dpi", DPI], capsys)
    assert code == 0, err
    after = json.loads((out / "training_family.json").read_text())
    assert paper_refusal(after, paper=APPENDIX) is None
    from ruleforge.core.artifacts import compute_scenario_id

    removed, added = compute_scenario_id({key: f"{key} {REMOVED_INDEX}" for key in ("prompt", "response")} | {
        "source": source_of(REMOVED_INDEX)}), compute_scenario_id(ADDED)
    campaigns = {campaign["root"]: campaign for campaign in after["provenance"]["campaigns"]}
    for name, label in (("A", "base"), ("B", "final")):
        record = campaigns[f"artifact://sampling/{name}-pop"]
        assert {key: record[key] for key in ("code_revision", "code_dirty", "code_state_sha256")} == CODE["rebase"]
        written = json.loads((rebased[name] / "rebase.json").read_text())
        assert record["rebase"] == {
            "parent_campaign_id": f"training-{name}",
            "parent_scenario_manifest_sha256": before["provenance"]["scenario_manifest_sha256"],
            "parent_sampling_manifest_sha256": written["parent"]["sampling_manifest_sha256"],
            "removed_ids": [removed], "added_ids": [added],
            "chain": [{"parent": {key: written["parent"][key] for key in HOP_PARENT_KEYS},
                       "target": rebase_hop(written)["target"], "removed_ids": [removed], "added_ids": [added]}],
            "run_code_states": {f"{PREFIX}/{label}": [
                {**CODE[phase], "draws": ["draw-000", "draw-001"]}
                for phase in (name, "resume", *(("retry",) if name == "B" else ()))]},
        }
        assert "root" not in record["rebase"]["chain"][0]["parent"]
        assert written["parent"]["root"] == ("artifact://A/root" if name == "A" else "/scratch/elsewhere/B/root")
    assert list(absolute_strings(after)) == []
    lines = after["disclosure"]["lines"]
    assert lines[1] == (
        "artifact://sampling/A-pop is a rebase of campaign training-A's root from its population (scenario "
        f"manifest {before['provenance']['scenario_manifest_sha256'][:12]}...): scenario(s) {removed[:12]}... removed "
        f"and {added[:12]}... added in their place, the added judged by a resume under the parent's campaign id, at "
        "the seeds it gives them; every other judgment is the parent's. Its sampling manifest's code state is the "
        "rebase's; its draws' run logs record these code states, each logged by a run that started on the draw, "
        f"whether it judged a row there or found nothing left to judge: {PREFIX}/base: commit aaaaaaaaaaaa, sources "
        "111111111111 (draw-000, draw-001), commit dddddddddddd, sources 444444444444 (draw-000, draw-001).")
    # the retry's state among B's, said to be a run's that started on the draws, not one that judged them
    assert lines[2].startswith("artifact://sampling/B-pop is a rebase of campaign training-B's root")
    assert lines[2].endswith(
        f"found nothing left to judge: {PREFIX}/final: commit bbbbbbbbbbbb, sources 222222222222 (draw-000, draw-001), "
        "commit dddddddddddd, sources 444444444444 (draw-000, draw-001), commit eeeeeeeeeeee, sources 555555555555 "
        "(draw-000, draw-001).")
    assert not any("judged its draws" in line or "runs that judged" in line for line in lines)
    assert not any("is a rebase" in line for line in before["disclosure"]["lines"])
    assert "rebase" not in json.dumps(before["provenance"]["campaigns"])
    golden = tmp_path / "repo" / QWEN_GOLDEN
    golden.parent.mkdir(parents=True)
    shutil.copyfile(out / "training_family.json", golden)
    rerendered = tmp_path / "rerendered"
    for suffix in ("pdf", "png"):
        figures = [word for panel in "a" for word in (f"--figure-{panel}", rerendered / f"training_{panel}.{suffix}")]
        assert run(driver, ["--from_golden", golden, *figures, "--dpi", DPI], capsys)[0] == 0
    for name, path in zip(PANEL_FILES, appendix_panels(tmp_path)):
        assert (rerendered / name).read_bytes() == path.read_bytes(), name
PAPER_MANIFEST = "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"
EARLIER_MANIFEST = "6f528dcba7feac97055e6f45c909d1e23943dcc15651b6ab4174332ccd44b9b3"
FIRST_MANIFEST = "8f74b7195f2724176f36dd15fd4579500688c6b76ddf3d635f4e265ec7d9d10a"
FROZEN_500_MANIFEST = "338d262861a49975516615130b7d79580d705095b6b348eb5163af073894668c"
REMOVED_287 = "85c5861ce0957f36297b0e442e94f2a299cc5b755fe0c5a01b89f0ef6b04006e"
ADDED_R = "354d1fb5d13cd4c813601e98ee9cb3619bd7af47d3fa76a22d619ef71790313e"
REMOVED_FIVE = ["cdafa56b9a733b982d67e3432367cc47855e02a98ecdb34e66f4a9d8e2ce7689",
                "f1f8265924bc5316f6e90f3762f2b5ebab10d595632ec8407140d5ccfea15df3",
                "6c7ff3470d1289bd508e83c22369181dc7395dd2ec1b293cfa2032d90b6ec299",
                "5cdb45d83ce2aa18933c4257ebc0ec2f69a04fc9744ede47cb4b7091e62a231d",
                "3a982a00b0febc80cb221571ba9de2f6ec0798218adc80ae03dd002e3f00b659"]
ADDED_FIVE = ["1cf46bc7eb8518c6dd4bacde4983a9736ff73480d583663bf34629757f7279c3",
              "90c97139bcdcb737e43fe41aa1edfc5df582bbf81b8b83ab5c3de8e0a67c0a8a",
              "2ce5ba24a8aa8fcb24dd56b9a5bc8fb2854e1f99631523d85224974bc81a6279",
              "898eb72a2e045393d6a3d0865edcc98862b848f3a586fa008c5710a23d780ae3",
              "f319558563c7042812c447495831719214537553fa223f41b8b5b5db67b109fc"]
PARENTS = {"Qwen3-8B/figure5/base": ("fig5-grpo-v2-fp32-2026-09-26", "2594706495aa8690e307b8961ddac2a5b6fd1910",
                                     "bb40e94c98023b8a6b73df848149a242fd5301ea00b3cd8faddc8afcf580526c",
                                     "4406bc4e40bbc2ed1deecddc100366ecebc815b497ba9aaf301a1ef9f719945e",
                                     "76e694db490cfa62360124bdb86fb06d53d14b785b90b9ea9b53b7d285f88464"),
           "Qwen3-8B/figure5/pick": ("fig5-qwen-mix-2026-10-11-checkpoint-300",
                                     "882e55c3e219ee45b4cc050e9fce45bbce173cc6",
                                     "126230343a15fdde9b7e800c604a9f0599ea4aac082fdcf95598ddfd7b09663d",
                                     "2d3cedde6c89baf6d69c539db3533f4663791586e21309504165034c5ea7c67f",
                                     "f7de9819b809f0f12efb6b2c406e86621b07e14f50e6ce2b85e2c6c60b11d42f")}
#: The checkpoint-300 merge manifest and parity B reports, as the Hub publishes them, are pinned by SHA-256.
RECORDS = {"merge_manifest": "c3fa949f4773e80198c39e8f655c60758e506f788fefd91216c6fdf127c39591",
           "prefill": "1eb2c3125e938c90244153a82df933d9c0c2ff95bd226eed009229b42e6e51d7",
           "decode": "f39cf8c3a888385e3e0c9dc90abd3a640675218ab7c2bd896d9233daa9644a92"}
FIRST_REBASE_COMMIT = "55aa6a2d456ecc025912645a5a1fc37f3feca46e"
SECOND_REBASE_COMMIT = "87a6ba4f68777c6284f13786aa3c94ef421e3efb"
PAPER_REBASE = "57bfa3dc8b77fc39c6314dd9faed3c4f80961906"
CAMPAIGNS = {
    "artifact://sampling/fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-a22af097": (
        "fig5-grpo-v2-fp32-2026-09-26", "48407fa7bf5844c0ff3f16d1073eae572700374675ac722f3f237671d1f1b55a",
        ["Qwen3-8B/figure5/base"]),
    "artifact://sampling/fig5-qwen-mix-2026-10-11-checkpoint-300-pop-a22af097": (
        "fig5-qwen-mix-2026-10-11-checkpoint-300", "35b7ea36190fd4f069a3cfbc68e84841d38e02669fbcddeaee7668b071cd122f",
        ["Qwen3-8B/figure5/pick"]),
}


@functools.lru_cache(maxsize=None)
def _population_swap() -> tuple[list[str], list[str]]:
    swap = json.loads((REPO_ROOT / "tests" / "fixtures" / "training" / "population_swap.json").read_text())
    return swap["removed"], swap["added"]


@pytest.fixture(scope="module")
def installed(repo_root):
    """The installed golden, byte for byte the readout of the two published roots."""
    return json.loads((repo_root / QWEN_GOLDEN).read_text(encoding="utf-8"))


@pytest.mark.frozen
def test_the_installed_golden_is_the_appendix_family_on_the_paper_population(installed):
    assert paper_refusal(installed, paper=APPENDIX) is None and list(absolute_strings(installed)) == []
    assert (installed["source"], installed["schema_version"], installed["metric"]) == ("training_family", 1, "flat")
    assert installed["policies"] == ["base", "pick"] and installed["reference_policy"] == "base"
    provenance = installed["provenance"]
    assert provenance["family"] == {"declared_on": "the command line", "labels": ["base", "pick"], "reference": "base",
                                    "members": ["pick"], "holm_family_size": 1}
    assert (provenance["scenario_manifest_sha256"], provenance["n_scenarios"], provenance["n_decoder_draws"]) == (
        PAPER_MANIFEST, 500, 5)
    assert (provenance["scenario_set"], provenance["scenario_source_sha256"]) == (
        "frozen_500_r3.jsonl", "c16f66f4ad3fc93fad46bf31efaa8325f30fb750104d1e13a9326b7cd46b10d4")
    assert provenance["allow_code_mismatch"] is True and provenance["complete"] is True
    campaigns = {campaign["root"]: campaign for campaign in provenance["campaigns"]}
    assert list(campaigns) == [arm.root for arm in QWEN_FAMILY.arms] == list(CAMPAIGNS)
    for root, (campaign_id, sha256, conditions) in CAMPAIGNS.items():
        assert (campaigns[root]["campaign_id"], campaigns[root]["estimates_sha256"], campaigns[root]["conditions"]) == (
            campaign_id, sha256, conditions)
        assert campaigns[root]["estimates"] == f"{root}/aggregate/estimates.json"
        # a rebased root's sampling manifest records the rebase's code state, the merge clean
        assert (campaigns[root]["code_revision"], campaigns[root]["code_dirty"], campaigns[root]["state"]) == (
            PAPER_REBASE, False, "complete")
        assert (campaigns[root]["max_semantic_attempts"], campaigns[root]["retry_scope"]) == (20, "component")
    for arm in QWEN_FAMILY.arms:
        rebase = campaigns[arm.root]["rebase"]
        parent, commit, frozen_manifest, first_manifest, earlier_manifest = PARENTS[arm.condition_id]
        removed, added = _population_swap()
        assert len(removed) == len(added) == 95
        assert (rebase["parent_campaign_id"], rebase["parent_scenario_manifest_sha256"],
                rebase["parent_sampling_manifest_sha256"]) == (parent, EARLIER_MANIFEST, earlier_manifest)
        assert (rebase["removed_ids"], rebase["added_ids"]) == (removed, added)
        assert [(hop["parent"]["campaign_id"], hop["parent"]["scenario_manifest_sha256"],
                 hop["parent"]["sampling_manifest_sha256"], hop["target"]["scenario_manifest_sha256"],
                 hop["removed_ids"], hop["added_ids"]) for hop in rebase["chain"]] == [
            (parent, FROZEN_500_MANIFEST, frozen_manifest, FIRST_MANIFEST, [REMOVED_287], [ADDED_R]),
            (parent, FIRST_MANIFEST, first_manifest, EARLIER_MANIFEST, REMOVED_FIVE, ADDED_FIVE),
            (parent, EARLIER_MANIFEST, earlier_manifest, PAPER_MANIFEST, removed, added)]
        states = rebase["run_code_states"][arm.condition_id]
        assert [state["code_revision"] for state in states] == [commit, FIRST_REBASE_COMMIT, SECOND_REBASE_COMMIT, PAPER_REBASE]
        assert all(state["code_dirty"] is False and state["draws"] == [f"draw-{d:03d}" for d in range(5)]
                   for state in states)
    weights = provenance["base_weights"]
    assert weights["sha256"] == "668868d5ab4d43cc5c396592bdba527bfabb53b69bd4d5af6598edc7a427091b"
    assert weights["reference"]["checked"] == "directory_name"
    assert weights["members"]["pick"]["merge_manifest"] == QWEN_FAMILY.merge_manifests["pick"]
    assert weights["members"]["pick"]["merge_manifest_sha256"] == RECORDS["merge_manifest"]
    assert {arm: record["report_sha256"] for arm, record in installed["disclosure"]["parity_b"]["pick"].items()} == {
        arm: RECORDS[arm] for arm in ("prefill", "decode")}
    assert provenance["bootstrap"] == {"ci_level": 0.95, "n_resamples": 10_000, "seed": 16}
    assert provenance["randomization_test"] == {"n_permutations": 100_000, "seed": 16}
    assert provenance["judge"]["prompt_style"] == "forced-grpo" and provenance["strict_think"] is True
    assert (provenance["engine"]["tp_size"], provenance["temperature"], provenance["max_semantic_attempts"]) == (1, 1.0, 20)


@pytest.mark.frozen
def test_the_two_arms_on_the_paper_population_with_five_draws(installed):
    base, pick = installed["arms"]["base"], installed["arms"]["pick"]
    assert (base["label"], pick["label"]) == ("Qwen3-8B", "GRPO (checkpoint-300)")
    assert (base["model_path"], base["model_weights_sha256"]) == ("Qwen3-8B", None)
    assert (pick["model_path"], pick["model_weights_sha256"]) == (
        "checkpoint-300", "9cbc91ce67ad525417242fe07d328c998b2493fbf644c693b01b8acaac4017dc")
    # flat inconsistency [95% scenario-cluster bootstrap CI]: 2,080 and 158 of 2,500 judgments
    assert [base["flat"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx([0.8320, 0.8100, 0.8532])
    assert [pick["flat"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx([0.0632, 0.0508, 0.0764])
    assert [round(arm["flat"]["estimate"] * 2500) for arm in (base, pick)] == [2080, 158]
    # Cohen's kappa between the verdict and the per-rule max: both near chance, where an always-yes judge sits
    assert [base["kappa"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.00959, 0.00571, 0.01347], abs=5e-6)
    assert [pick["kappa"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.02035, -0.00604, 0.05305], abs=5e-6)
    # the full-set yes share and the always-yes floor
    assert (base["yes_rate"]["estimate"], pick["yes_rate"]["estimate"]) == pytest.approx((0.1540, 0.9896))
    assert (base["always_yes_floor"]["estimate"], pick["always_yes_floor"]["estimate"]) == pytest.approx(
        (0.0592, 0.0548))
    strata = {label: {stratum: (row["estimate"], row["n_draws"]) for stratum, row in arm["m_stratified"].items()}
              for label, arm in installed["arms"].items()}
    assert strata["base"]["M=yes"] == pytest.approx((0.83929, 2352), abs=5e-6)
    assert strata["base"]["M!=yes"] == pytest.approx((0.71622, 148), abs=5e-6)
    assert strata["base"]["M=no"] == pytest.approx((0.02326, 43), abs=5e-6)
    assert strata["pick"]["M=yes"] == pytest.approx((0.00931, 2363), abs=5e-6)
    assert strata["pick"]["M!=yes"] == pytest.approx((0.99270, 137), abs=5e-6)
    assert strata["pick"]["M=no"] == pytest.approx((0.96667, 30), abs=5e-6)
    assert (strata["base"]["M=unsure"], strata["pick"]["M=unsure"]) == ((1.0, 105), (1.0, 107))
    # panel c's error bars: the M = yes and M != yes intervals
    for label, stratum, bounds in (("base", "M=yes", [0.81627, 0.86172]), ("base", "M!=yes", [0.63226, 0.79870]),
                                   ("pick", "M=yes", [0.00507, 0.01482]), ("pick", "M!=yes", [0.97600, 1.0])):
        row = installed["arms"][label]["m_stratified"][stratum]
        assert [row["ci_low"], row["ci_high"]] == pytest.approx(bounds, abs=5e-6), (label, stratum)
    yes_outside = {label: round(arm["yes_rate"]["estimate"] * 2500
                                - (1 - arm["m_stratified"]["M=yes"]["estimate"]) * arm["m_stratified"]["M=yes"]["n_draws"])
                   for label, arm in installed["arms"].items()}
    assert yes_outside == {"base": 7, "pick": 133}
    assert (base["same_call"]["n_complete_maps"], pick["same_call"]["n_complete_maps"]) == (1702, 2498)
    assert (base["first_attempt_acceptance_share"], pick["first_attempt_acceptance_share"]) == (0.9964, 0.9984)


@pytest.mark.frozen
def test_one_test_per_holm_family_and_the_claim_the_rule_allows(installed):
    comparison = installed["comparisons"]["pick"]
    flat, kappa = comparison["flat"], comparison["kappa"]
    assert [flat[key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx([0.7688, 0.7444, 0.7928])
    assert flat["p_value"] == flat["p_holm"] == pytest.approx(FLOOR_P) and flat["significant_holm"] is True
    assert [kappa[key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.01076, -0.01610, 0.04423], abs=5e-6)
    assert kappa["p_value"] == kappa["p_holm"] == pytest.approx(0.53295, abs=5e-6) and kappa["significant_holm"] is False
    for row in (flat, kappa):
        assert row["family_size"] == 1 and (row["policy_a"], row["policy_b"]) == ("base", "pick")
    assert comparison["claim"] == "yes_shift"
    assert comparison["harm_flag"] == {"flagged": True, "stratum": "M!=yes", "threshold": 0.1}
    differences = comparison["stratified_differences"]["differences"]
    assert [differences["M!=yes"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.27648, 0.19065, 0.36301], abs=5e-6)
    assert differences["M=unsure"]["estimate"] == 0.0  # 1.0 in both arms
    assert (differences["M=no"]["estimate"], differences["M=yes"]["estimate"]) == pytest.approx(
        (0.943411, -0.82998), abs=5e-6)


@pytest.mark.frozen
def test_the_yes_shift_baseline_explains_almost_all_of_the_gain(installed):
    shift = installed["yes_shift"]["pick"]
    assert shift["applicable"] is True and shift["descriptive"] is True
    assert (shift["yes_rate_reference"], shift["yes_rate_member"]) == pytest.approx((0.1540, 0.9896))
    assert (shift["rate_reference"], shift["rate_member"]) == pytest.approx((0.8320, 0.0632))
    assert shift["p"] == pytest.approx(0.98771, abs=5e-6)
    assert [shift["shifted_rate"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.06870, 0.05779, 0.07986], abs=5e-6)
    assert shift["shifted_rate"]["p_held_fixed"] is True
    assert shift["share_of_gain_explained"] == pytest.approx(0.99285, abs=5e-6)


@pytest.mark.frozen
def test_panel_a_draws_the_two_bars_their_intervals_and_labels_and_no_marks(installed):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    from ruleforge.experiments.training.paper_families import drawn_names

    assert drawn_names(installed) == {"base": "Base Qwen", "pick": "Trained Qwen"}
    assert_panel_a_draws_the_flat_rates_alone(installed, ["0.832", "0.063"],
                                              [("Base Qwen", 10.0), ("Trained Qwen", 10.0)], drawn_names(installed))


@pytest.mark.frozen
def test_the_disclosure_names_the_rebases_the_arms_origins_and_the_parity(installed):
    lines = installed["disclosure"]["lines"]
    text = "\n".join(lines)
    assert installed["disclosure"]["regime"] == "merged_fp32"
    base_root, pick_root = (arm.root for arm in QWEN_FAMILY.arms)
    removed, added = _population_swap()
    for phrase in (
        "reference base (Qwen3-8B), members pick (GRPO (checkpoint-300))",
        f"{base_root} is a rebase of campaign fig5-grpo-v2-fp32-2026-09-26's root from its population "
        f"(scenario manifest {EARLIER_MANIFEST[:12]}...): scenario(s) "
        + ", ".join(f"{scenario[:12]}..." for scenario in removed) + " removed and "
        + ", ".join(f"{scenario[:12]}..." for scenario in added)
        + " added in their place, the added judged by a resume under the parent's campaign id",
        f"{pick_root} is a rebase of campaign fig5-qwen-mix-2026-10-11-checkpoint-300's root",
        "Qwen3-8B/figure5/base: commit 2594706495aa, sources bba06331f5c2 (draw-000, draw-001, draw-002, draw-003, "
        "draw-004), commit 55aa6a2d456e, sources 100d560137c2 (draw-000, draw-001, draw-002, draw-003, draw-004), "
        "commit 87a6ba4f6877, sources 18acb6cf186d (draw-000, draw-001, draw-002, draw-003, draw-004), "
        "commit 57bfa3dc8b77, sources 47d982a23576 (draw-000, draw-001, draw-002, draw-003, draw-004).",
        "Qwen3-8B/figure5/pick: commit 882e55c3e219, sources 3540d30ea192 (draw-000, draw-001, draw-002, draw-003, "
        "draw-004), commit 55aa6a2d456e, sources 100d560137c2 (draw-000, draw-001, draw-002, draw-003, draw-004), "
        "commit 87a6ba4f6877, sources 18acb6cf186d (draw-000, draw-001, draw-002, draw-003, draw-004), "
        "commit 57bfa3dc8b77, sources 47d982a23576 (draw-000, draw-001, draw-002, draw-003, draw-004).",
        f"one scenario manifest ({PAPER_MANIFEST[:12]}...) of 500 scenarios",
        f"The reference arm base comes from {base_root}; paired with it by (scenario, decoder draw index)",
        "over the members named here (pick): a family of 1",
        "The yes-shift baseline beside each member is descriptive",
        "pick: prefill rho = 1.0003 [0.9998, 1.0009], decode rho = 1.0005 [1.0003, 1.0008].",
        "it is not the bf16 Qwen3-8B of the ruleset-size experiment",
    ):
        assert phrase in text, phrase
    assert sum("is a rebase" in line for line in lines) == 2


@pytest.mark.frozen
def test_the_table_builds_the_appendix_family_from_its_roots():
    population = "pop-a22af097"
    step = next(step for row in table.TABLE for step in row.builds if step.golden and step.golden.path == QWEN_GOLDEN)
    assert step.words == ["python", "-m", PROGRAM, "--family", APPENDIX, "--population", population, "--out-dir",
                          f"{{out}}/{APPENDIX}"]
    assert QWEN_FAMILY.on_population(population).arguments() == QWEN_FAMILY.arguments()
    assert table.QWEN_RUNS == tuple(arm.root for arm in QWEN_FAMILY.arms)


@pytest.mark.frozen
def test_the_installed_golden_s_panels(installed, repo_root, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    module = importlib.import_module("ruleforge.experiments.training.build")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    for suffix in ("pdf", "png"):
        figures = [word for panel in "a" for word in (
            f"--figure-{panel}", str(tmp_path / f"{QWEN_FIGURES[panel].split('/')[-1]}.{suffix}"))]
        assert module.main(["--from_golden", str(repo_root / QWEN_GOLDEN), *figures]) == 0
    for panel in "a":
        for suffix in ("pdf", "png"):
            assert (tmp_path / f"{QWEN_FIGURES[panel].split('/')[-1]}.{suffix}").stat().st_size > 0


@pytest.mark.frozen
@pytest.mark.cluster
def test_the_installed_golden_rebuilds_byte_for_byte_from_the_paper_root(installed, repo_root, tmp_path, monkeypatch):
    import subprocess
    import sys

    from ruleforge.core.paths import resolve
    from tests.test_training_gemma_frozen import CORETYPE, THREADS, cpu_unlike_the_golden_s

    pytest.importorskip("matplotlib", reason="needs the viz group: the build draws the panels")
    if not REAL_ARTIFACT_ROOT:
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    # both resolve() here and the build's process (its env is built from os.environ below) read the paper's root
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", REAL_ARTIFACT_ROOT)
    if not all(resolve(f"{arm.root}/aggregate/estimates.json").is_file() for arm in QWEN_FAMILY.arms):
        pytest.skip("the appendix's roots are not in this artifact root")
    if (reason := cpu_unlike_the_golden_s()) is not None:
        pytest.skip(reason)
    env = {**{key: value for key, value in os.environ.items() if key != CORETYPE}, **THREADS, "SOURCE_DATE_EPOCH": "0"}
    built = subprocess.run([sys.executable, "-m", PROGRAM, *QWEN_FAMILY.arguments(), "--out-dir", str(tmp_path)],
                           cwd=repo_root, env=env, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    assert (tmp_path / "training_family.json").read_bytes() == (repo_root / QWEN_GOLDEN).read_bytes()
