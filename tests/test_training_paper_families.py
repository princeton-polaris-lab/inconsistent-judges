from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.experiments.training import paper_families
from ruleforge.experiments.training.family_readout import FamilyArm
from ruleforge.experiments.training.paper_families import (
    GEMMA_FAMILY,
    GEMMA_FIGURES,
    GEMMA_GOLDEN,
    PaperFamily,
    drawn_names,
    paper_refusal,
)
from tests.support.training_campaign import _roots  # noqa: F401 - an autouse fixture
from tests.test_training_family_readout import LABELS, PREFIX, analysed, campaign, read

REPO_ROOT = Path(__file__).resolve().parents[1]


def copy_into_root(source_root: Path, dest_root: Path, paths: list[str]) -> dict[str, Path]:
    """Copy each of ``paths`` (a directory or a file) from ``source_root`` to the same place under ``dest_root``."""
    copies = {}
    for relative in paths:
        source, dest = source_root / relative, dest_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        copies[relative] = shutil.copytree(source, dest) if source.is_dir() else shutil.copy2(source, dest)
    return copies
#: Where the synthetic member's merge manifest sits under the artifact root (``run_training_campaign``'s merges).
MERGE = "B/grpo/merged/final/merge_manifest.json"
PANEL_FILES = [f"training_a.{suffix}" for suffix in ("pdf", "png")]
DPI = "40"


def family(tmp_path: Path, **changes) -> PaperFamily:
    """The synthetic stand-in for :data:`GEMMA_FAMILY`: base (root A) against final (root B)."""
    return dataclasses.replace(PaperFamily(
        arms=(FamilyArm("base", "artifact://A/root", f"{PREFIX}/base"),
              FamilyArm("final", "artifact://B/root", f"{PREFIX}/final")),
        labels={"base": LABELS["base"], "final": LABELS["final"]},
        rule_set=str(tmp_path / "A" / "rules.jsonl"),
        merge_manifests={"final": f"artifact://{MERGE}"},
    ), **changes)


@pytest.fixture
def two_roots(tmp_path, modelspec):
    """Root A (base and grpo) and root B (final alone), analysed."""
    a = analysed(campaign(tmp_path / "A", modelspec, ("base", "grpo")))
    b = campaign(tmp_path / "B", modelspec, ("final",), family="base+final")
    return {"base": a, "grpo": a, "final": analysed(b, allow_incomplete_conditions=True)}


@pytest.fixture
def driver(monkeypatch, tmp_path):
    """The script as a module, over a scratch checkout, the synthetic family and reproducibly dated PDFs."""
    module = importlib.import_module("ruleforge.experiments.training.build")
    monkeypatch.setattr(module, "repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paper_families, "GEMMA_FAMILY", family(tmp_path))
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    return module


def run(driver, argv, capsys) -> tuple[int, str, str]:
    try:
        code = driver.main([str(arg) for arg in argv])
    except SystemExit as exit_info:
        code = exit_info.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def paper_panels(tmp_path: Path) -> list[Path]:
    return [tmp_path / "repo" / f"{GEMMA_FIGURES[panel]}.{suffix}" for panel in "a" for suffix in ("pdf", "png")]


def test_the_switch_is_gemma_s_base_against_checkpoint_600_alone_from_the_paper_s_root():
    assert [(arm.label, arm.root, arm.condition_id) for arm in GEMMA_FAMILY.arms] == [
        ("base", "artifact://sampling/fig5-gemma-2026-10-07-pop-a22af097", "gemma-4-31B-it/figure5/base"),
        ("final", "artifact://sampling/fig5-gemma-ckpt600-2026-10-09-pop-a22af097", "gemma-4-31B-it/figure5/final"),
    ]
    assert GEMMA_FAMILY.reference == "base"
    assert GEMMA_FAMILY.labels == {"base": "Gemma 4 31B-it", "final": "GRPO (checkpoint-600)"}
    assert (REPO_ROOT / GEMMA_FAMILY.rule_set).is_file()
    assert GEMMA_FAMILY.merge_manifests == {"final": "artifact://selection/gemma-grpo-mix-2026-10-04/modal_root/"
                                             "finetuning/gemma-grpo-mix-2026-10-04/merged/checkpoint-600/"
                                             "merge_manifest.json"}
    assert GEMMA_GOLDEN == "results/training/gemma.json"
    assert GEMMA_FIGURES == {"a": "figures/grpo_inconsistency"}


#: Each paper family's arms as its panels draw them: base, then trained.
DRAWN = {"gemma": ("Base Gemma", "Trained Gemma"), "qwen": ("Base Qwen", "Trained Qwen")}


@pytest.mark.parametrize("name", sorted(DRAWN))
def test_each_paper_family_s_panels_name_its_arms_by_model_with_no_checkpoint_number(name):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    from ruleforge.experiments.training.figure import build_figure

    outputs = paper_families.PAPER_OUTPUTS[name]
    recorded = [outputs.family.labels[arm.label] for arm in outputs.family.arms]
    assert list(outputs.drawn) == [arm.label for arm in outputs.family.arms] and "checkpoint" in recorded[1]
    assert tuple(outputs.drawn.values()) == DRAWN[name]
    golden = json.loads((REPO_ROOT / outputs.golden).read_text(encoding="utf-8"))
    assert [golden["arms"][policy]["label"] for policy in golden["policies"]] == recorded
    assert drawn_names(golden) == outputs.drawn
    for panel in "a":
        figure = build_figure(golden, panel, drawn_names(golden))
        figure.canvas.draw()
        texts = [text.get_text() for text in figure.findobj(Text) if text.get_text()]
        plt.close(figure)
        assert set(DRAWN[name]) <= set(texts) and not set(recorded) & set(texts), (panel, texts)
        assert not [text for text in texts if "checkpoint" in text.lower()], (panel, texts)


def test_the_paper_s_names_are_drawn_for_a_paper_family_s_readout_alone(tmp_path, two_roots):
    readout = read(tmp_path, two_roots, arms=list(family(tmp_path).arms), merge_manifests={"final": f"artifact://{MERGE}"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(paper_families, "GEMMA_FAMILY", family(tmp_path))
        assert drawn_names(readout) == {"base": "Base Gemma", "final": "Trained Gemma"}
        relabelled = copy.deepcopy(readout)
        relabelled["arms"]["final"]["label"] = "GRPO (checkpoint-300)"
        three = read(tmp_path, two_roots)
        assert drawn_names(relabelled) == drawn_names(three) == {}
    assert drawn_names(readout) == {}


@pytest.mark.parametrize("name", sorted(DRAWN))
def test_every_value_a_paper_family_s_panel_prints_is_bold_as_the_ruleforge_panel_s(name):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from ruleforge.experiments import style
    from ruleforge.experiments.training.figure import build_figure
    from ruleforge.experiments.ruleforge import plot

    def styles(figure):
        return [(text.get_fontweight(), text.get_fontsize(), to_hex(text.get_color()).upper())
                for axes in figure.axes for text in axes.texts]

    expected = ("bold", style.VALUE_LABEL["fontsize"], style.VALUE_LABEL["color"])
    drawn = plot.build_figure(json.loads((REPO_ROOT / "results/ruleforge/ruleforge.json").read_text()), "a")
    assert [text.get_text() for text in drawn.axes[0].texts] == ["0.727", "0.589", "0.552"]
    assert set(styles(drawn)) == {expected}
    plt.close(drawn)
    golden = json.loads((REPO_ROOT / paper_families.PAPER_OUTPUTS[name].golden).read_text(encoding="utf-8"))
    for panel in "a":
        figure = build_figure(golden, panel)
        assert len(styles(figure)) in (2, 4) and set(styles(figure)) == {expected}, panel
        plt.close(figure)


def test_the_arguments_are_the_family_flag_by_flag(tmp_path):
    assert family(tmp_path).arguments() == [
        "--arm", "base", "artifact://A/root", f"{PREFIX}/base", "--arm", "final", "artifact://B/root", f"{PREFIX}/final",
        "--reference", "base", "--rule-set", str(tmp_path / "A" / "rules.jsonl"),
        "--label", f"base={LABELS['base']}", "--label", f"final={LABELS['final']}",
        "--merge-manifest", f"final=artifact://{MERGE}",
    ]


def test_a_family_on_another_population_names_its_roots_as_a_rebase_does():
    on_earlier = GEMMA_FAMILY.on_population("pop-6f528dcb")
    assert [arm.root for arm in on_earlier.arms] == ["artifact://sampling/fig5-gemma-2026-10-07-pop-6f528dcb",
                                                "artifact://sampling/fig5-gemma-ckpt600-2026-10-09-pop-6f528dcb"]
    assert dataclasses.replace(on_earlier, arms=GEMMA_FAMILY.arms) == GEMMA_FAMILY
    appendix = paper_families.QWEN_FAMILY
    assert [arm.root for arm in appendix.on_population("pop-6f528dcb").arms] == [
        "artifact://sampling/fig5-grpo-v2-fp32-2026-09-26-checkpoint-200-pop-6f528dcb",
        "artifact://sampling/fig5-qwen-mix-2026-10-11-checkpoint-300-pop-6f528dcb"]
    own = {re.search(r"-(pop-[0-9a-f]{8})$", arm.root).group(1) for arm in appendix.arms}
    assert len(own) == 1 and appendix.on_population(own.pop()) == appendix
    for population in ("6f528dcb", "pop-6F528DCB", "pop-6f528dc", "pop-6f528dcb-x"):
        with pytest.raises(ValueError, match="is not pop-<first 8 hex of a scenario manifest>"):
            GEMMA_FAMILY.on_population(population)


def _set(path: str, value):
    """A change to one key of a readout, its path dotted (value ``DROP``: the key removed)."""
    def change(readout):
        *parents, last = path.split(".")
        node = readout
        for key in parents:
            node = node[key]
        if value is DROP:
            del node[last]
        else:
            node[last] = value
    return change


DROP = object()
HEAD = "not the paper's Gemma family (ruleforge.experiments.training.paper_families.GEMMA_FAMILY): "
#: How a readout naming other arms anywhere is refused, before what it names is listed.
NOT_THE_ARMS = (HEAD + "its arms are not the paper's ['base', 'final'] (reference base), which it shows alone and in "
                "that order, wherever it names them: ")


def _third_arm_drawn(readout):
    readout["arms"]["grpo"] = {**readout["arms"]["final"], "label": "GRPO (checkpoint-200)"}
    readout["policies"].append("grpo")


def test_only_a_readout_of_exactly_the_paper_s_family_is_the_paper_s(tmp_path, two_roots):
    arms = family(tmp_path).arms
    readout = read(tmp_path, two_roots, arms=list(arms), merge_manifests={"final": f"artifact://{MERGE}"})
    assert paper_refusal(readout, family(tmp_path)) is None
    # The switch itself, looked up when it is called (so a test can stand in for it): the same two labels, not its roots.
    assert (f"arm base is {PREFIX}/base of artifact://A/root, not {PREFIX}/base of "
            "artifact://sampling/fig5-gemma-2026-10-07") in paper_refusal(readout)
    head = HEAD
    cases = [
        (_set("source", "stochastic_campaign"), "it is no training family readout (source 'stochastic_campaign'"),
        (_set("schema_version", 2), "schema_version 2;"),
        (_set("provenance.arms.final.root", "artifact://C/root"),
         f"arm final is {PREFIX}/final of artifact://C/root, not {PREFIX}/final of artifact://B/root"),
        (_set("provenance.arms.base.condition_id", f"{PREFIX}/grpo"), f"arm base is {PREFIX}/grpo of artifact://A/root"),
        (_set("arms.final.label", "GRPO"), "arm final is labelled 'GRPO', not 'GRPO (checkpoint-600)'"),
        (_set("provenance.base_weights.members.final.merge_manifest", "artifact://elsewhere/merge_manifest.json"),
         "final's merge manifest is not the paper's"),
        (_set("provenance.complete", False), "it is not complete"),
        (_set("provenance.rule_set", {"name": "rules.jsonl", "sha256": "0" * 64}), "the same-call rate was reread against"),
        (_set("provenance.allow_code_mismatch", True), "it was read with allow_code_mismatch"),
        (_set("provenance.engine.stack", "/opt/vllm"), "it holds absolute paths (at ['provenance.engine.stack'])"),
        (_set("policies", ["final"]), NOT_THE_ARMS + "policies is ['final']"),
        (_set("policies", ["final", "base"]), NOT_THE_ARMS + "policies is ['final', 'base']"),
        (_third_arm_drawn, NOT_THE_ARMS + "policies is ['base', 'final', 'grpo']; arms is ['base', 'final', 'grpo']"),
        (_set("reference_policy", "final"), NOT_THE_ARMS + "reference_policy is 'final'"),
        (_set("comparisons.grpo", {}), NOT_THE_ARMS + "comparisons is ['final', 'grpo']"),
        (_set("yes_shift.final", DROP), NOT_THE_ARMS + "yes_shift is []"),
        (_set("provenance.family.labels", ["base", "grpo"]), NOT_THE_ARMS + "provenance.family.labels is ['base', 'grpo']"),
        (_set("provenance.family.reference", "final"), NOT_THE_ARMS + "provenance.family.reference is 'final'"),
        (_set("provenance.family.members", []), NOT_THE_ARMS + "provenance.family.members is []"),
        (_set("provenance.family.holm_family_size", 2), NOT_THE_ARMS + "provenance.family.holm_family_size is 2"),
        (_set("provenance.arms.grpo", {}), NOT_THE_ARMS + "provenance.arms is ['base', 'final', 'grpo']"),
        (_set("provenance.base_weights.members", None), NOT_THE_ARMS + "provenance.base_weights.members is None"),
        (_set("provenance", DROP), NOT_THE_ARMS + "provenance.family.labels is None; provenance.family.reference is "
                                                  "None; provenance.family.members is None"),
        (_set("provenance.arms.final", "final"), f"arm final is None of None, not {PREFIX}/final of artifact://B/root"),
        (_set("arms.final.condition_id", f"{PREFIX}/grpo"),
         f"arms.final.condition_id is '{PREFIX}/grpo', not '{PREFIX}/final'"),
        (_set("arms.base.reference", False), "arms.base.reference is False, not True"),
        (_set("arms.final.model_path", "checkpoint-200"), "arms.final.model_path is 'checkpoint-200', not "),
        (_set("arms.final.model_weights_sha256", "0" * 64), "arms.final.model_weights_sha256 is '0000"),
        (_set("comparisons.final.flat.policy_a", "grpo"),
         f"final's flat contrast is not base against final ({PREFIX}/base against {PREFIX}/final) in a Holm family "
         "of 1"),
        (_set("comparisons.final.kappa.family_size", 2), "final's kappa contrast is not base against final"),
        (_set("comparisons.final.kappa", DROP), "final's kappa contrast is not base against final"),
    ]
    for change, message in cases:
        changed = copy.deepcopy(readout)
        change(changed)
        refusal = paper_refusal(changed, family(tmp_path))
        assert refusal is not None and refusal.startswith(head) and message in refusal, (message, refusal)
    assert "arm base is labelled 'Gemma 4 31B-it', not 'base'" in paper_refusal(
        readout, family(tmp_path, labels={"final": LABELS["final"]}))
    three = read(tmp_path, two_roots)
    assert paper_refusal(three, family(tmp_path)) == NOT_THE_ARMS + (
        "policies is ['base', 'grpo', 'final']; arms is ['base', 'final', 'grpo']; comparisons is ['final', 'grpo']; "
        "yes_shift is ['final', 'grpo']; provenance.family.labels is ['base', 'grpo', 'final']; "
        "provenance.family.members is ['grpo', 'final']; provenance.family.holm_family_size is 2; provenance.arms is "
        "['base', 'final', 'grpo']; provenance.base_weights.members is ['final', 'grpo']")
    grpo = family(tmp_path, arms=(arms[0], FamilyArm("grpo", "artifact://A/root", f"{PREFIX}/grpo")))
    assert paper_refusal(readout, grpo).startswith(
        HEAD + "its arms are not the paper's ['base', 'grpo'] (reference base), which it shows alone and in that "
        "order, wherever it names them: policies is ['base', 'final']; arms is ['base', 'final']")


def test_paper_writes_the_paper_s_panels_and_the_readout_to_scratch(tmp_path, two_roots, driver, capsys):
    out = tmp_path / "out"
    argv = [*family(tmp_path).arguments(), "--paper", "--out-dir", out, "--dpi", DPI]
    code, printed, _ = run(driver, argv, capsys)
    assert code == 0 and printed == (out / "training_family.md").read_text()
    assert sorted(path.name for path in out.iterdir()) == ["training_family.json", "training_family.md"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paper_panels(tmp_path))
    assert paper_refusal(json.loads((out / "training_family.json").read_text())) is None
    code, _, err = run(driver, [*argv[:-5], "--paper", "--out-dir", tmp_path / "again", "--dpi", DPI], capsys)
    assert code == 2 and "grpo_inconsistency.pdf" in err and "pass --force" in err
    assert run(driver, [*argv, "--force"], capsys)[0] == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--arm", "grpo", "artifact://A/root", f"{PREFIX}/grpo"],
         "--paper: this readout is " + NOT_THE_ARMS + "policies is ['base', 'final', 'grpo']"),
        (["DROP-LABELS", "--label", "final=GRPO"], "arm final is labelled 'GRPO'"),
    ],
)
def test_paper_refuses_any_other_family_and_writes_nothing(tmp_path, two_roots, driver, capsys, extra, message):
    argv = family(tmp_path).arguments()
    if "DROP-LABELS" in extra:
        argv = argv[:argv.index("--label")] + argv[argv.index("--label") + 4:]
    extra = [word for word in extra if not word.startswith("DROP-")]
    code, _, err = run(driver, [*argv, *extra, "--paper", "--out-dir", tmp_path / "out"], capsys)
    assert code == 2 and message in err
    assert not (tmp_path / "out").exists() and not (tmp_path / "repo").exists()


def test_a_build_writes_its_readout_and_panels_under_the_checkout_s_figures(tmp_path, two_roots, driver, capsys):
    out = tmp_path / "repo" / "figures" / "family"
    code, printed, err = run(driver, [*family(tmp_path).arguments(), "--out-dir", out, "--dpi", DPI], capsys)
    assert code == 0, err
    assert printed == (out / "training_family.md").read_text()
    assert sorted(path.name for path in out.iterdir()) == sorted(["training_family.json", "training_family.md",
                                                                  *PANEL_FILES])


def test_paper_still_keeps_the_readout_out_of_the_paper_s_files(tmp_path, two_roots, driver, capsys):
    out = tmp_path / "repo" / "results" / "gemma"
    code, _, err = run(driver, [*family(tmp_path).arguments(), "--paper", "--out-dir", out], capsys)
    assert code == 2 and "training_family.json" in err and "would be among the paper's files" in err


def test_from_golden_re_renders_the_paper_s_panels_and_nothing_into_results(tmp_path, two_roots, driver, capsys):
    built = tmp_path / "built"
    assert run(driver, [*family(tmp_path).arguments(), "--out-dir", built, "--dpi", DPI], capsys)[0] == 0
    golden = built / "training_family.json"
    figures = [flag for panel, path in zip("a", paper_panels(tmp_path)[::2])
               for flag in (f"--figure-{panel}", path)]
    code, printed, _ = run(driver, ["--from_golden", golden, *figures, "--dpi", DPI], capsys)
    assert code == 0 and printed == (built / "training_family.md").read_text()
    for name, path in zip(PANEL_FILES[::2], paper_panels(tmp_path)[::2]):
        assert path.read_bytes() == (built / name).read_bytes(), name
    path = tmp_path / "repo" / "results" / "training_a.png"
    code, _, err = run(driver, ["--from_golden", golden, "--figure-a", path, "--dpi", DPI], capsys)
    assert code == 2 and f"--figure-a {path} would be among the paper's files (results)" in err and not path.exists()
    three = tmp_path / "three"
    argv = [*family(tmp_path).arguments(), "--arm", "grpo", "artifact://A/root", f"{PREFIX}/grpo", "--out-dir", three]
    assert run(driver, [*argv, "--dpi", DPI], capsys)[0] == 0
    path = tmp_path / "repo" / "figures" / "a.png"
    code, _, _ = run(driver, ["--from_golden", three / "training_family.json", "--figure-a", path, "--dpi", DPI],
                     capsys)
    assert code == 0 and path.stat().st_size > 0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--from_golden", "GOLDEN", "--figure-a", "a.pdf", "--paper", "--arm", "x", "y", "z"],
         "--from_golden re-renders an existing golden; do not also pass --arm, --paper"),
        (["--from_golden", "GOLDEN"], "pass it"),
        (["--from_golden", "ONE_ROOT", "--figure-a", "a.pdf"],
         "is no training family readout (source 'stochastic_campaign', schema_version 1)"),
        (["--arm", "base", "artifact://A/root", "x", "--figure-a", "a.pdf"],
         "--figure-a name a re-render's panel (--from_golden); a build writes its own"),
        (["--arm", "base", "artifact://A/root", "x", "--reference", "base"],
         "a build takes --arm (one per arm), --reference, --rule-set and --out-dir; or pass --from_golden"),
    ],
)
def test_a_re_render_and_a_build_take_their_own_flags(tmp_path, driver, capsys, argv, message):
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"source": "training_family", "schema_version": 1}))
    one_root = REPO_ROOT / "tests" / "fixtures" / "training" / "synthetic_readout.json"
    argv = [{"GOLDEN": golden, "ONE_ROOT": one_root}.get(arg, arg) for arg in argv]
    code, _, err = run(driver, argv, capsys)
    assert code == 2 and message in err


def test_the_golden_and_panels_reproduce_byte_for_byte_from_the_roots_copied_into_another_root(
        tmp_path, tmp_path_factory, two_roots, driver, capsys, monkeypatch):
    original = tmp_path / "original"
    argv = [*family(tmp_path).arguments(), "--dpi", DPI]
    assert run(driver, [*argv, "--out-dir", original], capsys)[0] == 0
    paper_root = tmp_path_factory.mktemp("paper_root")  # another root: outside the first, which is tmp_path
    records = copy_into_root(tmp_path, paper_root, ["A/root", "B/root", MERGE])
    assert set(records) == {"A/root", "B/root", MERGE}
    shutil.move(str(tmp_path / "B" / "grpo"), str(tmp_path / "weights_moved_away"))
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(paper_root))
    without = [word for word in argv if not word.startswith("final=artifact://")]
    without.remove("--merge-manifest")
    code, _, err = run(driver, [*without, "--out-dir", tmp_path / "unnamed"], capsys)
    assert code == 2 and "merge_manifest.json is not readable here" in err
    copied = tmp_path / "copied"
    assert run(driver, [*argv, "--paper", "--out-dir", copied], capsys)[0] == 0
    for name in ("training_family.json", "training_family.md"):
        assert (copied / name).read_bytes() == (original / name).read_bytes(), name
    for name, path in zip(PANEL_FILES, paper_panels(tmp_path)):
        assert path.read_bytes() == (original / name).read_bytes(), name
    golden = tmp_path / "repo" / GEMMA_GOLDEN
    golden.parent.mkdir(parents=True)
    shutil.copyfile(copied / "training_family.json", golden)
    rerendered = tmp_path / "rerendered"
    for suffix in ("pdf", "png"):
        figures = [word for panel in "a" for word in (f"--figure-{panel}", rerendered / f"training_{panel}.{suffix}")]
        assert run(driver, ["--from_golden", golden, *figures, "--dpi", DPI], capsys)[0] == 0
    for name in PANEL_FILES:
        assert (rerendered / name).read_bytes() == (original / name).read_bytes(), name
