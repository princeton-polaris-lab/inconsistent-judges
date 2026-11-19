from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.experiments import table
from ruleforge.core.paths import resolve
from ruleforge.experiments.training import judgment_changes as jc
from ruleforge.experiments.training.paper_families import PAPER_OUTPUTS

pytestmark = pytest.mark.frozen

GOLDEN = "results/training/judgment_changes.json"
PROGRAM = "ruleforge.experiments.training.judgment_changes"
POPULATION = "pop-a22af097"
#: Each family's panel files, without their suffix.
PANELS = {"gemma": {"changes": "figures/grpo_judgment_changes", "counts": "figures/grpo_judgment_counts"},
          "qwen": {"changes": "figures/grpo_qwen_judgment_changes",
                                    "counts": "figures/grpo_qwen_judgment_counts"}}
#: The numbers, by family: each kind's (pairs, changed) and the trained arm's (yes, unsure, no) at draw 0.
CHANGES = {"gemma": {"per_rule": (28_500, 1_057), "aggregate": (500, 36), "full_set": (500, 182)},
           "qwen": {"per_rule": (28_500, 2_932), "aggregate": (500, 38), "full_set": (500, 408)}}
COUNTS = {"gemma": {"per_rule": (2_462, 5, 26_033), "aggregate": (468, 0, 32), "full_set": (292, 0, 208)},
          "qwen": {"per_rule": (1_689, 1_180, 25_631), "aggregate": (469, 27, 4), "full_set": (495, 0, 5)}}
#: Each arm's changed judgments between two of its draws: over the ten pairs, the fewest and the most in one pair.
DRAW_TO_DRAW = {
    "gemma": {"base": {"per_rule": (10_307, 1_000, 1_069), "aggregate": (322, 24, 39), "full_set": (264, 19, 36)},
              "final": {"per_rule": (10_148, 971, 1_068), "aggregate": (356, 28, 42), "full_set": (670, 52, 75)}},
    "qwen": {"base": {"per_rule": (28_403, 2_788, 2_882), "aggregate": (504, 42, 60), "full_set": (964, 86, 107)},
             "pick": {"per_rule": (29_569, 2_899, 3_011), "aggregate": (437, 37, 55), "full_set": (92, 8, 10)}},
}
#: The panels' printed values, by family and panel.
PRINTED = {"gemma": {"changes": ["3.7%", "7.2%", "36.4%"], "counts": ["2,462", "292", "5", "0", "26,033", "208"]},
           "qwen": {"changes": ["10.3%", "7.6%", "81.6%"], "counts": ["1,689", "495", "1,180", "0", "25,631", "5"]}}
#: Each panel's tick labels: the changes panel all three kinds, the counts panel two.
TICKS = {"changes": ["Individual-rule judgments\n(28,500 judgments)", "Aggregated rule judgments\n(500 judgments)",
                     "Full-ruleset judgments\n(500 judgments)"],
         "counts": ["Individual-rule judgments\n(n = 28,500)", "Full-ruleset judgments\n(n = 500)"]}
DRAWN = {"gemma": ("Base Gemma", "Trained Gemma"), "qwen": ("Base Qwen", "Trained Qwen")}


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


def _training_golden(repo_root, name):
    return json.loads((repo_root / PAPER_OUTPUTS[name].golden).read_text(encoding="utf-8"))


def test_the_golden_holds_both_paper_families_at_draw_0_on_the_paper_population(payload):
    assert (payload["source"], payload["schema_version"], payload["draw"], payload["population"]) == (
        "judgment_changes", 1, 0, POPULATION)
    assert payload["definition"] == jc.DEFINITION and list(payload["families"]) == list(PAPER_OUTPUTS)


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_family_s_changes_and_counts(payload, name):
    family = payload["families"][name]
    for kind, (n, changed) in CHANGES[name].items():
        assert {key: family["changes"][kind][key] for key in ("n", "changed")} == {"n": n, "changed": changed}
        assert family["changes"][kind]["change_rate"] == changed / n
        matrix = family["transitions"][jc.TRANSITIONS[kind]]
        assert n - sum(matrix[label][label] for label in matrix) == changed
    for kind, (yes, unsure, no) in COUNTS[name].items():
        assert family["member_counts"][kind] == {"yes": yes, "unsure": unsure, "no": no, "n": yes + unsure + no}
        assert {label: sum(row[label] for row in family["transitions"][jc.TRANSITIONS[kind]].values())
                for label in ("yes", "unsure", "no")} == {"yes": yes, "unsure": unsure, "no": no}
    # 500 scenarios, 57 rules, every pair valid
    assert family["transitions"]["n_common"] == family["transitions"]["n_base"] == family["transitions"]["n_tuned"] == 500


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_arm_s_draw_to_draw_changes_beside_the_base_to_trained_ones(payload, name):
    family = payload["families"][name]
    assert list(family["draw_to_draw"]) == [family["reference"], family["member"]]
    for arm, kinds in DRAW_TO_DRAW[name].items():
        within = family["draw_to_draw"][arm]
        assert within["n_pairs"] == 10 and [pair["draws"] for pair in within["pairs"]] == [
            [first, second] for first in range(5) for second in range(first + 1, 5)]
        for kind, (changed, fewest, most) in kinds.items():
            n = CHANGES[name][kind][0]
            assert [pair[kind]["n"] for pair in within["pairs"]] == [n] * 10
            counts = [pair[kind]["changed"] for pair in within["pairs"]]
            assert (sum(counts), min(counts), max(counts)) == (changed, fewest, most)
            assert within[kind] == {"n": 10 * n, "changed": changed, "change_rate": changed / (10 * n),
                                    "min_change_rate": fewest / n, "max_change_rate": most / n}


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_family_is_its_training_golden_s_arms(repo_root, payload, name):
    family, golden = payload["families"][name], _training_golden(repo_root, name)
    switch = PAPER_OUTPUTS[name].family.on_population(POPULATION)
    assert [family["reference"], family["member"]] == [arm.label for arm in switch.arms] == golden["policies"]
    assert family["labels"] == {label: golden["arms"][label]["label"] for label in golden["policies"]}
    for arm in switch.arms:
        ours, theirs = family["provenance"]["arms"][arm.label], golden["provenance"]["arms"][arm.label]
        assert (ours["root"], ours["condition_id"]) == (arm.root, arm.condition_id)
        assert {key: ours[key] for key in ("root", "campaign_id", "condition_id", "condition_fingerprint")} == {
            key: theirs[key] for key in ("root", "campaign_id", "condition_id", "condition_fingerprint")}
        assert ours["prompt_style"] == "forced-grpo"
    assert [(c["root"], c["estimates_sha256"]) for c in family["provenance"]["campaigns"]] == [
        (c["root"], c["estimates_sha256"]) for c in golden["provenance"]["campaigns"]]
    assert family["provenance"]["scenario_manifest_sha256"] == golden["provenance"]["scenario_manifest_sha256"]


def test_the_table_builds_it_from_both_families_roots():
    step = next(step for row in table.TABLE for step in row.builds if step.golden and step.golden.path == GOLDEN)
    assert step.words[2:] == [PROGRAM, "--family", "gemma", "--family", "qwen", "--population",
                              POPULATION, "--golden", f"{{out}}/{GOLDEN}"]
    assert list(step.inputs) == [arm.root for name in PAPER_OUTPUTS for arm in PAPER_OUTPUTS[name].family.arms]


def test_the_panels_draw_the_golden_s_numbers_under_the_training_arm_names(payload):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt

    from ruleforge.experiments.training.family_arms import arm_names
    from ruleforge.experiments.training.figure import _tick_labels_overlap
    from ruleforge.experiments.training.judgment_changes_figure import build_figure

    for name in PAPER_OUTPUTS:
        assert arm_names(payload, name) == DRAWN[name]
        for panel in ("changes", "counts"):
            figure = build_figure(payload, name, panel)
            texts = [text.get_text() for text in figure.axes[0].texts]
            ticks = [text.get_text() for text in figure.axes[0].get_xticklabels()]
            overlap = _tick_labels_overlap(figure, figure.axes[0])
            legend = figure.axes[0].get_legend()
            shown = [text.get_text() for text in legend.get_texts()] + [legend.get_title().get_text()]
            plt.close(figure)
            assert texts == PRINTED[name][panel], (name, panel)
            assert ticks == TICKS[panel] and not overlap, (name, panel)
            assert ([f"{DRAWN[name][0]} → {DRAWN[name][1]}"] if panel == "changes" else [DRAWN[name][1]]) == [
                text for text in shown if "Gemma" in text or "Qwen" in text]
            assert not [text for text in texts + shown if "checkpoint" in text.lower()]


def test_the_panels_render_from_the_golden(repo_root, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    for name, panels in PANELS.items():
        for suffix in ("pdf", "png"):
            out = {panel: tmp_path / f"{Path(path).name}.{suffix}" for panel, path in panels.items()}
            assert jc.main(["--from_golden", str(repo_root / GOLDEN), "--paper", name, "--figure-changes",
                            str(out["changes"]), "--figure-counts", str(out["counts"])]) == 0
            assert all(path.stat().st_size > 0 for path in out.values())


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_paper_root(repo_root, tmp_path):
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("no artifact root is set")
    roots = [arm.root for name in PAPER_OUTPUTS for arm in PAPER_OUTPUTS[name].family.arms]
    if not all((resolve(root) / "aggregate" / "estimates.json").is_file() for root in roots):
        pytest.skip("the training roots are not in this artifact root")
    out = tmp_path / "judgment_changes.json"
    built = subprocess.run([sys.executable, "-m", PROGRAM, "--family", "gemma", "--family", "qwen",
                            "--population", POPULATION, "--golden", str(out)], cwd=repo_root,
                           env={**os.environ, "SOURCE_DATE_EPOCH": "0"}, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    assert out.read_bytes() == (repo_root / GOLDEN).read_bytes()
