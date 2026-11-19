from __future__ import annotations

import importlib.util
import json
import re

import pytest

from ruleforge.evaluate import api
from ruleforge.experiments import table
from ruleforge.experiments.training.paper_families import GEMMA_GOLDEN, PAPER_OUTPUTS, QWEN_GOLDEN
from tests.conftest import REPO_ROOT
from tests.identity import published

#: The runs the paper's goldens read; a published root missing here is left writable, with no error.
PUBLISHED_RUNS = {f"artifact://sampling/{name}" for name in published.ROOTS}
STEPS = [step for row in table.TABLE for step in (*row.builds, *row.plots)]


def _row(name: str) -> table.Experiment:
    return next(row for row in table.TABLE if row.name == name)


def _config_roots(path: str) -> set[str]:
    text = (REPO_ROOT / path).read_text(encoding="utf-8").replace("${pop}", "pop-a22af097")
    return set(re.findall(r"artifact://sampling/[^\s,}]+", text))


def test_the_pinned_roots_are_the_published_runs():
    assert set(table.published_runs()) == PUBLISHED_RUNS
    assert set(api.pinned_roots()) == PUBLISHED_RUNS


@pytest.mark.parametrize("place", ["", "/inner"])
def test_a_published_run_is_refused_as_a_write_at_or_below_it(tmp_path, monkeypatch, place):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    run = sorted(PUBLISHED_RUNS)[0].removeprefix("artifact://")
    with pytest.raises(api.CampaignError, match="a published run of ruleset-size: it is never written"):
        api.refuse_protected(tmp_path / f"{run}{place}", "the campaign root")
    api.refuse_protected(tmp_path / "sampling" / "mine", "the campaign root")


def test_every_build_reads_only_protected_runs():
    named = {word for step in STEPS for word in (*step.words, *step.inputs) if word.startswith("artifact://sampling/")}
    roots = {"/".join(word.split("/")[:4]) for word in named}
    assert roots and roots <= PUBLISHED_RUNS


@pytest.mark.parametrize(("experiment", "config"), [("prompt-ablation", "configs/paper/prompt_ablation_arms.yaml"),
                                                    ("hierarchy", "configs/paper/hierarchy_pairs.yaml")])
def test_the_runs_of_a_build_that_reads_a_map_are_the_map_s(experiment, config):
    assert set(_row(experiment).runs) == _config_roots(config)
    # The rebuild checks a step's inputs before any step runs; the map's roots are not on its command line.
    for step in _row(experiment).builds:
        assert set(step.inputs) == _config_roots(config), step.id


def test_a_training_build_declares_every_input_its_family_resolves():
    for step, name in zip(_row("training").builds, ("gemma", "qwen")):
        family = PAPER_OUTPUTS[name].family
        expected = {*re.findall(r"artifact://\S+", " ".join(family.arguments())), f"repo://{family.rule_set}"}
        assert set(step.inputs) == expected, name
    assert (table.GEMMA.path, table.QWEN.path) == (GEMMA_GOLDEN, QWEN_GOLDEN)


def test_every_committed_result_is_a_golden_of_the_table_and_the_table_s_goldens_load():
    files = {path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / "results").rglob("*") if path.is_file()}
    committed = {path for path in files if path.endswith(".json")}
    assert files - committed == {"results/README.md"}  # the directory's description, beside the goldens
    assert committed == {golden.path for golden in table.goldens()} == {f"results/{path}" for path in published.GOLDENS}
    assert all(json.loads((REPO_ROOT / path).read_text(encoding="utf-8")) for path in committed)


def test_every_step_runs_a_module_of_this_package_on_inputs_that_are_committed():
    for step in STEPS:
        words = step.words
        assert words[:2] == ["python", "-m"] and importlib.util.find_spec(words[2]) is not None, step.id
        for word in (*words, *step.inputs):
            if word.startswith("repo://"):
                assert (REPO_ROOT / word.removeprefix("repo://")).is_file(), (step.id, word)


def test_a_step_names_out_only_as_the_start_of_a_path_and_no_two_steps_write_one_place():
    placed = []
    for step in STEPS:
        assert all(word.startswith("{out}/") for word in step.words if "{out}" in word), step.id
        placed += [word for word in step.words if word.startswith("{out}/")]
    assert len(placed) == len(set(placed))
    logs = [step.title for row in table.TABLE for step in row.builds]
    assert all(logs) and len(logs) == len(set(logs))


def test_each_plot_of_a_golden_reads_that_golden():
    for row in table.TABLE:
        paths = {f"repo://{step.golden.path}" for step in row.builds if step.golden}
        read = {word for step in row.plots for word in step.words if word.startswith("repo://results/")}
        assert read == paths, row.name


#: Each training drawing's file, by the golden it reads, the family it draws (``--paper``; none for a family's own
#: golden) and the flag that names the panel.
TRAINING_PLOTS = {
    ("results/training/gemma.json", None, "--figure-a"): "grpo_inconsistency",
    ("results/training/qwen.json", None, "--figure-a"): "grpo_qwen_inconsistency",
    ("results/training/judgment_changes.json", "gemma", "--figure-changes"): "grpo_judgment_changes",
    ("results/training/judgment_changes.json", "gemma", "--figure-counts"): "grpo_judgment_counts",
    ("results/training/judgment_changes.json", "qwen", "--figure-changes"): "grpo_qwen_judgment_changes",
    ("results/training/judgment_changes.json", "qwen", "--figure-counts"): "grpo_qwen_judgment_counts",
    ("results/training/frozen_anchors.json", "gemma", "--figure-rates"): "grpo_frozen_anchors",
    ("results/training/frozen_anchors.json", "qwen", "--figure-rates"): "grpo_qwen_frozen_anchors",
}


def test_each_training_drawing_writes_its_family_s_panel_under_that_panel_s_name():
    found: dict = {}
    for step in _row("training").plots:
        words = step.words
        golden = words[words.index("--from_golden") + 1].removeprefix("repo://")
        paper = words[words.index("--paper") + 1] if "--paper" in words else None
        for flag, target in zip(words, words[1:]):
            if flag.startswith("--figure"):
                found.setdefault((golden, paper, flag), set()).add(target)
    assert found == {key: {f"{{out}}/{name}.pdf", f"{{out}}/{name}.png"} for key, name in TRAINING_PLOTS.items()}


def test_the_rerun_tolerances_are_the_published_ones():
    assert {golden.name: golden.rerun_tolerance for golden in table.goldens()} == {
        "ruleset_size_curves": 0.005, "ruleset_size_repeated_draw_baseline": 0.005, "prompt_ablation": 0.005, "hierarchy": 0.005, "hierarchy_ladder": 0.005,
        "leniency": 0.0,
        "ruleforge_final_eval": 0.0, "training_gemma": 0.0, "training_qwen": 0.0,
        "grpo_judgment_changes": 0.0, "grpo_frozen_anchors": 0.0}
