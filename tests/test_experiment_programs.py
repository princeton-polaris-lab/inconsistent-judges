from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import pytest

from ruleforge.core.optional import MissingDependency
from ruleforge.core.paths import repository_root
from tests.conftest import requires_git_worktree

#: Each paper program, by the module ``python -m MODULE`` runs.
PROGRAMS = (
    "ruleforge.experiments.ruleset_size.build",
    "ruleforge.experiments.ruleset_size.build_repeated_draw_baseline",
    "ruleforge.experiments.ruleset_size.generate_subsets",
    "ruleforge.experiments.ruleset_size.verify_subsets",
    "ruleforge.experiments.ruleforge.build",
    "ruleforge.experiments.ruleforge.plot",
    "ruleforge.experiments.prompt_ablation.build",
    "ruleforge.experiments.leniency.build",
    "ruleforge.experiments.prompt_ablation.templates",
    "ruleforge.experiments.hierarchy.build",
    "ruleforge.experiments.hierarchy.build_ladder",
    "ruleforge.experiments.reproduce.verify_rerun",
    "ruleforge.experiments.training.build",
)


@requires_git_worktree
def test_no_script_is_left_in_the_tree(repo_root):
    tracked = subprocess.run(["git", "ls-files", "--", "scripts"], cwd=repo_root, capture_output=True, text=True,
                             check=True).stdout.split()
    assert tracked == []


@pytest.mark.parametrize("module", sorted(PROGRAMS))
def test_a_module_s_command_line_has_its_help_and_runs_as_a_program(module, capsys):
    with pytest.raises(SystemExit) as exit_info:
        importlib.import_module(module).main(["--help"])
    assert exit_info.value.code == 0 and capsys.readouterr().out.startswith("usage: ")
    tree = ast.parse(importlib.util.find_spec(module).loader.get_source(module))
    [block] = [node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"]
    assert "main()" in ast.unparse(block)


def _refused(main, argv, capsys) -> str:
    with pytest.raises(SystemExit) as exit_info:
        main([str(word) for word in argv])
    assert exit_info.value.code == 2
    return capsys.readouterr().err


def test_a_table_without_an_arm_map_is_refused_among_the_paper_s_files(capsys, tmp_path):
    from ruleforge.experiments.prompt_ablation import build as build_table

    repo = repository_root()
    golden = json.loads((repo / "results/prompt_ablation/prompt_ablation.json").read_text(encoding="utf-8"))
    del golden["provenance"]["arm_map"]
    source = tmp_path / "table.json"
    source.write_text(json.dumps(golden), encoding="utf-8")
    target = repo / "results" / "never_written.csv"
    err = _refused(build_table.main, ["--from_golden", source, "--csv", target], capsys)
    assert "hold(s) the paper's files; this output is not one of them" in err
    assert not target.exists()


def test_a_leniency_golden_of_another_forced_style_is_refused_among_the_paper_s_files(capsys, monkeypatch):
    from ruleforge.experiments.leniency import build as build_leniency

    repo = repository_root()
    monkeypatch.setattr(build_leniency, "FORCED", "forced-grpo")
    target = repo / "results" / "prompt_ablation" / "never_written.csv"
    err = _refused(build_leniency.main, ["--from_golden", repo / "results/leniency/leniency.json",
                                             "--csv", target], capsys)
    assert "hold(s) the paper's files; this output is not one of them" in err
    assert not target.exists()


def test_templates_other_than_the_paper_s_set_are_refused_among_the_paper_s_files(capsys):
    from ruleforge.experiments.prompt_ablation import templates

    target = repository_root() / "results" / "never_written"
    err = _refused(templates.main, ["--template", "naive", "--output-dir", target], capsys)
    assert "hold(s) the paper's files" in err and not target.exists()


def test_a_training_panel_and_readout_are_refused_among_the_paper_s_files(capsys, monkeypatch):
    from ruleforge.experiments.training import build as build_family

    repo = repository_root()
    monkeypatch.chdir(repo)  # the family's words name its ruleset relative to the checkout, where it runs
    target = repo / "results" / "never_written.pdf"
    err = _refused(build_family.main, ["--from_golden", repo / "results/training/gemma.json", "--figure-a", target],
                   capsys)
    assert "would be among the paper's files" in err and not target.exists()
    readout = repo / "results" / "training" / "never_written"
    err = _refused(build_family.main, ["--family", "gemma", "--population", "pop-a22af097", "--out-dir", readout],
                   capsys)
    assert "would be among the paper's files" in err and not readout.exists()


def test_an_arm_map_variable_needs_an_arm_map(capsys):
    from ruleforge.experiments.prompt_ablation import build as build_table

    err = _refused(build_table.main, ["--estimates", "x.json", "--arm-map-var", "pop=pop-a22af097"], capsys)
    assert "--arm-map-var gives an arm map's variables; it needs --arm-map" in err



@dataclass(frozen=True)
class GoldenFirst:
    """A program that writes its golden before its figures, run so that it writes a committed golden again."""

    committed: str  # the committed golden
    words: tuple[str, ...]  # its command line, with {committed}, {golden}, {first}, {later}, {out} and {rules} filled in
    replaced: tuple[str, ...] = ()  # a build's functions, each replaced by one returning the committed golden
    golden: str = "golden.json"  # where it writes the golden, under the test's directory
    first: Optional[str] = "first.pdf"  # a figure it draws before the later one (None: none)
    later: str = "later.pdf"


GOLDEN_FIRST = {
    "ruleforge.experiments.leniency.build": GoldenFirst(
        "results/leniency/leniency.json",
        ("--from_golden", "{committed}", "--golden", "{golden}", "--figure", "{first}", "--figure", "{later}")),
    "ruleforge.experiments.prompt_ablation.build": GoldenFirst(
        "results/prompt_ablation/prompt_ablation.json",
        ("--from_golden", "{committed}", "--golden", "{golden}", "--figure_styles", "{first}", "--figure_methods",
         "{later}")),
    "ruleforge.experiments.ruleset_size.build": GoldenFirst(
        "results/ruleset_size/curves.json", ("--from_golden", "{committed}", "--golden", "{golden}", "--figure",
                                             "{later}"), first=None),
    "ruleforge.experiments.hierarchy.build": GoldenFirst(
        "results/hierarchy/hierarchy.json",
        ("--judge", "artifact://flat", "f", "artifact://hierarchy", "h", "--golden", "{golden}", "--figure", "{first}",
         "--figure", "{later}"), ("build_payload",)),
    "ruleforge.experiments.hierarchy.build_ladder": GoldenFirst(
        "results/hierarchy/ladder.json",
        ("--pairs", "pairs.yaml", "--golden", "{golden}", "--figure", "{first}", "--figure", "{later}"),
        ("load_pairs", "build_payload")),
    "ruleforge.experiments.ruleset_size.build_repeated_draw_baseline": GoldenFirst(
        "results/ruleset_size/repeated_draw_baseline.json",
        ("--estimates", "estimates.json", "--golden", "{golden}", "--figure", "{later}"), ("build_payload",),
        first=None),
    "ruleforge.experiments.training.build": GoldenFirst(
        "results/training/gemma.json",
        ("--arm", "Base", "artifact://base", "c", "--reference", "Base", "--rule-set", "{rules}", "--out-dir", "{out}",
         "--force"), ("build_readout",), golden="out/training_family.json", first="out/training_a.pdf",
        later="out/training_a.png"),
}


def _golden_first(module, tmp_path, monkeypatch):
    case = GOLDEN_FIRST[module]
    program = importlib.import_module(module)
    committed = repository_root() / case.committed
    for name in case.replaced:
        monkeypatch.setattr(program, name, lambda *args, **kwargs: json.loads(committed.read_text(encoding="utf-8")))
    monkeypatch.chdir(tmp_path)
    golden = tmp_path / case.golden
    golden.parent.mkdir(exist_ok=True)
    golden.write_text("an older golden\n", encoding="utf-8")
    places = {"committed": committed, "golden": golden, "first": tmp_path / str(case.first),
              "later": tmp_path / case.later, "out": golden.parent,
              "rules": repository_root() / "data" / "rules" / "modelspec" / "openai_rules_all.jsonl"}
    return program.main, [word.format(**places) for word in case.words], committed, golden


@pytest.mark.parametrize("module", sorted(GOLDEN_FIRST))
def test_a_figure_that_fails_to_write_leaves_this_run_s_golden_not_an_older_one(module, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    main, words, committed, golden = _golden_first(module, tmp_path, monkeypatch)
    case = GOLDEN_FIRST[module]
    (tmp_path / case.later).mkdir()
    with pytest.raises(OSError):
        main(words)
    assert json.loads(golden.read_text(encoding="utf-8")) == json.loads(committed.read_text(encoding="utf-8"))
    assert case.first is None or (tmp_path / case.first).read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("module", sorted(GOLDEN_FIRST))
def test_a_run_that_would_draw_without_the_plotting_library_writes_nothing(module, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    main, words, _, golden = _golden_first(module, tmp_path, monkeypatch)
    with pytest.raises(MissingDependency, match="^drawing the figures needs matplotlib, .* --group viz"):
        main(words)
    assert sorted(tmp_path.rglob("*")) == sorted({golden, golden.parent} - {tmp_path})
    assert golden.read_text(encoding="utf-8") == "an older golden\n"
