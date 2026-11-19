from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.experiments.training import family_arms, paper_families
from ruleforge.experiments.training import judgment_changes as jc
from ruleforge.experiments.training.family_readout import FamilyArm
from ruleforge.experiments.training.paper_families import PaperFamily
from tests.support.training_campaign import _roots  # noqa: F401 -- an autouse fixture
from tests import test_training_family_readout as family_tests
from tests.test_sampling_campaign import set_recorded_runtime
from tests.test_training_family_readout import LABELS, PREFIX

#: Root A (base and grpo) and root B (final alone, its reference in A), analysed: the family tests' fixture.
two_roots = family_tests.two_roots
POP = "pop-0123abcd"
MEMBERS = {"gemma": ("final", "artifact://B/root"), "qwen": ("grpo", "artifact://A/root")}
#: What each family's golden holds, derived by hand from the fake judge's answers at draw 0: every arm's rule
#: judgments are (yes, no) in 11 scenarios and (no, no) in one, so each aggregate is yes 11 times and no once.
EXPECTED = {
    "gemma": {
        "changes": {"per_rule": (24, 0), "aggregate": (12, 0), "full_set": (12, 12)},
        "member_counts": {"per_rule": {"yes": 11, "unsure": 0, "no": 13, "n": 24},
                          "aggregate": {"yes": 11, "unsure": 0, "no": 1, "n": 12},
                          "full_set": {"yes": 12, "unsure": 0, "no": 0, "n": 12}},
    },
    "qwen": {
        "changes": {"per_rule": (24, 0), "aggregate": (12, 0), "full_set": (12, 11)},
        "member_counts": {"per_rule": {"yes": 11, "unsure": 0, "no": 13, "n": 24},
                          "aggregate": {"yes": 11, "unsure": 0, "no": 1, "n": 12},
                          "full_set": {"yes": 11, "unsure": 0, "no": 1, "n": 12}},
    },
}
#: Each arm's (n, changed) between its draws 0 and 1, by kind, derived by hand from the fake judge's answers: final's
#: draw 1 turns one scenario's (yes, no) into (no, no), grpo's one into (no, unsure), an aggregate yes to unsure.
DRAW_TO_DRAW = {"base": {"per_rule": (24, 0), "aggregate": (12, 0), "full_set": (12, 1)},
                "grpo": {"per_rule": (24, 2), "aggregate": (12, 1), "full_set": (12, 1)},
                "final": {"per_rule": (24, 1), "aggregate": (12, 1), "full_set": (12, 0)}}
#: Two synthetic arms over three scenarios and two draws: each bundle's (full-ruleset label, rule 1's, rule 2's).
#: Their aggregates are tied at yes (base s1 draw 0), no (base s2 draw 1, s3 draw 0) and unsure (trained s3 draw 0).
BASE = {"s1": [("yes", "yes", "yes"), ("yes", "yes", "no")],
        "s2": [("no", "unsure", "no"), ("no", "no", "no")],
        "s3": [("no", "no", "no"), ("unsure", "no", "unsure")]}
TRAINED = {"s1": [("yes", "no", "yes"), ("yes", "yes", "yes")],
           "s2": [("no", "no", "no"), ("no", "no", "no")],
           "s3": [("unsure", "unsure", "unsure"), ("no", "no", "no")]}
#: What the two arms give, by hand. Draw 0, base to trained: s1 changes one of its two tied rules and keeps its yes
#: aggregate; s2's one unsure rule turns no, and with it the aggregate; s3 turns both rules, its aggregate and its
#: full-ruleset judgment from no to unsure.
SYNTHETIC = {
    "changes": {"per_rule": (6, 4), "aggregate": (3, 2), "full_set": (3, 1)},
    "member_counts": {"per_rule": {"yes": 1, "unsure": 2, "no": 3, "n": 6},
                      "aggregate": {"yes": 1, "unsure": 1, "no": 1, "n": 3},
                      "full_set": {"yes": 1, "unsure": 1, "no": 1, "n": 3}},
    # draw 0 against draw 1: base s1 rule 2, s2 rule 1 and s3 rule 2 (aggregates of s2 and s3; s3's full set);
    # trained s1 rule 1 and s3 both rules (s3's aggregate and full set)
    "draw_to_draw": {"base": {"per_rule": (6, 3), "aggregate": (3, 2), "full_set": (3, 1)},
                     "trained": {"per_rule": (6, 3), "aggregate": (3, 1), "full_set": (3, 1)}},
    # the aggregates' transitions, rows the base and columns the trained arm
    "anchors": {"yes": {"yes": 1, "unsure": 0, "no": 0}, "unsure": {"yes": 0, "unsure": 0, "no": 1},
                "no": {"yes": 0, "unsure": 1, "no": 0}},
}


def bundle(full: str, *rules: str) -> dict:
    """A bundle outcome of these labels, its stored aggregate and inconsistency its labels' own."""
    top = max(rules, key=PHI_MAP.get)
    return {"full_set_judgment": full, "rule_judgments": {f"Rule {k + 1}": label for k, label in enumerate(rules)},
            "max_rule_phi": PHI_MAP[top], "contradiction": float(full != top)}


def cells(arm: dict) -> dict:
    return {(scenario, draw): bundle(*labels) for scenario, draws in arm.items() for draw, labels in enumerate(draws)}


def synthetic(member: str, root: str, *arms: FamilyArm) -> PaperFamily:
    return PaperFamily(
        arms=(FamilyArm("base", "artifact://A/root", f"{PREFIX}/base"), FamilyArm(member, root, f"{PREFIX}/{member}"),
              *arms),
        labels={"base": LABELS["base"], member: LABELS[member]}, rule_set="unused", merge_manifests={})


@pytest.fixture
def renamed(tmp_path, two_roots, monkeypatch):
    """The two roots under their rebased names, each switch standing for a synthetic family, in a scratch checkout."""
    for name in ("A", "B"):
        shutil.copytree(tmp_path / name / "root", tmp_path / name / f"root-{POP}")
    monkeypatch.setattr(paper_families, "GEMMA_FAMILY", synthetic(*MEMBERS["gemma"]))
    monkeypatch.setattr(paper_families, "QWEN_FAMILY", synthetic(*MEMBERS["qwen"]))
    monkeypatch.setattr(family_arms, "repository_root", lambda: tmp_path / "repo")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    return tmp_path


def run(argv, capsys) -> tuple[int, str]:
    try:
        code = jc.main([str(arg) for arg in argv])
    except SystemExit as exit_info:
        code = exit_info.code
    return code, capsys.readouterr().err


def build(renamed, capsys, *families: str) -> dict:
    golden = renamed / "out" / "judgment_changes.json"
    code, err = run(["--population", POP, "--golden", golden,
                     *[word for name in families or MEMBERS for word in ("--family", name)]], capsys)
    assert code == 0, err
    return json.loads(golden.read_text())


def test_each_family_s_changes_and_counts_are_the_fake_judge_s_at_draw_0(renamed, capsys):
    golden = build(renamed, capsys)
    assert (golden["source"], golden["schema_version"], golden["draw"], golden["population"]) == (
        jc.SOURCE, jc.SCHEMA_VERSION, 0, POP)
    assert golden["definition"] == jc.DEFINITION and list(golden["families"]) == list(MEMBERS)
    for name, (member, root) in MEMBERS.items():
        family = golden["families"][name]
        assert (family["reference"], family["member"]) == ("base", member)
        assert family["labels"] == {"base": LABELS["base"], member: LABELS[member]}
        assert {kind: (row["n"], row["changed"]) for kind, row in family["changes"].items()} == EXPECTED[name]["changes"]
        for kind, label in jc.COMPARISONS:
            row = family["changes"][kind]
            assert row["label"] == label and row["change_rate"] == row["changed"] / row["n"]
            matrix = family["transitions"][jc.TRANSITIONS[kind]]
            assert sum(map(sum, (r.values() for r in matrix.values()))) == row["n"]
            assert row["n"] - sum(matrix[label][label] for label in matrix) == row["changed"]
            assert {label: sum(r[label] for r in matrix.values()) for label in matrix} == {
                label: family["member_counts"][kind][label] for label in matrix}
        assert family["member_counts"] == EXPECTED[name]["member_counts"]
        assert family["transitions"]["n_common"] == 12
        # each arm against itself, draw 0 against draw 1 (two draws: one pair), pooled as its one pair
        assert list(family["draw_to_draw"]) == ["base", member]
        for arm, within in family["draw_to_draw"].items():
            expected = {kind: {"n": n, "changed": changed} for kind, (n, changed) in DRAW_TO_DRAW[arm].items()}
            assert within["n_pairs"] == 1 and within["pairs"] == [{"draws": [0, 1], **expected}]
            for kind, (n, changed) in DRAW_TO_DRAW[arm].items():
                assert within[kind] == {"n": n, "changed": changed, "change_rate": changed / n,
                                        "min_change_rate": changed / n, "max_change_rate": changed / n}
        arms = family["provenance"]["arms"]
        assert [arms[label]["root"] for label in ("base", member)] == [f"artifact://A/root-{POP}", f"{root}-{POP}"]
        assert arms[member]["condition_id"] == f"{PREFIX}/{member}" and arms[member]["prompt_style"] == "forced-grpo"
        assert [campaign["root"] for campaign in family["provenance"]["campaigns"]] == list(
            dict.fromkeys([f"artifact://A/root-{POP}", f"{root}-{POP}"]))
        assert all(campaign["estimates_sha256"] for campaign in family["provenance"]["campaigns"])
        assert family["provenance"]["n_decoder_draws"] == 2


def test_the_counts_are_the_trained_arm_s_own_draw_0_labels(renamed, capsys):
    golden = build(renamed, capsys, "gemma")
    rows = [json.loads(line) for line in
            (renamed / "B" / f"root-{POP}" / PREFIX / "final" / "draw-000" / "bundle_outcomes.jsonl").open()]
    found = {"per_rule": [label for row in rows for label in row["rule_judgments"].values()],
             "aggregate": [max(row["rule_judgments"].values(), key=PHI_MAP.get) for row in rows],
             "full_set": [row["full_set_judgment"] for row in rows]}
    counts = golden["families"]["gemma"]["member_counts"]
    for kind, labels in found.items():
        assert counts[kind] == {**{label: labels.count(label) for label in ("yes", "unsure", "no")},
                                "n": len(labels)}, kind


def test_every_change_count_and_transition_of_two_synthetic_arms_is_the_hand_computed_one():
    found = jc.family_changes({"base": cells(BASE), "trained": cells(TRAINED)}, "base", "trained", 2, "here")
    assert {kind: (row["n"], row["changed"]) for kind, row in found["changes"].items()} == SYNTHETIC["changes"]
    assert {kind: row["label"] for kind, row in found["changes"].items()} == dict(jc.COMPARISONS)
    assert found["member_counts"] == SYNTHETIC["member_counts"]
    for arm, kinds in SYNTHETIC["draw_to_draw"].items():
        assert {kind: (found["draw_to_draw"][arm][kind]["n"], found["draw_to_draw"][arm][kind]["changed"])
                for kind in kinds} == kinds, arm
    assert found["transitions"]["anchors"] == SYNTHETIC["anchors"]


#: A bundle of each arm at each draw whose labels are all no: (arm, scenario, draw).
ALL_NO = [("base", "s3", 0), ("base", "s2", 1), ("trained", "s2", 0), ("trained", "s2", 1)]


@pytest.mark.parametrize(("arm", "scenario", "draw"), ALL_NO)
@pytest.mark.parametrize(("field", "value", "message"), [
    ("max_rule_phi", 0.5, "its max_rule_phi 0.5 is not its rule judgments' largest (no)"),
    ("contradiction", 1.0, "its contradiction 1.0 is not whether its full-ruleset judgment (no) differs from its "
                           "aggregate (no)"),
])
def test_a_bundle_whose_stored_aggregate_or_inconsistency_is_not_its_labels_is_refused_in_either_arm_at_any_draw(
        arm, scenario, draw, field, value, message):
    arms = {"base": cells(BASE), "trained": cells(TRAINED)}
    arms[arm][(scenario, draw)][field] = value
    expected = f"here arm {arm}: scenario {scenario} draw {draw}: {message}"
    with pytest.raises(jc.JudgmentChangesError, match=re.escape(expected)):
        jc.family_changes(arms, "base", "trained", 2, "here")


def test_a_family_of_other_than_two_arms_and_a_label_of_no_kind_are_refused(renamed):
    third = FamilyArm("grpo", "artifact://A/root", f"{PREFIX}/grpo")
    with pytest.raises(jc.JudgmentChangesError, match="3 arms; this reads a base arm and one trained arm"):
        jc.family_readout(synthetic(*MEMBERS["gemma"], third), "here")
    outcomes = renamed / "B" / "root" / PREFIX / "final" / "draw-000" / "bundle_outcomes.jsonl"
    recorded = outcomes.read_text()
    rows = [json.loads(line) for line in recorded.splitlines()]
    rows[0]["rule_judgments"]["Rule 2"] = "maybe"
    outcomes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(jc.JudgmentChangesError, match="here: unparseable rule judgment 'maybe'"):
        jc.family_readout(synthetic(*MEMBERS["gemma"]), "here")
    outcomes.write_text(recorded)
    later = outcomes.parent.parent / "draw-001" / "bundle_outcomes.jsonl"
    rows = [json.loads(line) for line in later.open()]
    rows[0]["full_set_judgment"] = "maybe"
    later.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(jc.JudgmentChangesError, match="here arm final draws 0 and 1: unparseable"):
        jc.family_readout(synthetic(*MEMBERS["gemma"]), "here")


def test_a_family_of_one_decoder_draw_is_refused(renamed, monkeypatch):
    opened = family_arms.open_roots

    def one_draw(*args, **kwargs):
        roots = opened(*args, **kwargs)
        for root in roots.values():
            root.payload["n_decoder_draws"] = 1
        return roots

    monkeypatch.setattr(family_arms, "open_roots", one_draw)
    with pytest.raises(jc.JudgmentChangesError, match="here: 1 decoder draw; the draw-to-draw rates need two or more"):
        jc.family_readout(synthetic(*MEMBERS["gemma"]), "here")


@pytest.mark.parametrize(("change", "message"), [
    (lambda root: [family_tests._edit(path, lambda m: m.update(max_semantic_attempts=40))
                   for path in (root / "sampling_manifest.json", family_tests.estimates_of(root))],
     r"here: the roots' designs differ in max_semantic_attempts \(artifact://A/root 5, artifact://B/root 40\)"),
    (lambda root: set_recorded_runtime(root / family_tests.FINAL, transformers_version="5.5.0"),
     r"here: the arms' judgments come from different inference stacks .*transformers 5.5.0"),
    (lambda root: family_tests.edit_provenance(root / family_tests.FINAL, lambda p: p.update(tp_size=4), ("draw-001",)),
     r"here: the arms ran at tensor-parallel sizes"),
])
def test_arms_of_another_retry_design_or_engine_are_refused_as_the_family_readout_refuses_them(renamed, change, message):
    change(renamed / "B" / "root")
    with pytest.raises(jc.JudgmentChangesError, match=message):
        jc.family_readout(synthetic(*MEMBERS["gemma"]), "here")


@pytest.mark.parametrize(("argv", "message"), [
    (["--family", "gemma", "--population", POP, "--golden", "g.json", "--paper", "gemma"],
     "--paper draw a golden's panels"),
    (["--family", "gemma", "--golden", "g.json"], "a build takes --family (one or more), --population and --golden"),
    (["--family", "gemma", "--population", "pop-XYZ", "--golden", "g.json"],
     "population 'pop-XYZ' is not pop-<first 8 hex"),
])
def test_a_build_is_refused_its_render_flags_a_missing_flag_and_a_bad_population(renamed, capsys, monkeypatch, argv,
                                                                                  message):
    monkeypatch.chdir(renamed)
    code, err = run(argv, capsys)
    assert code == 2 and message in err


def test_a_build_never_writes_into_the_checkout_s_results_and_never_over_a_golden_without_force(renamed, capsys):
    inside = renamed / "repo" / "results" / "training" / "judgment_changes.json"
    code, err = run(["--family", "gemma", "--population", POP, "--golden", inside], capsys)
    assert code == 2 and "would be among the paper's files (results)" in err and not inside.exists()
    first = build(renamed, capsys, "gemma")
    golden = renamed / "out" / "judgment_changes.json"
    code, err = run(["--family", "gemma", "--population", POP, "--golden", golden], capsys)
    assert code == 2 and "exists already; pass --force" in err
    code, err = run(["--family", "gemma", "--family", "gemma", "--population", POP, "--golden", golden, "--force"],
                    capsys)
    assert code == 0 and json.loads(golden.read_text()) == first


@pytest.fixture
def golden_path(renamed, capsys) -> Path:
    build(renamed, capsys)
    return renamed / "out" / "judgment_changes.json"


@pytest.mark.parametrize("paper", sorted(MEMBERS))
def test_a_render_draws_each_family_s_two_panels(golden_path, capsys, paper):
    out = golden_path.parent
    code, err = run(["--from_golden", golden_path, "--paper", paper, "--figure-changes", out / f"{paper}_changes.pdf",
                     "--figure-counts", out / f"{paper}_counts.png", "--dpi", "40"], capsys)
    assert code == 0, err
    assert (out / f"{paper}_changes.pdf").stat().st_size > 0 and (out / f"{paper}_counts.png").stat().st_size > 0
    code, err = run(["--from_golden", golden_path, "--paper", paper, "--figure-counts", out / "alone.pdf"], capsys)
    assert code == 0 and (out / "alone.pdf").stat().st_size > 0


@pytest.mark.parametrize("paper", sorted(MEMBERS))
def test_a_render_draws_each_flag_s_panel_of_the_family_asked_for_into_that_flag_s_file(golden_path, capsys,
                                                                                         monkeypatch, paper):
    from ruleforge.experiments.training import judgment_changes_figure

    drawn = []
    monkeypatch.setattr(judgment_changes_figure, "draw", lambda golden, family, panel, path, dpi: drawn.append(
        (family, panel, Path(path).name, dpi)))
    code, err = run(["--from_golden", golden_path, "--paper", paper, "--figure-counts", "counts.png",
                     "--figure-changes", "changes.pdf", "--dpi", "40"], capsys)
    assert code == 0, err
    assert drawn == [(paper, "changes", "changes.pdf", 40), (paper, "counts", "counts.png", 40)]


def test_a_render_is_refused_build_flags_missing_outputs_and_what_is_no_such_golden(golden_path, capsys):
    out = golden_path.parent
    figure = ["--figure-changes", out / "x.pdf"]
    other = out / "other.json"
    other.write_text(json.dumps({"source": "training_family", "schema_version": 1}))
    broken = out / "broken.json"
    broken.write_text("not json")
    for argv, message in (
        (["--paper", "gemma", *figure, "--population", POP], "do not also pass --population"),
        (["--paper", "gemma"], "pass --paper and --figure-changes or --figure-counts"),
        ([*figure], "pass --paper and --figure-changes or --figure-counts"),
    ):
        code, err = run(["--from_golden", golden_path, *argv], capsys)
        assert code == 2 and message in err, argv
    for source, message in ((other, "is no judgment-changes golden"), (broken, "Expecting value"),
                            (out / "absent.json", "No such file")):
        code, err = run(["--from_golden", source, "--paper", "gemma", *figure], capsys)
        assert code == 2 and message in err, source
    golden = json.loads(golden_path.read_text())
    del golden["families"]["qwen"]
    golden_path.write_text(json.dumps(golden))
    code, err = run(["--from_golden", golden_path, "--paper", "qwen", *figure], capsys)
    assert code == 2 and "holds no family 'qwen' (it holds ['gemma'])" in err
    assert not (out / "x.pdf").exists()


def test_the_panels_draw_the_golden_s_values_under_the_arms_display_labels(golden_path):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from ruleforge.experiments import style
    from ruleforge.experiments.training.family_arms import arm_names
    from ruleforge.experiments.training.judgment_changes_figure import build_figure, panel_values

    golden = json.loads(golden_path.read_text())
    assert arm_names(golden, "gemma") == ("Base Gemma", "Trained Gemma")
    assert panel_values(golden, "gemma", "changes") == [
        {"label": "Individual-rule judgments", "n": 24, "rate": 0.0},
        {"label": "Aggregated rule judgments", "n": 12, "rate": 0.0},
        {"label": "Full-ruleset judgments", "n": 12, "rate": 1.0}]
    assert panel_values(golden, "qwen", "counts") == [
        {"label": "Individual-rule judgments", "yes": 11, "unsure": 0, "no": 13, "n": 24},
        {"label": "Full-ruleset judgments", "yes": 11, "unsure": 0, "no": 1, "n": 12}]
    with pytest.raises(ValueError, match="unknown panel 'flips'"):
        panel_values(golden, "gemma", "flips")
    bold = ("bold", style.VALUE_LABEL["fontsize"], style.VALUE_LABEL["color"])

    changes = build_figure(golden, "gemma", "changes")
    axes = changes.axes[0]
    assert [patch.get_height() for patch in axes.patches] == [0.0, 0.0, 1.0]
    assert [text.get_text() for text in axes.texts] == ["0.0%", "0.0%", "100.0%"]
    assert [text.get_text() for text in axes.get_xticklabels()] == [
        "Individual-rule judgments\n(24 judgments)", "Aggregated rule judgments\n(12 judgments)",
        "Full-ruleset judgments\n(12 judgments)"]
    assert axes.get_ylabel() == "Base-to-GRPO change rate" and axes.get_ylim() == (0.0, 1.0)
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["Base Gemma → Trained Gemma"]
    assert {(t.get_fontweight(), t.get_fontsize(), to_hex(t.get_color()).upper()) for t in axes.texts} == {bold}
    plt.close(changes)

    counts = build_figure(golden, "gemma", "counts")
    axes = counts.axes[0]
    # yes, unsure, no for each kind: bars from 1 to their counts, a zero count no bar, its label at the floor
    assert [patch.get_height() for patch in axes.patches] == [10, 11, 0, 0, 12, 0]
    assert [text.get_text() for text in axes.texts] == ["11", "12", "0", "0", "13", "0"]
    assert [text.get_position()[1] for text in axes.texts][2] == pytest.approx(1.18)
    assert axes.get_yscale() == "log" and axes.get_ylim() == pytest.approx((1, 100_000))
    assert axes.get_ylabel() == "Judgment count (log scale)"
    assert [text.get_text() for text in axes.get_xticklabels()] == [
        "Individual-rule judgments\n(n = 24)", "Full-ruleset judgments\n(n = 12)"]
    legend = axes.get_legend()
    assert legend.get_title().get_text() == "Trained Gemma"
    assert [text.get_text() for text in legend.get_texts()] == ["Yes", "Unsure", "No"]
    assert {(t.get_fontweight(), t.get_fontsize(), to_hex(t.get_color()).upper()) for t in axes.texts} == {bold}
    plt.close(counts)
