from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from ruleforge.evaluate.scoring import PHI_MAP
from ruleforge.experiments.training import frozen_anchors as fa
from ruleforge.experiments.training import judgment_changes as jc
from ruleforge.experiments.training import family_arms, paper_families
from tests.support.training_campaign import _roots  # noqa: F401 -- an autouse fixture
from tests import test_training_family_readout as family_tests
from tests.test_judgment_changes import MEMBERS, POP, cells, synthetic
from tests.test_training_family_readout import FINAL, PREFIX, rewrite_condition

#: Root A (base and grpo) and root B (final alone, its reference in A), analysed: the family tests' fixture.
two_roots = family_tests.two_roots
BOOTSTRAP = {"n_resamples": 300, "ci_level": 0.95, "seed": 16}
#: Two synthetic arms over two scenarios and three draws: each bundle's (full-ruleset label, rule 1's, rule 2's).
#: They hold every label, the unsure aggregate (base, s1 draw 2) and two rules tied at the maximum (base, s1 draw 1).
BASE = {"s1": [("no", "no", "yes"), ("yes", "yes", "yes"), ("unsure", "unsure", "no")],
        "s2": [("no", "no", "no"), ("no", "unsure", "no"), ("yes", "no", "unsure")]}
TRAINED = {"s1": [("yes", "no", "yes"), ("yes", "unsure", "yes"), ("yes", "yes", "no")],
           "s2": [("unsure", "unsure", "no"), ("no", "no", "no"), ("no", "no", "unsure")]}
#: Every estimate the two arms give, derived by hand from the label counts per scenario: across two sets of three
#: draws, 1 - sum_v n_v m_v / 9; within one, 1 - sum_v n_v (n_v - 1) / 6; then the mean of the two scenarios.
EXPECTED = {
    # base: F (n, y, u | n, n, y) against A (y, y, u | n, u, u): 1 - 3/9 and 1 - 2/9
    ("inconsistency", "base"): 13 / 18,
    # trained, its own A (y, y, y | u, n, u): 1 - 9/9 and 1 - 4/9
    ("inconsistency", "trained_updated"): 5 / 18,
    # trained F (y, y, y | u, n, n) against the base's A: 1 - 6/9 and 1 - 4/9
    ("inconsistency", "trained_frozen"): 8 / 18,
    ("inconsistency", "differences", "base_minus_trained_frozen"): 5 / 18,
    ("inconsistency", "differences", "base_minus_trained_updated"): 8 / 18,
    ("inconsistency", "differences", "trained_frozen_minus_trained_updated"): 3 / 18,
    # an always-yes judge against the base's A disagrees wherever it is not yes: always_yes of the base, 2/3
    ("inconsistency", "differences", "trained_frozen_minus_always_yes"): 8 / 18 - 2 / 3,
    # by the base's A: yes at s1 draws 0 and 1, 6 pairs, of which the base's F (n, y, u) differs in 4 and the
    # trained F (y, y, y) in none; not yes at s1 draw 2 and every s2 draw, 3 + 9 pairs, base 2 + 7, trained 3 + 5
    ("inconsistency", "by_anchor", "yes", "share"): 1 / 3,
    ("inconsistency", "by_anchor", "yes", "base"): 4 / 6,
    ("inconsistency", "by_anchor", "yes", "trained_frozen"): 0.0,
    ("inconsistency", "by_anchor", "yes", "base_minus_trained_frozen"): 4 / 6,
    ("inconsistency", "by_anchor", "not_yes", "share"): 2 / 3,
    ("inconsistency", "by_anchor", "not_yes", "base"): 9 / 12,
    ("inconsistency", "by_anchor", "not_yes", "trained_frozen"): 8 / 12,
    ("inconsistency", "by_anchor", "not_yes", "base_minus_trained_frozen"): 1 / 12,
    # draw by draw: base 1/3 and 2/3, trained 0 and 1/3
    ("inconsistency", "same_draw", "base"): 1 / 2,
    ("inconsistency", "same_draw", "trained"): 1 / 6,
    # rules: per scenario the mean of the two rules' (s1 6/9 and 4/9, s2 4/9 and 4/9)
    ("changes", "per_rule", "base_to_trained"): 1 / 2,
    ("changes", "per_rule", "draw_to_draw", "base"): 3 / 4,
    ("changes", "per_rule", "draw_to_draw", "trained"): 3 / 4,
    ("changes", "per_rule", "resampling"): 3 / 4,
    ("changes", "per_rule", "excess"): -1 / 4,
    # aggregate: across 1 - 6/9 and 1 - 5/9; within base 2/3 and 2/3, trained 0 and 2/3
    ("changes", "aggregate", "base_to_trained"): 7 / 18,
    ("changes", "aggregate", "draw_to_draw", "base"): 2 / 3,
    ("changes", "aggregate", "draw_to_draw", "trained"): 1 / 3,
    ("changes", "aggregate", "resampling"): 1 / 2,
    ("changes", "aggregate", "excess"): -1 / 9,
    # full set: across 1 - 3/9 and 1 - 4/9; within base 1 and 2/3, trained 0 and 2/3
    ("changes", "full_set", "base_to_trained"): 11 / 18,
    ("changes", "full_set", "draw_to_draw", "base"): 5 / 6,
    ("changes", "full_set", "draw_to_draw", "trained"): 1 / 3,
    ("changes", "full_set", "resampling"): 7 / 12,
    ("changes", "full_set", "excess"): 1 / 36,
    **{("label_shares", "base", "full_set", label): share for label, share in (("yes", 1 / 3), ("unsure", 1 / 6),
                                                                                ("no", 1 / 2))},
    **{("label_shares", "base", "aggregate", label): share for label, share in (("yes", 1 / 3), ("unsure", 1 / 2),
                                                                                 ("no", 1 / 6))},
    **{("label_shares", "trained", "full_set", label): share for label, share in (("yes", 1 / 2), ("unsure", 1 / 6),
                                                                                   ("no", 1 / 3))},
    **{("label_shares", "trained", "aggregate", label): share for label, share in (("yes", 1 / 2), ("unsure", 1 / 3),
                                                                                    ("no", 1 / 6))},
    ("always_yes", "base"): 2 / 3,
    ("always_yes", "trained"): 1 / 2,
}


def numbers(base: dict, trained: dict, n_draws: int = 3) -> dict:
    return fa.family_numbers({"base": base, "trained": trained}, "base", "trained", ["s1", "s2"], n_draws, BOOTSTRAP,
                             "here")


def at(node: dict, path: tuple) -> dict:
    for key in path:
        node = node[key]
    return node


def test_every_estimate_of_two_synthetic_arms_is_the_hand_computed_one():
    found = numbers(cells(BASE), cells(TRAINED))
    assert (found["n_scenarios"], found["n_rules"]) == (2, 2)
    for path, expected in EXPECTED.items():
        assert at(found, path)["estimate"] == pytest.approx(expected, abs=1e-12), path
    # every place the golden fills is one of the hand-computed ones, or a kind's panel label
    places = {path for path in EXPECTED}
    filled = set()

    def walk(node, path=()):
        if "estimate" in node:
            filled.add(path)
            return
        for key, value in node.items():
            if isinstance(value, dict):
                walk(value, (*path, key))
    walk(found)
    assert filled == places
    assert {kind: found["changes"][kind]["label"] for kind, _ in fa.KINDS} == dict(fa.KINDS)


def test_each_interval_holds_its_estimate_and_comes_from_the_one_bootstrap():
    found = numbers(cells(BASE), cells(TRAINED))
    for path in EXPECTED:
        entry = at(found, path)
        assert entry["ci_low"] <= entry["estimate"] + 1e-12 and entry["estimate"] <= entry["ci_high"] + 1e-12, path
        assert entry["interval_basis"] == "scenario_cluster_bootstrap", path
        # a resample of s2 alone holds no pair whose frozen anchor is yes
        split = path[:3] == ("inconsistency", "by_anchor", "yes") and path[3] != "share"
        assert (0 < entry["n_undefined_replicates"] < BOOTSTRAP["n_resamples"]) if split else (
            entry["n_undefined_replicates"] == 0), path


def test_a_rate_by_frozen_anchor_is_undefined_where_no_frozen_anchor_is_that_label():
    never_yes = {scenario: [("no", "no", "unsure")] * 3 for scenario in BASE}
    found = numbers(cells(never_yes), cells(TRAINED))["inconsistency"]["by_anchor"]
    for part in ("base", "trained_frozen", "base_minus_trained_frozen"):
        assert (found["yes"][part]["estimate"], found["yes"][part]["ci_low"], found["yes"][part]["sd"]) == (
            None, None, None)
        assert found["yes"][part]["n_undefined_replicates"] == BOOTSTRAP["n_resamples"]
    assert found["yes"]["share"]["estimate"] == 0.0 and found["not_yes"]["share"]["estimate"] == 1.0
    # the base's F (no) differs from its A (unsure) in every pair; the trained F (y, y, y | u, n, n) in 9 + 6 of 18
    assert (found["not_yes"]["base"]["estimate"], found["not_yes"]["trained_frozen"]["estimate"]) == pytest.approx(
        (1.0, 15 / 18))


def test_a_difference_s_interval_is_paired_scenario_by_scenario():
    """Trained rule judgments equal the base's, so a joint bootstrap's difference is 0 in every replicate."""
    trained = {scenario: [(full, *base[1:]) for full, base in zip(("yes", "no", "yes"), draws)]
               for scenario, draws in BASE.items()}
    found = numbers(cells(BASE), cells(trained))
    difference = found["inconsistency"]["differences"]["trained_frozen_minus_trained_updated"]
    assert (difference["estimate"], difference["ci_low"], difference["ci_high"], difference["sd"]) == (0.0, 0.0, 0.0,
                                                                                                        0.0)
    assert found["inconsistency"]["trained_frozen"]["sd"] > 0


def _reordered(arm: dict) -> dict:
    """An arm's bundles stored in the other order, each holding its rule judgments in the other order."""
    rows = {key: {**row, "rule_judgments": dict(reversed(row["rule_judgments"].items()))}
            for key, row in cells(arm).items()}
    return dict(reversed(rows.items()))


def test_the_arms_pair_by_scenario_draw_and_rule_whatever_order_their_bundles_and_rules_are_stored_in():
    canonical = numbers(cells(BASE), cells(TRAINED))
    assert numbers(cells(BASE), _reordered(TRAINED)) == canonical
    assert numbers(_reordered(BASE), cells(TRAINED)) == canonical


def _broken(edit) -> dict:
    trained = cells(TRAINED)
    edit(trained[("s2", 1)])
    return trained


@pytest.mark.parametrize(("edit", "message"), [
    (lambda row: row.update(full_set_judgment="maybe"), "here arm trained: scenario s2 draw 1 holds the label 'maybe', "
                                                        "not yes, unsure or no"),
    (lambda row: row["rule_judgments"].update({"Rule 2": "Yes"}), "holds the label 'Yes', not yes, unsure or no"),
    (lambda row: row["rule_judgments"].pop("Rule 2"), "scenario s2 draw 1 does not hold one judgment for each of the "
                                                       "family's 2 rules"),
    (lambda row: row["rule_judgments"].update({"Rule 3": "no"}), "does not hold one judgment for each"),
    (lambda row: row.update(max_rule_phi=0.5), "scenario s2 draw 1: its max_rule_phi 0.5 is not its rule judgments' "
                                               "largest (no)"),
    (lambda row: row.update(contradiction=1.0), "its contradiction 1.0 is not whether its full-ruleset judgment (no) "
                                                "differs from its aggregate (no)"),
])
def test_a_bundle_of_another_label_rule_set_aggregate_or_inconsistency_is_refused(edit, message):
    with pytest.raises(fa.FrozenAnchorsError, match=message.replace("(", r"\(").replace(")", r"\)")):
        numbers(cells(BASE), _broken(edit))


def test_arms_of_other_scenarios_or_draws_one_label_twice_and_a_single_draw_are_refused():
    trained = cells(TRAINED)
    del trained[("s2", 2)]
    with pytest.raises(fa.FrozenAnchorsError, match="arm trained's bundles are not the 2 scenarios x 3 draws"):
        numbers(cells(BASE), trained)
    with pytest.raises(fa.FrozenAnchorsError, match="here: both arms are labelled 'base'"):
        fa.family_numbers({"base": cells(BASE)}, "base", "base", ["s1", "s2"], 3, BOOTSTRAP, "here")
    single = {key: row for key, row in cells(BASE).items() if key[1] == 0}
    with pytest.raises(fa.FrozenAnchorsError, match="here: 1 decoder draw; the draw-to-draw rates need two or more"):
        numbers(single, {key: row for key, row in cells(TRAINED).items() if key[1] == 0}, n_draws=1)


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
        code = fa.main([str(arg) for arg in argv])
    except SystemExit as exit_info:
        code = exit_info.code
    return code, capsys.readouterr().err


def build(renamed, capsys, *families: str) -> dict:
    golden = renamed / "out" / "frozen_anchors.json"
    code, err = run(["--population", POP, "--golden", golden, "--force",
                     *[word for name in families or MEMBERS for word in ("--family", name)]], capsys)
    assert code == 0, err
    return json.loads(golden.read_text())


def _outcomes(renamed: Path, root: str, label: str, n_draws: int) -> dict:
    """An arm's bundles straight from its draws' outcome files, ``{scenario: [row of each draw]}``."""
    found: dict = {}
    for draw in range(n_draws):
        path = renamed / root.removeprefix("artifact://") / PREFIX / label / f"draw-{draw:03d}" / "bundle_outcomes.jsonl"
        for line in path.open():
            row = json.loads(line)
            found.setdefault(row["scenario_id"], []).append(row)
    return found


def _across(first, second) -> float:
    return sum(a != b for a in first for b in second) / (len(first) * len(second))


def _within(labels) -> float:
    pairs = [(a, b) for i, a in enumerate(labels) for j, b in enumerate(labels) if i != j]
    return sum(a != b for a, b in pairs) / len(pairs)


def test_a_build_reads_both_families_as_the_draws_give_them(renamed, capsys):
    """Every rate again, by plain loops over the outcome files, apart from the readers and the arrays."""
    golden = build(renamed, capsys)
    assert (golden["source"], golden["schema_version"], golden["population"]) == (fa.SOURCE, fa.SCHEMA_VERSION, POP)
    assert golden["definition"] == fa.DEFINITION and list(golden["families"]) == list(MEMBERS)
    for name, (member, root) in MEMBERS.items():
        family = golden["families"][name]
        n_draws = family["provenance"]["n_decoder_draws"]
        base = _outcomes(renamed, f"artifact://A/root-{POP}", "base", n_draws)
        trained = _outcomes(renamed, f"{root}-{POP}", member, n_draws)
        assert (family["reference"], family["member"], family["n_scenarios"]) == ("base", member, len(base))

        def full(arm, scenario):
            return [row["full_set_judgment"] for row in arm[scenario]]

        def aggregate(arm, scenario):
            return [max(row["rule_judgments"].values(), key=PHI_MAP.get) for row in arm[scenario]]

        def rule(arm, scenario, rule_id):
            return [row["rule_judgments"][rule_id] for row in arm[scenario]]

        rules = list(next(iter(base.values()))[0]["rule_judgments"])
        assert family["n_rules"] == len(rules)

        def mean(values):
            return sum(values) / len(values)

        expected = {
            ("inconsistency", "base"): mean([_across(full(base, s), aggregate(base, s)) for s in base]),
            ("inconsistency", "trained_updated"): mean([_across(full(trained, s), aggregate(trained, s)) for s in base]),
            ("inconsistency", "trained_frozen"): mean([_across(full(trained, s), aggregate(base, s)) for s in base]),
            ("changes", "full_set", "base_to_trained"): mean([_across(full(trained, s), full(base, s)) for s in base]),
            ("changes", "aggregate", "base_to_trained"): mean([_across(aggregate(trained, s), aggregate(base, s))
                                                              for s in base]),
            ("changes", "per_rule", "base_to_trained"): mean([mean([_across(rule(trained, s, r), rule(base, s, r))
                                                                    for r in rules]) for s in base]),
            ("changes", "full_set", "draw_to_draw", member): mean([_within(full(trained, s)) for s in base]),
            ("changes", "aggregate", "draw_to_draw", "base"): mean([_within(aggregate(base, s)) for s in base]),
            ("changes", "per_rule", "draw_to_draw", "base"): mean([mean([_within(rule(base, s, r)) for r in rules])
                                                                   for s in base]),
            ("always_yes", "base"): mean([mean([a != "yes" for a in aggregate(base, s)]) for s in base]),
        }
        for path, value in expected.items():
            assert at(family, path)["estimate"] == pytest.approx(value, abs=1e-12), (name, path)
        # the same-draw rates are each root's own flat rate
        for label, arm_root in (("base", f"artifact://A/root-{POP}"), (member, f"{root}-{POP}")):
            estimates = json.loads((renamed / arm_root.removeprefix("artifact://") / "aggregate" /
                                    "estimates.json").read_text())
            assert family["inconsistency"]["same_draw"][label]["estimate"] == pytest.approx(
                estimates["conditions"][f"{PREFIX}/{label}"]["estimate"], abs=1e-12)
        assert family["bootstrap"] == {"n_resamples": family_tests.N_RESAMPLES, "ci_level": 0.95, "seed": 16,
                                       "root": f"artifact://A/root-{POP}"}


def test_the_bootstrap_is_the_reference_root_s_whatever_the_trained_arm_s_root_records(renamed, monkeypatch):
    family = synthetic(*MEMBERS["gemma"]).on_population(POP)
    expected = fa.family_readout(family, "here")
    read = fa.read_family_arms

    def other_settings(*args, **kwargs):
        arms = read(*args, **kwargs)
        payload = arms.roots[arms.member.root].payload
        payload["bootstrap"] = {**payload["bootstrap"], "n_resamples": 50, "seed": 99}
        return arms

    monkeypatch.setattr(fa, "read_family_arms", other_settings)
    assert fa.family_readout(family, "here") == expected
    assert expected["bootstrap"] == {"n_resamples": family_tests.N_RESAMPLES, "ci_level": 0.95, "seed": 16,
                                     "root": f"artifact://A/root-{POP}"}


def test_a_family_whose_same_draw_rate_is_not_its_root_s_flat_rate_is_refused(renamed, monkeypatch):
    family = synthetic(*MEMBERS["gemma"]).on_population(POP)
    read = fa.read_family_arms

    def shifted(*args, **kwargs):
        arms = read(*args, **kwargs)
        conditions = copy.deepcopy(arms.roots[arms.member.root].payload["conditions"])
        conditions[f"{PREFIX}/final"]["estimate"] += 0.01
        arms.roots[arms.member.root].payload["conditions"] = conditions
        return arms

    monkeypatch.setattr(fa, "read_family_arms", shifted)
    with pytest.raises(fa.FrozenAnchorsError, match="arm final's same-draw rate .* is not its root's flat rate"):
        fa.family_readout(family, "here")


@pytest.mark.parametrize(("module", "error"), [(fa, fa.FrozenAnchorsError), (jc, jc.JudgmentChangesError)])
def test_a_family_of_other_than_two_arms_or_of_arms_on_another_ruleset_is_refused_in_each_golden_s_error(
        renamed, module, error):
    third = family_tests.FamilyArm("grpo", "artifact://A/root", f"{PREFIX}/grpo")
    with pytest.raises(error, match="3 arms; this reads a base arm and one trained arm"):
        module.family_readout(synthetic(*MEMBERS["gemma"], third), "here")
    rewrite_condition(renamed / "B" / f"root-{POP}", FINAL, rule_set_sha256="0" * 64)
    with pytest.raises(error, match=r"here: arm final differs from the reference base in \['rule_set_sha256'\], not "
                                    "only in the weights it serves"):
        module.family_readout(synthetic(*MEMBERS["gemma"]).on_population(POP), "here")


@pytest.mark.parametrize(("argv", "message"), [
    (["--family", "gemma", "--population", POP, "--golden", "g.json", "--figure-rates", "x.pdf"],
     "--figure-rates draw a golden's panels"),
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
    inside = renamed / "repo" / "results" / "training" / "frozen_anchors.json"
    code, err = run(["--family", "gemma", "--population", POP, "--golden", inside], capsys)
    assert code == 2 and "would be among the paper's files (results)" in err and not inside.exists()
    first = build(renamed, capsys, "gemma")
    golden = renamed / "out" / "frozen_anchors.json"
    code, err = run(["--family", "gemma", "--population", POP, "--golden", golden], capsys)
    assert code == 2 and "exists already; pass --force" in err
    assert build(renamed, capsys, "gemma", "gemma") == first


def test_a_build_writes_its_golden_under_the_checkout_s_figures(renamed, capsys):
    """The checkout commits no figures/: a build writes its golden there as anywhere else."""
    golden = renamed / "repo" / "figures" / "frozen_anchors.json"
    code, err = run(["--family", "gemma", "--population", POP, "--golden", golden], capsys)
    assert code == 0, err
    assert json.loads(golden.read_text()) == build(renamed, capsys, "gemma")


@pytest.fixture
def golden_path(renamed, capsys) -> Path:
    build(renamed, capsys)
    return renamed / "out" / "frozen_anchors.json"


@pytest.mark.parametrize("paper", sorted(MEMBERS))
def test_a_render_draws_each_family_s_rates_panel(golden_path, capsys, paper):
    out = golden_path.parent
    for path in (out / f"{paper}_rates.pdf", out / f"{paper}_rates.png"):
        code, err = run(["--from_golden", golden_path, "--paper", paper, "--figure-rates", path, "--dpi", "40"],
                        capsys)
        assert code == 0 and path.stat().st_size > 0, err


@pytest.mark.parametrize("paper", sorted(MEMBERS))
def test_a_render_draws_each_flag_s_panel_of_the_family_asked_for_into_that_flag_s_file(golden_path, capsys,
                                                                                         monkeypatch, paper):
    from ruleforge.experiments.training import frozen_anchors_figure

    drawn = []
    monkeypatch.setattr(frozen_anchors_figure, "draw", lambda golden, family, panel, path, dpi: drawn.append(
        (family, panel, Path(path).name, dpi)))
    code, err = run(["--from_golden", golden_path, "--paper", paper, "--figure-rates", "rates.pdf", "--dpi", "40"],
                    capsys)
    assert code == 0, err
    assert drawn == [(paper, "rates", "rates.pdf", 40)]


def test_a_render_is_refused_build_flags_missing_outputs_and_what_is_no_such_golden(golden_path, capsys):
    out = golden_path.parent
    figure = ["--figure-rates", out / "x.pdf"]
    other = out / "other.json"
    other.write_text(json.dumps({"source": "judgment_changes", "schema_version": 1}))
    broken = out / "broken.json"
    broken.write_text("not json")
    for argv, message in (
        (["--paper", "gemma", *figure, "--population", POP], "do not also pass --population"),
        (["--paper", "gemma"], "pass --paper and --figure-rates"),
        ([*figure], "pass --paper and --figure-rates"),
        # the frozen-anchors panels draw only the rates
        (["--paper", "gemma", *figure, "--figure-changes", out / "y.pdf"], "unrecognized arguments: --figure-changes"),
    ):
        code, err = run(["--from_golden", golden_path, *argv], capsys)
        assert code == 2 and message in err, argv
    for source, message in ((other, "is no frozen-anchors golden"), (broken, "Expecting value"),
                            (out / "absent.json", "No such file")):
        code, err = run(["--from_golden", source, "--paper", "gemma", *figure], capsys)
        assert code == 2 and message in err, source
    golden = json.loads(golden_path.read_text())
    del golden["families"]["qwen"]
    golden_path.write_text(json.dumps(golden))
    code, err = run(["--from_golden", golden_path, "--paper", "qwen", *figure], capsys)
    assert code == 2 and "holds no family 'qwen' (it holds ['gemma'])" in err
    assert not (out / "x.pdf").exists()


def test_the_rates_panel_draws_the_golden_s_values_in_the_rate_styles(golden_path):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from ruleforge.experiments import style
    from ruleforge.experiments.training.frozen_anchors_figure import build_figure, panel_values

    golden = json.loads(golden_path.read_text())
    family = golden["families"]["gemma"]
    with pytest.raises(ValueError, match="unknown panel 'flips'"):
        panel_values(golden, "gemma", "flips")
    bold = ("bold", style.VALUE_LABEL["fontsize"], style.VALUE_LABEL["color"])

    rates = [family["inconsistency"][name]["estimate"] for name in ("base", "trained_updated", "trained_frozen")]
    assert [row["estimate"] for row in panel_values(golden, "gemma", "rates")] == rates
    figure = build_figure(golden, "gemma", "rates")
    axes = figure.axes[0]
    assert [patch.get_height() for patch in axes.patches] == rates
    # hatched: against its own rule judgments; plain coral: against the frozen base rule judgments
    assert [(to_hex(patch.get_facecolor()).upper(), patch.get_hatch() or "") for patch in axes.patches] == [
        (style.PASTEL_BLUE, ""), (style.PASTEL_CORAL, "//"), (style.PASTEL_CORAL, "")]
    assert [text.get_text() for text in axes.texts] == [f"{rate:.3f}" for rate in rates]
    assert [text.get_text() for text in axes.get_xticklabels()] == [
        "Base Gemma", "Trained Gemma\n(own rule\njudgments)", "Trained Gemma\n(frozen base\nrule judgments)"]
    assert axes.get_ylabel() == "Inconsistency rate" and axes.get_ylim() == (0.0, 1.12)
    assert axes.get_legend() is None and axes.get_title() == ""
    assert {(t.get_fontweight(), t.get_fontsize(), to_hex(t.get_color()).upper()) for t in axes.texts} == {bold}
    plt.close(figure)
