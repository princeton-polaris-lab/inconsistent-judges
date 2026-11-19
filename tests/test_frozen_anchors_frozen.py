from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ruleforge.core.paths import resolve
from ruleforge.experiments import table
from ruleforge.experiments.training import frozen_anchors as fa
from ruleforge.experiments.training.paper_families import PAPER_OUTPUTS

pytestmark = pytest.mark.frozen

GOLDEN = "results/training/frozen_anchors.json"
PROGRAM = "ruleforge.experiments.training.frozen_anchors"
POPULATION = "pop-a22af097"
#: Scenarios, decoder draws and rules of each family.
N, D, K = 500, 5, 57
#: Each family's panel file, without its suffix.
PANELS = {"gemma": "figures/grpo_frozen_anchors", "qwen": "figures/grpo_qwen_frozen_anchors"}
#: The inconsistency rates: pairs of draws whose full-ruleset judgment differs from the aggregate (of N D^2), with the
#: interval to four places.
RATES = {
    "gemma": {"base": (8_866, 0.6738, 0.7438), "trained_updated": (4_645, 0.3374, 0.4059),
              "trained_frozen": (4_559, 0.3305, 0.3990)},
    "qwen": {"base": (10_399, 0.8099, 0.8532), "trained_updated": (782, 0.0505, 0.0756),
             "trained_frozen": (833, 0.0545, 0.0790)},
}
#: The rates' paired differences, each (estimate, low, high) to four places.
DIFFERENCES = {
    "gemma": {"base_minus_trained_frozen": (0.3446, 0.3091, 0.3798),
              "base_minus_trained_updated": (0.3377, 0.3004, 0.3747),
              "trained_frozen_minus_trained_updated": (-0.0069, -0.0166, 0.0022),
              "trained_frozen_minus_always_yes": (0.2923, 0.2537, 0.3306)},
    "qwen": {"base_minus_trained_frozen": (0.7653, 0.7414, 0.7890),
             "base_minus_trained_updated": (0.7694, 0.7450, 0.7931),
             "trained_frozen_minus_trained_updated": (0.0041, -0.0070, 0.0150),
             "trained_frozen_minus_always_yes": (0.0074, 0.0039, 0.0122)},
}
#: The rates against the frozen anchors split by the anchor: the pairs whose base F, and whose trained F, differs from
#: an anchor that is yes or is not yes (of D per such bundle of the base, SHARES' aggregate counts), and base minus
#: trained_frozen, (estimate, low, high) to four places.
BY_ANCHOR = {
    "gemma": {"yes": (8_858, 4_437, (0.3813, 0.3445, 0.4173)), "not_yes": (8, 122, (-0.1260, -0.1959, -0.0679))},
    "qwen": {"yes": (9_870, 105, (0.8304, 0.8072, 0.8525)), "not_yes": (529, 728, (-0.2689, -0.3464, -0.1912))},
}
#: Each arm's inconsistent bundles at the same draw (of N D): its root's flat rate.
SAME_DRAW = {"gemma": {"base": 1_772, "final": 937}, "qwen": {"base": 2_080, "pick": 158}}
#: The change rates by kind: base to trained, pairs that differ (of N D^2, times K for the rules), and each arm's draw
#: to draw (of N D (D - 1), times K); resampling and excess, each (estimate, low, high) to four places.
CHANGES = {
    "gemma": {
        "per_rule": (25_675, {"base": 20_614, "final": 20_296}, (0.0359, 0.0340, 0.0379), (0.0001, -0.0002, 0.0006)),
        "aggregate": (854, {"base": 644, "final": 712}, (0.0678, 0.0544, 0.0818), (0.0005, -0.0032, 0.0046)),
        "full_set": (4_559, {"base": 528, "final": 1_340}, (0.0934, 0.0824, 0.1044), (0.2713, 0.2379, 0.3050)),
    },
    "qwen": {
        "per_rule": (72_528, {"base": 56_806, "pick": 59_138}, (0.1017, 0.0978, 0.1057), (0.0001, -0.0005, 0.0007)),
        "aggregate": (1_174, {"base": 1_008, "pick": 874}, (0.0941, 0.0797, 0.1093), (-0.0002, -0.0029, 0.0026)),
        "full_set": (10_458, {"base": 1_928, "pick": 184}, (0.1056, 0.0948, 0.1165), (0.7310, 0.7014, 0.7598)),
    },
}
#: Each arm's (yes, unsure, no) bundles, by its full-ruleset judgment and by its aggregate (of N D).
SHARES = {
    "gemma": {"base": {"full_set": (549, 0, 1_951), "aggregate": (2_319, 0, 181)},
              "final": {"full_set": (1_456, 0, 1_044), "aggregate": (2_335, 0, 165)}},
    "qwen": {"base": {"full_set": (385, 36, 2_079), "aggregate": (2_352, 105, 43)},
             "pick": {"full_set": (2_474, 1, 25), "aggregate": (2_363, 107, 30)}},
}
#: The rates panel's printed values, by family.
PRINTED = {"gemma": ["0.709", "0.372", "0.365"], "qwen": ["0.832", "0.063", "0.067"]}
DRAWN = {"gemma": ("Base Gemma", "Trained Gemma"), "qwen": ("Base Qwen", "Trained Qwen")}


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GOLDEN).read_text(encoding="utf-8"))


def _rounded(entry) -> tuple[float, float, float]:
    return tuple(round(entry[key], 4) for key in ("estimate", "ci_low", "ci_high"))


def test_the_golden_holds_both_paper_families_on_the_paper_population(payload):
    assert (payload["source"], payload["schema_version"], payload["population"]) == ("frozen_anchors", 1, POPULATION)
    assert payload["definition"] == fa.DEFINITION and list(payload["families"]) == list(PAPER_OUTPUTS)
    for name, family in payload["families"].items():
        assert (family["n_scenarios"], family["n_rules"], family["provenance"]["n_decoder_draws"]) == (N, K, D)
        assert family["bootstrap"] == {"n_resamples": 10_000, "ci_level": 0.95, "seed": 16,
                                       "root": PAPER_OUTPUTS[name].family.on_population(POPULATION).arms[0].root}


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_family_s_rates_against_its_own_and_the_frozen_base_rule_judgments(payload, name):
    family = payload["families"][name]
    rates = family["inconsistency"]
    for rate, (pairs, low, high) in RATES[name].items():
        assert rates[rate]["estimate"] == pytest.approx(pairs / (N * D * D), abs=1e-12), rate
        assert _rounded(rates[rate])[1:] == (low, high), rate
    # an always-yes judge's rate against the frozen anchors is the base arm's always_yes
    against = {**{rate: rates[rate]["estimate"] for rate in RATES[name]},
               "always_yes": family["always_yes"][family["reference"]]["estimate"]}
    for difference, rounded in DIFFERENCES[name].items():
        first, second = fa.DIFFERENCES[difference]
        assert _rounded(rates["differences"][difference]) == rounded, difference
        assert rates["differences"][difference]["estimate"] == pytest.approx(against[first] - against[second],
                                                                             abs=1e-12)
    for arm, bundles in SAME_DRAW[name].items():
        assert rates["same_draw"][arm]["estimate"] == pytest.approx(bundles / (N * D), abs=1e-12)


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_family_s_rates_by_frozen_anchor_split_its_rates_against_the_frozen_anchors(payload, name):
    family = payload["families"][name]
    aggregate = SHARES[name][family["reference"]]["aggregate"]
    anchors = {"yes": aggregate[0], "not_yes": sum(aggregate[1:])}
    for anchor, (base, trained, difference) in BY_ANCHOR[name].items():
        split = family["inconsistency"]["by_anchor"][anchor]
        assert split["share"]["estimate"] == pytest.approx(anchors[anchor] / (N * D), abs=1e-12)
        assert split["base"]["estimate"] == pytest.approx(base / (anchors[anchor] * D), abs=1e-12)
        assert split["trained_frozen"]["estimate"] == pytest.approx(trained / (anchors[anchor] * D), abs=1e-12)
        assert _rounded(split["base_minus_trained_frozen"]) == difference, anchor
        assert all(entry["n_undefined_replicates"] == 0 for entry in split.values())
    # the two splits hold every pair of each rate against the frozen anchors
    assert sum(pairs[0] for pairs in BY_ANCHOR[name].values()) == RATES[name]["base"][0]
    assert sum(pairs[1] for pairs in BY_ANCHOR[name].values()) == RATES[name]["trained_frozen"][0]


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_family_s_change_rates_and_label_shares(payload, name):
    family = payload["families"][name]
    for kind, (pairs, within, resampling, excess) in CHANGES[name].items():
        rules = K if kind == "per_rule" else 1
        changes = family["changes"][kind]
        assert changes["label"] == dict(fa.KINDS)[kind]
        assert changes["base_to_trained"]["estimate"] == pytest.approx(pairs / (N * D * D * rules), abs=1e-12)
        assert {arm: entry["estimate"] for arm, entry in changes["draw_to_draw"].items()} == pytest.approx(
            {arm: count / (N * D * (D - 1) * rules) for arm, count in within.items()}, abs=1e-12)
        assert (_rounded(changes["resampling"]), _rounded(changes["excess"])) == (resampling, excess), kind
    for arm, which in SHARES[name].items():
        for judgment, counts in which.items():
            assert [family["label_shares"][arm][judgment][label]["estimate"] for label in ("yes", "unsure", "no")] == (
                pytest.approx([count / (N * D) for count in counts], abs=1e-12))
        # an always-yes judge disagrees with each aggregate that is not yes
        assert family["always_yes"][arm]["estimate"] == pytest.approx(sum(which["aggregate"][1:]) / (N * D), abs=1e-12)


@pytest.mark.parametrize("name", list(PAPER_OUTPUTS))
def test_each_family_s_arms_rates_and_draw_to_draw_are_its_other_goldens(repo_root, payload, name):
    """The same-draw rates are the training golden's flat rates; draw to draw, the judgment-changes golden's."""
    family = payload["families"][name]
    training = json.loads((repo_root / PAPER_OUTPUTS[name].golden).read_text(encoding="utf-8"))
    changes = json.loads((repo_root / table.CHANGES.path).read_text(encoding="utf-8"))["families"][name]
    for arm in (family["reference"], family["member"]):
        assert family["inconsistency"]["same_draw"][arm]["estimate"] == pytest.approx(
            training["arms"][arm]["flat"]["estimate"], abs=1e-12)
        for kind, _ in fa.KINDS:
            assert family["changes"][kind]["draw_to_draw"][arm]["estimate"] == pytest.approx(
                changes["draw_to_draw"][arm][kind]["change_rate"], abs=1e-12)
    assert {key: family[key] for key in ("reference", "member", "labels", "provenance")} == {
        key: changes[key] for key in ("reference", "member", "labels", "provenance")}


def test_the_table_builds_it_from_both_families_roots():
    step = next(step for row in table.TABLE for step in row.builds if step.golden and step.golden.path == GOLDEN)
    assert step.words[2:] == [PROGRAM, "--family", "gemma", "--family", "qwen", "--population",
                              POPULATION, "--golden", f"{{out}}/{GOLDEN}"]
    assert list(step.inputs) == [arm.root for name in PAPER_OUTPUTS for arm in PAPER_OUTPUTS[name].family.arms]


def test_the_rates_panel_draws_the_golden_s_numbers_under_the_training_arm_names(payload):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    import matplotlib.pyplot as plt

    from ruleforge.experiments.training.frozen_anchors_figure import build_figure

    for name in PAPER_OUTPUTS:
        base, trained = DRAWN[name]
        figure = build_figure(payload, name, "rates")
        axes = figure.axes[0]
        texts = [text.get_text() for text in axes.texts]
        ticks = [text.get_text() for text in axes.get_xticklabels()]
        legend = axes.get_legend()
        plt.close(figure)
        assert texts == PRINTED[name], name
        assert ticks == [base, f"{trained}\n(own rule\njudgments)", f"{trained}\n(frozen base\nrule judgments)"]
        assert legend is None
        assert not [text for text in texts + ticks if "checkpoint" in text.lower()]


def test_the_rates_panel_renders_from_the_golden(repo_root, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    for name, path in PANELS.items():
        for suffix in ("pdf", "png"):
            out = tmp_path / f"{Path(path).name}.{suffix}"
            assert fa.main(["--from_golden", str(repo_root / GOLDEN), "--paper", name, "--figure-rates",
                            str(out)]) == 0
            assert out.stat().st_size > 0


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_paper_root(repo_root, tmp_path):
    """Tier R: the table's build, run from ``$RULEFORGE_ARTIFACT_ROOT``, writes the committed golden's bytes."""
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("no artifact root is set")
    roots = [arm.root for name in PAPER_OUTPUTS for arm in PAPER_OUTPUTS[name].family.arms]
    if not all((resolve(root) / "aggregate" / "estimates.json").is_file() for root in roots):
        pytest.skip("the training roots are not in this artifact root")
    out = tmp_path / "frozen_anchors.json"
    built = subprocess.run([sys.executable, "-m", PROGRAM, "--family", "gemma", "--family", "qwen",
                            "--population", POPULATION, "--golden", str(out)], cwd=repo_root,
                           env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                                "SOURCE_DATE_EPOCH": "0"}, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    assert out.read_bytes() == (repo_root / GOLDEN).read_bytes()
