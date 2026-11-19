from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

from ruleforge.experiments import table
from ruleforge.experiments.training.paper_families import GEMMA_FAMILY, GEMMA_FIGURES, GEMMA_GOLDEN, paper_refusal
from ruleforge.core.paths import absolute_strings, resolve

pytestmark = pytest.mark.frozen

PROGRAM = "ruleforge.experiments.training.build"
CAMPAIGNS = {
    "artifact://sampling/fig5-gemma-2026-10-07-pop-a22af097": (
        "fig5-gemma-2026-10-07", "61729f6e8671ee5804b904c23b59da3112e4e1f84c4c7b5b22b4592950fa973c",
        ["gemma-4-31B-it/figure5/base"]),
    "artifact://sampling/fig5-gemma-ckpt600-2026-10-09-pop-a22af097": (
        "fig5-gemma-ckpt600-2026-10-09", "63fd5434d7a16f2d1fb1d5cb96f405f9420fb2726274cd92d5bb5fc6532c1952",
        ["gemma-4-31B-it/figure5/final"]),
}
CHAIN = ["338d262861a49975516615130b7d79580d705095b6b348eb5163af073894668c",
         "8f74b7195f2724176f36dd15fd4579500688c6b76ddf3d635f4e265ec7d9d10a",
         "6f528dcba7feac97055e6f45c909d1e23943dcc15651b6ab4174332ccd44b9b3",
         "a22af097639cfdaa15b4508c66631af1cb6d45780d8ada93a9fb1d5a29390462"]
FLOOR_P = 1 / 100_001  # the randomization test's floor at 100,000 permutations
KAPPA_P = 2 / 10_001  # the paired bootstrap test's, two of 10,000 resamples
THREADS = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
#: An override of the kernel OpenBLAS picks for the CPU, which the rebuild drops.
CORETYPE = "OPENBLAS_CORETYPE"


@pytest.fixture(scope="module")
def payload(repo_root):
    return json.loads((repo_root / GEMMA_GOLDEN).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def driver(repo_root):
    module = importlib.import_module("ruleforge.experiments.training.build")
    return module


@pytest.fixture(scope="module")
def rerendered(repo_root, driver, tmp_path_factory):
    """The six panels re-rendered from the golden (``--from_golden``, ``SOURCE_DATE_EPOCH=0``), by committed name."""
    pytest.importorskip("matplotlib", reason="needs the viz group")
    out = tmp_path_factory.mktemp("gemma_rerendered")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SOURCE_DATE_EPOCH", "0")
        for suffix in ("pdf", "png"):
            figures = [word for panel in "a" for word in (
                f"--figure-{panel}", str(out / f"{GEMMA_FIGURES[panel].split('/')[-1]}.{suffix}"))]
            assert driver.main(["--from_golden", str(repo_root / GEMMA_GOLDEN), *figures]) == 0
    return out


def cpu_unlike_the_golden_s() -> Optional[str]:
    """Why a rebuild here would not reproduce the golden's bytes, or None."""
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    if not any(line.startswith("flags") and "avx512f" in line.split() for line in cpuinfo.splitlines()):
        return ("no avx512f in /proc/cpuinfo here: OpenBLAS runs another kernel than the golden's (SkylakeX), which "
                "moves a few bootstrap bounds by about 1e-16; run Tier R on an AVX-512 Intel CPU")
    if len(os.sched_getaffinity(0)) < 2:
        return ("one CPU: OpenBLAS runs single-threaded whatever its pin, and the golden is the multi-threaded value; "
                "run Tier R with two or more")
    return None


def test_the_golden_is_the_paper_s_family_and_nothing_else(payload):
    assert paper_refusal(payload) is None
    assert (payload["source"], payload["schema_version"], payload["metric"]) == ("training_family", 1, "flat")
    assert payload["policies"] == ["base", "final"] and payload["reference_policy"] == "base"
    family = payload["provenance"]["family"]
    assert family == {"declared_on": "the command line", "labels": ["base", "final"], "reference": "base",
                      "members": ["final"], "holm_family_size": 1}
    assert list(absolute_strings(payload)) == []


def test_the_two_arms_on_the_paper_population_with_five_draws(payload):
    base, final = payload["arms"]["base"], payload["arms"]["final"]
    assert (base["label"], final["label"]) == ("Gemma 4 31B-it", "GRPO (checkpoint-600)")
    assert (base["model_path"], base["model_weights_sha256"]) == ("gemma-4-31B-it", None)
    assert (final["model_path"], final["model_weights_sha256"]) == (
        "checkpoint-600", "03a8b16f1032b403a6f47cb8085fe294e838729ce06a84f73509c46914395be3")
    # flat inconsistency [95% scenario-cluster bootstrap CI]: 1,772 and 937 of 2,500 judgments
    assert [base["flat"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx([0.7088, 0.6732, 0.7436])
    assert [final["flat"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.3748, 0.3412, 0.4104], abs=5e-6)
    assert [round(arm["flat"]["estimate"] * 2500) for arm in (base, final)] == [1772, 937]
    # Cohen's kappa between the verdict and the per-rule max
    assert [base["kappa"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.03868, 0.02733, 0.05132], abs=5e-6)
    assert [final["kappa"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.12014, 0.08507, 0.15758], abs=5e-6)
    # the full-set yes share and the always-yes floor
    assert (base["yes_rate"]["estimate"], final["yes_rate"]["estimate"]) == pytest.approx((0.2196, 0.5824))
    assert (base["always_yes_floor"]["estimate"], final["always_yes_floor"]["estimate"]) == pytest.approx(
        (0.0724, 0.0660))
    # the M-stratified rates (M = the per-rule max): no M = unsure cell in either arm
    strata = {label: {stratum: (row["estimate"], row["n_draws"]) for stratum, row in arm["m_stratified"].items()}
              for label, arm in payload["arms"].items()}
    assert strata["base"]["M=yes"] == pytest.approx((0.76369, 2319), abs=5e-6)
    assert strata["base"]["M!=yes"] == pytest.approx((0.00552, 181), abs=5e-6)
    assert strata["final"]["M=yes"] == pytest.approx((0.38887, 2335), abs=5e-6)
    assert strata["final"]["M!=yes"] == pytest.approx((0.17576, 165), abs=5e-6)
    assert strata["base"]["M=unsure"] == strata["final"]["M=unsure"] == (None, 0)
    assert (base["same_call"]["n_complete_maps"], final["same_call"]["n_complete_maps"]) == (2307, 2500)
    assert base["first_attempt_acceptance_share"] == final["first_attempt_acceptance_share"] == 1.0


def test_one_test_per_holm_family_and_the_claim_the_rule_allows(payload):
    comparison = payload["comparisons"]["final"]
    flat, kappa = comparison["flat"], comparison["kappa"]
    assert [flat[key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.3340, 0.2968, 0.3708], abs=5e-6)
    assert flat["p_value"] == flat["p_holm"] == pytest.approx(FLOOR_P)
    assert [kappa[key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.08146, 0.05035, 0.11344], abs=5e-6)
    assert kappa["p_value"] == kappa["p_holm"] == pytest.approx(KAPPA_P)
    for row in (flat, kappa):
        assert row["family_size"] == 1 and row["significant_holm"] is True
        assert (row["policy_a"], row["policy_b"]) == ("base", "final")
    assert comparison["claim"] == "more_consistent"
    # the harm flag on the M != yes stratum: the flag reads the estimate (over 0.1) and the bound (over 0)
    assert comparison["harm_flag"] == {"flagged": True, "stratum": "M!=yes", "threshold": 0.1}
    difference = comparison["stratified_differences"]["differences"]["M!=yes"]
    assert [difference[key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.17023, 0.10021, 0.25000], abs=5e-6)
    assert 0 < difference["ci_low"] and comparison["harm_flag"]["threshold"] < difference["estimate"]


def test_the_yes_shift_baseline_explains_most_of_the_gain(payload):
    """Descriptive, no test: the base's verdicts turned yes blindly as often as checkpoint-600's yes share rose."""
    shift = payload["yes_shift"]["final"]
    assert shift["applicable"] is True and shift["descriptive"] is True
    assert (shift["yes_rate_reference"], shift["yes_rate_member"]) == pytest.approx((0.2196, 0.5824))
    assert shift["p"] == pytest.approx(0.46489, abs=5e-6)
    assert [shift["shifted_rate"][key] for key in ("estimate", "ci_low", "ci_high")] == pytest.approx(
        [0.41294, 0.39412, 0.43043], abs=5e-6)
    assert shift["shifted_rate"]["p_held_fixed"] is True
    assert shift["share_of_gain_explained"] == pytest.approx(0.88580, abs=5e-6)


def test_the_disclosure_says_where_the_arms_came_from_and_what_was_waived(payload):
    lines = payload["disclosure"]["lines"]
    text = "\n".join(lines)
    assert payload["disclosure"]["regime"] == "merged_fp32"
    for phrase in (
        "reference base (Gemma 4 31B-it), members final (GRPO (checkpoint-600))",
        "The reference arm base comes from artifact://sampling/fig5-gemma-2026-10-07-pop-a22af097; paired with it by "
        "(scenario, decoder draw index)",
        "of campaign fig5-gemma-2026-10-07's root from its population (scenario manifest 6f528dcba7fe...)",
        "one scenario manifest (a22af097639c...) of 500 scenarios",
        "over the members named here (final): a family of 1",
        "The yes-shift baseline beside each member is descriptive",
        "Parity B was not built for gemma-4-31B-it",
        "it is not the bf16 gemma-4-31B-it of the ruleset-size experiment",
    ):
        assert phrase in text, phrase


def test_the_provenance_pins_the_two_roots_and_the_merge(payload):
    provenance = payload["provenance"]
    assert provenance["complete"] is True and provenance["allow_code_mismatch"] is False
    campaigns = {campaign["root"]: campaign for campaign in provenance["campaigns"]}
    assert list(campaigns) == list(CAMPAIGNS)
    for root, (campaign_id, sha256, conditions) in CAMPAIGNS.items():
        assert (campaigns[root]["campaign_id"], campaigns[root]["estimates_sha256"], campaigns[root]["conditions"]) == (
            campaign_id, sha256, conditions)
        assert campaigns[root]["estimates"] == f"{root}/aggregate/estimates.json"
        assert (campaigns[root]["code_revision"], campaigns[root]["code_dirty"]) == (
            "57bfa3dc8b77fc39c6314dd9faed3c4f80961906", False)
        # the frozen cap and the component retry the training campaigns ran under, no retry ladder
        assert (campaigns[root]["max_semantic_attempts"], campaigns[root]["retry_scope"]) == (20, "component")
        chain = campaigns[root]["rebase"]["chain"]
        assert [hop["parent"]["scenario_manifest_sha256"] for hop in chain] + [chain[-1]["target"][
            "scenario_manifest_sha256"]] == CHAIN
        assert [hop["parent"]["campaign_id"] for hop in chain] == [campaign_id] * 3
    weights = provenance["base_weights"]
    assert weights["sha256"] == "e264a8bf89fc6630e7fffb89d60cf175a55d68b89c6255fee20439d72987b29c"
    assert weights["reference"]["checked"] == "directory_name"
    merge = weights["members"]["final"]
    assert merge["merge_manifest"] == GEMMA_FAMILY.merge_manifests["final"]
    # the merge manifest as the Hub publishes it
    assert merge["merge_manifest_sha256"] == "ca4ab806d959d1022cbfec16299ffbf8d98104e3a09776de363eae6d69f8b8de"
    assert provenance["scenario_manifest_sha256"] == CHAIN[-1]
    assert (provenance["n_scenarios"], provenance["n_decoder_draws"], provenance["temperature"]) == (500, 5, 1.0)
    assert provenance["bootstrap"] == {"ci_level": 0.95, "n_resamples": 10_000, "seed": 16}
    assert provenance["randomization_test"] == {"n_permutations": 100_000, "seed": 16}
    assert provenance["judge"]["prompt_style"] == "forced-grpo" and provenance["strict_think"] is False
    assert provenance["engine"]["tp_size"] == 2


def test_the_table_builds_the_paper_family_from_both_roots():
    population = f"pop-{CHAIN[-1][:8]}"
    step = next(step for row in table.TABLE for step in row.builds if step.golden and step.golden.path == GEMMA_GOLDEN)
    assert step.words == ["python", "-m", PROGRAM, "--family", "gemma", "--population", population, "--out-dir",
                          "{out}/gemma"]
    assert GEMMA_FAMILY.on_population(population).arguments() == GEMMA_FAMILY.arguments()
    assert table.GEMMA_RUNS == tuple(CAMPAIGNS)


def test_the_panels_re_render_from_the_golden_with_its_values(payload, rerendered):
    from ruleforge.experiments.training.figure import panel_values

    for suffix in ("pdf", "png"):
        assert (rerendered / f"{GEMMA_FIGURES['a'].split('/')[-1]}.{suffix}").stat().st_size > 0
    labels = ["Gemma 4 31B-it", "GRPO (checkpoint-600)"]
    values = panel_values(payload, "a")
    assert list(values) == labels
    assert [values[label]["estimate"] for label in labels] == [payload["arms"][policy]["flat"]["estimate"]
                                                              for policy in ("base", "final")]
    strata = {policy: payload["arms"][policy]["m_stratified"] for policy in ("base", "final")}
    assert {stratum: [strata[policy][stratum]["n_draws"] for policy in strata] for stratum in ("M=yes", "M!=yes")} == {
        "M=yes": [2319, 2335], "M!=yes": [181, 165]}


def test_panel_a_draws_the_two_bars_their_intervals_and_labels_and_no_marks(payload, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib", reason="needs the viz group")
    from ruleforge.experiments.training.figure import draw
    from ruleforge.experiments.training.paper_families import drawn_names
    from tests.support.training_campaign import assert_panel_a_draws_the_flat_rates_alone

    assert drawn_names(payload) == {"base": "Base Gemma", "final": "Trained Gemma"}
    assert_panel_a_draws_the_flat_rates_alone(payload, ["0.709", "0.375"],
                                              [("Base Gemma", 10.0), ("Trained Gemma", 10.0)], drawn_names(payload))

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    without = {key: value for key, value in payload.items() if key != "yes_shift"}
    without["arms"] = {policy: {key: value for key, value in arm.items() if key != "always_yes_floor"}
                       for policy, arm in payload["arms"].items()}
    for panel in "a":
        for suffix in ("pdf", "png"):
            draw(payload, tmp_path / f"with_{panel}.{suffix}", panel)
            draw(without, tmp_path / f"without_{panel}.{suffix}", panel)
            assert (tmp_path / f"with_{panel}.{suffix}").read_bytes() == (
                tmp_path / f"without_{panel}.{suffix}").read_bytes(), (panel, suffix)


@pytest.fixture(scope="module")
def rebuilt(repo_root, tmp_path_factory):
    """The paper's readout rebuilt from ``$RULEFORGE_ARTIFACT_ROOT`` as the table's command builds it."""
    pytest.importorskip("matplotlib", reason="needs the viz group: the build draws the panels")
    if not os.environ.get("RULEFORGE_ARTIFACT_ROOT"):
        pytest.skip("RULEFORGE_ARTIFACT_ROOT is not set")
    estimates = {root: resolve(f"{root}/aggregate/estimates.json") for root in CAMPAIGNS}
    if not all(path.is_file() for path in estimates.values()):
        pytest.skip("the training roots are not in this artifact root")
    if (reason := cpu_unlike_the_golden_s()) is not None:
        pytest.skip(reason)
    for root, path in estimates.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == CAMPAIGNS[root][1], root
    out = tmp_path_factory.mktemp("gemma_rebuilt")
    env = {**{key: value for key, value in os.environ.items() if key != CORETYPE}, **THREADS, "SOURCE_DATE_EPOCH": "0"}
    built = subprocess.run([sys.executable, "-m", PROGRAM, *GEMMA_FAMILY.arguments(), "--out-dir",
                            str(out)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr[-2000:]
    return out


@pytest.mark.cluster
def test_the_golden_rebuilds_byte_for_byte_from_the_paper_root(repo_root, rebuilt):
    assert (rebuilt / "training_family.json").read_bytes() == (repo_root / GEMMA_GOLDEN).read_bytes()
    for panel in "a":
        for suffix in ("pdf", "png"):
            assert (rebuilt / f"training_{panel}.{suffix}").stat().st_size > 0
