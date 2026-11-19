from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ruleforge.experiments.training import family_readout as family_module
from ruleforge.evaluate.estimates import CampaignEstimatesError
from ruleforge.experiments.training.campaign_readout import TrainingReadoutError, build_from_campaign
from ruleforge.experiments.training.figure import TICK_FONT_SIZES, build_figure, drawn_labels, panel_values
from ruleforge.experiments.training.paper_families import drawn_names
from ruleforge.experiments.training.family_readout import (
    YES_SHIFT_DEFINITION,
    FamilyArm,
    TrainingFamilyError,
    build_readout,
    identity_differences,
    summary_markdown,
    yes_shift,
)
from ruleforge.evaluate.inference import DrawGrid
from ruleforge.grpo import merging as merge_module
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.campaign import condition_payload, load_conditions
from ruleforge.core.digests import sha256_file
from ruleforge.evaluate.discrimination import VerdictGrids
from ruleforge.evaluate.runner import CampaignRunError
from tests.support.training_campaign import (
    GEMMA,
    POLICIES,
    _edit,
    _roots as _roots,  # an autouse fixture: imported, it applies here too
    assert_panel_a_draws_the_flat_rates_alone,
    mark_final,
    run_training_campaign,
)
from tests.test_sampling_campaign import set_recorded_runtime
from tests.test_prompt_ablation_arms import UNPRIVILEGED, read_only

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Past the bootstrap's first chunk (``repeated_sampling._CHUNK``, 500 replicates).
N_RESAMPLES = 600
N_PERMUTATIONS = 2000
PREFIX = GEMMA["prefix"]
LABELS = {"base": "Gemma 4 31B-it", "grpo": "GRPO (checkpoint-200)", "final": "GRPO (checkpoint-600)"}
#: The code state a launch from a clean checkout records (``provenance.code_state``), as both real roots do.
CLEAN_CODE = {"code_revision": "2393b36a965a679eefd74298251d623c986a38cf", "code_dirty": False,
              "code_state_sha256": "f6623e5d61df91ad5cbdfa764bc49934aff55e53dd3d7ef7b38c67e5dc75ca02"}


def campaign(directory: Path, modelspec, policies, **kwargs) -> Path:
    """A synthetic training root of ``policies`` (``run_training_campaign``) in ``directory``, on Gemma's engine."""
    directory.mkdir()
    return run_training_campaign(directory, modelspec, policies, judge=GEMMA, campaign_id=f"training-{directory.name}",
                                **kwargs)


def analysed(root: Path, **flags) -> Path:
    """``root`` analysed, final, and recording a launch from a clean checkout (the synthetic runs record no code)."""
    analyze_campaign(root, n_resamples=N_RESAMPLES, n_permutations=N_PERMUTATIONS,
                     rule_set_path=root.parent / "rules.jsonl", **flags)
    mark_final(root)
    _edit(root / "sampling_manifest.json", lambda manifest: manifest.update(CLEAN_CODE))
    return root


def uri(root: Path, tmp_path: Path) -> str:
    return "artifact://" + root.relative_to(tmp_path).as_posix()


def arms_in(tmp_path: Path, roots: dict[str, Path]) -> list[FamilyArm]:
    """``base``, ``grpo``, ``final`` in the family's order, each from the root ``roots`` names for it."""
    return [FamilyArm(label, uri(roots[label], tmp_path), f"{PREFIX}/{label}") for label in POLICIES]


@pytest.fixture
def two_roots(tmp_path, modelspec):
    """Root A (base and grpo, declaring base+grpo) and root B (final alone, declaring base+final), analysed."""
    a = analysed(campaign(tmp_path / "A", modelspec, ("base", "grpo")))
    b = campaign(tmp_path / "B", modelspec, ("final",), family="base+final")
    return {"base": a, "grpo": a, "final": analysed(b, allow_incomplete_conditions=True)}


@pytest.fixture
def one_root(tmp_path, modelspec):
    root = analysed(campaign(tmp_path / "one", modelspec, POLICIES))
    return {label: root for label in POLICIES}


def read(tmp_path: Path, roots: dict[str, Path], arms=None, **kwargs):
    arms = arms or arms_in(tmp_path, roots)
    kwargs.setdefault("reference", "base")
    kwargs.setdefault("labels", {arm.label: LABELS[arm.label] for arm in arms if arm.label in LABELS})
    return build_readout(arms, rule_set_path=roots["base"].parent / "rules.jsonl", **kwargs)


def estimates_of(root: Path) -> Path:
    return root / "aggregate" / "estimates.json"


def test_a_root_holding_a_member_without_its_reference_is_analysed_with_the_reference_reported_missing(
        tmp_path, modelspec):
    root = campaign(tmp_path / "B", modelspec, ("final",), family="base+final")
    with pytest.raises(CampaignRunError, match=r"missing declared policies .*\['base'\]"):
        analyze_campaign(root, n_resamples=200, n_permutations=200, rule_set_path=tmp_path / "B" / "rules.jsonl")
    estimates = analyze_campaign(root, n_resamples=200, n_permutations=200, allow_incomplete_conditions=True,
                                 rule_set_path=tmp_path / "B" / "rules.jsonl")
    (block,) = estimates["policy_families"].values()
    assert block["missing_policies"] == ["base"] and block["contrasts"] == [] and list(block["arms"]) == ["final"]
    assert block["arms"]["final"]["method_b"]["n_outputs"] == 24
    assert estimates["state"] == "complete" and estimates["excluded_conditions"] == {}


def test_the_family_across_two_roots_is_the_one_root_family_number_for_number(tmp_path, one_root, two_roots):
    producer = build_from_campaign(estimates_of(one_root["base"]), parity={}, labels=LABELS)
    single = read(tmp_path, one_root)
    split = read(tmp_path, two_roots)
    for key in ("arms", "comparisons", "strata", "policies", "reference_policy"):
        assert single[key] == producer[key], key
        assert split[key] == single[key], key
    assert split["yes_shift"] == single["yes_shift"]
    two_arm = json.loads(estimates_of(two_roots["base"]).read_text())["conditions"][f"{PREFIX}/base"]
    assert split["arms"]["base"]["flat"]["estimate"] == two_arm["estimate"]
    assert split["arms"]["base"]["flat"]["ci_low"] != two_arm["ci_low"]
    comparisons = split["comparisons"]
    assert {label: (row["flat"]["family_size"], row["kappa"]["family_size"]) for label, row in comparisons.items()} == {
        "grpo": (2, 2), "final": (2, 2)}
    assert (comparisons["grpo"]["claim"], comparisons["final"]["claim"]) == ("more_consistent", "yes_shift")
    assert comparisons["final"]["harm_flag"]["flagged"] and not comparisons["grpo"]["harm_flag"]["flagged"]
    assert split["arms"]["final"]["flat"]["estimate"] == pytest.approx(3 / 24)
    core = producer["disclosure"]["lines"]
    assert single["disclosure"]["lines"][-len(core):] == core == split["disclosure"]["lines"][-len(core):]
    assert [campaign["root"] for campaign in split["provenance"]["campaigns"]] == ["artifact://A/root",
                                                                                    "artifact://B/root"]
    assert [campaign["conditions"] for campaign in split["provenance"]["campaigns"]] == [
        [f"{PREFIX}/base", f"{PREFIX}/grpo"], [f"{PREFIX}/final"]]


@UNPRIVILEGED
def test_read_only_roots_give_the_same_readout_byte_for_byte(tmp_path, two_roots):
    writable = json.dumps(read(tmp_path, two_roots), sort_keys=True)
    with read_only(tmp_path / "A"), read_only(tmp_path / "B"):
        assert json.dumps(read(tmp_path, two_roots), sort_keys=True) == writable


def test_the_reference_and_one_member_of_one_root_read_as_the_two_arm_producer_reads_them(tmp_path, two_roots):
    labels = {label: LABELS[label] for label in ("base", "grpo")}
    producer = build_from_campaign(estimates_of(two_roots["base"]), parity={}, labels=labels)
    readout = read(tmp_path, two_roots, arms=arms_in(tmp_path, two_roots)[:2])
    for key in ("arms", "comparisons", "strata", "policies"):
        assert readout[key] == producer[key], key
    assert readout["comparisons"]["grpo"]["flat"]["family_size"] == 1
    assert readout["provenance"]["family"]["holm_family_size"] == 1
    text = "\n".join(readout["disclosure"]["lines"])
    assert "from another root" not in text
    assert "joint scenario-cluster bootstrap" not in text


def test_an_arm_left_out_of_the_family_is_outside_its_holm_family(tmp_path, two_roots):
    arms = arms_in(tmp_path, two_roots)
    readout = read(tmp_path, two_roots, arms=[arms[0], arms[2]])
    text = "\n".join(readout["disclosure"]["lines"])
    assert "over the members named here (final): a family of 1" in text
    assert readout["comparisons"]["final"]["flat"]["family_size"] == 1 and list(readout["comparisons"]) == ["final"]


def test_the_disclosure_and_the_record_say_where_each_arm_came_from(tmp_path, two_roots):
    readout = read(tmp_path, two_roots)
    text = "\n".join(readout["disclosure"]["lines"])
    for phrase in (
        "The family is named on the command line: reference base (Gemma 4 31B-it), members grpo (GRPO "
        "(checkpoint-200)), final (GRPO (checkpoint-600))",
        "artifact://A/root (campaign training-A, commit 2393b36a965a, sources f6623e5d61df: base, grpo); artifact://B/root "
        "(campaign training-B, commit 2393b36a965a, sources f6623e5d61df: final)",
        "label_only validity with strict_think off", "bootstrap 600 at seed 16, randomization 2,000 permutations",
        "named by that directory alone", "this readout checks the name, not that the reference's files are those",
        "The reference arm base comes from artifact://A/root; paired with it by (scenario, decoder draw index) over the "
        "one scenario manifest, from another root: final.",
        "over the members named here (grpo, final): a family of 2",
        "Each arm's flat-rate interval is the joint scenario-cluster bootstrap over the arms named here",
        "(artifact://A/root: gemma-4-31B-it/figure5/base, gemma-4-31B-it/figure5/grpo; artifact://B/root: "
        "gemma-4-31B-it/figure5/final). The point estimates, and each contrast's estimate, interval and unadjusted p, "
        "do not depend on it.",
        "descriptive: no test, in no family", "Parity B was not built for gemma-4-31B-it",
    ):
        assert phrase in text, phrase
    provenance = readout["provenance"]
    assert provenance["complete"] is True and provenance["allow_code_mismatch"] is False
    assert provenance["family"] == {
        "declared_on": "the command line", "labels": list(POLICIES), "reference": "base", "members": ["grpo", "final"],
        "holm_family_size": 2}
    assert provenance["arms"]["final"]["declared"] == {"policy": "final", "policy_family": "base+final",
                                                       "policy_reference": "base"}
    assert provenance["arms"]["base"]["model_weights_sha256"] is None
    assert provenance["arms"]["final"]["model_path"] == "final"
    assert [{key: campaign[key] for key in CLEAN_CODE} for campaign in provenance["campaigns"]] == [CLEAN_CODE] * 2
    base = merge_module.weights_digest(tmp_path / "A" / "models" / "gemma-4-31B-it")
    weights = provenance["base_weights"]
    assert weights["sha256"] == base and weights["reference"]["checked"] == "directory_name"
    assert {label: merge["base_directory"] for label, merge in weights["members"].items()} == {
        "grpo": "gemma-4-31B-it", "final": "gemma-4-31B-it"}
    assert weights["members"]["final"]["merge_manifest"] == "artifact://B/grpo/merged/final/merge_manifest.json"
    assert provenance["engine"]["tp_size"] is None and provenance["engine"]["short"].startswith("vLLM 0.28.0+cu129")
    assert provenance["rule_set"]["name"] == "rules.jsonl" and provenance["judge"]["model_name"] == "gemma-4-31B-it"
    assert [campaign["campaign_id"] for campaign in provenance["campaigns"]] == ["training-A", "training-B"]


def test_the_yes_shift_baseline_of_each_member_is_the_hand_computation(tmp_path, two_roots):
    """base never says yes and its per-rule max is yes in 22 of its 24 cells (inconsistent in exactly those)."""
    shifts = read(tmp_path, two_roots)["yes_shift"]
    grpo, final = shifts["grpo"], shifts["final"]
    assert (grpo["yes_rate_reference"], grpo["yes_rate_member"], grpo["p"]) == pytest.approx((0.0, 21 / 24, 21 / 24))
    p = 21 / 24
    assert grpo["shifted_rate"]["estimate"] == pytest.approx((22 * (1 - p) + 2 * p) / 24)
    assert grpo["share_of_gain_explained"] == pytest.approx((22 / 24 - (22 * (1 - p) + 2 * p) / 24) / (22 / 24))
    assert (final["p"], final["shifted_rate"]["estimate"], final["share_of_gain_explained"]) == pytest.approx(
        (1.0, 2 / 24, (20 / 24) / (19 / 24)))
    assert final["shifted_rate"]["p_held_fixed"] is True and final["shifted_rate"]["ci_low"] <= 2 / 24
    assert all(shift["descriptive"] and shift["applicable"] for shift in shifts.values())


def verdicts(cid: str, full, rule_max) -> VerdictGrids:
    ids = [f"s{i}" for i in range(len(full))]
    return VerdictGrids(DrawGrid(cid, "full_set_phi", ids, np.array(full, dtype=float)),
                        DrawGrid(cid, "max_rule_phi", ids, np.array(rule_max, dtype=float)))


REFERENCE = ([[1, 0], [0.5, 0], [0, 0]], [[1, 1], [1, 1], [0, 0.5]])
MEMBER = ([[1, 1], [1, 1], [0, 0]], [[1, 1], [1, 0], [0, 0]])


def test_the_yes_shift_keeps_the_reference_s_yes_cells_and_moves_its_others_by_p():
    shift = yes_shift(verdicts("ref", *REFERENCE), verdicts("mem", *MEMBER), n_resamples=200, ci_level=0.95, seed=16)
    p = (4 / 6 - 1 / 6) / (1 - 1 / 6)
    expected = (0 + 3 * (1 - p) + p + 1) / 6
    assert (shift["p"], shift["shifted_rate"]["estimate"]) == pytest.approx((0.6, expected))
    assert (shift["rate_reference"], shift["rate_member"]) == pytest.approx((4 / 6, 1 / 6))
    assert shift["share_of_gain_explained"] == pytest.approx((4 / 6 - expected) / (4 / 6 - 1 / 6))
    assert shift["shifted_rate"]["ci_low"] <= expected <= shift["shifted_rate"]["ci_high"]


def test_the_yes_shift_is_not_applicable_unless_the_member_says_yes_more_and_is_less_inconsistent():
    backwards = yes_shift(verdicts("mem", *MEMBER), verdicts("ref", *REFERENCE), n_resamples=200, ci_level=0.95,
                          seed=16)
    assert backwards["applicable"] is False and (backwards["p"], backwards["shifted_rate"]) == (None, None)
    assert backwards["reason"] == ("the member's full-set yes share is not above the reference's; the member is not "
                                   "less inconsistent than the reference")
    worse = verdicts("worse", [[1, 1], [1, 1], [1, 1]], [[0, 0], [0, 0], [0, 0]])
    shifted = yes_shift(verdicts("ref", *REFERENCE), worse, n_resamples=200, ci_level=0.95, seed=16)
    assert shifted["reason"] == "the member is not less inconsistent than the reference"
    assert shifted["share_of_gain_explained"] is None and shifted["yes_rate_member"] == 1.0


def rewrite_condition(root: Path, condition_id: str, **changes) -> None:
    """Re-register a condition with ``changes`` (its fingerprint recomputed), as if it had been launched so."""
    spec = dataclasses.replace(load_conditions(root)[condition_id], **changes)
    (root / condition_id / "condition.json").write_text(json.dumps(condition_payload(spec)))


def edit_provenance(condition_dir: Path, change, draws=("draw-000", "draw-001")) -> None:
    for draw in draws:
        _edit(condition_dir / draw / "draw_provenance.json", change)


def rewrite_merge(condition_dir: Path, change) -> Path:
    """Change a member's merge manifest and, as if it had been merged so, what its draws recorded of it."""
    record = json.loads((condition_dir / "draw-000" / "draw_provenance.json").read_text())["model_weights"]
    path = Path(record["path"]) / merge_module.MANIFEST_FNAME
    _edit(path, change)
    digest = sha256_file(path)
    edit_provenance(condition_dir, lambda p: p["model_weights"].update(manifest_sha256=digest))
    return path


def _grpo_weights(roots):
    return load_conditions(roots["grpo"])[f"{PREFIX}/grpo"].model_weights_sha256


FINAL = f"{PREFIX}/final"


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda roots: _edit(estimates_of(roots["final"]), lambda e: e.update(scenario_manifest_sha256="0" * 64)),
         CampaignEstimatesError, "scenario_manifest_sha256 .* these roots are not one experiment"),
        (lambda roots: [_edit(path, lambda m: m.update(max_semantic_attempts=40)) for path in (
            roots["final"] / "sampling_manifest.json", estimates_of(roots["final"]))],
         TrainingFamilyError, r"designs differ in max_semantic_attempts \(artifact://A/root 5, artifact://B/root 40\)"),
        (lambda roots: _edit(estimates_of(roots["final"]), lambda e: e["bootstrap"].update(seed=17)),
         CampaignEstimatesError, "bootstrap .* differs"),
        (lambda roots: rewrite_condition(roots["final"], FINAL, label_rule="bare"),
         TrainingFamilyError, r"arm final differs from the reference base in \['label_rule'\]"),
        (lambda roots: rewrite_condition(roots["final"], FINAL, rule_set_sha256="0" * 64),
         TrainingFamilyError, r"in \['rule_set_sha256'\]"),
        (lambda roots: rewrite_condition(roots["final"], FINAL, max_new_tokens=2048),
         TrainingFamilyError, r"in \['max_new_tokens'\]"),
        (lambda roots: set_recorded_runtime(roots["final"] / FINAL, transformers_version="5.5.0"),
         TrainingFamilyError, r"different inference stacks or serving settings \(base vs final: .*transformers 5.5.0"),
        (lambda roots: edit_provenance(roots["final"] / FINAL, lambda p: p.update(tp_size=4), ("draw-001",)),
         TrainingFamilyError, r"tensor-parallel sizes .*'final': \[4, None\]"),
        (lambda roots: _edit(estimates_of(roots["final"]), lambda e: e.update(state="in_progress")),
         CampaignEstimatesError, "needs a complete campaign"),
        (lambda roots: _edit(roots["final"] / "sampling_manifest.json", lambda m: m.update(campaign_mode="pilot")),
         CampaignEstimatesError, "campaign_mode 'pilot'"),
        (lambda roots: rewrite_condition(roots["final"], FINAL, model_weights_sha256=_grpo_weights(roots)),
         TrainingFamilyError, r"arms \['final', 'grpo'\] serve the same weights"),
        (lambda roots: rewrite_merge(roots["final"] / FINAL, lambda m: m["base"].update(weights_sha256="f" * 64)),
         TrainingFamilyError, "the members' merges record different base weights"),
        (lambda roots: _edit(roots["final"].parent / "grpo" / "merged" / "final" / "merge_manifest.json",
                             lambda m: m.update(note="edited")),
         TrainingFamilyError, "is not the merge manifest its draws recorded"),
        (lambda roots: _edit(roots["final"] / "sampling_manifest.json", lambda m: m.update(code_revision="abc123")),
         TrainingFamilyError, "the roots ran different code .*artifact://B/root commit abc123, sources f6623e5d61df"),
        (lambda roots: _edit(roots["final"] / "sampling_manifest.json",
                             lambda m: m.update(code_dirty=True, code_state_sha256="b" * 64)),
         TrainingFamilyError, r"the roots ran different code; roots \['artifact://B/root'\] record no clean commit .*"
                             r"artifact://B/root commit 2393b36a965a \(dirty\), sources bbbbbbbbbbbb\); pass "
                             "allow_code_mismatch"),
        (lambda roots: _edit(roots["final"] / "sampling_manifest.json", lambda m: m.update(code_state_sha256="b" * 64)),
         TrainingFamilyError, "the roots ran different code .*sources bbbbbbbbbbbb"),
        (lambda roots: [_edit(root / "sampling_manifest.json", lambda m: m.update(code_revision=None))
                        for root in (roots["base"], roots["final"])],
         TrainingFamilyError, r"^training family: roots \['artifact://A/root', 'artifact://B/root'\] record no clean "
                             "commit"),
    ],
)
def test_the_readout_refuses_arms_that_are_not_one_family(tmp_path, two_roots, change, error, message):
    change(two_roots)
    with pytest.raises(error, match=message):
        read(tmp_path, two_roots)


def test_different_code_is_read_only_when_allowed_and_then_recorded(tmp_path, two_roots):
    _edit(two_roots["final"] / "sampling_manifest.json", lambda m: m.update(code_dirty=True, code_state_sha256="b" * 64))
    readout = read(tmp_path, two_roots, allow_code_mismatch=True)
    assert readout["provenance"]["allow_code_mismatch"] is True
    assert [(campaign["code_dirty"], campaign["code_state_sha256"]) for campaign in readout["provenance"]["campaigns"]] == [
        (False, CLEAN_CODE["code_state_sha256"]), (True, "b" * 64)]
    assert "artifact://B/root (campaign training-B, commit 2393b36a965a (dirty), sources bbbbbbbbbbbb: final)" in "\n".join(
        readout["disclosure"]["lines"])


def test_two_arms_of_one_display_label_are_refused(tmp_path, two_roots):
    with pytest.raises(TrainingReadoutError, match=r"display labels \['GRPO'\] name more than one arm"):
        read(tmp_path, two_roots, labels={"grpo": "GRPO", "final": "GRPO"})


def test_the_yes_shift_is_read_against_each_arm_s_stored_contradiction(tmp_path, two_roots, monkeypatch):
    class Shifted(VerdictGrids):
        @classmethod
        def from_cells(cls, condition_id, *args, **kwargs):
            grids = VerdictGrids.from_cells(condition_id, *args, **kwargs)
            if condition_id.endswith("/final"):
                grids.full.values[0, 0] = 1.0 - grids.full.values[0, 0]
            return grids

    monkeypatch.setattr(family_module, "VerdictGrids", Shifted)
    with pytest.raises(TrainingFamilyError, match="arm final's stored contradiction is not phi"):
        read(tmp_path, two_roots)


@pytest.mark.parametrize(
    ("arms", "kwargs", "message"),
    [
        ([("base", "A", "base")], {}, "a family is its reference and at least one member"),
        ([("base", "A", "base"), ("grpo", "A", "grpo"), ("grpo", "B", "final")], {},
         r"labels \['grpo'\] name more than one arm"),
        ([("base", "A", "base"), ("grpo", "A", "grpo"), ("final", "A", "grpo")], {}, "named more than once"),
        ([("base", "A", "base"), ("grpo", "A", "grpo")], {"reference": "nobody"},
         r"the reference 'nobody' is not one of the family's labels \['base', 'grpo'\]"),
        ([("grpo", "A", "grpo"), ("base", "A", "base")], {},
         r"the reference 'base' is not the first arm named \(\['grpo', 'base'\]\)"),
        ([("base", "/abs/root", "base"), ("grpo", "A", "grpo")], {}, r"roots \['/abs/root'\] are not artifact://"),
        ([("base", "A", "base"), ("grpo", "A", "grpo")], {"labels": {"final": "x"}},
         r"display labels for \['final'\], which are not arms"),
        ([("base", "A", "base"), ("grpo", "A", "grpo")], {"merge_manifests": {"base": "artifact://m.json"}},
         r"merge manifests for \['base'\], which are not members"),
        ([("base", "A", "base"), ("grpo", "A", "grpo")], {"merge_manifests": {"grpo": "/abs/merge_manifest.json"}},
         r"merge manifests \['/abs/merge_manifest.json'\] are not artifact:// URIs"),
    ],
)
def test_the_command_line_must_name_a_family(arms, kwargs, message):
    named = [FamilyArm(label, root if root.startswith("/") else f"artifact://{root}/root", f"{PREFIX}/{cid}")
             for label, root, cid in arms]
    with pytest.raises(TrainingFamilyError, match=message):
        build_readout(named, reference=kwargs.pop("reference", "base"), rule_set_path="rules.jsonl", **kwargs)


def test_each_arm_is_the_policy_it_declared_with_the_reference_it_declared(tmp_path, two_roots):
    arms = arms_in(tmp_path, two_roots)
    with pytest.raises(TrainingFamilyError, match="arm ckpt600 .* declared policy 'final', not 'ckpt600'"):
        read(tmp_path, two_roots, arms=[*arms[:2], dataclasses.replace(arms[2], label="ckpt600")])
    with pytest.raises(TrainingFamilyError, match="arm base declared the reference 'base', not 'grpo'"):
        read(tmp_path, two_roots, arms=[arms[1], arms[0], arms[2]], reference="grpo")


def test_identity_differences_name_extra_keys_and_ignore_the_policy_fields(two_roots):
    specs = {**load_conditions(two_roots["base"]), **load_conditions(two_roots["final"])}
    base, final = specs[f"{PREFIX}/base"], specs[FINAL]
    assert identity_differences(base, final) == []
    other = dataclasses.replace(final, extra=tuple({**dict(final.extra), "figure": "figure4"}.items()))
    assert identity_differences(base, other) == ["extra.figure"]


def weights_inputs(two_roots):
    specs = {label: load_conditions(root)[f"{PREFIX}/{label}"] for label, root in two_roots.items()}
    dirs = {label: root / PREFIX / label for label, root in two_roots.items()}
    return specs, dirs


def check_weights(specs, dirs):
    records = {label: family_module.draw_provenances(directory, 2) for label, directory in dirs.items()}
    return family_module._check_weights(specs, records, "base", "family")


def test_the_reference_is_the_members_base_by_content_where_its_identity_names_it_so(two_roots):
    specs, dirs = weights_inputs(two_roots)
    base = check_weights(specs, dirs)["sha256"]
    by_content = {**specs, "base": dataclasses.replace(specs["base"], model_weights_sha256=base)}
    assert check_weights(by_content, dirs)["reference"]["checked"] == "content"
    with pytest.raises(TrainingFamilyError, match="the reference base serves weights 0+, not the base"):
        check_weights({**specs, "base": dataclasses.replace(specs["base"], model_weights_sha256="0" * 64)}, dirs)
    with pytest.raises(TrainingFamilyError, match=r"name the base directory \['gemma-4-31B-it'\], not the reference "
                                                  "base's 'gemma-4-31B-it-other'"):
        check_weights({**specs, "base": dataclasses.replace(specs["base"], model_path="gemma-4-31B-it-other")}, dirs)
    record = check_weights(by_content, dirs)
    assert family_module._base_line("base", ["grpo", "final"], record).endswith(
        "; the reference base serves those weights, named by content.")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda specs, dirs: specs.update(final=dataclasses.replace(
            specs["final"], model_weights_sha256=None, adapter_path="checkpoint-600/adapter_model.safetensors",
            adapter_sha256="a" * 64, adapter_config_sha256="c" * 64)),
         "member final .* serves no merged checkpoint named by content"),
        (lambda specs, dirs: edit_provenance(dirs["final"], lambda p: p["model_weights"].update(sha256="0" * 64),
                                             ("draw-001",)),
         "its draws record served weights .* not one merged checkpoint"),
        (lambda specs, dirs: edit_provenance(dirs["final"], lambda p: p.pop("model_weights")),
         "its draws record served weights"),
        (lambda specs, dirs: edit_provenance(dirs["final"], lambda p: p["model_weights"].update(path="/nowhere")),
         "its merge manifest /nowhere/merge_manifest.json is not readable here"),
        (lambda specs, dirs: rewrite_merge(dirs["final"], lambda m: m["result"].update(weights_sha256="0" * 64)),
         "records weights 0+, not the arm's"),
        (lambda specs, dirs: rewrite_merge(dirs["final"], lambda m: m.pop("base")), "records no base weights digest"),
        (lambda specs, dirs: rewrite_merge(dirs["final"], lambda m: m["base"].pop("path")),
         "records no base weights digest and directory"),
    ],
)
def test_a_member_must_be_a_merged_checkpoint_whose_recorded_merge_names_its_base(two_roots, change, message):
    specs, dirs = weights_inputs(two_roots)
    change(specs, dirs)
    with pytest.raises(TrainingFamilyError, match=message):
        check_weights(specs, dirs)


def test_a_merge_manifest_named_on_the_command_line_is_read_there_and_held_to_the_draws_digest(tmp_path, two_roots):
    """A root copied into another artifact root still records the weights directory of the root it ran under."""
    recorded = two_roots["final"].parent / "grpo" / "merged" / "final" / "merge_manifest.json"
    elsewhere = tmp_path / "copied" / "final" / "merge_manifest.json"
    elsewhere.parent.mkdir(parents=True)
    recorded.rename(elsewhere)
    with pytest.raises(TrainingFamilyError, match="its merge manifest .*merge_manifest.json is not readable here"):
        read(tmp_path, two_roots)
    readout = read(tmp_path, two_roots, merge_manifests={"final": "artifact://copied/final/merge_manifest.json"})
    merge = readout["provenance"]["base_weights"]["members"]["final"]
    assert merge["merge_manifest"] == "artifact://copied/final/merge_manifest.json"
    elsewhere.rename(recorded)
    merge["merge_manifest"] = "artifact://B/grpo/merged/final/merge_manifest.json"
    assert readout == read(tmp_path, two_roots, merge_manifests={})
    elsewhere.write_text(recorded.read_text().replace('"base"', '"base" ', 1))
    with pytest.raises(TrainingFamilyError, match="is not the merge manifest its draws recorded"):
        read(tmp_path, two_roots, merge_manifests={"final": "artifact://copied/final/merge_manifest.json"})
    for uri, message in (("artifact://copied/absent.json", "absent.json is not readable here"),
                         ("artifact://../outside.json", "its merge manifest artifact://../outside.json does not "
                                                        "resolve: .*escapes the artifact root")):
        with pytest.raises(TrainingFamilyError, match=f"member final .*{message}"):
            read(tmp_path, two_roots, merge_manifests={"final": uri})


def test_the_summary_is_a_function_of_the_readout(tmp_path, two_roots):
    readout = json.loads(json.dumps(read(tmp_path, two_roots)))
    text = summary_markdown(readout)
    assert text.startswith("# Training readout over a named policy family\n")
    assert "Reference Gemma 4 31B-it; members GRPO (checkpoint-200) (grpo), GRPO (checkpoint-600) (final)" in text
    final = readout["provenance"]["arms"]["final"]["model_weights_sha256"]
    assert f"| GRPO (checkpoint-600) | gemma-4-31B-it/figure5/final | final ({final[:12]}...) | 0.1250 [" in text
    assert "| Gemma 4 31B-it | gemma-4-31B-it/figure5/base | gemma-4-31B-it (by directory) | " in text
    assert "| 22/24 |" in text and YES_SHIFT_DEFINITION in text
    assert "| FLAGGED |\n" in text and "| no |\n" in text
    contrasts = text.split("## Contrasts")[1].split("## Yes-shift")[0].splitlines()
    claims = {label: [cell.strip() for cell in line.split("|")][8] for label in readout["provenance"]["family"]["members"]
              for line in contrasts if line.startswith(f"| {readout['arms'][label]['label']} |")}
    assert claims == {label: row["claim"] for label, row in readout["comparisons"].items()}
    assert sorted(claims.values()) == ["more_consistent", "yes_shift"]
    assert "| 0.0000 -> 0.8750 | 0.8750 | 0.1875 [" in text
    assert "## Disclosure (merged_fp32)" in text
    readout["arms"]["grpo"]["same_call"] = None
    readout["arms"]["final"]["kappa"] = {"estimate": None, "ci_low": None, "ci_high": None}
    readout["yes_shift"]["final"] = {**readout["yes_shift"]["final"], "applicable": False, "reason": "why"}
    text = summary_markdown(readout)
    assert "| - | 0.9583 |" in text and "| undefined |" in text and "| - | not applicable: why |" in text


def test_long_display_labels_wrap_only_where_one_line_would_overlap(tmp_path, two_roots):
    import matplotlib.pyplot as plt

    readout = read(tmp_path, two_roots)
    committed = json.loads((REPO_ROOT / "tests" / "fixtures" / "training" / "synthetic_readout.json").read_text())
    assert drawn_names(readout) == {} and drawn_names(committed) == {}
    for panel in "a":
        wrapped, kept = build_figure(readout, panel), build_figure(committed, panel)
        assert [text.get_text() for text in wrapped.axes[0].get_xticklabels()] == [
            "Gemma 4\n31B-it", "GRPO\n(checkpoint-200)", "GRPO\n(checkpoint-600)"]
        assert [text.get_text() for text in kept.axes[0].get_xticklabels()] == ["Qwen3-8B", "GRPO", "GRPO (final)"]
        plt.close(wrapped)
        plt.close(kept)


def test_a_label_is_broken_at_its_middle_space_unless_it_is_already_broken():
    from ruleforge.experiments.training.figure import _two_lines

    assert _two_lines("Base Gemma") == "Base\nGemma" and _two_lines("Gemma") == "Gemma"
    assert _two_lines("Trained Gemma\n(own rule judgments)") == "Trained Gemma\n(own rule judgments)"


def test_names_a_caller_passes_are_drawn_for_their_policies_and_two_arms_drawn_alike_are_refused():
    import matplotlib.pyplot as plt

    golden = json.loads((REPO_ROOT / "tests" / "fixtures" / "training" / "synthetic_readout.json").read_text())
    names = {"base": "Base Qwen", "final": "Trained Qwen"}
    assert drawn_labels(golden, names) == {"Qwen3-8B": "Base Qwen", "GRPO": "GRPO",
                                           "GRPO (final)": "Trained Qwen"}
    figure = build_figure(golden, "a", names)
    assert [text.get_text() for text in figure.axes[0].get_xticklabels()] == ["Base Qwen", "GRPO", "Trained Qwen"]
    plt.close(figure)
    with pytest.raises(ValueError, match=r"drawn labels \['GRPO'\] name more than one arm"):
        build_figure(golden, "a", {"final": "GRPO"})


def test_a_family_that_is_not_the_paper_s_is_drawn_under_its_own_labels_built_or_re_rendered(
        tmp_path, two_roots, capsys, monkeypatch):
    import matplotlib.pyplot as plt

    from ruleforge.experiments.training import build as build_family

    drawn = []

    def recording(golden, out_path, panel, dpi=None, names=None):
        """A ``draw`` that records the names it is passed and each arm's text in the panel."""
        figure = build_figure(golden, panel, names)
        axes = figure.axes[0]
        texts = axes.get_xticklabels()
        drawn.append((panel, dict(names or {}), [text.get_text().replace("\n", " ") for text in texts]))
        plt.close(figure)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"")

    monkeypatch.setattr(build_family, "draw", recording)
    for index, labels in enumerate((
            {"base": "Gemma 4 31B-it", "grpo": "GRPO (checkpoint-200)", "final": "GRPO (checkpoint-600)"},
            {"base": "Qwen3-8B", "grpo": "GRPO (checkpoint-300)", "final": "GRPO (final)"})):
        out = tmp_path / f"out{index}"
        argv = [*cli_arms(tmp_path, two_roots), "--reference", "base", "--rule-set", str(tmp_path / "A" / "rules.jsonl"),
                *[word for policy, text in labels.items() for word in ("--label", f"{policy}={text}")],
                "--out-dir", str(out)]
        code, _, err = run_script(argv, capsys)
        assert code == 0, err
        code, _, err = run_script(["--from_golden", str(out / "training_family.json"), "--figure-a",
                                   str(out / "a.pdf")], capsys)
        assert code == 0, err
        built = [("a", {}, list(labels.values())) for _ in ("pdf", "png")]
        assert drawn == [*built, ("a", {}, list(labels.values()))]
        drawn.clear()


@pytest.mark.parametrize(
    ("labels", "sizes", "fits"),
    [
        (["Gemma-4-31B-it", "GRPO-checkpoint-200", "GRPO-checkpoint-600"], set(TICK_FONT_SIZES[1:]), True),
        # nothing fits: the smallest font, and a warning
        (["A" * 40, "B" * 40, "C" * 40], {TICK_FONT_SIZES[-1]}, False),
    ],
)
def test_tick_labels_that_still_overlap_are_drawn_smaller_and_a_warning_says_when_none_fits(labels, sizes, fits,
                                                                                            caplog):
    import matplotlib.pyplot as plt

    golden = json.loads((REPO_ROOT / "tests" / "fixtures" / "training" / "synthetic_readout.json").read_text())
    for policy, label in zip(golden["policies"], labels):
        golden["arms"][policy]["label"] = label
    with caplog.at_level("WARNING", logger="ruleforge.experiments.training.figure"):
        figure = build_figure(golden, "a")
    ticks = figure.axes[0].get_xticklabels()
    [chosen] = {text.get_fontsize() for text in ticks}
    assert [text.get_text() for text in ticks] == labels and chosen in sizes
    figure.canvas.draw()
    boxes = [text.get_window_extent() for text in ticks]
    assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:])) is fits
    assert ("overlap even at 7 pt" in caplog.text) is not fits
    plt.close(figure)


def not_applicable(readout: dict, policy: str) -> dict:
    """``readout`` with ``policy``'s yes-shift baseline not applicable, as :func:`yes_shift` reports one."""
    changed = json.loads(json.dumps(readout))
    changed["yes_shift"][policy] = {**changed["yes_shift"][policy], "applicable": False, "reason": "why", "p": None,
                                    "shifted_rate": None, "share_of_gain_explained": None}
    return changed


def test_panel_a_draws_the_bars_alone_whatever_the_readout_s_baselines(tmp_path, two_roots):
    readout = json.loads(json.dumps(read(tmp_path, two_roots)))
    assert all(readout["yes_shift"][policy]["applicable"] for policy in ("grpo", "final"))
    assert [readout["arms"][policy]["label"] for policy in readout["policies"]] == [LABELS[p] for p in POLICIES]
    assert_panel_a_draws_the_flat_rates_alone(readout, ["0.917", "0.000", "0.125"])

    no_floors = json.loads(json.dumps(readout))
    for arm in no_floors["arms"].values():
        del arm["always_yes_floor"]
    variants = (
        not_applicable(readout, "grpo"),
        not_applicable(not_applicable(readout, "grpo"), "final"),
        {key: value for key, value in readout.items() if key != "yes_shift"},
        no_floors,
    )
    assert all(panel_values(variant, "a") == panel_values(readout, "a") for variant in variants)


def run_script(argv, capsys) -> tuple[int, str, str]:
    """``python -m ruleforge.experiments.training.build ARGV``."""
    from ruleforge.experiments.training.build import main

    try:
        code = main(list(argv))
    except SystemExit as exit_info:
        code = exit_info.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def cli_arms(tmp_path, roots, labels=POLICIES) -> list[str]:
    return [arg for label in labels for arg in ("--arm", label, uri(roots[label], tmp_path), f"{PREFIX}/{label}")]


def test_the_command_line_writes_the_readout_its_summary_and_the_panels(tmp_path, two_roots, capsys):
    out = tmp_path / "out"
    argv = [*cli_arms(tmp_path, two_roots), "--reference", "base", "--rule-set", str(tmp_path / "A" / "rules.jsonl"),
            "--label", "base=Gemma 4 31B-it", "--label", "grpo=GRPO (checkpoint-200)",
            "--label", "final=GRPO (checkpoint-600)",
            "--out-dir", str(out), "--dpi", "40"]
    code, printed, _ = run_script(argv, capsys)
    assert code == 0 and printed == (out / "training_family.md").read_text()
    written = json.loads((out / "training_family.json").read_text())
    expected = read(tmp_path, two_roots)
    assert written == json.loads(json.dumps(expected))
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["training_family.json", "training_family.md"] + [f"training_{p}.{s}" for p in "a" for s in ("pdf", "png")])
    assert all(path.stat().st_size > 0 for path in out.iterdir())
    code, _, err = run_script(argv, capsys)
    assert code == 2 and "exist already; pass --force" in err
    code, _, _ = run_script([*argv, "--force"], capsys)
    assert code == 0


def test_the_command_line_says_it_has_no_scratch_build_where_a_reader_suggests_one(tmp_path, two_roots, capsys):
    _edit(two_roots["final"] / "sampling_manifest.json", lambda m: m.update(campaign_mode="pilot"))
    argv = [*cli_arms(tmp_path, two_roots), "--reference", "base", "--rule-set", str(tmp_path / "A" / "rules.jsonl"),
            "--out-dir", str(tmp_path / "out")]
    code, _, err = run_script(argv, capsys)
    assert code == 2 and "campaign_mode 'pilot'" in err
    assert "has no --allow-incomplete: finish or re-analyse the root" in err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--out-dir", str(REPO_ROOT / "results" / "training" / "training_family")], "would be among the paper's files"),
        (["--out-dir", "OUT", "--rule-set", "/no/such/rules.jsonl"], "no such file: --rule-set"),
        (["--out-dir", "OUT", "--arm", "x", "artifact://A/root", "x"], "4 arms; the panels draw at most 3"),
        (["--out-dir", "OUT", "--label", "grpo=a", "--label", "grpo=b"], "unique POLICY=VALUE"),
        (["--out-dir", "OUT", "--reference", "nobody"], "the reference 'nobody' is not one of the family's labels"),
    ],
)
def test_the_command_line_refuses(tmp_path, two_roots, capsys, extra, message):
    rules = ["--rule-set", str(tmp_path / "A" / "rules.jsonl")] if "--rule-set" not in extra else []
    reference = ["--reference", "base"] if "--reference" not in extra else []
    argv = [*cli_arms(tmp_path, two_roots), *reference, *rules,
            *[str(tmp_path / "out") if arg == "OUT" else arg for arg in extra]]
    code, _, err = run_script(argv, capsys)
    assert code == 2 and message in err
    assert not (tmp_path / "out").exists()
