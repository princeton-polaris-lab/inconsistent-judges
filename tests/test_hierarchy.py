from __future__ import annotations

import dataclasses
import itertools
import json
import shutil

import numpy as np
import pytest

from ruleforge.evaluate.inference import clustered_paired_randomization_p, holm_adjust
from ruleforge.evaluate.scoring import PHI_MAP, calculate_inconsistency
from ruleforge.experiments.hierarchy import analysis as ha
from ruleforge.experiments.hierarchy import figure as hf
from ruleforge.experiments import campaign_roots as cr
from ruleforge.experiments.labels import CAMPAIGN_MODEL_LABELS
from ruleforge.experiments.arm_records import bootstrap_settings
from ruleforge.prompts.styles import NAIVE_RULES_ABOVE, NAIVE_HIERARCHY
from ruleforge.rulesets.levels import LEVEL_ORDER, load_rule_levels
from tests.test_leniency_campaign import N, PERMUTATIONS, RULES, S, artifact_root, recorded_id, write_rules
from tests.test_rule_levels import STUDY_LEVELS, write_levels
from tests.test_prompt_ablation_arms import GEMMA, LUNA, NEW, OLD, write_condition

OPEN, LUNA_FLAT, LUNA_HIER = "ha-open", "ha-luna-published", "ha-luna-hierarchy"
URI = {name: f"artifact://sampling/{name}" for name in (OPEN, LUNA_FLAT, LUNA_HIER)}
FLAT_STYLE, HIER_STYLE = NAIVE_RULES_ABOVE, NAIVE_HIERARCHY
#: The roots registered as a published campaign recorded them: each condition under its style's recorded key.
RECORDED = (OPEN,)
LEVELS = {"r": "root", "s": "system", "d": "developer", "u": "user", "g": "guideline"}
RANK = {rule: LEVEL_ORDER.index(level) for rule, level in LEVELS.items()}
STUDY_RANK = {f"Rule {k}": LEVEL_ORDER.index(level) for k, level in STUDY_LEVELS.items()}


def directory(name, model, style):
    """The directory (the registered id) of ``model``'s ``style`` condition in root ``name``."""
    return recorded_id(model, style) if name in RECORDED else f"{model}/{style}"


def registered(record, side):
    """The registered id of a judge record's ``side`` arm, which the record names by its public id."""
    return directory(record[side]["root"].rsplit("/", 1)[1], record["model_name"], record[side]["prompt_style"])


def rescore(root):
    """Each bundle's ``contradiction`` as the campaign scores it (``calculate_inconsistency``)."""
    for path in sorted(root.glob("*/*/draw-*/bundle_outcomes.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["contradiction"] = calculate_inconsistency(row["full_set_judgment"], list(row["rule_judgments"].values()))[
                "contradiction"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def bundles(root, condition_id):
    """Each draw's and scenario's bundle row, in manifest order: the draws as written."""
    ids = json.loads((root / "scenario_manifest.json").read_text())["scenario_ids"]
    out = np.empty((N, S), dtype=object)
    for s in range(S):
        rows = {row["scenario_id"]: row for row in map(
            json.loads, (root / condition_id / f"draw-{s:03d}" / "bundle_outcomes.jsonl").read_text().splitlines())}
        for i, sid in enumerate(ids):
            out[i, s] = rows[sid]
    return out


@pytest.mark.parametrize(
    ("full", "labels", "inconsistent", "over", "under", "top", "root_floor"),
    [
        # An over-flag: the full set flags what no rule does -- in the root floor whatever the levels.
        ("yes", {"r": "no", "s": "no", "d": "no", "u": "no", "g": "no"}, True, True, False, None, True),
        ("unsure", {"r": "no", "s": "no", "d": "no", "u": "no", "g": "no"}, True, True, False, None, True),
        # An under-flag a root rule reaches: no hierarchy excuses it.
        ("no", {"r": "yes", "s": "no", "d": "no", "u": "no", "g": "yes"}, True, False, True, "root", True),
        # An under-flag only a guideline sets: outside the root floor.
        ("no", {"r": "no", "s": "no", "d": "no", "u": "no", "g": "yes"}, True, False, True, "guideline", False),
        # M_H is the user rule's yes; the root rule's unsure exceeds the full set's no but does not reach M_H.
        ("no", {"r": "unsure", "s": "no", "d": "no", "u": "yes", "g": "no"}, True, False, True, "user", False),
        ("unsure", {"r": "unsure", "s": "yes", "d": "no", "u": "yes", "g": "no"}, True, False, True, "system", False),
        ("no", {"r": "no", "s": "no", "d": "unsure", "u": "no", "g": "no"}, True, False, True, "developer", False),
        ("yes", {"r": "no", "s": "no", "d": "no", "u": "no", "g": "yes"}, False, False, False, None, False),
        ("unsure", {"r": "unsure", "s": "no", "d": "no", "u": "no", "g": "no"}, False, False, False, None, False),
        ("no", {"r": "no", "s": "no", "d": "no", "u": "no", "g": "no"}, False, False, False, None, False),
    ],
)
def test_the_cell_outcome_by_hand(full, labels, inconsistent, over, under, top, root_floor):
    cell = ha.cell_outcome(full, labels, RANK)
    assert (cell.inconsistent, cell.over, cell.under, cell.root_floor) == (inconsistent, over, under, root_floor)
    assert (None if cell.top_rank is None else LEVEL_ORDER[cell.top_rank]) == top
    assert cell.phi_full == PHI_MAP[full] and cell.phi_max == max(PHI_MAP[label] for label in labels.values())


def test_the_definition_is_separate_call_s_on_every_cell_of_three_rules_at_every_placement():
    for full, labels, placement in itertools.product(
        PHI_MAP, itertools.product(PHI_MAP, repeat=3), itertools.product(LEVEL_ORDER, repeat=3)
    ):
        rank = {rule: LEVEL_ORDER.index(level) for rule, level in zip("abc", placement)}
        cell = ha.cell_outcome(full, dict(zip("abc", labels)), rank)
        assert int(cell.inconsistent) == calculate_inconsistency(full, list(labels))["contradiction"]
        assert not (cell.over and cell.under) and cell.inconsistent == (cell.over or cell.under)
        top = max(PHI_MAP[label] for label in labels)
        reaching = [rank[rule] for rule, label in zip("abc", labels) if PHI_MAP[label] == top]
        assert cell.top_rank == (min(reaching) if cell.under else None)
        assert cell.root_floor == (cell.over or (cell.under and 0 in reaching))


def test_with_every_rule_at_one_level_the_definition_is_the_flat_rate_s():
    for level, full, labels in itertools.product(LEVEL_ORDER, PHI_MAP, itertools.product(PHI_MAP, repeat=3)):
        rank = dict.fromkeys("abc", LEVEL_ORDER.index(level))
        assert int(ha.cell_outcome(full, dict(zip("abc", labels)), rank).inconsistent) == calculate_inconsistency(
            full, list(labels))["contradiction"]


def levels_sha(base):
    return load_rule_levels(base / "rules.jsonl", base / "rules_full.jsonl").levels_file["sha256"]


def hierarchy_design(base, **extra):
    return {"individual_rule_context": "superiors", "rule_levels_sha256": levels_sha(base), **extra}


def build_study(base, *, open_runtime_hier=NEW, luna_hier_design=None, hier_design=None):
    """The study's three roots under ``base``, aggregated."""
    write_rules(base / "rules.jsonl")
    write_levels(base / "rules_full.jsonl")
    write_condition(base, OPEN, model=GEMMA, style=FLAT_STYLE, seed=3, runtime=NEW, recorded=True)
    write_condition(base, OPEN, model=GEMMA, style=HIER_STYLE, seed=14, runtime=open_runtime_hier, recorded=True,
                    design=hierarchy_design(base, **(hier_design or {})))
    write_condition(base, LUNA_FLAT, model=LUNA, style=FLAT_STYLE, seed=5, model_type="openai_batch")
    write_condition(base, LUNA_HIER, model=LUNA, style=HIER_STYLE, seed=9, model_type="openai_batch",
                    design=hierarchy_design(base, **(luna_hier_design or {})))
    roots = {}
    for name in (OPEN, LUNA_FLAT, LUNA_HIER):
        roots[name] = base / "sampling" / name
        rescore(roots[name])
        from ruleforge.evaluate.aggregate import analyze_campaign

        analyze_campaign(roots[name], n_permutations=PERMUTATIONS)
    return roots


def pairs():
    return [ha.JudgePair(flat=(URI[LUNA_FLAT], f"{LUNA}/{FLAT_STYLE}"), hierarchy=(URI[LUNA_HIER], f"{LUNA}/{HIER_STYLE}")),
            ha.JudgePair(flat=(URI[OPEN], f"{GEMMA}/{FLAT_STYLE}"), hierarchy=(URI[OPEN], f"{GEMMA}/{HIER_STYLE}"))]


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    base = tmp_path_factory.mktemp("hier")
    with artifact_root(base):
        roots = build_study(base)
        payload = ha.build_payload(pairs=pairs(), rules_path=base / "rules.jsonl", levels_path=base / "rules_full.jsonl")
    return {"base": base, "roots": roots, "payload": payload}


@pytest.fixture
def env(study, monkeypatch):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(study["base"]))
    return study


def by_hand(flat_rows, hier_rows):
    """Every per-cell quantity recounted with no analysis code."""
    out = {name: np.zeros(flat_rows.shape) for name in (
        "flat", "hierarchy", "fh_i", "f_c", "flat_over", "flat_under", "hier_over", "hier_under", "root_floor",
        *(f"shift_{level}" for level in LEVEL_ORDER))}
    tops = dict.fromkeys(LEVEL_ORDER, 0)
    for index, flat in np.ndenumerate(flat_rows):
        hier = hier_rows[index]

        def phi_max(judgments):
            return max(PHI_MAP[label] for label in judgments.values())

        f, fh = PHI_MAP[flat["full_set_judgment"]], PHI_MAP[hier["full_set_judgment"]]
        mi, mc = phi_max(flat["rule_judgments"]), phi_max(hier["rule_judgments"])
        out["flat"][index], out["hierarchy"][index] = f != mi, fh != mc
        out["fh_i"][index], out["f_c"][index] = fh != mi, f != mc
        out["flat_over"][index], out["flat_under"][index] = f > mi, f < mi
        out["hier_over"][index], out["hier_under"][index] = fh > mc, fh < mc
        roots_reach = any(PHI_MAP[label] == mc and STUDY_RANK[rule] == 0 for rule, label in hier["rule_judgments"].items())
        out["root_floor"][index] = fh > mc or (fh < mc and roots_reach)
        if fh < mc:
            tops[LEVEL_ORDER[min(STUDY_RANK[rule] for rule, label in hier["rule_judgments"].items()
                                 if PHI_MAP[label] == mc)]] += 1
        for level in LEVEL_ORDER:
            members = [rule for rule, rank in STUDY_RANK.items() if LEVEL_ORDER[rank] == level]
            if members:
                out[f"shift_{level}"][index] = np.mean([PHI_MAP[hier["rule_judgments"][r]] - PHI_MAP[
                    flat["rule_judgments"][r]] for r in members])
    return out, tops


def test_every_quantity_is_recounted_by_hand_and_the_rates_are_separate_call_s(study):
    records = {record["judge"]: record for record in study["payload"]["judges"]}
    assert list(records) == ["gemma4-31b", "gpt-5.6-luna"]
    for record in records.values():
        flat_root = study["roots"][record["flat"]["root"].rsplit("/", 1)[1]]
        hier_root = study["roots"][record["hierarchy"]["root"].rsplit("/", 1)[1]]
        values, tops = by_hand(bundles(flat_root, registered(record, "flat")),
                               bundles(hier_root, registered(record, "hierarchy")))
        primary, secondary = record["primary"], record["secondary"]
        for side, root in (("flat", flat_root), ("hierarchy", hier_root)):
            estimates = json.loads((root / "aggregate" / "estimates.json").read_text())["conditions"][
                registered(record, side)]
            assert primary[f"{side}_rate"] == {key: estimates[key] for key in ("estimate", "ci_low", "ci_high", "sd")}
            assert primary[f"{side}_rate"]["estimate"] == pytest.approx(values[side].mean(), abs=1e-12)
            assert record[side]["grid_digest"] == estimates["grid_digest"]
        difference = primary["difference"]
        assert difference["estimate"] == pytest.approx(values["hierarchy"].mean() - values["flat"].mean(), abs=1e-12)
        assert difference["ci_low"] <= difference["estimate"] <= difference["ci_high"]
        assert difference["inner_resample"] == "independent" and primary["difference_of"] == "hierarchy - flat"
        cells = secondary["two_by_two"]
        for name, key in (("flat", "flat"), ("hierarchy", "hierarchy"), ("hierarchy_full_set_isolated_rules", "fh_i"),
                          (ha.MISMATCHED, "f_c")):
            assert cells[name]["estimate"] == pytest.approx(values[key].mean(), abs=1e-12)
        assert [name for name, cell in cells.items() if cell["mismatched"]] == [ha.MISMATCHED]
        assert (cells[ha.MISMATCHED]["full_set"], cells[ha.MISMATCHED]["reference"]) == ("F", "C")
        split = secondary["over_under"]
        for side, prefix in (("flat", "flat"), ("hierarchy", "hier")):
            assert split[side]["over"]["estimate"] == pytest.approx(values[f"{prefix}_over"].mean(), abs=1e-12)
            assert split[side]["under"]["estimate"] == pytest.approx(values[f"{prefix}_under"].mean(), abs=1e-12)
            assert split[side]["over"]["estimate"] + split[side]["under"]["estimate"] == pytest.approx(
                primary[f"{side}_rate"]["estimate"])
        assert secondary["root_floor"]["estimate"] == pytest.approx(values["root_floor"].mean(), abs=1e-12)
        assert secondary["root_floor"]["estimate"] <= primary["hierarchy_rate"]["estimate"] + 1e-12
        # No system rule in the study: its level has no shift. Root is the placebo.
        assert list(secondary["level_shift"]) == ["root", "developer", "user", "guideline"]
        assert [level for level, shift in secondary["level_shift"].items() if shift["placebo"]] == ["root"]
        for level, shift in secondary["level_shift"].items():
            assert shift["estimate"] == pytest.approx(values[f"shift_{level}"].mean(), abs=1e-12)
            assert shift["n_rules"] == list(STUDY_LEVELS.values()).count(level)
        top_levels = secondary["under_flags_by_top_level"]
        assert {level: entry["count"] for level, entry in top_levels.items()} == tops
        assert sum(entry["count"] for entry in top_levels.values()) == values["hier_under"].sum()
        assert sum(entry["share_of_under_flags"] or 0 for entry in top_levels.values()) == pytest.approx(1)


def test_the_primary_difference_is_the_paired_randomization_test_s_and_holm_runs_over_the_judges(study):
    records = study["payload"]["judges"]
    for record in records:
        flat_root = study["roots"][record["flat"]["root"].rsplit("/", 1)[1]]
        hier_root = study["roots"][record["hierarchy"]["root"].rsplit("/", 1)[1]]
        with artifact_root(study["base"]):
            roots = cr.open_roots({record[side]["root"]: record[side]["root"] for side in ("flat", "hierarchy")}, "t",
                                  allow_incomplete=False, allow_stack_mismatch=False)
        grids = {side: cr.arm_outcomes(roots[record[side]["root"]],
                                       roots[record[side]["root"]].conditions[registered(record, side)], S, "t")[1]
                 for side in ("flat", "hierarchy")}
        test = clustered_paired_randomization_p(grids["hierarchy"], grids["flat"], n_permutations=PERMUTATIONS, seed=16)
        assert record["primary"]["difference"]["p_value"] == test["p_value"]
        assert record["primary"]["difference"]["n_permutations"] == PERMUTATIONS
        assert flat_root.is_dir() and hier_root.is_dir()
    holm = holm_adjust([{"p_value": record["primary"]["difference"]["p_value"]} for record in records], alpha=0.05)
    assert [record["primary"]["difference"]["p_holm"] for record in records] == [row["p_holm"] for row in holm]
    assert {record["primary"]["difference"]["family_size"] for record in records} == {2}
    assert {record["primary"]["difference"]["test_method"] for record in records} == {ha.TEST_METHOD}


def test_the_payload_says_what_it_is_and_where_it_came_from(study):
    payload = study["payload"]
    assert payload["analysis"] == "hierarchy" and payload["schema_version"] == ha.SCHEMA_VERSION
    assert "(scenario, decoder draw index)" in payload["pairing"] and "Holm" in payload["multiplicity"]
    assert "Descriptive, no test" in payload["secondary"] and "M_H" in payload["definition"]
    assert payload["levels"]["order"] == list(LEVEL_ORDER) and payload["levels"]["rules"]["system"] == []
    provenance = payload["provenance"]
    levels = load_rule_levels(study["base"] / "rules.jsonl", study["base"] / "rules_full.jsonl")
    assert (provenance["ruleset"], provenance["levels"]) == (levels.ruleset, levels.levels_file)
    assert provenance["approved_prompts_sha256"] == ha.APPROVED_PROMPT_SHA256
    assert provenance["pairs"][0] == {"flat": list(pairs()[0].flat), "hierarchy": list(pairs()[0].hierarchy)}
    campaigns = {entry["root"]: entry for entry in provenance["campaigns"]}
    assert set(campaigns) == set(URI.values())
    for name in study["roots"]:
        entry = campaigns[URI[name]]
        assert "root_digest" not in entry and entry["estimates"] == f"{URI[name]}/aggregate/estimates.json"
        assert entry["campaign_id"] == name and entry["label_replay"] is None
    assert "scenario_exclusion" not in provenance
    assert campaigns[URI[OPEN]]["conditions"] == [f"{GEMMA}/{HIER_STYLE}", f"{GEMMA}/{FLAT_STYLE}"]
    assert provenance["bootstrap"]["n_permutations"] == PERMUTATIONS and provenance["n_decoder_draws"] == S
    gemma = payload["judges"][0]
    # The open root's conditions are registered under their styles' recorded keys; the golden names them publicly.
    assert [(gemma[side]["condition_id"], gemma[side]["prompt_style"]) for side in ("flat", "hierarchy")] == [
        (f"{GEMMA}/{FLAT_STYLE}", FLAT_STYLE), (f"{GEMMA}/{HIER_STYLE}", HIER_STYLE)]
    assert directory(OPEN, GEMMA, HIER_STYLE) != f"{GEMMA}/{HIER_STYLE}"
    assert gemma["flat"]["stack"]["kind"] == "local" and payload["judges"][1]["flat"]["stack"]["kind"] == "provider"
    assert gemma["hierarchy"]["rule_levels_sha256"] == levels.levels_file["sha256"]
    assert gemma["flat"]["rule_levels_sha256"] is None and gemma["n_scenarios"] == N


def test_one_condition_fed_to_both_slots_gives_a_zero_difference_and_separate_call_s_rate(env):
    with artifact_root(env["base"]):
        roots = cr.open_roots({URI[OPEN]: URI[OPEN]}, "t", allow_incomplete=False, allow_stack_mismatch=False)
    root = roots[URI[OPEN]]
    spec = root.conditions[directory(OPEN, GEMMA, FLAT_STYLE)]
    cells, grid = cr.arm_outcomes(root, spec, S, "t")
    arm = ha.Arm(root, spec, cells, grid, cr.arm_stack(root, spec, "t"))
    levels = load_rule_levels(env["base"] / "rules.jsonl", env["base"] / "rules_full.jsonl")
    settings = bootstrap_settings(root.payload)
    record = ha.compare("same", arm, arm, levels, settings)
    difference = record["primary"]["difference"]
    assert (difference["estimate"], difference["ci_low"], difference["ci_high"], difference["p_value"]) == (0, 0, 0, 1)
    estimates = json.loads((env["roots"][OPEN] / "aggregate" / "estimates.json").read_text())["conditions"][
        spec.condition_id]
    assert record["primary"]["flat_rate"]["estimate"] == record["primary"]["hierarchy_rate"]["estimate"] == (
        estimates["estimate"])
    assert record["flat"]["grid_digest"] == estimates["grid_digest"] == grid.digest
    shifts = record["secondary"]["level_shift"]
    assert {level: (shift["estimate"], shift["ci_low"], shift["ci_high"]) for level, shift in shifts.items()} == {
        level: (0, 0, 0) for level in shifts}
    cells_2x2 = record["secondary"]["two_by_two"]
    assert len({cell["estimate"] for cell in cells_2x2.values()}) == 1


def test_two_arms_that_do_not_cover_the_same_grid_are_refused(env):
    with artifact_root(env["base"]):
        roots = cr.open_roots({URI[OPEN]: URI[OPEN]}, "t", allow_incomplete=False, allow_stack_mismatch=False)
    root = roots[URI[OPEN]]
    spec = root.conditions[directory(OPEN, GEMMA, FLAT_STYLE)]
    cells, grid = cr.arm_outcomes(root, spec, S, "t")
    arm = ha.Arm(root, spec, cells, grid, {})
    short = dataclasses.replace(arm, grid=grid.restricted_to(grid.scenario_ids[:-1]))
    levels = load_rule_levels(env["base"] / "rules.jsonl", env["base"] / "rules_full.jsonl")
    with pytest.raises(ha.HierarchyError, match="do not cover the same scenarios and draws"):
        ha.compare("same", arm, short, levels, bootstrap_settings(root.payload))


def _build(base, the_pairs=None, levels=None):
    with artifact_root(base):
        return ha.build_payload(pairs=pairs() if the_pairs is None else the_pairs, rules_path=base / "rules.jsonl",
                                levels_path=levels or base / "rules_full.jsonl")


@pytest.mark.parametrize(
    ("the_pairs", "match"),
    [
        ([], "no judge's arms given"),
        ([ha.JudgePair(flat=("sampling/ha-open", f"{GEMMA}/{FLAT_STYLE}"), hierarchy=(URI[OPEN], f"{GEMMA}/{HIER_STYLE}"))],
         "is not an artifact:// URI"),
        ([*pairs(), pairs()[1]], r"\['gemma4-31b'\] given twice"),
        ([ha.JudgePair(flat=(URI[OPEN], f"{GEMMA}/{HIER_STYLE}"), hierarchy=(URI[OPEN], f"{GEMMA}/{HIER_STYLE}"))],
         r"the flat arm gemma-4-31B-it/naive-hierarchy is 'naive-hierarchy' with its rules judged 'superiors', not"),
        ([ha.JudgePair(flat=(URI[OPEN], f"{GEMMA}/{FLAT_STYLE}"), hierarchy=(URI[OPEN], f"{GEMMA}/{FLAT_STYLE}"))],
         r"the hierarchy arm gemma-4-31B-it/naive-rules-above is 'naive-rules-above' with its rules judged 'isolated'"),
        ([ha.JudgePair(flat=(URI[LUNA_FLAT], f"{LUNA}/{FLAT_STYLE}"), hierarchy=(URI[OPEN], f"{GEMMA}/{HIER_STYLE}"))],
         r"its arms differ in \[.*'model_name'.*\], not only in"),
        ([ha.JudgePair(flat=(URI[OPEN], f"{GEMMA}/nothing"), hierarchy=(URI[OPEN], f"{GEMMA}/{HIER_STYLE}"))],
         "no registered condition has the public id 'gemma-4-31B-it/nothing'"),
    ],
)
def test_arms_that_are_not_one_judge_s_flat_and_hierarchy_conditions_are_refused(study, the_pairs, match):
    with pytest.raises(cr.CampaignEstimatesError, match=match):
        _build(study["base"], the_pairs)


def test_a_judge_with_no_golden_label_is_refused_not_named_by_its_model(study, monkeypatch):
    monkeypatch.delitem(CAMPAIGN_MODEL_LABELS, GEMMA)
    with pytest.raises(cr.CampaignEstimatesError, match="no golden label for judge 'gemma-4-31B-it'"):
        _build(study["base"])


def test_a_levels_file_that_does_not_align_or_is_not_the_hierarchy_arm_s_is_refused(study, tmp_path):
    base = study["base"]
    misaligned = write_levels(tmp_path / "full.jsonl", levels={**STUDY_LEVELS, 1: "platform"})
    with pytest.raises(ha.HierarchyError, match="no level, or one not in"):
        _build(base, levels=misaligned)
    other = write_levels(tmp_path / "other.jsonl", levels={**STUDY_LEVELS, 1: "root"})
    with pytest.raises(ha.HierarchyError, match="the hierarchy arm read levels file .*, not other.jsonl"):
        _build(base, levels=other)


def test_another_ruleset_is_refused(study, tmp_path):
    rules = write_rules(tmp_path / "rules.jsonl", [{**RULES[0], "rule": RULES[0]["rule"] + "!"}, *RULES[1:]])
    levels = write_levels(tmp_path / "full.jsonl", rules=[{**RULES[0], "rule": RULES[0]["rule"] + "!"}, *RULES[1:]])
    with artifact_root(study["base"]), pytest.raises(ha.HierarchyError, match="judged against ruleset rules.jsonl"):
        ha.build_payload(pairs=pairs()[1:], rules_path=rules, levels_path=levels)


def test_a_label_rule_or_another_setting_beyond_the_arm_s_fields_is_refused(tmp_path):
    with artifact_root(tmp_path):
        build_study(tmp_path, luna_hier_design={"label_rule": "lenient"})
    with pytest.raises(ha.HierarchyError, match=r"its arms differ in \['label_rule'\], not only in"):
        _build(tmp_path, pairs()[:1])


def test_a_flat_control_on_another_stack_is_refused(tmp_path):
    with artifact_root(tmp_path):
        build_study(tmp_path, open_runtime_hier=OLD)
    with pytest.raises(ha.HierarchyError, match="different inference stacks .* a same-stack control"):
        _build(tmp_path, pairs()[1:])


def test_roots_with_other_campaign_settings_are_refused(tmp_path):
    with artifact_root(tmp_path):
        build_study(tmp_path)
    for name in ("sampling_manifest.json", "aggregate/estimates.json"):
        path = tmp_path / "sampling" / LUNA_HIER / name
        path.write_text(json.dumps({**json.loads(path.read_text()), "max_semantic_attempts": 5}))
    with pytest.raises(cr.CampaignRootError, match="its arms' roots differ in"):
        _build(tmp_path, pairs()[:1])


def _edit_draw(root, condition_id, edit):
    path = root / condition_id / "draw-001" / "bundle_outcomes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit(rows[0])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows[0]["scenario_id"]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda row: row["rule_judgments"].pop("Rule 3"), "its rule_judgments do not name exactly the ruleset's"),
        (lambda row: row.update(rule_judgments=None), "its rule_judgments do not name exactly the ruleset's"),
        (lambda row: row["rule_judgments"].update({"Rule 3": "maybe"}), "a judgment outside"),
        (lambda row: row.update(full_set_judgment=None), "a judgment outside"),
    ],
)
def test_damaged_bundles_are_refused(tmp_path, edit, match):
    with artifact_root(tmp_path):
        roots = build_study(tmp_path)
    sid = _edit_draw(roots[OPEN], directory(OPEN, GEMMA, HIER_STYLE), edit)
    with pytest.raises(ha.HierarchyError, match=rf"{sid[:12]}/draw 1 \(hierarchy\): {match}"):
        _build(tmp_path, pairs()[1:])


def test_labels_changed_under_a_kept_contradiction_and_an_edited_estimate_are_refused(tmp_path):
    with artifact_root(tmp_path):
        roots = build_study(tmp_path)

    def flip(row):
        row["full_set_judgment"] = "no" if row["contradiction"] == 1 else "yes"
        row["rule_judgments"] = {rule: "no" for rule in row["rule_judgments"]}

    _edit_draw(roots[OPEN], directory(OPEN, GEMMA, FLAT_STYLE), flip)
    with pytest.raises(ha.HierarchyError, match="the flat arm's recomputed inconsistency is not its stored"):
        _build(tmp_path, pairs()[1:])
    shutil.rmtree(tmp_path / "sampling")
    with artifact_root(tmp_path):
        roots = build_study(tmp_path)
    estimates = roots[LUNA_HIER] / "aggregate" / "estimates.json"
    payload = json.loads(estimates.read_text())
    payload["conditions"][f"{LUNA}/{HIER_STYLE}"]["estimate"] += 0.5
    estimates.write_text(json.dumps(payload))
    with pytest.raises(ha.HierarchyError, match="the hierarchy arm's rate .* is not its estimates'"):
        _build(tmp_path, pairs()[:1])


def test_the_csv_rows_carry_every_quantity_and_the_p_values_on_the_difference_alone(study):
    rows = ha.csv_rows(study["payload"])
    assert set(rows[0]) == set(ha.CSV_COLUMNS)
    per_judge = len(rows) // 2
    assert [row["quantity"] for row in rows[:per_judge]][:3] == [
        "flat_rate", "hierarchy_rate", "difference_hierarchy_minus_flat"]
    assert {row["family"] for row in rows[:3]} == {"primary"} and {row["family"] for row in rows[3:per_judge]} == {
        "secondary"}
    difference = rows[2]
    assert difference["p_holm"] == study["payload"]["judges"][0]["primary"]["difference"]["p_holm"]
    assert all(row["p_value"] == "" for row in rows if row["quantity"] != "difference_hierarchy_minus_flat")
    assert f"two_by_two:{ha.MISMATCHED}" in {row["quantity"] for row in rows}
    assert {"level_shift:root", "root_floor", "over:hierarchy", "under:flat"} <= {row["quantity"] for row in rows}
    tops = study["payload"]["judges"][0]["secondary"]["under_flags_by_top_level"]
    by_top = {row["quantity"].split(":")[1]: row for row in rows[:per_judge]
              if row["quantity"].startswith("under_flags_top_level:")}
    assert list(by_top) == list(LEVEL_ORDER)
    assert all(row["estimate"] == tops[level]["rate"] and row["ci_low"] == row["ci_high"] == ""
               for level, row in by_top.items())


def test_the_summary_renders_the_same_from_the_written_json(study):
    text = ha.summary_markdown(study["payload"])
    assert text == ha.summary_markdown(json.loads(json.dumps(study["payload"], sort_keys=True)))
    assert "## Primary (Holm over the judges)" in text and "F vs C (mismatched)" in text
    assert "| gemma4-31b |" in text and "root: the placebo" in text
    assert "| - |" in text  # the system level holds no rule in the study
    # The flat control as it is: an open judge's same-stack re-run, an API judge's published arm with its dates.
    primary = text.split("## Secondary")[0].splitlines()
    gemma, luna = (next(line for line in primary if line.startswith(f"| {record['judge']} |"))
                   for record in study["payload"]["judges"])
    assert gemma.endswith("| same stack (" + study["payload"]["judges"][0]["flat"]["stack"]["short"] + ") |")
    assert "| published arm at the provider (requests " in luna and "; hierarchy " in luna


def _figure_parts(figure):
    """The drawn artists a test reads: the one axes, its bars by arm and its rate labels."""
    from matplotlib.container import BarContainer

    [axes] = figure.axes
    containers = [container for container in axes.containers if isinstance(container, BarContainer)]
    bars = {side: container for side, container in zip(hf.ARMS, containers, strict=True)}
    return axes, bars, axes.texts


def _tick_marks(axes, judges):
    """Each tick label as it reads, (the judge's display name, an asterisk after it), checked to hold nothing else."""
    from ruleforge.experiments.labels import MODEL_DISPLAY_LABELS

    marks = []
    for tick, record in zip(axes.get_xticklabels(), judges, strict=True):
        name, text = MODEL_DISPLAY_LABELS.get(record["judge"], record["judge"]), tick.get_text()
        assert text in (name, f"{name}*"), text
        marks.append((name, text == f"{name}*"))
    return marks


def _calls(judges):
    return [record["primary"]["difference"]["significant_holm"] for record in judges]


def _legend_rows(figure):
    """The legend strip's words by drawn row, top to bottom, each row left to right."""
    [axes] = figure.axes
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    rows = {}
    for text in axes.get_legend().get_texts():
        box = text.get_window_extent(renderer)
        rows.setdefault(round((box.y0 + box.y1) / 2), []).append((box.x0, text.get_text()))
    return [[words for _, words in sorted(row)] for _, row in sorted(rows.items(), reverse=True)]


def _assert_no_difference_drawn(figure, judges):
    from matplotlib.text import Text

    [axes] = figure.axes
    figure.canvas.draw()
    drawn = [text.get_text() for text in figure.findobj(Text) if text.get_visible() and text.get_text()]
    assert sorted(drawn) == sorted(text.get_text() for text in [
        *axes.get_xticklabels(), *axes.get_yticklabels(), axes.yaxis.label, *axes.get_legend().get_texts(),
        *axes.texts])
    assert not [text for text in drawn if "Δ" in text or "†" in text]
    values = {form for record in judges for value in [f"{record['primary']['difference']['estimate']:+.3f}"]
              for form in (value, value.replace("-", "−"))}
    assert not [text for text in drawn if any(value in text for value in values)]


def _assert_labels_placed(figure, judges):
    axes, bars, labels = _figure_parts(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    points = figure.dpi / 72
    to_display = axes.transData.transform
    for side in hf.ARMS:
        half_cap = max(cap.get_markeredgewidth() for cap in bars[side].errorbar.lines[1]) / 2 * points
        for bar, record in zip(bars[side].patches, judges, strict=True):
            centre, top = bar.get_x() + bar.get_width() / 2, record["primary"][f"{side}_rate"]["ci_high"]
            [label] = [label for label in labels if label.xy[0] == pytest.approx(centre)]
            assert label.xy == pytest.approx((centre, top))
            assert label.get_window_extent(renderer).y0 >= to_display((centre, top))[1] + half_cap
    assert len(labels) == 2 * len(judges) and not axes.artists and not axes.images
    assert {line.get_marker() for line in axes.lines} == {"_"}
    assert [star for _, star in _tick_marks(axes, judges)] == _calls(judges)
    _assert_no_difference_drawn(figure, judges)
    frame, strip = axes.get_window_extent(renderer), axes.get_legend().get_window_extent(renderer)
    assert strip.y0 >= frame.y1 and strip.y1 <= figure.bbox.y1
    assert strip.x0 == pytest.approx(frame.x0, abs=0.5) and strip.x1 <= frame.x1


def _bars_panel(labels, part):
    from ruleforge.experiments.prompt_ablation import figure as pf

    bars = {bar for part_bars in pf.FIGURE_PARTS.values() for bar in part_bars}
    block = {label: {"all": {bar: {"mean": 0.5, "ci_low": 0.45, "ci_high": 0.55, "n": 500} for bar in bars}}
             for label in labels}
    return pf.build_figure(block, "all", part)


def _panel_style(figure):
    from matplotlib.colors import to_hex
    from matplotlib.container import BarContainer

    [axes] = figure.axes
    containers = [container for container in axes.containers if isinstance(container, BarContainer)]
    legend, frame = axes.get_legend(), axes.get_legend().get_frame()
    params = figure.subplotpars
    return {
        "size": tuple(figure.get_size_inches()),
        "margins": (params.bottom, params.left, params.right),
        "font": {tuple(text.get_fontfamily()) for text in [*axes.get_xticklabels(), *axes.get_yticklabels(),
                                                           axes.yaxis.label, *legend.get_texts()]},
        "text sizes": ({tick.get_fontsize() for tick in axes.get_xticklabels()},
                       {tick.get_fontsize() for tick in axes.get_yticklabels()}, axes.yaxis.label.get_fontsize(),
                       {text.get_fontsize() for text in legend.get_texts()}),
        "y label": (axes.get_ylabel(), to_hex(axes.yaxis.label.get_color())),
        "bar edge width": {bar.get_linewidth() for container in containers for bar in container.patches},
        "caps": {(cap.get_markersize(), cap.get_markeredgewidth(), to_hex(cap.get_markeredgecolor()),
                  to_hex(cap.get_color())) for container in containers for cap in container.errorbar.lines[1]},
        "stems": {(width, to_hex(colour)) for container in containers for stems in container.errorbar.lines[2]
                  for width, colour in zip(stems.get_linewidths(), stems.get_colors())},
        "grid": ({(line.get_visible(), to_hex(line.get_color()), line.get_linewidth(), line.get_linestyle())
                  for line in axes.get_ygridlines()}, {line.get_visible() for line in axes.get_xgridlines()}),
        "spines": {side for side, spine in axes.spines.items() if spine.get_visible()},
        "legend": (legend._loc, legend.handlelength,
                   legend.handletextpad, legend.columnspacing, legend.borderpad, legend.borderaxespad,
                   to_hex(frame.get_facecolor()), to_hex(frame.get_edgecolor()), frame.get_linewidth()),
    }


def _bar_styles(figure):
    """Each bar container's (fill, edge, hatch), in drawing order (left to right within a group)."""
    from matplotlib.colors import to_hex
    from matplotlib.container import BarContainer

    [axes] = figure.axes
    return [{(to_hex(bar.get_facecolor()), to_hex(bar.get_edgecolor()), bar.get_hatch() or "")
             for bar in container.patches}
            for container in axes.containers if isinstance(container, BarContainer)]


def _group_geometry(figure):
    """Each bar's drawn width and its centre's offset from its group's tick, in drawing order."""
    from matplotlib.container import BarContainer

    [axes] = figure.axes
    return [value for container in axes.containers if isinstance(container, BarContainer)
            for bar, tick in zip(container.patches, axes.get_xticks(), strict=True)
            for value in (bar.get_width(), bar.get_x() + bar.get_width() / 2 - tick)]


SIGNIFICANCE_KEY = "Holm-significant (hierarchy − flat)"
STUDY_CALLS = [False, False]
STUDY_TICKS = ["Gemma-4-31B", "GPT-5.6 Luna"]


def test_the_figure_is_the_prompt_ablation_panel_with_the_training_values_and_an_asterisk_for_holm(study, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from ruleforge.experiments import style as theme

    judges = hf.drawn_judges(study["payload"])
    figure = hf.build_figure(study["payload"])
    try:
        axes, bars, labels = _figure_parts(figure)
        assert tuple(figure.get_size_inches()) == hf.PANEL_SIZE
        assert not figure.texts and figure._suptitle is None and axes.get_title() == ""
        assert _calls(judges) == STUDY_CALLS
        assert judges[1]["flat"]["stack"]["kind"] == "provider"
        assert [tick.get_text() for tick in axes.get_xticklabels()] == STUDY_TICKS
        assert {tick.get_fontsize() for tick in axes.get_xticklabels()} == {hf.TICK_FONTSIZE}
        assert axes.get_ylabel() == hf.Y_LABEL and axes.yaxis.label.get_fontsize() == hf.Y_LABEL_FONTSIZE
        assert axes.get_ylim() == (0.0, 1.0) and list(axes.get_yticks()) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1])
        for side, (_, (fill, edge, hatch)) in hf.ARMS.items():
            rates = [record["primary"][f"{side}_rate"] for record in judges]
            assert [bar.get_height() for bar in bars[side].patches] == [rate["estimate"] for rate in rates]
            assert {(to_hex(bar.get_facecolor()), to_hex(bar.get_edgecolor()), bar.get_hatch() or "")
                    for bar in bars[side].patches} == {(fill.lower(), edge.lower(), hatch)}
        whiskers = [y for side in hf.ARMS for segment in bars[side].errorbar.lines[2][0].get_segments()
                    for y in segment[:, 1]]
        assert whiskers == pytest.approx([record["primary"][f"{side}_rate"][bound] for side in hf.ARMS
                                          for record in judges for bound in ("ci_low", "ci_high")])
        assert sorted(label.get_text() for label in labels) == sorted(
            f"{record['primary'][f'{side}_rate']['estimate']:.3f}" for side in hf.ARMS for record in judges)
        assert {(label.get_fontweight(), label.get_fontsize(), label.get_color()) for label in labels} == {
            ("bold", theme.VALUE_LABEL["fontsize"], theme.VALUE_LABEL["color"])}
        assert _legend_rows(figure) == [["flat", "hierarchy", SIGNIFICANCE_KEY]]
        _assert_labels_placed(figure, judges)
    finally:
        plt.close(figure)
    for suffix in ("png", "pdf"):
        hf.draw(study["payload"], tmp_path / f"f.{suffix}", dpi=40)
        assert (tmp_path / f"f.{suffix}").stat().st_size > 0


def test_the_legend_keys_each_arm_as_the_prompt_ablation_panel_does_and_the_asterisk_as_the_ticks_draw_it(study):
    pytest.importorskip("matplotlib")
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as Outline
    from matplotlib.textpath import TextPath

    from ruleforge.experiments import style as theme
    from ruleforge.experiments.prompt_ablation.figure import BAR_STYLE
    from ruleforge.prompts.styles import ENCOURAGED

    payload = _judges_payload(study, THREE)
    judges = hf.drawn_judges(payload)
    assert sorted(set(_calls(judges))) == [False, True]
    figure = hf.build_figure(payload)
    try:
        [axes] = figure.axes
        legend = axes.get_legend()
        keys = list(zip((text.get_text() for text in legend.get_texts()), legend.legend_handles, strict=True))
        assert [(text, (to_hex(handle.get_facecolor()), to_hex(handle.get_edgecolor()), handle.get_hatch() or ""))
                for text, handle in keys[:2]] == [
            (label, (to_hex(fill), to_hex(edge), hatch))
            for label, (fill, edge, hatch) in (("flat", BAR_STYLE["styles"]["naive"]),
                                               ("hierarchy", BAR_STYLE["styles"][ENCOURAGED]))]
        [(text, key)] = keys[2:]
        assert text == SIGNIFICANCE_KEY and isinstance(key, Line2D) and key.get_linestyle() == "None"
        # The key's outline, at its size in points, is the tick font's asterisk centred on the key's point.
        tick = next(tick for tick, (_, star) in zip(axes.get_xticklabels(), _tick_marks(axes, judges)) if star)
        with matplotlib.rc_context(theme.SOFT_STYLE):
            glyph = TextPath((0, 0), "*", size=tick.get_fontsize(), prop=tick.get_fontproperties())
        box = glyph.get_extents()
        drawn = MarkerStyle(key.get_marker())
        outline = drawn.get_transform().transform(drawn.get_path().vertices) * key.get_markersize()
        shown = drawn.get_path().codes != Outline.CLOSEPOLY
        assert outline[shown] == pytest.approx(
            glyph.vertices[glyph.codes != Outline.CLOSEPOLY] - ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))
        assert (to_hex(key.get_markerfacecolor()), to_hex(key.get_markeredgecolor()), key.get_markeredgewidth()) == (
            to_hex(tick.get_color()), to_hex(tick.get_color()), 0.0)
        assert to_hex(tick.get_color()) == to_hex(theme.TICK_COLOR)
        assert [star for _, star in _tick_marks(axes, judges)] == _calls(judges)
        assert [tick.get_text() for tick in axes.get_xticklabels()] == ["Olmo-3-32B", "Gemma-4-31B*", "GPT-OSS-120B*"]
    finally:
        plt.close(figure)


def test_the_figure_refuses_a_holm_family_other_than_the_judges_it_draws(study):
    payload = {**study["payload"], "judges": study["payload"]["judges"][:1]}
    with pytest.raises(ValueError, match=r"families of \[2\] judges, the figure draws 1"):
        hf.drawn_judges(payload)
    with pytest.raises(ValueError, match="Holm over every judge drawn"):
        hf.build_figure(payload)


SEVEN = [("claude-haiku-5-5", 0.55, True, False), ("Qwen3-8B", 0.95, False, True), ("gpt-oss-120b", 0.63, False, True),
         ("Olmo3-32B", 0.76, False, False), ("gpt-5.6-luna", 0.35, True, True), ("gemma4-31b", 0.72, False, True),
         ("muse-glimmer-30b", 0.44, False, False)]
THREE = [row for row in SEVEN if row[0] in ("Olmo3-32B", "gemma4-31b", "gpt-oss-120b")]


def _judges_payload(study, rows):
    template = study["payload"]["judges"][0]
    judges = []
    for label, rate, provider, significant in rows:
        record = json.loads(json.dumps(template))
        record.update(judge=label)
        record["flat"]["stack"] = {"kind": "provider"} if provider else template["flat"]["stack"]
        primary = record["primary"]
        primary["flat_rate"].update(estimate=rate, ci_low=rate - 0.03, ci_high=min(1.0, rate + 0.04))
        primary["hierarchy_rate"].update(estimate=rate - 0.05, ci_low=rate - 0.08, ci_high=rate - 0.02)
        primary["difference"].update(estimate=-0.05, ci_low=-0.07, ci_high=-0.03, family_size=len(rows),
                                     significant_holm=significant)
        judges.append(record)
    return {**study["payload"], "judges": judges}


def test_the_figure_is_drawn_in_the_prompt_ablation_panel_style(study):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from ruleforge.experiments.prompt_ablation.figure import BAR_STYLE
    from ruleforge.prompts.styles import ENCOURAGED

    payload = _judges_payload(study, THREE)
    labels = [record["judge"] for record in payload["judges"]]
    figure, styles, methods = (hf.build_figure(payload), _bars_panel(labels, "styles"),
                               _bars_panel(labels, "methods"))
    try:
        assert hf.ARMS == {"flat": ("flat", BAR_STYLE["styles"]["naive"]),
                           "hierarchy": ("hierarchy", BAR_STYLE["styles"][ENCOURAGED])}
        ours, theirs = _panel_style(figure), _panel_style(styles)
        sizes = ours.pop("text sizes"), theirs.pop("text sizes")
        assert ours == theirs
        assert sizes == (({11}, {11}, 12, {11}), ({15}, {15}, 16, {15}))
        assert ours["size"] == hf.PANEL_SIZE
        assert _bar_styles(figure) == _bar_styles(styles)[:2]
        assert _group_geometry(figure) == pytest.approx(_group_geometry(methods))
        [axes], [their_axes] = figure.axes, styles.axes
        # The legend strip's anchor (above the axes, at their left edge), in axes fractions.
        anchors = [panel.get_legend().get_bbox_to_anchor().transformed(panel.transAxes.inverted()).bounds
                   for panel in (axes, their_axes)]
        assert anchors[0] == pytest.approx(anchors[1])
        judges = hf.drawn_judges(payload)
        assert [name for name, _ in _tick_marks(axes, judges)] == [
            tick.get_text() for tick in their_axes.get_xticklabels()]
        _assert_labels_placed(figure, judges)
        assert _legend_rows(figure) == [["flat", "hierarchy", SIGNIFICANCE_KEY]]
    finally:
        for drawn in (figure, styles, methods):
            plt.close(drawn)


def test_a_legend_strip_whose_one_row_is_wider_than_the_axes_takes_two():
    pytest.importorskip("matplotlib")
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from ruleforge.experiments import style as theme

    def strip(width):
        """The strip on axes ``width`` inches wide: its rows, whether it fits, and its width in inches."""
        figure = plt.figure(figsize=(width + 2.0, 3.0))
        axes = figure.add_axes((1.0 / (width + 2.0), 0.2, width / (width + 2.0), 0.5))
        try:
            hf._legend(figure, axes, [Patch(), Patch()])
            rows = _legend_rows(figure)
            renderer = figure.canvas.get_renderer()
            box = axes.get_legend().get_window_extent(renderer)
            return rows, box.x1 <= axes.get_window_extent(renderer).x1, box.width / figure.dpi
        finally:
            plt.close(figure)

    one_row, two_rows = [["flat", "hierarchy", SIGNIFICANCE_KEY]], [["flat", SIGNIFICANCE_KEY], ["hierarchy"]]
    with matplotlib.rc_context(theme.SOFT_STYLE):
        rows, fits, wide = strip(20.0)
        assert (rows, fits) == (one_row, True)
        rows, fits, narrow = strip(0.5)
        assert (rows, fits) == (two_rows, False) and narrow < wide
        assert strip(wide + 0.01)[:2] == (one_row, True)
        assert strip((wide + narrow) / 2)[:2] == (two_rows, True)


def test_the_figure_grows_to_seven_judges_in_paper_order_with_every_label_apart_and_inside(study):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from ruleforge.experiments.style import PUBLISHED_GROUPS, TICK_LABEL_GAP

    payload = _judges_payload(study, SEVEN)
    figure = hf.build_figure(payload)
    try:
        axes, bars, labels = _figure_parts(figure)
        assert [tick.get_text() for tick in axes.get_xticklabels()] == [
            "Olmo-3-32B", "Gemma-4-31B*", "GPT-OSS-120B*", "Muse-Glimmer-30B", "Qwen3-8B*", "GPT-5.6 Luna*",
            "Claude Haiku 5.5"]
        assert _legend_rows(figure) == [["flat", "hierarchy", SIGNIFICANCE_KEY]]
        width, height = figure.get_size_inches()
        assert width >= hf.PANEL_SIZE[0] * len(SEVEN) / PUBLISHED_GROUPS and height == hf.PANEL_SIZE[1]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        gap = TICK_LABEL_GAP * figure.dpi / 72
        for row in (axes.get_xticklabels(), labels):
            extents = sorted((artist.get_window_extent(renderer) for artist in row), key=lambda box: box.x0)
            assert all(right.x0 - left.x1 >= gap - 0.5 for left, right in zip(extents, extents[1:]))
        # Qwen3-8B's whisker reaches 0.99: the axis rises just enough to hold its labels, its ticks still end at 1.
        frame = axes.get_window_extent(renderer)
        assert axes.get_ylim()[1] > 1.0 and max(axes.get_yticks()) == pytest.approx(1.0)
        assert max(label.get_window_extent(renderer).y1 for label in labels) <= frame.y1 + 0.5
        _assert_labels_placed(figure, hf.drawn_judges(payload))
    finally:
        plt.close(figure)
    # Fewer than seven, the same figure keeps its axis at 1: the ceiling rises only for a label that would leave it.
    assert hf.build_figure(study["payload"]).axes[0].get_ylim() == (0.0, 1.0)
    plt.close("all")


def test_a_figure_whose_labels_need_more_room_widens_by_the_least_factor(study, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from ruleforge.experiments import style

    # A gap no font's labels leave at the proportional width, so the figure widens whatever fonts the machine has.
    monkeypatch.setattr(style, "TICK_LABEL_GAP", 60.0)
    figure = hf.build_figure(_judges_payload(study, SEVEN))
    try:
        axes, _, labels = _figure_parts(figure)
        width, height = figure.get_size_inches()
        assert width > hf.PANEL_SIZE[0] * len(SEVEN) / style.PUBLISHED_GROUPS and height == hf.PANEL_SIZE[1]
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        gaps = []
        for row in (axes.get_xticklabels(), labels):
            extents = sorted((artist.get_window_extent(renderer) for artist in row), key=lambda box: box.x0)
            gaps += [right.x0 - left.x1 for left, right in zip(extents, extents[1:])]
        assert min(gaps) == pytest.approx(60.0 * figure.dpi / 72, abs=1.0)
    finally:
        plt.close(figure)


def test_the_caption_notes_carry_what_the_figure_leaves_out(study):
    judges = hf.drawn_judges(study["payload"])
    notes = hf.caption_notes(study["payload"])
    assert notes.startswith("## The figure's caption notes (not drawn)\n") and notes.endswith(".\n")
    assert (f"\n\nThe published versions' title and subtitle: \"{hf.PUBLISHED_TITLE}\". {hf.PUBLISHED_SUBTITLE}.\n\n"
            in notes)
    assert (hf.PUBLISHED_TITLE, hf.PUBLISHED_SUBTITLE) == (
        "Inconsistency Rate With the Model Spec's Hierarchy Shown",
        "Flat: rules judged alone. Hierarchy: levels shown, each rule judged with the rules that outrank it")
    assert ("\n\nBars: each arm's inconsistency rate with its 95% scenario-cluster bootstrap interval. An asterisk after "
            "a judge's name marks a paired difference hierarchy - flat that is Holm-significant (alpha 0.05, Holm over "
            "the 2 judges drawn); a name without one, a difference that is not. Per judge, the difference with its 95% "
            "scenario-cluster bootstrap interval and Holm-adjusted p (scenario-clustered paired randomization test):"
            "\n\n") in notes
    for record, tick in zip(judges, STUDY_TICKS, strict=True):
        difference = record["primary"]["difference"]
        assert (f"- {tick}: {hf.difference_label(difference)} "
                f"(n = {record['n_scenarios']} scenarios x {record['n_decoder_draws']} draws)") in notes
    assert notes.rstrip("\n").endswith(hf.stack_note(judges) + ".")
    gemma, luna = judges
    assert [hf.flat_control(record) for record in judges] == ["rerun", "provider"]
    rerun = "Flat arm re-run beside the hierarchy arm, on its inference stack"
    provider = ("Flat arm the API judge's published naive arm, at the provider (a provider change since is not "
                "observable)")
    assert hf.stack_note([gemma]) == f"{rerun}: Gemma-4-31B"
    assert hf.stack_note([luna]) == f"{provider}: GPT-5.6 Luna"
    assert notes.rstrip("\n").endswith(f"{rerun}: Gemma-4-31B. {provider}: GPT-5.6 Luna.")
    published = json.loads(json.dumps(gemma))
    published.update(judge="muse-glimmer-30b")
    published["flat"]["root"] = "artifact://sampling/its-published-naive-arm"
    assert hf.flat_control(published) == "published"
    assert hf.stack_note([published]) == (
        "Flat arm the judge's published naive arm, on the hierarchy arm's inference stack: Muse-Glimmer-30B")
    olmo = {**gemma, "judge": "Olmo3-32B"}
    assert hf.stack_note([olmo, gemma, published, luna]) == ". ".join([
        f"{rerun}: Olmo-3-32B, Gemma-4-31B",
        "Flat arm the judge's published naive arm, on the hierarchy arm's inference stack: Muse-Glimmer-30B",
        f"{provider}: GPT-5.6 Luna"])
    assert hf.difference_label({"estimate": 0.01, "ci_low": -0.02, "ci_high": 0.04, "p_holm": 0.5}) == (
        "+0.010 [-0.020, +0.040], p(Holm) = 0.5")
    assert "marker" not in notes and "filled" not in notes and "hollow" not in notes and "†" not in notes


@pytest.fixture
def driver():
    from ruleforge.experiments.hierarchy import build as build_hierarchy

    return build_hierarchy


def _rules_and_levels(env):
    return ["--rules", str(env["base"] / "rules.jsonl"), "--levels", str(env["base"] / "rules_full.jsonl")]


def _judges():
    return [item for pair in pairs() for item in ("--judge", *pair.flat, *pair.hierarchy)]


def write_pairs(path, the_pairs):
    """A pair map of ``the_pairs``, in their order."""
    path.write_text(json.dumps({"version": 1, "judges": [
        {side: {"root": root, "condition": condition} for side, (root, condition) in
         (("flat", pair.flat), ("hierarchy", pair.hierarchy))} for pair in the_pairs]}), encoding="utf-8")
    return path


def test_the_producer_writes_what_it_is_asked_from_a_pair_map_or_the_command_line(driver, env, tmp_path, capsys,
                                                                                   monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")  # the PDF's creation date, so two renders compare
    out = tmp_path / "out"
    pairs_file = write_pairs(tmp_path / "pairs.yaml", pairs())
    outputs = ["--golden", str(out / "results" / "golden.json"), "--csv", str(out / "a.csv"), "--summary",
               str(out / "a.md"), "--figure", str(out / "a.pdf"), "--figure", str(out / "a.png"), "--dpi", "40"]
    assert driver.main(["--pairs", str(pairs_file), *_rules_and_levels(env), *outputs]) == 0
    golden_order = json.loads(json.dumps(env["payload"], sort_keys=True))
    text = ha.summary_markdown(golden_order) + "\n" + hf.caption_notes(golden_order)
    assert capsys.readouterr().out == text == (out / "a.md").read_text()
    written = (out / "results" / "golden.json").read_text()
    assert written == json.dumps(env["payload"], indent=2, sort_keys=True) + "\n"
    assert (out / "a.csv").read_text().splitlines()[0] == ",".join(ha.CSV_COLUMNS)
    assert len((out / "a.csv").read_text().splitlines()) == 1 + len(ha.csv_rows(env["payload"]))
    assert driver.main([*_judges(), *_rules_and_levels(env), "--golden", str(tmp_path / "again.json")]) == 0
    assert (tmp_path / "again.json").read_text() == written
    assert sorted(path.name for path in tmp_path.iterdir()) == ["again.json", "out", "pairs.yaml"]
    monkeypatch.delenv("RULEFORGE_ARTIFACT_ROOT")
    capsys.readouterr()
    again = tmp_path / "again"
    assert driver.main(["--from_golden", str(out / "results" / "golden.json"), "--figure", str(again / "a.pdf"),
                        "--figure", str(again / "a.png"), "--csv", str(again / "a.csv"), "--summary",
                        str(again / "a.md"), "--dpi", "40"]) == 0
    for name in ("a.pdf", "a.png", "a.csv", "a.md"):
        assert (again / name).read_bytes() == (out / name).read_bytes(), name
    assert capsys.readouterr().out == text


@pytest.mark.parametrize(("document", "message"), [
    ("[]", "must hold exactly"),
    ('{"version": 2, "judges": [{}]}', "must hold exactly"),
    ('{"version": 1, "judges": []}', "must hold exactly"),
    ('{"version": 1, "judges": [{"flat": {"root": "a", "condition": "b"}}]}', "judge 0 must name its flat and"),
    ('{"version": 1, "judges": [{"flat": {"root": "a"}, "hierarchy": {"root": "a", "condition": "b"}}]}',
     "judge 0 must name"),
    ('{"version": 1, "judges": [{"flat": {"root": "a", "condition": 1}, "hierarchy": {"root": "a", "condition": "b"}}]}',
     "judge 0 must name"),
    ('{"version": 1, "judges": ["x"]}', "judge 0 must name"),
    ("{unclosed", "unreadable"),
])
def test_a_pair_map_that_is_not_one_is_refused(tmp_path, document, message):
    path = tmp_path / "pairs.yaml"
    path.write_text(document, encoding="utf-8")
    with pytest.raises(ha.HierarchyError, match=message):
        ha_pairs(path)
    with pytest.raises(ha.HierarchyError, match="unreadable"):
        ha_pairs(tmp_path / "absent.yaml")


def ha_pairs(path):
    from ruleforge.experiments.hierarchy.build import load_pairs

    return load_pairs(path)


def test_the_paper_s_pair_map_names_six_judges_by_their_public_ids(repo_root):
    the_pairs = ha_pairs(repo_root / "configs" / "paper" / "hierarchy_pairs.yaml")
    assert [pair.flat[1].split("/")[0] for pair in the_pairs] == [
        "Olmo-3-32B-Think", "gemma-4-31B-it", "gpt-oss-120b", "Muse-Glimmer-30B", "claude-haiku-5-5", "gpt-5.6-luna"]
    for pair in the_pairs:
        model = pair.flat[1].split("/")[0]
        assert (pair.flat[1], pair.hierarchy[1]) == (f"{model}/{NAIVE_RULES_ABOVE}", f"{model}/{NAIVE_HIERARCHY}")
        assert all(root.startswith("artifact://sampling/") for root in (pair.flat[0], pair.hierarchy[0]))


def test_the_producer_refuses_a_golden_of_another_kind_a_missing_file_and_a_refused_build(driver, env, tmp_path,
                                                                                          capsys):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"analysis": "hierarchy_ladder", "schema_version": ha.SCHEMA_VERSION}))
    old = tmp_path / "old.json"
    old.write_text(json.dumps({**env["payload"], "schema_version": 1}))
    broken = tmp_path / "broken.json"
    broken.write_text("{")
    levels = write_levels(tmp_path / "full.jsonl", levels={**STUDY_LEVELS, 1: "platform"})
    refused = {
        "is not a hierarchy payload of schema 2": ["--from_golden", str(other)],
        f"{old} is not a hierarchy payload": ["--from_golden", str(old)],
        "cannot be read as JSON": ["--from_golden", str(broken)],
        "--from_golden re-renders an existing golden": ["--from_golden", str(old), "--golden", str(tmp_path / "g")],
        "do not also pass --allow-incomplete": ["--from_golden", str(old), "--allow-incomplete"],
        f"no such file: --levels {tmp_path / 'x'}": [*_judges(), *_rules_and_levels(env), "--levels",
                                                      str(tmp_path / "x")],
        "no level, or one not in": [*_judges(), *_rules_and_levels(env), "--levels", str(levels)],
        "not allowed with argument": [*_judges(), "--from_golden", str(old)],
        "pair map absent.yaml: unreadable": ["--pairs", str(tmp_path / "absent.yaml"), *_rules_and_levels(env)],
    }
    for message, arguments in refused.items():
        with pytest.raises(SystemExit):
            driver.main([*arguments, "--figure", str(tmp_path / "refused" / "f.png")])
        assert message in capsys.readouterr().err, message
    assert not (tmp_path / "refused").exists() and not (tmp_path / "g").exists()


def test_the_summary_discloses_a_campaign_s_cap_amendment_in_the_producers_words(study):
    from ruleforge.evaluate.estimates import amendment_sentence

    assert "## Disclosures" not in ha.summary_markdown(study["payload"])
    amendment = {"field": "max_semantic_attempts", "old_cap": 20, "new_cap": 30, "decided_by": "a reviewer",
                 "decided_on": "2026-10-28", "ledger": "an entry", "reason": "two bundles out of attempts",
                 "bundles": [{"condition_id": f"{GEMMA}/{HIER_STYLE}", "decoder_draw_id": 1, "scenario_id": "f" * 64,
                              "outcome": "accepted", "attempts": 23}],
                 "code_states": {f"{GEMMA}/{HIER_STYLE}": {"clean": ["draw-001"]}}}
    payload = json.loads(json.dumps(study["payload"]))
    payload["provenance"]["campaigns"][0]["amendments"] = [amendment]
    campaign_id = payload["provenance"]["campaigns"][0]["campaign_id"]
    text = ha.summary_markdown(payload)
    assert text.endswith(f"\n## Disclosures\n\n- {amendment_sentence(campaign_id, amendment)}\n")
