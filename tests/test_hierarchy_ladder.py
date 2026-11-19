from __future__ import annotations

import itertools
import json
import shutil

import pytest

from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.evaluate.estimates import amendment_sentence
from ruleforge.evaluate.inference import INFERENCE_METHOD, PRIMARY_INTERVAL_BASIS, grid_from_cells, hierarchical_bootstrap
from ruleforge.evaluate.scoring import PHI_MAP, calculate_inconsistency
from ruleforge.experiments.goldens import interval_record
from ruleforge.experiments.hierarchy import analysis as ha
from ruleforge.experiments.hierarchy import build_ladder
from ruleforge.experiments.hierarchy import figure as hf
from ruleforge.experiments.hierarchy import ladder as hl
from ruleforge.experiments.hierarchy import ladder_figure as lf
from ruleforge.experiments.hierarchy.analysis import HierarchyError, JudgePair
from ruleforge.rulesets.levels import LEVEL_ORDER, load_rule_levels
from tests.test_hierarchy import build_study, bundles, registered, write_pairs
from tests.test_hierarchy import pairs as hierarchy_pairs
from tests.test_leniency_campaign import N, PERMUTATIONS, S, artifact_root

LEVELS = {"r": "root", "s": "system", "d": "developer", "u": "user", "g": "guideline"}
RANK = {rule: LEVEL_ORDER.index(level) for rule, level in LEVELS.items()}
#: The study's judges in paper order, and each one's two arms.
JUDGES = ("gemma4-31b", "gpt-5.6-luna")
SIDES = ("flat", "hierarchy")


def by_hand(full, labels, k):
    """Rung k by the paragraph's definition: inconsistent iff phi(F) lies outside [M_k, M]."""
    phi = {rule: PHI_MAP[label] for rule, label in labels.items()}
    m_k = max(value for rule, value in phi.items() if RANK[rule] < k)
    return not (m_k <= PHI_MAP[full] <= max(phi.values()))


def unsure_by_hand(full, labels, k):
    """A miss at rung k whose top-k maximum is unsure: the full set says no and no top-k rule says yes alone."""
    top = [label for rule, label in labels.items() if RANK[rule] < k]
    return full == "no" and "unsure" in top and "yes" not in top


@pytest.mark.parametrize(("full", "labels", "rungs"), [
    # An over-flag: no rule alone flags, so it counts at every rung.
    ("yes", dict(r="no", s="no", d="no", u="no", g="no"), (1, 1, 1, 1, 1)),
    # A miss only a guideline rule flags: excused until the guideline level is considered.
    ("no", dict(r="no", s="no", d="no", u="no", g="yes"), (0, 0, 0, 0, 1)),
    # A miss of a root rule counts from the first rung.
    ("no", dict(r="yes", s="no", d="no", u="no", g="yes"), (1, 1, 1, 1, 1)),
    # A full-set flag only a guideline rule supports is consistent at every rung.
    ("yes", dict(r="no", s="no", d="no", u="no", g="yes"), (0, 0, 0, 0, 0)),
    # unsure is 0.5: an unsure root rule does not exceed an unsure full set; a user rule's yes does.
    ("unsure", dict(r="unsure", s="no", d="no", u="yes", g="no"), (0, 0, 0, 1, 1)),
    ("no", dict(r="no", s="no", d="unsure", u="no", g="no"), (0, 0, 1, 1, 1)),
])
def test_a_cell_s_ladder_by_hand(full, labels, rungs):
    assert tuple(map(int, hl.cell_ladder(full, labels, RANK).inconsistent)) == rungs


@pytest.mark.parametrize(("full", "labels", "unsure"), [
    # A system rule judged unsure against a no full set: an unsure miss from the system rung on.
    ("no", dict(r="no", s="unsure", d="no", u="no", g="no"), (0, 1, 1, 1, 1)),
    # Until a user rule's yes lifts M_k to 1: from there the miss is no longer an unsure one.
    ("no", dict(r="no", s="unsure", d="no", u="yes", g="no"), (0, 1, 1, 0, 0)),
    # An unsure full set misses nothing an unsure rule says.
    ("unsure", dict(r="unsure", s="unsure", d="no", u="no", g="no"), (0, 0, 0, 0, 0)),
])
def test_a_cell_s_unsure_misses_by_hand(full, labels, unsure):
    assert tuple(map(int, hl.cell_ladder(full, labels, RANK).under_unsure)) == unsure


def test_every_labelling_of_one_rule_per_level_meets_the_definition():
    for full, *labels in itertools.product(PHI_MAP, repeat=1 + len(LEVELS)):
        judgments = dict(zip(LEVELS, labels))
        ladder = hl.cell_ladder(full, judgments, RANK)
        rungs = ladder.inconsistent
        assert rungs == tuple(by_hand(full, judgments, k) for k in hl.RUNGS)
        assert list(rungs) == sorted(rungs)
        assert rungs[-1] == bool(calculate_inconsistency(full, labels)["contradiction"])
        assert ladder.over == (PHI_MAP[full] > max(PHI_MAP[label] for label in labels))
        assert ladder.under_unsure == tuple(unsure_by_hand(full, judgments, k) for k in hl.RUNGS)


def test_a_rung_is_named_by_the_level_it_adds():
    assert [hl.rung_label(LEVEL_ORDER[:k]) for k in hl.RUNGS] == [
        "root", "+system", "+developer", "+user", "+guideline"]


# A synthetic study: the hierarchy comparison's roots, each judge's flat arm and hierarchy arm read as that
# comparison reads them.


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    base = tmp_path_factory.mktemp("ladder")
    with artifact_root(base):
        roots = build_study(base)
        payload = hl.build_payload(pairs=hierarchy_pairs(), rules_path=base / "rules.jsonl",
                                   levels_path=base / "rules_full.jsonl")
        comparison = ha.build_payload(pairs=hierarchy_pairs(), rules_path=base / "rules.jsonl",
                                      levels_path=base / "rules_full.jsonl")
    return {"base": base, "roots": roots, "payload": payload, "comparison": comparison}


def _build(base, the_pairs=None):
    with artifact_root(base):
        return hl.build_payload(pairs=hierarchy_pairs() if the_pairs is None else the_pairs,
                                rules_path=base / "rules.jsonl", levels_path=base / "rules_full.jsonl")


def _copy(study, tmp_path):
    shutil.copytree(study["base"], tmp_path / "study")
    return tmp_path / "study"


def _refusal(error):
    where, checks = str(error.value).split(": ", 1)
    return where, checks.split("; ")


def _where(arm):
    return f"{arm['judge']} {arm['arm']} ({arm['condition_id']} in {arm['root']})"


def _arm(study, judge, side):
    [arm] = [arm for arm in study["payload"]["arms"] if (arm["judge"], arm["arm"]) == (judge, side)]
    return arm


def _root(base, arm):
    return base / "sampling" / arm["root"].rsplit("/", 1)[1]


def _hand_grids(root, condition_dir, rank):
    rows = bundles(root, condition_dir)
    ids = json.loads((root / "scenario_manifest.json").read_text())["scenario_ids"]
    flags = {name: {} for name in (*(f"rung_{k}" for k in hl.RUNGS), *(f"under_unsure_{k}" for k in hl.RUNGS),
                                   "over")}
    for i, s in itertools.product(range(N), range(S)):
        full = PHI_MAP[rows[i, s]["full_set_judgment"]]
        phi = {rule: PHI_MAP[label] for rule, label in rows[i, s]["rule_judgments"].items()}
        for k in hl.RUNGS:
            m_k = max((value for rule, value in phi.items() if rank[rule] < k), default=0.0)
            flags[f"rung_{k}"][(ids[i], s)] = float(not m_k <= full <= max(phi.values()))
            flags[f"under_unsure_{k}"][(ids[i], s)] = float(full < m_k == 0.5)
        flags["over"][(ids[i], s)] = float(full > max(phi.values()))
    return {name: grid_from_cells(condition_dir, name, ids, S, cells) for name, cells in flags.items()}


def test_each_judge_s_flat_and_hierarchy_arms_are_recounted_from_their_own_bundles(study):
    base, payload = study["base"], study["payload"]
    assert [(arm["judge"], arm["arm"]) for arm in payload["arms"]] == [
        (judge, side) for judge in JUDGES for side in SIDES]
    rank = load_rule_levels(base / "rules.jsonl", base / "rules_full.jsonl").rank
    bootstrap = payload["provenance"]["bootstrap"]
    last = f"rung_{hl.RUNGS[-1]}"
    records = {record["judge"]: record for record in study["comparison"]["judges"]}
    for arm in payload["arms"]:
        record = records[arm["judge"]]
        condition_dir = registered(record, arm["arm"])
        grids = _hand_grids(study["roots"][arm["root"].rsplit("/", 1)[1]], condition_dir, rank)
        summary = json.loads((_root(base, arm) / "aggregate" / "estimates.json").read_text())["conditions"][
            condition_dir]
        # Every rung but the last, the unsure misses and the over-flag are the hand grids' rates under one bootstrap;
        # the last rung is the arm's estimates' rate and interval.
        assert grids[last].rate == summary["estimate"]
        boot = hierarchical_bootstrap({name: grid for name, grid in grids.items() if name != last},
                                      shared_inner=[[name for name in grids if name != last]], **bootstrap)
        intervals = {name: interval_record(entry) for name, entry in boot["conditions"].items()}
        intervals[last] = interval_record(summary)
        assert (arm["n_scenarios"], arm["n_decoder_draws"]) == (N, S)
        assert [rung["rate"] for rung in arm["rungs"]] == [intervals[f"rung_{k}"] for k in hl.RUNGS]
        assert [rung["under_unsure"] for rung in arm["rungs"]] == [intervals[f"under_unsure_{k}"] for k in hl.RUNGS]
        assert arm["over"] == intervals["over"]
        side = record[arm["arm"]]
        assert arm["rungs"][-1]["rate"] == record["primary"][f"{arm['arm']}_rate"]
        assert (arm["condition_id"], arm["grid_digest"], arm["prompt_style"], arm["individual_rule_context"]) == (
            side["condition_id"], side["grid_digest"], *ha.SIDES[arm["arm"]])
    assert any(arm["rungs"][0]["rate"]["estimate"] < arm["rungs"][-1]["rate"]["estimate"] for arm in payload["arms"])
    assert any(rung["under_unsure"]["estimate"] > 0 for arm in payload["arms"] for rung in arm["rungs"])


def test_the_payload_names_its_levels_rungs_and_pairs_and_shares_the_comparison_s_sources(study):
    payload = study["payload"]
    assert payload["definition"] == hl.DEFINITION and f"{hl.LAST_RUNG.capitalize()} is the arm's" in hl.DEFINITION
    assert [rung["levels"] for rung in payload["rungs"]] == [list(LEVEL_ORDER[:k]) for k in hl.RUNGS]
    assert payload["levels"]["order"] == list(LEVEL_ORDER)
    provenance, comparison = payload["provenance"], study["comparison"]["provenance"]
    assert provenance["pairs"] == comparison["pairs"] and provenance["allow_incomplete"] is False
    for key in ("ruleset", "levels", "approved_prompts_sha256", "campaigns", "scenario_manifest_sha256", "n_scenarios",
                "n_decoder_draws", "inference_method", "interval_basis"):
        assert provenance[key] == comparison[key], key
    assert provenance["bootstrap"] == {key: value for key, value in comparison["bootstrap"].items()
                                       if key != "n_permutations"}


def test_a_pair_the_comparison_refuses_is_refused_in_its_words(study):
    swapped = [JudgePair(flat=pair.hierarchy, hierarchy=pair.flat) for pair in hierarchy_pairs()[1:]]
    levels_sha = load_rule_levels(study["base"] / "rules.jsonl", study["base"] / "rules_full.jsonl").levels_file["sha256"]
    with pytest.raises(HierarchyError) as refused:
        _build(study["base"], swapped)
    flat, hierarchy = swapped[0].flat, swapped[0].hierarchy
    assert str(refused.value) == (
        f"gemma4-31b: flat {flat[1]} in {flat[0]}, hierarchy {hierarchy[1]} in {hierarchy[0]}: "
        "the flat arm gemma-4-31B-it/naive-hierarchy is 'naive-hierarchy' with its rules judged 'superiors', not "
        "('naive-rules-above', 'isolated'); the hierarchy arm gemma-4-31B-it/naive-rules-above is "
        "'naive-rules-above' with its rules judged 'isolated', not ('naive-hierarchy', 'superiors'); "
        "the hierarchy arm read levels file None..., not rules_full.jsonl (" + levels_sha[:12] + "...)")
    with pytest.raises(HierarchyError) as refused:
        _build(study["base"], [])
    assert _refusal(refused) == ("hierarchy ladder", ["no judge's arms given"])


def test_a_stored_grid_of_the_same_rate_but_other_cells_is_refused(study, tmp_path):
    base = _copy(study, tmp_path)
    arm = _arm(study, "gpt-5.6-luna", "flat")
    root = _root(base, arm)
    # Swap one scenario's contradiction between two draws: each scenario's mean, so the rate, stays the same.
    paths = [root / arm["condition_id"] / f"draw-{s:03d}" / "bundle_outcomes.jsonl" for s in (0, 1)]
    draws = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    k = next(k for k, (a, b) in enumerate(zip(*draws)) if a["contradiction"] != b["contradiction"])
    draws[0][k]["contradiction"], draws[1][k]["contradiction"] = draws[1][k]["contradiction"], draws[0][k]["contradiction"]
    for path, rows in zip(paths, draws):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with artifact_root(base):
        analyze_campaign(root, n_permutations=PERMUTATIONS)
    with pytest.raises(HierarchyError) as refused:
        _build(base)
    assert _refusal(refused) == (
        _where(arm), [f"{hl.LAST_RUNG} is not its stored inconsistency (the stored contradiction) cell for cell"])


def test_an_estimate_other_than_the_stored_grid_s_rate_is_refused(study, tmp_path):
    base = _copy(study, tmp_path)
    arm = _arm(study, "gpt-5.6-luna", "hierarchy")
    path = _root(base, arm) / "aggregate" / "estimates.json"
    estimates = json.loads(path.read_text())
    stored = estimates["conditions"][arm["condition_id"]]
    rate, stored["estimate"] = stored["estimate"], stored["estimate"] + 0.125  # its grid_digest is left as it was
    path.write_text(json.dumps(estimates))
    with pytest.raises(HierarchyError) as refused:
        _build(base)
    assert _refusal(refused) == (
        _where(arm), [f"{hl.LAST_RUNG}'s rate {rate!r} is not its estimates' {stored['estimate']!r}"])


def test_an_interval_on_another_basis_or_another_inference_method_is_refused(study, tmp_path):
    base = _copy(study, tmp_path)
    arm = study["payload"]["arms"][0]
    [record] = [record for record in study["comparison"]["judges"] if record["judge"] == arm["judge"]]
    for path in sorted((base / "sampling").glob("*/aggregate/estimates.json")):
        estimates = json.loads(path.read_text())
        estimates["inference_method"] = "another_method"
        if path.parent.parent == _root(base, arm):
            estimates["conditions"][registered(record, arm["arm"])]["interval_basis"] = "two_stage"
        path.write_text(json.dumps(estimates))
    with pytest.raises(HierarchyError) as refused:
        _build(base)
    assert _refusal(refused) == (_where(arm), [
        f"its interval basis is 'two_stage', not {PRIMARY_INTERVAL_BASIS!r}",
        f"its campaign's inference method is 'another_method', not {INFERENCE_METHOD!r}"])


def test_the_csv_and_summary_render_from_the_written_json(study):
    payload = study["payload"]
    written = json.loads(json.dumps(payload, sort_keys=True))
    text = hl.summary_markdown(written)
    assert text == hl.summary_markdown(payload)
    assert "| Judge | Arm | root | +system | +developer | +user | +guideline | over-flag |\n" in text
    assert "| Judge | Arm | root | +system | +developer | +user | +guideline |\n" in text
    assert "## Disclosures" not in text
    rows = hl.csv_rows(written)
    assert len(rows) == len(payload["arms"]) * len(hl.RUNGS) and list(rows[0]) == list(hl.CSV_COLUMNS)
    first = payload["arms"][0]
    assert rows[0]["under_unsure_ci_high"] == first["rungs"][0]["under_unsure"]["ci_high"]
    assert rows[-1]["over"] == payload["arms"][-1]["over"]["estimate"]


def test_the_summary_discloses_a_campaign_s_cap_amendment(study):
    amendment = {"field": "max_semantic_attempts", "old_cap": 20, "new_cap": 30, "decided_by": "a reviewer",
                 "decided_on": "2026-10-28", "ledger": "an entry", "reason": "two bundles out of attempts",
                 "bundles": [{"condition_id": study["payload"]["arms"][0]["condition_id"], "decoder_draw_id": 1,
                              "scenario_id": "f" * 64, "outcome": "accepted", "attempts": 23}],
                 "code_states": {study["payload"]["arms"][0]["condition_id"]: {"clean": ["draw-001"]}}}
    payload = json.loads(json.dumps(study["payload"]))
    payload["provenance"]["campaigns"][0]["amendments"] = [amendment]
    campaign_id = payload["provenance"]["campaigns"][0]["campaign_id"]
    assert hl.summary_markdown(payload).endswith(
        f"\n## Disclosures\n\n- {amendment_sentence(campaign_id, amendment)}\n")


def _figure(payload):
    pytest.importorskip("matplotlib")
    return lf.build_figure(json.loads(json.dumps(payload, sort_keys=True)))


def _one_side(payload, side):
    return {**payload, "arms": [arm for arm in payload["arms"] if arm["arm"] == side]}


def test_the_figure_has_a_panel_per_judge_a_key_for_two_arms_and_no_title(study):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    figure = _figure(study["payload"])
    try:
        panels = [axes for axes in figure.axes if axes.get_visible()]
        assert len(panels) == len(figure.axes) == len(JUDGES)
        ticks = panels[0].get_xticklabels()
        assert [text.get_text() for text in ticks] == ["root", "+system", "+developer", "+user", "+guideline"]
        assert {text.get_rotation() for text in ticks} == {lf.TICK_ROTATION}
        assert figure.get_supxlabel() == lf.X_LABEL == "Hierarchy levels considered for inconsistency"
        assert figure.get_supylabel() == lf.Y_LABEL == "Inconsistency rate"
        assert figure.get_suptitle() == "" and all(axes.get_ylim() == (0.0, 1.0) for axes in panels)
        [legend] = figure.legends
        assert [text.get_text() for text in legend.get_texts()] == list(SIDES) and legend.get_frame_on()
        assert [container.lines[0].get_marker() for container in panels[0].containers] == ["o", "s"]
        xs = [container.lines[0].get_xdata()[0] for container in panels[0].containers]
        assert xs == pytest.approx([1 - lf.DODGE / 2, 1 + lf.DODGE / 2])
    finally:
        plt.close(figure)


@pytest.mark.parametrize(("side", "marker"), [("hierarchy", "s"), ("flat", "o")])
def test_a_figure_of_one_arm_draws_it_in_the_comparison_s_colour_without_a_key(study, side, marker):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    figure = _figure(_one_side(study["payload"], side))
    try:
        assert not figure.legends and len(figure.axes) == len(JUDGES)
        assert {(to_hex(container.lines[0].get_color()), container.lines[0].get_marker())
                for axes in figure.axes for container in axes.containers} == {(to_hex(hf.ARMS[side][1][1]), marker)}
        assert lf.ARM_STYLES[side] == (hf.ARMS[side][1][1], marker)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(("judges", "shape", "width"),
                         [(2, (1, 2), 6.8), (5, (2, 3), 10.2), (6, (2, 3), 10.2), (7, (2, 4), 10.2)])
def test_the_judges_fill_rows_of_at_most_four_split_evenly_no_wider_than_three_panels(study, judges, shape, width):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    first = _one_side(study["payload"], "hierarchy")["arms"][:1]
    names = ("Olmo3-32B", "gemma4-31b", "gpt-oss-120b", "muse-glimmer-30b", "Qwen3-8B", "gpt-5.6-luna",
             "claude-haiku-5-5")[:judges]
    figure = _figure({**study["payload"], "arms": [{**arm, "judge": name} for name in names for arm in first]})
    try:
        assert figure.axes[0].get_gridspec().get_geometry() == shape
        assert sum(axes.get_visible() for axes in figure.axes) == judges
        assert figure.get_size_inches()[0] == pytest.approx(width)
    finally:
        plt.close(figure)


def _main(study, tmp_path, *words):
    base = study["base"]
    pair_map = write_pairs(tmp_path / "pairs.yaml", hierarchy_pairs())
    with artifact_root(base):
        return build_ladder.main(["--pairs", str(pair_map), "--rules", str(base / "rules.jsonl"),
                                  "--levels", str(base / "rules_full.jsonl"), *map(str, words)])


def _recorded_draws(monkeypatch):
    drawn, original = [], lf.draw

    def draw(payload, out, **kwargs):
        drawn.append(payload)
        return original(payload, out, **kwargs)

    monkeypatch.setattr(lf, "draw", draw)
    return drawn


def test_the_program_builds_the_golden_csv_summary_and_one_arm_s_figure_from_the_pair_map(study, tmp_path, capsys,
                                                                                          monkeypatch):
    pytest.importorskip("matplotlib")
    drawn = _recorded_draws(monkeypatch)
    out = tmp_path / "out"
    golden, table, summary, figure = (out / name for name in ("ladder.json", "ladder.csv", "ladder.md", "ladder.png"))
    assert _main(study, tmp_path, "--golden", golden, "--csv", table, "--summary", summary, "--arm", "hierarchy",
                 "--figure", figure, "--dpi", "40") == 0
    written = json.loads(golden.read_text())
    assert written == json.loads(json.dumps(study["payload"], sort_keys=True))
    assert summary.read_text() == hl.summary_markdown(written) == capsys.readouterr().out
    assert [[arm["arm"] for arm in payload["arms"]] for payload in drawn] == [["hierarchy"] * len(JUDGES)]
    assert {arm["arm"] for arm in written["arms"]} == set(SIDES)
    assert table.read_text().splitlines()[0] == ",".join(hl.CSV_COLUMNS)
    assert len(table.read_text().splitlines()) == 1 + len(hl.csv_rows(written))
    assert {row.split(",")[1] for row in table.read_text().splitlines()[1:]} == set(SIDES)
    assert figure.stat().st_size and build_ladder.load_golden(golden) == written


@pytest.mark.parametrize(("words", "arms"), [(["--arm", "flat"], ["flat"] * len(JUDGES)),
                                             ([], [side for _ in JUDGES for side in SIDES])])
def test_the_program_draws_the_arms_arm_names_or_every_arm_without_it(study, tmp_path, monkeypatch, words, arms):
    pytest.importorskip("matplotlib")
    drawn = _recorded_draws(monkeypatch)
    assert _main(study, tmp_path, *words, "--figure", tmp_path / "ladder.png", "--dpi", "40") == 0
    assert [[arm["arm"] for arm in payload["arms"]] for payload in drawn] == [arms]


def test_the_program_redraws_from_a_golden_and_prints_its_summary(study, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    golden = tmp_path / "ladder.json"
    golden.write_text(json.dumps(study["payload"]))
    assert build_ladder.main(["--from_golden", str(golden), "--figure", str(tmp_path / "ladder.png"),
                              "--dpi", "50"]) == 0
    assert (tmp_path / "ladder.png").stat().st_size and sorted(path.name for path in tmp_path.iterdir()) == [
        "ladder.json", "ladder.png"]
    assert capsys.readouterr().out == hl.summary_markdown(json.loads(golden.read_text()))


@pytest.mark.parametrize(("words", "message"), [
    (["--from_golden", "g.json", "--golden", "again.json"],
     "--from_golden re-renders an existing golden; do not also pass --golden"),
    (["--from_golden", "g.json", "--allow-incomplete"],
     "--from_golden re-renders an existing golden; do not also pass --allow-incomplete"),
    (["--from_golden", "other.json"], "other.json is not a hierarchy_ladder payload of schema 1"),
    (["--from_golden", "g.json", "--arm", "naive"], "argument --arm: invalid choice: "),
])
def test_the_program_refuses_a_re_render_that_reads_roots_or_another_golden(study, tmp_path, monkeypatch, capsys,
                                                                             words, message):
    (tmp_path / "g.json").write_text(json.dumps(study["payload"]))
    (tmp_path / "other.json").write_text(json.dumps({"analysis": "hierarchy", "schema_version": 1}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        build_ladder.main(words)
    assert exit_info.value.code == 2 and f"error: {message}" in capsys.readouterr().err.splitlines()[-1]


@pytest.mark.parametrize("flag", ["--rules", "--levels"])
def test_the_program_refuses_a_missing_rules_or_levels_file(study, tmp_path, capsys, flag):
    with pytest.raises(SystemExit) as exit_info:
        _main(study, tmp_path, flag, tmp_path / "missing.jsonl")
    assert exit_info.value.code == 2
    assert capsys.readouterr().err.splitlines()[-1].endswith(f"error: no such file: {flag} {tmp_path / 'missing.jsonl'}")
