from copy import deepcopy
import hashlib
import importlib
import json

import numpy as np
import pytest

from ruleforge.evaluate.estimates import public_amendment
from ruleforge.experiments.ruleforge import comparison as ruleset_comparison
from ruleforge.evaluate.inference import (
    DrawGrid,
    load_condition_outcomes,
    outcome_grid,
)
from ruleforge.experiments.ruleforge.comparison import (
    build_from_campaign,
    build_summary,
    expected_transition_shares,
)
from ruleforge.evaluate.aggregate import analyze_campaign
from ruleforge.scenarios.manifest import ScenarioManifest
from ruleforge.core.files import write_json_sidecar
from tests.test_sampling_ruleforge import run_ruleset_conditions, RULESETS

LABELS = dict(zip(RULESETS, ("Initial", "Winner", "Runner-up"), strict=True))


def _close(figure):
    import matplotlib.pyplot as plt

    plt.close(figure)


def grid(values, ids=("a", "b"), name="c"):
    return DrawGrid(name, "contradiction", ids, np.asarray(values))


def test_transition_shares_equal_all_cross_draw_pairs_not_matched_indices():
    before = grid([[1, 0], [1, 1]])
    after = grid([[1, 0], [0, 1]], name="d")
    shares = expected_transition_shares(before, after)
    pairs = [
        (p, q)
        for a, b in zip(before.values, after.values, strict=True)
        for p in a
        for q in b
    ]
    assert shares == pytest.approx(
        {
            "both_consistent": sum(p == q == 0 for p, q in pairs) / len(pairs),
            "fixed": sum(p == 1 and q == 0 for p, q in pairs) / len(pairs),
            "regressed": sum(p == 0 and q == 1 for p, q in pairs) / len(pairs),
            "both_inconsistent": sum(p == q == 1 for p, q in pairs) / len(pairs),
        }
    )
    assert shares["fixed"] != np.mean((before.values == 1) & (after.values == 0))
    assert sum(shares.values()) == pytest.approx(1)
    assert shares["regressed"] - shares["fixed"] == pytest.approx(
        after.rate - before.rate
    )
    assert expected_transition_shares(before, after.restricted_to(("b", "a"))) == shares
    assert (
        expected_transition_shares(before, grid(after.values[:, ::-1], name="d"))
        == shares
    )


@pytest.mark.parametrize(
    "other",
    [
        grid([[0, 1], [1, 0]], ids=("a", "c")),
        grid([[0], [1]]),
        grid([[0, 0.5], [1, 0]]),
    ],
)
def test_transition_rejects_wrong_population_single_draw_and_graded_values(other):
    with pytest.raises(ValueError):
        expected_transition_shares(grid([[0, 1], [1, 0]]), other)


@pytest.fixture
def campaign(tmp_path, monkeypatch, modelspec):
    monkeypatch.setenv("RULEFORGE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("RULEFORGE_MODEL_ROOT", str(tmp_path / "models"))
    root = run_ruleset_conditions(tmp_path, modelspec)
    estimates = analyze_campaign(root, n_permutations=100, n_resamples=100)
    path = root / "sampling_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["campaign_mode"] = "final"
    path.write_text(json.dumps(manifest))
    scenarios = ScenarioManifest.load(root / "scenario_manifest.json")
    grids = {}
    for cid in estimates["conditions"]:
        cells, _ = load_condition_outcomes(
            root / cid, expected_scenario_ids=scenarios.scenario_ids, expected_draws=2
        )
        grids[cid] = outcome_grid(
            cid, cells, scenario_ids=scenarios.scenario_ids, n_draws=2
        )
    labels = {
        cid: LABELS[row["extra"]["ruleset"]]
        for cid, row in estimates["conditions"].items()
    }
    return root, estimates, grids, labels


def test_producer_preserves_rates_reverses_intervals_and_keeps_holm(campaign):
    root, estimates, grids, labels = campaign
    result = build_from_campaign(root / "aggregate/estimates.json", LABELS)
    assert result["data_version"] == 6
    assert result["transition_units"] == "share"
    assert result["provenance"]["complete"]
    # The key says sha256, so it holds the whole file's digest, not a fingerprint.
    assert result["provenance"]["estimates"]["sha256"] == hashlib.sha256(
        (root / "aggregate/estimates.json").read_bytes()
    ).hexdigest()
    for cid, label in labels.items():
        assert result["rulesets"][label]["mean"] == grids[cid].rate
    contrasts = next(iter(estimates["ruleset_families"].values()))["contrasts"]
    for row in contrasts:
        key = f"{labels[row['b']]} - {labels[row['a']]}"
        delta = result["paired_deltas"][key]
        assert delta["mean"] == -row["estimate"]
        assert (delta["ci_low"], delta["ci_high"]) == (-row["ci_high"], -row["ci_low"])
        assert delta["p_holm"] == row["p_holm"]
        t = result["paired_transitions"][key]
        assert t["regressed"] - t["fixed"] == pytest.approx(delta["mean"])
    reversed_labels = dict(reversed(list(labels.items())))
    reversed_result = build_summary(estimates, grids, reversed_labels)
    row = contrasts[0]
    assert (
        reversed_result["paired_deltas"][
            f"{labels[row['a']]} - {labels[row['b']]} ".strip()
        ]["mean"]
        == row["estimate"]
    )


def test_the_estimates_read_are_recorded_by_the_digest_of_the_bytes_read(campaign, monkeypatch):
    path = campaign[0] / "aggregate/estimates.json"
    read = hashlib.sha256(path.read_bytes()).hexdigest()
    real = ruleset_comparison.build_summary

    def summary(*args, **kwargs):
        write_json_sidecar(path, {**json.loads(path.read_text()), "created_at": "2026-10-11T00:00:00Z"})
        return real(*args, **kwargs)

    monkeypatch.setattr(ruleset_comparison, "build_summary", summary)
    built = build_from_campaign(path, LABELS)
    assert built["provenance"]["estimates"]["sha256"] == read != hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        "state",
        "method",
        "labels",
        "family",
        "digest",
        "interval",
        "basis",
        "pairs",
        "p",
        "holm",
    ],
)
def test_summary_refuses_incomplete_stale_or_mislabelled_results(campaign, mutation):
    _, original, grids, labels = campaign
    estimates = deepcopy(original)
    row = next(iter(estimates["conditions"].values()))
    family = next(iter(estimates["ruleset_families"].values()))
    if mutation == "state":
        estimates["state"] = "in_progress"
    elif mutation == "method":
        estimates["inference_method"] = "mcnemar"
    elif mutation == "labels":
        labels = dict(list(labels.items())[:-1])
    elif mutation == "family":
        family["missing_rulesets"] = ["absent"]
    elif mutation == "digest":
        row["grid_digest"] = "wrong"
    elif mutation == "interval":
        row["ci_low"] = float("nan")
    elif mutation == "basis":
        row["interval_basis"] = "two_stage"
    elif mutation == "pairs":
        family["contrasts"] = family["contrasts"][:-1]
    elif mutation == "p":
        family["contrasts"][0]["p_value"] = 2
    elif mutation == "holm":
        family["contrasts"][0]["p_holm"] = -1
    with pytest.raises(ValueError):
        build_summary(estimates, grids, labels)


@pytest.mark.parametrize(
    "mutation",
    [
        "mode",
        "manifest",
        "population",
        "family",
        "labels",
        "fingerprint",
        "draw_fingerprint",
        "materialized",
        "outcome",
    ],
)
def test_campaign_loader_rejects_identity_drift(campaign, mutation):
    root, estimates, grids, labels = campaign
    cid = next(iter(labels))
    path = root / "sampling_manifest.json"
    content = json.loads(path.read_text())
    if mutation == "mode":
        content["campaign_mode"] = "pilot"
    elif mutation == "manifest":
        content["temperature"] = 0.5
    elif mutation == "population":
        content["expected_scenarios"] = 99
    elif mutation == "family":
        path = root / cid / "condition.json"
        content = json.loads(path.read_text())
        content["extra"]["ruleset_family"] += "+missing"
        from ruleforge.evaluate.identity import ConditionSpec

        spec = {
            k: v
            for k, v in content.items()
            if k not in ("fingerprint", "schema_version")
        }
        content["fingerprint"] = ConditionSpec.from_dict(spec).fingerprint
    elif mutation == "labels":
        with pytest.raises(ValueError):
            build_from_campaign(root / "aggregate/estimates.json", {"wrong": "Wrong"})
        return
    elif mutation == "fingerprint":
        path = root / "aggregate/estimates.json"
        content = estimates
        content["conditions"][cid]["condition_fingerprint"] = "wrong"
    elif mutation in ("draw_fingerprint", "materialized"):
        for d in range(2):
            target = root / cid / f"draw-{d:03d}/completion.json"
            c = json.loads(target.read_text())
            c[
                "condition_fingerprint"
                if mutation == "draw_fingerprint"
                else "materialized"
            ] = "wrong" if mutation == "draw_fingerprint" else False
            target.write_text(json.dumps(c))
    elif mutation == "outcome":
        target = root / cid / "draw-000/bundle_outcomes.jsonl"
        lines = target.read_text().splitlines()
        target.write_text("\n".join(lines[:-1]) + "\n")
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        build_from_campaign(root / "aggregate/estimates.json", LABELS)


def test_renderer_uses_shares_without_dividing_by_n_or_printing_counts(
    campaign, repo_root, tmp_path
):
    root, _, _, _ = campaign
    stats = build_from_campaign(root / "aggregate/estimates.json", LABELS)
    plot = importlib.import_module(PLOT)
    fig = plot.build_figure(stats, panel="b")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Expected share per draw"
    assert [p.get_width() for p in ax.patches] == pytest.approx(
        [-1 / 3, -1 / 3, 1 / 3, 0]
    )
    assert "33.3%" in [t.get_text() for t in ax.texts]
    assert all("(" not in t.get_text() for t in ax.texts if "%" in t.get_text())
    _close(fig)
    stats["transition_units"] = "count"
    with pytest.raises(ValueError, match="expected-share provenance"):
        plot.build_figure(stats, panel="b")


def test_ambiguous_display_labels_cannot_silently_overwrite_a_contrast():
    labels = dict(zip(("a", "b", "c", "d"), ("A", "B - A", "C - B", "C"), strict=True))
    estimates = {
        "state": "complete",
        "inference_method": "hierarchical_bootstrap_v2",
        "conditions": dict.fromkeys(labels),
    }
    with pytest.raises(ValueError, match="ambiguous contrast keys"):
        build_summary(estimates, dict.fromkeys(labels), labels)


def test_declared_but_absent_ruleset_cannot_shrink_the_holm_family(campaign):
    from ruleforge.evaluate.identity import ConditionSpec

    root, _, _, labels = campaign
    for cid in labels:
        path = root / cid / "condition.json"
        content = json.loads(path.read_text())
        content["extra"]["ruleset_family"] += "+uncollected"
        spec = ConditionSpec.from_dict(
            {
                k: v
                for k, v in content.items()
                if k not in ("fingerprint", "schema_version")
            }
        )
        content["fingerprint"] = spec.fingerprint
        path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="declared ruleset family is incomplete"):
        build_from_campaign(root / "aggregate/estimates.json", LABELS)


def _report_a_stack_mismatch(campaign):
    """The campaign's aggregate as analysed under allow_stack_mismatch: (its path, the one mismatch it reports)."""
    root, _, _, labels = campaign
    path = root / "aggregate/estimates.json"
    first, second = sorted(labels)[:2]
    record = {"scope": "ruleset_contrasts", "conditions": [first, second],
              "reason": "vllm 0.19 / torch 2.10 / transformers 5.5 against vllm 0.28 / torch 2.13 / transformers 5.17"}
    payload = json.loads(path.read_text())
    payload.update(allow_stack_mismatch=True, stack_mismatches=[record])
    path.write_text(json.dumps(payload))
    return path, record


BUILD, PLOT = "ruleforge.experiments.ruleforge.build", "ruleforge.experiments.ruleforge.plot"


def test_an_aggregate_reporting_a_stack_mismatch_is_built_only_on_the_record(campaign, repo_root, tmp_path, monkeypatch):
    root = campaign[0]
    clean = build_from_campaign(root / "aggregate/estimates.json", LABELS)
    assert "stack_mismatches" not in clean["provenance"]
    assert all("inference_stacks" not in row for row in clean["rulesets"].values())
    path, record = _report_a_stack_mismatch(campaign)
    with pytest.raises(ValueError, match=r"reports 1 inference-stack mismatch\(es\), analysed under allow_stack_mismatch"):
        build_from_campaign(path, LABELS)
    assert build_from_campaign(path, LABELS, allow_stack_mismatch=True)["provenance"]["stack_mismatches"] == [record]
    cli = importlib.import_module(BUILD)
    golden = tmp_path / "ruleforge.json"
    argv = ["build_ruleset_campaign.py", "--estimates", str(path), "--golden", str(golden),
            *[f"--ruleset={tag}={label}" for tag, label in LABELS.items()]]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="inference-stack mismatch"):
        cli.main()
    monkeypatch.setattr("sys.argv", [*argv, "--allow-stack-mismatch"])
    cli.main()
    assert json.loads(golden.read_text())["provenance"]["stack_mismatches"] == [record]


def test_a_golden_naming_a_stack_mismatch_is_drawn_only_on_the_record(campaign, repo_root, tmp_path, monkeypatch):
    path, _ = _report_a_stack_mismatch(campaign)
    stats = build_from_campaign(path, LABELS, allow_stack_mismatch=True)
    plot = importlib.import_module(PLOT)
    with pytest.raises(ValueError, match=r"the RuleForge golden reports 1 inference-stack mismatch\(es\)"):
        plot.build_figure(stats, panel="b")
    _close(plot.build_figure(stats, panel="b", allow_stack_mismatch=True))
    cli = importlib.import_module(BUILD)
    golden, built = tmp_path / "ruleforge.json", tmp_path / "built_b.png"
    monkeypatch.setattr("sys.argv", [
        "build_ruleset_campaign.py", "--estimates", str(path), "--golden", str(golden), "--figure-b", str(built),
        *[f"--ruleset={tag}={label}" for tag, label in LABELS.items()], "--allow-stack-mismatch",
    ])
    cli.main()
    assert built.is_file()
    redrawn = tmp_path / "redrawn_b.png"
    argv = ["plot.py", "--from_golden", str(golden), "--figure_b", str(redrawn)]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit):
        plot.main()
    assert not redrawn.exists()
    monkeypatch.setattr("sys.argv", [*argv, "--allow-stack-mismatch"])
    plot.main()
    assert redrawn.is_file()


def test_an_amended_root_s_frozen_cap_is_recorded_with_its_amendment(campaign):
    from tests.test_campaign_estimates import record

    root, _, _, _ = campaign
    plain = build_from_campaign(root / "aggregate/estimates.json", LABELS)["provenance"]
    assert "amendments" not in plain
    frozen = plain["max_semantic_attempts"]
    disclosed = {"old_cap": frozen, "new_cap": 50, "bundles": []}
    for name, fields in (("sampling_manifest.json", {"max_semantic_attempts": 50,
                                                     "amendments": [record(old_cap=frozen, new_cap=50)]}),
                         ("aggregate/estimates.json", {"max_semantic_attempts": 50, "amendments": [disclosed]})):
        path = root / name
        path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))
    provenance = build_from_campaign(root / "aggregate/estimates.json", LABELS)["provenance"]
    assert (provenance["max_semantic_attempts"], provenance["amendments"]) == (frozen, [public_amendment(disclosed)])


def test_a_root_spelling_its_style_by_the_recorded_key_gives_the_public_golden(campaign):
    from ruleforge.core import record_names

    root, estimates, _, _ = campaign
    path = root / "aggregate/estimates.json"
    public = build_from_campaign(path, LABELS)
    key = record_names.identity_key("naive-rules-above")
    assert key != "naive-rules-above"
    for condition in root.glob("gemma/figure4/*/condition.json"):
        payload = json.loads(condition.read_text())
        assert payload["prompt_style"] == "naive-rules-above"
        condition.write_text(json.dumps({**payload, "prompt_style": key}))
    recorded_estimates = deepcopy(estimates)
    for row in recorded_estimates["conditions"].values():
        row["prompt_style"] = key
    path.write_text(json.dumps(recorded_estimates))
    recorded = build_from_campaign(path, LABELS)
    assert {row["prompt_style"] for row in recorded["rulesets"].values()} == {"naive-rules-above"}
    assert sorted(row["condition_id"] for row in recorded["rulesets"].values()) == sorted(estimates["conditions"])
    for result in (public, recorded):
        del result["provenance"]["estimates"]["sha256"]
    assert recorded == public
    assert {row["prompt_style"] for row in build_summary(
        recorded_estimates, campaign[2], campaign[3])["rulesets"].values()} == {key}
