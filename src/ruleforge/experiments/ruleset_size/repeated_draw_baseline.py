from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ruleforge.core.errors import RuleForgeError
from ruleforge.core.paths import resolve, to_uri
from ruleforge.evaluate.bank import (
    BANK_ROOT_KEY,
    ModelPhi,
    SharedBankDesign,
    default_contrasts,
    group_by_model,
    model_phi,
    read_conditions,
    subset_grid_digest,
)
from ruleforge.evaluate.bootstrap import validate_bootstrap_args
from ruleforge.evaluate.campaign import SCENARIO_MANIFEST_FNAME
from ruleforge.evaluate.inference import (
    CURVE_INFERENCE_METHOD,
    CURVE_INTERVAL_BASIS,
    RepeatedSamplingError,
    check_curve_array,
    crossed_cluster_indices,
    crossed_cluster_means,
    curve_contrast_plans,
    curve_intervals,
)
from ruleforge.evaluate.scoring import PHI_CLASSES
from ruleforge.experiments.checks import require
from ruleforge.experiments.goldens import INTERVAL_COLUMNS, format_interval, interval_record
from ruleforge.experiments.ruleset_size.build import estimates_root, joined_roots, judge_label, load_estimates
from ruleforge.experiments.ruleset_size.figure import legend_order
from ruleforge.scenarios.manifest import ScenarioManifest

__all__ = ["ANALYSIS", "BASE", "CSV_COLUMNS", "DEFINITION", "SCHEMA_VERSION", "STATISTICS",
           "RepeatedDrawBaselineError", "build_payload", "cell_statistics", "csv_rows", "draw_median",
           "joint_bootstrap", "read_phi", "repeat_disagreement", "summary"]

ANALYSIS = "ruleset_size_repeated_draw_baseline"
SCHEMA_VERSION = 1
#: The per-cell statistics the bootstrap resamples, then those computed from them within each replicate.
BASE = ("inconsistency", "full_repeat", "separate_repeat", "lenient", "strict", "rule_median_inconsistency",
        "median_inconsistency")
STATISTICS = ("inconsistency", "full_repeat", "separate_repeat", "repeated_draw_baseline", "excess", "lenient",
              "strict", "lenient_minus_strict", "rule_median_inconsistency", "median_inconsistency")

DEFINITION = {
    "cell": "phi(yes)=1, phi(unsure)=0.5, phi(no)=0. A cell is one (ruleset size K, scenario, subset seed) of a "
            "judge's shared-bank campaign, with D decoder draws. F_d is phi of the subset condition's full-set "
            "judgment in draw d; M_d is the maximum, over the subset's rules, of phi of the bank's separate-call "
            "judgment in draw d. Every call is sampled separately (a local judge's with its own seed), so F_d and M_d "
            "are independent given the scenario and rules.",
    "inconsistency": "mean over d of 1[F_d != M_d]: the ruleset-size curve's cell value.",
    "full_repeat": "mean over the D(D-1) ordered pairs d != d' of 1[F_d != F_d']: two full-set calls disagree.",
    "separate_repeat": "mean over the D(D-1) ordered pairs d != d' of 1[M_d != M_d']: two separate-call maxima "
                       "disagree.",
    "repeated_draw_baseline": "(full_repeat + separate_repeat) / 2. separate_repeat is the rate expected if the "
                              "full-set judgment were one more draw of the separate-call maximum, and full_repeat "
                              "the rate expected if the maximum were one more draw of the full-set judgment.",
    "excess": "inconsistency - repeated_draw_baseline.",
    "lenient": "mean over d of 1[F_d < M_d]: the full-set judgment is below the strongest single-rule judgment (no "
               "where the strongest rule says unsure or yes, or unsure where it says yes).",
    "strict": "mean over d of 1[F_d > M_d]: the full set exceeds every rule alone.",
    "lenient_minus_strict": "lenient - strict.",
    "rule_median_inconsistency": "mean over d of 1[F_d != M~], with M~ the maximum, over the subset's rules, of "
                                 "each rule's median phi over the D bank draws (the upper median when D is even).",
    "median_inconsistency": "1[F~ != M~], with F~ the full-set judgment's median phi over the D draws (the upper "
                            "median when D is even) and M~ as in rule_median_inconsistency: the full call's median "
                            "judgment against the strongest rule's.",
    "identity": "For independent A ~ a and B ~ b over the three labels, P(A != B) = (w_a + w_b) / 2 + ||a - b||^2 / 2 "
                "with w_a = 1 - sum_v a_v^2. With f and m a cell's distributions of F_d and M_d, "
                "repeated_draw_baseline estimates the first term, the disagreement the two calls' own draw-to-draw "
                "variability contributes, and the expected excess is the mean of ||f - m||^2 / 2 over cells: >= 0, "
                "and 0 only when every cell's two distributions coincide. Then the expected inconsistency equals the "
                "repeated-draw baseline, so does each call's expected repeat disagreement, and lenient and strict "
                "have equal expectation.",
    "null": "The null f = m takes the one full-set judgment to be distributed as the maximum of the K separate-call "
            "judgments.",
    "u_statistic": "full_repeat and separate_repeat average over distinct draw pairs, so each is unbiased for its w.",
    "interval": "A size's estimate is the mean of its cells. Every statistic's interval and sd come from one crossed "
                "scenario x subset-seed cluster bootstrap of cell means: one set of resampled scenarios and seeds, "
                "drawn as the ruleset-size curve's primary interval draws them, serves every statistic, so "
                "inconsistency's estimates and intervals are that curve's. repeated_draw_baseline, excess and "
                "lenient_minus_strict are computed from the base statistics within each replicate, and a contrast "
                "is the mean of its minuend sizes minus the mean of its subtrahend sizes within each replicate.",
}

CSV_COLUMNS = ("label", "model_name", "size",
               *(f"{name}{suffix}" for name in STATISTICS for suffix, _ in INTERVAL_COLUMNS))
_POINT_KEYS = ("size", "estimate", "ci_low", "ci_high", "sd")
_CONTRAST_KEYS = ("minuend_sizes", "subtrahend_sizes", *_POINT_KEYS[1:])
_SUMMARY_STATISTICS = ("inconsistency", "repeated_draw_baseline", "excess", "lenient_minus_strict",
                       "rule_median_inconsistency", "median_inconsistency")
_SUMMARY_CONTRAST = "net_rise"


class RepeatedDrawBaselineError(ValueError, RuleForgeError):
    """The repeated-draw baseline cannot be built from these estimates or roots."""


_require = partial(require, error=RepeatedDrawBaselineError)


def read_phi(root: Path | str, model: str, *, n_draws: int,
             bank_root: Optional[Path] = None) -> Tuple[SharedBankDesign, ModelPhi]:
    """Return a shared-bank root's design and ``model``'s phi arrays, joined to ``bank_root`` as its analysis joined
    them, each subset draw joined to the bank's draw of the same index, refused as the root's analysis refuses them."""
    root = Path(root)
    found = read_conditions(root, bank_root)
    manifest = ScenarioManifest.load(root / SCENARIO_MANIFEST_FNAME)
    by_model = group_by_model(found.registered, found.design)
    if model not in by_model:
        raise RepeatedDrawBaselineError(f"{root} registers no conditions of {model!r}; it holds {sorted(by_model)}")
    return found.design, model_phi(found.dirs, by_model[model], found.design, scenario_ids=manifest.scenario_ids,
                                   n_draws=n_draws, scenario_manifest_sha256=manifest.manifest_sha256)


def repeat_disagreement(phi: np.ndarray) -> np.ndarray:
    """Return, per cell, the share of ordered pairs of distinct draws (the last axis) whose labels differ."""
    n_draws = phi.shape[-1]
    pairs = n_draws * (n_draws - 1)
    counts = np.stack([(phi == value).sum(axis=-1) for value in PHI_CLASSES])
    return (pairs - (counts * (counts - 1)).sum(axis=0)) / pairs


def draw_median(phi: np.ndarray) -> np.ndarray:
    """Return, per cell, the median phi over the draws (the last axis), the upper median when the draws are even, as
    the rules' medians are taken."""
    return np.sort(phi, axis=-1)[..., phi.shape[-1] // 2]


def cell_statistics(
    full: np.ndarray, separate: np.ndarray, max_rule_median: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return each base statistic's ``(sizes, scenarios, subset_seeds)`` cells and its per-size estimate."""
    cells: Dict[str, np.ndarray] = {}
    observed: Dict[str, np.ndarray] = {}
    for name, flags in (("inconsistency", full != separate), ("lenient", full < separate),
                        ("strict", full > separate), ("rule_median_inconsistency", full != max_rule_median[..., None])):
        values = flags.astype(float)
        # A size's estimate is taken over all its draws at once, as the curve's is, so inconsistency's is the curve's
        # bitwise.
        cells[name], observed[name] = values.mean(axis=3), values.mean(axis=(1, 2, 3))
    cells["median_inconsistency"] = (draw_median(full) != max_rule_median).astype(float)
    for name, phi in (("full_repeat", full), ("separate_repeat", separate)):
        cells[name] = repeat_disagreement(phi)
    for name in ("median_inconsistency", "full_repeat", "separate_repeat"):
        observed[name] = cells[name].mean(axis=(1, 2))
    return cells, observed


def _derive(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    baseline = (values["full_repeat"] + values["separate_repeat"]) / 2
    derived = {**values, "repeated_draw_baseline": baseline, "excess": values["inconsistency"] - baseline,
               "lenient_minus_strict": values["lenient"] - values["strict"]}
    return {name: derived[name] for name in STATISTICS}


def joint_bootstrap(
    full: np.ndarray,
    separate: np.ndarray,
    max_rule_median: np.ndarray,
    *,
    sizes: Sequence[int],
    contrasts: Sequence[Mapping[str, Any]],
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Return every statistic's ``{"points", "contrasts"}`` under one crossed scenario x subset-seed bootstrap."""
    validate_bootstrap_args(n_resamples, ci_level)
    full, separate = check_curve_array(full, sizes), check_curve_array(separate, sizes)
    max_rule_median = np.asarray(max_rule_median, dtype=float)
    if full.shape != separate.shape or max_rule_median.shape != full.shape[:3]:
        raise RepeatedSamplingError(f"the full-set array is {full.shape}, the separate-call array {separate.shape} "
                                    f"and its rule-median array {max_rule_median.shape}")
    if not np.isfinite(max_rule_median).all():
        raise RepeatedSamplingError("a rule-median cell is missing (NaN)")
    _, n_scen, n_seed, _ = full.shape
    sizes = [int(size) for size in sizes]
    plans = curve_contrast_plans(sizes, contrasts)
    cells, observed = cell_statistics(full, separate, max_rule_median)
    scen, seeds = crossed_cluster_indices(np.random.default_rng(seed), n_resamples, n_scen, n_seed)
    replicates = {name: crossed_cluster_means(cells[name], scen, seeds) for name in BASE}
    observed, replicates = _derive(observed), _derive(replicates)
    found = {}
    for name in STATISTICS:
        points, rows = curve_intervals(observed[name], replicates[name], sizes=sizes, plans=plans, ci_level=ci_level)
        found[name] = {"points": points, "contrasts": rows}
    return found


def _keys(row: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {key: row[key] for key in keys}


def judge_record(payload: Mapping[str, Any], model: str) -> Dict[str, Any]:
    """Return one judge's statistics from its root, refused unless its grids and inconsistency rows are its estimates'
    exactly."""
    entry = payload["models"][model]
    model_name = entry.get("model_name", model)
    estimates = to_uri(payload["_path"])
    where = f"{model} in {estimates}"
    bank = payload.get(BANK_ROOT_KEY)
    design, phi = read_phi(estimates_root(payload), model, n_draws=int(payload["n_decoder_draws"]),
                           bank_root=None if bank is None else resolve(bank["root"]))
    sizes = list(design.subset_sizes)
    _require([(design.to_dict() == payload["design"], "its root's design is not its estimates'"),
              (phi.bank_digest == entry["bank"]["grid_digest"], "its bank's grid digest is not its estimates'"),
              (subset_grid_digest(phi.inconsistency) == entry["subsets"]["grid_digest"],
               "its inconsistency grid's digest is not its estimates'")], where)
    bootstrap = payload["bootstrap"]
    found = joint_bootstrap(phi.full, phi.separate, phi.max_rule_median, sizes=sizes,
                            contrasts=default_contrasts(sizes), n_resamples=int(bootstrap["n_resamples"]),
                            ci_level=float(bootstrap["ci_level"]), seed=int(bootstrap["seed"]))
    inconsistency = found["inconsistency"]
    _require([(inconsistency["points"] == [_keys(point, _POINT_KEYS) for point in entry["points"]],
               "its inconsistency rows are not its estimates' points"),
              (inconsistency["contrasts"] == {name: _keys(row, _CONTRAST_KEYS)
                                              for name, row in entry["contrasts"].items()},
               "its inconsistency contrasts are not its estimates'")], where)
    return {
        "label": judge_label(model_name),
        "model_name": model_name,
        "campaign": {"campaign_id": (bank or payload)["campaign_id"], "estimates": estimates,
                     "estimates_sha256": payload["_sha256"], **joined_roots(payload)},
        "grid_digests": {"bank": phi.bank_digest, "subsets": entry["subsets"]["grid_digest"]},
        "points": [{"size": size, **{name: interval_record(found[name]["points"][index]) for name in STATISTICS}}
                   for index, size in enumerate(sizes)],
        "contrasts": {name: found[name]["contrasts"] for name in STATISTICS},
    }


def build_payload(paths: Sequence[Path], *, workers: int = 1) -> Dict[str, Any]:
    """Return the repeated-draw baseline of every judge the estimates name, in legend order, one judge per process when
    ``workers`` exceeds one."""
    payloads = load_estimates(paths, allow_incomplete=False)
    first = payloads[0]
    _require([(first.get("interval_basis") == CURVE_INTERVAL_BASIS,
               f"their interval basis is {first.get('interval_basis')!r}, not {CURVE_INTERVAL_BASIS!r}"),
              (first.get("inference_method") == CURVE_INFERENCE_METHOD,
               f"their inference method is {first.get('inference_method')!r}, not {CURVE_INFERENCE_METHOD!r}")],
             "the estimates")
    tasks = [(payload, model) for payload in payloads for model in payload["models"]]
    processes = min(workers, len(tasks))
    if processes <= 1:
        judges = [judge_record(payload, model) for payload, model in tasks]
    else:
        with ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn")) as pool:
            judges = list(pool.map(judge_record, *zip(*tasks)))
    judges = legend_order(judges, lambda judge: judge["label"])
    sizes = [int(size) for size in first["subset_sizes"]]
    return {
        "analysis": ANALYSIS,
        "schema_version": SCHEMA_VERSION,
        "definition": DEFINITION,
        "statistics": list(STATISTICS),
        "bootstrap": {**first["bootstrap"], "interval_basis": CURVE_INTERVAL_BASIS},
        "subset_sizes": sizes,
        "n_scenarios": int(first["n_scenarios"]),
        "n_subset_seeds": int(first["n_subset_seeds"]),
        "n_decoder_draws": int(first["n_decoder_draws"]),
        "contrasts": default_contrasts(sizes),
        "judges": judges,
    }


def csv_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return one row per judge and size, in :data:`CSV_COLUMNS`."""
    rows = []
    for judge in payload["judges"]:
        for point in judge["points"]:
            row = {"label": judge["label"], "model_name": judge["model_name"], "size": point["size"]}
            for name in STATISTICS:
                for suffix, key in INTERVAL_COLUMNS:
                    row[f"{name}{suffix}"] = point[name][key]
            rows.append(row)
    return rows


def summary(payload: Mapping[str, Any]) -> str:
    """Return each judge's statistics at the smallest and largest size and their net rise, one judge per block."""
    lines = []
    for judge in payload["judges"]:
        lines.append(judge["label"])
        for point in (judge["points"][0], judge["points"][-1]):
            lines.append(f"  {'K=' + str(point['size']):<9}" + "  ".join(
                f"{name} {format_interval(point[name])}" for name in _SUMMARY_STATISTICS))
        lines.append(f"  {'net rise':<9}" + "  ".join(
            f"{name} {format_interval(judge['contrasts'][name][_SUMMARY_CONTRAST], signed=True)}"
            for name in _SUMMARY_STATISTICS))
    lines.append(f"{len(payload['judges'])} judge(s); N={payload['n_scenarios']} x J={payload['n_subset_seeds']} "
                 f"x D={payload['n_decoder_draws']}; {payload['bootstrap']['interval_basis']}")
    return "\n".join(lines) + "\n"
