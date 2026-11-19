from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from ruleforge.core.errors import RuleForgeError
from ruleforge.evaluate.bootstrap import (
    DEFAULT_CI_LEVEL,
    DEFAULT_RESAMPLES,
    percentile_bounds,
    validate_bootstrap_args,
)
from ruleforge.evaluate.multiplicity import adjust_for_multiplicity
from ruleforge.evaluate.identity import BUNDLE_OUTCOMES_FNAME, COMPLETION_FNAME, draw_dirname

INFERENCE_METHOD = "hierarchical_bootstrap_v2"
# The primary interval is the scenario-level cluster bootstrap of per-scenario draw means (Davison & Hinkley 1997,
# section 3.8). The two-stage resample is reported as conservative: its decoder component over-counts by (2S-1)/S.
PRIMARY_INTERVAL_BASIS = "scenario_cluster_bootstrap"
PROJECTION_N = 500
PROJECTION_DRAWS = (1, 2, 5, 10, 20)
TEST_METHOD = "scenario_clustered_paired_randomization_v1"
BOOTSTRAP_TEST_METHOD = "scenario_cluster_bootstrap_percentile_test_v1"
DEFAULT_PERMUTATIONS = 100_000
MIN_PERMUTATIONS = 100
# Replicates per vectorised chunk; bounds memory at about ``chunk * N * S`` floats per condition.
_CHUNK = 500


class RepeatedSamplingError(ValueError, RuleForgeError):
    """Raised when a grid or a resampling request is not what the design needs."""


@dataclass(frozen=True)
class DrawGrid:
    """One condition's outcomes as a dense ``(n_scenarios, n_draws)`` array, whose digest covers ids, draw count and
    values."""

    condition_id: str
    quantity: str
    scenario_ids: tuple[str, ...]
    values: np.ndarray
    identity_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise RepeatedSamplingError(
                f"{self.condition_id}: values must be (n_scenarios, n_draws), got "
                f"shape {values.shape}"
            )
        if values.shape[0] != len(self.scenario_ids) or values.shape[0] == 0:
            raise RepeatedSamplingError(
                f"{self.condition_id}: {len(self.scenario_ids)} scenario ids for "
                f"{values.shape[0]} rows"
            )
        if values.shape[1] < 1:
            raise RepeatedSamplingError(f"{self.condition_id}: a grid needs at least one draw")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise RepeatedSamplingError(f"{self.condition_id}: duplicate scenario ids")
        if not np.all(np.isfinite(values)):
            raise RepeatedSamplingError(f"{self.condition_id}: a cell is missing (NaN)")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "scenario_ids", tuple(self.scenario_ids))

    @property
    def n_scenarios(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_draws(self) -> int:
        return int(self.values.shape[1])

    @property
    def scenario_means(self) -> np.ndarray:
        return self.values.mean(axis=1)

    @property
    def rate(self) -> float:
        """``mean_i(mean_s(Y))`` -- equal to the pooled mean on a complete grid."""
        return float(self.scenario_means.mean())

    @property
    def digest(self) -> str:
        payload = {
            "condition_id": self.condition_id if self.identity_id is None else self.identity_id,
            "quantity": self.quantity,
            "scenario_ids": list(self.scenario_ids),
            "n_draws": self.n_draws,
            "values": [[float(v) for v in row] for row in self.values],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def restricted_to(self, scenario_ids: Sequence[str]) -> "DrawGrid":
        """The same grid over ``scenario_ids`` in that order; refuses a missing id."""
        position = {sid: i for i, sid in enumerate(self.scenario_ids)}
        missing = [sid for sid in scenario_ids if sid not in position]
        if missing:
            raise RepeatedSamplingError(
                f"{self.condition_id}: {len(missing)} requested scenario(s) absent "
                f"from the grid (first {missing[0][:12]})"
            )
        rows = [position[sid] for sid in scenario_ids]
        return DrawGrid(self.condition_id, self.quantity, tuple(scenario_ids), self.values[rows], self.identity_id)


def grid_from_cells(
    condition_id: str,
    quantity: str,
    scenario_ids: Sequence[str],
    n_draws: int,
    cells: Mapping[tuple[str, int], float],
    *,
    identity_id: Optional[str] = None,
) -> DrawGrid:
    """Return a grid from ``{(scenario_id, draw): value}``, refusing holes and extras."""
    expected = {(sid, draw) for sid in scenario_ids for draw in range(n_draws)}
    missing = sorted(expected - set(cells))
    unexpected = sorted(set(cells) - expected)
    if missing or unexpected:
        raise RepeatedSamplingError(
            f"{condition_id}/{quantity}: grid is not exact -- {len(missing)} missing, "
            f"{len(unexpected)} unexpected cell(s)"
            + (f"; first missing {missing[0]}" if missing else "")
            + (f"; first unexpected {unexpected[0]}" if unexpected else "")
        )
    values = np.array(
        [[float(cells[(sid, draw)]) for draw in range(n_draws)] for sid in scenario_ids],
        dtype=float,
    )
    return DrawGrid(condition_id, quantity, tuple(scenario_ids), values, identity_id)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepeatedSamplingError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RepeatedSamplingError(f"{path} is not a JSON object")
    return payload


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_draw_outcomes(
    draw_dir: Path,
    draw: int,
    *,
    expected_scenario_ids: Sequence[str],
    scenario_manifest_sha256: Optional[str] = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return one complete draw's outcomes, ``{scenario_id: row}``, and its completion record, refusing anything off the
    design."""
    completion = _read_json(draw_dir / COMPLETION_FNAME)
    if completion.get("state") != "complete":
        raise RepeatedSamplingError(
            f"{draw_dir}: completion state is {completion.get('state')!r}, not "
            "'complete'; an incomplete draw is never analysed"
        )
    if completion.get("decoder_draw_id") != draw:
        raise RepeatedSamplingError(
            f"{draw_dir}: completion record is for draw {completion.get('decoder_draw_id')!r}"
        )
    if (
        scenario_manifest_sha256 is not None
        and completion.get("scenario_manifest_sha256") != scenario_manifest_sha256
    ):
        raise RepeatedSamplingError(
            f"{draw_dir}: scenario manifest {completion.get('scenario_manifest_sha256')!r} "
            f"is not the campaign's {scenario_manifest_sha256!r}"
        )
    expected_ids = set(expected_scenario_ids)
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(draw_dir / BUNDLE_OUTCOMES_FNAME):
        sid = str(row["scenario_id"])
        if sid in rows:
            raise RepeatedSamplingError(f"{draw_dir}: scenario {sid[:12]} appears twice")
        if int(row.get("decoder_draw_id", -1)) != draw:
            raise RepeatedSamplingError(
                f"{draw_dir}: an outcome row claims draw {row.get('decoder_draw_id')!r}"
            )
        rows[sid] = row
    if set(rows) != expected_ids:
        raise RepeatedSamplingError(
            f"{draw_dir}: outcomes cover {len(rows)} scenarios, the manifest "
            f"{len(expected_ids)} ({len(expected_ids - set(rows))} missing, "
            f"{len(set(rows) - expected_ids)} extra)"
        )
    return rows, completion


def load_condition_outcomes(
    condition_root: Path | str,
    *,
    expected_scenario_ids: Sequence[str],
    expected_draws: int,
    scenario_manifest_sha256: Optional[str] = None,
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    """Return every accepted bundle outcome of one condition, keyed by (scenario, draw), and a provenance block."""
    root = Path(condition_root)
    draw_dirs = sorted(p for p in root.glob("draw-*") if p.is_dir())
    expected_names = [draw_dirname(draw) for draw in range(expected_draws)]
    names = [p.name for p in draw_dirs]
    if names != expected_names:
        raise RepeatedSamplingError(
            f"{root}: expected draw directories {expected_names}, found {names}"
        )
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    completions: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for draw, draw_dir in enumerate(draw_dirs):
        rows, completion = read_draw_outcomes(
            draw_dir, draw, expected_scenario_ids=expected_scenario_ids,
            scenario_manifest_sha256=scenario_manifest_sha256,
        )
        fingerprints.add(str(completion.get("condition_fingerprint")))
        completions.append(completion)
        cells.update(((sid, draw), row) for sid, row in rows.items())
    if len(fingerprints) != 1:
        raise RepeatedSamplingError(
            f"{root}: draws disagree on the condition fingerprint: {sorted(fingerprints)}"
        )
    provenance = {
        "condition_fingerprint": fingerprints.pop(),
        "n_draws": expected_draws,
        "draws": completions,
    }
    return cells, provenance


def outcome_grid(
    condition_id: str,
    cells: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    scenario_ids: Sequence[str],
    n_draws: int,
    quantity: str = "contradiction",
    identity_id: Optional[str] = None,
) -> DrawGrid:
    """Return a grid of one scalar field of the bundle outcomes; a null field refuses the grid."""
    values: dict[tuple[str, int], float] = {}
    for key, row in cells.items():
        value = row.get(quantity)
        if value is None:
            raise RepeatedSamplingError(
                f"{condition_id}: outcome {key[0][:12]}/draw {key[1]} has no "
                f"{quantity!r}; this condition does not carry that quantity"
            )
        values[key] = float(value)
    return grid_from_cells(condition_id, quantity, scenario_ids, n_draws, values, identity_id=identity_id)


def rule_grids(
    condition_id: str,
    cells: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    scenario_ids: Sequence[str],
    n_draws: int,
    rule_id: str,
    trigger: str = "yes",
    field: str = "rule_judgments",
    identity_id: Optional[str] = None,
) -> DrawGrid:
    """Return grids of ``1[judgment == trigger]`` for one rule, judged alone or in the same forced response's own
    entry."""
    values: dict[tuple[str, int], float] = {}
    for key, row in cells.items():
        judgments = row.get(field)
        if not isinstance(judgments, Mapping) or rule_id not in judgments:
            raise RepeatedSamplingError(
                f"{condition_id}: outcome {key[0][:12]}/draw {key[1]} carries no "
                f"{field}[{rule_id!r}]"
            )
        values[key] = float(judgments[rule_id] == trigger)
    return grid_from_cells(condition_id, f"{field}:{rule_id}=={trigger}", scenario_ids, n_draws, values,
                           identity_id=identity_id)


def _resampled_rates(
    grid: DrawGrid,
    scenario_index: np.ndarray,
    draw_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # ``draw_index`` is read per selected scenario occurrence, so a scenario drawn twice gets two independent inner
    # samples.
    values = grid.values
    two_stage = values[scenario_index[:, :, None], draw_index].mean(axis=(1, 2))
    scenario_level = grid.scenario_means[scenario_index].mean(axis=1)
    return two_stage, scenario_level


def variance_components(values: np.ndarray, *, projected_n: int = PROJECTION_N) -> dict[str, Any]:
    """Return the one-way random-effects decomposition of a balanced ``N x S`` grid and the projected standard error."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 2:
        raise RepeatedSamplingError(
            f"variance components need at least two scenarios and two draws, got {array.shape}"
        )
    n, s = array.shape
    within = float(array.var(axis=1, ddof=1).mean())
    between = max(0.0, float(array.mean(axis=1).var(ddof=1)) - within / s)
    total = between + within

    def se(scenarios: int, draws: float) -> float:
        return float(np.sqrt(between / scenarios + within / (scenarios * draws)))

    return {
        "n_scenarios": n,
        "n_decoder_draws": s,
        "within_draw": within,
        "between_scenario": between,
        "icc": between / total if total > 0 else None,
        "se_analytic": se(n, s),
        # The share of the estimator's variance that decoder noise contributes.
        "decoder_share_of_variance": (within / s) / (between + within / s) if total > 0 else None,
        "se_projection": {
            "n_scenarios": projected_n,
            "by_draws": {str(draws): se(projected_n, draws) for draws in PROJECTION_DRAWS},
            "infinite_draws": float(np.sqrt(between / projected_n)),
        },
    }


def _validate_grids(grids: Mapping[str, DrawGrid]) -> tuple[str, ...]:
    if not grids:
        raise RepeatedSamplingError("no grids to resample")
    ids = None
    for name, grid in grids.items():
        if grid.n_draws < 2:
            raise RepeatedSamplingError(
                f"{name}: {grid.n_draws} decoder draw(s); decoder variance cannot be "
                "estimated from one draw, so a joint interval is refused"
            )
        if ids is None:
            ids = grid.scenario_ids
        elif grid.scenario_ids != ids:
            raise RepeatedSamplingError(
                f"{name}: scenario ids differ from the other grids; a contrast needs "
                "the same scenarios in the same order"
            )
    return ids


def _inner_groups(
    names: Sequence[str],
    grids: Mapping[str, DrawGrid],
    shared_inner: Sequence[Iterable[str]],
) -> dict[str, int]:
    # Only quantities from the same generated artifact share an inner resample.
    group_of = {name: index for index, name in enumerate(names)}
    for members in shared_inner:
        members = list(members)
        unknown = [m for m in members if m not in group_of]
        if unknown:
            raise RepeatedSamplingError(f"shared_inner names unknown grid(s) {unknown}")
        draws = {grids[m].n_draws for m in members}
        if len(draws) != 1:
            raise RepeatedSamplingError(
                f"shared_inner group {members} mixes draw counts {sorted(draws)}"
            )
        # Merge transitively, so overlapping shared_inner sets form one group in any order.
        merged = {group_of[m] for m in members}
        target = min(merged)
        for name in names:
            if group_of[name] in merged:
                group_of[name] = target
    return group_of


def hierarchical_bootstrap(
    grids: Mapping[str, DrawGrid],
    *,
    contrasts: Sequence[tuple[str, str]] = (),
    shared_inner: Sequence[Iterable[str]] = (),
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 16,
) -> dict[str, Any]:
    """Return rates and paired contrasts under a scenario-level bootstrap; ``two_stage`` is the conservative reading.
    Conditions named together in ``shared_inner`` share an inner resample."""
    validate_bootstrap_args(n_resamples, ci_level)
    ids = _validate_grids(grids)
    names = list(grids)
    for a, b in contrasts:
        if a not in grids or b not in grids:
            raise RepeatedSamplingError(f"contrast ({a!r}, {b!r}) names an unknown grid")
    group_of = _inner_groups(names, grids, shared_inner)
    n = len(ids)
    rng = np.random.default_rng(seed)
    two_stage = {name: np.empty(n_resamples) for name in names}
    scenario_level = {name: np.empty(n_resamples) for name in names}
    for start in range(0, n_resamples, _CHUNK):
        stop = min(start + _CHUNK, n_resamples)
        scenario_index = rng.integers(0, n, size=(stop - start, n))
        draw_index: dict[int, np.ndarray] = {}
        for name in names:
            group = group_of[name]
            if group not in draw_index:
                draw_index[group] = rng.integers(
                    0, grids[name].n_draws, size=(stop - start, n, grids[name].n_draws)
                )
            t, c = _resampled_rates(grids[name], scenario_index, draw_index[group])
            two_stage[name][start:stop], scenario_level[name][start:stop] = t, c

    def summarize(point: float, primary: np.ndarray, conservative: np.ndarray) -> dict[str, Any]:
        low, high = percentile_bounds(primary, ci_level)
        low2, high2 = percentile_bounds(conservative, ci_level)
        return {
            "estimate": float(point),
            "ci_low": low,
            "ci_high": high,
            "sd": float(primary.std(ddof=0)),
            "interval_basis": PRIMARY_INTERVAL_BASIS,
            "two_stage": {"ci_low": low2, "ci_high": high2, "sd": float(conservative.std(ddof=0))},
        }

    conditions = {
        name: {
            **summarize(grids[name].rate, scenario_level[name], two_stage[name]),
            "variance_components": variance_components(grids[name].values),
            "n_scenarios": grids[name].n_scenarios,
            "n_decoder_draws": grids[name].n_draws,
            "grid_digest": grids[name].digest,
        }
        for name in names
    }
    contrast_rows = []
    for a, b in contrasts:
        paired = grids[a].scenario_means - grids[b].scenario_means
        contrast_rows.append(
            {
                "a": a,
                "b": b,
                "inner_resample": "joint" if group_of[a] == group_of[b] else "independent",
                **summarize(
                    grids[a].rate - grids[b].rate,
                    scenario_level[a] - scenario_level[b],
                    two_stage[a] - two_stage[b],
                ),
                "se_analytic": float(paired.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            }
        )
    return {
        "method": INFERENCE_METHOD,
        "resampling": {
            "primary": "scenario positions with replacement, shared across conditions; "
                       "the observed per-scenario draw means are the resampled units "
                       "(a balanced nested design resamples its top level only)",
            "two_stage": "scenario positions, then decoder draws with replacement within "
                         "each selected cell (independent per condition unless named in "
                         "shared_inner); conservative on the decoder component by (2S-1)/S",
            "n_resamples": n_resamples,
            "ci_level": ci_level,
            "seed": seed,
            "shared_inner": [sorted(group) for group in shared_inner],
        },
        "n_scenarios": n,
        "conditions": conditions,
        "contrasts": contrast_rows,
    }


def clustered_paired_randomization_p(
    grid_a: DrawGrid,
    grid_b: DrawGrid,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 16,
) -> dict[str, Any]:
    """Return the two-sided p for ``rate(a) - rate(b)``, swapping complete draw vectors within each scenario."""
    if grid_a.scenario_ids != grid_b.scenario_ids:
        raise RepeatedSamplingError("the two grids must cover the same scenarios in order")
    if n_permutations < MIN_PERMUTATIONS:
        raise RepeatedSamplingError(
            f"n_permutations must be at least {MIN_PERMUTATIONS}, got {n_permutations}"
        )
    differences = grid_a.scenario_means - grid_b.scenario_means
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    threshold = abs(observed) - 1e-12
    for start in range(0, n_permutations, _CHUNK * 4):
        count = min(_CHUNK * 4, n_permutations - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, differences.size))
        stats = signs @ differences / differences.size
        extreme += int(np.sum(np.abs(stats) >= threshold))
    return {
        "method": TEST_METHOD,
        "observed_difference": observed,
        "p_value": (extreme + 1) / (n_permutations + 1),
        "n_permutations": n_permutations,
        "seed": seed,
        "n_scenarios": int(differences.size),
    }


STATISTIC_RESAMPLING = "recomputed_on_every_replicate"
# An undefined permuted statistic counts as at least as extreme as the observed one.
UNDEFINED_PERMUTATION_RULE = "counted_as_extreme"

# ``(n, K) scenario-mean features -> (n, J) statistics``, row by row.
Statistic = Callable[[np.ndarray], np.ndarray]


def _as_features(features: np.ndarray, name: str = "features") -> np.ndarray:
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise RepeatedSamplingError(f"{name} must be (n_scenarios >= 2, n_features >= 1), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise RepeatedSamplingError(f"{name} hold a missing (NaN) cell")
    return array


def _statistic_rows(statistic: Statistic, means: np.ndarray) -> np.ndarray:
    values = np.asarray(statistic(means), dtype=float)
    return values.reshape(means.shape[0], -1)


def scenario_cluster_bootstrap(
    features: np.ndarray,
    statistic: Statistic,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the point value and ``(n_resamples, J)`` replicates of a function of scenario means; an undefined
    replicate is NaN."""
    validate_bootstrap_args(n_resamples, ci_level)
    array = _as_features(features)
    n = array.shape[0]
    point = _statistic_rows(statistic, array.mean(axis=0, keepdims=True))[0]
    replicates = np.empty((n_resamples, point.size))
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, _CHUNK):
        stop = min(start + _CHUNK, n_resamples)
        picks = rng.integers(0, n, size=(stop - start, n))
        # Multiplicities of each scenario in each replicate, then the mean row.
        offsets = (picks + n * np.arange(stop - start)[:, None]).ravel()
        counts = np.bincount(offsets, minlength=(stop - start) * n).reshape(stop - start, n)
        replicates[start:stop] = _statistic_rows(statistic, counts @ array / n)
    return point, replicates


def statistic_interval(point: float, replicates: np.ndarray, ci_level: float) -> dict[str, Any]:
    """Return one statistic's estimate and percentile interval over its defined replicates."""
    values = np.asarray(replicates, dtype=float)
    defined = values[np.isfinite(values)]
    low, high = percentile_bounds(defined, ci_level) if defined.size else (None, None)
    return {
        "estimate": float(point) if np.isfinite(point) else None,
        "ci_low": low,
        "ci_high": high,
        "sd": float(defined.std(ddof=0)) if defined.size else None,
        "interval_basis": PRIMARY_INTERVAL_BASIS,
        "statistic_resampling": STATISTIC_RESAMPLING,
        "n_undefined_replicates": int(values.size - defined.size),
    }


def percentile_bootstrap_p(replicates: np.ndarray) -> float:
    """Return the two-sided p, at most 1, for a difference of 0 from its paired bootstrap replicates (the difference
    itself, not exchangeability)."""
    values = np.asarray(replicates, dtype=float).ravel()
    defined = values[np.isfinite(values)]
    tail = min(int(np.sum(defined <= 1e-12)), int(np.sum(defined >= -1e-12)))
    return float(min(1.0, 2 * (tail + 1) / (defined.size + 1)))


def clustered_paired_randomization_statistic(
    features_a: np.ndarray,
    features_b: np.ndarray,
    statistic: Statistic,
    *,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 16,
) -> dict[str, Any]:
    """Return the two-sided p for ``statistic(a) - statistic(b)``, recomputed after swapping feature rows within each
    scenario."""
    a = _as_features(features_a, "features_a")
    b = _as_features(features_b, "features_b")
    if a.shape != b.shape:
        raise RepeatedSamplingError(f"the two arms' features differ in shape: {a.shape} vs {b.shape}")
    if n_permutations < MIN_PERMUTATIONS:
        raise RepeatedSamplingError(
            f"n_permutations must be at least {MIN_PERMUTATIONS}, got {n_permutations}"
        )
    n = a.shape[0]
    mean_a, mean_b = a.mean(axis=0, keepdims=True), b.mean(axis=0, keepdims=True)
    observed = float(_statistic_rows(statistic, mean_a)[0, 0] - _statistic_rows(statistic, mean_b)[0, 0])
    result = {
        "method": TEST_METHOD,
        "observed_difference": observed if np.isfinite(observed) else None,
        "p_value": None,
        "n_permutations": n_permutations,
        "seed": seed,
        "n_scenarios": n,
        "statistic_resampling": STATISTIC_RESAMPLING,
        "undefined_permutations": UNDEFINED_PERMUTATION_RULE,
        "n_undefined_permutations": 0,
    }
    if not np.isfinite(observed):
        return result
    shift = (b - a) / n
    rng = np.random.default_rng(seed)
    extreme = undefined = 0
    threshold = abs(observed) - 1e-12
    for start in range(0, n_permutations, _CHUNK * 4):
        count = min(_CHUNK * 4, n_permutations - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, n))
        moved = (signs < 0).astype(float) @ shift
        stats = _statistic_rows(statistic, mean_a + moved)[:, 0] - _statistic_rows(statistic, mean_b - moved)[:, 0]
        finite = np.isfinite(stats)
        undefined += int(np.sum(~finite))
        extreme += int(np.sum(~finite)) + int(np.sum(np.abs(stats[finite]) >= threshold))
    result["p_value"] = (extreme + 1) / (n_permutations + 1)
    result["n_undefined_permutations"] = undefined
    return result


def holm_adjust(
    rows: Sequence[Mapping[str, Any]], alpha: float = 0.05, *, test_method: str = TEST_METHOD
) -> list[dict[str, Any]]:
    """Holm-adjust ``p_value`` across one family of rows, stamping the test that gave it."""
    adjusted = []
    for row in adjust_for_multiplicity(rows, method="holm", alpha=alpha):
        row["p_holm"] = row.pop("p_adjusted")
        row["significant_holm"] = row.pop("significant_adjusted")
        row.pop("adjustment_method")
        row["test_method"] = test_method
        adjusted.append(row)
    return adjusted


CURVE_INFERENCE_METHOD = "crossed_scenario_subset_bankshared_decoder_bootstrap_v2"
CURVE_INTERVAL_BASIS = "scenario_subset_cluster_bootstrap"
# Replicates per gather; bounds memory.
_CURVE_CHUNK = 50
_CURVE_CHUNK_TWO_STAGE = 10


def _curve_interval(replicates: np.ndarray, ci_level: float) -> dict[str, Any]:
    low, high = percentile_bounds(replicates, ci_level)
    return {"ci_low": low, "ci_high": high, "sd": float(replicates.std(ddof=0))}


def check_curve_array(values: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Return ``values`` as a float ``(n_sizes, n_scenarios, n_subset_seeds, n_draws)`` array, refusing one the curve
    bootstrap cannot resample."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 4:
        raise RepeatedSamplingError(
            f"values must be (sizes, scenarios, subset_seeds, draws), got {array.shape}"
        )
    n_size, n_scen, n_seed, n_draw = array.shape
    sizes = [int(size) for size in sizes]
    if len(sizes) != n_size or len(set(sizes)) != n_size:
        raise RepeatedSamplingError(f"{n_size} size slices for sizes {sizes}")
    if min(n_scen, n_seed, n_draw) < 2:
        raise RepeatedSamplingError(
            "the bank-shared curve bootstrap needs at least two scenarios, two subset "
            f"seeds and two draws, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise RepeatedSamplingError("a cell is missing (NaN)")
    return array


class ContrastPlan(NamedTuple):
    """A contrast's name, its minuend and subtrahend sizes, and those sizes' positions among the curve's."""

    name: str
    minuend: list[int]
    subtrahend: list[int]
    minuend_idx: list[int]
    subtrahend_idx: list[int]


def curve_contrast_plans(sizes: Sequence[int], contrasts: Sequence[Mapping[str, Any]]) -> list[ContrastPlan]:
    """Return each contrast's plan, refusing one off ``sizes``."""
    position = {int(size): index for index, size in enumerate(sizes)}
    plans: list[ContrastPlan] = []
    for contrast in contrasts:
        name = str(contrast["name"])
        minuend = [int(size) for size in contrast["minuend"]]
        subtrahend = [int(size) for size in contrast["subtrahend"]]
        unknown = [size for size in minuend + subtrahend if size not in position]
        if unknown:
            raise RepeatedSamplingError(f"contrast {name!r} names sizes {unknown} outside {list(position)}")
        if not minuend or not subtrahend:
            raise RepeatedSamplingError(f"contrast {name!r} needs sizes on both sides")
        if any(plan.name == name for plan in plans):
            raise RepeatedSamplingError(f"contrast {name!r} is named twice")
        plans.append(ContrastPlan(name, minuend, subtrahend, [position[s] for s in minuend],
                                  [position[s] for s in subtrahend]))
    return plans


def crossed_cluster_indices(
    rng: np.random.Generator, n_resamples: int, n_scenarios: int, n_subset_seeds: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the replicates' scenario and subset-seed resamples, drawn from ``rng`` in that order."""
    scen = rng.integers(0, n_scenarios, size=(n_resamples, n_scenarios), dtype=np.int32)
    seeds = rng.integers(0, n_subset_seeds, size=(n_resamples, n_subset_seeds), dtype=np.int32)
    return scen, seeds


def crossed_cluster_means(cell: np.ndarray, scen: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Return the ``(n_resamples, n_sizes)`` replicate means of a ``(n_sizes, n_scenarios, n_subset_seeds)`` cell
    array under the given resamples."""
    n_resamples = scen.shape[0]
    replicates = np.empty((n_resamples, cell.shape[0]))
    for start in range(0, n_resamples, _CURVE_CHUNK):
        stop = min(start + _CURVE_CHUNK, n_resamples)
        selected = cell[:, scen[start:stop, :, None], seeds[start:stop, None, :]]
        replicates[start:stop] = selected.mean(axis=(2, 3)).T
    return replicates


def _contrast_difference(replicates: np.ndarray, minuend_idx: Sequence[int], subtrahend_idx: Sequence[int]) -> np.ndarray:
    return replicates[:, minuend_idx].mean(axis=1) - replicates[:, subtrahend_idx].mean(axis=1)


def curve_intervals(
    observed: np.ndarray,
    replicates: np.ndarray,
    *,
    sizes: Sequence[int],
    plans: Sequence[ContrastPlan],
    ci_level: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return each size's estimate and percentile interval, and each planned contrast's, from per-size estimates and
    their replicates."""
    points = [
        {"size": int(size), "estimate": float(observed[index]), **_curve_interval(replicates[:, index], ci_level)}
        for index, size in enumerate(sizes)
    ]
    contrasts = {
        plan.name: {
            "minuend_sizes": plan.minuend,
            "subtrahend_sizes": plan.subtrahend,
            "estimate": float(observed[plan.minuend_idx].mean() - observed[plan.subtrahend_idx].mean()),
            **_curve_interval(_contrast_difference(replicates, plan.minuend_idx, plan.subtrahend_idx), ci_level),
        }
        for plan in plans
    }
    return points, contrasts


def bank_shared_curve_bootstrap(
    values: np.ndarray,
    *,
    sizes: Sequence[int],
    contrasts: Sequence[Mapping[str, Any]] = (),
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = 16,
) -> dict[str, Any]:
    """Return per-size rates for a ``(n_sizes, n_scenarios, n_subset_seeds, n_draws)`` array under a crossed cluster
    bootstrap (``two_stage``: draws resampled per scenario, one list across seeds and sizes); ``contrasts`` are
    ``{"name", "minuend": [sizes], "subtrahend": [sizes]}``."""
    validate_bootstrap_args(n_resamples, ci_level)
    array = check_curve_array(values, sizes)
    n_size, n_scen, n_seed, n_draw = array.shape
    sizes = [int(size) for size in sizes]
    plans = curve_contrast_plans(sizes, contrasts)

    rng = np.random.default_rng(seed)
    scen, seeds = crossed_cluster_indices(rng, n_resamples, n_scen, n_seed)
    # One draw list per (replicate, scenario): a bank and the full-set responses of its draws travel together.
    draws = rng.integers(0, n_draw, size=(n_resamples, n_scen, n_draw), dtype=np.int32)

    primary = crossed_cluster_means(array.mean(axis=3), scen, seeds)
    two_stage = np.empty((n_resamples, n_size))
    for start in range(0, n_resamples, _CURVE_CHUNK_TWO_STAGE):
        stop = min(start + _CURVE_CHUNK_TWO_STAGE, n_resamples)
        selected = array[
            :,
            scen[start:stop, :, None, None],
            seeds[start:stop, None, :, None],
            draws[start:stop, :, None, :],
        ]
        two_stage[start:stop] = selected.mean(axis=(2, 3, 4)).T

    points, out_contrasts = curve_intervals(array.mean(axis=(1, 2, 3)), primary, sizes=sizes, plans=plans,
                                            ci_level=ci_level)
    for index, point in enumerate(points):
        point["two_stage"] = _curve_interval(two_stage[:, index], ci_level)
    for plan in plans:
        out_contrasts[plan.name]["two_stage"] = _curve_interval(
            _contrast_difference(two_stage, plan.minuend_idx, plan.subtrahend_idx), ci_level)
    return {
        "method": CURVE_INFERENCE_METHOD,
        "interval_basis": CURVE_INTERVAL_BASIS,
        "points": points,
        "contrasts": out_contrasts,
        "n_scenarios": n_scen,
        "n_subset_seeds": n_seed,
        "n_decoder_draws": n_draw,
        "n_resamples": n_resamples,
        "ci_level": ci_level,
        "seed": seed,
    }
