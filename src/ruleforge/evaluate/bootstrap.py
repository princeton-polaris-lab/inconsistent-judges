from __future__ import annotations

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_CI_LEVEL = 0.95
MIN_RESAMPLES = 100


def percentile_bounds(
    samples: np.ndarray,
    ci_level: float,
) -> tuple[float, float]:
    """Two-sided percentile bounds at ``ci_level``."""
    alpha = 1.0 - ci_level
    return (
        float(np.percentile(samples, alpha / 2 * 100)),
        float(np.percentile(samples, (1 - alpha / 2) * 100)),
    )


def validate_bootstrap_args(n_resamples: int, ci_level: float) -> None:
    # Few replicates give an erratic or zero-width interval that still looks like a confidence interval.
    if n_resamples < MIN_RESAMPLES:
        raise ValueError(
            f"n_resamples must be at least {MIN_RESAMPLES}, got {n_resamples}"
        )
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)``; a single observation's interval is the point itself."""
    validate_bootstrap_args(n_resamples, ci_level)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("values is empty; nothing to bootstrap")
    if values.size < 2:
        point = float(values.mean())
        return point, point, point

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    ci_low, ci_high = percentile_bounds(means, ci_level)
    return float(values.mean()), ci_low, ci_high


def paired_bootstrap_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return ``(mean_diff, ci_low, ci_high)`` for ``mean(a) - mean(b)``, applying one scenario resample to both
    arrays."""
    validate_bootstrap_args(n_resamples, ci_level)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"values_a and values_b must be aligned, got {a.shape} vs {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("values are empty; nothing to bootstrap")

    mean_diff = float(a.mean() - b.mean())
    if a.size < 2:
        return mean_diff, mean_diff, mean_diff

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, a.size, size=(n_resamples, a.size))
    diffs = a[indices].mean(axis=1) - b[indices].mean(axis=1)
    ci_low, ci_high = percentile_bounds(diffs, ci_level)
    return mean_diff, ci_low, ci_high
