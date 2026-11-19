from __future__ import annotations

import numpy as np
import pytest

from ruleforge.evaluate.bootstrap import (
    percentile_bounds,
    bootstrap_ci,
    paired_bootstrap_diff_ci,
)


def test_interval_brackets_the_mean():
    values = np.array([0.0, 1.0] * 50)
    mean, low, high = bootstrap_ci(values, n_resamples=2000, seed=0)
    assert mean == pytest.approx(0.5)
    assert low < mean < high


def test_interval_is_reproducible_for_a_given_seed():
    values = np.random.default_rng(1).random(200)
    assert bootstrap_ci(values, n_resamples=500, seed=7) == bootstrap_ci(
        values, n_resamples=500, seed=7
    )


def test_a_constant_sample_has_a_zero_width_interval():
    mean, low, high = bootstrap_ci(np.full(50, 0.25), n_resamples=200, seed=0)
    assert (mean, low, high) == (0.25, 0.25, 0.25)


def test_a_single_observation_collapses_rather_than_inventing_width():
    assert bootstrap_ci(np.array([0.4])) == (0.4, 0.4, 0.4)


def test_two_observations_are_resampled_not_collapsed():
    mean, low, high = bootstrap_ci(np.array([0.0, 1.0]), n_resamples=1000, seed=0)
    assert mean == pytest.approx(0.5)
    assert (low, high) == (0.0, 1.0)


def test_two_paired_observations_are_resampled_not_collapsed():
    mean, low, high = paired_bootstrap_diff_ci(
        np.array([0.0, 1.0]), np.array([1.0, 0.0]), n_resamples=1000, seed=0
    )
    assert mean == pytest.approx(0.0)
    assert (low, high) == (-1.0, 1.0)


def test_a_wider_confidence_level_gives_a_wider_interval():
    values = np.random.default_rng(2).random(100)
    _, low95, high95 = bootstrap_ci(values, n_resamples=4000, ci_level=0.95, seed=3)
    _, low50, high50 = bootstrap_ci(values, n_resamples=4000, ci_level=0.50, seed=3)
    assert (high95 - low95) > (high50 - low50)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_resamples": 0}, "n_resamples must be at least"),
        ({"n_resamples": 99}, "n_resamples must be at least"),
        ({"ci_level": 0.0}, r"ci_level must be in \(0, 1\)"),
        ({"ci_level": 1.0}, r"ci_level must be in \(0, 1\)"),
    ],
)
def test_bootstrap_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bootstrap_ci(np.array([1.0, 2.0]), **kwargs)


def test_bootstrap_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="nothing to bootstrap"):
        bootstrap_ci(np.array([]))


def test_bootstrap_rejects_a_two_dimensional_sample():
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_ci(np.zeros((2, 2)))


def test_a_constant_offset_has_a_zero_width_paired_interval():
    base = np.random.default_rng(4).random(100)
    tuned = base - 0.2
    mean, low, high = paired_bootstrap_diff_ci(tuned, base, n_resamples=1000, seed=0)
    assert mean == pytest.approx(-0.2)
    assert low == pytest.approx(-0.2)
    assert high == pytest.approx(-0.2)


def test_pairing_gives_a_tighter_interval_than_ignoring_it():
    rng = np.random.default_rng(5)
    base = rng.random(200)
    tuned = base - 0.1 + rng.normal(0, 0.01, 200)
    _, plow, phigh = paired_bootstrap_diff_ci(tuned, base, n_resamples=2000, seed=0)

    unpaired = np.random.default_rng(0)
    diffs = [
        tuned[unpaired.integers(0, 200, 200)].mean()
        - base[unpaired.integers(0, 200, 200)].mean()
        for _ in range(2000)
    ]
    ulow, uhigh = np.percentile(diffs, [2.5, 97.5])
    assert (phigh - plow) < (uhigh - ulow)


def test_a_clear_improvement_excludes_zero():
    base = np.ones(200)
    tuned = np.concatenate([np.zeros(150), np.ones(50)])
    _, _, high = paired_bootstrap_diff_ci(tuned, base, n_resamples=2000, seed=0)
    assert high < 0


def test_paired_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="must be aligned"):
        paired_bootstrap_diff_ci(np.zeros(3), np.zeros(4))


def test_paired_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        paired_bootstrap_diff_ci(np.zeros((2, 2)), np.zeros((2, 2)))


def test_paired_rejects_empty_input():
    with pytest.raises(ValueError, match="nothing to bootstrap"):
        paired_bootstrap_diff_ci(np.array([]), np.array([]))


def test_paired_single_pair_collapses():
    assert paired_bootstrap_diff_ci(np.array([1.0]), np.array([0.25])) == (
        0.75,
        0.75,
        0.75,
    )


def test_paired_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="n_resamples must be at least"):
        paired_bootstrap_diff_ci(np.zeros(3), np.zeros(3), n_resamples=-1)


@pytest.mark.frozen
def test_bootstrap_ci_matches_its_frozen_value():
    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert bootstrap_ci(values, n_resamples=1000, seed=7) == (0.6, 0.3, 0.9)


@pytest.mark.frozen
def test_paired_bootstrap_matches_its_frozen_value():
    a = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    mean, low, high = paired_bootstrap_diff_ci(a, b, n_resamples=1000, seed=7)
    assert mean == pytest.approx(-0.1)
    assert (low, high) == (-0.5, pytest.approx(0.3))


@pytest.mark.parametrize(
    "ci_level, expected",
    [(0.95, (2.5, 97.5)), (0.80, (10.0, 90.0)), (0.50, (25.0, 75.0))],
)
def test_the_percentile_convention_is_two_sided_and_symmetric(ci_level, expected):
    bounds = percentile_bounds(np.arange(101.0), ci_level)
    assert bounds == pytest.approx(expected)


def test_a_single_resample_is_not_a_confidence_interval():
    with pytest.raises(ValueError, match="n_resamples must be at least"):
        bootstrap_ci(np.array([0.0, 1.0]), n_resamples=1)
