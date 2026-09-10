"""Tests for registration-quality metrics and cohort-relative flagging."""

from __future__ import annotations

import numpy as np
import pytest

from neuropro.qc_metrics import (
    CATASTROPHIC_NMI,
    correlation,
    flag_low_outliers,
    normalized_mutual_information,
    overlap_mask,
)


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.random((24, 24, 24))


# --------------------------------------------------------------------------
# normalized mutual information
# --------------------------------------------------------------------------


def test_nmi_of_identical_images_is_maximal(volume):
    """Studholme NMI is 2.0 when the images are the same."""
    assert normalized_mutual_information(volume, volume) == pytest.approx(2.0)


def test_nmi_of_independent_images_is_near_one(volume):
    rng = np.random.default_rng(1)
    other = rng.random(volume.shape)
    assert normalized_mutual_information(volume, other) < 1.1


def test_nmi_survives_a_monotonic_intensity_change(volume):
    """MI is the right metric across modalities: it ignores the mapping."""
    remapped = np.exp(volume * 3)
    assert normalized_mutual_information(volume, remapped) > 1.5


def test_nmi_detects_misalignment(volume):
    """A shifted copy must score below a matched one."""
    smooth = np.zeros((24, 24, 24))
    smooth[6:18, 6:18, 6:18] = 1.0
    shifted = np.roll(smooth, 8, axis=0)
    assert normalized_mutual_information(
        smooth, smooth
    ) > normalized_mutual_information(smooth, shifted)


def test_catastrophic_floor_sits_above_independence(volume):
    """The retry floor must actually catch an unregistered pair."""
    rng = np.random.default_rng(2)
    independent = rng.random(volume.shape)
    assert normalized_mutual_information(volume, independent) <= CATASTROPHIC_NMI
    assert normalized_mutual_information(volume, volume) > CATASTROPHIC_NMI


def test_nmi_of_constant_image_is_nan(volume):
    assert np.isnan(normalized_mutual_information(volume, np.ones_like(volume)))


def test_nmi_ignores_non_finite_voxels(volume):
    spoiled = volume.copy()
    spoiled[0, 0, 0] = np.nan
    spoiled[0, 0, 1] = np.inf
    assert np.isfinite(normalized_mutual_information(spoiled, volume))


def test_nmi_honours_a_mask(volume):
    mask = np.zeros(volume.shape, dtype=bool)
    mask[:12] = True
    assert np.isfinite(normalized_mutual_information(volume, volume, mask=mask))


def test_nmi_of_empty_mask_is_nan(volume):
    mask = np.zeros(volume.shape, dtype=bool)
    assert np.isnan(normalized_mutual_information(volume, volume, mask=mask))


# --------------------------------------------------------------------------
# correlation and overlap
# --------------------------------------------------------------------------


def test_correlation_basics(volume):
    assert correlation(volume, volume) == pytest.approx(1.0)
    assert correlation(volume, -volume) == pytest.approx(-1.0)
    assert np.isnan(correlation(volume, np.ones_like(volume)))


def test_overlap_mask_selects_shared_foreground():
    rng = np.random.default_rng(3)
    a = rng.random((10, 10, 10)) * 0.1
    b = rng.random((10, 10, 10)) * 0.1
    a[:6] += 10.0  # a's slab
    b[4:] += 10.0  # b's slab, overlapping in the 4:6 band
    mask = overlap_mask(a, b)
    assert mask[:4].sum() == 0
    assert mask[6:].sum() == 0
    assert mask[4:6].sum() > 0


def test_overlap_mask_falls_back_for_near_binary_images():
    """A percentile can land on the foreground value and select nothing."""
    a = np.zeros((10, 10, 10))
    b = np.zeros((10, 10, 10))
    a[:6] = 1.0
    b[4:] = 1.0
    mask = overlap_mask(a, b)
    assert mask.any()
    assert mask[4:6].all()


# --------------------------------------------------------------------------
# cohort-relative flagging
# --------------------------------------------------------------------------


def test_flags_a_clear_outlier():
    values = [1.5, 1.52, 1.48, 1.51, 1.49, 1.50, 1.02]
    flags = flag_low_outliers(values)
    assert flags[-1] is True
    assert not any(flags[:-1])


def test_does_not_flag_a_uniform_cohort():
    assert not any(flag_low_outliers([1.5, 1.52, 1.48, 1.51, 1.49, 1.50]))


def test_only_low_values_are_flagged():
    """A registration that scores unusually well is not a problem."""
    values = [1.5, 1.52, 1.48, 1.51, 1.49, 1.50, 1.95]
    assert not any(flag_low_outliers(values))


def test_small_cohort_is_never_flagged():
    """With three subjects 'outlier' is not a well-defined idea."""
    assert not any(flag_low_outliers([1.5, 1.5, 1.02]))


def test_majority_failure_does_not_hide_itself():
    """MAD-based thresholds resist a few failures dragging the bar down."""
    values = [1.5, 1.51, 1.49, 1.50, 1.52, 1.03, 1.02]
    flags = flag_low_outliers(values)
    assert flags[-1] and flags[-2]


def test_handles_missing_and_non_finite_values():
    values = [1.5, 1.52, 1.48, 1.51, 1.49, None, float("nan"), 1.01]
    flags = flag_low_outliers(values)
    assert len(flags) == len(values)
    assert flags[-1] is True
    assert flags[5] is False and flags[6] is False


def test_zero_spread_cohort_is_not_flagged():
    assert not any(flag_low_outliers([1.5] * 8))


def test_empty_cohort():
    assert flag_low_outliers([]) == []
