"""Quantitative registration-quality metrics, and cohort-relative flagging.

The QC snapshots show a human whether a registration worked.  That does not
scale: at n=100 nobody reliably eyeballs 300 PNGs, so a failed coregistration
reaches the ROI stage looking exactly like a successful one in the file
listing.  These metrics put a number on each registration so the group table
can point at the subjects worth looking at.

Deliberately no absolute pass/fail thresholds.  What counts as a good mutual
information score depends on the sequence, the field strength and the FOV, so
a threshold tuned on CABIN would mislabel every 7T Berkeley scan.  Instead
each metric is compared against the rest of the cohort (median +/- a multiple
of the MAD), which adapts to whatever dataset is being run.  The one absolute
number here is CATASTROPHIC_NMI, a floor low enough to mean "these two images
are essentially unregistered" on any sequence.
"""

from __future__ import annotations

import numpy as np

#: Normalised mutual information at or below this means the registration did
#: not find any correspondence at all.  Studholme NMI is 1.0 for statistically
#: independent images, so this is "barely above independent" rather than a
#: quality bar -- it triggers a retry, not a warning.
CATASTROPHIC_NMI = 1.02


def normalized_mutual_information(a, b, bins: int = 64, mask=None) -> float:
    """Studholme normalised mutual information, (H(a)+H(b)) / H(a,b).

    1.0 for independent images, 2.0 for identical ones.  Unlike a plain
    correlation this is meaningful across modalities, which is what we need
    for NM-to-T1w and T1w-to-template.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size == 0 or a.std() == 0 or b.std() == 0:
        return float("nan")

    hist, _, _ = np.histogram2d(a, b, bins=bins)
    joint = hist / hist.sum()
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    def entropy(p):
        nz = p[p > 0]
        return float(-np.sum(nz * np.log(nz)))

    h_joint = entropy(joint)
    if h_joint <= 0:
        return float("nan")
    return (entropy(p_a) + entropy(p_b)) / h_joint


def correlation(a, b, mask=None) -> float:
    """Pearson correlation, over a mask when given."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a, b = a[mask], b[mask]
    else:
        a, b = a.ravel(), b.ravel()

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def shared_coverage_mask(a, b, background_fraction: float = 0.02):
    """Voxels where both images actually hold data.

    Scoring the whole volume would mostly measure how much background the two
    images share -- the NM slab covers about a fifth of the T1w -- so the
    metric is restricted to where they overlap.  The threshold is deliberately
    low, because it is a *coverage* test rather than a foreground one: keeping
    the full range of tissue intensities is what gives mutual information
    something to work with.  Selecting only bright voxels instead collapses
    the joint histogram onto a narrow intensity band and drives the score
    towards statistical independence no matter how good the alignment is.

    Thresholds are set from the 99.5th percentile rather than the maximum, so
    a single hot voxel cannot drag them up.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_finite, b_finite = a[np.isfinite(a)], b[np.isfinite(b)]
    if a_finite.size == 0 or b_finite.size == 0:
        return np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=bool)

    a_high = np.percentile(a_finite, 99.5)
    b_high = np.percentile(b_finite, 99.5)
    both = (a > a_high * background_fraction) & (b > b_high * background_fraction)
    if both.any():
        return both
    # A near-binary image (a mask, say) can put every voxel at or below the
    # threshold.  Fall back to plain nonzero rather than handing the caller an
    # empty mask, which would score as NaN.
    return (a > 0) & (b > 0)


def flag_low_outliers(values, n_mad: float = 3.0, min_cohort: int = 5) -> list[bool]:
    """Which values sit far below the cohort median.

    Uses the median and MAD rather than mean/SD so a couple of genuinely
    failed registrations cannot drag the threshold down far enough to hide
    themselves.  Returns all-False for a cohort too small to have a
    meaningful spread -- with three subjects, "outlier" is not a
    well-defined idea, and a false alarm on every run trains people to
    ignore the column.
    """
    values = list(values)
    usable = [v for v in values if v is not None and np.isfinite(v)]
    if len(usable) < min_cohort:
        return [False] * len(values)

    median = float(np.median(usable))
    mad = float(np.median(np.abs(np.asarray(usable) - median)))
    if mad == 0:
        return [False] * len(values)

    # 1.4826 makes the MAD comparable to a standard deviation for normal data
    threshold = median - n_mad * 1.4826 * mad
    # bool() so callers get plain Python booleans rather than np.bool_ leaking
    # out of the isfinite check
    return [bool(v is not None and np.isfinite(v) and v < threshold) for v in values]
