"""Choosing which T1w run to use when a subject has several.

Kept separate from the Snakemake script so the ranking rules can be unit
tested without a workflow around them.

The ordering is:

1. an explicit per-subject override from ``--t1w_choices``
2. protocol preference -- a recognised 3D T1 sequence beats an unrecognised
   one, and both beat a distortion-uncorrected (``_ND``) export
3. ``--t1w_strategy`` among the survivors

Every step degrades rather than failing: an unrecognised protocol name is
merely unpreferred, not disqualifying, and when SeriesNumber is missing the
strategy falls back to the ``run-`` entity and then to image sharpness, so a
dataset without sidecars still gets a defensible, reproducible choice instead
of whatever the filesystem happened to list first.
"""

from __future__ import annotations

import re

import nibabel as nib
import numpy as np
from scipy.ndimage import laplace

#: Vendor names for the 3D T1-weighted gradient-echo sequences NM-MRI studies
#: pair with an NM acquisition.  Siemens MPRAGE/MP2RAGE, GE BRAVO/FSPGR,
#: Philips TFE.  Matched case-insensitively against SeriesDescription.
PREFERRED_PROTOCOLS = (
    "MPRAGE",
    "MP2RAGE",
    "MPR",
    "BRAVO",
    "FSPGR",
    "SPGR",
    "TFE",
    "MPNRAGE",
)

STRATEGIES = ("last", "first", "sharpest")


def protocol_rank(description: str) -> int:
    """0 = recognised 3D T1 sequence, 1 = unrecognised, 2 = uncorrected export.

    Rank 2 is for the ``_ND`` (no distortion correction) copy the scanner
    exports alongside the corrected one; it is usable but never preferred.
    """
    text = (description or "").upper()
    if text.endswith("_ND"):
        return 2
    if any(name in text for name in PREFERRED_PROTOCOLS):
        return 0
    return 1


def run_entity(filename: str) -> int | None:
    """The BIDS ``run-`` index from a filename, when it has one."""
    match = re.search(r"(?:^|_)run-(\d+)", filename)
    return int(match.group(1)) if match else None


def sharpness(nii_path: str) -> float:
    """Variance of the Laplacian over foreground voxels.

    A motion-corrupted or otherwise blurred run scores lower than a clean one.
    Normalising by the foreground mean makes the number comparable between
    runs with different receiver gain.  Returns 0.0 for degenerate images
    rather than raising, so one unreadable candidate cannot fail the job.
    """
    data = np.asanyarray(nib.load(nii_path).dataobj).astype(np.float32)
    data = np.squeeze(data)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if data.size == 0:
        return 0.0
    foreground = data > np.percentile(data, 50)
    if foreground.sum() == 0 or data[foreground].var() == 0:
        return 0.0
    scaled = data / (data[foreground].mean() + 1e-9)
    return float(laplace(scaled)[foreground].var())


def choose(candidates: list[dict], strategy: str) -> tuple[dict, str]:
    """Pick one candidate and describe how it was picked.

    ``candidates`` are dicts with ``filename``, ``series_number`` (int or
    None), and ``sharpness``.  Returns the winner and a human-readable
    selection method for the output sidecar and the group QC table.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown --t1w_strategy '{strategy}'; expected one of "
            f"{', '.join(STRATEGIES)}"
        )

    best_rank = min(c["protocol_rank"] for c in candidates)
    pool = [c for c in candidates if c["protocol_rank"] == best_rank]
    if len(pool) == 1:
        return pool[0], f"only candidate at protocol rank {best_rank}"

    if strategy == "sharpest":
        chosen = max(pool, key=lambda c: c["sharpness"])
        return chosen, "sharpest of the preferred protocol"

    reverse = strategy == "last"

    # SeriesNumber is the intended ordering: a re-scan is usually acquired
    # because the earlier run was motion-corrupted.  Fall back only when it
    # is genuinely unavailable, so the choice stays reproducible.
    if all(c["series_number"] is not None for c in pool):
        pool = sorted(pool, key=lambda c: c["series_number"], reverse=reverse)
        return pool[0], f"protocol preference + {strategy} SeriesNumber"

    runs = {c["filename"]: run_entity(c["filename"]) for c in pool}
    if all(value is not None for value in runs.values()):
        pool = sorted(pool, key=lambda c: runs[c["filename"]], reverse=reverse)
        return pool[0], (
            f"protocol preference + {strategy} run- entity "
            "(SeriesNumber unavailable)"
        )

    chosen = max(pool, key=lambda c: c["sharpness"])
    return chosen, (
        "sharpest of the preferred protocol "
        "(neither SeriesNumber nor run- entity available)"
    )
