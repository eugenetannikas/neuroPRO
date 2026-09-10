"""Rigid coregistration of the NM average to the preprocessed T1w.

Uses a Mattes mutual-information metric (equivalent to SPM Coregister's
normalised mutual information).  The NM image is a thin slab covering the
brainstem; scanner coordinates from the same session already provide a good
initialization, and the rigid registration refines it.

That initialization is only as good as the headers, though, and it is the
assumption most likely to break on data from another site.  So the result is
scored, and a search-based initialization is tried only when the header-based
one produced something essentially unregistered -- which leaves datasets that
already work bit-for-bit untouched.
"""

import json
import os
import shutil

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants
import numpy as np

from neuropro.qc_metrics import (
    CATASTROPHIC_NMI,
    correlation,
    normalized_mutual_information,
    overlap_mask,
)

nm = ants.image_read(snakemake.input.nm).clone("float")
t1w = ants.image_read(snakemake.input.t1w).clone("float")


def attempt(initial_transform=None):
    """Register, resample, and score the result."""
    reg = ants.registration(
        fixed=t1w,
        moving=nm,
        type_of_transform="Rigid",
        aff_metric="mattes",
        initial_transform=initial_transform,
    )
    warped = ants.apply_transforms(
        fixed=t1w,
        moving=nm,
        transformlist=reg["fwdtransforms"],
        interpolator="bSpline",
    )
    moved, fixed = warped.numpy(), t1w.numpy()
    # score where the slab and the head actually overlap, otherwise the
    # metric mostly measures how much shared background there is
    mask = overlap_mask(moved, fixed)
    return {
        "reg": reg,
        "warped": warped,
        "nmi": normalized_mutual_information(moved, fixed, mask=mask),
        "correlation": correlation(moved, fixed, mask=mask),
    }


best = attempt()
method = "rigid, initialized from scanner coordinates"

if not np.isfinite(best["nmi"]) or best["nmi"] <= CATASTROPHIC_NMI:
    initializer = ants.affine_initializer(t1w, nm)
    retry = attempt(initial_transform=initializer)
    if np.isfinite(retry["nmi"]) and (
        not np.isfinite(best["nmi"]) or retry["nmi"] > best["nmi"]
    ):
        best = retry
        method = (
            "rigid, initialized by affine_initializer search "
            "(scanner-coordinate initialization failed)"
        )

# keep the rigid transform for later composition with the template warp
shutil.copyfile(best["reg"]["fwdtransforms"][0], snakemake.output.xfm)

arr = best["warped"].numpy()
arr[arr < 0] = 0
ants.image_write(best["warped"].new_image_like(arr), snakemake.output.nm_in_t1w)

with open(snakemake.output.metrics, "w") as f:
    json.dump(
        {
            "Description": "NM average to T1w rigid coregistration quality",
            "Method": method,
            "NormalizedMutualInformation": best["nmi"],
            "Correlation": best["correlation"],
        },
        f,
        indent=2,
    )
