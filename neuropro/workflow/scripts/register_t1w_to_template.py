"""Affine + SyN registration of the preprocessed T1w to the template.

This provides the nonlinear deformation to standard space (the scriptable
equivalent of the forward deformation field from SPM unified segmentation).
Composite forward/inverse transforms are written as single .h5 files.

The result is scored against the template so a silently failed registration
shows up as a number in the group QC table rather than as a plausible-looking
output file.  A search-based initialization is retried only when the first
attempt is essentially unregistered, so this costs nothing on data that
already works.
"""

import json
import os
import shutil

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads))

import ants
import numpy as np

from neuropro.qc_metrics import (
    CATASTROPHIC_NMI,
    correlation,
    normalized_mutual_information,
    shared_coverage_mask,
)

t1w = ants.image_read(snakemake.input.t1w).clone("float")
template = ants.image_read(snakemake.input.template).clone("float")


def first_transform(entry):
    # with write_composite_transform=True antspyx returns the composite .h5
    # path as a plain string instead of a list
    return entry if isinstance(entry, str) else entry[0]


def attempt(initial_transform=None):
    reg = ants.registration(
        fixed=template,
        moving=t1w,
        type_of_transform="SyN",
        write_composite_transform=True,
        initial_transform=initial_transform,
    )
    warped = ants.apply_transforms(
        fixed=template,
        moving=t1w,
        transformlist=[first_transform(reg["fwdtransforms"])],
        interpolator="bSpline",
    )
    moved, fixed = warped.numpy(), template.numpy()
    mask = shared_coverage_mask(moved, fixed)
    return {
        "reg": reg,
        "nmi": normalized_mutual_information(moved, fixed, mask=mask),
        "correlation": correlation(moved, fixed, mask=mask),
    }


best = attempt()
method = "affine + SyN"

if not np.isfinite(best["nmi"]) or best["nmi"] <= CATASTROPHIC_NMI:
    initializer = ants.affine_initializer(template, t1w)
    retry = attempt(initial_transform=initializer)
    if np.isfinite(retry["nmi"]) and (
        not np.isfinite(best["nmi"]) or retry["nmi"] > best["nmi"]
    ):
        best = retry
        method = "affine + SyN after affine_initializer search (first attempt failed)"

reg = best["reg"]
shutil.copyfile(first_transform(reg["fwdtransforms"]), snakemake.output.fwd_xfm)
shutil.copyfile(first_transform(reg["invtransforms"]), snakemake.output.inv_xfm)

# resample the template grid to the requested output resolution and write
# the normalized T1w (for registration QC)
out_res = float(snakemake.params.out_res)
ref = template
if any(abs(s - out_res) > 1e-6 for s in template.spacing):
    ref = ants.resample_image(
        template, (out_res, out_res, out_res), use_voxels=False, interp_type=0
    )

t1w_tpl = ants.apply_transforms(
    fixed=ref,
    moving=t1w,
    transformlist=[snakemake.output.fwd_xfm],
    interpolator="bSpline",
)
ants.image_write(t1w_tpl, snakemake.output.t1w_tpl)

# report the quality of the image that was actually written, which sits on the
# --out_res grid rather than the template's own
moved, fixed = t1w_tpl.numpy(), ref.numpy()
mask = shared_coverage_mask(moved, fixed)
reported_nmi = normalized_mutual_information(moved, fixed, mask=mask)
reported_correlation = correlation(moved, fixed, mask=mask)

with open(snakemake.output.metrics, "w") as f:
    json.dump(
        {
            "Description": "T1w to template registration quality",
            "Method": method,
            "NormalizedMutualInformation": reported_nmi,
            "Correlation": reported_correlation,
        },
        f,
        indent=2,
    )
