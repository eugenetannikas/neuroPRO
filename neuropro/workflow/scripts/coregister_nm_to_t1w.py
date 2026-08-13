"""Rigid coregistration of the NM average to the preprocessed T1w.

Uses a Mattes mutual-information metric (equivalent to SPM Coregister's
normalised mutual information).  The NM image is a thin slab covering the
brainstem; scanner coordinates from the same session already provide a good
initialization, and the rigid registration refines it.
"""

import os
import shutil

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

nm = ants.image_read(snakemake.input.nm).clone("float")
t1w = ants.image_read(snakemake.input.t1w).clone("float")

reg = ants.registration(
    fixed=t1w,
    moving=nm,
    type_of_transform="Rigid",
    aff_metric="mattes",
)

# keep the rigid transform for later composition with the template warp
shutil.copyfile(reg["fwdtransforms"][0], snakemake.output.xfm)

nm_in_t1w = ants.apply_transforms(
    fixed=t1w,
    moving=nm,
    transformlist=[snakemake.output.xfm],
    interpolator="bSpline",
)
arr = nm_in_t1w.numpy()
arr[arr < 0] = 0
ants.image_write(nm_in_t1w.new_image_like(arr), snakemake.output.nm_in_t1w)
