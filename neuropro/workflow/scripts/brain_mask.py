"""Brain mask in native T1w space, from the template mask.

The template brain mask is warped into the subject's T1w space with the
inverse of the T1w->template transform, thresholded, and cleaned up (largest
connected component, holes filled).  It is the same mask the former
segmentation step produced as a by-product, kept as its own output because
downstream tools (the CABIN template build) consume it.
"""

import os

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

t1w = ants.image_read(snakemake.input.t1w).clone("float")
mask = ants.apply_transforms(
    fixed=t1w,
    moving=ants.image_read(snakemake.input.template_mask).clone("float"),
    transformlist=[snakemake.input.inv_xfm],
    interpolator="linear",
)
mask = ants.threshold_image(mask, 0.5, 1e9)
mask = ants.iMath(mask, "GetLargestComponent")
mask = ants.iMath(mask, "FillHoles")
ants.image_write(mask, snakemake.output.mask)
