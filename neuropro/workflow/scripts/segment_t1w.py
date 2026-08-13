"""Atropos 3-class tissue segmentation of the T1w using template priors.

Template CSF/GM/WM probability maps and the brain mask are warped to native
space with the inverse of the T1w->template transform, then used as priors
for Atropos (prior-based finite mixture segmentation; the scriptable
equivalent of SPM unified segmentation tissue classes 1-3).

Output dseg labels: 1=CSF, 2=GM, 3=WM.
"""

import os

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

t1w = ants.image_read(snakemake.input.t1w).clone("float")
inv_xfm = snakemake.input.inv_xfm


def warp_to_native(path):
    moving = ants.image_read(path).clone("float")
    return ants.apply_transforms(
        fixed=t1w,
        moving=moving,
        transformlist=[inv_xfm],
        interpolator="linear",
    )


priors = [
    warp_to_native(snakemake.input.prior_csf),
    warp_to_native(snakemake.input.prior_gm),
    warp_to_native(snakemake.input.prior_wm),
]

mask = warp_to_native(snakemake.input.template_mask)
mask = ants.threshold_image(mask, 0.5, 1e9)
mask = ants.iMath(mask, "GetLargestComponent")
mask = ants.iMath(mask, "FillHoles")
ants.image_write(mask, snakemake.output.mask)

seg = ants.atropos(
    a=t1w,
    x=mask,
    i=priors,
    m="[0.1,1x1x1]",
    c="[5,0]",
    priorweight=0.25,
)

ants.image_write(seg["probabilityimages"][0], snakemake.output.probseg_csf)
ants.image_write(seg["probabilityimages"][1], snakemake.output.probseg_gm)
ants.image_write(seg["probabilityimages"][2], snakemake.output.probseg_wm)
ants.image_write(seg["segmentation"], snakemake.output.dseg)
