"""Affine + SyN registration of the preprocessed T1w to the template.

This provides the nonlinear deformation to standard space (the scriptable
equivalent of the forward deformation field from SPM unified segmentation).
Composite forward/inverse transforms are written as single .h5 files.
"""

import os
import shutil

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

t1w = ants.image_read(snakemake.input.t1w).clone("float")
template = ants.image_read(snakemake.input.template).clone("float")

reg = ants.registration(
    fixed=template,
    moving=t1w,
    type_of_transform="SyN",
    write_composite_transform=True,
)

def first_transform(entry):
    # with write_composite_transform=True antspyx returns the composite .h5
    # path as a plain string instead of a list
    return entry if isinstance(entry, str) else entry[0]


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
