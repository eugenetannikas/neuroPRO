"""Normalize the NM average to template space.

Composes the NM->T1w rigid with the T1w->template (affine+SyN) transform and
resamples the native NM average onto the template grid at the requested
resolution in a single interpolation step (the scriptable equivalent of SPM
Normalise: Write applied to the averaged NM image).
"""

import os

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

nm = ants.image_read(snakemake.input.nm).clone("float")
template = ants.image_read(snakemake.input.template).clone("float")

out_res = float(snakemake.params.out_res)
ref = template
if any(abs(s - out_res) > 1e-6 for s in template.spacing):
    ref = ants.resample_image(
        template, (out_res, out_res, out_res), use_voxels=False, interp_type=0
    )

# transforms are applied to points fixed->moving; list order means the
# rigid (last) is the first transform on the moving-image side
nm_tpl = ants.apply_transforms(
    fixed=ref,
    moving=nm,
    transformlist=[snakemake.input.t1w_to_tpl, snakemake.input.nm_to_t1w],
    interpolator="bSpline",
)
arr = nm_tpl.numpy()
arr[arr < 0] = 0
ants.image_write(nm_tpl.new_image_like(arr), snakemake.output.nm_tpl)
