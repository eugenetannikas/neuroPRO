"""N4 bias field correction of the selected T1w."""

import os

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

t1w = ants.image_read(snakemake.input.nii)
corrected = ants.n4_bias_field_correction(t1w)
ants.image_write(corrected, snakemake.output.nii)
