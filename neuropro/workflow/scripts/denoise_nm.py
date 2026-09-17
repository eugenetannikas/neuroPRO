"""Non-local-means denoising of the realigned NM average, in native space.

Replaces the Gaussian smoothing of the SPM workflow. NLM averages each voxel
with the voxels whose surrounding patches look alike, wherever they are in the
image, so noise is suppressed without blurring the edges that a Gaussian kernel
smears -- the LC and SN are 2-3 mm structures and keeping their boundaries is
the point. The raw average is kept as a separate output so noise statistics
(background SD, within-ROI SD) can still be measured on undenoised data.

Uses ANTs DenoiseImage (Manjon et al. 2010) with a Rician noise model, which is
the right model for magnitude MR images.
"""

import json
import os

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants

img = ants.image_read(snakemake.input.nii).clone("float")
denoised = ants.denoise_image(img, noise_model="Rician")
ants.image_write(denoised, snakemake.output.nii)

with open(snakemake.output.json, "w") as f:
    json.dump(
        {
            "Description": (
                "Realigned, averaged NM-GRE image, non-local-means denoised "
                "(native space)"
            ),
            "Sources": [snakemake.input.nii],
            "DenoisingMethod": "ANTs DenoiseImage (non-local means)",
            "NoiseModel": "Rician",
        },
        f,
        indent=2,
    )
