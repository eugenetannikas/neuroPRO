"""Isotropic Gaussian smoothing of the normalized NM image (SPM Smooth)."""

import json
import math

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

fwhm = float(snakemake.params.fwhm)

img = nib.load(snakemake.input.nii)
data = np.asanyarray(img.dataobj).astype(np.float32)

zooms = img.header.get_zooms()[:3]
sigma_mm = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
sigma_vox = [sigma_mm / z for z in zooms]

smoothed = gaussian_filter(data, sigma=sigma_vox)

out = nib.Nifti1Image(smoothed, img.affine, img.header)
out.header.set_data_dtype(np.float32)
nib.save(out, snakemake.output.nii)

with open(snakemake.output.json, "w") as f:
    json.dump(
        {
            "Description": (
                "Realigned, averaged NM-GRE image coregistered to T1w, "
                "normalized to template space and smoothed"
            ),
            "SmoothingFWHMmm": fwhm,
            "Space": snakemake.config["template_name"],
            "ResolutionMM": snakemake.config["out_res"],
        },
        f,
        indent=2,
    )
