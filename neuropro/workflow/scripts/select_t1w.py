"""Choose the best T1w run for a subject.

Ranking:
1. per-subject override from the --t1w_choices TSV, if given
2. protocol preference: MPRAGE > FLASH (distortion corrected) > FLASH_ND
3. among the preferred protocol, --t1w_strategy 'last'/'first' by SeriesNumber

A simple sharpness metric (variance of the Laplacian over foreground voxels,
normalized by foreground intensity variance) is computed for every candidate
and stored in the output sidecar so choices can be audited.
"""

import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import laplace


def protocol_rank(desc):
    desc_upper = desc.upper()
    if "MPRAGE" in desc_upper:
        return 0
    if desc_upper.endswith("_ND"):
        return 2
    return 1


def sharpness(nii_path):
    img = nib.load(nii_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    data = np.squeeze(data)
    fg = data > np.percentile(data, 50)
    if fg.sum() == 0 or data[fg].var() == 0:
        return 0.0
    lap = laplace(data / (data[fg].mean() + 1e-9))
    return float(lap[fg].var())


subject = snakemake.wildcards.subject
niis = list(snakemake.input.niis)
jsons = list(snakemake.input.jsons)

candidates = []
for nii, sidecar in zip(niis, jsons):
    meta = {}
    if Path(sidecar).exists():
        with open(sidecar) as f:
            meta = json.load(f)
    candidates.append(
        {
            "path": nii,
            "filename": Path(nii).name,
            "series_description": meta.get("SeriesDescription", ""),
            "series_number": meta.get("SeriesNumber", -1),
            "protocol_rank": protocol_rank(meta.get("SeriesDescription", "")),
            "sharpness": sharpness(nii),
            "meta": meta,
        }
    )

# manual override
chosen = None
selection_method = None
if snakemake.params.choices_tsv:
    with open(snakemake.params.choices_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row_sub = row["subject"].removeprefix("sub-")
            if row_sub == subject:
                matches = [
                    c for c in candidates if c["filename"] == row["filename"]
                ]
                if not matches:
                    raise ValueError(
                        f"--t1w_choices entry '{row['filename']}' for "
                        f"sub-{subject} does not match any T1w candidate: "
                        f"{[c['filename'] for c in candidates]}"
                    )
                chosen = matches[0]
                selection_method = "manual (t1w_choices)"

if chosen is None:
    best_rank = min(c["protocol_rank"] for c in candidates)
    pool = [c for c in candidates if c["protocol_rank"] == best_rank]
    reverse = snakemake.params.strategy == "last"
    pool.sort(key=lambda c: c["series_number"], reverse=reverse)
    chosen = pool[0]
    selection_method = (
        f"protocol preference + {snakemake.params.strategy} SeriesNumber"
    )

shutil.copyfile(chosen["path"], snakemake.output.nii)

sidecar_out = dict(chosen["meta"])
sidecar_out["SourceFile"] = chosen["path"]
sidecar_out["SelectionMethod"] = selection_method
sidecar_out["Candidates"] = [
    {
        "filename": c["filename"],
        "series_description": c["series_description"],
        "series_number": c["series_number"],
        "protocol_rank": c["protocol_rank"],
        "sharpness": round(c["sharpness"], 6),
        "chosen": c is chosen,
    }
    for c in candidates
]
with open(snakemake.output.json, "w") as f:
    json.dump(sidecar_out, f, indent=2)
