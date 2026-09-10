"""Choose the best T1w run for a subject and copy it into the derivatives.

The ranking rules live in neuropro/t1w_selection.py so they can be unit
tested; this script is the Snakemake wrapper around them.  A sharpness metric
is computed for every candidate and stored in the output sidecar, so a choice
can be audited (and so --t1w_strategy sharpest has something to sort on).
"""

import csv
import json
import shutil
from pathlib import Path

import nibabel as nib

from neuropro.discovery import sidecar_for, strip_nifti_ext
from neuropro.t1w_selection import choose, protocol_rank, sharpness

subject = snakemake.wildcards.subject
niis = list(snakemake.input.niis)

# Sidecars are matched by name rather than by position: only the candidates
# that actually have one are passed in as inputs, so the two lists are not
# necessarily parallel.
sidecars = {Path(p).name: p for p in snakemake.input.jsons}

candidates = []
for nii in niis:
    meta = {}
    sidecar = sidecar_for(Path(nii))
    if sidecar is not None and sidecar.name in sidecars:
        with open(sidecar) as f:
            meta = json.load(f)
    series_number = meta.get("SeriesNumber")
    candidates.append(
        {
            "path": nii,
            "filename": Path(nii).name,
            "series_description": meta.get("SeriesDescription", ""),
            "series_number": series_number if isinstance(series_number, int) else None,
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
            if row["subject"].removeprefix("sub-") != subject:
                continue
            matches = [c for c in candidates if c["filename"] == row["filename"]]
            if not matches:
                raise ValueError(
                    f"--t1w_choices entry '{row['filename']}' for "
                    f"sub-{subject} does not match any T1w candidate: "
                    f"{[c['filename'] for c in candidates]}"
                )
            chosen = matches[0]
            selection_method = "manual (t1w_choices)"

if chosen is None:
    chosen, selection_method = choose(candidates, snakemake.params.strategy)

# The chosen run may be uncompressed (.nii) while the output is always
# .nii.gz, so copy the bytes only when the compression already matches.
if strip_nifti_ext(chosen["path"]) + ".nii.gz" == chosen["path"]:
    shutil.copyfile(chosen["path"], snakemake.output.nii)
else:
    nib.save(nib.load(chosen["path"]), snakemake.output.nii)

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
