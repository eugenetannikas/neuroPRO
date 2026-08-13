"""
Common planning-time logic for neuroPRO.

- resolves template resource paths
- collects T1w candidate runs per subject (from pybids)
- discovers NM-GRE magnitude images in sourcedata/ (non-BIDS naming),
  selecting the requested distortion-correction variant and echoes
"""

import json
import re
from pathlib import Path


def resources_path(path):
    """Get path relative to the bundled resources folder"""
    return str(Path(workflow.basedir).parent / "resources" / path)


template_name = config["template_name"]

if config["template_dir"] is not None:
    template_dir = Path(os.path.expandvars(config["template_dir"]))
else:
    template_dir = Path(resources_path(f"tpl-{template_name}"))

template_t1w = str(template_dir / f"tpl-{template_name}_res-01_T1w.nii.gz")
template_mask = str(
    template_dir / f"tpl-{template_name}_res-01_desc-brain_mask.nii.gz"
)
template_probseg = {
    tissue: str(
        template_dir / f"tpl-{template_name}_res-01_label-{tissue}_probseg.nii.gz"
    )
    for tissue in ["CSF", "GM", "WM"]
}

for _f in [template_t1w, template_mask, *template_probseg.values()]:
    if not Path(_f).exists():
        raise FileNotFoundError(
            f"Template file not found: {_f}\n"
            "Provide --template_dir or add the files to the bundled resources."
        )


# ---- subjects and their T1w candidate runs ----


def _requested_subjects():
    """All sub-* dirs in bids_dir, filtered by --participant-label /
    --exclude-participant-label"""
    found = sorted(
        p.name.removeprefix("sub-")
        for p in Path(config["bids_dir"]).glob("sub-*")
        if p.is_dir()
    )
    keep = config.get("participant_label") or None
    drop = config.get("exclude_participant_label") or None
    if keep:
        keep = {s.removeprefix("sub-") for s in keep}
        found = [s for s in found if s in keep]
    if drop:
        drop = {s.removeprefix("sub-") for s in drop}
        found = [s for s in found if s not in drop]
    return found


t1w_by_subject = {}
for _subject in _requested_subjects():
    _niis = sorted(
        (Path(config["bids_dir"]) / f"sub-{_subject}" / "anat").glob(
            f"sub-{_subject}*_T1w.nii.gz"
        )
    )
    if _niis:
        t1w_by_subject[_subject] = [str(p) for p in _niis]
    else:
        print(f"WARNING: no T1w found for sub-{_subject}; it will be skipped.")


# ---- NM-GRE magnitude images from sourcedata/ ----


def discover_nm_files(subject):
    """Return the sorted list of NM-GRE magnitude niftis for a subject.

    Each NM acquisition is exported twice by the scanner: distortion-corrected
    (SeriesDescription 'NM-GRE') and uncorrected ('NM-GRE_ND'); each version
    has one image per echo (e1..e3), and phase images carry a '_ph' filename
    suffix.  We keep magnitude images of the requested variant and echoes.
    """
    nm_dir = (
        Path(config["bids_dir"])
        / f"sub-{subject}"
        / "sourcedata"
        / f"sub-{subject}"
        / "nm-gre"
    )
    found = []
    for sidecar in sorted(nm_dir.glob("NM-GRE_s*_e*.json")):
        m = re.fullmatch(r"NM-GRE_s(\d+)_e(\d+)", sidecar.stem)
        if m is None:
            # phase images (_ph) or unexpected names
            continue
        series, echo = int(m.group(1)), int(m.group(2))
        if echo not in config["nm_echoes"]:
            continue
        with open(sidecar) as f:
            desc = json.load(f).get("SeriesDescription", "")
        is_nd = desc.endswith("_ND")
        if (config["nm_variant"] == "uncorrected") != is_nd:
            continue
        nii = sidecar.with_suffix(".nii.gz")
        if nii.exists():
            found.append((series, echo, str(nii)))
    return [path for _, _, path in sorted(found)]


nm_by_subject = {}
for _subject in sorted(t1w_by_subject):
    _files = discover_nm_files(_subject)
    if _files:
        nm_by_subject[_subject] = _files

_missing_nm = sorted(set(t1w_by_subject) - set(nm_by_subject))
if _missing_nm:
    print(
        "WARNING: no NM-GRE magnitude images found for subject(s) "
        f"{', '.join('sub-' + s for s in _missing_nm)}; they will be skipped."
    )

subjects = sorted(nm_by_subject)

if len(subjects) == 0:
    raise ValueError("No subjects with both T1w and NM-GRE images were found")


def get_t1w_candidates(wildcards):
    return t1w_by_subject[wildcards.subject]


def get_t1w_candidate_jsons(wildcards):
    return [
        str(Path(p).with_name(Path(p).name.replace(".nii.gz", ".json")))
        for p in t1w_by_subject[wildcards.subject]
    ]


def get_nm_files(wildcards):
    return nm_by_subject[wildcards.subject]
