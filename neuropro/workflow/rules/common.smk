"""
Common planning-time logic for neuroPRO.

- resolves template resource paths
- scans the dataset for NM-GRE and T1w images (see neuropro/discovery.py,
  which handles the several naming conventions we encounter in practice and
  is unit tested against all of them)
- decides whether the dataset is sessioned, and exposes the wildcard set the
  rules build their output paths from
"""

import sys
from pathlib import Path

from neuropro.discovery import scan_dataset, summarize


def resources_path(path):
    """Get path relative to the bundled resources folder"""
    return str(Path(workflow.basedir).parent / "resources" / path)


# ---- template resources ----

template_name = config["template_name"]

if config["template_dir"] is not None:
    template_dir = Path(os.path.expandvars(config["template_dir"]))
else:
    template_dir = Path(resources_path(f"tpl-{template_name}"))


def template_file(suffix, **entities):
    """Locate a template resource, tolerating entity spelling variations.

    TemplateFlow names are not uniform across templates: the resolution entity
    may be written 'res-01' or 'res-1', and cohort-specific (e.g. paediatric)
    templates carry an extra 'cohort-' entity.  Rather than hard-coding one
    spelling, glob on the entities that actually identify the file.  When
    several resolutions are present the highest (numerically smallest 'res-')
    is preferred, since everything is resampled to --out_res anyway.
    """
    parts = "".join(f"*{key}-{value}" for key, value in entities.items())
    matches = sorted(template_dir.glob(f"tpl-{template_name}{parts}*_{suffix}.nii.gz"))
    if not matches:
        listing = "\n".join(f"    {p.name}" for p in sorted(template_dir.glob("*")))
        raise FileNotFoundError(
            f"No template file for '{suffix}' ({entities}) in {template_dir}\n"
            f"  looked for: tpl-{template_name}{parts}*_{suffix}.nii.gz\n"
            f"  directory contains:\n{listing or '    (empty)'}\n"
            "Point --template_dir at a directory using TemplateFlow naming."
        )
    return str(matches[0])


template_t1w = template_file("T1w", res="01")
template_mask = template_file("mask", res="01", desc="brain")
template_probseg = {
    tissue: template_file("probseg", res="01", label=tissue)
    for tissue in ["CSF", "GM", "WM"]
}


# ---- dataset scan ----

entries = scan_dataset(
    bids_dir=Path(config["bids_dir"]),
    participant_label=config.get("participant_label"),
    exclude_participant_label=config.get("exclude_participant_label"),
    nm_layout=config["nm_layout"],
    nm_pattern=config["nm_pattern"],
    nm_variant=config["nm_variant"],
    nm_echoes=config["nm_echoes"],
    include_combecho=config["include_combecho"],
)

print(summarize(entries), file=sys.stderr)

usable = [e for e in entries if e.nm and e.t1w]

_no_nm = [e.label for e in entries if not e.nm]
if _no_nm:
    print(
        f"WARNING: no NM-GRE images found for {', '.join(_no_nm)}; skipping.\n"
        "  If this dataset uses a naming convention neuroPRO does not know, "
        "set --nm_layout or --nm_pattern.",
        file=sys.stderr,
    )

_no_t1w = [e.label for e in entries if e.nm and not e.t1w]
if _no_t1w:
    print(
        f"WARNING: no T1w found for {', '.join(_no_t1w)}; skipping.",
        file=sys.stderr,
    )

if not usable:
    raise ValueError(
        "No subjects with both NM-GRE and T1w images were found in "
        f"{config['bids_dir']}.\n"
        "See the scan summary above for what was found per subject."
    )


# ---- sessions ----
#
# Rules build static output paths, so the session entity is either present for
# the whole run or absent for the whole run.  A dataset that mixes the two is
# genuinely ambiguous rather than merely awkward, so it is rejected with an
# explanation instead of guessed at.

_sessioned = [e for e in usable if e.session is not None]
if _sessioned and len(_sessioned) != len(usable):
    _flat = ", ".join(e.label for e in usable if e.session is None)
    raise ValueError(
        "This dataset mixes sessioned and unsessioned subjects, which cannot "
        "share one set of output paths.\n"
        f"  without a session: {_flat}\n"
        "Process the two groups separately with --participant-label."
    )

use_sessions = bool(_sessioned)

#: Wildcard placeholders every rule splats into bids(), so that the session
#: entity appears in output paths only when the dataset actually has sessions.
subj_wildcards = {"subject": "{subject}"}
if use_sessions:
    subj_wildcards["session"] = "{session}"

#: Parallel lists for zip-expanding a target over every subject/session pair.
target_entities = {"subject": [e.subject for e in usable]}
if use_sessions:
    target_entities["session"] = [e.session for e in usable]

inputs_by_key = {e.key: e for e in usable}


def entry_for(wildcards):
    """The SubjectInputs matching a rule's wildcards."""
    session = wildcards.session if use_sessions else None
    return inputs_by_key[(wildcards.subject, session)]


def get_t1w_candidates(wildcards):
    return [t.path for t in entry_for(wildcards).t1w]


def get_t1w_candidate_jsons(wildcards):
    """Existing sidecars for the T1w candidates.

    A nonexistent input file would make Snakemake refuse to run the job, so
    absent sidecars are omitted; select_t1w treats their metadata as unknown.
    """
    from neuropro.discovery import sidecar_for

    found = []
    for candidate in entry_for(wildcards).t1w:
        sidecar = sidecar_for(Path(candidate.path))
        if sidecar is not None:
            found.append(str(sidecar))
    return found


def get_nm_files(wildcards):
    return [i.path for i in entry_for(wildcards).nm]
