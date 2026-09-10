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
from neuropro.templates import find_template_file


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
    """Locate a template resource (see neuropro/templates.py)."""
    return find_template_file(template_dir, template_name, suffix, **entities)


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

processable = [e for e in entries if e.nm]
full_entries = [e for e in processable if e.t1w]
nmonly_entries = [e for e in processable if not e.t1w]

_no_nm = [e.label for e in entries if not e.nm]
if _no_nm:
    print(
        f"WARNING: no NM-GRE images found for {', '.join(_no_nm)}; skipping.\n"
        "  If this dataset uses a naming convention neuroPRO does not know, "
        "set --nm_layout or --nm_pattern.",
        file=sys.stderr,
    )

if nmonly_entries:
    print(
        f"NOTE: no T1w for {', '.join(e.label for e in nmonly_entries)}. "
        "These are processed as far as the data allows -- realignment, "
        "averaging and motion QC in native NM space -- but cannot be "
        "coregistered, segmented or normalized to the template.",
        file=sys.stderr,
    )

if not processable:
    raise ValueError(
        "No subjects with NM-GRE images were found in "
        f"{config['bids_dir']}.\n"
        "See the scan summary above for what was found per subject.  If the "
        "images are there but were not recognised, set --nm_layout or "
        "--nm_pattern."
    )


# ---- sessions ----
#
# Rules build static output paths, so the session entity is either present for
# the whole run or absent for the whole run.  A dataset that mixes the two is
# genuinely ambiguous rather than merely awkward, so it is rejected with an
# explanation instead of guessed at.

_sessioned = [e for e in processable if e.session is not None]
if _sessioned and len(_sessioned) != len(processable):
    _flat = ", ".join(e.label for e in processable if e.session is None)
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


def entities_of(entry_list):
    """Parallel lists for zip-expanding a target over the given entries."""
    values = {"subject": [e.subject for e in entry_list]}
    if use_sessions:
        values["session"] = [e.session for e in entry_list]
    return values


inputs_by_key = {e.key: e for e in processable}


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
