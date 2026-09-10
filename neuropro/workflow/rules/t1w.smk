"""
T1w anatomical handling: choose the best run among candidates, then apply
N4 bias field correction (the scriptable equivalent of SPM's save-bias-corrected
output of unified segmentation).
"""


rule select_t1w:
    """Choose the best T1w run for a subject.

    Protocol preference: MPRAGE > FLASH (distortion corrected) > FLASH_ND.
    Among runs of the preferred protocol, --t1w_strategy picks by SeriesNumber
    ('last' by default, since re-scans are usually acquired because earlier runs
    were motion-corrupted).  A --t1w_choices TSV can override per subject.
    A sharpness metric is computed for every candidate and recorded in the
    output sidecar for QC review.
    """
    input:
        niis=get_t1w_candidates,
        jsons=get_t1w_candidate_jsons,
    params:
        strategy=config["t1w_strategy"],
        choices_tsv=config["t1w_choices"],
    output:
        nii=bids(
            root=root,
            datatype="anat",
            desc="selected",
            suffix="T1w.nii.gz",
            **subj_wildcards,
        ),
        json=bids(
            root=root,
            datatype="anat",
            desc="selected",
            suffix="T1w.json",
            **subj_wildcards,
        ),
    script:
        "../scripts/select_t1w.py"


rule n4_t1w:
    """N4 bias field correction of the selected T1w."""
    input:
        nii=rules.select_t1w.output.nii,
    output:
        nii=bids(
            root=root,
            datatype="anat",
            desc="preproc",
            suffix="T1w.nii.gz",
            **subj_wildcards,
        ),
    threads: 4
    script:
        "../scripts/n4_t1w.py"
