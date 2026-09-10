"""
Quality control snapshots (participant level) and group-level aggregation.
"""


rule qc_coreg:
    """NM average in T1w space with T1w edge overlay."""
    input:
        bg=rules.coregister_nm_to_t1w.output.nm_in_t1w,
        overlay=rules.n4_t1w.output.nii,
    params:
        mode="coreg",
    output:
        png=bids(
            root=root,
            datatype="qc",
            desc="coreg",
            suffix="qc.png",
            **subj_wildcards,
        ),
    script:
        "../scripts/qc_snapshots.py"


rule qc_seg:
    """T1w with tissue segmentation boundaries."""
    input:
        bg=rules.n4_t1w.output.nii,
        overlay=rules.segment_t1w.output.dseg,
    params:
        mode="seg",
    output:
        png=bids(
            root=root,
            datatype="qc",
            desc="seg",
            suffix="qc.png",
            **subj_wildcards,
        ),
    script:
        "../scripts/qc_snapshots.py"


rule qc_norm:
    """Template with normalized T1w edges + normalized NM and brainstem zoom."""
    input:
        template=template_t1w,
        t1w_tpl=rules.register_t1w_to_template.output.t1w_tpl,
        nm_tpl=rules.smooth_nm.output.nii,
    params:
        mode="norm",
    output:
        png=bids(
            root=root,
            datatype="qc",
            desc="norm",
            suffix="qc.png",
            **subj_wildcards,
        ),
    script:
        "../scripts/qc_snapshots.py"


rule qc_nm:
    """Native-space NM average, for entries with no T1w to overlay."""
    input:
        nm=rules.realign_average_nm.output.avg,
    params:
        mode="nm",
    output:
        png=bids(
            root=root,
            datatype="qc",
            desc="nm",
            suffix="qc.png",
            **subj_wildcards,
        ),
    script:
        "../scripts/qc_snapshots.py"


rule group_qc:
    """Aggregate per-subject QC info into a group summary TSV and HTML index.

    Entries without a T1w contribute their NM motion numbers and a status of
    'nm-only', so a partially processed subject is visible in the summary
    rather than missing from it.
    """
    input:
        t1w_jsons=for_each(
            full_entries, datatype="anat", desc="selected", suffix="T1w.json"
        ),
        full_motion=for_each(
            full_entries, datatype="qc", desc="motion", suffix="NM.tsv"
        ),
        nmonly_motion=for_each(
            nmonly_entries, datatype="qc", desc="motion", suffix="NM.tsv"
        ),
        pngs=(
            [
                for_each(full_entries, datatype="qc", desc=qc, suffix="qc.png")
                for qc in ["coreg", "seg", "norm"]
            ]
            + [for_each(nmonly_entries, datatype="qc", desc="nm", suffix="qc.png")]
        ),
    params:
        full=[
            {"subject": e.subject, "session": e.session, "label": e.label}
            for e in full_entries
        ],
        nmonly=[
            {"subject": e.subject, "session": e.session, "label": e.label}
            for e in nmonly_entries
        ],
    output:
        tsv=os.path.join(root, "group", "qc_summary.tsv"),
        html=os.path.join(root, "group", "qc_index.html"),
    script:
        "../scripts/aggregate_qc.py"
