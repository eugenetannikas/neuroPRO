"""
NM-MRI (neuromelanin-sensitive GRE) processing:

- rigid realignment of all magnitude echo images to their mean (two-pass,
  the scriptable equivalent of SPM Realign: estimate & reslice, register to
  mean) followed by averaging (SPM ImCalc (i1+i2+..+iN)/N)
- non-local-means denoising of the average in native space (in place of
  SPM Smooth; edge-preserving, and applied before any resampling)
- rigid coregistration of the NM average to the subject's T1w
  (SPM Coregister with normalised mutual information)
"""


rule realign_average_nm:
    """Two-pass rigid realignment to the mean, then average all NM images."""
    input:
        nm=get_nm_files,
    output:
        avg=bids(
            root=root,
            datatype="anat",
            desc="avg",
            suffix="NM.nii.gz",
            **subj_wildcards,
        ),
        json=bids(
            root=root,
            datatype="anat",
            desc="avg",
            suffix="NM.json",
            **subj_wildcards,
        ),
        motion=bids(
            root=root,
            datatype="qc",
            desc="motion",
            suffix="NM.tsv",
            **subj_wildcards,
        ),
    threads: 4
    script:
        "../scripts/realign_average_nm.py"


rule denoise_nm:
    """Non-local-means denoising of the realigned NM average (native space)."""
    input:
        nii=rules.realign_average_nm.output.avg,
    output:
        nii=bids(
            root=root,
            datatype="anat",
            desc="denoised",
            suffix="NM.nii.gz",
            **subj_wildcards,
        ),
        json=bids(
            root=root,
            datatype="anat",
            desc="denoised",
            suffix="NM.json",
            **subj_wildcards,
        ),
    threads: 4
    script:
        "../scripts/denoise_nm.py"


rule coregister_nm_to_t1w:
    """Rigid coregistration of the NM average to the preprocessed T1w."""
    input:
        nm=rules.realign_average_nm.output.avg,
        t1w=rules.n4_t1w.output.nii,
    output:
        nm_in_t1w=bids(
            root=root,
            datatype="anat",
            space="T1w",
            desc="avg",
            suffix="NM.nii.gz",
            **subj_wildcards,
        ),
        xfm=bids(
            root=root,
            datatype="xfm",
            from_="NM",
            to="T1w",
            suffix="xfm.mat",
            **subj_wildcards,
        ),
        metrics=bids(
            root=root,
            datatype="qc",
            desc="coreg",
            suffix="metrics.json",
            **subj_wildcards,
        ),
    threads: 4
    script:
        "../scripts/coregister_nm_to_t1w.py"
