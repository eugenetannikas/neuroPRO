"""
Spatial normalization:

- T1w -> template registration (affine + SyN; the scriptable equivalent of the
  deformation field estimated by SPM unified segmentation)
- normalization of the NM average to template space by composing the
  NM->T1w rigid with the T1w->template warp (SPM Normalise: Write)
- Gaussian smoothing (SPM Smooth)
"""


rule register_t1w_to_template:
    """Affine + SyN registration of the preprocessed T1w to the template."""
    input:
        t1w=rules.n4_t1w.output.nii,
        template=template_t1w,
    params:
        out_res=config["out_res"],
    output:
        fwd_xfm=bids(
            root=root,
            datatype="xfm",
            from_="T1w",
            to=template_name,
            suffix="xfm.h5",
            **subj_wildcards,
        ),
        inv_xfm=bids(
            root=root,
            datatype="xfm",
            from_=template_name,
            to="T1w",
            suffix="xfm.h5",
            **subj_wildcards,
        ),
        t1w_tpl=bids(
            root=root,
            datatype="anat",
            space=template_name,
            desc="preproc",
            suffix="T1w.nii.gz",
            **subj_wildcards,
        ),
        metrics=bids(
            root=root,
            datatype="qc",
            desc="norm",
            suffix="metrics.json",
            **subj_wildcards,
        ),
    threads: 8
    script:
        "../scripts/register_t1w_to_template.py"


rule normalize_nm:
    """Resample the NM average into template space (rigid + warp composed)."""
    input:
        nm=rules.realign_average_nm.output.avg,
        nm_to_t1w=rules.coregister_nm_to_t1w.output.xfm,
        t1w_to_tpl=rules.register_t1w_to_template.output.fwd_xfm,
        template=template_t1w,
    params:
        out_res=config["out_res"],
    output:
        nm_tpl=bids(
            root=root,
            datatype="anat",
            space=template_name,
            desc="avg",
            suffix="NM.nii.gz",
            **subj_wildcards,
        ),
    threads: 4
    script:
        "../scripts/normalize_nm.py"


rule smooth_nm:
    """Isotropic Gaussian smoothing of the normalized NM average."""
    input:
        nii=rules.normalize_nm.output.nm_tpl,
    params:
        fwhm=config["fwhm"],
    output:
        nii=bids(
            root=root,
            datatype="anat",
            space=template_name,
            desc="smoothed",
            suffix="NM.nii.gz",
            **subj_wildcards,
        ),
        json=bids(
            root=root,
            datatype="anat",
            space=template_name,
            desc="smoothed",
            suffix="NM.json",
            **subj_wildcards,
        ),
    script:
        "../scripts/smooth_nm.py"
