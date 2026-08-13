# neuroPRO

Snakebids/Snakemake BIDS app that preprocesses neuromelanin-sensitive MRI
(NM-MRI) so the resulting images are ready for locus coeruleus (LC) and
substantia nigra (SN) segmentation.

It is a scriptable re-implementation of a manual SPM12 workflow
(Realign → ImCalc average → Coregister → Segment → Normalise → Smooth),
using ANTs (via ANTsPy) instead of the SPM GUI:

| Step | SPM (manual) | neuroPRO |
|---|---|---|
| 1. Realign & reslice NM images | SPM Realign (register to mean) | two-pass rigid registration to the mean (Mattes MI, B-spline reslice) |
| 2. Average | ImCalc `(i1+i2+i3)/3` | mean of all realigned NM magnitude images |
| 3. Co-registration | SPM Coregister (NMI) | rigid NM avg → T1w (Mattes MI) |
| 4. Segmentation | SPM unified segmentation | Atropos with template tissue priors (CSF/GM/WM) warped to native space |
| 5. Normalise (write), 1 mm | deformation field `y_` | affine+SyN T1w → MNI152NLin2009cAsym, composed with the rigid from step 3, single-interpolation resample |
| 6. Smooth, 1 mm FWHM | SPM Smooth | Gaussian smoothing |

## Input data layout

A BIDS-ish dataset:

```
bids_dir/
└── sub-XXX/
    ├── anat/sub-XXX[_acq-flash]_run-N_T1w.nii.gz (+ .json)
    └── sourcedata/sub-XXX/nm-gre/NM-GRE_s<series>_e<echo>[_ph].nii.gz (+ .json)
```

- T1w anatomicals are indexed with pybids; when a subject has several runs
  the app picks one (see below).
- NM-GRE images are discovered with custom logic (they are not BIDS-named).
  Each acquisition is exported twice — distortion corrected (SeriesDescription
  `NM-GRE`) and uncorrected (`NM-GRE_ND`) — with one magnitude image per echo
  (e1–e3) and optional phase (`_ph`) series, which are ignored.
  By default the distortion-corrected magnitude echoes are used
  (`--nm_variant uncorrected` to switch, `--nm_echoes` to subset echoes).
- Subjects without any NM-GRE magnitude images are skipped with a warning.

### T1w run selection

Protocol preference MPRAGE > FLASH > FLASH_ND; among runs of the preferred
protocol, `--t1w_strategy last` (default) takes the highest SeriesNumber —
re-scans are usually acquired because earlier runs were motion-corrupted.
Every candidate gets a sharpness metric recorded in the
`desc-selected_T1w.json` sidecar, and the group QC table lists the choice per
subject; individual subjects can be overridden with
`--t1w_choices choices.tsv` (columns: `subject`, `filename`).

## Outputs (per subject)

```
sub-XXX/anat/
  sub-XXX_desc-selected_T1w.nii.gz          chosen T1w run (+ provenance json)
  sub-XXX_desc-preproc_T1w.nii.gz           N4 bias-corrected T1w
  sub-XXX_desc-avg_NM.nii.gz                realigned + averaged NM (native)
  sub-XXX_space-T1w_desc-avg_NM.nii.gz      NM average coregistered to T1w
  sub-XXX_label-{CSF,GM,WM}_probseg.nii.gz  tissue probabilities (native T1w)
  sub-XXX_dseg.nii.gz                       1=CSF 2=GM 3=WM
  sub-XXX_desc-brain_mask.nii.gz
  sub-XXX_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
  sub-XXX_space-MNI152NLin2009cAsym_desc-avg_NM.nii.gz
  sub-XXX_space-MNI152NLin2009cAsym_desc-smoothed_NM.nii.gz   <- final image
sub-XXX/xfm/    NM→T1w rigid (.mat), T1w↔MNI composite warps (.h5)
sub-XXX/qc/     coreg / seg / norm snapshot PNGs, NM motion table
```

The `space-MNI152NLin2009cAsym_desc-smoothed_NM` image is the equivalent of
the `sw*` file in the SPM workflow — use it to draw the LC/SN ROIs.

## Running

```bash
cd neuroPRO
pixi install

# everything, 4 subjects at a time
pixi run neuropro /path/to/bids_dir /path/to/derivatives participant --cores 16

# single subject
pixi run neuropro /path/to/bids_dir /path/to/derivatives participant \
    --participant-label 003 --cores 8

# group-level QC summary (after participant level)
pixi run neuropro /path/to/bids_dir /path/to/derivatives group --cores 1
```

Useful options: `--nm_variant`, `--nm_echoes`, `--t1w_strategy`,
`--t1w_choices`, `--fwhm` (default 1 mm), `--out_res` (default 1 mm),
`--template_dir`/`--template_name`.  All Snakemake flags work too (`-n` for
dry-run, `--slurm` etc.).

## Template

`tpl-MNI152NLin2009cAsym` (res-01 T1w, brain mask, CSF/GM/WM probsegs from
TemplateFlow) is bundled in `neuropro/resources/`.  Note this is an adult
template; for pediatric cohorts consider a pediatric template via
`--template_dir`/`--template_name` (files must follow the same naming).

## Group QC

`group/qc_summary.tsv` lists, per subject, the chosen T1w run and NM motion
summary; `group/qc_index.html` embeds all snapshot PNGs for quick review.
