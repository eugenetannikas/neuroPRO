# neuroPRO

Snakebids/Snakemake BIDS app that preprocesses neuromelanin-sensitive MRI
(NM-MRI) so the resulting images are ready for locus coeruleus (LC) and
substantia nigra (SN) segmentation.

It is a scriptable re-implementation of a manual SPM12 workflow
(Realign → ImCalc average → Coregister → Normalise → Smooth),
using ANTs (via ANTsPy) instead of the SPM GUI:

| Step | SPM (manual) | neuroPRO |
|---|---|---|
| 1. Realign & reslice NM images | SPM Realign (register to mean) | two-pass rigid registration to the mean (Mattes MI, B-spline reslice) |
| 2. Average | ImCalc `(i1+i2+i3)/3` | mean of all realigned NM magnitude images |
| 3. Co-registration | SPM Coregister (NMI) | rigid NM avg → T1w (Mattes MI) |
| 5. Normalise (write), 1 mm | deformation field `y_` | affine+SyN T1w → MNI152NLin2009cAsym, composed with the rigid from step 3, single-interpolation resample |
| 6. Smooth, 1 mm FWHM | SPM Smooth | Gaussian smoothing |

## Input data layout

A subject directory per participant, optionally with BIDS sessions:

```
bids_dir/
└── sub-XXX/[ses-YYY/]
    ├── anat/sub-XXX[_ses-YYY][_acq-flash][_run-N]_T1w.nii[.gz] (+ .json)
    └── <NM-GRE images, see below>
```

### NM-GRE discovery

NM images rarely arrive BIDS-named, so three layouts are recognised and tried
in turn (`--nm_layout auto`, the default); the first that matches a subject
wins:

| Layout | Looks like | Where it comes from |
|---|---|---|
| `cfmm` | `sourcedata/sub-XXX/nm-gre/NM-GRE_s<series>_e<echo>[_ph].nii.gz` | CFMM sourcedata export (CABIN, Bright Start) |
| `dcm2niix` | `45_NM_MT-GRE_PAT4(2x2)_PSOFF_e1_20260903123857.nii.gz` | raw dcm2niix output with no renaming (Berkeley 7T) |
| `bids` | `sub-XXX[_ses-YYY]_acq-NM_echo-N_part-mag_MEGRE.nii.gz` under `anat/` | BIDS-native NM |

Pin one with `--nm_layout`, or bypass all three with `--nm_pattern`, a glob
relative to the subject directory (e.g. `--nm_pattern 'nm/*_mag.nii.gz'`).

Discovery is driven by the *images*, not by their sidecars, so:

- both `.nii` and `.nii.gz` are found;
- a missing or malformed `.json` degrades that image's metadata to "unknown"
  rather than hiding the file;
- phase images are excluded by filename (`_ph`, `part-phase`) *and* by
  `ImageType`, so they are dropped even when named like magnitude images;
- scanner-side combined-echo series are excluded by default, since averaging
  them alongside the echoes they derive from double-counts that signal
  (`--include_combecho` to keep them).

Siemens exports each acquisition twice — distortion corrected
(SeriesDescription `NM-GRE`) and uncorrected (`NM-GRE_ND`); `--nm_variant`
picks between them and `--nm_echoes` subsets echoes. **Filters only ever
subset what is present.** If the requested variant or echoes do not exist in
a dataset, the filter is relaxed and the reason reported, rather than
silently leaving the subject with nothing — an empty result is nearly always
a convention mismatch, not a real absence of data.

Every run prints a scan summary saying what was found per subject, which
layout matched, and any filter that had to be relaxed.

### Subjects with no T1w

A subject with NM images but no anatomical is **not** skipped. Realignment,
averaging and motion QC need no T1w, so those run and produce
`desc-avg_NM.nii.gz`, the motion TSV and a native-space QC snapshot. Only
coregistration and normalization are skipped. The group summary
lists these with a status of `nm-only`.

### Sessions

Multi-session datasets are supported end to end; outputs carry the `ses-`
entity. Because rules build static output paths, a dataset must be either
wholly sessioned or wholly unsessioned — one that mixes the two is rejected
with an explanation rather than being collapsed onto colliding paths.

### T1w run selection

Protocol preference ranks a recognised 3D T1 sequence (MPRAGE, MP2RAGE,
BRAVO, FSPGR, SPGR, TFE, ...) above an unrecognised one, and both above a
distortion-uncorrected `_ND` export. Among the preferred protocol,
`--t1w_strategy` picks `last` (default; the highest SeriesNumber, since a
re-scan is usually acquired because earlier runs were motion-corrupted),
`first`, or `sharpest` (highest Laplacian-variance sharpness).

When SeriesNumber is unavailable — an anonymised dataset with no sidecars —
`last`/`first` fall back to the `run-` entity and then to sharpness, so the
choice stays reproducible instead of following filesystem order. The rule
that actually decided is recorded in `SelectionMethod` in the
`desc-selected_T1w.json` sidecar and in the group QC table. Individual
subjects can be overridden with `--t1w_choices choices.tsv` (columns:
`subject`, `filename`).

## Outputs (per subject)

```
sub-XXX/anat/
  sub-XXX_desc-selected_T1w.nii.gz          chosen T1w run (+ provenance json)
  sub-XXX_desc-preproc_T1w.nii.gz           N4 bias-corrected T1w
  sub-XXX_desc-avg_NM.nii.gz                realigned + averaged NM (native)
  sub-XXX_space-T1w_desc-avg_NM.nii.gz      NM average coregistered to T1w
  sub-XXX_desc-brain_mask.nii.gz
  sub-XXX_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
  sub-XXX_space-MNI152NLin2009cAsym_desc-avg_NM.nii.gz
  sub-XXX_space-MNI152NLin2009cAsym_desc-smoothed_NM.nii.gz   <- final image
sub-XXX/xfm/    NM→T1w rigid (.mat), T1w↔MNI composite warps (.h5)
sub-XXX/qc/     coreg / norm snapshot PNGs, NM motion table,
                desc-{coreg,norm}_metrics.json registration quality

(paths gain a ses-YYY entity on sessioned datasets; a subject with no
T1w gets only desc-avg_NM, the motion table and a desc-nm QC snapshot)
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

Useful options: `--nm_layout`, `--nm_pattern`, `--nm_variant`,
`--nm_echoes`, `--include_combecho`, `--t1w_strategy`, `--t1w_choices`,
`--fwhm` (default 1 mm), `--out_res` (default 1 mm),
`--template_dir`/`--template_name`.  All Snakemake flags work too (`-n` for
dry-run, `--keep-going` to let other subjects finish when one fails,
`--slurm` etc.).

## Tests

```bash
pixi run -e dev pytest
```

The suite covers input discovery across every supported naming convention,
T1w selection and its fallbacks, template resolution, and the QC metrics --
on synthetic fixtures, so it runs in a couple of seconds and can be run on
every change.

## Template

`tpl-MNI152NLin2009cAsym` (res-01 T1w from TemplateFlow) is bundled in `neuropro/resources/`.  Note this is an adult
template; for pediatric cohorts pass a pediatric template via
`--template_dir`/`--template_name`.  Entities are matched by glob, so
`res-1` as well as `res-01`, and the extra `cohort-` entity that
cohort-specific TemplateFlow templates carry, both resolve.

## Group QC

`group/qc_summary.tsv` lists, per subject/session, a processing status, the
chosen T1w run and how it was chosen, the NM motion summary, and
normalized mutual information + correlation for both the NM→T1w
coregistration and the T1w→template registration.
`group/qc_index.html` embeds all snapshot PNGs for quick review.

Snapshots do not scale: at n=100 nobody reliably eyeballs 300 PNGs, so a
failed registration reaches the ROI stage looking exactly like a successful
one.  The `qc_flags` column therefore marks subjects whose registration
scores sit far below the rest of the cohort (median − 3×MAD).  The
comparison is deliberately cohort-relative, not an absolute threshold: what
counts as a good NMI depends on sequence, field strength and FOV, so a
number tuned on CABIN would mislabel every 7T scan.  Flags point at
subjects worth looking at; they are not a pass/fail verdict.
