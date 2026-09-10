"""QC snapshot PNGs.

Modes:
- coreg: NM average (T1w space) as background with T1w edge overlay
- seg:   T1w as background with tissue-class boundary overlay
- norm:  template with normalized-T1w edges; normalized NM with template
         edges; brainstem (SN/LC region) zoom on the normalized NM
- nm:    native-space NM average alone, for entries with no T1w
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import sobel


def load(path):
    img = nib.as_closest_canonical(nib.load(path))
    data = np.squeeze(np.asanyarray(img.dataobj)).astype(np.float32)
    return img, data


def norm_range(data, mask=None, plow=1, phigh=99):
    vals = data[mask] if mask is not None else data[data != 0]
    if vals.size == 0:
        return 0, 1
    lo, hi = np.percentile(vals, [plow, phigh])
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def edges(slice2d, pct=92):
    g = np.hypot(sobel(slice2d, 0), sobel(slice2d, 1))
    nz = g[g > 0]
    if nz.size == 0:
        return np.zeros_like(slice2d, dtype=bool)
    return g > np.percentile(nz, pct)


def get_slice(data, axis, index):
    sl = [slice(None)] * 3
    sl[axis] = index
    return np.rot90(data[tuple(sl)])


def slice_aspect(zooms, axis):
    # after rot90, rows/cols swap: aspect = row spacing / col spacing
    other = [z for i, z in enumerate(zooms) if i != axis]
    return other[1] / other[0]


def bbox_indices(data, axis, n, threshold=0):
    """n slice indices spanning the nonzero extent along axis"""
    nz = np.any(data > threshold, axis=tuple(i for i in range(3) if i != axis))
    idx = np.where(nz)[0]
    if idx.size == 0:
        idx = np.arange(data.shape[axis])
    fracs = np.linspace(0.15, 0.85, n)
    return [int(idx[0] + f * (idx[-1] - idx[0])) for f in fracs]


def snapshot_overlay(bg_path, ov_path, out_png, title):
    """rows of axial/coronal/sagittal slices: bg grayscale + overlay edges"""
    bg_img, bg = load(bg_path)
    _, ov = load(ov_path)
    zooms = bg_img.header.get_zooms()[:3]
    lo, hi = norm_range(bg)

    n_cols = 5
    fig, axes = plt.subplots(
        3, n_cols, figsize=(3 * n_cols, 9), facecolor="black"
    )
    for row, axis in enumerate([2, 1, 0]):  # axial, coronal, sagittal
        for col, index in enumerate(bbox_indices(bg, axis, n_cols)):
            ax = axes[row, col]
            bg_sl = get_slice(bg, axis, index)
            ov_sl = get_slice(ov, axis, index)
            ax.imshow(
                bg_sl,
                cmap="gray",
                vmin=lo,
                vmax=hi,
                aspect=slice_aspect(zooms, axis),
                interpolation="nearest",
            )
            edge = np.ma.masked_where(~edges(ov_sl), np.ones_like(ov_sl))
            ax.imshow(
                edge,
                cmap="autumn",
                vmin=0,
                vmax=1,
                alpha=0.8,
                aspect=slice_aspect(zooms, axis),
                interpolation="nearest",
            )
            ax.axis("off")
    fig.suptitle(title, color="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor="black")
    plt.close(fig)


def snapshot_seg(bg_path, dseg_path, out_png, title):
    bg_img, bg = load(bg_path)
    _, dseg = load(dseg_path)
    zooms = bg_img.header.get_zooms()[:3]
    lo, hi = norm_range(bg)
    colors = {1: (0.2, 0.4, 1.0), 2: (1.0, 0.4, 0.2), 3: (0.2, 1.0, 0.4)}

    n_cols = 5
    fig, axes = plt.subplots(
        3, n_cols, figsize=(3 * n_cols, 9), facecolor="black"
    )
    for row, axis in enumerate([2, 1, 0]):
        for col, index in enumerate(bbox_indices(dseg, axis, n_cols)):
            ax = axes[row, col]
            bg_sl = get_slice(bg, axis, index)
            seg_sl = get_slice(dseg, axis, index)
            ax.imshow(
                bg_sl,
                cmap="gray",
                vmin=lo,
                vmax=hi,
                aspect=slice_aspect(zooms, axis),
                interpolation="nearest",
            )
            rgba = np.zeros(seg_sl.shape + (4,))
            for label, color in colors.items():
                # boundary voxels of this class
                m = seg_sl == label
                if not m.any():
                    continue
                interior = (
                    m
                    & np.roll(m, 1, 0)
                    & np.roll(m, -1, 0)
                    & np.roll(m, 1, 1)
                    & np.roll(m, -1, 1)
                )
                boundary = m & ~interior
                rgba[boundary] = color + (0.9,)
            ax.imshow(
                rgba,
                aspect=slice_aspect(zooms, axis),
                interpolation="nearest",
            )
            ax.axis("off")
    fig.suptitle(f"{title}  (CSF=blue GM=orange WM=green)", color="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor="black")
    plt.close(fig)


def snapshot_norm(template_path, t1w_tpl_path, nm_tpl_path, out_png, title):
    tpl_img, tpl = load(template_path)
    _, t1w_tpl = load(t1w_tpl_path)
    nm_img, nm = load(nm_tpl_path)
    zooms = tpl_img.header.get_zooms()[:3]

    fig = plt.figure(figsize=(15, 12), facecolor="black")
    n_cols = 5

    # row 1: template bg + normalized T1w edges (axial)
    lo, hi = norm_range(tpl)
    for col, index in enumerate(bbox_indices(tpl, 2, n_cols)):
        ax = fig.add_subplot(4, n_cols, col + 1)
        bg_sl = get_slice(tpl, 2, index)
        ov_sl = get_slice(t1w_tpl, 2, index)
        ax.imshow(bg_sl, cmap="gray", vmin=lo, vmax=hi,
                  aspect=slice_aspect(zooms, 2), interpolation="nearest")
        edge = np.ma.masked_where(~edges(ov_sl), np.ones_like(ov_sl))
        ax.imshow(edge, cmap="autumn", vmin=0, vmax=1, alpha=0.8,
                  aspect=slice_aspect(zooms, 2), interpolation="nearest")
        ax.axis("off")
        if col == 0:
            ax.set_title("template + T1w edges", color="white", fontsize=9,
                         loc="left")

    # row 2: normalized NM bg + template edges (axial, within NM slab)
    lo_nm, hi_nm = norm_range(nm)
    for col, index in enumerate(bbox_indices(nm, 2, n_cols)):
        ax = fig.add_subplot(4, n_cols, n_cols + col + 1)
        bg_sl = get_slice(nm, 2, index)
        ov_sl = get_slice(tpl, 2, index)
        ax.imshow(bg_sl, cmap="gray", vmin=lo_nm, vmax=hi_nm,
                  aspect=slice_aspect(zooms, 2), interpolation="nearest")
        edge = np.ma.masked_where(~edges(ov_sl), np.ones_like(ov_sl))
        ax.imshow(edge, cmap="autumn", vmin=0, vmax=1, alpha=0.6,
                  aspect=slice_aspect(zooms, 2), interpolation="nearest")
        ax.axis("off")
        if col == 0:
            ax.set_title("normalized NM + template edges", color="white",
                         fontsize=9, loc="left")

    # rows 3-4: brainstem zoom on the normalized NM (SN around z=-14,
    # LC around z=-28 in MNI space)
    inv_aff = np.linalg.inv(nm_img.affine)
    z_coords = [-8, -12, -16, -20, -24, -28, -32, -36, -40, -44]
    for i, z_mm in enumerate(z_coords):
        ax = fig.add_subplot(4, n_cols, 2 * n_cols + i + 1)
        ijk = inv_aff @ np.array([0, -25, z_mm, 1])
        index = int(round(ijk[2]))
        index = np.clip(index, 0, nm.shape[2] - 1)
        # crop around brainstem: x in [-35,35], y in [-55,15] mm
        x0 = int(round((inv_aff @ np.array([-35, 0, 0, 1]))[0]))
        x1 = int(round((inv_aff @ np.array([35, 0, 0, 1]))[0]))
        y0 = int(round((inv_aff @ np.array([0, -55, 0, 1]))[1]))
        y1 = int(round((inv_aff @ np.array([0, 15, 0, 1]))[1]))
        x0, x1 = sorted((np.clip(x0, 0, nm.shape[0] - 1),
                         np.clip(x1, 0, nm.shape[0] - 1)))
        y0, y1 = sorted((np.clip(y0, 0, nm.shape[1] - 1),
                         np.clip(y1, 0, nm.shape[1] - 1)))
        crop = np.rot90(nm[x0:x1, y0:y1, index])
        vals = crop[crop > 0]
        if vals.size:
            lo_c, hi_c = np.percentile(vals, [2, 99.5])
        else:
            lo_c, hi_c = 0, 1
        ax.imshow(crop, cmap="gray", vmin=lo_c, vmax=hi_c,
                  aspect=slice_aspect(zooms, 2), interpolation="nearest")
        ax.set_title(f"z={z_mm}", color="white", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{title}  (bottom: brainstem zoom, SN/LC region)",
                 color="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor="black")
    plt.close(fig)


def snapshot_nm(nm_path, out_png, title):
    """The NM average on its own, for entries with no T1w to overlay.

    Three rows of slices through the acquired slab.  NM-GRE is a thin
    brainstem slab, so the sagittal/coronal views are coarse -- they are there
    to show gross realignment failure and slab coverage, not anatomy.
    """
    nm_img, nm = load(nm_path)
    zooms = nm_img.header.get_zooms()[:3]
    lo, hi = norm_range(nm)

    n_cols = 5
    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9), facecolor="black")
    for row, (axis, name) in enumerate([(2, "axial"), (1, "coronal"), (0, "sagittal")]):
        for col, index in enumerate(bbox_indices(nm, axis, n_cols)):
            ax = axes[row, col]
            ax.imshow(
                get_slice(nm, axis, index),
                cmap="gray",
                vmin=lo,
                vmax=hi,
                aspect=slice_aspect(zooms, axis),
                interpolation="nearest",
            )
            ax.axis("off")
            if col == 0:
                ax.set_title(name, color="white", fontsize=9, loc="left")
    fig.suptitle(title, color="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor="black")
    plt.close(fig)


mode = snakemake.params.mode
label = f"sub-{snakemake.wildcards.subject}"
if hasattr(snakemake.wildcards, "session"):
    label += f"_ses-{snakemake.wildcards.session}"

if mode == "coreg":
    snapshot_overlay(
        snakemake.input.bg,
        snakemake.input.overlay,
        snakemake.output.png,
        f"{label}: NM average in T1w space (T1w edges)",
    )
elif mode == "seg":
    snapshot_seg(
        snakemake.input.bg,
        snakemake.input.overlay,
        snakemake.output.png,
        f"{label}: tissue segmentation",
    )
elif mode == "norm":
    snapshot_norm(
        snakemake.input.template,
        snakemake.input.t1w_tpl,
        snakemake.input.nm_tpl,
        snakemake.output.png,
        f"{label}: normalization",
    )
elif mode == "nm":
    snapshot_nm(
        snakemake.input.nm,
        snakemake.output.png,
        f"{label}: NM average (native space, no T1w available)",
    )
else:
    raise ValueError(f"unknown qc mode: {mode}")
