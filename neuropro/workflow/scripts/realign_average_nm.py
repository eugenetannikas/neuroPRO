"""Combine echoes within each run, realign the runs, and average.

An NM acquisition is read out at several echo times from one excitation, so
its echoes are already in register: they are combined by a plain mean, never
registered to each other (registering an image to a differently-weighted copy
of itself is at best pointless).  Separate runs -- repeated acquisitions --
do move relative to each other, so those are rigidly realigned: pass 1
registers every run to the first and forms a mean, pass 2 registers every
original run to that mean and averages again (the scriptable equivalent of
SPM Realign 'register to mean' + ImCalc average).  Registration uses rigid
transforms with a Mattes mutual-information metric; runs are resliced with
B-spline interpolation.

A single run passes straight through: no registration, no reslicing.
"""

import json
import math
import os

os.environ.setdefault(
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(snakemake.threads)
)

import ants
import numpy as np


def load_volumes(path):
    """The 3D volumes in a NIfTI, as (label, image) pairs.

    Some sites export the repeats of an NM acquisition as one 4D series
    rather than as separate files.  Splitting them here means those repeats
    are realigned to each other exactly like separately-exported ones,
    instead of reaching ANTs as a 4D image it cannot register.
    """
    img = ants.image_read(path).clone("float")
    if img.dimension < 4:
        return [(path, img)]
    volumes = ants.ndimage_to_list(img)
    return [
        (f"{path}[{index}]", volume.clone("float"))
        for index, volume in enumerate(volumes)
    ]


def combine_echoes(label, paths):
    """The runs represented by one group of echo images.

    Normally a group is the echoes of one 3D acquisition and yields one run:
    their voxelwise mean, no registration.  A 4D file holds repeats along its
    fourth axis, so a group of 4D echoes yields one run per repeat, combining
    echo i of every repeat.
    """
    per_path = [load_volumes(p) for p in paths]
    n_rep = {len(v) for v in per_path}
    if len(n_rep) != 1:
        raise ValueError(
            f"run {label}: echo files hold different numbers of volumes "
            f"({sorted(n_rep)}); cannot pair them"
        )
    runs = []
    for rep in range(n_rep.pop()):
        vols = [v[rep][1] for v in per_path]
        shapes = {tuple(v.shape) for v in vols}
        if len(shapes) != 1:
            raise ValueError(
                f"run {label}: echoes have different grids {sorted(shapes)}; "
                "echoes of one acquisition must share a grid"
            )
        if len(vols) == 1:
            combined = vols[0]
        else:
            mean = np.zeros(vols[0].shape, dtype=np.float64)
            for v in vols:
                mean += v.numpy()
            mean /= len(vols)
            combined = vols[0].new_image_like(mean.astype(np.float32))
        suffix = f"[{rep}]" if len(per_path[0]) > 1 else ""
        runs.append((f"{label}{suffix}", combined, len(vols)))
    return runs


run_groups = [list(g) for g in snakemake.params.runs]
nm_paths = []
images = []
echoes_per_run = []
for _i, _paths in enumerate(run_groups):
    _label = os.path.basename(_paths[0]) if len(_paths) == 1 else f"run-{_i + 1}"
    for _rlabel, _image, _n_echo in combine_echoes(_label, _paths):
        nm_paths.append(_rlabel)
        images.append(_image)
        echoes_per_run.append(_n_echo)


def register_all(target, images, paths):
    """Rigidly register each image to target; return resliced images and
    per-image motion parameters."""
    resliced = []
    motion = []
    for img, path in zip(images, paths):
        if img is target:
            resliced.append(img)
            motion.append((path, 0.0, 0.0))
            continue
        reg = ants.registration(
            fixed=target,
            moving=img,
            type_of_transform="Rigid",
            aff_metric="mattes",
        )
        out = ants.apply_transforms(
            fixed=target,
            moving=img,
            transformlist=reg["fwdtransforms"],
            interpolator="bSpline",
        )
        resliced.append(out)

        tx = ants.read_transform(reg["fwdtransforms"][0])
        params = np.asarray(tx.parameters)
        matrix = params[:9].reshape(3, 3)
        translation = params[9:12]
        trans_mm = float(np.linalg.norm(translation))
        cos_angle = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = float(math.degrees(math.acos(cos_angle)))
        motion.append((path, trans_mm, rot_deg))
    return resliced, motion


def average(images):
    mean = images[0].numpy().astype(np.float64)
    for img in images[1:]:
        mean += img.numpy()
    mean /= len(images)
    return images[0].new_image_like(mean.astype(np.float32))


if len(images) == 1:
    # one run: nothing to realign, and no reslicing to blur it
    avg = images[0]
    motion = [(nm_paths[0], 0.0, 0.0)]
else:
    # pass 1: register every run to the first and form a mean
    resliced, _ = register_all(images[0], images, nm_paths)
    mean1 = average(resliced)
    # pass 2: register the original runs to the pass-1 mean
    resliced, motion = register_all(mean1, images, nm_paths)
    avg = average(resliced)

# negative overshoot from B-spline interpolation is not meaningful here
arr = avg.numpy()
arr[arr < 0] = 0
avg = avg.new_image_like(arr)

ants.image_write(avg, snakemake.output.avg)

with open(snakemake.output.json, "w") as f:
    json.dump(
        {
            "Description": (
                "NM-GRE magnitude image: echoes of each run combined by "
                "plain mean, runs rigidly realigned and averaged"
            ),
            "SourceFiles": list(snakemake.input.nm),
            "Runs": [list(g) for g in run_groups],
            "RunsRealigned": len(images),
            "EchoesPerRun": echoes_per_run,
            "EchoCombination": "voxelwise mean, no registration",
            "NMVariant": snakemake.config["nm_variant"],
            "Echoes": snakemake.config["nm_echoes"],
            "RealignmentPasses": 1 if len(images) == 1 else 2,
        },
        f,
        indent=2,
    )

with open(snakemake.output.motion, "w") as f:
    f.write("run\ttranslation_mm\trotation_deg\n")
    for path, trans_mm, rot_deg in motion:
        f.write(f"{os.path.basename(path)}\t{trans_mm:.4f}\t{rot_deg:.4f}\n")
