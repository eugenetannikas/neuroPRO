"""Two-pass rigid realignment of all NM magnitude images to their mean,
followed by averaging.

Pass 1 registers every image to the first image and forms a mean; pass 2
registers every original image to that mean and averages again (the
scriptable equivalent of SPM Realign 'register to mean' + ImCalc average).
Registration uses rigid transforms with a Mattes mutual-information metric,
so echoes with different contrast can be aligned to each other.  Images are
resliced with B-spline interpolation.

If only one image is present, it is simply resampled/copied through.
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


nm_paths = []
images = []
for _path in list(snakemake.input.nm):
    for _label, _image in load_volumes(_path):
        nm_paths.append(_label)
        images.append(_image)


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
    avg = images[0]
    motion = [(nm_paths[0], 0.0, 0.0)]
else:
    # pass 1: register everything to the first image
    resliced, _ = register_all(images[0], images, nm_paths)
    mean1 = average(resliced)
    # pass 2: register the original images to the pass-1 mean
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
            "Description": "Average of rigidly realigned NM-GRE magnitude images",
            "SourceFiles": list(snakemake.input.nm),
            "VolumesRealigned": len(images),
            "NMVariant": snakemake.config["nm_variant"],
            "Echoes": snakemake.config["nm_echoes"],
            "RealignmentPasses": 1 if len(images) == 1 else 2,
        },
        f,
        indent=2,
    )

with open(snakemake.output.motion, "w") as f:
    f.write("source_file\ttranslation_mm\trotation_deg\n")
    for path, trans_mm, rot_deg in motion:
        f.write(f"{os.path.basename(path)}\t{trans_mm:.4f}\t{rot_deg:.4f}\n")
