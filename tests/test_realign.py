"""Integration test for the NM realign-and-average step.

Runs the actual workflow script against tiny synthetic images, with a stand-in
for Snakemake's `snakemake` global.  Slower than the pure-logic tests because
it does real ANTs registration, but it is the only thing that checks the step
end to end -- including that a 4D series is split into its volumes rather than
handed to ANTs as something it cannot register.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "neuropro"
    / "workflow"
    / "scripts"
    / "realign_average_nm.py"
)


def blob(shape=(16, 16, 8), shift=0, seed=0):
    """A small image with structure ANTs can actually register."""
    rng = np.random.default_rng(seed)
    data = rng.random(shape).astype(np.float32) * 0.05
    data[4 + shift : 11 + shift, 5:12, 2:6] += 1.0
    return data


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return str(path)


def run_script(tmp_path, inputs):
    out = SimpleNamespace(
        avg=str(tmp_path / "avg.nii.gz"),
        json=str(tmp_path / "avg.json"),
        motion=str(tmp_path / "motion.tsv"),
    )
    fake = SimpleNamespace(
        input=SimpleNamespace(nm=inputs),
        output=out,
        threads=1,
        config={"nm_variant": "corrected", "nm_echoes": [1, 2, 3]},
    )
    import builtins

    builtins.snakemake = fake
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        del builtins.snakemake
    return out


def motion_rows(path):
    lines = Path(path).read_text().strip().splitlines()
    return lines[1:]  # drop the header


@pytest.mark.slow
def test_realigns_and_averages_separate_files(tmp_path):
    inputs = [
        write(tmp_path / f"nm{i}.nii.gz", blob(shift=i, seed=i)) for i in range(3)
    ]
    out = run_script(tmp_path, inputs)

    assert Path(out.avg).exists()
    averaged = np.asanyarray(nib.load(out.avg).dataobj)
    assert averaged.shape == (16, 16, 8)
    assert np.all(averaged >= 0)          # negative B-spline overshoot clipped
    assert len(motion_rows(out.motion)) == 3


@pytest.mark.slow
def test_splits_a_4d_series_into_its_volumes(tmp_path):
    """A 4D export must be realigned volume-by-volume, not rejected."""
    stack = np.stack([blob(shift=i, seed=i) for i in range(3)], axis=-1)
    inputs = [write(tmp_path / "nm4d.nii.gz", stack)]
    out = run_script(tmp_path, inputs)

    assert Path(out.avg).exists()
    assert np.asanyarray(nib.load(out.avg).dataobj).shape == (16, 16, 8)
    # one motion row per volume, even though there was a single input file
    assert len(motion_rows(out.motion)) == 3


@pytest.mark.slow
def test_single_image_passes_through(tmp_path):
    inputs = [write(tmp_path / "nm.nii.gz", blob())]
    out = run_script(tmp_path, inputs)
    assert Path(out.avg).exists()
    assert len(motion_rows(out.motion)) == 1
