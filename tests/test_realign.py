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


def run_script(tmp_path, inputs, runs=None):
    """runs: the echo grouping the rule would pass; default = one run per file."""
    if runs is None:
        runs = [[p] for p in inputs]
    out = SimpleNamespace(
        avg=str(tmp_path / "avg.nii.gz"),
        json=str(tmp_path / "avg.json"),
        motion=str(tmp_path / "motion.tsv"),
    )
    fake = SimpleNamespace(
        input=SimpleNamespace(nm=inputs),
        params=SimpleNamespace(runs=runs),
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
    # untouched: no registration, no reslicing
    assert np.allclose(np.asanyarray(nib.load(out.avg).dataobj), blob())


@pytest.mark.slow
def test_echoes_of_one_run_are_averaged_not_realigned(tmp_path):
    """Echoes share an excitation: combine by mean, never register them."""
    e1 = blob(seed=1)
    e2 = blob(seed=2) * 0.7          # later echo: same anatomy, less signal
    inputs = [write(tmp_path / "s5_e1.nii.gz", e1), write(tmp_path / "s5_e2.nii.gz", e2)]
    out = run_script(tmp_path, inputs, runs=[inputs])   # one run, two echoes

    averaged = np.asanyarray(nib.load(out.avg).dataobj)
    assert np.allclose(averaged, (e1 + e2) / 2, atol=1e-5)   # exact voxelwise mean
    assert len(motion_rows(out.motion)) == 1                # one run, one row
    assert "\t0.0000\t0.0000" in motion_rows(out.motion)[0]  # nothing moved


@pytest.mark.slow
def test_runs_are_realigned_but_their_echoes_are_not(tmp_path):
    """Three runs x two echoes: three motion rows, not six."""
    inputs, runs = [], []
    for r in range(3):
        paths = [
            write(tmp_path / f"s{r}_e{e}.nii.gz", blob(shift=r, seed=10 * r + e))
            for e in (1, 2)
        ]
        inputs += paths
        runs.append(paths)
    out = run_script(tmp_path, inputs, runs=runs)
    assert len(motion_rows(out.motion)) == 3
    import json
    meta = json.loads(Path(out.json).read_text())
    assert meta["EchoesPerRun"] == [2, 2, 2]
    assert meta["RunsRealigned"] == 3
