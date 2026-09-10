"""Tests for template resource resolution."""

from __future__ import annotations

import pytest

from neuropro.templates import find_template_file


def make_template(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_finds_bundled_style_names(tmp_path):
    """The naming the bundled MNI152NLin2009cAsym resources actually use."""
    d = make_template(
        tmp_path,
        "tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
        "tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz",
        "tpl-MNI152NLin2009cAsym_res-01_label-CSF_probseg.nii.gz",
    )
    assert find_template_file(d, "MNI152NLin2009cAsym", "T1w", res="01").endswith(
        "res-01_T1w.nii.gz"
    )
    assert find_template_file(
        d, "MNI152NLin2009cAsym", "mask", res="01", desc="brain"
    ).endswith("desc-brain_mask.nii.gz")
    assert find_template_file(
        d, "MNI152NLin2009cAsym", "probseg", res="01", label="CSF"
    ).endswith("label-CSF_probseg.nii.gz")


def test_tolerates_unpadded_resolution_entity(tmp_path):
    """'res-1' must be found when 'res-01' was requested."""
    d = make_template(tmp_path, "tpl-MNIPediatricAsym_res-1_T1w.nii.gz")
    assert find_template_file(d, "MNIPediatricAsym", "T1w", res="01").endswith(
        "res-1_T1w.nii.gz"
    )


def test_tolerates_cohort_entity(tmp_path):
    """Paediatric templates carry an extra cohort- entity."""
    d = make_template(tmp_path, "tpl-MNIPediatricAsym_cohort-2_res-1_T1w.nii.gz")
    assert find_template_file(d, "MNIPediatricAsym", "T1w", res="01").endswith(
        "cohort-2_res-1_T1w.nii.gz"
    )


def test_prefers_finest_resolution(tmp_path):
    d = make_template(
        tmp_path,
        "tpl-X_res-01_T1w.nii.gz",
        "tpl-X_res-02_T1w.nii.gz",
    )
    assert find_template_file(d, "X", "T1w", res="01").endswith("res-01_T1w.nii.gz")


def test_does_not_confuse_suffixes(tmp_path):
    """A mask must not be returned when the T1w was asked for."""
    d = make_template(
        tmp_path,
        "tpl-X_res-01_T1w.nii.gz",
        "tpl-X_res-01_desc-brain_mask.nii.gz",
    )
    assert find_template_file(d, "X", "T1w", res="01").endswith("_T1w.nii.gz")


def test_does_not_confuse_tissue_labels(tmp_path):
    d = make_template(
        tmp_path,
        "tpl-X_res-01_label-CSF_probseg.nii.gz",
        "tpl-X_res-01_label-GM_probseg.nii.gz",
        "tpl-X_res-01_label-WM_probseg.nii.gz",
    )
    for tissue in ("CSF", "GM", "WM"):
        got = find_template_file(d, "X", "probseg", res="01", label=tissue)
        assert got.endswith(f"label-{tissue}_probseg.nii.gz")


def test_missing_file_lists_directory_contents(tmp_path):
    """The error must show what is actually there, not just what was wanted."""
    d = make_template(tmp_path, "tpl-X_res-01_T1w.nii.gz")
    with pytest.raises(FileNotFoundError) as excinfo:
        find_template_file(d, "X", "probseg", res="01", label="CSF")
    message = str(excinfo.value)
    assert "tpl-X_res-01_T1w.nii.gz" in message
    assert "--template_dir" in message


def test_empty_directory_reports_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="empty"):
        find_template_file(tmp_path, "X", "T1w", res="01")
