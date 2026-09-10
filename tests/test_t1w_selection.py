"""Tests for T1w run selection."""

from __future__ import annotations

import pytest

from neuropro.t1w_selection import choose, protocol_rank, run_entity


def candidate(filename, description="", series_number=None, sharp=0.0):
    return {
        "filename": filename,
        "series_description": description,
        "series_number": series_number,
        "protocol_rank": protocol_rank(description),
        "sharpness": sharp,
    }


# --------------------------------------------------------------------------
# protocol ranking
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description,expected",
    [
        # the CABIN / Bright Start protocol, and other Siemens spellings
        ("T1 SAG MPRAGE grappa2", 0),
        ("t1_mprage_sag_p2", 0),
        ("MP2RAGE_UNI", 0),
        # other vendors, which the old substring check missed
        ("BRAVO", 0),
        ("3D FSPGR BRAVO", 0),
        ("T1TFE", 0),
        ("SAG 3D SPGR", 0),
        # unrecognised but usable
        ("FLASH", 1),
        ("3D T1", 1),
        ("", 1),
        # distortion-uncorrected export is never preferred
        ("FLASH_ND", 2),
        ("MPRAGE_ND", 2),
    ],
)
def test_protocol_rank(description, expected):
    assert protocol_rank(description) == expected


def test_nd_suffix_beats_protocol_match():
    """An _ND copy of a preferred sequence still ranks last."""
    assert protocol_rank("MPRAGE_ND") == 2
    assert protocol_rank("MPRAGE") == 0


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("sub-01_run-2_T1w.nii.gz", 2),
        ("sub-01_run-11_T1w.nii.gz", 11),
        ("sub-01_T1w.nii.gz", None),
        ("sub-run01_T1w.nii.gz", None),
    ],
)
def test_run_entity(filename, expected):
    assert run_entity(filename) == expected


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_prefers_recognised_protocol_over_unrecognised():
    chosen, _ = choose(
        [
            candidate("a.nii.gz", "FLASH", series_number=9),
            candidate("b.nii.gz", "MPRAGE", series_number=2),
        ],
        "last",
    )
    assert chosen["filename"] == "b.nii.gz"


def test_prefers_corrected_over_nd():
    chosen, _ = choose(
        [
            candidate("nd.nii.gz", "FLASH_ND", series_number=9),
            candidate("ok.nii.gz", "FLASH", series_number=2),
        ],
        "last",
    )
    assert chosen["filename"] == "ok.nii.gz"


def test_last_and_first_by_series_number():
    pool = [
        candidate("a.nii.gz", "MPRAGE", series_number=2),
        candidate("b.nii.gz", "MPRAGE", series_number=7),
    ]
    assert choose(pool, "last")[0]["filename"] == "b.nii.gz"
    assert choose(pool, "first")[0]["filename"] == "a.nii.gz"


def test_sharpest_strategy():
    chosen, method = choose(
        [
            candidate("blurred.nii.gz", "MPRAGE", series_number=7, sharp=0.01),
            candidate("crisp.nii.gz", "MPRAGE", series_number=2, sharp=0.09),
        ],
        "sharpest",
    )
    assert chosen["filename"] == "crisp.nii.gz"
    assert "sharpest" in method


def test_falls_back_to_run_entity_without_series_numbers():
    """A dataset with no sidecars still orders reproducibly."""
    chosen, method = choose(
        [
            candidate("sub-01_run-1_T1w.nii.gz", "MPRAGE"),
            candidate("sub-01_run-2_T1w.nii.gz", "MPRAGE"),
        ],
        "last",
    )
    assert chosen["filename"] == "sub-01_run-2_T1w.nii.gz"
    assert "run-" in method


def test_falls_back_to_sharpness_without_series_or_run():
    chosen, method = choose(
        [
            candidate("x.nii.gz", "MPRAGE", sharp=0.01),
            candidate("y.nii.gz", "MPRAGE", sharp=0.5),
        ],
        "last",
    )
    assert chosen["filename"] == "y.nii.gz"
    assert "sharpness" in method or "sharpest" in method


def test_partial_series_numbers_do_not_sort_by_none():
    """One missing SeriesNumber must not make None the sort key."""
    chosen, method = choose(
        [
            candidate("sub-01_run-1_T1w.nii.gz", "MPRAGE", series_number=3),
            candidate("sub-01_run-2_T1w.nii.gz", "MPRAGE", series_number=None),
        ],
        "last",
    )
    # falls through to the run- entity rather than comparing int with None
    assert chosen["filename"] == "sub-01_run-2_T1w.nii.gz"
    assert "SeriesNumber unavailable" in method


def test_single_candidate_short_circuits():
    chosen, method = choose([candidate("only.nii.gz", "MPRAGE")], "last")
    assert chosen["filename"] == "only.nii.gz"
    assert "only candidate" in method


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown --t1w_strategy"):
        choose([candidate("a.nii.gz", "MPRAGE")], "nonsense")


def test_cabin_case_is_unchanged():
    """The real CABIN/Bright Start shape: MPRAGE runs, pick the later one."""
    chosen, method = choose(
        [
            candidate("sub-004_run-1_T1w.nii.gz", "T1 SAG MPRAGE grappa2", 2),
            candidate("sub-004_run-2_T1w.nii.gz", "T1 SAG MPRAGE grappa2", 14),
        ],
        "last",
    )
    assert chosen["filename"] == "sub-004_run-2_T1w.nii.gz"
    assert method == "protocol preference + last SeriesNumber"
