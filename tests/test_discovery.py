"""Tests for neuroPRO input discovery.

Discovery never opens the image data, so the fixtures here write empty
``.nii.gz`` files plus real JSON sidecars.  That keeps the whole convention
matrix -- CFMM sourcedata, raw dcm2niix, BIDS-native, sessioned, sidecar-less
-- fast enough to run on every change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuropro.discovery import (
    NM_LAYOUTS,
    discover_nm,
    discover_t1w,
    filter_nm,
    find_sessions,
    is_phase,
    missing_participants,
    parse_bids_entities,
    scan_dataset,
    strip_nifti_ext,
    variant_of,
)


def write_nii(path: Path, sidecar: dict | None = None) -> Path:
    """Create an empty NIfTI and, optionally, its JSON sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    if sidecar is not None:
        json_path = path.with_name(strip_nifti_ext(path.name) + ".json")
        json_path.write_text(json.dumps(sidecar))
    return path


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a_b.nii.gz", "a_b"),
        ("a_b.nii", "a_b"),
        ("sub-01_T1w.nii.gz", "sub-01_T1w"),
        ("no_extension", "no_extension"),
        ("NM_MT-GRE_PAT4(2x2).nii.gz", "NM_MT-GRE_PAT4(2x2)"),
    ],
)
def test_strip_nifti_ext(name, expected):
    assert strip_nifti_ext(name) == expected


def test_parse_bids_entities():
    entities = parse_bids_entities("sub-01_ses-A_acq-NM_echo-2_part-mag_MEGRE.nii.gz")
    assert entities["sub"] == "01"
    assert entities["ses"] == "A"
    assert entities["acq"] == "NM"
    assert entities["echo"] == "2"
    assert entities["part"] == "mag"
    assert entities["suffix"] == "MEGRE"


def test_variant_from_sidecar_and_filename(tmp_path):
    nii = tmp_path / "NM-GRE_s1_e1.nii.gz"
    assert variant_of({"SeriesDescription": "NM-GRE_ND"}, nii) == "uncorrected"
    assert variant_of({"SeriesDescription": "NM-GRE"}, nii) == "corrected"
    # no sidecar -> fall back to the filename
    assert variant_of({}, tmp_path / "NM-GRE_ND.nii.gz") == "uncorrected"
    assert variant_of({}, nii) == "corrected"


# --------------------------------------------------------------------------
# CFMM sourcedata layout (CABIN / Bright Start)
# --------------------------------------------------------------------------


@pytest.fixture
def cfmm_subject(tmp_path):
    """One CABIN-style subject: 2 series x 3 echoes, both variants, + phase."""
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    for echo in (1, 2, 3):
        write_nii(
            nm_dir / f"NM-GRE_s6_e{echo}.nii.gz", {"SeriesDescription": "NM-GRE_ND"}
        )
        write_nii(nm_dir / f"NM-GRE_s7_e{echo}.nii.gz", {"SeriesDescription": "NM-GRE"})
        write_nii(
            nm_dir / f"NM-GRE_s8_e{echo}_ph.nii.gz", {"SeriesDescription": "NM-GRE"}
        )
    write_nii(
        tmp_path / "sub-01" / "anat" / "sub-01_run-1_T1w.nii.gz",
        {"SeriesDescription": "T1 SAG MPRAGE grappa2", "SeriesNumber": 2},
    )
    return tmp_path


def test_cfmm_finds_magnitude_only(cfmm_subject):
    images, _ = discover_nm(cfmm_subject / "sub-01")
    assert len(images) == 6
    assert all("_ph" not in Path(i.path).name for i in images)
    assert {i.layout for i in images} == {"cfmm"}


def test_cfmm_orders_by_series_then_echo(cfmm_subject):
    images, _ = discover_nm(cfmm_subject / "sub-01")
    assert [(i.series, i.echo) for i in images] == [
        (6, 1),
        (6, 2),
        (6, 3),
        (7, 1),
        (7, 2),
        (7, 3),
    ]


def test_cfmm_variant_filter(cfmm_subject):
    images, _ = discover_nm(cfmm_subject / "sub-01")
    corrected, _ = filter_nm(images, "corrected", [1, 2, 3])
    assert [i.series for i in corrected] == [7, 7, 7]
    uncorrected, _ = filter_nm(images, "uncorrected", [1, 2, 3])
    assert [i.series for i in uncorrected] == [6, 6, 6]


def test_cfmm_echo_subset(cfmm_subject):
    images, _ = discover_nm(cfmm_subject / "sub-01")
    subset, _ = filter_nm(images, "corrected", [1])
    assert [i.echo for i in subset] == [1]


def test_uncompressed_nii_is_found(tmp_path):
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    write_nii(nm_dir / "NM-GRE_s1_e1.nii", {"SeriesDescription": "NM-GRE"})
    images, _ = discover_nm(tmp_path / "sub-01")
    assert len(images) == 1
    assert images[0].path.endswith(".nii")


def test_missing_sidecar_does_not_hide_image(tmp_path):
    """A sidecar-less image is still discovered, with degraded metadata."""
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    write_nii(nm_dir / "NM-GRE_s1_e1.nii.gz", sidecar=None)
    images, _ = discover_nm(tmp_path / "sub-01")
    assert len(images) == 1
    assert images[0].series == 1 and images[0].echo == 1


def test_malformed_sidecar_is_treated_as_absent(tmp_path):
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    nii = write_nii(nm_dir / "NM-GRE_s1_e1.nii.gz", sidecar=None)
    nii.with_name("NM-GRE_s1_e1.json").write_text("{not valid json")
    images, _ = discover_nm(tmp_path / "sub-01")
    assert len(images) == 1


def test_phase_excluded_via_image_type(tmp_path):
    """Phase images are dropped even when not named '_ph'."""
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    write_nii(
        nm_dir / "NM-GRE_s1_e1.nii.gz",
        {"SeriesDescription": "NM-GRE", "ImageType": ["ORIGINAL", "PRIMARY", "PHASE"]},
    )
    write_nii(nm_dir / "NM-GRE_s1_e2.nii.gz", {"SeriesDescription": "NM-GRE"})
    images, _ = discover_nm(tmp_path / "sub-01")
    assert [i.echo for i in images] == [2]


# --------------------------------------------------------------------------
# raw dcm2niix layout (Berkeley 7T)
# --------------------------------------------------------------------------


@pytest.fixture
def dcm2niix_subject(tmp_path):
    """Berkeley-style raw dcm2niix export, with distractor series."""
    d = tmp_path / "sub-S1" / "nifti"
    stamp = "20260903123857"
    for echo in (1, 2, 3):
        write_nii(
            d / f"45_NM_MT-GRE_PAT4(2x2)_PSOFF_e{echo}_{stamp}.nii.gz",
            {"SeriesDescription": "NM_MT-GRE_PAT4(2x2)_PSOFF"},
        )
        write_nii(
            d / f"44_NM_MT-GRE_PAT4(2x2)_PSOFF_e{echo}_{stamp}_ph.nii.gz",
            {"SeriesDescription": "NM_MT-GRE_PAT4(2x2)_PSOFF"},
        )
    write_nii(
        d / f"46_NM_MT-GRE_PAT4(2x2)_PSOFF_CombEcho_e1_{stamp}.nii.gz",
        {"SeriesDescription": "NM_MT-GRE_PAT4(2x2)_PSOFF_CombEcho"},
    )
    write_nii(d / f"1_localizer_e1_{stamp}.nii.gz", {"SeriesDescription": "localizer"})
    write_nii(d / f"2_t1_mprage_e1_{stamp}.nii.gz", {"SeriesDescription": "t1_mprage"})
    return tmp_path


def test_dcm2niix_finds_nm_only(dcm2niix_subject):
    images, notes = discover_nm(dcm2niix_subject / "sub-S1")
    assert len(images) == 3
    assert {i.layout for i in images} == {"dcm2niix"}
    assert all(i.series == 45 for i in images)
    assert any("dcm2niix" in n for n in notes)


def test_dcm2niix_excludes_localizer_and_anat(dcm2niix_subject):
    images, _ = discover_nm(dcm2niix_subject / "sub-S1")
    names = [Path(i.path).name for i in images]
    assert not any("localizer" in n or "mprage" in n for n in names)


def test_dcm2niix_excludes_phase_after_timestamp(dcm2niix_subject):
    """'_ph' trails the timestamp in dcm2niix names, not the echo."""
    images, _ = discover_nm(dcm2niix_subject / "sub-S1")
    assert not any(Path(i.path).name.endswith("_ph.nii.gz") for i in images)


def test_dcm2niix_combecho_opt_in(dcm2niix_subject):
    without, _ = discover_nm(dcm2niix_subject / "sub-S1")
    with_combecho, _ = discover_nm(
        dcm2niix_subject / "sub-S1", layout="dcm2niix", include_combecho=True
    )
    assert len(with_combecho) == len(without) + 1


# --------------------------------------------------------------------------
# BIDS-native layout
# --------------------------------------------------------------------------


@pytest.fixture
def bids_subject(tmp_path):
    anat = tmp_path / "sub-01" / "anat"
    for echo in (1, 2, 3):
        write_nii(
            anat / f"sub-01_acq-NM_echo-{echo}_part-mag_MEGRE.nii.gz",
            {"SeriesDescription": "NM-GRE", "SeriesNumber": 7},
        )
        write_nii(
            anat / f"sub-01_acq-NM_echo-{echo}_part-phase_MEGRE.nii.gz",
            {"SeriesDescription": "NM-GRE", "SeriesNumber": 7},
        )
    # an unrelated multi-echo GRE that must NOT be picked up
    write_nii(
        anat / "sub-01_acq-QSM_echo-1_part-mag_MEGRE.nii.gz",
        {"SeriesDescription": "QSM"},
    )
    write_nii(anat / "sub-01_T1w.nii.gz", {"SeriesDescription": "MPRAGE"})
    return tmp_path


def test_bids_layout_finds_nm_magnitude(bids_subject):
    images, _ = discover_nm(bids_subject / "sub-01")
    assert len(images) == 3
    assert {i.layout for i in images} == {"bids"}
    assert [i.echo for i in images] == [1, 2, 3]


def test_bids_layout_excludes_phase_and_unrelated_gre(bids_subject):
    images, _ = discover_nm(bids_subject / "sub-01")
    names = [Path(i.path).name for i in images]
    assert not any("part-phase" in n for n in names)
    assert not any("QSM" in n for n in names)


def test_bids_nm_suffix_needs_no_hint(tmp_path):
    """A bare '_NM' suffix is unambiguous on its own."""
    write_nii(tmp_path / "sub-01" / "anat" / "sub-01_echo-1_NM.nii.gz", {})
    images, _ = discover_nm(tmp_path / "sub-01")
    assert len(images) == 1


# --------------------------------------------------------------------------
# filter fallbacks -- the "never silently return nothing" rule
# --------------------------------------------------------------------------


def test_variant_falls_back_when_only_one_exported(tmp_path):
    """Berkeley exports no _ND copy; asking for it must not empty the list."""
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    for echo in (1, 2, 3):
        write_nii(nm_dir / f"NM-GRE_s1_e{echo}.nii.gz", {"SeriesDescription": "NM-GRE"})
    images, _ = discover_nm(tmp_path / "sub-01")
    kept, notes = filter_nm(images, "uncorrected", [1, 2, 3])
    assert len(kept) == 3
    assert any("only one" in n for n in notes)


def test_absent_echoes_fall_back_with_note(tmp_path):
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    write_nii(nm_dir / "NM-GRE_s1_e1.nii.gz", {"SeriesDescription": "NM-GRE"})
    images, _ = discover_nm(tmp_path / "sub-01")
    kept, notes = filter_nm(images, "corrected", [2, 3])
    assert len(kept) == 1
    assert any("not present" in n for n in notes)


def test_unlabelled_echoes_ignore_echo_filter(tmp_path):
    images, notes = discover_nm(tmp_path, pattern="*.nii.gz")
    assert images == []
    write_nii(tmp_path / "nm_scan.nii.gz", {"SeriesDescription": "NM"})
    images, _ = discover_nm(tmp_path, pattern="*.nii.gz")
    kept, notes = filter_nm(images, "corrected", [1, 2, 3])
    assert len(kept) == 1
    assert any("no echo numbers" in n for n in notes)


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


@pytest.fixture
def sessioned_dataset(tmp_path):
    for session in ("20250108", "20260430"):
        base = tmp_path / "sub-01" / f"ses-{session}"
        write_nii(
            base / "anat" / f"sub-01_ses-{session}_T1w.nii.gz",
            {"SeriesDescription": "MPRAGE", "SeriesNumber": 2},
        )
        nm_dir = base / "sourcedata" / "sub-01" / "nm-gre"
        for echo in (1, 2, 3):
            write_nii(
                nm_dir / f"NM-GRE_s5_e{echo}.nii.gz", {"SeriesDescription": "NM-GRE"}
            )
    return tmp_path


def test_find_sessions(sessioned_dataset):
    assert find_sessions(sessioned_dataset / "sub-01") == ["20250108", "20260430"]


def test_scan_yields_one_entry_per_session(sessioned_dataset):
    entries = scan_dataset(sessioned_dataset)
    assert [e.key for e in entries] == [("01", "20250108"), ("01", "20260430")]
    for entry in entries:
        assert len(entry.nm) == 3
        assert len(entry.t1w) == 1
        # each session must get its own images, not the other session's
        assert f"ses-{entry.session}" in entry.nm[0].path
        assert f"ses-{entry.session}" in entry.t1w[0].path


def test_unsessioned_dataset_yields_none_session(cfmm_subject):
    entries = scan_dataset(cfmm_subject)
    assert [e.key for e in entries] == [("01", None)]


# --------------------------------------------------------------------------
# T1w discovery and whole-dataset scanning
# --------------------------------------------------------------------------


def test_discover_t1w_reads_metadata(cfmm_subject):
    candidates = discover_t1w(cfmm_subject / "sub-01")
    assert len(candidates) == 1
    assert candidates[0].series_number == 2
    assert candidates[0].has_sidecar
    assert "MPRAGE" in candidates[0].description


def test_discover_t1w_without_sidecar(tmp_path):
    write_nii(tmp_path / "sub-01" / "anat" / "sub-01_T1w.nii.gz", sidecar=None)
    candidates = discover_t1w(tmp_path / "sub-01")
    assert len(candidates) == 1
    assert candidates[0].series_number is None
    assert not candidates[0].has_sidecar


def test_t1w_ignores_sourcedata_copies(tmp_path):
    """A raw copy under sourcedata/ must not become a candidate run."""
    write_nii(tmp_path / "sub-01" / "sourcedata" / "sub-01" / "sub-01_T1w.nii.gz", {})
    nm = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    write_nii(nm / "NM-GRE_s1_e1.nii.gz", {"SeriesDescription": "NM-GRE"})
    assert discover_t1w(tmp_path / "sub-01") == []


def test_nm_only_subject_is_retained(tmp_path):
    """No T1w must not drop the subject -- native-space output is still valid."""
    nm_dir = tmp_path / "sub-01" / "sourcedata" / "sub-01" / "nm-gre"
    write_nii(nm_dir / "NM-GRE_s1_e1.nii.gz", {"SeriesDescription": "NM-GRE"})
    entries = scan_dataset(tmp_path)
    assert len(entries) == 1
    assert entries[0].nm and entries[0].t1w == []


def test_participant_label_filtering(tmp_path):
    for subject in ("01", "02", "03"):
        nm = tmp_path / f"sub-{subject}" / "sourcedata" / f"sub-{subject}" / "nm-gre"
        write_nii(nm / "NM-GRE_s1_e1.nii.gz", {"SeriesDescription": "NM-GRE"})
    assert [e.subject for e in scan_dataset(tmp_path, participant_label=["02"])] == [
        "02"
    ]
    # a 'sub-' prefix on the label is tolerated
    assert [
        e.subject for e in scan_dataset(tmp_path, participant_label=["sub-02"])
    ] == ["02"]
    assert [
        e.subject for e in scan_dataset(tmp_path, exclude_participant_label=["01"])
    ] == ["02", "03"]


def test_pattern_escape_hatch(tmp_path):
    d = tmp_path / "sub-01" / "weird"
    write_nii(d / "totally_custom_name_001.nii.gz", {"SeriesDescription": "whatever"})
    write_nii(d / "totally_custom_name_002.nii.gz", {"SeriesDescription": "whatever"})
    images, _ = discover_nm(
        tmp_path / "sub-01", pattern="weird/totally_custom_*.nii.gz"
    )
    assert len(images) == 2
    assert {i.layout for i in images} == {"pattern"}


def test_pattern_still_excludes_phase(tmp_path):
    """The escape hatch has no layout regex, so is_phase() is its only guard."""
    d = tmp_path / "sub-01" / "weird"
    write_nii(d / "custom_001.nii.gz", {"SeriesDescription": "NM"})
    write_nii(d / "custom_002_ph.nii.gz", {"SeriesDescription": "NM"})
    write_nii(d / "custom_003_part-phase.nii.gz", {"SeriesDescription": "NM"})
    write_nii(
        d / "custom_004.nii.gz",
        {"SeriesDescription": "NM", "ImageType": ["ORIGINAL", "PHASE"]},
    )
    images, _ = discover_nm(tmp_path / "sub-01", pattern="weird/custom_*.nii.gz")
    assert [Path(i.path).name for i in images] == ["custom_001.nii.gz"]


def test_pattern_reads_echo_from_either_convention(tmp_path):
    d = tmp_path / "sub-01" / "weird"
    write_nii(d / "scan_e2.nii.gz", {"SeriesDescription": "NM"})
    write_nii(d / "scan_echo-3.nii.gz", {"SeriesDescription": "NM"})
    images, _ = discover_nm(tmp_path / "sub-01", pattern="weird/scan_*.nii.gz")
    assert sorted(i.echo for i in images) == [2, 3]


@pytest.mark.parametrize(
    "name,meta,expected",
    [
        ("scan_ph.nii.gz", {}, True),
        ("sub-01_part-phase_MEGRE.nii.gz", {}, True),
        ("sub-01_part-mag_MEGRE.nii.gz", {}, False),
        ("scan.nii.gz", {"ImageType": ["ORIGINAL", "PRIMARY", "PHASE"]}, True),
        ("scan.nii.gz", {"ImageType": ["ORIGINAL", "PRIMARY", "M"]}, False),
        ("scan.nii.gz", {"ImageType": "not-a-list"}, False),
        ("scan.nii.gz", {}, False),
    ],
)
def test_is_phase(tmp_path, name, meta, expected):
    assert is_phase(tmp_path / name, meta) is expected


def test_unknown_layout_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown NM layout"):
        discover_nm(tmp_path, layout="nonsense")


def test_empty_subject_dir_yields_no_images(tmp_path):
    (tmp_path / "sub-01" / "anat").mkdir(parents=True)
    entries = scan_dataset(tmp_path)
    assert len(entries) == 1 and entries[0].nm == []


@pytest.mark.parametrize("layout", NM_LAYOUTS)
def test_every_layout_handles_missing_directory(tmp_path, layout):
    """Discovery on a nonexistent path returns empty, never raises."""
    images, _ = discover_nm(tmp_path / "does-not-exist", layout=layout)
    assert images == []


def test_mixed_dataset_splits_into_full_and_nm_only(tmp_path):
    """Subjects with and without a T1w are both kept, and distinguishable."""
    for subject, with_t1w in (("01", True), ("02", False), ("03", True)):
        base = tmp_path / f"sub-{subject}"
        nm = base / "sourcedata" / f"sub-{subject}" / "nm-gre"
        write_nii(nm / "NM-GRE_s1_e1.nii.gz", {"SeriesDescription": "NM-GRE"})
        if with_t1w:
            write_nii(
                base / "anat" / f"sub-{subject}_T1w.nii.gz",
                {"SeriesDescription": "MPRAGE", "SeriesNumber": 2},
            )

    entries = scan_dataset(tmp_path)
    full = [e for e in entries if e.nm and e.t1w]
    nm_only = [e for e in entries if e.nm and not e.t1w]
    assert [e.subject for e in full] == ["01", "03"]
    assert [e.subject for e in nm_only] == ["02"]


def test_missing_participants_are_reported(tmp_path):
    """A label matching nothing must be reported, not silently dropped."""
    for subject in ("01", "02"):
        nm = tmp_path / f"sub-{subject}" / "sourcedata" / f"sub-{subject}" / "nm-gre"
        write_nii(nm / "NM-GRE_s1_e1.nii.gz", {"SeriesDescription": "NM-GRE"})

    assert missing_participants(tmp_path, ["01", "99"]) == ["99"]
    assert missing_participants(tmp_path, ["sub-99", "sub-02"]) == ["99"]
    assert missing_participants(tmp_path, ["01", "02"]) == []
    assert missing_participants(tmp_path, None) == []
    assert missing_participants(tmp_path, []) == []
