"""Input discovery for neuroPRO.

Locates NM-GRE magnitude images and T1w anatomicals across the naming
conventions we actually encounter, rather than the single convention the
first CABIN export happened to use.

Three layouts are recognised (tried in order; the first that yields images
for a subject wins, unless ``--nm_layout`` pins one):

``cfmm``
    ``sub-X/**/nm-gre/NM-GRE_s<series>_e<echo>[_ph].nii[.gz]`` -- the CFMM
    sourcedata export used by the CABIN and Bright Start datasets.

``dcm2niix``
    ``<series>_<description>_e<echo>[_<timestamp>][_ph].nii[.gz]`` -- plain
    dcm2niix output with no BIDS renaming, e.g. the Berkeley 7T MT-GRE data
    (``45_NM_MT-GRE_PAT4(2x2)_PSOFF_e1_20260903123857.nii.gz``).

``bids``
    ``sub-X[_ses-Y][_acq-Z][_echo-N][_part-mag]_<suffix>.nii[.gz]`` under
    ``anat/``, with an NM hint in the ``acq``/``desc`` entity or a ``NM``
    suffix.

Design rules that keep this robust on unfamiliar data:

- Discovery is driven by the *images*, not by their sidecars.  A missing
  ``.json`` degrades metadata to "unknown"; it never hides a file.
- Both ``.nii`` and ``.nii.gz`` are found.
- Filters (variant, echo) only ever subset what is present.  If a filter
  would leave a subject with nothing, it is reported and relaxed rather
  than silently dropping the subject -- an empty result is nearly always a
  convention mismatch, not a real absence of data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NIFTI_SUFFIXES = (".nii.gz", ".nii")

#: Layout names in the order ``auto`` tries them.
NM_LAYOUTS = ("cfmm", "dcm2niix", "bids")

#: BIDS suffixes that a neuromelanin GRE acquisition is plausibly stored as.
NM_BIDS_SUFFIXES = ("NM", "MEGRE", "GRE", "T2starw")

#: Matches an NM-ish series description / acquisition label.  Deliberately
#: loose: "NM-GRE", "NM_MT-GRE_PAT4(2x2)_PSOFF", "acq-NM", "neuromelanin".
NM_HINT_RE = re.compile(r"(?:^|[^A-Za-z])(?:NM|neuromelanin)(?:[^A-Za-z]|$)", re.I)

_CFMM_RE = re.compile(r"^NM-GRE_s(?P<series>\d+)_e(?P<echo>\d+)(?P<ph>_ph)?$")
_DCM2NIIX_RE = re.compile(
    r"^(?P<series>\d+)_(?P<desc>.+?)"
    r"(?:_e(?P<echo>\d+))?"
    r"(?:_(?P<stamp>\d{8,}))?"
    r"(?P<ph>_ph)?$"
)


def strip_nifti_ext(name: str) -> str:
    """Filename without its NIfTI extension ('a_b.nii.gz' -> 'a_b')."""
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def sidecar_for(nii: Path) -> Path | None:
    """The JSON sidecar beside a NIfTI, or None when there isn't one."""
    candidate = nii.with_name(strip_nifti_ext(nii.name) + ".json")
    return candidate if candidate.exists() else None


def read_sidecar(nii: Path) -> dict:
    """Sidecar contents, or {} when missing or unparseable.

    Unparseable sidecars are treated as absent: a malformed JSON file is a
    metadata problem, and refusing to process the image because of it would
    be worse than proceeding with unknown metadata.
    """
    path = sidecar_for(nii)
    if path is None:
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def iter_niftis(directory: Path) -> list[Path]:
    """All NIfTI files directly in a directory, sorted, .nii.gz before .nii."""
    if not directory.is_dir():
        return []
    found = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.name.endswith(NIFTI_SUFFIXES)
    ]
    return sorted(found, key=lambda p: p.name)


def parse_bids_entities(name: str) -> dict[str, str]:
    """Parse 'sub-01_ses-A_acq-NM_echo-1_part-mag_MEGRE' into a dict.

    The trailing suffix is returned under the key ``suffix``.
    """
    stem = strip_nifti_ext(name)
    parts = stem.split("_")
    entities: dict[str, str] = {}
    for part in parts:
        key, sep, value = part.partition("-")
        if sep:
            entities.setdefault(key, value)
        else:
            entities["suffix"] = part
    return entities


def is_phase(nii: Path, meta: dict) -> bool:
    """True when an image holds phase rather than magnitude data.

    Checked from the filename first (``_ph`` from dcm2niix, ``part-phase``
    from BIDS) and then from ``ImageType``, so phase images are excluded even
    when they are named like magnitude ones.
    """
    stem = strip_nifti_ext(nii.name)
    if stem.endswith("_ph") or "part-phase" in stem:
        return True
    image_type = meta.get("ImageType") or []
    if isinstance(image_type, (list, tuple)):
        return any(str(t).upper() == "PHASE" for t in image_type)
    return False


def variant_of(meta: dict, nii: Path) -> str:
    """Distortion-correction variant: 'corrected' or 'uncorrected'.

    Siemens exports each NM acquisition twice, the uncorrected copy carrying
    an ``_ND`` (no distortion correction) suffix on its SeriesDescription.
    Falls back to the filename when there is no sidecar.
    """
    desc = str(meta.get("SeriesDescription", ""))
    haystack = desc or strip_nifti_ext(nii.name)
    return "uncorrected" if haystack.upper().endswith("_ND") else "corrected"


def looks_like_nm(*texts: str) -> bool:
    """True when any of the given strings names a neuromelanin acquisition."""
    return any(NM_HINT_RE.search(t or "") for t in texts)


@dataclass(frozen=True)
class NMImage:
    """One NM magnitude image, with whatever metadata we could recover."""

    path: str
    series: int | None = None
    echo: int | None = None
    variant: str = "corrected"
    layout: str = "unknown"
    description: str = ""

    @property
    def sort_key(self) -> tuple:
        # None sorts last so unlabelled images trail labelled ones instead of
        # raising on a mixed comparison
        return (
            self.series is None,
            self.series or 0,
            self.echo is None,
            self.echo or 0,
            self.path,
        )


@dataclass(frozen=True)
class T1wImage:
    """One candidate T1w anatomical run."""

    path: str
    description: str = ""
    series_number: int | None = None
    has_sidecar: bool = False


@dataclass
class SubjectInputs:
    """Everything discovery found for one subject (or subject/session)."""

    subject: str
    session: str | None = None
    nm: list[NMImage] = field(default_factory=list)
    t1w: list[T1wImage] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str | None]:
        return (self.subject, self.session)

    @property
    def label(self) -> str:
        return f"sub-{self.subject}" + (f"_ses-{self.session}" if self.session else "")


# --------------------------------------------------------------------------
# per-layout NM collectors
# --------------------------------------------------------------------------


def _collect_cfmm(search_root: Path) -> list[NMImage]:
    """CFMM sourcedata export: any nm-gre/ directory below the search root."""
    images = []
    nm_dirs = sorted({p for p in search_root.glob("**/nm-gre") if p.is_dir()})
    for nm_dir in nm_dirs:
        for nii in iter_niftis(nm_dir):
            match = _CFMM_RE.match(strip_nifti_ext(nii.name))
            if match is None:
                continue
            meta = read_sidecar(nii)
            if match.group("ph") or is_phase(nii, meta):
                continue
            images.append(
                NMImage(
                    path=str(nii),
                    series=int(match.group("series")),
                    echo=int(match.group("echo")),
                    variant=variant_of(meta, nii),
                    layout="cfmm",
                    description=str(meta.get("SeriesDescription", "")),
                )
            )
    return images


def _collect_dcm2niix(search_root: Path, include_combecho: bool) -> list[NMImage]:
    """Raw dcm2niix output, e.g. '45_NM_MT-GRE_PSOFF_e1_20260903123857.nii.gz'.

    Only series whose description looks neuromelanin-ish are kept, so this
    does not sweep up localisers or other series sitting in the same folder.
    """
    images = []
    candidate_dirs = {search_root}
    candidate_dirs.update(p for p in search_root.glob("**/") if p.is_dir())
    for directory in sorted(candidate_dirs):
        for nii in iter_niftis(directory):
            match = _DCM2NIIX_RE.match(strip_nifti_ext(nii.name))
            if match is None:
                continue
            desc = match.group("desc")
            meta = read_sidecar(nii)
            series_desc = str(meta.get("SeriesDescription", "")) or desc
            if not looks_like_nm(desc, series_desc):
                continue
            if match.group("ph") or is_phase(nii, meta):
                continue
            # CombEcho series are a scanner-side combination of the individual
            # echoes; averaging them alongside those echoes would double-count
            if not include_combecho and "combecho" in desc.lower():
                continue
            echo = match.group("echo")
            images.append(
                NMImage(
                    path=str(nii),
                    series=int(match.group("series")),
                    echo=int(echo) if echo else None,
                    variant=variant_of(meta, nii),
                    layout="dcm2niix",
                    description=series_desc,
                )
            )
    return images


def _collect_bids(search_root: Path) -> list[NMImage]:
    """BIDS-named NM images under anat/ (or anywhere below the subject)."""
    images = []
    candidate_dirs = {search_root}
    candidate_dirs.update(p for p in search_root.glob("**/") if p.is_dir())
    for directory in sorted(candidate_dirs):
        for nii in iter_niftis(directory):
            entities = parse_bids_entities(nii.name)
            suffix = entities.get("suffix", "")
            if suffix not in NM_BIDS_SUFFIXES:
                continue
            acq = entities.get("acq", "")
            desc = entities.get("desc", "")
            meta = read_sidecar(nii)
            series_desc = str(meta.get("SeriesDescription", ""))
            # a bare MEGRE/GRE/T2starw is only ours if something says NM
            if suffix != "NM" and not looks_like_nm(acq, desc, series_desc):
                continue
            if entities.get("part") == "phase" or is_phase(nii, meta):
                continue
            echo = entities.get("echo")
            series = meta.get("SeriesNumber")
            images.append(
                NMImage(
                    path=str(nii),
                    series=int(series) if isinstance(series, int) else None,
                    echo=int(echo) if echo and echo.isdigit() else None,
                    variant=variant_of(meta, nii),
                    layout="bids",
                    description=series_desc or acq,
                )
            )
    return images


def collect_nm_by_layout(
    search_root: Path, layout: str, include_combecho: bool = False
) -> list[NMImage]:
    """Run a single named layout collector."""
    if layout == "cfmm":
        return _collect_cfmm(search_root)
    if layout == "dcm2niix":
        return _collect_dcm2niix(search_root, include_combecho)
    if layout == "bids":
        return _collect_bids(search_root)
    raise ValueError(
        f"Unknown NM layout '{layout}'; expected one of "
        f"{', '.join(NM_LAYOUTS)} or 'auto'"
    )


def collect_nm_by_pattern(search_root: Path, pattern: str) -> list[NMImage]:
    """Escape hatch: a user-supplied glob, relative to the subject directory.

    Metadata is taken from sidecars where available; filenames are still
    inspected for echo/phase markers so the standard filters keep working.
    """
    images = []
    for nii in sorted(search_root.glob(pattern)):
        if not nii.is_file() or not nii.name.endswith(NIFTI_SUFFIXES):
            continue
        meta = read_sidecar(nii)
        if is_phase(nii, meta):
            continue
        stem = strip_nifti_ext(nii.name)
        echo_match = re.search(r"(?:_e|_echo-)(\d+)", stem)
        series = meta.get("SeriesNumber")
        images.append(
            NMImage(
                path=str(nii),
                series=int(series) if isinstance(series, int) else None,
                echo=int(echo_match.group(1)) if echo_match else None,
                variant=variant_of(meta, nii),
                layout="pattern",
                description=str(meta.get("SeriesDescription", "")),
            )
        )
    return images


def discover_nm(
    search_root: Path,
    layout: str = "auto",
    pattern: str | None = None,
    include_combecho: bool = False,
) -> tuple[list[NMImage], list[str]]:
    """Find NM magnitude images below ``search_root``.

    Returns the images (unfiltered) and any notes worth surfacing to the user.
    """
    notes: list[str] = []
    if pattern:
        images = collect_nm_by_pattern(search_root, pattern)
        if not images:
            notes.append(f"--nm_pattern '{pattern}' matched no NIfTI files")
        return sorted(images, key=lambda i: i.sort_key), notes

    layouts = NM_LAYOUTS if layout == "auto" else (layout,)
    for name in layouts:
        images = collect_nm_by_layout(search_root, name, include_combecho)
        if images:
            if layout == "auto" and name != NM_LAYOUTS[0]:
                notes.append(f"NM images matched the '{name}' layout")
            return sorted(images, key=lambda i: i.sort_key), notes
    return [], notes


def filter_nm(
    images: list[NMImage],
    variant: str,
    echoes: list[int] | None,
) -> tuple[list[NMImage], list[str]]:
    """Apply the --nm_variant / --nm_echoes filters.

    A filter that would empty the list is relaxed and reported instead: on an
    unfamiliar dataset an empty result almost always means the requested
    facet does not exist here, and dropping the subject would hide that.
    """
    notes: list[str] = []
    if not images:
        return images, notes

    variants_present = {img.variant for img in images}
    if variant in variants_present:
        images = [img for img in images if img.variant == variant]
    elif len(variants_present) == 1:
        # only one variant was exported; using it is what the user meant
        only = next(iter(variants_present))
        notes.append(
            f"no '{variant}' NM images here; using the '{only}' variant "
            "(this dataset exports only one)"
        )

    if echoes:
        wanted = set(echoes)
        matched = [img for img in images if img.echo in wanted]
        unlabelled = [img for img in images if img.echo is None]
        if matched:
            # keep unlabelled images only when nothing carried an echo number
            images = matched
        elif unlabelled:
            images = unlabelled
            notes.append(
                f"no echo numbers on these NM images; ignoring --nm_echoes "
                f"{sorted(wanted)} and using all {len(unlabelled)} of them"
            )
        else:
            present = sorted({img.echo for img in images if img.echo is not None})
            notes.append(
                f"requested echoes {sorted(wanted)} not present "
                f"(found {present}); using all of them"
            )

    return images, notes


# --------------------------------------------------------------------------
# T1w discovery
# --------------------------------------------------------------------------


def discover_t1w(search_root: Path) -> list[T1wImage]:
    """Find T1w candidate runs for a subject/session.

    Looks in ``anat/`` where BIDS puts them, and falls back to a recursive
    search so flatter exports still work.  ``sourcedata/`` is excluded so raw
    copies are not mistaken for the converted anatomicals.
    """
    directories = [search_root / "anat"]
    if not directories[0].is_dir():
        directories = [
            p
            for p in sorted(search_root.glob("**/"))
            if p.is_dir() and "sourcedata" not in p.relative_to(search_root).parts
        ]

    found = []
    for directory in directories:
        for nii in iter_niftis(directory):
            entities = parse_bids_entities(nii.name)
            if entities.get("suffix") != "T1w":
                continue
            meta = read_sidecar(nii)
            series = meta.get("SeriesNumber")
            found.append(
                T1wImage(
                    path=str(nii),
                    description=str(meta.get("SeriesDescription", "")),
                    series_number=int(series) if isinstance(series, int) else None,
                    has_sidecar=sidecar_for(nii) is not None,
                )
            )
    return sorted(found, key=lambda t: t.path)


# --------------------------------------------------------------------------
# top-level dataset scan
# --------------------------------------------------------------------------


def find_sessions(subject_dir: Path) -> list[str]:
    """Session labels under a subject directory, empty when unsessioned."""
    return sorted(
        p.name.removeprefix("ses-") for p in subject_dir.glob("ses-*") if p.is_dir()
    )


def missing_participants(
    bids_dir: Path, participant_label: list[str] | None
) -> list[str]:
    """Requested subject labels that have no directory in the dataset.

    A label that matches nothing is silently dropped by the filter, which is
    the wrong default when the list came from a spreadsheet or another
    cohort: half of it can be missing without anything saying so.
    """
    if not participant_label:
        return []
    present = {
        p.name.removeprefix("sub-") for p in Path(bids_dir).glob("sub-*") if p.is_dir()
    }
    requested = {s.removeprefix("sub-") for s in participant_label}
    return sorted(requested - present)


def scan_dataset(
    bids_dir: Path,
    participant_label: list[str] | None = None,
    exclude_participant_label: list[str] | None = None,
    nm_layout: str = "auto",
    nm_pattern: str | None = None,
    nm_variant: str = "corrected",
    nm_echoes: list[int] | None = None,
    include_combecho: bool = False,
) -> list[SubjectInputs]:
    """Scan a dataset and return one SubjectInputs per subject/session.

    Entries with no NM images at all are omitted; entries with NM but no T1w
    are kept, so they can still be processed to native-space outputs.
    """
    bids_dir = Path(bids_dir)
    subjects = sorted(
        p.name.removeprefix("sub-") for p in bids_dir.glob("sub-*") if p.is_dir()
    )
    if participant_label:
        keep = {s.removeprefix("sub-") for s in participant_label}
        subjects = [s for s in subjects if s in keep]
    if exclude_participant_label:
        drop = {s.removeprefix("sub-") for s in exclude_participant_label}
        subjects = [s for s in subjects if s not in drop]

    entries: list[SubjectInputs] = []
    for subject in subjects:
        subject_dir = bids_dir / f"sub-{subject}"
        sessions: list[str | None] = list(find_sessions(subject_dir)) or [None]
        for session in sessions:
            # NM lives under sourcedata/ in the CFMM layout, which sits at the
            # subject level even when the anatomicals are sessioned, so NM is
            # searched from the subject root and filtered by session below
            search_root = subject_dir / f"ses-{session}" if session else subject_dir
            nm_root = (
                search_root if session and _has_content(search_root) else subject_dir
            )
            images, notes = discover_nm(
                nm_root, nm_layout, nm_pattern, include_combecho
            )
            if session:
                scoped = [i for i in images if f"ses-{session}" in i.path]
                if scoped:
                    images = scoped
            images, filter_notes = filter_nm(images, nm_variant, nm_echoes)
            entry = SubjectInputs(
                subject=subject,
                session=session,
                nm=images,
                t1w=discover_t1w(search_root),
                notes=notes + filter_notes,
            )
            entries.append(entry)
    return entries


def _has_content(directory: Path) -> bool:
    """True when a directory exists and holds anything at all."""
    return directory.is_dir() and any(directory.iterdir())


def summarize(entries: list[SubjectInputs]) -> str:
    """Human-readable planning summary, printed once at workflow start."""
    usable = [e for e in entries if e.nm]
    with_t1w = [e for e in usable if e.t1w]
    lines = [
        f"neuroPRO: {len(usable)} subject/session entries with NM images "
        f"({len(with_t1w)} with a T1w, "
        f"{len(usable) - len(with_t1w)} NM-only)"
    ]
    for entry in entries:
        if not entry.nm:
            lines.append(f"  {entry.label}: SKIPPED -- no NM images found")
            continue
        layouts = {i.layout for i in entry.nm}
        detail = (
            f"  {entry.label}: {len(entry.nm)} NM image(s) "
            f"[{'/'.join(sorted(layouts))}], {len(entry.t1w)} T1w candidate(s)"
        )
        lines.append(detail)
        for note in entry.notes:
            lines.append(f"      note: {note}")
    return "\n".join(lines)
