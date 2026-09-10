"""Locating template resources.

TemplateFlow filenames are not uniform across templates: the resolution
entity may be written 'res-01' or 'res-1', and cohort-specific (e.g.
paediatric) templates carry an extra 'cohort-' entity.  Hard-coding one
spelling makes --template_dir useless for exactly the templates it exists to
support, so entities are matched by glob instead.
"""

from __future__ import annotations

import re
from pathlib import Path


def _resolution(path: Path) -> float:
    """Numeric value of a filename's res- entity, for ordering matches."""
    match = re.search(r"_res-(\d+)", path.name)
    return float(match.group(1)) if match else float("inf")


def find_template_file(
    template_dir: Path, template_name: str, suffix: str, **entities: str
) -> str:
    """Path to one template resource.

    ``entities`` are matched loosely, so ``res="01"`` also finds ``res-1``.
    When several resolutions match, the finest (numerically smallest ``res-``)
    wins; everything is resampled to --out_res downstream anyway.
    """
    template_dir = Path(template_dir)
    parts = "".join(f"*{key}-{value}" for key, value in entities.items())
    pattern = f"tpl-{template_name}{parts}*_{suffix}.nii.gz"
    matches = sorted(template_dir.glob(pattern))

    if not matches and "res" in entities:
        # tolerate 'res-1' where 'res-01' was asked for, and vice versa
        relaxed = {k: v for k, v in entities.items() if k != "res"}
        relaxed_parts = "".join(f"*{k}-{v}" for k, v in relaxed.items())
        pattern = f"tpl-{template_name}{relaxed_parts}*_{suffix}.nii.gz"
        matches = sorted(template_dir.glob(pattern))

    if not matches:
        listing = "\n".join(f"    {p.name}" for p in sorted(template_dir.glob("*")))
        raise FileNotFoundError(
            f"No template file for '{suffix}' ({entities}) in {template_dir}\n"
            f"  looked for: {pattern}\n"
            f"  directory contains:\n{listing or '    (empty)'}\n"
            "Point --template_dir at a directory using TemplateFlow naming."
        )
    return str(min(matches, key=_resolution))
