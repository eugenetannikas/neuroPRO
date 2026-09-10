"""Aggregate per-subject QC into a group summary TSV and an HTML index.

Every entry that was processed appears in the summary, including those that
only got as far as the native-space NM average (no T1w to coregister to).
Their status column says so, and their T1w columns are blank -- a partially
processed subject should be visible in the table rather than missing from it.
"""

import html
import json
import os

import pandas as pd


def motion_summary(motion_tsv):
    motion = pd.read_csv(motion_tsv, sep="\t")
    return {
        "nm_n_images": len(motion),
        "nm_max_translation_mm": motion["translation_mm"].max(),
        "nm_max_rotation_deg": motion["rotation_deg"].max(),
    }


def identity(entry):
    row = {"subject": f"sub-{entry['subject']}"}
    if entry.get("session"):
        row["session"] = f"ses-{entry['session']}"
    return row


rows = []

for entry, t1w_json, motion_tsv in zip(
    snakemake.params.full,
    snakemake.input.t1w_jsons,
    snakemake.input.full_motion,
):
    with open(t1w_json) as f:
        sel = json.load(f)
    chosen = next((c for c in sel.get("Candidates", []) if c.get("chosen")), {})
    row = identity(entry)
    row.update(
        {
            "status": "complete",
            "t1w_chosen": os.path.basename(sel.get("SourceFile", "")),
            "t1w_protocol": chosen.get("series_description", ""),
            "t1w_series_number": chosen.get("series_number", ""),
            "t1w_n_candidates": len(sel.get("Candidates", [])),
            "t1w_sharpness": chosen.get("sharpness", ""),
            "t1w_selection_method": sel.get("SelectionMethod", ""),
        }
    )
    row.update(motion_summary(motion_tsv))
    rows.append(row)

for entry, motion_tsv in zip(snakemake.params.nmonly, snakemake.input.nmonly_motion):
    row = identity(entry)
    row["status"] = "nm-only (no T1w)"
    row.update(motion_summary(motion_tsv))
    rows.append(row)

df = pd.DataFrame(rows)
sort_cols = [c for c in ("subject", "session") if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols).reset_index(drop=True)
df.to_csv(snakemake.output.tsv, sep="\t", index=False)

# simple html index with links to the per-subject snapshots.  Snapshots are
# grouped by the subject _and_ session entities of their filename, so the
# sessions of one subject stay in separate sections.
group_dir = os.path.dirname(snakemake.output.html)
png_by_entry = {}
for png in snakemake.input.pngs:
    name_parts = os.path.basename(png).split("_")
    key = "_".join(p for p in name_parts if p.startswith(("sub-", "ses-")))
    png_by_entry.setdefault(key, []).append(os.path.relpath(png, group_dir))

parts = [
    "<html><head><title>neuroPRO QC</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#eee}",
    "table{border-collapse:collapse}td,th{border:1px solid #555;",
    "padding:4px 8px}a{color:#8cf}img{max-width:100%}</style></head><body>",
    "<h1>neuroPRO QC summary</h1>",
    df.to_html(index=False, border=0),
    "<h1>Snapshots</h1>",
]
for key in sorted(png_by_entry):
    parts.append(f"<h2>{html.escape(key)}</h2>")
    for rel in sorted(png_by_entry[key]):
        parts.append(
            f'<p><a href="{rel}">{html.escape(os.path.basename(rel))}</a>'
            f'<br><img src="{rel}" loading="lazy"></p>'
        )
parts.append("</body></html>")

with open(snakemake.output.html, "w") as f:
    f.write("\n".join(parts))
