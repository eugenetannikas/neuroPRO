"""Aggregate per-subject QC into a group summary TSV and an HTML index."""

import html
import json
import os

import pandas as pd

rows = []
for subject, t1w_json, motion_tsv in zip(
    snakemake.params.subjects,
    snakemake.input.t1w_jsons,
    snakemake.input.motion_tsvs,
):
    with open(t1w_json) as f:
        sel = json.load(f)
    chosen = next(
        (c for c in sel.get("Candidates", []) if c.get("chosen")), {}
    )
    motion = pd.read_csv(motion_tsv, sep="\t")
    rows.append(
        {
            "subject": f"sub-{subject}",
            "t1w_chosen": os.path.basename(sel.get("SourceFile", "")),
            "t1w_protocol": chosen.get("series_description", ""),
            "t1w_series_number": chosen.get("series_number", ""),
            "t1w_n_candidates": len(sel.get("Candidates", [])),
            "t1w_sharpness": chosen.get("sharpness", ""),
            "t1w_selection_method": sel.get("SelectionMethod", ""),
            "nm_n_images": len(motion),
            "nm_max_translation_mm": motion["translation_mm"].max(),
            "nm_max_rotation_deg": motion["rotation_deg"].max(),
        }
    )

df = pd.DataFrame(rows)
df.to_csv(snakemake.output.tsv, sep="\t", index=False)

# simple html index with links to the per-subject snapshots
group_dir = os.path.dirname(snakemake.output.html)
png_by_subject = {}
for png in snakemake.input.pngs:
    sub = os.path.basename(png).split("_")[0]
    png_by_subject.setdefault(sub, []).append(
        os.path.relpath(png, group_dir)
    )

parts = [
    "<html><head><title>neuroPRO QC</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#eee}",
    "table{border-collapse:collapse}td,th{border:1px solid #555;",
    "padding:4px 8px}a{color:#8cf}img{max-width:100%}</style></head><body>",
    "<h1>neuroPRO QC summary</h1>",
    df.to_html(index=False, border=0),
    "<h1>Snapshots</h1>",
]
for sub in sorted(png_by_subject):
    parts.append(f"<h2>{html.escape(sub)}</h2>")
    for rel in sorted(png_by_subject[sub]):
        parts.append(
            f'<p><a href="{rel}">{html.escape(os.path.basename(rel))}</a>'
            f'<br><img src="{rel}" loading="lazy"></p>'
        )
parts.append("</body></html>")

with open(snakemake.output.html, "w") as f:
    f.write("\n".join(parts))
