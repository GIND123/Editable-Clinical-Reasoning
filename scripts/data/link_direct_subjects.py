"""Link DiReCT notes to MIMIC-IV subject_ids (needed for patient-level CV and for
excluding DiReCT patients from MIMIC-IV-Note silver data).

Matches each DiReCT note's HPI (input2) against MIMIC-IV-Note discharge
summaries by a normalised prefix + fuzzy confirmation. Runs locally only.

    python scripts/data/link_direct_subjects.py \
        --direct data/raw/physionet/mimic-iv-ext-direct/samples \
        --discharge data/raw/physionet/mimic-iv-note/note/discharge.csv.gz \
        --out data/processed/direct_subject_map.json
"""

from __future__ import annotations

import argparse
import json
import re

from clinblueprint.data.direct import load_direct


def norm(s: str, n: int = 160) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())[:n]


def main() -> None:
    import pandas as pd
    from rapidfuzz import fuzz

    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", required=True)
    ap.add_argument("--discharge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_score", type=float, default=90.0)
    a = ap.parse_args()
    recs = load_direct(a.direct)
    want = {norm(r.sections.get("History of Present Illness", ""), 60): r for r in recs}
    want.pop("", None)
    found = {}
    for chunk in pd.read_csv(a.discharge, chunksize=5000, usecols=["note_id", "subject_id", "text"]):
        for _, row in chunk.iterrows():
            t = norm(row["text"], 10**9)
            for key, rec in list(want.items()):
                if key and key in t:
                    hpi = rec.sections.get("History of Present Illness", "")
                    if fuzz.partial_ratio(hpi[:400].lower(), str(row["text"]).lower()) >= a.min_score:
                        found[rec.meta["file"]] = str(row["subject_id"])
                        want.pop(key)
        if not want:
            break
    json.dump(found, open(a.out, "w"), indent=1)
    print(f"linked {len(found)}/{len(recs)} DiReCT notes; unmatched: {len(want)}")


if __name__ == "__main__":
    main()
