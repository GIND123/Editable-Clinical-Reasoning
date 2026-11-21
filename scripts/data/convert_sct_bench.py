"""Convert SCT-Bench / concor.dance items to the repo's SCTItem JSONL.

The released format is only known after access is granted, so columns are
mapped by command-line arguments (edit the defaults once you see the files).
Panel distributions are expected either as one column per answer (-2..+2
counts) or as a JSON column.

    python scripts/data/convert_sct_bench.py --input items.csv --out data/raw/sct_bench/sct_bench.jsonl \
        --id-col id --vignette-col vignette --hypothesis-col hypothesis --info-col new_information \
        --panel-cols m2,m1,z,p1,p2
"""

from __future__ import annotations

import argparse
import json

LIKERT = [-2, -1, 0, 1, 2]


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--case-col", default=None)
    ap.add_argument("--vignette-col", default="vignette")
    ap.add_argument("--hypothesis-col", default="hypothesis")
    ap.add_argument("--info-col", default="new_information")
    ap.add_argument("--panel-cols", default=None, help="comma list of 5 count columns for -2..+2")
    ap.add_argument("--panel-json-col", default=None)
    ap.add_argument("--source", default="sct_bench")
    ap.add_argument("--provenance", default="dua")
    a = ap.parse_args()
    df = pd.read_csv(a.input) if a.input.endswith(".csv") else pd.read_json(a.input, lines=True)
    with open(a.out, "w") as fh:
        for _, r in df.iterrows():
            if a.panel_cols:
                panel = {l: int(r[c]) for l, c in zip(LIKERT, a.panel_cols.split(","))}
            elif a.panel_json_col:
                panel = {int(k): int(v) for k, v in json.loads(r[a.panel_json_col]).items()}
            else:
                panel = None
            gold = max(panel, key=panel.get) if panel else None
            fh.write(json.dumps({
                "item_id": f"{a.source}:{r[a.id_col]}", "case_id": f"{a.source}:{r[a.case_col] if a.case_col else r[a.id_col]}",
                "vignette": str(r[a.vignette_col]), "hypothesis": str(r[a.hypothesis_col]), "new_info": str(r[a.info_col]),
                "panel": panel, "gold_update": gold, "source": a.source, "provenance": a.provenance}) + "\n")
    print(f"wrote {len(df)} items")


if __name__ == "__main__":
    main()
