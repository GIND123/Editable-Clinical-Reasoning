"""Export clinician review sheets (CSV) for the silver-graph audit and edit verification.

    python scripts/export_audit_sheets.py silver  --graphs data/silver/medcasereasoning_train.silver.jsonl --n 100 \
        --out clinician/review/silver_audit.csv
    python scripts/export_audit_sheets.py edits   --cases data/benchmark/clinedit_faith/public.cases.jsonl \
        --edits data/benchmark/clinedit_faith/public.verify_sample.jsonl --out clinician/review/edit_verification.csv

Credentialed portions must be reviewed inside the approved environment only.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from clinblueprint.data.records import CaseRecord
from clinblueprint.edits.types import Edit
from clinblueprint.schema import ReasoningGraph
from clinblueprint.utils.io import read_jsonl

SILVER_COLS = ["case_id", "final_dx", "graph", "obs_verbatim_and_polarity_ok(1-5)", "rationales_correct(1-5)",
               "diagnosis_correct(Y/N)", "mnm_sensible(Y/N)", "critical_error(Y/N)", "comments"]
EDIT_COLS = ["edit_id", "case_id", "edit_class", "mode", "description", "measured_diagnosis", "expected_effect",
             "agree(Y/N)", "your_expected_effect(increase/decrease/no_change/flag)", "confidence(1-5)", "comments"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["silver", "edits"])
    ap.add_argument("--graphs")
    ap.add_argument("--cases")
    ap.add_argument("--edits")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        if a.kind == "silver":
            rows = read_jsonl(a.graphs)
            random.Random(a.seed).shuffle(rows)
            w = csv.writer(fh)
            w.writerow(SILVER_COLS)
            for r in rows[: a.n]:
                rec = CaseRecord(**r["record"])
                w.writerow([rec.case_id, rec.final_dx, ReasoningGraph.from_dict(r["graph"]).summary()] +
                           [""] * (len(SILVER_COLS) - 3))
        else:
            cases = {r["case_id"]: r for r in read_jsonl(a.cases)}
            w = csv.writer(fh)
            w.writerow(EDIT_COLS)
            for d in read_jsonl(a.edits):
                e = Edit(**d)
                if e.case_id not in cases:
                    continue
                w.writerow([e.edit_id, e.case_id, e.edit_class.value, e.mode.value, e.describe(),
                            e.measured_dx.split("::")[-1], e.expected_effect.value] + [""] * 4)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
