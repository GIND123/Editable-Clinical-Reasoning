"""ER-Reason (UCSF ED; PhysioNet credentialed; second-site validation).

Files (from the official repo README, github.com/AlaaLab/ER-Reason):
  er_reason.csv                one row per encounter (notes, one-liner, ED dx, ...)
  icd_10_codes.csv             ground-truth ICD-10 per encounter
  annotated_sct.csv            encounterkey, One_Sentence_Extracted, differential_1..5, evidence_1..5
  sct-annotations.csv          master list of the 194 SCT encounterkeys
  sct_cleaned_annotations.csv  encounterkey, differential, rationale (physician)
  gt_clean.csv                 encounterkey, differential, dxupdate (-2..+2), dxtrajectory (dict dx->rank)

Licence note: PhysioNet lists ER-Reason under the Contributor Review Health
Data License (per-study review). Check the terms before any derived release.

Leakage: the current-visit ED provider note contains the assessment and
diagnosis. ``input_mode="one_liner_plus_history"`` (default) uses the one-liner
plus *prior* notes; ``"ed_note_truncated"`` cuts the ED note at the first
assessment/MDM/diagnosis header. Records whose input still contains the label
string are marked ``meta.label_in_input``. Match the official final_diagnosis.py
protocol before reporting numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from clinblueprint.data.records import CaseRecord, build_note
from clinblueprint.data.sct import SCTItem
from clinblueprint.schema.graph import dx_key, normalize_label

TYPO_FIXES = {  # copied from ER-Reason sct_eval.py so differential strings match gold
    "aspergillosus": "aspergillosis",
    "arrythmia": "arrhythmia",
    "Arrythmia": "Arrhythmia",
    "electrolyte-induced arrythmia": "electrolyte-induced arrhythmia",
    "intraparynchemal hemorrhage": "intraparenchymal hemorrhage",
    "intracranial hemorrhage:": "intracranial hemorrhage",
}
HISTORY_COLS = [
    ("Prior Discharge Summary", "Discharge_Summary_Text"),
    ("Prior Progress Note", "Progress_Note_Text"),
    ("Prior H&P", "HP_Note_Text"),
    ("Prior Echo", "Echo_Text"),
    ("Prior Imaging", "Imaging_Text"),
    ("Prior Consult", "Consult_Text"),
    ("Prior ECG", "ECG_Text"),
]
_ASSESS_RE = re.compile(r"\n\s*(assessment|medical decision making|mdm|impression|diagnos[ie]s|ed course)\b",
                        re.IGNORECASE)


def _fix(dx: str) -> str:
    s = str(dx).strip().rstrip(":").strip()
    return TYPO_FIXES.get(s, s)


def _val(row, col) -> str:
    v = row.get(col)
    return "" if v is None or (isinstance(v, float) and v != v) else str(v)


def load_er_reason_cases(root: str | Path, input_mode: str = "one_liner_plus_history",
                         max_chars_per_note: int = 6000) -> List[CaseRecord]:
    import pandas as pd

    root = Path(root)
    df = pd.read_csv(root / "er_reason.csv")
    icd = {}
    if (root / "icd_10_codes.csv").exists():
        idf = pd.read_csv(root / "icd_10_codes.csv")
        for _, r in idf.iterrows():
            icd.setdefault(str(r["encounterkey"]), []).append(
                {"code": r.get("value"), "name": r.get("displaystring")})
    out: List[CaseRecord] = []
    for _, row in df.iterrows():
        row = row.to_dict()
        enc = str(row["encounterkey"])
        secs = [("Chief Complaint", _val(row, "primarychiefcomplaintname")),
                ("One-liner", _val(row, "One_Sentence_Extracted"))]
        if input_mode == "one_liner_plus_history":
            secs += [(name, _val(row, col)[:max_chars_per_note]) for name, col in HISTORY_COLS if _val(row, col)]
        elif input_mode == "ed_note_truncated":
            ed = _val(row, "ED_Provider_Notes_Text")
            m = _ASSESS_RE.search(ed)
            secs.append(("ED Provider Note (truncated)", (ed[:m.start()] if m else ed)[:max_chars_per_note]))
        else:
            raise ValueError(input_mode)
        note, offs = build_note([s for s in secs if s[1]])
        label = _val(row, "primaryeddiagnosisname")
        out.append(CaseRecord(
            case_id=f"er:{enc}", dataset="er_reason", provenance="physionet",
            patient_id=str(row.get("patientdurablekey")), note=note, sections=dict(secs), section_offsets=offs,
            final_dx=dx_key("open", label), split="test",
            meta={"icd10": icd.get(enc, []), "acuity": row.get("acuitylevel"),
                  "disposition": row.get("eddisposition"), "input_mode": input_mode,
                  "label_in_input": bool(label) and normalize_label(label) in normalize_label(note)},
        ))
    return out


def load_er_reason_sct(root: str | Path) -> List[SCTItem]:
    import pandas as pd

    root = Path(root)
    ann = pd.read_csv(root / "annotated_sct.csv")
    keys = set(pd.read_csv(root / "sct-annotations.csv")["encounterkey"].astype(str).str.strip())
    gt = pd.read_csv(root / "gt_clean.csv")
    rats = pd.read_csv(root / "sct_cleaned_annotations.csv") if (root / "sct_cleaned_annotations.csv").exists() else None

    gold: Dict = {}
    for _, r in gt.iterrows():
        k = (str(r["encounterkey"]).strip(), _fix(r["differential"]).lower())
        traj = r.get("dxtrajectory")
        ranking = None
        if isinstance(traj, str) and traj.strip():
            try:
                t = {_fix(a): b for a, b in json.loads(traj).items()}
                ranking = [d for d, _ in sorted(t.items(), key=lambda x: x[1])]
            except json.JSONDecodeError:
                ranking = None
        upd = r.get("dxupdate")
        gold[k] = (int(upd) if upd == upd and upd is not None else None, ranking)
    rationale: Dict = {}
    if rats is not None:
        for _, r in rats.iterrows():
            rationale[(str(r["encounterkey"]).strip(), _fix(r["differential"]).lower())] = r.get("rationale")

    items: List[SCTItem] = []
    ann = ann.drop_duplicates("encounterkey")
    for _, row in ann.iterrows():
        row = row.to_dict()
        enc = str(row["encounterkey"]).strip()
        if enc not in keys:
            continue
        dxs = [_fix(row[f"differential_{t}"]) for t in range(1, 6)
               if _val(row, f"differential_{t}").strip()]
        for t in range(1, 6):
            dx, ev = _val(row, f"differential_{t}").strip(), _val(row, f"evidence_{t}").strip()
            if not dx or not ev:
                continue
            dx = _fix(dx)
            upd, ranking = gold.get((enc, dx.lower()), (None, None))
            items.append(SCTItem(
                item_id=f"er-sct:{enc}:{t}", case_id=f"er:{enc}", vignette=_val(row, "One_Sentence_Extracted"),
                hypothesis=dx, new_info=ev, gold_update=upd, timestep=t, differential=dxs,
                gold_ranking=ranking, physician_rationale=rationale.get((enc, dx.lower())),
                source="er_reason_sct", provenance="physionet",
            ))
    return items
