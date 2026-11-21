"""MIMIC-IV-Ext-CDM (Hager et al., Nat Med 2024; PhysioNet credentialed; 2,400 cases).

Static full-information use only (interactive test ordering is out of scope to
keep clear of DEFER-Dx). Files per ConvertPhysionet.py in
github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset:
history_of_present_illness.csv (hpi), physical_examination.csv (pe),
laboratory_tests.csv (itemid, valuestr, ref_range_lower/upper),
microbiology.csv, radiology_reports.csv (modality, region, exam_name, text),
discharge_diagnosis.csv, icd_diagnosis.csv, lab_test_mapping.csv,
pathology_ids.json (pathology -> hadm_ids; appendicitis, cholecystitis,
diverticulitis, pancreatitis).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from clinblueprint.data.records import CaseRecord, build_note
from clinblueprint.schema.graph import dx_key

PATHOLOGIES = ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def load_mimic_cdm(root: str | Path, setting: str = "full_information", max_labs: int = 200) -> List[CaseRecord]:
    import pandas as pd

    root = Path(root)
    hpi = pd.read_csv(root / "history_of_present_illness.csv")
    pe = pd.read_csv(root / "physical_examination.csv")
    with open(root / "pathology_ids.json") as fh:
        patho = json.load(fh)
    label_of: Dict[int, str] = {}
    for p, ids in patho.items():
        for i in ids:
            label_of[int(i)] = p

    name_of: Dict[int, str] = {}
    mp = root / "lab_test_mapping.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        label_col = next((c for c in ("label", "name", "test_name") if c in m.columns), None)
        if label_col and "itemid" in m.columns:
            name_of = dict(zip(m["itemid"].astype(int), m[label_col].astype(str)))

    pe_of = dict(zip(pe["hadm_id"].astype(int), pe["pe"].astype(str)))
    labs: Dict[int, List[str]] = {}
    micro: Dict[int, List[str]] = {}
    rad: Dict[int, List[str]] = {}
    if setting == "full_information":
        lab = pd.read_csv(root / "laboratory_tests.csv")
        for _, r in lab.iterrows():
            nm = name_of.get(int(r["itemid"]), f"item {r['itemid']}")
            rng = ""
            if r.get("ref_range_lower") == r.get("ref_range_lower") and r.get("ref_range_upper") == r.get("ref_range_upper"):
                rng = f" (ref {r['ref_range_lower']}-{r['ref_range_upper']})"
            labs.setdefault(int(r["hadm_id"]), []).append(f"{nm}: {r['valuestr']}{rng}")
        if (root / "microbiology.csv").exists():
            mic = pd.read_csv(root / "microbiology.csv")
            for _, r in mic.iterrows():
                micro.setdefault(int(r["hadm_id"]), []).append(f"{r.get('test_itemid')}: {r.get('valuestr')}")
        if (root / "radiology_reports.csv").exists():
            rr = pd.read_csv(root / "radiology_reports.csv")
            for _, r in rr.iterrows():
                rad.setdefault(int(r["hadm_id"]), []).append(
                    f"{r.get('modality')} {r.get('region')} ({r.get('exam_name')}): {r.get('text')}")

    out: List[CaseRecord] = []
    for _, r in hpi.iterrows():
        hid = int(r["hadm_id"])
        if hid not in label_of:
            continue
        secs = [("History of Present Illness", str(r["hpi"])), ("Physical Exam", pe_of.get(hid, ""))]
        if setting == "full_information":
            secs += [("Laboratory Tests", "\n".join(labs.get(hid, [])[:max_labs])),
                     ("Microbiology", "\n".join(micro.get(hid, []))),
                     ("Radiology", "\n\n".join(rad.get(hid, [])))]
        note, offs = build_note([s for s in secs if s[1]])
        out.append(CaseRecord(
            case_id=f"cdm:{hid}", dataset="mimic_cdm", provenance="physionet", patient_id=f"hadm:{hid}",
            note=note, sections=dict(secs), section_offsets=offs, final_dx=dx_key("open", label_of[hid]),
            candidates=[dx_key("open", p) for p in PATHOLOGIES], split="test", meta={"setting": setting},
        ))
    return out
