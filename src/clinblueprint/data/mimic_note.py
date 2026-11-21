"""MIMIC-IV-Note discharge summaries -> DiReCT-style 6-section inputs (silver data).

Only the six DiReCT input sections are kept. Leakage sections (Brief Hospital
Course, Discharge Diagnosis, Discharge Condition/Instructions, Medications on
Discharge, Followup) are dropped from the input; Discharge Diagnosis is kept
as the *label* only. DiReCT subjects are excluded (``exclude_subjects``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set

from clinblueprint.data.records import CaseRecord, build_note
from clinblueprint.schema.graph import dx_key

INPUT_HEADERS = {
    "Chief Complaint": [r"chief complaint"],
    "History of Present Illness": [r"history of present illness", r"hpi"],
    "Past Medical History": [r"past medical history", r"pmh"],
    "Family History": [r"family history"],
    "Physical Exam": [r"physical exam(?:ination)?"],
    "Pertinent Results": [r"pertinent results"],
}
LABEL_HEADERS = [r"discharge diagnos[ie]s", r"final diagnos[ie]s"]
ALL_HEADERS = (
    [h for hs in INPUT_HEADERS.values() for h in hs] + LABEL_HEADERS +
    [r"brief hospital course", r"discharge condition", r"discharge instructions", r"discharge medications",
     r"medications on admission", r"discharge disposition", r"followup instructions", r"social history",
     r"allergies", r"major surgical or invasive procedure", r"attending", r"service", r"facility"]
)
_HDR = re.compile(r"^\s*(" + "|".join(ALL_HEADERS) + r")\s*:\s*$", re.IGNORECASE | re.MULTILINE)


def split_sections(text: str) -> dict:
    out = {}
    hits = list(_HDR.finditer(text))
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        out[m.group(1).strip().lower()] = text[m.end():end].strip()
    return out


def _pick(secs: dict, patterns) -> str:
    for k, v in secs.items():
        if any(re.fullmatch(p, k, re.IGNORECASE) for p in patterns):
            return v
    return ""


def discharge_to_record(note_id: str, subject_id: str, hadm_id: Optional[str], text: str) -> Optional[CaseRecord]:
    secs = split_sections(text)
    label = _pick(secs, LABEL_HEADERS)
    inputs = [(name, _pick(secs, pats)) for name, pats in INPUT_HEADERS.items()]
    if not label or sum(bool(v) for _, v in inputs) < 3:
        return None
    primary = re.split(r"\n|;", re.sub(r"(?i)primary( diagnos[ie]s)?\s*:?", "", label).strip())[0].strip(" -*:")
    note, offs = build_note([(n, v) for n, v in inputs if v])
    # Brief Hospital Course = clinician-written reasoning: TEACHER-ONLY context for silver graphs,
    # never part of the model input (``note``).
    bhc = _pick(secs, [r"brief hospital course"])[:6000]
    return CaseRecord(
        case_id=f"mimicnote:{note_id}", dataset="mimic_iv_note", provenance="physionet", patient_id=str(subject_id),
        note=note, sections={n: v for n, v in inputs if v}, section_offsets=offs, final_dx=dx_key("open", primary),
        rationale=bhc or None, split="train", meta={"hadm_id": hadm_id, "discharge_diagnosis_raw": label[:2000]},
    )


def iter_mimic_discharge(path: str | Path, exclude_subjects: Optional[Set[str]] = None,
                         limit: Optional[int] = None, chunksize: int = 5000) -> Iterator[CaseRecord]:
    """Stream ``note/discharge.csv.gz`` (columns note_id, subject_id, hadm_id, text)."""
    import pandas as pd

    n = 0
    excl = exclude_subjects or set()
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=["note_id", "subject_id", "hadm_id", "text"]):
        for _, r in chunk.iterrows():
            if str(r["subject_id"]) in excl:
                continue
            rec = discharge_to_record(str(r["note_id"]), str(r["subject_id"]), str(r["hadm_id"]), str(r["text"]))
            if rec is None:
                continue
            yield rec
            n += 1
            if limit and n >= limit:
                return


def subjects_of(records: Iterable[CaseRecord]) -> Set[str]:
    return {r.patient_id for r in records if r.patient_id}
