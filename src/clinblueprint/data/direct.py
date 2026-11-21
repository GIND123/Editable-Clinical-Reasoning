"""MIMIC-IV-Ext-DiReCT loader (PhysioNet, doi:10.13026/yf96-kc87; 511 notes).

Directory layout (per the official repo)::

    <root>/<Disease Category>/<PDD Category>/<note>.json   (or <Category>/<note>.json)

Each JSON has ``input1..input6`` (Chief Complaint, HPI, PMH, Family History,
Physical Exam, Pertinent Results) plus one nested tree whose keys are
``"<content>$<Type>"`` with Type in {Intermedia_k (diagnosis), Cause_k
(rationale), InputN (observation from section N)}. The top-level Intermedia key
is the final (most specific) diagnosis; nested Intermedia children are more
general diagnoses (the "chain" from suspected to PDD).

We convert this tree into a ReasoningGraph:
  Intermedia -> Diagnosis (parent-specific *refines* child-general)
  Cause      -> Rationale (target = enclosing Intermedia; premises = child Inputs)
  InputN     -> Observation (section = DIRECT_SECTIONS[inputN])

VERIFY ON ACCESS: file names. If they are MIMIC-IV-Note note_ids
("<subject_id>-DS-<n>") we recover subject_id for patient-level CV and for
excluding DiReCT patients from silver MIMIC data. Otherwise link via MIMIC-IV-Note.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

from clinblueprint.data.records import CaseRecord, build_note
from clinblueprint.schema.graph import Diagnosis, Observation, Rationale, ReasoningGraph, dx_key

DIRECT_SECTIONS = {
    "input1": "Chief Complaint",
    "input2": "History of Present Illness",
    "input3": "Past Medical History",
    "input4": "Family History",
    "input5": "Physical Exam",
    "input6": "Pertinent Results",
}
_NOTE_ID = re.compile(r"(\d{8})-DS-(\d+)")


def _split_key(key: str) -> Tuple[str, str]:
    if "$" not in key:
        raise ValueError(f"unexpected DiReCT key without '$': {key[:60]!r}")
    content, typ = key.rsplit("$", 1)
    return content.strip(), typ.strip()


def parse_direct_tree(data: dict, case_id: str, category: str) -> Tuple[ReasoningGraph, Optional[str]]:
    G = ReasoningGraph(case_id=case_id, meta={"source": "direct_gold"})
    final_leaf: Optional[str] = None

    def visit(key: str, sub: dict, parent_dx: Optional[str], parent_rat: Optional[str]) -> None:
        nonlocal final_leaf
        content, typ = _split_key(key)
        kind = typ.split("_")[0]
        if kind == "Intermedia":
            did = G.new_id("D")
            G.add(Diagnosis(id=did, category=category, leaf=content))
            if parent_dx is not None:
                G.diagnoses[parent_dx].refines = did  # specific refines general
            elif final_leaf is None:
                final_leaf = content
            for k, v in (sub or {}).items():
                visit(k, v, did, None)
        elif kind == "Cause":
            if parent_dx is None:
                warnings.warn(f"{case_id}: rationale outside any diagnosis; skipped")
                return
            rid = G.new_id("R")
            G.add(Rationale(id=rid, premises=[], claim=content, target=parent_dx))
            for k, v in (sub or {}).items():
                visit(k, v, parent_dx, rid)
        elif kind.lower().startswith("input"):
            sec = DIRECT_SECTIONS.get(kind.lower())
            oid = G.new_id("O")
            G.add(Observation(id=oid, text=content, section=sec))
            if parent_rat is not None:
                G.rationales[parent_rat].premises.append(oid)
            elif parent_dx is not None:
                G.add(Rationale(id=G.new_id("R"), premises=[oid], claim="", target=parent_dx))
            for k, v in (sub or {}).items():
                visit(k, v, parent_dx, parent_rat)
        else:
            warnings.warn(f"{case_id}: unknown node type {typ!r}")

    for key, val in data.items():
        if key.lower().startswith("input"):
            continue
        visit(key, val, None, None)
    return G, final_leaf


def parse_direct_file(path: str | Path, root: str | Path, provenance: str = "physionet") -> CaseRecord:
    path, root = Path(path), Path(root)
    rel = path.relative_to(root)
    category = rel.parts[0]
    pdd = rel.parts[1] if len(rel.parts) > 2 else None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    secs = [(DIRECT_SECTIONS[k], data.get(k, "")) for k in sorted(DIRECT_SECTIONS) if k in data]
    note, offsets = build_note(secs)
    case_id = f"direct:{rel.with_suffix('').as_posix()}"
    G, final_leaf = parse_direct_tree(data, case_id, category)
    m = _NOTE_ID.search(path.stem)
    patient_id = m.group(1) if m else None
    return CaseRecord(
        case_id=case_id, dataset="direct", provenance=provenance, patient_id=patient_id, note=note,
        sections=dict(secs), section_offsets=offsets,
        final_dx=dx_key(category, final_leaf) if final_leaf else None, category=category,
        gold_graph=G.to_dict(),
        meta={"pdd_folder": pdd, "file": rel.as_posix(), "note_id": m.group(0) if m else None},
    )


def load_direct(root: str | Path, provenance: str = "physionet") -> List[CaseRecord]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no DiReCT json under {root}")
    recs = [parse_direct_file(f, root, provenance) for f in files]
    if not any(r.patient_id for r in recs):
        warnings.warn("DiReCT file names carry no subject_id; patient-level CV falls back to note-level "
                      "unless you link note ids via MIMIC-IV-Note (scripts/data/link_direct_subjects.py)")
    return recs
