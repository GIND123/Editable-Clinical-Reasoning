"""MedCaseReasoning (public, CC-BY 4.0; HF: zou-lab/MedCaseReasoning).

14,489 PMC-OA case reports: 13,092 train / 897 test, with fields
``case_prompt`` (presentation before the differential), ``diagnostic_reasoning``
(numbered clinician reasoning statements) and ``final_diagnosis``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from clinblueprint.data.records import CaseRecord
from clinblueprint.schema.graph import dx_key
from clinblueprint.utils.io import read_jsonl

HF_ID = "zou-lab/MedCaseReasoning"
_PMCID_KEYS = ("pmcid", "PMCID", "accession_id", "pmc_id", "article_id")


def _row_to_record(row: dict, split: str, idx: int) -> CaseRecord:
    pmcid = next((str(row[k]) for k in _PMCID_KEYS if row.get(k)), None)
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    note = row.get("case_prompt") or row.get("prompt") or ""
    return CaseRecord(
        case_id=f"mcr:{split}:{pmcid or idx}", dataset="medcasereasoning", provenance="public",
        patient_id=pmcid or f"mcr-{split}-{idx}", note=note, sections={"Case": note},
        section_offsets={"Case": (0, len(note))},
        final_dx=dx_key("open", str(row.get("final_diagnosis", "")).strip()),
        rationale=row.get("diagnostic_reasoning"), split=split,
        meta={"pmcid": pmcid, "journal": row.get("journal"), "title": row.get("title"),
              "article_link": row.get("article_link"), "publication_date": row.get("publication_date")},
    )


def load_medcasereasoning(split: str = "train", local_jsonl: Optional[str | Path] = None,
                          cache_dir: Optional[str] = None) -> List[CaseRecord]:
    if local_jsonl:
        rows = read_jsonl(local_jsonl)
    else:
        from datasets import load_dataset  # lazy

        try:
            ds = load_dataset(HF_ID, split=split, cache_dir=cache_dir)
        except Exception:  # some versions expose splits as configs
            ds = load_dataset(HF_ID, split, cache_dir=cache_dir)
            ds = ds[split] if hasattr(ds, "keys") and split in ds else ds[list(ds.keys())[0]]
        rows = [dict(r) for r in ds]
    return [_row_to_record(r, split, i) for i, r in enumerate(rows)]
