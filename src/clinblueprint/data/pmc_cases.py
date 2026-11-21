"""Extra public PMC case-report corpora for silver-graph scale-up (optional).

* MultiCaRe (Nievas Offidani et al., Data in Brief 2023, doi:10.1016/j.dib.2023.110008):
  96,428 clinical cases from 75,382 PMC-OA articles (1990-2023). Zenodo.
* PMC-Patients (Zhao et al., Sci Data 2023): ~167k patient summaries. HF: zhengyun21/PMC-Patients.

CONTAMINATION: both overlap the PMC-OA source of MedCaseReasoning. Every
record carries its PMCID, and ``dedup.drop_overlap`` removes any article
present in the MedCaseReasoning *test* split before training.
These corpora have no explicit "final diagnosis" field: labels come from the
local teacher and are filtered strictly (data/silver.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from clinblueprint.data.records import CaseRecord


def iter_pmc_patients(limit: Optional[int] = None, cache_dir: Optional[str] = None) -> Iterator[CaseRecord]:
    from datasets import load_dataset  # lazy

    ds = load_dataset("zhengyun21/PMC-Patients", split="train", cache_dir=cache_dir)
    for i, r in enumerate(ds):
        pmid = str(r.get("PMID") or "")
        text = r.get("patient") or ""
        yield CaseRecord(case_id=f"pmcp:{r.get('patient_uid', i)}", dataset="pmc_patients", provenance="public",
                         patient_id=str(r.get("patient_uid", i)), note=text, sections={"Case": text},
                         section_offsets={"Case": (0, len(text))}, split="train",
                         meta={"pmid": pmid, "title": r.get("title")})
        if limit and i + 1 >= limit:
            return


def iter_multicare(cases_csv: str | Path, limit: Optional[int] = None) -> Iterator[CaseRecord]:
    """MultiCaRe ``cases.csv`` (columns include case_id, case_text, article_id / pmcid). VERIFY column names."""
    import pandas as pd

    df = pd.read_csv(cases_csv)
    txt_col = next(c for c in ("case_text", "text", "case") if c in df.columns)
    pmc_col = next((c for c in ("pmcid", "article_id", "pmc_id") if c in df.columns), None)
    for i, r in df.iterrows():
        text = str(r[txt_col])
        pmcid = str(r[pmc_col]) if pmc_col else None
        yield CaseRecord(case_id=f"multicare:{r.get('case_id', i)}", dataset="multicare", provenance="public",
                         patient_id=str(r.get("case_id", i)), note=text, sections={"Case": text},
                         section_offsets={"Case": (0, len(text))}, split="train", meta={"pmcid": pmcid})
        if limit and i + 1 >= limit:
            return
