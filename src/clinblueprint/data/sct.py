"""Script Concordance Test items (ER-Reason SCT, SCT-Bench / concor.dance)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from clinblueprint.utils.io import read_jsonl


class SCTItem(BaseModel):
    item_id: str
    case_id: str
    vignette: str
    hypothesis: str
    new_info: str
    gold_update: Optional[int] = Field(default=None, ge=-2, le=2)  # consensus / modal answer
    panel: Optional[Dict[int, int]] = None   # answer -> number of panel experts (aggregate scoring)
    timestep: Optional[int] = None
    differential: List[str] = Field(default_factory=list)  # fixed list to re-rank (ER-Reason)
    gold_ranking: Optional[List[str]] = None
    physician_rationale: Optional[str] = None
    source: str = "sct"
    provenance: str = "dua"


def load_sct_jsonl(path: str | Path, source: str = "sct_bench", provenance: str = "dua") -> List[SCTItem]:
    """Generic SCT JSONL (one item per line) with the SCTItem fields.

    SCT-Bench public items and held-out access (concor.dance, DUA) arrive in
    their own format; convert them once to this schema with
    scripts/data/convert_sct_bench.py (fill in after access is granted).
    """
    out = []
    for d in read_jsonl(path):
        d.setdefault("source", source)
        d.setdefault("provenance", provenance)
        if d.get("panel"):
            d["panel"] = {int(k): int(v) for k, v in d["panel"].items()}
        out.append(SCTItem(**d))
    return out
