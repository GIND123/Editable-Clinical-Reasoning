"""Unified case record shared by every loader."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


class CaseRecord(BaseModel):
    case_id: str
    dataset: str
    provenance: str = "public"  # public | dua | physionet  (see compliance.guard)
    patient_id: Optional[str] = None
    note: str
    sections: Dict[str, str] = Field(default_factory=dict)
    section_offsets: Dict[str, Tuple[int, int]] = Field(default_factory=dict)
    final_dx: Optional[str] = None   # dx key "<category>::<leaf>" or "open::<free text>"
    category: Optional[str] = None
    candidates: Optional[List[str]] = None  # fixed candidate list when the task provides one
    gold_graph: Optional[Dict[str, Any]] = None
    rationale: Optional[str] = None  # clinician-authored reasoning text, if any
    split: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


def build_note(sections: Sequence[Tuple[str, str]], sep: str = "\n\n") -> Tuple[str, Dict[str, Tuple[int, int]]]:
    """Concatenate (name, text) sections as ``Name:\\ntext`` and return char offsets of each body."""
    parts: List[str] = []
    offsets: Dict[str, Tuple[int, int]] = {}
    pos = 0
    for i, (name, text) in enumerate(sections):
        text = (text or "").replace("﻿", "").strip()
        head = f"{name}:\n"
        if i:
            parts.append(sep)
            pos += len(sep)
        parts.append(head)
        pos += len(head)
        offsets[name] = (pos, pos + len(text))
        parts.append(text)
        pos += len(text)
    return "".join(parts), offsets
