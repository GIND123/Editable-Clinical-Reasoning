"""Train/test contamination control.

1. Identifier overlap (PMCID / PMID / subject_id) between training corpora and
   any evaluation split, so the same patient or article is never in both.
2. Near-duplicate text: character 8-gram shingles + MinHash-style signatures
   with LSH banding; candidate pairs are confirmed by Jaccard >= threshold.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from clinblueprint.data.records import CaseRecord


def ids_of(records: Iterable[CaseRecord], keys: Sequence[str] = ("pmcid", "pmid")) -> Set[str]:
    out: Set[str] = set()
    for r in records:
        for k in keys:
            v = r.meta.get(k)
            if v:
                out.add(str(v).upper().replace("PMC", ""))
        if r.patient_id:
            out.add(f"pt:{r.patient_id}")
    return out


def drop_overlap(train: Iterable[CaseRecord], eval_ids: Set[str], keys: Sequence[str] = ("pmcid", "pmid")) -> List[CaseRecord]:
    kept = []
    for r in train:
        rid = {str(r.meta.get(k)).upper().replace("PMC", "") for k in keys if r.meta.get(k)}
        if r.patient_id:
            rid.add(f"pt:{r.patient_id}")
        if not (rid & eval_ids):
            kept.append(r)
    return kept


def _shingles(text: str, k: int = 8) -> Set[str]:
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def _sig(sh: Set[str], n: int = 64) -> Tuple[int, ...]:
    return tuple(
        min(int.from_bytes(hashlib.blake2b(f"{i}:{s}".encode(), digest_size=8).digest(), "big") for s in sh)
        for i in range(n)
    )


def near_duplicates(a: Sequence[CaseRecord], b: Sequence[CaseRecord], threshold: float = 0.8,
                    n_perm: int = 64, bands: int = 16) -> List[Tuple[str, str, float]]:
    rows = n_perm // bands
    sh_a = {r.case_id: _shingles(r.note) for r in a}
    sh_b = {r.case_id: _shingles(r.note) for r in b}
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[str]] = {}
    for cid, sh in sh_b.items():
        s = _sig(sh, n_perm)
        for band in range(bands):
            buckets.setdefault((band, s[band * rows:(band + 1) * rows]), []).append(cid)
    out = []
    for cid, sh in sh_a.items():
        s = _sig(sh, n_perm)
        cands = set()
        for band in range(bands):
            cands.update(buckets.get((band, s[band * rows:(band + 1) * rows]), []))
        for c in cands:
            j = len(sh & sh_b[c]) / max(1, len(sh | sh_b[c]))
            if j >= threshold:
                out.append((cid, c, j))
    return out
