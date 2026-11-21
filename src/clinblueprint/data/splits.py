"""Patient-level, category-stratified K-fold splits (DiReCT 5-fold CV).

No patient appears in more than one fold. Category balance is maintained by
greedy assignment of patient groups (largest first) to the fold that is
currently most under-represented for that group's majority category.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from clinblueprint.data.records import CaseRecord


def grouped_stratified_kfold(records: Sequence[CaseRecord], k: int = 5, seed: int = 0) -> Dict[str, int]:
    """Return {case_id: fold}."""
    groups: Dict[str, List[CaseRecord]] = defaultdict(list)
    for r in records:
        groups[r.patient_id or r.case_id].append(r)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -len(kv[1]))
    fold_cat = [Counter() for _ in range(k)]
    fold_size = [0] * k
    assign: Dict[str, int] = {}
    for _, recs in items:
        cat = Counter(r.category or "none" for r in recs).most_common(1)[0][0]
        f = min(range(k), key=lambda i: (fold_cat[i][cat], fold_size[i], rng.random()))
        for r in recs:
            assign[r.case_id] = f
            fold_cat[f][r.category or "none"] += 1
        fold_size[f] += len(recs)
    return assign


def fold_split(records: Sequence[CaseRecord], assign: Dict[str, int], fold: int, dev_fold_offset: int = 1):
    """(train, dev, test) for ``fold``; dev = the next fold (used for thresholds / early stopping)."""
    k = max(assign.values()) + 1
    dev = (fold + dev_fold_offset) % k
    tr = [r for r in records if assign[r.case_id] not in (fold, dev)]
    dv = [r for r in records if assign[r.case_id] == dev]
    te = [r for r in records if assign[r.case_id] == fold]
    return tr, dv, te
