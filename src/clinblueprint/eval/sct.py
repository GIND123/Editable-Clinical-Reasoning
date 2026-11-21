"""Script Concordance Test scoring (Claim A: graph-mediated belief updating).

SCT item: "If you were thinking of <hypothesis> and then you find <new
information>, this hypothesis becomes: −2 much less likely … +2 much more
likely". Expert-panel *aggregate* scoring gives credit
    credit(answer) = n_panel(answer) / n_panel(modal answer).

Mapping a model to the Likert scale: for graph-mediated updates we take the
change in log-odds of the hypothesis after the finding is inserted, and map it
to {−2..+2} with symmetric thresholds (t1, t2) fitted on a *dev* split (never
on test). ER-Reason metrics are reproduced: DxUpdate (linear-weighted κ per
timestep, averaged), direction accuracy, DxTrajectory (Spearman ρ of
rankings).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LIKERT = (-2, -1, 0, 1, 2)


def delta_to_likert(delta: float, t1: float, t2: float) -> int:
    a = abs(delta)
    if a < t1:
        return 0
    s = 1 if delta > 0 else -1
    return s * (2 if a >= t2 else 1)


def weighted_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = LIKERT,
                   weights: str = "linear") -> float:
    a, b = list(a), list(b)
    if not a:
        return float("nan")
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    W = np.abs(i - j) / (k - 1) if weights == "linear" else ((i - j) / (k - 1)) ** 2
    denom = (W * E).sum()
    return float(1.0 - (W * O).sum() / denom) if denom > 0 else 1.0


def fit_likert_thresholds(deltas: Sequence[float], gold: Sequence[int],
                          grid: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """Choose (t1 < t2) maximising linear-weighted κ on a dev split."""
    d = np.asarray(deltas, dtype=float)
    if grid is None:
        qs = np.quantile(np.abs(d), np.linspace(0.05, 0.95, 19)) if d.size else np.array([0.1, 1.0])
        grid = np.unique(np.concatenate([qs, [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]]))
    best, best_t = -np.inf, (0.25, 1.0)
    for t1 in grid:
        for t2 in grid:
            if t2 <= t1:
                continue
            k = weighted_kappa([delta_to_likert(x, t1, t2) for x in d], gold)
            if k > best:
                best, best_t = k, (float(t1), float(t2))
    return best_t


def aggregate_credit(answer: int, panel_counts: Dict[int, int]) -> float:
    modal = max(panel_counts.values()) if panel_counts else 0
    return (panel_counts.get(answer, 0) / modal) if modal else 0.0


def sct_score(answers: Sequence[int], panels: Sequence[Dict[int, int]]) -> float:
    return float(np.mean([aggregate_credit(a, p) for a, p in zip(answers, panels)])) if answers else float("nan")


def direction_accuracy(pred: Sequence[int], gold: Sequence[int]) -> float:
    return float(np.mean([np.sign(p) == np.sign(g) for p, g in zip(pred, gold)])) if pred else float("nan")


def dxupdate_kappa(rows: Iterable[Tuple[int, int, int]]) -> Dict[str, float]:
    """ER-Reason DxUpdate: rows of (timestep, gold, pred). Mean per-timestep κ (n>=2)."""
    by: Dict[int, List[Tuple[int, int]]] = {}
    for t, g, p in rows:
        by.setdefault(t, []).append((g, p))
    ks = {t: weighted_kappa([g for g, _ in v], [p for _, p in v]) for t, v in by.items() if len(v) >= 2}
    vals = [k for k in ks.values() if not np.isnan(k)]
    allg = [g for v in by.values() for g, _ in v]
    allp = [p for v in by.values() for _, p in v]
    return {
        "mean_kappa": float(np.mean(vals)) if vals else float("nan"),
        "std_kappa": float(np.std(vals)) if vals else float("nan"),
        "exact_match": float(np.mean([g == p for g, p in zip(allg, allp)])) if allg else float("nan"),
        "direction": direction_accuracy(allp, allg),
        "per_timestep": {int(t): float(k) for t, k in ks.items()},
    }


def spearman_rankings(pred_ranked: Sequence[str], gold_ranked: Sequence[str]) -> Optional[float]:
    """DxTrajectory: Spearman ρ over diagnoses present in both rankings."""
    from scipy.stats import spearmanr

    m1 = {d.lower().strip(): i for i, d in enumerate(pred_ranked)}
    m2 = {d.lower().strip(): i for i, d in enumerate(gold_ranked)}
    common = sorted(set(m1) & set(m2))
    if len(common) < 2:
        return None
    r1 = [m1[c] for c in common]
    r2 = [m2[c] for c in common]
    if len(set(r1)) < 2 or len(set(r2)) < 2:
        return None
    return float(spearmanr(r1, r2).correlation)
