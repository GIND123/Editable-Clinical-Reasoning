"""Cluster (case/patient-level) bootstrap confidence intervals.

Edits are nested in cases, so resampling must be over cases. Paired
differences between two systems on the same edits use the same resampled
clusters (``paired_cluster_bootstrap``). Claim A's test is "CI of the paired
difference excludes 0 at each site".
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np


def _groups(clusters: Sequence) -> Dict:
    g: Dict = {}
    for i, c in enumerate(clusters):
        g.setdefault(c, []).append(i)
    return g


def cluster_bootstrap(values: Sequence[float], clusters: Sequence, stat: Callable = np.mean,
                      n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    groups = list(_groups(clusters).values())
    rng = np.random.default_rng(seed)
    est = float(stat(v))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        boots[b] = stat(v[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def paired_cluster_bootstrap(a: Sequence[float], b: Sequence[float], clusters: Sequence,
                             n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    return cluster_bootstrap(a - b, clusters, np.mean, n_boot, alpha, seed)
