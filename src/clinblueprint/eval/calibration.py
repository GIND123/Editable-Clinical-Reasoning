"""Calibration metrics for differentials and the validator."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def ece(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.size == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins) - 1, 0, n_bins - 1)
    return float(sum(abs(c[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(n_bins) if (idx == b).any()))


def brier(prob_true_class: Sequence[float]) -> float:
    p = np.asarray(prob_true_class, dtype=float)
    return float(np.mean((1 - p) ** 2)) if p.size else float("nan")


def reliability(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> Dict[str, list]:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, bins) - 1, 0, n_bins - 1)
    out = {"bin_center": [], "accuracy": [], "confidence": [], "count": []}
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out["bin_center"].append(float((bins[b] + bins[b + 1]) / 2))
            out["accuracy"].append(float(y[m].mean()))
            out["confidence"].append(float(c[m].mean()))
            out["count"].append(int(m.sum()))
    return out
