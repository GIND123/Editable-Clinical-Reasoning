"""Learn-then-Test (Angelopoulos, Bates, Candès, Jordan & Lei 2021) for choosing the
risk threshold τ of the node-repair gate.

Guarantee: with probability >= 1-δ over the calibration set, the selected τ
controls the expected risk (e.g. rate of *unrepaired wrong nodes* that change
the top-1 diagnosis) at level α. Fixed-sequence testing over τ ordered from
most conservative (repair everything) to least.
"""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

import numpy as np
from scipy.stats import binom


def hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: risk > alpha (losses in [0, 1])."""
    if n == 0:
        return 1.0
    r = min(risk_hat, alpha)

    def h1(a, b):
        a = min(max(a, 1e-12), 1 - 1e-12)
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))

    hoeff = math.exp(-n * h1(r, alpha)) if r < alpha else 1.0
    bentkus = math.e * binom.cdf(math.ceil(n * risk_hat), n, alpha)
    return float(min(1.0, hoeff, bentkus))


def learn_then_test(lambdas: Sequence[float], loss_fn: Callable[[float], np.ndarray], alpha: float,
                    delta: float) -> List[float]:
    """Return the λ values certified by fixed-sequence testing (in the given order)."""
    valid: List[float] = []
    for lam in lambdas:
        losses = np.asarray(loss_fn(lam), dtype=float)
        p = hb_pvalue(float(losses.mean()) if losses.size else 1.0, losses.size, alpha)
        if p <= delta:
            valid.append(lam)
        else:
            break
    return valid
