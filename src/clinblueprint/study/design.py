"""Randomisation, power simulation and primary analysis for the crossover study.

Arms: A = editable graph, B = CoT text, C = answer only.
Design: Williams design (balanced for first-order carry-over) over clinicians;
cases are split into blocks stratified by (model correct?, specialty) so each
arm sees the same natural error rate. Cases are never shown twice to one
clinician.
Primary outcome: error detection = clinician rejects / corrects an incorrect
model answer (cases where the model is wrong). Primary analysis: GEE logistic
model, outcome ~ arm + period, exchangeable correlation within clinician,
robust SEs; contrast A vs B.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

ARMS = ("A_graph", "B_cot", "C_answer_only")
WILLIAMS_3 = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2), (2, 1, 0)]


def assign(clinicians: Sequence[str], cases: Sequence[Dict], seed: int = 0) -> List[Dict]:
    """cases: dicts with case_id, model_correct (bool), specialty. Returns schedule rows."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for c in cases:
        strata[(bool(c["model_correct"]), c.get("specialty", "any"))].append(c)
    blocks: List[List[Dict]] = [[], [], []]
    for key in sorted(strata, key=str):
        cs = strata[key][:]
        rng.shuffle(cs)
        for i, c in enumerate(cs):
            blocks[i % 3].append(c)
    seqs = WILLIAMS_3[:]
    rng.shuffle(seqs)
    rows = []
    for k, clin in enumerate(clinicians):
        seq = seqs[k % len(seqs)]
        for period, b in enumerate(range(3)):
            order = blocks[b][:]
            rng.shuffle(order)
            for pos, c in enumerate(order):
                rows.append({"clinician": clin, "period": period, "block": b, "arm": ARMS[seq[period]],
                             "case_id": c["case_id"], "position": pos, "model_correct": bool(c["model_correct"])})
    return rows


def simulate_power(n_clinicians: int, n_cases: int, model_error_rate: float, p_detect: Dict[str, float],
                   clinician_sd: float = 0.5, n_sim: int = 500, alpha: float = 0.05, seed: int = 0) -> float:
    """Power for A vs B on detection, using a paired test on per-clinician detection rates."""
    from scipy.stats import ttest_rel

    rng = np.random.default_rng(seed)
    n_err_per_arm = max(1, int(round(n_cases * model_error_rate / 3)))
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    hits = 0
    for _ in range(n_sim):
        u = rng.normal(0, clinician_sd, n_clinicians)
        ra = rng.binomial(n_err_per_arm, 1 / (1 + np.exp(-(logit(p_detect["A_graph"]) + u)))) / n_err_per_arm
        rb = rng.binomial(n_err_per_arm, 1 / (1 + np.exp(-(logit(p_detect["B_cot"]) + u)))) / n_err_per_arm
        if np.allclose(ra, rb):
            continue
        if ttest_rel(ra, rb).pvalue < alpha and ra.mean() > rb.mean():
            hits += 1
    return hits / n_sim


def analyze(rows: List[Dict], outcome: str = "detected") -> Optional[Dict[str, float]]:
    """GEE logistic regression on model-error trials. rows need clinician, arm, period, outcome."""
    try:
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return None
    df = pd.DataFrame([r for r in rows if not r.get("model_correct", False)])
    df["arm"] = pd.Categorical(df["arm"], categories=["B_cot", "A_graph", "C_answer_only"])
    m = smf.gee(f"{outcome} ~ arm + C(period)", "clinician", df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    b = m.params["arm[T.A_graph]"]
    lo, hi = m.conf_int().loc["arm[T.A_graph]"]
    return {"log_or_A_vs_B": float(b), "ci_low": float(lo), "ci_high": float(hi),
            "p": float(m.pvalues["arm[T.A_graph]"]), "n_trials": int(len(df))}
