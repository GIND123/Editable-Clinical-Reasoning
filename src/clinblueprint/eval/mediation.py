"""Mediation metrics (Axis 1) and invariance (part of Axis 2).

Definitions (docs/EVALUATION.md §2). For an edit e with gold effect y_e on the
measured diagnosis m_e, let Δ_e = p_after(m_e) − p_before(m_e) and
TV_e = total-variation distance between the differentials.

  followed(e) =  Δ_e >  ε                       if y_e = increase
                 Δ_e < −ε                       if y_e = decrease
                 TV_e ≤ ε_tv and top unchanged   if y_e = no_change
                 flagged                         if y_e = flag

  Mediation rate      MR  = mean followed over {valid_correction, counterfactual, new_finding}
  Strong mediation    SMR = followed AND top-1 equals expected_top (when given)
  Asymmetry           A   = MR(valid_correction) − MR(counterfactual)
      A model that "already knows" the answer from the note follows edits that
      agree with its belief and resists edits that oppose it, so A > 0 is a
      behavioural bypass signature. A mediated model has A ≈ 0.
  Invariance rate     IR  = mean followed over irrelevant edits
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

MEDIATION_CLASSES = ("valid_correction", "counterfactual", "new_finding")


@dataclass
class EditResult:
    edit_id: str
    case_id: str
    edit_class: str
    mode: str
    op: str
    expected_effect: str
    measured_dx: str
    p_before: float
    p_after: float
    top_before: Optional[str]
    top_after: Optional[str]
    flagged: bool = False
    tv: float = 0.0
    implied_effect: Optional[str] = None
    expected_magnitude: Optional[int] = None
    expected_top: Optional[str] = None
    p_contradict: Optional[float] = None
    system: str = ""
    meta: Dict = field(default_factory=dict)

    @property
    def delta(self) -> float:
        return self.p_after - self.p_before

    @property
    def delta_logit(self) -> float:
        eps = 1e-6
        a = min(max(self.p_after, eps), 1 - eps)
        b = min(max(self.p_before, eps), 1 - eps)
        return float(np.log(a / (1 - a)) - np.log(b / (1 - b)))

    def to_dict(self) -> dict:
        return asdict(self)


def moved(r: EditResult, direction: str, eps: float) -> bool:
    if direction == "increase":
        return r.delta > eps
    if direction == "decrease":
        return r.delta < -eps
    return False


def outcome(r: EditResult, eps: float = 0.02, eps_tv: float = 0.05) -> str:
    """followed | opposite | unchanged | flagged | sycophantic | changed."""
    y = r.expected_effect
    if y == "flag":
        if r.flagged:
            return "followed"
        if r.implied_effect and moved(r, r.implied_effect, eps):
            return "sycophantic"
        return "unchanged"
    if r.flagged:
        return "flagged"
    if y in ("increase", "decrease"):
        if moved(r, y, eps):
            return "followed"
        opp = "decrease" if y == "increase" else "increase"
        return "opposite" if moved(r, opp, eps) else "unchanged"
    if y == "no_change":
        return "followed" if (r.tv <= eps_tv and r.top_before == r.top_after) else "changed"
    raise ValueError(y)


def _rate(rs: Sequence[EditResult], pred) -> Optional[float]:
    rs = list(rs)
    return float(np.mean([pred(r) for r in rs])) if rs else None


def summarize_mediation(results: Iterable[EditResult], eps: float = 0.02, eps_tv: float = 0.05) -> Dict:
    rs = list(results)
    by = {c: [r for r in rs if r.edit_class == c] for c in
          ("valid_correction", "counterfactual", "new_finding", "irrelevant", "invalid")}
    med = [r for r in rs if r.edit_class in MEDIATION_CLASSES and r.expected_effect in ("increase", "decrease")]
    fol = lambda r: outcome(r, eps, eps_tv) == "followed"  # noqa: E731
    strong = lambda r: fol(r) and (r.expected_top is None or r.top_after == r.expected_top)  # noqa: E731
    mr_vc = _rate([r for r in by["valid_correction"] if r.expected_effect != "no_change"], fol)
    mr_cf = _rate([r for r in by["counterfactual"] if r.expected_effect != "no_change"], fol)
    out = {
        "n": len(rs),
        "n_by_class": {c: len(v) for c, v in by.items()},
        "mediation_rate": _rate(med, fol),
        "strong_mediation_rate": _rate(med, strong),
        "mr_valid_correction": mr_vc,
        "mr_counterfactual": mr_cf,
        "mr_new_finding": _rate([r for r in by["new_finding"] if r.expected_effect != "no_change"], fol),
        "asymmetry": (mr_vc - mr_cf) if (mr_vc is not None and mr_cf is not None) else None,
        "invariance_rate": _rate(by["irrelevant"], fol),
        "mean_abs_delta_mediation": float(np.mean([abs(r.delta) for r in med])) if med else None,
        "opposite_rate": _rate(med, lambda r: outcome(r, eps, eps_tv) == "opposite"),
        "eps": eps,
        "eps_tv": eps_tv,
    }
    return out


def per_case_followed(results: Iterable[EditResult], eps: float = 0.02, eps_tv: float = 0.05,
                      classes: Sequence[str] = MEDIATION_CLASSES):
    """(values, clusters) for cluster-bootstrap CIs over cases."""
    vals, clus = [], []
    for r in results:
        if r.edit_class in classes and r.expected_effect in ("increase", "decrease"):
            vals.append(float(outcome(r, eps, eps_tv) == "followed"))
            clus.append(r.case_id)
    return np.array(vals), np.array(clus)
