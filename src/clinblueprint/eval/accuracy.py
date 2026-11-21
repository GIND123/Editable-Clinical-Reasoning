"""Diagnostic accuracy and reasoning-quality metrics.

* ``dx_match``: exact / hierarchical (KG ancestor) / fuzzy label matching.
* ``direct_metrics``: re-implementation of DiReCT's statistics.py formulas
  (completeness precision/recall/coverage, faithfulness ob / "all"). The
  similarity judge is pluggable. DiReCT used a Llama-3-8B judge; we default
  to a string judge and offer a *local* LLM judge.
* ``reasoning_recall``: MedCaseReasoning-style recall of clinician reasoning
  statements.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from clinblueprint.schema.graph import ReasoningGraph, normalize_label, split_dx_key

try:
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None

Judge = Callable[[str, str], bool]


def token_f1(a: str, b: str) -> float:
    ta = re.findall(r"[a-z0-9]+", a.lower())
    tb = re.findall(r"[a-z0-9]+", b.lower())
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for t in ta:
        if t in pool:
            pool.remove(t)
            common += 1
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def string_judge(threshold: float = 0.6) -> Judge:
    def judge(a: str, b: str) -> bool:
        if normalize_label(a) == normalize_label(b):
            return True
        if _fuzz is not None and _fuzz.token_set_ratio(a.lower(), b.lower()) >= 90:
            return True
        return token_f1(a, b) >= threshold
    return judge


def dx_match(pred: str, gold: str, *, kg=None, mode: str = "exact", judge: Optional[Judge] = None) -> bool:
    pc, pl = split_dx_key(pred)
    gc, gl = split_dx_key(gold)
    if normalize_label(pl) == normalize_label(gl):
        return True
    if mode == "hierarchical" and kg is not None and gc in kg.categories:
        # predicted a more specific node under the gold node -> correct at gold granularity
        return any(normalize_label(a) == normalize_label(gl) for a in kg.ancestors(gc, pl))
    if mode == "fuzzy":
        return (judge or string_judge())(pl, gl)
    return False


def graph_to_direct_triples(G: ReasoningGraph) -> Dict[str, Tuple[str, Optional[str], str]]:
    """{observation text: (rationale claim, section, diagnosis leaf)} like DiReCT's deduction_assemble."""
    out: Dict[str, Tuple[str, Optional[str], str]] = {}
    for r in G.rationales.values():
        d = G.diagnoses.get(r.target)
        if d is None:
            continue
        for p in r.premises:
            o = G.observations.get(p)
            if o is not None and o.text not in out:
                out[o.text] = (r.claim, o.section, d.leaf)
    return out


def direct_metrics(gt: Dict[str, Tuple[str, Optional[str], str]], pred: Dict[str, Tuple[str, Optional[str], str]],
                   obs_judge: Optional[Judge] = None, rat_judge: Optional[Judge] = None) -> Dict[str, float]:
    """Mirrors DiReCT statistics.statistic_one_pred (incl. its +1 denominators)."""
    obs_judge = obs_judge or string_judge(0.5)
    rat_judge = rat_judge or string_judge(0.4)
    g_obs, p_obs = list(gt), list(pred)
    used: set = set()
    pairs: List[Tuple[int, int]] = []
    for i, go in enumerate(g_obs):
        for j, po in enumerate(p_obs):
            if j in used:
                continue
            if obs_judge(go, po):
                pairs.append((i, j))
                used.add(j)
                break
    n_gt, n_pred, n_pair = len(g_obs), len(p_obs), len(pairs)
    faithful = 0
    for i, j in pairs:
        g_rat, _, g_dx = gt[g_obs[i]]
        p_rat, _, p_dx = pred[p_obs[j]]
        if normalize_label(g_dx) == normalize_label(p_dx) and rat_judge(g_rat, p_rat):
            faithful += 1
    union = n_gt + n_pred - n_pair
    return {
        "comp_pre": n_pair / (n_pred + 1),
        "comp_re": n_pair / (n_gt + 1),
        "comp_coverage": n_pair / union if union else 0.0,
        "faith_ob": faithful / n_pair if n_pair else 0.0,
        "faith_all": faithful / union if union else 0.0,
    }


def reasoning_recall(gold_statements: Sequence[str], pred_text: str, judge: Optional[Judge] = None) -> float:
    """Fraction of clinician reasoning statements covered by the model explanation."""
    if not gold_statements:
        return float("nan")
    judge = judge or (lambda g, p: token_f1(g, p) >= 0.3 or (_fuzz is not None and _fuzz.partial_ratio(g, p) >= 80))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", pred_text) if s.strip()] or [pred_text]
    return float(sum(any(judge(g, s) for s in sents) for g in gold_statements) / len(gold_statements))


def split_reasoning_statements(text: str) -> List[str]:
    """MedCaseReasoning's diagnostic_reasoning is a numbered list of statements."""
    parts = re.split(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+", text or "")
    return [p.strip() for p in parts if p and p.strip()]
