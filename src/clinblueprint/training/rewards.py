"""Verifiable rewards for optional GRPO on the drafter (run only if SFT+DPO trails
same-size baselines by more than 3 points; docs/DECISION_GATES.md).

No posterior-coherence, deferral or escalation rewards: those belong to
DEFER-Dx / clinical-coherence-rl and are out of scope.
"""

from __future__ import annotations

from typing import Dict, Optional

from clinblueprint.checks.hard_checks import check_graph
from clinblueprint.eval.accuracy import string_judge
from clinblueprint.readout.hard import HardReadout
from clinblueprint.schema.graph import ReasoningGraph, normalize_label, split_dx_key
from clinblueprint.schema.serialize import GraphParseError, parse_graph_json

DEFAULT_WEIGHTS = {"parse": 0.2, "grounding": 0.3, "kg_valid": 0.1, "graph_f1": 0.2, "correct": 1.0}


def graph_f1(pred: ReasoningGraph, gold: ReasoningGraph) -> float:
    judge = string_judge(0.5)
    P = [o.text for o in pred.observations.values()]
    T = [o.text for o in gold.observations.values()]
    if not P or not T:
        return 0.0
    tp = sum(any(judge(p, t) for t in T) for p in P)
    prec, rec = tp / len(P), sum(any(judge(t, p) for p in P) for t in T) / len(T)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def drafter_reward(completion: str, *, case_id: str, note: str, gold_dx: str, kg=None,
                   gold_graph: Optional[ReasoningGraph] = None, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    try:
        G = parse_graph_json(completion, case_id)
    except GraphParseError:
        return {"total": 0.0, "parse": 0.0}
    G2, rep = check_graph(G, note, kg)
    kg_err = rep.codes().get("INVALID_KG_NODE", 0)
    top = HardReadout(kg)(G2).argmax
    correct = float(top is not None and normalize_label(split_dx_key(top)[1]) == normalize_label(split_dx_key(gold_dx)[1]))
    parts = {
        "parse": 1.0,
        "grounding": rep.grounding_rate,
        "kg_valid": 1.0 if kg_err == 0 else 0.0,
        "graph_f1": graph_f1(G2, gold_graph) if gold_graph is not None else 0.0,
        "correct": correct,
    }
    parts["total"] = sum(w[k] * v for k, v in parts.items() if k in w)
    return parts
