"""Node risk scoring and the gate used by ClinBlueprint.run (repair_budget > 0).

Features per node (cheap, graph-local): grounding match score, source, rationale
strength, number of premises, verifier contradiction probability (optional).
A logistic model is fitted on dev graphs with labels "node is wrong vs gold"
(unmatched observation / rationale with a wrong target), then τ is chosen by LTT.
Kept only if it improves the accuracy-vs-tokens front (decision gate in docs).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from clinblueprint.schema.graph import EvidenceSource, ReasoningGraph


def node_features(G: ReasoningGraph, nid: str) -> np.ndarray:
    t = G.node_type(nid).value
    if t == "observation":
        o = G.observations[nid]
        gs = (o.span.score if o.span else 0.0) / 100.0
        return np.array([1, 0, gs, float(o.source == EvidenceSource.NOTE), 0.0, len(G.rationales_using(nid)), 1.0])
    if t == "rationale":
        r = G.rationales[nid]
        return np.array([0, 1, 1.0, 1.0, r.strength, len(r.premises), 1.0])
    return np.array([0, 0, 1.0, 1.0, 1.0, 0, 1.0])


class NodeRiskGate:
    def __init__(self, weights: Optional[np.ndarray] = None, threshold: float = 0.5,
                 extra: Optional[Callable[[ReasoningGraph, str, str], float]] = None):
        self.w = weights if weights is not None else np.array([0.5, 0.5, -3.0, -1.0, -1.0, 0.0, 1.0])
        self.threshold = threshold
        self.extra = extra

    def risk(self, G: ReasoningGraph, nid: str, note: str) -> float:
        z = float(node_features(G, nid) @ self.w)
        if self.extra is not None:
            z += self.extra(G, nid, note)
        return 1.0 / (1.0 + np.exp(-z))

    def select(self, G: ReasoningGraph, note: str) -> List[str]:
        ids = list(G.observations) + list(G.rationales)
        return [n for n in ids if self.risk(G, n, note) > self.threshold]

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1.0 / l2, fit_intercept=False, max_iter=2000).fit(X, y)
        self.w = clf.coef_[0]
