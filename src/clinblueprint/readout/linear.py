"""Learned *linear* hard readout with sign constraints (still structurally mediated).

Features for candidate L (all computed from G only):
  0 support at L                 weight >= 0
  1 refute  at L                 weight <= 0
  2 decayed support at ancestors weight >= 0
  3 decayed refute  at ancestors weight <= 0
  4 log1p(#distinct valid observations on L's path)  weight >= 0
  5 log base-rate prior of L (case independent)       free
  6 [ablation] L was drafted as a diagnosis node       free  <- bypass channel
  7 [ablation] log drafter self-reported p             free  <- bypass channel

Fitted as a conditional logit (softmax over each case's candidate set) with
projected gradient ascent so monotonicity, hence mediation, is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clinblueprint.readout.differential import Differential, softmax
from clinblueprint.readout.hard import HardReadout, ReadoutConfig
from clinblueprint.schema.graph import ReasoningGraph, Relation, split_dx_key

FEATURES = [
    "support_leaf", "refute_leaf", "support_anc", "refute_anc",
    "log_n_obs", "log_prior", "drafted", "log_drafter_p",
]
_SIGN = np.array([+1, -1, +1, -1, +1, 0, 0, 0])  # +1: w>=0, -1: w<=0, 0: free


@dataclass
class LinearFitReport:
    n_cases: int
    log_likelihood: float
    accuracy: float
    weights: Dict[str, float]


class LinearReadout:
    def __init__(self, kg=None, base: Optional[ReadoutConfig] = None, allow_bypass_features: bool = False,
                 temperature: float = 1.0):
        self.base = HardReadout(kg, base or ReadoutConfig())
        self.allow_bypass = allow_bypass_features
        self.temperature = temperature
        self.w = np.array([1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
        self.log_prior: Dict[str, float] = {}

    # ------------------------------------------------------------- features
    def features(self, G: ReasoningGraph, fixed: Optional[Sequence[str]] = None) -> Tuple[List[str], np.ndarray]:
        S, F = self.base.support_refute(G)
        keys = self.base.candidates(G, fixed)
        X = np.zeros((len(keys), len(FEATURES)))
        decay = self.base.cfg.ancestor_decay
        drafted = {G.diagnoses[d].key for d in G.terminal_diagnoses()}
        for i, key in enumerate(keys):
            obs_ids = set()
            for ids, depth in self.base.path(G, key):
                s = sum(S.get(d, 0.0) for d in ids)
                f = sum(F.get(d, 0.0) for d in ids)
                if depth == 0:
                    X[i, 0], X[i, 1] = s, f
                else:
                    X[i, 2] += (decay ** depth) * s
                    X[i, 3] += (decay ** depth) * f
                for d in ids:
                    for r in G.rationales_for(d, Relation.SUPPORTS):
                        if self.base.rationale_factor(G, r.id) > 0:
                            obs_ids.update(p for p in r.premises if p in G.observations)
            X[i, 4] = math.log1p(len(obs_ids))
            X[i, 5] = self.log_prior.get(key, 0.0)
            if self.allow_bypass:
                X[i, 6] = 1.0 if key in drafted else 0.0
                cat, leaf = split_dx_key(key)
                ps = [G.diagnoses[d].p for d in self.base._graph_matches(G, cat, leaf) if G.diagnoses[d].p]
                X[i, 7] = math.log(max(ps)) if ps else 0.0
        return keys, X

    def _project(self, w: np.ndarray) -> np.ndarray:
        w = w.copy()
        w[_SIGN > 0] = np.maximum(w[_SIGN > 0], 0.0)
        w[_SIGN < 0] = np.minimum(w[_SIGN < 0], 0.0)
        if not self.allow_bypass:
            w[6:] = 0.0
        return w

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        graphs: Sequence[ReasoningGraph],
        gold: Sequence[str],
        *,
        fixed: Optional[Sequence[Optional[Sequence[str]]]] = None,
        lr: float = 0.1,
        epochs: int = 300,
        l2: float = 1e-3,
        prior_counts: Optional[Dict[str, int]] = None,
        verbose: bool = False,
    ) -> LinearFitReport:
        if prior_counts:
            tot = sum(prior_counts.values())
            self.log_prior = {k: math.log((c + 1) / (tot + len(prior_counts))) for k, c in prior_counts.items()}
        data = []
        for i, (G, y) in enumerate(zip(graphs, gold)):
            keys, X = self.features(G, fixed[i] if fixed else None)
            yi = [j for j, k in enumerate(keys) if k == y or split_dx_key(k)[1].lower() == split_dx_key(y)[1].lower()]
            if not yi or len(keys) < 2:
                continue
            data.append((X, yi[0]))
        if not data:
            raise ValueError("no trainable cases (gold not in candidate set)")
        w = self._project(self.w)
        for ep in range(epochs):
            grad = -l2 * w
            ll = 0.0
            for X, y in data:
                p = softmax(X @ w, self.temperature)
                ll += math.log(max(p[y], 1e-12))
                grad += (X[y] - p @ X) / self.temperature
            w = self._project(w + lr * grad / len(data))
            if verbose and ep % 50 == 0:
                print(f"epoch {ep} ll={ll / len(data):.4f}")
        self.w = w
        acc = float(np.mean([int(np.argmax(X @ w) == y) for X, y in data]))
        ll = float(np.mean([math.log(max(softmax(X @ w, self.temperature)[y], 1e-12)) for X, y in data]))
        return LinearFitReport(len(data), ll, acc, dict(zip(FEATURES, map(float, w))))

    def __call__(self, G: ReasoningGraph, fixed: Optional[Sequence[str]] = None) -> Differential:
        keys, X = self.features(G, fixed)
        s = X @ self.w if len(keys) else np.zeros(0)
        return Differential.from_scores(dict(zip(keys, s)), self.temperature)

    def state_dict(self) -> dict:
        return {"w": self.w.tolist(), "log_prior": self.log_prior, "allow_bypass": self.allow_bypass,
                "temperature": self.temperature}

    def load_state_dict(self, d: dict) -> None:
        self.w = np.array(d["w"], dtype=float)
        self.log_prior = dict(d.get("log_prior", {}))
        self.allow_bypass = bool(d.get("allow_bypass", False))
        self.temperature = float(d.get("temperature", 1.0))
