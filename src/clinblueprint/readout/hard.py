"""Hard bottleneck readout: the differential is a deterministic function of G.

For a candidate diagnosis label L with KG path  L -> parent -> ... -> root::

    score(L) = log_prior(L) + sum_k  decay^k * E(path_k)
    E(d)     = w_s * sum_{supporting r -> d} strength(r) * g(r)
             - w_r * sum_{refuting  r -> d} strength(r) * g(r)
    g(r)     = product over premises of {1 if valid, uncertain_factor if the
               observation is UNCERTAIN, 0 if invalid}
    p        = softmax(score / T) over the candidate set

Properties (unit-tested in tests/test_hard_readout.py):
  * Mediation is structural: the note never enters the readout.
  * Monotone: adding a valid supporting rationale for a node on L's path never
    decreases p(L); removing/negating a premise of such a rationale never
    increases it.
  * Locality: evidence for nodes off L's path changes p(L) only through the
    softmax normalisation.

Candidate-selection leakage: if candidates were only the drafter's own
diagnoses, the drafter could bypass the evidence by choosing the candidate set.
``candidate_mode="kg_expand"`` (default) scores every KG leaf of every category
the graph touches, so an edit can promote a diagnosis the drafter never named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from clinblueprint.readout.differential import Differential
from clinblueprint.schema.graph import (
    OPEN_CATEGORY,
    Polarity,
    ReasoningGraph,
    Relation,
    dx_key,
    normalize_label,
    split_dx_key,
)


@dataclass
class ReadoutConfig:
    w_support: float = 1.0
    w_refute: float = 1.0
    ancestor_decay: float = 0.5
    temperature: float = 1.0
    uncertain_factor: float = 0.5
    candidate_mode: str = "kg_expand"  # drafted | kg_expand | category_leaves | fixed
    use_drafter_p: bool = False        # ABLATION ONLY: opens a bypass channel
    drafter_p_weight: float = 0.0
    log_prior: Dict[str, float] = field(default_factory=dict)  # case-independent base rates
    unknown_prior: float = 0.0


class HardReadout:
    def __init__(self, kg=None, cfg: Optional[ReadoutConfig] = None):
        self.kg = kg
        self.cfg = cfg or ReadoutConfig()

    # --------------------------------------------------------------- evidence
    def rationale_factor(self, G: ReasoningGraph, rid: str, _stack: Optional[Set[str]] = None) -> float:
        r = G.rationales.get(rid)
        if r is None:
            return 0.0
        stack = set() if _stack is None else _stack
        if rid in stack:
            return 0.0  # cycle
        stack = stack | {rid}
        g = 1.0
        for p in r.premises:
            if p in G.observations:
                o = G.observations[p]
                want = r.expected_polarity(p)
                if o.polarity == want:
                    continue
                if o.polarity == Polarity.UNCERTAIN or want == Polarity.UNCERTAIN:
                    g *= self.cfg.uncertain_factor
                else:
                    return 0.0
            elif p in G.rationales:
                g *= self.rationale_factor(G, p, stack)
            else:
                return 0.0
            if g == 0.0:
                return 0.0
        return g

    def support_refute(self, G: ReasoningGraph) -> Tuple[Dict[str, float], Dict[str, float]]:
        S = {d: 0.0 for d in G.diagnoses}
        F = {d: 0.0 for d in G.diagnoses}
        for r in G.rationales.values():
            if r.target not in G.diagnoses:
                continue
            v = r.strength * self.rationale_factor(G, r.id)
            if r.relation == Relation.SUPPORTS:
                S[r.target] += v
            else:
                F[r.target] += v
        return S, F

    def node_evidence(self, G: ReasoningGraph) -> Dict[str, float]:
        S, F = self.support_refute(G)
        return {d: self.cfg.w_support * S[d] - self.cfg.w_refute * F[d] for d in G.diagnoses}

    # -------------------------------------------------------------- candidates
    def _is_kg_cat(self, cat: str) -> bool:
        return self.kg is not None and cat in self.kg.categories

    def candidates(self, G: ReasoningGraph, fixed: Optional[Sequence[str]] = None) -> List[str]:
        mode = "fixed" if fixed is not None else self.cfg.candidate_mode
        if mode == "fixed":
            return [k if "::" in k else dx_key(OPEN_CATEGORY, k) for k in (fixed or [])]
        drafted = [G.diagnoses[d].key for d in G.terminal_diagnoses()]
        out: List[str] = []
        if mode == "drafted" or self.kg is None:
            out = drafted
        else:
            cats = sorted({d.category for d in G.diagnoses.values() if self._is_kg_cat(d.category)})
            for c in cats:
                out.extend(dx_key(c, leaf) for leaf in self.kg.leaves(c))
            # non-KG (open) drafted diagnoses are always candidates
            out.extend(k for k in drafted if not self._is_kg_cat(split_dx_key(k)[0]))
            if mode == "kg_expand":
                out.extend(k for k in drafted if k not in out)
        seen: Set[str] = set()
        return [k for k in out if not (k in seen or seen.add(k))]

    # ------------------------------------------------------------------ path
    def _graph_matches(self, G: ReasoningGraph, cat: str, leaf: str) -> List[str]:
        n = normalize_label(leaf)
        return [
            d.id for d in G.diagnoses.values()
            if normalize_label(d.leaf) == n and (cat == OPEN_CATEGORY or d.category in (cat, OPEN_CATEGORY))
        ]

    def path(self, G: ReasoningGraph, key: str) -> List[Tuple[List[str], int]]:
        """[(graph dx ids matching the path node, depth)] from the candidate up to the root."""
        cat, leaf = split_dx_key(key)
        out = [(self._graph_matches(G, cat, leaf), 0)]
        if self._is_kg_cat(cat):
            for k, anc in enumerate(self.kg.ancestors(cat, leaf), start=1):
                out.append((self._graph_matches(G, cat, anc), k))
        else:
            seen: Set[str] = set()
            frontier = out[0][0]
            depth = 0
            while frontier:
                depth += 1
                nxt = []
                for d in frontier:
                    for a in G.dx_ancestors(d)[:1]:
                        if a not in seen:
                            seen.add(a)
                            nxt.append(a)
                if nxt:
                    out.append((nxt, depth))
                frontier = nxt
        return out

    # ---------------------------------------------------------------- scoring
    def scores(self, G: ReasoningGraph, fixed: Optional[Sequence[str]] = None) -> Dict[str, float]:
        E = self.node_evidence(G)
        cfg = self.cfg
        out: Dict[str, float] = {}
        for key in self.candidates(G, fixed):
            s = cfg.log_prior.get(key, cfg.unknown_prior)
            for ids, depth in self.path(G, key):
                s += (cfg.ancestor_decay ** depth) * sum(E.get(i, 0.0) for i in ids)
            if cfg.use_drafter_p and cfg.drafter_p_weight:
                cat, leaf = split_dx_key(key)
                ps = [G.diagnoses[i].p for i in self._graph_matches(G, cat, leaf) if G.diagnoses[i].p]
                if ps:
                    s += cfg.drafter_p_weight * float(np.log(max(ps)))
            out[key] = s
        return out

    def __call__(self, G: ReasoningGraph, fixed: Optional[Sequence[str]] = None) -> Differential:
        return Differential.from_scores(self.scores(G, fixed), self.cfg.temperature)

    def explain(self, G: ReasoningGraph, key: str) -> List[Tuple[str, float]]:
        """Per-rationale contribution to score(key): [(rationale id, contribution)]."""
        contrib: List[Tuple[str, float]] = []
        for ids, depth in self.path(G, key):
            w = self.cfg.ancestor_decay ** depth
            for d in ids:
                for r in G.rationales_for(d):
                    sign = self.cfg.w_support if r.relation == Relation.SUPPORTS else -self.cfg.w_refute
                    contrib.append((r.id, w * sign * r.strength * self.rationale_factor(G, r.id)))
        return sorted(contrib, key=lambda t: -abs(t[1]))
