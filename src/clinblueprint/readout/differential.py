"""A differential diagnosis = probability distribution over dx keys."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from clinblueprint.schema.graph import normalize_label, split_dx_key


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if x.size == 0:
        return x
    z = x / max(temperature, 1e-8)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Differential:
    keys: List[str] = field(default_factory=list)
    probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def from_scores(cls, scores: Dict[str, float], temperature: float = 1.0) -> "Differential":
        keys = list(scores)
        s = np.array([scores[k] for k in keys], dtype=float)
        return cls(keys=keys, probs=softmax(s, temperature), scores=s)

    @classmethod
    def from_probs(cls, probs: Dict[str, float]) -> "Differential":
        keys = list(probs)
        p = np.array([max(0.0, probs[k]) for k in keys], dtype=float)
        p = p / p.sum() if p.sum() > 0 else p
        return cls(keys=keys, probs=p, scores=np.log(np.clip(p, 1e-12, 1.0)))

    def __len__(self) -> int:
        return len(self.keys)

    def _match(self, name: str) -> List[int]:
        if name in self.keys:
            return [self.keys.index(name)]
        _, leaf = split_dx_key(name)
        n = normalize_label(leaf)
        return [i for i, k in enumerate(self.keys) if normalize_label(split_dx_key(k)[1]) == n]

    def p(self, name: str) -> float:
        idx = self._match(name)
        return float(sum(self.probs[i] for i in idx)) if idx else 0.0

    def logit(self, name: str, eps: float = 1e-6) -> float:
        p = min(max(self.p(name), eps), 1 - eps)
        return float(np.log(p / (1 - p)))

    def top(self, k: int = 1) -> List[Tuple[str, float]]:
        order = np.argsort(-self.probs)[:k]
        return [(self.keys[i], float(self.probs[i])) for i in order]

    @property
    def argmax(self) -> Optional[str]:
        return self.keys[int(np.argmax(self.probs))] if self.keys else None

    def rank_of(self, name: str) -> Optional[int]:
        idx = self._match(name)
        if not idx:
            return None
        order = list(np.argsort(-self.probs))
        return 1 + min(order.index(i) for i in idx)

    def as_dict(self) -> Dict[str, float]:
        return {k: float(p) for k, p in zip(self.keys, self.probs)}

    def tv(self, other: "Differential") -> float:
        """Total-variation distance (keys missing on one side count as 0 mass)."""
        a, b = self.as_dict(), other.as_dict()
        keys = set(a) | set(b)
        return 0.5 * float(sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))
