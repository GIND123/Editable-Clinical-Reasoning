"""Retarget a benchmark edit (written against the gold graph) onto a model's own graph.

Edits that target nodes are aligned by text: the model graph's observation (or
rationale) most similar to ``edit.target_text``. If no node matches, the model
did not use that evidence, and the edit is *not applicable* to it. Coverage is
reported per system so that "followed more edits" can never come from silently
skipping hard ones.
"""

from __future__ import annotations

from typing import Optional

from clinblueprint.edits.types import Edit, EditOp
from clinblueprint.eval.accuracy import token_f1
from clinblueprint.schema.graph import ReasoningGraph

try:
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None


def _sim(a: str, b: str) -> float:
    s = token_f1(a, b)
    if _fuzz is not None:
        s = max(s, _fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0 * 0.9)
    return s


def retarget(edit: Edit, G: ReasoningGraph, min_score: float = 0.5) -> Optional[Edit]:
    if edit.op == EditOp.INSERT:
        return edit
    tid, text = edit.target_node, edit.target_text
    if tid in G:
        node = G.node(tid)
        node_text = getattr(node, "text", None) or getattr(node, "claim", "")
        if text is None or _sim(text, node_text) >= min_score:
            return edit
    if not text:
        return None
    pool = list(G.observations.values()) if (tid is None or tid.startswith("O")) else list(G.rationales.values())
    best, best_s = None, 0.0
    for n in pool:
        s = _sim(text, getattr(n, "text", None) or getattr(n, "claim", ""))
        if s > best_s:
            best, best_s = n, s
    if best is None or best_s < min_score:
        return None
    return edit.model_copy(update={"target_node": best.id, "meta": {**edit.meta, "retarget_score": best_s}})
