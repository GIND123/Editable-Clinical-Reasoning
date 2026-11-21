"""Three-path decomposition of how diagnosis information reaches the answer.

For an in-context model reading [note][graph] -> answer, compare log p(gold):

  full        note + graph, normal attention
  knockout    note + graph, answer positions cannot attend note tokens
              (information must flow through graph positions)
  graph_only  graph text alone (information carried by the graph's *text*)
  note_only   note alone (no graph; the CoT-free baseline)

  text path            = graph_only
  hidden-state surplus = knockout - graph_only   (note information smuggled into
                                                  graph-token hidden states beyond their text)
  direct note path     = full - knockout

A causally mediated model has a small direct path and a small surplus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from clinblueprint.mech.knockout import masked_candidate_logprobs
from clinblueprint.mech.segments import segment_prompt
from clinblueprint.models.backbone import chat_prompt
from clinblueprint.models.scoring import continuation_logprobs

ANSWER_PREFIX = "Final diagnosis:"


def _msgs(user: str):
    return [{"role": "user", "content": user}]


def _norm(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max()
    p = np.exp(z)
    return np.log(p / p.sum())


@dataclass
class ThreePathResult:
    logp: Dict[str, float]           # condition -> log p(gold) (normalised over candidates)
    top: Dict[str, str]
    shares: Dict[str, float]


def three_path(model, tok, note: str, graph_text: str, candidates: Sequence[str], gold_idx: int) -> ThreePathResult:
    def full_prompt(n: str, g: str) -> str:
        body = (f"Clinical note:\n{n}\n\n" if n else "") + (f"Reasoning graph:\n{g}\n\n" if g else "")
        return chat_prompt(tok, _msgs(body + "Give the final diagnosis.")) + ANSWER_PREFIX

    conts = [" " + c for c in candidates]
    res: Dict[str, np.ndarray] = {}
    res["full"] = _norm(continuation_logprobs(model, tok, full_prompt(note, graph_text), conts))
    res["graph_only"] = _norm(continuation_logprobs(model, tok, full_prompt("", graph_text), conts))
    res["note_only"] = _norm(continuation_logprobs(model, tok, full_prompt(note, ""), conts))
    p = full_prompt(note, graph_text)
    seg = segment_prompt(tok, p, [("note", note), ("graph", graph_text)])
    cand_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in conts]
    res["knockout"] = _norm(masked_candidate_logprobs(model, seg.input_ids, cand_ids, seg.segments["note"],
                                                      block_from=seg.last))
    logp = {k: float(v[gold_idx]) for k, v in res.items()}
    top = {k: candidates[int(np.argmax(v))] for k, v in res.items()}
    pf, pk, pg = (np.exp(logp[k]) for k in ("full", "knockout", "graph_only"))
    tot = max(pf, 1e-9)
    shares = {"text_path": pg / tot, "hidden_surplus": (pk - pg) / tot, "direct_note": (pf - pk) / tot}
    return ThreePathResult(logp, top, shares)


def summarize(results: List[ThreePathResult]) -> Dict[str, float]:
    keys = results[0].shares.keys() if results else []
    return {k: float(np.mean([r.shares[k] for r in results])) for k in keys}
