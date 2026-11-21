"""Re-implementation sketch of the counterfactual multi-agent diagnosis baseline
(You et al., arXiv 2603.27820; training-free, model-agnostic).

Our reading of the method: perturb clinical findings in the *input case*
(negate / remove / replace / weaken / intensify / insert), measure how
each candidate diagnosis' probability shifts (the Counterfactual Probability
Gap, CPG), and aggregate the evidence into a final ranking. Run fully locally.

VERIFY against the authors' paper or released code before reporting numbers.
Any deviation must be listed in the Supplementary.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Sequence

import numpy as np

from clinblueprint.models import prompts as P
from clinblueprint.models.backbone import chat_prompt, generate

_SENT = re.compile(r"[^.\n]+[.\n]")


def _probs(model, tok, note: str, cands: Sequence[str]) -> np.ndarray:
    out = generate(model, tok, [chat_prompt(tok, [{"role": "user", "content": P.COUNTERFACTUAL_AGENT_USER.format(
        note=note, candidates=", ".join(cands))}])], max_new_tokens=300)[0]
    try:
        d = json.loads(out[out.find("{"):out.rfind("}") + 1])
        p = np.array([float(d.get(c, 0.0)) for c in cands])
    except Exception:
        p = np.ones(len(cands))
    p = np.clip(p, 1e-6, None)
    return p / p.sum()


def counterfactual_rank(model, tok, note: str, cands: Sequence[str], max_findings: int = 8) -> Dict[str, float]:
    base = _probs(model, tok, note, cands)
    sents = [m.group().strip() for m in _SENT.finditer(note) if len(m.group().strip()) > 15][:max_findings]
    support = np.zeros(len(cands))
    for s in sents:
        removed = note.replace(s, "")
        cpg = base - _probs(model, tok, removed, cands)  # >0: the finding supported that candidate
        support += cpg
    score = np.log(base) + support
    z = np.exp(score - score.max())
    return dict(zip(cands, (z / z.sum()).tolist()))
