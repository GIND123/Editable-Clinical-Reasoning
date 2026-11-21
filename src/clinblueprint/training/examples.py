"""Build SFT / DPO text examples from records + gold/silver graphs.

Verbaliser targets are *template renderings* of the graph, so they are faithful
by construction. The verbaliser learns to explain the graph, not to bring
in knowledge from elsewhere. Reasoning recall against clinician rationales
(MedCaseReasoning) is evaluated, not trained on directly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from clinblueprint.data.records import CaseRecord
from clinblueprint.models import prompts as P
from clinblueprint.models.drafter import kg_block, training_categories
from clinblueprint.readout.differential import Differential
from clinblueprint.schema.graph import Polarity, ReasoningGraph, Relation, natural_id_key, split_dx_key
from clinblueprint.schema.serialize import graph_to_json_dict, serialize_graph


@dataclass
class TextExample:
    messages: List[Dict[str, str]]
    completion: str
    kind: str
    case_id: str


def drafter_example(rec: CaseRecord, G: ReasoningGraph, kg, rng: random.Random, n_distractors: int = 2,
                    use_premise: bool = True) -> TextExample:
    if kg is not None and rec.category in getattr(kg, "categories", []):
        cats = training_categories(kg, rec.category, n_distractors, rng)
        kgt = kg_block(kg, cats, use_premise)
    else:
        kgt = "(open-ended: name diagnoses freely, category 'open')"
    return TextExample(P.drafter_messages(rec.note, kgt), json.dumps(graph_to_json_dict(G)), "drafter", rec.case_id)


def render_explanation(G: ReasoningGraph, ddx: Differential, k: int = 3) -> str:
    top = ddx.top(k)
    if not top:
        return "No diagnosis is supported by the current reasoning graph."
    lines = [f"Most likely: {split_dx_key(top[0][0])[1]} ({top[0][1]:.0%})."]
    sup, ref = [], []
    for r in sorted(G.rationales.values(), key=lambda n: natural_id_key(n.id)):
        d = G.diagnoses.get(r.target)
        if d is None:
            continue
        prem = ", ".join(
            f"{G.observations[p].text}{'' if G.observations[p].polarity == Polarity.PRESENT else ' (absent)'} [{p}]"
            for p in r.premises if p in G.observations)
        item = f"- {prem} -> {d.leaf}: {r.claim} [{r.id}]"
        (sup if r.relation == Relation.SUPPORTS else ref).append(item)
    if sup:
        lines += ["Supporting evidence:"] + sup
    if ref:
        lines += ["Evidence against:"] + ref
    if len(top) > 1:
        lines.append("Alternatives: " + "; ".join(f"{split_dx_key(kk)[1]} ({p:.0%})" for kk, p in top[1:]))
    for m in sorted(G.must_not_miss.values(), key=lambda n: natural_id_key(n.id)):
        lines.append(f"Must-not-miss: {m.dx} - {m.status.value.replace('_', ' ')} [{m.id}]")
    return "\n".join(lines)


def verbalizer_example(G: ReasoningGraph, ddx: Differential) -> TextExample:
    from clinblueprint.models.verbalizer import ddx_text

    msgs = P.verbalizer_messages(serialize_graph(G).text, ddx_text(ddx))
    return TextExample(msgs, render_explanation(G, ddx), "verbalizer", G.case_id)


def tokenize_example(tok, ex: TextExample, max_len: int) -> Optional[dict]:
    from clinblueprint.models.backbone import chat_prompt

    prompt = chat_prompt(tok, ex.messages, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(ex.completion + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        keep = max_len - len(c_ids)
        if keep < 256:
            return None  # completion alone too long: drop rather than truncate the target
        p_ids = p_ids[:keep]
    ids = p_ids + c_ids
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + c_ids, "attention_mask": [1] * len(ids)}


def pad_collate(batch: List[dict], pad_id: int):
    import torch

    L = max(len(b["input_ids"]) for b in batch)
    out = {k: torch.full((len(batch), L), v, dtype=torch.long)
           for k, v in (("input_ids", pad_id), ("labels", -100), ("attention_mask", 0))}
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        for k in out:
            out[k][i, :n] = torch.tensor(b[k])
    return out
