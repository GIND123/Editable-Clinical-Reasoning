"""Verbaliser: graph -> clinician-facing explanation.

The verbaliser is conditioned on the serialised graph and the readout only.
The note is never in its context, so its text cannot carry information that
bypasses the graph. ``audit`` flags node ids that do not exist and numbers
that do not occur in the graph text (hallucination guard; reported in Ext. Data).
"""

from __future__ import annotations

import re
from typing import Dict, List

from clinblueprint.models import prompts as P
from clinblueprint.models.backbone import chat_prompt, generate
from clinblueprint.readout.differential import Differential
from clinblueprint.schema.graph import ReasoningGraph, split_dx_key
from clinblueprint.schema.serialize import serialize_graph


def ddx_text(ddx: Differential, k: int = 5) -> str:
    return "\n".join(f"{split_dx_key(key)[1]}: {p:.2f}" for key, p in ddx.top(k))


class Verbalizer:
    def __init__(self, model, tokenizer, max_new_tokens: int = 600):
        self.model, self.tok, self.max_new_tokens = model, tokenizer, max_new_tokens

    def verbalize(self, G: ReasoningGraph, ddx: Differential) -> str:
        msgs = P.verbalizer_messages(serialize_graph(G).text, ddx_text(ddx))
        return generate(self.model, self.tok, [chat_prompt(self.tok, msgs)], max_new_tokens=self.max_new_tokens)[0]


def audit(text: str, G: ReasoningGraph) -> Dict[str, List[str]]:
    gtext = serialize_graph(G).text
    ids = set(re.findall(r"\[([ORDM]\d+)\]", text))
    bad_ids = sorted(i for i in ids if i not in G)
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    bad_nums = sorted(n for n in nums if n not in gtext)
    return {"unknown_node_ids": bad_ids, "numbers_not_in_graph": bad_nums}
