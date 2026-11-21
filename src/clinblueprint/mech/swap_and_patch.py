"""Graph-swap test and minimal-pair activation patching.

Graph swap (behavioural, length-agnostic): feed note_A with graph_B (cases with
different diagnoses). A mediated model follows graph_B; a bypassing model
stays with A's diagnosis. Report flip-to-graph rate.

Minimal-pair patching (mechanistic): clean graph G and corrupted G' differ by one
token-length-preserving edit (e.g. "present" -> "absent" in one observation
slot). For each layer l, patch the residual at the edited slot's positions from
the corrupted run into the clean run and measure the recovered fraction of the
corrupted logit difference:
    recovery_l = (LD_patched - LD_clean) / (LD_corrupt - LD_clean)
Patching the same positions in the *note* (identical in both runs) is the
negative control and should recover ~0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from clinblueprint.mech.hooks import capture, decoder_layers, patch
from clinblueprint.mech.segments import segment_prompt
from clinblueprint.models.backbone import chat_prompt
from clinblueprint.schema.graph import Polarity, ReasoningGraph
from clinblueprint.schema.serialize import serialize_graph

ANSWER_PREFIX = "Final diagnosis:"


def _prompt(tok, note: str, graph_text: str) -> str:
    body = f"Clinical note:\n{note}\n\nReasoning graph:\n{graph_text}\n\nGive the final diagnosis."
    return chat_prompt(tok, [{"role": "user", "content": body}]) + ANSWER_PREFIX


def graph_swap_flip(model, tok, note_a: str, graph_b_text: str, candidates: Sequence[str], dx_a: int, dx_b: int) -> str:
    from clinblueprint.models.scoring import candidate_distribution

    p = candidate_distribution(model, tok, _prompt(tok, note_a, graph_b_text), candidates)
    top = int(np.argmax(p))
    return "follows_graph" if top == dx_b else ("follows_note" if top == dx_a else "other")


def minimal_pair(G: ReasoningGraph, obs_id: str) -> Tuple[ReasoningGraph, ReasoningGraph]:
    """Flip the polarity of one observation (rationales left in place so the text differs by one word)."""
    Gc = G.copy_deep()
    Gx = G.copy_deep()
    o = Gx.observations[obs_id]
    o.polarity = Polarity.ABSENT if o.polarity == Polarity.PRESENT else Polarity.PRESENT
    return Gc, Gx


def _logit_diff(model, ids: List[int], a_ids: List[int], b_ids: List[int]) -> float:
    """log p(first token of a) - log p(first token of b) at the answer position."""
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return float(lp[a_ids[0]] - lp[b_ids[0]])


def layerwise_patching(model, tok, note: str, G: ReasoningGraph, obs_id: str, cand_a: str, cand_b: str,
                       control_positions: Optional[List[int]] = None) -> Optional[Dict[str, List[float]]]:
    import torch

    Gc, Gx = minimal_pair(G, obs_id)
    sc, sx = serialize_graph(Gc), serialize_graph(Gx)
    pc, px = _prompt(tok, note, sc.text), _prompt(tok, note, sx.text)
    segc = segment_prompt(tok, pc, [("note", note), ("graph", sc.text)], graph=sc)
    segx = segment_prompt(tok, px, [("note", note), ("graph", sx.text)], graph=sx)
    if len(segc.input_ids) != len(segx.input_ids):
        return None  # edit not token-length preserving for this tokenizer: skip pair
    pos = [i for i in segc.slot_positions.get(obs_id, []) if segc.input_ids[i] != segx.input_ids[i]] \
        or segc.slot_positions.get(obs_id, [])
    a_ids = tok(" " + cand_a, add_special_tokens=False)["input_ids"]
    b_ids = tok(" " + cand_b, add_special_tokens=False)["input_ids"]
    ld_c = _logit_diff(model, segc.input_ids, a_ids, b_ids)
    ld_x = _logit_diff(model, segx.input_ids, a_ids, b_ids)
    n_layers = len(decoder_layers(model))
    dev = next(model.parameters()).device
    with capture(model, range(n_layers)) as acts:
        with torch.no_grad():
            model(input_ids=torch.tensor([segx.input_ids], device=dev))
    ctrl = control_positions if control_positions is not None else segc.segments["note"][: len(pos)]
    rec_slot, rec_ctrl = [], []
    denom = (ld_x - ld_c) if abs(ld_x - ld_c) > 1e-6 else float("nan")
    for layer in range(n_layers):
        for positions, out in ((pos, rec_slot), (ctrl, rec_ctrl)):
            with patch(model, layer, positions, acts[layer][0, positions]):
                ld_p = _logit_diff(model, segc.input_ids, a_ids, b_ids)
            out.append((ld_p - ld_c) / denom)
    return {"slot": rec_slot, "control_note": rec_ctrl, "ld_clean": [ld_c], "ld_corrupt": [ld_x]}
