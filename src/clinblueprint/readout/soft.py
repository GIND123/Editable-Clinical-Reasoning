"""Soft bottleneck: a small learned head over slot embeddings.

Bottleneck-strength spectrum (a headline analysis; Fig. 3/5):

  hard            deterministic / linear readout of G             (mediation by construction)
  soft-isolated   head over slot embeddings encoded from the       (information can only enter
                  serialised graph ALONE (encoding="graph_only")    through the graph's text)
  soft-in-context head over slot embeddings encoded with the note  (hidden-state bypass possible;
                  in context (encoding="note_in_context")           measured, trained against)
  CoT             free text                                          (no bottleneck)

Anti-bypass defaults: diagnosis slots are excluded from the head's keys
(``exclude_dx_slots``). Candidates are embedded from their *label text*, not
from the graph, so the head must reach a candidate through the rationales
that name it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from clinblueprint.schema.graph import ReasoningGraph, split_dx_key
from clinblueprint.schema.serialize import SerializedGraph, serialize_graph

SLOT_TYPE_IDS = {"observation": 0, "rationale": 1, "diagnosis": 2, "must_not_miss": 3}


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_head(d_model: int, d_hidden: int = 512, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
    torch, nn = _torch()

    class SlotReadoutHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.slot_proj = nn.Linear(d_model, d_hidden)
            self.cand_proj = nn.Linear(d_model, d_hidden)
            self.type_emb = nn.Embedding(len(SLOT_TYPE_IDS), d_hidden)
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "attn": nn.MultiheadAttention(d_hidden, n_heads, dropout=dropout, batch_first=True),
                    "ln1": nn.LayerNorm(d_hidden),
                    "ff": nn.Sequential(nn.Linear(d_hidden, 4 * d_hidden), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(4 * d_hidden, d_hidden)),
                    "ln2": nn.LayerNorm(d_hidden),
                }) for _ in range(n_layers)
            ])
            self.out = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
            # learned "no evidence" key so attention is defined when all slots are masked
            self.null_key = nn.Parameter(torch.zeros(1, 1, d_hidden))

        def forward(self, slots, slot_types, slot_mask, cands, cand_mask):
            """slots [B,S,d], slot_types [B,S], slot_mask [B,S] (True=keep), cands [B,K,d], cand_mask [B,K]."""
            B = slots.shape[0]
            kv = self.slot_proj(slots) + self.type_emb(slot_types)
            kv = torch.cat([self.null_key.expand(B, 1, -1), kv], dim=1)
            keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=slots.device), slot_mask.bool()], dim=1)
            q = self.cand_proj(cands)
            for layer in self.layers:
                a, _ = layer["attn"](q, kv, kv, key_padding_mask=~keep)
                q = layer["ln1"](q + a)
                q = layer["ln2"](q + layer["ff"](q))
            logits = self.out(q).squeeze(-1)
            return logits.masked_fill(~cand_mask.bool(), float("-inf"))

    return SlotReadoutHead()


@dataclass
class EncodedGraph:
    slot_emb: "object"     # torch.Tensor [S, d]
    slot_types: List[int]
    slot_ids: List[str]


class SlotEncoder:
    """Run the backbone and mean-pool hidden states over each slot's token span."""

    def __init__(self, model, tokenizer, layer: int = -1, encoding: str = "graph_only",
                 exclude_dx_slots: bool = True, max_length: int = 8192):
        self.model, self.tok = model, tokenizer
        self.layer, self.encoding = layer, encoding
        self.exclude_dx_slots = exclude_dx_slots
        self.max_length = max_length
        self._cand_cache: Dict[str, "object"] = {}

    def _text(self, G: ReasoningGraph, note: Optional[str]) -> Tuple[str, SerializedGraph, int]:
        sg = serialize_graph(G, include_dx_slots=not self.exclude_dx_slots)
        if self.encoding == "note_in_context" and note:
            prefix = f"Clinical note:\n{note}\n\n"
            return prefix + sg.text, sg, len(prefix)
        return sg.text, sg, 0

    def _hidden(self, text: str, grad: bool):
        torch, _ = _torch()
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True,
                       max_length=self.max_length, add_special_tokens=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(next(self.model.parameters()).device) for k, v in enc.items()}
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states[self.layer][0], offsets

    def encode(self, G: ReasoningGraph, note: Optional[str] = None, grad: bool = False) -> EncodedGraph:
        torch, _ = _torch()
        text, sg, shift = self._text(G, note)
        H, offsets = self._hidden(text, grad)
        embs, types, ids = [], [], []
        for s in sg.slots:
            a, b = s.start + shift, s.end + shift
            tok_idx = [i for i, (x, y) in enumerate(offsets) if y > a and x < b and y > x]
            if not tok_idx:
                continue
            embs.append(H[tok_idx].mean(0))
            types.append(SLOT_TYPE_IDS[s.node_type])
            ids.append(s.node_id)
        d = H.shape[-1]
        emb = torch.stack(embs) if embs else torch.zeros(0, d, device=H.device, dtype=H.dtype)
        return EncodedGraph(emb, types, ids)

    def encode_candidates(self, keys: Sequence[str]):
        torch, _ = _torch()
        out = []
        for k in keys:
            if k not in self._cand_cache:
                cat, leaf = split_dx_key(k)
                H, _ = self._hidden(f"Diagnosis: {leaf} (category: {cat})", grad=False)
                self._cand_cache[k] = H.mean(0).detach()
            out.append(self._cand_cache[k])
        return torch.stack(out)


class SoftReadout:
    """Callable readout (same interface as HardReadout): (G, fixed) -> Differential.

    Candidate set comes from the HardReadout candidate logic (kg_expand), so
    hard and soft variants score the same labels.
    """

    def __init__(self, encoder: SlotEncoder, head, candidate_source, note_lookup=None, temperature: float = 1.0):
        self.encoder, self.head = encoder, head
        self.cands = candidate_source  # HardReadout (uses .candidates)
        self.note_lookup = note_lookup or {}
        self.temperature = temperature

    def logits(self, G: ReasoningGraph, fixed=None, grad: bool = False):
        torch, _ = _torch()
        keys = self.cands.candidates(G, fixed)
        enc = self.encoder.encode(G, self.note_lookup.get(G.case_id), grad=grad)
        ce = self.encoder.encode_candidates(keys)
        batch = collate([enc], [ce])
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lg = self.head(*[b.float() if b.is_floating_point() else b for b in batch])[0]
        return keys, lg

    def __call__(self, G: ReasoningGraph, fixed=None):
        from clinblueprint.readout.differential import Differential

        keys, lg = self.logits(G, fixed)
        return Differential.from_scores(dict(zip(keys, lg.float().cpu().numpy().tolist())), self.temperature)


def collate(encoded: Sequence[EncodedGraph], cand_embs: Sequence["object"]):
    """Pad a batch for the head."""
    torch, _ = _torch()
    B = len(encoded)
    S = max(1, max(e.slot_emb.shape[0] for e in encoded))
    K = max(c.shape[0] for c in cand_embs)
    d = cand_embs[0].shape[-1]
    dev, dt = cand_embs[0].device, cand_embs[0].dtype
    slots = torch.zeros(B, S, d, device=dev, dtype=dt)
    types = torch.zeros(B, S, dtype=torch.long, device=dev)
    smask = torch.zeros(B, S, dtype=torch.bool, device=dev)
    cands = torch.zeros(B, K, d, device=dev, dtype=dt)
    cmask = torch.zeros(B, K, dtype=torch.bool, device=dev)
    for i, (e, c) in enumerate(zip(encoded, cand_embs)):
        n = e.slot_emb.shape[0]
        if n:
            slots[i, :n] = e.slot_emb.to(dt)
            types[i, :n] = torch.tensor(e.slot_types, device=dev)
            smask[i, :n] = True
        cands[i, :c.shape[0]] = c
        cmask[i, :c.shape[0]] = True
    return slots, types, smask, cands, cmask
