"""Attribution of the diagnosis logit to prompt segments.

1. Integrated gradients over input embeddings (model-agnostic; always available).
2. circuit-tracer attribution graphs (transcoders): optional; supported for
   Qwen3-4B and Gemma models with released transcoders. VERIFY the API of the
   installed circuit-tracer version: the field names below follow v0.2.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def integrated_gradients(model, input_ids: List[int], target_ids: List[int], n_steps: int = 16) -> np.ndarray:
    """Per-token attribution of sum log p(target continuation) (IG, zero-embedding baseline)."""
    import torch

    dev = next(model.parameters()).device
    emb_layer = model.get_input_embeddings()
    ids = torch.tensor([input_ids + target_ids], device=dev)
    with torch.no_grad():
        x = emb_layer(ids).float()
    base = torch.zeros_like(x)
    total = torch.zeros(x.shape[1], device=dev)
    n_prompt = len(input_ids)
    for k in range(1, n_steps + 1):
        xi = (base + (k / n_steps) * (x - base)).detach().requires_grad_(True)
        out = model(inputs_embeds=xi.to(next(model.parameters()).dtype))
        lp = torch.log_softmax(out.logits[0, :-1].float(), -1)
        tgt = ids[0, 1:]
        score = lp.gather(-1, tgt[:, None]).squeeze(-1)[n_prompt - 1:].sum()
        (g,) = torch.autograd.grad(score, xi)
        total += (g * (x - base)).sum(-1)[0] / n_steps
    return total[:n_prompt].detach().cpu().numpy()


def segment_shares(attr: np.ndarray, segments: Dict[str, Sequence[int]]) -> Dict[str, float]:
    mag = np.abs(attr)
    tot = mag.sum() or 1.0
    return {k: float(mag[list(v)].sum() / tot) for k, v in segments.items() if len(v)}


def circuit_tracer_shares(prompt: str, model_name: str, transcoder_set: str, segments: Dict[str, Sequence[int]],
                          **attr_kwargs) -> Dict[str, float]:
    """Share of total influence on the top logit flowing from nodes at each segment's positions."""
    import torch
    from circuit_tracer import ReplacementModel, attribute

    rm = ReplacementModel.from_pretrained(model_name, transcoder_set, dtype=torch.bfloat16)
    graph = attribute(prompt=prompt, model=rm, **({"max_n_logits": 5, "desired_logit_prob": 0.9,
                                                    "max_feature_nodes": 4096} | attr_kwargs))
    A = graph.adjacency_matrix.float().abs()
    A = A / A.sum(1, keepdim=True).clamp_min(1e-9)
    n = A.shape[0]
    n_logits = len(graph.logit_tokens)
    logit_rows = list(range(n - n_logits, n))
    # total influence via power series of the row-normalised adjacency
    infl = A[logit_rows].clone()
    acc = infl.clone()
    for _ in range(50):
        infl = infl @ A
        acc += infl
        if infl.abs().sum() < 1e-6:
            break
    w = acc.sum(0)
    pos = node_positions(graph)
    out = {}
    for name, ps in segments.items():
        sel = [i for i, p in enumerate(pos) if p in set(ps)]
        out[name] = float(w[sel].sum() / w.sum().clamp_min(1e-9))
    return out


def node_positions(graph) -> List[int]:
    """Token position of every node (features, error nodes, embeddings; logits -> -1)."""
    n_pos = len(graph.input_tokens)
    feats = [int(p) for p in graph.active_features[graph.selected_features][:, 1].tolist()]
    n_layers = graph.cfg.n_layers if hasattr(graph, "cfg") else None
    errors = [p for _ in range(n_layers or 0) for p in range(n_pos)]
    embeds = list(range(n_pos))
    logits = [-1] * len(graph.logit_tokens)
    return feats + errors + embeds + logits
