"""Attention knockout and masked candidate scoring.

``knockout_mask`` builds an additive 4D mask = causal + (-inf on edges from
query positions to blocked key positions). Used to cut the *direct* path from
the note to the answer while leaving note -> graph attention intact.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def knockout_mask(seq_len: int, block_queries: Sequence[int], block_keys: Sequence[int], dtype, device):
    import torch

    neg = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), neg, dtype=dtype, device=device).triu(1)
    if len(block_queries) and len(block_keys):
        q = torch.tensor(list(block_queries), device=device)
        k = torch.tensor(list(block_keys), device=device)
        m[q.unsqueeze(1), k.unsqueeze(0)] = neg
    return m[None, None]


def masked_candidate_logprobs(model, prompt_ids: List[int], cand_ids: List[List[int]], block_keys: Sequence[int],
                              block_from: int, length_normalize: bool = True) -> np.ndarray:
    """Score candidates with queries at positions >= ``block_from`` unable to attend ``block_keys``.

    ``block_from`` is normally the last prompt position (the answer position).
    """
    import torch

    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if not dtype.is_floating_point:
        dtype = torch.bfloat16
    out = []
    for c in cand_ids:
        ids = prompt_ids + c
        L = len(ids)
        mask = knockout_mask(L, list(range(block_from, L)), list(block_keys), dtype, dev)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids], device=dev), attention_mask=mask).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1)
        tgt = torch.tensor(ids[1:], device=dev)
        tok = lp.gather(-1, tgt[:, None]).squeeze(-1)
        s = tok[len(prompt_ids) - 1:].sum().item()
        out.append(s / max(1, len(c)) if length_normalize else s)
    return np.array(out)
