"""Candidate scoring by continuation log-likelihood.

Used wherever a *distribution* over diagnoses is needed from a generative LM:
the CoT baseline (after its, possibly edited, trace), the in-context graph
reader, the QA sanity suites and the mechanistic suite.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def continuation_logprobs(model, tokenizer, prompt: str, continuations: Sequence[str], *,
                          length_normalize: bool = True, batch_size: int = 8) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    scores: List[float] = []
    for i in range(0, len(continuations), batch_size):
        chunk = continuations[i:i + batch_size]
        seqs, cont_lens = [], []
        for c in chunk:
            c_ids = tokenizer(c, add_special_tokens=False)["input_ids"]
            seqs.append(p_ids + c_ids)
            cont_lens.append(len(c_ids))
        L = max(map(len, seqs))
        pad = tokenizer.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):  # right padding for scoring
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:].to(device)
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for j, s in enumerate(seqs):
            n = cont_lens[j]
            lp = tok_lp[j, len(s) - 1 - n:len(s) - 1].sum().item()
            scores.append(lp / max(1, n) if length_normalize else lp)
    return np.array(scores)


def candidate_distribution(model, tokenizer, prompt: str, candidates: Sequence[str], *,
                           prefix: str = " ", length_normalize: bool = True) -> np.ndarray:
    lp = continuation_logprobs(model, tokenizer, prompt, [prefix + c for c in candidates],
                               length_normalize=length_normalize)
    z = lp - lp.max()
    p = np.exp(z)
    return p / p.sum()
