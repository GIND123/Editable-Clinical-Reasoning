"""Loss functions (torch). Kept free of model code so they can be unit-tested."""

from __future__ import annotations


def _t():
    import torch
    import torch.nn.functional as F

    return torch, F


def counterfactual_direction_loss(logp_before, logp_after, direction, margin: float = 0.5):
    """Hinge on the change of the measured candidate's log-probability.

    direction: tensor of +1 (gold says increase) / -1 (decrease).
    loss = relu(margin - direction * (logp_after - logp_before))
    """
    torch, F = _t()
    return F.relu(margin - direction * (logp_after - logp_before)).mean()


def invariance_loss(logits_before, logits_after, mask=None):
    """Symmetric KL between candidate distributions for irrelevant edits."""
    torch, F = _t()
    if mask is not None:
        logits_before = logits_before.masked_fill(~mask, float("-inf"))
        logits_after = logits_after.masked_fill(~mask, float("-inf"))
    lp, lq = F.log_softmax(logits_before, -1), F.log_softmax(logits_after, -1)
    p, q = lp.exp(), lq.exp()
    kl_pq = torch.where(p > 0, p * (lp - lq), torch.zeros_like(p)).sum(-1)
    kl_qp = torch.where(q > 0, q * (lq - lp), torch.zeros_like(q)).sum(-1)
    return 0.5 * (kl_pq + kl_qp).mean()


def task_ce(logits, gold_idx):
    torch, F = _t()
    return F.cross_entropy(logits, gold_idx)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    """Rafailov et al. 2023. Inputs are per-sequence summed log-probs."""
    torch, F = _t()
    logits = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean()


def sequence_logprob(model, input_ids, attention_mask, completion_mask):
    """Sum of log p(token) over completion positions. completion_mask aligns with input_ids."""
    torch, F = _t()
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logp = F.log_softmax(out.logits[:, :-1].float(), -1)
    tgt = input_ids[:, 1:]
    tok = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = completion_mask[:, 1:].float()
    return (tok * m).sum(-1)
