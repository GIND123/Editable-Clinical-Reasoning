"""SAE disease-feature ablation and leakage probes.

SAE ablation: pick the features most selective for a diagnosis (mean activation
on slot tokens in cases with that diagnosis minus others), then zero them
(keeping the SAE reconstruction error) either inside graph slots or inside the
note, and compare the drop in log p(gold). Bypass predicts a large note-side
effect; mediation predicts a large slot-side effect.

Leakage probe: can the diagnosis be decoded linearly from hidden states at
non-graph positions when the slots are ablated? Report selectivity against a
control task with shuffled labels (Hewitt & Liang 2019).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Sequence

import numpy as np

from clinblueprint.mech.hooks import decoder_layers


def load_sae(release: str, sae_id: str, device: str = "cuda"):
    from sae_lens import SAE

    out = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return out[0] if isinstance(out, tuple) else out


def select_features(acts_pos: np.ndarray, acts_neg: np.ndarray, k: int = 20) -> List[int]:
    """acts_*: [n_tokens, n_features] SAE activations. Highest mean difference first."""
    diff = acts_pos.mean(0) - acts_neg.mean(0)
    return list(np.argsort(-diff)[:k])


@contextmanager
def sae_feature_ablation(model, sae, layer: int, positions: Sequence[int], features: Sequence[int]):
    import torch

    L = decoder_layers(model)
    feats = torch.tensor(list(features))

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        x = h[:, list(positions)].float()
        f = sae.encode(x)
        err = x - sae.decode(f)
        f[..., feats.to(f.device)] = 0
        h = h.clone()
        h[:, list(positions)] = (sae.decode(f) + err).to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hd = L[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


def probe_selectivity(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Linear probe accuracy vs a label-shuffled control (grouped split by case)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    y_ctrl = rng.permutation(y)
    accs, ctrl = [], []
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        for lab, store in ((y, accs), (y_ctrl, ctrl)):
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(X[tr], lab[tr])
            store.append(clf.score(X[te], lab[te]))
    return {"probe_acc": float(np.mean(accs)), "control_acc": float(np.mean(ctrl)),
            "selectivity": float(np.mean(accs) - np.mean(ctrl))}
