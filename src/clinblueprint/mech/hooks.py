"""Residual-stream capture / patch / ablate hooks for HF decoder models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional


def base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def decoder_layers(model):
    m = base_model(model)
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers",
                 "model.model.layers", "transformer.h"):
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate decoder layers; add the path for this architecture")


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextmanager
def capture(model, layers: Iterable[int]):
    """Yield dict layer -> residual output [B, L, d] (filled after the forward pass)."""
    store: Dict[int, object] = {}
    hs = []
    L = decoder_layers(model)
    for i in layers:
        def hook(mod, inp, out, i=i):
            store[i] = _hidden(out).detach()
        hs.append(L[i].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in hs:
            h.remove()


@contextmanager
def patch(model, layer: int, positions: List[int], values, batch_index: Optional[int] = None):
    """Overwrite residual output of ``layer`` at ``positions`` with ``values`` [len(positions), d]."""
    L = decoder_layers(model)

    def hook(mod, inp, out):
        h = _hidden(out).clone()
        idx = slice(None) if batch_index is None else batch_index
        h[idx, positions] = values.to(h.dtype).to(h.device)
        return _replace(out, h)

    hd = L[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


@contextmanager
def ablate(model, layer: int, positions: List[int], mean_vector=None):
    """Zero- (or mean-) ablate the residual at ``positions`` after ``layer``."""
    L = decoder_layers(model)

    def hook(mod, inp, out):
        h = _hidden(out).clone()
        h[:, positions] = 0 if mean_vector is None else mean_vector.to(h.dtype).to(h.device)
        return _replace(out, h)

    hd = L[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        hd.remove()


@contextmanager
def layer_attention_mask(model, layers: Iterable[int], mask4d):
    """Replace the attention mask seen by selected decoder layers (attention knockout in a
    layer window). Requires attn_implementation='eager' (or sdpa accepting a float 4D mask).
    Sanity-check with an all-zero knockout before trusting results (scripts/mech/check_tooling.py)."""
    L = decoder_layers(model)
    hs = []
    for i in layers:
        def pre(mod, args, kwargs):
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = mask4d.to(kwargs["attention_mask"].dtype if kwargs["attention_mask"]
                                                     is not None else mask4d.dtype)
            return args, kwargs
        hs.append(L[i].register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for h in hs:
            h.remove()
