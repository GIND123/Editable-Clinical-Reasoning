"""Sanity checks before trusting any mechanistic result (run on the GPU host).

1. Decoder layers are found for the backbone.
2. A *null* attention knockout (blocking nothing) reproduces the normal logits.
3. Patching a layer with its own activations is a no-op.
4. Optional: circuit-tracer transcoders / SAELens SAEs load for the backbone.

    python scripts/mech/check_tooling.py --model Qwen/Qwen3-4B [--transcoders qwen3-4b-transcoders] [--sae-release ...]
"""

from __future__ import annotations

import argparse


def main() -> None:
    import torch

    from clinblueprint.mech.hooks import capture, decoder_layers, patch
    from clinblueprint.mech.knockout import knockout_mask
    from clinblueprint.models.backbone import BackboneConfig, load_model, load_tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-class", default="causal_lm")
    ap.add_argument("--transcoders", default=None)
    ap.add_argument("--sae-release", default=None)
    ap.add_argument("--sae-id", default=None)
    a = ap.parse_args()
    cfg = BackboneConfig(a.model, attn_implementation="eager", model_class=a.model_class)
    tok, model = load_tokenizer(cfg), load_model(cfg)
    layers = decoder_layers(model)
    print(f"[ok] {len(layers)} decoder layers")
    ids = tok("Chest pain radiating to the left arm. Final diagnosis:", return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        ref = model(input_ids=ids).logits.float()
        mask = knockout_mask(ids.shape[1], [], [], torch.bfloat16 if ref.dtype != torch.float32 else torch.float32,
                             model.device)
        ko = model(input_ids=ids, attention_mask=mask.to(next(model.parameters()).dtype)).logits.float()
    d = (ref - ko).abs().max().item()
    print(f"[{'ok' if d < 1e-2 else 'FAIL'}] null knockout max |Δlogit| = {d:.2e}")
    with capture(model, [len(layers) // 2]) as acts, torch.no_grad():
        model(input_ids=ids)
    pos = list(range(ids.shape[1]))
    with patch(model, len(layers) // 2, pos, acts[len(layers) // 2][0]), torch.no_grad():
        pp = model(input_ids=ids).logits.float()
    d2 = (ref - pp).abs().max().item()
    print(f"[{'ok' if d2 < 1e-2 else 'FAIL'}] self-patch max |Δlogit| = {d2:.2e}")
    if a.transcoders:
        try:
            from circuit_tracer import ReplacementModel

            ReplacementModel.from_pretrained(a.model, a.transcoders, dtype=torch.bfloat16)
            print("[ok] circuit-tracer transcoders load")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] circuit-tracer: {e}")
    if a.sae_release and a.sae_id:
        try:
            from clinblueprint.mech.sae_probes import load_sae

            load_sae(a.sae_release, a.sae_id)
            print("[ok] SAE loads")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] SAE: {e}")


if __name__ == "__main__":
    main()
