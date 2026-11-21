# Mechanistic suite: "no bypass" (Axis 3)

**Question.** In the models where bypass is *possible* (soft-in-context, in-context generative, CoT), does diagnosis information reach the answer through the graph, or around it straight from the note?

Methods must **agree in direction** (decision gate G4). Attribution graphs are approximations, so we triangulate.

| # | Method | What a mediated model shows | What a bypassing model shows | Code |
|---|---|---|---|---|
| 1 | **Three-path decomposition**: log p(gold) under (full, note→answer knockout, graph-only, note-only) | Knockout ≈ full (small direct path); graph-only ≈ knockout (small hidden surplus) | Large drop under knockout; graph-only ≪ full | `mech/three_path.py` |
| 2 | **Graph swap**: note_A + graph_B | Follows graph_B | Stays with A's diagnosis | `mech/swap_and_patch.py` |
| 3 | **Minimal-pair activation patching**: flip one observation's polarity (token-length preserving); patch the residual at that slot layer by layer | High recovery at slot positions; ~0 for the note-position control | Low slot recovery; the effect lives elsewhere | `mech/swap_and_patch.py` |
| 4 | **Attribution share**: integrated gradients per segment; circuit-tracer total influence on the logit by token position | Most attribution on graph tokens | Most on note tokens | `mech/attribution.py` |
| 5 | **SAE disease-feature ablation**: zero diagnosis-selective features inside slots vs inside the note | Slot-side ablation hurts more | Note-side ablation hurts more | `mech/sae_probes.py` |
| 6 | **Leakage probe**: decode the diagnosis linearly from non-graph positions with slots ablated; selectivity vs shuffled-label control | Near-control accuracy | High selectivity | `mech/sae_probes.py` |

## Comparison set
- In-context generative ClinBlueprint (DPO) vs **matched CoT-SFT** (same backbone, same data) vs the untuned backbone.
- Soft-in-context head vs soft-isolated head (method 1 applies to their encoders).
- Reference taxonomy: the pattern-matching vs reasoning pathways reported for Gemma-2-2B (AAAI 2026). We test whether the bottleneck *suppresses* the pattern-matching pathway.

## Tooling and controls
- Run `scripts/mech/check_tooling.py` before any mechanistic run. It checks layer discovery, null knockout ≡ normal logits, and self-patch ≡ identity, and that transcoders and SAEs load.
- Attention knockout requires `attn_implementation="eager"`. The mech config forces it.
- **Transcoders**: Qwen3-4B (circuit-tracer; Neuronpedia `qwen3-4b`); Gemma-3 via Gemma Scope 2 (`google/gemma-scope-2-4b-it`) and `saillab/medgemma-circuit-tools`. **VERIFY** the set names in the installed versions.
- **Negative controls**: patching identical note positions; label-shuffled probes; random-feature ablations of matched size.
- Report per-case distributions, not only means. Disagreement between methods is itself a reported result.
