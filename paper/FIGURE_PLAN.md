# Figure plan (NMI Article; main figures 1–6 + Extended Data)

| Fig | Panels | Data source (outputs/) | Producer |
|---|---|---|---|
| **1** Concept | (a) Note → graph → readout schematic; (b) a valid edit propagating; (c) an invalid assertion flagged with evidence; (d) bottleneck-strength spectrum | Toy / real example graph | `examples/toy_demo.py` (adapted), hand-drawn schematic |
| **2** Accuracy | Accuracy across DiReCT, MedCaseReasoning, ER-Reason, MIMIC-CDM, QA suites; 2 families × 2 sizes vs baselines; DiReCT explanation metrics | `eval/*/*.accuracy.summary.json` | `paper/figures/make_fig2.py` (to write) |
| **3** Mediation and non-sycophancy | (a) MR by edit class per system, plus asymmetry; (b) flag-sensitivity vs false-flag operating curves; (c) scatter: follow-valid vs flag-invalid per model; (d) bottleneck-strength frontier (MR vs accuracy) | `*.edits.jsonl`, `*.summary.json` | `eval/nonsycophancy.operating_points` |
| **4** Belief updating | SCT credit / DxUpdate κ / direction: graph-mediated vs text, at BIDMC, UCSF and SCT-Bench, with paired CIs | ER-Reason + SCT-Bench runs | `eval/sct.py` |
| **5** Mechanism | (a) three-path shares; (b) graph-swap flip rates; (c) layerwise patching recovery (slot vs note control); (d) attribution share; (e) SAE ablation Δ; contrast with the pattern-matching pathway | `mech/*` | `mech/run.py` |
| **6** Clinician study | Detection rate by arm (GEE OR, CI); post-correction accuracy; automation bias; time; trust calibration | study analysis | `study/design.analyze` |

**Extended Data**
- Ablations (hard vs soft, no validator, no counterfactual loss, no DPO, silver volume, backbone family, slot layer, candidate mode).
- Stress tests (distractors, section reordering, specialty hold-out).
- Claim C front.
- Leakage probes.
- ε sensitivity.
- Physician-verified subset.
- Verbaliser audit.
- Failure cases.
- Data flow and contamination controls.

Style: follow the NMI figure guide (fonts ≥ 5 pt at final size, colour-blind-safe palette, no 3D), error bars = 95% cluster-bootstrap CIs, n shown in each panel, small cells suppressed.
