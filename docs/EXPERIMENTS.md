# Claims ledger: claim → experiment → config → metric → figure → gate

| ID | Claim | Experiment | Config(s) | Primary metric | Figure | Gate |
|---|---|---|---|---|---|---|
| E0 | The problem is real | Baselines on ClinEdit-Faith (CoT base, CoT-SFT, note-edit, schema prompting) | `eval/evaluate.yaml` | MR, asymmetry, SCT κ | Fig. 3 (baseline arm) | G0 |
| E1 | Accuracy survives the bottleneck | ClinBlueprint hard / linear vs CoT-SFT; 2 families, 2 sizes | `train/sft_*.yaml`, `train/readout.yaml`, `eval/evaluate.yaml` | DiReCT acc + "all", MCR acc + recall, ER-Reason, CDM, QA sanity | Fig. 2 | G1 |
| E2 | Mediation (Axis 1) | Oracle-graph + model-graph regimes, frozen + re_reason | `eval/clinedit_faith.yaml`, `eval/evaluate.yaml` | MR, strong MR, asymmetry, coverage | Fig. 3a | – |
| E3 | Non-sycophancy (Axis 2) | Invalid / irrelevant arms; validator operating curve | `train/validator.yaml` | Flag sensitivity, false-flag, sycophancy rate, AUROC, invariance | Fig. 3b,c | G2 |
| E4 | Claim A: belief updating at two sites | New-finding inserts (re_reason) vs CoT text updates | `eval/evaluate.yaml` (SCT portions) | SCT credit, DxUpdate κ, direction; paired CI per site | Fig. 4 | – |
| E5 | Soft faithfulness / frontier | Soft-isolated, soft-in-context, in-context DPO vs hard | `train/faithfulness.yaml`, `train/dpo.yaml` | MR and accuracy across the bottleneck-strength spectrum | Fig. 3d / ED | G3 |
| E6 | No bypass (Axis 3) | Mech suite on in-context DPO vs CoT-SFT | `eval/mech.yaml` | Direct-note path share, flip-to-graph rate, slot patch recovery, attribution share, SAE Δ, probe selectivity | Fig. 5 | G4 |
| E7 | Human benefit (Axis 4) | Randomised crossover, 12–20 clinicians, ~120 public cases | `eval/study_power.yaml`, `clinician/study/` | Error detection (GEE A vs B), post-correction accuracy, time, automation bias, trust calibration | Fig. 6 | – |
| E8 | Claim C (optional) | Risk gate + LTT vs stop-only | (risk module) | Accuracy vs generated tokens; risk-exceedance rate | ED | G-C |

## Ablations (Extended Data)
- Hard vs soft bottleneck; no validator; no counterfactual loss; no DPO; silver-data volume; backbone family; layer for slot pooling.
- Candidate mode (`drafted` vs `kg_expand`); drafter-p bypass feature on; dx slots included in the soft head.
- Readout fitted on gold vs model-drafted graphs.

## Stress tests and analyses
- Distractor injection (MedDistractQA-style), note-section reordering, specialty hold-out.
- ε / ε_tv sensitivity; physician-verified subset only.
- Verbaliser audit: unknown node ids, numbers not in the graph.
- Failure-case gallery, reviewed by the clinician.
