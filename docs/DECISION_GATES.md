# Decision gates and pivots

Gates are ordered by dependency, not by date. Each gate has a measurable criterion and a pre-agreed consequence. Every pivot is still an NMI-shaped paper.

| Gate | Checked after | Criterion | If it passes | If it fails |
|---|---|---|---|---|
| **G0: the problem is real** | Baselines (CoT, CoT-SFT, untuned backbone) on ClinEdit-Faith | Baseline mediation ≤ 0.7, or poor SCT concordance, or asymmetry > 0 with CI excluding 0 | Proceed | Reframe around non-sycophancy + human benefit |
| **G1: the bottleneck keeps accuracy** | Graph SFT + hard readout | DiReCT / MedCaseReasoning accuracy ≥ CoT-SFT − 2 points | Proceed with the model paper | **Pivot A (Analysis article)**: "Clinical language models are not causally editable". Mediation, sycophancy and SCT updating across 20+ open models at two hospitals, with mechanistic evidence and the ClinEdit-Faith release |
| **G2: the validator works** | Validator training | AUROC ≥ 0.80 on dev; false-flag ≤ 0.10 at sensitivity 0.80 | Claim B | Report rules-only validator; frame Claim B as a trade-off curve |
| **G3: soft faithfulness** | Faithfulness + DPO | Soft mediation ≥ 0.80 *and* non-sycophancy targets met | Full four-axis claim | **Pivot B**: lead with the hard bottleneck + validator + mechanism + clinician study; soft results as the frontier |
| **G4: mechanism agrees** | Mech suite (≥ 3 methods) | Methods agree in direction | Axis 3 claimed | Report disagreement honestly; restrict the claim to behavioural mediation |
| **G-C: Claim C earns its place** | Risk gate + LTT | Improves the accuracy-vs-tokens front with a valid LTT certificate | Include as a secondary result | Drop to Extended Data or remove |
| **G-RL: GRPO needed?** | After SFT+DPO | Trails same-size baselines by > 3 points | Run GRPO with verifiable rewards | Skip |

**Editorial signal.** Send the NMI presubmission enquiry (abstract + Figs 2–5) once G1–G3 have outcomes (`paper/presubmission_enquiry.md`).
