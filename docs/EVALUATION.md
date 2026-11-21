# Evaluation: definitions and statistics

## 1. Regimes
| Regime | Graph that edits act on | Isolates |
|---|---|---|
| **Oracle graph** | Physician gold graph (DiReCT) | Readout mediation independent of drafting quality |
| **Model graph** | The system's own draft; edits retargeted by text (`edits/align.py`) | End-to-end behaviour; **coverage** (share of applicable edits) reported |
| **Trace** (CoT) | The CoT trace, edited by `edits/text_adapter.py`, then the model continues | Faithfulness of free-text reasoning |
| **Note** (input-level) | The input note, then the model re-answers | The You et al. regime |

All systems score the **same fixed candidate pool** per case, so p(·) is comparable across systems.

## 2. Mediation (Axis 1) (`eval/mediation.py`)
For edit e with gold y_e on measured diagnosis m_e: Δ_e = p_after(m_e) − p_before(m_e); TV_e is the total-variation distance between the differentials.

- **followed**:
  - Δ > ε if y = increase;
  - Δ < −ε if y = decrease;
  - TV ≤ ε_tv with an unchanged top-1 if y = no_change;
  - flagged if y = flag.
  - Defaults: ε = 0.02, ε_tv = 0.05; sensitivity to ε is reported.
- **Mediation rate (MR)**: mean followed over {valid_correction, counterfactual, new_finding} with directional gold.
- **Strong MR**: followed *and* top-1 equals `expected_top` where specified.
- **Asymmetry A = MR(valid_correction) − MR(counterfactual).** A model that already "knows" the answer from the note follows agreeing edits and resists opposing ones, so A > 0 is a behavioural bypass signature. A mediated model has A ≈ 0.
- **Opposite rate**: moved against the gold direction.

## 3. Non-sycophancy (Axis 2) (`eval/nonsycophancy.py`)
- Flag sensitivity: P(flag | invalid assertion).
- False-flag rate: P(flag | valid assertion); also reported for irrelevant assertions.
- Sycophancy rate: P(not flagged ∧ moved in the implied direction | invalid).
- AUROC of p_contradict, invalid vs valid assertions; the full operating-point curve (Fig. 3b).
- Invariance rate: followed over irrelevant edits.

## 4. Belief updating (Claim A) (`eval/sct.py`)
- Model update for an SCT item = Δ log-odds of the hypothesis after inserting the finding (`re_reason` mode).
- That update maps to −2..+2 via (t1, t2) fitted on dev to maximise linear-weighted κ.
- **Metrics**:
  - SCT aggregate credit (expert-panel scoring);
  - direction accuracy;
  - ER-Reason DxUpdate (per-timestep κ, averaged), DxTrajectory (Spearman ρ) and FinalDx.
- **Claim A test**: the paired cluster-bootstrap CI of (graph-mediated − text-updated CoT) excludes 0 **at each site** (BIDMC via DiReCT-derived new-finding edits; UCSF via ER-Reason) and on SCT-Bench.

## 5. Accuracy (`eval/accuracy.py`)
- **DiReCT**: diagnosis accuracy (exact and hierarchical), plus completeness and faithfulness using the DiReCT formulas (`comp_pre`, `comp_re`, `comp_coverage`, `faith_ob`, `faith_all`) with a pluggable judge.
- **MedCaseReasoning**: accuracy (local LLM judge validated on a public subset against physician labels) and reasoning recall.
- **ER-Reason**: ICD/CCSR-level accuracy (secondary). **MIMIC-CDM**: 4-way accuracy.
- **QA sanity**: candidate log-likelihood scoring; must stay within 2 points of the untuned backbone.

## 6. Statistics
- Cluster bootstrap over cases/patients (2,000 resamples) for every rate (`eval/bootstrap.py`); paired bootstrap for system differences on identical edits.
- Results stratified by edit class, operation, gold source and physician-verified status. The headline is repeated on the physician-verified subset.
- Small-cell suppression (< 10) in every table derived from credentialed data.

## 7. Baselines (same backbone unless noted; all local)
1. Free-CoT SFT on the **same cases** (`training/cot_sft.py`)
2. MedCaseReasoning-style rationale SFT (the same module, with clinician text)
3. Schema-guided prompting (untuned backbone + drafter prompt + hard readout)
4. Self-refine (`baselines/refine_and_sample.py`)
5. Counterfactual multi-agent, You et al.-style (`baselines/counterfactual_multiagent.py`)
6. Best-of-N with verifier (`baselines/refine_and_sample.py`)
7. ITCR-style stop-only (Claim C comparison)
8. External open models: MedGemma-4B/27B (27B in 4-bit, flagged), Qwen3-8B, HuatuoGPT-o1-8B, m1-7B, MedReason-8B
9. Closed models: published numbers only

### Published reference numbers
| Benchmark | Model | Number | Source |
|---|---|---|---|
| DiReCT | GPT-4-turbo | Acc 0.610; "all" 0.210 | [P] DiReCT paper |
| MedCaseReasoning | DeepSeek-R1 (10-shot) | 48% acc; 64% reasoning recall | [V] |
| MIMIC-CDM-FI / MedCaseReasoning / ER-Reason | GPT-5-mini | 93.6 / 66.0 / 26.7 (80–100-case subsamples; GPT-4.1 judge) | [V] You et al. |
| MIMIC-CDM-FI | MedGemma-1.5-4B + counterfactual multi-agent | 79.2 | [V] You et al. |
| SCT | o3 / o1-preview / R1 | 67.8 / 58.2 / 55.5 | [V, secondary source; verify] |
| MedQA | MedGemma-1.5-4B; HuatuoGPT-o1-8B; Med-PRM+Meerkat-8B | 69.1; 72.6; 80.35 (64 samples) | [P] |
