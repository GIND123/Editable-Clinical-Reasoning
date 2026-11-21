# Literature and dataset search log

Purpose: a record of what was searched, where, and what was found, so the novelty check can be re-run before submission. Searches from earlier rounds are summarised in [IDEA.md](IDEA.md).

## Round 4b (2026-11-21): PubMed, medRxiv, GitHub, web

### Repository and format verification (GitHub)
| Source | Verified |
|---|---|
| github.com/wbw520/DiReCT | Annotation tree format (`$Intermedia/$Cause/$InputN`), 6 input sections, `diagnostic_kg/` (25 categories) structure, evaluation formulas (`statistics.py`), PhysioNet DOI 10.13026/yf96-kc87 |
| github.com/kevinwu23/Stanford-MedCaseReasoning | HF id `zou-lab/MedCaseReasoning`; fields `case_prompt`, `diagnostic_reasoning`, `final_diagnosis`; splits 13,092 / 897; CC-BY 4.0 |
| github.com/AlaaLab/ER-Reason | File list (`er_reason.csv`, `annotated_sct.csv`, `gt_clean.csv`, …), column descriptions, DxUpdate / DxTrajectory / FinalDx implementations, typo fixes |
| github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset | PhysioNet CSV layout (`ConvertPhysionet.py`) |

### Web search
- ER-Reason on PhysioNet lists **version 1.1.0** (the paper links 1.0.0); licence is the **Contributor Review Health Data License 1.5.0**.
- The MIMIC-IV-Ext-CDM project page confirms `history_of_present_illness.csv` / `physical_examination.csv`.
- "Do Language Models Think Like Doctors?" (medRxiv 10.1101/2025.02.11.25321822): 750 SCT questions from 10 datasets, concor.dance platform, held-out test via DUA.
- MedDistractQA: code at github.com/nyuolab/MedDistractQA; Bystander and Non-literal variants.
- **Interpretability tooling**:
  - circuit-tracer supports Qwen3-4B transcoders (Neuronpedia `qwen3-4b`) and Gemma-3 through the NNSight backend.
  - Gemma Scope 2 (SAEs, transcoders, cross-layer transcoders for Gemma 3) is at `google/gemma-scope-2-4b-it`.
  - `saillab/medgemma-circuit-tools` exists on Hugging Face.

### PubMed
*According to PubMed*, the following were retrieved and are relevant (DOIs given):

| Topic | Article | DOI | Use |
|---|---|---|---|
| Explanations vs biased AI | Jabbour et al., *JAMA* 2023: randomised clinical vignette survey | 10.1001/jama.2023.22295 | Human-study design precedent; motivation |
| LLM and diagnostic reasoning RCT | Goh et al., *JAMA Netw Open* 2024 | 10.1001/jamanetworkopen.2024.40969 | Outcome rubric precedent; motivation |
| LLM and management reasoning RCT | Goh et al., *Nat Med* 2025 | 10.1038/s41591-024-03456-y | Related human study |
| Automation bias RCT | Qazi et al., *NEJM AI* | 10.1056/AIoa2501001 | Automation-bias outcome definition |
| UK replication | Healy et al., *Diagnosis* 2025 | 10.1515/dx-2025-0176 | Related human study |
| Sycophancy (illogical requests) | *npj Digit Med* 2025 | 10.1038/s41746-025-02008-z | External sycophancy stress test |
| Sycophancy commentary | *npj Digit Med* 2025 | 10.1038/s41746-025-02135-7 | Background |
| Sycophancy + diagnostic instability | *Artif Intell Med* 2026 (120 vignettes; MultiCaRe + MedMCQA) | 10.1016/j.artmed.2026.103516 | External protocol; closest clinical sycophancy evaluation (analysis only) |
| Warmth training → sycophancy | *Nature* 2026 | 10.1038/s41586-026-10410-0 | Motivation |
| Persona-driven sycophancy | medRxiv 2026 | 10.64898/2026.05.17.26353406 | Background |
| MultiCaRe dataset | Nievas Offidani et al., *Data Brief* 2023 | 10.1016/j.dib.2023.110008 | Extra public cases (dedup!) |
| MedNLI-Pain | *JAMIA Open* 2026 | 10.1093/jamiaopen/ooag143 | Related clinical NLI; validator background |
| Explainability of decoder-only clinical LLMs (scoping review) | *Artif Intell Med* 2026 | 10.1016/j.artmed.2026.103501 | Related-work section |
| LLMs generating SCT items | *Med Teach* 2025; *JMIR Form Res* 2025 | 10.1080/0142159X.2025.2497888; 10.2196/76618 | SCT background (not data) |
| LLM on SCT in palliative pharmacy | *J Palliat Med* 2025 | 10.1177/10966218251376436 | SCT background |

No PubMed result trained a clinical model for **graph-mediated, non-sycophantic editability**. That is consistent with the novelty verdict in IDEA.md §6.

### medRxiv / bioRxiv
- The connector supports date / category browsing only (no keyword search). A 30-day browse of medRxiv found one adjacent item: "From clinician review to measurable human oversight of AI-drafted paediatric discharge documents" (10.64898/2026.10.20.26365977). It measures whether review intercepts generation errors and is relevant to the human-study framing.
- Re-check before submission: the SCT-Bench comparative assessment (10.64898/2026.03.13.26347597) for any SCT-*trained* models.

## Before submission: re-run checklist
- [ ] arXiv: "causal mediation" + "reasoning graph" + clinical; "faithful" + "editable" + diagnosis; clinical versions of Breaking the Chain / CSR / ITCR
- [ ] PubMed: `large language model AND (sycophancy OR faithfulness) AND diagnosis`; `script concordance AND language model`
- [ ] medRxiv health-informatics browse, last 6 months
- [ ] Citations of DiReCT, ER-Reason, MedCaseReasoning, You et al. 2603.27820, Position-Not-Provenance
