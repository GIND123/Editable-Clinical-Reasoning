# Causally Editable Clinical Reasoning (ClinBlueprint)

**Clinical language models whose diagnoses follow valid clinician corrections and resist invalid ones.**

ClinBlueprint is a small (4–8B), locally run clinical language model. Its diagnosis is computed *through* an explicit, typed reasoning graph that a clinician can edit. The graph holds observations quoted verbatim from the note, rationales, diagnoses from a diagnostic knowledge graph, and must-not-miss items. The project defines and tests **causal editability**, a property that can be measured and trained for, and ships the **ClinEdit-Faith** benchmark.

Target venue: *Nature Machine Intelligence* (Article). If you are the clinician co-author, start with **[CLINICAL.md](CLINICAL.md)**.

---

## Contents
1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [Contributions and novelty](#2-contributions-and-novelty)
3. [How it works](#3-how-it-works)
4. [Datasets (names, links, access)](#4-datasets)
5. [Models and interpretability tooling](#5-models-and-interpretability-tooling)
6. [ClinEdit-Faith benchmark](#6-clinedit-faith-benchmark)
7. [Metrics and targets](#7-metrics-and-targets)
8. [Baselines](#8-baselines)
9. [Mechanistic suite (no bypass)](#9-mechanistic-suite-no-bypass)
10. [Clinician crossover study](#10-clinician-crossover-study)
11. [Paper: figures and claims ledger](#11-paper-figures-and-claims-ledger)
12. [Decision gates and pivots](#12-decision-gates-and-pivots)
13. [Compliance and data governance](#13-compliance-and-data-governance)
14. [Installation and quickstart](#14-installation-and-quickstart)
15. [Full pipeline: commands, configs, outputs](#15-full-pipeline)
16. [Repository map (every file)](#16-repository-map)
17. [Implementation and test status](#17-implementation-and-test-status)
18. [Verification checklist (before real runs and submission)](#18-verification-checklist)
19. [Documentation index](#19-documentation-index)
20. [Licence and citation](#20-licence-and-citation)

---

## 1. The idea in one minute

**Problem.** Model reasoning text is often not the reason for the answer.
- Models verbalise the hints they actually used only 25–39% of the time.
- In one clinical audit, only 19% of cited symptoms drove the diagnosis.
- Models are sycophantic: they comply with false claims and flip correct answers under pressure.
- Reasoning-optimised models score *worse* than ordinary ones on Script Concordance Tests (SCT).
- Explanations have not protected clinicians from AI errors (Jabbour, *JAMA* 2023; Goh, *JAMA Netw Open* 2024; Qazi, *NEJM AI*).

**Property: causal editability**, tested on four axes:

| Axis | Question | Measured by |
|---|---|---|
| 1. Mediation | Does a valid edit move the diagnosis the way experts expect? The edit can be a correction or a new finding. | ClinEdit-Faith mediation rate vs physician gold (DiReCT trees, SCT expert panels) |
| 2. Non-sycophancy | Are edits that contradict the record flagged, and irrelevant edits ignored? | Flag sensitivity, false-flag rate, sycophancy rate, invariance rate |
| 3. No bypass | Is the diagnosis computed through the graph, not around it? | Three-path decomposition, graph swap, activation patching, attribution, SAE ablation, leakage probes |
| 4. Human benefit | Do clinicians catch and fix model errors better through the graph than through CoT text? | Randomised crossover study: graph vs CoT vs answer-only |

**Two goals.**
- **Science.** The property can be measured and trained, and same-size CoT models fail axes 1–3.
- **Model.** ClinBlueprint-4B/8B beats same-size baselines on DiReCT, MedCaseReasoning, ER-Reason and SCT.

The full dossier is in [docs/IDEA.md](docs/IDEA.md).

## 2. Contributions and novelty

| Claim | Status vs prior art |
|---|---|
| **Core.** A clinical LM trained so its diagnosis is causally mediated by its own editable graph, scored against physician-gold update directions at two hospitals | Defensibly novel (closest: Breaking the Chain, CSR, both non-clinical) |
| **A. Graph-mediated belief updating** matches SCT expert panels better than text updates (BIDMC, UCSF, SCT-Bench) | Novel as a trained, graph-mediated property (SCT as evaluation is not new) |
| **B. Corrigible but not sycophantic**: accept valid edits, flag invalid ones, ignore irrelevant ones; *hypothetical vs assertion* edits | Defensibly novel (Position-Not-Provenance is analysis only) |
| **Mechanism.** The bottleneck suppresses the pattern-matching bypass | Defensibly novel (the AAAI 2026 Gemma-2 study is descriptive) |
| **Human study.** Error correction via graph vs CoT vs answer-only | Defensibly novel as designed |
| **C (optional).** Node-local risk-gated repair with a Learn-then-Test guarantee | Secondary (ITCR stop-only; CaRDI question-level) |
| Not claimed | That graph structure improves accuracy (GraphDx, KGARevion, DocCHA, ECR-Agent, Causal-Audit); input-level counterfactual editing (You et al. 2603.27820) |

Design refinements this repository adds:
- the hypothetical-vs-assertion distinction;
- correction–counterfactual **asymmetry** as a bypass signature;
- the **bottleneck-strength spectrum**;
- **candidate-selection leakage** control;
- **three-path** information-flow decomposition;
- a **hard-readout teacher** for DPO;
- a **matched-data CoT-SFT** baseline.

Details: [docs/IDEA.md §6–7](docs/IDEA.md). Anticipated objections: [docs/REVIEWER_FAQ.md](docs/REVIEWER_FAQ.md).

## 3. How it works

```
note ──▶ drafter (LoRA LM, constrained JSON) ──▶ reasoning graph G ──▶ hard checks ──▶ readout(G) ──▶ differential
                                                    ▲                                     │
                    clinician edit ── validator ────┘ (assertion: flag if contradicted)   └─▶ verbaliser(G) ─▶ explanation
```

- **Graph schema** ([`schema/graph.py`](src/clinblueprint/schema/graph.py)), aligned with DiReCT:
  - `Observation(text, span, polarity, source, section)`
  - `Rationale(premises, claim, target, relation, strength)`
  - `Diagnosis(category, leaf, refines)`
  - `MustNotMiss(dx, status, evidence)`
  - Edges are derived from these fields.
- **Hard checks** ([`checks/hard_checks.py`](src/clinblueprint/checks/hard_checks.py)): verbatim span grounding; KG validity; a premise cascade that prunes invalid rationales (this is how edits propagate); a support check; must-not-miss coverage.
- **Readouts: the bottleneck-strength spectrum.**
  - Hard: deterministic evidence-path score ([`readout/hard.py`](src/clinblueprint/readout/hard.py)).
  - Hard-linear: sign-constrained conditional logit ([`readout/linear.py`](src/clinblueprint/readout/linear.py)).
  - Soft-isolated and soft-in-context: a slot-attention head ([`readout/soft.py`](src/clinblueprint/readout/soft.py)).
  - In-context generative: DPO-trained ([`training/dpo.py`](src/clinblueprint/training/dpo.py)).
  - The note never enters the hard readouts. The drafter's self-reported probability is ignored except in an ablation. Candidates include every KG leaf of every touched category, to block candidate-selection leakage.
- **Edits** ([`edits/`](src/clinblueprint/edits/)):
  - Operations (from You et al.): negate, remove, replace, weaken, intensify, insert.
  - Modes: **hypothetical** (what-if; always answered) and **assertion** (a claim about the patient; validated).
  - Classes: valid_correction, counterfactual, new_finding, irrelevant, invalid.
  - Reason modes: frozen, and re_reason (the model links new findings itself).
  - A flag is not a refusal: overrides are applied and logged.
- **Validator** ([`edits/validator.py`](src/clinblueprint/edits/validator.py)): NegEx-style polarity and numeric-contradiction rules, plus a learned NLI verifier with isotonic calibration ([`training/validator_train.py`](src/clinblueprint/training/validator_train.py)).
- **Drafter** ([`models/drafter.py`](src/clinblueprint/models/drafter.py)): two-stage (KG category selection, then graph); emits evidence first; optional constrained decoding. Re-reasoning hooks: `link_new_observation`, `ground_or_repair`.
- **Verbaliser** ([`models/verbalizer.py`](src/clinblueprint/models/verbalizer.py)): sees only the graph and the readout, never the note; an audit flags invented ids and numbers.
- **Claim C** ([`risk/`](src/clinblueprint/risk/)): a node-risk gate plus Learn-then-Test (Hoeffding–Bentkus).

Full specification: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 4. Datasets

**Access legend.**
- **Public**: download directly.
- **PhysioNet**: credentialed (account + CITI "Data or Specimens Only Research" + signed DUA per project).
- **DUA**: separate agreement with the dataset owner.

**Rule:** PhysioNet / DUA text is processed **only by local models**, enforced in code by [`compliance/guard.py`](src/clinblueprint/compliance/guard.py).

### 4.1 Core datasets (required)

| # | Dataset | Role | Access | Links |
|---|---|---|---|---|
| 1 | **MIMIC-IV-Ext-DiReCT** (511 discharge notes with physician reasoning trees; BIDMC) | Gold graphs; SFT gold; 5-fold patient-level CV; edit gold; DiReCT metrics | PhysioNet | Data: <https://physionet.org/content/mimic-iv-ext-direct/> (DOI [10.13026/yf96-kc87](https://doi.org/10.13026/yf96-kc87)) · Code + public KG: <https://github.com/wbw520/DiReCT> · Paper: <https://arxiv.org/abs/2408.01933> |
| 2 | **MedCaseReasoning** (14,489 PMC case reports with clinician reasoning; 13,092 / 897) | Silver-graph training; global external test (accuracy + reasoning recall); public benchmark portion; clinician-study cases | Public (CC-BY 4.0) | Data: <https://huggingface.co/datasets/zou-lab/MedCaseReasoning> · Code: <https://github.com/kevinwu23/Stanford-MedCaseReasoning> · Paper: <https://arxiv.org/abs/2505.11733> |
| 3 | **ER-Reason** (UCSF ED: 25,174 notes / 3,984 encounters; 194 SCT cases, 2,555 physician annotations) | Second site; physician-gold belief-update direction and magnitude (DxUpdate, DxTrajectory, FinalDx) | PhysioNet (Contributor Review licence; v1.1.0 listed) | Data: <https://physionet.org/content/er-reason/> · Code: <https://github.com/AlaaLab/ER-Reason> · Paper: <https://arxiv.org/abs/2505.22919> |
| 4 | **SCT-Bench / concor.dance** (750 SCT questions, 10 datasets; held-out leaderboard) | Public expert-panel belief-update benchmark | Public items + DUA held-out | <https://www.concor.dance/> · Paper: <https://www.medrxiv.org/content/10.1101/2025.02.11.25321822> · Comparative assessment: <https://www.medrxiv.org/content/10.64898/2026.03.13.26347597> |
| 5 | **MIMIC-IV-Ext-CDM** (2,400 abdominal-pathology cases) | Same-site transfer, static full-information only | PhysioNet | <https://physionet.org/content/mimic-iv-ext-cdm/> · Code: <https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset> · Paper: <https://www.nature.com/articles/s41591-024-03097-1> |
| 6 | **MIMIC-IV-Note** (discharge summaries) | Silver graphs (local teacher only; DiReCT patients excluded) | PhysioNet | <https://physionet.org/content/mimic-iv-note/> |
| 7 | **MIMIC-IV** (hosp module) | ICD codes and admission links; DiReCT subject linking | PhysioNet | <https://physionet.org/content/mimiciv/> |

### 4.2 QA sanity suites (no-regression check)

| Dataset | Access | Link |
|---|---|---|
| MedQA (USMLE, 4 options) | Public | <https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options> |
| MedMCQA | Public | <https://huggingface.co/datasets/openlifescienceai/medmcqa> |
| PubMedQA (PQA-L) | Public | <https://huggingface.co/datasets/qiaojin/PubMedQA> |
| MedXpertQA (text) | Public | <https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA> |

### 4.3 Additional datasets (from the PubMed / medRxiv search)

| Dataset | Why | Access | Links |
|---|---|---|---|
| **MedDistractQA** | Seed library for *irrelevant* edits | Public | <https://github.com/nyuolab/MedDistractQA> · <https://arxiv.org/abs/2504.01201> |
| **MedNLI** | Validator NLI pre-training | PhysioNet | <https://physionet.org/content/mednli/> |
| **MultiCaRe** (96,428 PMC cases) | Extra public silver data. **De-duplicate by PMCID against the MedCaseReasoning test set.** | Public (Zenodo) | <https://doi.org/10.1016/j.dib.2023.110008> |
| **PMC-Patients** (~167k summaries) | Extra public silver data (same de-duplication) | Public | <https://huggingface.co/datasets/zhengyun21/PMC-Patients> |
| **n2c2 2010 assertions** | Optional polarity / hypothetical classifier | DUA | <https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/> |
| **DDXPlus** | Optional synthetic counterfactual pre-training (exact gold) | Public | <https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374> |
| **AHRQ CCSR** | ER-Reason category-level diagnosis match | Public | <https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp> |
| **UMLS** | Optional concept normalisation | Licence | <https://www.nlm.nih.gov/research/umls/> |

External sycophancy protocols (stress tests, not training data):
- [10.1038/s41746-025-02008-z](https://doi.org/10.1038/s41746-025-02008-z)
- [10.1016/j.artmed.2026.103516](https://doi.org/10.1016/j.artmed.2026.103516)

Search log with all DOIs: [docs/LITERATURE_SEARCH_LOG.md](docs/LITERATURE_SEARCH_LOG.md).

### 4.4 Access checklist (these lead times set the schedule)
1. Everyone who will see credentialed text, the clinician included, needs a PhysioNet account plus CITI "Data or Specimens Only Research" training.
2. Request access to MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-Ext-DiReCT, MIMIC-IV-Ext-CDM, ER-Reason (Contributor Review: describe the study accurately) and, optionally, MedNLI.
3. Apply for SCT-Bench held-out access through concor.dance (DUA).
4. Accept the model licences on Hugging Face (MedGemma / Gemma are gated).
5. Run `bash scripts/download/download_public.sh`, then `PHYSIONET_USER=… bash scripts/download/download_physionet.sh`, then `python scripts/check_access.py`.

### 4.5 Leakage and contamination controls
Verified formats and loader mappings are in [docs/DATASETS.md](docs/DATASETS.md). The controls:
1. Patient-level folds.
2. DiReCT subjects excluded from MIMIC silver data.
3. PMCID / PMID overlap removal plus a MinHash near-duplicate audit ([`data/dedup.py`](src/clinblueprint/data/dedup.py)).
4. No label-bearing sections in model inputs. ER-Reason uses the one-liner plus *prior* notes, and `label_in_input` is counted.
5. Teacher-only context (clinician rationales, Brief Hospital Course) never reaches the model.
6. Thresholds are fitted on dev folds only.
7. Every system scores the same fixed candidate pool.

## 5. Models and interpretability tooling

| Item | Use | Link |
|---|---|---|
| Qwen3-4B | Primary backbone; circuit-tracer transcoders | <https://huggingface.co/Qwen/Qwen3-4B> · <https://www.neuronpedia.org/qwen3-4b> |
| MedGemma 4B (Gemma-3) | Second family; Gemma Scope 2 | <https://huggingface.co/google/medgemma-4b-it> (VERIFY the MedGemma 1.5 id) · <https://huggingface.co/google/gemma-scope-2-4b-it> · <https://huggingface.co/saillab/medgemma-circuit-tools> |
| Qwen3-8B | Scale check | <https://huggingface.co/Qwen/Qwen3-8B> |
| Qwen3-32B (4-bit) | Local teacher for silver graphs | <https://huggingface.co/Qwen/Qwen3-32B> |
| circuit-tracer | Attribution graphs | <https://github.com/safety-research/circuit-tracer> |
| SAELens | SAEs | <https://github.com/jbloomAus/SAELens> |
| External comparators (published or local) | MedGemma-27B (4-bit), HuatuoGPT-o1-8B, m1-7B, MedReason-8B | HF model pages (VERIFY ids) |

Configs: [`configs/model/`](configs/model/). Pin revisions before the first real run.

## 6. ClinEdit-Faith benchmark

Each item is (case with a fixed candidate pool, edit, gold expected effect):

| Class | Mode | Construction | Gold |
|---|---|---|---|
| valid_correction | assertion | Physician-tree evidence the model's graph lacks | increase |
| counterfactual | hypothetical | Remove / negate / weaken / intensify key evidence on the gold path | decrease / increase |
| new_finding | assertion | SCT new information (ER-Reason, SCT-Bench); the model links it itself | panel sign and magnitude |
| irrelevant | assertion | Clinician-vetted bystander facts | no_change |
| invalid | assertion | Contradicts a finding the record states; numeric swap | flag |

**Portions.**
- Credentialed: DiReCT + ER-Reason, released via PhysioNet only.
- Public: MedCaseReasoning test, open release.

About 500 stratified items are physician-verified (with κ). Schemas are generated from the code ([`benchmark/clinedit_faith/schema/`](benchmark/clinedit_faith/schema/)). Specification: [benchmark/clinedit_faith/README.md](benchmark/clinedit_faith/README.md); datasheet: [DATASHEET.md](benchmark/clinedit_faith/DATASHEET.md).

**Evaluation regimes.**
- **Oracle graph**: isolates the readout.
- **Model graph**: edits are retargeted by text; coverage is reported.
- **Trace**: CoT trace intervention.
- **Note**: input-level editing.

## 7. Metrics and targets

| Axis | Metrics ([docs/EVALUATION.md](docs/EVALUATION.md)) | Target [E] |
|---|---|---|
| Accuracy | DiReCT accuracy + completeness/faithfulness (`comp_pre/re/coverage`, `faith_ob/all`); MedCaseReasoning accuracy (local judge) + reasoning recall; ER-Reason (ICD/CCSR); MIMIC-CDM; QA sanity | DiReCT ≥ 0.61, "all" ≥ 0.30; MCR ≥ rationale-SFT + 3 (stretch ≥ 48% at 8B); QA within 2 points |
| Mediation | Mediation rate, strong MR, **asymmetry** (MR_correction − MR_counterfactual), opposite rate, coverage | ≥ 0.90 hard / ≥ 0.80 soft vs CoT ~0.3–0.7 |
| Non-sycophancy | Flag sensitivity, false-flag rate, sycophancy rate, AUROC, operating curve, invariance | Sensitivity ≥ 0.80, false-flag ≤ 0.10, invariance ≥ 0.95 |
| Belief updating | SCT aggregate credit, direction accuracy, ER-Reason DxUpdate κ / DxTrajectory ρ / FinalDx | Paired CI (graph − text) excludes 0 at each site |
| Mechanism | Direct-note path share, hidden surplus, flip-to-graph rate, slot patch recovery, attribution share, SAE Δ, probe selectivity | Methods agree in direction (G4) |
| Human study | Detection rate (GEE odds ratio A vs B), post-correction accuracy, time, automation bias, trust calibration | Pre-registered |

Statistics: case/patient-level cluster bootstrap (2,000 resamples); paired bootstrap for system differences; small-cell suppression (< 10). Published reference numbers: [docs/EVALUATION.md §7](docs/EVALUATION.md).

## 8. Baselines

All baselines use the same backbone unless noted and run locally ([`baselines/`](src/clinblueprint/baselines/), [`configs/eval/evaluate.yaml`](configs/eval/evaluate.yaml)):

1. Free-CoT SFT on the **same cases** ([`training/cot_sft.py`](src/clinblueprint/training/cot_sft.py))
2. Rationale SFT (MedCaseReasoning-style; same module using clinician text)
3. Schema-guided prompting (untuned backbone + drafter prompt + hard readout)
4. Self-refine
5. Best-of-N with a verifier
6. Counterfactual multi-agent, You et al.-style (input-level, training-free)
7. Note-level editing for every model
8. Oracle-graph upper bound
9. ITCR-style stop-only (Claim C)
10. External open models (MedGemma-27B 4-bit, HuatuoGPT-o1-8B, m1-7B, MedReason-8B)
11. Closed models: published numbers only

## 9. Mechanistic suite (no bypass)

Code: [`mech/`](src/clinblueprint/mech/). Details: [docs/MECHANISTIC.md](docs/MECHANISTIC.md).

| Method | Mediated signature |
|---|---|
| Three-path decomposition (note→answer attention knockout, graph-only, note-only) | Small direct-note path; small hidden surplus |
| Graph swap (note_A + graph_B) | Follows graph_B |
| Minimal-pair activation patching (one polarity flip, layer by layer) | High slot recovery; ~0 for the note control |
| Integrated gradients + circuit-tracer influence by segment | Attribution mostly on graph tokens |
| SAE disease-feature ablation (slots vs note) | Slot-side ablation hurts more |
| Leakage probes with control tasks | Near-control selectivity |

Comparison: in-context DPO ClinBlueprint vs matched CoT-SFT vs the untuned backbone, against the pattern-matching vs reasoning pathway taxonomy of the AAAI 2026 study. Run [`scripts/mech/check_tooling.py`](scripts/mech/check_tooling.py) first (null knockout, self-patch, transcoder/SAE loading).

## 10. Clinician crossover study

- **Arms.** A = editable graph; B = CoT text; C = answer only.
- **Design.** Williams design over 12–20 clinicians; ~120 public MedCaseReasoning cases, stratified by natural model correctness. No planted errors.
- **Primary outcome.** Error detection, analysed with GEE logistic regression (arm + period, exchangeable within clinician).
- **Tooling.** Randomisation, power simulation and analysis are in [`study/design.py`](src/clinblueprint/study/design.py).
- **Commands.** `clinblueprint study-power --config configs/eval/study_power.yaml` and `clinblueprint study-randomize --config <yaml>`.
- **Documents.** [protocol](clinician/study/protocol.md), [pre-registration](clinician/study/preregistration.md), [participant instructions](clinician/study/rater_instructions.md).

Clinician tasks (must-not-miss lists, silver audit, edit verification, bystander facts, validator adjudication) are in [`clinician/`](clinician/).

## 11. Paper: figures and claims ledger

- **Fig. 1** concept and architecture
- **Fig. 2** accuracy
- **Fig. 3** mediation and non-sycophancy (incl. the bottleneck-strength frontier)
- **Fig. 4** SCT at two hospitals + SCT-Bench
- **Fig. 5** mechanism
- **Fig. 6** clinician study
- **Extended Data**: ablations, stress tests, Claim C, leakage probes, failure cases

Plan: [paper/FIGURE_PLAN.md](paper/FIGURE_PLAN.md). Claim → experiment → config → metric → figure: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). Manuscript skeleton with a draft abstract: [paper/main.tex](paper/main.tex); bibliography: [paper/references.bib](paper/references.bib); presubmission enquiry: [paper/presubmission_enquiry.md](paper/presubmission_enquiry.md).

## 12. Decision gates and pivots

Details: [docs/DECISION_GATES.md](docs/DECISION_GATES.md).

| Gate | Criterion | Consequence |
|---|---|---|
| G0 | Problem is real: baseline MR ≤ 0.7, poor SCT concordance, or asymmetry > 0 | Otherwise reframe |
| G1 | Accuracy ≥ CoT-SFT − 2 | Fail → **Pivot A**: NMI Analysis "Clinical LMs are not causally editable" (20+ models, two hospitals, mechanism, benchmark) |
| G2 | Validator AUROC ≥ 0.80 | Fail → rules-only validator; trade-off framing |
| G3 | Soft mediation ≥ 0.80 with non-sycophancy targets | Fail → **Pivot B**: lead with hard bottleneck + validator + mechanism + study |
| G4 | Mechanistic methods agree | Fail → report disagreement; behavioural claim only |
| G-C / G-RL | Claim C improves the front / GRPO needed (> 3-point gap) | Include or drop / run or skip |

## 13. Compliance and data governance

Details: [docs/COMPLIANCE.md](docs/COMPLIANCE.md).
- Local-only processing of credentialed text; `assert_local` is enforced before every model call.
- Per-person PhysioNet credentialing.
- `data/` and `outputs/` are never committed.
- Local judges only; small-cell suppression.
- MIMIC- and ER-Reason-derived weights and benchmark portions are released only via PhysioNet.
- An IRB / exempt determination is required for the human study.

## 14. Installation and quickstart

```bash
git clone https://github.com/GIND123/Editable-Clinical-Reasoning && cd Editable-Clinical-Reasoning
python -m venv .venv && source .venv/bin/activate          # Python >= 3.10
pip install -e ".[dev]"                                    # core: CPU only
pytest                                                     # 35 tests on synthetic fixtures
python examples/toy_demo.py                                # edit propagation / flag / override / invariance

# GPU host (one 48 GB card)
pip install -e ".[train,constrained,mech,study,dev]"       # + optional: .[rl] (TRL), .[serve] (vLLM)
uv pip compile pyproject.toml --extra train -o requirements.lock   # freeze the working environment
```

`make install | install-gpu | test | demo | schemas | check-access | download-public | lint` are shortcuts.

## 15. Full pipeline

Every stage reads one YAML (`--set key=value` overrides; `defaults:` includes). Stages run per backbone and per DiReCT fold. [`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) chains them.

| # | Command | Config | Produces |
|---|---|---|---|
| 0 | `python scripts/check_access.py` | – | Presence report |
| 0b | `python scripts/data/link_direct_subjects.py …` | – | `data/processed/direct_subject_map.json` |
| 1 | `clinblueprint prepare-data` | `configs/data/prepare.yaml` | `data/processed/*.jsonl`, `direct.folds.json`, `prepare_report.json` |
| 2 | `clinblueprint build-silver` (×3: MCR train, MCR test reference, MIMIC-IV-Note) | `configs/data/silver.yaml` | `data/silver/*.silver.jsonl` + stats |
| 3 | `clinblueprint train-sft` | `configs/train/sft_qwen3_4b.yaml` | `outputs/sft/<run>/adapter` |
| 4 | `clinblueprint train-cot-sft` | `configs/train/cot_sft.yaml` | `outputs/cot_sft/<run>/adapter` |
| 5 | `clinblueprint draft-graphs` | `configs/eval/draft_graphs.yaml` | `data/drafts/*.jsonl` |
| 6 | `clinblueprint train-readout` | `configs/train/readout.yaml` | `linear_readout.json`, fit report |
| 7 | `clinblueprint train-validator` | `configs/train/validator.yaml` | Verifier adapter + `calibration.json` |
| 8 | `clinblueprint train-faithfulness` | `configs/train/faithfulness.yaml` | `soft_head.pt` |
| 9 | `clinblueprint train-dpo` | `configs/train/dpo.yaml` | In-context DPO adapter |
| 10 | `clinblueprint build-benchmark` | `configs/eval/clinedit_faith.yaml` | `data/benchmark/clinedit_faith/{portion}.{cases,edits,verify_sample}.jsonl` |
| 11 | `python scripts/export_audit_sheets.py silver/edits …` | – | Clinician review CSVs |
| 12 | `clinblueprint evaluate` | `configs/eval/evaluate.yaml` | Per-system `*.edits.jsonl`, `*.summary.json`, accuracy files |
| 12b | `clinblueprint evaluate-sct` | `configs/eval/sct.yaml` | Sequential SCT rows + DxUpdate κ / direction / DxTrajectory / final top-1 (Claim A) |
| 13 | `clinblueprint mech` | `configs/eval/mech.yaml` | Three-path, graph-swap, patching and attribution results |
| 14 | `clinblueprint study-power` / `study-randomize` | `configs/eval/study_power.yaml` | Power estimate / schedule CSV |
| – | `clinblueprint demo` | – | Toy demo |

Memory budget, hyper-parameters and fold protocol: [docs/TRAINING.md](docs/TRAINING.md).

## 16. Repository map

```
README.md                    this file (single entry point)
CLINICAL.md                  guide for the clinician co-author
LICENSE · CITATION.cff · Makefile · pyproject.toml · .env.example · .gitignore (data/ and outputs/ never committed)

src/clinblueprint/
  schema/graph.py            typed graph, dx keys, integrity checks, human-readable summary
  schema/serialize.py        drafter JSON schema, tolerant JSON parsing, slot-text serialisation with char spans
  kg/direct_kg.py            DiReCT diagnostic KG (tree, premises, ancestors, leaves, canonical names)
  kg/must_not_miss.py        clinician must-not-miss loader (DRAFT / APPROVED gate)
  checks/hard_checks.py      grounding, KG validity, premise cascade, support, MNM coverage
  readout/differential.py    probability distribution over dx keys (p, logit, rank, TV)
  readout/hard.py            deterministic evidence-path readout (structural mediation)
  readout/linear.py          sign-constrained conditional-logit readout (+ bypass-feature ablation)
  readout/soft.py            slot encoder, slot-attention head, SoftReadout
  edits/types.py             Edit / modes / classes / expected effects
  edits/ops.py               graph edit operators
  edits/validator.py         rule checker (NegEx-style, numeric) + learned verifier interface
  edits/generator.py         ClinEdit-Faith edit generation (all five classes, SCT items)
  edits/text_adapter.py      the same edits applied to CoT traces / notes
  edits/align.py             retargeting benchmark edits onto model graphs
  pipeline.py                ClinBlueprint (run / intervene / apply_clinician_edit / override), StaticDrafter
  models/backbone.py         HF + PEFT loading, 4/8-bit, chat templates, generation, constrained JSON
  models/prompts.py          every prompt template (versioned)
  models/drafter.py          two-stage LM graph drafter + re-reasoning hooks
  models/verbalizer.py       graph-only verbaliser + audit
  models/scoring.py          candidate log-likelihood scoring
  models/draft_corpus.py     batch drafting of graphs
  training/examples.py       SFT example builders, template explanations, tokenisation
  training/sft.py            graph SFT (drafter + verbaliser)
  training/cot_sft.py        matched-data CoT-SFT baseline
  training/readout_train.py  linear readout fitting
  training/validator_train.py  NLI verifier training + isotonic calibration + HFEntailmentVerifier
  training/losses.py         direction hinge, invariance KL, DPO, sequence log-prob
  training/faithfulness.py   soft-head training with counterfactual / invariance losses
  training/dpo.py            DPO with hard-readout teacher and adapter-disabled reference
  training/rewards.py        verifiable rewards for optional GRPO
  eval/accuracy.py           dx matching, DiReCT metrics, reasoning recall
  eval/judge.py              local LLM judge (compliance-guarded)
  eval/mediation.py          mediation, strong mediation, asymmetry, invariance
  eval/nonsycophancy.py      flag sensitivity, false-flag rate, sycophancy rate, AUROC, operating points
  eval/sct.py                SCT credit, Likert mapping, weighted κ, DxUpdate, DxTrajectory
  eval/bootstrap.py          cluster and paired bootstrap
  eval/calibration.py        ECE, Brier, reliability
  eval/runner.py             benchmark + accuracy runner, summaries with CIs
  eval/sct_runner.py         sequential SCT evaluation (findings accumulate across timesteps; dev-fitted Likert thresholds)
  eval/system_factory.py     YAML → systems (ClinBlueprint variants, CoT, note-edit, in-context, CF multi-agent)
  baselines/systems.py       EditableSystem adapters
  baselines/refine_and_sample.py   self-refine, best-of-N
  baselines/counterfactual_multiagent.py   You et al.-style re-implementation
  mech/hooks.py · segments.py · knockout.py · three_path.py · swap_and_patch.py · attribution.py · sae_probes.py · run.py
  data/records.py · direct.py · medcasereasoning.py · er_reason.py · mimic_cdm.py · mimic_note.py · sct.py
  data/qa_suites.py · pmc_cases.py · dedup.py · splits.py · silver.py · prepare.py
  benchmark/build.py         candidate pools, benchmark files, physician-verification sample
  risk/node_risk.py · risk/ltt.py   Claim C
  study/design.py            Williams randomisation, power simulation, GEE analysis
  compliance/guard.py        local-only guard, small-cell suppression
  utils/config.py · utils/io.py
  cli.py                     `clinblueprint <command>`

configs/model/   qwen3_4b · medgemma_4b · qwen3_8b · qwen3_32b_teacher
configs/data/    prepare · silver
configs/train/   sft_qwen3_4b · cot_sft · readout · validator · faithfulness · dpo
configs/eval/    clinedit_faith · evaluate · sct · draft_graphs · mech · study_power
configs/schema/must_not_miss.yaml       DRAFT for clinician sign-off
configs/benchmark/bystander_facts.yaml  DRAFT for clinician sign-off

scripts/download/   download_public.sh · download_physionet.sh · download_models.sh
scripts/data/       link_direct_subjects.py · convert_sct_bench.py
scripts/mech/       check_tooling.py
scripts/            check_access.py · export_audit_sheets.py · run_pipeline.sh
examples/toy_demo.py                    synthetic end-to-end demo
tests/                                  35 unit tests + synthetic fixtures (DiReCT-format case, toy KG)
benchmark/clinedit_faith/               spec, datasheet, JSON schemas
clinician/                              task instructions (01–05), study protocol, pre-registration, rater instructions
paper/                                  main.tex, references.bib, FIGURE_PLAN.md, presubmission_enquiry.md
docs/                                   IDEA, ARCHITECTURE, DATASETS, TRAINING, EVALUATION, MECHANISTIC,
                                        DECISION_GATES, COMPLIANCE, EXPERIMENTS, REVIEWER_FAQ, LITERATURE_SEARCH_LOG
data/ · outputs/                        git-ignored (README placeholders only)
```

## 17. Implementation and test status

| Status | Components |
|---|---|
| **Implemented and unit-tested on CPU** (35 tests, synthetic data) | Graph schema and serialisation; DiReCT parser; KG; hard checks and cascade; hard and linear readouts (monotonicity, drafter-p ignored, sign constraints); edit operators; validator rules (polarity, numeric, new information); hypothetical / assertion / override pipeline; edit generator; text adapter; retargeting; mediation, non-sycophancy, SCT, bootstrap and DiReCT metrics; oracle-graph benchmark run end to end; sequential SCT driver; compliance guard; patient-level folds; Learn-then-Test; study randomisation and power; GRPO rewards |
| **Implemented, needs a GPU run and real data** | Backbone loading; drafter; verbaliser; candidate scoring; graph SFT; CoT-SFT; validator training; faithfulness head; DPO; silver generation; draft corpus; CoT, note-edit, in-context and counterfactual multi-agent systems; local LLM judge; full mechanistic suite; real-dataset loaders (formats verified against official repositories but not yet run on downloaded files) |
| **To build** | Clinician-study web interface (interactive graph editing; the participant instructions describe it); figure scripts (`paper/figures/make_fig*.py`); TRL GRPO wrapper (rewards exist); ITCR stop-only baseline; CoT text-update comparator for sequential SCT (ER-Reason `sct.py` protocol run with a local model) |

## 18. Verification checklist

Do these before real runs and before submission:

**Data**
- [ ] PhysioNet versions and folder names (DiReCT 1.0.0, ER-Reason 1.1.0, CDM 1.1, MIMIC-IV-Note 2.2, MIMIC-IV 3.1, MedNLI) in `download_physionet.sh` / `prepare.yaml`.
- [ ] Whether DiReCT file names carry note ids. If not, run `link_direct_subjects.py`.
- [ ] MedCaseReasoning PMCID field name; HF split loading.
- [ ] Match ER-Reason's `final_diagnosis.py` input protocol; check the Contributor Review licence terms.
- [ ] MIMIC-CDM `lab_test_mapping.csv` label column.
- [ ] SCT-Bench file format (`convert_sct_bench.py` column mapping).
- [ ] MultiCaRe column names.

**Models and tooling**
- [ ] MedGemma 1.5 4B model id and licence; pin all model revisions.
- [ ] circuit-tracer transcoder set names and graph API fields; Gemma Scope 2 SAELens release / `sae_id`; attention-knockout support in the installed transformers (`check_tooling.py`).

**Results and literature**
- [ ] SCT reference numbers (o3 / o1-preview / R1) against the primary paper.
- [ ] Read the remaining ~25% of You et al. 2603.27820 and align the re-implementation.
- [ ] Every `note={VERIFY}` entry in `paper/references.bib`.
- [ ] Re-run the novelty search ([docs/LITERATURE_SEARCH_LOG.md](docs/LITERATURE_SEARCH_LOG.md#before-submission-re-run-checklist)).

**Clinician sign-off**
- [ ] `must_not_miss.yaml` and `bystander_facts.yaml` from DRAFT to APPROVED.

## 19. Documentation index

| File | Contents |
|---|---|
| [CLINICAL.md](CLINICAL.md) | For the clinician co-author |
| [docs/IDEA.md](docs/IDEA.md) | Research dossier: problem, prior art, novelty verdict, claims, targets, risks |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Schema, pipeline, readouts, edits, validator, drafter, Claim C |
| [docs/DATASETS.md](docs/DATASETS.md) | Verified formats, loaders, roles, leakage controls |
| [docs/TRAINING.md](docs/TRAINING.md) | Stage-by-stage recipe, memory budget, folds, seeds |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Regimes, metric definitions, statistics, baselines, reference numbers |
| [docs/MECHANISTIC.md](docs/MECHANISTIC.md) | No-bypass methods, signatures, controls, tooling |
| [docs/DECISION_GATES.md](docs/DECISION_GATES.md) | Gates G0–G4, G-C, G-RL and pivots |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Claims ledger, ablations, stress tests |
| [docs/REVIEWER_FAQ.md](docs/REVIEWER_FAQ.md) | Anticipated objections |
| [docs/COMPLIANCE.md](docs/COMPLIANCE.md) | Data governance and releases |
| [docs/LITERATURE_SEARCH_LOG.md](docs/LITERATURE_SEARCH_LOG.md) | PubMed / medRxiv / GitHub verification log with DOIs |
| [benchmark/clinedit_faith/](benchmark/clinedit_faith/) | Benchmark spec, datasheet, schemas |
| [clinician/](clinician/) | Clinician tasks and study documents |
| [paper/](paper/) | Manuscript skeleton, figure plan, bibliography, presubmission enquiry |

## 20. Licence and citation

- **Code:** MIT ([LICENSE](LICENSE)). Datasets and model weights keep their own licences; artefacts derived from MIMIC or ER-Reason are released only through PhysioNet's credentialed channel.
- **Citation:** see [CITATION.cff](CITATION.cff). A paper citation will be added on preprint release.
