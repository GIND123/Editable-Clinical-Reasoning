# Causally Editable Clinical Reasoning (ClinBlueprint)

**Clinical language models whose diagnoses follow valid clinician corrections and resist invalid ones.**

ClinBlueprint is a small (4–8B), locally run clinical language model. Its diagnosis is computed *through* an explicit, typed reasoning graph that a clinician can edit. The graph holds observations quoted from the note, rationales, diagnoses from a diagnostic knowledge graph, and must-not-miss items. The project defines **causal editability**, a property that can be measured and trained for, and tests it on four axes:

| Axis | Question | How it is measured |
|---|---|---|
| 1. Mediation | Does a valid edit (a correction, or a new finding) move the diagnosis the way experts expect? | ClinEdit-Faith mediation rate against physician gold (DiReCT trees, SCT expert panels) |
| 2. Non-sycophancy | Are edits that contradict the record flagged rather than followed, and are irrelevant edits ignored? | Flag sensitivity, false-flag rate, sycophancy rate, invariance rate |
| 3. No bypass | Is the diagnosis computed through the graph, not around it? | Three-path decomposition, graph swap, activation patching, attribution graphs, SAE ablations, leakage probes |
| 4. Human benefit | Do clinicians catch and fix model errors better through the graph than through chain-of-thought text? | Randomised crossover study: graph vs CoT vs answer-only |

Target venue: *Nature Machine Intelligence* (Article). The research rationale, the novelty audit and the paper plan are in [`docs/`](docs/). If you are the clinician co-author, start with [`CLINICAL.md`](CLINICAL.md).

---

## Datasets

**Access legend.** **Public**: download directly. **PhysioNet**: credentialed access (PhysioNet account, CITI "Data or Specimens Only Research" training, signed DUA for each project). **DUA**: separate data-use agreement with the dataset owner.

**Compliance rule.** PhysioNet and DUA text is processed **only by local models**. It never goes to hosted APIs, LLM judges or simulators. [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) gives the details, and `clinblueprint.compliance.guard` enforces the rule in code.

### Core datasets (required)

| # | Dataset | Role in this project | Access | Links |
|---|---|---|---|---|
| 1 | **MIMIC-IV-Ext-DiReCT** (511 discharge notes with physician reasoning trees; BIDMC) | Gold reasoning graphs; supervised fine-tuning gold; 5-fold patient-level CV; edit gold (which evidence supports which diagnosis) | PhysioNet | Data: <https://physionet.org/content/mimic-iv-ext-direct/> (DOI [10.13026/yf96-kc87](https://doi.org/10.13026/yf96-kc87)) · Code + diagnostic knowledge graph (`diagnostic_kg/`, public): <https://github.com/wbw520/DiReCT> · Paper: <https://arxiv.org/abs/2408.01933> |
| 2 | **MedCaseReasoning** (14,489 PMC case reports with clinician reasoning; 13,092 train / 897 test) | Scale training (silver graphs); global external test (accuracy + reasoning recall); public half of the benchmark | Public (CC-BY 4.0) | Data: <https://huggingface.co/datasets/zou-lab/MedCaseReasoning> · Code: <https://github.com/kevinwu23/Stanford-MedCaseReasoning> · Paper: <https://arxiv.org/abs/2505.11733> |
| 3 | **ER-Reason** (UCSF ED: 25,174 notes, 3,984 encounters; 194 physician-authored SCT cases, 2,555 annotations) | Second-site validation; physician-gold belief-update direction and magnitude (DxUpdate, DxTrajectory) | PhysioNet (Contributor Review licence) | Data: <https://physionet.org/content/er-reason/> · Code: <https://github.com/AlaaLab/ER-Reason> · Paper: <https://arxiv.org/abs/2505.22919> |
| 4 | **SCT-Bench / concor.dance** (750 Script Concordance Test questions from 10 datasets; held-out leaderboard) | Public expert-panel belief-update benchmark | Public items + DUA for held-out set | Platform: <https://www.concor.dance/> · Paper: <https://www.medrxiv.org/content/10.1101/2025.02.11.25321822> · Comparative assessment: <https://www.medrxiv.org/content/10.64898/2026.03.13.26347597> |
| 5 | **MIMIC-IV-Ext-CDM** (2,400 abdominal-pathology cases) | Same-site transfer, static full-information setting only | PhysioNet | Data: <https://physionet.org/content/mimic-iv-ext-cdm/> · Code: <https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset> · Paper (Hager et al., *Nat Med* 2024): <https://www.nature.com/articles/s41591-024-03097-1> |
| 6 | **MIMIC-IV-Note** (discharge summaries) | Silver-graph training data, local teacher only; DiReCT patients excluded | PhysioNet | <https://physionet.org/content/mimic-iv-note/> |
| 7 | **MIMIC-IV** (hosp module) | ICD codes and admission links for MIMIC-IV-Note / CDM; linking DiReCT notes to subject IDs | PhysioNet | <https://physionet.org/content/mimiciv/> |

### Sanity suites (no-regression check)

| Dataset | Access | Link |
|---|---|---|
| MedQA (USMLE, 4 options) | Public | <https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options> |
| MedMCQA | Public | <https://huggingface.co/datasets/openlifescienceai/medmcqa> |
| PubMedQA (PQA-L) | Public | <https://huggingface.co/datasets/qiaojin/PubMedQA> |
| MedXpertQA (text subset) | Public | <https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA> |

### Additional datasets (from the PubMed and medRxiv literature search)

| Dataset | Why it helps | Access | Links |
|---|---|---|---|
| **MedDistractQA** (MedQA with Bystander / Non-literal distractors) | Seed library for *irrelevant* edits (the invariance test) | Public | Code: <https://github.com/nyuolab/MedDistractQA> · Paper: <https://arxiv.org/abs/2504.01201> |
| **MedNLI** (clinician-written NLI on MIMIC notes) | Pre-training the edit validator's entailment / contradiction verifier | PhysioNet | <https://physionet.org/content/mednli/> |
| **MultiCaRe** (96,428 cases from 75,382 PMC-OA articles) | Extra public scale for silver graphs. **Must be de-duplicated by PMCID against the MedCaseReasoning test set.** | Public (Zenodo; see paper) | Paper: <https://doi.org/10.1016/j.dib.2023.110008> |
| **PMC-Patients** (~167k patient summaries) | Extra public scale; same PMCID de-duplication | Public | <https://huggingface.co/datasets/zhengyun21/PMC-Patients> |
| **n2c2 2010 assertion** (present / absent / possible / hypothetical) | Optional assertion-status classifier for observation polarity and "hypothetical" edits | DUA (n2c2 portal) | <https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/> |
| **DDXPlus** (synthetic patients with ground-truth differentials) | Optional synthetic pre-training of the faithfulness loss where the exact counterfactual answer is known | Public | <https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374> |
| **AHRQ CCSR** (diagnosis groupings) | ER-Reason diagnosis evaluation (category-level match) | Public | <https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp> |
| **UMLS Metathesaurus** | Optional concept normalisation of observations | Licence (UTS) | <https://www.nlm.nih.gov/research/umls/> |

Published sycophancy protocols reused as external stress tests (not training data):
- "When helpfulness backfires", *npj Digit Med* 2025, [10.1038/s41746-025-02008-z](https://doi.org/10.1038/s41746-025-02008-z)
- "Too agreeable to be accurate?", *Artif Intell Med* 2026, [10.1016/j.artmed.2026.103516](https://doi.org/10.1016/j.artmed.2026.103516)

The complete search log, with queries and DOIs, is in [`docs/LITERATURE_SEARCH_LOG.md`](docs/LITERATURE_SEARCH_LOG.md).

### Models and interpretability tooling

| Item | Use | Link |
|---|---|---|
| Qwen3-4B | Primary backbone; circuit-tracer transcoders available | <https://huggingface.co/Qwen/Qwen3-4B> · transcoders: <https://www.neuronpedia.org/qwen3-4b> |
| MedGemma (Gemma-3 4B) | Second-family backbone; Gemma Scope 2 SAEs and transcoders | <https://huggingface.co/google/medgemma-4b-it> (check the MedGemma 1.5 ID) · <https://huggingface.co/google/gemma-scope-2-4b-it> · <https://huggingface.co/saillab/medgemma-circuit-tools> |
| Qwen3-8B | Scale check | <https://huggingface.co/Qwen/Qwen3-8B> |
| Qwen3-32B (4-bit) | Local teacher for silver graphs | <https://huggingface.co/Qwen/Qwen3-32B> |
| circuit-tracer | Attribution graphs | <https://github.com/safety-research/circuit-tracer> |
| SAELens | SAE loading and feature ablation | <https://github.com/jbloomAus/SAELens> |

Before downloading anything, confirm every model ID, revision and licence. Then pin the revisions in `configs/model/*.yaml`.

### Access checklist (the lead times that set the schedule)

1. Create a PhysioNet account and complete CITI "Data or Specimens Only Research" training, for **every** team member who will see MIMIC- or ER-Reason-derived text, including the clinician.
2. Request access on PhysioNet to MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-Ext-DiReCT, MIMIC-IV-Ext-CDM, ER-Reason and (optionally) MedNLI. ER-Reason uses the Contributor Review licence, so describe the study accurately.
3. Apply for SCT-Bench held-out access through concor.dance (DUA).
4. Download the public sets with `scripts/download/download_public.sh`, and the credentialed sets with `scripts/download/download_physionet.sh` (your PhysioNet credentials stay on your machine).

---

## Repository layout

```
src/clinblueprint/
  schema/       typed reasoning graph (Observation / Rationale / Diagnosis / MustNotMiss), JSON + slot serialisation
  kg/           DiReCT diagnostic knowledge graph, clinician must-not-miss lists
  checks/       hard checks: span grounding, KG validity, premise cascade, support, must-not-miss
  readout/      hard (deterministic / sign-constrained linear) and soft (slot-attention head) bottlenecks
  edits/        edit vocabulary (hypothetical vs assertion), graph operators, text adapter, validator, generator
  models/       backbone + LoRA loading, prompts, graph drafter, verbaliser, candidate scoring
  training/     graph SFT, readout fitting, validator, faithfulness losses, DPO, GRPO rewards
  eval/         accuracy (DiReCT metrics, reasoning recall), mediation, non-sycophancy, SCT, cluster bootstrap
  mech/         hooks, attention knockout, three-path decomposition, graph swap, patching, attribution, SAE, probes
  baselines/    CoT, note-level editing (You et al.-style), schema prompting, self-refine, best-of-N
  data/         loaders (DiReCT, MedCaseReasoning, ER-Reason, MIMIC-CDM, MIMIC-IV-Note, SCT, QA, PMC corpora), dedup, splits
  compliance/   local-only guard, small-cell suppression
configs/        model / data / train / eval YAML
scripts/        download, data preparation, pipeline runners, mechanistic tooling checks
benchmark/      ClinEdit-Faith specification, item schema, datasheet
clinician/      clinician tasks, templates and the crossover-study protocol
paper/          NMI manuscript skeleton, figure plan, references
docs/           idea dossier, architecture, datasets, training, evaluation, mechanism, compliance, decision gates
tests/          unit tests on synthetic fixtures (no patient data)
```

## Quickstart

```bash
# core (CPU): schema, checks, readouts, edits, metrics, tests
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

# GPU host (one 48 GB card): training + mechanistic extras
pip install -e ".[train,constrained,mech,study]"

# synthetic end-to-end demo: edit propagation, flagging, invariance
python examples/toy_demo.py
```

Pipeline stages (each reads one YAML config; see [`docs/TRAINING.md`](docs/TRAINING.md)):

```bash
clinblueprint prepare-data      --config configs/data/prepare.yaml
clinblueprint build-silver      --config configs/data/silver.yaml
clinblueprint train-sft         --config configs/train/sft_qwen3_4b.yaml
clinblueprint train-readout     --config configs/train/readout.yaml
clinblueprint train-validator   --config configs/train/validator.yaml
clinblueprint train-faithfulness --config configs/train/faithfulness.yaml
clinblueprint train-dpo         --config configs/train/dpo.yaml
clinblueprint build-benchmark   --config configs/eval/clinedit_faith.yaml
clinblueprint evaluate          --config configs/eval/evaluate.yaml
clinblueprint mech              --config configs/eval/mech.yaml
```

## Documentation

| File | Contents |
|---|---|
| [`CLINICAL.md`](CLINICAL.md) | For the clinician co-author: what is being built, why, and what we need from you |
| `docs/IDEA.md` | Full research dossier: problem, claims, prior art, honest novelty verdict |
| `docs/ARCHITECTURE.md` | Graph schema, bottleneck variants, validator, edit semantics |
| `docs/DATASETS.md` | Per-dataset access, formats, roles, leakage controls |
| `docs/EVALUATION.md` | Formal metric definitions and statistics |
| `docs/MECHANISTIC.md` | No-bypass methods and how they triangulate |
| `docs/TRAINING.md` | Stage-by-stage recipe and memory budget |
| `docs/DECISION_GATES.md` | Go/no-go gates and the paper pivots they trigger |
| `docs/COMPLIANCE.md` | PhysioNet / DUA rules and release policy |
| `benchmark/clinedit_faith/` | Benchmark specification |
| `clinician/study/` | Crossover-study protocol and pre-registration draft |
| `paper/` | NMI manuscript skeleton and figure plan |

## Licence

Code: MIT (see `LICENSE`). Datasets and model weights keep their own licences. Weights or graphs derived from MIMIC or ER-Reason are released only through PhysioNet's credentialed channel.

## Citation

A citation entry will be added on preprint release.
