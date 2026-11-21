# Datasets: access, formats, roles, leakage controls

The summary table with links is in the [README](../README.md#datasets). This file records **verified formats**, how each loader maps them, and the controls that keep evaluation clean. Formats marked *verified* were checked against the official repositories during this build. Anything marked **VERIFY** must be confirmed on first access.

## Directory layout (all git-ignored)
```
data/raw/direct_kg/                     DiReCT KG json (public, from GitHub)
data/raw/physionet/<project>/           credentialed downloads
data/raw/MedDistractQA/, ccsr/, multicare/, sct_bench/
data/processed/                         unified JSONL (clinblueprint prepare-data)
data/silver/                            teacher graphs (clinblueprint build-silver)
data/drafts/                            model-drafted graphs (clinblueprint draft-graphs)
data/benchmark/clinedit_faith/          cases + edits (clinblueprint build-benchmark)
```

## Core datasets

### MIMIC-IV-Ext-DiReCT: gold reasoning graphs (PhysioNet, doi:10.13026/yf96-kc87)
- **Format** (*verified*, github.com/wbw520/DiReCT):
  - Files are laid out as `<Disease Category>/<PDD>/<note>.json` (some notes sit directly under the category).
  - Keys `input1..input6` hold Chief Complaint, HPI, PMH, Family History, Physical Exam and Pertinent Results.
  - One nested tree has keys `"<content>$<Type>"`. `Intermedia_k` is a diagnosis: the top-level node is the final, most specific one, and nested children are more general. `Cause_k` is a rationale. `InputN` is an observation from section N.
  - Observation strings can be truncated fragments (e.g. "ltiple organisms consistent with …"), so grounding uses substring and fuzzy matching.
- **KG** (*verified*): one JSON per category, `{"diagnostic": nested tree with [] leaves, "knowledge": {node: premise str | {Risk Factors, Symptoms}}}`. There are 25 categories.
- **Loader**: `data/direct.py` → CaseRecord + ReasoningGraph (Intermedia → Diagnosis with specific-`refines`-general; Cause → Rationale; Input → Observation).
- **Splits**: 5-fold, patient-level, category-stratified (`data/splits.py`). **VERIFY** whether file names are MIMIC note_ids (`<subject_id>-DS-<n>`). If not, run `scripts/data/link_direct_subjects.py` against MIMIC-IV-Note.
- **Role**: SFT gold, oracle-graph regime, edit gold (key observations on the gold path), DiReCT metrics.

### MedCaseReasoning (public, CC-BY 4.0)
- **Format** (*verified*): HF `zou-lab/MedCaseReasoning`, with fields `case_prompt`, `diagnostic_reasoning` (numbered statements) and `final_diagnosis`. Metadata includes the PMC id (**VERIFY** field name; the loader tries `pmcid`, `PMCID`, `accession_id`, …).
- **Role**:
  - train → silver graphs (teacher sees case + reasoning + final dx);
  - test → external accuracy + reasoning recall;
  - test silver graphs → reference graphs for the *public* benchmark portion (never trained on).
- **Licences**: check each article's licence before redistributing derived graphs.

### ER-Reason (UCSF; PhysioNet, Contributor Review licence)
- **Files** (*verified*, github.com/AlaaLab/ER-Reason README):
  - `er_reason.csv`: per encounter, notes, `One_Sentence_Extracted`, `primaryeddiagnosisname`, acuity, disposition.
  - `icd_10_codes.csv`
  - `annotated_sct.csv`: `differential_1..5`, `evidence_1..5`.
  - `sct-annotations.csv`: the 194 SCT encounter keys.
  - `sct_cleaned_annotations.csv`: physician rationales.
  - `gt_clean.csv`: `dxupdate` −2..+2 and `dxtrajectory` (rank dict).
- **Metrics reproduced**:
  - DxUpdate: linear-weighted κ per timestep, averaged; exact and direction agreement.
  - DxTrajectory: Spearman ρ.
  - FinalDx: top-1 / top-3 accuracy and ρ.
- **Leakage**: the current ED provider note contains the assessment. The default input is the one-liner + *prior* notes; `meta.label_in_input` counts residual label strings. Match `final_diagnosis.py` before reporting accuracy.
- **Role**: second site; SCT is the primary ER-Reason endpoint (Claim A).

### SCT-Bench / concor.dance (public items + DUA held-out)
- The format is known only after access. Convert with `scripts/data/convert_sct_bench.py` to `SCTItem` (vignette, hypothesis, new information, panel counts −2..+2).
- **Scoring**: aggregate credit (n_panel(answer)/n_panel(modal)). The model's Likert answer comes from Δ log-odds via thresholds fitted on a *dev* split.

### MIMIC-IV-Ext-CDM (PhysioNet; 2,400 abdominal cases)
- **Files** (*verified*, ConvertPhysionet.py):
  - `history_of_present_illness.csv` (hpi), `physical_examination.csv` (pe)
  - `laboratory_tests.csv`, `microbiology.csv`, `radiology_reports.csv`
  - `discharge_diagnosis.csv`, `icd_diagnosis.csv`, `lab_test_mapping.csv`
  - `pathology_ids.json` (appendicitis, cholecystitis, diverticulitis, pancreatitis)
- **Role**: static full-information transfer only. No interactive test ordering (DEFER-Dx territory).

### MIMIC-IV-Note (PhysioNet)
- `note/discharge.csv.gz` (note_id, subject_id, hadm_id, text).
- `data/mimic_note.py` keeps the six DiReCT input sections. Discharge Diagnosis is the label only, and Brief Hospital Course is **teacher-only** context, never model input. DiReCT subjects are excluded.

## Additional datasets (PubMed / medRxiv search)
| Dataset | Use | Notes |
|---|---|---|
| MedDistractQA | Seeds for irrelevant edits | Bystander / non-literal distractors; the clinician vets context-dependence |
| MedNLI (PhysioNet) | Validator pre-training | MIMIC-III-derived; local only |
| MultiCaRe, PMC-Patients | Extra public silver data | **PMCID / PMID de-duplication against MedCaseReasoning test** (`data/dedup.py`), plus a near-duplicate MinHash audit |
| n2c2 2010 assertions | Optional polarity / hypothetical classifier | DUA |
| DDXPlus | Optional synthetic counterfactual pre-training (exact gold) | Synthetic; never used for headline claims |
| AHRQ CCSR | ER-Reason category-level match | Public |

## Contamination and leakage controls (all reported in the paper)
1. Patient-level folds; DiReCT subjects excluded from MIMIC silver data.
2. PMCID/PMID overlap removal between every training corpus and every evaluation split, plus a near-duplicate audit (`data/dedup.py`).
3. No label-bearing sections in model inputs (MIMIC-IV-Note, ER-Reason).
4. Teacher reasoning inputs (Brief Hospital Course, clinician rationales) are never model inputs.
5. Thresholds (SCT Likert mapping, validator τ, LTT τ) are fitted on dev folds only.
6. Same fixed candidate pool for every system.
