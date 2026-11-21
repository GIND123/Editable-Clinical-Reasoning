# ClinEdit-Faith: a benchmark for causal editability of clinical reasoning

ClinEdit-Faith tests whether a clinical reasoning system:
- **follows** valid edits to its reasoning in the direction experts expect;
- **flags** asserted edits that contradict the record;
- **ignores** irrelevant ones.

Any system can be scored. Graph-based systems take the edit on their graph; CoT systems take it on their trace; any model can take it on the input note.

## Item = (case, edit, gold effect)
- **Case** (`schema/case.schema.json`): the note, sections and offsets, the gold diagnosis, the **fixed candidate pool** (shared by all systems), and the reference graph.
- **Edit** (`schema/edit.schema.json`), with these fields:
  - `op`: negate | remove | replace | weaken | intensify | insert
  - `mode`: hypothetical | assertion
  - `edit_class`: valid_correction | counterfactual | new_finding | irrelevant | invalid
  - `target_node` / `target_text`, plus `new_text` / `links` for inserts
  - `measured_dx`
  - `expected_effect`: increase | decrease | no_change | flag
  - `implied_effect`, for invalid edits
  - `expected_magnitude`: SCT −2..+2
  - `gold_source`: direct_gold | er_reason_sct | sct_bench | kg_premise | construction | physician
  - `physician_verified`

## Edit classes and gold
| Class | Mode | Construction | Gold |
|---|---|---|---|
| valid_correction | assertion | Insert physician-tree evidence the model's graph lacks (with the physician's rationale links) | increase (gold dx) |
| counterfactual | hypothetical | Remove / negate key observations on the gold path; weaken / intensify the leaf rationale | decrease / increase |
| new_finding | assertion | SCT "new information" (ER-Reason, SCT-Bench); the model must link it itself (`re_reason`) | sign and magnitude of the panel / consensus update |
| irrelevant | assertion | Clinician-vetted bystander facts (`configs/benchmark/bystander_facts.yaml`) | no_change |
| invalid | assertion | Negate an observation the record states unambiguously; numeric swap of a recorded value | flag (implied: decrease) |

## Portions
| Portion | Source | Release |
|---|---|---|
| credentialed | DiReCT gold trees (BIDMC) + ER-Reason SCT (UCSF) | PhysioNet credentialed channel only |
| public | MedCaseReasoning test (teacher reference graphs; physician-verified sample) | Open, after per-article licence check |

## Scoring
Use `clinblueprint evaluate` (`eval/runner.py`). The definitions are in `docs/EVALUATION.md`:
- mediation rate, strong mediation and asymmetry;
- invariance;
- flag sensitivity, false-flag rate, sycophancy rate and AUROC;
- SCT credit and DxUpdate κ;
- coverage.

Each comes with a cluster-bootstrap CI.

## Build
```bash
clinblueprint build-benchmark --config configs/eval/clinedit_faith.yaml
python scripts/export_audit_sheets.py edits --cases ... --edits ...verify_sample.jsonl --out clinician/review/edit_verification.csv
```
`*.verify_sample.jsonl` is a stratified sample (edit class × gold source) for physician verification. Verified labels are merged back as `physician_verified` and `gold_source: physician` where the physician changed the gold.

See `DATASHEET.md` for composition, collection, intended use and limitations.
