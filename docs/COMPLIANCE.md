# Compliance and release policy

## Rules
1. **Local processing only.** No MIMIC-IV, MIMIC-IV-Note, DiReCT, MIMIC-CDM, ER-Reason, MedNLI or SCT-Bench held-out text goes to any hosted API, LLM judge, simulator or web tool. This is enforced in code: `compliance.guard.assert_local(backend, endpoint, provenance)` is called before teacher, drafter, evaluation and mechanistic runs. Hosted backends and non-private endpoints raise `ComplianceError` for `physionet` / `dua` provenance.
2. **Credentialing.** Everyone who sees credentialed text must hold their own PhysioNet credential: CITI "Data or Specimens Only Research" training plus a signed DUA per project. That includes clinicians reviewing items.
3. **Storage.** Credentialed data stays inside the approved environment. `data/` and `outputs/` are git-ignored; never commit raw or derived credentialed text.
4. **Graders.** Accuracy grading on credentialed data uses *local* judges. Each is validated against physician labels on a public subset first.
5. **Small cells.** Counts of 1–9 are suppressed ("<10") in any table derived from credentialed data (`compliance.guard.suppress_small_cells`).
6. **ER-Reason licence.** The PhysioNet Contributor Review Health Data License requires per-study review. Describe the study accurately when requesting access, and check the terms before releasing anything derived.

## Releases
| Artefact | Channel |
|---|---|
| Code | This repository (MIT) |
| Weights trained on MIMIC- or ER-Reason-derived data | PhysioNet credentialed channel only |
| ClinEdit-Faith credentialed portion (DiReCT, ER-Reason) | PhysioNet credentialed channel only |
| ClinEdit-Faith public portion (MedCaseReasoning) | Open release, after checking each article's licence |
| Weights trained only on public data | Hugging Face (with model card) |

## Human-subjects research
The clinician crossover study needs an IRB or exempt determination before recruitment (see `clinician/study/protocol.md`). Rater responses are stored outside git (`clinician/study/responses/` is ignored) and de-identified before analysis.
