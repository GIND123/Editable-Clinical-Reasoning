# Datasheet: ClinEdit-Faith (Gebru et al. format; fill the bracketed fields at release)

## Motivation
- **Purpose.** Measure whether clinical reasoning systems are *causally editable*: they follow valid edits, flag contradicted assertions and ignore irrelevant facts.
- **Creators / funding.** [authors, affiliations, funding]

## Composition
- **Instances.** (case, edit, gold expected effect) triples. Counts per class and portion: [from `*.stats.json`; suppress cells < 10 for the credentialed portion].
- **Sources.** DiReCT (MIMIC-IV-Note discharge summaries with physician reasoning trees; BIDMC); ER-Reason SCT (UCSF ED); MedCaseReasoning (PMC-OA case reports).
- **Labels.** Expected effect + gold source. Physician-verified subset of ~500 items, with κ for a doubly rated subset: [κ].
- **Missing information.** Model-graph retargeting may make some edits not applicable to a given system; coverage is reported.
- **Confidential data.** The credentialed portion contains de-identified clinical text governed by PhysioNet DUAs.

## Collection process
- Edits are generated programmatically (`src/clinblueprint/edits/generator.py`, `benchmark/build.py`) from physician trees, SCT panels, KG criteria and clinician-vetted bystander facts.
- Physician verification uses `scripts/export_audit_sheets.py`.

## Preprocessing
- Sections and offsets follow DiReCT. Span grounding is exact, then normalised, then fuzzy (≥ 90).
- SCT typo fixes follow ER-Reason's `sct_eval.py`.

## Uses
- **Intended.** Research evaluation of faithfulness, mediation and sycophancy in clinical reasoning systems.
- **Not intended.** Clinical decision-making, model certification, or any use on identifiable data.

## Distribution
- The credentialed portion goes through PhysioNet only. The public portion uses an open licence compatible with the source articles: [licence].

## Maintenance
- [contact; versioning; erratum process]
