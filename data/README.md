# data/ (git-ignored except this file)

Nothing in this directory is ever committed. Credentialed data (MIMIC-IV, MIMIC-IV-Note, DiReCT, MIMIC-IV-Ext-CDM, ER-Reason, MedNLI, SCT-Bench held-out) must stay inside the approved environment.

```
raw/direct_kg/          DiReCT diagnostic KG (public)        scripts/download/download_public.sh
raw/physionet/<proj>/   credentialed PhysioNet downloads     scripts/download/download_physionet.sh
raw/MedDistractQA/      irrelevant-edit seeds (public)
raw/ccsr/, raw/multicare/, raw/sct_bench/, raw/ddxplus/   manual / optional
processed/              unified JSONL                        clinblueprint prepare-data
silver/                 teacher graphs                       clinblueprint build-silver
drafts/                 model-drafted graphs                 clinblueprint draft-graphs
benchmark/clinedit_faith/  cases + edits                     clinblueprint build-benchmark
```
See `docs/DATASETS.md` for formats and `scripts/check_access.py` for what is present.
