# NMI presubmission enquiry (draft; send once decision gates G1–G3 have outcomes)

**To:** Nature Machine Intelligence editors
**Subject:** Presubmission enquiry: Causally editable clinical reasoning

Dear Editors,

Your October 2026 editorial asked whether the reasoning traces of language models are explanations or decoration. We would like to ask whether a manuscript answering this question for clinical diagnosis would interest *Nature Machine Intelligence*.

The manuscript introduces **causal editability**. A clinical model's diagnosis should follow valid corrections to its reasoning, flag corrections that contradict the patient record, and ignore irrelevant ones, and it should do so *because* the decision is computed through that reasoning. We present:

1. **ClinBlueprint**, a 4–8B model, run locally, whose diagnosis is mediated by a typed, clinician-editable reasoning graph;
2. **ClinEdit-Faith**, a benchmark of edits with physician-derived expected effects from two hospitals (BIDMC and UCSF) and a global case-report set;
3. **mechanistic evidence** that the decision is routed through the graph rather than around it;
4. a **randomised clinician crossover study** on whether clinicians catch more model errors with the graph than with chain-of-thought text.

Key results (Figs 2–5 attached): \[accuracy\], \[mediation\], \[non-sycophancy\], \[mechanism\].

Graph-structured clinical reasoning for accuracy, and input-level counterfactual analysis, have been reported before. Our contribution is the trained property, its measurement against physician gold, and its mechanistic and human validation.

Sincerely,
\[corresponding author\]
