# Task 3: Edit verification (~500 benchmark items)

**Why it matters.** Each benchmark item claims "after this edit, diagnosis X should go up / go down / stay the same / be flagged". Your agreement rate (and κ with a second rater) is the evidence that the benchmark's gold is clinically valid.

**What you receive.** `clinician/review/edit_verification.csv`, a stratified sample across edit types and gold sources. For each row you see the edit in plain language (e.g. "What if 'sputum culture: *S. pneumoniae*' were the opposite of what is recorded?"), the diagnosis being measured, and the expected effect.

**For each row**
1. `agree`: Y/N.
2. If N, write `your_expected_effect`: increase / decrease / no_change / flag.
3. `confidence`: 1–5.
4. `comments`: e.g. "depends on pre-test probability", "edit wording is misleading".

**Remember the two edit modes**
- **Hypothetical** ("What if…"): judge the *direction* an expert's belief should move.
- **Assertion** ("Clinician states…"): if the statement contradicts what the note says, the right answer is **flag**, even if the clinician might be correct in reality.

**Second rater.** A second clinician rates a random ~20% subset independently, for κ. Please do not discuss items before both have finished.
