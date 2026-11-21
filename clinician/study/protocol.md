# Protocol (draft): explanation format and clinician correction of AI diagnostic errors

## Objective
Do clinicians detect and correct incorrect AI diagnoses more often when the AI's reasoning is shown as an **editable, causally faithful reasoning graph** than as **chain-of-thought text** or **no explanation**?

## Design
- **Randomised, three-period crossover.** Each participant uses all three arms on different cases.
- **Arms.**
  - A: editable reasoning graph (ClinBlueprint).
  - B: CoT text explanation (matched-data CoT-SFT model, same backbone).
  - C: answer only.
- **Sequences.** Williams design for 3 treatments (6 sequences, balanced for first-order carry-over); see `src/clinblueprint/study/design.py`.
- **Cases.** About 120 MedCaseReasoning *test* cases (public, so no credentialing is needed). They are split into 3 blocks stratified by (model correct / incorrect) and specialty, so each arm sees the same natural error rate. No case is shown twice to the same participant.
- **Errors are natural.** No errors are planted; the model is wrong on a realistic fraction of cases. The model output for each case is fixed in advance.
- **Precedents.** Jabbour et al. (*JAMA* 2023) used a randomised clinical vignette survey with systematically biased models. Goh et al. (*JAMA Netw Open* 2024) used a randomised design with a structured reflection rubric. Qazi et al. (*NEJM AI*) measured automation bias.

## Participants
12–20 clinicians (attending physicians and residents; internal and emergency medicine). Inclusion: currently in clinical practice or training. Recruitment through the clinician co-author's network. Participants are compensated at [rate].

## Procedure (per case)
1. Read the case.
2. See the AI's diagnosis, with or without an explanation depending on the arm.
3. Decide: accept, or correct (Arm A: edit the graph; Arms B/C: write the corrected diagnosis).
4. Give a final diagnosis and confidence (0–100).
5. Time is recorded automatically.

## Outcomes
- **Primary:** error detection = proportion of AI-incorrect cases the participant rejects or corrects.
- **Secondary:**
  - final accuracy after correction (all cases);
  - time per case;
  - automation bias (acceptance of incorrect AI answers);
  - unnecessary overrides of correct AI answers;
  - trust calibration (the relation between confidence and AI correctness);
  - in Arm A, the type and correctness of edits.

## Analysis (pre-registered)
- Primary: GEE logistic regression on AI-incorrect trials, outcome ~ arm + period, exchangeable correlation within participant, robust SEs. Contrast A vs B (two-sided α = 0.05); A vs C secondary.
- Sensitivity analyses: mixed-effects logistic regression with random participant and case intercepts; per-protocol analysis excluding cases with technical failures.
- Code: `src/clinblueprint/study/design.py::analyze`.

## Sample size
- Simulation (`clinblueprint study-power --config configs/eval/study_power.yaml`) with planning values: 16 clinicians, 120 cases, 35% model error rate, detection 0.65 vs 0.45, clinician SD 0.5 on the logit scale. This gives power ≈ 0.95. These are placeholder values [E].
- **Replace them with pilot estimates** (pilot: 3 clinicians × 30 cases, not included in the main analysis).

## Ethics and data
- IRB / exempt determination before recruitment.
- Cases are published, de-identified case reports.
- Participant data: pseudonymous IDs, stored outside git, deleted on study completion except for de-identified analysis files.
- Informed consent via an information sheet.

## Blinding
Participants cannot be blinded to the arm. Outcome assessment (correctness of final diagnoses) uses a pre-specified judge validated against physician labels on a public subset, blinded to arm.
