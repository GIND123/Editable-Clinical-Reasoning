# Pre-registration template (OSF / AsPredicted style)

1. **Hypotheses.**
   - H1: error detection in Arm A (graph) > Arm B (CoT).
   - H2: A > C (answer only).
   - H3: automation bias A < B.
2. **Dependent variables.**
   - Primary: error detection on AI-incorrect cases.
   - Secondary: as in `protocol.md`.
3. **Conditions.** A / B / C, within-participant, Williams sequences.
4. **Analyses.** GEE logistic (outcome ~ arm + period), exchangeable within participant, robust SEs; two-sided α = 0.05; Holm correction for H2–H3.
5. **Outliers and exclusions.**
   - Participants completing < 50% of cases are excluded.
   - Trials with time < 10 s are treated as non-engagement and excluded.
   - The ambiguity exclusion rule for validator items is defined in `clinician/tasks/05_validator_adjudication.md`.
6. **Sample size.** N = [from the pilot-informed power analysis]; stopping rule: fixed N.
7. **Other.** Model outputs are frozen before recruitment; the case list and the model correctness per case are deposited before data collection.
