# Anticipated reviewer objections and pre-emptive answers

| Objection | Answer (and the experiment that backs it) |
|---|---|
| "A deterministic readout makes faithfulness trivial." | Yes, by design. The contributions are that (i) accuracy survives the bottleneck (G1, Fig. 2), (ii) invalid edits are resisted (Fig. 3), (iii) the *soft* and *in-context* variants' faithfulness is measured, forming a bottleneck-strength frontier, (iv) mechanism (Fig. 5) and (v) human benefit (Fig. 6). |
| "This is Breaking the Chain / CSR applied to medicine." | Those are non-clinical with deterministic targets. Here: probabilistic clinical graphs, physician-gold update directions at two sites, non-sycophancy, mechanism, clinician study. We also run their methods (schema-guided intermediates, causal-consistency loss) as ablations and show they alone underperform. |
| "Isn't this the counterfactual multi-agent paper?" | That work edits *inputs* at inference, training-free, to inform agents. We edit the model's *own reasoning state*, train the property, and score it against expert gold. Their method is a baseline (`baselines/counterfactual_multiagent.py`), and the note-edit regime is reported for every model. |
| "Edit-following is just sycophancy." | The hypothetical-vs-assertion distinction and the invalid / irrelevant arms test exactly this. We report the sensitivity / false-flag trade-off, the sycophancy rate and the correction–counterfactual **asymmetry**. |
| "Small gold data." | 5-fold patient-level CV on DiReCT; 13K MedCaseReasoning training cases; external tests from two hospitals plus a global set; cluster-bootstrap CIs; headline repeated on the physician-verified subset. |
| "Attribution graphs are approximate." | Six methods triangulated (three-path, swap, patching, IG, circuit-tracer, SAE / probes). They must agree in direction (gate G4); disagreements are reported. |
| "Why not GPT-5?" | PhysioNet rules forbid hosted APIs without verified zero retention, and local models serve deployment and privacy. We compare against published frontier numbers. |
| "Overlap with DEFER-Dx." | No deferral, escalation, interactive test ordering or posterior-coherence rewards here. We cite it as complementary work. |
| "The candidate set leaks the answer." | Candidates are every KG leaf of the touched categories, plus distractor categories in a pool shared by all systems (`benchmark/build.py`). The drafter-only candidate mode is an ablation. |
| "The edit gold is constructed, not real." | Gold sources are stratified (physician trees, expert SCT panels, KG criteria, construction). ~500 items are physician-verified with κ, and results are shown per gold source. |
| "CoT was handicapped." | The matched-data CoT-SFT uses identical cases and reasoning text. CoT gets a flag channel (flag prompting) and both trace-level and note-level edit regimes. |
| "Retargeting edits to model graphs hides failures." | Coverage (applicable / total) is reported per system, and oracle-graph results are reported separately. |
| "Thresholds were tuned on test." | ε, ε_tv are fixed a priori. The SCT Likert thresholds, validator τ and LTT τ are fitted on dev folds only. |
| "The validator will block true corrections when the chart is wrong." | A flag is not a refusal; overrides are applied and logged. We report clinician adjudication of borderline flags (`clinician/tasks/05_validator_adjudication.md`). |
| "Silver graphs are noisy." | Strict filters, a 100-graph clinician audit with reported error rates, and a silver-volume ablation. |
