# Causally Editable Clinical Reasoning: research dossier

Labels: **[V]** verified against a source; **[P]** carried over from earlier literature rounds and not re-checked; **[E]** estimate or projection. No one can guarantee novelty: §6 says what is defensible, what overlaps, and with whom. Re-run the prior-art search ([LITERATURE_SEARCH_LOG.md](LITERATURE_SEARCH_LOG.md)) shortly before submission.

---

## 1. Thesis

We train a small local clinical LM (4–8B) whose diagnosis is **causally mediated** by an explicit, typed, clinician-editable reasoning graph. We define and demonstrate a property we call **causal editability**, with four tests:

1. **Mediation.** A valid edit to the graph changes the diagnosis in the direction experts expect. The edit can be a clinician correction or a new finding.
2. **Non-sycophancy.** Edits that contradict the patient's record are flagged, not blindly followed, and irrelevant edits change nothing.
3. **No bypass.** Mechanistically, the diagnosis is computed *through* the graph's slot representations, not around them.
4. **Human benefit.** In a randomised crossover, clinicians catch and fix model errors better through the graph than through chain-of-thought text.

The project has two goals:

- **Science.** Causal editability is measurable and trainable, and same-size chain-of-thought (CoT) models fail axes 1–3.
- **Model.** "ClinBlueprint-4B/8B" beats same-size baselines on DiReCT, MedCaseReasoning, ER-Reason and the SCT benchmarks, and approaches much larger models.

### Why it is NMI-shaped
- It answers a question the editors raised in October 2026 ([NMI editorial](https://www.nature.com/articles/s42256-026-01333-6)): *are reasoning traces explanations or decoration?* The answer comes as a model, a measurable property, mechanistic evidence and a human study.
- The framing is general: high-stakes AI decisions, with medicine as the flagship case.
- It ships a reusable benchmark (ClinEdit-Faith).

---

## 2. The problem and the hook

**Reasoning text is often not the reason [P/V].**
- Claude 3.7 Sonnet verbalised the hints it used about 25% of the time, and DeepSeek-R1 about 39%.
- *Right Answer, Wrong Reason* ([2609.32817](https://arxiv.org/abs/2609.32817)): only 19.2% of cited symptom concepts actually drove the answer.
- *Breaking the Chain* ([2603.16475](https://arxiv.org/abs/2603.16475)): 18–23-point faithfulness gaps for schema-guided intermediate structures, in non-clinical tasks.
- NMI's October 2026 editorial and the Levy et al. Perspective framed reasoning tokens as not faithful explanations.

**Clinical belief updating is weak.** On SCT-Bench, reasoning-optimised models scored *worse* than non-reasoning models despite beating them elsewhere. It is the first medical benchmark where more reasoning hurt.

**More reasoning and more context can hurt [P].**
- 86.3% of 95 LLMs degraded with CoT on clinical text.
- On MIMIC-CDM, models were *less* accurate with more information, and were order-sensitive (Hager et al., *Nat Med* 2024).
- Always-on medical RAG lowered factuality.

**Clinicians do not reliably catch AI errors through explanations [V, PubMed].**
- Explanations did not mitigate a systematically biased model (Jabbour et al., *JAMA* 2023, [10.1001/jama.2023.22295](https://doi.org/10.1001/jama.2023.22295)).
- LLM access did not improve physicians' diagnostic reasoning (Goh et al., *JAMA Netw Open* 2024, [10.1001/jamanetworkopen.2024.40969](https://doi.org/10.1001/jamanetworkopen.2024.40969)).
- Automation bias persisted after AI-literacy training (Qazi et al., *NEJM AI*, [10.1056/AIoa2501001](https://doi.org/10.1056/AIoa2501001)).

**Sycophancy is a measured clinical risk [V, PubMed].**
- High compliance with illogical drug-equivalence requests ([10.1038/s41746-025-02008-z](https://doi.org/10.1038/s41746-025-02008-z)).
- Diagnostic instability under certainty challenges ([10.1016/j.artmed.2026.103516](https://doi.org/10.1016/j.artmed.2026.103516)).
- Warmth training increases sycophancy (*Nature* 2026, [10.1038/s41586-026-10410-0](https://doi.org/10.1038/s41586-026-10410-0)).

**The clinical need.** Clinicians need reasoning they can inspect, correct and trust to respond correctly. That means it must follow valid corrections and resist wrong ones. No current clinical LM is built or evaluated for this.

---

## 3. Landscape (condensed from earlier rounds) [P]

**Architectures**
- Plan-then-generate: QA blueprints (Narayan, TACL 2023).
- Prompt-level structure: ToT/GoT (crowded).
- Concept-level generation: Large Concept Models / SONAR-LLM.
- Auxiliary objectives and verifiers: LLM-JEPA, Energy-Based Transformers.
- Token-level self-correction (crowded): masked-diffusion LMs (LLaDA, Dream, Mercury, Gemini Diffusion) with remasking (ReMDM, PRISM, SCDD).
- Latent reasoning: Coconut (poorly interpretable).
- Concept bottlenecks: CB-LLM (ICLR 2025), which leaks in generation. Rudin (NMI 2019) argued for inherently interpretable models.
- Clinical KG+LLM (crowded *for accuracy*): GraphDx, DR.KNOWS, D2KGMed, KGARevion, MedKGI, RE-MCDF, Lung-R1.
- Interpretability tooling: frontier internals are undisclosed, but circuit tracing and attribution graphs are public (circuit-tracer, Neuronpedia, SAELens, Gemma Scope).

**Adaptive computation.** Always-on retrieval, reasoning or self-correction often hurts.
- Simple uncertainty gates matched complex adaptive-RAG pipelines (Moskvoretskii 2025).
- Only 22% of top-16 medical passages were relevant (Kim 2025).
- MedDistractQA distractors cut accuracy by up to 17.9%.
- A CXR study found retrieval-induced hallucination.
- m1 found a ~4K-token reasoning optimum.
- Intrinsic self-correction is unreliable without external feedback (Kamoi, TACL 2024); as-needed decomposition helps (ADaPT).
- Tools: semantic-entropy probes, Med-PRM, TRM/HRM halting, Conformal Thinking, Thinkless/AdaptThink, CaRDI, StratCP.

**Novelty audit of neighbours.**
- **DEFER-Dx** (Clinical-Agent-Stanford): RL with ASK/TEST/COMMIT/DEFER actions on MIMIC-CDM, counterfactual escalation value, Brier-scored differential handoff. Its sibling, clinical-coherence-rl, uses posterior-coherence rewards. **Learned deferral, severity-weighted rewards, interactive MIMIC-CDM RL and posterior-coherence rewards are all off-limits** as contributions here.
- General-domain threats: *Breaking the Chain* (mediation protocol, non-clinical); Rubin-Toles et al., ICLR 2025 (conformal coherent factuality over reasoning graphs); ITCR ([2606.08831](https://arxiv.org/abs/2606.08831), conformal stopping during graph generation); CSR ([2509.01544](https://arxiv.org/abs/2509.01544), a causal-consistency loss).
- Clinical neighbours: DiReCT; Clinical Reasoning Graphs ([2606.29876](https://arxiv.org/abs/2606.29876), evaluation only); DocCHA (graph refinement, no guarantees); CaRDI (question-level risk).

---

## 4. New data that fixes the single-centre problem [V]

**MedCaseReasoning** (public, global)
- 14,489 PMC-OA case reports (13,092 train / 897 test) from 800+ journals and 30+ specialties, with clinician-authored reasoning.
- DeepSeek-R1 reached 48% 10-shot accuracy and recalled 64% of clinician reasoning statements.
- Fine-tuning open models raised accuracy by 29% and reasoning recall by 41% (average relative gains).

**ER-Reason** (UCSF ED; PhysioNet)
- 25,174 notes across 3,984 encounters.
- 194 physician-authored SCT cases with 2,555 annotations: two board-certified emergency physicians annotated each timestep and a third adjudicated.
- Metrics: DxUpdate, DxTrajectory, FinalDx.
- This is the second site, and it supplies physician-gold direction-of-update labels.
- File layout is verified in [DATASETS.md](DATASETS.md).

**SCT-Bench / concor.dance**
- 750 SCT questions from ten datasets; models consistently fall short of expert clinicians.
- The held-out test set is available through a DUA.
- Reported scores (secondary source; verify against the paper): o3 67.8%, o1-preview 58.2%, R1 55.5%.

**Already in hand [P].** MIMIC-IV-Ext-DiReCT (511 notes with physician reasoning trees), MIMIC-CDM (2,400 cases), MIMIC-IV-Note, MedQA/MedMCQA/PubMedQA/MedXpertQA.

**Additional sources found in this round's PubMed / medRxiv search.** MultiCaRe (96,428 PMC cases; PMCID de-duplication required), PMC-Patients, MedNLI (validator pre-training), n2c2-2010 assertions, DDXPlus (optional synthetic counterfactuals), and published sycophancy and automation-bias protocols. See the [README dataset table](../README.md#datasets).

---

## 5. Prior art found this round, and why it does not pre-empt us [V]

**Counterfactual multi-agent diagnosis** (You et al., [2603.27820](https://arxiv.org/abs/2603.27820))
- Agents edit findings in the *input case*, measure how competing diagnoses shift (the "Counterfactual Probability Gap"), and use that to inform agents.
- Training-free; 80–100 cases per dataset; human evaluation of 12 trace pairs by two physicians.
- Reported: GPT-5-mini 93.6 / 66.0 / 26.7 on MIMIC / MedCaseReasoning / ER-Reason; MedGemma-1.5-4B 79.2 on MIMIC; Llama-3.1-8B 75.4 / 22.7 / 18.8.
- **Difference.** We edit the *model's own reasoning graph*, require the output to be mediated by it, *train* for it, score against physician-gold update directions, test non-sycophancy, and verify the mechanism.
- Their operation set (negate, remove, replace, weaken, intensify, insert) is our edit vocabulary; cite it. Their method is a baseline (`baselines/counterfactual_multiagent.py`). Coverage note: only ~75% of that PDF was read.

**Position, Not Provenance** ([2607.27304](https://arxiv.org/abs/2607.27304))
- Perturbs attributes inside a medical VLM's own CoT and classifies outcomes as faithful mediation, decoupled, or reactive-incoherent. It explicitly separates mediation from sycophancy.
- **Difference.** That work analyses existing VLMs. We build a text clinical model that *achieves* mediation without sycophancy, scored against physician gold. Cite it as motivation.

**Mechanistic medical diagnosis** (AAAI 2026)
- Transcoders and attribution graphs on Gemma-2-2B reveal two parallel pathways: pattern-matching and reasoning.
- **Difference.** Descriptive. We test whether a graph bottleneck plus faithfulness training *suppresses* the pattern-matching bypass, which is a causal architectural claim. Their pathway taxonomy is our mechanistic baseline.

**Causal-Audit** (ACL Findings 2026) **and ECR-Agent** (ACL 2026)
- Explicit causal graphs at inference.
- **Difference.** They are pipelines and agents with no guarantee that the output is mediated by the graph, no clinician-edit protocol and no mechanistic test. They are accuracy competitors only.

**Clinical Reasoning Graphs** ([2606.29876](https://arxiv.org/abs/2606.29876)): graphs for evaluation only.

---

## 6. Honest novelty verdict

| Claim | Verdict | Closest work |
|---|---|---|
| Graph-structured clinical reasoning improves accuracy | Not novel | GraphDx, KGARevion, DocCHA, ECR-Agent, Causal-Audit |
| Counterfactual case editing to test evidence importance | Not novel (input-level, training-free) | You et al. 2603.27820 |
| Measuring belief updating with SCT | Not novel as evaluation | SCT-Bench, ER-Reason |
| Mediation-vs-sycophancy testing of model reasoning | Not novel as analysis (VLM CoT) | Position, Not Provenance |
| Descriptive mechanistic analysis of medical diagnosis | Not novel | AAAI 2026 Gemma-2 attribution study |
| **Training a clinical LM so its diagnosis is causally mediated by its own editable graph, scored against physician-gold update directions (DiReCT + SCT) across two hospitals** | **Defensibly novel** (no counter-example found) | Breaking the Chain, CSR (non-clinical, deterministic targets) |
| **Corrigible-but-not-sycophantic editing (accept valid, flag invalid, ignore irrelevant) as a trained property with clinical gold**, incl. the *hypothetical vs assertion* distinction | **Defensibly novel** | Position, Not Provenance (analysis only) |
| **Mechanistic evidence that a bottleneck suppresses the pattern-matching bypass** (three-path decomposition, patching, attribution, SAE) | **Defensibly novel** | AAAI 2026 (descriptive only) |
| **Randomised clinician crossover: error correction via graph vs CoT vs answer-only** | **Defensibly novel as designed** | Earlier studies rated traces or measured accuracy, not correction through an editable artefact |
| Node-local risk-gated compute over the model's own clinical graph (Claim C) | Defensibly novel but secondary | ITCR (stop-only), CaRDI (question-level) |

The novelty lies in **the combination and the trained property**, not in any single ingredient. Credit each ingredient explicitly in the paper.

---

## 7. Contribution (final)

**Core claim.** Causal editability is a measurable, trainable property of clinical LMs. A 4–8B local model with a typed reasoning-graph bottleneck, trained with counterfactual-edit objectives, achieves it on four axes without losing accuracy:

1. mediation of valid edits;
2. resistance to invalid edits;
3. mechanistic routing through the graph;
4. improved clinician error correction.

Same-size CoT models fail on axes 1–3.

**Supporting claim A: graph-mediated belief updating.** When a new finding is added as a graph node, the diagnosis update matches expert SCT judgements better than when the same finding is added as text to a CoT model. This holds at two hospitals (BIDMC via DiReCT; UCSF via ER-Reason SCT) and on public SCT-Bench.

**Supporting claim B: corrigible, not sycophantic.** An edit validator (span-grounding rules plus a learned, calibrated entailment verifier) lets the model accept corrections consistent with the record and flag those that contradict it. Results are reported as a sensitivity / false-flag trade-off.

**Supporting claim C (optional, gated): node-local compute.** Grounding or repair is spent only on nodes whose calibrated risk is high, under a Learn-then-Test guarantee. It is kept only if it moves the accuracy-vs-tokens front.

**Design refinements this repository adds to the dossier.** These are intended to strengthen the paper:
- **Hypothetical vs assertion** speech acts. Counterfactual what-ifs are always answered, which is how mediation is measured. Factual assertions are validated, which is how non-sycophancy is measured. The same graph operation is judged differently according to intent. This removes the apparent conflict between "follow edits" and "resist edits".
- **Correction–counterfactual asymmetry** as a behavioural bypass signature. A model that "knows" the answer from the note follows agreeing edits and resists opposing ones.
- **Bottleneck-strength spectrum**: hard → soft-isolated (graph-only encoding) → soft-in-context → CoT. It is reported as a faithfulness–accuracy frontier.
- **Candidate-selection leakage control**: the readout scores every KG leaf of every touched category, not only the drafter's own diagnoses.
- **Three-path decomposition** of information flow: graph text, hidden-state surplus, and direct note path.
- **Hard-readout teacher for DPO**: the structural readout of the edited graph supplies the "chosen" answer, distilling mediation into the in-context generative variant.
- **Matched-data CoT-SFT baseline** trained on identical cases and reasoning content.

**Explicitly out of scope** (to avoid DEFER-Dx and related work): RL-learned deferral, counterfactual escalation credit, scored handoffs, interactive test ordering, posterior-coherence rewards.

---

## 8. Targets that would "prove it right" [E]

| Benchmark | Target |
|---|---|
| DiReCT | Accuracy ≥ 0.61 at 4–8B (GPT-4-turbo 0.610 [P]); explanation "all" ≥ 0.30 (GPT-4-turbo 0.210) |
| MedCaseReasoning | Beat same-size rationale-SFT by ≥ 3 points; stretch ≥ 48% (R1 10-shot) at 8B |
| ER-Reason | Beat same-size baselines. Labels are symptom-level, so SCT is the main ER-Reason endpoint |
| SCT concordance | Graph-mediated updates beat the backbone's text updates, with CIs excluding 0 at both sites |
| Mediation | ≥ 0.90 (hard) / ≥ 0.80 (soft), vs a measured CoT baseline expected around 0.3–0.7 |
| Non-sycophancy | Flag sensitivity ≥ 0.80, false-flag ≤ 0.10, irrelevant-edit invariance ≥ 0.95 |
| QA sanity | Within 2 points of the untuned backbone |

Published reference numbers are in [EVALUATION.md §7](EVALUATION.md).

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Silver-graph noise | Strict filters (`data/silver.py`), 100-graph clinician audit, oracle-graph upper bound reported |
| Bottleneck accuracy ceiling | Report honestly; soft variant; bottleneck-strength frontier |
| Tooling gaps (no transcoders/SAEs for a backbone) | Qwen3-4B (circuit-tracer) + Gemma-3/MedGemma (Gemma Scope 2); `scripts/mech/check_tooling.py` before committing |
| Pre-emption | Monthly arXiv/PubMed/medRxiv re-search ([LITERATURE_SEARCH_LOG.md](LITERATURE_SEARCH_LOG.md)) for clinical versions of Breaking the Chain, ITCR follow-ups, SCT-trained models |
| Clinician recruitment | Clinician co-author recruited; second rater for κ; public cases for the human study (no credentialing) |
| ER-Reason labels are symptom-level | SCT is the main ER-Reason endpoint; diagnosis accuracy secondary; CCSR grouping |
| SCT-Bench held-out DUA | Apply early; public items meanwhile |
| Validator over-flags true chart errors | Flag ≠ refusal; override logged; clinician adjudication of borderline flags |
| Edit gold noise | Physician verification of ~500 stratified edits, κ reported, analyses restricted to verified items as a sensitivity check |
