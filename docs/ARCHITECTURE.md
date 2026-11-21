# Architecture: ClinBlueprint

## 1. Graph schema (aligned with DiReCT)

| Node | Fields | DiReCT origin | Code |
|---|---|---|---|
| Observation | `id, text, span{start,end,text,section,score}, polarity{present,absent,uncertain}, source{note,clinician,new_finding,hypothetical}, section, concept?` | `"<text>$InputN"` | `schema/graph.py` |
| Rationale | `id, premises[], claim, target(dx id), relation{supports,refutes}, strength∈[0,1], premise_polarity{}` | `"<claim>$Cause_k"` | |
| Diagnosis | `id, category, leaf (KG node), refines (more general dx id), p (drafter self-report; ignored by default)` | `"<dx>$Intermedia_k"` | |
| MustNotMiss | `id, dx, status{excluded,open,not_addressed}, evidence[]` | (new) | |

Edges are **derived** from node fields, so there is a single source of truth:

```
premise ──premise──▶ rationale ──supports | refutes──▶ diagnosis ──refines──▶ diagnosis
```

## 2. Pipeline

```python
G   = drafter.draft(note)                 # two-stage: KG category selection, then constrained JSON
G   = check(G, note, kg, mnm)             # grounding, KG validity, premise cascade, support, MNM
[for risky nodes v (Claim C, LTT-gated): G = drafter.ground_or_repair(G, v, note); G = check(G)]
ddx = readout(G)                          # the bottleneck: a function of G only
txt = verbalizer(G, ddx)                  # sees only serialised G and ddx, never the note

intervene(G, edit)                        # hypothetical: apply -> [re_reason] -> check -> readout
apply_clinician_edit(G, edit)             # assertion: validate; flag if contradicted; else intervene
```

The implementation is `pipeline.ClinBlueprint`.

### Hard checks (`checks/hard_checks.py`)
1. **Grounding.** Every note-sourced observation must be a verbatim span, tried in order: exact match, case/whitespace-insensitive, then fuzzy ≥ 90. Ungrounded observations are removed.
2. **KG validity.** The leaf must exist in its category, and `refines` must point to a KG ancestor; refinements are repaired from the KG.
3. **Premise cascade.** A rationale holds only if every premise exists with the polarity it expects. Invalid rationales are pruned to a fixed point. *This is how edits propagate.*
4. **Support.** A diagnosis with no supporting rationale is flagged (warning; the readout gives it only its prior).
5. **Must-not-miss.** For each active category, missing clinician-listed items are inserted as `not_addressed`.

## 3. Readouts: the bottleneck-strength spectrum

| Variant | Decision computed from | Mediation | Code |
|---|---|---|---|
| **Hard** | Deterministic evidence-path score | Structural (by construction) | `readout/hard.py` |
| **Hard-linear** | Sign-constrained conditional logit over graph features | Structural (monotone weights) | `readout/linear.py` |
| **Soft-isolated** | Slot-attention head over slot embeddings from the *serialised graph alone* | Only through graph text; measured | `readout/soft.py` (`encoding="graph_only"`) |
| **Soft-in-context** | Same head, slots encoded *with the note in context* | Hidden-state bypass possible; measured and trained against | `encoding="note_in_context"` |
| **In-context generative** | LM reads note + graph (+ edit) → answer / FLAG; DPO-trained | Measured (mech suite) | `training/dpo.py`, `baselines/systems.InContextGraphSystem` |
| CoT (baseline) | Free text | None | `baselines/systems.CoTSystem` |

### Hard readout
For candidate label L with KG path L → parent → … → root:

```
score(L) = log_prior(L) + Σ_k decay^k · E(path_k)
E(d)     = w_s Σ_{r supports d} s_r g_r  −  w_r Σ_{r refutes d} s_r g_r
g(r)     = Π_premises {1 valid, u if UNCERTAIN, 0 invalid}
p        = softmax(score / T)
```

Properties, unit-tested in `tests/test_kg_checks_readout.py`:
- **Structural mediation.** The note never enters.
- **Monotonicity.** A valid supporting rationale on L's path never lowers p(L).
- **Locality.** Off-path evidence affects p(L) only through normalisation.
- The drafter's self-reported `p` is ignored unless `use_drafter_p` is set; that switch exists only for the bypass ablation.

### Candidate-selection leakage
If the candidates were only the drafter's diagnoses, the drafter could choose the answer by choosing the candidate set. `candidate_mode="kg_expand"` (the default) scores **every KG leaf of every category the graph touches**, plus the drafted terminals. For evaluation, all systems score the same **fixed candidate pool** per case: the gold category's leaves plus leaves of 2 distractor categories, or the task's own list (`benchmark/build.py`).

### Soft head anti-bypass defaults
- Diagnosis slots are excluded from the keys, so a rationale slot must name its target.
- Candidates are embedded from their *label text*, not from the graph.
- A learned null key is used when no evidence remains.

## 4. Edits (`edits/`)

| Concept | Values |
|---|---|
| Operation | negate, remove, replace, weaken, intensify, insert (the You et al. 2603.27820 set, applied to the model's own graph) |
| **Mode** | **hypothetical**: what-if, always answered, never validated. **assertion**: a claim about the patient, validated against the record |
| Class (benchmark) | valid_correction, counterfactual, new_finding, irrelevant, invalid |
| Gold effect | increase, decrease, no_change, flag (+ SCT magnitude −2..+2, + `implied_effect` for invalid edits) |
| Reason mode | **frozen**: graph edit → readout. **re_reason**: inserted findings are linked by the model itself (`drafter.link_new_observation`), the step SCT items test |

Two design rules:
- **A flag is not a refusal.** `apply_clinician_edit(..., override=True)` applies the edit and records `overridden=True`.
- **Retargeting.** Benchmark edits are written against the gold graph. For model graphs they are retargeted by text similarity (`edits/align.py`); edits with no matching node are *not applicable*, and per-system coverage is reported.

### Validator (`edits/validator.py`)
1. **Rules** (high precision):
   - NegEx-style polarity contradiction against the note (trigger window, clause-bounded);
   - numeric contradiction on anchored measurements.
2. **Learned verifier**:
   - a 3-way NLI classifier (LoRA on the backbone) over an evidence window;
   - isotonic-calibrated, with the threshold chosen on dev for target sensitivity (`training/validator_train.py`).
3. **Decision**: p = max(rule, calibrated verifier) ≥ τ → `contradicts_evidence`. A REMOVE assertion means "not relevant", so it is not contradiction-checked.

## 5. Drafter and verbaliser
- **Drafter** (`models/drafter.py`): two stages.
  1. Choose ≤ 3 KG categories.
  2. Emit the JSON graph with the chosen trees (+ premises) in the prompt.
  - Training uses gold + 2 distractor categories, so the format matches inference.
  - Output order is *evidence first* (observations → rationales → diagnoses).
  - Optional constrained decoding (lm-format-enforcer); parse retries with temperature.
- **Verbaliser** (`models/verbalizer.py`):
  - Conditioned on the serialised graph + readout only.
  - Trained on template renderings of gold/silver graphs, so its targets are faithful by construction.
  - `audit()` flags unknown node ids and numbers not present in the graph.

## 6. Claim C: node-local compute (optional)
- `risk/node_risk.py` computes a logistic risk over graph-local features.
- `risk/ltt.py` provides Learn-then-Test (Hoeffding–Bentkus p-values, fixed-sequence testing) for τ.
- It runs only when `repair_budget > 0`. It is reported only if it improves accuracy per generated token.
