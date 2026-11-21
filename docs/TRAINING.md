# Training recipe (one 48 GB GPU)

All numbers marked [E] are estimates to be replaced by measured values.

| Stage | Command | Config | Trains | Data |
|---|---|---|---|---|
| 0 Prepare | `prepare-data` | `configs/data/prepare.yaml` | – | raw → `data/processed` |
| 1 Silver graphs | `build-silver` | `configs/data/silver.yaml` | – (teacher inference) | MedCaseReasoning train (+ MIMIC-IV-Note, local only) |
| 2 Graph SFT | `train-sft` | `configs/train/sft_qwen3_4b.yaml` | drafter + verbaliser (LoRA) | DiReCT train folds + silver |
| 2b CoT-SFT baseline | `train-cot-sft` | `configs/train/cot_sft.yaml` | free-CoT model (LoRA) | **same cases** |
| 3 Draft corpus | `draft-graphs` | `configs/eval/draft_graphs.yaml` | – | model graphs on train cases |
| 4 Readout | `train-readout` | `configs/train/readout.yaml` | sign-constrained linear readout | model-drafted train graphs |
| 5 Validator | `train-validator` | `configs/train/validator.yaml` | 3-way NLI verifier (LoRA) + isotonic | generated triples (+ MedNLI) |
| 6 Faithfulness | `train-faithfulness` | `configs/train/faithfulness.yaml` | soft slot head | graphs + generated edits |
| 7 DPO | `train-dpo` | `configs/train/dpo.yaml` | in-context generative variant | preference pairs from edits |
| 8 GRPO (optional) | (TRL; rewards in `training/rewards.py`) | – | drafter | only if SFT+DPO trails same-size baselines by > 3 points |
| 9 Claim C (optional) | `risk/` | – | risk gate + LTT τ | dev graphs |

## Stage details

**1. Silver graphs.**
- The teacher is Qwen3-32B in 4-bit, run locally. It gets the case text, the authors' reasoning and the final diagnosis, plus the KG block when the category is known.
- Hard checks, then the quality filter: ≥ 2 grounded observations, grounding ≥ 0.8, a terminal diagnosis matching the reference, a supported path, and no error violations.
- Target ~30–50K graphs [E], about 1–3 GPU-days [E].
- The clinician audits 100.

**2. Graph SFT.**
- Prompt: note + KG block (gold category + 2 distractors, shuffled) → completion: graph JSON.
- Verbaliser examples: serialised graph + readout → template explanation.
- LoRA r=16, lr 1e-4, 2 epochs, effective batch 16; about 12–20 h/epoch at 4B [E].

**3–4. Readout on model drafts.**
- Fitting on the drafter's *own* training-set graphs makes the readout robust to the drafter's real error profile.
- Fitting on gold graphs gives the oracle upper bound (ablation).

**5. Validator.**
- Triples: grounded finding → entailment; negated finding / numeric swap → contradiction; another case's unmentioned finding → neutral.
- Isotonic calibration on dev; τ set for 0.80 sensitivity; dev false-flag rate reported.

**6. Faithfulness (soft head).**
- Loss: L = CE(gold) + λ_cf · hinge(direction of Δ log p on counterfactual / correction edits) + λ_inv · symmetric-KL(irrelevant edits).
- By default only the head trains, on frozen SFT-encoder slot embeddings, so the slot representation is fixed for mechanistic analysis.

**7. DPO (generative in-context variant).**

| Edit type | Chosen | Rejected |
|---|---|---|
| Valid / counterfactual | Hard-readout diagnosis on the edited graph | Stubborn pre-edit diagnosis |
| Invalid assertion | `FLAG: <evidence>` | Following the edit |
| Irrelevant | Unchanged diagnosis | A changed one |

- The reference model is the policy with its LoRA disabled, so no second copy is held in memory.

## Memory budget [E]
| Setting | Approx. peak |
|---|---|
| 4B LoRA, 8K context, gradient checkpointing, bf16 | 20–30 GB |
| 8B LoRA, 8K context | 35–45 GB (use 4-bit if needed) |
| 32B teacher inference, 4-bit | ~22–26 GB + KV cache |

Dedicate the GPU. DEFER-Dx reported out-of-memory crashes on a shared card.

## Protocol
- **Folds.** 5 folds on DiReCT: for fold *k*, test = *k*, dev = *k*+1, train = the rest. Silver data is always train. Report mean ± bootstrap CI over folds.
- **Backbones.**
  - Qwen3-4B: primary (circuit-tracer transcoders).
  - MedGemma/Gemma-3-4B: second family (Gemma Scope 2).
  - Qwen3-8B: scale check.
  - Run `scripts/mech/check_tooling.py` before committing to a backbone.
- **Seeds.** 3 seeds for the headline configuration; 1 for ablations.
- **Freeze environments.** `uv pip compile pyproject.toml --extra train -o requirements.lock` on the GPU host, and record model revisions in `configs/model/*.yaml`.
