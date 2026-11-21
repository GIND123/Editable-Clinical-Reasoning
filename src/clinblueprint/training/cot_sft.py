"""Baseline: free chain-of-thought SFT on the SAME cases and the same reasoning content.

Targets: MedCaseReasoning clinician reasoning (or DiReCT/silver rationales
rendered as prose) followed by "Final diagnosis: <dx>". This is the
matched-data comparator for every ClinBlueprint claim. It differs from
ClinBlueprint only in the absence of a graph bottleneck.

    clinblueprint train-cot-sft --config configs/train/cot_sft.yaml
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

from clinblueprint.models import prompts as P
from clinblueprint.schema.graph import Polarity, ReasoningGraph, Relation, natural_id_key, split_dx_key
from clinblueprint.training.examples import TextExample, pad_collate, tokenize_example
from clinblueprint.training.sft import load_graph_corpus
from clinblueprint.utils.io import ensure_dir, seed_everything, write_json


def graph_to_prose(G: ReasoningGraph) -> str:
    sents = []
    for r in sorted(G.rationales.values(), key=lambda n: natural_id_key(n.id)):
        d = G.diagnoses.get(r.target)
        if d is None:
            continue
        prem = " and ".join(
            (G.observations[p].text if G.observations[p].polarity == Polarity.PRESENT
             else f"absence of {G.observations[p].text}") for p in r.premises if p in G.observations)
        verb = "supports" if r.relation == Relation.SUPPORTS else "argues against"
        sents.append(f"{prem.capitalize()} {verb} {d.leaf}. {r.claim}".strip())
    return " ".join(sents)


def main(cfg: Dict[str, Any]) -> None:
    import json

    import torch
    from transformers import Trainer, TrainingArguments

    from clinblueprint.models.backbone import LoraSpec, from_dict, load_model, load_tokenizer

    seed_everything(cfg.get("seed", 0))
    fold_map = json.load(open(cfg["fold_file"])) if cfg.get("fold_file") else None
    corpus = load_graph_corpus(cfg["train_files"], cfg.get("exclude_folds"), fold_map)
    exs = []
    for rec, G in corpus:
        reasoning = rec.rationale if (cfg.get("prefer_clinician_text", True) and rec.rationale) else graph_to_prose(G)
        cands = [split_dx_key(k)[1] for k in rec.candidates] if rec.candidates else None
        exs.append(TextExample(P.cot_messages(rec.note, cands),
                               f"{reasoning}\nFinal diagnosis: {split_dx_key(rec.final_dx)[1]}", "cot", rec.case_id))
    random.Random(cfg.get("seed", 0)).shuffle(exs)
    bcfg = from_dict(cfg["model"])
    tok = load_tokenizer(bcfg)
    tok.padding_side = "right"
    toks = [t for t in (tokenize_example(tok, e, bcfg.max_length) for e in exs) if t is not None]
    model = load_model(bcfg, lora=LoraSpec(**cfg.get("lora", {})), for_training=True)
    t = cfg.get("train", {})

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(toks)

        def __getitem__(self, i):
            return toks[i]

    args = TrainingArguments(
        output_dir=cfg["output_dir"], per_device_train_batch_size=t.get("batch_size", 1),
        gradient_accumulation_steps=t.get("grad_accum", 16), learning_rate=t.get("lr", 1e-4),
        num_train_epochs=t.get("epochs", 2), warmup_ratio=t.get("warmup_ratio", 0.03), logging_steps=10,
        save_steps=t.get("save_steps", 500), save_total_limit=2, bf16=True,
        gradient_checkpointing=bcfg.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=t.get("report_to", "none"), remove_unused_columns=False, seed=cfg.get("seed", 0))
    Trainer(model=model, args=args, train_dataset=_DS(), data_collator=lambda b: pad_collate(b, tok.pad_token_id)).train()
    out = ensure_dir(cfg["output_dir"])
    model.save_pretrained(str(Path(out) / "adapter"))
    tok.save_pretrained(str(Path(out) / "adapter"))
    write_json(Path(out) / "cot_sft_summary.json", {"n_examples": len(toks)})
