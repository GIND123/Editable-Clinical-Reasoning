"""Stage 2: graph SFT (drafter + verbaliser), LoRA on a 4-8B backbone, one 48 GB GPU.

    clinblueprint train-sft --config configs/train/sft_qwen3_4b.yaml
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from clinblueprint.data.records import CaseRecord
from clinblueprint.readout.hard import HardReadout
from clinblueprint.schema.graph import ReasoningGraph
from clinblueprint.training.examples import drafter_example, pad_collate, tokenize_example, verbalizer_example
from clinblueprint.utils.io import ensure_dir, read_jsonl, seed_everything, write_json


def load_graph_corpus(paths: List[str], exclude_folds=None, fold_map=None) -> List[tuple]:
    """Rows of processed jsonl: {"record": CaseRecord, "graph": ReasoningGraph, "quality": {...}}."""
    out = []
    for p in paths:
        for row in read_jsonl(p):
            rec = CaseRecord(**row["record"])
            if exclude_folds is not None and fold_map is not None and fold_map.get(rec.case_id) in exclude_folds:
                continue
            G = ReasoningGraph.from_dict(row["graph"])
            out.append((rec, G))
    return out


def build_examples(corpus, kg, cfg: Dict[str, Any]) -> list:
    rng = random.Random(cfg.get("seed", 0))
    ro = HardReadout(kg)
    exs = []
    for rec, G in corpus:
        exs.append(drafter_example(rec, G, kg, rng, cfg.get("n_distractor_categories", 2)))
        if rng.random() < cfg.get("verbalizer_fraction", 0.5):
            exs.append(verbalizer_example(G, ro(G)))
    rng.shuffle(exs)
    return exs


def main(cfg: Dict[str, Any]) -> None:
    import torch
    from transformers import Trainer, TrainingArguments

    from clinblueprint.kg import DiagnosticKG
    from clinblueprint.models.backbone import LoraSpec, from_dict, load_model, load_tokenizer

    seed_everything(cfg.get("seed", 0))
    out_dir = ensure_dir(cfg["output_dir"])
    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    fold_map = None
    if cfg.get("fold_file"):
        import json

        fold_map = json.load(open(cfg["fold_file"]))
    excl = cfg.get("exclude_folds")
    corpus = load_graph_corpus(cfg["train_files"], excl, fold_map)
    exs = build_examples(corpus, kg, cfg)

    bcfg = from_dict(cfg["model"])
    tok = load_tokenizer(bcfg)
    tok.padding_side = "right"
    toks = [t for t in (tokenize_example(tok, e, bcfg.max_length) for e in exs) if t is not None]
    print(f"[sft] {len(corpus)} graphs -> {len(exs)} examples -> {len(toks)} tokenised")
    lora = LoraSpec(**cfg.get("lora", {}))
    model = load_model(bcfg, lora=lora, for_training=True)

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(toks)

        def __getitem__(self, i):
            return toks[i]

    t = cfg.get("train", {})
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=t.get("batch_size", 1),
        gradient_accumulation_steps=t.get("grad_accum", 16), learning_rate=t.get("lr", 1e-4),
        num_train_epochs=t.get("epochs", 2), warmup_ratio=t.get("warmup_ratio", 0.03),
        lr_scheduler_type=t.get("scheduler", "cosine"), logging_steps=10, save_steps=t.get("save_steps", 500),
        save_total_limit=2, bf16=True, gradient_checkpointing=bcfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=t.get("report_to", "none"),
        remove_unused_columns=False, seed=cfg.get("seed", 0), dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=args, train_dataset=_DS(),
                      data_collator=lambda b: pad_collate(b, tok.pad_token_id))
    trainer.train()
    model.save_pretrained(str(Path(out_dir) / "adapter"))
    tok.save_pretrained(str(Path(out_dir) / "adapter"))
    write_json(Path(out_dir) / "sft_summary.json", {"n_graphs": len(corpus), "n_examples": len(toks), "cfg": cfg})
