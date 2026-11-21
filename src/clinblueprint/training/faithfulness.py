"""Stage 5: faithfulness training of the *soft* bottleneck.

    L = L_task(CE on gold) + λ_cf · L_direction(counterfactual / correction edits)
                           + λ_inv · L_invariance(irrelevant edits)

Edits come from EditGenerator applied to training graphs (gold or silver);
the edited graph G' is re-checked (cascade) before encoding. Only the head is
trained by default (``train_encoder_lora: false``), so the backbone's slot
representations are fixed. Enabling LoRA on the encoder lets slot
representations adapt but must then be re-audited mechanistically.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from clinblueprint.checks.hard_checks import check_graph
from clinblueprint.edits.generator import EditGenerator
from clinblueprint.edits.ops import apply_edit
from clinblueprint.edits.types import EditClass, ExpectedEffect
from clinblueprint.readout.hard import HardReadout
from clinblueprint.training.sft import load_graph_corpus
from clinblueprint.utils.io import ensure_dir, seed_everything, write_json


def main(cfg: Dict[str, Any]) -> None:
    import json

    import torch
    import yaml

    from clinblueprint.kg import DiagnosticKG
    from clinblueprint.models.backbone import from_dict, load_model, load_tokenizer
    from clinblueprint.readout.soft import SlotEncoder, build_head, collate
    from clinblueprint.training.losses import counterfactual_direction_loss, invariance_loss, task_ce

    seed_everything(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    fold_map = json.load(open(cfg["fold_file"])) if cfg.get("fold_file") else None
    corpus = load_graph_corpus(cfg["train_files"], cfg.get("exclude_folds"), fold_map)
    bystander = yaml.safe_load(open(cfg["bystander_facts"]))["facts"] if cfg.get("bystander_facts") else []
    gen = EditGenerator(kg, bystander, seed=cfg.get("seed", 0))
    hard = HardReadout(kg)

    bcfg = from_dict(cfg["model"])
    tok = load_tokenizer(bcfg)
    tok.padding_side = "right"
    model = load_model(bcfg, adapter_path=cfg.get("encoder_adapter"), for_training=False)
    enc = SlotEncoder(model, tok, layer=cfg.get("layer", -1), encoding=cfg.get("encoding", "graph_only"),
                      exclude_dx_slots=cfg.get("exclude_dx_slots", True))
    d_model = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size
    head = build_head(d_model, **cfg.get("head", {})).to(next(model.parameters()).device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=0.01)
    lam_cf, lam_inv = cfg.get("lambda_cf", 1.0), cfg.get("lambda_inv", 1.0)
    margin = cfg.get("margin", 0.5)
    notes = {rec.case_id: rec.note for rec, _ in corpus}

    def forward(G, keys):
        e = enc.encode(G, notes.get(G.case_id))
        c = enc.encode_candidates(keys)
        b = collate([e], [c])
        return head(*[x.float() if x.is_floating_point() else x for x in b])[0]

    log: List[Dict[str, float]] = []
    for epoch in range(cfg.get("epochs", 3)):
        rng.shuffle(corpus)
        for step, (rec, G) in enumerate(corpus):
            keys = hard.candidates(G)
            if rec.final_dx not in keys or len(keys) < 2:
                continue
            gi = keys.index(rec.final_dx)
            lg0 = forward(G, keys)
            loss = task_ce(lg0.unsqueeze(0), torch.tensor([gi], device=lg0.device))
            parts = {"task": float(loss)}
            edits = gen.counterfactuals(G, rec.final_dx) + gen.irrelevant(G, rec.final_dx, k=1)
            for e in rng.sample(edits, min(cfg.get("edits_per_case", 3), len(edits))):
                G2, _ = check_graph(apply_edit(G, e), rec.note, kg)
                lg1 = forward(G2, keys)  # same candidate set before/after
                if e.edit_class == EditClass.IRRELEVANT:
                    li = invariance_loss(lg0.unsqueeze(0), lg1.unsqueeze(0))
                    loss = loss + lam_inv * li
                    parts["inv"] = parts.get("inv", 0.0) + float(li)
                elif e.expected_effect in (ExpectedEffect.INCREASE, ExpectedEffect.DECREASE):
                    mi = keys.index(e.measured_dx) if e.measured_dx in keys else gi
                    d = torch.tensor([1.0 if e.expected_effect == ExpectedEffect.INCREASE else -1.0], device=lg0.device)
                    lp0 = torch.log_softmax(lg0, -1)[mi:mi + 1]
                    lp1 = torch.log_softmax(lg1, -1)[mi:mi + 1]
                    lc = counterfactual_direction_loss(lp0, lp1, d, margin)
                    loss = loss + lam_cf * lc
                    parts["cf"] = parts.get("cf", 0.0) + float(lc)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                print(f"[faith] epoch {epoch} step {step} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
                log.append({"epoch": epoch, "step": step, **parts})
    out = ensure_dir(cfg["output_dir"])
    torch.save(head.state_dict(), Path(out) / "soft_head.pt")
    write_json(Path(out) / "faith_log.json", log)
