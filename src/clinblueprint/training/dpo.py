"""Stage 6: DPO for the generative in-context variant (note + graph -> diagnosis).

Preference pairs (built by ``build_pairs``):
  valid correction / counterfactual  chosen = diagnosis the HARD readout gives on
                                     the edited graph (the structural readout acts as
                                     teacher); rejected = the "stubborn" pre-edit
                                     diagnosis
  invalid assertion                  chosen = "FLAG: <evidence>"; rejected = following it
  irrelevant                         chosen = unchanged diagnosis; rejected = a changed one

The reference model is the policy with its LoRA adapter disabled
(``peft.disable_adapter``), so no second model copy is held in memory.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from clinblueprint.checks.hard_checks import check_graph
from clinblueprint.edits.generator import EditGenerator
from clinblueprint.edits.ops import apply_edit
from clinblueprint.edits.types import Edit, EditClass, EditMode
from clinblueprint.edits.validator import RuleChecker
from clinblueprint.readout.hard import HardReadout
from clinblueprint.schema.graph import split_dx_key
from clinblueprint.schema.serialize import serialize_graph
from clinblueprint.utils.io import ensure_dir, seed_everything, write_json

IN_CONTEXT_SYSTEM = ("You are a clinical reasoning assistant. Your diagnosis must follow the reasoning graph. "
                     "Hypothetical edits must be answered. Clinician assertions that contradict the record must be "
                     "flagged with the contradicting evidence instead of being followed.")


@dataclass
class Pair:
    messages: List[Dict[str, str]]
    chosen: str
    rejected: str
    kind: str
    case_id: str


def in_context_messages(note: str, graph_text: str, edit: Optional[Edit] = None) -> List[Dict[str, str]]:
    edit_block = ""
    if edit is not None:
        tag = "HYPOTHETICAL" if edit.mode == EditMode.HYPOTHETICAL else "CLINICIAN ASSERTION"
        edit_block = f"\n\n{tag}: {edit.describe()}"
    return [{"role": "system", "content": IN_CONTEXT_SYSTEM},
            {"role": "user", "content": f"Clinical note:\n{note}\n\nReasoning graph:\n{graph_text}{edit_block}\n\n"
                                        "Answer with 'Final diagnosis: <diagnosis>' or 'FLAG: <reason>'."}]


def build_pairs(corpus, kg, bystander, seed: int = 0) -> List[Pair]:
    rng = random.Random(seed)
    gen = EditGenerator(kg, bystander, seed=seed)
    hard = HardReadout(kg)
    rules = RuleChecker()
    pairs: List[Pair] = []
    for rec, G in corpus:
        top0 = hard(G).argmax
        if top0 is None:
            continue
        before = f"Final diagnosis: {split_dx_key(top0)[1]}"
        for e in gen.generate_case(G, rec.note, rec.final_dx):
            msgs = in_context_messages(rec.note, serialize_graph(G).text, e)
            if e.edit_class == EditClass.INVALID:
                _, reasons, _ = rules.check(e, rec.note, G)
                flag = "FLAG: " + (reasons[0] if reasons else "the assertion contradicts the record")
                G2, _ = check_graph(apply_edit(G, e), rec.note, kg)
                follow = hard(G2).argmax or top0
                pairs.append(Pair(msgs, flag, f"Final diagnosis: {split_dx_key(follow)[1]}", "invalid", rec.case_id))
                continue
            G2, _ = check_graph(apply_edit(G, e), rec.note, kg)
            top1 = hard(G2).argmax
            if top1 is None:
                continue
            after = f"Final diagnosis: {split_dx_key(top1)[1]}"
            if e.edit_class == EditClass.IRRELEVANT:
                alts = [k for k in hard.candidates(G) if k != top0]
                if alts:
                    pairs.append(Pair(msgs, before, f"Final diagnosis: {split_dx_key(rng.choice(alts))[1]}",
                                      "irrelevant", rec.case_id))
            elif top1 != top0:
                pairs.append(Pair(msgs, after, before, e.edit_class.value, rec.case_id))
    return pairs


def main(cfg: Dict[str, Any]) -> None:
    import json

    import torch
    import yaml

    from clinblueprint.kg import DiagnosticKG
    from clinblueprint.models.backbone import LoraSpec, chat_prompt, from_dict, load_model, load_tokenizer
    from clinblueprint.training.losses import dpo_loss, sequence_logprob
    from clinblueprint.training.sft import load_graph_corpus

    seed_everything(cfg.get("seed", 0))
    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    fold_map = json.load(open(cfg["fold_file"])) if cfg.get("fold_file") else None
    corpus = load_graph_corpus(cfg["train_files"], cfg.get("exclude_folds"), fold_map)
    bystander = yaml.safe_load(open(cfg["bystander_facts"]))["facts"] if cfg.get("bystander_facts") else []
    pairs = build_pairs(corpus, kg, bystander, cfg.get("seed", 0))
    print(f"[dpo] {len(pairs)} preference pairs")

    bcfg = from_dict(cfg["model"])
    tok = load_tokenizer(bcfg)
    model = load_model(bcfg, lora=None if cfg.get("init_adapter") else LoraSpec(**cfg.get("lora", {})),
                       adapter_path=cfg.get("init_adapter"), for_training=True)
    dev = next(model.parameters()).device
    beta = cfg.get("beta", 0.1)

    def encode(p: Pair, completion: str):
        prompt = chat_prompt(tok, p.messages)
        pi = tok(prompt, add_special_tokens=False)["input_ids"][-(bcfg.max_length - 64):]
        ci = tok(completion + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        ids = torch.tensor([pi + ci], device=dev)
        cm = torch.tensor([[0] * len(pi) + [1] * len(ci)], device=dev)
        return ids, torch.ones_like(ids), cm

    # reference log-probs (adapter disabled) computed once
    refs = []
    with torch.no_grad(), model.disable_adapter():
        for p in pairs:
            refs.append(tuple(float(sequence_logprob(model, *encode(p, c))) for c in (p.chosen, p.rejected)))
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=cfg.get("lr", 5e-6))
    accum = cfg.get("grad_accum", 16)
    order = list(range(len(pairs)))
    log = []
    for epoch in range(cfg.get("epochs", 1)):
        random.Random(epoch).shuffle(order)
        for step, i in enumerate(order):
            p = pairs[i]
            pc = sequence_logprob(model, *encode(p, p.chosen))
            pr = sequence_logprob(model, *encode(p, p.rejected))
            rc, rr = refs[i]
            loss, acc = dpo_loss(pc, pr, torch.tensor([rc], device=dev), torch.tensor([rr], device=dev), beta)
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
            if step % 100 == 0:
                print(f"[dpo] epoch {epoch} step {step} loss={float(loss):.4f} acc={float(acc):.2f}")
                log.append({"epoch": epoch, "step": step, "loss": float(loss)})
    out = ensure_dir(cfg["output_dir"])
    model.save_pretrained(str(Path(out) / "adapter"))
    write_json(Path(out) / "dpo_log.json", {"n_pairs": len(pairs), "log": log,
                                             "pair_kinds": {k: sum(p.kind == k for p in pairs) for k in
                                                            {p.kind for p in pairs}}})
