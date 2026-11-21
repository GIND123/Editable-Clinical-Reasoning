"""Stage 4: learned entailment verifier for the edit validator.

Training triples (premise window from the note, claim, label) are generated
from gold/silver graphs:
  entailment     "The patient has <grounded observation>."
  contradiction  negated grounded observation; numeric swap of a recorded value
  neutral        a finding from a *different* case not mentioned in this note
Optionally pre-trained on MedNLI (PhysioNet; local only). Calibrated with
isotonic regression on a dev split; the threshold is chosen for a target
sensitivity (default 0.80) and the false-flag rate is reported.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from clinblueprint.edits.validator import note_polarity
from clinblueprint.schema.graph import EvidenceSource, Polarity
from clinblueprint.utils.io import ensure_dir, seed_everything, write_json

LABELS = ["entailment", "neutral", "contradiction"]


def make_triples(corpus, rng: random.Random, window: int = 600) -> List[Tuple[str, str, int]]:
    pool = [o.text for _, G in corpus for o in G.observations.values()]
    out = []
    for rec, G in corpus:
        for o in G.observations.values():
            if o.source != EvidenceSource.NOTE or o.span is None or o.polarity != Polarity.PRESENT:
                continue
            if note_polarity(rec.note, o.text)[0] != Polarity.PRESENT:
                continue
            lo = max(0, o.span.start - window // 2)
            prem = rec.note[lo:lo + window]
            out.append((prem, f"The patient has {o.text}.", 0))
            out.append((prem, f"The patient does not have {o.text}.", 2))
            nums = re.findall(r"\d+(?:\.\d+)?", o.text)
            if nums:
                v = float(nums[0])
                out.append((prem, f"The patient has {o.text.replace(nums[0], f'{v * 0.5:g}', 1)}.", 2))
            other = rng.choice(pool)
            if other.lower() not in rec.note.lower():
                out.append((prem, f"The patient has {other}.", 1))
    rng.shuffle(out)
    return out


class HFEntailmentVerifier:
    """Wraps a sequence-classification model (3 labels) for EditValidator."""

    def __init__(self, model, tokenizer, max_length: int = 1024):
        self.model, self.tok, self.max_length = model, tokenizer, max_length

    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        import torch

        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation="only_first", max_length=self.max_length)
        enc = {k: v.to(next(self.model.parameters()).device) for k, v in enc.items()}
        with torch.no_grad():
            p = torch.softmax(self.model(**enc).logits.float()[0], -1).tolist()
        return dict(zip(LABELS, p))


def main(cfg: Dict[str, Any]) -> None:
    import json

    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from sklearn.isotonic import IsotonicRegression
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from clinblueprint.training.sft import load_graph_corpus

    seed_everything(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    fold_map = json.load(open(cfg["fold_file"])) if cfg.get("fold_file") else None
    corpus = load_graph_corpus(cfg["train_files"], cfg.get("exclude_folds"), fold_map)
    triples = make_triples(corpus, rng)
    n_dev = max(50, int(0.1 * len(triples)))
    dev, train = triples[:n_dev], triples[n_dev:]
    print(f"[validator] train={len(train)} dev={len(dev)}")

    name = cfg["model"]["name_or_path"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, torch_dtype=torch.bfloat16,
                                                               device_map="auto")
    model.config.pad_token_id = tok.pad_token_id
    lc = cfg.get("lora", {})
    model = get_peft_model(model, LoraConfig(r=lc.get("r", 16), lora_alpha=lc.get("alpha", 32),
                                             lora_dropout=lc.get("dropout", 0.05), task_type="SEQ_CLS",
                                             target_modules=lc.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])))
    dev_ = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-4))
    bs = cfg.get("batch_size", 8)
    for epoch in range(cfg.get("epochs", 2)):
        rng.shuffle(train)
        model.train()
        for i in range(0, len(train), bs):
            b = train[i:i + bs]
            enc = tok([x[0] for x in b], [x[1] for x in b], return_tensors="pt", padding=True,
                      truncation="only_first", max_length=cfg.get("max_length", 1024)).to(dev_)
            out = model(**enc, labels=torch.tensor([x[2] for x in b], device=dev_))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            if (i // bs) % 100 == 0:
                print(f"[validator] epoch {epoch} batch {i // bs} loss={float(out.loss):.4f}")
    model.eval()
    ver = HFEntailmentVerifier(model, tok)
    p_con = np.array([ver.predict(p, h)["contradiction"] for p, h, _ in dev])
    y = np.array([int(lbl == 2) for _, _, lbl in dev])
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_con, y)
    cal = iso.predict(p_con)
    target = cfg.get("target_sensitivity", 0.80)
    thr = float(np.quantile(cal[y == 1], 1 - target)) if y.sum() else 0.5
    ffr = float((cal[y == 0] >= thr).mean()) if (y == 0).any() else None
    out = ensure_dir(cfg["output_dir"])
    model.save_pretrained(str(Path(out) / "adapter"))
    tok.save_pretrained(str(Path(out) / "adapter"))
    write_json(Path(out) / "calibration.json", {
        "isotonic_x": iso.X_thresholds_.tolist(), "isotonic_y": iso.y_thresholds_.tolist(), "threshold": thr,
        "dev_false_flag_rate": ffr, "target_sensitivity": target, "n_dev": len(dev)})
