"""Fit the linear hard readout on (model-drafted or gold) training graphs per fold.

Fitting on *model-drafted* graphs (default) makes the readout robust to the
drafter's actual errors; fitting on gold graphs gives the "oracle graph" upper
bound (reported as an ablation).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

from clinblueprint.readout.hard import ReadoutConfig
from clinblueprint.readout.linear import LinearReadout
from clinblueprint.training.sft import load_graph_corpus
from clinblueprint.utils.io import ensure_dir, write_json


def main(cfg: Dict[str, Any]) -> None:
    import json

    from clinblueprint.kg import DiagnosticKG

    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    fold_map = json.load(open(cfg["fold_file"])) if cfg.get("fold_file") else None
    corpus = load_graph_corpus(cfg["train_files"], cfg.get("exclude_folds"), fold_map)
    graphs = [G for _, G in corpus]
    gold = [r.final_dx for r, _ in corpus]
    prior = Counter(gold)
    lr = LinearReadout(kg, ReadoutConfig(**cfg.get("readout", {})),
                       allow_bypass_features=cfg.get("allow_bypass_features", False))
    rep = lr.fit(graphs, gold, prior_counts=prior, epochs=cfg.get("epochs", 300), lr=cfg.get("lr", 0.1),
                 l2=cfg.get("l2", 1e-3), verbose=True)
    out = ensure_dir(cfg["output_dir"])
    write_json(Path(out) / "linear_readout.json", lr.state_dict())
    write_json(Path(out) / "fit_report.json", rep.__dict__)
    print(rep)
