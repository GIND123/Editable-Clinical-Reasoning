"""Run the mechanistic suite on a fitted in-context model (and the CoT baseline).

    clinblueprint mech --config configs/eval/mech.yaml
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

from clinblueprint.compliance.guard import assert_local
from clinblueprint.data.records import CaseRecord
from clinblueprint.schema.graph import ReasoningGraph, split_dx_key
from clinblueprint.schema.serialize import serialize_graph
from clinblueprint.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def main(cfg: Dict[str, Any]) -> None:
    from clinblueprint.mech.attribution import integrated_gradients, segment_shares
    from clinblueprint.mech.segments import segment_prompt
    from clinblueprint.mech.swap_and_patch import _prompt, graph_swap_flip, layerwise_patching
    from clinblueprint.mech.three_path import summarize, three_path
    from clinblueprint.models.backbone import from_dict, load_model, load_tokenizer

    assert_local("hf", None, cfg.get("provenance", "physionet"))
    rng = random.Random(cfg.get("seed", 0))
    out = ensure_dir(cfg["output_dir"])
    rows = read_jsonl(cfg["cases"])
    cases = [CaseRecord(**r) for r in rows][: cfg.get("max_cases", 100)]
    graphs = {c.case_id: ReasoningGraph.from_dict(c.gold_graph) for c in cases if c.gold_graph}
    bcfg = from_dict({**cfg["model"], "attn_implementation": "eager"})
    tok = load_tokenizer(bcfg)
    model = load_model(bcfg, adapter_path=cfg.get("adapter"))

    def names(c):
        return [split_dx_key(k)[1] for k in c.candidates or []]

    # 1. three-path decomposition
    tp = []
    for c in cases:
        cands = names(c)
        gold = split_dx_key(c.final_dx)[1] if c.final_dx else None
        if gold not in cands or c.case_id not in graphs:
            continue
        r = three_path(model, tok, c.note, serialize_graph(graphs[c.case_id]).text, cands, cands.index(gold))
        tp.append(r)
        write_jsonl(Path(out) / "three_path.jsonl", [x.__dict__ for x in tp])
    write_json(Path(out) / "three_path.summary.json", summarize(tp))

    # 2. graph swap
    swaps = []
    pool = [c for c in cases if c.case_id in graphs and c.final_dx]
    for _ in range(cfg.get("n_swaps", 100)):
        a, b = rng.sample(pool, 2)
        if a.final_dx == b.final_dx:
            continue
        cands = sorted(set(names(a)) | set(names(b)))
        res = graph_swap_flip(model, tok, a.note, serialize_graph(graphs[b.case_id]).text, cands,
                              cands.index(split_dx_key(a.final_dx)[1]), cands.index(split_dx_key(b.final_dx)[1]))
        swaps.append({"a": a.case_id, "b": b.case_id, "result": res})
    write_jsonl(Path(out) / "graph_swap.jsonl", swaps)
    write_json(Path(out) / "graph_swap.summary.json",
               {k: float(np.mean([s["result"] == k for s in swaps])) for k in ("follows_graph", "follows_note", "other")}
               if swaps else {})

    # 3. minimal-pair patching on key observations
    from clinblueprint.edits.generator import EditGenerator

    gen = EditGenerator()
    patches = []
    for c in pool[: cfg.get("n_patch_cases", 30)]:
        G = graphs[c.case_id]
        keys = gen.key_observations(G, c.final_dx, k=1)
        alts = [n for n in names(c) if n != split_dx_key(c.final_dx)[1]]
        if not keys or not alts:
            continue
        r = layerwise_patching(model, tok, c.note, G, keys[0], split_dx_key(c.final_dx)[1], alts[0])
        if r is not None:
            patches.append({"case_id": c.case_id, **r})
    write_jsonl(Path(out) / "patching.jsonl", patches)

    # 4. attribution share (IG)
    attr = []
    for c in pool[: cfg.get("n_attr_cases", 30)]:
        gtext = serialize_graph(graphs[c.case_id]).text
        p = _prompt(tok, c.note, gtext)
        seg = segment_prompt(tok, p, [("note", c.note), ("graph", gtext)])
        tgt = tok(" " + split_dx_key(c.final_dx)[1], add_special_tokens=False)["input_ids"]
        a = integrated_gradients(model, seg.input_ids, tgt, n_steps=cfg.get("ig_steps", 16))
        attr.append({"case_id": c.case_id, **segment_shares(a, {k: seg.segments[k] for k in ("note", "graph", "other")})})
    write_jsonl(Path(out) / "attribution.jsonl", attr)
    if attr:
        write_json(Path(out) / "attribution.summary.json",
                   {k: float(np.mean([x[k] for x in attr if k in x])) for k in ("note", "graph", "other")})
