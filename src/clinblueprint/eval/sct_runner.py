"""Sequential SCT evaluation (Claim A): findings accumulate across timesteps.

For each ER-Reason encounter (or SCT-Bench case):
  1. Draft an initial graph from the vignette (one-liner).
  2. For timestep t: insert evidence_t as a NEW_FINDING; the model links it to
     the fixed differential (re_reason); re-check; read out.
  3. Record Δ log-odds of differential_t, and the full ranking after step t.
Δ values are mapped to −2..+2 with thresholds fitted on a DEV split of encounters.
Reports DxUpdate κ (per-timestep mean), direction, exact match, DxTrajectory ρ,
final top-1/top-3, with encounter-level bootstrap CIs.

    clinblueprint evaluate-sct --config configs/eval/sct.yaml
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from clinblueprint.data.sct import SCTItem
from clinblueprint.edits.ops import apply_edit
from clinblueprint.edits.types import Edit, EditClass, EditMode, EditOp, ExpectedEffect
from clinblueprint.eval.bootstrap import cluster_bootstrap
from clinblueprint.eval.sct import delta_to_likert, dxupdate_kappa, fit_likert_thresholds, spearman_rankings
from clinblueprint.schema.graph import ReasoningGraph, dx_key, split_dx_key


def run_sequential(pipe, items: Sequence[SCTItem]) -> List[Dict[str, Any]]:
    by: Dict[str, List[SCTItem]] = defaultdict(list)
    for it in items:
        by[it.case_id].append(it)
    rows: List[Dict[str, Any]] = []
    for cid, steps in by.items():
        steps = sorted(steps, key=lambda s: s.timestep or 0)
        vignette = steps[0].vignette
        fixed = [dx_key("open", d) for d in steps[0].differential]
        try:
            G = pipe.drafter.draft(vignette, cid)
        except KeyError:
            G = ReasoningGraph(case_id=cid)
        G, _ = pipe.check(G, vignette)
        for i, it in enumerate(steps):
            before = pipe.readout(G, fixed)
            e = Edit(edit_id=it.item_id, case_id=cid, op=EditOp.INSERT, mode=EditMode.ASSERTION,
                     edit_class=EditClass.NEW_FINDING, new_text=it.new_info,
                     measured_dx=dx_key("open", it.hypothesis), expected_effect=ExpectedEffect.NO_CHANGE)
            G2 = apply_edit(G, e)
            if hasattr(pipe.drafter, "link_new_observation"):
                G2 = pipe.drafter.link_new_observation(G2, G2.meta["last_inserted"], vignette, candidates=fixed)
            G2, _ = pipe.check(G2, vignette)
            after = pipe.readout(G2, fixed)
            G = G2
            rows.append({
                "case_id": cid, "item_id": it.item_id, "timestep": it.timestep, "hypothesis": it.hypothesis,
                "gold_update": it.gold_update, "delta_logit": after.logit(e.measured_dx) - before.logit(e.measured_dx),
                "ranking": [split_dx_key(k)[1] for k, _ in after.top(len(fixed))],
                "gold_ranking": it.gold_ranking, "is_final": i == len(steps) - 1,
            })
    return rows


def score(rows: List[Dict[str, Any]], dev_frac: float = 0.3, seed: int = 0, n_boot: int = 2000) -> Dict[str, Any]:
    cases = sorted({r["case_id"] for r in rows})
    random.Random(seed).shuffle(cases)
    dev = set(cases[: int(len(cases) * dev_frac)])
    dv = [r for r in rows if r["case_id"] in dev and r["gold_update"] is not None]
    te = [r for r in rows if r["case_id"] not in dev and r["gold_update"] is not None]
    t1, t2 = fit_likert_thresholds([r["delta_logit"] for r in dv], [r["gold_update"] for r in dv]) if dv else (0.25, 1.0)
    for r in rows:
        r["pred_update"] = delta_to_likert(r["delta_logit"], t1, t2)
    k = dxupdate_kappa([(r["timestep"], r["gold_update"], r["pred_update"]) for r in te])
    rhos = [spearman_rankings(r["ranking"], r["gold_ranking"]) for r in te if r.get("gold_ranking")]
    rhos = [x for x in rhos if x is not None]
    fin = [r for r in te if r["is_final"] and r.get("gold_ranking")]
    top1 = [float(r["ranking"][0].lower() == r["gold_ranking"][0].lower()) for r in fin if r["ranking"]]
    direction = [float(np.sign(r["pred_update"]) == np.sign(r["gold_update"])) for r in te]
    return {
        "thresholds": [t1, t2], "n_dev_cases": len(dev), "n_test_items": len(te), "dxupdate": k,
        "direction_ci95": cluster_bootstrap(direction, [r["case_id"] for r in te], n_boot=n_boot) if te else None,
        "dxtrajectory_rho": float(np.mean(rhos)) if rhos else None,
        "final_top1": float(np.mean(top1)) if top1 else None,
    }


def main(cfg: Dict[str, Any]) -> None:
    from clinblueprint.eval.system_factory import build_system
    from clinblueprint.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl

    items = [SCTItem(**d) for d in read_jsonl(cfg["sct_items"])]
    out = ensure_dir(cfg["output_dir"])
    for spec in cfg["systems"]:
        system = build_system(spec, cfg, {})
        rows = run_sequential(system.pipe, items)
        write_jsonl(Path(out) / f"{spec['name']}.sct_rows.jsonl", rows)
        write_json(Path(out) / f"{spec['name']}.sct_summary.json", score(rows, cfg.get("dev_frac", 0.3)))
