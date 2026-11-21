"""Run ClinEdit-Faith and accuracy evaluation for any EditableSystem.

    clinblueprint evaluate --config configs/eval/evaluate.yaml
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from clinblueprint.data.records import CaseRecord
from clinblueprint.edits.types import Edit
from clinblueprint.eval.accuracy import dx_match
from clinblueprint.eval.bootstrap import cluster_bootstrap
from clinblueprint.eval.mediation import EditResult, per_case_followed, summarize_mediation
from clinblueprint.eval.nonsycophancy import summarize_nonsycophancy
from clinblueprint.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def run_edits(system, cases: Dict[str, CaseRecord], edits: Sequence[Edit], reason_mode: str = "frozen",
              progress: bool = True) -> Dict[str, Any]:
    by_case: Dict[str, List[Edit]] = defaultdict(list)
    for e in edits:
        by_case[e.case_id].append(e)
    results: List[EditResult] = []
    n_na = 0
    it = list(by_case.items())
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, desc=f"{system.name}")
        except ImportError:
            pass
    for cid, es in it:
        case = cases.get(cid)
        if case is None:
            continue
        state = system.prepare(case)
        d0 = system.differential(case, state)
        for e in es:
            r = system.apply(case, state, e, reason_mode)
            if not r.applicable:
                n_na += 1
                continue
            results.append(EditResult(
                edit_id=e.edit_id, case_id=cid, edit_class=e.edit_class.value, mode=e.mode.value, op=e.op.value,
                expected_effect=e.expected_effect.value, measured_dx=e.measured_dx,
                p_before=d0.p(e.measured_dx), p_after=r.after.p(e.measured_dx),
                top_before=d0.argmax, top_after=r.after.argmax, flagged=r.flagged, tv=d0.tv(r.after),
                implied_effect=e.implied_effect.value if e.implied_effect else None,
                expected_magnitude=e.expected_magnitude, expected_top=e.expected_top,
                p_contradict=r.p_contradict, system=system.name, meta={"gold_source": e.gold_source},
            ))
    return {"results": results, "n_not_applicable": n_na, "coverage": len(results) / max(1, len(results) + n_na)}


def run_accuracy(system, cases: Iterable[CaseRecord], kg=None, mode: str = "exact") -> List[Dict[str, Any]]:
    rows = []
    for case in cases:
        state = system.prepare(case)
        d = system.differential(case, state)
        top = [k for k, _ in d.top(3)]
        rows.append({"case_id": case.case_id, "patient_id": case.patient_id, "gold": case.final_dx,
                     "top1": top[0] if top else None, "top3": top,
                     "correct": bool(top) and dx_match(top[0], case.final_dx, kg=kg, mode=mode),
                     "correct_top3": any(dx_match(t, case.final_dx, kg=kg, mode=mode) for t in top)})
    return rows


def summarize(edit_run: Dict[str, Any], eps: float = 0.02, eps_tv: float = 0.05, n_boot: int = 2000) -> Dict[str, Any]:
    rs = edit_run["results"]
    s = {"mediation": summarize_mediation(rs, eps, eps_tv), "nonsycophancy": summarize_nonsycophancy(rs, eps),
         "coverage": edit_run["coverage"], "n_not_applicable": edit_run["n_not_applicable"]}
    v, c = per_case_followed(rs, eps, eps_tv)
    if v.size:
        s["mediation"]["mediation_rate_ci95"] = cluster_bootstrap(v, c, n_boot=n_boot)[1:]
    return s


def load_benchmark(path_cases: str, path_edits: str):
    cases = {r["case_id"]: CaseRecord(**r) for r in read_jsonl(path_cases)}
    edits = [Edit(**e) for e in read_jsonl(path_edits)]
    return cases, edits


def main(cfg: Dict[str, Any]) -> None:
    from clinblueprint.eval.system_factory import build_system

    out = ensure_dir(cfg["output_dir"])
    cases, edits = load_benchmark(cfg["benchmark"]["cases"], cfg["benchmark"]["edits"])
    shared: Dict[str, Any] = {}
    for spec in cfg["systems"]:
        system = build_system(spec, cfg, shared)
        for mode in cfg.get("reason_modes", ["frozen"]):
            run = run_edits(system, cases, edits, reason_mode=mode)
            tag = f"{system.name}__{mode}"
            write_jsonl(Path(out) / f"{tag}.edits.jsonl", [r.to_dict() for r in run["results"]])
            write_json(Path(out) / f"{tag}.summary.json", summarize(run, cfg.get("eps", 0.02), cfg.get("eps_tv", 0.05)))
        if cfg.get("accuracy", True):
            acc = run_accuracy(system, cases.values(), mode=cfg.get("match_mode", "exact"))
            write_jsonl(Path(out) / f"{system.name}.accuracy.jsonl", acc)
            vals = [float(r["correct"]) for r in acc]
            write_json(Path(out) / f"{system.name}.accuracy.summary.json", {
                "accuracy": cluster_bootstrap(vals, [r["patient_id"] or r["case_id"] for r in acc]),
                "n": len(acc)})
