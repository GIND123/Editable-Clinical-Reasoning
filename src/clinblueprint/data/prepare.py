"""Build processed, unified JSONL files from raw downloads.

    clinblueprint prepare-data --config configs/data/prepare.yaml

Outputs (under data/processed/):
  direct.graphs.jsonl          {"record", "graph"} with checked gold graphs
  direct.folds.json            {case_id: fold} patient-level, category-stratified
  medcasereasoning.{train,test}.records.jsonl
  er_reason.records.jsonl, er_reason.sct.jsonl
  mimic_cdm.records.jsonl
  mimic_note.records.jsonl     (DiReCT subjects excluded)
  dedup_report.json            PMCID / subject overlap and near-duplicate audit
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from clinblueprint.utils.io import ensure_dir, write_json, write_jsonl


def main(cfg: Dict[str, Any]) -> None:
    out = ensure_dir(cfg.get("output_dir", "data/processed"))
    report: Dict[str, Any] = {}
    kg = None
    if cfg.get("kg_dir") and Path(cfg["kg_dir"]).exists():
        from clinblueprint.kg import DiagnosticKG

        kg = DiagnosticKG.from_dir(cfg["kg_dir"])

    direct_recs = []
    if cfg.get("direct_dir") and Path(cfg["direct_dir"]).exists():
        from clinblueprint.checks import check_graph
        from clinblueprint.data.direct import load_direct
        from clinblueprint.data.splits import grouped_stratified_kfold
        from clinblueprint.schema import ReasoningGraph

        direct_recs = load_direct(cfg["direct_dir"])
        if cfg.get("direct_subject_map") and Path(cfg["direct_subject_map"]).exists():
            import json

            m = json.load(open(cfg["direct_subject_map"]))
            for r in direct_recs:
                r.patient_id = r.patient_id or m.get(r.meta.get("file"))
        rows, gr = [], []
        for r in direct_recs:
            G, rep = check_graph(ReasoningGraph.from_dict(r.gold_graph), r.note, kg,
                                 section_offsets=r.section_offsets, repair=False)
            gr.append(rep.grounding_rate)
            rows.append({"record": r.model_dump(mode="json"), "graph": G.to_dict(),
                         "check": {"grounding": rep.grounding_rate, "codes": dict(rep.codes())}})
        write_jsonl(out / "direct.graphs.jsonl", rows)
        folds = grouped_stratified_kfold(direct_recs, k=cfg.get("folds", 5), seed=cfg.get("seed", 0))
        write_json(out / "direct.folds.json", folds)
        report["direct"] = {"n": len(rows), "mean_gold_grounding": sum(gr) / max(1, len(gr))}

    mcr_test = []
    if cfg.get("medcasereasoning", {}).get("enabled", False):
        from clinblueprint.data.medcasereasoning import load_medcasereasoning

        for split in ("train", "test"):
            recs = load_medcasereasoning(split, cfg["medcasereasoning"].get(f"{split}_jsonl"))
            write_jsonl(out / f"medcasereasoning.{split}.records.jsonl", [r.model_dump(mode="json") for r in recs])
            report[f"medcasereasoning_{split}"] = len(recs)
            if split == "test":
                mcr_test = recs

    if cfg.get("er_reason_dir") and Path(cfg["er_reason_dir"]).exists():
        from clinblueprint.data.er_reason import load_er_reason_cases, load_er_reason_sct

        cases = load_er_reason_cases(cfg["er_reason_dir"], cfg.get("er_reason_input_mode", "one_liner_plus_history"))
        write_jsonl(out / "er_reason.records.jsonl", [r.model_dump(mode="json") for r in cases])
        items = load_er_reason_sct(cfg["er_reason_dir"])
        write_jsonl(out / "er_reason.sct.jsonl", [i.model_dump(mode="json") for i in items])
        report["er_reason"] = {"cases": len(cases), "sct_items": len(items),
                               "label_in_input": sum(bool(c.meta.get("label_in_input")) for c in cases)}

    if cfg.get("mimic_cdm_dir") and Path(cfg["mimic_cdm_dir"]).exists():
        from clinblueprint.data.mimic_cdm import load_mimic_cdm

        recs = load_mimic_cdm(cfg["mimic_cdm_dir"])
        write_jsonl(out / "mimic_cdm.records.jsonl", [r.model_dump(mode="json") for r in recs])
        report["mimic_cdm"] = len(recs)

    if cfg.get("mimic_note_discharge") and Path(cfg["mimic_note_discharge"]).exists():
        from clinblueprint.data.mimic_note import iter_mimic_discharge, subjects_of

        excl = subjects_of(direct_recs)
        recs = list(iter_mimic_discharge(cfg["mimic_note_discharge"], excl, cfg.get("mimic_note_limit")))
        write_jsonl(out / "mimic_note.records.jsonl", [r.model_dump(mode="json") for r in recs])
        report["mimic_note"] = {"n": len(recs), "excluded_direct_subjects": len(excl)}

    if mcr_test and cfg.get("extra_public_corpora"):
        from clinblueprint.data.dedup import drop_overlap, ids_of
        from clinblueprint.data.pmc_cases import iter_multicare, iter_pmc_patients

        test_ids = ids_of(mcr_test)
        for name, spec in cfg["extra_public_corpora"].items():
            it = iter_pmc_patients(spec.get("limit")) if name == "pmc_patients" else iter_multicare(spec["path"], spec.get("limit"))
            recs = list(it)
            kept = drop_overlap(recs, test_ids)
            write_jsonl(out / f"{name}.records.jsonl", [r.model_dump(mode="json") for r in kept])
            report[name] = {"n": len(recs), "kept_after_dedup": len(kept)}

    write_json(out / "prepare_report.json", report)
    print(report)
