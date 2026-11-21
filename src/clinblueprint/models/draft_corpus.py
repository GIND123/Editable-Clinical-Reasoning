"""Run a (fine-tuned) drafter over records and save model-drafted graphs.

Used for: fitting the readout on the drafter's real errors, building
valid-correction edits (model graph vs gold graph), and caching graphs for
evaluation / the clinician study.

    clinblueprint draft-graphs --config configs/eval/draft_graphs.yaml
"""

from __future__ import annotations

from typing import Any, Dict

from clinblueprint.checks.hard_checks import check_graph
from clinblueprint.compliance.guard import assert_local
from clinblueprint.data.records import CaseRecord
from clinblueprint.utils.io import read_jsonl, write_jsonl


def main(cfg: Dict[str, Any]) -> None:
    from clinblueprint.kg import DiagnosticKG, load_must_not_miss
    from clinblueprint.models.backbone import from_dict, load_model, load_tokenizer
    from clinblueprint.models.drafter import DrafterConfig, LMGraphDrafter

    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    mnm = load_must_not_miss(cfg["must_not_miss"]) if cfg.get("must_not_miss") else None
    rows = []
    for p in cfg["input_files"]:
        for r in read_jsonl(p):
            rows.append(CaseRecord(**(r["record"] if "record" in r else r)))
    for prov in {r.provenance for r in rows}:
        assert_local("hf", None, prov)
    b = from_dict(cfg["model"])
    drafter = LMGraphDrafter(load_model(b, adapter_path=cfg.get("adapter")), load_tokenizer(b), kg,
                             DrafterConfig(**cfg.get("drafter", {})))
    out = []
    for rec in rows[: cfg.get("limit") or None]:
        G = drafter.draft(rec.note, rec.case_id)
        G, rep = check_graph(G, rec.note, kg, mnm, section_offsets=rec.section_offsets)
        out.append({"record": rec.model_dump(mode="json"), "graph": G.to_dict(),
                    "check": {"grounding": rep.grounding_rate, "codes": dict(rep.codes())}})
    write_jsonl(cfg["output_file"], out)
    print(f"[draft] wrote {len(out)} graphs to {cfg['output_file']}")
