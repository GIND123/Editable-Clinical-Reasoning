"""Build ClinEdit-Faith: cases (with a fixed candidate pool) + edits with gold effects.

    clinblueprint build-benchmark --config configs/eval/clinedit_faith.yaml

Portions:
  credentialed  DiReCT (gold physician trees) + ER-Reason SCT  -> PhysioNet release only
  public        MedCaseReasoning test (teacher graphs, physician-verified sample) -> open release

Candidate pool per case (shared by every system): gold category leaves + leaves of
``n_distractor_categories`` other categories (fixed seed), or the task's own list
(ER-Reason SCT differential, MIMIC-CDM pathologies).
Physician-verification sample: stratified by edit class x gold source.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from clinblueprint.data.records import CaseRecord
from clinblueprint.edits.generator import EditGenerator
from clinblueprint.schema.graph import ReasoningGraph, dx_key
from clinblueprint.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


def candidate_pool(rec: CaseRecord, kg, rng: random.Random, n_distractors: int = 2) -> List[str]:
    if rec.candidates:
        return rec.candidates
    if kg is None or rec.category not in kg.categories:
        return [rec.final_dx] if rec.final_dx else []
    cats = [rec.category] + rng.sample([c for c in kg.categories if c != rec.category], n_distractors)
    return [dx_key(c, leaf) for c in cats for leaf in kg.leaves(c)]


def verification_sample(edits, n: int, seed: int = 0):
    by = defaultdict(list)
    for e in edits:
        by[(e.edit_class.value, e.gold_source)].append(e)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by)))
    out = []
    for k in sorted(by):
        out += rng.sample(by[k], min(per, len(by[k])))
    return out[:n]


def main(cfg: Dict[str, Any]) -> None:
    import yaml

    from clinblueprint.data.sct import SCTItem
    from clinblueprint.kg import DiagnosticKG

    rng = random.Random(cfg.get("seed", 0))
    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    facts = yaml.safe_load(open(cfg["bystander_facts"]))["facts"] if cfg.get("bystander_facts") else []
    gen = EditGenerator(kg, facts, seed=cfg.get("seed", 0))
    out = ensure_dir(cfg["output_dir"])

    for portion, spec in cfg["portions"].items():
        cases, edits = [], []
        for row in read_jsonl(spec["graphs"]) if spec.get("graphs") else []:
            rec = CaseRecord(**row["record"])
            G = ReasoningGraph.from_dict(row["graph"])
            rec.candidates = candidate_pool(rec, kg, rng, cfg.get("n_distractor_categories", 2))
            rec.gold_graph = G.to_dict()
            cases.append(rec)
            edits += gen.generate_case(G, rec.note, rec.final_dx)
        if spec.get("sct"):
            items = [SCTItem(**d) for d in read_jsonl(spec["sct"])]
            seen = set()
            for it in items:
                if it.case_id not in seen:
                    seen.add(it.case_id)
                    cases.append(CaseRecord(case_id=it.case_id, dataset=it.source, provenance=it.provenance,
                                            note=it.vignette, candidates=[dx_key("open", d) for d in it.differential],
                                            final_dx=None, split="test"))
            edits += gen.from_sct(items)
        write_jsonl(Path(out) / f"{portion}.cases.jsonl", [c.model_dump(mode="json") for c in cases])
        write_jsonl(Path(out) / f"{portion}.edits.jsonl", [e.model_dump(mode="json") for e in edits])
        ver = verification_sample(edits, spec.get("n_verify", 250), cfg.get("seed", 0))
        write_jsonl(Path(out) / f"{portion}.verify_sample.jsonl", [e.model_dump(mode="json") for e in ver])
        counts = defaultdict(int)
        for e in edits:
            counts[e.edit_class.value] += 1
        write_json(Path(out) / f"{portion}.stats.json", {"n_cases": len(cases), "n_edits": len(edits),
                                                         "by_class": dict(counts), "n_verify": len(ver)})
        print(f"[benchmark] {portion}: {len(cases)} cases, {len(edits)} edits")
