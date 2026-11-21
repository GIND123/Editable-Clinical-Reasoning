"""Stage 1: silver reasoning graphs from clinician rationales with a LOCAL teacher.

    clinblueprint build-silver --config configs/data/silver.yaml

Per case: teacher prompt (case text + authors' reasoning + final diagnosis [+ KG
block if the category is known]) -> JSON graph -> hard checks -> quality filter:

  * parse OK and >= ``min_observations`` grounded observations
  * grounding rate >= ``min_grounding`` (default 0.8)
  * a terminal diagnosis matches the reference final diagnosis (string judge)
  * that diagnosis has >= 1 valid supporting rationale path
  * no error-severity violations after repair

The clinician audits a random 100 kept graphs (clinician/tasks/02_silver_graph_audit.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from clinblueprint.checks.hard_checks import check_graph
from clinblueprint.compliance.guard import assert_local
from clinblueprint.data.records import CaseRecord
from clinblueprint.eval.accuracy import string_judge
from clinblueprint.schema.graph import Relation, ReasoningGraph, split_dx_key
from clinblueprint.schema.serialize import GraphParseError, parse_graph_json
from clinblueprint.utils.io import ensure_dir, write_json, write_jsonl


def quality(G: ReasoningGraph, rep, rec: CaseRecord, min_obs: int = 2, min_grounding: float = 0.8) -> Tuple[bool, List[str]]:
    reasons = []
    if len(G.observations) < min_obs:
        reasons.append("too_few_observations")
    if rep.grounding_rate < min_grounding:
        reasons.append("low_grounding")
    if not rep.ok:
        reasons.append("error_violations")
    judge = string_judge(0.6)
    gold_leaf = split_dx_key(rec.final_dx or "")[1]
    term = [G.diagnoses[d] for d in G.terminal_diagnoses()]
    match = [d for d in term if judge(d.leaf, gold_leaf)]
    if not match:
        reasons.append("final_dx_mismatch")
    elif not any(G.rationales_for(d.id, Relation.SUPPORTS) for d in match):
        reasons.append("final_dx_unsupported")
    return not reasons, reasons


def silver_one(rec: CaseRecord, gen_fn, kg=None, mnm=None, n_retries: int = 2, kg_text_fn=None,
               min_grounding: float = 0.8) -> Tuple[Optional[ReasoningGraph], Dict[str, Any]]:
    from clinblueprint.models import prompts as P

    kg_text = kg_text_fn(rec) if kg_text_fn else None
    msgs = P.teacher_messages(rec.note, rec.rationale or "", split_dx_key(rec.final_dx or "")[1], kg_text)
    last: Dict[str, Any] = {}
    for attempt in range(1 + n_retries):
        text = gen_fn(msgs, attempt)
        try:
            G = parse_graph_json(text, rec.case_id)
        except GraphParseError as e:
            last = {"kept": False, "reasons": ["parse_error"], "error": str(e), "attempt": attempt}
            continue
        G, rep = check_graph(G, rec.note, kg, mnm, section_offsets=rec.section_offsets)
        ok, reasons = quality(G, rep, rec, min_grounding=min_grounding)
        last = {"kept": ok, "reasons": reasons, "grounding": rep.grounding_rate, "attempt": attempt,
                "violations": dict(rep.codes())}
        if ok:
            G.meta.update({"source": "silver", "teacher_attempt": attempt})
            return G, last
    return None, last


def main(cfg: Dict[str, Any]) -> None:
    from clinblueprint.kg import DiagnosticKG, load_must_not_miss
    from clinblueprint.models.backbone import chat_prompt, from_dict, generate, load_model, load_tokenizer
    from clinblueprint.models.drafter import kg_block
    from clinblueprint.utils.io import read_jsonl

    backend = cfg.get("teacher", {}).get("backend", "hf")
    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    mnm = load_must_not_miss(cfg["must_not_miss"]) if cfg.get("must_not_miss") else None
    recs = [CaseRecord(**r) for p in cfg["input_files"] for r in read_jsonl(p)]
    if cfg.get("limit"):
        recs = recs[: cfg["limit"]]
    for prov in {r.provenance for r in recs}:
        assert_local(backend, cfg.get("teacher", {}).get("endpoint"), prov)

    bcfg = from_dict(cfg["teacher"]["model"])
    tok = load_tokenizer(bcfg)
    model = load_model(bcfg)

    def gen_fn(msgs, attempt):
        return generate(model, tok, [chat_prompt(tok, msgs)], max_new_tokens=cfg.get("max_new_tokens", 3072),
                        temperature=0.0 if attempt == 0 else 0.7)[0]

    def kg_text_fn(rec):
        if kg is not None and rec.category in kg.categories:
            return kg_block(kg, [rec.category])
        return None

    kept, stats = [], []
    for rec in recs:
        G, st = silver_one(rec, gen_fn, kg, mnm, cfg.get("n_retries", 2), kg_text_fn, cfg.get("min_grounding", 0.8))
        stats.append({"case_id": rec.case_id, **st})
        if G is not None:
            kept.append({"record": rec.model_dump(mode="json"), "graph": G.to_dict(), "quality": st})
    out = ensure_dir(Path(cfg["output_file"]).parent)
    write_jsonl(cfg["output_file"], kept)
    write_jsonl(Path(out) / (Path(cfg["output_file"]).stem + ".stats.jsonl"), stats)
    write_json(Path(out) / (Path(cfg["output_file"]).stem + ".summary.json"),
               {"n_input": len(recs), "n_kept": len(kept), "keep_rate": len(kept) / max(1, len(recs))})
    print(f"[silver] kept {len(kept)}/{len(recs)}")
