"""QA sanity suites (no-regression check vs the untuned backbone).

Hugging Face IDs (verify revisions and pin them in configs/data/qa_suites.yaml):
  MedQA (USMLE, 4 options)  GBaker/MedQA-USMLE-4-options
  MedMCQA                   openlifescienceai/medmcqa   (validation split; test has no labels)
  PubMedQA (PQA-L)          qiaojin/PubMedQA, config pqa_labeled
  MedXpertQA (text)         TsinghuaC3I/MedXpertQA, config Text
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

SUITES = {
    "medqa": {"hf_id": "GBaker/MedQA-USMLE-4-options", "config": None, "split": "test"},
    "medmcqa": {"hf_id": "openlifescienceai/medmcqa", "config": None, "split": "validation"},
    "pubmedqa": {"hf_id": "qiaojin/PubMedQA", "config": "pqa_labeled", "split": "train"},
    "medxpertqa": {"hf_id": "TsinghuaC3I/MedXpertQA", "config": "Text", "split": "test"},
}


class MCQItem(BaseModel):
    item_id: str
    suite: str
    question: str
    options: Dict[str, str]
    answer: str
    context: Optional[str] = None


def _medqa(r, i):
    opts = r.get("options") or {}
    if isinstance(opts, list):
        opts = {o["key"]: o["value"] for o in opts}
    return MCQItem(item_id=f"medqa:{i}", suite="medqa", question=r["question"], options=opts,
                   answer=r.get("answer_idx") or r.get("answer"))


def _medmcqa(r, i):
    keys = ["A", "B", "C", "D"]
    opts = dict(zip(keys, [r["opa"], r["opb"], r["opc"], r["opd"]]))
    return MCQItem(item_id=f"medmcqa:{r.get('id', i)}", suite="medmcqa", question=r["question"], options=opts,
                   answer=keys[int(r["cop"])])


def _pubmedqa(r, i):
    ctx = r.get("context") or {}
    ctx_text = " ".join(ctx.get("contexts", [])) if isinstance(ctx, dict) else str(ctx)
    return MCQItem(item_id=f"pubmedqa:{r.get('pubid', i)}", suite="pubmedqa", question=r["question"],
                   options={"A": "yes", "B": "no", "C": "maybe"},
                   answer={"yes": "A", "no": "B", "maybe": "C"}[r["final_decision"]], context=ctx_text)


def _medxpert(r, i):
    return MCQItem(item_id=f"medxpertqa:{r.get('id', i)}", suite="medxpertqa", question=r["question"],
                   options=r["options"], answer=r["label"])


_PARSERS = {"medqa": _medqa, "medmcqa": _medmcqa, "pubmedqa": _pubmedqa, "medxpertqa": _medxpert}


def load_suite(name: str, limit: Optional[int] = None, cache_dir: Optional[str] = None) -> List[MCQItem]:
    from datasets import load_dataset  # lazy

    spec = SUITES[name]
    ds = load_dataset(spec["hf_id"], spec["config"], split=spec["split"], cache_dir=cache_dir) if spec["config"] \
        else load_dataset(spec["hf_id"], split=spec["split"], cache_dir=cache_dir)
    out = []
    for i, r in enumerate(ds):
        out.append(_PARSERS[name](dict(r), i))
        if limit and len(out) >= limit:
            break
    return out
