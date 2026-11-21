"""LM graph drafter: note -> ReasoningGraph (two-stage: category selection, then graph).

Training/inference consistency: at SFT time the KG block contains the gold
category tree plus ``n_distractor_categories`` random other trees, shuffled.
At test time it contains the top-k categories chosen in stage 1. Either way the
model must pick the right tree.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from clinblueprint.models import prompts as P
from clinblueprint.models.backbone import chat_prompt, generate, json_prefix_fn
from clinblueprint.schema.graph import (
    Diagnosis,
    EvidenceSource,
    Observation,
    Rationale,
    ReasoningGraph,
    Relation,
    dx_key,
)
from clinblueprint.schema.serialize import (
    DRAFTER_JSON_SCHEMA,
    GraphParseError,
    extract_json_object,
    parse_graph_json,
    serialize_graph,
)


@dataclass
class DrafterConfig:
    k_categories: int = 3
    max_new_tokens: int = 3072
    temperature: float = 0.0
    retry_temperature: float = 0.7
    n_retries: int = 2
    constrained: bool = True
    include_schema_in_prompt: bool = True
    max_input_tokens: int = 7000
    use_premise: bool = True


def kg_block(kg, categories: Sequence[str], use_premise: bool = True) -> str:
    parts = []
    for c in categories:
        tree = kg.describe(c) if use_premise else "\n".join(f"- {n}" for n in kg.nodes(c))
        parts.append(f"## Category: {c}\n{tree}")
    return "\n\n".join(parts)


def training_categories(kg, gold_category: str, n_distractors: int, rng: random.Random) -> List[str]:
    others = [c for c in kg.categories if c != gold_category]
    cats = [gold_category] + rng.sample(others, min(n_distractors, len(others)))
    rng.shuffle(cats)
    return cats


class LMGraphDrafter:
    def __init__(self, model, tokenizer, kg=None, cfg: Optional[DrafterConfig] = None, gen_kwargs: Optional[dict] = None):
        self.model, self.tok, self.kg = model, tokenizer, kg
        self.cfg = cfg or DrafterConfig()
        self.gen_kwargs = gen_kwargs or {}
        self._prefix = json_prefix_fn(tokenizer, DRAFTER_JSON_SCHEMA) if self.cfg.constrained else None

    def _gen(self, messages, temperature: float, constrained: bool = False, max_new_tokens: Optional[int] = None) -> str:
        prompt = chat_prompt(self.tok, messages)
        return generate(self.model, self.tok, [prompt], temperature=temperature,
                        max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                        prefix_allowed_tokens_fn=self._prefix if constrained else None,
                        max_input_tokens=self.cfg.max_input_tokens, **self.gen_kwargs)[0]

    def select_categories(self, note: str) -> List[str]:
        if self.kg is None:
            return []
        out = self._gen(P.category_messages(note, self.kg.categories, self.cfg.k_categories), 0.0, max_new_tokens=128)
        try:
            cats = json.loads(out[out.find("["):out.rfind("]") + 1])
        except Exception:
            cats = [c for c in self.kg.categories if c.lower() in out.lower()]
        cats = [c for c in cats if c in self.kg.categories][: self.cfg.k_categories]
        return cats or self.kg.categories[: self.cfg.k_categories]

    def draft(self, note: str, case_id: str, categories: Optional[Sequence[str]] = None, **kw: Any) -> ReasoningGraph:
        cats = list(categories) if categories is not None else self.select_categories(note)
        kgt = kg_block(self.kg, cats, self.cfg.use_premise) if (self.kg is not None and cats) else \
            "(open-ended: name diagnoses freely, category 'open')"
        msgs = P.drafter_messages(note, kgt, self.cfg.include_schema_in_prompt)
        last_err = None
        for attempt in range(1 + self.cfg.n_retries):
            temp = self.cfg.temperature if attempt == 0 else self.cfg.retry_temperature
            text = self._gen(msgs, temp, constrained=self._prefix is not None)
            try:
                G = parse_graph_json(text, case_id)
                G.meta.update({"categories_offered": cats, "attempts": attempt + 1})
                return G
            except GraphParseError as e:
                last_err = str(e)
        return ReasoningGraph(case_id=case_id, meta={"parse_failed": True, "error": last_err, "categories_offered": cats})

    # ------------------------------------------------------- re-reasoning hooks
    def candidate_labels(self, G: ReasoningGraph) -> List[str]:
        labels = {d.key for d in G.diagnoses.values()}
        if self.kg is not None:
            for c in {d.category for d in G.diagnoses.values()} & set(self.kg.categories):
                labels.update(dx_key(c, leaf) for leaf in self.kg.leaves(c))
        return sorted(labels)

    def link_new_observation(self, G: ReasoningGraph, obs_id: str, note: str, **kw: Any) -> ReasoningGraph:
        o = G.observations[obs_id]
        cands = kw.get("candidates") or self.candidate_labels(G)
        text = self._gen(P.link_messages(serialize_graph(G).text, cands, o.text, o.source.value), 0.0, max_new_tokens=512)
        s, e = text.find("["), text.rfind("]")
        try:
            links = json.loads(text[s:e + 1]) if s >= 0 and e > s else []
        except json.JSONDecodeError:
            links = []
        G = G.copy_deep()
        for ln in links if isinstance(links, list) else []:
            if not isinstance(ln, dict) or not ln.get("leaf"):
                continue
            cat = ln.get("category") or "open"
            d = G.find_dx(ln["leaf"], cat)
            if d is None:
                d = Diagnosis(id=G.new_id("D"), category=cat, leaf=str(ln["leaf"]))
                G.add(d)
            try:
                strength = min(1.0, max(0.0, float(ln.get("strength", 0.5))))
            except (TypeError, ValueError):
                strength = 0.5
            rel = Relation.REFUTES if str(ln.get("relation", "")).lower().startswith("ref") else Relation.SUPPORTS
            G.add(Rationale(id=G.new_id("R"), premises=[obs_id], claim=str(ln.get("claim", "")), target=d.id,
                            relation=rel, strength=strength, premise_polarity={obs_id: o.polarity}))
        G.meta.setdefault("links_proposed", {})[obs_id] = links
        return G

    def ground_or_repair(self, G: ReasoningGraph, node_id: str, note: str, problem: str = "high risk") -> ReasoningGraph:
        node = G.node(node_id)
        text = self._gen([{"role": "user", "content": P.REPAIR_USER.format(
            problem=problem, node=node.model_dump_json(), note=note)}], 0.0, max_new_tokens=512)
        obj = extract_json_object(text) or {}
        G = G.copy_deep()
        if obj.get("delete"):
            G.remove(node_id)
            return G
        if isinstance(node, Observation) and obj.get("text"):
            G.observations[node_id].text = str(obj["text"])
            G.observations[node_id].span = None
            G.observations[node_id].source = EvidenceSource.NOTE
        elif isinstance(node, Rationale) and obj.get("claim"):
            G.rationales[node_id].claim = str(obj["claim"])
        return G
