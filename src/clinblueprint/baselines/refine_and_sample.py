"""Graph-producing baselines that reuse the ClinBlueprint readout.

* SchemaPromptingDrafter: the *untuned* backbone prompted with the drafter
  prompt (use LMGraphDrafter with the base model; nothing extra needed).
* SelfRefineDrafter: draft -> critique -> revise (Madaan et al. 2023).
  Intrinsic self-correction is unreliable without external feedback (Kamoi et
  al., TACL 2024); included because reviewers will ask for it.
* BestOfNDrafter: sample N graphs, keep the one with the best verifier score
  (hard-check grounding rate + optional PRM-style scorer).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from clinblueprint.checks.hard_checks import check_graph
from clinblueprint.models import prompts as P
from clinblueprint.models.backbone import chat_prompt, generate
from clinblueprint.schema.graph import ReasoningGraph
from clinblueprint.schema.serialize import GraphParseError, parse_graph_json, serialize_graph


class SelfRefineDrafter:
    def __init__(self, drafter, rounds: int = 1):
        self.d, self.rounds = drafter, rounds

    def draft(self, note: str, case_id: str, **kw: Any) -> ReasoningGraph:
        G = self.d.draft(note, case_id, **kw)
        for _ in range(self.rounds):
            g = serialize_graph(G).text
            crit = generate(self.d.model, self.d.tok, [chat_prompt(self.d.tok, [
                {"role": "user", "content": P.CRITIQUE_USER.format(note=note, graph=g)}])], max_new_tokens=600)[0]
            rev = generate(self.d.model, self.d.tok, [chat_prompt(self.d.tok, [
                {"role": "user", "content": P.REVISE_USER.format(critique=crit, note=note, graph=g)}])],
                max_new_tokens=self.d.cfg.max_new_tokens)[0]
            try:
                G = parse_graph_json(rev, case_id)
            except GraphParseError:
                break
        return G


class BestOfNDrafter:
    def __init__(self, drafter, n: int = 8, temperature: float = 0.8, kg=None,
                 scorer: Optional[Callable[[ReasoningGraph, str], float]] = None):
        self.d, self.n, self.temperature, self.kg = drafter, n, temperature, kg
        self.scorer = scorer

    def _score(self, G: ReasoningGraph, note: str) -> float:
        G2, rep = check_graph(G, note, self.kg)
        s = rep.grounding_rate - 0.1 * sum(v.severity == "error" for v in rep.violations)
        return s + (self.scorer(G2, note) if self.scorer else 0.0)

    def draft(self, note: str, case_id: str, **kw: Any) -> ReasoningGraph:
        old = self.d.cfg.temperature
        self.d.cfg.temperature = self.temperature
        try:
            cands = [self.d.draft(note, case_id, **kw) for _ in range(self.n)]
        finally:
            self.d.cfg.temperature = old
        return max(cands, key=lambda G: self._score(G, note))
