"""ClinBlueprint end-to-end (the algorithm in docs/ARCHITECTURE.md §2).

    G    = drafter.draft(note)                      # constrained decoding to schema
    G    = check(G, note, kg)                       # grounding, KG validity, cascade
    [optional Claim C: risk-gated ground_or_repair loop over high-risk nodes]
    ddx  = readout(G)                               # bottleneck: function of G only
    text = verbalizer(G)                            # sees only G slots + cited spans

Edits:
    intervene(G, edit)            hypothetical / causal intervention; never validated
    apply_clinician_edit(G, edit) assertion: validated; contradictions are flagged
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from clinblueprint.checks.hard_checks import CheckReport, check_graph
from clinblueprint.edits.ops import apply_edit
from clinblueprint.edits.types import Edit, EditMode, EditOp
from clinblueprint.edits.validator import EditValidator, Verdict
from clinblueprint.readout.differential import Differential
from clinblueprint.schema.graph import ReasoningGraph


class Drafter(Protocol):
    def draft(self, note: str, case_id: str, **kw: Any) -> ReasoningGraph: ...


@dataclass
class PipelineResult:
    differential: Differential
    graph: ReasoningGraph
    report: CheckReport
    text: Optional[str] = None
    repair_steps: int = 0


@dataclass
class EditOutcome:
    edit: Edit
    flagged: bool
    before: Differential
    after: Differential
    graph_after: ReasoningGraph
    verdict: Optional[Verdict] = None
    report_after: Optional[CheckReport] = None
    overridden: bool = False

    @property
    def delta(self) -> float:
        return self.after.p(self.edit.measured_dx) - self.before.p(self.edit.measured_dx)


class StaticDrafter:
    """Returns pre-built graphs (gold graphs, cached model drafts, or the toy demo).

    ``links`` maps an inserted observation text to rationale links so
    ``re_reason`` mode can be exercised without a model.
    """

    def __init__(self, graphs: Dict[str, ReasoningGraph], links: Optional[Dict[str, list]] = None):
        self.graphs = graphs
        self.links = links or {}

    def draft(self, note: str, case_id: str, **kw: Any) -> ReasoningGraph:
        return self.graphs[case_id].copy_deep()

    def link_new_observation(self, G: ReasoningGraph, obs_id: str, note: str, **kw: Any) -> ReasoningGraph:
        from clinblueprint.schema.graph import Diagnosis, Rationale

        text = G.observations[obs_id].text
        G = G.copy_deep()
        for link in self.links.get(text, []):
            d = G.find_dx(link.leaf, link.category)
            if d is None:
                d = Diagnosis(id=G.new_id("D"), category=link.category, leaf=link.leaf)
                G.add(d)
            G.add(Rationale(id=G.new_id("R"), premises=[obs_id], claim=link.claim, target=d.id,
                            relation=link.relation, strength=link.strength,
                            premise_polarity={obs_id: G.observations[obs_id].polarity}))
        return G


class ClinBlueprint:
    def __init__(
        self,
        drafter: Drafter,
        readout: Callable[..., Differential],
        *,
        kg=None,
        mnm: Optional[Dict[str, List[str]]] = None,
        validator: Optional[EditValidator] = None,
        verbalizer=None,
        risk=None,                 # Optional[risk.node_risk.NodeRiskGate] (Claim C)
        repair_budget: int = 0,
        check_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.drafter = drafter
        self.readout = readout
        self.kg = kg
        self.mnm = mnm
        self.validator = validator or EditValidator()
        self.verbalizer = verbalizer
        self.risk = risk
        self.repair_budget = repair_budget
        self.check_kwargs = check_kwargs or {}

    # ------------------------------------------------------------------ run
    def check(self, G: ReasoningGraph, note: str, section_offsets=None):
        return check_graph(G, note, self.kg, self.mnm, section_offsets=section_offsets, **self.check_kwargs)

    def run(self, note: str, case_id: str, *, fixed_candidates: Optional[Sequence[str]] = None,
            section_offsets=None, verbalize: bool = False) -> PipelineResult:
        G = self.drafter.draft(note, case_id)
        G, rep = self.check(G, note, section_offsets)
        steps = 0
        if self.risk is not None and self.repair_budget > 0 and hasattr(self.drafter, "ground_or_repair"):
            for _ in range(self.repair_budget):
                risky = self.risk.select(G, note)
                if not risky:
                    break
                for nid in risky:
                    if nid in G:
                        G = self.drafter.ground_or_repair(G, nid, note)
                G, rep = self.check(G, note, section_offsets)
                steps += 1
        ddx = self.readout(G, fixed_candidates)
        text = self.verbalizer.verbalize(G, ddx) if (verbalize and self.verbalizer is not None) else None
        return PipelineResult(ddx, G, rep, text, steps)

    # ---------------------------------------------------------------- edits
    def intervene(self, G: ReasoningGraph, edit: Edit, note: str, *, reason_mode: str = "frozen",
                  fixed_candidates: Optional[Sequence[str]] = None, section_offsets=None) -> EditOutcome:
        before = self.readout(G, fixed_candidates)
        G2 = apply_edit(G, edit)
        if (reason_mode == "re_reason" and edit.op == EditOp.INSERT and not edit.links
                and hasattr(self.drafter, "link_new_observation")):
            G2 = self.drafter.link_new_observation(G2, G2.meta["last_inserted"], note)
        G2, rep = self.check(G2, note, section_offsets)
        after = self.readout(G2, fixed_candidates)
        return EditOutcome(edit, False, before, after, G2, None, rep)

    def apply_clinician_edit(self, G: ReasoningGraph, edit: Edit, note: str, *, reason_mode: str = "frozen",
                             fixed_candidates: Optional[Sequence[str]] = None, override: bool = False,
                             section_offsets=None) -> EditOutcome:
        verdict = None
        if edit.mode == EditMode.ASSERTION:
            verdict = self.validator.validate(edit, note, G)
            if verdict.flagged and not override:
                d0 = self.readout(G, fixed_candidates)
                return EditOutcome(edit, True, d0, d0, G, verdict, None)
        out = self.intervene(G, edit, note, reason_mode=reason_mode, fixed_candidates=fixed_candidates,
                             section_offsets=section_offsets)
        out.verdict = verdict
        out.overridden = bool(override and verdict is not None and verdict.flagged)
        return out
