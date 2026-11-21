"""Generate ClinEdit-Faith edit items with gold expected effects.

Gold sources, from strongest to weakest:
  direct_gold   physician reasoning tree (DiReCT): which observations support
                the gold diagnosis
  er_reason_sct / sct_bench   expert-panel belief-update judgements
  kg_premise    DiReCT KG diagnostic criteria (premise text overlap)
  construction  by construction (bystander facts; contradictions of cited spans)

A stratified sample (~500 items) is physician-verified
(clinician/tasks/03_edit_verification.md); the agreement rate is reported.
"""

from __future__ import annotations

import random
import re
from typing import Iterable, List, Optional, Sequence

from clinblueprint.edits.types import Edit, EditClass, EditMode, EditOp, ExpectedEffect, InsertLink
from clinblueprint.edits.validator import note_polarity
from clinblueprint.schema.graph import (
    EvidenceSource,
    Polarity,
    ReasoningGraph,
    Relation,
    normalize_label,
    split_dx_key,
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return {w for w in _WORD.findall((s or "").lower()) if len(w) > 2}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(1, len(ta)) if ta else 0.0


class EditGenerator:
    def __init__(self, kg=None, bystander_facts: Optional[Sequence[str]] = None, seed: int = 0):
        self.kg = kg
        self.bystander = list(bystander_facts or [])
        self.rng = random.Random(seed)
        self._n = 0

    def _id(self, case_id: str, tag: str) -> str:
        self._n += 1
        return f"{case_id}:{tag}:{self._n}"

    # ------------------------------------------------------------- helpers
    def gold_path_nodes(self, G: ReasoningGraph, gold_key: str) -> List[str]:
        cat, leaf = split_dx_key(gold_key)
        d = G.find_dx(leaf, cat if cat != "open" else None)
        if d is None:
            return []
        return [d.id] + G.dx_ancestors(d.id)

    def key_observations(self, G: ReasoningGraph, gold_key: str, k: int = 3) -> List[str]:
        """Observations premising supporting rationales on the gold path, most specific first."""
        cat, leaf = split_dx_key(gold_key)
        premise_text = ""
        if self.kg is not None and cat in getattr(self.kg, "categories", []):
            premise_text = " ".join(self.kg.premise(cat, n) for n in [leaf] + self.kg.ancestors(cat, leaf))
        scored = []
        for depth, did in enumerate(self.gold_path_nodes(G, gold_key)):
            for r in G.rationales_for(did, Relation.SUPPORTS):
                for p in r.premises:
                    if p in G.observations:
                        o = G.observations[p]
                        scored.append((depth, -_overlap(o.text + " " + r.claim, premise_text), p))
        seen, out = set(), []
        for _, _, p in sorted(scored):
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out[:k]

    # ------------------------------------------------------- counterfactual
    def counterfactuals(self, G_gold: ReasoningGraph, gold_key: str, k: int = 2) -> List[Edit]:
        out: List[Edit] = []
        cid = G_gold.case_id
        for oid in self.key_observations(G_gold, gold_key, k):
            o = G_gold.observations[oid]
            for op in (EditOp.REMOVE, EditOp.NEGATE):
                out.append(Edit(
                    edit_id=self._id(cid, f"cf-{op.value}"), case_id=cid, op=op, mode=EditMode.HYPOTHETICAL,
                    edit_class=EditClass.COUNTERFACTUAL, target_node=oid, target_text=o.text,
                    measured_dx=gold_key, expected_effect=ExpectedEffect.DECREASE, gold_source="direct_gold",
                ))
        path = self.gold_path_nodes(G_gold, gold_key)
        if path:
            for r in G_gold.rationales_for(path[0], Relation.SUPPORTS)[:1]:
                out.append(Edit(
                    edit_id=self._id(cid, "cf-weaken"), case_id=cid, op=EditOp.WEAKEN, mode=EditMode.HYPOTHETICAL,
                    edit_class=EditClass.COUNTERFACTUAL, target_node=r.id, target_text=r.claim, strength_factor=0.25,
                    measured_dx=gold_key, expected_effect=ExpectedEffect.DECREASE, gold_source="direct_gold",
                ))
                if r.strength <= 0.9:
                    out.append(Edit(
                        edit_id=self._id(cid, "cf-intensify"), case_id=cid, op=EditOp.INTENSIFY,
                        mode=EditMode.HYPOTHETICAL, edit_class=EditClass.COUNTERFACTUAL, target_node=r.id,
                        target_text=r.claim, strength_factor=4.0, measured_dx=gold_key,
                        expected_effect=ExpectedEffect.INCREASE, gold_source="direct_gold",
                    ))
        return out

    # ---------------------------------------------------- valid corrections
    def valid_corrections(self, G_model: ReasoningGraph, G_gold: ReasoningGraph, gold_key: str,
                          k: int = 2) -> List[Edit]:
        """Supply physician-tree evidence the model missed (assertion mode; it is in the note)."""
        cid = G_gold.case_id
        model_texts = [o.text for o in G_model.observations.values()]
        out: List[Edit] = []
        for oid in self.key_observations(G_gold, gold_key, k=6):
            o = G_gold.observations[oid]
            if any(_overlap(o.text, t) >= 0.6 for t in model_texts):
                continue
            links = []
            for r in G_gold.rationales_using(oid):
                d = G_gold.diagnoses.get(r.target)
                if d is not None:
                    par = G_gold.diagnoses.get(d.refines) if d.refines else None
                    links.append(InsertLink(category=d.category, leaf=d.leaf, relation=r.relation,
                                            strength=r.strength, claim=r.claim,
                                            refines_leaf=par.leaf if par else None))
            if not links:
                continue
            out.append(Edit(
                edit_id=self._id(cid, "vc-insert"), case_id=cid, op=EditOp.INSERT, mode=EditMode.ASSERTION,
                edit_class=EditClass.VALID_CORRECTION, new_text=o.text, new_polarity=o.polarity,
                new_section=o.section, links=links, measured_dx=gold_key,
                expected_effect=ExpectedEffect.INCREASE, gold_source="direct_gold",
                meta={"kind": "missed_evidence", "gold_obs": oid},
            ))
            if len(out) >= k:
                break
        return out

    # --------------------------------------------------------------- invalid
    def invalid(self, G: ReasoningGraph, note: str, gold_key: str, k: int = 2) -> List[Edit]:
        """Clinician asserts the opposite of a finding the record states explicitly."""
        cid = G.case_id
        out: List[Edit] = []
        for oid in self.key_observations(G, gold_key, k=6):
            o = G.observations[oid]
            if o.source != EvidenceSource.NOTE or o.polarity != Polarity.PRESENT:
                continue
            pol, _ = note_polarity(note, o.text)
            if pol != Polarity.PRESENT:
                continue  # only contradict findings the record states unambiguously
            out.append(Edit(
                edit_id=self._id(cid, "inv-negate"), case_id=cid, op=EditOp.NEGATE, mode=EditMode.ASSERTION,
                edit_class=EditClass.INVALID, target_node=oid, target_text=o.text, measured_dx=gold_key,
                expected_effect=ExpectedEffect.FLAG, implied_effect=ExpectedEffect.DECREASE,
                gold_source="construction",
            ))
            nums = re.findall(r"\d+(?:\.\d+)?", o.text)
            if nums:
                v = float(nums[0])
                alt = f"{v * 0.5:g}" if v > 2 else f"{v * 3:g}"
                out.append(Edit(
                    edit_id=self._id(cid, "inv-number"), case_id=cid, op=EditOp.REPLACE, mode=EditMode.ASSERTION,
                    edit_class=EditClass.INVALID, target_node=oid, target_text=o.text,
                    new_text=o.text.replace(nums[0], alt, 1), measured_dx=gold_key,
                    expected_effect=ExpectedEffect.FLAG, implied_effect=ExpectedEffect.DECREASE,
                    gold_source="construction",
                ))
            if len(out) >= k:
                break
        return out[:k]

    # ------------------------------------------------------------ irrelevant
    def irrelevant(self, G: ReasoningGraph, measured_dx: str, k: int = 2) -> List[Edit]:
        cid = G.case_id
        facts = self.rng.sample(self.bystander, min(k, len(self.bystander))) if self.bystander else []
        return [
            Edit(edit_id=self._id(cid, "irr"), case_id=cid, op=EditOp.INSERT, mode=EditMode.ASSERTION,
                 edit_class=EditClass.IRRELEVANT, new_text=f, measured_dx=measured_dx,
                 expected_effect=ExpectedEffect.NO_CHANGE, gold_source="construction")
            for f in facts
        ]

    # ------------------------------------------------------------- SCT items
    def from_sct(self, items: Iterable) -> List[Edit]:
        """SCT items (data.sct.SCTItem) -> NEW_FINDING inserts; the model must link them itself."""
        out = []
        for it in items:
            if it.gold_update is None:
                continue
            eff = (ExpectedEffect.INCREASE if it.gold_update > 0 else
                   ExpectedEffect.DECREASE if it.gold_update < 0 else ExpectedEffect.NO_CHANGE)
            out.append(Edit(
                edit_id=f"{it.item_id}", case_id=it.case_id, op=EditOp.INSERT, mode=EditMode.ASSERTION,
                edit_class=EditClass.NEW_FINDING, new_text=it.new_info, measured_dx=f"open::{it.hypothesis}",
                expected_effect=eff, expected_magnitude=int(it.gold_update), gold_source=it.source,
                meta={"timestep": it.timestep, "differential": list(it.differential)},
            ))
        return out

    # -------------------------------------------------------------- all-in-one
    def generate_case(self, G_gold: ReasoningGraph, note: str, gold_key: str,
                      G_model: Optional[ReasoningGraph] = None) -> List[Edit]:
        edits = self.counterfactuals(G_gold, gold_key)
        if G_model is not None:
            edits += self.valid_corrections(G_model, G_gold, gold_key)
        edits += self.invalid(G_gold, note, gold_key)
        edits += self.irrelevant(G_gold, gold_key)
        return edits


def measured_leaf_matches(a: str, b: str) -> bool:
    return normalize_label(split_dx_key(a)[1]) == normalize_label(split_dx_key(b)[1])
