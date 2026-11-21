"""Apply an Edit to a ReasoningGraph (pure function; run ``check_graph`` afterwards).

The hard check's cascade then prunes rationales whose premises no longer
hold, which is how an edit propagates to the readout.
"""

from __future__ import annotations

from clinblueprint.edits.types import Edit, EditClass, EditMode, EditOp
from clinblueprint.schema.graph import (
    Diagnosis,
    EvidenceSource,
    Observation,
    Polarity,
    Rationale,
    ReasoningGraph,
    Relation,
)


class EditError(ValueError):
    pass


def _source(edit: Edit) -> EvidenceSource:
    if edit.edit_class == EditClass.NEW_FINDING and edit.op == EditOp.INSERT:
        return EvidenceSource.NEW_FINDING
    return EvidenceSource.HYPOTHETICAL if edit.mode == EditMode.HYPOTHETICAL else EvidenceSource.CLINICIAN


def _flip(p: Polarity) -> Polarity:
    return Polarity.ABSENT if p in (Polarity.PRESENT, Polarity.UNCERTAIN) else Polarity.PRESENT


def _scale_strength(s: float, factor: float) -> float:
    if factor <= 1.0:
        return max(0.0, s * factor)
    # intensify: move toward 1 without exceeding it
    return min(1.0, s + (1.0 - s) * (1.0 - 1.0 / factor))


def apply_edit(G: ReasoningGraph, edit: Edit) -> ReasoningGraph:
    G2 = G.copy_deep()
    op, tid = edit.op, edit.target_node
    if op != EditOp.INSERT and (tid is None or tid not in G2):
        raise EditError(f"edit {edit.edit_id}: target {tid!r} not in graph")

    if op == EditOp.NEGATE:
        if tid in G2.observations:
            o = G2.observations[tid]
            o.polarity = edit.new_polarity or _flip(o.polarity)
            o.source = _source(edit)
        elif tid in G2.rationales:
            r = G2.rationales[tid]
            r.relation = Relation.REFUTES if r.relation == Relation.SUPPORTS else Relation.SUPPORTS
        else:
            raise EditError("NEGATE applies to observations or rationales")

    elif op == EditOp.REMOVE:
        G2.remove(tid)

    elif op == EditOp.REPLACE:
        if tid in G2.observations:
            o = G2.observations[tid]
            if edit.new_text:
                o.text = edit.new_text
            if edit.new_polarity is not None:
                o.polarity = edit.new_polarity
            o.span = None
            o.source = _source(edit)
        elif tid in G2.diagnoses:
            d = G2.diagnoses[tid]
            if not edit.replace_leaf:
                raise EditError("REPLACE on a diagnosis needs replace_leaf")
            d.leaf = edit.replace_leaf
            if edit.replace_category:
                d.category = edit.replace_category
            d.p = None
        elif tid in G2.rationales:
            if edit.new_text:
                G2.rationales[tid].claim = edit.new_text
        else:
            raise EditError("REPLACE target type not supported")

    elif op in (EditOp.WEAKEN, EditOp.INTENSIFY):
        factor = edit.strength_factor or (0.5 if op == EditOp.WEAKEN else 2.0)
        if (op == EditOp.WEAKEN) != (factor <= 1.0):
            raise EditError("WEAKEN needs factor <= 1, INTENSIFY needs factor > 1")
        if tid in G2.rationales:
            targets = [G2.rationales[tid]]
        elif tid in G2.observations:
            targets = G2.rationales_using(tid)
        else:
            raise EditError("WEAKEN/INTENSIFY apply to rationales or observations")
        for r in targets:
            r.strength = _scale_strength(r.strength, factor)

    elif op == EditOp.INSERT:
        if not edit.new_text:
            raise EditError("INSERT needs new_text")
        oid = G2.new_id("O")
        pol = edit.new_polarity or Polarity.PRESENT
        G2.add(Observation(id=oid, text=edit.new_text, polarity=pol, source=_source(edit),
                           section=edit.new_section))
        G2.meta["last_inserted"] = oid
        for link in edit.links:
            d = G2.find_dx(link.leaf, link.category)
            if d is None:
                parent = G2.find_dx(link.refines_leaf, link.category) if link.refines_leaf else None
                d = Diagnosis(id=G2.new_id("D"), category=link.category, leaf=link.leaf,
                              refines=parent.id if parent else None)
                G2.add(d)
            G2.add(Rationale(
                id=G2.new_id("R"), premises=[oid], claim=link.claim, target=d.id,
                relation=link.relation, strength=link.strength, premise_polarity={oid: pol},
            ))
    else:  # pragma: no cover
        raise EditError(f"unknown op {op}")

    G2.meta.setdefault("edits", []).append(edit.edit_id)
    return G2
