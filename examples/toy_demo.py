"""End-to-end demo on a SYNTHETIC case (no patient data, no GPU).

    python examples/toy_demo.py

Shows: the reasoning graph, the differential computed from it, and how the
system handles a hypothetical counterfactual, a valid correction, an invalid
assertion (flagged with evidence; then overridden), and an irrelevant fact.
"""

from __future__ import annotations

from pathlib import Path

from clinblueprint.checks import check_graph
from clinblueprint.data.direct import load_direct
from clinblueprint.edits import Edit, EditClass, EditMode, EditOp, ExpectedEffect, InsertLink
from clinblueprint.kg import DiagnosticKG
from clinblueprint.pipeline import ClinBlueprint, StaticDrafter
from clinblueprint.readout import HardReadout
from clinblueprint.schema import ReasoningGraph

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
BP, VP = "Pneumonia::Bacterial Pneumonia", "Pneumonia::Viral Pneumonia"


def show(title, out):
    b, a = out.before.as_dict(), out.after.as_dict()
    print(f"\n== {title}\n   {out.edit.describe()}")
    if out.flagged:
        print("   FLAGGED:", "; ".join(out.verdict.reasons))
        for s, e, t in out.verdict.cited_spans:
            print(f"   evidence [{s}:{e}]: {t!r}")
    for k in a:
        print(f"   {k.split('::')[1]:<22} {b.get(k, 0):.2f} -> {a[k]:.2f}")


def main():
    kg = DiagnosticKG.from_dir(FIX / "toy_kg")
    rec = load_direct(FIX / "direct", provenance="public")[0]
    G, rep = check_graph(ReasoningGraph.from_dict(rec.gold_graph), rec.note, kg, section_offsets=rec.section_offsets,
                         mnm={"Pneumonia": ["Pulmonary embolism", "Sepsis"]})
    pipe = ClinBlueprint(StaticDrafter({rec.case_id: G}), HardReadout(kg), kg=kg)
    print(rec.note, "\n")
    print(G.summary())
    print("\nDifferential (computed from the graph only):", {k.split('::')[1]: round(v, 2) for k, v in pipe.readout(G).as_dict().items()})

    sputum = next(o.id for o in G.observations.values() if o.text.startswith("Sputum"))
    show("Hypothetical counterfactual", pipe.apply_clinician_edit(G, Edit(
        edit_id="d1", case_id=rec.case_id, op=EditOp.NEGATE, mode=EditMode.HYPOTHETICAL,
        edit_class=EditClass.COUNTERFACTUAL, target_node=sputum, target_text="Sputum culture: Streptococcus pneumoniae",
        measured_dx=BP, expected_effect=ExpectedEffect.DECREASE), rec.note))

    show("New finding pointing elsewhere", pipe.apply_clinician_edit(G, Edit(
        edit_id="d2", case_id=rec.case_id, op=EditOp.INSERT, mode=EditMode.ASSERTION,
        edit_class=EditClass.NEW_FINDING, new_text="respiratory viral PCR positive for influenza A",
        links=[InsertLink(category="Pneumonia", leaf="Viral Pneumonia", claim="Positive viral PCR")],
        measured_dx=VP, expected_effect=ExpectedEffect.INCREASE), rec.note))

    invalid = Edit(edit_id="d3", case_id=rec.case_id, op=EditOp.NEGATE, mode=EditMode.ASSERTION,
                   edit_class=EditClass.INVALID, target_node=sputum,
                   target_text="Sputum culture: Streptococcus pneumoniae", measured_dx=BP,
                   expected_effect=ExpectedEffect.FLAG, description="Clinician states: the sputum culture was negative.")
    show("Invalid assertion (contradicts the record)", pipe.apply_clinician_edit(G, invalid, rec.note))
    show("... clinician overrides the flag (logged)", pipe.apply_clinician_edit(G, invalid, rec.note, override=True))

    show("Irrelevant fact", pipe.apply_clinician_edit(G, Edit(
        edit_id="d4", case_id=rec.case_id, op=EditOp.INSERT, mode=EditMode.ASSERTION,
        edit_class=EditClass.IRRELEVANT, new_text="The patient prefers morning appointments.",
        measured_dx=BP, expected_effect=ExpectedEffect.NO_CHANGE), rec.note))


if __name__ == "__main__":
    main()
