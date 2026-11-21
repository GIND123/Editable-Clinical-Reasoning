from clinblueprint.edits import Edit, EditClass, EditMode, EditOp, ExpectedEffect
from clinblueprint.edits.generator import EditGenerator
from clinblueprint.edits.text_adapter import apply_edit_to_text
from clinblueprint.edits.validator import EditValidator, note_polarity
from clinblueprint.pipeline import ClinBlueprint, StaticDrafter
from clinblueprint.schema import Polarity

from conftest import obs_id

BP = "Pneumonia::Bacterial Pneumonia"


def _pipe(toy_graph, readout, kg):
    return ClinBlueprint(StaticDrafter({toy_graph.case_id: toy_graph}), readout, kg=kg)


def test_note_polarity(toy_record):
    assert note_polarity(toy_record.note, "productive cough")[0] == Polarity.PRESENT
    assert note_polarity(toy_record.note, "leg swelling")[0] == Polarity.ABSENT
    assert note_polarity(toy_record.note, "recent travel")[0] == Polarity.ABSENT
    assert note_polarity(toy_record.note, "hemoptysis")[0] is None


def test_hypothetical_vs_assertion(toy_graph, toy_record, readout, kg):
    pipe = _pipe(toy_graph, readout, kg)
    oid = obs_id(toy_graph, "Sputum culture: Streptococcus pneumoniae")
    common = dict(case_id="c", op=EditOp.NEGATE, target_node=oid, measured_dx=BP)
    hypo = Edit(edit_id="h", mode=EditMode.HYPOTHETICAL, edit_class=EditClass.COUNTERFACTUAL,
                expected_effect=ExpectedEffect.DECREASE, **common)
    asrt = Edit(edit_id="a", mode=EditMode.ASSERTION, edit_class=EditClass.INVALID,
                expected_effect=ExpectedEffect.FLAG, **common)
    out_h = pipe.apply_clinician_edit(toy_graph, hypo, toy_record.note)
    out_a = pipe.apply_clinician_edit(toy_graph, asrt, toy_record.note)
    assert not out_h.flagged and out_h.delta < 0          # what-if is answered
    assert out_a.flagged and out_a.delta == 0             # contradicting the chart is flagged
    assert out_a.verdict.cited_spans                      # ... with the evidence
    out_o = pipe.apply_clinician_edit(toy_graph, asrt, toy_record.note, override=True)
    assert out_o.overridden and out_o.delta < 0           # clinician can override; logged


def test_numeric_contradiction_and_new_information(toy_graph, toy_record, readout, kg):
    v = EditValidator()
    oid = obs_id(toy_graph, "fevers to 39.2 C")
    e = Edit(edit_id="n", case_id="c", op=EditOp.REPLACE, mode=EditMode.ASSERTION, edit_class=EditClass.INVALID,
             target_node=oid, new_text="fevers to 37.0 C", measured_dx=BP, expected_effect=ExpectedEffect.FLAG)
    assert v.validate(e, toy_record.note, toy_graph).flagged
    ins = Edit(edit_id="i", case_id="c", op=EditOp.INSERT, mode=EditMode.ASSERTION,
               edit_class=EditClass.NEW_FINDING, new_text="D-dimer elevated", measured_dx=BP,
               expected_effect=ExpectedEffect.NO_CHANGE)
    assert v.validate(ins, toy_record.note, toy_graph).label == "new_information"
    leg = Edit(edit_id="l", case_id="c", op=EditOp.INSERT, mode=EditMode.ASSERTION, edit_class=EditClass.INVALID,
               new_text="leg swelling", measured_dx=BP, expected_effect=ExpectedEffect.FLAG)
    assert v.validate(leg, toy_record.note, toy_graph).flagged  # note: "He denies leg swelling"


def test_irrelevant_insert_is_invariant(toy_graph, toy_record, readout, kg):
    pipe = _pipe(toy_graph, readout, kg)
    e = Edit(edit_id="irr", case_id="c", op=EditOp.INSERT, mode=EditMode.ASSERTION, edit_class=EditClass.IRRELEVANT,
             new_text="The patient prefers morning appointments.", measured_dx=BP,
             expected_effect=ExpectedEffect.NO_CHANGE)
    out = pipe.apply_clinician_edit(toy_graph, e, toy_record.note)
    assert not out.flagged and out.before.tv(out.after) == 0.0


def test_generator_produces_all_classes(toy_graph, toy_record, kg):
    gen = EditGenerator(kg, bystander_facts=["The patient prefers morning appointments."])
    edits = gen.generate_case(toy_graph, toy_record.note, BP, G_model=toy_graph.copy_deep())
    classes = {e.edit_class.value for e in edits}
    assert {"counterfactual", "invalid", "irrelevant"} <= classes
    # most specific evidence first: the sputum culture is a key observation
    keys = gen.key_observations(toy_graph, BP, k=2)
    assert any("Sputum" in toy_graph.observations[k].text for k in keys)


def test_text_adapter(toy_record, toy_graph):
    oid = obs_id(toy_graph, "productive cough")
    e = Edit(edit_id="t", case_id="c", op=EditOp.NEGATE, target_node=oid, edit_class=EditClass.COUNTERFACTUAL,
             measured_dx=BP, expected_effect=ExpectedEffect.DECREASE)
    trace = "The patient has a productive cough. Imaging shows consolidation."
    out = apply_edit_to_text(trace, e, toy_graph)
    assert "there is no productive cough" in out and "Imaging shows consolidation." in out
