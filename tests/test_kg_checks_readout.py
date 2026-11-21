from clinblueprint.checks import check_graph
from clinblueprint.edits import Edit, EditClass, EditOp, ExpectedEffect, InsertLink, apply_edit
from clinblueprint.readout import HardReadout, LinearReadout, ReadoutConfig
from clinblueprint.schema import Diagnosis, Observation, Rationale

from conftest import obs_id

BP = "Pneumonia::Bacterial Pneumonia"
VP = "Pneumonia::Viral Pneumonia"


def test_kg_queries(kg):
    assert set(kg.leaves("Pneumonia")) == {"Bacterial Pneumonia", "Viral Pneumonia"}
    assert kg.ancestors("Pneumonia", "Viral Pneumonia") == ["Pneumonia", "Suspected Pneumonia"]
    assert kg.is_valid_refinement("Pneumonia", "Bacterial Pneumonia", "Suspected Pneumonia")
    assert not kg.is_valid_refinement("Pneumonia", "Pneumonia", "Bacterial Pneumonia")
    assert kg.canonical("Pneumonia", "bacterial pneumonia") == "Bacterial Pneumonia"
    assert "culture" in kg.premise("Pneumonia", "Bacterial Pneumonia")


def test_checks_flag_ungrounded_and_invalid_kg(toy_graph, toy_record, kg):
    G = toy_graph.copy_deep()
    G.add(Observation(id="O99", text="bilateral ground-glass opacities"))
    G.add(Diagnosis(id="D99", category="Pneumonia", leaf="Fungal Pneumonia"))
    G2, rep = check_graph(G, toy_record.note, kg)
    codes = rep.codes()
    assert codes["UNGROUNDED_SPAN"] == 1 and codes["INVALID_KG_NODE"] == 1
    assert "O99" not in G2 and "D99" not in G2


def test_mnm_inserted(toy_graph, toy_record, kg):
    G2, rep = check_graph(toy_graph, toy_record.note, kg, mnm={"Pneumonia": ["Pulmonary embolism", "Sepsis"]})
    assert {m.dx for m in G2.must_not_miss.values()} == {"Pulmonary embolism", "Sepsis"}
    assert rep.codes()["MISSING_MUST_NOT_MISS"] == 2


def test_hard_readout_top_and_candidates(toy_graph, readout):
    ddx = readout(toy_graph)
    assert set(ddx.keys) == {BP, VP}  # kg_expand: every leaf of the touched category is scored
    assert ddx.argmax == BP


def _edit(op, target=None, **kw):
    return Edit(edit_id="e", case_id="c", op=op, edit_class=EditClass.COUNTERFACTUAL, target_node=target,
                measured_dx=BP, expected_effect=ExpectedEffect.DECREASE, **kw)


def test_monotone_removal_and_negation(toy_graph, toy_record, readout, kg):
    p0 = readout(toy_graph).p(BP)
    for op in (EditOp.REMOVE, EditOp.NEGATE):
        oid = obs_id(toy_graph, "Sputum culture: Streptococcus pneumoniae")
        G2, _ = check_graph(apply_edit(toy_graph, _edit(op, oid)), toy_record.note, kg)
        assert readout(G2).p(BP) < p0


def test_monotone_insert_support_for_alternative(toy_graph, toy_record, readout, kg):
    p0 = readout(toy_graph).p(VP)
    e = Edit(edit_id="e2", case_id="c", op=EditOp.INSERT, edit_class=EditClass.COUNTERFACTUAL,
             new_text="respiratory viral PCR positive for influenza A",
             links=[InsertLink(category="Pneumonia", leaf="Viral Pneumonia", claim="viral PCR positive")],
             measured_dx=VP, expected_effect=ExpectedEffect.INCREASE)
    G2, _ = check_graph(apply_edit(toy_graph, e), toy_record.note, kg)
    assert readout(G2).p(VP) > p0


def test_readout_never_reads_note(toy_graph, readout):
    # Same graph, different hypothetical notes -> identical differential (structural mediation)
    a = readout(toy_graph).as_dict()
    b = readout(toy_graph.copy_deep()).as_dict()
    assert a == b


def test_drafter_p_ignored_by_default(toy_graph, kg):
    G = toy_graph.copy_deep()
    for d in G.diagnoses.values():
        d.p = 0.01 if d.leaf == "Bacterial Pneumonia" else 0.99
    assert HardReadout(kg)(G).as_dict() == HardReadout(kg)(toy_graph).as_dict()
    bypass = HardReadout(kg, ReadoutConfig(use_drafter_p=True, drafter_p_weight=5.0))
    assert bypass(G).p(BP) < HardReadout(kg)(G).p(BP)


def test_linear_readout_respects_signs(toy_graph, kg):
    graphs, gold = [], []
    for i in range(6):
        G = toy_graph.copy_deep()
        if i % 2:
            G.add(Diagnosis(id="D50", category="Pneumonia", leaf="Viral Pneumonia", refines=None))
            G.add(Observation(id="O50", text="viral PCR positive", source="clinician"))
            G.add(Rationale(id="R50", premises=["O50"], target="D50", strength=1.0))
            G.add(Rationale(id="R51", premises=["O50"], target=G.find_dx("Bacterial Pneumonia").id,
                            relation="refutes", strength=1.0))
            for r in list(G.rationales.values()):
                if r.target == G.find_dx("Bacterial Pneumonia").id and r.relation.value == "supports":
                    G.remove(r.id)
            gold.append(VP)
        else:
            gold.append(BP)
        graphs.append(G)
    lr = LinearReadout(kg)
    rep = lr.fit(graphs, gold, epochs=200)
    w = rep.weights
    assert w["support_leaf"] >= 0 and w["refute_leaf"] <= 0 and w["support_anc"] >= 0 and w["refute_anc"] <= 0
    assert w["drafted"] == 0 and w["log_drafter_p"] == 0
    assert rep.accuracy == 1.0
