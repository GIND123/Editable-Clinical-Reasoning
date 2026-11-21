import numpy as np

from clinblueprint.baselines.systems import ClinBlueprintSystem
from clinblueprint.benchmark.build import candidate_pool
from clinblueprint.edits.align import retarget
from clinblueprint.edits.generator import EditGenerator
from clinblueprint.eval.runner import run_edits, summarize
from clinblueprint.pipeline import ClinBlueprint, StaticDrafter
from clinblueprint.risk.ltt import hb_pvalue, learn_then_test
from clinblueprint.schema import ReasoningGraph
from clinblueprint.study.design import ARMS, assign, simulate_power
from clinblueprint.training.rewards import drafter_reward
from clinblueprint.schema.serialize import graph_to_json_dict

BP = "Pneumonia::Bacterial Pneumonia"


def test_oracle_graph_benchmark_run(toy_record, readout, kg):
    import random

    rec = toy_record.model_copy(deep=True)
    rec.candidates = candidate_pool(rec, kg, random.Random(0), n_distractors=1)
    G = ReasoningGraph.from_dict(rec.gold_graph)
    edits = EditGenerator(kg, ["The patient prefers morning appointments."]).generate_case(G, rec.note, BP)
    pipe = ClinBlueprint(StaticDrafter({}), readout, kg=kg)
    run = run_edits(ClinBlueprintSystem(pipe, regime="gold_graph"), {rec.case_id: rec}, edits, progress=False)
    s = summarize(run, n_boot=100)
    # the hard bottleneck on the oracle graph: counterfactuals followed, invalid flagged, irrelevant invariant
    assert s["mediation"]["mr_counterfactual"] == 1.0
    assert s["nonsycophancy"]["flag_sensitivity"] == 1.0
    assert s["mediation"]["invariance_rate"] == 1.0
    assert s["coverage"] == 1.0


def test_retarget_by_text(toy_graph):
    G2 = toy_graph.copy_deep()
    oid = next(o.id for o in G2.observations.values() if o.text.startswith("Sputum"))
    o = G2.observations.pop(oid)
    G2.observations["O77"] = o.model_copy(update={"id": "O77"})
    for r in G2.rationales.values():
        r.premises = ["O77" if p == oid else p for p in r.premises]
    from clinblueprint.edits import Edit, EditClass, EditOp, ExpectedEffect

    e = Edit(edit_id="x", case_id="c", op=EditOp.REMOVE, edit_class=EditClass.COUNTERFACTUAL, target_node=oid,
             target_text="Sputum culture: Streptococcus pneumoniae", measured_dx=BP,
             expected_effect=ExpectedEffect.DECREASE)
    assert retarget(e, G2).target_node == "O77"
    e2 = e.model_copy(update={"target_text": "bilateral pleural effusions on ultrasound", "target_node": "O999"})
    assert retarget(e2, G2) is None


def test_ltt():
    assert hb_pvalue(0.01, 1000, 0.1) < 1e-6
    assert hb_pvalue(0.2, 1000, 0.1) == 1.0
    lam = [0.1, 0.3, 0.5, 0.7, 0.9]
    loss = lambda t: np.r_[np.zeros(int(1000 * (1 - t * 0.2))), np.ones(int(1000 * t * 0.2))]  # noqa: E731
    valid = learn_then_test(lam, loss, alpha=0.15, delta=0.05)
    assert valid == [0.1, 0.3, 0.5]


def test_study_design_balanced():
    cases = [{"case_id": f"c{i}", "model_correct": i % 4 != 0, "specialty": ["im", "em"][i % 2]} for i in range(60)]
    rows = assign([f"md{i}" for i in range(6)], cases)
    for clin in {r["clinician"] for r in rows}:
        mine = [r for r in rows if r["clinician"] == clin]
        assert {r["arm"] for r in mine} == set(ARMS)
        assert len({r["case_id"] for r in mine}) == len(mine) == 60
    p = simulate_power(16, 120, 0.3, {"A_graph": 0.7, "B_cot": 0.45}, n_sim=100)
    assert 0.0 <= p <= 1.0


def test_reward(toy_graph, toy_record, kg):
    import json

    r = drafter_reward(json.dumps(graph_to_json_dict(toy_graph)), case_id="c", note=toy_record.note, gold_dx=BP,
                       kg=kg, gold_graph=toy_graph)
    assert r["parse"] == 1.0 and r["correct"] == 1.0 and r["grounding"] == 1.0 and r["graph_f1"] == 1.0
    assert drafter_reward("garbage", case_id="c", note="x", gold_dx=BP)["total"] == 0.0
