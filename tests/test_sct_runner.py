from clinblueprint.data.sct import SCTItem
from clinblueprint.edits.types import InsertLink
from clinblueprint.eval.sct_runner import run_sequential, score
from clinblueprint.pipeline import ClinBlueprint, StaticDrafter
from clinblueprint.readout import HardReadout
from clinblueprint.schema import ReasoningGraph
from clinblueprint.schema.graph import Relation


def test_sequential_sct_accumulates_and_scores():
    dxs = ["pulmonary embolism", "pneumonia"]
    links = {
        "D-dimer markedly elevated": [InsertLink(category="open", leaf="pulmonary embolism", strength=1.0)],
        "Lobar consolidation on chest x-ray": [
            InsertLink(category="open", leaf="pneumonia", strength=1.0),
            InsertLink(category="open", leaf="pulmonary embolism", relation=Relation.REFUTES, strength=0.5)],
    }
    items = []
    for c in range(6):
        cid = f"er:{c}"
        items += [
            SCTItem(item_id=f"{cid}:1", case_id=cid, vignette="Dyspnoea and pleuritic chest pain.",
                    hypothesis="pulmonary embolism", new_info="D-dimer markedly elevated", gold_update=2, timestep=1,
                    differential=dxs, gold_ranking=["pulmonary embolism", "pneumonia"]),
            SCTItem(item_id=f"{cid}:2", case_id=cid, vignette="Dyspnoea and pleuritic chest pain.",
                    hypothesis="pneumonia", new_info="Lobar consolidation on chest x-ray", gold_update=1, timestep=2,
                    differential=dxs, gold_ranking=["pneumonia", "pulmonary embolism"]),
        ]
    drafter = StaticDrafter({f"er:{c}": ReasoningGraph(case_id=f"er:{c}") for c in range(6)}, links=links)
    pipe = ClinBlueprint(drafter, HardReadout())
    rows = run_sequential(pipe, items)
    assert len(rows) == 12
    assert all(r["delta_logit"] > 0 for r in rows)            # both findings raise their hypothesis
    assert rows[1]["ranking"][0] == "pneumonia"                # evidence accumulated across timesteps
    s = score(rows, dev_frac=0.5, n_boot=100)
    assert s["dxupdate"]["direction"] == 1.0
