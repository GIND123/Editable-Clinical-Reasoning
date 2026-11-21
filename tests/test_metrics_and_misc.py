import numpy as np
import pytest

from clinblueprint.compliance import ComplianceError, assert_local, suppress_small_cells
from clinblueprint.data.records import CaseRecord
from clinblueprint.data.splits import grouped_stratified_kfold
from clinblueprint.eval.accuracy import direct_metrics, graph_to_direct_triples, split_reasoning_statements
from clinblueprint.eval.bootstrap import cluster_bootstrap, paired_cluster_bootstrap
from clinblueprint.eval.mediation import EditResult, outcome, summarize_mediation
from clinblueprint.eval.nonsycophancy import auroc, summarize_nonsycophancy
from clinblueprint.eval.sct import (
    aggregate_credit,
    delta_to_likert,
    dxupdate_kappa,
    fit_likert_thresholds,
    weighted_kappa,
)


def R(cls, eff, pb, pa, flagged=False, mode="hypothetical", implied=None, tv=None, case="c1", pc=None):
    return EditResult(edit_id=f"{cls}{pb}{pa}", case_id=case, edit_class=cls, mode=mode, op="x",
                      expected_effect=eff, measured_dx="d", p_before=pb, p_after=pa, top_before="d",
                      top_after="d", flagged=flagged, tv=abs(pa - pb) if tv is None else tv,
                      implied_effect=implied, p_contradict=pc)


def test_outcomes_and_asymmetry():
    rs = [R("valid_correction", "increase", .2, .6), R("valid_correction", "increase", .2, .7),
          R("counterfactual", "decrease", .8, .79), R("counterfactual", "decrease", .8, .3),
          R("irrelevant", "no_change", .5, .5), R("invalid", "flag", .8, .8, flagged=True, mode="assertion")]
    assert outcome(rs[0]) == "followed" and outcome(rs[2]) == "unchanged"
    s = summarize_mediation(rs)
    assert s["mr_valid_correction"] == 1.0 and s["mr_counterfactual"] == 0.5
    assert s["asymmetry"] == pytest.approx(0.5) and s["invariance_rate"] == 1.0


def test_nonsycophancy():
    rs = [R("invalid", "flag", .8, .8, flagged=True, mode="assertion", pc=.9),
          R("invalid", "flag", .8, .2, mode="assertion", implied="decrease", pc=.4),
          R("valid_correction", "increase", .2, .6, mode="assertion", pc=.1),
          R("new_finding", "increase", .2, .3, mode="assertion", flagged=True, pc=.7)]
    s = summarize_nonsycophancy(rs)
    assert s["flag_sensitivity"] == 0.5 and s["sycophancy_rate"] == 0.5 and s["false_flag_rate"] == 0.5
    assert auroc([.9, .4], [.1, .7]) == 0.75


def test_sct():
    assert delta_to_likert(0.05, 0.1, 1.0) == 0 and delta_to_likert(-0.5, 0.1, 1.0) == -1
    assert delta_to_likert(2.0, 0.1, 1.0) == 2
    assert weighted_kappa([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]) == pytest.approx(1.0)
    gold = [-2, -1, 0, 1, 2, 2, -2, 0]
    deltas = [-3.0, -0.6, 0.0, 0.5, 2.5, 3.1, -2.9, 0.05]
    t1, t2 = fit_likert_thresholds(deltas, gold)
    assert weighted_kappa([delta_to_likert(d, t1, t2) for d in deltas], gold) == pytest.approx(1.0)
    assert aggregate_credit(1, {1: 6, 2: 3}) == 1.0 and aggregate_credit(2, {1: 6, 2: 3}) == 0.5
    k = dxupdate_kappa([(1, 1, 1), (1, -1, -1), (2, 2, 1), (2, 0, 0)])
    assert k["direction"] == 1.0


def test_bootstrap():
    v = np.r_[np.ones(50), np.zeros(50)]
    cl = np.repeat(np.arange(20), 5)
    est, lo, hi = cluster_bootstrap(v, cl, n_boot=500)
    assert lo <= est <= hi and est == 0.5
    d, dlo, dhi = paired_cluster_bootstrap(np.ones(100), np.zeros(100), cl, n_boot=200)
    assert d == 1.0 and dlo == 1.0


def test_direct_metrics_perfect(toy_graph):
    t = graph_to_direct_triples(toy_graph)
    m = direct_metrics(t, t)
    assert m["faith_all"] == 1.0 and m["comp_coverage"] == 1.0


def test_reasoning_statements():
    assert split_reasoning_statements("1. Fever suggests infection.\n2. Culture positive.") == \
        ["Fever suggests infection.", "Culture positive."]


def test_compliance():
    assert_local("hf", None, "physionet")
    assert_local("openai", "https://api.openai.com/v1", "public")
    for backend, ep in [("openai", "https://api.openai.com/v1"), ("vllm_local", "https://example.org:8000"),
                        ("openrouter", None)]:
        with pytest.raises(ComplianceError):
            assert_local(backend, ep, "physionet")
    assert_local("vllm_local", "http://127.0.0.1:8000", "physionet")
    assert suppress_small_cells({"a": 3, "b": 12, "c": 0}) == {"a": "<10", "b": 12, "c": 0}


def test_patient_level_folds():
    recs = [CaseRecord(case_id=f"c{i}", dataset="t", note="x", patient_id=f"p{i // 3}", category=f"k{i % 2}")
            for i in range(60)]
    a = grouped_stratified_kfold(recs, k=5)
    by_pt = {}
    for r in recs:
        by_pt.setdefault(r.patient_id, set()).add(a[r.case_id])
    assert all(len(f) == 1 for f in by_pt.values())
    assert len(set(a.values())) == 5
