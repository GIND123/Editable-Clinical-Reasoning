"""Shared fixtures. All data here is synthetic (written for this repo; no patient data)."""

from pathlib import Path

import pytest

from clinblueprint.checks import check_graph
from clinblueprint.data.direct import load_direct
from clinblueprint.kg import DiagnosticKG
from clinblueprint.readout import HardReadout
from clinblueprint.schema import ReasoningGraph

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def kg():
    return DiagnosticKG.from_dir(FIX / "toy_kg")


@pytest.fixture(scope="session")
def toy_record():
    recs = load_direct(FIX / "direct", provenance="public")
    assert len(recs) == 1
    return recs[0]


@pytest.fixture()
def toy_graph(toy_record, kg):
    G = ReasoningGraph.from_dict(toy_record.gold_graph)
    G, rep = check_graph(G, toy_record.note, kg, section_offsets=toy_record.section_offsets)
    assert rep.grounding_rate == 1.0
    return G


@pytest.fixture()
def readout(kg):
    return HardReadout(kg)


def obs_id(G, text):
    for o in G.observations.values():
        if o.text == text:
            return o.id
    raise KeyError(text)
