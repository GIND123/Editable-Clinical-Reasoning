import json

import pytest

from clinblueprint.schema import (
    GraphParseError,
    ReasoningGraph,
    extract_json_object,
    graph_from_json_dict,
    graph_to_json_dict,
    parse_graph_json,
    serialize_graph,
)


def test_direct_parser_structure(toy_record):
    assert toy_record.final_dx == "Pneumonia::Bacterial Pneumonia"
    assert toy_record.category == "Pneumonia"
    G = ReasoningGraph.from_dict(toy_record.gold_graph)
    assert len(G.diagnoses) == 3 and len(G.observations) == 6 and len(G.rationales) == 6
    bp = G.find_dx("Bacterial Pneumonia")
    chain = [G.diagnoses[a].leaf for a in G.dx_ancestors(bp.id)]
    assert chain == ["Pneumonia", "Suspected Pneumonia"]
    assert G.terminal_diagnoses() == [bp.id]
    assert not G.integrity_errors()


def test_grounding_sets_verbatim_spans(toy_graph, toy_record):
    for o in toy_graph.observations.values():
        assert o.span is not None
        assert toy_record.note[o.span.start:o.span.end].lower() == o.text.lower()


def test_json_roundtrip(toy_graph):
    d = graph_to_json_dict(toy_graph)
    G2 = graph_from_json_dict(json.loads(json.dumps(d)), toy_graph.case_id)
    assert {o.text for o in G2.observations.values()} == {o.text for o in toy_graph.observations.values()}
    assert len(G2.rationales) == len(toy_graph.rationales)


def test_serialize_slot_offsets(toy_graph):
    sg = serialize_graph(toy_graph)
    lines = sg.text.split("\n")
    for s in sg.slots:
        assert sg.text[s.start:s.end] in lines
        assert sg.text[s.start:s.end].startswith(f"[{s.node_id} |")
    # rationale slots name their target diagnosis (anti-bypass readout needs it)
    rat_lines = [sg.text[s.start:s.end] for s in sg.slots if s.node_type == "rationale"]
    assert any("Bacterial Pneumonia" in l for l in rat_lines)


def test_extract_json_with_noise():
    txt = "<think>let me think {not json}</think>\n```json\n{\"observations\": [], \"rationales\": [],\n" \
          " \"diagnoses\": [{\"id\": \"D1\", \"category\": \"open\", \"leaf\": \"x\"},], \"must_not_miss\": []}\n```"
    obj = extract_json_object(txt)
    assert obj["diagnoses"][0]["leaf"] == "x"
    G = parse_graph_json(txt, "c")
    assert "D1" in G.diagnoses
    with pytest.raises(GraphParseError):
        parse_graph_json("no json here", "c")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        ReasoningGraph.from_dict({"case_id": "c", "observations": {"X1": {"id": "X1", "text": "a"}},
                                  "diagnoses": {"X1": {"id": "X1", "leaf": "b"}}})
