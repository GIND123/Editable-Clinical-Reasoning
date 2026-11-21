"""Graph <-> text / JSON.

Two formats:

1. **Drafter JSON** – what the drafter LM emits (constrained to
   ``DRAFTER_JSON_SCHEMA``). Order is *evidence first*: observations, then
   rationales, then diagnoses, so an autoregressive drafter commits to evidence
   before it names a conclusion.
2. **Slot text** – a line-per-node linearisation used by the soft readout, the
   verbaliser and the mechanistic suite. Each line is a *slot* whose character
   span is recorded so token positions can be pooled / patched.

Rationale lines carry the *name* of their target diagnosis so a readout that
excludes diagnosis slots (anti-bypass default) still knows which candidate the
evidence bears on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clinblueprint.schema.graph import (
    OPEN_CATEGORY,
    Diagnosis,
    EvidenceSource,
    MNMStatus,
    MustNotMiss,
    NodeType,
    Observation,
    Polarity,
    Rationale,
    ReasoningGraph,
    Relation,
    natural_id_key,
)


class GraphParseError(ValueError):
    pass


@dataclass
class SlotSpan:
    node_id: str
    node_type: str
    start: int  # char offset in SerializedGraph.text (inclusive)
    end: int    # char offset (exclusive)


@dataclass
class SerializedGraph:
    text: str
    slots: List[SlotSpan] = field(default_factory=list)

    def slot(self, node_id: str) -> Optional[SlotSpan]:
        for s in self.slots:
            if s.node_id == node_id:
                return s
        return None


def _sorted(nodes):
    return sorted(nodes, key=lambda n: natural_id_key(n.id))


def serialize_graph(
    G: ReasoningGraph,
    *,
    include_dx_slots: bool = True,
    include_mnm: bool = True,
    include_drafter_p: bool = False,
    header: bool = True,
) -> SerializedGraph:
    parts: List[str] = []
    slots: List[SlotSpan] = []
    pos = 0

    def emit(line: str, node_id: Optional[str] = None, node_type: Optional[str] = None) -> None:
        nonlocal pos
        if node_id is not None:
            slots.append(SlotSpan(node_id, node_type or "", pos, pos + len(line)))
        parts.append(line)
        pos += len(line) + 1  # newline

    if header:
        emit("<graph>")
    for o in _sorted(G.observations.values()):
        sec = o.section or "-"
        emit(
            f"[{o.id} | observation | {o.polarity.value} | {sec} | {o.source.value}] {o.text}",
            o.id, NodeType.OBSERVATION.value,
        )
    for r in _sorted(G.rationales.values()):
        tgt = G.diagnoses.get(r.target)
        tname = f"{tgt.category} > {tgt.leaf}" if tgt else r.target
        prem = ",".join(
            p if r.expected_polarity(p) == Polarity.PRESENT else f"{p}:{r.expected_polarity(p).value}"
            for p in r.premises
        )
        emit(
            f"[{r.id} | rationale | {r.relation.value} -> {tname} | strength={r.strength:.2f} | "
            f"premises={prem}] {r.claim}",
            r.id, NodeType.RATIONALE.value,
        )
    if include_dx_slots:
        for d in _sorted(G.diagnoses.values()):
            ref = ""
            if d.refines and d.refines in G.diagnoses:
                ref = f" | refines={G.diagnoses[d.refines].leaf}"
            pstr = f" | p={d.p:.2f}" if (include_drafter_p and d.p is not None) else ""
            emit(f"[{d.id} | diagnosis | {d.category} > {d.leaf}{ref}{pstr}]", d.id, NodeType.DIAGNOSIS.value)
    if include_mnm:
        for m in _sorted(G.must_not_miss.values()):
            ev = ",".join(m.evidence) or "-"
            emit(f"[{m.id} | must_not_miss | {m.dx} | {m.status.value} | evidence={ev}]",
                 m.id, NodeType.MUST_NOT_MISS.value)
    if header:
        emit("</graph>")
    return SerializedGraph(text="\n".join(parts), slots=slots)


# --------------------------------------------------------------------------- JSON
DRAFTER_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["observations", "rationales", "diagnoses", "must_not_miss"],
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "text", "polarity"],
                "properties": {
                    "id": {"type": "string", "pattern": "^O[0-9]+$"},
                    "text": {"type": "string", "description": "verbatim quote from the note"},
                    "polarity": {"enum": ["present", "absent", "uncertain"]},
                    "section": {"type": "string"},
                },
            },
        },
        "rationales": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "premises", "claim", "target", "relation", "strength"],
                "properties": {
                    "id": {"type": "string", "pattern": "^R[0-9]+$"},
                    "premises": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "claim": {"type": "string"},
                    "target": {"type": "string", "pattern": "^D[0-9]+$"},
                    "relation": {"enum": ["supports", "refutes"]},
                    "strength": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "category", "leaf"],
                "properties": {
                    "id": {"type": "string", "pattern": "^D[0-9]+$"},
                    "category": {"type": "string"},
                    "leaf": {"type": "string"},
                    "refines": {"type": ["string", "null"]},
                    "p": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                },
            },
        },
        "must_not_miss": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "dx", "status"],
                "properties": {
                    "id": {"type": "string", "pattern": "^M[0-9]+$"},
                    "dx": {"type": "string"},
                    "status": {"enum": ["excluded", "open", "not_addressed"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def graph_to_json_dict(G: ReasoningGraph, include_p: bool = False) -> Dict[str, Any]:
    obs = []
    for o in _sorted(G.observations.values()):
        d: Dict[str, Any] = {"id": o.id, "text": o.text, "polarity": o.polarity.value}
        if o.section:
            d["section"] = o.section
        obs.append(d)
    rats = [
        {
            "id": r.id, "premises": list(r.premises), "claim": r.claim, "target": r.target,
            "relation": r.relation.value, "strength": round(float(r.strength), 3),
        }
        for r in _sorted(G.rationales.values())
    ]
    dxs = []
    for d in _sorted(G.diagnoses.values()):
        item: Dict[str, Any] = {"id": d.id, "category": d.category, "leaf": d.leaf, "refines": d.refines}
        if include_p:
            item["p"] = d.p
        dxs.append(item)
    mnm = [
        {"id": m.id, "dx": m.dx, "status": m.status.value, "evidence": list(m.evidence)}
        for m in _sorted(G.must_not_miss.values())
    ]
    return {"observations": obs, "rationales": rats, "diagnoses": dxs, "must_not_miss": mnm}


def _enum(enum_cls, value, default):
    try:
        return enum_cls(str(value).strip().lower())
    except Exception:
        return default


def graph_from_json_dict(d: Dict[str, Any], case_id: str) -> ReasoningGraph:
    """Tolerant constructor for drafter / teacher output.

    Missing ids are assigned, unknown enum values fall back to defaults,
    duplicate ids are renamed. Referential problems are *not* fixed here; that
    is the job of ``checks.check_graph`` so violations get reported.
    """
    G = ReasoningGraph(case_id=case_id)

    def uid(prefix: str, proposed: Optional[str]) -> str:
        if proposed and isinstance(proposed, str) and proposed not in G:
            return proposed
        return G.new_id(prefix)

    for o in d.get("observations") or []:
        if not isinstance(o, dict) or not str(o.get("text", "")).strip():
            continue
        G.add(Observation(
            id=uid("O", o.get("id")), text=str(o["text"]).strip(),
            polarity=_enum(Polarity, o.get("polarity", "present"), Polarity.PRESENT),
            section=o.get("section"),
            source=_enum(EvidenceSource, o.get("source", "note"), EvidenceSource.NOTE),
        ))
    for dx in d.get("diagnoses") or []:
        if not isinstance(dx, dict) or not str(dx.get("leaf", "")).strip():
            continue
        p = dx.get("p")
        try:
            p = None if p is None else min(1.0, max(0.0, float(p)))
        except (TypeError, ValueError):
            p = None
        G.add(Diagnosis(
            id=uid("D", dx.get("id")), category=str(dx.get("category") or OPEN_CATEGORY).strip(),
            leaf=str(dx["leaf"]).strip(), refines=dx.get("refines") or None, p=p,
        ))
    for r in d.get("rationales") or []:
        if not isinstance(r, dict) or not r.get("target"):
            continue
        prem = r.get("premises") or []
        if isinstance(prem, str):
            prem = [p.strip() for p in prem.split(",") if p.strip()]
        try:
            strength = min(1.0, max(0.0, float(r.get("strength", 1.0))))
        except (TypeError, ValueError):
            strength = 1.0
        G.add(Rationale(
            id=uid("R", r.get("id")), premises=[str(p) for p in prem], claim=str(r.get("claim", "")),
            target=str(r["target"]), relation=_enum(Relation, r.get("relation", "supports"), Relation.SUPPORTS),
            strength=strength,
        ))
    for m in d.get("must_not_miss") or []:
        if not isinstance(m, dict) or not m.get("dx"):
            continue
        G.add(MustNotMiss(
            id=uid("M", m.get("id")), dx=str(m["dx"]),
            status=_enum(MNMStatus, m.get("status", "not_addressed"), MNMStatus.NOT_ADDRESSED),
            evidence=[str(e) for e in (m.get("evidence") or [])],
        ))
    return G


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?")


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Return the first balanced top-level JSON object in ``text`` (or None)."""
    if not text:
        return None
    t = _THINK_RE.sub("", text)
    t = _FENCE_RE.sub("", t)
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = t[start:i + 1]
                    for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                    break
        start = t.find("{", start + 1)
    return None


def parse_graph_json(text: str, case_id: str) -> ReasoningGraph:
    obj = extract_json_object(text)
    if obj is None:
        raise GraphParseError("no JSON object found in model output")
    G = graph_from_json_dict(obj, case_id)
    if not G.diagnoses:
        raise GraphParseError("graph has no diagnoses")
    return G
