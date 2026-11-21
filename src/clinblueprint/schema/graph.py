"""Typed clinical reasoning graph (aligned with the DiReCT annotation schema).

Node types
----------
Observation   a finding, ideally a verbatim span of the note (DiReCT "Input" node)
Rationale     premises (observations / rationales) -> claim about one diagnosis
              (DiReCT "Cause" node); carries a relation (supports/refutes) and
              a strength in [0, 1]
Diagnosis     a node of the diagnostic knowledge graph (DiReCT "Intermedia");
              ``refines`` points to the more general diagnosis it specialises
MustNotMiss   a dangerous alternative that must be explicitly addressed

Edges are *derived* from node fields (single source of truth):
    premise --premise--> rationale --supports|refutes--> diagnosis --refines--> diagnosis

Design rule that makes mediation possible: the decision readout reads only
this object. The drafter's self-reported ``Diagnosis.p`` is stored for analysis
but is NOT read by the hard readout unless explicitly enabled as an ablation.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Polarity(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class Relation(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"


class EvidenceSource(str, Enum):
    NOTE = "note"                  # extracted from the patient's record; must cite a span
    CLINICIAN = "clinician"        # asserted by a clinician as a fact about this patient
    NEW_FINDING = "new_finding"    # new information arriving later (e.g. SCT "new information")
    HYPOTHETICAL = "hypothetical"  # what-if intervention; never a claim about the patient


class MNMStatus(str, Enum):
    EXCLUDED = "excluded"            # considered and argued against
    OPEN = "open"                    # considered, not excluded
    NOT_ADDRESSED = "not_addressed"  # not yet considered (safety flag)


class NodeType(str, Enum):
    OBSERVATION = "observation"
    RATIONALE = "rationale"
    DIAGNOSIS = "diagnosis"
    MUST_NOT_MISS = "must_not_miss"


OPEN_CATEGORY = "open"  # used for open-ended tasks without a diagnostic KG


def dx_key(category: Optional[str], leaf: str) -> str:
    """Canonical key of a diagnosis label: ``"<category>::<leaf>"``."""
    return f"{category or OPEN_CATEGORY}::{leaf}"


def split_dx_key(key: str) -> Tuple[str, str]:
    if "::" in key:
        cat, leaf = key.split("::", 1)
        return cat, leaf
    return OPEN_CATEGORY, key


def normalize_label(s: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace (for label matching)."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/+]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_ID_RE = re.compile(r"^([A-Za-z_]+)(\d+)$")


def natural_id_key(node_id: str) -> Tuple[str, int, str]:
    m = _ID_RE.match(node_id)
    if m:
        return (m.group(1), int(m.group(2)), "")
    return (node_id, 10**9, node_id)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Span(_Base):
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    section: Optional[str] = None
    score: float = 100.0  # grounding match score; 100 = verbatim

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: int, info):  # noqa: ANN001
        start = info.data.get("start", 0)
        if v < start:
            raise ValueError("span end < start")
        return v


class Observation(_Base):
    id: str
    text: str
    span: Optional[Span] = None
    concept: Optional[str] = None  # optional normalised concept (e.g. UMLS CUI)
    polarity: Polarity = Polarity.PRESENT
    source: EvidenceSource = EvidenceSource.NOTE
    section: Optional[str] = None


class Rationale(_Base):
    id: str
    premises: List[str]
    claim: str = ""
    target: str  # Diagnosis id
    relation: Relation = Relation.SUPPORTS
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    # polarity each observation premise must have for this rationale to hold
    # (default PRESENT). A rationale about "absence of fever" sets ABSENT.
    premise_polarity: Dict[str, Polarity] = Field(default_factory=dict)

    def expected_polarity(self, premise_id: str) -> Polarity:
        return self.premise_polarity.get(premise_id, Polarity.PRESENT)


class Diagnosis(_Base):
    id: str
    category: str = OPEN_CATEGORY
    leaf: str
    refines: Optional[str] = None  # id of the more general Diagnosis this one specialises
    p: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # drafter's self-report (not read by default)

    @property
    def key(self) -> str:
        return dx_key(self.category, self.leaf)


class MustNotMiss(_Base):
    id: str
    dx: str
    status: MNMStatus = MNMStatus.NOT_ADDRESSED
    evidence: List[str] = Field(default_factory=list)


Node = Union[Observation, Rationale, Diagnosis, MustNotMiss]


class ReasoningGraph(_Base):
    case_id: str
    observations: Dict[str, Observation] = Field(default_factory=dict)
    rationales: Dict[str, Rationale] = Field(default_factory=dict)
    diagnoses: Dict[str, Diagnosis] = Field(default_factory=dict)
    must_not_miss: Dict[str, MustNotMiss] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _keys_match_ids(self) -> "ReasoningGraph":
        seen: set = set()
        for store in self._stores():
            for k, node in store.items():
                if k != node.id:
                    raise ValueError(f"key {k!r} != node id {node.id!r}")
                if k in seen:
                    raise ValueError(f"duplicate node id {k!r}")
                seen.add(k)
        return self

    # ------------------------------------------------------------------ access
    def _stores(self) -> Tuple[Dict[str, Any], ...]:
        return (self.observations, self.rationales, self.diagnoses, self.must_not_miss)

    def __contains__(self, node_id: str) -> bool:
        return any(node_id in s for s in self._stores())

    def node(self, node_id: str) -> Node:
        for s in self._stores():
            if node_id in s:
                return s[node_id]
        raise KeyError(node_id)

    def node_type(self, node_id: str) -> NodeType:
        if node_id in self.observations:
            return NodeType.OBSERVATION
        if node_id in self.rationales:
            return NodeType.RATIONALE
        if node_id in self.diagnoses:
            return NodeType.DIAGNOSIS
        if node_id in self.must_not_miss:
            return NodeType.MUST_NOT_MISS
        raise KeyError(node_id)

    def ids(self) -> List[str]:
        out: List[str] = []
        for s in self._stores():
            out.extend(sorted(s, key=natural_id_key))
        return out

    def new_id(self, prefix: str) -> str:
        i = 1
        while f"{prefix}{i}" in self:
            i += 1
        return f"{prefix}{i}"

    def add(self, node: Node) -> None:
        if node.id in self:
            raise ValueError(f"node id {node.id!r} already exists")
        if isinstance(node, Observation):
            self.observations[node.id] = node
        elif isinstance(node, Rationale):
            self.rationales[node.id] = node
        elif isinstance(node, Diagnosis):
            self.diagnoses[node.id] = node
        elif isinstance(node, MustNotMiss):
            self.must_not_miss[node.id] = node
        else:  # pragma: no cover
            raise TypeError(type(node))

    def remove(self, node_id: str) -> None:
        for s in self._stores():
            if node_id in s:
                del s[node_id]
                return
        raise KeyError(node_id)

    # --------------------------------------------------------------- structure
    def rationales_for(self, dx_id: str, relation: Optional[Relation] = None) -> List[Rationale]:
        return [
            r for r in self.rationales.values()
            if r.target == dx_id and (relation is None or r.relation == relation)
        ]

    def rationales_using(self, node_id: str) -> List[Rationale]:
        return [r for r in self.rationales.values() if node_id in r.premises]

    def dx_ancestors(self, dx_id: str) -> List[str]:
        out: List[str] = []
        cur = self.diagnoses.get(dx_id)
        seen = {dx_id}
        while cur is not None and cur.refines and cur.refines not in seen:
            out.append(cur.refines)
            seen.add(cur.refines)
            cur = self.diagnoses.get(cur.refines)
        return out

    def dx_children(self, dx_id: str) -> List[str]:
        return [d.id for d in self.diagnoses.values() if d.refines == dx_id]

    def terminal_diagnoses(self) -> List[str]:
        """Diagnoses that no other diagnosis refines (the most specific claims)."""
        parents = {d.refines for d in self.diagnoses.values() if d.refines}
        return sorted((d for d in self.diagnoses if d not in parents), key=natural_id_key)

    def find_dx(self, leaf: str, category: Optional[str] = None) -> Optional[Diagnosis]:
        target = normalize_label(leaf)
        for d in self.diagnoses.values():
            if normalize_label(d.leaf) == target and (category is None or d.category == category):
                return d
        return None

    def edges(self) -> List[Tuple[str, str, str, float]]:
        """(src, dst, type, weight) for every derived edge."""
        out: List[Tuple[str, str, str, float]] = []
        for r in self.rationales.values():
            for p in r.premises:
                out.append((p, r.id, "premise", 1.0))
            out.append((r.id, r.target, r.relation.value, r.strength))
        for d in self.diagnoses.values():
            if d.refines:
                out.append((d.id, d.refines, "refines", 1.0))
        for m in self.must_not_miss.values():
            for e in m.evidence:
                out.append((e, m.id, "mnm_evidence", 1.0))
        return out

    def integrity_errors(self) -> List[str]:
        errs: List[str] = []
        for r in self.rationales.values():
            for p in r.premises:
                if p not in self.observations and p not in self.rationales:
                    errs.append(f"{r.id}: premise {p} missing")
                if p == r.id:
                    errs.append(f"{r.id}: self-premise")
            if r.target not in self.diagnoses:
                errs.append(f"{r.id}: target {r.target} missing")
        for d in self.diagnoses.values():
            if d.refines and d.refines not in self.diagnoses:
                errs.append(f"{d.id}: refines {d.refines} missing")
            if d.refines == d.id:
                errs.append(f"{d.id}: refines itself")
        for m in self.must_not_miss.values():
            for e in m.evidence:
                if e not in self:
                    errs.append(f"{m.id}: evidence {e} missing")
        return errs

    # ----------------------------------------------------------- serialisation
    def copy_deep(self) -> "ReasoningGraph":
        return self.model_copy(deep=True)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReasoningGraph":
        return cls.model_validate(d)

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "ReasoningGraph":
        return cls.from_dict(json.loads(s))

    def fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("meta", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def summary(self) -> str:
        """Human-readable rendering (used in the toy demo and clinician audit sheets)."""
        lines = [f"Case {self.case_id}"]
        for o in sorted(self.observations.values(), key=lambda n: natural_id_key(n.id)):
            neg = "" if o.polarity == Polarity.PRESENT else f" [{o.polarity.value}]"
            src = "" if o.source == EvidenceSource.NOTE else f" ({o.source.value})"
            lines.append(f"  {o.id}: {o.text}{neg}{src}")
        for r in sorted(self.rationales.values(), key=lambda n: natural_id_key(n.id)):
            tgt = self.diagnoses.get(r.target)
            tname = tgt.leaf if tgt else r.target
            lines.append(
                f"  {r.id}: {'+'.join(r.premises)} --{r.relation.value}({r.strength:.2f})--> {tname}: {r.claim}"
            )
        for d in sorted(self.diagnoses.values(), key=lambda n: natural_id_key(n.id)):
            ref = f" (refines {self.diagnoses[d.refines].leaf})" if d.refines in self.diagnoses else ""
            lines.append(f"  {d.id}: {d.category} > {d.leaf}{ref}")
        for m in sorted(self.must_not_miss.values(), key=lambda n: natural_id_key(n.id)):
            lines.append(f"  {m.id}: MUST-NOT-MISS {m.dx} [{m.status.value}]")
        return "\n".join(lines)
