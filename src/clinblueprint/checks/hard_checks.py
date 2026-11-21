"""Hard (symbolic) checks applied after drafting and after every edit.

1. Span grounding   every note-sourced Observation cites a verbatim (or
                    near-verbatim, fuzzy >= threshold) span of the note.
2. KG validity      every Diagnosis in a KG category is a node of that
                    category; ``refines`` must point to a KG ancestor.
3. Premise validity a Rationale holds only if all premises exist and each
                    observation premise has the polarity the rationale
                    expects. Invalid rationales are pruned to a fixed point
                    (this is the *cascade* that makes edits propagate).
4. Support          every Diagnosis should have >= 1 supporting rationale.
5. Must-not-miss    the clinician list for each active category is present
                    (missing entries are inserted as ``not_addressed``).

``check_graph`` never mutates its input; it returns ``(G_checked, report)``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from clinblueprint.schema.graph import (
    EvidenceSource,
    MNMStatus,
    MustNotMiss,
    Observation,
    ReasoningGraph,
    Relation,
    Span,
    normalize_label,
)

try:  # rapidfuzz is a core dependency, but keep checks importable without it
    from rapidfuzz import fuzz as _fuzz
except Exception:  # pragma: no cover
    _fuzz = None


@dataclass
class Violation:
    code: str
    node_id: Optional[str]
    message: str
    severity: str = "error"  # error | warning


@dataclass
class CheckReport:
    violations: List[Violation] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    added_mnm: List[str] = field(default_factory=list)
    unsupported_dx: List[str] = field(default_factory=list)
    grounding_rate: float = 1.0
    n_note_observations: int = 0

    def add(self, code: str, node_id: Optional[str], message: str, severity: str = "error") -> None:
        self.violations.append(Violation(code, node_id, message, severity))

    @property
    def ok(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    def codes(self) -> Counter:
        return Counter(v.code for v in self.violations)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


# ----------------------------------------------------------------- grounding
def _span(note: str, start: int, end: int, section: Optional[str], score: float) -> Span:
    return Span(start=start, end=end, text=note[start:end], section=section, score=score)


def ground_observation(
    obs: Observation,
    note: str,
    section_range: Optional[Tuple[int, int]] = None,
    fuzzy_threshold: float = 90.0,
) -> Optional[Span]:
    """Locate ``obs.text`` in ``note``. Returns a Span or None.

    Order: existing valid span -> exact -> case/whitespace-insensitive ->
    fuzzy partial alignment (rapidfuzz) with score >= ``fuzzy_threshold``.
    """
    text = (obs.text or "").strip()
    if not text:
        return None
    if obs.span is not None:
        s = obs.span
        if 0 <= s.start <= s.end <= len(note) and note[s.start:s.end] == s.text and s.text:
            return s
    lo, hi = section_range if section_range else (0, len(note))
    hay = note[lo:hi]

    i = hay.find(text)
    if i >= 0:
        return _span(note, lo + i, lo + i + len(text), obs.section, 100.0)

    toks = text.split()
    if toks:
        pattern = r"\s+".join(re.escape(t) for t in toks)
        m = re.search(pattern, hay, re.IGNORECASE)
        if m:
            return _span(note, lo + m.start(), lo + m.end(), obs.section, 99.0)

    if _fuzz is not None and fuzzy_threshold < 100 and len(text) >= 4 and hay:
        a, b = text.lower(), hay.lower()
        if len(a) != len(text) or len(b) != len(hay):  # unicode case-mapping changed lengths
            a, b = text, hay
        al = _fuzz.partial_ratio_alignment(a, b, score_cutoff=fuzzy_threshold)
        if al is not None and al.score >= fuzzy_threshold and al.dest_end > al.dest_start:
            return _span(note, lo + al.dest_start, lo + al.dest_end, obs.section, float(al.score))
    return None


# --------------------------------------------------------------------- check
def check_graph(
    G: ReasoningGraph,
    note: str,
    kg=None,  # Optional[DiagnosticKG]
    mnm: Optional[Dict[str, List[str]]] = None,
    *,
    repair: bool = True,
    fuzzy_threshold: float = 90.0,
    section_offsets: Optional[Dict[str, Tuple[int, int]]] = None,
    add_missing_mnm: bool = True,
    unsupported_severity: str = "warning",
) -> Tuple[ReasoningGraph, CheckReport]:
    G = G.copy_deep()
    rep = CheckReport()

    # 1. grounding ---------------------------------------------------------
    n_note = n_ok = 0
    for o in list(G.observations.values()):
        if o.source != EvidenceSource.NOTE:
            continue
        n_note += 1
        rng = section_offsets.get(o.section) if (section_offsets and o.section) else None
        span = ground_observation(o, note, rng, fuzzy_threshold)
        if span is None and rng is not None:
            span = ground_observation(o, note, None, fuzzy_threshold)
        if span is None:
            rep.add("UNGROUNDED_SPAN", o.id, f"not found in note: {o.text[:80]!r}")
            if repair:
                G.remove(o.id)
                rep.removed.append(o.id)
        else:
            o.span = span
            n_ok += 1
    rep.n_note_observations = n_note
    rep.grounding_rate = (n_ok / n_note) if n_note else 1.0

    # 2. KG validity --------------------------------------------------------
    if kg is not None:
        for d in list(G.diagnoses.values()):
            if d.category not in kg.categories:
                continue  # open-ended / non-KG category: allowed
            canon = kg.canonical(d.category, d.leaf)
            if canon is None:
                rep.add("INVALID_KG_NODE", d.id, f"{d.leaf!r} is not a node of {d.category!r}")
                if repair:
                    G.remove(d.id)
                    rep.removed.append(d.id)
            else:
                d.leaf = canon
        for d in G.diagnoses.values():
            if not d.refines or d.refines not in G.diagnoses or d.category not in kg.categories:
                continue
            par = G.diagnoses[d.refines]
            if par.category == d.category and not kg.is_valid_refinement(d.category, d.leaf, par.leaf):
                rep.add("INVALID_REFINEMENT", d.id, f"{d.leaf!r} does not refine {par.leaf!r}", "warning")
                if repair:
                    kp = kg.parent(d.category, d.leaf)
                    match = G.find_dx(kp, d.category) if kp else None
                    d.refines = match.id if match is not None and match.id != d.id else None

    # 3. referential integrity + premise validity (fixed point) ---------------
    changed = True
    while changed:
        changed = False
        for d in G.diagnoses.values():
            if d.refines and d.refines not in G.diagnoses:
                rep.add("DANGLING_REFINES", d.id, f"refines missing node {d.refines}", "warning")
                d.refines = None
        for r in list(G.rationales.values()):
            problem: Optional[Tuple[str, str]] = None
            if r.target not in G.diagnoses:
                problem = ("DANGLING_TARGET", f"target {r.target} missing")
            elif not r.premises:
                problem = ("NO_PREMISES", "rationale has no premises")
            else:
                for p in r.premises:
                    if p == r.id:
                        problem = ("SELF_PREMISE", "rationale cites itself")
                    elif p in G.observations:
                        if G.observations[p].polarity != r.expected_polarity(p):
                            problem = (
                                "PREMISE_POLARITY_MISMATCH",
                                f"premise {p} is {G.observations[p].polarity.value}, "
                                f"rationale needs {r.expected_polarity(p).value}",
                            )
                    elif p not in G.rationales:
                        problem = ("DANGLING_PREMISE", f"premise {p} missing")
                    if problem:
                        break
            if problem:
                # polarity mismatches / pruned premises are the normal cascade after an
                # edit; report them as warnings so a post-edit graph can still be "ok"
                sev = "warning" if problem[0] in ("PREMISE_POLARITY_MISMATCH", "DANGLING_PREMISE") else "error"
                rep.add(problem[0], r.id, problem[1], sev)
                if repair:
                    G.remove(r.id)
                    rep.removed.append(r.id)
                    changed = True

    for m in G.must_not_miss.values():
        if repair:
            m.evidence = [e for e in m.evidence if e in G]

    # 4. support -------------------------------------------------------------
    for d in G.diagnoses.values():
        if not G.rationales_for(d.id, Relation.SUPPORTS):
            rep.unsupported_dx.append(d.id)
            rep.add("UNSUPPORTED_DIAGNOSIS", d.id, f"{d.leaf!r} has no supporting rationale",
                    unsupported_severity)

    # 5. must-not-miss coverage ------------------------------------------------
    if mnm:
        cats = sorted({d.category for d in G.diagnoses.values()})
        present = {normalize_label(m.dx) for m in G.must_not_miss.values()}
        for c in cats:
            for need in mnm.get(c, []):
                if normalize_label(need) in present:
                    continue
                rep.add("MISSING_MUST_NOT_MISS", None, f"{c}: {need}", "warning")
                if repair and add_missing_mnm:
                    nid = G.new_id("M")
                    G.add(MustNotMiss(id=nid, dx=need, status=MNMStatus.NOT_ADDRESSED))
                    rep.added_mnm.append(nid)
                present.add(normalize_label(need))
    return G, rep
