"""Edit validator: accept corrections consistent with the record, flag contradictions.

Two layers:

1. ``RuleChecker`` (symbolic, high precision)
   * polarity contradiction: the clinician asserts the opposite polarity of a
     finding the note states explicitly (NegEx-style trigger window);
   * numeric contradiction: an asserted measurement differs from the value
     the note records for the same anchor term (e.g. temperature).
2. ``EntailmentVerifier`` (learned; see training/validator_train.py)
   P(contradiction | note evidence window, claim), calibrated with isotonic
   regression on a dev split.

Only ASSERTION-mode edits are validated. Hypothetical queries are always
answered. REMOVE assertions mean "not relevant", not "false", so they are not
contradiction-checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from clinblueprint.edits.types import Edit, EditMode, EditOp
from clinblueprint.schema.graph import Polarity, ReasoningGraph

NEG_TRIGGERS = [
    "no", "not", "denies", "denied", "deny", "without", "negative for", "absence of", "absent",
    "free of", "ruled out", "rules out", "never", "none", "neither", "nor", "resolved",
]
_NEG_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(NEG_TRIGGERS, key=len, reverse=True)) + r")\b",
                     re.IGNORECASE)
_SENT_BOUND = re.compile(r"[.;\n]|\b(?:but|however|although|though|except|which|who|whereas)\b", re.IGNORECASE)


@dataclass
class Verdict:
    label: str  # consistent | contradicts_evidence | new_information | hypothetical
    p_contradict: float = 0.0
    reasons: List[str] = field(default_factory=list)
    cited_spans: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.label == "contradicts_evidence"


def _mentions(note: str, phrase: str) -> List[Tuple[int, int]]:
    phrase = phrase.strip().rstrip(".")
    if not phrase:
        return []
    pat = r"\b" + r"\s+".join(re.escape(t) for t in phrase.split()) + r"\b"
    return [(m.start(), m.end()) for m in re.finditer(pat, note, re.IGNORECASE)]


def mention_polarity(note: str, start: int, window_chars: int = 60) -> Polarity:
    """Polarity of the mention starting at ``start``: negation trigger in the same
    clause within ``window_chars`` before it => ABSENT."""
    lo = max(0, start - window_chars)
    pre = note[lo:start]
    b = list(_SENT_BOUND.finditer(pre))
    if b:
        pre = pre[b[-1].end():]
    return Polarity.ABSENT if _NEG_RE.search(pre) else Polarity.PRESENT


def note_polarity(note: str, phrase: str) -> Tuple[Optional[Polarity], List[Tuple[int, int]]]:
    hits = _mentions(note, phrase)
    if not hits:
        return None, []
    pols = [mention_polarity(note, s) for s, _ in hits]
    if any(p == Polarity.PRESENT for p in pols):
        return Polarity.PRESENT, [h for h, p in zip(hits, pols) if p == Polarity.PRESENT]
    return Polarity.ABSENT, hits


def strip_negation(text: str) -> Tuple[str, Polarity]:
    t = text.strip().rstrip(".")
    m = re.match(r"^(?:there is |there are |patient has |patient )?(no|denies|without|negative for|absent)\s+(.*)$",
                 t, re.IGNORECASE)
    if m:
        return m.group(2), Polarity.ABSENT
    return t, Polarity.PRESENT


_NUM_RE = re.compile(r"([A-Za-z][A-Za-z ]{1,20}?)\s*(?:of|was|is|=|:)?\s*(\d+(?:\.\d+)?)")


def _anchored_numbers(text: str) -> List[Tuple[str, float, int, int]]:
    out = []
    for m in _NUM_RE.finditer(text):
        anchor = m.group(1).strip().lower().split()[-1]
        out.append((anchor, float(m.group(2)), m.start(2), m.end(2)))
    return out


class RuleChecker:
    def __init__(self, numeric_rel_tol: float = 0.05):
        self.tol = numeric_rel_tol

    def check(self, edit: Edit, note: str, G: Optional[ReasoningGraph]) -> Tuple[Optional[float], List[str], List]:
        reasons: List[str] = []
        spans: List[Tuple[int, int, str]] = []
        obs = G.observations.get(edit.target_node) if (G is not None and edit.target_node) else None

        claims: List[Tuple[str, Polarity]] = []
        if edit.op == EditOp.NEGATE and obs is not None:
            new_pol = edit.new_polarity or (Polarity.ABSENT if obs.polarity != Polarity.ABSENT else Polarity.PRESENT)
            claims.append((obs.text, new_pol))
        elif edit.op == EditOp.REPLACE and obs is not None and edit.new_text:
            phrase, pol = strip_negation(edit.new_text)
            claims.append((phrase, edit.new_polarity or pol))
            # replacing a recorded finding is itself a claim that the original is absent
            claims.append((obs.text, Polarity.ABSENT))
        elif edit.op == EditOp.INSERT and edit.new_text:
            phrase, pol = strip_negation(edit.new_text)
            claims.append((phrase, edit.new_polarity or pol))
        elif edit.op == EditOp.WEAKEN and obs is not None and edit.new_text:
            claims.append(strip_negation(edit.new_text))

        hit = False
        for phrase, pol in claims:
            npol, where = note_polarity(note, phrase)
            if npol is not None and npol != pol:
                hit = True
                s, e = where[0]
                spans.append((s, e, note[s:e]))
                reasons.append(f"record states '{note[s:e]}' as {npol.value}; edit asserts {pol.value}")

        # numeric contradictions
        if edit.new_text:
            note_nums = _anchored_numbers(note)
            for anchor, val, _, _ in _anchored_numbers(edit.new_text):
                for a2, v2, s2, e2 in note_nums:
                    if a2 == anchor and abs(v2 - val) > self.tol * max(abs(v2), 1e-6):
                        hit = True
                        spans.append((s2, e2, note[s2:e2]))
                        reasons.append(f"record has {anchor} {v2:g}; edit asserts {val:g}")
                        break
        if hit:
            return 1.0, reasons, spans
        return (None if not claims else 0.0), reasons, spans


class EntailmentVerifier(Protocol):
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Return probabilities with keys 'entailment', 'neutral', 'contradiction'."""


def claim_text(edit: Edit, G: Optional[ReasoningGraph]) -> str:
    obs = G.observations.get(edit.target_node) if (G is not None and edit.target_node) else None
    if edit.op == EditOp.NEGATE and obs is not None:
        return (f"The patient does not have {obs.text}." if obs.polarity == Polarity.PRESENT
                else f"The patient has {obs.text}.")
    if edit.op in (EditOp.INSERT, EditOp.REPLACE) and edit.new_text:
        return f"The patient: {edit.new_text}."
    return edit.describe()


def evidence_window(note: str, spans: List[Tuple[int, int, str]], obs_span=None, max_chars: int = 1500) -> str:
    centers = [s for s, _, _ in spans]
    if obs_span is not None:
        centers.append(obs_span.start)
    if not centers:
        return note[:max_chars]
    c = centers[0]
    lo = max(0, c - max_chars // 2)
    return note[lo:lo + max_chars]


class EditValidator:
    def __init__(
        self,
        rules: Optional[RuleChecker] = None,
        verifier: Optional[EntailmentVerifier] = None,
        threshold: float = 0.5,
        calibrator: Optional[Callable[[float], float]] = None,
    ):
        self.rules = rules or RuleChecker()
        self.verifier = verifier
        self.threshold = threshold
        self.calibrator = calibrator

    def validate(self, edit: Edit, note: str, G: Optional[ReasoningGraph] = None) -> Verdict:
        if edit.mode == EditMode.HYPOTHETICAL:
            return Verdict("hypothetical", 0.0, ["what-if query: answered, not validated against the record"])
        if edit.op == EditOp.REMOVE:
            return Verdict("consistent", 0.0, ["removal = 'not relevant'; not a factual claim"])
        p_rule, reasons, spans = self.rules.check(edit, note, G)
        p_nli = None
        if self.verifier is not None:
            obs = G.observations.get(edit.target_node) if (G is not None and edit.target_node) else None
            prem = evidence_window(note, spans, obs.span if obs is not None else None)
            probs = self.verifier.predict(prem, claim_text(edit, G))
            p_nli = float(probs.get("contradiction", 0.0))
            if self.calibrator is not None:
                p_nli = float(self.calibrator(p_nli))
            reasons.append(f"verifier P(contradiction)={p_nli:.2f}")
        p = max(p_rule or 0.0, p_nli or 0.0)
        if p >= self.threshold:
            return Verdict("contradicts_evidence", p, reasons, spans)
        if edit.op == EditOp.INSERT and p_rule is not None and not spans:
            phrase, _ = strip_negation(edit.new_text or "")
            if not _mentions(note, phrase):
                return Verdict("new_information", p, reasons or ["not mentioned in the record"], spans)
        return Verdict("consistent", p, reasons, spans)
