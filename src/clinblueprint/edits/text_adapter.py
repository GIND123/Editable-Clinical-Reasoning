"""Apply a graph Edit to *free text* (a CoT trace or the input note).

This is what makes the comparison with chain-of-thought fair: the same edit,
with the same gold effect, is applied

* to ClinBlueprint's graph (``edits.ops.apply_edit``),
* to a CoT model's reasoning trace, after which the model continues from the
  edited trace (trace intervention, cf. Lanham et al. 2023), and
* to the input note (the input-level regime of You et al. 2603.27820).

The templates are deliberately plain. The clinician co-author reviews them in
clinician/tasks/03_edit_verification.md.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from clinblueprint.edits.types import Edit, EditMode, EditOp
from clinblueprint.schema.graph import Polarity, ReasoningGraph

_SENT_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)


def sentences(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _SENT_RE.finditer(text) if m.group().strip()]


def _find(text: str, needle: str) -> List[Tuple[int, int]]:
    if not needle:
        return []
    pat = r"\s+".join(re.escape(t) for t in needle.split())
    hits = [(m.start(), m.end()) for m in re.finditer(pat, text, re.IGNORECASE)]
    if hits:
        return hits
    # fall back to the longest content words
    words = [w for w in re.findall(r"[A-Za-z0-9.%]+", needle) if len(w) > 3]
    if not words:
        return []
    key = max(words, key=len)
    return [(m.start(), m.end()) for m in re.finditer(re.escape(key), text, re.IGNORECASE)]


def _containing_sentences(text: str, needle: str) -> List[Tuple[int, int]]:
    hits = _find(text, needle)
    out = []
    for s, e in sentences(text):
        if any(s <= h0 < e for h0, _ in hits):
            out.append((s, e))
    return out


def _target_text(edit: Edit, G: Optional[ReasoningGraph]) -> str:
    if edit.target_text:
        return edit.target_text
    if G is not None and edit.target_node in getattr(G, "observations", {}):
        return G.observations[edit.target_node].text
    return ""


def _prefix(edit: Edit) -> str:
    return "Suppose that " if edit.mode == EditMode.HYPOTHETICAL else "Clinician correction: "


def negation_sentence(phrase: str, currently: Polarity) -> str:
    phrase = phrase.strip().rstrip(".")
    if currently == Polarity.PRESENT:
        return f"there is no {phrase}"
    return f"{phrase} is present"


def apply_edit_to_text(text: str, edit: Edit, G: Optional[ReasoningGraph] = None) -> str:
    tgt = _target_text(edit, G)
    cur = Polarity.PRESENT
    if G is not None and edit.target_node in getattr(G, "observations", {}):
        cur = G.observations[edit.target_node].polarity

    def replace_sentences(new: str) -> str:
        spans = _containing_sentences(text, tgt)
        if not spans:
            return text.rstrip() + "\n" + new
        out, last = [], 0
        for i, (s, e) in enumerate(spans):
            out.append(text[last:s])
            if i == 0:
                out.append(new)
            last = e
        out.append(text[last:])
        return "".join(out)

    if edit.op == EditOp.NEGATE:
        return replace_sentences(f"{_prefix(edit)}{negation_sentence(tgt, cur)}.")
    if edit.op == EditOp.REMOVE:
        return replace_sentences("")
    if edit.op == EditOp.REPLACE:
        hits = _find(text, tgt)
        if hits and edit.new_text:
            s, e = hits[0]
            return text[:s] + edit.new_text + text[e:]
        return text.rstrip() + f"\n{_prefix(edit)}{edit.new_text or ''}."
    if edit.op == EditOp.WEAKEN:
        return replace_sentences(f"{_prefix(edit)}the finding '{tgt}' is equivocal and of limited weight.")
    if edit.op == EditOp.INTENSIFY:
        return replace_sentences(f"{_prefix(edit)}the finding '{tgt}' is marked and unequivocal.")
    if edit.op == EditOp.INSERT:
        body = edit.new_text or ""
        if edit.new_polarity == Polarity.ABSENT:
            body = negation_sentence(body, Polarity.PRESENT)
        return text.rstrip() + f"\n{_prefix(edit)}{body}."
    return text
