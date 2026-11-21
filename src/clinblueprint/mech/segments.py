"""Map token positions to prompt segments (note / graph slots / edit / answer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from clinblueprint.schema.serialize import SerializedGraph


@dataclass
class Segmented:
    text: str
    input_ids: List[int]
    offsets: List[Tuple[int, int]]
    segments: Dict[str, List[int]] = field(default_factory=dict)       # name -> token positions
    slot_positions: Dict[str, List[int]] = field(default_factory=dict)  # node id -> token positions

    @property
    def last(self) -> int:
        return len(self.input_ids) - 1


def _tok_in(offsets, a: int, b: int) -> List[int]:
    return [i for i, (x, y) in enumerate(offsets) if y > a and x < b and y > x]


def segment_prompt(tokenizer, prompt: str, parts: Sequence[Tuple[str, str]],
                   graph: Optional[SerializedGraph] = None, graph_part: str = "graph") -> Segmented:
    """``prompt`` is the full (chat-templated) string; ``parts`` are (name, text) substrings of it,
    located left-to-right. If ``graph`` is given, slot spans are mapped inside the graph part."""
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    seg = Segmented(prompt, enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]])
    cursor = 0
    for name, text in parts:
        a = prompt.find(text, cursor)
        if a < 0:
            raise ValueError(f"segment {name!r} not found in prompt")
        b = a + len(text)
        seg.segments[name] = _tok_in(seg.offsets, a, b)
        if graph is not None and name == graph_part:
            for s in graph.slots:
                seg.slot_positions[s.node_id] = _tok_in(seg.offsets, a + s.start, a + s.end)
        cursor = b
    covered = {p for v in seg.segments.values() for p in v}
    seg.segments["other"] = [i for i in range(len(seg.input_ids)) if i not in covered]
    return seg
