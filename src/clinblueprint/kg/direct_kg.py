"""DiReCT diagnostic knowledge graph.

The public DiReCT repository (github.com/wbw520/DiReCT, folder
``diagnostic_kg/``) ships one JSON file per disease category::

    {"diagnostic": {"Suspected ACS": {"Strongly Suspected ACS":
                        {"NSTE-ACS": {"UA": [], "NSTEMI": []}, "STEMI-ACS": []}}},
     "knowledge":  {"Suspected ACS": {"Risk Factors": "...", "Symptoms": "..."},
                    "NSTEMI": "The peak hs-cTn exceeded the 99th percentile ...", ...}}

``diagnostic`` is a nested tree whose leaves are ``[]`` (or ``{}``);
``knowledge`` maps nodes to premise text (the "golden standard" criteria).
Node names are unique within a category, not necessarily globally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from clinblueprint.schema.graph import normalize_label


class DiagnosticKG:
    def __init__(self, categories: Dict[str, dict]):
        self._raw = categories
        self._parent: Dict[Tuple[str, str], Optional[str]] = {}
        self._children: Dict[Tuple[str, str], List[str]] = {}
        self._depth: Dict[Tuple[str, str], int] = {}
        self._premise: Dict[Tuple[str, str], str] = {}
        self._norm: Dict[Tuple[str, str], str] = {}  # (cat, normalized) -> canonical
        for cat, data in categories.items():
            self._walk(cat, data.get("diagnostic", {}) or {}, None, 0)
            for node, prem in (data.get("knowledge", {}) or {}).items():
                self._premise[(cat, node)] = self._flatten_premise(prem)

    # ----------------------------------------------------------------- loading
    @classmethod
    def from_dir(cls, path: str | Path) -> "DiagnosticKG":
        path = Path(path)
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no KG json files in {path} (run scripts/download/download_public.sh)")
        cats = {}
        for f in files:
            with open(f, encoding="utf-8") as fh:
                cats[f.stem] = json.load(fh)
        return cls(cats)

    @staticmethod
    def _flatten_premise(prem) -> str:
        if isinstance(prem, dict):
            return "; ".join(f"{k}: {v}" for k, v in prem.items())
        if isinstance(prem, list):
            return "; ".join(map(str, prem))
        return str(prem)

    def _walk(self, cat: str, tree, parent: Optional[str], depth: int) -> None:
        if not isinstance(tree, dict):
            return
        for node, sub in tree.items():
            key = (cat, node)
            self._parent[key] = parent
            self._depth[key] = depth
            self._children.setdefault(key, [])
            self._norm[(cat, normalize_label(node))] = node
            if parent is not None:
                self._children[(cat, parent)].append(node)
            self._walk(cat, sub, node, depth + 1)

    # ------------------------------------------------------------------ queries
    @property
    def categories(self) -> List[str]:
        return sorted(self._raw)

    def nodes(self, cat: str) -> List[str]:
        return [n for (c, n) in self._parent if c == cat]

    def has_node(self, cat: str, node: str) -> bool:
        return self.canonical(cat, node) is not None

    def canonical(self, cat: str, node: str) -> Optional[str]:
        if (cat, node) in self._parent:
            return node
        return self._norm.get((cat, normalize_label(node)))

    def parent(self, cat: str, node: str) -> Optional[str]:
        c = self.canonical(cat, node)
        return None if c is None else self._parent.get((cat, c))

    def children(self, cat: str, node: str) -> List[str]:
        c = self.canonical(cat, node)
        return [] if c is None else list(self._children.get((cat, c), []))

    def ancestors(self, cat: str, node: str) -> List[str]:
        out: List[str] = []
        cur = self.parent(cat, node)
        while cur is not None:
            out.append(cur)
            cur = self._parent.get((cat, cur))
        return out

    def depth(self, cat: str, node: str) -> int:
        c = self.canonical(cat, node)
        return -1 if c is None else self._depth[(cat, c)]

    def is_leaf(self, cat: str, node: str) -> bool:
        c = self.canonical(cat, node)
        return c is not None and not self._children.get((cat, c))

    def leaves(self, cat: str) -> List[str]:
        return [n for n in self.nodes(cat) if not self._children.get((cat, n))]

    def roots(self, cat: str) -> List[str]:
        return [n for n in self.nodes(cat) if self._parent[(cat, n)] is None]

    def is_valid_refinement(self, cat: str, child: str, parent: str) -> bool:
        """True if ``parent`` is an ancestor of ``child`` in category ``cat``."""
        p = self.canonical(cat, parent)
        return p is not None and p in self.ancestors(cat, child)

    def premise(self, cat: str, node: str) -> str:
        c = self.canonical(cat, node)
        return "" if c is None else self._premise.get((cat, c), "")

    def categories_of(self, node: str) -> List[str]:
        n = normalize_label(node)
        return [c for (c, nn) in self._norm if nn == n]

    def iter_leaves(self) -> Iterable[Tuple[str, str]]:
        for cat in self.categories:
            for leaf in self.leaves(cat):
                yield cat, leaf

    def describe(self, cat: str) -> str:
        """Indented tree with premises (used in drafter prompts with ``use_premise``)."""
        lines: List[str] = []

        def rec(node: str, ind: int) -> None:
            prem = self.premise(cat, node)
            lines.append("  " * ind + f"- {node}" + (f": {prem}" if prem else ""))
            for ch in self._children.get((cat, node), []):
                rec(ch, ind + 1)

        for r in self.roots(cat):
            rec(r, 0)
        return "\n".join(lines)
