"""Edit vocabulary for ClinEdit-Faith.

Two *speech acts* share the same graph operations:

* HYPOTHETICAL  "What if the troponin were normal?"  A what-if query (or a
                researcher's causal intervention, the do-operator). Never
                validated against the chart; the system must answer it, which
                is how mediation is measured.
* ASSERTION     "The patient has no fever."  A clinician states a fact about
                this patient. It is validated against the record. If it
                contradicts cited evidence, the system FLAGS it (with the
                evidence) instead of silently following it. That is the
                non-sycophancy test. A flag is not a refusal: the clinician can
                override, and the override is logged.

Operations follow the counterfactual operation set of You et al. (arXiv
2603.27820): negate, remove, replace, weaken, intensify, insert. Here they act
on the model's own reasoning graph rather than on the input case.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from clinblueprint.schema.graph import Polarity, Relation


class EditOp(str, Enum):
    NEGATE = "negate"
    REMOVE = "remove"
    REPLACE = "replace"
    WEAKEN = "weaken"
    INTENSIFY = "intensify"
    INSERT = "insert"


class EditMode(str, Enum):
    HYPOTHETICAL = "hypothetical"
    ASSERTION = "assertion"


class EditClass(str, Enum):
    VALID_CORRECTION = "valid_correction"  # moves the graph toward physician gold
    COUNTERFACTUAL = "counterfactual"      # removes/negates key support (what-if)
    NEW_FINDING = "new_finding"            # SCT-style new information
    IRRELEVANT = "irrelevant"              # bystander fact; must change nothing
    INVALID = "invalid"                    # asserted fact contradicts the record


class ExpectedEffect(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"
    FLAG = "flag"


class InsertLink(BaseModel):
    """A rationale to create alongside an inserted observation (gold-provided).

    In ``reason_mode="re_reason"`` links are left empty and the model must
    propose them itself (this is what SCT items test).
    """

    model_config = ConfigDict(extra="forbid")
    category: str
    leaf: str
    relation: Relation = Relation.SUPPORTS
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    claim: str = ""
    refines_leaf: Optional[str] = None


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edit_id: str
    case_id: str
    op: EditOp
    mode: EditMode = EditMode.HYPOTHETICAL
    edit_class: EditClass

    target_node: Optional[str] = None       # node acted upon (all ops except INSERT)
    target_text: Optional[str] = None       # text of the target (for text/note adapters)
    new_text: Optional[str] = None          # REPLACE / INSERT observation text
    new_polarity: Optional[Polarity] = None
    new_section: Optional[str] = None
    strength_factor: Optional[float] = None  # WEAKEN (<1) / INTENSIFY (>1)
    links: List[InsertLink] = Field(default_factory=list)
    replace_leaf: Optional[str] = None       # REPLACE on a diagnosis node
    replace_category: Optional[str] = None

    measured_dx: str                         # dx key whose probability the gold refers to
    expected_effect: ExpectedEffect
    implied_effect: Optional[ExpectedEffect] = None  # for INVALID: what following it would do
    expected_magnitude: Optional[int] = Field(default=None, ge=-2, le=2)  # SCT Likert
    expected_top: Optional[str] = None       # dx key expected to be top-1 after the edit

    gold_source: str = "construction"  # direct_gold | kg_premise | er_reason_sct | sct_bench | physician | construction
    physician_verified: Optional[bool] = None
    description: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    def describe(self) -> str:
        if self.description:
            return self.description
        tgt = self.target_text or self.target_node or ""
        q = "What if" if self.mode == EditMode.HYPOTHETICAL else "Clinician states:"
        if self.op == EditOp.NEGATE:
            return f"{q} '{tgt}' were the opposite of what is recorded?"
        if self.op == EditOp.REMOVE:
            return f"{q} '{tgt}' were not part of the picture?"
        if self.op == EditOp.REPLACE:
            return f"{q} '{tgt}' were instead '{self.new_text}'?"
        if self.op == EditOp.WEAKEN:
            return f"{q} the weight of '{tgt}' were reduced?"
        if self.op == EditOp.INTENSIFY:
            return f"{q} the weight of '{tgt}' were increased?"
        return f"{q} new finding: '{self.new_text}'"
