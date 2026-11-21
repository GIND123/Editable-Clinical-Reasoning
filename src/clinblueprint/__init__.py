"""ClinBlueprint: causally editable clinical reasoning.

A clinical language model whose diagnosis is computed *through* an explicit,
typed, clinician-editable reasoning graph. See docs/IDEA.md and CLINICAL.md.

Importing the package never requires torch; GPU-only modules import their
heavy dependencies lazily.
"""

__version__ = "0.1.0"

from clinblueprint.schema.graph import (  # noqa: F401
    Diagnosis,
    EvidenceSource,
    MNMStatus,
    MustNotMiss,
    Observation,
    Polarity,
    Rationale,
    ReasoningGraph,
    Relation,
    Span,
    dx_key,
)
