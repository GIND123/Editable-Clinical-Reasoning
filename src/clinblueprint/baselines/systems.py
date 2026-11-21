"""Editable-system adapters. Every system receives the SAME edits with the SAME gold.

* ClinBlueprintSystem   edits act on the model's own graph (or the gold graph:
                        "oracle-graph" regime, which isolates the readout)
* CoTSystem             edits act on the model's chain-of-thought trace; the model
                        continues from the edited trace (trace intervention).
                        Optional flag prompting gives it a fair non-sycophancy channel.
* NoteEditSystem        edits act on the input note (the input-level regime of
                        You et al. 2603.27820); the model answers again
* InContextGraphSystem  LM reads note + graph + edit and answers or FLAGs
                        (the DPO-trained generative variant)

All systems score the same fixed candidate pool per case, so probabilities are
comparable across systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from clinblueprint.data.records import CaseRecord
from clinblueprint.edits.align import retarget
from clinblueprint.edits.text_adapter import apply_edit_to_text
from clinblueprint.edits.types import Edit, EditMode
from clinblueprint.readout.differential import Differential
from clinblueprint.schema.graph import ReasoningGraph, split_dx_key
from clinblueprint.schema.serialize import serialize_graph


@dataclass
class SystemEditResult:
    applicable: bool
    flagged: bool
    after: Optional[Differential]
    p_contradict: Optional[float] = None
    note: str = ""


class EditableSystem(Protocol):
    name: str

    def prepare(self, case: CaseRecord) -> Any: ...

    def differential(self, case: CaseRecord, state: Any) -> Differential: ...

    def apply(self, case: CaseRecord, state: Any, edit: Edit, reason_mode: str = "frozen") -> SystemEditResult: ...


def labels(case: CaseRecord) -> List[str]:
    return [split_dx_key(k)[1] for k in (case.candidates or [])]


def _dist(keys: Sequence[str], p: np.ndarray) -> Differential:
    return Differential.from_probs(dict(zip(keys, p.tolist())))


# --------------------------------------------------------------------------- ClinBlueprint
class ClinBlueprintSystem:
    def __init__(self, pipeline, name: str = "clinblueprint", regime: str = "model_graph",
                 graph_cache: Optional[Dict[str, ReasoningGraph]] = None):
        self.pipe, self.name, self.regime = pipeline, name, regime
        self.cache = graph_cache or {}

    def prepare(self, case: CaseRecord) -> Dict[str, Any]:
        if self.regime == "gold_graph":
            G, _ = self.pipe.check(ReasoningGraph.from_dict(case.gold_graph), case.note, case.section_offsets)
        elif case.case_id in self.cache:
            G = self.cache[case.case_id]
        else:
            G = self.pipe.run(case.note, case.case_id, section_offsets=case.section_offsets).graph
        return {"G": G}

    def differential(self, case: CaseRecord, state) -> Differential:
        return self.pipe.readout(state["G"], case.candidates)

    def apply(self, case, state, edit, reason_mode="frozen") -> SystemEditResult:
        e = edit if self.regime == "gold_graph" else retarget(edit, state["G"])
        if e is None:
            return SystemEditResult(False, False, None, note="target not in model graph")
        out = self.pipe.apply_clinician_edit(state["G"], e, case.note, reason_mode=reason_mode,
                                             fixed_candidates=case.candidates, section_offsets=case.section_offsets)
        pc = out.verdict.p_contradict if out.verdict is not None else None
        return SystemEditResult(True, out.flagged, out.after, pc)


# --------------------------------------------------------------------------- LM-based systems
class _LMBase:
    def __init__(self, model, tokenizer, name: str, flag_prompting: bool = True, max_new_tokens: int = 1024):
        self.model, self.tok, self.name = model, tokenizer, name
        self.flag_prompting = flag_prompting
        self.max_new_tokens = max_new_tokens

    def _cand_dist(self, prompt: str, case: CaseRecord) -> Differential:
        from clinblueprint.models.scoring import candidate_distribution

        return _dist(case.candidates or [], candidate_distribution(self.model, self.tok, prompt, labels(case)))

    def _p_flag(self, prompt: str) -> float:
        from clinblueprint.models.scoring import continuation_logprobs

        lp = continuation_logprobs(self.model, self.tok, prompt, [" FLAG", " Final diagnosis"], length_normalize=False)
        z = np.exp(lp - lp.max())
        return float(z[0] / z.sum())


class CoTSystem(_LMBase):
    def __init__(self, model, tokenizer, name: str = "cot", **kw):
        super().__init__(model, tokenizer, name, **kw)

    def _prompt(self, case: CaseRecord) -> str:
        from clinblueprint.models import prompts as P
        from clinblueprint.models.backbone import chat_prompt

        return chat_prompt(self.tok, P.cot_messages(case.note, labels(case)))

    def prepare(self, case: CaseRecord) -> Dict[str, Any]:
        from clinblueprint.models.backbone import generate

        out = generate(self.model, self.tok, [self._prompt(case)], max_new_tokens=self.max_new_tokens)[0]
        trace = out.split("Final diagnosis:")[0].rstrip()
        return {"trace": trace}

    def differential(self, case, state) -> Differential:
        return self._cand_dist(self._prompt(case) + state["trace"] + "\nFinal diagnosis:", case)

    def apply(self, case, state, edit, reason_mode="frozen") -> SystemEditResult:
        trace2 = apply_edit_to_text(state["trace"], edit)
        base = self._prompt(case) + trace2
        pc = None
        if self.flag_prompting and edit.mode == EditMode.ASSERTION:
            pc = self._p_flag(base + "\nIf the clinician correction contradicts the record, answer FLAG. Otherwise:")
            if pc > 0.5:
                return SystemEditResult(True, True, self.differential(case, state), pc)
        return SystemEditResult(True, False, self._cand_dist(base + "\nFinal diagnosis:", case), pc)


class NoteEditSystem(_LMBase):
    """Input-level counterfactual editing (training-free, model-agnostic regime)."""

    def __init__(self, model, tokenizer, name: str = "note_edit", **kw):
        super().__init__(model, tokenizer, name, **kw)

    def _prompt(self, note: str, case: CaseRecord) -> str:
        from clinblueprint.models.backbone import chat_prompt

        cands = "\n".join(f"- {c}" for c in labels(case))
        return chat_prompt(self.tok, [{"role": "user", "content":
                                       f"Clinical note:\n{note}\n\nCandidate diagnoses:\n{cands}\n\n"
                                       "Give the single most likely diagnosis."}]) + "Final diagnosis:"

    def prepare(self, case):
        return {}

    def differential(self, case, state):
        return self._cand_dist(self._prompt(case.note, case), case)

    def apply(self, case, state, edit, reason_mode="frozen"):
        return SystemEditResult(True, False, self._cand_dist(self._prompt(apply_edit_to_text(case.note, edit), case), case))


class CounterfactualMultiAgentSystem(_LMBase):
    """You et al.-style input-level counterfactual reasoning as an editable system (edits act on the note)."""

    def __init__(self, model, tokenizer, name: str = "counterfactual_multiagent", max_findings: int = 8, **kw):
        super().__init__(model, tokenizer, name, **kw)
        self.max_findings = max_findings

    def _dist_for(self, note: str, case: CaseRecord) -> Differential:
        from clinblueprint.baselines.counterfactual_multiagent import counterfactual_rank

        names = labels(case)
        probs = counterfactual_rank(self.model, self.tok, note, names, self.max_findings)
        return _dist(case.candidates or [], np.array([probs[n] for n in names]))

    def prepare(self, case):
        return {}

    def differential(self, case, state):
        return self._dist_for(case.note, case)

    def apply(self, case, state, edit, reason_mode="frozen"):
        return SystemEditResult(True, False, self._dist_for(apply_edit_to_text(case.note, edit), case))


class InContextGraphSystem(_LMBase):
    """Generative variant: note + serialised graph (+ edit) -> answer or FLAG."""

    def __init__(self, model, tokenizer, graph_source, name: str = "in_context_graph", **kw):
        super().__init__(model, tokenizer, name, **kw)
        self.graph_source = graph_source  # callable(case) -> ReasoningGraph

    def _prompt(self, case, G, edit=None) -> str:
        from clinblueprint.models.backbone import chat_prompt
        from clinblueprint.training.dpo import in_context_messages

        return chat_prompt(self.tok, in_context_messages(case.note, serialize_graph(G).text, edit))

    def prepare(self, case):
        return {"G": self.graph_source(case)}

    def differential(self, case, state):
        return self._cand_dist(self._prompt(case, state["G"]) + "Final diagnosis:", case)

    def apply(self, case, state, edit, reason_mode="frozen"):
        from clinblueprint.edits.ops import apply_edit

        e = retarget(edit, state["G"])
        if e is None:
            return SystemEditResult(False, False, None, note="target not in graph")
        p = self._prompt(case, state["G"], e)
        pc = self._p_flag(p) if e.mode == EditMode.ASSERTION else None
        if pc is not None and pc > 0.5:
            return SystemEditResult(True, True, self.differential(case, state), pc)
        G2 = apply_edit(state["G"], e)
        return SystemEditResult(True, False, self._cand_dist(self._prompt(case, G2, e) + "Final diagnosis:", case), pc)
