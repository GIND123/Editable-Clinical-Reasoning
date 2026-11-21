"""Build evaluation systems from YAML specs (configs/eval/evaluate.yaml).

    systems:
      - {type: clinblueprint, name: cb_hard_qwen3_4b, readout: linear, readout_path: ..., adapter: ...}
      - {type: clinblueprint, name: cb_oracle_graph, regime: gold_graph}
      - {type: cot, name: cot_sft_qwen3_4b, adapter: ...}
      - {type: note_edit, name: note_edit_qwen3_4b}
      - {type: in_context, name: ic_dpo_qwen3_4b, adapter: ..., graph_from: clinblueprint}
"""

from __future__ import annotations

import json
from typing import Any, Dict

from clinblueprint.compliance.guard import assert_local


def _model(spec: Dict[str, Any], cfg: Dict[str, Any], shared: Dict[str, Any]):
    from clinblueprint.models.backbone import from_dict, load_model, load_tokenizer

    mcfg = {**cfg.get("model", {}), **spec.get("model", {})}
    key = (mcfg["name_or_path"], spec.get("adapter"))
    if key not in shared:
        assert_local(spec.get("backend", "hf"), spec.get("endpoint"), cfg.get("provenance", "physionet"))
        b = from_dict(mcfg)
        shared[key] = (load_model(b, adapter_path=spec.get("adapter")), load_tokenizer(b))
    return shared[key]


def build_system(spec: Dict[str, Any], cfg: Dict[str, Any], shared: Dict[str, Any]):
    from clinblueprint.baselines.systems import ClinBlueprintSystem, CoTSystem, InContextGraphSystem, NoteEditSystem
    from clinblueprint.edits.validator import EditValidator
    from clinblueprint.kg import DiagnosticKG, load_must_not_miss
    from clinblueprint.pipeline import ClinBlueprint, StaticDrafter
    from clinblueprint.readout import HardReadout, LinearReadout

    kg = DiagnosticKG.from_dir(cfg["kg_dir"]) if cfg.get("kg_dir") else None
    mnm = load_must_not_miss(cfg["must_not_miss"]) if cfg.get("must_not_miss") else None
    t = spec["type"]
    if t == "clinblueprint":
        if spec.get("readout", "hard") == "linear":
            ro = LinearReadout(kg)
            ro.load_state_dict(json.load(open(spec["readout_path"])))
        else:
            ro = HardReadout(kg)
        if spec.get("regime") == "gold_graph":
            drafter = StaticDrafter({})
        else:
            # no adapter = untuned backbone + drafter prompt = the schema-guided prompting baseline
            from clinblueprint.models.drafter import LMGraphDrafter

            model, tok = _model(spec, cfg, shared)
            drafter = LMGraphDrafter(model, tok, kg)
            wrap = spec.get("drafter_wrapper")
            if wrap == "self_refine":
                from clinblueprint.baselines.refine_and_sample import SelfRefineDrafter

                drafter = SelfRefineDrafter(drafter, rounds=spec.get("rounds", 1))
            elif wrap == "best_of_n":
                from clinblueprint.baselines.refine_and_sample import BestOfNDrafter

                drafter = BestOfNDrafter(drafter, n=spec.get("n", 8), kg=kg)
        validator = EditValidator()
        if spec.get("validator_adapter"):
            from clinblueprint.training.validator_train import HFEntailmentVerifier

            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            base = AutoModelForSequenceClassification.from_pretrained(cfg["model"]["name_or_path"], num_labels=3)
            vm = PeftModel.from_pretrained(base, spec["validator_adapter"]).eval()
            cal = json.load(open(spec["validator_calibration"]))
            import numpy as np

            validator = EditValidator(verifier=HFEntailmentVerifier(vm, AutoTokenizer.from_pretrained(spec["validator_adapter"])),
                                      threshold=cal["threshold"],
                                      calibrator=lambda p: float(np.interp(p, cal["isotonic_x"], cal["isotonic_y"])))
        pipe = ClinBlueprint(drafter, ro, kg=kg, mnm=mnm, validator=validator)
        return ClinBlueprintSystem(pipe, name=spec["name"], regime=spec.get("regime", "model_graph"))
    model, tok = _model(spec, cfg, shared)
    if t == "cot":
        return CoTSystem(model, tok, name=spec["name"], flag_prompting=spec.get("flag_prompting", True))
    if t == "note_edit":
        return NoteEditSystem(model, tok, name=spec["name"])
    if t == "counterfactual_multiagent":
        from clinblueprint.baselines.systems import CounterfactualMultiAgentSystem

        return CounterfactualMultiAgentSystem(model, tok, name=spec["name"], max_findings=spec.get("max_findings", 8))
    if t == "in_context":
        from clinblueprint.models.drafter import LMGraphDrafter
        from clinblueprint.schema.graph import ReasoningGraph

        if spec.get("graph_from") == "gold":
            src = lambda case: ReasoningGraph.from_dict(case.gold_graph)  # noqa: E731
        else:
            gd = LMGraphDrafter(*_model({"adapter": spec.get("drafter_adapter")}, cfg, shared), kg)
            src = lambda case: gd.draft(case.note, case.case_id)  # noqa: E731
        return InContextGraphSystem(model, tok, src, name=spec["name"])
    raise ValueError(f"unknown system type {t}")
