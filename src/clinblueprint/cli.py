"""Command-line entry point:  clinblueprint <command> --config <yaml> [--set key=value ...]"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

COMMANDS = {
    "prepare-data": ("clinblueprint.data.prepare", "main"),
    "build-silver": ("clinblueprint.data.silver", "main"),
    "train-sft": ("clinblueprint.training.sft", "main"),
    "train-cot-sft": ("clinblueprint.training.cot_sft", "main"),
    "draft-graphs": ("clinblueprint.models.draft_corpus", "main"),
    "train-readout": ("clinblueprint.training.readout_train", "main"),
    "train-validator": ("clinblueprint.training.validator_train", "main"),
    "train-faithfulness": ("clinblueprint.training.faithfulness", "main"),
    "train-dpo": ("clinblueprint.training.dpo", "main"),
    "build-benchmark": ("clinblueprint.benchmark.build", "main"),
    "evaluate": ("clinblueprint.eval.runner", "main"),
    "evaluate-sct": ("clinblueprint.eval.sct_runner", "main"),
    "mech": ("clinblueprint.mech.run", "main"),
}


def _study(args, cfg) -> None:
    from clinblueprint.study.design import assign, simulate_power

    if args.command == "study-power":
        p = simulate_power(cfg["n_clinicians"], cfg["n_cases"], cfg["model_error_rate"], cfg["p_detect"],
                           cfg.get("clinician_sd", 0.5), cfg.get("n_sim", 500))
        print(json.dumps({"power": p, **cfg}, indent=2))
    else:
        import csv

        from clinblueprint.utils.io import read_jsonl

        rows = assign(cfg["clinicians"], read_jsonl(cfg["cases"]), cfg.get("seed", 0))
        with open(cfg["output"], "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} schedule rows to {cfg['output']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="clinblueprint")
    ap.add_argument("command", choices=sorted(COMMANDS) + ["study-power", "study-randomize", "demo"])
    ap.add_argument("--config", required=False)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.lr=1e-4")
    args = ap.parse_args(argv)
    if args.command == "demo":
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
        import toy_demo  # type: ignore

        toy_demo.main()
        return
    if not args.config:
        ap.error("--config is required")
    from clinblueprint.utils.config import load_config

    cfg = load_config(args.config, args.set)
    if args.command.startswith("study-"):
        _study(args, cfg)
        return
    mod, fn = COMMANDS[args.command]
    getattr(importlib.import_module(mod), fn)(cfg)


if __name__ == "__main__":
    main()
