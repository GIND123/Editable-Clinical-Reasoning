"""Mechanistic suite (Axis 3: no bypass). See docs/MECHANISTIC.md.

Four methods that must agree on direction (decision gate G4):
  1. three-path decomposition (graph text / graph hidden-state surplus / direct note path)
  2. graph-swap + minimal-pair activation patching at slot positions
  3. attribution share through slot tokens (gradient x input / IG; circuit-tracer graphs)
  4. SAE disease-feature ablation inside slots vs inside the note, plus leakage probes
"""
