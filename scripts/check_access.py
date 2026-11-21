"""Report which datasets are present locally and which are still missing.

    python scripts/check_access.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

CHECKS = [
    ("DiReCT knowledge graph (public)", RAW / "direct_kg", "*.json"),
    ("MIMIC-IV-Ext-DiReCT (PhysioNet)", RAW / "physionet" / "mimic-iv-ext-direct", "**/*.json"),
    ("ER-Reason main csv (PhysioNet)", RAW / "physionet" / "er-reason", "**/er_reason.csv"),
    ("ER-Reason SCT gold (PhysioNet)", RAW / "physionet" / "er-reason", "**/gt_clean.csv"),
    ("MIMIC-IV-Ext-CDM (PhysioNet)", RAW / "physionet" / "mimic-iv-ext-cdm", "**/history_of_present_illness.csv"),
    ("MIMIC-IV-Note discharge (PhysioNet)", RAW / "physionet" / "mimic-iv-note", "**/discharge.csv.gz"),
    ("MedNLI (PhysioNet, optional)", RAW / "physionet" / "mednli", "**/*.jsonl"),
    ("MedDistractQA (public)", RAW / "MedDistractQA", "*"),
    ("AHRQ CCSR (public, manual)", RAW / "ccsr", "*"),
    ("SCT-Bench converted jsonl (DUA)", RAW / "sct_bench", "*.jsonl"),
]


def main() -> None:
    width = max(len(n) for n, _, _ in CHECKS)
    for name, base, pattern in CHECKS:
        hits = list(base.glob(pattern)) if base.exists() else []
        status = f"OK ({len(hits)} files)" if hits else "MISSING"
        print(f"{name:<{width}}  {status}")
    try:
        from huggingface_hub import scan_cache_dir

        repos = {r.repo_id for r in scan_cache_dir().repos}
        for rid in ("zou-lab/MedCaseReasoning", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "google/medgemma-4b-it", "Qwen/Qwen3-32B"):
            print(f"{'HF cache: ' + rid:<{width}}  {'OK' if rid in repos else 'MISSING'}")
    except Exception:
        print("(huggingface_hub not installed: skipping HF cache check)")


if __name__ == "__main__":
    main()
