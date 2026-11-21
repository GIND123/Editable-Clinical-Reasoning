#!/usr/bin/env bash
# Download PUBLIC resources into data/raw/. Safe to re-run.
# Credentialed PhysioNet data: scripts/download/download_physionet.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

echo "== DiReCT diagnostic knowledge graph (public, github.com/wbw520/DiReCT)"
if [ ! -d "$RAW/direct_repo" ]; then
  git clone --depth 1 https://github.com/wbw520/DiReCT.git "$RAW/direct_repo"
fi
mkdir -p "$RAW/direct_kg"
cp "$RAW/direct_repo/diagnostic_kg/"*.json "$RAW/direct_kg/"
echo "   $(ls "$RAW/direct_kg" | wc -l | tr -d ' ') category files -> data/raw/direct_kg"

echo "== MedDistractQA (irrelevant-edit seed library)"
[ -d "$RAW/MedDistractQA" ] || git clone --depth 1 https://github.com/nyuolab/MedDistractQA.git "$RAW/MedDistractQA"

echo "== Hugging Face datasets (MedCaseReasoning + QA sanity suites) -> HF cache"
python - <<'PY'
from datasets import load_dataset
specs = [
    ("zou-lab/MedCaseReasoning", None),
    ("GBaker/MedQA-USMLE-4-options", None),
    ("openlifescienceai/medmcqa", None),
    ("qiaojin/PubMedQA", "pqa_labeled"),
    ("TsinghuaC3I/MedXpertQA", "Text"),
]
for hf_id, conf in specs:
    try:
        ds = load_dataset(hf_id, conf) if conf else load_dataset(hf_id)
        print(f"   ok  {hf_id} {conf or ''}: " + ", ".join(f"{k}={len(v)}" for k, v in ds.items()))
    except Exception as e:  # noqa: BLE001
        print(f"   FAIL {hf_id}: {e}")
PY

cat <<'TXT'
== Manual / optional public downloads
   AHRQ CCSR (ER-Reason dx grouping): https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp -> data/raw/ccsr/
   MultiCaRe (Zenodo; see doi:10.1016/j.dib.2023.110008)       -> data/raw/multicare/
   PMC-Patients is loaded from HF on demand (zhengyun21/PMC-Patients)
   DDXPlus (optional): https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374 -> data/raw/ddxplus/
   SCT-Bench: public items + held-out DUA via https://www.concor.dance/
TXT
