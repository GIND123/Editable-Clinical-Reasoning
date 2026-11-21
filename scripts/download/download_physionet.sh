#!/usr/bin/env bash
# Download CREDENTIALED PhysioNet projects (requires approved access for each project).
#   PHYSIONET_USER=<your username> bash scripts/download/download_physionet.sh [project ...]
# You will be prompted for your password; it is never stored by this script.
# VERIFY the version folders below against each project page before running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/data/raw/physionet"
mkdir -p "$DEST"
: "${PHYSIONET_USER:?set PHYSIONET_USER}"

declare -A PROJECTS=(
  [mimic-iv-ext-direct]="mimic-iv-ext-direct/1.0.0"
  [er-reason]="er-reason/1.1.0"
  [mimic-iv-ext-cdm]="mimic-iv-ext-cdm/1.1"
  [mimic-iv-note]="mimic-iv-note/2.2"
  [mimiciv]="mimiciv/3.1"
  [mednli]="mednli/1.0.0"
)

want=("$@")
[ ${#want[@]} -eq 0 ] && want=(mimic-iv-ext-direct er-reason mimic-iv-ext-cdm mimic-iv-note)

for p in "${want[@]}"; do
  path="${PROJECTS[$p]:-}"
  if [ -z "$path" ]; then echo "unknown project $p"; exit 1; fi
  echo "== $p ($path)"
  wget -r -N -c -np -nH --cut-dirs=3 --user "$PHYSIONET_USER" --ask-password \
       -P "$DEST/$p" "https://physionet.org/files/$path/"
done

# MIMIC-IV is large: fetch only what we use (hosp diagnoses/admissions) if you prefer:
#   wget ... https://physionet.org/files/mimiciv/3.1/hosp/diagnoses_icd.csv.gz
echo "Done. Never copy these files outside the approved environment (docs/COMPLIANCE.md)."
