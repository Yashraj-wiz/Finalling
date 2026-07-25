#!/usr/bin/env bash
# setup.sh — create venv and install all dependencies for AIP-Speech.
# Run once from inside aip-speech/:  bash setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
DATA="$ROOT/data"
CACHE="$ROOT/models/cache"

echo "=== AIP-Speech environment setup ==="

# ── 1. Python virtual environment ─────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  echo "[setup] venv created at $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 2. Core Python deps ───────────────────────────────────────────────────────
pip install --quiet --upgrade pip

if [ -f "$ROOT/requirements.txt" ]; then
  pip install --quiet -r "$ROOT/requirements.txt"
else
  echo "[setup] ERROR: requirements.txt not found!"
  exit 1
fi

echo "[setup] Python packages installed."

# ── 3. Ensure local data/cache directories exist ─────────────────────────────
mkdir -p "$DATA"/{bg,bg_raw,speech_asr,speech_kws,speech_saa,esc50,ms_snsd,musan,noisex}
mkdir -p "$ROOT"/{descriptors,itembanks,prereg,manifests,inference,scoring,results}
mkdir -p "$ROOT"/checks/inspection
mkdir -p "$CACHE"

echo "[setup] Directories created."

# ── 4. Remind user of env vars ────────────────────────────────────────────────
cat <<EOF

[setup] DONE. Activate with:
  source $VENV/bin/activate

All scripts set HF_HOME and TORCH_HOME to:
  $CACHE
EOF
