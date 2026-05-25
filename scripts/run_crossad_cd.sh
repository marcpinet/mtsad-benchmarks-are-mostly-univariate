#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$ROOT/models/crossad/run_SYNTH.py" \
    --variant cd \
    --window-sizes "16,32,64,128" \
    --latent-dims "16,32,64,128" \
    "$@"
