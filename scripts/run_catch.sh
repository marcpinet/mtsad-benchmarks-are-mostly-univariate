#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$ROOT/models/catch/run_SYNTH.py" \
    --window-sizes "16,32,64,128" \
    --latent-dims "16,32,64,128" \
    "$@"
