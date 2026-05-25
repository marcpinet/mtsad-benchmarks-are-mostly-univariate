#!/usr/bin/env bash
# Run every per-model runner on both synthetic datasets (NPROLL and NOISEFLIP)
# with all 3 seeds. Output CSVs are written to each models/<name>/results/
# folder following the dataset-name convention used by previously committed
# files (e.g. results-cd-NPROLL-latent[...]-window[...].csv), one file per
# (variant, dataset) combination with one row per (seed, latent, window).
#
# Usage:
#   ./scripts/run_all_synth.sh                # run everything
#   ./scripts/run_all_synth.sh --datasets nproll
#   ./scripts/run_all_synth.sh --runners run_catch.sh,run_ae_cd.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASETS=(nproll noiseflip)
RUNNERS=(
    run_ae_cd.sh
    run_ae_ci.sh
    run_linearae_cd.sh
    run_linearae_ci.sh
    run_crossad_cd.sh
    run_crossad_ci.sh
    run_catch.sh
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) IFS=',' read -r -a DATASETS <<< "$2"; shift 2 ;;
        --runners)  IFS=',' read -r -a RUNNERS  <<< "$2"; shift 2 ;;
        -h|--help)
            sed -n '2,11p' "$0"
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "Datasets: ${DATASETS[*]}"
echo "Runners : ${RUNNERS[*]}"
echo

for dataset in "${DATASETS[@]}"; do
    for runner in "${RUNNERS[@]}"; do
        runner_path="$ROOT/scripts/$runner"
        if [[ ! -x "$runner_path" ]]; then
            echo "Skipping missing/non-executable runner: $runner_path" >&2
            continue
        fi
        echo "======================================================================"
        echo "  ${runner}  --multi-seed --dataset ${dataset}"
        echo "======================================================================"
        "$runner_path" --multi-seed --dataset "$dataset"
    done
done

echo
echo "All runs complete."
