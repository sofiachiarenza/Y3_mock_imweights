#!/bin/bash
# Photo pilot: all variants on a few act_dr6 mocks per tracer, then the
# variant comparison. Writes to outputs/pilot (not the production tree).
# Usage (compute node): bash tests/photo_pilot.sh [N_SIMS=5]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-5}
PY=python   # env from slurm/_env.sh
for T in BGS LRG1 LRG2 LRG3; do
    start=$(date +%s)
    $PY -u regress_photo_mocks.py --tracer $T --lensmap act_dr6 --sim-start 1 --sim-stop $N --output-root outputs/pilot
    echo "[$T] $N sims in $(( $(date +%s) - start ))s"
done
$PY -u compare_variants.py --sample photo --lensmap act_dr6 --tracers BGS LRG1 LRG2 LRG3 \
    --sim-start 1 --sim-stop $N --output-root outputs/pilot --tag _pilot
