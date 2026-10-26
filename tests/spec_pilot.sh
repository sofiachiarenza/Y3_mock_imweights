#!/bin/bash
# Spec pilot: both variants on N_SIMS mocks for every (tracer, lensmap) in
# config.yaml (timing + sanity), then the PNG_DR2 QSO comparison.
# Writes to outputs/pilot. Usage (compute node): bash tests/spec_pilot.sh [N_SIMS=2]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-2}
PY=python   # env from slurm/_env.sh
run() {  # tracer lensmap first_sim
    local start=$(date +%s)
    $PY -u regress_spec_mocks.py --tracer $1 --lensmap $2 --sim-start $3 --sim-stop $(( $3 + N - 1 )) --output-root outputs/pilot
    echo "[$1 $2] $N sims in $(( $(date +%s) - start ))s"
}
for T in BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2; do run $T act_dr6 1; done
for T in LRG1 LRG2 LRG3 QSO; do run $T planck_pr4 260; done
$PY -u tests/test_spec_vs_png_dr2_qso.py --sidecar-root outputs/pilot --sims $(seq 260 $(( 259 + N )))
