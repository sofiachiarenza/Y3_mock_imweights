#!/bin/bash
#SBATCH --job-name=y3imw_spec
#SBATCH --account=rrg-wperciva
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-43
#SBATCH --output=slurm/logs/spec_%A_%a.out
#SBATCH --error=slurm/logs/spec_%A_%a.out
# Spec production: 11 (tracer, lensmap) combos x 4 chunks of 25 sims (hdu 1).
# Pilot timing (both variants): 45-133 s/mock, 11 GB peak -> <= ~1 h per task.
# Submit from the repo root:  sbatch slurm/regress_spec.sbatch [--variants V1 V2 ...]
# (--skip-existing appends only the variants missing from existing sidecars)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
COMBOS=(BGS:act_dr6 LRG1:act_dr6 LRG2:act_dr6 LRG3:act_dr6 QSO:act_dr6 QSO1:act_dr6 QSO2:act_dr6
        LRG1:planck_pr4 LRG2:planck_pr4 LRG3:planck_pr4 QSO:planck_pr4)
ID=$SLURM_ARRAY_TASK_ID
C=${COMBOS[$(( ID / 4 ))]}
T=${C%%:*}; L=${C##*:}
FIRST=$([ "$L" = act_dr6 ] && echo 1 || echo 260)   # config.yaml: sims
START=$(( FIRST + (ID % 4) * 25 ))
STOP=$(( START + 24 ))
echo "task $ID: $T $L sims $START-$STOP"
env/bin/python -u regress_spec_mocks.py --tracer "$T" --lensmap "$L" \
    --sim-start "$START" --sim-stop "$STOP" --skip-existing "$@"
