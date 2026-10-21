#!/bin/bash
#SBATCH --job-name=y3imw_photo
#SBATCH --account=rrg-wperciva
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-79
#SBATCH --output=slurm/logs/photo_%A_%a.out
#SBATCH --error=slurm/logs/photo_%A_%a.out
# Photo production: 4 tracers x 2 lensmaps x 10 chunks of 10 sims (hdu 1).
# Pilot timing (all variants): BGS ~11 min/mock, LRG 3-6 min/mock -> <= ~2 h per task.
# Submit from the repo root:  sbatch slurm/regress_photo.sbatch
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
TRACERS=(BGS LRG1 LRG2 LRG3)
LENSMAPS=(act_dr6 planck_pr4)
FIRST=(1 260)                         # first sim per lensmap (config.yaml: sims)
ID=$SLURM_ARRAY_TASK_ID
T=${TRACERS[$(( ID / 20 ))]}
L=$(( (ID / 10) % 2 ))
CHUNK=$(( ID % 10 ))
START=$(( FIRST[L] + CHUNK * 10 ))
STOP=$(( START + 9 ))
echo "task $ID: $T ${LENSMAPS[$L]} sims $START-$STOP"
env/bin/python -u regress_photo_mocks.py --tracer "$T" --lensmap "${LENSMAPS[$L]}" \
    --sim-start "$START" --sim-stop "$STOP" --skip-existing
