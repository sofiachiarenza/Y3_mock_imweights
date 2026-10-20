#!/bin/bash
#SBATCH --job-name=y3imw_photo_pilot
#SBATCH --account=rrg-wperciva
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm/logs/photo_pilot_%j.out
#SBATCH --error=slurm/logs/photo_pilot_%j.out
# Photo pilot (all variants, 5 act_dr6 mocks per tracer + variant comparison).
# Submit from the repo root:  sbatch slurm/photo_pilot.sbatch [N_SIMS]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash tests/photo_pilot.sh "${1:-5}"
