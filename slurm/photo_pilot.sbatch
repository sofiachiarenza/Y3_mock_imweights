#!/bin/bash
#SBATCH --job-name=y3imw_photo_pilot
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/logs/photo_pilot_%j.out
#SBATCH --error=slurm/logs/photo_pilot_%j.out
# Photo pilot (all variants, 5 act_dr6 mocks per tracer + variant comparison).
# Submit from the repo root:  sbatch slurm/photo_pilot.sbatch [N_SIMS]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
bash tests/photo_pilot.sh "${1:-5}"
