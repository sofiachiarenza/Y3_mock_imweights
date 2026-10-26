#!/bin/bash
#SBATCH --job-name=y3imw_spec_pilot
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/logs/spec_pilot_%j.out
#SBATCH --error=slurm/logs/spec_pilot_%j.out
# Spec pilot (both variants, N mocks per tracer x lensmap + PNG_DR2 QSO comparison).
# Submit from the repo root:  sbatch slurm/spec_pilot.sbatch [N_SIMS]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
bash tests/spec_pilot.sh "${1:-2}"
