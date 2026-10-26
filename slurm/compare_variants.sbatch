#!/bin/bash
#SBATCH --job-name=y3imw_compare
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/logs/compare_%x_%j.out
#SBATCH --error=slurm/logs/compare_%x_%j.out
# Full 100-mock variant comparison for one sample, both lensmaps.
#   sbatch --dependency=afterany:<array job> slurm/compare_variants.sbatch photo
#   sbatch --dependency=afterany:<array job> slurm/compare_variants.sbatch spec
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
SAMPLE=$1
if [ "$SAMPLE" = photo ]; then
    python -u compare_variants.py --sample photo --lensmap act_dr6 --tracers BGS LRG1 LRG2 LRG3
    python -u compare_variants.py --sample photo --lensmap planck_pr4 --tracers BGS LRG1 LRG2 LRG3
else
    python -u compare_variants.py --sample spec --lensmap act_dr6 --tracers BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2
    python -u compare_variants.py --sample spec --lensmap planck_pr4 --tracers LRG1 LRG2 LRG3 QSO
fi
