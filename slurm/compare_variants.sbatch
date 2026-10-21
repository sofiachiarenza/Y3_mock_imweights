#!/bin/bash
#SBATCH --job-name=y3imw_compare
#SBATCH --account=rrg-wperciva
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/compare_%x_%j.out
#SBATCH --error=slurm/logs/compare_%x_%j.out
# Full 100-mock variant comparison for one sample, both lensmaps.
#   sbatch --dependency=afterany:<array job> slurm/compare_variants.sbatch photo
#   sbatch --dependency=afterany:<array job> slurm/compare_variants.sbatch spec
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SAMPLE=$1
if [ "$SAMPLE" = photo ]; then
    env/bin/python -u compare_variants.py --sample photo --lensmap act_dr6 --tracers BGS LRG1 LRG2 LRG3
    env/bin/python -u compare_variants.py --sample photo --lensmap planck_pr4 --tracers BGS LRG1 LRG2 LRG3
else
    env/bin/python -u compare_variants.py --sample spec --lensmap act_dr6 --tracers BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2
    env/bin/python -u compare_variants.py --sample spec --lensmap planck_pr4 --tracers LRG1 LRG2 LRG3 QSO
fi
