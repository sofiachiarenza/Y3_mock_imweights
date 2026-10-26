#!/bin/bash
#SBATCH --job-name=y3imw_elg_compare
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm/logs/elg_compare_%j.out
#SBATCH --error=slurm/logs/elg_compare_%j.out
# ELG SysNet mode-removal check: build the ELG mock randoms cache (once), then
# <Cgg(WEIGHT_SN_MOCK)>/<Cgg(unweighted)> over the mocks that have the sidecar column.
# Submit from the repo root:  sbatch --dependency=afterany:<sysnet array> slurm/elg_compare.sbatch [SIM_STOP_ACT SIM_STOP_PLANCK]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env.sh
[ -f outputs/randoms_cache/spec_ELG.fits ] || python -u build_randoms_cache.py --sample spec --tracer ELG
python -u compare_variants.py --sample spec --lensmap act_dr6 --tracers ELG --sim-stop "${1:-100}"
python -u compare_variants.py --sample spec --lensmap planck_pr4 --tracers ELG --sim-stop "${2:-359}"
