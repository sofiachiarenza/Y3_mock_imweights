#!/bin/bash
#SBATCH --job-name=y3imw_data
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/data_%j.out
#SBATCH --error=slurm/logs/data_%j.out
# Real-data tests 1-4 on one GPU node (charged to desi_g; the work is CPU-side):
#   while the regressions run: cib_check.py (test 4: all four Lenz maps) and dust_map_check.py (env check,
#     should reproduce the committed figures/dust_map_check.txt; see git diff)
#   in parallel: regress_photo_data.py x {BGS, LRG1-3} and regress_spec_data.py x {BGS, LRG1-3, QSO, QSO1, QSO2}
#   then: compare_data_variants.py (tests 1-2: Cgg / Cgk ratios) and euclid_data_check.py (test 3)
# 11 regression processes x 4 threads (64 cores). Peak memory per process is a few GB, below the 256 GB node.
# Submit from the repo root after the pilot passes:  sbatch slurm/data_regress.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
source slurm/_env.sh
L=slurm/logs/data_${SLURM_JOB_ID}
mkdir -p figures/data_variants

python -u cib_check.py > ${L}_cib_check.log 2>&1 &
python -u dust_map_check.py > ${L}_dust_map_check.log 2>&1 &

pids=()
for T in BGS LRG1 LRG2 LRG3; do
    python -u regress_photo_data.py --tracer $T > ${L}_photo_$T.log 2>&1 & pids+=($!)
done
for T in BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2; do
    python -u regress_spec_data.py --tracer $T > ${L}_spec_$T.log 2>&1 & pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=$((fail + 1)); done
echo "regressions done, $fail failed"
wait   # cib_check writes the CIB caches that euclid_data_check reads

python -u compare_data_variants.py --sample photo --tracers BGS LRG1 LRG2 LRG3 > ${L}_compare_photo.log 2>&1 &
python -u compare_data_variants.py --sample spec --tracers BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2 > ${L}_compare_spec.log 2>&1 &
python -u euclid_data_check.py --sample photo --tracers BGS LRG1 LRG2 LRG3 > ${L}_euclid_photo.log 2>&1 &
python -u euclid_data_check.py --sample spec --tracers BGS LRG1 LRG2 LRG3 QSO QSO1 QSO2 > ${L}_euclid_spec.log 2>&1 &
wait
grep -h "refit WEIGHT_IMLIN vs official" ${L}_spec_*.log
grep -lE "Traceback|Error" ${L}_*.log && echo "^ logs with errors"
echo "data job done"
