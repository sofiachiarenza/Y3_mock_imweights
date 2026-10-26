#!/bin/bash
#SBATCH --job-name=y3imw_pilot
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/pilot_%j.out
#SBATCH --error=slurm/logs/pilot_%j.out
# Pilot on one GPU debug node, everything at once:
#  1. regress_photo_data.py --mock: should reproduce outputs/photo/act_dr6/LRG1/mock_sim0001 exactly
#  2. regress_spec_data.py  --mock: should reproduce outputs/spec/act_dr6/LRG1/mock_sim0001 to < 1e-5
#  3. ELG act_dr6 sim 1: prep, then SysNet training twice from the same prep (CPU, and
#     GPU with --gpus=1) for timing, then add_sysnet_mocks.py on the CPU run -> sidecar
# Submit from the repo root:  sbatch slurm/pilot_gpu.sbatch
# Check: grep -E "VALIDATION|TIMING|WEIGHT_SN_MOCK|device" slurm/logs/pilot_<jobid>.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
WORK=/pscratch/sd/s/schiaren/y3imw_sysnet
SIM=1

( source slurm/_env.sh; python -u regress_photo_data.py --tracer LRG1 --mock act_dr6 $SIM ) > slurm/logs/pilot_${SLURM_JOB_ID}_photo.log 2>&1 &
( source slurm/_env.sh; python -u regress_spec_data.py --tracer LRG1 --mock act_dr6 $SIM ) > slurm/logs/pilot_${SLURM_JOB_ID}_spec.log 2>&1 &

(
    source slurm/_env_sysnet.sh
    python -u sysnet/prepare_sysnet_mocks.py --lensmap act_dr6 --sims $SIM --workdir $WORK
    CPU=$WORK/act_dr6/ELG/mock_sim0001
    GPU=$WORK/pilot_gpu/act_dr6/ELG/mock_sim0001
    mkdir -p $GPU && cp $CPU/prep_*.fits $GPU/
    ( t0=$(date +%s); sysnet/run_sysnet.sh $CPU 25 > slurm/logs/pilot_${SLURM_JOB_ID}_sysnet_cpu.log 2>&1; ec=$?
      echo "TIMING sysnet CPU (25 ranks): $(( $(date +%s) - t0 )) s, exit $ec" ) &
    ( t0=$(date +%s); SYSNET_SRUN_EXTRA="--gpus=1" sysnet/run_sysnet.sh $GPU 25 > slurm/logs/pilot_${SLURM_JOB_ID}_sysnet_gpu.log 2>&1; ec=$?
      echo "TIMING sysnet GPU (25 ranks, 1 GPU): $(( $(date +%s) - t0 )) s, exit $ec" ) &
    wait
    grep -h "device" slurm/logs/pilot_${SLURM_JOB_ID}_sysnet_*.log $CPU/*/train.log $GPU/*/train.log 2>/dev/null | sort | uniq -c
    python -u sysnet/add_sysnet_mocks.py --lensmap act_dr6 --sim $SIM --workdir $WORK
    python - <<EOF
import fitsio, numpy as np
for reg in ("N", "S"):
    h = fitsio.read("$CPU/1.1_1.6_%s/nn-weights.fits" % reg)
    print("hpix", reg, h["hpix"].min(), h["hpix"].max(), "weight shape", h["weight"].shape)
EOF
) &

wait
grep -h "VALIDATION\|max|dw|" slurm/logs/pilot_${SLURM_JOB_ID}_photo.log slurm/logs/pilot_${SLURM_JOB_ID}_spec.log
echo "pilot done"
