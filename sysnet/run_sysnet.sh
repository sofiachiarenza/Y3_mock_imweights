#!/bin/bash
# SysNet training for one ELG mock, N and S (port of Y3_AMR/act_scripts/run_sysnet.sh).
# Same hyperparameters: N lr 0.009 bs 256 nn 3x10; S lr 0.007 bs 1024 nn 4x20; 5 chains, 100 epochs,
# -k, pnll/dnnp, snapshot ensemble (sysnet_app_snapshot_mpi.py, T_0=5 -> 20 snapshots/chain).
# Fixes: ${tracer} / ${srun_flags} were undefined (paths happened to work); srun -t no longer
# exceeds the job limit; runs inside a packed allocation (srun --exact on a core slice).
#   usage: sysnet/run_sysnet.sh MOCKDIR [NTASKS]   (MOCKDIR holds prep_1.1_1.6_{N,S}.fits)
# SYSNET_SRUN_EXTRA is appended to the training srun, e.g. "--gpus=1" to give the step one GPU
# (sysnet uses cuda:0 when torch sees one; all ranks of the step share it). Unset -> CPU.
set -euo pipefail
mockdir=$1
ntasks=${2:-25}
here=$(cd "$(dirname "$0")" && pwd)

export OMP_NUM_THREADS=1
# CPU mode: hide the GPUs from torch inside each srun step (srun resets CUDA_VISIBLE_DEVICES, so an
# export here is not enough; the pilot's "CPU" run trained on cuda:0)
py="python"
if [ -z "${SYSNET_SRUN_EXTRA:-}" ]; then py="env CUDA_VISIBLE_DEVICES= python"; fi
sysnet_app=/global/cfs/cdirs/desicollab/users/schiarenza/LSS/py/LSS/imaging/sysnet_app_mpi.py
sysnet_app_snapshot=$here/sysnet_app_snapshot_mpi.py

regions=("N" "S")
zlims=(1.1 1.6)     # each pair is a redshift range
LR=(0.009 0.007)    # learning rate
NBATCH=(256 1024)   # batchsize, powers of 2
NNS=(3 10 4 20)     # each pair is a NN structure (# layers, # units)
NCHAIN=5            # number of chains
NEPOCH=100          # number of epochs

lrfinder_flags="-ax all --model dnnp --loss pnll -fl"
train_flags="-ax all --model dnnp --loss pnll --eta_min 0.00001 -k --snapshot_ensemble"

for ((i=0; i<${#regions[@]}; i++)); do
    NNlayers="${NNS[$((i*2))]}"
    NNunits="${NNS[$((i*2+1))]}"
    for ((j=0; j<$(( ${#zlims[@]} / 2 )); j++)); do
        zmin="${zlims[$((j*2))]}"
        zmax="${zlims[$((j*2 + 1))]}"
        region_flags="-lr ${LR[$i]} -bs ${NBATCH[$i]} --nn_structure $NNlayers $NNunits -ne $NEPOCH -nc $NCHAIN -i $mockdir/prep_${zmin}_${zmax}_${regions[$i]}.fits -o $mockdir/${zmin}_${zmax}_${regions[$i]}"
        echo "$region_flags"
        # learning-rate finder: informational only, it always ends with "loss diverged" (exit 1)
        srun --exact -n 1 -c 1 -t 5 $py $sysnet_app $lrfinder_flags $region_flags || true
        # training (snapshot ensemble)
        srun --exact -n "$ntasks" -c 1 -t 30 ${SYSNET_SRUN_EXTRA:-} $py $sysnet_app_snapshot $train_flags $region_flags
    done
done
