#!/bin/bash
#SBATCH --job-name=y3imw_sysnet
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm/logs/sysnet_%A_%a.out
#SBATCH --error=slurm/logs/sysnet_%A_%a.out
# ELG SysNet on the new mocks (CLAUDE.md task 5): 2 lensmaps x 10 chunks of 10 sims (hdu 1),
# act_dr6 1-100 and planck_pr4 260-359. One GPU node per chunk, split into LANES parallel lanes.
# Each lane runs prepare_sysnet_mocks.py on its sims (randoms read once), then for each sim:
# run_sysnet.sh (N, S) -> add_sysnet_mocks.py (sidecar column WEIGHT_SN_MOCK).
# Sims whose sidecar already has WEIGHT_SN_MOCK are skipped, so resubmitting is safe.
# Tune from the pilot TIMING lines:
#   LANES     parallel mocks per node (default 2: 2 x 25 ranks fits the 64 cores)
#   NTASKS    MPI ranks per training (default 25 as in Y3_AMR)
#   GPU=1     train on GPUs (lane i gets --gpus=1); default CPU. Pilot 2026-10-26: GPU 20-27 min/mock with 25
#             ranks sharing one GPU, vs ~2 min/region on CPU in Y3_AMR, so stay on CPU
# Submit from the repo root:  sbatch slurm/sysnet_elg.sbatch    (or: sbatch --export=ALL,LANES=4,GPU=1 ...)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/_env_sysnet.sh
LANES=${LANES:-2}
NTASKS=${NTASKS:-25}
WORK=/pscratch/sd/s/schiaren/y3imw_sysnet
ID=$SLURM_ARRAY_TASK_ID
LENSMAPS=(act_dr6 planck_pr4)
FIRST=(1 260)                         # config.yaml: sims
L=$(( ID / 10 ))
LENS=${LENSMAPS[$L]}
START=$(( FIRST[L] + (ID % 10) * 10 ))
echo "task $ID: ELG $LENS sims $START-$(( START + 9 )), $LANES lanes, $NTASKS ranks, GPU=${GPU:-0}"

todo=()
for s in $(seq $START $(( START + 9 ))); do
    side=outputs/spec/$LENS/ELG/$(printf "mock_sim%04d_hdu1.h5" $s)
    if [ -f "$side" ] && python -c "import h5py,sys; sys.exit(0 if 'WEIGHT_SN_MOCK' in h5py.File('$side','r') else 1)"; then
        echo "sim $s: done already"; continue
    fi
    todo+=($s)
done

lane() {
    local sims=("$@")
    [ ${#sims[@]} -eq 0 ] && return 0
    python -u sysnet/prepare_sysnet_mocks.py --lensmap $LENS --sims "${sims[@]}" --workdir $WORK
    for s in "${sims[@]}"; do
        local mdir=$WORK/$LENS/ELG/$(printf "mock_sim%04d" $s)
        local t0=$(date +%s)
        if sysnet/run_sysnet.sh $mdir $NTASKS > $mdir/run_sysnet.log 2>&1 \
           && python -u sysnet/add_sysnet_mocks.py --lensmap $LENS --sim $s --workdir $WORK; then
            echo "sim $s: done in $(( $(date +%s) - t0 )) s"
        else
            echo "sim $s: FAILED, see $mdir/run_sysnet.log"
        fi
    done
}

for ((i=0; i<LANES; i++)); do
    sims=()
    for ((k=i; k<${#todo[@]}; k+=LANES)); do sims+=(${todo[$k]}); done
    (
        if [ "${GPU:-0}" = 1 ]; then export SYSNET_SRUN_EXTRA="--gpus=1"; fi
        lane "${sims[@]}"
    ) > slurm/logs/sysnet_${SLURM_ARRAY_JOB_ID}_${ID}_lane$i.log 2>&1 &
done
wait
grep -h "sim " slurm/logs/sysnet_${SLURM_ARRAY_JOB_ID}_${ID}_lane*.log | grep -E "done|FAILED"
