# Environment for SysNet (as in Y3_AMR/act_scripts/sub_sysnet.sh): cosmodesi main + amry3_env,
# Sofia's LSS checkout for LSS.common_tools / LSS.imaging. Source from the repo root.
case $- in *u*) _y3imw_u=1; set +u ;; *) _y3imw_u=0 ;; esac   # conda/module scripts are not set -u clean
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module load conda
conda activate amry3_env
export PYTHONPATH=/global/cfs/cdirs/desicollab/users/schiarenza/LSS/py:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
if [ "$_y3imw_u" = 1 ]; then set -u; fi; unset _y3imw_u
