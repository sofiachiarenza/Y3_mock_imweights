# Environment for the linear-regression / C_ell scripts on Perlmutter (source from the repo root).
# c3y3_regr has jax, iminuit, regressis, healpy, h5py, fitsio; the upstream DESI_Y3_x_CMB repo
# (real-catalog loaders) goes on PYTHONPATH. The Narval env/ venv does not work here.
case $- in *u*) _y3imw_u=1; set +u ;; *) _y3imw_u=0 ;; esac   # conda/module scripts are not set -u clean
module load conda
conda activate c3y3_regr
export PYTHONPATH=/pscratch/sd/s/schiaren/DESI_Y3_x_CMB_upstream:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=FALSE
export MPICH_GPU_SUPPORT_ENABLED=0           # GPU nodes: otherwise importing mpi4py aborts (GTL not linked)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # jax on a GPU node: don't grab the GPU memory
export JAX_PLATFORMS=cpu                     # the fits are tiny; keep jax off the GPUs
if [ "$_y3imw_u" = 1 ]; then set -u; fi; unset _y3imw_u
