#!/bin/bash
# Build the Python env for Y3_mock_imweights (Narval / Compute Canada wheels).
# Elsewhere: drop the module line and use any Python >= 3.11.
# Usage: bash create_env.sh            (creates ./env next to this script)
set -euo pipefail
cd "$(dirname "$0")"
module load python/3.11 2>/dev/null || true
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
pip install -c constraints.txt git+https://github.com/echaussidon/regressis@a751bc1a1f6efa66b920f5225d6fdf38f75438bc
python -c "import jax, iminuit, healpy, h5py, regressis; print('env OK: jax', jax.__version__, 'iminuit', iminuit.__version__)"
