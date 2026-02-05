#!/bin/bash -l
#SBATCH --job-name=thesis-combo-cpu-test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=400G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

set -euo pipefail

echo "Node: $(hostname)"

# --- initialize module system (handled by -l), then load proxy BEFORE running python
module purge
module load proxy/default || { echo "proxy/default load failed"; module avail |& grep -i proxy || true; exit 1; }
module load gurobi/9.0.1 || { echo "gurobi load failed"; module avail |& grep -i proxy || true; exit 1; }


: "${MAMBA_ROOT_PREFIX:=../.mamba}"
export MAMBA_ROOT_PREFIX
# (optional) if you hit SSL cert errors, uncomment the next line and re-run
# export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

# conda env
eval "$(../tools/bin/micromamba shell hook -s bash)"
micromamba activate ../.venv

# Ensure relative paths resolve from the repo root
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# thread throttles
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# experiment knobs (can override with --export)
export PNODE_ID=${PNODE_ID:-2156113094}
export TORCH_MATMUL_PRECISION=tf32
export TORCH_PRECISION=bf16-mixed
export NUM_EVALS=${NUM_EVALS:-5}
export HF_HORIZON=${HF_HORIZON:-6}
export MIN_HF=${MIN_HF:-1}
export MAX_HF=${MAX_HF:-10}
export MIN_LF=${MIN_LF:-1}
export MAX_LF=${MAX_LF:-10}
export TEST_SIZE=${TEST_SIZE:-None}
export SEED=${SEED:-None}
export MAX_WORKERS=${MAX_WORKERS:-6}
export RUN_NAME=${RUN_NAME:-None}
export DEV=0

PYBIN="$(command -v python)"

# quick diagnostics (these lines appear in logs/%x_%j.out)
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
$PYBIN -c "import sys,torch; print(sys.executable); print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

# launch job: test best HF/LF model combinations
# (env-driven; avoids stdin launch so multiprocessing spawn can re-import __main__)
srun -u "$PYBIN" -m forecasting.test_best_model_combinations