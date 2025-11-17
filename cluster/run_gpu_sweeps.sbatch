#!/bin/bash -l
#SBATCH --job-name=thesis-gpu-sweeps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
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

# thread throttles
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# experiment knobs (can override with --export)
export PNODES=${PNODES:-2156113094}
export MODELS=${MODELS:-ALL}
export RUNS_PER_MODEL=${RUNS_PER_MODEL:-50}
export MAX_PROC=${MAX_PROC:-1}
export SUBSET=${SUBSET:-1.0}
export TORCH_MATMUL_PRECISION=tf32
export TORCH_PRECISION=bf16-mixed

PYBIN="$(command -v python)"

# quick diagnostics (these lines appear in logs/%x_%j.out)
nvidia-smi || true
$PYBIN -c "import sys,torch; print(sys.executable); print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

# launch job
srun -u "$PYBIN" -m forecasting.cluster_runner \
  --pnode "$PNODES" \
  --models "$MODELS" \
  --runs-per-model "$RUNS_PER_MODEL" \
  --max-proc "$MAX_PROC" \
  --subset-data-size "$SUBSET" \
  --use-gpus