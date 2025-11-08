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

# Always start from the directory where you ran `sbatch`
cd "${SLURM_SUBMIT_DIR:?}"
echo "Node: $(hostname)"
echo "CWD : $(pwd)"

module purge
module load proxy/default || { echo "proxy/default load failed"; module avail |& grep -i proxy || true; exit 1; }

# --- micromamba paths (absolute, not ../)
MMBIN="$PWD/tools/bin/micromamba"
ENV_PATH="$PWD/.venv"

# Define this BEFORE the shell hook (needed under `set -u`)
: "${MAMBA_ROOT_PREFIX:=$PWD/.mamba}"
export MAMBA_ROOT_PREFIX

# Bootstrap micromamba if missing
if [[ ! -x "$MMBIN" ]]; then
  mkdir -p "$PWD/tools"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$PWD/tools" bin/micromamba
fi

# Enable hook & activate env
eval "$("$MMBIN" shell hook -s bash)"
micromamba activate "$ENV_PATH"

# Threads
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# Experiment knobs
export PNODES=${PNODES:-2156113094}
export MODELS=${MODELS:-ALL}
export RUNS_PER_MODEL=${RUNS_PER_MODEL:-5}
export MAX_PROC=${MAX_PROC:-1}
export SUBSET=${SUBSET:-0.5}

PYBIN="$(command -v python)"

# Diagnostics
nvidia-smi || true
"$PYBIN" -c "import sys, torch; print(sys.executable); print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

# Launch
srun -u "$PYBIN" -m forecasting.cluster_runner \
  --pnode "$PNODES" \
  --models "$MODELS" \
  --runs-per-model "$RUNS_PER_MODEL" \
  --max-proc "$MAX_PROC" \
  --subset-data-size "$SUBSET"
