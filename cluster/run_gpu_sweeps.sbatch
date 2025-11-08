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

# show proxy vars so we know it took effect
echo "HTTP_PROXY=$HTTP_PROXY"
echo "HTTPS_PROXY=$HTTPS_PROXY"
echo "NO_PROXY=$NO_PROXY"

# (optional) if you hit SSL cert errors, uncomment the next line and re-run
# export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

# conda env
micromamba activate ./.venv

# thread throttles
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# experiment knobs (can override with --export)
export PNODES=${PNODES:-2156113094}
export MODELS=${MODELS:-ALL}
export RUNS_PER_MODEL=${RUNS_PER_MODEL:-5}
export MAX_PROC=${MAX_PROC:-1}
export SUBSET=${SUBSET:-0.5}

PYBIN=/scratch/network/$USER/envs/gpu311/bin/python

# quick diagnostics (these lines appear in logs/%x_%j.out)
nvidia-smi || true
$PYBIN -c "import sys,torch; print(sys.executable); print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

# verify connectivity via proxy from THIS node
curl -I --max-time 10 https://api.wandb.ai || echo "curl failed"
$PYBIN - <<'PY'
import os, requests
print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))
try:
    r = requests.get("https://api.wandb.ai", timeout=10)
    print("requests status:", r.status_code)
except Exception as e:
    print("requests error:", repr(e))
PY

# launch job
srun -u "$PYBIN" -m forecasting.cluster_runner \
  --pnode "$PNODES" \
  --models "$MODELS" \
  --runs-per-model "$RUNS_PER_MODEL" \
  --max-proc "$MAX_PROC" \
  --subset-data-size "$SUBSET"
