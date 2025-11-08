#!/bin/bash -l
#SBATCH --job-name=thesis-run-all-nodes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --export=ALL

set -euo pipefail

echo "Node: $(hostname)"

# modules and proxy
module purge
module load proxy/default || { echo "proxy/default load failed"; module avail |& grep -i proxy || true; exit 1; }

: "${MAMBA_ROOT_PREFIX:=../.mamba}"
export MAMBA_ROOT_PREFIX

# conda/mamba env (same pattern as GPU sweeps)
eval "$(../tools/bin/micromamba shell hook -s bash)"
micromamba activate ../.venv

# thread throttles (tune as needed)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

PYBIN="$(command -v python)"

# quick diagnostics
$PYBIN -V
$PYBIN -c 'import sys; print(sys.executable)'

# run deterministic objectives with default CLI args
srun -u "$PYBIN" deterministic/run_all_nodes.py
