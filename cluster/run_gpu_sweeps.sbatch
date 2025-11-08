#!/bin/bash

#SBATCH --job-name=thesis-gpu-sweeps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
source /scratch/network/$USER/miniforge/etc/profile.d/conda.sh
conda activate /scratch/network/$USER/envs/gpu311

# Thread throttles (reasonable defaults for single-GPU math libs)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# Optional: Weights & Biases
# export WANDB_API_KEY=...

# Experiment knobs (safe defaults; can override with sbatch --export or env)
export PNODES=${PNODES:-2156113094}
export MODELS=${MODELS:-ALL}
export RUNS_PER_MODEL=${RUNS_PER_MODEL:-5}
export MAX_PROC=${MAX_PROC:-1}
export SUBSET=${SUBSET:-0.5}

# Diagnostics (nice to keep)
nvidia-smi
python -c "import torch; print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

# Launch under SLURM
srun -u python -m forecasting.cluster_runner \
  --pnode "$PNODES" \
  --models "$MODELS" \
  --runs-per-model "$RUNS_PER_MODEL" \
  --max-proc "$MAX_PROC" \
  --subset-data-size "$SUBSET"
