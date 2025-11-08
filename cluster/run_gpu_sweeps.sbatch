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

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# W&B auth
# export WANDB_API_KEY=...

mkdir -p logs

# Choose models that benefit from GPU
PNODES=${PNODES:-2156113094}
MODELS=${MODELS:-"ALL"}
RUNS_PER_MODEL=${RUNS_PER_MODEL:-5}
MAX_PROC=${MAX_PROC:-1}  # 1 process per GPU is safest
SUBSET=${SUBSET:-0.5}

# Ensure Torch uses the allocated GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_ACCELERATOR=gpu
export TORCH_DEVICES=1

python -m forecasting.cluster_runner \
  --pnode "$PNODES" \
  --models "$MODELS" \
  --runs-per-model "$RUNS_PER_MODEL" \
  --max-proc "$MAX_PROC" \
  --subset-data-size "$SUBSET" \