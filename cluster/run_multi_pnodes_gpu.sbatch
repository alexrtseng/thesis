#!/bin/bash

#SBATCH --job-name=thesis-multi-pnodes-gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load anaconda3/2023.3 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.2 2>/dev/null || true

# source activate thesis

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}

# W&B API key assumed exported prior to sbatch submission

cd /Users/alextseng/MacDocuments/GitHub/thesis || exit 1
mkdir -p logs

# Comma-separated list of PNODES to sweep concurrently per model
PNODES=${PNODES:-"2156113094,2156114000"}
MODELS=${MODELS:-"RNNModel,TCNModel,TransformerModel"}
RUNS_PER_MODEL=${RUNS_PER_MODEL:-5}
MAX_PROC=${MAX_PROC:-2}  # match number of GPUs generally
SUBSET=${SUBSET:-1.0}

# Two GPUs available; allow cluster_runner to round-robin
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TORCH_ACCELERATOR=gpu
export TORCH_DEVICES=1

python -m forecasting.cluster_runner \
  --pnode "$PNODES" \
  --models "$MODELS" \
  --runs-per-model "$RUNS_PER_MODEL" \
  --max-proc "$MAX_PROC" \
  --subset-data-size "$SUBSET" \
  --use-gpus
