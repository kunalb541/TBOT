#!/bin/bash
#SBATCH --job-name=crypto_bot
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================================================
# SLURM Job Script for Crypto Trading Bot Training
# ============================================================================
# 
# Usage:
#   Single GPU:   sbatch --gres=gpu:1 train_hpc.slurm
#   Multi GPU:    sbatch --gres=gpu:4 train_hpc.slurm
#   Multi Node:   sbatch --nodes=2 --gres=gpu:4 train_hpc.slurm
#
# Adjust the partition name to match your cluster!
# Common names: gpu, a100, v100, p100, nvidia
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "============================================"

# Load required modules (adjust to your cluster)
# Common module names:
module load python/3.10 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9 2>/dev/null || true
module load nccl 2>/dev/null || true

# Alternatively, if using conda:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate crypto_bot

# Create log directory
mkdir -p logs

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))

# For better NCCL performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# PyTorch settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"

# ============================================================================
# Training Configuration
# ============================================================================
N_CANDLES=${N_CANDLES:-10000}          # Number of candles
BATCH_SIZE=${BATCH_SIZE:-64}           # Batch size per GPU
EPOCHS=${EPOCHS:-100}                   # Number of epochs
LEARNING_RATE=${LR:-1e-4}              # Learning rate
SYMBOL=${SYMBOL:-BTCUSDT}              # Trading symbol
USE_REAL_DATA=${USE_REAL_DATA:-true}   # Use real Binance data

echo ""
echo "Training Configuration:"
echo "  Candles: $N_CANDLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Symbol: $SYMBOL"
echo "  Use real data: $USE_REAL_DATA"
echo ""

# ============================================================================
# Run Training
# ============================================================================

if [ $WORLD_SIZE -gt 1 ]; then
    echo "Starting distributed training with $WORLD_SIZE GPUs..."
    
    # Use torchrun for distributed training
    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        main.py train \
        --candles $N_CANDLES \
        --symbol $SYMBOL
else
    echo "Starting single GPU training..."
    
    python main.py train \
        --candles $N_CANDLES \
        --symbol $SYMBOL
fi

# ============================================================================
# Post-training
# ============================================================================

echo ""
echo "============================================"
echo "Job finished at: $(date)"
echo "============================================"

# Copy checkpoints to a permanent location if needed
# cp -r checkpoints/ /path/to/permanent/storage/

# Optional: Run quick evaluation
# python main.py backtest
