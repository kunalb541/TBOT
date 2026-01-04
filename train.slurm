#!/bin/bash
#SBATCH -p gpu_a100_short
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=94000mb
#SBATCH --cpus-per-gpu=12
#SBATCH -t 00:30:00
#SBATCH -o logs/crypto_train_%j.out
#SBATCH -e logs/crypto_train_%j.err
#SBATCH --job-name=crypto_bot
#SBATCH --exclusive
#SBATCH --signal=B:USR1@120

# ============================================================================
# SLURM Job Script for Crypto Trading Bot Training on bw3.0 HPC
# ============================================================================
#
# Usage:
#   Single node (4 GPUs):   sbatch --nodes=1 train.slurm
#   Multi node (8 GPUs):    sbatch --nodes=2 train.slurm
#   Max scale (48 GPUs):    sbatch --nodes=12 train.slurm
#
# Partition: gpu_a100_short
#   - GPU nodes with Ice Lake + NVIDIA A100 x4
#   - mem-per-gpu=94000mb, cpus-per-gpu=12
#   - Max time=30min, Max nodes=12
# ============================================================================

set -euo pipefail

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "Start time: $(date)"
echo "============================================"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crypto_bot 2>/dev/null || {
    echo "Creating crypto_bot environment..."
    conda create -n crypto_bot python=3.10 -y
    conda activate crypto_bot
    pip install torch numpy pandas tqdm requests
}

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Create directories
mkdir -p logs checkpoints

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================

# Thread settings - avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Python settings
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

# NCCL settings for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# PyTorch distributed settings (use TORCH_ prefix for PyTorch 2.0+)
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Remove deprecated variables
unset NCCL_ASYNC_ERROR_HANDLING 2>/dev/null || true
unset NCCL_BLOCKING_WAIT 2>/dev/null || true

# Distributed training setup
export LOCAL_WORLD_SIZE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

echo ""
echo "Distributed Setup:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"
echo "  Total GPUs: $((SLURM_NNODES * LOCAL_WORLD_SIZE))"
echo ""

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

N_CANDLES=${N_CANDLES:-10000}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LR:-1e-4}
SYMBOL=${SYMBOL:-BTCUSDT}
USE_REAL_DATA=${USE_REAL_DATA:-true}

echo "Training Configuration:"
echo "  Candles: $N_CANDLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Symbol: $SYMBOL"
echo "  Use real data: $USE_REAL_DATA"
echo ""

# =============================================================================
# SIGNAL HANDLING
# =============================================================================

handle_preemption() {
    echo "[$(date)] Received preemption signal (USR1)"
    echo "Saving checkpoint and exiting gracefully..."
    # The Python script handles SIGTERM for graceful shutdown
    exit 0
}
trap handle_preemption USR1

cleanup() {
    echo "[$(date)] Cleaning up..."
    # Kill any remaining processes
    pkill -f "torchrun.*main.py" 2>/dev/null || true
}
trap cleanup EXIT

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "============================================"
echo "Starting Training"
echo "============================================"

if [ "$SLURM_NNODES" -gt 1 ]; then
    echo "Multi-node distributed training with $SLURM_NNODES nodes..."
    
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc_per_node=$LOCAL_WORLD_SIZE \
            --rdzv_id="crypto-${SLURM_JOB_ID}" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            main.py train \
                --candles $N_CANDLES \
                --symbol $SYMBOL
else
    echo "Single-node training with 4 GPUs..."
    
    torchrun \
        --standalone \
        --nproc_per_node=$LOCAL_WORLD_SIZE \
        main.py train \
            --candles $N_CANDLES \
            --symbol $SYMBOL
fi

TRAIN_EXIT_CODE=$?

# =============================================================================
# POST-TRAINING
# =============================================================================

echo ""
echo "============================================"
echo "Training Complete"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo "============================================"

# Run backtest if training succeeded
if [ $TRAIN_EXIT_CODE -eq 0 ] && [ -f "checkpoints/final_model.pt" ]; then
    echo ""
    echo "Running backtest..."
    python main.py backtest --checkpoint checkpoints/final_model.pt
fi

exit $TRAIN_EXIT_CODE
