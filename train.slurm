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
#SBATCH --mail-user=kunal29bhatia@gmail.com
#SBATCH --mail-type=ALL

# ============================================================================
# SLURM Job Script for Crypto Trading Bot Training on bw3.0 HPC
# ============================================================================
#
# Usage:
#   Single node (4 GPUs):   sbatch --nodes=1 train_hpc.slurm
#   Multi node (8 GPUs):    sbatch --nodes=2 train_hpc.slurm
#   Max scale (48 GPUs):    sbatch --nodes=12 train_hpc.slurm
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bot
# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Create directories
mkdir -p logs checkpoints

# =============================================================================
# CONFIGURATION
# =============================================================================

# Training hyperparameters (can be overridden with env vars)
N_CANDLES=${N_CANDLES:-10000}
BATCH_SIZE=${BATCH_SIZE:-64}
EPOCHS=${EPOCHS:-100}
LEARNING_RATE=${LR:-0.0001}
SYMBOL=${SYMBOL:-BTCUSDT}
USE_REAL_DATA=${USE_REAL_DATA:-true}

# Model architecture
D_MODEL=${D_MODEL:-128}
N_LAYERS=${N_LAYERS:-4}
DROPOUT=${DROPOUT:-0.1}
NUM_HEADS=${NUM_HEADS:-8}

# Progress tracking
PROGRESS_FILE="checkpoints/.train_progress"
EXPERIMENT_FILE="checkpoints/.current_experiment"

echo ""
echo "Training Configuration:"
echo "  Candles: $N_CANDLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Symbol: $SYMBOL"
echo "  Use real data: $USE_REAL_DATA"
echo "  D_model: $D_MODEL"
echo "  Layers: $N_LAYERS"
echo ""

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
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

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
unset PYTORCH_CUDA_ALLOC_CONF 2>/dev/null || true

# Distributed training setup
export LOCAL_WORLD_SIZE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}

echo "Distributed Setup:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  LOCAL_WORLD_SIZE: $LOCAL_WORLD_SIZE"
echo "  Total GPUs: $((SLURM_NNODES * LOCAL_WORLD_SIZE))"
echo ""

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

mark_stage_complete() {
    local stage="$1"
    echo "${stage}:complete:$(date +%s)" >> "${PROGRESS_FILE}"
    log_info "Stage complete: ${stage}"
}

is_stage_complete() {
    local stage="$1"
    [[ -f "${PROGRESS_FILE}" ]] && grep -q "^${stage}:complete:" "${PROGRESS_FILE}"
}

get_checkpoint_epoch() {
    local checkpoint_path="$1"
    if [[ -f "${checkpoint_path}" ]]; then
        python3 -c "
import torch
try:
    ckpt = torch.load('${checkpoint_path}', map_location='cpu', weights_only=False)
    print(int(ckpt.get('epoch', 0)))
except Exception:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# =============================================================================
# CLEANUP FUNCTIONS
# =============================================================================

cleanup_tmp() {
    log_info "Cleaning /tmp..."
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
        rm -f /tmp/crypto_cache_*.pkl 2>/dev/null || true
        echo \"Node \$(hostname): /tmp cleaned\"
    "
}
# Check for latest checkpoint (prioritize numbered checkpoints, then best)
if ls checkpoints/checkpoint_epoch_*.pt 1> /dev/null 2>&1; then
    LATEST_CHECKPOINT=$(ls -t checkpoints/checkpoint_epoch_*.pt | head -1)
elif [[ -f "checkpoints/best_model.pt" ]]; then
    LATEST_CHECKPOINT="checkpoints/best_model.pt"
else
    LATEST_CHECKPOINT=""
fi
    "
}

cleanup_all() {
    log_info "Running cleanup..."
    cleanup_tmp
    cleanup_shm
    
    # Kill any remaining Python processes
    pkill -f "torchrun.*main.py" 2>/dev/null || true
    
    log_info "Cleanup complete"
}

trap cleanup_all EXIT

# =============================================================================
# AUTO-CONTINUATION SETUP
# =============================================================================

setup_continuation() {
    log_info "Setting up continuation job..."
    local cont_job
    cont_job=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
                      --kill-on-invalid-dep=yes "$0")
    
    if [[ -n "${cont_job}" ]]; then
        echo "${cont_job}" > "checkpoints/.continuation_job_id"
        log_info "Continuation job scheduled: ${cont_job}"
    fi
}

cancel_continuation() {
    if [[ -f "checkpoints/.continuation_job_id" ]]; then
        local cont_job
        cont_job=$(cat "checkpoints/.continuation_job_id")
        scancel "${cont_job}" 2>/dev/null || true
        rm -f "checkpoints/.continuation_job_id"
        log_info "Cancelled continuation job: ${cont_job}"
    fi
}

handle_preemption() {
    log_info "Received USR1 preemption signal"
    log_info "Training will checkpoint and continuation job will resume"
    exit 0
}

trap handle_preemption USR1

# =============================================================================
# CHECK FOR RESUME
# =============================================================================

RESUME_FLAG=""
OUTPUT_DIR_FLAG=""

# Check if we have a checkpoint to resume from
LATEST_CHECKPOINT="checkpoints/checkpoint_latest.pt"
BEST_CHECKPOINT="checkpoints/best_model.pt"

if [[ -f "${LATEST_CHECKPOINT}" ]]; then
    CURRENT_EPOCH=$(get_checkpoint_epoch "${LATEST_CHECKPOINT}")
    log_info "Found checkpoint at epoch ${CURRENT_EPOCH}"
    
    if [[ "${CURRENT_EPOCH}" -ge "${EPOCHS}" ]]; then
        log_info "Training already complete (${CURRENT_EPOCH} >= ${EPOCHS})"
        log_info "Starting backtest..."
        python main.py backtest --checkpoint checkpoints/final_model.pt
        exit 0
    fi
    
    log_info "RESUMING training from epoch ${CURRENT_EPOCH}"
    RESUME_FLAG="${LATEST_CHECKPOINT}"
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

log_info "============================================"
log_info "STARTING TRAINING"
log_info "============================================"

# Setup continuation for preemption
setup_continuation

# Build training command
TRAIN_CMD="main.py train --candles $N_CANDLES --symbol $SYMBOL"

if [[ "${USE_REAL_DATA}" != "true" ]]; then
    TRAIN_CMD="${TRAIN_CMD} --fake-data"
fi

# Run training
if [ "$SLURM_NNODES" -gt 1 ]; then
    log_info "Multi-node distributed training with $SLURM_NNODES nodes..."
    
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
        torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc_per_node=$LOCAL_WORLD_SIZE \
            --rdzv_id="crypto-${SLURM_JOB_ID}" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
            $TRAIN_CMD
else
    log_info "Single-node training with 4 GPUs..."
    
    torchrun \
        --standalone \
        --nproc_per_node=$LOCAL_WORLD_SIZE \
        $TRAIN_CMD
fi

TRAIN_EXIT_CODE=$?

# =============================================================================
# POST-TRAINING
# =============================================================================

log_info ""
log_info "============================================"
log_info "TRAINING PHASE COMPLETE"
log_info "Exit code: $TRAIN_EXIT_CODE"
log_info "End time: $(date)"
log_info "============================================"

# If training succeeded, mark complete and cancel continuation
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    mark_stage_complete "training"
    cancel_continuation
    
    # Run backtest if final model exists
    if [ -f "checkpoints/final_model.pt" ]; then
        log_info ""
        log_info "============================================"
        log_info "STARTING BACKTEST"
        log_info "============================================"
        
        python main.py backtest --checkpoint checkpoints/final_model.pt
        BACKTEST_EXIT_CODE=$?
        
        if [ $BACKTEST_EXIT_CODE -eq 0 ]; then
            mark_stage_complete "backtest"
            log_info "BACKTEST COMPLETE"
        else
            log_error "Backtest failed with exit code $BACKTEST_EXIT_CODE"
        fi
    fi
    
    # Print summary
    log_info ""
    log_info "============================================"
    log_info "PIPELINE COMPLETE"
    log_info "============================================"
    
    if [[ -f "checkpoints/final_model.pt" ]]; then
        log_info "Final model: checkpoints/final_model.pt"
    fi
    
    if [[ -f "backtest_results.json" ]]; then
        log_info "Backtest results: backtest_results.json"
        python3 -c "
import json
try:
    with open('backtest_results.json') as f:
        results = json.load(f)
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
except Exception as e:
    print(f"  Error reading results: {e}")
    fi
else
    log_error "Training failed with exit code $TRAIN_EXIT_CODE"
    log_info "Continuation job will retry..."
fi

exit $TRAIN_EXIT_CODE
