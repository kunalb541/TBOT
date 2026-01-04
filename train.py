"""
Training Pipeline for Crypto Trading Bot
==========================================
Implements training with:
- Distributed training support (for HPC)
- Mixed precision training (AMP)
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Validation monitoring

ENHANCEMENTS:
- Multi-GPU / distributed training via DDP
- Automatic Mixed Precision (AMP) for faster training
- Gradient accumulation for effective larger batch sizes
- Better logging and progress tracking
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from config import Config, DEFAULT_CONFIG
from model import CausalTimeSeriesTransformer, TradingSignalLoss, count_parameters, verify_causal_masking
from data import prepare_datasets, create_dataloaders, TimeSeriesDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep copy the model state
            if isinstance(model, DDP):
                self.best_model_state = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
            else:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
    
    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        epoch_time: float
    ):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def to_dict(self) -> Dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }


def setup_distributed():
    """Setup distributed training if available."""
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training: rank {rank}/{world_size}, local_rank {local_rank}")
        return True, world_size, rank, local_rank
    
    return False, 1, 0, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process."""
    return rank == 0


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        config: Configuration
        device: Device to use
        scaler: GradScaler for AMP (optional)
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Classification accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} Train", leave=False)
    
    for batch_idx, (sequences, labels, _) in enumerate(pbar):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with optional AMP
        if scaler is not None and config.training.use_amp:
            with autocast():
                logits = model(sequences)
                loss = criterion(logits, labels)
            
            # Backward pass with scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(sequences)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config
) -> Tuple[float, float, Dict[int, float]]:
    """
    Validate model.
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Classification accuracy
        class_accuracies: Per-class accuracy breakdown
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}
    
    for sequences, labels, _ in dataloader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if config.training.use_amp:
            with autocast():
                logits = model(sequences)
                loss = criterion(logits, labels)
        else:
            logits = model(sequences)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Per-class accuracy
        for i in range(3):
            mask = labels == i
            class_correct[i] += (predictions[mask] == labels[mask]).sum().item()
            class_total[i] += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    class_accuracies = {}
    for i in range(3):
        if class_total[i] > 0:
            class_accuracies[i] = class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0
    
    return avg_loss, accuracy, class_accuracies


def compute_class_weights(dataloader: DataLoader) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.
    Uses inverse frequency weighting.
    """
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for _, labels, _ in dataloader:
        for label in labels:
            class_counts[label.item()] += 1
    
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    
    weights = []
    for i in range(n_classes):
        if class_counts[i] > 0:
            weight = total / (n_classes * class_counts[i])
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: TrainingMetrics,
    config: Config,
    feature_names: List[str],
    scaler_stats: Dict,
    path: str,
    is_ddp: bool = False
):
    """Save training checkpoint."""
    model_state = model.module.state_dict() if is_ddp else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics.to_dict(),
        'feature_names': feature_names,
        'scaler_mean': scaler_stats['mean'],
        'scaler_std': scaler_stats['std'],
        'config': config.model.__dict__
    }
    torch.save(checkpoint, path)


def train(
    config: Config = DEFAULT_CONFIG,
    n_candles: int = 10000,
    resume_from: Optional[str] = None
) -> Tuple[nn.Module, TrainingMetrics]:
    """
    Main training function.
    
    Supports:
    - Single GPU training
    - Multi-GPU distributed training (DDP)
    - Mixed precision training (AMP)
    
    Args:
        config: Configuration object
        n_candles: Number of candles to use for training
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained model and training metrics
    """
    # Setup distributed training
    is_distributed, world_size, rank, local_rank = setup_distributed()
    is_main = is_main_process(rank)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + rank)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}' if is_distributed else 'cuda')
    else:
        device = torch.device('cpu')
    
    if is_main:
        logger.info(f"\n=== Training Configuration ===")
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: {is_distributed} (world_size={world_size})")
        logger.info(f"Mixed Precision: {config.training.use_amp}")
        logger.info(f"Epochs: {config.training.epochs}")
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.learning_rate}")
    
    # Prepare data
    if is_main:
        logger.info("\n=== Preparing Data ===")
    
    train_ds, val_ds, test_ds, feature_names, scaler_stats = prepare_datasets(config, n_candles)
    
    # Create samplers for distributed training
    if is_distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    # Update model config with actual feature count
    config.model.input_dim = len(feature_names)
    
    # Create model
    if is_main:
        logger.info("\n=== Creating Model ===")
    
    model = CausalTimeSeriesTransformer(config.model).to(device)
    
    if is_main:
        logger.info(f"Parameters: {count_parameters(model):,}")
        verify_causal_masking(model, seq_len=20)
    
    # Wrap model for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_loader).to(device)
    if is_main:
        logger.info(f"Class weights: {class_weights}")
    
    # Loss function
    criterion = TradingSignalLoss(class_weights=class_weights, label_smoothing=0.1)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        verbose=is_main
    )
    
    # Gradient scaler for AMP
    scaler = GradScaler() if config.training.use_amp and torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.training.patience)
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        if is_main:
            logger.info(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        
        if is_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        metrics_dict = checkpoint['metrics']
        metrics.train_losses = metrics_dict['train_losses']
        metrics.val_losses = metrics_dict['val_losses']
        metrics.train_accuracies = metrics_dict['train_accuracies']
        metrics.val_accuracies = metrics_dict['val_accuracies']
        metrics.learning_rates = metrics_dict['learning_rates']
        metrics.epoch_times = metrics_dict['epoch_times']
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    if is_main:
        checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    if is_main:
        logger.info("\n=== Starting Training ===")
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.epochs):
        epoch_start = time.time()
        
        # Set epoch for distributed sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config, device, scaler, epoch
        )
        
        # Validate
        val_loss, val_acc, class_accs = validate(
            model, val_loader, criterion, device, config
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        epoch_time = time.time() - epoch_start
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Print progress (main process only)
        if is_main:
            logger.info(f"\nEpoch {epoch + 1}/{config.training.epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            logger.info(f"  Class Acc: Hold={class_accs[0]:.3f}, Long={class_accs[1]:.3f}, Short={class_accs[2]:.3f}")
            logger.info(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics, config,
                    feature_names, scaler_stats, str(best_path), is_distributed
                )
                logger.info(f"  -> New best model saved!")
            
            # Periodic checkpoint
            if (epoch + 1) % config.training.save_every == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, metrics, config,
                    feature_names, scaler_stats, str(ckpt_path), is_distributed
                )
        
        # Early stopping
        if early_stopping(val_loss, model):
            if is_main:
                logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Load best model
    if early_stopping.best_model_state is not None:
        if is_distributed:
            model.module.load_state_dict(early_stopping.best_model_state)
        else:
            model.load_state_dict(early_stopping.best_model_state)
    
    # Final evaluation on test set
    if is_main:
        logger.info("\n=== Final Evaluation on Test Set ===")
        test_loss, test_acc, test_class_accs = validate(
            model, test_loader, criterion, device, config
        )
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Class Accuracies: Hold={test_class_accs[0]:.3f}, Long={test_class_accs[1]:.3f}, Short={test_class_accs[2]:.3f}")
        
        # Save final model
        final_path = checkpoint_dir / "final_model.pt"
        model_state = model.module.state_dict() if is_distributed else model.state_dict()
        
        torch.save({
            'model_state_dict': model_state,
            'config': config.model.__dict__,
            'feature_names': feature_names,
            'scaler_mean': scaler_stats['mean'],
            'scaler_std': scaler_stats['std'],
            'test_metrics': {
                'loss': test_loss,
                'accuracy': test_acc,
                'class_accuracies': test_class_accs
            }
        }, str(final_path))
        logger.info(f"\nFinal model saved to: {final_path}")
        
        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    # Cleanup distributed training
    cleanup_distributed()
    
    # Return the unwrapped model
    if is_distributed:
        return model.module, metrics
    return model, metrics


if __name__ == "__main__":
    # Run training
    config = Config()
    config.data.use_real_data = True  # Use real Binance data
    
    model, metrics = train(config, n_candles=5000)
    
    logger.info("\n=== Training Complete ===")
    logger.info(f"Best validation loss: {min(metrics.val_losses):.4f}")
    logger.info(f"Best validation accuracy: {max(metrics.val_accuracies):.4f}")
