"""
Training Pipeline for Crypto Trading Bot
==========================================
Implements training with:
- Early stopping
- Learning rate scheduling
- Gradient clipping
- Checkpointing
- Validation monitoring
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import Config, DEFAULT_CONFIG
from model import CausalTimeSeriesTransformer, TradingSignalLoss, count_parameters, verify_causal_masking
from data import prepare_datasets, create_dataloaders


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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Classification accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (sequences, labels, _) in enumerate(pbar):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
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
    device: torch.device
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
        sequences = sequences.to(device)
        labels = labels.to(device)
        
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
    scheduler: ReduceLROnPlateau,
    epoch: int,
    metrics: TrainingMetrics,
    config: Config,
    feature_names: List[str],
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics.to_dict(),
        'feature_names': feature_names,
        'config': config.model.__dict__
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[ReduceLROnPlateau] = None
) -> Tuple[int, TrainingMetrics]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    metrics = TrainingMetrics()
    metrics_dict = checkpoint['metrics']
    metrics.train_losses = metrics_dict['train_losses']
    metrics.val_losses = metrics_dict['val_losses']
    metrics.train_accuracies = metrics_dict['train_accuracies']
    metrics.val_accuracies = metrics_dict['val_accuracies']
    metrics.learning_rates = metrics_dict['learning_rates']
    metrics.epoch_times = metrics_dict['epoch_times']
    
    return checkpoint['epoch'], metrics


def train(
    config: Config = DEFAULT_CONFIG,
    n_candles: int = 10000,
    resume_from: Optional[str] = None
) -> Tuple[nn.Module, TrainingMetrics]:
    """
    Main training function.
    
    Args:
        config: Configuration object
        n_candles: Number of candles to use for training
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Trained model and training metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device(config.training.device)
    print(f"\n=== Training Configuration ===")
    print(f"Device: {device}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Prepare data
    print("\n=== Preparing Data ===")
    train_ds, val_ds, test_ds, feature_names, scaler_stats = prepare_datasets(config, n_candles)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, config
    )
    
    # Update model config with actual feature count
    config.model.input_dim = len(feature_names)
    
    # Create model
    print("\n=== Creating Model ===")
    model = CausalTimeSeriesTransformer(config.model).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Verify causal masking
    verify_causal_masking(model, seq_len=20)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_loader).to(device)
    print(f"Class weights: {class_weights}")
    
    # Loss function
    criterion = TradingSignalLoss(class_weights=class_weights)
    
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
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.training.patience)
    
    # Metrics tracking
    metrics = TrainingMetrics()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        start_epoch, metrics = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        start_epoch += 1
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print("\n=== Starting Training ===")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config, device
        )
        
        # Validate
        val_loss, val_acc, class_accs = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track metrics
        epoch_time = time.time() - epoch_start
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Print progress
        print(f"\nEpoch {epoch + 1}/{config.training.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Class Acc: Hold={class_accs[0]:.3f}, Long={class_accs[1]:.3f}, Short={class_accs[2]:.3f}")
        print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.model.__dict__,
                'feature_names': feature_names,
                'scaler_mean': scaler_stats['mean'],
                'scaler_std': scaler_stats['std'],
                'val_loss': val_loss,
                'val_acc': val_acc
            }, best_path)
            print(f"  -> New best model saved!")
        
        # Periodic checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, feature_names, str(ckpt_path))
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    # Load best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    test_loss, test_acc, test_class_accs = validate(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Class Accuracies: Hold={test_class_accs[0]:.3f}, Long={test_class_accs[1]:.3f}, Short={test_class_accs[2]:.3f}")
    
    # Save final model with all necessary info
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
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
    print(f"\nFinal model saved to: {final_path}")
    
    # Save training history
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    return model, metrics


if __name__ == "__main__":
    # Run training
    config = Config()
    model, metrics = train(config, n_candles=5000)
    
    print("\n=== Training Complete ===")
    print(f"Best validation loss: {min(metrics.val_losses):.4f}")
    print(f"Best validation accuracy: {max(metrics.val_accuracies):.4f}")
