"""
Configuration for Crypto Trading Bot
=====================================
All hyperparameters and settings in one place.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class DataConfig:
    """Data-related configuration."""
    symbol: str = "BTCUSDT"
    interval: str = "1h"  # 1h candles for less noise
    lookback_window: int = 168  # 7 days of hourly data
    prediction_horizon: int = 1  # Predict next candle
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Features to use
    ohlcv_cols: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume"
    ])
    
    # Technical indicator periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])


@dataclass
class ModelConfig:
    """Transformer model configuration."""
    # Input/Output
    input_dim: int = 32  # Number of input features after engineering
    output_dim: int = 3  # 0: Hold, 1: Long, 2: Short
    
    # Transformer architecture
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Positional encoding
    max_seq_len: int = 512
    
    # Causal attention (CRITICAL for no lookahead)
    causal: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15  # Early stopping patience
    
    # Learning rate scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5


@dataclass
class TradingConfig:
    """Trading/Backtesting configuration."""
    initial_capital: float = 10000.0
    position_size: float = 0.1  # 10% of capital per trade
    leverage: int = 1  # No leverage for safety
    
    # Fees (Binance-like)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 500.0
    daily_profit_target: float = 500.0
    
    # Slippage simulation
    slippage_pct: float = 0.001  # 0.1%
    
    # Minimum confidence threshold for trades
    min_confidence: float = 0.6


@dataclass
class RetrainingConfig:
    """Auto-retraining configuration."""
    enabled: bool = True
    retrain_interval_hours: int = 24 * 7  # Weekly
    min_new_samples: int = 168  # At least 1 week of new data
    performance_threshold: float = 0.0  # Retrain if PnL drops below this
    rolling_window_days: int = 90  # Use last 90 days for retraining


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        # Use math.isclose for float comparison to avoid floating point issues
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-9):
            raise ValueError(
                f"Data split ratios must sum to 1.0, got {total_ratio} "
                f"({self.data.train_ratio} + {self.data.val_ratio} + {self.data.test_ratio})"
            )
        
        if self.model.d_model % self.model.nhead != 0:
            raise ValueError(
                f"d_model ({self.model.d_model}) must be divisible by nhead ({self.model.nhead})"
            )
        
        if not (0 < self.trading.position_size <= 1.0):
            raise ValueError(
                f"position_size must be in (0, 1], got {self.trading.position_size}"
            )


# Default config instance
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    # Print config for verification
    config = Config()
    print("=== Crypto Bot Configuration ===")
    print(f"Symbol: {config.data.symbol}")
    print(f"Interval: {config.data.interval}")
    print(f"Lookback: {config.data.lookback_window}")
    print(f"Model: d_model={config.model.d_model}, layers={config.model.num_encoder_layers}")
    print(f"Causal Attention: {config.model.causal}")
    print(f"Device: {config.training.device}")
    print(f"\nData splits: train={config.data.train_ratio}, val={config.data.val_ratio}, test={config.data.test_ratio}")
    print(f"Sum: {config.data.train_ratio + config.data.val_ratio + config.data.test_ratio}")
