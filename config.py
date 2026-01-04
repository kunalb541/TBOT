"""
Configuration for Crypto Trading Bot
=====================================
All hyperparameters and settings in one place.

FIXES:
- Added Binance API configuration
- Added proper environment variable loading
- Fixed float comparison issue
- Fixed num_workers for HPC compatibility
"""

import os
import math
from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class BinanceConfig:
    """Binance API configuration."""
    # Load from environment variables for security
    api_key: str = field(default_factory=lambda: os.environ.get("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("BINANCE_API_SECRET", ""))
    
    # Use testnet for safety during development
    testnet: bool = False
    testnet_api_url: str = "https://testnet.binancefuture.com"
    
    # Rate limiting
    max_requests_per_minute: int = 1200
    request_timeout: int = 30


@dataclass
class DataConfig:
    """Data-related configuration."""
    symbol: str = "BTCUSDT"
    interval: str = "15m"  # 1h candles for less noise
    lookback_window: int = 200  # 7 days of hourly data
    prediction_horizon: int = 1  # Predict next candle
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Use real data from Binance
    use_real_data: bool = True
    
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
    input_dim: int = 22  # Number of input features after engineering
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
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 500
    patience: int = 15  # Early stopping patience
    
    # Learning rate scheduler
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Device - auto-detect
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Multi-GPU / Distributed training
    distributed: bool = True
    world_size: int = 4
    local_rank: int = 0
    
    # Mixed precision training
    use_amp: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    
    # Number of workers for data loading
    # Set to 0 for HPC to avoid issues with distributed training
    num_workers: int = 0


@dataclass
class TradingConfig:
    """Trading/Backtesting configuration."""
    initial_capital: float = 10000.0
    position_size: float = 0.3  # 10% of capital per trade
    leverage: int = 75
    
    # Fees (Binance-like)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    
    # Risk management
    stop_loss_pct: float = 0.10  
    take_profit_pct: float = 0.20  
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 500.0
    daily_profit_target: float = 500.0
    
    # Slippage simulation
    slippage_pct: float = 0.001  # 0.1%
    
    # Minimum confidence threshold for trades
    min_confidence: float = 0.33


@dataclass
class RetrainingConfig:
    """Auto-retraining configuration."""
    enabled: bool = True
    retrain_interval_hours: int = 24 * 7  # Weekly
    min_new_samples: int = 200  
    performance_threshold: float = 0.0  # Retrain if PnL drops below this
    rolling_window_days: int = 90  # Use last 90 days for retraining


@dataclass
class Config:
    """Master configuration."""
    binance: BinanceConfig = field(default_factory=BinanceConfig)
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
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        # Would need custom parsing here
        return cls()
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        import yaml
        # Would need custom serialization here
        pass


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
    print(f"Use Real Data: {config.data.use_real_data}")
    print(f"\nData splits: train={config.data.train_ratio}, val={config.data.val_ratio}, test={config.data.test_ratio}")
    print(f"Sum: {config.data.train_ratio + config.data.val_ratio + config.data.test_ratio}")
