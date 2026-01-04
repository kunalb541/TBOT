"""
Data Module for Crypto Trading Bot
====================================
Handles data fetching, preprocessing, and dataset creation.
Uses fake data generation for POC (replace with real API calls).

FIXED ISSUES:
- Proper lookback context for val/test sets
- Correct sequence/label alignment
- Returns scaler statistics for inference
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from config import Config, DEFAULT_CONFIG


def generate_fake_ohlcv(
    n_candles: int = 10000,
    start_price: float = 40000.0,
    volatility: float = 0.02,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate realistic fake OHLCV data for testing.
    In production, replace with actual API calls.
    
    Uses geometric Brownian motion for price simulation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate timestamps (hourly)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=n_candles)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_candles)
    
    # Generate price series using GBM
    dt = 1 / 24  # 1 hour in days
    drift = 0.0001  # Slight upward drift
    
    log_returns = np.random.normal(
        loc=drift * dt,
        scale=volatility * np.sqrt(dt),
        size=n_candles
    )
    
    # Add some regime changes for realism
    regime_changes = np.random.choice(n_candles, size=min(10, n_candles // 100), replace=False)
    for idx in regime_changes:
        regime_shift = np.random.choice([-0.05, 0.05])
        log_returns[idx] += regime_shift
    
    close_prices = start_price * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC from close
    intrabar_vol = volatility * 0.5
    
    data = {
        'timestamp': timestamps,
        'open': np.zeros(n_candles),
        'high': np.zeros(n_candles),
        'low': np.zeros(n_candles),
        'close': close_prices,
        'volume': np.zeros(n_candles)
    }
    
    for i in range(n_candles):
        noise = np.random.uniform(-intrabar_vol, intrabar_vol, 3)
        
        if i == 0:
            data['open'][i] = close_prices[i] * (1 + noise[0])
        else:
            data['open'][i] = close_prices[i - 1]  # Open = prev close
        
        # High and Low
        data['high'][i] = max(data['open'][i], close_prices[i]) * (1 + abs(noise[1]))
        data['low'][i] = min(data['open'][i], close_prices[i]) * (1 - abs(noise[2]))
        
        # Volume (correlated with volatility)
        base_volume = 100 + abs(log_returns[i]) * 10000
        data['volume'][i] = base_volume * np.random.uniform(0.8, 1.2)
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


class CryptoDataFetcher:
    """
    Fetches cryptocurrency OHLCV data.
    For POC uses fake data; replace with real API in production.
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._cache = {}
    
    def fetch_historical(
        self,
        symbol: str,
        interval: str,
        n_candles: int = 10000,
        use_fake: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1h")
            n_candles: Number of candles to fetch
            use_fake: Use fake data for POC
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{interval}_{n_candles}"
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        if use_fake:
            df = generate_fake_ohlcv(n_candles=n_candles, seed=self.config.seed)
        else:
            # In production, implement real API call here
            raise NotImplementedError("Real API not implemented in POC")
        
        self._cache[cache_key] = df
        return df.copy()
    
    def fetch_latest(self, symbol: str, interval: str, n_candles: int = 1) -> pd.DataFrame:
        """
        Fetch latest candles for live trading.
        In POC, generates new fake data point.
        """
        # Use a different seed each time for variety
        return generate_fake_ohlcv(n_candles=n_candles, seed=None)


def compute_technical_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute technical indicators with NO LOOKAHEAD BIAS.
    
    CRITICAL: All indicators must be computed using ONLY past data.
    Rolling operations naturally use only past data.
    """
    df = df.copy()
    
    # Price-based features (normalized by close)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # High-Low range normalized
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Body size
    df['body_pct'] = (df['close'] - df['open']) / df['close']
    
    # Upper/Lower wicks
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=config.data.rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=config.data.rsi_period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Normalize to [-1, 1]
    
    # MACD
    ema_fast = df['close'].ewm(span=config.data.macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=config.data.macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=config.data.macd_signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # Normalize MACD by price
    df['macd_normalized'] = df['macd'] / df['close']
    df['macd_hist_normalized'] = df['macd_hist'] / df['close']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=config.data.bb_period, min_periods=1).mean()
    bb_std = df['close'].rolling(window=config.data.bb_period, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * config.data.bb_std)
    df['bb_lower'] = df['bb_middle'] - (bb_std * config.data.bb_std)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
    df['bb_width'] = bb_range / (df['bb_middle'] + 1e-10)
    
    # ATR (Average True Range)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=config.data.atr_period, min_periods=1).mean()
    df['atr_normalized'] = df['atr'] / df['close']
    
    # EMAs
    for period in config.data.ema_periods:
        ema = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}'] = (df['close'] - ema) / (ema + 1e-10)  # Distance from EMA
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['volume_normalized'] = np.log1p(df['volume']) - np.log1p(df['volume_ma'])
    
    # Momentum indicators
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(periods=period)
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(window=20, min_periods=1).std()
    vol_ma = df['volatility_20'].rolling(window=50, min_periods=1).mean()
    df['volatility_normalized'] = df['volatility_20'] / (vol_ma + 1e-10)
    
    return df


def create_labels(
    df: pd.DataFrame,
    prediction_horizon: int = 1,
    threshold: float = 0.002  # 0.2% minimum move
) -> pd.Series:
    """
    Create classification labels based on future returns.
    
    Labels:
        0: Hold (small move)
        1: Long (price goes up significantly)
        2: Short (price goes down significantly)
    
    CRITICAL: Labels are based on FUTURE data, which is correct for training.
    The label at index i represents what happens between close[i] and close[i+horizon].
    """
    # Future return: (close[i+horizon] - close[i]) / close[i]
    future_returns = df['close'].shift(-prediction_horizon) / df['close'] - 1
    
    labels = pd.Series(index=df.index, dtype=int, data=0)
    labels[future_returns > threshold] = 1   # Long signal
    labels[future_returns < -threshold] = 2  # Short signal
    # Everything else stays 0 (Hold)
    
    return labels


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with STRICT temporal ordering.
    
    ANTI-CHEATING MEASURES:
    1. No shuffling of time series data
    2. Sequences are contiguous in time
    3. Features are computed using only past data
    4. Labels are based on future returns (only known in training)
    
    IMPORTANT: The sequence for predicting label[i] includes features UP TO AND INCLUDING index i.
    This is because at time i, we know all data up to i and want to predict what happens next.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        lookback_window: int,
        feature_names: List[str],
        start_offset: int = 0  # For tracking global position
    ):
        """
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
            timestamps: Timestamp array for verification
            lookback_window: Number of past candles to use
            feature_names: Names of features for debugging
            start_offset: Global offset for this dataset (for debugging)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.timestamps = timestamps
        self.lookback_window = lookback_window
        self.feature_names = feature_names
        self.start_offset = start_offset
        
        # Valid indices: must have enough history AND valid label
        # Sequence ending at index i uses features[i-lookback+1 : i+1]
        # So we need i >= lookback - 1
        self.valid_indices = list(range(lookback_window - 1, len(features)))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sequence ending at the given index.
        
        For predicting label at position i:
        - Use features from [i - lookback + 1, i] inclusive
        - This gives us `lookback` consecutive features ending at i
        
        Returns:
            features: (lookback_window, n_features)
            label: scalar - what happens AFTER this sequence
            actual_idx: the actual index in the original data (for debugging)
        """
        actual_idx = self.valid_indices[idx]
        
        # Sequence: features from (actual_idx - lookback + 1) to actual_idx inclusive
        start_idx = actual_idx - self.lookback_window + 1
        end_idx = actual_idx + 1  # +1 because slice is exclusive
        
        sequence = self.features[start_idx:end_idx]  # (lookback_window, n_features)
        label = self.labels[actual_idx]
        
        return sequence, label, actual_idx + self.start_offset
    
    def get_timestamp(self, idx: int) -> datetime:
        """Get timestamp for debugging."""
        actual_idx = self.valid_indices[idx]
        return self.timestamps[actual_idx]


def prepare_datasets(
    config: Config = DEFAULT_CONFIG,
    n_candles: int = 10000
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset, List[str], Dict]:
    """
    Prepare train/val/test datasets with STRICT TEMPORAL SPLIT.
    
    CRITICAL FOR NO LOOKAHEAD:
    - Train < Val < Test in time
    - No overlap between splits
    - Features computed before splitting to ensure consistency
    - Val/Test have access to lookback context from previous split
    
    Returns:
        train_dataset, val_dataset, test_dataset, feature_names, scaler_stats
    """
    # Fetch data
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol=config.data.symbol,
        interval=config.data.interval,
        n_candles=n_candles
    )
    
    # Compute technical indicators
    df = compute_technical_indicators(df, config)
    
    # Create labels
    labels = create_labels(df, config.data.prediction_horizon)
    df['label'] = labels
    
    # Drop rows with NaN (from rolling calculations at the start)
    # and rows without valid labels (at the end due to shift)
    df = df.dropna()
    
    # Also remove the last prediction_horizon rows since they don't have valid labels
    df = df.iloc[:-config.data.prediction_horizon] if config.data.prediction_horizon > 0 else df
    
    # Select features
    feature_cols = [
        'returns', 'log_returns', 'range_pct', 'body_pct',
        'upper_wick', 'lower_wick',
        'rsi_normalized', 'macd_normalized', 'macd_hist_normalized',
        'bb_position', 'bb_width', 'atr_normalized',
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'volume_normalized', 'volume_ratio',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_normalized'
    ]
    
    # Ensure all features exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Extract arrays
    features = df[feature_cols].values
    labels_arr = df['label'].values
    timestamps = df.index.values
    
    # Update config with actual input dim
    config.model.input_dim = len(feature_cols)
    
    # TEMPORAL SPLIT indices
    n_samples = len(df)
    train_end = int(n_samples * config.data.train_ratio)
    val_end = int(n_samples * (config.data.train_ratio + config.data.val_ratio))
    
    lookback = config.data.lookback_window
    
    print(f"\n=== Data Split ===")
    print(f"Total samples: {n_samples}")
    print(f"Train: 0 to {train_end} ({train_end} samples)")
    print(f"Val: {train_end} to {val_end} ({val_end - train_end} samples)")
    print(f"Test: {val_end} to {n_samples} ({n_samples - val_end} samples)")
    print(f"Lookback window: {lookback}")
    print(f"Features: {len(feature_cols)}")
    
    # Normalize features using ONLY training data statistics
    train_features = features[:train_end]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
    
    features_normalized = (features - mean) / std
    
    # Store scaler stats for inference
    scaler_stats = {
        'mean': mean,
        'std': std
    }
    
    # Create datasets
    # IMPORTANT: Val and Test need lookback context from previous data
    # So we include the previous lookback-1 samples
    
    # Training set: just use training data
    train_dataset = TimeSeriesDataset(
        features=features_normalized[:train_end],
        labels=labels_arr[:train_end],
        timestamps=timestamps[:train_end],
        lookback_window=lookback,
        feature_names=feature_cols,
        start_offset=0
    )
    
    # Validation set: include lookback context from end of training
    val_start_with_context = max(0, train_end - lookback + 1)
    val_dataset = TimeSeriesDataset(
        features=features_normalized[val_start_with_context:val_end],
        labels=labels_arr[val_start_with_context:val_end],
        timestamps=timestamps[val_start_with_context:val_end],
        lookback_window=lookback,
        feature_names=feature_cols,
        start_offset=val_start_with_context
    )
    # Adjust valid_indices to only include actual validation samples
    val_dataset.valid_indices = [
        i for i in val_dataset.valid_indices 
        if i + val_start_with_context >= train_end
    ]
    
    # Test set: include lookback context from end of validation
    test_start_with_context = max(0, val_end - lookback + 1)
    test_dataset = TimeSeriesDataset(
        features=features_normalized[test_start_with_context:],
        labels=labels_arr[test_start_with_context:],
        timestamps=timestamps[test_start_with_context:],
        lookback_window=lookback,
        feature_names=feature_cols,
        start_offset=test_start_with_context
    )
    # Adjust valid_indices to only include actual test samples
    test_dataset.valid_indices = [
        i for i in test_dataset.valid_indices 
        if i + test_start_with_context >= val_end
    ]
    
    # Print label distribution
    print(f"\n=== Label Distribution ===")
    for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        if len(dataset) > 0:
            labels_list = [dataset.labels[dataset.valid_indices[i]].item() for i in range(len(dataset))]
            unique, counts = np.unique(labels_list, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"{split_name}: {dist} (total: {len(dataset)})")
        else:
            print(f"{split_name}: empty")
    
    return train_dataset, val_dataset, test_dataset, feature_cols, scaler_stats


def create_dataloaders(
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders.
    
    NOTE: shuffle=True only for training. Val/Test maintain temporal order.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,  # OK to shuffle training data within the split
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,  # Never shuffle test
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data pipeline
    config = Config()
    
    print("Testing data pipeline...")
    train_ds, val_ds, test_ds, features, scaler = prepare_datasets(config, n_candles=5000)
    
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
    # Test a sample
    if len(train_ds) > 0:
        seq, label, idx = train_ds[0]
        print(f"\nSample shape: {seq.shape}")
        print(f"Label: {label}")
        print(f"Global index: {idx}")
    
    print(f"\nFeatures: {features}")
    print(f"\nScaler mean shape: {scaler['mean'].shape}")
    print(f"Scaler std shape: {scaler['std'].shape}")
