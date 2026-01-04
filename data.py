"""
Data Module for Crypto Trading Bot
====================================
Handles data fetching, preprocessing, and dataset creation.

FEATURES:
- Real Binance data fetching (works without API keys for public endpoints)
- Fake data fallback for testing
- Proper temporal split with lookback context
- Anti-lookahead measures

FIXES:
- Fixed sequence/label alignment
- Proper lookback context for val/test sets
- Returns scaler statistics for inference
- Handles Binance API rate limits
- Better retry logic and error handling
- Fixed connection timeout issues
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Union
from datetime import datetime, timedelta
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from config import Config, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# BINANCE DATA FETCHER
# ============================================================================

class BinanceDataFetcher:
    """
    Fetches cryptocurrency OHLCV data from Binance.
    
    NOTE: Public market data endpoints don't require API keys!
    API keys are only needed for trading, account info, etc.
    """
    
    # Binance public API endpoints
    SPOT_BASE_URL = "https://api.binance.com"
    FUTURES_BASE_URL = "https://fapi.binance.com"
    
    # Interval mapping
    INTERVAL_MS = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
    }
    
    def __init__(self, config: Config = DEFAULT_CONFIG, use_futures: bool = True):
        """
        Initialize Binance data fetcher.
        
        Args:
            config: Configuration object
            use_futures: If True, use Futures API; else use Spot API
        """
        self.config = config
        self.use_futures = use_futures
        self.base_url = self.FUTURES_BASE_URL if use_futures else self.SPOT_BASE_URL
        self._cache = {}
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Setup session with retry logic
        self._session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry logic."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
        
    def _rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[List]:
        """Make API request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self._session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code == 418:  # IP ban
                    logger.error("IP banned by Binance. Wait and try again later.")
                    time.sleep(60 * (attempt + 1))
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch kline/candlestick data from Binance.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1h", "4h", "1d")
            limit: Number of candles (max 1500 for futures, 1000 for spot)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            DataFrame with OHLCV data
        """
        endpoint = "/fapi/v1/klines" if self.use_futures else "/api/v3/klines"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500 if self.use_futures else 1000)
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        data = self._make_request(endpoint, params)
        
        if data is None:
            return None
        
        # Parse response
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Select and rename columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        
        return df
    
    def fetch_historical(
        self,
        symbol: str,
        interval: str,
        n_candles: int = 10000,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with pagination.
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            n_candles: Total number of candles to fetch
            end_time: End time for data fetch (default: now)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{interval}_{n_candles}"
        if cache_key in self._cache:
            logger.info(f"Using cached data for {cache_key}")
            return self._cache[cache_key].copy()
        
        logger.info(f"Fetching {n_candles} candles for {symbol} ({interval})...")
        
        all_data = []
        remaining = n_candles
        current_end_time = int(end_time.timestamp() * 1000) if end_time else None
        
        max_per_request = 1500 if self.use_futures else 1000
        
        while remaining > 0:
            batch_size = min(remaining, max_per_request)
            
            df_batch = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=batch_size,
                end_time=current_end_time
            )
            
            if df_batch is None or len(df_batch) == 0:
                logger.warning(f"No more data available. Got {sum(len(d) for d in all_data)} candles.")
                break
            
            all_data.append(df_batch)
            remaining -= len(df_batch)
            
            # Update end_time for next batch (go further back in time)
            oldest_time = df_batch.index[0]
            current_end_time = int(oldest_time.timestamp() * 1000) - 1
            
            logger.info(f"  Fetched {len(df_batch)} candles, {remaining} remaining...")
            
            # Small delay to be nice to the API
            time.sleep(0.1)
        
        if not all_data:
            logger.error("No data fetched. Using fake data as fallback.")
            return generate_fake_ohlcv(n_candles=n_candles, seed=self.config.seed)
        
        # Combine and sort
        df = pd.concat(all_data)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
        
        logger.info(f"Total candles fetched: {len(df)}")
        
        # Cache the result
        self._cache[cache_key] = df
        
        return df.copy()
    
    def fetch_latest(self, symbol: str, interval: str, n_candles: int = 1) -> pd.DataFrame:
        """Fetch latest candles for live trading."""
        return self.fetch_klines(symbol=symbol, interval=interval, limit=n_candles)


# ============================================================================
# FAKE DATA GENERATOR (for testing without API)
# ============================================================================

def generate_fake_ohlcv(
    n_candles: int = 10000,
    start_price: float = 40000.0,
    volatility: float = 0.02,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate realistic fake OHLCV data for testing.
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
    regime_changes = np.random.choice(
        n_candles, 
        size=min(10, max(1, n_candles // 100)), 
        replace=False
    )
    for idx in regime_changes:
        regime_shift = np.random.choice([-0.05, 0.05])
        log_returns[idx] += regime_shift
    
    close_prices = start_price * np.exp(np.cumsum(log_returns))
    
    # Generate OHLC from close
    intrabar_vol = volatility * 0.5
    
    data = {
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
    
    df = pd.DataFrame(data, index=timestamps)
    df.index.name = 'timestamp'
    
    return df


# ============================================================================
# CRYPTO DATA FETCHER (UNIFIED INTERFACE)
# ============================================================================

class CryptoDataFetcher:
    """
    Unified data fetcher that handles both real Binance data and fake data.
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self._binance_fetcher = None
        self._cache = {}
    
    @property
    def binance(self) -> BinanceDataFetcher:
        """Lazy initialization of Binance fetcher."""
        if self._binance_fetcher is None:
            self._binance_fetcher = BinanceDataFetcher(self.config)
        return self._binance_fetcher
    
    def fetch_historical(
        self,
        symbol: str,
        interval: str,
        n_candles: int = 10000,
        use_fake: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1h")
            n_candles: Number of candles to fetch
            use_fake: Override config setting for fake data
            
        Returns:
            DataFrame with OHLCV data
        """
        if use_fake is None:
            use_fake = not self.config.data.use_real_data
        
        cache_key = f"{symbol}_{interval}_{n_candles}_{use_fake}"
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        if use_fake:
            logger.info("Using fake data generator...")
            df = generate_fake_ohlcv(n_candles=n_candles, seed=self.config.seed)
        else:
            logger.info("Fetching real data from Binance...")
            try:
                df = self.binance.fetch_historical(
                    symbol=symbol,
                    interval=interval,
                    n_candles=n_candles
                )
            except Exception as e:
                logger.error(f"Failed to fetch from Binance: {e}")
                logger.info("Falling back to fake data...")
                df = generate_fake_ohlcv(n_candles=n_candles, seed=self.config.seed)
        
        self._cache[cache_key] = df
        return df.copy()
    
    def fetch_latest(self, symbol: str, interval: str, n_candles: int = 1) -> pd.DataFrame:
        """Fetch latest candles for live trading."""
        if self.config.data.use_real_data:
            return self.binance.fetch_latest(symbol, interval, n_candles)
        else:
            return generate_fake_ohlcv(n_candles=n_candles, seed=None)


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

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


# ============================================================================
# DATASET CLASS
# ============================================================================

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series with STRICT temporal ordering.
    
    ANTI-CHEATING MEASURES:
    1. No shuffling of time series data
    2. Sequences are contiguous in time
    3. Features are computed using only past data
    4. Labels are based on future returns (only known in training)
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        lookback_window: int,
        feature_names: List[str],
        start_offset: int = 0
    ):
        """
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
            timestamps: Timestamp array for verification
            lookback_window: Number of past candles to use
            feature_names: Names of features for debugging
            start_offset: Global offset for this dataset
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.timestamps = timestamps
        self.lookback_window = lookback_window
        self.feature_names = feature_names
        self.start_offset = start_offset
        
        # Valid indices: must have enough history AND valid label
        self.valid_indices = list(range(lookback_window - 1, len(features)))
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sequence ending at the given index.
        
        Returns:
            features: (lookback_window, n_features)
            label: scalar
            actual_idx: the actual index in the original data
        """
        actual_idx = self.valid_indices[idx]
        
        start_idx = actual_idx - self.lookback_window + 1
        end_idx = actual_idx + 1
        
        sequence = self.features[start_idx:end_idx]
        label = self.labels[actual_idx]
        
        return sequence, label, actual_idx + self.start_offset
    
    def get_timestamp(self, idx: int) -> datetime:
        """Get timestamp for debugging."""
        actual_idx = self.valid_indices[idx]
        return self.timestamps[actual_idx]


# ============================================================================
# DATA PREPARATION
# ============================================================================

# Define feature columns globally for consistency
FEATURE_COLS = [
    'returns', 'log_returns', 'range_pct', 'body_pct',
    'upper_wick', 'lower_wick',
    'rsi_normalized', 'macd_normalized', 'macd_hist_normalized',
    'bb_position', 'bb_width', 'atr_normalized',
    'ema_9', 'ema_21', 'ema_50', 'ema_200',
    'volume_normalized', 'volume_ratio',
    'momentum_5', 'momentum_10', 'momentum_20',
    'volatility_normalized'
]


def prepare_datasets(
    config: Config = DEFAULT_CONFIG,
    n_candles: int = 10000
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset, List[str], Dict]:
    """
    Prepare train/val/test datasets with STRICT TEMPORAL SPLIT.
    
    CRITICAL FOR NO LOOKAHEAD:
    - Train < Val < Test in time
    - No overlap between splits
    - Features computed before splitting
    - Val/Test have access to lookback context
    
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
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Remove last prediction_horizon rows (no valid labels)
    if config.data.prediction_horizon > 0:
        df = df.iloc[:-config.data.prediction_horizon]
    
    # Select features
    feature_cols = [col for col in FEATURE_COLS if col in df.columns]
    
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
    
    logger.info(f"\n=== Data Split ===")
    logger.info(f"Total samples: {n_samples}")
    logger.info(f"Train: 0 to {train_end} ({train_end} samples)")
    logger.info(f"Val: {train_end} to {val_end} ({val_end - train_end} samples)")
    logger.info(f"Test: {val_end} to {n_samples} ({n_samples - val_end} samples)")
    logger.info(f"Lookback window: {lookback}")
    logger.info(f"Features: {len(feature_cols)}")
    
    # Normalize features using ONLY training data statistics
    train_features = features[:train_end]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    
    features_normalized = (features - mean) / std
    
    # Store scaler stats for inference
    scaler_stats = {
        'mean': mean,
        'std': std
    }
    
    # Create datasets with proper lookback context
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
    test_dataset.valid_indices = [
        i for i in test_dataset.valid_indices 
        if i + test_start_with_context >= val_end
    ]
    
    # Print label distribution
    logger.info(f"\n=== Label Distribution ===")
    for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        if len(dataset) > 0:
            labels_list = [dataset.labels[dataset.valid_indices[i]].item() for i in range(len(dataset))]
            unique, counts = np.unique(labels_list, return_counts=True)
            dist = dict(zip(unique, counts))
            logger.info(f"{split_name}: {dist} (total: {len(dataset)})")
        else:
            logger.info(f"{split_name}: empty")
    
    return train_dataset, val_dataset, test_dataset, feature_cols, scaler_stats


def create_dataloaders(
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    config: Config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data pipeline
    config = Config()
    config.data.use_real_data = True  # Try real data
    
    print("Testing data pipeline...")
    
    # Test Binance fetcher directly
    print("\n=== Testing Binance Data Fetch ===")
    fetcher = BinanceDataFetcher(config)
    df = fetcher.fetch_klines("BTCUSDT", "1h", limit=100)
    if df is not None:
        print(f"Fetched {len(df)} candles")
        print(df.head())
        print(df.tail())
    
    # Test full pipeline
    print("\n=== Testing Full Pipeline ===")
    train_ds, val_ds, test_ds, features, scaler = prepare_datasets(config, n_candles=2000)
    
    print(f"\nTrain dataset size: {len(train_ds)}")
    print(f"Val dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
    if len(train_ds) > 0:
        seq, label, idx = train_ds[0]
        print(f"\nSample shape: {seq.shape}")
        print(f"Label: {label}")
        print(f"Global index: {idx}")
    
    print(f"\nFeatures: {features}")
    print(f"Scaler mean shape: {scaler['mean'].shape}")
