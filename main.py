"""
Crypto Trading Bot - Main Entry Point
=======================================
Complete pipeline: Train → Backtest → Paper Trade

Usage:
    python main.py train        # Train the model
    python main.py backtest     # Run backtest
    python main.py paper        # Run paper trading
    python main.py full         # Full pipeline
    python main.py demo         # Quick demo (no training)
    python main.py test-data    # Test Binance data fetching
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from config import Config, DEFAULT_CONFIG
from data import (
    prepare_datasets,
    create_dataloaders,
    CryptoDataFetcher,
    BinanceDataFetcher,
    compute_technical_indicators,
    FEATURE_COLS
)
from model import (
    CausalTimeSeriesTransformer,
    count_parameters,
    verify_causal_masking
)
from train import train, TrainingMetrics
from backtest import Backtester, print_backtest_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_binance_data(config: Config):
    """Test Binance data fetching."""
    print("\n" + "="*60)
    print("TESTING BINANCE DATA FETCH")
    print("="*60)
    
    print("\n1. Testing direct Binance API fetch...")
    fetcher = BinanceDataFetcher(config)
    
    # Test klines fetch
    df = fetcher.fetch_klines("BTCUSDT", "1h", limit=100)
    if df is not None:
        print(f"   ✓ Successfully fetched {len(df)} candles")
        print(f"   Time range: {df.index[0]} to {df.index[-1]}")
        print(f"   Latest close: ${df['close'].iloc[-1]:,.2f}")
    else:
        print("   ✗ Failed to fetch data")
        return
    
    print("\n2. Testing historical data fetch (pagination)...")
    df = fetcher.fetch_historical("BTCUSDT", "1h", n_candles=2000)
    print(f"   ✓ Successfully fetched {len(df)} candles")
    print(f"   Time range: {df.index[0]} to {df.index[-1]}")
    
    print("\n3. Testing technical indicators...")
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    print(f"   ✓ Computed {len(FEATURE_COLS)} indicators")
    print(f"   Valid samples after dropna: {len(df)}")
    
    print("\n4. Sample data:")
    print(df[['close', 'rsi_normalized', 'macd_normalized', 'bb_position']].tail())
    
    print("\n" + "="*60)
    print("DATA TEST COMPLETE")
    print("="*60)


def run_training(config: Config, n_candles: int = 8000):
    """Run model training."""
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    model, metrics = train(config, n_candles=n_candles)
    
    print("\n✓ Training complete!")
    print(f"  Best validation accuracy: {max(metrics.val_accuracies):.4f}")
    print(f"  Model saved to: checkpoints/")
    
    return model, metrics


def run_backtest(config: Config, checkpoint_path: str = "checkpoints/final_model.pt"):
    """Run backtesting on test data."""
    print("\n" + "="*60)
    print("BACKTESTING PHASE")
    print("="*60)
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run training first: python main.py train")
        return None
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device, weights_only=False)
    
    from config import ModelConfig
    model_config = ModelConfig(**checkpoint['config'])
    model = CausalTimeSeriesTransformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.training.device)
    model.eval()
    
    feature_names = checkpoint['feature_names']
    scaler_mean = checkpoint.get('scaler_mean')
    scaler_std = checkpoint.get('scaler_std')
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Features: {len(feature_names)}")
    
    # Fetch fresh data for backtesting
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol=config.data.symbol,
        interval=config.data.interval,
        n_candles=3000
    )
    
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    available_features = [f for f in feature_names if f in df.columns]
    if len(available_features) != len(feature_names):
        print(f"WARNING: Missing features: {set(feature_names) - set(available_features)}")
    
    if scaler_mean is None or scaler_std is None:
        print("WARNING: No saved scaler stats, computing from first 70% of data")
        features = df[available_features].values
        train_end = int(len(features) * 0.7)
        scaler_mean = features[:train_end].mean(axis=0)
        scaler_std = features[:train_end].std(axis=0)
        scaler_std = np.where(scaler_std < 1e-10, 1.0, scaler_std)
    
    # Use last 15% as test
    val_end = int(len(df) * 0.85)
    test_df = df.iloc[val_end:]
    
    print(f"\nBacktest period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Backtest samples: {len(test_df)}")
    
    backtester = Backtester(config)
    results = backtester.run(
        model=model,
        df=test_df,
        feature_cols=available_features,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std
    )
    
    print_backtest_results(results)
    
    # Save results
    results_path = Path("backtest_results.json")
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'initial_capital': results.initial_capital,
        'final_capital': results.final_capital,
        'total_return_pct': results.total_return_pct,
        'total_trades': results.total_trades,
        'win_rate': results.win_rate,
        'sharpe_ratio': results.sharpe_ratio,
        'max_drawdown_pct': results.max_drawdown_pct
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Backtest results saved to: {results_path}")
    
    return results


def run_full_pipeline(config: Config):
    """Run the complete pipeline: Train → Backtest"""
    print("\n" + "#"*60)
    print("# CRYPTO TRADING BOT - FULL PIPELINE")
    print("#"*60)
    
    # Step 1: Training
    model, metrics = run_training(config, n_candles=8000)
    
    # Step 2: Backtesting
    results = run_backtest(config)
    
    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    
    return model, results


def quick_demo(config: Config):
    """Quick demo without training."""
    print("\n" + "="*60)
    print("QUICK DEMO - No Training")
    print("="*60)
    
    feature_names = FEATURE_COLS
    config.model.input_dim = len(feature_names)
    model = CausalTimeSeriesTransformer(config.model)
    model.to(config.training.device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Verify causal masking
    print("\nVerifying causal masking...")
    is_causal = verify_causal_masking(model, seq_len=20)
    
    # Fetch real data
    print("\nFetching data...")
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol=config.data.symbol,
        interval=config.data.interval,
        n_candles=1000
    )
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    print(f"Fetched {len(df)} candles with {len(feature_names)} features")
    
    # Quick backtest
    print("\nRunning quick backtest with random model...")
    available_features = [f for f in feature_names if f in df.columns]
    features = df[available_features].values
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    
    backtester = Backtester(config)
    results = backtester.run(model, df, available_features, mean, std)
    
    print_backtest_results(results)
    
    print("\n✓ Quick demo complete!")
    print("  Note: This used a random (untrained) model, so results are meaningless.")
    print("  To run full training, use: python main.py train")


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot with Causal Time Series Transformer"
    )
    parser.add_argument(
        'mode',
        choices=['train', 'backtest', 'full', 'demo', 'test-data'],
        nargs='?',
        default='demo',
        help='Mode to run'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/final_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--candles',
        type=int,
        default=8000,
        help='Number of candles for training'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--fake-data',
        action='store_true',
        help='Use fake data instead of real Binance data'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.data.symbol = args.symbol
    config.data.use_real_data = not args.fake_data
    
    if args.device:
        config.training.device = args.device
    
    # Print configuration
    print("\n=== Configuration ===")
    print(f"Symbol: {config.data.symbol}")
    print(f"Interval: {config.data.interval}")
    print(f"Lookback: {config.data.lookback_window}")
    print(f"Device: {config.training.device}")
    print(f"Use Real Data: {config.data.use_real_data}")
    
    # Check CUDA availability
    if config.training.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        config.training.device = "cpu"
    
    # Run selected mode
    if args.mode == 'test-data':
        test_binance_data(config)
    
    elif args.mode == 'train':
        run_training(config, n_candles=args.candles)
    
    elif args.mode == 'backtest':
        run_backtest(config, checkpoint_path=args.checkpoint)
    
    elif args.mode == 'full':
        run_full_pipeline(config)
    
    elif args.mode == 'demo':
        quick_demo(config)


if __name__ == "__main__":
    main()
