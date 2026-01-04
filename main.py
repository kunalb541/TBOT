"""
Crypto Trading Bot - Main Entry Point
=======================================
Complete pipeline: Train → Backtest → Paper Trade

Usage:
    python main.py train        # Train the model
    python main.py backtest     # Run backtest
    python main.py paper        # Run paper trading
    python main.py full         # Full pipeline
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

from config import Config, DEFAULT_CONFIG
from data import (
    prepare_datasets,
    create_dataloaders,
    CryptoDataFetcher,
    compute_technical_indicators
)
from model import (
    CausalTimeSeriesTransformer,
    count_parameters,
    verify_causal_masking
)
from train import train, TrainingMetrics
from backtest import Backtester, print_backtest_results
from paper_trade import PaperTrader


def run_training(config: Config, n_candles: int = 8000):
    """
    Run model training.
    """
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    model, metrics = train(config, n_candles=n_candles)
    
    print("\n✓ Training complete!")
    print(f"  Best validation accuracy: {max(metrics.val_accuracies):.4f}")
    print(f"  Model saved to: checkpoints/")
    
    return model, metrics


def run_backtest(config: Config, checkpoint_path: str = "checkpoints/final_model.pt"):
    """
    Run backtesting on test data.
    """
    print("\n" + "="*60)
    print("BACKTESTING PHASE")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
    
    from config import ModelConfig
    model_config = ModelConfig(**checkpoint['config'])
    model = CausalTimeSeriesTransformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.training.device)
    model.eval()
    
    feature_names = checkpoint['feature_names']
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Features: {len(feature_names)}")
    
    # Fetch data for backtesting
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol=config.data.symbol,
        interval=config.data.interval,
        n_candles=3000  # Last 3000 candles for backtest
    )
    
    # Compute indicators
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    # Get scaler statistics from first 70% (simulating train data)
    available_features = [f for f in feature_names if f in df.columns]
    features = df[available_features].values
    train_end = int(len(features) * 0.7)
    scaler_mean = features[:train_end].mean(axis=0)
    scaler_std = features[:train_end].std(axis=0)
    
    # Run backtest on last 30% (test period)
    test_df = df.iloc[train_end:]
    
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


def run_paper_trading(
    config: Config,
    checkpoint_path: str = "checkpoints/final_model.pt",
    duration_minutes: int = 5
):
    """
    Run paper trading simulation.
    """
    print("\n" + "="*60)
    print("PAPER TRADING PHASE")
    print("="*60)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=config.training.device)
    
    from config import ModelConfig
    model_config = ModelConfig(**checkpoint['config'])
    model = CausalTimeSeriesTransformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.training.device)
    model.eval()
    
    feature_names = checkpoint['feature_names']
    
    print(f"Loaded model from: {checkpoint_path}")
    print(f"Paper trading duration: {duration_minutes} minutes")
    
    # Create paper trader
    trader = PaperTrader(
        model=model,
        config=config,
        feature_names=feature_names
    )
    
    # Run paper trading
    trader.run(
        duration_minutes=duration_minutes,
        prediction_interval=30  # Predict every 30 seconds
    )
    
    # Save session data
    session_data = trader.get_session_data()
    session_path = Path(f"paper_trading_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(session_path, 'w') as f:
        json.dump(session_data, f, indent=2, default=str)
    
    print(f"\n✓ Session data saved to: {session_path}")
    
    return session_data


def run_full_pipeline(config: Config):
    """
    Run the complete pipeline: Train → Backtest → Paper Trade
    """
    print("\n" + "#"*60)
    print("# CRYPTO TRADING BOT - FULL PIPELINE")
    print("#"*60)
    
    # Step 1: Training
    model, metrics = run_training(config, n_candles=8000)
    
    # Step 2: Backtesting
    results = run_backtest(config)
    
    # Step 3: Paper Trading (short demo)
    print("\nRunning paper trading demo for 2 minutes...")
    session_data = run_paper_trading(config, duration_minutes=2)
    
    print("\n" + "#"*60)
    print("# PIPELINE COMPLETE")
    print("#"*60)
    
    return model, results, session_data


def quick_demo(config: Config):
    """
    Quick demo without training - just shows the system works.
    """
    print("\n" + "="*60)
    print("QUICK DEMO - No Training")
    print("="*60)
    
    # Create random model
    feature_names = [
        'returns', 'log_returns', 'range_pct', 'body_pct',
        'upper_wick', 'lower_wick',
        'rsi_normalized', 'macd_normalized', 'macd_hist_normalized',
        'bb_position', 'bb_width', 'atr_normalized',
        'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'volume_normalized', 'volume_ratio',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_normalized'
    ]
    
    config.model.input_dim = len(feature_names)
    model = CausalTimeSeriesTransformer(config.model)
    model.to(config.training.device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Verify causal masking
    print("\nVerifying causal masking...")
    verify_causal_masking(model, seq_len=20)
    
    # Generate some fake data
    print("\nGenerating fake price data...")
    from data import generate_fake_ohlcv
    df = generate_fake_ohlcv(n_candles=1000)
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    print(f"Generated {len(df)} candles with {len(feature_names)} features")
    
    # Quick backtest
    print("\nRunning quick backtest...")
    available_features = [f for f in feature_names if f in df.columns]
    features = df[available_features].values
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    
    backtester = Backtester(config)
    results = backtester.run(model, df, available_features, mean, std)
    
    print_backtest_results(results)
    
    print("\n✓ Quick demo complete!")
    print("  To run full training, use: python main.py train")


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot with Causal Time Series Transformer"
    )
    parser.add_argument(
        'mode',
        choices=['train', 'backtest', 'paper', 'full', 'demo'],
        help='Mode to run: train, backtest, paper (paper trading), full (complete pipeline), demo (quick test)'
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
        '--duration',
        type=int,
        default=5,
        help='Paper trading duration in minutes'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.data.symbol = args.symbol
    
    # Print configuration
    print("\n=== Configuration ===")
    print(f"Symbol: {config.data.symbol}")
    print(f"Interval: {config.data.interval}")
    print(f"Lookback: {config.data.lookback_window}")
    print(f"Device: {config.training.device}")
    
    # Run selected mode
    if args.mode == 'train':
        run_training(config, n_candles=args.candles)
    
    elif args.mode == 'backtest':
        run_backtest(config, checkpoint_path=args.checkpoint)
    
    elif args.mode == 'paper':
        run_paper_trading(
            config,
            checkpoint_path=args.checkpoint,
            duration_minutes=args.duration
        )
    
    elif args.mode == 'full':
        run_full_pipeline(config)
    
    elif args.mode == 'demo':
        quick_demo(config)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default to demo mode if no arguments
        sys.argv.append('demo')
    
    main()
