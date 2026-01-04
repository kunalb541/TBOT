"""
Backtest RL Trading Agent
==========================
Evaluate trained RL model on test data and compare to baseline.

Target: Beat 13.71 Sharpe (BNB 1d classification baseline)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO

from config import Config, DEFAULT_CONFIG
from data import CryptoDataFetcher, compute_technical_indicators, FEATURE_COLS
from trading_env import TradingEnv
from backtest import Trade, BacktestResults, print_backtest_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RLBacktester:
    """
    Backtester for RL models.
    
    Runs the trained policy on test data and collects metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Config = DEFAULT_CONFIG
    ):
        """
        Args:
            model_path: Path to trained RL model (.zip file)
            config: Configuration object
        """
        self.config = config
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)
        logger.info(f"Model loaded successfully")
        
        # Load training info
        model_dir = Path(model_path).parent
        info_files = list(model_dir.glob('*_info.json'))
        
        if info_files:
            with open(info_files[0], 'r') as f:
                self.training_info = json.load(f)
            logger.info(f"Loaded training info: {info_files[0]}")
        else:
            logger.warning("No training info file found")
            self.training_info = {}
    
    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        initial_balance: float = 10000.0,
        deterministic: bool = True
    ) -> BacktestResults:
        """
        Run backtest on test data.
        
        Args:
            df: Test data DataFrame
            feature_cols: Feature column names
            scaler_mean: Feature means for normalization
            scaler_std: Feature stds for normalization
            initial_balance: Starting capital
            deterministic: Use deterministic policy
            
        Returns:
            BacktestResults with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING RL BACKTEST")
        logger.info(f"{'='*60}")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Candles: {len(df)}")
        logger.info(f"Initial capital: ${initial_balance:,.2f}")
        logger.info(f"Deterministic: {deterministic}")
        
        # Normalize features
        df_normalized = df.copy()
        df_normalized[feature_cols] = (df[feature_cols].values - scaler_mean) / (scaler_std + 1e-10)
        
        # Create environment
        env = TradingEnv(
            df=df_normalized,
            feature_cols=feature_cols,
            config=self.config,
            initial_balance=initial_balance,
            lookback_window=self.config.data.lookback_window,
            reward_scaling=0.01
        )
        
        # Run episode
        obs, info = env.reset(options={'start_step': env.lookback_window - 1})
        
        episode_reward = 0
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Get action from policy
            action, _states = self.model.predict(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, done, truncated, info = env.step(int(action))
            episode_reward += reward
            step += 1
            
            # Progress logging
            if step % 100 == 0:
                logger.info(f"Step {step}: Balance=${env.balance:,.2f}, Trades={len(env.trades)}")
        
        logger.info(f"\nBacktest complete!")
        logger.info(f"Final balance: ${env.balance:,.2f}")
        logger.info(f"Total trades: {len(env.trades)}")
        
        # Convert to BacktestResults format
        results = self._create_results(env, initial_balance, df)
        
        return results
    
    def _create_results(
        self,
        env: TradingEnv,
        initial_capital: float,
        df: pd.DataFrame
    ) -> BacktestResults:
        """Create BacktestResults from environment state."""
        final_capital = env.balance
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        trades = env.trades
        total_trades = len(trades)
        
        # Trade statistics
        if total_trades > 0:
            pnls = [t['pnl'] for t in trades]
            winning_trades = sum(1 for pnl in pnls if pnl > 0)
            losing_trades = sum(1 for pnl in pnls if pnl <= 0)
            win_rate = winning_trades / total_trades
            
            avg_trade_pnl = np.mean(pnls)
            
            winning_pnls = [pnl for pnl in pnls if pnl > 0]
            losing_pnls = [pnl for pnl in pnls if pnl < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
            
            total_wins = sum(winning_pnls) if winning_pnls else 0.0
            total_losses = abs(sum(losing_pnls)) if losing_pnls else 1.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            avg_trade_pnl = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        # Calculate Sharpe ratio (daily returns)
        # Create equity curve
        equity_curve = [initial_capital]
        running_balance = initial_capital
        
        for trade in trades:
            running_balance += trade['pnl']
            equity_curve.append(running_balance)
        
        if len(equity_curve) > 1:
            equity_array = np.array(equity_curve)
            returns = np.diff(equity_array) / equity_array[:-1]
            
            if len(returns) > 1 and np.std(returns) > 0:
                # Annualized Sharpe (assuming daily data)
                sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
            else:
                sharpe_ratio = 0.0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(negative_returns)
            else:
                sortino_ratio = 0.0
            
            # Max drawdown
            peak = np.maximum.accumulate(equity_array)
            drawdown = (peak - equity_array) / peak
            max_drawdown_pct = np.max(drawdown) * 100
            max_drawdown = np.max(peak - equity_array)
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
        
        # Create equity DataFrame
        timestamps = [df.index[0]] + [df.index[min(i, len(df)-1)] for i in range(len(trades))]
        equity_df = pd.DataFrame({
            'timestamp': timestamps[:len(equity_curve)],
            'equity': equity_curve
        })
        
        # Convert env trades to Trade objects
        trade_objects = []
        for t in trades:
            trade_obj = Trade(
                entry_time=df.index[0],  # Placeholder
                entry_price=t['entry_price'],
                exit_time=df.index[0],  # Placeholder
                exit_price=t['exit_price'],
                position_type=None,  # Will be determined from position_type string
                size=t['size'],
                pnl=t['pnl']
            )
            trade_objects.append(trade_obj)
        
        return BacktestResults(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=0.0,  # Not tracked in RL env
            trades=trade_objects,
            equity_curve=equity_df
        )


def compare_to_baseline(
    rl_results: BacktestResults,
    baseline_path: str = None
):
    """
    Compare RL results to classification baseline.
    
    Args:
        rl_results: RL backtest results
        baseline_path: Path to baseline results JSON
    """
    print("\n" + "="*60)
    print("COMPARISON: RL vs Classification Baseline")
    print("="*60)
    
    # RL results
    print(f"\n--- RL Model ---")
    print(f"Total Return:    {rl_results.total_return_pct:>12.2f}%")
    print(f"Sharpe Ratio:    {rl_results.sharpe_ratio:>12.2f}")
    print(f"Win Rate:        {rl_results.win_rate*100:>11.2f}%")
    print(f"Total Trades:    {rl_results.total_trades:>12}")
    print(f"Profit Factor:   {rl_results.profit_factor:>12.2f}")
    
    # Load baseline if available
    if baseline_path and Path(baseline_path).exists():
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        print(f"\n--- Classification Baseline ---")
        print(f"Total Return:    {baseline.get('total_return_pct', 0):>12.2f}%")
        print(f"Sharpe Ratio:    {baseline.get('sharpe_ratio', 0):>12.2f}")
        print(f"Win Rate:        {baseline.get('win_rate', 0)*100:>11.2f}%")
        print(f"Total Trades:    {baseline.get('total_trades', 0):>12}")
        
        print(f"\n--- Improvement ---")
        sharpe_improvement = rl_results.sharpe_ratio - baseline.get('sharpe_ratio', 0)
        return_improvement = rl_results.total_return_pct - baseline.get('total_return_pct', 0)
        
        print(f"Sharpe Δ:        {sharpe_improvement:>+12.2f}")
        print(f"Return Δ:        {return_improvement:>+11.2f}%")
        
        if rl_results.sharpe_ratio > baseline.get('sharpe_ratio', 0):
            print(f"\n🎉 RL MODEL IS BETTER! 🎉")
        else:
            print(f"\n⚠️  Baseline still better (needs tuning)")
    else:
        print(f"\n--- Target Baseline (BNB 1d) ---")
        print(f"Sharpe Ratio:    {13.71:>12.2f}")
        print(f"Total Return:    {12.17:>12.2f}%")
        
        print(f"\n--- Comparison ---")
        if rl_results.sharpe_ratio > 13.71:
            print(f"🎉 BEAT THE BASELINE! 🎉")
            print(f"Improvement: +{rl_results.sharpe_ratio - 13.71:.2f} Sharpe")
        else:
            print(f"Gap to baseline: {13.71 - rl_results.sharpe_ratio:.2f} Sharpe")
            print(f"Keep tuning hyperparameters!")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Backtest RL trading agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip)')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Trading symbol (if different from training)')
    parser.add_argument('--interval', type=str, default=None,
                       help='Candle interval (if different from training)')
    parser.add_argument('--candles', type=int, default=3000,
                       help='Number of candles for testing')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline results JSON for comparison')
    parser.add_argument('--save-results', type=str, default='backtest_rl_results.json',
                       help='Path to save results')
    parser.add_argument('--fake-data', action='store_true',
                       help='Use fake data')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.data.use_real_data = not args.fake_data
    
    # Load model and get training info
    backtester = RLBacktester(args.model, config)
    
    # Get symbol/interval from training info or args
    symbol = args.symbol or backtester.training_info.get('symbol', 'BNBUSDT')
    interval = args.interval or backtester.training_info.get('interval', '1d')
    feature_cols = backtester.training_info.get('feature_cols', FEATURE_COLS)
    
    config.data.symbol = symbol
    config.data.interval = interval
    
    logger.info(f"\n=== Backtest Configuration ===")
    logger.info(f"Model: {args.model}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Test candles: {args.candles}")
    
    # Load scaler stats
    model_dir = Path(args.model).parent
    scaler_files = list(model_dir.glob('*_scaler.npz'))
    
    if scaler_files:
        scaler_data = np.load(scaler_files[0])
        scaler_mean = scaler_data['mean']
        scaler_std = scaler_data['std']
        logger.info(f"Loaded scaler from: {scaler_files[0]}")
    else:
        logger.warning("No scaler file found, will compute from data")
        scaler_mean = None
        scaler_std = None
    
    # Fetch test data
    from data import CryptoDataFetcher
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(symbol, interval, n_candles=args.candles)
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    # Use last 15% as test (matching train.py split)
    test_start = int(len(df) * 0.85)
    test_df = df.iloc[test_start:]
    
    logger.info(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Compute scaler if not loaded
    if scaler_mean is None:
        features = df[feature_cols].values
        train_end = int(len(features) * 0.7)
        scaler_mean = features[:train_end].mean(axis=0)
        scaler_std = features[:train_end].std(axis=0)
        scaler_std = np.where(scaler_std < 1e-10, 1.0, scaler_std)
    
    # Run backtest
    results = backtester.run(
        df=test_df,
        feature_cols=feature_cols,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        initial_balance=10000.0,
        deterministic=True
    )
    
    # Print results
    print_backtest_results(results)
    
    # Compare to baseline
    compare_to_baseline(results, args.baseline)
    
    # Save results
    if args.save_results:
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model,
            'symbol': symbol,
            'interval': interval,
            'initial_capital': results.initial_capital,
            'final_capital': results.final_capital,
            'total_return_pct': results.total_return_pct,
            'total_trades': results.total_trades,
            'win_rate': results.win_rate,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown_pct': results.max_drawdown_pct,
            'profit_factor': results.profit_factor,
            'avg_trade_pnl': results.avg_trade_pnl
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"\n✓ Results saved to: {args.save_results}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"{'='*60}")
    
    # Print verdict
    if results.sharpe_ratio > 13.71:
        logger.info(f"\n✅ SUCCESS! Beat baseline by {results.sharpe_ratio - 13.71:.2f} Sharpe!")
        logger.info(f"Next: Train on ETH and other symbols")
    elif results.sharpe_ratio > 10.0:
        logger.info(f"\n📊 Good performance ({results.sharpe_ratio:.2f} Sharpe)")
        logger.info(f"Close to baseline, try tuning hyperparameters")
    else:
        logger.info(f"\n⚠️  Needs improvement ({results.sharpe_ratio:.2f} Sharpe)")
        logger.info(f"Suggestions:")
        logger.info(f"  - Increase training timesteps")
        logger.info(f"  - Adjust reward scaling")
        logger.info(f"  - Tune PPO hyperparameters")


if __name__ == "__main__":
    main()
