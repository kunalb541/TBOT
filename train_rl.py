"""
Reinforcement Learning Training for Crypto Trading
===================================================
Train transformer-based policy using PPO to maximize trading profit.

This is the KEY difference from classification:
- Classification optimizes: accuracy on predicting labels
- RL optimizes: ACTUAL PROFIT (the reward function)

Expected improvement over 13.71 Sharpe baseline!
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor

from config import Config, DEFAULT_CONFIG
from data import CryptoDataFetcher, compute_technical_indicators, FEATURE_COLS
from trading_env import TradingEnv
from transformer_policy import TransformerActorCriticPolicy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading-specific metrics during training.
    
    Logs:
    - Episode returns
    - Number of trades per episode
    - Win rate
    - Sharpe ratio
    
    Compatible with both DummyVecEnv and SubprocVecEnv.
    """
    
    def __init__(self, verbose: int = 0, log_freq: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_returns = []
        self.episode_trades = []
        self.episode_steps = []
    
    def _on_step(self) -> bool:
        # Get infos from the rollout
        # For SubprocVecEnv, we need to get metrics from the info dict
        # which is populated when episodes end
        
        # Check if we have episode info in the locals
        infos = self.locals.get('infos', [])
        
        if not infos:
            return True
        
        # Process each info dict (one per parallel environment)
        for info in infos:
            # Check if episode is done (SB3 adds episode info to info dict)
            if 'episode' in info:
                # Standard SB3 episode stats
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                # Log to tensorboard
                self.logger.record('rollout/ep_rew_mean', episode_reward)
                self.logger.record('rollout/ep_len_mean', episode_length)
            
            # Check for custom trading metrics
            # These should be added by the environment when episode ends
            if 'balance' in info:
                balance = info['balance']
                initial_balance = info.get('initial_balance', 10000.0)
                episode_return = (balance - initial_balance) / initial_balance
                
                self.episode_returns.append(episode_return)
                
                # Log trading metrics
                self.logger.record('trading/episode_return_pct', episode_return * 100)
                self.logger.record('trading/final_balance', balance)
            
            # Log trade statistics if available
            if 'num_trades' in info:
                n_trades = info['num_trades']
                self.episode_trades.append(n_trades)
                self.logger.record('trading/num_trades', n_trades)
            
            if 'win_rate' in info and info.get('num_trades', 0) > 0:
                self.logger.record('trading/win_rate', info['win_rate'])
            
            if 'avg_trade_pnl' in info and info.get('num_trades', 0) > 0:
                self.logger.record('trading/avg_trade_pnl', info['avg_trade_pnl'])
            
            # Calculate Sharpe ratio over recent episodes
            if len(self.episode_returns) >= 30:
                recent_returns = self.episode_returns[-30:]
                if np.std(recent_returns) > 0:
                    sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                    self.logger.record('trading/sharpe_ratio', sharpe)
        
        return True


class ProgressCallback(BaseCallback):
    """Callback to print training progress."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Log progress
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Progress - Step {self.n_calls}")
            logger.info(f"{'='*60}")
            
            # Get recent episode rewards
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                
                logger.info(f"Mean episode reward: {mean_reward:.4f}")
                logger.info(f"Mean episode length: {mean_length:.1f}")
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    logger.info(f"NEW BEST! Mean reward: {mean_reward:.4f}")
        
        return True


def create_trading_env(
    df,
    feature_cols: list,
    config: Config,
    initial_balance: float = 10000.0,
    lookback_window: int = 200,
    reward_scaling: float = 1.0
) -> gym.Env:
    """
    Create and wrap a trading environment.
    
    Args:
        df: Market data DataFrame
        feature_cols: Feature column names
        config: Configuration
        initial_balance: Starting capital
        lookback_window: Sequence length
        reward_scaling: Scale rewards
        
    Returns:
        Wrapped environment
    """
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        config=config,
        initial_balance=initial_balance,
        lookback_window=lookback_window,
        reward_scaling=reward_scaling
    )
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    return env


def prepare_data(
    config: Config,
    symbol: str,
    interval: str,
    n_candles: int = 10000
) -> tuple:
    """
    Prepare training and evaluation data.
    
    Returns:
        train_df, val_df, test_df, feature_cols, scaler_stats
    """
    logger.info(f"\n=== Preparing Data ===")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Candles: {n_candles}")
    
    # Fetch data
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol=symbol,
        interval=interval,
        n_candles=n_candles
    )
    
    # Compute indicators
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    logger.info(f"Total samples after indicators: {len(df)}")
    
    # Get feature columns
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    logger.info(f"Features: {len(feature_cols)}")
    
    # Normalize features (using full dataset for RL - we're not predicting future!)
    features = df[feature_cols].values
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    
    # Normalize
    df[feature_cols] = (features - mean) / std
    
    # Split: train 70%, val 15%, test 15% (temporal order preserved)
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Train: {len(train_df)} candles")
    logger.info(f"Val: {len(val_df)} candles")
    logger.info(f"Test: {len(test_df)} candles")
    
    # Save scaler stats
    scaler_stats = {'mean': mean, 'std': std}
    
    return train_df, val_df, test_df, feature_cols, scaler_stats


def train_rl(
    config: Config,
    symbol: str = 'BNBUSDT',
    interval: str = '1d',
    n_candles: int = 2000,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    save_dir: str = 'rl_models',
    tensorboard_log: str = 'tensorboard_logs',
    checkpoint_freq: int = 50_000,
    eval_freq: int = 10_000
) -> PPO:
    """
    Train RL agent using PPO.
    
    Args:
        config: Configuration
        symbol: Trading symbol
        interval: Candle interval
        n_candles: Number of candles to use
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        tensorboard_log: TensorBoard log directory
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        
    Returns:
        Trained PPO model
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRANSFORMER + RL TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"Goal: Beat {symbol} classification baseline")
    logger.info(f"Baseline Sharpe: 13.71 (BNB 1d)")
    logger.info(f"{'='*60}\n")
    
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Prepare data
    train_df, val_df, test_df, feature_cols, scaler_stats = prepare_data(
        config, symbol, interval, n_candles
    )
    
    # Save scaler stats
    scaler_path = save_dir / f'{symbol}_{interval}_scaler.npz'
    np.savez(scaler_path, mean=scaler_stats['mean'], std=scaler_stats['std'])
    logger.info(f"Saved scaler to: {scaler_path}")
    
    # Create training environments
    logger.info(f"\n=== Creating Environments ===")
    logger.info(f"Parallel environments: {n_envs}")
    
    def make_env(df, rank):
        def _init():
            return create_trading_env(
                df=df,
                feature_cols=feature_cols,
                config=config,
                initial_balance=10000.0,
                lookback_window=config.data.lookback_window,
                reward_scaling=0.01  # Scale rewards to ~[-1, 1] range
            )
        return _init
    
    # Create vectorized environments
    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(train_df, i) for i in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env(train_df, 0)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(val_df, 0)])
    
    logger.info(f"Environments created successfully")
    
    # Create policy
    logger.info(f"\n=== Creating Policy ===")
    logger.info(f"Architecture: Transformer (d_model=128, layers=2, heads=4)")
    logger.info(f"Same architecture that achieved 13.71 Sharpe!")
    
    policy_kwargs = dict(
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        net_arch=[]  # We use custom feature extractor
    )
    
    # Create PPO model
    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,  # Collect 2048 steps before update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.2,  # PPO clip range
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Entropy coefficient (exploration)
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.seed,
        device='auto'
    )
    
    logger.info(f"Model created")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Setup callbacks
    logger.info(f"\n=== Setting up Callbacks ===")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / 'best_model'),
        log_path=str(save_dir / 'eval_logs'),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_dir / 'checkpoints'),
        name_prefix=f'{symbol}_{interval}_rl_model',
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # Trading metrics callback
    metrics_callback = TradingMetricsCallback(verbose=1, log_freq=1)
    
    # Progress callback
    progress_callback = ProgressCallback(check_freq=10000, verbose=1)
    
    # Combine callbacks
    callback = CallbackList([
        eval_callback,
        checkpoint_callback,
        metrics_callback,
        progress_callback
    ])
    
    # Train!
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"This will optimize DIRECTLY for profit!")
    logger.info(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            tb_log_name=f'{symbol}_{interval}_ppo',
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING COMPLETE!")
        logger.info(f"{'='*60}")
        
        # Save final model
        final_model_path = save_dir / f'{symbol}_{interval}_final.zip'
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Save training info
        training_info = {
            'symbol': symbol,
            'interval': interval,
            'n_candles': n_candles,
            'total_timesteps': total_timesteps,
            'feature_cols': feature_cols,
            'config': {
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'lookback_window': config.data.lookback_window
            },
            'timestamp': datetime.now().isoformat()
        }
        
        info_path = save_dir / f'{symbol}_{interval}_info.json'
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        logger.info(f"Training info saved to: {info_path}")
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        # Save current model
        interrupt_path = save_dir / f'{symbol}_{interval}_interrupted.zip'
        model.save(interrupt_path)
        logger.info(f"Model saved to: {interrupt_path}")
    
    finally:
        # Clean up
        train_env.close()
        eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train RL trading agent')
    parser.add_argument('--symbol', type=str, default='BNBUSDT',
                       help='Trading symbol (default: BNBUSDT - our best performer!)')
    parser.add_argument('--interval', type=str, default='1d',
                       help='Candle interval (default: 1d - proven to work)')
    parser.add_argument('--candles', type=int, default=2000,
                       help='Number of candles for training')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--save-dir', type=str, default='rl_models',
                       help='Directory to save models')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--fake-data', action='store_true',
                       help='Use fake data instead of real Binance data')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.data.symbol = args.symbol
    config.data.interval = args.interval
    config.data.use_real_data = not args.fake_data
    
    if args.device:
        config.training.device = args.device
    
    # Print configuration
    logger.info(f"\n=== RL Training Configuration ===")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Candles: {args.candles}")
    logger.info(f"Total timesteps: {args.timesteps:,}")
    logger.info(f"Parallel envs: {args.n_envs}")
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Use real data: {config.data.use_real_data}")
    
    # Train
    model = train_rl(
        config=config,
        symbol=args.symbol,
        interval=args.interval,
        n_candles=args.candles,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"NEXT STEPS:")
    logger.info(f"1. Run backtest: python backtest_rl.py --model {args.save_dir}/{args.symbol}_{args.interval}_final.zip")
    logger.info(f"2. Compare to baseline (13.71 Sharpe)")
    logger.info(f"3. If better, train on other symbols!")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
