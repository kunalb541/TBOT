"""
Trading Environment for Reinforcement Learning
===============================================
Gymnasium-compatible environment for cryptocurrency trading.

This environment:
- Provides realistic trading simulation with fees and slippage
- Rewards profitable trades directly (not classification accuracy)
- Maintains strict temporal ordering (no lookahead)
- Supports HOLD, LONG, and SHORT actions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces

from config import Config, DEFAULT_CONFIG


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    LONG = 1
    SHORT = 2


class Position(IntEnum):
    """Current position state."""
    FLAT = 0
    LONG = 1
    SHORT = 2


class TradingEnv(gym.Env):
    """
    Cryptocurrency Trading Environment for RL.
    
    State Space:
        - Sequence of OHLCV + technical indicators (lookback_window, n_features)
        - Current position (FLAT, LONG, SHORT)
        - Entry price (if in position)
        - Unrealized P&L (if in position)
    
    Action Space:
        - 0: HOLD (close position if open, otherwise stay flat)
        - 1: LONG (open long if flat, close short if short, hold if already long)
        - 2: SHORT (open short if flat, close long if long, hold if already short)
    
    Reward:
        - Realized profit/loss after fees when closing a position
        - Small penalty for holding to encourage action
        - Risk penalty for large drawdowns
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        config: Config = DEFAULT_CONFIG,
        initial_balance: float = 10000.0,
        lookback_window: int = 200,
        reward_scaling: float = 1.0,
        fee_penalty_multiplier: float = 1.0
    ):
        """
        Initialize trading environment.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            feature_cols: List of feature column names
            config: Configuration object
            initial_balance: Starting capital
            lookback_window: Number of historical candles in state
            reward_scaling: Scale rewards for better learning
            fee_penalty_multiplier: Multiply fee costs in reward
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.config = config
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.fee_penalty_multiplier = fee_penalty_multiplier
        
        # Extract feature matrix
        self.features = df[feature_cols].values
        self.prices = df['close'].values  # For stop loss/take profit checks
        self.opens = df['open'].values  # For realistic trade execution
        self.highs = df['high'].values
        self.lows = df['low'].values
        
        # Validate data
        if len(self.features) < lookback_window + 1:
            raise ValueError(
                f"Need at least {lookback_window + 1} samples, got {len(self.features)}"
            )
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # HOLD, LONG, SHORT
        
        # Observation: (lookback_window, n_features) + position info
        n_features = len(feature_cols)
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(lookback_window, n_features),
                dtype=np.float32
            ),
            'position': spaces.Discrete(3),  # FLAT, LONG, SHORT
            'position_value': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            )
        })
        
        # Initialize episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.position_size = 0.0
        self.total_reward = 0.0
        self.trades = []
        self.episode_start_balance = initial_balance
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation."""
        # Market data: lookback window ending at current step
        start_idx = self.current_step - self.lookback_window + 1
        end_idx = self.current_step + 1
        market_data = self.features[start_idx:end_idx].astype(np.float32)
        
        # Position information
        position_value = 0.0
        if self.position != Position.FLAT:
            current_price = self.prices[self.current_step]
            if self.position == Position.LONG:
                position_value = (current_price - self.entry_price) * self.position_size
            else:  # SHORT
                position_value = (self.entry_price - current_price) * self.position_size
        
        return {
            'market_data': market_data,
            'position': int(self.position),
            'position_value': np.array([position_value], dtype=np.float32)
        }
    
    def _calculate_slippage(self, price: float, is_entry: bool, is_long: bool) -> float:
        """Apply slippage to execution price."""
        slippage = price * self.config.trading.slippage_pct
        
        if is_entry:
            return price + slippage if is_long else price - slippage
        else:
            return price - slippage if is_long else price + slippage
    
    def _calculate_fee(self, price: float, size: float) -> float:
        """Calculate trading fee."""
        return price * size * self.config.trading.taker_fee
    
    def _open_position(
        self,
        position_type: Position,
        price: float
    ) -> Tuple[float, Dict]:
        """
        Open a new position.
        
        Returns:
            reward: Immediate reward (negative due to fees)
            info: Additional information
        """
        is_long = position_type == Position.LONG
        
        # Calculate position size (use a fraction of balance)
        capital_for_trade = self.balance * self.config.trading.position_size
        
        # Apply slippage
        exec_price = self._calculate_slippage(price, is_entry=True, is_long=is_long)
        
        # Calculate size after fees
        fee = self._calculate_fee(exec_price, capital_for_trade / exec_price)
        position_size = (capital_for_trade - fee) / exec_price
        
        # Update state
        self.position = position_type
        self.entry_price = exec_price
        self.position_size = position_size
        
        # Penalty for fees (encourage fewer trades)
        reward = -fee * self.fee_penalty_multiplier * self.reward_scaling
        
        info = {
            'action': 'open',
            'position_type': 'LONG' if is_long else 'SHORT',
            'entry_price': exec_price,
            'size': position_size,
            'fee': fee
        }
        
        return reward, info
    
    def _close_position(self, price: float) -> Tuple[float, Dict]:
        """
        Close current position.
        
        Returns:
            reward: Realized profit/loss (THE KEY REWARD SIGNAL!)
            info: Additional information
        """
        is_long = self.position == Position.LONG
        
        # Apply slippage
        exec_price = self._calculate_slippage(price, is_entry=False, is_long=is_long)
        
        # Calculate P&L
        if is_long:
            gross_pnl = (exec_price - self.entry_price) * self.position_size
        else:
            gross_pnl = (self.entry_price - exec_price) * self.position_size
        
        # Subtract exit fee
        exit_fee = self._calculate_fee(exec_price, self.position_size)
        net_pnl = gross_pnl - exit_fee
        
        # Update balance
        self.balance += net_pnl
        
        # Record trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': exec_price,
            'position_type': 'LONG' if is_long else 'SHORT',
            'pnl': net_pnl,
            'size': self.position_size
        })
        
        # Reward is the actual profit/loss (this is what we want to maximize!)
        reward = net_pnl * self.reward_scaling
        
        info = {
            'action': 'close',
            'position_type': 'LONG' if is_long else 'SHORT',
            'exit_price': exec_price,
            'pnl': net_pnl,
            'balance': self.balance
        }
        
        # Reset position
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.position_size = 0.0
        
        return reward, info
    
    def _check_stop_loss_take_profit(self) -> Tuple[bool, float, str]:
        """
        Check if stop loss or take profit is hit.
        
        Returns:
            hit: Whether SL/TP was hit
            exit_price: Price at which to exit
            reason: 'stop_loss' or 'take_profit'
        """
        if self.position == Position.FLAT:
            return False, 0.0, ''
        
        high = self.highs[self.current_step]
        low = self.lows[self.current_step]
        
        if self.position == Position.LONG:
            stop_loss = self.entry_price * (1 - self.config.trading.stop_loss_pct)
            take_profit = self.entry_price * (1 + self.config.trading.take_profit_pct)
            
            if low <= stop_loss:
                return True, stop_loss, 'stop_loss'
            if high >= take_profit:
                return True, take_profit, 'take_profit'
        
        else:  # SHORT
            stop_loss = self.entry_price * (1 + self.config.trading.stop_loss_pct)
            take_profit = self.entry_price * (1 - self.config.trading.take_profit_pct)
            
            if high >= stop_loss:
                return True, stop_loss, 'stop_loss'
            if low <= take_profit:
                return True, take_profit, 'take_profit'
        
        return False, 0.0, ''
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        ANTI-LOOKAHEAD MEASURE:
        - Model sees data up to current_step (t)
        - Decision made based on that data
        - Execution happens at current_step+1 open price (t+1)
        
        Args:
            action: 0=HOLD, 1=LONG, 2=SHORT
            
        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        action = Action(action)
        current_price = self.prices[self.current_step]  # For stop loss checks only
        reward = 0.0
        info = {'step': self.current_step}
        
        # Check if we can execute (need next candle for execution)
        if self.current_step + 1 >= len(self.opens):
            # End of data
            terminated = True
            truncated = False
            observation = self._get_observation()
            info['termination_reason'] = 'end_of_data'
            
            # Add final trading metrics
            info['balance'] = self.balance
            info['total_return'] = (self.balance - self.episode_start_balance) / self.episode_start_balance
            info['initial_balance'] = self.episode_start_balance
            info['num_trades'] = len(self.trades)
            
            if self.trades:
                pnls = [t['pnl'] for t in self.trades]
                winning = sum(1 for pnl in pnls if pnl > 0)
                info['win_rate'] = winning / len(self.trades) if self.trades else 0.0
                info['avg_trade_pnl'] = np.mean(pnls) if pnls else 0.0
            
            return observation, reward, terminated, truncated, info
        
        # CRITICAL: Execute at NEXT candle's open (no lookahead!)
        execution_price = self.opens[self.current_step + 1]
        
        # Check stop loss / take profit FIRST (use intrabar highs/lows)
        sl_tp_hit, exit_price, reason = self._check_stop_loss_take_profit()
        if sl_tp_hit:
            reward, close_info = self._close_position(exit_price)
            info.update(close_info)
            info['sl_tp_reason'] = reason
        
        # Execute action at NEXT candle's open
        if action == Action.HOLD:
            if self.position != Position.FLAT:
                # Close position at next candle's open
                close_reward, close_info = self._close_position(execution_price)
                reward += close_reward
                info.update(close_info)
            else:
                # Small penalty for staying flat (encourage action)
                reward = -0.0001 * self.reward_scaling
                info['action'] = 'hold_flat'
        
        elif action == Action.LONG:
            if self.position == Position.SHORT:
                # Close short first at next candle's open
                close_reward, _ = self._close_position(execution_price)
                reward += close_reward
            
            if self.position == Position.FLAT:
                # Open long at next candle's open
                open_reward, open_info = self._open_position(Position.LONG, execution_price)
                reward += open_reward
                info.update(open_info)
            else:
                # Already long, small penalty
                reward = -0.0001 * self.reward_scaling
                info['action'] = 'already_long'
        
        elif action == Action.SHORT:
            if self.position == Position.LONG:
                # Close long first at next candle's open
                close_reward, _ = self._close_position(execution_price)
                reward += close_reward
            
            if self.position == Position.FLAT:
                # Open short at next candle's open
                open_reward, open_info = self._open_position(Position.SHORT, execution_price)
                reward += open_reward
                info.update(open_info)
            else:
                # Already short, small penalty
                reward = -0.0001 * self.reward_scaling
                info['action'] = 'already_short'
        
        # Move to next step
        self.current_step += 1
        self.total_reward += reward
        
        # Check if episode is done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        
        # Check if balance is too low (risk management)
        if self.balance < self.initial_balance * 0.5:
            terminated = True
            truncated = True
            reward -= 10.0 * self.reward_scaling  # Heavy penalty for losing 50%
            info['termination_reason'] = 'balance_too_low'
        
        # Get new observation
        observation = self._get_observation()
        
        # Add final info
        info['balance'] = self.balance
        info['total_return'] = (self.balance - self.episode_start_balance) / self.episode_start_balance
        
        # Add trading metrics when episode ends
        if terminated or truncated:
            info['initial_balance'] = self.episode_start_balance
            info['num_trades'] = len(self.trades)
            
            if self.trades:
                pnls = [t['pnl'] for t in self.trades]
                winning = sum(1 for pnl in pnls if pnl > 0)
                info['win_rate'] = winning / len(self.trades) if self.trades else 0.0
                info['avg_trade_pnl'] = np.mean(pnls) if pnls else 0.0
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset to random starting point (or fixed if testing)
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            # Random start (ensuring enough lookback and future)
            min_start = self.lookback_window - 1
            max_start = len(self.prices) - 500  # Leave at least 500 steps
            self.current_step = self.np_random.integers(min_start, max_start)
        
        self.balance = self.initial_balance
        self.position = Position.FLAT
        self.entry_price = 0.0
        self.position_size = 0.0
        self.total_reward = 0.0
        self.trades = []
        self.episode_start_balance = self.initial_balance
        
        observation = self._get_observation()
        info = {
            'start_step': self.current_step,
            'initial_balance': self.initial_balance
        }
        
        return observation, info
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Price: ${self.prices[self.current_step]:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {Position(self.position).name}")
            if self.position != Position.FLAT:
                print(f"Entry: ${self.entry_price:.2f}")
                print(f"Size: {self.position_size:.6f}")
            print(f"Total Trades: {len(self.trades)}")
            print(f"Total Reward: {self.total_reward:.4f}")
    
    def close(self):
        """Clean up environment."""
        pass


if __name__ == "__main__":
    # Test the environment
    from config import Config
    from data import CryptoDataFetcher, compute_technical_indicators, FEATURE_COLS
    
    config = Config()
    config.data.use_real_data = True
    
    # Fetch data
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol='BNBUSDT',
        interval='1d',
        n_candles=1000
    )
    
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    # Create environment
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        config=config,
        initial_balance=10000.0,
        lookback_window=200
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test random episode
    obs, info = env.reset()
    print(f"\nInitial observation keys: {obs.keys()}")
    print(f"Market data shape: {obs['market_data'].shape}")
    print(f"Position: {obs['position']}")
    
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode finished:")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final balance: ${env.balance:.2f}")
    print(f"Total trades: {len(env.trades)}")
    print(f"Return: {((env.balance - 10000) / 10000 * 100):.2f}%")
