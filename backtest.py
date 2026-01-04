"""
Backtesting Engine for Crypto Trading Bot
===========================================
Implements a realistic backtesting framework with:
- Strict temporal ordering (no lookahead)
- Realistic fee simulation
- Slippage modeling
- Risk management
- Performance metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import torch.nn as nn

from config import Config, DEFAULT_CONFIG
from model import CausalTimeSeriesTransformer
from data import CryptoDataFetcher, compute_technical_indicators


class PositionType(Enum):
    """Position type."""
    NONE = 0
    LONG = 1
    SHORT = 2


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    position_type: PositionType = PositionType.NONE
    size: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestState:
    """Current state of the backtest."""
    capital: float
    position: PositionType = PositionType.NONE
    position_size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Performance tracking
    peak_capital: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


@dataclass
class BacktestResults:
    """Backtest results and metrics."""
    # Basic metrics
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Average metrics
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time metrics
    avg_trade_duration: float  # in hours
    
    # Detailed data
    trades: List[Trade]
    equity_curve: pd.DataFrame


class Backtester:
    """
    Backtesting engine with strict anti-lookahead measures.
    
    ANTI-CHEATING MEASURES:
    1. Process candles in strict chronological order
    2. Decisions based only on data available at decision time
    3. Execute trades at NEXT candle open (not current close)
    4. Realistic fee and slippage simulation
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.state = None
    
    def _calculate_slippage(self, price: float, position_type: PositionType) -> float:
        """
        Apply slippage to execution price.
        Long entry / Short exit: price increases (worse fill)
        Short entry / Long exit: price decreases (worse fill)
        """
        slippage = price * self.config.trading.slippage_pct
        
        if position_type == PositionType.LONG:
            return price + slippage  # Worse fill for long entry
        else:
            return price - slippage  # Worse fill for short entry
    
    def _calculate_fee(self, price: float, size: float, is_taker: bool = True) -> float:
        """Calculate trading fee."""
        fee_rate = self.config.trading.taker_fee if is_taker else self.config.trading.maker_fee
        return price * size * fee_rate
    
    def _open_position(
        self,
        timestamp: datetime,
        price: float,
        position_type: PositionType
    ) -> bool:
        """Open a new position."""
        if self.state.position != PositionType.NONE:
            return False
        
        # Calculate position size
        capital_for_trade = self.state.capital * self.config.trading.position_size
        
        # Apply slippage
        exec_price = self._calculate_slippage(price, position_type)
        
        # Calculate size and fee
        fee = self._calculate_fee(exec_price, capital_for_trade / exec_price)
        position_size = (capital_for_trade - fee) / exec_price
        
        # Set stop loss and take profit
        if position_type == PositionType.LONG:
            stop_loss = exec_price * (1 - self.config.trading.stop_loss_pct)
            take_profit = exec_price * (1 + self.config.trading.take_profit_pct)
        else:
            stop_loss = exec_price * (1 + self.config.trading.stop_loss_pct)
            take_profit = exec_price * (1 - self.config.trading.take_profit_pct)
        
        # Update state
        self.state.position = position_type
        self.state.position_size = position_size
        self.state.entry_price = exec_price
        self.state.entry_time = timestamp
        self.state.stop_loss = stop_loss
        self.state.take_profit = take_profit
        
        return True
    
    def _close_position(
        self,
        timestamp: datetime,
        price: float,
        reason: str = "signal"
    ) -> Optional[Trade]:
        """Close current position."""
        if self.state.position == PositionType.NONE:
            return None
        
        # Apply slippage (opposite direction)
        opposite_type = (PositionType.SHORT if self.state.position == PositionType.LONG 
                        else PositionType.LONG)
        exec_price = self._calculate_slippage(price, opposite_type)
        
        # Calculate fees
        exit_fee = self._calculate_fee(exec_price, self.state.position_size)
        
        # Calculate PnL
        if self.state.position == PositionType.LONG:
            gross_pnl = (exec_price - self.state.entry_price) * self.state.position_size
        else:  # SHORT
            gross_pnl = (self.state.entry_price - exec_price) * self.state.position_size
        
        # Approximate entry fee (we stored the position size after fees)
        entry_value = self.state.entry_price * self.state.position_size
        entry_fee = entry_value * self.config.trading.taker_fee / (1 - self.config.trading.taker_fee)
        
        net_pnl = gross_pnl - exit_fee
        pnl_pct = net_pnl / (entry_value + entry_fee)
        
        # Create trade record
        trade = Trade(
            entry_time=self.state.entry_time,
            entry_price=self.state.entry_price,
            exit_time=timestamp,
            exit_price=exec_price,
            position_type=self.state.position,
            size=self.state.position_size,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )
        
        # Update capital
        self.state.capital += net_pnl
        
        # Reset position
        self.state.position = PositionType.NONE
        self.state.position_size = 0.0
        self.state.entry_price = 0.0
        self.state.entry_time = None
        
        return trade
    
    def _check_stop_loss_take_profit(
        self,
        timestamp: datetime,
        high: float,
        low: float
    ) -> Optional[Trade]:
        """Check if stop loss or take profit is hit."""
        if self.state.position == PositionType.NONE:
            return None
        
        if self.state.position == PositionType.LONG:
            # Check stop loss
            if low <= self.state.stop_loss:
                return self._close_position(timestamp, self.state.stop_loss, "stop_loss")
            # Check take profit
            if high >= self.state.take_profit:
                return self._close_position(timestamp, self.state.take_profit, "take_profit")
        
        else:  # SHORT
            # Check stop loss
            if high >= self.state.stop_loss:
                return self._close_position(timestamp, self.state.stop_loss, "stop_loss")
            # Check take profit
            if low <= self.state.take_profit:
                return self._close_position(timestamp, self.state.take_profit, "take_profit")
        
        return None
    
    def _update_equity(self, timestamp: datetime, price: float):
        """Update equity curve."""
        if self.state.position == PositionType.NONE:
            equity = self.state.capital
        else:
            if self.state.position == PositionType.LONG:
                unrealized_pnl = (price - self.state.entry_price) * self.state.position_size
            else:
                unrealized_pnl = (self.state.entry_price - price) * self.state.position_size
            equity = self.state.capital + unrealized_pnl
        
        self.state.equity_curve.append(equity)
        self.state.timestamps.append(timestamp)
        
        # Track peak for drawdown calculation
        if equity > self.state.peak_capital:
            self.state.peak_capital = equity
    
    def _signal_to_action(
        self,
        prediction: int,
        confidence: float
    ) -> Optional[PositionType]:
        """
        Convert model prediction to trading action.
        
        Args:
            prediction: 0=Hold, 1=Long, 2=Short
            confidence: Confidence score
            
        Returns:
            Position type to open, or None
        """
        # Check confidence threshold
        if confidence < self.config.trading.min_confidence:
            return None
        
        if prediction == 1:
            return PositionType.LONG
        elif prediction == 2:
            return PositionType.SHORT
        return None
    
    def run(
        self,
        model: nn.Module,
        df: pd.DataFrame,
        feature_cols: List[str],
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray
    ) -> BacktestResults:
        """
        Run backtest.
        
        CRITICAL: Processes data in STRICT chronological order.
        Each decision is based ONLY on data available at that time.
        
        Args:
            model: Trained model
            df: OHLCV DataFrame with computed indicators
            feature_cols: Feature column names
            scaler_mean: Feature mean for normalization
            scaler_std: Feature std for normalization
            
        Returns:
            BacktestResults object
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Initialize state
        self.state = BacktestState(
            capital=self.config.trading.initial_capital,
            peak_capital=self.config.trading.initial_capital
        )
        
        lookback = self.config.data.lookback_window
        
        print(f"\n=== Running Backtest ===")
        print(f"Initial capital: ${self.state.capital:,.2f}")
        print(f"Period: {df.index[lookback]} to {df.index[-1]}")
        print(f"Total candles: {len(df) - lookback}")
        
        # Process each candle chronologically
        for i in range(lookback, len(df)):
            timestamp = df.index[i]
            current_candle = df.iloc[i]
            
            # Get historical data UP TO current point (no lookahead!)
            history = df.iloc[i - lookback:i]
            
            # Extract features and normalize
            features = history[feature_cols].values
            features_normalized = (features - scaler_mean) / (scaler_std + 1e-10)
            
            # Check stop loss / take profit first
            sl_tp_trade = self._check_stop_loss_take_profit(
                timestamp,
                current_candle['high'],
                current_candle['low']
            )
            if sl_tp_trade:
                self.state.trades.append(sl_tp_trade)
            
            # Get model prediction
            with torch.no_grad():
                x = torch.FloatTensor(features_normalized).unsqueeze(0).to(device)
                pred, conf = model.predict(x)
                prediction = pred[0].item()
                confidence = conf[0].item()
            
            # Execute trading logic using NEXT candle open price
            # This simulates realistic execution
            if i + 1 < len(df):
                next_open = df.iloc[i + 1]['open']
                
                # If we have a position and signal changes, close it
                if self.state.position != PositionType.NONE:
                    action = self._signal_to_action(prediction, confidence)
                    should_close = (
                        prediction == 0 or  # Hold signal
                        (self.state.position == PositionType.LONG and prediction == 2) or
                        (self.state.position == PositionType.SHORT and prediction == 1)
                    )
                    if should_close:
                        trade = self._close_position(timestamp, next_open, "signal")
                        if trade:
                            self.state.trades.append(trade)
                
                # If no position, check for entry
                if self.state.position == PositionType.NONE:
                    action = self._signal_to_action(prediction, confidence)
                    if action is not None:
                        self._open_position(timestamp, next_open, action)
            
            # Update equity curve
            self._update_equity(timestamp, current_candle['close'])
            
            # Check max drawdown limit
            if self.state.equity_curve:
                current_dd = (self.state.peak_capital - self.state.equity_curve[-1]) / self.state.peak_capital
                if current_dd >= self.config.trading.max_drawdown_pct:
                    print(f"\nMax drawdown limit reached at {timestamp}")
                    if self.state.position != PositionType.NONE:
                        trade = self._close_position(timestamp, current_candle['close'], "max_drawdown")
                        if trade:
                            self.state.trades.append(trade)
                    break
        
        # Close any remaining position
        if self.state.position != PositionType.NONE:
            final_price = df.iloc[-1]['close']
            trade = self._close_position(df.index[-1], final_price, "end_of_backtest")
            if trade:
                self.state.trades.append(trade)
        
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> BacktestResults:
        """Calculate backtest metrics."""
        initial_capital = self.config.trading.initial_capital
        final_capital = self.state.capital
        
        # Basic returns
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Trade statistics
        trades = self.state.trades
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl <= 0])
            win_rate = winning_trades / total_trades
            
            pnls = [t.pnl for t in trades]
            avg_trade_pnl = np.mean(pnls)
            
            winning_pnls = [t.pnl for t in trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in trades if t.pnl < 0]
            
            avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
            avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
            
            total_wins = sum(winning_pnls) if winning_pnls else 0.0
            total_losses = abs(sum(losing_pnls)) if losing_pnls else 1.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Average trade duration
            durations = []
            for t in trades:
                if t.entry_time and t.exit_time:
                    duration = (t.exit_time - t.entry_time).total_seconds() / 3600  # hours
                    durations.append(duration)
            avg_trade_duration = np.mean(durations) if durations else 0.0
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            avg_trade_pnl = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            avg_trade_duration = 0.0
        
        # Risk metrics from equity curve
        equity_curve = np.array(self.state.equity_curve)
        
        if len(equity_curve) > 1:
            # Drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdown_pct = np.max(drawdown) * 100
            max_drawdown = np.max(peak - equity_curve)
            
            # Returns for Sharpe/Sortino
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            # Sharpe ratio (annualized, assuming hourly data)
            if np.std(returns) > 0:
                sharpe_ratio = np.sqrt(24 * 365) * np.mean(returns) / np.std(returns)
            else:
                sharpe_ratio = 0.0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                sortino_ratio = np.sqrt(24 * 365) * np.mean(returns) / np.std(negative_returns)
            else:
                sortino_ratio = 0.0
        else:
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame({
            'timestamp': self.state.timestamps,
            'equity': self.state.equity_curve
        })
        
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
            avg_trade_duration=avg_trade_duration,
            trades=trades,
            equity_curve=equity_df
        )


def print_backtest_results(results: BacktestResults):
    """Print formatted backtest results."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    print(f"\n--- Capital ---")
    print(f"Initial Capital:     ${results.initial_capital:>12,.2f}")
    print(f"Final Capital:       ${results.final_capital:>12,.2f}")
    print(f"Total Return:        ${results.total_return:>12,.2f} ({results.total_return_pct:+.2f}%)")
    
    print(f"\n--- Trade Statistics ---")
    print(f"Total Trades:        {results.total_trades:>12}")
    print(f"Winning Trades:      {results.winning_trades:>12}")
    print(f"Losing Trades:       {results.losing_trades:>12}")
    print(f"Win Rate:            {results.win_rate*100:>11.2f}%")
    print(f"Profit Factor:       {results.profit_factor:>12.2f}")
    
    print(f"\n--- Average Trade Metrics ---")
    print(f"Avg Trade PnL:       ${results.avg_trade_pnl:>12,.2f}")
    print(f"Avg Win:             ${results.avg_win:>12,.2f}")
    print(f"Avg Loss:            ${results.avg_loss:>12,.2f}")
    print(f"Avg Duration:        {results.avg_trade_duration:>11.1f}h")
    
    print(f"\n--- Risk Metrics ---")
    print(f"Max Drawdown:        ${results.max_drawdown:>12,.2f} ({results.max_drawdown_pct:.2f}%)")
    print(f"Sharpe Ratio:        {results.sharpe_ratio:>12.2f}")
    print(f"Sortino Ratio:       {results.sortino_ratio:>12.2f}")
    
    print("\n" + "="*60)
    
    # Trade breakdown by exit reason
    if results.trades:
        reasons = {}
        for t in results.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        
        print("\n--- Exit Reasons ---")
        for reason, count in sorted(reasons.items()):
            print(f"{reason:20s}: {count}")


def run_backtest_from_checkpoint(
    checkpoint_path: str,
    config: Config = DEFAULT_CONFIG,
    n_candles: int = 3000
) -> BacktestResults:
    """
    Run backtest using a saved model checkpoint.
    """
    import torch
    from model import CausalTimeSeriesTransformer
    from data import CryptoDataFetcher, compute_technical_indicators
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model_config = checkpoint['config']
    feature_names = checkpoint['feature_names']
    
    # Create model
    from config import ModelConfig
    mc = ModelConfig(**model_config)
    model = CausalTimeSeriesTransformer(mc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Fetch data
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(
        symbol=config.data.symbol,
        interval=config.data.interval,
        n_candles=n_candles
    )
    
    # Compute indicators
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    # Normalize features (using saved statistics would be better in production)
    features = df[feature_names].values
    scaler_mean = features[:int(len(features)*0.7)].mean(axis=0)
    scaler_std = features[:int(len(features)*0.7)].std(axis=0)
    
    # Run backtest
    backtester = Backtester(config)
    results = backtester.run(
        model=model,
        df=df,
        feature_cols=feature_names,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std
    )
    
    return results


if __name__ == "__main__":
    # Test backtest with a random model
    config = Config()
    config.model.input_dim = 22
    
    from model import CausalTimeSeriesTransformer
    model = CausalTimeSeriesTransformer(config.model)
    
    # Generate fake data
    from data import CryptoDataFetcher, compute_technical_indicators
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(config.data.symbol, config.data.interval, n_candles=3000)
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
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
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    features = df[feature_cols].values
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    
    backtester = Backtester(config)
    results = backtester.run(model, df, feature_cols, mean, std)
    
    print_backtest_results(results)
