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
import logging

import torch
import torch.nn as nn

from config import Config, DEFAULT_CONFIG
from model import CausalTimeSeriesTransformer
from data import CryptoDataFetcher, compute_technical_indicators, FEATURE_COLS

logger = logging.getLogger(__name__)


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
        """Apply slippage to execution price."""
        slippage = price * self.config.trading.slippage_pct
        
        if position_type == PositionType.LONG:
            return price + slippage
        else:
            return price - slippage
    
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
        
        capital_for_trade = self.state.capital * self.config.trading.position_size
        exec_price = self._calculate_slippage(price, position_type)
        fee = self._calculate_fee(exec_price, capital_for_trade / exec_price)
        position_size = (capital_for_trade - fee) / exec_price
        
        if position_type == PositionType.LONG:
            stop_loss = exec_price * (1 - self.config.trading.stop_loss_pct)
            take_profit = exec_price * (1 + self.config.trading.take_profit_pct)
        else:
            stop_loss = exec_price * (1 + self.config.trading.stop_loss_pct)
            take_profit = exec_price * (1 - self.config.trading.take_profit_pct)
        
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
        
        opposite_type = (PositionType.SHORT if self.state.position == PositionType.LONG 
                        else PositionType.LONG)
        exec_price = self._calculate_slippage(price, opposite_type)
        exit_fee = self._calculate_fee(exec_price, self.state.position_size)
        
        if self.state.position == PositionType.LONG:
            gross_pnl = (exec_price - self.state.entry_price) * self.state.position_size
        else:
            gross_pnl = (self.state.entry_price - exec_price) * self.state.position_size
        
        entry_value = self.state.entry_price * self.state.position_size
        entry_fee = entry_value * self.config.trading.taker_fee / (1 - self.config.trading.taker_fee)
        
        net_pnl = gross_pnl - exit_fee
        pnl_pct = net_pnl / (entry_value + entry_fee) if (entry_value + entry_fee) > 0 else 0
        
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
        
        self.state.capital += net_pnl
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
            if low <= self.state.stop_loss:
                return self._close_position(timestamp, self.state.stop_loss, "stop_loss")
            if high >= self.state.take_profit:
                return self._close_position(timestamp, self.state.take_profit, "take_profit")
        else:
            if high >= self.state.stop_loss:
                return self._close_position(timestamp, self.state.stop_loss, "stop_loss")
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
        
        if equity > self.state.peak_capital:
            self.state.peak_capital = equity
    
    def _signal_to_action(
        self,
        prediction: int,
        confidence: float
    ) -> Optional[PositionType]:
        """Convert model prediction to trading action."""
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
        """
        model.eval()
        device = next(model.parameters()).device
        
        self.state = BacktestState(
            capital=self.config.trading.initial_capital,
            peak_capital=self.config.trading.initial_capital
        )
        
        lookback = self.config.data.lookback_window
        
        logger.info(f"\n=== Running Backtest ===")
        logger.info(f"Initial capital: ${self.state.capital:,.2f}")
        logger.info(f"Period: {df.index[lookback]} to {df.index[-1]}")
        logger.info(f"Total candles: {len(df) - lookback}")
        
        for i in range(lookback, len(df)):
            timestamp = df.index[i]
            current_candle = df.iloc[i]
            
            # Get historical data UP TO current point (no lookahead!)
            history = df.iloc[i - lookback:i]
            
            # Extract features and normalize
            available_features = [c for c in feature_cols if c in history.columns]
            features = history[available_features].values
            
            # Ensure scaler dimensions match
            if len(scaler_mean) != features.shape[1]:
                logger.warning(f"Feature dimension mismatch: {features.shape[1]} vs {len(scaler_mean)}")
                continue
                
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
            if i + 1 < len(df):
                next_open = df.iloc[i + 1]['open']
                
                if self.state.position != PositionType.NONE:
                    should_close = (
                        prediction == 0 or
                        (self.state.position == PositionType.LONG and prediction == 2) or
                        (self.state.position == PositionType.SHORT and prediction == 1)
                    )
                    if should_close:
                        trade = self._close_position(timestamp, next_open, "signal")
                        if trade:
                            self.state.trades.append(trade)
                
                if self.state.position == PositionType.NONE:
                    action = self._signal_to_action(prediction, confidence)
                    if action is not None:
                        self._open_position(timestamp, next_open, action)
            
            self._update_equity(timestamp, current_candle['close'])
            
            # Check max drawdown limit
            if self.state.equity_curve:
                current_dd = (self.state.peak_capital - self.state.equity_curve[-1]) / self.state.peak_capital
                if current_dd >= self.config.trading.max_drawdown_pct:
                    logger.info(f"\nMax drawdown limit reached at {timestamp}")
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
        
        total_return = final_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
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
            
            durations = []
            for t in trades:
                if t.entry_time and t.exit_time:
                    duration = (t.exit_time - t.entry_time).total_seconds() / 3600
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
        
        equity_curve = np.array(self.state.equity_curve)
        
        if len(equity_curve) > 1:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdown_pct = np.max(drawdown) * 100
            max_drawdown = np.max(peak - equity_curve)
            
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            if np.std(returns) > 0:
                sharpe_ratio = np.sqrt(24 * 365) * np.mean(returns) / np.std(returns)
            else:
                sharpe_ratio = 0.0
            
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
    
    if results.trades:
        reasons = {}
        for t in results.trades:
            reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        
        print("\n--- Exit Reasons ---")
        for reason, count in sorted(reasons.items()):
            print(f"{reason:20s}: {count}")


if __name__ == "__main__":
    config = Config()
    config.model.input_dim = 22
    config.data.use_real_data = True
    
    from model import CausalTimeSeriesTransformer
    model = CausalTimeSeriesTransformer(config.model)
    
    from data import CryptoDataFetcher
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical(config.data.symbol, config.data.interval, n_candles=3000)
    df = compute_technical_indicators(df, config)
    df = df.dropna()
    
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    features = df[feature_cols].values
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    
    backtester = Backtester(config)
    results = backtester.run(model, df, feature_cols, mean, std)
    
    print_backtest_results(results)
