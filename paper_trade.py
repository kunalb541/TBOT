"""
Paper Trading Simulator for Crypto Trading Bot
=================================================
Simulates live trading with fake money to validate strategies.

Features:
- Real-time price simulation
- Daily P&L tracking
- Position management
- Risk controls
- Auto-retraining triggers
"""

import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import Config, DEFAULT_CONFIG
from model import CausalTimeSeriesTransformer
from data import (
    CryptoDataFetcher,
    compute_technical_indicators,
    generate_fake_ohlcv
)
from backtest import PositionType, Trade


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: datetime
    start_capital: float
    end_capital: float
    pnl: float
    trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown: float


@dataclass
class TradingSession:
    """Current trading session state."""
    capital: float
    position: PositionType = PositionType.NONE
    position_size: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Daily tracking
    daily_start_capital: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    current_day: datetime = None
    
    # Performance tracking
    peak_capital: float = 0.0
    total_trades: List[Trade] = field(default_factory=list)
    daily_stats: List[DailyStats] = field(default_factory=list)


class PriceSimulator:
    """
    Simulates real-time price updates.
    In production, replace with real API streaming.
    """
    
    def __init__(self, config: Config, update_interval: float = 5.0):
        """
        Args:
            config: Configuration object
            update_interval: Seconds between price updates
        """
        self.config = config
        self.update_interval = update_interval
        self.current_price = 40000.0  # Starting price
        self.price_history: List[Dict] = []
        self._running = False
        self._thread = None
        self._price_queue = queue.Queue()
        
        # Volatility parameters
        self.volatility = 0.001  # Per update volatility
        self.drift = 0.00001  # Slight upward drift
    
    def start(self):
        """Start price simulation."""
        self._running = True
        self._thread = threading.Thread(target=self._simulate_prices)
        self._thread.daemon = True
        self._thread.start()
        logger.info("Price simulator started")
    
    def stop(self):
        """Stop price simulation."""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Price simulator stopped")
    
    def _simulate_prices(self):
        """Generate simulated price updates."""
        while self._running:
            # Generate price movement
            return_val = np.random.normal(self.drift, self.volatility)
            self.current_price *= (1 + return_val)
            
            # Generate OHLC for this interval
            noise = np.random.uniform(-self.volatility, self.volatility, 3)
            high = self.current_price * (1 + abs(noise[0]))
            low = self.current_price * (1 - abs(noise[1]))
            open_price = self.current_price * (1 + noise[2] * 0.5)
            
            candle = {
                'timestamp': datetime.now(),
                'open': open_price,
                'high': high,
                'low': low,
                'close': self.current_price,
                'volume': np.random.uniform(100, 1000)
            }
            
            self.price_history.append(candle)
            
            # Keep only recent history
            if len(self.price_history) > 1000:
                self.price_history = self.price_history[-500:]
            
            self._price_queue.put(candle)
            
            time.sleep(self.update_interval)
    
    def get_latest_price(self) -> Optional[Dict]:
        """Get the latest price update."""
        try:
            return self._price_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_current_price(self) -> float:
        """Get current price."""
        return self.current_price
    
    def get_history_df(self, n_candles: int = 200) -> pd.DataFrame:
        """Get price history as DataFrame."""
        if len(self.price_history) < n_candles:
            # Generate more historical data
            fake_df = generate_fake_ohlcv(n_candles=n_candles - len(self.price_history))
            fake_records = fake_df.reset_index().to_dict('records')
            full_history = fake_records + self.price_history
        else:
            full_history = self.price_history[-n_candles:]
        
        df = pd.DataFrame(full_history)
        df.set_index('timestamp', inplace=True)
        return df


class PaperTrader:
    """
    Paper trading system for strategy validation.
    
    Features:
    - Simulated order execution
    - Daily P&L tracking
    - Risk management
    - Performance logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config = DEFAULT_CONFIG,
        feature_names: List[str] = None,
        scaler_mean: np.ndarray = None,
        scaler_std: np.ndarray = None
    ):
        self.model = model
        self.config = config
        self.feature_names = feature_names or []
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        
        self.device = next(model.parameters()).device
        self.model.eval()
        
        # Initialize session
        self.session = TradingSession(
            capital=config.trading.initial_capital,
            peak_capital=config.trading.initial_capital,
            daily_start_capital=config.trading.initial_capital,
            current_day=datetime.now().date()
        )
        
        # Price simulator
        self.price_sim = PriceSimulator(config)
        
        # State
        self._running = False
        self._last_prediction_time = None
    
    def _apply_slippage(self, price: float, is_entry: bool, is_long: bool) -> float:
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
        price: float,
        position_type: PositionType
    ) -> bool:
        """Open a new position."""
        if self.session.position != PositionType.NONE:
            logger.warning("Cannot open position: already in a position")
            return False
        
        # Calculate position size
        capital_for_trade = self.session.capital * self.config.trading.position_size
        
        # Apply slippage
        is_long = position_type == PositionType.LONG
        exec_price = self._apply_slippage(price, is_entry=True, is_long=is_long)
        
        # Calculate size and fee
        fee = self._calculate_fee(exec_price, capital_for_trade / exec_price)
        position_size = (capital_for_trade - fee) / exec_price
        
        # Set stop loss and take profit
        if is_long:
            stop_loss = exec_price * (1 - self.config.trading.stop_loss_pct)
            take_profit = exec_price * (1 + self.config.trading.take_profit_pct)
        else:
            stop_loss = exec_price * (1 + self.config.trading.stop_loss_pct)
            take_profit = exec_price * (1 - self.config.trading.take_profit_pct)
        
        # Update session
        self.session.position = position_type
        self.session.position_size = position_size
        self.session.entry_price = exec_price
        self.session.entry_time = datetime.now()
        self.session.stop_loss = stop_loss
        self.session.take_profit = take_profit
        
        direction = "LONG" if is_long else "SHORT"
        logger.info(
            f"OPENED {direction} | Entry: ${exec_price:.2f} | "
            f"Size: {position_size:.6f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
        )
        
        return True
    
    def _close_position(self, price: float, reason: str = "signal") -> Optional[Trade]:
        """Close current position."""
        if self.session.position == PositionType.NONE:
            return None
        
        is_long = self.session.position == PositionType.LONG
        exec_price = self._apply_slippage(price, is_entry=False, is_long=is_long)
        
        # Calculate PnL
        if is_long:
            gross_pnl = (exec_price - self.session.entry_price) * self.session.position_size
        else:
            gross_pnl = (self.session.entry_price - exec_price) * self.session.position_size
        
        fee = self._calculate_fee(exec_price, self.session.position_size)
        net_pnl = gross_pnl - fee
        
        # Create trade record
        trade = Trade(
            entry_time=self.session.entry_time,
            entry_price=self.session.entry_price,
            exit_time=datetime.now(),
            exit_price=exec_price,
            position_type=self.session.position,
            size=self.session.position_size,
            pnl=net_pnl,
            exit_reason=reason
        )
        
        # Update capital
        self.session.capital += net_pnl
        self.session.daily_pnl += net_pnl
        self.session.daily_trades += 1
        self.session.total_trades.append(trade)
        
        # Update peak capital
        if self.session.capital > self.session.peak_capital:
            self.session.peak_capital = self.session.capital
        
        direction = "LONG" if is_long else "SHORT"
        logger.info(
            f"CLOSED {direction} | Exit: ${exec_price:.2f} | "
            f"PnL: ${net_pnl:+.2f} | Reason: {reason}"
        )
        
        # Reset position
        self.session.position = PositionType.NONE
        self.session.position_size = 0.0
        self.session.entry_price = 0.0
        self.session.entry_time = None
        
        return trade
    
    def _check_stop_loss_take_profit(self, high: float, low: float) -> Optional[Trade]:
        """Check if SL/TP is hit."""
        if self.session.position == PositionType.NONE:
            return None
        
        if self.session.position == PositionType.LONG:
            if low <= self.session.stop_loss:
                return self._close_position(self.session.stop_loss, "stop_loss")
            if high >= self.session.take_profit:
                return self._close_position(self.session.take_profit, "take_profit")
        else:
            if high >= self.session.stop_loss:
                return self._close_position(self.session.stop_loss, "stop_loss")
            if low <= self.session.take_profit:
                return self._close_position(self.session.take_profit, "take_profit")
        
        return None
    
    def _check_daily_limits(self) -> bool:
        """Check if daily limits are reached."""
        if self.session.daily_pnl <= -self.config.trading.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: ${self.session.daily_pnl:.2f}")
            return True
        
        if self.session.daily_pnl >= self.config.trading.daily_profit_target:
            logger.info(f"Daily profit target reached: ${self.session.daily_pnl:.2f}")
            return True
        
        return False
    
    def _reset_daily_tracking(self):
        """Reset daily tracking for new day."""
        # Save previous day stats
        if self.session.current_day:
            winning = len([t for t in self.session.total_trades[-self.session.daily_trades:] if t.pnl > 0])
            losing = self.session.daily_trades - winning
            
            stats = DailyStats(
                date=self.session.current_day,
                start_capital=self.session.daily_start_capital,
                end_capital=self.session.capital,
                pnl=self.session.daily_pnl,
                trades=self.session.daily_trades,
                winning_trades=winning,
                losing_trades=losing,
                max_drawdown=self.session.peak_capital - self.session.capital
            )
            self.session.daily_stats.append(stats)
            
            logger.info(
                f"Day ended: {self.session.current_day} | "
                f"PnL: ${self.session.daily_pnl:+.2f} | "
                f"Trades: {self.session.daily_trades}"
            )
        
        # Reset for new day
        self.session.current_day = datetime.now().date()
        self.session.daily_start_capital = self.session.capital
        self.session.daily_pnl = 0.0
        self.session.daily_trades = 0
        
        logger.info(f"New trading day: {self.session.current_day}")
    
    def _get_prediction(self, df: pd.DataFrame) -> tuple:
        """Get model prediction."""
        # Compute features
        df = compute_technical_indicators(df, self.config)
        df = df.dropna()
        
        if len(df) < self.config.data.lookback_window:
            return 0, 0.0  # Hold if not enough data
        
        # Get latest window
        history = df.iloc[-self.config.data.lookback_window:]
        
        # Extract and normalize features
        available_features = [f for f in self.feature_names if f in history.columns]
        if len(available_features) != len(self.feature_names):
            logger.warning(f"Missing features: {set(self.feature_names) - set(available_features)}")
            return 0, 0.0
        
        features = history[available_features].values
        
        if self.scaler_mean is not None and self.scaler_std is not None:
            features = (features - self.scaler_mean) / (self.scaler_std + 1e-10)
        
        # Get prediction
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            pred, conf = self.model.predict(x)
            return pred[0].item(), conf[0].item()
    
    def run(
        self,
        duration_minutes: int = 60,
        prediction_interval: int = 60  # seconds
    ):
        """
        Run paper trading simulation.
        
        Args:
            duration_minutes: How long to run
            prediction_interval: Seconds between predictions
        """
        self._running = True
        self.price_sim.start()
        
        logger.info(f"Starting paper trading for {duration_minutes} minutes")
        logger.info(f"Initial capital: ${self.session.capital:,.2f}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            while self._running and datetime.now() < end_time:
                # Check for new day
                if datetime.now().date() != self.session.current_day:
                    if self.session.position != PositionType.NONE:
                        self._close_position(
                            self.price_sim.get_current_price(),
                            "day_end"
                        )
                    self._reset_daily_tracking()
                
                # Check daily limits
                if self._check_daily_limits():
                    if self.session.position != PositionType.NONE:
                        self._close_position(
                            self.price_sim.get_current_price(),
                            "daily_limit"
                        )
                    logger.info("Waiting for next trading day...")
                    time.sleep(60)
                    continue
                
                # Get latest price update
                price_update = self.price_sim.get_latest_price()
                
                if price_update:
                    current_price = price_update['close']
                    high = price_update['high']
                    low = price_update['low']
                    
                    # Check SL/TP
                    self._check_stop_loss_take_profit(high, low)
                    
                    # Make prediction periodically
                    should_predict = (
                        self._last_prediction_time is None or
                        (datetime.now() - self._last_prediction_time).seconds >= prediction_interval
                    )
                    
                    if should_predict:
                        self._last_prediction_time = datetime.now()
                        
                        # Get historical data
                        df = self.price_sim.get_history_df(n_candles=300)
                        
                        # Get prediction
                        prediction, confidence = self._get_prediction(df)
                        
                        # Log status
                        position_str = (
                            "FLAT" if self.session.position == PositionType.NONE
                            else ("LONG" if self.session.position == PositionType.LONG else "SHORT")
                        )
                        logger.info(
                            f"Price: ${current_price:.2f} | Position: {position_str} | "
                            f"Pred: {prediction} ({confidence:.2f}) | "
                            f"Capital: ${self.session.capital:,.2f} | "
                            f"Daily PnL: ${self.session.daily_pnl:+.2f}"
                        )
                        
                        # Execute trading logic
                        if confidence >= self.config.trading.min_confidence:
                            if self.session.position != PositionType.NONE:
                                # Check if we should close
                                should_close = (
                                    prediction == 0 or
                                    (self.session.position == PositionType.LONG and prediction == 2) or
                                    (self.session.position == PositionType.SHORT and prediction == 1)
                                )
                                if should_close:
                                    self._close_position(current_price, "signal")
                            
                            # Check if we should open
                            if self.session.position == PositionType.NONE:
                                if prediction == 1:
                                    self._open_position(current_price, PositionType.LONG)
                                elif prediction == 2:
                                    self._open_position(current_price, PositionType.SHORT)
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        
        finally:
            self._running = False
            
            # Close any open position
            if self.session.position != PositionType.NONE:
                self._close_position(
                    self.price_sim.get_current_price(),
                    "shutdown"
                )
            
            self.price_sim.stop()
            self._print_summary()
    
    def stop(self):
        """Stop trading."""
        self._running = False
    
    def _print_summary(self):
        """Print trading session summary."""
        print("\n" + "="*60)
        print("PAPER TRADING SUMMARY")
        print("="*60)
        
        initial = self.config.trading.initial_capital
        final = self.session.capital
        total_pnl = final - initial
        total_pnl_pct = (total_pnl / initial) * 100
        
        print(f"\n--- Capital ---")
        print(f"Initial:    ${initial:>12,.2f}")
        print(f"Final:      ${final:>12,.2f}")
        print(f"Total PnL:  ${total_pnl:>12,.2f} ({total_pnl_pct:+.2f}%)")
        
        trades = self.session.total_trades
        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:  {len(trades)}")
        
        if trades:
            winning = len([t for t in trades if t.pnl > 0])
            losing = len(trades) - winning
            win_rate = winning / len(trades) * 100
            
            print(f"Winning:       {winning}")
            print(f"Losing:        {losing}")
            print(f"Win Rate:      {win_rate:.1f}%")
            
            avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if winning > 0 else 0
            avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0]) if losing > 0 else 0
            
            print(f"Avg Win:       ${avg_win:,.2f}")
            print(f"Avg Loss:      ${avg_loss:,.2f}")
        
        # Exit reasons breakdown
        if trades:
            reasons = {}
            for t in trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            
            print(f"\n--- Exit Reasons ---")
            for reason, count in sorted(reasons.items()):
                print(f"{reason:15s}: {count}")
        
        print("\n" + "="*60)
    
    def get_session_data(self) -> Dict:
        """Get session data for saving/analysis."""
        return {
            'capital': self.session.capital,
            'total_pnl': self.session.capital - self.config.trading.initial_capital,
            'trades': [
                {
                    'entry_time': str(t.entry_time),
                    'exit_time': str(t.exit_time),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position': t.position_type.name,
                    'pnl': t.pnl,
                    'exit_reason': t.exit_reason
                }
                for t in self.session.total_trades
            ],
            'daily_stats': [
                {
                    'date': str(d.date),
                    'pnl': d.pnl,
                    'trades': d.trades,
                    'win_rate': d.winning_trades / d.trades if d.trades > 0 else 0
                }
                for d in self.session.daily_stats
            ]
        }


if __name__ == "__main__":
    # Test paper trading with random model
    from config import Config
    from model import CausalTimeSeriesTransformer
    
    config = Config()
    
    # Feature names (must match training)
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
    
    # Create trader
    trader = PaperTrader(
        model=model,
        config=config,
        feature_names=feature_names
    )
    
    # Run for 2 minutes as demo
    trader.run(duration_minutes=2, prediction_interval=10)
