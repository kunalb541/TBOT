#!/usr/bin/env python3
"""
Desktop Paper Trading with Real-Time Status Updates
====================================================
Enhanced version for desktop with:
- Frequent status updates (every 5 minutes)
- Live dashboard file
- Progress tracking
- Automatic recovery
"""
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import Config, DEFAULT_CONFIG
from model import CausalTimeSeriesTransformer
from data import CryptoDataFetcher, compute_technical_indicators, FEATURE_COLS
from backtest import PositionType, Trade

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("paper_trade_desktop.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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
    
    # Performance tracking
    peak_capital: float = 0.0
    total_trades: List[Trade] = field(default_factory=list)
    equity_history: List[Dict] = field(default_factory=list)
    
    # Session stats
    session_start: datetime = field(default_factory=datetime.now)
    last_check_time: Optional[datetime] = None
    next_check_time: Optional[datetime] = None
    checks_completed: int = 0
    total_runtime_hours: float = 0.0


class DesktopPaperTrader:
    """
    Enhanced paper trading system for desktop with real-time updates.
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
            peak_capital=config.trading.initial_capital
        )
        
        # Data fetcher for real prices
        self.fetcher = CryptoDataFetcher(config)
        
        # State
        self._running = False
        self._latest_price = None
        self._latest_data = None
        self._latest_prediction = None
        self._latest_confidence = None
        
        # Dashboard file
        self.dashboard_file = "trading_dashboard.json"
        self.status_file = "trading_status.txt"
    
    def _fetch_latest_data(self, n_candles: int = 300) -> pd.DataFrame:
        """Fetch latest real market data from Binance."""
        try:
            df = self.fetcher.fetch_historical(
                symbol=self.config.data.symbol,
                interval=self.config.data.interval,
                n_candles=n_candles
            )
            
            # Compute technical indicators
            df = compute_technical_indicators(df, self.config)
            df = df.dropna()
            
            self._latest_data = df
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return None
    
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
            logger.warning("⚠️  Cannot open position: already in a position")
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
            f"🟢 OPENED {direction} | Entry: ${exec_price:.2f} | "
            f"Size: {position_size:.6f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
        )
        
        self._update_dashboard()
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
        self.session.total_trades.append(trade)
        
        # Update peak capital
        if self.session.capital > self.session.peak_capital:
            self.session.peak_capital = self.session.capital
        
        direction = "LONG" if is_long else "SHORT"
        pnl_pct = (net_pnl / (self.session.entry_price * self.session.position_size)) * 100
        logger.info(
            f"🔴 CLOSED {direction} | Exit: ${exec_price:.2f} | "
            f"PnL: ${net_pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}"
        )
        
        # Reset position
        self.session.position = PositionType.NONE
        self.session.position_size = 0.0
        self.session.entry_price = 0.0
        self.session.entry_time = None
        
        self._update_dashboard()
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
    
    def _get_prediction(self, df: pd.DataFrame) -> tuple:
        """Get model prediction from real data."""
        if len(df) < self.config.data.lookback_window:
            return 0, 0.0
        
        # Get latest window
        history = df.iloc[-self.config.data.lookback_window:]
        
        # Extract and normalize features
        available_features = [f for f in self.feature_names if f in history.columns]
        if len(available_features) != len(self.feature_names):
            logger.warning(f"⚠️  Missing features: {set(self.feature_names) - set(available_features)}")
            return 0, 0.0
        
        features = history[available_features].values
        
        if self.scaler_mean is not None and self.scaler_std is not None:
            features = (features - self.scaler_mean) / (self.scaler_std + 1e-10)
        
        # Get prediction
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            pred, conf = self.model.predict(x)
            return pred[0].item(), conf[0].item()
    
    def _update_dashboard(self):
        """Update dashboard JSON file."""
        try:
            # Calculate metrics
            unrealized_pnl = 0
            if self.session.position != PositionType.NONE and self._latest_price:
                if self.session.position == PositionType.LONG:
                    unrealized_pnl = (self._latest_price - self.session.entry_price) * self.session.position_size
                else:
                    unrealized_pnl = (self.session.entry_price - self._latest_price) * self.session.position_size
            
            current_equity = self.session.capital + unrealized_pnl
            total_pnl = self.session.capital - self.config.trading.initial_capital
            total_pnl_pct = (total_pnl / self.config.trading.initial_capital) * 100
            
            winning_trades = len([t for t in self.session.total_trades if t.pnl > 0])
            losing_trades = len(self.session.total_trades) - winning_trades
            win_rate = (winning_trades / len(self.session.total_trades) * 100) if self.session.total_trades else 0
            
            # Calculate runtime
            runtime = datetime.now() - self.session.session_start
            runtime_hours = runtime.total_seconds() / 3600
            
            dashboard_data = {
                "last_updated": datetime.now().isoformat(),
                "status": "RUNNING" if self._running else "STOPPED",
                "symbol": self.config.data.symbol,
                "interval": self.config.data.interval,
                
                "capital": {
                    "initial": self.config.trading.initial_capital,
                    "current": round(self.session.capital, 2),
                    "equity": round(current_equity, 2),
                    "total_pnl": round(total_pnl, 2),
                    "total_pnl_pct": round(total_pnl_pct, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2)
                },
                
                "position": {
                    "status": self.session.position.name,
                    "entry_price": self.session.entry_price if self.session.position != PositionType.NONE else None,
                    "entry_time": self.session.entry_time.isoformat() if self.session.entry_time else None,
                    "size": self.session.position_size,
                    "stop_loss": self.session.stop_loss if self.session.position != PositionType.NONE else None,
                    "take_profit": self.session.take_profit if self.session.position != PositionType.NONE else None
                },
                
                "market": {
                    "latest_price": self._latest_price,
                    "prediction": ["HOLD", "LONG", "SHORT"][self._latest_prediction] if self._latest_prediction is not None else None,
                    "confidence": round(self._latest_confidence * 100, 2) if self._latest_confidence else None
                },
                
                "trading_stats": {
                    "total_trades": len(self.session.total_trades),
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": round(win_rate, 2)
                },
                
                "session_info": {
                    "start_time": self.session.session_start.isoformat(),
                    "runtime_hours": round(runtime_hours, 2),
                    "checks_completed": self.session.checks_completed,
                    "last_check": self.session.last_check_time.isoformat() if self.session.last_check_time else None,
                    "next_check": self.session.next_check_time.isoformat() if self.session.next_check_time else None
                },
                
                "recent_trades": [
                    {
                        "entry_time": t.entry_time.isoformat(),
                        "exit_time": t.exit_time.isoformat(),
                        "type": t.position_type.name,
                        "entry_price": round(t.entry_price, 2),
                        "exit_price": round(t.exit_price, 2),
                        "pnl": round(t.pnl, 2),
                        "reason": t.exit_reason
                    }
                    for t in self.session.total_trades[-5:]  # Last 5 trades
                ]
            }
            
            with open(self.dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            # Also create simple text status
            self._update_status_text(dashboard_data)
            
        except Exception as e:
            logger.error(f"❌ Error updating dashboard: {e}")
    
    def _update_status_text(self, data: dict):
        """Create simple text status file."""
        try:
            status_text = f"""
╔═══════════════════════════════════════════════════════════╗
║       CRYPTO TRADING BOT - LIVE STATUS                   ║
╚═══════════════════════════════════════════════════════════╝

📊 MARKET INFO
   Symbol:          {data['symbol']}
   Latest Price:    ${data['market']['latest_price']:.2f}
   Prediction:      {data['market']['prediction'] or 'N/A'}
   Confidence:      {data['market']['confidence']}%

💼 POSITION
   Status:          {data['position']['status']}
   Entry Price:     ${data['position']['entry_price'] or 0:.2f}
   Stop Loss:       ${data['position']['stop_loss'] or 0:.2f}
   Take Profit:     ${data['position']['take_profit'] or 0:.2f}

💰 CAPITAL
   Initial:         ${data['capital']['initial']:,.2f}
   Current:         ${data['capital']['current']:,.2f}
   Equity:          ${data['capital']['equity']:,.2f}
   Total P&L:       ${data['capital']['total_pnl']:+,.2f} ({data['capital']['total_pnl_pct']:+.2f}%)
   Unrealized P&L:  ${data['capital']['unrealized_pnl']:+,.2f}

📈 TRADING STATS
   Total Trades:    {data['trading_stats']['total_trades']}
   Winning:         {data['trading_stats']['winning_trades']}
   Losing:          {data['trading_stats']['losing_trades']}
   Win Rate:        {data['trading_stats']['win_rate']:.1f}%

⏱️  SESSION INFO
   Started:         {data['session_info']['start_time'][:19]}
   Runtime:         {data['session_info']['runtime_hours']:.1f} hours
   Checks:          {data['session_info']['checks_completed']}
   Last Check:      {data['session_info']['last_check'][:19] if data['session_info']['last_check'] else 'N/A'}
   Next Check:      {data['session_info']['next_check'][:19] if data['session_info']['next_check'] else 'N/A'}

═══════════════════════════════════════════════════════════
Last Updated: {data['last_updated'][:19]}
═══════════════════════════════════════════════════════════
"""
            
            with open(self.status_file, 'w') as f:
                f.write(status_text)
                
        except Exception as e:
            logger.error(f"❌ Error updating status text: {e}")
    
    def run(
        self,
        duration_hours: float = 168,  # 1 week default
        market_check_interval: int = 4 * 3600,  # Check market every 4 hours
        status_update_interval: int = 5 * 60  # Update status every 5 minutes
    ):
        """
        Run paper trading with frequent status updates.
        
        Args:
            duration_hours: How long to run (hours)
            market_check_interval: Seconds between market checks (4 hours for 1d)
            status_update_interval: Seconds between status file updates (5 min)
        """
        self._running = True
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        logger.info("\n" + "="*60)
        logger.info("🚀 DESKTOP PAPER TRADING STARTED")
        logger.info("="*60)
        logger.info(f"Symbol: {self.config.data.symbol}")
        logger.info(f"Interval: {self.config.data.interval}")
        logger.info(f"Initial Capital: ${self.session.capital:,.2f}")
        logger.info(f"Duration: {duration_hours} hours")
        logger.info(f"Market Check: Every {market_check_interval/3600:.1f} hours")
        logger.info(f"Status Update: Every {status_update_interval/60:.0f} minutes")
        logger.info("="*60 + "\n")
        
        last_market_check = datetime.now() - timedelta(seconds=market_check_interval)  # Check immediately
        last_status_update = datetime.now()
        
        try:
            while self._running and datetime.now() < end_time:
                current_time = datetime.now()
                
                # Check if it's time for market check
                if (current_time - last_market_check).total_seconds() >= market_check_interval:
                    self.session.checks_completed += 1
                    self.session.last_check_time = current_time
                    self.session.next_check_time = current_time + timedelta(seconds=market_check_interval)
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"📡 MARKET CHECK #{self.session.checks_completed} @ {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"{'='*60}")
                    
                    # Fetch latest data
                    df = self._fetch_latest_data(n_candles=300)
                    
                    if df is None or len(df) == 0:
                        logger.error("❌ Failed to fetch data, waiting...")
                        last_market_check = current_time
                        time.sleep(60)
                        continue
                    
                    # Get current price
                    current_price = df.iloc[-1]['close']
                    current_high = df.iloc[-1]['high']
                    current_low = df.iloc[-1]['low']
                    self._latest_price = current_price
                    
                    logger.info(f"💰 Real {self.config.data.symbol} Price: ${current_price:.2f}")
                    logger.info(f"📊 High: ${current_high:.2f} | Low: ${current_low:.2f}")
                    
                    # Check SL/TP
                    sl_tp_trade = self._check_stop_loss_take_profit(current_high, current_low)
                    
                    # Get prediction
                    prediction, confidence = self._get_prediction(df)
                    self._latest_prediction = prediction
                    self._latest_confidence = confidence
                    
                    # Log position and prediction
                    position_str = self.session.position.name
                    logger.info(f"📈 Position: {position_str}")
                    logger.info(f"🤖 Prediction: {['HOLD', 'LONG', 'SHORT'][prediction]} (confidence: {confidence:.2%})")
                    
                    # Calculate current state
                    unrealized_pnl = 0
                    if self.session.position != PositionType.NONE:
                        if self.session.position == PositionType.LONG:
                            unrealized_pnl = (current_price - self.session.entry_price) * self.session.position_size
                        else:
                            unrealized_pnl = (self.session.entry_price - current_price) * self.session.position_size
                    
                    current_equity = self.session.capital + unrealized_pnl
                    logger.info(f"💼 Capital: ${self.session.capital:,.2f} | Equity: ${current_equity:,.2f}")
                    
                    if unrealized_pnl != 0:
                        logger.info(f"💸 Unrealized P&L: ${unrealized_pnl:+,.2f}")
                    
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
                    else:
                        logger.info(f"⏸️  No trade: confidence {confidence:.2%} < threshold {self.config.trading.min_confidence:.2%}")
                    
                    last_market_check = current_time
                    self._update_dashboard()
                    
                    logger.info(f"⏰ Next market check in {market_check_interval/3600:.1f} hours")
                    logger.info(f"{'='*60}\n")
                
                # Update status file more frequently
                if (current_time - last_status_update).total_seconds() >= status_update_interval:
                    self._update_dashboard()
                    last_status_update = current_time
                    logger.info(f"💾 Status updated @ {current_time.strftime('%H:%M:%S')}")
                
                # Sleep for a bit
                time.sleep(30)  # Check every 30 seconds if it's time to do something
        
        except KeyboardInterrupt:
            logger.info("\n⚠️  Trading interrupted by user")
        
        finally:
            self._running = False
            
            # Close any open position
            if self.session.position != PositionType.NONE:
                df = self._fetch_latest_data(n_candles=10)
                if df is not None and len(df) > 0:
                    final_price = df.iloc[-1]['close']
                    self._close_position(final_price, "shutdown")
            
            self._print_summary()
            self.save_results()
    
    def _print_summary(self):
        """Print trading session summary."""
        print("\n" + "="*60)
        print("📊 PAPER TRADING SUMMARY")
        print("="*60)
        
        initial = self.config.trading.initial_capital
        final = self.session.capital
        total_pnl = final - initial
        total_pnl_pct = (total_pnl / initial) * 100
        
        print(f"\n💰 Capital")
        print(f"   Initial:    ${initial:>12,.2f}")
        print(f"   Final:      ${final:>12,.2f}")
        print(f"   Total P&L:  ${total_pnl:>12,.2f} ({total_pnl_pct:+.2f}%)")
        
        trades = self.session.total_trades
        print(f"\n📈 Trade Statistics")
        print(f"   Total Trades:  {len(trades):>12}")
        
        if trades:
            winning = len([t for t in trades if t.pnl > 0])
            losing = len(trades) - winning
            win_rate = winning / len(trades) * 100
            
            print(f"   Winning:       {winning:>12}")
            print(f"   Losing:        {losing:>12}")
            print(f"   Win Rate:      {win_rate:>11.1f}%")
            
            avg_win = np.mean([t.pnl for t in trades if t.pnl > 0]) if winning > 0 else 0
            avg_loss = np.mean([t.pnl for t in trades if t.pnl < 0]) if losing > 0 else 0
            
            print(f"   Avg Win:       ${avg_win:>12,.2f}")
            print(f"   Avg Loss:      ${avg_loss:>12,.2f}")
            
            # Trade details
            print(f"\n📋 Trade History")
            for i, t in enumerate(trades, 1):
                direction = "LONG" if t.position_type == PositionType.LONG else "SHORT"
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                print(f"   {i}. {direction:5s} | Entry: ${t.entry_price:8.2f} | Exit: ${t.exit_price:8.2f} | "
                      f"P&L: ${t.pnl:+8.2f} | {duration:.1f}h | {t.exit_reason}")
        
        runtime = datetime.now() - self.session.session_start
        print(f"\n⏱️  Session Info")
        print(f"   Runtime:       {runtime.total_seconds() / 3600:>11.1f} hours")
        print(f"   Checks:        {self.session.checks_completed:>12}")
        
        print("\n" + "="*60)
    
    def save_results(self, filename: str = "desktop_paper_trade_results.json"):
        """Save session data to JSON."""
        results = {
            'symbol': self.config.data.symbol,
            'interval': self.config.data.interval,
            'session_start': self.session.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'runtime_hours': (datetime.now() - self.session.session_start).total_seconds() / 3600,
            'initial_capital': self.config.trading.initial_capital,
            'final_capital': self.session.capital,
            'total_pnl': self.session.capital - self.config.trading.initial_capital,
            'total_pnl_pct': ((self.session.capital - self.config.trading.initial_capital) / 
                              self.config.trading.initial_capital) * 100,
            'total_trades': len(self.session.total_trades),
            'checks_completed': self.session.checks_completed,
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position': t.position_type.name,
                    'pnl': t.pnl,
                    'exit_reason': t.exit_reason
                }
                for t in self.session.total_trades
            ],
            'equity_history': self.session.equity_history
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Results saved to: {filename}")


if __name__ == "__main__":
    from config import Config
    
    # Load config
    config = Config()
    
    # Load trained model
    checkpoint_path = 'production_models/BNBUSDT_1d_sharpe_11.48.pt'
    if not Path(checkpoint_path).exists():
        checkpoint_path = 'checkpoints/final_model.pt'
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config.model.input_dim = checkpoint['config']['input_dim']
    model = CausalTimeSeriesTransformer(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    feature_names = checkpoint['feature_names']
    scaler_mean = checkpoint['scaler_mean']
    scaler_std = checkpoint['scaler_std']
    
    print(f"\n✅ Loaded model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trader
    trader = DesktopPaperTrader(
        model=model,
        config=config,
        feature_names=feature_names,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std
    )
    
    # Run paper trading
    # Market check: every 4 hours (respects 1d strategy)
    # Status update: every 5 minutes (frequent updates)
    trader.run(
        duration_hours=168,  # 1 week
        market_check_interval=4 * 3600,  # 4 hours
        status_update_interval=5 * 60  # 5 minutes
    )
