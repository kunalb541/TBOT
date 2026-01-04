# Crypto Trading Bot with Causal Time Series Transformer

A production-ready proof-of-concept for an AI-powered cryptocurrency futures trading bot using a **Causal Time Series Transformer** with strict anti-lookahead measures.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAUSAL TRANSFORMER                           │
├─────────────────────────────────────────────────────────────────┤
│  Input Projection → Positional Encoding → Encoder Layers → Head │
│                           ↓                                     │
│            ┌────────────────────────────────┐                   │
│            │   CAUSAL SELF-ATTENTION        │                   │
│            │   Position i → attends to ≤ i  │                   │
│            │   NO FUTURE INFORMATION LEAK   │                   │
│            └────────────────────────────────┘                   │
├─────────────────────────────────────────────────────────────────┤
│  Output: [Hold=0, Long=1, Short=2] with confidence score        │
└─────────────────────────────────────────────────────────────────┘
```

## 🔐 Anti-Lookahead Measures

This system implements **strict anti-cheating measures** to prevent lookahead bias:

### 1. Causal Attention Masking
```python
# Each position can ONLY attend to previous positions
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
attn_scores.masked_fill_(mask, float('-inf'))
```

### 2. Temporal Data Splitting
```
|-------- Train --------|--- Val ---|--- Test ---|
        t=0                t=0.7       t=0.85     t=1.0

NO OVERLAP between splits
Features computed BEFORE splitting
```

### 3. Realistic Trade Execution
- Trades execute at **NEXT candle open** (not current close)
- Slippage simulation applied
- Realistic fee calculation

### 4. Feature Engineering
- All features computed using ONLY historical data
- Rolling windows use `.shift(1)` to prevent peeking

## 📁 Project Structure

```
crypto_bot/
├── config.py         # All configuration parameters
├── data.py           # Data fetching, preprocessing, dataset creation
├── model.py          # Causal Transformer architecture
├── train.py          # Training pipeline with checkpointing
├── backtest.py       # Backtesting engine with risk management
├── paper_trade.py    # Paper trading simulator
├── main.py           # Main entry point
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick demo (no training required)
python main.py demo

# Full pipeline: Train → Backtest → Paper Trade
python main.py full

# Individual steps
python main.py train              # Train the model
python main.py backtest           # Run backtesting
python main.py paper --duration 5 # Paper trade for 5 minutes
```

## 📊 Model Architecture

### Input Features (22 dimensions)
- **Price-based**: returns, log_returns, range_pct, body_pct, wicks
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Moving Averages**: EMA distances (9, 21, 50, 200)
- **Volume**: normalized volume, volume ratio
- **Momentum**: 5, 10, 20-period momentum
- **Volatility**: rolling volatility normalized

### Transformer Configuration
- `d_model`: 128
- `nhead`: 8 attention heads
- `num_layers`: 4 encoder layers
- `dim_feedforward`: 512
- `dropout`: 0.1
- `max_seq_len`: 512

### Output Classes
- **0**: Hold (no action)
- **1**: Long (buy)
- **2**: Short (sell)

## 📈 Risk Management

```python
TradingConfig:
    initial_capital: 10000.0
    position_size: 0.1           # 10% per trade
    stop_loss_pct: 0.02          # 2% stop loss
    take_profit_pct: 0.04        # 4% take profit
    max_drawdown_pct: 0.15       # 15% max drawdown
    daily_loss_limit: 500.0      # Daily loss limit
    daily_profit_target: 500.0   # Daily profit target
    min_confidence: 0.6          # Minimum prediction confidence
```

## 🔬 Verification

The system includes built-in verification of causal masking:

```python
def verify_causal_masking(model, seq_len=10):
    """
    Test: Changing future inputs should NOT affect current outputs.
    """
    # Get output at position 5
    out1 = model(x)[:, 5, :]
    
    # Modify positions 6-9 (future)
    x_modified[:, 6:, :] = random_data
    
    # Get output at position 5 again
    out2 = model(x_modified)[:, 5, :]
    
    # Should be identical with causal masking
    assert torch.allclose(out1, out2)
```

## 🔄 Auto-Retraining (Planned)

The system is designed to support auto-retraining:

```python
RetrainingConfig:
    enabled: True
    retrain_interval_hours: 168    # Weekly
    min_new_samples: 168           # Minimum new data
    performance_threshold: 0.0     # Retrain on performance drop
    rolling_window_days: 90        # Training window
```

## 📝 Extending to Real Trading

To connect to real exchanges:

1. **Install exchange SDK**:
   ```bash
   pip install python-binance ccxt
   ```

2. **Replace `CryptoDataFetcher.fetch_historical()`**:
   ```python
   from binance.client import Client
   
   def fetch_from_binance(self, symbol, interval, n_candles):
       klines = self.client.get_historical_klines(
           symbol, interval, limit=n_candles
       )
       return pd.DataFrame(klines, columns=[...])
   ```

3. **Replace `PriceSimulator`** with real-time streaming:
   ```python
   from binance import BinanceSocketManager
   
   def start_real_stream(self, symbol):
       bsm = BinanceSocketManager(self.client)
       conn = bsm.kline_socket(symbol, callback=self.process_message)
   ```

## ⚠️ Disclaimer

This is a **proof-of-concept** for educational purposes. Real cryptocurrency trading involves significant financial risk. The fake data generator produces unrealistic price movements. Always:

1. Test thoroughly with paper trading
2. Start with small positions
3. Never risk money you can't afford to lose
4. Understand that past performance doesn't guarantee future results

## 📚 References

- [Temporal Fusion Transformers (TFT)](https://arxiv.org/abs/1912.09363)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Informer for Long Sequence Time-Series](https://arxiv.org/abs/2012.07436)
- [Lookahead Bias Prevention](https://quantjourney.substack.com/p/advanced-look-ahead-bias-prevention)

## 📄 License

MIT License - Use at your own risk.
