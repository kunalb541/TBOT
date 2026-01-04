# Crypto Trading Bot - Causal Transformer

AI-powered cryptocurrency futures trading bot with strict anti-lookahead measures.

## 🚀 Quick Start

### Local Testing
```bash
# Install
pip install -r requirements.txt

# Test Binance data (no API key needed!)
python main.py test-data

# Quick demo
python main.py demo

# Train model
python main.py train --candles 10000 --symbol BTCUSDT

# Backtest
python main.py backtest

# Full pipeline
python main.py full
```

### HPC Training (bw3.0)
```bash
# Setup environment (first time only)
conda create -n crypto_bot python=3.10 -y
conda activate crypto_bot
pip install -r requirements.txt

# Submit job
mkdir -p logs
sbatch --nodes=1 train.slurm    # 1 node  = 4 GPUs
sbatch --nodes=2 train.slurm    # 2 nodes = 8 GPUs
sbatch --nodes=12 train.slurm   # Max scale = 48 GPUs

# Monitor
tail -f logs/crypto_train_*.out
squeue -u $USER

# Cancel if needed
scancel <JOB_ID>
```

## 📊 Data Source

**Real Binance data - no API keys required!**

- Symbols: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, etc.
- Intervals: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`, etc.
- Fetches via public endpoints
- Auto-pagination for 10k+ candles
- Fallback to synthetic data if API fails

## 🏗️ What Makes This Different

1. **Causal Masking**: Position i can ONLY see positions ≤ i (no future leakage)
2. **Realistic Execution**: Trades at NEXT candle open + slippage + fees
3. **Temporal Splitting**: Train → Val → Test (no overlap in time)
4. **HPC Ready**: Multi-node distributed training with auto-resume

## ⚙️ Configuration

Edit `config.py`:
```python
# Data
symbol = "BTCUSDT"
interval = "1h"
lookback_window = 168  # 7 days
use_real_data = True

# Model
d_model = 128
num_encoder_layers = 4
causal = True  # KEEP THIS TRUE!

# Trading
initial_capital = 10000.0
position_size = 0.1  # 10% per trade
stop_loss_pct = 0.02  # 2%
min_confidence = 0.6
```

## 📈 Expected Results

| Metric | Random | Trained |
|--------|--------|---------|
| Accuracy | ~33% | 40-50% |
| Win Rate | ~50% | 50-60% |
| Sharpe | ~0 | 0.5-1.5 |

## 🔧 HPC Environment Variables

```bash
# Custom training config
export N_CANDLES=20000
export BATCH_SIZE=128
export EPOCHS=100
export SYMBOL=ETHUSDT
sbatch train.slurm
```

## 📁 Files

```
├── config.py          # All settings
├── data.py            # Binance fetcher + preprocessing
├── model.py           # Causal Transformer
├── train.py           # Training (DDP + AMP)
├── backtest.py        # Backtesting engine
├── main.py            # Entry point
├── train.slurm        # HPC job script
└── requirements.txt   # Dependencies
```

## ✅ What Works

- ✅ Real Binance data (no API key)
- ✅ Multi-node distributed training
- ✅ Mixed precision (2x speedup)
- ✅ Auto-resume after preemption
- ✅ Causal masking verified
- ✅ Realistic backtesting

## 🐛 Common Issues

**"No data fetched"**
```bash
# Use fake data as fallback
python main.py train --fake-data
```

**"CUDA out of memory"**
```bash
# Reduce batch size in config.py
batch_size = 32  # instead of 64
```

**"Job keeps failing on HPC"**
```bash
# Check logs
cat logs/crypto_train_*.err

# Verify environment
conda activate crypto_bot
python -c "import torch; print(torch.cuda.is_available())"
```

## ⚠️ Important

1. **Not financial advice** - This is educational/research code
2. **No real trading** - Use paper trading to validate
3. **Past performance ≠ future results**
4. **API keys** - Never commit to git (use env vars)
5. **Risk management** - Always use stop losses

## 🚦 Next Steps

1. Test locally: `python main.py demo`
2. Train small: `python main.py train --candles 2000`
3. Validate backtest results
4. If good → scale up on HPC
5. Paper trade before considering real money

## 📚 Key Concepts

**Causal Masking**: Prevents model from "seeing the future"
```python
# Position 5 can only see [0,1,2,3,4,5]
# Position 5 CANNOT see [6,7,8,...]
```

**Temporal Split**: Data ordered by time
```
Train: Jan-Jun → Val: Jul-Aug → Test: Sep-Oct
```

**Realistic Execution**: 
```
Signal at candle[i] → Execute at candle[i+1].open
```

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| No CUDA | Add `--device cpu` to commands |
| NCCL errors on HPC | Check SLURM logs, nodes may be down |
| Low accuracy | Normal! 40-50% is good for trading |
| Overfitting | Reduce model size or add dropout |

## 🎯 Performance Tuning

**For speed:**
- Enable AMP: `use_amp = True` 
- Increase batch size
- Use multiple nodes

**For accuracy:**
- More data: `--candles 20000`
- Tune hyperparameters
- Try different symbols/intervals
- Ensemble models

## 📄 License

MIT - Use at your own risk

---

**Ready to start?**
```bash
python main.py test-data  # Verify setup
python main.py demo       # Quick test
python main.py train      # Full training
```

