# WINNING CONFIGURATION - Sharpe 11.48
Date: 2026-01-05
Symbol: BNBUSDT
Interval: 1d
Lookback: 100

## Results
- Sharpe Ratio: 11.48
- Return: +19.39% (7 months)
- Win Rate: 61.90%
- Profit Factor: 4.30
- Max Drawdown: 3.03%
- Total Trades: 21

## Model Architecture
class ModelConfig:
    """Transformer model configuration."""
    # Input/Output
    input_dim: int = 22  
    output_dim: int = 3 
    
    # Transformer architecture
    d_model: int = 32
    nhead: int = 4
    num_encoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Positional encoding
    max_seq_len: int = 512
    
