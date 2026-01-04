"""
Causal Time Series Transformer for Crypto Trading
===================================================
Implements a Transformer with CAUSAL MASKING to prevent lookahead bias.

Key Anti-Cheating Measures:
1. Causal attention mask: each position can only attend to previous positions
2. No future information leakage in positional encoding
3. Proper temporal ordering in input sequences
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import Config, ModelConfig


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Encodes position information without any future knowledge.
    Position 0 is the oldest data point, position T-1 is the most recent.
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with CAUSAL MASKING.
    
    CRITICAL: The causal mask ensures that position i can only attend
    to positions j <= i. This prevents any lookahead bias.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal mask where mask[i, j] = True if j > i (future positions).
        
        This mask is added to attention scores before softmax:
        - Positions where mask is True get -inf, resulting in 0 attention weight
        - Positions where mask is False are attended to normally
        """
        # Upper triangular matrix (excluding diagonal) = future positions
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
            causal: Whether to apply causal masking
            
        Returns:
            Output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, nhead, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # (batch, nhead, seq, seq)
        
        # Apply causal mask
        if causal:
            causal_mask = self._generate_causal_mask(seq_len, x.device)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch, nhead, seq, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block with Causal Self-Attention.
    
    Architecture:
    - Causal Multi-Head Self-Attention
    - Add & Norm
    - Feed-Forward Network
    - Add & Norm
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            causal: Whether to use causal masking
            
        Returns:
            Output: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual
        attn_out = self.self_attn(x, causal=causal)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class CausalTimeSeriesTransformer(nn.Module):
    """
    Causal Time Series Transformer for Trading Signal Classification.
    
    Architecture:
    1. Input Projection: Map features to d_model dimensions
    2. Positional Encoding: Add temporal information
    3. Transformer Encoder: Stack of causal attention blocks
    4. Output Head: Classify trading signal
    
    ANTI-LOOKAHEAD MEASURES:
    - Causal masking in all attention layers
    - Output at position t only depends on positions 0 to t
    - For prediction, we use the output at the LAST position only
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_positions: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with causal attention.
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            return_all_positions: If True, return outputs for all positions
            
        Returns:
            If return_all_positions:
                (batch_size, seq_len, output_dim)
            Else:
                (batch_size, output_dim) - output at last position only
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through encoder layers with causal masking
        for layer in self.encoder_layers:
            x = layer(x, causal=self.config.causal)
        
        # Output projection
        logits = self.output_head(x)  # (batch, seq, output_dim)
        
        if return_all_positions:
            return logits
        else:
            # Return only the last position (most recent prediction)
            return logits[:, -1, :]  # (batch, output_dim)
    
    def predict(
        self,
        x: torch.Tensor,
        return_confidence: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Make predictions with optional confidence scores.
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            return_confidence: Whether to return confidence scores
            
        Returns:
            predictions: (batch_size,) - predicted class
            confidences: (batch_size,) - confidence scores (if requested)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, return_all_positions=False)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            if return_confidence:
                confidences = torch.max(probs, dim=-1)[0]
                return predictions, confidences
            return predictions, None
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """
        Get attention weights for visualization.
        Useful for understanding what the model is attending to.
        """
        # This would require modifying forward to store attention weights
        # Placeholder for future implementation
        raise NotImplementedError("Attention weight extraction not implemented")


class TradingSignalLoss(nn.Module):
    """
    Custom loss function for trading signals.
    
    Combines:
    1. Cross-entropy for classification
    2. Optional class weighting (to handle imbalanced labels)
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
            
        Returns:
            Scalar loss
        """
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
        
        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        return loss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_causal_masking(model: CausalTimeSeriesTransformer, seq_len: int = 10) -> bool:
    """
    Verify that causal masking is working correctly.
    
    Test: Changing future inputs should NOT affect current outputs.
    
    Returns:
        True if causal masking is working correctly
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create test input
    batch_size = 2
    input_dim = model.config.input_dim
    x = torch.randn(batch_size, seq_len, input_dim, device=device)
    
    # Get output at position 5
    with torch.no_grad():
        out1 = model(x, return_all_positions=True)[:, 5, :]
    
    # Modify positions 6-9 (future relative to position 5)
    x_modified = x.clone()
    x_modified[:, 6:, :] = torch.randn(batch_size, seq_len - 6, input_dim, device=device)
    
    # Get output at position 5 again
    with torch.no_grad():
        out2 = model(x_modified, return_all_positions=True)[:, 5, :]
    
    # Check if outputs are the same (they should be with causal masking)
    is_causal = torch.allclose(out1, out2, atol=1e-5)
    
    if is_causal:
        print("✓ Causal masking verified: Future changes don't affect past outputs")
    else:
        print("✗ WARNING: Causal masking may not be working correctly!")
        print(f"  Max difference: {(out1 - out2).abs().max().item()}")
    
    return is_causal


if __name__ == "__main__":
    # Test model
    from config import Config
    
    config = Config()
    config.model.input_dim = 22  # Example feature count
    
    model = CausalTimeSeriesTransformer(config.model)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 168
    x = torch.randn(batch_size, seq_len, config.model.input_dim)
    
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Verify causal masking
    verify_causal_masking(model, seq_len=20)
    
    # Test prediction
    preds, confs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Confidences: {confs}")
