"""
Transformer Policy Network for Reinforcement Learning
======================================================
Wraps the existing CausalTimeSeriesTransformer for use with PPO/A2C.

Key Features:
- Reuses proven transformer architecture
- Adds actor (policy) and critic (value) heads
- Compatible with Stable-Baselines3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym

from model import (
    PositionalEncoding,
    TransformerEncoderBlock,
    CausalTimeSeriesTransformer
)
from config import ModelConfig


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using Causal Transformer for temporal patterns.
    
    This is the core of our RL policy - it processes market data
    using the same transformer architecture that worked so well
    for classification (13.71 Sharpe on BNB!).
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        features_dim: int = 128
    ):
        """
        Args:
            observation_space: Observation space (Dict with 'market_data', 'position', etc.)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            features_dim: Output feature dimension
        """
        super().__init__(observation_space, features_dim=features_dim)
        
        # Get input dimensions from observation space
        market_data_shape = observation_space['market_data'].shape  # (lookback, n_features)
        self.lookback_window = market_data_shape[0]
        self.n_features = market_data_shape[1]
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=512,
            dropout=dropout
        )
        
        # Transformer encoder layers (REUSE existing architecture!)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Position embedding (add current position state)
        self.position_embed = nn.Embedding(3, 32)  # FLAT, LONG, SHORT
        
        # Combine transformer output with position info
        self.feature_combine = nn.Sequential(
            nn.Linear(d_model + 32 + 1, features_dim),  # +1 for position_value
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Dict with:
                - market_data: (batch, lookback, n_features)
                - position: (batch,) - current position
                - position_value: (batch, 1) - unrealized P&L
                
        Returns:
            features: (batch, features_dim)
        """
        market_data = observations['market_data']
        position = observations['position']
        position_value = observations['position_value']
        
        # FIXED: Robust handling of position_value dimensions
        # Always ensure it's (batch, 1)
        while position_value.dim() > 2:
            position_value = position_value.squeeze(-1)
        
        if position_value.dim() == 1:
            position_value = position_value.unsqueeze(-1)
        
        # At this point, position_value should be (batch, 1)
        
        # Project market data
        x = self.input_proj(market_data)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer layers with causal masking
        for layer in self.encoder_layers:
            x = layer(x, causal=True)
        
        # Final norm
        x = self.final_norm(x)
        
        # Use last position output (most recent state)
        x = x[:, -1, :]  # (batch, d_model)
        
        # Embed position state
        # Handle position being 2D from vectorized envs
        if position.dim() > 1:
            position = position.squeeze(-1)
        pos_embed = self.position_embed(position.long())  # (batch, 32)
        
        # Combine all features (all should be (batch, feature_dim) now)
        combined = torch.cat([x, pos_embed, position_value], dim=-1)
        features = self.feature_combine(combined)
        
        return features


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using Transformer feature extractor.
    
    This policy:
    - Uses transformer to extract temporal patterns (ACTOR)
    - Estimates state value for learning (CRITIC)
    - Compatible with PPO, A2C, etc.
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            **kwargs: Additional arguments for ActorCriticPolicy
        """
        # Store transformer config
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """
        Build the feature extractor.
        
        We override this to use our Transformer-based extractor
        instead of the default MLP.
        """
        self.features_extractor = TransformerFeatureExtractor(
            observation_space=self.observation_space,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            features_dim=128
        )
        
        # Create dummy mlp_extractor to satisfy stable-baselines3
        # The features_extractor already does all the work
        latent_dim = self.d_model
        
        class DummyMLPExtractor(nn.Module):
            """Dummy MLP extractor - features already extracted by transformer."""
            def __init__(self, latent_dim_pi, latent_dim_vf):
                super().__init__()
                self.latent_dim_pi = latent_dim_pi
                self.latent_dim_vf = latent_dim_vf
                self.policy_net = nn.Identity()
                self.value_net = nn.Identity()
            
            def forward(self, features):
                return features, features
        
        self.mlp_extractor = DummyMLPExtractor(latent_dim, latent_dim)


def create_transformer_policy(
    env: gym.Env,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 512,
    dropout: float = 0.1
):
    """
    Create a Transformer-based policy for the environment.
    
    This is a factory function to make it easy to create policies
    with different hyperparameters.
    
    Args:
        env: Trading environment
        d_model: Transformer dimension (default matches our best model)
        nhead: Number of heads (default 4, proven to work)
        num_layers: Number of layers (default 2, proven to work)
        dim_feedforward: FFN dimension
        dropout: Dropout rate
        
    Returns:
        Policy class configured for this environment
    """
    from functools import partial
    
    return partial(
        TransformerActorCriticPolicy,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )


class PretrainedTransformerPolicy(nn.Module):
    """
    Optional: Load pre-trained transformer weights from classification model.
    
    This allows us to:
    1. Load the proven transformer that got 13.71 Sharpe
    2. Add RL heads on top
    3. Fine-tune with RL (or freeze transformer, only train heads)
    """
    
    def __init__(
        self,
        pretrained_model_path: str,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        freeze_transformer: bool = False
    ):
        """
        Args:
            pretrained_model_path: Path to saved classification model
            observation_space: RL observation space
            action_space: RL action space
            freeze_transformer: Whether to freeze transformer weights
        """
        super().__init__()
        
        # Load pretrained model
        checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        
        # Create transformer with same config
        model_config = ModelConfig(**checkpoint['config'])
        self.transformer = CausalTimeSeriesTransformer(model_config)
        
        # Load pretrained weights
        self.transformer.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally freeze transformer weights
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Add position embedding
        self.position_embed = nn.Embedding(3, 32)
        
        # Policy head (actor)
        d_model = model_config.d_model
        self.policy_head = nn.Sequential(
            nn.Linear(d_model + 32 + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_space.n)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(d_model + 32 + 1, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits: (batch, n_actions)
            values: (batch, 1)
        """
        market_data = observations['market_data']
        position = observations['position']
        position_value = observations['position_value']
        
        # Get transformer features
        # Note: transformer expects (batch, seq, features)
        transformer_out = self.transformer(market_data, return_all_positions=False)
        
        # Add position info
        if position.dim() > 1:
            position = position.squeeze(-1)
        pos_embed = self.position_embed(position.long())
        
        # Ensure position_value is 2D
        while position_value.dim() > 2:
            position_value = position_value.squeeze(-1)
        if position_value.dim() == 1:
            position_value = position_value.unsqueeze(-1)
        
        combined = torch.cat([transformer_out, pos_embed, position_value], dim=-1)
        
        # Policy and value outputs
        action_logits = self.policy_head(combined)
        values = self.value_head(combined)
        
        return action_logits, values


if __name__ == "__main__":
    # Test the policy
    import gymnasium as gym
    from trading_env import TradingEnv
    from data import CryptoDataFetcher, compute_technical_indicators, FEATURE_COLS
    from config import Config
    
    config = Config()
    config.data.use_real_data = True
    
    # Fetch data
    fetcher = CryptoDataFetcher(config)
    df = fetcher.fetch_historical('BNBUSDT', '1d', n_candles=1000)
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
    
    print("Testing Transformer Policy...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create feature extractor
    extractor = TransformerFeatureExtractor(
        observation_space=env.observation_space,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        features_dim=128
    )
    
    print(f"\nFeature extractor created")
    print(f"Parameters: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # Test forward pass
    obs, _ = env.reset()
    
    # Convert to torch tensors and add batch dimension
    obs_tensor = {
        'market_data': torch.FloatTensor(obs['market_data']).unsqueeze(0),
        'position': torch.LongTensor([obs['position']]),
        'position_value': torch.FloatTensor(obs['position_value']).unsqueeze(0)
    }
    
    features = extractor(obs_tensor)
    print(f"\nFeature output shape: {features.shape}")
    print(f"Features: {features[0, :5]}")
    
    print("\n✓ Policy test successful!")
