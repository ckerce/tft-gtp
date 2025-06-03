# configs/model_configs.py
"""
Model configurations for TFT and other transformer variants.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TFTConfig:
    """Configuration for Token-Factored Transformer with ALiBi."""
    
    # Model architecture
    vocab_size: int = 50257
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = None  # Will be set to 4 * d_model if None
    dropout: float = 0.1
    bias: bool = False
    
    # TFT-specific parameters
    use_v: bool = False
    use_proj: bool = False
    
    # ALiBi parameters
    block_size: int = 128  # Training sequence length
    max_position_embeddings: int = 512  # Max inference length
    
    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.max_position_embeddings >= self.block_size, "max_position_embeddings must be >= block_size"
    
    @classmethod
    def small(cls, **kwargs):
        defaults = dict(n_layers=6, n_heads=6, d_model=384, block_size=128)
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def medium(cls, **kwargs):
        defaults = dict(n_layers=12, n_heads=12, d_model=768, block_size=256)
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def large(cls, **kwargs):
        defaults = dict(n_layers=24, n_heads=16, d_model=1024, block_size=512)
        defaults.update(kwargs)
        return cls(**kwargs)


# Configuration presets
CONFIG_PRESETS = {
    'tiny': TFTConfig(
        n_layers=2,
        n_heads=2,
        d_model=128,
        block_size=128,
        max_position_embeddings=512,
        dropout=0.1,
        learning_rate=5e-4,
    ),
    'small': TFTConfig.small(),
    'medium': TFTConfig.medium(),
    'large': TFTConfig.large(),
    'debug': TFTConfig(
        n_layers=1,
        n_heads=2,
        d_model=64,
        block_size=16,
        max_position_embeddings=32,
        dropout=0.0,
    ),
}


def get_config(preset: str, **overrides) -> TFTConfig:
    """Get a configuration preset with optional overrides."""
    if preset not in CONFIG_PRESETS:
        available = list(CONFIG_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
    
    # Get the preset config
    config = CONFIG_PRESETS[preset]
    
    # Create a new config with overrides
    config_dict = {
        'vocab_size': config.vocab_size,
        'n_layers': config.n_layers,
        'n_heads': config.n_heads,
        'd_model': config.d_model,
        'd_ff': config.d_ff,
        'dropout': config.dropout,
        'bias': config.bias,
        'use_v': config.use_v,
        'use_proj': config.use_proj,
        'block_size': config.block_size,
        'max_position_embeddings': config.max_position_embeddings,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
    }
    
    # Apply overrides
    config_dict.update(overrides)
    
    return TFTConfig(**config_dict)


def print_config(config: TFTConfig, title: str = "TFT Configuration"):
    """Pretty print configuration."""
    print("=" * 60)
    print(f"{title.upper()}")
    print("=" * 60)
    
    print(f"\n🏗️  ARCHITECTURE:")
    print(f"  Model Type:           Token-Factored Transformer")
    print(f"  Layers:               {config.n_layers}")
    print(f"  Attention Heads:      {config.n_heads}")
    print(f"  Model Dimension:      {config.d_model}")
    print(f"  Feed-Forward Dim:     {config.d_ff}")
    print(f"  Vocabulary Size:      {config.vocab_size:,}")
    
    print(f"\n🔧 TFT FEATURES:")
    print(f"  Value Factorization:  {config.use_v}")
    print(f"  Output Projection:    {config.use_proj}")
    
    print(f"\n📏 SEQUENCE HANDLING:")
    print(f"  Training Length:      {config.block_size}")
    print(f"  Max Inference Length: {config.max_position_embeddings}")
    print(f"  Extrapolation Ratio:  {config.max_position_embeddings / config.block_size:.1f}x")
    
    print(f"\n⚙️  TRAINING:")
    print(f"  Learning Rate:        {config.learning_rate}")
    print(f"  Weight Decay:         {config.weight_decay}")
    print(f"  Dropout:              {config.dropout}")
    print(f"  Bias in Linear:       {config.bias}")
    
    # Estimate parameters
    token_emb_params = config.vocab_size * config.d_model
    layer_params = config.n_layers * (
        # Layer norms
        2 * config.d_model * 2 +
        # Attention (Q,K,V projections)
        config.d_model * 3 * config.d_model +
        # MLP
        config.d_model * config.d_ff + config.d_ff * config.d_model
    )
    estimated_params = token_emb_params + layer_params
    
    print(f"\n📊 ESTIMATED STATS:")
    print(f"  Parameters:           ~{estimated_params/1e6:.1f}M")
    print(f"  Memory per token:     ~{config.d_model * 4 / 1024:.1f} KB")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Test different configs
    for preset in ['tiny', 'small', 'medium']:
        print(f"\n{preset.upper()} PRESET:")
        config = get_config(preset)
        print(f"  {config.n_layers}L-{config.n_heads}H-{config.d_model}D")
        print(f"  {config.vocab_size:,} vocab, {config.block_size} block size")
        print(f"  ~{(config.vocab_size * config.d_model + config.n_layers * config.d_model * config.d_ff * 8)/1e6:.1f}M params")
    
    # Test overrides
    print(f"\nCUSTOM CONFIG:")
    custom = get_config('small', n_layers=8, use_v=True)
    print(f"  {custom.n_layers}L (overridden), factorization={custom.use_v}")