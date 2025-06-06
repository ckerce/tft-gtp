# src/config/model_configs.py (updated)
"""
Model configurations for TFT and other transformer variants.
Updated to include dictionary FFN parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TFTConfig:
    """Configuration for Token-Factored Transformer with ALiBi and optional Dictionary FFN."""
    
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
    
    # Dictionary FFN parameters
    use_dict_ffn: bool = False
    dict_vocab_size: Optional[int] = None  # Use subset of vocab, None = full vocab
    dict_loss_weight: float = 1.0  # Weight for dictionary reconstruction loss
    
    # ALiBi parameters
    block_size: int = 128  # Training sequence length
    max_position_embeddings: int = 512  # Max inference length
    
    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        # Set default dict vocab size to full vocab if not specified
        if self.dict_vocab_size is None:
            self.dict_vocab_size = self.vocab_size
            
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.max_position_embeddings >= self.block_size, "max_position_embeddings must be >= block_size"
        assert self.dict_vocab_size <= self.vocab_size, "dict_vocab_size cannot exceed vocab_size"
    
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
        return cls(**defaults)


# Configuration presets (updated)
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
    # Dictionary FFN presets for experimentation
    'tiny-dict': TFTConfig(
        n_layers=2,
        n_heads=2,
        d_model=128,
        block_size=128,
        max_position_embeddings=512,
        dropout=0.1,
        learning_rate=5e-4,
        use_dict_ffn=True,
        dict_vocab_size=1000,  # Reduced vocab for initial experiments
        dict_loss_weight=1.0,
    ),
    'small-dict': TFTConfig(
        n_layers=6,
        n_heads=6,
        d_model=384,
        block_size=128,
        use_dict_ffn=True,
        dict_vocab_size=5000,  # Reduced vocab
        dict_loss_weight=1.0,
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
        'use_dict_ffn': config.use_dict_ffn,
        'dict_vocab_size': config.dict_vocab_size,
        'dict_loss_weight': config.dict_loss_weight,
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
    
    print(f"\n📖 DICTIONARY FFN:")
    print(f"  Use Dictionary FFN:   {config.use_dict_ffn}")
    if config.use_dict_ffn:
        print(f"  Dict Vocab Size:      {config.dict_vocab_size:,}")
        print(f"  Dict Loss Weight:     {config.dict_loss_weight}")
        
        # Calculate dictionary parameters
        dict_params_per_layer = config.n_heads * config.dict_vocab_size * (config.d_model // config.n_heads)
        total_dict_params = config.n_layers * dict_params_per_layer
        print(f"  Dict Params/Layer:    {dict_params_per_layer/1e6:.1f}M")
        print(f"  Total Dict Params:    {total_dict_params/1e6:.1f}M")
    
    print(f"\n📏 SEQUENCE HANDLING:")
    print(f"  Training Length:      {config.block_size}")
    print(f"  Max Inference Length: {config.max_position_embeddings}")
    print(f"  Extrapolation Ratio:  {config.max_position_embeddings / config.block_size:.1f}x")
    
    print(f"\n⚙️  TRAINING:")
    print(f"  Learning Rate:        {config.learning_rate}")
    print(f"  Weight Decay:         {config.weight_decay}")
    print(f"  Dropout:              {config.dropout}")
    print(f"  Bias in Linear:       {config.bias}")
    
    print(f"\n📊 MODEL STATS:")
    print(f"  Use model.get_num_params() for actual parameter count")
    print(f"  Memory per token:     ~{config.d_model * 4 / 1024:.1f} KB")

    print("=" * 60)