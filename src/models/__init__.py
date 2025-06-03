# src/models/__init__.py
"""
Model registry for TFT and other transformer variants.
"""

from .model_tft_alibi import TokenFactoredTransformer
from .model_vanilla import VanillaTransformer
from config.model_configs import TFTConfig
from typing import Dict, Type, Any

# Model registry for extensibility
MODEL_REGISTRY: Dict[str, Type] = {
    'tft': TokenFactoredTransformer,
    'tft-alibi': TokenFactoredTransformer,  # Alias
    'vanilla': VanillaTransformer,
    'baseline': VanillaTransformer,  # Alias
}


def register_model(name: str, model_class: Type):
    """Register a new model type."""
    MODEL_REGISTRY[name] = model_class


def get_model(model_type: str, config: TFTConfig):
    """Get a model instance by type."""
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config)


def list_models() -> list:
    """List available model types."""
    return list(MODEL_REGISTRY.keys())


# Export main classes
__all__ = [
    'TokenFactoredTransformer',
    'VanillaTransformer',
    'TFTConfig', 
    'get_model',
    'register_model',
    'list_models',
    'MODEL_REGISTRY'
]