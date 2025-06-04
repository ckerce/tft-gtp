# src/models/__init__.py
"""
Model registry for TFT variants.
"""

from .model_tft_alibi import TokenFactoredTransformer
from .model_vanilla import VanillaTransformer
from .model_tft_dict import TokenFactoredTransformerDict


def get_model(model_type: str, config):
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: Type of model ('tft', 'vanilla', 'tft-dict')
        config: Model configuration
        
    Returns:
        Instantiated model
    """
    model_registry = {
        'tft': TokenFactoredTransformer,
        'tft-alibi': TokenFactoredTransformer,  # Alias
        'vanilla': VanillaTransformer,
        'tft-dict': TokenFactoredTransformerDict,
        'dict': TokenFactoredTransformerDict,  # Short alias
    }
    
    if model_type not in model_registry:
        available_models = list(model_registry.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available_models}")
    
    model_class = model_registry[model_type]
    return model_class(config)


def list_models():
    """List all available model types."""
    return {
        'tft': 'Token-Factored Transformer with ALiBi',
        'vanilla': 'Standard Transformer baseline',
        'tft-dict': 'Token-Factored Transformer with Dictionary FFN',
    }


# Export main components
__all__ = [
    'get_model',
    'list_models',
    'TokenFactoredTransformer',
    'VanillaTransformer', 
    'TokenFactoredTransformerDict'
]