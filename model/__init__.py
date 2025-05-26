# model/__init__.py
"""
Simple model registry for TFT-GPT.
"""

from .model_token_factored_alibi import FactoredTransformerModelALiBi

def get_model(model_type, config):
    """
    Simple model factory function.
    
    Args:
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Initialized model instance
    """
    if model_type.lower() in ['factored', 'factoredalibi', 'factored_alibi']:
        return FactoredTransformerModelALiBi(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

__all__ = ['FactoredTransformerModelALiBi', 'get_model']
