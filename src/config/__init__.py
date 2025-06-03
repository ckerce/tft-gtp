# configs/__init__.py
"""
Configuration system for TFT models.
"""

from .model_configs import TFTConfig, get_config, print_config, CONFIG_PRESETS

# Export main functions
__all__ = ["TFTConfig", "get_config", "print_config", "CONFIG_PRESETS"]