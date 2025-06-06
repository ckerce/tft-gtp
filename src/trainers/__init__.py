# src/trainers/__init__.py (Fixed with validation support)
"""
Trainers module for TFT training with validation support.
"""

from .simple_trainer import SimpleTrainer
from .base_trainer import BaseTrainer, Callback

# Import accelerate trainer if available
try:
    from .accelerate_trainer import AccelerateTrainer
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def get_trainer(trainer_type: str = 'simple', **kwargs):
    """
    Get a trainer instance with validation support.
    
    Args:
        trainer_type: Type of trainer ('simple' or 'accelerate')
        **kwargs: Arguments to pass to trainer constructor
        
    Returns:
        Trainer instance
    """
    if trainer_type == 'simple':
        return SimpleTrainer(**kwargs)  # Pass all kwargs directly
    
    elif trainer_type == 'accelerate':
        if not ACCELERATE_AVAILABLE:
            raise ImportError("AccelerateTrainer not available. Install accelerate or use 'simple' trainer.")
        return AccelerateTrainer(**kwargs)  # Pass all kwargs directly
    
    else:
        available_trainers = ['simple']
        if ACCELERATE_AVAILABLE:
            available_trainers.append('accelerate')
        raise ValueError(f"Unknown trainer type: {trainer_type}. Available: {available_trainers}")


# Export main classes and functions
__all__ = [
    'BaseTrainer',
    'SimpleTrainer', 
    'Callback',
    'get_trainer'
]

if ACCELERATE_AVAILABLE:
    __all__.append('AccelerateTrainer')