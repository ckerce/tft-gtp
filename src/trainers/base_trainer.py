# src/trainers/base_trainer.py (Enhanced with validation support)
"""
Base trainer class with validation support and callback system.
"""

import time
import logging
import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class for training events."""
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each batch."""
        pass
    
    def on_evaluate_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of evaluation."""
        pass
    
    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of evaluation."""
        pass
    
    def on_validate_begin(self, logs: Optional[Dict[str, Any]] = None):
        """NEW: Called at the beginning of validation."""
        pass
    
    def on_validate_end(self, logs: Optional[Dict[str, Any]] = None):
        """NEW: Called at the end of validation."""
        pass


class BaseTrainer(ABC):
    """
    Base trainer class with callback system and validation support.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 output_dir: Optional[str] = None,
                 callbacks: Optional[List[Callback]] = None,
                 val_dataloader: Optional[DataLoader] = None,  # NEW: validation dataloader
                 validate_every_n_epochs: int = 1):             # NEW: validation frequency
        """
        Initialize the base trainer.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Device for training.
            output_dir: Directory to save outputs.
            callbacks: List of callback instances.
            val_dataloader: Optional validation DataLoader.
            validate_every_n_epochs: Run validation every N epochs.
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.callbacks = callbacks or []
        self.val_dataloader = val_dataloader        # NEW
        self.validate_every_n_epochs = validate_every_n_epochs  # NEW
        
        # Trainer state for callbacks
        self.trainer_state = {
            'model_name': self.model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'device': str(device),
            'output_dir': output_dir,
            'has_validation': val_dataloader is not None,  # NEW
        }
        
        # Move model to device
        self.model.to(device)
        
        logger.info(f"BaseTrainer initialized on {device}")
        if self.val_dataloader:  # NEW
            logger.info(f"Validation enabled: {len(self.val_dataloader)} batches every {validate_every_n_epochs} epochs")

    def _trigger_callbacks(self, method_name: str, *args, **kwargs):
        """Trigger a callback method on all registered callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                try:
                    getattr(callback, method_name)(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback {callback.__class__.__name__}.{method_name} failed: {e}")

    def validate(self) -> Dict[str, Any]:
        """NEW: Run validation on the validation dataset."""
        if self.val_dataloader is None:
            logger.warning("No validation dataloader provided")
            return {}
        
        logger.info("Running validation...")
        self._trigger_callbacks('on_validate_begin', logs=self.trainer_state)
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_dataloader):
                # Move data to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                outputs = self.model(**batch_data)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Validation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    continue

                batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                num_batches_processed += 1

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        val_metrics = {'val_loss': avg_loss}
        
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            val_metrics['val_perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            val_metrics['val_perplexity'] = float('nan')

        self.model.train()
        
        logger.info(f"Validation results: Loss: {val_metrics['val_loss']:.6f}, Perplexity: {val_metrics['val_perplexity']:.6f}")
        
        # Update trainer state and trigger callbacks
        self.trainer_state.update(val_metrics)
        self._trigger_callbacks('on_validate_end', logs=self.trainer_state)

        return val_metrics

    def log_batch(self, batch_idx: int, loss: float, epoch: Optional[int] = None):
        """Log batch-level metrics."""
        log_msg = f"Batch {batch_idx}: loss={loss:.6f}"
        if epoch is not None:
            log_msg = f"Epoch {epoch}, " + log_msg
        logger.info(log_msg)

    def log_epoch(self, epoch: int, loss: float):
        """Log epoch-level metrics."""
        logger.info(f"Epoch {epoch}: avg_loss={loss:.6f}")

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Execute the training loop. Must be implemented by subclasses."""
        pass

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model. Can be overridden by subclasses."""
        logger.info("Base evaluation method - should be overridden by subclasses")
        return {}

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, **kwargs):
        """Save a training checkpoint."""
        if self.output_dir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                **kwargs
            }
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint