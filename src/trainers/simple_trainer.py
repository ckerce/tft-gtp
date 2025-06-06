# src/trainers/simple_trainer.py (Enhanced with validation support)
"""
Simple trainer implementation with validation support.
"""

import time
import logging
import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer, Callback

logger = logging.getLogger(__name__)


class SimpleTrainer(BaseTrainer):
    """
    Simple trainer implementation with validation support.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,
                 callbacks: Optional[List[Callback]] = None,
                 val_dataloader: Optional[DataLoader] = None,    # NEW
                 validate_every_n_epochs: int = 1):              # NEW
        """
        Initialize the simple trainer.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Device for training.
            num_epochs: Number of training epochs.
            output_dir: Directory to save outputs.
            clip_grad_norm: Maximum norm for gradient clipping.
            log_interval: Number of batches between logging.
            callbacks: List of callback instances.
            val_dataloader: Optional validation DataLoader.
            validate_every_n_epochs: Run validation every N epochs.
        """
        super().__init__(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            output_dir=output_dir,
            callbacks=callbacks,
            val_dataloader=val_dataloader,
            validate_every_n_epochs=validate_every_n_epochs
        )
        
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        logger.info(f"SimpleTrainer initialized:")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Device: {self.device}")
        if self.clip_grad_norm:
            logger.info(f"  Gradient Clipping: {self.clip_grad_norm}")
        if self.val_dataloader:
            logger.info(f"  Validation: Every {self.validate_every_n_epochs} epochs")
        if self.callbacks:
            logger.info(f"  Callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")

    def train(self) -> Dict[str, Any]:
        """Execute the training loop with validation support."""
        logger.info("Starting training...")
        
        self.model.train()
        
        # Trigger callbacks
        self.trainer_state['num_epochs'] = self.num_epochs
        self._trigger_callbacks('on_train_begin', logs=self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': float('nan'),
            'training_time': 0.0,
            'validation_losses': []  # NEW: track validation losses
        }

        for epoch in range(1, self.num_epochs + 1):
            self.trainer_state['current_epoch'] = epoch
            self._trigger_callbacks('on_epoch_begin', epoch, logs=self.trainer_state)
            
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches_processed = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                self.trainer_state['current_batch_idx'] = batch_idx
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                # Move data to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}

                # Forward pass
                outputs = self.model(**batch_data)
                loss = outputs.get('loss')

                if loss is None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue
                
                if torch.isnan(loss):
                    logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping.")
                    self.trainer_state['status'] = 'NaN Loss'
                    self._trigger_callbacks('on_train_end', logs=self.trainer_state)
                    training_metrics['training_time'] = time.time() - total_start_time
                    return training_metrics

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.optimizer.step()

                batch_loss_item = loss.item()
                epoch_loss += batch_loss_item
                num_batches_processed += 1

                # Update progress and log
                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item})
                progress_bar.set_postfix({"loss": f"{batch_loss_item:.4f}"})

                if batch_idx % self.log_interval == 0:
                    self.log_batch(batch_idx, batch_loss_item, epoch=epoch)

            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)

            epoch_duration = time.time() - epoch_start_time
            self.log_epoch(epoch, avg_epoch_loss)

            epoch_end_logs = {'loss': avg_epoch_loss, 'epoch_duration': epoch_duration}
            self.trainer_state.update(epoch_end_logs)
            
            # NEW: Run validation if enabled and it's time
            if (self.val_dataloader is not None and 
                epoch % self.validate_every_n_epochs == 0):
                val_metrics = self.validate()
                if val_metrics and 'val_loss' in val_metrics:
                    training_metrics['validation_losses'].append(val_metrics['val_loss'])
                    epoch_end_logs.update(val_metrics)
                    self.trainer_state.update(val_metrics)
            
            self._trigger_callbacks('on_epoch_end', epoch, logs=self.trainer_state)

            # Save checkpoint
            if self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                checkpoint_metrics = {'loss': avg_epoch_loss}
                if 'val_loss' in epoch_end_logs:
                    checkpoint_metrics['val_loss'] = epoch_end_logs['val_loss']
                self.save_checkpoint(checkpoint_path, epoch=epoch, **checkpoint_metrics)

        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")
        if training_metrics['validation_losses']:
            logger.info(f"Final validation loss: {training_metrics['validation_losses'][-1]:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model."""
        logger.info("Starting evaluation...")
        
        if eval_dataloader is None:
            logger.warning("eval_dataloader not provided. Using training dataloader.")
            eval_dataloader = self.dataloader

        self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)
        self._trigger_callbacks('on_evaluate_begin', logs=self.trainer_state)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            eval_iter = tqdm(eval_dataloader, desc="Evaluating")
            
            for batch_idx, batch_data in enumerate(eval_iter):
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                # Move data to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}

                outputs = self.model(**batch_data)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Eval Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue

                batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                num_batches_processed += 1

                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': loss.item()})

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        eval_metrics = {'loss': avg_loss}
        
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            eval_metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            eval_metrics['perplexity'] = float('nan')

        self.model.train()

        logger.info(f"Evaluation results: Loss: {eval_metrics['loss']:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
        self.trainer_state.update(eval_metrics)
        self._trigger_callbacks('on_evaluate_end', logs=self.trainer_state)

        return eval_metrics