# ./trainers/simple_trainer.py
"""
Simple Trainer Implementation with Validation
Basic training loop with progress tracking, callback integration, and simple validation.
"""

import time
import logging
from typing import Dict, Any, Optional, List
import os
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer, Callback

logger = logging.getLogger(__name__)

class SimpleTrainer(BaseTrainer):
    """
    Simple trainer implementation with validation support.
    Automatically splits training data for validation if no eval dataloader is provided.
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
                 # Validation parameters
                 validation_split: float = 0.1,
                 validate_every_n_epochs: int = 1):
        """
        Initialize the simple trainer with validation.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Device to train on.
            num_epochs: Number of training epochs.
            output_dir: Directory to save outputs.
            clip_grad_norm: Maximum norm for gradient clipping.
            log_interval: Number of batches between logging.
            callbacks: List of Callback instances.
            validation_split: Fraction of training data to use for validation.
            validate_every_n_epochs: Run validation every N epochs.
        """
        
        # Split training data for validation
        self.validate_every_n_epochs = validate_every_n_epochs
        self.eval_dataloader = None
        
        if validation_split > 0:
            # Split training data for validation
            train_dataloader, eval_dataloader = self._split_dataloader(dataloader, validation_split)
            dataloader = train_dataloader
            self.eval_dataloader = eval_dataloader
            logger.info(f"Split data: {validation_split:.1%} for validation")
        
        super().__init__(model, dataloader, optimizer, device, output_dir, callbacks)
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval

        logger.info(f"SimpleTrainer initialized with {self.num_epochs} epochs.")
        if self.clip_grad_norm:
            logger.info(f"Gradient clipping enabled with max norm: {self.clip_grad_norm}")
        if self.eval_dataloader:
            logger.info(f"Validation enabled, running every {validate_every_n_epochs} epoch(s)")
        if self.callbacks:
            logger.info(f"Attached callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")

    def _split_dataloader(self, dataloader: DataLoader, validation_split: float):
        """Split dataloader into train and validation sets."""
        dataset = dataloader.dataset
        dataset_size = len(dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        # Create indices for train/val split
        indices = list(range(dataset_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        # Create new dataloaders with same parameters
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,  # Keep shuffling for training
            collate_fn=dataloader.collate_fn,
            num_workers=dataloader.num_workers,
            drop_last=dataloader.drop_last
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,  # No shuffling for validation
            collate_fn=dataloader.collate_fn,
            num_workers=dataloader.num_workers,
            drop_last=False
        )
        
        logger.info(f"Data split: {len(train_dataset)} train, {len(val_dataset)} validation")
        return train_dataloader, val_dataloader

    def train(self) -> Dict[str, Any]:
        """Execute the training loop with validation."""
        logger.info("Starting training...")
        self.model.to(self.device)
        self.model.train()

        self.trainer_state['num_epochs'] = self.num_epochs
        self._trigger_callbacks('on_train_begin', logs=self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'val_losses': [],
            'final_loss': float('nan'),
            'final_val_loss': float('nan'),
            'training_time': 0.0
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

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping optimization.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue
                if torch.isnan(loss):
                    logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping training.")
                    self.trainer_state['status'] = 'NaN Loss'
                    self._trigger_callbacks('on_train_end', logs=self.trainer_state)
                    training_metrics['training_time'] = time.time() - total_start_time
                    return training_metrics

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_grad_norm
                    )

                self.optimizer.step()

                batch_loss_item = loss.item()
                epoch_loss += batch_loss_item
                num_batches_processed += 1

                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item})
                progress_bar.set_postfix({"loss": f"{batch_loss_item:.4f}"})

                if batch_idx % self.log_interval == 0:
                    self.log_batch(batch_idx, batch_loss_item, epoch=epoch)

            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)

            epoch_duration = time.time() - epoch_start_time
            
            # Run validation if available and scheduled
            val_metrics = {}
            if self.eval_dataloader and epoch % self.validate_every_n_epochs == 0:
                val_results = self.evaluate(self.eval_dataloader)
                val_loss = val_results.get('loss', float('nan'))
                training_metrics['val_losses'].append(val_loss)
                val_metrics.update(val_results)
                logger.info(f"Validation - Loss: {val_loss:.6f}, Perplexity: {val_results.get('perplexity', 'N/A'):.6f}")

            # Log epoch with validation metrics
            log_metrics = {'val_loss': val_metrics.get('loss'), 'val_perplexity': val_metrics.get('perplexity')}
            self.log_epoch(epoch, avg_epoch_loss, metrics=log_metrics)

            epoch_end_logs = {
                'loss': avg_epoch_loss, 
                'epoch_duration': epoch_duration,
                **val_metrics
            }
            self.trainer_state.update(epoch_end_logs)
            self._trigger_callbacks('on_epoch_end', epoch, logs=self.trainer_state)

            if self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                checkpoint_data = {'loss': avg_epoch_loss}
                if val_metrics:
                    checkpoint_data['val_loss'] = val_metrics.get('loss')
                self.save_checkpoint(checkpoint_path, epoch=epoch, **checkpoint_data)

        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        if training_metrics['val_losses']:
            training_metrics['final_val_loss'] = training_metrics['val_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Final training loss: {training_metrics['final_loss']:.6f}")
        if training_metrics['final_val_loss'] != float('nan'):
            logger.info(f"Final validation loss: {training_metrics['final_val_loss']:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model on validation data."""
        if eval_dataloader is None:
            eval_dataloader = self.eval_dataloader
            
        if eval_dataloader is None:
            logger.warning("No validation data available. Skipping evaluation.")
            return {}

        logger.debug("Starting validation...")
        self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)
        self._trigger_callbacks('on_evaluate_begin', logs=self.trainer_state)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_dataloader, desc="Validating", leave=False)):
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Validation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue

                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
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
        
        logger.debug(f"Validation results: Loss: {eval_metrics['loss']:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
        self.trainer_state.update(eval_metrics)
        self._trigger_callbacks('on_evaluate_end', logs=self.trainer_state)

        return eval_metrics