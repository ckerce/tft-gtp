# src/trainers/accelerate_trainer.py
"""
Accelerate-enhanced trainer for distributed training and mixed precision.
Drop-in replacement for SimpleTrainer with scaling capabilities.
"""

import time
import logging
import os
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

from .base_trainer import BaseTrainer, Callback

logger = logging.getLogger(__name__)


class AccelerateTrainer(BaseTrainer):
    """
    Accelerate-enhanced trainer with multi-GPU support and mixed precision.
    
    This trainer is a drop-in replacement for SimpleTrainer but adds:
    - Multi-GPU/multi-node training
    - Mixed precision (fp16/bf16)
    - Gradient accumulation
    - Better memory management
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device = None,  # Will be handled by Accelerator
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,
                 callbacks: Optional[List[Callback]] = None,
                 # Accelerate-specific options
                 gradient_accumulation_steps: int = 1,
                 mixed_precision: str = "no",  # "no", "fp16", "bf16"
                 seed: Optional[int] = None,
                 dataloader_config: Optional[Dict] = None):
        """
        Initialize the Accelerate trainer.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Ignored - Accelerator handles device placement.
            num_epochs: Number of training epochs.
            output_dir: Directory to save outputs.
            clip_grad_norm: Maximum norm for gradient clipping.
            log_interval: Number of batches between logging.
            callbacks: List of callback instances.
            gradient_accumulation_steps: Steps to accumulate gradients.
            mixed_precision: Mixed precision mode ("no", "fp16", "bf16").
            seed: Random seed for reproducibility.
            dataloader_config: Additional dataloader configuration.
        """
        # Initialize accelerator first
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=None,  # We handle logging ourselves
            project_dir=output_dir,
        )
        
        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)
            self.seed = seed
        else:
            self.seed = None
        
        # Prepare model, optimizer, and dataloader with Accelerator
        model, optimizer, dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )
        
        # Initialize base trainer with prepared components
        # Note: device is now handled by accelerator
        super().__init__(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=self.accelerator.device,
            output_dir=output_dir,
            callbacks=callbacks
        )
        
        # Store accelerate-specific parameters
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Update trainer state with accelerate info
        self.trainer_state.update({
            'accelerator_state': {
                'num_processes': self.accelerator.num_processes,
                'process_index': self.accelerator.process_index,
                'device': str(self.accelerator.device),
                'mixed_precision': mixed_precision,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'is_main_process': self.accelerator.is_main_process,
            }
        })
        
        # Only log from main process
        if self.accelerator.is_main_process:
            logger.info(f"AccelerateTrainer initialized:")
            logger.info(f"  Processes: {self.accelerator.num_processes}")
            logger.info(f"  Device: {self.accelerator.device}")
            logger.info(f"  Mixed Precision: {mixed_precision}")
            logger.info(f"  Gradient Accumulation: {gradient_accumulation_steps}")
            logger.info(f"  Epochs: {self.num_epochs}")
            if self.clip_grad_norm:
                logger.info(f"  Gradient Clipping: {self.clip_grad_norm}")
            if self.callbacks:
                logger.info(f"  Callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")

    def train(self) -> Dict[str, Any]:
        """Execute the training loop with Accelerator."""
        if self.accelerator.is_main_process:
            logger.info("Starting training with Accelerator...")
        
        self.model.train()
        
        # Trigger callbacks only on main process
        if self.accelerator.is_main_process:
            self.trainer_state['num_epochs'] = self.num_epochs
            self._trigger_callbacks('on_train_begin', logs=self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': float('nan'),
            'training_time': 0.0
        }

        for epoch in range(1, self.num_epochs + 1):
            if self.accelerator.is_main_process:
                self.trainer_state['current_epoch'] = epoch
                self._trigger_callbacks('on_epoch_begin', epoch, logs=self.trainer_state)
            
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches_processed = 0

            # Create progress bar only on main process
            if self.accelerator.is_main_process:
                progress_bar = tqdm(
                    self.dataloader,
                    desc=f"Epoch {epoch}/{self.num_epochs}",
                    leave=False
                )
                dataloader_iter = progress_bar
            else:
                dataloader_iter = self.dataloader

            for batch_idx, batch_data in enumerate(dataloader_iter):
                if self.accelerator.is_main_process:
                    self.trainer_state['current_batch_idx'] = batch_idx
                    batch_logs = {'batch_data_keys': list(batch_data.keys())}
                    self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                # Data is already on the right device thanks to accelerator.prepare()
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch_data)
                    loss = outputs.get('loss')

                    if loss is None:
                        if self.accelerator.is_main_process:
                            logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping.")
                            self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                        continue
                    
                    if torch.isnan(loss):
                        if self.accelerator.is_main_process:
                            logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping.")
                            self.trainer_state['status'] = 'NaN Loss'
                            self._trigger_callbacks('on_train_end', logs=self.trainer_state)
                        training_metrics['training_time'] = time.time() - total_start_time
                        return training_metrics

                    # Backward pass with accelerator
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.clip_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Gather loss from all processes for logging
                loss_for_logging = self.accelerator.gather(loss).mean()
                batch_loss_item = loss_for_logging.item()
                
                epoch_loss += batch_loss_item
                num_batches_processed += 1

                # Update progress and log (main process only)
                if self.accelerator.is_main_process:
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item})
                    progress_bar.set_postfix({"loss": f"{batch_loss_item:.4f}"})

                    if batch_idx % self.log_interval == 0:
                        self.log_batch(batch_idx, batch_loss_item, epoch=epoch)

            # Wait for all processes to finish the epoch
            self.accelerator.wait_for_everyone()

            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)

            if self.accelerator.is_main_process:
                epoch_duration = time.time() - epoch_start_time
                self.log_epoch(epoch, avg_epoch_loss)

                epoch_end_logs = {'loss': avg_epoch_loss, 'epoch_duration': epoch_duration}
                self.trainer_state.update(epoch_end_logs)
                self._trigger_callbacks('on_epoch_end', epoch, logs=self.trainer_state)

                # Save checkpoint (only main process)
                if self.output_dir:
                    checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                    self.save_checkpoint(checkpoint_path, epoch=epoch, loss=avg_epoch_loss)

        if training_metrics['epoch_losses']:
            training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time

        if self.accelerator.is_main_process:
            logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
            logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")

            self.trainer_state['status'] = 'Completed'
            self.trainer_state.update(training_metrics)
            self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Evaluate the model with Accelerator support."""
        if self.accelerator.is_main_process:
            logger.info("Starting evaluation with Accelerator...")
        
        if eval_dataloader is None:
            if self.accelerator.is_main_process:
                logger.warning("eval_dataloader not provided. Using training dataloader.")
            eval_dataloader = self.dataloader
        else:
            # Prepare eval dataloader if it's not already prepared
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        if self.accelerator.is_main_process:
            self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)
            self._trigger_callbacks('on_evaluate_begin', logs=self.trainer_state)

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        num_batches_processed = 0

        with torch.no_grad():
            # Create progress bar only on main process
            if self.accelerator.is_main_process:
                eval_iter = tqdm(eval_dataloader, desc="Evaluating")
            else:
                eval_iter = eval_dataloader

            for batch_idx, batch_data in enumerate(eval_iter):
                if self.accelerator.is_main_process:
                    batch_logs = {'batch_data_keys': list(batch_data.keys())}
                    self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                outputs = self.model(**batch_data)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    if self.accelerator.is_main_process:
                        logger.warning(f"Eval Batch {batch_idx}: Loss is None or NaN. Skipping.")
                        self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue

                # Gather loss from all processes
                loss_gathered = self.accelerator.gather(loss)
                batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).size(0)
                
                # Accumulate metrics
                total_loss += loss_gathered.sum().item()
                total_samples += batch_size * self.accelerator.num_processes
                num_batches_processed += 1

                if self.accelerator.is_main_process:
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': loss.item()})

        # Wait for all processes
        self.accelerator.wait_for_everyone()

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        eval_metrics = {'loss': avg_loss}
        
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            eval_metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            eval_metrics['perplexity'] = float('nan')

        self.model.train()

        if self.accelerator.is_main_process:
            logger.info(f"Evaluation results: Loss: {eval_metrics['loss']:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
            self.trainer_state.update(eval_metrics)
            self._trigger_callbacks('on_evaluate_end', logs=self.trainer_state)

        return eval_metrics

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, **kwargs):
        """Save checkpoint using Accelerator (only on main process)."""
        if not self.accelerator.is_main_process:
            return
        
        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        checkpoint = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'accelerator_state': self.trainer_state.get('accelerator_state', {}),
            'seed': self.seed,
            **kwargs
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str, map_location: Optional[str] = None):
        """Load checkpoint with Accelerator support."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # Load on CPU first, then let accelerator handle device placement
        checkpoint = torch.load(path, map_location='cpu')
        
        # Unwrap model for loading
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning("Optimizer state not found in checkpoint.")

        if self.accelerator.is_main_process:
            logger.info(f"Checkpoint loaded from: {path}")
        
        return checkpoint