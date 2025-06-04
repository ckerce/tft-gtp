# src/utils/json_logger.py (Enhanced with perplexity)
"""
Enhanced JSON logger for training metrics with perplexity calculation.
"""

import json
import os
import time
import math
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import torch

logger = logging.getLogger(__name__)


class JSONLogger:
    """
    Enhanced JSON logger for training metrics with automatic perplexity calculation.
    """
    
    def __init__(self, log_file: str, run_name: Optional[str] = None):
        """
        Initialize the JSON logger.
        
        Args:
            log_file: Path to JSON log file
            run_name: Optional name for this training run
        """
        self.log_file = log_file
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize log structure
        self.log_data = {
            "run_info": {
                "run_name": self.run_name,
                "start_time": datetime.now().isoformat(),
                "status": "running"
            },
            "config": {},
            "metrics": {
                "train": [],
                "eval": [],
                "epochs": [],
                "steps": []
            },
            "events": []
        }
        
        # Save initial structure
        self._save()
        logger.info(f"JSON logger initialized: {log_file}")
    
    def _calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss."""
        try:
            if loss is None or math.isnan(loss) or math.isinf(loss):
                return float('nan')
            # Perplexity = exp(loss)
            perplexity = math.exp(loss)
            # Cap extremely large perplexities for numerical stability
            return min(perplexity, 1e6)
        except (OverflowError, ValueError):
            return float('inf')
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.log_data["config"] = config
        self._save()
        logger.info("Configuration logged")
    
    def log_step(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """
        Log metrics for a training step with automatic perplexity calculation.
        
        Args:
            step: Step number
            metrics: Dictionary of metrics (loss, lr, etc.)
            phase: Training phase ('train', 'eval')
        """
        entry = {
            "step": step,
            "timestamp": time.time(),
            "phase": phase,
            **metrics
        }
        
        # Add perplexity if loss is present
        if 'loss' in metrics:
            entry['perplexity'] = self._calculate_perplexity(metrics['loss'])
        
        self.log_data["metrics"]["steps"].append(entry)
        self._save()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log epoch-level metrics with automatic perplexity calculation.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics
        }
        
        # Add perplexity calculations for various loss types
        for loss_key in ['loss', 'avg_loss', 'train_loss', 'eval_loss']:
            if loss_key in metrics:
                perplexity_key = loss_key.replace('loss', 'perplexity')
                entry[perplexity_key] = self._calculate_perplexity(metrics[loss_key])
        
        self.log_data["metrics"]["epochs"].append(entry)
        self._save()
        
        # Enhanced logging message
        loss_info = f"loss: {metrics.get('loss', metrics.get('avg_loss', 'N/A'))}"
        if 'perplexity' in entry:
            loss_info += f", perplexity: {entry['perplexity']:.2f}"
        logger.info(f"Epoch {epoch} metrics logged: {loss_info}")
    
    def log_train_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log training metrics with perplexity."""
        entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        # Add perplexity if loss is present
        if 'loss' in metrics:
            entry['perplexity'] = self._calculate_perplexity(metrics['loss'])
        
        self.log_data["metrics"]["train"].append(entry)
        self._save()
    
    def log_eval_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics with perplexity."""
        entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        # Add perplexity for evaluation metrics
        for loss_key in ['eval_loss', 'loss']:
            if loss_key in metrics:
                perplexity_key = loss_key.replace('loss', 'perplexity') if loss_key != 'loss' else 'perplexity'
                entry[perplexity_key] = self._calculate_perplexity(metrics[loss_key])
        
        self.log_data["metrics"]["eval"].append(entry)
        self._save()
    
    def log_event(self, event: str, data: Optional[Dict[str, Any]] = None):
        """
        Log a training event (checkpoint saved, early stopping, etc.).
        
        Args:
            event: Event description
            data: Optional additional data
        """
        entry = {
            "timestamp": time.time(),
            "event": event,
            "data": data or {}
        }
        
        self.log_data["events"].append(entry)
        self._save()
        logger.info(f"Event logged: {event}")
    
    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """Mark training as finished with final perplexity."""
        self.log_data["run_info"]["end_time"] = datetime.now().isoformat()
        self.log_data["run_info"]["status"] = "completed"
        
        if final_metrics:
            enhanced_final_metrics = final_metrics.copy()
            # Add final perplexity
            if 'final_loss' in final_metrics:
                enhanced_final_metrics['final_perplexity'] = self._calculate_perplexity(final_metrics['final_loss'])
            self.log_data["run_info"]["final_metrics"] = enhanced_final_metrics
        
        self._save()
        logger.info("Training run marked as completed")
    
    def _save(self):
        """Save current log data to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save JSON log: {e}")


class MetricsAggregator:
    """
    Enhanced utility to aggregate metrics with perplexity calculation.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += float(value)
            self.counts[key] += 1
    
    def get_averages(self) -> Dict[str, float]:
        """Get averaged metrics with automatic perplexity calculation - FIXED VERSION."""
        averages = {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }
        
        # Add perplexity for any loss metrics
        # FIX: Create a list of items first to avoid modifying dict during iteration
        loss_items = [(key, value) for key, value in averages.items() if 'loss' in key.lower()]
        
        for key, value in loss_items:
            perplexity_key = key.replace('loss', 'perplexity').replace('Loss', 'Perplexity')
            try:
                averages[perplexity_key] = math.exp(value) if not math.isnan(value) else float('nan')
            except (OverflowError, ValueError):
                averages[perplexity_key] = float('inf')
        
        return averages
    
    def get_totals(self) -> Dict[str, float]:
        """Get total metrics."""
        return self.metrics.copy()


# Enhanced callback for automatic perplexity logging
class JSONLoggingCallback:
    """
    Enhanced callback that automatically logs metrics with perplexity to JSON.
    """
    
    def __init__(self, output_dir: str, run_name: Optional[str] = None, 
                 log_every_n_steps: int = 50):
        """
        Initialize the callback.
        
        Args:
            output_dir: Directory to save the log file
            run_name: Optional run name
            log_every_n_steps: Log detailed step metrics every N steps
        """
        self.output_dir = output_dir
        self.log_every_n_steps = log_every_n_steps
        
        # Create logger
        log_file = os.path.join(output_dir, "training_metrics.json")
        self.logger = JSONLogger(log_file, run_name)
        
        # Metrics aggregation
        self.epoch_aggregator = MetricsAggregator()
        self.step_count = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        if logs:
            # Extract config from logs if available
            config_data = {}
            for key, value in logs.items():
                if key not in ['current_epoch', 'current_batch_idx']:
                    config_data[key] = value
            self.logger.log_config(config_data)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        self.epoch_aggregator.reset()
    
    def on_batch_end(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each batch."""
        if logs and 'loss' in logs:
            loss = logs['loss']
            if loss is not None and not math.isnan(loss):
                self.epoch_aggregator.update(loss=loss)
                
                # Log detailed step metrics periodically
                if self.step_count % self.log_every_n_steps == 0:
                    step_metrics = {'loss': loss}
                    # Add learning rate if available
                    if 'learning_rate' in logs:
                        step_metrics['learning_rate'] = logs['learning_rate']
                    self.logger.log_step(self.step_count, step_metrics, phase='train')
                
                self.step_count += 1
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        # Get aggregated metrics
        epoch_metrics = self.epoch_aggregator.get_averages()
        
        # Add any additional metrics from logs
        if logs:
            for key, value in logs.items():
                if key not in ['current_epoch', 'current_batch_idx'] and value is not None:
                    epoch_metrics[key] = value
        
        # Log epoch metrics (perplexity will be calculated automatically)
        if epoch_metrics:
            self.logger.log_epoch(epoch, epoch_metrics)
    
    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called after evaluation."""
        if logs:
            # Extract evaluation metrics
            eval_metrics = {}
            for key, value in logs.items():
                if 'eval' in key.lower() or key in ['loss', 'perplexity']:
                    eval_metrics[key] = value
            
            if eval_metrics:
                self.logger.log_eval_metrics(
                    epoch=logs.get('current_epoch', 0),
                    step=self.step_count,
                    metrics=eval_metrics
                )
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        final_metrics = {}
        if logs:
            # Extract final metrics
            for key, value in logs.items():
                if 'final' in key.lower() or 'training_time' in key:
                    final_metrics[key] = value
        
        self.logger.finish(final_metrics)


# Convenience function for quick setup
def setup_json_logging(output_dir: str, run_name: Optional[str] = None) -> JSONLogger:
    """
    Quick setup for JSON logging with perplexity calculation.
    
    Args:
        output_dir: Output directory
        run_name: Optional run name
        
    Returns:
        Configured JSONLogger with perplexity support
    """
    log_file = os.path.join(output_dir, "training_log.json")
    return JSONLogger(log_file, run_name)