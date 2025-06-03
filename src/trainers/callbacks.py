# src/trainers/callbacks.py
"""
Training callbacks including JSON logging for metrics tracking.
"""

import os
import logging
import time
from typing import Dict, Optional, Any

from .base_trainer import Callback
from src.utils.json_logger import JSONLogger, MetricsAggregator

logger = logging.getLogger(__name__)


class JSONLoggingCallback(Callback):
    """
    Callback that logs all training metrics to JSON for easy plotting.
    """
    
    def __init__(self, 
                 output_dir: str,
                 run_name: Optional[str] = None,
                 log_every_n_steps: int = 50,
                 log_eval_metrics: bool = True):
        """
        Initialize JSON logging callback.
        
        Args:
            output_dir: Directory to save logs
            run_name: Name for this training run
            log_every_n_steps: Log step metrics every N steps
            log_eval_metrics: Whether to log evaluation metrics
        """
        super().__init__()
        
        self.output_dir = output_dir
        self.run_name = run_name
        self.log_every_n_steps = log_every_n_steps
        self.log_eval_metrics = log_eval_metrics
        
        # Will be initialized in on_train_begin
        self.json_logger = None
        self.metrics_agg = MetricsAggregator()
        
        # Tracking
        self.step_count = 0
        self.epoch_start_time = None
        self.train_start_time = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize logging at start of training."""
        # Create JSON logger
        log_file = os.path.join(self.output_dir, "training_metrics.json")
        self.json_logger = JSONLogger(log_file, self.run_name)
        
        # Log configuration
        config_to_log = {}
        
        # Get model config if available
        if hasattr(self.model, 'config'):
            model_config = self.model.config
            if hasattr(model_config, '__dict__'):
                config_to_log['model'] = vars(model_config)
            elif hasattr(model_config, '_asdict'):  # namedtuple
                config_to_log['model'] = model_config._asdict()
        
        # Get training config
        if logs:
            config_to_log['training'] = logs
        
        # Add system info
        config_to_log['system'] = {
            'device': str(self.device) if self.device else 'unknown',
            'num_epochs': getattr(self, 'num_epochs', None),
        }
        
        self.json_logger.log_config(config_to_log)
        self.json_logger.log_event("training_started")
        
        self.train_start_time = time.time()
        logger.info("JSON logging started")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Finalize logging at end of training."""
        if self.json_logger:
            # Calculate final metrics
            total_time = time.time() - self.train_start_time if self.train_start_time else 0
            
            final_metrics = {
                'total_training_time': total_time,
                'total_steps': self.step_count,
            }
            
            if logs:
                final_metrics.update(logs)
            
            self.json_logger.finish(final_metrics)
            self.json_logger.log_event("training_completed", {
                'total_time': total_time,
                'total_steps': self.step_count
            })
            
            logger.info(f"JSON logging completed. Total time: {total_time:.2f}s, Steps: {self.step_count}")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Reset metrics at start of epoch."""
        self.metrics_agg.reset()
        self.epoch_start_time = time.time()
        
        if self.json_logger:
            self.json_logger.log_event("epoch_started", {'epoch': epoch})
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log epoch metrics."""
        if not self.json_logger:
            return
        
        # Calculate epoch duration
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Get averaged metrics for this epoch
        avg_metrics = self.metrics_agg.get_averages()
        
        # Add epoch-specific info
        epoch_metrics = {
            'duration': epoch_duration,
            **avg_metrics
        }
        
        # Add any additional metrics from logs
        if logs:
            epoch_metrics.update(logs)
        
        # Log to JSON
        self.json_logger.log_epoch(epoch, epoch_metrics)
        self.json_logger.log_event("epoch_completed", {
            'epoch': epoch,
            'duration': epoch_duration
        })
    
    def on_batch_end(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """Log batch metrics and aggregate for epoch."""
        if not logs or 'loss' not in logs:
            return
        
        self.step_count += 1
        
        # Update aggregated metrics
        batch_metrics = {'loss': logs['loss']}
        
        # Add any other metrics from logs
        for key, value in logs.items():
            if key != 'loss' and isinstance(value, (int, float)):
                batch_metrics[key] = value
        
        self.metrics_agg.update(**batch_metrics)
        
        # Log individual steps periodically
        if self.json_logger and self.step_count % self.log_every_n_steps == 0:
            step_metrics = {
                'loss': logs['loss'],
                'learning_rate': self._get_learning_rate(),
            }
            
            # Add other metrics
            for key, value in logs.items():
                if key not in ['loss'] and isinstance(value, (int, float)):
                    step_metrics[key] = value
            
            self.json_logger.log_step(self.step_count, step_metrics, "train")
    
    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None):
        """Log evaluation metrics."""
        if not self.log_eval_metrics or not self.json_logger or not logs:
            return
        
        eval_metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                eval_metrics[f"eval_{key}"] = value
        
        if eval_metrics:
            # Use current epoch if available
            current_epoch = getattr(self, 'current_epoch', 0)
            self.json_logger.log_eval_metrics(current_epoch, self.step_count, eval_metrics)
            self.json_logger.log_event("evaluation_completed", eval_metrics)
    
    def _get_learning_rate(self) -> float:
        """Extract current learning rate from optimizer."""
        if self.optimizer and hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0].get('lr', 0.0)
        return 0.0


class ProgressLoggingCallback(Callback):
    """
    Simple callback for progress logging (complements JSON logging).
    """
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.batch_count = 0
    
    def on_batch_end(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """Log training progress."""
        self.batch_count += 1
        
        if self.batch_count % self.log_interval == 0 and logs and 'loss' in logs:
            epoch = getattr(self, 'current_epoch', 0)
            lr = self._get_learning_rate()
            
            logger.info(f"Epoch {epoch}, Step {self.batch_count}, "
                       f"Loss: {logs['loss']:.4f}, LR: {lr:.2e}")
    
    def _get_learning_rate(self) -> float:
        """Get current learning rate."""
        if self.optimizer and hasattr(self.optimizer, 'param_groups'):
            return self.optimizer.param_groups[0].get('lr', 0.0)
        return 0.0