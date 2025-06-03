# src/utils/json_logger.py
"""
Simple JSON logger for training metrics.
Makes it easy to plot convergence and compare runs.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class JSONLogger:
    """
    Simple JSON logger for training metrics.
    Logs everything to a structured JSON file for easy plotting.
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
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.log_data["config"] = config
        self._save()
        logger.info("Configuration logged")
    
    def log_step(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """
        Log metrics for a training step.
        
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
        
        self.log_data["metrics"]["steps"].append(entry)
        self._save()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics
        }
        
        self.log_data["metrics"]["epochs"].append(entry)
        self._save()
        logger.info(f"Epoch {epoch} metrics logged: {metrics}")
    
    def log_train_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log training metrics."""
        entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
        self.log_data["metrics"]["train"].append(entry)
        self._save()
    
    def log_eval_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        entry = {
            "epoch": epoch,
            "step": step,
            "timestamp": time.time(),
            **metrics
        }
        
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
        """Mark training as finished."""
        self.log_data["run_info"]["end_time"] = datetime.now().isoformat()
        self.log_data["run_info"]["status"] = "completed"
        
        if final_metrics:
            self.log_data["run_info"]["final_metrics"] = final_metrics
        
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
    Simple utility to aggregate metrics for logging.
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
        """Get averaged metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics
            if self.counts[key] > 0
        }
    
    def get_totals(self) -> Dict[str, float]:
        """Get total metrics (useful for loss sums)."""
        return self.metrics.copy()


# Convenience function for quick setup
def setup_json_logging(output_dir: str, run_name: Optional[str] = None) -> JSONLogger:
    """
    Quick setup for JSON logging.
    
    Args:
        output_dir: Output directory
        run_name: Optional run name
        
    Returns:
        Configured JSONLogger
    """
    log_file = os.path.join(output_dir, "training_log.json")
    return JSONLogger(log_file, run_name)