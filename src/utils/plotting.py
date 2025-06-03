# src/utils/plotting.py
"""
Utilities for plotting training metrics from JSON logs.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd


def load_training_log(log_file: str) -> Dict:
    """Load training log from JSON file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def plot_training_curves(log_files: Union[str, List[str]], 
                        output_file: Optional[str] = None,
                        title: str = "Training Curves",
                        figsize: tuple = (12, 8)):
    """
    Plot training curves from one or more log files.
    
    Args:
        log_files: Single log file or list of log files to compare
        output_file: Optional file to save plot
        title: Plot title
        figsize: Figure size
    """
    if isinstance(log_files, str):
        log_files = [log_files]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for i, log_file in enumerate(log_files):
        data = load_training_log(log_file)
        run_name = data.get('run_info', {}).get('run_name', f'Run {i+1}')
        color = colors[i]
        
        # Plot epoch loss
        if data['metrics']['epochs']:
            epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
            losses = [entry.get('loss', entry.get('avg_loss', np.nan)) for entry in data['metrics']['epochs']]
            
            axes[0, 0].plot(epochs, losses, label=run_name, color=color, marker='o')
            axes[0, 0].set_title('Training Loss per Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot step loss (if available)
        if data['metrics']['steps']:
            steps = [entry['step'] for entry in data['metrics']['steps'] if entry['phase'] == 'train']
            step_losses = [entry['loss'] for entry in data['metrics']['steps'] if entry['phase'] == 'train']
            
            if steps and step_losses:
                axes[0, 1].plot(steps, step_losses, label=run_name, color=color, alpha=0.7)
                axes[0, 1].set_title('Training Loss per Step')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        if data['metrics']['steps']:
            steps = [entry['step'] for entry in data['metrics']['steps'] if 'learning_rate' in entry]
            lrs = [entry['learning_rate'] for entry in data['metrics']['steps'] if 'learning_rate' in entry]
            
            if steps and lrs:
                axes[1, 0].semilogy(steps, lrs, label=run_name, color=color)
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot evaluation metrics (if available)
        if data['metrics']['eval']:
            eval_steps = [entry['step'] for entry in data['metrics']['eval']]
            eval_losses = [entry.get('eval_loss', np.nan) for entry in data['metrics']['eval']]
            
            if eval_steps and not all(np.isnan(eval_losses)):
                axes[1, 1].plot(eval_steps, eval_losses, label=f'{run_name} (eval)', 
                               color=color, linestyle='--', marker='s')
                axes[1, 1].set_title('Evaluation Loss')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Eval Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig


def plot_convergence_comparison(log_files: List[str], 
                               metric: str = 'loss',
                               output_file: Optional[str] = None,
                               title: Optional[str] = None):
    """
    Compare convergence of multiple training runs.
    
    Args:
        log_files: List of log files to compare
        metric: Metric to compare ('loss', 'eval_loss', etc.)
        output_file: Optional file to save plot
        title: Optional plot title
    """
    plt.figure(figsize=(10, 6))
    
    for i, log_file in enumerate(log_files):
        data = load_training_log(log_file)
        run_name = data.get('run_info', {}).get('run_name', Path(log_file).stem)
        
        # Extract data based on metric
        if metric.startswith('eval_'):
            # Evaluation metric
            entries = data['metrics']['eval']
            steps = [entry['step'] for entry in entries if metric in entry]
            values = [entry[metric] for entry in entries if metric in entry]
        else:
            # Training metric from epochs
            entries = data['metrics']['epochs']
            steps = [entry['epoch'] for entry in entries if metric in entry]
            values = [entry[metric] for entry in entries if metric in entry]
        
        if steps and values:
            plt.plot(steps, values, label=run_name, marker='o', alpha=0.8)
    
    plt.title(title or f'{metric.replace("_", " ").title()} Comparison')
    plt.xlabel('Epoch' if not metric.startswith('eval_') else 'Step')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    
    plt.show()


def generate_training_report(log_file: str, output_dir: str = "reports"):
    """
    Generate a comprehensive training report with plots.
    
    Args:
        log_file: Path to training log
        output_dir: Directory to save report
    """
    data = load_training_log(log_file)
    run_name = data.get('run_info', {}).get('run_name', 'training_run')
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate main training curves plot
    plot_file = Path(output_dir) / f"{run_name}_training_curves.png"
    plot_training_curves(log_file, str(plot_file), f"Training Report: {run_name}")
    
    # Generate summary statistics
    summary_file = Path(output_dir) / f"{run_name}_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"Training Report: {run_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic info
        run_info = data.get('run_info', {})
        f.write(f"Start time: {run_info.get('start_time', 'Unknown')}\n")
        f.write(f"End time: {run_info.get('end_time', 'Unknown')}\n")
        f.write(f"Status: {run_info.get('status', 'Unknown')}\n\n")
        
        # Training metrics
        if data['metrics']['epochs']:
            epochs = data['metrics']['epochs']
            initial_loss = epochs[0].get('loss', epochs[0].get('avg_loss', 0))
            final_loss = epochs[-1].get('loss', epochs[-1].get('avg_loss', 0))
            
            f.write("Training Metrics:\n")
            f.write(f"  Total epochs: {len(epochs)}\n")
            f.write(f"  Initial loss: {initial_loss:.6f}\n")
            f.write(f"  Final loss: {final_loss:.6f}\n")
            f.write(f"  Loss reduction: {initial_loss - final_loss:.6f}\n")
            f.write(f"  Relative improvement: {(initial_loss - final_loss) / initial_loss * 100:.2f}%\n\n")
        
        # Configuration
        config = data.get('config', {})
        if config:
            f.write("Configuration:\n")
            for section, params in config.items():
                f.write(f"  {section}:\n")
                if isinstance(params, dict):
                    for key, value in params.items():
                        f.write(f"    {key}: {value}\n")
                else:
                    f.write(f"    {params}\n")
                f.write("\n")
    
    print(f"Training report generated in {output_dir}")
    print(f"  - Plots: {plot_file}")
    print(f"  - Summary: {summary_file}")


def compare_runs(log_files: List[str], output_dir: str = "comparisons"):
    """
    Compare multiple training runs and generate comparison plots.
    
    Args:
        log_files: List of log files to compare
        output_dir: Directory to save comparison plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Overall comparison
    comparison_file = Path(output_dir) / "run_comparison.png"
    plot_training_curves(log_files, str(comparison_file), "Training Run Comparison")
    
    # Loss convergence comparison
    loss_file = Path(output_dir) / "loss_comparison.png"
    plot_convergence_comparison(log_files, "loss", str(loss_file), "Loss Convergence Comparison")
    
    print(f"Comparison plots saved to {output_dir}")


# Example usage functions
def quick_plot(log_file: str):
    """Quick plot of a single training run."""
    plot_training_curves(log_file)
    plt.show()


def plot_multiple(log_files: List[str]):
    """Quick comparison of multiple runs."""
    plot_training_curves(log_files)
    plt.show()