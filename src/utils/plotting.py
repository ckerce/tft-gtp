# src/utils/plotting.py (Enhanced with perplexity plotting)
"""
Enhanced utilities for plotting training metrics with perplexity visualization.
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
                        figsize: tuple = (15, 10)):
    """
    Plot enhanced training curves including perplexity from one or more log files.
    
    Args:
        log_files: Single log file or list of log files to compare
        output_file: Optional file to save plot
        title: Plot title
        figsize: Figure size
    """
    if isinstance(log_files, str):
        log_files = [log_files]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
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
        
        # Plot epoch perplexity
        if data['metrics']['epochs']:
            epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
            perplexities = [entry.get('perplexity', entry.get('avg_perplexity', np.nan)) for entry in data['metrics']['epochs']]
            
            # Filter out nan/inf values for cleaner plots
            valid_data = [(e, p) for e, p in zip(epochs, perplexities) 
                         if not (np.isnan(p) or np.isinf(p))]
            
            if valid_data:
                valid_epochs, valid_perplexities = zip(*valid_data)
                axes[0, 1].plot(valid_epochs, valid_perplexities, label=run_name, color=color, marker='o')
                axes[0, 1].set_title('Training Perplexity per Epoch')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Perplexity')
                axes[0, 1].set_yscale('log')  # Log scale for perplexity
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot step loss (if available)
        if data['metrics']['steps']:
            steps = [entry['step'] for entry in data['metrics']['steps'] if entry['phase'] == 'train']
            step_losses = [entry['loss'] for entry in data['metrics']['steps'] if entry['phase'] == 'train']
            
            if steps and step_losses:
                axes[0, 2].plot(steps, step_losses, label=run_name, color=color, alpha=0.7)
                axes[0, 2].set_title('Training Loss per Step')
                axes[0, 2].set_xlabel('Step')
                axes[0, 2].set_ylabel('Loss')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
        
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
            eval_losses = [entry.get('eval_loss', entry.get('loss', np.nan)) for entry in data['metrics']['eval']]
            
            if eval_steps and not all(np.isnan(eval_losses)):
                axes[1, 1].plot(eval_steps, eval_losses, label=f'{run_name} (eval loss)', 
                               color=color, linestyle='--', marker='s')
                axes[1, 1].set_title('Evaluation Loss')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Eval Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # Plot evaluation perplexity (if available)
        if data['metrics']['eval']:
            eval_steps = [entry['step'] for entry in data['metrics']['eval']]
            eval_perplexities = [entry.get('eval_perplexity', entry.get('perplexity', np.nan)) for entry in data['metrics']['eval']]
            
            # Filter valid perplexity values
            valid_eval_data = [(s, p) for s, p in zip(eval_steps, eval_perplexities) 
                              if not (np.isnan(p) or np.isinf(p))]
            
            if valid_eval_data:
                valid_eval_steps, valid_eval_perplexities = zip(*valid_eval_data)
                axes[1, 2].plot(valid_eval_steps, valid_eval_perplexities, 
                               label=f'{run_name} (eval ppl)', 
                               color=color, linestyle='--', marker='s')
                axes[1, 2].set_title('Evaluation Perplexity')
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Eval Perplexity')
                axes[1, 2].set_yscale('log')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig


def plot_perplexity_comparison(log_files: List[str], 
                              output_file: Optional[str] = None,
                              title: Optional[str] = None,
                              figsize: tuple = (12, 6)):
    """
    Compare perplexity convergence of multiple training runs.
    
    Args:
        log_files: List of log files to compare
        output_file: Optional file to save plot
        title: Optional plot title
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for i, log_file in enumerate(log_files):
        data = load_training_log(log_file)
        run_name = data.get('run_info', {}).get('run_name', Path(log_file).stem)
        
        # Training perplexity
        if data['metrics']['epochs']:
            epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
            perplexities = [entry.get('perplexity', entry.get('avg_perplexity', np.nan)) 
                           for entry in data['metrics']['epochs']]
            
            valid_data = [(e, p) for e, p in zip(epochs, perplexities) 
                         if not (np.isnan(p) or np.isinf(p))]
            
            if valid_data:
                valid_epochs, valid_perplexities = zip(*valid_data)
                ax1.plot(valid_epochs, valid_perplexities, label=run_name, marker='o', alpha=0.8)
        
        # Evaluation perplexity
        if data['metrics']['eval']:
            eval_steps = [entry['step'] for entry in data['metrics']['eval']]
            eval_perplexities = [entry.get('eval_perplexity', entry.get('perplexity', np.nan)) 
                                for entry in data['metrics']['eval']]
            
            valid_eval_data = [(s, p) for s, p in zip(eval_steps, eval_perplexities) 
                              if not (np.isnan(p) or np.isinf(p))]
            
            if valid_eval_data:
                valid_eval_steps, valid_eval_perplexities = zip(*valid_eval_data)
                ax2.plot(valid_eval_steps, valid_eval_perplexities, label=run_name, marker='s', alpha=0.8)
    
    # Configure training perplexity plot
    ax1.set_title('Training Perplexity Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Perplexity')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configure evaluation perplexity plot
    ax2.set_title('Evaluation Perplexity Comparison')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Eval Perplexity')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title or 'Perplexity Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Perplexity comparison plot saved to {output_file}")
    
    plt.show()


def plot_convergence_comparison(log_files: List[str], 
                               metric: str = 'loss',
                               output_file: Optional[str] = None,
                               title: Optional[str] = None):
    """
    Compare convergence of multiple training runs for any metric.
    
    Args:
        log_files: List of log files to compare
        metric: Metric to compare ('loss', 'perplexity', 'eval_loss', etc.)
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
        
        # Filter out invalid values for perplexity
        if 'perplexity' in metric:
            valid_data = [(s, v) for s, v in zip(steps, values) 
                         if not (np.isnan(v) or np.isinf(v))]
            if valid_data:
                steps, values = zip(*valid_data)
        
        if steps and values:
            plt.plot(steps, values, label=run_name, marker='o', alpha=0.8)
    
    plt.title(title or f'{metric.replace("_", " ").title()} Comparison')
    plt.xlabel('Epoch' if not metric.startswith('eval_') else 'Step')
    plt.ylabel(metric.replace('_', ' ').title())
    
    # Use log scale for perplexity
    if 'perplexity' in metric:
        plt.yscale('log')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_file}")
    
    plt.show()


def generate_training_report(log_file: str, output_dir: str = "reports"):
    """
    Generate a comprehensive training report with plots including perplexity.
    
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
        
        # Training metrics with perplexity
        if data['metrics']['epochs']:
            epochs = data['metrics']['epochs']
            initial_loss = epochs[0].get('loss', epochs[0].get('avg_loss', 0))
            final_loss = epochs[-1].get('loss', epochs[-1].get('avg_loss', 0))
            
            initial_ppl = epochs[0].get('perplexity', epochs[0].get('avg_perplexity', np.nan))
            final_ppl = epochs[-1].get('perplexity', epochs[-1].get('avg_perplexity', np.nan))
            
            f.write("Training Metrics:\n")
            f.write(f"  Total epochs: {len(epochs)}\n")
            f.write(f"  Initial loss: {initial_loss:.6f}\n")
            f.write(f"  Final loss: {final_loss:.6f}\n")
            f.write(f"  Loss reduction: {initial_loss - final_loss:.6f}\n")
            f.write(f"  Relative improvement: {(initial_loss - final_loss) / initial_loss * 100:.2f}%\n")
            
            if not np.isnan(initial_ppl) and not np.isnan(final_ppl):
                f.write(f"  Initial perplexity: {initial_ppl:.2f}\n")
                f.write(f"  Final perplexity: {final_ppl:.2f}\n")
                f.write(f"  Perplexity reduction: {initial_ppl - final_ppl:.2f}\n")
                f.write(f"  Perplexity reduction %: {(initial_ppl - final_ppl) / initial_ppl * 100:.2f}%\n")
            f.write("\n")
        
        # Final metrics from run_info
        final_metrics = run_info.get('final_metrics', {})
        if final_metrics:
            f.write("Final Metrics:\n")
            for key, value in final_metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
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


# Quick plotting functions
def quick_plot(log_file: str):
    """Quick plot of a single training run with perplexity."""
    plot_training_curves(log_file)
    plt.show()


def quick_perplexity_plot(log_files: Union[str, List[str]]):
    """Quick perplexity comparison plot."""
    if isinstance(log_files, str):
        log_files = [log_files]
    plot_perplexity_comparison(log_files)


def plot_multiple(log_files: List[str]):
    """Quick comparison of multiple runs with perplexity."""
    plot_training_curves(log_files)
    plt.show()