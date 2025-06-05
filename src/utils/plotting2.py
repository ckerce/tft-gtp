# src/utils/plotting.py (Enhanced with validation loss plotting)
"""
Enhanced utilities for plotting training metrics with comprehensive validation loss visualization.
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
                        figsize: tuple = (16, 12),
                        show_validation: bool = True):
    """
    Plot enhanced training curves including validation loss and perplexity.
    
    Args:
        log_files: Single log file or list of log files to compare
        output_file: Optional file to save plot
        title: Plot title
        figsize: Figure size
        show_validation: Whether to include validation plots
    """
    if isinstance(log_files, str):
        log_files = [log_files]
    
    # Determine subplot layout based on validation data availability
    if show_validation:
        fig, axes = plt.subplots(3, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
    fig.suptitle(title, fontsize=16)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for i, log_file in enumerate(log_files):
        data = load_training_log(log_file)
        run_name = data.get('run_info', {}).get('run_name', f'Run {i+1}')
        color = colors[i]
        
        # Plot training loss per epoch
        if data['metrics']['epochs']:
            epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
            losses = [entry.get('loss', entry.get('avg_loss', np.nan)) for entry in data['metrics']['epochs']]
            
            axes[0, 0].plot(epochs, losses, label=f'{run_name} (train)', color=color, marker='o', linewidth=2)
            axes[0, 0].set_title('Training Loss per Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot training perplexity per epoch
        if data['metrics']['epochs']:
            epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
            perplexities = [entry.get('perplexity', entry.get('avg_perplexity', np.nan)) for entry in data['metrics']['epochs']]
            
            # Filter out nan/inf values
            valid_data = [(e, p) for e, p in zip(epochs, perplexities) 
                         if not (np.isnan(p) or np.isinf(p))]
            
            if valid_data:
                valid_epochs, valid_perplexities = zip(*valid_data)
                axes[0, 1].plot(valid_epochs, valid_perplexities, label=f'{run_name} (train)', 
                               color=color, marker='o', linewidth=2)
                axes[0, 1].set_title('Training Perplexity per Epoch')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Perplexity')
                axes[0, 1].set_yscale('log')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot step-wise training loss
        if data['metrics']['steps']:
            train_steps = [entry['step'] for entry in data['metrics']['steps'] if entry.get('phase') == 'train']
            train_step_losses = [entry['loss'] for entry in data['metrics']['steps'] if entry.get('phase') == 'train']
            
            if train_steps and train_step_losses:
                axes[0, 2].plot(train_steps, train_step_losses, label=f'{run_name} (train)', 
                               color=color, alpha=0.7, linewidth=1)
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
                axes[1, 0].semilogy(steps, lrs, label=run_name, color=color, linewidth=2)
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot validation loss (enhanced)
        validation_plotted = False
        
        # Try multiple sources for validation data
        eval_data_sources = [
            ('eval', data['metrics'].get('eval', [])),
            ('train', [entry for entry in data['metrics']['steps'] if entry.get('phase') == 'eval']),
            ('epochs', [entry for entry in data['metrics']['epochs'] if 'eval_loss' in entry])
        ]
        
        for source_name, eval_entries in eval_data_sources:
            if not eval_entries:
                continue
                
            if source_name == 'epochs':
                # Validation from epoch data
                eval_epochs = [entry['epoch'] for entry in eval_entries]
                eval_losses = [entry.get('eval_loss', entry.get('val_loss', np.nan)) for entry in eval_entries]
                x_data, x_label = eval_epochs, 'Epoch'
            else:
                # Validation from step data
                eval_steps = [entry.get('step', entry.get('epoch', 0)) for entry in eval_entries]
                eval_losses = [entry.get('eval_loss', entry.get('loss', entry.get('val_loss', np.nan))) for entry in eval_entries]
                x_data, x_label = eval_steps, 'Step'
            
            # Filter valid data
            valid_eval_data = [(x, loss) for x, loss in zip(x_data, eval_losses) 
                              if not (np.isnan(loss) or np.isinf(loss))]
            
            if valid_eval_data:
                valid_x, valid_losses = zip(*valid_eval_data)
                axes[1, 1].plot(valid_x, valid_losses, 
                               label=f'{run_name} (val)', 
                               color=color, linestyle='--', marker='s', linewidth=2)
                validation_plotted = True
                break
        
        if validation_plotted:
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].set_xlabel(x_label)
            axes[1, 1].set_ylabel('Validation Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Validation Data\nAvailable', 
                           transform=axes[1, 1].transAxes, ha='center', va='center',
                           fontsize=12, alpha=0.6)
            axes[1, 1].set_title('Validation Loss (No Data)')
        
        # Plot validation perplexity
        val_perplexity_plotted = False
        
        for source_name, eval_entries in eval_data_sources:
            if not eval_entries:
                continue
                
            if source_name == 'epochs':
                eval_epochs = [entry['epoch'] for entry in eval_entries]
                eval_perplexities = [entry.get('eval_perplexity', entry.get('val_perplexity', 
                                    np.exp(entry.get('eval_loss', np.nan)) if 'eval_loss' in entry else np.nan)) 
                                   for entry in eval_entries]
                x_data, x_label = eval_epochs, 'Epoch'
            else:
                eval_steps = [entry.get('step', entry.get('epoch', 0)) for entry in eval_entries]
                eval_perplexities = [entry.get('eval_perplexity', entry.get('perplexity', 
                                    np.exp(entry.get('eval_loss', entry.get('loss', np.nan))))) 
                                   for entry in eval_entries]
                x_data, x_label = eval_steps, 'Step'
            
            # Filter valid perplexity data
            valid_eval_ppl_data = [(x, ppl) for x, ppl in zip(x_data, eval_perplexities) 
                                  if not (np.isnan(ppl) or np.isinf(ppl))]
            
            if valid_eval_ppl_data:
                valid_x, valid_ppls = zip(*valid_eval_ppl_data)
                axes[1, 2].plot(valid_x, valid_ppls, 
                               label=f'{run_name} (val)', 
                               color=color, linestyle='--', marker='s', linewidth=2)
                val_perplexity_plotted = True
                break
        
        if val_perplexity_plotted:
            axes[1, 2].set_title('Validation Perplexity')
            axes[1, 2].set_xlabel(x_label)
            axes[1, 2].set_ylabel('Validation Perplexity')
            axes[1, 2].set_yscale('log')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Validation\nPerplexity Data', 
                           transform=axes[1, 2].transAxes, ha='center', va='center',
                           fontsize=12, alpha=0.6)
            axes[1, 2].set_title('Validation Perplexity (No Data)')
        
        # Add combined train/val loss comparison if validation data exists and show_validation is True
        if show_validation and validation_plotted:
            # Training loss from epochs
            if data['metrics']['epochs']:
                epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
                train_losses = [entry.get('loss', entry.get('avg_loss', np.nan)) for entry in data['metrics']['epochs']]
                axes[2, 0].plot(epochs, train_losses, label=f'{run_name} (train)', 
                               color=color, marker='o', linewidth=2)
            
            # Validation loss overlay
            axes[2, 0].plot(valid_x, valid_losses, label=f'{run_name} (val)', 
                           color=color, linestyle='--', marker='s', linewidth=2, alpha=0.8)
            axes[2, 0].set_title('Training vs Validation Loss')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # Combined perplexity comparison
            if data['metrics']['epochs']:
                epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
                train_ppls = [entry.get('perplexity', entry.get('avg_perplexity', np.nan)) for entry in data['metrics']['epochs']]
                valid_train_ppl = [(e, p) for e, p in zip(epochs, train_ppls) if not (np.isnan(p) or np.isinf(p))]
                
                if valid_train_ppl:
                    valid_epochs, valid_train_ppls = zip(*valid_train_ppl)
                    axes[2, 1].plot(valid_epochs, valid_train_ppls, label=f'{run_name} (train)', 
                                   color=color, marker='o', linewidth=2)
            
            if val_perplexity_plotted:
                axes[2, 1].plot(valid_x, valid_ppls, label=f'{run_name} (val)', 
                               color=color, linestyle='--', marker='s', linewidth=2, alpha=0.8)
            
            axes[2, 1].set_title('Training vs Validation Perplexity')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Perplexity')
            axes[2, 1].set_yscale('log')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
    
        # If not showing validation, use the space for other metrics
        elif not show_validation:
            # Additional metrics can go here
            pass
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    return fig


def plot_validation_comparison(log_files: List[str], 
                              output_file: Optional[str] = None,
                              title: Optional[str] = None,
                              figsize: tuple = (15, 5)):
    """
    Compare validation performance across multiple training runs.
    
    Args:
        log_files: List of log files to compare
        output_file: Optional file to save plot
        title: Optional plot title
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for i, log_file in enumerate(log_files):
        data = load_training_log(log_file)
        run_name = data.get('run_info', {}).get('run_name', Path(log_file).stem)
        color = plt.cm.tab10(i)
        
        # Extract validation data from multiple sources
        validation_data = []
        
        # Check eval metrics
        if data['metrics'].get('eval'):
            for entry in data['metrics']['eval']:
                step = entry.get('step', entry.get('epoch', 0))
                val_loss = entry.get('eval_loss', entry.get('loss', entry.get('val_loss')))
                if val_loss is not None and not np.isnan(val_loss):
                    validation_data.append((step, val_loss))
        
        # Check epoch metrics for eval data
        if data['metrics'].get('epochs'):
            for entry in data['metrics']['epochs']:
                if 'eval_loss' in entry or 'val_loss' in entry:
                    epoch = entry['epoch']
                    val_loss = entry.get('eval_loss', entry.get('val_loss'))
                    if val_loss is not None and not np.isnan(val_loss):
                        validation_data.append((epoch, val_loss))
        
        if validation_data:
            steps, val_losses = zip(*validation_data)
            
            # Plot validation loss
            axes[0].plot(steps, val_losses, label=run_name, marker='o', linewidth=2, color=color)
            
            # Plot validation perplexity
            val_ppls = [np.exp(loss) for loss in val_losses]
            axes[1].plot(steps, val_ppls, label=run_name, marker='o', linewidth=2, color=color)
            
            # Plot validation improvement over time
            if len(val_losses) > 1:
                improvement = [(val_losses[0] - loss) / val_losses[0] * 100 for loss in val_losses]
                axes[2].plot(steps, improvement, label=run_name, marker='o', linewidth=2, color=color)
    
    # Configure plots
    axes[0].set_title('Validation Loss Comparison')
    axes[0].set_xlabel('Epoch/Step')
    axes[0].set_ylabel('Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Validation Perplexity Comparison')
    axes[1].set_xlabel('Epoch/Step')
    axes[1].set_ylabel('Validation Perplexity')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Validation Improvement (%)')
    axes[2].set_xlabel('Epoch/Step')
    axes[2].set_ylabel('Improvement from Initial (%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.suptitle(title or 'Validation Performance Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Validation comparison plot saved to {output_file}")
    
    plt.show()


def plot_train_val_curves(log_file: str, 
                         output_file: Optional[str] = None,
                         title: Optional[str] = None,
                         figsize: tuple = (12, 8)):
    """
    Plot detailed training vs validation curves for a single run.
    
    Args:
        log_file: Path to log file
        output_file: Optional file to save plot
        title: Optional plot title
        figsize: Figure size
    """
    data = load_training_log(log_file)
    run_name = data.get('run_info', {}).get('run_name', Path(log_file).stem)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Training data
    if data['metrics']['epochs']:
        epochs = [entry['epoch'] for entry in data['metrics']['epochs']]
        train_losses = [entry.get('loss', entry.get('avg_loss', np.nan)) for entry in data['metrics']['epochs']]
        train_ppls = [entry.get('perplexity', entry.get('avg_perplexity', np.nan)) for entry in data['metrics']['epochs']]
    else:
        epochs, train_losses, train_ppls = [], [], []
    
    # Validation data (try multiple sources)
    val_epochs, val_losses, val_ppls = [], [], []
    
    # From eval metrics
    if data['metrics'].get('eval'):
        for entry in data['metrics']['eval']:
            epoch = entry.get('epoch', entry.get('step', 0))
            val_loss = entry.get('eval_loss', entry.get('loss', entry.get('val_loss')))
            if val_loss is not None and not np.isnan(val_loss):
                val_epochs.append(epoch)
                val_losses.append(val_loss)
                val_ppls.append(np.exp(val_loss))
    
    # From epoch data
    if not val_epochs and data['metrics']['epochs']:
        for entry in data['metrics']['epochs']:
            if 'eval_loss' in entry or 'val_loss' in entry:
                val_epochs.append(entry['epoch'])
                val_loss = entry.get('eval_loss', entry.get('val_loss'))
                val_losses.append(val_loss)
                val_ppls.append(np.exp(val_loss))
    
    # Plot 1: Loss comparison
    if epochs and train_losses:
        axes[0, 0].plot(epochs, train_losses, label='Training', marker='o', linewidth=2, color='blue')
    if val_epochs and val_losses:
        axes[0, 0].plot(val_epochs, val_losses, label='Validation', marker='s', linewidth=2, color='red', linestyle='--')
    
    axes[0, 0].set_title('Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Perplexity comparison
    if epochs and train_ppls:
        valid_train_ppl = [(e, p) for e, p in zip(epochs, train_ppls) if not (np.isnan(p) or np.isinf(p))]
        if valid_train_ppl:
            valid_epochs, valid_ppls = zip(*valid_train_ppl)
            axes[0, 1].plot(valid_epochs, valid_ppls, label='Training', marker='o', linewidth=2, color='blue')
    
    if val_epochs and val_ppls:
        valid_val_ppl = [(e, p) for e, p in zip(val_epochs, val_ppls) if not (np.isnan(p) or np.isinf(p))]
        if valid_val_ppl:
            valid_val_epochs, valid_val_ppls = zip(*valid_val_ppl)
            axes[0, 1].plot(valid_val_epochs, valid_val_ppls, label='Validation', marker='s', linewidth=2, color='red', linestyle='--')
    
    axes[0, 1].set_title('Perplexity Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss difference (overfitting indicator)
    if val_epochs and val_losses and epochs and train_losses:
        # Find common epochs
        common_epochs = sorted(set(epochs) & set(val_epochs))
        if len(common_epochs) > 1:
            train_common = [train_losses[epochs.index(e)] for e in common_epochs if e in epochs]
            val_common = [val_losses[val_epochs.index(e)] for e in common_epochs if e in val_epochs]
            
            if len(train_common) == len(val_common) == len(common_epochs):
                loss_diff = [v - t for v, t in zip(val_common, train_common)]
                axes[1, 0].plot(common_epochs, loss_diff, marker='d', linewidth=2, color='purple')
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 0].set_title('Overfitting Indicator (Val - Train Loss)')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss Difference')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add interpretation text
                latest_diff = loss_diff[-1] if loss_diff else 0
                if latest_diff > 0.5:
                    status = "High overfitting risk"
                    color = 'red'
                elif latest_diff > 0.2:
                    status = "Moderate overfitting"
                    color = 'orange'
                elif latest_diff > 0:
                    status = "Good generalization"
                    color = 'green'
                else:
                    status = "Underfitting possible"
                    color = 'blue'
                
                axes[1, 0].text(0.02, 0.98, f"Status: {status}", 
                               transform=axes[1, 0].transAxes, 
                               verticalalignment='top', color=color, fontweight='bold')
    
    # Plot 4: Learning curves summary
    axes[1, 1].text(0.1, 0.9, f"Training Summary: {run_name}", transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
    
    summary_text = []
    if epochs and train_losses:
        summary_text.append(f"Training epochs: {len(epochs)}")
        summary_text.append(f"Initial train loss: {train_losses[0]:.4f}")
        summary_text.append(f"Final train loss: {train_losses[-1]:.4f}")
        summary_text.append(f"Train improvement: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    if val_epochs and val_losses:
        summary_text.append(f"Validation points: {len(val_epochs)}")
        summary_text.append(f"Best val loss: {min(val_losses):.4f}")
        summary_text.append(f"Final val loss: {val_losses[-1]:.4f}")
    
    summary_text_str = '\n'.join(summary_text)
    axes[1, 1].text(0.1, 0.8, summary_text_str, transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='top')
    axes[1, 1].axis('off')
    
    plt.suptitle(title or f'Training vs Validation Analysis: {run_name}', fontsize=14)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Train-val analysis plot saved to {output_file}")
    
    plt.show()


# Enhanced quick plotting functions
def quick_plot_with_validation(log_file: str):
    """Quick plot including validation data if available."""
    plot_training_curves(log_file, show_validation=True)
    plt.show()


def quick_validation_analysis(log_file: str):
    """Quick validation-focused analysis."""
    plot_train_val_curves(log_file)


def plot_multiple_with_validation(log_files: List[str]):
    """Quick comparison of multiple runs with validation emphasis."""
    plot_training_curves(log_files, show_validation=True)
    plt.show()
    
    # Also show validation-specific comparison
    plot_validation_comparison(log_files)