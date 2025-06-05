#!/usr/bin/env python3
"""
Post-hoc validation evaluation script.
Evaluates saved model checkpoints on validation data and adds results to training logs.
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, 'src')

from models import get_model
from config.model_configs import TFTConfig
from mytokenizers import create_tokenizer
from utils.data_utils import load_and_prepare_data


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint and extract config."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config - try different possible locations
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif 'model_config' in checkpoint:
        config = checkpoint['model_config']
    else:
        # Try to infer from args if available
        args = checkpoint.get('args', {})
        preset = args.get('preset', 'small')
        from config.model_configs import get_config
        config = get_config(preset)
        print(f"⚠️  Config not found in checkpoint, using preset: {preset}")
    
    return checkpoint, config


def evaluate_model_on_validation(model, val_dataloader, device, max_batches=None):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_batches = 0
    
    print(f"Evaluating on validation data...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
                
            # Move data to device
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Forward pass
            outputs = model(**batch_data)
            loss = outputs.get('loss')
            
            if loss is None or torch.isnan(loss):
                print(f"⚠️  Warning: NaN loss at batch {batch_idx}")
                continue
            
            # Accumulate metrics
            batch_size = batch_data.get('input_ids', next(iter(batch_data.values()))).size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            num_batches += 1
    
    if total_samples == 0:
        return {'eval_loss': float('nan'), 'eval_perplexity': float('nan')}
    
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    
    metrics = {
        'eval_loss': avg_loss,
        'eval_perplexity': perplexity,
        'eval_samples': total_samples,
        'eval_batches': num_batches
    }
    
    print(f"Validation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Samples: {total_samples}")
    
    return metrics


def update_training_log_with_validation(log_file: str, validation_metrics: dict, epoch: int):
    """Add validation metrics to existing training log."""
    if not os.path.exists(log_file):
        print(f"⚠️  Training log not found: {log_file}")
        return
    
    # Load existing log
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Add validation metrics to the appropriate epoch
    if 'metrics' not in log_data:
        log_data['metrics'] = {'eval': [], 'epochs': []}
    
    if 'eval' not in log_data['metrics']:
        log_data['metrics']['eval'] = []
    
    # Find the epoch entry and add validation metrics
    for epoch_entry in log_data['metrics']['epochs']:
        if epoch_entry['epoch'] == epoch:
            epoch_entry.update(validation_metrics)
            break
    
    # Also add to eval metrics list
    eval_entry = {
        'epoch': epoch,
        'step': epoch * 1000,  # Approximate step count
        'timestamp': epoch_entry.get('timestamp', 0),
        **validation_metrics
    }
    log_data['metrics']['eval'].append(eval_entry)
    
    # Save updated log
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"✅ Updated training log: {log_file}")


def evaluate_experiment_checkpoints(experiment_dir: str, 
                                   dataset: str = "wikimedia/wikipedia",
                                   dataset_config: str = "20231101.en",
                                   max_val_samples: int = 10000,
                                   max_eval_batches: int = None,
                                   batch_size: int = 32):
    """Evaluate all checkpoints in an experiment directory."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    # Find checkpoints
    checkpoint_files = list(experiment_path.glob("checkpoint_epoch_*.pt"))
    final_model = experiment_path / "tft_model.pt"
    
    if final_model.exists():
        checkpoint_files.append(final_model)
    
    if not checkpoint_files:
        print(f"❌ No checkpoints found in {experiment_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints in {experiment_dir}")
    
    # Load first checkpoint to get config
    checkpoint, config = load_checkpoint(str(checkpoint_files[0]))
    
    # Create tokenizer and validation data
    print("Creating tokenizer and loading validation data...")
    tokenizer = create_tokenizer('gpt2')
    config.vocab_size = tokenizer.vocab_size
    
    try:
        val_dataloader, _ = load_and_prepare_data(
            dataset_name=dataset,
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            max_samples=max_val_samples,
            max_seq_length=config.block_size,
            batch_size=batch_size,
            split='validation',  # Try validation first
            shuffle=False
        )
        print(f"✅ Loaded validation data: {len(val_dataloader)} batches")
    except:
        try:
            # Fallback to test split
            val_dataloader, _ = load_and_prepare_data(
                dataset_name=dataset,
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                max_samples=max_val_samples,
                max_seq_length=config.block_size,
                batch_size=batch_size,
                split='test',
                shuffle=False
            )
            print(f"✅ Loaded test data as validation: {len(val_dataloader)} batches")
        except:
            # Fallback to train split (use different samples)
            val_dataloader, _ = load_and_prepare_data(
                dataset_name=dataset,
                dataset_config=dataset_config,
                tokenizer=tokenizer,
                max_samples=max_val_samples,
                max_seq_length=config.block_size,
                batch_size=batch_size,
                split=f'train[90%:95%]',  # Use a slice of train data
                shuffle=False
            )
            print(f"✅ Loaded train slice as validation: {len(val_dataloader)} batches")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Evaluate each checkpoint
    results = {}
    
    for checkpoint_file in sorted(checkpoint_files):
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_file.name}")
        
        # Extract epoch from filename
        if "epoch_" in checkpoint_file.name:
            epoch = int(checkpoint_file.name.split("epoch_")[1].split(".")[0])
        else:
            epoch = 999  # Final model
        
        # Load checkpoint
        checkpoint, config = load_checkpoint(str(checkpoint_file))
        
        # Create model
        model_type = checkpoint.get('args', {}).get('model', 'tft')
        model = get_model(model_type, config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Evaluate
        val_metrics = evaluate_model_on_validation(
            model, val_dataloader, device, max_eval_batches
        )
        
        results[epoch] = val_metrics
        
        # Update training log
        log_file = experiment_path / "training_metrics.json"
        if epoch != 999:  # Don't update for final model
            update_training_log_with_validation(str(log_file), val_metrics, epoch)
    
    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    for epoch, metrics in sorted(results.items()):
        epoch_str = f"Epoch {epoch}" if epoch != 999 else "Final"
        print(f"{epoch_str:>10}: Loss {metrics['eval_loss']:.4f}, PPL {metrics['eval_perplexity']:.2f}")
    
    # Find best epoch
    best_epoch = min(results.keys(), key=lambda e: results[e]['eval_loss'])
    best_metrics = results[best_epoch]
    print(f"\nBest validation: Epoch {best_epoch}, Loss {best_metrics['eval_loss']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Post-hoc validation evaluation')
    parser.add_argument('experiment_dir', help='Experiment directory containing checkpoints')
    parser.add_argument('--dataset', default='wikimedia/wikipedia', help='Dataset name')
    parser.add_argument('--dataset_config', default='20231101.en', help='Dataset config')
    parser.add_argument('--max_val_samples', type=int, default=10000, help='Max validation samples')
    parser.add_argument('--max_eval_batches', type=int, default=None, help='Max evaluation batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Evaluation batch size')
    
    args = parser.parse_args()
    
    evaluate_experiment_checkpoints(
        args.experiment_dir,
        args.dataset,
        args.dataset_config,
        args.max_val_samples,
        args.max_eval_batches,
        args.batch_size
    )


# Convenience functions for your specific experiments
def evaluate_wiki_experiments():
    """Evaluate all your Wikipedia experiments post-hoc."""
    experiment_dirs = [
        "outputs/wiki_compare/vanilla",
        "outputs/wiki_compare/tft_basic", 
        "outputs/wiki_compare/tft_factored"
    ]
    
    for exp_dir in experiment_dirs:
        if Path(exp_dir).exists():
            print(f"\n🔍 Evaluating {exp_dir}")
            try:
                evaluate_experiment_checkpoints(
                    exp_dir,
                    dataset="wikimedia/wikipedia",
                    dataset_config="20231101.en",
                    max_val_samples=5000,  # Smaller for speed
                    max_eval_batches=100   # Limit for speed
                )
            except Exception as e:
                print(f"❌ Failed to evaluate {exp_dir}: {e}")
        else:
            print(f"⚠️  Directory not found: {exp_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments, evaluate wiki experiments
        evaluate_wiki_experiments()
    else:
        main()