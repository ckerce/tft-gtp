#!/usr/bin/env python3
"""
Quick script to check parameter counts in your TFT model outputs.
Based on your directory structure from the training metrics.
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def count_params_simple(checkpoint_path):
    """Simple parameter counting from checkpoint."""
    try:
        print(f"Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(param.numel() for param in state_dict.values())
            
            # Get model info if available
            model_info = ""
            if 'args' in checkpoint:
                args = checkpoint['args']
                model_type = args.get('model', 'unknown')
                preset = args.get('preset', 'unknown')
                model_info = f" ({model_type}, {preset})"
            
            print(f"  ✅ {total_params:,} parameters ({total_params/1e6:.2f}M){model_info}")
            return total_params
        else:
            print(f"  ❌ No model_state_dict found")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """Check your outputs directory structure."""
    outputs_dir = Path("./outputs")
    
    if not outputs_dir.exists():
        print("❌ ./outputs directory not found")
        return
    
    print("🔍 Scanning outputs directory...")
    
    # Based on your structure, look for model files
    patterns = [
        "**/*_model.pt",           # Your main model saves
        "**/checkpoint_*.pt",      # Epoch checkpoints  
        "**/*.pt"                  # Any other .pt files
    ]
    
    found_files = []
    for pattern in patterns:
        found_files.extend(outputs_dir.glob(pattern))
    
    # Remove duplicates and sort
    found_files = sorted(set(found_files))
    
    if not found_files:
        print("❌ No .pt files found in outputs")
        return
    
    print(f"Found {len(found_files)} checkpoint files:\n")
    
    results = {}
    
    for checkpoint_file in found_files:
        rel_path = checkpoint_file.relative_to(outputs_dir)
        print(f"📁 {rel_path}")
        
        num_params = count_params_simple(checkpoint_file)
        if num_params:
            results[str(rel_path)] = num_params
        print()
    
    # Summary
    if results:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        # Group by directory (experiment)
        by_experiment = {}
        for path, params in results.items():
            exp_name = Path(path).parts[0] if '/' in path else 'root'
            if exp_name not in by_experiment:
                by_experiment[exp_name] = []
            by_experiment[exp_name].append((path, params))
        
        for exp_name, files in by_experiment.items():
            print(f"\n📊 {exp_name}:")
            for path, params in sorted(files, key=lambda x: x[1], reverse=True):
                filename = Path(path).name
                print(f"  {filename:<30} {params:>10,} ({params/1e6:>6.2f}M)")
        
        # Find unique parameter counts
        unique_params = sorted(set(results.values()), reverse=True)
        print(f"\n🔢 Unique parameter counts found:")
        for params in unique_params:
            count = sum(1 for p in results.values() if p == params)
            print(f"  {params:,} parameters ({params/1e6:.2f}M) - {count} model(s)")

if __name__ == "__main__":
    main()