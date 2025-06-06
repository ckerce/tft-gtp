import torch
import os
from typing import Dict, Any

def analyze_pt_file_parameters(pt_file_path: str, detailed: bool = True) -> Dict[str, Any]:
    """
    Analyze parameter count from a saved .pt file without loading the full model.
    
    Args:
        pt_file_path: Path to the .pt file
        detailed: If True, show detailed breakdown
        
    Returns:
        Dictionary with parameter analysis
    """
    print(f"🔍 ANALYZING: {pt_file_path}")
    print("=" * 60)
    
    if not os.path.exists(pt_file_path):
        print(f"❌ File not found: {pt_file_path}")
        return {}
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(pt_file_path, map_location='cpu')
        print(f"✅ File loaded successfully")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return {}
    
    # Check what's in the checkpoint
    print(f"\n📋 CHECKPOINT CONTENTS:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  {key}: {len(checkpoint[key])} items")
        else:
            print(f"  {key}: {type(checkpoint[key])}")
    
    # Look for model state dict
    state_dict = None
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\n✅ Found model_state_dict")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"\n✅ Found state_dict")
    else:
        # Maybe the checkpoint IS the state dict
        if all(isinstance(k, str) and '.' in k for k in checkpoint.keys()):
            state_dict = checkpoint
            print(f"\n✅ Checkpoint appears to be a state_dict")
        else:
            print(f"\n❌ No model state dict found")
            return {}
    
    # Analyze parameters
    total_params = 0
    trainable_params = 0
    component_breakdown = {}
    
    print(f"\n🔢 PARAMETER ANALYSIS:")
    print("-" * 40)
    
    for param_name, param_tensor in state_dict.items():
        param_count = param_tensor.numel()
        total_params += param_count
        trainable_params += param_count  # Assume all are trainable in state dict
        
        # Group by component (first part before first dot)
        component = param_name.split('.')[0] if '.' in param_name else param_name
        
        if component not in component_breakdown:
            component_breakdown[component] = {
                'params': 0,
                'tensors': [],
                'memory_mb': 0
            }
        
        component_breakdown[component]['params'] += param_count
        component_breakdown[component]['memory_mb'] += param_count * 4 / (1024**2)  # float32
        component_breakdown[component]['tensors'].append({
            'name': param_name,
            'shape': list(param_tensor.shape),
            'params': param_count
        })
    
    # Print results
    print(f"📊 TOTAL PARAMETERS: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"💾 MEMORY: {total_params * 4 / (1024**2):.1f} MB")
    
    print(f"\n📋 COMPONENT BREAKDOWN:")
    print("-" * 60)
    
    # Sort components by parameter count
    sorted_components = sorted(component_breakdown.items(), 
                             key=lambda x: x[1]['params'], reverse=True)
    
    for component, info in sorted_components:
        percentage = (info['params'] / total_params) * 100
        print(f"{component:25} {info['params']:>10,} ({percentage:5.1f}%) {info['memory_mb']:6.1f}MB")
        
        if detailed and len(info['tensors']) > 1:
            # Show individual tensors in this component
            for tensor_info in sorted(info['tensors'], key=lambda x: x['params'], reverse=True):
                if tensor_info['params'] > 1000:  # Only show significant tensors
                    print(f"  ├─ {tensor_info['name']:35} {str(tensor_info['shape']):>20} {tensor_info['params']:>8,}")
    
    # Check for config information
    config_info = {}
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\n⚙️ MODEL CONFIG:")
        if hasattr(config, '__dict__'):
            config_dict = vars(config)
        elif hasattr(config, '_asdict'):
            config_dict = config._asdict()
        else:
            config_dict = config if isinstance(config, dict) else {}
            
        for key, value in config_dict.items():
            if key in ['n_layers', 'd_model', 'n_heads', 'vocab_size', 'd_ff', 'use_dict_ffn']:
                config_info[key] = value
                print(f"  {key}: {value}")
    
    return {
        'total_params': total_params,
        'component_breakdown': component_breakdown,
        'config_info': config_info,
        'file_size_mb': os.path.getsize(pt_file_path) / (1024**2)
    }