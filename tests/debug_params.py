import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def analyze_model_parameters(model: nn.Module, detailed: bool = True) -> Dict:
    """
    Comprehensive parameter analysis for TFT models.
    
    Args:
        model: The model to analyze
        detailed: If True, show parameter breakdown by component
        
    Returns:
        Dictionary with parameter analysis
    """
    total_params = 0
    trainable_params = 0
    component_breakdown = {}
    
    print("🔍 PARAMETER ANALYSIS")
    print("=" * 60)
    
    # Group parameters by component
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        # Extract component name (first part of parameter name)
        component = name.split('.')[0] if '.' in name else name
        
        if component not in component_breakdown:
            component_breakdown[component] = {
                'params': 0,
                'details': []
            }
        
        component_breakdown[component]['params'] += param_count
        component_breakdown[component]['details'].append({
            'name': name,
            'shape': list(param.shape),
            'params': param_count,
            'memory_mb': param_count * 4 / (1024**2)  # Assuming float32
        })
    
    # Print summary
    print(f"📊 TOTAL PARAMETERS: {total_params:,}")
    print(f"🎯 TRAINABLE: {trainable_params:,}")
    print(f"❄️ FROZEN: {total_params - trainable_params:,}")
    print(f"💾 MEMORY: {total_params * 4 / (1024**2):.1f} MB")
    
    print("\n📋 COMPONENT BREAKDOWN:")
    print("-" * 60)
    
    # Sort components by parameter count
    sorted_components = sorted(component_breakdown.items(), 
                             key=lambda x: x[1]['params'], reverse=True)
    
    for component, info in sorted_components:
        percentage = (info['params'] / total_params) * 100
        print(f"{component:20} {info['params']:>10,} ({percentage:5.1f}%)")
        
        if detailed and len(info['details']) > 1:
            # Show breakdown of sub-components
            for detail in sorted(info['details'], key=lambda x: x['params'], reverse=True):
                if detail['params'] > 1000:  # Only show significant parameters
                    print(f"  ├─ {detail['name']:30} {detail['params']:>8,} {str(detail['shape']):>15}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'component_breakdown': component_breakdown
    }

def compare_tft_vs_vanilla(tft_model, vanilla_model):
    """Compare parameter counts between TFT and vanilla models."""
    print("\n🆚 TFT vs VANILLA COMPARISON")
    print("=" * 60)
    
    tft_analysis = analyze_model_parameters(tft_model, detailed=False)
    print("\n" + "-" * 60)
    vanilla_analysis = analyze_model_parameters(vanilla_model, detailed=False)
    
    diff = tft_analysis['total_params'] - vanilla_analysis['total_params']
    print(f"\n📈 DIFFERENCE: {diff:,} parameters ({diff/1e6:.1f}M)")
    
    if diff > 0:
        print("⚠️ TFT model has MORE parameters than vanilla!")
    else:
        print("✅ TFT model has fewer/same parameters as vanilla")

def estimate_expected_parameters(config):
    """Estimate expected parameter count for TFT model."""
    print("\n🧮 EXPECTED PARAMETER CALCULATION")
    print("=" * 60)
    
    vocab_size = config.vocab_size
    d_model = config.d_model
    n_layers = config.n_layers
    n_heads = config.n_heads
    d_ff = config.d_ff
    
    # Token embeddings (shared with lm_head)
    token_emb = vocab_size * d_model
    print(f"Token embeddings: {token_emb:,} ({vocab_size:,} × {d_model})")
    
    # Per-layer parameters
    layer_params = 0
    
    # Attention (Q, K projections - no V in your factored version)
    qk_proj = 2 * d_model * d_model  # Q and K projections
    layer_params += qk_proj
    print(f"QK projections per layer: {qk_proj:,} (2 × {d_model} × {d_model})")
    
    # Optional V and output factorizations
    if hasattr(config, 'use_v') and config.use_v:
        v_fact = n_heads * n_heads
        layer_params += v_fact
        print(f"V factorization per layer: {v_fact:,} ({n_heads} × {n_heads})")
    
    if hasattr(config, 'use_proj') and config.use_proj:
        proj_fact = n_heads * n_heads  
        layer_params += proj_fact
        print(f"Output factorization per layer: {proj_fact:,} ({n_heads} × {n_heads})")
    
    # MLP
    mlp_params = d_model * d_ff + d_ff * d_model  # fc + proj
    layer_params += mlp_params
    print(f"MLP per layer: {mlp_params:,} ({d_model} × {d_ff} × 2)")
    
    # LayerNorms
    ln_params = 4 * d_model  # ln1, ln2, each with weight + bias
    layer_params += ln_params
    print(f"LayerNorms per layer: {ln_params:,} (4 × {d_model})")
    
    # Dictionary FFN additional parameters
    dict_params = 0
    if hasattr(config, 'use_dict_ffn') and config.use_dict_ffn:
        # Additional LayerNorms for heads
        dict_ln_params = n_heads * d_model // n_heads * 2  # weight + bias per head
        dict_params += dict_ln_params
        print(f"Dict LayerNorms per layer: {dict_ln_params:,} ({n_heads} × {d_model//n_heads} × 2)")
        
        layer_params += dict_params
    
    # Total layer parameters
    total_layer_params = layer_params * n_layers
    print(f"\nPer layer: {layer_params:,}")
    print(f"All layers: {total_layer_params:,} ({n_layers} × {layer_params:,})")
    
    # Final LayerNorm
    final_ln = d_model * 2  # weight + bias
    print(f"Final LayerNorm: {final_ln:,}")
    
    # Total expected
    expected_total = token_emb + total_layer_params + final_ln
    print(f"\n🎯 EXPECTED TOTAL: {expected_total:,} ({expected_total/1e6:.1f}M)")
    
    return expected_total

def find_parameter_culprits(model, expected_params):
    """Find what's causing unexpected parameter growth."""
    print("\n🕵️ CULPRIT DETECTION")
    print("=" * 60)
    
    analysis = analyze_model_parameters(model, detailed=False)
    actual_params = analysis['total_params']
    excess = actual_params - expected_params
    
    print(f"Expected: {expected_params:,}")
    print(f"Actual:   {actual_params:,}")
    print(f"Excess:   {excess:,} ({excess/1e6:.1f}M)")
    
    if excess > 1e6:  # More than 1M excess parameters
        print("\n🚨 SUSPICIOUS COMPONENTS:")
        
        # Look for unusually large components
        for component, info in analysis['component_breakdown'].items():
            component_params = info['params']
            
            # Check if this component seems too large
            if 'embedding' in component.lower() and component_params > 20e6:
                print(f"⚠️ {component}: {component_params:,} (embedding seems large)")
            elif 'head_layer_norms' in component and component_params > 1e6:
                print(f"⚠️ {component}: {component_params:,} (too many head norms?)")
            elif component_params > 10e6 and 'token' not in component:
                print(f"⚠️ {component}: {component_params:,} (unexpectedly large)")
        
        # Detailed analysis of largest components
        print("\n🔍 DETAILED ANALYSIS OF LARGE COMPONENTS:")
        sorted_components = sorted(analysis['component_breakdown'].items(), 
                                 key=lambda x: x[1]['params'], reverse=True)
        
        for component, info in sorted_components[:3]:  # Top 3 components
            print(f"\n{component} ({info['params']:,} params):")
            for detail in info['details'][:5]:  # Top 5 details
                print(f"  {detail['name']:40} {detail['shape']} → {detail['params']:,}")

def debug_dict_ffn_parameters(model):
    """Specifically debug dictionary FFN parameter usage."""
    print("\n📚 DICTIONARY FFN PARAMETER ANALYSIS")
    print("=" * 60)
    
    dict_related_params = 0
    
    for name, param in model.named_parameters():
        if any(keyword in name.lower() for keyword in ['dict', 'head_layer_norm']):
            param_count = param.numel()
            dict_related_params += param_count
            print(f"{name:50} {list(param.shape):>20} {param_count:>10,}")
    
    print(f"\nTotal dict-related parameters: {dict_related_params:,} ({dict_related_params/1e6:.1f}M)")
    
    # Check if parameters are being duplicated
    print("\n🔍 CHECKING FOR PARAMETER DUPLICATION:")
    
    # Look for shared vs separate embeddings
    token_emb_id = id(model.token_embedding.weight)
    lm_head_id = id(model.lm_head.weight)
    
    if token_emb_id == lm_head_id:
        print("✅ Token embedding and LM head share weights (good)")
    else:
        print("⚠️ Token embedding and LM head have separate weights (duplicated!)")
        dup_params = model.token_embedding.weight.numel()
        print(f"   Duplicated parameters: {dup_params:,} ({dup_params/1e6:.1f}M)")

# Comprehensive debugging function
def full_parameter_debug(model, config):
    """Run complete parameter debugging analysis."""
    print("🐛 COMPREHENSIVE PARAMETER DEBUGGING")
    print("=" * 80)
    
    # 1. Basic analysis
    analyze_model_parameters(model, detailed=True)
    
    # 2. Expected vs actual
    expected = estimate_expected_parameters(config)
    
    # 3. Find culprits
    find_parameter_culprits(model, expected)
    
    # 4. Dictionary FFN specific
    if hasattr(config, 'use_dict_ffn') and config.use_dict_ffn:
        debug_dict_ffn_parameters(model)
    
    print("\n" + "=" * 80)

# Quick parameter check function
def quick_param_check(model):
    """Quick parameter count with breakdown."""
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total:,} ({total/1e6:.1f}M)")
    
    # Quick breakdown
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        if module_params > 0:
            percentage = (module_params / total) * 100
            print(f"  {name:20} {module_params:>10,} ({percentage:5.1f}%)")
    
    return total