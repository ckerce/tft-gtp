#!/usr/bin/env python3
"""
Detailed breakdown of how transformer parameters are calculated.
Shows the math behind parameter counting for different transformer variants.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ModelConfig:
    """Model configuration for parameter calculation."""
    vocab_size: int = 50257      # GPT-2 tokenizer
    n_layers: int = 6            # Number of transformer layers
    n_heads: int = 6             # Number of attention heads
    d_model: int = 768           # Model dimension
    d_ff: int = None             # FFN dimension (4 * d_model if None)
    bias: bool = False           # Whether to use bias terms
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


def calculate_vanilla_transformer_params(config: ModelConfig) -> Dict[str, int]:
    """Calculate parameters for a standard transformer."""
    params = {}
    
    # 1. TOKEN EMBEDDINGS
    # Embedding table: vocab_size × d_model
    params['token_embedding'] = config.vocab_size * config.d_model
    
    # 2. POSITIONAL EMBEDDINGS (if used)
    # Note: Your TFT uses ALiBi, so no positional embeddings
    params['positional_embedding'] = 0  # ALiBi uses no parameters
    
    # 3. PER LAYER COMPONENTS
    layer_params = {}
    
    # 3.1 LAYER NORMALIZATION (2 per layer: pre-attention, pre-FFN)
    # LayerNorm has: weight (d_model) + bias (d_model if bias=True)
    bias_params = config.d_model if config.bias else 0
    layer_params['layernorm'] = 2 * (config.d_model + bias_params)
    
    # 3.2 MULTI-HEAD ATTENTION
    # Query, Key, Value projections: d_model → d_model each
    # Output projection: d_model → d_model
    qkv_params = 3 * config.d_model * config.d_model  # Q, K, V
    if config.bias:
        qkv_params += 3 * config.d_model  # bias for Q, K, V
    
    output_proj_params = config.d_model * config.d_model
    if config.bias:
        output_proj_params += config.d_model
    
    layer_params['attention'] = qkv_params + output_proj_params
    
    # 3.3 FEED-FORWARD NETWORK
    # Linear 1: d_model → d_ff
    # Linear 2: d_ff → d_model
    ffn_params = (config.d_model * config.d_ff) + (config.d_ff * config.d_model)
    if config.bias:
        ffn_params += config.d_ff + config.d_model  # bias terms
    
    layer_params['ffn'] = ffn_params
    
    # Total per layer
    params_per_layer = sum(layer_params.values())
    layer_params['total_per_layer'] = params_per_layer
    
    # 4. ALL LAYERS
    params['all_layers'] = config.n_layers * params_per_layer
    params['layer_breakdown'] = layer_params
    
    # 5. FINAL LAYER NORM
    params['final_layernorm'] = config.d_model + (config.d_model if config.bias else 0)
    
    # 6. OUTPUT HEAD (usually tied with token embedding)
    # If tied: 0 additional parameters
    # If not tied: vocab_size × d_model
    params['output_head'] = 0  # Assuming weight tying (which your models use)
    
    # 7. TOTAL
    params['total'] = (params['token_embedding'] + 
                      params['positional_embedding'] + 
                      params['all_layers'] + 
                      params['final_layernorm'] + 
                      params['output_head'])
    
    return params


def calculate_tft_params(config: ModelConfig, use_v: bool = False, use_proj: bool = False) -> Dict[str, int]:
    """Calculate parameters for Token-Factored Transformer."""
    params = {}
    
    # Start with vanilla transformer base
    base_params = calculate_vanilla_transformer_params(config)
    params.update(base_params)
    
    # TFT modifications per layer
    tft_modifications = {}
    
    if use_v:
        # V factorization: n_heads × n_heads matrix per layer
        v_factorization_params = config.n_heads * config.n_heads
        tft_modifications['v_factorization_per_layer'] = v_factorization_params
        tft_modifications['v_factorization_total'] = config.n_layers * v_factorization_params
    
    if use_proj:
        # Output projection factorization: n_heads × n_heads matrix per layer
        proj_factorization_params = config.n_heads * config.n_heads
        tft_modifications['proj_factorization_per_layer'] = proj_factorization_params
        tft_modifications['proj_factorization_total'] = config.n_layers * proj_factorization_params
    
    # Add TFT-specific parameters
    tft_additional = sum(v for k, v in tft_modifications.items() if k.endswith('_total'))
    params['tft_additional'] = tft_additional
    params['tft_breakdown'] = tft_modifications
    
    # Update total
    params['total'] += tft_additional
    
    return params


def calculate_dictionary_ffn_params(config: ModelConfig, dict_vocab_size: int = None) -> Dict[str, int]:
    """Calculate additional parameters for dictionary FFN."""
    if dict_vocab_size is None:
        dict_vocab_size = config.vocab_size
    
    d_head = config.d_model // config.n_heads
    
    # Dictionary FFN replaces standard FFN
    # Standard FFN params per layer: d_model * d_ff + d_ff * d_model
    standard_ffn_per_layer = 2 * config.d_model * config.d_ff
    if config.bias:
        standard_ffn_per_layer += config.d_ff + config.d_model
    
    # Dictionary FFN per head:
    # - Standard FFN: d_model -> d_ff -> d_head
    # - LayerNorm: d_head params
    # - Dictionary embedding: dict_vocab_size × d_head
    dict_ffn_per_head = (
        config.d_model * config.d_ff +  # FC layer
        config.d_ff * d_head +           # Projection to head
        d_head +                         # LayerNorm weight
        (d_head if config.bias else 0) + # LayerNorm bias
        dict_vocab_size * d_head         # Dictionary embedding
    )
    
    dict_ffn_per_layer = config.n_heads * dict_ffn_per_head
    
    # Difference from standard FFN
    additional_params_per_layer = dict_ffn_per_layer - standard_ffn_per_layer
    total_additional = config.n_layers * additional_params_per_layer
    
    return {
        'standard_ffn_per_layer': standard_ffn_per_layer,
        'dict_ffn_per_layer': dict_ffn_per_layer,
        'additional_per_layer': additional_params_per_layer,
        'total_additional': total_additional,
        'dict_vocab_size': dict_vocab_size,
        'embeddings_per_head': dict_vocab_size * d_head,
        'total_embedding_params': config.n_layers * config.n_heads * dict_vocab_size * d_head
    }


def print_parameter_breakdown(config: ModelConfig):
    """Print detailed parameter breakdown for different model variants."""
    print("="*80)
    print(f"PARAMETER BREAKDOWN")
    print("="*80)
    print(f"Configuration:")
    print(f"  Vocab Size: {config.vocab_size:,}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Model Dim: {config.d_model}")
    print(f"  FFN Dim: {config.d_ff}")
    print(f"  Bias: {config.bias}")
    print()
    
    # 1. Vanilla Transformer
    print("1. VANILLA TRANSFORMER")
    print("-"*40)
    vanilla_params = calculate_vanilla_transformer_params(config)
    
    print(f"Token Embedding:     {vanilla_params['token_embedding']:>12,} params")
    print(f"Positional Embedding: {vanilla_params['positional_embedding']:>11,} params (ALiBi = 0)")
    print(f"All Layers:          {vanilla_params['all_layers']:>12,} params")
    print(f"  - Per Layer:       {vanilla_params['layer_breakdown']['total_per_layer']:>12,} params")
    print(f"    - LayerNorms:    {vanilla_params['layer_breakdown']['layernorm']:>12,} params")
    print(f"    - Attention:     {vanilla_params['layer_breakdown']['attention']:>12,} params")
    print(f"    - FFN:           {vanilla_params['layer_breakdown']['ffn']:>12,} params")
    print(f"Final LayerNorm:     {vanilla_params['final_layernorm']:>12,} params")
    print(f"Output Head:         {vanilla_params['output_head']:>12,} params (tied)")
    print(f"{'TOTAL:':>17} {vanilla_params['total']:>12,} params ({vanilla_params['total']/1e6:.2f}M)")
    print()
    
    # 2. TFT Variants
    for use_v, use_proj in [(False, False), (True, False), (False, True), (True, True)]:
        variant_name = f"TFT"
        if use_v or use_proj:
            features = []
            if use_v: features.append("V-factorization")
            if use_proj: features.append("Proj-factorization")
            variant_name += f" ({', '.join(features)})"
        
        print(f"2. {variant_name.upper()}")
        print("-"*(len(variant_name) + 3))
        
        tft_params = calculate_tft_params(config, use_v=use_v, use_proj=use_proj)
        
        print(f"Base Transformer:    {vanilla_params['total']:>12,} params")
        
        if 'tft_breakdown' in tft_params:
            for key, value in tft_params['tft_breakdown'].items():
                if key.endswith('_total'):
                    feature_name = key.replace('_total', '').replace('_', ' ').title()
                    print(f"{feature_name:>17}: {value:>12,} params")
        
        if tft_params.get('tft_additional', 0) > 0:
            print(f"TFT Additional:      {tft_params['tft_additional']:>12,} params")
        
        print(f"{'TOTAL:':>17} {tft_params['total']:>12,} params ({tft_params['total']/1e6:.2f}M)")
        print()
    
    # 3. Dictionary FFN
    print("3. DICTIONARY FFN ANALYSIS")
    print("-"*30)
    
    for dict_vocab in [1000, 5000, config.vocab_size]:
        dict_params = calculate_dictionary_ffn_params(config, dict_vocab)
        base_total = vanilla_params['total']
        dict_total = base_total + dict_params['total_additional']
        
        print(f"Dict Vocab Size: {dict_vocab:,}")
        print(f"  Additional params:   {dict_params['total_additional']:>12,} ({dict_params['total_additional']/1e6:.2f}M)")
        print(f"  Total model:         {dict_total:>12,} ({dict_total/1e6:.2f}M)")
        print(f"  Overhead:            {dict_params['total_additional']/base_total*100:>11.1f}%")
        print()


def verify_with_actual_model():
    """Verify calculations against actual model implementation."""
    print("4. VERIFICATION WITH ACTUAL MODELS")
    print("-"*40)
    
    try:
        import sys
        sys.path.insert(0, 'src')
        from config.model_configs import get_config
        from models import get_model
        
        # Test with your actual config
        config_obj = get_config('small')  # 6L-6H-384D
        
        print(f"Actual config from your code:")
        print(f"  Layers: {config_obj.n_layers}")
        print(f"  Heads: {config_obj.n_heads}")
        print(f"  Model Dim: {config_obj.d_model}")
        print(f"  FFN Dim: {config_obj.d_ff}")
        print()
        
        # Create actual models and count
        models_to_test = [
            ('vanilla', 'vanilla'),
            ('tft', 'tft'),
            ('tft-dict', 'tft-dict')
        ]
        
        for model_name, model_type in models_to_test:
            try:
                model = get_model(model_type, config_obj)
                actual_params = model.get_num_params()
                
                # Calculate theoretical
                calc_config = ModelConfig(
                    vocab_size=config_obj.vocab_size,
                    n_layers=config_obj.n_layers,
                    n_heads=config_obj.n_heads,
                    d_model=config_obj.d_model,
                    d_ff=config_obj.d_ff,
                    bias=config_obj.bias
                )
                
                if model_type == 'vanilla':
                    theoretical = calculate_vanilla_transformer_params(calc_config)['total']
                else:
                    theoretical = calculate_tft_params(calc_config)['total']
                
                print(f"{model_name:>12}: {actual_params:>10,} params (actual)")
                print(f"{'':>12}  {theoretical:>10,} params (calculated)")
                print(f"{'':>12}  {'✅ Match' if actual_params == theoretical else f'❌ Diff: {actual_params - theoretical:+,}'}")
                print()
                
            except Exception as e:
                print(f"{model_name:>12}: ❌ Error creating model: {e}")
        
    except ImportError as e:
        print(f"❌ Cannot import model code: {e}")
        print("Run this script from your project root directory")


def main():
    """Main function to demonstrate parameter calculations."""
    # Your model configuration
    config = ModelConfig(
        vocab_size=50257,  # GPT-2 tokenizer
        n_layers=6,
        n_heads=6,
        d_model=768,       # This matches your outputs
        bias=False         # Your models use bias=False
    )
    
    print_parameter_breakdown(config)
    verify_with_actual_model()
    
    # Quick formulas
    print("="*80)
    print("QUICK FORMULAS")
    print("="*80)
    print("For a transformer with:")
    print("  V = vocab_size, L = layers, H = heads, D = d_model, F = d_ff")
    print()
    print("Embeddings:        V × D")
    print("Per Layer:         2×D (LayerNorms) + 4×D² (Attention) + 2×D×F (FFN)")
    print("All Layers:        L × (2×D + 4×D² + 2×D×F)")
    print("Total (no bias):   V×D + L×(2×D + 4×D² + 2×D×F) + D")
    print()
    print("For your config (V=50257, L=6, H=6, D=768, F=3072):")
    v, l, h, d, f = 50257, 6, 6, 768, 3072
    embedding_params = v * d
    layer_params = l * (2*d + 4*d*d + 2*d*f)
    final_ln = d
    total = embedding_params + layer_params + final_ln
    
    print(f"Embeddings:        {embedding_params:,}")
    print(f"All Layers:        {layer_params:,}")
    print(f"Final LayerNorm:   {final_ln:,}")
    print(f"TOTAL:             {total:,} ({total/1e6:.2f}M)")


if __name__ == "__main__":
    main()