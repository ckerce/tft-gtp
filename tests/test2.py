#!/usr/bin/env python3
"""
Quick experiment script to test Dictionary FFN implementation.
Usage: python exp/test_dict_ffn.py
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
from models import get_model
from config.model_configs import get_config, print_config


def test_dict_ffn():
    """Test the dictionary FFN implementation."""
    print("🧪 Testing Dictionary FFN Implementation\n")
    
    # Test configurations
    configs_to_test = [
        ('tiny', False),          # Baseline TFT
        ('tiny-dict', True),      # TFT with dictionary FFN
    ]
    
    for preset, should_have_dict in configs_to_test:
        print(f"Testing {preset}...")
        
        # Get config
        config = get_config(preset)
        print_config(config, f"{preset} Configuration")
        
        # Create model
        model_type = 'tft-dict' if config.use_dict_ffn else 'tft'
        model = get_model(model_type, config)
        
        # Print parameter count
        total_params = model.get_num_params()
        print(f"✅ Model created successfully: {total_params/1e6:.1f}M parameters\n")
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        print(f"Testing forward pass with input shape: {input_ids.shape}")
        
        # Forward pass
        model.train()
        outputs = model(input_ids, labels=labels)
        
        print(f"✅ Forward pass successful!")
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Loss: {outputs['loss'].item():.4f}")
        
        # Check dictionary-specific outputs
        if should_have_dict and config.use_dict_ffn:
            if 'dict_loss' in outputs:
                print(f"   Dictionary loss: {outputs['dict_loss']:.4f}")
            if 'dict_weights' in outputs:
                dict_weights_shape = outputs['dict_weights'].shape
                print(f"   Dict weights shape: {dict_weights_shape}")  # [B, L, H, T, V]
                
                # Analyze dictionary attention patterns
                dict_weights = outputs['dict_weights']
                # Get the top-5 dictionary entries for first sample, last token, first head, first layer
                sample_weights = dict_weights[0, 0, 0, -1, :]  # [V]
                top_k = 5
                top_indices = torch.topk(sample_weights, top_k)[1]
                top_weights = sample_weights[top_indices]
                
                print(f"   Top-{top_k} dictionary entries for head 0, layer 0:")
                for i, (idx, weight) in enumerate(zip(top_indices, top_weights)):
                    print(f"     {i+1}. Token {idx.item()}: {weight.item():.4f}")
        
        print(f"✅ {preset} test completed!\n" + "="*60 + "\n")
    
    # Test generation
    print("🎨 Testing generation...")
    config = get_config('tiny-dict')
    model = get_model('tft-dict', config)
    model.eval()
    
    # Simple generation test
    prompt = torch.tensor([[1, 2, 3]])  # Simple prompt
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    
    print(f"✅ Generation test successful!")
    print(f"   Input: {prompt.tolist()}")
    print(f"   Generated: {generated.tolist()}")
    print()
    
    print("🎉 All tests passed! Dictionary FFN implementation is working.")


def compare_parameter_counts():
    """Compare parameter counts between standard and dictionary FFN."""
    print("📊 Parameter Count Comparison\n")
    
    presets = ['tiny', 'small']
    
    for preset in presets:
        print(f"{preset.upper()} Configuration:")
        
        # Standard TFT
        config_standard = get_config(preset, use_dict_ffn=False)
        model_standard = get_model('tft', config_standard)
        params_standard = model_standard.get_num_params()
        
        # Dictionary TFT (full vocab)
        config_dict_full = get_config(preset, use_dict_ffn=True, dict_vocab_size=config_standard.vocab_size)
        model_dict_full = get_model('tft-dict', config_dict_full)
        params_dict_full = model_dict_full.get_num_params()
        
        # Dictionary TFT (reduced vocab)
        config_dict_reduced = get_config(preset, use_dict_ffn=True, dict_vocab_size=1000)
        model_dict_reduced = get_model('tft-dict', config_dict_reduced)
        params_dict_reduced = model_dict_reduced.get_num_params()
        
        print(f"  Standard TFT:        {params_standard/1e6:6.1f}M params")
        print(f"  +Dict FFN (full):    {params_dict_full/1e6:6.1f}M params ({params_dict_full/params_standard:.1f}x)")
        print(f"  +Dict FFN (1K vocab): {params_dict_reduced/1e6:6.1f}M params ({params_dict_reduced/params_standard:.1f}x)")
        
        overhead_full = params_dict_full - params_standard
        overhead_reduced = params_dict_reduced - params_standard
        print(f"  Overhead (full):     {overhead_full/1e6:6.1f}M params")
        print(f"  Overhead (reduced):  {overhead_reduced/1e6:6.1f}M params")
        print()


if __name__ == "__main__":
    print("🚀 Dictionary FFN Testing Suite\n")
    
    try:
        test_dict_ffn()
        print("\n" + "="*60 + "\n")
        compare_parameter_counts()
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)