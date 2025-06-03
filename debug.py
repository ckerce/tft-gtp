#!/usr/bin/env python3
# debug_original_model.py
"""
Debug just the original model to understand its requirements.
"""

import sys
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

def debug_original_model():
    """Debug the original model to understand its interface."""
    print("🔍 Debugging original model...")
    
    try:
        from models.old_model import FactoredTransformerModelALiBi
        from config.old_config import GPTConfigALiBi
        
        # Try different config settings to find what works
        configs_to_try = [
            # Config 1: Very small
            {
                'vocab_size': 100,
                'n_layer': 1,
                'n_head': 2,
                'n_embd': 32,
                'block_size': 8,
                'dropout': 0.0
            },
            # Config 2: Default-ish
            {
                'vocab_size': 50257,  # GPT-2 vocab size
                'n_layer': 2,
                'n_head': 4,
                'n_embd': 128,
                'block_size': 16,
                'dropout': 0.0
            },
            # Config 3: Match expected dimensions
            {
                'vocab_size': 1024,  # Match the error message
                'n_layer': 2,
                'n_head': 4,
                'n_embd': 64,
                'block_size': 8,
                'dropout': 0.0
            }
        ]
        
        working_config = None
        working_model = None
        
        for i, config_params in enumerate(configs_to_try):
            print(f"\n--- Trying Config {i+1} ---")
            print(f"  vocab_size: {config_params['vocab_size']}")
            print(f"  n_embd: {config_params['n_embd']}")
            print(f"  block_size: {config_params['block_size']}")
            
            try:
                config = GPTConfigALiBi(**config_params)
                model = FactoredTransformerModelALiBi(config)
                print(f"  ✅ Model created successfully")
                print(f"  Parameters: {model.get_num_params():,}")
                
                # Test forward pass with different input sizes
                for seq_len in [2, 4, 8]:
                    try:
                        input_ids = torch.randint(0, config_params['vocab_size'], (1, seq_len))
                        print(f"  Testing input shape: {input_ids.shape}")
                        
                        # Try forward pass
                        output = model(input_ids=input_ids)
                        print(f"    ✅ Forward pass works: output shape {output['logits'].shape}")
                        
                        # Try with labels
                        output_with_labels = model(input_ids=input_ids, labels=input_ids)
                        if output_with_labels.get('loss') is not None:
                            print(f"    ✅ Loss computation works: {output_with_labels['loss'].item():.4f}")
                        
                        # Try generation
                        gen_output = model.generate(idx=input_ids, max_new_tokens=2)
                        print(f"    ✅ Generation works: {gen_output.shape}")
                        
                        # This config works!
                        working_config = config
                        working_model = model
                        print(f"  🎉 Config {i+1} works perfectly!")
                        break
                        
                    except Exception as e:
                        print(f"    ❌ Seq len {seq_len} failed: {str(e)[:100]}...")
                
                if working_config:
                    break
                    
            except Exception as e:
                print(f"  ❌ Config {i+1} failed: {str(e)[:100]}...")
        
        if working_config:
            print(f"\n🎉 Found working configuration!")
            print(f"  vocab_size: {working_config.vocab_size}")
            print(f"  n_layer: {working_config.n_layer}")
            print(f"  n_head: {working_config.n_head}")
            print(f"  n_embd: {working_config.n_embd}")
            print(f"  block_size: {working_config.block_size}")
            print(f"  max_position_embeddings: {working_config.max_position_embeddings}")
            
            # Test edge cases
            print(f"\n🧪 Testing edge cases...")
            
            # Test larger sequences
            max_working_seq = working_config.block_size
            for seq_len in [max_working_seq, max_working_seq + 1, max_working_seq * 2]:
                try:
                    if seq_len <= working_config.max_position_embeddings:
                        input_ids = torch.randint(0, working_config.vocab_size, (1, seq_len))
                        output = working_model(input_ids=input_ids)
                        print(f"    ✅ Seq len {seq_len} works")
                    else:
                        print(f"    ⚠️ Seq len {seq_len} exceeds max_position_embeddings")
                except Exception as e:
                    print(f"    ❌ Seq len {seq_len} fails: {str(e)[:100]}...")
            
            return working_config, working_model
        else:
            print(f"\n❌ No working configuration found!")
            return None, None
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def inspect_model_internals(model, config):
    """Inspect the internal structure of the working model."""
    print(f"\n🔍 Inspecting model internals...")
    
    try:
        print(f"Model structure:")
        print(f"  Embedding: {model.transformer.wte}")
        print(f"  Layers: {len(model.transformer.h)}")
        print(f"  LM Head: {model.lm_head}")
        
        # Check first layer structure
        first_layer = model.transformer.h[0]
        print(f"  First layer attention: {first_layer.attn}")
        print(f"  Has use_v: {hasattr(first_layer.attn, 'v_tmp')}")
        print(f"  Has use_proj: {hasattr(first_layer.attn, 'proj_tmp')}")
        
        # Check ALiBi slopes
        if hasattr(first_layer.attn, 'alibi_slopes'):
            print(f"  ALiBi slopes shape: {first_layer.attn.alibi_slopes.shape}")
            print(f"  ALiBi slopes: {first_layer.attn.alibi_slopes}")
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")

def main():
    print("=" * 60)
    print("🔬 ORIGINAL MODEL DEBUG")
    print("=" * 60)
    
    working_config, working_model = debug_original_model()
    
    if working_config and working_model:
        inspect_model_internals(working_model, working_config)
        
        print(f"\n" + "=" * 60)
        print("📋 SUMMARY")
        print("=" * 60)
        print("✅ Original model is working!")
        print(f"Use these parameters for comparison:")
        print(f"  vocab_size = {working_config.vocab_size}")
        print(f"  n_layer = {working_config.n_layer}")
        print(f"  n_head = {working_config.n_head}")
        print(f"  n_embd = {working_config.n_embd}")
        print(f"  block_size = {working_config.block_size}")
        print(f"  max_position_embeddings = {working_config.max_position_embeddings}")
        
    else:
        print(f"\n" + "=" * 60)
        print("❌ Could not get original model working")
        print("=" * 60)

if __name__ == "__main__":
    main()