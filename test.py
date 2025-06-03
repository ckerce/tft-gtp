#!/usr/bin/env python3
# test_original_simple.py
"""
Test the original model with the exact settings that should work.
"""

import sys
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_original_basic():
    """Test original model with basic settings that should work."""
    print("🔍 Testing original model with basic settings...")
    
    try:
        from models.old_model import FactoredTransformerModelALiBi
        from config.old_config import GPTConfigALiBi
        
        # Create a simple tokenizer-like object
        class SimpleTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
            def decode(self, tokens):
                return " ".join([f"tok_{t}" for t in tokens])
        
        # Use the exact same settings as your working examples but smaller
        config = GPTConfigALiBi(
            vocab_size=1000,   # Smaller vocab for testing
            n_layer=1,         # Single layer first
            n_head=2,          # Fewer heads
            n_embd=64,         # Smaller embedding
            block_size=16,     # Smaller block
            dropout=0.0,
            bias=False,
            use_v=False,       # Start with no factorization
            use_proj=False,
            max_position_embeddings=32
        )
        
        print(f"Config: {config.n_layer}L, {config.n_head}H, {config.n_embd}D")
        print(f"Block size: {config.block_size}, Max pos: {config.max_position_embeddings}")
        print(f"Use factorization: v={config.use_v}, proj={config.use_proj}")
        
        model = FactoredTransformerModelALiBi(config)
        
        # Set the tokenizer that your model expects
        model.tokenizer = SimpleTokenizer(config.vocab_size)
        
        print(f"✅ Model created: {model.get_num_params()/1e6:.2f}M params")
        
        # Test with a sequence that fits in block_size
        seq_len = 8  # Use small sequence
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        print(f"Testing with input shape: {input_ids.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids)
            print(f"✅ Forward pass: {output['logits'].shape}")
            
            # Test with labels
            output = model(input_ids=input_ids, labels=input_ids)
            print(f"✅ Forward with labels: loss={output['loss'].item():.4f}")
            
            # Test generation
            gen_output = model.generate(idx=input_ids, max_new_tokens=2)
            print(f"✅ Generation: {gen_output.shape}")
        
        return True, config, model
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        
        # Let's check what specific operation is failing
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        return False, None, None

def test_new_model_matching():
    """Test new model with matching configuration."""
    print("\n🔍 Testing new model with matching config...")
    
    try:
        from models.model_tft_alibi import TokenFactoredTransformer, TFTConfig
        
        # Create matching config for new model
        config = TFTConfig(
            vocab_size=50257,
            n_layers=2,
            n_heads=6,
            d_model=384,
            block_size=128,
            dropout=0.0,
            bias=False,
            use_value_factorization=False,
            use_output_projection=False,
            max_position_embeddings=256
        )
        
        print(f"New config: {config.n_layers}L, {config.n_heads}H, {config.d_model}D")
        
        model = TokenFactoredTransformer(config)
        print(f"✅ New model created: {model.get_num_params()/1e6:.2f}M params")
        
        # Test with same input as original
        seq_len = 16
        input_ids = torch.randint(0, 1000, (1, seq_len))
        print(f"Testing with input shape: {input_ids.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
            print(f"✅ Forward pass: {output['logits'].shape}")
            
            output = model(input_ids, labels=input_ids)
            print(f"✅ Forward with labels: loss={output['loss'].item():.4f}")
            
            gen_output = model.generate(input_ids, max_new_tokens=3)
            print(f"✅ Generation: {gen_output.shape}")
        
        return True, config, model
        
    except Exception as e:
        print(f"❌ New model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    print("=" * 60)
    print("🧪 SIMPLE MODEL TESTING")
    print("=" * 60)
    
    # Test original model first
    orig_success, orig_config, orig_model = test_original_basic()
    
    if orig_success:
        print("\n🎉 Original model works!")
        
        # Now test new model
        new_success, new_config, new_model = test_new_model_matching()
        
        if new_success:
            print("\n🎉 Both models work!")
            
            # Quick comparison
            print("\n📊 Quick comparison:")
            print(f"Original params: {orig_model.get_num_params():,}")
            print(f"New params:      {new_model.get_num_params():,}")
            
            param_diff = abs(orig_model.get_num_params() - new_model.get_num_params())
            print(f"Difference:      {param_diff:,} ({param_diff/orig_model.get_num_params()*100:.1f}%)")
            
        else:
            print("\n❌ New model failed")
    else:
        print("\n❌ Original model failed - need to fix this first")

if __name__ == "__main__":
    main()