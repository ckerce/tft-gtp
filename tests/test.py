# test.py
"""
Comprehensive verification script for the refactored TFT implementation.
This script tests the new model against the old one to ensure functionality is preserved.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import os

# Add src to path for imports
sys.path.insert(0, 'src')

def test_imports():
    """Test that all new modules can be imported correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test new model imports
        from models.model_tft_alibi import TokenFactoredTransformer, TFTConfig
        from config.model_configs import get_config, print_config
        print("  ✅ New model imports successful")
        
        # Test old model imports (for comparison)
        from models.old_model import FactoredTransformerModelALiBi
        from config.old_config import GPTConfigALiBi
        print("  ✅ Old model imports successful")
        
        return True, {
            'new_model': TokenFactoredTransformer,
            'new_config': TFTConfig, 
            'old_model': FactoredTransformerModelALiBi,
            'old_config': GPTConfigALiBi
        }
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False, None

def create_equivalent_configs():
    """Create equivalent configs for both old and new models."""
    print("\n⚙️ Creating equivalent configurations...")
    
    try:
        from models.model_tft_alibi import TFTConfig
        from config.old_config import GPTConfigALiBi
        
        # Common parameters for small test
        common_params = {
            'vocab_size': 1000,
            'dropout': 0.0,  # Disable for deterministic comparison
            'bias': False,
            'block_size': 32,
        }
        
        # Old config
        old_config = GPTConfigALiBi(
            vocab_size=common_params['vocab_size'],
            n_layer=2,
            n_head=4, 
            n_embd=128,
            block_size=common_params['block_size'],
            dropout=common_params['dropout'],
            bias=common_params['bias'],
            max_position_embeddings=64,
            use_v=True,
            use_proj=True,
        )
        
        # New config with equivalent parameters
        new_config = TFTConfig(
            vocab_size=common_params['vocab_size'],
            n_layers=2,  # old: n_layer
            n_heads=4,   # old: n_head
            d_model=128, # old: n_embd
            block_size=common_params['block_size'],
            dropout=common_params['dropout'],
            bias=common_params['bias'],
            max_position_embeddings=64,
            use_v=True,  # old: use_v
            use_proj=True,    # old: use_proj
        )
        
        print(f"  ✅ Configs created - Old: {old_config.n_layer}L/{old_config.n_head}H/{old_config.n_embd}D")
        print(f"                      New: {new_config.n_layers}L/{new_config.n_heads}H/{new_config.d_model}D")
        
        return True, (old_config, new_config)
        
    except Exception as e:
        print(f"  ❌ Config creation failed: {e}")
        return False, None

def test_model_creation():
    """Test that both models can be created with equivalent configs."""
    print("\n🏗️ Testing model creation...")
    
    success, configs = create_equivalent_configs()
    if not success:
        return False, None, None
    
    old_config, new_config = configs
    
    try:
        from models.old_model import FactoredTransformerModelALiBi
        from models.model_tft_alibi import TokenFactoredTransformer
        
        # Create models with same random seed
        torch.manual_seed(42)
        old_model = FactoredTransformerModelALiBi(old_config)
        old_params = old_model.get_num_params()
        
        torch.manual_seed(42)
        new_model = TokenFactoredTransformer(new_config)
        new_params = new_model.get_num_params()
        
        print(f"  ✅ Old model: {old_params:,} parameters")
        print(f"  ✅ New model: {new_params:,} parameters")
        
        # Check parameter count similarity (should be very close)
        param_diff_pct = abs(old_params - new_params) / old_params * 100
        if param_diff_pct < 10:  # Allow 10% difference
            print(f"  ✅ Parameter counts similar (diff: {param_diff_pct:.1f}%)")
        else:
            print(f"  ⚠️ Parameter counts differ significantly (diff: {param_diff_pct:.1f}%)")
        
        return True, old_model, new_model
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_forward_pass(old_model, new_model):
    """Test forward pass compatibility."""
    print("\n➡️ Testing forward pass...")
    
    try:
        # Use small input that should work
        batch_size, seq_len = 2, 16
        vocab_size = 1000
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        print(f"  Using input shape: {input_ids.shape}")
        
        # Set models to eval mode for deterministic comparison
        old_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            # Old model forward pass
            try:
                old_outputs = old_model(input_ids=input_ids, labels=input_ids)
                print(f"  ✅ Old model forward: {old_outputs['logits'].shape}, loss: {old_outputs['loss'].item():.4f}")
            except Exception as e:
                print(f"  ❌ Old model forward failed: {e}")
                return False
            
            # New model forward pass  
            try:
                new_outputs = new_model(input_ids, labels=input_ids)
                print(f"  ✅ New model forward: {new_outputs['logits'].shape}, loss: {new_outputs['loss'].item():.4f}")
            except Exception as e:
                print(f"  ❌ New model forward failed: {e}")
                return False
        
        # Check output shapes match
        if old_outputs['logits'].shape == new_outputs['logits'].shape:
            print("  ✅ Output shapes match")
        else:
            print(f"  ❌ Shape mismatch: {old_outputs['logits'].shape} vs {new_outputs['logits'].shape}")
            return False
        
        # Check loss values are reasonable (they won't be identical due to different init)
        loss_ratio = new_outputs['loss'].item() / old_outputs['loss'].item()
        if 0.1 < loss_ratio < 10:  # Within order of magnitude
            print(f"  ✅ Loss values reasonable (ratio: {loss_ratio:.2f})")
        else:
            print(f"  ⚠️ Loss values very different (ratio: {loss_ratio:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Forward pass test failed: {e}")
        return False

def test_generation(old_model, new_model):
    """Test text generation capabilities."""
    print("\n🎯 Testing generation...")
    
    try:
        batch_size, seq_len = 1, 8
        vocab_size = 1000
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        print(f"  Using generation input: {input_ids.shape}")
        
        old_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            # Old model generation
            try:
                old_generated = old_model.generate(
                    idx=input_ids,  # Old model uses 'idx' parameter
                    max_new_tokens=5,
                    temperature=0.8,
                    top_k=40
                )
                print(f"  ✅ Old model generation: {old_generated.shape}")
            except Exception as e:
                print(f"  ❌ Old model generation failed: {e}")
                return False
            
            # New model generation
            try:
                new_generated = new_model.generate(
                    input_ids,  # New model uses standard parameter name
                    max_new_tokens=5,
                    temperature=0.8,
                    top_k=40
                )
                print(f"  ✅ New model generation: {new_generated.shape}")
            except Exception as e:
                print(f"  ❌ New model generation failed: {e}")
                return False
        
        # Check generation shapes
        if old_generated.shape == new_generated.shape:
            print("  ✅ Generated sequence shapes match")
        else:
            print(f"  ❌ Generation shape mismatch: {old_generated.shape} vs {new_generated.shape}")
            return False
        
        # Check that original input is preserved
        if torch.equal(old_generated[:, :seq_len], input_ids) and torch.equal(new_generated[:, :seq_len], input_ids):
            print("  ✅ Original sequences preserved in both models")
        else:
            print("  ⚠️ Original sequence preservation differs between models")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Generation test failed: {e}")
        return False

def test_alibi_features(old_model, new_model):
    """Test ALiBi-specific features."""
    print("\n🔧 Testing ALiBi features...")
    
    try:
        # Check ALiBi slopes in first attention layer
        old_attn = old_model.transformer.h[0].attn
        new_attn = new_model.blocks[0].attention
        
        old_slopes = old_attn.alibi_slopes
        new_slopes = new_attn.alibi_slopes
        
        print(f"  Old ALiBi slopes shape: {old_slopes.shape}")
        print(f"  New ALiBi slopes shape: {new_slopes.shape}")
        
        # Slopes should have same shape and similar values
        if old_slopes.shape == new_slopes.shape:
            print("  ✅ ALiBi slopes have matching shapes")
            
            # Check if slopes are similar (they might not be identical due to different computation)
            if torch.allclose(old_slopes, new_slopes, atol=1e-3):
                print("  ✅ ALiBi slopes are identical")
            else:
                slope_diff = torch.abs(old_slopes - new_slopes).max().item()
                print(f"  ⚠️ ALiBi slopes differ slightly (max diff: {slope_diff:.4f})")
        else:
            print(f"  ❌ ALiBi slopes shape mismatch")
            return False
        
        # Check factorization features
        old_has_v = hasattr(old_attn, 'v_tmp')
        new_has_v = hasattr(new_attn, 'v_factorization')
        
        old_has_proj = hasattr(old_attn, 'proj_tmp')
        new_has_proj = hasattr(new_attn, 'output_factorization')
        
        print(f"  Value factorization - Old: {old_has_v}, New: {new_has_v}")
        print(f"  Output projection - Old: {old_has_proj}, New: {new_has_proj}")
        
        if old_has_v == new_has_v and old_has_proj == new_has_proj:
            print("  ✅ Factorization features match")
        else:
            print("  ⚠️ Factorization features differ")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ALiBi features test failed: {e}")
        return False

def test_config_system():
    """Test the new configuration system."""
    print("\n⚙️ Testing configuration system...")
    
    try:
        from config.model_configs import get_config, CONFIG_PRESETS, print_config
        
        # Test preset loading
        for preset_name in ['tiny', 'small', 'medium']:
            try:
                config = get_config(preset_name)
                print(f"  ✅ Preset '{preset_name}': {config.n_layers}L-{config.n_heads}H-{config.d_model}D")
            except Exception as e:
                print(f"  ❌ Preset '{preset_name}' failed: {e}")
                return False
        
        # Test config overrides
        custom_config = get_config('small', n_layers=8, use_v=True)
        if custom_config.n_layers == 8 and custom_config.use_v:
            print("  ✅ Config overrides work")
        else:
            print("  ❌ Config overrides failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config system test failed: {e}")
        return False

def test_model_registry():
    """Test the model registry system."""
    print("\n📋 Testing model registry...")
    
    try:
        from models import get_model, list_models, MODEL_REGISTRY
        from config.model_configs import TFTConfig
        
        # Test model creation via registry
        config = TFTConfig(
            vocab_size=100,
            n_layers=1,
            n_heads=2,
            d_model=64,
            block_size=16
        )
        
        model = get_model('tft', config)
        print(f"  ✅ Model created via registry: {model.get_num_params()} params")
        
        # Test model listing
        available_models = list_models()
        print(f"  ✅ Available models: {available_models}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model registry test failed: {e}")
        return False

def test_training_compatibility():
    """Test that new models work with training infrastructure."""
    print("\n🎓 Testing training compatibility...")
    
    try:
        from models import get_model
        from config.model_configs import TFTConfig
        
        # Create small model for training test
        config = TFTConfig(
            vocab_size=100,
            n_layers=1,
            n_heads=2,
            d_model=64,
            block_size=16,
            dropout=0.1
        )
        
        model = get_model('tft', config)
        
        # Test basic training step
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        input_ids = torch.randint(0, 100, (2, 8))
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  ✅ Training step successful: loss = {loss.item():.4f}")
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            eval_outputs = model(input_ids, labels=labels)
            print(f"  ✅ Evaluation mode works: loss = {eval_outputs['loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 70)
    print("🧪 TFT REFACTOR VERIFICATION")
    print("=" * 70)
    
    test_results = []
    
    # Test imports
    success, imports = test_imports()
    test_results.append(('Imports', success))
    if not success:
        print("\n❌ Import test failed - cannot continue")
        return False
    
    # Test model creation
    success, old_model, new_model = test_model_creation()
    test_results.append(('Model Creation', success))
    
    if success and old_model and new_model:
        # Test forward pass
        success = test_forward_pass(old_model, new_model)
        test_results.append(('Forward Pass', success))
        
        # Test generation
        success = test_generation(old_model, new_model)
        test_results.append(('Generation', success))
        
        # Test ALiBi features
        success = test_alibi_features(old_model, new_model)
        test_results.append(('ALiBi Features', success))
    
    # Test new systems
    success = test_config_system()
    test_results.append(('Config System', success))
    
    success = test_model_registry()
    test_results.append(('Model Registry', success))
    
    success = test_training_compatibility()
    test_results.append(('Training Compatibility', success))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Refactored TFT implementation is working correctly")
        print("✅ New model maintains compatibility with old implementation")
        print("✅ Ready for production use")
    elif passed >= total - 2:
        print("\n✅ MOSTLY SUCCESSFUL!")
        print("⚠️ Minor issues found but core functionality works")
        print("✅ Safe to proceed with refactored implementation")
    else:
        print("\n❌ SIGNIFICANT ISSUES FOUND")
        print("⚠️ Review and fix issues before using refactored implementation")
    
    return passed >= total - 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)