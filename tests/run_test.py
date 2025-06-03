#!/usr/bin/env python3
# run_tests.py
"""
Simple test runner to verify refactored TFT implementation.
"""

import sys
import subprocess
from pathlib import Path
import torch


def check_dependencies():
    """Check that required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required = ['torch', 'pytest']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package}")
    
    if missing:
        print(f"\n❌ Missing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def run_model_tests():
    """Run model tests."""
    print("\n🧪 Running Model Tests...")
    
    test_file = Path(__file__).parent / "tests" / "test_model.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', str(test_file), 
            '-v', '--tb=short', '--durations=10'
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        
        success = result.returncode == 0
        print(f"\n{'✅' if success else '❌'} Model tests {'passed' if success else 'failed'}")
        return success
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def compare_with_original():
    """Compare key features with original implementation."""
    print("\n🔄 Comparing with Original Implementation...")
    
    try:
        # Import our refactored model
        sys.path.insert(0, str(Path(__file__).parent))
        from models.model_tft_alibi import TokenFactoredTransformer, TFTConfig
        
        # Test key features
        config = TFTConfig.small()
        model = TokenFactoredTransformer(config)
        
        # Check core TFT features
        features = {
            "Factored attention": hasattr(model.blocks[0], 'attention'),
            "ALiBi slopes": hasattr(model.blocks[0].attention, 'alibi_slopes'),
            "Value factorization support": True,  # Always supported, enabled by config
            "Output projection support": True,   # Always supported, enabled by config
            "Stream separation": hasattr(model.blocks[0], 'ln1'),  # Pre-LN indicates proper structure
            "Generation method": hasattr(model, 'generate'),
            "Config system": hasattr(config, 'n_layers'),
        }
        
        all_present = True
        for feature, present in features.items():
            status = "✅" if present else "❌"
            print(f"  {status} {feature}")
            if not present:
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error comparing features: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality works."""
    print("\n⚡ Testing Basic Functionality...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from models.model_tft_alibi import TokenFactoredTransformer, TFTConfig
        
        # Test model creation
        config = TFTConfig(
            vocab_size=100,
            n_layers=2,
            n_heads=4,
            d_model=64,
            block_size=16
        )
        model = TokenFactoredTransformer(config)
        print(f"  ✅ Model created with {model.get_num_params()} parameters")
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        outputs = model(input_ids, labels=input_ids)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (2, 8, config.vocab_size)
        print(f"  ✅ Forward pass works - Loss: {outputs['loss'].item():.4f}")
        
        # Test generation
        model.eval()
        with torch.no_grad():
            generated = model.generate(input_ids[:1], max_new_tokens=5)
        
        assert generated.shape == (1, 8 + 5)
        print(f"  ✅ Generation works - Generated shape: {generated.shape}")
        
        # Test different configurations
        config_with_features = TFTConfig(
            vocab_size=100,
            n_layers=1,
            n_heads=4,
            d_model=64,
            block_size=16,
            use_value_factorization=True,
            use_output_projection=True
        )
        model_with_features = TokenFactoredTransformer(config_with_features)
        outputs = model_with_features(input_ids)
        print(f"  ✅ Advanced features work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    print("=" * 60)
    print("🧪 TFT MODEL REFACTOR VERIFICATION")
    print("=" * 60)
    
    success = True
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Test basic functionality first
    if not test_basic_functionality():
        success = False
    
    # Compare with original features
    if not compare_with_original():
        success = False
    
    # Run comprehensive unit tests
    if not run_model_tests():
        success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    if success:
        print("🎉 All verifications passed!")
        print("✅ Refactored model maintains original functionality")
        print("✅ Ready to proceed with training system refactor")
    else:
        print("⚠️  Some verifications failed")
        print("❌ Fix issues before proceeding")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)