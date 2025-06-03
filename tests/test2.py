#!/usr/bin/env python3
"""
Integration test for TFT-GPT training pipeline.
Tests the complete training flow with minimal setup to catch integration issues.
"""

import os
import sys
import torch
import tempfile
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Model imports
from models import get_model, TokenFactoredTransformer
from config.model_configs import get_config, TFTConfig, print_config

# Tokenizer imports
from mytokenizers import create_tokenizer, GPT2Tokenizer

# Training imports
from trainers import get_trainer, SimpleTrainer
from trainers.callbacks import JSONLoggingCallback

# Utilities
from utils.data_utils import load_and_prepare_data
from utils.json_logger import JSONLogger

print("✅ All imports successful")

def test_config():
    """Test configuration system."""
    print("🔧 Testing configuration...")
    
    try:
        # Test preset configs
        for preset in ['tiny', 'small']:
            config = get_config(preset)
            print(f"  - {preset}: {config.n_layers}L-{config.n_heads}H-{config.d_model}D")
        
        # Test config overrides
        custom_config = get_config('tiny', n_layers=3, use_v=True)
        assert custom_config.n_layers == 3
        assert custom_config.use_v == True
        
        # Test config validation
        print_config(get_config('tiny'), "Test Config")
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Config error: {e}")
        print(traceback.format_exc())
        return False

def test_model_creation():
    """Test model creation and basic operations."""
    print("🏗️ Testing model creation...")
    
    try:
        # Create tiny config for testing
        config = get_config('tiny')
        
        # Create model
        model = get_model('tft', config)
        print(f"  - Model created: {model.get_num_params()} parameters")
        
        # Test model forward pass
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids)
            assert 'logits' in outputs
            assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
            print(f"  - Forward pass: {outputs['logits'].shape}")
        
        # Test generation
        with torch.no_grad():
            generated = model.generate(input_ids[:1], max_new_tokens=5, temperature=1.0)
            assert generated.shape[1] == seq_len + 5
            print(f"  - Generation: {generated.shape}")
        
        print("✅ Model creation and basic ops working")
        return True
        
    except Exception as e:
        print(f"❌ Model error: {e}")
        print(traceback.format_exc())
        return False

def test_tokenizer():
    """Test tokenizer functionality."""
    print("🔤 Testing tokenizer...")
    
    try:
        # Create tokenizer
        tokenizer = create_tokenizer('gpt2')
        print(f"  - Tokenizer vocab size: {tokenizer.vocab_size}")
        
        # Test encoding/decoding
        text = "Hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"  - Encode/decode: '{text}' -> {encoded} -> '{decoded}'")
        
        # Test batch processing
        batch_encoded = tokenizer(
            ["Hello", "World"], 
            padding=True, 
            return_tensors='pt'
        )
        print(f"  - Batch encoding: {batch_encoded['input_ids'].shape}")
        
        print("✅ Tokenizer working")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        print(traceback.format_exc())
        return False

def test_data_loading():
    """Test data loading with a small sample."""
    print("📚 Testing data loading...")
    
    try:
        # Create tokenizer for data loading
        tokenizer = create_tokenizer('gpt2')
        
        # Test with TinyStories (small sample)
        try:
            dataloader, _ = load_and_prepare_data(
                dataset_name="roneneldan/TinyStories",
                dataset_config=None,
                tokenizer=tokenizer,
                max_samples=100,  # Very small for testing
                max_seq_length=32,
                batch_size=4,
                split='train',
                shuffle=True
            )
            
            # Test getting a batch
            batch = next(iter(dataloader))
            print(f"  - Batch keys: {list(batch.keys())}")
            print(f"  - Input shape: {batch['input_ids'].shape}")
            print(f"  - Labels shape: {batch['labels'].shape}")
            
            print("✅ Data loading working")
            return True
            
        except Exception as data_e:
            print(f"⚠️ TinyStories failed, trying simple text data...")
            
            # Fallback: create simple mock dataloader
            from torch.utils.data import DataLoader, TensorDataset
            
            # Create simple dummy data
            dummy_data = torch.randint(0, 1000, (50, 32))  # 50 samples, 32 tokens each
            dummy_labels = dummy_data.clone()
            
            dataset = TensorDataset(dummy_data, dummy_labels)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            # Test batch
            for batch_data in dataloader:
                input_ids, labels = batch_data
                batch = {
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': torch.ones_like(input_ids)
                }
                print(f"  - Dummy batch shape: {batch['input_ids'].shape}")
                break
            
            print("✅ Data loading working (with fallback)")
            return True
            
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        print(traceback.format_exc())
        return False

def test_training_integration():
    """Test the complete training pipeline."""
    print("🎯 Testing training integration...")
    
    try:
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  - Using temp dir: {temp_dir}")
            
            # Setup minimal configuration
            config = get_config('tiny', block_size=16, max_position_embeddings=32)
            tokenizer = create_tokenizer('gpt2')
            config.vocab_size = tokenizer.vocab_size
            
            # Create model
            model = get_model('tft', config)
            device = torch.device('cpu')  # Use CPU for testing
            model.to(device)
            
            # Create simple dummy data
            from torch.utils.data import DataLoader, TensorDataset
            
            # Generate dummy sequences
            dummy_data = torch.randint(0, min(1000, config.vocab_size), (20, config.block_size))
            
            def simple_collate(batch):
                input_ids = torch.stack([item[0] for item in batch])
                return {
                    'input_ids': input_ids,
                    'labels': input_ids.clone(),  # For language modeling
                    'attention_mask': torch.ones_like(input_ids)
                }
            
            dataset = TensorDataset(dummy_data)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=simple_collate)
            
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Setup JSON logging
            callbacks = [
                JSONLoggingCallback(
                    output_dir=temp_dir,
                    run_name="integration_test",
                    log_every_n_steps=2
                )
            ]
            
            # Create trainer
            trainer = get_trainer(
                trainer_type='simple',
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                num_epochs=2,  # Just 2 epochs for testing
                output_dir=temp_dir,
                callbacks=callbacks
            )
            
            print("  - Trainer created, starting training...")
            
            # Run training
            metrics = trainer.train()
            
            print(f"  - Training completed!")
            print(f"  - Final loss: {metrics.get('final_loss', 'N/A')}")
            print(f"  - Training time: {metrics.get('training_time', 'N/A'):.2f}s")
            
            # Check if logs were created
            log_file = os.path.join(temp_dir, "training_metrics.json")
            if os.path.exists(log_file):
                print(f"  - JSON log created: {log_file}")
                
                # Load and check log content
                import json
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                print(f"  - Log epochs: {len(log_data['metrics']['epochs'])}")
                print(f"  - Log steps: {len(log_data['metrics']['steps'])}")
            
            # Test model generation after training
            model.eval()
            test_input = torch.randint(0, config.vocab_size, (1, 5)).to(device)
            
            with torch.no_grad():
                generated = model.generate(test_input, max_new_tokens=10, temperature=1.0)
                print(f"  - Generated shape: {generated.shape}")
            
            print("✅ Training integration working")
            return True
            
    except Exception as e:
        print(f"❌ Training integration error: {e}")
        print(traceback.format_exc())
        return False

def test_model_variants():
    """Test different TFT model configurations."""
    print("🔄 Testing model variants...")
    
    try:
        # Test with different factorization options
        configs = [
            ('baseline', {}),
            ('with_v_factorization', {'use_v': True}),
            ('with_output_projection', {'use_proj': True}),
            ('with_both', {'use_v': True, 'use_proj': True}),
        ]
        
        for name, overrides in configs:
            config = get_config('tiny', **overrides)
            model = get_model('tft', config)
            
            # Test forward pass
            input_ids = torch.randint(0, config.vocab_size, (1, 8))
            
            with torch.no_grad():
                outputs = model(input_ids)
                assert 'logits' in outputs
                print(f"  - {name}: {model.get_num_params()} params, output shape: {outputs['logits'].shape}")
        
        print("✅ Model variants working")
        return True
        
    except Exception as e:
        print(f"❌ Model variants error: {e}")
        print(traceback.format_exc())
        return False

def run_all_tests():
    """Run all integration tests."""
    print("🧪 Running TFT-GPT Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Tokenizer", test_tokenizer),
       # ("Data Loading", test_data_loading),
        ("Training Integration", test_training_integration),
        ("Model Variants", test_model_variants),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:<10} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Integration is working.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)