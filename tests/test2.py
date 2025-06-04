# tests/test2.py
"""
Comprehensive integration tests for TFT-GPT before running multi-GPU training.
Tests that all components work together: config, models, trainers, accelerate, etc.
"""

import os
import sys
import torch
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, 'src')

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class IntegrationTestSuite:
    """Comprehensive integration test suite."""
    
    def __init__(self):
        self.temp_dir = None
        self.test_results = []
        self.verbose = True
        
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="tft_test_")
        print(f"🏗️  Test directory: {self.temp_dir}")
        
    def cleanup(self):
        """Cleanup test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up: {self.temp_dir}")
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        status = "✅" if success else "❌"
        self.test_results.append((test_name, success, details))
        if self.verbose:
            print(f"  {status} {test_name}" + (f": {details}" if details else ""))
        return success
    
    def test_imports(self) -> bool:
        """Test all critical imports work."""
        print("\n🔍 Testing Critical Imports")
        print("-" * 40)
        
        imports_to_test = [
            # Models
            ("models.model_tft_alibi", "TokenFactoredTransformer"),
            ("models.model_vanilla", "VanillaTransformer"),
            ("models", "get_model"),
            
            # Config system
            ("config.model_configs", "TFTConfig"),
            ("config.model_configs", "get_config"),
            
            # Tokenizers
            ("mytokenizers", "create_tokenizer"),
            ("mytokenizers", "GPT2Tokenizer"),
            
            # Training
            ("trainers", "get_trainer"),
            ("trainers.simple_trainer", "SimpleTrainer"),
            ("trainers.accelerate_trainer", "AccelerateTrainer"),
            ("trainers.callbacks", "JSONLoggingCallback"),
            
            # Utils
            ("utils.data_utils", "load_and_prepare_data"),
            ("utils.json_logger", "JSONLogger"),
            ("utils.plotting", "plot_training_curves"),
            
            # Inference
            ("inference", "run_generation"),
        ]
        
        all_success = True
        for module_name, class_name in imports_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                self.log_test(f"Import {module_name}.{class_name}", True)
            except Exception as e:
                self.log_test(f"Import {module_name}.{class_name}", False, str(e))
                all_success = False
        
        return all_success
    
    def test_accelerate_availability(self) -> bool:
        """Test if accelerate is available and working."""
        print("\n🚀 Testing Accelerate Availability")
        print("-" * 40)
        
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            self.log_test("Accelerate import", True)
            self.log_test("Accelerator creation", True, f"Device: {accelerator.device}")
            self.log_test("CUDA availability", torch.cuda.is_available(), f"GPUs: {device_count}")
            
            return True
        except Exception as e:
            self.log_test("Accelerate setup", False, str(e))
            return False
    
    def test_config_system(self) -> bool:
        """Test configuration system."""
        print("\n⚙️ Testing Configuration System")
        print("-" * 40)
        
        try:
            from config.model_configs import get_config, CONFIG_PRESETS, TFTConfig
            
            # Test all presets
            for preset in ['tiny', 'small', 'medium']:
                config = get_config(preset)
                self.log_test(f"Config preset '{preset}'", True, 
                             f"{config.n_layers}L-{config.n_heads}H-{config.d_model}D")
            
            # Test custom overrides
            custom_config = get_config('tiny', n_layers=3, use_v=True, learning_rate=1e-3)
            success = (custom_config.n_layers == 3 and 
                      custom_config.use_v and 
                      custom_config.learning_rate == 1e-3)
            self.log_test("Config overrides", success)
            
            # Test config validation
            try:
                invalid_config = TFTConfig(d_model=100, n_heads=7)  # Not divisible
                self.log_test("Config validation", False, "Should have failed")
                return False
            except AssertionError:
                self.log_test("Config validation", True, "Correctly caught invalid config")
            
            return True
            
        except Exception as e:
            self.log_test("Config system", False, str(e))
            return False
    
    def test_model_creation(self) -> bool:
        """Test model creation and basic functionality."""
        print("\n🏗️ Testing Model Creation")
        print("-" * 40)
        
        try:
            from models import get_model
            from config.model_configs import get_config
            
            config = get_config('tiny')  # Small model for testing
            
            # Test both model types
            for model_type in ['tft', 'vanilla']:
                model = get_model(model_type, config)
                param_count = model.get_num_params()
                
                self.log_test(f"Create {model_type} model", True, f"{param_count:,} params")
                
                # Test forward pass
                batch_size, seq_len = 2, 8
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                
                model.eval()
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    
                self.log_test(f"{model_type} forward pass", True, 
                             f"Loss: {outputs['loss'].item():.4f}")
                
                # Test generation
                generated = model.generate(input_ids[:1], max_new_tokens=3, temperature=0.8)
                self.log_test(f"{model_type} generation", True, 
                             f"Shape: {generated.shape}")
            
            return True
            
        except Exception as e:
            self.log_test("Model creation", False, str(e))
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_tokenizer_system(self) -> bool:
        """Test tokenizer functionality."""
        print("\n🔤 Testing Tokenizer System")
        print("-" * 40)
        
        try:
            from mytokenizers import create_tokenizer, from_pretrained
            
            # Test tokenizer creation
            tokenizer = create_tokenizer('gpt2')
            self.log_test("Create GPT-2 tokenizer", True, f"Vocab size: {tokenizer.vocab_size}")
            
            # Test basic functionality
            test_text = "Hello, world! This is a test."
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            
            self.log_test("Tokenizer encode/decode", True, 
                         f"'{test_text}' -> {len(encoded)} tokens")
            
            # Test batch processing
            batch_texts = ["First text", "Second text", "Third text"]
            batch_output = tokenizer(batch_texts, padding=True, return_tensors='pt')
            
            expected_shape = (len(batch_texts), batch_output['input_ids'].shape[1])
            success = batch_output['input_ids'].shape == expected_shape
            self.log_test("Tokenizer batch processing", success, 
                         f"Shape: {batch_output['input_ids'].shape}")
            
            return True
            
        except Exception as e:
            self.log_test("Tokenizer system", False, str(e))
            return False
    
    def test_data_loading(self) -> bool:
        """Test data loading functionality."""
        print("\n📊 Testing Data Loading")
        print("-" * 40)
        
        try:
            from utils.data_utils import load_and_prepare_data
            from mytokenizers import create_tokenizer
            
            tokenizer = create_tokenizer('gpt2')
            
            # Test with TinyStories (small sample)
            dataloader, _ = load_and_prepare_data(
                dataset_name="roneneldan/TinyStories",
                dataset_config=None,
                tokenizer=tokenizer,
                max_samples=100,  # Very small for testing
                max_seq_length=32,
                batch_size=4,
                split='train'
            )
            
            self.log_test("Load TinyStories dataset", True, f"{len(dataloader)} batches")
            
            # Test batch iteration
            batch = next(iter(dataloader))
            expected_keys = {'input_ids', 'attention_mask', 'labels'}
            has_keys = expected_keys.issubset(batch.keys())
            
            self.log_test("Data batch format", has_keys, 
                         f"Keys: {list(batch.keys())}")
            
            self.log_test("Batch shapes", True, 
                         f"input_ids: {batch['input_ids'].shape}")
            
            return True
            
        except Exception as e:
            self.log_test("Data loading", False, str(e))
            return False
    
    def test_simple_trainer(self) -> bool:
        """Test SimpleTrainer with minimal setup."""
        print("\n🎓 Testing SimpleTrainer")
        print("-" * 40)
        
        try:
            from trainers import get_trainer
            from trainers.callbacks import JSONLoggingCallback
            from models import get_model
            from config.model_configs import get_config
            from mytokenizers import create_tokenizer
            from utils.data_utils import load_and_prepare_data
            
            # Setup minimal training
            config = get_config('tiny', block_size=16)
            model = get_model('tft', config)
            tokenizer = create_tokenizer('gpt2')
            config.vocab_size = tokenizer.vocab_size
            
            # Create minimal dataset
            dataloader, _ = load_and_prepare_data(
                dataset_name="roneneldan/TinyStories",
                dataset_config=None,
                tokenizer=tokenizer,
                max_samples=50,
                max_seq_length=16,
                batch_size=2,
                split='train'
            )
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            device = torch.device('cpu')  # Use CPU for testing
            
            # Setup callbacks
            test_output = os.path.join(self.temp_dir, "simple_trainer_test")
            callbacks = [JSONLoggingCallback(test_output, "test_run")]
            
            # Create trainer
            trainer = get_trainer(
                trainer_type='simple',
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                num_epochs=1,
                output_dir=test_output,
                callbacks=callbacks
            )
            
            self.log_test("Create SimpleTrainer", True)
            
            # Run training
            metrics = trainer.train()
            
            self.log_test("SimpleTrainer.train()", True, 
                         f"Final loss: {metrics.get('final_loss', 'N/A')}")
            
            # Check outputs
            log_file = os.path.join(test_output, "training_metrics.json")
            logs_exist = os.path.exists(log_file)
            self.log_test("Training logs created", logs_exist)
            
            if logs_exist:
                with open(log_file) as f:
                    log_data = json.load(f)
                has_epochs = len(log_data['metrics']['epochs']) > 0
                self.log_test("Training metrics logged", has_epochs)
            
            return True
            
        except Exception as e:
            self.log_test("SimpleTrainer", False, str(e))
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_accelerate_trainer(self) -> bool:
        """Test AccelerateTrainer with minimal setup."""
        print("\n🚀 Testing AccelerateTrainer")
        print("-" * 40)
        
        try:
            from trainers.accelerate_trainer import AccelerateTrainer
            from trainers.callbacks import JSONLoggingCallback
            from models import get_model
            from config.model_configs import get_config
            from mytokenizers import create_tokenizer
            from utils.data_utils import load_and_prepare_data
            
            # Setup minimal training
            config = get_config('tiny', block_size=16)
            model = get_model('tft', config)
            tokenizer = create_tokenizer('gpt2')
            config.vocab_size = tokenizer.vocab_size
            
            # Create minimal dataset
            dataloader, _ = load_and_prepare_data(
                dataset_name="roneneldan/TinyStories",
                dataset_config=None,
                tokenizer=tokenizer,
                max_samples=30,
                max_seq_length=16,
                batch_size=2,
                split='train'
            )
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Setup callbacks
            test_output = os.path.join(self.temp_dir, "accelerate_trainer_test")
            callbacks = [JSONLoggingCallback(test_output, "accelerate_test")]
            
            # Create AccelerateTrainer directly
            trainer = AccelerateTrainer(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                num_epochs=1,
                output_dir=test_output,
                callbacks=callbacks,
                mixed_precision="no",  # Disable for testing
                gradient_accumulation_steps=1
            )
            
            self.log_test("Create AccelerateTrainer", True, 
                         f"Device: {trainer.device}")
            
            # Run training
            metrics = trainer.train()
            
            self.log_test("AccelerateTrainer.train()", True, 
                         f"Final loss: {metrics.get('final_loss', 'N/A')}")
            
            # Check outputs
            log_file = os.path.join(test_output, "training_metrics.json")
            logs_exist = os.path.exists(log_file)
            self.log_test("Accelerate training logs", logs_exist)
            
            return True
            
        except Exception as e:
            self.log_test("AccelerateTrainer", False, str(e))
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_trainer_registry(self) -> bool:
        """Test trainer registry system."""
        print("\n📋 Testing Trainer Registry")
        print("-" * 40)
        
        try:
            from trainers import get_trainer, TRAINER_REGISTRY
            
            # Check available trainers
            available = list(TRAINER_REGISTRY.keys())
            self.log_test("Trainer registry", True, f"Available: {available}")
            
            expected_trainers = ['simple']
            for trainer_type in expected_trainers:
                if trainer_type in available:
                    self.log_test(f"Trainer '{trainer_type}' registered", True)
                else:
                    self.log_test(f"Trainer '{trainer_type}' registered", False)
            
            return True
            
        except Exception as e:
            self.log_test("Trainer registry", False, str(e))
            return False
    
    def test_generation_pipeline(self) -> bool:
        """Test complete generation pipeline."""
        print("\n🎨 Testing Generation Pipeline")
        print("-" * 40)
        
        try:
            from models import get_model
            from config.model_configs import get_config
            from mytokenizers import create_tokenizer
            from inference import run_generation
            
            # Setup
            config = get_config('tiny')
            model = get_model('tft', config)
            tokenizer = create_tokenizer('gpt2')
            device = torch.device('cpu')
            
            # Test generation
            prompt = "Once upon a time"
            generated_ids, generated_text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=device,
                max_new_tokens=10,
                temperature=0.8,
                show_progress=False
            )
            
            self.log_test("Generation pipeline", True, 
                         f"Generated {len(generated_ids)} tokens")
            
            self.log_test("Generated text quality", len(generated_text) > len(prompt),
                         f"'{generated_text[:50]}...'")
            
            return True
            
        except Exception as e:
            self.log_test("Generation pipeline", False, str(e))
            return False
    
    def test_plotting_system(self) -> bool:
        """Test plotting and logging utilities."""
        print("\n📊 Testing Plotting System")
        print("-" * 40)
        
        try:
            from utils.json_logger import JSONLogger
            from utils.plotting import plot_training_curves
            
            # Create fake training log
            log_file = os.path.join(self.temp_dir, "test_log.json")
            logger = JSONLogger(log_file, "test_run")
            
            # Log some fake data
            logger.log_config({"model": {"n_layers": 2}})
            
            for epoch in range(1, 4):
                logger.log_epoch(epoch, {"loss": 2.0 - epoch * 0.3})
                
            logger.finish({"final_loss": 1.1})
            
            self.log_test("JSON logging", True, f"Log: {log_file}")
            
            # Test plotting (will fail without display, but we check import)
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                
                plot_file = os.path.join(self.temp_dir, "test_plot.png")
                fig = plot_training_curves(log_file, plot_file, "Test Plot")
                
                plot_exists = os.path.exists(plot_file)
                self.log_test("Plot generation", plot_exists)
                
            except Exception as plot_e:
                self.log_test("Plot generation", False, f"Plot error: {plot_e}")
            
            return True
            
        except Exception as e:
            self.log_test("Plotting system", False, str(e))
            return False
    
    def test_full_integration(self) -> bool:
        """Test complete integration with all components."""
        print("\n🎯 Testing Full Integration")
        print("-" * 40)
        
        try:
            # This is a mini version of the training script
            from models import get_model
            from config.model_configs import get_config
            from mytokenizers import create_tokenizer
            from utils.data_utils import load_and_prepare_data
            from trainers import get_trainer
            from trainers.callbacks import JSONLoggingCallback
            
            # Configuration
            config = get_config('tiny', block_size=16, learning_rate=1e-3)
            
            # Tokenizer
            tokenizer = create_tokenizer('gpt2')
            config.vocab_size = tokenizer.vocab_size
            
            # Data
            dataloader, _ = load_and_prepare_data(
                dataset_name="roneneldan/TinyStories",
                dataset_config=None,
                tokenizer=tokenizer,
                max_samples=40,
                max_seq_length=16,
                batch_size=2,
                split='train'
            )
            
            # Model
            model = get_model('tft', config)
            
            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01
            )
            
            # Training setup
            device = torch.device('cpu')
            output_dir = os.path.join(self.temp_dir, "full_integration_test")
            
            callbacks = [
                JSONLoggingCallback(
                    output_dir=output_dir,
                    run_name="integration_test",
                    log_every_n_steps=5
                )
            ]
            
            # Trainer
            trainer = get_trainer(
                trainer_type='simple',
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                num_epochs=1,
                output_dir=output_dir,
                callbacks=callbacks
            )
            
            # Train
            metrics = trainer.train()
            
            self.log_test("Full integration training", True, 
                         f"Loss: {metrics.get('final_loss', 'N/A')}")
            
            # Test generation
            from inference import run_generation
            
            generated_ids, generated_text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text="Hello",
                device=device,
                max_new_tokens=5,
                show_progress=False
            )
            
            self.log_test("Full integration generation", True,
                         f"Generated: '{generated_text}'")
            
            # Check all outputs exist
            expected_files = [
                "training_metrics.json",
                "checkpoint_epoch_1.pt"
            ]
            
            for filename in expected_files:
                file_path = os.path.join(output_dir, filename)
                exists = os.path.exists(file_path)
                self.log_test(f"Output file {filename}", exists)
            
            return True
            
        except Exception as e:
            self.log_test("Full integration", False, str(e))
            import traceback
            if self.verbose:
                traceback.print_exc()
            return False
    
    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🧪 TFT-GPT INTEGRATION TEST SUITE")
        print("=" * 50)
        
        self.setup()
        
        try:
            test_methods = [
                self.test_imports,
                self.test_accelerate_availability,
                self.test_config_system,
                self.test_tokenizer_system,
                self.test_model_creation,
                self.test_data_loading,
                self.test_trainer_registry,
                self.test_simple_trainer,
                self.test_accelerate_trainer,
                self.test_generation_pipeline,
                self.test_plotting_system,
                self.test_full_integration,
            ]
            
            for test_method in test_methods:
                try:
                    test_method()
                except Exception as e:
                    print(f"❌ Test {test_method.__name__} crashed: {e}")
                    self.test_results.append((test_method.__name__, False, str(e)))
            
            # Summary
            self.print_summary()
            
            # Return overall success
            passed = sum(1 for _, success, _ in self.test_results if success)
            total = len(self.test_results)
            
            return passed >= total * 0.8  # 80% pass rate
            
        finally:
            self.cleanup()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📋 INTEGRATION TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"\nTests passed: {passed}/{total} ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Ready for multi-GPU training")
        elif passed >= total * 0.9:
            print("\n✅ EXCELLENT! Almost all tests passed")
            print("✅ Safe to proceed with multi-GPU training")
        elif passed >= total * 0.8:
            print("\n⚠️ GOOD - Most tests passed")
            print("✅ Probably safe for multi-GPU training")
        else:
            print("\n❌ ISSUES FOUND")
            print("⚠️ Fix issues before multi-GPU training")
        
        # Show failed tests
        failed_tests = [(name, details) for name, success, details in self.test_results if not success]
        if failed_tests:
            print(f"\n❌ Failed tests ({len(failed_tests)}):")
            for name, details in failed_tests:
                print(f"   • {name}: {details}")
        
        print(f"\n💡 Next steps:")
        if passed >= total * 0.8:
            print("   1. ✅ Run single-GPU training test:")
            print("      python exptrain_tft.py --preset tiny --epochs 1 --max_samples 100")
            print("   2. ✅ Run multi-GPU comparison:")
            print("      bash exprun_train_compare.sh 2 small 3")
        else:
            print("   1. ❌ Fix failed tests first")
            print("   2. ❌ Re-run integration tests")
            print("   3. ✅ Then proceed with multi-GPU testing")


def main():
    """Run integration tests."""
    suite = IntegrationTestSuite()
    success = suite.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)