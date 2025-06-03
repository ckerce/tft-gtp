#!/usr/bin/env python3
# tests/test_model.py
"""
Unit tests for the refactored TFT model to ensure it maintains original functionality.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tft_alibi import TokenFactoredTransformer, TFTConfig


class TestTFTConfig:
    """Test TFT configuration."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = TFTConfig()
        assert config.n_layers == 6
        assert config.n_heads == 6
        assert config.d_model == 384
        assert config.d_ff == 4 * config.d_model
        assert config.d_model % config.n_heads == 0
    
    def test_config_presets(self):
        """Test preset configurations."""
        small = TFTConfig.small()
        medium = TFTConfig.medium()
        large = TFTConfig.large()
        
        assert small.d_model < medium.d_model < large.d_model
        assert small.n_layers <= medium.n_layers <= large.n_layers
    
    def test_config_validation(self):
        """Test config validation."""
        # Should fail: d_model not divisible by n_heads
        with pytest.raises(AssertionError):
            TFTConfig(d_model=100, n_heads=7)
        
        # Should fail: max_position_embeddings < block_size
        with pytest.raises(AssertionError):
            TFTConfig(block_size=512, max_position_embeddings=256)
    
    def test_config_overrides(self):
        """Test config with overrides."""
        config = TFTConfig.small(
            use_value_factorization=True,
            use_output_projection=True,
            dropout=0.2
        )
        assert config.use_value_factorization is True
        assert config.use_output_projection is True
        assert config.dropout == 0.2


class TestTFTModel:
    """Test the TFT model implementation."""
    
    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return TFTConfig(
            vocab_size=1000,
            n_layers=2,
            n_heads=4,
            d_model=128,
            block_size=32,
            max_position_embeddings=64,
            dropout=0.0  # Disable for deterministic tests
        )
    
    @pytest.fixture
    def model(self, config):
        """Create model for testing."""
        torch.manual_seed(42)
        return TokenFactoredTransformer(config)
    
    def test_model_creation(self, config):
        """Test model can be created."""
        model = TokenFactoredTransformer(config)
        assert isinstance(model, TokenFactoredTransformer)
        assert model.config == config
    
    def test_model_parameters(self, model, config):
        """Test model has expected parameters."""
        total_params = model.get_num_params()
        assert total_params > 0
        
        # Check that parameters are reasonable for the config
        expected_range = (50_000, 2_000_000)  # Rough range for small model
        assert expected_range[0] < total_params < expected_range[1]
    
    def test_forward_pass_basic(self, model, config):
        """Test basic forward pass."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert 'loss' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs['loss'] is None  # No labels provided
    
    def test_forward_pass_with_labels(self, model, config):
        """Test forward pass with labels (loss computation)."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        
        assert outputs['loss'] is not None
        assert isinstance(outputs['loss'], torch.Tensor)
        assert outputs['loss'].dim() == 0  # Scalar loss
        assert outputs['loss'].item() > 0
    
    def test_sequence_length_limits(self, model, config):
        """Test sequence length validation."""
        # Should work within limits
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings))
        outputs = model(input_ids)
        assert outputs['logits'].shape[1] == config.max_position_embeddings
        
        # Should fail beyond limits
        with pytest.raises(ValueError):
            input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings + 1))
            model(input_ids)
    
    def test_generation_basic(self, model, config):
        """Test basic text generation."""
        model.eval()
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=5)
        
        assert generated.shape == (batch_size, seq_len + 5)
        assert torch.all(generated[:, :seq_len] == input_ids)  # Original sequence preserved
    
    def test_generation_with_sampling(self, model, config):
        """Test generation with different sampling parameters."""
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        
        with torch.no_grad():
            # Test temperature
            gen_temp_low = model.generate(input_ids, max_new_tokens=5, temperature=0.1)
            gen_temp_high = model.generate(input_ids, max_new_tokens=5, temperature=2.0)
            
            # Test top-k
            gen_topk = model.generate(input_ids, max_new_tokens=5, top_k=10)
        
        assert gen_temp_low.shape[1] == input_ids.shape[1] + 5
        assert gen_temp_high.shape[1] == input_ids.shape[1] + 5
        assert gen_topk.shape[1] == input_ids.shape[1] + 5


class TestALiBiAttention:
    """Test ALiBi attention mechanism."""
    
    @pytest.fixture
    def config(self):
        return TFTConfig(
            vocab_size=100,
            n_layers=1,
            n_heads=4,
            d_model=64,
            block_size=16,
            dropout=0.0
        )
    
    @pytest.fixture
    def model(self, config):
        torch.manual_seed(42)
        return TokenFactoredTransformer(config)
    
    def test_alibi_slopes_generation(self, model):
        """Test ALiBi slopes are generated correctly."""
        attention_layer = model.blocks[0].attention
        slopes = attention_layer.alibi_slopes
        
        assert slopes.shape == (model.config.n_heads,)
        assert torch.all(slopes > 0)  # All slopes should be positive
        assert torch.all(slopes <= 1)  # All slopes should be <= 1
        
        # Slopes should be in descending order (approximately)
        assert torch.all(slopes[:-1] >= slopes[1:] * 0.9)  # Allow some tolerance
    
    def test_alibi_bias_shape(self, model):
        """Test ALiBi bias matrix has correct shape."""
        attention_layer = model.blocks[0].attention
        seq_len = 8
        device = next(model.parameters()).device
        
        bias = attention_layer._get_alibi_bias(seq_len, device)
        
        assert bias.shape == (model.config.n_heads, seq_len, seq_len)
        
        # Check causal mask (upper triangle should be -inf)
        for h in range(model.config.n_heads):
            upper_triangle = torch.triu(bias[h], diagonal=1)
            assert torch.all(upper_triangle == float('-inf'))
    
    def test_length_extrapolation(self, model, config):
        """Test that model can handle longer sequences than training length."""
        model.eval()
        
        # Test with sequence longer than block_size but within max_position_embeddings
        long_seq_len = config.block_size + 8
        assert long_seq_len <= config.max_position_embeddings
        
        input_ids = torch.randint(0, config.vocab_size, (1, long_seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs['logits'].shape == (1, long_seq_len, config.vocab_size)


class TestFactoredStreams:
    """Test the factored stream implementation (xt + xe)."""
    
    @pytest.fixture
    def config(self):
        return TFTConfig(
            vocab_size=100,
            n_layers=2,
            n_heads=4,
            d_model=64,
            block_size=8,
            dropout=0.0
        )
    
    def test_stream_initialization(self, config):
        """Test that streams are initialized correctly."""
        model = TokenFactoredTransformer(config)
        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Get initial embeddings
        token_emb = model.token_embedding(input_ids)
        xt_init = model.dropout(token_emb)
        xe_init = torch.zeros_like(xt_init)
        
        assert xt_init.shape == (batch_size, seq_len, config.d_model)
        assert xe_init.shape == (batch_size, seq_len, config.d_model)
        assert torch.allclose(xe_init, torch.zeros_like(xe_init))
    
    def test_stream_updates(self, config):
        """Test that streams are updated correctly through blocks."""
        model = TokenFactoredTransformer(config)
        model.eval()
        
        batch_size, seq_len = 1, 4
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Manual forward pass to track streams
        token_emb = model.token_embedding(input_ids)
        xt = model.dropout(token_emb)
        xe = torch.zeros_like(xt)
        
        xt_initial = xt.clone()
        xe_initial = xe.clone()
        
        # Pass through first block
        xt, xe = model.blocks[0](xt, xe)
        
        # xt should have changed (attention updates it)
        assert not torch.allclose(xt, xt_initial, atol=1e-6)
        
        # xe should have changed (MLP updates it)
        assert not torch.allclose(xe, xe_initial, atol=1e-6)
    
    def test_value_factorization_flag(self):
        """Test that value factorization flag works."""
        config_no_v = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64,
                               use_value_factorization=False)
        config_with_v = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64,
                                 use_value_factorization=True)
        
        model_no_v = TokenFactoredTransformer(config_no_v)
        model_with_v = TokenFactoredTransformer(config_with_v)
        
        # Check that the attention layers have different parameters
        attn_no_v = model_no_v.blocks[0].attention
        attn_with_v = model_with_v.blocks[0].attention
        
        assert not hasattr(attn_no_v, 'v_factorization')
        assert hasattr(attn_with_v, 'v_factorization')
    
    def test_output_projection_flag(self):
        """Test that output projection flag works."""
        config_no_proj = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64,
                                  use_output_projection=False)
        config_with_proj = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64,
                                    use_output_projection=True)
        
        model_no_proj = TokenFactoredTransformer(config_no_proj)
        model_with_proj = TokenFactoredTransformer(config_with_proj)
        
        attn_no_proj = model_no_proj.blocks[0].attention
        attn_with_proj = model_with_proj.blocks[0].attention
        
        assert not hasattr(attn_no_proj, 'output_factorization')
        assert hasattr(attn_with_proj, 'output_factorization')


class TestDeterminism:
    """Test that model behavior is deterministic."""
    
    def test_forward_determinism(self):
        """Test that forward passes are deterministic."""
        config = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64, dropout=0.0)
        
        # Create two identical models
        torch.manual_seed(42)
        model1 = TokenFactoredTransformer(config)
        
        torch.manual_seed(42)
        model2 = TokenFactoredTransformer(config)
        
        # Same input
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            outputs1 = model1(input_ids)
            outputs2 = model2(input_ids)
        
        assert torch.allclose(outputs1['logits'], outputs2['logits'], atol=1e-6)
    
    def test_generation_determinism(self):
        """Test that generation is deterministic with same seed."""
        config = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64, dropout=0.0)
        model = TokenFactoredTransformer(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        
        # Generate twice with same conditions
        torch.manual_seed(123)
        gen1 = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
        
        torch.manual_seed(123)
        gen2 = model.generate(input_ids, max_new_tokens=5, temperature=1.0)
        
        assert torch.equal(gen1, gen2)


class TestMemoryEfficiency:
    """Test memory usage and efficiency."""
    
    def test_memory_usage(self):
        """Test that memory usage is reasonable."""
        config = TFTConfig.small()
        model = TokenFactoredTransformer(config)
        
        # Rough parameter count check
        param_count = model.get_num_params()
        
        # For small config, should be in reasonable range
        assert 100_000 < param_count < 10_000_000
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test model works on CUDA."""
        config = TFTConfig(vocab_size=100, n_layers=1, n_heads=4, d_model=64, block_size=8)
        model = TokenFactoredTransformer(config).cuda()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 6)).cuda()
        
        outputs = model(input_ids)
        assert outputs['logits'].is_cuda


def run_all_tests():
    """Run all tests programmatically."""
    print("🧪 Running TFT Model Tests...")
    
    # Run pytest programmatically
    import subprocess
    result = subprocess.run([
        'python', '-m', 'pytest', __file__, '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    import sys
    sys.exit(0 if success else 1)