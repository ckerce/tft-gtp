# models/tft_alibi.py
"""
Token-Factored Transformer with ALiBi positional encoding.
Clean implementation focusing on the core TFT concept.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class TFTConfig:
    """Configuration for Token-Factored Transformer with ALiBi."""
    
    # Model architecture
    vocab_size: int = 50257
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = None  # Will be set to 4 * d_model if None
    dropout: float = 0.1
    bias: bool = False
    
    # TFT-specific parameters
    use_value_factorization: bool = False
    use_output_projection: bool = False
    
    # ALiBi parameters
    block_size: int = 128  # Training sequence length
    max_position_embeddings: int = 512  # Max inference length
    
    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.max_position_embeddings >= self.block_size, "max_position_embeddings must be >= block_size"
    
    @classmethod
    def small(cls, **kwargs):
        defaults = dict(n_layers=6, n_heads=6, d_model=384, block_size=128)
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def medium(cls, **kwargs):
        defaults = dict(n_layers=12, n_heads=12, d_model=768, block_size=256)
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def large(cls, **kwargs):
        defaults = dict(n_layers=24, n_heads=16, d_model=1024, block_size=512)
        defaults.update(kwargs)
        return cls(**defaults)


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class ALiBiAttention(nn.Module):
    """Factored Causal Self-Attention with ALiBi positional encoding."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.use_value_factorization = config.use_value_factorization
        self.use_output_projection = config.use_output_projection
        
        # Q, K projections (always needed)
        self.qk_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=config.bias)
        
        # Optional V factorization
        if self.use_value_factorization:
            self.v_factorization = nn.Parameter(torch.randn(config.n_heads, config.n_heads) * 0.02)
        
        # Optional output projection
        if self.use_output_projection:
            self.output_factorization = nn.Parameter(torch.randn(config.n_heads, config.n_heads) * 0.02)
            self.resid_dropout = nn.Dropout(config.dropout)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        
        # ALiBi slopes
        slopes = self._get_alibi_slopes(config.n_heads)
        self.register_buffer("alibi_slopes", slopes, persistent=False)
    
    def _get_alibi_slopes(self, n_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes for each attention head."""
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if (n_heads & (n_heads - 1)) == 0:  # Power of 2
            slopes = get_slopes_power_of_2(n_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            if n_heads > closest_power_of_2:
                extra_base = 2**(-(2**-(math.log2(2*closest_power_of_2)-3)))
                num_remaining = n_heads - closest_power_of_2
                extra_slopes = [extra_base * (extra_base**i) for i in range(num_remaining)]
                slopes.extend(extra_slopes)
        
        return torch.tensor(slopes[:n_heads], dtype=torch.float32)
    
    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate ALiBi bias matrix."""
        # Create relative position matrix
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        relative_pos = positions[None, :] - positions[:, None]
        
        # Apply slopes and causal mask
        alibi_bias = self.alibi_slopes[:, None, None] * relative_pos[None, :, :]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float('-inf'))
        
        return alibi_bias
    
    def _kronecker_lift(self, matrix: torch.Tensor) -> torch.Tensor:
        """Lift n_heads x n_heads matrix to d_model x d_model block-diagonal."""
        n_heads, _ = matrix.shape
        head_dim = self.d_model // n_heads
        
        lifted = torch.zeros(self.d_model, self.d_model, device=matrix.device, dtype=matrix.dtype)
        for i in range(n_heads):
            for j in range(n_heads):
                start_i, end_i = i * head_dim, (i + 1) * head_dim
                start_j, end_j = j * head_dim, (j + 1) * head_dim
                lifted[start_i:end_i, start_j:end_j] = matrix[i, j] * torch.eye(head_dim, device=matrix.device, dtype=matrix.dtype)
        
        return lifted
    
    def forward(self, x_norm: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for factored attention.
        
        Args:
            x_norm: Normalized combined state (xt + xe) for Q,K computation
            xt: Token stream for values
        """
        B, T, C = x_norm.size()
        
        # Compute Q, K from normalized combined state
        q, k = self.qk_proj(x_norm).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Handle values based on factorization setting
        if self.use_value_factorization:
            v_matrix = self._kronecker_lift(self.v_factorization)
            v_flat = torch.matmul(xt.view(-1, C), v_matrix.t())
            v = v_flat.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            v = xt.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        
        # Add ALiBi bias
        if T > 1:
            alibi_bias = self._get_alibi_bias(T, x_norm.device)
            attn_scores = attn_scores + alibi_bias[None, :, :, :]
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        y = attn_weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Optional output projection
        if self.use_output_projection:
            proj_matrix = self._kronecker_lift(self.output_factorization)
            y_flat = torch.matmul(y.view(-1, C), proj_matrix.t())
            y = y_flat.view(B, T, C)
            y = self.resid_dropout(y)
        
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.activation(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TFTBlock(nn.Module):
    """Single Token-Factored Transformer block with Pre-LN."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        self.attention = ALiBiAttention(config)
        self.ln2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, xt: torch.Tensor, xe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for TFT block.
        
        Args:
            xt: Token stream
            xe: Embedding stream
            
        Returns:
            Updated (xt, xe) streams
        """
        # Attention updates xt using combined state for Q,K but xt for V
        norm_combined = self.ln1(xt + xe)
        attn_out = self.attention(norm_combined, xt)
        xt = xt + attn_out
        
        # MLP updates xe using combined state
        norm_combined = self.ln2(xt + xe)
        mlp_out = self.mlp(norm_combined)
        xe = xe + mlp_out
        
        return xt, xe


class TokenFactoredTransformer(nn.Module):
    """Token-Factored Transformer with ALiBi positional encoding."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (no positional embeddings with ALiBi)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TFTBlock(config) for _ in range(config.n_layers)])
        
        # Final layer norm and output head
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, ALiBiAttention):
            if hasattr(module, 'v_factorization'):
                torch.nn.init.normal_(module.v_factorization, mean=0.0, std=0.02)
            if hasattr(module, 'output_factorization'):
                torch.nn.init.normal_(module.output_factorization, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, T = input_ids.size()
        
        # Check sequence length
        if T > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {T} exceeds max_position_embeddings {self.config.max_position_embeddings}")
        
        # Token embeddings only (no positional with ALiBi)
        token_emb = self.token_embedding(input_ids)
        
        # Initialize factored streams
        xt = self.dropout(token_emb)  # Token stream
        xe = torch.zeros_like(xt)     # Embedding stream
        
        # Pass through transformer blocks
        for block in self.blocks:
            xt, xe = block(xt, xe)
        
        # Final processing
        x_final = xt + xe
        x_final = self.ln_f(x_final)
        logits = self.lm_head(x_final)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate new tokens autoregressively."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if too long
            context = input_ids if input_ids.size(1) <= self.config.max_position_embeddings else input_ids[:, -self.config.max_position_embeddings:]
            
            # Forward pass
            outputs = self(context)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        self.train()
        return input_ids
    
    def get_num_params(self) -> int:
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> 'TokenFactoredTransformer':
        """Create model from config dictionary."""
        config = TFTConfig(**config_dict)
        return cls(config)