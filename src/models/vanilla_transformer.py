# src/models/vanilla_transformer.py
"""
Vanilla Transformer baseline model for comparison with TFT.
Uses the same config system and interface as TFT.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Any

# Import your existing config
from config.model_configs import TFTConfig


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (same as TFT implementation)."""
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Standard multi-head self-attention with causal mask."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Key, Query, Value projections for all heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Register causal mask
        self.register_buffer(
            "causal_mask", 
            torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                  .view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Calculate Q, K, V for all heads
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        y = attn_weights @ v  # (B, n_heads, T, head_dim)
        
        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.output_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network (same as TFT implementation)."""
    
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


class TransformerBlock(nn.Module):
    """Standard transformer block with Pre-Layer Normalization."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN architecture
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VanillaTransformer(nn.Module):
    """
    Vanilla Transformer baseline model.
    Uses same config and interface as TFT for fair comparison.
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
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
        elif isinstance(module, CausalSelfAttention):
            # Special initialization for output projections
            torch.nn.init.normal_(module.output_proj.weight, mean=0.0, 
                                 std=0.02/math.sqrt(2 * self.config.n_layers))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass (same interface as TFT)."""
        B, T = input_ids.size()
        
        # Check sequence length
        if T > self.config.max_position_embeddings:
            raise ValueError(f"Sequence length {T} exceeds max_position_embeddings {self.config.max_position_embeddings}")
        
        # Create position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        
        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(token_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final processing
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
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
        """Generate new tokens autoregressively (same interface as TFT)."""
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
        """Get number of parameters (same interface as TFT)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)