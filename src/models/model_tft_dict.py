# models/model_tft_alibi.py
"""
Factored Transformer model with Pre-Layer Normalization and ALiBi positional encoding.
This model incorporates separate token-like (xt) and embedding (xe) streams to represent
the typical internal transformer state vector (x = xt + xe).
- xt is updated by the attention mechanism, which serves as a symbolic manipulation
  operator on the token-like states -- thus providing a restricted symbolic reasoning
  process.
- xe is updated by the MLP, potentially introducing context not present in the original
  tokenized input prompt.
- ALiBi replaces traditional positional embeddings with linear biases in attention. This
  positional representation avoids modulating the token-like state with non-token-like
  information, nor similarly modulating the embedding-like states, thereby preserving
  the natural informational structure of the internal states.
- The factored representation provides a "dimensional analysis" accounting of the
  internal states -- a central concept in the analysis of anything we should try to
  measure (see e.g. P.W. Bridgman's "Dimensional Analysis").
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Any

# Import config from separate file
from config.model_configs import TFTConfig


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
        self.use_v = config.use_v
        self.use_proj = config.use_proj
        
        # Q, K projections (always needed)
        self.qk_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=config.bias)
        
        # Optional V factorization
        if self.use_v:
            self.v_factorization = nn.Parameter(torch.randn(config.n_heads, config.n_heads) * 0.02)
        
        # Optional output projection
        if self.use_proj:
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
        if self.use_v:
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
        if self.use_proj:
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
    """Fallback to your original sequential implementation."""
    
    def __init__(self, config: TFTConfig, shared_dict_embedding: nn.Embedding = None):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model, bias=config.bias)
        self.attention = ALiBiAttention(config)
        self.ln2 = LayerNorm(config.d_model, bias=config.bias)
        
        if config.use_dict_ffn and shared_dict_embedding is not None:
            self.mlp = MLP(config)
            self.dict_embedding = shared_dict_embedding
            self.dict_vocab_size = config.dict_vocab_size or config.vocab_size
            self.n_heads = config.n_heads
            self.d_head = config.d_model // config.n_heads
            
            # Back to individual LayerNorms per head
            self.head_layer_norms = nn.ModuleList([
                LayerNorm(self.d_head, bias=config.bias) for _ in range(self.n_heads)
            ])
            
            print("🔄 Fallback to sequential processing (original implementation)")
            
            self.use_dict_ffn = True
        else:
            self.mlp = MLP(config)
            self.use_dict_ffn = False
    
    def forward(self, xt: torch.Tensor, xe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        aux_outputs = {}
        
        # Attention updates xt
        norm_combined = self.ln1(xt + xe)
        attn_out = self.attention(norm_combined, xt)
        xt = xt + attn_out
        
        # MLP updates xe
        norm_combined = self.ln2(xt + xe)
        
        if self.use_dict_ffn:
            mlp_out = self.mlp(norm_combined)
            B, T, C = mlp_out.size()
            xe_new = torch.zeros_like(xe)
            total_dict_loss = 0.0
            
            # Back to sequential processing
            for h in range(self.n_heads):
                start_idx = h * self.d_head
                end_idx = (h + 1) * self.d_head
                
                h_head = mlp_out[:, :, start_idx:end_idx]
                h_head_norm = self.head_layer_norms[h](h_head)
                
                dict_emb_head = self.dict_embedding.weight[:self.dict_vocab_size, start_idx:end_idx]
                
                dict_logits = torch.matmul(h_head_norm, dict_emb_head.T)
                dict_weights = F.softmax(dict_logits, dim=-1)
                xe_head = torch.matmul(dict_weights, dict_emb_head)
                
                xe_new[:, :, start_idx:end_idx] = xe_head
                dict_loss = F.mse_loss(h_head_norm, xe_head)
                total_dict_loss += dict_loss
                
                # Memory cleanup after each head
                del h_head, h_head_norm, dict_emb_head, dict_logits, dict_weights, xe_head
                if h % 2 == 0:
                    torch.cuda.empty_cache()
            
            xe = xe + xe_new
            aux_outputs['dict_loss'] = total_dict_loss / self.n_heads
            
        else:
            mlp_out = self.mlp(norm_combined)
            xe = xe + mlp_out
        
        return xt, xe, aux_outputs

class TokenFactoredTransformerDict(nn.Module):
    """Token-Factored Transformer with ALiBi positional encoding."""
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (no positional embeddings with ALiBi)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TFTBlock(config, shared_dict_embedding=self.token_embedding) 
            for _ in range(config.n_layers)
        ])
        
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
    
    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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
        # Accumulate auxiliary outputs
        total_dict_loss = 0.0
        all_dict_weights = []

        # Pass through transformer blocks
        for block in self.blocks:
            xt, xe, aux_outputs = block(xt, xe)
            
            if 'dict_loss' in aux_outputs:
                total_dict_loss += aux_outputs['dict_loss']
                #all_dict_weights.append(aux_outputs['dict_weights'])
        
        # Final processing
        x_final = xt + xe
        x_final = self.ln_f(x_final)
        logits = self.lm_head(x_final)
        
        # Calculate loss if labels provided
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            lm_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add dictionary loss if using dictionary FFN
            if self.config.use_dict_ffn and total_dict_loss > 0:
                dict_loss_avg = total_dict_loss / self.config.n_layers
                loss = lm_loss + self.config.dict_loss_weight * dict_loss_avg
            else:
                loss = lm_loss

        outputs = {"logits": logits, "loss": loss}

        # Add auxiliary outputs for analysis
        if self.config.use_dict_ffn and all_dict_weights:
            outputs["dict_weights"] = torch.stack(all_dict_weights, dim=1)
            outputs["dict_loss"] = total_dict_loss / self.config.n_layers if total_dict_loss > 0 else 0.0

        return outputs
    
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