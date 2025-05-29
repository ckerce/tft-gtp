import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


# Global configuration
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"


@dataclass
class TFTDeepSeekConfig:
    """
    Configuration for Token-Factored DeepSeek model with ALiBi positional encoding.
    
    Key differences from standard DeepSeek:
    - No RoPE (replaced with ALiBi)
    - V pathway uses Kronecker-lifted projections from pure token stream
    - Maintains Q/K compression for efficiency where interpretability not needed
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    
    # MoE configuration
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    
    # TFT-compatible attention (no RoPE)
    q_lora_rank: int = 0  # Can use compression for Q
    kv_lora_rank: int = 512  # Can use compression for K
    qk_nope_head_dim: int = 128  # No positional component needed
    v_head_dim: int = 128
    
    # ALiBi configuration
    alibi_bias: bool = True
    dropout: float = 0.1
    
    # Training stability
    gradient_checkpointing: bool = False
    use_cache: bool = True


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Linear transformation with support for quantized weights.
    Simplified version focusing on bf16 operations for TFT implementation.
    """
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    """
    Custom linear layer supporting the TFT-DeepSeek architecture.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype or Linear.dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Linear layer with column parallelism for distributed training."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        if world_size > 1:
            assert out_features % world_size == 0
            part_out_features = out_features // world_size
        else:
            part_out_features = out_features
        super().__init__(in_features, part_out_features, bias, dtype)


class RowParallelLinear(Linear):
    """Linear layer with row parallelism for distributed training."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        if world_size > 1:
            assert in_features % world_size == 0
            part_in_features = in_features // world_size
        else:
            part_in_features = in_features
        super().__init__(part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class TFTDeepSeekAttention(nn.Module):
    """
    Token-Factored DeepSeek Multi-Head Attention with ALiBi positional encoding.
    
    Key innovations:
    - Q/K can use compressed representations (interpretability not required)
    - V must come from pure token stream via Kronecker-lifted projections
    - ALiBi replaces RoPE for position encoding without embedding contamination
    """
    def __init__(self, config: TFTDeepSeekConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_local_heads = config.n_heads // world_size if world_size > 1 else config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_nope_head_dim  # No RoPE component
        
        # Q projection (can be compressed for efficiency)
        if config.q_lora_rank > 0:
            self.wq_a = Linear(config.dim, config.q_lora_rank)
            self.q_norm = RMSNorm(config.q_lora_rank)
            self.wq_b = ColumnParallelLinear(config.q_lora_rank, self.n_local_heads * self.qk_head_dim)
        else:
            self.wq = ColumnParallelLinear(config.dim, self.n_local_heads * self.qk_head_dim)
        
        # K projection (can be compressed - interpretability not needed for attention scores)
        self.wk_a = Linear(config.dim, config.kv_lora_rank)
        self.k_norm = RMSNorm(config.kv_lora_rank)
        self.wk_b = ColumnParallelLinear(config.kv_lora_rank, self.n_local_heads * self.qk_head_dim)
        
        # V projection - CRITICAL: Must preserve token interpretability
        # Use Kronecker-lifted structure for efficient cross-head interactions
        self.v_kronecker = nn.Parameter(torch.randn(self.n_local_heads, self.n_local_heads) * 0.02)
        
        # Output projection
        self.wo = RowParallelLinear(self.n_local_heads * self.v_head_dim, config.dim)
        
        # ALiBi slopes for positional encoding
        slopes = self._get_alibi_slopes(self.n_local_heads)
        self.register_buffer("alibi_slopes", slopes, persistent=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling for attention scores
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)
        
        # Cache for inference (optional)
        if config.use_cache:
            self.register_buffer("k_cache", torch.zeros(
                config.max_batch_size, config.max_seq_len, self.n_local_heads, self.qk_head_dim
            ), persistent=False)
            self.register_buffer("v_cache", torch.zeros(
                config.max_batch_size, config.max_seq_len, self.n_local_heads, self.v_head_dim
            ), persistent=False)

    def _get_alibi_slopes(self, n_heads):
        """Compute ALiBi slopes for positional bias."""
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        def get_slopes(n):
            if n <= 0:
                return []
            if (n & (n - 1)) == 0:  # Power of 2
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                if n > closest_power_of_2:
                    extra_base = 2**(-(2**-(math.log2(2*closest_power_of_2)-3)))
                    num_remaining = n - closest_power_of_2
                    extra_slopes = [extra_base * (extra_base**i) for i in range(num_remaining)]
                    slopes.extend(extra_slopes)
                return slopes[:n]

        slopes = get_slopes(n_heads)
        return torch.tensor(slopes, dtype=torch.float32)

    def _get_alibi_bias(self, seq_len, device):
        """Generate ALiBi bias matrix."""
        # Create relative position matrix
        context_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        memory_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        relative_position = memory_position[None, :] - context_position[:, None]
        
        # Apply slopes and causal masking
        alibi_bias = self.alibi_slopes[:, None, None] * relative_position[None, :, :]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float('-inf'))
        
        return alibi_bias

    def _kronecker_lift(self, small_matrix):
        """
        Lift H×H matrix to create structured V projection that preserves token semantics.
        Maps from full embedding dimension to V head dimensions with cross-head interactions.
        """
        n_heads = small_matrix.shape[0]
        input_head_dim = self.dim // n_heads
        output_head_dim = self.v_head_dim
        
        # Create structured projection matrix
        lifted = torch.zeros(
            self.dim, n_heads * output_head_dim, 
            device=small_matrix.device, dtype=small_matrix.dtype
        )
        
        for i in range(n_heads):
            for j in range(n_heads):
                # Map input head i to output head j with cross-head scaling
                input_start = i * input_head_dim
                input_end = (i + 1) * input_head_dim
                output_start = j * output_head_dim
                output_end = (j + 1) * output_head_dim
                
                # Create block preserving token structure
                min_dim = min(input_head_dim, output_head_dim)
                lifted[input_start:input_start+min_dim, 
                       output_start:output_start+min_dim] = (
                    small_matrix[i, j] * torch.eye(min_dim, device=small_matrix.device, dtype=small_matrix.dtype)
                )
        
        return lifted

    def forward(self, x_combined, xt_pure, start_pos: int = 0, use_cache: bool = False):
        """
        Forward pass for TFT-DeepSeek attention.
        
        Args:
            x_combined: Combined state (XT + XE) for Q/K computation
            xt_pure: Pure token stream for V computation (preserves interpretability)
            start_pos: Starting position for cached inference
            use_cache: Whether to use/update cache
        """
        bsz, seqlen, _ = x_combined.size()
        end_pos = start_pos + seqlen
        
        # Q computation (can use compressed representation)
        if hasattr(self, 'wq'):
            q = self.wq(x_combined)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x_combined)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        
        # K computation (can use compressed representation)
        k_compressed = self.k_norm(self.wk_a(x_combined))
        k = self.wk_b(k_compressed)
        k = k.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        
        # V computation - CRITICAL: Must come from pure token stream
        v_matrix = self._kronecker_lift(self.v_kronecker)
        v_flat = xt_pure.view(-1, xt_pure.size(-1))
        v_projected = torch.matmul(v_flat, v_matrix.t())
        v = v_projected.view(bsz, seqlen, self.n_local_heads, self.v_head_dim)
        
        # Update cache if using incremental generation
        if use_cache and hasattr(self, 'k_cache'):
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            k = self.k_cache[:bsz, :end_pos]
            v = self.v_cache[:bsz, :end_pos]
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add ALiBi positional bias
        if seqlen > 1 or (use_cache and end_pos > 1):
            target_len = k.size(2)  # Use actual sequence length including cache
            alibi_bias = self._get_alibi_bias(target_len, x_combined.device)
            if use_cache and start_pos > 0:
                # For incremental generation, only apply bias to relevant positions
                alibi_bias = alibi_bias[:, -seqlen:, :]
            scores = scores + alibi_bias
        
        # Apply attention weights
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x_combined)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        output = torch.matmul(attn_weights, v)  # (bsz, n_heads, seqlen, v_head_dim)
        output = output.transpose(1, 2).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, self.n_local_heads * self.v_head_dim)
        
        # Final output projection
        return self.wo(output)


class TFTDeepSeekMLP(nn.Module):
    """Standard MLP for dense layers in TFT-DeepSeek."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TFTDeepSeekGate(nn.Module):
    """Gating mechanism for MoE routing in TFT-DeepSeek."""
    def __init__(self, config: TFTDeepSeekConfig):
        super().__init__()
        self.dim = config.dim
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        # Optional bias for specific configurations
        self.bias = nn.Parameter(torch.empty(config.n_routed_experts)) if config.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
            
        # Group-based routing if configured
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # Select top-k experts
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        
        return weights.type_as(x), indices


class TFTDeepSeekExpert(nn.Module):
    """Individual expert in MoE layer."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TFTDeepSeekMoE(nn.Module):
    """
    Mixture of Experts layer for TFT-DeepSeek.
    Only updates the contextual stream (XE) to maintain stream separation.
    """
    def __init__(self, config: TFTDeepSeekConfig):
        super().__init__()
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        if world_size > 1:
            assert config.n_routed_experts % world_size == 0
            self.n_local_experts = config.n_routed_experts // world_size
            self.experts_start_idx = rank * self.n_local_experts
            self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        else:
            self.n_local_experts = config.n_routed_experts
            self.experts_start_idx = 0
            self.experts_end_idx = config.n_routed_experts
        
        self.n_activated_experts = config.n_activated_experts
        
        # Gating mechanism
        self.gate = TFTDeepSeekGate(config)
        
        # Expert networks
        self.experts = nn.ModuleList([
            TFTDeepSeekExpert(config.dim, config.moe_inter_dim) 
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        
        # Shared experts
        self.shared_experts = TFTDeepSeekMLP(config.dim, config.n_shared_experts * config.moe_inter_dim)

    def forward(self, x_combined: torch.Tensor, xe_current: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE layer.
        
        Args:
            x_combined: Combined state for routing decisions
            xe_current: Current contextual stream to be updated
            
        Returns:
            Updated contextual stream
        """
        shape = x_combined.size()
        x_flat = x_combined.view(-1, self.dim)
        
        # Route to experts based on combined state
        weights, indices = self.gate(x_flat)
        
        # Compute expert outputs
        expert_output = torch.zeros_like(x_flat)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0 or self.experts[i] is None:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if len(idx) > 0:
                expert_output[idx] += expert(x_flat[idx]) * weights[idx, top, None]
        
        # Add shared expert output
        shared_output = self.shared_experts(x_flat)
        total_output = expert_output + shared_output
        
        # All-reduce for distributed training
        if world_size > 1:
            dist.all_reduce(total_output)
        
        # Only update contextual stream
        return xe_current + total_output.view(shape)


class TFTDeepSeekBlock(nn.Module):
    """
    Token-Factored DeepSeek transformer block.
    Maintains explicit separation between symbolic (XT) and contextual (XE) streams.
    """
    def __init__(self, layer_id: int, config: TFTDeepSeekConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        
        # Attention mechanism (updates symbolic stream)
        self.attn = TFTDeepSeekAttention(config)
        
        # Feed-forward network (updates contextual stream)
        if layer_id < config.n_dense_layers:
            self.ffn = TFTDeepSeekMLP(config.dim, config.inter_dim)
            self.use_moe = False
        else:
            self.ffn = TFTDeepSeekMoE(config)
            self.use_moe = True
        
        # Layer normalization
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self, xt: torch.Tensor, xe: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass maintaining stream separation.
        
        Args:
            xt: Symbolic stream (token-like representations)
            xe: Contextual stream (embedding-like representations)
            start_pos: Starting position for cached inference
            use_cache: Whether to use/update attention cache
            
        Returns:
            Updated (xt, xe) streams
        """
        # Attention sub-layer: updates symbolic stream only
        x_combined_attn = self.attn_norm(xt + xe)
        attn_output = self.attn(x_combined_attn, xt, start_pos, use_cache)
        xt_new = xt + attn_output
        
        # FFN sub-layer: updates contextual stream only
        x_combined_ffn = self.ffn_norm(xt_new + xe)
        if self.use_moe:
            xe_new = self.ffn(x_combined_ffn, xe)
        else:
            xe_new = xe + self.ffn(x_combined_ffn)
        
        return xt_new, xe_new


class ParallelEmbedding(nn.Module):
    """Embedding layer with distributed training support."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        if world_size > 1:
            assert vocab_size % world_size == 0
            self.part_vocab_size = vocab_size // world_size
            self.vocab_start_idx = rank * self.part_vocab_size
            self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        else:
            self.part_vocab_size = vocab_size
            self.vocab_start_idx = 0
            self.vocab_end_idx = vocab_size
        
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


class TFTDeepSeekTransformer(nn.Module):
    """
    Token-Factored DeepSeek Transformer model.
    
    Key features:
    - Explicit symbolic (XT) and contextual (XE) stream separation
    - ALiBi positional encoding instead of RoPE
    - Kronecker-lifted attention for token interpretability
    - Stream-aware MoE that only updates contextual stream
    """
    def __init__(self, config: TFTDeepSeekConfig):
        super().__init__()
        global world_size, rank
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        
        self.config = config
        Linear.dtype = torch.float8_e4m3fn if config.dtype == "fp8" else torch.bfloat16
        
        # Token embeddings (no positional embeddings - using ALiBi)
        self.embed = ParallelEmbedding(config.vocab_size, config.dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TFTDeepSeekBlock(layer_id, config) for layer_id in range(config.n_layers)
        ])
        
        # Final normalization and output projection
        self.norm = RMSNorm(config.dim)
        self.head = ColumnParallelLinear(config.dim, config.vocab_size, dtype=torch.get_default_dtype())
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights following standard practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) or isinstance(module, ParallelEmbedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, TFTDeepSeekAttention):
            # Initialize Kronecker parameters
            if hasattr(module, 'v_kronecker'):
                torch.nn.init.normal_(module.v_kronecker, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """Count model parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.embed, 'weight'):
            n_params -= self.embed.weight.numel()
        return n_params

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0, use_cache: bool = False):
        """
        Forward pass with explicit stream separation.
        
        Args:
            tokens: Input token IDs
            start_pos: Starting position for cached inference
            use_cache: Whether to use attention caching
            
        Returns:
            Logits for next token prediction
        """
        bsz, seqlen = tokens.size()
        
        # Initialize streams
        # XT: Pure token embeddings (symbolic stream)
        xt = self.embed(tokens)
        # XE: Zero initialization (contextual stream)
        xe = torch.zeros_like(xt)
        
        # Pass through transformer layers maintaining stream separation
        for layer in self.layers:
            xt, xe = layer(xt, xe, start_pos, use_cache)
        
        # Final processing: combine streams and project to vocabulary
        h = self.norm(xt + xe)
        
        # For generation, only need last token logits
        if use_cache and seqlen == 1:
            h = h[:, -1]
        else:
            h = h[:, -1]  # Always take last token for next token prediction
        
        logits = self.head(h)
        
        # Gather distributed logits
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
            
        return logits

    def forward_with_loss(self, tokens: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass with loss computation for training.
        
        Args:
            tokens: Input token IDs
            labels: Target token IDs
            
        Returns:
            Dictionary with loss and logits
        """
        bsz, seqlen = tokens.size()
        
        # Initialize streams
        xt = self.embed(tokens)
        xe = torch.zeros_like(xt)
        
        # Pass through transformer layers
        for layer in self.layers:
            xt, xe = layer(xt, xe, start_pos=0, use_cache=False)
        
        # Final processing
        h = self.norm(xt + xe)
        logits = self.head(h)
        
        # Gather distributed logits
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        
        # Compute loss
        loss = None
        if labels is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten and compute cross-entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits
        }

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: Optional[int] = None, 
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        
        Args:
            tokens: Initial token sequence
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token sequence
        """
        self.eval()
        device = tokens.device
        bsz, seqlen = tokens.size()
        
        # Initialize generation
        generated = tokens.clone()
        
        for step in range(max_new_tokens):
            # Get current sequence length
            current_len = generated.size(1)
            
            # Truncate if exceeding max length
            if current_len > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]
                current_len = self.config.max_seq_len
            
            # Forward pass
            if self.config.use_cache and step > 0:
                # For cached generation, only process last token
                input_tokens = generated[:, -1:]
                logits = self.forward(input_tokens, start_pos=current_len-1, use_cache=True)
            else:
                # Process full sequence
                logits = self.forward(generated, start_pos=0, use_cache=False)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        self.train()
        return generated

    def get_stream_representations(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the symbolic and contextual stream representations for interpretability analysis.
        
        Args:
            tokens: Input token IDs
            
        Returns:
            Tuple of (XT, XE) final representations
        """
        with torch.no_grad():
            bsz, seqlen = tokens.size()
            
            # Initialize streams
            xt = self.embed(tokens)
            xe = torch.zeros_like(xt)
            
            # Pass through all layers
            for layer in self.layers:
                xt, xe = layer(xt, xe, start_pos=0, use_cache=False)
            
            return xt, xe

    def analyze_stream_specialization(self, tokens: torch.Tensor, layer_idx: Optional[int] = None):
        """
        Analyze how well streams specialize for different types of reasoning.
        
        Args:
            tokens: Input token IDs
            layer_idx: Specific layer to analyze (None for final layer)
            
        Returns:
            Dictionary with specialization metrics
        """
        with torch.no_grad():
            bsz, seqlen = tokens.size()
            
            # Initialize streams
            xt = self.embed(tokens)
            xe = torch.zeros_like(xt)
            
            # Forward to specified layer
            target_layer = layer_idx if layer_idx is not None else len(self.layers) - 1
            
            for i, layer in enumerate(self.layers):
                xt, xe = layer(xt, xe, start_pos=0, use_cache=False)
                if i == target_layer:
                    break
            
            # Compute specialization metrics
            # 1. Token space proximity for XT
            token_embeddings = self.embed.weight
            if world_size > 1:
                # Gather all embedding weights for analysis
                all_embeddings = [torch.empty_like(token_embeddings) for _ in range(world_size)]
                dist.all_gather(all_embeddings, token_embeddings)
                token_embeddings = torch.cat(all_embeddings, dim=0)
            
            # Compute distances from XT to token embedding space
            xt_flat = xt.view(-1, xt.size(-1))
            distances_to_tokens = torch.cdist(xt_flat, token_embeddings)
            min_token_distances = distances_to_tokens.min(dim=1)[0]
            
            # 2. Orthogonality between streams
            xt_norm = F.normalize(xt.view(-1, xt.size(-1)), dim=1)
            xe_norm = F.normalize(xe.view(-1, xe.size(-1)), dim=1)
            stream_similarity = torch.sum(xt_norm * xe_norm, dim=1).abs()
            
            # 3. Stream magnitudes
            xt_magnitude = torch.norm(xt, dim=-1)
            xe_magnitude = torch.norm(xe, dim=-1)
            
            return {
                'layer': target_layer,
                'xt_token_proximity': min_token_distances.mean().item(),
                'stream_orthogonality': 1.0 - stream_similarity.mean().item(),  # Higher is better
                'xt_magnitude_mean': xt_magnitude.mean().item(),
                'xe_magnitude_mean': xe_magnitude.mean().item(),
                'magnitude_ratio': (xe_magnitude.mean() / xt_magnitude.mean()).item()
            }


# Utility functions for model creation and loading
def create_tft_deepseek_model(config: TFTDeepSeekConfig) -> TFTDeepSeekTransformer:
    """Create a TFT-DeepSeek model with the given configuration."""
    model = TFTDeepSeekTransformer(config)
    
    print(f"Created TFT-DeepSeek model with {model.get_num_params()/1e6:.2f}M parameters")
    print(f"Configuration: {config.n_layers} layers, {config.n_heads} heads, {config.dim} dim")
    print(f"Using ALiBi positional encoding (no RoPE)")
    print(f"Token-factored streams with Kronecker-lifted attention")
    
    return model


def get_default_config() -> TFTDeepSeekConfig:
    """Get default configuration for TFT-DeepSeek model."""
    return TFTDeepSeekConfig()


# Example usage and testing
if __name__ == "__main__":
    # Set default dtype and device
    torch.set_default_dtype(torch.bfloat16)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    
    # Create model with default configuration
    config = get_default_config()
    
    # Adjust config for testing
    config.n_layers = 4  # Smaller for testing
    config.n_heads = 8
    config.dim = 512
    config.vocab_size = 1000
    config.max_seq_len = 128
    config.max_batch_size = 2
    
    model = create_tft_deepseek_model(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nTesting forward pass with input shape: {tokens.shape}")
    
    # Test inference
    logits = model(tokens)
    print(f"Output logits shape: {logits.shape}")
    
    # Test training forward pass
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    result = model.forward_with_loss(tokens, labels)
    print(f"Training loss: {result['loss'].item():.4f}")
    
    # Test stream analysis
    xt, xe = model.get_stream_representations(tokens)
    print(f"XT stream shape: {xt.shape}")
    print(f"XE stream shape: {xe.shape}")
    
    # Test specialization analysis
    analysis = model.analyze_stream_specialization(tokens)
    print(f"\nStream specialization analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Test generation
    print(f"\nTesting generation...")
    generated = model.generate(tokens[:1, :10], max_new_tokens=20, temperature=0.8)
    print(f"Generated sequence shape: {generated.shape}")
    
    print(f"\nTFT-DeepSeek model test completed successfully!")
    print(f"Model maintains explicit symbolic/contextual stream separation")
    print(f"Uses ALiBi instead of RoPE for position encoding")
    print(f"Kronecker-lifted attention preserves token interpretability")
