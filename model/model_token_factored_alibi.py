# ./model/model_token_factored_alibi.py
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
import os
import torch.serialization
from config_alibi import GPTConfigALiBi, print_config_alibi


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        """Applies Layer Normalization with improved stability."""
        return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)


class FactoredCausalSelfAttentionALiBi(nn.Module):
    """
    Causal self-attention mechanism for the Factored Transformer with ALiBi positional encoding.
    CORRECTED: Uses only Q and K projections, with xt directly as values (unless use_v=True).
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Always have Key and Query projections
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        
        # Store config for conditional behavior
        self.use_v = getattr(config, 'use_v', False)
        self.use_proj = getattr(config, 'use_proj', False)
        
        # Conditional V and output projection matrices
        if self.use_v:
            # Create n_heads x n_heads trainable parameters for V
            self.v_tmp = nn.Parameter(torch.randn(config.n_head, config.n_head) * 0.02)
            
        if self.use_proj:
            # Create n_heads x n_heads trainable parameters for output projection
            self.proj_tmp = nn.Parameter(torch.randn(config.n_head, config.n_head) * 0.02)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        if self.use_proj:
            self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # ALiBi slopes - computed once and cached
        slopes = self._get_alibi_slopes(config.n_head)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def _get_kronecker_lifted_tensor(self, v):
        '''
        v should have dimensions n_heads x n_heads (number of heads by number of heads)
        v_out will be lifted to n_embd x n_embd

        If v_in is a matrix {v_in_ij}, then
        v_out_ij = v_in_ij * eye(n_embd // n_heads)
        
        This creates a block-diagonal structure where each n_heads x n_heads block
        is scaled by the corresponding element in v.
        '''
        n_heads = v.shape[0]
        head_dim = self.n_embd // n_heads
        
        # Create the lifted tensor
        v_out = torch.zeros(self.n_embd, self.n_embd, device=v.device, dtype=v.dtype)
        
        for i in range(n_heads):
            for j in range(n_heads):
                # Create identity matrix scaled by v[i,j]
                start_i, end_i = i * head_dim, (i + 1) * head_dim
                start_j, end_j = j * head_dim, (j + 1) * head_dim
                v_out[start_i:end_i, start_j:end_j] = v[i, j] * torch.eye(head_dim, device=v.device, dtype=v.dtype)
        
        return v_out

    def _get_alibi_slopes(self, n_heads):
        """
        Compute ALiBi slopes for each attention head.
        Implementation based on the original ALiBi paper.
        """
        def get_slopes_power_of_2(n_heads):
            start = 2**(-(2**-(math.log2(n_heads)-3)))
            ratio = start
            return [start*ratio**i for i in range(n_heads)]

        def get_slopes(n_heads):
            if n_heads <= 0:
                return []
            
            # Check if n_heads is a power of 2
            if (n_heads & (n_heads - 1)) == 0:
                return get_slopes_power_of_2(n_heads)
            else:
                # Handle non-power-of-2 case
                closest_power_of_2 = 2**math.floor(math.log2(n_heads))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                
                # Get additional slopes for remaining heads
                if n_heads > closest_power_of_2:
                    extra_base = 2**(-(2**-(math.log2(2*closest_power_of_2)-3)))
                    num_remaining = n_heads - closest_power_of_2
                    extra_slopes = [extra_base * (extra_base**i) for i in range(num_remaining)]
                    slopes.extend(extra_slopes)
                
                return slopes[:n_heads]

        slopes = get_slopes(n_heads)
        return torch.tensor(slopes, dtype=torch.float32)

    def _get_alibi_bias(self, seq_len, device):
        """
        Generate ALiBi bias matrix for the given sequence length.
        Returns bias matrix of shape (n_head, seq_len, seq_len).
        """
        # Create position indices
        context_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        memory_position = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Compute relative distances (memory - context)
        # Shape: (seq_len, seq_len)
        relative_position = memory_position[None, :] - context_position[:, None]
        
        # For causal attention, future positions should have large negative bias
        # We use the absolute distance but make future positions very negative
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        
        # Apply slopes to relative positions
        # Negative relative positions (past) get small negative bias
        # Future positions will be masked with -inf
        alibi_bias = self.alibi_slopes[:, None, None] * relative_position[None, :, :]
        
        # Apply causal masking
        alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float('-inf'))
        
        return alibi_bias

    def forward(self, x_norm_for_qk, xt_current_for_v):
        """
        Forward pass for FactoredCausalSelfAttentionALiBi.
        x_norm_for_qk used for Q,K; xt_current_for_v used directly as values (or projected if use_v=True).
        """
        B, T, C = x_norm_for_qk.size()

        # Calculate only query and key from normalized combined state
        q, k = self.c_attn(x_norm_for_qk).split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Handle values based on use_v flag
        if self.use_v:
            # Apply Kronecker-lifted V matrix to xt
            v_matrix = self._get_kronecker_lifted_tensor(self.v_tmp)
            # Reshape xt_current_for_v for matrix multiplication: (B*T, C)
            xt_flat = xt_current_for_v.view(-1, C)
            # Apply V matrix: (B*T, C) @ (C, C) -> (B*T, C)
            v_flat = torch.matmul(xt_flat, v_matrix.t())
            # Reshape back and prepare for multi-head attention
            v = v_flat.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        else:
            # Use xt directly as values (no projection, no modulation)
            v = xt_current_for_v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores with proper scaling
        scale = 1.0 / math.sqrt(self.head_dim)
        att_scores = (q @ k.transpose(-2, -1)) * scale

        # Add ALiBi bias
        if T > 1:  # Only apply ALiBi for sequences longer than 1
            alibi_bias = self._get_alibi_bias(T, x_norm_for_qk.device)
            att_scores = att_scores + alibi_bias[None, :, :, :]

        # Apply softmax
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        # Apply attention to values
        y = att_weights @ v  # (B, nh, T, hs)

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply output projection if enabled
        if self.use_proj:
            proj_matrix = self._get_kronecker_lifted_tensor(self.proj_tmp)
            # Reshape for matrix multiplication: (B*T, C)
            y_flat = y.view(-1, C)
            # Apply projection: (B*T, C) @ (C, C) -> (B*T, C)
            y_flat = torch.matmul(y_flat, proj_matrix.t())
            # Reshape back
            y = y_flat.view(B, T, C)
            # Apply residual dropout
            y = self.resid_dropout(y)
        
        return y


class FactoredMLP(nn.Module):
    """MLP for the Factored Transformer."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x_norm):
        """Forward pass for FactoredMLP."""
        x = self.c_fc(x_norm)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class FactoredPreLNBlockALiBi(nn.Module):
    """Transformer block with Pre-Layer Normalization for the Factored Transformer with ALiBi."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = FactoredCausalSelfAttentionALiBi(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FactoredMLP(config)

    def forward(self, xt, xe, return_ffn_out = False):
        """Forward pass for FactoredPreLNBlockALiBi."""
        # Attention path - updates xt
        # Use combined state (xt + xe) for Q,K computation, but xt directly for values
        norm_for_attn = self.ln_1(xt + xe)
        attn_output = self.attn(x_norm_for_qk=norm_for_attn, xt_current_for_v=xt)
        xt = xt + attn_output

        # MLP path - updates xe
        norm_for_mlp = self.ln_2(xt + xe)
        mlp_output = self.mlp(norm_for_mlp)
        xe = xe + mlp_output

        if return_ffn_out:
            return xt, xe, mlp_output, attn_output
        return xt, xe


class FactoredTransformerModelALiBi(nn.Module):
    """
    Factored Transformer model with ALiBi positional encoding.
    CORRECTED version with proper symbolic structure preservation.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)
        
        # Store use_v flag for proper initialization
        self.use_v = getattr(config, 'use_v', False)
        self.use_proj = getattr(config, 'use_proj', False)

        # Model components (no positional embeddings with ALiBi)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([FactoredPreLNBlockALiBi(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language model head (shared weights with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Special initialization for output projections and factored attention
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        self._probe_file = None

        print_config_alibi(config)
        print(f"FactoredTransformerModelALiBi initialized with {self.get_num_params()/1e6:.2f}M parameters")
        print(f"Using factored attention with use_v={self.use_v}, use_proj={self.use_proj}")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, FactoredCausalSelfAttentionALiBi):
            # Initialize the factored attention parameters
            if hasattr(module, 'v_tmp'):
                # Initialize V matrix parameters
                torch.nn.init.normal_(module.v_tmp, mean=0.0, std=0.02)
            if hasattr(module, 'proj_tmp'):
                # Initialize output projection parameters  
                torch.nn.init.normal_(module.proj_tmp, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for the FactoredTransformerModelALiBi."""
        device = input_ids.device
        b, t = input_ids.size()

        # Check sequence length limits
        max_len = getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)
        if t > max_len:
            raise ValueError(f"Sequence length {t} exceeds maximum supported length {max_len}")

        # Token embeddings only (no positional embeddings with ALiBi)
        tok_emb = self.transformer.wte(input_ids)
        
        # Initialize xt and xe streams
        xt = self.transformer.drop(tok_emb)
        xe = torch.zeros_like(xt, device=device)

        # Set up output file once
        if self._probe_file is None and not self.training:
            self._probe_file = open("outputs/xe_symbolic_projection_probe.txt", "w")
            self._probe_file.write("=== Symbolic Projection Diagnostic ===\n")


        # Pass through transformer blocks
        ffn_outputs = []
        for block in self.transformer.h:
            xt, xe, ffn_out, attn_out = block(xt, xe, return_ffn_out = True)
            ffn_outputs.append(self.transformer.ln_f(xe))

            # === Probe after each layer ===
            with torch.no_grad():
                E = self.transformer.wte.weight             # [V, d]
                E_T = E.T                                   # [d, V]
                xe_flat = xe.view(-1, xe.size(-1))          # [B*T, d]

                # No LayerNorm
                q1 = xe_flat
                logits1 = q1 @ E_T
                alpha1 = F.softmax(logits1, dim=-1)
                xe_proj1 = alpha1 @ E
                cos_sim1 = F.cosine_similarity(q1, xe_proj1, dim=-1)
                l2_diff1 = torch.norm(q1 - xe_proj1, dim=-1)

                # With LayerNorm
                q2 = F.layer_norm(q1, q1.shape[-1:])
                logits2 = q2 @ E_T
                alpha2 = F.softmax(logits2, dim=-1)
                xe_proj2 = alpha2 @ E
                cos_sim2 = F.cosine_similarity(q1, xe_proj2, dim=-1)
                l2_diff2 = torch.norm(q1 - xe_proj2, dim=-1)

                # Log
                if hasattr(self, "_probe_file"):
                    self._probe_file.write(f"\n--- Layer {layer_idx} ---\n")
                    self._probe_file.write(f"[No LN]  mean cosine: {cos_sim1.mean():.4f}, mean L2: {l2_diff1.mean():.4f}\n")
                    self._probe_file.write(f"[With LN] mean cosine: {cos_sim2.mean():.4f}, mean L2: {l2_diff2.mean():.4f}\n")


        if hasattr(self, "_probe_file") and not self.training:
            self._probe_file.close()


        # Logit Lens decoding
        # if not self.training:
        #     log_file = "outputs/logit_lens_output_xe_norm.txt"
        #     with open(log_file, "w") as f:
        #         f.write("=== Logit Lens Output ===\n")

        #         with torch.no_grad():
        #             for i, ffn_out in enumerate(ffn_outputs):
        #                 logits = torch.matmul(ffn_out, self.transformer.wte.weight.T)
        #                 probs = F.softmax(logits / 1, dim=-1)
        #                 top_probs, top_ids = torch.topk(probs, k=5, dim=-1)

        #                 input_tokens = [self.tokenizer.decode([id.item()]) for id in input_ids[0]]

        #                 f.write(f"\n--- Layer {i} ---\n")
        #                 for t in range(top_ids.shape[1]):
        #                     decoded_token = input_tokens[t]
        #                     guesses = [self.tokenizer.decode([id.item()]) for id in top_ids[0, t]]
        #                     f.write(f'Pos {t}: "{decoded_token}" → {guesses}\n')

        #     print(f"[Logit Lens] Written decoded outputs with tokens to: {log_file}")

        # Final processing
        x_final = xt + xe
        x_final = self.transformer.ln_f(x_final)
        logits = self.lm_head(x_final)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively with improved sampling.
        """
        self.eval()
        
        # Maximum total sequence length
        max_total_length = getattr(self.config, 'max_position_embeddings', self.config.block_size * 4)
        
        for _ in range(max_new_tokens):
            # Truncate context if too long
            idx_cond = idx if idx.size(1) <= max_total_length else idx[:, -max_total_length:]
            
            # Forward pass
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            # Get logits for the last position and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            
            # Add small amount of noise to prevent getting stuck
            if temperature > 0:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu'):
        """
        Loads a FactoredTransformerModelALiBi and its tokenizer from a checkpoint.

        Args:
            checkpoint_path (str): The path to the .pt checkpoint file.
            device (str): The device to load the model onto ('cpu' or 'cuda').

        Returns:
            tuple: (FactoredTransformerModelALiBi, object)
                     The loaded model instance, and the loaded tokenizer.
                     Returns (None, None) if loading fails.
        Example:
            from model.model_token_factored_alibi import FactoredTransformerModelALiBi as ModelALiBi
            model, tokenizer = ModelALiBi.load_from_checkpoint('output_alibi/alibi_model.pt',device='cuda')
        """
        # This is now the primary loading logic
        target_device = torch.device(device) # Ensure it's a torch.device
        print(f"Loading checkpoint using {cls.__name__}.load_from_checkpoint from: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            return None, None

        try:
            torch.serialization.add_safe_globals([GPTConfigALiBi])
            checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

            config = checkpoint.get('config')
            model_state_dict = checkpoint.get('model_state_dict')
            tokenizer = checkpoint.get('tokenizer')

            if not config or not model_state_dict:
                raise ValueError("Checkpoint must contain 'config' and 'model_state_dict'.")

            if not isinstance(config, GPTConfigALiBi):
                 raise TypeError(f"Config in checkpoint is not GPTConfigALiBi. Type: {type(config)}")

            model = cls(config) # 'cls' refers to FactoredTransformerModelALiBi
            model.load_state_dict(model_state_dict)
            model.to(target_device) # Already moved by map_location, but good for clarity
            model.eval()

            print(f"Model loaded successfully via class method ({model.get_num_params()/1e6:.2f}M params).")
            if tokenizer:
                print(f"Tokenizer loaded successfully (type: {type(tokenizer)}).")
            else:
                print("Warning: Tokenizer not found in checkpoint.")

            return model, tokenizer

        except Exception as e:
            print(f"An unexpected error occurred during {cls.__name__}.load_from_checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return None, None

# Ensure both are available if needed, but the class method is the primary one.
__all__ = ['FactoredTransformerModelALiBi']
