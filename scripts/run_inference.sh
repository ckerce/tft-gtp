#!/bin/bash

python run_inference.py \
  --model_path checkpoints/alibi_model.pt \
  --model_type Factored \
  --prompt "The capital of France is" \
  --max_new_tokens 20 \
  --temperature 0.8 \
  --top_k 40