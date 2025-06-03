#!/bin/bash

python scripts/run_inference.py \
  --model_path model/checkpoints/alibi_model.pt \
  --prompt "The cat sat on the" \
  --max_new_tokens 1 \
  --temperature 1 \
  --top_k 40
