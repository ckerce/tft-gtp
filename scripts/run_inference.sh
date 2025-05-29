#!/bin/bash

python scripts/run_inference.py \
  --model_path model/checkpoints/alibi_model.pt \
  --interactive
  --max_new_tokens 25 \
  --temperature 0.8 \
  --top_k 40
