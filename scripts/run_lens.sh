#!/bin/bash

python examples/train_lens.py \ 
  --model_ckpt model/checkpoints/alibi_model.pt \
  --dataset "wikimedia/wikipedia" \
  --dataset_config "20231101.en" \
  --output_dir ./outputs/tuned_lens_heads \
  --epochs 5
