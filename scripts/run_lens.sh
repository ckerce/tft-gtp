python train_tuned_lens_heads.py \
  --model_ckpt ./outputs/alibi_model.pt \
  --dataset "roneneldan/TinyStories" \
  --output_dir ./outputs/tuned_lens_heads \
  --epochs 5
