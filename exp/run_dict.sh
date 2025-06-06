#!/bin/bash

# Dictionary model training with configurable parameters
set -e

if [ "${LOCAL_RANK:-0}" != "0" ]; then
    exec > /dev/null 2>&1
fi

# === Configuration ===
DATASET="wikimedia/wikipedia"
DATASET_CONFIG="20231101.en"
EPOCHS=3
BATCH_SIZE=512
MAX_SAMPLES=5000000

# Model size - leave empty to use individual params
PRESET=""
N_LAYERS=6
N_HEADS=6
D_MODEL=768

BLOCK_SIZE=128
LR=0.0005
WEIGHT_DECAY=0.01
DICT_LOSS_WEIGHT=1.0
SEED=42

# Hardware
USE_ACCELERATE=true
NUM_GPUS=4
MIXED_PRECISION="bf16"

OUTPUT_DIR="./outputs/dict_training"

echo "🚀 Dictionary Model Training"
echo "Dataset: $DATASET ($DATASET_CONFIG)"
if [ -n "$PRESET" ]; then
    echo "Config: $PRESET | Epochs: $EPOCHS | Samples: $MAX_SAMPLES"
else
    echo "Config: ${N_LAYERS}L-${N_HEADS}H-${D_MODEL}D | Epochs: $EPOCHS | Samples: $MAX_SAMPLES"
fi
echo "Dictionary Loss Weight: $DICT_LOSS_WEIGHT"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "=== Training: TFT-DICT ==="

cmd_args=(
    --model tft-dict
    --use_dict_ffn
    --dataset "$DATASET"
    --dataset_config "$DATASET_CONFIG"
    --epochs $EPOCHS
    --batch_size $BATCH_SIZE
    --max_samples $MAX_SAMPLES
    --block_size $BLOCK_SIZE
    --lr $LR
    --weight_decay $WEIGHT_DECAY
    --dict_loss_weight $DICT_LOSS_WEIGHT
    --seed $SEED
    --output_dir "$OUTPUT_DIR"
    --run_name "tft_dict_wiki"
    --validation_split 0.1
    --validate_every_n_epochs 1
)

# Add preset or individual architecture params
if [ -n "$PRESET" ]; then
    cmd_args+=(--preset "$PRESET")
else
    # Use individual architecture parameters
    cmd_args+=(--preset tiny)  # Start with tiny preset as base
    cmd_args+=(--n_layers $N_LAYERS)
    cmd_args+=(--n_heads $N_HEADS)
    cmd_args+=(--d_model $D_MODEL)
fi

if [ "$USE_ACCELERATE" = true ]; then
    cmd_args+=(--trainer accelerate --mixed_precision "$MIXED_PRECISION")
    if [ $NUM_GPUS -gt 1 ]; then
        accelerate launch --num_processes $NUM_GPUS --mixed_precision "$MIXED_PRECISION" \
            exp/train_tft.py "${cmd_args[@]}"
    else
        python exp/train_tft.py "${cmd_args[@]}"
    fi
else
    python exp/train_tft.py "${cmd_args[@]}"
fi

echo "🎉 Dictionary model training completed! Results in $OUTPUT_DIR"