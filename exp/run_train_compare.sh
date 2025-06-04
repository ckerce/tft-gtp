#!/bin/bash

# Wikipedia model comparison with configurable parameters
set -e

# === Configuration ===
DATASET="wikimedia/wikipedia"
DATASET_CONFIG="20231101.en"
EPOCHS=3
BATCH_SIZE=128
MAX_SAMPLES=5000000

# Model size - leave empty to use individual params
PRESET=""
N_LAYERS=6
N_HEADS=6
D_MODEL=768

BLOCK_SIZE=128
LR=0.0005
WEIGHT_DECAY=0.01
SEED=42

# Hardware
USE_ACCELERATE=false
NUM_GPUS=1
MIXED_PRECISION="bf16"

OUTPUT_BASE="./outputs/wiki_compare"

echo "🚀 Wikipedia Model Comparison"
echo "Dataset: $DATASET ($DATASET_CONFIG)"
if [ -n "$PRESET" ]; then
    echo "Config: $PRESET | Epochs: $EPOCHS | Samples: $MAX_SAMPLES"
else
    echo "Config: ${N_LAYERS}L-${N_HEADS}H-${D_MODEL}D | Epochs: $EPOCHS | Samples: $MAX_SAMPLES"
fi

rm -rf "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE"

# Base training command
run_training() {
    local model=$1
    local suffix=$2
    local extra_args=$3
    
    local output_dir="$OUTPUT_BASE/${model}${suffix}"
    if [ -n "$PRESET" ]; then
        local run_name="${model}${suffix}_wiki_$PRESET"
    else
        local run_name="${model}${suffix}_wiki_${N_LAYERS}L${N_HEADS}H${D_MODEL}D"
    fi
    
    echo "=== Training: $model$suffix ==="
    
    local cmd_args=(
        --model "$model"
        --dataset "$DATASET"
        --dataset_config "$DATASET_CONFIG"
        --epochs $EPOCHS
        --batch_size $BATCH_SIZE
        --max_samples $MAX_SAMPLES
        --block_size $BLOCK_SIZE
        --lr $LR
        --weight_decay $WEIGHT_DECAY
        --seed $SEED
        --output_dir "$output_dir"
        --run_name "$run_name"
    )
    
    # Add preset or individual params
    if [ -n "$PRESET" ]; then
        cmd_args+=(--preset "$PRESET")
    else
        # Override with custom architecture
        cmd_args+=(--preset tiny)  # Use tiny as base, then override
        # Add overrides for custom architecture here if train_tft.py supports them
        # cmd_args+=(--n_layers $N_LAYERS --n_heads $N_HEADS --d_model $D_MODEL)
    fi
    
    if [ "$USE_ACCELERATE" = true ]; then
        cmd_args+=(--trainer accelerate --mixed_precision "$MIXED_PRECISION")
        if [ $NUM_GPUS -gt 1 ]; then
            accelerate launch --num_processes $NUM_GPUS --mixed_precision "$MIXED_PRECISION" \
                exptrain_tft.py "${cmd_args[@]}" $extra_args
        else
            python exptrain_tft.py "${cmd_args[@]}" $extra_args
        fi
    else
        python exptrain_tft.py "${cmd_args[@]}" $extra_args
    fi
}

# Run comparisons
run_training "vanilla" "" ""
run_training "tft" "_basic" ""
run_training "tft" "_factored" "--use_v --use_proj"

echo "✅ All training complete!"

# Quick results
echo "=== Results ==="
for dir in vanilla tft_basic tft_factored; do
    if [ -f "$OUTPUT_BASE/$dir/training_metrics.json" ]; then
        final_loss=$(python -c "
import json
try:
    with open('$OUTPUT_BASE/$dir/training_metrics.json') as f:
        data = json.load(f)
        epochs = data['metrics']['epochs']
        print(f'{epochs[-1][\"loss\"]:.4f}' if epochs else 'N/A')
except: print('ERROR')
")
        echo "$dir: $final_loss"
    else
        echo "$dir: FAILED"
    fi
done