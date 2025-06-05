#!/bin/bash

# Dictionary FFN experiments on Wikipedia with same config as run_train_compare
set -e

if [ "${LOCAL_RANK:-0}" != "0" ]; then
    exec > /dev/null 2>&1
fi

# === Configuration (same as run_train_compare.sh) ===
DATASET="wikimedia/wikipedia"
DATASET_CONFIG="20231101.en"
EPOCHS=3
BATCH_SIZE=$((128*4))
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

# Dictionary FFN variants to test
DICT_CONFIGS=(
    "factored:true:true"          # Dict FFN + factorizations
    "basic:false:false"           # Basic dict FFN
)

# Hardware
USE_ACCELERATE=true
NUM_GPUS=4
MIXED_PRECISION="bf16"

OUTPUT_BASE="./outputs/wiki_dict"

echo "🔤 Wikipedia Dictionary FFN Experiments"
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
    local suffix=$1
    local use_v=$2
    local use_proj=$3
    local extra_args=$4
    
    local output_dir="$OUTPUT_BASE/tft_dict${suffix}"
    if [ -n "$PRESET" ]; then
        local run_name="tft_dict${suffix}_wiki_$PRESET"
    else
        local run_name="tft_dict${suffix}_wiki_${N_LAYERS}L${N_HEADS}H${D_MODEL}D"
    fi
    
    echo "=== Training: TFT Dictionary${suffix} ==="
    
    local cmd_args=(
        --model "tft-dict"
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
    
    # Add preset - USE DICTIONARY PRESETS to enable dict FFN
    if [ -n "$PRESET" ]; then
        # Use dictionary version of the preset
        case "$PRESET" in
            "tiny")
                cmd_args+=(--preset "tiny-dict")
                ;;
            "small")
                cmd_args+=(--preset "small-dict")
                ;;
            *)
                # Fallback to tiny-dict for other presets
                cmd_args+=(--preset "tiny-dict")
                ;;
        esac
    else
        # Use dictionary preset that enables dict FFN
        cmd_args+=(--preset "small-dict")
    fi
    
    # Add TFT factorizations if specified
    if [ "$use_v" = true ]; then
        cmd_args+=(--use_v)
    fi
    if [ "$use_proj" = true ]; then
        cmd_args+=(--use_proj)
    fi
    
    if [ "$USE_ACCELERATE" = true ]; then
        cmd_args+=(--trainer accelerate --mixed_precision "$MIXED_PRECISION")
        if [ $NUM_GPUS -gt 1 ]; then
            accelerate launch --num_processes $NUM_GPUS --mixed_precision "$MIXED_PRECISION" \
                exp/train_tft.py "${cmd_args[@]}" $extra_args
        else
            python exp/train_tft.py "${cmd_args[@]}" $extra_args
        fi
    else
        python exp/train_tft.py "${cmd_args[@]}" $extra_args
    fi
}

# Run dictionary FFN variants
for config in "${DICT_CONFIGS[@]}"; do
    IFS=':' read -r suffix use_v use_proj <<< "$config"
    run_training "_$suffix" "$use_v" "$use_proj" ""
done