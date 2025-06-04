#!/bin/bash

set -e

echo "🚀 Model Comparison Script"
echo "========================="

# === Config with Defaults ===
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
DATASET=${DATASET:-"roneneldan/TinyStories"}
BLOCK_SIZE=${BLOCK_SIZE:-128}
BATCH_SIZE=${BATCH_SIZE:-32}
OUTPUT_BASE=${OUTPUT_BASE:-"./outputs"}
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_SAMPLES=${MAX_SAMPLES:-100000}

PRESET=${PRESET:-""}
N_LAYER=${N_LAYER:-6}
N_HEAD=${N_HEAD:-6}
N_EMBD=${N_EMBD:-384}
DROPOUT=${DROPOUT:-0.1}

LEARNING_RATE=${LEARNING_RATE:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}
GRAD_ACCUM=${GRAD_ACCUM:-2}
SEED=${SEED:-42}
DATASET_CONFIG=${DATASET_CONFIG:-""}

# === Show Configuration Summary ===
echo "GPUs: $NUM_GPUS | Dataset: $DATASET ${DATASET_CONFIG:+($DATASET_CONFIG)}"
echo "Arch: ${N_LAYER}L-${N_HEAD}H-${N_EMBD}D ${PRESET:+(preset: $PRESET)}"
echo "Epochs: $NUM_EPOCHS | Max Samples: $MAX_SAMPLES | LR: $LEARNING_RATE | Dropout: $DROPOUT"
echo ""

# === Build Command Arguments ===
build_args() {
    local model=$1
    local output=$2
    local extra=$3

    args=(--model "$model" --output_dir "$output" --dataset "$DATASET" --block_size "$BLOCK_SIZE" \
          --batch_size "$BATCH_SIZE" --epochs "$NUM_EPOCHS" --max_samples "$MAX_SAMPLES" \
          --lr "$LEARNING_RATE" --weight_decay "$WEIGHT_DECAY" --trainer "accelerate" \
          --mixed_precision "$MIXED_PRECISION" --gradient_accumulation_steps "$GRAD_ACCUM" \
          --seed "$SEED" --clip_grad_norm "1.0")

    [[ -n "$PRESET" ]] && args+=(--preset "$PRESET")
    [[ -z "$PRESET" ]] && args+=(--n_layer "$N_LAYER" --n_head "$N_HEAD" --n_embd "$N_EMBD")
    [[ -n "$PRESET" && "$N_LAYER" != 6 ]] && args+=(--n_layer "$N_LAYER")
    [[ -n "$PRESET" && "$N_HEAD" != 6 ]] && args+=(--n_head "$N_HEAD")
    [[ -n "$PRESET" && "$N_EMBD" != 384 ]] && args+=(--n_embd "$N_EMBD")
    [[ "$DROPOUT" != "0.1" ]] && args+=(--dropout "$DROPOUT")
    [[ -n "$DATASET_CONFIG" ]] && args+=(--dataset_config "$DATASET_CONFIG")
    [[ -n "$extra" ]] && args+=($extra)

    printf "%s\n" "${args[@]}"
}

# === Run Training ===
run_training() {
    local model=$1
    local output=$2
    local extra=$3

    echo "▶️ Training: $model -> $output"
    mapfile -t args < <(build_args "$model" "$output" "$extra")

    echo "Command: accelerate launch --num_processes $NUM_GPUS --mixed_precision $MIXED_PRECISION experiments/train_tft.py \\"
    printf "  %s \\\n" "${args[@]}"
    echo ""

    local start=$(date +%s)
    if accelerate launch --num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION" experiments/train_tft.py "${args[@]}"; then
        local duration=$(( $(date +%s) - start ))
        echo "$duration" > "$output/.duration"
        echo "✅ Done in ${duration}s"
        return 0
    else
        echo "❌ Failed"
        echo "FAILED" > "$output/.duration"
        return 1
    fi
}

# === Setup Output Dirs ===
mkdir -p "$OUTPUT_BASE"
VANILLA_OUT="$OUTPUT_BASE/vanilla"
TFT_OUT="$OUTPUT_BASE/tft"
TFT_FACTORED_OUT="$OUTPUT_BASE/tft_factored"

# === Run All Trainings ===
echo "=== 1/3: Vanilla Transformer ==="
run_training "vanilla" "$VANILLA_OUT" "" || echo "⚠️ Skipped Vanilla failure"

echo "=== 2/3: TFT Basic ==="
run_training "tft" "$TFT_OUT" "" || echo "⚠️ Skipped TFT Basic failure"

echo "=== 3/3: TFT Factorized ==="
run_training "tft" "$TFT_FACTORED_OUT" "--use_v --use_proj" || echo "⚠️ TFT Factorized failed"

# === Summary Report ===
VANILLA_TIME=$(cat "$VANILLA_OUT/.duration" 2>/dev/null || echo "FAILED")
TFT_TIME=$(cat "$TFT_OUT/.duration" 2>/dev/null || echo "FAILED")
TFT_FACTORED_TIME=$(cat "$TFT_FACTORED_OUT/.duration" 2>/dev/null || echo "FAILED")

echo ""
echo "✅ All trainings done!"
echo "Vanilla:        ${VANILLA_TIME}s"
echo "TFT Basic:      ${TFT_TIME}s"
echo "TFT Factorized: ${TFT_FACTORED_TIME}s"

# === Save Summary ===
cat > "$OUTPUT_BASE/comparison_summary.txt" <<EOF
Model Comparison Summary
========================
Dataset: $DATASET${DATASET_CONFIG:+ ($DATASET_CONFIG)}
Architecture: ${N_LAYER}L-${N_HEAD}H-${N_EMBD}D${PRESET:+ (preset: $PRESET)}
Epochs: $NUM_EPOCHS | Max Samples: $MAX_SAMPLES
Batch Size: $BATCH_SIZE | Block Size: $BLOCK_SIZE
LR: $LEARNING_RATE | Dropout: $DROPOUT

Results:
- Vanilla Transformer:  ${VANILLA_TIME}s
- TFT Basic:            ${TFT_TIME}s
- TFT Factorized:       ${TFT_FACTORED_TIME}s

Generated on: $(date)
EOF

echo "📄 Summary saved to $OUTPUT_BASE/comparison_summary.txt"
