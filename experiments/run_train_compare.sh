#!/bin/bash
# scripts/compare_models.sh
# Simple comparison script with customizable parameters
# Example Usage: 
# DATASET="wikimedia/wikipedia" \
# DATASET_CONFIG="20231101.en" \
# BLOCK_SIZE=128 \
# BATCH_SIZE=128 \
# OUTPUT_BASE="./outputs/wikipedia_run" \
# NUM_EPOCHS=3 \
# MAX_SAMPLES=5000000 \
# N_LAYER=6 \
# N_HEAD=6 \
# N_EMBD=768 \
# LEARNING_RATE=0.0005 \
# ./compare_models.sh

set -e

echo "🚀 Model Comparison Script"
echo "========================="

# Default values (can be overridden)
NUM_GPUS=${NUM_GPUS:-2}
DATASET=${DATASET:-"roneneldan/TinyStories"}
DATASET_CONFIG=${DATASET_CONFIG:-""}
BLOCK_SIZE=${BLOCK_SIZE:-128}
BATCH_SIZE=${BATCH_SIZE:-32}
OUTPUT_BASE=${OUTPUT_BASE:-"./outputs"}
TOKENIZER_TYPE=${TOKENIZER_TYPE:-"gpt2"}
NUM_EPOCHS=${NUM_EPOCHS:-5}
MAX_SAMPLES=${MAX_SAMPLES:-100000}
N_LAYER=${N_LAYER:-6}
N_HEAD=${N_HEAD:-6}
N_EMBD=${N_EMBD:-384}
DROPOUT=${DROPOUT:-0.1}
LEARNING_RATE=${LEARNING_RATE:-0.0003}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}
GRAD_ACCUM=${GRAD_ACCUM:-2}
SEED=${SEED:-42}

# Show configuration
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Dataset: $DATASET"
if [[ -n "$DATASET_CONFIG" ]]; then
    echo "  Dataset Config: $DATASET_CONFIG"
fi
echo "  Architecture: ${N_LAYER}L-${N_HEAD}H-${N_EMBD}D"
echo "  Block Size: $BLOCK_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Build dataset config argument
DATASET_CONFIG_ARG=""
if [[ -n "$DATASET_CONFIG" ]]; then
    DATASET_CONFIG_ARG="--dataset_config $DATASET_CONFIG"
fi

# Common arguments for all models
COMMON_ARGS="--dataset $DATASET $DATASET_CONFIG_ARG --block_size $BLOCK_SIZE --batch_size $BATCH_SIZE --tokenizer_type $TOKENIZER_TYPE --num_epochs $NUM_EPOCHS --max_samples $MAX_SAMPLES --n_layer $N_LAYER --n_head $N_HEAD --n_embd $N_EMBD --dropout $DROPOUT --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --trainer accelerate --mixed_precision $MIXED_PRECISION --gradient_accumulation_steps $GRAD_ACCUM --seed $SEED"

# Function to run training
run_training() {
    local model_type=$1
    local output_dir=$2
    local extra_args=$3
    
    echo "🎯 Training $model_type..."
    echo "   Output: $output_dir"
    
    local start_time=$(date +%s)
    
    accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision $MIXED_PRECISION \
        experiments/train_tft.py \
        --model $model_type \
        --output_dir $output_dir \
        $extra_args \
        $COMMON_ARGS
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "✅ $model_type completed in ${duration}s"
    echo ""
    
    return $duration
}

# Create output directories
VANILLA_OUTPUT="$OUTPUT_BASE/vanilla"
TFT_OUTPUT="$OUTPUT_BASE/tft"
TFT_FACTORED_OUTPUT="$OUTPUT_BASE/tft_factored"

mkdir -p "$OUTPUT_BASE"

# Train models
echo "=== 1/3: Vanilla Transformer ==="
run_training "vanilla" "$VANILLA_OUTPUT" ""
VANILLA_TIME=$?

echo "=== 2/3: TFT Basic ==="
run_training "tft" "$TFT_OUTPUT" ""
TFT_TIME=$?

echo "=== 3/3: TFT Factorized ==="
run_training "tft" "$TFT_FACTORED_OUTPUT" "--use_v --use_proj"
TFT_FACTORED_TIME=$?

echo "🎉 All training completed!"
echo ""
echo "Training Times:"
echo "  Vanilla:        ${VANILLA_TIME}s"
echo "  TFT Basic:      ${TFT_TIME}s"
echo "  TFT Factorized: ${TFT_FACTORED_TIME}s"
echo ""
echo "Results saved to: $OUTPUT_BASE"