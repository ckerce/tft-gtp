#!/bin/bash

# Enhanced Wikipedia model comparison with Dictionary FFN and validation support
set -e

if [ "${LOCAL_RANK:-0}" != "0" ]; then
    exec > /dev/null 2>&1
fi

# === Configuration ===

# Hardware
USE_ACCELERATE=true
NUM_GPUS=4
MIXED_PRECISION="bf16"

# Broad
DATASET="wikimedia/wikipedia"
DATASET_CONFIG="20231101.en"
EPOCHS=3
BATCH_SIZE=$((128*NUM_GPUS))
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

# NEW: Validation settings
VAL_SPLIT=0.1
MAX_VAL_SAMPLES=10000
VALIDATE_EVERY_N_EPOCHS=1


OUTPUT_BASE="./outputs/wiki_compare_enhanced"

echo "🚀 Enhanced Wikipedia Model Comparison with Dictionary FFN & Validation"
echo "Dataset: $DATASET ($DATASET_CONFIG)"
if [ -n "$PRESET" ]; then
    echo "Config: $PRESET | Epochs: $EPOCHS | Samples: $MAX_SAMPLES"
else
    echo "Config: ${N_LAYERS}L-${N_HEADS}H-${D_MODEL}D | Epochs: $EPOCHS | Samples: $MAX_SAMPLES"
fi
echo "Validation: 10% split, every $VALIDATE_EVERY_N_EPOCHS epochs"

rm -rf "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE"

# Base training command with validation support
run_training() {
    local model=$1
    local suffix=$2
    local extra_args=$3
    
    local output_dir="$OUTPUT_BASE/${model}${suffix}"
    if [ -n "$PRESET" ]; then
        local run_name="${model}${suffix}_wiki_val_$PRESET"
    else
        local run_name="${model}${suffix}_wiki_val_${N_LAYERS}L${N_HEADS}H${D_MODEL}D"
    fi
    
    echo "=== Training: $model$suffix (with validation) ==="
    
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
        # NEW: Validation arguments
        --val_split $VAL_SPLIT
        --max_val_samples $MAX_VAL_SAMPLES
        --validate_every_n_epochs $VALIDATE_EVERY_N_EPOCHS
        # Enable plotting to visualize validation curves
        --plot
    )
    
    # Add preset or individual params
    if [ -n "$PRESET" ]; then
        # Handle dictionary FFN presets
        if [[ "$model" == *"dict"* ]]; then
            case "$PRESET" in
                "tiny")
                    cmd_args+=(--preset "tiny-dict")
                    ;;
                "small")
                    cmd_args+=(--preset "small-dict")
                    ;;
                *)
                    # Default to small-dict for other presets
                    cmd_args+=(--preset "small-dict")
                    ;;
            esac
        else
            cmd_args+=(--preset "$PRESET")
        fi
    else
        # Override with custom architecture
        if [[ "$model" == *"dict"* ]]; then
            cmd_args+=(--preset "small-dict")  # Use dict preset as base for custom arch
        else
            cmd_args+=(--preset tiny)  # Use tiny as base, then override
        fi
        # Add overrides for custom architecture here if train_tft.py supports them
        # cmd_args+=(--n_layers $N_LAYERS --n_heads $N_HEADS --d_model $D_MODEL)
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

# Enhanced model comparison matrix
echo ""
echo "🏗️ Model Comparison Matrix:"
echo "1. Vanilla Transformer (baseline)"
echo "2. TFT Basic (no factorizations)"
echo "3. TFT with Value factorization"
echo "4. TFT with Output projection factorization"
echo "5. TFT Fully factored (both V and Proj)"
echo "6. TFT Dictionary FFN (basic)"
echo "7. TFT Dictionary FFN + Value factorization"
echo "8. TFT Dictionary FFN + Output projection"
echo "9. TFT Dictionary FFN + Full factorization"
echo ""

# Run enhanced comparisons
echo "🚀 Starting enhanced model comparison runs..."

# 1. Vanilla transformer baseline
run_training "vanilla" "_baseline" ""

# 2. Basic TFT (no factorizations)
run_training "tft" "_basic" ""

# 3. TFT with value factorization
run_training "tft" "_value_fact" "--use_v"

# 4. TFT with output projection factorization
run_training "tft" "_proj_fact" "--use_proj"

# 5. TFT with both factorizations
run_training "tft" "_full_fact" "--use_v --use_proj"

# 6. NEW: TFT Dictionary FFN (basic)
run_training "tft-dict" "_dict_basic" ""

# 7. NEW: TFT Dictionary FFN + Value factorization
run_training "tft-dict" "_dict_value" "--use_v"

# 8. NEW: TFT Dictionary FFN + Output projection
run_training "tft-dict" "_dict_proj" "--use_proj"

# 9. NEW: TFT Dictionary FFN + Full factorization
run_training "tft-dict" "_dict_full" "--use_v --use_proj"

# Generate comparison report
echo ""
echo "📊 Generating comparison report..."

cat > "$OUTPUT_BASE/comparison_summary.md" << EOF
# Enhanced Wikipedia Model Comparison Results

## Experiment Configuration
- **Dataset**: $DATASET ($DATASET_CONFIG)
- **Samples**: $MAX_SAMPLES training, $MAX_VAL_SAMPLES validation
- **Architecture**: ${N_LAYERS}L-${N_HEADS}H-${D_MODEL}D (if custom) or $PRESET preset
- **Training**: $EPOCHS epochs, batch size $BATCH_SIZE, LR $LR
- **Validation**: 10% split, every $VALIDATE_EVERY_N_EPOCHS epochs
- **Hardware**: $NUM_GPUS GPUs, $MIXED_PRECISION precision

## Models Compared

### Standard Transformers
1. **Vanilla Transformer**: Standard transformer baseline
2. **TFT Basic**: Token-factored streams, no matrix factorizations
3. **TFT Value**: TFT + value matrix factorization in attention
4. **TFT Projection**: TFT + output projection factorization
5. **TFT Full**: TFT + both value and projection factorizations

### Dictionary FFN Variants
6. **TFT Dict Basic**: TFT + dictionary FFN (interpretable feed-forward)
7. **TFT Dict Value**: Dictionary FFN + value factorization
8. **TFT Dict Projection**: Dictionary FFN + output projection factorization
9. **TFT Dict Full**: Dictionary FFN + both factorizations

## Results Analysis

Check individual model directories for:
- \`training_metrics.json\`: Detailed training and validation curves
- \`*.png\`: Automatically generated loss/perplexity plots
- \`*_model.pt\`: Saved model checkpoints

### Key Metrics to Compare
- **Training Loss**: Final training loss convergence
- **Validation Loss**: Generalization performance
- **Perplexity**: Both training and validation perplexity
- **Training Time**: Computational efficiency
- **Parameter Count**: Model size efficiency
- **Dictionary Interpretability**: For dict FFN variants, analyze dictionary weights

### Expected Insights
1. **Baseline Performance**: How does vanilla transformer perform?
2. **Factorization Impact**: Do matrix factorizations help or hurt performance?
3. **Dictionary Benefit**: Does dictionary FFN improve interpretability without sacrificing performance?
4. **Validation Gaps**: Which models generalize best (smallest train-val gap)?
5. **Efficiency Trade-offs**: Performance vs parameter count vs training time

## Generated Files
- Each model directory contains complete training logs and plots
- Use \`comparison_analysis.py\` to aggregate results across all models
- Validation curves show generalization performance over training

EOF

echo "✅ Enhanced comparison completed!"
echo ""
echo "📁 Results saved to: $OUTPUT_BASE"
echo "📋 Summary report: $OUTPUT_BASE/comparison_summary.md"
echo ""
echo "🔍 Next steps:"
echo "1. Check individual model directories for detailed results"
echo "2. Compare validation curves to assess generalization"
echo "3. Analyze dictionary FFN interpretability (models 6-9)"
echo "4. Run inference tests on best-performing models"
echo ""
echo "📊 Example analysis commands:"
echo "  # Compare final validation losses:"
echo "  grep 'Final validation loss' $OUTPUT_BASE/*/train_tft.log"
echo ""
echo "  # View training plots:"
echo "  ls $OUTPUT_BASE/*/*.png"
echo ""
echo "  # Analyze JSON logs:"
echo "  python -c \"import json; [print(f'{d}: {json.load(open(f\"{d}/training_metrics.json\"))[\"run_info\"][\"final_metrics\"]}') for d in ['$OUTPUT_BASE/vanilla_baseline', '$OUTPUT_BASE/tft_basic', '$OUTPUT_BASE/tft-dict_dict_full']]\""