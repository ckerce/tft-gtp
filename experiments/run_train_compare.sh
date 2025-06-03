#!/bin/bash
# scripts/compare_accelerate_multi_gpu.sh
# Comprehensive multi-GPU comparison of TFT vs Vanilla Transformer

set -e  # Exit on any error

echo "🚀 Multi-GPU Accelerate Model Comparison"
echo "========================================"

# Configuration
NUM_GPUS=${1:-2}  # Default to 2 GPUs, can override with first argument
PRESET=${2:-small}  # Default to small preset
EPOCHS=${3:-10}     # Default to 10 epochs
BATCH_SIZE=32
GRAD_ACCUM=2
MIXED_PRECISION=bf16
SEED=42
DATASET="roneneldan/TinyStories"
MAX_SAMPLES=100000

# Output directories
BASE_OUTPUT="./outputs/multi_gpu_comparison"
TFT_OUTPUT="$BASE_OUTPUT/tft"
TFT_FACTORED_OUTPUT="$BASE_OUTPUT/tft_factored"
VANILLA_OUTPUT="$BASE_OUTPUT/vanilla"

echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Preset: $PRESET"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Mixed Precision: $MIXED_PRECISION"
echo "  Dataset: $DATASET"
echo "  Max Samples: $MAX_SAMPLES"
echo ""

# Check if accelerate is available
if ! command -v accelerate &> /dev/null; then
    echo "❌ Accelerate not found. Please install with: pip install accelerate"
    exit 1
fi

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "⚠️  Warning: Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
    echo "   Continuing with available GPUs..."
    NUM_GPUS=$GPU_COUNT
fi

echo "🔍 Available GPUs: $GPU_COUNT, Using: $NUM_GPUS"
echo ""

# Common training arguments
COMMON_ARGS="--preset $PRESET --epochs $EPOCHS --batch_size $BATCH_SIZE --trainer accelerate --mixed_precision $MIXED_PRECISION --gradient_accumulation_steps $GRAD_ACCUM --seed $SEED --dataset $DATASET --max_samples $MAX_SAMPLES --clip_grad_norm 1.0"

# Function to run training with timing
run_training() {
    local model_type=$1
    local output_dir=$2
    local run_name=$3
    local extra_args=$4
    
    echo "🎯 Training $model_type..."
    echo "   Output: $output_dir"
    echo "   Run name: $run_name"
    
    local start_time=$(date +%s)
    
    # Run with accelerate launch
    accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision $MIXED_PRECISION \
        experiments/train_tft.py \
        --model $model_type \
        --output_dir $output_dir \
        --run_name $run_name \
        $extra_args \
        $COMMON_ARGS \
        --plot
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "✅ $model_type completed in ${duration}s"
    echo ""
    
    return $duration
}

# Create base output directory
mkdir -p "$BASE_OUTPUT"

echo "🚀 Starting multi-GPU training comparison..."
echo ""

# 1. Train Vanilla Transformer (baseline)
echo "=== 1/3: Vanilla Transformer (Baseline) ==="
run_training "vanilla" "$VANILLA_OUTPUT" "vanilla_${PRESET}_${NUM_GPUS}gpu" ""
VANILLA_TIME=$?

# 2. Train basic TFT (no factorization)
echo "=== 2/3: Token-Factored Transformer (Basic) ==="
run_training "tft" "$TFT_OUTPUT" "tft_basic_${PRESET}_${NUM_GPUS}gpu" ""
TFT_BASIC_TIME=$?

# 3. Train TFT with factorization
echo "=== 3/3: Token-Factored Transformer (Factorized) ==="
run_training "tft" "$TFT_FACTORED_OUTPUT" "tft_factored_${PRESET}_${NUM_GPUS}gpu" "--use_v --use_proj"
TFT_FACTORED_TIME=$?

echo "🎉 All training completed!"
echo ""

# Generate comparison report
REPORT_FILE="$BASE_OUTPUT/comparison_report.md"

cat > "$REPORT_FILE" << EOF
# Multi-GPU Training Comparison Report

**Configuration:**
- GPUs: $NUM_GPUS
- Preset: $PRESET  
- Epochs: $EPOCHS
- Batch Size: $BATCH_SIZE
- Gradient Accumulation: $GRAD_ACCUM
- Mixed Precision: $MIXED_PRECISION
- Dataset: $DATASET
- Max Samples: $MAX_SAMPLES

**Training Times:**
- Vanilla Transformer: ${VANILLA_TIME}s
- TFT (Basic): ${TFT_BASIC_TIME}s  
- TFT (Factorized): ${TFT_FACTORED_TIME}s

## Models Compared

### 1. Vanilla Transformer
- Standard transformer with learned positional embeddings
- QKV attention with standard projections
- Location: \`$VANILLA_OUTPUT\`

### 2. TFT (Basic)
- Token-factored streams (xt + xe)
- ALiBi positional encoding
- No value/output factorization
- Location: \`$TFT_OUTPUT\`

### 3. TFT (Factorized) 
- Token-factored streams (xt + xe)
- ALiBi positional encoding
- Value matrix factorization (--use_v)
- Output projection factorization (--use_proj)
- Location: \`$TFT_FACTORED_OUTPUT\`

## Results

Check the individual output directories for:
- \`training_metrics.json\` - Detailed training logs
- \`*_model.pt\` - Saved model checkpoints
- \`training_curves.png\` - Training plots (if generated)

## Commands to Analyze Results

\`\`\`bash
# Compare training curves
python -c "
from utils.plotting import compare_runs
compare_runs([
    '$VANILLA_OUTPUT/training_metrics.json',
    '$TFT_OUTPUT/training_metrics.json', 
    '$TFT_FACTORED_OUTPUT/training_metrics.json'
], '$BASE_OUTPUT/model_comparison')
"

# Generate detailed reports
python -c "
from utils.plotting import generate_training_report
generate_training_report('$VANILLA_OUTPUT/training_metrics.json', '$BASE_OUTPUT/vanilla_report')
generate_training_report('$TFT_OUTPUT/training_metrics.json', '$BASE_OUTPUT/tft_basic_report')
generate_training_report('$TFT_FACTORED_OUTPUT/training_metrics.json', '$BASE_OUTPUT/tft_factored_report')
"
\`\`\`

EOF

echo "📊 Comparison Results Summary"
echo "=============================="
echo "Training Times:"
echo "  Vanilla:        ${VANILLA_TIME}s"
echo "  TFT Basic:      ${TFT_BASIC_TIME}s"
echo "  TFT Factorized: ${TFT_FACTORED_TIME}s"
echo ""
echo "📄 Full report saved to: $REPORT_FILE"
echo ""

# Extract final losses if available
echo "📈 Final Training Losses:"
for dir in "$VANILLA_OUTPUT" "$TFT_OUTPUT" "$TFT_FACTORED_OUTPUT"; do
    if [ -f "$dir/training_metrics.json" ]; then
        model_name=$(basename "$dir")
        final_loss=$(python -c "
import json
try:
    with open('$dir/training_metrics.json', 'r') as f:
        data = json.load(f)
    if data['metrics']['epochs']:
        print(f\"{data['metrics']['epochs'][-1].get('loss', 'N/A'):.6f}\")
    else:
        print('N/A')
except:
    print('N/A')
" 2>/dev/null)
        printf "  %-15s %s\n" "$model_name:" "$final_loss"
    fi
done
echo ""

# Generate comparison plots
echo "📊 Generating comparison plots..."
python3 << 'EOF'
import sys
import os
sys.path.insert(0, 'src')

try:
    from utils.plotting import compare_runs
    
    log_files = []
    labels = []
    
    base_output = os.environ.get('BASE_OUTPUT', './outputs/multi_gpu_comparison')
    
    for subdir, label in [('vanilla', 'Vanilla'), ('tft', 'TFT Basic'), ('tft_factored', 'TFT Factorized')]:
        log_file = f"{base_output}/{subdir}/training_metrics.json"
        if os.path.exists(log_file):
            log_files.append(log_file)
            print(f"Found log: {log_file}")
    
    if len(log_files) >= 2:
        print(f"Generating comparison plots for {len(log_files)} models...")
        compare_runs(log_files, f"{base_output}/comparison_plots")
        print(f"✅ Plots saved to {base_output}/comparison_plots/")
    else:
        print("❌ Not enough log files found for comparison")
        
except Exception as e:
    print(f"❌ Error generating plots: {e}")
EOF

echo ""
echo "🎉 Multi-GPU comparison complete!"
echo ""
echo "📁 Results structure:"
echo "   $BASE_OUTPUT/"
echo "   ├── vanilla/           # Vanilla transformer results"
echo "   ├── tft/              # Basic TFT results"  
echo "   ├── tft_factored/     # Factorized TFT results"
echo "   ├── comparison_plots/ # Model comparison plots"
echo "   └── comparison_report.md # Summary report"
echo ""
echo "🔍 Next steps:"
echo "   1. Read the report: cat $REPORT_FILE"
echo "   2. View plots: open $BASE_OUTPUT/comparison_plots/"
echo "   3. Analyze training logs in each model directory"
echo ""
echo "💡 To run again with different settings:"
echo "   bash scripts/compare_accelerate_multi_gpu.sh [num_gpus] [preset] [epochs]"
echo "   Example: bash scripts/compare_accelerate_multi_gpu.sh 4 medium 20"