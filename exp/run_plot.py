"""
Script to visualize TFT training results using the existing plotting utilities.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from utils.plotting import (
    plot_training_curves, 
    plot_convergence_comparison, 
    compare_runs,
    generate_training_report
)

def main():
    # Define the log files from your training runs
    log_files = [
        "outputs/wiki_compare/vanilla/training_metrics.json",
        "outputs/wiki_compare/tft_basic/training_metrics.json", 
        "outputs/wiki_compare/tft_factored/training_metrics.json",
        "outputs/wiki_dict/tft_dict_factored/training_metrics.json"
    ]
    
    # Check if log files exist
    existing_files = []
    for log_file in log_files:
        if os.path.exists(log_file):
            existing_files.append(log_file)
            print(f"✅ Found: {log_file}")
        else:
            print(f"❌ Missing: {log_file}")
    
    if not existing_files:
        print("No log files found! Make sure the training runs completed.")
        return
    
    print(f"\n📊 Visualizing {len(existing_files)} training runs...")
    
    # Create output directory for plots
    output_dir = "outputs/wiki_compare/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Generate overall comparison
    print("🔄 Creating training curves comparison...")
    plot_training_curves(
        existing_files, 
        output_file=f"{output_dir}/training_comparison.png",
        title="TFT vs Vanilla: Wikipedia Training Comparison"
    )
    
    # 2. Generate loss convergence comparison
    print("🔄 Creating loss convergence comparison...")
    plot_convergence_comparison(
        existing_files,
        metric="loss",
        output_file=f"{output_dir}/loss_convergence.png",
        title="Loss Convergence: TFT Variants vs Vanilla"
    )
    
    # 3. Generate comprehensive comparison plots
    print("🔄 Creating comprehensive comparison...")
    compare_runs(existing_files, output_dir=output_dir)
    
    # 4. Generate individual reports for each run
    for log_file in existing_files:
        model_name = Path(log_file).parent.name
        print(f"🔄 Generating report for {model_name}...")
        
        report_dir = f"{output_dir}/{model_name}_report"
        generate_training_report(log_file, output_dir=report_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  📈 training_comparison.png - Overall training curves")
    print("  📉 loss_convergence.png - Loss convergence comparison") 
    print("  📊 run_comparison.png - Detailed run comparison")
    print("  📁 Individual model reports in subdirectories")
    
    # Print summary statistics
    print("\n📋 TRAINING SUMMARY:")
    print("=" * 60)
    
    import json
    for log_file in existing_files:
        with open(log_file, 'r') as f:
            data = json.load(f)
        
        run_name = data['run_info']['run_name']
        final_loss = data['run_info']['final_metrics']['final_loss']
        training_time = data['run_info']['final_metrics']['training_time']
        total_steps = data['run_info']['final_metrics']['total_steps']
        
        print(f"{run_name}:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Training Time: {training_time/3600:.2f} hours")
        print(f"  Total Steps: {total_steps:,}")
        print(f"  Steps/sec: {total_steps/training_time:.1f}")
        print()

if __name__ == "__main__":
    main()