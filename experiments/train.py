#!/usr/bin/env python3
"""
Clean, simple training script for both Token-Factored Transformer and Vanilla Transformer.
No verbose logging, just clean training with JSON metrics.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, 'src')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import get_model, list_models
from config.model_configs import get_config
from mytokenizers import create_tokenizer
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from trainers.callbacks import JSONLoggingCallback
from utils.plotting import quick_plot


def parse_args():
    """Simple argument parser."""
    parser = argparse.ArgumentParser(description='Train Transformer models')
    
    # Model selection
    parser.add_argument('--model', default='tft', choices=['tft', 'tft-alibi', 'vanilla', 'baseline'], 
                        help='Model type to train')
    parser.add_argument('--preset', default='small', choices=['tiny', 'small', 'medium', 'large', 'debug'])
    
    # Data
    parser.add_argument('--dataset', default='roneneldan/TinyStories')
    parser.add_argument('--dataset_config', default=None)
    parser.add_argument('--max_samples', type=int, default=50000)
    
    # Training
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--block_size', type=int, default=None)
    
    # TFT-specific features (ignored for vanilla models)
    parser.add_argument('--use_v', action='store_true', help='Use value matrix in attention (TFT only)')
    parser.add_argument('--use_proj', action='store_true', help='Use output projection (TFT only)')
    
    # Output
    parser.add_argument('--output_dir', default='./outputs/clean_run')
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--plot', action='store_true', help='Plot results after training')
    
    # System
    parser.add_argument('--device', default='auto')
    
    return parser.parse_args()


def get_model_display_name(model_type: str, config) -> str:
    """Get a nice display name for the model."""
    base_name = {
        'tft': 'Token-Factored Transformer',
        'tft-alibi': 'Token-Factored Transformer', 
        'vanilla': 'Vanilla Transformer',
        'baseline': 'Vanilla Transformer'
    }.get(model_type, model_type)
    
    # Add TFT features if applicable
    if model_type in ['tft', 'tft-alibi'] and hasattr(config, 'use_v'):
        features = []
        if config.use_v:
            features.append('V-incl')
        if config.use_proj:
            features.append('Proj-incl')
        if features:
            base_name += f" ({'+'.join(features)})"
    
    return base_name


def main():
    """Main training function."""
    args = parse_args()
    
    model_display = get_model_display_name(args.model, args)
    print(f"🚀 Training {model_display}: {args.preset} on {args.dataset}")
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load config with overrides
    config_overrides = {}
    if args.block_size:
        config_overrides['block_size'] = args.block_size
    if args.lr:
        config_overrides['learning_rate'] = args.lr
    
    # Add TFT-specific overrides (will be ignored for vanilla models)
    if args.use_v:
        config_overrides['use_v'] = True
    if args.use_proj:
        config_overrides['use_proj'] = True
    
    # Get appropriate config type
    if args.model in ['tft', 'tft-alibi']:
        config = get_config('tft', args.preset, **config_overrides)
        features_str = []
        if config.use_v:
            features_str.append("V-fact")
        if config.use_proj:
            features_str.append("Proj-fact")
        features_display = f" [{'+'.join(features_str)}]" if features_str else ""
    else:  # vanilla/baseline
        config = get_config('base', args.preset, **config_overrides)
        features_display = ""
        # Warn if TFT features were specified but will be ignored
        if args.use_v or args.use_proj:
            print("⚠️ TFT features (--use_v, --use_proj) specified but ignored for vanilla model")
    
    print(f"Config: {config.n_layers}L-{config.n_heads}H-{config.d_model}D, block_size={config.block_size}{features_display}")
    
    # Create tokenizer
    tokenizer = create_tokenizer('gpt2')
    config.vocab_size = len(tokenizer)
    
    # Load data
    print(f"Loading {args.dataset}...")
    dataloader, _ = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=config.block_size,
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )
    print(f"Data loaded: {len(dataloader)} batches")
    
    # Create model
    model = get_model(args.model, config)
    total_params = model.get_num_params()
    print(f"Model created: {total_params/1e6:.2f}M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 0.01)
    )
    
    # Setup logging
    base_name = args.dataset.split('/')[-1] if '/' in args.dataset else args.dataset
    run_name = args.run_name or f"{args.model}_{args.preset}_{base_name}"
    
    # Add feature suffix to run name for TFT
    if args.model in ['tft', 'tft-alibi'] and hasattr(config, 'use_v'):
        feature_suffix = []
        if config.use_v:
            feature_suffix.append("v")
        if config.use_proj:
            feature_suffix.append("proj")
        if feature_suffix:
            run_name += f"_{'_'.join(feature_suffix)}"
    
    callbacks = [
        JSONLoggingCallback(
            output_dir=args.output_dir,
            run_name=run_name,
            log_every_n_steps=50
        )
    ]
    
    # Create trainer
    trainer = get_trainer(
        trainer_type='simple',
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        callbacks=callbacks
    )
    
    # Train
    print(f"🎯 Starting training for {args.epochs} epochs...")
    metrics = trainer.train()
    print(f"✅ Training completed! Final loss: {metrics.get('final_loss', 'N/A'):.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{run_name}_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': args.model,
        'tokenizer': tokenizer,
        'metrics': metrics,
        'args': vars(args)  # Save training arguments for reference
    }, model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Test generation
    print("🎨 Testing generation...")
    model.eval()
    test_prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(test_prompt)]).to(device)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
        generated_text = tokenizer.decode(generated[0].tolist())
    
    print(f"Generated: {generated_text}")
    
    # Print model comparison info
    positional_encoding = "ALiBi (attention bias)" if args.model in ['tft', 'tft-alibi'] else "Learned embeddings"
    print(f"\n📋 Model Summary:")
    print(f"  Type: {model_display}")
    print(f"  Positional Encoding: {positional_encoding}")
    print(f"  Parameters: {total_params/1e6:.2f}M")
    print(f"  Training Loss: {metrics.get('final_loss', 'N/A'):.4f}")
    
    # Plot results
    if args.plot:
        try:
            log_file = os.path.join(args.output_dir, 'training_metrics.json')
            if os.path.exists(log_file):
                print("📊 Generating plots...")
                quick_plot(log_file)
            else:
                print("⚠️ No metrics file found for plotting")
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
    
    print(f"🎉 All done! Results in {args.output_dir}")
    
    # Suggest comparison runs
    if args.model in ['tft', 'tft-alibi']:
        print(f"\n💡 To compare with vanilla transformer, run:")
        print(f"   python experiments/train_tft.py --model vanilla --preset {args.preset} --output_dir ./outputs/vanilla_comparison")
    else:
        print(f"\n💡 To compare with TFT, run:")
        print(f"   python experiments/train_tft.py --model tft --preset {args.preset} --output_dir ./outputs/tft_comparison")


if __name__ == "__main__":
    main()