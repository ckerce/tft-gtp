"""
Enhanced training script for Token-Factored Transformer with ALiBi.
Supports both single GPU and multi-GPU training via Accelerate.
"""

import argparse
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, 'src')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models import get_model
from config.model_configs import get_config, print_config
from mytokenizers import create_tokenizer
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from trainers.callbacks import JSONLoggingCallback
from utils.plotting import quick_plot


def parse_args():
    """Enhanced argument parser with multi-GPU support."""
    parser = argparse.ArgumentParser(description='Train TFT with ALiBi')
    
    # Model & data
    parser.add_argument('--preset', default='small', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size preset')
    parser.add_argument('--model', '--model_type', default='tft', 
                       choices=['tft', 'vanilla', 'tft-alibi'],
                       help='Model type to train')
    parser.add_argument('--dataset', default='roneneldan/TinyStories',
                       help='Dataset name')
    parser.add_argument('--dataset_config', default=None,
                       help='Dataset configuration')
    parser.add_argument('--max_samples', type=int, default=50000,
                       help='Maximum samples to use')
    
    # Training
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per device')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--block_size', type=int, default=None,
                       help='Sequence length (overrides config)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # TFT features
    parser.add_argument('--use_v', action='store_true', 
                       help='Use value matrix factorization in attention')
    parser.add_argument('--use_proj', action='store_true', 
                       help='Use output projection factorization')
    
    # Training infrastructure
    parser.add_argument('--trainer', default='simple', 
                       choices=['simple', 'accelerate'],
                       help='Trainer type to use')
    parser.add_argument('--mixed_precision', default='no',
                       choices=['no', 'fp16', 'bf16'],
                       help='Mixed precision mode')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Output & logging
    parser.add_argument('--output_dir', default='./outputs/training_run',
                       help='Output directory')
    parser.add_argument('--run_name', default=None,
                       help='Run name for logging')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log metrics every N steps')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate plots after training')
    parser.add_argument('--save_every_epoch', action='store_true',
                       help='Save checkpoint every epoch')
    
    # System
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def setup_device_and_trainer_type(args):
    """Setup device and determine appropriate trainer type."""
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Trainer type selection
    trainer_type = args.trainer
    
    # Check if we're running under accelerate launch
    accelerate_used = ('ACCELERATE_USE_FSDP' in os.environ or 
                      'ACCELERATE_USE_DEEPSPEED' in os.environ or
                      'LOCAL_RANK' in os.environ)
    
    if accelerate_used and trainer_type == 'simple':
        print("⚠️ Detected accelerate launch but trainer=simple. Switching to accelerate trainer.")
        trainer_type = 'accelerate'
    
    if trainer_type == 'accelerate':
        try:
            from accelerate import Accelerator
            # Test accelerate availability
            accelerator = Accelerator()
            print(f"✅ Accelerate available: {accelerator.num_processes} process(es)")
        except ImportError:
            print("❌ Accelerate not available. Falling back to simple trainer.")
            trainer_type = 'simple'
    
    return device, trainer_type


def main():
    """Main training function."""
    args = parse_args()
    
    # Normalize model name
    if args.model == 'tft-alibi':
        args.model = 'tft'
    
    print(f"🚀 Training {args.model.upper()}: {args.preset} on {args.dataset}")
    if args.verbose:
        print(f"Arguments: {vars(args)}")
    
    # Setup device and trainer
    device, trainer_type = setup_device_and_trainer_type(args)
    print(f"Using device: {device}, trainer: {trainer_type}")
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed set: {args.seed}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load config with overrides
    config_overrides = {}
    if args.block_size:
        config_overrides['block_size'] = args.block_size
    if args.lr:
        config_overrides['learning_rate'] = args.lr
    if args.weight_decay:
        config_overrides['weight_decay'] = args.weight_decay
    if args.use_v:
        config_overrides['use_v'] = True
    if args.use_proj:
        config_overrides['use_proj'] = True
    
    config = get_config(args.preset, **config_overrides)
    
    if args.verbose:
        print_config(config, f"{args.model.upper()} Configuration")
    else:
        print(f"Config: {config.n_layers}L-{config.n_heads}H-{config.d_model}D, "
              f"block_size={config.block_size}, factorization=v:{config.use_v}/proj:{config.use_proj}")
    
    # Create tokenizer
    print("🔤 Creating tokenizer...")
    tokenizer = create_tokenizer('gpt2')
    config.vocab_size = tokenizer.vocab_size
    
    # Load data
    print(f"📊 Loading {args.dataset}...")
    try:
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
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("This might be due to network issues or dataset access problems.")
        sys.exit(1)
    
    # Create model
    print(f"🏗️ Creating {args.model} model...")
    model = get_model(args.model, config)
    total_params = model.get_num_params()
    print(f"Model created: {total_params/1e6:.2f}M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup callbacks
    run_name = args.run_name or f"{args.preset}_{args.model}_{args.dataset.split('/')[-1]}"
    callbacks = [
        JSONLoggingCallback(
            output_dir=args.output_dir,
            run_name=run_name,
            log_every_n_steps=args.log_every_n_steps
        )
    ]
    
    # Create trainer with appropriate arguments
    trainer_kwargs = {
        'model': model,
        'dataloader': dataloader,
        'optimizer': optimizer,
        'device': device,
        'num_epochs': args.epochs,
        'output_dir': args.output_dir,
        'callbacks': callbacks,
        'clip_grad_norm': args.clip_grad_norm,
    }
    
    # Add trainer-specific arguments
    if trainer_type == 'accelerate':
        trainer_kwargs.update({
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'mixed_precision': args.mixed_precision,
            'seed': args.seed,
        })
        
        # Import accelerate trainer directly since get_trainer might not have it registered
        from trainers.accelerate_trainer import AccelerateTrainer
        trainer = AccelerateTrainer(**trainer_kwargs)
        
    else:  # simple trainer
        trainer = get_trainer(trainer_type='simple', **trainer_kwargs)
    
    # Train
    print(f"🎯 Starting training for {args.epochs} epochs...")
    try:
        metrics = trainer.train()
        final_loss = metrics.get('final_loss', 'N/A')
        training_time = metrics.get('training_time', 0)
        
        print(f"✅ Training completed! Final loss: {final_loss:.4f}, Time: {training_time:.1f}s")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model
    model_path = os.path.join(args.output_dir, f'{args.model}_model.pt')
    
    # Handle multi-GPU case where model might be wrapped
    model_to_save = model
    if hasattr(trainer, 'accelerator'):
        model_to_save = trainer.accelerator.unwrap_model(model)
    
    torch.save({
        'model_state_dict': model_to_save.state_dict(),
        'config': config,
        'args': vars(args),
        'metrics': metrics,
        'tokenizer_info': {
            'type': 'gpt2',
            'vocab_size': tokenizer.vocab_size
        }
    }, model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Test generation
    if trainer_type == 'simple' or (hasattr(trainer, 'accelerator') and trainer.accelerator.is_main_process):
        print("🎨 Testing generation...")
        try:
            model_to_save.eval()
            test_prompt = "Once upon a time"
            input_ids = torch.tensor([tokenizer.encode(test_prompt)]).to(device)
            
            with torch.no_grad():
                generated = model_to_save.generate(
                    input_ids, 
                    max_new_tokens=20, 
                    temperature=0.8,
                    top_k=50
                )
                generated_text = tokenizer.decode(generated[0].tolist())
            
            print(f"Generated: {generated_text}")
            
        except Exception as e:
            print(f"⚠️ Generation test failed: {e}")
    
    # Plot results
    if args.plot and (trainer_type == 'simple' or (hasattr(trainer, 'accelerator') and trainer.accelerator.is_main_process)):
        try:
            log_file = os.path.join(args.output_dir, 'training_metrics.json')
            if os.path.exists(log_file):
                print("📊 Generating plots...")
                quick_plot(log_file)
            else:
                print("⚠️ No metrics file found for plotting")
        except Exception as e:
            print(f"❌ Plotting failed: {e}")
    
    if trainer_type == 'simple' or (hasattr(trainer, 'accelerator') and trainer.accelerator.is_main_process):
        print(f"🎉 All done! Results in {args.output_dir}")


if __name__ == "__main__":
    main()