#!/usr/bin/env python3
"""
Check loss on vanilla transformer for comparison with TFT-Dict.
"""

import torch
import sys
sys.path.insert(0, 'src')

from models import get_model
from mytokenizers import create_tokenizer

# === MODIFY THIS PATH ===
VANILLA_CHECKPOINT_PATH = "./outputs/wiki_compare/vanilla/vanilla_model.pt"  # Your vanilla model path

def check_vanilla_loss():
    """Check vanilla model loss on same test texts."""
    
    print(f"🔍 Loading vanilla checkpoint: {VANILLA_CHECKPOINT_PATH}")
    
    # Load checkpoint
    checkpoint = torch.load(VANILLA_CHECKPOINT_PATH, map_location='cpu')
    config = checkpoint['config']
    
    print(f"📊 Vanilla Model: {config.n_layers}L-{config.n_heads}H-{config.d_model}D")
    
    # Create tokenizer
    tokenizer = create_tokenizer('gpt2')
    
    # Same test texts as TFT-Dict
    test_texts = [
        "The transformer architecture revolutionized natural language processing.",
        "Machine learning models require careful optimization and tuning.",
        "Artificial intelligence systems can process vast amounts of data.",
        "Neural networks learn complex patterns through gradient descent.",
        "Dictionary learning provides interpretable feature representations."
    ]
    
    print(f"\n🧪 Testing vanilla model on {len(test_texts)} sample texts...")
    
    # === VANILLA MODEL ===
    print(f"\n" + "="*60)
    print(f"🏛️ VANILLA TRANSFORMER (Trained)")
    print(f"="*60)
    
    vanilla_model = get_model('vanilla', config)
    vanilla_model.load_state_dict(checkpoint['model_state_dict'])
    vanilla_model.eval()
    
    total_loss = 0
    valid_samples = 0
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            input_ids = torch.tensor([tokenizer.encode(text)])
            outputs = vanilla_model(input_ids, labels=input_ids)
            
            loss = outputs.get('loss')
            
            if loss is not None:
                loss_val = loss.item()
                print(f"  Text {i+1}: Loss={loss_val:.4f} | '{text[:60]}...'")
                
                total_loss += loss_val
                valid_samples += 1
            else:
                print(f"  Text {i+1}: Loss=None | '{text[:60]}...'")
    
    if valid_samples > 0:
        avg_loss = total_loss / valid_samples
        
        print(f"\n📊 VANILLA MODEL SUMMARY:")
        print(f"  Average Loss:    {avg_loss:.4f}")
        print(f"  Perplexity:      {torch.exp(torch.tensor(avg_loss)).item():.2f}")
        print(f"  Valid samples:   {valid_samples}/{len(test_texts)}")
        
        # Compare with simple texts
        print(f"\n🧪 Testing on simpler texts for comparison...")
        
        simple_texts = [
            "The cat is happy today.",
            "She walked to the store quickly.",
            "The book is on the table.",
            "I like to read books.",
            "The sun is bright and warm."
        ]
        
        simple_total = 0
        simple_count = 0
        
        with torch.no_grad():
            for i, text in enumerate(simple_texts):
                input_ids = torch.tensor([tokenizer.encode(text)])
                outputs = vanilla_model(input_ids, labels=input_ids)
                
                loss = outputs.get('loss')
                if loss is not None:
                    loss_val = loss.item()
                    print(f"  Simple {i+1}: Loss={loss_val:.4f} | '{text}'")
                    simple_total += loss_val
                    simple_count += 1
        
        if simple_count > 0:
            simple_avg = simple_total / simple_count
            print(f"\n📈 COMPARISON:")
            print(f"  Technical text loss:  {avg_loss:.4f}")
            print(f"  Simple text loss:     {simple_avg:.4f}")
            print(f"  Difficulty ratio:     {avg_loss/simple_avg:.2f}x")
            
            print(f"\n🎯 ANALYSIS:")
            if simple_avg < 3.0:
                print(f"  ✅ Good performance on simple text (training-like data)")
            elif simple_avg < 5.0:
                print(f"  ⚠️ Moderate performance on simple text")
            else:
                print(f"  ❌ Poor performance even on simple text")
                
            if avg_loss / simple_avg < 2.0:
                print(f"  ✅ Technical text only moderately harder than simple text")
            elif avg_loss / simple_avg < 3.0:
                print(f"  ⚠️ Technical text significantly harder than simple text")
            else:
                print(f"  ❌ Technical text much harder than simple text")

def check_initial_vanilla():
    """Also check untrained vanilla for full comparison."""
    print(f"\n" + "="*60)
    print(f"🌱 VANILLA TRANSFORMER (Random Initialization)")
    print(f"="*60)
    
    # Load config but don't load weights
    checkpoint = torch.load(VANILLA_CHECKPOINT_PATH, map_location='cpu')
    config = checkpoint['config']
    
    initial_vanilla = get_model('vanilla', config)
    initial_vanilla.eval()
    
    tokenizer = create_tokenizer('gpt2')
    test_text = "The transformer architecture revolutionized natural language processing."
    
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(test_text)])
        outputs = initial_vanilla(input_ids, labels=input_ids)
        loss = outputs.get('loss')
        
        if loss is not None:
            print(f"  Initial vanilla loss: {loss.item():.4f}")
            print(f"  (Expected ~10-11 for random model)")

if __name__ == "__main__":
    try:
        check_vanilla_loss()
        check_initial_vanilla()
    except FileNotFoundError:
        print(f"❌ Vanilla checkpoint not found at: {VANILLA_CHECKPOINT_PATH}")
        print(f"Available model files might be:")
        import os
        for root, dirs, files in os.walk('./outputs'):
            for file in files:
                if 'vanilla' in file.lower() and file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")